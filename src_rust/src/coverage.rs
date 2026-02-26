//! Polygon-to-morton coverage: given a polygon defined by lat/lon vertices,
//! compute all morton indices at a given order that completely cover the polygon.
//!
//! Algorithm:
//! Phase A — Build contiguous boundary by casting vertices to cells and
//!           interpolating gaps along great-circle arcs (half cell-res spacing).
//! Phase B — Buffer the boundary, find connected components, classify
//!           inner vs outer via point-in-polygon on stereographic projection.
//!           Falls back to per-cell PIP when only 1 component (boundary gaps).
//! Phase C — Recursively expand inward from the inner buffer until the
//!           polygon interior is completely covered.
//! Phase D — PIP sweep: expand from coverage boundary, adding any adjacent
//!           cells whose centres are inside the polygon.  Catches cells
//!           missed by Phase C (e.g. at low orders).

use std::collections::{HashSet, VecDeque};
use std::f64::consts::PI;

use healpix::get;
use rayon::prelude::*;

use crate::geo2mort::{ang2pix_scalar, pix2ang_scalar};
use crate::morton::nested2mort;

// ── public entry point ───────────────────────────────────────────────────

/// Compute morton indices that completely cover a polygon.
///
/// # Arguments
/// * `lats` — vertex latitudes in degrees
/// * `lons` — vertex longitudes in degrees
/// * `order` — HEALPix depth (1–18)
///
/// # Returns
/// Sorted `Vec<i64>` of unique morton indices covering the polygon.
///
/// # Panics
/// * If `lats` and `lons` have different lengths
/// * If fewer than 3 vertices
/// * If order not in 1–18
pub fn polygon_to_morton_coverage(lats: &[f64], lons: &[f64], order: u8) -> Vec<i64> {
    assert_eq!(
        lats.len(),
        lons.len(),
        "lats and lons must have same length"
    );
    assert!(lats.len() >= 3, "Need at least 3 vertices for a polygon");
    assert!((1..=18).contains(&order), "Order must be 1-18");

    let depth = order;

    // Pre-compute PIP infrastructure (used in Phase B and Phase D)
    let (center_lat, center_lon) = polygon_centroid(lats, lons);
    let proj: Vec<(f64, f64)> = lats
        .iter()
        .zip(lons.iter())
        .map(|(&la, &lo)| stereographic_project(la, lo, center_lat, center_lon))
        .collect();
    let px: Vec<f64> = proj.iter().map(|p| p.0).collect();
    let py: Vec<f64> = proj.iter().map(|p| p.1).collect();

    // Phase A: build contiguous boundary (half cell-res interpolation)
    let boundary = ensure_boundary_contiguous(lats, lons, depth);

    // Buffer boundary by k=1 (exclusive — only the ring)
    let buffer_ring = nested_buffer_exclusive(&boundary, depth);

    let mut coverage: HashSet<u64> = boundary.clone();

    if !buffer_ring.is_empty() {
        // Phase B: classify buffer cells as inner/outer
        let components = connected_components(&buffer_ring, depth);

        let inner = if components.len() >= 2 {
            // Standard: classify entire components by sampling
            let (inner_set, _) = classify_components(&components, &px, &py, depth, center_lat, center_lon);
            inner_set
        } else {
            // Single component (boundary gap or tiny polygon) — classify
            // each buffer cell individually via PIP
            let mut inner_set = HashSet::new();
            for &cell in &buffer_ring {
                let (lon_deg, lat_deg) = pix2ang_scalar(depth, cell);
                let (sx, sy) =
                    stereographic_project(lat_deg, lon_deg, center_lat, center_lon);
                if point_in_polygon_ray_cast(sx, sy, &px, &py) {
                    inner_set.insert(cell);
                }
            }
            inner_set
        };

        if !inner.is_empty() {
            // Phase C: recursive inward fill from inner buffer
            let mut reference: HashSet<u64> = boundary;
            let mut frontier = inner;

            loop {
                let r_neighbors = nested_buffer_inclusive(&frontier, depth);
                let new_frontier: HashSet<u64> = r_neighbors
                    .iter()
                    .filter(|c| !frontier.contains(c) && !reference.contains(c))
                    .copied()
                    .collect();

                reference = frontier;
                frontier = new_frontier;

                let prev = coverage.len();
                coverage.extend(&r_neighbors);

                if coverage.len() == prev || frontier.is_empty() {
                    break;
                }
            }
        }
    }

    // Phase D: PIP sweep — expand outward from coverage, adding any adjacent
    // cells whose cell-centre is inside the polygon.  Catches cells that the
    // boundary/buffer/fill phases missed (e.g. at low orders or near gaps).
    loop {
        let adjacent = nested_buffer_exclusive(&coverage, depth);
        let mut new_cells = Vec::new();
        for &cell in &adjacent {
            let (lon_deg, lat_deg) = pix2ang_scalar(depth, cell);
            let (sx, sy) = stereographic_project(lat_deg, lon_deg, center_lat, center_lon);
            if point_in_polygon_ray_cast(sx, sy, &px, &py) {
                new_cells.push(cell);
            }
        }
        if new_cells.is_empty() {
            break;
        }
        coverage.extend(new_cells.iter());
    }

    nested_set_to_morton(&coverage, depth)
}

// ── Phase A helpers ──────────────────────────────────────────────────────

/// Cast polygon vertices to HEALPix cells and interpolate along edges
/// with half cell-res spacing to guarantee a contiguous boundary ring.
fn ensure_boundary_contiguous(lats: &[f64], lons: &[f64], depth: u8) -> HashSet<u64> {
    let n = lats.len();
    let cell_res = cell_resolution_rad(depth);
    // Use half cell-res for tighter interpolation — prevents gaps
    let spacing = cell_res * 0.5;
    let mut boundary = HashSet::new();

    for i in 0..n {
        let (lat1, lon1) = (lats[i], lons[i]);
        boundary.insert(ang2pix_scalar(depth, lon1, lat1));

        let j = (i + 1) % n;
        let (lat2, lon2) = (lats[j], lons[j]);

        let dist = great_circle_distance_rad(lat1, lon1, lat2, lon2);
        let n_seg = (dist / spacing).ceil() as usize;

        if n_seg > 1 {
            let n_interior = n_seg - 1;
            let interp = interpolate_great_circle(lat1, lon1, lat2, lon2, n_interior);
            for (la, lo) in interp {
                boundary.insert(ang2pix_scalar(depth, lo, la));
            }
        }
    }

    boundary
}

/// Great-circle distance in radians (Haversine formula).
fn great_circle_distance_rad(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (la1, lo1) = (lat1.to_radians(), lon1.to_radians());
    let (la2, lo2) = (lat2.to_radians(), lon2.to_radians());
    let dlat = (la2 - la1) * 0.5;
    let dlon = (lo2 - lo1) * 0.5;
    let a = dlat.sin().powi(2) + la1.cos() * la2.cos() * dlon.sin().powi(2);
    2.0 * a.sqrt().asin()
}

/// Angular resolution of one HEALPix cell at the given depth (radians).
fn cell_resolution_rad(depth: u8) -> f64 {
    let nside = (1u64 << depth) as f64;
    (PI / 3.0).sqrt() / nside
}

/// Interpolate `n` interior points along the great-circle arc.
/// Does not include the endpoints.
fn interpolate_great_circle(
    lat1: f64,
    lon1: f64,
    lat2: f64,
    lon2: f64,
    n: usize,
) -> Vec<(f64, f64)> {
    if n == 0 {
        return Vec::new();
    }

    let (la1, lo1) = (lat1.to_radians(), lon1.to_radians());
    let (la2, lo2) = (lat2.to_radians(), lon2.to_radians());
    let (c1, s1) = (la1.cos(), la1.sin());
    let (c2, s2) = (la2.cos(), la2.sin());
    let (x1, y1, z1) = (c1 * lo1.cos(), c1 * lo1.sin(), s1);
    let (x2, y2, z2) = (c2 * lo2.cos(), c2 * lo2.sin(), s2);

    let dot = (x1 * x2 + y1 * y2 + z1 * z2).clamp(-1.0, 1.0);
    let omega = dot.acos();
    // Threshold: ~1e-6 rad ≈ 6 m on Earth — well below any useful cell size
    if omega < 1e-6 {
        return Vec::new();
    }
    let sin_omega = omega.sin();

    let mut pts = Vec::with_capacity(n);
    for k in 1..=n {
        let f = k as f64 / (n + 1) as f64;
        let a = ((1.0 - f) * omega).sin() / sin_omega;
        let b = (f * omega).sin() / sin_omega;
        let x = a * x1 + b * x2;
        let y = a * y1 + b * y2;
        let z = a * z1 + b * z2;
        let lat = z.asin().to_degrees();
        let lon = y.atan2(x).to_degrees();
        pts.push((lat, lon));
    }
    pts
}

// ── buffer helpers (nested HEALPix space) ────────────────────────────────

/// Cells ∪ k=1-neighbors(cells).
fn nested_buffer_inclusive(cells: &HashSet<u64>, depth: u8) -> HashSet<u64> {
    let layer = get(depth);
    let cell_vec: Vec<u64> = cells.iter().copied().collect();

    cell_vec
        .par_iter()
        .fold(HashSet::new, |mut local, &cell| {
            for n in layer.kth_neighborhood(cell, 1) {
                local.insert(n);
            }
            local
        })
        .reduce(HashSet::new, |mut a, b| {
            a.extend(b);
            a
        })
}

/// k=1-neighbors(cells) − cells.
fn nested_buffer_exclusive(cells: &HashSet<u64>, depth: u8) -> HashSet<u64> {
    let inclusive = nested_buffer_inclusive(cells, depth);
    inclusive.difference(cells).copied().collect()
}

// ── Phase B helpers ──────────────────────────────────────────────────────

/// BFS connected-component decomposition on nested HEALPix cells.
fn connected_components(cells: &HashSet<u64>, depth: u8) -> Vec<HashSet<u64>> {
    let layer = get(depth);
    let mut remaining: HashSet<u64> = cells.clone();
    let mut components = Vec::new();

    while !remaining.is_empty() {
        let &start = remaining.iter().next().unwrap();
        remaining.remove(&start);

        let mut component = HashSet::new();
        component.insert(start);
        let mut queue = VecDeque::new();
        queue.push_back(start);

        while let Some(cell) = queue.pop_front() {
            for nbr in layer.kth_neighborhood(cell, 1) {
                if remaining.remove(&nbr) {
                    component.insert(nbr);
                    queue.push_back(nbr);
                }
            }
        }

        components.push(component);
    }

    components
}

/// Classify connected components as inner (inside polygon) or outer
/// using stereographic projection + ray-casting PIP.
///
/// Returns `(inner_cells, outer_cells)`.
fn classify_components(
    components: &[HashSet<u64>],
    px: &[f64],
    py: &[f64],
    depth: u8,
    center_lat: f64,
    center_lon: f64,
) -> (HashSet<u64>, HashSet<u64>) {
    let mut inner = HashSet::new();
    let mut outer = HashSet::new();

    for comp in components {
        // Adaptive sample count: floor=5, ceiling=50, ln(len) between
        let n_samples = {
            let ln_len = (comp.len() as f64).ln().ceil() as usize;
            ln_len.max(5).min(50)
        };

        let cells: Vec<u64> = comp.iter().copied().collect();
        let step = (cells.len() / n_samples).max(1);
        let mut inside = 0usize;
        let mut total = 0usize;

        for idx in (0..cells.len()).step_by(step).take(n_samples) {
            let (lon_deg, lat_deg) = pix2ang_scalar(depth, cells[idx]);
            let (sx, sy) = stereographic_project(lat_deg, lon_deg, center_lat, center_lon);
            if point_in_polygon_ray_cast(sx, sy, px, py) {
                inside += 1;
            }
            total += 1;
        }

        if total > 0 && inside > total / 2 {
            inner.extend(comp.iter());
        } else {
            outer.extend(comp.iter());
        }
    }

    (inner, outer)
}

/// Spherical centroid via Cartesian mean.
fn polygon_centroid(lats: &[f64], lons: &[f64]) -> (f64, f64) {
    let (mut cx, mut cy, mut cz) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..lats.len() {
        let la = lats[i].to_radians();
        let lo = lons[i].to_radians();
        cx += la.cos() * lo.cos();
        cy += la.cos() * lo.sin();
        cz += la.sin();
    }
    let n = lats.len() as f64;
    cx /= n;
    cy /= n;
    cz /= n;
    let lat = cz.atan2((cx * cx + cy * cy).sqrt()).to_degrees();
    let lon = cy.atan2(cx).to_degrees();
    (lat, lon)
}

/// Stereographic projection centred on `(center_lat, center_lon)`.
/// Returns `(x, y)` in the projected plane.
fn stereographic_project(
    lat: f64,
    lon: f64,
    center_lat: f64,
    center_lon: f64,
) -> (f64, f64) {
    let la = lat.to_radians();
    let lo = lon.to_radians();
    let cla = center_lat.to_radians();
    let clo = center_lon.to_radians();

    let cos_c = cla.sin() * la.sin() + cla.cos() * la.cos() * (lo - clo).cos();
    let k = 2.0 / (1.0 + cos_c);
    let x = k * la.cos() * (lo - clo).sin();
    let y = k * (cla.cos() * la.sin() - cla.sin() * la.cos() * (lo - clo).cos());
    (x, y)
}

/// 2-D ray-casting point-in-polygon test.
fn point_in_polygon_ray_cast(tx: f64, ty: f64, poly_x: &[f64], poly_y: &[f64]) -> bool {
    let n = poly_x.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        if ((poly_y[i] > ty) != (poly_y[j] > ty))
            && (tx
                < (poly_x[j] - poly_x[i]) * (ty - poly_y[i]) / (poly_y[j] - poly_y[i])
                    + poly_x[i])
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

// ── conversion helper ────────────────────────────────────────────────────

fn nested_set_to_morton(cells: &HashSet<u64>, depth: u8) -> Vec<i64> {
    let mut result: Vec<i64> = cells.iter().map(|&c| nested2mort(c, depth)).collect();
    result.sort();
    result
}

// ── tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle_basic() {
        let lats = vec![40.0, 50.0, 45.0];
        let lons = vec![-120.0, -120.0, -110.0];
        let result = polygon_to_morton_coverage(&lats, &lons, 4);
        assert!(!result.is_empty(), "Coverage should not be empty");
        for &m in &result {
            assert!(m != 0, "Morton index should not be zero");
        }
    }

    #[test]
    fn test_coverage_sorted_unique() {
        let lats = vec![40.0, 50.0, 45.0];
        let lons = vec![-120.0, -120.0, -110.0];
        let result = polygon_to_morton_coverage(&lats, &lons, 4);
        for i in 1..result.len() {
            assert!(result[i] > result[i - 1], "Result must be sorted and unique");
        }
    }

    #[test]
    fn test_square_coverage() {
        let lats = vec![40.0, 40.0, 50.0, 50.0];
        let lons = vec![-125.0, -115.0, -115.0, -125.0];
        let result = polygon_to_morton_coverage(&lats, &lons, 4);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_southern_hemisphere() {
        let lats = vec![-70.0, -80.0, -75.0];
        let lons = vec![30.0, 30.0, 50.0];
        let result = polygon_to_morton_coverage(&lats, &lons, 4);
        assert!(!result.is_empty());
        assert!(
            result.iter().any(|&m| m < 0),
            "Southern hemisphere should have negative morton indices"
        );
    }

    #[test]
    fn test_interpolation_basic() {
        let pts = interpolate_great_circle(0.0, 0.0, 10.0, 0.0, 3);
        assert_eq!(pts.len(), 3);
        for (lat, _lon) in &pts {
            assert!(*lat > 0.0 && *lat < 10.0);
        }
    }

    #[test]
    fn test_interpolation_same_point() {
        let pts = interpolate_great_circle(45.0, -120.0, 45.0, -120.0, 5);
        assert!(pts.is_empty(), "Same point should produce no interpolation");
    }

    #[test]
    fn test_great_circle_distance() {
        let d = great_circle_distance_rad(45.0, -120.0, 45.0, -120.0);
        assert!(d < 1e-10, "Same point should have zero distance");

        // ~0.01745 rad per degree at equator
        let d2 = great_circle_distance_rad(0.0, 0.0, 1.0, 0.0);
        assert!(
            (d2 - 0.01745).abs() < 0.001,
            "1 degree at equator ≈ 0.01745 rad"
        );
    }

    #[test]
    fn test_pip_inside() {
        let px = vec![0.0, 1.0, 1.0, 0.0];
        let py = vec![0.0, 0.0, 1.0, 1.0];
        assert!(point_in_polygon_ray_cast(0.5, 0.5, &px, &py));
    }

    #[test]
    fn test_pip_outside() {
        let px = vec![0.0, 1.0, 1.0, 0.0];
        let py = vec![0.0, 0.0, 1.0, 1.0];
        assert!(!point_in_polygon_ray_cast(2.0, 2.0, &px, &py));
    }

    #[test]
    fn test_different_orders() {
        let lats = vec![40.0, 50.0, 45.0];
        let lons = vec![-120.0, -120.0, -110.0];
        let r4 = polygon_to_morton_coverage(&lats, &lons, 4);
        let r6 = polygon_to_morton_coverage(&lats, &lons, 6);
        assert!(
            r6.len() > r4.len(),
            "Higher order should produce more cells"
        );
    }

    #[test]
    fn test_centroid_antimeridian() {
        let lats = vec![0.0, 0.0, 10.0, 10.0];
        let lons = vec![179.0, -179.0, -179.0, 179.0];
        let (clat, clon) = polygon_centroid(&lats, &lons);
        assert!(clat.abs() < 10.0);
        assert!(clon.abs() > 170.0);
    }

    #[test]
    fn test_stereographic_roundtrip() {
        let (x, y) = stereographic_project(45.0, -120.0, 45.0, -120.0);
        assert!(x.abs() < 1e-10 && y.abs() < 1e-10, "Center projects to origin");
    }

    #[test]
    #[should_panic(expected = "at least 3 vertices")]
    fn test_too_few_vertices() {
        polygon_to_morton_coverage(&[0.0, 1.0], &[0.0, 1.0], 4);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn test_mismatched_lengths() {
        polygon_to_morton_coverage(&[0.0, 1.0, 2.0], &[0.0, 1.0], 4);
    }

    #[test]
    fn test_square_superset() {
        // Coverage must include all cells whose centres are inside the polygon
        use crate::geo2mort::geo2mort_scalar;
        let lats = vec![40.0, 40.0, 50.0, 50.0];
        let lons = vec![-125.0, -115.0, -115.0, -125.0];
        let result = polygon_to_morton_coverage(&lats, &lons, 4);
        let coverage_set: HashSet<i64> = result.into_iter().collect();

        // Sample interior points
        for lat in [42.0, 45.0, 48.0] {
            for lon in [-123.0, -120.0, -117.0] {
                let m = geo2mort_scalar(lat, lon, 4);
                assert!(
                    coverage_set.contains(&m),
                    "Interior cell at ({}, {}) = {} not in coverage",
                    lat, lon, m
                );
            }
        }
    }
}

//! Polygon-to-morton coverage: given a polygon defined by lat/lon vertices,
//! compute all morton indices at a given order that completely cover the polygon.
//!
//! Algorithm:
//! Phase A — Build contiguous boundary by casting vertices to cells and
//!           interpolating gaps along great-circle arcs (half cell-res spacing).
//! Phase B — Buffer the boundary, find connected components, classify
//!           inner vs outer via gnomonic projection + winding-number PIP.
//!           Falls back to per-cell PIP when only 1 component (boundary gaps).
//! Phase C — Recursively expand inward from the inner buffer until the
//!           polygon interior is completely covered.

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

    // Pre-compute polygon vertices as unit 3D vectors (used in Phase B PIP)
    let poly_verts: Vec<[f64; 3]> = lats
        .iter()
        .zip(lons.iter())
        .map(|(&la, &lo)| latlon_to_unit_vec(la, lo))
        .collect();

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
            let (inner_set, _) = classify_components(&components, &poly_verts, depth);
            inner_set
        } else {
            // Single component (boundary gap or tiny polygon) — classify
            // each buffer cell individually via PIP
            let mut inner_set = HashSet::new();
            for &cell in &buffer_ring {
                let (lon_deg, lat_deg) = pix2ang_scalar(depth, cell);
                let center = latlon_to_unit_vec(lat_deg, lon_deg);
                if gnomonic_pip(&center, &poly_verts) {
                    inner_set.insert(cell);
                }
            }
            inner_set
        };

        if !inner.is_empty() {
            // Phase C: PIP-bounded flood fill from inner buffer cells.
            // Each new frontier cell is checked against the polygon via PIP
            // to prevent leaking through any boundary gaps.
            coverage.extend(&inner);
            let mut visited = coverage.clone();
            let mut frontier = inner;

            loop {
                let neighbors = nested_buffer_exclusive(&frontier, depth);
                let mut new_frontier = HashSet::new();
                for &cell in &neighbors {
                    if visited.contains(&cell) {
                        continue;
                    }
                    visited.insert(cell);
                    let (lon_deg, lat_deg) = pix2ang_scalar(depth, cell);
                    let center = latlon_to_unit_vec(lat_deg, lon_deg);
                    if gnomonic_pip(&center, &poly_verts) {
                        new_frontier.insert(cell);
                        coverage.insert(cell);
                    }
                }
                if new_frontier.is_empty() {
                    break;
                }
                frontier = new_frontier;
            }
        }
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
/// using gnomonic projection + winding-number PIP.
///
/// Returns `(inner_cells, outer_cells)`.
fn classify_components(
    components: &[HashSet<u64>],
    poly_verts: &[[f64; 3]],
    depth: u8,
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
            let center = latlon_to_unit_vec(lat_deg, lon_deg);
            if gnomonic_pip(&center, poly_verts) {
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

/// Convert lat/lon (degrees) to unit 3D vector on the sphere.
#[inline]
fn latlon_to_unit_vec(lat_deg: f64, lon_deg: f64) -> [f64; 3] {
    let la = lat_deg.to_radians();
    let lo = lon_deg.to_radians();
    let (sla, cla) = (la.sin(), la.cos());
    let (slo, clo) = (lo.sin(), lo.cos());
    [cla * clo, cla * slo, sla]
}

/// Gnomonic point-in-polygon test: projects polygon vertices onto the
/// tangent plane at `center` (gnomonic projection), then runs a 2D
/// winding-number test.  The test point projects to the origin (0, 0).
///
/// Gnomonic projection preserves great-circle edges as straight lines
/// and requires zero trig per vertex (just dot products).
fn gnomonic_pip(center: &[f64; 3], poly_verts: &[[f64; 3]]) -> bool {
    let n = poly_verts.len();
    if n < 3 {
        return false;
    }

    // Build orthonormal basis (a, b) for the tangent plane at `center`.
    // Pick a reference vector that isn't parallel to `center`.
    let ref_vec = if center[2].abs() < 0.9 {
        [0.0, 0.0, 1.0]
    } else {
        [1.0, 0.0, 0.0]
    };

    // a = normalize(ref_vec - (ref_vec·center) * center)
    let dot_rc = ref_vec[0] * center[0] + ref_vec[1] * center[1] + ref_vec[2] * center[2];
    let ax = ref_vec[0] - dot_rc * center[0];
    let ay = ref_vec[1] - dot_rc * center[1];
    let az = ref_vec[2] - dot_rc * center[2];
    let a_norm = (ax * ax + ay * ay + az * az).sqrt();
    let a = [ax / a_norm, ay / a_norm, az / a_norm];

    // b = center × a  (already unit length since center and a are orthonormal)
    let b = [
        center[1] * a[2] - center[2] * a[1],
        center[2] * a[0] - center[0] * a[2],
        center[0] * a[1] - center[1] * a[0],
    ];

    // Project polygon vertices onto the tangent plane.
    // For vertex v: dot = center·v, proj_x = a·v / dot, proj_y = b·v / dot
    // Test point (center) projects to (0, 0).
    let mut proj_x = Vec::with_capacity(n);
    let mut proj_y = Vec::with_capacity(n);
    for v in poly_verts {
        let dot = center[0] * v[0] + center[1] * v[1] + center[2] * v[2];
        // Skip vertices on the opposite hemisphere (dot <= 0 means > 90° away)
        // Use a small epsilon to avoid division by near-zero
        let d = if dot > 1e-12 { dot } else { 1e-12 };
        proj_x.push((a[0] * v[0] + a[1] * v[1] + a[2] * v[2]) / d);
        proj_y.push((b[0] * v[0] + b[1] * v[1] + b[2] * v[2]) / d);
    }

    winding_number_2d(0.0, 0.0, &proj_x, &proj_y)
}

/// 2D winding-number point-in-polygon test (Hormann & Agathos 2001).
/// Returns true if `(tx, ty)` is inside the polygon defined by
/// `(poly_x, poly_y)`.
fn winding_number_2d(tx: f64, ty: f64, poly_x: &[f64], poly_y: &[f64]) -> bool {
    let n = poly_x.len();
    let mut winding: i32 = 0;
    let mut j = n - 1;
    for i in 0..n {
        let yi = poly_y[j] - ty;
        let yj = poly_y[i] - ty;
        if yi <= 0.0 {
            if yj > 0.0 {
                // Upward crossing — check if test point is left of edge
                let cross = (poly_x[j] - tx) * yj - (poly_x[i] - tx) * yi;
                if cross > 0.0 {
                    winding += 1;
                }
            }
        } else if yj <= 0.0 {
            // Downward crossing — check if test point is right of edge
            let cross = (poly_x[j] - tx) * yj - (poly_x[i] - tx) * yi;
            if cross < 0.0 {
                winding -= 1;
            }
        }
        j = i;
    }
    winding != 0
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
    fn test_winding_number_inside() {
        let px = vec![0.0, 1.0, 1.0, 0.0];
        let py = vec![0.0, 0.0, 1.0, 1.0];
        assert!(winding_number_2d(0.5, 0.5, &px, &py));
    }

    #[test]
    fn test_winding_number_outside() {
        let px = vec![0.0, 1.0, 1.0, 0.0];
        let py = vec![0.0, 0.0, 1.0, 1.0];
        assert!(!winding_number_2d(2.0, 2.0, &px, &py));
    }

    #[test]
    fn test_gnomonic_pip_inside() {
        // Test point at (45, -120) inside a square polygon
        let center = latlon_to_unit_vec(45.0, -120.0);
        let poly = vec![
            latlon_to_unit_vec(40.0, -125.0),
            latlon_to_unit_vec(40.0, -115.0),
            latlon_to_unit_vec(50.0, -115.0),
            latlon_to_unit_vec(50.0, -125.0),
        ];
        assert!(gnomonic_pip(&center, &poly));
    }

    #[test]
    fn test_gnomonic_pip_outside() {
        // Test point at (60, -100) outside the square polygon
        let center = latlon_to_unit_vec(60.0, -100.0);
        let poly = vec![
            latlon_to_unit_vec(40.0, -125.0),
            latlon_to_unit_vec(40.0, -115.0),
            latlon_to_unit_vec(50.0, -115.0),
            latlon_to_unit_vec(50.0, -125.0),
        ];
        assert!(!gnomonic_pip(&center, &poly));
    }

    #[test]
    fn test_gnomonic_pip_south_pole() {
        // Test point near south pole inside a polar polygon
        let center = latlon_to_unit_vec(-85.0, 0.0);
        let poly = vec![
            latlon_to_unit_vec(-80.0, -90.0),
            latlon_to_unit_vec(-80.0, 0.0),
            latlon_to_unit_vec(-80.0, 90.0),
            latlon_to_unit_vec(-80.0, 180.0),
        ];
        assert!(gnomonic_pip(&center, &poly));
    }

    #[test]
    fn test_latlon_to_unit_vec_norm() {
        // Unit vector should have magnitude 1
        for (lat, lon) in [(0.0, 0.0), (45.0, -120.0), (-90.0, 0.0), (90.0, 180.0)] {
            let v = latlon_to_unit_vec(lat, lon);
            let mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!((mag - 1.0).abs() < 1e-12, "Unit vec at ({lat},{lon}) has mag {mag}");
        }
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

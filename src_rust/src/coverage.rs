//! Polygon-to-morton coverage via a **top-down hierarchical region coverer**
//! (issue #30).
//!
//! Starting from the 12 HEALPix base cells, each cell is classified against the
//! polygon ring-set (`cell_geom::classify`) as inside / outside / straddling:
//! `outside` subtrees are pruned, `inside` cells are kept whole at their coarse
//! order, and `straddle` cells are refined into their 4 children down to the
//! target order — where any remaining straddler is a boundary leaf.  There is no
//! boundary rasterization, buffer ring, or flood fill, so the result is a pure,
//! deterministic function of the inputs (fixes the issue #28 class of bug by
//! construction).
//!
//! The descent emits a Multi-Order Coverage map; [`polygon_to_morton_coverage`]
//! flattens it to a single order (back-compatible), while
//! [`polygon_to_morton_moc`] returns the compact mixed-order form.

use std::f64::consts::PI;

use rayon::prelude::*;

use crate::cell_geom::{classify, Cap, Classification};
use crate::geo2mort::ang2pix_scalar;
use crate::morton::nested2mort;
use crate::sphere::{choose_backend, latlon_to_unit_vec, PipBackend, Vec3};

// ── public entry points ──────────────────────────────────────────────────

/// Compute morton indices covering a polygon, as a flat list at `order`.
///
/// # Arguments
/// * `lats` — vertex latitudes in degrees
/// * `lons` — vertex longitudes in degrees
/// * `order` — HEALPix depth (1–18)
///
/// # Returns
/// Sorted unique `Vec<i64>` of morton indices at `order` whose cells intersect
/// the closed polygon (contract (a): the cover is a superset of the polygon).
///
/// # Panics
/// * If `lats`/`lons` differ in length, fewer than 3 vertices, or order ∉ 1–18.
pub fn polygon_to_morton_coverage(lats: &[f64], lons: &[f64], order: u8) -> Vec<i64> {
    let moc = polygon_descend(lats, lons, order);
    crate::moc::to_order(&moc, order)
}

/// Compute polygon coverage as a compact, normalized Multi-Order Coverage map:
/// coarse cells for the interior, fine cells (at `order`) along the boundary.
pub fn polygon_to_morton_moc(lats: &[f64], lons: &[f64], order: u8) -> Vec<i64> {
    let moc = polygon_descend(lats, lons, order);
    crate::moc::normalize(&moc)
}

// ── descent ──────────────────────────────────────────────────────────────

/// Validate inputs, build the ring-set, run the descent, and return its cells
/// as (un-normalized, mixed-order) morton indices.
fn polygon_descend(lats: &[f64], lons: &[f64], order: u8) -> Vec<i64> {
    assert_eq!(
        lats.len(),
        lons.len(),
        "lats and lons must have same length"
    );
    assert!(lats.len() >= 3, "Need at least 3 vertices for a polygon");
    assert!((1..=18).contains(&order), "Order must be 1-18");

    let rings = vec![build_ring(lats, lons)];
    descend(&rings, order)
        .iter()
        .map(|&(pixel, depth)| nested2mort(pixel, depth))
        .collect()
}

/// Convert lat/lon vertices to a closed ring of unit vectors, dropping a
/// duplicate closing vertex if present.
fn build_ring(lats: &[f64], lons: &[f64]) -> Vec<Vec3> {
    let mut ring: Vec<Vec3> = lats
        .iter()
        .zip(lons.iter())
        .map(|(&la, &lo)| latlon_to_unit_vec(la, lo))
        .collect();
    if ring.len() > 3 {
        let (f, l) = (ring[0], ring[ring.len() - 1]);
        if (f[0] - l[0]).abs() < 1e-12 && (f[1] - l[1]).abs() < 1e-12 && (f[2] - l[2]).abs() < 1e-12
        {
            ring.pop();
        }
    }
    ring
}

/// Top-down descent producing the covering cells as `(nested_pixel, depth)`.
///
/// The 12 base subtrees are independent and explored in parallel; each is a
/// deterministic stack DFS with a fixed child order, so the merged result is
/// order-independent (callers sort/normalize it).
fn descend(rings: &[Vec<Vec3>], order: u8) -> Vec<(u64, u8)> {
    let backend: PipBackend = choose_backend(rings);
    let cap = Cap::of_rings(rings);

    // Exact vertex-in-cell containment: each vertex's leaf cell at `order`,
    // from which its ancestor at any depth is a bit-shift (see `cell_geom`).
    let vert_cells: Vec<Vec<u64>> = rings
        .iter()
        .map(|ring| {
            ring.iter()
                .map(|v| {
                    let lon = v[1].atan2(v[0]).to_degrees();
                    let lat = v[2].clamp(-1.0, 1.0).asin().to_degrees();
                    ang2pix_scalar(order, lon, lat)
                })
                .collect()
        })
        .collect();

    (0..12u64)
        .into_par_iter()
        .flat_map_iter(|base| {
            let mut out: Vec<(u64, u8)> = Vec::new();
            let mut stack: Vec<(u64, u8)> = vec![(base, 0u8)];
            while let Some((pixel, depth)) = stack.pop() {
                match classify(depth, pixel, rings, &vert_cells, order, backend, &cap) {
                    Classification::Outside => {}
                    Classification::Inside => out.push((pixel, depth)),
                    Classification::Straddle => {
                        if depth < order {
                            for child in 0..4u64 {
                                stack.push((pixel * 4 + child, depth + 1));
                            }
                        } else {
                            out.push((pixel, depth));
                        }
                    }
                }
            }
            out
        })
        .collect()
}

// ── great-circle helpers (shared with `linestring`) ──────────────────────

/// Great-circle distance in radians (Haversine formula).
pub(crate) fn great_circle_distance_rad(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (la1, lo1) = (lat1.to_radians(), lon1.to_radians());
    let (la2, lo2) = (lat2.to_radians(), lon2.to_radians());
    let dlat = (la2 - la1) * 0.5;
    let dlon = (lo2 - lo1) * 0.5;
    let a = dlat.sin().powi(2) + la1.cos() * la2.cos() * dlon.sin().powi(2);
    2.0 * a.sqrt().asin()
}

/// Angular resolution of one HEALPix cell at the given depth (radians).
pub(crate) fn cell_resolution_rad(depth: u8) -> f64 {
    let nside = (1u64 << depth) as f64;
    (PI / 3.0).sqrt() / nside
}

/// Interpolate `n` interior points along the great-circle arc.
/// Does not include the endpoints.
pub(crate) fn interpolate_great_circle(
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
    fn test_different_orders() {
        let lats = vec![40.0, 50.0, 45.0];
        let lons = vec![-120.0, -120.0, -110.0];
        let r4 = polygon_to_morton_coverage(&lats, &lons, 4);
        let r6 = polygon_to_morton_coverage(&lats, &lons, 6);
        assert!(r6.len() > r4.len(), "Higher order should produce more cells");
    }

    #[test]
    fn test_moc_is_compact_and_densifies_to_flat() {
        // The MOC must be no larger than the flat cover and must densify back
        // to exactly the flat cover (densify-invariance).
        let lats = vec![40.0, 40.0, 50.0, 50.0];
        let lons = vec![-125.0, -115.0, -115.0, -125.0];
        let flat = polygon_to_morton_coverage(&lats, &lons, 8);
        let moc = polygon_to_morton_moc(&lats, &lons, 8);
        assert!(moc.len() <= flat.len(), "MOC should be compact");
        assert!(moc.len() < flat.len(), "interior should collapse to coarse cells");
        assert_eq!(crate::moc::to_order(&moc, 8), flat, "MOC must densify to flat");
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
    fn test_polar_polygon_deterministic() {
        // Regression test for issue #28: a thin near-polar lon-strip used to
        // produce one of two different cell sets at random.  The hierarchical
        // coverer is deterministic by construction and fills the interior.
        let lats = vec![-89.0, -59.09804617, -59.09804617, -89.0];
        let lons = vec![105.5108378, 105.5108378, 106.5108378, 106.5108378];

        let first = polygon_to_morton_coverage(&lats, &lons, 10);
        for _ in 0..50 {
            let r = polygon_to_morton_coverage(&lats, &lons, 10);
            assert_eq!(r, first, "coverage must be deterministic across calls");
        }
        // The buggy boundary-only result was 1166 cells; the correct filled
        // result is in the thousands.  Guard against regressing to boundary-only.
        assert!(
            first.len() > 2000,
            "expected filled interior, got {} cells",
            first.len()
        );
    }

    #[test]
    fn test_square_superset() {
        // Coverage must include all cells whose centres are inside the polygon
        use crate::geo2mort::geo2mort_scalar;
        use std::collections::HashSet;
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
                    lat,
                    lon,
                    m
                );
            }
        }
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

        let d2 = great_circle_distance_rad(0.0, 0.0, 1.0, 0.0);
        assert!(
            (d2 - 0.01745).abs() < 0.001,
            "1 degree at equator ≈ 0.01745 rad"
        );
    }
}

//! Unit tests for the polygon coverage descent (issue #30/#22).
//!
//! Split out of `coverage.rs` to keep that module under the project's
//! ~1000-line soft limit; wired back in via `#[cfg(test)] mod tests;`.

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
fn test_donut_carves_hole() {
    use crate::geo2mort::geo2mort_scalar;
    use std::collections::HashSet;
    // 20° outer box with a centred 6° hole, both around (45, -120).
    let lats = vec![
        vec![35.0, 35.0, 55.0, 55.0],
        vec![42.0, 42.0, 48.0, 48.0],
    ];
    let lons = vec![
        vec![-130.0, -110.0, -110.0, -130.0],
        vec![-123.0, -117.0, -117.0, -123.0],
    ];
    let cov: HashSet<i64> = multipolygon_to_morton_coverage(&lats, &lons, 7)
        .into_iter()
        .collect();
    // Hole interior must be carved out; annulus must be covered.
    assert!(
        !cov.contains(&geo2mort_scalar(45.0, -120.0, 7)),
        "hole interior must not be covered"
    );
    assert!(
        cov.contains(&geo2mort_scalar(37.0, -120.0, 7)),
        "annulus must be covered"
    );
}

#[test]
fn test_multipart_disjoint_equals_union() {
    use std::collections::HashSet;
    let a_la = vec![40.0, 50.0, 45.0];
    let a_lo = vec![-120.0, -120.0, -110.0];
    let b_la = vec![10.0, 20.0, 15.0];
    let b_lo = vec![-80.0, -80.0, -70.0];
    let union: HashSet<i64> = polygon_to_morton_coverage(&a_la, &a_lo, 6)
        .into_iter()
        .chain(polygon_to_morton_coverage(&b_la, &b_lo, 6))
        .collect();
    let multi: HashSet<i64> = multipolygon_to_morton_coverage(
        &vec![a_la, b_la],
        &vec![a_lo, b_lo],
        6,
    )
    .into_iter()
    .collect();
    assert_eq!(multi, union, "disjoint multipart should equal the union");
}

#[test]
fn test_tolerance_coarsens_boundary() {
    // A tolerance stop must yield no more cells than the exact order-10 MOC,
    // and must still cover the interior (superset of a coarse exact cover).
    let lats = vec![40.0, 40.0, 50.0, 50.0];
    let lons = vec![-125.0, -115.0, -115.0, -125.0];
    let exact = polygon_to_morton_moc(&lats, &lons, 10);
    let tol = polygon_to_morton_moc_tolerance(&lats, &lons, 10, 2.0_f64.to_radians());
    assert!(!tol.is_empty());
    assert!(
        tol.len() <= exact.len(),
        "tolerance cover ({}) should not exceed exact ({})",
        tol.len(),
        exact.len()
    );
    // Determinism.
    assert_eq!(tol, polygon_to_morton_moc_tolerance(&lats, &lons, 10, 2.0_f64.to_radians()));
}

#[test]
fn test_budget_respects_cap() {
    // The best-first budget must keep the cell count near the target and be
    // deterministic.
    let lats = vec![40.0, 40.0, 50.0, 50.0];
    let lons = vec![-125.0, -115.0, -115.0, -125.0];
    for budget in [20usize, 50, 200] {
        let (cov, effective) = polygon_to_morton_moc_budget(&lats, &lons, 12, budget);
        assert!(!cov.is_empty());
        // Soft target: at most one split (×4) past the effective budget.
        assert!(
            cov.len() <= effective + 4,
            "budget {} (eff {}) produced {} cells",
            budget,
            effective,
            cov.len()
        );
        assert_eq!(cov, polygon_to_morton_moc_budget(&lats, &lons, 12, budget).0);
    }
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
fn test_polar_boundary_bulge_covered() {
    // Regression test for issue #32.  HEALPix cell edges are not great-circle
    // arcs; near the poles the true cell bulges outside the 4-corner geodesic
    // quad.  This real ATL06 cycle-22 granule (near the south pole, wrapping
    // the antimeridian) grazes the curved boundary of order-6 cell -6111131
    // — overlap that S2 and EPSG:3031 shapely both see.  The corners-only
    // straddle test pruned that cell at order 6, dropping it at every order;
    // the densified-boundary straddle test must now cover it.
    let lats = vec![
        -78.97166, -78.94929, -79.42733, -80.73793, -82.03867, -83.31985,
        -85.7817, -86.88342, -87.71538, -87.87437, -87.80769, -87.61281,
        -87.35613, -87.0609, -86.04897, -85.66109, -83.79971, -83.41805,
        -83.03733, -82.61643, -80.39302, -79.93122, -79.49362, -78.94943,
        -78.97178, -79.51702, -79.95551, -80.41857, -82.64886, -83.07129,
        -83.45395, -83.83797, -85.71503, -86.10799, -87.14043, -87.44464,
        -87.71114, -87.91512, -87.98525, -87.81823, -86.95811, -85.83679,
        -83.35487, -82.06846, -80.76386, -79.45036, -78.97166,
    ];
    let lons = vec![
        -37.60041, -38.19306, -38.71944, -40.42942, -42.66808, -45.72009,
        -57.13684, -69.29219, -92.07313, -102.96984, -139.16366, -149.26852,
        -157.63891, -164.28406, -177.35124, 179.5542, 170.5179, 169.31794,
        168.26288, 167.22786, 163.25287, 162.63733, 162.10694, 161.50445,
        160.91177, 161.48734, 161.99211, 162.57855, 166.3654, 167.35214,
        168.3599, 169.50821, 178.19957, -178.79874, -165.9332, -159.26658,
        -150.74238, -140.28136, -102.10046, -90.7387, -67.65821, -55.75165,
        -44.77371, -41.86274, -39.73073, -38.10328, -37.60041,
    ];
    let cover = polygon_to_morton_coverage(&lats, &lons, 8);
    // A covered order-8 morton is a child of order-6 cell -6111131 iff its
    // order-6 ancestor (two decimal digits stripped) equals it.
    let hits = cover.iter().filter(|&&m| m / 100 == -6111131).count();
    assert!(
        hits > 0,
        "issue #32: order-8 cover misses near-pole boundary cell -6111131 \
         (granule grazes the cell's curved edge, outside its geodesic quad)"
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
fn test_covers_complement_detects_hemisphere_plus() {
    // A band ring at lat -10° traversed so its CCW interior is the >hemisphere
    // northern region (the #22 "everything except Antarctica" shape). The
    // cap-axis antipode must test inside ⇒ complement detected. A small
    // mid-latitude square is sub-hemisphere ⇒ not complement.
    let band: Vec<Vec3> = (0..36)
        .map(|k| latlon_to_unit_vec(-10.0, k as f64 * 10.0))
        .collect();
    let cap = Cap::of_rings(&[band.clone()]);
    assert!(
        covers_complement(&[band], &cap),
        "hemisphere+ band must be detected as complement"
    );

    let square: Vec<Vec3> = [(40.0, -125.0), (40.0, -115.0), (50.0, -115.0), (50.0, -125.0)]
        .iter()
        .map(|&(la, lo)| latlon_to_unit_vec(la, lo))
        .collect();
    let cap2 = Cap::of_rings(&[square.clone()]);
    assert!(
        !covers_complement(&[square], &cap2),
        "sub-hemisphere square must not be complement"
    );
}

#[test]
fn test_complement_guard_keeps_antipodal_base_cell() {
    // Phase 2 regression (#22): for a hemisphere+ polygon the bounding cap
    // bounds only the boundary vertices, so a base cell near the cap antipode
    // sits far from the cap axis and the vertex-cap cull would prune it —
    // even though it is deep in the (large) interior. The complement guard
    // must keep it. Prove both: the un-guarded cull prunes the cell, and the
    // guarded path retains it.
    let band: Vec<Vec3> = (0..36)
        .map(|k| latlon_to_unit_vec(-10.0, k as f64 * 10.0))
        .collect();
    let rings = vec![band];
    let edges = build_edges(&rings, 4);
    let cap = Cap::of_rings(&rings);
    assert!(covers_complement(&rings, &cap), "precondition: hemisphere+");
    let backend = choose_backend(&rings);

    // The base cell whose centre is closest to the cap antipode is the one the
    // vertex-cap cull is most prone to wrongly prune.
    let antipode = [-cap.axis[0], -cap.axis[1], -cap.axis[2]];
    let far_base = (0..12u64)
        .max_by(|&a, &b| {
            let da = dot(&antipode, &cell_center_vec(0, a));
            let db = dot(&antipode, &cell_center_vec(0, b));
            da.total_cmp(&db)
        })
        .unwrap();

    // Without the guard (complement=false) this base is pruned …
    assert!(
        base_node(far_base, &edges, &rings, &cap, backend, false).is_none(),
        "un-guarded cap cull should prune the antipodal base cell"
    );
    // … with the guard (complement=true) it is kept.
    assert!(
        base_node(far_base, &edges, &rings, &cap, backend, true).is_some(),
        "complement guard must keep the antipodal base cell"
    );
}

#[test]
fn test_covers_complement_multipart_two_caps() {
    // Multipart hemisphere+: two large polar caps (north and south) whose
    // combined CCW interior is >hemisphere. The even-odd parity at a single
    // antipode can misreport; probing every pruned base centre detects it.
    let north: Vec<Vec3> = (0..24)
        .map(|k| latlon_to_unit_vec(20.0, k as f64 * 15.0))
        .collect();
    let south: Vec<Vec3> = (0..24)
        .map(|k| latlon_to_unit_vec(-20.0, (24 - k) as f64 * 15.0))
        .collect();
    let rings = vec![north, south];
    let cap = Cap::of_rings(&rings);
    assert!(
        covers_complement(&rings, &cap),
        "multipart >hemisphere geometry must be detected as complement"
    );
}

#[test]
fn test_complement_guard_preserves_subhemisphere_coverage() {
    // The guard must be a no-op for sub-hemisphere polygons: coverage of a
    // mid-latitude square is unchanged (complement=false keeps the original
    // cull exactly). Byte-identical to the pre-guard behaviour.
    let lats = vec![40.0, 40.0, 50.0, 50.0];
    let lons = vec![-125.0, -115.0, -115.0, -125.0];
    let result = polygon_to_morton_coverage(&lats, &lons, 6);
    assert!(!result.is_empty());
    // Determinism / no spurious antipodal cells: a square this small must not
    // pull in any far-side base cell.
    let rings = vec![build_ring(&lats, &lons)];
    let cap = Cap::of_rings(&rings);
    assert!(!covers_complement(&rings, &cap));
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

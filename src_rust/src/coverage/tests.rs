//! Unit tests for the polygon coverage descent (issue #30/#22).
//!
//! Split out of `coverage.rs` to keep that module under the project's
//! ~1000-line soft limit; wired back in via `#[cfg(test)] mod tests;`.

use super::*;
use crate::sphere::{parity_filled_robust, parity_filled_with, ring_winding_sign};

#[test]
fn test_triangle_basic() {
    let lats = vec![40.0, 50.0, 45.0];
    let lons = vec![-120.0, -120.0, -110.0];
    let result = polygon_to_morton_coverage(&lats, &lons, 4, true);
    assert!(!result.is_empty(), "Coverage should not be empty");
    for &m in &result {
        assert!(m != 0, "Morton index should not be zero");
    }
}

#[test]
fn test_coverage_sorted_unique() {
    let lats = vec![40.0, 50.0, 45.0];
    let lons = vec![-120.0, -120.0, -110.0];
    let result = polygon_to_morton_coverage(&lats, &lons, 4, true);
    for i in 1..result.len() {
        assert!(
            result[i] > result[i - 1],
            "Result must be sorted and unique"
        );
    }
}

#[test]
fn test_square_coverage() {
    let lats = vec![40.0, 40.0, 50.0, 50.0];
    let lons = vec![-125.0, -115.0, -115.0, -125.0];
    let result = polygon_to_morton_coverage(&lats, &lons, 4, true);
    assert!(!result.is_empty());
}

#[test]
fn test_southern_hemisphere() {
    let lats = vec![-70.0, -80.0, -75.0];
    let lons = vec![30.0, 30.0, 50.0];
    let result = polygon_to_morton_coverage(&lats, &lons, 4, true);
    assert!(!result.is_empty());
    assert!(
        result.iter().any(|&m| m >= 1u64 << 63),
        "Southern hemisphere should set bit 63 on its morton words"
    );
}

#[test]
fn test_different_orders() {
    let lats = vec![40.0, 50.0, 45.0];
    let lons = vec![-120.0, -120.0, -110.0];
    let r4 = polygon_to_morton_coverage(&lats, &lons, 4, true);
    let r6 = polygon_to_morton_coverage(&lats, &lons, 6, true);
    assert!(
        r6.len() > r4.len(),
        "Higher order should produce more cells"
    );
}

#[test]
fn test_ingest_seeded_reference_matches_the_derived_one() {
    // `build_ring` hands the descent a `RingRef` it already established, so
    // `ring_reference` never runs for a ring ingest normalized.  That saves an
    // `O(V)` turning sum per ring per cover, and it is only sound because
    // normalization leaves the ring unambiguously counter-clockwise: a ring it
    // left alone read `+1` or `0`, one it reversed read `-1`, and reversing
    // negates every exterior angle.  If that ever stops holding, the seeded and
    // derived references disagree and the cover silently changes — so pin them
    // against each other directly.
    let cases: [(&str, Vec<Vec<f64>>, Vec<Vec<f64>>); 5] = [
        (
            "ccw square",
            vec![vec![40.0, 40.0, 50.0, 50.0]],
            vec![vec![-125.0, -115.0, -115.0, -125.0]],
        ),
        (
            "cw square (ingest must reverse it)",
            vec![vec![50.0, 50.0, 40.0, 40.0]],
            vec![vec![-125.0, -115.0, -115.0, -125.0]],
        ),
        (
            "donut: outer ccw, hole cw",
            vec![vec![35.0, 35.0, 55.0, 55.0], vec![48.0, 48.0, 42.0, 42.0]],
            vec![
                vec![-130.0, -110.0, -110.0, -130.0],
                vec![-123.0, -117.0, -117.0, -123.0],
            ],
        ),
        (
            "non-convex, axis in the concavity",
            vec![vec![10.0, 50.0, -10.0, -70.0, -10.0]],
            vec![vec![45.0, 45.0, 170.0, 225.0, 280.0]],
        ),
        (
            "thin sliver",
            vec![vec![0.0, 0.0, 1e-5, 1e-5]],
            vec![vec![0.0, 20.0, 20.0, 0.0]],
        ),
    ];
    for (name, lats, lons) in cases {
        let (rings, seeded) = build_rings(&lats, &lons, true);
        // The same rings, but with every reference derived from scratch.
        let derived = RingRefs::of_rings(&rings);
        let mut checked = 0;
        for lat in [-88.0, -55.0, -20.0, -3.0, 0.0, 7.0, 33.0, 45.0, 61.0, 89.0] {
            for lon in [-179.0, -123.0, -120.0, -60.0, 0.0, 44.0, 91.0, 178.0] {
                let p = latlon_to_unit_vec(lat, lon);
                assert_eq!(
                    parity_filled_with(&p, &rings, &seeded),
                    parity_filled_with(&p, &rings, &derived),
                    "{name}: seeded and derived references disagree at ({lat},{lon})"
                );
                checked += 1;
            }
        }
        assert!(checked == 80);
    }
}

#[test]
fn test_donut_carves_hole() {
    use crate::geo2mort::geo2mort_scalar;
    use std::collections::HashSet;
    // 20° outer box with a centred 6° hole, both around (45, -120).
    let lats = vec![vec![35.0, 35.0, 55.0, 55.0], vec![42.0, 42.0, 48.0, 48.0]];
    let lons = vec![
        vec![-130.0, -110.0, -110.0, -130.0],
        vec![-123.0, -117.0, -117.0, -123.0],
    ];
    let cov: HashSet<u64> = multipolygon_to_morton_coverage(&lats, &lons, 7, true)
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
fn test_issue11_meridian_box_no_descent_flood() {
    use crate::geo2mort::geo2mort_scalar;
    use std::collections::HashSet;
    // #11 at the descent level (the sphere-side PIP test only covers point
    // probes).  A 2°×2° box whose left edge lies exactly on the lon-45
    // base-cell-centre meridian — base cell 0's centre (lat 41.81°, lon 45°)
    // sits on that edge's great circle.  Seeding the even-odd fill walk from
    // that on-edge centre used to flip parity unstably and flood base cell 0's
    // whole interior (a depth-1 cell kept whole ⇒ 1024 spurious order-6 cells).
    // The SoS-robust crossing in `arc_crossing_parity` keeps it tight.
    let lats = vec![40.0, 40.0, 42.0, 42.0];
    let lons = vec![45.0, 47.0, 47.0, 45.0];
    let cov: HashSet<u64> = polygon_to_morton_coverage(&lats, &lons, 6, true)
        .into_iter()
        .collect();
    // A 2°×2° box at order 6 needs ~10 cells; a flood is >1000.  The tight
    // oracle (≤8 spurious vs cdshealpix) lives in the Python test; this is a
    // coarse Rust backstop with margin for cross-platform boundary-cell jitter.
    assert!(cov.len() < 40, "meridian box flooded: {} cells", cov.len());
    // The interior is covered; the far-east cell the flood wrongly filled is not.
    assert!(
        cov.contains(&geo2mort_scalar(41.0, 46.0, 6)),
        "interior covered"
    );
    assert!(
        !cov.contains(&geo2mort_scalar(41.81, 67.5, 6)),
        "far-east cell (the old flood) must not be covered"
    );
}

#[test]
fn test_multipart_disjoint_equals_union() {
    use std::collections::HashSet;
    let a_la = vec![40.0, 50.0, 45.0];
    let a_lo = vec![-120.0, -120.0, -110.0];
    let b_la = vec![10.0, 20.0, 15.0];
    let b_lo = vec![-80.0, -80.0, -70.0];
    let union: HashSet<u64> = polygon_to_morton_coverage(&a_la, &a_lo, 6, true)
        .into_iter()
        .chain(polygon_to_morton_coverage(&b_la, &b_lo, 6, true))
        .collect();
    let multi: HashSet<u64> =
        multipolygon_to_morton_coverage(&vec![a_la, b_la], &vec![a_lo, b_lo], 6, true)
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
    let exact = polygon_to_morton_moc(&lats, &lons, 10, true);
    let tol = polygon_to_morton_moc_tolerance(&lats, &lons, 10, 2.0_f64.to_radians(), true);
    assert!(!tol.is_empty());
    assert!(
        tol.len() <= exact.len(),
        "tolerance cover ({}) should not exceed exact ({})",
        tol.len(),
        exact.len()
    );
    // Determinism.
    assert_eq!(
        tol,
        polygon_to_morton_moc_tolerance(&lats, &lons, 10, 2.0_f64.to_radians(), true)
    );
}

#[test]
fn test_budget_respects_cap() {
    // The best-first budget must keep the cell count near the target and be
    // deterministic.
    let lats = vec![40.0, 40.0, 50.0, 50.0];
    let lons = vec![-125.0, -115.0, -115.0, -125.0];
    for budget in [20usize, 50, 200] {
        let (cov, effective) = polygon_to_morton_moc_budget(&lats, &lons, 12, budget, true);
        assert!(!cov.is_empty());
        // Soft target: at most one split (×4) past the effective budget.
        assert!(
            cov.len() <= effective + 4,
            "budget {} (eff {}) produced {} cells",
            budget,
            effective,
            cov.len()
        );
        assert_eq!(
            cov,
            polygon_to_morton_moc_budget(&lats, &lons, 12, budget, true).0
        );
    }
}

#[test]
fn test_moc_is_compact_and_densifies_to_flat() {
    // The MOC must be no larger than the flat cover and must densify back
    // to exactly the flat cover (densify-invariance).
    let lats = vec![40.0, 40.0, 50.0, 50.0];
    let lons = vec![-125.0, -115.0, -115.0, -125.0];
    let flat = polygon_to_morton_coverage(&lats, &lons, 8, true);
    let moc = polygon_to_morton_moc(&lats, &lons, 8, true);
    assert!(moc.len() <= flat.len(), "MOC should be compact");
    assert!(
        moc.len() < flat.len(),
        "interior should collapse to coarse cells"
    );
    assert_eq!(
        crate::moc::to_order(&moc, 8),
        flat,
        "MOC must densify to flat"
    );
}

#[test]
#[should_panic(expected = "at least 3 vertices")]
fn test_too_few_vertices() {
    polygon_to_morton_coverage(&[0.0, 1.0], &[0.0, 1.0], 4, true);
}

#[test]
#[should_panic(expected = "same length")]
fn test_mismatched_lengths() {
    polygon_to_morton_coverage(&[0.0, 1.0, 2.0], &[0.0, 1.0], 4, true);
}

#[test]
fn test_polar_polygon_deterministic() {
    // Regression test for issue #28: a thin near-polar lon-strip used to
    // produce one of two different cell sets at random.  The hierarchical
    // coverer is deterministic by construction and fills the interior.
    let lats = vec![-89.0, -59.09804617, -59.09804617, -89.0];
    let lons = vec![105.5108378, 105.5108378, 106.5108378, 106.5108378];

    let first = polygon_to_morton_coverage(&lats, &lons, 10, true);
    for _ in 0..50 {
        let r = polygon_to_morton_coverage(&lats, &lons, 10, true);
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
        -78.97166, -78.94929, -79.42733, -80.73793, -82.03867, -83.31985, -85.7817, -86.88342,
        -87.71538, -87.87437, -87.80769, -87.61281, -87.35613, -87.0609, -86.04897, -85.66109,
        -83.79971, -83.41805, -83.03733, -82.61643, -80.39302, -79.93122, -79.49362, -78.94943,
        -78.97178, -79.51702, -79.95551, -80.41857, -82.64886, -83.07129, -83.45395, -83.83797,
        -85.71503, -86.10799, -87.14043, -87.44464, -87.71114, -87.91512, -87.98525, -87.81823,
        -86.95811, -85.83679, -83.35487, -82.06846, -80.76386, -79.45036, -78.97166,
    ];
    let lons = vec![
        -37.60041, -38.19306, -38.71944, -40.42942, -42.66808, -45.72009, -57.13684, -69.29219,
        -92.07313, -102.96984, -139.16366, -149.26852, -157.63891, -164.28406, -177.35124,
        179.5542, 170.5179, 169.31794, 168.26288, 167.22786, 163.25287, 162.63733, 162.10694,
        161.50445, 160.91177, 161.48734, 161.99211, 162.57855, 166.3654, 167.35214, 168.3599,
        169.50821, 178.19957, -178.79874, -165.9332, -159.26658, -150.74238, -140.28136,
        -102.10046, -90.7387, -67.65821, -55.75165, -44.77371, -41.86274, -39.73073, -38.10328,
        -37.60041,
    ];
    let cover = polygon_to_morton_coverage(&lats, &lons, 8, true);
    // A covered order-8 packed word is a child of order-6 cell -6111131 iff its
    // order-6 ancestor equals that cell's packed word. After the issue #48 flip
    // ancestry is a kernel coarsen, not a decimal-digit divide.
    let target = crate::decimal_morton::from_legacy_decimal(-6111131);
    let hits = cover
        .iter()
        .filter(|&&m| crate::decimal_morton::coarsen(m, 6) == Some(target))
        .count();
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
    let result = polygon_to_morton_coverage(&lats, &lons, 4, true);
    let coverage_set: HashSet<u64> = result.into_iter().collect();

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
    let bands = [band];
    let cap = Cap::of_rings(&bands);
    assert!(
        covers_complement(&bands, &cap, &RingRefs::of_rings(&bands)),
        "hemisphere+ band must be detected as complement"
    );

    let square: Vec<Vec3> = [
        (40.0, -125.0),
        (40.0, -115.0),
        (50.0, -115.0),
        (50.0, -125.0),
    ]
    .iter()
    .map(|&(la, lo)| latlon_to_unit_vec(la, lo))
    .collect();
    let squares = [square];
    let cap2 = Cap::of_rings(&squares);
    assert!(
        !covers_complement(&squares, &cap2, &RingRefs::of_rings(&squares)),
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
    assert!(
        covers_complement(&rings, &cap, &RingRefs::of_rings(&rings)),
        "precondition: hemisphere+"
    );

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
        base_node(far_base, &edges, false, &cap, false).is_none(),
        "un-guarded cap cull should prune the antipodal base cell"
    );
    // … with the guard (complement=true) it is kept.
    assert!(
        base_node(far_base, &edges, false, &cap, true).is_some(),
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
        covers_complement(&rings, &cap, &RingRefs::of_rings(&rings)),
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
    let result = polygon_to_morton_coverage(&lats, &lons, 6, true);
    assert!(!result.is_empty());
    // Determinism / no spurious antipodal cells: a square this small must not
    // pull in any far-side base cell.
    let rings = vec![build_ring(&lats, &lons, true).0];
    let cap = Cap::of_rings(&rings);
    assert!(!covers_complement(
        &rings,
        &cap,
        &RingRefs::of_rings(&rings)
    ));
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

// ── orientation normalization at ingest (Phase 3, #22) ───────────────────

#[test]
fn test_build_ring_normalizes_cw_subhemisphere() {
    // A sub-hemisphere square handed in CW must be reversed to CCW by build_ring,
    // so a clearly-interior point reads inside under the robust winding fill.
    let lats = vec![40.0, 50.0, 50.0, 40.0]; // CW: down-the-other-way order
    let lons = vec![-125.0, -125.0, -115.0, -115.0];
    let ring = build_ring(&lats, &lons, true).0;
    assert_eq!(
        ring_winding_sign(&ring),
        1,
        "build_ring must emit a CCW (interior-on-left) ring"
    );
    assert!(
        parity_filled_robust(&latlon_to_unit_vec(45.0, -120.0), &[ring]),
        "centre of the normalized box must classify inside"
    );
}

#[test]
fn test_cw_and_ccw_input_give_same_coverage() {
    // The whole point of ingest normalization: a sub-hemisphere polygon and its
    // vertex-reversed twin must produce identical coverage (no inversion).
    let lats_ccw = vec![40.0, 40.0, 50.0, 50.0];
    let lons_ccw = vec![-125.0, -115.0, -115.0, -125.0];
    let mut lats_cw = lats_ccw.clone();
    let mut lons_cw = lons_ccw.clone();
    lats_cw.reverse();
    lons_cw.reverse();
    let ccw = polygon_to_morton_coverage(&lats_ccw, &lons_ccw, 5, true);
    let cw = polygon_to_morton_coverage(&lats_cw, &lons_cw, 5, true);
    assert!(!ccw.is_empty());
    assert_eq!(ccw, cw, "CW input must give the same cover as CCW input");
}

#[test]
fn test_build_ring_trusts_order_for_hemisphere_plus_vertices() {
    // A ring whose *vertices* do not fit in a sub-hemisphere cap (they span the
    // whole sphere here) is past the normalization regime: build_ring must trust
    // the vertex order exactly, never reorder. Proof: the as-given ring and its
    // reversed twin select *complementary* interiors (one classifies a probe
    // inside, the other outside) — impossible if ingest forced one orientation.
    let lats = vec![80.0, 0.0, -80.0, 0.0];
    let lons = vec![0.0, 90.0, 180.0, -90.0];
    let ring = build_ring(&lats, &lons, true).0;
    let mut lats_rev = lats.clone();
    let mut lons_rev = lons.clone();
    lats_rev.reverse();
    lons_rev.reverse();
    let ring_rev = build_ring(&lats_rev, &lons_rev, true).0;
    let probe = latlon_to_unit_vec(0.0, 0.0);
    assert_ne!(
        parity_filled_robust(&probe, &[ring]),
        parity_filled_robust(&probe, &[ring_rev]),
        "hemisphere+ (whole-sphere-spanning) vertices must keep their order: \
         reversing selects the complement, so ingest did not normalize"
    );
}

#[test]
fn test_build_ring_subhemisphere_takes_smaller_side() {
    // Documented sub-hemisphere policy (#22): when a ring's vertices fit inside a
    // hemisphere, the *smaller* of the two regions is the interior, regardless of
    // input winding. A lat -10 band across all longitudes has sub-hemisphere
    // vertices (an ~80° cap about the south pole), so its smaller region — the
    // south cap — is interior; the north pole is OUTSIDE. Both windings agree,
    // because ingest normalizes the CW one. (To select the large north side a
    // caller must express it as hemisphere+ vertices or a world-minus-hole.)
    // Both vertex orderings (longitude increasing and decreasing) must give the
    // same cover: whichever is wound CW about the cap axis is reversed at ingest.
    let lats: Vec<f64> = (0..36).map(|_| -10.0).collect();
    let lons_inc: Vec<f64> = (0..36).map(|k| k as f64 * 10.0).collect();
    let lons_dec: Vec<f64> = lons_inc.iter().rev().copied().collect();
    for lons in [&lons_inc, &lons_dec] {
        let ring = build_ring(&lats, lons, true).0;
        assert_eq!(
            ring_winding_sign(&ring),
            1,
            "normalized band must be CCW (small side on the left)"
        );
        assert!(
            parity_filled_robust(&latlon_to_unit_vec(-80.0, 0.0), &[ring.clone()]),
            "south cap (smaller side) is interior"
        );
        assert!(
            !parity_filled_robust(&latlon_to_unit_vec(80.0, 0.0), &[ring]),
            "north (larger side) is outside the normalized sub-hemisphere band"
        );
    }
}

#[test]
fn test_build_ring_normalize_false_trusts_subhemisphere_order() {
    // normalize=false must skip the ingest reorder even for a sub-hemisphere ring.
    // A correctly-oriented (CCW, interior-on-left) sub-hemisphere square: with the
    // flag off, build_ring leaves it CCW and its centre classifies inside.
    let lats_ccw = vec![40.0, 40.0, 50.0, 50.0];
    let lons_ccw = vec![-125.0, -115.0, -115.0, -125.0];
    let ring = build_ring(&lats_ccw, &lons_ccw, false).0;
    assert_eq!(
        ring_winding_sign(&ring),
        1,
        "normalize=false must leave a CCW ring untouched"
    );
    assert!(
        parity_filled_robust(&latlon_to_unit_vec(45.0, -120.0), &[ring]),
        "centre of a correctly-wound box classifies inside with normalize=false"
    );

    // And the CW twin is NOT corrected: with normalize=false it selects the
    // complement (centre reads outside), whereas normalize=true would reverse it.
    let lats_cw: Vec<f64> = lats_ccw.iter().rev().copied().collect();
    let lons_cw: Vec<f64> = lons_ccw.iter().rev().copied().collect();
    let ring_cw = build_ring(&lats_cw, &lons_cw, false).0;
    assert_eq!(
        ring_winding_sign(&ring_cw),
        -1,
        "normalize=false must preserve a CW ring's clockwise order"
    );
    assert!(
        !parity_filled_robust(&latlon_to_unit_vec(45.0, -120.0), &[ring_cw]),
        "CW ring left as-is selects the complement (centre outside) under normalize=false"
    );
}

#[test]
fn test_base_fills_chain_no_antipodal_phantom() {
    // Phase-2 review reproducer (#103): a ring placed so the bare
    // two-straddle test fires on the *far* (antipodal) intersection for one
    // of base_fills' long donor→seed chords, silently inverting the seed.
    // The chain now uses the full symbolic predicate unconditionally: every
    // seed the winding backend can classify unambiguously must agree with it.
    let lats = vec![10.0, 50.0, -10.0, -70.0, -10.0];
    let lons = vec![45.0, 45.0, 170.0, 225.0, 280.0];
    let (rings, refs) = build_rings(&[lats], &[lons], true);
    let edges = build_edges(&rings, 6);
    let cap = Cap::of_rings(&rings);
    let complement = covers_complement(&rings, &cap, &refs);
    let fills = base_fills(&edges, &rings, &cap, complement, &refs);
    let units: Vec<Vec3> = edges.iter().map(|e| normalize(&e.n_ab)).collect();
    for b in 0..12u64 {
        let c = cell_center_vec(0, b);
        if seed_fill(&c, center_id(0, b), &units, &rings, &refs).is_some() {
            assert_eq!(
                fills[b as usize],
                parity_filled_robust(&c, &rings),
                "seed {b} inverted"
            );
        }
    }
}

#[test]
fn test_descent_hemisphere_ring_collinear_edge_oracle() {
    // The issue #107 acceptance gate, formerly #[ignore]d: the full descent
    // under the debug parity oracle on the phase-2 review's adversarial ring
    // (first edge 40° along the lon-45 meridian, collinear with the probe
    // lattice).  The pre-phase-1 desync was the float reference layer, not
    // the chain (see sphere.rs, "chain consistency & S2 correspondence");
    // with the reference repaired the oracle validates every uniform cell.
    let lats = vec![10.0, 50.0, -10.0, -70.0, -10.0];
    let lons = vec![45.0, 45.0, 170.0, 225.0, 280.0];
    let cov = polygon_to_morton_coverage(&lats, &lons, 6, true);
    assert!(!cov.is_empty());
}

#[test]
fn test_descent_collinear_edge_matrix() {
    // The oracle gate above, widened along the two adversarial axes: vertex
    // rotation (renumbers every SoS id) and winding (the reversed hemisphere+
    // ring selects the complement).  Each descent runs under the debug parity
    // oracle, which validates every uniform cell's fill against the reference;
    // on top of that, rotation must be a no-op on the *cover itself* — a
    // rotation-dependent cover would pass a per-configuration check while
    // still proving the SoS renumbering leaked into the verdict, so each
    // rotation is compared cell-for-cell against rotation 0 of its winding.
    let lats = [10.0, 50.0, -10.0, -70.0, -10.0];
    let lons = [45.0, 45.0, 170.0, 225.0, 280.0];
    for rev in [false, true] {
        let mut reference: Option<std::collections::HashSet<u64>> = None;
        for rot in 0..5 {
            let mut la: Vec<f64> = lats.to_vec();
            let mut lo: Vec<f64> = lons.to_vec();
            la.rotate_left(rot);
            lo.rotate_left(rot);
            if rev {
                la.reverse();
                lo.reverse();
            }
            let cov: std::collections::HashSet<u64> = polygon_to_morton_coverage(&la, &lo, 5, true)
                .into_iter()
                .collect();
            assert!(!cov.is_empty(), "rot {rot} rev {rev}");
            match &reference {
                None => reference = Some(cov),
                Some(r) => assert_eq!(
                    &cov,
                    r,
                    "vertex rotation changed the cover at rot {rot} rev {rev} \
                     ({} cells differ)",
                    cov.symmetric_difference(r).count()
                ),
            }
        }
    }
}

#[test]
fn test_descent_collinear_edge_nudge_is_a_boundary_band() {
    // Release-mode symptom of the pre-repair defect: nudging the collinear
    // edge off the lattice by 1e-6° moved the cover by 8 221 of ~25k cells at
    // order 6 (a subtree inversion).  Post-repair the two covers differ by a
    // boundary band only — measured 9 cells.  The bound is 12, not something
    // looser: the un-nudged cover's MOC has 70 uniform cells at depth 4, each
    // worth 4^2 = 16 order-6 cells, so any bound at or above 16 would admit a
    // whole-subtree inversion — the one thing this assertion exists to
    // exclude.  (In a `cargo test` build the debug parity oracle inside
    // `polygon_to_morton_coverage` fires first on an inversion; this is the
    // release-visible symptom.)
    //
    // `normalize=false` since phase 4: this pins the *chain's* stability on
    // the as-given winding, independent of the decision-(A) ingest flip
    // (issue #144) — under `normalize=true` both rings now flip to the small
    // side and the band along the flipped boundary measures 29 cells, which
    // says nothing this test is after.  The (A) flip itself is pinned by
    // `test_descent_collinear_cover_count_pinned`.
    let lats = vec![10.0, 50.0, -10.0, -70.0, -10.0];
    let lons_exact = vec![45.0, 45.0, 170.0, 225.0, 280.0];
    let lons_nudged = vec![45.0, 45.000001, 170.0, 225.0, 280.0];
    let a: std::collections::HashSet<u64> =
        polygon_to_morton_coverage(&lats, &lons_exact, 6, false)
            .into_iter()
            .collect();
    let b: std::collections::HashSet<u64> =
        polygon_to_morton_coverage(&lats, &lons_nudged, 6, false)
            .into_iter()
            .collect();
    let sym_diff = a.symmetric_difference(&b).count();
    assert!(
        sym_diff <= 12,
        "collinear-edge nudge moved {sym_diff} cells — subtree inversion?"
    );
}

#[test]
fn test_base_fills_oversize_cap_classifies_all_seeds() {
    // A ring whose vertex cap exceeds 90°: cap convexity no longer bounds
    // the boundary, so the cull-skip must disengage and every seed the
    // winding backend can classify must carry its own verdict (PR #109
    // review follow-up) — a culled false would not be provably exact here.
    let lats = vec![80.0, -10.0, -60.0, -10.0, 80.0];
    let lons = vec![0.0, 60.0, 130.0, 200.0, 260.0];
    let (rings, refs) = build_rings(&[lats], &[lons], true);
    let edges = build_edges(&rings, 6);
    let cap = Cap::of_rings(&rings);
    assert!(
        cap.radius > std::f64::consts::FRAC_PI_2,
        "cap must be oversize"
    );
    let complement = covers_complement(&rings, &cap, &refs);
    let fills = base_fills(&edges, &rings, &cap, complement, &refs);
    let units: Vec<Vec3> = edges.iter().map(|e| normalize(&e.n_ab)).collect();
    for b in 0..12u64 {
        let c = cell_center_vec(0, b);
        if seed_fill(&c, center_id(0, b), &units, &rings, &refs).is_some() {
            assert_eq!(
                fills[b as usize],
                parity_filled_robust(&c, &rings),
                "seed {b} diverged under an oversize cap"
            );
        }
    }
}

#[test]
fn test_descent_collinear_cover_count_pinned() {
    // Two exact pins on the reproducer at order 6, deterministic (exact
    // predicates, fixed traversal):
    //
    // * `normalize=false` (the as-given winding) pins 25 586.  It read
    //   25 577 through phase 3's identity threading (the id-rank invariant:
    //   `PROBE_ID` and `center_id` rank identically against every vertex id,
    //   so threading moved nothing) and grew by 9 cells (+0.04%) when the
    //   closed-set incidence branch went from a bit-exact zero to
    //   `straddle_error_bound` (issue #117 item 1).  Growth only: those nine
    //   cells are boundary-touching under the #103 contract and had been
    //   resolved to one side by rounding.
    // * `normalize=true` pins the **decision-(A)** flip (issue #144): this
    //   simple hemisphere-plus ring's as-given interior is the *larger*
    //   region (52.0% of the sphere), so ingest reverses it and covers the
    //   smaller side — 24 213 cells (49.3%; the counts overlap on the
    //   boundary fringe).  Expressing the 52% side takes `normalize=false`.
    let lats = vec![10.0, 50.0, -10.0, -70.0, -10.0];
    let lons = vec![45.0, 45.0, 170.0, 225.0, 280.0];
    let as_given = polygon_to_morton_coverage(&lats, &lons, 6, false);
    assert_eq!(as_given.len(), 25586, "as-given cover moved");
    let normalized = polygon_to_morton_coverage(&lats, &lons, 6, true);
    assert_eq!(
        normalized.len(),
        24213,
        "decision-(A) normalized cover moved"
    );
    assert!(
        normalized.len() < as_given.len(),
        "(A) must pick the small side"
    );
}

#[test]
fn test_build_ring_normalizes_the_crescent_and_hemisphere_plus() {
    // Decision (A) at the cover level (issue #144).  The crescent reproducer
    // — lat 5–10°, 300° of longitude, defeats the vertex-sum cap — must now
    // normalize: both windings cover the *small* side, and the covers are
    // identical.  Before (A), the clockwise spelling covered 98.3% of the
    // sphere at order 5 (12 077 of 12 288 cells).
    let lats: Vec<f64> = (0..40).map(|_| 5.0).chain((0..40).map(|_| 10.0)).collect();
    let lons: Vec<f64> = (0..40)
        .map(|k| k as f64 * 300.0 / 39.0)
        .chain((0..40).map(|k| 300.0 - k as f64 * 300.0 / 39.0))
        .collect();
    let ccw: std::collections::HashSet<u64> = polygon_to_morton_coverage(&lats, &lons, 5, true)
        .into_iter()
        .collect();
    let lats_r: Vec<f64> = lats.iter().rev().copied().collect();
    let lons_r: Vec<f64> = lons.iter().rev().copied().collect();
    let cw: std::collections::HashSet<u64> = polygon_to_morton_coverage(&lats_r, &lons_r, 5, true)
        .into_iter()
        .collect();
    assert_eq!(ccw, cw, "both windings must give the same normalized cover");
    // S2 puts the interior at 3.61% of the sphere; the cover is that plus a
    // boundary fringe, nowhere near the 98.3% complement.
    let frac = ccw.len() as f64 / 12288.0;
    assert!(
        (0.03..0.08).contains(&frac),
        "crescent cover fraction {frac:.4} is not the small side"
    );
    // And the escape hatch still expresses the big side.
    let cw_raw = polygon_to_morton_coverage(&lats_r, &lons_r, 5, false);
    assert!(
        cw_raw.len() as f64 / 12288.0 > 0.9,
        "normalize=false must keep the as-given complement"
    );
}

// ── closed-set incidence: the vertex-point-touch gap (#117 item 1, in #107) ──

/// `straddle_error_bound` must dominate the actual rounding of the determinant
/// it bounds: perturb a point off a great circle by well under the bound and
/// the computed determinant must stay inside it.
#[test]
fn test_straddle_error_bound_dominates_the_rounding_it_bounds() {
    // Two cell corners define the circle; the third point is a corner of the
    // *neighbouring* cell at the same geometric position, i.e. the same point
    // reached by a different computation — exactly the configuration the bound
    // exists for.
    let mut checked = 0usize;
    for depth in 3..9u8 {
        for pixel in [7u64, 40, 111, 250] {
            let c = cell_corners(depth, pixel);
            for i in 0..4 {
                let (c1, c2) = (c[i], c[(i + 1) % 4]);
                // a point on the arc: the normalized midpoint.  Its determinant
                // against the arc's own normal is zero up to rounding.
                let mid = normalize(&[c1[0] + c2[0], c1[1] + c2[1], c1[2] + c2[2]]);
                let d = dot(&cross(&c1, &c2), &mid);
                let bound = straddle_error_bound(&c1, &c2, &mid);
                assert!(
                    d.abs() <= bound,
                    "depth {depth} pixel {pixel} side {i}: |d| = {d:e} exceeds bound {bound:e}"
                );
                checked += 1;
            }
        }
    }
    assert!(checked >= 90, "expected a real sample, got {checked}");
}

/// A point provably off the great circle must stay *outside* the bound, so the
/// widening cannot swallow genuine sidedness.
#[test]
fn test_straddle_error_bound_does_not_swallow_a_real_offset() {
    let c = cell_corners(5, 100);
    let (c1, c2) = (c[0], c[1]);
    let n = normalize(&cross(&c1, &c2));
    let mid = normalize(&[c1[0] + c2[0], c1[1] + c2[1], c1[2] + c2[2]]);
    // step off the circle by 1e-12 rad — a million times the bound's scale
    let off = normalize(&[
        mid[0] + 1e-12 * n[0],
        mid[1] + 1e-12 * n[1],
        mid[2] + 1e-12 * n[2],
    ]);
    let d = dot(&cross(&c1, &c2), &off);
    assert!(
        d.abs() > straddle_error_bound(&c1, &c2, &off),
        "a 1e-12 rad offset must remain provably off the circle"
    );
}

/// The closed-set contract at a shared cell corner: a polygon whose only
/// contact with a cell is one vertex landing on that cell's corner must still
/// put the cell in the cover.  Before the error-bounded incidence test this
/// held only for the vertex's own leaf-owning cell (issue #117 item 1).
#[test]
fn test_vertex_on_shared_corner_includes_every_incident_cell() {
    let order: u8 = 5;
    // north corner of the cell holding (32, 47) — shared by four cells
    let v_lat = 34.228866327812575_f64;
    let v_lon = 46.40625_f64;
    let incident = [
        1424263382155919365u64,
        1426515181969604613,
        1427641081876447237,
        1429892881690132485,
    ];
    // apex exactly on the corner, body inside the cell south of it
    let (c_lat, c_lon) = (32.79716829582364_f64, 46.40625_f64);
    let (m_lat, m_lon) = (
        v_lat + (c_lat - v_lat) * 0.15,
        v_lon + (c_lon - v_lon) * 0.15,
    );
    let (p_lat, p_lon) = (-(c_lon - v_lon) * 0.15, (c_lat - v_lat) * 0.15);
    let lats = vec![v_lat, m_lat + p_lat, m_lat - p_lat];
    let lons = vec![v_lon, m_lon + p_lon, m_lon - p_lon];
    let cov = polygon_to_morton_coverage(&lats, &lons, order, true);
    for cell in incident {
        assert!(
            cov.contains(&cell),
            "cell {cell} shares the touched corner but is missing from {cov:?}"
        );
    }
}

/// The widening never removes covered *area* — it turns "provably one side"
/// into "indistinguishable from incidence", and incidence is included.
///
/// Note the invariant is area, not cell count: on the MOC paths an added
/// boundary cell can complete a parent's four children and merge, so the count
/// can fall while the area rises (measured: 3 cells over 3 leaves becomes 1
/// cell over 4).  Flat covers at a fixed order have no merge step, so there the
/// count is the area and growth is the whole story.  Sizes are pinned against
/// the values measured before the widening so a future change has to say so.
///
/// Phase 2's vertex-neighbourhood clause leaves all four unchanged: it selects
/// only the cells that share the touched side or corner, and on these polygons
/// those were already reached by the quad test.  What it adds is the
/// point-touch case the descent used to prune — see
/// `test_apex_on_shared_corner_survives_the_ancestor_walk`.
#[test]
fn test_closed_set_widening_never_loses_area() {
    // (lats, lons, order, size before the widening, size after)
    let cases: [(&[f64], &[f64], u8, usize, usize); 4] = [
        (
            &[0.0, 0.0, 11.25, 11.25],
            &[0.0, 11.25, 11.25, 0.0],
            6,
            198,
            199,
        ),
        (
            &[0.0, 0.0, 22.5, 22.5],
            &[45.0, 67.5, 67.5, 45.0],
            4,
            63,
            64,
        ),
        (&[40.0, 50.0, 45.0], &[-120.0, -120.0, -110.0], 5, 21, 23),
        (&[70.0, 80.0, 75.0], &[10.0, 10.0, 40.0], 5, 23, 24),
    ];
    for (lats, lons, order, before, after) in cases {
        let cov = polygon_to_morton_coverage(lats, lons, order, true);
        assert!(
            cov.len() >= before,
            "cover shrank: {} < {before} (order {order})",
            cov.len()
        );
        assert_eq!(cov.len(), after, "cover size moved (order {order})");
    }
}

/// The MOC entry points share the predicate, so the area invariant has to hold
/// there too — and there the *cell count* legitimately falls when an added
/// boundary cell completes a parent's four children.  Comparing densified leaf
/// sets keeps the assertion on area rather than representation.
#[test]
fn test_moc_entry_points_never_lose_leaf_area() {
    let lats = vec![40.0, 40.0, 50.0, 50.0];
    let lons = vec![-125.0, -115.0, -115.0, -125.0];
    let order = 7;
    let flat: std::collections::HashSet<u64> =
        polygon_to_morton_coverage(&lats, &lons, order, true)
            .into_iter()
            .collect();
    // plain MOC, tolerance MOC and budget MOC all densify to a superset of the
    // flat cover — the flat cover is the same descent without the merge step.
    let moc = polygon_to_morton_moc(&lats, &lons, order, true);
    let dense: std::collections::HashSet<u64> =
        crate::moc::to_order(&moc, order).into_iter().collect();
    assert_eq!(dense, flat, "MOC densifies to a different leaf set");

    let coarse = polygon_to_morton_moc_tolerance(&lats, &lons, order, 0.01, true);
    let coarse_dense: std::collections::HashSet<u64> =
        crate::moc::to_order(&coarse, order).into_iter().collect();
    assert!(
        flat.is_subset(&coarse_dense),
        "tolerance MOC dropped {} leaves the flat cover holds",
        flat.difference(&coarse_dense).count()
    );

    let (budget, effective) = polygon_to_morton_moc_budget(&lats, &lons, order, 64, true);
    let budget_dense: std::collections::HashSet<u64> =
        crate::moc::to_order(&budget, order).into_iter().collect();
    assert!(
        flat.is_subset(&budget_dense),
        "budget MOC (effective {effective}) dropped {} leaves",
        flat.difference(&budget_dense).count()
    );
}

/// The scale gate is the one tuned number in the change, so exercise it
/// directly at both ends: an order-6-scale arc keeps the widening, an
/// order-29-scale arc gives it up and defers to the symbolic path.
#[test]
fn test_incidence_slack_gate_disengages_on_sub_ulp_arcs() {
    let long = cell_corners(6, ang2pix_scalar(6, -76.55, 38.9));
    let (l1, l2) = (long[0], long[1]);
    let l_mid = normalize(&[l1[0] + l2[0], l1[1] + l2[1], l1[2] + l2[2]]);
    let n_long = cross(&l1, &l2);
    assert!(
        indistinguishable_from_zero(
            dot(&n_long, &l_mid),
            &l1,
            &l2,
            &l_mid,
            dot(&n_long, &n_long)
        ),
        "an order-6 arc must keep the widened incidence test"
    );

    // mid-latitude, so the vectors' components are O(1) and the permanent is
    // not shrunk by proximity to a pole (which would shrink the bound too).
    let tiny = cell_corners(29, ang2pix_scalar(29, -76.55, 38.9));
    let (t1, t2) = (tiny[0], tiny[1]);
    let t_mid = normalize(&[t1[0] + t2[0], t1[1] + t2[1], t1[2] + t2[2]]);
    let n_tiny = cross(&t1, &t2);
    assert!(
        !indistinguishable_from_zero(
            dot(&n_tiny, &t_mid),
            &t1,
            &t2,
            &t_mid,
            dot(&n_tiny, &n_tiny)
        ),
        "an order-29 arc must disengage the gate and defer to arcs_cross_sos"
    );
}

// ── the point-touch descent residual (issue #107 phase 2, option (a′)) ──────

/// The ancestor-walk case from the [corrected option
/// list](https://github.com/espg/mortie/pull/148#issuecomment-5090814161), and
/// the acceptance gate for the combinatorial vertex clause.
///
/// A triangle whose apex sits on the corner shared by four order-4 cells, its
/// body inside `1418633882621706244`.  Phase 1 made every one of the four
/// register the touch *at depth 4* — and two of them were still missing from
/// the cover, because the apex lies on the boundary between **depth-3** cells
/// 14 and 15 and cell 15's four-corner quad is a chord that the apex is
/// provably off, so the whole subtree under it was pruned before depth 4 was
/// reached.  `1423137482249076740` is the cell from that walk;
/// `1432144681503817732` is the other side of the same corner.
#[test]
fn test_apex_on_shared_corner_survives_the_ancestor_walk() {
    let lats = vec![32.79716829582364, 33.21904329582364, 32.37529329582364];
    let lons = vec![42.1875, 41.765625, 41.765625];
    let cov = polygon_to_morton_coverage(&lats, &lons, 4, true);
    for cell in [
        1409626683366965252u64,
        1418633882621706244, // the body's own cell (covered before phase 2)
        1423137482249076740, // the ancestor-walk case: pruned at depth 3
        1432144681503817732,
    ] {
        assert!(
            cov.contains(&cell),
            "cell {cell} shares the touched corner but is missing from {cov:?}"
        );
    }
    // Nothing beyond the four: the 8-neighbourhood is an upper bound on what
    // the touch can reach, and the cells it does not reach are culled by
    // `edge_relevant` before clause (1b) ever runs.
    assert_eq!(cov.len(), 4, "the expansion pulled in extra cells: {cov:?}");
}

/// The gate, asserted from both sides: a vertex well inside its leaf expands to
/// nothing, and a vertex on the boundary expands to exactly the neighbours that
/// share what it lands on.  Without the gate every polygon vertex would drag
/// its neighbours into the descent; without the selection a corner touch would
/// drag in all eight.
///
/// The expected sets are built from the neighbours' *own* corners — geometry
/// the clause itself never consults — so this pins the side→direction map
/// (`[N, W, S, E]` corner order ⇒ NW / SW / SE / NE across the sides)
/// independently of the code under test.
#[test]
fn test_neighbourhood_expansion_is_gated_and_selective() {
    let order: u8 = 6;
    let leaf = ang2pix_scalar(order, 47.0, 32.0);
    let center = cell_center_vec(order, leaf);
    assert!(
        boundary_incident_neighbourhood(&center, leaf, order).is_empty(),
        "a vertex at the cell centre must not expand"
    );

    // Loose on purpose.  Adjacent cells report their shared corner up to
    // ~2.6e-8 rad apart (the PR's question (2)), which is exactly why the
    // clause itself never asks the neighbours where their corners are — but a
    // test may, at a tolerance far above that quantization and far below the
    // ~1.6e-2 rad an order-6 cell spans, where the answer is unambiguous.
    const SAME_POINT: f64 = 1e-6;
    let sep = |p: &Vec3, q: &Vec3| dot(p, q).clamp(-1.0, 1.0).acos();
    let ring = healpix::get(order).kth_neighborhood(leaf, 1);
    let touching = |v: &Vec3| -> Vec<u64> {
        let mut hit: Vec<u64> = ring
            .iter()
            .copied()
            .filter(|&h| {
                h != leaf
                    && cell_corners(order, h)
                        .iter()
                        .any(|c| sep(c, v) < SAME_POINT)
            })
            .collect();
        hit.sort_unstable();
        hit
    };

    let corners = cell_corners(order, leaf);
    for (i, corner) in corners.iter().enumerate() {
        let mut got = boundary_incident_neighbourhood(corner, leaf, order).to_vec();
        got.sort_unstable();
        assert_eq!(
            got.len(),
            3,
            "corner {i} must reach three neighbours: {got:?}"
        );
        assert_eq!(got, touching(corner), "corner {i} reached the wrong cells");
    }

    // A side's midpoint is on one side only, so it reaches one cell — the one
    // that shares *both* endpoints of that side.
    for i in 0..4 {
        let (c1, c2) = (corners[i], corners[(i + 1) % 4]);
        let mid = normalize(&[c1[0] + c2[0], c1[1] + c2[1], c1[2] + c2[2]]);
        let got = boundary_incident_neighbourhood(&mid, leaf, order);
        assert_eq!(got.len(), 1, "side {i} must reach one neighbour: {got:?}");
        let across = cell_corners(order, got[0]);
        for (which, end) in [("start", c1), ("end", c2)] {
            assert!(
                across.iter().any(|c| sep(c, &end) < SAME_POINT),
                "the cell across side {i} does not share the side's {which}"
            );
        }
    }
}

/// The same gate through the descent: a triangle strictly inside one cell
/// covers exactly that cell.  A regression that dropped the gate would ring it
/// with its neighbours.
#[test]
fn test_interior_triangle_does_not_drag_in_neighbours() {
    let lats = vec![32.59716829582364, 32.99716829582365, 32.797168295823646];
    let lons = vec![44.8, 44.8, 45.2];
    let cov = polygon_to_morton_coverage(&lats, &lons, 4, true);
    assert_eq!(cov, vec![1423137482249076740], "cover grew: {cov:?}");
}

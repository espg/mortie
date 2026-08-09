//! Tests for the dissolve pipeline (`src_rust/src/dissolve.rs`) --- split
//! into a sibling file per the `sphere/tests.rs` layout so the module body
//! stays reviewable (issue #147).

use super::*;

#[test]
fn cut_no_crossing_returns_whole() {
    let ring = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
    let got = cut_at_antimeridian(&ring).unwrap();
    assert_eq!(got.len(), ring.len() + 1); // closed
    assert_eq!(got[0], got[got.len() - 1]);
}

#[test]
fn cut_two_crossings_splits_each_side() {
    // box straddling +/-180: two segments, one per hemisphere side.
    let ring = [(170.0, 0.0), (-170.0, 0.0), (-170.0, 10.0), (170.0, 10.0)];
    let segs = cut_at_antimeridian(&ring).unwrap_err();
    assert_eq!(segs.len(), 2);
    for s in &segs {
        assert!((s[0].0.abs() - 180.0).abs() < 1e-9);
        assert!((s[s.len() - 1].0.abs() - 180.0).abs() < 1e-9);
    }
    // no pole wrap needed -> each segment closes on its own side.
    let rings = stitch_segments(segs).unwrap();
    assert_eq!(rings.len(), 2);
    for r in &rings {
        let span = r.iter().map(|p| p.0).fold(f64::MIN, f64::max)
            - r.iter().map(|p| p.0).fold(f64::MAX, f64::min);
        assert!(span <= 180.0 + 1e-9);
    }
}

// ── point-sampling oracle (issues #147/#181 fixtures) ──────────────────

/// Even-odd planar point test (mirrors `mortie/dissolve.py::_point_in_ring`).
fn planar_point_in_ring(x: f64, y: f64, ring: &[(f64, f64)]) -> bool {
    let n = ring.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = ring[i];
        let (xj, yj) = ring[j];
        if (yi > y) != (yj > y) {
            let x_cross = xi + (y - yi) / (yj - yi) * (xj - xi);
            if x < x_cross {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// Even-odd planar containment over all emitted rings (shells + holes).
fn emitted_contains(out: &ClassifiedRings, lon: f64, lat: f64) -> bool {
    let mut inside = false;
    for ring in out.shells.iter().chain(out.holes.iter()) {
        if planar_point_in_ring(lon, lat, ring) {
            inside = !inside;
        }
    }
    inside
}

/// Point-sample an emitted outline against its source cover: at every
/// cell centre of `sample_order`, planar containment in the emitted
/// rings must equal exact cover membership.  Centres exactly on the
/// ±180 seam are nudged 1e-4° west (membership is re-evaluated at the
/// nudged point, so the comparison stays exact); all other centres are
/// strictly interior to their covering cell, hence bounded away from
/// the emitted outline.
fn assert_point_sampled(cover: &[u64], out: &ClassifiedRings, sample_order: u8) {
    let depths: Vec<u8> = cover.iter().map(|&w| mort2nested(w).1).collect();
    let max_depth = *depths.iter().max().unwrap();
    let flat: Vec<u64> = if depths.iter().any(|&d| d != max_depth) {
        moc::to_order(cover, max_depth)
    } else {
        cover.to_vec()
    };
    let covered: std::collections::HashSet<u64> = flat.iter().copied().collect();
    assert!(
        sample_order >= max_depth,
        "sample centres coarser than the cover sit on cell corners"
    );
    // The emit is a straight-chord (step-driven) discretization of the true
    // cell-edge arcs, so pointwise agreement with the exact cover is only
    // required for samples farther from the emitted boundary than the
    // chord-vs-arc gap; the band shrinks with cell size.
    let band = 4.0 / f64::from(1u32 << max_depth);
    let ncells = 12u64 << (2 * sample_order);
    let mut checked = 0u64;
    for pix in 0..ncells {
        let (mut lon, lat) = crate::geo2mort::pix2ang_scalar(sample_order, pix);
        if lon > 180.0 {
            lon -= 360.0;
        }
        if (lon.abs() - 180.0).abs() < 1e-9 {
            lon = 180.0 - 1e-4;
        }
        let w = crate::geo2mort::geo2mort_scalar(lat, lon, max_depth);
        if emitted_contains(out, lon, lat) != covered.contains(&w) {
            let d = planar_distance_to_rings(out, lon, lat);
            assert!(
                d < band,
                "sample pix {pix} at lon {lon} lat {lat}: mismatch {d} deg \
                 from the emitted boundary (band {band})"
            );
        }
        checked += 1;
    }
    assert_eq!(checked, ncells);
}

/// Min planar distance (degrees) from a point to any emitted ring segment.
fn planar_distance_to_rings(out: &ClassifiedRings, x: f64, y: f64) -> f64 {
    let mut best = f64::MAX;
    for ring in out.shells.iter().chain(out.holes.iter()) {
        for w in ring.windows(2) {
            let ((x0, y0), (x1, y1)) = (w[0], w[1]);
            let (dx, dy) = (x1 - x0, y1 - y0);
            let len2 = dx * dx + dy * dy;
            let t = if len2 > 0.0 {
                (((x - x0) * dx + (y - y0) * dy) / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let (px, py) = (x0 + t * dx, y0 + t * dy);
            let d = ((x - px) * (x - px) + (y - py) * (y - py)).sqrt();
            best = best.min(d);
        }
    }
    best
}

fn base_cells_cover(bases: &[u64], order: u8) -> Vec<u64> {
    let n = 1u64 << (2 * order);
    bases
        .iter()
        .flat_map(|&b| (b * n..(b + 1) * n).map(|c| crate::morton::nested2mort(c, order)))
        .collect()
}

/// Exact covered area (steradians) of a cover: Σ π/(3·4^depth).
fn cover_area(morton: &[u64]) -> f64 {
    morton
        .iter()
        .map(|&w| std::f64::consts::PI / (3.0 * 4f64.powi(mort2nested(w).1 as i32)))
        .sum()
}

// ── issue #147: hemisphere+ covers dissolve (fail-loud fixtures flip) ──

#[test]
fn hemisphere_cover_dissolves() {
    // 24 order-1 cells (base cells 0-5) tile exactly half the sphere
    // (area = 2π) — the flagship input the PR #111 guard rejected.  The
    // winding-free classifier needs no winding sign, so it dissolves,
    // point-sampled correct against the source cover (issue #147).
    let cover = base_cells_cover(&[0, 1, 2, 3, 4, 5], 1);
    assert!((cover_area(&cover) - std::f64::consts::TAU).abs() < 1e-12);
    let got = dissolve(&cover, 1).unwrap();
    assert!(!got.shells.is_empty());
    assert_point_sampled(&cover, &got, 3);
}

#[test]
fn equatorial_band_dissolves() {
    // A thin equatorial band (~1.3 sr): both boundary rings enclose more
    // than a hemisphere, which wrapped the old fan sum mod 4π (rejected
    // by the PR #111 cross-check).  Planar classification does not care:
    // the two circles stitch into one seam-closed shell (issue #147).
    let mut band: Vec<u64> = Vec::new();
    for ilon in 0..720 {
        for ilat in -7..8 {
            let lat = ilat as f64 * 0.5;
            let lon = -180.0 + ilon as f64 * 0.5;
            band.push(crate::geo2mort::geo2mort_scalar(lat, lon, 4));
        }
    }
    band.sort_unstable();
    band.dedup();
    let got = dissolve(&band, 1).unwrap();
    assert_eq!(got.shells.len(), 1);
    assert!(got.holes.is_empty());
    assert_point_sampled(&band, &got, 4);
}

#[test]
fn sweep_exemplar_1_2_7_dissolves() {
    // Base cells {1, 2, 7} at order 1 (π sr) — the PR #179 review
    // sweep's second exemplar (12/12 stitcher panics post-#179, the
    // issue #181 measurement).  Its region wraps a pole only in the
    // net-winding accounting, not in fact; the local pole rule stitches
    // it correctly (issue #147 plan delta, acceptance fixture).
    let cover = base_cells_cover(&[1, 2, 7], 1);
    let got = dissolve(&cover, 1).unwrap();
    assert!(!got.shells.is_empty());
    assert_point_sampled(&cover, &got, 3);
}

#[test]
fn sweep_exemplar_4_6_7_10_dissolves() {
    // Base cells {4, 6, 7, 10} at order 1 (4.19 sr, over-hemisphere) —
    // the PR #179 review sweep's first exemplar (12/12 guard rejections
    // post-#179).  Issue #147 plan delta, acceptance fixture.
    let cover = base_cells_cover(&[4, 6, 7, 10], 1);
    let got = dissolve(&cover, 1).unwrap();
    assert!(!got.shells.is_empty());
    assert_point_sampled(&cover, &got, 3);
}

#[test]
fn over_hemisphere_cover_dissolves() {
    // Base cells 0-6 (7.33 sr, well past a hemisphere): previously the
    // area guard rejected anything past 0.98·2π; the planar classifier
    // emits the honest outline of the larger region (issue #147).
    let cover = base_cells_cover(&[0, 1, 2, 3, 4, 5, 6], 1);
    let got = dissolve(&cover, 1).unwrap();
    assert!(!got.shells.is_empty());
    assert_point_sampled(&cover, &got, 3);
}

#[test]
fn world_minus_hole_gets_frame_shell() {
    // Every order-2 cell but one: the covered region encloses the whole
    // seam and both poles, so its only spherical boundary ring is a
    // planar hole (covered-on-left reads CW around the uncovered cell)
    // and the shell is the explicit map frame (issue #147).
    let mut nested: Vec<u64> = (0..192).collect();
    nested.remove(100);
    let cover: Vec<u64> = nested
        .iter()
        .map(|&c| crate::morton::nested2mort(c, 2))
        .collect();
    let got = dissolve(&cover, 1).unwrap();
    assert_eq!(got.shells.len(), 1);
    assert_eq!(got.shells[0], frame_ring());
    assert_eq!(got.holes.len(), 1);
    assert_point_sampled(&cover, &got, 3);
}

#[test]
fn whole_sphere_cover_emits_frame() {
    // All 48 order-1 cells: every edge cancels, no boundary ring
    // survives — the planar image is the frame itself (issue #147).
    let bases: Vec<u64> = (0..12).collect();
    let cover = base_cells_cover(&bases, 1);
    assert!((cover_area(&cover) - 2.0 * std::f64::consts::TAU).abs() < 1e-12);
    let got = dissolve(&cover, 1).unwrap();
    assert_eq!(got.shells, vec![frame_ring()]);
    assert!(got.holes.is_empty());
    assert_point_sampled(&cover, &got, 3);
}

#[test]
fn north_pole_wrap_picks_nearest_start() {
    // Latent stitcher bug fixture (found during issue #147): a north
    // polar cap plus a disjoint seam-straddling box.  The old global
    // net-winding rule set pole=+90, and after the cap's pole seam the
    // `want_min` arm picked the *minimum*-latitude start on -180 (the
    // box's), fusing cap and box into one self-intersecting ring —
    // shapely reported the emit invalid.  The local rule resumes at the
    // start nearest the crossed pole (the cap's own): two clean parts.
    let mut cover: Vec<u64> = Vec::new();
    for ilat in 0..10 {
        let lat = 86.0 + 0.4 * ilat as f64;
        for ilon in -180..180 {
            cover.push(crate::geo2mort::geo2mort_scalar(lat, ilon as f64, 4));
        }
    }
    for ilat in 0..21 {
        let lat = 10.0 + 0.5 * ilat as f64;
        for ilon in 0..33 {
            let lon = 172.0 + 0.5 * ilon as f64;
            let lon = if lon > 180.0 { lon - 360.0 } else { lon };
            cover.push(crate::geo2mort::geo2mort_scalar(lat, lon, 4));
        }
    }
    cover.sort_unstable();
    cover.dedup();
    let got = dissolve(&cover, 1).unwrap();
    // cap (one seam-wrapped ring) + the box's east and west halves.
    assert_eq!(got.shells.len(), 3, "cap and box must stay separate");
    assert!(got.holes.is_empty());
    assert_point_sampled(&cover, &got, 4);
}

// ── issue #147 phase 4: the regenerated PR #179 review-sweep corpus ────────

#[test]
fn sweep_corpus_order1_combos_dissolve_point_sampled() {
    // The PR #179 review sweep's coarse family, regenerated and extended:
    // every non-empty base-cell mask at order 1 — all 4095, a superset of
    // the sweep's 1-to-5-cell family (1585) whose 6-to-12-cell masks are
    // the hemisphere+ regime this issue exists for.  Under the old
    // classifier 194 of the sweep masks deterministically failed the PR
    // #111 guards and 79 reached the stitcher assert (and every mask past
    // 5 cells was area-guard-rejected); the winding-free classifier must
    // dissolve every one, each emitted region point-sampled against the
    // source cover (issue #147 plan delta).  The first 60 masks also run
    // at step 3, mirroring the sweep's step slice.
    let mut count = 0u32;
    for mask in 1u16..(1 << 12) {
        let bases: Vec<u64> = (0..12).filter(|&b| mask >> b & 1 == 1).collect();
        let cover = base_cells_cover(&bases, 1);
        let got = dissolve(&cover, 1).unwrap_or_else(|e| panic!("mask {mask:#014b} failed: {e}"));
        assert_point_sampled(&cover, &got, 3);
        if count < 60 {
            let got3 = dissolve(&cover, 3)
                .unwrap_or_else(|e| panic!("mask {mask:#014b} step 3 failed: {e}"));
            assert_point_sampled(&cover, &got3, 3);
        }
        count += 1;
    }
    assert_eq!(count, 4095);
}

#[test]
fn sweep_corpus_fine_order_families_dissolve() {
    // The sweep's realistic families, regenerated: contiguous nested runs
    // at orders 2-5 (300 per order) plus 300 scattered order-3 sets,
    // deterministically seeded.  Every cover dissolves; its own cell
    // centres must all fall inside the emit, and a global order-1 lattice
    // point-samples membership (the chord-band exemption applies near the
    // boundary).
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut rng = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 33
    };
    let mut covers: Vec<(u8, Vec<u64>)> = Vec::new();
    for order in 2u8..=5 {
        let ncell = 12u64 << (2 * order);
        for _ in 0..300 {
            let len = 4 + rng() % 37;
            let start = rng() % (ncell - len);
            let cover: Vec<u64> = (start..start + len)
                .map(|c| crate::morton::nested2mort(c, order))
                .collect();
            covers.push((order, cover));
        }
    }
    for _ in 0..300 {
        let cover: Vec<u64> = (0..8)
            .map(|_| crate::morton::nested2mort(rng() % 768, 3))
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        covers.push((3, cover));
    }
    for (order, cover) in covers {
        let got = dissolve(&cover, 1).unwrap_or_else(|e| panic!("order {order} cover failed: {e}"));
        let covered: std::collections::HashSet<u64> = cover.iter().copied().collect();
        let band = 4.0 / f64::from(1u32 << order);
        // every cover cell's own centre is inside the emit.
        for &w in &cover {
            let (nest, _) = mort2nested(w);
            let (mut lon, lat) = crate::geo2mort::pix2ang_scalar(order, nest);
            if lon > 180.0 {
                lon -= 360.0;
            }
            if (lon.abs() - 180.0).abs() < 1e-9 {
                lon = 180.0 - 1e-4;
            }
            if !emitted_contains(&got, lon, lat) {
                assert!(
                    planar_distance_to_rings(&got, lon, lat) < band,
                    "own centre of {w} at lon {lon} lat {lat} outside emit"
                );
            }
        }
        // a global order-1 lattice must classify by exact membership.
        for pix in 0..48u64 {
            let (mut lon, lat) = crate::geo2mort::pix2ang_scalar(1, pix);
            if lon > 180.0 {
                lon -= 360.0;
            }
            let w = crate::geo2mort::geo2mort_scalar(lat, lon, order);
            if emitted_contains(&got, lon, lat) != covered.contains(&w) {
                assert!(
                    planar_distance_to_rings(&got, lon, lat) < band,
                    "order {order}: lattice pix {pix} misclassified"
                );
            }
        }
    }
}

/// A jagged polar cap of the `n` northernmost order-4 cells.
fn polar_cap_cells(n: usize) -> Vec<u64> {
    let mut by_lat: Vec<(f64, u64)> = (0..3072u64)
        .map(|nest| {
            let (_, lat) = crate::geo2mort::pix2ang_scalar(4, nest);
            (lat, nest)
        })
        .collect();
    by_lat.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
    by_lat[..n]
        .iter()
        .map(|&(_, nest)| crate::morton::nested2mort(nest, 4))
        .collect()
}

#[test]
fn polar_caps_straddling_hemisphere_dissolve() {
    // Polar caps just under / at / just over a hemisphere (issue #147):
    // 1475 / 1536 / 1597 of 3072 order-4 cells = 96% / 100% / 104% of 2π
    // sr.  The 96% cap dissolved correctly before PR #111's 2% margin
    // rejected it — regression-pinned as working again — and the exact- and
    // over-hemisphere caps flip from raising to correct outlines.
    for n in [1475usize, 1536, 1597] {
        let cover = polar_cap_cells(n);
        let got = dissolve(&cover, 1).unwrap_or_else(|e| panic!("cap of {n} cells failed: {e}"));
        assert!(!got.shells.is_empty(), "cap of {n} cells: no shells");
        assert_point_sampled(&cover, &got, 4);
    }
}

#[test]
fn mixed_order_hemisphere_moc_dissolves() {
    // A mixed-order hemisphere+ MOC (review fold, issue #147): base cells
    // 0-5 at order 1 plus base cell 6 as its sixteen order-2 children
    // (7.33 sr).  Exercises the path where the per-call orientation
    // calibration reads morton[0]'s own order while the boundary is built
    // at the densified finest order — emission handedness is
    // order-uniform, so the calibration must still hold.
    let mut cover = base_cells_cover(&[0, 1, 2, 3, 4, 5], 1);
    cover.extend((6 * 16..7 * 16).map(|c| crate::morton::nested2mort(c, 2)));
    let got = dissolve(&cover, 1).unwrap();
    assert!(!got.shells.is_empty());
    assert_point_sampled(&cover, &got, 3);
}

#[test]
fn both_polar_caps_multipart_dissolves() {
    // A multipart hemisphere-straddling cover: both polar caps (|centre
    // lat| > 60 at order 3).  The two boundary circles' net longitude
    // windings cancel — the case the old global pole selection could not
    // stitch (each cap must wrap its own pole).
    let cover: Vec<u64> = (0..768u64)
        .filter(|&nest| {
            let (_, lat) = crate::geo2mort::pix2ang_scalar(3, nest);
            lat.abs() > 60.0
        })
        .map(|nest| crate::morton::nested2mort(nest, 3))
        .collect();
    let got = dissolve(&cover, 1).unwrap();
    assert_eq!(got.shells.len(), 2, "two caps, two shells");
    assert!(got.holes.is_empty());
    assert_point_sampled(&cover, &got, 3);
}

#[test]
fn stitch_missing_partner_errs_curated() {
    // Issue #181: the stitching path's defensive failure states surface
    // as the curated Err convention (PR #111), never as a raw panic.
    // A lone segment starting and ending on +180 with no start above its
    // end wraps the north pole and finds no partner on -180.
    let segs = vec![vec![(180.0, 10.0), (0.0, 15.0), (180.0, 20.0)]];
    let err = stitch_segments(segs).expect_err("must err");
    assert!(err.contains("no partner segment"), "{err}");
}

#[test]
fn sub_hemisphere_cover_still_dissolves() {
    // Base cells 0-3 (2/3 of a hemisphere) stay outside the guard.
    let cover: Vec<u64> = (0..16u64)
        .map(|nest| crate::morton::nested2mort(nest, 1))
        .collect();
    let got = dissolve(&cover, 1).unwrap();
    assert!(!got.shells.is_empty());
}

#[test]
fn sub_hemisphere_cover_dissolves_deterministically() {
    // issue #155: dissolve on this fixed cover (base cells 0-3 at order
    // 1) flaked ~10-15% per call.  Phase-1 instrumentation finding: the
    // cover's boundary graph is a single simple 16-cycle -- there is no
    // branching vertex at all (neither a true corner-touch nor a
    // snap-merge artifact; no near-coincident vertex pairs, no duplicate
    // directed edges), so the ring *decomposition* is unique.  The
    // nondeterminism was purely the HashMap-seeded starting rotation of
    // that one ring: classify_and_split fans spherical_signed_area from
    // ring[0], and 2 of the 16 possible anchors (the equatorial spike
    // tips at lon +/-45, lat 0) wrap the fan to 0.790 sr vs the true
    // 4.090 sr, tripping the hemisphere-ring guard -- predicting 2/16 =
    // 12.5% (measured 48/400 pre-fix).  Each dissolve call builds fresh
    // HashMaps, so the flake reproduces in-process: pre-fix, 50
    // iterations fail with p ~= 1 - (7/8)^50 ~= 0.999.
    let cover: Vec<u64> = (0..16u64)
        .map(|nest| crate::morton::nested2mort(nest, 1))
        .collect();
    for i in 0..50 {
        let got = dissolve(&cover, 1).unwrap_or_else(|e| panic!("iteration {i} failed: {e}"));
        assert!(!got.shells.is_empty(), "iteration {i}: no shells");
    }
}

#[test]
fn dissolve_output_is_byte_identical_across_calls() {
    // issue #155 phase 3: dissolve is a pure function of its input --
    // two calls on identical input return byte-identical ClassifiedRings,
    // on each fixture shape (plain cover, step > 1 densified edges, the
    // pole/antimeridian stitch path, an equatorial cell).
    let north: Vec<u64> = (0..16u64)
        .map(|n| crate::morton::nested2mort(n, 1))
        .collect();
    let south: Vec<u64> = (32..48u64)
        .map(|n| crate::morton::nested2mort(n, 1))
        .collect();
    let eq: Vec<u64> = (16..20u64)
        .map(|n| crate::morton::nested2mort(n, 1))
        .collect();
    for (cover, step) in [(&north, 1u32), (&north, 4), (&south, 1), (&eq, 1), (&eq, 3)] {
        let a = dissolve(cover, step).unwrap();
        let b = dissolve(cover, step).unwrap();
        assert_eq!(a.shells, b.shells);
        assert_eq!(a.holes, b.holes);
        assert!(!a.shells.is_empty());
    }
}

fn lonlat_xyz(lon: f64, lat: f64) -> Vec3 {
    let (rlon, rlat) = (lon.to_radians(), lat.to_radians());
    [rlat.cos() * rlon.cos(), rlat.cos() * rlon.sin(), rlat.sin()]
}

// Two diamonds east and west of a shared corner-touch vertex T (id 0),
// both wound anticlockwise (interior on the same side).
fn corner_touch_fixture() -> (Vec<Vec3>, Vec<(u32, u32)>) {
    let id_xyz = vec![
        lonlat_xyz(0.0, 0.0),   // 0: T, the corner-touch vertex
        lonlat_xyz(5.0, 5.0),   // 1: east diamond, north corner
        lonlat_xyz(10.0, 0.0),  // 2: east diamond, east corner
        lonlat_xyz(5.0, -5.0),  // 3: east diamond, south corner
        lonlat_xyz(-5.0, 5.0),  // 4: west diamond, north corner
        lonlat_xyz(-10.0, 0.0), // 5: west diamond, west corner
        lonlat_xyz(-5.0, -5.0), // 6: west diamond, south corner
    ];
    let edges = vec![
        (0, 3),
        (3, 2),
        (2, 1),
        (1, 0), // east: T -> S -> E -> N -> T
        (0, 4),
        (4, 5),
        (5, 6),
        (6, 0), // west: T -> N -> W -> S -> T
    ];
    (id_xyz, edges)
}

// chain_rings on every LCG-shuffled permutation of `edges` (plus the
// reverse-sorted order) must equal its output on the canonical order.
fn assert_permutation_invariant(edges: &[(u32, u32)], id_xyz: &[Vec3]) {
    let baseline = chain_rings(edges, id_xyz);
    assert!(!baseline.is_empty());
    // deterministic Fisher-Yates via an LCG (no rand dependency).
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut rng = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    let mut shuffled = edges.to_vec();
    for _ in 0..200 {
        for i in (1..shuffled.len()).rev() {
            let j = rng() % (i + 1);
            shuffled.swap(i, j);
        }
        assert_eq!(chain_rings(&shuffled, id_xyz), baseline);
    }
    shuffled.sort_unstable();
    shuffled.reverse();
    assert_eq!(chain_rings(&shuffled, id_xyz), baseline);
}

#[test]
fn chain_rings_is_order_independent() {
    // issue #155 phase 3: chain_rings output is a pure function of the
    // edge *set* -- every permutation of the survivor slice yields the
    // identical ring list (this is the invariant, not one lucky order).
    // Three shapes: the real cover's survivors (a simple 16-cycle --
    // pins seed/rotation order), the corner-touch fixture (a branching
    // vertex -- pins the rotational pairing itself), and a net > 1
    // duplicate-edge set (pins the canonical-index tie-break at
    // identical azimuths).
    let cover: Vec<u64> = (0..16u64)
        .map(|n| crate::morton::nested2mort(n, 1))
        .collect();
    let (edges, id_xyz) = survivor_edges(&cover, 1);
    assert_permutation_invariant(&edges, &id_xyz);

    let (id_xyz, edges) = corner_touch_fixture();
    assert_permutation_invariant(&edges, &id_xyz);

    let mut dup = edges.clone();
    dup.extend_from_slice(&edges[..4]); // east diamond twice (net = 2)
    assert_permutation_invariant(&dup, &id_xyz);
    let rings = chain_rings(&dup, &id_xyz);
    assert_eq!(rings.len(), 3, "duplicated diamond must chain twice");
    assert_eq!(rings[0], rings[1]); // the two east copies are identical
}

#[test]
fn corner_touch_pairs_into_simple_rings() {
    // issue #155 ruling: at a non-manifold corner-touch vertex the
    // rotational pairing pinches the boundary into simple rings touching
    // at a point (GeoJSON-valid) -- never one self-intersecting
    // figure-eight.
    let (id_xyz, edges) = corner_touch_fixture();
    let rings = chain_rings(&edges, &id_xyz);
    assert_eq!(
        rings.len(),
        2,
        "pinch must yield two rings, not a figure-eight"
    );
    // exact pairing: each ring stays inside its own diamond, T once each.
    assert_eq!(rings[0], vec![id_xyz[0], id_xyz[3], id_xyz[2], id_xyz[1]]);
    assert_eq!(rings[1], vec![id_xyz[0], id_xyz[4], id_xyz[5], id_xyz[6]]);
}

// ── issue #147 phase 1: the stitcher orientation contract ──────────────

#[test]
fn cell_boundary_emission_orientation_is_uniform() {
    // Phase 1 of issue #147: the classifier's orientation contract rests
    // on every cell boundary being emitted with a fixed handedness.  The
    // HEALPix boundary functions put the cell interior on the LEFT at
    // step == 1 (CCW, positive spherical signed area) and on the RIGHT at
    // step > 1 (CW, negative) — uniformly over every base cell and order,
    // with the loop's fan area matching the exact cell area.
    for order in 0u8..=5 {
        let n: u64 = 1 << (2 * order);
        let exact = std::f64::consts::PI / (3.0 * 4f64.powi(order as i32));
        for step in [1u32, 2, 3, 4, 8] {
            for base in 0u64..12 {
                for nest in [base * n, base * n + n / 2, base * n + n - 1] {
                    let a = spherical_signed_area(&cell_boundary_loop(order, nest, step));
                    assert_eq!(
                        a > 0.0,
                        step == 1,
                        "order {order} step {step} nest {nest}: area {a}"
                    );
                    assert!(
                        (a.abs() - exact).abs() < 0.35 * exact,
                        "order {order} step {step} nest {nest}: |{a}| vs {exact}"
                    );
                }
            }
        }
    }
}

/// Covers used by the orientation audit: `(name, nested cells, order)`.
/// Includes exact-hemisphere and over-hemisphere covers (the audit drives
/// `boundary_rings_xyz` directly, below any guard) and the PR #179 review
/// sweep's two exemplar covers.
fn orientation_audit_covers() -> Vec<(&'static str, Vec<u64>, u8)> {
    let cells = |bases: &[u64], order: u8| -> Vec<u64> {
        let n = 1u64 << (2 * order);
        bases.iter().flat_map(|&b| b * n..(b + 1) * n).collect()
    };
    let mut world: Vec<u64> = (0..192).collect();
    world.remove(100);
    vec![
        ("base 0-3", cells(&[0, 1, 2, 3], 1), 1),
        ("hemisphere 0-5", cells(&[0, 1, 2, 3, 4, 5], 1), 1),
        ("over-hemisphere 0-6", cells(&[0, 1, 2, 3, 4, 5, 6], 1), 1),
        ("exemplar 4/6/7/10", cells(&[4, 6, 7, 10], 1), 1),
        ("exemplar 1/2/7", cells(&[1, 2, 7], 1), 1),
        ("scattered", vec![16, 24, 25, 28, 40, 43], 1),
        ("world minus one order-2 cell", world, 2),
    ]
}

#[test]
fn chained_rings_carry_cover_on_left() {
    // Phase 1 of issue #147 — the orientation contract, end to end: after
    // one per-call calibration (reverse every ring iff the emission
    // handedness is CW, read off one cell's own loop), every chained ring
    // carries the covered region on its LEFT.  The property is local
    // (each surviving directed edge bounds exactly one covered cell, on
    // the emission side; chaining preserves edge direction), so it holds
    // at any cover scale — verified here on exact-hemisphere,
    // over-hemisphere and world-minus-cell covers by displacing each edge
    // midpoint a quarter edge-length to the left (must land covered) and
    // right (must land uncovered).
    for (name, nested, order) in orientation_audit_covers() {
        for step in [1u32, 3] {
            let covered: std::collections::HashSet<u64> = nested
                .iter()
                .map(|&c| crate::morton::nested2mort(c, order))
                .collect();
            let cover: Vec<u64> = covered.iter().copied().collect();
            let mut rings = boundary_rings_xyz(&cover, step);
            let ccw = spherical_signed_area(&cell_boundary_loop(order, nested[0], step)) > 0.0;
            if !ccw {
                for r in rings.iter_mut() {
                    r.reverse();
                }
            }
            assert!(!rings.is_empty(), "{name}: no rings");
            for ring in &rings {
                let n = ring.len();
                for i in 0..n {
                    let (a, b) = (ring[i], ring[(i + 1) % n]);
                    let t = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                    let elen = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
                    let m = crate::sphere::normalize(&[a[0] + b[0], a[1] + b[1], a[2] + b[2]]);
                    let left = crate::sphere::normalize(&cross(&m, &t));
                    for (s, want) in [(1.0, true), (-1.0, false)] {
                        let p = crate::sphere::normalize(&[
                            m[0] + s * 0.25 * elen * left[0],
                            m[1] + s * 0.25 * elen * left[1],
                            m[2] + s * 0.25 * elen * left[2],
                        ]);
                        let (lon, lat) = xyz_to_lonlat(&p);
                        let w = crate::geo2mort::geo2mort_scalar(lat, lon, order);
                        assert_eq!(
                            covered.contains(&w),
                            want,
                            "{name} step {step}: probe side {s} at lon {lon} lat {lat}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn pole_cap_stitches_to_one_ring_with_pole_vertex() {
    // one segment running +180 -> -180 around the pole; the local rule
    // (end on -180 with no start below) wraps -90 with no pole parameter.
    let segs = vec![vec![
        (180.0, -80.0),
        (90.0, -80.0),
        (0.0, -80.0),
        (-90.0, -80.0),
        (-180.0, -80.0),
    ]];
    let rings = stitch_segments(segs).unwrap();
    assert_eq!(rings.len(), 1);
    assert!(rings[0].iter().any(|p| (p.1 + 90.0).abs() < 1e-9)); // pole vertex
}

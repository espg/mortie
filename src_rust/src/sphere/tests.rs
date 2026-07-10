//! Unit tests for the spherical-geometry primitives (issue #78).
//!
//! Split out of `sphere.rs` to keep that module under the project's
//! ~1000-line soft limit; wired back in via `#[cfg(test)] mod tests;`.

use super::*;

fn ring(pts: &[(f64, f64)]) -> Vec<Vec3> {
    pts.iter()
        .map(|&(la, lo)| latlon_to_unit_vec(la, lo))
        .collect()
}

#[test]
fn test_orient_basis() {
    let x = [1.0, 0.0, 0.0];
    let y = [0.0, 1.0, 0.0];
    let z = [0.0, 0.0, 1.0];
    assert!(orient(&x, &y, &z) > 0.0, "right-handed basis is positive");
    assert!(orient(&x, &z, &y) < 0.0, "swapped is negative");
    // Coplanar vectors give zero.
    assert!(orient(&x, &y, &x).abs() < 1e-15);
}

#[test]
fn test_arcs_cross_basic() {
    // Two arcs straddling (0,0): a N-S arc and an E-W arc.
    let ns = (ring(&[(-10.0, 0.0)])[0], ring(&[(10.0, 0.0)])[0]);
    let ew = (ring(&[(0.0, -10.0)])[0], ring(&[(0.0, 10.0)])[0]);
    assert!(
        arcs_cross(&ns.0, &ns.1, &ew.0, &ew.1),
        "should cross at origin"
    );
    // Parallel-ish, offset arcs that do not cross.
    let a = ring(&[(20.0, -10.0), (20.0, 10.0)]);
    let b = ring(&[(40.0, -10.0), (40.0, 10.0)]);
    assert!(!arcs_cross(&a[0], &a[1], &b[0], &b[1]), "disjoint arcs");
}

#[test]
fn test_arcs_cross_n_matches_arcs_cross() {
    // arcs_cross_n must equal arcs_cross for every non-degenerate quadruple.
    // The two formulations of the side test (a·(b×c) vs (a×b)·c) are exactly
    // equal in real arithmetic but can disagree in float *sign* only when the
    // triple product is a tie (≈0) — the touching/coplanar case arcs_cross
    // already documents as sign-arbitrary.  Skip those: a shared endpoint or
    // a near-zero orientation.
    let pts = ring(&[
        (-10.0, 0.0),
        (10.0, 0.0),
        (0.0, -10.0),
        (0.0, 10.0),
        (40.0, -125.0),
        (50.0, -115.0),
        (-72.0, 30.0),
        (12.0, 88.0),
    ]);
    for a in 0..pts.len() {
        for b in 0..pts.len() {
            for c in 0..pts.len() {
                for d in 0..pts.len() {
                    if [b, c, d].contains(&a) || b == c || b == d || c == d {
                        continue;
                    }
                    let (pa, pb, pc, pd) = (pts[a], pts[b], pts[c], pts[d]);
                    // Skip ties where the sign of an exact-zero triple product
                    // is meaningless (and differs between formulations).
                    let orients = [
                        orient(&pa, &pb, &pc),
                        orient(&pa, &pb, &pd),
                        orient(&pc, &pd, &pa),
                        orient(&pc, &pd, &pb),
                    ];
                    if orients.iter().any(|o| o.abs() < 1e-9) {
                        continue;
                    }
                    let n_ab = cross(&pa, &pb);
                    let n_cd = cross(&pc, &pd);
                    assert_eq!(
                        arcs_cross(&pa, &pb, &pc, &pd),
                        arcs_cross_n(&pa, &pb, &n_ab, &pc, &pd, &n_cd),
                        "mismatch at ({a},{b},{c},{d})"
                    );
                }
            }
        }
    }
}

#[test]
fn test_arcs_cross_endpoint_disjoint() {
    // Arcs sharing no span: one near equator, one far away.
    let a = ring(&[(0.0, 0.0), (0.0, 5.0)]);
    let b = ring(&[(50.0, 50.0), (55.0, 55.0)]);
    assert!(!arcs_cross(&a[0], &a[1], &b[0], &b[1]));
}

// ── robust backend (issue #22 / #11) ─────────────────────────────────

#[test]
fn test_orient_sos_matches_orient_when_nonzero() {
    // Where the geometric determinant is non-zero, SoS must echo its sign.
    let x = [1.0, 0.0, 0.0];
    let y = [0.0, 1.0, 0.0];
    let z = [0.0, 0.0, 1.0];
    assert_eq!(orient_sos(&x, &y, &z, 0, 1, 2), 1);
    assert_eq!(orient_sos(&x, &z, &y, 0, 1, 2), -1);
}

#[test]
fn test_orient_sos_breaks_coplanar_consistently() {
    // Three points on a common, NON-axis-aligned great circle (an earlier
    // version of this test used axis-aligned equatorial points, where every
    // argument permutation happens to round to exact 0.0 — it passed by
    // coordinate luck even with the antisymmetry bug).  Build a tilted great
    // circle and place three distinct points on it so the triple is coplanar
    // with the origin but the f64 determinant is sensitive to arg order.
    let n = normalize(&[0.37, -0.81, 0.45]); // great-circle normal
    let u = normalize(&cross(&n, &[1.0, 0.0, 0.0]));
    let v = cross(&n, &u); // {u, v, n} orthonormal; u, v span the circle
    let on = |ang: f64| -> Vec3 {
        normalize(&[
            u[0] * ang.cos() + v[0] * ang.sin(),
            u[1] * ang.cos() + v[1] * ang.sin(),
            u[2] * ang.cos() + v[2] * ang.sin(),
        ])
    };
    let a = on(0.3);
    let b = on(1.9);
    let c = on(4.4);
    assert!(orient(&a, &b, &c).abs() < 1e-15, "must be exactly coplanar");
    let s = orient_sos(&a, &b, &c, 10, 20, 30);
    assert_ne!(s, 0, "SoS must never return zero");
    // Antisymmetry: swapping two arguments (points and ids together) flips
    // the sign — the property that keeps crossing parity consistent.
    assert_eq!(orient_sos(&b, &a, &c, 20, 10, 30), -s);
    assert_eq!(orient_sos(&a, &c, &b, 10, 30, 20), -s);
    assert_eq!(orient_sos(&c, &b, &a, 30, 20, 10), -s);
    // Cyclic invariance: even permutations preserve the sign.
    assert_eq!(orient_sos(&b, &c, &a, 20, 30, 10), s);
    assert_eq!(orient_sos(&c, &a, &b, 30, 10, 20), s);
    // Re-evaluation is deterministic.
    assert_eq!(orient_sos(&a, &b, &c, 10, 20, 30), s);
}

#[test]
fn test_orient_sos_antisymmetric_brute_force() {
    // Brute-force the permutation-robustness contract over thousands of
    // triples — both exactly-coplanar (points sharing a great circle, the
    // reproduction in the bug report) and random general-position points.
    // For every triple, all 6 argument permutations must agree up to their
    // permutation parity, and SoS must never return 0.  This FAILS on the
    // pre-fix code (~28% of coplanar triples violated antisymmetry because
    // the exact-0.0 gate was applied to the as-given order, not a canonical
    // one) and passes after the canonical-order fix.
    let mut rng: u64 = 0x9e3779b97f4a7c15;
    let mut next = || {
        // splitmix64 — deterministic, no rand dependency.
        rng = rng.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z = z ^ (z >> 31);
        // uniform-ish f64 in [-1, 1]
        (z as f64 / u64::MAX as f64) * 2.0 - 1.0
    };

    let check = |a: &Vec3, b: &Vec3, c: &Vec3| {
        // ids must be distinct and consistently ordered
        let (ia, ib, ic) = (1u64, 2u64, 3u64);
        let s = orient_sos(a, b, c, ia, ib, ic);
        assert_ne!(s, 0, "SoS must always break ties (never 0)");
        // odd permutations flip
        assert_eq!(orient_sos(b, a, c, ib, ia, ic), -s, "swap(0,1)");
        assert_eq!(orient_sos(a, c, b, ia, ic, ib), -s, "swap(1,2)");
        assert_eq!(orient_sos(c, b, a, ic, ib, ia), -s, "swap(0,2)");
        // even (cyclic) permutations preserve
        assert_eq!(orient_sos(b, c, a, ib, ic, ia), s, "cycle b,c,a");
        assert_eq!(orient_sos(c, a, b, ic, ia, ib), s, "cycle c,a,b");
    };

    let mut coplanar = 0u32;
    let mut general = 0u32;
    for _ in 0..5000 {
        // --- general-position triple ---
        let a = normalize(&[next(), next(), next()]);
        let b = normalize(&[next(), next(), next()]);
        let c = normalize(&[next(), next(), next()]);
        check(&a, &b, &c);
        general += 1;

        // --- exactly-coplanar triple: three points on one great circle ---
        let nrm = normalize(&[next(), next(), next()]);
        // a stable in-plane basis from the normal
        let seed = if nrm[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let u = normalize(&cross(&nrm, &seed));
        let v = cross(&nrm, &u);
        let on = |ang: f64| -> Vec3 {
            normalize(&[
                u[0] * ang.cos() + v[0] * ang.sin(),
                u[1] * ang.cos() + v[1] * ang.sin(),
                u[2] * ang.cos() + v[2] * ang.sin(),
            ])
        };
        let pa = on(next() * std::f64::consts::PI);
        let pb = on(next() * std::f64::consts::PI);
        let pc = on(next() * std::f64::consts::PI);
        // confirm coplanarity (within fp noise) before asserting robustness
        assert!(
            orient(&pa, &pb, &pc).abs() < 1e-12,
            "constructed triple must be coplanar"
        );
        check(&pa, &pb, &pc);
        coplanar += 1;
    }
    assert_eq!(general, 5000);
    assert_eq!(coplanar, 5000);
}

#[test]
fn test_robust_crossing_basic() {
    // N-S arc and E-W arc straddling the origin cross; offset arcs don't.
    let ns = (
        latlon_to_unit_vec(-10.0, 0.0),
        latlon_to_unit_vec(10.0, 0.0),
    );
    let ew = (
        latlon_to_unit_vec(0.0, -10.0),
        latlon_to_unit_vec(0.0, 10.0),
    );
    assert!(arcs_cross_sos(&ns.0, &ns.1, &ew.0, &ew.1, 0, 1, 2, 3));
    let a = ring(&[(20.0, -10.0), (20.0, 10.0)]);
    let b = ring(&[(40.0, -10.0), (40.0, 10.0)]);
    assert!(!arcs_cross_sos(&a[0], &a[1], &b[0], &b[1], 0, 1, 2, 3));
}

#[test]
fn test_robust_crossing_rejects_far_intersection() {
    // The crux of major-arc correctness: a long arc from a generic point to
    // the south pole straddles the great circle of a north-cap edge, but the
    // *segments* do not meet (the meeting point is on the antipodal halves).
    // A bare 4-orientation test counts this as a crossing; robust_crossing
    // must not. (Mirrors the #22 hemisphere+ failure traced in this PR.)
    let r = latlon_to_unit_vec(12.3, 75.7);
    let p = latlon_to_unit_vec(-89.0, 0.0);
    let c = latlon_to_unit_vec(80.0, 180.0);
    let d = latlon_to_unit_vec(80.0, 200.0);
    assert!(
        !arcs_cross_sos(&r, &p, &c, &d, 0, 1, 2, 3),
        "long arc must not falsely cross a far north-cap edge"
    );
}

/// Build the orthonormal frame `{u, v, n}` of a tilted (non-axis-aligned)
/// great circle and return a closure placing unit points on it by angle.
/// Mirrors `test_orient_sos_breaks_coplanar_consistently` so on-circle probes
/// are exactly coplanar with the origin (the `on_minor_arc` degeneracy).
fn great_circle(normal: [f64; 3]) -> impl Fn(f64) -> Vec3 {
    let n = normalize(&normal);
    let seed = if n[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let u = normalize(&cross(&n, &seed));
    let v = cross(&n, &u);
    move |ang: f64| -> Vec3 {
        normalize(&[
            u[0] * ang.cos() + v[0] * ang.sin(),
            u[1] * ang.cos() + v[1] * ang.sin(),
            u[2] * ang.cos() + v[2] * ang.sin(),
        ])
    }
}

/// `robust_crossing` over every reordering of the same geometry — both
/// endpoints reversed within each arc **and** the two arc roles swapped
/// (AB↔CD, which a crossing predicate must be symmetric under) — with point
/// ids carried in lockstep.  Returns the set of distinct boolean results
/// (size 1 ⇒ traversal-order-independent, the #78 invariant).
#[allow(clippy::too_many_arguments)]
fn crossing_under_reorder(
    a: &Vec3,
    b: &Vec3,
    c: &Vec3,
    d: &Vec3,
    ia: PointId,
    ib: PointId,
    ic: PointId,
    id: PointId,
) -> std::collections::BTreeSet<bool> {
    let mut out = std::collections::BTreeSet::new();
    // (e1, e2) ranges over both arc-role assignments; each arc's endpoints
    // are then independently reversed.
    for &(e1, e2) in &[
        ((a, b, ia, ib), (c, d, ic, id)),
        ((c, d, ic, id), (a, b, ia, ib)),
    ] {
        let (p0, p1, i0, i1) = e1;
        let (q0, q1, j0, j1) = e2;
        for &(pa, pb, pia, pib) in &[(p0, p1, i0, i1), (p1, p0, i1, i0)] {
            for &(pc, pd, pic, pid) in &[(q0, q1, j0, j1), (q1, q0, j1, j0)] {
                out.insert(arcs_cross_sos(pa, pb, pc, pd, pia, pib, pic, pid));
            }
        }
    }
    out
}

#[test]
fn test_robust_crossing_probe_on_edge_great_circle() {
    // #78: the antipodal-disambiguation `on_minor_arc` step still used raw f64
    // signs, so a probe whose intersection lands *exactly* on the edge's great
    // circle (both half-plane dot products round near zero) could flip the
    // crossing result under argument/edge reordering — a silent wrong PIP with
    // no error raised.  Place an edge AB on a tilted great circle and a probe
    // arc CD that crosses through a point exactly on AB's great circle; the
    // result must be the same for every traversal order.
    let on = great_circle([0.37, -0.81, 0.45]);
    let nrm = normalize(&[0.37, -0.81, 0.45]);
    let a = on(0.2);
    let b = on(1.9);
    // Probe whose crossing point sits exactly on AB's great circle (mid-arc).
    let xm = on(1.0);
    assert!(
        orient(&a, &b, &xm).abs() < 1e-15,
        "probe point must be exactly on edge AB's great circle"
    );
    let c = normalize(&[
        xm[0] + 0.3 * nrm[0],
        xm[1] + 0.3 * nrm[1],
        xm[2] + 0.3 * nrm[2],
    ]);
    let d = normalize(&[
        xm[0] - 0.3 * nrm[0],
        xm[1] - 0.3 * nrm[1],
        xm[2] - 0.3 * nrm[2],
    ]);
    let res = crossing_under_reorder(&a, &b, &c, &d, 10, 20, 30, 40);
    assert_eq!(
        res.len(),
        1,
        "on-great-circle crossing must be order-independent, got {res:?}"
    );
    // Pin the value: this probe genuinely crosses the interior of the arc.
    assert!(
        *res.iter().next().unwrap(),
        "mid-arc on-circle probe must register as a crossing"
    );
}

#[test]
fn test_robust_crossing_probe_through_edge_endpoint() {
    // The sharpest `on_minor_arc` degeneracy: the probe's intersection lands
    // exactly on an edge *endpoint*, where a half-plane determinant is exactly
    // zero (`cross(a, x)` with x ≡ ±a) and the raw-f64 sign is pure noise.  The
    // robust result must still be a single, definite value across reorderings.
    let on = great_circle([0.37, -0.81, 0.45]);
    let nrm = normalize(&[0.37, -0.81, 0.45]);
    let a = on(0.0);
    let b = on(1.6);
    // CD passes through endpoint a, so the great-circle intersection is ±a.
    let c = normalize(&[
        a[0] + 0.4 * nrm[0],
        a[1] + 0.4 * nrm[1],
        a[2] + 0.4 * nrm[2],
    ]);
    let d = normalize(&[
        a[0] - 0.4 * nrm[0],
        a[1] - 0.4 * nrm[1],
        a[2] - 0.4 * nrm[2],
    ]);
    let res = crossing_under_reorder(&a, &b, &c, &d, 10, 20, 30, 40);
    assert_eq!(
        res.len(),
        1,
        "endpoint-coincident crossing must be order-independent, got {res:?}"
    );
    // Pin the previously-flippable value: with the probe's intersection landing
    // exactly on endpoint a and CD reaching across it on the antipodal side, the
    // SoS-hardened half-open `[a, b)` rule resolves this to a definite "no
    // crossing" in every traversal order (rather than a noise-driven coin flip).
    assert!(
        !*res.iter().next().unwrap(),
        "endpoint-coincident probe must resolve to a definite no-crossing"
    );
}

#[test]
fn test_robust_pip_midlatitude() {
    // Mid-latitude square: clearly inside / outside points.
    let sq = ring(&[
        (40.0, -125.0),
        (40.0, -115.0),
        (50.0, -115.0),
        (50.0, -125.0),
    ]);
    assert!(point_in_ring_robust(&latlon_to_unit_vec(45.0, -120.0), &sq));
    assert!(!point_in_ring_robust(&latlon_to_unit_vec(0.0, 0.0), &sq));
    assert!(!point_in_ring_robust(&latlon_to_unit_vec(45.0, -90.0), &sq));
}

#[test]
fn test_ring_winding_sign() {
    // A sub-hemisphere mid-latitude square wound CCW (interior on the left)
    // reads +1; the same vertices reversed (CW) read -1.
    let ccw = ring(&[
        (40.0, -125.0),
        (40.0, -115.0),
        (50.0, -115.0),
        (50.0, -125.0),
    ]);
    assert_eq!(ring_winding_sign(&ccw), 1, "CCW square");
    let mut cw = ccw.clone();
    cw.reverse();
    assert_eq!(ring_winding_sign(&cw), -1, "reversed (CW) square");
    // A southern triangle, both orientations.
    let tri_ccw = ring(&[(-80.0, 30.0), (-70.0, 60.0), (-70.0, 0.0)]);
    assert_eq!(ring_winding_sign(&tri_ccw), 1);
    let mut tri_cw = tri_ccw.clone();
    tri_cw.reverse();
    assert_eq!(ring_winding_sign(&tri_cw), -1);
}

#[test]
fn test_robust_parity_matches_oracle_within_hemisphere() {
    // Correctness proof on a sub-hemisphere CCW square: the single robust
    // backend must agree with the independent trig winding oracle at every
    // probe point (the gnomonic backend it used to be checked against was
    // removed at the Phase-3 cutover, #22).
    let sq = ring(&[
        (40.0, -125.0),
        (40.0, -115.0),
        (50.0, -115.0),
        (50.0, -125.0),
    ]);
    let rings = vec![sq.clone()];
    for lat in [38.0, 41.0, 45.0, 49.0, 52.0] {
        for lon in [-128.0, -123.0, -120.0, -117.0, -112.0] {
            let p = latlon_to_unit_vec(lat, lon);
            let oracle = winding_inside(&p, &sq);
            let r = parity_filled_robust(&p, &rings);
            assert_eq!(
                r, oracle,
                "robust vs winding-oracle disagree at ({lat},{lon})"
            );
        }
    }
}

#[test]
fn test_robust_parity_southern_polygon() {
    // Near-polar southern triangle (mirrors test_edgecross_southern_polygon).
    // CCW order (reversed from the gnomonic test's ring) so the robust
    // backend's CCW-interior convention selects the triangle as inside.
    let tri = ring(&[(-80.0, 30.0), (-70.0, 60.0), (-70.0, 0.0)]);
    let rings = vec![tri];
    assert!(parity_filled_robust(
        &latlon_to_unit_vec(-75.0, 30.0),
        &rings
    ));
    assert!(!parity_filled_robust(
        &latlon_to_unit_vec(-60.0, 30.0),
        &rings
    ));
}

#[test]
fn test_robust_pip_donut_and_multipart() {
    // Holes (even-odd) and multipart geometry under the robust backend.
    let outer = ring(&[
        (35.0, -130.0),
        (35.0, -110.0),
        (55.0, -110.0),
        (55.0, -130.0),
    ]);
    let hole = ring(&[
        (42.0, -123.0),
        (42.0, -117.0),
        (48.0, -117.0),
        (48.0, -123.0),
    ]);
    let donut = vec![outer, hole];
    assert!(
        parity_filled_robust(&latlon_to_unit_vec(38.0, -120.0), &donut),
        "annulus"
    );
    assert!(
        !parity_filled_robust(&latlon_to_unit_vec(45.0, -120.0), &donut),
        "hole empty"
    );
    assert!(
        !parity_filled_robust(&latlon_to_unit_vec(10.0, -120.0), &donut),
        "outside"
    );

    let part_a = ring(&[
        (40.0, -125.0),
        (40.0, -120.0),
        (45.0, -120.0),
        (45.0, -125.0),
    ]);
    let part_b = ring(&[
        (40.0, -110.0),
        (40.0, -105.0),
        (45.0, -105.0),
        (45.0, -110.0),
    ]);
    let parts = vec![part_a, part_b];
    assert!(parity_filled_robust(
        &latlon_to_unit_vec(42.0, -122.0),
        &parts
    ));
    assert!(parity_filled_robust(
        &latlon_to_unit_vec(42.0, -107.0),
        &parts
    ));
    assert!(!parity_filled_robust(
        &latlon_to_unit_vec(42.0, -115.0),
        &parts
    ));
}

#[test]
fn test_robust_pip_antimeridian() {
    // Box straddling the ±180° antimeridian (lon 170 → -170 the short way).
    let box_am = ring(&[(40.0, 170.0), (40.0, -170.0), (50.0, -170.0), (50.0, 170.0)]);
    assert!(
        point_in_ring_robust(&latlon_to_unit_vec(45.0, 180.0), &box_am),
        "on antimeridian"
    );
    assert!(point_in_ring_robust(
        &latlon_to_unit_vec(45.0, 175.0),
        &box_am
    ));
    assert!(point_in_ring_robust(
        &latlon_to_unit_vec(45.0, -175.0),
        &box_am
    ));
    assert!(
        !point_in_ring_robust(&latlon_to_unit_vec(45.0, 0.0), &box_am),
        "far side out"
    );
}

/// Independent oracle: signed spherical winding of `ring` seen from `x`
/// (sum of the signed angles each directed edge subtends).  `> π` ⇒ `x` is
/// in the counter-clockwise interior.  This is the trig-based,
/// reference-free ground truth the orientation-only backend must reproduce
/// at any size; it is deliberately a *different* computation from the code
/// under test (no fixed reference, no SoS) so agreement is meaningful.
fn winding_inside(x: &Vec3, ring: &[Vec3]) -> bool {
    let n = ring.len();
    let mut total = 0.0;
    for i in 0..n {
        let a = &ring[i];
        let b = &ring[(i + 1) % n];
        let da = dot(a, x);
        let db = dot(b, x);
        let pa = normalize(&[a[0] - da * x[0], a[1] - da * x[1], a[2] - da * x[2]]);
        let pb = normalize(&[b[0] - db * x[0], b[1] - db * x[1], b[2] - db * x[2]]);
        let ang = dot(&pa, &pb).clamp(-1.0, 1.0).acos();
        total += if dot(&cross(&pa, &pb), x) >= 0.0 {
            ang
        } else {
            -ang
        };
    }
    // CCW interior (matches point_in_ring_robust): winding ≈ +2π inside.
    total > std::f64::consts::PI
}

#[test]
fn test_robust_pip_polar_cap() {
    // A ring near the north pole (lat 80°): the robust backend must agree
    // with the winding oracle everywhere, including the pole and the far
    // antipodal point where a pole-centred gnomonic projection would clamp.
    let cap: Vec<Vec3> = (0..24)
        .map(|k| latlon_to_unit_vec(80.0, k as f64 * 15.0))
        .collect();
    for lat in [89.0, 85.0, 80.0, 45.0, 0.0, -45.0, -89.0] {
        for lon in [0.0, 37.0, 123.0, 200.0, 315.0] {
            let p = latlon_to_unit_vec(lat, lon);
            assert_eq!(
                point_in_ring_robust(&p, &cap),
                winding_inside(&p, &cap),
                "robust vs winding-oracle disagree at ({lat},{lon})"
            );
        }
    }
}

#[test]
fn test_robust_pip_hemisphere_plus_band() {
    // The hemisphere+ case (#22): a band ring at lat −10° whose CCW interior
    // is a region larger than a hemisphere.  We (1) prove it really is
    // >hemisphere by exhibiting two interior points more than 90° apart —
    // impossible for any sub-hemisphere cap, and the regime where gnomonic /
    // the cap-axis edge-cross both fail — and (2) check the robust backend
    // matches the winding oracle across the whole sphere.
    let band: Vec<Vec3> = (0..36)
        .map(|k| latlon_to_unit_vec(-10.0, k as f64 * 10.0))
        .collect();

    // (1) >hemisphere witness: north pole and a far mid-latitude point, both
    // interior, separated by well over 90°.
    let far_a = latlon_to_unit_vec(13.0, 15.0);
    let far_b = latlon_to_unit_vec(13.0, 195.0);
    assert!(
        winding_inside(&far_a, &band) && winding_inside(&far_b, &band),
        "witnesses interior"
    );
    let sep = dot(&far_a, &far_b).clamp(-1.0, 1.0).acos().to_degrees();
    assert!(
        sep > 90.0,
        "interior spans >hemisphere (witness sep {sep}°)"
    );
    assert!(point_in_ring_robust(&far_a, &band) && point_in_ring_robust(&far_b, &band));

    // (2) full-sphere parity against the oracle.
    for lat in [89.0, 60.0, 30.0, 0.0, -9.0, -11.0, -30.0, -89.0] {
        for lon in [0.0, 55.0, 123.0, 180.0, 250.0, 305.0] {
            let p = latlon_to_unit_vec(lat, lon);
            assert_eq!(
                point_in_ring_robust(&p, &band),
                winding_inside(&p, &band),
                "robust vs winding-oracle disagree at ({lat},{lon})"
            );
        }
    }
}

#[test]
#[ignore = "known defect found in #107 phase 1: the short-way subtended-angle \
sum is antisymmetric under x → −x (a and b project to the same vectors on the \
plane ⊥x for both poles; only the sign term flips), so it computes \
k(x) − k(−x) instead of the winding k(x) — every interior point whose \
antipode is also interior reads outside.  Sub-hemisphere interiors cannot \
contain an antipodal pair, so only the hemisphere+ regime (#22) is affected. \
Un-ignore when the winding backend is repaired (decision on issue #107)"]
fn test_point_in_ring_hemisphere_plus_antipodal_interior() {
    // #22's flagship shape: the lat −10 band, CCW interior = everything north
    // of −10 (hemisphere+).  Every point with lat ∈ (−10, +10) has its
    // antipode interior as well; the winding sum cancels to ~0 there and the
    // backend calls a truly interior point outside.  (The existing band test
    // above only probes latitudes ≥ 13 or ≤ −11, and its cross-check oracle
    // is another short-way winding sum — wrong in exactly the same region —
    // which is how this slipped through.)
    let band: Vec<Vec3> = (0..36)
        .map(|k| latlon_to_unit_vec(-10.0, k as f64 * 10.0))
        .collect();
    for lon in [0.0, 33.0, 120.0, 260.0] {
        for lat in [-5.0, 0.0, 5.0] {
            assert!(
                point_in_ring_robust(&latlon_to_unit_vec(lat, lon), &band),
                "({lat},{lon}) is interior (north of the −10 band) but reads \
                 outside: its antipode is interior too, so the antisymmetric \
                 winding sum cancels"
            );
        }
    }
}

#[test]
fn test_robust_pip_issue_11_meridian_box() {
    // #11 regression: the over-coverage flood was triggered by a polygon edge
    // lying exactly on a base-cell-centre meridian (lon ∈ {45,90,135,…}),
    // where the orientation determinant hits exact zero at HEALPix cell
    // centres. The robust PIP's SoS tie-break must classify such a box
    // cleanly. Base cell 0's centre is at (lat 41.81°, lon 45°); the lon-45
    // edge's great circle passes through it.
    // Box from the #11 reproducer: lat[40,42], left edge exactly on lon 45.
    let box45 = ring(&[(40.0, 47.0), (42.0, 47.0), (42.0, 45.0), (40.0, 45.0)]);
    // The base-cell centre sits on the edge meridian but north of the box's
    // top edge → must be classified OUTSIDE (it was flooded IN before).
    let base_center = latlon_to_unit_vec(41.81, 45.0);
    assert!(
        !point_in_ring_robust(&base_center, &box45),
        "base-cell centre on lon-45 meridian must be outside the box (#11)"
    );
    // A point genuinely inside the box.
    assert!(point_in_ring_robust(
        &latlon_to_unit_vec(41.0, 46.0),
        &box45
    ));
    // Points on the far (west) side of the lon-45 meridian must be outside —
    // this half is exactly what flooded in the bug.
    assert!(!point_in_ring_robust(
        &latlon_to_unit_vec(41.0, 44.0),
        &box45
    ));
    assert!(!point_in_ring_robust(
        &latlon_to_unit_vec(41.0, 30.0),
        &box45
    ));
    // Same check for the lon-90 meridian family.
    let box90 = ring(&[(40.0, 92.0), (42.0, 92.0), (42.0, 90.0), (40.0, 90.0)]);
    assert!(point_in_ring_robust(
        &latlon_to_unit_vec(41.0, 91.0),
        &box90
    ));
    assert!(!point_in_ring_robust(
        &latlon_to_unit_vec(41.0, 89.0),
        &box90
    ));
}

#[test]
fn test_robust_pip_consistency_under_vertex_rotation() {
    // Rotating the start vertex of a ring must not change any classification
    // — a direct check that SoS keeps the crossing parity traversal-order
    // independent (the property the #11 fix relies on).
    let base = ring(&[(40.0, 47.0), (42.0, 47.0), (42.0, 45.0), (40.0, 45.0)]);
    let rotated = ring(&[(42.0, 47.0), (42.0, 45.0), (40.0, 45.0), (40.0, 47.0)]);
    for lat in [39.0, 41.0, 41.81, 43.0] {
        for lon in [30.0, 44.0, 45.0, 46.0, 48.0] {
            let p = latlon_to_unit_vec(lat, lon);
            assert_eq!(
                point_in_ring_robust(&p, &base),
                point_in_ring_robust(&p, &rotated),
                "rotation changed classification at ({lat},{lon})"
            );
        }
    }
}

#[test]
fn test_robust_crossing_meridian_edge_on_circle() {
    // #78 on the #11 meridian family: a polygon edge lying exactly on a
    // base-cell-centre meridian (lon ∈ {45, 90}) is the canonical degeneracy.
    // Probe arcs whose intersection lands exactly on that meridian's great
    // circle must give a single, traversal-order-independent crossing result —
    // the `on_minor_arc` step is now SoS-hardened, not raw-f64.
    for &lon in &[45.0_f64, 90.0] {
        // Edge AB straddling the equator on the meridian's great circle.
        let a = latlon_to_unit_vec(-15.0, lon);
        let b = latlon_to_unit_vec(20.0, lon);
        // Probe CD crossing the edge; its intersection lies on the lon meridian
        // great circle (both endpoints at the same latitude, symmetric in lon).
        let c = latlon_to_unit_vec(2.0, lon - 8.0);
        let d = latlon_to_unit_vec(2.0, lon + 8.0);
        let res = crossing_under_reorder(&a, &b, &c, &d, 10, 20, 30, 40);
        assert_eq!(
            res.len(),
            1,
            "meridian-edge crossing must be order-independent at lon {lon}, got {res:?}"
        );
        assert!(
            *res.iter().next().unwrap(),
            "probe crossing the lon-{lon} edge must register"
        );
    }
}

#[test]
fn test_robust_crossing_tilted_circle_brute_force_invariance() {
    // Brute-force the #78 invariant over many edge/probe pairs on a tilted,
    // non-axis-aligned great circle (built like
    // test_orient_sos_breaks_coplanar_consistently), with every probe placed so
    // its intersection lies EXACTLY on the edge's great circle — the on-arc
    // degeneracy.  For each pair, all argument/edge reorderings must agree
    // (`crossing_under_reorder` size 1).  splitmix64 keeps it dependency-free
    // and deterministic.
    let on = great_circle([0.37, -0.81, 0.45]);
    let nrm = normalize(&[0.37, -0.81, 0.45]);
    let mut rng: u64 = 0xd1b54a32d192ed03;
    let mut next = || {
        rng = rng.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z = z ^ (z >> 31);
        (z as f64 / u64::MAX as f64) * 2.0 - 1.0
    };
    let mut checked = 0u32;
    for _ in 0..4000 {
        let a = on(next() * std::f64::consts::PI);
        let b = on(next() * std::f64::consts::PI);
        // Probe whose intersection sits exactly on AB's great circle.
        let xm = on(next() * std::f64::consts::PI);
        assert!(
            orient(&a, &b, &xm).abs() < 1e-12,
            "probe must be on edge's great circle"
        );
        let off = 0.2 + 0.5 * (next() + 1.0); // positive offset, well clear of x
        let c = normalize(&[
            xm[0] + off * nrm[0],
            xm[1] + off * nrm[1],
            xm[2] + off * nrm[2],
        ]);
        let d = normalize(&[
            xm[0] - off * nrm[0],
            xm[1] - off * nrm[1],
            xm[2] - off * nrm[2],
        ]);
        let res = crossing_under_reorder(&a, &b, &c, &d, 10, 20, 30, 40);
        assert_eq!(res.len(), 1, "reorder flip on tilted-circle probe: {res:?}");
        checked += 1;
    }
    assert_eq!(checked, 4000);
}

/// Crossing-number PIP built *directly* on [`robust_crossing`]: `p` is inside
/// iff the arc from a fixed far reference `r` to `p` crosses the ring boundary
/// an odd number of times.  This is the path that actually exercises the #78
/// `on_minor_arc` hardening (unlike [`point_in_ring_robust`], which uses the
/// winding number), so comparing it against the independent [`winding_inside`]
/// trig oracle is a genuine check of the SoS crossing predicate — including at
/// probes whose reference arc grazes an edge's great circle.  `r` carries SoS
/// id 0 and `p` id 1 (probe ids, below the ring's); ring vertex `i` gets id
/// `i + 2`.
///
/// Precondition for the oracle agreement: `r` must lie outside the ring and the
/// ring must be wound CCW, since `winding_inside`'s `> π` test picks the CCW
/// interior.  Every ring fed here ([`ring`] of a mid-latitude box, un-reversed)
/// satisfies both.
fn crossing_pip(r: &Vec3, p: &Vec3, ring: &[Vec3]) -> bool {
    let n = ring.len();
    let mut crossings = 0u32;
    for i in 0..n {
        let a = &ring[i];
        let b = &ring[(i + 1) % n];
        let (ia, ib) = (i as PointId + 2, ((i + 1) % n) as PointId + 2);
        if arcs_cross_sos(r, p, a, b, 0, 1, ia, ib) {
            crossings += 1;
        }
    }
    crossings & 1 == 1
}

#[test]
fn test_crossing_pip_matches_winding_oracle_on_edge() {
    // The crossing-number PIP (which DOES route through robust_crossing /
    // on_minor_arc) must agree with the independent winding-number oracle, for
    // the lon-45/lon-90 meridian boxes (#11 family) and a tilted box, at probes
    // sitting on or beside a polygon edge's great circle — the #78 inputs.  A
    // reference well outside every ring (south pole) anchors the parity.
    let r = latlon_to_unit_vec(-89.0, 0.0);
    let box45 = ring(&[(40.0, 47.0), (42.0, 47.0), (42.0, 45.0), (40.0, 45.0)]);
    let box90 = ring(&[(40.0, 92.0), (42.0, 92.0), (42.0, 90.0), (40.0, 90.0)]);
    for (bx, lon) in [(&box45, 45.0_f64), (&box90, 90.0)] {
        for lat in [39.5, 40.0, 41.0, 41.81, 42.0, 43.0] {
            for dlon in [-0.5, 0.5] {
                let p = latlon_to_unit_vec(lat, lon + dlon);
                assert_eq!(
                    crossing_pip(&r, &p, bx),
                    winding_inside(&p, bx),
                    "crossing-PIP vs winding-oracle disagree at ({lat},{}) for lon-{lon} box",
                    lon + dlon
                );
            }
        }
    }

    // Tilted (non-axis-aligned) box: rotate a mid-latitude square off the axes
    // so none of its edges lie on a coordinate plane, then probe interior and
    // exterior points (rotated the same way).  Reference is rotated too so it
    // stays outside.
    let axis = normalize(&[1.0, 0.3, 0.5]);
    let rot = |v: &Vec3| rotate_about(v, &axis, 0.9);
    let square = ring(&[
        (40.0, -125.0),
        (40.0, -115.0),
        (50.0, -115.0),
        (50.0, -125.0),
    ]);
    let tilted: Vec<Vec3> = square.iter().map(&rot).collect();
    let r_t = rot(&latlon_to_unit_vec(-89.0, 0.0));
    for &(lat, lon) in &[
        (45.0, -120.0),
        (45.0, -90.0),
        (41.0, -118.0),
        (48.0, -123.0),
    ] {
        let p = rot(&latlon_to_unit_vec(lat, lon));
        assert_eq!(
            crossing_pip(&r_t, &p, &tilted),
            winding_inside(&p, &tilted),
            "tilted crossing-PIP vs winding-oracle disagree at ({lat},{lon})"
        );
    }
}

#[test]
fn test_crossing_pip_invariant_under_vertex_rotation() {
    // Rotating the ring's start vertex must not change any crossing-PIP
    // classification — the traversal-order independence the #78 SoS hardening
    // guarantees end-to-end, on the meridian-box family where an edge lies on a
    // base-cell-centre great circle.
    let r = latlon_to_unit_vec(-89.0, 0.0);
    let base = ring(&[(40.0, 47.0), (42.0, 47.0), (42.0, 45.0), (40.0, 45.0)]);
    let rotated = ring(&[(42.0, 47.0), (42.0, 45.0), (40.0, 45.0), (40.0, 47.0)]);
    for lat in [39.0, 41.0, 41.81, 43.0] {
        for lon in [30.0, 44.0, 45.0, 46.0, 48.0] {
            let p = latlon_to_unit_vec(lat, lon);
            assert_eq!(
                crossing_pip(&r, &p, &base),
                crossing_pip(&r, &p, &rotated),
                "vertex rotation changed crossing-PIP at ({lat},{lon})"
            );
        }
    }
}

/// Rotate unit vector `v` by `theta` about unit `axis` (Rodrigues' formula).
/// Lets a test build a non-axis-aligned polygon by rotating an axis-aligned one.
fn rotate_about(v: &Vec3, axis: &Vec3, theta: f64) -> Vec3 {
    let (c, s) = (theta.cos(), theta.sin());
    let k_cross_v = cross(axis, v);
    let k_dot_v = dot(axis, v);
    [
        v[0] * c + k_cross_v[0] * s + axis[0] * k_dot_v * (1.0 - c),
        v[1] * c + k_cross_v[1] * s + axis[1] * k_dot_v * (1.0 - c),
        v[2] * c + k_cross_v[2] * s + axis[2] * k_dot_v * (1.0 - c),
    ]
}

// ── arcs_cross_sos: the uniform symbolic crossing predicate (issue #103) ──

#[test]
fn test_arcs_cross_sos_matches_arcs_cross_hemisphere_confined() {
    // Differential check against the plain geometric arcs_cross.  The bare
    // straddle test is only a valid oracle when both arcs sit inside one open
    // hemisphere (the two great circles then meet at most once there, so a
    // straddle is a crossing); the antipodal counterexample is pinned in the
    // rejection test below.  Skip shared endpoints and near-zero orientations
    // — exactly the configurations the symbolic predicate exists to fix.
    let pts = ring(&[
        (-10.0, 0.0),
        (10.0, 0.0),
        (0.0, -10.0),
        (0.0, 10.0),
        (40.0, -55.0),
        (50.0, -45.0),
        (-32.0, 30.0),
        (12.0, 48.0),
        (33.0, 71.0),
        (-45.0, -20.0),
    ]);
    let degenerate = |a: &Vec3, b: &Vec3, c: &Vec3| orient(a, b, c).abs() < 1e-9;
    let mut checked = 0;
    for a in 0..pts.len() {
        for b in 0..pts.len() {
            for c in 0..pts.len() {
                for d in 0..pts.len() {
                    if a == b || a == c || a == d || b == c || b == d || c == d {
                        continue;
                    }
                    let (pa, pb, pc, pd) = (&pts[a], &pts[b], &pts[c], &pts[d]);
                    let quad = [pa, pb, pc, pd];
                    let confined = (0..4).all(|i| (i + 1..4).all(|j| dot(quad[i], quad[j]) > 0.1));
                    if !confined
                        || degenerate(pa, pc, pb)
                        || degenerate(pc, pb, pd)
                        || degenerate(pb, pd, pa)
                        || degenerate(pd, pa, pc)
                    {
                        continue;
                    }
                    assert_eq!(
                        arcs_cross_sos(pa, pb, pc, pd, a as u64, b as u64, c as u64, d as u64),
                        arcs_cross(pa, pb, pc, pd),
                        "disagrees with arcs_cross at ({a},{b},{c},{d})"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 1000, "fuzz grid too sparse ({checked})");
}

#[test]
fn test_arcs_cross_sos_rejects_antipodal_straddle() {
    // Both straddle conditions hold for two tiny arcs in antipodal regions
    // (their great circles meet on the far side), yet the arcs do not cross.
    // The old pipeline needed the constructed-x on_minor_arc stage for this;
    // the four-sign identity encodes it directly.
    let a = latlon_to_unit_vec(0.0, -5.0);
    let b = latlon_to_unit_vec(0.0, 5.0);
    let c = latlon_to_unit_vec(-5.0, 180.0);
    let d = latlon_to_unit_vec(5.0, 180.0);
    // The bare 4-orientation straddle is fooled ...
    let n_ab = cross(&a, &b);
    let n_cd = cross(&c, &d);
    assert!(
        (dot(&n_ab, &c) > 0.0) != (dot(&n_ab, &d) > 0.0)
            && (dot(&n_cd, &a) > 0.0) != (dot(&n_cd, &b) > 0.0),
        "precondition: straddle gates alone accept the antipodal pair"
    );
    // ... the symbolic predicate is not.
    assert!(!arcs_cross_sos(&a, &b, &c, &d, 0, 1, 2, 3));
    // And the genuine crossing at the origin still reports.
    let e = latlon_to_unit_vec(-5.0, 0.0);
    let f = latlon_to_unit_vec(5.0, 0.0);
    assert!(arcs_cross_sos(&a, &b, &e, &f, 0, 1, 2, 3));
}

#[test]
fn test_arcs_cross_sos_issue_103_meridian_vertex_graze() {
    // The exact parity-breaking configuration from issue #103: the descent
    // probe leg between the centres of nested pixels 19@depth1 and 79@depth2
    // (both on the lon-0 meridian, y == 0 bit-exact) against the box
    // lat[20,25] lon[0,5], whose west edge lies ON that meridian.  The old
    // pipeline counted the graze at vertex (20,0) once (bit-exact coincident
    // hit) and the one at (25,0) zero times (wedge determinant rounded to
    // +8e-20, so the SoS tie-break never engaged) — odd parity, subtree-wide
    // fill inversion.  The symbolic predicate counts each vertex passage
    // exactly once: edges E0 and E2 cross, E1 and the probe-collinear E3 do
    // not — parity even, matching both endpoints lying outside the box.
    use crate::cell_geom::cell_center_vec;

    let p = cell_center_vec(1, 19);
    let q = cell_center_vec(2, 79);
    assert_eq!((p[1], q[1]), (0.0, 0.0), "probe endpoints bit-exact on y=0");
    let v = [
        latlon_to_unit_vec(20.0, 0.0),
        latlon_to_unit_vec(20.0, 5.0),
        latlon_to_unit_vec(25.0, 5.0),
        latlon_to_unit_vec(25.0, 0.0),
    ];
    let crossings: Vec<bool> = (0..4)
        .map(|i| {
            let j = (i + 1) % 4;
            arcs_cross_sos(&p, &q, &v[i], &v[j], 0, 1, 2 + i as u64, 2 + j as u64)
        })
        .collect();
    assert_eq!(
        crossings,
        vec![true, false, true, false],
        "E0/E2 cross once each; E1 and the collinear E3 do not"
    );
}

#[test]
fn test_arcs_cross_sos_parity_even_under_id_relabeling() {
    // Individual crossing attributions may legitimately move between incident
    // edges under a different SoS id assignment; the load-bearing invariant is
    // the total parity.  Both probe endpoints are outside the box, so the
    // crossing count must stay even for every vertex-id rotation.
    use crate::cell_geom::cell_center_vec;

    let p = cell_center_vec(1, 19);
    let q = cell_center_vec(2, 79);
    let v = [
        latlon_to_unit_vec(20.0, 0.0),
        latlon_to_unit_vec(20.0, 5.0),
        latlon_to_unit_vec(25.0, 5.0),
        latlon_to_unit_vec(25.0, 0.0),
    ];
    for rot in 0..4u64 {
        let count = (0..4)
            .filter(|&i| {
                let j = (i + 1) % 4;
                arcs_cross_sos(
                    &p,
                    &q,
                    &v[i],
                    &v[j],
                    0,
                    1,
                    2 + (i as u64 + rot) % 4,
                    2 + (j as u64 + rot) % 4,
                )
            })
            .count();
        assert_eq!(count % 2, 0, "odd parity under id rotation {rot}");
    }
}

#[test]
fn test_arcs_cross_sos_vertex_graze_counts_once() {
    // A probe passing exactly through a shared ring vertex whose two incident
    // edges continue to opposite sides of the probe circle must count exactly
    // one crossing across the pair — the emergent half-open convention.  Run
    // it on the axis-aligned meridian family and on tilted copies so the
    // degeneracy is not coordinate luck.
    let axis_cases: Vec<([Vec3; 3], [Vec3; 2])> = vec![
        // (u, v, w): v on the probe circle (lon 0), u west, w east.
        (
            [
                latlon_to_unit_vec(30.0, -8.0),
                latlon_to_unit_vec(35.0, 0.0),
                latlon_to_unit_vec(32.0, 7.0),
            ],
            [latlon_to_unit_vec(20.0, 0.0), latlon_to_unit_vec(50.0, 0.0)],
        ),
        // equator probe, vertex exactly on it
        (
            [
                latlon_to_unit_vec(-6.0, 100.0),
                latlon_to_unit_vec(0.0, 103.0),
                latlon_to_unit_vec(9.0, 106.0),
            ],
            [
                latlon_to_unit_vec(0.0, 95.0),
                latlon_to_unit_vec(0.0, 110.0),
            ],
        ),
    ];
    let tilt_axis = normalize(&[0.3, -0.5, 0.81]);
    for (tilt, theta) in [(false, 0.0), (true, 0.4), (true, 1.1)] {
        for (tri, probe) in &axis_cases {
            let mv = |x: &Vec3| {
                if tilt {
                    rotate_about(x, &tilt_axis, theta)
                } else {
                    *x
                }
            };
            let (u, v, w) = (mv(&tri[0]), mv(&tri[1]), mv(&tri[2]));
            let (p, q) = (mv(&probe[0]), mv(&probe[1]));
            let c1 = arcs_cross_sos(&p, &q, &u, &v, 0, 1, 2, 3);
            let c2 = arcs_cross_sos(&p, &q, &v, &w, 0, 1, 3, 4);
            assert_eq!(
                c1 as u32 + c2 as u32,
                1,
                "vertex passage must count exactly once (tilt {theta})"
            );
        }
    }
}

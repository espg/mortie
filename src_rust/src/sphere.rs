//! Spherical primitives for the hierarchical region coverer.
//!
//! Everything here operates on **unit 3-vectors** on the sphere.  The two core
//! predicates are [`orient`] (the sign of a scalar triple product) and
//! [`arcs_cross`] (do two great-circle arcs cross?), built from it.  On top of
//! those sits the single point-in-polygon path — [`point_in_ring_robust`]
//! (spherical winding number, correct at any polygon size including
//! hemisphere+, issue #22) — plus [`parity_filled_robust`], the even-odd rule
//! over a *ring-set* that gives holes and multipart geometry for free (see issue
//! #30).  [`orient_sos`] / [`robust_crossing`] add a Simulation-of-Simplicity
//! tie-break for the descent's degenerate cell-centre crossings (issue #11).
//! Ring orientation (RFC 7946 / S2 right-hand rule) is normalized at ingest by
//! [`crate::coverage`]; see [`point_in_ring_robust`] for the winding contract.

/// Unit 3-vector on the sphere.
pub type Vec3 = [f64; 3];

// ── vector helpers ───────────────────────────────────────────────────────

#[inline]
pub fn dot(a: &Vec3, b: &Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub fn cross(a: &Vec3, b: &Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub fn norm(a: &Vec3) -> f64 {
    dot(a, a).sqrt()
}

#[inline]
pub fn normalize(a: &Vec3) -> Vec3 {
    let n = norm(a);
    if n == 0.0 {
        *a
    } else {
        [a[0] / n, a[1] / n, a[2] / n]
    }
}

/// Convert lat/lon (degrees) to a unit 3-vector.
#[inline]
pub fn latlon_to_unit_vec(lat_deg: f64, lon_deg: f64) -> Vec3 {
    let la = lat_deg.to_radians();
    let lo = lon_deg.to_radians();
    let (sla, cla) = (la.sin(), la.cos());
    let (slo, clo) = (lo.sin(), lo.cos());
    [cla * clo, cla * slo, sla]
}

// ── core predicates ──────────────────────────────────────────────────────

/// Orientation of three unit vectors: the scalar triple product
/// `a · (b × c) = det[a b c]`.
///
/// Equivalently `(a × b) · c`, so its sign tells which side of the directed
/// great circle `a → b` the point `c` lies on: `> 0` left, `< 0` right,
/// `== 0` on the great circle.
#[inline]
pub fn orient(a: &Vec3, b: &Vec3, c: &Vec3) -> f64 {
    dot(a, &cross(b, c))
}

/// Do the great-circle arcs `a → b` and `c → d` cross?
///
/// Uses the standard 4-orientation test: the arcs cross iff `c` and `d` lie on
/// opposite sides of great circle `AB` **and** `a` and `b` lie on opposite
/// sides of great circle `CD`.
///
/// Precondition: each arc is **minor** (shorter than a hemisphere), which holds
/// for HEALPix cell edges and for polygon edges between consecutive vertices.
/// An exactly-touching configuration (a zero orientation) reports `false`; the
/// classifier pairs this with a vertex-in-cell check to catch grazes.
pub fn arcs_cross(a: &Vec3, b: &Vec3, c: &Vec3, d: &Vec3) -> bool {
    let d1 = orient(a, b, c);
    let d2 = orient(a, b, d);
    if (d1 > 0.0) == (d2 > 0.0) {
        return false; // c, d on the same side of AB
    }
    let d3 = orient(c, d, a);
    let d4 = orient(c, d, b);
    (d3 > 0.0) != (d4 > 0.0) // a, b on opposite sides of CD
}

/// Like [`arcs_cross`], but with the two great-circle normals supplied by the
/// caller: `n_ab = a × b` and `n_cd = c × d`.  Since `orient(a, b, x) =
/// (a × b) · x`, the four side tests become plain dot products — no cross
/// product in the inner loop.  The descent hot path reuses these normals (a
/// polygon edge's `n_ab` is fixed across every cell it is tested against, and
/// the probe arc's normal is computed once per fan of edges), so this is the
/// per-cell form of [`arcs_cross`].  Identical result to `arcs_cross(a, b, c,
/// d)` for unit inputs.
#[inline]
pub fn arcs_cross_n(a: &Vec3, b: &Vec3, n_ab: &Vec3, c: &Vec3, d: &Vec3, n_cd: &Vec3) -> bool {
    let d1 = dot(n_ab, c);
    let d2 = dot(n_ab, d);
    if (d1 > 0.0) == (d2 > 0.0) {
        return false; // c, d on the same side of AB
    }
    let d3 = dot(n_cd, a);
    let d4 = dot(n_cd, b);
    (d3 > 0.0) != (d4 > 0.0) // a, b on opposite sides of CD
}

// ── robust any-size point-in-ring (issue #22 / #11) ──────────────────────
//
// This is the single point-in-ring path (the gnomonic / cap-axis-edge-cross
// backends were removed at the Phase-3 cutover, #22).  It is correct at **any**
// polygon size — including hemisphere+ rings such as "everything except
// Antarctica" — and degeneracy-free on edges whose great circle passes exactly
// through HEALPix cell centres (issue #11).
//
// Two layers, kept separate on purpose:
//
//   1. The point-in-ring decision [`point_in_ring_robust`] uses the signed
//      spherical **winding number** ([`ring_winding_at`], Bevis & Chatelain
//      1989): the sum of signed angles the directed edges subtend at the test
//      point is `≈ +2π` inside the counter-clockwise interior and `≈ 0` outside,
//      with no reference point, no projection, and no minor-arc precondition —
//      so it is correct for hemisphere+ rings by construction.  This is the
//      backend wired into the seed PIP.  It runs only at the 12 base-cell seeds,
//      so its per-vertex trig is off the hot path.
//
//   2. The orientation primitives [`orient_sos`] and [`robust_crossing`] are the
//      **degeneracy-free building blocks** the hierarchical descent's per-cell
//      parity flips (`arc_crossing_parity`, a later phase) will consume to fix
//      issue #11 — where the descent predicate hits an exact-zero triple product
//      at HEALPix cell centres.  `orient_sos` is the scalar triple product of
//      [`orient`] with its exact-zero (coplanar) case broken by **Simulation of
//      Simplicity** (Edelsbrunner & Mücke 1990): a symbolic perturbation keyed
//      to each vertex's stable identity, so a coplanar triple resolves to a
//      definite, consistent side regardless of traversal order — the f64+SoS
//      approach @espg signed off on (#22).  `robust_crossing` builds an S2
//      `CrossingSign`-style great-circle-*segment* test on top, valid for arcs
//      up to ~180° (it verifies the intersection lies on both segments, which a
//      bare 4-orientation straddle does not guarantee for long arcs).
//
// An *edge-crossing* PIP built on layer 2 (rather than the winding number) is
// deferred: its long-arc / scalloped-boundary behaviour still needs validating
// against the winding reference.

/// Stable identity of a point feeding [`orient_sos`], used by its Simulation-of-
/// Simplicity tie-break.  For ring vertices this is the vertex index; the
/// symbolic perturbation is a strictly increasing function of it, so identities
/// only need to be **distinct and consistently ordered**, not contiguous.
pub type PointId = u64;

/// Robust orientation sign of three unit vectors as `-1 | 0 | +1`.
///
/// Returns the sign of the scalar triple product `a · (b × c)` (see [`orient`]),
/// but the decision is taken on a **canonical** (identity-sorted) evaluation of
/// that determinant, with the parity of the sort reapplied.  This is essential:
/// the f64 triple product is not antisymmetric under argument permutation (the
/// same coplanar points can round to `0.0` in one ordering and `~1e-17` in
/// another), so gating on the as-given det being `0.0` would let different
/// permutations disagree.  Deciding from one canonical evaluation makes every
/// permutation reduce to that result times its own sign — true antisymmetry and
/// cyclic invariance.  When the canonical determinant is exactly `0.0` — the
/// three points coplanar with the origin, e.g. an edge great circle passing
/// through the test point — the tie is broken with Simulation of Simplicity
/// using the points' identities `ia, ib, ic`.
///
/// SoS imagines each point's coordinates perturbed by successively smaller
/// powers of an infinitesimal `ε → 0⁺`; the first non-vanishing term of the
/// perturbed determinant decides the sign.  For the orientation predicate that
/// expansion reduces to a fixed-order sequence of 2×2 sub-determinants of the
/// *unperturbed* coordinates, whose final term is a pure function of the
/// identity order and is non-zero — so the predicate is **total** (never
/// returns 0 once identities are distinct).  The construction is antisymmetric:
/// swapping two points flips the sign, exactly like the geometric determinant,
/// which is what keeps edge-crossing parity consistent.
#[inline]
pub fn orient_sos(a: &Vec3, b: &Vec3, c: &Vec3, ia: PointId, ib: PointId, ic: PointId) -> i32 {
    // The f64 triple product is NOT antisymmetric under argument permutation: for
    // the same coplanar points one ordering can round to exact 0.0 while a
    // permuted ordering rounds to ~1e-17.  Gating SoS on the *as-given* det being
    // 0.0 therefore breaks antisymmetry (some permutations take the geometric
    // branch, others the symbolic one).  The fix is to decide everything from a
    // single **canonical** (identity-sorted) evaluation, then reapply the parity
    // of the sort: every permutation reduces to the same canonical result times
    // its own sign, so antisymmetry and cyclic invariance hold by construction.
    let mut pts: [(PointId, &Vec3); 3] = [(ia, a), (ib, b), (ic, c)];
    let mut perm_sign = 1i32;
    if pts[0].0 > pts[1].0 {
        pts.swap(0, 1);
        perm_sign = -perm_sign;
    }
    if pts[1].0 > pts[2].0 {
        pts.swap(1, 2);
        perm_sign = -perm_sign;
    }
    if pts[0].0 > pts[1].0 {
        pts.swap(0, 1);
        perm_sign = -perm_sign;
    }
    let (p, q, r) = (pts[0].1, pts[1].1, pts[2].1);
    // Evaluate the geometric determinant ONCE, on the canonical order.
    let det = orient(p, q, r);
    let canon = if det > 0.0 {
        1
    } else if det < 0.0 {
        -1
    } else {
        // True degeneracy on the canonical order: symbolic perturbation.
        sos_sorted_sign(p, q, r)
    };
    perm_sign * canon
}

/// SoS tie-break for three coplanar points already ordered by identity
/// (`p < q < r`).  Returns a guaranteed non-zero `-1 | +1`.
///
/// The perturbation expands the `[p q r]` determinant into 2×2 minors of the
/// real coordinates, evaluated in the canonical Edelsbrunner–Mücke order; the
/// first non-zero minor decides the sign (with its attached parity).  The final
/// `+1` fallback is reached only if every minor vanishes — i.e. the points are
/// identical in every coordinate, impossible for three distinct unit vectors —
/// so it is purely a total-function guard.
#[inline]
fn sos_sorted_sign(p: &Vec3, q: &Vec3, r: &Vec3) -> i32 {
    let minors = [
        (1.0, q[0] * r[1] - q[1] * r[0]),
        (-1.0, q[0] * r[2] - q[2] * r[0]),
        (1.0, q[1] * r[2] - q[2] * r[1]),
        (-1.0, p[0] * r[1] - p[1] * r[0]),
        (1.0, p[0] * r[2] - p[2] * r[0]),
        (-1.0, p[1] * r[2] - p[2] * r[1]),
        (1.0, p[0] * q[1] - p[1] * q[0]),
        (-1.0, p[0] * q[2] - p[2] * q[0]),
        (1.0, p[1] * q[2] - p[2] * q[1]),
    ];
    for (sgn, val) in minors {
        if val > 0.0 {
            return sgn as i32;
        }
        if val < 0.0 {
            return -(sgn as i32);
        }
    }
    1
}

/// Is unit vector `x` on the **minor** great-circle arc from `a` to `b`,
/// counting the arc as half-open `[a, b)` — the start endpoint is on the arc,
/// the end endpoint is not?
///
/// `x` is between `a` and `b` on the shorter arc iff it lies the same rotational
/// direction from `a` as `b` does and continues that direction to `b`.  The
/// half-open convention is what makes the crossing count correct when the
/// reference→point arc passes exactly **through a ring vertex**: the vertex is
/// shared by two edges, and `[start, end)` assigns the grazing crossing to
/// exactly one of them, so it is counted once rather than twice or zero times.
/// Used by [`robust_crossing`] to confirm a great-circle intersection falls on
/// the real segments and not their antipodal halves (the failure mode of a bare
/// 4-orientation straddle on long arcs).
#[inline]
fn on_minor_arc(x: &Vec3, a: &Vec3, b: &Vec3) -> bool {
    let ab = cross(a, b);
    dot(&cross(a, x), &ab) >= 0.0 && dot(&cross(x, b), &ab) > 0.0
}

/// Robust great-circle-*segment* crossing using [`orient_sos`].
///
/// Returns `true` iff the arcs `a → b` and `c → d` cross at a point interior to
/// both.  A 4-orientation straddle test (each pair on opposite sides of the
/// other's great circle) is necessary but **not** sufficient for arcs longer
/// than a quadrant — the two great circles meet at two antipodal points, and the
/// straddle can be satisfied by the one on the *far* halves.  So we additionally
/// locate the intersection and require it to lie on both minor arcs
/// ([`on_minor_arc`]).  This keeps the test correct for the long reference→point
/// arc of the crossing-number PIP, where [`arcs_cross`]'s minor-arc precondition
/// does not hold.  SoS makes the straddle decisions total, so coplanar and
/// shared-endpoint configurations resolve to a definite, consistent side.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn robust_crossing(
    a: &Vec3,
    b: &Vec3,
    c: &Vec3,
    d: &Vec3,
    ia: PointId,
    ib: PointId,
    ic: PointId,
    id: PointId,
) -> bool {
    // c, d must straddle great circle AB and a, b must straddle great circle CD.
    let abc = orient_sos(a, b, c, ia, ib, ic);
    let abd = orient_sos(a, b, d, ia, ib, id);
    if abc == abd {
        return false;
    }
    let cda = orient_sos(c, d, a, ic, id, ia);
    let cdb = orient_sos(c, d, b, ic, id, ib);
    if cda == cdb {
        return false;
    }
    // Disambiguate which antipodal intersection the straddle refers to: it must
    // lie on both minor arcs.  (The intersection is along ±(AB × CD).)
    let mut x = normalize(&cross(&cross(a, b), &cross(c, d)));
    if !on_minor_arc(&x, a, b) {
        x = [-x[0], -x[1], -x[2]];
    }
    on_minor_arc(&x, a, b) && on_minor_arc(&x, c, d)
}

/// Signed spherical winding of `ring` as seen from `x`: the sum of the signed
/// angles each directed edge subtends at `x` (Bevis & Chatelain 1989).  It is
/// `≈ +2π` when `x` lies inside the ring's counter-clockwise interior (the
/// region to the left of the directed edges), `≈ 0` when outside, and `≈ −2π`
/// inside a clockwise ring — so `|winding| > π` is the inside test.  This is the
/// any-size point-in-ring decision used by [`point_in_ring_robust`]; it needs no
/// reference point and no minor-arc precondition, so it is correct for
/// hemisphere+ rings and for edges whose great circle passes through a cell
/// centre.  It runs only at the base-cell seeds, so its per-vertex trig is not
/// on a hot path.
fn ring_winding_at(x: &Vec3, ring: &[Vec3]) -> f64 {
    let n = ring.len();
    let mut total = 0.0;
    for i in 0..n {
        let a = &ring[i];
        let b = &ring[(i + 1) % n];
        // Project a, b onto the plane perpendicular to x and measure the signed
        // angle between the projections.
        let da = dot(a, x);
        let db = dot(b, x);
        let pa = normalize(&[a[0] - da * x[0], a[1] - da * x[1], a[2] - da * x[2]]);
        let pb = normalize(&[b[0] - db * x[0], b[1] - db * x[1], b[2] - db * x[2]]);
        let cos_t = dot(&pa, &pb).clamp(-1.0, 1.0);
        let ang = cos_t.acos();
        let sgn = dot(&cross(&pa, &pb), x);
        total += if sgn >= 0.0 { ang } else { -ang };
    }
    total
}

/// Robust spherical point-in-ring test, valid at **any** ring size.
///
/// `p` is inside iff the ring's signed spherical winding at `p`
/// ([`ring_winding_at`]) exceeds `π` — i.e. `p` is in the counter-clockwise
/// interior (the region to the left of the directed edges, the same convention
/// the even-odd fill assumes).  There is no projection centre to go singular and
/// no sub-hemisphere precondition, so this is correct for hemisphere+ rings such
/// as "everything except Antarctica" (#22) and degeneracy-free when an edge's
/// great circle passes through a HEALPix cell centre (#11).
///
/// # Winding (orientation) contract
///
/// Ring vertex order **carries meaning** and is the caller's responsibility.
/// mortie adopts the RFC 7946 §3.1.6 / S2 **right-hand rule**: an exterior ring
/// is wound **counter-clockwise** (CCW) so its interior — the smaller of the two
/// regions the ring divides the sphere into for sub-hemisphere rings — lies to
/// the **left** of each directed edge; **holes are wound clockwise** (CW). Under
/// even-odd fill ([`parity_filled_robust`]) a CW ring simply winds the opposite
/// way, which is exactly what carves a hole.
///
/// This orientation convention is *the* disambiguation that lets the test work
/// for hemisphere-plus rings: on a sphere a closed ring bounds two complementary
/// regions of equal standing, so "inside" is undefined by the vertex set alone —
/// only the winding direction picks which side is interior. A ≤-hemisphere ring
/// has an unambiguous "smaller side", so [`crate::coverage`] auto-normalizes its
/// orientation at ingest (see `build_ring`); past a hemisphere that shortcut
/// breaks and the right-hand rule is required, so those rings are passed through
/// untouched. A ring supplied with reversed orientation selects the
/// complementary region — not a bug, the documented contract.
///
/// The companion SoS predicates [`orient_sos`] and [`robust_crossing`] are the
/// orientation-only building blocks the descent's per-cell parity flips will use
/// in a later phase; an *edge-crossing* PIP built on them is deferred while its
/// long-arc / scalloped-boundary behaviour is validated against this winding
/// reference.
pub fn point_in_ring_robust(p: &Vec3, ring: &[Vec3]) -> bool {
    if ring.len() < 3 {
        return false;
    }
    ring_winding_at(p, ring) > std::f64::consts::PI
}

/// Is `p` inside the filled region defined by `rings` under the **even-odd**
/// rule — i.e. inside an *odd* number of rings?  The any-size robust point-in-
/// ring backend ([`point_in_ring_robust`]) is the single path (the gnomonic /
/// cap-axis-edge-cross backends were removed at the Phase-3 cutover, #22), so
/// holes (a point in the hole is inside both the outer and the hole ring → even
/// → empty) and multipart geometry (separate outer rings) fall out of the rule
/// for free, correct at any polygon size including hemisphere+.
///
/// Rings must follow the RFC 7946 §3.1.6 / S2 right-hand-rule winding contract
/// documented on [`point_in_ring_robust`] (CCW exterior, CW holes); past a
/// hemisphere, orientation is the only thing that makes "inside" well-defined.
/// [`crate::coverage`] normalizes sub-hemisphere rings to this convention at
/// ingest, so callers feeding everyday (possibly CW) input do not invert.
pub fn parity_filled_robust(p: &Vec3, rings: &[Vec<Vec3>]) -> bool {
    let mut inside = false;
    for ring in rings {
        if point_in_ring_robust(p, ring) {
            inside = !inside;
        }
    }
    inside
}

/// Signed winding direction of a sub-hemisphere `ring`: `+1` if it is wound
/// counter-clockwise (interior — the smaller side — to the **left** of the
/// directed edges, the RFC 7946 / S2 convention), `-1` if clockwise, `0` if the
/// winding is too small to call (degenerate / collinear ring).
///
/// This is only meaningful for a ring that fits within a hemisphere, where the
/// two regions the ring bounds are unambiguously "small" and "large" and the
/// small side is the intended interior.  It probes the ring's own cap axis (the
/// normalized vertex sum), which for a sub-hemisphere ring lies on the small
/// side: the signed spherical winding there is `≈ +2π` for a CCW ring and
/// `≈ −2π` for a CW ring ([`ring_winding_at`]).  Used by [`crate::coverage`] to
/// auto-correct everyday CW input; it must **not** be used to "normalize" a
/// hemisphere+ ring, where area alone cannot pick the interior side (#22).
pub fn ring_winding_sign(ring: &[Vec3]) -> i32 {
    if ring.len() < 3 {
        return 0;
    }
    let mut s = [0.0, 0.0, 0.0];
    for v in ring {
        s[0] += v[0];
        s[1] += v[1];
        s[2] += v[2];
    }
    if norm(&s) < 1e-12 {
        return 0; // balanced ⇒ not sub-hemisphere; caller must not normalize
    }
    ring_winding_sign_at(ring, &normalize(&s))
}

/// [`ring_winding_sign`] with the ring's cap `axis` (its normalized vertex sum)
/// supplied by the caller.  A caller that already holds the axis — e.g.
/// [`crate::coverage`]'s ingest normalization, which computes it to size the
/// ring's bounding cap — passes it here instead of having the sign test
/// recompute the vertex sum.  `axis` must be the unit normalized vertex sum (the
/// small side of a sub-hemisphere ring); callers without it use
/// [`ring_winding_sign`].
pub fn ring_winding_sign_at(ring: &[Vec3], axis: &Vec3) -> i32 {
    if ring.len() < 3 {
        return 0;
    }
    let w = ring_winding_at(axis, ring);
    if w > std::f64::consts::PI {
        1
    } else if w < -std::f64::consts::PI {
        -1
    } else {
        0
    }
}

// ── tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
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
        assert!(robust_crossing(&ns.0, &ns.1, &ew.0, &ew.1, 0, 1, 2, 3));
        let a = ring(&[(20.0, -10.0), (20.0, 10.0)]);
        let b = ring(&[(40.0, -10.0), (40.0, 10.0)]);
        assert!(!robust_crossing(&a[0], &a[1], &b[0], &b[1], 0, 1, 2, 3));
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
            !robust_crossing(&r, &p, &c, &d, 0, 1, 2, 3),
            "long arc must not falsely cross a far north-cap edge"
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
}

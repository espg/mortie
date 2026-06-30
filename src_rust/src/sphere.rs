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

/// SoS identity of the *intersection* point `x` inside [`robust_crossing`].  `x`
/// is a derived point (the AB×CD meet), not one of the four arc endpoints, so it
/// needs its own identity for the half-plane tie-break in [`on_minor_arc`].  A
/// reserved top id keeps it distinct from every real vertex and orders it
/// consistently against `a, b, c, d` in both the AB and CD wedge tests, so the
/// same physical `x` perturbs the same way regardless of which arc it is checked
/// against.  Endpoint *coincidence* is settled before SoS (see [`on_minor_arc`]),
/// so this id only governs the rare canonical-order tie of an interior on-circle
/// `x`.
const INTERSECTION_ID: PointId = PointId::MAX;

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

/// Robust sign of the half-plane determinant `(u × v) · n` as `-1 | +1`, where
/// `n` is the edge's great-circle normal (`a × b`).  This is the per-endpoint
/// wedge test inside [`on_minor_arc`]; on the edge's great circle `u × v` is
/// parallel to `±n`, so the sign says whether `v` lies the same rotational way
/// from `u` as `b` does from `a`.
///
/// Like [`orient_sos`], the decision is taken on a **canonical** (identity-
/// sorted) evaluation with the sort parity reapplied: the determinant is
/// antisymmetric in `(u, v)`, so the same geometry rounds the same way in every
/// argument order, killing the f64-noise sign flip when `x` lies exactly on the
/// edge's great circle.  An exact-`0.0` determinant (`u`, `v` parallel — the
/// probe coincident with an endpoint) is broken by Simulation of Simplicity: the
/// canonical-lower-id point is perturbed first along `e₀, e₁, e₂`, and the first
/// non-vanishing term decides.  The result is **total** (never `0`) and
/// antisymmetric, matching the `orient_sos` contract the straddle gates rely on.
#[inline]
fn half_plane_sign(u: &Vec3, v: &Vec3, n: &Vec3, iu: PointId, iv: PointId) -> i32 {
    // Canonical order by identity, parity tracked (the det negates under a swap).
    let (p, q, perm) = if iu <= iv { (u, v, 1) } else { (v, u, -1) };
    let det = dot(&cross(p, q), n);
    let canon = if det > 0.0 {
        1
    } else if det < 0.0 {
        -1
    } else {
        // p ∥ q: symbolic perturbation, lower-id point (p) first then q.
        sos_half_plane_sign(p, q, n)
    };
    perm * canon
}

/// SoS tie-break for [`half_plane_sign`] when `(p × q) · n` vanishes (the two
/// points are parallel).  Perturbs the canonical-lower point `p` along the unit
/// axes `e₀, e₁, e₂` (largest perturbation), then `q`; the first non-zero term
/// `(eₖ × q) · n` / `(p × eₖ) · n` decides.  Returns a guaranteed non-zero
/// `-1 | +1`; the final `+1` is reached only if `n` is the zero vector (a
/// degenerate edge `a ∥ b`), which the descent never feeds, so it is a
/// total-function guard.
#[inline]
fn sos_half_plane_sign(p: &Vec3, q: &Vec3, n: &Vec3) -> i32 {
    let e = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for ek in &e {
        let t = dot(&cross(ek, q), n);
        if t > 0.0 {
            return 1;
        }
        if t < 0.0 {
            return -1;
        }
    }
    for ek in &e {
        let t = dot(&cross(p, ek), n);
        if t > 0.0 {
            return 1;
        }
        if t < 0.0 {
            return -1;
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
///
/// `x` is on the great circle in this call (it is the AB×CD intersection), so the
/// two wedge determinants are O(1) in the arc interior and the raw f64 sign is
/// only noise-prone when `x` lands near an endpoint (`±a` / `±b`) — the #78
/// degeneracy.  Each wedge sign goes through [`half_plane_sign`], decided on a
/// canonical, identity-keyed order, so it cannot flip under argument/edge
/// reordering.  The half-open `[a, b)` convention is kept without a tolerance: an
/// `x` coinciding **exactly** with the start vertex `a` is included and one
/// coinciding with the end vertex `b` is excluded (coincidence ⇔ the cross
/// product is exactly the zero vector and the points are not antipodal), exactly
/// as the original `>= 0.0` / `> 0.0` bounds did.  `ix`, `ia`, `ib` are the SoS
/// identities of `x`, `a`, `b`.
#[inline]
fn on_minor_arc(x: &Vec3, a: &Vec3, b: &Vec3, ix: PointId, ia: PointId, ib: PointId) -> bool {
    let n = cross(a, b);
    // Start bound is inclusive at a, end bound exclusive at b: settle exact
    // endpoint coincidence first (order-independent), then the robust wedge sign.
    if coincident(x, b) {
        return false; // x ≡ b ⇒ end excluded
    }
    // Start is inclusive at a: x ≡ a satisfies it outright, else the robust wedge.
    let s_start = half_plane_sign(a, x, &n, ia, ix);
    let s_end = half_plane_sign(x, b, &n, ix, ib);
    // Invariant: the SoS-hardened wedge sign is total — never undecided (#78).
    debug_assert!(
        s_start != 0 && s_end != 0,
        "on_minor_arc wedge sign must be total"
    );
    (coincident(x, a) || s_start > 0) && s_end > 0
}

/// Do unit vectors `p` and `q` point in exactly the same direction?  True iff
/// `p × q` is the exact zero vector (parallel) and `p · q > 0` (not antipodal).
/// This is a tolerance-free coincidence test for the endpoint cases of
/// [`on_minor_arc`].
#[inline]
fn coincident(p: &Vec3, q: &Vec3) -> bool {
    cross(p, q) == [0.0, 0.0, 0.0] && dot(p, q) > 0.0
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
    // lie on both minor arcs.  (The intersection is along ±(AB × CD).)  The
    // on-arc tests carry SoS identities so an `x` landing exactly on an endpoint
    // resolves to a definite, traversal-order-independent side (#78).
    let xi = INTERSECTION_ID;
    let mut x = normalize(&cross(&cross(a, b), &cross(c, d)));
    if !on_minor_arc(&x, a, b, xi, ia, ib) {
        x = [-x[0], -x[1], -x[2]];
    }
    on_minor_arc(&x, a, b, xi, ia, ib) && on_minor_arc(&x, c, d, xi, ic, id)
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
mod tests;

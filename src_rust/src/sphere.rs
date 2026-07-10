//! Spherical primitives for the hierarchical region coverer.
//!
//! Everything here operates on **unit 3-vectors** on the sphere.  The two core
//! predicates are [`orient`] (the sign of a scalar triple product) and
//! [`arcs_cross`] (do two great-circle arcs cross?), built from it.  On top of
//! those sits the single point-in-polygon path — [`point_in_ring_robust`]
//! (spherical winding number, correct at any polygon size including
//! hemisphere+, issue #22) — plus [`parity_filled_robust`], the even-odd rule
//! over a *ring-set* that gives holes and multipart geometry for free (see issue
//! #30).  [`orient_sos`] / [`arcs_cross_sos`] add a Simulation-of-Simplicity
//! tie-break for the descent's degenerate cell-centre crossings (issues #11,
//! #103).
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
//   2. The orientation primitives [`orient_sos`] and [`arcs_cross_sos`] are the
//      **degeneracy-free building blocks** the hierarchical descent's per-cell
//      parity flips (`arc_crossing_parity`) consume to fix issues #11/#103 —
//      where the descent predicate hits an exact-zero triple product at HEALPix
//      cell centres.  `orient_sos` is the scalar triple product of [`orient`]
//      with its exact-zero (coplanar) case broken by **Simulation of
//      Simplicity** (Edelsbrunner & Mücke 1990): a symbolic perturbation keyed
//      to each vertex's stable identity, so a coplanar triple resolves to a
//      definite, consistent side regardless of traversal order — the f64+SoS
//      approach @espg signed off on (#22).  `arcs_cross_sos` builds the
//      great-circle-*segment* test on top purely from those signs (the S2
//      `SimpleCrossing` identity), valid for minor arcs (< 180°) — no
//      constructed intersection point, so no derived-point rounding for a
//      degeneracy to hide in (the issue #103 failure mode of the retired
//      two-stage `robust_crossing`).
//
// An *edge-crossing* PIP built on layer 2 (rather than the winding number) is
// deferred: its long-arc / scalloped-boundary behaviour still needs validating
// against the winding reference.

/// Stable identity of a point feeding [`orient_sos`], used by its Simulation-of-
/// Simplicity tie-break.  For ring vertices this is the vertex index; the
/// symbolic perturbation is a strictly increasing function of it, so identities
/// only need to be **distinct and consistently ordered**, not contiguous.
pub type PointId = u64;

// ── exact determinant sign (Shewchuk error-free expansions) ───────────────
//
// SoS breaks ties only at an **exact** zero, but f64 evaluation of the triple
// product turns a geometrically degenerate configuration into ~1e-17 noise
// whenever the inputs are not bit-exactly coplanar (e.g. HEALPix points on the
// lon-45 meridians, where cos(π/4) and sin(π/4) round differently).  Noise
// signs are individually stable under permutation (the canonical evaluation)
// but **jointly inconsistent** — the set of signs need not describe any real
// point configuration — which is exactly how issue #103's parity chain broke
// off the bit-exact grid.  The fix is the classical one (Shewchuk 1997):
// decide the sign of the determinant **exactly** with error-free float
// expansions, so every sign describes the true configuration of the actual
// f64 points and the predicate axioms hold jointly; SoS then only ever
// arbitrates true zeros, where its global perturbation is consistent by
// construction.  A cheap error-bound filter keeps the fast path fast.

/// Error-free sum: `a + b = s + e` exactly.
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bv = s - a;
    let av = s - bv;
    (s, (a - av) + (b - bv))
}

/// Error-free product via FMA: `a * b = p + e` exactly.
#[inline]
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

/// Exact sign of `Σ terms`, where every term is an exact f64 component.
///
/// Builds a nonoverlapping expansion by repeated GROW-EXPANSION (Shewchuk
/// 1997, Thm. 10): each term is absorbed with an error-free `two_sum` cascade
/// over the expansion-so-far (kept in increasing magnitude), so the result's
/// components are nonoverlapping and ascending, and the **last** component
/// alone carries the sign of the exact sum.  O(n²) in the term count (≤ 24
/// here) and only reached when the fast filtered path abstains.
fn exact_sum_sign(terms: &[f64]) -> i32 {
    let mut h: Vec<f64> = Vec::with_capacity(terms.len() + 4);
    let mut tmp: Vec<f64> = Vec::with_capacity(terms.len() + 4);
    for &t in terms {
        if t == 0.0 {
            continue;
        }
        tmp.clear();
        let mut q = t;
        for &hi in &h {
            let (sum, err) = two_sum(q, hi);
            q = sum;
            if err != 0.0 {
                tmp.push(err);
            }
        }
        if q != 0.0 {
            tmp.push(q);
        }
        std::mem::swap(&mut h, &mut tmp);
    }
    match h.last() {
        None => 0,
        Some(&m) if m > 0.0 => 1,
        _ => -1,
    }
}

/// Exact sign of the scalar triple product `a · (b × c)` as `-1 | 0 | +1`.
///
/// A relative-error filter accepts the fast f64 evaluation when its magnitude
/// provably dominates the rounding error; otherwise the determinant is
/// re-evaluated exactly as a 12-term error-free expansion.
fn orient_exact_sign(a: &Vec3, b: &Vec3, c: &Vec3) -> i32 {
    let det = orient(a, b, c);
    // Permanent-style magnitude bound on the six products.
    let perm = (b[1] * c[2]).abs()
        + (b[2] * c[1]).abs()
        + (b[2] * c[0]).abs()
        + (b[0] * c[2]).abs()
        + (b[0] * c[1]).abs()
        + (b[1] * c[0]).abs();
    let mag = a[0].abs().max(a[1].abs()).max(a[2].abs()) * perm;
    // ~2^-50: comfortably above the true bound (~1e-15 · mag) for a 6-product
    // sum-of-products; anything larger is decided by the float sign.
    if det.abs() > mag * 1e-15 {
        return if det > 0.0 { 1 } else { -1 };
    }
    // Exact: each a_i · (b_j c_k − b_k c_j) contributes 4 exact components.
    let mut terms: Vec<f64> = Vec::with_capacity(12);
    for (ai, bj, ck, bk, cj) in [
        (a[0], b[1], c[2], b[2], c[1]),
        (a[1], b[2], c[0], b[0], c[2]),
        (a[2], b[0], c[1], b[1], c[0]),
    ] {
        let (p1, e1) = two_product(bj, ck);
        let (p2, e2) = two_product(bk, cj);
        for m in [p1, e1, -p2, -e2] {
            let (q, f) = two_product(ai, m);
            terms.push(q);
            terms.push(f);
        }
    }
    exact_sum_sign(&terms)
}

/// Exact sign of the 2×2 minor `w·x − y·z` as `-1 | 0 | +1` (for the SoS
/// tie-break sequence, whose minors need the same exactness as the
/// determinant itself).
fn minor_exact_sign(w: f64, x: f64, y: f64, z: f64) -> i32 {
    let (p1, e1) = two_product(w, x);
    let (p2, e2) = two_product(y, z);
    exact_sum_sign(&[p1, e1, -p2, -e2])
}

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
    // Decide the geometric sign ONCE, on the canonical order — and decide it
    // **exactly** ([`orient_exact_sign`]): a near-zero f64 determinant carries
    // a noise sign that is stable per-triple but jointly inconsistent across
    // triples (issue #103), so only the exact sign keeps the predicate axioms.
    let canon = match orient_exact_sign(p, q, r) {
        0 => sos_sorted_sign(p, q, r), // true degeneracy: symbolic perturbation
        sign => sign,
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
    // Each minor's sign is decided exactly ([`minor_exact_sign`]): the tie-
    // break sequence needs the same joint consistency as the determinant, and
    // a rounded minor near zero would reintroduce the very noise SoS exists
    // to remove.  Evaluated lazily — the first minor usually decides.
    let minors: [(i32, (f64, f64, f64, f64)); 9] = [
        (1, (q[0], r[1], q[1], r[0])),
        (-1, (q[0], r[2], q[2], r[0])),
        (1, (q[1], r[2], q[2], r[1])),
        (-1, (p[0], r[1], p[1], r[0])),
        (1, (p[0], r[2], p[2], r[0])),
        (-1, (p[1], r[2], p[2], r[1])),
        (1, (p[0], q[1], p[1], q[0])),
        (-1, (p[0], q[2], p[2], q[0])),
        (1, (p[1], q[2], p[2], q[1])),
    ];
    for (sgn, (w, x, y, z)) in minors {
        let s = minor_exact_sign(w, x, y, z);
        if s != 0 {
            return sgn * s;
        }
    }
    1
}

/// Uniform symbolic minor-arc crossing, decided purely by [`orient_sos`] signs
/// on the **input** points (issue #103).
///
/// Arcs `a → b` and `c → d` — each **minor** (< 180°, which holds for HEALPix
/// probe arcs and polygon edges between consecutive vertices) — cross at a
/// point interior to both iff the four orientations `[a c b]`, `[c b d]`,
/// `[b d a]`, `[d a c]` share one sign (the S2 `SimpleCrossing` identity; the
/// four signs jointly encode which antipodal intersection the straddle refers
/// to, so no constructed intersection point is needed).
///
/// This replaces the retired two-stage `robust_crossing` pipeline
/// (straddle gates + float-constructed intersection + `on_minor_arc`): that
/// pipeline resolved the same degeneracy in three different implicit ways, and
/// issue #103 showed they can disagree by one crossing — a vertex graze counted
/// by luck through a bit-exact `coincident` hit on one edge and dropped on the
/// other because the deciding wedge determinant rounded to `+8e-20` (nonzero,
/// so the SoS tie-break never engaged).  Here every sidedness question goes
/// through [`orient_sos`]'s canonical, identity-keyed evaluation, so the same
/// physical point resolves to the same side in **every** test that consults it
/// (e.g. the shared vertex of two incident edges appears as `[p v q]` in one
/// edge's test and `[q v p]` in the other — exact negations by construction).
/// A probe passing exactly through a ring vertex therefore counts **exactly
/// one** crossing across the two incident edges when the boundary passes
/// through the probe circle (and zero or two when it grazes), keeping the
/// even-odd fill parity consistent: the half-open `[a, b)` convention emerges
/// instead of being hand-maintained.  Total and reorder-invariant **provided
/// the four SoS identities are pairwise distinct** — a duplicated id makes the
/// symbolic perturbation ill-defined and voids the invariance; every call site
/// (probe ids, vertex ids, corner ids) draws from disjoint ranges.  `ia, ib,
/// ic, id` are the identities of the four endpoints.
#[inline]
#[allow(clippy::too_many_arguments)] // 4 points + 4 SoS ids, same shape as robust_crossing
pub fn arcs_cross_sos(
    a: &Vec3,
    b: &Vec3,
    c: &Vec3,
    d: &Vec3,
    ia: PointId,
    ib: PointId,
    ic: PointId,
    id: PointId,
) -> bool {
    let acb = orient_sos(a, c, b, ia, ic, ib);
    let cbd = orient_sos(c, b, d, ic, ib, id);
    if acb != cbd {
        return false;
    }
    let bda = orient_sos(b, d, a, ib, id, ia);
    if cbd != bda {
        return false;
    }
    let dac = orient_sos(d, a, c, id, ia, ic);
    bda == dac
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
/// The companion SoS predicates [`orient_sos`] and [`arcs_cross_sos`] are the
/// orientation-only building blocks the descent's per-cell parity flips use; an
/// *edge-crossing* PIP built on them is deferred while its long-arc /
/// scalloped-boundary behaviour is validated against this winding reference.
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

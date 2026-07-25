//! Spherical primitives for the hierarchical region coverer.
//!
//! Everything here operates on **unit 3-vectors** on the sphere.  The two core
//! predicates are [`orient`] (the sign of a scalar triple product) and
//! [`arcs_cross`] (do two great-circle arcs cross?), built from it.  On top of
//! those sits the single point-in-polygon path — [`point_in_ring_robust`]
//! (spherical winding number where that is definitive, edge-crossing parity
//! from an anchor where it is not; correct at any polygon size including
//! hemisphere+, issues #22/#107) — plus [`parity_filled_robust`], the even-odd rule
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
//   1. The point-in-ring decision [`point_in_ring_robust`] reads the signed
//      spherical **subtended-angle sum** ([`ring_winding_at`], Bevis & Chatelain
//      1989) where that sum is *definitive* (`|sum| ≈ 2π`), and falls back to
//      **edge-crossing parity from an anchor** where it is not (issue #107).  The
//      sum alone is not the winding indicator: it is antisymmetric under
//      `x → −x`, so it reports `2π·[k(x) − k(−x)]` and cancels to `0` for a point
//      whose antipode is also interior — see [`ring_winding_at`].  It runs only
//      at the 12 base-cell seeds, so its per-vertex trig is off the hot path.
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
// The *edge-crossing* PIP built on layer 2 is no longer deferred: issue #107
// showed the subtended-angle sum cannot decide the antipodal-lens case at all,
// so [`point_in_ring_robust`] now classifies those points as `anchor_fill XOR
// crossing_parity(anchor → x)` over layer 2's [`arcs_cross_sos`].  Only the
// degenerate class pays for it; the definitive-magnitude fast path keeps the
// clean case at one angle sum.

/// Stable identity of a point feeding [`orient_sos`], used by its Simulation-of-
/// Simplicity tie-break.  For ring vertices this is the vertex index; the
/// symbolic perturbation is a strictly increasing function of it, so identities
/// only need to be **distinct and consistently ordered**, not contiguous.
pub type PointId = u64;

// ── SoS identity allocation ──────────────────────────────────────────────
//
// [`arcs_cross_sos`] is total and reorder-invariant **only when its four point
// identities are pairwise distinct**, so every endpoint that can reach it needs
// an identity from a range no other call site can hit.  The crate-wide
// allocation, lowest first (low ids perturb first, so the *polygon* is nudged
// off a degenerate lattice before any probe point is):
//
// | range                             | points                                          |
// |-----------------------------------|-------------------------------------------------|
// | [`RING_VERTEX_ID_BASE`] `..`      | ring vertices, by global index across a ring-set |
// | `(1 << 63) ..`                    | HEALPix cell centres (`crate::coverage::center_id`) |
// | [`PROBE_ID`] (`MAX - 3`)          | the synthesized test point of the bare-geometry PIP |
// | [`ANCHOR_ID`] (`MAX - 2`)         | the crossing-PIP anchor                          |
// | `MAX - 1`, `MAX`                  | `crate::coverage`'s `CORNER_ID_B` / `CORNER_ID_A` |
//
// Centre ids top out around `(1 << 63) + (4 << 58)` ≈ `1.04e19`, well clear of
// `MAX - 3` ≈ `1.84e19`, so the four ranges cannot overlap.

/// First identity of a ring-set's vertices; vertex `i` (counted globally across
/// the ring-set, as `crate::coverage`'s `build_edges` counts them) is
/// `RING_VERTEX_ID_BASE + i`.  Deliberately the same value as that module's
/// `VERTEX_ID_BASE` so a ring vertex carries one identity whichever layer tests
/// it.
pub const RING_VERTEX_ID_BASE: PointId = 2;

/// Identity the bare-geometry [`point_in_ring_robust`] / [`parity_filled_robust`]
/// wrappers synthesize for the test point.  A caller that holds a *stable*
/// identity for the point (a cell centre, say) should pass it instead — see
/// [`point_in_ring_robust`]'s note on why the synthesized id is only
/// self-consistent within one call.
pub const PROBE_ID: PointId = PointId::MAX - 3;

/// Identity of the crossing-PIP anchor ([`ring_inside_anchor`]).  One anchor is
/// in flight at a time and it is never a ring vertex, so a single reserved id
/// suffices.
pub const ANCHOR_ID: PointId = PointId::MAX - 2;

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

/// Signed spherical subtended-angle sum of `ring` as seen from `x`: the sum of
/// the signed angles each directed edge subtends at `x` (Bevis & Chatelain
/// 1989).
///
/// # What it actually measures (issue #107)
///
/// It is **not** the winding indicator.  Writing `k(x) = 1` when `x` is in the
/// ring's counter-clockwise interior and `0` otherwise, this function returns
///
/// ```text
///     w(x) = 2π · [ k(x) − k(−x) ]
/// ```
///
/// — it is antisymmetric under `x → −x`.  The proof is three lines of the code
/// below: `da = dot(a, x)` is *odd* in `x`, so the product `da * x` is **even**,
/// so `pa`/`pb` and therefore `ang = acos(pa·pb) ≥ 0` are *invariant* under
/// `x → −x`; only `sgn` flips, so every term negates.
///
/// The consequences, and how [`point_in_ring_robust`] uses them:
///
/// * `w > π` (i.e. `≈ +2π`) ⇒ `x` **inside** and `−x` outside.  Definitive.
/// * `w < −π` (i.e. `≈ −2π`) ⇒ `x` **outside** and `−x` inside.  Definitive.
/// * `|w| ≤ π` (i.e. `≈ 0`) ⇒ `x` and `−x` are on the **same** side, and the sum
///   cancels without saying which.  *Not* a verdict of "outside" — reading it as
///   one is the defect this documents: a hemisphere+ interior contains antipodal
///   pairs, and every such interior point read outside.  A sub-hemisphere
///   interior cannot contain an antipodal pair, so `w` is definitive everywhere
///   for those rings and the whole pre-#107 behaviour was sound there.
///
/// A point *on* the boundary lands in none of the three cases cleanly: its
/// on-edge term is `±π` with the sign decided by rounding, so `w` can be any of
/// `0`, `±π`, `±2π`.  Boundary points are therefore also routed to the anchor
/// construction, which is symbolic and total.
///
/// Needs no reference point and no minor-arc precondition; runs only at the
/// base-cell seeds, so its per-vertex trig is not on a hot path.
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

/// How many ring edges the anchor search may sample.
///
/// Each sampled edge offers its midpoint's **left** flank as a candidate and
/// its right flank as the witness that can prove that candidate
/// ([`ring_flanks`]); a candidate is taken only once a proof places it inside,
/// never on the strength of the construction alone.  The bound on the whole
/// search is `O(SAMPLES × V)`, paid once per ring ([`RingAnchors`]) rather than
/// once per probe — and in the ordinary case the first edge already yields a
/// proved anchor, so the real cost is `O(V)`.
const ANCHOR_EDGE_SAMPLES: usize = 8;

/// Tolerance for resolving an angle sum to the nearest whole turn.
///
/// Both quantities [`ring_inside_anchor`] rounds this way — a candidate's own
/// sum, and the gap between a flank pair — are exact integer multiples of `2π`
/// in exact arithmetic, since `w = 2π·[W(x) − W(−x)]` with both terms integers
/// ([`ring_winding_at`]).  A quarter-turn window therefore cannot admit a
/// neighbouring multiple, while sitting far above the sum's accumulated error
/// (`~V × 1e-16`, i.e. `~1e-12` even for a 22 k-vertex basin).
const ANCHOR_TURN_TOL: f64 = std::f64::consts::FRAC_PI_2;

/// Angle between two unit vectors.
///
/// `atan2` of the cross-product magnitude against the dot product, which stays
/// accurate where `dot(a, b).acos()` collapses: below ~`2e-8` rad the dot
/// product rounds to exactly `1.0` and `acos` returns exactly `0.0`.  That
/// underflow was issue #107's 6 mm-edge regression — a zero-length anchor step
/// left the anchor sitting *on* the boundary.
fn angle_between(a: &Vec3, b: &Vec3) -> f64 {
    norm(&cross(a, b)).atan2(dot(a, b))
}

/// Angular distance from `p` to the arc `u → v`: the perpendicular foot when it
/// falls within the segment, otherwise the nearer endpoint.
fn arc_distance(p: &Vec3, u: &Vec3, v: &Vec3) -> f64 {
    let ends = angle_between(p, u).min(angle_between(p, v));
    let n = cross(u, v);
    if norm(&n) < 1e-15 {
        return ends; // degenerate edge: only its endpoints are defined
    }
    let n = normalize(&n);
    let d = dot(p, &n);
    let f = [p[0] - d * n[0], p[1] - d * n[1], p[2] - d * n[2]];
    if norm(&f) < 1e-15 {
        return ends; // p is the edge's pole: every foot is equidistant
    }
    let f = normalize(&f);
    if dot(&cross(u, &f), &n) >= 0.0 && dot(&cross(&f, v), &n) >= 0.0 {
        return d.abs().clamp(0.0, 1.0).asin();
    }
    ends
}

/// The two points flanking edge `i`'s midpoint, stepped off the edge's great
/// circle by **half the distance to the nearest other strand of the ring**.
/// `left` is the `a × b` side — the interior side under the counter-clockwise
/// winding contract of [`point_in_ring_robust`].
///
/// # Invariant
///
/// Let `d = min over j ≠ i of arc_distance(mid, edge_j)`.  By construction the
/// ball `B(mid, d)` meets the boundary only in edge `i`, and `d ≤ ρ`, the
/// edge's own half-length — the adjacent edge shares the endpoint `a`, which
/// sits exactly `ρ` from `mid`, so the minimum cannot exceed it.  Edge `i`
/// therefore crosses `B` end to end and cuts it into exactly two components,
/// and a step of `d/2 < d` lands one flank in each.  Two consequences, both
/// load-bearing for [`ring_inside_anchor`]:
///
/// 1. Both flanks are strictly **off** the boundary.
/// 2. `W(left) = W(right) + 1` **exactly** — they are separated by one directed
///    edge crossing and nothing else.
///
/// # What (2) does *not* say about the angle sum
///
/// The sums are antisymmetric, `w(x) = 2π·[W(x) − W(−x)]`, so (2) gives
///
/// ```text
/// w(left) − w(right) = 2π·[1 − (W(−left) − W(−right))]
/// ```
///
/// which collapses to the `2π` gap [`ring_inside_anchor`]'s witness proof tests
/// for **only when `W(−left) = W(−right)`** — when the ring does not separate
/// the two flanks' *antipodes*.  That precondition is not free: for a ring of
/// angular radius near `90°` a flank's antipode falls on the far side of the
/// boundary and the gap measures `4π` instead (a 24-gon at radius `89°` steps
/// `3.75°`, putting `−right` at colatitude `87.3°` — inside the `89°` cap).
/// The witness proof therefore fails *closed* on such rings; they are decided
/// by pass 1, which needs no gap.
///
/// This bound is what #107's first attempt got wrong: it capped the step at a
/// fixed `1e-6` rad and otherwise scaled it by the edge's *own* length, which
/// says nothing about how close another strand runs.  Any feature thinner than
/// ~6.4 m was stepped clean across, inverting the cover.
///
/// # Returns
///
/// `None` when edge `i` is degenerate (equal or antipodal endpoints, hence no
/// great circle) or when `d` is zero (another strand touches the midpoint).
/// Either way the edge offers no room and the caller samples another.
fn ring_flanks(ring: &[Vec3], i: usize) -> Option<(Vec3, Vec3)> {
    let n = ring.len();
    let (a, b) = (&ring[i], &ring[(i + 1) % n]);
    let normal = cross(a, b);
    if norm(&normal) < 1e-15 {
        return None;
    }
    let normal = normalize(&normal);
    let mid = normalize(&[a[0] + b[0], a[1] + b[1], a[2] + b[2]]);
    let off = 0.5
        * (0..n)
            .filter(|&j| j != i)
            .map(|j| arc_distance(&mid, &ring[j], &ring[(j + 1) % n]))
            .fold(f64::INFINITY, f64::min);
    if !off.is_finite() || off <= 0.0 {
        return None; // another strand touches the midpoint: no room to step
    }
    let (co, so) = (off.cos(), off.sin());
    let step = |s: f64| {
        normalize(&[
            mid[0] * co + s * normal[0] * so,
            mid[1] * co + s * normal[1] * so,
            mid[2] * co + s * normal[2] * so,
        ])
    };
    Some((step(1.0), step(-1.0)))
}

/// An anchor **proved** to lie inside `ring`, for the crossing-based
/// point-in-ring path.
///
/// Candidates come from [`ring_flanks`] over [`ANCHOR_EDGE_SAMPLES`] sampled
/// edges — `O(1)` generation, no search over the `O(V²)` pairs it would take to
/// hunt for a point with `|w| ≈ 2π` directly.  But the construction is only a
/// *generator*: nothing about "step to the left of an edge" survives a
/// self-intersecting ring, so every candidate must clear a proof before it is
/// returned, and a candidate that fails is discarded and the next one tried.
///
/// # Pass 1 — self-proof
///
/// A candidate whose own angle sum clears layer 1's threshold (`w > π`) is
/// inside *by the very predicate layer 1 decides with*, so accepting it assumes
/// nothing whatsoever about the step.
///
/// Among those, prefer the sum closest to a single turn.  `w = 2π·[W(x) −
/// W(−x)]` ([`ring_winding_at`]), so `w ≈ 2π` is a winding difference of exactly
/// one — **odd**, hence in the same even-odd class that layer 2's crossing
/// parity counts in.  The preference is not cosmetic: 54 of the 72 left flanks
/// of the polar comb in `mortie/tests/test_geometry.py` sit at `w ≈ 4π`, which
/// layer 1 calls inside but which is an *even* winding difference, i.e. the
/// exterior parity class.  Anchoring there labels the whole exterior inside —
/// the 24 → 3072 blow-up seen on #107's first attempt.
///
/// A candidate already at a single turn is optimal, so the scan returns on the
/// first one and the ordinary ring never pays for a second edge.  Only the left
/// flank is a candidate: the interior lies to the left of the directed edges by
/// contract, and that holds under either winding — reverse a simple ring and
/// the region left of its edges becomes the complementary one, which is
/// precisely the interior layer 1 then reports.
///
/// # Pass 2 — witness proof
///
/// If no candidate is layer-1-definitive, that is the antipodal-lens signature
/// itself: the interior flank *and its antipode* are both interior, so the sum
/// cancels to `≈ 0` and layer 1 is blind (issue #107).  The **opposite** flank
/// is generally not in the lens, though — for the PR #112 wobbly ring all 96
/// right flanks read a clean `−2π` — and layer 1 is definitive there.
///
/// So: a right flank that layer 1 calls definitively **outside** (`w < −π`),
/// together with the one-crossing gap `w(l) − w(r) ≈ 2π` that [`ring_flanks`]
/// guarantees, proves the left flank inside without the angle sum ever ruling
/// on it directly.  This is the positive check that replaces #107's first
/// attempt at a `w < −1.5π` veto, which could not fire in the regime that
/// needed it: a wrong-side flank of a sub-hemisphere ring reads `w ≈ 0`, nowhere
/// near `−1.5π`.
///
/// # No-anchor policy
///
/// `None` when neither pass proves a candidate — every sampled edge degenerate,
/// or no candidate provable.  Callers then degrade to the pre-#107 behaviour
/// exactly: [`point_in_ring_with`] reports the (already layer-1-ambiguous)
/// point outside, and [`ring_winding_sign_at`] reports `0`, "undecidable, do
/// not normalize".  Declining beats guessing, and it is never *worse* than
/// `main`.
fn ring_inside_anchor(ring: &[Vec3]) -> Option<Vec3> {
    let n = ring.len();
    if n < 3 {
        return None;
    }
    let (pi, tau) = (std::f64::consts::PI, std::f64::consts::TAU);
    let mut pairs: Vec<(Vec3, Vec3)> = Vec::new();
    let mut best: Option<(Vec3, f64)> = None;

    for i in (0..n).step_by((n / ANCHOR_EDGE_SAMPLES).max(1)) {
        let Some((l, r)) = ring_flanks(ring, i) else {
            continue;
        };
        let wl = ring_winding_at(&l, ring);
        // A single turn is the ideal and no later candidate can improve on it,
        // so stop here.  This is the fast path in every ordinary case, and it
        // is why the search costs ~2 sweeps of the ring rather than 24: only
        // this one edge's flanks and one angle sum get built.
        if (wl - tau).abs() < ANCHOR_TURN_TOL {
            return Some(l);
        }
        let better = match best {
            None => true,
            Some((_, bw)) => (wl - tau).abs() < (bw - tau).abs(),
        };
        if wl > pi && better {
            best = Some((l, wl));
        }
        pairs.push((l, r));
    }
    if let Some((c, _)) = best {
        return Some(c);
    }

    // Pass 2.  The right flank's own sum is only built here, on the rare path.
    pairs.iter().find_map(|&(l, r)| {
        let (wl, wr) = (ring_winding_at(&l, ring), ring_winding_at(&r, ring));
        // `wl > −π` rejects a pair sitting a whole turn lower (`wr ≈ −4π`,
        // `wl ≈ −2π`), where the gap still holds but layer 1 calls the left
        // flank outright outside.
        let proved = wr < -pi && wl > -pi && (wl - wr - tau).abs() < ANCHOR_TURN_TOL;
        proved.then_some(l)
    })
}

/// Per-ring anchors for a ring-set: computed at most **once** each, and only if
/// some probe actually reaches layer 2.
///
/// [`ring_inside_anchor`] costs `O(ANCHOR_EDGE_SAMPLES × V)`, so recomputing it
/// inside every probe made the layer-2 path `O(V)` per cell centre across a
/// whole descent.  `crate::coverage` builds this once per descent and threads it
/// into [`parity_filled_with`] instead.
///
/// The cell is lazy so that a ring-set pays only for the rings some probe
/// actually reaches layer 2 on — for multipart geometry that is often a
/// minority of them.  [`std::sync::OnceLock`] keeps that sound under the
/// rayon-parallel descent, which shares one `RingAnchors` across threads.
pub struct RingAnchors(Vec<std::sync::OnceLock<Option<Vec3>>>);

impl RingAnchors {
    /// One empty slot per ring; nothing is computed until [`Self::get`] asks.
    pub fn of_rings(rings: &[Vec<Vec3>]) -> Self {
        RingAnchors(rings.iter().map(|_| std::sync::OnceLock::new()).collect())
    }

    /// The anchor of ring `i`, computing it on first use.  `ring` must be the
    /// `i`-th ring of the set this was built from.
    fn get(&self, i: usize, ring: &[Vec3]) -> Option<Vec3> {
        *self.0[i].get_or_init(|| ring_inside_anchor(ring))
    }
}

/// Parity of the boundary crossings of the minor arc `a → x` against `ring`,
/// counted with the exact symbolic predicate [`arcs_cross_sos`].
///
/// `true` means an odd number of crossings, i.e. `a` and `x` are on **opposite**
/// sides of the ring (the Jordan-curve argument on the sphere).  Vertex ids run
/// `vid_base + i`; the arc endpoints carry the caller-supplied `ia` and `ix`, so
/// the four identities of every call are pairwise distinct as `arcs_cross_sos`
/// requires.  A zero-length edge (duplicate consecutive vertices) traces no
/// boundary and is skipped, matching `crate::coverage`'s `build_edges`, which
/// keeps vertex ids positional across the skip.
fn ring_crossing_parity(
    a: &Vec3,
    x: &Vec3,
    ia: PointId,
    ix: PointId,
    ring: &[Vec3],
    vid_base: PointId,
) -> bool {
    let n = ring.len();
    let mut crossings = 0usize;
    for i in 0..n {
        let j = (i + 1) % n;
        let (u, v) = (&ring[i], &ring[j]);
        if (u[0] - v[0]).abs() < 1e-12 && (u[1] - v[1]).abs() < 1e-12 && (u[2] - v[2]).abs() < 1e-12
        {
            continue;
        }
        if arcs_cross_sos(
            a,
            x,
            u,
            v,
            ia,
            ix,
            vid_base + i as PointId,
            vid_base + j as PointId,
        ) {
            crossings += 1;
        }
    }
    crossings % 2 == 1
}

/// [`point_in_ring_robust`] with the test point's SoS identity `ix`, the ring's
/// vertex-id base, and the ring's precomputed `anchor` supplied by the caller.
///
/// Two layers, cheapest first:
///
/// 1. **Definitive angle sum.**  `|ring_winding_at(p, ring)| > π` decides `p`
///    outright ([`ring_winding_at`]).  This is every point of a sub-hemisphere
///    ring and every point of a hemisphere+ ring outside the antipodal lens, so
///    the common case costs exactly what it cost before #107.
/// 2. **Crossing parity from an anchor.**  Otherwise `p` is in the ambiguous
///    class (`p` and `−p` on the same side, or `p` on the boundary).  Take the
///    proved-inside anchor ([`ring_inside_anchor`]) and return `NOT
///    crossing_parity(anchor → p)` ([`ring_crossing_parity`]).
///
/// Layer 1 is bit-for-bit the pre-#107 decision, so the repair only fills in the
/// class the old test could not decide — no previously-definitive answer moves.
///
/// # On the anchor→`p` arc
///
/// #107's first attempt claimed this is always a well-defined minor arc because
/// `|w(anchor)| > π` would have routed `p` through layer 1.  That argument is
/// **false in exactly this branch**: the witness proof of
/// [`ring_inside_anchor`] exists precisely so a lens anchor with `w ≈ 0` can be
/// used, and then `w(anchor)` says nothing about `p`.  What actually holds is
/// weaker but sufficient: crossing parity along *any* path decides the even-odd
/// class, the minor arc is such a path for any non-antipodal pair, both
/// endpoints are strictly off the boundary ([`ring_flanks`]), and
/// [`arcs_cross_sos`] is total under SoS for distinct identities — so the
/// measure-zero coincidence of `p` equal or antipodal to the anchor still
/// returns a consistent verdict rather than being undefined.
/// `anchor` is a thunk: layer 1 decides most probes on its own, and for a
/// sub-hemisphere ring it decides *every* probe, so the anchor must not be
/// built unless this call actually reaches layer 2.
fn point_in_ring_with(
    p: &Vec3,
    ix: PointId,
    ring: &[Vec3],
    vid_base: PointId,
    anchor: impl FnOnce() -> Option<Vec3>,
) -> bool {
    if ring.len() < 3 {
        return false;
    }
    let w = ring_winding_at(p, ring);
    if w > std::f64::consts::PI {
        return true;
    }
    if w < -std::f64::consts::PI {
        return false;
    }
    match anchor() {
        // The anchor is proved inside, so `p` is inside iff the arc between
        // them crosses the boundary an even number of times.
        Some(a) => !ring_crossing_parity(&a, p, ANCHOR_ID, ix, ring, vid_base),
        None => false, // see `ring_inside_anchor`'s no-anchor policy
    }
}

/// [`point_in_ring_with`] computing the ring's anchor itself.  For a caller
/// that tests many points against the same ring-set this is the wrong entry
/// point — build a [`RingAnchors`] once and use [`parity_filled_with`].
fn point_in_ring_ids(p: &Vec3, ix: PointId, ring: &[Vec3], vid_base: PointId) -> bool {
    point_in_ring_with(p, ix, ring, vid_base, || ring_inside_anchor(ring))
}

/// Robust spherical point-in-ring test, valid at **any** ring size.
///
/// `p` is inside iff it is in the ring's counter-clockwise interior (the region
/// to the left of the directed edges, the same convention the even-odd fill
/// assumes).  There is no projection centre to go singular and no sub-hemisphere
/// precondition, so this is correct for hemisphere+ rings such as "everything
/// except Antarctica" (#22) and degeneracy-free when an edge's great circle
/// passes through a HEALPix cell centre (#11).
///
/// The decision is the two-layer construction of [`point_in_ring_ids`]: the
/// subtended-angle sum where it is definitive, edge-crossing parity from an
/// anchor where it is not (issue #107).  This wrapper synthesizes the SoS
/// identities the crossing layer needs — [`PROBE_ID`] for `p` and
/// [`RING_VERTEX_ID_BASE`] for the vertices.  Those are only **self-consistent
/// within one call**: a caller that holds a stable identity for `p` (a cell
/// centre carrying `crate::coverage`'s `center_id`, say) gets a verdict that
/// agrees with the descent's own probes at exact boundary coincidences, where a
/// synthesized id can legitimately perturb the other way.  Threading real
/// identities through to here is phase 3 of #107.
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
/// orientation-only building blocks the descent's per-cell parity flips use, and
/// layer 2 of this test now consumes them directly.
pub fn point_in_ring_robust(p: &Vec3, ring: &[Vec3]) -> bool {
    point_in_ring_ids(p, PROBE_ID, ring, RING_VERTEX_ID_BASE)
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
///
/// Vertex identities are numbered globally across the ring-set from
/// [`RING_VERTEX_ID_BASE`], advancing by `ring.len()` per ring exactly as
/// `crate::coverage`'s `build_edges` does, so a vertex has the same id here and
/// in the descent.  See [`point_in_ring_robust`] on the synthesized test-point
/// id.
pub fn parity_filled_robust(p: &Vec3, rings: &[Vec<Vec3>]) -> bool {
    parity_filled_with(p, rings, &RingAnchors::of_rings(rings))
}

/// [`parity_filled_robust`] against anchors the caller has already computed.
///
/// `anchors` must come from `RingAnchors::of_rings(rings)` for the *same*
/// ring-set; it is positional.  This is the entry point for a descent, which
/// probes thousands of cell centres against one ring-set and must not rebuild
/// the anchors each time.
pub fn parity_filled_with(p: &Vec3, rings: &[Vec<Vec3>], anchors: &RingAnchors) -> bool {
    let mut inside = false;
    let mut vid = RING_VERTEX_ID_BASE;
    for (i, ring) in rings.iter().enumerate() {
        if point_in_ring_with(p, PROBE_ID, ring, vid, || anchors.get(i, ring)) {
            inside = !inside;
        }
        vid += ring.len() as PointId;
    }
    inside
}

/// Signed winding direction of a sub-hemisphere `ring`: `+1` if it is wound
/// counter-clockwise (interior — the smaller side — to the **left** of the
/// directed edges, the RFC 7946 / S2 convention), `-1` if clockwise, `0` if the
/// ring is degenerate (fewer than three vertices, or a balanced vertex sum that
/// means the ring is not sub-hemisphere at all).
///
/// This is only meaningful for a ring that fits within a hemisphere, where the
/// two regions the ring bounds are unambiguously "small" and "large" and the
/// small side is the intended interior.  Used by [`crate::coverage`] to
/// auto-correct everyday CW input; it must **not** be used to "normalize" a
/// hemisphere+ ring, where area alone cannot pick the interior side (#22).
///
/// # Why this no longer reads the angle sum at the cap axis (issue #107)
///
/// The pre-#107 test read [`ring_winding_at`] at the cap axis directly and
/// called the ring CCW on `+2π`, CW on `−2π`, undecidable otherwise.  That is
/// sound only when the axis is inside the small side — true for a convex ring,
/// **false for a non-convex one**, whose normalized vertex sum can fall in the
/// large region.  Then the axis and its antipode are on the *same* side, the sum
/// cancels to `0` (see [`ring_winding_at`]), the test reports "undecidable", and
/// [`crate::coverage`]'s ingest silently declines to normalize — leaving a
/// clockwise ring clockwise and selecting the complementary region.  The
/// Antarctic drainage basins in `mortie/tests/` are exactly this shape (their
/// vertex sum lands in the basin's concavity); the miss was invisible before
/// only because the equally broken point-in-ring test read those un-normalized
/// rings as their small side anyway.
///
/// The replacement asks the same question with crossings alone.  The antipode of
/// the cap axis is *provably* in the large region for any sub-hemisphere ring —
/// the whole boundary lies within a cap of radius `< 90°` about the axis, so the
/// antipode is at least `90°` from every vertex — and
/// [`ring_inside_anchor`] gives a point that is interior by definition.  If a
/// boundary-crossing walk between the two finds them on **opposite** sides the
/// interior is the small side (CCW, `+1`); on the same side it is the large one
/// (CW, `-1`).  No angle sum enters, so neither the antipodal-lens defect nor
/// the winding-number misreading can reach this decision.
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
/// recompute the vertex sum.  `axis` must be the unit normalized vertex sum of a
/// sub-hemisphere ring; callers without it use [`ring_winding_sign`].
pub fn ring_winding_sign_at(ring: &[Vec3], axis: &Vec3) -> i32 {
    if ring.len() < 3 {
        return 0;
    }
    let Some(anchor) = ring_inside_anchor(ring) else {
        return 0; // no usable left side — caller must not normalize
    };
    let antipode = [-axis[0], -axis[1], -axis[2]];
    // Crossing parity directly, *not* [`point_in_ring_robust`]: that would take
    // its layer-1 shortcut on the angle sum, and the sum's sign at a point this
    // far from the boundary is a winding-number difference rather than an
    // inside/outside flag (see [`ring_inside_anchor`]).
    if ring_crossing_parity(
        &anchor,
        &antipode,
        ANCHOR_ID,
        PROBE_ID,
        ring,
        RING_VERTEX_ID_BASE,
    ) {
        1 // opposite sides ⇒ the far region is exterior ⇒ interior is the small side
    } else {
        -1 // same side ⇒ the interior is the large region ⇒ clockwise
    }
}

// ── tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;

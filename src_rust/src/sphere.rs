//! Spherical primitives for the hierarchical region coverer.
//!
//! Everything here operates on **unit 3-vectors** on the sphere.  The core
//! predicate is [`orient`] (the sign of a scalar triple product); [`orient_sos`]
//! makes it total with an exact determinant plus a Simulation-of-Simplicity
//! tie-break, and [`arcs_cross_sos`] builds the great-circle-segment crossing
//! test on top of it (issues #11, #103).  On top of those sits the single
//! point-in-polygon path — [`point_in_ring_robust`], correct at any polygon size
//! including hemisphere+ (issues #22/#107) — plus [`parity_filled_robust`], the
//! even-odd rule over a *ring-set* that gives holes and multipart geometry for
//! free (see issue #30).
//!
//! There is deliberately **no plain float crossing test** here.  The obvious one
//! — "each arc straddles the other's great circle" — is not a crossing test on a
//! sphere: two great circles meet at an *antipodal pair*, and each arc can
//! straddle at the opposite member, so the predicate accepts disjoint arcs in
//! antipodal regions.  `arcs_cross`/`arcs_cross_n` carried that body under a doc
//! claiming the four-orientation test and were pruned with #107; the
//! four-orientation identity lives in [`arcs_cross_sos`], which is what every
//! caller uses.
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

/// Identity of the crossing walk's reference point ([`ring_witness`]).  One
/// witness is in flight at a time and it is never a ring vertex, so a single
/// reserved id suffices.
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

// ── the normalization constant (issue #107) ──────────────────────────────
//
// [`ring_winding_at`] returns `w(x) = 2π·[W(x) − W(−x)]` — a *difference* of
// winding numbers.  That is not a shortcoming of the formula; on a sphere it is
// all a ring defines by itself.  A closed curve cuts the sphere into regions of
// equal standing, so the winding number `W` is fixed only up to one global
// additive constant, and "inside" *is* the choice of that constant.  The
// right-hand rule (RFC 7946 §3.1.6 / S2) supplies it: the interior lies to the
// **left** of the directed edges.
//
// #107's first two attempts tried to recover the constant *locally*, by
// constructing a point just off an edge and reading `w` there.  That cannot
// work — `w` is constant-free, so no reading of it at a constructed point can
// carry the constant — and every failure found in review is that same failure
// wearing a different hat: an anchor left sitting on the boundary (an edge
// under ~2e-8 rad), an anchor stepped clean across the ring (a feature thinner
// than the step), an anchor at `w ≈ 4π` (a winding *difference* of two, read as
// "inside"), and an anchor on the wrong lobe of a self-intersecting ring (the
// flank gap `w(l) − w(r) = 2π` constrains the difference, never the constant).
//
// The constant is a *global* property of the ring, so it is computed globally,
// by [`ring_turning`].  Gauss–Bonnet on the unit sphere gives
//
//     area(region to the left of the edges) = 2π − turning
//
// for any closed piecewise-geodesic ring, so `sign(turning)` says whether the
// right-hand-rule interior is the small side or the big one.  It is a sum of
// per-vertex angles: no sampling, no step size, no reference point, and it is
// invariant both under rotating the vertex list and under subdividing edges —
// the two axes #107's stride was steered along.  S2 takes the same route, for
// the same reason: `S2Loop::GetCurvature`, with `IsNormalized() ⟺ curvature ≥
// −maxError`, exists because its surface-integral area is accurate only modulo
// `4π` and the curvature is what resolves the branch.
//
// With the constant in hand the *point* test needs no constructed point at all
// whenever the ring's vertices fit inside an open hemisphere about their
// normalized sum `A` — the test [`crate::coverage`] already applies at ingest.
// The whole ring then lies in a cap of radius `< 90°`; the complement of that
// cap is a connected curve-free region, so `W = 0` throughout it, and for any
// `x` with `dot(x, A) > 0` the antipode `−x` lies in it.  Hence `W(−x) = 0` and
//
//     W(x) = w(x) / 2π      exactly.
//
// Only a genuinely hemisphere-plus ring — vertices that fit in no cap about
// their own sum — still needs a reference point, and [`ring_witness`] takes one
// whose verdict `w` *proves*, rather than one the construction merely hopes for.

/// How many candidate edges [`ring_witness`] may try.
///
/// Only hemisphere-plus rings reach it; everything else is decided in closed
/// form. For a simple ring the first candidate always succeeds (see
/// [`ring_witness`]), so the bound binds only on self-intersecting
/// hemisphere-plus input, and the cost is `O(SAMPLES × V)` paid once per ring
/// ([`RingRefs`]) rather than once per probe.
const WITNESS_EDGE_SAMPLES: usize = 8;

/// Angle between two vectors, which need not be unit-length.
///
/// `atan2` of the cross-product magnitude against the dot product, which stays
/// accurate where `dot(a, b).acos()` collapses: below ~`2e-8` rad the dot
/// product rounds to exactly `1.0` and `acos` returns exactly `0.0`.  That
/// underflow was issue #107's 6 mm-edge regression — a zero-length flank step
/// left the reference point sitting *on* the boundary.
fn angle_between(a: &Vec3, b: &Vec3) -> f64 {
    norm(&cross(a, b)).atan2(dot(a, b))
}

/// Sum of the exterior (turn) angles of `ring` — its total geodesic curvature.
///
/// By Gauss–Bonnet on the unit sphere, a **simple** closed piecewise-geodesic
/// ring bounding the region `L` to the left of its directed edges satisfies
///
/// ```text
///     |L| = 2π − turning
/// ```
///
/// so `turning > 0` exactly when the right-hand-rule interior is the *smaller*
/// of the two regions the ring bounds, and `turning < 0` when it is the larger.
/// That sign is the ring's winding direction ([`ring_winding_sign`]), and it is
/// the normalization constant the rest of this module needs.
///
/// Three properties are what make it the right instrument, and they are exactly
/// the ones an edge-sampling test lacks: it is invariant under **rotating** the
/// vertex list, invariant under **subdividing** edges (adding a vertex along an
/// edge adds a zero turn), and it needs no reference point, no step size and no
/// tolerance beyond its own sign.  A ring with no net orientation — a balanced
/// figure-eight, whose lobes wind opposite ways — reads `≈ 0`, which is the
/// honest answer rather than a coin flip.
///
/// A repeated vertex leaves one incident edge without a great circle; its turn
/// is undefined and none is due, so it contributes zero rather than `NaN`.
fn ring_turning(ring: &[Vec3]) -> f64 {
    let n = ring.len();
    let mut total = 0.0;
    for i in 0..n {
        let a = &ring[(i + n - 1) % n];
        let b = &ring[i];
        let c = &ring[(i + 1) % n];
        let (n_ab, n_bc) = (cross(a, b), cross(b, c));
        if norm(&n_ab) < 1e-15 || norm(&n_bc) < 1e-15 {
            continue;
        }
        let ang = angle_between(&n_ab, &n_bc);
        total += if orient(a, b, c) >= 0.0 { ang } else { -ang };
    }
    total
}

/// The ring's bounding cap: its normalized vertex sum, and the smallest dot
/// product any vertex has with it.
///
/// `min_dot > 0` means every vertex lies strictly inside the open hemisphere
/// about the axis — and therefore so does every edge, an open hemisphere being
/// convex — so the ring is confined to a cap of radius `< 90°`.  `None` when
/// the vertex sum is balanced, which is itself a hemisphere-plus signature.
///
/// Deliberately the same axis and the same `min_dot > 0` test that
/// [`crate::coverage`]'s `normalize_ring_orientation` applies at ingest, so the
/// two paths never disagree about which rings are sub-hemisphere.
fn ring_cap(ring: &[Vec3]) -> Option<(Vec3, f64)> {
    let mut s = [0.0, 0.0, 0.0];
    for v in ring {
        s[0] += v[0];
        s[1] += v[1];
        s[2] += v[2];
    }
    if norm(&s) < 1e-12 {
        return None;
    }
    let axis = normalize(&s);
    let min_dot = ring
        .iter()
        .map(|v| dot(&axis, v))
        .fold(f64::INFINITY, f64::min);
    Some((axis, min_dot))
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
/// and a step of `d/2 < d` lands one flank in each.  So:
///
/// 1. Both flanks are strictly **off** the boundary.
/// 2. `W(left) = W(right) + 1` **exactly** — they are separated by one directed
///    edge crossing and nothing else.
///
/// The nearest-strand bound is not an optimization to skip.  #107's first
/// attempt capped the step at a fixed `1e-6` rad and otherwise scaled it by the
/// edge's *own* length, which says nothing about how close another strand runs;
/// any feature thinner than ~6.4 m was stepped clean across, inverting the
/// cover.  The `O(V)` sweep is affordable here because only hemisphere-plus
/// rings reach [`ring_witness`] at all, and its result is cached per ring
/// ([`RingRefs`]).
///
/// Note what (2) is *not*.  The sums are antisymmetric, so
/// `w(left) − w(right) = 2π·[1 − (W(−left) − W(−right))]`: the flank pair pins
/// a **difference** of winding numbers and never the constant, which is why
/// #107's "witness proof" — reading a `2π` gap as proof that the left flank is
/// interior — was unsound, and why the caller now takes only points `w` decides
/// outright.
///
/// # Returns
///
/// `None` when edge `i` is degenerate (equal or antipodal endpoints, hence no
/// great circle) or when `d` is zero (another strand touches the midpoint).
/// Either way the edge offers no room and the caller tries another.
fn ring_flanks(ring: &[Vec3], i: usize) -> Option<(Vec3, Vec3)> {
    let n = ring.len();
    let (a, b) = (&ring[i], &ring[(i + 1) % n]);
    let normal = cross(a, b);
    if norm(&normal) < 1e-15 {
        return None;
    }
    let normal = normalize(&normal);
    let mid = normalize(&[a[0] + b[0], a[1] + b[1], a[2] + b[2]]);
    let clearance = (0..n)
        .filter(|&j| j != i)
        .map(|j| arc_distance(&mid, &ring[j], &ring[(j + 1) % n]))
        .fold(f64::INFINITY, f64::min);
    let off = 0.5 * clearance;
    if !off.is_finite() || off <= 0.0 {
        return None; // no room to step
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

/// Everything a ring needs, computed once, to turn [`ring_winding_at`]'s
/// winding *difference* into an absolute inside/outside verdict.
#[derive(Clone, Copy, Debug)]
enum RingRef {
    /// The vertices fit strictly inside an open hemisphere about `axis`, so
    /// `W(x) = w(x)/2π` for every `x` on that side of the sphere and `W(x) = 0`
    /// on the other — no reference point needed.  `cw` records that the
    /// right-hand-rule interior is the *big* side, which shifts the whole
    /// winding function up by one.
    Cap { axis: Vec3, cw: bool },
    /// Hemisphere-plus: a point whose verdict the angle sum settled outright,
    /// carrying the constant for the crossing walk.
    Witness { point: Vec3, inside: bool },
    /// The ring traces no boundary — fewer than three vertices, or every edge
    /// degenerate — so it encloses nothing and contains nothing.  Distinct from
    /// [`RingRef::Undecidable`]: this is an answer, not the absence of one, and
    /// it is the reason a ring of coincident vertices cannot report its own
    /// vertex as interior off the back of a meaningless angle sum.
    Empty,
    /// Nothing could be established; the caller falls back to the bare sum,
    /// which is exactly the pre-#107 behaviour.
    Undecidable,
}

/// The reference [`point_in_ring_with`] decides against, computed once per ring.
fn ring_reference(ring: &[Vec3]) -> RingRef {
    let n = ring.len();
    if n < 3 || !(0..n).any(|i| norm(&cross(&ring[i], &ring[(i + 1) % n])) >= 1e-15) {
        return RingRef::Empty;
    }
    if let Some((axis, min_dot)) = ring_cap(ring) {
        if min_dot > 0.0 {
            // `winding_sign_in_cap`, not the raw sign of `ring_turning`: a ring
            // with no net orientation reads `±0` on numerical noise, and taking
            // that at face value would shift the whole winding function on the
            // strength of it — inverting the cover of a balanced figure-eight
            // depending on which lobe happened to carry more vertices.  Sharing
            // the banded test with `ring_winding_sign` is also what stops ingest
            // and this path from ever disagreeing about a ring's orientation.
            return RingRef::Cap {
                axis,
                cw: winding_sign_in_cap(ring, min_dot) < 0,
            };
        }
    }
    ring_witness(ring)
}

/// A point of a hemisphere-plus `ring` whose inside/outside verdict the angle
/// sum decides on its own.
///
/// # Why this is sound where #107's anchor was not
///
/// Nothing here assumes the construction landed on any particular side.  A
/// candidate is taken only when `|w| > π`, and *that* is already a verdict:
/// `w > π` means `W(x) > W(−x)`, which for a simple ring (`W ∈ {0, 1}`) forces
/// `W(x) = 1`, and `w < −π` forces `W(x) = 0`.  [`ring_flanks`] is demoted from
/// prover to generator — it only says *where to look*.
///
/// # Why looking succeeds
///
/// For a simple ring **every** edge yields a witness, so the first candidate
/// all but always settles it.  Take any edge's flank pair `(l, r)`: they lie in
/// the ring's two different regions, `W(l) = 1` and `W(r) = 0`.  If
/// `W(−l) = W(−r) = 0` then `w(l) = 2π` and `l` is the witness; if both are `1`
/// then `w(r) = −2π` and `r` is; if they differ, one of the two is again `±2π`.
/// Both flanks read `w ≈ 0` only when `−mid` lands exactly on the boundary — a
/// measure-zero coincidence the next candidate edge clears.
///
/// # Why the candidates are ordered by edge length
///
/// Longest edge first, ties by index.  That keeps the choice invariant under
/// rotating the vertex list, and — unlike a fixed index stride — it cannot be
/// steered by vertex **density**.  Crowding one part of a ring with vertices is
/// precisely how #107's `step_by(n / 8)` was made to sample only the clockwise
/// lobe of a self-intersecting ring, and every sampled candidate there agreed
/// on the wrong answer.
fn ring_witness(ring: &[Vec3]) -> RingRef {
    let n = ring.len();
    let pi = std::f64::consts::PI;
    // Longest edge first.  `dot` is monotone decreasing in arc length, so the
    // smallest dot is the longest edge and no trig is needed to rank them.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        let (di, dj) = (
            dot(&ring[i], &ring[(i + 1) % n]),
            dot(&ring[j], &ring[(j + 1) % n]),
        );
        di.partial_cmp(&dj)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(i.cmp(&j))
    });

    // A left flank the sum calls outright *outside*, or a right flank it calls
    // inside, can only happen on a self-intersecting ring.  Both are still
    // verdicts, so they are kept as a fallback, but the ordinary reading is
    // tried across every candidate first.
    let mut fallback: Option<(Vec3, bool)> = None;
    for &i in order.iter().take(WITNESS_EDGE_SAMPLES) {
        let Some((l, r)) = ring_flanks(ring, i) else {
            continue;
        };
        let wl = ring_winding_at(&l, ring);
        if wl > pi {
            return RingRef::Witness {
                point: l,
                inside: true,
            };
        }
        let wr = ring_winding_at(&r, ring);
        if wr < -pi {
            return RingRef::Witness {
                point: r,
                inside: false,
            };
        }
        if fallback.is_none() {
            if wl < -pi {
                fallback = Some((l, false));
            } else if wr > pi {
                fallback = Some((r, true));
            }
        }
    }
    match fallback {
        Some((point, inside)) => RingRef::Witness { point, inside },
        None => RingRef::Undecidable,
    }
}

/// Per-ring [`RingRef`]s for a ring-set: computed at most **once** each, on
/// first use.
///
/// [`ring_reference`] is `O(V)` for a sub-hemisphere ring and
/// `O(WITNESS_EDGE_SAMPLES × V)` for a hemisphere-plus one, so recomputing it
/// inside every probe would make the whole descent `O(V)` per cell centre.
/// `crate::coverage` builds this once per descent and threads it into
/// [`parity_filled_with`] instead.
///
/// [`std::sync::OnceLock`] keeps the laziness sound under the rayon-parallel
/// descent, which shares one `RingRefs` across threads.
pub struct RingRefs(Vec<std::sync::OnceLock<RingRef>>);

impl RingRefs {
    /// One empty slot per ring; nothing is computed until [`Self::get`] asks.
    pub fn of_rings(rings: &[Vec<Vec3>]) -> Self {
        RingRefs(rings.iter().map(|_| std::sync::OnceLock::new()).collect())
    }

    /// Record that ring `i` is sub-hemisphere about `axis` **and already
    /// normalized counter-clockwise**, so [`Self::get`] never has to derive it.
    ///
    /// This is the one thing [`crate::coverage`]'s ingest already knows and
    /// [`ring_reference`] would otherwise rediscover: `normalize_ring_orientation`
    /// computes the bounding cap and the turning angle to decide whether to
    /// reverse the ring, and after it has run the answer is fixed. A ring it
    /// left alone read `+1` or `0`; one it reversed read `-1`, and reversing
    /// negates every exterior angle, so the reversed ring reads `+1`. Either
    /// way `cw` is false, and seeding it here saves a second `O(V)` turning sum
    /// per ring per cover.
    ///
    /// `axis` must be the ring's normalized vertex sum. Seeding a ring that is
    /// *not* sub-hemisphere, or one wound clockwise, silently selects the wrong
    /// region — the only caller is ingest, immediately after establishing both.
    pub fn seed_normalized_cap(&mut self, i: usize, axis: Vec3) {
        let _ = self.0[i].set(RingRef::Cap { axis, cw: false });
    }

    /// The reference of ring `i`, computing it on first use.  `ring` must be
    /// the `i`-th ring of the set this was built from.
    fn get(&self, i: usize, ring: &[Vec3]) -> RingRef {
        *self.0[i].get_or_init(|| ring_reference(ring))
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
/// vertex-id base, and the ring's [`RingRef`] supplied by the caller.
///
/// One angle sum ([`ring_winding_at`]) plus whatever the ring's reference needs
/// to turn that sum's winding *difference* into a verdict:
///
/// * **[`RingRef::Cap`]** — vertices confined to an open hemisphere about
///   `axis`, so `W(p) = w/2π` where `dot(p, axis) > 0` and `W(p) = 0` elsewhere.
///   `p` is inside iff `W(p) ≥ 1`, or `W(p) ≥ 0` when the interior is the big
///   side.  Exact, and no crossing walk at all — this is the common case, and it
///   costs one dot product more than the pre-#107 path.
/// * **[`RingRef::Witness`]** — hemisphere-plus.  `|w| > π` still decides `p`
///   outright; otherwise walk from the witness, whose verdict is known, and flip
///   it on an odd crossing count ([`ring_crossing_parity`]).
/// * **[`RingRef::Undecidable`]** — degrade to the bare `w > π`, i.e. exactly
///   the pre-#107 behaviour.
///
/// The `Cap` arm's `dot(p, axis) > 0` guard is not a micro-optimization: without
/// it a ring with a clockwise lobe reports its lobe's whole *antipodal image* as
/// interior, because `w > π` there reads `W(−p) ≤ −1` rather than `W(p) ≥ 1`.
/// That is `main`'s behaviour and it is wrong; see [`point_in_ring_robust`]'s
/// note on reversed rings.
///
/// # On the witness→`p` arc
///
/// Crossing parity along *any* path decides the even-odd class; the minor arc is
/// such a path for any non-antipodal pair, both endpoints are strictly off the
/// boundary ([`ring_flanks`]), and [`arcs_cross_sos`] is total under SoS for
/// distinct identities — so even the measure-zero coincidence of `p` equal or
/// antipodal to the witness returns a consistent verdict rather than being
/// undefined.
///
/// `reference` is a thunk so a caller holding a cache ([`RingRefs`]) pays for a
/// ring at most once.
fn point_in_ring_with(
    p: &Vec3,
    ix: PointId,
    ring: &[Vec3],
    vid_base: PointId,
    reference: impl FnOnce() -> RingRef,
) -> bool {
    if ring.len() < 3 {
        return false;
    }
    let pi = std::f64::consts::PI;
    let w = ring_winding_at(p, ring);
    match reference() {
        RingRef::Cap { axis, cw } => {
            let near = dot(p, &axis) > 0.0;
            if cw {
                !(near && w < -pi) // W(p) + 1 ≥ 1
            } else {
                near && w > pi //     W(p) ≥ 1
            }
        }
        RingRef::Witness { point, inside } => {
            if w > pi {
                true
            } else if w < -pi {
                false
            } else {
                inside != ring_crossing_parity(&point, p, ANCHOR_ID, ix, ring, vid_base)
            }
        }
        RingRef::Empty => false,
        RingRef::Undecidable => w > pi,
    }
}

/// [`point_in_ring_with`] computing the ring's [`RingRef`] itself.  For a caller
/// that tests many points against the same ring-set this is the wrong entry
/// point — build a [`RingRefs`] once and use [`parity_filled_with`].
fn point_in_ring_ids(p: &Vec3, ix: PointId, ring: &[Vec3], vid_base: PointId) -> bool {
    point_in_ring_with(p, ix, ring, vid_base, || ring_reference(ring))
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
/// The decision is [`point_in_ring_with`]'s: one subtended-angle sum, read
/// against the ring's [`RingRef`], which carries the winding convention's
/// normalization constant (issue #107).  This wrapper synthesizes the SoS
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
/// **Behaviour change in #107.** That last sentence is now true for rings of
/// *every* size; before, it held only past a hemisphere. Reverse a small ring
/// and `main` reported neither the ring nor its complement but the ring's
/// **antipodal image** — `w > π` at a point whose antipode is the clockwise
/// interior — which is not the indicator of any region a caller asked for.
/// [`ring_turning`] supplies the winding direction, so the complement is now
/// returned as documented. Cover paths are unaffected: ingest normalizes
/// sub-hemisphere rings before they get here, and after normalization the two
/// readings coincide.
///
/// # Self-intersecting rings
///
/// The right-hand rule is self-contradictory on a ring that crosses itself —
/// different edges put "the left side" in different regions — so no
/// implementation can be both rotation-invariant and locally-left-consistent
/// there, and S2 declines the input outright (`S2Loop::IsValid`). mortie does
/// not validate, and resolves the ambiguity by convention instead: the interior
/// is the **positively wound** region, `W ≥ 1`, which is the reading
/// [`ring_turning`] falls back to when the ring has no net orientation. That
/// convention is rotation-invariant and density-invariant, and agrees with a
/// simple ring wherever the two notions both apply.
///
/// The companion SoS predicates [`orient_sos`] and [`arcs_cross_sos`] are the
/// orientation-only building blocks the descent's per-cell parity flips use, and
/// the hemisphere-plus crossing walk consumes them directly.
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
    parity_filled_with(p, rings, &RingRefs::of_rings(rings))
}

/// [`parity_filled_robust`] against references the caller has already computed.
///
/// `refs` must come from `RingRefs::of_rings(rings)` for the *same* ring-set; it
/// is positional.  This is the entry point for a descent, which probes thousands
/// of cell centres against one ring-set and must not rebuild the references each
/// time.
pub fn parity_filled_with(p: &Vec3, rings: &[Vec<Vec3>], refs: &RingRefs) -> bool {
    debug_assert_eq!(
        refs.0.len(),
        rings.len(),
        "RingRefs built for another ring-set"
    );
    let mut inside = false;
    let mut vid = RING_VERTEX_ID_BASE;
    for (i, ring) in rings.iter().enumerate() {
        if point_in_ring_with(p, PROBE_ID, ring, vid, || refs.get(i, ring)) {
            inside = !inside;
        }
        vid += ring.len() as PointId;
    }
    inside
}

/// Signed winding direction of a sub-hemisphere `ring`: `+1` if it is wound
/// counter-clockwise (interior — the smaller side — to the **left** of the
/// directed edges, the RFC 7946 / S2 convention), `-1` if clockwise, `0` when
/// no direction is defined (fewer than three vertices, a ring that is not
/// sub-hemisphere, or one with no net orientation at all).
///
/// This is only meaningful for a ring that fits within a hemisphere, where the
/// two regions the ring bounds are unambiguously "small" and "large" and the
/// small side is the intended interior.  Used by [`crate::coverage`] to
/// auto-correct everyday CW input; it must **not** be used to "normalize" a
/// hemisphere+ ring, where area alone cannot pick the interior side (#22).
///
/// # Why this is the turning angle (issue #107)
///
/// The pre-#107 test read [`ring_winding_at`] at the cap axis and called the
/// ring CCW on `+2π`, CW on `−2π`, undecidable otherwise.  That is sound only
/// when the axis is inside the small side — true for a convex ring, **false for
/// a non-convex one**, whose normalized vertex sum can fall in the large region.
/// Then the axis and its antipode are on the *same* side, the sum cancels to
/// `0`, the test reports "undecidable", and ingest silently declines to
/// normalize — leaving a clockwise ring clockwise and selecting the
/// complementary region.  The Antarctic drainage basins in `mortie/tests/` are
/// exactly this shape (their vertex sum lands in the basin's concavity).
///
/// #107's first repair replaced it with a crossing walk from a constructed
/// anchor, which inherited every one of that construction's failures.  Both
/// approaches were asking a *global* question — which side is the interior? —
/// with a *local* instrument.  [`ring_turning`] answers it globally: by
/// Gauss–Bonnet the region to the left has area `2π − turning`, so the interior
/// is the small side exactly when `turning > 0`.  No reference point, no
/// sampling, no step size; invariant under rotating the vertex list and under
/// subdividing edges.  It is `S2Loop::IsNormalized`, which tests
/// `GetCurvature() ≥ −maxError` for the same reason.
///
/// # The undecided band
///
/// A sub-hemisphere ring lies in a cap of angular radius `ρ` with
/// `cos ρ = min_dot > 0`, so its small side has area at most
/// `2π·(1 − min_dot)`.  A **simple** such ring therefore has
/// `|turning| ≥ 2π·min_dot`, whether it is CCW (`|L| ≤ cap`) or CW
/// (`|L| ≥ 4π − cap`) — a guaranteed margin, not a guess.  Anything landing
/// well inside that margin cannot be a simple ring, and in practice is a
/// self-intersecting one whose lobes wind opposite ways and cancel; those get
/// `0`, so ingest leaves them alone and [`point_in_ring_robust`] reads them
/// under the positive-winding convention rather than flipping on the sign of
/// numerical noise.  Half the margin is used, which cannot reject a simple ring
/// and leaves a factor of two over the accumulated float error (`~V × 1e-16`).
pub fn ring_winding_sign(ring: &[Vec3]) -> i32 {
    if ring.len() < 3 {
        return 0;
    }
    let Some((_, min_dot)) = ring_cap(ring) else {
        return 0; // balanced vertex sum ⇒ hemisphere+; caller must not normalize
    };
    if min_dot <= 0.0 {
        return 0; // hemisphere+ ⇒ area cannot pick the interior side
    }
    winding_sign_in_cap(ring, min_dot)
}

/// [`ring_winding_sign`] for a ring already known to be sub-hemisphere, with
/// its cap's `min_dot` in hand.  The single place the turning angle is turned
/// into a direction, shared by [`ring_reference`] and by [`crate::coverage`]'s
/// ingest normalization so the point test and ingest cannot disagree — and so
/// neither pays for the other's bounding-cap pass.
pub fn winding_sign_in_cap(ring: &[Vec3], min_dot: f64) -> i32 {
    let turning = ring_turning(ring);
    let band = std::f64::consts::PI * min_dot; // half of 2π·min_dot
    if turning > band {
        1
    } else if turning < -band {
        -1
    } else {
        0 // no net orientation
    }
}

// ── tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;

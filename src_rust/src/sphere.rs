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
// Issue #107 showed the subtended-angle sum cannot decide the antipodal-lens
// case at all.  [`point_in_ring_robust`] therefore reads that sum against the
// ring's [`RingRef`], which carries the winding convention's normalization
// constant: for a ring confined to a cap that is a closed form and layer 2 is
// never reached, and only a genuinely hemisphere-plus ring falls through to
// `witness_verdict XOR crossing_parity(witness → x)` over layer 2's
// [`arcs_cross_sos`].

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

/// Identity of the crossing walk's reference point (`ring_witness`).  One
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

// ── chain consistency & S2 correspondence (issue #107 phase 2) ───────────
//
// The descent chains even-odd fill along probe arcs (`crate::coverage`'s
// `arc_crossing_parity`), so its correctness is a **joint** property of many
// crossing verdicts, not a per-call one: the verdicts along a chain, and the
// seed verdicts they extend, must all describe one consistent world.  Issue
// #107's "collinear overlap" configuration — a probe arc lying along a polygon
// edge's great circle for tens of degrees, exiting through a shared vertex
// with a transversal sibling edge — is the adversarial case, because there
// every deciding quantity is a ±1-ulp residue.
//
// **Why the chain cannot desynchronize.**  Every sidedness fact any caller
// consumes — a probe leg against an edge, a donor chain against the same edge,
// "which incident edge carries the graze" at a shared vertex, the seed's side
// of the near-plane — is decided by [`orient_sos`] over a triple of identified
// points wherever it is close enough to matter.  [`orient_sos`] evaluates each
// *unordered* triple exactly once, in canonical (identity-sorted) order, with
// an exact determinant sign ([`orient_exact_sign`]), and hands every
// permutation that same result times the permutation parity.  Two call sites
// consulting the same three identified points therefore cannot disagree,
// whichever edges or probe legs they arrived from: the shared vertex's side of
// a probe circle is *one* exact sign, consumed identically by both incident
// edges' four-sign tests.  The invariant #107 asked for ("graze attribution
// must agree with the seed's side of the same near-plane") holds by
// construction, not statistically — subject to three qualifications that are
// what the surrounding code engineers:
//
// * *The float fast path is a triage, not a second opinion.*  The descent's
//   hot path (`crate::coverage::edge_crosses_probe`) decides from four raw f64
//   determinants and calls [`arcs_cross_sos`] only when one of them is within
//   `ORIENT_EPS` (1e-12) of zero.  That is per-call rounding, and it is sound
//   rather than lucky: a cross-then-dot on unit vectors carries ~1e-16 of
//   error, four orders below the gate, so any determinant clearing
//   `ORIENT_EPS` already has the sign exact arithmetic would give.  The two
//   layers cannot return different answers, only cheaper ones.
// * *The seed verdicts split the same way.*  "The seed's side of the
//   near-plane reduces to [`orient_sos`]" is true of the seeds
//   `crate::coverage::seed_fill` declines — it returns `None` for any centre
//   within `ORIENT_EPS` of some edge's plane (4 of the 12 bases on #107's
//   reproducer, including base 0, the one the issue fingered), and
//   `base_fills` chains those from an unambiguous donor through the descent's
//   own SoS parity.  The rest keep the float `parity_filled_with` verdict,
//   consistent for the same reason the fast path is: the gate guarantees they
//   are off every edge plane.
// * *Identities must rank stably — the phase-3 obligation.*  "The same three
//   points" means the same three *identities*, and two id schemes here are
//   one-shot by design: `crate::coverage`'s `CORNER_ID_A`/`CORNER_ID_B` are
//   positional per call, and the bare-geometry PIP synthesizes [`PROBE_ID`]
//   for its test point.  Both are safe, for reasons worth stating rather than
//   assuming.  The corner ids feed `edge_hits_cell_edge`'s straddle boolean
//   only — a per-cell inclusion decision, never a chained parity, so no second
//   call site consumes them.  [`PROBE_ID`] (`MAX - 3`) is safe because of
//   where it *ranks*: like a real cell centre's `center_id` (`>= 2^63`) it
//   sits above every ring-vertex id (see the allocation table above), so every
//   `{p, v_i, v_j}` triple canonicalises in the same order under either and
//   [`orient_sos`] returns the same sign — measured identical on 714 probes
//   laid along the reproducer's edge great circles, including the exact
//   vertex.  Only an id ranked *below* the vertex ids diverges, and only at an
//   exact vertex.  **That rank invariant — probe and centre identities above
//   all polygon-vertex identities — is the load-bearing premise, and what
//   phase 3 must preserve when it threads real identities through to
//   [`point_in_ring_robust`].**
//
// The pre-#107 desync was real, but the inconsistent party was the *float
// reference layer*, not the chain: `ring_winding_at`'s subtended-angle sum
// fed seed verdicts whose rounding could contradict the chain's exact world
// (see "the normalization constant" above).  With the reference repaired
// (phase 1), the reproducer's chain agrees with the reference at every
// uniform cell, and nudging the collinear edge by 1e-6° moves the cover by a
// boundary band (9 cells at order 6) instead of inverting a subtree (8 221
// cells on the pre-repair code) — pinned by
// `test_descent_collinear_edge_matrix` in `coverage/tests.rs` and the
// path-independence tests in `sphere/tests.rs`.
//
// **S2 correspondence, and the perturbation-convention compatibility
// argument.**  s2geometry factors the same problem into `S2::CrossingSign`
// (+1 transversal crossing / 0 shared vertex / −1 otherwise, exact
// arithmetic + symbolic perturbation underneath) and `S2::VertexCrossing`
// (the half-open tie-break at a shared vertex), composed as
// `EdgeOrVertexCrossing` so chained parity stays consistent.  mortie reaches
// the same jointly-consistent end through one mechanism instead of two:
//
// * Four points in general position (every orientation determinant exactly
//   non-zero): [`arcs_cross_sos`] is the same four-orientation identity S2's
//   `CrossingSign` decides, on exact signs of the same coordinates — the
//   verdicts are **provably equal**, and the committed differential fixture
//   (`s2_crossing_fixture.tsv`, generated from the C++ reference — see
//   `mortie/tests/generate_s2_crossing_fixtures.py`) pins the equality on
//   both generic and 1-ulp near-coplanar configurations.
// * Exact degeneracies (a zero determinant, or two edges meeting at a shared
//   coordinate): S2 perturbs **coordinates** (its `Sign` never returns 0, and
//   equal coordinates are *the same point*, handled by `VertexCrossing`);
//   mortie perturbs **identities** (a shared coordinate under two ids is two
//   infinitesimally separated points).  The conventions pick different total
//   orders — measured to disagree on ~29% of exactly-collinear triples
//   (issue #145 notes) — and neither is wrong: each yields one globally
//   consistent perturbed world.  What the chain needs is consistency *within*
//   a world, never agreement *between* them.  For probe endpoints off the
//   boundary, closed-ring crossing parity is the same in both worlds (it
//   equals `fill(p) XOR fill(q)`, and an infinitesimal perturbation of the
//   boundary cannot move a strictly-off-boundary point across it), so the
//   convention choice is invisible off the boundary.  *On* it the verdict is
//   a convention in any implementation — and that case is systematic here,
//   not negligible: HEALPix routinely puts cell centres exactly on polygon
//   edges (the substance of #11, #103 and #107), and 4 of the 12 base seeds
//   on this PR's own reproducer are boundary-coincident.  Which is why mortie
//   does not leave those to a convention at all: `seed_fill` refuses a
//   boundary-coincident seed and `edge_hits_cell_edge` applies the closed-set
//   rule ("a cell whose boundary the polygon touches exactly is always
//   included").  The divergence domain is recorded, not asserted away: the
//   fixture's `coplanar` and `shared_vertex` rows carry both libraries'
//   verdicts, under production-shaped identities (probe arc ranked above edge
//   arc, per the allocation table above) — the perturbation ranking the
//   library actually computes, which is what a drift pin has to pin.
//
// mortie has no `VertexCrossing` because identity perturbation makes the
// shared-vertex case *generic*: the two incident edges see the vertex as one
// identified point whose virtual side of any circle is a single canonical
// sign, so a probe through a ring vertex counts exactly one crossing across
// the incident edge pair when the boundary passes through the probe circle
// (and zero or two on a graze) — the half-open convention S2 hand-maintains
// emerges from the algebra (see [`arcs_cross_sos`] below).  There is likewise
// no stateful `S2EdgeCrosser` analogue: S2 amortizes per-*vertex* state while
// walking one query edge along a polygon chain, whereas the descent's shape
// fixes one probe arc and fans over culled edges — the probe-side state
// (`n_pq`) is hoisted once per leg in `arc_crossing_parity`, the per-edge
// normals are precomputed at ingest (`Edge::n_ab`), and the symbolic path is
// cold behind `ORIENT_EPS` triage, which is the same triage→exact→symbolic
// ladder S2 descends per predicate.  Restructuring the fan into a stateful
// walk would forfeit `edge_hits_cell_edge`'s early exit — the descent's
// measured hot-path contract (#106) — to amortize work the fan does not
// repeat.
//
// The crossing rules implemented here follow the *documented semantics* of
// s2geometry's edge-crossing predicates (Apache-2.0); the implementation is
// mortie's own, per the no-new-dependency decision on issue #107.

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
    debug_assert!(
        ia != ib && ia != ic && ia != id && ib != ic && ib != id && ic != id,
        "arcs_cross_sos requires pairwise-distinct SoS identities \
         (got {ia}, {ib}, {ic}, {id}): a duplicated id makes the symbolic \
         perturbation ill-defined and voids reorder-invariance"
    );
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
/// `0`, `±π`, `±2π`.  Nothing rescues that: for a ring taking the
/// [`RingRef::Cap`] path the verdict at an exactly-on-boundary point is
/// whatever the rounded sum says, the same as it was before #107.  The descent
/// does not depend on it — a cell whose centre lies on an edge is resolved by
/// the SoS crossing predicates, not by this sum.
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
/// Degenerate input is reduced to the ring's real **edges** first.  A repeated
/// vertex traces no edge, but the turn *across* it is a genuine turn — the one
/// between the edge arriving at it and the edge leaving it — and dropping the
/// corner instead of the edge makes the sum depend on vertex duplication, which
/// would destroy the subdivision invariance this whole approach rests on.  (An
/// earlier cut skipped any corner with a degenerate incident edge; duplicating
/// one vertex of a triangle then moved its turning from `6.27` to `4.24`, and
/// duplicating a few spike tips could flip the sign of a valid ring and invert
/// its cover.)
///
/// Test-facing accessor: production code consumes the turning angle only
/// through [`turning_sign`], which needs the paired error bound.
#[cfg(test)]
fn ring_turning(ring: &[Vec3]) -> f64 {
    ring_turning_with_error(ring).0
}

/// The turning sum plus a per-ring **rounding-error bound**: the sum and a
/// `max_error` such that the true turning lies within `± max_error` of it
/// (see the caveat on the sign branch below).
///
/// The bound is derived against *this* summation, per the issue #144 plan —
/// not S2's `GetCurvatureMaxError` constant (`11.25 ε` per vertex), which is
/// an accounting of S2's `RobustCrossProd`/Kahan pipeline, not ours:
///
/// * **Normal direction.**  Each edge normal is a plain `cross(u, v)` with
///   componentwise absolute error `≤ 2ε` (one product rounding, one
///   subtraction), so `≤ 2√3 ε ≈ 3.5ε` as a vector — an **angular** error of
///   up to `3.5ε / |n|`, where `|n| = sin(edge length)` for unit inputs.
///   Unlike S2 we do not robustify the cross product, so short edges cost
///   accuracy: the bound charges each edge `8ε / |n|` (a ×2 cushion over
///   3.5ε), counted once per adjacent turn — i.e. `16ε / |n|` per edge in
///   total, since each normal feeds two turns.  For ordinary rings
///   (edges ≳ 1e-3 rad) this is ~1e-13 per edge; for a ring full of
///   just-above-tolerance edges it honestly balloons, and the sign functions
///   below then refuse to decide rather than flip on noise.
/// * **Per-turn evaluation.**  `angle_between` is `atan2(|a × b|, a·b)`:
///   scale-invariant, and its own rounding contributes a few ε absolute;
///   charged at `8ε` per turn (again ×2 over the ~3–4ε accounting of the
///   norm, dot, and `atan2` roundings).
/// * **Accumulation.**  The sum is Kahan-compensated (this function is the
///   reason it is), so accumulation contributes `O(ε)` per term rather than
///   growing with the running total; it is inside the `8ε` per-turn charge.
///
/// The constants are deliberately loose — the bound's job is to be *safely
/// above* the true error while staying far below any real decision margin
/// (a decisive ring has `|turning|` of order 1) — and they are validated
/// empirically by `test_ring_turning_error_bound_holds`, which measures the
/// actual error against closed forms and reversal antisymmetry at orders of
/// magnitude below the bound.
///
/// # What `max_error` does not cover
///
/// The bound is on the turn *magnitudes*.  Each turn's **sign** is a discrete
/// branch — `dot(a, n_out) >= 0.0` — and that branch is exact only to its own
/// rounding: `dot(a, n_out) = |n_out|·sin(dist(a, GC(b, c)))`, so it is
/// undecidable when a turn is within `~4ε / |n_out|` of `π`, and flipping it
/// moves `turning` by **2π**, not by `max_error`.  A measured witness: a
/// four-vertex ring doubling back on itself with a lateral offset of `±1e-12`
/// rad reads `+5.996` versus `−0.287` under a reported bound of `3.6e-9`.
///
/// This is unreachable for a **simple** ring.  In that witness both readings
/// are correct — the ring crosses from simple to self-intersecting as the
/// offset passes zero — and the branch tracks the truth down to `~1e-15` rad,
/// because `dot` there is linear in the offset against a few ε of rounding.
/// The undecidable set is "within ~1e-15 rad of self-touching at a doubling-
/// back vertex", below the resolution of the input coordinates; ruling it out
/// properly is what issue #145's simplicity check would do.  For multiply-wound
/// or garbage input the guard in [`turning_sign`] catches only part of the
/// range: a 2π shift lands past `2π + π/2` and returns `0` only when the true
/// `|turning| > π/2`.
fn ring_turning_with_error(ring: &[Vec3]) -> (f64, f64) {
    let n = ring.len();
    // Does edge `i -> i+1` trace anything?  Duplicate vertices go by the same
    // `1e-12` componentwise test `ring_crossing_parity` and `build_edges` use;
    // antipodal endpoints define no great circle at all.
    let traces = |i: usize| -> Option<(Vec3, f64)> {
        let (u, v) = (&ring[i], &ring[(i + 1) % n]);
        let dup = (u[0] - v[0]).abs() < 1e-12
            && (u[1] - v[1]).abs() < 1e-12
            && (u[2] - v[2]).abs() < 1e-12;
        if dup {
            return None;
        }
        let nrm = cross(u, v);
        let mag = norm(&nrm);
        (mag >= 1e-15).then_some((nrm, mag))
    };
    // The turn where an edge with normal `n_in`, starting at `a`, meets an edge
    // with normal `n_out`.  `orient(a, b, c) = dot(a, b × c)`
    // and `b × c` is `n_out`, so the sign costs a dot product rather than a
    // third cross.
    let turn = |a: &Vec3, n_in: &Vec3, n_out: &Vec3| {
        let ang = angle_between(n_in, n_out);
        if dot(a, n_out) >= 0.0 {
            ang
        } else {
            -ang
        }
    };
    // One pass, one cross product per edge, no allocation — this runs per ring
    // per cover and the ring can carry a million vertices.  Each edge's normal
    // is carried forward to serve as the *incoming* normal of the next turn:
    // duplicates between two real edges are the same point, so the previous
    // edge's normal is still the direction arriving at this edge's start.
    // (start-vertex index, edge normal, normal magnitude) of a traced edge.
    type Traced = Option<(usize, Vec3, f64)>;
    let (mut first, mut prev): (Traced, Traced) = (None, None);
    let (mut total, mut comp) = (0.0_f64, 0.0_f64); // Kahan sum + compensation
    let (mut kept, mut max_err) = (0usize, 0.0_f64);
    const EPS: f64 = f64::EPSILON;
    let add = |total: &mut f64, comp: &mut f64, term: f64| {
        let y = term - *comp;
        let t = *total + y;
        *comp = (t - *total) - y;
        *total = t;
    };
    for i in 0..n {
        let Some((n_out, mag_out)) = traces(i) else {
            continue;
        };
        max_err += 16.0 * EPS / mag_out; // both turns this normal feeds
        match prev {
            Some((p, ref n_in, _)) => {
                add(&mut total, &mut comp, turn(&ring[p], n_in, &n_out));
                kept += 1;
                max_err += 8.0 * EPS;
            }
            None => first = Some((i, n_out, mag_out)),
        }
        prev = Some((i, n_out, mag_out));
    }
    match (first, prev) {
        (Some((f, ref n_f, _)), Some((p, ref n_p, _))) if f != p => {
            add(&mut total, &mut comp, turn(&ring[p], n_p, n_f));
            kept += 1;
            max_err += 8.0 * EPS;
        }
        _ => return (0.0, 0.0),
    }
    if kept < 3 {
        return (0.0, 0.0); // no closed piecewise-geodesic curve to measure
    }
    (total, max_err)
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
pub fn ring_cap(ring: &[Vec3]) -> Option<(Vec3, f64)> {
    // The vertex sum first: `O(V)`, no allocation, and it is what `main` used,
    // so the overwhelming majority of rings are decided here at no extra cost.
    let mut s = [0.0, 0.0, 0.0];
    for v in ring {
        s[0] += v[0];
        s[1] += v[1];
        s[2] += v[2];
    }
    let sum_cap = (norm(&s) >= 1e-12).then(|| {
        let axis = normalize(&s);
        (axis, min_dot_of(ring, &axis))
    });
    if let Some((_, min_dot)) = sum_cap {
        if min_dot > 0.0 {
            return sum_cap;
        }
    }
    // NOTE: the vertex sum is a *heuristic* enclosing axis, not an exact test
    // for "does this ring fit in an open hemisphere", and it is a poor one for
    // a ring whose vertices are spread unevenly.  A crescent between lat 5° and
    // 10° spanning 300° of longitude sits inside an 85° cap about the north
    // pole, yet sums to `min_dot = −0.62`.  Since decision (A) (issue #144)
    // this no longer affects *correctness* — normalization keys on the turning
    // sign, which needs no cap — it only costs the closed-form fast path: a
    // ring the heuristic misses takes the hemisphere-plus witness route.  The
    // closed form below is correct either way, because it needs only "every
    // vertex is within 90° of `axis`".  Better axis candidates (the minor
    // principal axis; the exact GJK optimum) remain #144's performance half.
    sum_cap
}

/// The smallest dot product any vertex has with `axis`.
fn min_dot_of(ring: &[Vec3], axis: &Vec3) -> f64 {
    ring.iter()
        .map(|v| dot(axis, v))
        .fold(f64::INFINITY, f64::min)
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
    // Three real edges are the minimum that closes a curve.  Counting merely
    // *one* non-degenerate edge was not enough: `[p, −p, r]` has two, encloses
    // nothing, and was being handed to the angle sum, which claimed 88% of the
    // sphere for it — and it would have fed an exactly-180° edge to
    // `arcs_cross_sos`, whose precondition is a minor arc.
    let real_edges = (0..n)
        .filter(|&i| norm(&cross(&ring[i], &ring[(i + 1) % n])) >= 1e-15)
        .count();
    if n < 3 || real_edges < 3 {
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
/// [`RING_VERTEX_ID_BASE`] for the vertices.  Phase 3 of #107 threaded real
/// identities through every `crate::coverage` call site
/// ([`parity_filled_with_id`] under `center_id`s), so the synthesized id
/// remains only in these bare-geometry wrappers, where it is sound for the
/// reason the id-rank invariant states: [`PROBE_ID`] ranks above every
/// ring-vertex id exactly as a `center_id` does, so both canonicalise every
/// `{p, v_i, v_j}` triple identically and return the same verdict even at an
/// exact boundary coincidence (measured identical on 1 842 probes: the #107
/// reproducer's edge great circles, its vertices bit-exactly, and a 1e-3°
/// sweep around the collinear edge's endpoints, under both windings; pinned
/// by `test_probe_id_and_center_id_verdicts_agree`).  An id ranked *below* the
/// vertex ids would perturb the other way at an exact vertex — see "chain
/// consistency & S2 correspondence" above.
///
/// # Winding (orientation) contract
///
/// Ring vertex order **carries meaning** and is the caller's responsibility.
/// The interior of a ring is the region to the **left** of each directed edge,
/// so an exterior ring is wound **counter-clockwise** (CCW) when its interior
/// is the smaller of the two regions it divides the sphere into. This
/// predicate sits *below* ingest normalization, so the same rule binds every
/// ring: under even-odd fill ([`parity_filled_robust`]) a hole carves only
/// when it too is wound so its own small region is on its left — CCW, like its
/// exterior. A CW hole selects its complement instead and inverts the fill.
/// (RFC 7946 §3.1.6's CCW-exterior/CW-hole spelling is what
/// [`crate::coverage`]'s `normalize=true` ingest *delivers*, not what this
/// predicate asks for.)
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
/// The turning sign (`ring_turning_with_error`) supplies the winding
/// direction, so the complement is now returned as documented.
///
/// Cover paths are unaffected under the default, because ingest normalizes
/// every simple ring whose interior decisively reads as the larger region —
/// decision (A) of issue #144, keyed on the turning sign rather than any cap
/// test, so the crescent family the vertex-sum heuristic misses normalizes
/// too (pinned against the C++ S2 goldens in
/// `mortie/tests/test_normalization_s2_goldens.py`).  A reversed ring reaches
/// here as-given only under `normalize=false`, where selecting the complement
/// is the caller's stated intent.
///
/// # Self-intersecting rings
///
/// The right-hand rule is self-contradictory on a ring that crosses itself —
/// different edges put "the left side" in different regions — so no
/// implementation can be both rotation-invariant and locally-left-consistent
/// there, and S2 declines the input outright (`S2Loop::IsValid`). mortie does
/// not validate, and resolves the ambiguity by convention instead: the interior
/// is the **positively wound** region, `W ≥ 1`, which is the reading the
/// turning sum (`ring_turning_with_error`) falls back to when the ring has no
/// net orientation. That
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
/// is positional.  Synthesizes [`PROBE_ID`] for the test point; a caller that
/// holds a stable identity — every `crate::coverage` call site holds a
/// `center_id` — uses [`parity_filled_with_id`] instead (issue #107 phase 3).
pub fn parity_filled_with(p: &Vec3, rings: &[Vec<Vec3>], refs: &RingRefs) -> bool {
    parity_filled_with_id(p, PROBE_ID, rings, refs)
}

/// [`parity_filled_with`] with the test point's own SoS identity `ip`.
///
/// This is the descent's entry point: probing a cell centre under its stable
/// `center_id` makes the verdict *the same fact* the chained parity flips
/// consult — one global perturbed world, not one per call — which is what
/// closes the "self-consistent within one call" caveat the synthesized id
/// carried (issue #107 phase 3).
///
/// `ip` must sit in the band the allocation table reserves for it: **above
/// every ring-vertex id** (the id-rank invariant from "chain consistency & S2
/// correspondence": vertex ids perturb first, so the polygon is nudged off a
/// degenerate lattice before any probe point is) **and below [`ANCHOR_ID`]**
/// (the crossing walk pairs `ip` with the anchor in one triple, and the
/// corner ids sit above the anchor).  Both [`PROBE_ID`] and
/// `crate::coverage`'s `center_id` range satisfy both halves by construction;
/// the debug assert — checked before any verdict is computed — keeps a future
/// caller from silently breaking either.
pub fn parity_filled_with_id(p: &Vec3, ip: PointId, rings: &[Vec<Vec3>], refs: &RingRefs) -> bool {
    debug_assert_eq!(
        refs.0.len(),
        rings.len(),
        "RingRefs built for another ring-set"
    );
    let vid_end = RING_VERTEX_ID_BASE + rings.iter().map(|r| r.len() as PointId).sum::<PointId>();
    debug_assert!(
        ip >= vid_end && ip < ANCHOR_ID,
        "id-rank invariant: the test point's id must rank above every \
         ring-vertex id and below ANCHOR_ID (got {ip}, vertex ids run below \
         {vid_end})"
    );
    let mut inside = false;
    let mut vid = RING_VERTEX_ID_BASE;
    for (i, ring) in rings.iter().enumerate() {
        if point_in_ring_with(p, ip, ring, vid, || refs.get(i, ring)) {
            inside = !inside;
        }
        vid += ring.len() as PointId;
    }
    inside
}

/// A pair of **non-adjacent** vertex positions in a ring-set holding
/// bit-identical coordinates, or `None` when no vertex coordinate recurs
/// outside a consecutive run — issue #145's identity-consistency check,
/// folded into #107 phase 3.  The check is existential: which conflicting
/// pair is returned is the first in the internal coordinate-bit sort order,
/// which is arbitrary, *not* the first in ring/index order.
///
/// The identity-keyed Simulation of Simplicity is stated over *distinct
/// points*; a repeated coordinate under two ids is two virtual points an
/// infinitesimal apart.  Inside the coverage pipeline that is provably
/// benign, for two structural reasons this check exists to keep true:
///
/// * Both ids of a repeated coordinate reach **one predicate triple** only as
///   the endpoints of a single edge — triples mix at most two vertex ids, and
///   always one edge's — and a bit-equal edge is degenerate, which
///   `build_edges` and [`ring_crossing_parity`] skip.
/// * In every *third-party* triple (the repeated coordinate against a probe,
///   corner, anchor, or another vertex), the two instances resolve
///   **identically** — and the reason is the allocation table, not vertex
///   numbering.  The tie-break minors are functions of the triple's
///   coordinates and of the instances' ranks against the other two ids; equal
///   coordinates give equal minors, and the other two ids can never *rank
///   between* the two instances, whatever positions they hold: a triple pairs
///   a vertex with at most one other vertex (its edge partner, the case
///   above), and every remaining identity in the crate — [`PROBE_ID`],
///   [`ANCHOR_ID`], the corner ids, `crate::coverage`'s `center_id`s — ranks
///   *above the whole vertex range* by construction.  So the instances share a
///   rank order against every id they can meet.
///
/// That argument is about ranks, not adjacency, which is what makes it cover
/// the cases the positional check below allows: the closing vertex sits at
/// positions `0` and `n - 1`, i.e. ids `base` and `base + n - 1` with every
/// other vertex of the ring numbered *between* them (allowed here, and sound
/// — no non-vertex id lands in that span), and any wrap-around run is the
/// same shape.  It also explains what the check flags conservatively: `vid`
/// advances by `ring.len()` per ring, so ring 0's last vertex and ring 1's
/// first carry consecutive ids, yet a coordinate shared across two rings is
/// still reported — the check is positional, and a bit-shared vertex between
/// rings is worth surfacing regardless.
///
/// The rank sensitivity being guarded is real, not decorative: on 20 000
/// exactly-coplanar triples (where `orient_exact_sign` returns `0` and the SoS
/// branch always runs), moving one point's id from below to above another's
/// flips `orient_sos` **10 085** times.  That is precisely why a *non-adjacent*
/// recurrence — where the pinch's two strands are separated by the tie-break
/// rather than by geometry — gets flagged.
///
/// So *consecutive* duplicates — including the closing vertex `build_ring`
/// pops, and runs of three or more — are allowed, and **identity-invisible**:
/// the two instances resolve the same way in every predicate.  (Measured over
/// 95 628 probes on 3 000 rings — every vertex, every vertex antipode, every
/// edge midpoint and normal, plus random points — a wrap-duplicated ring and
/// an interior-duplicated one, which differ only in vertex *rank* order, agree
/// on **95 628 / 95 628**; crossing-walk path independence
/// `parity(A→P) ⊕ parity(A→Q) == parity(P→Q)` holds with 0 violations / 99 069
/// triples on each.)  Identity-invisible is not verdict-invariant, though, and
/// the difference is a float path with no identity in it: [`ring_cap`]'s axis
/// is `normalize(Σ vertices)`, so *any* repetition or densification of the
/// vertex list moves the bounding cap and can flip the `Cap` arm's
/// `dot(p, axis) > 0` guard near the cap's 90° circle (measured: deduped vs
/// either duplicate family, 1 191 / 95 628, 742 of them on `Cap` rings, which
/// consult no identity at all).  That is the heuristic-axis concern issue #144
/// already records, not an SoS one.
///
/// What gets
/// flagged is a coordinate recurring at non-adjacent positions: a
/// figure-eight pinched through one bit-exact point, a ring revisiting a
/// vertex, two rings of a set sharing a vertex bitwise.  Such input stays
/// *accepted* (mortie does not validate — the verdict is the documented
/// positive-winding convention, and it stays chain-consistent), but at the
/// pinch the convention's tie-break is keyed to **id order**, i.e. to vertex
/// *numbering*: renumbering the ring can legitimately re-resolve which way
/// the virtual strands separate — the same coordinate-vs-identity divergence
/// family the S2 fixture's `shared_vertex` rows record at the bare-predicate
/// level.  A caller asking "is this input's verdict convention-free?" needs
/// that surfaced, which is what #145's follow-on validity API will consume.
/// Reserved derived identities ([`PROBE_ID`], [`ANCHOR_ID`], corner ids) sit
/// outside the vertex range by construction and cannot conflict here.
///
/// # Scope: equality is bit-exact (modulo `-0.0`)
///
/// Recurrence is keyed to the coordinate's bits, with `-0.0` normalized to
/// `0.0` so numerically equal points share a key.  That is deliberate, and it
/// is the *identity*-relevant equality: only an exactly equal coordinate
/// forces an exactly-zero determinant, which is the one case that hands the
/// verdict to the id-order tie-break.  Two strands that meet at `1e-13` are
/// near-degenerate but not degenerate — their determinant has a sign, and the
/// identities are never consulted.  The `1e-12` tolerance `build_edges` and
/// [`ring_crossing_parity`] apply is a *different* question (is this **edge**
/// degenerate, i.e. too short to carry a direction?) and deliberately not the
/// one asked here; a caller wanting "is this input numerically fragile?"
/// rather than "is its verdict convention-free?" needs a tolerance-based
/// check, which this is not.
///
/// `O(V log V)`: one sort of `(coordinate bits, position)` over the set.
pub fn ring_set_identity_conflict(rings: &[Vec<Vec3>]) -> Option<((usize, usize), (usize, usize))> {
    // `-0.0 == 0.0` numerically and to every predicate here, but the two have
    // different bit patterns — fold them together before keying.
    let key = |t: f64| (if t == 0.0 { 0.0f64 } else { t }).to_bits();
    let bits = |v: &Vec3| [key(v[0]), key(v[1]), key(v[2])];
    let mut keyed: Vec<([u64; 3], (usize, usize))> = rings
        .iter()
        .enumerate()
        .flat_map(|(r, ring)| ring.iter().enumerate().map(move |(i, v)| (bits(v), (r, i))))
        .collect();
    keyed.sort_unstable();
    let mut start = 0;
    while start < keyed.len() {
        let mut end = start + 1;
        while end < keyed.len() && keyed[end].0 == keyed[start].0 {
            end += 1;
        }
        // All positions of one repeated coordinate (sorted by ring, then
        // index).  Allowed iff they lie in one ring as one consecutive
        // *cyclic* block — at most one gap around the cycle, counting the
        // wrap-around gap.  Report the pair across the second gap otherwise.
        let group = &keyed[start..end];
        if end - start > 1 {
            let (r0, _) = group[0].1;
            if let Some(w) = group.windows(2).find(|w| w[0].1 .0 != w[1].1 .0) {
                return Some((w[0].1, w[1].1)); // spans two rings
            }
            let n = rings[r0].len();
            let mut breaks = group
                .windows(2)
                .filter(|w| w[1].1 .1 - w[0].1 .1 > 1)
                .count();
            let (first, last) = (group[0].1 .1, group[end - start - 1].1 .1);
            if first + n - last > 1 {
                breaks += 1; // the wrap-around gap
            }
            if breaks > 1 {
                let w = group
                    .windows(2)
                    .find(|w| w[1].1 .1 - w[0].1 .1 > 1)
                    .expect("breaks > 1 implies an interior gap");
                return Some((w[0].1, w[1].1));
            }
        }
        start = end;
    }
    None
}

/// Signed winding direction of `ring`: `+1` if it is wound counter-clockwise
/// (interior — the smaller side — to the **left** of the directed edges, the
/// RFC 7946 / S2 convention), `-1` if clockwise, `0` when no direction can be
/// vouched for (fewer than three vertices, a reading inside the applicable
/// margin, or a multiply-wound sum).
///
/// **Decision (A), issue #144:** meaningful for *any* simple ring, not only
/// cap-certified ones — Gauss–Bonnet needs no hemisphere.  The pre-(A)
/// contract ("must not be used to normalize a hemisphere+ ring") conflated
/// two things: winding *magnitude* genuinely cannot pick an interior past a
/// hemisphere, but the turning *sign* reads the winding of any simple ring
/// exactly.  What changed is only which margin guards the sign: cap-certified
/// rings keep the conservative geometric band ([`winding_sign_in_cap`]),
/// capless rings use the summation's rounding bound
/// ([`ring_turning_with_error`]).  Used by [`crate::coverage`] to
/// auto-correct input under `normalize=true`; `normalize=false` remains the
/// way to express a big-side interior.
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
/// with a *local* instrument.  The turning sum answers it globally: by
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
    match ring_cap(ring) {
        Some((_, min_dot)) if min_dot > 0.0 => winding_sign_in_cap(ring, min_dot),
        // Decision (A), issue #144: no cap does not mean no direction.  The
        // turning sign is exact for any simple ring — cap-fitting or not —
        // so a ring the vertex-sum heuristic misses (the lat 5–10° crescent)
        // and a genuinely hemisphere-plus simple ring both read their true
        // winding here, against the rounding bound alone.
        _ => turning_sign(ring, 0.0),
    }
}

/// [`ring_winding_sign`] for a ring already known to be sub-hemisphere, with
/// its cap's `min_dot` in hand.  The single place a cap-certified ring's
/// turning angle is turned into a direction, shared by [`ring_reference`] and
/// by [`crate::coverage`]'s ingest normalization so the point test and ingest
/// cannot disagree — and so neither pays for the other's bounding-cap pass.
///
/// Keeps the **geometric band** `π·min_dot`, and keeps it *alone*: a *simple*
/// ring in a cap of `min_dot` satisfies `|turning| ≥ 2π·min_dot` either way
/// round, so a reading inside half that margin cannot be a simple ring — in
/// practice a self-intersecting one whose lobes nearly cancel — and flipping
/// those on a small-but-decisive residue would change which lobe the
/// positive-winding convention selects on the strength of an area imbalance.
/// This is exactly the pre-decision-(A) in-cap behaviour, unchanged: the band
/// is the simplicity guard here, and the summation's rounding bound governs
/// only the capless regime (A) opened, where no cap exists to derive a
/// geometric margin from.  Combining the two with `.max()` would let the
/// rounding bound un-decide a cap-certified ring the band decided (see
/// `turning_sign`).  When issue #145's simplicity check lands,
/// verified-simple rings can drop to the rounding bound alone.
pub fn winding_sign_in_cap(ring: &[Vec3], min_dot: f64) -> i32 {
    turning_sign(ring, std::f64::consts::PI * min_dot)
}

/// [`turning_sign`] against the rounding bound alone — the capless arm of
/// decision (A): exact for any simple ring, `0` when the bound (or the
/// multiply-wound guard) cannot vouch for the sign.
pub(crate) fn ring_turning_sign(ring: &[Vec3]) -> i32 {
    turning_sign(ring, 0.0)
}

/// The banded sign of the turning sum: `+1` counter-clockwise (right-hand-
/// rule interior is the smaller region), `-1` clockwise, `0` undecided.
///
/// Exactly one margin guards the sign, and which one depends on the caller.
/// A positive `geometric_band` is used **alone** — it is the cap-certified
/// arm's simplicity guard (`π·min_dot`, see [`winding_sign_in_cap`]), which
/// dominates the rounding error for any ring not hugging a great circle, and
/// using it alone is precisely the pre-decision-(A) in-cap behaviour: nothing
/// that used to normalize stops normalizing.  Taking `geometric_band.max(
/// max_err)` instead would not be pure extra conservatism — `max_err` can
/// *exceed* `π·min_dot` and un-decide a ring the geometric band decided (a
/// 200 000-vertex circle at colatitude 89.9998° reads `turning = 2.19e-5`,
/// `π·min_dot = 1.10e-5`, `max_err = 2.26e-5`), eating the deliberate factor
/// of two in the geometric band.
///
/// A capless caller passes `0.0` and the rounding bound from
/// [`ring_turning_with_error`] governs instead — the new regime decision (A)
/// opened, where there is no cap to derive a geometric margin from.  It never
/// flips on float noise either: a ring of just-above-tolerance edges has an
/// honestly huge bound and reads `0` here.
fn turning_sign(ring: &[Vec3], geometric_band: f64) -> i32 {
    let (turning, max_err) = ring_turning_with_error(ring);
    // A *simple* ring has `turning = 2π − |L|` with `|L| ∈ (0, 4π)`, so
    // `|turning| < 2π`.  Past that the ring is provably multiply wound, and it
    // has no single winding direction to report: a circle traversed twice
    // clockwise reads `turning ≈ −4π` and holds `W = −2` inside, which the
    // one-step shift of [`RingRef::Cap`] cannot describe — applying it anyway
    // put the *whole sphere* in the cover.  The `π/2` slack keeps a vanishingly
    // small clockwise ring, whose turning approaches `−2π` from above, on the
    // right side of the test.
    if turning.abs() >= std::f64::consts::TAU + std::f64::consts::FRAC_PI_2 {
        return 0;
    }
    let band = if geometric_band > 0.0 {
        geometric_band
    } else {
        max_err
    };
    if turning > band {
        1
    } else if turning < -band {
        -1
    } else {
        0 // no net orientation the margins can vouch for
    }
}

// ── the hemisphere-plus witness construction ─────────────────────────────

mod witness;
use witness::{angle_between, ring_witness};

// ── tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;

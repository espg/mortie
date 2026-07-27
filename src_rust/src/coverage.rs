//! Polygon-to-morton coverage via a **top-down hierarchical region coverer**
//! (issue #30).
//!
//! Starting from the 12 HEALPix base cells, each cell is classified against the
//! polygon ring-set as inside / outside / straddling:
//! `outside` subtrees are pruned, `inside` cells are kept whole at their coarse
//! order, and `straddle` cells are refined into their 4 children down to the
//! target order — where any remaining straddler is a boundary leaf.  There is no
//! boundary rasterization, buffer ring, or flood fill, so the result is a pure,
//! deterministic function of the inputs (fixes the issue #28 class of bug by
//! construction).
//!
//! The descent emits a Multi-Order Coverage map; [`polygon_to_morton_coverage`]
//! flattens it to a single order (back-compatible), while
//! [`polygon_to_morton_moc`] returns the compact mixed-order form.
//!
//! # Ring winding contract
//!
//! Vertex order is meaningful: the interior is the region to the **left** of
//! each directed edge. The single robust winding backend
//! ([`crate::sphere::parity_filled_robust`], #22) that handles hemisphere-plus
//! rings *requires* a direction — past a hemisphere a ring's two sides have
//! equal standing, so only the winding disambiguates which is interior.
//!
//! With `normalize=true` the authored winding does not matter: [`build_ring`]
//! auto-corrects any **simple** ring whose interior decisively reads as the
//! larger region (the Gauss–Bonnet turning sign — S2's `S2Loop::Normalize`
//! convention, decision (A) of issue #144), so CW and CCW spellings of a
//! polygon give the same cover. The RFC 7946 §3.1.6 form — CCW exteriors, CW
//! holes — is what ingest *delivers*, not what it demands.
//!
//! With `normalize=false` nothing is reordered, so every ring must be wound so
//! its own intended region lies to its left — for a carved hole that means
//! **counter-clockwise, like its exterior**, not the RFC's clockwise: a CW
//! hole selects its complement and inverts the even-odd fill. Trusting the
//! supplied order is also how a big-side interior is expressed as a lone ring.

/// Cause-tagged straddle instrumentation for the issue #90 measurement; a
/// dev-only cargo feature (cfg flag, no dependency), absent from release builds.
#[cfg(feature = "descent-stats")]
pub mod descent_stats;

/// Batch (ragged, arrow-list-layout) coverage over many polygons (issue #153).
pub mod batch;

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::f64::consts::PI;

use rayon::prelude::*;
use smallvec::SmallVec;

use crate::cell_geom::{cell_center_vec, cell_corners, Cap};
use crate::geo2mort::{ang2pix_scalar, boundaries_step_scalar};
use crate::morton::nested2mort;
use crate::sphere::{
    arcs_cross_sos, cross, dot, latlon_to_unit_vec, normalize, parity_filled_with,
    parity_filled_with_id, ring_cap, ring_turning_sign, winding_sign_in_cap, PointId, RingRefs,
    Vec3,
};

// ── public entry points ──────────────────────────────────────────────────

/// Compute morton indices covering a polygon, as a flat list at `order`.
///
/// # Arguments
/// * `lats` — vertex latitudes in degrees
/// * `lons` — vertex longitudes in degrees
/// * `order` — HEALPix depth (1–29)
///
/// # Returns
/// Sorted unique `Vec<u64>` of morton indices at `order` whose cells intersect
/// the closed polygon (contract (a): the cover is a superset of the polygon).
///
/// `normalize` toggles the ingest orientation auto-correction (see
/// [`build_ring`]): `true` reverses a sub-hemisphere CW ring to CCW, `false`
/// trusts the supplied vertex order exactly.
///
/// # Panics
/// * If `lats`/`lons` differ in length, fewer than 3 vertices, or order ∉ 1–29.
pub fn polygon_to_morton_coverage(
    lats: &[f64],
    lons: &[f64],
    order: u8,
    normalize: bool,
) -> Vec<u64> {
    let moc = polygon_descend(lats, lons, order, None, normalize);
    crate::moc::to_order(&moc, order).expect("polygon_descend asserted order 1-29")
}

/// Compute polygon coverage as a compact, normalized Multi-Order Coverage map:
/// coarse cells for the interior, fine cells (at `order`) along the boundary.
pub fn polygon_to_morton_moc(lats: &[f64], lons: &[f64], order: u8, normalize: bool) -> Vec<u64> {
    let moc = polygon_descend(lats, lons, order, None, normalize);
    crate::moc::normalize(&moc)
}

/// MOC coverage with a **tolerance** stop: a boundary cell stops refining once
/// its angular radius (radians) drops to `tolerance`, even if that is coarser
/// than `order`.  Produces an approximate cover with a coarser boundary; cheaper.
pub fn polygon_to_morton_moc_tolerance(
    lats: &[f64],
    lons: &[f64],
    order: u8,
    tolerance: f64,
    normalize: bool,
) -> Vec<u64> {
    let moc = polygon_descend(lats, lons, order, Some(tolerance), normalize);
    crate::moc::normalize(&moc)
}

/// MOC coverage with a **`max_cells`** budget: refine the largest boundary cells
/// first (best-first) until the cell count reaches `max_cells`, then stop.
/// Produces an adaptive mixed-order boundary — finer where it wiggles, coarser
/// where it is straight.  `max_cells` is a soft target.
///
/// Returns `(cells, effective_budget)`: if `max_cells` is below the minimum
/// needed to represent the polygon at base resolution, the budget is raised to
/// that floor and `effective_budget > max_cells` signals the caller to warn.
pub fn polygon_to_morton_moc_budget(
    lats: &[f64],
    lons: &[f64],
    order: u8,
    max_cells: usize,
    normalize: bool,
) -> (Vec<u64>, usize) {
    assert_eq!(
        lats.len(),
        lons.len(),
        "lats and lons must have same length"
    );
    assert!(lats.len() >= 3, "Need at least 3 vertices for a polygon");
    assert!((1..=29).contains(&order), "Order must be 1-29");

    let (ring, cap) = build_ring(lats, lons, normalize);
    let rings = vec![ring];
    let refs = ring_refs(&rings, &[cap]);
    let (nodes, effective) = descend_best_first(&rings, &refs, order, max_cells);
    let moc: Vec<u64> = nodes.iter().map(|&(p, d)| nested2mort(p, d)).collect();
    (crate::moc::normalize(&moc), effective)
}

// ── multipart / holes (ring-set, even-odd fill) ──────────────────────────

/// Coverage of a **ring-set** as a flat list at `order`.  All rings are fed to
/// one even-odd descent, so multipart polygons and holes are handled uniformly:
/// a point is covered iff it lies inside an *odd* number of rings (nesting →
/// holes).  A shared interior border between disjoint parts is therefore not a
/// boundary — no per-part seams.  Wind rings per the module's right-hand-rule
/// contract (CCW exterior, CW holes; RFC 7946 §3.1.6).
pub fn multipolygon_to_morton_coverage(
    lats: &[Vec<f64>],
    lons: &[Vec<f64>],
    order: u8,
    normalize: bool,
) -> Vec<u64> {
    validate_multi(lats, lons, order);
    let (rings, refs) = build_rings(lats, lons, normalize);
    let moc = nodes_to_morton(&descend_parallel(&rings, &refs, order, None));
    crate::moc::to_order(&moc, order).expect("validate_multi asserted order 1-29")
}

/// MOC coverage of a ring-set, with optional `tolerance` or `max_cells` stop.
/// Returns `(cells, effective_budget)` (see [`polygon_to_morton_moc_budget`]).
pub fn multipolygon_to_morton_moc(
    lats: &[Vec<f64>],
    lons: &[Vec<f64>],
    order: u8,
    tolerance: Option<f64>,
    max_cells: Option<usize>,
    normalize: bool,
) -> (Vec<u64>, usize) {
    validate_multi(lats, lons, order);
    let (rings, refs) = build_rings(lats, lons, normalize);
    if let Some(budget) = max_cells {
        let (nodes, effective) = descend_best_first(&rings, &refs, order, budget);
        (crate::moc::normalize(&nodes_to_morton(&nodes)), effective)
    } else {
        let nodes = descend_parallel(&rings, &refs, order, tolerance);
        (crate::moc::normalize(&nodes_to_morton(&nodes)), 0)
    }
}

fn validate_multi(lats: &[Vec<f64>], lons: &[Vec<f64>], order: u8) {
    assert!(!lats.is_empty(), "Need at least one ring");
    assert_eq!(
        lats.len(),
        lons.len(),
        "lats and lons must have the same number of rings"
    );
    assert!((1..=29).contains(&order), "Order must be 1-29");
    for (la, lo) in lats.iter().zip(lons.iter()) {
        assert_eq!(la.len(), lo.len(), "ring lats/lons length mismatch");
        assert!(la.len() >= 3, "each ring needs at least 3 vertices");
    }
}

fn build_rings(
    lats: &[Vec<f64>],
    lons: &[Vec<f64>],
    normalize: bool,
) -> (Vec<Vec<Vec3>>, RingRefs) {
    let (rings, caps): (Vec<Vec<Vec3>>, Vec<Option<Vec3>>) = lats
        .iter()
        .zip(lons.iter())
        .map(|(la, lo)| build_ring(la, lo, normalize))
        .unzip();
    let refs = ring_refs(&rings, &caps);
    (rings, refs)
}

/// A [`RingRefs`] for `rings`, pre-seeded wherever ingest already settled the
/// ring's bounding cap and orientation (see [`normalize_ring_orientation`]).
fn ring_refs(rings: &[Vec<Vec3>], caps: &[Option<Vec3>]) -> RingRefs {
    debug_assert_eq!(rings.len(), caps.len());
    let mut refs = RingRefs::of_rings(rings);
    for (i, cap) in caps.iter().enumerate() {
        if let Some(axis) = *cap {
            refs.seed_normalized_cap(i, axis);
        }
    }
    refs
}

#[inline]
fn nodes_to_morton(nodes: &[(u64, u8)]) -> Vec<u64> {
    nodes.iter().map(|&(p, d)| nested2mort(p, d)).collect()
}

// ── descent ──────────────────────────────────────────────────────────────

/// Validate inputs, build the ring-set, run the descent, and return its cells
/// as (un-normalized, mixed-order) morton indices.
fn polygon_descend(
    lats: &[f64],
    lons: &[f64],
    order: u8,
    tolerance: Option<f64>,
    normalize: bool,
) -> Vec<u64> {
    assert_eq!(
        lats.len(),
        lons.len(),
        "lats and lons must have same length"
    );
    assert!(lats.len() >= 3, "Need at least 3 vertices for a polygon");
    assert!((1..=29).contains(&order), "Order must be 1-29");

    let (ring, cap) = build_ring(lats, lons, normalize);
    let rings = vec![ring];
    let refs = ring_refs(&rings, &[cap]);
    descend_parallel(&rings, &refs, order, tolerance)
        .iter()
        .map(|&(pixel, depth)| nested2mort(pixel, depth))
        .collect()
}

/// Convert lat/lon vertices to a closed ring of unit vectors, dropping a
/// duplicate closing vertex if present, then (when `normalize` is `true`)
/// **normalize its orientation** so the smaller region lies to the left of the
/// directed edges — which yields the RFC 7946 §3.1.6 / S2 right-hand-rule form
/// (CCW exterior, CW holes) as its *output*.
///
/// # Expected ring orientation (the contract)
///
/// The PIP backend is winding-direction-aware: the interior is the region to
/// the **left** of each directed edge, and a ring wound the other way selects
/// the *complementary* region.  So under `normalize == false` every ring —
/// holes included — must be wound so its own intended region lies to its left:
/// for a carved hole that is **counter-clockwise, like its exterior**, not the
/// RFC's clockwise, since a CW hole selects its complement and inverts the
/// even-odd fill.  Under `normalize == true` the authored winding is free; RFC
/// 7946 input (CCW exteriors, CW holes) is accepted and reproduced.
///
/// # Orientation normalization (Phase 3, #22) — the `normalize` flag
///
/// `normalize == false` trusts the supplied vertex order exactly and skips all
/// reordering (use this when the caller already winds every ring to the
/// contract above).  `normalize == true` (the default at the public entry points)
/// applies the auto-correction below — a non-breaking convenience for everyday,
/// often-clockwise GIS input.
///
/// The single robust point-in-ring path ([`parity_filled_robust`]) is
/// winding-direction-aware: it treats the region to the *left* of the directed
/// edges as interior, so a ring wound the wrong way selects the complementary
/// region.  To keep everyday (often clockwise) input from silently inverting,
/// **`normalize == true` reverses any ring whose right-hand-rule interior is
/// decisively the larger region** — the S2 convention (`S2Loop::Normalize`),
/// adopted as decision (A) on issue #144.  The instrument is the Gauss–Bonnet
/// turning angle, which is exact for any *simple* ring and needs no bounding
/// cap: `sign(turning) < 0` ⟺ the smaller region lies to the right ⟺ reverse.
/// This is the standard GIS "smaller-area-is-interior" behaviour, so CW and
/// CCW spellings of the same polygon give the same cover — now for every
/// simple ring, not only those the vertex-sum cap heuristic certified (the
/// lat 5–10°, 300°-of-longitude crescent reads `min_dot = −0.62` yet
/// normalizes correctly here; issue #144's reproducer).
///
/// The sign is never taken on faith: cap-certified rings keep the geometric
/// band ([`winding_sign_in_cap`] — a reading inside it cannot be a simple
/// ring), capless rings use the summation's own rounding bound, and a
/// multiply-wound or balanced-figure-eight reading gets `0` — those rings are
/// left exactly as supplied under the positive-winding convention.
///
/// Consequence, superseding the pre-#144 "hemisphere+ vertices are never
/// reordered" rule: a **big-side interior** — "everything except Antarctica"
/// spelled as a reversed Antarctica-hugging ring, or a hemisphere-plus ring
/// whose intended interior is the larger side — is expressed either the way
/// GeoJSON authors it anyway (a whole-world outer ring with a hole) or by
/// passing **`normalize == false`**, the documented escape hatch, which
/// trusts the supplied vertex order exactly.  The point predicates themselves
/// are unchanged: below ingest, winding is always read as given.
fn build_ring(lats: &[f64], lons: &[f64], normalize: bool) -> (Vec<Vec3>, Option<Vec3>) {
    let mut ring: Vec<Vec3> = lats
        .iter()
        .zip(lons.iter())
        .map(|(&la, &lo)| latlon_to_unit_vec(la, lo))
        .collect();
    if ring.len() > 3 {
        let (f, l) = (ring[0], ring[ring.len() - 1]);
        if (f[0] - l[0]).abs() < 1e-12 && (f[1] - l[1]).abs() < 1e-12 && (f[2] - l[2]).abs() < 1e-12
        {
            ring.pop();
        }
    }
    let cap_axis = if normalize {
        normalize_ring_orientation(&mut ring)
    } else {
        None
    };
    (ring, cap_axis)
}

/// Auto-correct a ring wound clockwise — one whose right-hand-rule interior
/// is the **larger** region — to the CCW, small-side-on-the-left convention.
/// See [`build_ring`] for the rationale and the decision-(A) semantics.
///
/// **Decision (A), issue #144:** the flip decision keys on the sign of the
/// turning angle alone (Gauss–Bonnet: exact for any simple ring), so it no
/// longer needs the ring to fit a bounding cap.  The cap survives here for
/// two narrower jobs: cap-certified rings keep the *more conservative*
/// geometric band on their sign ([`sphere::winding_sign_in_cap`] — protection
/// against flipping a nearly-cancelling self-intersecting ring on an area
/// imbalance), and a cap is still what a returned axis certifies for
/// [`RingRefs::seed_normalized_cap`].  Capless rings — genuinely
/// hemisphere-plus simple rings; since issue #144's performance half the
/// crescent family is cap-certified by the principal-axis candidate rather
/// than landing here — now normalize on the rounding-bounded sign
/// ([`sphere::ring_turning_sign`]); expressing a big-side interior takes
/// `normalize=False`, the documented escape hatch.
///
/// Returns the ring's bounding-cap axis when the vertices fit one (the ring
/// is then unambiguously CCW *and* sub-hemisphere, the two facts seeding
/// requires); `None` means only that no cap was certified — the ring may
/// still have been reversed — and the reference stays lazy.
fn normalize_ring_orientation(ring: &mut [Vec3]) -> Option<Vec3> {
    if ring.len() < 3 {
        return None;
    }
    // `sphere::ring_cap`, not a second copy of the vertex-sum computation.  The
    // copy that used to live here meant ingest and the point predicate derived
    // the cap independently, so improving one silently left the other behind —
    // which is exactly what happened when the enclosing axis of issue #144 was
    // first attempted.
    let cap = ring_cap(ring).filter(|&(_, min_dot)| min_dot > 0.0);
    // The same banded turning-angle tests the point predicate uses
    // (`ring_reference` / `ring_winding_sign` dispatch identically), which is
    // what stops ingest and `point_in_ring_robust` from ever disagreeing
    // about a ring's orientation.
    let sign = match cap {
        Some((_, min_dot)) => winding_sign_in_cap(ring, min_dot),
        None => ring_turning_sign(ring),
    };
    if sign < 0 {
        ring.reverse();
    }
    // Reversing negates every exterior angle, so the ring is now unambiguously
    // counter-clockwise whichever branch was taken.  The vertex sum is
    // order-independent, so a certified axis still describes the ring.
    cap.map(|(axis, _)| axis)
}

/// A polygon edge as a great-circle arc `a→b`, with a bounding cap (`mid`,
/// angular half-length `rho`) for culling and the leaf cell of endpoint `a`
/// (at the target order) for exact vertex-in-cell tests.
struct Edge {
    a: Vec3,
    b: Vec3,
    /// Great-circle normal `a × b`, precomputed once so the per-cell crossing
    /// tests reduce to dot products (two dot products per edge).
    n_ab: Vec3,
    mid: Vec3,
    cos_rho: f64,
    sin_rho: f64,
    leaf: u64,
    /// Stable Simulation-of-Simplicity identities of the endpoints `a`, `b`
    /// (their global vertex index across the ring-set).  Feed the descent's
    /// symbolic crossing test ([`arcs_cross_sos`]) so a probe arc whose endpoint
    /// lies exactly on this edge's great circle — e.g. a HEALPix base-cell centre
    /// on a base-cell-centre meridian (issues #11/#103) — resolves to a definite,
    /// traversal-order-independent side instead of a sign-unstable plain test.
    ia: PointId,
    ib: PointId,
}

/// Symbolic identity of a cell centre: its HEALPix UNIQ code with the top bit
/// set, so it is unique per `(depth, pixel)`, stable across every probe arc
/// that touches the centre, and disjoint from the ring-vertex id range.
///
/// Stability is load-bearing (#103): the even-odd fill parity is chained along
/// centre→centre probe arcs, and when a centre lies **exactly** on a polygon
/// edge its SoS-resolved virtual side must be the same in the arc that arrives
/// at it and in every arc that leaves it — otherwise the chain is
/// path-dependent and one boundary-coincident centre inverts its whole
/// subtree.  Positional ids (source=0 / target=1, the pre-#103 scheme) broke
/// exactly this.  With per-point ids the symbolic perturbation is one global,
/// consistent perturbation of the configuration; centres carry the top-bit
/// range, so ring vertices (low ids) perturb first — the polygon is nudged off
/// the degenerate grid once, the same way for every probe.
#[inline]
fn center_id(depth: u8, pixel: u64) -> PointId {
    (1u64 << 63) | ((4u64 << (2 * depth as u32)) + pixel)
}

/// SoS ids for one-shot derived endpoints (cell corners, densified near-pole
/// boundary points).  These feed only the straddle *boolean* — never the
/// chained fill parity — so they need distinctness within a call, not global
/// stability.  Top-of-range keeps them clear of vertex and centre ids.
const CORNER_ID_A: PointId = PointId::MAX;
const CORNER_ID_B: PointId = PointId::MAX - 1;
/// Ring vertices are numbered from here (their global index across the
/// ring-set), below every centre / corner id.
const VERTEX_ID_BASE: PointId = 2;

/// A straddle determinant (a scalar triple product of unit vectors) below this
/// magnitude is treated as a degeneracy and routed to the SoS-robust crossing
/// test.  Clean crossings give O(1) determinants, so this only fires near an
/// exact-on-edge configuration (the #11 cell-centre-on-meridian case ≈ 1e-18);
/// the margin keeps it off the common path while still catching the degeneracy.
const ORIENT_EPS: f64 = 1e-12;

/// Forward error bound for the straddle determinant `dot(a × b, c)` as this
/// module computes it — `cross` then `dot`, both in plain `f64`.
///
/// The closed-set contract (#103) has to recognise *exact incidence*: a point
/// lying on a great circle makes the determinant zero in exact arithmetic.  In
/// `f64` it is zero only when the rounding happens to cancel, which is why
/// testing `d == 0.0` caught the on-grid family (where the arithmetic is
/// symmetric) but missed a polygon vertex placed on a HEALPix cell corner from
/// outside — the vertex arrives through `lat/lon → Vec3` while the corner comes
/// from `cell_corners`, so the two agree to ~1e-16 and never bit-exactly.  That
/// was the documented vertex-point-touch gap (issue #117 item 1).  The honest
/// predicate is "not provably nonzero": `|d| <= bound` means the sign is below
/// the noise floor of its own computation, so the configuration is
/// indistinguishable from incidence and the closed set claims it.
///
/// Derivation, with `u = EPSILON / 2` the unit roundoff and
/// `P_i = |a_j b_k| + |a_k b_j|` the permanent of component `i`'s 2×2 minor:
///
/// * `n_i = fl(a_j b_k - a_k b_j)` rounds twice in the products and once in the
///   difference, so `|Δn_i| ≤ 2u · P_i` to first order.
/// * `fl(Σ n_i c_i)` over three terms rounds three times in the products and
///   twice in the sums: `≤ 4u · Σ|n_i c_i| ≤ 4u · Σ P_i |c_i|`, since
///   `|n_i| ≤ P_i`.
///
/// The two contributions sum to `6u · Σ P_i |c_i|`; taking `8u` rounds the
/// second-order terms up rather than tracking them.  The bound is exact-zero
/// when the inputs are (a degenerate edge gives `P_i = 0`), so it never widens
/// a configuration that carries no rounding to begin with.
/// The widened incidence test is only meaningful while the slack it implies
/// stays small compared with the arc it is testing against.  `bound / |n|` is
/// that slack in radians and `|n| = |a x b|` is (to first order) the arc's own
/// length, so the criterion is `bound <= FRACTION * |n|^2`.
///
/// It matters because `cross` of two nearly-parallel unit vectors cancels: at
/// order 29 a cell edge is ~1.9e-9 rad, the determinant's rounding is ~9e-16
/// regardless, and the implied slack is ~5e-7 rad — hundreds of cells wide.
/// Widening there would not recognise incidence, it would erase sidedness (a
/// 3 mm order-29 triangle went from 3 cells to 288 before this gate).  Below
/// the threshold the configuration falls through to `arcs_cross_sos` for an
/// exact symbolic verdict, exactly as it did before the widening — so the
/// closed set degrades to the old bit-exact behaviour at very fine orders
/// rather than misbehaving.  The crossover sits near order 21.
const INCIDENCE_SLACK_FRACTION: f64 = 0.01;

/// Is `d = dot(a x b, c)` indistinguishable from zero at working precision,
/// on an arc long enough for that question to mean anything?
#[inline]
fn indistinguishable_from_zero(d: f64, a: &Vec3, b: &Vec3, c: &Vec3, n_len2: f64) -> bool {
    let bound = straddle_error_bound(a, b, c);
    d.abs() <= bound && bound <= INCIDENCE_SLACK_FRACTION * n_len2
}

#[inline]
fn straddle_error_bound(a: &Vec3, b: &Vec3, c: &Vec3) -> f64 {
    const F: f64 = 8.0 * (f64::EPSILON / 2.0);
    let p = |j: usize, k: usize| (a[j] * b[k]).abs() + (a[k] * b[j]).abs();
    F * (p(1, 2) * c[0].abs() + p(2, 0) * c[1].abs() + p(0, 1) * c[2].abs())
}

/// Build the edge list for a ring-set, each with its bounding cap, the leaf
/// cell of its start vertex, and the global SoS ids of its endpoints.
fn build_edges(rings: &[Vec<Vec3>], order: u8) -> Vec<Edge> {
    let mut edges = Vec::new();
    let mut vid: PointId = VERTEX_ID_BASE;
    for ring in rings {
        let m = ring.len();
        if m < 2 {
            vid += m as PointId;
            continue;
        }
        for i in 0..m {
            let a = ring[i];
            let b = ring[(i + 1) % m];
            // Skip a zero-length edge (a duplicate consecutive vertex): it traces
            // no boundary, and its degenerate `a × b ≈ 0` normal would only feed
            // noise into the crossing tests.  SoS ids are positional (`vid + i`),
            // so dropping the edge leaves the other edges' ids untouched.
            if (a[0] - b[0]).abs() < 1e-12
                && (a[1] - b[1]).abs() < 1e-12
                && (a[2] - b[2]).abs() < 1e-12
            {
                continue;
            }
            let mid = normalize(&[a[0] + b[0], a[1] + b[1], a[2] + b[2]]);
            let cos_rho = dot(&mid, &a).clamp(-1.0, 1.0);
            let sin_rho = (1.0 - cos_rho * cos_rho).max(0.0).sqrt();
            let lon = a[1].atan2(a[0]).to_degrees();
            let lat = a[2].clamp(-1.0, 1.0).asin().to_degrees();
            edges.push(Edge {
                a,
                b,
                n_ab: cross(&a, &b),
                mid,
                cos_rho,
                sin_rho,
                leaf: ang2pix_scalar(order, lon, lat),
                ia: vid + i as PointId,
                ib: vid + ((i + 1) % m) as PointId,
            });
        }
        vid += m as PointId;
    }
    edges
}

/// `(cos, sin)` of a cell's circumradius (max angular distance centre→corner).
fn cell_cos_radius(center: &Vec3, corners: &[Vec3; 4]) -> (f64, f64) {
    let cos_cr = corners
        .iter()
        .map(|c| dot(center, c))
        .fold(1.0_f64, f64::min);
    let sin_cr = (1.0 - cos_cr * cos_cr).max(0.0).sqrt();
    (cos_cr, sin_cr)
}

/// Could edge `e`'s cap overlap a cell's cap (centre, circumradius cr)?  True
/// iff `angle(e.mid, center) <= rho + cr`, i.e. `cos(angle) >= cos(rho + cr)`.
#[inline]
fn edge_relevant(e: &Edge, center: &Vec3, cos_cr: f64, sin_cr: f64) -> bool {
    let cos_sum = e.cos_rho * cos_cr - e.sin_rho * sin_cr;
    dot(&e.mid, center) >= cos_sum - 1e-12
}

/// Parity (odd?) of how many of `relevant` edges the short arc `p→q` crosses.
/// Flips the even-odd fill state between two nearby points.
///
/// The crossing test goes through [`arcs_cross_sos`]: the descent walks `fill`
/// from a cell centre to a neighbour, and when an edge's great circle passes
/// exactly through one endpoint (the issue #11/#103 degeneracy — a cell centre
/// or a polygon edge sitting on a base-cell meridian), a plain sign test is
/// noise-unstable and flip-flops the parity, flooding the cell's whole subtree.
/// The symbolic predicate decides every sidedness question on a canonical,
/// identity-keyed [`orient_sos`] evaluation with no constructed intersection
/// point, so a probe passing exactly through a ring vertex counts exactly one
/// crossing across the two incident edges (#103) and the parity is stable for
/// the whole on-grid family.  This runs on the descent's per-cell fan, but only
/// over the `relevant` edges (those whose cap reaches the cell), so it stays
/// off the bulk path.
///
/// The symbolic path is only needed at a degeneracy (a straddle orientation
/// rounding near zero), which is rare; away from it the plain four-orientation
/// test is exact and far cheaper.  So each edge takes the fast sign path unless
/// one of its four straddle determinants is within [`ORIENT_EPS`] of zero, in
/// which case it falls back to [`arcs_cross_sos`].  This keeps the common
/// descent path at dot-product speed while preserving the #11/#103 fix.
#[inline]
fn arc_crossing_parity(
    p: &Vec3,
    q: &Vec3,
    ip: PointId,
    iq: PointId,
    relevant: &[usize],
    edges: &[Edge],
) -> bool {
    let n_pq = cross(p, q);
    let mut crossings = 0u32;
    for &i in relevant {
        let e = &edges[i];
        if edge_crosses_probe(p, q, ip, iq, &n_pq, e) {
            crossings += 1;
        }
    }
    crossings & 1 == 1
}

/// Does polygon edge `e` cross the probe arc `p→q` (normal `n_pq = p × q`)?
/// Fast sign test unless a straddle determinant is near-degenerate, then the
/// symbolic [`arcs_cross_sos`].  See [`arc_crossing_parity`].
#[inline]
fn edge_crosses_probe(p: &Vec3, q: &Vec3, ip: PointId, iq: PointId, n_pq: &Vec3, e: &Edge) -> bool {
    // The four straddle determinants of the standard arcs-cross test.  Near-zero
    // in any of them is the cell-centre-on-edge degeneracy (#11) where the plain
    // sign test is unstable; defer those to SoS.
    let d_pq_a = dot(n_pq, &e.a);
    let d_pq_b = dot(n_pq, &e.b);
    let d_ab_p = dot(&e.n_ab, p);
    let d_ab_q = dot(&e.n_ab, q);
    if d_pq_a.abs() < ORIENT_EPS
        || d_pq_b.abs() < ORIENT_EPS
        || d_ab_p.abs() < ORIENT_EPS
        || d_ab_q.abs() < ORIENT_EPS
    {
        return arcs_cross_sos(p, q, &e.a, &e.b, ip, iq, e.ia, e.ib);
    }
    // Fast path: p, q straddle edge AB's great circle and a, b straddle PQ's.
    (d_pq_a > 0.0) != (d_pq_b > 0.0) && (d_ab_p > 0.0) != (d_ab_q > 0.0)
}

/// Does polygon edge `e` cross **or exactly touch** the cell edge `c1 → c2`
/// (normal `n_c = c1 × c2`)?  The straddle test's closed-set form (#103).
///
/// The hot path keeps the exact two-dot shape of the retired `arcs_cross_n`
/// (pruned in #107):
/// corners strictly (beyond [`ORIENT_EPS`]) on one side of the polygon edge's
/// great circle exit immediately — the CodSpeed-visible cost of computing all
/// four determinants up front was a ~19% coverage regression.  Degenerate
/// configurations take the closed-set logic:
///
/// * A determinant **at or below its own rounding error**
///   ([`straddle_error_bound`]) with the incident point inside the *segment*
///   cap (angular slack, `sin ρ`-scaled) is true geometric incidence — the
///   cell is boundary-touching → refined → included ("a cell whose boundary
///   the polygon touches exactly is always included").
/// * A determinant above that bound but still near zero (< [`ORIENT_EPS`])
///   routes to the symbolic [`arcs_cross_sos`] for an exact verdict: it is
///   provably off the great circle, so it resolves deterministically to one
///   side rather than counting as a touch.
///
/// The error-bounded incidence test is what closes the vertex-point-touch gap
/// (issue #117 item 1, folded into #107 on espg's direction): a polygon vertex
/// placed on a HEALPix cell corner reaches the descent through
/// `lat/lon → Vec3` while the corner comes from `cell_corners`, so the two
/// agree to ~1e-16 and a `d == 0.0` test saw only the vertex's own leaf-owning
/// cell.  Every cell whose boundary the vertex lands on is now included.
/// Edge-collinear touches — the systematic on-grid family the closed-set
/// contract exists for — make `d1`/`d2` degenerate as before and are
/// unaffected.
///
/// Cell corners are one-shot derived points ([`CORNER_ID_A`]/[`CORNER_ID_B`]):
/// only distinctness matters within a call — this feeds the straddle boolean,
/// never the chained fill parity.
#[inline(always)]
fn edge_hits_cell_edge(e: &Edge, c1: &Vec3, c2: &Vec3, n_c: &Vec3) -> bool {
    let d1 = dot(&e.n_ab, c1);
    let d2 = dot(&e.n_ab, c2);
    let degenerate12 = d1.abs() < ORIENT_EPS || d2.abs() < ORIENT_EPS;
    if !degenerate12 && (d1 > 0.0) == (d2 > 0.0) {
        return false; // hot exit: corners strictly on one side
    }
    let d3 = dot(n_c, &e.a);
    let d4 = dot(n_c, &e.b);
    if degenerate12 || d3.abs() < ORIENT_EPS || d4.abs() < ORIENT_EPS {
        return edge_touches_cell_edge_degenerate(e, c1, c2, n_c, d1, d2, d3, d4);
    }
    // d1, d2 straddle (established above); the plain sign test decides.
    let crossed = (d3 > 0.0) != (d4 > 0.0);
    #[cfg(feature = "descent-stats")]
    if crossed {
        descent_stats::note_touch(false);
    }
    crossed
}

/// Cold half of [`edge_hits_cell_edge`]: the closed-set / symbolic logic for
/// degenerate configurations.  Outlined (`#[cold]`) so the hot shell above
/// stays small enough to inline into the quad loop — as a single function the
/// degenerate branch pushed it past the inlining threshold and the per-test
/// call overhead showed up as a CodSpeed regression on every coverage bench.
#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn edge_touches_cell_edge_degenerate(
    e: &Edge,
    c1: &Vec3,
    c2: &Vec3,
    n_c: &Vec3,
    d1: f64,
    d2: f64,
    d3: f64,
    d4: f64,
) -> bool {
    // Exact incidence ⇒ closed-set touch, gated to the segment span (the
    // infinite great circle runs to the poles; gating on it alone emitted
    // a spurious strip of cells far beyond the polygon).
    let span = |cos_rho: f64, sin_rho: f64, d: f64| {
        d >= cos_rho - sin_rho * ORIENT_EPS - ORIENT_EPS * ORIENT_EPS
    };
    let n_ab2 = dot(&e.n_ab, &e.n_ab);
    let on12 = |d: f64, c: &Vec3| indistinguishable_from_zero(d, &e.a, &e.b, c, n_ab2);
    if (on12(d1, c1) && span(e.cos_rho, e.sin_rho, dot(&e.mid, c1)))
        || (on12(d2, c2) && span(e.cos_rho, e.sin_rho, dot(&e.mid, c2)))
    {
        #[cfg(feature = "descent-stats")]
        descent_stats::note_touch(true);
        return true;
    }
    let n_c2 = dot(n_c, n_c);
    let on34 = |d: f64, v: &Vec3| indistinguishable_from_zero(d, c1, c2, v, n_c2);
    if on34(d3, &e.a) || on34(d4, &e.b) {
        let mid = normalize(&[c1[0] + c2[0], c1[1] + c2[1], c1[2] + c2[2]]);
        let cos_rho = dot(&mid, c1);
        let sin_rho = (1.0 - cos_rho * cos_rho).max(0.0).sqrt();
        if (on34(d3, &e.a) && span(cos_rho, sin_rho, dot(&mid, &e.a)))
            || (on34(d4, &e.b) && span(cos_rho, sin_rho, dot(&mid, &e.b)))
        {
            #[cfg(feature = "descent-stats")]
            descent_stats::note_touch(true);
            return true;
        }
    }
    let crossed = arcs_cross_sos(c1, c2, &e.a, &e.b, CORNER_ID_A, CORNER_ID_B, e.ia, e.ib);
    #[cfg(feature = "descent-stats")]
    if crossed {
        descent_stats::note_touch(false);
    }
    crossed
}

/// A descent node: cell `(pixel, depth)`, its centre, even-odd fill state, and
/// the subset of polygon edge indices whose caps reach the cell.
struct Node {
    pixel: u64,
    depth: u8,
    center: Vec3,
    /// The cell's four corners and the cosine of its circumradius, cached at
    /// construction (both are computed there anyway to cull edges).  Caching
    /// lets the straddle and radius tests reuse them instead of refetching
    /// corners from HEALPix — the descent hot loop.
    corners: [Vec3; 4],
    cos_cr: f64,
    fill: bool,
    /// Polygon edge indices whose caps reach the cell.  Inline-stored for the
    /// common ≤8-edge case to avoid a heap allocation per descent node, with
    /// transparent heap spill beyond.
    relevant: SmallVec<[usize; 8]>,
}

/// Would the vertex-cap cull prune a base cell that is actually **inside** the
/// polygon (the complement / hemisphere+ case)?  The bounding [`Cap`] only
/// encloses the ring *vertices*, so it bounds the boundary, not the interior.
/// For a sub-hemisphere polygon the interior lies inside that cap and the cull in
/// [`base_node`] is sound.  For a hemisphere+ polygon ("everything except
/// Antarctica") the interior is the *large* region wrapping the cap's antipode,
/// and culling base cells by distance from the cap axis would wrongly prune cells
/// deep inside.
///
/// We answer the question **exactly** at base-cell granularity with the any-size
/// robust PIP ([`parity_filled_robust`], issue #22): probe every base-cell
/// **centre the cull would prune** (those beyond `radius + cr` of the axis), plus
/// the cap-axis antipode; if any tests inside the filled region, the cull would
/// drop an interior cell, so it must be disabled.  Probing the cull's own
/// candidates — not the antipode alone — makes detection exact for multipart /
/// holed geometry too, where the antipode's even-odd parity need not reflect the
/// large region.  Computed once per descent (≤13 PIPs, off the hot path).
fn covers_complement(rings: &[Vec<Vec3>], cap: &Cap, refs: &RingRefs) -> bool {
    // The antipode is a derived, one-shot point with no stable identity of
    // its own — like the corner ids, its verdict feeds a boolean decision
    // (cull on/off), never the chained fill parity, so the synthesized
    // [`sphere::PROBE_ID`] of the bare wrapper is the right identity for it.
    let antipode = [-cap.axis[0], -cap.axis[1], -cap.axis[2]];
    if parity_filled_with(&antipode, rings, refs) {
        return true;
    }
    (0..12u64).any(|base| {
        let center = cell_center_vec(0, base);
        let corners = cell_corners(0, base);
        let (cos_cr, sin_cr) = cell_cos_radius(&center, &corners);
        // Cells the cull would prune (beyond radius + cr of the axis) that still
        // test inside ⇒ the cull would drop an interior cell.  Centres carry
        // their stable `center_id` (issue #107 phase 3).
        cap.excludes(&center, cos_cr, sin_cr)
            && parity_filled_with_id(&center, center_id(0, base), rings, refs)
    })
}

/// Even-odd fill verdict at `x` from the winding backend, or `None` when the
/// verdict is **untrustworthy** because `x` lies (numerically) on some edge's
/// great circle.
///
/// The winding backend cannot flag its own boundary tie: for a point exactly
/// on an edge, that edge's subtended angle is ±π with the *sign* decided by
/// rounding noise, so the total lands on a legitimate-looking `0` or `±2π` —
/// the #103 seed failure (a base centre exactly on a polygon edge tips an
/// entire subtree, in whichever direction the noise fell).  The tie is
/// detected geometrically instead: `x` within [`ORIENT_EPS`] of any edge's
/// great-circle plane makes the seed ambiguous, and [`base_fills`] classifies
/// it through the SoS parity chain, whose answer is exact in the symbolically
/// perturbed world the descent's own probes live in.
fn seed_fill(
    x: &Vec3,
    ix: PointId,
    unit_normals: &[Vec3],
    rings: &[Vec<Vec3>],
    refs: &RingRefs,
) -> Option<bool> {
    for n in unit_normals {
        if dot(n, x).abs() < ORIENT_EPS {
            return None;
        }
    }
    // `ix` is the centre's stable `center_id` (issue #107 phase 3): the seed
    // verdict and every chained probe that later consults this centre resolve
    // it in the same globally perturbed world.
    Some(parity_filled_with_id(x, ix, rings, refs))
}

/// Are base cells `a` and `b` centred on antipodal points?  The 12 HEALPix
/// base centres pair up as north-cap/south-cap (0,10) (1,11) (2,8) (3,9) and
/// equatorial (4,6) (5,7); a chain probe between an antipodal pair would be a
/// degenerate (non-minor) arc, so [`base_fills`] never picks one.
fn antipodal_base(a: u64, b: u64) -> bool {
    matches!(
        (a.min(b), a.max(b)),
        (0, 10) | (1, 11) | (2, 8) | (3, 9) | (4, 6) | (5, 7)
    )
}

/// Fill states for the 12 base-cell centres (#103 seed hardening).
///
/// Each seed takes the winding verdict when it is trustworthy
/// ([`seed_fill`]); a seed whose centre lies numerically **on** some edge's
/// great circle is instead chained from an unambiguous donor centre through the
/// same SoS crossing parity the descent itself uses
/// ([`arc_crossing_parity`] over the full edge list), so its verdict is exact
/// and consistent with every later probe flip — the raw winding tie can no
/// longer invert a base subtree.  If every centre is ambiguous (the boundary
/// passes through all 12 — pathological), the raw winding verdicts are kept.
fn base_fills(
    edges: &[Edge],
    rings: &[Vec<Vec3>],
    cap: &Cap,
    complement: bool,
    refs: &RingRefs,
) -> [bool; 12] {
    let centers: Vec<Vec3> = (0..12).map(|b| cell_center_vec(0, b)).collect();
    // Unit edge normals once (not per seed): the plane-proximity test is the
    // only consumer and 12 × E normalizations showed up in the profile.
    let unit_normals: Vec<Vec3> = edges.iter().map(|e| normalize(&e.n_ab)).collect();
    let mut fill = [false; 12];
    let mut known = [false; 12];
    // Classify only the seeds the descent will actually use: a base cell the
    // vertex-cap cull rejects never reaches base_node, and paying a full
    // winding PIP for it was the dominant fixed cost CodSpeed flagged on
    // small polygons (12 PIPs where pre-#103 code paid ~2).  Culled bases
    // keep fill = false; base_node returns None for them regardless.
    // The cull-skip's exactness argument (an excluded centre is provably
    // outside the polygon, so fill = false is a valid donor verdict) needs
    // the boundary contained in the vertex cap, which cap convexity only
    // gives for sub-hemisphere caps: a minor arc between in-cap vertices can
    // exit a > 90° cap by up to a quadrant.  Gate the skip accordingly —
    // oversize caps classify all 12 seeds, the pre-#109 behavior.
    let cullable = !complement && cap.radius <= std::f64::consts::FRAC_PI_2;
    for b in 0..12u64 {
        if cullable {
            let center = &centers[b as usize];
            let corners = cell_corners(0, b);
            let (cos_cr, sin_cr) = cell_cos_radius(center, &corners);
            if cap.excludes(center, cos_cr, sin_cr) {
                known[b as usize] = true; // culled: fill value is never read
                continue;
            }
        }
        if let Some(f) = seed_fill(
            &centers[b as usize],
            center_id(0, b),
            &unit_normals,
            rings,
            refs,
        ) {
            fill[b as usize] = f;
            known[b as usize] = true;
        }
    }
    if known.iter().all(|&k| k) {
        return fill; // common case: no seed on the boundary
    }
    if known.iter().all(|&k| !k) {
        // Pathological: boundary through all 12 centres; keep raw winding.
        for b in 0..12u64 {
            fill[b as usize] =
                parity_filled_with_id(&centers[b as usize], center_id(0, b), rings, refs);
        }
        return fill;
    }
    // NOTE: a culled base is marked known (fill unused), so the donor search
    // below can only pick a *classified* seed when at least one un-culled,
    // unambiguous seed exists; if every un-culled seed was ambiguous, the
    // donors include culled entries whose fill is false — the correct verdict
    // for a base whose cap the polygon's vertices cannot reach.
    for b in 0..12u64 {
        if known[b as usize] {
            continue;
        }
        // A non-antipodal known donor almost always exists (each base has
        // exactly one antipodal partner).  The pathological residue — every
        // candidate ambiguous and the only known seed antipodal to `b` —
        // falls back to the raw winding verdict rather than panicking.
        let Some(d) = (0..12u64).find(|&d| known[d as usize] && !antipodal_base(b, d)) else {
            fill[b as usize] =
                parity_filled_with_id(&centers[b as usize], center_id(0, b), rings, refs);
            known[b as usize] = true;
            continue;
        };
        // Chain arcs between base centres reach ~122°, where the bare
        // two-straddle fast path of `edge_crosses_probe` also fires on the
        // *far* intersection of the two great circles (an edge antipodal to
        // the chord) and silently inverts the seed.  The full symbolic
        // predicate is unconditional here — 11 arcs at most, off every hot
        // path.
        let crossings = edges
            .iter()
            .filter(|e| {
                arcs_cross_sos(
                    &centers[d as usize],
                    &centers[b as usize],
                    &e.a,
                    &e.b,
                    center_id(0, d),
                    center_id(0, b),
                    e.ia,
                    e.ib,
                )
            })
            .count();
        fill[b as usize] = fill[d as usize] ^ (crossings % 2 == 1);
        known[b as usize] = true;
    }
    fill
}

/// Build a base-cell node, or `None` if the base cell is entirely outside the
/// polygon's bounding cap.  Computes the only full O(V) even-odd parity per base.
///
/// `complement` disables the vertex-cap cull: for a hemisphere+ polygon (see
/// [`covers_complement`]) the interior wraps the cap's antipode, so no base cell
/// can be pruned by cap distance — every base is descended and the even-odd fill
/// classifies its interior cells.
///
/// The base seed's `fill` comes precomputed from [`base_fills`] — the winding
/// backend when unambiguous, the SoS parity chain when the centre lies on the
/// boundary (#103) — correct at any polygon size, so the seed is classified
/// the same whether the polygon is a small box or a hemisphere+ ring.
fn base_node(base: u64, edges: &[Edge], fill: bool, cap: &Cap, complement: bool) -> Option<Node> {
    let center = cell_center_vec(0, base);
    let corners = cell_corners(0, base);
    let (cos_cr, sin_cr) = cell_cos_radius(&center, &corners);
    if !complement && cap.excludes(&center, cos_cr, sin_cr) {
        return None;
    }
    let relevant: SmallVec<[usize; 8]> = (0..edges.len())
        .filter(|&i| edge_relevant(&edges[i], &center, cos_cr, sin_cr))
        .collect();
    Some(Node {
        pixel: base,
        depth: 0,
        center,
        corners,
        cos_cr,
        fill,
        relevant,
    })
}

/// Does this cell sit close enough to a pole that its HEALPix edges deviate
/// meaningfully from great-circle arcs?  HEALPix cell edges are not great
/// circles; near the 4-way base junctions (the poles) the true cell bulges
/// outside the 4-corner geodesic quad, so a polygon grazing that bulge would be
/// missed by a corners-only straddle test (issue #32).
///
/// The deviation is governed by the cell's distance to the nearest pole measured
/// in cell widths; HEALPix is self-similar, so at fixed `1 - |z|` it falls ~4×
/// per finer order.  `one_minus_absz` is `1 - max|corner.z|` (smallest for cells
/// hugging a pole).  The threshold reproduces the measured boundary — real
/// misses sit at `1 - |z| ≲ 0.005` (order 6); mid-latitude cells and the 3-way
/// junctions stay well above it — and is cheap (no extra HEALPix calls).
#[inline]
fn near_pole_curved(depth: u8, one_minus_absz: f64) -> bool {
    // `1 - |z|` threshold = min(0.03, 0.02·4^(6-depth)): ~0.02 at order 6, ×4 per
    // coarser order (self-similar curvature), with margin above the measured
    // real-miss boundary (1 - |z| ≈ 0.005 at order 6).  Capped so only the
    // genuinely near-pole regime is densified — NOT the whole polar cap
    // (|z| > 2/3, i.e. |lat| > 41.8°): mid-cap cells (e.g. lat 45–75) have only
    // mild curvature (rel-dev ≲ 0.02) and would otherwise pay for densification
    // they don't need.  The cap means extremely coarse coverage (order ≲ 5) at
    // moderate-high latitude can retain a sub-cell boundary graze gap, far below
    // any practical intersection use.  A `match` (not `powi`) keeps it hot-path
    // cheap.
    let thresh = match depth {
        0..=5 => 0.03,
        6 => 0.02,
        7 => 0.005,
        8 => 0.001_25,
        _ => return false, // sub-degree cells: deviation already negligible
    };
    one_minus_absz < thresh
}

/// Densification step for a near-pole cell's true boundary: coarser cells bulge
/// more, so use a larger step (curvature falls ~quadratically in the step).
#[inline]
fn near_pole_step(depth: u8) -> u32 {
    (1u32 << (7u8.saturating_sub(depth))).clamp(4, 16)
}

/// Does the polygon boundary pass through this cell?  True if a polygon vertex
/// lies in it, a relevant edge crosses **or exactly touches** a cell edge
/// ([`edge_hits_cell_edge`], the #103 closed-set contract), or a relevant edge
/// crosses centre→boundary (a clipped corner/edge).
///
/// The cheap 4-corner geodesic-quad test runs first and settles every solid
/// overlap.  HEALPix cell edges are not great circles, so near the poles the
/// true cell bulges outside the quad; a polygon can *graze* that bulge without
/// crossing the quad (issue #32).  Only when the quad test fails **and** the
/// cell is a near-pole curved cell with a nearby edge do we pay to re-test
/// against the densified true boundary — so the common path is unchanged.
fn node_straddles(node: &Node, edges: &[Edge], order: u8) -> bool {
    // No polygon edge reaches this cell ⇒ the boundary cannot pass through it.
    // (A vertex inside the cell would put its incident edges in `relevant`.)
    // Early-out before fetching corners — interior/empty cells cost nothing.
    if node.relevant.is_empty() {
        return false;
    }

    let shift = 2 * (order - node.depth) as u32;
    // (1) a polygon vertex's leaf cell falls in this cell — exact via HEALPix.
    if node
        .relevant
        .iter()
        .any(|&i| edges[i].leaf >> shift == node.pixel)
    {
        #[cfg(feature = "descent-stats")]
        descent_stats::set_cause(descent_stats::Cause::VertexLeaf);
        return true;
    }

    // (2) cheap 4-corner geodesic-quad straddle test (the common path).  Corners
    // are cached on the node; precompute the four cell-edge normals once so each
    // edge-vs-edge test is dot-products only.
    let corners = &node.corners;
    let n_quad: [Vec3; 4] = [
        cross(&corners[0], &corners[1]),
        cross(&corners[1], &corners[2]),
        cross(&corners[2], &corners[3]),
        cross(&corners[3], &corners[0]),
    ];
    if node.relevant.iter().any(|&i| {
        let e = &edges[i];
        (0..4).any(|ci| edge_hits_cell_edge(e, &corners[ci], &corners[(ci + 1) % 4], &n_quad[ci]))
    }) {
        // The short-circuiting loops stop at the deciding hit, so under
        // `descent-stats` its touch/cross tag is still current here.
        #[cfg(feature = "descent-stats")]
        descent_stats::set_cause(descent_stats::quad_cause());
        return true;
    }
    {
        let cid = center_id(node.depth, node.pixel);
        if corners
            .iter()
            .any(|c| arc_crossing_parity(&node.center, c, cid, CORNER_ID_A, &node.relevant, edges))
        {
            #[cfg(feature = "descent-stats")]
            descent_stats::set_cause(descent_stats::Cause::CornerParity);
            return true;
        }
    }

    // (3) near-pole bulge graze: re-test against the true (curved) boundary, but
    // only for the few near-pole cells the quad missed (#32).
    let one_minus_absz = 1.0 - corners.iter().fold(0.0_f64, |m, c| m.max(c[2].abs()));
    if !near_pole_curved(node.depth, one_minus_absz) {
        return false;
    }
    let bnd = boundaries_step_scalar(node.depth, node.pixel, near_pole_step(node.depth));
    let n = bnd.len();
    let n_bnd: Vec<Vec3> = (0..n)
        .map(|ci| cross(&bnd[ci], &bnd[(ci + 1) % n]))
        .collect();
    let bulge_hit = node.relevant.iter().any(|&i| {
        let e = &edges[i];
        (0..n).any(|ci| edge_hits_cell_edge(e, &bnd[ci], &bnd[(ci + 1) % n], &n_bnd[ci]))
    }) || {
        let cid = center_id(node.depth, node.pixel);
        bnd.iter()
            .any(|b| arc_crossing_parity(&node.center, b, cid, CORNER_ID_A, &node.relevant, edges))
    };
    #[cfg(feature = "descent-stats")]
    if bulge_hit {
        descent_stats::set_cause(descent_stats::Cause::NearPoleBulge);
    }
    bulge_hit
}

/// The four children of a node, each with re-culled edges and propagated fill.
fn node_children(node: &Node, edges: &[Edge]) -> Vec<Node> {
    (0..4u64)
        .map(|ch| {
            let pixel = node.pixel * 4 + ch;
            let depth = node.depth + 1;
            let corners = cell_corners(depth, pixel);
            let center = cell_center_vec(depth, pixel);
            let (cos_cr, sin_cr) = cell_cos_radius(&center, &corners);
            let relevant: SmallVec<[usize; 8]> = node
                .relevant
                .iter()
                .copied()
                .filter(|&i| edge_relevant(&edges[i], &center, cos_cr, sin_cr))
                .collect();
            let fill = node.fill
                ^ arc_crossing_parity(
                    &node.center,
                    &center,
                    center_id(node.depth, node.pixel),
                    center_id(depth, pixel),
                    &node.relevant,
                    edges,
                );
            Node {
                pixel,
                depth,
                center,
                corners,
                cos_cr,
                fill,
                relevant,
            }
        })
        .collect()
}

/// A cell's angular radius (centre→corner), in radians.
fn node_radius(node: &Node) -> f64 {
    node.cos_cr.clamp(-1.0, 1.0).acos()
}

/// Top-down descent producing the covering cells as `(nested_pixel, depth)`.
///
/// Each node carries the subset of polygon edges whose caps reach the cell and
/// the cell centre's even-odd fill state, both maintained in `O(local edges)`:
/// children inherit the parent's edge subset (re-culled) and fill (flipped by
/// the parity of edges crossing the short centre→centre arc).  Only the 12 base
/// cells pay one full even-odd parity each; a cell with no nearby edge is
/// uniform — kept whole if filled, pruned if empty, with no PIP call.
///
/// Stop rule: refine straddle cells until `depth == order` (exact), or — if set
/// — until a cell's radius ≤ `tolerance` (approximate, coarser boundary), or
/// until a `max_cells` budget is hit (best-first, adaptive boundary).
///
/// Parallel stack-DFS over the 12 base subtrees (fixed-order, optional
/// tolerance).  Deterministic — the merged result is order-independent.
fn descend_parallel(
    rings: &[Vec<Vec3>],
    refs: &RingRefs,
    order: u8,
    tolerance: Option<f64>,
) -> Vec<(u64, u8)> {
    let edges = build_edges(rings, order);
    let cap = Cap::of_rings(rings);
    // One `RingRef` per ring for the whole descent, supplied by ingest: deriving
    // one is O(V), so rebuilding it per probe would make every cell centre O(V).
    let complement = covers_complement(rings, &cap, refs);
    let fills = base_fills(&edges, rings, &cap, complement, refs);

    (0..12u64)
        .into_par_iter()
        .flat_map_iter(|base| {
            let mut out: Vec<(u64, u8)> = Vec::new();
            let Some(seed) = base_node(base, &edges, fills[base as usize], &cap, complement) else {
                return out;
            };
            let mut stack = vec![seed];
            while let Some(node) = stack.pop() {
                if node_straddles(&node, &edges, order) {
                    #[cfg(feature = "descent-stats")]
                    let cause = descent_stats::take_cause();
                    let stop = node.depth >= order
                        || tolerance.is_some_and(|tol| node_radius(&node) <= tol);
                    if stop {
                        #[cfg(feature = "descent-stats")]
                        descent_stats::record_leaf(&node, cause);
                        out.push((node.pixel, node.depth));
                    } else {
                        #[cfg(feature = "descent-stats")]
                        descent_stats::record_internal(cause);
                        stack.extend(node_children(&node, &edges));
                    }
                } else {
                    // #103 parity oracle: a uniform (boundary-free) cell's
                    // incremental even-odd fill must agree with the independent
                    // winding PIP at its centre; a mismatch means a probe-chain
                    // parity flip upstream.  Debug builds only — this is the
                    // assert that would have caught #11, #78, and #103 at once.
                    // The oracle probes under the centre's own `center_id`
                    // (issue #107 phase 3), so chain and oracle share one
                    // perturbed world even at exact boundary coincidences.
                    debug_assert_eq!(
                        node.fill,
                        parity_filled_with_id(
                            &node.center,
                            center_id(node.depth, node.pixel),
                            rings,
                            refs
                        ),
                        "parity oracle diverged at uniform cell ({}, {})",
                        node.pixel,
                        node.depth
                    );
                    if node.fill {
                        out.push((node.pixel, node.depth));
                    }
                }
            }
            out
        })
        .collect()
}

/// Best-first descent: refine the largest (coarsest) straddle cell first until
/// the cell count reaches `max_cells`, then emit the remaining frontier coarse.
/// Single-threaded (the priority frontier is global) but bounded by the budget.
///
/// Returns `(cells, effective_budget)`.  The budget is floored at the base-level
/// cover size — fewer cells than that cannot represent the polygon — so a
/// too-low `max_cells` is raised and the larger `effective_budget` reported.
fn descend_best_first(
    rings: &[Vec<Vec3>],
    refs: &RingRefs,
    order: u8,
    max_cells: usize,
) -> (Vec<(u64, u8)>, usize) {
    let edges = build_edges(rings, order);
    let cap = Cap::of_rings(rings);
    let complement = covers_complement(rings, &cap, refs);
    let fills = base_fills(&edges, rings, &cap, complement, refs);

    let mut out: Vec<(u64, u8)> = Vec::new();
    let mut frontier: BinaryHeap<HeapNode> = BinaryHeap::new();

    for base in 0..12u64 {
        if let Some(node) = base_node(base, &edges, fills[base as usize], &cap, complement) {
            consider_node(node, &edges, rings, order, refs, &mut out, &mut frontier);
        }
    }

    // Floor: a budget too small to even refine the base frontier once yields a
    // degenerate cover.  Require room for one refinement pass (each base
    // straddle cell → 4 children); below that, raise it so the caller can warn.
    let floor = out.len() + 4 * frontier.len();
    let budget = max_cells.max(floor);

    while let Some(HeapNode(node)) = frontier.pop() {
        // Splitting trades 1 cell for up to 4; stop before exceeding the budget.
        if out.len() + frontier.len() + 1 >= budget {
            out.push((node.pixel, node.depth));
            out.extend(frontier.drain().map(|HeapNode(n)| (n.pixel, n.depth)));
            break;
        }
        for child in node_children(&node, &edges) {
            consider_node(child, &edges, rings, order, refs, &mut out, &mut frontier);
        }
    }
    (out, budget)
}

/// Route a node into the output (uniform interior, or finest boundary leaf) or
/// onto the frontier (a straddle cell still above `order`).
fn consider_node(
    node: Node,
    edges: &[Edge],
    rings: &[Vec<Vec3>],
    order: u8,
    refs: &RingRefs,
    out: &mut Vec<(u64, u8)>,
    frontier: &mut BinaryHeap<HeapNode>,
) {
    if node_straddles(&node, edges, order) {
        #[cfg(feature = "descent-stats")]
        let cause = descent_stats::take_cause();
        if node.depth >= order {
            #[cfg(feature = "descent-stats")]
            descent_stats::record_leaf(&node, cause);
            out.push((node.pixel, node.depth));
        } else {
            #[cfg(feature = "descent-stats")]
            descent_stats::record_internal(cause);
            frontier.push(HeapNode(node));
        }
    } else {
        // #103 parity oracle — see descend_parallel.
        debug_assert_eq!(
            node.fill,
            parity_filled_with_id(&node.center, center_id(node.depth, node.pixel), rings, refs),
            "parity oracle diverged at uniform cell ({}, {})",
            node.pixel,
            node.depth
        );
        if node.fill {
            out.push((node.pixel, node.depth));
        }
    }
}

/// Heap wrapper ordering nodes so the **coarsest** (smallest depth) is popped
/// first, ties broken by pixel for determinism.
struct HeapNode(Node);
impl PartialEq for HeapNode {
    fn eq(&self, other: &Self) -> bool {
        self.0.depth == other.0.depth && self.0.pixel == other.0.pixel
    }
}
impl Eq for HeapNode {}
impl Ord for HeapNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Smaller depth = larger cell = greater priority; then smaller pixel.
        other
            .0
            .depth
            .cmp(&self.0.depth)
            .then_with(|| other.0.pixel.cmp(&self.0.pixel))
    }
}
impl PartialOrd for HeapNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ── great-circle helpers (shared with `linestring`) ──────────────────────

/// Great-circle distance in radians (Haversine formula).
pub(crate) fn great_circle_distance_rad(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (la1, lo1) = (lat1.to_radians(), lon1.to_radians());
    let (la2, lo2) = (lat2.to_radians(), lon2.to_radians());
    let dlat = (la2 - la1) * 0.5;
    let dlon = (lo2 - lo1) * 0.5;
    let a = dlat.sin().powi(2) + la1.cos() * la2.cos() * dlon.sin().powi(2);
    2.0 * a.sqrt().asin()
}

/// Angular resolution of one HEALPix cell at the given depth (radians).
pub(crate) fn cell_resolution_rad(depth: u8) -> f64 {
    let nside = (1u64 << depth) as f64;
    (PI / 3.0).sqrt() / nside
}

/// Invoke `f(lat, lon)` for each of the `n` interior points along the
/// great-circle arc (endpoints excluded), in order.  Allocation-free — callers
/// that only stream the points (e.g. linestring rasterization) avoid the
/// per-segment `Vec` that [`interpolate_great_circle`] would build.
pub(crate) fn for_each_great_circle_point(
    lat1: f64,
    lon1: f64,
    lat2: f64,
    lon2: f64,
    n: usize,
    mut f: impl FnMut(f64, f64),
) {
    if n == 0 {
        return;
    }

    let (la1, lo1) = (lat1.to_radians(), lon1.to_radians());
    let (la2, lo2) = (lat2.to_radians(), lon2.to_radians());
    let (c1, s1) = (la1.cos(), la1.sin());
    let (c2, s2) = (la2.cos(), la2.sin());
    let (x1, y1, z1) = (c1 * lo1.cos(), c1 * lo1.sin(), s1);
    let (x2, y2, z2) = (c2 * lo2.cos(), c2 * lo2.sin(), s2);

    let dot = (x1 * x2 + y1 * y2 + z1 * z2).clamp(-1.0, 1.0);
    let omega = dot.acos();
    // Threshold: ~1e-6 rad ≈ 6 m on Earth — well below any useful cell size
    if omega < 1e-6 {
        return;
    }
    let sin_omega = omega.sin();

    for k in 1..=n {
        let f_ = k as f64 / (n + 1) as f64;
        let a = ((1.0 - f_) * omega).sin() / sin_omega;
        let b = (f_ * omega).sin() / sin_omega;
        let x = a * x1 + b * x2;
        let y = a * y1 + b * y2;
        let z = a * z1 + b * z2;
        let lat = z.asin().to_degrees();
        let lon = y.atan2(x).to_degrees();
        f(lat, lon);
    }
}

/// Interpolate `n` interior points along the great-circle arc, collected into a
/// `Vec` (endpoints excluded).  Thin wrapper over [`for_each_great_circle_point`]
/// for callers that need the materialized points (the descent tests).
#[cfg(test)]
pub(crate) fn interpolate_great_circle(
    lat1: f64,
    lon1: f64,
    lat2: f64,
    lon2: f64,
    n: usize,
) -> Vec<(f64, f64)> {
    let mut pts = Vec::with_capacity(n);
    for_each_great_circle_point(lat1, lon1, lat2, lon2, n, |lat, lon| pts.push((lat, lon)));
    pts
}

// ── tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;

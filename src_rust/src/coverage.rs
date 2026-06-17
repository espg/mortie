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
//! Vertex order is meaningful. mortie follows the RFC 7946 §3.1.6 / S2
//! **right-hand rule**: exterior rings counter-clockwise (interior on the left),
//! holes clockwise. The single robust winding backend
//! ([`crate::sphere::parity_filled_robust`], #22) that handles hemisphere-plus
//! rings *requires* this convention — past a hemisphere a ring's two sides have
//! equal standing, so only the winding direction disambiguates which is
//! interior. As a convenience, [`build_ring`] auto-corrects a **sub-hemisphere**
//! ring wound the wrong way (cheap signed-winding check, where the smaller side
//! is unambiguously the interior); hemisphere-plus rings are never reordered, so
//! supply those CCW (holes CW) exactly as the right-hand rule prescribes.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::f64::consts::PI;

use rayon::prelude::*;
use smallvec::SmallVec;

use crate::cell_geom::{cell_center_vec, cell_corners, Cap};
use crate::geo2mort::{ang2pix_scalar, boundaries_step_scalar};
use crate::morton::nested2mort;
use crate::sphere::{
    arcs_cross_n, cross, dot, latlon_to_unit_vec, norm, normalize, parity_filled_robust,
    ring_winding_sign, robust_crossing, PointId, Vec3,
};

// ── public entry points ──────────────────────────────────────────────────

/// Compute morton indices covering a polygon, as a flat list at `order`.
///
/// # Arguments
/// * `lats` — vertex latitudes in degrees
/// * `lons` — vertex longitudes in degrees
/// * `order` — HEALPix depth (1–18)
///
/// # Returns
/// Sorted unique `Vec<i64>` of morton indices at `order` whose cells intersect
/// the closed polygon (contract (a): the cover is a superset of the polygon).
///
/// # Panics
/// * If `lats`/`lons` differ in length, fewer than 3 vertices, or order ∉ 1–18.
pub fn polygon_to_morton_coverage(lats: &[f64], lons: &[f64], order: u8) -> Vec<i64> {
    let moc = polygon_descend(lats, lons, order, None);
    crate::moc::to_order(&moc, order)
}

/// Compute polygon coverage as a compact, normalized Multi-Order Coverage map:
/// coarse cells for the interior, fine cells (at `order`) along the boundary.
pub fn polygon_to_morton_moc(lats: &[f64], lons: &[f64], order: u8) -> Vec<i64> {
    let moc = polygon_descend(lats, lons, order, None);
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
) -> Vec<i64> {
    let moc = polygon_descend(lats, lons, order, Some(tolerance));
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
) -> (Vec<i64>, usize) {
    assert_eq!(
        lats.len(),
        lons.len(),
        "lats and lons must have same length"
    );
    assert!(lats.len() >= 3, "Need at least 3 vertices for a polygon");
    assert!((1..=18).contains(&order), "Order must be 1-18");

    let rings = vec![build_ring(lats, lons)];
    let (nodes, effective) = descend_best_first(&rings, order, max_cells);
    let moc: Vec<i64> = nodes.iter().map(|&(p, d)| nested2mort(p, d)).collect();
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
) -> Vec<i64> {
    validate_multi(lats, lons, order);
    let rings = build_rings(lats, lons);
    let moc = nodes_to_morton(&descend_parallel(&rings, order, None));
    crate::moc::to_order(&moc, order)
}

/// MOC coverage of a ring-set, with optional `tolerance` or `max_cells` stop.
/// Returns `(cells, effective_budget)` (see [`polygon_to_morton_moc_budget`]).
pub fn multipolygon_to_morton_moc(
    lats: &[Vec<f64>],
    lons: &[Vec<f64>],
    order: u8,
    tolerance: Option<f64>,
    max_cells: Option<usize>,
) -> (Vec<i64>, usize) {
    validate_multi(lats, lons, order);
    let rings = build_rings(lats, lons);
    if let Some(budget) = max_cells {
        let (nodes, effective) = descend_best_first(&rings, order, budget);
        (crate::moc::normalize(&nodes_to_morton(&nodes)), effective)
    } else {
        let nodes = descend_parallel(&rings, order, tolerance);
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
    assert!((1..=18).contains(&order), "Order must be 1-18");
    for (la, lo) in lats.iter().zip(lons.iter()) {
        assert_eq!(la.len(), lo.len(), "ring lats/lons length mismatch");
        assert!(la.len() >= 3, "each ring needs at least 3 vertices");
    }
}

fn build_rings(lats: &[Vec<f64>], lons: &[Vec<f64>]) -> Vec<Vec<Vec3>> {
    lats.iter()
        .zip(lons.iter())
        .map(|(la, lo)| build_ring(la, lo))
        .collect()
}

#[inline]
fn nodes_to_morton(nodes: &[(u64, u8)]) -> Vec<i64> {
    nodes.iter().map(|&(p, d)| nested2mort(p, d)).collect()
}

// ── descent ──────────────────────────────────────────────────────────────

/// Validate inputs, build the ring-set, run the descent, and return its cells
/// as (un-normalized, mixed-order) morton indices.
fn polygon_descend(lats: &[f64], lons: &[f64], order: u8, tolerance: Option<f64>) -> Vec<i64> {
    assert_eq!(
        lats.len(),
        lons.len(),
        "lats and lons must have same length"
    );
    assert!(lats.len() >= 3, "Need at least 3 vertices for a polygon");
    assert!((1..=18).contains(&order), "Order must be 1-18");

    let rings = vec![build_ring(lats, lons)];
    descend_parallel(&rings, order, tolerance)
        .iter()
        .map(|&(pixel, depth)| nested2mort(pixel, depth))
        .collect()
}

/// Convert lat/lon vertices to a closed ring of unit vectors, dropping a
/// duplicate closing vertex if present, then **normalize its orientation** to
/// the module's RFC 7946 §3.1.6 / S2 right-hand-rule contract (interior to the
/// **left** of the directed edges: CCW exterior, CW holes).
///
/// # Orientation normalization (Phase 3, #22)
///
/// The single robust point-in-ring path ([`parity_filled_robust`]) is
/// winding-direction-aware: it treats the region to the *left* of the directed
/// edges as interior, so a ring wound the wrong way selects the complementary
/// region.  To keep everyday (often clockwise) input from silently inverting we
/// auto-correct **only rings whose vertices fit within a hemisphere**, where the
/// two regions split into an unambiguous "smaller" and "larger" side and the
/// smaller one is the intended interior:
///
/// * **Sub-hemisphere vertices** (the ring's vertex bounding cap has radius
///   `< 90°`): read the winding sign at the cap axis ([`ring_winding_sign`]) —
///   `+1` CCW, `-1` CW — and reverse a CW ring so the **smaller** region ends up
///   on the left (CCW).  One cheap O(V) signed-winding pass, done once.  This is
///   the standard GIS "smaller-area-is-interior" behaviour for ordinary
///   polygons, so CW and CCW spellings of the same box give the same cover.
/// * **Hemisphere+ vertices** (cap radius `≥ 90°`, or a balanced vertex sum): we
///   **never** reorder.  Past a hemisphere the two sides have equal standing, so
///   winding *magnitude* cannot pick the interior — only the vertex order can,
///   and it is trusted exactly as supplied (reversing here provably picks the
///   wrong side).
///
/// Consequence for the hemisphere+ feature (#22): a region whose *interior* is
/// larger than a hemisphere but whose *boundary vertices* still fit in one (the
/// classic "everything except Antarctica", whose Antarctica-hugging ring sits in
/// a sub-hemisphere cap) must be expressed the way GeoJSON authors it anyway — a
/// whole-world outer ring with a small hole, or vertices that genuinely span
/// `> 90°` — not as a lone sub-hemisphere-vertex ring relying on reversed
/// winding, which ingest would normalize back to the small side.
fn build_ring(lats: &[f64], lons: &[f64]) -> Vec<Vec3> {
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
    normalize_ring_orientation(&mut ring);
    ring
}

/// Auto-correct a **sub-hemisphere** ring wound clockwise to the right-hand-rule
/// (CCW, interior-on-the-left) convention; leave hemisphere+ rings untouched.
/// See [`build_ring`] for the rationale.
fn normalize_ring_orientation(ring: &mut [Vec3]) {
    if ring.len() < 3 {
        return;
    }
    // Bounding cap of this ring's vertices: axis = normalized vertex sum,
    // radius = max angular distance to a vertex.  A radius ≥ 90° (or a balanced
    // sum) means the ring is not sub-hemisphere, so orientation must be trusted.
    let mut s = [0.0, 0.0, 0.0];
    for v in ring.iter() {
        s[0] += v[0];
        s[1] += v[1];
        s[2] += v[2];
    }
    if norm(&s) < 1e-12 {
        return; // balanced ⇒ hemisphere+; never normalize
    }
    let axis = normalize(&s);
    let radius = ring
        .iter()
        .map(|v| dot(&axis, v).clamp(-1.0, 1.0).acos())
        .fold(0.0_f64, f64::max);
    if radius >= std::f64::consts::FRAC_PI_2 {
        return; // hemisphere+ ⇒ winding magnitude can't pick the interior side
    }
    // Sub-hemisphere: reverse a clockwise ring so the small side is on the left.
    if ring_winding_sign(ring) < 0 {
        ring.reverse();
    }
}

/// A polygon edge as a great-circle arc `a→b`, with a bounding cap (`mid`,
/// angular half-length `rho`) for culling and the leaf cell of endpoint `a`
/// (at the target order) for exact vertex-in-cell tests.
struct Edge {
    a: Vec3,
    b: Vec3,
    /// Great-circle normal `a × b`, precomputed once so the per-cell crossing
    /// tests reduce to dot products (see [`arcs_cross_n`]).
    n_ab: Vec3,
    mid: Vec3,
    cos_rho: f64,
    sin_rho: f64,
    leaf: u64,
    /// Stable Simulation-of-Simplicity identities of the endpoints `a`, `b`
    /// (their global vertex index across the ring-set).  Feed the descent's
    /// robust crossing test ([`robust_crossing`]) so a probe arc whose endpoint
    /// lies exactly on this edge's great circle — e.g. a HEALPix base-cell centre
    /// on a base-cell-centre meridian (issue #11) — resolves to a definite,
    /// traversal-order-independent side instead of a sign-unstable `arcs_cross_n`.
    ia: PointId,
    ib: PointId,
}

/// SoS identities reserved for the two endpoints of a descent **probe** arc
/// (cell centre → centre/corner).  Ring vertices are numbered from
/// [`VERTEX_ID_BASE`] upward, so the probe endpoints take ids `0` and `1` — i.e.
/// they sort **before** every ring vertex.  This matters when a probe endpoint
/// lies exactly on an edge's great circle (the issue #11 base-cell-centre-on-a-
/// meridian degeneracy): SoS resolves the tie by perturbing the lowest-id point
/// first, so the probe point — not a ring vertex — is the one nudged decisively
/// off the edge, and it is nudged the **same** way every time it anchors a walk.
/// That keeps the descent's incremental fill parity consistent with itself.
const PROBE_ID_P: PointId = 0;
const PROBE_ID_Q: PointId = 1;
/// Ring vertices are numbered from here so their ids never collide with the two
/// reserved probe ids above.
const VERTEX_ID_BASE: PointId = 2;

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
/// The crossing test goes through the SoS-robust [`robust_crossing`]: the
/// descent walks `fill` from a cell centre to a neighbour, and when an edge's
/// great circle passes exactly through one endpoint (the issue #11 degeneracy —
/// a base-cell centre sitting on a base-cell-centre meridian), a plain
/// `arcs_cross_n` is sign-unstable and flip-flops the parity, flooding the
/// cell's whole subtree.  Simulation of Simplicity breaks that exact-zero
/// orientation to a definite, traversal-order-independent side, so the parity is
/// stable.  This runs on the descent's per-cell fan, but only over the `relevant`
/// edges (those whose cap reaches the cell), so it stays off the bulk path.
#[inline]
fn arc_crossing_parity(p: &Vec3, q: &Vec3, relevant: &[usize], edges: &[Edge]) -> bool {
    let mut crossings = 0u32;
    for &i in relevant {
        let e = &edges[i];
        if robust_crossing(p, q, &e.a, &e.b, PROBE_ID_P, PROBE_ID_Q, e.ia, e.ib) {
            crossings += 1;
        }
    }
    crossings & 1 == 1
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
fn covers_complement(rings: &[Vec<Vec3>], cap: &Cap) -> bool {
    let antipode = [-cap.axis[0], -cap.axis[1], -cap.axis[2]];
    if parity_filled_robust(&antipode, rings) {
        return true;
    }
    (0..12u64).any(|base| {
        let center = cell_center_vec(0, base);
        let corners = cell_corners(0, base);
        let (cos_cr, _) = cell_cos_radius(&center, &corners);
        let cr = cos_cr.clamp(-1.0, 1.0).acos();
        dot(&cap.axis, &center).clamp(-1.0, 1.0).acos() > cap.radius + cr
            && parity_filled_robust(&center, rings)
    })
}

/// Build a base-cell node, or `None` if the base cell is entirely outside the
/// polygon's bounding cap.  Computes the only full O(V) even-odd parity per base.
///
/// `complement` disables the vertex-cap cull: for a hemisphere+ polygon (see
/// [`covers_complement`]) the interior wraps the cap's antipode, so no base cell
/// can be pruned by cap distance — every base is descended and the even-odd fill
/// classifies its interior cells.
///
/// The base seed's fill state is the only full O(V) point-in-polygon evaluation
/// per base cell, and it goes through the single robust winding backend
/// ([`parity_filled_robust`], #22) — correct at any polygon size, so the seed is
/// classified the same whether the polygon is a small box or a hemisphere+ ring.
fn base_node(
    base: u64,
    edges: &[Edge],
    rings: &[Vec<Vec3>],
    cap: &Cap,
    complement: bool,
) -> Option<Node> {
    let center = cell_center_vec(0, base);
    let corners = cell_corners(0, base);
    let (cos_cr, sin_cr) = cell_cos_radius(&center, &corners);
    let cr = cos_cr.clamp(-1.0, 1.0).acos();
    if !complement && dot(&cap.axis, &center).clamp(-1.0, 1.0).acos() > cap.radius + cr {
        return None;
    }
    let relevant: SmallVec<[usize; 8]> = (0..edges.len())
        .filter(|&i| edge_relevant(&edges[i], &center, cos_cr, sin_cr))
        .collect();
    let fill = parity_filled_robust(&center, rings);
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
/// lies in it, a relevant edge crosses a cell edge, or a relevant edge crosses
/// centre→boundary (a clipped corner/edge).
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
        return true;
    }

    // (2) cheap 4-corner geodesic-quad straddle test (the common path).  Corners
    // are cached on the node; precompute the four cell-edge normals once so each
    // edge-vs-edge test is dot-products only (`arcs_cross_n`).
    let corners = &node.corners;
    let n_quad: [Vec3; 4] = [
        cross(&corners[0], &corners[1]),
        cross(&corners[1], &corners[2]),
        cross(&corners[2], &corners[3]),
        cross(&corners[3], &corners[0]),
    ];
    let quad_straddles = node.relevant.iter().any(|&i| {
        let e = &edges[i];
        (0..4).any(|ci| {
            arcs_cross_n(
                &e.a,
                &e.b,
                &e.n_ab,
                &corners[ci],
                &corners[(ci + 1) % 4],
                &n_quad[ci],
            )
        })
    }) || corners
        .iter()
        .any(|c| arc_crossing_parity(&node.center, c, &node.relevant, edges));
    if quad_straddles {
        return true;
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
    node.relevant.iter().any(|&i| {
        let e = &edges[i];
        (0..n).any(|ci| {
            arcs_cross_n(
                &e.a,
                &e.b,
                &e.n_ab,
                &bnd[ci],
                &bnd[(ci + 1) % n],
                &n_bnd[ci],
            )
        })
    }) || bnd
        .iter()
        .any(|b| arc_crossing_parity(&node.center, b, &node.relevant, edges))
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
            let fill =
                node.fill ^ arc_crossing_parity(&node.center, &center, &node.relevant, edges);
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
fn descend_parallel(rings: &[Vec<Vec3>], order: u8, tolerance: Option<f64>) -> Vec<(u64, u8)> {
    let edges = build_edges(rings, order);
    let cap = Cap::of_rings(rings);
    let complement = covers_complement(rings, &cap);

    (0..12u64)
        .into_par_iter()
        .flat_map_iter(|base| {
            let mut out: Vec<(u64, u8)> = Vec::new();
            let Some(seed) = base_node(base, &edges, rings, &cap, complement) else {
                return out;
            };
            let mut stack = vec![seed];
            while let Some(node) = stack.pop() {
                if node_straddles(&node, &edges, order) {
                    let stop = node.depth >= order
                        || tolerance.is_some_and(|tol| node_radius(&node) <= tol);
                    if stop {
                        out.push((node.pixel, node.depth));
                    } else {
                        stack.extend(node_children(&node, &edges));
                    }
                } else if node.fill {
                    out.push((node.pixel, node.depth));
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
fn descend_best_first(rings: &[Vec<Vec3>], order: u8, max_cells: usize) -> (Vec<(u64, u8)>, usize) {
    let edges = build_edges(rings, order);
    let cap = Cap::of_rings(rings);
    let complement = covers_complement(rings, &cap);

    let mut out: Vec<(u64, u8)> = Vec::new();
    let mut frontier: BinaryHeap<HeapNode> = BinaryHeap::new();

    for base in 0..12u64 {
        if let Some(node) = base_node(base, &edges, rings, &cap, complement) {
            consider_node(node, &edges, order, &mut out, &mut frontier);
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
            consider_node(child, &edges, order, &mut out, &mut frontier);
        }
    }
    (out, budget)
}

/// Route a node into the output (uniform interior, or finest boundary leaf) or
/// onto the frontier (a straddle cell still above `order`).
fn consider_node(
    node: Node,
    edges: &[Edge],
    order: u8,
    out: &mut Vec<(u64, u8)>,
    frontier: &mut BinaryHeap<HeapNode>,
) {
    if node_straddles(&node, edges, order) {
        if node.depth >= order {
            out.push((node.pixel, node.depth));
        } else {
            frontier.push(HeapNode(node));
        }
    } else if node.fill {
        out.push((node.pixel, node.depth));
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

/// Interpolate `n` interior points along the great-circle arc.
/// Does not include the endpoints.
pub(crate) fn interpolate_great_circle(
    lat1: f64,
    lon1: f64,
    lat2: f64,
    lon2: f64,
    n: usize,
) -> Vec<(f64, f64)> {
    if n == 0 {
        return Vec::new();
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
        return Vec::new();
    }
    let sin_omega = omega.sin();

    let mut pts = Vec::with_capacity(n);
    for k in 1..=n {
        let f = k as f64 / (n + 1) as f64;
        let a = ((1.0 - f) * omega).sin() / sin_omega;
        let b = (f * omega).sin() / sin_omega;
        let x = a * x1 + b * x2;
        let y = a * y1 + b * y2;
        let z = a * z1 + b * z2;
        let lat = z.asin().to_degrees();
        let lon = y.atan2(x).to_degrees();
        pts.push((lat, lon));
    }
    pts
}

// ── tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;

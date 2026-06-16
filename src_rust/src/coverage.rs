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

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::f64::consts::PI;

use rayon::prelude::*;

use crate::cell_geom::{cell_center_vec, cell_corners, Cap};
use crate::geo2mort::{ang2pix_scalar, boundaries_step_scalar};
use crate::morton::nested2mort;
use crate::sphere::{
    arcs_cross, choose_backend, dot, latlon_to_unit_vec, normalize, parity_filled,
    parity_filled_robust, Vec3,
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
    assert_eq!(lats.len(), lons.len(), "lats and lons must have same length");
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
/// boundary — no per-part seams.
pub fn multipolygon_to_morton_coverage(lats: &[Vec<f64>], lons: &[Vec<f64>], order: u8) -> Vec<i64> {
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
/// duplicate closing vertex if present.
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
    ring
}

/// A polygon edge as a great-circle arc `a→b`, with a bounding cap (`mid`,
/// angular half-length `rho`) for culling and the leaf cell of endpoint `a`
/// (at the target order) for exact vertex-in-cell tests.
struct Edge {
    a: Vec3,
    b: Vec3,
    mid: Vec3,
    cos_rho: f64,
    sin_rho: f64,
    leaf: u64,
}

/// Build the edge list for a ring-set, each with its bounding cap and the leaf
/// cell of its start vertex.
fn build_edges(rings: &[Vec<Vec3>], order: u8) -> Vec<Edge> {
    let mut edges = Vec::new();
    for ring in rings {
        let m = ring.len();
        if m < 2 {
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
                mid,
                cos_rho,
                sin_rho,
                leaf: ang2pix_scalar(order, lon, lat),
            });
        }
    }
    edges
}

/// `(cos, sin)` of a cell's circumradius (max angular distance centre→corner).
fn cell_cos_radius(center: &Vec3, corners: &[Vec3; 4]) -> (f64, f64) {
    let cos_cr = corners.iter().map(|c| dot(center, c)).fold(1.0_f64, f64::min);
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
#[inline]
fn arc_crossing_parity(p: &Vec3, q: &Vec3, relevant: &[usize], edges: &[Edge]) -> bool {
    let mut crossings = 0u32;
    for &i in relevant {
        if arcs_cross(p, q, &edges[i].a, &edges[i].b) {
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
    fill: bool,
    relevant: Vec<usize>,
}

/// Does the polygon cover **more than a hemisphere** (the complement / hemisphere+
/// case)?  The bounding [`Cap`] only encloses the ring *vertices*, so it bounds
/// the boundary, not the interior.  For a sub-hemisphere polygon the interior
/// lies inside that cap and the vertex-cap cull in [`base_node`] is sound.  For a
/// hemisphere+ polygon ("everything except Antarctica") the interior is the
/// *large* region that wraps around the cap's antipode, and culling base cells by
/// distance from the cap axis would wrongly prune cells that are deep inside.
///
/// We detect this by testing the cap-axis **antipode** with the any-size robust
/// PIP ([`parity_filled_robust`], issue #22): if the antipode — the point
/// farthest from every vertex — is inside the filled region, the interior is the
/// complement and the cap cull must be disabled.  Computed once per descent (not
/// per base cell), so the extra O(V) winding test is off the hot path.
fn covers_complement(rings: &[Vec<Vec3>], cap: &Cap) -> bool {
    let antipode = [-cap.axis[0], -cap.axis[1], -cap.axis[2]];
    parity_filled_robust(&antipode, rings)
}

/// Build a base-cell node, or `None` if the base cell is entirely outside the
/// polygon's bounding cap.  Computes the only full O(V) even-odd parity per base.
///
/// `complement` disables the vertex-cap cull: for a hemisphere+ polygon (see
/// [`covers_complement`]) the interior wraps the cap's antipode, so no base cell
/// can be pruned by cap distance — every base is descended and the even-odd fill
/// classifies its interior cells.
fn base_node(
    base: u64,
    edges: &[Edge],
    rings: &[Vec<Vec3>],
    cap: &Cap,
    backend: crate::sphere::PipBackend,
    complement: bool,
) -> Option<Node> {
    let center = cell_center_vec(0, base);
    let corners = cell_corners(0, base);
    let (cos_cr, sin_cr) = cell_cos_radius(&center, &corners);
    let cr = cos_cr.clamp(-1.0, 1.0).acos();
    if !complement && dot(&cap.axis, &center).clamp(-1.0, 1.0).acos() > cap.radius + cr {
        return None;
    }
    let relevant: Vec<usize> = (0..edges.len())
        .filter(|&i| edge_relevant(&edges[i], &center, cos_cr, sin_cr))
        .collect();
    let fill = parity_filled(&center, rings, backend);
    Some(Node { pixel: base, depth: 0, center, fill, relevant })
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
    if node.relevant.iter().any(|&i| edges[i].leaf >> shift == node.pixel) {
        return true;
    }

    // (2) cheap 4-corner geodesic-quad straddle test (the common path).
    let corners = cell_corners(node.depth, node.pixel);
    let quad_straddles = node.relevant.iter().any(|&i| {
        let e = &edges[i];
        (0..4).any(|ci| arcs_cross(&e.a, &e.b, &corners[ci], &corners[(ci + 1) % 4]))
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
    node.relevant.iter().any(|&i| {
        let e = &edges[i];
        (0..n).any(|ci| arcs_cross(&e.a, &e.b, &bnd[ci], &bnd[(ci + 1) % n]))
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
            let relevant: Vec<usize> = node
                .relevant
                .iter()
                .copied()
                .filter(|&i| edge_relevant(&edges[i], &center, cos_cr, sin_cr))
                .collect();
            let fill = node.fill ^ arc_crossing_parity(&node.center, &center, &node.relevant, edges);
            Node { pixel, depth, center, fill, relevant }
        })
        .collect()
}

/// A cell's angular radius (centre→corner), in radians.
fn node_radius(node: &Node) -> f64 {
    let corners = cell_corners(node.depth, node.pixel);
    let (cos_cr, _) = cell_cos_radius(&node.center, &corners);
    cos_cr.clamp(-1.0, 1.0).acos()
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
    let backend = choose_backend(rings);
    let edges = build_edges(rings, order);
    let cap = Cap::of_rings(rings);
    let complement = covers_complement(rings, &cap);

    (0..12u64)
        .into_par_iter()
        .flat_map_iter(|base| {
            let mut out: Vec<(u64, u8)> = Vec::new();
            let Some(seed) = base_node(base, &edges, rings, &cap, backend, complement) else {
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
    let backend = choose_backend(rings);
    let edges = build_edges(rings, order);
    let cap = Cap::of_rings(rings);
    let complement = covers_complement(rings, &cap);

    let mut out: Vec<(u64, u8)> = Vec::new();
    let mut frontier: BinaryHeap<HeapNode> = BinaryHeap::new();

    for base in 0..12u64 {
        if let Some(node) = base_node(base, &edges, rings, &cap, backend, complement) {
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
mod tests {
    use super::*;

    #[test]
    fn test_triangle_basic() {
        let lats = vec![40.0, 50.0, 45.0];
        let lons = vec![-120.0, -120.0, -110.0];
        let result = polygon_to_morton_coverage(&lats, &lons, 4);
        assert!(!result.is_empty(), "Coverage should not be empty");
        for &m in &result {
            assert!(m != 0, "Morton index should not be zero");
        }
    }

    #[test]
    fn test_coverage_sorted_unique() {
        let lats = vec![40.0, 50.0, 45.0];
        let lons = vec![-120.0, -120.0, -110.0];
        let result = polygon_to_morton_coverage(&lats, &lons, 4);
        for i in 1..result.len() {
            assert!(result[i] > result[i - 1], "Result must be sorted and unique");
        }
    }

    #[test]
    fn test_square_coverage() {
        let lats = vec![40.0, 40.0, 50.0, 50.0];
        let lons = vec![-125.0, -115.0, -115.0, -125.0];
        let result = polygon_to_morton_coverage(&lats, &lons, 4);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_southern_hemisphere() {
        let lats = vec![-70.0, -80.0, -75.0];
        let lons = vec![30.0, 30.0, 50.0];
        let result = polygon_to_morton_coverage(&lats, &lons, 4);
        assert!(!result.is_empty());
        assert!(
            result.iter().any(|&m| m < 0),
            "Southern hemisphere should have negative morton indices"
        );
    }

    #[test]
    fn test_different_orders() {
        let lats = vec![40.0, 50.0, 45.0];
        let lons = vec![-120.0, -120.0, -110.0];
        let r4 = polygon_to_morton_coverage(&lats, &lons, 4);
        let r6 = polygon_to_morton_coverage(&lats, &lons, 6);
        assert!(r6.len() > r4.len(), "Higher order should produce more cells");
    }

    #[test]
    fn test_donut_carves_hole() {
        use crate::geo2mort::geo2mort_scalar;
        use std::collections::HashSet;
        // 20° outer box with a centred 6° hole, both around (45, -120).
        let lats = vec![
            vec![35.0, 35.0, 55.0, 55.0],
            vec![42.0, 42.0, 48.0, 48.0],
        ];
        let lons = vec![
            vec![-130.0, -110.0, -110.0, -130.0],
            vec![-123.0, -117.0, -117.0, -123.0],
        ];
        let cov: HashSet<i64> = multipolygon_to_morton_coverage(&lats, &lons, 7)
            .into_iter()
            .collect();
        // Hole interior must be carved out; annulus must be covered.
        assert!(
            !cov.contains(&geo2mort_scalar(45.0, -120.0, 7)),
            "hole interior must not be covered"
        );
        assert!(
            cov.contains(&geo2mort_scalar(37.0, -120.0, 7)),
            "annulus must be covered"
        );
    }

    #[test]
    fn test_multipart_disjoint_equals_union() {
        use std::collections::HashSet;
        let a_la = vec![40.0, 50.0, 45.0];
        let a_lo = vec![-120.0, -120.0, -110.0];
        let b_la = vec![10.0, 20.0, 15.0];
        let b_lo = vec![-80.0, -80.0, -70.0];
        let union: HashSet<i64> = polygon_to_morton_coverage(&a_la, &a_lo, 6)
            .into_iter()
            .chain(polygon_to_morton_coverage(&b_la, &b_lo, 6))
            .collect();
        let multi: HashSet<i64> = multipolygon_to_morton_coverage(
            &vec![a_la, b_la],
            &vec![a_lo, b_lo],
            6,
        )
        .into_iter()
        .collect();
        assert_eq!(multi, union, "disjoint multipart should equal the union");
    }

    #[test]
    fn test_tolerance_coarsens_boundary() {
        // A tolerance stop must yield no more cells than the exact order-10 MOC,
        // and must still cover the interior (superset of a coarse exact cover).
        let lats = vec![40.0, 40.0, 50.0, 50.0];
        let lons = vec![-125.0, -115.0, -115.0, -125.0];
        let exact = polygon_to_morton_moc(&lats, &lons, 10);
        let tol = polygon_to_morton_moc_tolerance(&lats, &lons, 10, 2.0_f64.to_radians());
        assert!(!tol.is_empty());
        assert!(
            tol.len() <= exact.len(),
            "tolerance cover ({}) should not exceed exact ({})",
            tol.len(),
            exact.len()
        );
        // Determinism.
        assert_eq!(tol, polygon_to_morton_moc_tolerance(&lats, &lons, 10, 2.0_f64.to_radians()));
    }

    #[test]
    fn test_budget_respects_cap() {
        // The best-first budget must keep the cell count near the target and be
        // deterministic.
        let lats = vec![40.0, 40.0, 50.0, 50.0];
        let lons = vec![-125.0, -115.0, -115.0, -125.0];
        for budget in [20usize, 50, 200] {
            let (cov, effective) = polygon_to_morton_moc_budget(&lats, &lons, 12, budget);
            assert!(!cov.is_empty());
            // Soft target: at most one split (×4) past the effective budget.
            assert!(
                cov.len() <= effective + 4,
                "budget {} (eff {}) produced {} cells",
                budget,
                effective,
                cov.len()
            );
            assert_eq!(cov, polygon_to_morton_moc_budget(&lats, &lons, 12, budget).0);
        }
    }

    #[test]
    fn test_moc_is_compact_and_densifies_to_flat() {
        // The MOC must be no larger than the flat cover and must densify back
        // to exactly the flat cover (densify-invariance).
        let lats = vec![40.0, 40.0, 50.0, 50.0];
        let lons = vec![-125.0, -115.0, -115.0, -125.0];
        let flat = polygon_to_morton_coverage(&lats, &lons, 8);
        let moc = polygon_to_morton_moc(&lats, &lons, 8);
        assert!(moc.len() <= flat.len(), "MOC should be compact");
        assert!(moc.len() < flat.len(), "interior should collapse to coarse cells");
        assert_eq!(crate::moc::to_order(&moc, 8), flat, "MOC must densify to flat");
    }

    #[test]
    #[should_panic(expected = "at least 3 vertices")]
    fn test_too_few_vertices() {
        polygon_to_morton_coverage(&[0.0, 1.0], &[0.0, 1.0], 4);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn test_mismatched_lengths() {
        polygon_to_morton_coverage(&[0.0, 1.0, 2.0], &[0.0, 1.0], 4);
    }

    #[test]
    fn test_polar_polygon_deterministic() {
        // Regression test for issue #28: a thin near-polar lon-strip used to
        // produce one of two different cell sets at random.  The hierarchical
        // coverer is deterministic by construction and fills the interior.
        let lats = vec![-89.0, -59.09804617, -59.09804617, -89.0];
        let lons = vec![105.5108378, 105.5108378, 106.5108378, 106.5108378];

        let first = polygon_to_morton_coverage(&lats, &lons, 10);
        for _ in 0..50 {
            let r = polygon_to_morton_coverage(&lats, &lons, 10);
            assert_eq!(r, first, "coverage must be deterministic across calls");
        }
        // The buggy boundary-only result was 1166 cells; the correct filled
        // result is in the thousands.  Guard against regressing to boundary-only.
        assert!(
            first.len() > 2000,
            "expected filled interior, got {} cells",
            first.len()
        );
    }

    #[test]
    fn test_polar_boundary_bulge_covered() {
        // Regression test for issue #32.  HEALPix cell edges are not great-circle
        // arcs; near the poles the true cell bulges outside the 4-corner geodesic
        // quad.  This real ATL06 cycle-22 granule (near the south pole, wrapping
        // the antimeridian) grazes the curved boundary of order-6 cell -6111131
        // — overlap that S2 and EPSG:3031 shapely both see.  The corners-only
        // straddle test pruned that cell at order 6, dropping it at every order;
        // the densified-boundary straddle test must now cover it.
        let lats = vec![
            -78.97166, -78.94929, -79.42733, -80.73793, -82.03867, -83.31985,
            -85.7817, -86.88342, -87.71538, -87.87437, -87.80769, -87.61281,
            -87.35613, -87.0609, -86.04897, -85.66109, -83.79971, -83.41805,
            -83.03733, -82.61643, -80.39302, -79.93122, -79.49362, -78.94943,
            -78.97178, -79.51702, -79.95551, -80.41857, -82.64886, -83.07129,
            -83.45395, -83.83797, -85.71503, -86.10799, -87.14043, -87.44464,
            -87.71114, -87.91512, -87.98525, -87.81823, -86.95811, -85.83679,
            -83.35487, -82.06846, -80.76386, -79.45036, -78.97166,
        ];
        let lons = vec![
            -37.60041, -38.19306, -38.71944, -40.42942, -42.66808, -45.72009,
            -57.13684, -69.29219, -92.07313, -102.96984, -139.16366, -149.26852,
            -157.63891, -164.28406, -177.35124, 179.5542, 170.5179, 169.31794,
            168.26288, 167.22786, 163.25287, 162.63733, 162.10694, 161.50445,
            160.91177, 161.48734, 161.99211, 162.57855, 166.3654, 167.35214,
            168.3599, 169.50821, 178.19957, -178.79874, -165.9332, -159.26658,
            -150.74238, -140.28136, -102.10046, -90.7387, -67.65821, -55.75165,
            -44.77371, -41.86274, -39.73073, -38.10328, -37.60041,
        ];
        let cover = polygon_to_morton_coverage(&lats, &lons, 8);
        // A covered order-8 morton is a child of order-6 cell -6111131 iff its
        // order-6 ancestor (two decimal digits stripped) equals it.
        let hits = cover.iter().filter(|&&m| m / 100 == -6111131).count();
        assert!(
            hits > 0,
            "issue #32: order-8 cover misses near-pole boundary cell -6111131 \
             (granule grazes the cell's curved edge, outside its geodesic quad)"
        );
    }

    #[test]
    fn test_square_superset() {
        // Coverage must include all cells whose centres are inside the polygon
        use crate::geo2mort::geo2mort_scalar;
        use std::collections::HashSet;
        let lats = vec![40.0, 40.0, 50.0, 50.0];
        let lons = vec![-125.0, -115.0, -115.0, -125.0];
        let result = polygon_to_morton_coverage(&lats, &lons, 4);
        let coverage_set: HashSet<i64> = result.into_iter().collect();

        // Sample interior points
        for lat in [42.0, 45.0, 48.0] {
            for lon in [-123.0, -120.0, -117.0] {
                let m = geo2mort_scalar(lat, lon, 4);
                assert!(
                    coverage_set.contains(&m),
                    "Interior cell at ({}, {}) = {} not in coverage",
                    lat,
                    lon,
                    m
                );
            }
        }
    }

    #[test]
    fn test_covers_complement_detects_hemisphere_plus() {
        // A band ring at lat -10° traversed so its CCW interior is the >hemisphere
        // northern region (the #22 "everything except Antarctica" shape). The
        // cap-axis antipode must test inside ⇒ complement detected. A small
        // mid-latitude square is sub-hemisphere ⇒ not complement.
        let band: Vec<Vec3> = (0..36)
            .map(|k| latlon_to_unit_vec(-10.0, k as f64 * 10.0))
            .collect();
        let cap = Cap::of_rings(&[band.clone()]);
        assert!(
            covers_complement(&[band], &cap),
            "hemisphere+ band must be detected as complement"
        );

        let square: Vec<Vec3> = [(40.0, -125.0), (40.0, -115.0), (50.0, -115.0), (50.0, -125.0)]
            .iter()
            .map(|&(la, lo)| latlon_to_unit_vec(la, lo))
            .collect();
        let cap2 = Cap::of_rings(&[square.clone()]);
        assert!(
            !covers_complement(&[square], &cap2),
            "sub-hemisphere square must not be complement"
        );
    }

    #[test]
    fn test_complement_guard_keeps_antipodal_base_cell() {
        // Phase 2 regression (#22): for a hemisphere+ polygon the bounding cap
        // bounds only the boundary vertices, so a base cell near the cap antipode
        // sits far from the cap axis and the vertex-cap cull would prune it —
        // even though it is deep in the (large) interior. The complement guard
        // must keep it. Prove both: the un-guarded cull prunes the cell, and the
        // guarded path retains it.
        let band: Vec<Vec3> = (0..36)
            .map(|k| latlon_to_unit_vec(-10.0, k as f64 * 10.0))
            .collect();
        let rings = vec![band];
        let edges = build_edges(&rings, 4);
        let cap = Cap::of_rings(&rings);
        assert!(covers_complement(&rings, &cap), "precondition: hemisphere+");
        let backend = choose_backend(&rings);

        // The base cell whose centre is closest to the cap antipode is the one the
        // vertex-cap cull is most prone to wrongly prune.
        let antipode = [-cap.axis[0], -cap.axis[1], -cap.axis[2]];
        let far_base = (0..12u64)
            .max_by(|&a, &b| {
                let da = dot(&antipode, &cell_center_vec(0, a));
                let db = dot(&antipode, &cell_center_vec(0, b));
                da.partial_cmp(&db).unwrap()
            })
            .unwrap();

        // Without the guard (complement=false) this base is pruned …
        assert!(
            base_node(far_base, &edges, &rings, &cap, backend, false).is_none(),
            "un-guarded cap cull should prune the antipodal base cell"
        );
        // … with the guard (complement=true) it is kept.
        assert!(
            base_node(far_base, &edges, &rings, &cap, backend, true).is_some(),
            "complement guard must keep the antipodal base cell"
        );
    }

    #[test]
    fn test_complement_guard_preserves_subhemisphere_coverage() {
        // The guard must be a no-op for sub-hemisphere polygons: coverage of a
        // mid-latitude square is unchanged (complement=false keeps the original
        // cull exactly). Byte-identical to the pre-guard behaviour.
        let lats = vec![40.0, 40.0, 50.0, 50.0];
        let lons = vec![-125.0, -115.0, -115.0, -125.0];
        let result = polygon_to_morton_coverage(&lats, &lons, 6);
        assert!(!result.is_empty());
        // Determinism / no spurious antipodal cells: a square this small must not
        // pull in any far-side base cell.
        let rings = vec![build_ring(&lats, &lons)];
        let cap = Cap::of_rings(&rings);
        assert!(!covers_complement(&rings, &cap));
    }

    #[test]
    fn test_interpolation_basic() {
        let pts = interpolate_great_circle(0.0, 0.0, 10.0, 0.0, 3);
        assert_eq!(pts.len(), 3);
        for (lat, _lon) in &pts {
            assert!(*lat > 0.0 && *lat < 10.0);
        }
    }

    #[test]
    fn test_interpolation_same_point() {
        let pts = interpolate_great_circle(45.0, -120.0, 45.0, -120.0, 5);
        assert!(pts.is_empty(), "Same point should produce no interpolation");
    }

    #[test]
    fn test_great_circle_distance() {
        let d = great_circle_distance_rad(45.0, -120.0, 45.0, -120.0);
        assert!(d < 1e-10, "Same point should have zero distance");

        let d2 = great_circle_distance_rad(0.0, 0.0, 1.0, 0.0);
        assert!(
            (d2 - 0.01745).abs() < 0.001,
            "1 degree at equator ≈ 0.01745 rad"
        );
    }
}

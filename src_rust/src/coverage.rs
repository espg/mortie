//! Polygon-to-morton coverage via a **top-down hierarchical region coverer**
//! (issue #30).
//!
//! Starting from the 12 HEALPix base cells, each cell is classified against the
//! polygon ring-set (`cell_geom::classify`) as inside / outside / straddling:
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
use crate::geo2mort::ang2pix_scalar;
use crate::morton::nested2mort;
use crate::sphere::{
    arcs_cross, choose_backend, dot, latlon_to_unit_vec, normalize, parity_filled, Vec3,
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

/// Build a base-cell node, or `None` if the base cell is entirely outside the
/// polygon's bounding cap.  Computes the only full O(V) even-odd parity per base.
fn base_node(
    base: u64,
    edges: &[Edge],
    rings: &[Vec<Vec3>],
    cap: &Cap,
    backend: crate::sphere::PipBackend,
) -> Option<Node> {
    let center = cell_center_vec(0, base);
    let corners = cell_corners(0, base);
    let (cos_cr, sin_cr) = cell_cos_radius(&center, &corners);
    let cr = cos_cr.clamp(-1.0, 1.0).acos();
    if dot(&cap.axis, &center).clamp(-1.0, 1.0).acos() > cap.radius + cr {
        return None;
    }
    let relevant: Vec<usize> = (0..edges.len())
        .filter(|&i| edge_relevant(&edges[i], &center, cos_cr, sin_cr))
        .collect();
    let fill = parity_filled(&center, rings, backend);
    Some(Node { pixel: base, depth: 0, center, fill, relevant })
}

/// Does the polygon boundary pass through this cell?  True if a polygon vertex
/// lies in it, a relevant edge crosses a cell edge, or a relevant edge crosses
/// centre→corner (a clipped corner).  Cost is O(relevant edges).
fn node_straddles(node: &Node, edges: &[Edge], order: u8) -> bool {
    let corners = cell_corners(node.depth, node.pixel);
    let shift = 2 * (order - node.depth) as u32;
    node.relevant.iter().any(|&i| edges[i].leaf >> shift == node.pixel)
        || node.relevant.iter().any(|&i| {
            let e = &edges[i];
            (0..4).any(|ci| arcs_cross(&e.a, &e.b, &corners[ci], &corners[(ci + 1) % 4]))
        })
        || corners
            .iter()
            .any(|c| arc_crossing_parity(&node.center, c, &node.relevant, edges))
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

    (0..12u64)
        .into_par_iter()
        .flat_map_iter(|base| {
            let mut out: Vec<(u64, u8)> = Vec::new();
            let Some(seed) = base_node(base, &edges, rings, &cap, backend) else {
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

    let mut out: Vec<(u64, u8)> = Vec::new();
    let mut frontier: BinaryHeap<HeapNode> = BinaryHeap::new();

    for base in 0..12u64 {
        if let Some(node) = base_node(base, &edges, rings, &cap, backend) {
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

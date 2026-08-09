//! Native dissolved-outline emit for a morton cover (issue #71, phase 6).
//!
//! This is the Rust port of `mortie/geometry.py`'s edge-cancellation dissolve
//! (the Python remains the reference/oracle in the tests).  The pipeline:
//!
//! 1. Each cell's boundary is `step` points per edge of unit vectors; every
//!    boundary point is integer-snapped to a vertex id, so a corner/sub-edge
//!    point shared by two neighbours collapses to one id and their shared edge
//!    cancels exactly (no floating tolerance search).
//! 2. The surviving directed edges chain into rings via an upfront rotational
//!    pairing — a pure, order-independent successor function (issue #155); at
//!    a non-manifold corner-touch vertex each arriving edge pairs with the
//!    smallest-turn departing edge, pinching the boundary into simple rings
//!    touching at a point (never a self-intersecting figure-eight).
//! 3. Rings are oriented **covered-region-on-the-left** by one per-call
//!    calibration (the phase-1 contract of issue #147: cell loops are emitted
//!    with a uniform handedness, so at most one global reversal is needed and
//!    the orientation is a *local* fact — valid at any cover scale, hemisphere
//!    and beyond).  Crossing rings are cut at +/-180 and reconnected by the
//!    GeoJSON convention, with pole seams chosen locally from the seam side
//!    (covered-on-left puts the covered seam interval *upward* of a cut end
//!    on +180 and *downward* on -180).
//! 4. Exterior/hole classification happens in the **plane**, where "inside"
//!    is absolute: an emitted planar ring with the covered region on its left
//!    is an exterior iff its shoelace area is positive.  A covered region
//!    enclosing the whole map frame (e.g. world-minus-hole) makes the total
//!    emitted shoelace negative and gets the explicit world-frame shell.
//!    No step keys off a mod-4π winding sign — hemisphere+ covers dissolve
//!    instead of raising (issue #147).
//!
//! The entry point returns classified planar rings (shells and holes) as
//! `(lon, lat)` degree pairs; the Python side builds the backend Polygons and
//! nests holes (both need the shapely codec anyway).

use std::collections::HashMap;

use crate::geo2mort::{boundaries_scalar, boundaries_step_scalar};
use crate::moc;
use crate::morton::mort2nested;
use crate::sphere::{cross, dot, Vec3};

// Snap scale for vertex identity (mirrors `_DISSOLVE_SNAP` in geometry.py): a
// shared HEALPix corner that both adjacent cells compute identically collapses
// to one integer-keyed vertex, so their shared edge cancels exactly.
const SNAP: f64 = 1e10;

/// A closed lon/lat ring (degrees): a list of `(lon, lat)` vertices.
type Ring = Vec<(f64, f64)>;

/// Classified planar (lon, lat) rings: exterior shells and hole rings.
pub struct ClassifiedRings {
    pub shells: Vec<Ring>,
    pub holes: Vec<Ring>,
}

/// Dissolve a morton cover into classified planar (lon, lat) rings.
///
/// Handles covers of any size — hemisphere and beyond (issue #147): the
/// classifier keys off the phase-1 orientation contract (covered region on
/// each ring's left after one per-call calibration) plus planar shoelace
/// signs, never a mod-4π winding sum.  Errs (with a message for a Python
/// `ValueError`, the curated PR #111 convention) only on genuinely ill-posed
/// boundary output: a self-crossing boundary ring, or a stitch state that
/// cannot close (issue #181).
pub fn dissolve(morton: &[u64], step: u32) -> Result<ClassifiedRings, String> {
    if morton.is_empty() {
        return Ok(ClassifiedRings {
            shells: Vec::new(),
            holes: Vec::new(),
        });
    }
    let mut rings = boundary_rings_xyz(morton, step);
    // Orientation calibration (issue #147 phase 1): cell loops are emitted
    // with one uniform handedness (CCW at step == 1, CW at step > 1 — pinned
    // by `cell_boundary_emission_orientation_is_uniform`), each surviving
    // directed edge bounds exactly one covered cell on the emission side, and
    // chaining preserves edge direction — so one reversal iff the emission
    // reads CW puts the covered region on every ring's LEFT, at any cover
    // scale.  One cell's own fan area (~π/3·4^order, far above float noise)
    // is the per-call witness.
    let (nest0, order0) = mort2nested(morton[0]);
    let ccw = spherical_signed_area(&cell_boundary_loop(order0, nest0, step)) > 0.0;
    if !ccw {
        for r in rings.iter_mut() {
            r.reverse();
        }
    }
    #[cfg(debug_assertions)]
    debug_assert_cover_on_left(&rings, morton);
    classify_and_split(rings)
}

/// The retired PR #111 guards (`HEMISPHERE_MARGIN`, the Σ-vs-exact-area wrap
/// cross-check), reduced to what they actually protected: debug-build
/// verification that every oriented ring carries the covered region on its
/// left.  Each ring's first edge midpoint, displaced a quarter edge-length
/// leftward, must land in a covered cell.  (A Σ-fan-area cross-check is
/// *not* usable at hemisphere+ scale: the fan formula's wraps are
/// anchor-dependent and not clean 4π multiples — PR #179 measured 4.0899 sr
/// vs 0.7901 sr fans for the same ring — so no area identity survives; the
/// membership probe is exact instead.)
#[cfg(debug_assertions)]
fn debug_assert_cover_on_left(rings: &[Vec<Vec3>], morton: &[u64]) {
    use crate::sphere::normalize;
    let depths: Vec<u8> = morton.iter().map(|&w| mort2nested(w).1).collect();
    let max_depth = *depths.iter().max().unwrap();
    let flat: Vec<u64> = if depths.iter().any(|&d| d != max_depth) {
        moc::to_order(morton, max_depth)
    } else {
        morton.to_vec()
    };
    let covered: std::collections::HashSet<u64> = flat.into_iter().collect();
    for ring in rings {
        if ring.len() < 2 {
            continue;
        }
        let (a, b) = (ring[0], ring[1]);
        let t = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let elen = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
        let m = normalize(&[a[0] + b[0], a[1] + b[1], a[2] + b[2]]);
        let left = normalize(&cross(&m, &t));
        let p = normalize(&[
            m[0] + 0.25 * elen * left[0],
            m[1] + 0.25 * elen * left[1],
            m[2] + 0.25 * elen * left[2],
        ]);
        let (lon, lat) = xyz_to_lonlat(&p);
        debug_assert!(
            covered.contains(&crate::geo2mort::geo2mort_scalar(lat, lon, max_depth)),
            "dissolve internal inconsistency: a boundary ring does not carry \
             the covered region on its left"
        );
    }
}

// ── edge-cancellation: cover → boundary rings (unit vectors) ───────────────

/// One cell's boundary loop as unit vectors, exactly as emitted (`step`
/// points per edge) — the traversal order every surviving directed edge and
/// the per-call orientation calibration inherit.
fn cell_boundary_loop(order: u8, nest: u64, step: u32) -> Vec<Vec3> {
    if step == 1 {
        let xyz = boundaries_scalar(order, nest); // [[x;4],[y;4],[z;4]]
        (0..4).map(|c| [xyz[0][c], xyz[1][c], xyz[2][c]]).collect()
    } else {
        boundaries_step_scalar(order, nest, step) // Vec<[f64;3]>
    }
}

fn boundary_rings_xyz(morton: &[u64], step: u32) -> Vec<Vec<Vec3>> {
    let (survivors, id_xyz) = survivor_edges(morton, step);
    chain_rings(&survivors, &id_xyz)
}

/// Surviving directed boundary edges of a cover, plus the snapped vertex
/// coordinates they index (split out of [`boundary_rings_xyz`] so tests can
/// drive [`chain_rings`] with permuted edge slices, issue #155).
fn survivor_edges(morton: &[u64], step: u32) -> (Vec<(u32, u32)>, Vec<Vec3>) {
    if morton.is_empty() {
        return (Vec::new(), Vec::new());
    }
    // Decode depth per word; densify a mixed-order MOC to its finest order so
    // every cell carries unit-length edges that cancel against their neighbours.
    let depths: Vec<u8> = morton.iter().map(|&w| mort2nested(w).1).collect();
    let max_depth = *depths.iter().max().unwrap();
    let min_depth = *depths.iter().min().unwrap();
    let flat: Vec<u64> = if min_depth != max_depth {
        moc::to_order(morton, max_depth)
    } else {
        morton.to_vec()
    };
    let order = max_depth;

    // Boundary points per cell, in boundary order, as unit vectors.
    let mut all_pts: Vec<Vec<Vec3>> = Vec::with_capacity(flat.len());
    for &w in &flat {
        all_pts.push(cell_boundary_loop(order, mort2nested(w).0, step));
    }

    // Integer-snap every boundary point to a vertex id.
    let mut id_of: HashMap<[i64; 3], u32> = HashMap::new();
    let mut id_xyz: Vec<Vec3> = Vec::new();
    let mut cell_ids: Vec<Vec<u32>> = Vec::with_capacity(all_pts.len());
    for cell in &all_pts {
        let mut ids = Vec::with_capacity(cell.len());
        for p in cell {
            let key = [
                (p[0] * SNAP).round() as i64,
                (p[1] * SNAP).round() as i64,
                (p[2] * SNAP).round() as i64,
            ];
            let id = *id_of.entry(key).or_insert_with(|| {
                id_xyz.push(*p);
                (id_xyz.len() - 1) as u32
            });
            ids.push(id);
        }
        cell_ids.push(ids);
    }

    // Directed edges around every cell boundary; the net direction per
    // undirected edge survives (an interior edge appears as (a,b) in one cell
    // and (b,a) in its neighbour and cancels).
    let mut counts: HashMap<(u32, u32), i64> = HashMap::new();
    for ids in &cell_ids {
        let n = ids.len();
        for i in 0..n {
            let a = ids[i];
            let b = ids[(i + 1) % n];
            if a != b {
                *counts.entry((a, b)).or_insert(0) += 1;
            }
        }
    }
    let mut survivors: Vec<(u32, u32)> = Vec::new();
    for (&(a, b), &c) in &counts {
        let net = c - counts.get(&(b, a)).copied().unwrap_or(0);
        for _ in 0..net.max(0) {
            survivors.push((a, b));
        }
    }
    (survivors, id_xyz)
}

// ── ring chaining (angular / right-hand rule at non-manifold vertices) ─────

fn tangent_azimuth(p: &Vec3, q: &Vec3) -> f64 {
    let qp = dot(q, p);
    let d = [q[0] - qp * p[0], q[1] - qp * p[1], q[2] - qp * p[2]];
    let nd = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if nd < 1e-15 {
        return 0.0;
    }
    let d = [d[0] / nd, d[1] / nd, d[2] / nd];
    let mut east = cross(&[0.0, 0.0, 1.0], p);
    let ne = (east[0] * east[0] + east[1] * east[1] + east[2] * east[2]).sqrt();
    if ne < 1e-9 {
        east = [1.0, 0.0, 0.0];
    } else {
        east = [east[0] / ne, east[1] / ne, east[2] / ne];
    }
    let north = cross(p, &east);
    dot(&d, &east).atan2(dot(&d, &north))
}

/// Chain surviving directed edges into rings.
///
/// Deterministic and order-independent by construction (issue #155):
/// permuting `survivors` cannot change the output.  The edge stream is
/// canonicalized by sorting, and `successor(edge)` is a **pure function of
/// the edge set** computed up front by a rotational sweep at every vertex:
/// arriving edges (at their reversed-arrival azimuth) and departing edges
/// (at their forward azimuth) are swept in ascending angle, cyclically, and
/// each departing edge closes the most recently opened arrival — **LIFO
/// bracket matching**, which is the contract.  Where arrivals and
/// departures alternate around the vertex (the generic case for a
/// consistently wound boundary) LIFO coincides with the old smallest-turn
/// rule freed of its history-dependent aliveness filter; where snap
/// degeneracy breaks alternation, LIFO governs, because nested wedges then
/// resolve to non-crossing rings.  The pairing is interior-consistent: at a
/// non-manifold corner-touch vertex the boundary pinches into simple rings
/// touching at a point, never a self-intersecting figure-eight.  Each
/// emitted ring starts at its minimal vertex id — rings are seeded from
/// their smallest sorted edge, whose start vertex is that minimum — giving
/// [`classify_and_split`]'s fan a deterministic anchor.  Do not reintroduce
/// history-dependent successor choices (e.g. filtering candidates by which
/// edges an earlier ring consumed) — ring output must stay a pure function
/// of the edge *set*, or dissolve's classification goes nondeterministic
/// again (fan-anchor wrap, issue #155).
fn chain_rings(survivors: &[(u32, u32)], id_xyz: &[Vec3]) -> Vec<Vec<Vec3>> {
    // Canonical edge stream: sorted, duplicate directed edges adjacent.
    let mut edges: Vec<(u32, u32)> = survivors.to_vec();
    edges.sort_unstable();
    let n = edges.len();

    // Rotational sweep entries per vertex: (angle, kind, edge) with kind
    // 0 = arriving edge at its reversed-arrival azimuth, kind 1 = departing
    // edge at its forward azimuth.  At equal angles an arrival sorts first,
    // so a departing edge lying exactly along a reversed arrival pairs with
    // it (zero turn — the old rule's minimum); remaining ties break by
    // canonical edge index.
    let mut at_vertex: HashMap<u32, Vec<(f64, u8, usize)>> = HashMap::new();
    for (i, &(a, b)) in edges.iter().enumerate() {
        let pa = &id_xyz[a as usize];
        let pb = &id_xyz[b as usize];
        at_vertex
            .entry(a)
            .or_default()
            .push((tangent_azimuth(pa, pb), 1, i));
        at_vertex
            .entry(b)
            .or_default()
            .push((tangent_azimuth(pb, pa), 0, i));
    }

    // successor[i] = the edge following edge i in its ring.  Cyclic bracket
    // matching per vertex: sweep entries in ascending angle (twice, for the
    // wrap-around); a departing edge closes the most recently opened arrival
    // (LIFO), so nested wedges resolve to separate simple rings instead of
    // crossing.  Balanced in/out degrees (guaranteed by edge cancellation)
    // leave nothing unpaired; an unbalanced vertex leaves `None`, emitted
    // below as an open chain exactly like the old dead-end break.
    let mut successor: Vec<Option<usize>> = vec![None; n];
    for entries in at_vertex.values_mut() {
        entries.sort_by(|x, y| {
            x.0.partial_cmp(&y.0)
                .unwrap()
                .then(x.1.cmp(&y.1))
                .then(x.2.cmp(&y.2))
        });
        let mut pending: Vec<usize> = Vec::new();
        let mut matched: Vec<bool> = vec![false; entries.len()];
        for pass in 0..2 {
            for (e, &(_, kind, i)) in entries.iter().enumerate() {
                if kind == 0 {
                    if pass == 0 {
                        pending.push(i);
                    }
                } else if !matched[e] {
                    if let Some(arrived) = pending.pop() {
                        successor[arrived] = Some(i);
                        matched[e] = true;
                    }
                }
            }
        }
    }

    // Ring extraction: cycle-following in the functional graph, seeds in
    // canonical edge order.
    let mut rings = Vec::new();
    let mut visited = vec![false; n];
    for seed in 0..n {
        if visited[seed] {
            continue;
        }
        let mut chain: Vec<u32> = Vec::new();
        let mut cur = seed;
        loop {
            visited[cur] = true;
            chain.push(edges[cur].0);
            match successor[cur] {
                Some(nxt) if nxt == seed => break, // ring closed
                Some(nxt) if !visited[nxt] => cur = nxt,
                _ => break, // open chain (unbalanced vertex)
            }
        }
        // The seed is the ring's smallest sorted edge, so `chain` already
        // starts at the ring's minimal vertex id (and, via the shared
        // successor path, in its lexicographically smallest rotation) — no
        // explicit re-rotation needed for a canonical start.
        for cycle in split_at_revisits(chain) {
            rings.push(cycle.iter().map(|&i| id_xyz[i as usize]).collect());
        }
    }
    rings
}

/// Split a closed vertex-id walk into minimal simple cycles.
///
/// The rotational pairing traces boundaries of the *covered* faces; where
/// the covered region is pinched at a vertex (two uncovered cells touching
/// only diagonally, so the covered face's own boundary passes the pinch
/// twice) the traced walk legitimately revisits that vertex — a correct
/// even-odd boundary, but not an OGC-simple ring.  Splitting at each
/// revisit yields the equivalent decomposition into simple rings touching
/// at a point (the dual of the covered-side corner-touch pinch the pairing
/// already resolves).  A split at a *pole* vertex re-pairs that pole's
/// passages onto their adjacent wedges; [`ring_to_planar`]'s empty-bracket
/// span rule renders each re-paired passage over exactly its own wedge, so
/// the planar shoelace budget stays consistent.  Each cycle is rotated to
/// start at its minimal vertex id, keeping the canonical-start determinism
/// of issue #155.
fn split_at_revisits(chain: Vec<u32>) -> Vec<Vec<u32>> {
    let mut out: Vec<Vec<u32>> = Vec::new();
    let mut stack: Vec<u32> = Vec::new();
    let mut pos: HashMap<u32, usize> = HashMap::new();
    for id in chain {
        if let Some(&p) = pos.get(&id) {
            let cycle: Vec<u32> = stack.drain(p..).collect();
            for &c in &cycle {
                pos.remove(&c);
            }
            out.push(rotate_to_min(cycle));
        }
        pos.insert(id, stack.len());
        stack.push(id);
    }
    if !stack.is_empty() {
        out.push(rotate_to_min(stack));
    }
    out
}

/// Split a **closed** planar ring at exactly-repeated vertices into simple
/// closed rings (the planar mirror of [`split_at_revisits`], keyed on exact
/// coordinate bits; drops degenerate two-vertex slivers).
fn split_planar_at_revisits(ring: Ring) -> Vec<Ring> {
    let open = &ring[..ring.len() - 1];
    let key = |p: &(f64, f64)| (p.0.to_bits(), p.1.to_bits());
    let mut out: Vec<Ring> = Vec::new();
    let mut stack: Vec<(f64, f64)> = Vec::new();
    let mut pos: HashMap<(u64, u64), usize> = HashMap::new();
    let mut emit = |cycle: Vec<(f64, f64)>| {
        if cycle.len() >= 3 {
            let mut closed = cycle;
            closed.push(closed[0]);
            out.push(closed);
        }
    };
    for &p in open {
        if let Some(&at) = pos.get(&key(&p)) {
            let cycle: Vec<(f64, f64)> = stack.drain(at..).collect();
            for c in &cycle {
                pos.remove(&key(c));
            }
            emit(cycle);
        }
        pos.insert(key(&p), stack.len());
        stack.push(p);
    }
    emit(stack);
    out
}

/// Rotate a simple cycle to start at its minimal vertex id (unique, since a
/// simple cycle visits each vertex once).
fn rotate_to_min(mut cycle: Vec<u32>) -> Vec<u32> {
    let k = cycle
        .iter()
        .enumerate()
        .min_by_key(|&(_, &v)| v)
        .map(|(i, _)| i)
        .unwrap_or(0);
    cycle.rotate_left(k);
    cycle
}

// ── classification + pole/antimeridian split (GeoJSON convention) ──────────

fn spherical_signed_area(ring: &[Vec3]) -> f64 {
    if ring.len() < 3 {
        return 0.0;
    }
    let a = ring[0];
    let mut total = 0.0;
    for i in 1..ring.len() - 1 {
        let b = ring[i];
        let c = ring[i + 1];
        let num = dot(&a, &cross(&b, &c));
        let den = 1.0 + dot(&b, &a) + dot(&b, &c) + dot(&c, &a);
        total += 2.0 * num.atan2(den);
    }
    total
}

fn xyz_to_lonlat(v: &Vec3) -> (f64, f64) {
    let z = v[2].clamp(-1.0, 1.0);
    let lat = z.asin().to_degrees();
    let lon = v[1].atan2(v[0]).to_degrees();
    (lon, lat)
}

/// Convert an oriented spherical ring to planar lon/lat vertices, expanding
/// any pole vertex into an explicit lat-±90 traverse.
///
/// A vertex at the pole has no longitude (`atan2(0, 0)`); its planar image
/// is a lat-±90 *edge* — a cap over the wedge of longitudes this passage
/// encloses at the pole.  (The old single `atan2(0,0) = 0` planar vertex
/// silently sliced off planar area next to the pole: pointwise-wrong emit
/// for any cover whose boundary passes *through* a pole, e.g. base cells
/// {0, 1} at order 1 — a defect predating issue #147, exposed by its
/// point-sampled acceptance tests.)
///
/// Which wedge a passage encloses is the **empty-bracket rule**: of the two
/// lon-intervals between the arriving and departing meridians, the enclosed
/// one contains no *other* pole-incident boundary meridian (`meridians`,
/// collected over the whole ring set by [`classify_and_split`]) — after
/// [`split_at_revisits`], each passage brackets exactly its own wedge.  At
/// a degree-2 pole both intervals are empty and the covered side wins: the
/// walk pairs the passage across the covered wedge, which lies westward of
/// the arrival at +90 and eastward at -90 (covered-on-left along the
/// meridians).  A cap through the seam inserts an exact ±180 pair that
/// [`cut_at_antimeridian`] splits without interpolation.
fn ring_to_planar(ring: &[Vec3], north_m: &[f64], south_m: &[f64]) -> Vec<(f64, f64)> {
    let n = ring.len();
    let lonlat = normalized_lonlat(ring);
    let mut out: Vec<(f64, f64)> = Vec::with_capacity(n + 4);
    for i in 0..n {
        let v = ring[i];
        if v[2].abs() > 1.0 - 1e-9 {
            let pole = if v[2] > 0.0 { 90.0 } else { -90.0 };
            let (arr, _) = lonlat[(i + n - 1) % n];
            let (dep, _) = lonlat[(i + 1) % n];
            let meridians = if pole > 0.0 { north_m } else { south_m };
            // spans measured from the arrival, mod 360, exclusive of the
            // endpoints (rel 0 is arr itself; rel >= d is dep or beyond).
            let rel_w = |x: f64| (arr - x).rem_euclid(360.0);
            let rel_e = |x: f64| (x - arr).rem_euclid(360.0);
            let dw = if rel_w(dep) == 0.0 { 360.0 } else { rel_w(dep) };
            let de = if rel_e(dep) == 0.0 { 360.0 } else { rel_e(dep) };
            let west_clear = meridians.iter().all(|&m| rel_w(m) == 0.0 || rel_w(m) >= dw);
            let east_clear = meridians.iter().all(|&m| rel_e(m) == 0.0 || rel_e(m) >= de);
            let go_west = if west_clear && east_clear {
                pole > 0.0 // covered side: westward at +90, eastward at -90
            } else {
                west_clear
            };
            out.push((arr, pole));
            if go_west {
                if dep < arr {
                    pole_run(&mut out, arr, dep, pole);
                } else {
                    pole_run(&mut out, arr, -180.0, pole);
                    out.push((180.0, pole));
                    pole_run(&mut out, 180.0, dep, pole);
                }
            } else if dep > arr {
                pole_run(&mut out, arr, dep, pole);
            } else {
                pole_run(&mut out, arr, 180.0, pole);
                out.push((-180.0, pole));
                pole_run(&mut out, -180.0, dep, pole);
            }
        } else {
            out.push(lonlat[i]);
        }
    }
    out
}

/// Append a monotone lat-±90 run from `from` to `to` (exclusive of `from`),
/// subdivided so no planar step reaches 180° — the pole edge is spherically
/// degenerate, so extra vertices are free, and a covered polar wedge can
/// legitimately span more than 180° of longitude (e.g. three of the four
/// polar base cells), which `cut_at_antimeridian` would misread as a seam
/// wrap if emitted as one step.
fn pole_run(out: &mut Vec<(f64, f64)>, from: f64, to: f64, pole: f64) {
    let dir = (to - from).signum();
    let mut cur = from;
    while (to - cur).abs() > 150.0 {
        cur += dir * 120.0;
        out.push((cur, pole));
    }
    out.push((to, pole));
}

/// Per-vertex `(lon, lat)` with seam-lying vertices' longitude **signs
/// normalized by traversal direction**.  A vertex exactly on the ±180
/// meridian gets an arbitrary sign from `atan2` (its `y` is a rounding
/// residual), and a whole boundary *edge* can lie on the seam (base-cell
/// meridians at lon 180, e.g. base cells {9, 10}'s shared edge) — mixed
/// signs there fabricate spurious ±360 cuts mid-edge.  With the covered
/// region on the ring's left, a seam-lying edge walked *northward* has the
/// covered side west, so it belongs on the map's east edge (+180);
/// southward, on −180.  A vertex with a seam neighbour takes that edge's
/// direction; an isolated seam touch takes its off-seam neighbours' side.
fn normalized_lonlat(ring: &[Vec3]) -> Vec<(f64, f64)> {
    let n = ring.len();
    let mut lonlat: Vec<(f64, f64)> = ring.iter().map(xyz_to_lonlat).collect();
    let is_pole = |j: usize| ring[j][2].abs() > 1.0 - 1e-9;
    let on_seam = |ll: &[(f64, f64)], j: usize| !is_pole(j) && (ll[j].0.abs() - 180.0).abs() < 1e-9;
    // Effective seam latitude of a neighbour when the edge to it runs along
    // the seam: a seam vertex's own latitude, or ±90 for a pole vertex (any
    // great-circle edge into a pole from a seam vertex is the seam meridian).
    let seam_lat = |ll: &[(f64, f64)], j: usize| -> Option<f64> {
        if is_pole(j) {
            Some(if ring[j][2] > 0.0 { 90.0 } else { -90.0 })
        } else if on_seam(ll, j) {
            Some(ll[j].1)
        } else {
            None
        }
    };
    let orig = lonlat.clone();
    for i in 0..n {
        if !on_seam(&orig, i) {
            continue;
        }
        let (prev, next) = ((i + n - 1) % n, (i + 1) % n);
        let side = if let Some(la) = seam_lat(&orig, next) {
            // seam edge onward: northward ⟹ covered west ⟹ map east (+180).
            if la > orig[i].1 {
                180.0
            } else {
                -180.0
            }
        } else if let Some(la) = seam_lat(&orig, prev) {
            if orig[i].1 > la {
                180.0
            } else {
                -180.0
            }
        } else {
            // isolated touch: stay on the off-seam neighbours' side.
            if orig[prev].0 >= 0.0 {
                180.0
            } else {
                -180.0
            }
        };
        lonlat[i].0 = side;
    }
    lonlat
}

/// Shoelace signed area (deg²) of a closed planar lon/lat ring (first vertex
/// repeated) — positive iff the traversal is CCW in the plane.  With every
/// ring carrying the covered region on its left (the phase-1 orientation
/// contract survives the cylindrical projection and the seam stitch),
/// positive ⟺ exterior shell, negative ⟺ hole; the sign is decisive because
/// an emitted ring bounds at least one cell's planar image.
fn planar_shoelace(ring: &[(f64, f64)]) -> f64 {
    let open = &ring[..ring.len() - 1];
    let n = open.len();
    let mut acc = 0.0;
    for i in 0..n {
        let (x0, y0) = open[i];
        let (x1, y1) = open[(i + 1) % n];
        acc += x0 * y1 - x1 * y0;
    }
    0.5 * acc
}

/// Planar area (deg²) of the whole lon/lat map — the frame ring's shoelace.
const MAP_AREA: f64 = 360.0 * 180.0;

/// The whole-map frame shell `[-180, 180] × [-90, 90]`, CCW (positive
/// shoelace): the planar boundary of a covered region that encloses the
/// entire antimeridian seam and both poles (e.g. world-minus-hole), which no
/// spherical boundary ring supplies.
fn frame_ring() -> Ring {
    vec![
        (-180.0, -90.0),
        (180.0, -90.0),
        (180.0, 90.0),
        (-180.0, 90.0),
        (-180.0, -90.0),
    ]
}

/// Cut an open lon/lat ring at +/-180.  `Ok(whole)` when the ring never
/// crosses; `Err(segments)` with each segment an open polyline whose free ends
/// sit on +/-180.
fn cut_at_antimeridian(coords: &[(f64, f64)]) -> Result<Ring, Vec<Ring>> {
    let n = coords.len();
    let mut segments: Vec<Vec<(f64, f64)>> = Vec::new();
    let mut cur: Vec<(f64, f64)> = Vec::new();
    for i in 0..n {
        let (lo0, la0) = coords[i];
        let (lo1, la1) = coords[(i + 1) % n];
        cur.push((lo0, la0));
        if (lo1 - lo0).abs() > 180.0 {
            let lo1u = if lo1 > lo0 { lo1 - 360.0 } else { lo1 + 360.0 };
            // An exact ±180 → ∓180 pair (a pole traverse's explicit seam
            // crossing, `ring_to_planar`) unwraps to a zero-length step:
            // nothing to interpolate, and the cut side is the pair's own
            // first side (the generic `lo1u > lo0` test cannot tell).
            let boundary = if lo1u == lo0 {
                lo0
            } else if lo1u > lo0 {
                180.0
            } else {
                -180.0
            };
            let frac = if lo1u == lo0 {
                0.0
            } else {
                (boundary - lo0) / (lo1u - lo0)
            };
            let la_x = la0 + frac * (la1 - la0);
            cur.push((boundary, la_x));
            segments.push(std::mem::take(&mut cur));
            cur = vec![(-boundary, la_x)];
        }
    }
    if segments.is_empty() {
        let mut whole = coords.to_vec();
        whole.push(coords[0]);
        return Ok(whole);
    }
    // the wrap-around segment closes the first.
    let mut first = std::mem::take(&mut cur);
    first.append(&mut segments[0]);
    segments[0] = first;
    Err(segments)
}

/// Reconnect antimeridian-cut `segments` into closed lon/lat rings.
///
/// Pole seams are chosen **locally** (issue #147): the phase-1 orientation
/// contract (covered region on every ring's left) means the covered seam
/// interval always runs *upward* from a cut end on +180 and *downward* on
/// -180 — so an end with no same-side start in that direction wraps the pole
/// on that side, unconditionally.  No global winding parameter exists to be
/// wrong, and one ring may wrap both poles (a covered region enclosing the
/// whole seam).  This replaces the net-longitude-winding `pole` argument,
/// whose single global value mis-stitched covers mixing a pole region with
/// other seam-crossing parts.
///
/// Errs (with a message for a Python `ValueError`, the curated convention
/// from PR #111 — issue #181) instead of panicking on stitch states that
/// cannot be closed: a missing partner segment after a pole seam, or a
/// non-converging walk.
fn stitch_segments(segments: Vec<Vec<(f64, f64)>>) -> Result<Vec<Vec<(f64, f64)>>, String> {
    let segs = segments;
    let n = segs.len();
    let mut used = vec![false; n];
    let mut rings: Vec<Vec<(f64, f64)>> = Vec::new();
    for seed in 0..n {
        if used[seed] {
            continue;
        }
        let mut ring: Vec<(f64, f64)> = Vec::new();
        let mut idx = Some(seed);
        let mut guard = 0usize;
        while let Some(i) = idx {
            if used[i] {
                break;
            }
            guard += 1;
            if guard > 8 * n + 16 {
                return Err(
                    "dissolved cover's antimeridian stitch did not converge (an \
                     internal dissolve inconsistency); pass dissolve=False for \
                     per-cell polygons"
                        .to_string(),
                );
            }
            used[i] = true;
            ring.extend_from_slice(&segs[i]);
            idx = next_segment(&segs, &used, &mut ring, seed)?;
        }
        if let Some(&first) = ring.first() {
            ring.push(first);
        }
        rings.push(ring);
    }
    Ok(rings)
}

fn next_segment(
    segs: &[Vec<(f64, f64)>],
    used: &[bool],
    ring: &mut Vec<(f64, f64)>,
    seed: usize,
) -> Result<Option<usize>, String> {
    let &(side, end_lat) = ring.last().unwrap();
    // candidate starts on the same +/-180 side (the seed is allowed, to close).
    // The Python oracle keys min/max on the full (lat, index) tuple; here we
    // compare on lat only, but iterate `0..segs.len()` in index order, so
    // `min_by`/`max_by` (first-/last-of-equal) reproduce the same tie-break.
    // Distinct crossing points never share a latitude within 1e-9°, so the tie
    // is unreachable in practice anyway.
    let same_side = |i: usize| (segs[i][0].0 - side).abs() < 1e-9 && (!used[i] || i == seed);
    let pick: Option<(f64, usize)> = if side > 0.0 {
        (0..segs.len())
            .filter(|&i| same_side(i) && segs[i][0].1 >= end_lat - 1e-9)
            .map(|i| (segs[i][0].1, i))
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
    } else {
        (0..segs.len())
            .filter(|&i| same_side(i) && segs[i][0].1 <= end_lat + 1e-9)
            .map(|i| (segs[i][0].1, i))
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
    };
    if let Some((la, i)) = pick {
        ring.push((side, la));
        return Ok(if i == seed && used[seed] {
            None
        } else {
            Some(i)
        });
    }

    // No same-side start in that direction: the covered seam interval runs to
    // the pole on this side (upward on +180, downward on -180 — the phase-1
    // orientation contract), so the region wraps that pole.  Cross it and
    // resume on the other side at the start nearest the pole: the highest
    // start after a north wrap (walking down -180), the lowest after a south
    // wrap (walking up +180).
    let pole = if side > 0.0 { 90.0 } else { -90.0 };
    let other = -side;
    ring.push((side, pole));
    ring.push((other, pole));
    let ocands: Vec<(f64, usize)> = (0..segs.len())
        .filter(|&i| (segs[i][0].0 - other).abs() < 1e-9 && (!used[i] || i == seed))
        .map(|i| (segs[i][0].1, i))
        .collect();
    if ocands.is_empty() {
        return Err(
            "dissolved cover's antimeridian stitch found no partner segment \
             after a pole seam (an internal dissolve inconsistency, issue \
             #181); pass dissolve=False for per-cell polygons"
                .to_string(),
        );
    }
    let (la, i) = if pole > 0.0 {
        *ocands
            .iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .unwrap()
    } else {
        *ocands
            .iter()
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .unwrap()
    };
    ring.push((other, la));
    Ok(if i == seed && used[seed] {
        None
    } else {
        Some(i)
    })
}

/// Classify covered-on-left boundary rings into planar shells and holes —
/// the winding-free classifier (issue #147).
///
/// `rings_xyz` must already carry the covered region on every ring's left
/// (the phase-1 orientation contract, established per-call in [`dissolve`]).
/// Classification is then planar and local: cut/stitch at the antimeridian
/// (pole seams locally decided), and read each emitted planar ring's
/// shoelace sign — positive (CCW, covered inside) ⟹ shell, negative ⟹ hole.
/// A negative *total* shoelace means the covered region encloses the whole
/// map frame (world-minus-hole covers), which gets the explicit
/// [`frame_ring`] shell.  Nothing consults a mod-4π winding sum: the PR #111
/// guards this replaces (`HEMISPHERE_MARGIN`, the Σ-vs-exact-area wrap
/// cross-check) survive only as [`dissolve`]'s debug assertion on the
/// orientation contract ([`debug_assert_cover_on_left`]).
///
/// Errs (curated, for a Python `ValueError`) only on genuinely ill-posed
/// boundary output: a self-crossing boundary ring
/// ([`crate::sphere::ring_set_validity`]), or a stitch state that cannot
/// close (issue #181).  Identity conflicts — one snapped coordinate at two
/// non-adjacent ring positions — are *accepted*: they are the working
/// representation of a corner-touch pinch resolved into touching simple
/// rings (the issue #155 ruling), pinned by the corner-touch dissolve tests.
fn classify_and_split(rings_xyz: Vec<Vec<Vec3>>) -> Result<ClassifiedRings, String> {
    let mut out = ClassifiedRings {
        shells: Vec::new(),
        holes: Vec::new(),
    };
    if rings_xyz.is_empty() {
        // A nonempty cover with no surviving boundary edge is the whole
        // sphere: every edge cancelled, and the planar image is the frame.
        out.shells.push(frame_ring());
        return Ok(out);
    }

    let validity = crate::sphere::ring_set_validity(&rings_xyz);
    if let Some((r, i, j)) = validity.crossing {
        return Err(format!(
            "dissolved cover produced a self-crossing boundary ring (ring \
             {r}, edges at vertices {i} and {j}), so its outline is not \
             classifiable; pass dissolve=False for per-cell polygons"
        ));
    }

    let classify = |ring: Ring, out: &mut ClassifiedRings| {
        // A planar ring can revisit a vertex exactly — e.g. a ring that both
        // expands a pole traverse and wraps the same pole's seam passes the
        // ±180/±90 corner twice — which even-odd fill reads correctly but
        // OGC does not allow in one ring; split into simple rings touching
        // at the point.
        for piece in split_planar_at_revisits(ring) {
            if planar_shoelace(&piece) >= 0.0 {
                out.shells.push(piece);
            } else {
                out.holes.push(piece);
            }
        }
    };
    // Pole-incident boundary meridians over the whole ring set, for
    // `ring_to_planar`'s empty-bracket cap rule.
    let mut north_m: Vec<f64> = Vec::new();
    let mut south_m: Vec<f64> = Vec::new();
    for ring in rings_xyz.iter() {
        let n = ring.len();
        let lonlat = normalized_lonlat(ring);
        for i in 0..n {
            if ring[i][2].abs() > 1.0 - 1e-9 {
                let m = if ring[i][2] > 0.0 {
                    &mut north_m
                } else {
                    &mut south_m
                };
                m.push(lonlat[(i + n - 1) % n].0);
                m.push(lonlat[(i + 1) % n].0);
            }
        }
    }

    let mut segments: Vec<Vec<(f64, f64)>> = Vec::new();
    for ring in rings_xyz.iter() {
        let ll = ring_to_planar(ring, &north_m, &south_m);
        match cut_at_antimeridian(&ll) {
            Ok(whole) => classify(whole, &mut out),
            Err(segs) => segments.extend(segs),
        }
    }
    if !segments.is_empty() {
        for piece in stitch_segments(segments)? {
            classify(piece, &mut out);
        }
    }

    // Frame rule: Σ shoelace over the emitted rings is the covered region's
    // planar area when its boundary is complete, and that minus the full map
    // when the region encloses the frame (both poles and the whole seam
    // covered, with no ring crossing the seam) — the two cases are separated
    // by sign, since a nonempty cover has positive planar area.
    let total: f64 = out
        .shells
        .iter()
        .chain(out.holes.iter())
        .map(|r| planar_shoelace(r))
        .sum();
    let framed = total < 0.0;
    if framed {
        out.shells.insert(0, frame_ring());
    }
    let planar_area = total + if framed { MAP_AREA } else { 0.0 };
    debug_assert!(
        planar_area > 0.0 && planar_area < MAP_AREA + 1.0,
        "dissolve internal inconsistency: emitted planar area {planar_area} \
         deg^2 outside (0, whole map]"
    );
    Ok(out)
}

// ── tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;

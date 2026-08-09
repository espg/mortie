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
//! 3. Rings are classified exterior/hole by spherical signed area (global
//!    winding normalised so the covered area is positive), then crossing rings
//!    are cut at +/-180 and reconnected by the GeoJSON convention — explicit
//!    +/-90 pole vertices stitched down the antimeridian for a pole-enclosing
//!    region.
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

// Hemisphere guard (issue #108): exterior/hole classification keys off the
// sign of Σ ring signed areas, which is defined mod 4π — at 2π a cover reads
// the same as its complement wound the other way, and past 2π the sign
// silently inverts (the fan formula can also wrap per-ring at that scale).
// The covered area itself is exact (equal-area cells), so gate on it: 2% of 2π
// keeps a comfortable distance from the breakdown point while excluding only
// covers within 2% of half the sphere.
const HEMISPHERE_MARGIN: f64 = std::f64::consts::TAU * 0.02;

/// Exact covered area (steradians) of a morton cover: Σ π/(3·4^depth).
/// Assumes disjoint, non-duplicated cells (the dissolve precondition anyway —
/// duplicate words would break edge cancellation); duplicates double-count.
fn cover_area(morton: &[u64]) -> f64 {
    morton
        .iter()
        .map(|&w| std::f64::consts::PI / (3.0 * 4f64.powi(mort2nested(w).1 as i32)))
        .sum()
}

/// Dissolve a morton cover into classified planar (lon, lat) rings.
///
/// Errs (with a message for a Python `ValueError`) when the cover spans
/// near or over a hemisphere, or when a boundary ring encloses more than a
/// hemisphere (the mod-4π fan sum wraps) — both make the winding-sign
/// normalisation of [`classify_and_split`] untrustworthy.
pub fn dissolve(morton: &[u64], step: u32) -> Result<ClassifiedRings, String> {
    let area = cover_area(morton);
    if area > std::f64::consts::TAU - HEMISPHERE_MARGIN {
        return Err(format!(
            "dissolved cover spans {area:.6} sr — within 2% of a hemisphere \
             (2π sr) or beyond — so its exterior/hole winding is ambiguous; \
             split the cover into sub-hemisphere parts or pass dissolve=False \
             for per-cell polygons"
        ));
    }
    let rings = boundary_rings_xyz(morton, step);
    classify_and_split(rings, area)
}

// ── edge-cancellation: cover → boundary rings (unit vectors) ───────────────

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
        let nest = mort2nested(w).0;
        if step == 1 {
            let xyz = boundaries_scalar(order, nest); // [[x;4],[y;4],[z;4]]
            let cell: Vec<Vec3> = (0..4).map(|c| [xyz[0][c], xyz[1][c], xyz[2][c]]).collect();
            all_pts.push(cell);
        } else {
            all_pts.push(boundaries_step_scalar(order, nest, step)); // Vec<[f64;3]>
        }
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
        rings.push(chain.iter().map(|&i| id_xyz[i as usize]).collect());
    }
    rings
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

fn net_winding(coords: &[(f64, f64)]) -> f64 {
    let n = coords.len();
    let mut net = 0.0;
    for i in 0..n {
        let lo0 = coords[i].0;
        let lo1 = coords[(i + 1) % n].0;
        let d = lo1 - lo0;
        net += (d + 180.0).rem_euclid(360.0) - 180.0;
    }
    net
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
            let boundary = if lo1u > lo0 { 180.0 } else { -180.0 };
            let frac = (boundary - lo0) / (lo1u - lo0);
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

/// Reconnect antimeridian-cut `segments` into closed lon/lat rings.  `pole` is
/// the pole (+/-90) the filled region encloses (0.0 = none).
fn stitch_segments(segments: Vec<Vec<(f64, f64)>>, pole: f64) -> Vec<Vec<(f64, f64)>> {
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
            assert!(guard <= 8 * n + 16, "antimeridian stitch did not converge");
            used[i] = true;
            ring.extend_from_slice(&segs[i]);
            idx = next_segment(&segs, &used, &mut ring, pole, seed);
        }
        if let Some(&first) = ring.first() {
            ring.push(first);
        }
        rings.push(ring);
    }
    rings
}

fn next_segment(
    segs: &[Vec<(f64, f64)>],
    used: &[bool],
    ring: &mut Vec<(f64, f64)>,
    pole: f64,
    seed: usize,
) -> Option<usize> {
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
        return if i == seed && used[seed] {
            None
        } else {
            Some(i)
        };
    }

    // No same-side start in that direction: the region wraps `pole`.
    assert!(
        pole != 0.0,
        "unbalanced antimeridian segments but no pole enclosed"
    );
    let other = -side;
    ring.push((side, pole));
    ring.push((other, pole));
    let ocands: Vec<(f64, usize)> = (0..segs.len())
        .filter(|&i| (segs[i][0].0 - other).abs() < 1e-9 && (!used[i] || i == seed))
        .map(|i| (segs[i][0].1, i))
        .collect();
    if ocands.is_empty() {
        return None;
    }
    let want_min = (other > 0.0) == (pole < 0.0);
    let (la, i) = if want_min {
        *ocands
            .iter()
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .unwrap()
    } else {
        *ocands
            .iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .unwrap()
    };
    ring.push((other, la));
    if i == seed && used[seed] {
        None
    } else {
        Some(i)
    }
}

fn ring_signed_area_lonlat(ring: &[(f64, f64)]) -> f64 {
    // ring is closed (first == last); drop the repeat for the area fan.
    let open = &ring[..ring.len() - 1];
    let v: Vec<Vec3> = open
        .iter()
        .map(|&(lon, lat)| {
            let rlat = lat.to_radians();
            let rlon = lon.to_radians();
            [rlat.cos() * rlon.cos(), rlat.cos() * rlon.sin(), rlat.sin()]
        })
        .collect();
    spherical_signed_area(&v)
}

fn classify_and_split(
    mut rings_xyz: Vec<Vec<Vec3>>,
    cover_area: f64,
) -> Result<ClassifiedRings, String> {
    let mut out = ClassifiedRings {
        shells: Vec::new(),
        holes: Vec::new(),
    };
    if rings_xyz.is_empty() {
        return Ok(out);
    }
    // Normalise global winding so the covered area (exteriors minus holes) is
    // positive — the boundary point order differs between step==1 and step>1.
    // The fan formula wraps mod 4π when a single ring encloses more than a
    // hemisphere (possible even for a small cover, e.g. an equatorial band),
    // which would flip the sign here; an honest |Σ| matches the exact covered
    // area to within chord discretization (≲0.1 sr at step==1) while any wrap
    // is off by ~4π, so a π tolerance separates them cleanly.
    let mut areas: Vec<f64> = rings_xyz.iter().map(|r| spherical_signed_area(r)).collect();
    let total: f64 = areas.iter().sum();
    if (total.abs() - cover_area).abs() > std::f64::consts::PI {
        return Err(format!(
            "dissolved cover has a boundary ring enclosing more than a \
             hemisphere (|Σ ring areas| = {:.6} sr vs covered area {cover_area:.6} \
             sr), so its exterior/hole winding cannot be classified; split the \
             cover into sub-hemisphere parts or pass dissolve=False for \
             per-cell polygons",
            total.abs()
        ));
    }
    if total < 0.0 {
        for r in rings_xyz.iter_mut() {
            r.reverse();
        }
        for a in areas.iter_mut() {
            *a = -*a;
        }
    }

    let mut segments: Vec<Vec<(f64, f64)>> = Vec::new();
    let mut total_net = 0.0;
    for (ring, &area) in rings_xyz.iter().zip(areas.iter()) {
        let ll: Vec<(f64, f64)> = ring.iter().map(xyz_to_lonlat).collect();
        total_net += net_winding(&ll);
        match cut_at_antimeridian(&ll) {
            Ok(whole) => {
                if area < 0.0 {
                    out.holes.push(whole);
                } else {
                    out.shells.push(whole);
                }
            }
            Err(segs) => segments.extend(segs),
        }
    }

    if !segments.is_empty() {
        let pole = if total_net.abs() > 180.0 {
            if total_net > 0.0 {
                90.0
            } else {
                -90.0
            }
        } else {
            0.0
        };
        for piece in stitch_segments(segments, pole) {
            if ring_signed_area_lonlat(&piece) >= 0.0 {
                out.shells.push(piece);
            } else {
                out.holes.push(piece);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cut_no_crossing_returns_whole() {
        let ring = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let got = cut_at_antimeridian(&ring).unwrap();
        assert_eq!(got.len(), ring.len() + 1); // closed
        assert_eq!(got[0], got[got.len() - 1]);
    }

    #[test]
    fn cut_two_crossings_splits_each_side() {
        // box straddling +/-180: two segments, one per hemisphere side.
        let ring = [(170.0, 0.0), (-170.0, 0.0), (-170.0, 10.0), (170.0, 10.0)];
        let segs = cut_at_antimeridian(&ring).unwrap_err();
        assert_eq!(segs.len(), 2);
        for s in &segs {
            assert!((s[0].0.abs() - 180.0).abs() < 1e-9);
            assert!((s[s.len() - 1].0.abs() - 180.0).abs() < 1e-9);
        }
        // no pole enclosed -> each segment closes on its own side.
        let rings = stitch_segments(segs, 0.0);
        assert_eq!(rings.len(), 2);
        for r in &rings {
            let span = r.iter().map(|p| p.0).fold(f64::MIN, f64::max)
                - r.iter().map(|p| p.0).fold(f64::MAX, f64::min);
            assert!(span <= 180.0 + 1e-9);
        }
    }

    #[test]
    fn net_winding_detects_pole_wrap() {
        // a ring marching once around the globe wraps a pole (net ~ +/-360).
        let ring: Vec<(f64, f64)> = (-180..180)
            .step_by(30)
            .map(|lo| (lo as f64, -80.0))
            .collect();
        assert!(net_winding(&ring).abs() > 180.0);
    }

    #[test]
    fn hemisphere_cover_fails_loud() {
        // 24 order-1 cells (base cells 0-5) tile exactly half the sphere
        // (area = 2π), where exterior/hole winding is ambiguous (issue #108).
        let cover: Vec<u64> = (0..24u64)
            .map(|nest| crate::morton::nested2mort(nest, 1))
            .collect();
        assert!((cover_area(&cover) - std::f64::consts::TAU).abs() < 1e-12);
        let err = dissolve(&cover, 1)
            .err()
            .expect("hemisphere must be rejected");
        assert!(err.contains("hemisphere"), "{err}");
    }

    #[test]
    fn equatorial_band_fails_loud_not_cryptic() {
        // A thin equatorial band (~1.3 sr, far under the area guard) has two
        // boundary rings that each enclose more than a hemisphere, so the fan
        // sum wraps mod 4π: the Σ-vs-exact-area cross-check must reject it
        // with the curated message, not die in the antimeridian stitcher.
        let mut band: Vec<u64> = Vec::new();
        for ilon in 0..720 {
            for ilat in -7..8 {
                let lat = ilat as f64 * 0.5;
                let lon = -180.0 + ilon as f64 * 0.5;
                band.push(crate::geo2mort::geo2mort_scalar(lat, lon, 4));
            }
        }
        band.sort_unstable();
        band.dedup();
        let err = dissolve(&band, 1).err().expect("band must be rejected");
        assert!(err.contains("hemisphere"), "{err}");
    }

    #[test]
    fn sub_hemisphere_cover_still_dissolves() {
        // Base cells 0-3 (2/3 of a hemisphere) stay outside the guard.
        let cover: Vec<u64> = (0..16u64)
            .map(|nest| crate::morton::nested2mort(nest, 1))
            .collect();
        let got = dissolve(&cover, 1).unwrap();
        assert!(!got.shells.is_empty());
    }

    #[test]
    fn sub_hemisphere_cover_dissolves_deterministically() {
        // issue #155: dissolve on this fixed cover (base cells 0-3 at order
        // 1) flaked ~10-15% per call.  Phase-1 instrumentation finding: the
        // cover's boundary graph is a single simple 16-cycle -- there is no
        // branching vertex at all (neither a true corner-touch nor a
        // snap-merge artifact; no near-coincident vertex pairs, no duplicate
        // directed edges), so the ring *decomposition* is unique.  The
        // nondeterminism was purely the HashMap-seeded starting rotation of
        // that one ring: classify_and_split fans spherical_signed_area from
        // ring[0], and 2 of the 16 possible anchors (the equatorial spike
        // tips at lon +/-45, lat 0) wrap the fan to 0.790 sr vs the true
        // 4.090 sr, tripping the hemisphere-ring guard -- predicting 2/16 =
        // 12.5% (measured 48/400 pre-fix).  Each dissolve call builds fresh
        // HashMaps, so the flake reproduces in-process: pre-fix, 50
        // iterations fail with p ~= 1 - (7/8)^50 ~= 0.999.
        let cover: Vec<u64> = (0..16u64)
            .map(|nest| crate::morton::nested2mort(nest, 1))
            .collect();
        for i in 0..50 {
            let got = dissolve(&cover, 1).unwrap_or_else(|e| panic!("iteration {i} failed: {e}"));
            assert!(!got.shells.is_empty(), "iteration {i}: no shells");
        }
    }

    #[test]
    fn dissolve_output_is_byte_identical_across_calls() {
        // issue #155 phase 3: dissolve is a pure function of its input --
        // two calls on identical input return byte-identical ClassifiedRings,
        // on each fixture shape (plain cover, step > 1 densified edges, the
        // pole/antimeridian stitch path, an equatorial cell).
        let north: Vec<u64> = (0..16u64)
            .map(|n| crate::morton::nested2mort(n, 1))
            .collect();
        let south: Vec<u64> = (32..48u64)
            .map(|n| crate::morton::nested2mort(n, 1))
            .collect();
        let eq: Vec<u64> = (16..20u64)
            .map(|n| crate::morton::nested2mort(n, 1))
            .collect();
        for (cover, step) in [(&north, 1u32), (&north, 4), (&south, 1), (&eq, 1), (&eq, 3)] {
            let a = dissolve(cover, step).unwrap();
            let b = dissolve(cover, step).unwrap();
            assert_eq!(a.shells, b.shells);
            assert_eq!(a.holes, b.holes);
            assert!(!a.shells.is_empty());
        }
    }

    fn lonlat_xyz(lon: f64, lat: f64) -> Vec3 {
        let (rlon, rlat) = (lon.to_radians(), lat.to_radians());
        [rlat.cos() * rlon.cos(), rlat.cos() * rlon.sin(), rlat.sin()]
    }

    // Two diamonds east and west of a shared corner-touch vertex T (id 0),
    // both wound anticlockwise (interior on the same side).
    fn corner_touch_fixture() -> (Vec<Vec3>, Vec<(u32, u32)>) {
        let id_xyz = vec![
            lonlat_xyz(0.0, 0.0),   // 0: T, the corner-touch vertex
            lonlat_xyz(5.0, 5.0),   // 1: east diamond, north corner
            lonlat_xyz(10.0, 0.0),  // 2: east diamond, east corner
            lonlat_xyz(5.0, -5.0),  // 3: east diamond, south corner
            lonlat_xyz(-5.0, 5.0),  // 4: west diamond, north corner
            lonlat_xyz(-10.0, 0.0), // 5: west diamond, west corner
            lonlat_xyz(-5.0, -5.0), // 6: west diamond, south corner
        ];
        let edges = vec![
            (0, 3),
            (3, 2),
            (2, 1),
            (1, 0), // east: T -> S -> E -> N -> T
            (0, 4),
            (4, 5),
            (5, 6),
            (6, 0), // west: T -> N -> W -> S -> T
        ];
        (id_xyz, edges)
    }

    // chain_rings on every LCG-shuffled permutation of `edges` (plus the
    // reverse-sorted order) must equal its output on the canonical order.
    fn assert_permutation_invariant(edges: &[(u32, u32)], id_xyz: &[Vec3]) {
        let baseline = chain_rings(edges, id_xyz);
        assert!(!baseline.is_empty());
        // deterministic Fisher-Yates via an LCG (no rand dependency).
        let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        let mut shuffled = edges.to_vec();
        for _ in 0..200 {
            for i in (1..shuffled.len()).rev() {
                let j = rng() % (i + 1);
                shuffled.swap(i, j);
            }
            assert_eq!(chain_rings(&shuffled, id_xyz), baseline);
        }
        shuffled.sort_unstable();
        shuffled.reverse();
        assert_eq!(chain_rings(&shuffled, id_xyz), baseline);
    }

    #[test]
    fn chain_rings_is_order_independent() {
        // issue #155 phase 3: chain_rings output is a pure function of the
        // edge *set* -- every permutation of the survivor slice yields the
        // identical ring list (this is the invariant, not one lucky order).
        // Three shapes: the real cover's survivors (a simple 16-cycle --
        // pins seed/rotation order), the corner-touch fixture (a branching
        // vertex -- pins the rotational pairing itself), and a net > 1
        // duplicate-edge set (pins the canonical-index tie-break at
        // identical azimuths).
        let cover: Vec<u64> = (0..16u64)
            .map(|n| crate::morton::nested2mort(n, 1))
            .collect();
        let (edges, id_xyz) = survivor_edges(&cover, 1);
        assert_permutation_invariant(&edges, &id_xyz);

        let (id_xyz, edges) = corner_touch_fixture();
        assert_permutation_invariant(&edges, &id_xyz);

        let mut dup = edges.clone();
        dup.extend_from_slice(&edges[..4]); // east diamond twice (net = 2)
        assert_permutation_invariant(&dup, &id_xyz);
        let rings = chain_rings(&dup, &id_xyz);
        assert_eq!(rings.len(), 3, "duplicated diamond must chain twice");
        assert_eq!(rings[0], rings[1]); // the two east copies are identical
    }

    #[test]
    fn corner_touch_pairs_into_simple_rings() {
        // issue #155 ruling: at a non-manifold corner-touch vertex the
        // rotational pairing pinches the boundary into simple rings touching
        // at a point (GeoJSON-valid) -- never one self-intersecting
        // figure-eight.
        let (id_xyz, edges) = corner_touch_fixture();
        let rings = chain_rings(&edges, &id_xyz);
        assert_eq!(
            rings.len(),
            2,
            "pinch must yield two rings, not a figure-eight"
        );
        // exact pairing: each ring stays inside its own diamond, T once each.
        assert_eq!(rings[0], vec![id_xyz[0], id_xyz[3], id_xyz[2], id_xyz[1]]);
        assert_eq!(rings[1], vec![id_xyz[0], id_xyz[4], id_xyz[5], id_xyz[6]]);
    }

    #[test]
    fn pole_cap_stitches_to_one_ring_with_pole_vertex() {
        // one segment running +180 -> -180 around the pole, stitched through -90.
        let segs = vec![vec![
            (180.0, -80.0),
            (90.0, -80.0),
            (0.0, -80.0),
            (-90.0, -80.0),
            (-180.0, -80.0),
        ]];
        let rings = stitch_segments(segs, -90.0);
        assert_eq!(rings.len(), 1);
        assert!(rings[0].iter().any(|p| (p.1 + 90.0).abs() < 1e-9)); // pole vertex
    }
}

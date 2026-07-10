//! Native dissolved-outline emit for a morton cover (issue #71, phase 6).
//!
//! This is the Rust port of `mortie/geometry.py`'s edge-cancellation dissolve
//! (the Python remains the reference/oracle in the tests).  The pipeline:
//!
//! 1. Each cell's boundary is `step` points per edge of unit vectors; every
//!    boundary point is integer-snapped to a vertex id, so a corner/sub-edge
//!    point shared by two neighbours collapses to one id and their shared edge
//!    cancels exactly (no floating tolerance search).
//! 2. The surviving directed edges chain into rings; at a non-manifold
//!    corner-touch vertex the next edge is the smallest anticlockwise turn from
//!    the reversed arrival (right-hand rule), keeping rings simple.
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
fn cover_area(morton: &[u64]) -> f64 {
    morton
        .iter()
        .map(|&w| std::f64::consts::PI / (3.0 * 4f64.powi(mort2nested(w).1 as i32)))
        .sum()
}

/// Dissolve a morton cover into classified planar (lon, lat) rings.
///
/// Errs (with a message for a Python `ValueError`) when the cover spans
/// near or over a hemisphere, where the winding-sign normalisation of
/// [`classify_and_split`] is ambiguous.
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
    Ok(classify_and_split(rings))
}

// ── edge-cancellation: cover → boundary rings (unit vectors) ───────────────

fn boundary_rings_xyz(morton: &[u64], step: u32) -> Vec<Vec<Vec3>> {
    if morton.is_empty() {
        return Vec::new();
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
    chain_rings(&survivors, &id_xyz)
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

fn chain_rings(survivors: &[(u32, u32)], id_xyz: &[Vec3]) -> Vec<Vec<Vec3>> {
    let two_pi = std::f64::consts::TAU;
    let az: Vec<f64> = survivors
        .iter()
        .map(|&(a, b)| tangent_azimuth(&id_xyz[a as usize], &id_xyz[b as usize]))
        .collect();
    let mut alive: Vec<bool> = vec![true; survivors.len()];
    let mut by_start: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, &(a, _)) in survivors.iter().enumerate() {
        by_start.entry(a).or_default().push(i);
    }

    let mut rings = Vec::new();
    for seed in 0..survivors.len() {
        if !alive[seed] {
            continue;
        }
        let seed_start = survivors[seed].0;
        let mut cur = seed;
        let mut chain: Vec<u32> = Vec::new();
        loop {
            if !alive[cur] {
                break;
            }
            alive[cur] = false;
            chain.push(survivors[cur].0);
            let v = survivors[cur].1;
            if v == seed_start {
                break; // ring closed
            }
            let cands: Vec<usize> = by_start
                .get(&v)
                .map(|ix| ix.iter().copied().filter(|&i| alive[i]).collect())
                .unwrap_or_default();
            if cands.is_empty() {
                break;
            }
            cur = if cands.len() == 1 {
                cands[0]
            } else {
                // smallest turn anticlockwise from the reversed arrival.
                let back = tangent_azimuth(&id_xyz[v as usize], &id_xyz[survivors[cur].0 as usize]);
                *cands
                    .iter()
                    .min_by(|&&i, &&j| {
                        let ti = (az[i] - back).rem_euclid(two_pi);
                        let tj = (az[j] - back).rem_euclid(two_pi);
                        ti.partial_cmp(&tj).unwrap()
                    })
                    .unwrap()
            };
        }
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

// Winding-guard margin (issue #108): spherical signed area is defined mod 4π,
// so a cover whose |Σ ring areas| lands near 2π reads the same as its
fn classify_and_split(mut rings_xyz: Vec<Vec<Vec3>>) -> ClassifiedRings {
    let mut out = ClassifiedRings {
        shells: Vec::new(),
        holes: Vec::new(),
    };
    if rings_xyz.is_empty() {
        return out;
    }
    // Normalise global winding so the covered area (exteriors minus holes) is
    // positive — the boundary point order differs between step==1 and step>1.
    // (Spherical signed area is defined mod 4π; the hemisphere guard in
    // `dissolve` keeps the cover well under 2π, so the sign is trustworthy.)
    let mut areas: Vec<f64> = rings_xyz.iter().map(|r| spherical_signed_area(r)).collect();
    let total: f64 = areas.iter().sum();
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
    out
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
    fn sub_hemisphere_cover_still_dissolves() {
        // Base cells 0-3 (2/3 of a hemisphere) stay outside the guard.
        let cover: Vec<u64> = (0..16u64)
            .map(|nest| crate::morton::nested2mort(nest, 1))
            .collect();
        let got = dissolve(&cover, 1).unwrap();
        assert!(!got.shells.is_empty());
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

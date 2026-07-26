//! Ring-set validity: the bucketed self-intersection check (issue #145).
//!
//! [`ring_is_simple`] answers "does this ring cross itself?" with the same
//! architecture the C++ reference uses for `S2Loop::FindValidationError`:
//! bucket edges into a spatial index and test only pairs that share a bucket
//! with the exact crossing predicate — **not** a sweep line.  S2 buckets into
//! a `MutableS2ShapeIndex`; here the buckets are HEALPix cells refined
//! adaptively until each holds few edges, reusing the same cap-overlap
//! culling the coverage descent applies to its own edge lists.  The pairwise
//! predicate is [`super::arcs_cross_sos`] — the identity-keyed exact test the
//! whole crate stands on — so the check inherits its total, reorder-invariant
//! semantics and adds no new geometric primitive.
//!
//! # What "simple" means here, precisely
//!
//! A crossing is a **transversal intersection of two non-adjacent edges**,
//! where adjacency is between consecutive *real* (non-degenerate) edges in
//! cyclic ring order.  Three boundary families are deliberately outside this
//! predicate's scope, each already owned elsewhere:
//!
//! * **Degenerate edges** (consecutive duplicate vertices, by the crate-wide
//!   `1e-12` componentwise test) trace no boundary and are skipped, exactly
//!   as `build_edges` and `ring_crossing_parity` skip them.  The two real
//!   edges joined *through* a duplicate run are treated as adjacent — their
//!   junction is a vertex joint, not a crossing.
//! * **Adjacent edges** share a vertex identity, so testing them would both
//!   violate [`super::arcs_cross_sos`]'s pairwise-distinct-ids precondition
//!   and ask a question ("does an edge cross its own continuation?") whose
//!   transversal reading is always *no*; a spur that doubles back is a
//!   repeated-coordinate configuration, which is
//!   [`super::ring_set_identity_conflict`]'s domain.
//! * **Bit-exact pinches** (the same coordinate revisited at non-adjacent
//!   positions) resolve to cross-or-not by the identity-keyed symbolic
//!   convention — deterministic, but keyed to vertex numbering, which is
//!   again what [`super::ring_set_identity_conflict`] surfaces.
//!
//! # Why a shared bucket must exist
//!
//! The index guarantees no crossing escapes: HEALPix children tile their
//! parent and the 12 base cells tile the sphere, so the intersection point
//! `p` of any crossing pair lies in some leaf cell.  An edge whose cap
//! contains `p` is within `rho` of `p`, hence within `rho + circumradius` of
//! that leaf's centre — precisely the relevance test each leaf's edge list is
//! filtered by (with the same `1e-12` slack the descent uses).  Both edges of
//! the pair therefore appear in that leaf, and the pair is tested.

use std::collections::HashSet;

use crate::cell_geom::{cell_center_vec, cell_corners};

use super::{arcs_cross_sos, cross, dot, norm, normalize, PointId, Vec3, RING_VERTEX_ID_BASE};

/// A leaf bucket stops splitting once it holds this few edges: pair-testing
/// `n ≤ 8` edges costs at most 28 predicate calls, cheaper than another
/// round of 4× relevance filtering.
const LEAF_EDGES: usize = 8;

/// Hard refinement floor.  A bucket that stays crowded at this depth is a
/// genuine convergence point (many edges meeting near one coordinate — e.g. a
/// dense pinch), where further splitting cannot separate the edges; the
/// bucket is pair-tested exhaustively instead.  Order-16 cells are ~3 arcsec
/// across, far below any realistic edge length.
const MAX_DEPTH: u8 = 16;

/// Work budget on index nodes, as a multiple of the real-edge count (plus a
/// floor for tiny rings).  The two-part relevance test below makes the
/// refinement effectively linear on realistic input, but no cap/band test is
/// a partition — an adversarial ring can keep whole subtrees crowded — so
/// the budget converts the worst case from exponential *time* into the
/// honest quadratic of pair-testing whatever buckets remain when it runs
/// out.  Correctness is unaffected either way: a bucket that stops refining
/// is tested exhaustively.
const NODE_BUDGET_PER_EDGE: usize = 48;

/// One real (non-degenerate) edge of the ring, with its bounding cap and the
/// SoS identities of its endpoints.
struct SimpleEdge {
    a: Vec3,
    b: Vec3,
    mid: Vec3,
    cos_rho: f64,
    sin_rho: f64,
    ia: PointId,
    ib: PointId,
    /// Index of the edge's start vertex in the caller's ring — what a
    /// reported crossing pair points back into.
    start: usize,
    /// Position in the filtered real-edge sequence; cyclic neighbours in
    /// this sequence are the adjacent pairs the check must skip.
    seq: usize,
}

/// First pair of non-adjacent edges of `ring` that cross, as the indices of
/// the two edges' start vertices, or `None` when the ring is **simple**.
///
/// See the module docs for the exact scope (transversal crossings of
/// non-adjacent real edges; degenerate edges skipped; pinches and spurs are
/// [`super::ring_set_identity_conflict`]'s domain) and for the argument that
/// the bucketing cannot miss a crossing.  `O(V log V)`-ish on realistic
/// input: the index build touches each edge once per overlapped cell along
/// the refinement path, and pair tests run only inside leaves.
///
/// The verdict is invariant under rotating the vertex list and under
/// reversing the ring (both permute edge identities; the crossing set is
/// geometric).  *Which* pair is reported first is deterministic for a given
/// vertex order but not otherwise canonical.
pub fn ring_is_simple(ring: &[Vec3]) -> Option<(usize, usize)> {
    let n = ring.len();
    if n < 4 {
        // A triangle's edge pairs are all adjacent: nothing to test.
        return None;
    }
    let mut edges: Vec<SimpleEdge> = Vec::with_capacity(n);
    for i in 0..n {
        let (u, v) = (&ring[i], &ring[(i + 1) % n]);
        let dup = (u[0] - v[0]).abs() < 1e-12
            && (u[1] - v[1]).abs() < 1e-12
            && (u[2] - v[2]).abs() < 1e-12;
        if dup || norm(&cross(u, v)) < 1e-15 {
            continue; // traces no boundary; same skip as build_edges
        }
        let mid = normalize(&[u[0] + v[0], u[1] + v[1], u[2] + v[2]]);
        let cos_rho = dot(&mid, u).clamp(-1.0, 1.0);
        let seq = edges.len();
        edges.push(SimpleEdge {
            a: *u,
            b: *v,
            mid,
            cos_rho,
            sin_rho: (1.0 - cos_rho * cos_rho).max(0.0).sqrt(),
            ia: RING_VERTEX_ID_BASE + i as PointId,
            ib: RING_VERTEX_ID_BASE + ((i + 1) % n) as PointId,
            start: i,
            seq,
        });
    }
    let m = edges.len();
    if m < 4 {
        return None; // ≤ 3 real edges: every pair is cyclically adjacent
    }

    // Adjacent in the real-edge sequence (cyclically): the vertex-joint
    // pairs the transversal predicate must not consult.
    let adjacent = |k: usize, l: usize| {
        let d = k.abs_diff(l);
        d == 1 || d == m - 1
    };
    let mut tested: HashSet<(usize, usize)> = HashSet::new();
    let mut test_pair = |e: &SimpleEdge, f: &SimpleEdge| -> bool {
        if adjacent(e.seq, f.seq) {
            return false;
        }
        let key = (e.seq.min(f.seq), e.seq.max(f.seq));
        if !tested.insert(key) {
            return false;
        }
        arcs_cross_sos(&e.a, &e.b, &f.a, &f.b, e.ia, e.ib, f.ia, f.ib)
    };

    // Adaptive HEALPix bucket descent.  Relevance is two conservative tests
    // ANDed, both O(1) dot products:
    //
    // * the coverage descent's cap-overlap cull — the edge's bounding cap
    //   within `rho + circumradius` of the cell centre (same 1e-12 slack) —
    //   which is sharp for short edges but useless for long ones (a 120°
    //   edge's cap covers half the sphere), and
    // * a great-circle band test — the cell centre within `circumradius` of
    //   the edge's great-circle *plane* — which is sharp for long edges
    //   (their plane still separates most cells) and is what keeps a ring of
    //   continent-scale edges from flooding every bucket.
    //
    // Soundness: an intersection point lies ON both edges, so it is inside
    // both caps and on both great circles; any cell containing it passes
    // both tests for both edges.
    let unit_normals: Vec<Vec3> = edges
        .iter()
        .map(|e| normalize(&cross(&e.a, &e.b)))
        .collect();
    let relevant = |k: usize, center: &Vec3, cos_cr: f64, sin_cr: f64| {
        let e = &edges[k];
        let cos_sum = e.cos_rho * cos_cr - e.sin_rho * sin_cr;
        dot(&e.mid, center) >= cos_sum - 1e-12
            && dot(&unit_normals[k], center).abs() <= sin_cr + 1e-12
    };
    let cell_cos_cr = |center: &Vec3, corners: &[Vec3; 4]| {
        let cos_cr = corners
            .iter()
            .map(|c| dot(center, c))
            .fold(1.0_f64, f64::min);
        (cos_cr, (1.0 - cos_cr * cos_cr).max(0.0).sqrt())
    };

    let mut stack: Vec<(u64, u8, Vec<usize>)> = (0..12u64)
        .map(|base| {
            let center = cell_center_vec(0, base);
            let corners = cell_corners(0, base);
            let (cos_cr, sin_cr) = cell_cos_cr(&center, &corners);
            let idx = (0..m)
                .filter(|&k| relevant(k, &center, cos_cr, sin_cr))
                .collect();
            (base, 0u8, idx)
        })
        .collect();

    let mut node_budget = NODE_BUDGET_PER_EDGE * m + 4096;
    while let Some((pixel, depth, idx)) = stack.pop() {
        if idx.len() < 2 {
            continue;
        }
        node_budget = node_budget.saturating_sub(1);
        if idx.len() <= LEAF_EDGES || depth >= MAX_DEPTH || node_budget == 0 {
            for (p, &k) in idx.iter().enumerate() {
                for &l in idx.iter().skip(p + 1) {
                    if test_pair(&edges[k], &edges[l]) {
                        let (e, f) = (&edges[k], &edges[l]);
                        return Some((e.start.min(f.start), e.start.max(f.start)));
                    }
                }
            }
            continue;
        }
        for ch in 0..4u64 {
            let pixel = pixel * 4 + ch;
            let depth = depth + 1;
            let center = cell_center_vec(depth, pixel);
            let corners = cell_corners(depth, pixel);
            let (cos_cr, sin_cr) = cell_cos_cr(&center, &corners);
            let child: Vec<usize> = idx
                .iter()
                .copied()
                .filter(|&k| relevant(k, &center, cos_cr, sin_cr))
                .collect();
            if child.len() >= 2 {
                stack.push((pixel, depth, child));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::super::{latlon_to_unit_vec, parity_filled_robust};
    use super::*;

    fn ring(pts: &[(f64, f64)]) -> Vec<Vec3> {
        pts.iter()
            .map(|&(la, lo)| latlon_to_unit_vec(la, lo))
            .collect()
    }

    /// The naive `O(V²)` reference: every non-adjacent real-edge pair,
    /// straight through the same predicate.  Test-local by design — the
    /// bucketed version must agree with it exactly.
    fn all_pairs_is_simple(ringv: &[Vec3]) -> Option<(usize, usize)> {
        let n = ringv.len();
        if n < 4 {
            return None;
        }
        let mut real: Vec<(usize, Vec3, Vec3, PointId, PointId)> = Vec::new();
        for i in 0..n {
            let (u, v) = (&ringv[i], &ringv[(i + 1) % n]);
            let dup = (u[0] - v[0]).abs() < 1e-12
                && (u[1] - v[1]).abs() < 1e-12
                && (u[2] - v[2]).abs() < 1e-12;
            if dup || norm(&cross(u, v)) < 1e-15 {
                continue;
            }
            real.push((
                i,
                *u,
                *v,
                RING_VERTEX_ID_BASE + i as PointId,
                RING_VERTEX_ID_BASE + ((i + 1) % n) as PointId,
            ));
        }
        let m = real.len();
        if m < 4 {
            return None;
        }
        let mut hit: Option<(usize, usize)> = None;
        for k in 0..m {
            for l in (k + 1)..m {
                if l - k == 1 || (k == 0 && l == m - 1) {
                    continue;
                }
                let (si, a, b, ia, ib) = &real[k];
                let (sj, c, d, ic, id) = &real[l];
                if arcs_cross_sos(a, b, c, d, *ia, *ib, *ic, *id) {
                    let pair = (*si.min(sj), *sj.max(si));
                    if hit.is_none_or(|h| pair < h) {
                        hit = Some(pair);
                    }
                }
            }
        }
        hit
    }

    /// Verdict (not pair) agreement between the bucketed check and the
    /// all-pairs reference, plus rotation/reversal invariance of the verdict.
    fn assert_agrees(name: &str, ringv: &[Vec3]) {
        let bucketed = ring_is_simple(ringv).is_some();
        let naive = all_pairs_is_simple(ringv).is_some();
        assert_eq!(bucketed, naive, "{name}: bucketed vs all-pairs verdict");
        let mut rot = ringv.to_vec();
        rot.rotate_left(ringv.len() / 3);
        assert_eq!(
            ring_is_simple(&rot).is_some(),
            naive,
            "{name}: verdict changed under rotation"
        );
        let rev: Vec<Vec3> = ringv.iter().rev().copied().collect();
        assert_eq!(
            ring_is_simple(&rev).is_some(),
            naive,
            "{name}: verdict changed under reversal"
        );
    }

    #[test]
    fn test_simple_fixtures_are_simple() {
        // The families every other suite leans on: boxes, the crescent, the
        // lat band, the wobbly ring, the hemisphere-plus basin.
        let cases: Vec<(&str, Vec<Vec3>)> = vec![
            (
                "box",
                ring(&[
                    (40.0, -125.0),
                    (40.0, -115.0),
                    (50.0, -115.0),
                    (50.0, -125.0),
                ]),
            ),
            ("crescent", {
                let mut v: Vec<Vec3> = (0..40)
                    .map(|k| latlon_to_unit_vec(5.0, k as f64 * 300.0 / 39.0))
                    .collect();
                v.extend(
                    (0..40).map(|k| latlon_to_unit_vec(10.0, 300.0 - k as f64 * 300.0 / 39.0)),
                );
                v
            }),
            ("band", {
                (0..36)
                    .map(|k| latlon_to_unit_vec(-10.0, k as f64 * 10.0))
                    .collect()
            }),
            (
                "basin",
                ring(&[
                    (10.0, 45.0),
                    (50.0, 45.0),
                    (-10.0, 170.0),
                    (-70.0, 225.0),
                    (-10.0, 280.0),
                ]),
            ),
            ("wobbly", {
                let centre = latlon_to_unit_vec(45.0, 0.0);
                let e1 = normalize(&cross(&[0.0, 0.0, 1.0], &centre));
                let e2 = cross(&centre, &e1);
                (0..96)
                    .map(|k| {
                        let th = k as f64 * std::f64::consts::TAU / 96.0;
                        let r = (97.5 + 12.5 * (3.0 * th).sin()).to_radians();
                        let (sr, cr) = (r.sin(), r.cos());
                        normalize(&[
                            cr * centre[0] + sr * (th.cos() * e1[0] + th.sin() * e2[0]),
                            cr * centre[1] + sr * (th.cos() * e1[1] + th.sin() * e2[1]),
                            cr * centre[2] + sr * (th.cos() * e1[2] + th.sin() * e2[2]),
                        ])
                    })
                    .collect()
            }),
        ];
        for (name, r) in &cases {
            assert_eq!(ring_is_simple(r), None, "{name} wrongly flagged");
            assert_agrees(name, r);
        }
    }

    #[test]
    fn test_self_intersecting_fixtures_are_flagged() {
        // The bowtie: edges 0 and 2 cross transversally.
        let bowtie = ring(&[(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)]);
        let pair = ring_is_simple(&bowtie);
        assert_eq!(pair, all_pairs_is_simple(&bowtie));
        assert_eq!(
            pair,
            Some((1, 3)),
            "the bowtie's crossing pair is its two diagonals"
        );

        // The lemniscate from the #107 suite: a genuine figure-eight.
        let lemni: Vec<Vec3> = (0..72)
            .map(|k| {
                let t = k as f64 * std::f64::consts::TAU / 72.0;
                latlon_to_unit_vec(30.0 * (2.0 * t).sin() / 2.0, 40.0 * t.cos())
            })
            .collect();
        assert!(
            ring_is_simple(&lemni).is_some(),
            "lemniscate must be flagged"
        );
        assert_agrees("lemniscate", &lemni);

        // A long collinear overlap family ring with a transversal pinch:
        // the meridian spike revisits the lon-45 plane.
        let spike = ring(&[
            (10.0, 45.0),
            (50.0, 45.0),
            (30.0, 60.0),
            (40.0, 30.0),
            (20.0, 60.0),
        ]);
        assert_eq!(
            ring_is_simple(&spike).is_some(),
            all_pairs_is_simple(&spike).is_some()
        );
    }

    #[test]
    fn test_degenerate_and_tiny_rings() {
        let p = latlon_to_unit_vec(10.0, 10.0);
        let q = latlon_to_unit_vec(20.0, 10.0);
        let r = latlon_to_unit_vec(15.0, 20.0);
        assert_eq!(ring_is_simple(&[p, q, r]), None, "triangle");
        assert_eq!(ring_is_simple(&[p, q]), None, "degenerate 2-ring");
        assert_eq!(ring_is_simple(&[p, p, p, p]), None, "all-duplicate ring");
        // Duplicate vertex mid-ring: the joined edges are adjacent, not
        // crossing; the deduped twin agrees.
        let dup = vec![p, p, q, r];
        assert_eq!(ring_is_simple(&dup), None, "duplicate-vertex ring");
    }

    #[test]
    fn test_randomized_agreement_with_all_pairs() {
        // splitmix64, as the #103 fuzz uses: deterministic, no new deps.
        let mut state: u64 = 0x51_a5e1;
        let mut next = || {
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            (z ^ (z >> 31)) as f64 / u64::MAX as f64
        };
        let (mut simple_seen, mut crossing_seen) = (0u32, 0u32);
        for case in 0..60 {
            let n = 8 + (next() * 56.0) as usize;
            let star = case % 2 == 0;
            let clat = -60.0 + 120.0 * next();
            let clon = 360.0 * next();
            let centre = latlon_to_unit_vec(clat, clon);
            let e1 = normalize(&cross(
                &centre,
                if centre[0].abs() < 0.9 {
                    &[1.0, 0.0, 0.0]
                } else {
                    &[0.0, 1.0, 0.0]
                },
            ));
            let e2 = cross(&centre, &e1);
            // Stars (monotone azimuth, jittered radius) are simple by
            // construction — a radial-monotone boundary cannot revisit an
            // azimuth.  Tangles take the same vertices in shuffled order,
            // which is self-intersecting almost surely for n ≥ 6.
            let mut ringv: Vec<Vec3> = (0..n)
                .map(|k| {
                    let th = k as f64 * std::f64::consts::TAU / n as f64;
                    let r = (20.0 + 8.0 * next()).to_radians();
                    let (sr, cr) = (r.sin(), r.cos());
                    normalize(&[
                        cr * centre[0] + sr * (th.cos() * e1[0] + th.sin() * e2[0]),
                        cr * centre[1] + sr * (th.cos() * e1[1] + th.sin() * e2[1]),
                        cr * centre[2] + sr * (th.cos() * e1[2] + th.sin() * e2[2]),
                    ])
                })
                .collect();
            if !star {
                // Fisher–Yates on the same splitmix stream.
                for k in (1..n).rev() {
                    let j = (next() * (k + 1) as f64) as usize % (k + 1);
                    ringv.swap(k, j);
                }
            }
            let naive = all_pairs_is_simple(&ringv).is_some();
            assert_eq!(
                ring_is_simple(&ringv).is_some(),
                naive,
                "case {case} (n={n}, star={star})"
            );
            if naive {
                crossing_seen += 1;
            } else {
                simple_seen += 1;
            }
        }
        // The sweep must exercise both verdicts, or it proves nothing.
        assert!(simple_seen >= 10, "only {simple_seen} simple cases");
        assert!(crossing_seen >= 10, "only {crossing_seen} crossing cases");
    }

    #[test]
    #[ignore = "wall-clock scaling evidence; run explicitly with --release"]
    fn test_bucketed_scaling_smoke() {
        // Local stand-in for the criterion benches (which need the Linux CI
        // link): the bucketed check at basin scale and at 1M vertices must
        // land in interactive time under --release.  Bounds are generous —
        // this guards against a return of the exponential-refinement failure
        // mode, not against millisecond drift (CodSpeed owns that).
        let wiggly: Vec<Vec3> = (0..22_000)
            .map(|i| {
                let th = std::f64::consts::TAU * (i as f64) / 22_000.0;
                let r = 12.0 + 3.0 * (7.0 * th).sin();
                latlon_to_unit_vec(-60.0 + r * th.cos(), 30.0 + 1.5 * r * th.sin())
            })
            .collect();
        let t = std::time::Instant::now();
        assert_eq!(ring_is_simple(&wiggly), None);
        let wiggly_ms = t.elapsed().as_millis();
        // 200k here rather than 1M: this runs in the unoptimized test
        // profile (~25x release), where 1M costs ~130 s; the criterion bench
        // carries the 1M case on the optimized CI build.
        let dense: Vec<Vec3> = (0..200_000)
            .map(|i| {
                let th = std::f64::consts::TAU * (i as f64) / 2e5;
                latlon_to_unit_vec(10.0 + 6.0 * th.cos(), 6.0 * th.sin())
            })
            .collect();
        let t = std::time::Instant::now();
        assert_eq!(ring_is_simple(&dense), None);
        let dense_ms = t.elapsed().as_millis();
        eprintln!("ring_is_simple (debug): 22k wiggly {wiggly_ms} ms, 200k circle {dense_ms} ms");
        assert!(wiggly_ms < 5_000, "22k took {wiggly_ms} ms");
        assert!(dense_ms < 60_000, "200k took {dense_ms} ms");
    }

    #[test]
    fn test_simplicity_consistent_with_the_predicates() {
        // A ring the checker calls simple must satisfy the W ∈ {0, 1}
        // assumption the hemisphere-plus branch documents: spot-check that
        // a simple hemisphere-plus ring's parity verdicts partition the
        // sphere into exactly the two documented regions (no W = 2 pocket).
        let basin = ring(&[
            (10.0, 45.0),
            (50.0, 45.0),
            (-10.0, 170.0),
            (-70.0, 225.0),
            (-10.0, 280.0),
        ]);
        assert_eq!(ring_is_simple(&basin), None);
        let inside: u32 = (0..500)
            .filter(|&k| {
                let lat = -85.0 + (k / 20) as f64 * 7.0;
                let lon = (k % 20) as f64 * 18.0;
                parity_filled_robust(&latlon_to_unit_vec(lat, lon), std::slice::from_ref(&basin))
            })
            .count() as u32;
        // The as-given interior is ~52% of the sphere; a winding pocket
        // would push the sampled fraction far outside this window.
        assert!(
            (150..350).contains(&inside),
            "sampled inside = {inside}/500"
        );
    }
}

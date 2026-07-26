//! The hemisphere-plus witness construction (issue #107), split from
//! `sphere.rs` per the module-size discussion on PR #138 (question 2,
//! option (b)) — a pure move, no behavior change.  That split was deliberately
//! scope-limited: it took `sphere.rs` from 1775 to 1579 lines, still well over
//! the project's ~1000-line soft limit.
//!
//! Only a ring that fits no bounding cap needs a constructed reference point;
//! everything here serves [`ring_witness`], which finds one whose verdict the
//! subtended-angle sum proves outright.  See `sphere.rs`'s "the normalization
//! constant" section for why the witness carries a *proved* verdict rather
//! than an assumed one, and [`super::RingRef`] for what it feeds.

use super::{cross, dot, norm, normalize, ring_winding_at, RingRef, Vec3};

/// How many candidate edges [`ring_witness`] may try.
///
/// The loop returns on the first candidate that works, so this is a bound on
/// *failure*, not a cost the ordinary ring pays — and it is set high because
/// the previous value of `8` was not a bound on anything useful.  On a `k`-fold
/// symmetric ring the edge lengths tie in groups of `k`, so eight candidates
/// bought `8 / k` distinct geometries: review found ~9% of simple
/// hemisphere-plus rings returning [`RingRef::Undecidable`] with two thirds of
/// their edges definitive, simply never reached.  Only hemisphere-plus rings
/// get here at all, the result is cached per ring ([`super::RingRefs`]), and the
/// walk stops at the first witness.
const WITNESS_EDGE_SAMPLES: usize = 256;

/// Angle between two vectors, which need not be unit-length.
///
/// `atan2` of the cross-product magnitude against the dot product, which stays
/// accurate where `dot(a, b).acos()` collapses: below ~`2e-8` rad the dot
/// product rounds to exactly `1.0` and `acos` returns exactly `0.0`.  That
/// underflow was issue #107's 6 mm-edge regression — a zero-length flank step
/// left the reference point sitting *on* the boundary.
pub(super) fn angle_between(a: &Vec3, b: &Vec3) -> f64 {
    norm(&cross(a, b)).atan2(dot(a, b))
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
/// winding contract of [`super::point_in_ring_robust`].
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
/// ([`super::RingRefs`]).
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
pub(super) fn ring_flanks(ring: &[Vec3], i: usize) -> Option<(Vec3, Vec3)> {
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
/// Longest edge first, ties broken by index.  Two honest caveats: exact ties
/// are resolved by index, which rotation changes, and vertex density *does*
/// influence edge length, so an adversary retains some grip on the order.  What
/// it buys over #107's `step_by(n / 8)` is that a stride samples a fixed
/// *fraction* of a crowded region — that is how the lemniscate got all eight of
/// its candidates onto one lobe — whereas length ordering has no such handle,
/// and a run of tied lengths is now covered by trying many more candidates
/// rather than eight.
///
/// None of this is load-bearing for the ring's *orientation*, which is where
/// #107's sampling actually went wrong: that comes from
/// [`super::ring_turning_with_error`], a sum
/// over every vertex. The ordering only decides which candidate is examined
/// first, and every candidate must clear the same proof.
pub(super) fn ring_witness(ring: &[Vec3]) -> RingRef {
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

//! Spherical primitives for the hierarchical region coverer.
//!
//! Everything here operates on **unit 3-vectors** on the sphere.  The two core
//! predicates are [`orient`] (the sign of a scalar triple product) and
//! [`arcs_cross`] (do two great-circle arcs cross?), built from it.  On top of
//! those sit two interchangeable point-in-polygon backends — [`gnomonic_pip`]
//! (fast, valid within a hemisphere of the test point) and
//! [`point_in_ring_edgecross`] (orientation-only, the basis for hemisphere+
//! support, issue #22) — and [`parity_filled`], the even-odd rule over a
//! *ring-set* that gives holes and multipart geometry for free (see issue #30).
//!
//! NOTE: `dead_code` is allowed while this module is built out ahead of being
//! wired into `coverage.rs` (Phase C of the #30 plan); the allow is removed once
//! the descent consumes these functions.

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

/// Do the great-circle arcs `a → b` and `c → d` cross?
///
/// Uses the standard 4-orientation test: the arcs cross iff `c` and `d` lie on
/// opposite sides of great circle `AB` **and** `a` and `b` lie on opposite
/// sides of great circle `CD`.
///
/// Precondition: each arc is **minor** (shorter than a hemisphere), which holds
/// for HEALPix cell edges and for polygon edges between consecutive vertices.
/// An exactly-touching configuration (a zero orientation) reports `false`; the
/// classifier pairs this with a vertex-in-cell check to catch grazes.
pub fn arcs_cross(a: &Vec3, b: &Vec3, c: &Vec3, d: &Vec3) -> bool {
    let d1 = orient(a, b, c);
    let d2 = orient(a, b, d);
    if (d1 > 0.0) == (d2 > 0.0) {
        return false; // c, d on the same side of AB
    }
    let d3 = orient(c, d, a);
    let d4 = orient(c, d, b);
    (d3 > 0.0) != (d4 > 0.0) // a, b on opposite sides of CD
}

// ── point-in-ring backends ───────────────────────────────────────────────

/// Gnomonic point-in-ring test: project the ring onto the tangent plane at
/// `center` (gnomonic projection turns great-circle edges into straight lines)
/// and run a 2-D winding-number test, with the test point at the origin.
///
/// Valid when every ring vertex lies within a hemisphere of `center` (the
/// default fast path); for larger rings use [`point_in_ring_edgecross`].
pub fn gnomonic_pip(center: &Vec3, ring: &[Vec3]) -> bool {
    let n = ring.len();
    if n < 3 {
        return false;
    }

    // Orthonormal tangent basis (a, b) at `center`.
    let ref_vec = if center[2].abs() < 0.9 {
        [0.0, 0.0, 1.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let dot_rc = dot(&ref_vec, center);
    let a = normalize(&[
        ref_vec[0] - dot_rc * center[0],
        ref_vec[1] - dot_rc * center[1],
        ref_vec[2] - dot_rc * center[2],
    ]);
    let b = cross(center, &a); // unit length: center ⟂ a, both unit

    let mut proj_x = Vec::with_capacity(n);
    let mut proj_y = Vec::with_capacity(n);
    for v in ring {
        let d = dot(center, v);
        // Vertices on the far hemisphere (d <= 0) are clamped to avoid a sign
        // flip through infinity; for ≤hemisphere rings this never triggers.
        let d = if d > 1e-12 { d } else { 1e-12 };
        proj_x.push(dot(&a, v) / d);
        proj_y.push(dot(&b, v) / d);
    }
    winding_number_2d(0.0, 0.0, &proj_x, &proj_y)
}

/// 2-D winding-number point-in-polygon test (Hormann & Agathos 2001).
pub fn winding_number_2d(tx: f64, ty: f64, poly_x: &[f64], poly_y: &[f64]) -> bool {
    let n = poly_x.len();
    let mut winding: i32 = 0;
    let mut j = n - 1;
    for i in 0..n {
        let yi = poly_y[j] - ty;
        let yj = poly_y[i] - ty;
        if yi <= 0.0 {
            if yj > 0.0 {
                let cross = (poly_x[j] - tx) * yj - (poly_x[i] - tx) * yi;
                if cross > 0.0 {
                    winding += 1;
                }
            }
        } else if yj <= 0.0 {
            let cross = (poly_x[j] - tx) * yj - (poly_x[i] - tx) * yi;
            if cross < 0.0 {
                winding -= 1;
            }
        }
        j = i;
    }
    winding != 0
}

/// Axis of the smallest cap loosely bounding `ring`: the normalized vertex sum.
/// Returns `+z` as a fallback for a balanced (≈ whole-sphere) ring.
pub fn ring_cap_axis(ring: &[Vec3]) -> Vec3 {
    let mut s = [0.0, 0.0, 0.0];
    for v in ring {
        s[0] += v[0];
        s[1] += v[1];
        s[2] += v[2];
    }
    if norm(&s) < 1e-12 {
        [0.0, 0.0, 1.0]
    } else {
        normalize(&s)
    }
}

/// Edge-crossing point-in-ring test using only [`orient`].
///
/// Counts how many ring edges the arc from a reference point to `p` crosses;
/// `p` is inside iff that count is odd and the reference is outside.  The
/// reference is the antipode of the ring's cap axis, which is outside the ring
/// whenever the ring fits within a hemisphere.  This is the orientation-only
/// path that hemisphere+ support (#22) will generalize; it does not assume a
/// gnomonic projection is valid.
pub fn point_in_ring_edgecross(p: &Vec3, ring: &[Vec3]) -> bool {
    let n = ring.len();
    if n < 3 {
        return false;
    }
    let axis = ring_cap_axis(ring);
    // Reference ~90° from the cap axis: outside any sub-hemisphere ring, and —
    // unlike the antipode of the axis — never near-antipodal to an interior test
    // point, so the arc reference→p stays a well-behaved minor arc.
    let t = if axis[2].abs() < 0.9 {
        [0.0, 0.0, 1.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let reference = normalize(&[
        t[0] - dot(&t, &axis) * axis[0],
        t[1] - dot(&t, &axis) * axis[1],
        t[2] - dot(&t, &axis) * axis[2],
    ]);
    let mut crossings = 0u32;
    let mut j = n - 1;
    for i in 0..n {
        if arcs_cross(&reference, p, &ring[j], &ring[i]) {
            crossings += 1;
        }
        j = i;
    }
    crossings % 2 == 1
}

// ── ring-set fill (even-odd) ─────────────────────────────────────────────

/// Which point-in-ring backend to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipBackend {
    /// Gnomonic projection + winding number (fast; ≤ hemisphere).
    Gnomonic,
    /// Orientation-only edge crossing (hemisphere+; issue #22).
    EdgeCross,
}

/// Pick a backend from the geometry: gnomonic while the whole ring-set fits
/// comfortably inside a hemisphere, otherwise edge-crossing.
pub fn choose_backend(rings: &[Vec<Vec3>]) -> PipBackend {
    // Combined cap axis over every vertex of every ring.
    let mut s = [0.0, 0.0, 0.0];
    for ring in rings {
        for v in ring {
            s[0] += v[0];
            s[1] += v[1];
            s[2] += v[2];
        }
    }
    let axis = if norm(&s) < 1e-12 {
        return PipBackend::EdgeCross; // balanced ⇒ spans a hemisphere or more
    } else {
        normalize(&s)
    };
    let extent = rings
        .iter()
        .flat_map(|r| r.iter())
        .map(|v| dot(&axis, v).clamp(-1.0, 1.0).acos())
        .fold(0.0_f64, f64::max);
    // ~85°: stay clear of the gnomonic projection's 90° singularity.
    if extent < 85.0_f64.to_radians() {
        PipBackend::Gnomonic
    } else {
        PipBackend::EdgeCross
    }
}

/// Is `p` inside the filled region defined by `rings` under the **even-odd**
/// rule — i.e. inside an *odd* number of rings?
///
/// This is orientation-free and handles holes (a point in the hole is inside
/// both the outer ring and the hole ring → even → empty) and multipart geometry
/// (separate outer rings) with one predicate.
pub fn parity_filled(p: &Vec3, rings: &[Vec<Vec3>], backend: PipBackend) -> bool {
    let mut inside = false;
    for ring in rings {
        let in_ring = match backend {
            PipBackend::Gnomonic => gnomonic_pip(p, ring),
            PipBackend::EdgeCross => point_in_ring_edgecross(p, ring),
        };
        if in_ring {
            inside = !inside;
        }
    }
    inside
}

// ── tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(pts: &[(f64, f64)]) -> Vec<Vec3> {
        pts.iter().map(|&(la, lo)| latlon_to_unit_vec(la, lo)).collect()
    }

    #[test]
    fn test_orient_basis() {
        let x = [1.0, 0.0, 0.0];
        let y = [0.0, 1.0, 0.0];
        let z = [0.0, 0.0, 1.0];
        assert!(orient(&x, &y, &z) > 0.0, "right-handed basis is positive");
        assert!(orient(&x, &z, &y) < 0.0, "swapped is negative");
        // Coplanar vectors give zero.
        assert!(orient(&x, &y, &x).abs() < 1e-15);
    }

    #[test]
    fn test_arcs_cross_basic() {
        // Two arcs straddling (0,0): a N-S arc and an E-W arc.
        let ns = (ring(&[(-10.0, 0.0)])[0], ring(&[(10.0, 0.0)])[0]);
        let ew = (ring(&[(0.0, -10.0)])[0], ring(&[(0.0, 10.0)])[0]);
        assert!(arcs_cross(&ns.0, &ns.1, &ew.0, &ew.1), "should cross at origin");
        // Parallel-ish, offset arcs that do not cross.
        let a = ring(&[(20.0, -10.0), (20.0, 10.0)]);
        let b = ring(&[(40.0, -10.0), (40.0, 10.0)]);
        assert!(!arcs_cross(&a[0], &a[1], &b[0], &b[1]), "disjoint arcs");
    }

    #[test]
    fn test_arcs_cross_endpoint_disjoint() {
        // Arcs sharing no span: one near equator, one far away.
        let a = ring(&[(0.0, 0.0), (0.0, 5.0)]);
        let b = ring(&[(50.0, 50.0), (55.0, 55.0)]);
        assert!(!arcs_cross(&a[0], &a[1], &b[0], &b[1]));
    }

    #[test]
    fn test_gnomonic_vs_edgecross_parity() {
        // A mid-latitude square; both backends must agree on a grid of points.
        let sq = ring(&[(40.0, -125.0), (40.0, -115.0), (50.0, -115.0), (50.0, -125.0)]);
        let rings = vec![sq.clone()];
        for lat in [38.0, 42.0, 45.0, 48.0, 52.0] {
            for lon in [-128.0, -123.0, -120.0, -117.0, -112.0] {
                let p = latlon_to_unit_vec(lat, lon);
                let g = parity_filled(&p, &rings, PipBackend::Gnomonic);
                let e = parity_filled(&p, &rings, PipBackend::EdgeCross);
                assert_eq!(g, e, "backends disagree at ({lat},{lon})");
            }
        }
    }

    #[test]
    fn test_even_odd_donut() {
        // Outer 20°-ish box with a smaller hole, both centered at (45,-120).
        let outer = ring(&[(35.0, -130.0), (35.0, -110.0), (55.0, -110.0), (55.0, -130.0)]);
        let hole = ring(&[(42.0, -123.0), (42.0, -117.0), (48.0, -117.0), (48.0, -123.0)]);
        let rings = vec![outer, hole];
        for backend in [PipBackend::Gnomonic, PipBackend::EdgeCross] {
            // In the annulus (inside outer, outside hole) → filled.
            let annulus = latlon_to_unit_vec(38.0, -120.0);
            assert!(parity_filled(&annulus, &rings, backend), "annulus filled ({backend:?})");
            // In the hole → empty.
            let in_hole = latlon_to_unit_vec(45.0, -120.0);
            assert!(!parity_filled(&in_hole, &rings, backend), "hole empty ({backend:?})");
            // Outside everything → empty.
            let outside = latlon_to_unit_vec(10.0, -120.0);
            assert!(!parity_filled(&outside, &rings, backend), "outside empty ({backend:?})");
        }
    }

    #[test]
    fn test_multipart_parity() {
        // Two disjoint boxes; a point in either is filled, between them empty.
        let part_a = ring(&[(40.0, -125.0), (40.0, -120.0), (45.0, -120.0), (45.0, -125.0)]);
        let part_b = ring(&[(40.0, -110.0), (40.0, -105.0), (45.0, -105.0), (45.0, -110.0)]);
        let rings = vec![part_a, part_b];
        let in_a = latlon_to_unit_vec(42.0, -122.0);
        let in_b = latlon_to_unit_vec(42.0, -107.0);
        let between = latlon_to_unit_vec(42.0, -115.0);
        for backend in [PipBackend::Gnomonic, PipBackend::EdgeCross] {
            assert!(parity_filled(&in_a, &rings, backend));
            assert!(parity_filled(&in_b, &rings, backend));
            assert!(!parity_filled(&between, &rings, backend));
        }
    }

    #[test]
    fn test_choose_backend() {
        let small = vec![ring(&[(40.0, -125.0), (40.0, -115.0), (50.0, -115.0), (50.0, -125.0)])];
        assert_eq!(choose_backend(&small), PipBackend::Gnomonic);
        // A ring spanning most of a hemisphere forces edge-crossing.
        let huge = vec![ring(&[(80.0, 0.0), (0.0, 90.0), (-80.0, 180.0), (0.0, -90.0)])];
        assert_eq!(choose_backend(&huge), PipBackend::EdgeCross);
    }

    #[test]
    fn test_edgecross_southern_polygon() {
        // Near-polar triangle (southern hemisphere) — edge-cross must agree with
        // gnomonic on clearly-inside and clearly-outside points.
        let tri = ring(&[(-70.0, 0.0), (-70.0, 60.0), (-80.0, 30.0)]);
        let rings = vec![tri];
        // NB: the top edge is a great-circle arc bulging poleward to ~-72.5° at
        // lon 30, so the interior point must be safely below that.
        let inside = latlon_to_unit_vec(-75.0, 30.0);
        let outside = latlon_to_unit_vec(-60.0, 30.0);
        for backend in [PipBackend::Gnomonic, PipBackend::EdgeCross] {
            assert!(parity_filled(&inside, &rings, backend), "inside ({backend:?})");
            assert!(!parity_filled(&outside, &rings, backend), "outside ({backend:?})");
        }
    }
}

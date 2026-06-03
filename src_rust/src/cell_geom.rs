//! HEALPix cell geometry and ring-set classification for the hierarchical
//! coverer (issue #30, Phase B).
//!
//! A cell is classified against a *ring-set* (outer rings + holes, even-odd
//! fill — see [`crate::sphere`]) as one of:
//!
//! * [`Classification::Inside`]  — the whole cell is filled,
//! * [`Classification::Outside`] — the whole cell is empty,
//! * [`Classification::Straddle`] — some ring boundary passes through the cell.
//!
//! The descent keeps `Inside`, prunes `Outside`, and refines `Straddle`.
//!
//! NOTE: `dead_code` is allowed until the descent (Phase C) consumes these.
#![allow(dead_code)]

use crate::geo2mort::boundaries_scalar;
use crate::sphere::{arcs_cross, normalize, orient, parity_filled, PipBackend, Vec3};

/// Result of classifying a cell against a ring-set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Classification {
    Inside,
    Outside,
    Straddle,
}

/// The four corner unit-vectors of a NESTED cell, in boundary order.
#[inline]
pub fn cell_corners(depth: u8, pixel: u64) -> [Vec3; 4] {
    let xyz = boundaries_scalar(depth, pixel); // [x[4], y[4], z[4]]
    let mut corners = [[0.0; 3]; 4];
    for i in 0..4 {
        corners[i] = [xyz[0][i], xyz[1][i], xyz[2][i]];
    }
    corners
}

/// Centroid (normalized vertex mean) of a cell — a point strictly interior to
/// the convex cell, used as a known-inside reference.
#[inline]
pub fn cell_centroid(corners: &[Vec3; 4]) -> Vec3 {
    normalize(&[
        corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0],
        corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1],
        corners[0][2] + corners[1][2] + corners[2][2] + corners[3][2],
    ])
}

/// Is `p` strictly inside the (convex) spherical cell?
///
/// A HEALPix cell is convex, so `p` is inside iff it lies on the same side of
/// every directed cell edge as the cell centroid.  Orientation-only, so it does
/// not depend on the cell's corner winding direction.  Points on an edge count
/// as outside.
pub fn point_in_cell(p: &Vec3, corners: &[Vec3; 4]) -> bool {
    let centroid = cell_centroid(corners);
    for i in 0..4 {
        let j = (i + 1) % 4;
        let s_p = orient(&corners[i], &corners[j], p);
        let s_c = orient(&corners[i], &corners[j], &centroid);
        if (s_p > 0.0) != (s_c > 0.0) {
            return false;
        }
    }
    true
}

/// Classify a cell against a ring-set.
///
/// The cell is **uniform** (entirely filled or entirely empty) iff no ring edge
/// crosses any cell edge *and* no ring vertex lies inside the cell; in that case
/// the cell centroid's even-odd parity decides `Inside` vs `Outside`.  Any other
/// configuration means a ring boundary passes through the cell → `Straddle`.
///
/// The two checks together catch every boundary-through-cell case: an edge
/// slicing the cell (corner-free crossing included), and a whole small ring —
/// e.g. a tiny polygon or a hole — sitting inside the cell with no edge crossing.
pub fn classify(
    depth: u8,
    pixel: u64,
    rings: &[Vec<Vec3>],
    backend: PipBackend,
) -> Classification {
    let corners = cell_corners(depth, pixel);

    // (1) Any ring edge crossing any cell edge → boundary passes through.
    // TODO(perf, Phase C): cull rings/edges by a per-subtree bounding cap so
    // deep cells only test nearby edges (keeps this ≈ O(local edges)).
    for ring in rings {
        let m = ring.len();
        if m < 2 {
            continue;
        }
        let mut rj = m - 1;
        for ri in 0..m {
            for ci in 0..4 {
                let cj = (ci + 1) % 4;
                if arcs_cross(&ring[rj], &ring[ri], &corners[ci], &corners[cj]) {
                    return Classification::Straddle;
                }
            }
            rj = ri;
        }
    }

    // (2) Any ring vertex inside the cell → a whole ring (tiny polygon / hole)
    // sits within the cell even though no edge crosses a cell edge.
    for ring in rings {
        for v in ring {
            if point_in_cell(v, &corners) {
                return Classification::Straddle;
            }
        }
    }

    // (3) Uniform cell: decide by the centroid's even-odd parity.
    let centroid = cell_centroid(&corners);
    if parity_filled(&centroid, rings, backend) {
        Classification::Inside
    } else {
        Classification::Outside
    }
}

// ── tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo2mort::ang2pix_scalar;
    use crate::sphere::latlon_to_unit_vec;

    fn ring(pts: &[(f64, f64)]) -> Vec<Vec3> {
        pts.iter().map(|&(la, lo)| latlon_to_unit_vec(la, lo)).collect()
    }

    /// NESTED pixel at `depth` whose cell contains (lat, lon).
    fn cell_at(depth: u8, lat: f64, lon: f64) -> u64 {
        ang2pix_scalar(depth, lon, lat)
    }

    // A 20°×20° square, mid-latitude, plus a centered 6° hole (for donut tests).
    fn outer() -> Vec<Vec3> {
        ring(&[(35.0, -130.0), (35.0, -110.0), (55.0, -110.0), (55.0, -130.0)])
    }
    fn hole() -> Vec<Vec3> {
        ring(&[(42.0, -123.0), (42.0, -117.0), (48.0, -117.0), (48.0, -123.0)])
    }

    #[test]
    fn test_classify_inside() {
        let rings = vec![outer()];
        // Small deep cell well inside the square, away from edges.
        let pix = cell_at(8, 45.0, -120.0);
        assert_eq!(classify(8, pix, &rings, PipBackend::Gnomonic), Classification::Inside);
    }

    #[test]
    fn test_classify_outside() {
        let rings = vec![outer()];
        let pix = cell_at(8, 0.0, 0.0); // far away
        assert_eq!(classify(8, pix, &rings, PipBackend::Gnomonic), Classification::Outside);
    }

    #[test]
    fn test_classify_straddle_edge() {
        let rings = vec![outer()];
        // Cell containing a polygon vertex → boundary passes through it.
        let pix = cell_at(8, 35.0, -130.0);
        assert_eq!(classify(8, pix, &rings, PipBackend::Gnomonic), Classification::Straddle);
    }

    #[test]
    fn test_classify_contains_tiny_polygon() {
        // A coarse (large) cell with a tiny polygon entirely inside it: no edge
        // crosses a cell edge, but the ring vertices are inside the cell.
        let depth = 3;
        let pix = cell_at(depth, 45.0, -120.0);
        let (lon, lat) = crate::geo2mort::pix2ang_scalar(depth, pix);
        // Tiny 0.2° square around the cell centre.
        let tiny = ring(&[
            (lat - 0.1, lon - 0.1),
            (lat - 0.1, lon + 0.1),
            (lat + 0.1, lon + 0.1),
            (lat + 0.1, lon - 0.1),
        ]);
        assert_eq!(classify(depth, pix, &vec![tiny], PipBackend::Gnomonic), Classification::Straddle);
    }

    #[test]
    fn test_classify_donut_annulus_inside() {
        let rings = vec![outer(), hole()];
        // Inside outer, outside hole → filled.
        let pix = cell_at(9, 38.0, -120.0);
        assert_eq!(classify(9, pix, &rings, PipBackend::Gnomonic), Classification::Inside);
    }

    #[test]
    fn test_classify_donut_hole_outside() {
        let rings = vec![outer(), hole()];
        // Inside the hole → empty.
        let pix = cell_at(9, 45.0, -120.0);
        assert_eq!(classify(9, pix, &rings, PipBackend::Gnomonic), Classification::Outside);
    }

    #[test]
    fn test_classify_donut_hole_rim_straddle() {
        let rings = vec![outer(), hole()];
        // Cell on the hole boundary (contains a hole vertex).
        let pix = cell_at(9, 42.0, -123.0);
        assert_eq!(classify(9, pix, &rings, PipBackend::Gnomonic), Classification::Straddle);
    }

    #[test]
    fn test_point_in_cell() {
        let depth = 6;
        let pix = cell_at(depth, 45.0, -120.0);
        let corners = cell_corners(depth, pix);
        let centroid = cell_centroid(&corners);
        assert!(point_in_cell(&centroid, &corners), "centroid is inside");
        // A point clearly outside the small cell.
        let far = latlon_to_unit_vec(45.0, -100.0);
        assert!(!point_in_cell(&far, &corners));
    }
}

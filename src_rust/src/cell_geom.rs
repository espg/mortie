//! HEALPix cell geometry helpers for the hierarchical coverer (issue #30).
//!
//! Just the primitives the descent (`coverage.rs`) needs: a cell's corner and
//! centre unit-vectors, and the bounding [`Cap`] of a ring-set used to prune
//! subtrees that cannot intersect the filled region.
//!
//! NOTE: HEALPix cell edges are *not* great-circle arcs, so a cell is never
//! treated as a great-circle quad for containment — vertex-in-cell tests in the
//! descent are decided exactly by HEALPix (a vertex's leaf cell, shifted to the
//! current depth), which stays consistent between a parent and its children.

use crate::geo2mort::{boundaries_scalar, pix2ang_scalar};
use crate::sphere::{dot, latlon_to_unit_vec, Vec3};

/// Bounding cap of a ring-set: a unit `axis` and angular `radius` (radians)
/// such that every ring vertex lies within `radius` of `axis`.
#[derive(Clone, Copy, Debug)]
pub struct Cap {
    pub axis: Vec3,
    pub radius: f64,
}

impl Cap {
    /// Smallest cap over all ring vertices (axis = normalized vertex sum,
    /// radius = max angular distance to a vertex).
    pub fn of_rings(rings: &[Vec<Vec3>]) -> Cap {
        let mut s = [0.0, 0.0, 0.0];
        for ring in rings {
            for v in ring {
                s[0] += v[0];
                s[1] += v[1];
                s[2] += v[2];
            }
        }
        let n = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
        let axis = if n < 1e-12 {
            [0.0, 0.0, 1.0]
        } else {
            [s[0] / n, s[1] / n, s[2] / n]
        };
        let radius = rings
            .iter()
            .flat_map(|r| r.iter())
            .map(|v| dot(&axis, v).clamp(-1.0, 1.0).acos())
            .fold(0.0_f64, f64::max);
        Cap { axis, radius }
    }
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

/// The cell centre as a unit vector (HEALPix centre — strictly interior).
#[inline]
pub fn cell_center_vec(depth: u8, pixel: u64) -> Vec3 {
    let (lon, lat) = pix2ang_scalar(depth, pixel);
    latlon_to_unit_vec(lat, lon)
}

// ── tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo2mort::ang2pix_scalar;
    use crate::sphere::latlon_to_unit_vec;

    #[test]
    fn test_cell_corners_are_unit_vectors() {
        let pix = ang2pix_scalar(6, -120.0, 45.0);
        for c in cell_corners(6, pix) {
            let mag = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
            assert!((mag - 1.0).abs() < 1e-9, "corner not unit length: {mag}");
        }
    }

    #[test]
    fn test_cell_center_near_request() {
        // The HEALPix centre of the cell containing (45, -120) is close to it.
        let pix = ang2pix_scalar(8, -120.0, 45.0);
        let c = cell_center_vec(8, pix);
        let want = latlon_to_unit_vec(45.0, -120.0);
        let ang = dot(&c, &want).clamp(-1.0, 1.0).acos().to_degrees();
        assert!(ang < 1.0, "centre {ang}° from request");
    }

    #[test]
    fn test_cap_of_rings() {
        // A ~10° box around (45, -120): axis near the centre, radius ~ corner dist.
        let ring: Vec<Vec3> = [(40.0, -125.0), (40.0, -115.0), (50.0, -115.0), (50.0, -125.0)]
            .iter()
            .map(|&(la, lo)| latlon_to_unit_vec(la, lo))
            .collect();
        let cap = Cap::of_rings(&[ring]);
        let axis_ll = (
            cap.axis[2].asin().to_degrees(),
            cap.axis[1].atan2(cap.axis[0]).to_degrees(),
        );
        assert!((axis_ll.0 - 45.0).abs() < 2.0 && (axis_ll.1 + 120.0).abs() < 2.0);
        assert!(cap.radius.to_degrees() < 12.0 && cap.radius.to_degrees() > 5.0);
    }
}

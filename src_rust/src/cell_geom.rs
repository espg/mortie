//! HEALPix cell geometry and ring-set classification for the hierarchical
//! coverer (issue #30, Phase B).
//!
//! A cell is classified against a *ring-set* (outer rings + holes, even-odd
//! fill — see [`crate::sphere`]) as one of:
//!
//! * [`Classification::Inside`]  — the whole cell is filled,
//! * [`Classification::Outside`] — the whole cell is empty,
//! * [`Classification::Straddle`] — some ring boundary passes through the cell,
//!   or a whole ring (tiny polygon / hole) sits inside the cell.
//!
//! The descent keeps `Inside`, prunes `Outside`, and refines `Straddle`.
//!
//! IMPORTANT: HEALPix cell edges are *not* great-circle arcs, so a cell cannot
//! be treated as a great-circle quad for containment — doing so disagrees with
//! HEALPix's own `ang2pix` near boundaries, and (fatally for a tree descent)
//! disagrees between a parent and its child.  Vertex-in-cell containment is
//! therefore decided *exactly* by HEALPix: a vertex's leaf cell at the target
//! order is computed once, and its ancestor at any depth is a bit-shift away.
//!
//! NOTE: `dead_code` is allowed until the descent (Phase C) consumes these.
#![allow(dead_code)]

use crate::geo2mort::{boundaries_scalar, pix2ang_scalar};
use crate::sphere::{arcs_cross, latlon_to_unit_vec, parity_filled, PipBackend, Vec3};

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

/// The cell centre as a unit vector (HEALPix centre — strictly interior).
#[inline]
pub fn cell_center_vec(depth: u8, pixel: u64) -> Vec3 {
    let (lon, lat) = pix2ang_scalar(depth, pixel);
    latlon_to_unit_vec(lat, lon)
}

/// Classify a cell against a ring-set.
///
/// `vert_cells[r][k]` is the leaf cell (NESTED pixel at `order`) of ring `r`'s
/// vertex `k`; its ancestor at `depth` is `leaf >> 2*(order-depth)`.  This makes
/// vertex-in-cell containment exact and parent/child-consistent.
///
/// A cell is `Straddle` if any ring vertex lies in it, any ring edge crosses a
/// cell edge, or its corners and centre disagree on fill (a ring boundary clips
/// the cell).  Otherwise it is uniformly `Inside` or `Outside` by the centre's
/// even-odd parity.
pub fn classify(
    depth: u8,
    pixel: u64,
    rings: &[Vec<Vec3>],
    vert_cells: &[Vec<u64>],
    order: u8,
    backend: PipBackend,
) -> Classification {
    // (1) Exact vertex-in-cell: a whole ring (incl. a tiny polygon or hole)
    // sitting inside the cell is caught here, parent/child-consistently.
    let shift = 2 * (order - depth) as u32;
    for rc in vert_cells {
        for &leaf in rc {
            if (leaf >> shift) == pixel {
                return Classification::Straddle;
            }
        }
    }

    let corners = cell_corners(depth, pixel);

    // (2) Any ring edge crossing any cell edge → boundary slices the cell.
    // (Cell edges are great-circle approximations here; the sub-degree error
    // only affects grazing crossings, which the parity check below also guards.)
    // TODO(perf, Phase C): cull rings/edges by a per-subtree bounding cap.
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

    // (3) Corner/centre parity: if any corner's fill differs from the centre's,
    // a ring boundary clips the cell (e.g. shaves a corner with no vertex in it).
    let center = cell_center_vec(depth, pixel);
    let filled = parity_filled(&center, rings, backend);
    for c in &corners {
        if parity_filled(c, rings, backend) != filled {
            return Classification::Straddle;
        }
    }

    if filled {
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

    fn ring(pts: &[(f64, f64)]) -> Vec<Vec3> {
        pts.iter().map(|&(la, lo)| latlon_to_unit_vec(la, lo)).collect()
    }

    /// NESTED pixel at `depth` whose cell contains (lat, lon).
    fn cell_at(depth: u8, lat: f64, lon: f64) -> u64 {
        ang2pix_scalar(depth, lon, lat)
    }

    /// Precompute each ring vertex's leaf cell at `order`.
    fn leaf_cells(rings: &[Vec<Vec3>], order: u8) -> Vec<Vec<u64>> {
        rings
            .iter()
            .map(|ring| {
                ring.iter()
                    .map(|v| {
                        let lon = v[1].atan2(v[0]).to_degrees();
                        let lat = v[2].clamp(-1.0, 1.0).asin().to_degrees();
                        ang2pix_scalar(order, lon, lat)
                    })
                    .collect()
            })
            .collect()
    }

    /// Classify a single cell at `depth`, with target order = depth.
    fn classify_at(depth: u8, pixel: u64, rings: &[Vec<Vec3>]) -> Classification {
        let vc = leaf_cells(rings, depth);
        classify(depth, pixel, rings, &vc, depth, PipBackend::Gnomonic)
    }

    fn outer() -> Vec<Vec3> {
        ring(&[(35.0, -130.0), (35.0, -110.0), (55.0, -110.0), (55.0, -130.0)])
    }
    fn hole() -> Vec<Vec3> {
        ring(&[(42.0, -123.0), (42.0, -117.0), (48.0, -117.0), (48.0, -123.0)])
    }

    #[test]
    fn test_classify_inside() {
        let rings = vec![outer()];
        let pix = cell_at(8, 45.0, -120.0);
        assert_eq!(classify_at(8, pix, &rings), Classification::Inside);
    }

    #[test]
    fn test_classify_outside() {
        let rings = vec![outer()];
        let pix = cell_at(8, 0.0, 0.0);
        assert_eq!(classify_at(8, pix, &rings), Classification::Outside);
    }

    #[test]
    fn test_classify_straddle_edge() {
        let rings = vec![outer()];
        let pix = cell_at(8, 35.0, -130.0); // contains a polygon vertex
        assert_eq!(classify_at(8, pix, &rings), Classification::Straddle);
    }

    #[test]
    fn test_classify_contains_tiny_polygon() {
        // Tiny polygon entirely inside a coarse cell: no edge crosses a cell
        // edge, but the vertices' leaf cells are inside it.
        let depth = 3;
        let pix = cell_at(depth, 45.0, -120.0);
        let (lon, lat) = pix2ang_scalar(depth, pix);
        let tiny = ring(&[
            (lat - 0.05, lon - 0.05),
            (lat - 0.05, lon + 0.05),
            (lat + 0.05, lon + 0.05),
            (lat + 0.05, lon - 0.05),
        ]);
        assert_eq!(classify_at(depth, pix, &vec![tiny]), Classification::Straddle);
    }

    #[test]
    fn test_classify_donut_annulus_inside() {
        let rings = vec![outer(), hole()];
        let pix = cell_at(9, 38.0, -120.0); // inside outer, outside hole
        assert_eq!(classify_at(9, pix, &rings), Classification::Inside);
    }

    #[test]
    fn test_classify_donut_hole_outside() {
        let rings = vec![outer(), hole()];
        let pix = cell_at(9, 45.0, -120.0); // inside the hole
        assert_eq!(classify_at(9, pix, &rings), Classification::Outside);
    }

    #[test]
    fn test_classify_donut_hole_rim_straddle() {
        let rings = vec![outer(), hole()];
        let pix = cell_at(9, 42.0, -123.0); // contains a hole vertex
        assert_eq!(classify_at(9, pix, &rings), Classification::Straddle);
    }

    #[test]
    fn test_classify_consistent_parent_child() {
        // Regression: a point inside a child cell must classify its parent as
        // non-Outside (the great-circle-quad bug pruned correct subtrees).
        let tiny = ring(&[(45.0, -120.0), (45.001, -120.0), (45.0005, -119.999)]);
        let rings = vec![tiny];
        let vc = leaf_cells(&rings, 4);
        let parent = cell_at(3, 45.0, -120.0);
        let child = cell_at(4, 45.0, -120.0);
        assert_eq!(child >> 2, parent, "child must descend from parent");
        assert_ne!(
            classify(3, parent, &rings, &vc, 4, PipBackend::Gnomonic),
            Classification::Outside,
            "parent of a covered child must not be pruned"
        );
    }
}

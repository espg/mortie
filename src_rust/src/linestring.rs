//! Linestring-to-morton coverage: given an open polyline defined by lat/lon
//! vertices, compute the set of morton indices at a given order that traces
//! the line as a contiguous sequence of cells.
//!
//! Algorithm — reuses Phase A of `coverage.rs`:
//!   1. Rasterize each vertex to a HEALPix NESTED cell.
//!   2. Along each segment between consecutive vertices, sample the
//!      great-circle arc at half-cell-resolution spacing and rasterize
//!      each sample. This guarantees the cell sequence along each segment
//!      is contiguous (neighboring cells between consecutive samples).
//!
//! Unlike polygon coverage, the line is *open* — the last vertex is not
//! connected back to the first. Output is sorted/unique across the whole line.

use rayon::prelude::*;

use crate::coverage::{
    cell_resolution_rad, for_each_great_circle_point, great_circle_distance_rad,
};
use crate::geo2mort::ang2pix_scalar;
use crate::morton::nested2mort;

/// Compute sorted, unique morton indices tracing a linestring.
///
/// # Arguments
/// * `lats` — vertex latitudes in degrees
/// * `lons` — vertex longitudes in degrees
/// * `order` — HEALPix depth (1–18)
///
/// # Panics
/// * If `lats` and `lons` have different lengths
/// * If fewer than 2 vertices
/// * If `order` not in 1..=18
pub fn linestring_to_morton_coverage(lats: &[f64], lons: &[f64], order: u8) -> Vec<i64> {
    assert_eq!(
        lats.len(),
        lons.len(),
        "lats and lons must have same length"
    );
    assert!(lats.len() >= 2, "Need at least 2 vertices for a linestring");
    assert!((1..=18).contains(&order), "Order must be 1-18");

    let depth = order;
    let cells = rasterize_linestring(lats, lons, depth);

    let mut result: Vec<i64> = cells.iter().map(|&c| nested2mort(c, depth)).collect();
    result.sort();
    result
}

/// Rasterize an open polyline to a sorted, unique list of NESTED HEALPix cells,
/// interpolating along each segment at half cell-res spacing so the result is
/// contiguous.
///
/// Each segment (`i -> i+1`) is rasterized independently (its start vertex plus
/// the great-circle samples up to, but not including, its end vertex), so the
/// work parallelizes per-segment with rayon; the final vertex is added once.  The
/// per-segment cell lists are concatenated, then deduplicated by `sort_unstable`
/// then `dedup` — no per-cell `HashSet` insert and no throwaway interpolation
/// `Vec`, since samples stream straight into each segment's buffer via
/// [`for_each_great_circle_point`].
pub(crate) fn rasterize_linestring(lats: &[f64], lons: &[f64], depth: u8) -> Vec<u64> {
    let cell_res = cell_resolution_rad(depth);
    let spacing = cell_res * 0.5;

    let n = lats.len();

    // One buffer per segment `i -> i+1`, gathered in parallel.  Each owns its
    // start vertex and interior samples; end vertices are the next segment's
    // start, except the very last, appended below.
    let mut cells: Vec<u64> = (0..n - 1)
        .into_par_iter()
        .flat_map_iter(|i| {
            let (lat1, lon1) = (lats[i], lons[i]);
            let (lat2, lon2) = (lats[i + 1], lons[i + 1]);
            let mut seg: Vec<u64> = vec![ang2pix_scalar(depth, lon1, lat1)];

            let dist = great_circle_distance_rad(lat1, lon1, lat2, lon2);
            let n_seg = (dist / spacing).ceil() as usize;
            if n_seg > 1 {
                for_each_great_circle_point(lat1, lon1, lat2, lon2, n_seg - 1, |la, lo| {
                    seg.push(ang2pix_scalar(depth, lo, la));
                });
            }
            seg
        })
        .collect();

    // The open line's final vertex (no trailing segment to carry it).
    cells.push(ang2pix_scalar(depth, lons[n - 1], lats[n - 1]));

    cells.sort_unstable();
    cells.dedup();
    cells
}

// ── tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo2mort::geo2mort_scalar;
    use healpix::get;
    use std::collections::HashSet;

    #[test]
    fn test_two_vertex_line() {
        let lats = vec![40.0, 50.0];
        let lons = vec![-120.0, -120.0];
        let result = linestring_to_morton_coverage(&lats, &lons, 6);
        assert!(!result.is_empty());
        // Sorted + unique
        for w in result.windows(2) {
            assert!(w[1] > w[0]);
        }
    }

    #[test]
    fn test_endpoints_present() {
        let lats = vec![40.0, 50.0, 45.0];
        let lons = vec![-120.0, -110.0, -100.0];
        let order = 6;
        let result = linestring_to_morton_coverage(&lats, &lons, order);
        let set: HashSet<i64> = result.iter().copied().collect();
        for (la, lo) in lats.iter().zip(lons.iter()) {
            let m = geo2mort_scalar(*la, *lo, order);
            assert!(
                set.contains(&m),
                "Endpoint ({}, {}) -> {} missing from coverage",
                la,
                lo,
                m
            );
        }
    }

    #[test]
    fn test_southern_hemisphere_sign() {
        // An all-southern-hemisphere line should give all-negative mortons
        let lats = vec![-70.0, -80.0, -75.0];
        let lons = vec![30.0, 30.0, 50.0];
        let result = linestring_to_morton_coverage(&lats, &lons, 6);
        assert!(!result.is_empty());
        assert!(
            result.iter().all(|&m| m < 0),
            "Southern hemisphere line should have all negative morton indices"
        );
    }

    #[test]
    fn test_northern_hemisphere_sign() {
        let lats = vec![40.0, 50.0, 45.0];
        let lons = vec![-120.0, -110.0, -100.0];
        let result = linestring_to_morton_coverage(&lats, &lons, 6);
        assert!(!result.is_empty());
        assert!(
            result.iter().all(|&m| m > 0),
            "Northern hemisphere line should have all positive morton indices"
        );
    }

    #[test]
    fn test_segment_contiguity() {
        // Verify that along a single segment each cell in the raster chain
        // has a kth=1 neighbor also in the chain (cells form a connected path)
        let lats = vec![10.0, 20.0];
        let lons = vec![30.0, 40.0];
        let depth: u8 = 5;
        let cells = rasterize_linestring(&lats, &lons, depth);
        assert!(cells.len() >= 2);
        let layer = get(depth);

        // For each cell, at least one kth=1 neighbor (or the cell itself) is
        // in the set -- i.e., no cell is isolated.
        for &c in &cells {
            let mut connected = false;
            for nbr in layer.kth_neighborhood(c, 1) {
                if cells.contains(&nbr) && nbr != c {
                    connected = true;
                    break;
                }
            }
            assert!(connected, "Cell {} is isolated in the raster chain", c);
        }
    }

    #[test]
    fn test_higher_order_more_cells() {
        let lats = vec![10.0, 30.0];
        let lons = vec![40.0, 60.0];
        let r4 = linestring_to_morton_coverage(&lats, &lons, 4);
        let r8 = linestring_to_morton_coverage(&lats, &lons, 8);
        assert!(r8.len() > r4.len());
    }

    #[test]
    #[should_panic(expected = "at least 2 vertices")]
    fn test_too_few_vertices() {
        linestring_to_morton_coverage(&[0.0], &[0.0], 4);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn test_mismatched_lengths() {
        linestring_to_morton_coverage(&[0.0, 1.0], &[0.0], 4);
    }

    #[test]
    #[should_panic(expected = "Order must be")]
    fn test_bad_order() {
        linestring_to_morton_coverage(&[0.0, 1.0], &[0.0, 1.0], 19);
    }

    #[test]
    fn test_repeated_vertex_no_crash() {
        let lats = vec![0.0, 0.0, 1.0];
        let lons = vec![0.0, 0.0, 1.0];
        let result = linestring_to_morton_coverage(&lats, &lons, 5);
        assert!(!result.is_empty());
    }
}

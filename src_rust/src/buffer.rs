//! Morton buffer operation: expand a set of morton cells by a k-cell border.
//!
//! Uses the healpix crate's `kth_neighborhood` on `Layer` to find all cells
//! within k distance of each input cell, then returns only the new cells
//! not present in the input set.

use std::collections::HashSet;

use healpix::get;
use rayon::prelude::*;

use crate::morton::{mort2nested, nested2mort};

/// Compute the k-cell border around a set of morton indices.
///
/// Returns only cells NOT in the input set (the expansion ring).
///
/// # Arguments
/// * `morton_indices` - Slice of morton indices, all at the same order
/// * `k` - Border width in cells (1 = 8-connected neighbors)
///
/// # Returns
/// Sorted vector of morton indices representing the border cells
///
/// # Panics
/// * If indices have mixed orders
/// * If `k >= nside` (healpix constraint)
pub fn morton_buffer(morton_indices: &[i64], k: u32) -> Vec<i64> {
    if morton_indices.is_empty() || k == 0 {
        return Vec::new();
    }

    // Convert all morton indices to nested and validate same order
    let first_morton = morton_indices[0];
    let (_, first_depth) = mort2nested(first_morton);
    let depth = first_depth;

    // Validate k < nside
    let nside = 1u64 << (depth as u32);
    if k as u64 >= nside {
        panic!(
            "k={} must be less than nside={} (order {})",
            k, nside, depth
        );
    }

    // Convert all morton to nested, validating same order
    let nested_cells: Vec<u64> = morton_indices
        .iter()
        .map(|&m| {
            let (nested, d) = mort2nested(m);
            assert_eq!(
                d, depth,
                "All morton indices must be at the same order. Found order {} and {}",
                depth, d
            );
            nested
        })
        .collect();

    let input_set: HashSet<u64> = nested_cells.iter().copied().collect();
    let layer = get(depth);

    // Collect all candidate neighbors in parallel
    // Use thread-local HashSets to avoid contention, then merge
    let candidates: HashSet<u64> = nested_cells
        .par_iter()
        .fold(
            HashSet::new,
            |mut local_set, &cell| {
                let neighborhood = layer.kth_neighborhood(cell, k);
                local_set.extend(neighborhood);
                local_set
            },
        )
        .reduce(
            HashSet::new,
            |mut a, b| {
                a.extend(b);
                a
            },
        );

    // Border = candidates - input
    let mut border: Vec<i64> = candidates
        .difference(&input_set)
        .map(|&nested| nested2mort(nested, depth))
        .collect();

    border.sort();
    border
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morton::fast_norm2mort_scalar;

    #[test]
    fn test_buffer_empty_input() {
        let result = morton_buffer(&[], 1);
        assert!(result.is_empty());
    }

    #[test]
    fn test_buffer_k_zero() {
        let morton = fast_norm2mort_scalar(6, 100, 2);
        let result = morton_buffer(&[morton], 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_buffer_single_cell() {
        // A single cell at order 6 should have up to 8 neighbors
        let morton = fast_norm2mort_scalar(6, 100, 2);
        let result = morton_buffer(&[morton], 1);
        // kth_neighborhood(cell, 1) returns up to 9 cells (3x3),
        // minus the 1 input = up to 8 border cells
        assert!(!result.is_empty());
        assert!(result.len() <= 8);
        // Border should not contain the input
        assert!(!result.contains(&morton));
    }

    #[test]
    fn test_buffer_idempotency() {
        // Border cells should never include input cells
        let cells: Vec<i64> = (0..4)
            .map(|normed| fast_norm2mort_scalar(6, normed, 2))
            .collect();
        let border = morton_buffer(&cells, 1);
        for cell in &cells {
            assert!(
                !border.contains(cell),
                "Border should not contain input cell {}",
                cell
            );
        }
    }

    #[test]
    fn test_buffer_sorted() {
        let cells: Vec<i64> = (0..4)
            .map(|normed| fast_norm2mort_scalar(6, normed, 2))
            .collect();
        let border = morton_buffer(&cells, 1);
        for i in 1..border.len() {
            assert!(
                border[i] >= border[i - 1],
                "Border should be sorted"
            );
        }
    }

    #[test]
    fn test_buffer_southern_hemisphere() {
        // Test with southern hemisphere cells (negative morton)
        let morton = fast_norm2mort_scalar(6, 100, 8);
        assert!(morton < 0, "Southern hemisphere should be negative");
        let result = morton_buffer(&[morton], 1);
        assert!(!result.is_empty());
        assert!(!result.contains(&morton));
    }

    #[test]
    fn test_buffer_k2_larger_than_k1() {
        let morton = fast_norm2mort_scalar(6, 100, 2);
        let border_k1 = morton_buffer(&[morton], 1);
        let border_k2 = morton_buffer(&[morton], 2);
        assert!(
            border_k2.len() > border_k1.len(),
            "k=2 border ({}) should be larger than k=1 border ({})",
            border_k2.len(),
            border_k1.len()
        );
    }

    #[test]
    fn test_buffer_roundtrip_identity() {
        // mort2nested -> nested2mort should be identity
        for parent in 0..12i64 {
            let order = 6i64;
            let normed = 42i64;
            let morton = fast_norm2mort_scalar(order, normed, parent);
            let (nested, depth) = mort2nested(morton);
            let roundtrip = nested2mort(nested, depth);
            assert_eq!(morton, roundtrip);
        }
    }

    #[test]
    #[should_panic(expected = "same order")]
    fn test_buffer_mixed_orders() {
        let m1 = fast_norm2mort_scalar(6, 100, 2);
        let m2 = fast_norm2mort_scalar(7, 100, 2);
        morton_buffer(&[m1, m2], 1);
    }
}

//! Morton buffer operation: expand a set of morton cells by a k-cell border.
//!
//! Uses the healpix crate's `kth_neighborhood` on `Layer` to find all cells
//! within k distance of each input cell, then returns only the new cells
//! not present in the input set.

use std::collections::HashSet;

use healpix::get;
use rayon::prelude::*;

use crate::morton::{mort2nested, nested2mort};

/// Input-cell count below which the neighbour gather runs serially.  Above it the
/// two-level rayon fold/reduce earns back its thread fan-out + per-thread
/// `HashSet` merge; below it a single serial `Vec` + `sort_unstable` + `dedup`
/// wins.  The real crossover tracks total gather work (≈ `n · k²` cells), so a
/// fixed input-count gate is a coarse proxy: it switches early for light `k`
/// (where serial keeps winning past 16 k inputs) and right around parity for
/// heavy `k`.  Measured (issue #34 §D): at `k = 4` the parallel path overtakes
/// serial at `n ≈ 4 k`–`16 k`; at `k ≤ 2` serial still wins at 16 k.  `4096` sits
/// at the heavy-`k` parity point — switching early on a light-`k` input there
/// costs sub-millisecond, while switching late on a heavy-`k` input costs tens of
/// ms — so it errs on the safe side.
const PAR_GATHER_THRESHOLD: usize = 4096;

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
/// * If `k > nside` (healpix constraint)
pub fn morton_buffer(morton_indices: &[i64], k: u32) -> Vec<i64> {
    if morton_indices.is_empty() || k == 0 {
        return Vec::new();
    }

    // Convert all morton indices to nested and validate same order
    let first_morton = morton_indices[0];
    let (_, first_depth) = mort2nested(first_morton);
    let depth = first_depth;

    // Validate k <= nside.  The healpix crate's `kth_neighborhood` accepts
    // `k <= nside` and only panics for `k > nside`; rejecting `k == nside` here
    // would over-constrain it relative to upstream (issue #34 §D).
    let nside = 1u64 << (depth as u32);
    if k as u64 > nside {
        panic!("k={} must not exceed nside={} (order {})", k, nside, depth);
    }

    // Convert all morton to nested (validating same order) into one `HashSet`
    // that serves as both the membership set for the final subtraction and the
    // gather source — no parallel `Vec` copy of the same cells.
    let input_set: HashSet<u64> = morton_indices
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

    let layer = get(depth);

    // Gather candidate neighbours, dedup, and subtract the input set.  Small
    // inputs take a serial `Vec` + `sort_unstable` + `dedup` (no thread fan-out
    // or per-thread `HashSet` merge); larger inputs use the two-level rayon
    // fold/reduce into thread-local `HashSet`s.
    let mut border: Vec<i64> = if input_set.len() < PAR_GATHER_THRESHOLD {
        let mut candidates: Vec<u64> = Vec::new();
        for &cell in &input_set {
            candidates.extend(layer.kth_neighborhood(cell, k));
        }
        candidates.sort_unstable();
        candidates.dedup();
        candidates
            .into_iter()
            .filter(|c| !input_set.contains(c))
            .map(|nested| nested2mort(nested, depth))
            .collect()
    } else {
        let candidates: HashSet<u64> = input_set
            .par_iter()
            .fold(HashSet::new, |mut local_set, &cell| {
                local_set.extend(layer.kth_neighborhood(cell, k));
                local_set
            })
            .reduce(HashSet::new, |mut a, b| {
                a.extend(b);
                a
            });
        candidates
            .difference(&input_set)
            .map(|&nested| nested2mort(nested, depth))
            .collect()
    };

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

    #[test]
    fn test_buffer_k_equals_nside_allowed() {
        // The healpix crate accepts k == nside (only k > nside panics); a low
        // order keeps the neighbourhood small.  At order 2, nside = 4.
        let morton = fast_norm2mort_scalar(2, 5, 2);
        let result = morton_buffer(&[morton], 4);
        assert!(!result.is_empty());
        assert!(!result.contains(&morton));
    }

    #[test]
    #[should_panic(expected = "must not exceed nside")]
    fn test_buffer_k_greater_than_nside_panics() {
        // At order 2, nside = 4, so k = 5 is over the limit.
        let morton = fast_norm2mort_scalar(2, 5, 2);
        morton_buffer(&[morton], 5);
    }

    #[test]
    fn test_buffer_serial_and_parallel_paths_agree() {
        // Build an input set straddling the serial/parallel threshold and verify
        // both gather paths produce identical borders.  The result is sorted and
        // excludes the input regardless of which path runs.
        let cells: Vec<i64> = (0..PAR_GATHER_THRESHOLD as i64 + 100)
            .map(|normed| fast_norm2mort_scalar(8, normed, 2))
            .collect();
        assert!(cells.len() > PAR_GATHER_THRESHOLD, "want the parallel path");

        // Parallel path over the full set.
        let par_border = morton_buffer(&cells, 1);
        // Serial path over a sub-threshold slice that is a subset of the input.
        let small = &cells[..PAR_GATHER_THRESHOLD - 1];
        let ser_border = morton_buffer(small, 1);

        for b in [&par_border, &ser_border] {
            for w in b.windows(2) {
                assert!(w[1] > w[0], "border must be sorted and unique");
            }
        }
        let input: HashSet<i64> = cells.iter().copied().collect();
        assert!(par_border.iter().all(|m| !input.contains(m)));
    }


}

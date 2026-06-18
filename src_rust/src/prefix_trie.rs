//! Morton bounding box: build a compacted prefix trie over morton indices
//! and return a flat list of nodes plus one shared permutation buffer.
//!
//! A morton index is a decimal self-describing address: the first digit is
//! the base cell and every later digit is in `1..=4` (a base-4 path step).
//! The trie therefore branches on decimal-digit columns of the (left-padded)
//! decimal rendering, but we never materialise strings — each value is read
//! as a fixed-width array of integer *slots* via integer arithmetic:
//!
//! * `SLOT_SPACE` (`-2`) — left-pad position (a shorter value's high columns)
//! * `SLOT_MINUS` (`-1`) — the sign column of a negative value
//! * `0..=9`            — a decimal digit
//!
//! Column 0 is always the sign/pad column and column 1 the first digit/pad,
//! mirroring the historical `rjust`-padded char grid exactly so the
//! `split_children` Rust-vs-Python parity contract holds bit-for-bit.
//!
//! Membership is stored once as a single `permutation` buffer; each node
//! carries an `(idx_start, idx_len)` slice into it instead of cloning its
//! index list. The Python side reconstructs `MortonChild` objects from this
//! flat output and slices the shared buffer with numpy.

use rayon::prelude::*;

/// Slot sentinel for a left-pad (space) column.
const SLOT_SPACE: i8 = -2;
/// Slot sentinel for the `'-'` sign column.
const SLOT_MINUS: i8 = -1;

/// A flat trie node returned to Python.
///
/// Fields: `(characteristic, count, idx_start, idx_len, child_node_ids, depth)`
/// where `(idx_start, idx_len)` is a slice into the shared permutation buffer.
pub type FlatNode = (String, usize, usize, usize, Vec<usize>, usize);

/// Render a slot value back to its decimal-string character.
#[inline]
fn slot_char(slot: i8) -> char {
    match slot {
        SLOT_SPACE => ' ',
        SLOT_MINUS => '-',
        d => (b'0' + d as u8) as char,
    }
}

/// Build the fixed-width slot grid (row-major) matching `str(v).rjust(width)`.
///
/// Returns `(grid, ncols)`. Row `i` holds the right-justified slots for
/// `morton_array[i]`; high columns beyond the value's length are `SLOT_SPACE`.
fn build_slot_grid(morton_array: &[i64]) -> (Vec<i8>, usize) {
    let n = morton_array.len();

    // String length of each value: decimal digits (+1 for a leading '-').
    let str_lens: Vec<usize> = morton_array
        .par_iter()
        .map(|&v| {
            let digits = decimal_digits(v.unsigned_abs());
            digits + if v < 0 { 1 } else { 0 }
        })
        .collect();

    let max_len = str_lens.iter().copied().max().unwrap();
    let has_negatives = morton_array.iter().any(|&v| v < 0);
    // Ensure column 0 is always a sign/pad column even when all positive.
    let ncols = if has_negatives { max_len } else { max_len + 1 };

    let mut grid = vec![SLOT_SPACE; n * ncols];
    grid.par_chunks_mut(ncols)
        .zip(morton_array.par_iter())
        .for_each(|(row, &v)| {
            let abs = v.unsigned_abs();
            let ndig = decimal_digits(abs);
            let slen = ndig + if v < 0 { 1 } else { 0 };
            let start = ncols - slen;
            let mut col = start;
            if v < 0 {
                row[col] = SLOT_MINUS;
                col += 1;
            }
            // Write decimal digits most-significant first.
            let mut divisor = 10u64.pow((ndig - 1) as u32);
            let mut rem = abs;
            for _ in 0..ndig {
                row[col] = (rem / divisor) as i8;
                rem %= divisor;
                divisor /= 10;
                col += 1;
            }
        });

    (grid, ncols)
}

/// Count decimal digits in `val` (a `0` value has one digit).
#[inline]
fn decimal_digits(val: u64) -> usize {
    if val == 0 {
        return 1;
    }
    (val.ilog10() + 1) as usize
}

/// Build a compacted prefix trie and return all nodes plus the permutation.
///
/// # Arguments
/// * `morton_array` – signed 64-bit morton indices
/// * `max_depth`    – optional branching depth limit (None = unlimited)
///
/// # Returns
/// `(Vec<FlatNode>, Vec<usize>)`. Each node references its membership as an
/// `(idx_start, idx_len)` slice of the returned permutation buffer (the
/// original positions in `morton_array`). Nodes are emitted depth-first;
/// roots come first.
pub fn split_children_flat(
    morton_array: &[i64],
    max_depth: Option<usize>,
) -> (Vec<FlatNode>, Vec<usize>) {
    let n = morton_array.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let (grid, ncols) = build_slot_grid(morton_array);
    let slot = |i: usize, col: usize| grid[i * ncols + col];

    // Single permutation buffer; nodes own contiguous slices of it. We sort a
    // working copy column-by-column so each group is contiguous, mirroring the
    // recursive partition order of the historical algorithm.
    let mut perm: Vec<usize> = (0..n).collect();
    let mut nodes: Vec<FlatNode> = Vec::new();

    // Root grouping: by (sign column, first-digit column), in slot order.
    // Sort by (col0, col1) so each root group is a contiguous run.
    perm.sort_by(|&a, &b| (slot(a, 0), slot(a, 1)).cmp(&(slot(b, 0), slot(b, 1))));

    let mut i = 0usize;
    while i < n {
        let s0 = slot(perm[i], 0);
        let s1 = slot(perm[i], 1);
        let mut j = i + 1;
        while j < n && slot(perm[j], 0) == s0 && slot(perm[j], 1) == s1 {
            j += 1;
        }

        // Root characteristic: '-' + digit for negatives, else just the digit.
        let mut characteristic = String::new();
        if s0 == SLOT_MINUS {
            characteristic.push('-');
        }
        characteristic.push(slot_char(s1));

        compact_group(
            &grid,
            ncols,
            &mut perm,
            i,
            j,
            2, // start scanning at column 2
            characteristic,
            0, // depth
            max_depth,
            &mut nodes,
        );

        i = j;
    }

    (nodes, perm)
}

/// Recursively compact the permutation slice `perm[lo..hi]` and append nodes.
///
/// Returns the node id (index into `out`) of the node created for this group.
#[allow(clippy::too_many_arguments)]
fn compact_group(
    grid: &[i8],
    ncols: usize,
    perm: &mut [usize],
    lo: usize,
    hi: usize,
    start_col: usize,
    mut characteristic: String,
    depth: usize,
    max_depth: Option<usize>,
    out: &mut Vec<FlatNode>,
) -> usize {
    let slot = |i: usize, col: usize| grid[i * ncols + col];
    let mut col = start_col;

    while col < ncols {
        let first = slot(perm[lo], col);
        let all_same = perm[lo..hi].iter().all(|&i| slot(i, col) == first);

        if all_same {
            characteristic.push(slot_char(first));
            col += 1;
        } else {
            // Divergence — branch if depth allows.
            if max_depth.is_some() && depth >= max_depth.unwrap() {
                break;
            }

            // Reserve this node's slot; children fill its child_ids later.
            let node_id = out.len();
            out.push((
                characteristic.clone(),
                hi - lo,
                lo,
                hi - lo,
                Vec::new(),
                depth,
            ));

            // Sort the slice by this column so each child group is contiguous,
            // preserving relative order within a child (stable) so deeper
            // columns stay grouped.
            perm[lo..hi].sort_by_key(|&i| slot(i, col));

            let mut child_ids = Vec::new();
            let mut g = lo;
            while g < hi {
                let cv = slot(perm[g], col);
                let mut k = g + 1;
                while k < hi && slot(perm[k], col) == cv {
                    k += 1;
                }
                let mut child_char = characteristic.clone();
                child_char.push(slot_char(cv));
                let cid = compact_group(
                    grid,
                    ncols,
                    perm,
                    g,
                    k,
                    col + 1,
                    child_char,
                    depth + 1,
                    max_depth,
                    out,
                );
                child_ids.push(cid);
                g = k;
            }

            out[node_id].4 = child_ids;
            return node_id;
        }
    }

    // Leaf node (uniform through all columns, or hit max_depth).
    let node_id = out.len();
    out.push((characteristic, hi - lo, lo, hi - lo, Vec::new(), depth));
    node_id
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Collect (characteristic, count) for nodes at a given depth.
    fn at_depth(nodes: &[FlatNode], depth: usize) -> Vec<(String, usize)> {
        nodes
            .iter()
            .filter(|n| n.5 == depth)
            .map(|n| (n.0.clone(), n.1))
            .collect()
    }

    #[test]
    fn test_identical_indices() {
        let arr = vec![1234i64, 1234, 1234];
        let (nodes, perm) = split_children_flat(&arr, None);
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].0, "1234"); // characteristic
        assert_eq!(nodes[0].1, 3); // count
        assert!(nodes[0].4.is_empty()); // no children
        assert_eq!(nodes[0].3, 3); // idx_len
        assert_eq!(perm.len(), 3);
    }

    #[test]
    fn test_divergence() {
        let arr = vec![1231i64, 1232, 1233];
        let (nodes, _) = split_children_flat(&arr, None);
        assert_eq!(nodes[0].0, "123");
        assert_eq!(nodes[0].4.len(), 3);
    }

    #[test]
    fn test_negative_indices() {
        let arr = vec![-123i64, -124, -234];
        let (nodes, _) = split_children_flat(&arr, None);
        let roots = at_depth(&nodes, 0);
        assert!(roots.iter().any(|(c, _)| c.starts_with("-1")));
        assert!(roots.iter().any(|(c, _)| c.starts_with("-2")));
    }

    #[test]
    fn test_mixed_sign() {
        let arr = vec![-111i64, -112, 111, 112];
        let (nodes, _) = split_children_flat(&arr, None);
        let roots = at_depth(&nodes, 0);
        assert!(roots.iter().any(|(c, _)| c.starts_with('-')));
        assert!(roots.iter().any(|(c, _)| !c.starts_with('-')));
    }

    #[test]
    fn test_max_depth_zero() {
        let arr = vec![1111i64, 1122, 1211, 1222];
        let (nodes, _) = split_children_flat(&arr, Some(0));
        for n in &nodes {
            if n.5 == 0 {
                assert!(n.4.is_empty());
            }
        }
    }

    #[test]
    fn test_coverage() {
        let arr = vec![-5112i64, -5121, -6131, -6132, -6133];
        let (nodes, perm) = split_children_flat(&arr, None);
        let root_count: usize = at_depth(&nodes, 0).iter().map(|(_, c)| c).sum();
        assert_eq!(root_count, arr.len());
        assert_eq!(perm.len(), arr.len());
    }

    #[test]
    fn test_empty() {
        let arr: Vec<i64> = vec![];
        let (nodes, perm) = split_children_flat(&arr, None);
        assert!(nodes.is_empty());
        assert!(perm.is_empty());
    }

    #[test]
    fn test_mixed_order() {
        // Shorter and longer values left-pad differently (space-aligned).
        let arr = vec![123i64, 1231, 1232];
        let (nodes, _) = split_children_flat(&arr, None);
        // The order-2 value 123 splits off from the order-3 values via the
        // pad/digit column, so there must be more than one root-or-leaf node.
        assert!(nodes.len() >= 2);
        // Coverage is preserved.
        let total: usize = at_depth(&nodes, 0).iter().map(|(_, c)| c).sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn test_slices_partition_permutation() {
        // Every leaf's (start,len) slice maps to its original morton values,
        // and the leaf slices tile the whole permutation buffer.
        let arr = vec![1211i64, 1211, 1222, 1233, 1233, 1233];
        let (nodes, perm) = split_children_flat(&arr, None);
        let mut covered = vec![false; arr.len()];
        for n in &nodes {
            if n.4.is_empty() {
                // leaf
                for &p in &perm[n.2..n.2 + n.3] {
                    covered[p] = true;
                }
            }
        }
        assert!(covered.iter().all(|&c| c));
    }

    #[test]
    fn test_decimal_digits() {
        assert_eq!(decimal_digits(0), 1);
        assert_eq!(decimal_digits(9), 1);
        assert_eq!(decimal_digits(10), 2);
        assert_eq!(decimal_digits(999), 3);
        assert_eq!(decimal_digits(1000), 4);
    }
}

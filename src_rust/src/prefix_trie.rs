//! Morton bounding box: build a compacted prefix trie over morton indices
//! and return a flat list of nodes plus one shared permutation buffer.
//!
//! Each morton word carries a self-describing address as its **decimal repr**
//! ([`crate::decimal_morton::to_decimal_repr`]): the first digit is the base
//! cell (`base+1` north / `base-5` south, with a leading `-` for the southern
//! hemisphere) and every later digit is in `1..=4` (one base-4 path step per
//! HEALPix order). After the issue #48 packed-`u64` flip the bare-`i64` word is
//! no longer the decimal value itself, so the trie branches on the columns of
//! the *decoded repr string* rather than on raw decimal digits of the `i64`.
//! The repr is the order-0..=29 generalization of the legacy decimal form, so
//! the column structure (sign column, then one digit per order) — and therefore
//! the characteristic strings and the `MortonChild.cell_area` digit-count → order
//! contract — is unchanged.
//!
//! Each value is read as a fixed-width array of integer *slots*:
//!
//! * `SLOT_SPACE` (`-2`) — left-pad position (a shorter value's high columns)
//! * `SLOT_MINUS` (`-1`) — the sign column of a southern value
//! * `0..=9`            — a decimal digit of the repr
//!
//! Column 0 is always the sign/pad column and column 1 the first digit/pad, so
//! a shorter (coarser) word left-pads against the deepest one in the set.
//!
//! Membership is stored once as a single `permutation` buffer; each node
//! carries an `(idx_start, idx_len)` slice into it instead of cloning its
//! index list. The Python side reconstructs `MortonChild` objects from this
//! flat output and slices the shared buffer with numpy.

use rayon::prelude::*;

use crate::decimal_morton::to_decimal_repr;

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

/// Decode each packed word to its decimal-repr string (issue #48).
///
/// A panic-free decode: the empty sentinel / an invalid prefix has no repr, so
/// it maps to an empty string and lands in the all-pad row (it shares no prefix
/// with any real cell and falls out as its own degenerate group).
fn reprs(morton_array: &[i64]) -> Vec<String> {
    morton_array
        .par_iter()
        .map(|&v| to_decimal_repr(v as u64).unwrap_or_default())
        .collect()
}

/// Build the fixed-width slot grid (row-major) over the words' decimal reprs.
///
/// Returns `(grid, ncols)`. Row `i` holds the right-justified slots for the repr
/// of `morton_array[i]`; high columns beyond the repr's length are `SLOT_SPACE`.
fn build_slot_grid(morton_array: &[i64]) -> (Vec<i8>, usize) {
    let n = morton_array.len();
    let reprs = reprs(morton_array);

    let max_len = reprs.iter().map(|s| s.len()).max().unwrap();
    let has_negatives = reprs.iter().any(|s| s.starts_with('-'));
    // Ensure column 0 is always a sign/pad column even when all positive.
    let ncols = if has_negatives { max_len } else { max_len + 1 };

    let mut grid = vec![SLOT_SPACE; n * ncols];
    grid.par_chunks_mut(ncols)
        .zip(reprs.par_iter())
        .for_each(|(row, s)| {
            let start = ncols - s.len();
            for (off, ch) in s.bytes().enumerate() {
                row[start + off] = match ch {
                    b'-' => SLOT_MINUS,
                    d => (d - b'0') as i8,
                };
            }
        });

    (grid, ncols)
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
    use crate::decimal_morton::encode;

    /// Build a packed word (bit-reinterpreted to `i64`) whose decimal repr is the
    /// legacy-style digit string `digits` (e.g. "1234" -> base 0 north, tuples
    /// [0,1,2]; "-5112" -> base 8 south, tuples [0,0,1]). Each digit after the
    /// leading base digit is `1..=4`, stored as `digit-1`.
    fn word(digits: &str) -> i64 {
        let southern = digits.starts_with('-');
        let body = digits.trim_start_matches('-');
        let lead = body.as_bytes()[0] - b'0'; // base+1 (north) or base-5 (south)
        let base = if southern { lead + 5 } else { lead - 1 };
        let tuples: Vec<u8> = body.bytes().skip(1).map(|c| c - b'0' - 1).collect();
        let order = tuples.len() as u8;
        encode(base, &tuples, order) as i64
    }

    /// Map a digit-string list to packed words.
    fn words(digits: &[&str]) -> Vec<i64> {
        digits.iter().map(|d| word(d)).collect()
    }

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
        let arr = words(&["1234", "1234", "1234"]);
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
        let arr = words(&["1231", "1232", "1233"]);
        let (nodes, _) = split_children_flat(&arr, None);
        assert_eq!(nodes[0].0, "123");
        assert_eq!(nodes[0].4.len(), 3);
    }

    #[test]
    fn test_negative_indices() {
        let arr = words(&["-123", "-124", "-234"]);
        let (nodes, _) = split_children_flat(&arr, None);
        let roots = at_depth(&nodes, 0);
        assert!(roots.iter().any(|(c, _)| c.starts_with("-1")));
        assert!(roots.iter().any(|(c, _)| c.starts_with("-2")));
    }

    #[test]
    fn test_mixed_sign() {
        let arr = words(&["-111", "-112", "111", "112"]);
        let (nodes, _) = split_children_flat(&arr, None);
        let roots = at_depth(&nodes, 0);
        assert!(roots.iter().any(|(c, _)| c.starts_with('-')));
        assert!(roots.iter().any(|(c, _)| !c.starts_with('-')));
    }

    #[test]
    fn test_max_depth_zero() {
        let arr = words(&["1111", "1122", "1211", "1222"]);
        let (nodes, _) = split_children_flat(&arr, Some(0));
        for n in &nodes {
            if n.5 == 0 {
                assert!(n.4.is_empty());
            }
        }
    }

    #[test]
    fn test_coverage() {
        let arr = words(&["-5112", "-5121", "-6131", "-6132", "-6133"]);
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
        // Shorter and longer reprs left-pad differently (space-aligned).
        let arr = words(&["123", "1231", "1232"]);
        let (nodes, _) = split_children_flat(&arr, None);
        // The order-2 value "123" splits off from the order-3 values via the
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
        let arr = words(&["1211", "1211", "1222", "1233", "1233", "1233"]);
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
    fn test_repr_columns_match_digit_string() {
        // The slot grid is built over the decode-through-kernel decimal repr, so
        // a word's repr characters reappear as the trie characteristic.
        let arr = words(&["1234"]);
        let (nodes, _) = split_children_flat(&arr, None);
        assert_eq!(nodes[0].0, "1234");
        let arr = words(&["-5112"]);
        let (nodes, _) = split_children_flat(&arr, None);
        assert_eq!(nodes[0].0, "-5112");
    }
}

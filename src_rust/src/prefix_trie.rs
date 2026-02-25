//! Morton bounding box: build a compacted prefix trie over morton index
//! strings and return a flat list of nodes.
//!
//! The Python side reconstructs `MortonChild` objects from this flat output.

use rayon::prelude::*;
use std::collections::BTreeMap;

/// A flat trie node returned to Python.
///
/// Fields: (characteristic, count, original_indices, child_node_ids, depth)
pub type FlatNode = (String, usize, Vec<usize>, Vec<usize>, usize);

/// Build a compacted prefix trie and return all nodes as a flat `Vec`.
///
/// # Arguments
/// * `morton_array` – signed 64-bit morton indices
/// * `max_depth`    – optional branching depth limit (None = unlimited)
///
/// # Returns
/// `Vec<FlatNode>` where each tuple is
/// `(characteristic, count, original_indices, child_node_ids, depth)`.
///
/// Nodes are emitted depth-first; roots come first.
pub fn split_children_flat(
    morton_array: &[i64],
    max_depth: Option<usize>,
) -> Vec<FlatNode> {
    let n = morton_array.len();
    if n == 0 {
        return Vec::new();
    }

    // 1. Convert i64 → padded strings (parallel with rayon)
    let strings: Vec<String> = morton_array
        .par_iter()
        .map(|v| v.to_string())
        .collect();

    // Determine max string length; ensure column 0 is sign/pad
    let max_len = strings.iter().map(|s| s.len()).max().unwrap();
    let has_negatives = strings.iter().any(|s| s.starts_with('-'));
    let padded_len = if has_negatives { max_len } else { max_len + 1 };

    // Right-pad each string with spaces, then build a flat byte grid
    let padded: Vec<Vec<u8>> = strings
        .par_iter()
        .map(|s| {
            let mut v = Vec::with_capacity(padded_len);
            let bytes = s.as_bytes();
            // left-pad with spaces
            for _ in 0..(padded_len - bytes.len()) {
                v.push(b' ');
            }
            v.extend_from_slice(bytes);
            v
        })
        .collect();

    // 2. Group by (sign_col, first_digit) → root groups
    //    sign_col = padded[i][0], digit_col = padded[i][1]
    let mut root_groups: BTreeMap<(u8, u8), Vec<usize>> = BTreeMap::new();
    for i in 0..n {
        let sign = padded[i][0];
        let digit = padded[i][1];
        root_groups.entry((sign, digit)).or_default().push(i);
    }

    // 3. Build trie iteratively, collecting nodes into flat vec
    let mut nodes: Vec<FlatNode> = Vec::new();

    for ((sign, digit), indices) in &root_groups {
        let characteristic = if *sign == b'-' {
            format!("-{}", *digit as char)
        } else {
            format!("{}", *digit as char)
        };

        // Recursively compact this group starting at column 2
        compact_group(
            &padded,
            indices,
            2, // start scanning at column 2
            padded_len,
            characteristic,
            0, // depth
            max_depth,
            &mut nodes,
        );
    }

    nodes
}

/// Recursively compact a group of indices and append nodes to `out`.
///
/// Returns the node id (index into `out`) of the node created for this group.
fn compact_group(
    grid: &[Vec<u8>],
    indices: &[usize],
    start_col: usize,
    ncols: usize,
    mut characteristic: String,
    depth: usize,
    max_depth: Option<usize>,
    out: &mut Vec<FlatNode>,
) -> usize {
    let mut col = start_col;

    // Extend characteristic while the column is uniform
    while col < ncols {
        let first_byte = grid[indices[0]][col];
        let all_same = indices.iter().all(|&i| grid[i][col] == first_byte);

        if all_same {
            characteristic.push(first_byte as char);
            col += 1;
        } else {
            // Divergence — branch if depth allows
            if max_depth.is_some() && depth >= max_depth.unwrap() {
                break;
            }

            // Reserve a slot for this node, fill children later
            let node_id = out.len();
            out.push((
                characteristic.clone(),
                indices.len(),
                indices.to_vec(),
                Vec::new(), // child_node_ids placeholder
                depth,
            ));

            // Group by the diverging byte
            let mut child_groups: BTreeMap<u8, Vec<usize>> = BTreeMap::new();
            for &i in indices {
                child_groups.entry(grid[i][col]).or_default().push(i);
            }

            let mut child_ids = Vec::with_capacity(child_groups.len());
            for (byte_val, child_indices) in &child_groups {
                let child_char = format!("{}{}", characteristic, *byte_val as char);
                let cid = compact_group(
                    grid,
                    child_indices,
                    col + 1,
                    ncols,
                    child_char,
                    depth + 1,
                    max_depth,
                    out,
                );
                child_ids.push(cid);
            }

            // Patch in the child_node_ids
            out[node_id].3 = child_ids;
            return node_id;
        }
    }

    // Leaf node (no divergence within columns, or hit max_depth)
    let node_id = out.len();
    out.push((
        characteristic,
        indices.len(),
        indices.to_vec(),
        Vec::new(),
        depth,
    ));
    node_id
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_indices() {
        let arr = vec![1234i64, 1234, 1234];
        let nodes = split_children_flat(&arr, None);
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].0, "1234"); // characteristic
        assert_eq!(nodes[0].1, 3);       // count
        assert!(nodes[0].3.is_empty());  // no children
    }

    #[test]
    fn test_divergence() {
        let arr = vec![1231i64, 1232, 1233];
        let nodes = split_children_flat(&arr, None);
        // Root "123" with 3 children
        assert_eq!(nodes[0].0, "123");
        assert_eq!(nodes[0].3.len(), 3);
    }

    #[test]
    fn test_negative_indices() {
        let arr = vec![-123i64, -124, -234];
        let nodes = split_children_flat(&arr, None);
        // Should have roots for -1 and -2 groups
        let root_chars: Vec<&str> = nodes
            .iter()
            .filter(|n| n.4 == 0) // depth == 0
            .map(|n| n.0.as_str())
            .collect();
        assert!(root_chars.iter().any(|c| c.starts_with("-1")));
        assert!(root_chars.iter().any(|c| c.starts_with("-2")));
    }

    #[test]
    fn test_mixed_sign() {
        let arr = vec![-111i64, -112, 111, 112];
        let nodes = split_children_flat(&arr, None);
        let root_chars: Vec<&str> = nodes
            .iter()
            .filter(|n| n.4 == 0)
            .map(|n| n.0.as_str())
            .collect();
        assert!(root_chars.iter().any(|c| c.starts_with("-")));
        assert!(root_chars.iter().any(|c| !c.starts_with("-")));
    }

    #[test]
    fn test_max_depth_zero() {
        let arr = vec![1111i64, 1122, 1211, 1222];
        let nodes = split_children_flat(&arr, Some(0));
        // All root nodes should have no children
        for n in &nodes {
            if n.4 == 0 {
                assert!(n.3.is_empty());
            }
        }
    }

    #[test]
    fn test_coverage() {
        let arr = vec![-5112i64, -5121, -6131, -6132, -6133];
        let nodes = split_children_flat(&arr, None);
        // Total count across root nodes should equal input length
        let root_count: usize = nodes
            .iter()
            .filter(|n| n.4 == 0)
            .map(|n| n.1)
            .sum();
        assert_eq!(root_count, arr.len());
    }

    #[test]
    fn test_empty() {
        let arr: Vec<i64> = vec![];
        let nodes = split_children_flat(&arr, None);
        assert!(nodes.is_empty());
    }
}

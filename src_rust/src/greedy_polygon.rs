//! Greedy polygon subdivision for morton indices
//!
//! This module provides a greedy algorithm to generate compact morton index
//! coverage for arbitrary geometries with configurable constraints.

use std::collections::HashSet;

/// A box in the subdivision tree
#[derive(Clone)]
struct MortonBox {
    prefix: String,
    indices: Vec<String>,
}

/// Estimate order from a morton prefix string
fn estimate_order_from_prefix(prefix: &str) -> usize {
    // Remove minus sign and count digits
    let digits = prefix.trim_start_matches('-');
    if digits.is_empty() {
        return 0;
    }
    digits.len() - 1
}

/// Find longest common prefix among strings
fn find_common_prefix(strings: &[String]) -> String {
    if strings.is_empty() {
        return String::new();
    }

    if strings.len() == 1 {
        return strings[0].clone();
    }

    let first = &strings[0];
    let mut prefix_len = 0;

    for (i, ch) in first.chars().enumerate() {
        // Check if all strings have same character at position i
        let all_match = strings[1..].iter().all(|s| {
            s.chars().nth(i).map_or(false, |c| c == ch)
        });

        if all_match {
            prefix_len = i + 1;
        } else {
            break;
        }
    }

    first.chars().take(prefix_len).collect()
}

/// Try to split a box into sub-boxes
fn try_split(prefix: &str, indices: &[String]) -> Option<Vec<MortonBox>> {
    // Check if all indices are identical
    let unique: HashSet<&String> = indices.iter().collect();
    if unique.len() == 1 {
        return None;
    }

    let divergence_pos = prefix.len();

    // Check if we've exhausted string length
    let min_len = indices.iter().map(|s| s.len()).min().unwrap_or(0);
    if divergence_pos >= min_len {
        return None;
    }

    // Get unique characters at divergence position
    let mut unique_chars: Vec<char> = indices
        .iter()
        .filter_map(|s| s.chars().nth(divergence_pos))
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    if unique_chars.len() <= 1 {
        return None;
    }

    unique_chars.sort();

    // Create sub-boxes
    let mut sub_boxes = Vec::new();

    for ch in unique_chars {
        let group_indices: Vec<String> = indices
            .iter()
            .filter(|s| s.chars().nth(divergence_pos) == Some(ch))
            .cloned()
            .collect();

        if !group_indices.is_empty() {
            let group_prefix = find_common_prefix(&group_indices);
            sub_boxes.push(MortonBox {
                prefix: group_prefix,
                indices: group_indices,
            });
        }
    }

    if sub_boxes.len() > 1 {
        Some(sub_boxes)
    } else {
        None
    }
}

/// Greedy subdivision of morton indices
///
/// # Arguments
/// * `morton_strings` - Morton indices as strings
/// * `max_boxes` - Maximum number of boxes allowed
/// * `ordermax` - Optional maximum order for subdivision
///
/// # Returns
/// Vector of morton index prefix strings
pub fn greedy_subdivide(
    morton_strings: Vec<String>,
    max_boxes: usize,
    ordermax: Option<usize>,
) -> Vec<String> {
    // Start with one box containing all indices
    let mut boxes = vec![MortonBox {
        prefix: find_common_prefix(&morton_strings),
        indices: morton_strings,
    }];

    while boxes.len() < max_boxes {

        // Find which box can be split
        let mut best_box_idx: Option<usize> = None;
        let mut best_split: Option<Vec<MortonBox>> = None;
        let mut max_split_count = 0;
        let mut best_order = usize::MAX;

        for (i, box_item) in boxes.iter().enumerate() {
            // Check if this box has reached ordermax
            if let Some(om) = ordermax {
                let box_order = estimate_order_from_prefix(&box_item.prefix);
                if box_order >= om {
                    continue;
                }
            }

            // Try to split this box
            if let Some(split_result) = try_split(&box_item.prefix, &box_item.indices) {
                let n_splits = split_result.len();
                let new_total = boxes.len() - 1 + n_splits;

                if new_total <= max_boxes && n_splits > 1 {
                    // Priority: lower order first, then more splits
                    let box_order = estimate_order_from_prefix(&box_item.prefix);

                    if best_box_idx.is_none()
                        || box_order < best_order
                        || (box_order == best_order && n_splits > max_split_count)
                    {
                        best_box_idx = Some(i);
                        best_split = Some(split_result);
                        max_split_count = n_splits;
                        best_order = box_order;
                    }
                }
            }
        }

        // If no splits possible, break
        let Some(idx) = best_box_idx else {
            break;
        };

        // Apply the best split
        let mut new_split = best_split.unwrap();

        // Clip children to ordermax if necessary
        if let Some(om) = ordermax {
            for child in &mut new_split {
                let child_order = estimate_order_from_prefix(&child.prefix);
                if child_order > om {
                    // Clip prefix to ordermax
                    // ordermax + 2 for minus sign and parent digit
                    let target_len = om + 2;
                    if child.prefix.len() > target_len {
                        child.prefix.truncate(target_len);
                    }
                }
            }
        }

        // Remove old box and add new sub-boxes
        boxes.remove(idx);
        boxes.extend(new_split);
    }

    // Extract prefixes
    boxes.iter().map(|b| b.prefix.clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_common_prefix_empty() {
        let strings = vec![];
        assert_eq!(find_common_prefix(&strings), "");
    }

    #[test]
    fn test_find_common_prefix_single() {
        let strings = vec!["-511".to_string()];
        assert_eq!(find_common_prefix(&strings), "-511");
    }

    #[test]
    fn test_find_common_prefix_multiple() {
        let strings = vec!["-511111".to_string(), "-511222".to_string(), "-511333".to_string()];
        assert_eq!(find_common_prefix(&strings), "-511");
    }

    #[test]
    fn test_estimate_order() {
        assert_eq!(estimate_order_from_prefix("-6"), 0);
        assert_eq!(estimate_order_from_prefix("-61"), 1);
        assert_eq!(estimate_order_from_prefix("-611"), 2);
        assert_eq!(estimate_order_from_prefix("-6111"), 3);
    }

    #[test]
    fn test_greedy_subdivide_basic() {
        let morton_strings = vec![
            "-311111".to_string(),
            "-311222".to_string(),
            "-411111".to_string(),
            "-411222".to_string(),
        ];

        let result = greedy_subdivide(morton_strings, 10, None);
        assert!(result.len() >= 2);
        assert!(result.len() <= 10);
    }

    #[test]
    fn test_greedy_subdivide_respects_max_boxes() {
        let morton_strings = vec![
            "-311111".to_string(),
            "-322222".to_string(),
            "-411111".to_string(),
            "-422222".to_string(),
        ];

        let result = greedy_subdivide(morton_strings, 3, None);
        assert!(result.len() <= 3);
    }

    #[test]
    fn test_greedy_subdivide_ordermax() {
        let morton_strings = vec![
            "-3111111111".to_string(),
            "-3222222222".to_string(),
        ];

        let result = greedy_subdivide(morton_strings, 10, Some(2));

        // All results should be order <= 2 (3 digits or less)
        for prefix in &result {
            let order = estimate_order_from_prefix(prefix);
            assert!(order <= 2, "Order {} exceeds ordermax 2 for prefix {}", order, prefix);
        }
    }
}

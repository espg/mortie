//! Multi-Order Coverage (MOC) helpers for the hierarchical coverer (issue #30).
//!
//! The hierarchical descent emits cells at mixed orders — coarse cells for the
//! interior, fine cells along the boundary.  Because a mortie morton index
//! self-encodes its order (decimal digit count) and ancestry is a decimal-digit
//! prefix, a mixed-order coverage is just a `Vec<i64>`; these helpers convert
//! between the compact MOC form and a flat single-order list.
//!
//! - [`normalize`] collapses complete sibling sets into their parent and drops
//!   any cell already contained in a coarser one → the canonical compact MOC.
//! - [`to_order`] densifies every cell to a single target order → the flat,
//!   back-compatible form.
//!
//! `HashSet`/`HashMap` are used only for membership; every result is sorted, so
//! output is independent of hash iteration order (unlike the issue #28 bug).

use std::collections::{HashMap, HashSet};

use crate::morton::{mort2nested, nested2mort};

/// Densify a (possibly mixed-order) morton set to a flat list at `order`.
///
/// Cells coarser than `order` are expanded to their `4^(order-depth)`
/// descendants; cells already at `order` are kept; cells finer than `order`
/// (unusual) are coarsened to their ancestor at `order`.  Returns sorted unique.
pub fn to_order(morton: &[i64], order: u8) -> Vec<i64> {
    let mut out = Vec::with_capacity(morton.len());
    for &m in morton {
        let (nested, depth) = mort2nested(m);
        if depth == order {
            out.push(m);
        } else if depth < order {
            let shift = 2 * (order - depth) as u32;
            let base = nested << shift;
            let count = 1u64 << shift;
            for k in 0..count {
                out.push(nested2mort(base + k, order));
            }
        } else {
            let up = 2 * (depth - order) as u32;
            out.push(nested2mort(nested >> up, order));
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Collapse a morton set into its canonical compact MOC.
///
/// First drops any cell that already has an ancestor in the set, then
/// repeatedly merges any 4 complete sibling cells into their parent.  Returns
/// sorted unique morton indices at mixed orders.
pub fn normalize(morton: &[i64]) -> Vec<i64> {
    let mut set: HashSet<(u64, u8)> = morton.iter().map(|&m| mort2nested(m)).collect();

    // (1) Ancestor-prune: a cell contained in a coarser cell is redundant.
    let snapshot: Vec<(u64, u8)> = set.iter().copied().collect();
    for &(n, d) in &snapshot {
        let (mut nn, mut dd) = (n, d);
        while dd > 0 {
            nn >>= 2;
            dd -= 1;
            if set.contains(&(nn, dd)) {
                set.remove(&(n, d));
                break;
            }
        }
    }

    // (2) Sibling-merge: collapse 4 complete children into their parent, until
    // no further merges are possible (handles multi-level collapse over passes).
    loop {
        let mut by_parent: HashMap<(u64, u8), Vec<u64>> = HashMap::new();
        for &(n, d) in &set {
            if d > 0 {
                by_parent.entry((n >> 2, d - 1)).or_default().push(n & 3);
            }
        }
        let mut changed = false;
        for (parent, child_slots) in by_parent {
            let distinct: HashSet<u64> = child_slots.iter().copied().collect();
            if distinct.len() == 4 {
                let (pn, pd) = parent;
                for slot in 0..4u64 {
                    set.remove(&((pn << 2) | slot, pd + 1));
                }
                set.insert((pn, pd));
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut out: Vec<i64> = set.iter().map(|&(n, d)| nested2mort(n, d)).collect();
    out.sort_unstable();
    out
}

// ── tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_order_expands_coarse() {
        // One cell at depth 2 → 4^(5-2) = 64 leaves at order 5.
        let coarse = nested2mort(7, 2);
        let flat = to_order(&[coarse], 5);
        assert_eq!(flat.len(), 64);
        // All leaves must be descendants (same nested prefix at depth 2).
        for &m in &flat {
            let (n, d) = mort2nested(m);
            assert_eq!(d, 5);
            assert_eq!(n >> (2 * 3), 7, "leaf must descend from cell 7@2");
        }
    }

    #[test]
    fn test_to_order_keeps_same_order() {
        let cells: Vec<i64> = (10..20).map(|n| nested2mort(n, 6)).collect();
        let flat = to_order(&cells, 6);
        let mut expected = cells.clone();
        expected.sort_unstable();
        assert_eq!(flat, expected);
    }

    #[test]
    fn test_normalize_merges_siblings() {
        // The 4 children of cell 3@4 collapse to the single parent 3@3.
        let children: Vec<i64> = (0..4).map(|s| nested2mort((3 << 2) | s, 4)).collect();
        let norm = normalize(&children);
        assert_eq!(norm, vec![nested2mort(3, 3)]);
    }

    #[test]
    fn test_normalize_partial_siblings_unchanged() {
        // Only 3 of 4 children present → no merge.
        let children: Vec<i64> = (0..3).map(|s| nested2mort((3 << 2) | s, 4)).collect();
        let norm = normalize(&children);
        let mut expected = children.clone();
        expected.sort_unstable();
        assert_eq!(norm, expected);
    }

    #[test]
    fn test_normalize_ancestor_prune() {
        // A coarse cell plus one of its descendants → the descendant is dropped.
        let parent = nested2mort(5, 3);
        let child = nested2mort((5 << 2) | 2, 4);
        let norm = normalize(&[parent, child]);
        assert_eq!(norm, vec![parent]);
    }

    #[test]
    fn test_normalize_then_to_order_roundtrip() {
        // Densify-invariance: normalizing must not change the flattened cover.
        let children: Vec<i64> = (0..4).map(|s| nested2mort((9 << 2) | s, 5)).collect();
        let direct = to_order(&children, 5);
        let viamoc = to_order(&normalize(&children), 5);
        assert_eq!(direct, viamoc);
    }
}

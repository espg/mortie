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
//! [`normalize`] runs as a single sorted-stream pass (no hash-set fixpoint), so
//! output is deterministic and independent of any iteration order (unlike the
//! issue #28 bug).

use crate::morton::{mort2nested, nested2mort};

/// Maximum HEALPix depth mortie supports (order 18 → 64-bit morton limit).
/// Cells are mapped to half-open ranges over the uniform grid at this depth so a
/// single sort linearizes ancestry: a coarser cell's range strictly contains its
/// descendants'.
const MAX_DEPTH: u8 = 18;

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

/// Half-open range `[start, end)` a cell covers on the uniform `MAX_DEPTH` grid.
#[inline]
fn cell_range(nested: u64, depth: u8) -> (u64, u64) {
    let shift = 2 * (MAX_DEPTH - depth) as u32;
    let start = nested << shift;
    (start, start + (1u64 << shift))
}

/// Collapse a morton set into its canonical compact MOC.
///
/// Runs in a single sorted-stream pass: cells are linearized by their range on
/// the uniform `MAX_DEPTH` grid, contained (descendant) cells are pruned in one
/// sweep, then a stack merge collapses every complete sibling quartet into its
/// parent, cascading without a fixpoint loop.  Returns sorted unique morton
/// indices at mixed orders.
pub fn normalize(morton: &[i64]) -> Vec<i64> {
    if morton.is_empty() {
        return Vec::new();
    }

    // Linearize: sort by range start ascending, then by range end descending so
    // a coarser cell sorts ahead of any descendant sharing its start.
    let mut cells: Vec<(u64, u64, u64, u8)> = morton
        .iter()
        .map(|&m| {
            let (n, d) = mort2nested(m);
            let (start, end) = cell_range(n, d);
            (start, end, n, d)
        })
        .collect();
    cells.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));

    // (1) Ancestor-prune + dedup in one sweep: keep a cell only if it starts at
    // or beyond the end of the last kept cell (anything else is contained in it).
    let mut stack: Vec<(u64, u8)> = Vec::with_capacity(cells.len());
    let mut last_end: u64 = 0;
    let mut first = true;
    for (start, end, n, d) in cells {
        if first || start >= last_end {
            // (2) Stack merge: push, then collapse complete sibling quartets.
            stack.push((n, d));
            while stack.len() >= 4 {
                let len = stack.len();
                let (n0, d0) = stack[len - 4];
                let (n1, d1) = stack[len - 3];
                let (n2, d2) = stack[len - 2];
                let (n3, d3) = stack[len - 1];
                let parent = n0 >> 2;
                let complete = d0 > 0
                    && d1 == d0
                    && d2 == d0
                    && d3 == d0
                    && n1 >> 2 == parent
                    && n2 >> 2 == parent
                    && n3 >> 2 == parent
                    && n0 & 3 == 0
                    && n1 & 3 == 1
                    && n2 & 3 == 2
                    && n3 & 3 == 3;
                if !complete {
                    break;
                }
                stack.truncate(len - 4);
                stack.push((parent, d0 - 1));
            }
            last_end = end;
            first = false;
        }
    }

    let mut out: Vec<i64> = stack.iter().map(|&(n, d)| nested2mort(n, d)).collect();
    out.sort_unstable();
    out
}

// ── boolean set-algebra (issue #50, Option D: no new dependency) ───────────
//
// Decision #50 is Option D: fast set-algebra with no new crate. We first tried
// routing through the `healpix 0.3.2` BMOC (`or`/`and`/`minus`) we already
// depend on — `or`/`and` work, but its `minus` hangs on mixed-order MOC inputs
// (an upstream loop bug). Rather than ship a mix where two ops use BMOC and one
// is in-house, all three are implemented in-house as the same single-pass
// sorted-range linear merge that [`normalize`] already uses: map each cover to
// disjoint half-open ranges on the uniform `MAX_DEPTH` grid, merge the two
// sorted range streams, then decompose the result back into HEALPix-aligned
// cells. O(n+m), deterministic, no second HEALPix crate.

/// Map a morton set to its sorted, coalesced disjoint `[start, end)` ranges on
/// the uniform `MAX_DEPTH` grid.
///
/// Runs through [`normalize`] first so the cells are a proper MOC (no overlap),
/// then merges touching ranges into maximal runs — the canonical interval form
/// the merges operate on.
fn morton_to_ranges(morton: &[i64]) -> Vec<(u64, u64)> {
    let cells = normalize(morton);
    // normalize() returns cells sorted by signed-morton value, which is NOT the
    // grid-range order (mixed depths and the southern-hemisphere sign flip break
    // it), so sort the ranges explicitly before coalescing touching runs.
    let mut ranges: Vec<(u64, u64)> = cells
        .iter()
        .map(|&m| {
            let (n, d) = mort2nested(m);
            cell_range(n, d)
        })
        .collect();
    ranges.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
    for (start, end) in ranges {
        match merged.last_mut() {
            Some(last) if last.1 == start => last.1 = end,
            _ => merged.push((start, end)),
        }
    }
    merged
}

/// Decompose disjoint sorted `[start, end)` ranges back into a canonical compact
/// MOC (mixed-order morton, sorted).
///
/// Each range is greedily split into the largest HEALPix-aligned cells that fit
/// (the standard range→MOC decomposition), then [`normalize`] collapses any
/// complete sibling quartets that span adjacent ranges.
fn ranges_to_morton(ranges: &[(u64, u64)]) -> Vec<i64> {
    let mut cells: Vec<i64> = Vec::new();
    for &(mut start, end) in ranges {
        while start < end {
            // Largest aligned cell at `start`: limited by start's alignment
            // (trailing-zero pairs) and by the remaining span.
            let align = if start == 0 {
                MAX_DEPTH
            } else {
                (start.trailing_zeros() / 2).min(MAX_DEPTH as u32) as u8
            };
            let span = end - start;
            // Largest power-of-four <= span caps the cell size too.
            let span_levels = (63 - span.leading_zeros()) / 2;
            let levels = (align as u32).min(span_levels) as u8;
            let depth = MAX_DEPTH - levels;
            let nested = start >> (2 * levels as u32);
            cells.push(nested2mort(nested, depth));
            start += 1u64 << (2 * levels as u32);
        }
    }
    normalize(&cells)
}

/// Union of two morton sets → canonical compact MOC (range OR).
///
/// Equivalent to `normalize(concat(a, b))` but as a linear merge over the two
/// sorted range streams.
pub fn union(a: &[i64], b: &[i64]) -> Vec<i64> {
    let (ra, rb) = (morton_to_ranges(a), morton_to_ranges(b));
    let mut out: Vec<(u64, u64)> = Vec::with_capacity(ra.len() + rb.len());
    let push = |out: &mut Vec<(u64, u64)>, s: u64, e: u64| match out.last_mut() {
        Some(last) if last.1 >= s => last.1 = last.1.max(e),
        _ => out.push((s, e)),
    };
    let (mut i, mut j) = (0, 0);
    while i < ra.len() && j < rb.len() {
        if ra[i].0 <= rb[j].0 {
            push(&mut out, ra[i].0, ra[i].1);
            i += 1;
        } else {
            push(&mut out, rb[j].0, rb[j].1);
            j += 1;
        }
    }
    while i < ra.len() {
        push(&mut out, ra[i].0, ra[i].1);
        i += 1;
    }
    while j < rb.len() {
        push(&mut out, rb[j].0, rb[j].1);
        j += 1;
    }
    ranges_to_morton(&out)
}

/// Intersection of two morton sets → canonical compact MOC (range AND).
///
/// Returns the cells covered by both inputs (empty when disjoint).
pub fn intersection(a: &[i64], b: &[i64]) -> Vec<i64> {
    let (ra, rb) = (morton_to_ranges(a), morton_to_ranges(b));
    let mut out: Vec<(u64, u64)> = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < ra.len() && j < rb.len() {
        let lo = ra[i].0.max(rb[j].0);
        let hi = ra[i].1.min(rb[j].1);
        if lo < hi {
            out.push((lo, hi));
        }
        // Advance whichever range ends first.
        if ra[i].1 < rb[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    ranges_to_morton(&out)
}

/// Difference `a \ b` → canonical compact MOC (range subtraction).
///
/// Returns the part of `a` not covered by `b`.
pub fn difference(a: &[i64], b: &[i64]) -> Vec<i64> {
    let (ra, rb) = (morton_to_ranges(a), morton_to_ranges(b));
    let mut out: Vec<(u64, u64)> = Vec::new();
    let mut j = 0;
    for &(start, end) in &ra {
        let mut cur = start;
        // Carve out every `b` range overlapping [start, end).
        while j < rb.len() && rb[j].1 <= cur {
            j += 1;
        }
        let mut k = j;
        while k < rb.len() && rb[k].0 < end {
            let (bs, be) = rb[k];
            if bs > cur {
                out.push((cur, bs.min(end)));
            }
            cur = cur.max(be);
            if cur >= end {
                break;
            }
            k += 1;
        }
        if cur < end {
            out.push((cur, end));
        }
    }
    ranges_to_morton(&out)
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

    #[test]
    fn test_normalize_empty() {
        assert_eq!(normalize(&[]), Vec::<i64>::new());
    }

    #[test]
    fn test_normalize_multilevel_cascade() {
        // 16 leaves at depth 5 under cell 2@3 → cascade two levels to 2@3.
        let mut leaves = Vec::new();
        for c in 0..4u64 {
            for s in 0..4u64 {
                leaves.push(nested2mort(((2 << 2 | c) << 2) | s, 5));
            }
        }
        assert_eq!(normalize(&leaves), vec![nested2mort(2, 3)]);
    }

    #[test]
    fn test_normalize_dedup() {
        // Duplicate inputs collapse to one cell.
        let c = nested2mort(42, 6);
        assert_eq!(normalize(&[c, c, c]), vec![c]);
    }

    #[test]
    fn test_normalize_distinct_parents_unchanged() {
        // Complete quartets under two different parents stay as two parents.
        let mut cells: Vec<i64> = (0..4).map(|s| nested2mort((1 << 2) | s, 4)).collect();
        cells.extend((0..4).map(|s| nested2mort((7 << 2) | s, 4)));
        let mut expected = vec![nested2mort(1, 3), nested2mort(7, 3)];
        expected.sort_unstable();
        assert_eq!(normalize(&cells), expected);
    }

    /// Brute-force reference: prune descendants, then fixpoint sibling-merge via
    /// sorted vectors (no hashing), used to differentially test the fast path.
    fn normalize_reference(morton: &[i64]) -> Vec<i64> {
        use std::collections::BTreeSet;
        let mut set: BTreeSet<(u8, u64)> = morton
            .iter()
            .map(|&m| {
                let (n, d) = mort2nested(m);
                (d, n)
            })
            .collect();
        // ancestor-prune
        let snap: Vec<(u8, u64)> = set.iter().copied().collect();
        for &(d, n) in &snap {
            let (mut dd, mut nn) = (d, n);
            while dd > 0 {
                dd -= 1;
                nn >>= 2;
                if set.contains(&(dd, nn)) {
                    set.remove(&(d, n));
                    break;
                }
            }
        }
        // fixpoint sibling-merge
        loop {
            let mut merged = false;
            let snap: Vec<(u8, u64)> = set.iter().copied().collect();
            for &(d, n) in &snap {
                if d == 0 || n & 3 != 0 {
                    continue;
                }
                if (1..4).all(|s| set.contains(&(d, n | s))) {
                    for s in 0..4 {
                        set.remove(&(d, n | s));
                    }
                    set.insert((d - 1, n >> 2));
                    merged = true;
                }
            }
            if !merged {
                break;
            }
        }
        let mut out: Vec<i64> = set.iter().map(|&(d, n)| nested2mort(n, d)).collect();
        out.sort_unstable();
        out
    }

    #[test]
    fn test_normalize_matches_reference() {
        // Random mixed-order covers: fast path must equal the brute-force ref.
        let mut state = 0x9e3779b97f4a7c15u64;
        let mut rng = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..200 {
            let k = (rng() % 60) as usize + 1;
            let cells: Vec<i64> = (0..k)
                .map(|_| {
                    let depth = (rng() % 6) as u8 + 1;
                    let nside_sq = 1u64 << (2 * depth as u32);
                    let base = rng() % 12;
                    let n = base * nside_sq + rng() % nside_sq;
                    nested2mort(n, depth)
                })
                .collect();
            assert_eq!(normalize(&cells), normalize_reference(&cells));
        }

        // Dense subtrees: emit many leaves under one shallow root so complete,
        // cascading sibling quartets are exercised (with random gaps so partial
        // quartets and ancestor cells also occur).
        for _ in 0..200 {
            let root_depth = (rng() % 3) as u8 + 1;
            let leaf_depth = root_depth + 2 + (rng() % 2) as u8;
            let base = rng() % 12;
            let root = base * (1u64 << (2 * root_depth as u32)) + rng() % 4;
            let span = 1u64 << (2 * (leaf_depth - root_depth) as u32);
            let base_leaf = root << (2 * (leaf_depth - root_depth) as u32);
            let mut cells: Vec<i64> = Vec::new();
            for off in 0..span {
                if rng() % 8 != 0 {
                    cells.push(nested2mort(base_leaf + off, leaf_depth));
                }
            }
            if rng() % 4 == 0 {
                cells.push(nested2mort(root, root_depth)); // ancestor present
            }
            if cells.is_empty() {
                continue;
            }
            assert_eq!(normalize(&cells), normalize_reference(&cells));
        }
    }

    // ── set-op tests (issue #50) ────────────────────────────────────────────

    /// Expand a morton set to the *set* of its descendant leaf cells at `order`
    /// (every cell densified to `order`), as a `BTreeSet` of nested ids — the
    /// independent ground truth for the set-ops. Every cell in the test inputs
    /// is at depth <= `order`.
    fn leaf_set(morton: &[i64], order: u8) -> std::collections::BTreeSet<u64> {
        let mut s = std::collections::BTreeSet::new();
        for &m in morton {
            let (n, d) = mort2nested(m);
            let shift = 2 * (order - d) as u32;
            let base = n << shift;
            for k in 0..(1u64 << shift) {
                s.insert(base + k);
            }
        }
        s
    }

    /// Densify a morton set and return its leaf-cell set at `order` (so a
    /// mixed-order result is compared by the flat region it covers, independent
    /// of how it is tiled).
    fn covered_set(result: &[i64], order: u8) -> std::collections::BTreeSet<u64> {
        leaf_set(result, order)
    }

    #[test]
    fn test_union_equals_normalize_concat() {
        // Bit-for-bit: union(a,b) must equal normalize(concat(a,b)).
        let a: Vec<i64> = (0..6).map(|n| nested2mort(n, 4)).collect();
        let b: Vec<i64> = (4..10).map(|n| nested2mort(n, 4)).collect();
        let mut concat = a.clone();
        concat.extend_from_slice(&b);
        assert_eq!(union(&a, &b), normalize(&concat));
    }

    #[test]
    fn test_intersection_brute_force() {
        // Overlapping equatorial cells; intersection covers exactly the overlap.
        let order = 6u8;
        let a: Vec<i64> = (0..20).map(|n| nested2mort(n, order)).collect();
        let b: Vec<i64> = (10..30).map(|n| nested2mort(n, order)).collect();
        let got = intersection(&a, &b);
        let expect: std::collections::BTreeSet<u64> = leaf_set(&a, order)
            .intersection(&leaf_set(&b, order))
            .copied()
            .collect();
        assert_eq!(covered_set(&got, order), expect);
    }

    #[test]
    fn test_difference_brute_force() {
        let order = 6u8;
        let a: Vec<i64> = (0..20).map(|n| nested2mort(n, order)).collect();
        let b: Vec<i64> = (10..30).map(|n| nested2mort(n, order)).collect();
        let got = difference(&a, &b);
        let expect: std::collections::BTreeSet<u64> = leaf_set(&a, order)
            .difference(&leaf_set(&b, order))
            .copied()
            .collect();
        assert_eq!(covered_set(&got, order), expect);
    }

    #[test]
    fn test_mixed_order_inputs() {
        // a holds a coarse parent (3@3); b holds two of its order-4 children.
        // intersection = those two children; difference = the other two.
        let order = 4u8;
        let a = vec![nested2mort(3, 3)];
        let b = vec![nested2mort((3 << 2) | 1, 4), nested2mort((3 << 2) | 2, 4)];
        let inter = intersection(&a, &b);
        assert_eq!(
            covered_set(&inter, order),
            leaf_set(&a, order)
                .intersection(&leaf_set(&b, order))
                .copied()
                .collect()
        );
        let diff = difference(&a, &b);
        assert_eq!(
            covered_set(&diff, order),
            leaf_set(&a, order)
                .difference(&leaf_set(&b, order))
                .copied()
                .collect()
        );
    }

    #[test]
    fn test_setops_empty_inputs() {
        let a: Vec<i64> = (0..4).map(|n| nested2mort(n, 4)).collect();
        // union with empty == normalize(a)
        assert_eq!(union(&a, &[]), normalize(&a));
        assert_eq!(union(&[], &a), normalize(&a));
        // intersection with empty == empty
        assert_eq!(intersection(&a, &[]), Vec::<i64>::new());
        assert_eq!(intersection(&[], &a), Vec::<i64>::new());
        // difference: a\∅ == normalize(a); ∅\a == ∅
        assert_eq!(difference(&a, &[]), normalize(&a));
        assert_eq!(difference(&[], &a), Vec::<i64>::new());
        // both empty
        assert_eq!(union(&[], &[]), Vec::<i64>::new());
        assert_eq!(intersection(&[], &[]), Vec::<i64>::new());
        assert_eq!(difference(&[], &[]), Vec::<i64>::new());
    }

    #[test]
    fn test_setops_both_hemispheres() {
        // Northern (base 2) and southern (base 8) cells in the same covers.
        let order = 5u8;
        let nside_sq = 1u64 << (2 * order as u32);
        let a: Vec<i64> = (0..8)
            .map(|k| nested2mort(2 * nside_sq + k, order))
            .chain((0..8).map(|k| nested2mort(8 * nside_sq + k, order)))
            .collect();
        let b: Vec<i64> = (4..12)
            .map(|k| nested2mort(2 * nside_sq + k, order))
            .chain((4..12).map(|k| nested2mort(8 * nside_sq + k, order)))
            .collect();
        for (got, op) in [
            (union(&a, &b), 0u8),
            (intersection(&a, &b), 1),
            (difference(&a, &b), 2),
        ] {
            let (sa, sb) = (leaf_set(&a, order), leaf_set(&b, order));
            let expect: std::collections::BTreeSet<u64> = match op {
                0 => sa.union(&sb).copied().collect(),
                1 => sa.intersection(&sb).copied().collect(),
                _ => sa.difference(&sb).copied().collect(),
            };
            assert_eq!(covered_set(&got, order), expect);
        }
    }

    #[test]
    fn test_setops_match_brute_force_random() {
        // Random mixed-order covers over a shallow common depth; each op is
        // checked against the leaf-set ground truth (densify both sides, do the
        // set-op on leaf ids). order kept small so densification is cheap.
        let order = 7u8;
        let mut state = 0x243f6a8885a308d3u64;
        let mut rng = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut gen_cover = |rng: &mut dyn FnMut() -> u64| -> Vec<i64> {
            let k = (rng() % 40) as usize + 1;
            (0..k)
                .map(|_| {
                    let depth = (rng() % order as u64) as u8 + 1;
                    let nside_sq = 1u64 << (2 * depth as u32);
                    let base = rng() % 12;
                    nested2mort(base * nside_sq + rng() % nside_sq, depth)
                })
                .collect()
        };
        for _ in 0..200 {
            let a = gen_cover(&mut rng);
            let b = gen_cover(&mut rng);
            let (sa, sb) = (leaf_set(&a, order), leaf_set(&b, order));

            let u_expect: std::collections::BTreeSet<u64> = sa.union(&sb).copied().collect();
            assert_eq!(covered_set(&union(&a, &b), order), u_expect);
            // union must also equal normalize(concat) bit-for-bit.
            let mut concat = a.clone();
            concat.extend_from_slice(&b);
            assert_eq!(union(&a, &b), normalize(&concat));

            let i_expect: std::collections::BTreeSet<u64> = sa.intersection(&sb).copied().collect();
            assert_eq!(covered_set(&intersection(&a, &b), order), i_expect);

            let d_expect: std::collections::BTreeSet<u64> = sa.difference(&sb).copied().collect();
            assert_eq!(covered_set(&difference(&a, &b), order), d_expect);
        }
    }
}

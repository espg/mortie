//! Multi-Order Coverage (MOC) helpers for the hierarchical coverer (issue #30).
//!
//! The hierarchical descent emits cells at mixed orders — coarse cells for the
//! interior, fine cells along the boundary.  Because a mortie morton word
//! self-encodes its order and ancestry, a mixed-order coverage is just a
//! `Vec<u64>`; these helpers convert between the compact MOC form and a flat
//! single-order list.  The word is an unsigned `u64` (issue #58), so a raw
//! `sort_unstable` over the words is the Z-order with no sign special-casing.
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
use healpix::bmoc::{Bmoc, MutableBmoc};

/// Maximum HEALPix depth mortie supports — tied to the packed-u64 kernel's
/// [`crate::decimal_morton::MAX_ORDER`] (29) so the two cannot drift (issue #60).
/// Cells are mapped to half-open ranges over the uniform grid at this depth so a
/// single sort linearizes ancestry: a coarser cell's range strictly contains its
/// descendants'.  The range start of a depth-`d` cell is `nested << 2*(29-d)`;
/// the deepest depth-29 nested hash is `12*4^29 - 1 < 2^61.6`, so a depth-29
/// range `[start, start + 4^(29-d))` stays well within u64 — no shift, add, or
/// `1 << shift` overflow at any depth in `1..=29` (the old `18` was the retired
/// decimal-i64 morton's ceiling, not a u64 limit).
const MAX_DEPTH: u8 = crate::decimal_morton::MAX_ORDER;

// ── BMOC-backed boolean set algebra (issue #50, Option D) ──────────────────
//
// `moc_or`/`moc_and`/`moc_minus` are thin wrappers around the `healpix` crate's
// BMOC `or`/`and`/`minus`.  A mortie cover is a `Vec<u64>` of packed decimal
// morton words at mixed orders; the BMOC works on `(depth, nested hash)` pairs.
// So each op follows the same decode → op → encode → normalize shape:
//
//   1. decode each morton u64 → `(depth, nested u64)` via `mort2nested`,
//   2. build a `MutableBmoc` (packed/valid) per input,
//   3. run the BMOC op,
//   4. re-encode the result cells → morton u64 via `nested2mort`,
//   5. `normalize` to mortie's canonical compact form.
//
// The base-cell prefix carries the hemisphere, which round-trips through the
// nested hash, so southern covers need no special handling at this boundary.

/// Build a packed, valid BMOC at `MAX_DEPTH` from a morton cover.
///
/// The BMOC ops require a canonical, non-overlapping input cover (a valid MOC):
/// `pack()` only merges complete sibling quartets, it does not prune a cell that
/// is a descendant of another in the same set.  `normalize` first delivers
/// exactly that — sorted, sibling-merged, ancestor-pruned — so a raw mixed-order
/// cover with overlaps is made valid before it reaches the BMOC.
fn build_bmoc(morton: &[u64]) -> MutableBmoc {
    let canonical = normalize(morton);
    let mut builder = MutableBmoc::<false>::with_capacity(MAX_DEPTH, canonical.len());
    for &m in &canonical {
        let (nested, depth) = mort2nested(m);
        builder.push_unchecked(depth, nested, true);
    }
    builder.into_packed()
}

/// Re-encode a BMOC's cells back to a normalized mortie morton cover.
///
/// Discarding each cell's full/partial flag is safe here: every cell mortie
/// pushes is `is_full = true`, so the BMOC only ever emits full cells — the flag
/// carries no information a `(hash, depth)` cover would lose.
fn bmoc_to_morton(bmoc: MutableBmoc) -> Vec<u64> {
    let cells: Vec<u64> = bmoc
        .into_cells()
        .map(|c| nested2mort(c.hash, c.depth))
        .collect();
    normalize(&cells)
}

/// Union (logical OR) of two morton covers.
///
/// Equivalent to `normalize(concat(a, b))`; backed by BMOC `or`.
pub fn moc_or(a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.is_empty() {
        return normalize(b);
    }
    if b.is_empty() {
        return normalize(a);
    }
    bmoc_to_morton(build_bmoc(a).or(&build_bmoc(b)))
}

/// Intersection (logical AND) of two morton covers.
///
/// Backed by BMOC `and`.
pub fn moc_and(a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    bmoc_to_morton(build_bmoc(a).and(&build_bmoc(b)))
}

/// Difference `a \ b` of two morton covers.
///
/// Backed by BMOC `minus`.  The `minus` infinite-loop on mixed-order inputs in
/// upstream `healpix` 0.3.2 is fixed in the pinned fork (see Cargo.toml).
pub fn moc_minus(a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.is_empty() {
        return Vec::new();
    }
    if b.is_empty() {
        return normalize(a);
    }
    bmoc_to_morton(build_bmoc(a).minus(&build_bmoc(b)))
}

/// Densify a (possibly mixed-order) morton set to a flat list at `order`.
///
/// Cells coarser than `order` are expanded to their `4^(order-depth)`
/// descendants; cells already at `order` are kept; cells finer than `order`
/// (unusual) are coarsened to their ancestor at `order`.  Returns sorted unique.
pub fn to_order(morton: &[u64], order: u8) -> Vec<u64> {
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
pub fn normalize(morton: &[u64]) -> Vec<u64> {
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

    let mut out: Vec<u64> = stack.iter().map(|&(n, d)| nested2mort(n, d)).collect();
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
        let cells: Vec<u64> = (10..20).map(|n| nested2mort(n, 6)).collect();
        let flat = to_order(&cells, 6);
        let mut expected = cells.clone();
        expected.sort_unstable();
        assert_eq!(flat, expected);
    }

    #[test]
    fn test_normalize_merges_siblings() {
        // The 4 children of cell 3@4 collapse to the single parent 3@3.
        let children: Vec<u64> = (0..4).map(|s| nested2mort((3 << 2) | s, 4)).collect();
        let norm = normalize(&children);
        assert_eq!(norm, vec![nested2mort(3, 3)]);
    }

    #[test]
    fn test_normalize_partial_siblings_unchanged() {
        // Only 3 of 4 children present → no merge.
        let children: Vec<u64> = (0..3).map(|s| nested2mort((3 << 2) | s, 4)).collect();
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
        let children: Vec<u64> = (0..4).map(|s| nested2mort((9 << 2) | s, 5)).collect();
        let direct = to_order(&children, 5);
        let viamoc = to_order(&normalize(&children), 5);
        assert_eq!(direct, viamoc);
    }

    #[test]
    fn test_normalize_empty() {
        assert_eq!(normalize(&[]), Vec::<u64>::new());
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
        let mut cells: Vec<u64> = (0..4).map(|s| nested2mort((1 << 2) | s, 4)).collect();
        cells.extend((0..4).map(|s| nested2mort((7 << 2) | s, 4)));
        let mut expected = vec![nested2mort(1, 3), nested2mort(7, 3)];
        expected.sort_unstable();
        assert_eq!(normalize(&cells), expected);
    }

    /// Brute-force reference: prune descendants, then fixpoint sibling-merge via
    /// sorted vectors (no hashing), used to differentially test the fast path.
    fn normalize_reference(morton: &[u64]) -> Vec<u64> {
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
        let mut out: Vec<u64> = set.iter().map(|&(d, n)| nested2mort(n, d)).collect();
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
            let cells: Vec<u64> = (0..k)
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
            let mut cells: Vec<u64> = Vec::new();
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

    // ── BMOC set-algebra tests (issue #50) ─────────────────────────────────

    /// Brute-force reference for a binary set op: densify both covers to a deep
    /// common order, run the op on the flat leaf sets, then normalize.  `order`
    /// must be >= the deepest cell in either input for the result to be exact.
    fn setop_reference(a: &[u64], b: &[u64], order: u8, op: fn(bool, bool) -> bool) -> Vec<u64> {
        use std::collections::BTreeSet;
        let la: BTreeSet<u64> = to_order(a, order).into_iter().collect();
        let lb: BTreeSet<u64> = to_order(b, order).into_iter().collect();
        let mut out: Vec<u64> = la
            .union(&lb)
            .filter(|&&m| op(la.contains(&m), lb.contains(&m)))
            .copied()
            .collect();
        out.sort_unstable();
        normalize(&out)
    }

    fn ref_or(a: &[u64], b: &[u64], order: u8) -> Vec<u64> {
        setop_reference(a, b, order, |x, y| x || y)
    }
    fn ref_and(a: &[u64], b: &[u64], order: u8) -> Vec<u64> {
        setop_reference(a, b, order, |x, y| x && y)
    }
    fn ref_minus(a: &[u64], b: &[u64], order: u8) -> Vec<u64> {
        setop_reference(a, b, order, |x, y| x && !y)
    }

    #[test]
    fn test_or_equals_normalize_concat() {
        // `moc_or` must be bit-for-bit == normalize(concat(a, b)).
        let a: Vec<u64> = (0..6).map(|n| nested2mort(n, 4)).collect();
        let b: Vec<u64> = (3..10).map(|n| nested2mort(n, 4)).collect();
        let mut concat = a.clone();
        concat.extend_from_slice(&b);
        assert_eq!(moc_or(&a, &b), normalize(&concat));
    }

    #[test]
    fn test_and_brute_force() {
        let a: Vec<u64> = (0..8).map(|n| nested2mort(n, 5)).collect();
        let b: Vec<u64> = (4..12).map(|n| nested2mort(n, 5)).collect();
        assert_eq!(moc_and(&a, &b), ref_and(&a, &b, 5));
    }

    #[test]
    fn test_minus_brute_force() {
        let a: Vec<u64> = (0..8).map(|n| nested2mort(n, 5)).collect();
        let b: Vec<u64> = (4..12).map(|n| nested2mort(n, 5)).collect();
        assert_eq!(moc_minus(&a, &b), ref_minus(&a, &b, 5));
    }

    #[test]
    fn test_disjoint_covers() {
        // Disjoint: and = empty, minus = a, or = a ∪ b.
        let a: Vec<u64> = (0..4).map(|n| nested2mort(n, 4)).collect();
        let b: Vec<u64> = (100..104).map(|n| nested2mort(n, 4)).collect();
        assert_eq!(moc_and(&a, &b), Vec::<u64>::new());
        assert_eq!(moc_minus(&a, &b), normalize(&a));
        let mut concat = a.clone();
        concat.extend_from_slice(&b);
        assert_eq!(moc_or(&a, &b), normalize(&concat));
    }

    #[test]
    fn test_self_minus_empty() {
        let a: Vec<u64> = (0..10).map(|n| nested2mort(n, 5)).collect();
        assert_eq!(moc_minus(&a, &a), Vec::<u64>::new());
        // Self-and = self, self-or = self (idempotent).
        assert_eq!(moc_and(&a, &a), normalize(&a));
        assert_eq!(moc_or(&a, &a), normalize(&a));
    }

    #[test]
    fn test_empty_inputs() {
        let a: Vec<u64> = (0..4).map(|n| nested2mort(n, 4)).collect();
        let empty: Vec<u64> = Vec::new();
        assert_eq!(moc_or(&a, &empty), normalize(&a));
        assert_eq!(moc_or(&empty, &a), normalize(&a));
        assert_eq!(moc_and(&a, &empty), Vec::<u64>::new());
        assert_eq!(moc_minus(&a, &empty), normalize(&a));
        assert_eq!(moc_minus(&empty, &a), Vec::<u64>::new());
        assert_eq!(moc_or(&empty, &empty), Vec::<u64>::new());
    }

    #[test]
    fn test_southern_hemisphere() {
        // Southern parents (6-11) set bit 63 → large u64 words; must round-trip.
        let south_base = 8u64 << (2 * 4); // base cell 8, depth 4
        let a: Vec<u64> = (0..6).map(|s| nested2mort(south_base + s, 4)).collect();
        let b: Vec<u64> = (3..10).map(|s| nested2mort(south_base + s, 4)).collect();
        assert!(
            a.iter().all(|&m| m >= 1u64 << 63),
            "southern morton must set bit 63"
        );
        assert_eq!(moc_and(&a, &b), ref_and(&a, &b, 4));
        assert_eq!(moc_minus(&a, &b), ref_minus(&a, &b, 4));
        let mut concat = a.clone();
        concat.extend_from_slice(&b);
        assert_eq!(moc_or(&a, &b), normalize(&concat));
    }

    #[test]
    fn test_normalize_returns_unsigned_z_order_across_hemispheres() {
        // A cover spanning northern (bases 0-6) and southern (bases 7-11, bit 63
        // set) cells must come back in ascending *unsigned* word order — the
        // Z-order. With the u64 channel `normalize`'s raw `sort_unstable` gives
        // this for free (an i64 sort would float the southern words to the front).
        let mut cells: Vec<u64> = Vec::new();
        for base in [0u64, 6, 7, 11] {
            for s in 0..3u64 {
                cells.push(nested2mort((base << (2 * 4)) + s, 4));
            }
        }
        let out = normalize(&cells);
        for w in out.windows(2) {
            assert!(w[1] > w[0], "normalize output must be ascending unsigned");
        }
        // The largest base (11, bit 63 set) must sort *after* the smallest (0).
        let north0 = nested2mort(0, 4);
        let south11 = nested2mort(11u64 << (2 * 4), 4);
        assert!(
            south11 > north0,
            "southern word must sort after northern under unsigned order"
        );
    }

    #[test]
    fn test_mixed_order_inputs() {
        // a: a coarse cell at depth 2; b: some of its depth-4 descendants plus a
        // disjoint coarse cell. Exercises the mixed-order code paths in all ops.
        let coarse = nested2mort(3, 2); // covers nested 48..64 at depth 4
        let a = vec![coarse, nested2mort(20, 3)];
        let b: Vec<u64> = (48..56)
            .map(|n| nested2mort(n, 4))
            .chain(std::iter::once(nested2mort(200, 4)))
            .collect();
        assert_eq!(moc_and(&a, &b), ref_and(&a, &b, 4));
        assert_eq!(moc_minus(&a, &b), ref_minus(&a, &b, 4));
        assert_eq!(moc_or(&a, &b), ref_or(&a, &b, 4));
    }

    /// Regression for the upstream `healpix` 0.3.2 BMOC `minus` infinite-loop on
    /// mixed-order inputs (issue #50 / PR #52): when a finer left cell is fully
    /// covered by a coarser full right cell, neither cursor advanced.  The pinned
    /// fork adds the terminal `else` arm; this asserts `minus` now terminates and
    /// returns the correct result on exactly that trigger condition — a cover
    /// whose finer cells are descendants of the other's coarse full cells.
    #[test]
    fn test_minus_mixed_order_terminates_regression() {
        // The hang is in `minus`'s `Greater` arm, which fires only when a *finer*
        // LEFT (minuend) cell is a descendant of a coarser *full* RIGHT
        // (subtrahend) cell — both `is_full`. So the minuend must be the FINER
        // cover and the subtrahend the coarse full ancestor. (An earlier cut had
        // the operands reversed, so only the always-advancing `Less`/`Equal` arms
        // ran and the test would have passed on buggy 0.3.2 too.)
        //
        // minuend: scattered depth-5 + depth-3 cells inside base-cell-0's depth-1
        // cell 0 (no complete quartets, so they stay finer than it through
        // normalize), plus one depth-1 cell in base cell 5 so the result is
        // non-empty.
        let mut fine: Vec<u64> = [0u64, 5, 17, 42, 130]
            .iter()
            .map(|&n| nested2mort(n, 5))
            .collect();
        fine.push(nested2mort(3, 3)); // depth-3 cell, also inside depth-1 cell 0
        let outside = nested2mort((5u64 << 2) + 1, 1); // depth-1 cell in base cell 5
        fine.push(outside);
        // subtrahend: the coarse FULL depth-1 ancestor (base-cell-0 cell 0) of
        // every "inside" cell above.
        let coarse = vec![nested2mort(0, 1)];
        // Minuend finer, subtrahend coarser-full => exercises the buggy `Greater`
        // arm. On 0.3.2 this loops forever; reaching the assert proves the fork
        // fix. Correctness is checked against the brute-force reference.
        let got = moc_minus(&fine, &coarse);
        assert_eq!(got, ref_minus(&fine, &coarse, 5));
        assert!(
            !got.is_empty(),
            "the `outside` cell is not subtracted, so the result must be non-empty"
        );
    }

    #[test]
    fn test_setops_match_reference_fuzz() {
        // Randomized fuzz over many mixed-order cover pairs in both hemispheres.
        let mut state = 0xd1b54a32d192ed03u64;
        let mut rng = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let gen_cover = |rng: &mut dyn FnMut() -> u64| -> (Vec<u64>, u8) {
            let k = (rng() % 30) as usize + 1;
            let mut max_depth = 1u8;
            let cells: Vec<u64> = (0..k)
                .map(|_| {
                    let depth = (rng() % 5) as u8 + 1;
                    max_depth = max_depth.max(depth);
                    let nside_sq = 1u64 << (2 * depth as u32);
                    let base = rng() % 12;
                    let n = base * nside_sq + rng() % nside_sq;
                    nested2mort(n, depth)
                })
                .collect();
            (cells, max_depth)
        };
        for _ in 0..300 {
            let (a, da) = gen_cover(&mut rng);
            let (b, db) = gen_cover(&mut rng);
            let order = da.max(db);
            assert_eq!(moc_or(&a, &b), ref_or(&a, &b, order), "or mismatch");
            assert_eq!(moc_and(&a, &b), ref_and(&a, &b, order), "and mismatch");
            assert_eq!(
                moc_minus(&a, &b),
                ref_minus(&a, &b, order),
                "minus mismatch"
            );
        }
    }

    // ── high-order (issue #60): MAX_DEPTH lifted 18 → 29 ───────────────────
    //
    // These pin `normalize` and each set-op at orders 22 and 29 — the orders
    // the old `MAX_DEPTH = 18` cap rejected (zagg's `child_order + 3 = 22`
    // case, and the kernel ceiling 29).  The brute-force references densify to
    // a common order via `to_order`, so to keep that cheap every cell is built
    // *at or near* the test order (a coarser cell would expand to `4^(29-d)`
    // leaves).  Sibling quartets are placed deep so normalize still exercises
    // the merge/cascade path at high depth.

    /// Adjacent sibling-quartet roots at `depth`: roots `r, r+4, r+8, …` each
    /// get their 4 children at `depth+1`, plus a couple of lone deep cells.
    fn deep_mixed_cover(depth: u8, base: u64, roots: u64) -> Vec<u64> {
        let nside_sq = 1u64 << (2 * depth as u32);
        let origin = base * nside_sq; // first nested hash in base cell
        let mut cells = Vec::new();
        for r in 0..roots {
            let root = origin + r * 4;
            for s in 0..4u64 {
                cells.push(nested2mort((root << 2) | s, depth + 1));
            }
        }
        // a lone cell at `depth` and one at `depth+1` (no full quartet) so the
        // ancestor-prune and partial-quartet arms are also hit.
        cells.push(nested2mort(origin + 4 * roots, depth));
        cells.push(nested2mort(((origin + 4 * roots + 1) << 2) | 2, depth + 1));
        cells
    }

    #[test]
    fn test_normalize_high_order_22_and_29() {
        for &depth in &[21u8, 28] {
            // depth+1 is 22 / 29 — the orders the old cap forbade.
            let cover = deep_mixed_cover(depth, 5, 6);
            let got = normalize(&cover);
            assert_eq!(
                got,
                normalize_reference(&cover),
                "normalize @ {}",
                depth + 1
            );
            // Each complete quartet must have collapsed to its `depth` parent.
            assert!(
                got.iter().any(|&m| mort2nested(m).1 == depth),
                "a sibling quartet must merge up to depth {depth}"
            );
            // Deepest input cell sits at depth+1 (22 or 29) and round-trips.
            assert!(
                got.iter().all(|&m| mort2nested(m).1 <= depth + 1),
                "no cell may exceed the input depth {}",
                depth + 1
            );
        }
    }

    #[test]
    fn test_setops_high_order_22_and_29() {
        // Overlapping deep covers at orders 22 and 29: each op must match the
        // brute-force reference densified at that order.  Cells are all at the
        // test depth (or one above), so `to_order` does no large expansion.
        for &order in &[22u8, 29] {
            let nside_sq = 1u64 << (2 * order as u32);
            let origin = 7 * nside_sq; // base cell 7 (southern, bit 63 set path)
            let a: Vec<u64> = (0..10).map(|n| nested2mort(origin + n, order)).collect();
            let b: Vec<u64> = (5..15).map(|n| nested2mort(origin + n, order)).collect();
            assert_eq!(moc_or(&a, &b), ref_or(&a, &b, order), "or @ {order}");
            assert_eq!(moc_and(&a, &b), ref_and(&a, &b, order), "and @ {order}");
            assert_eq!(
                moc_minus(&a, &b),
                ref_minus(&a, &b, order),
                "minus @ {order}"
            );
            // and is the 5 shared cells (5..10); densifying back to `order` must
            // recover exactly those, proving the deep BMOC round-trip is lossless.
            let shared: Vec<u64> = (5..10).map(|n| nested2mort(origin + n, order)).collect();
            assert_eq!(to_order(&moc_and(&a, &b), order), shared, "and @ {order}");
            // a \ b is the 5 cells only in a (0..5).
            let only_a: Vec<u64> = (0..5).map(|n| nested2mort(origin + n, order)).collect();
            assert_eq!(
                to_order(&moc_minus(&a, &b), order),
                only_a,
                "minus @ {order}"
            );
        }
    }

    #[test]
    fn test_setops_mixed_order_29_boundary() {
        // A coarse depth-27 cell vs. its own depth-29 descendants: the BMOC
        // path must encode/decode the full depth-29 range without overflow.
        let nside_sq = 1u64 << (2 * 27u32);
        let coarse_nested = 3 * nside_sq + 11; // some cell in base 3 at depth 27
        let coarse = vec![nested2mort(coarse_nested, 27)];
        // its 16 depth-29 descendants occupy nested [coarse<<4, coarse<<4 + 16)
        let leaves: Vec<u64> = (0..16)
            .map(|k| nested2mort((coarse_nested << 4) + k, 29))
            .collect();
        // and(coarse, all-its-leaves) = the leaves (normalized back to coarse).
        assert_eq!(moc_and(&coarse, &leaves), normalize(&coarse));
        // minus(coarse, half the leaves) drops only that half.
        let half = &leaves[..8];
        assert_eq!(
            moc_minus(&coarse, half),
            ref_minus(&coarse, half, 29),
            "mixed-order minus @ depth 29"
        );
    }
}

//! Dense-output batches over the packed-word hierarchy (issue #156, phase 3).
//!
//! The two ops here are the **dense** half of the batch sweep: their results
//! are fixed-shape arrays, not ragged ones, so neither needs an arrow offsets
//! pair on the way out.
//!
//! * [`common_ancestors`] — ragged in (groups of words), one word out per
//!   group: a segmented reduce of [`super::common_ancestor`].
//! * [`children_of`] — flat in, a fixed-width block out: every parent at the
//!   same order yields exactly `4**d` children at the target order, so the
//!   result is an `(n, 4**d)` matrix.
//!
//! Both scalar kernels are this module's parent ([`super::common_ancestor`],
//! and the [`super::to_nested`] / [`super::from_nested`] pair the child
//! generator is built from), which is why the batches live here rather than
//! beside [`crate::moc::batch`] — the batch submodule sits with the scalar
//! kernel it wraps, the same axis the Python side splits on (issue #156's
//! domain ruling).
//!
//! # Memory posture
//!
//! Dense output makes this simpler than the ragged batches, and the honest
//! statement is short.  The pyfunction must `to_vec()` its numpy inputs — a
//! borrowed `&[u64]` cannot cross `py.allow_threads` — so an input copy is
//! **mandatory**, the term issue #162 records as missing from the merged
//! ragged batches' docs:
//!
//! * [`common_ancestors`]: peak is **input copy + result + one chunk of
//!   `Result`s + the reduction's own scratch**.  The result is one word per
//!   group, so with `g` groups over `n` words it is `g / n` of the input copy —
//!   for the t-digest consumer (groups of 2-3 centroids) about a third of it —
//!   and the chunk term is 2048 x 32 B = 64 KiB regardless of batch size.  The
//!   scratch is [`super::common_ancestor`]'s own `others` buffer, 16 B per
//!   non-first word in the group it is reducing, held for that whole reduction:
//!   one per group in flight, so the term is
//!   `min(threads, groups) * 16 B * max_group_size`.  It scales with the
//!   **largest single group**, not with `n` — which is why it is invisible in
//!   the consumer shape and dominant in a skewed one.  Measured over 5M groups
//!   of 3 order-9 words: 152.6 MiB of input, a 38.1 MiB result, and a 191.9 MiB
//!   peak — 1.01x the `input + result` model, **5.0x the result alone**, and a
//!   scratch term under a kilobyte.  Two skewed shapes, same order and
//!   instrument: 40 groups of 1M words peaks at 458.6 MiB against a 305.2 MiB
//!   `input + result` model (**1.50x**, excess 153.4 MiB against the 152.6 MiB
//!   the formula predicts), and a single 20M-word group at 460.0 MiB against
//!   152.7 MiB (**3.01x**, excess 307.3 against a predicted 305.2).  So the
//!   input copy is the peak for small groups only; size a worker off the
//!   largest group when groups are large.
//! * [`children_of`]: peak is **input copy + result + one chunk of
//!   `Result`s**, and here the result dominates by construction — it is
//!   `4**d` times the input.  The output is allocated once at its exact final
//!   size (no growth reallocation, so no 1.5x transient) and written in place;
//!   the only transient is the same 64 KiB chunk of per-word `Result`s.
//!   Measured over 1M order-6 parents refined to order 9: 7.6 MiB of input, a
//!   488.3 MiB result, and a 497.1 MiB peak — 1.00x the model.
//!
//! Both numbers come from sampled RSS in a process that loads its input rather
//! than building it, which matters: when the input is *constructed* in-process
//! the `to_vec()` copy is served out of already-resident freed heap and the
//! measured peak silently under-counts it (the same 5M-group case reads as
//! 77.3 MiB that way).  Size a worker off `input + result`, not the result.
//!
//! # Allocation posture
//!
//! [`children_of`]'s result grows as `4**d`, so an over-ambitious `order` asks
//! for an allocation no machine can serve.  It is therefore allocated
//! **fallibly** — `try_reserve_exact`, not `vec![0u64; total]` — because the
//! latter goes through Rust's infallible allocator, whose failure path is
//! `handle_alloc_error` -> `abort()`: the process dies with no Python
//! traceback and nothing a caller can `except`.  The scalar loop this op
//! replaces raises a catchable `MemoryError` from `np.arange` at exactly those
//! sizes, so the fallible path is what keeps the batch's observable behaviour
//! matching the scalar's (the same reasoning issue #108 used to keep
//! `moc_to_order`'s ceiling a catchable `ValueError`).
//!
//! What this does **not** fence is an allocation the OS accepts and then cannot
//! back: on an overcommitting kernel a request far larger than RAM succeeds at
//! `try_reserve` time and the process is killed later, while it is being
//! written.  Refusing those needs a *policy* ceiling (a `max_cells`-style
//! budget), which is a separate question from making the allocator fallible.
//! The `checked_mul` above is not that fence either: it only trips when the
//! element count overflows `usize`, which is far past any allocation a machine
//! would attempt.
//!
//! # Error posture
//!
//! Fail-fast naming the **lowest-index** offending item, as in
//! [`crate::coverage::batch`] and [`crate::moc::batch`]: a serial pre-pass
//! scans in index order, and the parallel pass materializes a whole chunk's
//! outcomes (chunks themselves in index order) and walks them in index order
//! before any is allowed to fail the call.  The error is therefore
//! deterministic under any rayon schedule — `try_reduce`-style short-circuiting
//! was rejected in PR #154 for exactly this reason.
//!
//! The two ops differ in *which* pass owns their domain errors, and the split
//! is deliberate:
//!
//! * [`children_of`] validates every word up front (decodability, the target
//!   order, the shared parent order), because it must know the row width
//!   before it can allocate at all.  Its parallel pass cannot fail except by
//!   panic.
//! * [`common_ancestors`] validates only the *layout* up front and leaves the
//!   per-group failures (an empty group, an undecodable word, words spanning
//!   more than one base cell) to the parallel pass.  Hoisting them into the
//!   pre-pass would break the lowest-index rule, not preserve it: the cheap
//!   check (emptiness) would then out-rank the expensive one (mixed base
//!   cells) regardless of index, so an empty group 7 would be reported ahead of
//!   a mixed-base-cell group 2.

use rayon::prelude::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

use super::{common_ancestor, from_nested, to_nested, MAX_ORDER};

/// Items processed per parallel chunk.
///
/// Same value and same reason as [`crate::moc::batch`]'s: the batch runs chunk
/// by chunk so only one chunk's per-item outcomes are ever resident, and 2048
/// is large enough that the per-chunk fork-join is noise against the work.
const CHUNK: usize = 2048;

/// Validate a ragged arrow list layout over `n_values` values.
///
/// Checks, in order: a non-empty `offsets` starting at 0, then per group
/// (lowest index first) offset monotonicity and bounds, and finally the
/// endpoint requirement `offsets[n_groups] == n_values`.  The offsets must
/// **exactly cover** the value array — both endpoints are checked and each
/// names which one failed — so stale or misaligned offsets that happen to stay
/// in bounds cannot silently reduce a valid-looking wrong set.  Per-group
/// errors name the offending group and are raised ahead of the endpoint check,
/// so the lowest-index group is still what a caller sees first.
///
/// This is the offsets-only spelling of the same contract
/// [`crate::coverage::batch`] and [`crate::moc::batch`] enforce.  They are not
/// folded into one helper because each of those interleaves a *domain* check
/// (the 3-vertex ring minimum; the per-MOC `max_cells` budget) inside the
/// index-order loop, which is what makes their errors lowest-index across both
/// kinds of failure — a shared helper would have to take that as a callback.
/// This op has no such per-group pre-check (see the module's error posture), so
/// it is the degenerate case, not a duplicate of a callback-shaped abstraction
/// nothing else would use.
fn validate_ragged(n_values: usize, offsets: &[i64]) -> Result<usize, String> {
    if offsets.is_empty() {
        return Err("offsets must have at least one element".to_string());
    }
    if offsets[0] != 0 {
        return Err(format!("offsets must start at 0, got {}", offsets[0]));
    }
    let n_groups = offsets.len() - 1;
    for i in 0..n_groups {
        let (s, e) = (offsets[i], offsets[i + 1]);
        if e < s {
            return Err(format!(
                "group {i}: offsets must be monotonically non-decreasing ({e} < {s})"
            ));
        }
        if e as usize > n_values {
            return Err(format!(
                "group {i}: offset {e} exceeds value array length {n_values}"
            ));
        }
    }
    // Exact coverage: the last offset must land on the end of the value array.
    // Checked after the per-group pass so an out-of-bounds group is still
    // reported by index (the lowest-index rule).
    if offsets[n_groups] as usize != n_values {
        return Err(format!(
            "offsets must end at the value count: offsets[{n_groups}] is {} but \
             values has {n_values} entries",
            offsets[n_groups]
        ));
    }
    Ok(n_groups)
}

/// Run one item's kernel, turning a panic into an item-named error.
///
/// `label` is the noun the message leads with (`group` / `word`), so the
/// message reads in the caller's own vocabulary.
fn run_item<T, F>(label: &str, i: usize, kernel: F) -> Result<T, String>
where
    F: FnOnce() -> Result<T, String>,
{
    match catch_unwind(AssertUnwindSafe(kernel)) {
        Ok(r) => r.map_err(|e| format!("{label} {i}: {e}")),
        Err(e) => Err(format!(
            "{label} {i}: {}",
            crate::panic_msg(e, "batch kernel panicked")
        )),
    }
}

/// Reduce many independent groups of words to their deepest common ancestors.
///
/// Plural *ancestors*: one ancestor word per input group (many→many), the
/// segmented-reduce batch sibling of [`super::common_ancestor`].  Group `i` is
/// `values[offsets[i]..offsets[i + 1]]` (arrow list layout, the same one
/// [`crate::coverage::batch`] emits), and the result is dense — one `u64` per
/// group, `offsets.len() - 1` of them — because the reduction is many→one per
/// group, so no output offsets are needed.
///
/// Each output word is bit-identical to [`super::common_ancestor`] on that
/// group alone, including its single-element case (a group of one word returns
/// that word verbatim, kind preserved).
///
/// # Errors
/// The lowest-index offending group, named in the message: a layout problem
/// (see [`validate_ragged`]) or a group the scalar reduction refuses — empty,
/// holding an undecodable word, or spanning more than one base cell.
pub fn common_ancestors(values: &[u64], offsets: &[i64]) -> Result<Vec<u64>, String> {
    let n_groups = validate_ragged(values.len(), offsets)?;
    let mut out = Vec::with_capacity(n_groups);
    // Chunk the parallel pass so only one chunk's outcomes are resident, and
    // drain each chunk (in index order) before the next one is built.
    for base in (0..n_groups).step_by(CHUNK) {
        let end = (base + CHUNK).min(n_groups);
        let words: Vec<Result<u64, String>> = (base..end)
            .into_par_iter()
            .map(|i| {
                let (s, e) = (offsets[i] as usize, offsets[i + 1] as usize);
                run_item("group", i, || {
                    common_ancestor(&values[s..e]).map_err(|e| e.to_string())
                })
            })
            .collect();
        for word in words {
            out.push(word?);
        }
    }
    Ok(out)
}

/// A batch of children: a dense `(n_parents, width)` block, row-major.
#[derive(Debug)]
pub struct Children {
    /// The children of every parent, row-major: parent `i`'s children are
    /// `values[i * width..(i + 1) * width]`.
    pub values: Vec<u64>,
    /// `4**(order - parent_order)`, the children each parent yields.
    pub width: usize,
}

/// Validate the parent words against the target `order`, returning the row width.
///
/// One index-order pass doing all three checks per word — decodability, then
/// "not finer than the target", then agreement with word 0's order — so the
/// error names the genuinely lowest-index offender across all three, rather
/// than whichever check happens to run as its own earlier pass.
///
/// The shared-order requirement is what makes the result dense: rows of
/// different widths would be a ragged layout, which is the shape this op exists
/// to avoid.  It is also what every consumer already assumes — moczarr's
/// `np.stack([generate_morton_children(int(w), level) for w in words])`
/// (`dggs.py:310`) raises on rows of unequal width.
fn validate_words(words: &[u64], order: u8) -> Result<usize, String> {
    if order > MAX_ORDER {
        return Err(format!(
            "Order must be between 0 and {MAX_ORDER}, got {order}"
        ));
    }
    let mut parent_order: Option<u8> = None;
    for (i, &w) in words.iter().enumerate() {
        let (o, _) = to_nested(w).ok_or_else(|| {
            format!("word {i}: {w} is not a decodable morton word (empty sentinel or bad prefix)")
        })?;
        if o > order {
            return Err(format!(
                "word {i}: is at order {o}, finer than the requested order {order}; \
                 children_of only refines (use clip2order to coarsen)"
            ));
        }
        match parent_order {
            None => parent_order = Some(o),
            Some(p) if p != o => {
                return Err(format!(
                    "word {i}: is at order {o} but word 0 is at order {p}; children_of \
                     returns a dense (n, 4**d) block, so every parent must sit at one order"
                ))
            }
            Some(_) => {}
        }
    }
    // An empty batch has no parent order to derive the width from; 1 (== 4**0)
    // is the only width that is not a lie about a block with no rows.
    let depth = parent_order.map_or(0, |p| order - p);
    Ok(1usize << (2 * depth as u32))
}

/// Generate every parent's children at `order`, as a dense `(n, 4**d)` block.
///
/// The batch sibling of the Python-level `generate_morton_children`, whose
/// wrapper coerces its input to a *single* parent.  All parents must sit at one
/// order `p <= order`; `d = order - p`, so each yields exactly `4**d` children
/// and the result is a fixed-width matrix rather than a ragged list.
///
/// Row `i` is bit-identical to the scalar generator on `words[i]` alone,
/// including the `d == 0` case, where the parent comes back **verbatim** — that
/// is what preserves a [`super::Kind::Point`] word, which round-tripping
/// through [`super::from_nested`] would re-pack as an area cell.
///
/// # Errors
/// The lowest-index offending word, named in the message: undecodable, finer
/// than `order`, or at a different order from word 0.  Also an `order` past
/// [`MAX_ORDER`], a result whose element count overflows `usize`, or a result
/// the allocator refuses (see the module's allocation posture).
pub fn children_of(words: &[u64], order: u8) -> Result<Children, String> {
    let width = validate_words(words, order)?;
    let total = words
        .len()
        .checked_mul(width)
        .ok_or_else(|| format!("{} parents x {width} children each overflows", words.len()))?;
    // Allocated once at the exact final size (no growth realloc, so no
    // old-plus-new transient in the peak) and **fallibly**: `vec![0u64; total]`
    // goes through Rust's infallible allocator, whose failure path is
    // `handle_alloc_error` -> `abort()`, which kills the interpreter with no
    // catchable Python exception.  `try_reserve_exact` hands the failure back as
    // an `Err(String)` the binding turns into `ValueError`, matching the
    // catchable `MemoryError` the scalar loop this replaces raises.
    let mut out: Vec<u64> = Vec::new();
    out.try_reserve_exact(total).map_err(|_| {
        format!(
            "{} parents x {width} children each needs {} bytes; allocation failed \
             (refine fewer parents, or to a coarser order)",
            words.len(),
            total.saturating_mul(std::mem::size_of::<u64>()),
        )
    })?;
    out.resize(total, 0);

    for base in (0..words.len()).step_by(CHUNK) {
        let end = (base + CHUNK).min(words.len());
        let block = &mut out[base * width..end * width];
        let outcomes: Vec<Result<(), String>> = block
            .par_chunks_mut(width)
            .enumerate()
            .map(|(k, row)| {
                run_item("word", base + k, || {
                    children_row(words[base + k], order, row);
                    Ok(())
                })
            })
            .collect();
        for outcome in outcomes {
            outcome?;
        }
    }
    Ok(Children { values: out, width })
}

/// Write one parent's children at `order` into `row` (length `4**d`).
///
/// In NESTED space a cell's descendants at `order` are the contiguous block
/// `nested * 4**d + [0, 4**d)`, so the row is a single ascending run packed
/// back through [`super::from_nested`].
fn children_row(word: u64, order: u8, row: &mut [u64]) {
    let (o, nested) = to_nested(word).expect("validate_words accepted this word");
    let depth = order - o;
    if depth == 0 {
        // Parity with the scalar generator, which returns the parent verbatim
        // at `target_order == parent_order`.  Re-packing through `from_nested`
        // would answer with the order-29 *area* cell for a `Kind::Point` word.
        row[0] = word;
        return;
    }
    let first = nested << (2 * depth as u32);
    for (k, slot) in row.iter_mut().enumerate() {
        *slot = from_nested(first + k as u64, order);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decimal_morton::{encode, encode_point, kind_of, Kind};

    /// One area word in base `base` at `order`, tuples all `t`.
    fn word(base: u8, order: u8, t: u8) -> u64 {
        encode(base, &vec![t; order as usize], order)
    }

    // -- common_ancestors ---------------------------------------------------

    #[test]
    fn ancestors_match_the_scalar_per_group() {
        // Group 0: the four order-5 children of an order-4 cell.  Group 1: a
        // lone word.  Group 2: two cells sharing only their base cell.
        let parent = word(3, 4, 1);
        let (o4, n4) = to_nested(parent).unwrap();
        let kids: Vec<u64> = (0..4).map(|k| from_nested((n4 << 2) + k, o4 + 1)).collect();
        let mut values = kids.clone();
        values.push(word(7, 6, 2));
        values.push(word(9, 3, 0));
        values.push(word(9, 3, 3));
        let offsets = vec![0, 4, 5, 7];

        let out = common_ancestors(&values, &offsets).unwrap();
        assert_eq!(out.len(), 3);
        for i in 0..3 {
            let (s, e) = (offsets[i] as usize, offsets[i + 1] as usize);
            assert_eq!(out[i], common_ancestor(&values[s..e]).unwrap(), "group {i}");
        }
        assert_eq!(out[0], parent);
        // A group of one word comes back verbatim.
        assert_eq!(out[1], values[4]);
    }

    #[test]
    fn ancestors_span_more_than_one_chunk() {
        // 2x CHUNK + 1 groups, so the chunk seam and the trailing partial chunk
        // are both exercised.
        let n = 2 * CHUNK + 1;
        let values: Vec<u64> = (0..n as u64)
            .map(|k| from_nested(k % (12 << 12), 6))
            .collect();
        let offsets: Vec<i64> = (0..=n as i64).collect();
        let out = common_ancestors(&values, &offsets).unwrap();
        assert_eq!(out.len(), n);
        for i in 0..n {
            assert_eq!(out[i], values[i], "singleton group {i} returns itself");
        }
    }

    #[test]
    fn ancestors_empty_batch() {
        assert!(common_ancestors(&[], &[0]).unwrap().is_empty());
    }

    #[test]
    fn ancestors_errors_name_lowest_index_group() {
        let values = vec![word(0, 4, 1), word(5, 4, 1), word(0, 4, 2)];
        // An empty group has no ancestor, and is named.
        let err = common_ancestors(&values, &[0, 1, 1, 3]).unwrap_err();
        assert!(err.starts_with("group 1:"), "{err}");
        // Mixed base cells: group 0 offends first even though group 1 is empty,
        // which is the whole reason emptiness is not hoisted into the pre-pass.
        let err = common_ancestors(&values, &[0, 2, 2, 3]).unwrap_err();
        assert!(err.starts_with("group 0:"), "{err}");
        assert!(err.contains("multiple base cells"), "{err}");
        // Layout problems name the group too.
        let err = common_ancestors(&values, &[0, 3, 1]).unwrap_err();
        assert!(err.starts_with("group 1:"), "{err}");
        let err = common_ancestors(&values, &[0, 1, 99]).unwrap_err();
        assert!(err.starts_with("group 1:"), "{err}");
    }

    #[test]
    fn ancestors_offsets_must_exactly_cover_the_values() {
        let values = vec![word(0, 4, 1), word(0, 4, 2)];
        let err = common_ancestors(&values, &[1, 2]).unwrap_err();
        assert!(err.contains("must start at 0"), "{err}");
        let err = common_ancestors(&values, &[0, 1]).unwrap_err();
        assert!(err.contains("must end at the value count"), "{err}");
        assert!(common_ancestors(&[], &[]).is_err());
    }

    #[test]
    fn ancestors_lowest_index_holds_across_chunks() {
        // Offenders at the start, in the middle, at a chunk seam, and at the
        // end: each call must name the lowest-index one, not whichever rayon
        // finished first.
        let n = 3 * CHUNK;
        // Every group is a pair of base-cell-0 words; an offender's second word
        // comes from base cell 1, which has no common ancestor with the first.
        let offenders = [0usize, 7, CHUNK - 1, CHUNK, 2 * CHUNK + 5, n - 1];
        let other_base = from_nested((1 << 12) + 1, 6);
        let offsets: Vec<i64> = (0..=n as i64).map(|i| 2 * i).collect();
        let base_values: Vec<u64> = (0..2 * n as u64)
            .map(|k| from_nested(k % (1 << 12), 6))
            .collect();

        // Heal the offenders one at a time, lowest first: each call must name
        // the lowest *surviving* offender, across chunk seams and the tail.
        for healed in 0..offenders.len() {
            let mut values = base_values.clone();
            for (rank, &off) in offenders.iter().enumerate() {
                if rank >= healed {
                    values[2 * off + 1] = other_base;
                }
            }
            let err = common_ancestors(&values, &offsets).unwrap_err();
            let expect = offenders[healed];
            assert!(err.starts_with(&format!("group {expect}:")), "{err}");
            assert!(err.contains("multiple base cells"), "{err}");
        }
        // With every offender healed the batch goes through.
        assert_eq!(common_ancestors(&base_values, &offsets).unwrap().len(), n);
    }

    #[test]
    fn ancestors_kernel_panic_is_caught_and_named() {
        // No validated input reaches a panic in the reduction, so the capture
        // half of the lowest-index guarantee is driven directly.
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let err = run_item::<u64, _>("group", 9, || panic!("kernel exploded")).unwrap_err();
        std::panic::set_hook(hook);
        assert!(err.starts_with("group 9:"), "{err}");
        assert!(err.contains("kernel exploded"), "{err}");
    }

    // -- children_of --------------------------------------------------------

    /// The scalar reference: one parent's children at `order`.
    fn scalar_children(parent: u64, order: u8) -> Vec<u64> {
        let (o, nested) = to_nested(parent).unwrap();
        if o == order {
            return vec![parent];
        }
        let d = order - o;
        (0..(1u64 << (2 * d)))
            .map(|k| from_nested((nested << (2 * d)) + k, order))
            .collect()
    }

    #[test]
    fn children_match_the_scalar_per_word() {
        let words: Vec<u64> = (0..12).map(|b| word(b, 5, b % 4)).collect();
        let out = children_of(&words, 8).unwrap();
        assert_eq!(out.width, 64);
        assert_eq!(out.values.len(), 12 * 64);
        for (i, &w) in words.iter().enumerate() {
            let row = &out.values[i * out.width..(i + 1) * out.width];
            assert_eq!(row, &scalar_children(w, 8)[..], "word {i}");
        }
    }

    #[test]
    fn children_at_zero_depth_return_the_parent_verbatim() {
        // Area words round-trip either way; a point word only survives the
        // verbatim path, which is what the scalar promises.
        let area = word(4, 9, 2);
        let point = encode_point(4, &[2u8; MAX_ORDER as usize]);
        assert!(matches!(kind_of(point), Kind::Point));
        let out = children_of(&[area], 9).unwrap();
        assert_eq!((out.width, out.values.as_slice()), (1, &[area][..]));
        let out = children_of(&[point], MAX_ORDER).unwrap();
        assert_eq!((out.width, out.values.as_slice()), (1, &[point][..]));
    }

    #[test]
    fn children_order_zero_parents_and_max_order_target() {
        // Order-0 parents (whole base cells) refine like any other order.
        let bases: Vec<u64> = (0..12).map(|b| from_nested(b, 0)).collect();
        let out = children_of(&bases, 2).unwrap();
        assert_eq!(out.width, 16);
        for (i, &b) in bases.iter().enumerate() {
            assert_eq!(
                &out.values[i * 16..(i + 1) * 16],
                &scalar_children(b, 2)[..]
            );
        }
        // The max-order target at d == 1, so the block stays small.
        let parents = [word(0, MAX_ORDER - 1, 1), word(6, MAX_ORDER - 1, 3)];
        let out = children_of(&parents, MAX_ORDER).unwrap();
        assert_eq!(out.width, 4);
        for (i, &p) in parents.iter().enumerate() {
            assert_eq!(
                &out.values[i * 4..(i + 1) * 4],
                &scalar_children(p, MAX_ORDER)[..]
            );
        }
    }

    #[test]
    fn children_empty_and_single() {
        let out = children_of(&[], 6).unwrap();
        assert!(out.values.is_empty());
        assert_eq!(out.width, 1, "no parent order to derive a width from");
        let w = word(2, 3, 1);
        let out = children_of(&[w], 5).unwrap();
        assert_eq!(out.values, scalar_children(w, 5));
    }

    #[test]
    fn children_span_more_than_one_chunk() {
        let n = 2 * CHUNK + 1;
        let words: Vec<u64> = (0..n as u64)
            .map(|k| from_nested(k % (12 << 12), 6))
            .collect();
        let out = children_of(&words, 8).unwrap();
        assert_eq!(out.width, 16);
        for i in [0, 1, CHUNK - 1, CHUNK, CHUNK + 1, n - 1] {
            let row = &out.values[i * 16..(i + 1) * 16];
            assert_eq!(row, &scalar_children(words[i], 8)[..], "word {i}");
        }
    }

    #[test]
    fn children_errors_name_lowest_index_word() {
        let mut words: Vec<u64> = (0..4).map(|b| word(b, 5, 1)).collect();
        // Finer than the target.
        words.push(word(4, 9, 1));
        let err = children_of(&words, 7).unwrap_err();
        assert!(err.starts_with("word 4:"), "{err}");
        assert!(err.contains("finer than the requested order 7"), "{err}");
        // Coarser but disagreeing with word 0's order -> ragged rows, refused.
        words[4] = word(4, 6, 1);
        let err = children_of(&words, 8).unwrap_err();
        assert!(err.starts_with("word 4:"), "{err}");
        assert!(err.contains("word 0 is at order 5"), "{err}");
        // The empty sentinel is undecodable and named.
        words[2] = 0;
        let err = children_of(&words, 8).unwrap_err();
        assert!(err.starts_with("word 2:"), "{err}");
        assert!(err.contains("not a decodable morton word"), "{err}");
        // Word 0 itself finer than the target is named as index 0.
        let err = children_of(&[word(0, 9, 1), word(0, 5, 1)], 6).unwrap_err();
        assert!(err.starts_with("word 0:"), "{err}");
    }

    #[test]
    fn children_oversized_result_errors_instead_of_aborting() {
        // 12 order-0 parents at order 25 is 96 PiB.  `vec![0u64; total]` sent
        // this through Rust's infallible allocator, which aborts the process;
        // the fallible path names the byte count and returns.
        let bases: Vec<u64> = (0..12).map(|b| from_nested(b, 0)).collect();
        let err = children_of(&bases, 25).unwrap_err();
        assert!(err.contains("allocation failed"), "{err}");
        assert!(err.contains("108086391056891904 bytes"), "{err}");
        // The `checked_mul` guard still owns the >= 2**64 *element* case, which
        // is a different message and sits far above any real allocation.
        let err = children_of(&[from_nested(0, 0); 64], MAX_ORDER).unwrap_err();
        assert!(err.contains("children each overflows"), "{err}");
    }

    #[test]
    fn children_bad_order_rejected() {
        let err = children_of(&[word(0, 3, 1)], MAX_ORDER + 1).unwrap_err();
        assert!(err.contains("Order must be between 0 and 29"), "{err}");
    }

    #[test]
    fn children_lowest_index_holds_across_chunks() {
        // An offender at a chunk seam and one late in the batch: the pre-pass
        // scans in index order, so the earlier is always what surfaces.
        let n = 3 * CHUNK;
        let mut words: Vec<u64> = (0..n).map(|_| word(0, 4, 1)).collect();
        words[2 * CHUNK + 3] = word(0, 9, 1);
        words[CHUNK] = word(0, 9, 1);
        let err = children_of(&words, 6).unwrap_err();
        assert!(err.starts_with(&format!("word {CHUNK}:")), "{err}");
        words[CHUNK] = word(0, 4, 1);
        let err = children_of(&words, 6).unwrap_err();
        assert!(
            err.starts_with(&format!("word {}:", 2 * CHUNK + 3)),
            "{err}"
        );
    }

    #[test]
    fn children_deterministic_across_runs() {
        let words: Vec<u64> = (0..1000u64).map(|k| from_nested(k, 5)).collect();
        let a = children_of(&words, 8).unwrap();
        let b = children_of(&words, 8).unwrap();
        assert_eq!(a.values, b.values);
        assert_eq!(a.width, b.width);
    }
}

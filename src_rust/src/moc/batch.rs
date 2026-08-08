//! Batch MOC densify: one Python↔Rust crossing, rayon across MOCs (issue #156).
//!
//! The scalar [`super::to_order`] is a cheap kernel, so a Python loop over it
//! pays mostly boundary cost — argument conversion, allocation, result
//! wrapping — once per MOC, which is what dominates wall time when the loop
//! runs over a whole catalog (~556k granule covers).  This module crosses the
//! boundary once, in the same ragged arrow list layout the batch coverer
//! ([`crate::coverage::batch`]) emits: MOC `i` is
//! `values[offsets[i]..offsets[i + 1]]`, one flat cell list per MOC out, in
//! the same layout.  Chaining the two therefore needs no marshalling — the
//! coverer's `(values, offsets)` pair is this module's input verbatim.
//!
//! # Memory posture
//!
//! MOCs are densified a chunk at a time and each chunk is copied into the
//! ragged output as it lands, so the whole batch's per-MOC flat lists never
//! coexist with the concatenated result — peak ≈ result + one chunk, against
//! the ~2.5x of a densify-everything-then-concatenate pass (issue #153's
//! review, where that change took the 556k-item peak from 2.50x to 1.34x of
//! the returned array).
//!
//! # Error posture
//!
//! Fail-fast with the offending MOC named: structural problems (bad offsets)
//! and the per-MOC `max_cells` budget are rejected in a serial pre-validation
//! pass — before anything is densified, so a refusal costs no allocation, as
//! in the scalar path — and kernel panics are caught per MOC.  In both regimes
//! the error reported is the **lowest-index** offending MOC: pre-validation
//! scans in index order, and the parallel pass materializes a whole chunk's
//! results, then walks them in index order (chunks themselves in index order),
//! before any is allowed to fail the call.  So the error is deterministic
//! regardless of rayon's schedule.

use rayon::prelude::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

use super::{to_order, to_order_count};

/// MOCs densified per parallel chunk.
///
/// Same value and same reason as [`crate::coverage::batch`]'s: the batch runs
/// chunk by chunk so only one chunk's per-MOC `Vec`s are ever resident, and
/// 2048 is large enough that the per-chunk fork-join is noise against the
/// densifying work.
const CHUNK: usize = 2048;

/// A batch densify: ragged `values` / `offsets` in arrow list layout.
#[derive(Debug)]
pub struct BatchOrders {
    /// All MOCs' flat cells at the target order, concatenated.
    pub values: Vec<u64>,
    /// Arrow list offsets into `values`; MOC `i` densifies to
    /// `values[offsets[i]..offsets[i + 1]]`.
    pub offsets: Vec<i64>,
}

/// Validate the ragged layout and the per-MOC budget, returning the MOC count.
///
/// Checks, in order: `order` range, a non-empty `offsets` starting at 0, then
/// per MOC (lowest index first) offset monotonicity and bounds and — when
/// `max_cells` is set — the densified-count estimate against the budget, and
/// finally the endpoint requirement `offsets[n_mocs] == values.len()`.  The
/// offsets must **exactly cover** the value array — both endpoints are checked
/// and each names which one failed — so stale or misaligned offsets that
/// happen to stay in bounds cannot silently densify a valid-looking wrong set.
///
/// The budget lives here, not in the parallel pass, for the same reason the
/// scalar guard is pre-emptive (issue #80): the estimate is an O(n) pass over
/// the input words with no flat allocation, so a refusal happens before any
/// memory is committed.  Per-MOC errors name the offending MOC, and are raised
/// ahead of the endpoint check so the lowest-index MOC is still what a caller
/// sees first.
fn validate_batch(
    values: &[u64],
    offsets: &[i64],
    order: u8,
    max_cells: Option<u64>,
) -> Result<usize, String> {
    if !(1..=29).contains(&order) {
        return Err("Order must be between 1 and 29".to_string());
    }
    if offsets.is_empty() {
        return Err("offsets must have at least one element".to_string());
    }
    if offsets[0] != 0 {
        return Err(format!("offsets must start at 0, got {}", offsets[0]));
    }
    let n_mocs = offsets.len() - 1;
    for i in 0..n_mocs {
        let (s, e) = (offsets[i], offsets[i + 1]);
        if e < s {
            return Err(format!(
                "moc {i}: offsets must be monotonically non-decreasing ({e} < {s})"
            ));
        }
        if e as usize > values.len() {
            return Err(format!(
                "moc {i}: offset {e} exceeds value array length {}",
                values.len()
            ));
        }
        if let Some(budget) = max_cells {
            let estimated = to_order_count(&values[s as usize..e as usize], order);
            if estimated > budget {
                return Err(format!(
                    "moc {i}: moc_to_order would densify to ~{estimated} cells at \
                     order {order}, exceeding max_cells={budget}. Pass a larger \
                     max_cells, or max_cells=None to proceed (risking OOM), or \
                     densify to a coarser order."
                ));
            }
        }
    }
    // Exact coverage: the last offset must land on the end of the value array.
    // Checked after the per-MOC pass so an out-of-bounds MOC is still reported
    // by index (the lowest-index rule).
    if offsets[n_mocs] as usize != values.len() {
        return Err(format!(
            "offsets must end at the value count: offsets[{n_mocs}] is {} but \
             values has {} entries",
            offsets[n_mocs],
            values.len()
        ));
    }
    Ok(n_mocs)
}

impl BatchOrders {
    /// An empty batch result sized for `n_mocs` MOCs.
    fn new(n_mocs: usize) -> Self {
        let mut offsets = Vec::with_capacity(n_mocs + 1);
        offsets.push(0i64);
        BatchOrders {
            values: Vec::new(),
            offsets,
        }
    }

    /// Append one chunk's flat cell lists, or surface the lowest-index error.
    ///
    /// `base` is the chunk's first MOC index.  The lists are consumed in index
    /// order — each is copied and then **dropped**, so a chunk's allocations
    /// are released as the copy walks it — which also makes the surfaced error
    /// the lowest-index failure, deterministic under any rayon schedule.
    fn extend_chunk(&mut self, flats: Vec<Result<Vec<u64>, String>>) -> Result<(), String> {
        for flat in flats {
            let flat = flat?;
            self.values.extend_from_slice(&flat);
            self.offsets.push(self.values.len() as i64);
        }
        Ok(())
    }

    /// Reserve for the MOCs still to come, from the flat lists seen so far.
    ///
    /// A `Vec` realloc holds the old and new buffers at once, so growing the
    /// output by pure doubling puts a 1.5x-of-result transient in the peak.
    /// Extrapolating the mean densified size (with a margin) makes late
    /// reallocs rare; over-reserving is cheap because untouched pages cost
    /// address space, not resident memory.
    fn reserve_estimate(&mut self, done: usize, remaining: usize) {
        if done == 0 || remaining == 0 {
            return;
        }
        // Mean flat list so far, scaled by the MOCs left, plus a 1/16 margin.
        // `reserve` is a no-op when the spare capacity already covers it.
        let est = (self.values.len() / done).saturating_mul(remaining) * 17 / 16;
        self.values.reserve(est);
    }
}

/// Run one MOC's densify, turning a panic into a MOC-named error.
///
/// Defensive: `validate_batch` screens the one input the kernel cannot take
/// (an out-of-range `order`), so no known input reaches the panic arm — which
/// is why the tests drive this with an injected panicking kernel instead,
/// keeping the capture-and-name mechanism itself covered against future kernel
/// changes.
fn run_moc<F>(i: usize, kernel: F) -> Result<Vec<u64>, String>
where
    F: FnOnce() -> Vec<u64>,
{
    catch_unwind(AssertUnwindSafe(kernel))
        .map_err(|e| format!("moc {i}: {}", crate::panic_msg(e, "moc densify panicked")))
}

/// Densify many independent (mixed-order) MOCs to a flat order in one call.
///
/// Plural *orders*: one flat cell list per input MOC (many→many), the ragged
/// batch sibling of [`super::to_order`].
///
/// MOC `i` is `values[offsets[i]..offsets[i + 1]]` (arrow list layout).  The
/// offsets must **exactly cover** the value array — `offsets[0] == 0` and
/// `offsets[n] == values.len()` — so a sliced arrow array is re-based by the
/// caller rather than passed through verbatim; anything else is an error
/// naming the failing endpoint.  Returns a [`BatchOrders`] in the same layout:
/// MOC `i`'s flat cells at `order` are `values[offsets[i]..offsets[i + 1]]`,
/// each identical (sorted unique) to what [`super::to_order`] returns for that
/// MOC alone.
///
/// `max_cells` is a **single shared** per-MOC budget, applied to each MOC
/// exactly as the scalar path applies it to its one input: the densified-count
/// estimate is taken per MOC, and a MOC over budget refuses the whole call.
///
/// # Errors
/// The lowest-index offending MOC, named in the message (see the module docs
/// for the determinism argument), or an out-of-range `order`.
pub fn mocs_to_orders(
    values: &[u64],
    offsets: &[i64],
    order: u8,
    max_cells: Option<u64>,
) -> Result<BatchOrders, String> {
    let n_mocs = validate_batch(values, offsets, order, max_cells)?;
    let mut out = BatchOrders::new(n_mocs);
    // Chunk the parallel pass so only one chunk's flat lists are resident, and
    // copy each chunk out (in index order) before the next one is built.
    for base in (0..n_mocs).step_by(CHUNK) {
        let end = (base + CHUNK).min(n_mocs);
        let flats: Vec<Result<Vec<u64>, String>> = (base..end)
            .into_par_iter()
            .map(|i| {
                let (s, e) = (offsets[i] as usize, offsets[i + 1] as usize);
                run_moc(i, || to_order(&values[s..e], order))
            })
            .collect();
        out.extend_chunk(flats)?;
        out.reserve_estimate(end, n_mocs - end);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morton::nested2mort;

    /// Three MOCs: one order-6 cell, four order-4 cells, an empty cover.
    ///
    /// At order 8 they densify to 16, 1024 and 0 cells — MOC 1 the larger, so a
    /// budget between the two lands on MOC 1 rather than trivially on MOC 0.
    fn ragged() -> (Vec<u64>, Vec<i64>) {
        let mut values = vec![nested2mort(600, 6)];
        values.extend((0..4).map(|k| nested2mort(600 + k, 4)));
        (values, vec![0, 1, 5, 5])
    }

    #[test]
    fn batch_matches_scalar_per_moc() {
        let (values, offsets) = ragged();
        let out = mocs_to_orders(&values, &offsets, 8, None).unwrap();
        assert_eq!(out.offsets.len(), 4);
        assert_eq!(out.offsets[0], 0);
        assert_eq!(*out.offsets.last().unwrap() as usize, out.values.len());
        for i in 0..3 {
            let (s, e) = (offsets[i] as usize, offsets[i + 1] as usize);
            let scalar = to_order(&values[s..e], 8);
            let got = &out.values[out.offsets[i] as usize..out.offsets[i + 1] as usize];
            assert_eq!(got, &scalar[..]);
        }
        // The empty MOC densifies to nothing, keeping its (empty) slot.
        assert_eq!(out.offsets[2], out.offsets[3]);
    }

    #[test]
    fn batch_spans_more_than_one_chunk() {
        // 2x CHUNK + 1 MOCs, so the chunk seam and the trailing partial chunk
        // are both exercised (offsets must stay contiguous across them).
        let n = 2 * CHUNK + 1;
        let values: Vec<u64> = (0..n as u64).map(|k| nested2mort(k, 6)).collect();
        let offsets: Vec<i64> = (0..=n as i64).collect();
        let out = mocs_to_orders(&values, &offsets, 7, None).unwrap();
        assert_eq!(out.offsets.len(), n + 1);
        for i in 0..n {
            let scalar = to_order(&values[i..i + 1], 7);
            let got = &out.values[out.offsets[i] as usize..out.offsets[i + 1] as usize];
            assert_eq!(got, &scalar[..]);
        }
    }

    #[test]
    fn empty_batch() {
        let out = mocs_to_orders(&[], &[0], 6, None).unwrap();
        assert!(out.values.is_empty());
        assert_eq!(out.offsets, vec![0]);
    }

    #[test]
    fn budget_refuses_lowest_index_moc() {
        let (values, offsets) = ragged();
        // A budget under MOC 0's 16 cells refuses at index 0.
        let err = mocs_to_orders(&values, &offsets, 8, Some(10)).unwrap_err();
        assert!(err.starts_with("moc 0:"), "{err}");
        assert!(err.contains("exceeding max_cells=10"), "{err}");
        // A budget between the two refuses MOC 1 — the lowest-index offender
        // now that MOC 0 fits.  The empty MOC 2 never offends.
        let err = mocs_to_orders(&values, &offsets, 8, Some(100)).unwrap_err();
        assert!(err.starts_with("moc 1:"), "{err}");
        assert!(err.contains("~1024 cells"), "{err}");
        // Above both, the call goes through.
        assert!(mocs_to_orders(&values, &offsets, 8, Some(1024)).is_ok());
    }

    #[test]
    fn budget_is_per_moc_not_batch_total() {
        let (values, offsets) = ragged();
        // 16 + 1024 = 1040 cells in total, past a 1024 budget, but no single
        // MOC exceeds it — the guard is per item, exactly as the scalar guard
        // applies to its one input.
        let out = mocs_to_orders(&values, &offsets, 8, Some(1024)).unwrap();
        assert_eq!(out.values.len(), 16 + 1024);
    }

    #[test]
    fn budget_refusal_allocates_nothing() {
        // The refusal comes out of the serial pre-pass, so an over-budget MOC
        // at the *end* of a batch still refuses before the earlier MOCs are
        // densified: the count estimate is all that ran.
        let (values, offsets) = ragged();
        let err = mocs_to_orders(&values, &offsets, 20, Some(1)).unwrap_err();
        assert!(err.starts_with("moc 0:"), "{err}");
    }

    #[test]
    fn offsets_must_exactly_cover_the_values() {
        let (values, offsets) = ragged();
        let err = mocs_to_orders(&values, &offsets[1..], 8, None).unwrap_err();
        assert!(err.contains("must start at 0"), "{err}");
        let err = mocs_to_orders(&values, &[0, 1], 8, None).unwrap_err();
        assert!(err.contains("must end at the value count"), "{err}");
        // The exactly-covering spelling of that first MOC is accepted.
        let out = mocs_to_orders(&values[..1], &[0, 1], 8, None).unwrap();
        assert_eq!(out.values, to_order(&values[..1], 8));
    }

    #[test]
    fn errors_name_lowest_index_moc() {
        let (values, _) = ragged();
        // Non-monotone offsets name the MOC that shrinks.
        let err = mocs_to_orders(&values, &[0, 5, 1, 5], 8, None).unwrap_err();
        assert!(err.starts_with("moc 1:"), "{err}");
        // Out-of-bounds end offset: the per-MOC bounds check must fire ahead of
        // the batch-level endpoint check, so the MOC is named.
        let err = mocs_to_orders(&values, &[0, 1, 99], 8, None).unwrap_err();
        assert!(err.starts_with("moc 1:"), "{err}");
    }

    #[test]
    fn kernel_panic_is_caught_and_named_by_lowest_index() {
        // No input reaches the kernel's shift asserts (validate_batch screens
        // the order range), so the panic-capture half of the lowest-index
        // guarantee is driven here with injected panicking kernels through the
        // same assembly path.
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {})); // keep the test output clean
        let flats = vec![
            run_moc(0, || vec![7u64]),
            run_moc(1, || panic!("kernel exploded")),
            run_moc(2, || panic!("also exploded")),
        ];
        // A chunk that does not start at MOC 0 still names the true index.
        let late = vec![run_moc(9, || panic!("boom"))];
        std::panic::set_hook(hook);

        assert!(flats[0].is_ok());
        let err = BatchOrders::new(3).extend_chunk(flats).unwrap_err();
        assert!(err.starts_with("moc 1:"), "{err}");
        assert!(err.contains("kernel exploded"), "{err}");
        let err = BatchOrders::new(10).extend_chunk(late).unwrap_err();
        assert!(err.starts_with("moc 9:"), "{err}");
    }

    #[test]
    fn bad_order_and_layout_rejected() {
        let (values, offsets) = ragged();
        assert!(mocs_to_orders(&values, &offsets, 0, None).is_err());
        assert!(mocs_to_orders(&values, &offsets, 30, None).is_err());
        assert!(mocs_to_orders(&values, &[], 8, None).is_err());
        let err = mocs_to_orders(&values, &[-1, 1], 8, None).unwrap_err();
        assert!(err.contains("must start at 0"), "{err}");
    }
}

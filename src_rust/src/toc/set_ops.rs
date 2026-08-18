//! toc set algebra — the canonical cover form (issues #177 / #198).
//!
//! [`normalize`] collapses a toc word set into its **canonical cover**: the
//! unique sorted word set with the same decoded coverage — maximal merged
//! ranges plus the exact instants no range subsumes.  It is the equality and
//! construction basis for the `Toc` object (issue #198) and the root-word-set
//! normalizer for store sidecars (englacial/zagg#480).
//!
//! The rulings this implements are recorded on issue #177
//! (<https://github.com/espg/mortie/issues/177#issuecomment-5324957990>,
//! confirmed in
//! <https://github.com/espg/mortie/issues/177#issuecomment-5333585919>):
//!
//! - **Q1 — merge on decoded bounds; never bridge a decoded gap.**  Two range
//!   words coalesce iff their decoded half-open `[start, end)` envelopes
//!   overlap or abut *exactly* (`end_a == start_b` in exact ns — well-defined,
//!   both grids decode to exact integers).  A surviving decoded gap is a floor
//!   on the true gap (outward rounding only shrinks apparent gaps), so it is
//!   real information and is preserved however small.  Because decoded starts
//!   sit on the 2^31 ns grid and decoded ends on the 2^32 ns grid, a merged
//!   union's bounds are min/max of on-grid values — still on-grid, so the
//!   merge is **exact**: no rounding arm exists anywhere in this module.
//! - **Q2 — coverage semantics.**  A timestamp inside a covering range's
//!   decoded span adds no coverage and is absorbed; a timestamp no range
//!   subsumes survives **bit-identical** as an exact degenerate member.
//!   Equal timestamps deduplicate.  Timestamps never merge with each other
//!   or extend a range — re-encoding an instant into a range would round
//!   outward and *change* coverage, which normalize never does.
//!
//! Like [`super::merge`], everything here is total over arbitrary bit
//! patterns: junk words are garbage in, garbage out — deterministic, never a
//! panic.  The canonical form is normative for encoder-produced words and
//! pinned by the fixtures below.

use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;

use super::{decode, FLAG_MASK, LOW_MASK};

/// Re-splice an exact instant into its timestamp word, with no domain check.
///
/// Bit-identical to [`super::encode_timestamp`] on its domain; total (no
/// ceiling rejection) so that normalize can round-trip a junk "timestamp"
/// unchanged instead of erroring on input it merely passes through.
#[inline]
fn splice_timestamp(t_ns: u64) -> u64 {
    ((t_ns >> 31) << 32) | FLAG_MASK | (t_ns & LOW_MASK)
}

/// Re-encode a decoded on-grid envelope `[start_ns, end_ns)` as a range word.
///
/// The inverse of [`decode`] for range words: exact (no rounding) because
/// normalize only ever holds bounds that came off the grids — `start_ns` a
/// multiple of 2^31, `end_ns` a multiple of 2^32 — and min/max keep them so.
#[inline]
fn encode_envelope(start_ns: u64, end_ns: u64) -> u64 {
    ((start_ns >> 31) << 32) | (end_ns >> 32)
}

/// A cover in canonical parts (the shared intermediate of the set ops).
///
/// `ranges` are maximal merged decoded envelopes — exact half-open
/// `[start_ns, end_ns)` intervals, sorted, pairwise separated by a real
/// decoded gap.  `stamps` are the exact instants no range subsumes, sorted
/// and deduplicated.
pub(crate) struct Canonical {
    pub(crate) ranges: Vec<(u64, u64)>,
    pub(crate) stamps: Vec<u64>,
}

/// Decompose a word set into canonical parts.
///
/// One sorted-stream pass per part, the house pattern of
/// [`crate::moc::normalize`]: decode and split, sort ranges and sweep-merge
/// overlap-or-abut (Q1), then co-walk the sorted instants past the merged
/// ranges, dropping exactly the subsumed ones (Q2).
pub(crate) fn canonicalize(words: &[u64]) -> Canonical {
    let mut ranges: Vec<(u64, u64)> = Vec::new();
    let mut stamps: Vec<u64> = Vec::new();
    for &w in words {
        let (s, e, is_rng) = decode(w);
        if is_rng {
            ranges.push((s, e));
        } else {
            stamps.push(s);
        }
    }
    ranges.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
    for (s, e) in ranges {
        match merged.last_mut() {
            // Overlap or exact abutment: `s <= last_end` in exact ns.  A
            // strict `<` here would split abutting envelopes; a tolerance
            // would bridge a surviving decoded gap.  Both are wrong (Q1).
            Some((_, last_end)) if s <= *last_end => *last_end = (*last_end).max(e),
            _ => merged.push((s, e)),
        }
    }
    stamps.sort_unstable();
    stamps.dedup();
    let mut kept = Vec::with_capacity(stamps.len());
    let mut i = 0;
    for &t in &stamps {
        // `merged` is sorted by *start* (ends need not ascend — a junk word
        // decodes to an empty envelope).  A range ending at or before t can
        // subsume no later instant either, and `stamps` ascends, so the
        // cursor only ever moves forward; the check below is then a decision
        // for *all* remaining ranges, because their starts ascend.
        while i < merged.len() && merged[i].1 <= t {
            i += 1;
        }
        if i == merged.len() || t < merged[i].0 {
            kept.push(t);
        }
    }
    Canonical {
        ranges: merged,
        stamps: kept,
    }
}

/// Encode canonical parts back to the canonical word set (sorted u64s).
pub(crate) fn to_words(c: &Canonical) -> Vec<u64> {
    let mut out: Vec<u64> = c
        .ranges
        .iter()
        .map(|&(s, e)| encode_envelope(s, e))
        .chain(c.stamps.iter().map(|&t| splice_timestamp(t)))
        .collect();
    out.sort_unstable();
    out
}

/// Collapse a toc word set into its canonical cover form.
///
/// Sorted maximal merges: ranges coalesced iff their decoded envelopes
/// overlap or abut exactly (Q1), subsumed instants absorbed and free
/// instants kept bit-identical (Q2).  The output's decoded coverage equals
/// the input's **exactly** — order-independent and idempotent.  Empty in,
/// empty out.
///
/// Sortedness and duplicate-freeness are the canonical form over
/// **encoder-produced** words.  A junk word can decode to an empty
/// envelope (end below start), which subsumes nothing and does not
/// collapse even against a copy of itself, so junk can come back
/// duplicated — coverage is still exact and the output is still a
/// fixpoint, but junk in is junk out (pinned by
/// `arbitrary_bit_patterns_normalize_without_panicking`).
pub fn normalize(words: &[u64]) -> Vec<u64> {
    to_words(&canonicalize(words))
}

/// Canonicalize a toc word set: sorted maximal merges (issue #198 phase 1).
///
/// # Arguments
/// * `words` - Toc words (u64 NumPy array), any order, duplicates allowed
///
/// # Returns
/// The canonical cover as a sorted u64 NumPy array: maximal merged ranges
/// plus the exact instants no range subsumes.  Coverage-identical to the
/// input; see `mortie.toc.toc_normalize` for the direction table.
#[pyfunction]
pub fn rust_toc_normalize(py: Python<'_>, words: PyReadonlyArray1<u64>) -> PyResult<PyObject> {
    let w = words.to_vec()?;
    let out = py.allow_threads(|| normalize(&w));
    Ok(out.into_pyarray_bound(py).into_any().unbind())
}

// ── tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::{encode_range, encode_timestamp, is_range, Q_END_NS, Q_START_NS};
    use super::*;

    /// Deterministic PRNG (splitmix64) — no rand dependency.
    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn rand_time(state: &mut u64) -> u64 {
        splitmix64(state) % super::super::TOC_MAX_NS
    }

    /// A random valid word, timestamps and ranges mixed (short ranges too,
    /// so absorptions and near-misses both occur).
    fn rand_word(state: &mut u64) -> u64 {
        match splitmix64(state) % 4 {
            0 | 1 => encode_timestamp(rand_time(state)).unwrap(),
            2 => {
                let a = rand_time(state);
                let b = a + splitmix64(state) % (8 * Q_END_NS);
                encode_range(a, b.min(super::super::TOC_MAX_NS - 1)).unwrap()
            }
            _ => {
                let (x, y) = (rand_time(state), rand_time(state));
                encode_range(x.min(y), x.max(y)).unwrap()
            }
        }
    }

    /// Reference coverage membership over raw (un-normalized) words.
    fn covered(words: &[u64], t: u64) -> bool {
        words.iter().any(|&w| {
            let (s, e, is_rng) = decode(w);
            if is_rng {
                s <= t && t < e
            } else {
                s == t
            }
        })
    }

    // ── golden fixtures (normative — the canonical form persists) ────────

    #[test]
    fn golden_issue_177_absorption_example() {
        // espg's Q2 example (issue #177 decision record): t1, t2 inside a
        // covering range R absorb; t3 months later survives bit-identical;
        // the gap between R and t3 is preserved.
        let r = encode_range(100 * Q_END_NS, 200 * Q_END_NS).unwrap();
        let (rs, re, _) = decode(r);
        let t1 = encode_timestamp(rs + 7).unwrap(); // inside R's decoded span
        let t2 = encode_timestamp(re - 1).unwrap(); // last covered instant
        let t3 = encode_timestamp(5000 * Q_END_NS + 123).unwrap(); // far later
        let got = normalize(&[t1, r, t2, t3]);
        assert_eq!(got, vec![r, t3]);
    }

    #[test]
    fn golden_abutting_envelopes_merge_and_a_one_quantum_gap_survives() {
        // r1's decoded end (2^32 grid) meets r2's decoded start (2^31 grid)
        // exactly: end code 8 -> 8 * 2^32 = start code 16 * 2^31.
        let r1 = encode_range(3 * Q_START_NS, 8 * Q_END_NS - 5).unwrap();
        let r2 = encode_range(16 * Q_START_NS + 1, 20 * Q_END_NS - 5).unwrap();
        assert_eq!(decode(r1).1, decode(r2).0, "envelopes abut exactly");
        assert_eq!(
            normalize(&[r2, r1]),
            vec![encode_range(3 * Q_START_NS, 20 * Q_END_NS - 5).unwrap()]
        );
        // One start quantum later the decoded gap is 2^31 ns — the smallest
        // gap the grids can express — and it is never bridged.
        let r3 = encode_range(17 * Q_START_NS + 1, 20 * Q_END_NS - 5).unwrap();
        assert_eq!(normalize(&[r3, r1]), vec![r1, r3]);
    }

    // ── Q1: range merging ────────────────────────────────────────────────

    #[test]
    fn overlapping_and_nested_ranges_coalesce() {
        let a = encode_range(10 * Q_END_NS, 40 * Q_END_NS).unwrap();
        let b = encode_range(30 * Q_END_NS, 60 * Q_END_NS).unwrap();
        let nested = encode_range(12 * Q_END_NS, 20 * Q_END_NS).unwrap();
        let (s, _, _) = decode(a);
        let (_, e, _) = decode(b);
        let got = normalize(&[b, nested, a]);
        assert_eq!(got.len(), 1);
        assert_eq!(decode(got[0]), (s, e, true));
    }

    #[test]
    fn merged_bounds_stay_on_grid_no_rounding() {
        // The union of decoded envelopes re-encodes exactly: decode of the
        // output word gives back the min-start / max-end verbatim.
        let mut st = 0x198;
        for _ in 0..500 {
            let a = rand_word(&mut st);
            let b = rand_word(&mut st);
            for w in normalize(&[a, b]) {
                let (s, e, is_rng) = decode(w);
                if is_rng {
                    assert_eq!(decode(encode_envelope(s, e)), (s, e, true));
                }
            }
        }
    }

    // ── Q2: timestamps ───────────────────────────────────────────────────

    #[test]
    fn absorption_boundaries_are_exact() {
        let r = encode_range(50 * Q_END_NS, 70 * Q_END_NS).unwrap();
        let (s, e, _) = decode(r);
        let at_start = encode_timestamp(s).unwrap();
        let last_in = encode_timestamp(e - 1).unwrap();
        let at_end = encode_timestamp(e).unwrap(); // end exclusive: outside
        let before = encode_timestamp(s - 1).unwrap();
        assert_eq!(normalize(&[r, at_start]), vec![r]);
        assert_eq!(normalize(&[r, last_in]), vec![r]);
        assert_eq!(normalize(&[r, at_end]), vec![r, at_end]);
        // `before` sorts ahead of r: its start quantum precedes r's floor.
        assert_eq!(normalize(&[r, before]), vec![before, r]);
    }

    #[test]
    fn timestamps_never_merge_with_each_other() {
        // Adjacent instants stay two exact members — a merged range would
        // round outward and change coverage, which normalize never does.
        let t = 9 * Q_END_NS + 3;
        let a = encode_timestamp(t).unwrap();
        let b = encode_timestamp(t + 1).unwrap();
        assert_eq!(normalize(&[b, a]), vec![a, b]);
        // Equal instants deduplicate to the one word, still a timestamp.
        let got = normalize(&[a, a, a]);
        assert_eq!(got, vec![a]);
        assert!(!is_range(got[0]));
    }

    // ── canonical-form laws ──────────────────────────────────────────────

    #[test]
    fn empty_and_singletons_pass_through() {
        assert!(normalize(&[]).is_empty());
        let t = encode_timestamp(123_456_789).unwrap();
        let r = encode_range(5 * Q_END_NS, 6 * Q_END_NS).unwrap();
        assert_eq!(normalize(&[t]), vec![t]);
        assert_eq!(normalize(&[r]), vec![r]);
    }

    #[test]
    fn order_independent_and_idempotent() {
        let mut st = 0xCA11;
        for _ in 0..300 {
            let n = 1 + (splitmix64(&mut st) % 12) as usize;
            let mut words: Vec<u64> = (0..n).map(|_| rand_word(&mut st)).collect();
            let reference = normalize(&words);
            assert_eq!(normalize(&reference), reference, "idempotent");
            for _ in 0..4 {
                // Fisher-Yates on the deterministic PRNG.
                for i in (1..words.len()).rev() {
                    let j = (splitmix64(&mut st) % (i as u64 + 1)) as usize;
                    words.swap(i, j);
                }
                assert_eq!(normalize(&words), reference, "order-independent");
            }
        }
    }

    #[test]
    fn coverage_is_preserved_exactly() {
        // Membership at every decoded bound and its neighbors must agree
        // between the raw set and its canonical form — coverage-identical,
        // not conservatively-identical.
        let mut st = 0xC0DE;
        for _ in 0..200 {
            let n = 1 + (splitmix64(&mut st) % 10) as usize;
            let words: Vec<u64> = (0..n).map(|_| rand_word(&mut st)).collect();
            let canon = normalize(&words);
            let mut probes: Vec<u64> = Vec::new();
            for &w in words.iter().chain(canon.iter()) {
                let (s, e, _) = decode(w);
                probes.extend([s.saturating_sub(1), s, s + 1, e.saturating_sub(1), e, e + 1]);
            }
            probes.push(rand_time(&mut st));
            for t in probes {
                assert_eq!(covered(&words, t), covered(&canon, t), "probe {t}");
            }
        }
    }

    #[test]
    fn canonical_form_is_structurally_canonical() {
        let mut st = 0x5E7;
        for _ in 0..200 {
            let n = 1 + (splitmix64(&mut st) % 14) as usize;
            let words: Vec<u64> = (0..n).map(|_| rand_word(&mut st)).collect();
            let canon = normalize(&words);
            assert!(canon.windows(2).all(|p| p[0] < p[1]), "sorted, no dups");
            let parts = canonicalize(&canon);
            for pair in parts.ranges.windows(2) {
                let ((_, e0), (s1, _)) = (pair[0], pair[1]);
                assert!(e0 < s1, "ranges disjoint with a surviving gap");
            }
            for &t in &parts.stamps {
                assert!(
                    !parts.ranges.iter().any(|&(s, e)| s <= t && t < e),
                    "no stamp subsumed by a range"
                );
            }
        }
    }

    #[test]
    fn arbitrary_bit_patterns_normalize_without_panicking() {
        let mut st = 0xBADF00D;
        let junk: Vec<u64> = (0..256).map(|_| splitmix64(&mut st)).collect();
        let once = normalize(&junk);
        assert_eq!(normalize(&once), once, "junk output is still a fixpoint");
        // Scope of the canonical form: a junk "range" whose decoded end
        // falls below its decoded start has an empty envelope, so it
        // subsumes nothing and does not collapse against a copy of itself.
        // Duplicate-freeness holds for encoder-produced words only.
        let empty = (4u64 << 32) | 1;
        assert_eq!(decode(empty), (4 * Q_START_NS, Q_END_NS, true));
        assert_eq!(normalize(&[empty, empty]), vec![empty, empty]);
    }

    #[test]
    fn quantum_constants_are_the_decode_grids() {
        // The Q1 exactness argument leans on decoded starts being 2^31
        // multiples and ends 2^32 multiples; pin that against the constants.
        let mut st = 0x9;
        for _ in 0..200 {
            let w = rand_word(&mut st);
            let (s, e, is_rng) = decode(w);
            if is_rng {
                assert_eq!(s % Q_START_NS, 0);
                assert_eq!(e % Q_END_NS, 0);
            }
        }
    }
}

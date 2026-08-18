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
    let (_, kept) = split_stamps(&stamps, &merged);
    Canonical {
        ranges: merged,
        stamps: kept,
    }
}

/// Partition sorted instants by range membership: `(inside, outside)`.
///
/// The one stamp/range walk both set ops share: canonicalize keeps the
/// `outside` half (Q2 absorption) and [`intersect`] keeps the `inside` half
/// (an instant survives intersection with a cover that subsumes it).
/// `ranges` is sorted by *start* (ends need not ascend — a junk word decodes
/// to an empty envelope).  A range ending at or before t can subsume no
/// later instant either, and `stamps` ascends, so the cursor only ever
/// moves forward; the membership check is then a decision for *all*
/// remaining ranges, because their starts ascend.
fn split_stamps(stamps: &[u64], ranges: &[(u64, u64)]) -> (Vec<u64>, Vec<u64>) {
    let mut inside = Vec::new();
    let mut outside = Vec::new();
    let mut i = 0;
    for &t in stamps {
        while i < ranges.len() && ranges[i].1 <= t {
            i += 1;
        }
        if i < ranges.len() && ranges[i].0 <= t {
            inside.push(t);
        } else {
            outside.push(t);
        }
    }
    (inside, outside)
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

/// Intersect two canonical covers (the shared kernel of [`toc_and`]).
///
/// Ranges run a two-pointer sweep over the two sorted disjoint families:
/// each surviving piece is `[max(starts), min(ends))` — **exact by grid
/// closure** (Q3): the max of two 2^31-grid starts stays on the start grid
/// and the min of two 2^32-grid ends stays on the end grid, so every
/// intersection bound is exactly representable and no rounding arm exists.
/// Instants survive iff genuinely covered on both sides: a stamp inside the
/// other cover's ranges (the `inside` half of [`split_stamps`]) or present
/// as the identical stamp in both.
///
/// The output is canonical without a re-normalize, because the inputs are:
/// output ranges are sub-intervals of one side's disjoint non-abutting
/// ranges, separated by the other side's surviving gaps, so they are
/// disjoint and non-abutting; a surviving stamp lies outside its own
/// side's ranges (canonical), hence outside the output ranges those
/// contain; and the three stamp sources cannot overlap — a stamp equal on
/// both sides is inside neither side's ranges, so exactly one source
/// claims each instant and the sorted union is duplicate free.
fn intersect(a: &Canonical, b: &Canonical) -> Canonical {
    let mut ranges = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.ranges.len() && j < b.ranges.len() {
        let (sa, ea) = a.ranges[i];
        let (sb, eb) = b.ranges[j];
        let (s, e) = (sa.max(sb), ea.min(eb));
        if s < e {
            ranges.push((s, e));
        }
        // Advance whichever side's range ends first (both on a tie): the
        // finished range can intersect nothing later on the other side.
        if ea <= eb {
            i += 1;
        }
        if eb <= ea {
            j += 1;
        }
    }
    let mut stamps = split_stamps(&a.stamps, &b.ranges).0;
    stamps.extend(split_stamps(&b.stamps, &a.ranges).0);
    let (mut i, mut j) = (0, 0);
    while i < a.stamps.len() && j < b.stamps.len() {
        match a.stamps[i].cmp(&b.stamps[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                stamps.push(a.stamps[i]);
                i += 1;
                j += 1;
            }
        }
    }
    stamps.sort_unstable();
    Canonical { ranges, stamps }
}

/// Intersect two toc word sets: the canonical cover of the common coverage.
///
/// Both operands are canonicalized internally (the posture of
/// [`crate::moc::moc_intersects`]), so raw unsorted word sets are accepted.
/// Conservatism is preserved by construction — `A ⊇ X` and `B ⊇ Y` imply
/// `A ∩ B ⊇ X ∩ Y` — and the sweep itself is exact (see [`intersect`]);
/// per Q3, the difference/xor directions deliberately do not ship.
/// Total over junk like [`normalize`]: garbage in, garbage out, no panic.
pub fn toc_and(a: &[u64], b: &[u64]) -> Vec<u64> {
    to_words(&intersect(&canonicalize(a), &canonicalize(b)))
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

/// Intersect two toc word sets (issue #198 phase 2).
///
/// # Arguments
/// * `a` - Toc words (u64 NumPy array), any order, duplicates allowed
/// * `b` - Toc words (u64 NumPy array), the other operand
///
/// # Returns
/// The canonical cover of the common coverage as a sorted u64 NumPy array;
/// see `mortie.toc.toc_and` for the direction table.
#[pyfunction]
pub fn rust_toc_and(
    py: Python<'_>,
    a: PyReadonlyArray1<u64>,
    b: PyReadonlyArray1<u64>,
) -> PyResult<PyObject> {
    let wa = a.to_vec()?;
    let wb = b.to_vec()?;
    let out = py.allow_threads(|| toc_and(&wa, &wb));
    Ok(out.into_pyarray_bound(py).into_any().unbind())
}

// ── tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::{
        encode_range, encode_timestamp, is_range, Q_END_NS, Q_START_NS, TOC_MAX_NS,
    };
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

    #[test]
    fn golden_top_of_span_keeps_the_range_variant() {
        // At the top of the span the end field is completely full
        // (`end_ns >> 32 == LOW_MASK`): one more bit of end would land on
        // bit 31 and re-encode the range as a *timestamp*.  `toc.rs` guards
        // that edge on the encoder side (`top_of_span_is_rejected`), but
        // `set_ops` re-derives every word with its own unchecked
        // `encode_envelope`, so pin the maximal end here too.
        let r = encode_range(TOC_MAX_NS - 3 * Q_END_NS, TOC_MAX_NS - 1).unwrap();
        assert_eq!(r & LOW_MASK, LOW_MASK, "end field full");
        assert_eq!(decode(r).1, TOC_MAX_NS);
        // The last encodable instant is subsumed by r and absorbs; r comes
        // back through decode/re-encode bit-identical, still a range.
        let t = encode_timestamp(TOC_MAX_NS - 1).unwrap();
        assert_eq!(normalize(&[t, r]), vec![r]);
        // Merging an overlapping lower range moves the start and keeps the
        // maximal end — the arithmetic that would overflow into the flag.
        let lower = encode_range(TOC_MAX_NS - 9 * Q_END_NS, TOC_MAX_NS - 3 * Q_END_NS).unwrap();
        let got = normalize(&[t, r, lower]);
        assert_eq!(
            got,
            vec![encode_range(TOC_MAX_NS - 9 * Q_END_NS, TOC_MAX_NS - 1).unwrap()]
        );
        assert!(is_range(got[0]));
        assert_eq!(decode(got[0]).1, TOC_MAX_NS);
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
        // Read the expectation off the *inputs*: when two range words
        // coalesce, the merged word decodes to their min start and max end
        // verbatim — min/max of on-grid values, so no rounding arm.
        let mut st = 0x198;
        let mut merges = 0;
        for _ in 0..500 {
            let a = rand_word(&mut st);
            let b = rand_word(&mut st);
            let (sa, ea, a_rng) = decode(a);
            let (sb, eb, b_rng) = decode(b);
            let out = normalize(&[a, b]);
            // Only a two-range collapse has a min/max expectation: two
            // timestamps, or a timestamp/range pair, need not collapse.
            if a_rng && b_rng && out.len() == 1 {
                assert!(is_range(out[0]));
                assert_eq!(decode(out[0]), (sa.min(sb), ea.max(eb), true));
                merges += 1;
            }
        }
        assert!(merges > 0, "no two-range merge exercised");
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

    // ── toc_and (issue #198 phase 2) ─────────────────────────────────────

    #[test]
    fn golden_and_is_exact_by_grid_closure() {
        // Overlapping ranges: the intersection decodes to max(starts) /
        // min(ends) verbatim — Q3's closure, no rounding.
        let a = encode_range(10 * Q_END_NS, 40 * Q_END_NS).unwrap();
        let b = encode_range(30 * Q_END_NS + 5, 60 * Q_END_NS).unwrap();
        let (sa, ea, _) = decode(a);
        let (sb, eb, _) = decode(b);
        let got = toc_and(&[a], &[b]);
        assert_eq!(got.len(), 1);
        assert_eq!(decode(got[0]), (sa.max(sb), ea.min(eb), true));
        // The bounds land back on their grids exactly.
        assert_eq!(sa.max(sb) % Q_START_NS, 0);
        assert_eq!(ea.min(eb) % Q_END_NS, 0);
    }

    #[test]
    fn golden_disjoint_and_abutting_intersect_to_nothing() {
        let a = encode_range(3 * Q_START_NS, 8 * Q_END_NS - 5).unwrap();
        // Abutting envelopes (decoded end == decoded start) share no
        // instant: both are half-open, so the intersection is empty.
        let abutting = encode_range(16 * Q_START_NS + 1, 20 * Q_END_NS - 5).unwrap();
        assert_eq!(decode(a).1, decode(abutting).0);
        assert!(toc_and(&[a], &[abutting]).is_empty());
        let far = encode_range(100 * Q_END_NS, 200 * Q_END_NS).unwrap();
        assert!(toc_and(&[a], &[far]).is_empty());
    }

    #[test]
    fn nested_range_intersects_to_the_inner() {
        let outer = encode_range(10 * Q_END_NS, 60 * Q_END_NS).unwrap();
        let inner = encode_range(20 * Q_END_NS, 30 * Q_END_NS).unwrap();
        assert_eq!(toc_and(&[outer], &[inner]), vec![inner]);
    }

    #[test]
    fn one_range_against_many_fragments() {
        // A long a-range cut by three disjoint b-ranges: three pieces out,
        // each an exact pairwise intersection, gaps preserved.
        let a = encode_range(0, 100 * Q_END_NS).unwrap();
        let bs: Vec<u64> = [10u64, 40, 70]
            .iter()
            .map(|&k| encode_range(k * Q_END_NS, (k + 5) * Q_END_NS).unwrap())
            .collect();
        assert_eq!(toc_and(&[a], &bs), normalize(&bs));
    }

    #[test]
    fn and_timestamp_survival_is_exact() {
        let r = encode_range(50 * Q_END_NS, 70 * Q_END_NS).unwrap();
        let (s, e, _) = decode(r);
        let inside = encode_timestamp(e - 1).unwrap();
        let at_end = encode_timestamp(e).unwrap();
        let at_start = encode_timestamp(s).unwrap();
        // A stamp inside the other cover's range survives bit-identical …
        assert_eq!(toc_and(&[inside], &[r]), vec![inside]);
        assert_eq!(toc_and(&[r], &[inside]), vec![inside]);
        assert_eq!(toc_and(&[at_start], &[r]), vec![at_start]);
        // … the exclusive envelope end is outside …
        assert!(toc_and(&[at_end], &[r]).is_empty());
        // … identical stamps intersect to themselves, distinct ones to
        // nothing (an instant has no extent to share).
        assert_eq!(toc_and(&[inside], &[inside]), vec![inside]);
        assert!(toc_and(&[inside], &[at_end]).is_empty());
    }

    #[test]
    fn and_accepts_raw_word_sets() {
        // Operands are canonicalized internally: unsorted, duplicated,
        // absorbable words give the same answer as their canonical forms.
        let r1 = encode_range(10 * Q_END_NS, 30 * Q_END_NS).unwrap();
        let r2 = encode_range(25 * Q_END_NS, 50 * Q_END_NS).unwrap();
        let t = encode_timestamp(28 * Q_END_NS).unwrap(); // absorbed by r1|r2
        let q = encode_range(20 * Q_END_NS, 40 * Q_END_NS).unwrap();
        let raw = vec![t, r2, r1, r2];
        assert_eq!(toc_and(&raw, &[q]), toc_and(&normalize(&raw), &[q]));
        // q's decoded envelope sits wholly inside raw's merged coverage, so
        // the intersection is q itself, bit-identical.
        assert_eq!(toc_and(&raw, &[q]), vec![q]);
    }

    #[test]
    fn and_laws_identity_commutativity_empty() {
        let mut st = 0xA17D;
        for _ in 0..300 {
            let n = 1 + (splitmix64(&mut st) % 10) as usize;
            let m = 1 + (splitmix64(&mut st) % 10) as usize;
            let a: Vec<u64> = (0..n).map(|_| rand_word(&mut st)).collect();
            let b: Vec<u64> = (0..m).map(|_| rand_word(&mut st)).collect();
            assert_eq!(toc_and(&a, &a), normalize(&a), "A ∩ A = normalize(A)");
            assert_eq!(toc_and(&a, &b), toc_and(&b, &a), "commutative");
            assert!(toc_and(&a, &[]).is_empty());
            assert!(toc_and(&[], &b).is_empty());
        }
    }

    #[test]
    fn and_membership_matches_both_sides() {
        // The defining property: an instant is covered by A ∩ B iff it is
        // covered by A and by B — probed at every decoded bound ± 1.
        let mut st = 0xB007;
        for _ in 0..200 {
            let n = 1 + (splitmix64(&mut st) % 8) as usize;
            let m = 1 + (splitmix64(&mut st) % 8) as usize;
            let a: Vec<u64> = (0..n).map(|_| rand_word(&mut st)).collect();
            let b: Vec<u64> = (0..m).map(|_| rand_word(&mut st)).collect();
            let both = toc_and(&a, &b);
            let mut probes: Vec<u64> = Vec::new();
            for &w in a.iter().chain(b.iter()).chain(both.iter()) {
                let (s, e, _) = decode(w);
                probes.extend([s.saturating_sub(1), s, s + 1, e.saturating_sub(1), e, e + 1]);
            }
            probes.push(rand_time(&mut st));
            for t in probes {
                assert_eq!(
                    covered(&both, t),
                    covered(&a, t) && covered(&b, t),
                    "probe {t}"
                );
            }
        }
    }

    #[test]
    fn and_output_is_canonical() {
        // No re-normalize runs on the way out; the sweep must land in
        // canonical form on its own (idempotence pins it).
        let mut st = 0xCAB;
        for _ in 0..200 {
            let n = 1 + (splitmix64(&mut st) % 10) as usize;
            let m = 1 + (splitmix64(&mut st) % 10) as usize;
            let a: Vec<u64> = (0..n).map(|_| rand_word(&mut st)).collect();
            let b: Vec<u64> = (0..m).map(|_| rand_word(&mut st)).collect();
            let both = toc_and(&a, &b);
            assert_eq!(normalize(&both), both, "already canonical");
            assert!(both.windows(2).all(|p| p[0] < p[1]), "sorted, no dups");
        }
    }

    #[test]
    fn arbitrary_bit_patterns_intersect_without_panicking() {
        let mut st = 0xDEAD;
        let junk_a: Vec<u64> = (0..128).map(|_| splitmix64(&mut st)).collect();
        let junk_b: Vec<u64> = (0..128).map(|_| splitmix64(&mut st)).collect();
        let got = toc_and(&junk_a, &junk_b);
        assert_eq!(toc_and(&junk_a, &junk_b), got, "deterministic");
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

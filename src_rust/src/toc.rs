//! toc word — temporal order coverage (issue #175).
//!
//! One `u64`, a tagged union of an exact nanosecond **timestamp** and a
//! quantized, conservative **time range**.  All internal times are u64
//! nanoseconds since **1850-01-01T00:00:00** on a continuous, leap-free,
//! GPS-aligned timescale (leap seconds exist only at the UTC boundary in
//! `mortie/toc.py`).  This is *not* an IVOA T-MOC — see the design record.
//!
//! Layout (decision ledger on issue #175; design rationale on
//! englacial/zagg#410):
//!
//! ```text
//! [start: 32 bits, 2^31 ns units][flag: 1 bit][low: 31 bits]
//!   flag = 1 → timestamp: the word is t_ns with the flag bit spliced in at
//!              position 31 — bits 63..32 = t_ns >> 31, bits 30..0 =
//!              t_ns & (2^31 - 1).  Monotone in t_ns.
//!   flag = 0 → range: bits 63..32 = start code s (floor, 2^31 ns units),
//!              bits 30..0 = end code e (31 bits, 2^32 ns units); the
//!              encoded envelope is half-open [s * 2^31, e * 2^32) ns.
//! ```
//!
//! Decisions pinned here (normative — words persist on disk):
//! - 1 ns base quantum, 32/31 split
//!   (<https://github.com/espg/mortie/issues/175#issuecomment-5230004941>);
//! - flag at bit 31, polarity 1 = timestamp, and the `toc` name
//!   (<https://github.com/espg/mortie/issues/175#issuecomment-5230049902>);
//! - outward rounding with a strictly-greater end ceiling, the min/max
//!   semilattice merge, and the 1850 GPS-aligned epoch
//!   (<https://github.com/englacial/zagg/issues/410>).
//!
//! Sort property (normative): unsigned u64 order = order by conservative
//! encoded start; within a tied start quantum, ranges (flag 0) sort before
//! timestamps (flag 1) — correct, since a range's floored start is <= any
//! timestamp inside that quantum; within timestamps, exact ns; within
//! ranges, shorter-first by end code.
//!
//! The bit layout, constants, merge results, and sort order are normative
//! and fixture-pinned below; how they are computed (parallelism, chunking,
//! error text) is not.

/// Discriminator bit position: 1 = timestamp, 0 = range.
pub const FLAG_BIT: u32 = 31;
/// Mask of the flag bit.
pub const FLAG_MASK: u64 = 1 << FLAG_BIT;
/// Mask of the 31-bit low field (fine ns-offset or end code).
pub const LOW_MASK: u64 = FLAG_MASK - 1;
/// Start quantum: 2^31 ns (~2.15 s).  A range's start code floors to this.
pub const Q_START_NS: u64 = 1 << 31;
/// End quantum: 2^32 ns (~4.29 s).  A range's end code ceils to this.
pub const Q_END_NS: u64 = 1 << 32;
/// Exclusive ceiling on internal times: 2^63 - 2^32 ns past the epoch
/// (~4 s short of year 2142).  The bound is the end code's: the ceiling
/// `e = (t >> 32) + 1` must fit 31 bits, i.e. `t < (2^31 - 1) * 2^32`.
/// Applied to *both* encoders so that every encodable word is mergeable —
/// a timestamp in the last 2^32 ns would encode fine but its merge
/// envelope would overflow the end field, so it is rejected up front
/// rather than wrapping silently.
pub const TOC_MAX_NS: u64 = ((1u64 << 31) - 1) << 32;

/// Encode an exact instant (internal ns) as a timestamp word.
///
/// The word is `t_ns` with a 1 flag bit spliced in at position 31, so the
/// unsigned word order over timestamps is exactly the ns order.
pub fn encode_timestamp(t_ns: u64) -> Result<u64, String> {
    if t_ns >= TOC_MAX_NS {
        return Err(format!(
            "timestamp {t_ns} ns is at or beyond the toc span ceiling \
             (2^63 - 2^32 ns past 1850-01-01, ~year 2142)"
        ));
    }
    Ok(((t_ns >> 31) << 32) | FLAG_MASK | (t_ns & LOW_MASK))
}

/// Encode a real closed interval `[start_ns, end_ns]` as a range word.
///
/// Outward rounding with a strictly-greater end ceiling: `s = start >> 31`
/// (floor) and `e = (end >> 32) + 1` — uniformly, *including* when `end`
/// sits exactly on the 2^32 grid — so the half-open envelope
/// `[s * 2^31, e * 2^32)` always properly contains `end`.
pub fn encode_range(start_ns: u64, end_ns: u64) -> Result<u64, String> {
    if start_ns > end_ns {
        return Err(format!(
            "range start {start_ns} ns is after its end {end_ns} ns"
        ));
    }
    if end_ns >= TOC_MAX_NS {
        return Err(format!(
            "range end {end_ns} ns is at or beyond the toc span ceiling \
             (2^63 - 2^32 ns past 1850-01-01, ~year 2142)"
        ));
    }
    Ok(((start_ns >> 31) << 32) | ((end_ns >> 32) + 1))
}

/// True when `word` is a range (flag 0), false for a timestamp (flag 1).
#[inline]
pub fn is_range(word: u64) -> bool {
    word & FLAG_MASK == 0
}

/// Decode a word to `(start_ns, end_ns, is_range)`.
///
/// A timestamp yields its exact instant twice, `(t, t, false)`.  A range
/// yields its half-open envelope bounds `[start_ns, end_ns)` — `end_ns` is
/// *exclusive*, and strictly greater than every instant the range covers.
/// Total over encoder-produced words; an arbitrary bit pattern decodes
/// without complaint (garbage in, garbage out).
#[inline]
pub fn decode(word: u64) -> (u64, u64, bool) {
    if is_range(word) {
        ((word >> 32) << 31, (word & LOW_MASK) << 32, true)
    } else {
        let t = ((word >> 32) << 31) | (word & LOW_MASK);
        (t, t, false)
    }
}

/// Conservative `(start_code, end_code)` of a word, for merging.
///
/// A range carries its codes verbatim.  A timestamp's envelope for merging
/// purposes is `s = t >> 31` (= `word >> 32`, the shared high field) and
/// `e = (t >> 32) + 1` (= `(word >> 33) + 1` — bit 31 of `t` is bit 0 of
/// the high field, so `t >> 32` is one more shift of it).
#[inline]
fn codes(word: u64) -> (u64, u64) {
    if is_range(word) {
        (word >> 32, word & LOW_MASK)
    } else {
        (word >> 32, (word >> 33) + 1)
    }
}

/// Semilattice join of two words: min of start codes, max of end codes.
///
/// **Bitwise-equal inputs return that word unchanged** — required for
/// idempotence, because merging two equal timestamps must NOT produce
/// their range envelope.  Any other pair decodes to conservative
/// envelopes and emits a range word.  Exactly associative, commutative,
/// and idempotent on the fixed epoch-anchored lattice, so the result is
/// bit-identical under any fold tree (pinned by the tests below).
#[inline]
pub fn merge(a: u64, b: u64) -> u64 {
    if a == b {
        return a;
    }
    let (sa, ea) = codes(a);
    let (sb, eb) = codes(b);
    (sa.min(sb) << 32) | ea.max(eb)
}

// ── tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
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
        splitmix64(state) % TOC_MAX_NS
    }

    /// A random valid word: timestamps, ranges, and grid-aligned ends mixed.
    fn rand_word(state: &mut u64) -> u64 {
        match splitmix64(state) % 4 {
            0 => encode_timestamp(rand_time(state)).unwrap(),
            1 => {
                // End snapped exactly onto the 2^32 grid (the strictly-greater
                // ceiling edge case must circulate through the law tests too).
                let b = (rand_time(state) >> 32) << 32;
                let a = b.saturating_sub(splitmix64(state) % Q_END_NS);
                encode_range(a, b).unwrap()
            }
            _ => {
                let (x, y) = (rand_time(state), rand_time(state));
                encode_range(x.min(y), x.max(y)).unwrap()
            }
        }
    }

    // ── golden fixtures (normative; words persist on disk) ──────────────

    #[test]
    fn golden_epoch_timestamp() {
        // The epoch instant 1850-01-01T00:00:00 (t = 0 ns): only the flag
        // bit is set.
        assert_eq!(encode_timestamp(0).unwrap(), 0x8000_0000);
    }

    #[test]
    fn golden_2018_timestamp() {
        // 2018-01-01T00:00:00 UTC as internal ns, derived:
        //   proleptic-Gregorian days 1850-01-01 -> 2018-01-01 = 61,362
        //   (168 years * 365 + 41 leap days: 1852..2016 has 42 multiples of
        //   4, minus 1900, a century non-leap year);
        //   naive ns = 61,362 * 86,400 * 10^9 = 5,301,590,400e9;
        //   GPS - UTC at 2018 = TAI - UTC - 19 = 37 - 19 = 18 s;
        //   t = 5,301,590,400e9 + 18e9 = 5,301,590,418,000,000,000 ns.
        // (Cross-check: GPS seconds since 1980-01-06 for that instant are
        // 1,198,800,018, the published GPS time of 2018-01-01 UTC, and
        // 47,486 days separate 1850-01-01 from 1980-01-06:
        // 4,102,790,400e9 + 1,198,800,018e9 = the same t.)
        let t: u64 = 5_301_590_418_000_000_000;
        let w = encode_timestamp(t).unwrap();
        assert_eq!(w, 0x9326_10CA_E981_3400);
        assert_eq!(w, 10_603_180_836_377_408_512);
        assert_eq!(decode(w), (t, t, false));
    }

    #[test]
    fn golden_range_straddling_quantum_boundary() {
        // [3*2^32 - 5e8, 3*2^32 + 5e8] straddles the 2^32 grid line at
        // 3*2^32: s = floor(a / 2^31) = 5, e = floor(b / 2^32) + 1 = 4.
        let a: u64 = 3 * (1 << 32) - 500_000_000;
        let b: u64 = 3 * (1 << 32) + 500_000_000;
        let w = encode_range(a, b).unwrap();
        assert_eq!(w, 0x5_0000_0004);
        assert_eq!(w, 21_474_836_484);
        let (s, e, is_rng) = decode(w);
        assert!(is_rng);
        assert_eq!((s, e), (5 << 31, 4u64 << 32));
        assert!(s <= a && b < e);
    }

    #[test]
    fn golden_range_end_exactly_on_grid() {
        // b = 7*2^32 sits exactly on the 2^32 grid; the ceiling is
        // strictly greater: e = (b >> 32) + 1 = 8, never 7 — the envelope
        // must properly contain b.
        let a: u64 = 7 * (1 << 32) - 1_000_000_000;
        let b: u64 = 7 * (1 << 32);
        let w = encode_range(a, b).unwrap();
        assert_eq!(w, (13 << 32) | 8);
        let (_, e, _) = decode(w);
        assert_eq!(e, 8u64 << 32);
        assert!(b < e, "end ceiling must be strictly greater than b");
    }

    // ── encoder domain ───────────────────────────────────────────────────

    #[test]
    fn top_of_span_is_rejected() {
        assert!(encode_timestamp(TOC_MAX_NS).is_err());
        assert!(encode_timestamp(u64::MAX).is_err());
        assert!(encode_range(0, TOC_MAX_NS).is_err());
        // The last valid instant fills the end field exactly.
        let w = encode_range(0, TOC_MAX_NS - 1).unwrap();
        assert_eq!(w & LOW_MASK, LOW_MASK);
        assert!(encode_timestamp(TOC_MAX_NS - 1).is_ok());
        // Inverted interval.
        assert!(encode_range(5, 3).is_err());
    }

    #[test]
    fn timestamp_roundtrip_and_monotonicity() {
        let mut st = 0x175;
        let mut prev_t = 0u64;
        let mut prev_w = 0u64;
        let mut times: Vec<u64> = (0..2000).map(|_| rand_time(&mut st)).collect();
        times.extend([0, 1, Q_START_NS - 1, Q_START_NS, Q_END_NS, TOC_MAX_NS - 1]);
        times.sort_unstable();
        for &t in &times {
            let w = encode_timestamp(t).unwrap();
            assert_eq!(decode(w), (t, t, false), "roundtrip exact");
            assert!(!is_range(w));
            if t > prev_t {
                assert!(w > prev_w, "word order must be monotone in t_ns");
            }
            (prev_t, prev_w) = (t, w);
        }
    }

    // ── conservatism and width ───────────────────────────────────────────

    #[test]
    fn envelope_contains_interval_and_is_bounded() {
        let mut st = 0xC0FFEE;
        for i in 0..4000 {
            let (x, y) = (rand_time(&mut st), rand_time(&mut st));
            let (a, mut b) = (x.min(y), x.max(y));
            if i % 8 == 0 {
                // Force the strictly-greater ceiling edge: b on the 2^32 grid.
                b = (b >> 32) << 32;
                if b < a {
                    continue;
                }
            }
            let (s, e, _) = decode(encode_range(a, b).unwrap());
            assert!(s <= a, "start floors");
            assert!(b < e, "end is strictly greater, half-open envelope");
            // Width bound: envelope <= real span + one quantum per end.
            assert!(e - s <= (b - a) + Q_START_NS + Q_END_NS);
        }
    }

    #[test]
    fn merge_envelope_contains_every_input_instant() {
        let mut st = 0xDECADE;
        for _ in 0..1000 {
            let n = 2 + (splitmix64(&mut st) % 6) as usize;
            let words: Vec<u64> = (0..n).map(|_| rand_word(&mut st)).collect();
            let folded = words.iter().copied().reduce(merge).unwrap();
            let (fs, fe, _) = decode(folded);
            for &w in &words {
                let (s, e, is_rng) = decode(w);
                assert!(fs <= s);
                // A timestamp's instant must land inside; a range's whole
                // envelope must land inside.
                if is_rng {
                    assert!(e <= fe);
                } else {
                    assert!(s < fe);
                }
            }
        }
    }

    // ── semilattice laws ─────────────────────────────────────────────────

    #[test]
    fn merge_is_idempotent_including_timestamps() {
        let mut st = 0x1DEA;
        for _ in 0..2000 {
            let w = rand_word(&mut st);
            assert_eq!(merge(w, w), w, "merge(w, w) must return w unchanged");
        }
        // The load-bearing case: two equal timestamps must NOT become
        // their range envelope.
        let t = encode_timestamp(123_456_789_012).unwrap();
        assert_eq!(merge(t, t), t);
        assert!(!is_range(merge(t, t)));
    }

    #[test]
    fn merge_is_commutative_and_associative() {
        let mut st = 0xABBA;
        for _ in 0..2000 {
            let (a, b, c) = (rand_word(&mut st), rand_word(&mut st), rand_word(&mut st));
            assert_eq!(merge(a, b), merge(b, a));
            assert_eq!(merge(merge(a, b), c), merge(a, merge(b, c)));
            // Mixed case: an equal-timestamp pair inside a triple.
            assert_eq!(merge(merge(a, a), c), merge(a, merge(a, c)));
        }
    }

    #[test]
    fn fold_tree_independence() {
        /// Fold `words` with a random binary tree shape.
        fn fold_random(words: &[u64], st: &mut u64) -> u64 {
            match words.len() {
                1 => words[0],
                n => {
                    let cut = 1 + (splitmix64(st) % (n as u64 - 1)) as usize;
                    merge(
                        fold_random(&words[..cut], st),
                        fold_random(&words[cut..], st),
                    )
                }
            }
        }
        let mut st = 0xF01D;
        for _ in 0..300 {
            let n = 2 + (splitmix64(&mut st) % 14) as usize;
            let mut words: Vec<u64> = (0..n).map(|_| rand_word(&mut st)).collect();
            // Plant a duplicated timestamp so equal-word pairs occur inside
            // larger reductions.
            let dup = encode_timestamp(rand_time(&mut st)).unwrap();
            words.push(dup);
            words.push(dup);
            let reference = words.iter().copied().reduce(merge).unwrap();
            for _ in 0..8 {
                assert_eq!(
                    fold_random(&words, &mut st),
                    reference,
                    "every fold tree must produce the identical u64"
                );
            }
        }
    }

    // ── sort property ────────────────────────────────────────────────────

    #[test]
    fn sort_order_is_conservative_start_order() {
        let mut st = 0x50FA;
        let mut words: Vec<u64> = (0..4000).map(|_| rand_word(&mut st)).collect();
        words.sort_unstable();
        let mut prev_start = 0u64;
        for &w in &words {
            let start = (w >> 32) << 31; // conservative encoded start, both kinds
            assert!(start >= prev_start, "u64 order must be start order");
            assert_eq!(decode(w).0 >> 31 << 31, start);
            prev_start = start;
        }
    }

    #[test]
    fn tied_quantum_tiebreaks() {
        // Within one start quantum q: ranges (flag 0) sort before
        // timestamps (flag 1); ranges shorter-first by end code;
        // timestamps in exact ns order.
        let q: u64 = 1_000_000; // start quantum index
        let base = q * Q_START_NS;
        let ts_lo = encode_timestamp(base + 7).unwrap();
        let ts_hi = encode_timestamp(base + Q_START_NS - 1).unwrap();
        let rng_short = encode_range(base + 100, base + 200).unwrap();
        let rng_long = encode_range(base + 100, base + 10 * Q_END_NS).unwrap();
        assert_eq!(ts_lo >> 32, q);
        assert_eq!(rng_short >> 32, q);
        assert!(rng_short < rng_long, "shorter range first");
        assert!(
            rng_long < ts_lo,
            "ranges before timestamps in a tied quantum"
        );
        assert!(ts_lo < ts_hi, "timestamps in exact ns order");
    }
}

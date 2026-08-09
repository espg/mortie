"""Tests for the toc word (temporal order coverage, issue #175).

The golden u64 fixtures here are normative: words persist on disk, so the
bit layout, constants, merge results, and sort order they pin must never
change.  The property tests mirror the cargo suite in src_rust/src/toc.rs
at array level.
"""

from functools import reduce

import numpy as np
import pytest

from mortie.toc import (
    GPS_EPOCH_NS,
    Q_END_NS,
    Q_START_NS,
    TOC_MAX_NS,
    from_datetime64,
    from_gps_ns,
    span2toc,
    time2toc,
    to_datetime64,
    to_gps_ns,
    toc2time,
    toc_contains,
    toc_is_range,
    toc_merge,
    toc_overlaps,
    toc_reduce,
)

FLAG = 1 << 31
LOW_MASK = FLAG - 1


def py_timestamp(t):
    """Reference splice: t_ns with a 1 flag bit inserted at position 31."""
    return ((t >> 31) << 32) | FLAG | (t & LOW_MASK)


def py_range(a, b):
    """Reference range encode: floored start, strictly-greater end ceiling."""
    return ((a >> 31) << 32) | ((b >> 32) + 1)


def rand_times(rng, n):
    """Random valid internal times as uint64."""
    return rng.integers(0, TOC_MAX_NS, size=n, dtype=np.uint64)


def rand_words(rng, n):
    """A mixed batch of valid timestamp and range words."""
    t = time2toc(rand_times(rng, n))
    x, y = rand_times(rng, n), rand_times(rng, n)
    r = span2toc(np.minimum(x, y), np.maximum(x, y))
    pick = rng.integers(0, 2, size=n).astype(bool)
    return np.where(pick, t, r)


# ── golden fixtures (normative) ─────────────────────────────────────────


def test_golden_epoch_timestamp():
    # 1850-01-01T00:00:00 (t = 0): only the flag bit set.
    assert time2toc(0) == 0x8000_0000


def test_golden_2018_timestamp():
    # 2018-01-01T00:00:00 UTC = 61,362 proleptic-Gregorian days of 86,400 s
    # past 1850-01-01, plus GPS-UTC = 18 s: 5,301,590,418e9 ns internal.
    t = 5_301_590_418_000_000_000
    assert time2toc(t) == 10_603_180_836_377_408_512
    assert toc2time(10_603_180_836_377_408_512) == (t, t)


def test_golden_range_fixtures():
    # Straddling the 2^32 grid line at 3*2^32: s = 5, e = 4.
    assert span2toc(3 * 2**32 - 500_000_000,
                    3 * 2**32 + 500_000_000) == 21_474_836_484
    # End exactly on the grid at 7*2^32: e = 8, strictly greater.
    w = span2toc(7 * 2**32 - 10**9, 7 * 2**32)
    assert w == (13 << 32) | 8
    _, end = toc2time(w)
    assert end == 8 * 2**32 > 7 * 2**32


# ── parity with the Rust scalars ────────────────────────────────────────


def test_encode_parity_with_reference_splice():
    rng = np.random.default_rng(175)
    times = rand_times(rng, 500)
    words = time2toc(times)
    assert words.dtype == np.uint64
    for t, w in zip(times.tolist(), words.tolist()):
        assert w == py_timestamp(t)
    x, y = rand_times(rng, 500), rand_times(rng, 500)
    a, b = np.minimum(x, y), np.maximum(x, y)
    words = span2toc(a, b)
    for aa, bb, w in zip(a.tolist(), b.tolist(), words.tolist()):
        assert w == py_range(aa, bb)


def test_timestamp_roundtrip_exact():
    rng = np.random.default_rng(17)
    times = rand_times(rng, 1000)
    starts, ends = toc2time(time2toc(times))
    np.testing.assert_array_equal(starts, times)
    np.testing.assert_array_equal(ends, times)
    assert not toc_is_range(time2toc(times)).any()


def test_envelope_conservatism_and_width():
    rng = np.random.default_rng(23)
    x, y = rand_times(rng, 1000), rand_times(rng, 1000)
    a, b = np.minimum(x, y), np.maximum(x, y)
    # Force some ends exactly onto the 2^32 grid (the strictly-greater
    # ceiling edge); keep the pairs ordered.
    b[::8] = (b[::8] >> np.uint64(32)) << np.uint64(32)
    a = np.minimum(a, b)
    starts, ends = toc2time(span2toc(a, b))
    assert (starts <= a).all(), "start floors"
    assert (b < ends).all(), "end strictly greater, half-open envelope"
    assert (ends - starts <= (b - a) + np.uint64(Q_START_NS + Q_END_NS)).all()
    assert toc_is_range(span2toc(a, b)).all()


# ── merge laws (mirroring the cargo property suite) ─────────────────────


def test_merge_idempotent_including_timestamps():
    rng = np.random.default_rng(29)
    w = rand_words(rng, 1000)
    np.testing.assert_array_equal(toc_merge(w, w), w)
    # The load-bearing case: equal timestamps must stay timestamps.
    t = time2toc(123_456_789_012)
    assert toc_merge(t, t) == t
    assert not toc_is_range(t)


def test_merge_commutative_associative():
    rng = np.random.default_rng(31)
    a, b, c = (rand_words(rng, 1000) for _ in range(3))
    np.testing.assert_array_equal(toc_merge(a, b), toc_merge(b, a))
    np.testing.assert_array_equal(toc_merge(toc_merge(a, b), c),
                                  toc_merge(a, toc_merge(b, c)))
    # Mixed case: an equal-timestamp pair inside a triple.
    np.testing.assert_array_equal(toc_merge(toc_merge(a, a), c),
                                  toc_merge(a, toc_merge(a, c)))


def test_reduce_matches_pairwise_fold_and_permutations():
    rng = np.random.default_rng(37)
    for n in (1, 2, 3, 7, 100):
        words = rand_words(rng, n)
        # Plant a duplicated timestamp inside the larger reductions.
        if n >= 3:
            words[1] = words[0] = time2toc(int(rand_times(rng, 1)[0]))
        expected = reduce(toc_merge, (int(w) for w in words))
        assert toc_reduce(words) == expected
        assert toc_reduce(words[::-1].copy()) == expected
        assert toc_reduce(rng.permutation(words)) == expected


def test_reduce_singleton_and_all_equal():
    t = time2toc(42)
    assert toc_reduce([int(t)]) == t
    assert toc_reduce(np.full(17, t, dtype=np.uint64)) == t


def test_reduce_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        toc_reduce(np.empty(0, dtype=np.uint64))


def test_merged_envelope_contains_inputs():
    rng = np.random.default_rng(41)
    words = rand_words(rng, 200)
    fs, fe = toc2time(toc_reduce(words))
    starts, ends = toc2time(words)
    is_rng = toc_is_range(words)
    assert (fs <= starts).all()
    assert (ends[is_rng] <= fe).all()
    assert (starts[~is_rng] < fe).all()


# ── sort property ───────────────────────────────────────────────────────


def test_sort_order_is_conservative_start_order():
    rng = np.random.default_rng(43)
    words = np.sort(rand_words(rng, 2000))
    starts, _ = toc2time(words)
    quantized = (starts >> np.uint64(31)) << np.uint64(31)
    assert (np.diff(quantized.astype(np.int64)) >= 0).all()


def test_tied_quantum_tiebreaks():
    base = 1_000_000 * Q_START_NS
    ts_lo = time2toc(base + 7)
    ts_hi = time2toc(base + Q_START_NS - 1)
    rng_short = span2toc(base + 100, base + 200)
    rng_long = span2toc(base + 100, base + 10 * Q_END_NS)
    assert rng_short < rng_long < ts_lo < ts_hi


# ── window predicates ───────────────────────────────────────────────────


def test_timestamp_window_semantics_exact():
    t = 10**15
    w = time2toc(t)
    # Half-open window: inclusive start, exclusive end.
    assert toc_overlaps(w, t, t + 1)
    assert toc_contains(w, t, t + 1)
    assert not toc_overlaps(w, t + 1, t + 2)
    assert not toc_overlaps(w, t - 2, t)  # exclusive end excludes t
    assert not toc_contains(w, t - 2, t)


def test_range_window_uses_conservative_bounds():
    a, b = 10 * Q_END_NS + 100, 10 * Q_END_NS + 200
    w = span2toc(a, b)
    start, end = toc2time(w)
    # Envelope grazing: a window touching only the envelope slop still
    # reports an overlap (documented over-report, never an under-report).
    assert toc_overlaps(w, int(start), int(start) + 1)
    assert not toc_overlaps(w, 0, int(start))
    assert not toc_overlaps(w, int(end), int(end) + 1)
    # Containment is envelope containment: the real [a, b] fits a snug
    # window but the envelope does not (documented under-report) ...
    assert not toc_contains(w, a, b + 1)
    # ... and the envelope itself does.
    assert toc_contains(w, int(start), int(end))


def test_overlaps_never_under_reports_contains_never_over_reports():
    rng = np.random.default_rng(47)
    x, y = rand_times(rng, 2000), rand_times(rng, 2000)
    a, b = np.minimum(x, y), np.maximum(x, y)
    words = span2toc(a, b)
    q0, q1 = TOC_MAX_NS // 4, TOC_MAX_NS // 2
    hits = toc_overlaps(words, q0, q1)
    true_overlap = (a < q1) & (b >= q0)
    assert (hits | ~true_overlap).all(), "a real overlap must be reported"
    inside = toc_contains(words, q0, q1)
    truly_inside = (a >= q0) & (b < q1)
    assert (~inside | truly_inside).all(), "contains must not over-report"


def test_inverted_window_raises():
    with pytest.raises(ValueError, match="window"):
        toc_overlaps(time2toc(0), 10, 5)


# ── validation, shapes, and edges ───────────────────────────────────────


def test_empty_arrays():
    empty = np.empty(0, dtype=np.uint64)
    assert time2toc(empty).shape == (0,)
    assert span2toc(empty, empty).shape == (0,)
    starts, ends = toc2time(empty)
    assert starts.shape == ends.shape == (0,)
    assert toc_merge(empty, empty).shape == (0,)
    assert toc_is_range(empty).shape == (0,)
    assert toc_overlaps(empty, 0, 1).shape == (0,)
    assert toc_contains(empty, 0, 1).shape == (0,)


def test_scalar_in_scalar_out():
    assert isinstance(time2toc(0), int)
    assert isinstance(span2toc(0, 5), int)
    assert isinstance(toc2time(time2toc(0))[0], int)
    assert isinstance(toc_merge(time2toc(0), time2toc(1)), int)
    assert isinstance(toc_is_range(time2toc(0)), bool)
    assert isinstance(toc_overlaps(time2toc(0), 0, 1), bool)


def test_broadcasting():
    times = np.arange(4, dtype=np.uint64) * np.uint64(10**12)
    words = span2toc(0, times)  # scalar start broadcast over array ends
    assert words.shape == (4,)
    merged = toc_merge(int(words[0]), words)
    assert merged.shape == (4,)


def test_dtype_and_domain_validation():
    with pytest.raises(ValueError, match="integer"):
        time2toc(np.array([1.5]))
    with pytest.raises(ValueError, match="non-negative"):
        time2toc(np.array([-1]))
    with pytest.raises(ValueError, match="ceiling"):
        time2toc(TOC_MAX_NS)
    with pytest.raises(ValueError, match="ceiling"):
        span2toc(0, TOC_MAX_NS)
    with pytest.raises(ValueError, match="after its end"):
        span2toc(5, 3)
    # The last valid instant is fine for both encoders.
    assert time2toc(TOC_MAX_NS - 1) > 0
    assert span2toc(0, TOC_MAX_NS - 1) > 0


# ── timescale boundary (phase 3) ────────────────────────────────────────


def test_gps_epoch_constant_against_date_arithmetic():
    # The 1850 -> GPS-epoch constant is a plain proleptic-Gregorian day
    # count (the internal scale is leap-free and ticks with GPS).
    from datetime import date
    days = (date(1980, 1, 6) - date(1850, 1, 1)).days
    assert days == 47_486
    assert GPS_EPOCH_NS == days * 86_400 * 10**9


def test_gps_epoch_zero_offset_identity():
    # At the GPS epoch TAI - UTC = 19, so GPS - UTC = 0: the UTC and GPS
    # entry points agree exactly there.
    assert from_datetime64("1980-01-06") == GPS_EPOCH_NS
    assert from_gps_ns(0) == GPS_EPOCH_NS
    assert to_gps_ns(GPS_EPOCH_NS) == 0
    assert to_datetime64(GPS_EPOCH_NS) == np.datetime64("1980-01-06", "ns")


def test_epoch_identity_and_pre1972_convention():
    # Zero offset before 1972 pins the epoch identity exactly.
    assert from_datetime64("1850-01-01") == 0
    assert to_datetime64(0) == np.datetime64("1850-01-01", "ns")
    with pytest.raises(ValueError, match="before the 1850"):
        from_datetime64("1849-12-31T23:59:59")


def test_gps_conversion_exact_roundtrip():
    rng = np.random.default_rng(53)
    gps = rng.integers(0, TOC_MAX_NS - GPS_EPOCH_NS, size=500,
                       dtype=np.uint64)
    np.testing.assert_array_equal(to_gps_ns(from_gps_ns(gps)), gps)
    with pytest.raises(ValueError, match="before the GPS epoch"):
        to_gps_ns(GPS_EPOCH_NS - 1)


def test_2018_utc_conversion_matches_golden():
    # The same instant the golden timestamp fixture pins: 61,362 days of
    # 86,400 s past 1850, +18 s GPS-UTC.
    t = from_datetime64("2018-01-01")
    assert t == 5_301_590_418_000_000_000
    assert to_gps_ns(t) == 1_198_800_018 * 10**9  # published GPS seconds


def test_leap_second_boundary_pinned():
    # A leap second was inserted at the end of 2016-12-31: one UTC second
    # across the boundary spans two internal seconds.
    before = from_datetime64("2016-12-31T23:59:59")
    after = from_datetime64("2017-01-01T00:00:00")
    assert after - before == 2 * 10**9
    # An instant inside the inserted leap second renders into the
    # following UTC second (datetime64 cannot express 23:59:60).
    inside = before + int(1.5e9)
    assert to_datetime64(inside) == np.datetime64(
        "2017-01-01T00:00:00.500000000")


def test_datetime64_roundtrip_exact_modern_era():
    rng = np.random.default_rng(59)
    lo = np.datetime64("1972-01-01", "ns").astype(np.int64)
    hi = np.datetime64("2100-01-01", "ns").astype(np.int64)
    naive = rng.integers(lo, hi, size=2000, dtype=np.int64)
    dt = naive.astype("datetime64[ns]")
    np.testing.assert_array_equal(to_datetime64(from_datetime64(dt)), dt)
    # ... and through a toc timestamp word.
    t = from_datetime64(dt)
    starts, _ = toc2time(time2toc(t))
    np.testing.assert_array_equal(to_datetime64(starts), dt)


def test_datetime64_scalar_forms():
    t = from_datetime64(np.datetime64("2020-05-17T12:34:56.789"))
    assert isinstance(t, int)
    assert isinstance(to_datetime64(t), np.datetime64)
    with pytest.raises(ValueError, match="ceiling"):
        from_datetime64("2150-01-01")

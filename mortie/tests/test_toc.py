"""Tests for the toc word (temporal order coverage, issue #175).

The golden u64 fixtures here are normative: words persist on disk, so the
bit layout, constants, merge results, and sort order they pin must never
change.  The property tests mirror the cargo suite in src_rust/src/toc.rs
at array level.
"""

from functools import reduce

import numpy as np
import pytest

from mortie._toc import (
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
    tocs_reduce,
)

FLAG = 1 << 31
LOW_MASK = FLAG - 1

# The Rust batch steps through 2048 groups at a time, so offenders are placed
# either side of that seam rather than at arbitrary indices.
CHUNK = 2048


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
    # 2018-01-01T00:00:00 UTC = 61,361 proleptic-Gregorian days of 86,400 s
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


# ── segmented reduce (issue #177) ───────────────────────────────────────


def py_codes(w):
    """Reference conservative (start_code, end_code) of a word."""
    if w & FLAG:                      # timestamp
        return w >> 32, (w >> 33) + 1
    return w >> 32, w & LOW_MASK      # range: codes verbatim


def py_merge(a, b):
    """Reference semilattice join: equal words unchanged, else min/max."""
    if a == b:
        return a
    (sa, ea), (sb, eb) = py_codes(a), py_codes(b)
    return (min(sa, sb) << 32) | max(ea, eb)


def ragged(groups):
    """Concatenate word groups into (words, offsets) arrow list layout."""
    words = np.concatenate([np.asarray(g, dtype=np.uint64) for g in groups]
                           or [np.empty(0, np.uint64)])
    offsets = np.zeros(len(groups) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(g) for g in groups])
    return words, offsets


def random_groups(rng, n, size=(1, 9)):
    """*n* groups of 1-8 mixed timestamp/range words."""
    sizes = rng.integers(*size, size=n)
    flat = rand_words(rng, int(sizes.sum()))
    cuts = np.cumsum(sizes)[:-1]
    return np.split(flat, cuts)


def test_tocs_reduce_parity_with_a_scalar_loop():
    """Group i is bit-identical to toc_reduce on that group alone.

    Swept past the 2048-group chunk seam so the parallel path is exercised,
    not just the first chunk.
    """
    rng = np.random.default_rng(177)
    groups = random_groups(rng, CHUNK + 137)
    got = tocs_reduce(*ragged(groups))
    assert got.dtype == np.uint64 and got.shape == (len(groups),)
    for i, g in enumerate(groups):
        assert int(got[i]) == toc_reduce(g), f"group {i}"


def test_tocs_reduce_property_against_a_python_reference():
    """Randomized groups match a pure-Python fold of the reference merge."""
    rng = np.random.default_rng(1770)
    groups = random_groups(rng, 400, size=(1, 20))
    got = tocs_reduce(*ragged(groups))
    for i, g in enumerate(groups):
        expected = reduce(py_merge, (int(w) for w in g))
        assert int(got[i]) == expected, f"group {i}"


def test_tocs_reduce_permutation_invariant_within_a_group():
    """The join is commutative and associative, so group order cannot matter."""
    rng = np.random.default_rng(1771)
    groups = random_groups(rng, 300, size=(2, 12))
    reference = tocs_reduce(*ragged(groups))
    shuffled = [rng.permutation(g) for g in groups]
    np.testing.assert_array_equal(tocs_reduce(*ragged(shuffled)), reference)
    np.testing.assert_array_equal(
        tocs_reduce(*ragged([g[::-1].copy() for g in groups])), reference)


def test_tocs_reduce_preserves_instants_per_group():
    """A group of bitwise-equal timestamps stays that timestamp.

    The load-bearing case of the merge, per group: it must not collapse to
    the pair's range envelope.  A group of one comes back verbatim.
    """
    t = time2toc(123_456_789_012)
    groups = [[t], [t] * 17, [t] * 3]
    got = tocs_reduce(*ragged(groups))
    assert (got == t).all()
    assert not toc_is_range(got).any()


def test_tocs_reduce_mixed_instant_and_range_groups():
    """Instant-only, range-only and mixed groups in one call, each parity-checked."""
    t0, t1 = time2toc(10 * Q_END_NS), time2toc(11 * Q_END_NS)
    r0 = span2toc(3 * Q_END_NS, 5 * Q_END_NS)
    r1 = span2toc(20 * Q_END_NS, 21 * Q_END_NS)
    groups = [[t0], [t0, t1], [r0, r1], [t0, r0], [r0, t1, r1, t0]]
    got = tocs_reduce(*ragged(groups))
    for i, g in enumerate(groups):
        assert int(got[i]) == toc_reduce(np.asarray(g, np.uint64)), f"group {i}"
    # Only the singleton keeps the timestamp flag; every unequal pair merges
    # to a range word.
    np.testing.assert_array_equal(toc_is_range(got),
                                  [False, True, True, True, True])


def test_tocs_reduce_empty_segment_is_a_catchable_named_error():
    """An empty group refuses, names itself, and survives ``except Exception``.

    The merge has no identity element, so many->one over no words has no
    answer -- :func:`toc_reduce`'s ruling, inherited.  Asserted through the
    handler shape a consumer actually writes, not only ``pytest.raises``: the
    PR #160 / issue #185 lesson is that a ``pyo3_runtime.PanicException``
    derives from ``BaseException`` and escapes even ``except Exception``.
    """
    words = time2toc(np.array([1, 2, 3], dtype=np.uint64))
    with pytest.raises(ValueError, match=r"group 1: .*empty segment"):
        tocs_reduce(words, [0, 1, 1, 3])
    caught = None
    try:
        tocs_reduce(words, [0, 1, 1, 3])
    except Exception as exc:      # must not escape as PanicException
        caught = exc
    assert isinstance(caught, ValueError)
    assert "group 1" in str(caught) and "identity element" in str(caught)


def test_tocs_reduce_arbitrary_bit_patterns_do_not_panic():
    """Junk words are garbage-in-garbage-out, never an uncatchable panic.

    Unlike a morton word there is no malformed toc word -- the merge only
    shifts and compares, so every bit pattern decodes (the module docstring's
    "garbage in, garbage out").  What the batch must still guarantee is the
    issue #185 posture: junk cannot take the process down, and the answer is
    the sequential in-group fold.

    Parity with :func:`toc_reduce` is deliberately *not* asserted here: the
    merge is associative over encoder-produced words only, so on junk the two
    fold trees may legitimately differ (pinned in
    :func:`test_tocs_reduce_junk_fold_is_tree_dependent`).  The oracle is the
    same one the cargo twin uses -- a sequential fold of the reference merge
    (``arbitrary_bit_patterns_fold_without_panicking``, src_rust/src/toc.rs).
    """
    rng = np.random.default_rng(1772)
    junk = rng.integers(0, 1 << 63, size=64, dtype=np.uint64) * np.uint64(2)
    junk[:4] = [0, 1, np.uint64((1 << 64) - 1), FLAG]
    offsets = np.arange(0, 65, 8, dtype=np.int64)
    got = tocs_reduce(junk, offsets)
    for i in range(8):
        expected = reduce(py_merge, (int(w) for w in junk[8 * i:8 * (i + 1)]))
        assert int(got[i]) == expected, f"group {i}"


def test_tocs_reduce_junk_fold_is_tree_dependent():
    """Out-of-domain words can merge onto the flag bit, so the tree matters.

    ``merge``'s associativity is a *valid-word* property (see its doc comment
    in src_rust/src/toc.rs): an out-of-domain "timestamp" of ``2**64 - 1``
    decodes to an end code of ``2**31``, and the merged word carries that
    **on** the timestamp flag, so it re-reads as a timestamp and the answer
    stops being fold-tree independent.  Junk still gets the two guarantees
    that are promised -- no panic, and a deterministic answer from each entry
    point -- but only the segmented form is pinned to the sequential fold.
    """
    junk = np.array([0, 1, 1, 1, 1, 1, 1, 2 ** 64 - 1], dtype=np.uint64)
    sequential = reduce(py_merge, (int(w) for w in junk))
    assert sequential & FLAG          # merged onto the timestamp flag
    for _ in range(3):
        assert int(tocs_reduce(junk, [0, junk.size])[0]) == sequential
    # toc_reduce splits this group its own way; whatever tree it picks, the
    # answer is stable -- it just need not be ``sequential``.
    assert toc_reduce(junk) == toc_reduce(junk)


def test_tocs_reduce_offsets_guards():
    """Layout failures are catchable ValueErrors naming the group or endpoint."""
    words = time2toc(np.array([1, 2, 3], dtype=np.uint64))
    with pytest.raises(ValueError, match=r"group 1: .*monotonically"):
        tocs_reduce(words, [0, 3, 1])
    with pytest.raises(ValueError, match=r"group 1: offset 99 exceeds"):
        tocs_reduce(words, [0, 1, 99])
    with pytest.raises(ValueError, match="must start at 0"):
        tocs_reduce(words, [1, 3])
    with pytest.raises(ValueError, match="must end at the word count"):
        tocs_reduce(words, [0, 2])
    with pytest.raises(ValueError, match="at least one element"):
        tocs_reduce(words, np.empty(0, dtype=np.int64))
    # Offsets are integer-typed by the same rule words are: a float array
    # would otherwise cast silently, truncating a boundary rather than saying so.
    with pytest.raises(ValueError, match="offsets must be integer-typed"):
        tocs_reduce(words, [0.0, 1.5, 3.0])
    with pytest.raises(ValueError, match="words must be integer-typed"):
        tocs_reduce(np.array([1.5, 2.5]), [0, 2])
    with pytest.raises(ValueError, match="words must be non-negative"):
        tocs_reduce(np.array([-1, 2]), [0, 2])


def test_tocs_reduce_uint64_offsets_out_of_int64_range():
    """A uint64 offset the int64 cast cannot hold is named, not wrapped.

    ``uint64`` offsets are the natural output of ``np.cumsum`` over unsigned
    counts, and anything at or above ``2**63`` wraps negative on the cast.
    That failed closed only by accident -- every wrapped value is negative, so
    the ``offsets[0] == 0`` pin tripped the monotonicity check -- with a
    message describing the wrapped copy rather than the offset passed in.
    """
    words = time2toc(np.array([1, 2, 3, 4], dtype=np.uint64))
    with pytest.raises(ValueError, match=r"must fit in int64, got 9223372036854775813"):
        tocs_reduce(words, np.array([0, 2 ** 63 + 5], dtype=np.uint64))
    # In-range uint64 offsets stay accepted -- this is a range check, not a
    # rejection of the dtype cumsum hands you.
    offsets = np.array([0, 2, 4], dtype=np.uint64)
    np.testing.assert_array_equal(
        tocs_reduce(words, offsets),
        [toc_reduce(words[:2]), toc_reduce(words[2:])])


def test_tocs_reduce_lowest_index_offender_across_the_chunk_seam():
    """The named group is the lowest-index offender, wherever it sits.

    Collapsing ``offsets[i + 1]`` onto ``offsets[i]`` empties group ``i`` and
    hands its word to group ``i + 1``, so the layout stays exactly covering
    and only the empty-group refusal is under test.  rayon may finish any
    group first; the reported index must still be the lowest offender, so the
    calls are repeated.
    """
    n = 2 * CHUNK
    words = time2toc(np.arange(n, dtype=np.uint64) * np.uint64(10**9))
    offsets = np.arange(n + 1, dtype=np.int64)
    offenders = [7, CHUNK - 1, CHUNK + 3]
    for i in offenders:
        offsets[i + 1] = offsets[i]
    for k, expect in enumerate(offenders):
        for _ in range(3):
            with pytest.raises(ValueError, match=rf"group {expect}: "):
                tocs_reduce(words, offsets)
        offsets[expect + 1] = expect + 1       # heal the lowest survivor
    assert len(tocs_reduce(words, offsets)) == n


def test_tocs_reduce_empty_batch_and_group_of_one():
    got = tocs_reduce(np.empty(0, dtype=np.uint64), [0])
    assert got.shape == (0,) and got.dtype == np.uint64
    words = rand_words(np.random.default_rng(1773), 500)
    # Every group a singleton: each word comes back verbatim.
    np.testing.assert_array_equal(
        tocs_reduce(words, np.arange(len(words) + 1, dtype=np.int64)), words)


def test_tocs_reduce_deterministic_across_runs():
    rng = np.random.default_rng(1774)
    args = ragged(random_groups(rng, 5000))
    first = tocs_reduce(*args)
    for _ in range(9):
        np.testing.assert_array_equal(tocs_reduce(*args), first)


def test_tocs_reduce_consumer_shape_per_cell_fold():
    """The zagg#410 consumer shape: a per-cell fold equals the scalar loop.

    GEDI shot pooling -- many shots' instants folding into one word per cell --
    and the ATL03 overview cascade, where a level's per-cell words are folded
    again into the coarser level (envelope of envelopes).  The cascade is the
    stronger claim: folding twice must equal folding the leaves once, which is
    the merge's associativity carried through the segmented form.
    """
    rng = np.random.default_rng(410)
    shots = rand_words(rng, 4096)
    # Unique cut points: zagg's fold sites never present an empty cell (they
    # short-circuit before the fold), which is why the empty segment refuses.
    per_cell = np.unique(rng.integers(1, len(shots), size=255))
    cell_offsets = np.concatenate([[0], per_cell, [len(shots)]]).astype(np.int64)
    cells = tocs_reduce(shots, cell_offsets)
    # Scalar loop equivalence, the thing the batch replaces.
    loop = [toc_reduce(shots[a:b])
            for a, b in zip(cell_offsets[:-1], cell_offsets[1:])]
    np.testing.assert_array_equal(cells, np.asarray(loop, dtype=np.uint64))
    # One pyramid level up: 4 cells per parent, envelope of envelopes.
    parents = np.arange(0, len(cells) + 1, 4, dtype=np.int64)
    if parents[-1] != len(cells):
        parents = np.append(parents, len(cells))
    up = tocs_reduce(cells, parents)
    direct = [toc_reduce(shots[cell_offsets[a]:cell_offsets[b]])
              for a, b in zip(parents[:-1], parents[1:])]
    np.testing.assert_array_equal(up, np.asarray(direct, dtype=np.uint64))


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


def test_empty_window_matches_nothing():
    # An empty window intersects and contains nothing -- for both
    # variants, even when it sits strictly inside a range's envelope.
    q = 5 * Q_END_NS
    words = np.array([time2toc(q), span2toc(q - Q_END_NS, q + Q_END_NS)],
                     dtype=np.uint64)
    assert not toc_overlaps(words, q, q).any()
    assert not toc_contains(words, q, q).any()


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
    # The same instant the golden timestamp fixture pins: 61,361 days of
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
    # A 0-d ndarray classifies as array, matching the word ops.
    out = from_datetime64(np.array("2020-05-17", dtype="datetime64[ns]"))
    assert isinstance(out, np.ndarray) and out.shape == (1,)


def test_pre1972_zero_offset_and_1972_step():
    from datetime import date

    # Zero offset before 1972: exactly the proleptic day count.
    days = (date(1960, 1, 1) - date(1850, 1, 1)).days
    assert from_datetime64("1960-01-01") == days * 86_400 * 10**9
    # The 9 s backward step across 1972-01-01: the last 9 SI seconds of
    # 1971 alias early 1972 ...
    assert (from_datetime64("1971-12-31T23:59:55")
            == from_datetime64("1972-01-01T00:00:04"))
    assert (from_datetime64("1972-01-01")
            == from_datetime64("1971-12-31T23:59:51"))
    # ... and aliased instants render on the 1972 side of the step.
    assert to_datetime64(from_datetime64("1971-12-31T23:59:55")) == \
        np.datetime64("1972-01-01T00:00:04", "ns")
    # From 1972 on the mapping is invertible (the roundtrip test sweeps
    # the modern era; this pins the very first invertible instant).
    assert to_datetime64(from_datetime64("1972-01-01")) == \
        np.datetime64("1972-01-01", "ns")


def test_timescale_empty_arrays():
    assert from_datetime64(np.array([], dtype="datetime64[ns]")).shape == (0,)
    empty = np.empty(0, dtype=np.uint64)
    assert to_datetime64(empty).shape == (0,)
    assert from_gps_ns(empty).shape == (0,)
    assert to_gps_ns(empty).shape == (0,)


def test_timescale_domain_errors():
    with pytest.raises(ValueError, match="2\\*\\*63"):
        to_datetime64(1 << 63)
    with pytest.raises(ValueError, match="ceiling"):
        from_gps_ns(TOC_MAX_NS)
    with pytest.raises(ValueError, match="q_start_ns"):
        toc_overlaps(time2toc(0), -1, 5)

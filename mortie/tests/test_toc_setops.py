"""Tests for the toc set algebra (issues #177 / #198).

``toc_normalize`` is the canonical cover form the ``Toc`` object builds on:
sorted maximal merges, per the espg-confirmed #177 rulings — Q1 (merge on
decoded bounds, never bridge a surviving decoded gap) and Q2 (subsumed
timestamps absorb; free instants survive bit-identical).  ``toc_and`` is
the one set operation over it — pairwise ``[max(starts), min(ends))``,
exact by grid closure (Q3).  The goldens here are normative at the Python
surface the way the cargo fixtures are at the kernel: the canonical form
is what ``Toc.__eq__`` will compare.
"""

import numpy as np
import pytest

from mortie.toc import (
    Q_END_NS,
    Q_START_NS,
    from_datetime64,
    span2toc,
    time2toc,
    toc2time,
    toc_and,
    toc_is_range,
    toc_normalize,
)


def u64(values):
    """Shorthand: a uint64 array from Python ints."""
    return np.asarray(values, dtype=np.uint64)


def covered(words, t):
    """Reference membership: is instant t in the decoded coverage of words?"""
    starts, ends = toc2time(np.atleast_1d(u64(words)))
    rng = toc_is_range(np.atleast_1d(u64(words)))
    return bool(np.any(np.where(rng, (starts <= t) & (t < ends), starts == t)))


def rand_words(rng, n, base=None):
    """A mixed batch of valid words, clustered around one random base so
    overlaps, exact abutments and absorptions all occur.

    Drawing instants and range starts uniformly over the whole span would
    spread ~1e18 ns of domain over a dozen words a few quanta wide, so no
    two envelopes would ever touch and ``toc_normalize`` would degenerate
    to a sort -- the randomized laws below would pass vacuously.  The
    ``toc_and`` tests pass an explicit shared ``base`` so the two operands
    actually intersect (independent bases would make every intersection
    empty and those laws vacuous the same way).
    """
    if base is None:
        base = np.uint64(int(rng.integers(0, 1 << 25)) * Q_END_NS)

    def off(k):
        return rng.integers(0, k * Q_END_NS, size=n, dtype=np.uint64)

    t = time2toc(base + off(40))
    a = base + off(40)
    r = span2toc(a, a + off(12))
    pick = rng.integers(0, 2, size=n).astype(bool)
    return np.where(pick, t, r)


# ── goldens (normative) ─────────────────────────────────────────────────


def test_golden_issue_177_absorption_example():
    # espg's Q2 example: t1, t2 inside a covering range R absorb; t3 months
    # later survives bit-identical; the Mar–Jul gap is preserved.
    r = span2toc(from_datetime64("2020-03-01"), from_datetime64("2020-03-05"))
    t1, t2, t3 = time2toc(from_datetime64(
        ["2020-03-02", "2020-03-04", "2020-07-04"]))
    got = toc_normalize(u64([t1, r, t2, t3]))
    assert got.tolist() == sorted([r, int(t3)])
    assert toc_is_range(got).tolist() == [True, False]


def test_golden_abutting_envelopes_merge():
    # r1's decoded end (2^32 grid) meets r2's decoded start (2^31 grid)
    # exactly: end code 8 → 8 * 2^32 = start code 16 * 2^31.
    r1 = span2toc(3 * Q_START_NS, 8 * Q_END_NS - 5)
    r2 = span2toc(16 * Q_START_NS + 1, 20 * Q_END_NS - 5)
    assert toc2time(r1)[1] == toc2time(r2)[0]
    merged = toc_normalize(u64([r2, r1]))
    assert merged.tolist() == [span2toc(3 * Q_START_NS, 20 * Q_END_NS - 5)]


def test_golden_one_quantum_gap_survives():
    # 2^31 ns is the smallest decoded gap the grids can express; it is
    # never bridged (Q1: a surviving gap is a floor on the true gap).
    r1 = span2toc(3 * Q_START_NS, 8 * Q_END_NS - 5)
    r3 = span2toc(17 * Q_START_NS + 1, 20 * Q_END_NS - 5)
    assert toc_normalize(u64([r3, r1])).tolist() == [r1, r3]


# ── Q1: range merging ───────────────────────────────────────────────────


def test_overlapping_and_nested_ranges_coalesce():
    a = span2toc(10 * Q_END_NS, 40 * Q_END_NS)
    b = span2toc(30 * Q_END_NS, 60 * Q_END_NS)
    nested = span2toc(12 * Q_END_NS, 20 * Q_END_NS)
    got = toc_normalize(u64([b, nested, a]))
    assert got.size == 1
    s, e = toc2time(int(got[0]))
    assert (s, e) == (toc2time(a)[0], toc2time(b)[1])


# ── Q2: timestamps ──────────────────────────────────────────────────────


def test_absorption_boundaries_are_exact():
    r = span2toc(50 * Q_END_NS, 70 * Q_END_NS)
    s, e = toc2time(r)
    at_start, last_in, at_end, before = time2toc(u64([s, e - 1, e, s - 1]))
    assert toc_normalize(u64([r, at_start])).tolist() == [r]
    assert toc_normalize(u64([r, last_in])).tolist() == [r]
    # The envelope end is exclusive: the instant at e is outside.
    assert toc_normalize(u64([r, at_end])).tolist() == [r, at_end]
    assert toc_normalize(u64([r, before])).tolist() == [before, r]


def test_timestamps_never_merge_and_equal_ones_dedupe():
    t = 9 * Q_END_NS + 3
    a, b = time2toc(u64([t, t + 1]))
    assert toc_normalize(u64([b, a])).tolist() == [a, b]
    got = toc_normalize(u64([a, a, a]))
    assert got.tolist() == [a]
    assert not toc_is_range(got)[0]


# ── canonical-form laws ─────────────────────────────────────────────────


def test_empty_and_singletons_pass_through():
    assert toc_normalize(u64([])).tolist() == []
    assert toc_normalize(u64([])).dtype == np.uint64
    t = time2toc(123_456_789)
    r = span2toc(5 * Q_END_NS, 6 * Q_END_NS)
    assert toc_normalize(u64([t])).tolist() == [t]
    assert toc_normalize(u64([r])).tolist() == [r]


def test_order_independent_and_idempotent():
    rng = np.random.default_rng(198)
    shrank = 0
    for _ in range(50):
        words = rand_words(rng, int(rng.integers(1, 12)))
        reference = toc_normalize(words)
        shrank += reference.size < words.size
        assert toc_normalize(reference).tolist() == reference.tolist()
        for _ in range(3):
            assert (toc_normalize(rng.permutation(words)).tolist()
                    == reference.tolist())
    # Guard against a vacuous generator: the laws are only interesting on
    # sets where something actually merged or was absorbed.
    assert shrank, "no merge or absorption exercised"


def test_coverage_is_preserved_exactly():
    # Membership at every decoded bound and its neighbors agrees between
    # the raw set and its canonical form — coverage-identical.
    rng = np.random.default_rng(177)
    shrank = 0
    for _ in range(30):
        words = rand_words(rng, int(rng.integers(1, 10)))
        canon = toc_normalize(words)
        shrank += canon.size < words.size
        starts, ends = toc2time(words)
        probes = set()
        for s, e in zip(starts.tolist(), ends.tolist()):
            probes.update((max(s - 1, 0), s, s + 1, max(e - 1, 0), e, e + 1))
        for t in probes:
            assert covered(words, t) == covered(canon, t)
    assert shrank, "no merge or absorption exercised"


def test_canonical_output_is_sorted_and_duplicate_free():
    rng = np.random.default_rng(41)
    shrank = 0
    for _ in range(30):
        words = rand_words(rng, int(rng.integers(1, 14)))
        canon = toc_normalize(words)
        shrank += canon.size < words.size
        # Compare, never subtract: np.diff on uint64 wraps, so a descending
        # pair reads as a huge positive difference and slips through.
        assert np.all(canon[1:] > canon[:-1])
    assert shrank, "no merge or absorption exercised"


def test_validation_matches_the_word_ops():
    with pytest.raises(ValueError, match="integer-typed"):
        toc_normalize(np.array([1.5, 2.5]))
    with pytest.raises(ValueError, match="non-negative"):
        toc_normalize(np.array([-1, 2]))


def test_scalar_input_yields_the_one_word_set():
    t = time2toc(42)
    assert toc_normalize(t).tolist() == [t]


# ── toc_and (issue #198 phase 2) ────────────────────────────────────────


def test_golden_and_calendar_overlap_is_exact():
    # Two campaigns share exactly their overlap week: the intersection
    # bounds are b's decoded start and a's decoded end, verbatim (Q3 grid
    # closure — no rounding arm).
    a = span2toc(from_datetime64("2020-03-01"), from_datetime64("2020-03-15"))
    b = span2toc(from_datetime64("2020-03-10"), from_datetime64("2020-04-01"))
    both = toc_and([a], [b])
    assert both.size == 1
    s, e = toc2time(int(both[0]))
    assert s == toc2time(b)[0] and e == toc2time(a)[1]
    assert s % Q_START_NS == 0 and e % Q_END_NS == 0


def test_and_disjoint_and_abutting_share_nothing():
    a = span2toc(3 * Q_START_NS, 8 * Q_END_NS - 5)
    abutting = span2toc(16 * Q_START_NS + 1, 20 * Q_END_NS - 5)
    assert toc2time(a)[1] == toc2time(abutting)[0]
    assert toc_and([a], [abutting]).tolist() == []
    assert toc_and([a], [span2toc(100 * Q_END_NS, 200 * Q_END_NS)]).tolist() == []


def test_and_timestamp_survival_is_exact():
    r = span2toc(50 * Q_END_NS, 70 * Q_END_NS)
    s, e = toc2time(r)
    inside, at_end = time2toc(u64([e - 1, e]))
    assert toc_and([inside], [r]).tolist() == [inside]
    assert toc_and([r], [inside]).tolist() == [inside]
    assert toc_and([at_end], [r]).tolist() == []
    assert toc_and([inside], [inside]).tolist() == [inside]
    assert toc_and([inside], [at_end]).tolist() == []


def test_and_accepts_raw_word_sets():
    rng = np.random.default_rng(3)
    hits = 0
    for _ in range(20):
        base = np.uint64(int(rng.integers(0, 1 << 25)) * Q_END_NS)
        a = rand_words(rng, int(rng.integers(1, 10)), base)
        b = rand_words(rng, int(rng.integers(1, 10)), base)
        both = toc_and(a, b)
        hits += both.size
        assert (both.tolist()
                == toc_and(toc_normalize(a), toc_normalize(b)).tolist())
    assert hits > 0, "no nonempty intersection exercised"


def test_and_laws_identity_commutativity_empty():
    rng = np.random.default_rng(17)
    hits = 0
    for _ in range(30):
        base = np.uint64(int(rng.integers(0, 1 << 25)) * Q_END_NS)
        a = rand_words(rng, int(rng.integers(1, 10)), base)
        b = rand_words(rng, int(rng.integers(1, 10)), base)
        hits += toc_and(a, b).size
        assert toc_and(a, a).tolist() == toc_normalize(a).tolist()
        assert toc_and(a, b).tolist() == toc_and(b, a).tolist()
        assert toc_and(a, u64([])).tolist() == []
        assert toc_and(u64([]), b).tolist() == []
    # The shared ``base`` is what makes the operands meet at all (see
    # ``rand_words``); without this guard a generator tweak reduces the
    # laws to empty == empty and they pass vacuously.
    assert hits > 0, "no nonempty intersection exercised"


def test_and_membership_matches_both_sides():
    # The defining property: covered by A ∩ B iff covered by A and by B.
    rng = np.random.default_rng(29)
    hits = 0
    for _ in range(30):
        base = np.uint64(int(rng.integers(0, 1 << 25)) * Q_END_NS)
        a = rand_words(rng, int(rng.integers(1, 8)), base)
        b = rand_words(rng, int(rng.integers(1, 8)), base)
        both = toc_and(a, b)
        hits += both.size
        starts_a, ends_a = toc2time(a)
        starts_b, ends_b = toc2time(b)
        probes = set()
        for s, e in zip(np.append(starts_a, starts_b).tolist(),
                        np.append(ends_a, ends_b).tolist()):
            probes.update((max(s - 1, 0), s, s + 1, max(e - 1, 0), e, e + 1))
        for t in probes:
            assert covered(both, t) == (covered(a, t) and covered(b, t))
    assert hits > 0, "no nonempty intersection exercised"


def test_and_output_is_canonical():
    rng = np.random.default_rng(31)
    hits = 0
    for _ in range(30):
        base = np.uint64(int(rng.integers(0, 1 << 25)) * Q_END_NS)
        both = toc_and(rand_words(rng, int(rng.integers(1, 10)), base),
                       rand_words(rng, int(rng.integers(1, 10)), base))
        hits += both.size
        assert toc_normalize(both).tolist() == both.tolist()
        assert bool(np.all(both[1:] > both[:-1]))
    assert hits > 0, "no nonempty intersection exercised"


def test_and_validation_matches_the_word_ops():
    with pytest.raises(ValueError, match="integer-typed"):
        toc_and(np.array([1.5]), u64([1]))
    with pytest.raises(ValueError, match="non-negative"):
        toc_and(u64([1]), np.array([-1]))
    assert toc_and(u64([]), u64([])).dtype == np.uint64

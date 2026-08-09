"""Tests for the 1xN broadcast MOC set ops and their scalar twin (issue #173).

The batch contract is parity: for every MOC ``i`` in the ragged batch,
``mocs_and``'s slice ``values[out[i]:out[i+1]]`` is byte-identical to the
scalar :func:`mortie.moc_and` on that pair alone, and ``mocs_intersect[i]``
is exactly "is that slice non-empty".  Both operand orders are exercised
(``and`` is commutative and both appear in the wild).  Plus the compaction
trap (a fully-occupied subtree that compacts to its parent), the layout/error
surface shared with ``mocs_to_orders``, determinism, and the GIL-release
guarantee.
"""

import threading
import time
from pathlib import Path

import numpy as np
import pytest

import mortie

# ---------------------------------------------------------------------------
# Helpers (the test_moc_batch.py corpus, reused for the set ops)
# ---------------------------------------------------------------------------


def _load_basin(basin_id):
    """Load Antarctic basin vertices.  Returns (lats, lons) or skips test."""
    coords_file = Path("mortie/tests/Ant_Grounded_DrainageSystem_Polygons.txt")
    if not coords_file.exists():
        pytest.skip("Antarctic polygon data not found")
    data = np.loadtxt(coords_file)
    mask = data[:, 2] == basin_id
    return data[mask, 0], data[mask, 1]


def _simplify_vertices(lats, lons, target):
    """Uniformly subsample vertices to approximately *target* count."""
    n = len(lats)
    if n <= target:
        return lats, lons
    step = max(1, n // target)
    idx = np.arange(0, n, step)
    return lats[idx], lons[idx]


def _ragged(mocs):
    """Concatenate MOCs into (values, offsets) arrow list layout."""
    values = np.concatenate(
        [np.asarray(m, dtype=np.uint64) for m in mocs] or [np.empty(0, np.uint64)]
    )
    offsets = np.zeros(len(mocs) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(m) for m in mocs])
    return values, offsets


def _word(nested, order, base=0):
    """One packed morton word as a length-1 uint64 array."""
    return np.atleast_1d(
        np.asarray(mortie.norm2mort([nested], [base], order), np.uint64)
    )


def _random_rings(rng, n):
    """*n* random small convex-ish rings scattered over the sphere."""
    rings = []
    for _ in range(n):
        n_verts = rng.integers(3, 9)
        clat = rng.uniform(-85.0, 85.0)
        clon = rng.uniform(-180.0, 180.0)
        radius = rng.uniform(0.5, 6.0)
        angles = np.sort(rng.uniform(0.0, 2 * np.pi, n_verts))
        lats = np.clip(clat + radius * np.cos(angles), -89.9, 89.9)
        lons = (clon + radius * np.sin(angles) + 180.0) % 360.0 - 180.0
        rings.append((lats, lons))
    return rings


def _random_mocs(rng, n, order):
    """*n* MOCs of random rings, as a list of uint64 arrays."""
    return [
        mortie.morton_coverage_moc(la, lo, order=order)
        for la, lo in _random_rings(rng, n)
    ]


def _aoi(order=6):
    """A shared operand shaped like an AOI: one wide mid-latitude quad cover."""
    return mortie.morton_coverage_moc(
        [10.0, 10.0, 45.0, 45.0], [-60.0, 10.0, 10.0, -60.0], order=order
    )


def _whole_sphere():
    """The 12 order-0 base cells: the whole sphere as a cover."""
    return np.asarray(
        mortie.norm2mort([0] * 12, list(range(12)), 0), dtype=np.uint64
    )


def _assert_broadcast_parity(a, mocs):
    """Assert both batch ops == the scalar pair, per MOC, both operand orders."""
    values, offsets = _ragged(mocs)
    out_vals, out = mortie.mocs_and(a, values, offsets)
    hits = mortie.mocs_intersect(a, values, offsets)
    assert out.dtype == np.int64 and out_vals.dtype == np.uint64
    assert hits.dtype == np.bool_ and len(hits) == len(mocs)
    assert len(out) == len(mocs) + 1
    assert out[0] == 0 and out[-1] == len(out_vals)
    for i, moc in enumerate(mocs):
        got = out_vals[out[i]:out[i + 1]]
        # Byte parity against the scalar, in both operand orders (commutative).
        np.testing.assert_array_equal(got, mortie.moc_and(a, moc))
        np.testing.assert_array_equal(got, mortie.moc_and(moc, a))
        # The predicate agrees with "the mocs_and span is non-empty" and with
        # its own scalar twin.
        assert hits[i] == (out[i + 1] > out[i])
        assert hits[i] == mortie.moc_intersects(a, moc)
        assert hits[i] == mortie.moc_intersects(moc, a)


# ---------------------------------------------------------------------------
# The scalar twin
# ---------------------------------------------------------------------------


def test_moc_intersects_matches_moc_and_size():
    """The scalar predicate equals ``moc_and(a, b).size > 0``, both orders."""
    rng = np.random.default_rng(173)
    mocs = _random_mocs(rng, 20, order=6)
    a = _aoi()
    for m in mocs:
        expect = mortie.moc_and(a, m).size > 0
        assert mortie.moc_intersects(a, m) == expect
        assert mortie.moc_intersects(m, a) == expect
    assert isinstance(mortie.moc_intersects(a, mocs[0]), bool)


def test_moc_intersects_empty_and_self():
    a = _aoi()
    empty = np.empty(0, np.uint64)
    assert mortie.moc_intersects(a, a) is True
    assert mortie.moc_intersects(a, empty) is False
    assert mortie.moc_intersects(empty, a) is False
    assert mortie.moc_intersects(empty, empty) is False


def test_moc_intersects_fully_occupied_subtree():
    """The compaction trap: a dense region that compacts to its parent.

    All 4 order-5 children of an order-4 cell normalize to that parent, so no
    input word survives into the compacted cover; a membership test against it
    would answer False for a child.  The geometric predicate must answer True.
    """
    children = np.asarray(
        mortie.norm2mort([44 * 4 + s for s in range(4)], [3] * 4, 5), np.uint64
    )
    parent = _word(44, 4, base=3)
    np.testing.assert_array_equal(mortie.compress_moc(children), parent)
    one_child = _word(44 * 4 + 2, 5, base=3)
    assert mortie.moc_intersects(children, one_child) is True
    assert mortie.moc_intersects(one_child, children) is True
    assert mortie.moc_and(children, one_child).size > 0
    # A cousin outside the occupied subtree stays out.
    cousin = _word(45 * 4 + 1, 5, base=3)
    assert mortie.moc_intersects(children, cousin) is False


# ---------------------------------------------------------------------------
# Parity: batch == scalar per MOC, both operand orders
# ---------------------------------------------------------------------------


def test_randomized_mocs_broadcast_parity():
    """40 random-ring MOCs against an AOI-shaped shared operand."""
    rng = np.random.default_rng(173)
    _assert_broadcast_parity(_aoi(), _random_mocs(rng, 40, order=6))


def test_mixed_order_covers_parity():
    """Hand-built mixed-order covers on both sides of the broadcast."""
    coarse = np.asarray(mortie.norm2mort([0, 1, 2], [0, 3, 7], 3), np.uint64)
    finer = np.asarray(mortie.norm2mort([0, 1, 2, 3], [5] * 4, 9), np.uint64)
    mixed = np.concatenate([coarse, finer])
    a = np.asarray(mortie.norm2mort([0, 5], [0, 5], 1), np.uint64)
    _assert_broadcast_parity(a, [coarse, finer, mixed])


def test_antarctic_basin_parity():
    """Real pole+antimeridian data: one basin as the AOI, others as items."""
    la, lo = _load_basin(24)
    a = mortie.morton_coverage_moc(*_simplify_vertices(la, lo, 300), order=6)
    mocs = []
    for basin_id in (2, 24):
        la, lo = _load_basin(basin_id)
        mocs.append(
            mortie.morton_coverage_moc(*_simplify_vertices(la, lo, 200), order=6)
        )
    mocs.append(np.empty(0, np.uint64))
    _assert_broadcast_parity(a, mocs)


def test_one_word_items_parity():
    """Length-1 MOCs — the shape most of the surveyed call sites use."""
    a = _aoi()
    flat = mortie.moc_to_order(a, 7)
    inside = [np.atleast_1d(w) for w in flat[:3]]
    outside = [_word(5, 7, base=8)]
    _assert_broadcast_parity(a, inside + outside)


def test_whole_sphere_shared_operand():
    """A whole-sphere ``a`` intersects every non-empty item to itself."""
    rng = np.random.default_rng(12)
    mocs = _random_mocs(rng, 6, order=5) + [np.empty(0, np.uint64)]
    sphere = _whole_sphere()
    _assert_broadcast_parity(sphere, mocs)
    values, offsets = _ragged(mocs)
    hits = mortie.mocs_intersect(sphere, values, offsets)
    np.testing.assert_array_equal(hits, [True] * 6 + [False])


def test_empty_shared_operand():
    """An empty ``a`` keeps every slot, empty, and answers all-False."""
    rng = np.random.default_rng(13)
    values, offsets = _ragged(_random_mocs(rng, 5, order=5))
    empty = np.empty(0, np.uint64)
    out_vals, out = mortie.mocs_and(empty, values, offsets)
    assert out_vals.size == 0
    np.testing.assert_array_equal(out, np.zeros(6, np.int64))
    np.testing.assert_array_equal(
        mortie.mocs_intersect(empty, values, offsets), [False] * 5
    )


def test_deterministic_across_runs():
    """Repeated identical calls give identical output (rayon-order-free)."""
    rng = np.random.default_rng(3)
    a = _aoi()
    values, offsets = _ragged(_random_mocs(rng, 25, order=6))
    v1, o1 = mortie.mocs_and(a, values, offsets)
    v2, o2 = mortie.mocs_and(a, values, offsets)
    np.testing.assert_array_equal(v1, v2)
    np.testing.assert_array_equal(o1, o2)
    np.testing.assert_array_equal(
        mortie.mocs_intersect(a, values, offsets),
        mortie.mocs_intersect(a, values, offsets),
    )


def test_broadcast_hits_the_fully_occupied_subtree():
    """The compaction trap through both batch paths (materialize + predicate)."""
    children = np.asarray(
        mortie.norm2mort([44 * 4 + s for s in range(4)], [3] * 4, 5), np.uint64
    )
    inside = _word(44 * 4 + 2, 5, base=3)
    outside = _word(45 * 4 + 1, 5, base=3)
    values, offsets = _ragged([inside, outside])
    np.testing.assert_array_equal(
        mortie.mocs_intersect(children, values, offsets), [True, False]
    )
    out_vals, out = mortie.mocs_and(children, values, offsets)
    np.testing.assert_array_equal(out_vals, inside)
    np.testing.assert_array_equal(out, [0, 1, 1])


# ---------------------------------------------------------------------------
# Layout edge cases and errors (the mocs_to_orders contract, shared)
# ---------------------------------------------------------------------------


def test_empty_batch():
    a = _aoi()
    out_vals, out = mortie.mocs_and(a, [], [0])
    assert out_vals.size == 0 and out_vals.dtype == np.uint64
    np.testing.assert_array_equal(out, [0])
    hits = mortie.mocs_intersect(a, [], [0])
    assert hits.size == 0 and hits.dtype == np.bool_


def test_single_item():
    a = _aoi()
    moc = mortie.morton_coverage_moc(
        [20.0, 30.0, 25.0], [-50.0, -50.0, -40.0], order=6
    )
    out_vals, out = mortie.mocs_and(a, moc, [0, len(moc)])
    expected = mortie.moc_and(a, moc)
    np.testing.assert_array_equal(out_vals, expected)
    np.testing.assert_array_equal(out, [0, len(expected)])
    assert mortie.mocs_intersect(a, moc, [0, len(moc)])[0] == (expected.size > 0)


def test_empty_moc_keeps_its_slot():
    """A zero-length item inside a non-empty batch stays an empty/False slot."""
    a = _aoi()
    item = mortie.moc_to_order(a, 6)[:1]
    out_vals, out = mortie.mocs_and(a, item, [0, 0, 1, 1])
    np.testing.assert_array_equal(out, [0, 0, len(out_vals), len(out_vals)])
    np.testing.assert_array_equal(out_vals, mortie.moc_and(a, item))
    np.testing.assert_array_equal(
        mortie.mocs_intersect(a, item, [0, 0, 1, 1]), [False, True, False]
    )


def test_offsets_must_exactly_cover_the_values():
    """Strict contract: offsets[0] == 0 and offsets[-1] == len(values)."""
    a = _aoi()
    values = np.asarray(mortie.norm2mort([0, 1, 2], [0, 0, 0], 4), np.uint64)
    for fn in (mortie.mocs_and, mortie.mocs_intersect):
        with pytest.raises(ValueError, match="must start at 0"):
            fn(a, values, [1, 3])
        with pytest.raises(ValueError, match="must end at the value count"):
            fn(a, values, [0, 2])
        with pytest.raises(ValueError, match=r"moc 1: .*monotonically"):
            fn(a, values, [0, 3, 1])
        with pytest.raises(ValueError, match=r"moc 1: .*exceeds"):
            fn(a, values, [0, 1, 99])
        with pytest.raises(ValueError, match="at least one element"):
            fn(a, values, [])
    # ... and an empty shared operand still validates layout first.
    with pytest.raises(ValueError, match="must end at the value count"):
        mortie.mocs_and(np.empty(0, np.uint64), values, [0, 2])


def test_malformed_word_is_a_named_value_error():
    """A bad word (the empty word 0) is the documented ValueError, both sides.

    Layout validation does not screen the morton words, so the kernel panic on
    a malformed word is caught per item — naming the lowest index — and for
    the hoisted shared operand, which is named as such.  Callers must see the
    catchable :class:`ValueError` the contract promises, never a
    ``BaseException``-derived ``PanicException`` (the issue #108 posture).
    """
    a = _aoi()
    good = mortie.moc_to_order(a, 6)[:1]
    values = np.concatenate([good, np.zeros(2, np.uint64)])
    bad = np.zeros(1, np.uint64)
    for fn in (mortie.mocs_and, mortie.mocs_intersect):
        with pytest.raises(ValueError, match=r"moc 1: "):
            fn(a, values, [0, 1, 2, 3])
        with pytest.raises(ValueError, match="shared operand:"):
            fn(bad, good, [0, 1])


# ---------------------------------------------------------------------------
# GIL release
# ---------------------------------------------------------------------------


def test_gil_released_during_broadcast():
    """A pure-Python counter thread keeps its free rate *inside* batch calls.

    Stronger instrument than the sibling test in ``test_moc_batch.py``
    (adversarial-review finding): the counter's free-running rate is
    calibrated first (under ``time.sleep``, which releases the GIL), progress
    is sampled around each individual call so inter-call scheduling gaps do
    not count, and the assertion demands a large fraction of the free rate
    over the in-call wall time.  A held GIL starves the counter for the whole
    call — at most one stray ~5 ms interpreter slice leaks in — so removing
    ``allow_threads`` fails this by orders of magnitude, where a fixed
    "progressed > 1000" would still pass on the gaps.
    """
    a = mortie.morton_coverage_moc(
        [10.0, 10.0, 45.0, 45.0], [-60.0, 10.0, 10.0, -60.0], order=9
    )
    items = mortie.moc_to_order(a, 9)
    offsets = np.arange(len(items) + 1, dtype=np.int64)

    counter = [0]
    stop = threading.Event()
    started = threading.Event()

    def busy():
        started.set()
        while not stop.is_set():
            counter[0] += 1

    b = threading.Thread(target=busy)
    b.start()
    started.wait()
    try:
        pre = counter[0]
        t0 = time.perf_counter()
        time.sleep(0.1)
        free_rate = (counter[0] - pre) / (time.perf_counter() - t0)

        progressed = 0
        in_call = 0.0
        while in_call < 0.3:
            p0 = counter[0]
            t1 = time.perf_counter()
            mortie.mocs_and(a, items, offsets)
            in_call += time.perf_counter() - t1
            progressed += counter[0] - p0
    finally:
        stop.set()
        b.join()

    floor = 0.2 * free_rate * in_call
    assert progressed > floor, (
        f"Python thread progressed {progressed} increments over {in_call:.2f}s "
        f"of in-call time, below {floor:.0f} (20% of the free rate "
        f"{free_rate:.0f}/s); the GIL was likely held (allow_threads not in "
        "effect)"
    )

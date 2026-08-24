"""Tests for the batch MOC densify kernel (_mocs_to_orders, issue #156).

The batch contract is parity: for every MOC ``i`` in the ragged batch, the slice
``values[out_offsets[i]:out_offsets[i+1]]`` is byte-identical to the scalar
:func:`mortie.moc_to_order` on that MOC alone.  Plus the composition with
:func:`mortie.polygons_to_morton_mocs` (whose ragged output is this function's
input verbatim), the per-MOC ``max_cells`` refusal naming the lowest-index
offender, the layout/error surface, and the GIL-release guarantee.
"""

import threading
from pathlib import Path

import numpy as np
import pytest

import mortie
from mortie.batch import _mocs_to_orders
from mortie.coverage import _morton_coverage_moc

# ---------------------------------------------------------------------------
# Helpers
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


def _word(nested, order):
    """One packed morton word in base cell 0 as a length-1 uint64 array."""
    return np.atleast_1d(np.asarray(mortie.norm2mort([nested], [0], order), np.uint64))


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
        _morton_coverage_moc(la, lo, order=order)
        for la, lo in _random_rings(rng, n)
    ]


def _assert_batch_parity(mocs, order, **kwargs):
    """Assert batch output == the scalar densify per MOC, slice by slice."""
    values, offsets = _ragged(mocs)
    flat, out = _mocs_to_orders(values, offsets, order, **kwargs)
    assert out.dtype == np.int64 and flat.dtype == np.uint64
    assert len(out) == len(mocs) + 1
    assert out[0] == 0 and out[-1] == len(flat)
    for i, moc in enumerate(mocs):
        expected = mortie.moc_to_order(moc, order, **kwargs)
        np.testing.assert_array_equal(flat[out[i]:out[i + 1]], expected)


# ---------------------------------------------------------------------------
# Parity: batch == scalar per MOC
# ---------------------------------------------------------------------------


def test_randomized_mocs_parity():
    """40 MOCs of random rings match the scalar densify per MOC."""
    rng = np.random.default_rng(156)
    _assert_batch_parity(_random_mocs(rng, 40, order=6), order=8)


def test_mixed_order_and_coarsening_parity():
    """Hand-built mixed-order MOCs, including cells finer than the target.

    ``moc_to_order`` coarsens finer cells to their ancestor at ``order`` and
    dedups; the batch must reproduce that, not just the coarse-expansion half.
    """
    coarse = np.asarray(mortie.norm2mort([0, 1, 2], [0, 3, 7], 3), dtype=np.uint64)
    finer = np.asarray(mortie.norm2mort([0, 1, 2, 3], [5] * 4, 9), dtype=np.uint64)
    mixed = np.concatenate([coarse, finer])
    _assert_batch_parity([coarse, finer, mixed], order=6)


def test_order_zero_parity():
    """Order 0 is a coarsen target the batch must answer like the scalar.

    ``moc_to_order(moc, 0)`` returns the base cells a MOC touches — first-class
    in mortie (``_whole_sphere()`` is order-0 words), not a degenerate case — so
    the batch must accept it rather than inherit the coverer's ``1..=29``
    (issue #156 review).
    """
    rng = np.random.default_rng(1560)
    mocs = _random_mocs(rng, 8, order=5)
    # A MOC spanning two base cells coarsens to exactly those two base cells.
    mocs.append(np.asarray(mortie.norm2mort([0, 0], [2, 5], 4), dtype=np.uint64))
    _assert_batch_parity(mocs, order=0)
    flat, out = _mocs_to_orders(*_ragged(mocs[-1:]), 0)
    np.testing.assert_array_equal(flat, mortie.norm2mort([0, 0], [2, 5], 0))


def test_antimeridian_pole_and_basecell_edge_parity():
    """The classic hard rings, covered then densified."""
    rings = [
        # Antimeridian-crossing quad.
        (np.array([-10.0, -10.0, 10.0, 10.0]),
         np.array([170.0, -170.0, -170.0, 170.0])),
        # Pole-adjacent (north) triangle.
        (np.array([85.0, 86.0, 87.0]), np.array([-30.0, 60.0, -150.0])),
        # Square aligned on basecell edges (lon multiple of 45, equator).
        (np.array([-5.0, -5.0, 5.0, 5.0]),
         np.array([40.0, 50.0, 50.0, 40.0])),
    ]
    mocs = [_morton_coverage_moc(la, lo, order=6) for la, lo in rings]
    _assert_batch_parity(mocs, order=8)


def test_antarctic_basin_parity():
    """Real pole+antimeridian data: simplified basin covers batch == scalar."""
    mocs = []
    for basin_id in (24, 2):
        la, lo = _load_basin(basin_id)
        la, lo = _simplify_vertices(la, lo, 300)
        mocs.append(_morton_coverage_moc(la, lo, order=6))
    _assert_batch_parity(mocs, order=7)


def test_deterministic_across_runs():
    """Two identical calls give identical ragged output (rayon-order-free)."""
    rng = np.random.default_rng(3)
    values, offsets = _ragged(_random_mocs(rng, 25, order=6))
    v1, o1 = _mocs_to_orders(values, offsets, 8)
    v2, o2 = _mocs_to_orders(values, offsets, 8)
    np.testing.assert_array_equal(v1, v2)
    np.testing.assert_array_equal(o1, o2)


def test_slices_are_sorted_unique():
    """Each slice comes back sorted-unique, as the docstring promises."""
    rng = np.random.default_rng(9)
    values, offsets = _ragged(_random_mocs(rng, 12, order=5))
    flat, out = _mocs_to_orders(values, offsets, 8)
    for i in range(len(out) - 1):
        part = flat[out[i]:out[i + 1]]
        assert np.all(np.diff(part) > 0)
        np.testing.assert_array_equal(part, np.unique(part))


# ---------------------------------------------------------------------------
# Composition with the batch coverer (the #154 pair, verbatim)
# ---------------------------------------------------------------------------


def test_chains_with_polygons_to_morton_mocs():
    """The coverer's ragged pair is this call's input with no marshalling."""
    rng = np.random.default_rng(42)
    rings = _random_rings(rng, 30)
    lats = np.concatenate([la for la, _ in rings])
    lons = np.concatenate([lo for _, lo in rings])
    off_in = np.zeros(len(rings) + 1, dtype=np.int64)
    off_in[1:] = np.cumsum([len(la) for la, _ in rings])

    mocs, off = mortie.polygons_to_morton_mocs(lats, lons, off_in, order=6)
    flat, flat_off = _mocs_to_orders(mocs, off, 8)

    np.testing.assert_array_equal(off, np.asarray(off, dtype=np.int64))
    assert len(flat_off) == len(off)
    for i, (la, lo) in enumerate(rings):
        expected = mortie.moc_to_order(
            _morton_coverage_moc(la, lo, order=6), 8
        )
        np.testing.assert_array_equal(flat[flat_off[i]:flat_off[i + 1]], expected)


def test_chained_basin_rings_match_the_flat_cover():
    """Both stages over the Antarctic fixtures reproduce the flat cover.

    ``moc_to_order(_morton_coverage_moc(ring, order), order)`` is the flat
    ``morton_coverage`` of that ring (the identity pinned in
    ``test_coverage.py``), so the chained batch must reproduce it too.
    """
    rings = []
    for basin_id in (24, 2):
        la, lo = _load_basin(basin_id)
        rings.append(_simplify_vertices(la, lo, 200))
    lats = np.concatenate([la for la, _ in rings])
    lons = np.concatenate([lo for _, lo in rings])
    off_in = np.array([0, len(rings[0][0]), len(lats)], dtype=np.int64)

    mocs, off = mortie.polygons_to_morton_mocs(lats, lons, off_in, order=5)
    flat, flat_off = _mocs_to_orders(mocs, off, 5)
    for i, (la, lo) in enumerate(rings):
        expected = mortie.morton_coverage(la, lo, order=5)
        np.testing.assert_array_equal(flat[flat_off[i]:flat_off[i + 1]], expected)


# ---------------------------------------------------------------------------
# Layout edge cases
# ---------------------------------------------------------------------------


def test_empty_batch():
    flat, out = _mocs_to_orders([], [0], 7)
    assert flat.size == 0 and flat.dtype == np.uint64
    np.testing.assert_array_equal(out, [0])


def test_single_moc():
    moc = _morton_coverage_moc(
        [40.0, 50.0, 45.0], [-120.0, -120.0, -110.0], order=6
    )
    flat, out = _mocs_to_orders(moc, [0, len(moc)], 8)
    expected = mortie.moc_to_order(moc, 8)
    np.testing.assert_array_equal(flat, expected)
    np.testing.assert_array_equal(out, [0, len(expected)])


def test_empty_moc_keeps_its_slot():
    """A zero-length MOC is legal and densifies to an empty slice."""
    moc = _word(0, 4)
    flat, out = _mocs_to_orders(moc, [0, 0, 1, 1], 6)
    np.testing.assert_array_equal(out, [0, 0, len(flat), len(flat)])
    np.testing.assert_array_equal(flat, mortie.moc_to_order(moc, 6))


def test_offsets_must_exactly_cover_the_values():
    """Strict contract: offsets[0] == 0 and offsets[-1] == len(values)."""
    values = np.asarray(mortie.norm2mort([0, 1, 2], [0, 0, 0], 4), dtype=np.uint64)
    with pytest.raises(ValueError, match="must start at 0"):
        _mocs_to_orders(values, [1, 3], 6)
    with pytest.raises(ValueError, match="must end at the value count"):
        _mocs_to_orders(values, [0, 2], 6)
    # The re-based spelling of that same MOC is what the core accepts.
    flat, out = _mocs_to_orders(values[:2], [0, 2], 6)
    np.testing.assert_array_equal(flat, mortie.moc_to_order(values[:2], 6))
    np.testing.assert_array_equal(out, [0, len(flat)])


# ---------------------------------------------------------------------------
# Fail-fast errors naming the lowest-index MOC
# ---------------------------------------------------------------------------


def test_offsets_errors_name_moc_index():
    values = np.asarray(mortie.norm2mort([0, 1, 2], [0, 0, 0], 4), dtype=np.uint64)
    with pytest.raises(ValueError, match=r"moc 1: .*monotonically"):
        _mocs_to_orders(values, [0, 3, 1], 6)
    with pytest.raises(ValueError, match=r"moc 1: .*exceeds"):
        _mocs_to_orders(values, [0, 1, 99], 6)
    with pytest.raises(ValueError, match="must start at 0"):
        _mocs_to_orders(values, [-1, 3], 6)
    with pytest.raises(ValueError, match="at least one element"):
        _mocs_to_orders(values, [], 6)
    with pytest.raises(ValueError, match="Order must be"):
        _mocs_to_orders(values, [0, 3], 30)


def test_budget_refusal_names_lowest_index_moc():
    """The per-MOC ``max_cells`` refusal names the lowest-index offender.

    MOC 0 densifies to 16 cells at order 8, MOC 1 to 256, MOC 2 to 16 — so a
    budget of 100 is over-run only by MOC 1, and one of 10 by MOC 0 first.
    """
    small = _word(600, 6)
    big = _word(60, 4)
    values = np.concatenate([small, big, small])
    offsets = [0, 1, 2, 3]

    with pytest.raises(ValueError, match=r"moc 1: moc_to_order would densify"):
        _mocs_to_orders(values, offsets, 8, max_cells=100)
    with pytest.raises(ValueError, match=r"moc 0: "):
        _mocs_to_orders(values, offsets, 8, max_cells=10)
    # The message carries the scalar's escape hatches verbatim.
    with pytest.raises(ValueError, match="max_cells=None to proceed"):
        _mocs_to_orders(values, offsets, 8, max_cells=10)


def test_budget_is_per_moc_and_matches_the_scalar_boundary():
    """The budget applies per MOC, at the same >-boundary as the scalar."""
    small = _word(600, 6)
    big = _word(60, 4)
    values = np.concatenate([small, big])
    offsets = [0, 1, 2]

    # 272 cells in total, past a 256 budget, but no single MOC exceeds it.
    flat, out = _mocs_to_orders(values, offsets, 8, max_cells=256)
    assert len(flat) == 16 + 256
    assert out[-1] == len(flat)
    # Strict ``>``: exactly the estimate passes, one below refuses -- the same
    # boundary the scalar guard uses.
    mortie.moc_to_order(big, 8, max_cells=256)
    with pytest.raises(ValueError):
        mortie.moc_to_order(big, 8, max_cells=255)
    with pytest.raises(ValueError, match="moc 1:"):
        _mocs_to_orders(values, offsets, 8, max_cells=255)


def test_budget_default_is_the_single_flat_cover_threshold():
    """The default budget is the one ``_FLAT_COVER_WARN_THRESHOLD`` (issue #108).

    One number, not a second higher ceiling: the batch default must be the same
    line as the scalar's, and ``max_cells=None`` must lift it for both.
    """
    from mortie import batch, coverage

    threshold = coverage._FLAT_COVER_WARN_THRESHOLD
    assert mortie.moc_to_order.__defaults__ == (threshold,)
    assert batch._mocs_to_orders.__defaults__ == (threshold,)

    # An order-2 cell densifies to 4**19 cells at order 21 -- past the default.
    word = _word(3, 2)
    with pytest.raises(ValueError, match=f"max_cells={threshold}"):
        _mocs_to_orders(word, [0, 1], 21)
    with pytest.raises(ValueError, match=f"max_cells={threshold}"):
        mortie.moc_to_order(word, 21)
    # ... and the escape is a caller parameter, not a hard ceiling.
    flat, _ = _mocs_to_orders(word, [0, 1], 12, max_cells=None)
    np.testing.assert_array_equal(flat, mortie.moc_to_order(word, 12, max_cells=None))


def test_budget_domain_matches_the_scalar():
    """The batch accepts the same ``max_cells`` spellings the scalar does.

    The default-equality check above pins the default *value*; this pins the
    accepted *domain* (issue #156 review). A float budget and one past
    ``2 ** 64`` used to hit the binding's ``u64`` as ``TypeError`` /
    ``OverflowError`` where the scalar just compared; a negative budget used to
    be ``OverflowError`` where the scalar refuses with ``ValueError``.
    """
    word = _word(3, 2)  # 4**6 = 4096 cells at order 8
    offsets = [0, 1]
    for budget in (1e9, float(4 ** 6), 2 ** 64, 2 ** 70, np.uint64(4 ** 6)):
        flat, _ = _mocs_to_orders(word, offsets, 8, max_cells=budget)
        np.testing.assert_array_equal(
            flat, mortie.moc_to_order(word, 8, max_cells=budget)
        )
    # Floors like the scalar's comparison: 4095.5 refuses, 4096.5 does not.
    with pytest.raises(ValueError, match="moc 0:"):
        _mocs_to_orders(word, offsets, 8, max_cells=4095.5)
    with pytest.raises(ValueError):
        mortie.moc_to_order(word, 8, max_cells=4095.5)
    flat, _ = _mocs_to_orders(word, offsets, 8, max_cells=4096.5)
    assert len(flat) == 4 ** 6
    # A negative budget refuses in both, as a catchable ValueError.
    with pytest.raises(ValueError, match="non-negative"):
        _mocs_to_orders(word, offsets, 8, max_cells=-1)
    with pytest.raises(ValueError):
        mortie.moc_to_order(word, 8, max_cells=-1)


def test_budget_refusal_precedes_any_densify():
    """An over-budget MOC refuses the call before anything is materialized.

    Pinned by making the *last* MOC the offender at an order whose flat cover
    would be astronomic: the call must return promptly with a ValueError rather
    than densify the earlier MOCs first.
    """
    small = _word(0, 27)
    huge = _word(0, 1)
    values = np.concatenate([small, huge])
    with pytest.raises(ValueError, match="moc 1:"):
        _mocs_to_orders(values, [0, 1, 2], 29)


def test_malformed_word_is_a_named_value_error_at_both_budgets():
    """A bad word is the same named ``ValueError`` with the budget on or off.

    The budget estimate decodes every word in the serial pre-pass, so it panics
    in ``mort2nested`` on the empty word 0 exactly as the densify kernel does.
    That pass ran outside the kernel's panic capture, so the *default*
    ``max_cells`` turned the refusal into a ``pyo3_runtime.PanicException`` --
    ``BaseException``-derived, missed by ``except ValueError`` *and* by
    ``except Exception`` -- while ``max_cells=None`` gave the documented
    ``ValueError``.  Same input, opposite exception contract, chosen by a
    keyword default (issue #161).
    """
    from mortie import coverage

    values = np.concatenate([np.zeros(1, np.uint64), _word(600, 6)])
    for max_cells in (coverage._FLAT_COVER_WARN_THRESHOLD, None):
        with pytest.raises(ValueError, match=r"moc 0: Morton index cannot be zero"):
            _mocs_to_orders(values, [0, 2], 8, max_cells=max_cells)
    # The default is that same budget, and must behave identically.
    with pytest.raises(ValueError, match=r"moc 0: Morton index cannot be zero"):
        _mocs_to_orders(values, [0, 2], 8)
    # A plain ``except Exception`` must see it -- the PanicException lesson.
    caught = None
    try:
        _mocs_to_orders(values, [0, 2], 8)
    except Exception as exc:
        caught = exc
    assert isinstance(caught, ValueError), caught
    # The lowest-index rule holds when the bad word is not MOC 0.
    with pytest.raises(ValueError, match=r"moc 1: Morton index cannot be zero"):
        _mocs_to_orders(values[::-1].copy(), [0, 1, 2], 8)


# ---------------------------------------------------------------------------
# GIL release
# ---------------------------------------------------------------------------


def test_gil_released_during_batch():
    """A pure-Python counter thread keeps running during a large batch call.

    Same instrument as ``test_threading.test_gil_released_during_rust_compute``:
    with the GIL held for the whole call the counter cannot advance; released,
    it climbs by many thousands.
    """
    rng = np.random.default_rng(5)
    values, offsets = _ragged(_random_mocs(rng, 400, order=6))

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
        for _ in range(20):
            _mocs_to_orders(values, offsets, 12, max_cells=None)
        progressed = counter[0] - pre
    finally:
        stop.set()
        b.join()

    assert progressed > 1000, (
        f"Python thread made ~no progress during the batch call ({progressed}); "
        "the GIL was likely held (allow_threads not in effect)"
    )

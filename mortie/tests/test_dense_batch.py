"""Tests for the dense-output batches (``common_ancestors`` / ``children_of``).

Issue #156 phase 3.  Both ops are *dense*: their results are fixed-shape arrays
rather than a ragged (values, offsets) pair, which is why they are one phase.

The contract for both is per-item parity with the scalar twin — result ``i`` is
byte-identical to :func:`mortie.common_ancestor` on group ``i`` alone, and row
``i`` to :func:`mortie.generate_morton_children` on ``words[i]`` alone — over
randomized input and the in-tree Antarctic basin fixtures.  Plus the layout and
error surface (lowest-index fail-fast, proved across chunk boundaries),
determinism under rayon, and the GIL-release guarantee.
"""

import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

import mortie

# The Rust batches step through 2048 items at a time, so offenders are placed
# either side of these seams rather than at arbitrary indices.
CHUNK = 2048

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


def _basin_cells(basin_id, order, limit=None):
    """A real Antarctic basin's flat cover at ``order`` (uint64, one order)."""
    lats, lons = _simplify_vertices(*_load_basin(basin_id), 300)
    cells = np.asarray(mortie.morton_coverage(lats, lons, order=order), np.uint64)
    return cells[:limit] if limit else cells


def _ragged(groups):
    """Concatenate word groups into (values, offsets) arrow list layout."""
    values = np.concatenate(
        [np.asarray(g, dtype=np.uint64) for g in groups] or [np.empty(0, np.uint64)]
    )
    offsets = np.zeros(len(groups) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(g) for g in groups])
    return values, offsets


def _random_groups(rng, n, order, base=None, size=(1, 6)):
    """*n* groups of same-base-cell words, each 1-5 words at ``order``."""
    groups = []
    span = 1 << (2 * order)
    for _ in range(n):
        b = rng.integers(0, 12) if base is None else base
        k = rng.integers(*size)
        nested = int(b) * span + rng.integers(0, span, k)
        groups.append(_pack(nested, np.full(k, order, dtype=np.uint8)))
    return groups


def _pack(nested, depths):
    """Pack HEALPix NESTED ids at given depths into packed morton words."""
    from mortie import _rustie

    return np.asarray(
        _rustie.rust_nested2mort(
            np.ascontiguousarray(np.asarray(nested, dtype=np.uint64)),
            np.ascontiguousarray(np.asarray(depths, dtype=np.uint8)),
        ),
        dtype=np.uint64,
    )


def _assert_ancestor_parity(groups):
    """Assert the batch result == the scalar reduction per group."""
    values, offsets = _ragged(groups)
    got = mortie.common_ancestors(values, offsets)
    assert got.dtype == np.uint64 and got.shape == (len(groups),)
    for i, group in enumerate(groups):
        assert int(got[i]) == int(mortie.common_ancestor(group)), f"group {i}"


def _assert_children_parity(words, order):
    """Assert the batch block == the scalar generator per word."""
    words = np.asarray(words, dtype=np.uint64)
    got = mortie.children_of(words, order)
    assert got.dtype == np.uint64
    assert got.shape[0] == len(words)
    for i, w in enumerate(words):
        expected = mortie.generate_morton_children(int(w), order)
        np.testing.assert_array_equal(got[i], expected, err_msg=f"word {i}")


# ---------------------------------------------------------------------------
# common_ancestors: parity with the scalar
# ---------------------------------------------------------------------------


def test_ancestors_randomized_parity():
    """200 random same-base-cell groups match the scalar reduction."""
    rng = np.random.default_rng(156)
    _assert_ancestor_parity(_random_groups(rng, 200, order=7))


def test_ancestors_mixed_order_groups_parity():
    """Groups holding mixed-order words reduce like the scalar does."""
    rng = np.random.default_rng(1563)
    groups = []
    for _ in range(60):
        base = int(rng.integers(0, 12))
        parts = [
            _pack(base * (1 << (2 * o)) + rng.integers(0, 1 << (2 * o), 2),
                  np.full(2, o, dtype=np.uint8))
            for o in (3, 6, 9)
        ]
        groups.append(np.concatenate(parts))
    _assert_ancestor_parity(groups)


def test_ancestors_sibling_groups_reduce_to_their_parent():
    """The four order-(k+1) children of a cell reduce to that cell."""
    parents = np.asarray(mortie.norm2mort([11, 7, 3], [0, 3, 9], 4), dtype=np.uint64)
    groups = [mortie.generate_morton_children(int(p), 5) for p in parents]
    values, offsets = _ragged(groups)
    np.testing.assert_array_equal(mortie.common_ancestors(values, offsets), parents)


def test_ancestors_antarctic_basin_parity():
    """Real pole+antimeridian data: basin cover cells grouped by base cell."""
    groups = []
    for basin_id in (24, 2):
        cells = _basin_cells(basin_id, order=6)
        # Real covers straddle base cells; the scalar refuses those, so use the
        # partition mortie itself offers as the group-maker.
        groups.extend(mortie.split_base_cells(cells).values())
    assert len(groups) > 2, "fixture should span several base cells"
    _assert_ancestor_parity(groups)


def test_ancestors_group_of_one_returns_that_word():
    """A one-word group comes back verbatim, kind preserved (the scalar rule)."""
    cells = _basin_cells(24, order=6, limit=500)
    got = mortie.common_ancestors(cells, np.arange(len(cells) + 1, dtype=np.int64))
    np.testing.assert_array_equal(got, cells)


def test_ancestors_order_zero_and_max_order_edges():
    """Order-0 base cells and order-29 words both reduce like the scalar."""
    bases = _pack(np.arange(12), np.zeros(12, dtype=np.uint8))
    # Each base cell alone is its own ancestor; a pair from one base cell at
    # order 29 reduces to their enclosing area cell.
    deep = _pack([3 * (1 << 58), 3 * (1 << 58) + 1], np.full(2, 29, dtype=np.uint8))
    groups = [bases[i:i + 1] for i in range(12)] + [deep]
    _assert_ancestor_parity(groups)
    got = mortie.common_ancestors(*_ragged(groups))
    np.testing.assert_array_equal(got[:12], bases)


def test_ancestors_empty_batch_and_single_group():
    got = mortie.common_ancestors([], [0])
    assert got.shape == (0,) and got.dtype == np.uint64
    cells = _basin_cells(24, order=6, limit=64)
    one = mortie.split_base_cells(cells)
    group = next(iter(one.values()))
    got = mortie.common_ancestors(group, [0, len(group)])
    assert got.shape == (1,)
    assert int(got[0]) == int(mortie.common_ancestor(group))


# ---------------------------------------------------------------------------
# common_ancestors: layout and error surface
# ---------------------------------------------------------------------------


def test_ancestors_offsets_must_exactly_cover_the_values():
    values = np.asarray(mortie.norm2mort([0, 1, 2], [0, 0, 0], 4), dtype=np.uint64)
    with pytest.raises(ValueError, match="must start at 0"):
        mortie.common_ancestors(values, [1, 3])
    with pytest.raises(ValueError, match="must end at the value count"):
        mortie.common_ancestors(values, [0, 2])
    with pytest.raises(ValueError, match="at least one element"):
        mortie.common_ancestors(values, [])
    # The re-based spelling of that same group is what the core accepts.
    got = mortie.common_ancestors(values[:2], [0, 2])
    assert int(got[0]) == int(mortie.common_ancestor(values[:2]))


def test_ancestors_errors_name_the_group_index():
    values = np.asarray(mortie.norm2mort([0, 1, 2], [0, 0, 5], 4), dtype=np.uint64)
    with pytest.raises(ValueError, match=r"group 1: .*monotonically"):
        mortie.common_ancestors(values, [0, 3, 1])
    with pytest.raises(ValueError, match=r"group 1: .*exceeds"):
        mortie.common_ancestors(values, [0, 1, 99])
    # An empty group is refused (many->one over nothing has no answer), unlike
    # the ragged batches where an empty item is a legal empty slot.
    with pytest.raises(ValueError, match=r"group 1: .*empty"):
        mortie.common_ancestors(values, [0, 2, 2, 3])
    # Mixed base cells: the scalar's own refusal, per group.
    with pytest.raises(ValueError, match=r"group 0: .*multiple base cells"):
        mortie.common_ancestors(values, [0, 3])
    # An invalid word is named by its group.
    with pytest.raises(ValueError, match=r"group 0: "):
        mortie.common_ancestors([0, 0], [0, 2])


def test_ancestors_layout_errors_outrank_domain_errors():
    """Layout is a whole-batch pre-pass, so it outranks a lower-index domain error.

    The lowest-index rule holds *within* a pass, not across the two: a bad
    offset at group 7 is reported ahead of a mixed-base-cell group 0.  That is
    deliberate — a broken ``offsets`` array is what the group indices a domain
    error would be reported by are derived from — and pinned here so it cannot
    drift silently.
    """
    n = 10
    values = _pack(np.arange(2 * n) % 4096, np.full(2 * n, 6, np.uint8))
    values[1] = _pack([(1 << 12) + 1], [6])[0]  # group 0 spans two base cells
    offsets = np.arange(0, 2 * n + 1, 2, dtype=np.int64)
    with pytest.raises(ValueError, match=r"group 0: .*multiple base cells"):
        mortie.common_ancestors(values, offsets)
    broken = offsets.copy()
    broken[8] = 99
    with pytest.raises(ValueError, match=r"group 7: offset 99 exceeds"):
        mortie.common_ancestors(values, broken)


@pytest.mark.parametrize(
    "offender", [0, 1, 7, CHUNK - 1, CHUNK, CHUNK + 1, 2 * CHUNK + 5, 3 * CHUNK - 1]
)
def test_ancestors_lowest_index_holds_across_chunk_boundaries(offender):
    """The named group is the lowest-index offender, wherever it sits.

    One offender at a time, swept over the chunk seams: rayon may finish any
    group first, but the reported index must be the offending one — the batch
    materializes a whole chunk and walks it in index order before failing.
    """
    n = 3 * CHUNK
    base0 = _pack(np.arange(2 * n) % 4096, np.full(2 * n, 6, dtype=np.uint8))
    offsets = np.arange(0, 2 * n + 1, 2, dtype=np.int64)
    values = base0.copy()
    # Break one group by pulling its second word from another base cell.
    values[2 * offender + 1] = _pack([(1 << 12) + 1], [6])[0]
    for _ in range(5):  # repeated: a nondeterministic answer would show up here
        with pytest.raises(ValueError, match=rf"group {offender}: "):
            mortie.common_ancestors(values, offsets)


def test_ancestors_lowest_index_wins_over_later_offenders():
    """With offenders at several indices, the lowest is always the one named."""
    n = 3 * CHUNK
    values = _pack(np.arange(2 * n) % 4096, np.full(2 * n, 6, dtype=np.uint8))
    offsets = np.arange(0, 2 * n + 1, 2, dtype=np.int64)
    other = _pack([(1 << 12) + 1], [6])[0]
    offenders = [3, CHUNK - 1, CHUNK, 2 * CHUNK + 5, n - 1]
    for off in offenders:
        values[2 * off + 1] = other
    # Heal them lowest-first; each call names the lowest survivor.
    for i, expect in enumerate(offenders):
        for _ in range(3):
            with pytest.raises(ValueError, match=rf"group {expect}: "):
                mortie.common_ancestors(values, offsets)
        values[2 * offenders[i] + 1] = values[2 * offenders[i]]
    assert len(mortie.common_ancestors(values, offsets)) == n


def test_ancestors_deterministic_across_runs():
    rng = np.random.default_rng(3)
    values, offsets = _ragged(_random_groups(rng, 5000, order=8))
    first = mortie.common_ancestors(values, offsets)
    for _ in range(9):
        np.testing.assert_array_equal(mortie.common_ancestors(values, offsets), first)


# ---------------------------------------------------------------------------
# children_of: parity with the scalar
# ---------------------------------------------------------------------------


def test_children_randomized_parity():
    """1000 random order-5 parents refine to order 8 exactly as the scalar."""
    rng = np.random.default_rng(156)
    words = _pack(rng.integers(0, 12 << 10, 1000), np.full(1000, 5, np.uint8))
    _assert_children_parity(words, 8)


def test_children_antarctic_basin_parity():
    """Real cover cells (pole + antimeridian) refine like the scalar."""
    cells = _basin_cells(24, order=6, limit=400)
    _assert_children_parity(cells, 9)


def test_children_at_shipped_atl03_shard_orders():
    """Real shard keys at the shipped order-9 shardmap, refined per zagg.

    zagg's HEALPix grid refines each shard key to its sub-chunks and then each
    sub-chunk to its cells (``grids/healpix.py:199``), which is the two-step
    ``generate_morton_children`` loop this replaces.  Both steps must be
    byte-identical to the scalar on real Antarctic shard keys.
    """
    shard_keys = _basin_cells(2, order=9, limit=200)
    sub_chunks = mortie.children_of(shard_keys, 11)
    assert sub_chunks.shape == (len(shard_keys), 16)
    _assert_children_parity(shard_keys, 11)
    # Second step: the sub-chunks themselves refine to their cells.
    flat = sub_chunks[:8].ravel()
    _assert_children_parity(flat, 13)


def test_children_zero_depth_returns_the_parents_verbatim():
    """``order`` equal to the parents' order gives back an (n, 1) identity."""
    cells = _basin_cells(24, order=7, limit=200)
    got = mortie.children_of(cells, 7)
    assert got.shape == (len(cells), 1)
    np.testing.assert_array_equal(got[:, 0], cells)
    _assert_children_parity(cells, 7)


def test_children_zero_depth_preserves_a_point_word():
    """A point word survives ``d == 0`` — the reason it is returned verbatim.

    Re-packing through the nested round trip would hand back the order-29
    *area* cell instead, which is a different word and a different kind.
    """
    lats = np.array([-71.5, 40.25])
    lons = np.array([-122.75, 3.5])
    points = np.asarray(mortie.geo2mort(lats, lons, points=True), np.uint64)
    assert mortie.is_point(points).all()
    got = mortie.children_of(points, 29)
    np.testing.assert_array_equal(got[:, 0], points)
    assert mortie.is_point(got[:, 0]).all()


def test_children_order_zero_parents_and_max_order_target():
    """Order-0 base cells refine, and order 29 is a legal target."""
    bases = _pack(np.arange(12), np.zeros(12, dtype=np.uint8))
    got = mortie.children_of(bases, 3)
    assert got.shape == (12, 64)
    _assert_children_parity(bases, 3)
    # d == 1 into the kernel's max order.
    deep = _pack([3 * (1 << 56), 7 * (1 << 56) + 11], np.full(2, 28, np.uint8))
    got = mortie.children_of(deep, 29)
    assert got.shape == (2, 4)
    _assert_children_parity(deep, 29)


def test_children_empty_and_single():
    got = mortie.children_of([], 6)
    assert got.shape == (0, 1) and got.dtype == np.uint64
    parent = np.atleast_1d(np.asarray(mortie.norm2mort([11], [0], 4), dtype=np.uint64))
    got = mortie.children_of(parent, 7)
    assert got.shape == (1, 64)
    np.testing.assert_array_equal(
        got[0], mortie.generate_morton_children(int(parent[0]), 7)
    )


def test_children_stacked_scalar_loop_is_reproduced_exactly():
    """The consumer's own idiom, replaced verbatim.

    moczarr writes ``np.stack([generate_morton_children(int(w), level) for w in
    words])`` (``dggs.py:310``) with the comment that no vectorized many-parent
    kernel exists.  This is that expression, and the batch must equal it.
    """
    words = _basin_cells(2, order=8, limit=300)
    expected = np.stack([mortie.generate_morton_children(int(w), 10) for w in words])
    np.testing.assert_array_equal(mortie.children_of(words, 10), expected)


# ---------------------------------------------------------------------------
# children_of: layout and error surface
# ---------------------------------------------------------------------------


def test_children_refuse_a_word_finer_than_the_target():
    """A parent finer than ``order`` is refused, naming its index."""
    words = list(_basin_cells(24, order=6, limit=10))
    words.append(_pack([12345], [9])[0])
    with pytest.raises(ValueError, match=r"word 10: is at order 9, finer than"):
        mortie.children_of(np.asarray(words, np.uint64), 7)
    # Word 0 itself being too fine names index 0.
    with pytest.raises(ValueError, match=r"word 0: is at order 9, finer than"):
        mortie.children_of(np.asarray(words[::-1], np.uint64), 7)


def test_children_refuse_mixed_parent_orders():
    """Rows would be ragged, so a disagreeing order is refused by index."""
    words = list(_basin_cells(24, order=6, limit=10))
    words[4] = _pack([123], [5])[0]
    with pytest.raises(ValueError, match=r"word 4: is at order 5 but word 0"):
        mortie.children_of(np.asarray(words, np.uint64), 9)


def test_children_refuse_invalid_words_and_orders():
    words = np.asarray(list(_basin_cells(24, order=6, limit=5)) + [0], np.uint64)
    with pytest.raises(ValueError, match=r"word 5: .*not a decodable morton word"):
        mortie.children_of(words, 8)
    with pytest.raises(ValueError, match="Order must be between 0 and 29"):
        mortie.children_of(words[:5], 30)
    with pytest.raises(ValueError, match="Order must be between 0 and 29"):
        mortie.children_of(words[:5], -1)


def test_children_oversized_result_raises_instead_of_aborting():
    """An unservable result is a catchable ``ValueError``, not a process abort.

    ``vec![0u64; total]`` in the Rust kernel used Rust's *infallible* allocator,
    whose failure path is ``handle_alloc_error`` -> ``abort()`` — SIGABRT, no
    Python traceback, nothing to ``except``.  The scalar loop this op replaces
    raises a catchable ``MemoryError`` at the same sizes, so the batch has to be
    catchable too.  12 order-0 parents at order 25 asks for 96 PiB.
    """
    words = _pack(np.arange(12), np.zeros(12, np.uint8))
    with pytest.raises(ValueError, match="allocation failed"):
        mortie.children_of(words, 25)
    # A plain ``except ValueError`` must see it — the PR #160 lesson, where a
    # PanicException escaped even ``except Exception``.
    caught = None
    try:
        mortie.children_of(words[:1], 29)  # 2 EiB
    except ValueError as exc:
        caught = str(exc)
    assert caught is not None and "allocation failed" in caught
    # The overflow guard above it still answers separately (>= 2**64 elements).
    with pytest.raises(ValueError, match="children each overflows"):
        mortie.children_of(_pack(np.arange(64) % 12, np.zeros(64, np.uint8)), 29)


def test_children_oversized_result_leaves_the_process_alive():
    """The refusal happens in-process: a fresh interpreter exits 0.

    ``pytest.raises`` cannot observe an ``abort()`` — the whole run dies with
    it — so the guarantee that actually matters (the interpreter survives) is
    asserted from outside, on the exit status of a subprocess.
    """
    code = (
        "import numpy as np, mortie\n"
        "from mortie import _rustie\n"
        "w = np.asarray(_rustie.rust_nested2mort(\n"
        "    np.arange(12, dtype=np.uint64), np.zeros(12, np.uint8)), np.uint64)\n"
        "try:\n"
        "    mortie.children_of(w, 25)\n"
        "except ValueError as exc:\n"
        "    print('CAUGHT', 'allocation failed' in str(exc))\n"
        "print('ALIVE')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, f"exit {out.returncode}: {out.stderr}"
    assert "CAUGHT True" in out.stdout
    assert out.stdout.strip().endswith("ALIVE")


def test_children_large_but_servable_result_still_works():
    """The guard refuses only what the allocator refuses.

    100k parents refined by 5 orders is 819 MB — the size the docstring quotes
    as the reason to budget caller-side — and it must still go through, so the
    fallible allocation is not a de facto budget.
    """
    words = _pack(np.arange(100_000) % (12 << 12), np.full(100_000, 6, np.uint8))
    kids = mortie.children_of(words, 11)
    assert kids.shape == (100_000, 1024)
    np.testing.assert_array_equal(
        kids[7], mortie.generate_morton_children(int(words[7]), 11)
    )


def test_children_max_cells_unset_is_byte_identical_to_today():
    """``max_cells=None`` (the default) changes nothing at all.

    The point of the opt-in default is that adopters see no behaviour change,
    so the existing parity fixtures run both ways and must be byte-equal —
    including passing the parameter explicitly as ``None``.
    """
    for order, target in [(6, 8), (9, 11), (4, 4)]:
        words = _basin_cells(24, order=order, limit=500)
        default = mortie.children_of(words, target)
        np.testing.assert_array_equal(
            default, mortie.children_of(words, target, max_cells=None)
        )
        _assert_children_parity(words[:50], target)
    empty = np.asarray([], np.uint64)
    assert mortie.children_of(empty, 6).shape == (0, 1)
    assert mortie.children_of(empty, 6, max_cells=None).shape == (0, 1)


def test_children_max_cells_boundary_in_both_directions():
    """Exactly at the budget passes; one cell over refuses."""
    words = _basin_cells(24, order=6, limit=100)
    exact = len(words) * 4 ** 2  # d = 2 -> 16 children each
    ok = mortie.children_of(words, 8, max_cells=exact)
    np.testing.assert_array_equal(ok, mortie.children_of(words, 8))
    with pytest.raises(ValueError, match=rf"exceeding max_cells={exact - 1}"):
        mortie.children_of(words, 8, max_cells=exact - 1)
    np.testing.assert_array_equal(mortie.children_of(words, 8, max_cells=1 << 40), ok)
    # A zero budget refuses any non-empty refinement but passes an empty batch,
    # which is genuinely zero cells.
    with pytest.raises(ValueError, match="exceeding max_cells=0"):
        mortie.children_of(words, 8, max_cells=0)
    assert mortie.children_of(np.asarray([], np.uint64), 8, max_cells=0).shape == (0, 1)
    with pytest.raises(ValueError, match="max_cells must be non-negative"):
        mortie.children_of(words, 8, max_cells=-1)


def test_children_max_cells_refusal_is_a_plain_value_error():
    """A bare ``except ValueError`` sees the refusal.

    The PR #160 pattern: the analogous ``moc_to_order`` guard had a
    ``PanicException`` escaping even ``except Exception``, so the handler shape
    a consumer actually writes is what gets asserted, not ``pytest.raises``.
    """
    words = _basin_cells(24, order=6, limit=100)
    caught = None
    try:
        mortie.children_of(words, 14, max_cells=1 << 20)
    except ValueError as exc:
        caught = str(exc)
    assert caught is not None
    assert "exceeding max_cells=1048576" in caught
    assert "would generate" in caught
    # And the message says what to do about it, in moc_to_order's wording.
    assert "max_cells=None to proceed" in caught


def test_children_max_cells_outranks_the_overflow_guard():
    """The caller's budget answers before the internal overflow diagnostic.

    64 order-0 parents at order 29 is 2**64 elements — the ``checked_mul``
    guard's case when no budget is set.  With a budget set the budget answers,
    because it is compared in 128-bit and an explicit argument condition is more
    actionable than "the element count does not fit a usize".  Pinned because
    the ordering is deliberate, not incidental.
    """
    words = _pack(np.arange(64) % 12, np.zeros(64, np.uint8))
    with pytest.raises(ValueError, match="children each overflows"):
        mortie.children_of(words, 29)
    with pytest.raises(ValueError, match="exceeding max_cells=1048576"):
        mortie.children_of(words, 29, max_cells=1 << 20)
    # Same ordering ahead of the fallible-allocation refusal.
    twelve = _pack(np.arange(12), np.zeros(12, np.uint8))
    with pytest.raises(ValueError, match="allocation failed"):
        mortie.children_of(twelve, 25)
    with pytest.raises(ValueError, match="exceeding max_cells=1048576"):
        mortie.children_of(twelve, 25, max_cells=1 << 20)


def test_children_max_cells_refuses_the_zagg_d8_shape_catchably():
    """The case that motivated the ruling: a worker refuses instead of dying.

    1M order-6 parents refined to order 14 is the shipped ``d = 8`` shape at
    fleet scale — 488 GiB, which ``try_reserve`` cannot refuse because the OS
    accepts the reservation and kills the process later, mid-write.  With the
    budget a worker would set it comes back as a ``ValueError`` the worker can
    catch and retry at a coarser order.
    """
    words = _pack(np.arange(1_000_000) % (12 << 12), np.full(1_000_000, 6, np.uint8))
    budget = 1 << 27  # 2**27 cells = 1 GiB of uint64, a plausible worker ceiling
    caught = None
    try:
        mortie.children_of(words, 14, max_cells=budget)
    except ValueError as exc:
        caught = str(exc)
    assert caught is not None
    assert f"exceeding max_cells={budget}" in caught
    assert "would generate 65536000000 cells" in caught
    # The same shape under the budget still works, so the guard is a ceiling and
    # not a refusal of the shape itself.
    ok = mortie.children_of(words[:1000], 14, max_cells=budget)
    assert ok.shape == (1000, 65536)


@pytest.mark.parametrize(
    "offender", [0, 1, 7, CHUNK - 1, CHUNK, CHUNK + 1, 2 * CHUNK + 5, 3 * CHUNK - 1]
)
def test_children_lowest_index_holds_across_chunk_boundaries(offender):
    """The named word is the lowest-index offender, wherever it sits."""
    n = 3 * CHUNK
    words = _pack(np.arange(n) % 4096, np.full(n, 6, dtype=np.uint8))
    words[offender] = _pack([12345], [9])[0]
    for _ in range(5):
        with pytest.raises(ValueError, match=rf"word {offender}: "):
            mortie.children_of(words, 8)


def test_children_lowest_index_wins_over_later_offenders():
    n = 3 * CHUNK
    words = _pack(np.arange(n) % 4096, np.full(n, 6, dtype=np.uint8))
    bad = _pack([12345], [9])[0]
    offenders = [3, CHUNK - 1, CHUNK, 2 * CHUNK + 5, n - 1]
    healthy = words.copy()
    for off in offenders:
        words[off] = bad
    for i, expect in enumerate(offenders):
        for _ in range(3):
            with pytest.raises(ValueError, match=rf"word {expect}: "):
                mortie.children_of(words, 8)
        words[offenders[i]] = healthy[offenders[i]]
    assert mortie.children_of(words, 8).shape == (n, 16)


def test_children_deterministic_across_runs():
    rng = np.random.default_rng(11)
    words = _pack(rng.integers(0, 12 << 12, 20_000), np.full(20_000, 6, np.uint8))
    first = mortie.children_of(words, 8)
    for _ in range(9):
        np.testing.assert_array_equal(mortie.children_of(words, 8), first)


# ---------------------------------------------------------------------------
# GIL release
# ---------------------------------------------------------------------------


def _gil_probe(call):
    """Run *call* 20x with a pure-Python counter thread and return its progress."""
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
            call()
        return counter[0] - pre
    finally:
        stop.set()
        b.join()


def test_gil_released_during_common_ancestors():
    """A pure-Python counter thread keeps running during a large batch call.

    Same instrument as ``test_threading.test_gil_released_during_rust_compute``:
    with the GIL held for the whole call the counter cannot advance.
    """
    rng = np.random.default_rng(5)
    values, offsets = _ragged(_random_groups(rng, 40_000, order=9, size=(2, 6)))
    progressed = _gil_probe(lambda: mortie.common_ancestors(values, offsets))
    assert progressed > 1000, (
        f"Python thread made ~no progress during common_ancestors ({progressed}); "
        "the GIL was likely held (allow_threads not in effect)"
    )


def test_gil_released_during_children_of():
    rng = np.random.default_rng(6)
    words = _pack(rng.integers(0, 12 << 12, 20_000), np.full(20_000, 6, np.uint8))
    progressed = _gil_probe(lambda: mortie.children_of(words, 10))
    assert progressed > 1000, (
        f"Python thread made ~no progress during children_of ({progressed}); "
        "the GIL was likely held (allow_threads not in effect)"
    )

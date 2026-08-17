"""Tests for the polymorphic (one-function-per-operation) API (issue #187).

Each operation is meant to have **one** public entry point whose input shape
selects the form: the bare call is the single-item form, and passing the ragged
``offsets`` keyword makes the same call the batch form.  The contract asserted
here is that the polymorphic form is not a second semantics -- it is
byte-identical both to the plural sibling it delegates to and to a Python loop
over the single-item form -- and that the error surface passes through
unchanged (the batch refusals still name the lowest-index offender, and still
arrive as catchable :class:`ValueError`).
"""

import numpy as np
import pytest

import mortie

# ---------------------------------------------------------------------------
# Fixtures: a small ragged column of MOCs with a mixed-order, empty and
# single-cell item, so every slot shape the batch admits is exercised.
# ---------------------------------------------------------------------------


@pytest.fixture
def column():
    """Ragged (values, offsets) column of four MOCs, one of them empty."""
    base = mortie.norm2mort(np.arange(16), np.zeros(16, dtype=int), 2)
    mocs = [
        base[:4],                       # four siblings (compacts to a parent)
        base[5:9],                      # a straddling run
        base[:0],                       # empty MOC -- legal, keeps its slot
        base[10:11],                    # single cell
    ]
    values = np.concatenate(mocs).astype(np.uint64)
    offsets = np.cumsum([0] + [len(m) for m in mocs]).astype(np.int64)
    return values, offsets


def _slices(values, offsets):
    return [values[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


# ---------------------------------------------------------------------------
# moc_to_order
# ---------------------------------------------------------------------------


def test_moc_to_order_offsets_matches_plural(column):
    values, offsets = column
    got_v, got_o = mortie.moc_to_order(values, 4, offsets=offsets)
    want_v, want_o = mortie.mocs_to_orders(values, offsets, 4)
    np.testing.assert_array_equal(got_v, want_v)
    np.testing.assert_array_equal(got_o, want_o)


def test_moc_to_order_offsets_matches_scalar_loop(column):
    values, offsets = column
    got_v, got_o = mortie.moc_to_order(values, 4, offsets=offsets)
    for i, moc in enumerate(_slices(values, offsets)):
        np.testing.assert_array_equal(
            got_v[got_o[i]:got_o[i + 1]], mortie.moc_to_order(moc, 4)
        )


def test_moc_to_order_offsets_none_is_the_scalar_form(column):
    values, offsets = column
    moc = values[offsets[0]:offsets[1]]
    np.testing.assert_array_equal(
        mortie.moc_to_order(moc, 4, offsets=None), mortie.moc_to_order(moc, 4)
    )


def test_moc_to_order_offsets_keeps_per_item_budget_refusal(column):
    values, offsets = column
    with pytest.raises(ValueError, match="moc 0"):
        mortie.moc_to_order(values, 12, max_cells=8, offsets=offsets)


def test_moc_to_order_offsets_is_keyword_only(column):
    values, offsets = column
    with pytest.raises(TypeError):
        mortie.moc_to_order(values, 4, None, offsets)


# ---------------------------------------------------------------------------
# moc_and / moc_intersects
# ---------------------------------------------------------------------------


def test_moc_and_offsets_matches_plural(column):
    values, offsets = column
    shared = mortie.norm2mort(np.arange(8), np.zeros(8, dtype=int), 2)
    got_v, got_o = mortie.moc_and(shared, values, offsets=offsets)
    want_v, want_o = mortie.mocs_and(shared, values, offsets)
    np.testing.assert_array_equal(got_v, want_v)
    np.testing.assert_array_equal(got_o, want_o)


def test_moc_and_offsets_matches_scalar_loop(column):
    values, offsets = column
    shared = mortie.norm2mort(np.arange(8), np.zeros(8, dtype=int), 2)
    got_v, got_o = mortie.moc_and(shared, values, offsets=offsets)
    for i, moc in enumerate(_slices(values, offsets)):
        np.testing.assert_array_equal(
            got_v[got_o[i]:got_o[i + 1]], mortie.moc_and(shared, moc)
        )


def test_moc_intersects_offsets_matches_plural_and_loop(column):
    values, offsets = column
    shared = mortie.norm2mort(np.arange(8), np.zeros(8, dtype=int), 2)
    got = mortie.moc_intersects(shared, values, offsets=offsets)
    np.testing.assert_array_equal(
        got, mortie.mocs_intersect(shared, values, offsets)
    )
    np.testing.assert_array_equal(
        got,
        [mortie.moc_intersects(shared, m) for m in _slices(values, offsets)],
    )


def test_moc_intersects_offsets_agrees_with_moc_and_slots(column):
    values, offsets = column
    shared = mortie.norm2mort(np.arange(8), np.zeros(8, dtype=int), 2)
    hits = mortie.moc_intersects(shared, values, offsets=offsets)
    _, out_off = mortie.moc_and(shared, values, offsets=offsets)
    np.testing.assert_array_equal(hits, np.diff(out_off) > 0)


def test_moc_and_offsets_rejects_offsets_not_covering_values(column):
    values, offsets = column
    shared = values[:2]
    with pytest.raises(ValueError):
        mortie.moc_and(shared, values, offsets=offsets[:-1])


# ---------------------------------------------------------------------------
# common_ancestor (and its moc_min alias)
# ---------------------------------------------------------------------------


def test_common_ancestor_offsets_matches_plural_and_loop(column):
    values, offsets = column
    # The empty slot has no common ancestor, so reduce over the non-empty ones.
    keep = np.array([0, 4, 8, 9], dtype=np.int64)
    got = mortie.common_ancestor(values, offsets=keep)
    np.testing.assert_array_equal(got, mortie.common_ancestors(values, keep))
    np.testing.assert_array_equal(
        got, [mortie.common_ancestor(m) for m in _slices(values, keep)]
    )


def test_moc_min_alias_is_polymorphic_too(column):
    values, _ = column
    keep = np.array([0, 4, 8, 9], dtype=np.int64)
    np.testing.assert_array_equal(
        mortie.moc_min(values, offsets=keep),
        mortie.common_ancestor(values, offsets=keep),
    )


def test_common_ancestor_offsets_refusal_names_the_group(column):
    values, offsets = column
    with pytest.raises(ValueError, match="group 2|2:"):
        mortie.common_ancestor(values, offsets=offsets)


# ---------------------------------------------------------------------------
# toc_reduce
# ---------------------------------------------------------------------------


def test_toc_reduce_offsets_matches_plural_and_loop():
    words = np.array(
        [mortie.time2toc(t) for t in (10**9, 2 * 10**9, 5 * 10**9, 7 * 10**9)],
        dtype=np.uint64,
    )
    offsets = np.array([0, 2, 3, 4], dtype=np.int64)
    got = mortie.toc_reduce(words, offsets=offsets)
    np.testing.assert_array_equal(got, mortie.tocs_reduce(words, offsets))
    np.testing.assert_array_equal(
        got, [mortie.toc_reduce(g) for g in _slices(words, offsets)]
    )


def test_toc_reduce_offsets_none_is_the_whole_array_form():
    words = np.array([mortie.time2toc(10**9), mortie.time2toc(3 * 10**9)],
                     dtype=np.uint64)
    assert mortie.toc_reduce(words, offsets=None) == mortie.toc_reduce(words)


def test_toc_reduce_offsets_refuses_an_empty_group():
    words = np.array([mortie.time2toc(10**9)], dtype=np.uint64)
    with pytest.raises(ValueError):
        mortie.toc_reduce(words, offsets=np.array([0, 0, 1], dtype=np.int64))


# ---------------------------------------------------------------------------
# decimal_to_word
# ---------------------------------------------------------------------------


def test_decimal_to_word_array_matches_plural_and_loop():
    ids = np.array(["-31123", "12341", "6444"])
    got = mortie.decimal_to_word(ids)
    np.testing.assert_array_equal(got, mortie.decimals_to_words(ids))
    np.testing.assert_array_equal(
        got, [mortie.decimal_to_word(s) for s in ids]
    )
    assert got.dtype == np.uint64


def test_decimal_to_word_preserves_shape_and_scalar_form():
    ids = np.array([["-31123", "12341"], ["6444", "12341"]])
    assert mortie.decimal_to_word(ids).shape == (2, 2)
    # A bare str stays scalar -- not a 0-d array.
    assert isinstance(mortie.decimal_to_word("12341"), np.uint64)
    assert mortie.decimal_to_word("12341", dtype=int) == int(
        mortie.decimal_to_word("12341")
    )


def test_decimal_to_word_rejects_non_uint64_dtype_for_arrays():
    with pytest.raises(TypeError, match="always uint64"):
        mortie.decimal_to_word(["12341"], dtype=int)


# ---------------------------------------------------------------------------
# generate_morton_children
# ---------------------------------------------------------------------------


def test_generate_morton_children_array_matches_plural_and_loop():
    parents = mortie.norm2mort(np.arange(4), np.zeros(4, dtype=int), 3)
    got = mortie.generate_morton_children(parents, 5)
    np.testing.assert_array_equal(got, mortie.children_of(parents, 5))
    np.testing.assert_array_equal(
        got, [mortie.generate_morton_children(p, 5) for p in parents]
    )
    assert got.shape == (4, 16)


def test_generate_morton_children_scalar_stays_one_dimensional():
    parent = mortie.norm2mort(0, 0, 3)
    kids = mortie.generate_morton_children(parent, 5)
    assert kids.ndim == 1 and kids.shape == (16,)


def test_generate_morton_children_length_one_array_is_a_row():
    """A length-1 array used to describe only its first element (silently)."""
    # norm2mort squeezes a length-1 input to a scalar, so build the array here.
    parents = np.asarray([mortie.norm2mort(0, 0, 3)], dtype=np.uint64)
    got = mortie.generate_morton_children(parents, 5)
    assert got.shape == (1, 16)
    np.testing.assert_array_equal(
        got[0], mortie.generate_morton_children(parents[0], 5)
    )


def test_generate_morton_children_array_honours_max_cells():
    parents = mortie.norm2mort(np.arange(4), np.zeros(4, dtype=int), 3)
    with pytest.raises(ValueError):
        mortie.generate_morton_children(parents, 8, max_cells=4)

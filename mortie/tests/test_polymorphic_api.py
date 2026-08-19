"""Tests for the polymorphic (one-function-per-operation) API (issue #187).

Each operation is meant to have **one** public entry point whose input shape
selects the form: the bare call is the single-item form, and passing the ragged
``offsets`` keyword makes the same call the batch form.  The contract asserted
here is that the polymorphic form is not a second semantics -- it is
byte-identical both to the private batch kernel it delegates to (the plural
names retired in phase 4) and to a Python loop over the single-item form -- and that the error surface passes through
unchanged (the batch refusals still name the lowest-index offender, and still
arrive as catchable :class:`ValueError`).
"""

import pathlib
import re

import numpy as np
import pytest

import mortie
import mortie.arrow
from mortie._toc import _tocs_reduce
from mortie.batch import (
    _children_of,
    _common_ancestors,
    _from_wkbs,
    _mocs_and,
    _mocs_intersect,
    _mocs_to_orders,
)
from mortie.morton_index import _decimals_to_words

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
    want_v, want_o = _mocs_to_orders(values, offsets, 4)
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
    with pytest.raises(ValueError, match=r"^moc 0: moc_to_order would densify"):
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
    want_v, want_o = _mocs_and(shared, values, offsets)
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
        got, _mocs_intersect(shared, values, offsets)
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
    with pytest.raises(ValueError, match=r"offsets must end at the value count"):
        mortie.moc_and(shared, values, offsets=offsets[:-1])


# ---------------------------------------------------------------------------
# common_ancestor (and its moc_min alias)
# ---------------------------------------------------------------------------


def test_common_ancestor_offsets_matches_plural_and_loop(column):
    values, offsets = column
    # The empty slot has no common ancestor, so reduce over the non-empty ones.
    keep = np.array([0, 4, 8, 9], dtype=np.int64)
    got = mortie.common_ancestor(values, offsets=keep)
    np.testing.assert_array_equal(got, _common_ancestors(values, keep))
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
    with pytest.raises(
        ValueError, match=r"^group 2: empty input has no common ancestor"
    ):
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
    np.testing.assert_array_equal(got, _tocs_reduce(words, offsets))
    np.testing.assert_array_equal(
        got, [mortie.toc_reduce(g) for g in _slices(words, offsets)]
    )


def test_toc_reduce_offsets_none_is_the_whole_array_form():
    words = np.array([mortie.time2toc(10**9), mortie.time2toc(3 * 10**9)],
                     dtype=np.uint64)
    assert mortie.toc_reduce(words, offsets=None) == mortie.toc_reduce(words)


def test_toc_reduce_offsets_refuses_an_empty_group():
    words = np.array([mortie.time2toc(10**9)], dtype=np.uint64)
    # The kernel says "tocs_reduce"; the public wrapper renames the retired
    # delegate to the surviving entry point (issue #187, phase 4).
    with pytest.raises(ValueError, match=r"^group 0: toc_reduce of an empty"):
        mortie.toc_reduce(words, offsets=np.array([0, 0, 1], dtype=np.int64))


# ---------------------------------------------------------------------------
# decimal_to_word
# ---------------------------------------------------------------------------


def test_decimal_to_word_array_matches_plural_and_loop():
    ids = np.array(["-31123", "12341", "6444"])
    got = mortie.decimal_to_word(ids)
    np.testing.assert_array_equal(got, _decimals_to_words(ids))
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
    np.testing.assert_array_equal(got, _children_of(parents, 5))
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
    # A length-1 array stays an array through norm2mort (issue #187, phase 5),
    # so this is a genuine one-parent array rather than a re-boxed scalar.
    parents = np.asarray([mortie.norm2mort(0, 0, 3)], dtype=np.uint64)
    got = mortie.generate_morton_children(parents, 5)
    assert got.shape == (1, 16)
    np.testing.assert_array_equal(
        got[0], mortie.generate_morton_children(parents[0], 5)
    )


def test_generate_morton_children_rejects_higher_rank():
    parents = mortie.norm2mort(np.arange(4), np.zeros(4, dtype=int), 3)
    with pytest.raises(ValueError, match=r"scalar or 1-D, got 2-D"):
        mortie.generate_morton_children(parents.reshape(2, 2), 5)


def test_generate_morton_children_scalar_honours_max_cells():
    """The budget is refused pre-emptively in the scalar form too."""
    parent = mortie.norm2mort(0, 0, 3)
    with pytest.raises(ValueError, match=r"exceeding max_cells=4"):
        mortie.generate_morton_children(parent, 8, max_cells=4)
    assert mortie.generate_morton_children(parent, 5, max_cells=16).size == 16


def test_decimal_to_word_zero_dim_input_returns_a_scalar():
    """0-d in -> scalar out, the numpy semantics the API is meant to follow."""
    got = mortie.decimal_to_word(np.array("12341"))
    assert isinstance(got, np.uint64) and np.ndim(got) == 0
    assert got == mortie.decimal_to_word("12341")


def test_decimal_to_word_array_refuses_morton_index_scalar_dtype():
    """MortonIndexScalar is a uint64 subclass, so it must be ruled out first."""
    from mortie.morton_index import MortonIndexScalar

    with pytest.raises(TypeError, match=r"always uint64"):
        mortie.decimal_to_word(["12341"], dtype=MortonIndexScalar)


# ---------------------------------------------------------------------------
# Layout and budget edges shared by every ``offsets`` form
# ---------------------------------------------------------------------------


def test_offsets_accepts_a_plain_list(column):
    values, offsets = column
    got_v, got_o = mortie.moc_to_order(values, 4, offsets=list(map(int, offsets)))
    want_v, want_o = mortie.moc_to_order(values, 4, offsets=offsets)
    np.testing.assert_array_equal(got_v, want_v)
    np.testing.assert_array_equal(got_o, want_o)


def test_offsets_single_group_and_empty_column(column):
    values, _ = column
    one = np.array([0, len(values)], dtype=np.int64)
    got_v, got_o = mortie.moc_to_order(values, 4, offsets=one)
    np.testing.assert_array_equal(got_v, mortie.moc_to_order(values, 4))
    np.testing.assert_array_equal(got_o, [0, len(got_v)])
    # A column of zero MOCs is legal and yields nothing.
    empty_v, empty_o = mortie.moc_to_order(
        values[:0], 4, offsets=np.array([0], dtype=np.int64)
    )
    assert empty_v.size == 0
    np.testing.assert_array_equal(empty_o, [0])


def test_offsets_rejects_non_monotone_layout(column):
    values, _ = column
    bad = np.array([0, 5, 2, len(values)], dtype=np.int64)
    with pytest.raises(ValueError):
        mortie.moc_to_order(values, 4, offsets=bad)


def test_offsets_honours_max_cells_none(column):
    """``max_cells=None`` opts the whole column out, as it does one MOC."""
    values, offsets = column
    got_v, _ = mortie.moc_to_order(values, 12, max_cells=None, offsets=offsets)
    assert got_v.size > (1 << 20)


def test_moc_and_batch_operands_are_not_interchangeable(column):
    """``a`` is the shared cover and ``b`` the column -- swapping asks a
    different question, which is why the batch form is not commutative."""
    values, offsets = column
    shared = mortie.norm2mort(np.arange(8), np.zeros(8, dtype=int), 2)
    _, out_off = mortie.moc_and(shared, values, offsets=offsets)
    # Same group count, so the swap is silently well-formed rather than an error.
    swapped_off = np.array([0, 2, 4, 6, len(shared)], dtype=np.int64)
    _, other = mortie.moc_and(values, shared, offsets=swapped_off)
    assert len(out_off) == len(other)
    assert not np.array_equal(np.diff(out_off), np.diff(other))


def test_generate_morton_children_array_honours_max_cells():
    parents = mortie.norm2mort(np.arange(4), np.zeros(4, dtype=int), 3)
    with pytest.raises(ValueError, match=r"exceeding max_cells=4"):
        mortie.generate_morton_children(parents, 8, max_cells=4)


# ---------------------------------------------------------------------------
# Phase 4: the retirement (issue #187, ruled 2026-08-19) -- the plural names
# are gone outright, and the refusal surface names the survivors.
# ---------------------------------------------------------------------------

_RETIRED = (
    "mocs_to_orders", "mocs_and", "mocs_intersect", "common_ancestors",
    "tocs_reduce", "decimals_to_words", "children_of", "from_wkbs",
    "morton_coverage_moc",
)


def test_retired_names_are_gone_from_the_package_root():
    for name in _RETIRED:
        assert not hasattr(mortie, name), name
        assert name not in mortie.__all__, name


def test_retired_names_are_gone_from_their_modules():
    import mortie._toc
    import mortie.batch
    import mortie.coverage
    import mortie.morton_index

    for mod, name in (
        (mortie.batch, "from_wkbs"), (mortie.batch, "mocs_to_orders"),
        (mortie.batch, "mocs_and"), (mortie.batch, "mocs_intersect"),
        (mortie.batch, "common_ancestors"), (mortie.batch, "children_of"),
        (mortie._toc, "tocs_reduce"),
        (mortie.morton_index, "decimals_to_words"),
        (mortie.coverage, "morton_coverage_moc"),
        (mortie.arrow, "from_wkbs"),
    ):
        assert not hasattr(mod, name), f"{mod.__name__}.{name}"


def test_arrow_skin_kept_the_surviving_name():
    # The pyarrow skin renamed with the core (a column is inherently the
    # batch shape, so the one surviving name needs no dispatch there).
    assert callable(mortie.arrow.from_wkb)


def test_generate_morton_children_refusals_name_the_survivor():
    parents = mortie.norm2mort(np.arange(2), np.zeros(2, dtype=int), 4)
    with pytest.raises(ValueError, match=r"generate_morton_children only refines"):
        mortie.generate_morton_children(parents, 2)
    with pytest.raises(
        ValueError, match=r"^generate_morton_children would generate"
    ):
        mortie.generate_morton_children(parents, 14, max_cells=10)
    # The third kernel message carrying the retired name goes through the same
    # respelling and is just as public.
    mixed = np.concatenate([
        np.atleast_1d(mortie.norm2mort(np.arange(1), np.zeros(1, dtype=int), 4)),
        np.atleast_1d(mortie.norm2mort(np.arange(1), np.zeros(1, dtype=int), 5)),
    ])
    with pytest.raises(
        ValueError,
        match=r"generate_morton_children returns a dense \(n, 4\*\*d\) block",
    ):
        mortie.generate_morton_children(mixed, 8)


# ---------------------------------------------------------------------------
# from_wkb: the phase-4 collapse (issue #187 question 4, the "(a+)" design).
# Dispatch is exactly: offsets= => packed column; list/tuple/object-ndarray =>
# sequence batch; bytes / hex str / bytearray / memoryview / uint8 ndarray =>
# one blob.  moc is a tri-state whose default keeps both historical calls.
# ---------------------------------------------------------------------------


def _tri(lat0, lon0):
    """One small triangle as a WKB blob (built by hand, no backend)."""
    import struct

    pts = [(lon0, lat0), (lon0, lat0 + 5.0), (lon0 + 5.0, lat0 + 2.0),
           (lon0, lat0)]
    blob = struct.pack("<BII", 1, 3, 1) + struct.pack("<I", len(pts))
    for x, y in pts:
        blob += struct.pack("<dd", x, y)
    return blob


@pytest.fixture
def blobs():
    return [_tri(40.0, -120.0), _tri(10.0, -80.0)]


def test_from_wkb_scalar_default_is_the_flat_cover(blobs):
    """The bare scalar call is byte-identical to the old ``from_wkb``."""
    flat = mortie.from_wkb(blobs[0], order=6)
    assert flat.ndim == 1 and flat.dtype == np.uint64
    assert np.array_equal(flat, mortie.from_wkb(blobs[0], order=6, moc=False))
    # The old signature's positional moc slot survives the collapse.
    np.testing.assert_array_equal(
        mortie.from_wkb(blobs[0], 6, True),
        mortie.from_wkb(blobs[0], order=6, moc=True),
    )


def test_from_wkb_sequence_batch_matches_kernel_and_loop(blobs):
    """The batch call is byte-identical to the old ``from_wkbs``."""
    got_v, got_o = mortie.from_wkb(blobs, order=6)
    want_v, want_o = _from_wkbs(blobs, order=6)
    np.testing.assert_array_equal(got_v, want_v)
    np.testing.assert_array_equal(got_o, want_o)
    for i, blob in enumerate(blobs):
        np.testing.assert_array_equal(
            got_v[got_o[i]:got_o[i + 1]],
            mortie.from_wkb(blob, order=6, moc=True),
        )


def test_from_wkb_batch_default_and_moc_true_agree(blobs):
    got = mortie.from_wkb(blobs, order=6)
    explicit = mortie.from_wkb(blobs, order=6, moc=True)
    np.testing.assert_array_equal(got[0], explicit[0])
    np.testing.assert_array_equal(got[1], explicit[1])
    # tuple input is a sequence batch too.
    np.testing.assert_array_equal(
        mortie.from_wkb(tuple(blobs), order=6)[0], got[0]
    )
    # ... as is a numpy object array (the pandas .to_numpy() shape).
    arr = np.empty(len(blobs), dtype=object)
    arr[:] = blobs
    np.testing.assert_array_equal(mortie.from_wkb(arr, order=6)[0], got[0])


def test_from_wkb_batch_refuses_moc_false(blobs):
    with pytest.raises(ValueError, match=r"no flat-cover form"):
        mortie.from_wkb(blobs, order=6, moc=False)


def test_from_wkb_scalar_spellings_stay_scalar(blobs):
    """A single blob in any buffer spelling is one blob, never a batch."""
    flat = mortie.from_wkb(blobs[0], order=6)
    for spelling in (
        blobs[0].hex(), bytearray(blobs[0]), memoryview(blobs[0]),
        np.frombuffer(blobs[0], dtype=np.uint8),
    ):
        got = mortie.from_wkb(spelling, order=6)
        assert isinstance(got, np.ndarray)  # a cover, not a (values, off) pair
        np.testing.assert_array_equal(got, flat)


def test_from_wkb_packed_column_matches_the_sequence_form(blobs):
    packed = np.frombuffer(b"".join(blobs), dtype=np.uint8)
    offsets = np.cumsum([0] + [len(b) for b in blobs]).astype(np.int64)
    got_v, got_o = mortie.from_wkb(packed, order=6, offsets=offsets)
    want_v, want_o = mortie.from_wkb(blobs, order=6)
    np.testing.assert_array_equal(got_v, want_v)
    np.testing.assert_array_equal(got_o, want_o)
    # bytes and bytearray spellings of the packed buffer work identically.
    np.testing.assert_array_equal(
        mortie.from_wkb(b"".join(blobs), order=6, offsets=offsets)[0], got_v
    )


def test_from_wkb_packed_column_layout_errors(blobs):
    packed = np.frombuffer(b"".join(blobs), dtype=np.uint8)
    n0, n1 = len(blobs[0]), len(blobs[1])
    with pytest.raises(ValueError, match=r"^offsets must start at 0"):
        mortie.from_wkb(packed, order=6, offsets=[1, n0, n0 + n1])
    with pytest.raises(
        ValueError, match=r"^blob 1: offsets must be monotonically"
    ):
        mortie.from_wkb(packed, order=6, offsets=[0, n0, 2, n0 + n1])
    with pytest.raises(ValueError, match=r"^offsets must end at the byte count"):
        mortie.from_wkb(packed, order=6, offsets=[0, n0])
    with pytest.raises(ValueError, match=r"^blob 0: offset .* exceeds"):
        mortie.from_wkb(packed, order=6,
                        offsets=[0, n0 + n1 + 1, n0 + n1 + 2])
    with pytest.raises(TypeError, match=r"packed, contiguous buffer"):
        mortie.from_wkb(blobs, order=6, offsets=[0, n0, n0 + n1])
    # A wider-item buffer is refused by item size, as the scalar path refuses
    # it -- cast("B") alone would have reinterpreted it byte-wise.
    packed_bytes = b"".join(blobs)
    for dtype in (np.float64, np.int32):
        wide = np.frombuffer(
            packed_bytes + b"\0" * (-len(packed_bytes) % np.dtype(dtype).itemsize),
            dtype=dtype,
        )
        with pytest.raises(TypeError, match=r"got one of \d+-byte items"):
            mortie.from_wkb(wide, order=6, offsets=[0, n0, wide.nbytes])
    # An offset past int64 is a ValueError like every other offset refusal,
    # not the OverflowError numpy would raise from the coercion.
    with pytest.raises(ValueError, match=r"offsets must fit in int64"):
        mortie.from_wkb(packed, order=6, offsets=[0, 10 ** 19, n0 + n1])


def test_from_wkb_refuses_a_non_bool_moc(blobs):
    """A positionally-migrated ``tolerance`` must not bind to ``moc`` silently.

    ``from_wkbs(blobs, order, tol)`` respelled as ``from_wkb(blobs, order,
    tol)`` used to read the float as "MOC requested" and drop the tolerance,
    returning a much finer cover with no error at all.
    """
    for form in (blobs, blobs[0]):
        with pytest.raises(TypeError, match=r"third positional\s+is moc"):
            mortie.from_wkb(form, 8, 0.5)
    # The bool spellings the tri-state is made of keep working untouched.
    np.testing.assert_array_equal(
        mortie.from_wkb(blobs[0], 8, True),
        mortie.from_wkb(blobs[0], 8, np.True_),
    )
    assert mortie.from_wkb(blobs[0], 8, False).ndim == 1
    assert mortie.from_wkb(blobs, 8, np.True_)[1].size == len(blobs) + 1


def test_from_wkb_batch_dispatch_is_exhaustive(blobs):
    """The ruled dispatch does not sniff wider: other containers are refused.

    ``from_wkbs`` iterated anything; ``from_wkb`` reads exactly ``list`` /
    ``tuple`` / object-``ndarray`` as a batch (issue #187, ruled), so a
    generator or a bytes-dtype array lands on the scalar path and is named
    there.  Materializing is the documented migration.
    """
    for narrowed in (np.array(blobs), (b for b in blobs)):
        with pytest.raises(TypeError, match=r"WKB input must be"):
            mortie.from_wkb(narrowed, order=6)
    want_v, want_o = mortie.from_wkb(blobs, order=6)
    for materialized in (np.array(blobs).astype(object), list(iter(blobs))):
        got_v, got_o = mortie.from_wkb(materialized, order=6)
        np.testing.assert_array_equal(got_v, want_v)
        np.testing.assert_array_equal(got_o, want_o)


def test_from_wkb_batch_routes_through_the_chunked_kernel(blobs, monkeypatch):
    """Both batch forms reuse the byte-capped chunking machinery verbatim."""
    import mortie.batch as batch
    import mortie.geometry  # noqa: F401  (the module under patch's importer)

    calls = []
    real = batch._from_wkbs

    def spy(entries, **kwargs):
        calls.append((list(entries), kwargs))
        return real(entries, **kwargs)

    monkeypatch.setattr(batch, "_from_wkbs", spy)
    mortie.from_wkb(blobs, order=6, tolerance=2.0, normalize=False,
                    latitude="geodetic-spherical")
    packed = np.frombuffer(b"".join(blobs), dtype=np.uint8)
    offsets = np.cumsum([0] + [len(b) for b in blobs]).astype(np.int64)
    mortie.from_wkb(packed, order=6, offsets=offsets, tolerance=2.0,
                    normalize=False, latitude="geodetic-spherical")
    assert len(calls) == 2
    # The packed form hands the kernel zero-copy views of the caller's buffer.
    assert all(isinstance(e, memoryview) for e in calls[1][0])
    # ...and every knob reaches it intact, from both forms: dropping tolerance
    # or max_cells on the floor has to fail here.
    assert [kw for _, kw in calls] == [
        dict(order=6, tolerance=2.0, max_cells=None, normalize=False,
             latitude="geodetic-spherical")
    ] * 2


def test_from_wkb_batch_forwards_every_coverage_knob(blobs):
    """Both batch forms answer exactly as the kernel does, under each knob."""
    packed = b"".join(blobs)
    offsets = np.cumsum([0] + [len(b) for b in blobs]).astype(np.int64)
    for kw in (dict(tolerance=2.0), dict(max_cells=8), dict(normalize=False),
               dict(latitude="geodetic-spherical")):
        want_v, want_o = _from_wkbs(blobs, order=7, **kw)
        for got_v, got_o in (
            mortie.from_wkb(blobs, order=7, **kw),
            mortie.from_wkb(packed, order=7, offsets=offsets, **kw),
        ):
            np.testing.assert_array_equal(got_v, want_v)
            np.testing.assert_array_equal(got_o, want_o)


def test_from_wkb_batch_coverage_knobs_bind_behaviourally(blobs):
    """Not just parity: the budget and the tolerance really coarsen the cover."""
    packed = b"".join(blobs)
    offsets = np.cumsum([0] + [len(b) for b in blobs]).astype(np.int64)
    loose = mortie.from_wkb(blobs, order=7)[0]
    for tight in (mortie.from_wkb(blobs, order=7, max_cells=8)[0],
                  mortie.from_wkb(packed, order=7, offsets=offsets,
                                  max_cells=8)[0],
                  mortie.from_wkb(blobs, order=7, tolerance=2.0)[0]):
        assert tight.size < loose.size


def test_from_wkb_offsets_is_keyword_only(blobs):
    packed = np.frombuffer(b"".join(blobs), dtype=np.uint8)
    offsets = np.cumsum([0] + [len(b) for b in blobs]).astype(np.int64)
    with pytest.raises(
        TypeError, match=r"takes from 1 to 6 positional arguments but 7"
    ):
        mortie.from_wkb(packed, 6, None, True, None, None, offsets)


# ---------------------------------------------------------------------------
# Phase 5: the two ruled folds (issue #187 questions 5 and 6).
# ---------------------------------------------------------------------------


def test_norm2mort_keeps_a_length_one_array_an_array():
    """array in -> array out, the endorsed numpy semantics (question 5)."""
    got = mortie.norm2mort(np.arange(1), np.zeros(1, dtype=int), 3)
    assert np.shape(got) == (1,)
    # ... and it is the same word the scalar form returns, just boxed.
    scalar = mortie.norm2mort(0, 0, 3)
    assert np.ndim(scalar) == 0 and isinstance(scalar, np.uint64)
    assert int(got[0]) == int(scalar)


def test_norm2mort_form_follows_input_rank_not_size():
    """Both operands scalar -> scalar; either one an array -> array."""
    assert np.ndim(mortie.norm2mort(0, 0, 3)) == 0
    assert np.ndim(mortie.norm2mort(np.array(0), np.array(0), 3)) == 0  # 0-d
    assert np.shape(mortie.norm2mort([0], [0], 3)) == (1,)
    # A scalar broadcasts against an array operand, keeping the array form.
    assert np.shape(mortie.norm2mort(0, [0, 1, 2], 3)) == (3,)
    assert np.shape(mortie.norm2mort([0, 1, 2], 0, 3)) == (3,)


def test_mort2norm_form_follows_input_rank_not_size():
    """The inverse follows the same rule, so the round trip keeps its form."""
    assert np.shape(mortie.mort2norm(mortie.norm2mort([146], [9], 6))[0]) == (1,)
    assert np.shape(mortie.mort2norm(mortie.norm2mort([146], [9], 6))[1]) == (1,)
    scalars = mortie.mort2norm(mortie.norm2mort(146, 9, 6))
    assert np.ndim(scalars[0]) == 0 and np.ndim(scalars[1]) == 0
    # A 0-d word is a scalar; a length-1 array is not.
    word = mortie.norm2mort(146, 9, 6)
    assert np.ndim(mortie.mort2norm(np.asarray(word))[0]) == 0
    assert np.shape(mortie.mort2norm(np.atleast_1d(word))[0]) == (1,)
    # `order` is a plain int in every form -- the words share one order.
    for got in (mortie.mort2norm(word), mortie.mort2norm(np.atleast_1d(word))):
        assert isinstance(got[2], int)


def test_mort2norm_round_trip_is_form_preserving_both_ways():
    """`norm2mort` and `mort2norm` agree on what a length-1 array is."""
    array_word = mortie.norm2mort([146], [9], 6)
    n, p, o = mortie.mort2norm(array_word)
    assert np.shape(n) == (1,) and np.shape(p) == (1,)
    np.testing.assert_array_equal(mortie.norm2mort(n, p, o), array_word)
    assert np.shape(mortie.norm2mort(n, p, o)) == (1,)

    scalar_word = mortie.norm2mort(146, 9, 6)
    n, p, o = mortie.mort2norm(scalar_word)
    assert np.ndim(n) == 0 and np.ndim(p) == 0
    assert mortie.norm2mort(n, p, o) == scalar_word
    assert np.ndim(mortie.norm2mort(n, p, o)) == 0

    # Longer arrays were never in doubt, and are unchanged.
    many = mortie.norm2mort(np.arange(4), np.zeros(4, dtype=int), 6)
    n, p, o = mortie.mort2norm(many)
    np.testing.assert_array_equal(mortie.norm2mort(n, p, o), many)


def test_validate_morton_checks_every_element_order():
    """A mixed-order array used to pass on its first element alone (question 6)."""
    six = np.asarray(mortie.norm2mort([0, 1, 2], [0, 0, 0], 6), dtype=np.uint64)
    seven = np.asarray(mortie.norm2mort([0], [0], 7), dtype=np.uint64)
    mixed = np.concatenate([six, seven])
    # The first element is order 6, so the old check passed this array.
    assert mortie.validate_morton(six, order=6) is True
    with pytest.raises(
        ValueError,
        match=r"^Morton word decodes to order 7, expected 6 \(word 3 of 4\)",
    ):
        mortie.validate_morton(mixed, order=6)
    # The offender named is the lowest-index one, not merely the last.  This
    # needs *two* offenders at different indices to discriminate: `mixed` and
    # its reverse each hold exactly one, so for them lowest and last coincide.
    two_bad = np.concatenate([six[:1], seven, six[1:], seven])  # orders 6,7,6,6,7
    with pytest.raises(
        ValueError,
        match=r"^Morton word decodes to order 7, expected 6 \(word 1 of 5\)$",
    ):
        mortie.validate_morton(two_bad, order=6)
    # Without `order` there is nothing to disagree with; the decode still runs.
    assert mortie.validate_morton(mixed) is True


def test_validate_morton_refuses_a_non_scalar_order():
    """One order, checked against every word -- not a per-element expectation."""
    six = np.asarray(mortie.norm2mort([0, 1, 2], [0, 0, 0], 6), dtype=np.uint64)
    # The per-element comparison would happily broadcast these; the guard
    # keeps the refusal the scalar comparison used to give.
    for bad_order in ([6, 6, 6], np.array([6, 6, 6]), np.array([6, 6, 7]),
                      np.array([6, 6]), (6,), np.array([6])):
        with pytest.raises(TypeError, match=r"^order must be a single int"):
            mortie.validate_morton(six, order=bad_order)
    # A 0-d array is a scalar and still passes through.
    assert mortie.validate_morton(six, order=np.array(6)) is True
    assert mortie.validate_morton(six, order=np.uint8(6)) is True
    assert mortie.validate_morton(six, order=6) is True


def test_validate_morton_decode_refusal_wins_over_the_order_check():
    """The decode runs first, over the whole array, and names no index."""
    seven = np.asarray(mortie.norm2mort([0], [0], 7), dtype=np.uint64)
    # Index 0 disagrees with `order`; index 1 does not decode at all.  The
    # decode refusal wins even though the order offender is the lower index,
    # and it carries no `(word i of n)` suffix -- it is the kernel's message.
    both = np.concatenate([seven, np.zeros(1, dtype=np.uint64)])
    with pytest.raises(ValueError, match=r"^Morton index cannot be zero$"):
        mortie.validate_morton(both, order=6)
    # ... and the same refusal without an `order` to disagree with.
    with pytest.raises(ValueError, match=r"^Morton index cannot be zero$"):
        mortie.validate_morton(both)


def test_validate_morton_empty_is_vacuously_true():
    """No word disagrees with nothing -- for any order (issue #187)."""
    empty = np.zeros(0, dtype=np.uint64)
    assert mortie.validate_morton(empty) is True
    assert mortie.validate_morton(empty, order=6) is True
    # ... including an order no word could ever carry: both checks quantify
    # over the words, and there are none.  Before phase 5 this was IndexError.
    assert mortie.validate_morton(empty, order=99) is True
    assert mortie.validate_morton([], order=99) is True


def test_validate_morton_scalar_message_is_unchanged():
    """One *scalar* word in, no index suffix -- the message, verbatim."""
    word = int(np.asarray(mortie.norm2mort([0], [0], 6))[0])
    with pytest.raises(
        ValueError, match=r"^Morton word decodes to order 6, expected 7$"
    ):
        mortie.validate_morton(word, order=7)
    # A 0-d array is a scalar too.
    with pytest.raises(
        ValueError, match=r"^Morton word decodes to order 6, expected 7$"
    ):
        mortie.validate_morton(np.uint64(word), order=7)


def test_validate_morton_suffix_rule_is_rank_based_not_size_based():
    """A length-1 array is an array, and is indexed like one (issue #187)."""
    one = np.asarray(mortie.norm2mort([0], [0], 6), dtype=np.uint64)
    assert one.shape == (1,)
    with pytest.raises(
        ValueError,
        match=r"^Morton word decodes to order 6, expected 7 \(word 0 of 1\)$",
    ):
        mortie.validate_morton(one, order=7)
    # ... where the same word as a python int keeps the bare message, so the
    # two halves of the phase answer "is a length-1 array a scalar?" alike.
    with pytest.raises(
        ValueError, match=r"^Morton word decodes to order 6, expected 7$"
    ):
        mortie.validate_morton(int(one[0]), order=7)


# ---------------------------------------------------------------------------
# Phase 6: scalar-return unification on ``np.uint64`` (issue #187, the
# phase-3 fold's standing item, espg-approved).  Every function that returns a
# single **word**-valued scalar returns the same numpy type; the audit's
# out-of-scope boundary (times, orders, UNIQ ids, and the explicit
# ``dtype=`` escapes) is pinned alongside so a later drift in either
# direction is visible.
# ---------------------------------------------------------------------------


def _one_toc_word():
    return mortie.time2toc(10**9)


def test_morton_word_scalars_are_numpy_uint64():
    words = np.asarray(mortie.norm2mort([0, 1], [0, 0], 4), dtype=np.uint64)
    for name, got in (
        ("norm2mort", mortie.norm2mort(0, 0, 4)),
        ("common_ancestor", mortie.common_ancestor(words)),
        ("moc_min", mortie.moc_min(words)),
        ("decimal_to_word", mortie.decimal_to_word("-31123")),
    ):
        assert isinstance(got, np.uint64), f"{name} returned {type(got).__name__}"
        assert type(got) is np.uint64, f"{name} returned a uint64 subclass"
        assert np.ndim(got) == 0, name


def test_toc_word_scalars_are_numpy_uint64():
    """These four returned Python ``int`` before the unification."""
    t = _one_toc_word()
    column = np.array([t, mortie.time2toc(2 * 10**9)], dtype=np.uint64)
    for name, got in (
        ("time2toc", mortie.time2toc(10**9)),
        ("span2toc", mortie.span2toc(10**9, 2 * 10**9)),
        ("toc_merge", mortie.toc_merge(int(t), int(t))),
        ("toc_reduce", mortie.toc_reduce(column)),
    ):
        assert isinstance(got, np.uint64), f"{name} returned {type(got).__name__}"
        assert type(got) is np.uint64, f"{name} returned a uint64 subclass"
        assert np.ndim(got) == 0, name


def test_object_layer_element_access_is_still_a_uint64_word():
    """`MortonIndexScalar` satisfies the unification -- it *is* a uint64 word.

    Element access on the object layer cannot hand back a bare ``np.uint64``
    (a pandas ``ExtensionArray`` needs its own scalar type), so it hands back
    a subclass overriding nothing but the string presentation.  ``isinstance``
    is therefore the predicate a caller writes; ``type(...) is np.uint64`` is
    the stricter pin the *bare* functions above are held to.
    """
    pd = pytest.importorskip("pandas")
    from mortie.morton_index import MortonIndexScalar

    words = np.asarray(mortie.norm2mort([0, 1], [0, 0], 4), dtype=np.uint64)
    array = mortie.MortonIndexArray.from_words(words)
    for name, got in (
        ("__getitem__", array[0]),
        ("iter", next(iter(array))),
        ("tolist", array.tolist()[0]),
        ("take", array.take([0])[0]),
        ("unique", array.unique()[0]),
        ("Series.iloc", pd.Series(array).iloc[0]),
    ):
        assert isinstance(got, np.uint64), f"{name} returned {type(got).__name__}"
        assert type(got) is MortonIndexScalar, name
        assert int(got) == int(words[0]), name


def test_unified_scalars_keep_their_values():
    """The unification is a type change only -- every word is bit-identical."""
    column = mortie.time2toc(np.array([10**9, 2 * 10**9], dtype=np.uint64))
    # Each scalar call equals its slot in the array form, which never changed.
    assert int(mortie.time2toc(10**9)) == int(column[0])
    assert int(mortie.time2toc(2 * 10**9)) == int(column[1])
    # The reduce agrees with the pairwise merge over the same two words.
    assert int(mortie.toc_reduce(column)) == int(
        mortie.toc_merge(int(column[0]), int(column[1]))
    )
    # span2toc's scalar form equals its own one-element array form.
    span_array = mortie.span2toc(
        np.array([10**9], dtype=np.uint64), np.array([2 * 10**9], dtype=np.uint64)
    )
    assert int(mortie.span2toc(10**9, 2 * 10**9)) == int(span_array[0])


def test_word_scalars_stay_comparable_and_hashable():
    """uint64 keeps the operations a Python int was carrying at these sites.

    **Characterization, deliberately**: every assertion here passes on either
    return type, so this test does not fail if the unification is reverted --
    ``test_toc_word_scalars_are_numpy_uint64`` above is what pins the flip.
    What this one guards is the flip's *cost*: the CHANGELOG promises that
    comparisons, hashing, dict keys and f-strings are unaffected, and those
    are the operations a caller was relying on the old ``int`` for.  A future
    change that keeps ``np.uint64`` but breaks one of them is invisible to a
    type assertion and lands right here.
    """
    t = _one_toc_word()
    assert t == int(t)
    assert hash(t) == hash(int(t))
    assert {t: "cell"}[np.uint64(int(t))] == "cell"
    assert f"{t}" == f"{int(t)}"
    assert int(mortie.norm2mort(0, 0, 4)) == mortie.norm2mort(0, 0, 4)


def test_non_word_scalars_are_deliberately_not_unified():
    """The audit boundary: times, orders and UNIQ ids are not mortie words."""
    t = _one_toc_word()
    # Times in ns -- a quantity, not a word.
    start, end = mortie.toc2time(int(t))
    assert type(start) is int and type(end) is int
    assert type(mortie.from_gps_ns(10**9)) is int
    assert type(mortie.to_gps_ns(mortie.from_gps_ns(10**9))) is int
    assert type(mortie.from_datetime64(np.datetime64("2020-01-01", "ns"))) is int
    # A HEALPix order.
    assert type(mortie.infer_order_from_morton(mortie.norm2mort(0, 0, 4))) is int
    # UNIQ cell ids -- a different encoding, and inconsistent among themselves
    # today (two of the three are ``np.int64``); frozen here so the admitted
    # inconsistency cannot drift unnoticed in either direction.
    uniq = mortie.geo2uniq(0.0, 0.0, 4)
    assert type(uniq) is np.int64
    assert type(mortie.unique2parent(int(uniq))) is np.int64
    assert type(mortie.norm2uniq(0, 0, 4)) is int
    # The explicit dtype escapes on decimal_to_word are untouched.
    assert type(mortie.decimal_to_word("-31123", dtype=int)) is int
    from mortie.morton_index import MortonIndexScalar

    assert isinstance(
        mortie.decimal_to_word("-31123", dtype=MortonIndexScalar), MortonIndexScalar
    )


def test_toc_set_algebra_never_returns_a_bare_scalar():
    """`toc_normalize` / `toc_and` are cover-set ops: always an array out."""
    column = np.array(
        [mortie.time2toc(10**9), mortie.time2toc(2 * 10**9)], dtype=np.uint64
    )
    # Scalar-in is covered for both: it is exactly where the other four
    # functions grew a scalar return in this phase.
    for got in (mortie.toc_normalize(column), mortie.toc_and(column, column),
                mortie.toc_normalize(int(column[0])),
                mortie.toc_and(int(column[0]), int(column[0]))):
        assert isinstance(got, np.ndarray) and got.dtype == np.uint64


# ---------------------------------------------------------------------------
# Phase 7: the numpy floor the word semantics depend on (issue #187, ruled
# 2026-08-19).  Phase 6's ``np.uint64`` unification is only correct under
# NEP 50, so the floor is part of the contract rather than packaging trivia --
# and nothing else in the suite would notice a downgrade, because every CI job
# installs numpy unpinned.  These pin the *behaviour* the floor exists for, so
# an environment that drops below it fails loudly here instead of silently
# rounding words above 2**53.
# ---------------------------------------------------------------------------


def _declared_numpy_requirement():
    """The ``numpy`` requirement string from ``pyproject.toml``, or ``None``.

    ``importlib.metadata`` is not usable here: an editable install that has
    not been reinstalled still reports the *old* floor, so the declaration has
    to be read from the source of truth.  ``tomllib`` is stdlib only on 3.11+
    and ``requires-python`` is ``>=3.10``, so the fallback parses the list by
    hand rather than taking a parser dependency.
    """
    pyproject = pathlib.Path(__file__).parents[2] / "pyproject.toml"
    if not pyproject.exists():  # installed wheel: no source tree to read
        return None
    text = pyproject.read_text()
    try:
        import tomllib

        deps = tomllib.loads(text)["project"]["dependencies"]
    except ModuleNotFoundError:  # Python 3.10
        block = text.split("\ndependencies = [", 1)[1].split("\n]", 1)[0]
        deps = re.findall(r'"([^"]+)"', re.sub(r"#[^\n]*", "", block))
    return next((d for d in deps if d.startswith("numpy")), None)


def test_the_declared_numpy_floor_is_numpy_2():
    """The floor is pinned where it is *declared*, not only where it runs.

    Every CI job installs numpy unpinned, so numpy 2 lands regardless of what
    ``pyproject.toml`` says -- an edit that quietly lowered the floor back to
    ``>=1.20`` would be invisible to the whole matrix.  This is that guard.
    """
    declared = _declared_numpy_requirement()
    if declared is None:
        pytest.skip("no pyproject.toml alongside the package (installed wheel)")
    assert declared == "numpy>=2", f"declared numpy requirement is {declared!r}"


def test_numpy_is_at_or_above_the_declared_floor():
    assert int(np.__version__.split(".")[0]) >= 2, (
        f"mortie declares numpy>=2 (pyproject.toml); got {np.__version__}"
    )


def test_word_arithmetic_stays_uint64_under_nep50():
    """A word mixed with a Python ``int`` must not promote to ``float64``."""
    word = mortie.time2toc(10**9)
    for got in (word + 1, word - 1, word * 2, word // 2, word % 7):
        assert got.dtype == np.uint64, f"promoted to {got.dtype}"
    # Bitwise ops against a Python int work too -- a toc word is a bit-packed
    # struct, so masking and shifting it is the natural thing to do.  Every
    # operand here is a Python int deliberately: below numpy 2 these raise
    # TypeError, while the all-uint64 spellings were always legal and would
    # pin nothing.
    for got in (word | 1, word & 0xFF, word >> 32, word ^ 1):
        assert got.dtype == np.uint64, f"promoted to {got.dtype}"


def test_word_arithmetic_is_exact_near_the_top_of_the_range():
    """The failure a float64 promotion would cause, pinned by value.

    mortie words run near ``2**62``, far above float64's ``2**53`` exact
    integer range, so a promotion does not merely change dtype -- it returns
    the wrong word.
    """
    big = mortie.time2toc(mortie.TOC_MAX_NS - 3)
    assert int(big) > 2**53
    # The Python int on the right is the whole point: ``big + np.uint64(1)``
    # stays uint64 on numpy 1 too, so only the mixed form discriminates.
    # Below numpy 2 this returns 1.8446744065119617e+19 and the assert fails.
    assert int(big + 1) == int(big) + 1
    # ... which is exactly what a float64 round-trip would get wrong.
    assert int(np.float64(int(big)) + 1) != int(big) + 1

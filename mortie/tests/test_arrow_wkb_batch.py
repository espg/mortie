"""Tests for the Arrow skin of the WKB batch (issue #163).

``mortie.arrow.from_wkbs`` takes a geoparquet / STAC geometry column as it
comes off the file and returns the core's ragged ``(values, out_offsets)``
pair.  Its whole reason to exist is that hand-rolling the buffer extraction is
a **footgun**: four separate traps, the first three of which yield wrong data
with *no error*, the fourth a wrong diagnosis.  Each has a test here that
fails if its guard is removed:

1. a ``ChunkedArray`` has no ``buffers()`` at all (and that is the default
   shape a parquet column reads back as);
2. a **sliced** array's ``buffers()`` are the original array's, so ignoring
   ``offset`` reads a different, valid-looking set of blobs;
3. ``large_binary`` offsets are ``int64``, plain ``binary``'s are ``int32``;
4. a null spans **zero bytes**, so it sails through as an empty blob and the
   core reports it as a truncated geometry, not as a missing one.

Blobs are packed by hand via the core batch's helpers, so nothing here needs a
geometry backend except the dialect fixtures, which are easier to author with
shapely.
"""

import warnings

import numpy as np
import pytest

import mortie
from mortie.geometry import from_wkbs as core_from_wkbs
from mortie.tests.test_wkb_batch import (
    CHUNK,
    assert_ragged_contract,
    corpus,
    polygon_blob,
    quad,
    truncated_blob,
)

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

from mortie import arrow as marrow  # noqa: E402


def _binary(blobs, type=None):
    """Pack blobs into a pyarrow binary array.

    Parameters
    ----------
    blobs : list of bytes
        The WKB blobs.
    type : pyarrow.DataType, optional
        ``binary`` (the default) or ``large_binary``.

    Returns
    -------
    pyarrow.Array
        The column.
    """
    return pa.array(blobs, type=type or pa.binary())


def _assert_matches_core(column, blobs, **kwargs):
    """Assert the skin equals the core fed the same blobs, and keeps contract.

    Parameters
    ----------
    column : pyarrow.Array or pyarrow.ChunkedArray
        The Arrow column handed to the skin.
    blobs : list of bytes
        The same geometries as Python ``bytes``, for the core.
    **kwargs : dict
        Shared parameters forwarded to both calls.
    """
    values, offsets = marrow.from_wkbs(column, **kwargs)
    want_values, want_offsets = core_from_wkbs(blobs, **kwargs)
    np.testing.assert_array_equal(values, want_values)
    np.testing.assert_array_equal(offsets, want_offsets)
    assert_ragged_contract(values, offsets, len(blobs))


# ── trap 1: a ChunkedArray has no buffers() ────────────────────────────────


def test_a_chunked_column_has_no_buffers_and_is_walked_chunk_by_chunk():
    # The trap itself, pinned: the recipe's very first call does not exist on
    # the shape a parquet column reads back as.
    blobs = corpus(48)
    chunked = pa.chunked_array([_binary(blobs[:5]), _binary(blobs[5:29]),
                                _binary(blobs[29:])])
    assert not hasattr(chunked, "buffers")
    assert chunked.num_chunks == 3
    _assert_matches_core(chunked, blobs, order=6)


def test_a_parquet_column_reads_back_chunked_and_still_covers(tmp_path):
    # The default shape, end to end: this is how zagg's STAC catalog arrives.
    blobs = corpus(24)
    path = tmp_path / "footprints.parquet"
    pq.write_table(pa.table({"geometry": _binary(blobs)}), path)
    column = pq.read_table(path).column("geometry")
    assert isinstance(column, pa.ChunkedArray)
    _assert_matches_core(column, blobs, order=6)


# ── trap 2: a sliced array's buffers are the original array's ──────────────


def test_a_sliced_column_reads_its_own_rows_not_the_first_ones():
    # The demonstrated footgun (issue #163): `slice()` is zero-copy metadata,
    # so the offset-blind recipe returns rows 0..k with no error at all.
    arr = _binary([b"AAA", b"BBBB", b"CCCCC", b"DDDDDD"])
    sliced = arr.slice(2, 2)
    naive_offsets = np.frombuffer(sliced.buffers()[1], dtype=np.int32)
    naive_values = memoryview(sliced.buffers()[2])
    naive = [bytes(naive_values[naive_offsets[i]:naive_offsets[i + 1]])
             for i in range(len(sliced))]
    assert naive == [b"AAA", b"BBBB"]        # wrong, and silently so
    assert sliced.to_pylist() == [b"CCCCC", b"DDDDDD"]

    # And the skin, on real geometry: the slice covers *its own* blobs.
    blobs = corpus(12)
    column = _binary(blobs)
    assert column.slice(5, 4).offset == 5
    _assert_matches_core(column.slice(5, 4), blobs[5:9], order=6)


def test_a_sliced_and_chunked_column_offsets_every_chunk():
    # Every chunk carries its own offset, so a fix applied to the first chunk
    # only would still be wrong here.
    blobs = corpus(40)
    chunked = pa.chunked_array([
        _binary(blobs[:20]).slice(3, 9),      # blobs[3:12]
        _binary(blobs[20:]).slice(6, 11),     # blobs[26:37]
    ])
    _assert_matches_core(chunked, blobs[3:12] + blobs[26:37], order=6)


# ── trap 3: large_binary offsets are int64 ─────────────────────────────────


def test_large_binary_offsets_are_read_as_int64():
    blobs = corpus(16)
    column = _binary(blobs, type=pa.large_binary())
    # The trap: the same buffer read at int32 gives a different, non-erroring
    # sequence of spans -- the first blob still looks plausible.
    as_i64 = np.frombuffer(column.buffers()[1], dtype=np.int64)
    as_i32 = np.frombuffer(column.buffers()[1], dtype=np.int32)
    assert not np.array_equal(as_i64, as_i32[:as_i64.size])
    _assert_matches_core(column, blobs, order=6)


def test_large_binary_survives_slicing_and_chunking():
    blobs = corpus(24)
    lb = pa.large_binary()
    chunked = pa.chunked_array([
        _binary(blobs[:12], type=lb).slice(2, 7),      # blobs[2:9]
        _binary(blobs[12:], type=lb),                  # blobs[12:]
    ], type=lb)
    _assert_matches_core(chunked, blobs[2:9] + blobs[12:], order=6)


# ── trap 4: a null spans zero bytes ────────────────────────────────────────


def test_a_null_spans_zero_bytes_and_is_refused_by_index():
    # The trap, pinned on the arrow side: the offsets of `[b"AA", None,
    # b"BBB"]` are [0, 2, 2, 5], so a null reads back as an empty blob.
    probe = _binary([b"AA", None, b"BBB"])
    assert list(np.frombuffer(probe.buffers()[1], dtype=np.int32)) == [0, 2, 2, 5]

    blobs = corpus(6)
    column = _binary(blobs[:2] + [None] + blobs[3:])
    with pytest.raises(ValueError, match=r"^blob 2: null entry"):
        marrow.from_wkbs(column, order=6)


def test_a_null_is_named_in_the_column_frame_not_the_chunk_frame():
    blobs = corpus(30)
    chunked = pa.chunked_array([
        _binary(blobs[:10]),
        _binary(blobs[10:20]),
        _binary(blobs[20:23] + [None] + blobs[24:]),
    ])
    with pytest.raises(ValueError, match=r"^blob 23: null entry"):
        marrow.from_wkbs(chunked, order=6)


def test_a_null_in_a_sliced_chunk_is_named_by_its_logical_index():
    blobs = corpus(12)
    column = _binary(blobs[:7] + [None] + blobs[8:]).slice(4, 6)
    # Logical row 3 of the slice is the null; the slice is its own column.
    with pytest.raises(ValueError, match=r"^blob 3: null entry"):
        marrow.from_wkbs(column, order=6)


@pytest.mark.parametrize("chunked", [False, True], ids=["flat", "chunked"])
def test_a_null_preempts_a_lower_index_malformed_blob(chunked):
    # The documented cross-class order: the null scan is a whole-column
    # pre-pass, so it fires before the core parses blob 0. Within each class
    # the lowest index still wins (the reverse ordering, below).
    blobs = corpus(30)
    blobs[3] = truncated_blob()
    rows = blobs[:20] + [None] + blobs[21:]
    column = (pa.chunked_array([_binary(rows[:12]), _binary(rows[12:])])
              if chunked else _binary(rows))
    with pytest.raises(ValueError, match=r"^blob 20: null entry"):
        marrow.from_wkbs(column, order=6)
    # Same rows with the null spelled as an empty blob: the core reports the
    # malformed one, at its lower index.
    with pytest.raises(ValueError, match=r"^blob 3: "):
        core_from_wkbs(blobs[:20] + [b""] + blobs[21:], order=6)


def test_the_lowest_index_wins_within_each_failure_class():
    blobs = corpus(30)
    blobs[25] = truncated_blob()
    with pytest.raises(ValueError, match=r"^blob 3: null entry"):
        marrow.from_wkbs(_binary(blobs[:3] + [None] + blobs[4:]), order=6)
    blobs[3] = truncated_blob()
    with pytest.raises(ValueError, match=r"^blob 3: "):
        marrow.from_wkbs(_binary(blobs), order=6)


# ── shapes: the acceptance matrix ──────────────────────────────────────────


def test_binary_array_parity_with_the_core():
    blobs = corpus(64)
    _assert_matches_core(_binary(blobs), blobs, order=8)


def test_per_blob_parity_with_the_scalar():
    blobs = corpus(32)
    values, offsets = marrow.from_wkbs(_binary(blobs), order=8)
    for i, blob in enumerate(blobs):
        np.testing.assert_array_equal(
            values[offsets[i]:offsets[i + 1]],
            mortie.from_wkb(blob, order=8, moc=True),
        )


def test_a_single_element_column():
    blobs = corpus(1)
    _assert_matches_core(_binary(blobs), blobs, order=7)


@pytest.mark.parametrize("empty", ["array", "large", "chunked", "no_chunks",
                                   "empty_slice"])
def test_an_empty_column_keeps_the_contract(empty):
    column = {
        "array": lambda: _binary([]),
        "large": lambda: _binary([], type=pa.large_binary()),
        "chunked": lambda: pa.chunked_array([_binary([])]),
        "no_chunks": lambda: pa.chunked_array([], type=pa.binary()),
        "empty_slice": lambda: _binary(corpus(4)).slice(2, 0),
    }[empty]()
    values, offsets = marrow.from_wkbs(column, order=6)
    assert_ragged_contract(values, offsets, 0)
    assert values.size == 0


def test_zero_length_chunks_between_populated_ones():
    blobs = corpus(8)
    chunked = pa.chunked_array([_binary([]), _binary(blobs[:3]), _binary([]),
                                _binary(blobs[3:]), _binary([])])
    _assert_matches_core(chunked, blobs, order=6)


# ── dialects the ATL03 corpus lacks ────────────────────────────────────────


@pytest.mark.parametrize(
    "wkt",
    [
        "POLYGON ((10 -75, 40 -75, 40 -71, 10 -71, 10 -75))",
        "POLYGON ((10 -75, 40 -75, 40 -71, 10 -71, 10 -75),"
        "(20 -74, 30 -74, 30 -72, 20 -72, 20 -74))",
        "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)),((5 5, 6 5, 6 6, 5 6, 5 5)))",
        "POLYGON ((170 -20, -170 -20, -170 -10, 170 -10, 170 -20))",
        "POLYGON ((0 -89.5, 90 -89.5, 180 -89.5, -90 -89.5, 0 -89.5))",
        "POLYGON ((0 89.5, 90 89.5, 180 89.5, -90 89.5, 0 89.5))",
        "POLYGON Z ((10 -75 7, 40 -75 7, 40 -71 7, 10 -71 7, 10 -75 7))",
        "POLYGON M ((10 -75 7, 40 -75 7, 40 -71 7, 10 -71 7, 10 -75 7))",
    ],
)
@pytest.mark.parametrize("byte_order", [0, 1])
@pytest.mark.parametrize("kind", ["binary", "large_binary"])
def test_dialect_fixtures_match_the_scalar(wkt, byte_order, kind):
    shapely = pytest.importorskip("shapely")
    blob = shapely.to_wkb(shapely.from_wkt(wkt), byte_order=byte_order,
                          flavor="iso")
    column = _binary([blob], type=pa.large_binary() if kind == "large_binary"
                     else pa.binary())
    values, offsets = marrow.from_wkbs(column, order=7)
    assert_ragged_contract(values, offsets, 1)
    np.testing.assert_array_equal(values, mortie.from_wkb(blob, order=7, moc=True))


def test_ewkb_and_mixed_endianness_in_one_column():
    shapely = pytest.importorskip("shapely")
    geom = shapely.from_wkt("POLYGON ((10 -75, 40 -75, 40 -71, 10 -71, 10 -75))")
    blobs = [
        shapely.to_wkb(geom, byte_order=1),
        shapely.to_wkb(geom, byte_order=0),
        shapely.to_wkb(shapely.set_srid(geom, 4326), include_srid=True),
    ]
    _assert_matches_core(_binary(blobs), blobs, order=7)


# ── the scalar parameters, forwarded unchanged ─────────────────────────────


def test_shared_stop_criteria_are_forwarded():
    blobs = corpus(16)
    column = _binary(blobs)
    for kwargs in ({"tolerance": 0.5}, {"max_cells": 32}):
        _assert_matches_core(column, blobs, order=10, **kwargs)
    with pytest.raises(ValueError, match="at most one"):
        marrow.from_wkbs(column, order=8, tolerance=0.5, max_cells=32)


def test_normalize_is_forwarded():
    # A ring wound the other way, so the two spellings must disagree -- else
    # the parameter could be dropped and the test still pass.
    blob = polygon_blob([quad(0.0, 0.0)[::-1]])
    column = _binary([blob])
    covers = {}
    for norm in (True, False):
        values, _ = marrow.from_wkbs(column, order=6, normalize=norm)
        np.testing.assert_array_equal(
            values, mortie.from_wkb(blob, order=6, moc=True, normalize=norm)
        )
        covers[norm] = values
    assert not np.array_equal(covers[True], covers[False])


def test_order_is_forwarded_and_range_checked():
    blobs = corpus(4)
    column = _binary(blobs)
    for order in (4, 9):
        _assert_matches_core(column, blobs, order=order)
    for order in (0, 30):
        with pytest.raises(ValueError, match="Order must be between 1 and 29"):
            marrow.from_wkbs(column, order=order)


def test_the_budget_warning_still_names_the_lowest_index():
    column = _binary(corpus(8))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        marrow.from_wkbs(column, order=8, max_cells=1)
    raised = [w for w in caught if "max_cells=1" in str(w.message)]
    assert len(raised) == 1
    assert "blob 0" in str(raised[0].message)


# ── fail-fast, in the logical column's frame ───────────────────────────────


def test_a_malformed_blob_is_named_by_its_column_index_across_chunks():
    blobs = corpus(30)
    blobs[23] = truncated_blob()
    chunked = pa.chunked_array([_binary(blobs[:10]), _binary(blobs[10:20]),
                                _binary(blobs[20:])])
    with pytest.raises(ValueError, match=r"^blob 23: "):
        marrow.from_wkbs(chunked, order=6)


@pytest.mark.parametrize("bad_index", [0, CHUNK - 1, CHUNK, CHUNK + 1])
def test_the_rust_chunk_boundary_does_not_renumber_an_arrow_column(bad_index):
    # The arrow chunking and the Rust chunk loop are independent; neither may
    # shift the reported index.
    n = 2 * CHUNK + 16
    blobs = corpus(n)
    blobs[bad_index] = truncated_blob()
    chunked = pa.chunked_array([_binary(blobs[:777]), _binary(blobs[777:])])
    with pytest.raises(ValueError, match=rf"^blob {bad_index}: "):
        marrow.from_wkbs(chunked, order=5)


def test_a_malformed_blob_in_a_sliced_column_uses_the_slice_frame():
    blobs = corpus(12)
    blobs[8] = truncated_blob()
    column = _binary(blobs).slice(5, 6)
    with pytest.raises(ValueError, match=r"^blob 3: "):
        marrow.from_wkbs(column, order=6)


# ── geoarrow-typed columns ─────────────────────────────────────────────────
#
# The ATL03 column carries ``ARROW:extension:name = geoarrow.wkb`` in its field
# metadata and reads back as plain `binary` only because nothing registered the
# extension; with geoarrow-pyarrow installed the same file arrives as an
# ExtensionArray. `pa.ExtensionType` is pyarrow's own, so this needs no new
# dependency.


class _WkbExtType(pa.ExtensionType):
    """A stand-in for ``geoarrow.wkb``: binary storage, geoarrow's name."""

    def __init__(self, storage_type=pa.binary(), name="geoarrow.wkb"):
        super().__init__(storage_type, name)

    def __arrow_ext_serialize__(self):
        """Return the (empty) parameter payload."""
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        """Rebuild the type from its storage type."""
        return cls(storage_type)


def _extension(blobs, type=None, name="geoarrow.wkb"):
    """Wrap blobs in an extension array over binary/large_binary storage.

    Parameters
    ----------
    blobs : list of bytes
        The WKB blobs.
    type : pyarrow.DataType, optional
        The storage type; ``binary`` by default.
    name : str, optional
        The extension name.

    Returns
    -------
    pyarrow.ExtensionArray
        The wrapped column.
    """
    storage_type = type or pa.binary()
    return pa.ExtensionArray.from_storage(
        _WkbExtType(storage_type, name), _binary(blobs, type=storage_type)
    )


@pytest.mark.parametrize("type_", [pa.binary(), pa.large_binary()],
                         ids=["binary", "large_binary"])
def test_a_geoarrow_extension_column_is_unwrapped_to_its_storage(type_):
    blobs = corpus(12)
    column = _extension(blobs, type=type_)
    assert isinstance(column.type, pa.BaseExtensionType)
    assert column.type.extension_name == "geoarrow.wkb"
    _assert_matches_core(column, blobs, order=6)
    # Byte-identical to feeding the storage by hand, which is the escape hatch
    # a consumer would otherwise have to know about.
    for got, want in zip(marrow.from_wkbs(column, order=6),
                         marrow.from_wkbs(column.storage, order=6)):
        np.testing.assert_array_equal(got, want)


def test_a_chunked_geoarrow_extension_column_is_unwrapped():
    # The shape a geoparquet read actually produces: chunked *and* typed.
    blobs = corpus(20)
    chunked = pa.chunked_array([_extension(blobs[:7]), _extension(blobs[7:])])
    assert isinstance(chunked.type, pa.BaseExtensionType)
    _assert_matches_core(chunked, blobs, order=6)
    # Zero chunks still type-check off the column's own extension type.
    values, offsets = marrow.from_wkbs(
        pa.chunked_array([], type=_WkbExtType()), order=6)
    assert_ragged_contract(values, offsets, 0)


def test_a_sliced_geoarrow_extension_column_reads_its_own_rows():
    # The storage of a sliced extension array keeps the slice's offset, so
    # trap 2 has to survive the unwrap.
    blobs = corpus(9)
    _assert_matches_core(_extension(blobs).slice(4, 3), blobs[4:7], order=6)


def test_a_null_in_a_geoarrow_extension_column_is_still_refused_by_index():
    column = _extension(corpus(5)[:3] + [None] + corpus(5)[4:])
    with pytest.raises(ValueError, match="blob 3: null entry"):
        marrow.from_wkbs(column, order=6)


def test_an_extension_over_non_binary_storage_is_refused_by_its_name():
    column = pa.ExtensionArray.from_storage(
        _WkbExtType(pa.int64(), "geoarrow.point"), pa.array([1, 2], pa.int64())
    )
    with pytest.raises(TypeError, match="extension type geoarrow.point"):
        marrow.from_wkbs(column, order=6)


# ── input contract ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "column",
    [
        pytest.param(lambda: pa.array(["POLYGON ((0 0, 1 0, 1 1, 0 0))"]),
                     id="string"),
        pytest.param(lambda: pa.array([1, 2, 3]), id="int64"),
        pytest.param(lambda: pa.array([b"ab"], type=pa.binary(2)),
                     id="fixed_size_binary"),
        pytest.param(lambda: pa.chunked_array([pa.array([1.0])]), id="chunked_double"),
    ],
)
def test_a_non_binary_column_is_refused_by_type(column):
    with pytest.raises(TypeError, match="binary or large_binary"):
        marrow.from_wkbs(column(), order=6)


@pytest.mark.parametrize("type_", [pa.float64(), pa.int64(), pa.string(),
                                   pa.binary(2)])
def test_a_non_binary_column_with_zero_chunks_is_refused_by_type(type_):
    # The crossed case the two shapes above miss: a ChunkedArray carries its
    # .type with no chunks at all, so a per-chunk check would never run and
    # the column would be answered as empty instead of refused.
    column = pa.chunked_array([], type=type_)
    assert len(column.chunks) == 0 and column.type == type_
    with pytest.raises(TypeError, match="binary or large_binary"):
        marrow.from_wkbs(column, order=6)


@pytest.mark.parametrize("bad", [[b"\x01"], b"\x01", None, np.array([1])])
def test_a_non_arrow_input_is_refused_by_name(bad):
    with pytest.raises(TypeError, match="pyarrow Array or ChunkedArray"):
        marrow.from_wkbs(bad, order=6)


def test_no_python_bytes_object_is_built_per_blob():
    # The point of the skin: the blobs handed to the core are zero-copy views
    # into the column's own value buffer, not copies.
    from mortie.arrow import _wkb_blobs_from_arrow

    blobs = corpus(8)
    column = _binary(blobs)
    views = _wkb_blobs_from_arrow(pa, column)
    assert all(isinstance(v, memoryview) for v in views)
    assert [bytes(v) for v in views] == blobs
    # Every view is backed by the one value buffer, so nothing was copied.
    assert {v.obj.address for v in views} == {column.buffers()[2].address}

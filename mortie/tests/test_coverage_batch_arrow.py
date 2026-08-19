"""Tests for the Arrow skin of batch coverage (issue #153, phase 4).

``mortie.arrow.polygons_to_morton_mocs`` accepts a ``list<struct<lat, lon>>``
polygon column (or a paired ``list<double>`` spelling) and returns a
``ListArray`` whose values carry the registered ``morton_index`` extension
type — so a per-polygon MOC column drops straight into a parquet catalog.
"""

import numpy as np
import pytest

from mortie.coverage import _morton_coverage_moc

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

from mortie import arrow as marrow  # noqa: E402

TRIANGLES = [
    ([40.0, 50.0, 45.0], [-120.0, -120.0, -110.0]),
    ([10.0, 20.0, 15.0], [-80.0, -80.0, -70.0]),
    ([-10.0, -10.0, 10.0, 10.0], [170.0, -170.0, -170.0, 170.0]),  # antimeridian
]


def _struct_column(rings=TRIANGLES):
    """The rings as a ``list<struct<lat, lon>>`` pyarrow array."""
    return pa.array(
        [[{"lat": la, "lon": lo} for la, lo in zip(*ring)] for ring in rings]
    )


def _pair_columns(rings=TRIANGLES):
    """The rings as a ``(lats, lons)`` pair of ``list<double>`` arrays."""
    lats = pa.array([list(ring[0]) for ring in rings], type=pa.list_(pa.float64()))
    lons = pa.array([list(ring[1]) for ring in rings], type=pa.list_(pa.float64()))
    return lats, lons


def _assert_matches_scalar(mocs, rings, order, **kwargs):
    """Each list entry's words == the scalar MOC of the matching ring."""
    assert len(mocs) == len(rings)
    for i, (la, lo) in enumerate(rings):
        expected = _morton_coverage_moc(la, lo, order=order, **kwargs)
        got = mocs[i].values.to_numpy(zero_copy_only=False).astype(np.uint64)
        np.testing.assert_array_equal(got, expected)


def test_struct_input_parity_and_type():
    mocs = marrow.polygons_to_morton_mocs(_struct_column(), order=7)
    assert pa.types.is_list(mocs.type)
    assert mocs.type.value_type.extension_name == marrow.EXTENSION_NAME
    _assert_matches_scalar(mocs, TRIANGLES, order=7)


def test_pair_input_parity():
    mocs = marrow.polygons_to_morton_mocs(_pair_columns(), order=7)
    _assert_matches_scalar(mocs, TRIANGLES, order=7)


def test_pair_offsets_must_agree():
    lats, _ = _pair_columns()
    lons_short = pa.array(
        [[-120.0, -120.0, -110.0]], type=pa.list_(pa.float64())
    )
    with pytest.raises(ValueError, match="equal offsets"):
        marrow.polygons_to_morton_mocs((lats, lons_short), order=7)


def test_chunked_input():
    col = pa.chunked_array([_struct_column(TRIANGLES[:1]), _struct_column(TRIANGLES[1:])])
    mocs = marrow.polygons_to_morton_mocs(col, order=7)
    _assert_matches_scalar(mocs, TRIANGLES, order=7)


def _assert_same_mocs(a, b):
    """Two list arrays hold the same per-entry MOC words."""
    assert len(a) == len(b)
    for i in range(len(a)):
        np.testing.assert_array_equal(
            a[i].values.to_numpy(zero_copy_only=False),
            b[i].values.to_numpy(zero_copy_only=False),
        )


def test_sliced_input_nonzero_offsets():
    """A sliced list array is re-based: same MOCs as the unsliced equivalent.

    ``.values`` on a sliced ``ListArray`` is the *whole* unsliced child, and
    its offsets start above 0 — the skin slices the value window and re-bases
    the offsets together, which is what the strict core requires.
    """
    for start, length in ((1, None), (1, 1), (2, 1)):
        sliced = _struct_column().slice(start) if length is None \
            else _struct_column().slice(start, length)
        assert sliced.offsets[0].as_py() > 0  # the case under test
        rings = TRIANGLES[start:] if length is None \
            else TRIANGLES[start:start + length]
        mocs = marrow.polygons_to_morton_mocs(sliced, order=7)
        _assert_matches_scalar(mocs, rings, order=7)
        _assert_same_mocs(mocs, marrow.polygons_to_morton_mocs(
            _struct_column(rings), order=7))


def test_sliced_pair_input():
    """The paired spelling re-bases too, including a one-side-only slice."""
    lats, lons = _pair_columns()
    mocs = marrow.polygons_to_morton_mocs((lats.slice(1), lons.slice(1)), order=7)
    _assert_matches_scalar(mocs, TRIANGLES[1:], order=7)
    # Only one side sliced, the other built to match: logically identical
    # offsets once re-based, so this must be accepted (raw offsets differ).
    lons_tail = pa.array([list(r[1]) for r in TRIANGLES[1:]],
                         type=pa.list_(pa.float64()))
    assert not lats.slice(1).offsets.equals(lons_tail.offsets)
    mocs2 = marrow.polygons_to_morton_mocs((lats.slice(1), lons_tail), order=7)
    _assert_same_mocs(mocs, mocs2)


def test_chunked_and_sliced_input():
    """A slice of a chunked column gives the same MOCs as the plain rings."""
    col = pa.chunked_array(
        [_struct_column(TRIANGLES[:1]), _struct_column(TRIANGLES[1:])]
    )
    mocs = marrow.polygons_to_morton_mocs(col.slice(1), order=7)
    _assert_matches_scalar(mocs, TRIANGLES[1:], order=7)
    _assert_same_mocs(
        mocs, marrow.polygons_to_morton_mocs(_struct_column(TRIANGLES[1:]), order=7)
    )


def test_null_polygon_rejected_with_index():
    col = pa.array(
        [[{"lat": 40.0, "lon": -120.0}, {"lat": 50.0, "lon": -120.0},
          {"lat": 45.0, "lon": -110.0}], None],
        type=_struct_column().type,
    )
    with pytest.raises(ValueError, match=r"polygon 1: null"):
        marrow.polygons_to_morton_mocs(col, order=7)


def test_bad_layout_rejected():
    with pytest.raises(ValueError, match=r"list<struct"):
        marrow.polygons_to_morton_mocs(pa.array([[1.0, 2.0]]), order=7)


def test_shared_stop_criteria_forwarded():
    mocs = marrow.polygons_to_morton_mocs(_struct_column(), order=9, max_cells=64)
    _assert_matches_scalar(mocs, TRIANGLES, order=9, max_cells=64)
    mocs = marrow.polygons_to_morton_mocs(_struct_column(), order=9, tolerance=0.25)
    _assert_matches_scalar(mocs, TRIANGLES, order=9, tolerance=0.25)
    with pytest.raises(ValueError, match="at most one of"):
        marrow.polygons_to_morton_mocs(_struct_column(), tolerance=0.1, max_cells=8)


def test_error_names_polygon_index():
    col = pa.array(
        [[{"lat": 40.0, "lon": -120.0}, {"lat": 50.0, "lon": -120.0},
          {"lat": 45.0, "lon": -110.0}],
         [{"lat": 10.0, "lon": -80.0}, {"lat": 20.0, "lon": -80.0}]],
    )
    with pytest.raises(ValueError, match=r"polygon 1: needs at least 3"):
        marrow.polygons_to_morton_mocs(col, order=7)


def test_parquet_round_trip_keeps_extension():
    """The MOC column survives parquet with its morton_index identity."""
    mocs = marrow.polygons_to_morton_mocs(_struct_column(), order=7)
    table = pa.table({"footprint_cells": mocs})
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    back = pq.read_table(pa.BufferReader(sink.getvalue()))
    col = back.column("footprint_cells").combine_chunks()
    assert col.type.value_type.extension_name == marrow.EXTENSION_NAME
    _assert_matches_scalar(col, TRIANGLES, order=7)

"""Tests for the numpy-level point-kind encoder ``geo2mort(..., points=True)``.

Covers the contract from issue #96: a vectorized, numpy-in/numpy-out point
encoder on the ``geo2mort`` surface that routes through the kernel's order-29
point path (``Kind::Point``) and returns a plain contiguous ``uint64`` ndarray.
"""

import numpy as np
import pytest

from mortie import _rustie, common_ancestor, geo2mort
from mortie.tools import clip2order

# A spread of locations across hemispheres and near the poles/prime meridian.
_LATS = np.array([45.0, -80.0, 0.0, 12.3, 89.9, -89.9])
_LONS = np.array([-122.0, 120.0, 0.0, 45.6, 179.9, -179.9])


def _area(order=29):
    return geo2mort(_LATS, _LONS, order=order)


def _point():
    return geo2mort(_LATS, _LONS, order=29, points=True)


def test_point_differs_from_area_but_coarsens_identically():
    """(a) point != area at order 29, but coarsening is bit-identical below 29.

    This is the property zagg's grouping relies on: point and area words for the
    same location share the entire order-<29 prefix, so grouping by a coarsened
    word is independent of whether the words started as points or areas.
    """
    area = _area(29)
    point = _point()
    assert not np.array_equal(area, point), "point and area words must differ at 29"
    # At every coarser order the two collapse to the same area cell.
    for order in range(0, 29):
        np.testing.assert_array_equal(
            clip2order(order, point),
            clip2order(order, area),
            err_msg=f"point/area coarsening diverged at order {order}",
        )


def test_lone_point_survives_common_ancestor_unchanged():
    """(b) a single point word passes through common_ancestor unchanged (kind kept)."""
    for word in _point():
        got = common_ancestor(np.array([word], dtype=np.uint64))
        assert int(got) == int(word), "lone point word must be returned unchanged"


def test_points_true_wrong_order_raises():
    """(c) points=True with an explicit order != 29 raises ValueError."""
    for bad_order in (0, 18, 28):
        with pytest.raises(ValueError):
            geo2mort(1.0, 2.0, order=bad_order, points=True)
    # order=29 (or the default / None) is fine.
    geo2mort(1.0, 2.0, order=29, points=True)
    geo2mort(1.0, 2.0, points=True)
    geo2mort(1.0, 2.0, order=None, points=True)


def test_point_output_dtype_shape_contiguity():
    """(d) result is uint64, C-contiguous, same shape as the input."""
    out = _point()
    assert out.dtype == np.uint64
    assert out.flags["C_CONTIGUOUS"]
    assert out.shape == _LATS.shape
    # scalar in -> length-1 ndarray, matching the area path's always-ndarray return.
    scalar = geo2mort(45.0, -122.0, points=True)
    assert isinstance(scalar, np.ndarray)
    assert scalar.shape == (1,)
    assert scalar.dtype == np.uint64


def test_point_matches_from_latlon_kernel_path():
    """geo2mort(points=True) is bit-identical to the from_latlon internal path.

    This is the equivalence zagg needs to swap ``from_latlon(points=True)`` +
    unwrap for ``geo2mort(..., points=True)``. Cross-checked against the raw
    kernel bindings (ang2pix(29) -> from_nested_point) that from_latlon composes,
    so it holds without the optional pandas dependency.
    """
    via_geo = _point()
    nested = _rustie.rust_ang2pix(29, _LONS, _LATS)
    nested = np.ascontiguousarray(nested, dtype=np.uint64)
    via_kernel = _rustie.rust_mi_from_nested_point(nested)
    np.testing.assert_array_equal(via_geo, via_kernel)


def test_point_matches_morton_index_from_latlon():
    """Same equivalence, but against the public MortonIndexArray.from_latlon."""
    pytest.importorskip("pandas")
    from mortie.morton_index import MortonIndexArray

    via_geo = _point()
    arr = MortonIndexArray.from_latlon(_LATS, _LONS, points=True)
    via_arr = np.asarray(arr).astype(np.uint64)
    np.testing.assert_array_equal(via_geo, via_arr)


def test_area_default_backward_compatible_at_explicit_order():
    """points=False is byte-identical to the pre-#96 area encode at a given order."""
    for order in (0, 6, 12, 18, 29):
        np.testing.assert_array_equal(
            geo2mort(_LATS, _LONS, order=order),
            geo2mort(_LATS, _LONS, order=order, points=False),
        )


def test_nonfinite_handling_agrees_between_paths():
    """Non-finite lat/lon resolve identically on the area and point paths.

    The two share the order-29 hash step, so a non-finite input must land on the
    same HEALPix nested cell (they differ only in Kind), and must fail — or not —
    the same way. Today neither path raises (the hash maps non-finite inputs to a
    definite cell rather than the reserved ``0``); see the PR's "Questions for
    review" on whether the area path should instead reject non-finite input.
    ``BaseException`` is used so a would-be PyO3 panic is caught, not just
    ``Exception``.
    """
    from mortie.tools import mort2healpix

    for bad in (np.nan, np.inf, -np.inf):
        area_err = point_err = None
        area_word = point_word = None
        try:
            area_word = int(geo2mort(bad, 0.0, order=29, points=False)[0])
        except BaseException as exc:
            area_err = type(exc)
        try:
            point_word = int(geo2mort(bad, 0.0, order=29, points=True)[0])
        except BaseException as exc:
            point_err = type(exc)
        assert area_err == point_err, f"non-finite failure mode mismatch for {bad}"
        if area_err is None:
            # Both produced a word: they must share the underlying nested cell.
            assert mort2healpix(area_word)[0] == mort2healpix(point_word)[0], (
                f"non-finite nested cell mismatch for {bad}"
            )


def test_point_broadcast_matches_elementwise():
    """The points=True broadcast branch (scalar lat + array lon) is correct."""
    lons = np.array([-122.0, -121.0, -120.0])
    out = geo2mort(45.0, lons, order=29, points=True)
    assert out.dtype == np.uint64 and out.shape == (3,)
    for i, lon in enumerate(lons):
        assert int(out[i]) == int(geo2mort(45.0, lon, order=29, points=True)[0])

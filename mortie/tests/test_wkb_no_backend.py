"""WKB ingest with no geometry backend installed (issue #157, phase 2).

The headline test of the issue: block **both** ``shapely`` and ``spherely``
from importing and show that WKB → coverage still succeeds.  Before #157 this
raised ``ImportError`` — ``from_wkb`` decoded through a backend and then leaned
on shapely's ring introspection — which made ``_require_backend``'s own claim
("mortie's runtime is numpy-only, so the backend is an optional extra") untrue
for anyone who touched WKB.

Every blob here is packed by hand, so the module has no geometry-library
dependency of its own: what the tests assert is that the *runtime* path does
not acquire one either.
"""

import struct
import sys
from contextlib import contextmanager

import numpy as np
import pytest

import mortie
from mortie import codec

BLOCKED = ("shapely", "spherely")


class _Blocker:
    """A ``sys.meta_path`` finder that refuses the geometry backends."""

    def find_spec(self, fullname, path=None, target=None):
        """Raise for a blocked module, and defer on everything else.

        Parameters
        ----------
        fullname : str
            The module being imported.
        path : optional
            Unused; part of the finder protocol.
        target : optional
            Unused; part of the finder protocol.

        Returns
        -------
        None
            Always, for modules that are not blocked — deferring to the
            finders behind this one.

        Raises
        ------
        ImportError
            If *fullname* is (or is inside) a blocked package.
        """
        if fullname.split(".")[0] in BLOCKED:
            raise ImportError(f"{fullname} is blocked for this test")
        return None


@contextmanager
def no_geometry_backend():
    """Make ``shapely`` and ``spherely`` unimportable for the duration.

    Already-imported copies are pulled out of :data:`sys.modules` (and put
    back afterwards), and :mod:`mortie.codec`'s cached backend is cleared,
    so the lazy gate re-resolves inside the block.

    Yields
    ------
    None
        With both backends unimportable.
    """
    saved_modules = {
        k: v for k, v in sys.modules.items() if k.split(".")[0] in BLOCKED
    }
    for k in saved_modules:
        del sys.modules[k]
    blocker = _Blocker()
    sys.meta_path.insert(0, blocker)
    saved_backend = codec._BACKEND
    codec._BACKEND = None
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved_modules)
        codec._BACKEND = saved_backend


def polygon_blob(rings):
    """Pack a little-endian Polygon WKB from ``(lon, lat)`` rings.

    Parameters
    ----------
    rings : list of list of tuple
        One list of ``(lon, lat)`` degree pairs per ring, exterior first.

    Returns
    -------
    bytes
        The WKB blob.
    """
    out = struct.pack("<BII", 1, 3, len(rings))
    for ring in rings:
        out += struct.pack("<I", len(ring))
        out += b"".join(struct.pack("<dd", x, y) for x, y in ring)
    return out


def linestring_blob(pts):
    """Pack a little-endian LineString WKB from ``(lon, lat)`` pairs.

    Parameters
    ----------
    pts : list of tuple
        The vertices as ``(lon, lat)`` degree pairs.

    Returns
    -------
    bytes
        The WKB blob.
    """
    return (
        struct.pack("<BII", 1, 2, len(pts))
        + b"".join(struct.pack("<dd", x, y) for x, y in pts)
    )


# An asymmetric footprint (lats in [-75, -71], lons in [10, 40]) with a hole.
OUTER = [(10, -75), (40, -75), (40, -71), (10, -71), (10, -75)]
HOLE = [(20, -74), (30, -74), (30, -72), (20, -72), (20, -74)]
POLY = polygon_blob([OUTER])
POLY_HOLE = polygon_blob([OUTER, HOLE])
LINE = linestring_blob([(10, -75), (40, -75), (40, -71)])


def test_the_backends_are_really_blocked():
    # Guard the guard: if the block silently stopped working, the headline
    # test below would pass for the wrong reason.
    with no_geometry_backend():
        for name in BLOCKED:
            with pytest.raises(ImportError):
                __import__(name)
        with pytest.raises(ImportError, match="requires a geometry backend"):
            codec._require_backend()


@pytest.mark.parametrize("blob", [POLY, POLY_HOLE, LINE])
def test_from_wkb_covers_without_any_backend(blob):
    # The point of issue #157: bytes in, cells out, no geometry library.
    with_backend = mortie.from_wkb(blob, order=6)
    with no_geometry_backend():
        without = mortie.from_wkb(blob, order=6)
    np.testing.assert_array_equal(np.asarray(without), np.asarray(with_backend))
    assert np.asarray(without).size > 0


def test_from_wkb_moc_and_stop_criteria_work_without_a_backend():
    with no_geometry_backend():
        moc = mortie.from_wkb(POLY_HOLE, order=10, moc=True)
        tol = mortie.from_wkb(POLY, order=10, moc=True, tolerance=1.0)
        budget = mortie.from_wkb(POLY, order=10, moc=True, max_cells=64)
    np.testing.assert_array_equal(
        moc, mortie.from_wkb(POLY_HOLE, order=10, moc=True)
    )
    assert tol.size and budget.size


def test_reader_errors_are_backend_free_too():
    # The refusals must not go looking for a backend on their way out.
    with no_geometry_backend():
        with pytest.raises(ValueError, match="truncated WKB"):
            mortie.from_wkb(POLY[:12], order=6)
        with pytest.raises(ValueError, match="unsupported WKB geometry type"):
            mortie.from_wkb(struct.pack("<BId", 1, 1, 0.0) + b"\x00" * 8, order=6)
        with pytest.raises(ValueError, match="empty geometry"):
            mortie.from_wkb(polygon_blob([]), order=6)
        with pytest.raises(ValueError, match="is not closed"):
            mortie.from_wkb(polygon_blob([OUTER[:-1]]), order=6)


def test_the_input_contract_is_backend_free_too():
    # The hex/buffer coercion is pure Python plus the reader, so the accepted
    # spellings and the refusal hold with no geometry library around either.
    want = mortie.from_wkb(POLY, order=6)
    with no_geometry_backend():
        np.testing.assert_array_equal(mortie.from_wkb(POLY.hex(), order=6), want)
        np.testing.assert_array_equal(
            mortie.from_wkb(bytearray(POLY), order=6), want
        )
        with pytest.raises(TypeError, match="WKB input must be"):
            mortie.from_wkb(list(POLY), order=6)


def test_wkt_and_emit_still_require_a_backend():
    # The scope line of #157, pinned: only WKB ingest went backend-free.
    # `from_wkt` has no Rust parser behind it, and emit hands back a backend
    # geometry object by definition.
    with no_geometry_backend():
        with pytest.raises(ImportError, match="requires a geometry backend"):
            mortie.from_wkt("POLYGON ((10 -75, 40 -75, 40 -71, 10 -75))")
        with pytest.raises(ImportError, match="requires a geometry backend"):
            mortie.to_geometry(mortie.from_wkb(POLY, order=6))
        with pytest.raises(ImportError, match="requires a geometry backend"):
            codec._geometry_from_wkb(POLY)


def test_from_wkbs_batches_without_any_backend():
    # The plural twin is backend-free for the same reason the scalar is: one
    # Rust parser, no geometry object anywhere on the path (issue #157
    # phase 3).  Crossing a chunk boundary here too, so the chunked
    # copy-then-release loop is what runs, not just its first iteration.
    blobs = [POLY, POLY_HOLE] * 1100
    values, offsets = mortie.from_wkbs(blobs, order=6)
    with no_geometry_backend():
        got_values, got_offsets = mortie.from_wkbs(blobs, order=6)
        # And the refusals stay backend-free as well.
        with pytest.raises(ValueError, match=r"^blob 1: .*truncated WKB"):
            mortie.from_wkbs([POLY, POLY[:12]], order=6)
        with pytest.raises(ValueError, match=r"^blob 0: .*linear geometry"):
            mortie.from_wkbs([LINE], order=6)
    np.testing.assert_array_equal(got_values, values)
    np.testing.assert_array_equal(got_offsets, offsets)
    assert offsets[-1] == values.size and offsets[0] == 0

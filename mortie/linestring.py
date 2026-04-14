"""Linestring / multi-linestring morton coverage.

Compute morton indices tracing an open polyline (or list of polylines) at a
given HEALPix order. The result is a contiguous chain of cells: along each
segment between consecutive vertices, intermediate cells are filled in by
great-circle interpolation at half cell-resolution spacing.
"""

import numpy as np


def _is_multi(lats):
    """Return True if *lats* is a sequence of sequences (multi-linestring)."""
    if isinstance(lats, np.ndarray):
        return lats.ndim == 2
    if isinstance(lats, (list, tuple)) and len(lats) > 0:
        return isinstance(lats[0], (list, tuple, np.ndarray))
    return False


def _single_linestring_coverage(lats, lons, order):
    """Coverage for one open polyline."""
    lats = np.ascontiguousarray(np.asarray(lats, dtype=np.float64).ravel())
    lons = np.ascontiguousarray(np.asarray(lons, dtype=np.float64).ravel())

    if lats.shape != lons.shape:
        raise ValueError("lats and lons must have the same length")
    if lats.size < 2:
        raise ValueError("Need at least 2 vertices for a linestring")
    if not np.all(np.isfinite(lats)) or not np.all(np.isfinite(lons)):
        raise ValueError("lats and lons must not contain NaN or infinity")

    from . import _rustie

    return np.asarray(_rustie.rust_linestring_coverage(lats, lons, order))


def linestring_coverage(lats, lons, order=18):
    """Compute morton indices tracing a linestring.

    For a single open polyline, returns a 1-D ``numpy.ndarray`` of sorted,
    unique morton indices at the requested HEALPix order. Cells along each
    segment are contiguous: gaps between vertex cells are filled by sampling
    the great-circle arc at half cell-resolution spacing.

    For **multi-linestrings**, pass *lats* and *lons* as lists of 1-D arrays
    (one per line). The result is a ``list`` of ``numpy.ndarray``, one per
    input line, preserving per-line resolution. Per-line arrays are NOT
    deduplicated across lines — if the caller wants the union, they can
    concatenate and call ``np.unique`` themselves.

    Parameters
    ----------
    lats : array_like or list of array_like
        Vertex latitudes in degrees. For a single line, a 1-D array with at
        least 2 vertices. For a multi-linestring, a list of such arrays.
    lons : array_like or list of array_like
        Vertex longitudes in degrees. Must match the structure of *lats*.
    order : int, optional
        HEALPix depth / tessellation order (1–18). Default 18.

    Returns
    -------
    numpy.ndarray or list of numpy.ndarray
        Single line → 1-D int64 array, sorted and unique.
        Multi-linestring → list of such arrays, one per input line.

    Raises
    ------
    ValueError
        If fewer than 2 vertices, mismatched lengths, invalid *order*, or
        NaN/infinity in coordinates.

    Examples
    --------
    Single linestring::

        >>> import mortie
        >>> lats = [40.0, 50.0, 45.0]
        >>> lons = [-120.0, -110.0, -100.0]
        >>> cells = mortie.linestring_coverage(lats, lons, order=6)

    Multi-linestring (list of arrays; lengths may differ)::

        >>> lats_parts = [[40.0, 50.0], [10.0, 20.0, 15.0]]
        >>> lons_parts = [[-120.0, -120.0], [-80.0, -70.0, -60.0]]
        >>> per_line = mortie.linestring_coverage(lats_parts, lons_parts, order=6)
        >>> [arr.shape for arr in per_line]
    """
    if not 1 <= order <= 18:
        raise ValueError("Order must be between 1 and 18")

    if _is_multi(lats):
        if len(lats) != len(lons):
            raise ValueError("lats and lons must have the same number of parts")
        return [_single_linestring_coverage(la, lo, order) for la, lo in zip(lats, lons)]

    return _single_linestring_coverage(lats, lons, order)

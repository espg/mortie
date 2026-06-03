"""Polygon-to-morton coverage.

Compute the set of morton indices at a given order that completely cover
a polygon defined by lat/lon vertices.  Supports single and multipart
polygons.
"""

import numpy as np


def _is_multipart(lats):
    """Check if *lats* is a sequence of sequences (multipart polygon)."""
    if isinstance(lats, np.ndarray):
        return lats.ndim == 2
    if isinstance(lats, (list, tuple)) and len(lats) > 0:
        return isinstance(lats[0], (list, tuple, np.ndarray))
    return False


def _single_coverage(lats, lons, order):
    """Coverage for one polygon ring."""
    lats = np.asarray(lats, dtype=np.float64).ravel()
    lons = np.asarray(lons, dtype=np.float64).ravel()

    if lats.shape != lons.shape:
        raise ValueError("lats and lons must have the same length")
    if lats.size < 3:
        raise ValueError("Need at least 3 vertices for a polygon")
    if not np.all(np.isfinite(lats)) or not np.all(np.isfinite(lons)):
        raise ValueError("lats and lons must not contain NaN or infinity")

    # Strip duplicate closing vertex (first == last) if present
    if lats[0] == lats[-1] and lons[0] == lons[-1] and lats.size > 3:
        lats = lats[:-1].copy()
        lons = lons[:-1].copy()

    from . import _rustie

    return np.asarray(_rustie.rust_polygon_coverage(lats, lons, order))


def morton_coverage(lats, lons, order=18):
    """Compute morton indices covering a polygon defined by lat/lon vertices.

    Given a polygon (as arrays of vertex latitudes and longitudes), returns the
    set of morton indices at the requested HEALPix order that completely cover
    the polygon interior.  The coverage is optimally compact — it includes all
    boundary cells plus every cell whose centre lies inside the polygon.

    For **multipart polygons**, pass *lats* and *lons* as lists of arrays
    (one per part).  The coverage of all parts is unioned.

    Parameters
    ----------
    lats : array_like or list of array_like
        Vertex latitudes in degrees.  For a single polygon, a 1-D array
        with at least 3 vertices.  For multipart polygons, a list of
        such arrays.
    lons : array_like or list of array_like
        Vertex longitudes in degrees.  Must match the structure of *lats*.
    order : int, optional
        HEALPix depth / tessellation order (1–18).  Default 18.

    Returns
    -------
    numpy.ndarray
        Sorted 1-D array of unique morton indices (dtype ``int64``).

    Raises
    ------
    ValueError
        If fewer than 3 vertices, mismatched lengths, invalid order,
        or coordinates containing NaN/infinity.

    Notes
    -----
    - Self-intersecting polygons produce undefined results.
    - Polygons with holes are not supported; pass the outer ring only.
    - The algorithm uses gnomonic projection centered on each test point
      with a winding-number PIP test, which works correctly for polygons
      in any hemisphere.

    Examples
    --------
    Single polygon:

    >>> import mortie
    >>> lats = [40.0, 50.0, 45.0]
    >>> lons = [-120.0, -120.0, -110.0]
    >>> cells = mortie.morton_coverage(lats, lons, order=6)

    Multipart polygon:

    >>> lats_parts = [[40.0, 50.0, 45.0], [10.0, 20.0, 15.0]]
    >>> lons_parts = [[-120.0, -120.0, -110.0], [-80.0, -80.0, -70.0]]
    >>> cells = mortie.morton_coverage(lats_parts, lons_parts, order=6)
    """
    if not 1 <= order <= 18:
        raise ValueError("Order must be between 1 and 18")

    if _is_multipart(lats):
        if len(lats) != len(lons):
            raise ValueError("lats and lons must have the same number of parts")
        parts = [_single_coverage(la, lo, order) for la, lo in zip(lats, lons)]
        return np.unique(np.concatenate(parts))

    return _single_coverage(lats, lons, order)


def morton_coverage_moc(lats, lons, order=18, tolerance=None, max_cells=None):
    """Compute polygon coverage as a compact Multi-Order Coverage (MOC) map.

    Unlike :func:`morton_coverage`, which returns a flat list of cells all at
    ``order``, this returns a *mixed-order* set: coarse cells for the interior
    and fine cells (down to ``order``) along the boundary.  Because a mortie
    morton index self-encodes its order, the result is still a 1-D ``int64``
    array — typically far smaller than the flat cover.

    Optional **adaptive stop criteria** (mutually exclusive) trade boundary
    precision for fewer cells and faster runtime:

    Parameters
    ----------
    lats, lons : array_like
        Vertex latitudes / longitudes in degrees (single polygon ring).
    order : int, optional
        Finest HEALPix order (1–18).  Default 18.
    tolerance : float, optional
        Stop refining a boundary cell once its angular radius (in **degrees**)
        drops to this value, even if coarser than ``order``.  Approximate,
        coarser boundary.
    max_cells : int, optional
        Best-first budget: refine the largest boundary cells until about this
        many cells, giving an adaptive mixed-order boundary (fine where it
        wiggles, coarse where it is straight).  Soft target.

    Returns
    -------
    numpy.ndarray
        Sorted 1-D array of mixed-order morton indices (``int64``).

    See Also
    --------
    morton_coverage : flat single-order cover.
    """
    if not 1 <= order <= 18:
        raise ValueError("Order must be between 1 and 18")
    if tolerance is not None and max_cells is not None:
        raise ValueError("pass at most one of tolerance / max_cells")

    lats = np.asarray(lats, dtype=np.float64).ravel()
    lons = np.asarray(lons, dtype=np.float64).ravel()
    if lats.shape != lons.shape:
        raise ValueError("lats and lons must have the same length")
    if lats.size < 3:
        raise ValueError("Need at least 3 vertices for a polygon")
    if not np.all(np.isfinite(lats)) or not np.all(np.isfinite(lons)):
        raise ValueError("lats and lons must not contain NaN or infinity")
    if lats[0] == lats[-1] and lons[0] == lons[-1] and lats.size > 3:
        lats = lats[:-1].copy()
        lons = lons[:-1].copy()

    tol_rad = None if tolerance is None else np.radians(float(tolerance))

    from . import _rustie

    return np.asarray(
        _rustie.rust_polygon_coverage_moc(lats, lons, order, tol_rad, max_cells)
    )

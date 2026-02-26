"""Polygon-to-morton coverage.

Compute the set of morton indices at a given order that completely cover
a polygon defined by lat/lon vertices.
"""

import numpy as np


def morton_coverage(lats, lons, order=18):
    """Compute morton indices covering a polygon defined by lat/lon vertices.

    Given a polygon (as arrays of vertex latitudes and longitudes), returns the
    set of morton indices at the requested HEALPix order that completely cover
    the polygon interior.  The coverage is optimally compact — it includes all
    boundary cells plus every cell whose centre lies inside the polygon.

    Parameters
    ----------
    lats : array_like
        Vertex latitudes in degrees.  At least 3 vertices required.
    lons : array_like
        Vertex longitudes in degrees.  Same length as *lats*.
    order : int, optional
        HEALPix depth / tessellation order (1–18).  Default 18.

    Returns
    -------
    numpy.ndarray
        Sorted 1-D array of unique morton indices (dtype ``int64``).

    Raises
    ------
    NotImplementedError
        If *lats* / *lons* describe a polygon with holes (not yet supported).
    ValueError
        If fewer than 3 vertices, mismatched lengths, or invalid order.

    Examples
    --------
    >>> import mortie
    >>> lats = [40.0, 50.0, 45.0]
    >>> lons = [-120.0, -120.0, -110.0]
    >>> cells = mortie.morton_coverage(lats, lons, order=6)
    """
    lats = np.asarray(lats, dtype=np.float64).ravel()
    lons = np.asarray(lons, dtype=np.float64).ravel()

    if lats.shape != lons.shape:
        raise ValueError("lats and lons must have the same length")
    if lats.size < 3:
        raise ValueError("Need at least 3 vertices for a polygon")
    if not 1 <= order <= 18:
        raise ValueError("Order must be between 1 and 18")

    # Close the polygon if needed (first == last vertex)
    if lats[0] == lats[-1] and lons[0] == lons[-1] and lats.size > 3:
        lats = lats[:-1]
        lons = lons[:-1]

    from . import _rustie

    result = _rustie.rust_polygon_coverage(lats, lons, order)
    return np.asarray(result)

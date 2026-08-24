"""Cell-set dilation over morton sets.

:func:`morton_buffer` takes morton indices and returns morton indices -- the
k-cell expansion ring around a set, not a geometric operation on it -- and
:func:`morton_buffer_meters` is the convenience wrapper that picks ``k`` from a
metre width. That is why this is its own module rather than part of
``mortie.geometry``, and it is the Python side of ``src_rust/src/buffer.rs``.

Split out of ``mortie.tools`` (issue #159) so the Python surface mirrors the
Rust tree's own decomposition. The names stay flat on the package
(``mortie.morton_buffer``): this module is where they live, not how they are
spelled.
"""

import numpy as np

from . import _rustie
from .orders import infer_order_from_morton


def morton_buffer(morton_indices, k=1):
    """Compute the k-cell border around a set of morton indices.

    Returns only cells NOT in the input set (the expansion ring).
    User can union: ``np.union1d(morton_indices, border)``

    **Not batch vectorized**: one cell set per call — the ring is a set
    operation over the whole input, not an elementwise map.

    Parameters
    ----------
    morton_indices : array-like
        Morton indices, all at the same order.
    k : int, optional
        Border width in cells (default 1, 8-connected neighbors).
        k=1 gives the immediate ring, k=2 gives a 2-cell border, etc.

    Returns
    -------
    border : ndarray
        Sorted array of morton indices for the border cells.

    Raises
    ------
    ValueError
        If indices have mixed orders or k is out of range.
    """
    morton_indices = np.asarray(morton_indices, dtype=np.uint64)
    return _rustie.rust_morton_buffer(np.ascontiguousarray(morton_indices), k)


# Earth mean radius in meters (IUGG 2015 mean).
_EARTH_RADIUS_M = 6_371_008.7714


def morton_buffer_meters(morton_indices, width_m):
    """Approximate meter-width buffer around a set of morton cells.

    This is a convenience wrapper around :func:`morton_buffer` that picks
    ``k`` from the cells' HEALPix order so the resulting ring is roughly
    *width_m* meters wide. The input cells are assumed to all be at the same
    order.

    .. warning::
       **This is an approximate buffer.** The achieved width is rounded
       UP to the nearest whole HEALPix cell width — so the result always
       covers *at least* ``width_m`` meters, but may cover up to one cell
       width more. For order 18 cells (~30 m) the granularity is fine; at
       coarser orders it can be substantial. If you need a precise buffer,
       pick an order whose cell width is small relative to ``width_m`` and
       convert your input cells to that order first.

    The cell width used for the calculation is the HEALPix angular
    resolution ``sqrt(pi/3) / nside`` converted to meters via the Earth's
    mean radius (6,371,008.77 m).

    **Not batch vectorized**: one cell set per call, as :func:`morton_buffer`.

    Parameters
    ----------
    morton_indices : array-like
        Morton indices, all at the same HEALPix order.
    width_m : float
        Desired buffer width in meters (must be > 0).

    Returns
    -------
    border : ndarray
        Sorted array of morton indices for the border cells (NOT including
        the input cells). Union with the input if you want the filled ring:
        ``np.union1d(morton_indices, border)``.

    Raises
    ------
    ValueError
        If ``width_m`` is non-positive, the input array is empty, or the
        cells are at mixed orders.

    Examples
    --------
    >>> import mortie, numpy as np
    >>> cells = mortie.linestring_coverage([10.0, 20.0], [30.0, 40.0], order=10)
    >>> border = mortie.morton_buffer_meters(cells, width_m=5000.0)
    >>> expanded = np.union1d(cells, border)
    """
    morton_indices = np.asarray(morton_indices, dtype=np.uint64)
    if morton_indices.size == 0:
        raise ValueError("morton_indices must be non-empty")
    if not (width_m > 0):
        raise ValueError("width_m must be positive")

    # Infer order from the first cell. rust_morton_buffer will itself reject
    # mixed-order inputs downstream.
    order = infer_order_from_morton(int(morton_indices.flat[0]))
    if order < 1:
        raise ValueError("Could not infer a valid order from the input cells")

    nside = 1 << order
    cell_width_m = _EARTH_RADIUS_M * np.sqrt(np.pi / 3.0) / nside

    # Round UP so the buffer covers AT LEAST the requested width.
    k = int(np.ceil(width_m / cell_width_m))
    if k < 1:
        k = 1

    return morton_buffer(morton_indices, k=k)

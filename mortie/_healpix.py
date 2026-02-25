"""
HEALPix operations via the compiled Rust extension (``_rustie``).

Provides ``ang2pix``, ``pix2ang``, ``boundaries``, ``vec2ang``, and
``order2nside`` — all backed by the ``healpix`` Rust crate.

All public functions work in **degrees** and the **NESTED** scheme.
"""

import math
import numpy as np

from mortie._rustie import rust_ang2pix, rust_pix2ang, rust_boundaries, rust_vec2ang


def order2nside(order):
    """Convert HEALPix order to nside.  Trivial: ``2**order``."""
    return 2**order


def ang2pix(nside, lon, lat):
    """coords (degrees) → NESTED pixel index via Rust healpix crate."""
    depth = int(math.log2(nside))
    scalar = np.ndim(lon) == 0 and np.ndim(lat) == 0
    if scalar:
        result = rust_ang2pix(depth, float(lon), float(lat))
        return np.int64(result)
    lon_arr = np.ascontiguousarray(lon, dtype=np.float64)
    lat_arr = np.ascontiguousarray(lat, dtype=np.float64)
    return np.asarray(rust_ang2pix(depth, lon_arr, lat_arr), dtype=np.int64)


def pix2ang(nside, pixel):
    """NESTED pixel index → (lon, lat) in degrees via Rust healpix crate."""
    depth = int(math.log2(nside))
    scalar = np.ndim(pixel) == 0
    if scalar:
        result = rust_pix2ang(depth, int(pixel))
        return np.float64(result[0]), np.float64(result[1])
    pixel_arr = np.ascontiguousarray(pixel, dtype=np.int64)
    result = rust_pix2ang(depth, pixel_arr)
    return np.asarray(result[0], dtype=np.float64), \
        np.asarray(result[1], dtype=np.float64)


def boundaries(nside, pixel):
    """Cell boundary vertices as 3-D unit vectors.

    Returns ndarray of shape ``(3, 4)`` for scalar pixel or
    ``(N, 3, 4)`` for array pixel.
    """
    depth = int(math.log2(nside))
    if np.ndim(pixel) == 0:
        return rust_boundaries(depth, int(pixel))  # (3, 4)
    return rust_boundaries(depth,
                           np.ascontiguousarray(pixel, dtype=np.int64))


def vec2ang(vectors):
    """3-D unit vectors → (theta, phi) in radians.

    *vectors* has shape ``(N, 3)``.  Returns ``(theta, phi)`` where
    ``theta`` is colatitude and ``phi`` is longitude, both in radians.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim > 2:
        vectors = vectors.reshape(-1, 3)
    vectors = np.ascontiguousarray(vectors, dtype=np.float64)
    return rust_vec2ang(vectors)

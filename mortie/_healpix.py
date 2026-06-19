"""
HEALPix operations via the compiled Rust extension (``_rustie``).

Provides ``ang2pix``, ``pix2ang``, ``boundaries``, ``vec2ang``, and
``order2nside`` — all backed by the ``healpix`` Rust crate.

All public functions work in **degrees** and the **NESTED** scheme.  The Rust
layer is **depth-native** (it indexes by HEALPix order/depth, not nside), so
``ang2pix``/``pix2ang``/``boundaries`` take ``depth`` directly rather than
``log2``-ing an nside on every call — callers already hold the order.
"""

import numpy as np

from mortie._rustie import rust_ang2pix, rust_pix2ang, rust_boundaries, rust_vec2ang


def order2nside(order):
    """Convert HEALPix order to nside.  Trivial: ``2**order``."""
    return 2**order


def ang2pix(depth, lon, lat):
    """coords (degrees) → NESTED pixel index via Rust healpix crate.

    ``depth`` is the HEALPix order (``nside = 2**depth``)."""
    scalar = np.ndim(lon) == 0 and np.ndim(lat) == 0
    if scalar:
        result = rust_ang2pix(depth, float(lon), float(lat))
        return np.int64(result)
    lon_arr = np.ascontiguousarray(lon, dtype=np.float64)
    lat_arr = np.ascontiguousarray(lat, dtype=np.float64)
    return np.asarray(rust_ang2pix(depth, lon_arr, lat_arr), dtype=np.int64)


def pix2ang(depth, pixel):
    """NESTED pixel index → (lon, lat) in degrees via Rust healpix crate.

    ``depth`` is the HEALPix order (``nside = 2**depth``)."""
    scalar = np.ndim(pixel) == 0
    if scalar:
        result = rust_pix2ang(depth, int(pixel))
        return np.float64(result[0]), np.float64(result[1])
    pixel_arr = np.ascontiguousarray(pixel, dtype=np.int64)
    result = rust_pix2ang(depth, pixel_arr)
    return np.asarray(result[0], dtype=np.float64), \
        np.asarray(result[1], dtype=np.float64)


def boundaries(depth, pixel, step=1):
    """Cell boundary vertices as 3-D unit vectors.

    Parameters
    ----------
    depth : int
        HEALPix order/depth (``nside = 2**depth``).
    pixel : int or array-like
        NESTED pixel index(es).
    step : int, optional
        Points per side (default 1 = 4 corners only).
        Use step=32 for 128 boundary points that accurately trace
        curved cell edges near the poles.

    Returns
    -------
    ndarray
        Shape ``(3, 4*step)`` for scalar pixel or
        ``(N, 3, 4*step)`` for array pixel.
    """
    if np.ndim(pixel) == 0:
        return rust_boundaries(depth, int(pixel), step)
    return rust_boundaries(depth,
                           np.ascontiguousarray(pixel, dtype=np.int64),
                           step)


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

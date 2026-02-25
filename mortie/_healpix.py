"""
HEALPix backend abstraction layer.

Provides a unified API for HEALPix operations, supporting both ``healpy``
and ``cdshealpix`` as interchangeable backends.

Backend selection order:

1. ``MORTIE_HEALPIX_BACKEND`` environment variable (``healpy`` or ``cdshealpix``)
2. Try ``cdshealpix`` first (lighter weight, Rust-backed, ARM64 wheels)
3. Fall back to ``healpy``
4. Raise ``ImportError`` with helpful message if neither is installed

All public functions work in **degrees** and the **NESTED** scheme.
"""

import os
import math
import numpy as np

_BACKEND = None  # will be set at module load


def order2nside(order):
    """Convert HEALPix order to nside.  Trivial: ``2**order``."""
    return 2**order


# ---------------------------------------------------------------------------
# healpy backend
# ---------------------------------------------------------------------------

def _ang2pix_healpy(nside, lon, lat):
    """coords (degrees) → NESTED pixel index via healpy."""
    import healpy as hp
    return hp.ang2pix(nside, lon, lat, lonlat=True, nest=True)


def _pix2ang_healpy(nside, pixel):
    """NESTED pixel index → (lon, lat) in degrees via healpy."""
    import healpy as hp
    return hp.pix2ang(nside, pixel, nest=True, lonlat=True)


def _boundaries_healpy(nside, pixel):
    """Cell boundary vertices as 3-D unit vectors via healpy.

    Returns ndarray of shape ``(3, 4)`` for scalar pixel or
    ``(N, 3, 4)`` for array pixel.
    """
    import healpy as hp
    return hp.boundaries(nside, pixel, nest=True, step=1)


def _vec2ang_healpy(vectors):
    """3-D unit vectors → (theta, phi) in radians via healpy.

    *vectors* has shape ``(N, 3)``.  Returns ``(theta, phi)`` where
    ``theta`` is colatitude and ``phi`` is longitude, both in radians.
    """
    import healpy as hp
    return hp.vec2ang(vectors)


# ---------------------------------------------------------------------------
# cdshealpix backend
# ---------------------------------------------------------------------------

def _ang2pix_cds(nside, lon, lat):
    """coords (degrees) → NESTED pixel index via cdshealpix."""
    from cdshealpix import nested
    from astropy.units import deg

    depth = int(math.log2(nside))
    lon_arr = np.atleast_1d(np.asarray(lon, dtype=np.float64))
    lat_arr = np.atleast_1d(np.asarray(lat, dtype=np.float64))
    result = nested.lonlat_to_healpix(lon_arr * deg, lat_arr * deg, depth)
    # cdshealpix returns uint64 ndarray
    result = np.asarray(result, dtype=np.int64)
    if np.ndim(lon) == 0 and np.ndim(lat) == 0:
        return np.int64(result[0])  # numpy scalar, not Python int
    return result


def _pix2ang_cds(nside, pixel):
    """NESTED pixel index → (lon, lat) in degrees via cdshealpix."""
    from cdshealpix import nested

    depth = int(math.log2(nside))
    pixel_arr = np.atleast_1d(np.asarray(pixel, dtype=np.uint64))
    lon_obj, lat_obj = nested.healpix_to_lonlat(pixel_arr, depth)
    lon_deg = np.asarray(lon_obj.deg, dtype=np.float64)
    lat_deg = np.asarray(lat_obj.deg, dtype=np.float64)
    if np.ndim(pixel) == 0:
        return np.float64(lon_deg[0]), np.float64(lat_deg[0])
    return lon_deg, lat_deg


def _boundaries_cds(nside, pixel):
    """Cell boundary vertices as 3-D unit vectors via cdshealpix.

    Returns same shape as healpy: ``(3, 4)`` for scalar pixel or
    ``(N, 3, 4)`` for array pixel.
    """
    from cdshealpix import nested

    depth = int(math.log2(nside))
    scalar = np.ndim(pixel) == 0
    pixel_arr = np.atleast_1d(np.asarray(pixel, dtype=np.uint64))

    # vertices returns (N, 4) lon and (N, 4) lat Quantity arrays
    lon_q, lat_q = nested.vertices(pixel_arr, depth, step=1)
    lon_rad = np.asarray(lon_q.rad, dtype=np.float64)  # (N, 4)
    lat_rad = np.asarray(lat_q.rad, dtype=np.float64)  # (N, 4)

    # cdshealpix returns vertices in a different starting order than healpy;
    # roll by 2 positions to match healpy convention.
    lon_rad = np.roll(lon_rad, 2, axis=1)
    lat_rad = np.roll(lat_rad, 2, axis=1)

    # Convert lon/lat (radians) to 3-D unit vectors
    cos_lat = np.cos(lat_rad)
    x = cos_lat * np.cos(lon_rad)  # (N, 4)
    y = cos_lat * np.sin(lon_rad)  # (N, 4)
    z = np.sin(lat_rad)            # (N, 4)

    if scalar:
        # shape (3, 4) to match healpy scalar output
        return np.array([x[0], y[0], z[0]])
    else:
        # shape (N, 3, 4) to match healpy array output
        return np.stack([x, y, z], axis=1)


def _vec2ang_cds(vectors):
    """3-D unit vectors → (theta, phi) in radians.

    Pure math — no library dependency.  *vectors* has shape ``(N, 3)``.
    Returns ``(theta, phi)`` matching healpy convention:
    ``theta`` = colatitude (0 at north pole), ``phi`` = longitude.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    phi = np.arctan2(y, x)
    phi[phi < 0] += 2 * np.pi
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    return theta, phi


# ---------------------------------------------------------------------------
# Backend detection and dispatch
# ---------------------------------------------------------------------------

def _detect_backend():
    """Detect and return the backend name."""
    env = os.environ.get('MORTIE_HEALPIX_BACKEND', '').strip().lower()

    if env == 'healpy':
        try:
            import healpy  # noqa: F401
            return 'healpy'
        except ImportError:
            raise ImportError(
                "MORTIE_HEALPIX_BACKEND='healpy' but healpy is not installed. "
                "Install it with: pip install mortie[healpy]"
            )

    if env == 'cdshealpix':
        try:
            import cdshealpix  # noqa: F401
            return 'cdshealpix'
        except ImportError:
            raise ImportError(
                "MORTIE_HEALPIX_BACKEND='cdshealpix' but cdshealpix is not "
                "installed. Install it with: pip install mortie[cdshealpix]"
            )

    # Auto-detect: try cdshealpix first (lighter), then healpy
    if env == '' or env not in ('healpy', 'cdshealpix'):
        try:
            import cdshealpix  # noqa: F401
            return 'cdshealpix'
        except ImportError:
            pass

        try:
            import healpy  # noqa: F401
            return 'healpy'
        except ImportError:
            pass

    raise ImportError(
        "No HEALPix backend available. Install one with:\n"
        "  pip install mortie[healpy]       # healpy (C/Fortran)\n"
        "  pip install mortie[cdshealpix]   # cdshealpix (Rust, ARM64 wheels)\n"
        "  pip install mortie[all]          # both"
    )


_BACKEND = _detect_backend()

if _BACKEND == 'healpy':
    ang2pix = _ang2pix_healpy
    pix2ang = _pix2ang_healpy
    boundaries = _boundaries_healpy
    vec2ang = _vec2ang_healpy
elif _BACKEND == 'cdshealpix':
    ang2pix = _ang2pix_cds
    pix2ang = _pix2ang_cds
    boundaries = _boundaries_cds
    vec2ang = _vec2ang_cds

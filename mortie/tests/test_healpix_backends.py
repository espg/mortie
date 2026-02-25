"""
Cross-comparison tests for healpy vs cdshealpix backends.

Skipped automatically if only one backend is installed.  Run with both
backends available to verify numerical equivalence.
"""

import numpy as np
import pytest

# Check backend availability
_has_healpy = False
_has_cdshealpix = False

try:
    import healpy  # noqa: F401
    _has_healpy = True
except ImportError:
    pass

try:
    import cdshealpix  # noqa: F401
    _has_cdshealpix = True
except ImportError:
    pass

_both = _has_healpy and _has_cdshealpix
_reason = "Both healpy and cdshealpix must be installed for cross-backend tests"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _both, reason=_reason),
]


# Import both backend implementations directly
from mortie._healpix import (
    order2nside,
    _ang2pix_healpy,
    _ang2pix_cds,
    _pix2ang_healpy,
    _pix2ang_cds,
    _boundaries_healpy,
    _boundaries_cds,
    _vec2ang_healpy,
    _vec2ang_cds,
)


class TestOrder2Nside:
    """order2nside is trivial (2**order), but verify it."""

    def test_known_values(self):
        for order in range(19):
            assert order2nside(order) == 2**order


class TestAng2Pix:
    """ang2pix: lon/lat (degrees) → NESTED pixel index."""

    def test_scalar(self):
        nside = order2nside(10)
        for lon, lat in [(0.0, 0.0), (45.0, 45.0), (-132.0, -78.5), (180.0, 89.0)]:
            hp_result = _ang2pix_healpy(nside, lon, lat)
            cds_result = _ang2pix_cds(nside, lon, lat)
            assert hp_result == cds_result, f"Mismatch at ({lon}, {lat})"

    def test_array(self):
        nside = order2nside(18)
        rng = np.random.default_rng(42)
        lons = rng.uniform(-180, 180, 500)
        lats = rng.uniform(-90, 90, 500)
        hp_result = _ang2pix_healpy(nside, lons, lats)
        cds_result = _ang2pix_cds(nside, lons, lats)
        np.testing.assert_array_equal(hp_result, cds_result)


class TestPix2Ang:
    """pix2ang: NESTED pixel → (lon, lat) in degrees."""

    def test_scalar(self):
        nside = order2nside(10)
        for pixel in [0, 100, 12582911]:
            hp_lon, hp_lat = _pix2ang_healpy(nside, pixel)
            cds_lon, cds_lat = _pix2ang_cds(nside, pixel)
            assert abs(hp_lon - cds_lon) < 1e-10, f"lon mismatch at pixel {pixel}"
            assert abs(hp_lat - cds_lat) < 1e-10, f"lat mismatch at pixel {pixel}"

    def test_array(self):
        nside = order2nside(12)
        pixels = np.array([0, 10, 100, 1000, 10000], dtype=np.int64)
        hp_lon, hp_lat = _pix2ang_healpy(nside, pixels)
        cds_lon, cds_lat = _pix2ang_cds(nside, pixels)
        np.testing.assert_allclose(hp_lon, cds_lon, atol=1e-10)
        np.testing.assert_allclose(hp_lat, cds_lat, atol=1e-10)


class TestBoundaries:
    """boundaries: pixel → 3-D unit vectors of cell corners."""

    def test_scalar(self):
        nside = order2nside(6)
        for pixel in [0, 42, 49151]:
            hp_b = _boundaries_healpy(nside, pixel)
            cds_b = _boundaries_cds(nside, pixel)
            assert hp_b.shape == cds_b.shape == (3, 4)
            np.testing.assert_allclose(hp_b, cds_b, atol=1e-10)

    def test_array(self):
        nside = order2nside(6)
        pixels = np.array([0, 42, 100, 49151], dtype=np.int64)
        hp_b = _boundaries_healpy(nside, pixels)
        cds_b = _boundaries_cds(nside, pixels)
        assert hp_b.shape == cds_b.shape
        np.testing.assert_allclose(hp_b, cds_b, atol=1e-10)


class TestVec2Ang:
    """vec2ang: 3-D unit vectors → (theta, phi) in radians."""

    def test_known_vectors(self):
        vectors = np.array([
            [1.0, 0.0, 0.0],   # equator, prime meridian
            [0.0, 1.0, 0.0],   # equator, 90E
            [0.0, 0.0, 1.0],   # north pole
            [0.0, 0.0, -1.0],  # south pole
        ])
        hp_theta, hp_phi = _vec2ang_healpy(vectors)
        cds_theta, cds_phi = _vec2ang_cds(vectors)
        np.testing.assert_allclose(hp_theta, cds_theta, atol=1e-14)
        np.testing.assert_allclose(hp_phi, cds_phi, atol=1e-14)

    def test_random_unit_vectors(self):
        rng = np.random.default_rng(42)
        raw = rng.standard_normal((200, 3))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        vectors = raw / norms
        hp_theta, hp_phi = _vec2ang_healpy(vectors)
        cds_theta, cds_phi = _vec2ang_cds(vectors)
        np.testing.assert_allclose(hp_theta, cds_theta, atol=1e-14)
        np.testing.assert_allclose(hp_phi, cds_phi, atol=1e-14)


class TestEndToEnd:
    """Round-trip: ang2pix → pix2ang should return close to original."""

    def test_roundtrip_both_backends(self):
        nside = order2nside(12)
        rng = np.random.default_rng(42)
        lons = rng.uniform(-180, 180, 100)
        lats = rng.uniform(-85, 85, 100)

        # healpy round-trip
        hp_pix = _ang2pix_healpy(nside, lons, lats)
        hp_lon2, hp_lat2 = _pix2ang_healpy(nside, hp_pix)

        # cdshealpix round-trip
        cds_pix = _ang2pix_cds(nside, lons, lats)
        cds_lon2, cds_lat2 = _pix2ang_cds(nside, cds_pix)

        # Same pixels
        np.testing.assert_array_equal(hp_pix, cds_pix)
        # Same reconstructed coordinates
        np.testing.assert_allclose(hp_lon2, cds_lon2, atol=1e-10)
        np.testing.assert_allclose(hp_lat2, cds_lat2, atol=1e-10)

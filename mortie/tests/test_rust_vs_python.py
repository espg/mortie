"""
Golden-output regression tests for the Rust morton encoders.

The pure-Python parity twins for ``fastNorm2Mort``/``VaexNorm2Mort``/
``geo2mort``/``mort2norm`` were removed (issue #37).  These tests pin the
Rust output against golden values captured from the Rust path so future
Rust changes that alter the encoding are caught.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

# Captured from the Rust geo2mort output at order 18 on the Antarctic dataset.
_ANTARCTIC_SHA256 = (
    "13e81fd525fe378aad9caff1f188ee4c25e0faddb2968389a933c4e3d6e57b0f"
)


class TestFastNorm2Mort:
    """Golden output for fastNorm2Mort."""

    def test_scalar(self):
        from mortie import tools
        assert tools.fastNorm2Mort(18, 1000, 2) == 3111111111111144331

    def test_array(self):
        from mortie import tools
        orders = np.full(1000, 18, dtype=np.int64)
        normed = np.arange(1000, dtype=np.int64)
        parents = np.array([i % 12 for i in range(1000)], dtype=np.int64)

        result = np.asarray(
            tools.fastNorm2Mort(orders, normed, parents), dtype=np.int64
        )
        assert len(result) == 1000
        # Spot-check the ends and pin the whole array with a content hash.
        np.testing.assert_array_equal(
            result[:5],
            [1111111111111111111, 2111111111111111112, 3111111111111111113,
             4111111111111111114, 5111111111111111121],
        )
        np.testing.assert_array_equal(
            result[-5:],
            [-6111111111111144314, 1111111111111144321, 2111111111111144322,
             3111111111111144323, 4111111111111144324],
        )
        assert hashlib.sha256(result.tobytes()).hexdigest() == (
            "515eb8ccc7d9bdf5f7d10090f94260fc7125bcf29383fb49e9bea616530dc4f0"
        )

    @pytest.mark.parametrize("order,expected", [
        (6, 3112321),
        (10, 31111112321),
        (14, 311111111112321),
        (18, 3111111111111112321),
    ])
    def test_different_orders(self, order, expected):
        from mortie import tools
        result = np.asarray(
            tools.fastNorm2Mort(
                order,
                np.array([100], dtype=np.int64),
                np.array([2], dtype=np.int64),
            ),
            dtype=np.int64,
        )
        np.testing.assert_array_equal(result, [expected])


class TestVaexNorm2Mort:
    """Golden output for VaexNorm2Mort (order 18)."""

    def test_basic(self):
        from mortie import tools
        normed = np.array([100, 200, 300], dtype=np.int64)
        parents = np.array([2, 3, 8], dtype=np.int64)

        result = np.asarray(tools.VaexNorm2Mort(normed, parents), dtype=np.int64)
        np.testing.assert_array_equal(
            result,
            [3111111111111112321, 4111111111111114131, -3111111111111121341],
        )


class TestGeo2Mort:
    """Golden output for geo2mort (all-in-Rust healpix crate path)."""

    # A spread of latitudes/longitudes including poles and the antimeridian.
    LATS = [0.0, 45.0, -45.0, 89.0, -89.0, 12.34, -77.7, 30.0]
    LONS = [0.0, 90.0, -90.0, 179.0, -179.0, 56.78, 123.4, -60.0]

    def test_order18(self):
        from mortie import tools
        result = np.asarray(
            tools.geo2mort(np.array(self.LATS), np.array(self.LONS), order=18),
            dtype=np.int64,
        )
        np.testing.assert_array_equal(result, [
            5411111111111111111, 2333433333333344444, -5222122222222211111,
            2444442424224133132, -5111113131331422423, 1121123434111234114,
            -4113241312243444332, 4312232323232323232,
        ])

    def test_order10(self):
        from mortie import tools
        result = np.asarray(
            tools.geo2mort(np.array(self.LATS), np.array(self.LONS), order=10),
            dtype=np.int64,
        )
        np.testing.assert_array_equal(result, [
            54111111111, 23334333333, -52221222222, 24444424242,
            -51111131313, 11211234341, -41132413122, 43122323232,
        ])

    @pytest.mark.slow
    def test_antarctic_polygon_stable(self):
        """geo2mort is stable on the 1.2M-point Antarctic dataset.

        Pins a content hash of the order-18 output so a future Rust change
        that shifts the encoding is caught even without a Python reference.
        """
        coords_file = Path("mortie/tests/Ant_Grounded_DrainageSystem_Polygons.txt")
        if not coords_file.exists():
            pytest.skip("Antarctic polygon data not found")

        from mortie import tools
        data = np.loadtxt(coords_file)
        lats, lons = data[:, 0], data[:, 1]
        result = np.asarray(tools.geo2mort(lats, lons, order=18), dtype=np.int64)
        assert hashlib.sha256(result.tobytes()).hexdigest() == _ANTARCTIC_SHA256


class TestMort2Norm:
    """Golden output for mort2norm (Rust ``rust_mort2nested`` decode path).

    The public ``mort2norm`` is capped at the order-≤18 decimal encoding
    (a 30-digit order-29 morton overflows i64); order-29 support is v1.0
    release work tracked in #48, so these fixtures stop at order 18.
    """

    def test_order1_all_parents(self):
        """Order 1 covers every parent (0-11) and both hemispheres."""
        from mortie import tools
        mortons = np.array(
            [11, 22, 33, 44, 51, 62, -13, -24, -31, -42, -53, -64],
            dtype=np.int64,
        )
        normed, parent, order = tools.mort2norm(mortons)
        assert order == 1
        np.testing.assert_array_equal(normed, [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
        np.testing.assert_array_equal(parent, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    def test_order6_spread(self):
        from mortie import tools
        mortons = np.array(
            [1241243, 2313241, 3312131, 4222443,
             5441314, 6224324, -1333123, -2224442],
            dtype=np.int64,
        )
        normed, parent, order = tools.mort2norm(mortons)
        assert order == 6
        np.testing.assert_array_equal(
            normed, [1822, 2204, 2120, 1406, 3875, 1511, 2694, 1533]
        )
        np.testing.assert_array_equal(parent, [0, 1, 2, 3, 4, 5, 6, 7])

    def test_order18_max(self):
        """Order 18 is the maximum the public decimal encoding supports."""
        from mortie import tools
        mortons = np.array(
            [1232314314324131443, 2342433312442113214, 3212443211132331242,
             4122213422422421222, 5443121134343432144, 6321122423111421134],
            dtype=np.int64,
        )
        normed, parent, order = tools.mort2norm(mortons)
        assert order == 18
        np.testing.assert_array_equal(
            normed,
            [27440083518, 49300361363, 19298032157,
             5684813077, 66642177615, 38752750859],
        )
        np.testing.assert_array_equal(parent, [0, 1, 2, 3, 4, 5])

    def test_scalar_north(self):
        from mortie import tools
        assert tools.mort2norm(3312131) == (2120, 2, 6)

    def test_scalar_south(self):
        from mortie import tools
        assert tools.mort2norm(-2224442) == (1533, 7, 6)

    def test_roundtrip_against_fastnorm2mort(self):
        """mort2norm inverts fastNorm2Mort for a random order-14 spread."""
        from mortie import tools
        rng = np.random.default_rng(14)
        normed_in = rng.integers(0, 4**14, size=32, dtype=np.int64)
        parents_in = (np.arange(32) % 12).astype(np.int64)
        orders = np.full(32, 14, dtype=np.int64)
        mortons = np.asarray(
            tools.fastNorm2Mort(orders, normed_in, parents_in), dtype=np.int64
        )
        normed, parent, order = tools.mort2norm(mortons)
        assert order == 14
        np.testing.assert_array_equal(normed, normed_in)
        np.testing.assert_array_equal(parent, parents_in)

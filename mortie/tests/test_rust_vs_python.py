"""
Golden-output regression tests for the Rust morton encoders.

The pure-Python parity twins for ``fastNorm2Mort``/``VaexNorm2Mort``/
``geo2mort`` were removed (issue #37).  These tests pin the Rust output
against golden values captured from the Rust path so future Rust changes
that alter the encoding are caught.

``mort2norm`` still carries a Python reference, so its Rust-vs-Python
parity check remains below.
"""

import hashlib
import numpy as np
import os
import pytest
from pathlib import Path

# Captured from the Rust geo2mort output at order 18 on the Antarctic dataset.
_ANTARCTIC_SHA256 = (
    "13e81fd525fe378aad9caff1f188ee4c25e0faddb2968389a933c4e3d6e57b0f"
)


@pytest.fixture(autouse=True)
def _clean_env():
    """Ensure MORTIE_FORCE_PYTHON is unset before each test."""
    os.environ.pop('MORTIE_FORCE_PYTHON', None)
    yield
    os.environ.pop('MORTIE_FORCE_PYTHON', None)


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
            tools.fastNorm2Mort(order, np.array([100]), np.array([2])),
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


def _valid_mortons(order, n=64):
    """Build a spread of valid morton indices at a given order."""
    from mortie import tools
    rng = np.random.default_rng(order)
    normed = rng.integers(0, 4**order, size=n, dtype=np.int64)
    parents = (np.arange(n) % 12).astype(np.int64)
    orders = np.full(n, order, dtype=np.int64)
    return np.asarray(tools.fastNorm2Mort(orders, normed, parents), dtype=np.int64)


def _get_rust_result(func_name, *args, **kwargs):
    """Call a mortie.tools function with Rust enabled."""
    os.environ.pop('MORTIE_FORCE_PYTHON', None)
    from mortie import tools
    import importlib
    importlib.reload(tools)
    return getattr(tools, func_name)(*args, **kwargs)


def _get_python_result(func_name, *args, **kwargs):
    """Call a mortie.tools function with pure Python."""
    os.environ['MORTIE_FORCE_PYTHON'] = '1'
    from mortie import tools
    import importlib
    importlib.reload(tools)
    return getattr(tools, func_name)(*args, **kwargs)


class TestMort2Norm:
    """Rust vs Python for mort2norm (must be bit-identical)."""

    @pytest.mark.parametrize("order", [1, 6, 10, 14, 18])
    def test_array_parity(self, order):
        mortons = _valid_mortons(order)
        r_normed, r_parent, r_order = _get_rust_result('mort2norm', mortons)
        p_normed, p_parent, p_order = _get_python_result('mort2norm', mortons)
        assert r_order == p_order == order
        np.testing.assert_array_equal(r_normed, p_normed)
        np.testing.assert_array_equal(r_parent, p_parent)

    def test_scalar_parity(self):
        morton = int(_valid_mortons(6, n=1)[0])
        rust = _get_rust_result('mort2norm', morton)
        python = _get_python_result('mort2norm', morton)
        assert rust == python

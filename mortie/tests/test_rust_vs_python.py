"""
Direct comparison test: Rust vs Pure Python implementations.

Verifies that the Rust-accelerated functions produce identical results
to the pure-Python fallback for fastNorm2Mort/VaexNorm2Mort, and
near-identical results for geo2mort (which uses a different HEALPix
library in Rust vs Python).
"""

import numpy as np
import os
import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def _clean_env():
    """Ensure MORTIE_FORCE_PYTHON is unset before each test."""
    os.environ.pop('MORTIE_FORCE_PYTHON', None)
    yield
    os.environ.pop('MORTIE_FORCE_PYTHON', None)


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


class TestFastNorm2Mort:
    """Rust vs Python for fastNorm2Mort (must be bit-identical)."""

    def test_scalar(self):
        rust = _get_rust_result('fastNorm2Mort', 18, 1000, 2)
        python = _get_python_result('fastNorm2Mort', 18, 1000, 2)
        assert rust == python

    def test_array(self):
        orders = np.full(1000, 18, dtype=np.int64)
        normed = np.arange(1000, dtype=np.int64)
        parents = np.array([i % 12 for i in range(1000)], dtype=np.int64)

        rust = _get_rust_result('fastNorm2Mort', orders, normed, parents)
        python = _get_python_result('fastNorm2Mort', orders, normed, parents)
        np.testing.assert_array_equal(rust, python)

    @pytest.mark.parametrize("order", [6, 10, 14, 18])
    def test_different_orders(self, order):
        normed = np.array([100], dtype=np.int64)
        parents = np.array([2], dtype=np.int64)

        rust = _get_rust_result('fastNorm2Mort', order, normed, parents)
        python = _get_python_result('fastNorm2Mort', order, normed, parents)
        np.testing.assert_array_equal(rust, python)


class TestVaexNorm2Mort:
    """Rust vs Python for VaexNorm2Mort (must be bit-identical)."""

    def test_basic(self):
        normed = np.array([100, 200, 300], dtype=np.int64)
        parents = np.array([2, 3, 8], dtype=np.int64)

        rust = _get_rust_result('VaexNorm2Mort', normed, parents)
        python = _get_python_result('VaexNorm2Mort', normed, parents)
        np.testing.assert_array_equal(rust, python)


class TestGeo2Mort:
    """Rust vs Python for geo2mort.

    The Rust path uses the ``healpix`` Rust crate for the HEALPix hash,
    while the Python path uses healpy (or cdshealpix). At very high
    resolution (order 18), a handful of boundary-case pixels may hash
    differently between implementations. This is expected and acceptable.
    """

    @pytest.mark.slow
    def test_antarctic_polygon(self):
        """Compare on the 1.2M-point Antarctic polygon dataset."""
        coords_file = Path("mortie/tests/Ant_Grounded_DrainageSystem_Polygons.txt")
        if not coords_file.exists():
            pytest.skip("Antarctic polygon data not found")

        data = np.loadtxt(coords_file)
        lats, lons = data[:, 0], data[:, 1]

        rust = _get_rust_result('geo2mort', lats, lons, order=18)
        python = _get_python_result('geo2mort', lats, lons, order=18)

        mismatches = np.sum(rust != python)
        total = len(rust)
        mismatch_rate = mismatches / total

        # Allow up to 0.01% boundary-case differences between
        # the healpix Rust crate and healpy
        assert mismatch_rate < 1e-4, (
            f"{mismatches:,} mismatches out of {total:,} "
            f"({mismatch_rate:.6%}) exceeds 0.01% tolerance"
        )

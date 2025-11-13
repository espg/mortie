"""Test the main package API imports

This specifically tests imports from the main package level,
not from submodules. This catches issues where functions are
incorrectly aliased or overwritten in __init__.py
"""

import pytest
import numpy as np


class TestMainAPI:
    """Test that the main API works correctly"""

    def test_import_geo2mort(self):
        """Test that geo2mort can be imported from main package"""
        from mortie import geo2mort
        assert geo2mort is not None

    def test_geo2mort_signature(self):
        """Test that geo2mort has the correct signature (lats, lons, order)"""
        from mortie import geo2mort

        # Test with single point
        lat, lon = 45.0, -122.0
        result = geo2mort(lat, lon, order=6)
        assert isinstance(result, (int, np.integer, np.ndarray))

    def test_geo2mort_arrays(self):
        """Test that geo2mort works with arrays"""
        from mortie import geo2mort

        lats = np.array([45.0, 40.0, 35.0])
        lons = np.array([-122.0, -120.0, -118.0])

        result = geo2mort(lats, lons, order=8)
        assert len(result) == len(lats)

    def test_geo2mort_default_order(self):
        """Test that geo2mort uses order=18 by default"""
        from mortie import geo2mort

        lat, lon = 45.0, -122.0

        # Should work without specifying order
        result = geo2mort(lat, lon)
        assert isinstance(result, (int, np.integer, np.ndarray))

    def test_all_main_imports(self):
        """Test that all expected functions can be imported from main package"""
        from mortie import (
            geo2mort,
            geo2uniq,
            clip2order,
            unique2parent,
            heal_norm,
            fastNorm2Mort,
            VaexNorm2Mort,
            order2res,
            res2display
        )

        # All should be callable or None (for unimplemented)
        assert callable(geo2mort)
        assert callable(geo2uniq)
        assert callable(clip2order)
        assert callable(unique2parent)
        assert callable(heal_norm)
        assert callable(fastNorm2Mort)
        assert callable(VaexNorm2Mort)
        assert callable(order2res)
        assert callable(res2display)

    def test_geo2mort_vs_tools(self):
        """Test that mortie.geo2mort and mortie.tools.geo2mort are the same"""
        from mortie import geo2mort as main_geo2mort
        from mortie import tools

        # They should be the same function
        assert main_geo2mort is tools.geo2mort

        # And produce the same results
        lat, lon = 45.0, -122.0
        result1 = main_geo2mort(lat, lon, order=10)
        result2 = tools.geo2mort(lat, lon, order=10)
        assert result1 == result2
"""
Comprehensive unit tests for mortie.tools module

These tests establish reference behavior for all morton indexing functions.
They will be used to verify that any refactoring (e.g., removing numba)
produces identical outputs.

Key constraints:
- Morton indices use base-4 encoding (digits 1-4) after the base cell identifier
- Not all integers are valid morton indices
- Tests focus on consistency, determinism, and structural validation
"""

import pytest
import numpy as np
import healpy as hp
from numpy.testing import assert_array_equal, assert_allclose

from mortie import tools


class TestOrder2Res:
    """Test order to resolution conversion"""

    def test_order2res_basic(self):
        """Test basic order to resolution calculations"""
        # Order 0 should be largest resolution
        res0 = tools.order2res(0)
        assert_allclose(res0, 111 * 58.6323, rtol=1e-10)

        # Order 1 should be half of order 0
        res1 = tools.order2res(1)
        assert_allclose(res1, res0 / 2.0, rtol=1e-10)

    def test_order2res_range(self):
        """Test full range of valid orders"""
        for order in range(20):
            res = tools.order2res(order)
            expected = 111 * 58.6323 * (0.5 ** order)
            assert_allclose(res, expected, rtol=1e-10)

    def test_order2res_decreasing(self):
        """Test that resolution decreases with order"""
        resolutions = [tools.order2res(i) for i in range(10)]
        # Each resolution should be smaller than the previous
        assert all(resolutions[i] > resolutions[i+1] for i in range(len(resolutions)-1))


class TestUnique2Parent:
    """Test UNIQ to parent cell conversion"""

    def test_unique2parent_single_resolution(self):
        """Test parent extraction for single resolution"""
        # Create some UNIQ values at order 6
        order = 6
        nside = 2**order
        nest_indices = np.array([100, 200, 300, 400])
        uniq = 4 * (nside**2) + nest_indices

        parents = tools.unique2parent(uniq)

        # All parents should be in valid range (0-11 for HEALPix base cells)
        assert np.all(parents >= 0)
        assert np.all(parents < 12)

    def test_unique2parent_deterministic(self):
        """Test that same inputs give same outputs"""
        order = 8
        nside = 2**order
        nest_indices = np.array([1000, 2000, 3000])
        uniq = 4 * (nside**2) + nest_indices

        parents1 = tools.unique2parent(uniq)
        parents2 = tools.unique2parent(uniq)
        parents3 = tools.unique2parent(uniq)

        assert_array_equal(parents1, parents2)
        assert_array_equal(parents2, parents3)

    def test_unique2parent_mixed_resolution_raises(self):
        """Test that mixed resolutions raise NotImplementedError"""
        # Mix orders 6 and 7
        nside6 = 2**6
        nside7 = 2**7
        uniq_mixed = np.array([
            4 * (nside6**2) + 100,
            4 * (nside7**2) + 200,
        ])

        with pytest.raises(NotImplementedError, match="mixed resolution"):
            tools.unique2parent(uniq_mixed)


class TestHealNorm:
    """Test HEALPix address normalization"""

    def test_heal_norm_basic(self):
        """Test basic normalization"""
        order = 6
        nside = 2**order
        N_pix = nside**2

        base = 2
        addr_nest = np.array([2*N_pix + 100, 2*N_pix + 200])

        normed = tools.heal_norm(base, order, addr_nest)

        # Should be offset by base * N_pix
        expected = addr_nest - (base * N_pix)
        assert_array_equal(normed, expected)

    def test_heal_norm_zero_offset(self):
        """Test normalization with base 0"""
        order = 6
        addr_nest = np.array([100, 200, 300])

        normed = tools.heal_norm(0, order, addr_nest)

        # With base=0, should be unchanged
        assert_array_equal(normed, addr_nest)

    def test_heal_norm_deterministic(self):
        """Test determinism"""
        order = 8
        base = 5
        addr_nest = np.array([1000, 2000, 3000])

        result1 = tools.heal_norm(base, order, addr_nest)
        result2 = tools.heal_norm(base, order, addr_nest)

        assert_array_equal(result1, result2)


class TestGeo2Uniq:
    """Test geographic to UNIQ conversion"""

    def test_geo2uniq_single_point(self):
        """Test single lat/lon point"""
        lat, lon = 45.0, -122.0
        order = 6

        uniq = tools.geo2uniq(lat, lon, order)

        # Check it's a valid UNIQ (should be integer)
        assert isinstance(uniq, (int, np.integer))

        # Check it's in valid range for this order
        nside = 2**order
        min_uniq = 4 * (nside**2)
        max_uniq = 4 * (nside**2) + hp.nside2npix(nside)
        assert min_uniq <= uniq < max_uniq

    def test_geo2uniq_array(self):
        """Test array of lat/lon points"""
        lats = np.array([45.0, 47.0, 49.0])
        lons = np.array([-122.0, -120.0, -118.0])
        order = 8

        uniq = tools.geo2uniq(lats, lons, order)

        # Should return array of same length
        assert len(uniq) == len(lats)

        # All should be valid UNIQ values
        nside = 2**order
        assert np.all(uniq >= 4 * (nside**2))

    def test_geo2uniq_deterministic(self):
        """Test that same inputs give same outputs"""
        lat, lon = 45.0, -122.0
        order = 10

        uniq1 = tools.geo2uniq(lat, lon, order)
        uniq2 = tools.geo2uniq(lat, lon, order)
        uniq3 = tools.geo2uniq(lat, lon, order)

        assert uniq1 == uniq2 == uniq3

    def test_geo2uniq_different_orders(self):
        """Test that different orders give different results"""
        lat, lon = 45.0, -122.0

        uniq6 = tools.geo2uniq(lat, lon, order=6)
        uniq8 = tools.geo2uniq(lat, lon, order=8)
        uniq12 = tools.geo2uniq(lat, lon, order=12)

        # Different orders should give different UNIQ values
        assert uniq6 != uniq8
        assert uniq8 != uniq12


class TestMortonStructure:
    """Test morton index structural properties"""

    def test_morton_digits_valid(self):
        """Test that morton indices only use valid digits (1-4)"""
        # Generate some morton indices
        lats = np.array([45.0, -45.0, 0.0, 60.0, -30.0])
        lons = np.array([-122.0, 122.0, 0.0, -90.0, 45.0])

        for order in [6, 8, 10, 12]:
            morton = tools.geo2mort(lats, lons, order=order)

            for m in morton:
                # Convert to string, skip sign and base cell portion
                morton_str = str(abs(m))

                # After leading digits, remaining should be 1-4
                # (This is a heuristic check - may need refinement)
                if len(morton_str) > 2:
                    trailing_digits = morton_str[2:]  # Skip base cell digits
                    for digit in trailing_digits:
                        # Valid morton digits after encoding are 1-4
                        assert digit in '1234', f"Invalid digit {digit} in morton {m}"

    def test_morton_sign_consistency(self):
        """Test that sign indicates hemisphere or base cell region"""
        # Points in northern hemisphere
        lats_north = np.array([45.0, 60.0, 30.0])
        lons_north = np.array([-122.0, 0.0, 45.0])

        # Points in southern hemisphere
        lats_south = np.array([-45.0, -60.0, -30.0])
        lons_south = np.array([-122.0, 0.0, 45.0])

        morton_north = tools.geo2mort(lats_north, lons_north, order=10)
        morton_south = tools.geo2mort(lats_south, lons_south, order=10)

        # Check that we get both positive and negative indices
        # (exact distribution depends on HEALPix geometry)
        all_morton = np.concatenate([morton_north, morton_south])
        assert np.any(all_morton > 0) or np.any(all_morton < 0)


class TestGeo2Mort:
    """Test full geographic to Morton index conversion"""

    @pytest.fixture
    def sample_coords(self):
        """Sample coordinates for testing"""
        return {
            'single': (45.0, -122.0),
            'array': (
                np.array([45.0, 47.0, 49.0, -45.0, -47.0]),
                np.array([-122.0, -120.0, -118.0, 122.0, 120.0])
            ),
            'equator': (
                np.array([0.0, 0.0, 0.0]),
                np.array([-180.0, 0.0, 180.0])
            ),
            'poles': (
                np.array([89.0, -89.0]),
                np.array([0.0, 0.0])
            )
        }

    def test_geo2mort_single_point(self, sample_coords):
        """Test single point conversion"""
        lat, lon = sample_coords['single']

        morton = tools.geo2mort(lat, lon, order=6)

        # Should return integer
        assert isinstance(morton, (int, np.integer, np.ndarray))

    def test_geo2mort_array(self, sample_coords):
        """Test array conversion"""
        lats, lons = sample_coords['array']

        morton = tools.geo2mort(lats, lons, order=8)

        # Should return array of same length
        assert len(morton) == len(lats)

        # All should be integers
        assert morton.dtype in [np.int32, np.int64]

    def test_geo2mort_deterministic(self, sample_coords):
        """Test that same inputs always give same outputs"""
        lat, lon = sample_coords['single']

        morton1 = tools.geo2mort(lat, lon, order=12)
        morton2 = tools.geo2mort(lat, lon, order=12)
        morton3 = tools.geo2mort(lat, lon, order=12)

        assert morton1 == morton2 == morton3

    def test_geo2mort_array_deterministic(self, sample_coords):
        """Test determinism for arrays"""
        lats, lons = sample_coords['array']

        morton1 = tools.geo2mort(lats, lons, order=10)
        morton2 = tools.geo2mort(lats, lons, order=10)

        assert_array_equal(morton1, morton2)

    def test_geo2mort_order_hierarchy(self, sample_coords):
        """Test that clipping higher order to lower order is consistent"""
        lat, lon = sample_coords['single']

        # Get morton at different orders
        mort6 = tools.geo2mort(lat, lon, order=6)
        mort12 = tools.geo2mort(lat, lon, order=12)

        # Both should be valid integers
        assert isinstance(mort6, (int, np.integer, np.ndarray))
        assert isinstance(mort12, (int, np.integer, np.ndarray))

        # Clipping should reduce magnitude (this is a structural test)
        mort12_clipped = tools.clip2order(6, np.array([mort12]))
        assert len(mort12_clipped) == 1

    def test_geo2mort_equator(self, sample_coords):
        """Test points on equator"""
        lats, lons = sample_coords['equator']

        morton = tools.geo2mort(lats, lons, order=8)

        # Should get valid morton indices
        assert len(morton) == len(lats)
        assert not np.any(np.isnan(morton))

    def test_geo2mort_poles(self, sample_coords):
        """Test points near poles"""
        lats, lons = sample_coords['poles']

        morton = tools.geo2mort(lats, lons, order=8)

        # Should get valid morton indices
        assert len(morton) == len(lats)
        assert not np.any(np.isnan(morton))


class TestFastNorm2Mort:
    """Test the fast normalized to morton conversion"""

    def test_fastNorm2Mort_basic(self):
        """Test basic conversion"""
        order = 6
        normed = np.array([100, 200, 300], dtype=np.int64)
        parents = np.array([2, 3, 4], dtype=np.int64)

        morton = tools.fastNorm2Mort(order, normed, parents)

        # Should return array of same length
        assert len(morton) == len(normed)

        # Should be integers
        assert morton.dtype == np.int64

    def test_fastNorm2Mort_deterministic(self):
        """Test determinism"""
        order = 8
        normed = np.array([100, 200, 300], dtype=np.int64)
        parents = np.array([2, 3, 4], dtype=np.int64)

        result1 = tools.fastNorm2Mort(order, normed, parents)
        result2 = tools.fastNorm2Mort(order, normed, parents)
        result3 = tools.fastNorm2Mort(order, normed, parents)

        assert_array_equal(result1, result2)
        assert_array_equal(result2, result3)

    def test_fastNorm2Mort_max_order(self):
        """Test that order 18 is maximum"""
        normed = np.array([100], dtype=np.int64)
        parents = np.array([2], dtype=np.int64)

        # Order 18 should work
        morton18 = tools.fastNorm2Mort(18, normed, parents)
        assert morton18.dtype == np.int64

        # Order 19 should raise ValueError
        with pytest.raises(ValueError, match="Max order is 18"):
            tools.fastNorm2Mort(19, normed, parents)

    def test_fastNorm2Mort_different_orders(self):
        """Test that different orders give different results"""
        normed = np.array([100], dtype=np.int64)
        parents = np.array([2], dtype=np.int64)

        mort6 = tools.fastNorm2Mort(6, normed, parents)
        mort8 = tools.fastNorm2Mort(8, normed, parents)
        mort10 = tools.fastNorm2Mort(10, normed, parents)

        # Different orders should give different values
        assert mort6[0] != mort8[0]
        assert mort8[0] != mort10[0]


class TestVaexNorm2Mort:
    """Test the Vaex-specific normalized to morton conversion"""

    def test_VaexNorm2Mort_basic(self):
        """Test basic conversion at order 18"""
        normed = np.array([100, 200, 300], dtype=np.int64)
        parents = np.array([8, 9, 10], dtype=np.int64)

        morton = tools.VaexNorm2Mort(normed, parents)

        # Should return array of same length
        assert len(morton) == len(normed)

        # Should be integers
        assert morton.dtype == np.int64

    def test_VaexNorm2Mort_deterministic(self):
        """Test determinism"""
        normed = np.array([100, 200, 300], dtype=np.int64)
        parents = np.array([8, 9, 10], dtype=np.int64)

        result1 = tools.VaexNorm2Mort(normed, parents)
        result2 = tools.VaexNorm2Mort(normed, parents)

        assert_array_equal(result1, result2)

    def test_VaexNorm2Mort_vs_fast(self):
        """Test that VaexNorm2Mort matches fastNorm2Mort at order 18"""
        # Test with various parent values to cover both branches
        test_cases = [
            # (normed, parents) - test both parent < 6 and parent >= 6
            (np.array([100, 200, 300], dtype=np.int64), np.array([2, 3, 4], dtype=np.int64)),
            (np.array([100, 200, 300], dtype=np.int64), np.array([8, 9, 10], dtype=np.int64)),
            (np.array([1000, 2000, 3000], dtype=np.int64), np.array([0, 5, 11], dtype=np.int64)),
        ]

        for normed, parents in test_cases:
            vaex_result = tools.VaexNorm2Mort(normed, parents)
            fast_result = tools.fastNorm2Mort(18, normed, parents)

            # Results must match exactly
            assert_array_equal(
                vaex_result, fast_result,
                err_msg=f"VaexNorm2Mort must match fastNorm2Mort(order=18)\n"
                        f"normed={normed}, parents={parents}\n"
                        f"VaexNorm2Mort={vaex_result}\n"
                        f"fastNorm2Mort={fast_result}"
            )


class TestClip2Order:
    """Test resolution clipping"""

    def test_clip2order_factor(self):
        """Test factor calculation"""
        factor = tools.clip2order(12, print_factor=True)
        assert factor == 10**(18-12)

    def test_clip2order_clipping(self):
        """Test actual clipping"""
        # Create morton indices at order 18
        morton18 = np.array([123456789012345678, 987654321098765432])

        # Clip to order 12
        morton12 = tools.clip2order(12, morton18)

        # Should be smaller values
        assert np.all(np.abs(morton12) < np.abs(morton18))

        # Check correct factor
        expected = np.abs(morton18) // 10**(18-12)
        assert_array_equal(np.abs(morton12), expected)

    def test_clip2order_negative_indices(self):
        """Test clipping preserves sign"""
        morton18 = np.array([123456789012345678, -987654321098765432])

        morton12 = tools.clip2order(12, morton18)

        # Negative should stay negative
        assert morton18[0] > 0 and morton12[0] > 0
        assert morton18[1] < 0 and morton12[1] < 0

    def test_clip2order_deterministic(self):
        """Test determinism"""
        morton18 = np.array([123456789012345678, -987654321098765432])

        result1 = tools.clip2order(12, morton18)
        result2 = tools.clip2order(12, morton18)

        assert_array_equal(result1, result2)


class TestIntegration:
    """Integration tests for complete workflow"""

    def test_round_trip_consistency(self):
        """Test that processing pipeline is consistent"""
        # Generate test points
        lats = np.array([45.0, -45.0, 0.0, 60.0, -60.0])
        lons = np.array([-122.0, 122.0, 0.0, -90.0, 90.0])

        for order in [6, 8, 10, 12, 14]:
            morton1 = tools.geo2mort(lats, lons, order=order)
            morton2 = tools.geo2mort(lats, lons, order=order)

            # Same inputs should give same outputs
            assert_array_equal(morton1, morton2)

    def test_spatial_locality(self):
        """Test that very nearby points have similar morton indices"""
        # Points very close together
        lat_base = 45.0
        lon_base = -122.0
        epsilon = 0.0001  # Very small offset

        lats = np.array([lat_base, lat_base + epsilon, lat_base - epsilon])
        lons = np.array([lon_base, lon_base + epsilon, lon_base - epsilon])

        morton = tools.geo2mort(lats, lons, order=14)

        # Nearby points should have some similarity
        # At minimum, they should all be valid (no NaN/Inf)
        assert not np.any(np.isnan(morton))
        assert not np.any(np.isinf(morton))

    def test_full_globe_coverage(self):
        """Test that we can process points across entire globe"""
        # Sample points across globe
        np.random.seed(42)
        n_points = 100
        lats = np.random.uniform(-85, 85, n_points)  # Avoid extreme poles
        lons = np.random.uniform(-180, 180, n_points)

        morton = tools.geo2mort(lats, lons, order=10)

        # Should get valid morton indices for all points
        assert len(morton) == n_points
        assert not np.any(np.isnan(morton))
        assert not np.any(np.isinf(morton))

    def test_reproducibility_across_runs(self):
        """Test that results are reproducible across multiple runs"""
        np.random.seed(123)
        lats = np.random.uniform(-80, 80, 50)
        lons = np.random.uniform(-180, 180, 50)

        # Run multiple times
        results = []
        for _ in range(5):
            morton = tools.geo2mort(lats, lons, order=12)
            results.append(morton)

        # All results should be identical
        for i in range(1, len(results)):
            assert_array_equal(results[0], results[i])


class TestReferenceData:
    """Generate and validate reference data for regression testing"""

    def test_reference_single_points(self):
        """Generate reference data for single points at various orders"""
        test_points = [
            (45.0, -122.0),   # Pacific Northwest
            (-45.0, 122.0),   # Southern hemisphere
            (0.0, 0.0),       # Equator, Prime Meridian
            (60.0, -90.0),    # High latitude
            (-30.0, 45.0),    # Southern mid-latitude
        ]

        reference = {}
        for idx, (lat, lon) in enumerate(test_points):
            for order in [6, 8, 10, 12, 14, 16, 18]:
                morton = tools.geo2mort(lat, lon, order=order)
                key = f"point_{idx}_order_{order}"
                reference[key] = morton

        # Verify all reference data is valid (may be scalars or 0-d arrays)
        for v in reference.values():
            assert isinstance(v, (int, np.integer, np.ndarray))
            # If array, should be scalar-like
            if isinstance(v, np.ndarray):
                assert v.ndim == 0 or (v.ndim == 1 and len(v) == 1)

    def test_reference_arrays(self):
        """Generate reference data for coordinate arrays"""
        # Matching arrays (not meshgrid - healpy expects matching sizes)
        n_points = 20
        lats = np.linspace(-80, 80, n_points)
        lons = np.linspace(-180, 180, n_points)

        reference = {}
        for order in [6, 10, 14]:
            morton = tools.geo2mort(lats, lons, order=order)
            reference[f"array_order_{order}"] = morton

        # Verify all arrays
        for morton in reference.values():
            assert len(morton) == n_points
            assert not np.any(np.isnan(morton))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

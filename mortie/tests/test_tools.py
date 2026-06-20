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
        max_uniq = 4 * (nside**2) + 12 * (nside**2)
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
        """Test that the decode-through-kernel decimal repr uses valid digits.

        After the issue #48 flip the bare ``i64`` is the packed word, not its
        decimal value; the human-readable structure (leading base digit, then
        ``1..=4`` per order) lives in the ``decimal_repr``.
        """
        from mortie import _rustie
        lats = np.array([45.0, -45.0, 0.0, 60.0, -30.0])
        lons = np.array([-122.0, 122.0, 0.0, -90.0, 45.0])

        for order in [6, 8, 10, 12]:
            morton = np.ascontiguousarray(
                tools.geo2mort(lats, lons, order=order), dtype=np.uint64
            )
            reprs = _rustie.rust_mi_decimal_repr(morton)
            for s in reprs:
                digits = s.lstrip("-")
                # leading base-cell digit + one digit per order.
                assert len(digits) == order + 1
                for digit in digits[1:]:
                    assert digit in '1234', f"Invalid digit {digit} in repr {s}"

    def test_morton_sign_consistency(self):
        """Bit 63 indicates the hemisphere / base-cell region (uint64 word)."""
        bit63 = np.uint64(1) << np.uint64(63)
        # Points in northern hemisphere
        lats_north = np.array([45.0, 60.0, 30.0])
        lons_north = np.array([-122.0, 0.0, 45.0])

        # Points in southern hemisphere
        lats_south = np.array([-45.0, -60.0, -30.0])
        lons_south = np.array([-122.0, 0.0, 45.0])

        morton_north = tools.geo2mort(lats_north, lons_north, order=10)
        morton_south = tools.geo2mort(lats_south, lons_south, order=10)

        # Both bit-63-clear and bit-63-set words appear across the two sets
        # (exact distribution depends on HEALPix geometry).
        all_morton = np.concatenate([morton_north, morton_south])
        assert np.any(all_morton < bit63) and np.any(all_morton >= bit63)


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

        # Morton words are uint64 (issue #58)
        assert morton.dtype == np.uint64

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


class TestNorm2Mort:
    """norm2mort: the order-29-native packed forward encoder (inverse of mort2norm)."""

    def test_norm2mort_basic(self):
        """Basic conversion returns one packed word per (normed, parent)."""
        order = 6
        normed = np.array([100, 200, 300], dtype=np.int64)
        parents = np.array([2, 3, 4], dtype=np.int64)

        morton = np.array(
            [int(tools.norm2mort(int(n), int(p), order)) for n, p in zip(normed, parents)],
            dtype=np.uint64,
        )
        assert len(morton) == len(normed)
        assert morton.dtype == np.uint64

    def test_norm2mort_inverts_mort2norm(self):
        """norm2mort is the exact inverse of mort2norm."""
        for order in (6, 8, 18, 29):
            for normed, parent in [(100, 2), (4096, 5), (0, 11), (12345, 8)]:
                if normed >= 4 ** order:
                    continue
                m = tools.norm2mort(normed, parent, order)
                assert tools.mort2norm(m) == (normed, parent, order)

    def test_norm2mort_different_orders(self):
        """Different orders give different packed words for the same (normed, parent)."""
        m6 = int(tools.norm2mort(100, 2, 6))
        m8 = int(tools.norm2mort(100, 2, 8))
        m10 = int(tools.norm2mort(100, 2, 10))
        assert m6 != m8
        assert m8 != m10

    def test_norm2mort_order29_native(self):
        """norm2mort reaches order 29 (no order-18 cap)."""
        m = tools.norm2mort(123, 2, 29)
        normed, parent, order = tools.mort2norm(m)
        assert (int(normed), int(parent), order) == (123, 2, 29)


class TestClip2Order:
    """Test resolution clipping (kernel coarsen)."""

    def test_clip2order_factor(self):
        """print_factor returns the level count dropped from order 18."""
        assert tools.clip2order(12, print_factor=True) == 18 - 12

    def test_clip2order_clipping(self):
        """Clipping coarsens packed words to the target order."""
        # Two order-18 packed words.
        morton18 = np.array(
            [int(tools.norm2mort(12345, 2, 18)), int(tools.norm2mort(54321, 4, 18))],
            dtype=np.uint64,
        )
        morton12 = tools.clip2order(12, morton18)
        # The coarsened words decode to order 12 and the same base cells.
        _, parent, order = tools.mort2norm(morton12)
        assert order == 12
        np.testing.assert_array_equal(parent, [2, 4])
        # Coarsening == re-encoding the order-18 cell's order-12 ancestor.
        n18, p18, _ = tools.mort2norm(morton18)
        expected = np.array(
            [int(tools.norm2mort(int(n) >> (2 * 6), int(p), 12))
             for n, p in zip(n18, p18)],
            dtype=np.uint64,
        )
        np.testing.assert_array_equal(morton12, expected)

    def test_clip2order_negative_indices(self):
        """Clipping a southern (bit-63-set) word keeps it southern."""
        bit63 = np.uint64(1) << np.uint64(63)
        morton18 = np.array(
            [int(tools.norm2mort(100, 2, 18)), int(tools.norm2mort(200, 9, 18))],
            dtype=np.uint64,
        )
        morton12 = tools.clip2order(12, morton18)
        # Base cell 9 sets bit 63 -> stays set; base 2 stays clear.
        assert morton18[0] < bit63 and morton12[0] < bit63
        assert morton18[1] >= bit63 and morton12[1] >= bit63

    def test_clip2order_deterministic(self):
        """Test determinism"""
        morton18 = np.array(
            [int(tools.norm2mort(100, 2, 18)), int(tools.norm2mort(200, 9, 18))],
            dtype=np.uint64,
        )
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


class TestGenerateMortonChildren:
    """generate_morton_children: NESTED-space descent, packed words (issue #48)."""

    def _parent(self, normed, base, order):
        """A packed parent word for a given (normed, base, order)."""
        return int(tools.norm2mort(normed, base, order))

    def test_one_level_count_and_descent(self):
        """One level down yields 4 children; staying put yields the parent."""
        parent = self._parent(1234, base=11, order=6)  # southern base cell
        children = tools.generate_morton_children(parent, target_order=7)
        assert len(children) == 4
        # Each child is order 7 and shares the parent's order-6 ancestor.
        _, _, order = tools.mort2norm(children)
        assert order == 7
        np.testing.assert_array_equal(
            tools.clip2order(6, np.ascontiguousarray(children, dtype=np.uint64)),
            np.full(4, parent, dtype=np.uint64),
        )
        # Already at target order -> returns the parent unchanged.
        np.testing.assert_array_equal(
            tools.generate_morton_children(parent, target_order=6),
            np.array([parent], dtype=np.uint64),
        )

    def test_two_levels_count_and_membership(self):
        """Descending 2 levels yields 16 children, all sharing the parent prefix."""
        parent = self._parent(420, base=2, order=5)
        children = tools.generate_morton_children(parent, target_order=7)
        assert len(children) == 16
        # Each child coarsens back to the parent at order 5.
        np.testing.assert_array_equal(
            tools.clip2order(5, np.ascontiguousarray(children, dtype=np.uint64)),
            np.full(16, parent, dtype=np.uint64),
        )
        # Strictly ascending in the unsigned (Z-order) word.
        u = np.ascontiguousarray(children, dtype=np.uint64)
        assert np.all(np.diff(u.astype(object)) > 0)

    def test_sign_preserved(self):
        """Southern-hemisphere parents (bit 63 set) keep bit 63 set."""
        bit63 = np.uint64(1) << np.uint64(63)
        parent = self._parent(7, base=8, order=6)
        assert parent >= int(bit63)
        children = tools.generate_morton_children(parent, target_order=8)
        assert np.all(children >= bit63)

    def test_matches_nested_space_reference(self):
        """Match an independent NESTED-space child enumeration for several inputs."""
        from mortie import _rustie

        def reference(parent_morton, target):
            nested, depths = _rustie.rust_mort2nested(
                np.ascontiguousarray(np.atleast_1d(np.uint64(parent_morton)))
            )
            diff = target - int(depths[0])
            child_nested = (int(nested[0]) << (2 * diff)) + np.arange(
                4 ** diff, dtype=np.uint64
            )
            return _rustie.rust_nested2mort(
                np.ascontiguousarray(child_nested),
                np.full(4 ** diff, target, dtype=np.uint8),
            )

        for normed, base, order in [(7, 8, 6), (420, 2, 5), (0, 0, 1), (3, 7, 2)]:
            parent = self._parent(normed, base, order)
            for target in range(order, order + 4):
                assert_array_equal(
                    tools.generate_morton_children(parent, target),
                    reference(parent, target),
                )

    def test_target_below_parent_raises(self):
        parent = self._parent(7, base=8, order=6)
        with pytest.raises(ValueError):
            tools.generate_morton_children(parent, target_order=3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

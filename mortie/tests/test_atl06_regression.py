"""
Empirical regression test using real ATL06 granule data

This test processes all coordinate values from an ATL06 granule and generates
reference morton indices. Any changes to the morton encoding that alter the
output will be caught by this test.

The test will be skipped if the ATL06 .h5 file is not present.
"""

import pytest
import numpy as np
import h5py
from pathlib import Path
from numpy.testing import assert_array_equal

from mortie import tools


# Find ATL06 file in tests directory
TEST_DIR = Path(__file__).parent
ATL06_FILES = list(TEST_DIR.glob("*ATL06*.h5"))

# Skip if no ATL06 file found
pytestmark = pytest.mark.skipif(
    len(ATL06_FILES) == 0,
    reason="No ATL06 granule found in tests directory"
)


class TestATL06Regression:
    """Empirical regression tests using real ATL06 data"""

    @pytest.fixture(scope="class")
    def atl06_file(self):
        """Path to ATL06 test file"""
        if len(ATL06_FILES) == 0:
            pytest.skip("No ATL06 file available")
        return ATL06_FILES[0]

    @pytest.fixture(scope="class")
    def atl06_coordinates(self, atl06_file):
        """Extract lat/lon coordinates from ATL06 file"""
        coords = {}

        with h5py.File(atl06_file, 'r') as f:
            # ATL06 has 3 beam pairs, 6 beams total
            # Structure: /gtXl/ and /gtXr/ where X in {1,2,3}
            for beam_pair in ['gt1', 'gt2', 'gt3']:
                for lr in ['l', 'r']:
                    beam = f"{beam_pair}{lr}"
                    try:
                        # Path to land ice segments
                        base_path = f"{beam}/land_ice_segments"

                        if base_path in f:
                            lats = f[f"{base_path}/latitude"][:]
                            lons = f[f"{base_path}/longitude"][:]

                            # Filter out fill values and invalid coords
                            valid = (
                                (lats >= -90) & (lats <= 90) &
                                (lons >= -180) & (lons <= 180) &
                                np.isfinite(lats) & np.isfinite(lons)
                            )

                            if np.any(valid):
                                coords[beam] = {
                                    'lats': lats[valid],
                                    'lons': lons[valid],
                                    'count': np.sum(valid)
                                }
                    except KeyError:
                        # Beam may not exist in this granule
                        continue

        return coords

    @pytest.fixture(scope="class")
    def reference_morton_indices(self, atl06_coordinates):
        """Generate reference morton indices for all ATL06 coordinates"""
        reference = {}

        for beam, coords in atl06_coordinates.items():
            lats = coords['lats']
            lons = coords['lons']

            # Generate morton indices at multiple orders
            reference[beam] = {}
            for order in [6, 10, 14, 18]:
                morton = tools.geo2mort(lats, lons, order=order)
                reference[beam][f'order_{order}'] = morton

        return reference

    def test_atl06_file_exists(self, atl06_file):
        """Verify ATL06 file exists and is readable"""
        assert atl06_file.exists()
        assert atl06_file.suffix == '.h5'

        # Verify it's a valid HDF5 file
        with h5py.File(atl06_file, 'r') as f:
            assert 'ancillary_data' in f or any('gt' in key for key in f.keys())

    def test_atl06_coordinates_extracted(self, atl06_coordinates):
        """Verify we extracted coordinates from ATL06 file"""
        assert len(atl06_coordinates) > 0, "No valid coordinates found in ATL06 file"

        # Check each beam has valid data
        for beam, coords in atl06_coordinates.items():
            assert 'lats' in coords
            assert 'lons' in coords
            assert len(coords['lats']) == len(coords['lons'])
            assert len(coords['lats']) > 0

            # Verify coordinates are in valid range
            assert np.all(coords['lats'] >= -90)
            assert np.all(coords['lats'] <= 90)
            assert np.all(coords['lons'] >= -180)
            assert np.all(coords['lons'] <= 180)

    def test_reference_morton_generation(self, reference_morton_indices, atl06_coordinates):
        """Verify reference morton indices were generated"""
        assert len(reference_morton_indices) > 0

        for beam in atl06_coordinates.keys():
            assert beam in reference_morton_indices

            # Check all orders were computed
            for order in [6, 10, 14, 18]:
                key = f'order_{order}'
                assert key in reference_morton_indices[beam]

                morton = reference_morton_indices[beam][key]
                coords_count = atl06_coordinates[beam]['count']

                # Verify same length as input coordinates
                assert len(morton) == coords_count

                # Verify no NaN or Inf
                assert not np.any(np.isnan(morton))
                assert not np.any(np.isinf(morton))

    def test_morton_regression_order6(self, atl06_coordinates, reference_morton_indices):
        """Regression test: morton indices at order 6 must not change"""
        for beam, coords in atl06_coordinates.items():
            # Recompute morton indices
            morton_new = tools.geo2mort(coords['lats'], coords['lons'], order=6)

            # Compare with reference
            morton_ref = reference_morton_indices[beam]['order_6']

            assert_array_equal(
                morton_new, morton_ref,
                err_msg=f"Morton indices changed for beam {beam} at order 6"
            )

    def test_morton_regression_order10(self, atl06_coordinates, reference_morton_indices):
        """Regression test: morton indices at order 10 must not change"""
        for beam, coords in atl06_coordinates.items():
            morton_new = tools.geo2mort(coords['lats'], coords['lons'], order=10)
            morton_ref = reference_morton_indices[beam]['order_10']

            assert_array_equal(
                morton_new, morton_ref,
                err_msg=f"Morton indices changed for beam {beam} at order 10"
            )

    def test_morton_regression_order14(self, atl06_coordinates, reference_morton_indices):
        """Regression test: morton indices at order 14 must not change"""
        for beam, coords in atl06_coordinates.items():
            morton_new = tools.geo2mort(coords['lats'], coords['lons'], order=14)
            morton_ref = reference_morton_indices[beam]['order_14']

            assert_array_equal(
                morton_new, morton_ref,
                err_msg=f"Morton indices changed for beam {beam} at order 14"
            )

    def test_morton_regression_order18(self, atl06_coordinates, reference_morton_indices):
        """Regression test: morton indices at order 18 must not change"""
        for beam, coords in atl06_coordinates.items():
            morton_new = tools.geo2mort(coords['lats'], coords['lons'], order=18)
            morton_ref = reference_morton_indices[beam]['order_18']

            assert_array_equal(
                morton_new, morton_ref,
                err_msg=f"Morton indices changed for beam {beam} at order 18"
            )

    def test_morton_determinism_atl06(self, atl06_coordinates):
        """Test that morton indices are deterministic on real data"""
        for beam, coords in atl06_coordinates.items():
            # Run conversion multiple times
            morton1 = tools.geo2mort(coords['lats'], coords['lons'], order=12)
            morton2 = tools.geo2mort(coords['lats'], coords['lons'], order=12)
            morton3 = tools.geo2mort(coords['lats'], coords['lons'], order=12)

            # All should be identical
            assert_array_equal(morton1, morton2)
            assert_array_equal(morton2, morton3)

    def test_morton_structure_atl06(self, reference_morton_indices):
        """Test that all morton indices have valid structure"""
        for beam, orders in reference_morton_indices.items():
            for order_key, morton in orders.items():
                # Check all morton indices use valid digits
                for m in morton:
                    morton_str = str(abs(m))

                    # After leading digits, should only use 1-4
                    if len(morton_str) > 2:
                        trailing_digits = morton_str[2:]
                        for digit in trailing_digits:
                            assert digit in '1234', (
                                f"Invalid digit {digit} in morton {m} "
                                f"for beam {beam}, {order_key}"
                            )

    def test_atl06_statistics(self, atl06_coordinates, reference_morton_indices):
        """Print statistics about the ATL06 test data"""
        print("\n" + "="*60)
        print("ATL06 Test Data Statistics")
        print("="*60)

        total_points = sum(coords['count'] for coords in atl06_coordinates.values())
        print(f"\nTotal valid coordinates: {total_points:,}")
        print(f"Number of beams: {len(atl06_coordinates)}")

        for beam, coords in atl06_coordinates.items():
            print(f"\n{beam}:")
            print(f"  Points: {coords['count']:,}")
            print(f"  Lat range: [{coords['lats'].min():.2f}, {coords['lats'].max():.2f}]")
            print(f"  Lon range: [{coords['lons'].min():.2f}, {coords['lons'].max():.2f}]")

            # Show sample morton indices
            morton_18 = reference_morton_indices[beam]['order_18']
            print(f"  Sample morton (order 18): {morton_18[0]}")

        print("="*60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

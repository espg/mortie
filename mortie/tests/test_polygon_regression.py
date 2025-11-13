"""
Regression test using Antarctic Grounded Drainage System Polygons

This test processes 1.2+ million real coordinates from Antarctic polygon data
and verifies that morton indices remain unchanged. Any modifications to the
morton encoding algorithm will cause this test to fail.

Data source: Ant_Grounded_DrainageSystem_Polygons.txt (1,239,001 coordinates)
Reference: Ant_Grounded_DrainageSystem_Polygons_morton.npz (order=18)
"""

import pytest
import numpy as np
from pathlib import Path
from numpy.testing import assert_array_equal

from mortie import tools


# File paths
TEST_DIR = Path(__file__).parent
COORDS_FILE = TEST_DIR / "Ant_Grounded_DrainageSystem_Polygons.txt"
REFERENCE_FILE = TEST_DIR / "Ant_Grounded_DrainageSystem_Polygons_morton.npz"

# Skip if reference file doesn't exist
pytestmark = pytest.mark.skipif(
    not REFERENCE_FILE.exists(),
    reason="Reference morton file not generated. Run generate_morton_reference.py first."
)


class TestPolygonRegression:
    """Regression tests using Antarctic polygon coordinate data"""

    @pytest.fixture(scope="class")
    def polygon_coordinates(self):
        """Load coordinates from polygon file"""
        print(f"\nLoading coordinates from {COORDS_FILE.name}...")
        data = np.loadtxt(COORDS_FILE)

        coords = {
            'lats': data[:, 0],
            'lons': data[:, 1],
            'polygon_ids': data[:, 2].astype(np.int32),
            'count': len(data)
        }

        print(f"  Loaded {coords['count']:,} coordinates")
        print(f"  Lat range: [{coords['lats'].min():.2f}, {coords['lats'].max():.2f}]")
        print(f"  Lon range: [{coords['lons'].min():.2f}, {coords['lons'].max():.2f}]")

        return coords

    @pytest.fixture(scope="class")
    def reference_morton(self):
        """Load reference morton indices"""
        print(f"\nLoading reference morton indices from {REFERENCE_FILE.name}...")
        data = np.load(REFERENCE_FILE)

        reference = {
            'morton': data['morton_indices'],
            'order': int(data['order'][0]),
            'metadata': data['metadata']
        }

        print(f"  Loaded {len(reference['morton']):,} morton indices")
        print(f"  Order: {reference['order']}")
        print(f"  Morton range: [{reference['morton'].min()}, {reference['morton'].max()}]")

        return reference

    def test_files_exist(self):
        """Verify both coordinate and reference files exist"""
        assert COORDS_FILE.exists(), f"Coordinate file not found: {COORDS_FILE}"
        assert REFERENCE_FILE.exists(), f"Reference file not found: {REFERENCE_FILE}"

    def test_coordinates_loaded(self, polygon_coordinates):
        """Verify coordinates were loaded correctly"""
        assert polygon_coordinates['count'] > 0
        assert len(polygon_coordinates['lats']) == polygon_coordinates['count']
        assert len(polygon_coordinates['lons']) == polygon_coordinates['count']

        # Verify Antarctica coordinates
        assert np.all(polygon_coordinates['lats'] >= -90)
        assert np.all(polygon_coordinates['lats'] <= -60)  # Southern hemisphere, Antarctic region

    def test_reference_loaded(self, reference_morton):
        """Verify reference morton indices were loaded correctly"""
        assert len(reference_morton['morton']) > 0
        assert reference_morton['order'] == 18

    def test_morton_regression_full(self, polygon_coordinates, reference_morton):
        """
        CRITICAL REGRESSION TEST: Full morton index comparison

        This test verifies that ALL 1.2+ million morton indices match
        the reference. If this test fails, the morton encoding has changed.
        """
        lats = polygon_coordinates['lats']
        lons = polygon_coordinates['lons']
        order = reference_morton['order']

        print(f"\n{'='*70}")
        print(f"REGRESSION TEST: Computing morton indices for {len(lats):,} coordinates")
        print(f"{'='*70}")

        # Compute morton indices
        morton_new = tools.geo2mort(lats, lons, order=order)

        print(f"  Computed: {len(morton_new):,} indices")
        print(f"  Reference: {len(reference_morton['morton']):,} indices")

        # Compare with reference
        morton_ref = reference_morton['morton']

        # Check lengths match
        assert len(morton_new) == len(morton_ref), (
            f"Length mismatch: computed {len(morton_new):,} vs reference {len(morton_ref):,}"
        )

        # Check all values match
        assert_array_equal(
            morton_new, morton_ref,
            err_msg="Morton indices have changed! This indicates a regression in the morton encoding algorithm."
        )

        print(f"  ✓ All {len(morton_new):,} morton indices match reference")

    def test_morton_determinism(self, polygon_coordinates):
        """Test that morton computation is deterministic"""
        lats = polygon_coordinates['lats'][:10000]  # Sample first 10k
        lons = polygon_coordinates['lons'][:10000]

        # Compute multiple times
        morton1 = tools.geo2mort(lats, lons, order=18)
        morton2 = tools.geo2mort(lats, lons, order=18)
        morton3 = tools.geo2mort(lats, lons, order=18)

        # All should match
        assert_array_equal(morton1, morton2)
        assert_array_equal(morton2, morton3)

    def test_morton_structure_sample(self, reference_morton):
        """Verify morton indices have valid digit structure (sample)"""
        morton = reference_morton['morton']

        # Sample 10,000 random indices
        np.random.seed(42)
        sample_indices = np.random.choice(len(morton), size=min(10000, len(morton)), replace=False)
        morton_sample = morton[sample_indices]

        invalid_count = 0
        for m in morton_sample:
            morton_str = str(abs(m))
            if len(morton_str) > 2:
                trailing_digits = morton_str[2:]
                invalid_digits = [d for d in trailing_digits if d not in '1234']
                if invalid_digits:
                    invalid_count += 1

        assert invalid_count == 0, (
            f"Found {invalid_count} morton indices with invalid digits (not 1-4)"
        )

    def test_morton_subsample_regression(self, polygon_coordinates, reference_morton):
        """Regression test on stratified subsample (faster smoke test)"""
        # Take every 100th coordinate for quick verification
        lats = polygon_coordinates['lats'][::100]
        lons = polygon_coordinates['lons'][::100]
        morton_ref = reference_morton['morton'][::100]

        print(f"\nSubsample test: {len(lats):,} coordinates (every 100th)")

        morton_new = tools.geo2mort(lats, lons, order=18)

        assert_array_equal(
            morton_new, morton_ref,
            err_msg="Subsample morton indices don't match reference"
        )

        print(f"  ✓ Subsample matches reference")

    def test_polygon_id_consistency(self, polygon_coordinates):
        """Verify polygon IDs are reasonable"""
        polygon_ids = polygon_coordinates['polygon_ids']

        n_polygons = len(np.unique(polygon_ids))
        print(f"\nPolygon statistics:")
        print(f"  Total coordinates: {len(polygon_ids):,}")
        print(f"  Unique polygons: {n_polygons}")

        # Should have multiple polygons
        assert n_polygons > 1
        assert n_polygons < len(polygon_ids)  # But not one per coordinate

    def test_coordinate_coverage(self, polygon_coordinates):
        """Verify coordinates cover Antarctic region"""
        lats = polygon_coordinates['lats']
        lons = polygon_coordinates['lons']

        # Should cover substantial lat range in Antarctica
        lat_range = lats.max() - lats.min()
        assert lat_range > 10.0, "Latitude coverage too small"

        # Should cover substantial longitude range
        lon_range = lons.max() - lons.min()
        assert lon_range > 100.0, "Longitude coverage too small"

    def test_statistics_summary(self, polygon_coordinates, reference_morton):
        """Print comprehensive statistics about the dataset"""
        coords = polygon_coordinates
        morton = reference_morton['morton']

        print("\n" + "="*70)
        print("DATASET STATISTICS")
        print("="*70)

        print(f"\nCoordinates:")
        print(f"  Total points:     {coords['count']:,}")
        print(f"  Latitude range:   [{coords['lats'].min():.2f}, {coords['lats'].max():.2f}]")
        print(f"  Longitude range:  [{coords['lons'].min():.2f}, {coords['lons'].max():.2f}]")
        print(f"  Unique polygons:  {len(np.unique(coords['polygon_ids']))}")

        print(f"\nMorton Indices (order={reference_morton['order']}):")
        print(f"  Total indices:    {len(morton):,}")
        print(f"  Range:            [{morton.min()}, {morton.max()}]")
        print(f"  Dtype:            {morton.dtype}")
        print(f"  Memory (MB):      {morton.nbytes / 1024 / 1024:.2f}")

        # Sign distribution
        n_positive = np.sum(morton > 0)
        n_negative = np.sum(morton < 0)
        print(f"\nSign distribution:")
        print(f"  Positive:         {n_positive:,} ({100*n_positive/len(morton):.1f}%)")
        print(f"  Negative:         {n_negative:,} ({100*n_negative/len(morton):.1f}%)")

        # Sample values
        print(f"\nSample morton indices:")
        for i in [0, len(morton)//4, len(morton)//2, 3*len(morton)//4, -1]:
            print(f"  [{i:7d}]: {morton[i]}")

        print("="*70)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

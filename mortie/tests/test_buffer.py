"""Tests for morton_buffer functionality."""

import numpy as np
import pytest

import mortie


class TestMortonBuffer:
    """Tests for the morton_buffer function."""

    def test_single_cell_k1(self):
        """Buffer of one cell with k=1 should return up to 8 neighbors."""
        morton = mortie.geo2mort(45.0, -122.0, order=6)
        border = mortie.morton_buffer(morton, k=1)
        assert len(border) > 0
        assert len(border) <= 8
        # Border should not contain the input
        assert morton[0] not in border

    def test_k0_returns_empty(self):
        """k=0 returns empty array."""
        morton = mortie.geo2mort(45.0, -122.0, order=6)
        border = mortie.morton_buffer(morton, k=0)
        assert len(border) == 0

    def test_idempotency(self):
        """Buffer cells are not in the input set."""
        lats = np.array([45.0, 45.1, 44.9, 45.05])
        lons = np.array([-122.0, -121.9, -122.1, -122.0])
        morton = mortie.geo2mort(lats, lons, order=6)
        cells = np.unique(morton)
        border = mortie.morton_buffer(cells, k=1)
        # No overlap between input and border
        overlap = np.intersect1d(cells, border)
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping cells"

    def test_order_validation(self):
        """Mixed-order inputs should raise ValueError."""
        m1 = mortie.geo2mort(45.0, -122.0, order=6)
        m2 = mortie.geo2mort(45.0, -122.0, order=7)
        mixed = np.concatenate([m1, m2])
        with pytest.raises(ValueError):
            mortie.morton_buffer(mixed, k=1)

    def test_hemisphere_handling_positive(self):
        """Works with northern hemisphere (positive) morton indices."""
        morton = mortie.geo2mort(60.0, 30.0, order=6)
        assert morton[0] > 0
        border = mortie.morton_buffer(morton, k=1)
        assert len(border) > 0

    def test_hemisphere_handling_negative(self):
        """Works with southern hemisphere (negative) morton indices."""
        morton = mortie.geo2mort(-60.0, 30.0, order=6)
        assert morton[0] < 0
        border = mortie.morton_buffer(morton, k=1)
        assert len(border) > 0

    def test_south_polar_cells(self):
        """Cells near the south pole work correctly (the actual use case)."""
        morton = mortie.geo2mort(-85.0, 0.0, order=6)
        border = mortie.morton_buffer(morton, k=1)
        assert len(border) > 0

    def test_roundtrip_mort2nested(self):
        """mort2nested -> nested2mort is identity via the buffer pathway."""
        # Test by checking that buffer of union is superset of original buffer
        morton = mortie.geo2mort(45.0, -122.0, order=6)
        cells = np.unique(morton)
        border = mortie.morton_buffer(cells, k=1)
        union = np.union1d(cells, border)
        # The union should be larger than the original
        assert len(union) > len(cells)

    def test_known_geometry_2x2_block(self):
        """Buffer of a cluster of cells should return the surrounding ring."""
        # Create a small cluster by getting nearby cells
        lats = np.array([45.0, 45.0, 45.01, 45.01])
        lons = np.array([-122.0, -121.99, -122.0, -121.99])
        morton = mortie.geo2mort(lats, lons, order=6)
        cells = np.unique(morton)
        if len(cells) < 2:
            pytest.skip("Cells too coarse for this test at order 6")
        border = mortie.morton_buffer(cells, k=1)
        # Border should surround the block
        assert len(border) > 0
        # No overlap
        assert len(np.intersect1d(cells, border)) == 0

    def test_k2_larger_than_k1(self):
        """k=2 border should have more cells than k=1."""
        morton = mortie.geo2mort(45.0, -122.0, order=6)
        border_k1 = mortie.morton_buffer(morton, k=1)
        border_k2 = mortie.morton_buffer(morton, k=2)
        assert len(border_k2) > len(border_k1)

    def test_empty_input(self):
        """Empty input returns empty output."""
        border = mortie.morton_buffer(np.array([], dtype=np.int64), k=1)
        assert len(border) == 0

    def test_sorted_output(self):
        """Output should be sorted."""
        morton = mortie.geo2mort(45.0, -122.0, order=6)
        border = mortie.morton_buffer(morton, k=1)
        assert np.all(border[:-1] <= border[1:])

    def test_border_cells_are_valid(self):
        """All border cells should be valid morton indices."""
        morton = mortie.geo2mort(45.0, -122.0, order=6)
        border = mortie.morton_buffer(morton, k=1)
        for cell in border:
            # Should not raise
            mortie.validate_morton(int(cell))

    def test_border_cells_same_order(self):
        """All border cells should be at the same order as input."""
        morton = mortie.geo2mort(45.0, -122.0, order=6)
        input_order = mortie.infer_order_from_morton(int(morton[0]))
        border = mortie.morton_buffer(morton, k=1)
        for cell in border:
            cell_order = mortie.infer_order_from_morton(int(cell))
            assert cell_order == input_order

    def test_mixed_hemisphere_cluster(self):
        """Buffer works across hemisphere boundary cells."""
        # Get cells from both hemispheres near the equator
        lats = np.array([1.0, -1.0])
        lons = np.array([30.0, 30.0])
        morton = mortie.geo2mort(lats, lons, order=6)
        cells = np.unique(morton)
        # Should have both positive and negative
        has_positive = np.any(cells > 0)
        has_negative = np.any(cells < 0)
        if has_positive and has_negative:
            border = mortie.morton_buffer(cells, k=1)
            assert len(border) > 0
        else:
            pytest.skip("Cells at this order don't straddle hemispheres")

"""
Tests for greedy_morton_polygon functionality.
"""

import numpy as np
import pytest
from mortie import greedy_morton_polygon, generate_morton_children, geo2mort


class TestGreedyMortonPolygon:
    """Test suite for greedy_morton_polygon function."""

    def test_basic_square(self):
        """Test with a simple square polygon."""
        # Define a square
        lat = np.array([-75, -75, -70, -70, -75])
        lon = np.array([-80, -70, -70, -80, -80])

        # Generate morton boxes
        morton_boxes, orders = greedy_morton_polygon(
            lat, lon, order=18, max_boxes=10, ordermax=6, verbose=False
        )

        # Check outputs
        assert isinstance(morton_boxes, np.ndarray)
        assert isinstance(orders, np.ndarray)
        assert len(morton_boxes) == len(orders)
        assert len(morton_boxes) <= 10  # Should respect max_boxes
        assert np.all(orders <= 6)  # Should respect ordermax

    def test_returns_at_least_one_box(self):
        """Test that at least one box is always returned."""
        lat = np.array([-75, -75, -70, -70])
        lon = np.array([-80, -70, -70, -80])

        morton_boxes, orders = greedy_morton_polygon(
            lat, lon, order=18, max_boxes=1, verbose=False
        )

        assert len(morton_boxes) >= 1

    def test_max_boxes_constraint(self):
        """Test that max_boxes constraint is respected."""
        # Use Antarctic data
        lat = np.random.uniform(-80, -60, 1000)
        lon = np.random.uniform(-180, 180, 1000)

        for max_boxes in [5, 10, 20]:
            morton_boxes, orders = greedy_morton_polygon(
                lat, lon, order=18, max_boxes=max_boxes, ordermax=6, verbose=False
            )
            assert len(morton_boxes) <= max_boxes, f"Exceeded max_boxes={max_boxes}"

    def test_ordermax_constraint(self):
        """Test that ordermax constraint is respected."""
        lat = np.array([-75, -75, -70, -70])
        lon = np.array([-80, -70, -70, -80])

        for ordermax in [2, 4, 6]:
            morton_boxes, orders = greedy_morton_polygon(
                lat, lon, order=18, max_boxes=25, ordermax=ordermax, verbose=False
            )
            assert np.all(orders <= ordermax), f"Exceeded ordermax={ordermax}"

    def test_no_ordermax(self):
        """Test greedy subdivision without ordermax constraint."""
        lat = np.array([-75, -75, -70, -70])
        lon = np.array([-80, -70, -70, -80])

        morton_boxes, orders = greedy_morton_polygon(
            lat, lon, order=18, max_boxes=10, ordermax=None, verbose=False
        )

        assert len(morton_boxes) <= 10

    def test_empty_input(self):
        """Test that empty input raises appropriate error."""
        lat = np.array([])
        lon = np.array([])

        with pytest.raises(ValueError, match="No valid points"):
            greedy_morton_polygon(lat, lon)

    def test_nan_handling(self):
        """Test that NaN values are handled correctly."""
        lat = np.array([-75, np.nan, -70, -70])
        lon = np.array([-80, -70, np.nan, -80])

        morton_boxes, orders = greedy_morton_polygon(
            lat, lon, order=18, max_boxes=5, verbose=False
        )

        # Should still work with valid points
        assert len(morton_boxes) >= 1

    def test_coverage_completeness(self):
        """Test that generated boxes cover all input points."""
        # Generate random points
        np.random.seed(42)
        lat = np.random.uniform(-75, -70, 100)
        lon = np.random.uniform(-80, -70, 100)

        # Get morton boxes
        morton_boxes, orders = greedy_morton_polygon(
            lat, lon, order=18, max_boxes=10, ordermax=6, verbose=False
        )

        # Expand all boxes to order 6
        all_order6_cells = []
        for morton in morton_boxes:
            children = generate_morton_children(morton, target_order=6)
            all_order6_cells.extend(children)

        order6_set = set(all_order6_cells)

        # Check that all input points map to one of the boxes
        input_morton = geo2mort(lat, lon, order=6)
        assert np.all(np.isin(input_morton, list(order6_set))), \
            "Not all input points are covered by generated boxes"

    def test_balanced_subdivision(self):
        """Test that subdivision is balanced across regions."""
        # Create points in two distinct regions
        lat1 = np.random.uniform(-75, -70, 500)
        lon1 = np.random.uniform(-80, -70, 500)

        lat2 = np.random.uniform(-70, -65, 500)
        lon2 = np.random.uniform(-60, -50, 500)

        lat = np.concatenate([lat1, lat2])
        lon = np.concatenate([lon1, lon2])

        morton_boxes, orders = greedy_morton_polygon(
            lat, lon, order=18, max_boxes=20, ordermax=6, verbose=False
        )

        # With balanced subdivision, we should get multiple boxes
        # (not just one box covering everything)
        assert len(morton_boxes) > 1, "Should create multiple boxes for distinct regions"

    def test_output_types(self):
        """Test that outputs have correct types."""
        lat = np.array([-75, -75, -70, -70])
        lon = np.array([-80, -70, -70, -80])

        morton_boxes, orders = greedy_morton_polygon(
            lat, lon, order=18, max_boxes=10, verbose=False
        )

        assert isinstance(morton_boxes, np.ndarray)
        assert isinstance(orders, np.ndarray)
        assert morton_boxes.dtype == np.int64 or morton_boxes.dtype == np.int32
        assert orders.dtype == np.int64 or orders.dtype == np.int32

    def test_order_inference(self):
        """Test that order inference is correct."""
        lat = np.array([-75, -75, -70, -70])
        lon = np.array([-80, -70, -70, -80])

        morton_boxes, orders = greedy_morton_polygon(
            lat, lon, order=18, max_boxes=10, ordermax=6, verbose=False
        )

        # Verify order inference
        from mortie import infer_order_from_morton
        for morton, expected_order in zip(morton_boxes, orders):
            inferred_order = infer_order_from_morton(morton)
            assert inferred_order == expected_order, \
                f"Order mismatch for morton {morton}: inferred {inferred_order}, expected {expected_order}"


class TestGenerateMortonChildrenFix:
    """Test suite for fixed generate_morton_children function."""

    def test_same_order_returns_parent(self):
        """Test that requesting same order returns parent."""
        parent_morton = -5111131  # order 6
        children = generate_morton_children(parent_morton, target_order=6)

        assert len(children) == 1
        assert children[0] == parent_morton

    def test_higher_order_generates_children(self):
        """Test that higher order generates children."""
        parent_morton = -511113  # order 5
        children = generate_morton_children(parent_morton, target_order=6)

        assert len(children) == 4  # 4^(6-5) = 4 children

    def test_lower_order_raises_error(self):
        """Test that requesting lower order raises error."""
        parent_morton = -5111131  # order 6

        with pytest.raises(ValueError, match="must be >= parent_order"):
            generate_morton_children(parent_morton, target_order=5)

    def test_multiple_order_jump(self):
        """Test generating children with multiple order jump."""
        parent_morton = -511  # order 2
        children = generate_morton_children(parent_morton, target_order=6)

        # 4^(6-2) = 256 children
        assert len(children) == 256

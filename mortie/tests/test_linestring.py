"""Tests for linestring_coverage and morton_buffer_meters."""

import numpy as np
import pytest

import mortie


class TestSingleLinestring:
    def test_returns_ndarray(self):
        lats = [40.0, 50.0, 45.0]
        lons = [-120.0, -110.0, -100.0]
        result = mortie.linestring_coverage(lats, lons, order=6)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.dtype == np.int64
        assert result.size >= 3

    def test_default_order_is_18(self):
        lats = [10.0, 10.001]
        lons = [20.0, 20.001]
        r_default = mortie.linestring_coverage(lats, lons)
        r_18 = mortie.linestring_coverage(lats, lons, order=18)
        np.testing.assert_array_equal(r_default, r_18)

    def test_sorted_unique(self):
        lats = [40.0, 50.0, 45.0]
        lons = [-120.0, -110.0, -100.0]
        result = mortie.linestring_coverage(lats, lons, order=6)
        assert np.all(np.diff(result) > 0)

    def test_endpoints_present(self):
        lats = np.array([40.0, 50.0, 45.0])
        lons = np.array([-120.0, -110.0, -100.0])
        order = 6
        result = mortie.linestring_coverage(lats, lons, order=order)
        endpoint_cells = mortie.geo2mort(lats, lons, order=order)
        result_set = set(result.tolist())
        for m in endpoint_cells.tolist():
            assert m in result_set

    def test_two_vertex_minimum(self):
        # Exactly 2 vertices is the minimum for a linestring. Pick vertices
        # far enough apart that even at low order they land in different cells.
        lats = [0.0, 40.0]
        lons = [0.0, 40.0]
        result = mortie.linestring_coverage(lats, lons, order=6)
        assert result.size >= 2

    def test_higher_order_more_cells(self):
        lats = [10.0, 30.0]
        lons = [40.0, 60.0]
        r4 = mortie.linestring_coverage(lats, lons, order=4)
        r8 = mortie.linestring_coverage(lats, lons, order=8)
        assert r8.size > r4.size

    def test_northern_hemisphere_positive(self):
        lats = [40.0, 50.0, 45.0]
        lons = [-120.0, -110.0, -100.0]
        result = mortie.linestring_coverage(lats, lons, order=6)
        assert np.all(result > 0)

    def test_southern_hemisphere_negative(self):
        lats = [-70.0, -80.0, -75.0]
        lons = [30.0, 30.0, 50.0]
        result = mortie.linestring_coverage(lats, lons, order=6)
        assert np.all(result < 0)

    def test_interpolation_fills_gap(self):
        # Two vertices far enough apart that at order 10 (~10 km cells) the
        # raw endpoints would be non-adjacent; the result should contain more
        # than just the two endpoint cells (interpolation fills the gap).
        lats = [10.0, 11.0]  # ~111 km apart
        lons = [30.0, 30.0]
        result = mortie.linestring_coverage(lats, lons, order=10)
        # Endpoints alone would be 2 cells; interpolation must add many more
        assert result.size > 2

    def test_accepts_numpy_arrays(self):
        lats = np.array([40.0, 50.0], dtype=np.float64)
        lons = np.array([-120.0, -110.0], dtype=np.float64)
        result = mortie.linestring_coverage(lats, lons, order=6)
        assert result.size >= 2


class TestMultiLinestring:
    def test_returns_list_of_arrays(self):
        lats_parts = [[40.0, 50.0], [10.0, 20.0, 15.0]]
        lons_parts = [[-120.0, -110.0], [-80.0, -70.0, -60.0]]
        result = mortie.linestring_coverage(lats_parts, lons_parts, order=6)
        assert isinstance(result, list)
        assert len(result) == 2
        for arr in result:
            assert isinstance(arr, np.ndarray)
            assert arr.ndim == 1
            assert arr.dtype == np.int64

    def test_per_line_matches_single_call(self):
        lats_parts = [[40.0, 50.0, 45.0], [10.0, 20.0, 15.0]]
        lons_parts = [[-120.0, -110.0, -100.0], [-80.0, -70.0, -60.0]]
        multi = mortie.linestring_coverage(lats_parts, lons_parts, order=6)
        single_a = mortie.linestring_coverage(lats_parts[0], lons_parts[0], order=6)
        single_b = mortie.linestring_coverage(lats_parts[1], lons_parts[1], order=6)
        np.testing.assert_array_equal(multi[0], single_a)
        np.testing.assert_array_equal(multi[1], single_b)

    def test_per_line_no_cross_deduplication(self):
        # Two identical lines -> multi result should have identical arrays
        # (NOT unioned / deduplicated across lines)
        lats = [40.0, 50.0, 45.0]
        lons = [-120.0, -110.0, -100.0]
        result = mortie.linestring_coverage([lats, lats], [lons, lons], order=6)
        np.testing.assert_array_equal(result[0], result[1])

    def test_lengths_may_differ(self):
        lats_parts = [[40.0, 50.0], [10.0, 20.0, 15.0, 5.0, 0.0]]
        lons_parts = [[-120.0, -120.0001], [-80.0, -70.0, -60.0, -50.0, -40.0]]
        result = mortie.linestring_coverage(lats_parts, lons_parts, order=6)
        # The longer line should almost certainly have more cells
        assert result[1].size > result[0].size

    def test_mismatched_parts_raises(self):
        with pytest.raises(ValueError, match="same number of parts"):
            mortie.linestring_coverage([[0.0, 1.0], [2.0, 3.0]], [[0.0, 1.0]], order=6)


class TestValidation:
    def test_too_few_vertices(self):
        with pytest.raises(ValueError, match="at least 2 vertices"):
            mortie.linestring_coverage([0.0], [0.0], order=6)

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            mortie.linestring_coverage([0.0, 1.0, 2.0], [0.0, 1.0], order=6)

    def test_order_out_of_range_low(self):
        with pytest.raises(ValueError, match="Order must be"):
            mortie.linestring_coverage([0.0, 1.0], [0.0, 1.0], order=0)

    def test_order_out_of_range_high(self):
        with pytest.raises(ValueError, match="Order must be"):
            mortie.linestring_coverage([0.0, 1.0], [0.0, 1.0], order=19)

    def test_nan_coordinates(self):
        with pytest.raises(ValueError, match="NaN"):
            mortie.linestring_coverage([0.0, np.nan], [0.0, 1.0], order=6)

    def test_inf_coordinates(self):
        with pytest.raises(ValueError, match="NaN|infinity"):
            mortie.linestring_coverage([0.0, 1.0], [0.0, np.inf], order=6)


class TestMortonBufferMeters:
    def test_returns_border_array(self):
        cells = mortie.linestring_coverage([10.0, 20.0], [30.0, 40.0], order=8)
        border = mortie.morton_buffer_meters(cells, width_m=5000.0)
        assert isinstance(border, np.ndarray)
        assert border.dtype == np.int64
        assert border.size > 0
        # Border and input should be disjoint
        assert np.intersect1d(cells, border).size == 0

    def test_larger_width_gives_more_cells(self):
        cells = mortie.linestring_coverage([10.0, 20.0], [30.0, 40.0], order=8)
        b1 = mortie.morton_buffer_meters(cells, width_m=1000.0)
        b2 = mortie.morton_buffer_meters(cells, width_m=100_000.0)
        assert b2.size > b1.size

    def test_small_width_still_produces_k1_ring(self):
        # Even a tiny width_m should give at least k=1 (one-cell ring)
        cells = mortie.linestring_coverage([10.0, 20.0], [30.0, 40.0], order=6)
        expected_k1 = mortie.morton_buffer(cells, k=1)
        border = mortie.morton_buffer_meters(cells, width_m=1e-6)
        np.testing.assert_array_equal(np.sort(border), np.sort(expected_k1))

    def test_matches_equivalent_k(self):
        # Picking a width that equals exactly k=2 cell widths should produce
        # the same result as morton_buffer(cells, k=2).
        cells = mortie.linestring_coverage([10.0, 20.0], [30.0, 40.0], order=10)
        order = 10
        nside = 1 << order
        cell_width_m = 6_371_008.7714 * np.sqrt(np.pi / 3.0) / nside
        # Slightly less than 2 cell widths so ceil -> 2
        border = mortie.morton_buffer_meters(cells, width_m=1.999 * cell_width_m)
        expected = mortie.morton_buffer(cells, k=2)
        np.testing.assert_array_equal(np.sort(border), np.sort(expected))

    def test_non_positive_width_raises(self):
        cells = mortie.linestring_coverage([10.0, 20.0], [30.0, 40.0], order=6)
        with pytest.raises(ValueError, match="positive"):
            mortie.morton_buffer_meters(cells, width_m=0.0)
        with pytest.raises(ValueError, match="positive"):
            mortie.morton_buffer_meters(cells, width_m=-10.0)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            mortie.morton_buffer_meters(np.array([], dtype=np.int64), width_m=100.0)

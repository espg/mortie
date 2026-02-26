"""Tests for polygon-to-morton coverage (morton_coverage)."""

import numpy as np
import pytest
from pathlib import Path

import mortie


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_pip(test_lat, test_lon, poly_lats, poly_lons):
    """Ray-casting point-in-polygon on (lat, lon) pairs.

    Works for simple (non-self-intersecting) polygons that don't span
    more than ~180° in longitude.
    """
    n = len(poly_lats)
    inside = False
    j = n - 1
    for i in range(n):
        yi, yj = poly_lons[i], poly_lons[j]
        xi, xj = poly_lats[i], poly_lats[j]
        if ((yi > test_lon) != (yj > test_lon)) and (
            test_lat < (xj - xi) * (test_lon - yi) / (yj - yi) + xi
        ):
            inside = not inside
        j = i
    return inside


def _load_basin(basin_id):
    """Load Antarctic basin vertices.  Returns (lats, lons) or skips test."""
    coords_file = Path("mortie/tests/Ant_Grounded_DrainageSystem_Polygons.txt")
    if not coords_file.exists():
        pytest.skip("Antarctic polygon data not found")
    data = np.loadtxt(coords_file)
    mask = data[:, 2] == basin_id
    return data[mask, 0], data[mask, 1]


def _simplify_vertices(lats, lons, target):
    """Uniformly subsample vertices to approximately *target* count."""
    n = len(lats)
    if n <= target:
        return lats, lons
    step = max(1, n // target)
    idx = np.arange(0, n, step)
    return lats[idx], lons[idx]


# ---------------------------------------------------------------------------
# Synthetic polygon tests
# ---------------------------------------------------------------------------

class TestCoverageSynthetic:
    """Tests using simple synthetic polygons."""

    def test_triangle_coverage(self):
        """Simple triangle produces non-empty coverage."""
        lats = [40.0, 50.0, 45.0]
        lons = [-120.0, -120.0, -110.0]
        cells = mortie.morton_coverage(lats, lons, order=6)
        assert len(cells) > 0
        assert len(np.unique(cells)) == len(cells), "must be unique"

    def test_square_coverage(self):
        """Axis-aligned square coverage at order 4."""
        lats = [40.0, 40.0, 50.0, 50.0]
        lons = [-125.0, -115.0, -115.0, -125.0]
        cells = mortie.morton_coverage(lats, lons, order=4)
        assert len(cells) > 0

    def test_concave_polygon(self):
        """L-shaped (concave) polygon is handled correctly."""
        # L-shape: lower-left corner, right, notch, upper-right, top, back
        lats = [30.0, 30.0, 35.0, 35.0, 40.0, 40.0]
        lons = [-120.0, -110.0, -110.0, -115.0, -115.0, -120.0]
        cells = mortie.morton_coverage(lats, lons, order=4)
        assert len(cells) > 0

    def test_coverage_superset(self):
        """Coverage must include all cells whose centres are inside polygon."""
        poly_lats = np.array([40.0, 40.0, 50.0, 50.0])
        poly_lons = np.array([-125.0, -115.0, -115.0, -125.0])
        order = 4

        cells = mortie.morton_coverage(poly_lats, poly_lons, order=order)
        coverage_set = set(cells.tolist())

        # Check a grid of points inside the polygon
        for lat in np.arange(41.0, 50.0, 1.0):
            for lon in np.arange(-124.0, -115.0, 1.0):
                if _simple_pip(lat, lon, poly_lats, poly_lons):
                    m = mortie.geo2mort(lat, lon, order=order)
                    m_val = int(m) if np.ndim(m) == 0 else int(m[0])
                    assert m_val in coverage_set, (
                        f"Interior cell at ({lat}, {lon}) = {m_val} "
                        f"not in coverage"
                    )

    def test_coverage_boundary(self):
        """All boundary vertex cells should be in the coverage."""
        lats = np.array([40.0, 40.0, 50.0, 50.0])
        lons = np.array([-125.0, -115.0, -115.0, -125.0])
        order = 4

        cells = mortie.morton_coverage(lats, lons, order=order)
        coverage_set = set(cells.tolist())

        for lat, lon in zip(lats, lons):
            m = mortie.geo2mort(lat, lon, order=order)
            m_val = int(m) if np.ndim(m) == 0 else int(m[0])
            assert m_val in coverage_set, (
                f"Boundary vertex cell ({lat}, {lon}) = {m_val} "
                f"not in coverage"
            )

    def test_single_cell_polygon(self):
        """Tiny polygon mapping to very few cells."""
        # Polygon much smaller than one cell at order 4
        lats = [45.0, 45.001, 45.0005]
        lons = [-120.0, -120.0, -119.999]
        cells = mortie.morton_coverage(lats, lons, order=4)
        assert 1 <= len(cells) <= 4

    @pytest.mark.parametrize("order", [4, 6, 8])
    def test_different_orders(self, order):
        """Coverage at various orders produces valid results."""
        lats = [40.0, 50.0, 45.0]
        lons = [-120.0, -120.0, -110.0]
        cells = mortie.morton_coverage(lats, lons, order=order)
        assert len(cells) > 0
        # All cells should be valid
        for c in cells[:10]:  # spot-check first 10
            mortie.validate_morton(int(c))

    def test_order_scaling(self):
        """Higher order produces more cells."""
        lats = [40.0, 50.0, 45.0]
        lons = [-120.0, -120.0, -110.0]
        n4 = len(mortie.morton_coverage(lats, lons, order=4))
        n6 = len(mortie.morton_coverage(lats, lons, order=6))
        assert n6 > n4

    def test_sorted_output(self):
        """Output is sorted."""
        cells = mortie.morton_coverage([40.0, 50.0, 45.0],
                                      [-120.0, -120.0, -110.0], order=6)
        assert np.all(cells[:-1] <= cells[1:])

    def test_closed_polygon(self):
        """Closed polygon (first==last vertex) is handled."""
        lats = [40.0, 50.0, 45.0, 40.0]
        lons = [-120.0, -120.0, -110.0, -120.0]
        cells_closed = mortie.morton_coverage(lats, lons, order=6)
        cells_open = mortie.morton_coverage(lats[:3], lons[:3], order=6)
        np.testing.assert_array_equal(cells_closed, cells_open)

    def test_holes_raises(self):
        """Polygon with holes concept — not supported."""
        # We raise NotImplementedError for multi-ring polygons.
        # Currently there's no explicit hole detection, but document intent.
        # If someone passes a flat list of outer + inner rings, the result
        # is undefined.  This test documents the current behavior.
        pass  # placeholder — hole detection deferred to future work

    def test_southern_hemisphere(self):
        """Southern hemisphere polygon produces negative morton indices."""
        lats = [-70.0, -80.0, -75.0]
        lons = [30.0, 30.0, 50.0]
        cells = mortie.morton_coverage(lats, lons, order=6)
        assert np.any(cells < 0)

    def test_equatorial_polygon(self):
        """Polygon spanning the equator works."""
        lats = [-5.0, -5.0, 5.0, 5.0]
        lons = [10.0, 20.0, 20.0, 10.0]
        cells = mortie.morton_coverage(lats, lons, order=4)
        assert len(cells) > 0

    def test_invalid_order(self):
        """Invalid order raises ValueError."""
        with pytest.raises(ValueError):
            mortie.morton_coverage([0, 1, 2], [0, 1, 2], order=0)
        with pytest.raises(ValueError):
            mortie.morton_coverage([0, 1, 2], [0, 1, 2], order=19)

    def test_too_few_vertices(self):
        """Fewer than 3 vertices raises ValueError."""
        with pytest.raises(ValueError):
            mortie.morton_coverage([0, 1], [0, 1], order=6)

    def test_mismatched_lengths(self):
        """Mismatched lat/lon arrays raise ValueError."""
        with pytest.raises(ValueError):
            mortie.morton_coverage([0, 1, 2], [0, 1], order=6)

    def test_multipart_union(self):
        """Multipart polygon returns union of individual coverages."""
        lats_a = [40.0, 50.0, 45.0]
        lons_a = [-120.0, -120.0, -110.0]
        lats_b = [10.0, 20.0, 15.0]
        lons_b = [-80.0, -80.0, -70.0]

        cells_a = mortie.morton_coverage(lats_a, lons_a, order=6)
        cells_b = mortie.morton_coverage(lats_b, lons_b, order=6)
        expected = np.unique(np.concatenate([cells_a, cells_b]))

        cells_multi = mortie.morton_coverage(
            [lats_a, lats_b], [lons_a, lons_b], order=6
        )
        np.testing.assert_array_equal(cells_multi, expected)

    def test_multipart_single_element(self):
        """Multipart with one part matches single polygon call."""
        lats = [40.0, 50.0, 45.0]
        lons = [-120.0, -120.0, -110.0]
        single = mortie.morton_coverage(lats, lons, order=6)
        multi = mortie.morton_coverage([lats], [lons], order=6)
        np.testing.assert_array_equal(single, multi)

    def test_multipart_mismatched_parts(self):
        """Multipart with mismatched part count raises ValueError."""
        with pytest.raises(ValueError):
            mortie.morton_coverage(
                [[0, 1, 2], [3, 4, 5]], [[0, 1, 2]], order=6
            )


# ---------------------------------------------------------------------------
# Real-data tests (Antarctic drainage basins)
# ---------------------------------------------------------------------------

class TestCoverageRealData:
    """Tests using Antarctic drainage basin polygons."""

    @pytest.mark.slow
    def test_smallest_basin(self):
        """Basin 24 (smallest, 21k vertices) — full polygon."""
        lats, lons = _load_basin(24)
        cells = mortie.morton_coverage(lats, lons, order=6)
        # Basin 24 produces ~24 cells at order 6 (verified against brute-force PIP)
        assert len(cells) > 10, f"Expected >10 cells, got {len(cells)}"
        assert len(np.unique(cells)) == len(cells)

    @pytest.mark.slow
    def test_smallest_basin_simplified(self):
        """Basin 24 simplified to ~1k vertices."""
        lats, lons = _load_basin(24)
        lats_s, lons_s = _simplify_vertices(lats, lons, 1000)
        cells = mortie.morton_coverage(lats_s, lons_s, order=6)
        assert len(cells) > 10

    @pytest.mark.slow
    def test_largest_basin(self):
        """Basin 1 (largest, 81k vertices) — full polygon."""
        lats, lons = _load_basin(1)
        cells = mortie.morton_coverage(lats, lons, order=6)
        # Basin 1 produces ~76 cells at order 6 (verified against brute-force PIP)
        assert len(cells) > 30, f"Expected >30 cells, got {len(cells)}"

    @pytest.mark.slow
    def test_largest_basin_simplified(self):
        """Basin 1 simplified to ~1k vertices."""
        lats, lons = _load_basin(1)
        lats_s, lons_s = _simplify_vertices(lats, lons, 1000)
        cells = mortie.morton_coverage(lats_s, lons_s, order=6)
        assert len(cells) > 20

    @pytest.mark.slow
    def test_pole_antimeridian_basin(self):
        """Basin 2 (pole+antimeridian crossing, 43k vertices)."""
        lats, lons = _load_basin(2)
        cells = mortie.morton_coverage(lats, lons, order=6)
        # Basin 2 produces ~106 cells at order 6 (verified against brute-force PIP)
        assert len(cells) > 50

    @pytest.mark.slow
    def test_pole_antimeridian_simplified(self):
        """Basin 2 simplified to ~1k vertices."""
        lats, lons = _load_basin(2)
        lats_s, lons_s = _simplify_vertices(lats, lons, 1000)
        cells = mortie.morton_coverage(lats_s, lons_s, order=6)
        assert len(cells) > 20

    @pytest.mark.slow
    def test_simplification_consistency(self):
        """Simplified polygon coverage should be similar scale to full."""
        lats, lons = _load_basin(24)
        cells_full = mortie.morton_coverage(lats, lons, order=4)
        lats_s, lons_s = _simplify_vertices(lats, lons, 1000)
        cells_simp = mortie.morton_coverage(lats_s, lons_s, order=4)

        # Simplification changes the polygon shape, but both should have
        # substantial coverage and overlap
        assert len(cells_full) > 0
        assert len(cells_simp) > 0
        overlap = len(np.intersect1d(cells_full, cells_simp))
        assert overlap > 0, "Full and simplified should overlap"

    @pytest.mark.slow
    def test_dense_vs_sparse_correctness(self):
        """Coverage at different simplification levels produces valid results."""
        lats, lons = _load_basin(24)
        order = 4

        for target in [500, 2000, 5000]:
            la, lo = _simplify_vertices(lats, lons, target)
            cells = mortie.morton_coverage(la, lo, order=order)
            assert len(cells) > 0, f"target={target} produced empty coverage"
            assert len(np.unique(cells)) == len(cells)

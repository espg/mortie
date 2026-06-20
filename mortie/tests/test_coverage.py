"""Tests for polygon-to-morton coverage (morton_coverage)."""

import json

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

    def test_holes_not_supported(self):
        """Polygon with holes is not yet supported."""
        pytest.skip("Hole detection not yet implemented — see issue #20")

    def test_nan_raises(self):
        """NaN in coordinates raises ValueError."""
        with pytest.raises(ValueError, match="NaN or infinity"):
            mortie.morton_coverage([40.0, float("nan"), 45.0],
                                  [-120.0, -120.0, -110.0], order=6)

    def test_inf_raises(self):
        """Infinity in coordinates raises ValueError."""
        with pytest.raises(ValueError, match="NaN or infinity"):
            mortie.morton_coverage([40.0, 50.0, 45.0],
                                  [-120.0, float("inf"), -110.0], order=6)

    def test_southern_hemisphere(self):
        """Southern hemisphere polygon produces bit-63-set morton words."""
        lats = [-70.0, -80.0, -75.0]
        lons = [30.0, 30.0, 50.0]
        cells = mortie.morton_coverage(lats, lons, order=6)
        assert np.any(cells >= np.uint64(1) << np.uint64(63))

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
            mortie.morton_coverage([0, 1, 2], [0, 1, 2], order=30)

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


class TestCoverageDeterminism:
    """Regression tests for issue #28.

    A thin near-polar longitude strip used to return one of two different
    cell sets at random, because Phase B classified whole buffer components
    by sampling cells in (randomized) ``HashSet`` order and majority-voting.
    Coverage must be a pure function of its inputs and must fill the polygon
    interior rather than collapsing to the boundary ring.
    """

    # The exact polygon from issue #28 (closing vertex omitted; the wrapper
    # strips it anyway).
    POLAR_LATS = np.array([-89.0, -59.09804617, -59.09804617, -89.0])
    POLAR_LONS = np.array([105.5108378, 105.5108378, 106.5108378, 106.5108378])

    def test_polar_polygon_deterministic(self):
        """Repeated identical calls must return identical cell sets."""
        first = frozenset(
            int(x) for x in mortie.morton_coverage(
                self.POLAR_LATS, self.POLAR_LONS, order=10)
        )
        for _ in range(30):
            again = frozenset(
                int(x) for x in mortie.morton_coverage(
                    self.POLAR_LATS, self.POLAR_LONS, order=10)
            )
            assert again == first, "morton_coverage is not deterministic"

    def test_polar_polygon_fills_interior(self):
        """Result must include the interior, not just the boundary ring.

        The buggy boundary-only result was 1166 cells; the correct filled
        result is ~3074.  A threshold well above the boundary-only size
        guards against regressing to the dropped-interior behaviour.
        """
        cells = mortie.morton_coverage(self.POLAR_LATS, self.POLAR_LONS, order=10)
        assert len(cells) > 2000, (
            f"expected filled interior, got only {len(cells)} cells "
            "(regressed to boundary-only?)"
        )

    def test_polar_polygon_no_interior_holes(self):
        """Every cell whose centre is inside the polygon must be covered."""
        order = 10
        cells = mortie.morton_coverage(self.POLAR_LATS, self.POLAR_LONS, order=order)
        coverage_set = set(int(x) for x in cells)

        # Sample a grid of interior points (the strip spans lon ~105.5-106.5,
        # lat -89..-59) and require each one's cell to be covered.
        checked = 0
        for lat in np.arange(-88.0, -60.0, 1.0):
            for lon in np.arange(105.6, 106.5, 0.1):
                if _simple_pip(lat, lon, self.POLAR_LATS, self.POLAR_LONS):
                    m = mortie.geo2mort(lat, lon, order=order)
                    m_val = int(m) if np.ndim(m) == 0 else int(m[0])
                    assert m_val in coverage_set, (
                        f"Interior point ({lat}, {lon}) -> {m_val} "
                        "missing from coverage (hole)"
                    )
                    checked += 1
        assert checked > 0, "test sampled no interior points"


class TestCoveragePolarBoundary:
    """Regression tests for issue #32 (near-pole boundary under-coverage).

    HEALPix cell edges are *not* great-circle arcs: near the poles the true
    cell bulges outside the 4-corner geodesic quad.  The descent's straddle
    test was corners-only, so a polygon that grazed only that bulge (cell
    centre outside the polygon) was judged non-overlapping and the cell was
    pruned at a coarse level — dropping it from coverage at *every* order and
    violating the "includes all boundary cells" contract.  The fix densifies
    the cell boundary along the true HEALPix edge in the straddle test.

    The cases are real ATL06 cycle-22 near-pole granules that both S2
    (spherely) and EPSG:3031 shapely report as intersecting the listed shard
    cell, but which mortie 0.7.0 missed.  Each granule's order-8 cover must
    share at least one cell with the shard's order-8 children.  (mortie's
    *polygon* edges are already geodesic, so this is not a rhumb/geodesic
    issue — it is mortie's own cell-boundary model.)
    """

    DATA = Path(__file__).resolve().parent / "data" / "issue32_polar_pairs.json"

    def _pairs(self):
        if not self.DATA.exists():
            pytest.skip("issue #32 polar-pair data not found")
        return json.loads(self.DATA.read_text())

    @staticmethod
    def _packed(legacy_key):
        """Convert a legacy-decimal shard key in the data file to a packed word.

        The issue #32 fixture pins shard keys in the retired decimal encoding;
        the one-way converter lands them on today's packed wire format (#48).
        """
        from mortie import _rustie
        return int(
            _rustie.rust_mi_from_legacy(
                np.ascontiguousarray([int(legacy_key)], dtype=np.int64)
            )[0]
        )

    def test_near_pole_boundary_cells_covered(self):
        """Every previously-missed granule must now cover its shard cell."""
        pairs = self._pairs()
        assert pairs, "no test pairs loaded"
        missed = []
        for p in pairs:
            lats = np.asarray(p["lats"])
            lons = np.asarray(p["lons"])
            cover = set(int(c) for c in mortie.morton_coverage(lats, lons, order=8))
            shard = self._packed(p["shard_key"])
            children = set(
                int(c) for c in mortie.generate_morton_children(shard, 8)
            )
            if not (cover & children):
                missed.append((p["granule"], p["shard_key"]))
        assert not missed, (
            f"{len(missed)}/{len(pairs)} near-pole granule/shard pairs still "
            f"missed (issue #32 regression): {missed}"
        )

    def test_pruned_parent_now_covered(self):
        """The order-6 ancestor (shard parent) was pruned entirely pre-fix."""
        pairs = self._pairs()
        p = next((q for q in pairs if q["shard_key"] == -6111131), None)
        if p is None:
            pytest.skip("representative shard -6111131 not in data")
        lats = np.asarray(p["lats"])
        lons = np.asarray(p["lons"])
        cover6 = set(int(c) for c in mortie.morton_coverage(lats, lons, order=6))
        assert self._packed(-6111131) in cover6, (
            "order-6 shard parent -6111131 pruned from coverage (issue #32): "
            "the descent must refine a coarse polar cell whose true (curved) "
            "boundary the polygon grazes"
        )


class TestCoverageHolesMultipart:
    """Native ring-set coverage: holes carved, multipart unioned (even-odd)."""

    def _g(self, lat, lon, order):
        m = mortie.geo2mort(lat, lon, order=order)
        return int(np.atleast_1d(m)[0])

    def test_donut_carves_hole(self):
        # 20deg outer box, centred 6deg hole, around (45, -120).
        outer_la, outer_lo = [35.0, 35.0, 55.0, 55.0], [-130.0, -110.0, -110.0, -130.0]
        hole_la, hole_lo = [42.0, 42.0, 48.0, 48.0], [-123.0, -117.0, -117.0, -123.0]
        cov = set(int(x) for x in mortie.morton_coverage(
            [outer_la, hole_la], [outer_lo, hole_lo], order=7))
        assert self._g(45, -120, 7) not in cov, "hole interior must be carved out"
        assert self._g(37, -120, 7) in cov, "annulus must be covered"

    def test_donut_moc_densifies_to_flat(self):
        outer_la, outer_lo = [35.0, 35.0, 55.0, 55.0], [-130.0, -110.0, -110.0, -130.0]
        hole_la, hole_lo = [42.0, 42.0, 48.0, 48.0], [-123.0, -117.0, -117.0, -123.0]
        flat = set(int(x) for x in mortie.morton_coverage(
            [outer_la, hole_la], [outer_lo, hole_lo], order=8))
        moc = mortie.morton_coverage_moc(
            [outer_la, hole_la], [outer_lo, hole_lo], order=8)
        dens = set(int(x) for x in mortie.moc_to_order(moc, 8))
        assert dens == flat, "donut MOC must densify to the exact flat cover"
        assert len(moc) < len(flat), "MOC should be more compact"

    def test_disjoint_multipart_equals_union(self):
        a = ([40.0, 50.0, 45.0], [-120.0, -120.0, -110.0])
        b = ([10.0, 20.0, 15.0], [-80.0, -80.0, -70.0])
        union = np.unique(np.concatenate([
            mortie.morton_coverage(*a, order=6),
            mortie.morton_coverage(*b, order=6),
        ]))
        multi = mortie.morton_coverage([a[0], b[0]], [a[1], b[1]], order=6)
        np.testing.assert_array_equal(multi, union)

    def test_multipart_with_hole(self):
        # part A is a donut; part B is a disjoint solid triangle.
        outer_la, outer_lo = [35.0, 35.0, 55.0, 55.0], [-130.0, -110.0, -110.0, -130.0]
        hole_la, hole_lo = [42.0, 42.0, 48.0, 48.0], [-123.0, -117.0, -117.0, -123.0]
        tri_la, tri_lo = [10.0, 20.0, 15.0], [-80.0, -80.0, -70.0]
        cov = set(int(x) for x in mortie.morton_coverage(
            [outer_la, hole_la, tri_la], [outer_lo, hole_lo, tri_lo], order=7))
        assert self._g(45, -120, 7) not in cov, "hole still carved with extra part"
        assert self._g(37, -120, 7) in cov, "donut annulus covered"
        assert self._g(15, -75, 7) in cov, "disjoint triangle covered"


class TestCoverageMOCApi:
    """Single-polygon MOC output, adaptive stops, and MOC helpers."""

    SQ_LATS = [40.0, 40.0, 50.0, 50.0]
    SQ_LONS = [-125.0, -115.0, -115.0, -125.0]

    def test_moc_densifies_to_flat(self):
        flat = set(int(x) for x in mortie.morton_coverage(self.SQ_LATS, self.SQ_LONS, order=8))
        moc = mortie.morton_coverage_moc(self.SQ_LATS, self.SQ_LONS, order=8)
        assert len(moc) < len(flat), "MOC should be compact"
        dens = set(int(x) for x in mortie.moc_to_order(moc, 8))
        assert dens == flat

    def test_moc_tolerance_coarsens(self):
        exact = mortie.morton_coverage_moc(self.SQ_LATS, self.SQ_LONS, order=10)
        tol = mortie.morton_coverage_moc(self.SQ_LATS, self.SQ_LONS, order=10, tolerance=1.0)
        assert 0 < len(tol) <= len(exact)
        # deterministic
        np.testing.assert_array_equal(
            tol, mortie.morton_coverage_moc(self.SQ_LATS, self.SQ_LONS, order=10, tolerance=1.0))

    def test_moc_budget_bounds_cells(self):
        cov = mortie.morton_coverage_moc(self.SQ_LATS, self.SQ_LONS, order=12, max_cells=200)
        assert 0 < len(cov) <= 200 + 4

    def test_moc_budget_too_low_warns(self):
        with pytest.warns(UserWarning):
            mortie.morton_coverage_moc(self.SQ_LATS, self.SQ_LONS, order=12, max_cells=2)

    def test_moc_tolerance_and_budget_mutually_exclusive(self):
        with pytest.raises(ValueError):
            mortie.morton_coverage_moc(self.SQ_LATS, self.SQ_LONS, order=8,
                                       tolerance=1.0, max_cells=100)

    def test_moc_invalid_order(self):
        with pytest.raises(ValueError):
            mortie.morton_coverage_moc(self.SQ_LATS, self.SQ_LONS, order=0)
        with pytest.raises(ValueError):
            mortie.morton_coverage_moc(self.SQ_LATS, self.SQ_LONS, order=30)

    def test_moc_too_few_vertices(self):
        with pytest.raises(ValueError):
            mortie.morton_coverage_moc([0.0, 1.0], [0.0, 1.0], order=8)

    def test_moc_nan_raises(self):
        with pytest.raises(ValueError):
            mortie.morton_coverage_moc([0.0, 1.0, np.nan], [0.0, 1.0, 2.0], order=8)

    def test_compress_moc_idempotent_and_lossless(self):
        flat = mortie.morton_coverage(self.SQ_LATS, self.SQ_LONS, order=8)
        comp = mortie.compress_moc(flat)
        assert len(comp) < len(flat)
        # idempotent
        np.testing.assert_array_equal(comp, mortie.compress_moc(comp))
        # lossless
        assert set(int(x) for x in mortie.moc_to_order(comp, 8)) == set(int(x) for x in flat)

    def test_moc_to_order_expands(self):
        moc = mortie.morton_coverage_moc(self.SQ_LATS, self.SQ_LONS, order=8)
        flat = mortie.morton_coverage(self.SQ_LATS, self.SQ_LONS, order=8)
        assert len(mortie.moc_to_order(moc, 8)) == len(flat)

    def test_multipart_mismatched_ring_count(self):
        with pytest.raises(ValueError):
            mortie.morton_coverage_moc([[0, 1, 2], [3, 4, 5]], [[0, 1, 2]], order=6)

    def test_multipart_ring_too_few_vertices(self):
        with pytest.raises(ValueError):
            mortie.morton_coverage([[0, 1], [3, 4, 5]], [[0, 1], [3, 4, 5]], order=6)

    def test_moc_mismatched_lengths(self):
        with pytest.raises(ValueError):
            mortie.morton_coverage_moc([0.0, 1.0, 2.0], [0.0, 1.0], order=8)

    def test_moc_closed_ring_stripped(self):
        la = [40.0, 40.0, 50.0, 50.0, 40.0]
        lo = [-125.0, -115.0, -115.0, -125.0, -125.0]
        cov = mortie.morton_coverage_moc(la, lo, order=8)
        assert len(cov) > 0


class TestCoverageHighOrder:
    """Order 19–29 coverage (issue #60).

    The polygon/linestring/MOC paths capped order at 18 while ``geo2mort`` and
    the rest of the packed-u64 kernel already reached 29; the cap was a stale
    ``MAX_DEPTH = 18`` constant, not a u64 limit.  These pin the lifted ceiling
    at the orders the old cap rejected.  A *flat* cover scales as ``4**order``
    along the boundary, so the order-29 flat case uses a sub-cell polygon (a
    handful of cells) and the larger covers use the compact MOC form.
    """

    # ~3 mm triangle: sub-cell even at order 29 (cell edge ≈ 1.2 cm), so the
    # flat cover stays tiny at any order.
    TINY_LATS = [38.9, 38.9 + 3e-8, 38.9 + 1.5e-8]
    TINY_LONS = [-76.55, -76.55, -76.55 + 3e-8]

    # zagg's footprint ring (englacial/zagg#92): child_order = 19 → MOC order
    # child_order + 3 = 22, which the old order-18 cap rejected.
    RING_LATS = [38.85, 38.85, 38.93, 38.93, 38.85]
    RING_LONS = [-76.62, -76.59, -76.59, -76.62, -76.62]

    @pytest.mark.parametrize("order", [19, 22, 29])
    def test_flat_subcell_high_order(self, order):
        """A sub-cell polygon's flat cover at order 19/22/29 is a handful of
        cells, each self-encoding the requested order."""
        cells = mortie.morton_coverage(self.TINY_LATS, self.TINY_LONS, order=order)
        assert 1 <= len(cells) <= 16
        _, got_order = mortie.mort2healpix(cells)
        assert int(np.max(np.atleast_1d(got_order))) == order

    @pytest.mark.parametrize("order", [19, 22, 29])
    def test_moc_high_order(self, order):
        """The compact MOC form covers a ring at orders the old cap rejected.

        This sub-cell-scale ring is tiny enough that the adaptive descent may
        stop just short of ``order`` (no boundary cell needs the finest split),
        so the MOC's depth is bounded by ``order`` but need not reach it.  The
        meaningful, order-independent invariant is that the cover is non-empty,
        never exceeds ``order``, and densifies losslessly back to the flat cover
        at ``order`` — the round-trip zagg relies on.
        """
        # a ~3 m ring keeps the order-29 flat cover small (~6e4 cells) so the
        # densify round-trip below stays cheap in the default suite.
        sr_lats = [38.9, 38.9, 38.90003, 38.90003, 38.9]
        sr_lons = [-76.55, -76.54997, -76.54997, -76.55, -76.55]
        moc = mortie.morton_coverage_moc(sr_lats, sr_lons, order=order)
        assert len(moc) > 0
        orders = mortie.infer_order_from_morton(moc)
        assert int(np.max(orders)) <= order
        # Lossless densify: the MOC expands to exactly the flat order cover.
        flat = set(int(x) for x in mortie.morton_coverage(sr_lats, sr_lons, order=order))
        dens = set(int(x) for x in mortie.moc_to_order(moc, order))
        assert dens == flat

    def test_zagg_child_order_plus_three(self):
        """englacial/zagg#92: a footprint MOC at child_order + 3 = 22 must build
        (the old cap rejected order 22) and reach order 22 on the boundary."""
        moc = mortie.morton_coverage_moc(self.RING_LATS, self.RING_LONS, order=22)
        assert len(moc) > 0
        assert int(np.max(mortie.infer_order_from_morton(moc))) == 22

    def test_moc_densifies_to_flat_order_19(self):
        """MOC at order 19 densifies losslessly to the flat cover."""
        flat = set(
            int(x) for x in mortie.morton_coverage(self.RING_LATS, self.RING_LONS, order=19)
        )
        moc = mortie.morton_coverage_moc(self.RING_LATS, self.RING_LONS, order=19)
        dens = set(int(x) for x in mortie.moc_to_order(moc, 19))
        assert dens == flat

    def test_linestring_high_order_29(self):
        """linestring_coverage reaches order 29 too."""
        cells = mortie.linestring_coverage([38.9, 38.9001], [-76.55, -76.5499], order=29)
        assert len(cells) > 0
        assert int(np.max(mortie.infer_order_from_morton(cells))) == 29

    def test_large_flat_cover_warns(self, monkeypatch):
        """A flat cover above the cell-count threshold warns and names the MOC
        alternative (issue #60, phase 4)."""
        from mortie import coverage

        monkeypatch.setattr(coverage, "_FLAT_COVER_WARN_THRESHOLD", 1000)
        with pytest.warns(UserWarning, match="morton_coverage_moc"):
            mortie.morton_coverage(self.RING_LATS, self.RING_LONS, order=16)

    def test_small_flat_cover_does_not_warn(self, recwarn):
        """An ordinary small cover stays silent."""
        mortie.morton_coverage(self.RING_LATS, self.RING_LONS, order=6)
        assert not [w for w in recwarn.list if issubclass(w.category, UserWarning)]

    def test_warn_threshold_is_strict_boundary(self, monkeypatch, recwarn):
        """The warn check is strict ``>``: a cover of *exactly* the threshold is
        silent, one cell over warns.  Pins the boundary so a future
        ``>``→``>=`` slip is caught (issue #60, phase 4)."""
        from mortie import coverage

        n = mortie.morton_coverage(self.RING_LATS, self.RING_LONS, order=6).size
        # Threshold == cover size: strict ``>`` means no warning.
        monkeypatch.setattr(coverage, "_FLAT_COVER_WARN_THRESHOLD", int(n))
        mortie.morton_coverage(self.RING_LATS, self.RING_LONS, order=6)
        assert not [w for w in recwarn.list if issubclass(w.category, UserWarning)]
        # Threshold one below the cover size: now it must warn.
        monkeypatch.setattr(coverage, "_FLAT_COVER_WARN_THRESHOLD", int(n) - 1)
        with pytest.warns(UserWarning, match="morton_coverage_moc"):
            mortie.morton_coverage(self.RING_LATS, self.RING_LONS, order=6)

    def test_large_flat_cover_warns_default_threshold(self):
        """At the real (un-patched) ~1M-cell threshold the warning fires.

        An order-21 cover of the zagg ring is 2_394_698 cells (> ``1<<20``),
        built in ~0.2 s / ~19 MB — cheap enough to run in the default suite so
        the *real* threshold is exercised in CI, not only the monkeypatched one.
        """
        cover = mortie.morton_coverage(self.RING_LATS, self.RING_LONS, order=21)
        assert cover.size > (1 << 20), "fixture must exceed the warn threshold"
        with pytest.warns(UserWarning, match="morton_coverage_moc"):
            mortie.morton_coverage(self.RING_LATS, self.RING_LONS, order=21)


class TestMOCSetOps:
    """BMOC-backed boolean set algebra: moc_or / moc_and / moc_minus (issue #50).

    `and`/`minus` are checked against a brute-force reference computed by
    densifying both covers to a common deep order with `moc_to_order`, doing the
    set op on the flat leaf sets, then re-compressing with `compress_moc`.
    """

    # Two overlapping mid-latitude squares, plus a disjoint southern square.
    A_LATS = [40.0, 40.0, 50.0, 50.0]
    A_LONS = [-125.0, -115.0, -115.0, -125.0]
    B_LATS = [45.0, 45.0, 55.0, 55.0]
    B_LONS = [-120.0, -110.0, -110.0, -120.0]
    S_LATS = [-50.0, -50.0, -40.0, -40.0]
    S_LONS = [10.0, 20.0, 20.0, 10.0]

    def _cover(self, lats, lons, order=8):
        return mortie.morton_coverage_moc(lats, lons, order=order)

    def _ref(self, a, b, order, op):
        la = set(int(x) for x in mortie.moc_to_order(a, order))
        lb = set(int(x) for x in mortie.moc_to_order(b, order))
        leaves = sorted(op(la, lb))
        if not leaves:
            return set()
        comp = mortie.compress_moc(np.asarray(leaves, dtype=np.uint64))
        return set(int(x) for x in comp)

    def test_or_equals_compress_concat(self):
        a = self._cover(self.A_LATS, self.A_LONS)
        b = self._cover(self.B_LATS, self.B_LONS)
        got = mortie.moc_or(a, b)
        concat = np.concatenate([np.asarray(a), np.asarray(b)])
        np.testing.assert_array_equal(got, mortie.compress_moc(concat))

    def test_and_brute_force(self):
        a = self._cover(self.A_LATS, self.A_LONS)
        b = self._cover(self.B_LATS, self.B_LONS)
        got = set(int(x) for x in mortie.moc_and(a, b))
        assert got == self._ref(a, b, 8, lambda x, y: x & y)
        assert len(got) > 0, "the squares overlap, intersection must be non-empty"

    def test_minus_brute_force(self):
        a = self._cover(self.A_LATS, self.A_LONS)
        b = self._cover(self.B_LATS, self.B_LONS)
        got = set(int(x) for x in mortie.moc_minus(a, b))
        assert got == self._ref(a, b, 8, lambda x, y: x - y)

    def test_disjoint(self):
        a = self._cover(self.A_LATS, self.A_LONS)
        s = self._cover(self.S_LATS, self.S_LONS)
        # disjoint: and empty, minus is a, or is the union of both
        assert len(mortie.moc_and(a, s)) == 0
        np.testing.assert_array_equal(mortie.moc_minus(a, s), mortie.compress_moc(a))
        concat = np.concatenate([np.asarray(a), np.asarray(s)])
        np.testing.assert_array_equal(mortie.moc_or(a, s), mortie.compress_moc(concat))

    def test_self_minus_empty(self):
        a = self._cover(self.A_LATS, self.A_LONS)
        assert len(mortie.moc_minus(a, a)) == 0
        # self-and / self-or are idempotent (== compressed self)
        np.testing.assert_array_equal(mortie.moc_and(a, a), mortie.compress_moc(a))
        np.testing.assert_array_equal(mortie.moc_or(a, a), mortie.compress_moc(a))

    def test_mixed_order(self):
        # a finer cover (order 9) against a coarser one (order 6) of the same box.
        a = self._cover(self.A_LATS, self.A_LONS, order=9)
        b = self._cover(self.A_LATS, self.A_LONS, order=6)
        # densify to the deeper order for the reference
        got_and = set(int(x) for x in mortie.moc_and(a, b))
        assert got_and == self._ref(a, b, 9, lambda x, y: x & y)
        got_minus = set(int(x) for x in mortie.moc_minus(a, b))
        assert got_minus == self._ref(a, b, 9, lambda x, y: x - y)

    def test_empty(self):
        a = self._cover(self.A_LATS, self.A_LONS)
        empty = np.array([], dtype=np.uint64)
        np.testing.assert_array_equal(mortie.moc_or(a, empty), mortie.compress_moc(a))
        np.testing.assert_array_equal(mortie.moc_or(empty, a), mortie.compress_moc(a))
        assert len(mortie.moc_and(a, empty)) == 0
        np.testing.assert_array_equal(mortie.moc_minus(a, empty), mortie.compress_moc(a))
        assert len(mortie.moc_minus(empty, a)) == 0

    def test_southern_hemisphere(self):
        # Two overlapping southern boxes → bit-63-set words; must round-trip.
        b_lats = [-50.0, -50.0, -42.0, -42.0]
        b_lons = [15.0, 25.0, 25.0, 15.0]
        a = self._cover(self.S_LATS, self.S_LONS)
        b = self._cover(b_lats, b_lons)
        assert np.all(np.asarray(a) >= np.uint64(1) << np.uint64(63)), (
            "southern cover must set bit 63 on every word"
        )
        got = set(int(x) for x in mortie.moc_and(a, b))
        assert got == self._ref(a, b, 8, lambda x, y: x & y)
        got_m = set(int(x) for x in mortie.moc_minus(a, b))
        assert got_m == self._ref(a, b, 8, lambda x, y: x - y)

    def test_xor_brute_force(self):
        a = self._cover(self.A_LATS, self.A_LONS)
        b = self._cover(self.B_LATS, self.B_LONS)
        got = set(int(x) for x in mortie.moc_xor(a, b))
        assert got == self._ref(a, b, 8, lambda x, y: x ^ y)
        # the squares overlap and differ, so the symmetric difference is
        # non-empty and shares no leaf with the intersection.
        assert len(got) > 0
        and_leaves = set(int(x) for x in mortie.moc_to_order(mortie.moc_and(a, b), 8))
        got_arr = np.asarray(list(got), dtype=np.uint64)
        xor_leaves = set(int(x) for x in mortie.moc_to_order(got_arr, 8))
        assert xor_leaves.isdisjoint(and_leaves)

    def test_xor_equals_or_minus_and(self):
        # a △ b == (a ∪ b) \ (a ∩ b).
        a = self._cover(self.A_LATS, self.A_LONS)
        b = self._cover(self.B_LATS, self.B_LONS)
        got = mortie.moc_xor(a, b)
        expected = mortie.moc_minus(mortie.moc_or(a, b), mortie.moc_and(a, b))
        np.testing.assert_array_equal(got, expected)

    def test_xor_self_and_empty(self):
        a = self._cover(self.A_LATS, self.A_LONS)
        # a △ a = ∅.
        assert len(mortie.moc_xor(a, a)) == 0
        # a △ ∅ = a, ∅ △ a = a.
        empty = np.array([], dtype=np.uint64)
        np.testing.assert_array_equal(mortie.moc_xor(a, empty), mortie.compress_moc(a))
        np.testing.assert_array_equal(mortie.moc_xor(empty, a), mortie.compress_moc(a))

    def test_xor_disjoint_equals_or(self):
        # Disjoint covers share nothing, so xor == or.
        a = self._cover(self.A_LATS, self.A_LONS)
        s = self._cover(self.S_LATS, self.S_LONS)
        np.testing.assert_array_equal(mortie.moc_xor(a, s), mortie.moc_or(a, s))

    def test_xor_order_zero_operand(self):
        # An order-0 whole-base-cell operand against a finer cover that has cells
        # inside and outside that base cell (the deferred #53 coverage gap). The
        # inside cells cancel against the base cell's coverage; the outside cells
        # survive.
        base0 = mortie.norm2mort(0, 0, 0)  # base cell 0 at order 0
        a = np.array([base0], dtype=np.uint64)
        b = self._cover(self.S_LATS, self.S_LONS)  # southern → a different base cell
        got = set(int(x) for x in mortie.moc_xor(a, b))
        assert got == self._ref(a, b, 8, lambda x, y: x ^ y)
        # b is wholly outside base cell 0, so xor == or == base0 ∪ b.
        np.testing.assert_array_equal(mortie.moc_xor(a, b), mortie.moc_or(a, b))

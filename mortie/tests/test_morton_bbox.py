"""
Unit tests for mortie.morton_bbox module.

Tests cover:
- compact() behavior: extending characteristic on uniform columns, branching on divergence
- split_children(): root grouping by sign/first-digit, coverage preservation
- max_depth limiting
- refine_bbox(): budget constraint, coverage preservation, area reduction
- _cell_area(): area formula correctness
- Edge cases: single element, all identical indices
"""

import pytest
import numpy as np

from mortie.morton_bbox import (
    MortonChild,
    split_children,
    split_children_geo,
    refine_bbox,
    refine_bbox_geo,
    refine_bbox_morton,
    _cell_area,
    _auto_max_depth,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _total_len(nodes):
    """Sum of .len across a list of MortonChild."""
    return sum(n.len for n in nodes)


def _collect_leaves(nodes):
    """Recursively collect leaf nodes (nchildren == 0)."""
    leaves = []
    for n in nodes:
        if n.nchildren == 0:
            leaves.append(n)
        else:
            leaves.extend(_collect_leaves(n.children))
    return leaves


# ---------------------------------------------------------------------------
# Tests: _compact via split_children
# ---------------------------------------------------------------------------

class TestCompact:
    """Test that _compact extends characteristic on shared columns and branches on divergence."""

    def test_identical_indices_no_branching(self):
        """All identical indices → single root, characteristic is the full string, no children."""
        arr = np.array([1234, 1234, 1234], dtype=np.int64)
        roots = split_children(arr, max_depth=None)
        assert len(roots) == 1
        assert roots[0].characteristic == "1234"
        assert roots[0].nchildren == 0
        assert roots[0].len == 3

    def test_shared_prefix_then_divergence(self):
        """Indices sharing a prefix diverge → characteristic covers shared part, then branches."""
        arr = np.array([1231, 1232, 1233], dtype=np.int64)
        roots = split_children(arr, max_depth=None)
        assert len(roots) == 1
        root = roots[0]
        # Shared prefix is "123", then diverges on the last digit
        assert root.characteristic == "123"
        assert root.nchildren == 3
        child_chars = sorted(c.characteristic for c in root.children)
        assert child_chars == ["1231", "1232", "1233"]

    def test_immediate_divergence(self):
        """Indices that differ at the first digit → separate roots."""
        arr = np.array([11, 21, 31], dtype=np.int64)
        roots = split_children(arr, max_depth=None)
        assert len(roots) == 3
        chars = sorted(r.characteristic for r in roots)
        assert chars == ["11", "21", "31"]

    def test_children_have_correct_len(self):
        """Each child's .len matches the count of indices it covers."""
        arr = np.array([1211, 1211, 1222, 1233, 1233, 1233], dtype=np.int64)
        roots = split_children(arr, max_depth=None)
        assert len(roots) == 1
        root = roots[0]
        # "12" is shared, then diverges on '1', '2', '3'
        assert root.characteristic == "12"
        child_lens = {c.characteristic: c.len for c in root.children}
        assert child_lens["1211"] == 2
        assert child_lens["1222"] == 1
        assert child_lens["1233"] == 3

    def test_nchildren_always_ge_2(self):
        """Every node with children has nchildren >= 2 (compact skips single-unique columns)."""
        arr = np.array([1111, 1112, 1121, 1122, 1211, 1212], dtype=np.int64)
        roots = split_children(arr, max_depth=None)

        def check(node):
            if node.nchildren > 0:
                assert node.nchildren >= 2
            for c in node.children:
                check(c)

        for r in roots:
            check(r)

    def test_children_share_char_array(self):
        """All nodes in the tree reference the same underlying char_array (no copies)."""
        arr = np.array([1111, 1112, 1211, 1212], dtype=np.int64)
        roots = split_children(arr, max_depth=None)

        def check_same_array(node, expected_id):
            assert id(node._char_array) == expected_id
            for c in node.children:
                check_same_array(c, expected_id)

        root_array_id = id(roots[0]._char_array)
        for r in roots:
            check_same_array(r, root_array_id)

    def test_non_compressible_raises(self):
        """MortonChild raises ValueError when masked rows differ in sign column."""
        arr = np.array([-111, 222], dtype=np.int64)
        # Build char_array with mixed sign column: '-' vs ' '
        str_arr = np.array(["-111", " 222"])
        char_array = np.array([[ch for ch in s] for s in str_arr])
        # Mask selects both rows, but column 0 has '-' and ' '
        mask = np.array([True, True])
        with pytest.raises(ValueError, match="not compressible"):
            MortonChild(char_array, mask, start_col=1, characteristic="1",
                        original_array=arr)


# ---------------------------------------------------------------------------
# Tests: split_children
# ---------------------------------------------------------------------------

class TestSplitChildren:
    """Test split_children root-level grouping and coverage."""

    def test_negative_indices_grouped_by_sign_and_digit(self):
        """Negative indices get '-' prefix in characteristic."""
        arr = np.array([-123, -124, -234], dtype=np.int64)
        roots = split_children(arr, max_depth=None)
        chars = sorted(r.characteristic for r in roots)
        # -123 and -124 share "-12", -234 is "-234"
        assert len(roots) == 2
        assert chars[0].startswith("-1")
        assert chars[1].startswith("-2")

    def test_mixed_sign_indices(self):
        """Positive and negative indices produce separate root groups."""
        arr = np.array([-111, -112, 111, 112], dtype=np.int64)
        roots = split_children(arr, max_depth=None)
        neg_roots = [r for r in roots if r.characteristic.startswith("-")]
        pos_roots = [r for r in roots if not r.characteristic.startswith("-")]
        assert len(neg_roots) >= 1
        assert len(pos_roots) >= 1

    def test_coverage_preserved(self):
        """Total .len across roots equals input length."""
        arr = np.array([-5112, -5121, -6131, -6132, -6133], dtype=np.int64)
        roots = split_children(arr, max_depth=None)
        assert _total_len(roots) == len(arr)

    def test_leaf_coverage_preserved(self):
        """Total .len across all leaves equals input length."""
        arr = np.array([1111, 1122, 1211, 1222, 2111, 2122], dtype=np.int64)
        roots = split_children(arr, max_depth=None)
        leaves = _collect_leaves(roots)
        assert _total_len(leaves) == len(arr)

    def test_single_element(self):
        """Single-element input → one root, no children, full characteristic."""
        arr = np.array([42], dtype=np.int64)
        roots = split_children(arr, max_depth=None)
        assert len(roots) == 1
        assert roots[0].characteristic == "42"
        assert roots[0].nchildren == 0
        assert roots[0].len == 1

    def test_empty_raises(self):
        """Empty array raises ValueError."""
        with pytest.raises(ValueError):
            split_children(np.array([], dtype=np.int64))

    def test_2d_raises(self):
        """2-D array raises ValueError."""
        with pytest.raises(ValueError):
            split_children(np.array([[1, 2], [3, 4]], dtype=np.int64))


# ---------------------------------------------------------------------------
# Tests: max_depth
# ---------------------------------------------------------------------------

class TestMaxDepth:
    """Test that max_depth limits branching depth."""

    def test_max_depth_zero_no_children(self):
        """max_depth=0 → roots have no children (branching suppressed)."""
        arr = np.array([1111, 1122, 1211, 1222], dtype=np.int64)
        roots = split_children(arr, max_depth=0)
        for r in roots:
            assert r.nchildren == 0

    def test_max_depth_limits_tree_depth(self):
        """Tree depth does not exceed max_depth."""
        arr = np.array([11111, 11112, 11121, 11211, 12111], dtype=np.int64)

        def max_tree_depth(node, depth=0):
            if node.nchildren == 0:
                return depth
            return max(max_tree_depth(c, depth + 1) for c in node.children)

        for md in [1, 2, 3]:
            roots = split_children(arr, max_depth=md)
            for r in roots:
                assert max_tree_depth(r) <= md

    def test_max_depth_preserves_coverage(self):
        """Coverage is preserved regardless of max_depth."""
        arr = np.array([1111, 1122, 1211, 2111, 2122], dtype=np.int64)
        for md in [0, 1, 2, None]:
            roots = split_children(arr, max_depth=md)
            assert _total_len(roots) == len(arr)


# ---------------------------------------------------------------------------
# Tests: _cell_area
# ---------------------------------------------------------------------------

class TestCellArea:
    """Test the area formula: 4^(-(ndigits-1))."""

    def test_single_digit(self):
        """1 digit → area = 1 (base cell)."""
        # Use values that differ at digit 2 so the root characteristic stays at 1 digit
        arr = np.array([11, 12, 21, 22], dtype=np.int64)
        roots = split_children(arr, max_depth=0)
        for r in roots:
            # characteristic is a single digit like "1" or "2" (compaction stops at divergence)
            assert len(r.characteristic) == 1
            assert _cell_area(r) == pytest.approx(1.0)

    def test_two_digits(self):
        """2 digits → area = 1/4."""
        arr = np.array([11, 12], dtype=np.int64)
        roots = split_children(arr, max_depth=None)
        # Both are 2-digit, single root "1" with children "11", "12"
        for child in roots[0].children:
            assert _cell_area(child) == pytest.approx(0.25)

    def test_negative_sign_stripped(self):
        """Negative sign is stripped before counting digits."""
        arr = np.array([-123, -124], dtype=np.int64)
        roots = split_children(arr, max_depth=None)
        root = roots[0]
        # characteristic is "-12", ndigits = 2, area = 1/4
        ndigits = len(root.characteristic.lstrip("-"))
        assert ndigits == 2
        assert _cell_area(root) == pytest.approx(0.25)

    def test_area_decreases_with_depth(self):
        """Deeper nodes have smaller area."""
        arr = np.array([11111, 11112, 11121, 11122], dtype=np.int64)
        roots = split_children(arr, max_depth=None)
        root = roots[0]
        root_area = _cell_area(root)
        for child in root.children:
            assert _cell_area(child) < root_area


# ---------------------------------------------------------------------------
# Tests: refine_bbox
# ---------------------------------------------------------------------------

class TestRefineBbox:
    """Test the greedy expansion algorithm."""

    def test_budget_respected(self):
        """Output never exceeds n_cells."""
        arr = np.array([1111, 1122, 1211, 1222, 2111, 2222], dtype=np.int64)
        roots = split_children(arr)
        for budget in [2, 3, 4, 6, 10]:
            refined = refine_bbox(roots, n_cells=budget)
            assert len(refined) <= budget

    def test_coverage_preserved(self):
        """Sum of .len across refined cells equals input length."""
        arr = np.array([1111, 1122, 1211, 1222, 2111, 2222], dtype=np.int64)
        roots = split_children(arr)
        for budget in [2, 4, 8]:
            refined = refine_bbox(roots, n_cells=budget)
            assert _total_len(refined) == len(arr)

    def test_area_decreases(self):
        """Refined set has equal or smaller total area than root set."""
        arr = np.array([1111, 1122, 1211, 1222, 2111, 2222], dtype=np.int64)
        roots = split_children(arr)
        root_area = sum(_cell_area(r) for r in roots)
        refined = refine_bbox(roots, n_cells=6)
        refined_area = sum(_cell_area(r) for r in refined)
        assert refined_area <= root_area

    def test_budget_equal_to_roots(self):
        """Budget equal to len(roots) → no expansion, returns roots unchanged."""
        arr = np.array([1111, 2222], dtype=np.int64)
        roots = split_children(arr, max_depth=None)
        refined = refine_bbox(roots, n_cells=len(roots))
        assert len(refined) == len(roots)
        for r, ref in zip(roots, refined):
            assert r.characteristic == ref.characteristic

    def test_budget_one_returns_roots_if_single(self):
        """Budget=1 with a single root returns it as-is."""
        arr = np.array([1231, 1232, 1233], dtype=np.int64)
        roots = split_children(arr)
        assert len(roots) == 1
        refined = refine_bbox(roots, n_cells=1)
        assert len(refined) == 1
        assert refined[0].characteristic == roots[0].characteristic

    def test_all_leaves_are_valid_morton_children(self):
        """Every element in the refined list is a MortonChild."""
        arr = np.array([-5112, -5121, -6131, -6132, -6133], dtype=np.int64)
        roots = split_children(arr)
        refined = refine_bbox(roots, n_cells=4)
        for node in refined:
            assert isinstance(node, MortonChild)
            assert hasattr(node, "characteristic")
            assert hasattr(node, "len")
            assert node.len > 0

    def test_no_expansion_when_leaves_only(self):
        """If all roots are leaves (no children), refine returns them unchanged."""
        arr = np.array([1111, 2222, 3333], dtype=np.int64)
        roots = split_children(arr, max_depth=None)
        # Each is a unique leaf
        for r in roots:
            assert r.nchildren == 0
        refined = refine_bbox(roots, n_cells=10)
        assert len(refined) == len(roots)


# ---------------------------------------------------------------------------
# Tests: mantissa_array
# ---------------------------------------------------------------------------

class TestMantissaArray:
    """Test that mantissa_array returns the correct subset of original indices."""

    def test_root_mantissa_covers_all(self):
        """Each root's mantissa_array contains the expected indices."""
        arr = np.array([111, 112, 211, 212], dtype=np.int64)
        roots = split_children(arr, max_depth=None)
        all_mantissa = np.sort(np.concatenate([r.mantissa_array for r in roots]))
        np.testing.assert_array_equal(all_mantissa, np.sort(arr))

    def test_child_mantissa_subset_of_parent(self):
        """Each child's mantissa is a subset of its parent's mantissa."""
        arr = np.array([1211, 1212, 1221, 1222], dtype=np.int64)
        roots = split_children(arr, max_depth=None)
        for root in roots:
            parent_set = set(root.mantissa_array)
            for child in root.children:
                child_set = set(child.mantissa_array)
                assert child_set.issubset(parent_set)


# ---------------------------------------------------------------------------
# Tests: Rust vs Python parity
# ---------------------------------------------------------------------------

class TestRustParity:
    """Compare Rust and Python split_children output."""

    @staticmethod
    def _collect_characteristics(nodes):
        """Recursively collect all characteristics in depth-first order."""
        result = []
        for n in nodes:
            result.append(n.characteristic)
            result.extend(TestRustParity._collect_characteristics(n.children))
        return result

    @staticmethod
    def _get_python_roots(arr, max_depth=4):
        """Run pure-Python split_children regardless of Rust availability."""
        from mortie.morton_bbox import _split_children_python
        return _split_children_python(arr, max_depth=max_depth)

    def _assert_parity(self, arr, max_depth=4):
        """Assert Rust and Python produce equivalent trees."""
        rust_roots = split_children(arr, max_depth=max_depth)
        python_roots = self._get_python_roots(arr, max_depth=max_depth)

        # Same number of roots
        assert len(rust_roots) == len(python_roots)

        # Same characteristics (recursively)
        rust_chars = self._collect_characteristics(rust_roots)
        python_chars = self._collect_characteristics(python_roots)
        assert sorted(rust_chars) == sorted(python_chars)

        # Same lens at root level
        rust_lens = sorted((r.characteristic, r.len) for r in rust_roots)
        python_lens = sorted((r.characteristic, r.len) for r in python_roots)
        assert rust_lens == python_lens

        # Same total coverage
        assert sum(r.len for r in rust_roots) == sum(r.len for r in python_roots)

    def test_parity_positive(self):
        arr = np.array([1231, 1232, 1233], dtype=np.int64)
        self._assert_parity(arr, max_depth=None)

    def test_parity_negative(self):
        arr = np.array([-5112, -5121, -6131, -6132, -6133], dtype=np.int64)
        self._assert_parity(arr, max_depth=None)

    def test_parity_mixed_sign(self):
        arr = np.array([-111, -112, 111, 112], dtype=np.int64)
        self._assert_parity(arr, max_depth=None)

    def test_parity_max_depth_0(self):
        arr = np.array([1111, 1122, 1211, 1222], dtype=np.int64)
        self._assert_parity(arr, max_depth=0)

    def test_parity_max_depth_2(self):
        arr = np.array([11111, 11112, 11121, 11211, 12111], dtype=np.int64)
        self._assert_parity(arr, max_depth=2)

    def test_parity_identical(self):
        arr = np.array([1234, 1234, 1234], dtype=np.int64)
        self._assert_parity(arr, max_depth=None)

    def test_parity_single(self):
        arr = np.array([42], dtype=np.int64)
        self._assert_parity(arr, max_depth=None)

    def test_rust_mantissa_array(self):
        """mantissa_array works from Rust-reconstructed nodes."""
        arr = np.array([111, 112, 211, 212], dtype=np.int64)
        roots = split_children(arr, max_depth=None)
        all_mantissa = np.sort(np.concatenate([r.mantissa_array for r in roots]))
        np.testing.assert_array_equal(all_mantissa, np.sort(arr))

    def test_parity_large_random(self):
        """Parity on a larger random dataset."""
        rng = np.random.default_rng(42)
        arr = rng.integers(-999999, 999999, size=500, dtype=np.int64)
        self._assert_parity(arr, max_depth=4)


# ---------------------------------------------------------------------------
# Tests: Convenience methods
# ---------------------------------------------------------------------------

class TestConvenienceMethods:
    """Test split_children_geo and refine_bbox_geo."""

    def test_split_children_geo_returns_morton_children(self):
        """split_children_geo returns a list of MortonChild."""
        lats = np.array([-75, -75, -70, -70])
        lons = np.array([-80, -70, -70, -80])
        roots = split_children_geo(lats, lons, order=6, max_depth=4)
        assert len(roots) >= 1
        for r in roots:
            assert isinstance(r, MortonChild)
            assert r.len > 0

    def test_split_children_geo_coverage(self):
        """Total .len covers all input points."""
        lats = np.array([-75, -75, -70, -70, -72])
        lons = np.array([-80, -70, -70, -80, -75])
        roots = split_children_geo(lats, lons, order=6, max_depth=4)
        assert _total_len(roots) == len(lats)

    def test_refine_bbox_geo_respects_budget(self):
        """refine_bbox_geo respects the n_cells budget."""
        lats = np.array([-75, -75, -70, -70])
        lons = np.array([-80, -70, -70, -80])
        for budget in [2, 4, 8]:
            refined = refine_bbox_geo(lats, lons, n_cells=budget, order=6, max_depth=4)
            assert len(refined) <= budget

    def test_refine_bbox_geo_coverage(self):
        """refine_bbox_geo preserves coverage."""
        lats = np.array([-75, -75, -70, -70, -72])
        lons = np.array([-80, -70, -70, -80, -75])
        refined = refine_bbox_geo(lats, lons, n_cells=10, order=6, max_depth=4)
        assert _total_len(refined) == len(lats)

    def test_refine_bbox_geo_returns_morton_children(self):
        """All refined nodes are MortonChild."""
        lats = np.array([-75, -75, -70, -70])
        lons = np.array([-80, -70, -70, -80])
        refined = refine_bbox_geo(lats, lons, n_cells=8, order=6, max_depth=4)
        for r in refined:
            assert isinstance(r, MortonChild)


# ---------------------------------------------------------------------------
# Tests: _auto_max_depth optimization
# ---------------------------------------------------------------------------

class TestAutoMaxDepth:
    """Verify that _auto_max_depth produces sufficient trie depth.

    The formula is ceil(log2(n_cells)) + 1.  We test that:
    - The auto depth matches a deep reference (max_depth=10)
    - One level shallower (auto - 1) also matches (validates headroom)
    """

    @staticmethod
    def _get_refined_chars(morton_array, n_cells, max_depth):
        roots = split_children(morton_array, max_depth=max_depth)
        refined = refine_bbox(roots, n_cells=n_cells)
        return sorted(r.characteristic for r in refined)

    def test_formula_values(self):
        """Spot-check the formula for key n_cells values."""
        assert _auto_max_depth(4) == 3    # ceil(log2(4)) + 1 = 2 + 1
        assert _auto_max_depth(12) == 5   # ceil(log2(12)) + 1 = 4 + 1
        assert _auto_max_depth(1) == 1
        assert _auto_max_depth(2) == 2    # ceil(log2(2)) + 1 = 1 + 1

    # -- Empirical data (Antarctica P3 Chile) --

    @pytest.fixture
    def empirical_morton(self):
        """Morton indices from Antarctic flight-line parquet file."""
        import os
        parquet_path = os.path.join(
            os.path.dirname(__file__), '..', '..',
            '2002_Antarctica_P3chile_layers.parquet',
        )
        if not os.path.exists(parquet_path):
            pytest.skip("Parquet file not available")
        import pandas as pd
        import shapely
        df = pd.read_parquet(parquet_path)
        geoms = shapely.from_wkb(df['geometry'].values)
        from mortie import geo2mort
        return geo2mort(shapely.get_y(geoms), shapely.get_x(geoms), order=18)

    def test_empirical_n4_auto_matches_deep(self, empirical_morton):
        """n_cells=4: auto depth (3) matches deep reference (10)."""
        auto = self._get_refined_chars(empirical_morton, 4, _auto_max_depth(4))
        deep = self._get_refined_chars(empirical_morton, 4, 10)
        assert auto == deep

    def test_empirical_n4_shallow_matches_deep(self, empirical_morton):
        """n_cells=4: one level shallower (2) also matches deep reference."""
        shallow = self._get_refined_chars(empirical_morton, 4, _auto_max_depth(4) - 1)
        deep = self._get_refined_chars(empirical_morton, 4, 10)
        assert shallow == deep

    def test_empirical_n12_auto_matches_deep(self, empirical_morton):
        """n_cells=12: auto depth (5) matches deep reference (10)."""
        auto = self._get_refined_chars(empirical_morton, 12, _auto_max_depth(12))
        deep = self._get_refined_chars(empirical_morton, 12, 10)
        assert auto == deep

    def test_empirical_n12_shallow_matches_deep(self, empirical_morton):
        """n_cells=12: one level shallower (4) also matches deep reference."""
        shallow = self._get_refined_chars(empirical_morton, 12, _auto_max_depth(12) - 1)
        deep = self._get_refined_chars(empirical_morton, 12, 10)
        assert shallow == deep

    # -- Synthetic: clustered data (3 groups, ~8k points) --

    @pytest.fixture
    def clustered_morton(self):
        rng = np.random.default_rng(42)
        c1 = -5110000 + rng.integers(0, 9999, size=3000, dtype=np.int64)
        c2 = -6130000 + rng.integers(0, 9999, size=3000, dtype=np.int64)
        c3 = -5120000 + rng.integers(0, 9999, size=2000, dtype=np.int64)
        return np.concatenate([c1, c2, c3])

    def test_clustered_n4_auto_matches_deep(self, clustered_morton):
        auto = self._get_refined_chars(clustered_morton, 4, _auto_max_depth(4))
        deep = self._get_refined_chars(clustered_morton, 4, 10)
        assert auto == deep

    def test_clustered_n4_shallow_matches_deep(self, clustered_morton):
        shallow = self._get_refined_chars(clustered_morton, 4, _auto_max_depth(4) - 1)
        deep = self._get_refined_chars(clustered_morton, 4, 10)
        assert shallow == deep

    def test_clustered_n12_auto_matches_deep(self, clustered_morton):
        auto = self._get_refined_chars(clustered_morton, 12, _auto_max_depth(12))
        deep = self._get_refined_chars(clustered_morton, 12, 10)
        assert auto == deep

    def test_clustered_n12_shallow_matches_deep(self, clustered_morton):
        shallow = self._get_refined_chars(clustered_morton, 12, _auto_max_depth(12) - 1)
        deep = self._get_refined_chars(clustered_morton, 12, 10)
        assert shallow == deep

    # -- Synthetic: pathological binary splits --

    @pytest.fixture
    def pathological_morton(self):
        t1 = np.array([-5111111 + i for i in range(500)], dtype=np.int64)
        t2 = np.array([-5112222 + i for i in range(500)], dtype=np.int64)
        t3 = np.array([-6131111 + i for i in range(500)], dtype=np.int64)
        t4 = np.array([-6132222 + i for i in range(500)], dtype=np.int64)
        return np.concatenate([t1, t2, t3, t4])

    def test_pathological_n4_auto_matches_deep(self, pathological_morton):
        auto = self._get_refined_chars(pathological_morton, 4, _auto_max_depth(4))
        deep = self._get_refined_chars(pathological_morton, 4, 10)
        assert auto == deep

    def test_pathological_n4_shallow_matches_deep(self, pathological_morton):
        shallow = self._get_refined_chars(pathological_morton, 4, _auto_max_depth(4) - 1)
        deep = self._get_refined_chars(pathological_morton, 4, 10)
        assert shallow == deep

    def test_pathological_n12_auto_matches_deep(self, pathological_morton):
        auto = self._get_refined_chars(pathological_morton, 12, _auto_max_depth(12))
        deep = self._get_refined_chars(pathological_morton, 12, 10)
        assert auto == deep

    def test_pathological_n12_shallow_matches_deep(self, pathological_morton):
        shallow = self._get_refined_chars(pathological_morton, 12, _auto_max_depth(12) - 1)
        deep = self._get_refined_chars(pathological_morton, 12, 10)
        assert shallow == deep

    # -- refine_bbox_morton and refine_bbox_geo auto-depth --

    def test_refine_bbox_morton_auto_depth(self, clustered_morton):
        """refine_bbox_morton with default max_depth=None uses auto depth."""
        auto = refine_bbox_morton(clustered_morton, n_cells=4)
        explicit = refine_bbox_morton(clustered_morton, n_cells=4, max_depth=10)
        auto_chars = sorted(r.characteristic for r in auto)
        explicit_chars = sorted(r.characteristic for r in explicit)
        assert auto_chars == explicit_chars

    def test_refine_bbox_geo_auto_depth(self):
        """refine_bbox_geo with default max_depth=None uses auto depth."""
        lats = np.array([-75, -75, -70, -70])
        lons = np.array([-80, -70, -70, -80])
        auto = refine_bbox_geo(lats, lons, n_cells=4, order=6)
        explicit = refine_bbox_geo(lats, lons, n_cells=4, order=6, max_depth=10)
        auto_chars = sorted(r.characteristic for r in auto)
        explicit_chars = sorted(r.characteristic for r in explicit)
        assert auto_chars == explicit_chars

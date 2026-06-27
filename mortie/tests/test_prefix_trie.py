"""
Unit tests for mortie.prefix_trie module.

Tests cover:
- compact() behavior: extending characteristic on uniform columns, branching on divergence
- split_children(): root grouping by sign/first-digit, coverage preservation
- max_depth limiting
- morton_polygon(): budget constraint, coverage preservation, area reduction
- _cell_area(): area formula correctness
- Edge cases: single element, all identical indices
"""

import hashlib
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from mortie import _rustie
from mortie.prefix_trie import (
    MortonChild,
    _auto_max_depth,
    geo_morton_polygon,
    morton_polygon,
    morton_polygon_from_array,
    split_children,
    split_children_geo,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
#
# The bare-i64 "morton" wire format is now a packed-u64 bit layout (stored
# bit-reinterpreted as i64), not the legacy decimal encoding.  The trie still
# branches on each word's DECODE-THROUGH-KERNEL decimal repr (via
# ``to_decimal_repr``), which reproduces the legacy digit-string structure
# exactly, so characteristic strings are unchanged in form -- only the integer
# values fed in must be packed words.  These helpers convert legacy-style
# decimal mortons (e.g. 1234, -5111131) to today's packed words via the
# shipped one-way converter, so the hardcoded test literals keep their meaning.


def _packed(legacy):
    """Convert a legacy decimal morton int (e.g. 1234) to today's packed word."""
    return int(_rustie.rust_mi_from_legacy(
        np.ascontiguousarray([int(legacy)], dtype=np.int64))[0])


def _packed_arr(legacies):
    """Convert an iterable of legacy decimal mortons to a packed int64 array."""
    return np.ascontiguousarray(
        _rustie.rust_mi_from_legacy(
            np.ascontiguousarray(list(legacies), dtype=np.int64)),
        dtype=np.int64)


def _packed_nested(nested_ids, depth=6):
    """Pack HEALPix NESTED ids at ``depth`` into packed morton words.

    Used to build large clusters of *valid* morton words without relying on
    decimal arithmetic (which can produce malformed legacy digit strings).
    """
    return np.ascontiguousarray(
        _rustie.rust_mi_from_nested(
            np.ascontiguousarray(list(nested_ids), dtype=np.uint64), depth),
        dtype=np.int64)


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
        arr = _packed_arr([1234, 1234, 1234])
        roots = split_children(arr, max_depth=None)
        assert len(roots) == 1
        assert roots[0].characteristic == "1234"
        assert roots[0].nchildren == 0
        assert roots[0].len == 3

    def test_shared_prefix_then_divergence(self):
        """Indices sharing a prefix diverge → characteristic covers shared part, then branches."""
        arr = _packed_arr([1231, 1232, 1233])
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
        arr = _packed_arr([11, 21, 31])
        roots = split_children(arr, max_depth=None)
        assert len(roots) == 3
        chars = sorted(r.characteristic for r in roots)
        assert chars == ["11", "21", "31"]

    def test_children_have_correct_len(self):
        """Each child's .len matches the count of indices it covers."""
        arr = _packed_arr([1211, 1211, 1222, 1233, 1233, 1233])
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
        arr = _packed_arr([1111, 1112, 1121, 1122, 1211, 1212])
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
        arr = _packed_arr([1111, 1112, 1211, 1212])
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
        arr = _packed_arr([-123, -124, -234])
        roots = split_children(arr, max_depth=None)
        chars = sorted(r.characteristic for r in roots)
        # -123 and -124 share "-12", -234 is "-234"
        assert len(roots) == 2
        assert chars[0].startswith("-1")
        assert chars[1].startswith("-2")

    def test_mixed_sign_indices(self):
        """Positive and negative indices produce separate root groups."""
        arr = _packed_arr([-111, -112, 111, 112])
        roots = split_children(arr, max_depth=None)
        neg_roots = [r for r in roots if r.characteristic.startswith("-")]
        pos_roots = [r for r in roots if not r.characteristic.startswith("-")]
        assert len(neg_roots) >= 1
        assert len(pos_roots) >= 1

    def test_coverage_preserved(self):
        """Total .len across roots equals input length."""
        arr = _packed_arr([-5112, -5121, -6131, -6132, -6133])
        roots = split_children(arr, max_depth=None)
        assert _total_len(roots) == len(arr)

    def test_leaf_coverage_preserved(self):
        """Total .len across all leaves equals input length."""
        arr = _packed_arr([1111, 1122, 1211, 1222, 2111, 2122])
        roots = split_children(arr, max_depth=None)
        leaves = _collect_leaves(roots)
        assert _total_len(leaves) == len(arr)

    def test_single_element(self):
        """Single-element input → one root, no children, full characteristic."""
        arr = _packed_arr([42])
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
        arr = _packed_arr([1111, 1122, 1211, 1222])
        roots = split_children(arr, max_depth=0)
        for r in roots:
            assert r.nchildren == 0

    def test_max_depth_limits_tree_depth(self):
        """Tree depth does not exceed max_depth."""
        arr = _packed_arr([11111, 11112, 11121, 11211, 12111])

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
        arr = _packed_arr([1111, 1122, 1211, 2111, 2122])
        for md in [0, 1, 2, None]:
            roots = split_children(arr, max_depth=md)
            assert _total_len(roots) == len(arr)


# ---------------------------------------------------------------------------
# Tests: MortonChild.cell_area
# ---------------------------------------------------------------------------

class TestCellArea:
    """Test the cached area property: 4^(-(ndigits-1))."""

    def test_single_digit(self):
        """1 digit → area = 1 (base cell)."""
        # Use values that differ at digit 2 so the root characteristic stays at 1 digit
        arr = _packed_arr([11, 12, 21, 22])
        roots = split_children(arr, max_depth=0)
        for r in roots:
            # characteristic is a single digit like "1" or "2" (compaction stops at divergence)
            assert len(r.characteristic) == 1
            assert r.cell_area == pytest.approx(1.0)

    def test_two_digits(self):
        """2 digits → area = 1/4."""
        arr = _packed_arr([11, 12])
        roots = split_children(arr, max_depth=None)
        # Both are 2-digit, single root "1" with children "11", "12"
        for child in roots[0].children:
            assert child.cell_area == pytest.approx(0.25)

    def test_negative_sign_stripped(self):
        """Negative sign is stripped before counting digits."""
        arr = _packed_arr([-123, -124])
        roots = split_children(arr, max_depth=None)
        root = roots[0]
        # characteristic is "-12", ndigits = 2, area = 1/4
        ndigits = len(root.characteristic.lstrip("-"))
        assert ndigits == 2
        assert root.cell_area == pytest.approx(0.25)

    def test_area_decreases_with_depth(self):
        """Deeper nodes have smaller area."""
        arr = _packed_arr([11111, 11112, 11121, 11122])
        roots = split_children(arr, max_depth=None)
        root = roots[0]
        root_area = root.cell_area
        for child in root.children:
            assert child.cell_area < root_area

    def test_cell_area_is_cached(self):
        """Repeated access returns the same cached value."""
        arr = _packed_arr([11, 12])
        root = split_children(arr, max_depth=None)[0]
        first = root.cell_area
        assert root.cell_area is first or root.cell_area == first
        # mutating characteristic must not change the cached area
        root.characteristic = root.characteristic + "9"
        assert root.cell_area == first


# ---------------------------------------------------------------------------
# Tests: morton_polygon
# ---------------------------------------------------------------------------

class TestMortonPolygon:
    """Test the greedy expansion algorithm."""

    def test_budget_respected(self):
        """Output never exceeds n_cells."""
        arr = _packed_arr([1111, 1122, 1211, 1222, 2111, 2222])
        roots = split_children(arr)
        for budget in [2, 3, 4, 6, 10]:
            refined = morton_polygon(roots, n_cells=budget)
            assert len(refined) <= budget

    def test_coverage_preserved(self):
        """Sum of .len across refined cells equals input length."""
        arr = _packed_arr([1111, 1122, 1211, 1222, 2111, 2222])
        roots = split_children(arr)
        for budget in [2, 4, 8]:
            refined = morton_polygon(roots, n_cells=budget)
            assert _total_len(refined) == len(arr)

    def test_area_decreases(self):
        """Refined set has equal or smaller total area than root set."""
        arr = _packed_arr([1111, 1122, 1211, 1222, 2111, 2222])
        roots = split_children(arr)
        root_area = sum(r.cell_area for r in roots)
        refined = morton_polygon(roots, n_cells=6)
        refined_area = sum(r.cell_area for r in refined)
        assert refined_area <= root_area

    def test_budget_equal_to_roots(self):
        """Budget equal to len(roots) → no expansion, returns roots unchanged."""
        arr = _packed_arr([1111, 2222])
        roots = split_children(arr, max_depth=None)
        refined = morton_polygon(roots, n_cells=len(roots))
        assert len(refined) == len(roots)
        for r, ref in zip(roots, refined):
            assert r.characteristic == ref.characteristic

    def test_budget_one_returns_roots_if_single(self):
        """Budget=1 with a single root returns it as-is."""
        arr = _packed_arr([1231, 1232, 1233])
        roots = split_children(arr)
        assert len(roots) == 1
        refined = morton_polygon(roots, n_cells=1)
        assert len(refined) == 1
        assert refined[0].characteristic == roots[0].characteristic

    def test_all_leaves_are_valid_morton_children(self):
        """Every element in the refined list is a MortonChild."""
        arr = _packed_arr([-5112, -5121, -6131, -6132, -6133])
        roots = split_children(arr)
        refined = morton_polygon(roots, n_cells=4)
        for node in refined:
            assert isinstance(node, MortonChild)
            assert hasattr(node, "characteristic")
            assert hasattr(node, "len")
            assert node.len > 0

    def test_no_expansion_when_leaves_only(self):
        """If all roots are leaves (no children), refine returns them unchanged."""
        arr = _packed_arr([1111, 2222, 3333])
        roots = split_children(arr, max_depth=None)
        # Each is a unique leaf
        for r in roots:
            assert r.nchildren == 0
        refined = morton_polygon(roots, n_cells=10)
        assert len(refined) == len(roots)


# ---------------------------------------------------------------------------
# Tests: morton_polygon tie-break determinism (issue #83)
#
# The heap is keyed by ``(-efficiency, seq, node)`` with a unique monotonic
# ``seq`` (mortie/prefix_trie.py), so ties are resolved by insertion order — a
# total order with no hash/dict/set dependence.  These pin that determinism:
# a deliberately symmetric, tie-heavy input where several sibling subtrees have
# identical area/efficiency, so the *budget cutoff falls in the middle of a tie*
# and the seq tiebreak decides which symmetric subtree expands.
# ---------------------------------------------------------------------------

# Four identical-shape subtrees under base cells 1 and 2: every node at a given
# depth has the same area, so efficiencies tie across all siblings.
_TIE_HEAVY = [
    1111, 1112, 1121, 1122, 1211, 1212, 1221, 1222,
    2111, 2112, 2121, 2122, 2211, 2212, 2221, 2222,
]


class TestMortonPolygonDeterminism:
    """Pin ``morton_polygon`` tie-break determinism (issue #83)."""

    def _chars(self, n_cells):
        roots = split_children(_packed_arr(_TIE_HEAVY))
        return [n.characteristic for n in morton_polygon(roots, n_cells=n_cells)]

    def test_repeatable_within_process(self):
        """Many calls on a tie-rich input return a byte-identical cell set."""
        # Budget 6 cuts through a tie: only some of the symmetric subtrees can
        # expand, so the seq tiebreak picks which -- the exposed decision.
        first = self._chars(6)
        for _ in range(50):
            assert self._chars(6) == first
        # Several budgets, each rebuilt from scratch so node identity differs.
        for budget in [4, 6, 8, 10, 12]:
            assert self._chars(budget) == self._chars(budget)

    def test_stable_across_pythonhashseed(self):
        """Result is independent of PYTHONHASHSEED (no set/dict ordering leak).

        Run in subprocesses with different hash seeds; the emitted characteristic
        list must be identical.  This is the one real run-to-run exposure if
        hash-ordered containers were ever reintroduced on this path.
        """
        script = textwrap.dedent(
            """
            import numpy as np
            from mortie import _rustie
            from mortie.prefix_trie import split_children, morton_polygon
            legacies = [
                1111, 1112, 1121, 1122, 1211, 1212, 1221, 1222,
                2111, 2112, 2121, 2122, 2211, 2212, 2221, 2222,
            ]
            arr = np.ascontiguousarray(
                _rustie.rust_mi_from_legacy(
                    np.ascontiguousarray(legacies, dtype=np.int64)),
                dtype=np.int64)
            roots = split_children(arr)
            refined = morton_polygon(roots, n_cells=6)
            print(",".join(n.characteristic for n in refined))
            """
        )
        outputs = set()
        for seed in ("0", "1", "12345", "99999"):
            env = {**os.environ, "PYTHONHASHSEED": seed}
            res = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, env=env, check=True)
            outputs.add(res.stdout.strip())
        assert len(outputs) == 1, f"hash-seed-dependent output: {outputs}"

    def test_golden_tie_break(self):
        """Pinned golden for the symmetric tie-heavy input at budget 6.

        Captured from the (deterministic) Rust+Python path; if the tie-break
        order ever changes, this golden flags it.
        """
        chars = self._chars(6)
        assert chars == [
            "111", "112", "121", "122", "21", "22",
        ]
        assert hashlib.sha256("\n".join(chars).encode()).hexdigest() == (
            "f87e4659200f68b46cb9d68cbb48d65018228ca26d5c5fa4811c8ca774ee276d"
        )

    def test_golden_tie_break_deeper(self):
        """Second golden at budget 10 — a deeper mid-tie cut.

        Budget 6 resolves a tie at the depth-1 level; budget 10 resolves one a
        level deeper (the ``1111/1112/...`` frontier), so this guards the
        tie-break order at a deeper cut too.
        """
        chars = self._chars(10)
        assert chars == [
            "1111", "1112", "1121", "1122", "121", "122",
            "211", "212", "221", "222",
        ]
        assert hashlib.sha256("\n".join(chars).encode()).hexdigest() == (
            "3ec1d2bcb813fee598fa61c1c821a3a509a27e7cd593a3d8b88701fea82de515"
        )


# ---------------------------------------------------------------------------
# Tests: mantissa_array
# ---------------------------------------------------------------------------

class TestMantissaArray:
    """Test that mantissa_array returns the correct subset of original indices."""

    def test_root_mantissa_covers_all(self):
        """Each root's mantissa_array contains the expected indices."""
        arr = _packed_arr([111, 112, 211, 212])
        roots = split_children(arr, max_depth=None)
        all_mantissa = np.sort(np.concatenate([r.mantissa_array for r in roots]))
        np.testing.assert_array_equal(all_mantissa, np.sort(arr))

    def test_child_mantissa_subset_of_parent(self):
        """Each child's mantissa is a subset of its parent's mantissa."""
        arr = _packed_arr([1211, 1212, 1221, 1222])
        roots = split_children(arr, max_depth=None)
        for root in roots:
            parent_set = set(root.mantissa_array)
            for child in root.children:
                child_set = set(child.mantissa_array)
                assert child_set.issubset(parent_set)


# ---------------------------------------------------------------------------
# Tests: golden split_children output
#
# The pure-Python split_children twin was removed (issue #37).  These tests
# pin the Rust trie shape against golden values captured from the Rust path:
# the recursively-collected characteristics, the root-level (characteristic,
# len) pairs, and the total covered count.
# ---------------------------------------------------------------------------

class TestRustGolden:
    """Pin Rust split_children output against captured golden values."""

    @staticmethod
    def _collect_characteristics(nodes):
        """Recursively collect all characteristics in depth-first order."""
        result = []
        for n in nodes:
            result.append(n.characteristic)
            result.extend(TestRustGolden._collect_characteristics(n.children))
        return result

    def _assert_golden(self, arr, max_depth, chars, rootlens, total):
        """Assert the Rust trie matches the golden shape."""
        roots = split_children(arr, max_depth=max_depth)
        assert sorted(self._collect_characteristics(roots)) == sorted(chars)
        assert sorted((r.characteristic, r.len) for r in roots) == sorted(rootlens)
        assert sum(r.len for r in roots) == total

    def test_golden_positive(self):
        arr = _packed_arr([1231, 1232, 1233])
        self._assert_golden(
            arr, None,
            ['123', '1231', '1232', '1233'],
            [('123', 3)], 3,
        )

    def test_golden_negative(self):
        arr = _packed_arr([-5112, -5121, -6131, -6132, -6133])
        self._assert_golden(
            arr, None,
            ['-51', '-5112', '-5121', '-613', '-6131', '-6132', '-6133'],
            [('-51', 2), ('-613', 3)], 5,
        )

    def test_golden_mixed_sign(self):
        arr = _packed_arr([-111, -112, 111, 112])
        self._assert_golden(
            arr, None,
            ['-11', '-111', '-112', '11', '111', '112'],
            [('-11', 2), ('11', 2)], 4,
        )

    def test_golden_max_depth_0(self):
        arr = _packed_arr([1111, 1122, 1211, 1222])
        self._assert_golden(arr, 0, ['1'], [('1', 4)], 4)

    def test_golden_max_depth_2(self):
        arr = _packed_arr([11111, 11112, 11121, 11211, 12111])
        self._assert_golden(
            arr, 2,
            ['1', '11', '111', '11211', '12111'],
            [('1', 5)], 5,
        )

    def test_golden_identical(self):
        arr = _packed_arr([1234, 1234, 1234])
        self._assert_golden(arr, None, ['1234'], [('1234', 3)], 3)

    def test_golden_single(self):
        arr = _packed_arr([42])
        self._assert_golden(arr, None, ['42'], [('42', 1)], 1)

    def test_rust_mantissa_array(self):
        """mantissa_array works from Rust-reconstructed nodes."""
        arr = _packed_arr([111, 112, 211, 212])
        roots = split_children(arr, max_depth=None)
        all_mantissa = np.sort(np.concatenate([r.mantissa_array for r in roots]))
        np.testing.assert_array_equal(all_mantissa, np.sort(arr))

    def test_golden_large_random(self):
        """Golden shape on a larger random dataset (500 indices, seed 42).

        The legacy version drew raw decimal ints, which are no longer valid
        packed mortons.  We instead draw 500 random valid HEALPix NESTED ids
        (depth 6, spanning all 12 base cells) and pack them, then pin the
        resulting trie shape -- still a deterministic golden, captured from the
        Rust path on this seeded input.
        """
        rng = np.random.default_rng(42)
        nested = rng.integers(0, 12 * (4 ** 6), size=500, dtype=np.uint64)
        arr = _packed_nested(nested)
        roots = split_children(arr, max_depth=4)

        assert len(roots) == 12
        assert sum(r.len for r in roots) == 500

        chars = sorted(self._collect_characteristics(roots))
        assert len(chars) == 774
        assert hashlib.sha256("\n".join(chars).encode()).hexdigest() == (
            "85d539a2839f6b8ffe48984c2a6ef6776da478dc799cf75231dc403510717203"
        )

        rootlens = sorted((r.characteristic, r.len) for r in roots)
        assert hashlib.sha256(repr(rootlens).encode()).hexdigest() == (
            "6736ded34c2d1a76e523a961bd6791105771d771a0ba63e1dc127fe6b78e25e0"
        )


# ---------------------------------------------------------------------------
# Tests: Convenience methods
# ---------------------------------------------------------------------------

class TestConvenienceMethods:
    """Test split_children_geo and geo_morton_polygon."""

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

    def test_geo_morton_polygon_respects_budget(self):
        """geo_morton_polygon respects the n_cells budget."""
        lats = np.array([-75, -75, -70, -70])
        lons = np.array([-80, -70, -70, -80])
        for budget in [2, 4, 8]:
            refined = geo_morton_polygon(lats, lons, n_cells=budget, order=6, max_depth=4)
            assert len(refined) <= budget

    def test_geo_morton_polygon_coverage(self):
        """geo_morton_polygon preserves coverage."""
        lats = np.array([-75, -75, -70, -70, -72])
        lons = np.array([-80, -70, -70, -80, -75])
        refined = geo_morton_polygon(lats, lons, n_cells=10, order=6, max_depth=4)
        assert _total_len(refined) == len(lats)

    def test_geo_morton_polygon_returns_morton_children(self):

        """All refined nodes are MortonChild."""
        lats = np.array([-75, -75, -70, -70])
        lons = np.array([-80, -70, -70, -80])
        refined = geo_morton_polygon(lats, lons, n_cells=8, order=6, max_depth=4)
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
        refined = morton_polygon(roots, n_cells=n_cells)
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
        # Three clusters (~8k points) in distinct trie regions, two of which
        # share a base cell so the root-level grouping is non-trivial.  Built
        # from valid NESTED ids (depth 6) rather than decimal arithmetic, which
        # would yield malformed legacy mortons.  The fixed leading NESTED
        # orders reproduce the legacy prefixes -511 / -613 / -512: base cell 10
        # decodes to "-5", base cell 11 to "-6", and each leading NESTED digit
        # d shows up as decimal digit d+1.
        rng = np.random.default_rng(42)
        base, b5, b4 = 4 ** 6, 4 ** 5, 4 ** 4

        def cluster(base_cell, o1, o2, n):
            tail = rng.integers(0, b4, size=n, dtype=np.uint64)
            return base_cell * base + o1 * b5 + o2 * b4 + tail

        c1 = cluster(10, 0, 0, 3000)  # -511...
        c2 = cluster(11, 0, 2, 3000)  # -613...
        c3 = cluster(10, 0, 1, 2000)  # -512...
        return _packed_nested(np.concatenate([c1, c2, c3]))

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
        # Four tight clusters (500 each) forming a binary-split structure: base
        # cell 10 ("-5") splits into two sub-branches, base cell 11 ("-6") into
        # two more.  Built from valid NESTED ids (depth 6); 500 consecutive ids
        # stay within one leading NESTED order (4^5 = 1024), so each cluster
        # keeps a distinct two-digit prefix (-51/-52/-61/-62).
        base, b5 = 4 ** 6, 4 ** 5
        t1 = 10 * base + 0 * b5 + np.arange(500, dtype=np.uint64)  # -51...
        t2 = 10 * base + 1 * b5 + np.arange(500, dtype=np.uint64)  # -52...
        t3 = 11 * base + 0 * b5 + np.arange(500, dtype=np.uint64)  # -61...
        t4 = 11 * base + 1 * b5 + np.arange(500, dtype=np.uint64)  # -62...
        return _packed_nested(np.concatenate([t1, t2, t3, t4]))

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

    # -- morton_polygon_from_array and geo_morton_polygon auto-depth --

    def test_morton_polygon_from_array_auto_depth(self, clustered_morton):
        """morton_polygon_from_array with default max_depth=None uses auto depth."""
        auto = morton_polygon_from_array(clustered_morton, n_cells=4)
        explicit = morton_polygon_from_array(clustered_morton, n_cells=4, max_depth=10)
        auto_chars = sorted(r.characteristic for r in auto)
        explicit_chars = sorted(r.characteristic for r in explicit)
        assert auto_chars == explicit_chars

    def test_geo_morton_polygon_auto_depth(self):
        """geo_morton_polygon with default max_depth=None uses auto depth."""
        lats = np.array([-75, -75, -70, -70])
        lons = np.array([-80, -70, -70, -80])
        auto = geo_morton_polygon(lats, lons, n_cells=4, order=6)
        explicit = geo_morton_polygon(lats, lons, n_cells=4, order=6, max_depth=10)
        auto_chars = sorted(r.characteristic for r in auto)
        explicit_chars = sorted(r.characteristic for r in explicit)
        assert auto_chars == explicit_chars

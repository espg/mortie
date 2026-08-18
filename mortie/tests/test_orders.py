"""
Comprehensive unit tests for mortie.orders -- order query, change and validate

Split out of test_tools.py alongside the mortie.tools split itself (issue #159):
these are the classes whose subject is an order -- the resolution ladder,
UNIQ order decoding, coarsening and refinement.  The conversion classes moved to
test_convert.py.  Class bodies are unchanged.

These tests establish reference behavior for all morton indexing functions.
They will be used to verify that any refactoring (e.g., removing numba)
produces identical outputs.

Key constraints:
- Morton indices use base-4 encoding (digits 1-4) after the base cell identifier
- Not all integers are valid morton indices
- Tests focus on consistency, determinism, and structural validation
"""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mortie import convert
from mortie import orders as orders_mod


def _sphere_res(order):
    """Reference RMS cell spacing (km): sqrt of the equal-area cell area."""
    R = orders_mod.EARTH_RADIUS_KM
    return math.sqrt(4 * math.pi * R**2 / (12 * 4**order))


class TestOrder2Res:
    """Test order to resolution conversion"""

    def test_order2res_basic(self):
        """Test basic order to resolution calculations"""
        # Order 0 is the RMS cell spacing on the unified HEALPix sphere.
        res0 = orders_mod.order2res(0)
        assert_allclose(res0, _sphere_res(0), rtol=1e-10)

        # Each order halves the cell scale (area drops by 4).
        res1 = orders_mod.order2res(1)
        assert_allclose(res1, res0 / 2.0, rtol=1e-10)

    def test_order2res_range(self):
        """Test full range of valid orders against the sphere formula"""
        for order in range(20):
            res = orders_mod.order2res(order)
            assert_allclose(res, _sphere_res(order), rtol=1e-10)

    def test_order2res_decreasing(self):
        """Test that resolution decreases with order"""
        resolutions = [orders_mod.order2res(i) for i in range(10)]
        # Each resolution should be smaller than the previous
        assert all(resolutions[i] > resolutions[i+1] for i in range(len(resolutions)-1))


class TestRes2Display:
    """Test the resolution ladder (returns records; issue #68)"""

    def test_returns_one_record_per_order_through_max(self):
        """One record per order 0..MAX_ORDER -- no silent drop above 19"""
        levels = orders_mod.res2display()
        assert len(levels) == orders_mod.MAX_ORDER + 1
        assert [lvl.order for lvl in levels] == list(range(orders_mod.MAX_ORDER + 1))

    def test_returns_data_not_none(self, capsys):
        """The ladder is returned, and nothing is printed (issue #68)"""
        levels = orders_mod.res2display(max_order=3)
        assert capsys.readouterr().out == ""
        assert isinstance(levels, list)
        assert levels[0]._fields == ("order", "value", "unit", "km")

    def test_max_order_argument(self):
        """max_order bounds the range inclusively"""
        levels = orders_mod.res2display(max_order=5)
        assert len(levels) == 6
        assert levels[-1].order == 5

    def test_unit_ladder_km_m_cm(self):
        """Coarse orders read in km, sub-km in m, sub-m in cm"""
        by_order = {lvl.order: lvl for lvl in orders_mod.res2display()}
        # order 12 = 1.592 km, order 13 = 795.852 m (unified sphere, issue #119)
        assert (by_order[12].value, by_order[12].unit) == (1.592, 'km')
        assert (by_order[13].value, by_order[13].unit) == (795.852, 'm')
        # finest orders drop to cm rather than tiny km/m fractions
        assert (by_order[25].value, by_order[25].unit) == (19.43, 'cm')
        assert (by_order[29].value, by_order[29].unit) == (1.214, 'cm')

    def test_rounds_within_bracket(self):
        """Values are rounded to three decimals inside the chosen unit"""
        for lvl in orders_mod.res2display():
            assert round(lvl.value, 3) == lvl.value

    def test_km_field_is_unrounded_order2res(self):
        """The km field is the raw resolution, for callers doing arithmetic"""
        for lvl in orders_mod.res2display(max_order=6):
            assert lvl.km == orders_mod.order2res(lvl.order)

    def test_out_of_range_raises(self):
        """max_order outside 0..MAX_ORDER is rejected"""
        for bad in (-1, orders_mod.MAX_ORDER + 1):
            with pytest.raises(ValueError, match="max_order must be"):
                orders_mod.res2display(max_order=bad)


def _uniq_at(order, nest):
    """UNIQ cell number(s) for NESTED index/indices at ``order``."""
    return 4 * (4**order) + np.asarray(nest, dtype=np.int64)


class TestUniqOrders:
    """Test the per-element order decode underpinning the UNIQ decoders"""

    def test_decodes_every_order(self):
        """First, last and an interior UNIQ value decode to their own order"""
        for order in range(orders_mod.MAX_ORDER + 1):
            npix = 12 * (4**order)
            values = _uniq_at(order, [0, npix // 2, npix - 1])
            assert_array_equal(orders_mod.orders_of_uniq(values),
                               np.full(3, order, dtype=np.int64))

    def test_order_boundaries_are_exact(self):
        """The last value of order k and the first of k+1 are adjacent ints

        The retired ``log2(uniq / 4) // 2`` decode went through float64, which
        cannot separate these above 2**53 (issue #136).
        """
        for order in range(orders_mod.MAX_ORDER):
            last = 4 * (4**order) + 12 * (4**order) - 1
            first = 4 * (4 ** (order + 1))
            assert first == last + 1
            assert_array_equal(orders_mod.orders_of_uniq([last, first]),
                               np.array([order, order + 1], dtype=np.int64))

    def test_out_of_range_raises(self):
        """Values below the order-0 floor or above the MAX_ORDER ceiling"""
        for bad in (0, 3, 4 ** (orders_mod.MAX_ORDER + 2)):
            with pytest.raises(ValueError, match="valid UNIQ"):
                orders_mod.orders_of_uniq([bad])


class TestClip2Order:
    """Test resolution clipping (kernel coarsen)."""

    def test_clip2order_rejects_removed_print_factor(self):
        """The order-18-anchored print_factor flag is gone (issue #68).

        It returned ``18 - clip_order``, which went negative for the
        order-19..29 words this package now encodes. Pinned so the flag
        cannot quietly return.
        """
        with pytest.raises(TypeError):
            orders_mod.clip2order(12, print_factor=True)

    def test_clip2order_requires_words(self):
        """midx is now required -- there is no word-less call form."""
        with pytest.raises(TypeError):
            orders_mod.clip2order(12)

    def test_clip2order_clipping(self):
        """Clipping coarsens packed words to the target order."""
        # Two order-18 packed words.
        morton18 = np.array(
            [int(convert.norm2mort(12345, 2, 18)),
             int(convert.norm2mort(54321, 4, 18))],
            dtype=np.uint64,
        )
        morton12 = orders_mod.clip2order(12, morton18)
        # The coarsened words decode to order 12 and the same base cells.
        _, parent, order = convert.mort2norm(morton12)
        assert order == 12
        np.testing.assert_array_equal(parent, [2, 4])
        # Coarsening == re-encoding the order-18 cell's order-12 ancestor.
        n18, p18, _ = convert.mort2norm(morton18)
        expected = np.array(
            [int(convert.norm2mort(int(n) >> (2 * 6), int(p), 12))
             for n, p in zip(n18, p18)],
            dtype=np.uint64,
        )
        np.testing.assert_array_equal(morton12, expected)

    def test_clip2order_negative_indices(self):
        """Clipping a southern (bit-63-set) word keeps it southern."""
        bit63 = np.uint64(1) << np.uint64(63)
        morton18 = np.array(
            [int(convert.norm2mort(100, 2, 18)), int(convert.norm2mort(200, 9, 18))],
            dtype=np.uint64,
        )
        morton12 = orders_mod.clip2order(12, morton18)
        # Base cell 9 sets bit 63 -> stays set; base 2 stays clear.
        assert morton18[0] < bit63 and morton12[0] < bit63
        assert morton18[1] >= bit63 and morton12[1] >= bit63

    def test_clip2order_deterministic(self):
        """Test determinism"""
        morton18 = np.array(
            [int(convert.norm2mort(100, 2, 18)), int(convert.norm2mort(200, 9, 18))],
            dtype=np.uint64,
        )
        result1 = orders_mod.clip2order(12, morton18)
        result2 = orders_mod.clip2order(12, morton18)
        assert_array_equal(result1, result2)


class TestGenerateMortonChildren:
    """generate_morton_children: NESTED-space descent, packed words (issue #48)."""

    def _parent(self, normed, base, order):
        """A packed parent word for a given (normed, base, order)."""
        return int(convert.norm2mort(normed, base, order))

    def test_one_level_count_and_descent(self):
        """One level down yields 4 children; staying put yields the parent."""
        parent = self._parent(1234, base=11, order=6)  # southern base cell
        children = orders_mod.generate_morton_children(parent, target_order=7)
        assert len(children) == 4
        # Each child is order 7 and shares the parent's order-6 ancestor.
        _, _, order = convert.mort2norm(children)
        assert order == 7
        np.testing.assert_array_equal(
            orders_mod.clip2order(6, np.ascontiguousarray(children, dtype=np.uint64)),
            np.full(4, parent, dtype=np.uint64),
        )
        # Already at target order -> returns the parent unchanged.
        np.testing.assert_array_equal(
            orders_mod.generate_morton_children(parent, target_order=6),
            np.array([parent], dtype=np.uint64),
        )

    def test_two_levels_count_and_membership(self):
        """Descending 2 levels yields 16 children, all sharing the parent prefix."""
        parent = self._parent(420, base=2, order=5)
        children = orders_mod.generate_morton_children(parent, target_order=7)
        assert len(children) == 16
        # Each child coarsens back to the parent at order 5.
        np.testing.assert_array_equal(
            orders_mod.clip2order(5, np.ascontiguousarray(children, dtype=np.uint64)),
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
        children = orders_mod.generate_morton_children(parent, target_order=8)
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
                    orders_mod.generate_morton_children(parent, target),
                    reference(parent, target),
                )

    def test_target_below_parent_raises(self):
        parent = self._parent(7, base=8, order=6)
        with pytest.raises(ValueError):
            orders_mod.generate_morton_children(parent, target_order=3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

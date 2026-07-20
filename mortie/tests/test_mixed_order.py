"""Mixed-order support in the geo kernels (issue #116).

Covers the per-element introspection surface ``orders_of`` / ``is_point``:
pure-numpy suffix decodes of the packed word per the spec page's §1 suffix
table (``docs/specification.md``), golden-pinned against the table itself —
including the 28/47 preorder band boundaries and both hemispheres — and
cross-checked against the Rust kernel's depth decode so the two cannot drift.
"""

import numpy as np

from mortie import _rustie, geo2mort, is_point, orders_of

# One northern and one southern location: base cells 7-11 set bit 63, so the
# southern column exercises the large-unsigned half of the word space.
_LATS = np.array([45.0, -45.0])
_LONS = np.array([45.0, -120.0])


def _area(order):
    return geo2mort(_LATS, _LONS, order=order)


def _point():
    return geo2mort(_LATS, _LONS, points=True)


class TestOrdersOf:
    """Golden pins for the spec §1 suffix table."""

    def test_suffix_0_to_27_order_is_suffix(self):
        """Orders 0-27 store the order as the suffix value itself (spec §1)."""
        for order in range(28):
            words = _area(order)
            assert np.all((words & np.uint64(0x3F)) == order)
            np.testing.assert_array_equal(orders_of(words), order)

    def test_suffix_28_to_47_preorder_band(self):
        """The 28..=47 band is parent-first preorder r = t28*5 + (t29+1 | 0):
        each t28 owns a 5-block — the order-28 parent slot, then its four
        order-29 children (spec §1)."""
        # Craft every band suffix onto a real order-29 prefix+body (the body
        # is fully populated for suffixes >= 28, so every value is canonical).
        for base_word in _area(29):
            stem = np.uint64(base_word) & ~np.uint64(0x3F)
            for suffix in range(28, 48):
                word = stem | np.uint64(suffix)
                r = suffix - 28
                expected = 28 if r % 5 == 0 else 29
                assert orders_of(word)[0] == expected, (
                    f"suffix {suffix}: expected order {expected}"
                )
                assert not is_point(word)[0]

    def test_suffix_48_to_63_point_band(self):
        """Suffixes 48..=63 are order-29 points, max-encoded (spec §1)."""
        for base_word in _area(29):
            stem = np.uint64(base_word) & ~np.uint64(0x3F)
            for suffix in range(48, 64):
                word = stem | np.uint64(suffix)
                assert orders_of(word)[0] == 29
                assert is_point(word)[0]

    def test_matches_kernel_depth_decode(self):
        """orders_of agrees with the Rust kernel's depth decode elementwise —
        the numpy suffix table and the kernel cannot drift apart."""
        words = np.concatenate(
            [_area(o) for o in (0, 1, 6, 13, 19, 24, 27, 28, 29)] + [_point()]
        )
        _, depths = _rustie.rust_mort2nested(np.ascontiguousarray(words))
        np.testing.assert_array_equal(orders_of(words), depths)

    def test_shape_family_and_dtype(self):
        """Scalar in -> length-1 uint8 ndarray (the geo2mort shape family)."""
        got = orders_of(int(_area(6)[0]))
        assert got.shape == (1,) and got.dtype == np.uint8 and got[0] == 6
        assert orders_of(np.array([], dtype=np.uint64)).shape == (0,)


class TestIsPoint:
    def test_point_words_are_points(self):
        np.testing.assert_array_equal(is_point(_point()), True)

    def test_area_words_are_not_points(self):
        """Area words at every order — order-29 areas included — are not
        points: kind rides the suffix band, not the order (spec §4)."""
        for order in (0, 6, 28, 29):
            np.testing.assert_array_equal(is_point(_area(order)), False)

    def test_shape_family(self):
        got = is_point(int(_point()[0]))
        assert got.shape == (1,) and got.dtype == np.bool_ and got[0]

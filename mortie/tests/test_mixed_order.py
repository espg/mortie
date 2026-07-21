"""Mixed-order support in the geo kernels (issue #116).

Covers the per-element introspection surface ``orders_of`` / ``is_point``:
pure-numpy suffix decodes of the packed word per the spec page's §1 suffix
table (``docs/specification.md``), golden-pinned against the table itself —
including the 28/47 preorder band boundaries and both hemispheres — and
cross-checked against the Rust kernel's depth decode so the two cannot drift.
"""

import numpy as np
import pytest

from mortie import (
    _rustie,
    geo2mort,
    infer_order_from_morton,
    is_point,
    mort2bbox,
    mort2geo,
    mort2polygon,
    order2res,
    orders_of,
)

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


class TestInferOrderMixedRaise:
    """infer_order_from_morton returns one scalar order; mixed input raises
    (issue #116 — it previously returned the first element's order silently)."""

    def test_uniform_scalar_and_array_unchanged(self):
        for order in (0, 6, 29):
            words = _area(order)
            assert infer_order_from_morton(int(words[0])) == order
            assert infer_order_from_morton(words) == order
        assert infer_order_from_morton(_point()) == 29

    def test_mixed_raises_naming_orders(self):
        mixed = np.concatenate([_area(6), _area(19)])
        with pytest.raises(ValueError, match=r"Mixed orders.*\[6, 19\].*orders_of"):
            infer_order_from_morton(mixed)

    def test_mixed_area_point_is_uniform_29(self):
        """Order-29 areas and points share order 29 — kind is not order, so
        this is NOT a mixed-order array (spec §4)."""
        assert infer_order_from_morton(np.concatenate([_area(29), _point()])) == 29


_MIXED_ORDERS = (0, 6, 19, 24, 29)


def _mixed_area(rng_seed=42):
    """Interleaved mixed-order area words spanning both hemispheres."""
    words = np.concatenate([_area(o) for o in _MIXED_ORDERS])
    np.random.default_rng(rng_seed).shuffle(words)
    return words


class TestGroupDispatchOracle:
    """The correctness oracle (issue #116): group-by-order dispatch must equal
    the per-order uniform kernel calls, scattered back to input positions."""

    def test_mort2geo_matches_uniform(self):
        words = _mixed_area()
        orders = orders_of(words)
        lat, lon = mort2geo(words)
        assert lat.shape == lon.shape == words.shape
        for order in np.unique(orders):
            mask = orders == order
            ulat, ulon = mort2geo(np.ascontiguousarray(words[mask]))
            np.testing.assert_array_equal(lat[mask], ulat)
            np.testing.assert_array_equal(lon[mask], ulon)

    def test_mort2geo_points_group_with_29(self):
        """Point words dispatch through the order-29 group: locations match
        the uniform all-point call (a point's location IS its mort2geo)."""
        words = np.concatenate([_area(6), _point(), _area(19)])
        lat, lon = mort2geo(words)
        plat, plon = mort2geo(_point())
        np.testing.assert_array_equal(lat[2:4], plat)
        np.testing.assert_array_equal(lon[2:4], plon)

    def test_mort2bbox_matches_uniform(self):
        words = _mixed_area()
        orders = orders_of(words)
        bboxes = mort2bbox(words)
        assert len(bboxes) == words.size
        for order in np.unique(orders):
            (idx,) = np.nonzero(orders == order)
            uniform = mort2bbox(np.ascontiguousarray(words[idx]))
            assert [bboxes[i] for i in idx] == uniform

    def test_mort2polygon_matches_uniform(self):
        for step in (1, 2):
            words = _mixed_area()
            orders = orders_of(words)
            polygons = mort2polygon(words, step=step)
            assert len(polygons) == words.size
            # Rings are 4*step + 1 vertices at every order (closure included).
            assert all(len(p) == 4 * step + 1 for p in polygons)
            for order in np.unique(orders):
                (idx,) = np.nonzero(orders == order)
                uniform = mort2polygon(np.ascontiguousarray(words[idx]), step=step)
                assert [polygons[i] for i in idx] == uniform

    def test_singleton_group_scatter(self):
        """A group of exactly one element exercises the bare-dict/bare-ring
        return of the length-1 uniform call; scatter must still place it."""
        words = np.concatenate([_area(6), _area(19)[:1]])
        bboxes = mort2bbox(words)
        assert bboxes[2] == mort2bbox(int(words[2]))
        polygons = mort2polygon(words)
        assert polygons[2] == mort2polygon(int(words[2]))
        lat, lon = mort2geo(words)
        slat, slon = mort2geo(int(words[2]))
        assert lat[2] == slat[0] and lon[2] == slon[0]

    def test_uniform_input_unchanged(self):
        """Uniform arrays never enter the dispatch branch: results are the
        pre-#116 uniform kernel outputs (scalar/array postures alike)."""
        words = _area(6)
        lat, lon = mort2geo(words)
        slat, slon = mort2geo(int(words[0]))
        assert lat[0] == slat[0] and lon[0] == slon[0]
        assert mort2bbox(words)[0] == mort2bbox(int(words[0]))
        assert mort2polygon(words)[0] == mort2polygon(int(words[0]))


class TestPointPostures:
    """mort2polygon/mort2bbox on point words yield the geometry of the point's
    containing order-29 cell (issue #116, espg review): a point still has a
    well-defined containing box/polygon — the cell that contains it — and a
    group of points covers a well-defined area element by element. The oracle
    is the same location's order-29 AREA word: a point and the order-29 area
    word at that location decode to the identical HEALPix cell (spec §4)."""

    def test_bbox_points_yield_containing_cell(self):
        """A point's bbox equals the bbox of its containing order-29 cell (the
        order-29 area word at the same location)."""
        assert mort2bbox(_point()) == mort2bbox(_area(29))

    def test_polygon_points_yield_containing_cell(self):
        assert mort2polygon(_point()) == mort2polygon(_area(29))

    def test_scalar_point_yields_containing_cell(self):
        point_word = int(_point()[0])
        area_word = int(_area(29)[0])
        assert mort2bbox(point_word) == mort2bbox(area_word)
        assert mort2polygon(point_word) == mort2polygon(area_word)

    def test_mixed_area_point_scatters(self):
        """Points mixed with area words scatter correctly through the
        group-by-order dispatch: each point element yields its containing
        order-29 cell geometry at its input position."""
        words = np.concatenate([_area(6), _point()])
        bboxes = mort2bbox(words)
        assert len(bboxes) == words.size
        # The order-6 area words keep their uniform bboxes...
        assert bboxes[:2] == mort2bbox(_area(6))
        # ...and the trailing points scatter to their containing-cell bboxes.
        assert bboxes[2:] == mort2bbox(_area(29))

        polygons = mort2polygon(words)
        assert len(polygons) == words.size
        assert polygons[:2] == mort2polygon(_area(6))
        assert polygons[2:] == mort2polygon(_area(29))

    def test_order29_area_words_still_work(self):
        """Order-29 AREA words are genuine cells (spec §4) — points now group
        with them and share their geometry."""
        assert len(mort2bbox(_area(29))) == 2
        assert len(mort2polygon(_area(29))) == 2


class TestResolutionCrossCheck:
    def test_orders_of_indexes_order2res(self):
        """The per-element order indexes the resolution ladder exactly as the
        encoding order does (spec §3)."""
        words = np.concatenate([_area(o) for o in _MIXED_ORDERS])
        expected = np.repeat([order2res(o) for o in _MIXED_ORDERS], len(_LATS))
        np.testing.assert_allclose(order2res(orders_of(words)), expected, rtol=1e-12)

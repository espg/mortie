"""
Comprehensive unit tests for mortie.tools module

These tests establish reference behavior for all morton indexing functions.
They will be used to verify that any refactoring (e.g., removing numba)
produces identical outputs.

Key constraints:
- Morton indices use base-4 encoding (digits 1-4) after the base cell identifier
- Not all integers are valid morton indices
- Tests focus on consistency, determinism, and structural validation
"""

import math

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from mortie import tools
from mortie import _healpix as hp


def _sphere_res(order):
    """Reference RMS cell spacing (km): sqrt of the equal-area cell area."""
    R = tools.EARTH_RADIUS_KM
    return math.sqrt(4 * math.pi * R**2 / (12 * 4**order))


class TestOrder2Res:
    """Test order to resolution conversion"""

    def test_order2res_basic(self):
        """Test basic order to resolution calculations"""
        # Order 0 is the RMS cell spacing on the unified HEALPix sphere.
        res0 = tools.order2res(0)
        assert_allclose(res0, _sphere_res(0), rtol=1e-10)

        # Each order halves the cell scale (area drops by 4).
        res1 = tools.order2res(1)
        assert_allclose(res1, res0 / 2.0, rtol=1e-10)

    def test_order2res_range(self):
        """Test full range of valid orders against the sphere formula"""
        for order in range(20):
            res = tools.order2res(order)
            assert_allclose(res, _sphere_res(order), rtol=1e-10)

    def test_order2res_decreasing(self):
        """Test that resolution decreases with order"""
        resolutions = [tools.order2res(i) for i in range(10)]
        # Each resolution should be smaller than the previous
        assert all(resolutions[i] > resolutions[i+1] for i in range(len(resolutions)-1))


class TestRes2Display:
    """Test the resolution ladder (returns records; issue #68)"""

    def test_returns_one_record_per_order_through_max(self):
        """One record per order 0..MAX_ORDER -- no silent drop above 19"""
        levels = tools.res2display()
        assert len(levels) == tools.MAX_ORDER + 1
        assert [lvl.order for lvl in levels] == list(range(tools.MAX_ORDER + 1))

    def test_returns_data_not_none(self, capsys):
        """The ladder is returned, and nothing is printed (issue #68)"""
        levels = tools.res2display(max_order=3)
        assert capsys.readouterr().out == ""
        assert isinstance(levels, list)
        assert levels[0]._fields == ("order", "value", "unit", "km")

    def test_max_order_argument(self):
        """max_order bounds the range inclusively"""
        levels = tools.res2display(max_order=5)
        assert len(levels) == 6
        assert levels[-1].order == 5

    def test_unit_ladder_km_m_cm(self):
        """Coarse orders read in km, sub-km in m, sub-m in cm"""
        by_order = {lvl.order: lvl for lvl in tools.res2display()}
        # order 12 = 1.592 km, order 13 = 795.852 m (unified sphere, issue #119)
        assert (by_order[12].value, by_order[12].unit) == (1.592, 'km')
        assert (by_order[13].value, by_order[13].unit) == (795.852, 'm')
        # finest orders drop to cm rather than tiny km/m fractions
        assert (by_order[25].value, by_order[25].unit) == (19.43, 'cm')
        assert (by_order[29].value, by_order[29].unit) == (1.214, 'cm')

    def test_rounds_within_bracket(self):
        """Values are rounded to three decimals inside the chosen unit"""
        for lvl in tools.res2display():
            assert round(lvl.value, 3) == lvl.value

    def test_km_field_is_unrounded_order2res(self):
        """The km field is the raw resolution, for callers doing arithmetic"""
        for lvl in tools.res2display(max_order=6):
            assert lvl.km == tools.order2res(lvl.order)

    def test_out_of_range_raises(self):
        """max_order outside 0..MAX_ORDER is rejected"""
        for bad in (-1, tools.MAX_ORDER + 1):
            with pytest.raises(ValueError, match="max_order must be"):
                tools.res2display(max_order=bad)


class TestUnique2Parent:
    """Test UNIQ to parent cell conversion"""

    def test_unique2parent_single_resolution(self):
        """Test parent extraction for single resolution"""
        # Create some UNIQ values at order 6
        order = 6
        nside = 2**order
        nest_indices = np.array([100, 200, 300, 400])
        uniq = 4 * (nside**2) + nest_indices

        parents = tools.unique2parent(uniq)

        # All parents should be in valid range (0-11 for HEALPix base cells)
        assert np.all(parents >= 0)
        assert np.all(parents < 12)

    def test_unique2parent_deterministic(self):
        """Test that same inputs give same outputs"""
        order = 8
        nside = 2**order
        nest_indices = np.array([1000, 2000, 3000])
        uniq = 4 * (nside**2) + nest_indices

        parents1 = tools.unique2parent(uniq)
        parents2 = tools.unique2parent(uniq)
        parents3 = tools.unique2parent(uniq)

        assert_array_equal(parents1, parents2)
        assert_array_equal(parents2, parents3)

    def test_unique2parent_mixed_resolution(self):
        """Mixed resolutions are reduced against each cell's own order

        Superseded ``test_unique2parent_mixed_resolution_raises``: the
        ``NotImplementedError`` is gone as of issue #136.
        """
        # Mix orders 6 and 7
        nside6 = 2**6
        nside7 = 2**7
        uniq_mixed = np.array([
            4 * (nside6**2) + 100,
            4 * (nside7**2) + 200,
        ])

        parents = tools.unique2parent(uniq_mixed)

        # 100 // 4**6 == 0 and 200 // 4**7 == 0 -- both land in base cell 0.
        assert_array_equal(parents, np.array([0, 0]))

    def test_unique2parent_mixed_matches_per_order_calls(self):
        """Every base cell, at five orders, in one mixed array"""
        orders = [2, 5, 9, 18, 29]
        uniq = []
        expect = []
        for order in orders:
            for base in range(12):
                # A cell in the middle of each base cell's z-order block.
                uniq.append(4 * (4**order) + base * (4**order) + 4**order // 2)
                expect.append(base)
        uniq = np.array(uniq, dtype=np.int64)

        assert_array_equal(tools.unique2parent(uniq), np.array(expect))
        # Same answer as calling one order at a time.
        per_order = np.concatenate([
            tools.unique2parent(uniq[i * 12:(i + 1) * 12])
            for i in range(len(orders))
        ])
        assert_array_equal(tools.unique2parent(uniq), per_order)

    def test_unique2parent_matches_legacy_single_resolution(self):
        """Single-resolution answers are unchanged by the multi-res rework

        The retired body was ``(uniq // 4**(order-1) - 16) // 4`` after
        collapsing the decoded orders to one scalar.
        """
        rng = np.random.default_rng(2136)
        for order in range(0, 21):
            nest = rng.integers(0, 12 * (4**order), size=20, dtype=np.int64)
            uniq = 4 * (4**order) + nest
            legacy = (uniq // 4 ** (order - 1) - 16) // 4
            assert_array_equal(tools.unique2parent(uniq),
                               legacy.astype(np.int64))

    def test_unique2parent_scalar(self):
        """A scalar UNIQ returns a scalar parent"""
        parent = tools.unique2parent(int(4 * (4**6) + 7 * (4**6) + 11))

        assert np.ndim(parent) == 0
        assert parent == 7

    def test_unique2parent_rejects_invalid(self):
        """Not-a-UNIQ input raises instead of returning a nonsense parent"""
        with pytest.raises(ValueError, match="valid UNIQ"):
            tools.unique2parent(np.array([1], dtype=np.int64))


class TestGeo2Uniq:
    """Test geographic to UNIQ conversion"""

    def test_geo2uniq_single_point(self):
        """Test single lat/lon point"""
        lat, lon = 45.0, -122.0
        order = 6

        uniq = tools.geo2uniq(lat, lon, order)

        # Check it's a valid UNIQ (should be integer)
        assert isinstance(uniq, (int, np.integer))

        # Check it's in valid range for this order
        nside = 2**order
        min_uniq = 4 * (nside**2)
        max_uniq = 4 * (nside**2) + 12 * (nside**2)
        assert min_uniq <= uniq < max_uniq

    def test_geo2uniq_array(self):
        """Test array of lat/lon points"""
        lats = np.array([45.0, 47.0, 49.0])
        lons = np.array([-122.0, -120.0, -118.0])
        order = 8

        uniq = tools.geo2uniq(lats, lons, order)

        # Should return array of same length
        assert len(uniq) == len(lats)

        # All should be valid UNIQ values
        nside = 2**order
        assert np.all(uniq >= 4 * (nside**2))

    def test_geo2uniq_deterministic(self):
        """Test that same inputs give same outputs"""
        lat, lon = 45.0, -122.0
        order = 10

        uniq1 = tools.geo2uniq(lat, lon, order)
        uniq2 = tools.geo2uniq(lat, lon, order)
        uniq3 = tools.geo2uniq(lat, lon, order)

        assert uniq1 == uniq2 == uniq3

    def test_geo2uniq_different_orders(self):
        """Test that different orders give different results"""
        lat, lon = 45.0, -122.0

        uniq6 = tools.geo2uniq(lat, lon, order=6)
        uniq8 = tools.geo2uniq(lat, lon, order=8)
        uniq12 = tools.geo2uniq(lat, lon, order=12)

        # Different orders should give different UNIQ values
        assert uniq6 != uniq8
        assert uniq8 != uniq12


class TestUniqEncoderOrders:
    """Test scalar-or-array ``order`` on geo2uniq / norm2uniq (issue #136)"""

    def _points(self, n=40, seed=1136):
        rng = np.random.default_rng(seed)
        return (rng.uniform(-89.0, 89.0, size=n),
                rng.uniform(-180.0, 180.0, size=n),
                rng.integers(0, tools.MAX_ORDER + 1, size=n))

    def test_default_is_max_order(self):
        """Both encoders default to MAX_ORDER, not the legacy 18"""
        assert (tools.geo2uniq(41.0, -3.0)
                == tools.geo2uniq(41.0, -3.0, order=tools.MAX_ORDER))
        assert (tools.norm2uniq(5, 3)
                == tools.norm2uniq(5, 3, tools.MAX_ORDER))
        # ...and that is a different answer from the old default.
        assert tools.geo2uniq(41.0, -3.0) != tools.geo2uniq(41.0, -3.0, 18)

    def test_geo2uniq_uniform_array_order_equals_scalar(self):
        """Where the two forms overlap they agree exactly"""
        lats, lons, _ = self._points()
        for order in (0, 4, 13, 29):
            assert_array_equal(
                tools.geo2uniq(lats, lons, order=np.full(lats.size, order)),
                tools.geo2uniq(lats, lons, order=order))

    def test_geo2uniq_array_order_matches_elementwise(self):
        """Per-element orders match calling one element at a time"""
        lats, lons, orders = self._points()
        got = tools.geo2uniq(lats, lons, order=orders)
        expect = np.array([int(tools.geo2uniq(lats[i], lons[i],
                                              order=int(orders[i])))
                           for i in range(lats.size)], dtype=np.int64)

        assert_array_equal(got, expect)
        # The encoded orders are recoverable from the result.
        assert_array_equal(tools._uniq_orders(got), orders.astype(np.int64))

    def test_norm2uniq_uniform_array_order_equals_scalar(self):
        """Same overlap check for the address-space encoder"""
        rng = np.random.default_rng(3136)
        parent = rng.integers(0, 12, size=25, dtype=np.int64)
        for order in (0, 4, 13, 29):
            normed = rng.integers(0, 4**order, size=25, dtype=np.int64)
            assert_array_equal(
                tools.norm2uniq(normed, parent, np.full(25, order)),
                tools.norm2uniq(normed, parent, order))

    def test_norm2uniq_array_order_matches_elementwise(self):
        """Per-element orders broadcast without group-by-order dispatch"""
        rng = np.random.default_rng(4136)
        orders = rng.integers(0, tools.MAX_ORDER + 1, size=30)
        parent = rng.integers(0, 12, size=30, dtype=np.int64)
        normed = np.array([rng.integers(0, 4 ** int(o)) for o in orders],
                          dtype=np.int64)

        got = tools.norm2uniq(normed, parent, orders)
        expect = np.array([int(tools.norm2uniq(int(normed[i]), int(parent[i]),
                                               int(orders[i])))
                           for i in range(orders.size)], dtype=np.int64)

        assert_array_equal(got, expect)
        assert_array_equal(tools._uniq_orders(got), orders.astype(np.int64))

    def test_encoders_agree_on_mixed_orders(self):
        """geo2uniq and norm2uniq describe the same cells, order by order

        Closes the loop over the whole family: encode lat/lon at mixed orders,
        take the result apart with ``unique2parent`` (phase 2), and re-encode
        the pieces with ``norm2uniq``.
        """
        lats, lons, orders = self._points(seed=5136)
        uniq = tools.geo2uniq(lats, lons, order=orders)

        parents = tools.unique2parent(uniq)
        nest = uniq - 4 * (4 ** orders.astype(np.int64))
        normed = nest - parents * (4 ** orders.astype(np.int64))

        assert np.all((parents >= 0) & (parents < 12))
        assert_array_equal(tools.norm2uniq(normed, parents, orders), uniq)

    def test_mixed_order_round_trip_through_uniq2geo(self):
        """geo2uniq -> uniq2geo -> geo2uniq is the identity on mixed input"""
        lats, lons, orders = self._points(seed=6136)
        uniq = tools.geo2uniq(lats, lons, order=orders)

        out_lat, out_lon = tools.uniq2geo(uniq)

        assert_array_equal(tools.geo2uniq(out_lat, out_lon, order=orders),
                           uniq)

    def test_uniq_cannot_carry_the_point_kind(self):
        """An order-29 UNIQ is the max-resolution area cell, not a point

        Documented on both encoders: UNIQ is ``4 * 4**order + nested``, with no
        kind bit. Point-vs-area lives in the packed word's suffix instead, so
        the two words below are distinguishable and their UNIQ cells are not.
        """
        lat, lon = 12.5, -46.25
        area = tools.geo2mort(lat, lon, order=tools.MAX_ORDER)
        point = tools.geo2mort(lat, lon)

        assert not tools.is_point(area)[0]
        assert tools.is_point(point)[0]
        assert area[0] != point[0]

        cells = []
        for word in (area, point):
            normed, parent, order = tools.mort2norm(word)
            cells.append(int(tools.norm2uniq(normed, parent, order)))

        # Same UNIQ from an area word and a point word -- the kind is lost.
        assert cells[0] == cells[1]
        # And that is exactly what the geo2uniq default returns.
        assert int(tools.geo2uniq(lat, lon)) == cells[0]

    def test_order_out_of_range_raises(self):
        """Orders outside 0..MAX_ORDER are rejected, scalar or array"""
        for bad in (-1, tools.MAX_ORDER + 1):
            with pytest.raises(ValueError, match="order must be"):
                tools.geo2uniq(1.0, 2.0, order=bad)
            with pytest.raises(ValueError, match="order must be"):
                tools.norm2uniq(1, 2, order=bad)
            with pytest.raises(ValueError, match="order must be"):
                tools.geo2uniq(np.array([1.0]), np.array([2.0]),
                               order=np.array([bad]))

    def test_order_array_length_mismatch_raises(self):
        """An order array must carry one entry per input element"""
        lats = np.array([1.0, 2.0, 3.0])
        lons = np.array([4.0, 5.0, 6.0])

        with pytest.raises(ValueError, match="one entry per input element"):
            tools.geo2uniq(lats, lons, order=np.array([5, 6]))
        with pytest.raises(ValueError, match="one entry per input element"):
            tools.norm2uniq(np.array([1, 2, 3]), 0, order=np.array([5, 6]))


def _uniq_at(order, nest):
    """UNIQ cell number(s) for NESTED index/indices at ``order``."""
    return 4 * (4**order) + np.asarray(nest, dtype=np.int64)


class TestUniqOrders:
    """Test the per-element order decode underpinning the UNIQ decoders"""

    def test_decodes_every_order(self):
        """First, last and an interior UNIQ value decode to their own order"""
        for order in range(tools.MAX_ORDER + 1):
            npix = 12 * (4**order)
            values = _uniq_at(order, [0, npix // 2, npix - 1])
            assert_array_equal(tools._uniq_orders(values),
                               np.full(3, order, dtype=np.int64))

    def test_order_boundaries_are_exact(self):
        """The last value of order k and the first of k+1 are adjacent ints

        The retired ``log2(uniq / 4) // 2`` decode went through float64, which
        cannot separate these above 2**53 (issue #136).
        """
        for order in range(tools.MAX_ORDER):
            last = 4 * (4**order) + 12 * (4**order) - 1
            first = 4 * (4 ** (order + 1))
            assert first == last + 1
            assert_array_equal(tools._uniq_orders([last, first]),
                               np.array([order, order + 1], dtype=np.int64))

    def test_out_of_range_raises(self):
        """Values below the order-0 floor or above the MAX_ORDER ceiling"""
        for bad in (0, 3, 4 ** (tools.MAX_ORDER + 2)):
            with pytest.raises(ValueError, match="valid UNIQ"):
                tools._uniq_orders([bad])


class TestUniq2Geo:
    """Test UNIQ to lat/lon conversion (order decoded from the value)"""

    def test_takes_no_order_argument(self):
        """The order parameter is gone -- the old silent-wrong-answer trap

        Before issue #136 ``uniq2geo(uniq, order)`` trusted the caller's order
        and returned plausible but wrong coordinates when it disagreed with the
        UNIQ value. There is no longer an argument to disagree with.
        """
        uniq = tools.geo2uniq(45.0, -122.0, order=6)

        with pytest.raises(TypeError):
            tools.uniq2geo(uniq, 6)
        with pytest.raises(TypeError):
            tools.uniq2geo(uniq, order=6)

    def test_order_cannot_disagree_with_the_value(self):
        """No order other than the encoded one decodes a UNIQ value

        The removed parameter was never cross-checked against the value. The
        order is now read from the value, and the value admits exactly one:
        order k occupies ``[4**(k+1), 4**(k+2))`` and the ranges tile without
        overlap, so every other order reconstructs an out-of-range NESTED
        index (which is what made a mismatched argument a bug rather than a
        second opinion).
        """
        order = 6
        uniq = int(tools.geo2uniq(45.0, -122.0, order=order))
        assert int(tools._uniq_orders([uniq])[0]) == order

        for wrong in range(tools.MAX_ORDER + 1):
            if wrong == order:
                continue
            nest = uniq - 4 * (4**wrong)
            assert not 0 <= nest < 12 * (4**wrong)

        # The value-decoded answer is the one the correct order gives.
        lat, lon = tools.uniq2geo(uniq)
        good_lon, good_lat = hp.pix2ang(order, uniq - 4 * (4**order))
        assert_allclose([lat, lon], [good_lat, good_lon])

    def test_default_order_no_longer_decides_the_answer(self):
        """A bare call on non-order-18 data is now correct, not order-18

        ``order=18`` was the reachable form of the trap: any caller who omitted
        it got 18 whatever their data was (issue #136 (3)).
        """
        for order in (9, 18, 29):
            uniq = int(tools.geo2uniq(40.0, 15.0, order=order))
            lat, lon = tools.uniq2geo(uniq)
            expect_lon, expect_lat = hp.pix2ang(order, uniq - 4 * (4**order))
            assert_allclose([lat, lon], [expect_lat, expect_lon])

    def test_scalar_in_scalar_out(self):
        """A scalar UNIQ returns scalar lat/lon (mort2geo relies on this)"""
        uniq = int(tools.geo2uniq(45.0, -122.0, order=9))
        lat, lon = tools.uniq2geo(uniq)

        assert np.ndim(lat) == 0
        assert np.ndim(lon) == 0
        assert -90.0 <= lat <= 90.0

    def test_mixed_order_input(self):
        """Mixed-resolution UNIQ decodes element by element"""
        lat, lon = 45.0, -122.0
        orders = [3, 7, 12, 18, 29]
        uniq = np.array([int(tools.geo2uniq(lat, lon, order=o))
                         for o in orders], dtype=np.int64)

        lats, lons = tools.uniq2geo(uniq)

        assert lats.shape == (len(orders),)
        # Every element matches the single-resolution answer for its own order.
        for i, o in enumerate(orders):
            one_lat, one_lon = tools.uniq2geo(int(uniq[i]))
            assert_allclose([lats[i], lons[i]], [one_lat, one_lon])
            # ...and lands inside the cell it came from.
            assert tools.geo2uniq(lats[i], lons[i], order=o) == uniq[i]

    def test_round_trip_to_cell_centre(self):
        """geo2uniq -> uniq2geo recovers coordinates to cell-centre precision"""
        rng = np.random.default_rng(136)
        lats = rng.uniform(-85.0, 85.0, size=50)
        lons = rng.uniform(-180.0, 180.0, size=50)

        for order in (0, 1, 5, 11, 18, 24, 29):
            uniq = tools.geo2uniq(lats, lons, order=order)
            out_lat, out_lon = tools.uniq2geo(uniq)
            # Re-encoding the centre lands back in the same cell.
            assert_array_equal(tools.geo2uniq(out_lat, out_lon, order=order),
                               uniq)
            # Centres of fine cells sit within a cell scale of the input.
            # pix2ang returns longitude in [0, 360); compare on [-180, 180).
            if order >= 18:
                assert_allclose(out_lat, lats, atol=1e-3)
                assert_allclose((out_lon + 180.0) % 360.0 - 180.0, lons,
                                atol=1e-3)

    def test_empty_input(self):
        """Empty in -> empty out, no order to guess"""
        lat, lon = tools.uniq2geo(np.array([], dtype=np.int64))
        assert lat.size == 0
        assert lon.size == 0

    def test_invalid_uniq_raises(self):
        """A non-UNIQ integer is rejected rather than silently decoded"""
        with pytest.raises(ValueError, match="valid UNIQ"):
            tools.uniq2geo(np.array([2], dtype=np.int64))


class TestMortonStructure:
    """Test morton index structural properties"""

    def test_morton_digits_valid(self):
        """Test that the decode-through-kernel decimal repr uses valid digits.

        After the issue #48 flip the bare ``i64`` is the packed word, not its
        decimal value; the human-readable structure (leading base digit, then
        ``1..=4`` per order) lives in the ``decimal_repr``.
        """
        from mortie import _rustie
        lats = np.array([45.0, -45.0, 0.0, 60.0, -30.0])
        lons = np.array([-122.0, 122.0, 0.0, -90.0, 45.0])

        for order in [6, 8, 10, 12]:
            morton = np.ascontiguousarray(
                tools.geo2mort(lats, lons, order=order), dtype=np.uint64
            )
            reprs = _rustie.rust_mi_decimal_repr(morton)
            for s in reprs:
                digits = s.lstrip("-")
                # leading base-cell digit + one digit per order.
                assert len(digits) == order + 1
                for digit in digits[1:]:
                    assert digit in '1234', f"Invalid digit {digit} in repr {s}"

    def test_morton_sign_consistency(self):
        """Bit 63 indicates the hemisphere / base-cell region (uint64 word)."""
        bit63 = np.uint64(1) << np.uint64(63)
        # Points in northern hemisphere
        lats_north = np.array([45.0, 60.0, 30.0])
        lons_north = np.array([-122.0, 0.0, 45.0])

        # Points in southern hemisphere
        lats_south = np.array([-45.0, -60.0, -30.0])
        lons_south = np.array([-122.0, 0.0, 45.0])

        morton_north = tools.geo2mort(lats_north, lons_north, order=10)
        morton_south = tools.geo2mort(lats_south, lons_south, order=10)

        # Both bit-63-clear and bit-63-set words appear across the two sets
        # (exact distribution depends on HEALPix geometry).
        all_morton = np.concatenate([morton_north, morton_south])
        assert np.any(all_morton < bit63) and np.any(all_morton >= bit63)


class TestGeo2Mort:
    """Test full geographic to Morton index conversion"""

    @pytest.fixture
    def sample_coords(self):
        """Sample coordinates for testing"""
        return {
            'single': (45.0, -122.0),
            'array': (
                np.array([45.0, 47.0, 49.0, -45.0, -47.0]),
                np.array([-122.0, -120.0, -118.0, 122.0, 120.0])
            ),
            'equator': (
                np.array([0.0, 0.0, 0.0]),
                np.array([-180.0, 0.0, 180.0])
            ),
            'poles': (
                np.array([89.0, -89.0]),
                np.array([0.0, 0.0])
            )
        }

    def test_geo2mort_single_point(self, sample_coords):
        """Test single point conversion"""
        lat, lon = sample_coords['single']

        morton = tools.geo2mort(lat, lon, order=6)

        # Should return integer
        assert isinstance(morton, (int, np.integer, np.ndarray))

    def test_geo2mort_array(self, sample_coords):
        """Test array conversion"""
        lats, lons = sample_coords['array']

        morton = tools.geo2mort(lats, lons, order=8)

        # Should return array of same length
        assert len(morton) == len(lats)

        # Morton words are uint64 (issue #58)
        assert morton.dtype == np.uint64

    def test_geo2mort_deterministic(self, sample_coords):
        """Test that same inputs always give same outputs"""
        lat, lon = sample_coords['single']

        morton1 = tools.geo2mort(lat, lon, order=12)
        morton2 = tools.geo2mort(lat, lon, order=12)
        morton3 = tools.geo2mort(lat, lon, order=12)

        assert morton1 == morton2 == morton3

    def test_geo2mort_array_deterministic(self, sample_coords):
        """Test determinism for arrays"""
        lats, lons = sample_coords['array']

        morton1 = tools.geo2mort(lats, lons, order=10)
        morton2 = tools.geo2mort(lats, lons, order=10)

        assert_array_equal(morton1, morton2)

    def test_geo2mort_order_hierarchy(self, sample_coords):
        """Test that clipping higher order to lower order is consistent"""
        lat, lon = sample_coords['single']

        # Get morton at different orders
        mort6 = tools.geo2mort(lat, lon, order=6)
        mort12 = tools.geo2mort(lat, lon, order=12)

        # Both should be valid integers
        assert isinstance(mort6, (int, np.integer, np.ndarray))
        assert isinstance(mort12, (int, np.integer, np.ndarray))

        # Clipping should reduce magnitude (this is a structural test)
        mort12_clipped = tools.clip2order(6, np.array([mort12]))
        assert len(mort12_clipped) == 1

    def test_geo2mort_equator(self, sample_coords):
        """Test points on equator"""
        lats, lons = sample_coords['equator']

        morton = tools.geo2mort(lats, lons, order=8)

        # Should get valid morton indices
        assert len(morton) == len(lats)
        assert not np.any(np.isnan(morton))

    def test_geo2mort_poles(self, sample_coords):
        """Test points near poles"""
        lats, lons = sample_coords['poles']

        morton = tools.geo2mort(lats, lons, order=8)

        # Should get valid morton indices
        assert len(morton) == len(lats)
        assert not np.any(np.isnan(morton))


class TestNorm2Mort:
    """norm2mort: the order-29-native packed forward encoder (inverse of mort2norm)."""

    def test_norm2mort_basic(self):
        """Basic conversion returns one packed word per (normed, parent)."""
        order = 6
        normed = np.array([100, 200, 300], dtype=np.int64)
        parents = np.array([2, 3, 4], dtype=np.int64)

        morton = np.array(
            [int(tools.norm2mort(int(n), int(p), order)) for n, p in zip(normed, parents)],
            dtype=np.uint64,
        )
        assert len(morton) == len(normed)
        assert morton.dtype == np.uint64

    def test_norm2mort_inverts_mort2norm(self):
        """norm2mort is the exact inverse of mort2norm."""
        for order in (6, 8, 18, 29):
            for normed, parent in [(100, 2), (4096, 5), (0, 11), (12345, 8)]:
                if normed >= 4 ** order:
                    continue
                m = tools.norm2mort(normed, parent, order)
                assert tools.mort2norm(m) == (normed, parent, order)

    def test_norm2mort_different_orders(self):
        """Different orders give different packed words for the same (normed, parent)."""
        m6 = int(tools.norm2mort(100, 2, 6))
        m8 = int(tools.norm2mort(100, 2, 8))
        m10 = int(tools.norm2mort(100, 2, 10))
        assert m6 != m8
        assert m8 != m10

    def test_norm2mort_order29_native(self):
        """norm2mort reaches order 29 (no order-18 cap)."""
        m = tools.norm2mort(123, 2, 29)
        normed, parent, order = tools.mort2norm(m)
        assert (int(normed), int(parent), order) == (123, 2, 29)


class TestClip2Order:
    """Test resolution clipping (kernel coarsen)."""

    def test_clip2order_rejects_removed_print_factor(self):
        """The order-18-anchored print_factor flag is gone (issue #68).

        It returned ``18 - clip_order``, which went negative for the
        order-19..29 words this package now encodes. Pinned so the flag
        cannot quietly return.
        """
        with pytest.raises(TypeError):
            tools.clip2order(12, print_factor=True)

    def test_clip2order_requires_words(self):
        """midx is now required -- there is no word-less call form."""
        with pytest.raises(TypeError):
            tools.clip2order(12)

    def test_clip2order_clipping(self):
        """Clipping coarsens packed words to the target order."""
        # Two order-18 packed words.
        morton18 = np.array(
            [int(tools.norm2mort(12345, 2, 18)), int(tools.norm2mort(54321, 4, 18))],
            dtype=np.uint64,
        )
        morton12 = tools.clip2order(12, morton18)
        # The coarsened words decode to order 12 and the same base cells.
        _, parent, order = tools.mort2norm(morton12)
        assert order == 12
        np.testing.assert_array_equal(parent, [2, 4])
        # Coarsening == re-encoding the order-18 cell's order-12 ancestor.
        n18, p18, _ = tools.mort2norm(morton18)
        expected = np.array(
            [int(tools.norm2mort(int(n) >> (2 * 6), int(p), 12))
             for n, p in zip(n18, p18)],
            dtype=np.uint64,
        )
        np.testing.assert_array_equal(morton12, expected)

    def test_clip2order_negative_indices(self):
        """Clipping a southern (bit-63-set) word keeps it southern."""
        bit63 = np.uint64(1) << np.uint64(63)
        morton18 = np.array(
            [int(tools.norm2mort(100, 2, 18)), int(tools.norm2mort(200, 9, 18))],
            dtype=np.uint64,
        )
        morton12 = tools.clip2order(12, morton18)
        # Base cell 9 sets bit 63 -> stays set; base 2 stays clear.
        assert morton18[0] < bit63 and morton12[0] < bit63
        assert morton18[1] >= bit63 and morton12[1] >= bit63

    def test_clip2order_deterministic(self):
        """Test determinism"""
        morton18 = np.array(
            [int(tools.norm2mort(100, 2, 18)), int(tools.norm2mort(200, 9, 18))],
            dtype=np.uint64,
        )
        result1 = tools.clip2order(12, morton18)
        result2 = tools.clip2order(12, morton18)
        assert_array_equal(result1, result2)


class TestIntegration:
    """Integration tests for complete workflow"""

    def test_round_trip_consistency(self):
        """Test that processing pipeline is consistent"""
        # Generate test points
        lats = np.array([45.0, -45.0, 0.0, 60.0, -60.0])
        lons = np.array([-122.0, 122.0, 0.0, -90.0, 90.0])

        for order in [6, 8, 10, 12, 14]:
            morton1 = tools.geo2mort(lats, lons, order=order)
            morton2 = tools.geo2mort(lats, lons, order=order)

            # Same inputs should give same outputs
            assert_array_equal(morton1, morton2)

    def test_spatial_locality(self):
        """Test that very nearby points have similar morton indices"""
        # Points very close together
        lat_base = 45.0
        lon_base = -122.0
        epsilon = 0.0001  # Very small offset

        lats = np.array([lat_base, lat_base + epsilon, lat_base - epsilon])
        lons = np.array([lon_base, lon_base + epsilon, lon_base - epsilon])

        morton = tools.geo2mort(lats, lons, order=14)

        # Nearby points should have some similarity
        # At minimum, they should all be valid (no NaN/Inf)
        assert not np.any(np.isnan(morton))
        assert not np.any(np.isinf(morton))

    def test_full_globe_coverage(self):
        """Test that we can process points across entire globe"""
        # Sample points across globe
        np.random.seed(42)
        n_points = 100
        lats = np.random.uniform(-85, 85, n_points)  # Avoid extreme poles
        lons = np.random.uniform(-180, 180, n_points)

        morton = tools.geo2mort(lats, lons, order=10)

        # Should get valid morton indices for all points
        assert len(morton) == n_points
        assert not np.any(np.isnan(morton))
        assert not np.any(np.isinf(morton))

    def test_reproducibility_across_runs(self):
        """Test that results are reproducible across multiple runs"""
        np.random.seed(123)
        lats = np.random.uniform(-80, 80, 50)
        lons = np.random.uniform(-180, 180, 50)

        # Run multiple times
        results = []
        for _ in range(5):
            morton = tools.geo2mort(lats, lons, order=12)
            results.append(morton)

        # All results should be identical
        for i in range(1, len(results)):
            assert_array_equal(results[0], results[i])


class TestReferenceData:
    """Generate and validate reference data for regression testing"""

    def test_reference_single_points(self):
        """Generate reference data for single points at various orders"""
        test_points = [
            (45.0, -122.0),   # Pacific Northwest
            (-45.0, 122.0),   # Southern hemisphere
            (0.0, 0.0),       # Equator, Prime Meridian
            (60.0, -90.0),    # High latitude
            (-30.0, 45.0),    # Southern mid-latitude
        ]

        reference = {}
        for idx, (lat, lon) in enumerate(test_points):
            for order in [6, 8, 10, 12, 14, 16, 18]:
                morton = tools.geo2mort(lat, lon, order=order)
                key = f"point_{idx}_order_{order}"
                reference[key] = morton

        # Verify all reference data is valid (may be scalars or 0-d arrays)
        for v in reference.values():
            assert isinstance(v, (int, np.integer, np.ndarray))
            # If array, should be scalar-like
            if isinstance(v, np.ndarray):
                assert v.ndim == 0 or (v.ndim == 1 and len(v) == 1)

    def test_reference_arrays(self):
        """Generate reference data for coordinate arrays"""
        # Matching arrays (not meshgrid - healpy expects matching sizes)
        n_points = 20
        lats = np.linspace(-80, 80, n_points)
        lons = np.linspace(-180, 180, n_points)

        reference = {}
        for order in [6, 10, 14]:
            morton = tools.geo2mort(lats, lons, order=order)
            reference[f"array_order_{order}"] = morton

        # Verify all arrays
        for morton in reference.values():
            assert len(morton) == n_points
            assert not np.any(np.isnan(morton))


class TestGenerateMortonChildren:
    """generate_morton_children: NESTED-space descent, packed words (issue #48)."""

    def _parent(self, normed, base, order):
        """A packed parent word for a given (normed, base, order)."""
        return int(tools.norm2mort(normed, base, order))

    def test_one_level_count_and_descent(self):
        """One level down yields 4 children; staying put yields the parent."""
        parent = self._parent(1234, base=11, order=6)  # southern base cell
        children = tools.generate_morton_children(parent, target_order=7)
        assert len(children) == 4
        # Each child is order 7 and shares the parent's order-6 ancestor.
        _, _, order = tools.mort2norm(children)
        assert order == 7
        np.testing.assert_array_equal(
            tools.clip2order(6, np.ascontiguousarray(children, dtype=np.uint64)),
            np.full(4, parent, dtype=np.uint64),
        )
        # Already at target order -> returns the parent unchanged.
        np.testing.assert_array_equal(
            tools.generate_morton_children(parent, target_order=6),
            np.array([parent], dtype=np.uint64),
        )

    def test_two_levels_count_and_membership(self):
        """Descending 2 levels yields 16 children, all sharing the parent prefix."""
        parent = self._parent(420, base=2, order=5)
        children = tools.generate_morton_children(parent, target_order=7)
        assert len(children) == 16
        # Each child coarsens back to the parent at order 5.
        np.testing.assert_array_equal(
            tools.clip2order(5, np.ascontiguousarray(children, dtype=np.uint64)),
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
        children = tools.generate_morton_children(parent, target_order=8)
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
                    tools.generate_morton_children(parent, target),
                    reference(parent, target),
                )

    def test_target_below_parent_raises(self):
        parent = self._parent(7, base=8, order=6)
        with pytest.raises(ValueError):
            tools.generate_morton_children(parent, target_order=3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

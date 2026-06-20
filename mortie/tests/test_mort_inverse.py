"""
Test mort2geo and related inverse functions
"""
import pytest
import numpy as np
from mortie import tools


class TestMort2Geo:
    """Test the mort2geo inverse function"""

    def test_round_trip_single_point(self):
        """Test that geo2mort -> mort2geo -> geo2mort gives same morton"""
        for order in [6, 10, 14]:
            # Test various points
            test_points = [
                (0, 0),
                (45, -90),
                (89, 180),
                (30.5, -95.5),
            ]

            for lat, lon in test_points:
                # Convert to morton
                morton = tools.geo2mort(lat, lon, order=order)[0]

                # Convert back to lat/lon (pixel center)
                lat2, lon2 = tools.mort2geo(morton)

                # Convert center back to morton
                morton2 = tools.geo2mort(lat2[0], lon2[0], order=order)[0]

                # Should get the same morton index
                assert morton == morton2, f"Round trip failed for ({lat}, {lon}) at order {order}"

    def test_morton_validation(self):
        """Test morton index validation on packed words (issue #48)."""
        m6 = int(tools.norm2mort(1234, 2, 6))     # order 6, north
        m6s = int(tools.norm2mort(2345, 8, 6))    # order 6, south
        m7 = int(tools.norm2mort(5000, 0, 7))     # order 7
        # Valid packed words validate true.
        assert tools.validate_morton(m6)
        assert tools.validate_morton(m6s)
        assert tools.validate_morton(m7)

        # Order is read from the word's suffix, not decimal digits.
        assert tools.infer_order_from_morton(m6) == 6
        assert tools.infer_order_from_morton(m6s) == 6
        assert tools.infer_order_from_morton(m7) == 7

        # An order mismatch is rejected.
        with pytest.raises(ValueError):
            tools.validate_morton(m6, order=7)

        # The empty sentinel (0) is not a valid morton word.
        with pytest.raises(ValueError):
            tools.validate_morton(0)

    def test_mort2bbox(self):
        """Test bounding box generation"""
        morton = int(tools.norm2mort(2120, 2, 6))

        bbox = tools.mort2bbox(morton)

        # Check bbox structure
        assert 'west' in bbox
        assert 'east' in bbox
        assert 'north' in bbox
        assert 'south' in bbox

        # Check bbox validity
        assert bbox['west'] < bbox['east'] or bbox['west'] > 180  # handle dateline
        assert bbox['south'] < bbox['north']

        # Center should be within bbox
        lat, lon = tools.mort2geo(morton)
        lat, lon = lat[0], lon[0]

        assert bbox['south'] <= lat <= bbox['north']
        # Longitude check is complex due to wrapping

    def test_mort2polygon(self):
        """Test polygon generation"""
        morton = int(tools.norm2mort(2120, 2, 6))

        polygon = tools.mort2polygon(morton)

        # Should be a closed polygon (first == last)
        assert polygon[0] == polygon[-1]

        # Should have 5 points (4 corners + closing point)
        assert len(polygon) == 5

        # Each point should be [lat, lon] (standard geographic order)
        for point in polygon:
            assert len(point) == 2
            assert -90 <= point[0] <= 90    # latitude
            assert -180 <= point[1] <= 180  # longitude

    def test_array_input(self):
        """Test that array inputs work correctly"""
        mortons = np.array([
            int(tools.norm2mort(2120, 2, 6)),
            int(tools.norm2mort(2120, 8, 6)),
            int(tools.norm2mort(1402, 3, 6)),
        ])

        # Test mort2geo with array
        lats, lons = tools.mort2geo(mortons)
        assert len(lats) == len(mortons)
        assert len(lons) == len(mortons)

        # Test mort2bbox with array
        bboxes = tools.mort2bbox(mortons)
        assert len(bboxes) == len(mortons)
        assert all('west' in bbox for bbox in bboxes)

        # Test mort2polygon with array
        polygons = tools.mort2polygon(mortons)
        assert len(polygons) == len(mortons)
        assert all(poly[0] == poly[-1] for poly in polygons)

    def test_bbox_polygon_array_matches_scalar(self):
        """Array mort2bbox/mort2polygon must equal the per-element scalar calls.

        Regression: mort2bbox indexed the boundary array on the wrong axis for
        multi-cell input (it only avoided a shape error at length 3), silently
        returning wrong boxes.  The batched path fixes this and must agree with
        the scalar reference at every length, including an antimeridian cell.
        """
        lats = np.array([40.0, -70.0, 12.0, 5.0, 80.0])
        lons = np.array([-120.0, 30.0, 179.9, -179.9, 0.0])
        for n in (2, 3, 4, 5):
            mortons = tools.geo2mort(lats[:n], lons[:n], order=6)
            bbox_arr = tools.mort2bbox(mortons)
            bbox_ref = [tools.mort2bbox(int(m)) for m in mortons]
            assert bbox_arr == bbox_ref, f"mort2bbox mismatch at n={n}"

            poly_arr = tools.mort2polygon(mortons)
            poly_ref = [tools.mort2polygon(int(m)) for m in mortons]
            assert poly_arr == poly_ref, f"mort2polygon mismatch at n={n}"

            # step > 1 is a separate boundary branch (ncols = 4*step).
            poly2_arr = tools.mort2polygon(mortons, step=2)
            poly2_ref = [tools.mort2polygon(int(m), step=2) for m in mortons]
            assert poly2_arr == poly2_ref, f"mort2polygon(step=2) mismatch at n={n}"

    def test_negative_morton_hemisphere(self):
        """Bit 63 of the (uint64) word encodes polar proximity / hemisphere."""
        bit63 = np.uint64(1) << np.uint64(63)
        # Test high northern latitude - bit 63 clear
        lat, lon = 70.0, 0.0
        morton_north = tools.geo2mort(lat, lon, order=6)[0]
        assert morton_north < bit63, f"Morton at lat={lat} should leave bit 63 clear"

        # Test high southern latitude - bit 63 set
        lat, lon = -70.0, 0.0
        morton_south = tools.geo2mort(lat, lon, order=6)[0]
        assert morton_south >= bit63, f"Morton at lat={lat} should set bit 63"

        # Verify the inverse functions work
        lat_north, _ = tools.mort2geo(morton_north)
        assert lat_north[0] > 45, f"North morton {morton_north} should decode to lat > 45"

        lat_south, _ = tools.mort2geo(morton_south)
        assert lat_south[0] < -45, f"South morton {morton_south} should decode to lat < -45"

    def test_mort2norm_empty(self):
        """Empty input returns empty int64 arrays and order 0, not IndexError."""
        for empty_in in (np.array([]), np.array([], dtype=np.int64), []):
            normed, parent, order = tools.mort2norm(empty_in)
            assert order == 0
            assert normed.dtype == np.int64
            assert parent.dtype == np.int64
            assert normed.size == 0
            assert parent.size == 0

    def test_mort2norm_inverse(self):
        """Test the mort2norm conversion"""
        morton = int(tools.norm2mort(2120, 2, 6))

        # Decode morton (order is read from the packed word).
        normed, parent, order = tools.mort2norm(morton)

        # Check that order was correctly recovered.
        assert order == 6, f"Expected order 6, got {order}"

        # Parent should be 0-11
        assert 0 <= abs(parent) <= 11

        # Convert back through the chain
        uniq = tools.norm2uniq(normed, parent, order)
        lat, lon = tools.uniq2geo(uniq, order)

        # Should be able to get back to same morton
        morton2 = tools.geo2mort(lat, lon, order)[0]
        assert morton == morton2


class TestRustMortNested:
    """Direct tests for the rust_mort2nested / rust_nested2mort pyfunctions."""

    def _valid_mortons(self, order, n=50):
        rng = np.random.default_rng(order + 1)
        normed = rng.integers(0, 4**order, size=n, dtype=np.int64)
        parents = (np.arange(n) % 12).astype(np.int64)
        return np.asarray(
            [int(tools.norm2mort(int(no), int(p), order))
             for no, p in zip(normed, parents)],
            dtype=np.uint64,
        )

    def test_imports(self):
        from mortie import _rustie
        assert hasattr(_rustie, 'rust_mort2nested')
        assert hasattr(_rustie, 'rust_nested2mort')

    @pytest.mark.parametrize("order", [1, 6, 10, 14, 18])
    def test_roundtrip(self, order):
        from mortie import _rustie
        mortons = self._valid_mortons(order)
        nested, depths = _rustie.rust_mort2nested(mortons)
        assert np.all(depths == order)
        back = _rustie.rust_nested2mort(nested, depths)
        np.testing.assert_array_equal(back, mortons)

    def test_nested_matches_parent_normed(self):
        """nested == parent * nside^2 + normed, consistent with mort2norm."""
        from mortie import _rustie
        order = 6
        mortons = self._valid_mortons(order)
        nested, depths = _rustie.rust_mort2nested(mortons)
        normed, parent, o = tools.mort2norm(mortons)
        assert o == order
        nside_sq = (1 << (2 * order))
        np.testing.assert_array_equal(
            nested.astype(np.int64), parent * nside_sq + normed)

    def test_nested2mort_length_mismatch(self):
        from mortie import _rustie
        nested = np.array([0, 1], dtype=np.uint64)
        depths = np.array([6], dtype=np.uint8)
        with pytest.raises(ValueError, match="same length"):
            _rustie.rust_nested2mort(nested, depths)

    def test_mort2nested_zero_raises(self):
        from mortie import _rustie
        with pytest.raises(ValueError):
            _rustie.rust_mort2nested(np.array([0], dtype=np.uint64))

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
        """Test morton index validation"""
        # Valid morton indices
        assert tools.validate_morton(3122124)  # order 6 inferred
        assert tools.validate_morton(-5123123)  # order 6 inferred
        assert tools.validate_morton(13111111)  # order 7 inferred

        # Test order inference
        assert tools.infer_order_from_morton(3122124) == 6
        assert tools.infer_order_from_morton(-5123123) == 6
        assert tools.infer_order_from_morton(13111111) == 7

        # Invalid morton indices
        with pytest.raises(ValueError, match="digit"):
            tools.validate_morton(1234567)  # has digits > 4

        with pytest.raises(ValueError, match="digit"):
            tools.validate_morton(5123454)  # has digit 5

    def test_mort2bbox(self):
        """Test bounding box generation"""
        morton = 3122124

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
        morton = 3122124

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
        mortons = np.array([3122124, -3122124, 4231423])

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

    def test_negative_morton_hemisphere(self):
        """Test that negative morton indices encode polar proximity correctly"""
        # Test high northern latitude - should be positive morton
        lat, lon = 70.0, 0.0
        morton_north = tools.geo2mort(lat, lon, order=6)[0]
        assert morton_north > 0, f"Morton at lat={lat} should be positive, got {morton_north}"

        # Test high southern latitude - should be negative morton
        lat, lon = -70.0, 0.0
        morton_south = tools.geo2mort(lat, lon, order=6)[0]
        assert morton_south < 0, f"Morton at lat={lat} should be negative, got {morton_south}"

        # Verify the inverse functions work
        lat_north, _ = tools.mort2geo(morton_north)
        assert lat_north[0] > 45, f"Positive morton {morton_north} should decode to lat > 45"

        lat_south, _ = tools.mort2geo(morton_south)
        assert lat_south[0] < -45, f"Negative morton {morton_south} should decode to lat < -45"

    def test_mort2norm_inverse(self):
        """Test the mort2norm conversion"""
        morton = 3122124

        # Decode morton (order is inferred)
        normed, parent, order = tools.mort2norm(morton)

        # Check that order was correctly inferred
        assert order == 6, f"Expected order 6, got {order}"

        # Parent should be 0-11
        assert 0 <= abs(parent) <= 11

        # Convert back through the chain
        uniq = tools.norm2uniq(normed, parent, order)
        lat, lon = tools.uniq2geo(uniq, order)

        # Should be able to get back to same morton
        morton2 = tools.geo2mort(lat, lon, order)[0]
        assert morton == morton2
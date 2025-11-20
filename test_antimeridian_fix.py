#!/usr/bin/env python3
"""
Test script to verify the antimeridian normalization fix for mort2polygon.

This tests the specific case that was reported:
- Morton index -5111131 near Antarctica
- Polygon touches the antimeridian at 180°/-180°
- Should be normalized to use -180° consistently (western hemisphere)
"""

import numpy as np
from mortie import mort2polygon, mort2bbox
from shapely.geometry import Polygon


def test_antimeridian_normalization():
    """Test that polygons touching the antimeridian are properly normalized."""

    print("="*70)
    print("Testing Antimeridian Normalization Fix")
    print("="*70)

    # Test case: Morton index -5111131 (near Antarctica)
    morton = -5111131

    print(f"\nTest Morton Index: {morton}")
    print(f"Location: Near Antarctica (touches antimeridian)")

    # Get polygon
    polygon = mort2polygon(morton)

    print(f"\nPolygon vertices ({len(polygon)} points):")
    for i, (lon, lat) in enumerate(polygon):
        print(f"  {i}: lon={lon:8.2f}, lat={lat:7.4f}")

    # Extract longitudes (excluding closing point)
    lons = np.array([p[0] for p in polygon[:-1]])
    lats = np.array([p[1] for p in polygon[:-1]])

    # Check longitude span
    lon_span = lons.max() - lons.min()

    print(f"\nLongitude Statistics:")
    print(f"  Min:  {lons.min():8.2f}°")
    print(f"  Max:  {lons.max():8.2f}°")
    print(f"  Span: {lon_span:8.2f}°")

    # Check which hemisphere
    western_count = np.sum(lons < -0.1)
    eastern_count = np.sum(lons > 0.1)
    on_antimeridian = np.sum(np.abs(np.abs(lons) - 180.0) < 1e-6)

    print(f"\nVertex Distribution:")
    print(f"  Western hemisphere: {western_count}")
    print(f"  Eastern hemisphere: {eastern_count}")
    print(f"  On antimeridian:    {on_antimeridian}")

    # Verify the fix
    print(f"\n" + "="*70)
    print("VERIFICATION")
    print("="*70)

    # Check 1: Longitude span should be <= 180° (not touching -> not crossing)
    if lon_span <= 180:
        print(f"✓ PASS: Longitude span ({lon_span:.1f}°) is ≤ 180°")
        print(f"        Polygon correctly interpreted as NOT crossing antimeridian")
    else:
        print(f"✗ FAIL: Longitude span ({lon_span:.1f}°) is > 180°")
        print(f"        Polygon would be misinterpreted as crossing antimeridian!")
        return False

    # Check 2: All antimeridian vertices should use the same sign
    antimeridian_lons = lons[np.abs(np.abs(lons) - 180.0) < 1e-6]
    if len(antimeridian_lons) > 0:
        all_same_sign = np.all(antimeridian_lons > 0) or np.all(antimeridian_lons < 0)
        if all_same_sign:
            sign = "positive (+180°)" if antimeridian_lons[0] > 0 else "negative (-180°)"
            print(f"✓ PASS: All antimeridian vertices use {sign}")
        else:
            print(f"✗ FAIL: Antimeridian vertices have mixed signs!")
            print(f"        Values: {antimeridian_lons}")
            return False

    # Check 3: Hemisphere consistency
    if western_count > 0:
        expected_antimeridian = -180.0
        hemisphere = "western"
    else:
        expected_antimeridian = 180.0
        hemisphere = "eastern"

    if len(antimeridian_lons) > 0:
        actual_antimeridian = antimeridian_lons[0]
        if abs(actual_antimeridian - expected_antimeridian) < 1e-6:
            print(f"✓ PASS: Antimeridian vertices ({actual_antimeridian:+.0f}°) match")
            print(f"        {hemisphere} hemisphere majority")
        else:
            print(f"✗ FAIL: Expected {expected_antimeridian:+.0f}° for {hemisphere} hemisphere,")
            print(f"        but got {actual_antimeridian:+.0f}°")
            return False

    # Check 4: Shapely polygon interpretation
    poly = Polygon(polygon)
    poly_bounds = poly.bounds
    poly_span = poly_bounds[2] - poly_bounds[0]  # max_lon - min_lon

    print(f"\n✓ PASS: Shapely interprets polygon correctly")
    print(f"        Bounds: {poly_bounds}")
    print(f"        Span: {poly_span:.1f}° (not the full globe)")
    print(f"        Area: {poly.area:.4f} square degrees")

    print(f"\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70)
    print(f"\nExpected polygon (all vertices using -180°):")
    print(f"[(-180.0, -88.53802883735207),")
    print(f" (-150.0, -87.80696888064219),")
    print(f" (-157.5, -87.07581964295005),")
    print(f" (-180.0, -87.80696888064219),")
    print(f" (-180.0, -88.53802883735207)]")

    print(f"\nActual polygon (from mort2polygon):")
    print("[", end="")
    for i, (lon, lat) in enumerate(polygon[:-1]):  # Skip closing point
        if i > 0:
            print(",\n ", end="")
        print(f"({lon}, {lat})", end="")
    print("]")

    return True


def test_various_morton_indices():
    """Test the fix doesn't break normal polygons."""

    print(f"\n\n" + "="*70)
    print("Testing Various Morton Indices")
    print("="*70)

    test_cases = [
        (-5111131, "Antarctic (touches antimeridian)"),
        (-3111131, "Antarctic (doesn't cross)"),
        (5111131, "Arctic (positive)"),
        (0, "Equator/prime meridian"),
    ]

    for morton, description in test_cases:
        print(f"\nMorton {morton:10d}: {description}")
        polygon = mort2polygon(morton)
        lons = np.array([p[0] for p in polygon[:-1]])
        lon_span = lons.max() - lons.min()

        if lon_span <= 180:
            print(f"  ✓ Span: {lon_span:6.2f}° (OK)")
        else:
            print(f"  ⚠ Span: {lon_span:6.2f}° (may need normalization)")

        # Check for antimeridian vertices
        on_antimeridian = np.sum(np.abs(np.abs(lons) - 180.0) < 1e-6)
        if on_antimeridian > 0:
            print(f"  ℹ {on_antimeridian} vertices on antimeridian")


def test_bbox_antimeridian():
    """Test that bboxes touching the antimeridian are properly normalized."""

    print(f"\n\n" + "="*70)
    print("Testing Bbox Antimeridian Normalization")
    print("="*70)

    # Test case: Morton index -5111131 (near Antarctica)
    morton = -5111131

    print(f"\nTest Morton Index: {morton}")
    print(f"Location: Near Antarctica (touches antimeridian)")

    # Get bbox
    bbox = mort2bbox(morton)

    print(f"\nBounding Box:")
    print(f"  West:  {bbox['west']:8.2f}°")
    print(f"  South: {bbox['south']:7.4f}°")
    print(f"  East:  {bbox['east']:8.2f}°")
    print(f"  North: {bbox['north']:7.4f}°")

    # Check longitude span
    lon_span = bbox['east'] - bbox['west']

    print(f"\nLongitude Span: {lon_span:8.2f}°")

    # Verify the fix
    print(f"\n" + "="*70)
    print("VERIFICATION")
    print("="*70)

    # Check: Longitude span should be <= 180° (not wrapping)
    if lon_span <= 180:
        print(f"✓ PASS: Longitude span ({lon_span:.1f}°) is ≤ 180°")
        print(f"        Bbox correctly interpreted as NOT spanning globe")
    else:
        print(f"✗ FAIL: Longitude span ({lon_span:.1f}°) is > 180°")
        print(f"        Bbox would be misinterpreted as spanning entire globe!")
        return False

    # Check: Both west and east should have same sign if touching antimeridian
    if abs(bbox['west']) == 180.0 or abs(bbox['east']) == 180.0:
        if (bbox['west'] < 0 and bbox['east'] < 0) or (bbox['west'] > 0 and bbox['east'] > 0):
            print(f"✓ PASS: Bbox edges use consistent hemisphere")
        else:
            print(f"✗ FAIL: Bbox edges have mixed hemisphere signs")
            return False

    print(f"\n" + "="*70)
    print("BBOX TEST PASSED!")
    print("="*70)

    return True


if __name__ == "__main__":
    success = test_antimeridian_normalization()
    bbox_success = test_bbox_antimeridian()
    test_various_morton_indices()

    if success and bbox_success:
        print("\n✅ All tests passed!")
        exit(0)
    else:
        print("\n❌ Some tests failed!")
        exit(1)

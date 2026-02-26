"""
Performance benchmarks for morton_coverage (CodSpeed-compatible).

Run locally:
    pytest benchmarks/test_bench_coverage.py -v

Run with CodSpeed:
    pytest benchmarks/test_bench_coverage.py --codspeed
"""

import numpy as np
import pytest

from mortie import morton_coverage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def triangle():
    """Simple ~10° × 10° triangle."""
    return np.array([40.0, 50.0, 45.0]), np.array([-120.0, -120.0, -110.0])


@pytest.fixture
def square():
    """~10° × 10° square."""
    return np.array([40.0, 40.0, 50.0, 50.0]), np.array([-125.0, -115.0, -115.0, -125.0])


@pytest.fixture
def circle_100():
    """Circular polygon with 100 vertices (~5° radius, southern hemisphere)."""
    n = 100
    center_lat, center_lon, radius = -75.0, 0.0, 5.0
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    lats = center_lat + radius * np.cos(angles)
    lons = center_lon + radius * np.sin(angles)
    return lats, lons


@pytest.fixture
def circle_500():
    """Circular polygon with 500 vertices (~5° radius, southern hemisphere)."""
    n = 500
    center_lat, center_lon, radius = -75.0, 0.0, 5.0
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    lats = center_lat + radius * np.cos(angles)
    lons = center_lon + radius * np.sin(angles)
    return lats, lons


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def test_coverage_triangle_order4(benchmark, triangle):
    """morton_coverage: triangle at order 4."""
    lats, lons = triangle
    benchmark(morton_coverage, lats, lons, order=4)


def test_coverage_triangle_order6(benchmark, triangle):
    """morton_coverage: triangle at order 6."""
    lats, lons = triangle
    benchmark(morton_coverage, lats, lons, order=6)


def test_coverage_triangle_order8(benchmark, triangle):
    """morton_coverage: triangle at order 8."""
    lats, lons = triangle
    benchmark(morton_coverage, lats, lons, order=8)


def test_coverage_square_order6(benchmark, square):
    """morton_coverage: square at order 6."""
    lats, lons = square
    benchmark(morton_coverage, lats, lons, order=6)


def test_coverage_circle100_order6(benchmark, circle_100):
    """morton_coverage: 100-vertex circle at order 6."""
    lats, lons = circle_100
    benchmark(morton_coverage, lats, lons, order=6)


def test_coverage_circle500_order6(benchmark, circle_500):
    """morton_coverage: 500-vertex circle at order 6."""
    lats, lons = circle_500
    benchmark(morton_coverage, lats, lons, order=6)

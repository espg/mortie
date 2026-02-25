"""
Performance benchmarks for mortie (CodSpeed-compatible).

Run locally:
    pytest benchmarks/test_bench_cpu.py -v

Run with CodSpeed:
    pytest benchmarks/test_bench_cpu.py --codspeed
"""

import numpy as np
import pytest

from mortie import geo2mort, fastNorm2Mort, split_children, morton_polygon


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def coords_1k():
    rng = np.random.default_rng(42)
    lats = rng.uniform(-90, 90, 1_000)
    lons = rng.uniform(-180, 180, 1_000)
    return lats, lons


@pytest.fixture
def coords_100k():
    rng = np.random.default_rng(42)
    lats = rng.uniform(-90, 90, 100_000)
    lons = rng.uniform(-180, 180, 100_000)
    return lats, lons


@pytest.fixture
def norm_batch():
    """Normalized addresses and parents for batch encoding."""
    rng = np.random.default_rng(42)
    n = 10_000
    orders = np.full(n, 18, dtype=np.int64)
    normed = rng.integers(0, 2**36, size=n, dtype=np.int64)
    parents = rng.integers(0, 12, size=n, dtype=np.int64)
    return orders, normed, parents


@pytest.fixture
def morton_clustered():
    """Clustered morton indices for trie benchmarks."""
    rng = np.random.default_rng(42)
    c1 = -5110000 + rng.integers(0, 9999, size=3000, dtype=np.int64)
    c2 = -6130000 + rng.integers(0, 9999, size=3000, dtype=np.int64)
    c3 = -5120000 + rng.integers(0, 9999, size=2000, dtype=np.int64)
    return np.concatenate([c1, c2, c3])


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def test_geo2mort_small(benchmark, coords_1k):
    """geo2mort with 1K coordinates."""
    lats, lons = coords_1k
    benchmark(geo2mort, lats, lons, order=18)


def test_geo2mort_large(benchmark, coords_100k):
    """geo2mort with 100K coordinates."""
    lats, lons = coords_100k
    benchmark(geo2mort, lats, lons, order=18)


def test_fastNorm2Mort_batch(benchmark, norm_batch):
    """fastNorm2Mort batch encoding (10K values)."""
    orders, normed, parents = norm_batch
    benchmark(fastNorm2Mort, orders, normed, parents)


def test_morton_polygon_n4(benchmark, morton_clustered):
    """morton_polygon with n_cells=4 (bounding box)."""
    roots = split_children(morton_clustered, max_depth=3)
    benchmark(morton_polygon, roots, n_cells=4)


def test_morton_polygon_n12(benchmark, morton_clustered):
    """morton_polygon with n_cells=12 (polygon)."""
    roots = split_children(morton_clustered, max_depth=5)
    benchmark(morton_polygon, roots, n_cells=12)

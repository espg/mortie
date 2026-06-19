"""
Performance benchmarks for mortie (CodSpeed-compatible).

Run locally:
    pytest benchmarks/test_bench_cpu.py -v

Run with CodSpeed:
    pytest benchmarks/test_bench_cpu.py --codspeed
"""

import numpy as np
import pytest

from mortie import geo2mort, morton_polygon, norm2mort, split_children

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
    """Normalized addresses and parents for batch encoding (order 18)."""
    rng = np.random.default_rng(42)
    n = 10_000
    normed = rng.integers(0, 4**18, size=n, dtype=np.int64)
    parents = rng.integers(0, 12, size=n, dtype=np.int64)
    return normed, parents, 18


@pytest.fixture
def morton_clustered():
    """Clustered packed morton words for trie benchmarks.

    Three base cells, each a tight cluster of distinct order-6 cells (varying
    the in-base z-order), so the trie still sees three multi-cell groups.
    """
    rng = np.random.default_rng(42)
    order = 6
    span = 4 ** order
    c1 = norm2mort(rng.integers(0, span, size=3000, dtype=np.int64), 8, order)
    c2 = norm2mort(rng.integers(0, span, size=3000, dtype=np.int64), 10, order)
    c3 = norm2mort(rng.integers(0, span, size=2000, dtype=np.int64), 9, order)
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


def test_norm2mort_batch(benchmark, norm_batch):
    """norm2mort batch encoding (10K values)."""
    normed, parents, order = norm_batch
    benchmark(norm2mort, normed, parents, order)


def test_morton_polygon_n4(benchmark, morton_clustered):
    """morton_polygon with n_cells=4 (bounding box)."""
    roots = split_children(morton_clustered, max_depth=3)
    benchmark(morton_polygon, roots, n_cells=4)


def test_morton_polygon_n12(benchmark, morton_clustered):
    """morton_polygon with n_cells=12 (polygon)."""
    roots = split_children(morton_clustered, max_depth=5)
    benchmark(morton_polygon, roots, n_cells=12)

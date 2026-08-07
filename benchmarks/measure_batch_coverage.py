"""Measure batch vs per-call scalar-loop MOC coverage (issue #153).

The issue's acceptance criterion: at catalog scale the per-call fixed
overhead (Python->Rust boundary, allocation, wrapping) should vanish, leaving
batch wall time ~ scalar-loop rust time / cores.  This script times a Python
loop over ``morton_coverage_moc`` against one ``polygons_to_morton_mocs``
call on N synthetic ~1 degree granule-footprint quads.

Run:
    python benchmarks/measure_batch_coverage.py [N] [order]

Defaults: N=100_000 footprints, order=8.
"""

import os
import sys
import time

import numpy as np

import mortie


def footprints(n, rng):
    """N small (~1 deg) quads scattered over the mid-latitudes, ragged."""
    clat = rng.uniform(-60.0, 60.0, n)
    clon = rng.uniform(-180.0, 180.0, n)
    lats = np.column_stack(
        [clat - 0.5, clat - 0.5, clat + 0.5, clat + 0.5]
    ).ravel()
    lons = np.column_stack(
        [clon - 0.5, clon + 0.5, clon + 0.5, clon - 0.5]
    ).ravel()
    offsets = np.arange(0, 4 * n + 1, 4, dtype=np.int64)
    return lats, lons, offsets


def main():
    """Time the scalar loop vs the batch call and print the comparison."""
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    order = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    rng = np.random.default_rng(153)
    lats, lons, offsets = footprints(n, rng)

    t0 = time.perf_counter()
    values, out = mortie.polygons_to_morton_mocs(lats, lons, offsets, order=order)
    t_batch = time.perf_counter() - t0

    t0 = time.perf_counter()
    parts = [
        mortie.morton_coverage_moc(
            lats[offsets[i]:offsets[i + 1]],
            lons[offsets[i]:offsets[i + 1]],
            order=order,
        )
        for i in range(n)
    ]
    t_loop = time.perf_counter() - t0

    for i in (0, n // 2, n - 1):
        np.testing.assert_array_equal(values[out[i]:out[i + 1]], parts[i])

    cores = os.cpu_count()
    print(f"n={n} order={order} cores={cores} cells={len(values)}")
    print(f"scalar loop : {t_loop:8.2f} s  ({1e6 * t_loop / n:7.1f} us/polygon)")
    print(f"batch       : {t_batch:8.2f} s  ({1e6 * t_batch / n:7.1f} us/polygon)")
    print(f"speedup     : {t_loop / t_batch:8.2f}x")


if __name__ == "__main__":
    main()

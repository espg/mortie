"""Measure batch vs per-call scalar-loop MOC densify (issue #156).

The issue's acceptance criterion: at catalog scale the per-call fixed overhead
(Python->Rust boundary, allocation, wrapping) should vanish, leaving batch wall
time ~ scalar-loop rust time / cores.  This script times a Python loop over
``moc_to_order`` against one ``mocs_to_orders`` call on the MOCs of N synthetic
~1 degree granule-footprint quads — the same corpus
``measure_batch_coverage.py`` uses, so the two stages of the pipeline are
measured on identical inputs.

Run:
    python benchmarks/measure_mocs_to_orders.py [N] [moc_order] [flat_order]

Defaults: N=100_000 footprints, MOCs built at order 8, densified to order 9.
"""

import os
import sys
import time

import numpy as np

import mortie
from mortie.batch import _mocs_to_orders


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
    moc_order = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    flat_order = int(sys.argv[3]) if len(sys.argv) > 3 else 9
    rng = np.random.default_rng(156)
    lats, lons, off_in = footprints(n, rng)

    # Stage 1 (already batched, issue #153) builds the corpus this measures.
    mocs, offsets = mortie.polygons_to_morton_mocs(lats, lons, off_in, order=moc_order)

    t0 = time.perf_counter()
    values, out = _mocs_to_orders(mocs, offsets, flat_order)
    t_batch = time.perf_counter() - t0

    t0 = time.perf_counter()
    parts = [
        mortie.moc_to_order(mocs[offsets[i]:offsets[i + 1]], flat_order)
        for i in range(n)
    ]
    t_loop = time.perf_counter() - t0

    for i in (0, n // 2, n - 1):
        np.testing.assert_array_equal(values[out[i]:out[i + 1]], parts[i])

    cores = os.cpu_count()
    print(f"n={n} moc_order={moc_order} flat_order={flat_order} cores={cores}")
    print(f"moc cells={len(mocs)} flat cells={len(values)}")
    print(f"scalar loop : {t_loop:8.2f} s  ({1e6 * t_loop / n:7.1f} us/moc)")
    print(f"batch       : {t_batch:8.2f} s  ({1e6 * t_batch / n:7.1f} us/moc)")
    print(f"speedup     : {t_loop / t_batch:8.2f}x")


if __name__ == "__main__":
    main()

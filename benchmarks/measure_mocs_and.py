"""Measure batch vs per-call scalar-loop MOC intersection (issue #173).

The loop this replaces is zagg's stored-MOC fast path — per granule,
``moc_and(granule_moc, aoi_moc)`` — and moczarr's predicate comprehension —
``moc_and([w], aoi).size`` per item.  Both are 1 x N broadcasts: one shared
AOI operand against N small per-item covers.  The scalar rebuilds the shared
operand's BMOC (normalize + encode) on every call; ``mocs_and`` /
``mocs_intersect`` build it once, which is a term that *grows with N* on top
of the usual boundary amortization — so the gap is reported as a curve over N,
not a single ratio.

Run:
    python benchmarks/measure_mocs_and.py [N ...]      # timing sweep
    python benchmarks/measure_mocs_and.py --mem N      # one peak-RSS case

Defaults: N sweeps 1_000, 10_000, 100_000; item covers at order 6, AOI a wide
quad covered at order 8.
"""

import os
import resource
import sys
import time

import numpy as np

import mortie


def corpus(n, rng, order=6):
    """N small (~1 deg) quad covers plus offsets, and the shared AOI cover."""
    clat = rng.uniform(-60.0, 60.0, n)
    clon = rng.uniform(-180.0, 180.0, n)
    lats = np.column_stack([clat - 0.5, clat - 0.5, clat + 0.5, clat + 0.5]).ravel()
    lons = np.column_stack([clon - 0.5, clon + 0.5, clon + 0.5, clon - 0.5]).ravel()
    off_in = np.arange(0, 4 * n + 1, 4, dtype=np.int64)
    values, offsets = mortie.polygons_to_morton_mocs(lats, lons, off_in, order=order)
    aoi = mortie.morton_coverage_moc(
        [10.0, 10.0, 45.0, 45.0], [-60.0, 10.0, 10.0, -60.0], order=8
    )
    return aoi, values, offsets


def rss_mb():
    """Return peak resident set size so far, in MiB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS bytes.
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def one_size(n, rng):
    """Time both ops at one N, batch vs scalar loop, and check parity."""
    aoi, values, offsets = corpus(n, rng)

    t0 = time.perf_counter()
    out_vals, out = mortie.mocs_and(aoi, values, offsets)
    t_and_batch = time.perf_counter() - t0

    t0 = time.perf_counter()
    parts = [
        mortie.moc_and(aoi, values[offsets[i]:offsets[i + 1]]) for i in range(n)
    ]
    t_and_loop = time.perf_counter() - t0

    t0 = time.perf_counter()
    hits = mortie.mocs_intersect(aoi, values, offsets)
    t_pred_batch = time.perf_counter() - t0

    t0 = time.perf_counter()
    loop_hits = [
        mortie.moc_and(aoi, values[offsets[i]:offsets[i + 1]]).size > 0
        for i in range(n)
    ]
    t_pred_loop = time.perf_counter() - t0

    for i in (0, n // 2, n - 1):
        np.testing.assert_array_equal(out_vals[out[i]:out[i + 1]], parts[i])
        assert hits[i] == loop_hits[i]

    frac = np.mean(hits)
    print(f"n={n:>7}  hit-rate={frac:5.1%}  moc cells={len(values)}")
    print(f"  mocs_and       : {t_and_batch:8.3f} s   loop: {t_and_loop:8.3f} s"
          f"   speedup {t_and_loop / t_and_batch:6.2f}x")
    print(f"  mocs_intersect : {t_pred_batch:8.3f} s   loop: {t_pred_loop:8.3f} s"
          f"   speedup {t_pred_loop / t_pred_batch:6.2f}x")


def mem_case(n):
    """Peak RSS of one mocs_and + mocs_intersect call at N (run per process)."""
    rng = np.random.default_rng(173)
    aoi, values, offsets = corpus(n, rng)
    base = rss_mb()
    in_mb = (values.nbytes + offsets.nbytes) / (1024 * 1024)
    out_vals, out = mortie.mocs_and(aoi, values, offsets)
    hits = mortie.mocs_intersect(aoi, values, offsets)
    peak = rss_mb()
    out_mb = (out_vals.nbytes + out.nbytes + hits.nbytes) / (1024 * 1024)
    print(f"n={n} input={in_mb:.1f} MiB result={out_mb:.1f} MiB "
          f"resident-before={base:.1f} MiB peak={peak:.1f} MiB "
          f"growth={peak - base:.1f} MiB")


def main():
    """Run the timing sweep (or one --mem case) and print the comparison."""
    if len(sys.argv) > 2 and sys.argv[1] == "--mem":
        mem_case(int(sys.argv[2]))
        return
    sizes = [int(v) for v in sys.argv[1:]] or [1_000, 10_000, 100_000]
    rng = np.random.default_rng(173)
    print(f"cores={os.cpu_count()}")
    for n in sizes:
        one_size(n, rng)


if __name__ == "__main__":
    main()

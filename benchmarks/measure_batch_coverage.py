"""Measure batch vs per-call scalar-loop MOC coverage (issue #153).

The issue's acceptance criterion: at catalog scale the per-call fixed
overhead (Python->Rust boundary, allocation, wrapping) should vanish, leaving
batch wall time ~ scalar-loop rust time / cores.  This script times a Python
loop over ``morton_coverage_moc`` against one ``polygons_to_morton_mocs``
call on N synthetic ~1 degree granule-footprint quads.

Run:
    python benchmarks/measure_batch_coverage.py [N] [order]      # timing sweep
    python benchmarks/measure_batch_coverage.py --mem N [order]  # peak-memory case

Defaults: N=100_000 footprints, order=8.

``--mem`` re-runs the memory table quoted in
:func:`mortie.batch.polygons_to_morton_mocs` and in ``coverage::batch``'s
module docs, which the timing arm left as a claim on trust (issue #162
adversarial review).  It reports **input, result and peak**, since the peak is
``input copy + result + one chunk`` and quoting it against the result alone
omits the binding's ``to_vec()`` of the vertex arrays.
"""

import os
import resource
import sys
import threading
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


def rss_mb():
    """Current resident set size in MiB, or None where it is unreadable.

    ``/proc/self/statm`` field 2 is resident pages -- *current* residency, so
    sampling it across a call gives the call's true peak.  Only Linux has it;
    elsewhere the caller falls back to the ``ru_maxrss`` watermark, which is a
    lower bound (it reads as no growth whenever an earlier phase out-peaked
    the call).
    """
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2 ** 20
    except (OSError, ValueError, IndexError):
        return None


def watermark_mb():
    """Peak-RSS high-water mark so far, in MiB (Linux reports KiB, macOS bytes)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2 ** 20 if sys.platform == "darwin" else peak / 1024


def mem_case(n, order):
    """Peak of one ``polygons_to_morton_mocs`` call against input and result."""
    rng = np.random.default_rng(153)
    lats, lons, offsets = footprints(n, rng)
    in_mb = (lats.nbytes + lons.nbytes + offsets.nbytes) / 2 ** 20

    sampled = rss_mb()
    stop = threading.Event()
    peak = [sampled] if sampled is not None else []

    def sample():
        while not stop.wait(0.002):
            peak.append(rss_mb())

    base = sampled if sampled is not None else watermark_mb()
    sampler = threading.Thread(target=sample, daemon=True)
    if sampled is not None:
        sampler.start()
    values, out = mortie.polygons_to_morton_mocs(lats, lons, offsets, order=order)
    stop.set()
    if sampled is not None:
        sampler.join()
        top = max(peak)
        how = "statm sampled"
    else:
        top = watermark_mb()
        how = "ru_maxrss watermark (lower bound; no /proc/self/statm)"

    out_mb = (values.nbytes + out.nbytes) / 2 ** 20
    growth = top - base
    print(f"n={n} order={order} cells={len(values)}  [{how}]")
    print(f"input  : {in_mb:8.1f} MiB   result: {out_mb:8.1f} MiB")
    print(f"peak   : {growth:8.1f} MiB   = {growth / out_mb:5.2f}x the result "
          f"alone, {growth / (in_mb + out_mb):5.2f}x of input + result")


def main():
    """Time the scalar loop vs the batch call, or run the one --mem case."""
    if len(sys.argv) > 2 and sys.argv[1] == "--mem":
        mem_case(int(sys.argv[2]),
                 int(sys.argv[3]) if len(sys.argv) > 3 else 8)
        return
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

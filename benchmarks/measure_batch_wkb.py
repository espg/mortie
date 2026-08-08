"""Measure batch vs per-blob-loop WKB coverage (issue #157).

The issue's acceptance criterion: at catalog scale a per-blob Python loop is
dominated by the fixed cost of crossing into Rust once per blob (argument
coercion, a parse, a cover, a result wrap), and ``from_wkbs`` should pay that
once for the whole column.  This script times a Python loop over
``mortie.from_wkb(..., moc=True)`` against one ``mortie.from_wkbs`` call on N
synthetic ~1 degree granule-footprint quads, encoded as WKB.

Pass ``--peak`` to report peak RSS for the batch call instead of timings —
the memory posture the docstring claims (result + one chunk of copied input
bytes + one chunk of in-flight covers).

Run:
    python benchmarks/measure_batch_wkb.py [N] [order] [--peak]

Defaults: N=100_000 blobs, order=8.
"""

import os
import resource
import struct
import sys
import time

import numpy as np

import mortie


def footprint_blobs(n, rng):
    """Return *n* WKB Polygon blobs, ~1 degree quads over the mid-latitudes."""
    clat = rng.uniform(-60.0, 60.0, n)
    clon = rng.uniform(-180.0, 180.0, n)
    head = struct.pack("<BII", 1, 3, 1) + struct.pack("<I", 5)
    blobs = []
    for la, lo in zip(clat, clon):
        ring = (
            (lo - 0.5, la - 0.5),
            (lo + 0.5, la - 0.5),
            (lo + 0.5, la + 0.5),
            (lo - 0.5, la + 0.5),
            (lo - 0.5, la - 0.5),
        )
        blobs.append(head + b"".join(struct.pack("<dd", x, y) for x, y in ring))
    return blobs


def rss_mb():
    """Return peak resident set size so far, in MiB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS bytes.
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def main():
    """Time the per-blob loop against the batch call, or report peak RSS."""
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    peak_mode = "--peak" in sys.argv
    n = int(argv[0]) if argv else 100_000
    order = int(argv[1]) if len(argv) > 1 else 8
    rng = np.random.default_rng(157)
    blobs = footprint_blobs(n, rng)
    blob_mb = sum(len(b) for b in blobs) / (1024 * 1024)

    if peak_mode:
        base = rss_mb()
        values, offsets = mortie.from_wkbs(blobs, order=order)
        result_mb = (values.nbytes + offsets.nbytes) / (1024 * 1024)
        print(f"n={n} order={order} blobs={blob_mb:.1f} MiB")
        print(f"result      : {result_mb:8.1f} MiB")
        print(f"peak RSS    : {rss_mb():8.1f} MiB  (baseline {base:.1f} MiB)")
        print(f"growth      : {rss_mb() - base:8.1f} MiB "
              f"({(rss_mb() - base) / result_mb:.2f}x the result)")
        return

    t0 = time.perf_counter()
    values, offsets = mortie.from_wkbs(blobs, order=order)
    t_batch = time.perf_counter() - t0

    t0 = time.perf_counter()
    parts = [mortie.from_wkb(b, order=order, moc=True) for b in blobs]
    t_loop = time.perf_counter() - t0

    for i in (0, n // 2, n - 1):
        np.testing.assert_array_equal(values[offsets[i]:offsets[i + 1]], parts[i])

    cores = os.cpu_count()
    print(f"n={n} order={order} cores={cores} cells={len(values)} "
          f"blobs={blob_mb:.1f} MiB")
    print(f"per-blob loop : {t_loop:8.2f} s  ({1e6 * t_loop / n:7.1f} us/blob)")
    print(f"batch         : {t_batch:8.2f} s  ({1e6 * t_batch / n:7.1f} us/blob)")
    print(f"speedup       : {t_loop / t_batch:8.2f}x")


if __name__ == "__main__":
    main()

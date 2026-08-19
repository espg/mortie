"""Measure batch vs per-call scalar-loop child generation (issue #156).

The loop this replaces is written down in moczarr's own source —
``np.stack([generate_morton_children(int(w), level) for w in words])``
(``dggs.py:310``) — and in zagg's per-sub-chunk refinement on every worker
(``grids/healpix.py:199``).  So the scalar side timed here is exactly that
expression, ``np.stack`` included, since a caller cannot get a dense block out
of the scalar without it.

Note the result grows as ``4**d``: at large ``d`` this is a memory-bandwidth
measurement, not a boundary-cost one, and the speedup falls accordingly.  The
decay does **not** flatten inside the small-``d`` range — it keeps going, so a
figure quoted at ``d <= 4`` does not bound ``d = 8``, which is the shape zagg
ships (65,536 cells/shard).  Measure the ``d`` you intend to run.

Run:
    python benchmarks/measure_children_of.py [N] [parent_order] [target_order]

Defaults: N=100_000 order-6 parents refined to order 8 (16 children each).

Measured, 10 cores, macOS/arm64, **median of 5 runs** (single runs vary by up to
20% on a loaded box, so a one-shot number reads high as often as low):

===========================  ============  ==========  =========
shape                        result        median      range
===========================  ============  ==========  =========
100k, 6 -> 7   (d=1)              3.1 MiB     74.2x     68.5-84.8
100k, 6 -> 8   (d=2)             12.2 MiB     50.3x     48.1-55.9
100k, 6 -> 9   (d=3)             48.8 MiB     28.2x     26.3-28.8
100k, 6 -> 10  (d=4)            195.3 MiB     12.9x     12.3-13.1
2k,   6 -> 14  (d=8)           1000.0 MiB      7.4x      7.1-7.5
===========================  ============  ==========  =========
"""

import os
import sys
import time

import numpy as np

import mortie
from mortie import _rustie
from mortie.batch import _children_of


def parents(n, order, rng):
    """N random parent words at `order`, scattered over all 12 base cells."""
    span = 1 << (2 * order)
    nested = rng.integers(0, 12 * span, n).astype(np.uint64)
    depths = np.full(n, order, dtype=np.uint8)
    return np.asarray(
        _rustie.rust_nested2mort(np.ascontiguousarray(nested), depths), np.uint64
    )


def main():
    """Time the scalar loop vs the batch call and print the comparison."""
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    parent_order = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    target_order = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    rng = np.random.default_rng(156)
    words = parents(n, parent_order, rng)

    t0 = time.perf_counter()
    batch = _children_of(words, target_order)
    t_batch = time.perf_counter() - t0

    t0 = time.perf_counter()
    loop = np.stack(
        [mortie.generate_morton_children(int(w), target_order) for w in words]
    )
    t_loop = time.perf_counter() - t0

    np.testing.assert_array_equal(batch, loop)

    cores = os.cpu_count()
    width = batch.shape[1]
    mib = batch.nbytes / 2 ** 20
    print(f"n={n} {parent_order} -> {target_order} (d={target_order - parent_order}, "
          f"{width} children/parent) cores={cores}")
    print(f"result      : {mib:8.1f} MiB")
    print(f"scalar loop : {t_loop:8.2f} s  ({1e6 * t_loop / n:7.2f} us/parent)")
    print(f"batch       : {t_batch:8.2f} s  ({1e6 * t_batch / n:7.2f} us/parent)")
    print(f"speedup     : {t_loop / t_batch:8.2f}x")


if __name__ == "__main__":
    main()

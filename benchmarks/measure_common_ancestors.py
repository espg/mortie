"""Measure batch vs per-call scalar-loop common-ancestor reduction (issue #156).

The consumer shape this exists for is zagg's t-digest merge, which reduces one
tiny group per multi-member centroid inside a Python ``for`` loop
(``stats/tdigest.py:198``) — many groups, each a handful of words.  So the
default corpus is N small groups rather than a few large ones: the win being
measured is the per-call boundary cost, not the reduction itself.

Run:
    python benchmarks/measure_common_ancestors.py [N] [group_size] [order]

Defaults: N=100_000 groups of 3 order-9 words each.
"""

import os
import sys
import time

import numpy as np

import mortie
from mortie import _rustie


def groups(n, size, order, rng):
    """N groups of `size` words, each group inside one random base cell."""
    span = 1 << (2 * order)
    base = rng.integers(0, 12, n).repeat(size).astype(np.uint64)
    nested = base * np.uint64(span) + rng.integers(0, span, n * size).astype(np.uint64)
    depths = np.full(n * size, order, dtype=np.uint8)
    values = np.asarray(
        _rustie.rust_nested2mort(np.ascontiguousarray(nested), depths), np.uint64
    )
    offsets = np.arange(0, n * size + 1, size, dtype=np.int64)
    return values, offsets


def main():
    """Time the scalar loop vs the batch call and print the comparison."""
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    order = int(sys.argv[3]) if len(sys.argv) > 3 else 9
    rng = np.random.default_rng(156)
    values, offsets = groups(n, size, order, rng)

    t0 = time.perf_counter()
    batch = mortie.common_ancestors(values, offsets)
    t_batch = time.perf_counter() - t0

    t0 = time.perf_counter()
    loop = np.array(
        [
            mortie.common_ancestor(values[offsets[i]:offsets[i + 1]])
            for i in range(n)
        ],
        dtype=np.uint64,
    )
    t_loop = time.perf_counter() - t0

    np.testing.assert_array_equal(batch, loop)

    cores = os.cpu_count()
    print(f"n={n} group_size={size} order={order} cores={cores}")
    print(f"scalar loop : {t_loop:8.2f} s  ({1e6 * t_loop / n:7.2f} us/group)")
    print(f"batch       : {t_batch:8.2f} s  ({1e6 * t_batch / n:7.2f} us/group)")
    print(f"speedup     : {t_loop / t_batch:8.2f}x")


if __name__ == "__main__":
    main()

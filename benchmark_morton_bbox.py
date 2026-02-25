"""
Benchmark: Rust vs Python split_children performance.

Usage:
    python benchmark_morton_bbox.py
"""

import time
import importlib
import os
import numpy as np


def _run_split_children(morton_array, max_depth=4):
    """Import and run split_children (picks up current FORCE_PYTHON setting)."""
    import mortie.morton_bbox as mb
    importlib.reload(mb)
    return mb.split_children(morton_array, max_depth=max_depth)


def _collect_characteristics(roots):
    chars = []
    for r in roots:
        chars.append(r.characteristic)
        chars.extend(_collect_characteristics(r.children))
    return sorted(chars)


def bench(sizes=(1_000, 5_000, 10_000, 50_000, 100_000), max_depth=4, repeats=3):
    rng = np.random.default_rng(42)

    print(f"{'Size':>10s}  {'Rust (s)':>10s}  {'Python (s)':>10s}  {'Speedup':>8s}  {'Match':>5s}")
    print("-" * 55)

    for n in sizes:
        arr = rng.integers(-9_999_999_999, 9_999_999_999, size=n, dtype=np.int64)

        # --- Rust path ---
        os.environ['MORTIE_FORCE_PYTHON'] = '0'
        rust_times = []
        rust_roots = None
        for _ in range(repeats):
            t0 = time.perf_counter()
            rust_roots = _run_split_children(arr, max_depth=max_depth)
            rust_times.append(time.perf_counter() - t0)
        rust_best = min(rust_times)

        # --- Python path ---
        os.environ['MORTIE_FORCE_PYTHON'] = '1'
        py_times = []
        py_roots = None
        for _ in range(repeats):
            t0 = time.perf_counter()
            py_roots = _run_split_children(arr, max_depth=max_depth)
            py_times.append(time.perf_counter() - t0)
        py_best = min(py_times)

        # Verify parity
        rust_chars = _collect_characteristics(rust_roots)
        py_chars = _collect_characteristics(py_roots)
        match = rust_chars == py_chars

        speedup = py_best / rust_best if rust_best > 0 else float('inf')
        print(f"{n:>10,d}  {rust_best:>10.4f}  {py_best:>10.4f}  {speedup:>7.1f}x  {'OK' if match else 'FAIL'}")

    # Reset
    os.environ.pop('MORTIE_FORCE_PYTHON', None)


if __name__ == "__main__":
    bench()

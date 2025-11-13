#!/usr/bin/env python3
"""
Performance comparison: Rust vs Pure Python

Benchmarks the Rust-accelerated and pure Python implementations
of morton indexing functions.
"""

import numpy as np
import time
import os
from pathlib import Path

# Force clean imports
import sys
if 'mortie' in sys.modules:
    del sys.modules['mortie']
if 'mortie.tools' in sys.modules:
    del sys.modules['mortie.tools']

print("=" * 70)
print("RUST vs PURE PYTHON PERFORMANCE COMPARISON")
print("=" * 70)

# Benchmark 1: Scalar operations
print("\n[BENCHMARK 1] Scalar Operations")
print("-" * 70)

# Rust
os.environ.pop('MORTIE_FORCE_PYTHON', None)
import importlib
import mortie.tools
importlib.reload(mortie.tools)

iterations = 10000
start = time.perf_counter()
for i in range(iterations):
    result = mortie.tools.fastNorm2Mort(18, 1000, 2)
rust_scalar_time = (time.perf_counter() - start) / iterations

print(f"Rust:   {rust_scalar_time*1e6:.2f} µs per call")

# Pure Python
os.environ['MORTIE_FORCE_PYTHON'] = '1'
if 'mortie.tools' in sys.modules:
    del sys.modules['mortie.tools']
import mortie.tools
importlib.reload(mortie.tools)

start = time.perf_counter()
for i in range(iterations):
    result = mortie.tools.fastNorm2Mort(18, 1000, 2)
python_scalar_time = (time.perf_counter() - start) / iterations

print(f"Python: {python_scalar_time*1e6:.2f} µs per call")
print(f"Speedup: {python_scalar_time / rust_scalar_time:.2f}x")

# Benchmark 2: Small arrays (1,000 values)
print("\n[BENCHMARK 2] Small Arrays (1,000 values)")
print("-" * 70)

orders = np.full(1000, 18, dtype=np.int64)
normed = np.arange(1000, dtype=np.int64)
parents = np.array([i % 12 for i in range(1000)], dtype=np.int64)

# Rust
os.environ.pop('MORTIE_FORCE_PYTHON', None)
if 'mortie.tools' in sys.modules:
    del sys.modules['mortie.tools']
import mortie.tools

start = time.perf_counter()
rust_result = mortie.tools.fastNorm2Mort(orders, normed, parents)
rust_small_time = time.perf_counter() - start

print(f"Rust:   {rust_small_time*1000:.2f} ms ({len(normed)/rust_small_time:.0f} values/sec)")

# Pure Python
os.environ['MORTIE_FORCE_PYTHON'] = '1'
if 'mortie.tools' in sys.modules:
    del sys.modules['mortie.tools']
import mortie.tools

start = time.perf_counter()
python_result = mortie.tools.fastNorm2Mort(orders, normed, parents)
python_small_time = time.perf_counter() - start

print(f"Python: {python_small_time*1000:.2f} ms ({len(normed)/python_small_time:.0f} values/sec)")
print(f"Speedup: {python_small_time / rust_small_time:.2f}x")

# Verify correctness
if np.array_equal(rust_result, python_result):
    print("✓ Results match")
else:
    print("✗ MISMATCH!")

# Benchmark 3: Large arrays (100,000 values)
print("\n[BENCHMARK 3] Large Arrays (100,000 values)")
print("-" * 70)

orders = np.full(100000, 18, dtype=np.int64)
normed = np.arange(100000, dtype=np.int64)
parents = np.array([i % 12 for i in range(100000)], dtype=np.int64)

# Rust
os.environ.pop('MORTIE_FORCE_PYTHON', None)
if 'mortie.tools' in sys.modules:
    del sys.modules['mortie.tools']
import mortie.tools

start = time.perf_counter()
rust_result = mortie.tools.fastNorm2Mort(orders, normed, parents)
rust_large_time = time.perf_counter() - start

print(f"Rust:   {rust_large_time*1000:.2f} ms ({len(normed)/rust_large_time/1e6:.2f}M values/sec)")

# Pure Python
os.environ['MORTIE_FORCE_PYTHON'] = '1'
if 'mortie.tools' in sys.modules:
    del sys.modules['mortie.tools']
import mortie.tools

start = time.perf_counter()
python_result = mortie.tools.fastNorm2Mort(orders, normed, parents)
python_large_time = time.perf_counter() - start

print(f"Python: {python_large_time*1000:.2f} ms ({len(normed)/python_large_time/1e6:.2f}M values/sec)")
print(f"Speedup: {python_large_time / rust_large_time:.2f}x")

# Verify correctness
if np.array_equal(rust_result, python_result):
    print("✓ Results match")
else:
    print("✗ MISMATCH!")

# Benchmark 4: Antarctic polygon data (if available)
test_dir = Path("mortie/tests")
coords_file = test_dir / "Ant_Grounded_DrainageSystem_Polygons.txt"

if coords_file.exists():
    print("\n[BENCHMARK 4] Antarctic Polygon Data (1,239,001 coordinates)")
    print("-" * 70)

    data = np.loadtxt(coords_file)
    lats = data[:, 0]
    lons = data[:, 1]

    # Rust
    os.environ.pop('MORTIE_FORCE_PYTHON', None)
    if 'mortie.tools' in sys.modules:
        del sys.modules['mortie.tools']
    import mortie.tools

    start = time.perf_counter()
    rust_morton = mortie.tools.geo2mort(lats, lons, order=18)
    rust_real_time = time.perf_counter() - start

    print(f"Rust:   {rust_real_time*1000:.2f} ms ({len(lats)/rust_real_time/1e6:.2f}M coords/sec)")

    # Pure Python
    os.environ['MORTIE_FORCE_PYTHON'] = '1'
    if 'mortie.tools' in sys.modules:
        del sys.modules['mortie.tools']
    import mortie.tools

    start = time.perf_counter()
    python_morton = mortie.tools.geo2mort(lats, lons, order=18)
    python_real_time = time.perf_counter() - start

    print(f"Python: {python_real_time*1000:.2f} ms ({len(lats)/python_real_time/1e6:.2f}M coords/sec)")
    print(f"Speedup: {python_real_time / rust_real_time:.2f}x")

    # Verify correctness
    if np.array_equal(rust_morton, python_morton):
        print("✓ Results match")
    else:
        print("✗ MISMATCH!")
else:
    print("\n[BENCHMARK 4] SKIPPED: Antarctic data not found")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Scalar operations:  {python_scalar_time / rust_scalar_time:.1f}x faster with Rust")
print(f"Small arrays (1K):  {python_small_time / rust_small_time:.1f}x faster with Rust")
print(f"Large arrays (100K): {python_large_time / rust_large_time:.1f}x faster with Rust")
if coords_file.exists():
    print(f"Real-world (1.2M):  {python_real_time / rust_real_time:.1f}x faster with Rust")
print("=" * 70)

#!/usr/bin/env python3
"""
Direct comparison test: Rust vs Pure Python

This script explicitly compares Rust and pure Python implementations
to verify they produce bit-identical results.
"""

import numpy as np
import os
from pathlib import Path

# Force clean imports
import sys
if 'mortie' in sys.modules:
    del sys.modules['mortie']
if 'mortie.tools' in sys.modules:
    del sys.modules['mortie.tools']

print("=" * 70)
print("RUST VS PURE PYTHON COMPARISON TEST")
print("=" * 70)

# Test 1: Scalar comparison
print("\n[TEST 1] Scalar inputs")
print("-" * 70)

# Test with Rust
os.environ.pop('MORTIE_FORCE_PYTHON', None)
import importlib
import mortie.tools
importlib.reload(mortie.tools)
rust_scalar = mortie.tools.fastNorm2Mort(18, 1000, 2)
print(f"Rust:        fastNorm2Mort(18, 1000, 2) = {rust_scalar}")

# Test with pure Python
os.environ['MORTIE_FORCE_PYTHON'] = '1'
if 'mortie.tools' in sys.modules:
    del sys.modules['mortie.tools']
import mortie.tools
importlib.reload(mortie.tools)
python_scalar = mortie.tools.fastNorm2Mort(18, 1000, 2)
print(f"Pure Python: fastNorm2Mort(18, 1000, 2) = {python_scalar}")

if rust_scalar == python_scalar:
    print(f"✓ MATCH: Both produce {rust_scalar}")
else:
    print(f"✗ MISMATCH: Rust={rust_scalar}, Python={python_scalar}")
    sys.exit(1)

# Test 2: Array comparison
print("\n[TEST 2] Array inputs (1000 values)")
print("-" * 70)

orders = np.full(1000, 18, dtype=np.int64)
normed = np.arange(1000, dtype=np.int64)
parents = np.array([i % 12 for i in range(1000)], dtype=np.int64)

# Rust
os.environ.pop('MORTIE_FORCE_PYTHON', None)
if 'mortie.tools' in sys.modules:
    del sys.modules['mortie.tools']
import mortie.tools
rust_array = mortie.tools.fastNorm2Mort(orders, normed, parents)
print(f"Rust:        Computed {len(rust_array)} values")

# Pure Python
os.environ['MORTIE_FORCE_PYTHON'] = '1'
if 'mortie.tools' in sys.modules:
    del sys.modules['mortie.tools']
import mortie.tools
python_array = mortie.tools.fastNorm2Mort(orders, normed, parents)
print(f"Pure Python: Computed {len(python_array)} values")

if np.array_equal(rust_array, python_array):
    print(f"✓ MATCH: All {len(rust_array)} values identical")
else:
    mismatches = np.sum(rust_array != python_array)
    print(f"✗ MISMATCH: {mismatches} values differ")
    print(f"   First mismatch at index {np.where(rust_array != python_array)[0][0]}")
    sys.exit(1)

# Test 3: VaexNorm2Mort comparison
print("\n[TEST 3] VaexNorm2Mort (order 18 hardcoded)")
print("-" * 70)

normed = np.array([100, 200, 300], dtype=np.int64)
parents = np.array([2, 3, 8], dtype=np.int64)

# Rust
os.environ.pop('MORTIE_FORCE_PYTHON', None)
if 'mortie.tools' in sys.modules:
    del sys.modules['mortie.tools']
import mortie.tools
rust_vaex = mortie.tools.VaexNorm2Mort(normed, parents)
print(f"Rust:        {rust_vaex}")

# Pure Python
os.environ['MORTIE_FORCE_PYTHON'] = '1'
if 'mortie.tools' in sys.modules:
    del sys.modules['mortie.tools']
import mortie.tools
python_vaex = mortie.tools.VaexNorm2Mort(normed, parents)
print(f"Pure Python: {python_vaex}")

if np.array_equal(rust_vaex, python_vaex):
    print(f"✓ MATCH: VaexNorm2Mort results identical")
else:
    print(f"✗ MISMATCH!")
    sys.exit(1)

# Test 4: Antarctic polygon data (1.2M coordinates)
print("\n[TEST 4] Antarctic polygon data (1,239,001 coordinates)")
print("-" * 70)

test_dir = Path("mortie/tests")
coords_file = test_dir / "Ant_Grounded_DrainageSystem_Polygons.txt"

if coords_file.exists():
    data = np.loadtxt(coords_file)
    lats = data[:, 0]
    lons = data[:, 1]

    print(f"Loaded {len(lats):,} coordinates")

    # Rust
    os.environ.pop('MORTIE_FORCE_PYTHON', None)
    if 'mortie.tools' in sys.modules:
        del sys.modules['mortie.tools']
    import mortie.tools
    import time
    start = time.perf_counter()
    rust_morton = mortie.tools.geo2mort(lats, lons, order=18)
    rust_time = time.perf_counter() - start
    print(f"Rust:        {len(rust_morton):,} indices in {rust_time*1000:.2f} ms")

    # Pure Python
    os.environ['MORTIE_FORCE_PYTHON'] = '1'
    if 'mortie.tools' in sys.modules:
        del sys.modules['mortie.tools']
    import mortie.tools
    start = time.perf_counter()
    python_morton = mortie.tools.geo2mort(lats, lons, order=18)
    python_time = time.perf_counter() - start
    print(f"Pure Python: {len(python_morton):,} indices in {python_time*1000:.2f} ms")

    if np.array_equal(rust_morton, python_morton):
        print(f"✓ MATCH: All {len(rust_morton):,} morton indices identical")
        speedup = python_time / rust_time
        print(f"\nPerformance: Rust is {speedup:.1f}x faster than pure Python")
    else:
        mismatches = np.sum(rust_morton != python_morton)
        print(f"✗ MISMATCH: {mismatches:,} values differ")
        idx = np.where(rust_morton != python_morton)[0][0]
        print(f"   First mismatch at index {idx}")
        print(f"   Rust:        {rust_morton[idx]}")
        print(f"   Pure Python: {python_morton[idx]}")
        sys.exit(1)
else:
    print("⊘ SKIPPED: Antarctic polygon data not found")

# Test 5: Different orders
print("\n[TEST 5] Different tessellation orders")
print("-" * 70)

normed = np.array([100], dtype=np.int64)
parents = np.array([2], dtype=np.int64)

for order in [6, 10, 14, 18]:
    # Rust
    os.environ.pop('MORTIE_FORCE_PYTHON', None)
    if 'mortie.tools' in sys.modules:
        del sys.modules['mortie.tools']
    import mortie.tools
    rust_result = mortie.tools.fastNorm2Mort(order, normed, parents)

    # Pure Python
    os.environ['MORTIE_FORCE_PYTHON'] = '1'
    if 'mortie.tools' in sys.modules:
        del sys.modules['mortie.tools']
    import mortie.tools
    python_result = mortie.tools.fastNorm2Mort(order, normed, parents)

    if np.array_equal(rust_result, python_result):
        print(f"Order {order:2d}: ✓ MATCH ({rust_result[0]})")
    else:
        print(f"Order {order:2d}: ✗ MISMATCH (Rust={rust_result[0]}, Python={python_result[0]})")
        sys.exit(1)

print("\n" + "=" * 70)
print("ALL TESTS PASSED: Rust and pure Python produce identical results!")
print("=" * 70)

#!/usr/bin/env python3
"""
Direct comparison test: Rust vs Numba

This script explicitly compares Rust and numba implementations
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
print("RUST VS NUMBA COMPARISON TEST")
print("=" * 70)

# Test 1: Scalar comparison
print("\n[TEST 1] Scalar inputs")
print("-" * 70)

# Test with Rust
os.environ.pop('MORTIE_FORCE_NUMBA', None)
import importlib
import mortie.tools
importlib.reload(mortie.tools)
rust_scalar = mortie.tools.fastNorm2Mort(18, 1000, 2)
print(f"Rust:  fastNorm2Mort(18, 1000, 2) = {rust_scalar}")

# Test with numba
os.environ['MORTIE_FORCE_NUMBA'] = '1'
if 'mortie.tools' in sys.modules:
    del sys.modules['mortie.tools']
import mortie.tools
importlib.reload(mortie.tools)
numba_scalar = mortie.tools.fastNorm2Mort(18, 1000, 2)
print(f"Numba: fastNorm2Mort(18, 1000, 2) = {numba_scalar}")

if rust_scalar == numba_scalar:
    print(f"✓ MATCH: Both produce {rust_scalar}")
else:
    print(f"✗ MISMATCH: Rust={rust_scalar}, Numba={numba_scalar}")
    sys.exit(1)

# Test 2: Array comparison
print("\n[TEST 2] Array inputs (1000 values)")
print("-" * 70)

orders = np.full(1000, 18, dtype=np.int64)
normed = np.arange(1000, dtype=np.int64)
parents = np.array([i % 12 for i in range(1000)], dtype=np.int64)

# Rust
os.environ.pop('MORTIE_FORCE_NUMBA', None)
if 'mortie.tools' in sys.modules:
    del sys.modules['mortie.tools']
import mortie.tools
rust_array = mortie.tools.fastNorm2Mort(orders, normed, parents)
print(f"Rust:  Computed {len(rust_array)} values")

# Numba
os.environ['MORTIE_FORCE_NUMBA'] = '1'
if 'mortie.tools' in sys.modules:
    del sys.modules['mortie.tools']
import mortie.tools
numba_array = mortie.tools.fastNorm2Mort(orders, normed, parents)
print(f"Numba: Computed {len(numba_array)} values")

if np.array_equal(rust_array, numba_array):
    print(f"✓ MATCH: All {len(rust_array)} values identical")
else:
    mismatches = np.sum(rust_array != numba_array)
    print(f"✗ MISMATCH: {mismatches} values differ")
    print(f"   First mismatch at index {np.where(rust_array != numba_array)[0][0]}")
    sys.exit(1)

# Test 3: Antarctic polygon data (1.2M coordinates)
print("\n[TEST 3] Antarctic polygon data (1,239,001 coordinates)")
print("-" * 70)

test_dir = Path("mortie/tests")
coords_file = test_dir / "Ant_Grounded_DrainageSystem_Polygons.txt"

if coords_file.exists():
    data = np.loadtxt(coords_file)
    lats = data[:, 0]
    lons = data[:, 1]

    print(f"Loaded {len(lats):,} coordinates")

    # Rust
    os.environ.pop('MORTIE_FORCE_NUMBA', None)
    if 'mortie.tools' in sys.modules:
        del sys.modules['mortie.tools']
    import mortie.tools
    import time
    start = time.perf_counter()
    rust_morton = mortie.tools.geo2mort(lats, lons, order=18)
    rust_time = time.perf_counter() - start
    print(f"Rust:  {len(rust_morton):,} indices in {rust_time*1000:.2f} ms ({len(rust_morton)/rust_time:.0f} coords/sec)")

    # Numba
    os.environ['MORTIE_FORCE_NUMBA'] = '1'
    if 'mortie.tools' in sys.modules:
        del sys.modules['mortie.tools']
    import mortie.tools
    start = time.perf_counter()
    numba_morton = mortie.tools.geo2mort(lats, lons, order=18)
    numba_time = time.perf_counter() - start
    print(f"Numba: {len(numba_morton):,} indices in {numba_time*1000:.2f} ms ({len(numba_morton)/numba_time:.0f} coords/sec)")

    if np.array_equal(rust_morton, numba_morton):
        print(f"✓ MATCH: All {len(rust_morton):,} morton indices identical")
        speedup = numba_time / rust_time
        print(f"\nPerformance: Rust is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than numba")
    else:
        mismatches = np.sum(rust_morton != numba_morton)
        print(f"✗ MISMATCH: {mismatches:,} values differ")
        idx = np.where(rust_morton != numba_morton)[0][0]
        print(f"   First mismatch at index {idx}")
        print(f"   Rust:  {rust_morton[idx]}")
        print(f"   Numba: {numba_morton[idx]}")
        sys.exit(1)
else:
    print("⊘ SKIPPED: Antarctic polygon data not found")

print("\n" + "=" * 70)
print("ALL TESTS PASSED: Rust and numba produce identical results!")
print("=" * 70)

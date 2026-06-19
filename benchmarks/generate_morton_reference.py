#!/usr/bin/env python
"""
Generate reference morton indices for Ant_Grounded_DrainageSystem_Polygons.txt

This creates a compressed numpy array of morton indices that will be used
for regression testing. Any changes to morton encoding will cause tests to fail.
"""

from pathlib import Path

import numpy as np

from mortie import tools

# Paths
TEST_DIR = Path(__file__).parent / "mortie" / "tests"
INPUT_FILE = TEST_DIR / "Ant_Grounded_DrainageSystem_Polygons.txt"
OUTPUT_FILE = TEST_DIR / "Ant_Grounded_DrainageSystem_Polygons_morton.npz"

print("="*70)
print("Generating Morton Index Reference File")
print("="*70)

# Load coordinates
print(f"\nLoading coordinates from: {INPUT_FILE}")
data = np.loadtxt(INPUT_FILE)
print(f"  Shape: {data.shape}")
print(f"  Total points: {len(data):,}")

lats = data[:, 0]
lons = data[:, 1]
polygon_ids = data[:, 2].astype(np.int32)

print("\nCoordinate ranges:")
print(f"  Latitude:  [{lats.min():.2f}, {lats.max():.2f}]")
print(f"  Longitude: [{lons.min():.2f}, {lons.max():.2f}]")
print(f"  Unique polygons: {len(np.unique(polygon_ids))}")

# Generate morton indices at order 18
print("\nGenerating morton indices at order=18...")
order = 18
morton_indices = tools.geo2mort(lats, lons, order=order)

print(f"  Computed {len(morton_indices):,} morton indices")
print(f"  Range: [{morton_indices.min()}, {morton_indices.max()}]")
print(f"  Dtype: {morton_indices.dtype}")

# Validate structure via the decode-through-kernel decimal repr (issue #48):
# the packed i64 is no longer the decimal value, so structure lives in the repr.
print("\nValidating morton index structure...")
from mortie import _rustie  # noqa: E402

invalid_count = 0
sample = np.ascontiguousarray(morton_indices[:1000], dtype=np.int64)
for m, s in zip(sample, _rustie.rust_mi_decimal_repr(sample)):
    trailing = s.lstrip("-")[1:]  # drop sign + leading base-cell digit
    invalid = [d for d in trailing if d not in '1234']
    if invalid:
        invalid_count += 1
        if invalid_count < 5:  # Show first few
            print(f"  Warning: Invalid digits in morton {m}: {invalid}")

if invalid_count == 0:
    print("  ✓ All sampled morton indices have valid digit structure")
else:
    print(f"  ⚠️  Found {invalid_count}/1000 with invalid digits")

# Save compressed
print(f"\nSaving reference to: {OUTPUT_FILE}")
np.savez_compressed(
    OUTPUT_FILE,
    morton_indices=morton_indices,
    order=np.array([order]),
    metadata=np.array([
        len(morton_indices),
        lats.min(), lats.max(),
        lons.min(), lons.max()
    ])
)

# Verify we can load it back
print("\nVerifying saved file...")
loaded = np.load(OUTPUT_FILE)
assert np.array_equal(loaded['morton_indices'], morton_indices)
print("  ✓ Verification successful")

# Show file sizes
import os

input_size = os.path.getsize(INPUT_FILE) / 1024 / 1024
output_size = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
compression_ratio = input_size / output_size

print("\nFile sizes:")
print(f"  Input (txt):        {input_size:.2f} MB")
print(f"  Output (npz):       {output_size:.2f} MB")
print(f"  Compression ratio:  {compression_ratio:.2f}x")

print("\n" + "="*70)
print("Reference file generation complete!")
print("="*70)
print("\nNext step: Run regression test with:")
print("  pytest mortie/tests/test_polygon_regression.py -v -s")

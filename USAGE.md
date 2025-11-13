# Mortie Usage Guide

## Overview

Mortie provides morton indexing for HEALPix grids with automatic Rust acceleration. The library transparently uses the fastest available implementation:

1. **Rust** (default) - Pre-compiled binary extension for maximum performance
2. **Pure Python** (fallback) - Used when Rust is unavailable or explicitly requested

Both implementations produce **identical results** and have the **same API**.

## Basic Usage

### Converting Geographic Coordinates to Morton Indices

```python
import mortie.tools as mt
import numpy as np

# Single coordinate
lat, lon = -78.5, -132.0
morton = mt.geo2mort(lat, lon, order=18)
print(f"Morton index: {morton}")

# Multiple coordinates
lats = np.array([-78.5, -75.2, -80.1])
lons = np.array([-132.0, -145.5, -120.3])
morton_indices = mt.geo2mort(lats, lons, order=18)
print(f"Morton indices: {morton_indices}")
```

### Working with Normalized HEALPix Addresses

If you already have normalized HEALPix addresses and parent cells:

```python
import mortie.tools as mt
import numpy as np

# Single value
order = 18
normed = 1000
parent = 2
morton = mt.fastNorm2Mort(order, normed, parent)
print(f"Morton index: {morton}")

# Arrays
orders = np.array([18, 18, 18], dtype=np.int64)
normed = np.array([100, 200, 300], dtype=np.int64)
parents = np.array([2, 3, 8], dtype=np.int64)
morton_indices = mt.fastNorm2Mort(orders, normed, parents)
print(f"Morton indices: {morton_indices}")
```

### Vaex-Compatible Interface

For use with [Vaex](https://vaex.io/) dataframes (order hardcoded to 18):

```python
import mortie.tools as mt
import numpy as np

normed = np.array([100, 200, 300], dtype=np.int64)
parents = np.array([2, 3, 8], dtype=np.int64)
morton = mt.VaexNorm2Mort(normed, parents)
print(f"Morton indices: {morton}")
```

## Implementation Selection

### Default Behavior (Automatic)

By default, mortie automatically uses the fastest available implementation:

```python
import mortie.tools as mt

# Uses Rust if available, otherwise pure Python
morton = mt.geo2mort(-78.5, -132.0, order=18)
```

### Forcing Pure Python

To explicitly use the pure Python implementation (useful for testing or debugging):

```bash
# Set environment variable before importing
export MORTIE_FORCE_PYTHON=1
python your_script.py
```

Or in Python:

```python
import os
os.environ['MORTIE_FORCE_PYTHON'] = '1'

# Now import mortie - it will use pure Python
import mortie.tools as mt

morton = mt.geo2mort(-78.5, -132.0, order=18)
```

### Checking Which Implementation Is Active

```python
import mortie.tools as mt

# Check if Rust is available
if hasattr(mt, 'RUST_AVAILABLE'):
    if mt.RUST_AVAILABLE and not mt.FORCE_PYTHON:
        print("Using Rust implementation")
    else:
        print("Using pure Python implementation")
```

## Resolution Orders

Mortie supports tessellation orders from 1 to 18:

```python
import mortie.tools as mt

# View available resolutions
mt.res2display()

# Output:
# 6514.02758 km at tessellation order 0
# 3257.013790 km at tessellation order 1
# ...
# 0.00006361 km at tessellation order 18
```

Example with different orders:

```python
import mortie.tools as mt

lat, lon = -78.5, -132.0

# Low resolution (large cells)
morton_low = mt.geo2mort(lat, lon, order=6)   # ~407 km cells

# Medium resolution
morton_med = mt.geo2mort(lat, lon, order=12)  # ~6.3 km cells

# High resolution (small cells)
morton_high = mt.geo2mort(lat, lon, order=18) # ~64 m cells
```

## Clipping to Lower Resolutions

Convert high-resolution morton indices to lower resolutions:

```python
import mortie.tools as mt
import numpy as np

# Generate high-resolution morton indices
lats = np.array([-78.5, -75.2, -80.1])
lons = np.array([-132.0, -145.5, -120.3])
morton_18 = mt.geo2mort(lats, lons, order=18)

# Clip to order 12 (lower resolution)
morton_12 = mt.clip2order(12, morton_18)
print(f"Order 18: {morton_18}")
print(f"Order 12: {morton_12}")
```

## Performance Considerations

### When to Use Each Implementation

**Rust Implementation (Default):**
- ✅ Production workloads
- ✅ Large datasets (>1000 coordinates)
- ✅ Performance-critical applications
- ✅ Batch processing

**Pure Python Implementation:**
- ✅ Testing and validation
- ✅ Platforms without pre-built wheels
- ✅ Debugging morton encoding logic
- ✅ Understanding the algorithm

### Performance Comparison

| Dataset Size | Rust | Pure Python | Speedup |
|--------------|------|-------------|---------|
| 1,000 values | 1.93 ms | 4.14 ms | 2.1x |
| 100,000 values | 1.85 ms | 410.59 ms | 222x |
| 1.2M coordinates | 102.51 ms | 5.1 sec | 50x |

For small datasets (<100 values), the performance difference is minimal. For large datasets (>10,000 values), Rust provides dramatic speedups.

## API Reference

### `geo2mort(lats, lons, order=18)`

Convert geographic coordinates to morton indices.

**Parameters:**
- `lats` (float or array): Latitude(s) in degrees
- `lons` (float or array): Longitude(s) in degrees
- `order` (int): Tessellation order (1-18), default=18

**Returns:**
- Morton index/indices as int64

### `fastNorm2Mort(order, normed, parents)`

Convert normalized HEALPix addresses to morton indices.

**Parameters:**
- `order` (int or array): Tessellation order (1-18)
- `normed` (int or array): Normalized HEALPix address
- `parents` (int or array): Parent base cell (0-11)

**Returns:**
- Morton index/indices as int64

### `VaexNorm2Mort(normed, parents)`

Convert normalized HEALPix addresses to morton indices at order 18 (Vaex-compatible).

**Parameters:**
- `normed` (int or array): Normalized HEALPix address
- `parents` (int or array): Parent base cell (0-11)

**Returns:**
- Morton index/indices as int64

### `clip2order(clip_order, midx=None, print_factor=False)`

Clip morton indices to lower resolution.

**Parameters:**
- `clip_order` (int): Target resolution order
- `midx` (array): Morton indices to clip
- `print_factor` (bool): If True, return scaling factor instead of clipped values

**Returns:**
- Clipped morton indices or scaling factor

### `order2res(order)`

Calculate approximate resolution in km for a given order.

**Parameters:**
- `order` (int): Tessellation order

**Returns:**
- Resolution in kilometers (float)

### `res2display()`

Print resolution table for all tessellation orders (0-19).

## Advanced Usage

### Integration with Vaex DataFrames

```python
import vaex
import mortie.tools as mt

# Create a Vaex dataframe
df = vaex.from_arrays(
    lat=[-78.5, -75.2, -80.1],
    lon=[-132.0, -145.5, -120.3]
)

# Add morton indices as a virtual column
# Note: This requires the full geo2mort workflow
# For Vaex, you'd typically use VaexNorm2Mort after computing normed addresses
```

### Working with HEALPix Unique Identifiers

```python
import mortie.tools as mt
import numpy as np

# Convert UNIQ identifiers to morton indices
uniq = np.array([1234567890, 2345678901, 3456789012], dtype=np.int64)
parents = mt.unique2parent(uniq)

# Then use with normalized addresses
# (requires computing normed from uniq first via healpy)
```

## Troubleshooting

### "Cannot import mortie_rs" Error

This means the Rust extension is not available. Mortie will automatically fall back to pure Python. To verify:

```python
import mortie.tools as mt
print(f"Rust available: {mt.RUST_AVAILABLE}")
```

To build the Rust extension locally, see [BUILDING.md](BUILDING.md).

### Different Results Between Implementations

This should **never** happen. Both implementations are tested to produce bit-identical results. If you encounter different outputs, please [file an issue](https://github.com/espg/mortie/issues).

### Performance Issues

If you're experiencing slow performance:

1. Check which implementation is active:
   ```python
   import mortie.tools as mt
   print(f"Rust: {mt.RUST_AVAILABLE}, Force Python: {mt.FORCE_PYTHON}")
   ```

2. Ensure `MORTIE_FORCE_PYTHON=1` is not set in your environment

3. Verify you're using arrays (not lists) for large datasets:
   ```python
   # Good (NumPy array)
   lats = np.array([...])
   morton = mt.geo2mort(lats, lons, order=18)

   # Slower (Python list, gets converted internally)
   lats = [...]
   morton = mt.geo2mort(lats, lons, order=18)
   ```

## Examples

### Example 1: Processing Antarctic Data

```python
import mortie.tools as mt
import numpy as np

# Load Antarctic coordinate data
data = np.loadtxt('antarctica_coords.txt')
lats = data[:, 0]
lons = data[:, 1]

# Generate morton indices at high resolution
morton_indices = mt.geo2mort(lats, lons, order=18)

# Create a spatial index (example)
unique_cells = np.unique(morton_indices)
print(f"Data spans {len(unique_cells)} unique morton cells")
```

### Example 2: Multi-Resolution Analysis

```python
import mortie.tools as mt
import numpy as np

lats = np.array([-78.5, -75.2, -80.1])
lons = np.array([-132.0, -145.5, -120.3])

# Generate indices at multiple resolutions
for order in [6, 10, 14, 18]:
    morton = mt.geo2mort(lats, lons, order=order)
    res = mt.order2res(order)
    print(f"Order {order:2d} (~{res:8.2f} km): {morton}")
```

### Example 3: Benchmarking

```python
import mortie.tools as mt
import numpy as np
import time

# Generate test data
n = 100000
lats = np.random.uniform(-90, 90, n)
lons = np.random.uniform(-180, 180, n)

# Benchmark
start = time.perf_counter()
morton = mt.geo2mort(lats, lons, order=18)
elapsed = time.perf_counter() - start

print(f"Processed {n:,} coordinates in {elapsed*1000:.2f} ms")
print(f"Throughput: {n/elapsed/1e6:.2f} M coords/sec")
```

## Further Reading

- [BUILDING.md](BUILDING.md) - Build instructions for Rust extension
- [Youngren & Petty (2017)](https://doi.org/10.1016/j.heliyon.2017.e00332) - Multi-resolution HEALPix paper
- [HEALPix](https://healpix.jpl.nasa.gov/) - Hierarchical Equal Area isoLatitude Pixelization
- [Morton Ordering](https://en.wikipedia.org/wiki/Z-order_curve) - Z-order curve on Wikipedia

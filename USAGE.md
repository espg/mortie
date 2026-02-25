# Mortie Usage Guide

## Overview

Mortie provides morton indexing for HEALPix grids using a Rust-accelerated extension for high performance. The Rust extension is required and is included in all pip-installed wheels.

## Basic Usage

### Converting Geographic Coordinates to Morton Indices

```python
from mortie import geo2mort
import numpy as np

# Single coordinate
lat, lon = -78.5, -132.0
morton = geo2mort(lat, lon, order=18)
print(f"Morton index: {morton}")

# Multiple coordinates
lats = np.array([-78.5, -75.2, -80.1])
lons = np.array([-132.0, -145.5, -120.3])
morton_indices = geo2mort(lats, lons, order=18)
print(f"Morton indices: {morton_indices}")
```

### Working with Normalized HEALPix Addresses

If you already have normalized HEALPix addresses and parent cells:

```python
from mortie import fastNorm2Mort
import numpy as np

# Single value
order = 18
normed = 1000
parent = 2
morton = fastNorm2Mort(order, normed, parent)
print(f"Morton index: {morton}")

# Arrays
orders = np.array([18, 18, 18], dtype=np.int64)
normed = np.array([100, 200, 300], dtype=np.int64)
parents = np.array([2, 3, 8], dtype=np.int64)
morton_indices = fastNorm2Mort(orders, normed, parents)
print(f"Morton indices: {morton_indices}")
```

### Vaex-Compatible Interface

For use with [Vaex](https://vaex.io/) dataframes (order hardcoded to 18):

```python
from mortie import VaexNorm2Mort
import numpy as np

normed = np.array([100, 200, 300], dtype=np.int64)
parents = np.array([2, 3, 8], dtype=np.int64)
morton = VaexNorm2Mort(normed, parents)
print(f"Morton indices: {morton}")
```

## Resolution Orders

Morton encoding supports tessellation orders from 1 to 18. The `res2display()` function shows all orders 0-19 for reference:

```python
from mortie import res2display

# View available resolutions
res2display()

# Output:
# 6514.02758 km at tessellation order 0
# 3257.013790 km at tessellation order 1
# ...
# 0.00006361 km at tessellation order 18
```

Example with different orders:

```python
from mortie import geo2mort

lat, lon = -78.5, -132.0

# Low resolution (large cells)
morton_low = geo2mort(lat, lon, order=6)   # ~407 km cells

# Medium resolution
morton_med = geo2mort(lat, lon, order=12)  # ~6.3 km cells

# High resolution (small cells)
morton_high = geo2mort(lat, lon, order=18) # ~64 m cells
```

## Clipping to Lower Resolutions

Convert high-resolution morton indices to lower resolutions:

```python
from mortie import geo2mort, clip2order
import numpy as np

# Generate high-resolution morton indices
lats = np.array([-78.5, -75.2, -80.1])
lons = np.array([-132.0, -145.5, -120.3])
morton_18 = geo2mort(lats, lons, order=18)

# Clip to order 12 (lower resolution)
morton_12 = clip2order(12, morton_18)
print(f"Order 18: {morton_18}")
print(f"Order 12: {morton_12}")
```

## Morton Polygon / Bounding Box

Use `morton_polygon` (or `geo_morton_polygon`) to find the fewest prefix-cells
that span a set of morton indices:

```python
from mortie import geo_morton_polygon, split_children, morton_polygon
import numpy as np

lats = np.array([-75, -75, -70, -70, -72])
lons = np.array([-80, -70, -70, -80, -75])

# Bounding box (4 cells)
bbox_cells = geo_morton_polygon(lats, lons, n_cells=4, order=18)

# Tighter polygon (12 cells)
poly_cells = geo_morton_polygon(lats, lons, n_cells=12, order=18)

# Or from morton indices directly:
morton_indices = np.array([-5111131, -5111132, -5111133], dtype=np.int64)
roots = split_children(morton_indices)
refined = morton_polygon(roots, n_cells=4)
```

## Performance Considerations

### Performance Comparison

| Dataset Size | Rust | Python (reference) | Speedup |
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

### `split_children(morton_array, max_depth=4)`

Build a compacted prefix trie over morton indices.

**Parameters:**
- `morton_array` (array): Morton indices (signed integers)
- `max_depth` (int or None): Maximum branching depth (default 4)

**Returns:**
- List of `MortonChild` root-level nodes

### `morton_polygon(roots, n_cells)`

Greedily expand trie nodes to minimize area within a cell budget.

**Parameters:**
- `roots` (list of MortonChild): From `split_children()`
- `n_cells` (int): Maximum cells (4 = bounding box, 12 = polygon)

**Returns:**
- List of `MortonChild` refined prefix-cells

### `morton_buffer(morton_indices, k=1)`

Compute the k-cell border around a set of morton indices.

Returns only cells NOT in the input set (the expansion ring).

**Parameters:**
- `morton_indices` (array-like): Morton indices, all at the same order
- `k` (int): Border width in cells (default 1). k=1 gives immediate 8-connected neighbors, k=2 gives a 2-cell ring, etc.

**Returns:**
- Sorted NumPy array of border morton indices

**Raises:**
- `ValueError` if indices have mixed orders or k is out of range

### `geo_morton_polygon(lats, lons, n_cells, order=18, max_depth=None)`

Geographic convenience wrapper for `split_children` + `morton_polygon`.

**Parameters:**
- `lats`, `lons` (array): Coordinates in degrees
- `n_cells` (int): Maximum cells
- `order` (int): Tessellation order (default 18)
- `max_depth` (int or None): Trie depth (auto-derived if None)

**Returns:**
- List of `MortonChild` refined prefix-cells

## Advanced Usage

### Integration with Vaex DataFrames

```python
import vaex
from mortie import VaexNorm2Mort

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
from mortie import unique2parent
import numpy as np

# Convert UNIQ identifiers to morton indices
uniq = np.array([1234567890, 2345678901, 3456789012], dtype=np.int64)
parents = unique2parent(uniq)

# Then use with normalized addresses
# Then use with normalized addresses
```

## Troubleshooting

### Extension fails to load

If the Rust extension fails to load, try reinstalling:

```bash
pip install --force-reinstall mortie
```

To build the Rust extension locally, see [BUILDING.md](BUILDING.md).

### Performance Issues

If you're experiencing slow performance, verify you're using arrays (not lists) for large datasets:
   ```python
   # Good (NumPy array)
   lats = np.array([...])
   morton = geo2mort(lats, lons, order=18)

   # Slower (Python list, gets converted internally)
   lats = [...]
   morton = geo2mort(lats, lons, order=18)
   ```

## Examples

### Example 1: Processing Antarctic Data

```python
from mortie import geo2mort
import numpy as np

# Load Antarctic coordinate data
data = np.loadtxt('antarctica_coords.txt')
lats = data[:, 0]
lons = data[:, 1]

# Generate morton indices at high resolution
morton_indices = geo2mort(lats, lons, order=18)

# Create a spatial index (example)
unique_cells = np.unique(morton_indices)
print(f"Data spans {len(unique_cells)} unique morton cells")
```

### Example 2: Multi-Resolution Analysis

```python
from mortie import geo2mort, order2res
import numpy as np

lats = np.array([-78.5, -75.2, -80.1])
lons = np.array([-132.0, -145.5, -120.3])

# Generate indices at multiple resolutions
for order in [6, 10, 14, 18]:
    morton = geo2mort(lats, lons, order=order)
    res = order2res(order)
    print(f"Order {order:2d} (~{res:8.2f} km): {morton}")
```

### Example 3: Benchmarking

```python
from mortie import geo2mort
import numpy as np
import time

# Generate test data
n = 100000
lats = np.random.uniform(-90, 90, n)
lons = np.random.uniform(-180, 180, n)

# Benchmark
start = time.perf_counter()
morton = geo2mort(lats, lons, order=18)
elapsed = time.perf_counter() - start

print(f"Processed {n:,} coordinates in {elapsed*1000:.2f} ms")
print(f"Throughput: {n/elapsed/1e6:.2f} M coords/sec")
```

### Example 4: Morton Polygon

```python
from mortie import geo2mort, geo_morton_polygon
import numpy as np

# Antarctic flight line coordinates
lats = np.random.uniform(-80, -70, 5000)
lons = np.random.uniform(-140, -120, 5000)

# Get bounding box (4 prefix-cells)
bbox = geo_morton_polygon(lats, lons, n_cells=4, order=18)
print(f"Bounding box: {[c.characteristic for c in bbox]}")

# Get tighter polygon (12 prefix-cells)
poly = geo_morton_polygon(lats, lons, n_cells=12, order=18)
print(f"Polygon: {[c.characteristic for c in poly]}")
```

### Example 5: Spatial Buffer

```python
from mortie import geo2mort, clip2order, morton_buffer
import numpy as np

# Antarctic flight line at order 18, clipped to order 6
lats = np.random.uniform(-85, -70, 10000)
lons = np.random.uniform(-180, 180, 10000)
morton_18 = geo2mort(lats, lons, order=18)
cells_o6 = np.unique(clip2order(6, morton_18))

# Expand by 1-cell border to capture edge cells
border = morton_buffer(cells_o6, k=1)
expanded = np.union1d(cells_o6, border)

print(f"Original: {len(cells_o6)} cells")
print(f"Border:   {len(border)} new cells")
print(f"Expanded: {len(expanded)} total cells")
```

## Further Reading

- [BUILDING.md](BUILDING.md) - Build instructions for Rust extension
- [Youngren & Petty (2017)](https://doi.org/10.1016/j.heliyon.2017.e00332) - Multi-resolution HEALPix paper
- [HEALPix](https://healpix.jpl.nasa.gov/) - Hierarchical Equal Area isoLatitude Pixelization
- [Morton Ordering](https://en.wikipedia.org/wiki/Z-order_curve) - Z-order curve on Wikipedia

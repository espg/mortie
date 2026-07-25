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

If you already have a normalized HEALPix address and parent cell, `norm2mort`
packs it into a morton word (the exact inverse of `mort2norm`):

```python
from mortie import norm2mort
import numpy as np

# Single value (normed, parent, order)
normed = 1000
parent = 2
order = 18
morton = norm2mort(normed, parent, order)
print(f"Morton index: {morton}")

# Arrays of (normed, parent) at a fixed order
normed = np.array([100, 200, 300], dtype=np.int64)
parents = np.array([2, 3, 8], dtype=np.int64)
morton_indices = norm2mort(normed, parents, order)
print(f"Morton indices: {morton_indices}")
```

`norm2mort` reaches order 29 (the packed `decimal_morton` kernel's maximum).
The returned `int64` is the packed word (bit-reinterpreted; the prefix is
`base+1`, so the word is negative for base cells 7-11), not a human-readable
decimal — use `MortonIndexArray.decimal_repr()` for the readable form.

## Resolution Orders

Morton encoding supports tessellation orders from 0 to 29. The `res2display()` function shows orders 0-19 for reference:

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

# Or from morton indices directly (packed words, e.g. from geo2mort):
morton_indices = geo2mort(lats, lons, order=18)
roots = split_children(morton_indices)
refined = morton_polygon(roots, n_cells=4)
```

## Polygon Coverage

`morton_coverage` / `morton_coverage_moc` cover a polygon (given by lat/lon
vertices) with HEALPix cells, via a top-down hierarchical descent. Unlike the
bounding-box helpers above, these return the cells that actually intersect the
polygon.

```python
import mortie

lats = [40.0, 40.0, 50.0, 50.0]
lons = [-125.0, -115.0, -115.0, -125.0]

# Flat cover: every cell at the requested order
cells = mortie.morton_coverage(lats, lons, order=10)

# Multi-Order Coverage: coarse interior + fine boundary (usually far smaller)
moc = mortie.morton_coverage_moc(lats, lons, order=10)

# Approximate / adaptive boundary (cheaper, fewer cells)
moc_tol = mortie.morton_coverage_moc(lats, lons, order=10, tolerance=0.5)  # degrees
moc_bud = mortie.morton_coverage_moc(lats, lons, order=10, max_cells=500)

# Multipart + holes: pass a list of rings (even-odd fill)
donut = mortie.morton_coverage([outer_lat, hole_lat], [outer_lon, hole_lon], order=8)

# MOC <-> flat
flat = mortie.moc_to_order(moc, 10)         # densify back to a single order
compact = mortie.compress_moc(flat)         # merge 4-sibling groups
```

See [docs/coverage_methods.md](docs/coverage_methods.md) for the full
method/precision/runtime trade-offs and a benchmark matrix.

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
- `order` (int): Tessellation order (1-29), default=18

**Returns:**
- Morton index/indices as int64

### `norm2mort(normed, parent, order)`

Pack a normalized HEALPix address + base cell into a morton word (the exact
inverse of `mort2norm`).

**Parameters:**
- `normed` (int or array): Normalized HEALPix address (`0 <= normed < 4**order`)
- `parent` (int or array): Parent base cell (0-11)
- `order` (int): Tessellation order (0-29)

**Returns:**
- Packed morton word(s) as int64

### `clip2order(clip_order, midx=None, print_factor=False)`

Coarsen packed morton words to a lower resolution (kernel coarsen).

**Parameters:**
- `clip_order` (int): Target resolution order
- `midx` (array): Packed morton words to coarsen
- `print_factor` (bool): If True, return the level count dropped from order 18
  (`18 - clip_order`) instead of coarsening

**Returns:**
- Coarsened morton words or the level count

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

### `morton_coverage(lats, lons, order=18)`

Cells covering a polygon, as a **flat** sorted array at `order` (hierarchical
descent; contract: a cell is included iff it intersects the closed polygon).

**Parameters:**
- `lats`, `lons` (array, or **list of rings** for multipart/holes): vertices in degrees
- `order` (int): HEALPix order (1–29)

**Returns:**
- Sorted 1-D `int64` array of morton indices at `order`

### `morton_coverage_moc(lats, lons, order=18, tolerance=None, max_cells=None)`

Compact **Multi-Order Coverage** of a polygon (coarse interior, fine boundary).
The result is a plain `int64` array (each morton index self-encodes its order).

**Parameters:**
- `lats`, `lons`: as above (list of rings → multipart/holes, even-odd fill)
- `order` (int): finest HEALPix order
- `tolerance` (float or None): stop refining a boundary cell once its angular
  radius (degrees) drops below this — approximate, coarser boundary
- `max_cells` (int or None): best-first budget; refine the largest boundary cells
  until about this many cells (adaptive boundary). `tolerance`/`max_cells` are
  mutually exclusive; a too-low `max_cells` is raised with a warning.

**Returns:**
- Sorted 1-D `int64` array of mixed-order morton indices

### `compress_moc(morton)`

Collapse a morton set to its canonical compact MOC (merge any 4 complete sibling
cells into their parent; drop any cell contained in a coarser one). Lossless.

### `moc_to_order(morton, order)`

Densify a (mixed-order) morton set to a flat list at `order`.

## Advanced Usage

### Integration with DataFrames

```python
import vaex
from mortie import geo2mort

# Create a Vaex dataframe
df = vaex.from_arrays(
    lat=[-78.5, -75.2, -80.1],
    lon=[-132.0, -145.5, -120.3]
)

# Add morton indices as a column via the geo2mort workflow
df["morton"] = geo2mort(df.lat.values, df.lon.values, order=18)
```

For a first-class column type — a pandas `ExtensionArray` and a pyarrow
`ExtensionType` that carry the `morton_index` identity through DataFrames and
parquet, with a decimal-Morton repr and order-aware accessors — see
[docs/morton_index_datatype.md](docs/morton_index_datatype.md) (pandas / pyarrow
are optional extras; numpy stays the only runtime dependency).

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

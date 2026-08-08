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

Morton encoding supports tessellation orders from 0 to 29. `res2display()` returns
the resolution ladder as records — one `ResolutionLevel(order, value, unit, km)`
per order, with `value`/`unit` the display pair and `km` the unrounded resolution:

```python
from mortie import res2display

levels = res2display()

levels[0]
# ResolutionLevel(order=0, value=6519.623, unit='km', km=6519.623461602107)

# Format them however you like:
for lvl in res2display(max_order=2):
    print(f"{lvl.value} {lvl.unit} at tessellation order {lvl.order}")
# 6519.623 km at tessellation order 0
# 3259.812 km at tessellation order 1
# 1629.906 km at tessellation order 2

# The unit ladder switches to m below 1 km and cm below 1 m:
levels[18]
# ResolutionLevel(order=18, value=24.87, unit='m', km=0.024870389791878156)
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

### `clip2order(clip_order, midx)`

Coarsen packed morton words to a lower resolution (kernel coarsen).

**Parameters:**
- `clip_order` (int): Target resolution order
- `midx` (array): Packed morton words to coarsen

**Returns:**
- Coarsened morton words, one per input word

> The `print_factor` flag was removed for the 1.x freeze. It returned
> `18 - clip_order`, a level count anchored to the retired decimal encoding's
> order-18 ceiling, so it went negative for the order-19..29 words this package
> now encodes. The levels a word actually drops is `order - clip_order` against
> its own decoded order, available from `orders_of()`.

### `order2res(order)`

Calculate approximate resolution in km for a given order.

**Parameters:**
- `order` (int): Tessellation order

**Returns:**
- Resolution in kilometers (float)

### `res2display(max_order=29)`

Return the resolution ladder for tessellation orders `0..max_order` as a list of
`ResolutionLevel(order, value, unit, km)` named tuples. `value`/`unit` are the
display pair (km, m or cm, rounded to three decimals within the bracket); `km` is
the unrounded resolution for arithmetic.

**Returns:**
- `list[ResolutionLevel]`

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

### `morton_coverage(lats, lons, order=18, normalize=True)`

Cells covering a polygon, as a **flat** sorted array at `order` (hierarchical
descent; contract: a cell is included iff it intersects the closed polygon).

**Parameters:**
- `lats`, `lons` (array, or **list of rings** for multipart/holes): vertices in degrees
- `order` (int): HEALPix order (1–29)
- `normalize` (bool): auto-correct ring orientation at ingest (default `True`) —
  any simple ring whose interior decisively reads as the larger region is
  reversed, so CW and CCW spellings give the same cover (S2's convention).
  Pass `False` to trust the supplied winding exactly; that is the only way a
  lone ring expresses an interior larger than its complement.

**Returns:**
- Sorted 1-D `int64` array of morton indices at `order`

### `morton_coverage_moc(lats, lons, order=18, tolerance=None, max_cells=None, normalize=True)`

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
- `normalize` (bool): as `morton_coverage` above (default `True`)

**Returns:**
- Sorted 1-D `int64` array of mixed-order morton indices

### `polygons_to_morton_mocs(lats, lons, offsets, order=18, tolerance=None, max_cells=None, normalize=True)`

**Batch** MOC coverage of many independent polygons in one call — one MOC per
input polygon (result `i` is byte-identical to `morton_coverage_moc` on polygon
`i`). The whole ragged set crosses into Rust once, the GIL is released, and the
covers run in parallel across polygons.

**Parameters:**
- `lats`, `lons`: flat `float64` vertices in degrees, all rings concatenated
- `offsets` (array): `int64` arrow list offsets — polygon `i` is
  `lats[offsets[i]:offsets[i+1]]`, one **ring** per entry (no multipart/holes:
  decompose such a footprint yourself and cover it with `morton_coverage_moc`'s
  list-of-rings form). The offsets must exactly cover the vertex arrays
  (`offsets[0] == 0` and `offsets[-1] == len(lats)`); re-base a sliced arrow
  array's offsets first, as `mortie.arrow.polygons_to_morton_mocs` does.
- `order`, `tolerance`, `max_cells`, `normalize`: as `morton_coverage_moc`
  above, each a single shared setting applied to every polygon

**Returns:**
- `(values, out_offsets)`: all MOC words concatenated (`uint64`) plus the
  `int64` offsets into them — polygon `i`'s MOC is
  `values[out_offsets[i]:out_offsets[i+1]]`

```python
lats = np.array([40., 50., 45., 10., 20., 15.])
lons = np.array([-120., -120., -110., -80., -80., -70.])
values, off = mortie.polygons_to_morton_mocs(lats, lons, [0, 3, 6], order=8)
first = values[off[0]:off[1]]        # MOC of the first triangle
```

The Arrow-native spelling — a `list<struct<lat, lon>>` column in, a
`morton_index`-typed `ListArray` out (parquet-ready) — is
`mortie.arrow.polygons_to_morton_mocs`.

### `compress_moc(morton)`

Collapse a morton set to its canonical compact MOC (merge any 4 complete sibling
cells into their parent; drop any cell contained in a coarser one). Lossless.

### `moc_to_order(morton, order)`

Densify a (mixed-order) morton set to a flat list at `order`. Guarded
pre-emptively: `max_cells` (default `1 << 20`) refuses a densify whose estimated
flat cell count would exceed it, before allocating. `max_cells=None` opts out.

### `mocs_to_orders(values, offsets, order, max_cells=1 << 20)`

Densify **many** MOCs in one call — the ragged batch twin of `moc_to_order`.
One Python↔Rust crossing, GIL released, rayon across MOCs; slice `i` of the
result is byte-identical to `moc_to_order` on MOC `i` alone (sorted-unique, so a
downstream `np.unique` is redundant). The budget applies per MOC and names the
lowest-index offender.

Ragged in, ragged out, in the same arrow list layout `polygons_to_morton_mocs`
returns — so the two chain with no marshalling:

```python
mocs, off = mortie.polygons_to_morton_mocs(lats, lons, [0, 3, 6], order=8)
flat, flat_off = mortie.mocs_to_orders(mocs, off, 8)
first = flat[flat_off[0]:flat_off[1]]    # flat cover of the first triangle
```

### `common_ancestors(values, offsets)`

Reduce **many** groups of words to their deepest common ancestors in one call —
the batch twin of `common_ancestor` / `moc_min`. Ragged in (the same arrow list
layout), **dense out**: one `uint64` per group, because the reduction is
many→one per group. Result `i` is bit-identical to `common_ancestor` on group
`i` alone; an empty group is an error naming its index, since the scalar refuses
empty input.

```python
kids = np.concatenate([
    np.asarray(mortie.norm2mort([11 * 4 + s for s in range(4)], [0] * 4, 5)),
    np.asarray(mortie.norm2mort([7 * 4 + s for s in range(4)], [3] * 4, 5)),
])
parents = mortie.common_ancestors(kids, [0, 4, 8])    # -> 2 order-4 words
```

### `children_of(words, order)`

Refine **many** parents to their children at `order` — the batch twin of
`generate_morton_children`, which takes a single parent. Every parent must sit
at one order `p <= order`, so each yields `4**d` children for `d = order - p`
and the result is a dense `(n, 4**d)` block rather than a ragged pair. Row `i`
is bit-identical to `generate_morton_children(words[i], order)`.

```python
parents = np.asarray(mortie.norm2mort([11, 7], [0, 3], 4), dtype=np.uint64)
kids = mortie.children_of(parents, 6)     # shape (2, 16)
```

Size it before you call it: the result is `n * 4**d * 8` bytes, and there is no
budget guard (the scalar has none either).

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

UNIQ is the MOC cell number `4 * 4**order + nested`. It is self-describing —
the order is recoverable from the value — so the decoders read it from the
data rather than taking it as an argument, and mixed-resolution arrays work
throughout.

```python
import numpy as np
from mortie import geo2uniq, uniq2geo, unique2parent

lats = np.array([45.0, -33.9, 64.1])
lons = np.array([-122.7, 151.2, -21.9])

# Encode at one resolution...
geo2uniq(lats, lons, order=9)
# array([1683881, 3530931, 2051535])

# ...or one resolution per element.
uniq = geo2uniq(lats, lons, order=np.array([6, 12, 20]))
# array([26310, 225979639, 8604763086340])

# Decoders take no order: they read it back out of each value.
unique2parent(uniq)     # array([2, 9, 3])  -- parent base cells
lat, lon = uniq2geo(uniq)   # cell centres, element by element
```

`order` defaults to `MAX_ORDER` (29) on both encoders (`geo2uniq`,
`norm2uniq`). Note that UNIQ carries **no point/area kind** — there is no kind
bit in `4 * 4**order + nested` — so an order-29 UNIQ is the max-resolution
*area* cell containing the coordinate, not a point. For point semantics from
lat/lon use `geo2mort(lats, lons)` or
`MortonIndexArray.from_latlon(lats, lons, points=True)`, which return order-29
`Kind::Point` packed words.

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

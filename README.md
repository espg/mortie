mortie
======

[![Tests](https://github.com/espg/mortie/actions/workflows/test.yml/badge.svg)](https://github.com/espg/mortie/actions/workflows/test.yml)
[![CodSpeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge/json/espg/mortie&style=flat)](https://codspeed.io/espg/mortie?utm_source=badge)
[![codecov](https://codecov.io/gh/espg/mortie/branch/main/graph/badge.svg)](https://codecov.io/gh/espg/mortie)
[![PyPI version](https://badge.fury.io/py/mortie.svg)](https://badge.fury.io/py/mortie)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CodSpeed Badge](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/espg/mortie?utm_source=badge)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/espg/mortie/HEAD?labpath=examples%2Fmorton_set_algebra.ipynb)

![Morty using mortie](./morty.jpg)

Mortie is a library for applying morton indexing to healpix grids. Morton
numbering (also called z-ordering) facilitates several geospatial operators
such as buffering and neighborhood look-ups, and can generally be thought of as
a type of geohashing.

This particular implementation focuses on hierarchical healpix maps, and is
mostly inspired from [this paper](https://doi.org/10.1016/j.heliyon.2017.e00332).

The normative encoding and conventions — the packed-word bit layout, the
decimal string grammar, the order 0–29 resolution table, the morton-hive
store layout, and the coverage-MOC serializations, all frozen for the 1.x
series — are documented in
[docs/specification.md](docs/specification.md).

## Performance

Mortie uses **Rust-accelerated** morton indexing functions for high performance. The Rust implementation provides dramatic speedups:

| Dataset Size | Rust | Python (reference) | Speedup |
|--------------|------|--------------------|---------|
| 1,000 values | 1.93 ms | 4.14 ms | **2.1x** |
| 100,000 values | 1.85 ms | 410.59 ms | **222x** |
| 1.2M coordinates | 102.51 ms | 5.1 sec | **50x** |

Pre-built wheels are available for Linux, macOS, and Windows. The Rust extension is required and is included in all pip-installed wheels.

## Installation

```bash
pip install mortie
```

For development builds with Rust, see [BUILDING.md](BUILDING.md).

## Spatial Buffer

Mortie provides a `morton_buffer` function for expanding a set of morton cells by a configurable border ring. This is useful for... well, buffering.

```python
import numpy as np
import mortie

# Convert coordinates to morton cells at order 6
cells = np.unique(mortie.geo2mort(lats, lons, order=6))

# Expand by 1-cell ring (8-connected neighbors)
border = mortie.morton_buffer(cells, k=1)
expanded = np.union1d(cells, border)
```

All input indices must be at the same order. The function returns only the new border cells, not the input cells themselves.

## Polygon Coverage

`morton_coverage` computes the set of morton indices that cover a polygon defined by lat/lon vertices. It uses a **top-down hierarchical descent** over the HEALPix tree: starting from the 12 base cells it keeps cells inside the polygon, prunes cells outside, and refines cells the boundary passes through down to the requested order. Cost scales with the polygon's *boundary*, not its area — interior regions collapse to a few coarse cells, so a large but simple polygon is cheap. Vertex count still matters (a one-time O(V) edge/seed setup, plus per-boundary-cell work that grows with local edge density), but far more gently than the old `O(cells × vertices)` approach — a 1M-vertex polygon covers in ~1 s, roughly 40× faster than before.

```python
import mortie

# Define polygon vertices (lat, lon in degrees)
lats = [40.0, 40.0, 50.0, 50.0]
lons = [-125.0, -115.0, -115.0, -125.0]

# Flat cover — every cell at order 6
cells = mortie.morton_coverage(lats, lons, order=6)

# Compact Multi-Order Coverage — coarse interior, fine boundary (usually far smaller)
moc = mortie.morton_coverage_moc(lats, lons, order=10)

# Adaptive boundary: stop at an angular tolerance, or cap the cell count
moc_tol = mortie.morton_coverage_moc(lats, lons, order=10, tolerance=0.5)   # degrees
moc_bud = mortie.morton_coverage_moc(lats, lons, order=10, max_cells=500)
```

The function handles concave polygons, antimeridian-crossing polygons, and polar regions. **Multipart polygons and holes** are supported by passing a list of rings (even-odd fill): disjoint parts are unioned and a nested ring carves a hole, so a donut is `[outer, hole]`. Helpers `compress_moc` (merge 4-sibling groups) and `moc_to_order` (densify a MOC to a flat order) round out the API. See [docs/coverage_methods.md](docs/coverage_methods.md) for the full method/precision/runtime trade-offs and a benchmark matrix.

## Dependencies

**numpy**. All HEALPix operations use the Rust-native `healpix` crate bundled in the compiled extension — no external HEALPix library is needed.

## Funding
Initial funding of this work was supported by the ICESat-2 project science
office, at the Laboratory for Cryospheric Sciences (NASA Goddard, Section 615).

## References
<a id="1">[1]</a>
Youngren, Robert W., and Mikel D. Petty.
"A multi-resolution HEALPix data structure for spherically mapped point data."
Heliyon 3.6 (2017): e00332. [doi: 10.1016/j.heliyon.2017.e00332](https://doi.org/10.1016/j.heliyon.2017.e00332)

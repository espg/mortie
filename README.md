mortie
======

[![Tests](https://github.com/espg/mortie/actions/workflows/test.yml/badge.svg)](https://github.com/espg/mortie/actions/workflows/test.yml)
[![CodSpeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge/json/espg/mortie&style=flat)](https://codspeed.io/espg/mortie?utm_source=badge)
[![codecov](https://codecov.io/gh/espg/mortie/branch/main/graph/badge.svg)](https://codecov.io/gh/espg/mortie)
[![PyPI version](https://badge.fury.io/py/mortie.svg)](https://badge.fury.io/py/mortie)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CodSpeed Badge](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/espg/mortie?utm_source=badge)

![Morty using mortie](./morty.jpg)

Mortie is a library for applying morton indexing to healpix grids. Morton
numbering (also called z-ordering) facilitates several geospatial operators
such as buffering and neighborhood look-ups, and can generally be thought of as
a type of geohashing.

This particular implementation focuses on hierarchical healpix maps, and is
mostly inspired from [this paper](https://doi.org/10.1016/j.heliyon.2017.e00332).

## Performance

Mortie uses **Rust-accelerated** morton indexing functions for high performance, with an automatic fallback to pure Python if Rust is unavailable. The Rust implementation provides dramatic speedups:

| Dataset Size | Rust | Pure Python | Speedup |
|--------------|------|-------------|---------|
| 1,000 values | 1.93 ms | 4.14 ms | **2.1x** |
| 100,000 values | 1.85 ms | 410.59 ms | **222x** |
| 1.2M coordinates | 102.51 ms | 5.1 sec | **50x** |

Pre-built wheels are available for Linux, macOS, and Windows. If a wheel is unavailable for your platform, mortie will automatically use the pure Python fallback.

## Installation

```bash
pip install mortie
```

For development builds with Rust, see [BUILDING.md](BUILDING.md).

## Spatial Buffer

Mortie provides a `morton_buffer` function for expanding a set of morton cells by a configurable border ring. This is useful for capturing edge cells missed by sparse vertex sampling (e.g., near HEALPix pole holes).

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

mortie
======

[![Tests](https://github.com/espg/mortie/actions/workflows/test.yml/badge.svg)](https://github.com/espg/mortie/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/espg/mortie/branch/main/graph/badge.svg)](https://codecov.io/gh/espg/mortie)
[![PyPI version](https://badge.fury.io/py/mortie.svg)](https://badge.fury.io/py/mortie)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

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

TODO:

- [x] add paper reference
- [x] add funding information
- [ ] add tests
- [x] remove / prune dead code
- [ ] add example(s)
- [x] fix north / south bug
- [x] remove numba dependency
- [ ] update documentation
- [x] publish to pypi

Dependencies are **numpy** and **healpy**. The Rust-accelerated functions are optional - if unavailable, mortie will automatically fall back to a pure Python implementation. Although not a dependency, there are several functions that have been written to interface with the vaex project.

## Funding
Initial funding of this work was supported by the ICESat-2 project science
office, at the Laboratory for Cryospheric Sciences (NASA Goddard, Section 615). 

## References
<a id="1">[1]</a> 
Youngren, Robert W., and Mikel D. Petty. 
"A multi-resolution HEALPix data structure for spherically mapped point data." 
Heliyon 3.6 (2017): e00332. [doi: 10.1016/j.heliyon.2017.e00332](https://doi.org/10.1016/j.heliyon.2017.e00332)

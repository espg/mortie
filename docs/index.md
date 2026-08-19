# mortie

Mortie is a library for applying morton indexing to healpix grids. Morton
numbering (also called z-ordering) facilitates several geospatial operators such
as buffering and neighborhood look-ups, and can generally be thought of as a
type of geohashing.

This particular implementation focuses on hierarchical healpix maps, and is
mostly inspired from [this paper](https://doi.org/10.1016/j.heliyon.2017.e00332).

The morton core is a Rust extension and the sole runtime path — there is no
Python implementation to fall back on. Encoding (`geo2mort`) and decoding
(`mort2geo`) run at tens of millions of morton indices per second on one core.
`numpy` is the only runtime dependency; all HEALPix operations use the
Rust-native `healpix` crate bundled in the compiled extension.

## Installation

```bash
pip install mortie
```

Pre-built wheels are available for Linux, macOS, and Windows; the Rust
extension is included in all of them. For development builds from source, see
[BUILDING.md](https://github.com/espg/mortie/blob/main/BUILDING.md).

## Quick start

```python
import numpy as np
import mortie

# Encode lat/lon to packed morton words at order 6
cells = np.unique(mortie.geo2mort(lats, lons, order=6))

# Expand by a 1-cell ring (8-connected neighbours); returns only the new border
border = mortie.morton_buffer(cells, k=1)

# Cover a polygon — flat at a single order, or as a compact multi-order coverage
cover = mortie.morton_coverage(poly_lats, poly_lons, order=6)
moc, _ = mortie.polygons_to_morton_mocs(poly_lats, poly_lons,
                                        [0, len(poly_lats)], order=10)
```

> **Latitude convention.** Since 0.10 `geo2mort` and the coverage kernels take
> WGS84 **geodetic** latitude and map it to the authalic sphere, so cells are
> equal-area on the ellipsoid. Cell ids therefore differ from pre-0.10 mortie
> and from raw-spherical HEALPix libraries; pass
> `latitude="geodetic-spherical"` for the old behaviour. See
> [specification.md](specification.md#latitude-convention) §9.

## Documentation

- **[Specification](specification.md)** — the normative encoding and
  conventions frozen for the 1.x series: the packed 64-bit word layout, the
  decimal string grammar, the order 0–29 resolution table, the zarr DGGS
  convention block, the morton-hive store layout, and the coverage-MOC
  serializations.
- **[Morton index datatype](morton_index_datatype.md)** — the optional pandas
  `ExtensionArray` and its dtype.
- **[HEALPix interchange](healpix_interchange.md)** — moving a packed word to
  and from the wider HEALPix ecosystem (`cdshealpix` / `healpy`
  `(order, nested-pixel)` pairs).
- **[Arrow interchange](arrow_interchange.md)** — the pyarrow `ExtensionType`
  and the library-agnostic Arrow C Data Interface surface.
- **[Coverage methods](coverage_methods.md)** — polygon coverage and MOC
  method, precision, and runtime trade-offs.
- **[Benchmarks](benchmarks.md)** — the cross-order throughput table, regenerated
  in place by a committed script.

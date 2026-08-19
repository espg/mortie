# mortie.Moc — the coverage object

`mortie.moc(...)` builds a `Moc`: a multi-order coverage as an object, so that
coverage geometry reads as geometry.

```python
from mortie import moc

cali = moc(cali_geojson)     # multi-order coverage; no order argument
q    = moc(aoi_geojson)
assert cali.contains(q)
q9 = q.to_order(9)           # fixed-order cast when a consumer's grid wants one
```

## The two-layer rule

mortie's coverage surface is two layers and stays that way:

- **The kernel functions are the array/batch layer.** The free `moc_*` functions
  on [mortie MOC kernel](moc.md) are words in, words out, unchanged and
  un-deprecated, and the plural forms in [mortie.batch](batch.md) (`mocs_and`,
  `mocs_intersect`, `mocs_to_orders`, `polygons_to_morton_mocs`) stay
  function-shaped permanently — an offset-packed many-cover operation has no
  natural `self`. Array-first consumers keep calling these directly, at zero
  wrapping cost.
- **The object is ergonomics.** `Moc` is a thin view over the canonical `uint64`
  word array, never a new representation: **every method is a single delegation
  to a kernel function**. The array stays the interchange format —
  `Moc.__morton_moc__()` hands the canonical words back, and any object exposing
  that dunder is accepted wherever a `Moc` is.

## MOCpy crosswalk

HEALPix-MOC users already know these names, so the object mirrors them where
they apply. The word encoding is mortie's own frozen grammar
([specification](../specification.md) §1/§4) either way — only the vocabulary is
shared.

| MOCpy | mortie object | mortie kernel |
| --- | --- | --- |
| `MOC.from_polygon(lon, lat, max_depth=…)` | `Moc.from_polygon(lats, lons)`, or `moc(geojson)` | `morton_coverage_moc(lats, lons, order=…)` |
| `a.union(b)`, `a \| b` | `a.union(b)`, `a \| b` | `moc_or(a, b)` |
| `a.intersection(b)`, `a & b` | `a.intersection(b)`, `a & b` | `moc_and(a, b)` |
| `a.difference(b)`, `a - b` | `a.difference(b)`, `a - b` | `moc_minus(a, b)` |
| `a.symmetric_difference(b)` | `a.symmetric_difference(b)`, `a ^ b` | `moc_xor(a, b)` |
| `b.difference(a).empty()` | `a.contains(b)`, `b.within(a)` | `moc_minus(b, a).size == 0` |
| `a.contains_lonlat(lon, lat)` | — (kernel only) | `moc_intersects(a, geo2mort(lat, lon, order))` |
| — | `a.intersects(b)` | `moc_intersects(a, b)` |
| `a.degrade_to_order(n).flatten()` | `a.to_order(n)` | `moc_to_order(a, n)` |
| `a.complement()` | — (kernel only) | `moc_not(a, domain)` |
| `a.max_order` | `repr(a)` | `orders_of(a).max()` |

Three places the vocabulary matches but the meaning does not:

- **`from_polygon` takes its coordinates the other way round.** MOCpy is
  `MOC.from_polygon(lon, lat, …)`; mortie is `Moc.from_polygon(lats, lons, …)`.
  Same name, swapped order — transpose it and you get a cover somewhere else
  entirely.
- **MOCpy has no MOC-in-MOC `contains`.** `MOC.contains(lon, lat, …)` is a
  point-in-MOC mask (and is deprecated in favour of `contains_lonlat` /
  `contains_skycoords`); the MOCpy spelling of mortie's `a.contains(b)` is
  `b.difference(a).empty()`.
- **`a.to_order(n)` is not `degrade_to_order` and not `flatten`.**
  `degrade_to_order(n)` returns a coarsened *MOC* and `flatten()` takes no
  order, so the MOCpy equivalent is the pair `degrade_to_order(n).flatten()`.
  That covers the coarsening direction only: `to_order(n)` also **densifies** when
  `n` is finer than the cover, which MOCpy has no single call for.

Two mortie-specific notes. `a.to_order(n)` returns the flat **array**, not a `Moc`: a
single-order cell list is not a MOC, and re-normalizing it would collapse it
straight back to the compact form. And the predicates are *cover* algebra, not
*polygon* algebra — the conservative-direction table below says which way each
answer can err near a boundary.

::: mortie.moc_object
    options:
      members:
        - Moc

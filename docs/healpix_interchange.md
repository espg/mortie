# HEALPix-ecosystem interchange

A `mortie` packed 64-bit `morton_index` word *is* a HEALPix NESTED cell with its
order carried intrinsically (see [specification.md](specification.md) §1). This
guide shows how to move a word to and from the two common HEALPix carriers —
`(order, nested-pixel-id)` pairs as used by
[`cdshealpix`](https://cds-astro.github.io/cds-healpix-python/) and
[`healpy`](https://healpy.readthedocs.io/) — with runnable, cross-checked
snippets, then covers the three gotchas that bite at the boundary.

mortie does not reimplement the HEALPix geometry: its NESTED transforms
(`geo2mort`, `mort2geo`, `mort2healpix`, `norm2mort`, and the coverage kernels)
wrap the Rust [`healpix`](https://github.com/matt-cornell/healpix-rs) crate
(`matt-cornell/healpix-rs`, pinned in `Cargo.toml`) — the same crate used in
production for transforms to and from NESTED and the other wrapped HEALPix
routines. It is the only HEALPix implementation in the runtime path; there is no
C HEALPix library. `cdshealpix` and `healpy` appear below **only as independent
external oracles** — a second, unrelated implementation to cross-check that the
interchange is exact — not as runtime dependencies.

Only **numpy** is a `mortie` runtime dependency (the `healpix` crate above is
compiled into the extension, not a Python dependency). `cdshealpix` (with its
`astropy` dependency) ships in the `test` extra — it is the same oracle the
test-suite cross-checks against (`mortie/tests/test_morton_index.py`,
`test_coverage_hemisphere.py`). `healpy` is **not** in any `mortie` extra;
install it separately (`pip install healpy`) to run the healpy snippets below.

## The conversion API

| direction | function | returns |
|---|---|---|
| word → HEALPix | `mort2healpix(morton)` | `(nested_cell_id, order)` |
| word → normalized | `mort2norm(morton)` | `(normed, base_cell, order)` |
| normalized → word | `norm2mort(normed, base_cell, order)` | packed `uint64` word |
| order of a word | `infer_order_from_morton(morton)` | `int` |

`mort2healpix` is the one-call bridge to the ecosystem: it returns the NESTED
`ipix` and the `order` (the HEALPix `depth`) that every other library keys on.
The nested id is the standard `base_cell * nside**2 + normed` composition with
`nside = 2**order`, so `mort2norm` / `norm2mort` are just the split/join around
that same relation.

```python
import numpy as np
import mortie

# geo2mort takes (lat, lon) and returns a length-1 array for scalar input,
# so index [0] for a single word.
m = mortie.geo2mort(-80.0, 120.0, order=6)[0]
print(int(m))                        # 11570310392668225542

cell_id, order = mortie.mort2healpix(m)
print(cell_id, order)                # 37010 6

normed, base_cell, order = mortie.mort2norm(m)
print(normed, base_cell, order)      # 146 9 6

# nested id is base_cell * nside**2 + normed
nside = 2 ** order
print(cell_id == base_cell * nside**2 + normed)   # True

# round-trip back to the exact same word
m2 = mortie.norm2mort(normed, base_cell, order)
print(np.uint64(m2) == m)            # True

print(mortie.infer_order_from_morton(m))          # 6
```

For an array of words `mort2healpix` returns `(cell_ids_array, order)` with a
single `order` when the words share one order (it raises on mixed orders — pass
one order at a time).

## Round-trip against cdshealpix

`cdshealpix.nested.lonlat_to_healpix(lon, lat, depth)` is the NESTED-`ipix`
oracle; `healpix_to_lonlat(ipix, depth)` inverts it. The cell ids match
`mort2healpix` exactly, and the centers agree to floating-point tolerance.

```python
import numpy as np
import astropy.units as u
from cdshealpix.nested import lonlat_to_healpix, healpix_to_lonlat
import mortie

lats = np.array([0.0, 41.8, -41.8, 80.0, -80.0, 12.3, -67.9])
lons = np.array([0.0, 45.0, 135.0, 200.0, 305.0, 91.5, 270.2])
order = 14

words = mortie.geo2mort(lats, lons, order=order)
cell_ids, o = mortie.mort2healpix(words)

oracle = np.asarray(lonlat_to_healpix(lons * u.deg, lats * u.deg, depth=order),
                    dtype=np.int64)
print(np.array_equal(cell_ids, oracle))     # True
# cell_ids == [1275068416, 67108860, 2617245699, 800177021,
#              2962305922, 1556551876, 2997005193]

# centers, mortie vs cdshealpix
clon, clat = healpix_to_lonlat(cell_ids, depth=order)
mlat, mlon = mortie.mort2geo(words)
print(float(np.max(np.abs(mlat - clat.to_value(u.deg)))))   # ~1.98e-09
print(float(np.max(np.abs(mlon - clon.to_value(u.deg)))))   # 0.0
```

The `cell_ids` list and the tolerance figures above are captured from the
current build — regenerate them by rerunning the snippet rather than editing
them by hand.

Note the argument **order**: `geo2mort` takes `(lat, lon)`, while the cdshealpix
calls take `(lon, lat)` — a frequent source of transposed results.

### Importing a foreign nested id

To build a `mortie` word from a NESTED id produced elsewhere, split it into
`(normed, base_cell)` at that order and call `norm2mort`:

```python
foreign = int(oracle[3])            # 800177021, a cdshealpix NESTED ipix at order 14
nside_sq = (2 ** order) ** 2
base_cell, normed = divmod(foreign, nside_sq)
word = mortie.norm2mort(normed, base_cell, order)
print(mortie.mort2healpix(word))    # (800177021, 14)
```

## Round-trip against healpy

`healpy` keys on `nside = 2**order` (not the order itself) and needs
`nest=True`; pass `lonlat=True` to give it degrees in `(lon, lat)` order.

```python
import numpy as np
import healpy as hp
import mortie

lats = np.array([0.0, 41.8, -41.8, 80.0, -80.0, 12.3, -67.9])
lons = np.array([0.0, 45.0, 135.0, 200.0, 305.0, 91.5, 270.2])
order = 14
nside = 2 ** order

words = mortie.geo2mort(lats, lons, order=order)
cell_ids, _ = mortie.mort2healpix(words)

hpix = hp.ang2pix(nside, lons, lats, nest=True, lonlat=True)
print(np.array_equal(cell_ids, hpix))       # True

hlon, hlat = hp.pix2ang(nside, cell_ids, nest=True, lonlat=True)
mlat, mlon = mortie.mort2geo(words)
print(float(np.max(np.abs(mlat - hlat))))    # ~1.98e-09
```

## Gotchas at the boundary

### 1. The 4-bit prefix is `base_cell + 1`, and the word is unsigned

The top nibble of a word stores the HEALPix base cell as `base + 1` (so the 12
base cells occupy `1..=12`, and `0` is the empty/null sentinel). Base cells
**7–11 set bit 63**, making the word a large unsigned value. Store and exchange
the word as `uint64`; reinterpreting it as signed `int64` makes base cells
7–11 read back negative and corrupts the id.

```python
import numpy as np
import mortie

w = mortie.geo2mort(-80.0, 120.0, order=3)[0]   # lands in a southern base cell
print(int(w))                           # 11565243843087433731
print(int(w) >> 60)                     # 10   (prefix nibble == base_cell + 1)
_, base_cell, _ = mortie.mort2norm(w)
print(base_cell)                        # 9
print(bool(int(w) >> 63 & 1))           # True  (bit 63 set)
print(np.int64(w))                      # -6881500230622117885  <- WRONG if read signed
```

The "negative = southern" signed form only exists in the decimal *string*
presentation ([specification.md](specification.md) §2); it is never a storage
form. Keep interchange columns `uint64`.

### 2. Order is self-encoded — don't carry it out of band

The order is intrinsic to the word (`infer_order_from_morton` reads it back), so
a word never needs a companion order column. But every ecosystem call *does*
need the order/`depth`/`nside` explicitly — always pass the `order` that
`mort2healpix` returned, and never mix orders in a single `mort2healpix` array
call (it raises on mixed orders).

### 3. Points decode at order 29

A bare `geo2mort(lat, lon)` (no `order`) encodes an order-29 **point** — a
location with no area claim ([specification.md](specification.md) §1). It still
converts cleanly, but always lands at order 29:

```python
p = mortie.geo2mort(-80.0, 120.0)[0]        # order-29 point word
print(mortie.mort2healpix(p))               # (2604365600141906640, 29)
print(mortie.infer_order_from_morton(p))    # 29
```

If you need an area cell at a specific resolution for interchange, pass an
explicit `order` to `geo2mort`. The empty/null sentinel (`geo2mort` of a
non-finite coordinate) is the all-zero word `0`, which is not a valid HEALPix
cell — filter it before handing ids to an ecosystem library.

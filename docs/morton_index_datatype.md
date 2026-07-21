# The `morton_index` datatype

`mortie` ships a first-class datatype for its packed 64-bit decimal-Morton MOC
words: a pandas **`ExtensionArray`** (`MortonIndexArray` / `MortonIndexDtype`,
registered as `"morton_index"`) and a matching pyarrow **`ExtensionType`**. Both
are thin *skins* over the same Rust kernel — storage is the raw `uint64` packed
word (bit layout `[4-bit prefix | 54-bit body | 6-bit suffix]`), so wrapping a
column in the datatype is zero-copy and every domain operation delegates to the
vectorized Rust bindings. The datatype gives a column three things a bare
`uint64` array does not: a self-describing dtype that survives a DataFrame /
parquet round-trip, a **decimal-Morton repr** (`-31123` rather than the raw
word), and order-aware accessors (`.order()`, `.coarsen(k)`, …).

## Optional dependencies — numpy stays the only runtime dep

The library's sole hard runtime dependency is **numpy**. pandas and pyarrow are
**optional extras**: `import mortie` succeeds with neither installed, and the
extension classes are built *lazily* on first use.

```sh
pip install "mortie[pandas]"    # the MortonIndexArray ExtensionArray
pip install "mortie[pyarrow]"   # the pyarrow ExtensionType
pip install "mortie[pandas,pyarrow]"   # both
```

The names are exposed through a module-level `__getattr__`, so touching one
without its extra installed raises a clear `ImportError` rather than failing at
import time:

```python
# in a numpy-only environment:
import mortie
mortie.MortonIndexArray          # -> ImportError:
# "the morton_index ExtensionArray requires pandas; install it with
#  `pip install mortie[pandas]` (or `pip install pandas`)"

from mortie import arrow
arrow.morton_index_type()        # -> ImportError:
# "the morton_index Arrow extension type requires pyarrow; install it with
#  `pip install mortie[pyarrow]` (or `pip install pyarrow`)"
```

Every example below therefore requires the relevant extra. The pandas examples
assume `mortie[pandas]`; the Arrow examples assume `mortie[pyarrow]`.

## The pandas `ExtensionArray`

*Requires `mortie[pandas]`.*

### Construction

`MortonIndexArray` builds from HEALPix NESTED ids, from lat/lon, or from
already-packed `uint64` words:

```python
import numpy as np
import pandas as pd
from mortie import MortonIndexArray

# from NESTED ids at a fixed HEALPix order (depth)
arr = MortonIndexArray.from_nested([0, 1, 2, 3], depth=1)
print(repr(arr))
# MortonIndexArray([11, 12, 13, 14], len=4, order=1)

# from lat/lon in degrees (default order=29; here order=6)
pts = MortonIndexArray.from_latlon([40.0, -33.9], [-105.0, 151.2], order=6)
print(repr(pts))
# MortonIndexArray([3223213, -4243113], len=2, order=6)
```

The raw constructor (and `from_words`) expects the **packed `uint64` word**, not
the human-readable decimal id — pull the storage back out with `np.asarray(...,
dtype=np.uint64)` and it round-trips exactly:

```python
words = np.asarray(arr, dtype=np.uint64)   # the packed uint64 storage
print(words)
# [1152921504606846977 1441151880758558721 1729382256910270465 2017612633061982209]

back = MortonIndexArray.from_words(words)
print(repr(back))
# MortonIndexArray([11, 12, 13, 14], len=4, order=1)
```

Wrap it in a `Series` either from an existing `MortonIndexArray` or by naming the
registered dtype (packed words in, `"morton_index"` out):

```python
s = pd.Series(arr)
print(s.dtype, s.dtype.name)              # morton_index morton_index

s = pd.Series(words, dtype="morton_index")
print(s)
# 0    11
# 1    12
# 2    13
# 3    14
# dtype: morton_index
```

### What you see — the decimal repr

The array and its elements *display* as the decimal-Morton id, while the stored
value stays the packed word. Indexing yields a `MortonIndexScalar` (a
`numpy.uint64` subclass): it compares, hashes, and `int()`s as the raw word, but
`str`/`repr`/`format` render the decimal id. This is what makes an
`f"{shard_key}"` print `11` instead of `1152921504606846977`:

```python
elt = arr[0]
print(type(elt).__name__)   # MortonIndexScalar
print(elt)                  # 11                    (decimal-Morton id)
print(int(elt))             # 1152921504606846977   (raw packed word)
```

To materialize the decimal ids as plain strings, use `decimal_repr()` (a list)
or `to_decimal()` (a fixed-width `"<U32"` numpy array):

```python
print(arr.decimal_repr())   # ['11', '12', '13', '14']
print(arr.to_decimal())     # ['11' '12' '13' '14']   dtype='<U32'
```

### Order-aware accessors

Because each word self-encodes its HEALPix order, the array exposes order and
base-cell accessors that delegate to the kernel. The singular forms (`.order()`,
`.base_cell()`) return one value and *raise* on a mixed array, steering you to
the plural forms:

```python
print(arr.orders())          # [1 1 1 1]   per-element order (uint8)
print(arr.order())           # 1           single shared order
print(arr.is_fixed_order())  # True
print(arr.base_cells())      # [0 0 0 0]

mixed = MortonIndexArray(np.concatenate([np.asarray(arr), np.asarray(pts)]))
print(repr(mixed))
# MortonIndexArray([11, 12, 13, 14, 3223213, -4243113], len=6, order=mixed)
print(mixed.is_fixed_order())  # False
mixed.order()
# ValueError: array holds mixed orders; use .orders() for the per-element
#             orders or .coarsen(k) to cast to a fixed order
```

`coarsen(k)` rewrites every word to order `k` (a new array; elements already at
or below `k` are unchanged) — the way to cast a mixed array to a fixed order:

```python
fine = MortonIndexArray.from_nested([100, 101, 102, 103], depth=4)
print(repr(fine))            # MortonIndexArray([12321, 12322, 12323, 12324], len=4, order=4)
print(repr(fine.coarsen(2))) # MortonIndexArray([123, 123, 123, 123], len=4, order=2)
```

### Comparisons and sorting are the Z-order

The packed word is unsigned, so the *raw `uint64` order is the Morton Z-order* —
comparisons and `argsort` operate on the words directly, with no special casing
across the base-cell boundary. Equality is bit-identity. **No arithmetic
operators are defined** (see [What is and isn't supported](#what-is-and-isnt-supported-in-1x)):

```python
b = MortonIndexArray.from_nested([3, 0, 2, 1], depth=1)
print(repr(b))          # MortonIndexArray([14, 11, 13, 12], len=4, order=1)
print(np.argsort(b))    # [1 3 2 0]   -> sorts to 11, 12, 13, 14 (Z-order)
print(b == arr)         # [False False  True False]
```

### Missing values

The missing value is `pd.NA`, stored as the kernel's all-zero **empty sentinel**
word (prefix `0`). NA-bearing construction and `isna()` round-trip through it,
and the empty sentinel renders as `<NA>`:

```python
na = MortonIndexArray._from_sequence(
    [int(words[0]), pd.NA, int(words[2])], dtype=arr.dtype
)
print(repr(na))       # MortonIndexArray([11, <NA>, 13], len=3, order=mixed)
print(na.isna())      # [False  True False]
```

## The pyarrow `ExtensionType`

*Requires `mortie[pyarrow]`.*

`mortie.arrow` wraps the same `uint64` words in a pyarrow `ExtensionType`
(extension name `mortie.morton_index`) so a column keeps its `morton_index`
identity through IPC / parquet. Convert with `from_morton_index` /
`to_morton_index`:

```python
import pyarrow as pa
from mortie import MortonIndexArray
from mortie import arrow as marrow

arr = MortonIndexArray.from_nested([0, 1, 2, 3], depth=1)

pa_arr = marrow.from_morton_index(arr)
print(pa_arr.type)                 # extension<mortie.morton_index<MortonIndexType>>

back = marrow.to_morton_index(pa_arr)
print(repr(back))                  # MortonIndexArray([11, 12, 13, 14], len=4, order=1)
```

The pandas empty sentinel maps to an Arrow **null** (a real validity bitmap) and
back, so `isna()` survives the round-trip in both directions:

```python
w = np.asarray(arr)
na = MortonIndexArray.from_words(np.array([int(w[0]), 0, int(w[2])], dtype=np.uint64))
pa_na = marrow.from_morton_index(na)
print(pa_na.null_count)                       # 1
print(marrow.to_morton_index(pa_na).isna())   # [False  True False]
```

### Parquet round-trip and `table.to_pandas()`

The extension name rides along through parquet, and pandas' `__from_arrow__`
hook lands the column back as a `MortonIndexArray` (not a bare int column):

```python
import pyarrow.parquet as pq

tbl = pa.table({"cell": marrow.from_morton_index(arr)})
pq.write_table(tbl, "cells.parquet")

rt = pq.read_table("cells.parquet")
print(rt.schema.field("cell").type)   # extension<mortie.morton_index<MortonIndexType>>

df = rt.to_pandas()
print(df["cell"].dtype)               # morton_index
```

Because `MortonIndexArray` implements the Arrow **C Data Interface**
(`__arrow_c_array__`), you can also hand it straight to `pa.array(...)`:

```python
print(pa.array(arr).type)   # extension<mortie.morton_index<MortonIndexType>>
```

Note the reverse convenience does **not** exist: `pa.Table.from_pandas(df)` on a
`morton_index` column raises `ArrowTypeError` (the pandas array does not
implement `__arrow_array__`). Convert explicitly with `from_morton_index`, or use
`pa.array(series.array)` via the C-Data path shown above.

For the library-agnostic, **pyarrow-free** C Data Interface surface
(`export_c_array` / `import_c_array`, for arro3-core / polars on workers without
pyarrow), see [`arrow_interchange.md`](arrow_interchange.md).

## numpy ↔ pandas ↔ arrow round-trip

The three representations are lossless conversions of the same packed words:

| from → to | how |
|---|---|
| numpy `uint64` → array | `MortonIndexArray.from_words(words)` |
| array → numpy `uint64` | `np.asarray(arr, dtype=np.uint64)` |
| array → pandas `Series` | `pd.Series(arr)` |
| array → Arrow | `mortie.arrow.from_morton_index(arr)` / `pa.array(arr)` |
| Arrow → array | `mortie.arrow.to_morton_index(pa_arr)` / `MortonIndexArray.from_arrow(a)` |
| Arrow table → pandas | `table.to_pandas()` (lands as `morton_index`) |

NESTED ids and lat/lon are one-way *entry points* (`from_nested`,
`from_latlon`); `to_nested()` returns `(nested_ids, depths)` back out.

## What is and isn't supported in 1.x

The datatype deliberately ships a **narrow, total** surface for 1.x:

- **Supported:** construction, storage/round-trip (numpy ↔ pandas ↔ arrow ↔
  parquet), NA/sentinel semantics, the decimal repr, ordering/equality
  comparisons and `argsort` (the Z-order), and the domain accessors
  (`.orders()`/`.order()`, `.base_cells()`/`.base_cell()`, `.is_fixed_order()`,
  `.coarsen(k)`, `.to_nested()`, `.hive_path()`).
- **Not supported:** **arithmetic operators.** Add/subtract/multiply and the
  like are *not* defined, because raw arithmetic on a packed word (prefix + body
  + suffix bit-fields) is meaningless — it does not correspond to any spatial
  operation. Use the domain accessors instead.

A richer, **NEP-42**-style operator grammar for the datatype — spatial
predicates and set operations expressed through native operators / ufuncs, so
`morton_index` columns compose the way numpy dtypes do — is a **deferred,
forward-looking** design and is **not** part of 1.x. Nothing above will change
its meaning when that lands; the accessor methods are the stable 1.x contract.
The MOC set operations that back such a grammar already exist today as functions
(`moc_and`, `moc_or`, `moc_minus`, …) — see the coverage API.

## See also

- [`arrow_interchange.md`](arrow_interchange.md) — the two Arrow surfaces
  (pyarrow `ExtensionType` and the pyarrow-free C Data Interface) in depth.
- [`specification.md`](specification.md) — the packed-word bit layout, the
  decimal-Morton id grammar, and the hive-path convention.
- [`coverage_methods.md`](coverage_methods.md) — polygon/MOC coverage that
  produces `morton_index` cell lists.

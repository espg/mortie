# Arrow interchange for `morton_index`

`mortie` exposes its packed 64-bit `morton_index` words to the Arrow ecosystem
through **two** surfaces. Both carry the same `uint64` storage and the same
`mortie.morton_index` **extension type**, and both map the all-zero empty
sentinel to an Arrow **null** and back. Neither is a runtime dependency — numpy
stays the only hard dep.

| surface | module | needs | use for |
|---|---|---|---|
| pyarrow `ExtensionType` skin | `mortie.arrow.from_morton_index` / `to_morton_index` | `pyarrow` | parquet / IPC, `table.to_pandas()`, off-worker analysis |
| **library-agnostic C Data Interface** | `mortie.arrow.export_c_array` / `import_c_array` (+ `MortonIndexArray.__arrow_c_array__` / `from_arrow`) | **nothing beyond numpy** | zero-copy handoff to **arro3-core**, polars, pyarrow — including envs with no pyarrow |

The second surface (issue #93) is what lets a `morton_index` column travel
through **arro3-core** — the pyarrow-free Arrow carrier used on constrained
workers (e.g. an AWS Lambda layer without pyarrow). The raw Arrow C structs are
built in Rust (via the `arrow` crate), so nothing on the critical path imports
pyarrow.

## The pyarrow extension classes: `MortonIndexType` / `MortonIndexExtArray`

The pyarrow skin's two classes are public as `mortie.MortonIndexType` and
`mortie.MortonIndexExtArray` (and on `mortie.arrow`), but they are **defined
inside `_build_type()`** and handed out by a module `__getattr__` — they are
never bound as module attributes. That is why they have no rendered
[API page](api/arrow.md): mkdocstrings resolves modules statically, and
static resolution finds no such attribute to render, whether or not pyarrow
is installed. They are documented here instead.

pyarrow itself stays **optional**: importing mortie never *requires* it — a
numpy-only install imports fine, and touching either name there raises an
`ImportError` pointing at the missing extra. When pyarrow *is* installed,
`mortie.arrow` builds and registers the extension type eagerly at import, so
a parquet read resolves the `mortie.morton_index` extension name without the
user having touched the type first.

- **`MortonIndexType`** is the `pyarrow.ExtensionType` subclass over
  `uint64` storage with extension name `mortie.morton_index`. It carries no
  parameters — its serialized form is empty; the extension name is the whole
  identity — so the type survives parquet / IPC round-trips.
  `morton_index_type()` builds, registers, and returns the singleton
  instance; there is no reason to construct the class directly.
- **`MortonIndexExtArray`** is the matching `pyarrow.ExtensionArray`
  subclass: what `from_morton_index` returns, and what pyarrow hands back
  when the registered type resolves on read. Its one addition over the
  stock class is `to_numpy(**kwargs)`, which materializes the `uint64`
  storage (defaulting `zero_copy_only=False` so a null-bearing array
  converts); for the null → sentinel-`0` word mapping, go through
  `to_morton_index` instead.

## Producing a column (any Arrow lib)

`export_c_array` returns the `(schema_capsule, array_capsule)` pair of the
[Arrow PyCapsule C Data Interface][pycapsule] from raw `uint64` words (or a
`MortonIndexArray`). Wrap it in a tiny object exposing `__arrow_c_array__` and
hand it to any Arrow constructor:

```python
import numpy as np
from mortie import arrow as marrow

words = ...  # uint64 numpy array of packed morton_index words (0 == null)

class MortonColumn:
    def __arrow_c_array__(self, requested_schema=None):
        return marrow.export_c_array(words)
    def __arrow_c_schema__(self):
        return marrow.export_c_schema()

# arro3-core (no pyarrow needed):
from arro3.core import Array
a3 = Array.from_arrow(MortonColumn())
# a3.field.metadata_str["ARROW:extension:name"] == "mortie.morton_index"

# pyarrow, if installed, resolves the registered extension type:
import pyarrow as pa
pa_arr = pa.array(MortonColumn())          # type: extension<mortie.morton_index>
```

`MortonIndexArray` (the pandas extension array) implements `__arrow_c_array__`
directly, so it can be passed straight to `Array.from_arrow(...)` / `pa.array(...)`.

## Consuming a column back to words

`import_c_array` accepts either an object exposing `__arrow_c_array__` (an
arro3-core / pyarrow / polars array) or a raw `(schema_capsule, array_capsule)`
tuple, and returns a `uint64` numpy array (Arrow nulls come back as the empty
sentinel `0`):

```python
words = marrow.import_c_array(a3)                    # from arro3-core
mia = MortonIndexArray.from_arrow(a3)                # straight to the pandas array
```

## Extension metadata survives arro3-core

The exported schema carries the extension type as field metadata
(`ARROW:extension:name = mortie.morton_index`). **arro3-core `0.8.1` round-trips
this metadata** through the C-Data boundary (verified in
`mortie/tests/test_arrow_cdata.py::TestArro3Interop`), so a column stays typed
end-to-end with no fallback to bare `uint64`. If a future carrier drops the
metadata, the words still transfer (storage is `uint64`); re-attach the type at
the edge with `MortonIndexArray.from_arrow`.

## Null / sentinel semantics

The missing value is the all-zero empty word (`MortonIndexArray.isna()`), the
kernel's null sentinel. On export it becomes an Arrow null via a real **validity
bitmap**; on import a null becomes the sentinel again — byte-for-byte through any
Arrow lib, not just via pyarrow's `fill_null`.

## Running the arro3-core tests locally

arro3-core is **not** in the CI `test` extra, so the arro3 leg of
`test_arrow_cdata.py` skips in CI (the pyarrow leg runs there). To exercise it
locally, install the pinned carrier and run the suite:

```sh
pip install "mortie[arro3]"    # or: pip install arro3-core==0.8.1
pytest mortie/tests/test_arrow_cdata.py
```

[pycapsule]: https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html

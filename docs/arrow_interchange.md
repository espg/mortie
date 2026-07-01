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

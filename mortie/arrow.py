"""
The ``morton_index`` Arrow skin: a pyarrow :class:`pyarrow.ExtensionType` over
``uint64`` storage carrying the ``morton_index`` tag (issue #35, phase 4;
issue #58 flipped the storage to ``uint64``).

This is the Arrow-interop sibling of the pandas ExtensionArray in
:mod:`mortie.morton_index`. The packed 64-bit decimal-Morton words live in Rust
(``src_rust/src/decimal_morton.rs``); this module only wraps them so the same
words can travel through an Arrow array and survive a parquet round-trip with
their ``morton_index`` identity attached as extension metadata. Storage is the
raw ``uint64`` words verbatim (over the kernel's bit layout), so the raw word
order is the Z-order, the same convention as the pandas skin.

pyarrow is an **optional** dependency exactly like pandas: importing ``mortie``
succeeds with neither installed. The extension type is built lazily on first
use and a clear ``ImportError`` is raised if it is touched without pyarrow.
"""

import numpy as np

# The extension-type / array helpers are provided via module-level
# ``__getattr__`` (built lazily so a numpy-only install can import this module),
# so they are intentionally not named in ``__all__`` here.
__all__ = []

# The extension name registered with pyarrow (the metadata tag that travels
# through IPC / parquet and identifies the type on the way back).
EXTENSION_NAME = "mortie.morton_index"


def _require_pyarrow():
    """Import pyarrow lazily, raising a clear error if it is absent.

    pyarrow is an optional extra (the only hard runtime dep is numpy), so the
    extension type is built on top of whatever pyarrow provides at call time
    rather than at module import.
    """
    try:
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover - exercised via message only
        raise ImportError(
            "the morton_index Arrow extension type requires pyarrow; install it "
            "with `pip install mortie[pyarrow]` (or `pip install pyarrow`)"
        ) from exc
    return pa


# The extension type is created and registered once, on first access, so that a
# numpy-only install can import this module without pyarrow present.
_EXT_TYPE = None
_REGISTERED = False


def _build_type():
    """Define, instantiate, and register the pyarrow extension type.

    Returns the singleton ``MortonIndexType`` instance, building it once and
    caching it on the module. Splitting this out of import time is what keeps
    pyarrow an optional dependency.
    """
    global _EXT_TYPE, _REGISTERED
    if _EXT_TYPE is not None:
        return _EXT_TYPE

    pa = _require_pyarrow()

    class MortonIndexExtArray(pa.ExtensionArray):
        """Extension array whose ``.to_numpy()`` yields the packed words."""

        def to_numpy(self, **kwargs):
            kwargs.setdefault("zero_copy_only", False)
            return self.storage.to_numpy(**kwargs)

    class MortonIndexType(pa.ExtensionType):
        """pyarrow extension type for ``morton_index`` packed words.

        Storage is ``uint64`` (the raw packed Morton words, verbatim); the
        ``morton_index`` identity rides along as the extension name so it
        survives IPC / parquet serialization. Carries no parameters, so its
        serialized form is empty.
        """

        def __init__(self):
            super().__init__(pa.uint64(), EXTENSION_NAME)

        def __arrow_ext_serialize__(self):
            # No parameters to carry; the extension name is the whole identity.
            return b""

        @classmethod
        def __arrow_ext_deserialize__(cls, storage_type, serialized):
            return cls()

        def __arrow_ext_class__(self):
            return MortonIndexExtArray

        def to_pandas_dtype(self):
            """The matching pandas ExtensionDtype (``morton_index``)."""
            from .morton_index import MortonIndexDtype

            return MortonIndexDtype()

    inst = MortonIndexType()
    if not _REGISTERED:
        try:
            pa.register_extension_type(inst)
        except pa.ArrowKeyError:
            # Already registered (e.g. a prior build in the same interpreter).
            pass
        _REGISTERED = True
    _EXT_TYPE = inst
    return _EXT_TYPE


def morton_index_type():
    """Return the (registered) ``morton_index`` pyarrow extension type."""
    return _build_type()


def from_morton_index(array):
    """Wrap a :class:`~mortie.morton_index.MortonIndexArray` as an Arrow array.

    Builds a pyarrow ``ExtensionArray`` of the ``morton_index`` type over the
    same ``uint64`` words. ``array`` may also be a raw ``uint64`` array-like of
    words. Missing elements -- a ``MortonIndexArray`` for which :meth:`isna` is
    True, i.e. the all-zero empty sentinel word -- emit Arrow nulls, so a null
    survives the round-trip back through :func:`to_morton_index`. (The missing
    mask is read off the ``uint64`` words, so a sentinel word in a raw array is
    treated as a null too; an already-built Arrow array goes back through
    :func:`to_morton_index`, not here.)
    """
    pa = _require_pyarrow()
    ext_type = _build_type()
    data = np.asarray(getattr(array, "_data", array), dtype=np.uint64)
    # The empty sentinel (all-zero word, prefix 0) is the missing value on the
    # pandas side; mirror it as an Arrow null so isna() round-trips both ways.
    from .morton_index import MortonIndexArray

    mask = data == MortonIndexArray._SENTINEL
    storage = pa.array(data, type=pa.uint64(), mask=mask)
    return pa.ExtensionArray.from_storage(ext_type, storage)


def to_morton_index(array):
    """Convert an Arrow ``morton_index`` array back to a ``MortonIndexArray``.

    Accepts the extension array (or its plain ``uint64`` storage) and returns the
    pandas-side :class:`~mortie.morton_index.MortonIndexArray` over the same
    words. Arrow nulls come back as the all-zero empty sentinel word, so the
    pandas :meth:`isna` reports them as missing.
    """
    _require_pyarrow()
    from .morton_index import MortonIndexArray

    storage = getattr(array, "storage", array)
    # Fill nulls with the empty sentinel before materializing: a uint64 array
    # with a null buffer cannot go straight to numpy.
    if storage.null_count:
        storage = storage.fill_null(int(MortonIndexArray._SENTINEL))
    words = storage.to_numpy(zero_copy_only=False).astype(np.uint64, copy=False)
    return MortonIndexArray(words)


# ---------------------------------------------------------------------------
# Arrow C Data Interface (PyCapsule) surface -- library-agnostic, pyarrow-free
# (issue #93).
#
# These build/consume the raw Arrow C structs in Rust (via the ``arrow`` crate),
# so any Arrow lib that speaks the PyCapsule interface -- arro3-core (the carrier
# zagg runs on its Lambda worker, without pyarrow), pyarrow, polars -- can pull a
# typed ``morton_index`` column zero-copy and hand one back. The runtime stays
# numpy-only; nothing here imports pyarrow.
# ---------------------------------------------------------------------------


def export_c_array(words):
    """Export packed ``uint64`` words as an Arrow C Data Interface capsule pair.

    Returns ``(schema_capsule, array_capsule)`` PyCapsules carrying the words as
    a ``morton_index`` extension column (``ARROW:extension:name`` on the schema),
    with the all-zero empty sentinel mapped to an Arrow null via a real validity
    bitmap. Consumable by any Arrow lib without pandas or pyarrow. ``words`` is
    any ``uint64`` array-like (e.g. a raw numpy array or a ``MortonIndexArray``).
    """
    from . import _rustie

    data = np.ascontiguousarray(
        np.asarray(getattr(words, "_data", words), dtype=np.uint64)
    )
    return _rustie.rust_mi_export_c_array(data)


def export_c_schema():
    """Return the ``morton_index`` Arrow schema capsule (``__arrow_c_schema__``)."""
    from . import _rustie

    return _rustie.rust_mi_export_c_schema()


def import_c_array(source):
    """Import an Arrow C Data Interface array/stream as packed ``uint64`` words.

    ``source`` is one of:

    * an object exposing ``__arrow_c_array__`` (a contiguous arro3-core / pyarrow /
      polars array),
    * an object exposing ``__arrow_c_stream__`` (a **chunked** column / multi-batch
      source -- every chunk is concatenated),
    * or a ``(schema_capsule, array_capsule)`` tuple.

    Arrow nulls come back as the all-zero empty sentinel, so the null<->sentinel
    convention round-trips byte-for-byte. Returns a ``uint64`` numpy array. No
    pyarrow dependency on any path.
    """
    from . import _rustie

    # A single contiguous array is preferred when both are present; only a
    # chunked source (no __arrow_c_array__) goes through the stream path.
    if hasattr(source, "__arrow_c_array__"):
        schema_capsule, array_capsule = source.__arrow_c_array__()
        return _rustie.rust_mi_import_c_array(schema_capsule, array_capsule)
    if hasattr(source, "__arrow_c_stream__"):
        return _rustie.rust_mi_import_c_stream(source.__arrow_c_stream__())
    schema_capsule, array_capsule = source
    return _rustie.rust_mi_import_c_array(schema_capsule, array_capsule)


def __getattr__(name):
    """Lazily expose the extension type / array classes.

    Building the type touches pyarrow, so it is deferred until the names are
    actually requested (module import stays numpy-only).
    """
    if name in ("MortonIndexType", "MortonIndexExtArray"):
        inst = _build_type()
        if name == "MortonIndexType":
            return type(inst)
        return inst.__arrow_ext_class__()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Register the extension type eagerly *iff* pyarrow is already importable, so a
# parquet read of a previously-written file resolves the ``morton_index``
# extension name without the user first touching the type. A numpy-only
# environment skips this silently (the type still builds on demand).
try:
    import pyarrow as _pa  # noqa: F401

    _build_type()
except ImportError:
    pass

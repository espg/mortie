"""
The ``morton_index`` Arrow skin: a pyarrow :class:`pyarrow.ExtensionType` over
``int64`` storage carrying the ``morton_index`` tag (issue #35, phase 4).

This is the Arrow-interop sibling of the pandas ExtensionArray in
:mod:`mortie.morton_index`. The packed 64-bit decimal-Morton words live in Rust
(``src_rust/src/decimal_morton.rs``); this module only wraps them so the same
words can travel through an Arrow array and survive a parquet round-trip with
their ``morton_index`` identity attached as extension metadata. Storage is the
raw ``int64`` words verbatim (over the kernel's bit layout), the same
unsigned-Z-order convention as the pandas skin.

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

        Storage is ``int64`` (the raw packed Morton words, verbatim); the
        ``morton_index`` identity rides along as the extension name so it
        survives IPC / parquet serialization. Carries no parameters, so its
        serialized form is empty.
        """

        def __init__(self):
            super().__init__(pa.int64(), EXTENSION_NAME)

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
    same ``int64`` words. ``array`` may also be a raw ``int64`` array-like of
    words.
    """
    pa = _require_pyarrow()
    ext_type = _build_type()
    data = getattr(array, "_data", array)
    storage = pa.array(np.asarray(data, dtype=np.int64), type=pa.int64())
    return pa.ExtensionArray.from_storage(ext_type, storage)


def to_morton_index(array):
    """Convert an Arrow ``morton_index`` array back to a ``MortonIndexArray``.

    Accepts the extension array (or its plain ``int64`` storage) and returns the
    pandas-side :class:`~mortie.morton_index.MortonIndexArray` over the same
    words.
    """
    _require_pyarrow()
    from .morton_index import MortonIndexArray

    storage = getattr(array, "storage", array)
    words = storage.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    return MortonIndexArray(words)


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

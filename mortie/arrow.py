"""The ``morton_index`` Arrow skin: a pyarrow ExtensionType over the words.

A pyarrow :class:`pyarrow.ExtensionType` over ``uint64`` storage carrying the
``morton_index`` tag (issue #35, phase 4; issue #58 flipped the storage to
``uint64``).

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

    Returns
    -------
    module
        The imported ``pyarrow`` module.

    Raises
    ------
    ImportError
        If pyarrow is not installed, with the install command in the message.
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

    Splitting this out of import time is what keeps pyarrow an optional
    dependency.

    Returns
    -------
    pyarrow.ExtensionType
        The singleton ``MortonIndexType`` instance, built once and cached on
        the module.

    Raises
    ------
    ImportError
        If pyarrow is not installed (via :func:`_require_pyarrow`).
    """
    global _EXT_TYPE, _REGISTERED
    if _EXT_TYPE is not None:
        return _EXT_TYPE

    pa = _require_pyarrow()

    class MortonIndexExtArray(pa.ExtensionArray):
        """Extension array whose ``.to_numpy()`` yields the packed words."""

        def to_numpy(self, **kwargs):
            """Materialize the storage as a numpy array of packed words.

            Parameters
            ----------
            **kwargs
                Forwarded to ``pyarrow.Array.to_numpy``; ``zero_copy_only``
                defaults to ``False`` so a null-bearing array converts.

            Returns
            -------
            numpy.ndarray
                The ``uint64`` packed words.
            """
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
            """Build the type over ``uint64`` storage with the extension name."""
            super().__init__(pa.uint64(), EXTENSION_NAME)

        def __arrow_ext_serialize__(self):
            """Serialize the type's parameters.

            Returns
            -------
            bytes
                Empty: there are no parameters to carry; the extension name is
                the whole identity.
            """
            return b""

        @classmethod
        def __arrow_ext_deserialize__(cls, storage_type, serialized):
            """Rebuild the type from its serialized form.

            Parameters
            ----------
            storage_type : pyarrow.DataType
                The storage type read back from the metadata (always
                ``uint64``).
            serialized : bytes
                The serialized parameters (always empty).

            Returns
            -------
            MortonIndexType
                A fresh instance.
            """
            return cls()

        def __arrow_ext_class__(self):
            """Return the ExtensionArray class arrays of this type use.

            Returns
            -------
            type
                :class:`MortonIndexExtArray`.
            """
            return MortonIndexExtArray

        def to_pandas_dtype(self):
            """Return the matching pandas ExtensionDtype (``morton_index``).

            Returns
            -------
            MortonIndexDtype
                The pandas-side dtype, so ``to_pandas()`` lands the words in a
                ``MortonIndexArray``.
            """
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
    """Return the (registered) ``morton_index`` pyarrow extension type.

    Returns
    -------
    pyarrow.ExtensionType
        The singleton type instance, registered with pyarrow on first call.

    Raises
    ------
    ImportError
        If pyarrow is not installed.
    """
    return _build_type()


def from_morton_index(array):
    """Wrap a :class:`~mortie.morton_index.MortonIndexArray` as an Arrow array.

    Builds a pyarrow ``ExtensionArray`` of the ``morton_index`` type over the
    same ``uint64`` words. Missing elements -- a ``MortonIndexArray`` for which
    :meth:`isna` is True, i.e. the all-zero empty sentinel word -- emit Arrow
    nulls, so a null survives the round-trip back through
    :func:`to_morton_index`. (The missing mask is read off the ``uint64``
    words, so a sentinel word in a raw array is treated as a null too; an
    already-built Arrow array goes back through :func:`to_morton_index`, not
    here.)

    Parameters
    ----------
    array : MortonIndexArray or array_like
        The words to wrap; may also be a raw ``uint64`` array-like of words.

    Returns
    -------
    pyarrow.ExtensionArray
        A ``morton_index``-typed Arrow array over the same words.

    Raises
    ------
    ImportError
        If pyarrow is not installed.
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

    Arrow nulls come back as the all-zero empty sentinel word, so the pandas
    :meth:`isna` reports them as missing.

    Parameters
    ----------
    array : pyarrow.ExtensionArray or pyarrow.Array
        The extension array, or its plain ``uint64`` storage.

    Returns
    -------
    MortonIndexArray
        The pandas-side :class:`~mortie.morton_index.MortonIndexArray` over
        the same words.

    Raises
    ------
    ImportError
        If pyarrow is not installed.
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


def _ragged_from_arrow(pa, polygons):
    """Unpack an Arrow polygon batch into flat (lats, lons, offsets) numpy.

    Parameters
    ----------
    pa : module
        The imported ``pyarrow`` module.
    polygons : pyarrow.Array or pyarrow.ChunkedArray or tuple
        A ``list<struct<lat, lon>>`` array, or a ``(lats, lons)`` pair of
        ``list<double>`` arrays with identical offsets.

    Returns
    -------
    lats, lons : numpy.ndarray
        Flat ``float64`` vertex coordinates (the arrow child arrays).
    offsets : numpy.ndarray
        ``int64`` arrow list offsets (a sliced array's nonzero start offset
        passes straight through).

    Raises
    ------
    ValueError
        If the batch contains a null polygon (fail-fast, naming its index),
        the pair's offsets disagree, or the layout is not one of the two
        accepted forms.
    """

    def _plain(arr):
        if isinstance(arr, pa.ChunkedArray):
            arr = arr.combine_chunks()
        if arr.null_count:
            bad = int(np.flatnonzero(arr.is_null().to_numpy(zero_copy_only=False))[0])
            raise ValueError(f"polygon {bad}: null polygon in batch")
        return arr

    def _f64(child):
        return child.cast(pa.float64()).to_numpy(zero_copy_only=False)

    if isinstance(polygons, (tuple, list)) and len(polygons) == 2:
        lat_list, lon_list = (_plain(a) for a in polygons)
        if not lat_list.offsets.equals(lon_list.offsets):
            raise ValueError("lats and lons list arrays must have equal offsets")
        offsets = lat_list.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
        return _f64(lat_list.values), _f64(lon_list.values), offsets

    arr = _plain(polygons)
    if not pa.types.is_struct(arr.type.value_type):
        raise ValueError(
            "polygons must be a list<struct<lat, lon>> array or a "
            "(lats, lons) pair of list<double> arrays"
        )
    verts = arr.values
    offsets = arr.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
    return _f64(verts.field("lat")), _f64(verts.field("lon")), offsets


def polygons_to_morton_mocs(polygons, order=18, tolerance=None, max_cells=None,
                            normalize=True):
    """Batch MOC coverage over an Arrow polygon column (issue #153).

    The Arrow skin of :func:`mortie.polygons_to_morton_mocs` (plural *MOCs*:
    one MOC per input polygon, many→many — not the many→one ring union of the
    multipart scalar form): the ragged polygon batch goes in as an Arrow list
    array, its child arrays feed the numpy core directly, and the ragged
    result comes back as a ``ListArray`` whose values carry the registered
    ``morton_index`` extension type — parquet-ready, e.g. for a catalog's
    ``footprint_cells`` column.

    Parameters
    ----------
    polygons : pyarrow.Array or pyarrow.ChunkedArray or tuple
        Either a ``list<struct<lat, lon>>`` array (fields in degrees), or a
        ``(lats, lons)`` pair of ``list<double>`` arrays with identical
        offsets.  Chunked inputs are combined; nulls are rejected fail-fast
        with the polygon index named.
    order : int, optional
        Finest HEALPix order (1-29), shared by every polygon.  Default 18.
    tolerance, max_cells : float, int, optional
        The shared per-polygon stop criteria, exactly as on
        :func:`mortie.polygons_to_morton_mocs` (mutually exclusive;
        ``tolerance`` in degrees).
    normalize : bool, optional
        Ring-orientation handling, as on :func:`mortie.morton_coverage`.
        Default ``True``.

    Returns
    -------
    pyarrow.ListArray
        One entry per input polygon; entry ``i`` is that polygon's compact
        MOC as ``morton_index``-typed words, byte-identical to the scalar
        :func:`mortie.morton_coverage_moc` on that ring.  A
        ``LargeListArray`` is returned instead when the batch holds more than
        2**31 - 1 cells.

    Raises
    ------
    ImportError
        If pyarrow is not installed.
    ValueError
        Fail-fast with the lowest-index offending polygon named, as on
        :func:`mortie.polygons_to_morton_mocs`; also for null polygons or an
        unrecognized layout.

    Examples
    --------
    >>> import pyarrow as pa
    >>> from mortie import arrow as marrow
    >>> polys = pa.array(
    ...     [[{"lat": 40.0, "lon": -120.0}, {"lat": 50.0, "lon": -120.0},
    ...       {"lat": 45.0, "lon": -110.0}]])
    >>> mocs = marrow.polygons_to_morton_mocs(polys, order=6)
    >>> mocs.type
    ListType(list<item: extension<mortie.morton_index<MortonIndexType>>>)
    """
    pa = _require_pyarrow()
    from .coverage import polygons_to_morton_mocs as _batch

    lats, lons, offsets = _ragged_from_arrow(pa, polygons)
    values, out_offsets = _batch(
        lats, lons, offsets, order=order, tolerance=tolerance,
        max_cells=max_cells, normalize=normalize,
    )
    ext_values = pa.ExtensionArray.from_storage(
        _build_type(), pa.array(values, type=pa.uint64())
    )
    if out_offsets[-1] <= np.iinfo(np.int32).max:
        return pa.ListArray.from_arrays(
            pa.array(out_offsets.astype(np.int32), type=pa.int32()), ext_values
        )
    return pa.LargeListArray.from_arrays(
        pa.array(out_offsets, type=pa.int64()), ext_values
    )


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

    Consumable by any Arrow lib without pandas or pyarrow.

    Parameters
    ----------
    words : array_like
        Any ``uint64`` array-like (e.g. a raw numpy array or a
        ``MortonIndexArray``).

    Returns
    -------
    tuple of PyCapsule
        The ``(schema_capsule, array_capsule)`` pair, carrying the words as a
        ``morton_index`` extension column (``ARROW:extension:name`` on the
        schema), with the all-zero empty sentinel mapped to an Arrow null via
        a real validity bitmap.
    """
    from . import _rustie

    data = np.ascontiguousarray(
        np.asarray(getattr(words, "_data", words), dtype=np.uint64)
    )
    return _rustie.rust_mi_export_c_array(data)


def export_c_schema():
    """Return the ``morton_index`` Arrow schema capsule.

    The ``__arrow_c_schema__`` half of the C Data Interface surface.

    Returns
    -------
    PyCapsule
        An ``ArrowSchema`` capsule carrying the ``morton_index`` extension
        type.
    """
    from . import _rustie

    return _rustie.rust_mi_export_c_schema()


def import_c_array(source):
    """Import an Arrow C Data Interface array/stream as packed ``uint64`` words.

    Arrow nulls come back as the all-zero empty sentinel, so the null<->sentinel
    convention round-trips byte-for-byte. No pyarrow dependency on any path.

    Parameters
    ----------
    source : object or tuple
        One of:

        * an object exposing ``__arrow_c_array__`` (a contiguous arro3-core /
          pyarrow / polars array),
        * an object exposing ``__arrow_c_stream__`` (a **chunked** column /
          multi-batch source -- every chunk is concatenated),
        * or a ``(schema_capsule, array_capsule)`` tuple.

    Returns
    -------
    numpy.ndarray
        The packed words as a ``uint64`` array.
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

    Parameters
    ----------
    name : str
        The attribute being looked up on this module.

    Returns
    -------
    type
        ``MortonIndexType`` or ``MortonIndexExtArray``.

    Raises
    ------
    AttributeError
        For any other name.
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

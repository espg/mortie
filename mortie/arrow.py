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
        Flat ``float64`` vertex coordinates: only the window the listed rows
        actually address (``values[offsets[0]:offsets[-1]]``).
    offsets : numpy.ndarray
        ``int64`` arrow list offsets, **re-based** so ``offsets[0] == 0`` and
        ``offsets[-1] == len(lats)`` — the exact-coverage contract the core
        requires.  A sliced array's offsets start above 0 and its ``.values``
        is the *whole* unsliced child, so both are corrected together here.

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

    def _rebased(arr):
        """Return the listed rows' value window (zero-copy), 0-based offsets."""
        offsets = arr.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
        start, end = int(offsets[0]), int(offsets[-1])
        # List offsets are contiguous by construction, so [start, end) is
        # exactly the region the listed rows use.
        return arr.values.slice(start, end - start), offsets - start

    def _f64(child):
        return child.cast(pa.float64()).to_numpy(zero_copy_only=False)

    if isinstance(polygons, (tuple, list)) and len(polygons) == 2:
        lat_list, lon_list = (_plain(a) for a in polygons)
        lat_verts, offsets = _rebased(lat_list)
        lon_verts, lon_offsets = _rebased(lon_list)
        # Compare re-based offsets: a pair where only one side is sliced is
        # logically identical and must be accepted.
        if not np.array_equal(offsets, lon_offsets):
            raise ValueError("lats and lons list arrays must have equal offsets")
        return _f64(lat_verts), _f64(lon_verts), offsets

    arr = _plain(polygons)
    if not pa.types.is_struct(arr.type.value_type):
        raise ValueError(
            "polygons must be a list<struct<lat, lon>> array or a "
            "(lats, lons) pair of list<double> arrays"
        )
    verts, offsets = _rebased(arr)
    return _f64(verts.field("lat")), _f64(verts.field("lon")), offsets


def polygons_to_morton_mocs(polygons, order=18, tolerance=None, max_cells=None,
                            normalize=True, *, latitude="authalic"):
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
        offsets.  Each list entry is **one ring** — there is no multipart/hole
        spelling here, so a multi-ring footprint must be decomposed by the
        caller (and covered with :func:`mortie.morton_coverage_moc`'s
        list-of-rings form if the union is what is wanted).  Chunked inputs are combined; a **sliced** input is re-based
        (its offsets shifted to 0 and only its own vertex window passed on, so
        the untouched rest of the column is neither copied nor covered);
        nulls are rejected fail-fast with the polygon index named.
    order : int, optional
        Finest HEALPix order (1-29), shared by every polygon.  Default 18.
    tolerance, max_cells : float, int, optional
        The shared per-polygon stop criteria, exactly as on
        :func:`mortie.polygons_to_morton_mocs` (mutually exclusive;
        ``tolerance`` in degrees).
    normalize : bool, optional
        Ring-orientation handling, as on :func:`mortie.morton_coverage`.
        Default ``True``.
    latitude : str, optional
        Latitude convention of the input vertices, as on
        :func:`mortie.polygons_to_morton_mocs` (issue #186).  Default
        ``"authalic"``; ``"geodetic-spherical"`` is the legacy escape.

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
    from .batch import polygons_to_morton_mocs as _batch

    lats, lons, offsets = _ragged_from_arrow(pa, polygons)
    values, out_offsets = _batch(
        lats, lons, offsets, order=order, tolerance=tolerance,
        max_cells=max_cells, normalize=normalize, latitude=latitude,
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


def _wkb_blobs_from_arrow(pa, column):
    """Slice an Arrow binary column into one zero-copy blob view per row.

    The four ways a hand-rolled version of this goes wrong, all of them
    handled here once instead of in every consumer (issue #163).  The first
    three are **silent** -- they return different, valid-looking data with no
    error at all:

    1. a ``ChunkedArray`` has no ``buffers()`` at all -- which is the *default*
       shape, since a parquet column reads back chunked -- so the chunks are
       walked in order rather than combined (``combine_chunks`` copies the
       whole column, which is the cost this exists to avoid);
    2. ``slice()`` is zero-copy metadata, so a sliced array's ``buffers()``
       are the **original** array's and its rows start at ``offset`` -- the
       naive ``o[i]`` reads a *different, valid-looking* set of blobs;
    3. ``large_binary`` offsets are ``int64`` where ``binary``'s are
       ``int32``.

    The fourth is a **wrong diagnosis** rather than wrong data: a null spans
    **zero bytes** in the offsets, so it would otherwise pass through as an
    empty blob and be reported by the core as a truncated geometry (the right
    index, the wrong cause) instead of as the absence of one.

    Parameters
    ----------
    pa : module
        The imported ``pyarrow`` module.
    column : pyarrow.Array or pyarrow.ChunkedArray
        A ``binary`` or ``large_binary`` column of WKB/EWKB blobs, or an
        extension column over either (``geoarrow.wkb``), whose storage is
        used.

    Returns
    -------
    list of memoryview
        One view per row, in logical column order, each addressing that row's
        bytes in the column's own value buffer -- no ``bytes`` object is
        built.

    Raises
    ------
    TypeError
        For a non-Arrow input, or an Arrow column that is not ``binary`` /
        ``large_binary`` -- including an extension type over any other
        storage, named by its extension name.
    ValueError
        For a null entry, naming its index in the **logical column's** frame.
        This is a pre-pass over the whole column, so it fires before the core
        parses any blob -- a null preempts a lower-index malformed one.
    """
    if isinstance(column, pa.ChunkedArray):
        chunks = column.chunks
    elif isinstance(column, pa.Array):
        chunks = [column]
    else:
        raise TypeError(
            "WKB column must be a pyarrow Array or ChunkedArray of binary / "
            f"large_binary; got {type(column).__name__}"
        )
    # Checked on the column, not per chunk: a ``ChunkedArray`` carries its
    # ``.type`` with zero chunks (where a per-chunk check never runs and a
    # non-binary column would be answered as empty), and every chunk is that
    # type by construction.
    col_type = column.type
    extension = isinstance(col_type, pa.BaseExtensionType)
    if extension:
        # A geoparquet column read back with the geoarrow extension registered
        # arrives as ``geoarrow.wkb`` over binary storage -- the same file, the
        # same bytes, a different Python type -- so unwrap to the storage the
        # blobs actually live in rather than refusing the PR's own corpus.
        col_type = col_type.storage_type
        if not (pa.types.is_binary(col_type)
                or pa.types.is_large_binary(col_type)):
            raise TypeError(
                "WKB column must hold binary or large_binary values; got the "
                f"extension type {column.type.extension_name} over {col_type}"
            )
    large = pa.types.is_large_binary(col_type)
    if not (large or pa.types.is_binary(col_type)):
        raise TypeError(
            "WKB column must hold binary or large_binary values; got "
            f"{col_type}"
        )
    blobs = []
    base = 0
    for chunk in chunks:
        if extension:
            # Zero-copy metadata: a sliced extension array's storage keeps the
            # slice's own offset and length.
            chunk = chunk.storage
        n = len(chunk)
        if chunk.null_count:
            bad = int(np.flatnonzero(chunk.is_null().to_numpy(zero_copy_only=False))[0])
            raise ValueError(
                f"blob {base + bad}: null entry in WKB column; a null is the "
                "absence of a geometry, not an empty one, and covering "
                "nothing is not a cover"
            )
        if n:
            _, offset_buf, value_buf = chunk.buffers()
            offsets = np.frombuffer(offset_buf, dtype=np.int64 if large else np.int32)
            values = memoryview(b"" if value_buf is None else value_buf)
            # The row window starts at ``chunk.offset``, not at 0: the buffers
            # belong to whatever array this one was sliced out of.
            lo = offsets[chunk.offset:chunk.offset + n]
            hi = offsets[chunk.offset + 1:chunk.offset + n + 1]
            blobs.extend(values[a:b] for a, b in zip(lo, hi))
        base += n
    return blobs


def from_wkbs(column, order=18, tolerance=None, max_cells=None, normalize=True,
              *, latitude="authalic"):
    """Batch MOC coverage over an Arrow WKB column (issue #163).

    The Arrow skin of :func:`mortie.from_wkbs`: a geoparquet / STAC geometry
    column goes in as it comes off the file — ``binary`` or ``large_binary``,
    chunked or not, sliced or not — and the same ragged
    ``(values, out_offsets)`` pair comes back, with every scalar parameter
    forwarded unchanged.  Result ``i`` is byte-identical to the core called on
    blob ``i``.

    What this buys is **correctness, not speed**.  The core already accepts
    byte buffers, so a caller can hand it ``memoryview`` slices off the
    column's value buffer today — but doing that by hand has to get the array
    offset, the chunk boundaries and the offset width all right, and gets
    *different data with no error* if it misses any of them (issue #163) —
    and, for a null, an empty blob reported as a truncated geometry rather
    than as a missing one.  All four traps are handled once here, in
    :func:`_wkb_blobs_from_arrow`.

    Memory is the core's posture unchanged, because this *is* the core: the
    blobs are zero-copy views into the column's own buffer, so no ``bytes``
    object is built (the ~305 MB materialization ``to_pylist()`` costs on a
    555,867-blob column, englacial/zagg#408), and the byte-capped chunk loop
    still bounds the peak at the result plus one chunk.  What it does **not**
    remove is the per-chunk copy — releasing the GIL needs owned bytes — nor
    the one ``memoryview`` object per row.

    Parameters
    ----------
    column : pyarrow.Array or pyarrow.ChunkedArray
        A ``binary`` or ``large_binary`` column, one WKB/EWKB geometry per
        entry.  Chunked input is walked chunk by chunk (never combined, which
        would copy the column); a **sliced** input reads its own rows; nulls
        are rejected fail-fast with the index named.  An **extension** column
        over either storage — a geoparquet column read with the
        ``geoarrow.wkb`` extension registered — is unwrapped to its storage,
        so the same file covers the same whether or not geoarrow is installed.
    order : int, optional
        Finest HEALPix order (1-29), shared by every blob.  Default 18.
    tolerance : float, optional
        Shared per-blob stop radius in **degrees**, mutually exclusive with
        ``max_cells``, exactly as on :func:`mortie.from_wkbs`.
    max_cells : int, optional
        Shared per-blob cell budget, exactly as on :func:`mortie.from_wkbs`.
    normalize : bool, optional
        Ring-orientation handling, as on :func:`mortie.from_wkbs`.  Default
        ``True``.
    latitude : str, optional
        Latitude convention of the blobs' coordinates, as on
        :func:`mortie.from_wkbs` (issue #186).  Default ``"authalic"``;
        ``"geodetic-spherical"`` is the legacy escape.

    Returns
    -------
    values : numpy.ndarray
        Every blob's morton MOC words concatenated (``uint64``).
    out_offsets : numpy.ndarray
        ``int64`` arrow list offsets into *values*, length ``len(column) + 1``;
        ``out_offsets[0]`` is 0 and ``out_offsets[-1]`` is ``len(values)``.

    Raises
    ------
    ImportError
        If pyarrow is not installed.
    ValueError
        Fail-fast naming the offending blob in the logical column's frame (so
        an offender in the third chunk reports its column index, not its
        within-chunk one): a null entry, plus every failure class
        :func:`mortie.from_wkbs` raises.  See the Notes for which one wins
        when a column carries both.
    TypeError
        For a column that is not an Arrow ``binary`` / ``large_binary``, or
        an extension type over any other storage (named by its extension
        name).

    Notes
    -----
    **Two ordered gates**, as on :func:`mortie.from_wkbs` itself: nulls are
    screened by a vectorised pre-pass over the whole column, and only then
    are the blobs parsed and covered.  Each gate reports its own lowest-index
    offender, so a **null preempts a malformed blob at a lower index** — a
    null at row 20 is raised ahead of a truncated blob at row 3.  Within each
    class the lowest index wins.  The pre-pass is an earlier gate, not a
    competing one: it is what turns a null from the core's misleading
    ``truncated WKB`` into the absence of a geometry, and it is per column
    rather than per blob because ``is_null()`` is one vectorised call.

    See Also
    --------
    mortie.from_wkbs : the core batch, and the contract in full.

    Examples
    --------
    >>> import pyarrow as pa
    >>> import shapely                                     # doctest: +SKIP
    >>> from mortie import arrow as marrow                 # doctest: +SKIP
    >>> col = pa.array([shapely.to_wkb(geom)])             # doctest: +SKIP
    >>> values, off = marrow.from_wkbs(col, order=8)       # doctest: +SKIP
    """
    pa = _require_pyarrow()
    from .batch import from_wkbs as _batch

    return _batch(
        _wkb_blobs_from_arrow(pa, column), order=order, tolerance=tolerance,
        max_cells=max_cells, normalize=normalize, latitude=latitude,
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

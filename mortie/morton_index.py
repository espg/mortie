"""The ``morton_index`` datatype: a pandas ExtensionArray over packed words.

A pandas ExtensionArray skin over the packed 64-bit decimal-Morton MOC kernel
(issue #35, phase 5).

The kernel lives in Rust (``src_rust/src/decimal_morton.rs``); this module is the
user-facing surface. Storage is raw ``uint64`` packed words (issue #58; zero-copy
over the kernel's bit layout ``[4-bit prefix | 54-bit body | 6-bit suffix]``). The
word is unsigned, so the Z-order is simply the raw word order -- base cells 7..=11
(prefix 8..=12) set bit 63 and sort after the northern cells with no special
casing, and comparisons/sort operate on the words directly. Domain operations
(``coarsen``/``order``/``base_cell``) and the ``(nested, depth)`` <-> word bridge
delegate to the vectorized Rust bindings; **no arithmetic operators** are defined
(raw arithmetic on packed words is meaningless).

pandas is an **optional** dependency: importing ``mortie`` succeeds with only
numpy installed. The pandas machinery here is built lazily on first use, and a
clear ``ImportError`` is raised if the ExtensionArray is touched without pandas.
"""

import os

import numpy as np

from . import _rustie

# ``MortonIndexDtype`` / ``MortonIndexArray`` are provided via module-level
# ``__getattr__`` (built lazily so a numpy-only install can import this module),
# so they are intentionally not named in ``__all__`` here.
__all__ = ["MortonIndexScalar", "decimal_to_word", "decimals_to_words"]

# HEALPix orders this datatype reaches (0 = base cell, 29 = max resolution).
MAX_ORDER = 29


class MortonIndexScalar(np.uint64):
    """A packed ``morton_index`` word that displays as its decimal string.

    Element access and iteration on a ``MortonIndexArray`` yield this type
    (issue #104), so a downstream ``f"{shard_key}"`` prints the decimal Morton
    id (``-31123`` style) rather than the raw packed word. It subclasses
    ``numpy.uint64``: comparisons, hashing, and ``int()`` (the packed word)
    behave exactly like the word itself; only ``str``/``repr`` differ. The
    empty sentinel renders ``"<NA>"``; a word with an invalid prefix renders
    ``"<invalid 0x...>"`` rather than raising (a repr must never raise).

    Construct it from a packed word (an ``int`` or ``numpy.uint64``) exactly as
    you would a ``numpy.uint64``; the constructor is inherited unchanged.
    """

    def __str__(self):
        """Render the word as its decimal Morton string.

        Returns
        -------
        str
            The decimal Morton id, ``"<NA>"`` for the empty sentinel, or
            ``"<invalid 0x...>"`` for a word with an invalid prefix.
        """
        word = int(self)
        if word == 0:
            return "<NA>"
        try:
            return _rustie.rust_mi_decimal_repr(
                np.asarray([word], dtype=np.uint64)
            )[0]
        except ValueError:
            return f"<invalid {word:#018x}>"

    __repr__ = __str__

    def __format__(self, spec):
        """Format the decimal Morton string, not the packed word.

        numpy's numeric ``__format__`` would print the packed word; the display
        form of a morton_index is its decimal string, so ``f"{shard_key}"`` (and
        any string spec, e.g. ``">10"``) formats that instead. ``int(self)``
        remains the escape hatch to format the raw word numerically. Old-style
        ``"%d" % key`` bypasses ``__format__`` entirely and emits the raw word.

        Parameters
        ----------
        spec : str
            A standard format spec, applied to the decimal string.

        Returns
        -------
        str
            The formatted decimal Morton string.
        """
        return format(str(self), spec)

    def __reduce__(self):
        """Pickle as a ``MortonIndexScalar`` rather than a bare ``uint64``.

        numpy scalars pickle through ``multiarray.scalar``, which rebuilds the
        bare ``np.uint64`` and would silently drop the decimal display on any
        process boundary (multiprocessing/dask); rebuild the wrapper instead.

        Returns
        -------
        tuple
            The ``(callable, args)`` pair pickle uses to rebuild the wrapper.
        """
        return (type(self), (int(self),))


def decimal_to_word(s, dtype=np.uint64):
    """Parse one decimal Morton string into its packed word (issue #114).

    The scalar inverse of the decode-through-kernel repr, and the public
    counterpart to :meth:`MortonIndexArray.decimal_repr`: sign column +
    leading base digit (``1..6``), one ``1..4`` digit per order, and an
    optional terminal ``p`` kind suffix (spec section 4, issue #120). A
    ``p``-marked string (legal only at order 29) yields the POINT word; an
    unmarked string always yields the AREA word -- the tie-break for the one
    ambiguous form, and fully backward compatible (every pre-suffix string
    is unmarked).

    numpy-only: calling this imports no pandas, so it is usable from hot
    per-key parse paths. Use :func:`decimals_to_words` for arrays.

    Parameters
    ----------
    s : str
        The decimal Morton id, e.g. ``"-31123"``.
    dtype : type, optional
        The return shape. ``np.uint64`` (default) returns the bare packed
        word, staying numpy-native for hot loops; ``int`` returns a Python
        int; :class:`MortonIndexScalar` returns a word that displays back as
        its decimal string. ``"uint64"`` / ``np.dtype("uint64")`` are
        accepted spellings of the default.

    Returns
    -------
    numpy.uint64 or int or MortonIndexScalar
        The packed word, in the shape requested by ``dtype``.

    Raises
    ------
    ValueError
        If ``s`` is a malformed decimal Morton id.
    TypeError
        If ``dtype`` is not ``np.uint64`` (or a spelling of it), ``int``, or
        :class:`MortonIndexScalar`.

    See Also
    --------
    decimals_to_words : The vectorized array counterpart.
    """
    word = int(_rustie.rust_mi_from_decimal([s])[0])
    if dtype is int:
        return word
    # `issubclass`, not `is`: a MortonIndexScalar subclass must round-trip as
    # itself rather than silently downgrading to a bare uint64.
    if isinstance(dtype, type) and issubclass(dtype, MortonIndexScalar):
        return dtype(word)
    # Only a dtype *spelling* is accepted here -- `np.dtype(np.uint64(0))`
    # happens to succeed on an instance, which would let a stray value through.
    if isinstance(dtype, (type, str, np.dtype)):
        try:
            requested = np.dtype(dtype)
        except TypeError:
            requested = None
        if requested == np.uint64:
            return np.uint64(word)
    raise TypeError(
        f"decimal_to_word dtype must be np.uint64 (the default), int, or "
        f"MortonIndexScalar; got {dtype!r}"
    )


def decimals_to_words(decimals):
    """Parse an array of decimal Morton strings into packed words (issue #114).

    The vectorized inverse of :meth:`MortonIndexArray.to_decimal`, parsed in
    Rust in one pass. Shape is preserved; the result is always ``uint64``.
    numpy-only, like :func:`decimal_to_word`.

    Parameters
    ----------
    decimals : array_like of str
        Decimal Morton ids, of any shape. A bare ``str`` is rejected -- see
        ``Raises``.

    Returns
    -------
    numpy.ndarray
        ``uint64`` packed words, in the shape of ``decimals``.

    Raises
    ------
    ValueError
        Naming the first malformed id, in input order.
    TypeError
        For non-string input -- a scalar string included, since
        ``np.asarray`` would make it a 0-d array; use
        :func:`decimal_to_word`.
    """
    if isinstance(decimals, str):
        raise TypeError(
            "decimals_to_words expects a sequence of decimal Morton strings; "
            "for a single id use decimal_to_word"
        )
    if isinstance(decimals, (list, tuple)) and len(decimals) == 0:
        # numpy types an empty list as float64, which the dtype guard below
        # would reject. Handled here, ahead of the guard, so that the guard
        # stays purely dtype-driven -- an empty *array* of the wrong dtype is
        # still a wrong dtype, and must not pass just because it has no data.
        return np.empty(0, dtype=np.uint64)
    arr = np.asarray(decimals)
    if arr.dtype.kind not in ("U", "O"):
        # Do not let numpy's str-coercion silently turn e.g. the integer 1
        # into the order-0 id "1"; a parse surface takes strings only.
        raise TypeError(
            f"decimals_to_words expects decimal Morton strings, got an array "
            f"of dtype {arr.dtype!r}"
        )
    flat = arr.ravel().tolist()
    if arr.dtype.kind == "O" and not all(isinstance(s, str) for s in flat):
        # Object arrays can hold anything; name the surface and the offender
        # rather than leaking a bare PyO3 extraction message.
        bad = next(s for s in flat if not isinstance(s, str))
        raise TypeError(
            f"decimals_to_words expects decimal Morton strings, got "
            f"{type(bad).__name__} ({bad!r})"
        )
    return _rustie.rust_mi_from_decimal(flat).reshape(arr.shape)


def _decimal_to_word(s):
    """Parse one decimal Morton string into a Python ``int`` (issue #114).

    Deprecated private alias for :func:`decimal_to_word`. Kept through a
    deprecation cycle because downstream code (zagg's parse boundary) imports
    this name; returns a Python ``int`` exactly as it always has, and rejects
    exactly the same strings. New code should use the public
    :func:`decimal_to_word`.

    One deliberate difference: a non-``str`` argument now raises ``TypeError``
    (from the Rust binding) where the old pure-Python body raised
    ``AttributeError`` from ``s.endswith``. String behavior is unchanged.

    Parameters
    ----------
    s : str
        The decimal Morton id, e.g. ``"-31123"``.

    Returns
    -------
    int
        The packed word as a Python ``int``.

    Raises
    ------
    ValueError
        If ``s`` is a malformed decimal Morton id.
    TypeError
        If ``s`` is not a ``str``.
    """
    return decimal_to_word(s, dtype=int)


def _require_pandas():
    """Import pandas lazily, raising a clear error if it is absent.

    pandas is an optional extra (the only hard runtime dep is numpy), so the
    ExtensionArray classes are built on top of whatever pandas provides at
    call time rather than at module import.

    Returns
    -------
    module
        The imported ``pandas`` module.

    Raises
    ------
    ImportError
        If pandas is not installed, with the install command in the message.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised via message only
        raise ImportError(
            "the morton_index ExtensionArray requires pandas; install it with "
            "`pip install mortie[pandas]` (or `pip install pandas`)"
        ) from exc
    return pd


# The dtype / array classes are created once, on first access, so that a
# numpy-only install can import this module without pandas present.
_DTYPE = None
_ARRAY = None


def _build_classes():
    """Define and register the pandas ExtensionDtype / ExtensionArray.

    Splitting this out of import time is what keeps pandas an optional
    dependency.

    Returns
    -------
    tuple of type
        The ``(dtype_cls, array_cls)`` pair, built once and cached on the
        module.

    Raises
    ------
    ImportError
        If pandas is not installed (via :func:`_require_pandas`).
    """
    global _DTYPE, _ARRAY
    if _DTYPE is not None:
        return _DTYPE, _ARRAY

    pd = _require_pandas()
    from pandas.api.extensions import (
        ExtensionArray,
        ExtensionDtype,
        register_extension_dtype,
    )

    @register_extension_dtype
    class MortonIndexDtype(ExtensionDtype):
        """pandas dtype registered as ``"morton_index"``.

        Backed by ``uint64`` storage (the raw packed Morton words; issue #58).
        The missing value is ``pd.NA``, stored as the kernel's all-zero empty
        sentinel.

        Attributes
        ----------
        name : str
            The registered dtype name, ``"morton_index"``.
        type : type
            The scalar type backing the dtype, ``numpy.uint64``.
        kind : str
            The numpy kind character, ``"u"`` (unsigned integer).
        na_value : pandas.NA
            The missing value this dtype reports.
        """

        name = "morton_index"
        type = np.uint64
        kind = "u"
        na_value = pd.NA
        _is_numeric = False

        @classmethod
        def construct_array_type(cls):
            """Return the ExtensionArray class this dtype constructs.

            Returns
            -------
            type
                :class:`MortonIndexArray`.
            """
            return MortonIndexArray

        def __from_arrow__(self, array):
            """Build a :class:`MortonIndexArray` from a pyarrow array.

            This is the hook pandas calls on ``table.to_pandas()`` for a column
            tagged with the ``morton_index`` Arrow extension type, so the words
            land back as a ``MortonIndexArray`` (not a plain int64 Series).
            The pyarrow import stays lazy so this module remains numpy-only
            importable.

            Parameters
            ----------
            array : pyarrow.Array or pyarrow.ChunkedArray
                The Arrow column to convert; a chunked array is concatenated.

            Returns
            -------
            MortonIndexArray
                The packed words as a pandas ExtensionArray.
            """
            from .arrow import _require_pyarrow, to_morton_index

            pa = _require_pyarrow()
            if isinstance(array, pa.ChunkedArray):
                parts = [to_morton_index(chunk) for chunk in array.chunks]
                if not parts:
                    return MortonIndexArray(np.empty(0, dtype=np.uint64))
                return MortonIndexArray._concat_same_type(parts)
            return to_morton_index(array)

    class MortonIndexArray(ExtensionArray):
        """An array of packed 64-bit ``morton_index`` MOC words.

        Construct from raw words with the constructor, or from a HEALPix NESTED
        index via :meth:`from_nested` / a lat/lon via :meth:`from_latlon`.
        Comparisons and sorting use the raw ``uint64`` (the Z-order); the domain
        methods :meth:`coarsen`, :meth:`orders`/:meth:`order`,
        :meth:`base_cells`/:meth:`base_cell` and :meth:`is_fixed_order` delegate
        to the vectorized Rust bindings. No arithmetic operators are defined.

        Parameters
        ----------
        values : array_like
            1-D array-like of packed ``uint64`` words.
        copy : bool, optional
            Copy ``values`` instead of viewing it. Default ``False``.

        Raises
        ------
        ValueError
            If ``values`` is not 1-dimensional.
        """

        # The all-zero word is the kernel's empty/null sentinel (prefix 0).
        _SENTINEL = np.uint64(0)

        def __init__(self, values, copy=False):
            arr = np.asarray(values, dtype=np.uint64)
            if arr.ndim != 1:
                raise ValueError("morton_index values must be 1-dimensional")
            self._data = arr.copy() if copy else arr

        # -- construction ----------------------------------------------------

        @classmethod
        def from_nested(cls, nested, depth):
            """Pack HEALPix NESTED ids at ``depth`` into ``morton_index`` words.

            Parameters
            ----------
            nested : array_like
                NESTED cell ids.
            depth : int
                The scalar HEALPix order they were hashed at.

            Returns
            -------
            MortonIndexArray
                The packed words.
            """
            nested = np.ascontiguousarray(np.asarray(nested), dtype=np.uint64)
            words = _rustie.rust_mi_from_nested(nested, int(depth))
            return cls(words)

        @classmethod
        def from_words(cls, words, copy=False):
            """Wrap an array of already-packed ``uint64`` words.

            Parameters
            ----------
            words : array_like
                1-D array-like of packed ``uint64`` words.
            copy : bool, optional
                Copy ``words`` instead of viewing it. Default ``False``.

            Returns
            -------
            MortonIndexArray
                The wrapped words.
            """
            return cls(words, copy=copy)

        @classmethod
        def from_latlon(cls, lat, lon, order=MAX_ORDER, points=False):
            """Hash lat/lon (degrees) to ``morton_index`` words at ``order``.

            Routes through the Rust ``healpix`` bridge: lat/lon -> NESTED ids ->
            packed words, so it matches the cross-library nested representation.

            With ``points=False`` (the default) the result is an order-``order``
            **area** cell (``Kind::Area``). With ``points=True`` the location is
            encoded as a max-resolution **point** (``Kind::Point``); point
            encoding is order-29-only, so an explicit ``order != 29`` raised
            together with ``points=True`` is a ``ValueError`` (the default
            ``order`` is ``MAX_ORDER`` so the point path needs no extra argument).

            Parameters
            ----------
            lat : array_like
                Latitudes in degrees.
            lon : array_like
                Longitudes in degrees. Must have the same shape as ``lat``.
            order : int, optional
                HEALPix order to hash at. Default :data:`MAX_ORDER` (29).
            points : bool, optional
                Encode a max-resolution point instead of an area cell.
                Default ``False``.

            Returns
            -------
            MortonIndexArray
                The packed words, one per input coordinate.

            Raises
            ------
            ValueError
                If ``points=True`` is combined with an explicit
                ``order != 29``, or if ``lat`` and ``lon`` differ in shape.
            """
            if points and int(order) != MAX_ORDER:
                raise ValueError(
                    "points=True encodes an order-29 point; pass order=29 "
                    "(the default) or omit it"
                )
            lat = np.ascontiguousarray(np.asarray(lat), dtype=np.float64)
            lon = np.ascontiguousarray(np.asarray(lon), dtype=np.float64)
            if lat.shape != lon.shape:
                raise ValueError("lat and lon must have the same shape")
            if points:
                nested = _rustie.rust_ang2pix(MAX_ORDER, lon, lat)
                nested = np.ascontiguousarray(nested, dtype=np.uint64)
                words = _rustie.rust_mi_from_nested_point(nested)
            else:
                nested = _rustie.rust_ang2pix(int(order), lon, lat)
                nested = np.ascontiguousarray(nested, dtype=np.uint64)
                words = _rustie.rust_mi_from_nested(nested, int(order))
            return cls(words)

        @classmethod
        def from_legacy(cls, legacy):
            """Convert retired legacy decimal Morton ``int64`` values to words.

            One-way bridge (issue #48): the legacy decimal encoding is being
            retired in favour of the packed word, but the converter is kept for
            checking new output against old pinned values. There is no packed ->
            legacy inverse beyond the render-only :meth:`decimal_repr`.

            Parameters
            ----------
            legacy : array_like
                Legacy signed decimal Morton values, convertible to ``int64``.

            Returns
            -------
            MortonIndexArray
                The equivalent packed words.
            """
            legacy = np.ascontiguousarray(np.asarray(legacy), dtype=np.int64)
            words = _rustie.rust_mi_from_legacy(legacy)
            return cls(words)

        @classmethod
        def from_decimal(cls, decimals):
            """Parse decimal Morton strings into an array (issue #114).

            The inverse of :meth:`to_decimal`, and sugar over the numpy-only
            :func:`decimals_to_words` for pandas users -- ``to_decimal()``
            output round-trips straight back through it. An unmarked order-29
            id yields the AREA word; only a ``p``-marked one yields the POINT
            word (spec section 4), so point-ness does not survive a round-trip
            through an unmarked string.

            Parameters
            ----------
            decimals : array_like of str
                Decimal Morton ids, e.g. the output of :meth:`to_decimal`.

            Returns
            -------
            MortonIndexArray
                The parsed packed words.

            Raises
            ------
            ValueError
                Naming the first malformed id.
            """
            return cls(decimals_to_words(decimals))

        @classmethod
        def from_arrow(cls, source):
            """Build a ``MortonIndexArray`` from any Arrow C-Data array (#93).

            The words are pulled over the PyCapsule C Data Interface with **no
            pyarrow dependency**; Arrow nulls come back as the all-zero empty
            sentinel so :meth:`isna` round-trips. This is the
            library-agnostic sibling of :func:`mortie.arrow.to_morton_index`
            (the pyarrow ``ExtensionArray`` path).

            Parameters
            ----------
            source : object or tuple
                An object exposing ``__arrow_c_array__`` (a contiguous
                arro3-core / pyarrow / polars array), one exposing
                ``__arrow_c_stream__`` (a **chunked** column, concatenated),
                or a ``(schema_capsule, array_capsule)`` tuple.

            Returns
            -------
            MortonIndexArray
                The imported packed words.
            """
            from .arrow import import_c_array

            return cls(import_c_array(source))

        # -- Arrow C Data Interface (PyCapsule) -----------------------------
        # The library-agnostic export surface (#93): any Arrow lib that speaks
        # the PyCapsule interface can pull these zero-copy, carrying the
        # morton_index extension type, with no pyarrow on either side.

        def __arrow_c_schema__(self):
            """Return the Arrow C-Data schema capsule for ``morton_index``.

            Returns
            -------
            PyCapsule
                An ``ArrowSchema`` capsule carrying the ``morton_index``
                extension type.
            """
            from .arrow import export_c_schema

            return export_c_schema()

        def __arrow_c_array__(self, requested_schema=None):
            """Return Arrow C-Data ``(schema, array)`` capsules over the words.

            The empty sentinel is exported as an Arrow null (validity bitmap) and
            the schema carries the ``morton_index`` extension type.

            Parameters
            ----------
            requested_schema : PyCapsule, optional
                Accepted (per the protocol) but ignored: this array has a
                single fixed logical type.

            Returns
            -------
            tuple of PyCapsule
                The ``(ArrowSchema, ArrowArray)`` capsule pair.
            """
            from .arrow import export_c_array

            return export_c_array(self._data)

        @classmethod
        def _coerce_words(cls, scalars):
            """Map a sequence of words / NA markers to a uint64 array.

            Missing markers (``pd.NA``/``None``/``NaN``) become the all-zero
            empty sentinel so pandas' NA-bearing construction/assignment paths
            round-trip through :meth:`isna`.

            Parameters
            ----------
            scalars : sequence
                Packed words and/or missing markers.

            Returns
            -------
            numpy.ndarray
                A ``uint64`` array with missing markers replaced by the
                sentinel.
            """
            sentinel = int(cls._SENTINEL)
            out = [
                sentinel if (v is None or v is pd.NA or v != v) else int(v)
                for v in scalars
            ]
            return np.asarray(out, dtype=np.uint64)

        @classmethod
        def _from_sequence(cls, scalars, *, dtype=None, copy=False):
            """Build an array from a sequence of scalars (pandas protocol).

            An object or float sequence goes through :meth:`_coerce_words` so
            NA markers land on the empty sentinel; anything else is cast
            straight to ``uint64``.

            Parameters
            ----------
            scalars : sequence
                Packed words and/or missing markers.
            dtype : ExtensionDtype, optional
                Accepted for the pandas protocol; this array has one dtype.
            copy : bool, optional
                Copy the cast array instead of viewing it. Default ``False``.

            Returns
            -------
            MortonIndexArray
                The constructed array.
            """
            arr = np.asarray(scalars)
            if arr.dtype == object or arr.dtype.kind == "f":
                return cls(cls._coerce_words(scalars))
            return cls(arr.astype(np.uint64, copy=False), copy=copy)

        @classmethod
        def _from_factorized(cls, values, original):
            """Rebuild an array from factorized uniques (pandas protocol).

            Parameters
            ----------
            values : numpy.ndarray
                The unique packed words from :meth:`_values_for_factorize`.
            original : MortonIndexArray
                The array that was factorized (unused; the words are
                self-describing).

            Returns
            -------
            MortonIndexArray
                The uniques as a ``morton_index`` array.
            """
            return cls(values)

        # -- required ExtensionArray surface --------------------------------

        @property
        def dtype(self):
            """The ``morton_index`` ExtensionDtype of this array."""
            return MortonIndexDtype()

        def __len__(self):
            """Return the number of words in the array."""
            return len(self._data)

        def __getitem__(self, item):
            """Index or slice the array.

            Parameters
            ----------
            item : int or slice or array_like
                A scalar position, or any numpy-style selection.

            Returns
            -------
            MortonIndexScalar or MortonIndexArray
                A scalar position yields a :class:`MortonIndexScalar` (so it
                displays as its decimal id, issue #104); any other selection
                yields a new ``MortonIndexArray``.
            """
            result = self._data[item]
            if np.isscalar(result) or isinstance(result, np.integer):
                return MortonIndexScalar(result)
            return type(self)(result)

        def __setitem__(self, key, value):
            """Assign words in place, mapping NA markers to the sentinel.

            Parameters
            ----------
            key : int or slice or array_like
                The positions to assign.
            value : MortonIndexArray or scalar or sequence
                The replacement word(s). ``None`` / ``pd.NA`` / ``NaN`` become
                the all-zero empty sentinel (the dtype's NA value).
            """
            if isinstance(value, type(self)):
                value = value._data
            elif np.isscalar(value) or value is None or value is pd.NA:
                # accept the dtype's NA value (-> empty sentinel)
                value = (
                    int(self._SENTINEL)
                    if (value is None or value is pd.NA or value != value)
                    else int(value)
                )
                self._data[key] = value
                return
            else:
                value = self._coerce_words(value)
            self._data[key] = np.asarray(value, dtype=np.uint64)

        @property
        def nbytes(self):
            """Size of the packed-word storage in bytes."""
            return self._data.nbytes

        def isna(self):
            """Return a boolean mask of the missing elements.

            The empty sentinel (all-zero word, prefix 0) is the missing value.

            Returns
            -------
            numpy.ndarray
                A ``bool`` mask, ``True`` where the word is the sentinel.
            """
            return self._data == self._SENTINEL

        def copy(self):
            """Return a deep copy of the array."""
            return type(self)(self._data, copy=True)

        def take(self, indices, *, allow_fill=False, fill_value=None):
            """Take elements by position (pandas protocol).

            Parameters
            ----------
            indices : array_like of int
                Positions to take.
            allow_fill : bool, optional
                Treat ``-1`` in ``indices`` as a missing marker rather than a
                negative index. Default ``False``.
            fill_value : scalar, optional
                The value to fill with when ``allow_fill=True``; ``None`` /
                ``pd.NA`` fill with the all-zero empty sentinel.

            Returns
            -------
            MortonIndexArray
                The taken words.
            """
            from pandas.api.extensions import take

            if allow_fill and (fill_value is None or fill_value is pd.NA):
                fill_value = int(self._SENTINEL)
            result = take(
                self._data, indices, allow_fill=allow_fill, fill_value=fill_value
            )
            return type(self)(result)

        @classmethod
        def _concat_same_type(cls, to_concat):
            """Concatenate several ``morton_index`` arrays (pandas protocol).

            Parameters
            ----------
            to_concat : sequence of MortonIndexArray
                The arrays to join, in order.

            Returns
            -------
            MortonIndexArray
                The concatenation.
            """
            return cls(np.concatenate([a._data for a in to_concat]))

        def _values_for_argsort(self):
            """Return the sort keys (pandas protocol).

            The word is unsigned, so the raw uint64 order is the Z-order: base
            cells 7..=11 (prefix 8..=12) set bit 63 and sort after the northern
            cells with no special casing.

            Returns
            -------
            numpy.ndarray
                The packed ``uint64`` words themselves.
            """
            return self._data

        def _values_for_factorize(self):
            """Return the factorization keys and NA marker (pandas protocol).

            Returns
            -------
            tuple
                ``(words, na_value)`` -- the packed words and the all-zero
                empty sentinel.
            """
            return self._data, int(self._SENTINEL)

        # -- comparisons -----------------------------------------------------

        def _cmp(self, other, op):
            """Compare elementwise against another array or scalar.

            The word is unsigned, so the raw uint64 order is the Z-order across
            the bit-63 boundary (prefix >= 8 sets bit 63); equality is
            bit-identity.

            Parameters
            ----------
            other : MortonIndexArray or array_like or scalar
                The right-hand operand, cast to ``uint64``.
            op : callable
                A binary comparison from :mod:`operator`.

            Returns
            -------
            numpy.ndarray
                The elementwise ``bool`` result.
            """
            if isinstance(other, type(self)):
                other = other._data
            elif isinstance(other, (list, np.ndarray)):
                other = np.asarray(other, dtype=np.uint64)
            else:
                # scalar
                other = np.uint64(other)
            return op(self._data, np.asarray(other, dtype=np.uint64))

        def __eq__(self, other):
            """Elementwise equality (bit-identity of the packed words)."""
            import operator

            return self._cmp(other, operator.eq)

        def __ne__(self, other):
            """Elementwise inequality (bit-identity of the packed words)."""
            import operator

            return self._cmp(other, operator.ne)

        def __lt__(self, other):
            """Elementwise ``<`` in Z-order (raw ``uint64`` order)."""
            import operator

            return self._cmp(other, operator.lt)

        def __le__(self, other):
            """Elementwise ``<=`` in Z-order (raw ``uint64`` order)."""
            import operator

            return self._cmp(other, operator.le)

        def __gt__(self, other):
            """Elementwise ``>`` in Z-order (raw ``uint64`` order)."""
            import operator

            return self._cmp(other, operator.gt)

        def __ge__(self, other):
            """Elementwise ``>=`` in Z-order (raw ``uint64`` order)."""
            import operator

            return self._cmp(other, operator.ge)

        # -- domain operations (delegate to the Rust kernel) ----------------

        def orders(self):
            """Return the per-element HEALPix order.

            Returns
            -------
            numpy.ndarray
                A ``uint8`` array of per-element orders.
            """
            return _rustie.rust_mi_order_of(self._data)

        def order(self):
            """Return the single shared order of a fixed-order array.

            Returns
            -------
            int or None
                The shared HEALPix order, or ``None`` for an empty array.

            Raises
            ------
            ValueError
                If the array holds mixed orders; use :meth:`orders` for the
                per-element orders or :meth:`coarsen` to cast to a fixed order.
            """
            if not self.is_fixed_order():
                raise ValueError(
                    "array holds mixed orders; use .orders() for the per-element "
                    "orders or .coarsen(k) to cast to a fixed order"
                )
            return int(self.orders()[0]) if len(self) else None

        def base_cells(self):
            """Return the per-element HEALPix base cell (``0..=11``).

            Returns
            -------
            numpy.ndarray
                The base cell of each element; empty / invalid words map to
                ``255``.
            """
            return _rustie.rust_mi_base_cell_of(self._data)

        def base_cell(self):
            """Return the single shared base cell of the array.

            Returns
            -------
            int or None
                The shared base cell, or ``None`` for an empty array.

            Raises
            ------
            ValueError
                If the array spans multiple base cells; use :meth:`base_cells`
                instead.
            """
            bases = self.base_cells()
            if len(bases) == 0:
                return None
            if not np.all(bases == bases[0]):
                raise ValueError(
                    "array spans multiple base cells; use .base_cells() instead"
                )
            return int(bases[0])

        def is_fixed_order(self):
            """Report whether every element shares one HEALPix order.

            Returns
            -------
            bool
                ``True`` if the array is fixed-order (an empty array counts as
                fixed-order), ``False`` if it is mixed-order.
            """
            if len(self) == 0:
                return True
            ords = self.orders()
            return bool(np.all(ords == ords[0]))

        def coarsen(self, k):
            """Coarsen every word to order ``k`` (a new array; suffix rewrite).

            Parameters
            ----------
            k : int
                The target HEALPix order. Elements already at or below order
                ``k`` are returned unchanged.

            Returns
            -------
            MortonIndexArray
                A new array of coarsened words.
            """
            words = _rustie.rust_mi_coarsen(self._data, int(k))
            return type(self)(words)

        def to_nested(self):
            """Decode the words to HEALPix NESTED ids via the kernel.

            Returns
            -------
            tuple of numpy.ndarray
                The ``(nested ids, depths)`` arrays.
            """
            return _rustie.rust_mi_to_nested(self._data)

        def decimal_repr(self):
            """Decode each word to its decimal Morton string (issue #48).

            The human-readable decimal Morton form produced by *decoding* each
            word (the canonical render-only repr; backward-compatible with the
            legacy ``str(legacy_i64)`` for orders 0..=18, the natural extension
            for 19..=29). Point words carry the terminal ``p`` kind suffix
            (spec section 2, issue #120), so the repr is injective across kinds
            and the round-trip is lossless.

            Returns
            -------
            list of str
                One decimal Morton id per element.

            Raises
            ------
            ValueError
                On any empty / invalid word.
            """
            return _rustie.rust_mi_decimal_repr(self._data)

        def to_decimal(self):
            """Emit the decimal strings as a fixed-width numpy array.

            The always-strings interchange convention from issue #48. Point
            words carry the ``p`` suffix (issue #120).

            Returns
            -------
            numpy.ndarray
                The decode-through-kernel decimal strings as a ``"<U32"``
                array (sign + base digit + 29 order digits + the point ``p``
                kind suffix is the widest form, so the width is
                order-independent and stable across arrays).

            Raises
            ------
            ValueError
                On any empty / invalid word.
            """
            return np.asarray(self.decimal_repr(), dtype="<U32")

        def to_legacy_i64(self):
            """Emit the legacy signed decimal ``int64`` form (orders <= 18).

            The named escape hatch from issue #48's emit conventions --
            interchange is always the decimal *string* (:meth:`to_decimal`),
            storage the packed ``uint64``; this exists solely for testing new
            output against old pinned values (pairing with
            :meth:`from_legacy`).

            Returns
            -------
            numpy.ndarray
                The legacy signed decimal values as ``int64``.

            Raises
            ------
            ValueError
                If any element is above order 18 (the legacy decimal overflows
                ``int64`` above that), or on any empty / invalid word -- never
                truncated, never data-dependent.
            """
            strings = self.decimal_repr()  # raises on empty / invalid words
            orders = self.orders()
            if len(self) and int(orders.max()) > 18:
                raise ValueError(
                    f"to_legacy_i64 is capped at order 18 (array holds order "
                    f"{int(orders.max())}); use to_decimal() for orders 19-29"
                )
            return np.asarray([int(s) for s in strings], dtype=np.int64)

        def hive_path(self, root="", suffix=".zarr"):
            """Build the hive-layout path per element (issue #104; spec lands on #62).

            The ``morton-hive/1`` convention (zagg's sparse-coverage design
            record): one decimal digit per directory level, the full id as the
            leaf inside its own node --
            ``{root}/{sign+base}/{d1}/.../{d_order}/{full_id}{suffix}``, e.g.
            ``-31123`` -> ``-3/1/1/2/3/-31123.zarr`` and the order-0 ``-3`` ->
            ``-3/-3.zarr``. Every order is a node, so mixed shard orders nest
            naturally: a coarser shard's leaf sits in the same directory its
            finer siblings descend through.

            Parameters
            ----------
            root : str, optional
                Path prefix placed above the digit chain. Default ``""`` (no
                prefix).
            suffix : str, optional
                Extension appended to the leaf. Default ``".zarr"``.

            Returns
            -------
            list of str
                One hive path per element.

            Raises
            ------
            ValueError
                On any empty / invalid word, or for a point id -- points do
                not live in paths, and the kind suffix never enters a path
                component (spec section 2, issue #120).
            """
            prefix = root.rstrip("/") + "/" if root else ""
            paths = []
            for s in self.decimal_repr():
                if s.endswith("p"):
                    raise ValueError(
                        f"hive_path is undefined for point ids ({s!r}): "
                        f"points do not live in paths, and the kind suffix "
                        f"never enters a path component (spec section 2, "
                        f"issue #120)"
                    )
                head = 2 if s.startswith("-") else 1
                levels = "/".join([s[:head], *s[head:]])
                paths.append(f"{prefix}{levels}/{s}{suffix}")
            return paths

        @classmethod
        def from_hive_path(cls, paths, suffix=".zarr"):
            """Parse hive-layout paths back to words (inverse of hive_path).

            The leaf basename carries the full decimal id; when the path also
            carries the digit directories -- recognized by a
            ``{sign+base}``-shaped component sitting at its slot above the leaf
            -- the whole chain is checked against the leaf. A bare
            ``{full_id}.zarr``, or one under an arbitrary root without the
            digit chain, skips the check. Order-29 ids parse to the *area* word
            (see :func:`decimal_to_word`).

            Parameters
            ----------
            paths : str or os.PathLike or iterable
                A single slash-separated path or an iterable of them.
            suffix : str, optional
                The leaf extension to strip. Default ``".zarr"``.

            Returns
            -------
            MortonIndexArray
                The parsed packed words, in input order.

            Raises
            ------
            ValueError
                If a leaf does not end with ``suffix``, if a leaf carries the
                point kind suffix (points do not live in paths, spec section 2,
                issue #120), or for a mis-filed leaf (wrong base cell, wrong
                descent) when the digit chain is anchored.
            """
            import pathlib

            if isinstance(paths, (str, os.PathLike)):
                paths = [paths]
            decs = []
            for p in paths:
                if isinstance(p, pathlib.PurePath):
                    # honor the path's own flavor: a WindowsPath splits on
                    # backslashes too, which a raw "/"-split would not
                    parts = list(p.parts)
                else:
                    parts = str(p).rstrip("/").split("/")
                leaf = parts[-1]
                if suffix and not leaf.endswith(suffix):
                    raise ValueError(
                        f"hive leaf {leaf!r} does not end with {suffix!r}"
                    )
                dec = leaf[: len(leaf) - len(suffix)] if suffix else leaf
                if dec.endswith("p"):
                    raise ValueError(
                        f"hive leaf {leaf!r} carries the point kind suffix: "
                        f"points do not live in paths (spec section 2, issue "
                        f"#120)"
                    )
                head = 2 if dec.startswith("-") else 1
                levels = [dec[:head], *dec[head:]]
                # Enforce the directory cross-check only when the chain is
                # anchored: a {sign+base}-shaped component (optional "-" plus
                # one digit 1..6) at its expected slot. Anchoring on shape --
                # not equality with the leaf's own sign+base -- keeps a
                # mis-filed wrong-base leaf detectable while still treating
                # an arbitrary root as skippable (root components are
                # indistinguishable from digit dirs by count alone).
                anchor = parts[-1 - len(levels)] if (
                    len(parts) - 1 >= len(levels)
                ) else ""
                body = anchor[1:] if anchor.startswith("-") else anchor
                if len(body) == 1 and "1" <= body <= "6":
                    got = parts[-1 - len(levels):-1]
                    if got != levels:
                        raise ValueError(
                            f"hive path {p!r} directories {'/'.join(got)!r} "
                            f"do not match leaf id {dec!r}"
                        )
                decs.append(dec)
            # One vectorized Rust parse for the whole batch rather than a
            # per-leaf scalar call (issue #114): ~13x on large path lists.
            return cls(decimals_to_words(decs))

        # -- repr ------------------------------------------------------------

        def _word_repr(self, word):
            """Render the decimal-string label for one packed word (#104).

            Parameters
            ----------
            word : int or numpy.uint64
                The packed word to label.

            Returns
            -------
            str
                The decimal Morton id, or the ``<NA>`` / ``<invalid ...>``
                placeholder (see :class:`MortonIndexScalar`).
            """
            return str(MortonIndexScalar(word))

        def __repr__(self):
            """Render the array as decimal ids with its length and order."""
            n = len(self)
            if self.is_fixed_order():
                order = "empty" if n == 0 else f"order={int(self.orders()[0])}"
            else:
                order = "order=mixed"
            head = ", ".join(self._word_repr(w) for w in self._data[:3])
            if n > 6:
                tail = ", ".join(self._word_repr(w) for w in self._data[-3:])
                body = f"{head}, ..., {tail}"
            else:
                body = ", ".join(self._word_repr(w) for w in self._data)
            return f"MortonIndexArray([{body}], len={n}, {order})"

        def _formatter(self, boxed=False):
            """Return the per-element formatter pandas uses to print a Series.

            Parameters
            ----------
            boxed : bool, optional
                ``True`` when pandas is rendering inside a DataFrame; the
                formatter is the same either way. Default ``False``.

            Returns
            -------
            callable
                A function mapping one packed word to its decimal string.
            """
            return lambda w: self._word_repr(w)

    _DTYPE, _ARRAY = MortonIndexDtype, MortonIndexArray
    return _DTYPE, _ARRAY


def __getattr__(name):
    """Lazily expose ``MortonIndexDtype`` / ``MortonIndexArray``.

    Building the classes touches pandas, so it is deferred until the names are
    actually requested (module import stays numpy-only).

    Parameters
    ----------
    name : str
        The attribute being looked up on this module.

    Returns
    -------
    type
        :class:`MortonIndexDtype` or :class:`MortonIndexArray`.

    Raises
    ------
    AttributeError
        For any other name.
    """
    if name in ("MortonIndexDtype", "MortonIndexArray"):
        dtype_cls, array_cls = _build_classes()
        return dtype_cls if name == "MortonIndexDtype" else array_cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Register the dtype eagerly *iff* pandas is already importable, so that
# ``pd.Series(dtype="morton_index")`` resolves the registered name without the
# user first touching the classes. A numpy-only environment skips this silently
# (the classes still build on demand via __getattr__, raising a clear error).
try:
    import pandas as _pd  # noqa: F401

    _build_classes()
except ImportError:
    pass

"""mortie's pandas extension: the ``morton_index`` ExtensionArray.

This is **mortie's pandas integration layer, not pandas itself** -- the
ExtensionDtype / ExtensionArray pair over the packed 64-bit decimal-Morton
words. The numpy-only parse surface and the packed-word scalar live next door
in :mod:`mortie.morton_index`.

pandas stays an **optional** dependency: importing *this* module is what pulls
pandas in, and ``mortie`` imports it only on demand -- when the ExtensionArray
names are actually requested, or once at import time when pandas is already
installed, so the ``"morton_index"`` dtype string is registered. A numpy-only
install therefore imports ``mortie`` fine and gets the curated ``ImportError``
from :func:`mortie.morton_index._require_pandas` if it touches the classes.

The classes sit at module level here -- not inside a function, as they did
before issue #135 -- so they are ordinary, statically discoverable attributes:
``vars()``, ``dir()``, :func:`inspect.getmembers` and mkdocstrings all see them,
and ``__qualname__`` is the bare class name. That needs a module free to import
pandas at its own top level, which is what this one is for.
"""

import os

import numpy as np

from . import _rustie
from .morton_index import (
    MAX_ORDER,
    MortonIndexScalar,
    _require_pandas,
    decimals_to_words,
)

__all__ = ["MortonIndexArray", "MortonIndexDtype"]

# Guarded at module level (not per-class-body, as the pre-#135 builder did) so
# that a bare ``from mortie.pandas import MortonIndexArray`` on a numpy-only
# install raises the same curated ImportError as ``mortie.MortonIndexArray``,
# rather than a bare ModuleNotFoundError.
pd = _require_pandas()

# Bound as plain names so the class statements below read exactly as they did
# when they lived in ``morton_index._build_classes``; ``pandas.api.extensions``
# is reached through ``pd`` to keep every import in this file at the top.
ExtensionArray = pd.api.extensions.ExtensionArray
ExtensionDtype = pd.api.extensions.ExtensionDtype
register_extension_dtype = pd.api.extensions.register_extension_dtype


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

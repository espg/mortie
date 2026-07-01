"""
The ``morton_index`` datatype: a pandas ExtensionArray skin over the packed
64-bit decimal-Morton MOC kernel (issue #35, phase 5).

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

import numpy as np

from . import _rustie

# ``MortonIndexDtype`` / ``MortonIndexArray`` are provided via module-level
# ``__getattr__`` (built lazily so a numpy-only install can import this module),
# so they are intentionally not named in ``__all__`` here.
__all__ = []

# HEALPix orders this datatype reaches (0 = base cell, 29 = max resolution).
MAX_ORDER = 29


def _require_pandas():
    """Import pandas lazily, raising a clear error if it is absent.

    pandas is an optional extra (the only hard runtime dep is numpy), so the
    ExtensionArray classes are built on top of whatever pandas provides at
    call time rather than at module import.
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

    Returns the ``(dtype_cls, array_cls)`` pair, building them once and caching
    on the module. Splitting this out of import time is what keeps pandas an
    optional dependency.
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
        """

        name = "morton_index"
        type = np.uint64
        kind = "u"
        na_value = pd.NA
        _is_numeric = False

        @classmethod
        def construct_array_type(cls):
            return MortonIndexArray

        def __from_arrow__(self, array):
            """Build a :class:`MortonIndexArray` from a pyarrow array.

            This is the hook pandas calls on ``table.to_pandas()`` for a column
            tagged with the ``morton_index`` Arrow extension type, so the words
            land back as a ``MortonIndexArray`` (not a plain int64 Series).
            Accepts a ``pa.Array`` or a ``pa.ChunkedArray``; the pyarrow import
            stays lazy so this module remains numpy-only importable.
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

            ``nested`` is an array-like of NESTED cell ids; ``depth`` is the
            scalar HEALPix order they were hashed at.
            """
            nested = np.ascontiguousarray(np.asarray(nested), dtype=np.uint64)
            words = _rustie.rust_mi_from_nested(nested, int(depth))
            return cls(words)

        @classmethod
        def from_words(cls, words, copy=False):
            """Wrap an array of already-packed ``uint64`` words."""
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
            """
            legacy = np.ascontiguousarray(np.asarray(legacy), dtype=np.int64)
            words = _rustie.rust_mi_from_legacy(legacy)
            return cls(words)

        @classmethod
        def from_arrow(cls, source):
            """Build a ``MortonIndexArray`` from any Arrow C-Data array (#93).

            ``source`` is an object exposing ``__arrow_c_array__`` (a contiguous
            arro3-core / pyarrow / polars array), one exposing
            ``__arrow_c_stream__`` (a **chunked** column, concatenated), or a
            ``(schema_capsule, array_capsule)`` tuple. The words are pulled over
            the PyCapsule C Data Interface with **no pyarrow dependency**; Arrow
            nulls come back as the all-zero empty sentinel so :meth:`isna`
            round-trips. This is the library-agnostic sibling of
            :func:`mortie.arrow.to_morton_index` (the pyarrow ``ExtensionArray``
            path).
            """
            from .arrow import import_c_array

            return cls(import_c_array(source))

        # -- Arrow C Data Interface (PyCapsule) -----------------------------
        # The library-agnostic export surface (#93): any Arrow lib that speaks
        # the PyCapsule interface can pull these zero-copy, carrying the
        # morton_index extension type, with no pyarrow on either side.

        def __arrow_c_schema__(self):
            """Arrow C-Data schema capsule for the ``morton_index`` type."""
            from .arrow import export_c_schema

            return export_c_schema()

        def __arrow_c_array__(self, requested_schema=None):
            """Arrow C-Data ``(schema, array)`` capsules over the packed words.

            The empty sentinel is exported as an Arrow null (validity bitmap) and
            the schema carries the ``morton_index`` extension type. The
            ``requested_schema`` hint is accepted (per the protocol) but ignored:
            this array has a single fixed logical type.
            """
            from .arrow import export_c_array

            return export_c_array(self._data)

        @classmethod
        def _coerce_words(cls, scalars):
            """Map a sequence of words / NA markers to a uint64 array.

            Missing markers (``pd.NA``/``None``/``NaN``) become the all-zero
            empty sentinel so pandas' NA-bearing construction/assignment paths
            round-trip through :meth:`isna`.
            """
            sentinel = int(cls._SENTINEL)
            out = [
                sentinel if (v is None or v is pd.NA or v != v) else int(v)
                for v in scalars
            ]
            return np.asarray(out, dtype=np.uint64)

        @classmethod
        def _from_sequence(cls, scalars, *, dtype=None, copy=False):
            arr = np.asarray(scalars)
            if arr.dtype == object or arr.dtype.kind == "f":
                return cls(cls._coerce_words(scalars))
            return cls(arr.astype(np.uint64, copy=False), copy=copy)

        @classmethod
        def _from_factorized(cls, values, original):
            return cls(values)

        # -- required ExtensionArray surface --------------------------------

        @property
        def dtype(self):
            return MortonIndexDtype()

        def __len__(self):
            return len(self._data)

        def __getitem__(self, item):
            result = self._data[item]
            if np.isscalar(result) or isinstance(result, np.integer):
                return np.uint64(result)
            return type(self)(result)

        def __setitem__(self, key, value):
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
            return self._data.nbytes

        def isna(self):
            # The empty sentinel (all-zero word, prefix 0) is the missing value.
            return self._data == self._SENTINEL

        def copy(self):
            return type(self)(self._data, copy=True)

        def take(self, indices, *, allow_fill=False, fill_value=None):
            from pandas.api.extensions import take

            if allow_fill and (fill_value is None or fill_value is pd.NA):
                fill_value = int(self._SENTINEL)
            result = take(
                self._data, indices, allow_fill=allow_fill, fill_value=fill_value
            )
            return type(self)(result)

        @classmethod
        def _concat_same_type(cls, to_concat):
            return cls(np.concatenate([a._data for a in to_concat]))

        def _values_for_argsort(self):
            # The word is unsigned, so the raw uint64 order is the Z-order: base
            # cells 7..=11 (prefix 8..=12) set bit 63 and sort after the northern
            # cells with no special casing.
            return self._data

        def _values_for_factorize(self):
            return self._data, int(self._SENTINEL)

        # -- comparisons -----------------------------------------------------
        # The word is unsigned, so the raw uint64 order is the Z-order across the
        # bit-63 boundary (prefix >= 8 sets bit 63); equality is bit-identity.

        def _cmp(self, other, op):
            if isinstance(other, type(self)):
                other = other._data
            elif isinstance(other, (list, np.ndarray)):
                other = np.asarray(other, dtype=np.uint64)
            else:
                # scalar
                other = np.uint64(other)
            return op(self._data, np.asarray(other, dtype=np.uint64))

        def __eq__(self, other):
            import operator

            return self._cmp(other, operator.eq)

        def __ne__(self, other):
            import operator

            return self._cmp(other, operator.ne)

        def __lt__(self, other):
            import operator

            return self._cmp(other, operator.lt)

        def __le__(self, other):
            import operator

            return self._cmp(other, operator.le)

        def __gt__(self, other):
            import operator

            return self._cmp(other, operator.gt)

        def __ge__(self, other):
            import operator

            return self._cmp(other, operator.ge)

        # -- domain operations (delegate to the Rust kernel) ----------------

        def orders(self):
            """Per-element HEALPix order as a numpy ``uint8`` array."""
            return _rustie.rust_mi_order_of(self._data)

        def order(self):
            """The single shared order, or raise if the array is mixed-order."""
            if not self.is_fixed_order():
                raise ValueError(
                    "array holds mixed orders; use .orders() for the per-element "
                    "orders or .coarsen(k) to cast to a fixed order"
                )
            return int(self.orders()[0]) if len(self) else None

        def base_cells(self):
            """Per-element HEALPix base cell (``0..=11``) as a numpy array.

            Empty / invalid words map to ``255``.
            """
            return _rustie.rust_mi_base_cell_of(self._data)

        def base_cell(self):
            """The single shared base cell, or raise if the array is mixed."""
            bases = self.base_cells()
            if len(bases) == 0:
                return None
            if not np.all(bases == bases[0]):
                raise ValueError(
                    "array spans multiple base cells; use .base_cells() instead"
                )
            return int(bases[0])

        def is_fixed_order(self):
            """True if every element shares one HEALPix order (else mixed)."""
            if len(self) == 0:
                return True
            ords = self.orders()
            return bool(np.all(ords == ords[0]))

        def coarsen(self, k):
            """Coarsen every word to order ``k`` (a new array; suffix rewrite).

            Elements already at or below order ``k`` are returned unchanged.
            """
            words = _rustie.rust_mi_coarsen(self._data, int(k))
            return type(self)(words)

        def to_nested(self):
            """Return ``(nested ids, depths)`` numpy arrays via the kernel."""
            return _rustie.rust_mi_to_nested(self._data)

        def decimal_repr(self):
            """Decode-through-kernel decimal string repr per element (issue #48).

            Returns a list of ``str``: the human-readable decimal Morton form
            produced by *decoding* each word (the canonical render-only repr;
            backward-compatible with the legacy ``str(legacy_i64)`` for orders
            0..=18, the natural extension for 19..=29). Raises ``ValueError`` on
            any empty / invalid word.
            """
            return _rustie.rust_mi_decimal_repr(self._data)

        # -- repr ------------------------------------------------------------

        def _word_repr(self, word):
            """Compact ``base/order`` label for one packed word."""
            w = np.asarray([word], dtype=np.uint64)
            if word == int(self._SENTINEL):
                return "<NA>"
            base = int(_rustie.rust_mi_base_cell_of(w)[0])
            order = int(_rustie.rust_mi_order_of(w)[0])
            return f"base={base} order={order}"

        def __repr__(self):
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
            return lambda w: self._word_repr(w)

    _DTYPE, _ARRAY = MortonIndexDtype, MortonIndexArray
    return _DTYPE, _ARRAY


def __getattr__(name):
    """Lazily expose ``MortonIndexDtype`` / ``MortonIndexArray``.

    Building the classes touches pandas, so it is deferred until the names are
    actually requested (module import stays numpy-only).
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

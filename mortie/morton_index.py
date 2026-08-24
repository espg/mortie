"""The ``morton_index`` datatype: the numpy-only surface over packed words.

The scalar and decimal parse surface of the packed 64-bit decimal-Morton MOC
kernel (issue #35, phase 5). The pandas ExtensionArray skin over the same words
lives in :mod:`mortie.pandas` (issue #135) and is re-exported from here.

The kernel lives in Rust (``mortie-core/src/decimal_morton.rs``, re-exported as
``mortie_rustie::decimal_morton``); this module is the user-facing surface.
Storage is raw ``uint64`` packed words (issue #58; zero-copy over the kernel's
bit layout ``[4-bit prefix | 54-bit body | 6-bit suffix]``). The word is
unsigned, so the Z-order is simply the raw word order -- base cells 7..=11
(prefix 8..=12) set bit 63 and sort after the northern cells with no special
casing, and comparisons/sort operate on the words directly. Domain operations
(``coarsen``/``order``/``base_cell``) and the ``(nested, depth)`` <-> word bridge
delegate to the vectorized Rust bindings; **no arithmetic operators** are defined
(raw arithmetic on packed words is meaningless).

pandas is an **optional** dependency: importing ``mortie`` succeeds with only
numpy installed. Nothing in *this* module touches pandas; the ExtensionArray
names resolve by importing :mod:`mortie.pandas` on demand, and a clear
``ImportError`` is raised if they are touched without pandas installed.
"""

import numpy as np

from . import _rustie

# ``MortonIndexDtype`` / ``MortonIndexArray`` are re-exported from
# :mod:`mortie.pandas` via module-level ``__getattr__`` (imported on demand so a
# numpy-only install can import this module), so they are intentionally not
# named in ``__all__`` here.
__all__ = ["MortonIndexScalar", "decimal_to_word"]

# HEALPix orders this datatype reaches (0 = base cell, 29 = max resolution).
MAX_ORDER = 29

# Longest rendered argument an error message will carry (issue #152): a
# constructor error quotes what it was handed, and the caller controls that
# length -- a megabyte of garbage must not become a megabyte of exception.
_ERR_REPR_LIMIT = 64


def _clip(text, limit=_ERR_REPR_LIMIT):
    """Bound a caller-controlled fragment of an error message.

    Parameters
    ----------
    text : str
        The rendered fragment, e.g. ``repr(value)`` or ``str(exc)``.
    limit : int, optional
        Maximum length of the returned string, ellipsis included.

    Returns
    -------
    str
        ``text``, truncated with a trailing ``"..."`` if it would
        otherwise exceed ``limit`` characters.
    """
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


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
    you would a ``numpy.uint64``, or from the decimal Morton label itself: a
    ``str`` argument parses as a decimal label through :func:`decimal_to_word`
    (issue #152), so ``MortonIndexScalar("-31123")`` is the cell that displays
    as ``-31123``. The two forms are disambiguated by type alone -- the
    inherited ``numpy.uint64`` constructor used to read a label string as a
    base-10 *packed word*, silently constructing the wrong cell. An invalid
    label raises ``ValueError`` eagerly, at the boundary; display stays
    lazy/never-raise as above. Bytes-like input -- what an HDF5 attr reader
    hands back for a label -- is refused with a pointed ``TypeError``
    rather than guessed at: numpy reads ``bytes`` as a base-10 word and
    ``bytearray`` as a raw buffer, and neither reading is the label.
    """

    def __new__(cls, value=0):
        """Build from a packed word, or parse a decimal Morton label.

        Parameters
        ----------
        value : int-like or str
            A ``str`` is the decimal Morton label, e.g. ``"-31123"``
            (parsed via :func:`decimal_to_word`, terminal ``p`` point
            suffix included); a 0-d ``"U"`` array of one is unwrapped and
            read the same way. Anything else is a packed word, handed to
            ``numpy.uint64`` and taking *its* semantics whole -- ``bool``
            and ``float`` included, so ``1.9`` truncates to ``1``, kept as
            numpy parity by choice rather than tightened here. Bytes-like
            input is refused: see *Raises*.

        Returns
        -------
        MortonIndexScalar
            The packed word, displaying as its decimal label -- for the
            label form and for every int-like scalar. Parity has one
            edge: an input ``numpy.uint64`` turns into an *array* rather
            than a scalar (a buffer such as ``memoryview``, or an array
            of more than one element) comes back as numpy returns it, a
            plain ``ndarray``, not this type.

        Raises
        ------
        ValueError
            If a ``str`` ``value`` is not a well-formed decimal Morton
            label (sign column + base digit ``1..6``, one ``1..4`` digit
            per order, optional terminal ``p`` -- spec sections 2 and 4).
        TypeError
            If ``value`` is bytes-like -- ``bytes``/``numpy.bytes_``
            (which numpy reads as a base-10 *packed word*) or
            ``bytearray`` (which numpy reads as a raw *buffer*, giving an
            array). Neither reading is the decimal label a byte string
            from an HDF5 attr almost always is, so the input is refused
            rather than guessed at. Decode it
            (``value.decode("ascii")``) for a label, or pass an ``int``
            for a packed word. Non-``str``, non-int-like values raise
            whatever ``numpy.uint64`` raises for them.
        """
        if (
            isinstance(value, np.ndarray)
            and value.ndim == 0
            and value.dtype.kind in "SU"
        ):
            # A 0-d "U"/"S" array is a string handed over in an array
            # wrapper (h5py attrs, ``arr[()]``); numpy.uint64 would read
            # either as a base-10 word and slip past both guards below.
            # Unwrap so "U" takes the label path and "S" hits the refusal.
            # Numeric 0-d arrays are left alone -- numpy parity.
            value = value.item()
        if isinstance(value, (bytes, bytearray)):
            raise TypeError(
                f"MortonIndexScalar({_clip(repr(value))}): bytes-like "
                f"input is ambiguous here -- numpy reads bytes as a base-10 "
                f"packed word and bytearray as a raw buffer, and neither "
                f"reading is the decimal Morton label a byte string from an "
                f"HDF5 attr almost always is. Pass value.decode('ascii') "
                f"for a label, or an int for a packed word."
            )
        if isinstance(value, str):
            try:
                word = decimal_to_word(value, dtype=int)
            except ValueError as exc:
                raise ValueError(
                    f"MortonIndexScalar({_clip(repr(value))}): not a "
                    f"decimal Morton "
                    f"label (['-'] + base digit 1..6 + one 1..4 digit per "
                    f"order + optional terminal 'p' -- spec sections 2 and "
                    f"4): {_clip(str(exc), 160)}"
                ) from exc
            return super().__new__(cls, word)
        return super().__new__(cls, value)

    @property
    def decimal(self):
        """The decimal Morton label, exactly as ``str``/``repr`` render it.

        The canonical label string -- the same decode-through-kernel
        rendering as ``str(self)``, so the empty sentinel yields ``"<NA>"``
        and an invalid word yields ``"<invalid 0x...>"`` rather than
        raising (the lazy display posture ruled on issue #152).

        Returns
        -------
        str
            The decimal Morton id, ``"<NA>"``, or ``"<invalid 0x...>"``.
        """
        return str(self)

    @property
    def order(self):
        """The HEALPix order of this word (0-29), via :func:`mortie.orders_of`.

        Pure suffix decode, delegated to the existing kernel -- words are
        not validated (the empty sentinel decodes as order 0; use
        :func:`mortie.validate_morton` to reject malformed words), matching
        :func:`mortie.orders_of` exactly.

        Returns
        -------
        int
            The HEALPix order, 0-29.
        """
        # Lazy import: mortie.orders pulls in the batch/coverage/geometry
        # chain, and this module stays a leaf import (numpy + _rustie only).
        from .orders import orders_of

        return int(orders_of(self)[0])

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
    per-key parse paths.

    **Batch vectorized** (issue #187), with numpy semantics literally: a
    ``str`` in gives one word out, anything else is treated as an array of ids
    and gives an array of words back, parsed in Rust in one pass. The array
    form is always ``uint64``, so ``dtype`` applies to the scalar form only.

    Parameters
    ----------
    s : str or array_like of str
        The decimal Morton id, e.g. ``"-31123"``, or an array of them (any
        shape) for the vectorized form.
    dtype : type, optional
        The return shape. ``np.uint64`` (default) returns the bare packed
        word, staying numpy-native for hot loops; ``int`` returns a Python
        int; :class:`MortonIndexScalar` returns a word that displays back as
        its decimal string. ``"uint64"`` / ``np.dtype("uint64")`` are
        accepted spellings of the default.

    Returns
    -------
    numpy.uint64 or int or MortonIndexScalar or numpy.ndarray
        The packed word, in the shape requested by ``dtype``; for array input,
        a ``uint64`` array in the shape of ``s``.

    Raises
    ------
    ValueError
        If ``s`` is a malformed decimal Morton id -- for the array form,
        naming the first malformed *id* in input order, not its index, so a
        wide array gives no row to look at.
    TypeError
        If ``dtype`` is not ``np.uint64`` (or a spelling of it), ``int``, or
        :class:`MortonIndexScalar`; or if a non-``uint64`` ``dtype`` is asked
        for alongside array input, which has no scalar shape to return.

    See Also
    --------
    _decimals_to_words : the vectorized kernel the array form delegates to.
    """
    if not isinstance(s, str):
        # `MortonIndexScalar` is a uint64 subclass, so `np.dtype` resolves it
        # to uint64 -- it has to be ruled out by identity before that check, or
        # asking for it on array input would silently return bare words.
        if isinstance(dtype, type) and issubclass(dtype, MortonIndexScalar):
            uint64_asked = False
        else:
            try:
                uint64_asked = np.dtype(dtype) == np.uint64
            except TypeError:
                uint64_asked = False
        if not uint64_asked:
            raise TypeError(
                f"decimal_to_word dtype must be np.uint64 (the default) for "
                f"array input, which is always uint64; got {dtype!r}"
            )
        words = _decimals_to_words(s)
        # numpy semantics exactly: a 0-d input is a scalar, not a 0-d array.
        return words if words.ndim else np.uint64(words)
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


def _decimals_to_words(decimals):
    """Parse an array of decimal Morton strings into packed words (issue #114).

    The vectorized inverse of :meth:`MortonIndexArray.to_decimal`, parsed in
    Rust in one pass. Shape is preserved; the result is always ``uint64``.
    numpy-only, like :func:`decimal_to_word`.

    **Private kernel** (issue #187): reached polymorphically by
    :func:`decimal_to_word`'s array form; the public name
    ``decimals_to_words`` retired with the plural batch names.

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
        ``np.asarray`` would make it a 0-d array; :func:`decimal_to_word`
        routes a bare ``str`` down its scalar path before it can get here.
    """
    if isinstance(decimals, str):
        raise TypeError(
            "decimal_to_word's array form expects a sequence of decimal "
            "Morton strings; a single id takes the scalar path"
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
            f"decimal_to_word expects decimal Morton strings, got an array "
            f"of dtype {arr.dtype!r}"
        )
    flat = arr.ravel().tolist()
    if arr.dtype.kind == "O" and not all(isinstance(s, str) for s in flat):
        # Object arrays can hold anything; name the surface and the offender
        # rather than leaking a bare PyO3 extraction message.
        bad = next(s for s in flat if not isinstance(s, str))
        raise TypeError(
            f"decimal_to_word expects decimal Morton strings, got "
            f"{type(bad).__name__} ({bad!r})"
        )
    return _rustie.rust_mi_from_decimal(flat).reshape(arr.shape)


def _decimal_to_word(s):
    """Parse a decimal Morton string to ``int`` (deprecated; issue #114).

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
    ExtensionArray classes in :mod:`mortie.pandas` are built on top of whatever
    pandas provides at that submodule's import rather than at this module's.
    This is the single definition of the missing-pandas message: both the
    ``mortie.MortonIndexArray`` path and a direct
    ``from mortie.pandas import MortonIndexArray`` raise it from here, so the
    two cannot drift.

    Returns
    -------
    module
        The imported ``pandas`` module.

    Raises
    ------
    ImportError
        If pandas is not installed, with the install commands in the message.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised via message only
        raise ImportError(
            "the morton_index ExtensionArray requires pandas, which is not "
            "installed. Install it with `pip install pandas`, or with "
            "`pip install mortie[pandas]` to declare it as a mortie extra so "
            "it is installed alongside mortie in future environments."
        ) from exc
    return pd


def __getattr__(name):
    """Re-export ``MortonIndexDtype`` / ``MortonIndexArray`` from `mortie.pandas`.

    The classes are ordinary module-level definitions in :mod:`mortie.pandas`
    (issue #135); importing that submodule is what touches pandas, so the
    import is deferred until one of the names is actually requested and this
    module stays numpy-only importable. ``mortie.morton_index.MortonIndexArray``
    is a load-bearing downstream import path, which is why the aliases stay.

    Parameters
    ----------
    name : str
        The attribute being looked up on this module.

    Returns
    -------
    type
        :class:`mortie.pandas.MortonIndexDtype` or
        :class:`mortie.pandas.MortonIndexArray`.

    Raises
    ------
    AttributeError
        For any other name.
    ImportError
        If pandas is not installed (via :func:`_require_pandas`).
    """
    if name in ("MortonIndexDtype", "MortonIndexArray"):
        from . import pandas as _pandas_ext

        return getattr(_pandas_ext, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Import the pandas skin eagerly *iff* pandas is already importable, so that
# ``pd.Series(dtype="morton_index")`` resolves the registered name (the
# ``@register_extension_dtype`` runs at class creation) without the user first
# touching the classes. A numpy-only environment skips this silently (the
# classes still resolve on demand via __getattr__, raising a clear error).
try:
    import pandas as _pd  # noqa: F401

    from . import pandas as _pandas_ext  # noqa: F401
except ImportError:
    pass

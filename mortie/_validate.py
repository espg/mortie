"""Shared strict input validators for word and offset arrays (issue #194).

The toc module's validation discipline, hoisted to one home so the whole
family answers bad input the same way: refuse float-typed words and offsets
instead of truncating them, range-check before any narrowing cast instead of
wrapping, and name the parameter and the offending value.  The strict form
has caught two real bug classes -- the issue #185 uncatchable-panic arc and
PR #192's silent uint64 wrap -- so with issue #187's consolidation giving
each operation one polymorphic entry point, that posture is applied at every
choke point rather than kept as a toc-only stance.

Zero-size input is the one deliberate acceptance: an untyped empty container
(``[]``, ``()``, ``np.array([])``) is float64 by numpy's default, but it is
not numeric, it is empty (the ruling :class:`~mortie.toc_object.Toc` already
applied to its source argument) -- so it passes through as a typed empty
array rather than being refused for a dtype it never chose.
"""

import numpy as np


def _as_u64(values, name):
    """Validate non-negative integer input and return it as uint64.

    Float input is refused rather than truncated, and negative input is
    refused rather than wrapped -- packed words are unsigned, and a negative
    here is almost always a signed *reinterpretation* of a word whose top
    bit is set (base cells 7-11; spec section 1) or a legacy signed id, both
    of which would wrap into a different, possibly valid, word.

    Parameters
    ----------
    values : array_like
        Integer-typed values (any shape); zero-size input of any dtype is
        accepted as empty.
    name : str
        Parameter name to blame in refusal messages.

    Returns
    -------
    numpy.ndarray
        The values as ``uint64``, at least 1-D; no copy when the input is
        already ``uint64``.

    Raises
    ------
    ValueError
        If ``values`` is not integer-typed, or any value is negative --
        naming ``name`` and the first offending value.
    """
    arr = np.atleast_1d(np.asarray(values))
    if arr.size == 0:
        return arr.astype(np.uint64)
    if arr.dtype.kind not in "iu":
        raise ValueError(
            f"{name} must be integer-typed, got dtype {arr.dtype}")
    if arr.dtype.kind == "i":
        flat = arr.ravel()
        neg = flat[flat < 0]
        if neg.size:
            raise ValueError(
                f"{name} must be non-negative, got {int(neg[0])}")
    return arr.astype(np.uint64, copy=False)


def _as_i64(values, name):
    """Validate int64-representable integer input and return it as int64.

    The signed counterpart of :func:`_as_u64`, for encodings whose working
    dtype is ``int64`` (UNIQ ids, arrow offsets): float input is refused
    rather than truncated, and a value the cast cannot represent -- a
    ``uint64`` at or above ``2**63``, or a Python int outside int64 -- is
    refused naming the value rather than wrapped or left to numpy's own
    error.  Negative values pass: they are representable, and the caller's
    domain check owns their refusal (and its message).

    Parameters
    ----------
    values : array_like
        Integer-typed values (any shape); zero-size input of any dtype is
        accepted as empty.
    name : str
        Parameter name to blame in refusal messages.

    Returns
    -------
    numpy.ndarray
        The values as ``int64``, at least 1-D; no copy when the input is
        already ``int64``.

    Raises
    ------
    ValueError
        If ``values`` is not integer-typed, or a value does not fit in
        ``int64`` -- naming ``name`` and the first offending value.  Every
        refusal is this family's own message: no numpy cast error or warning
        (strings, ``None``, ``NaN``) surfaces in its place.
    """
    arr = np.atleast_1d(np.asarray(values))
    if arr.size == 0:
        return arr.astype(np.int64)
    if arr.dtype.kind not in "iu":
        # A Python int past int64 lands as float64 (or, further out, object)
        # in the untyped asarray above, so an oversized *integer* would
        # otherwise be blamed on its promoted dtype.  Look for one directly
        # rather than probe-casting (issue #194 review).
        #
        # Only an *untyped* container can hide an int inside a float64
        # promotion.  An input that already carries a numpy dtype (ndarray,
        # numpy scalar, pandas Series) holds numpy floats, and
        # `asarray(..., dtype=object)` on it yields Python floats -- so the
        # probe cannot fire, and would only materialize an object list the
        # size of the column in front of a refusal it cannot change (0.33 s
        # and ~96 MB on a 5M-element float64 UNIQ column, the likeliest way
        # to reach this validator).  Gate it on the two cases that can carry
        # an oversized int: an object array, or an untyped container.
        if arr.dtype.kind == "O" or getattr(values, "dtype", None) is None:
            flat = np.atleast_1d(np.asarray(values, dtype=object)).ravel()
            bad = next((v for v in flat.tolist() if isinstance(v, int)
                        and not -2**63 <= v < 2**63), None)
            if bad is not None:
                raise ValueError(f"{name} must fit in int64, got {bad}")
        raise ValueError(
            f"{name} must be integer-typed, got dtype {arr.dtype}")
    if arr.dtype.kind == "u":
        too_big = arr > np.iinfo(np.int64).max
        if too_big.any():
            raise ValueError(
                f"{name} must fit in int64, got {int(arr[too_big][0])}")
    return arr.astype(np.int64, copy=False)


def _as_offsets(offsets):
    """Validate arrow list offsets and return them as contiguous int64.

    Integer-typed by the same rule :func:`_as_u64` applies to words: a float
    offset array would otherwise cast silently, truncating ``2.9`` to a group
    boundary at 2 rather than saying so.  The same standard rules out the
    ``uint64`` values the cast cannot represent -- at or above ``2**63`` they
    would wrap negative, and the Rust validator would then describe the
    wrapped copy rather than the offset that was passed.  Monotonicity and
    bounds stay the Rust validator's job -- it names the offending group.

    Parameters
    ----------
    offsets : array_like
        Integer-typed arrow list offsets; zero-size input of any dtype is
        accepted and left for the kernel's own emptiness refusal.

    Returns
    -------
    numpy.ndarray
        The offsets as a contiguous 1-D ``int64`` array.

    Raises
    ------
    ValueError
        If ``offsets`` is not integer-typed, or a value is at or above
        ``2**63`` -- naming the first offending value.  Every refusal is
        this family's own message: no numpy cast error or warning (strings,
        ``None``, ``NaN``) is allowed to surface in its place.
    """
    return np.ascontiguousarray(_as_i64(offsets, "offsets").ravel())

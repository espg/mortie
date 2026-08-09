"""toc word -- temporal order coverage (issue #175).

One ``uint64``, a tagged union of an exact nanosecond **timestamp** and a
quantized, conservative **time range**, templated on what the morton word
does for space: self-describing, sortable as a plain unsigned integer, and
closed under a semilattice merge.  All internal times are u64 nanoseconds
since **1850-01-01T00:00:00** on a continuous, leap-free, GPS-aligned
timescale; leap seconds exist only at the UTC boundary
(:func:`from_datetime64` / :func:`to_datetime64`).

Layout (bit 0 = LSB)::

    [start: 32 bits, 2^31 ns units][flag: 1 bit][low: 31 bits]

- **timestamp** (flag = 1): the word is ``t_ns`` with the flag bit spliced
  in at position 31 -- monotone in ``t_ns``.
- **range** (flag = 0): ``low`` is the end code, 31 bits at 2^32 ns; the
  encoded envelope is half-open ``[start * 2^31, end * 2^32)``, rounded
  outward with a strictly-greater end ceiling so it always properly
  contains the real interval.

Unsigned word order is order by conservative encoded start; within a tied
start quantum, ranges sort before timestamps, then timestamps by exact ns
and ranges shorter-first.  The layout, epoch, merge results, and sort
order are normative (words persist on disk) and pinned by golden fixtures;
see the decision ledger on `issue #175
<https://github.com/espg/mortie/issues/175>`_ and the design record on
`zagg#410 <https://github.com/englacial/zagg/issues/410>`_.

Naming note: "toc" pleasantly echoes tick/tock and T-MOC, but this is
**not** an IVOA T-MOC and does not conform to the IVOA MOC 2.0
recommendation (different epoch, timescale, and cell model -- see the
T-MOC research on zagg#410 for why the hierarchical cell was rejected).

These flat-array elementwise ops are the type's scalar surface, mirroring
the relationship :mod:`mortie.moc` has to its ops over one cover; the
ragged many-cover plurals land in :mod:`mortie.batch` when the interval-set
algebra (issue #177) activates.
"""

import operator

import numpy as np

from . import _rustie

#: Start quantum: 2^31 ns (~2.15 s); a range's start code floors to this.
Q_START_NS = 1 << 31
#: End quantum: 2^32 ns (~4.29 s); a range's end code ceils to this.
Q_END_NS = 1 << 32
#: Exclusive ceiling on internal times: 2^63 - 2^32 ns past the epoch
#: (~4 s short of year 2142); the end code must fit its 31-bit field.
TOC_MAX_NS = (1 << 63) - (1 << 32)


def _as_u64(values, name):
    """Validate non-negative integer input and return it as uint64."""
    arr = np.atleast_1d(np.asarray(values))
    if arr.dtype.kind not in "iu":
        raise ValueError(
            f"{name} must be integer-typed, got dtype {arr.dtype}")
    if arr.dtype.kind == "i" and arr.size and np.any(arr < 0):
        raise ValueError(f"{name} must be non-negative")
    return arr.astype(np.uint64)


def _as_scalar_ns(value, name):
    """Validate a scalar ns argument and return it as a plain int."""
    value = operator.index(value)
    if not 0 <= value < 1 << 64:
        raise ValueError(f"{name} must lie in [0, 2**64)")
    return value


def time2toc(t_ns):
    """Encode exact instants (internal ns) as timestamp words.

    The word is ``t_ns`` with a 1 flag bit spliced in at position 31, so
    unsigned word order over timestamps is exactly the ns order.

    Parameters
    ----------
    t_ns : int or array-like
        Instant(s) in nanoseconds since 1850-01-01T00:00:00 on the
        continuous GPS-aligned scale, each in ``[0, TOC_MAX_NS)``.

    Returns
    -------
    int or ndarray
        Timestamp word(s), ``uint64`` for array input (scalar in ->
        ``int`` out).

    Raises
    ------
    ValueError
        If any instant is negative, non-integer-typed, or at or beyond
        ``TOC_MAX_NS``.

    See Also
    --------
    span2toc : encode a real interval as a range word.
    toc2time : decode words back to conservative bounds.
    """
    is_scalar = np.isscalar(t_ns)
    t = _as_u64(t_ns, "t_ns")
    words = _rustie.rust_time2toc(np.ascontiguousarray(t.ravel()))
    words = words.reshape(t.shape)
    if is_scalar:
        return int(words[0])
    return words


def span2toc(start_ns, end_ns):
    """Encode real closed intervals ``[start, end]`` as range words.

    Outward rounding with a strictly-greater end ceiling: the start code
    floors to the 2^31 ns grid and the end code is ``(end >> 32) + 1`` --
    uniformly, including when ``end`` sits exactly on the 2^32 ns grid --
    so the encoded envelope always properly contains the interval.
    ``start_ns`` and ``end_ns`` broadcast against each other.

    Parameters
    ----------
    start_ns : int or array-like
        Interval start(s) in internal ns.
    end_ns : int or array-like
        Interval end(s) in internal ns (inclusive real endpoint), each
        ``>= start_ns`` and ``< TOC_MAX_NS``.

    Returns
    -------
    int or ndarray
        Range word(s), ``uint64`` for array input (scalar in -> ``int``
        out).

    Raises
    ------
    ValueError
        If any start is after its end, or any end is at or beyond
        ``TOC_MAX_NS`` (the 31-bit end code would overflow -- ~4 s short
        of the year-2142 span ceiling; rejected rather than wrapped).

    See Also
    --------
    time2toc : encode an exact instant.
    toc2time : decode words back to conservative bounds.
    """
    is_scalar = np.isscalar(start_ns) and np.isscalar(end_ns)
    starts, ends = np.broadcast_arrays(_as_u64(start_ns, "start_ns"),
                                       _as_u64(end_ns, "end_ns"))
    words = _rustie.rust_span2toc(np.ascontiguousarray(starts.ravel()),
                                  np.ascontiguousarray(ends.ravel()))
    words = words.reshape(starts.shape)
    if is_scalar:
        return int(words[0])
    return words


def toc2time(words):
    """Decode toc words to conservative ``(start_ns, end_ns)`` bounds.

    A timestamp yields its exact instant twice, ``(t, t)``.  A range
    yields its half-open envelope bounds: ``end_ns`` is **exclusive**,
    strictly greater than every instant the range covers (the encoder's
    strictly-greater ceiling guarantees this even for interval ends that
    sat exactly on the 2^32 ns grid).

    Parameters
    ----------
    words : int or array-like
        Toc word(s) (``uint64``).

    Returns
    -------
    start_ns : int or ndarray
        Conservative start(s) in internal ns, ``uint64`` for array input
        (scalar in -> ``int`` out).
    end_ns : int or ndarray
        Exact instant (timestamps) or exclusive envelope end (ranges),
        same convention as ``start_ns``.

    Raises
    ------
    ValueError
        If ``words`` is negative or non-integer-typed.

    See Also
    --------
    toc_is_range : which variant each word is.
    """
    is_scalar = np.isscalar(words)
    w = _as_u64(words, "words")
    starts, ends = _rustie.rust_toc2time(np.ascontiguousarray(w.ravel()))
    starts, ends = starts.reshape(w.shape), ends.reshape(w.shape)
    if is_scalar:
        return int(starts[0]), int(ends[0])
    return starts, ends


def toc_merge(a, b):
    """Merge two toc words elementwise (the semilattice join).

    Bitwise-equal inputs return that word unchanged -- merging two equal
    timestamps must not produce their range envelope.  Any other pair
    merges conservative envelopes (min of start codes, max of end codes)
    into a range word.  Exactly associative, commutative, and idempotent,
    so any fold tree over the same words produces the identical ``uint64``.
    ``a`` and ``b`` broadcast against each other.

    Parameters
    ----------
    a : int or array-like
        Toc word(s) (``uint64``).
    b : int or array-like
        Toc word(s) (``uint64``).

    Returns
    -------
    int or ndarray
        Merged word(s), ``uint64`` for array input (scalar in -> ``int``
        out).

    Raises
    ------
    ValueError
        If either input is negative or non-integer-typed.

    See Also
    --------
    toc_reduce : merge a whole array to one word.
    """
    is_scalar = np.isscalar(a) and np.isscalar(b)
    wa, wb = np.broadcast_arrays(_as_u64(a, "a"), _as_u64(b, "b"))
    merged = _rustie.rust_toc_merge(np.ascontiguousarray(wa.ravel()),
                                    np.ascontiguousarray(wb.ravel()))
    merged = merged.reshape(wa.shape)
    if is_scalar:
        return int(merged[0])
    return merged


def toc_reduce(words):
    """Merge an array of toc words down to one word.

    The fold tree is unspecified (parallel under the hood) -- safe because
    the merge is exactly associative, commutative, and idempotent, so
    every fold tree produces the identical ``uint64``.

    Parameters
    ----------
    words : array-like
        Toc words (``uint64``), at least one.

    Returns
    -------
    int
        The merged word.

    Raises
    ------
    ValueError
        If ``words`` is empty (the merge has no identity element), or is
        negative or non-integer-typed.

    See Also
    --------
    toc_merge : the elementwise pairwise form.
    """
    w = _as_u64(words, "words")
    return int(_rustie.rust_toc_reduce(np.ascontiguousarray(w.ravel())))


def toc_is_range(words):
    """Test which variant each toc word is.

    Parameters
    ----------
    words : int or array-like
        Toc word(s) (``uint64``).

    Returns
    -------
    bool or ndarray
        True where the word is a range (flag bit 31 = 0), False for a
        timestamp; ``bool`` ndarray for array input (scalar in ->
        ``bool`` out).

    Raises
    ------
    ValueError
        If ``words`` is negative or non-integer-typed.
    """
    is_scalar = np.isscalar(words)
    w = _as_u64(words, "words")
    flags = _rustie.rust_toc_is_range(np.ascontiguousarray(w.ravel()))
    flags = flags.reshape(w.shape)
    if is_scalar:
        return bool(flags[0])
    return flags


def toc_overlaps(words, q_start_ns, q_end_ns):
    """Test which words intersect a half-open query window.

    The test runs on each word's **conservative encoded bounds** (a
    timestamp is its exact instant; a range is its outward-rounded
    envelope), against the half-open window ``[q_start_ns, q_end_ns)``.
    Consequently it may **over-report** near window edges by up to one
    quantum (a range whose envelope grazes the window without its real
    interval doing so), and it **never under-reports**: every word whose
    real time content intersects the window tests True.

    Parameters
    ----------
    words : int or array-like
        Toc word(s) (``uint64``).
    q_start_ns : int
        Window start in internal ns (inclusive).
    q_end_ns : int
        Window end in internal ns (exclusive), ``>= q_start_ns``.

    Returns
    -------
    bool or ndarray
        True where the word's conservative bounds intersect the window;
        ``bool`` ndarray for array input (scalar in -> ``bool`` out).

    Raises
    ------
    ValueError
        If the window is inverted, or ``words`` is negative or
        non-integer-typed.

    See Also
    --------
    toc_contains : containment in the window instead of intersection.
    """
    return _window(words, q_start_ns, q_end_ns, 0)


def toc_contains(words, q_start_ns, q_end_ns):
    """Test which words are contained in a half-open query window.

    The test runs on each word's **conservative encoded bounds** against
    the half-open window ``[q_start_ns, q_end_ns)``.  Consequently it
    **never over-reports** (the real interval is inside the envelope, so
    envelope-in-window implies interval-in-window), and it may
    **under-report** near window edges by up to one quantum -- a range
    whose real interval fits the window but whose outward-rounded
    envelope spills past an edge tests False.

    Parameters
    ----------
    words : int or array-like
        Toc word(s) (``uint64``).
    q_start_ns : int
        Window start in internal ns (inclusive).
    q_end_ns : int
        Window end in internal ns (exclusive), ``>= q_start_ns``.

    Returns
    -------
    bool or ndarray
        True where the word's conservative bounds fit inside the window;
        ``bool`` ndarray for array input (scalar in -> ``bool`` out).

    Raises
    ------
    ValueError
        If the window is inverted, or ``words`` is negative or
        non-integer-typed.

    See Also
    --------
    toc_overlaps : intersection with the window instead of containment.
    """
    return _window(words, q_start_ns, q_end_ns, 1)


def _window(words, q_start_ns, q_end_ns, mode):
    """Run one window predicate (0 = overlaps, 1 = contains)."""
    is_scalar = np.isscalar(words)
    w = _as_u64(words, "words")
    hits = _rustie.rust_toc_window(np.ascontiguousarray(w.ravel()),
                                   _as_scalar_ns(q_start_ns, "q_start_ns"),
                                   _as_scalar_ns(q_end_ns, "q_end_ns"),
                                   mode)
    hits = hits.reshape(w.shape)
    if is_scalar:
        return bool(hits[0])
    return hits

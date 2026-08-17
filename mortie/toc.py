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
the relationship :mod:`mortie.moc` has to its ops over one cover.
:func:`tocs_reduce` is the one ragged operator here: the segmented sibling
of :func:`toc_reduce`, kept beside its scalar because it is a fold over the
word type itself rather than an op over covers (issue #177).  The
many-*cover* plurals still land in :mod:`mortie.batch`, and wait on the
interval-set algebra, which stays deferred for want of a consumer.
"""

import operator

import numpy as np

from . import _rustie

Q_START_NS = 1 << 31
"""Start quantum: 2^31 ns (~2.15 s); a range's start code floors to this."""

Q_END_NS = 1 << 32
"""End quantum: 2^32 ns (~4.29 s); a range's end code ceils to this."""

TOC_MAX_NS = (1 << 63) - (1 << 32)
"""Exclusive ceiling on internal times: 2^63 - 2^32 ns past the epoch
(~4 s short of year 2142); the end code must fit its 31-bit field."""


def _as_u64(values, name):
    """Validate non-negative integer input and return it as uint64."""
    arr = np.atleast_1d(np.asarray(values))
    if arr.dtype.kind not in "iu":
        raise ValueError(
            f"{name} must be integer-typed, got dtype {arr.dtype}")
    if arr.dtype.kind == "i" and arr.size and np.any(arr < 0):
        raise ValueError(f"{name} must be non-negative")
    return arr.astype(np.uint64)


def _as_offsets(offsets):
    """Validate arrow list offsets and return them as contiguous int64.

    Integer-typed by the same rule :func:`_as_u64` applies to words: a float
    offset array would otherwise cast silently, truncating ``2.9`` to a group
    boundary at 2 rather than saying so.  Range and monotonicity are the Rust
    validator's job -- it names the offending group.
    """
    arr = np.atleast_1d(np.asarray(offsets))
    if arr.dtype.kind not in "iu":
        raise ValueError(
            f"offsets must be integer-typed, got dtype {arr.dtype}")
    return np.ascontiguousarray(arr.astype(np.int64).ravel())


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
    tocs_reduce : the segmented form, one word per group.
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
    tocs_reduce : the segmented form, one word per group.
    """
    w = _as_u64(words, "words")
    return int(_rustie.rust_toc_reduce(np.ascontiguousarray(w.ravel())))


def tocs_reduce(words, offsets):
    """Merge each group of toc words down to one word, in one call.

    The **segmented** sibling of :func:`toc_reduce` (issue #177), named in the
    batch family's plural convention (``mocs_and``, ``mocs_to_orders``): the
    whole ragged group set crosses the Python/Rust boundary once, the GIL is
    released for the batch, and Rust parallelizes across groups.  Result ``i``
    is bit-identical to ``toc_reduce(words[offsets[i]:offsets[i + 1]])`` — same
    join, same instant preservation (a group of bitwise-equal timestamps comes
    back as that timestamp, not as its range envelope), same fold-tree
    independence.

    Input is ragged in the arrow list layout the batch family uses; the
    **output is dense** — one ``uint64`` per group — because the reduction is
    many→one per group, so there are no output offsets to carry.  The consumer
    this exists for is a per-cell fold: zagg's GEDI shot pooling and its ATL03
    overview envelopes-of-envelopes both run the scalar reduce once per cell
    (`zagg#410 <https://github.com/englacial/zagg/issues/410>`_), which is the
    Python loop this call replaces.

    Parameters
    ----------
    words : array-like
        Flat toc words (``uint64``), all groups concatenated.
    offsets : array-like
        ``int64`` arrow list offsets: group ``i`` spans
        ``[offsets[i], offsets[i + 1])``, so there are ``len(offsets) - 1``
        groups.  The offsets must **exactly cover** ``words`` --
        ``offsets[0] == 0`` and ``offsets[-1] == len(words)`` -- so a sliced
        arrow array must be re-based before it gets here; anything else is an
        error naming the endpoint that failed.  An **empty group is an
        error**, not an empty slot: the merge has no identity element, so
        many→one over no words has no answer, exactly as :func:`toc_reduce`
        refuses an empty array.

    Returns
    -------
    numpy.ndarray
        ``uint64`` array of ``len(offsets) - 1`` merged words, one per group.

    Raises
    ------
    ValueError
        Fail-fast, naming the offending group (e.g. ``group 4217:
        tocs_reduce of an empty segment ...``): an empty group, or a *layout*
        failure -- non-monotone / out-of-bounds offsets, or offsets that do not
        exactly cover ``words`` (the message names which endpoint failed).
        Also if ``words`` is negative or non-integer-typed.  The index named
        is the lowest-index offender **within its pass**: layout is checked for
        the whole batch first, so a layout failure at a high index is reported
        ahead of an empty group at a low one -- deliberate, since the group
        indices an empty-group error is reported *by* are read out of
        ``offsets``.

    See Also
    --------
    toc_reduce : the whole-array (one group) form.
    toc_merge : the elementwise pairwise join both reduce with.

    Examples
    --------
    Two cells' shot times fold to one word each:

    >>> import mortie, numpy as np
    >>> w = mortie.time2toc(np.array([10, 20, 30, 40], dtype=np.uint64) * 10**9)
    >>> got = mortie.tocs_reduce(w, [0, 3, 4])
    >>> int(got[0]) == mortie.toc_reduce(w[:3]) and int(got[1]) == int(w[3])
    True
    """
    w = _as_u64(words, "words")
    return np.asarray(_rustie.rust_tocs_reduce(
        np.ascontiguousarray(w.ravel()), _as_offsets(offsets)))


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
    real time content intersects the window tests True.  An empty window
    (``q_start_ns == q_end_ns``) matches nothing.

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
    envelope spills past an edge tests False.  An empty window
    (``q_start_ns == q_end_ns``) contains nothing.

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


# ---------------------------------------------------------------------------
# Timescale boundary (issue #175 phase 3). The internal scale is continuous
# and leap-free; leap seconds are handled exactly once, here, when crossing
# to or from UTC. GPS interop is a pure constant offset.
# ---------------------------------------------------------------------------

GPS_EPOCH_NS = 47_486 * 86_400 * 10**9
"""Internal ns of the GPS epoch 1980-01-06T00:00:00: 47,486
proleptic-Gregorian days of 86,400 s past 1850-01-01 (the leap-free
scale ticks exactly with GPS, so the constant is a plain day count --
validated against ``datetime.date`` arithmetic in the test suite)."""

# Internal/naive ns between 1850-01-01 and the numpy datetime64 epoch
# 1970-01-01: 43,829 proleptic-Gregorian days of 86,400 s.
_EPOCH_1850_1970_NS = 43_829 * 86_400 * 10**9

# Static leap-second table: (UTC step date, TAI-UTC after the step).
# Data, not a dependency; complete as of the 2017-01-01 step (+18 s
# GPS-UTC), and no further leap second is currently scheduled.  Update by
# appending when the IERS announces one.
_LEAP_TABLE = (
    ("1972-01-01", 10),
    ("1972-07-01", 11),
    ("1973-01-01", 12),
    ("1974-01-01", 13),
    ("1975-01-01", 14),
    ("1976-01-01", 15),
    ("1977-01-01", 16),
    ("1978-01-01", 17),
    ("1979-01-01", 18),
    ("1980-01-01", 19),
    ("1981-07-01", 20),
    ("1982-07-01", 21),
    ("1983-07-01", 22),
    ("1985-07-01", 23),
    ("1988-01-01", 24),
    ("1990-01-01", 25),
    ("1991-01-01", 26),
    ("1992-07-01", 27),
    ("1993-07-01", 28),
    ("1994-07-01", 29),
    ("1996-01-01", 30),
    ("1997-07-01", 31),
    ("1999-01-01", 32),
    ("2006-01-01", 33),
    ("2009-01-01", 34),
    ("2012-07-01", 35),
    ("2015-07-01", 36),
    ("2017-01-01", 37),
)

# UTC step instants as naive ns since 1970 (datetime64 arithmetic), and the
# internal-ns offset in force from each step on: GPS - UTC = TAI - UTC - 19.
# _OFFSETS_NS[0] is the pre-1972 proleptic convention: zero (see
# from_datetime64); at the GPS epoch itself TAI - UTC = 19, so the offset
# is exactly zero there -- the identity the GPS alignment pins.
_UTC_STEPS_1970_NS = np.array(
    [np.datetime64(d, "ns").astype(np.int64) for d, _ in _LEAP_TABLE],
    dtype=np.int64)
_OFFSETS_NS = np.array(
    [0] + [(tai - 19) * 10**9 for _, tai in _LEAP_TABLE], dtype=np.int64)
# The same step instants on the internal scale (naive + the offset the step
# switches to), for the inverse lookup.
_INTERNAL_STEPS_NS = (
    _UTC_STEPS_1970_NS + _EPOCH_1850_1970_NS + _OFFSETS_NS[1:]
).astype(np.uint64)


def from_datetime64(when):
    """Convert UTC ``datetime64`` times to internal ns.

    The internal scale is continuous and GPS-aligned: from 1972 the offset
    to naive UTC day-count time is ``GPS - UTC = TAI - UTC - 19`` from the
    static in-module leap-second table (zero at the GPS epoch 1980-01-06,
    +18 s since 2017-01-01).  **Before 1972** the proleptic convention is
    **zero offset** (naive day-count seconds, no leap adjustment): it pins
    the epoch identity (1850-01-01T00:00:00 -> 0 ns exactly), whereas
    freezing the 1972 offset of -9 s would push the epoch itself to a
    negative, unrepresentable internal time.  Pre-1972 "UTC" was not
    SI-second aligned anyway, and such data carries hours-level precision.
    Cost: the mapping steps back 9 s across the 1972-01-01 boundary, so
    the last 9 SI seconds of 1971 are not invertible (they alias early
    1972); conversion is exact and invertible from 1972 on.

    Parameters
    ----------
    when : datetime64, str, or array-like
        UTC instant(s); anything ``np.asarray(when, 'datetime64[ns]')``
        accepts.

    Returns
    -------
    int or ndarray
        Internal ns (``uint64``) since 1850-01-01T00:00:00 on the
        continuous GPS-aligned scale; scalar in -> ``int`` out.

    Raises
    ------
    ValueError
        If any instant is before the 1850 epoch or at or beyond
        ``TOC_MAX_NS`` (the toc span ceiling, ~year 2142).

    See Also
    --------
    to_datetime64 : the inverse conversion.
    from_gps_ns : the leap-free GPS entry point.
    """
    # A 0-d ndarray classifies as array (matching np.isscalar in the word
    # ops); np.isscalar itself is unusable here -- it is False for a
    # np.datetime64 scalar.
    is_scalar = np.ndim(when) == 0 and not isinstance(when, np.ndarray)
    naive = np.atleast_1d(
        np.asarray(when, dtype="datetime64[ns]")).astype(np.int64)
    # Ceiling guard on the *internal* result, evaluated before the offset
    # addition so the int64 arithmetic below cannot overflow: the offset in
    # force anywhere near the ceiling is the table's last entry, so every
    # naive value at or past this limit maps to internal >= TOC_MAX_NS.
    limit = TOC_MAX_NS - _EPOCH_1850_1970_NS - int(_OFFSETS_NS[-1])
    if naive.size and int(naive.max()) >= limit:
        raise ValueError(
            "datetime64 input is at or beyond the toc span ceiling "
            "(~year 2142)")
    idx = np.searchsorted(_UTC_STEPS_1970_NS, naive, side="right")
    internal = naive + _EPOCH_1850_1970_NS + _OFFSETS_NS[idx]
    if internal.size and int(internal.min()) < 0:
        raise ValueError("datetime64 input is before the 1850-01-01 epoch")
    internal = internal.astype(np.uint64)
    if is_scalar:
        return int(internal[0])
    return internal


def to_datetime64(t_ns):
    """Convert internal ns to UTC ``datetime64[ns]`` times.

    The inverse of :func:`from_datetime64` (same leap table, same pre-1972
    zero-offset convention).  Internal instants that fall *inside* an
    inserted leap second render into the following UTC second --
    ``datetime64`` cannot express 23:59:60 -- so, e.g., the middle of the
    2016-12-31 leap second renders as 2017-01-01T00:00:00.5.  Roundtrip
    ``to_datetime64(from_datetime64(t))`` is exact for every ``datetime64``
    from 1972 on (no ``datetime64`` names a leap-second instant).

    Parameters
    ----------
    t_ns : int or array-like
        Internal ns (``uint64``), each below ``2**63``.

    Returns
    -------
    datetime64 or ndarray
        UTC instant(s) as ``datetime64[ns]``; scalar in -> ``np.datetime64``
        out.

    Raises
    ------
    ValueError
        If any value is negative, non-integer-typed, or at or beyond
        ``2**63`` (past every representable envelope bound).

    See Also
    --------
    from_datetime64 : the inverse conversion.
    to_gps_ns : the leap-free GPS exit point.
    """
    is_scalar = np.isscalar(t_ns)
    t = _as_u64(t_ns, "t_ns")
    if t.size and int(t.max()) >= 1 << 63:
        raise ValueError("t_ns must lie below 2**63")
    idx = np.searchsorted(_INTERNAL_STEPS_NS, t, side="right")
    naive = t.astype(np.int64) - _OFFSETS_NS[idx] - _EPOCH_1850_1970_NS
    out = naive.astype("datetime64[ns]")
    if is_scalar:
        return out[0]
    return out


def from_gps_ns(gps_ns):
    """Convert GPS time (ns since 1980-01-06T00:00:00) to internal ns.

    A pure constant offset: both scales are continuous and leap-free and
    tick together, so the conversion is ``gps_ns + GPS_EPOCH_NS``.  The
    ICESat-2 ingest path (``delta_time`` + ``atlas_sdp_gps_epoch`` -> GPS
    ns) composes with this directly.

    Parameters
    ----------
    gps_ns : int or array-like
        GPS time(s) in ns since the GPS epoch, each below
        ``TOC_MAX_NS - GPS_EPOCH_NS``.

    Returns
    -------
    int or ndarray
        Internal ns (``uint64``); scalar in -> ``int`` out.

    Raises
    ------
    ValueError
        If any value is negative, non-integer-typed, or maps at or beyond
        ``TOC_MAX_NS``.

    See Also
    --------
    to_gps_ns : the inverse conversion.
    """
    is_scalar = np.isscalar(gps_ns)
    g = _as_u64(gps_ns, "gps_ns")
    if g.size and int(g.max()) >= TOC_MAX_NS - GPS_EPOCH_NS:
        raise ValueError(
            "gps_ns input maps at or beyond the toc span ceiling "
            "(~year 2142)")
    internal = g + np.uint64(GPS_EPOCH_NS)
    if is_scalar:
        return int(internal[0])
    return internal


def to_gps_ns(t_ns):
    """Convert internal ns to GPS time (ns since 1980-01-06T00:00:00).

    The exact inverse of :func:`from_gps_ns`.  Internal times before the
    GPS epoch have no non-negative GPS representation and are rejected.

    Parameters
    ----------
    t_ns : int or array-like
        Internal ns (``uint64``), each at or after ``GPS_EPOCH_NS``.

    Returns
    -------
    int or ndarray
        GPS ns (``uint64``); scalar in -> ``int`` out.

    Raises
    ------
    ValueError
        If any value is negative, non-integer-typed, or before the GPS
        epoch.

    See Also
    --------
    from_gps_ns : the inverse conversion.
    """
    is_scalar = np.isscalar(t_ns)
    t = _as_u64(t_ns, "t_ns")
    if t.size and int(t.min()) < GPS_EPOCH_NS:
        raise ValueError(
            "t_ns is before the GPS epoch 1980-01-06 (internal ns "
            f"{GPS_EPOCH_NS}); no non-negative GPS time exists")
    gps = t - np.uint64(GPS_EPOCH_NS)
    if is_scalar:
        return int(gps[0])
    return gps

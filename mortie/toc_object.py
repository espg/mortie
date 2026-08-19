"""The time-first ``Toc`` object over the toc kernel (issue #198).

mortie's temporal-coverage surface is two layers, the same deliberate split
:mod:`mortie.moc_object` documents for space:

* the **kernel** -- the free ``toc_*`` functions in :mod:`mortie._toc`, words
  in and words out, unchanged and un-deprecated.  Array-first consumers
  (zagg's per-cell folds, the segmented :func:`~mortie.tocs_reduce`) keep
  calling them on plain ndarrays at zero wrapping cost.
* the **object** -- :class:`Toc`, here.  It is ergonomics and nothing else: a
  thin view over the canonical ``uint64`` word set
  (:func:`~mortie.toc_normalize`'s sorted maximal merges), never a new
  representation.  **Every public method is a single delegation to a kernel
  function.**  There is no algebra in this module; a method body that is not
  one kernel call is a bug, not a feature.

All three object methods delegate to the same kernel,
:func:`~mortie.toc_and` -- the one set operation the issue #177 call-site
audit ruled in.  The method names are the MOCpy-adjacent vocabulary (issue
#198 commitment 6), so they deliberately do **not** line up with the batch
layer's :func:`~mortie.toc_overlaps` / :func:`~mortie.toc_contains`, which
stay un-deprecated and ask a different question: those take a
``[q_start_ns, q_end_ns)`` query window and answer elementwise, per word,
while :meth:`Toc.overlaps` / :meth:`Toc.contains` compare two whole covers
and answer once.  "The object method is the kernel of the same name with
``self`` bound" is the wrong reading here; each method docstring names the
kernel it actually calls.

The interchange format stays the array.  :meth:`Toc.__toc_words__` hands back
the canonical words -- the temporal sibling of ``__morton_moc__()`` -- and any
object exposing that dunder is accepted wherever a ``Toc`` is: mortie owns the
word grammar, downstream stores own their own encodings, and the two meet at a
plain ``uint64`` array with neither importing the other's private grammar.

**Conservative directions.**  Every predicate below is *envelope* algebra, not
data algebra.  A range word's decoded envelope dilates the real interval it
was encoded from (starts floor to the 2^31 ns grid, ends ceil to 2^32 ns), so
the covered timeline is always a superset of the real one, on both sides of
the comparison.  Read the answers accordingly; each method's docstring points
back here.

| call | the question it answers exactly | as a question about the real data |
| --- | --- | --- |
| `a.overlaps(b)` | do the two covers share any decoded instant? | may say `True` for data that only comes within a quantum (~2-4 s) of each other; a `False` is decisive. The safe direction for "must I read this store?" -- never a false skip. |
| `a.contains(b)` | is `b`'s decoded coverage inside `a`'s? | may err either way by up to a quantum at span edges (each side's envelope dilates its own data). Exact for the question that matters -- "will a store whose coverage is `a` answer a query for `b`?" |
| `a & b`, `a.intersection(b)` | the canonical cover of the shared coverage | never under-covers the true intersection (`A ⊇ X` and `B ⊇ Y` imply `A ∩ B ⊇ X ∩ Y`); may over-cover near piece edges by up to one quantum per side, inherited from the operands. |
| either side empty | an empty cover is contained in everything and overlaps nothing | `a.contains(empty)` is `True` (vacuously -- there is no coverage of `empty` outside `a`) while `a.overlaps(empty)` is `False`. The two predicates disagree here by definition, not by accident: ask `contains` about coverage and `overlaps` about work to do. |
| `a == b` | identical canonical words | **not** equality of the underlying timeline: an instant and the degenerate span at the same time encode different words and compare unequal. |
"""

import warnings

import numpy as np

from . import _toc
from ._toc import (
    from_datetime64,
    span2toc,
    time2toc,
    to_datetime64,
    toc2time,
    toc_and,
    toc_is_range,
    toc_normalize,
)

# The public surface of the former ``mortie.toc`` submodule -- what the
# migration shim below still resolves (with a DeprecationWarning) for one minor
# version.  A frozen historical roster (the 0.9.9 surface: thirteen
# module-level functions plus the four grid/epoch constants), not a live view
# of the kernel: pinned as a literal, and pinned in the tests against that same
# history, so a name added to the kernel later does not join the deprecated
# namespace.  ``toc_normalize`` and ``toc_and`` are deliberately absent: born
# in this same PR, they never had a ``mortie.toc.<name>`` spelling any release
# could be using, so there is nothing there to deprecate.
_KERNEL_NAMES = (
    "GPS_EPOCH_NS",
    "Q_END_NS",
    "Q_START_NS",
    "TOC_MAX_NS",
    "from_datetime64",
    "from_gps_ns",
    "span2toc",
    "time2toc",
    "to_datetime64",
    "to_gps_ns",
    "toc2time",
    "toc_contains",
    "toc_is_range",
    "toc_merge",
    "toc_overlaps",
    "toc_reduce",
    "tocs_reduce",
)


def _words(operand):
    """Canonical ``uint64`` toc words of a set-operation operand.

    *Every* operand is canonicalized here (:func:`~mortie.toc_normalize`),
    protocol carriers included: nothing obliges a third-party
    ``__toc_words__()`` to hand back canonical form, and the equality-shaped
    delegation in :meth:`Toc.contains` compares canonical form against
    canonical form or it false-negatives.  Normalizing already-canonical
    words is a fixpoint, so the :class:`Toc`-to-:class:`Toc` path is
    unchanged.

    Parameters
    ----------
    operand : Toc or array_like
        A :class:`Toc`, anything exposing ``__toc_words__()``, or a raw toc
        word array.

    Returns
    -------
    numpy.ndarray
        The operand's canonical words as a 1-D ``uint64`` array.
    """
    protocol = getattr(operand, "__toc_words__", None)
    if protocol is not None:
        operand = protocol()
    return toc_normalize(operand)


def _as_time_ns(value, name):
    """Coerce one time endpoint to internal ns, refusing numeric input.

    Integers are always *words* in the :class:`Toc` constructor, and numpy
    would otherwise read a bare number as naive ns since 1970 -- silently, on
    the wrong epoch -- so numeric endpoints are refused rather than guessed.

    Parameters
    ----------
    value : str, datetime64, or array_like
        The endpoint(s): anything :func:`~mortie.from_datetime64` accepts.
    name : str
        Argument name for the error message.

    Returns
    -------
    int or numpy.ndarray
        Internal ns since 1850-01-01 (scalar in -> ``int`` out).

    Raises
    ------
    ValueError
        If the endpoint is numeric rather than datetime64 / ISO string, or
        :func:`~mortie.from_datetime64` rejects it.
    """
    if np.asarray(value).dtype.kind in "iufcb":
        raise ValueError(
            f"{name} must be a datetime64 / ISO-string time, got a numeric "
            f"value; integers are toc words in this constructor, and "
            f"internal-ns integers go through time2toc/span2toc explicitly")
    return from_datetime64(value)


def _source_words(source, end):
    """Resolve a constructor source to its toc words, before normalization.

    Parameters
    ----------
    source : object
        Any :class:`Toc` constructor source; see that class for the matrix.
    end : str, datetime64, array_like, or None
        The span-end side, for the time-pair form only.

    Returns
    -------
    int or numpy.ndarray
        Toc words (``uint64``), not yet normalized.

    Raises
    ------
    ValueError
        If ``end`` is given for a source that is already words -- it
        completes a time pair, which such a source is not, so it is refused
        rather than ignored -- or if an integer source is not 1-D: an
        ``(N, 2)`` array of ``[start, end]`` pairs is a natural thing to have
        in hand, and raveling it into words yields a plausible-looking cover
        of garbage, so the shape is gated the way :class:`~mortie.Moc` gates
        its own words branch.

    Notes
    -----
    A size-0 non-datetime, non-string source is the **empty word set** in
    every spelling -- ``[]``, ``()``, ``np.array([])`` and a typed
    ``uint64`` empty alike.  Untyped empty containers are ``float64`` by
    numpy's default, so without that route they would reach the time path
    and be refused as *numeric*, advising the caller to do exactly what they
    were already doing.
    """
    protocol = getattr(source, "__toc_words__", None)
    if protocol is not None:
        if end is not None:
            raise ValueError(
                "end= completes a time pair and applies to a time source "
                "only; this source is already a cover (__toc_words__)")
        return np.asarray(protocol(), dtype=np.uint64).ravel()
    arr = np.asarray(source)
    if arr.dtype.kind in "iu":
        if end is not None:
            raise ValueError(
                "end= completes a time pair and applies to a time source "
                "only; this source is already toc words")
        if arr.ndim > 1:
            raise ValueError(
                f"toc words must be a 1-D integer array or a single int, got "
                f"shape {arr.shape}; an (N, 2) array of time pairs goes "
                f"through Toc(starts, ends)")
        return source
    if arr.size == 0 and arr.dtype.kind not in "MUSO":
        # An untyped empty container ([], (), np.array([])) is float64 by
        # numpy's default, so it would fall to the time path and be reported
        # as *numeric*.  It is not numeric, it is empty: the empty cover.
        if end is not None:
            raise ValueError(
                "end= completes a time pair and applies to a time source "
                "only; this source is already toc words")
        return arr.astype(np.uint64)
    if end is None:
        return time2toc(_as_time_ns(source, "source"))
    return span2toc(_as_time_ns(source, "source"), _as_time_ns(end, "end"))


_COVERED_UNITS = (
    ("years", 31_557_600 * 10**9),
    ("days", 86_400 * 10**9),
    ("hours", 3_600 * 10**9),
    ("minutes", 60 * 10**9),
    ("seconds", 10**9),
    ("ms", 10**6),
    ("us", 10**3),
)


def _covered_display(total_ns):
    """Render a covered duration in the largest unit it fills.

    Parameters
    ----------
    total_ns : int
        Total covered nanoseconds.

    Returns
    -------
    str
        The duration with its unit, e.g. ``"517 days"``.
    """
    for unit, scale in _COVERED_UNITS:
        if total_ns >= scale:
            return f"{total_ns / scale:.4g} {unit}"
    return f"{total_ns} ns"


class Toc:
    """A temporal coverage as an object -- gappy time coverage that reads as time.

    A thin view over the canonical ``uint64`` toc word set and nothing more:
    the words *are* the coverage, this class is the ergonomics, and every
    public method is a single delegation to a kernel function (see the module
    docstring for the two-layer rule and the conservative-direction table the
    methods share).

    The canonical form is a word **set**, not a single envelope (issue #198):
    a store observed in campaigns has *gappy* coverage, and one merged
    envelope papers over the gaps exactly where they are most informative --
    k disjoint spans keep them.  The words are normalized eagerly
    (:func:`~mortie.toc_normalize`) and stored read-only, which makes ``==``
    and :func:`hash` well defined; the instance itself is immutable.

    Normalization is **lossy toward coverage, one way** (the #177 Q2 ruling):
    a timestamp subsumed by a range's decoded span adds no coverage and is
    absorbed at construction -- which subsumed instants existed, and how many
    times, is dropped.  Exact instants live in the sibling word arrays a
    cover is built from; a cover can be rebuilt from the arrays, never the
    arrays from a cover.  A timestamp no range subsumes survives
    bit-identical as an exact degenerate member, and a surviving decoded gap
    is never bridged (Q1).

    Union needs no method: construction normalizes, so
    ``Toc(np.append(a.words, b.words))`` is the union.  The #177 call-site
    audit ruled exactly one set operation in (:func:`~mortie.toc_and`, the
    :meth:`intersection` / ``&`` delegate); the difference and
    symmetric-difference directions deliberately do not ship -- conservative
    covers under-cover on subtraction, and no audited call site exists.

    Parameters
    ----------
    source : str, datetime64, array_like, or Toc
        What the coverage is.  A UTC instant -- an ISO 8601 string or
        ``numpy.datetime64`` -- or an array of them, encoded exactly via
        :func:`~mortie.time2toc`; with ``end`` given, the start side of one
        or more closed real intervals ``[source, end]``, enveloped
        conservatively via :func:`~mortie.span2toc` (``source`` and ``end``
        broadcast).  A 1-D integer array (or a single int) is taken as toc
        **words** as given; any object exposing ``__toc_words__()``, another
        :class:`Toc` included, contributes its canonical words.  Integers are
        always words: internal nanoseconds go through
        :func:`~mortie.time2toc` / :func:`~mortie.span2toc` explicitly, so a
        bare number can never be misread as a time on the wrong epoch.  Any
        size-0 source that is not datetimes or strings -- ``[]``, ``()``,
        ``np.array([])``, a typed empty -- is the empty cover.
    end : str, datetime64, or array_like, optional
        End side of the time-pair form, same forms as ``source``.  Only a
        time source takes it: with a words or protocol source it is refused
        rather than ignored.

    Raises
    ------
    ValueError
        If a time endpoint is numeric rather than datetime64 / ISO string, a
        span is inverted or outside the representable epoch range (see
        :func:`~mortie.span2toc` / :func:`~mortie.from_datetime64`), words
        are negative, non-integer-typed or not 1-D, or ``end`` is given for a
        source that is already words.

    See Also
    --------
    mortie.toc_normalize : the eager canonicalization applied to every
        instance.
    mortie.toc_and : the intersection kernel the set methods delegate to.

    Examples
    --------
    >>> import mortie
    >>> when = mortie.Toc("2020-01-01", "2021-06-01")
    >>> mortie.Toc("2020-06-15").overlaps(when)
    True
    >>> when.contains(mortie.Toc("2020-03-01", "2020-04-01"))
    True
    """

    __slots__ = ("words",)

    def __init__(self, source, end=None):
        words = toc_normalize(_source_words(source, end))
        words.setflags(write=False)
        object.__setattr__(self, "words", words)

    def __setattr__(self, name, value):
        """Refuse attribute assignment -- a Toc is immutable."""
        raise AttributeError(
            "Toc is immutable (its hash is its words); build a new one instead"
        )

    def __delattr__(self, name):
        """Refuse attribute deletion -- a Toc is immutable."""
        raise AttributeError("Toc is immutable; build a new one instead")

    def __reduce__(self):
        """Rebuild through the constructor -- pickle and copy cannot set slots.

        A ``__slots__`` class is restored by assigning its slot state, which
        the immutability guard above refuses; reconstructing from the words
        instead keeps ``Toc`` picklable (workers marshal their arguments) and
        deep-copyable at the cost of one ``toc_normalize`` on already-canonical
        words.

        Returns
        -------
        tuple
            The ``(callable, args)`` pair the pickle protocol rebuilds from.
        """
        return (Toc, (self.words,))

    def overlaps(self, other):
        """Whether this cover and ``other`` share any coverage.

        Envelope algebra, not data algebra -- see the module docstring's
        conservative-direction table: ``True`` can mean "within a quantum of
        each other", ``False`` is decisive.

        Parameters
        ----------
        other : Toc or array_like
            The cover to test against.

        Returns
        -------
        bool
            ``True`` if the two covers share any decoded instant.

        See Also
        --------
        mortie.toc_and : the kernel this delegates to.
        mortie.toc_overlaps : the batch layer's *window* predicate of the
            same name, a different question -- which individual words
            intersect a ``[q_start_ns, q_end_ns)`` window, elementwise.
        """
        return toc_and(self.words, _words(other)).size > 0

    def contains(self, other):
        """Whether ``other``'s coverage lies entirely inside this cover.

        Envelope algebra, not data algebra -- see the module docstring's
        conservative-direction table.  This is the "will a store whose
        coverage is ``self`` answer a query for ``other``?" test: the
        intersection with ``other`` must give ``other`` back whole, and both
        sides of that comparison are canonical -- the kernel's output by
        construction, the operand because :func:`_words` canonicalizes every
        operand it is handed -- so word equality is coverage equality.  The
        single-delegation shape costs a second :func:`_words` call (a foreign
        ``__toc_words__()`` is therefore invoked twice per test); binding it
        to a local would be a second statement, which the delegation pin
        refuses.

        Parameters
        ----------
        other : Toc or array_like
            The cover to test for containment.

        Returns
        -------
        bool
            ``True`` if ``other`` adds no coverage outside this cover.

        See Also
        --------
        mortie.toc_and : the kernel this delegates to.
        mortie.toc_contains : the batch layer's *window* predicate of the
            same name, and the mirror direction -- which individual words
            fall inside a ``[q_start_ns, q_end_ns)`` window, elementwise,
            rather than whether one cover sits inside another.
        """
        return np.array_equal(toc_and(self.words, _words(other)), _words(other))

    def intersection(self, other):
        """Intersect with ``other``: the canonical cover of the shared coverage (``&``).

        Never under-covers the true intersection and may over-cover near
        piece edges by up to one quantum per side, inherited from the
        operands' envelopes -- see the module docstring's table.

        Parameters
        ----------
        other : Toc or array_like
            The cover to intersect with.

        Returns
        -------
        Toc
            The intersection cover.
        """
        return Toc(toc_and(self.words, _words(other)))

    __and__ = intersection

    def __toc_words__(self):
        """Canonical toc words -- the interchange protocol (issue #198).

        Returns
        -------
        numpy.ndarray
            The read-only ``uint64`` word array backing this cover.
        """
        return self.words

    def __eq__(self, other):
        """Compare canonical words -- word identity, not timeline equality."""
        if not isinstance(other, Toc):
            return NotImplemented
        return np.array_equal(self.words, other.words)

    def __hash__(self):
        """Hash the canonical words; sound because a Toc is immutable."""
        return hash(self.words.tobytes())

    def __len__(self):
        """Count the words (disjoint spans and free instants) in the cover."""
        return int(self.words.size)

    def __iter__(self):
        """Iterate the cover's toc words."""
        return iter(self.words)

    def __repr__(self):
        """Show the span count, the UTC extent, and the covered duration.

        Both printed bounds round **outward** to the second -- the start
        floors, the end ceils -- so the extent shown is never narrower than
        what the cover covers.  A decoded end sits on the 2^32 ns grid
        (~4.295 s) and so is essentially never a whole second; flooring it
        would print a time strictly inside the envelope, the one direction
        the module docstring's conservative-direction table rules out.
        """
        if self.words.size == 0:
            return "Toc(0 spans)"
        ranges = toc_is_range(self.words)
        n_ranges, n_instants = int(ranges.sum()), int((~ranges).sum())
        kinds = []
        if n_ranges:
            kinds.append(f"{n_ranges} range" + ("s" if n_ranges > 1 else ""))
        if n_instants:
            kinds.append(
                f"{n_instants} instant" + ("s" if n_instants > 1 else ""))
        starts, ends = toc2time(self.words)
        first = np.datetime64(to_datetime64(int(starts.min())), "s")
        end = to_datetime64(int(ends.max()))
        last = np.datetime64(end, "s")
        if last != end:
            last += np.timedelta64(1, "s")
        covered = _covered_display(int((ends - starts).sum()))
        return (f"Toc({' + '.join(kinds)}, {first} to {last}, "
                f"{covered} covered)")


class _TocNamespace:
    """Callable stand-in for the retired ``mortie.toc`` submodule (issue #198).

    Calling it builds a :class:`Toc`; attribute access to the kernel functions
    and constants the submodule used to hold still resolves, with a
    :class:`DeprecationWarning`, for one minor version.  Statement-form
    ``import mortie.toc`` and ``from mortie.toc import ...`` break at the
    rename -- the module is gone -- which is why the shim covers attribute
    access and not the import system.

    The warning is raised on **every** access and the shim keeps no state, so
    policy is left entirely to the warnings filters.  Under the interpreter
    defaults that means ``DeprecationWarning`` is ignored outside ``__main__``
    and ``stacklevel=2`` charges it to the *calling* module, so a consumer
    module sees nothing until its filters ask -- ``-W``, ``PYTHONWARNINGS``, or
    a test runner that enables the category (pytest does).  ``always`` and
    ``error`` filters then see every occurrence, and a
    :func:`warnings.catch_warnings` block starts from a fresh registry; with
    no shim-side budget to exhaust, a downstream test suite still observes the
    warning however late in the process it runs.
    """

    __slots__ = ()

    def __call__(self, source, end=None):
        """Build a :class:`Toc`; see that class for the argument matrix."""
        return Toc(source, end)

    def __getattr__(self, name):
        """Resolve a retired submodule attribute, warning on every access."""
        if name not in _KERNEL_NAMES:
            raise AttributeError(
                f"'mortie.toc' is the Toc constructor (issue #198), not the "
                f"old submodule, and has no attribute {name!r}"
            )
        warnings.warn(
            f"mortie.toc.{name} is deprecated: mortie.toc is now the Toc "
            f"constructor, not a module. Use the top-level mortie.{name} "
            f"instead; this shim is removed in the next minor release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(_toc, name)

    def __dir__(self):
        """List the deprecated kernel names this shim still resolves."""
        return sorted(_KERNEL_NAMES)

    def __repr__(self):
        """Say what this object is, so `mortie.toc` is not mistaken for a module."""
        return "<mortie.toc: the Toc constructor; mortie.toc.<kernel> is deprecated>"


toc = _TocNamespace()
"""The :class:`Toc` constructor, bound where the submodule used to be."""

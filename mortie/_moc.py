"""Multi-Order Coverage (MOC) algebra over morton sets.

A mortie MOC is just a ``uint64`` array of packed morton words at mixed orders —
the word self-encodes its own order and ancestry — so every operation here is an
array in, array out.  :func:`compress_moc` is the canonical compaction,
:func:`moc_to_order` densifies back to a flat single-order list,
:func:`moc_or` / :func:`moc_and` /
:func:`moc_minus` / :func:`moc_xor` are the healpix-crate BMOC set algebra,
:func:`moc_intersects` the intersection predicate (no BMOC build, no
materialized result),
:func:`moc_not` its domain-bounded complement, and :func:`common_ancestor` /
:func:`split_base_cells` the ancestry reductions.  All of it is computed in Rust
— there is no Python-level MOC set algebra.

Split out of :mod:`mortie.coverage` by **domain** (issue #156): MOC algebra
against polygon coverage.  The ragged batch kernels behind the vectorized
operators here live in :mod:`mortie.batch` (issue #170) as private functions,
reached through each operator's keyword-only ``offsets=`` form — the plural
names that used to export them retired in the polymorphic consolidation
(issue #187).  The pyarrow skins stay in :mod:`mortie.arrow` (issue #154).
The names stay flat on the package (``mortie.moc_to_order``): the module is
where they live, not how they are spelled.

Renamed from ``mortie/moc.py`` to ``mortie/_moc.py`` for issue #196, which
frees the ``mortie.moc`` name for the :class:`~mortie.moc_object.Moc`
constructor.  Nothing here changed and nothing here is deprecated: these
free functions are the **kernel layer** — words in, words out, no wrapping
cost — and the array-first consumers keep calling them on plain ndarrays.
:class:`~mortie.moc_object.Moc` is the **object layer** over them, and every
one of its methods is a single delegation to a function on this page.
"""

import warnings

import numpy as np

from . import _rustie
from .batch import (
    _common_ancestors,
    _mocs_and,
    _mocs_intersect,
    _mocs_to_orders,
)
from .convert import norm2mort
from .coverage import _FLAT_COVER_WARN_THRESHOLD


def compress_moc(morton):
    """Compress a morton set into its canonical compact MOC.

    Merges any 4 complete sibling cells into their parent (repeatedly) and drops
    any cell already contained in a coarser one.  Use after unioning covers from
    several polygons / parts so that sibling groups spanning the seams collapse.

    **Not batch vectorized**: one MOC per call.

    Parameters
    ----------
    morton : array_like
        Morton indices (mixed order allowed).

    Returns
    -------
    numpy.ndarray
        Sorted, compacted morton indices (``uint64``).
    """
    morton = np.asarray(morton, dtype=np.uint64).ravel()
    return np.asarray(_rustie.rust_moc_normalize(morton))


def moc_to_order(morton, order, max_cells=_FLAT_COVER_WARN_THRESHOLD, *,
                 offsets=None):
    """Densify a (mixed-order) morton set to a flat list at ``order``.

    Unlike :func:`morton_coverage`'s post-hoc warning, the densify path can
    over-allocate to the point of OOM before any warning is reachable — a tiny
    compact MOC densifies to ``Σ 4**(order - depth)`` flat cells (issue #80).
    So this guards **pre-emptively**: an upper bound on the densified count is
    computed from the input set alone (an O(n) pass, no flat allocation) and,
    when it exceeds ``max_cells``, a :class:`ValueError` is raised *before*
    materializing.  The bound is exact unless ``morton`` holds cells finer than
    ``order`` (which coarsen and dedup on densify), where it is a safe over-count
    — so the guard never lets more than ``max_cells`` cells through.

    ``order`` is range-checked here, in the wrapper, for the same reason: the
    refusal must be catchable by the handlers consumers already have (issue
    #108), and a Rust panic surfaces as ``pyo3_runtime.PanicException``, which
    derives from :class:`BaseException` — so neither ``except ValueError`` nor
    ``except Exception`` catches it.  The kernel's densify shift is defined only
    over 0-29, and past that ``1 << (2 * (order - depth))`` used to wrap mod 64
    in a release build rather than trap: for depth-6 input the whole band
    ``order`` 38-48 estimated *under* the default budget and passed straight
    through to the panic.  That shift now refuses out of range in Rust too and
    the binding maps it to the same :class:`ValueError` (issue #161), so this
    check is defence in depth rather than the only defence.

    **Batch vectorized** (issue #187): one MOC in, one flat cover out; pass
    ``offsets`` and the same call densifies a whole ragged column of MOCs in
    one crossing, returning the ``(values, out_offsets)`` pair.  Passing
    ``offsets`` is what selects the form: a MOC is itself an array, so unlike
    :func:`~mortie.morton_index.decimal_to_word` there is no rank difference
    to dispatch on.

    Parameters
    ----------
    morton : array_like
        Morton indices (mixed order allowed).  With ``offsets``, the flat
        ``uint64`` concatenation of every MOC in the column.
    order : int
        Target HEALPix order (0-29) to densify to, shared by every MOC.
    max_cells : int or None, optional
        Pre-emptive budget on the densified flat cell count.  Raises
        :class:`ValueError` if the estimate exceeds it (default
        ``1 << 20`` — the same ~1M-cell line as the flat-cover warning).  Pass
        ``None`` to opt out and densify unconditionally.  With ``offsets`` the
        budget applies **per MOC**, and the refusal names the lowest-index
        offending MOC *within its kind* — offset-layout errors are screened in
        their own pass ahead of the budget, so a bad layout is reported before
        an over-budget MOC at a lower index.
    offsets : array_like or None, optional
        ``int64`` arrow list offsets selecting the ragged batch form: MOC ``i``
        spans ``morton[offsets[i]:offsets[i + 1]]``, and the offsets must
        exactly cover ``morton``.  ``None`` (default) is the single-MOC form.

    Returns
    -------
    numpy.ndarray or tuple of numpy.ndarray
        Without ``offsets``, a sorted 1-D array of flat morton indices at
        ``order`` (``uint64``).  With ``offsets``, the ragged
        ``(values, out_offsets)`` pair — slice ``i`` is byte-identical to the
        single-MOC result on MOC ``i`` alone.

    Raises
    ------
    ValueError
        If ``order`` is outside 0-29, or the estimated densified count exceeds
        ``max_cells``.  In the ragged form, also for offsets that are
        non-monotone, out of bounds, or do not exactly cover ``morton``.

    See Also
    --------
    morton_coverage : flat single-order cover (post-hoc large-cover warning).
    mortie.batch._mocs_to_orders : the ragged batch kernel this delegates to.
    """
    if offsets is not None:
        return _mocs_to_orders(morton, offsets, order, max_cells)
    morton = np.asarray(morton, dtype=np.uint64).ravel()
    if not 0 <= order <= 29:
        raise ValueError(f"Order must be between 0 and 29, got {order}")
    if max_cells is not None:
        estimated = int(_rustie.rust_moc_to_order_count(morton, order))
        if estimated > max_cells:
            raise ValueError(
                f"moc_to_order would densify to ~{estimated} cells at order "
                f"{order}, exceeding max_cells={max_cells}. Pass a larger "
                f"max_cells, or max_cells=None to proceed (risking OOM), or "
                f"densify to a coarser order."
            )
    return np.asarray(_rustie.rust_moc_to_order(morton, order))


def moc_or(a, b):
    r"""Union of two morton covers (the cells in ``a`` or ``b``).

    Equivalent to ``compress_moc(concatenate([a, b]))``, but computed by the
    healpix-crate BMOC ``or`` rather than a concatenate-then-compress pass.

    **Not batch vectorized**: one pair of covers per call — there is no batch
    kernel behind it, unlike :func:`moc_and`.

    Parameters
    ----------
    a, b : array_like
        Morton covers (mixed order allowed).

    Returns
    -------
    numpy.ndarray
        Sorted, compacted union (``uint64``).

    See Also
    --------
    moc_and : intersection of two covers.
    moc_minus : difference ``a \ b``.
    compress_moc : ``moc_or(a, b) == compress_moc(concatenate([a, b]))``.
    """
    a = np.asarray(a, dtype=np.uint64).ravel()
    b = np.asarray(b, dtype=np.uint64).ravel()
    return np.asarray(_rustie.rust_moc_or(a, b))


def moc_and(a, b, *, offsets=None):
    r"""Intersection of two morton covers (the cells in both ``a`` and ``b``).

    **Batch vectorized** (issue #187): pass ``offsets`` and ``b`` is read as a
    ragged column of MOCs, all intersected against the shared cover ``a`` in
    one crossing.  The broadcast builds ``a``'s BMOC once instead of once per
    item, which is the structural win a Python loop cannot get.

    Parameters
    ----------
    a, b : array_like
        Morton covers (mixed order allowed).  With ``offsets``, ``b`` is the
        flat concatenation of the column and ``a`` stays the shared operand —
        so the two are **not** interchangeable in the batch form even though
        the operation itself is commutative, and swapping them is a different
        question rather than an error.
    offsets : array_like or None, optional
        ``int64`` arrow list offsets selecting the ragged batch form: MOC ``i``
        spans ``b[offsets[i]:offsets[i + 1]]``, and the offsets must exactly
        cover ``b``.  ``None`` (default) is the two-cover form.

    Returns
    -------
    numpy.ndarray or tuple of numpy.ndarray
        Without ``offsets``, the sorted, compacted intersection (``uint64``).
        With ``offsets``, the ragged ``(values, out_offsets)`` pair; an empty
        intersection keeps its slot.

    See Also
    --------
    moc_or : union of two covers.
    moc_minus : difference ``a \ b``.
    moc_intersects : tests for overlap without materializing this result.
    mortie.batch._mocs_and : the 1 x N broadcast kernel this delegates to.
    """
    if offsets is not None:
        return _mocs_and(a, b, offsets)
    a = np.asarray(a, dtype=np.uint64).ravel()
    b = np.asarray(b, dtype=np.uint64).ravel()
    return np.asarray(_rustie.rust_moc_and(a, b))


def moc_intersects(a, b, *, offsets=None):
    """Whether two morton covers intersect (share any area at any order).

    The predicate twin of :func:`moc_and` (issue #173): ``moc_intersects(a, b)``
    equals ``moc_and(a, b).size > 0``, but materializes no intersection — both
    covers are normalized (the only allocation) and walked as sorted disjoint
    ranges, exiting on the first overlap.  It is compaction-safe by construction: it tests geometric
    overlap, never identity against a compacted cover, so a dense region that
    compacts to its parent still answers ``True`` for any cell inside it.

    **Batch vectorized** (issue #187): pass ``offsets`` and ``b`` is read as a
    ragged column of MOCs, each tested against the shared cover ``a``, giving
    one ``bool`` per item.

    Parameters
    ----------
    a, b : array_like
        Morton covers (mixed order allowed).  With ``offsets``, ``b`` is the
        flat concatenation of the column and ``a`` stays the shared operand —
        so the two are **not** interchangeable in the batch form even though
        the operation itself is commutative, and swapping them is a different
        question rather than an error.
    offsets : array_like or None, optional
        ``int64`` arrow list offsets selecting the ragged batch form: MOC ``i``
        spans ``b[offsets[i]:offsets[i + 1]]``, and the offsets must exactly
        cover ``b``.  ``None`` (default) is the two-cover form.

    Returns
    -------
    bool or numpy.ndarray
        Without ``offsets``, ``True`` if the two covers share any area.  With
        ``offsets``, a ``bool`` array of length ``len(offsets) - 1``, agreeing
        item-for-item with the non-empty slots of the ``offsets`` form of
        :func:`moc_and`.

    See Also
    --------
    moc_and : materializes the intersection this only tests.
    mortie.batch._mocs_intersect : the 1 x N broadcast kernel this delegates
        to.
    """
    if offsets is not None:
        return _mocs_intersect(a, b, offsets)
    a = np.asarray(a, dtype=np.uint64).ravel()
    b = np.asarray(b, dtype=np.uint64).ravel()
    return bool(_rustie.rust_moc_intersects(a, b))


def moc_minus(a, b):
    r"""Difference of two morton covers (the cells in ``a`` but not ``b``).

    Computes ``a \ b``.

    **Not batch vectorized**: one pair of covers per call — there is no batch
    kernel behind it, unlike :func:`moc_and`.

    Parameters
    ----------
    a, b : array_like
        Morton covers (mixed order allowed).

    Returns
    -------
    numpy.ndarray
        Sorted, compacted difference (``uint64``).

    See Also
    --------
    moc_or : union of two covers.
    moc_and : intersection of two covers.
    """
    a = np.asarray(a, dtype=np.uint64).ravel()
    b = np.asarray(b, dtype=np.uint64).ravel()
    return np.asarray(_rustie.rust_moc_minus(a, b))


def moc_xor(a, b):
    r"""Symmetric difference of two morton covers (cells in exactly one).

    Computes ``a △ b`` — the cells in ``a`` or ``b`` but not both, i.e.
    ``moc_minus(moc_or(a, b), moc_and(a, b))``.  Useful for "what changed"
    between two coverages: against an earlier cover ``a`` and a later cover
    ``b``, ``moc_xor`` is exactly the cells that gained *or* lost coverage.

    **Not batch vectorized**: one pair of covers per call — there is no batch
    kernel behind it, unlike :func:`moc_and`.

    Parameters
    ----------
    a, b : array_like
        Morton covers (mixed order allowed).

    Returns
    -------
    numpy.ndarray
        Sorted, compacted symmetric difference (``uint64``).

    See Also
    --------
    moc_or : union of two covers.
    moc_and : intersection of two covers.
    moc_minus : difference ``a \ b`` (the directional half of ``xor``).
    """
    a = np.asarray(a, dtype=np.uint64).ravel()
    b = np.asarray(b, dtype=np.uint64).ravel()
    return np.asarray(_rustie.rust_moc_xor(a, b))


def _whole_sphere():
    """Return the 12 order-0 HEALPix base cells as a morton cover.

    That cover is the whole sphere. Built via :func:`norm2mort` so it tracks
    the packed-u64 encoding (issue #58), not a hand-rolled constant.

    Returns
    -------
    numpy.ndarray
        The 12 order-0 base cells as packed morton words (``uint64``).
    """
    base = np.arange(12, dtype=np.int64)
    return np.asarray(norm2mort(np.zeros(12, dtype=np.int64), base, 0), dtype=np.uint64)


def moc_not(cover, domain=None):
    r"""Complement a morton cover within a domain.

    The result is the cells in ``domain`` but not ``cover``. A complement is
    only well-defined relative to a bounded domain, so ``moc_not`` is a
    domain-bounded difference: it returns ``domain \ cover``, i.e.
    ``moc_minus(domain, cover)``.

    **Not batch vectorized**: one cover per call — there is no batch kernel
    behind it, unlike :func:`moc_and`.

    Parameters
    ----------
    cover : array_like
        The morton cover to complement (mixed order allowed).
    domain : array_like, optional
        The morton cover to complement *within*.  A single morton index or a
        list/array of them (e.g. a coarse "shard" cell whose finer cells are
        enumerated in ``cover``).  Defaults to the whole sphere — the 12 order-0
        base cells.

    Returns
    -------
    numpy.ndarray
        Sorted, compacted complement ``domain \ cover`` (``uint64``).

    Warns
    -----
    UserWarning
        If ``cover`` contains cells outside ``domain``.  Such cells cannot be
        complemented within the domain, so they are **clipped**: the result is
        ``domain \ (cover ∩ domain)``, which equals ``domain \ cover`` whenever
        ``cover ⊆ domain``.

    See Also
    --------
    moc_minus : difference ``a \ b`` (``moc_not`` is ``moc_minus`` against a
        domain, with the whole-sphere default and an out-of-domain warning).

    Examples
    --------
    The shard case — a coarse cell with some finer cells enumerated inside it,
    asking for the finer cells *not* yet enumerated within the shard:

    >>> import mortie
    >>> shard = mortie.norm2mort(0, 0, 0)          # one order-0 base cell
    >>> enumerated = mortie.from_geometry(aoi, moc=True)              # doctest: +SKIP
    >>> gaps = mortie.moc_not(enumerated, domain=shard)               # doctest: +SKIP
    """
    cover = np.asarray(cover, dtype=np.uint64).ravel()
    if domain is None:
        domain = _whole_sphere()
    else:
        domain = np.asarray(domain, dtype=np.uint64).ravel()

    if domain.size == 0:
        # The complement within an empty domain is empty for any cover; the
        # out-of-domain warning would be vacuously true, so skip it.
        return np.asarray([], dtype=np.uint64)

    # Cells of `cover` outside `domain` cannot be complemented within it; warn
    # and clip them (the clip is implicit in `moc_minus(domain, cover)`, which
    # only ever subtracts the in-domain part of `cover`).
    if moc_minus(cover, domain).size > 0:
        warnings.warn(
            "moc_not: `cover` has cells outside `domain`; they cannot be "
            "complemented within the domain and are clipped away.",
            stacklevel=2,
        )

    return moc_minus(domain, cover)


def common_ancestor(morton, *, offsets=None):
    """Deepest common ancestor (highest-order common parent) of a morton set.

    The array-reduction sibling of :func:`clip2order` (coarsen): where coarsening
    lowers each word to a *caller-given* order, ``common_ancestor`` *discovers*
    the deepest order at which the whole input collapses to a single enclosing
    cell, and returns that one cell.  Because a packed morton word self-encodes
    its order and ancestry, this is the longest shared path prefix after the
    common base cell, capped at each word's own order — so mixed-order input is
    fine (each word is capped at its own order).

    **Batch vectorized** (issue #187): one group in, one word out; pass
    ``offsets`` and the same call reduces a whole ragged column of groups in
    one crossing, giving one word per group.

    Parameters
    ----------
    morton : array_like
        Morton indices (mixed order allowed).  A single index returns itself.
        With ``offsets``, the flat concatenation of every group in the column.
    offsets : array_like or None, optional
        ``int64`` arrow list offsets selecting the ragged batch form: group
        ``i`` spans ``morton[offsets[i]:offsets[i + 1]]``, and the offsets must
        exactly cover ``morton``.  ``None`` (default) is the one-group form.

    Returns
    -------
    numpy.uint64 or numpy.ndarray
        The packed morton index of the deepest cell that contains every input.
        A batch (more than one input) always yields an **area** cell — even when
        the inputs collapse to a single order-29 cell, since the shared cell is
        an enclosing area, not any one input point.  Only a single-element input
        is returned unchanged (its area/point kind preserved), so a lone area or
        point returns itself.  With ``offsets``, a ``uint64`` array holding that
        word for each group.

    Raises
    ------
    ValueError
        If ``morton`` is empty, contains an empty/invalid word, or spans more
        than one HEALPix base cell — there is then no common ancestor above the
        (non-existent) whole-sphere root.  In the ragged form the message names
        the lowest-index offending group *within its kind* (layout errors are
        screened in their own pass, ahead of the per-group content check), and
        bad offsets raise here too.

    See Also
    --------
    clip2order : coarsen each word to a fixed order (the elementwise form;
        ``common_ancestor`` is its reduce-by-common-coarsening reduction).
    mortie.batch._common_ancestors : the ragged batch kernel this delegates
        to.

    Examples
    --------
    The four order-5 children of an order-4 cell reduce to that parent:

    >>> import mortie, numpy as np
    >>> parent = mortie.norm2mort(11, 0, 4)              # one order-4 cell in base 0
    >>> kids = mortie.norm2mort([11 * 4 + s for s in range(4)], [0] * 4, 5)
    >>> int(mortie.common_ancestor(kids)) == int(parent)
    True
    """
    if offsets is not None:
        return _common_ancestors(morton, offsets)
    morton = np.asarray(morton, dtype=np.uint64).ravel()
    return np.uint64(_rustie.rust_moc_min(morton))


# ``moc_min`` is the MOC set-family alias for :func:`common_ancestor` (issue #61).
moc_min = common_ancestor


def split_base_cells(words, sort=False):
    """Partition a morton set by HEALPix base cell.

    Each group is keyed by its own :func:`moc_min`.
    The companion to :func:`moc_min` for the cross-base-cell case it refuses:
    where ``moc_min`` reduces a *single* base cell's words to one ancestor and
    raises on mixed base cells, ``split_base_cells`` groups the words by base
    cell and hands back each group untouched.  Every group is keyed by its own
    ``moc_min`` — the deepest cell enclosing that group — which is self-
    describing (a packed word the same 64 bits wide as the data) and from which
    the base cell id is cheap to recover (e.g. ``mort2healpix`` /
    ``MortonIndexArray.base_cell``).

    **Not batch vectorized**: one word set per call.

    Parameters
    ----------
    words : array_like
        Morton indices (mixed order and mixed base cell allowed).
    sort : bool, optional
        If ``False`` (default, the faster path) each group keeps the input
        order of its words.  If ``True`` each group's words are sorted, giving a
        canonical MOC per base cell.

    Returns
    -------
    dict[int, numpy.ndarray]
        Maps the ``int`` of each group's ``moc_min`` word to that group's
        ``uint64`` array of words.  Empty input returns ``{}``; a single base
        cell returns a one-entry dict.

    Raises
    ------
    ValueError
        If a group's ``moc_min`` reduction fails — e.g. ``words`` contains an
        empty/invalid word (``moc_min`` rejects it).

    See Also
    --------
    moc_min : the single-base-cell reduction this partitions for; its mixed-
        base-cell error points here.

    Examples
    --------
    >>> import mortie, numpy as np
    >>> a = np.atleast_1d(mortie.norm2mort(0, 2, 4))   # one cell in base 2
    >>> b = np.atleast_1d(mortie.norm2mort(0, 5, 4))   # one cell in base 5
    >>> groups = mortie.split_base_cells(np.concatenate([a, b]))
    >>> sorted(int(np.uint64(k) >> np.uint64(60)) - 1 for k in groups)
    [2, 5]
    """
    words = np.asarray(words, dtype=np.uint64).ravel()
    if words.size == 0:
        return {}

    bases = _rustie.rust_mi_base_cell_of(words)
    out = {}
    # Stable group-by: dict.fromkeys yields base cells in first-seen order, and
    # the boolean mask below keeps each group's words in input order.
    for base in dict.fromkeys(bases.tolist()):
        group = words[bases == base]
        if sort:
            group = np.sort(group)
        out[int(moc_min(group))] = group
    return out

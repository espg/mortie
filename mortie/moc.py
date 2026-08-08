"""Multi-Order Coverage (MOC) algebra over morton sets.

A mortie MOC is just a ``uint64`` array of packed morton words at mixed orders —
the word self-encodes its own order and ancestry — so every operation here is an
array in, array out.  :func:`compress_moc` is the canonical compaction,
:func:`moc_to_order` (and its ragged batch twin :func:`mocs_to_orders`) densifies
back to a flat single-order list, :func:`moc_or` / :func:`moc_and` /
:func:`moc_minus` / :func:`moc_xor` are the healpix-crate BMOC set algebra,
:func:`moc_not` its domain-bounded complement, and :func:`common_ancestor` /
:func:`split_base_cells` the ancestry reductions.  All of it is computed in Rust
— there is no Python-level MOC set algebra.

Split out of :mod:`mortie.coverage` (issue #156) so each scalar op sits beside
the plural batch twin it gains: the axis is the **domain** (MOC algebra vs
polygon coverage), not the arity, so ``moc_to_order`` and ``mocs_to_orders`` are
read together.  The names stay flat on the package (``mortie.moc_to_order``,
``mortie.mocs_to_orders``): this module is where they live, not how they are
spelled.
"""

import warnings

import numpy as np

from . import _rustie
from .coverage import _FLAT_COVER_WARN_THRESHOLD
from .tools import norm2mort


def compress_moc(morton):
    """Compress a morton set into its canonical compact MOC.

    Merges any 4 complete sibling cells into their parent (repeatedly) and drops
    any cell already contained in a coarser one.  Use after unioning covers from
    several polygons / parts so that sibling groups spanning the seams collapse.

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


def moc_to_order(morton, order, max_cells=_FLAT_COVER_WARN_THRESHOLD):
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

    Parameters
    ----------
    morton : array_like
        Morton indices (mixed order allowed).
    order : int
        Target HEALPix order to densify to.
    max_cells : int or None, optional
        Pre-emptive budget on the densified flat cell count.  Raises
        :class:`ValueError` if the estimate exceeds it (default
        ``1 << 20`` — the same ~1M-cell line as the flat-cover warning).  Pass
        ``None`` to opt out and densify unconditionally.

    Returns
    -------
    numpy.ndarray
        Sorted 1-D array of flat morton indices at ``order`` (``uint64``).

    Raises
    ------
    ValueError
        If the estimated densified count exceeds ``max_cells``.

    See Also
    --------
    morton_coverage : flat single-order cover (post-hoc large-cover warning).
    mocs_to_orders : the ragged batch form (many MOCs in one call).
    """
    morton = np.asarray(morton, dtype=np.uint64).ravel()
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


def mocs_to_orders(values, offsets, order, max_cells=_FLAT_COVER_WARN_THRESHOLD):
    """Densify many independent MOCs to a flat order in one call.

    The batch sibling of :func:`moc_to_order` (issue #156): the ragged MOC set
    crosses the Python/Rust boundary **once**, the GIL is released for the whole
    batch, and Rust parallelizes across MOCs — so the per-call fixed cost that
    dominates a Python loop over half a million covers is paid once.  Result
    ``i`` is byte-identical to :func:`moc_to_order` on MOC ``i`` alone.

    Input and output are ragged arrays in arrow list layout, **the same pair
    :func:`polygons_to_morton_mocs` returns** — so the two chain with no
    marshalling::

        cells, off = mortie.polygons_to_morton_mocs(lats, lons, off_in, order=8)
        flat, flat_off = mortie.mocs_to_orders(cells, off, 8)

    MOCs are densified in chunks and each chunk is copied into the ragged output
    as it lands, so peak memory is about the returned ``values`` array plus one
    chunk of in-flight lists — not the ~2.5x of holding every MOC's flat list to
    concatenate at the end.

    Parameters
    ----------
    values : array_like
        Flat ``uint64`` morton words, all MOCs concatenated.  Mixed orders
        allowed within each MOC, as in the scalar form.
    offsets : array_like
        ``int64`` arrow list offsets: MOC ``i`` spans
        ``[offsets[i], offsets[i + 1])``.  ``len(offsets) - 1`` MOCs.  The
        offsets must **exactly cover** ``values`` — ``offsets[0] == 0`` and
        ``offsets[-1] == len(values)`` — so a sliced arrow array must be
        re-based before it gets here; anything else is an error naming the
        endpoint that failed.  An empty MOC (``offsets[i] == offsets[i + 1]``)
        is legal and densifies to an empty slot.
    order : int
        Target HEALPix order (1-29) to densify to, shared by every MOC.
    max_cells : int or None, optional
        Pre-emptive budget on the densified flat cell count, applied **per
        MOC** exactly as :func:`moc_to_order` applies it to its one input
        (default ``1 << 20``).  A MOC whose estimate exceeds the budget raises
        :class:`ValueError` naming the **lowest-index** offending MOC, from a
        serial pre-pass — so the refusal costs no densify allocation, and it is
        the whole call that refuses, not that MOC alone.  Pass ``None`` to opt
        out.

    Returns
    -------
    values : numpy.ndarray
        All MOCs' flat cells at ``order`` concatenated (``uint64``).
    out_offsets : numpy.ndarray
        ``int64`` arrow list offsets into ``values``, length ``len(offsets)``;
        ``out_offsets[0]`` is always 0.

    Raises
    ------
    ValueError
        Fail-fast, naming the **lowest-index** offending MOC (e.g. ``moc 4217:
        moc_to_order would densify to ...``): a MOC over ``max_cells``, or
        non-monotone / out-of-bounds offsets.  Also for offsets that do not
        exactly cover ``values`` (``offsets[0] != 0``, or ``offsets[-1]`` short
        of or past ``len(values)`` — the message names which endpoint failed),
        or an ``order`` outside 1-29.

    See Also
    --------
    moc_to_order : the scalar (one MOC) form.
    polygons_to_morton_mocs : the batch coverer whose output feeds this
        verbatim.

    Notes
    -----
    Each slice comes back **sorted and unique** — the same guarantee
    :func:`moc_to_order` gives — so a downstream ``np.unique`` over a slice is
    redundant work, and ``np.searchsorted`` applies directly.

    Examples
    --------
    >>> import mortie, numpy as np
    >>> lats = np.array([40.0, 50.0, 45.0, 10.0, 20.0, 15.0])
    >>> lons = np.array([-120.0, -120.0, -110.0, -80.0, -80.0, -70.0])
    >>> mocs, off = mortie.polygons_to_morton_mocs(lats, lons, [0, 3, 6], order=6)
    >>> flat, flat_off = mortie.mocs_to_orders(mocs, off, 6)
    >>> first = flat[flat_off[0]:flat_off[1]]   # flat cover of the first triangle
    """
    values = np.ascontiguousarray(np.asarray(values, dtype=np.uint64).ravel())
    offsets = np.ascontiguousarray(np.asarray(offsets, dtype=np.int64).ravel())
    out_values, out_offsets = _rustie.rust_mocs_to_orders(
        values, offsets, order, max_cells
    )
    return np.asarray(out_values), np.asarray(out_offsets)


def moc_or(a, b):
    r"""Union of two morton covers (the cells in ``a`` or ``b``).

    Equivalent to ``compress_moc(concatenate([a, b]))``, but computed by the
    healpix-crate BMOC ``or`` rather than a concatenate-then-compress pass.

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


def moc_and(a, b):
    r"""Intersection of two morton covers (the cells in both ``a`` and ``b``).

    Parameters
    ----------
    a, b : array_like
        Morton covers (mixed order allowed).

    Returns
    -------
    numpy.ndarray
        Sorted, compacted intersection (``uint64``).

    See Also
    --------
    moc_or : union of two covers.
    moc_minus : difference ``a \ b``.
    """
    a = np.asarray(a, dtype=np.uint64).ravel()
    b = np.asarray(b, dtype=np.uint64).ravel()
    return np.asarray(_rustie.rust_moc_and(a, b))


def moc_minus(a, b):
    r"""Difference of two morton covers (the cells in ``a`` but not ``b``).

    Computes ``a \ b``.

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
    >>> enumerated = mortie.morton_coverage_moc(lats, lons, order=6)  # doctest: +SKIP
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


def common_ancestor(morton):
    """Deepest common ancestor (highest-order common parent) of a morton set.

    The array-reduction sibling of :func:`clip2order` (coarsen): where coarsening
    lowers each word to a *caller-given* order, ``common_ancestor`` *discovers*
    the deepest order at which the whole input collapses to a single enclosing
    cell, and returns that one cell.  Because a packed morton word self-encodes
    its order and ancestry, this is the longest shared path prefix after the
    common base cell, capped at each word's own order — so mixed-order input is
    fine (each word is capped at its own order).

    Parameters
    ----------
    morton : array_like
        Morton indices (mixed order allowed).  A single index returns itself.

    Returns
    -------
    numpy.uint64
        The packed morton index of the deepest cell that contains every input.
        A batch (more than one input) always yields an **area** cell — even when
        the inputs collapse to a single order-29 cell, since the shared cell is
        an enclosing area, not any one input point.  Only a single-element input
        is returned unchanged (its area/point kind preserved), so a lone area or
        point returns itself.

    Raises
    ------
    ValueError
        If ``morton`` is empty, contains an empty/invalid word, or spans more
        than one HEALPix base cell — there is then no common ancestor above the
        (non-existent) whole-sphere root.

    See Also
    --------
    clip2order : coarsen each word to a fixed order (the elementwise form;
        ``common_ancestor`` is its reduce-by-common-coarsening reduction).

    Examples
    --------
    The four order-5 children of an order-4 cell reduce to that parent:

    >>> import mortie, numpy as np
    >>> parent = mortie.norm2mort(11, 0, 4)              # one order-4 cell in base 0
    >>> kids = mortie.norm2mort([11 * 4 + s for s in range(4)], [0] * 4, 5)
    >>> int(mortie.common_ancestor(kids)) == int(parent)
    True
    """
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

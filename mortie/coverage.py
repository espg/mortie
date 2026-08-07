"""Polygon-to-morton coverage.

Compute the set of morton indices at a given order that completely cover
a polygon defined by lat/lon vertices.  Supports single and multipart
polygons.

Set algebra over covers (union / intersection / difference) is done in Rust via
:func:`moc_or`, :func:`moc_and`, and :func:`moc_minus` (healpix-crate BMOC), with
:func:`compress_moc` for the canonical compaction — there is no Python-level MOC
set algebra here.
"""

import warnings
from collections import namedtuple

import numpy as np

from . import _rustie
from .tools import norm2mort

# A flat cover's cell count scales as ~4**order along the boundary, so a flat
# `morton_coverage` at high order can grow to billions of cells and exhaust
# memory.  Above this many cells we warn and point at the compact MOC form
# (issue #60).  The warning is post-hoc (it fires after the cover is built), so
# it flags the hazard but cannot prevent an OOM at very high order — the MOC
# form is the real high-order path.  The true ceiling is this cell count, not
# the order itself.
_FLAT_COVER_WARN_THRESHOLD = 1 << 20  # ~1.05M cells (~8 MB of uint64)


def _warn_large_flat(n_cells, order):
    """Warn when a flat cover is large enough to be a memory hazard.

    Parameters
    ----------
    n_cells : int
        Number of cells in the materialized flat cover.
    order : int
        HEALPix order the cover was built at (reported in the warning).

    Warns
    -----
    UserWarning
        If ``n_cells`` exceeds ``_FLAT_COVER_WARN_THRESHOLD``.
    """
    if n_cells > _FLAT_COVER_WARN_THRESHOLD:
        warnings.warn(
            f"flat morton_coverage returned {n_cells} cells at order {order}; "
            f"a flat cover scales as ~4**order along the boundary and can "
            f"exhaust memory at high order. Use morton_coverage_moc(...) for a "
            f"compact mixed-order cover, or its max_cells= budget.",
            UserWarning,
            stacklevel=3,
        )


def _is_multipart(lats):
    """Check if *lats* is a sequence of sequences (multipart polygon).

    Parameters
    ----------
    lats : array_like or list of array_like
        Candidate vertex latitudes, single-ring or multipart.

    Returns
    -------
    bool
        ``True`` when *lats* holds one entry per ring rather than one per
        vertex.
    """
    if isinstance(lats, np.ndarray):
        return lats.ndim == 2
    if isinstance(lats, (list, tuple)) and len(lats) > 0:
        return isinstance(lats[0], (list, tuple, np.ndarray))
    return False


def _prep_rings(lats, lons):
    """Validate and convert a ring-set to parallel lists of float64 arrays.

    Parameters
    ----------
    lats, lons : list of array_like
        One entry per ring of vertex latitudes / longitudes in degrees.

    Returns
    -------
    la_rings, lo_rings : list of numpy.ndarray
        The same rings as contiguous 1-D ``float64`` arrays.

    Raises
    ------
    ValueError
        If the ring counts differ, a ring's lats and lons have different
        lengths, a ring has fewer than 3 vertices, or a coordinate is
        NaN/infinity.
    """
    if len(lats) != len(lons):
        raise ValueError("lats and lons must have the same number of rings")
    la_rings, lo_rings = [], []
    for la, lo in zip(lats, lons):
        la = np.asarray(la, dtype=np.float64).ravel()
        lo = np.asarray(lo, dtype=np.float64).ravel()
        if la.shape != lo.shape:
            raise ValueError("each ring's lats and lons must have the same length")
        if la.size < 3:
            raise ValueError("each ring needs at least 3 vertices")
        if not np.all(np.isfinite(la)) or not np.all(np.isfinite(lo)):
            raise ValueError("lats and lons must not contain NaN or infinity")
        la_rings.append(la)
        lo_rings.append(lo)
    return la_rings, lo_rings


def _single_coverage(lats, lons, order, normalize=True):
    """Coverage for one polygon ring.

    Parameters
    ----------
    lats, lons : array_like
        Vertex latitudes / longitudes in degrees for a single ring. A repeated
        closing vertex is stripped before the ring is handed to Rust.
    order : int
        HEALPix depth / tessellation order.
    normalize : bool, optional
        Ring-orientation handling. Identical in meaning to
        :func:`morton_coverage`'s ``normalize`` — see that function for the
        full ring-winding contract, including the hemisphere-plus caveat.

    Returns
    -------
    numpy.ndarray
        Sorted 1-D array of unique morton indices (``uint64``).

    Raises
    ------
    ValueError
        If lats and lons have different lengths, there are fewer than 3
        vertices, or a coordinate is NaN/infinity.
    """
    lats = np.asarray(lats, dtype=np.float64).ravel()
    lons = np.asarray(lons, dtype=np.float64).ravel()

    if lats.shape != lons.shape:
        raise ValueError("lats and lons must have the same length")
    if lats.size < 3:
        raise ValueError("Need at least 3 vertices for a polygon")
    if not np.all(np.isfinite(lats)) or not np.all(np.isfinite(lons)):
        raise ValueError("lats and lons must not contain NaN or infinity")

    # Strip duplicate closing vertex (first == last) if present
    if lats[0] == lats[-1] and lons[0] == lons[-1] and lats.size > 3:
        lats = lats[:-1].copy()
        lons = lons[:-1].copy()

    return np.asarray(_rustie.rust_polygon_coverage(lats, lons, order, normalize))


def morton_coverage(lats, lons, order=18, normalize=True):
    """Compute morton indices covering a polygon defined by lat/lon vertices.

    Given a polygon (as arrays of vertex latitudes and longitudes), returns the
    set of morton indices at the requested HEALPix order that completely cover
    the polygon interior.  The coverage is optimally compact — it includes all
    boundary cells plus every cell whose centre lies inside the polygon.

    For **multipart polygons and holes**, pass *lats* and *lons* as lists of
    rings.  All rings are covered by a single even-odd descent: a cell is
    covered iff its centre is inside an *odd* number of rings.  So disjoint
    outer rings are unioned (with no seam along shared interior borders), and a
    ring nested inside another carves a **hole** (a donut is ``[outer, hole]``).

    Parameters
    ----------
    lats : array_like or list of array_like
        Vertex latitudes in degrees.  For a single polygon, a 1-D array
        with at least 3 vertices.  For multipart polygons, a list of
        such arrays.
    lons : array_like or list of array_like
        Vertex longitudes in degrees.  Must match the structure of *lats*.
    order : int, optional
        HEALPix depth / tessellation order (1–29).  Default 18.
    normalize : bool, optional
        Auto-correct ring orientation at ingest.  Default ``True``: any
        *simple* ring whose interior decisively reads as the larger region is
        reversed so the smaller region is the interior — S2's normalization
        convention (issue #144, decision (A)) — so CW and CCW spellings of a
        polygon give the same cover, hemisphere-plus polygons included.  Pass
        ``False`` to **trust the supplied vertex order exactly** — the interior
        is taken as the region to the left of the directed edges with no
        reordering, so *every* ring, holes included, must be wound so its
        intended region lies to its left (see the **Ring winding** note); this
        is how a lone ring expresses a bigger-than-complement interior.

    Returns
    -------
    numpy.ndarray
        Sorted 1-D array of unique morton indices (dtype ``uint64``).

    Raises
    ------
    ValueError
        If fewer than 3 vertices, mismatched lengths, invalid order,
        or coordinates containing NaN/infinity.

    Warns
    -----
    UserWarning
        If the returned flat cover exceeds ~1M cells.  This is a **best-effort,
        post-hoc** signal — it fires only *after* the cover is materialized, so
        it does not prevent the blow-up: a flat cover's cell count grows as
        ``4**order`` along the boundary, so a large polygon and/or a high order
        can materialize billions of cells and exhaust memory *before* the
        warning is reached.  The hazard is the *cell count*, not the order
        alone.  Treat large flat covers as a footgun and use
        :func:`morton_coverage_moc` (optionally with its ``max_cells`` budget)
        for a compact mixed-order cover instead.

    Notes
    -----
    - Self-intersecting polygons produce undefined results.
    - Holes are supported via the multipart form: pass ``[outer, hole, ...]``
      (even-odd nesting carves the holes).
    - **Ring winding.**  The interior is the region to the **left** of each
      directed edge.  With ``normalize=True`` (default) the winding you author
      does not matter: any *simple* ring decisively enclosing the larger region
      is reversed at ingest, so the smaller side is taken either way (S2's
      convention; issue #144 decision (A)).  Author per RFC 7946 §3.1.6 (CCW
      exteriors, CW holes) or any other way — the RFC's CW-hole spelling is
      what ingest *delivers*, not what it requires.  With ``normalize=False``
      nothing is reordered, so you must wind **every** ring so its intended
      region lies to its left — for a carved hole that means
      counter-clockwise, like its exterior, *not* the RFC's clockwise.  A ring
      wound the other way selects its complement, which inverts the even-odd
      fill: a CW hole under ``normalize=False`` makes the "donut" larger than
      its own outer ring.  That same complement is the *only* way a lone ring
      covers a region larger than its complement.
    - The point-in-polygon test is a single robust spherical winding-number
      backend (issue #22): it is correct at any polygon size, including
      hemisphere-plus polygons, and degeneracy-free when an edge's great circle
      passes through a HEALPix cell centre (issue #11).

    Examples
    --------
    Single polygon:

    >>> import mortie
    >>> lats = [40.0, 50.0, 45.0]
    >>> lons = [-120.0, -120.0, -110.0]
    >>> cells = mortie.morton_coverage(lats, lons, order=6)

    Multipart polygon:

    >>> lats_parts = [[40.0, 50.0, 45.0], [10.0, 20.0, 15.0]]
    >>> lons_parts = [[-120.0, -120.0, -110.0], [-80.0, -80.0, -70.0]]
    >>> cells = mortie.morton_coverage(lats_parts, lons_parts, order=6)
    """
    if not 1 <= order <= 29:
        raise ValueError("Order must be between 1 and 29")

    if _is_multipart(lats):
        la, lo = _prep_rings(lats, lons)
        result = np.asarray(
            _rustie.rust_multipolygon_coverage(la, lo, order, normalize)
        )
    else:
        result = _single_coverage(lats, lons, order, normalize)

    _warn_large_flat(result.size, order)
    return result


def morton_coverage_moc(lats, lons, order=18, tolerance=None, max_cells=None,
                        normalize=True):
    """Compute polygon coverage as a compact Multi-Order Coverage (MOC) map.

    Unlike :func:`morton_coverage`, which returns a flat list of cells all at
    ``order``, this returns a *mixed-order* set: coarse cells for the interior
    and fine cells (down to ``order``) along the boundary.  Because a mortie
    morton index self-encodes its order, the result is still a 1-D ``uint64``
    array — typically far smaller than the flat cover.

    Optional **adaptive stop criteria** (mutually exclusive) trade boundary
    precision for fewer cells and faster runtime — the ``tolerance`` and
    ``max_cells`` parameters below.

    For **multipart / holes** (lists of rings), all rings are covered by one
    even-odd descent — disjoint parts union with no internal seam, and nested
    rings carve holes (a donut is ``[outer, hole]``).

    Parameters
    ----------
    lats, lons : array_like
        Vertex latitudes / longitudes in degrees (single polygon ring), or a
        list of such arrays for the multipart form.
    order : int, optional
        Finest HEALPix order (1–29).  Default 18.
    tolerance : float, optional
        Stop refining a boundary cell once its angular radius (in **degrees**)
        drops to this value, even if coarser than ``order``.  Approximate,
        coarser boundary.
    max_cells : int, optional
        Best-first budget: refine the largest boundary cells until about this
        many cells, giving an adaptive mixed-order boundary (fine where it
        wiggles, coarse where it is straight).  Soft target.
    normalize : bool, optional
        Auto-correct ring orientation at ingest (default True): any simple
        ring whose interior decisively reads as the larger region is reversed
        so the smaller region is covered, matching S2's normalization (issue
        #144, decision (A)).  Pass False to trust the supplied winding
        exactly — the escape hatch for covering a big-side interior with a
        lone ring.

    Returns
    -------
    numpy.ndarray
        Sorted 1-D array of mixed-order morton indices (``uint64``).

    Raises
    ------
    ValueError
        If ``order`` lies outside 1-29, both ``tolerance`` and ``max_cells``
        are given, the multipart form's lats and lons hold different numbers
        of rings, a ring's lats and lons have different lengths, a ring has
        fewer than 3 vertices, or a coordinate is NaN/infinity.

    See Also
    --------
    morton_coverage : flat single-order cover.
    compress_moc : merge 4-sibling groups in an existing morton set.
    """
    if not 1 <= order <= 29:
        raise ValueError("Order must be between 1 and 29")
    if tolerance is not None and max_cells is not None:
        raise ValueError("pass at most one of tolerance / max_cells")

    if _is_multipart(lats):
        la, lo = _prep_rings(lats, lons)
        tol_rad = None if tolerance is None else np.radians(float(tolerance))
        return np.asarray(
            _rustie.rust_multipolygon_coverage_moc(
                la, lo, order, tol_rad, max_cells, normalize
            )
        )

    lats = np.asarray(lats, dtype=np.float64).ravel()
    lons = np.asarray(lons, dtype=np.float64).ravel()
    if lats.shape != lons.shape:
        raise ValueError("lats and lons must have the same length")
    if lats.size < 3:
        raise ValueError("Need at least 3 vertices for a polygon")
    if not np.all(np.isfinite(lats)) or not np.all(np.isfinite(lons)):
        raise ValueError("lats and lons must not contain NaN or infinity")
    if lats[0] == lats[-1] and lons[0] == lons[-1] and lats.size > 3:
        lats = lats[:-1].copy()
        lons = lons[:-1].copy()

    tol_rad = None if tolerance is None else np.radians(float(tolerance))

    return np.asarray(
        _rustie.rust_polygon_coverage_moc(
            lats, lons, order, tol_rad, max_cells, normalize
        )
    )


def polygons_to_morton_mocs(lats, lons, offsets, order=18, tolerance=None,
                            max_cells=None, normalize=True):
    """Compute MOC coverage of many independent polygons in one call.

    The batch sibling of :func:`morton_coverage_moc` (issue #153): the ragged
    polygon set crosses the Python/Rust boundary **once**, the GIL is released
    for the whole batch, and Rust parallelizes across polygons — so the
    per-call fixed cost that dominates a Python loop over half a million
    footprints is paid once.  Identity-preserving: result ``i`` is exactly the
    cover of input polygon ``i`` (unlike the multipart form of
    :func:`morton_coverage_moc`, which unions its rings into one cover).  The
    plural *MOCs* in the name marks that many→many contract — one MOC per
    input polygon — against the many→one union of the multipart form.

    Input and output are ragged arrays in arrow list layout: polygon ``i`` is
    ``lats[offsets[i]:offsets[i+1]]`` / ``lons[offsets[i]:offsets[i+1]]``, and
    its MOC is ``values[out_offsets[i]:out_offsets[i+1]]`` in the result —
    byte-identical to ``morton_coverage_moc`` on that ring alone.

    Parameters
    ----------
    lats, lons : array_like
        Flat ``float64`` vertex latitudes / longitudes in degrees, all rings
        concatenated.
    offsets : array_like
        ``int64`` arrow list offsets: polygon ``i`` spans
        ``[offsets[i], offsets[i+1])``.  ``len(offsets) - 1`` polygons.  The
        offsets must **exactly cover** the vertex arrays — ``offsets[0] == 0``
        and ``offsets[-1] == len(lats) == len(lons)`` — so a sliced arrow
        array must be re-based before it gets here (:mod:`mortie.arrow` does
        that for you); anything else is an error naming the endpoint that
        failed.
    order : int, optional
        Finest HEALPix order (1-29), shared by every polygon.  Default 18.
    tolerance : float, optional
        Stop refining a boundary cell once its angular radius (in **degrees**)
        drops to this value — exactly :func:`morton_coverage_moc`'s
        ``tolerance``, applied as a **single shared setting** to every polygon
        in the batch.
    max_cells : int, optional
        Best-first cell budget per polygon — exactly
        :func:`morton_coverage_moc`'s ``max_cells``, shared by every polygon.
        A budget below some polygon's representable floor is raised for that
        polygon (soft target, as in the scalar path) and one summary warning
        is emitted.
    normalize : bool, optional
        Ring-orientation handling, identical in meaning to
        :func:`morton_coverage`'s ``normalize`` — see that function for the
        full ring-winding contract.  Default ``True``.

    Returns
    -------
    values : numpy.ndarray
        All polygons' morton MOC words concatenated (``uint64``).
    out_offsets : numpy.ndarray
        ``int64`` arrow list offsets into ``values``, length
        ``len(offsets)``; ``out_offsets[0]`` is always 0.

    Raises
    ------
    ValueError
        Fail-fast, naming the **lowest-index** offending polygon (e.g.
        ``polygon 4217: needs at least 3 vertices``): non-monotone or
        out-of-bounds offsets, a ring with fewer than 3 vertices, or a
        NaN/infinite coordinate.  Also for offsets that do not exactly cover
        the vertex arrays (``offsets[0] != 0``, or ``offsets[-1]`` short of or
        past ``len(lats)`` — the message names which endpoint failed),
        ``order`` outside 1-29, mismatched ``lats``/``lons`` lengths, or both
        ``tolerance`` and ``max_cells`` given.

    Warns
    -----
    UserWarning
        If ``max_cells`` is below the minimum needed to represent some
        polygon; the warning reports how many polygons were raised and names
        the lowest-index one.

    See Also
    --------
    morton_coverage_moc : the scalar (one polygon / one ring-set) form.

    Examples
    --------
    >>> import mortie, numpy as np
    >>> lats = np.array([40.0, 50.0, 45.0, 10.0, 20.0, 15.0])
    >>> lons = np.array([-120.0, -120.0, -110.0, -80.0, -80.0, -70.0])
    >>> values, off = mortie.polygons_to_morton_mocs(lats, lons, [0, 3, 6], order=6)
    >>> first = values[off[0]:off[1]]   # MOC of the first triangle
    """
    if tolerance is not None and max_cells is not None:
        raise ValueError("pass at most one of tolerance / max_cells")
    lats = np.ascontiguousarray(np.asarray(lats, dtype=np.float64).ravel())
    lons = np.ascontiguousarray(np.asarray(lons, dtype=np.float64).ravel())
    offsets = np.ascontiguousarray(np.asarray(offsets, dtype=np.int64).ravel())
    tol_rad = None if tolerance is None else np.radians(float(tolerance))
    values, out_offsets = _rustie.rust_polygons_coverage_mocs(
        lats, lons, offsets, order, tol_rad, max_cells, normalize
    )
    return np.asarray(values), np.asarray(out_offsets)


RingValidity = namedtuple("RingValidity", ["simple", "identity_consistent"])
"""Both ring-validity verdicts; see :func:`ring_validity`."""


def ring_validity(lats, lons):
    """Both validity verdicts for a ring -- whether any convention is in play.

    ``simple`` is :func:`ring_is_simple`'s verdict (no transversal
    self-intersection); ``identity_consistent`` is ``True`` when no vertex
    coordinate recurs at non-adjacent positions (a bit-exact pinch or a
    revisited vertex makes the symbolic tie-break depend on vertex
    *numbering*, so verdicts at the pinch can legitimately change under
    rotation).  Both ``True`` means every winding rule documented for
    coverage is exact for this ring, with no convention in play.

    Rings failing either check are still accepted by every coverage
    function -- the verdicts say which documented convention resolves the
    ambiguity, not that the cover is wrong.

    Parameters
    ----------
    lats, lons : array_like
        Vertex latitudes / longitudes in degrees (single ring, open or
        closed).

    Returns
    -------
    RingValidity
        Named tuple ``(simple, identity_consistent)`` of booleans.

    Raises
    ------
    ValueError
        If fewer than 3 vertices, mismatched lengths, or coordinates
        containing NaN/infinity.

    Examples
    --------
    >>> import mortie
    >>> mortie.ring_validity([0, 10, 10, 0], [0, 0, 10, 10])
    RingValidity(simple=True, identity_consistent=True)
    >>> mortie.ring_validity([0, 10, 0, 10], [0, 0, 10, 10]).simple
    False
    """
    lats = np.asarray(lats, dtype=np.float64).ravel()
    lons = np.asarray(lons, dtype=np.float64).ravel()
    if lats.shape != lons.shape:
        raise ValueError("lats and lons must have the same length")
    if lats.size < 3:
        raise ValueError("Need at least 3 vertices for a polygon")
    if not np.all(np.isfinite(lats)) or not np.all(np.isfinite(lons)):
        raise ValueError("lats and lons must not contain NaN or infinity")
    flags = np.asarray(_rustie.rust_ring_validity(lats, lons))
    return RingValidity(
        simple=not bool(flags[0]), identity_consistent=not bool(flags[1])
    )


def ring_is_simple(lats, lons):
    """Whether this polygon ring is free of transversal self-intersections.

    A ring whose edges cross has no single right-hand-rule interior, so
    mortie falls back to a documented convention (the positively wound
    region) and ingest normalization declines to trust the Gauss-Bonnet
    turning sign fully.  This check tells you *before* covering whether
    that convention is in play: ``True`` means no two non-adjacent edges
    cross transversally, and the winding rules read exactly on the
    geometry as given.

    It answers the crossing half of the question only.  A *bit-exact
    pinch* -- one coordinate repeated at non-adjacent positions -- is
    resolved by the symbolic identity convention instead, whose verdict
    can flip with where the vertex list starts (of 400 randomly pinched
    rings, 17 flip under a rotation of their vertices).  The full "is any
    convention in play?" question therefore needs the second verdict too:
    ``sphere::ring_set_identity_conflict``, returned alongside this one by
    the Rust ``sphere::ring_set_validity`` — or, from Python, the combined
    :func:`ring_validity`, which returns both verdicts.

    The check is the S2-architecture bucketed transversal-crossing test
    (issue #145): edges are indexed into adaptively refined HEALPix cells
    and only co-located, non-adjacent pairs are tested with the exact
    crossing predicate -- no sweep line, ``O(V log V)``-ish on realistic
    rings.  Consecutive duplicate vertices and a duplicated closing vertex
    are handled exactly as coverage ingest handles them.

    Parameters
    ----------
    lats, lons : array_like
        Vertex latitudes / longitudes in degrees (single ring, open or
        closed).

    Returns
    -------
    bool
        ``True`` when no two non-adjacent edges cross.

    Raises
    ------
    ValueError
        If lats and lons have different lengths, there are fewer than 3
        vertices, or a coordinate is NaN/infinity.

    Examples
    --------
    >>> import mortie
    >>> mortie.ring_is_simple([0, 10, 0, 10], [0, 0, 10, 10])   # a bowtie
    False
    >>> mortie.ring_is_simple([0, 10, 10, 0], [0, 0, 10, 10])   # untangled
    True
    """
    lats = np.asarray(lats, dtype=np.float64).ravel()
    lons = np.asarray(lons, dtype=np.float64).ravel()
    if lats.shape != lons.shape:
        raise ValueError("lats and lons must have the same length")
    if lats.size < 3:
        raise ValueError("Need at least 3 vertices for a polygon")
    if not np.all(np.isfinite(lats)) or not np.all(np.isfinite(lons)):
        raise ValueError("lats and lons must not contain NaN or infinity")
    return np.asarray(_rustie.rust_ring_is_simple(lats, lons)).size == 0


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

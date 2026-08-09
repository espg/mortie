"""Polygon-to-morton coverage.

Compute the set of morton indices at a given order that completely cover
a polygon defined by lat/lon vertices.  Supports single and multipart
polygons.

Set algebra over the covers this module produces — union / intersection /
difference, the canonical compaction, the densify back to a flat single order —
lives in :mod:`mortie.moc`, split out by domain (issue #156).  It is all
computed in Rust; there is no Python-level MOC set algebra in either module.
The plural batch twin of the coverers here,
:func:`~mortie.batch.polygons_to_morton_mocs`, lives in :mod:`mortie.batch`,
consolidated by arity with every other bulk operator the pyarrow skin does not
own (issue #170).  All three
modules' names stay flat on the package (``mortie.morton_coverage``,
``mortie.moc_and``, ``mortie.polygons_to_morton_mocs``).
"""

import warnings
from collections import namedtuple

import numpy as np

from . import _rustie

# A flat cover's cell count scales as ~4**order along the boundary, so a flat
# `morton_coverage` at high order can grow to billions of cells and exhaust
# memory.  Above this many cells we warn and point at the compact MOC form
# (issue #60).  The warning is post-hoc (it fires after the cover is built), so
# it flags the hazard but cannot prevent an OOM at very high order — the MOC
# form is the real high-order path.  The true ceiling is this cell count, not
# the order itself.
#
# One number, two expressions of it (espg's ruling on issue #108): the same line
# is the *pre-emptive refusal* in `mortie.moc.moc_to_order` /
# `mortie.batch.mocs_to_orders`, where the memory has not been spent yet and can
# still be declined.  There is
# deliberately no second, higher ceiling — and `max_cells` stays a caller
# parameter with a documented `max_cells=None` escape, so the default takes no
# functionality from anyone.
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
    mortie.batch.polygons_to_morton_mocs : the batch form (many polygons in
        one call).
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

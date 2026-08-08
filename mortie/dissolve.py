"""Spherical outline of a morton cover: edge-cancellation dissolve.

The whole ``dissolve=True`` emit path, from a cover's cell boundaries to the
exterior/hole rings :func:`mortie.to_geometry` hands to the backend.
:func:`_dissolved_polygons` is the runtime entry (the Rust
``_rustie.rust_dissolve`` fast path plus hole nesting);
:func:`_dissolved_rings_py` is the exact-verified pure-Python reference oracle
the tests check it against, and the rest are that oracle's parts -- boundary
extraction and edge cancellation, ring chaining, the GeoJSON antimeridian
cut/stitch, and the spherical/planar area and point-in-ring primitives the
winding decisions rest on.

Split out of :mod:`mortie.geometry` (issue #159) so the Python surface mirrors
the Rust tree's own decomposition -- this module is the Python side of
``src_rust/src/dissolve.rs``.  :mod:`mortie.geometry` keeps the
coverage<->geometry API and the WKB plumbing (the codec quartet and the backend
gate are :mod:`mortie.codec`'s, since issue #159 phase 4), and imports
:func:`_dissolved_polygons` from here; nothing here imports back.  Every name is
private: the public surface is unchanged, and reaches this module only through
``mortie.to_geometry`` / ``mortie.to_wkb`` / ``mortie.to_wkt``.
"""

import math

import numpy as np

# Snap scale for vertex identity in the dissolve edge-cancellation (rounding
# unit-vector components to 1e-10 makes a shared HEALPix corner — which both
# adjacent cells compute identically — a single integer-keyed vertex, so their
# shared edge cancels exactly without a floating tolerance search).
_DISSOLVE_SNAP = 1e10


# ── emit: dissolved-boundary outline (phase 4) ─────────────────────────────
#
# The dissolved outline is built natively (no backend spatial predicate): every
# cell contributes its boundary as a loop of directed edges; interior edges that
# two adjacent cells share are traversed in opposite directions and cancel, and
# the surviving edges chain into the outline rings.  Correctness is mortie's own
# job throughout — the backend is only asked to *construct* the final Polygon.


def _xyz_to_latlon(vecs):
    """Convert unit vectors to ``(lat, lon)`` degree arrays.

    Parameters
    ----------
    vecs : numpy.ndarray
        An ``(M, 3)`` array of unit vectors.

    Returns
    -------
    tuple of numpy.ndarray
        The ``(lat, lon)`` degree arrays, with lon in (-180, 180].
    """
    z = np.clip(vecs[:, 2], -1.0, 1.0)
    lat = np.degrees(np.arcsin(z))
    lon = np.degrees(np.arctan2(vecs[:, 1], vecs[:, 0]))
    return lat, lon


def _spherical_signed_area(ring_xyz):
    """Compute the signed area (steradians) of a spherical polygon.

    Uses the van Oosterom–Strackee signed-solid-angle sum over a fan from
    vertex 0.

    Parameters
    ----------
    ring_xyz : numpy.ndarray
        An ``(M, 3)`` array of unit vectors (open ring — first vertex not
        repeated).

    Returns
    -------
    float
        The signed area in steradians.  Positive = the region lies to the left
        of the directed boundary (an exterior ring); negative = the boundary
        winds the other way (a hole).  Zero for a degenerate ring of fewer
        than 3 vertices.
    """
    v = ring_xyz
    if v.shape[0] < 3:
        return 0.0
    a = v[0]
    b = v[1:-1]
    c = v[2:]
    num = np.einsum("j,ij->i", a, np.cross(b, c))
    den = 1.0 + b @ a + np.einsum("ij,ij->i", b, c) + c @ a
    return float(np.sum(2.0 * np.arctan2(num, den)))


def _boundary_rings_xyz(morton, step):
    """Dissolve a cover to its boundary rings by edge cancellation.

    A mixed-order MOC is densified to its finest order first so every cell
    carries unit-length edges that cancel against their neighbours.  Shared
    sub-edge points coincide between neighbours and cancel too.

    Parameters
    ----------
    morton : array_like of uint64
        A morton cover (flat or mixed-order MOC).
    step : int
        Samples ``step`` points per cell edge (``step>1`` traces the curved
        HEALPix boundary).

    Returns
    -------
    list of numpy.ndarray
        One ``(M, 3)`` array of unit vectors per boundary ring (open — first
        vertex not repeated); empty for an empty cover.
    """
    from . import _healpix as hp
    from .moc import moc_to_order
    from .orders import _rust_mort2nested

    morton = np.atleast_1d(np.asarray(morton, dtype=np.uint64))
    if morton.size == 0:
        return []

    nested, depths = _rust_mort2nested(np.ascontiguousarray(morton))
    udepths = np.unique(depths)
    if udepths.size > 1:
        morton = np.asarray(moc_to_order(morton, int(udepths.max())), dtype=np.uint64)
        nested, depths = _rust_mort2nested(np.ascontiguousarray(morton))
    order = int(depths[0])
    nest = np.ascontiguousarray(nested.astype(np.int64))

    bnd = hp.boundaries(order, nest, step=step)
    if bnd.ndim == 2:
        bnd = bnd[np.newaxis, ...]
    pts = np.transpose(bnd, (0, 2, 1))  # (N, K, 3), K = 4*step in boundary order
    n_cells, k = pts.shape[0], pts.shape[1]
    flat = pts.reshape(-1, 3)

    # Integer-snap every boundary point to a vertex id; a shared corner/sub-edge
    # point collapses to one id, so adjacent cells reference the same vertex.
    snapped = np.round(flat * _DISSOLVE_SNAP).astype(np.int64)
    _, first_idx, inv = np.unique(
        snapped, axis=0, return_index=True, return_inverse=True
    )
    id_xyz = flat[first_idx]  # representative unit vector per vertex id
    inv = inv.reshape(n_cells, k)

    # Directed edges (vertex id → vertex id) around every cell boundary.
    starts = inv.ravel()
    ends = np.roll(inv, -1, axis=1).ravel()
    keep = starts != ends  # drop any degenerate zero-length edge
    edges = list(zip(starts[keep].tolist(), ends[keep].tolist()))

    # An interior edge appears as (a, b) in one cell and (b, a) in its neighbour;
    # the surviving boundary is the net direction at each undirected edge.
    from collections import Counter

    counts = Counter(edges)
    survivors = []
    for (a, b), c in counts.items():
        net = c - counts.get((b, a), 0)
        survivors.extend([(a, b)] * net)
    return _chain_rings(survivors, id_xyz)


def _tangent_azimuth(p, q):
    """Compute the azimuth from unit vector *p* toward unit vector *q*.

    Measured in p's tangent plane (north-referenced).  Used to order edges
    around a vertex.

    Parameters
    ----------
    p : numpy.ndarray
        The unit vector at which the azimuth is measured.
    q : numpy.ndarray
        The unit vector the azimuth points toward.

    Returns
    -------
    float
        The azimuth in radians; ``0.0`` when *q* is (numerically) parallel
        to *p*.
    """
    d = q - np.dot(q, p) * p
    nd = np.linalg.norm(d)
    if nd < 1e-15:
        return 0.0
    d = d / nd
    east = np.cross([0.0, 0.0, 1.0], p)
    ne = np.linalg.norm(east)
    east = np.array([1.0, 0.0, 0.0]) if ne < 1e-9 else east / ne
    north = np.cross(p, east)
    return math.atan2(float(np.dot(d, east)), float(np.dot(d, north)))


def _chain_rings(survivors, id_xyz):
    """Chain surviving directed boundary edges into closed rings.

    At a non-manifold vertex (out-degree > 1 — e.g. two cells touching only at a
    corner) the next edge is chosen by angular order: the surviving edge whose
    departure azimuth is the smallest turn anticlockwise from the reversed
    arrival direction.  This right-hand-rule traversal yields *simple* rings
    (the cells' boundaries stay separate rather than crossing into a bowtie),
    independent of the cover's global winding.

    Parameters
    ----------
    survivors : list of tuple
        The surviving directed edges as ``(start_id, end_id)`` vertex-id pairs.
    id_xyz : numpy.ndarray
        An ``(N, 3)`` array of the representative unit vector per vertex id.

    Returns
    -------
    list of numpy.ndarray
        One ``(M, 3)`` array of unit vectors per closed ring (open — first
        vertex not repeated).
    """
    from collections import defaultdict

    az = {e: _tangent_azimuth(id_xyz[e[0]], id_xyz[e[1]]) for e in survivors}
    records = [[a, b, True] for a, b in survivors]
    by_start = defaultdict(list)
    for rec in records:
        by_start[rec[0]].append(rec)

    rings = []
    for seed in records:
        if not seed[2]:
            continue
        seed_start = seed[0]
        cur = seed
        chain = []
        while cur is not None and cur[2]:
            cur[2] = False
            chain.append(cur[0])
            v = cur[1]
            if v == seed_start:
                break  # returned to the start vertex — ring closed
            cand = [r for r in by_start[v] if r[2]]
            if not cand:
                break
            if len(cand) == 1:
                cur = cand[0]
            else:
                # Smallest turn anticlockwise from the reversed arrival keeps the
                # walk on the same face (no crossing) at a non-manifold vertex.
                back = _tangent_azimuth(id_xyz[v], id_xyz[cur[0]])
                cur = min(cand, key=lambda r: (az[(r[0], r[1])] - back) % (2 * math.pi))
        rings.append(id_xyz[np.asarray(chain)])
    return rings


def _antimeridian_winding(lon):
    """Measure a ring's net longitude winding and antimeridian crossings.

    Parameters
    ----------
    lon : numpy.ndarray
        The longitudes (degrees) of a closed ring, first vertex not repeated.

    Returns
    -------
    tuple
        ``(net, crossings)`` — the net signed longitude winding in degrees and
        the antimeridian-crossing count.  Net ≈ ±360 ⟺ the ring encircles a
        pole.
    """
    deltas = np.diff(np.concatenate([lon, lon[:1]]))
    crossings = int(np.sum(np.abs(deltas) > 180.0))
    net = float(np.sum((deltas + 180.0) % 360.0 - 180.0))
    return net, crossings


def _cut_at_antimeridian(coords):
    """Cut an open lon/lat ring at every ±180° crossing.

    This is the GeoJSON-convention building block — :func:`_stitch_segments`
    reconnects the segments along the meridian (and, for a pole-enclosing
    region, through a ±90° pole vertex).

    Parameters
    ----------
    coords : list of tuple
        The open ring as ``(lon, lat)`` degree pairs (first vertex not
        repeated).

    Returns
    -------
    tuple
        ``(whole, segments)``: a ring that never crosses gives
        ``(closed_ring, [])`` (the caller keeps it whole); a crossing ring
        gives ``(None, [seg, ...])`` where each segment is an open polyline
        whose two free ends sit on ±180° (latitude linearly interpolated at
        the cut).
    """
    n = len(coords)
    segments = []
    cur = []
    for i in range(n):
        lo0, la0 = coords[i]
        lo1, la1 = coords[(i + 1) % n]
        cur.append((lo0, la0))
        if abs(lo1 - lo0) > 180.0:
            lo1u = lo1 - 360.0 if lo1 > lo0 else lo1 + 360.0
            boundary = 180.0 if lo1u > lo0 else -180.0
            frac = (boundary - lo0) / (lo1u - lo0)
            la_x = la0 + frac * (la1 - la0)
            cur.append((boundary, la_x))
            segments.append(cur)
            cur = [(-boundary, la_x)]
    if not segments:
        return coords + [coords[0]], []
    segments[0] = cur + segments[0]  # the wrap-around segment closes the first
    return None, segments


def _stitch_segments(segments, pole):
    """Reconnect antimeridian-cut *segments* into closed lon/lat rings.

    Every segment runs from a free end on ±180° to another on ±180°.  Walking
    from a segment's end, the next segment is the one whose **start** sits on the
    **same ±180° side** at the next latitude inward — on +180° the next start
    above, on -180° the next start below — so the connector edge runs straight
    along the meridian without crossing the boundary.  When no same-side start
    lies in that direction the region wraps a pole: insert the ``pole`` (±90°)
    vertex, cross to the other side at that pole, and resume.

    This is the GeoJSON / ``antimeridian``-package convention: a single split
    ``MultiPolygon`` with explicit ±90° pole vertices stitched down ±180°.  It
    generalises the old two-crossing split (each segment closing on its own
    side) to any even crossing count, to pole-enclosing caps, and to
    antimeridian-crossing holes.

    Parameters
    ----------
    segments : list of list of tuple
        The cut segments from :func:`_cut_at_antimeridian`, each an open
        polyline of ``(lon, lat)`` degree pairs.
    pole : float
        The pole the **filled** region encloses (``+90``/``-90``), or ``0``
        when none is enclosed.  It is only ever reached when the segments are
        genuinely unbalanced, so a non-pole cover never touches it.

    Returns
    -------
    list of list of tuple
        The reconnected closed rings of ``(lon, lat)`` degree pairs.

    Raises
    ------
    RuntimeError
        If the stitch fails to converge (a guard), or if the segments are
        unbalanced with no pole enclosed.
    """
    segs = [list(s) for s in segments]
    used = [False] * len(segs)
    rings = []
    for seed in range(len(segs)):
        if used[seed]:
            continue
        ring = []
        idx = seed
        guard = 0
        while idx is not None and not used[idx]:
            guard += 1
            if guard > 8 * len(segs) + 16:  # pragma: no cover - convergence guard
                raise RuntimeError("antimeridian stitch did not converge")
            used[idx] = True
            ring.extend(segs[idx])
            idx = _next_segment(segs, used, ring, pole, seed)
        ring.append(ring[0])
        rings.append(ring)
    return rings


def _next_segment(segs, used, ring, pole, seed):
    """Append meridian/pole connectors and pick the next segment.

    Appends the connectors from the current ring end and returns the next
    segment index.  See :func:`_stitch_segments`.

    Closing back to the *seed* is the right stop: walking always advances toward
    the next start in one meridian direction, so the seed (the directional
    extremum on its side) is reached only when the loop has consumed every
    segment of this ring — it cannot be stepped past.  Same-side starts are
    matched within a 1e-9° latitude tolerance, which is far below HEALPix corner
    spacing at any order, so distinct crossing points never alias.

    Parameters
    ----------
    segs : list of list of tuple
        All antimeridian-cut segments.
    used : list of bool
        Per-segment consumed flags, updated by the caller.
    ring : list of tuple
        The ring being built; connector vertices are appended in place.
    pole : float
        The pole the filled region encloses (``+90``/``-90``), or ``0``.
    seed : int
        Index of the segment this ring started from.

    Returns
    -------
    int or None
        The next segment index, or ``None`` to close the ring.

    Raises
    ------
    RuntimeError
        If the segments are unbalanced but no pole is enclosed.
    """
    side, end_lat = ring[-1]
    cands = [(segs[i][0][1], i) for i in range(len(segs))
             if abs(segs[i][0][0] - side) < 1e-9 and (not used[i] or i == seed)]
    # +180° connects upward to the next start above; -180° downward to the next
    # start below — the direction that keeps the connector inside the region.
    if side > 0:
        pick = min(((la, i) for la, i in cands if la >= end_lat - 1e-9),
                   default=None)
    else:
        pick = max(((la, i) for la, i in cands if la <= end_lat + 1e-9),
                   default=None)
    if pick is not None:
        la, i = pick
        ring.append((side, la))
        return None if (i == seed and used[seed]) else i

    # No same-side start in that direction: the region wraps ``pole``.  Run the
    # seam to the pole, cross to the other side, and resume from the pole.
    if pole == 0:  # pragma: no cover - guarded by the caller's pole detection
        raise RuntimeError("unbalanced antimeridian segments but no pole enclosed")
    other = -side
    ring.append((side, pole))
    ring.append((other, pole))
    ocands = [(segs[i][0][1], i) for i in range(len(segs))
              if abs(segs[i][0][0] - other) < 1e-9 and (not used[i] or i == seed)]
    if not ocands:  # pragma: no cover - a closed boundary always has a partner
        return None
    if other > 0:
        la, i = min(ocands) if pole < 0 else max(ocands)
    else:
        la, i = max(ocands) if pole < 0 else min(ocands)
    ring.append((other, la))
    return None if (i == seed and used[seed]) else i


def _point_in_ring(x, y, ring):
    """Test a point against a ring by even-odd ray casting.

    Parameters
    ----------
    x, y : float
        The point's coordinates, in the same units as ``ring``.
    ring : list of tuple
        A closed list of ``(x, y)`` vertices.

    Returns
    -------
    bool
        ``True`` if the point is inside the ring.
    """
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > y) != (yj > y):
            x_cross = xi + (y - yi) / (yj - yi) * (xj - xi)
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def _planar_signed_area(ring):
    """Compute the shoelace signed area of a ring (for size ordering).

    Parameters
    ----------
    ring : list of tuple
        A closed list of ``(x, y)`` vertices.

    Returns
    -------
    float
        The planar signed area.
    """
    a = np.asarray(ring, dtype=np.float64)
    x, y = a[:, 0], a[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _ring_signed_area_lonlat(ring):
    """Compute the spherical signed area of a closed lon/lat-degree ring.

    Used to classify a stitched ring whose seam runs through a pole, where the
    planar shoelace sign is unreliable but the spherical area stays exact.

    Parameters
    ----------
    ring : list of tuple
        A closed list of ``(lon, lat)`` degree pairs (last vertex repeats the
        first).

    Returns
    -------
    float
        The signed area in steradians.  Positive ⟺ the ring winds CCW (an
        exterior); negative ⟺ a hole.
    """
    a = np.asarray(ring[:-1], dtype=np.float64)
    rlat = np.radians(a[:, 1])
    rlon = np.radians(a[:, 0])
    v = np.column_stack(
        [np.cos(rlat) * np.cos(rlon), np.cos(rlat) * np.sin(rlon), np.sin(rlat)]
    )
    return _spherical_signed_area(v)


def _reject_hemisphere_cover(morton):
    """Mirror of the Rust hemisphere guard (``src_rust/src/dissolve.rs``, issue
    #108): exterior/hole classification keys off the sign of the mod-4π
    spherical signed area, which is ambiguous once the cover nears 2π — fail
    loud on the exact covered area (Σ π/(3·4^depth), cells are equal-area)
    instead of silently swapping shells and holes.  Assumes disjoint,
    non-duplicated cells (the dissolve precondition anyway — duplicate words
    would break edge cancellation); duplicates double-count.  Returns the
    exact covered area (steradians) for the wrap cross-check downstream.
    """
    from .orders import _rust_mort2nested

    morton = np.atleast_1d(np.asarray(morton, dtype=np.uint64))
    if morton.size == 0:
        return 0.0
    _, depths = _rust_mort2nested(np.ascontiguousarray(morton))
    area = float(np.sum(np.pi / (3.0 * 4.0 ** depths.astype(np.float64))))
    if area > 2.0 * np.pi * 0.98:
        raise ValueError(
            f"dissolved cover spans {area:.6f} sr — within 2% of a hemisphere "
            "(2π sr) or beyond — so its exterior/hole winding is ambiguous; "
            "split the cover into sub-hemisphere parts or pass dissolve=False "
            "for per-cell polygons"
        )
    return area


def _dissolved_rings_py(morton, step):
    """Dissolve a cover to lon/lat rings in pure Python (the reference engine).

    This is the exact-verified reference engine kept as the test oracle for the
    Rust fast path (:func:`_dissolved_polygons` calls ``_rustie.rust_dissolve``
    at runtime — §7's Rust-only contract).  Rings that cross the ±180° meridian
    are cut and reconnected by the GeoJSON-convention splitter
    (:func:`_cut_at_antimeridian` / :func:`_stitch_segments`), which inserts
    explicit ±90° pole vertices for a pole-enclosing region.

    Parameters
    ----------
    morton : array_like of uint64
        A morton cover (flat or mixed-order MOC).
    step : int
        Boundary points per cell edge (1 = 4 corners).

    Returns
    -------
    tuple of list
        ``(ext_pieces, holes)`` — each entry a closed list of ``(lon, lat)``
        degree pairs; ``([], [])`` for an empty cover.
    """
    cover_area = _reject_hemisphere_cover(morton)
    rings_xyz = _boundary_rings_xyz(morton, step)
    if not rings_xyz:
        return [], []

    # Normalise global winding: the cover's net signed area (exteriors minus
    # holes) is the covered area, always positive.  HEALPix orders boundary
    # points one way for step==1 and the other for step>1, so key the
    # exterior/hole sign off this invariant rather than a fixed convention.
    # The fan formula wraps mod 4π when a single ring encloses more than a
    # hemisphere (possible even for a small cover, e.g. an equatorial band),
    # which would flip the sign here; an honest |Σ| matches the exact covered
    # area to within chord discretization (≲0.1 sr at step==1) while any wrap
    # is off by ~4π, so a π tolerance separates them cleanly (mirrors the Rust
    # cross-check in `src_rust/src/dissolve.rs::classify_and_split`).
    areas = [_spherical_signed_area(r) for r in rings_xyz]
    total = sum(areas)
    if abs(abs(total) - cover_area) > np.pi:
        raise ValueError(
            "dissolved cover has a boundary ring enclosing more than a "
            f"hemisphere (|Σ ring areas| = {abs(total):.6f} sr vs covered area "
            f"{cover_area:.6f} sr), so its exterior/hole winding cannot be "
            "classified; split the cover into sub-hemisphere parts or pass "
            "dissolve=False for per-cell polygons"
        )
    if total < 0.0:
        rings_xyz = [r[::-1] for r in rings_xyz]
        areas = [-a for a in areas]

    # Rings that never cross the antimeridian are emitted whole; crossing rings
    # contribute open segments that are stitched together below.  The pole the
    # filled region encloses is set by the cover's *total* net longitude winding
    # (an exterior and a hole that both wrap the pole cancel to net 0 — a band
    # that does not enclose the pole — so per-ring winding would be wrong here).
    ext_pieces = []
    holes = []
    segments = []
    total_net = 0.0
    for ring, area in zip(rings_xyz, areas):
        lat, lon = _xyz_to_latlon(ring)
        ll = list(zip(lon.tolist(), lat.tolist()))
        net, _ = _antimeridian_winding(lon)
        total_net += net
        whole, segs = _cut_at_antimeridian(ll)
        if whole is not None:
            (holes if area < 0.0 else ext_pieces).append(whole)
        else:
            segments.extend(segs)

    if segments:
        pole = 0.0
        if abs(total_net) > 180.0:  # net ≈ ±360° ⟺ the filled region wraps a pole
            pole = 90.0 if total_net > 0.0 else -90.0
        for piece in _stitch_segments(segments, pole):
            # Classify by spherical signed area — a pole-spanning ring's planar
            # shoelace sign is unreliable, but its spherical area is exact.  The
            # sign is meaningful because the global winding was normalised above
            # (exteriors CCW → positive); a stitched piece always encloses a
            # finite covered/uncovered region, so its area is never exactly zero.
            (ext_pieces if _ring_signed_area_lonlat(piece) >= 0.0 else holes).append(
                piece
            )
    return ext_pieces, holes


def _nest_and_build(mod, ext_pieces, holes):
    """Nest each hole into the smallest containing exterior and build Polygons.

    Parameters
    ----------
    mod : module
        The active geometry backend module (shapely).
    ext_pieces : list of list of tuple
        The exterior rings, each a closed list of ``(lon, lat)`` degree pairs.
    holes : list of list of tuple
        The hole rings, in the same form.

    Returns
    -------
    list
        One backend Polygon per exterior, carrying its nested holes.

    Raises
    ------
    NotImplementedError
        If a hole nests into no exterior (an unsupported self-touching
        outline); pass ``dissolve=False``.
    """
    hole_groups = [[] for _ in ext_pieces]
    ext_areas = [abs(_planar_signed_area(p)) for p in ext_pieces]
    for hole in holes:
        # A hole vertex lies strictly inside its surrounding exterior, so test a
        # vertex (a guaranteed-interior point) rather than the centroid, which a
        # concave or split ring can push outside the region.
        hx, hy = hole[0]
        best = None
        for idx, piece in enumerate(ext_pieces):
            if _point_in_ring(hx, hy, piece) and (
                best is None or ext_areas[idx] < ext_areas[best]
            ):
                best = idx
        if best is None:
            raise NotImplementedError(
                "dissolved emit could not nest a hole into any exterior (an "
                "unsupported self-touching outline); pass dissolve=False"
            )
        hole_groups[best].append(hole)
    return [
        mod.Polygon(ext_pieces[i], hole_groups[i]) for i in range(len(ext_pieces))
    ]


def _dissolved_polygons(mod, morton, step):
    """Build the dissolved outline of *morton* as a list of backend Polygons.

    The exterior/hole rings (edge-cancellation dissolve plus the GeoJSON
    pole/antimeridian split) are computed by the Rust fast path
    (``_rustie.rust_dissolve``); this nests holes into the exterior that
    contains them and constructs the backend Polygons.  Handles pole caps (the
    project's polar data), exteriors crossing the antimeridian any even number
    of times, and antimeridian-crossing holes.  The pure-Python
    :func:`_dissolved_rings_py` is the exact-verified reference oracle for this
    path in the tests.

    Parameters
    ----------
    mod : module
        The active geometry backend module (shapely).
    morton : array_like of uint64
        A morton cover (flat or mixed-order MOC).
    step : int
        Boundary points per cell edge (1 = 4 corners).

    Returns
    -------
    list
        The dissolved outline as backend Polygons; empty for an empty cover.
    """
    from . import _rustie

    morton = np.atleast_1d(np.asarray(morton, dtype=np.uint64))
    if morton.size == 0:
        return []
    shells, holes = _rustie.rust_dissolve(np.ascontiguousarray(morton), int(step))
    ext_pieces = [[tuple(v) for v in ring] for ring in shells]
    hole_rings = [[tuple(v) for v in ring] for ring in holes]
    if not ext_pieces:
        return []
    return _nest_and_build(mod, ext_pieces, hole_rings)

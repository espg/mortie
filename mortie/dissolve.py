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
        for cycle in _split_at_revisits(chain):
            rings.append(id_xyz[np.asarray(cycle)])
    return rings


def _split_at_revisits(chain):
    """Split a closed vertex-id walk into minimal simple cycles.

    The angular chaining traces boundaries of the *covered* faces; where the
    covered region is pinched at a vertex (two uncovered cells touching only
    diagonally, so the covered face's own boundary passes the pinch twice)
    the traced walk legitimately revisits that vertex — a correct even-odd
    boundary, but not an OGC-simple ring.  Splitting at each revisit yields
    the equivalent decomposition into simple rings touching at a point.  A
    split at a *pole* vertex re-pairs that pole's passages onto their
    adjacent wedges; :func:`_ring_to_planar`'s empty-bracket span rule
    renders each re-paired passage over exactly its own wedge, keeping the
    planar shoelace budget consistent.  Each cycle is rotated to start at
    its minimal vertex id.  Mirrors ``split_at_revisits`` in
    ``src_rust/src/dissolve.rs`` (issue #147).

    Parameters
    ----------
    chain : list of int
        The closed walk as vertex ids (first vertex not repeated).

    Returns
    -------
    list of list of int
        The simple cycles (each open, first vertex not repeated).
    """
    out = []
    stack = []
    pos = {}

    def _rotated(cycle):
        k = cycle.index(min(cycle))
        return cycle[k:] + cycle[:k]

    for vid in chain:
        if vid in pos:
            p = pos[vid]
            cycle = stack[p:]
            del stack[p:]
            for c in cycle:
                pos.pop(c, None)
            out.append(_rotated(cycle))
        pos[vid] = len(stack)
        stack.append(vid)
    if stack:
        out.append(_rotated(stack))
    return out


def _normalized_lonlat(ring_xyz):
    """Per-vertex ``(lat, lon)`` with seam vertices' longitude signs fixed.

    A vertex exactly on the ±180° meridian gets an arbitrary sign from
    ``atan2`` (its ``y`` is a rounding residual), and a whole boundary edge
    can lie on the seam (base-cell meridians at lon 180) — mixed signs there
    fabricate spurious ±360° cuts mid-edge.  With the covered region on the
    ring's left, a seam-lying edge walked northward has the covered side
    west, so it belongs on the map's east edge (+180°); southward, on -180°.
    A vertex with a seam (or pole) neighbour takes that edge's direction; an
    isolated seam touch takes its off-seam neighbours' side.  Mirrors
    ``normalized_lonlat`` in ``src_rust/src/dissolve.rs`` (issue #147).

    Parameters
    ----------
    ring_xyz : numpy.ndarray
        An ``(M, 3)`` array of unit vectors (open ring).

    Returns
    -------
    tuple of numpy.ndarray
        The ``(lat, lon)`` degree arrays with normalized seam longitudes.
    """
    lat, lon = _xyz_to_latlon(ring_xyz)
    lon = lon.copy()
    n = len(ring_xyz)

    def is_pole(j):
        return abs(ring_xyz[j][2]) > 1.0 - 1e-9

    def on_seam(j):
        return not is_pole(j) and abs(abs(orig[j]) - 180.0) < 1e-9

    def seam_lat(j):
        # effective seam latitude when the edge to neighbour j runs along
        # the seam (a pole neighbour's meridian edge is the seam meridian).
        if is_pole(j):
            return 90.0 if ring_xyz[j][2] > 0 else -90.0
        if on_seam(j):
            return float(lat[j])
        return None

    orig = lon.copy()
    for i in range(n):
        if not on_seam(i):
            continue
        prev, nxt = (i - 1) % n, (i + 1) % n
        la = seam_lat(nxt)
        if la is not None:
            side = 180.0 if la > lat[i] else -180.0
        else:
            la = seam_lat(prev)
            if la is not None:
                side = 180.0 if lat[i] > la else -180.0
            else:
                side = 180.0 if orig[prev] >= 0.0 else -180.0
        lon[i] = side
    return lat, lon


def _pole_run(out, start, stop, pole):
    """Append a monotone lat-±90 run from *start* to *stop* (exclusive of
    *start*), subdivided so no planar step reaches 180° — a covered polar
    wedge can legitimately span more than 180° of longitude, which
    :func:`_cut_at_antimeridian` would misread as a seam wrap if emitted as
    one step.  Mirrors ``pole_run`` in ``src_rust/src/dissolve.rs``.
    """
    direction = 1.0 if stop > start else -1.0
    cur = start
    while abs(stop - cur) > 150.0:
        cur += direction * 120.0
        out.append((cur, pole))
    out.append((stop, pole))


def _pole_meridians(rings_xyz):
    """Pole-incident boundary meridians over a ring set.

    For every pole vertex, the arriving and departing neighbours'
    (normalized) longitudes.  Feeds :func:`_ring_to_planar`'s empty-bracket
    cap rule.  Mirrors the collection in ``classify_and_split``
    (``src_rust/src/dissolve.rs``).

    Parameters
    ----------
    rings_xyz : list of numpy.ndarray
        The oriented boundary rings.

    Returns
    -------
    tuple of list
        ``(north, south)`` meridian longitude lists (degrees).
    """
    north, south = [], []
    for ring in rings_xyz:
        n = len(ring)
        _, lon = _normalized_lonlat(ring)
        for i in range(n):
            if abs(ring[i][2]) > 1.0 - 1e-9:
                dst = north if ring[i][2] > 0 else south
                dst.append(float(lon[(i - 1) % n]))
                dst.append(float(lon[(i + 1) % n]))
    return north, south


def _ring_to_planar(ring_xyz, north_m, south_m):
    """Convert an oriented spherical ring to planar ``(lon, lat)`` vertices.

    Expands any pole vertex into an explicit lat-±90 traverse: a vertex at
    the pole has no longitude (``atan2(0, 0)``); its planar image is a
    lat-±90 *edge* — a cap over the wedge of longitudes this passage
    encloses at the pole.  Which wedge is the **empty-bracket rule**: of the
    two lon-intervals between the arriving and departing meridians, the
    enclosed one contains no *other* pole-incident boundary meridian; at a
    degree-2 pole both are empty and the covered side wins (westward of the
    arrival at +90, eastward at -90).  A cap through the seam inserts an
    exact ±180 pair that :func:`_cut_at_antimeridian` splits without
    interpolation.  Mirrors ``ring_to_planar`` in
    ``src_rust/src/dissolve.rs`` (issue #147).

    Parameters
    ----------
    ring_xyz : numpy.ndarray
        An ``(M, 3)`` array of unit vectors (open ring, covered region on
        the left).
    north_m, south_m : list of float
        The pole-incident meridians from :func:`_pole_meridians`.

    Returns
    -------
    list of tuple
        The open planar ring as ``(lon, lat)`` degree pairs.
    """
    lat, lon = _normalized_lonlat(ring_xyz)
    n = len(ring_xyz)
    out = []
    for i in range(n):
        if abs(ring_xyz[i][2]) > 1.0 - 1e-9:
            pole = 90.0 if ring_xyz[i][2] > 0 else -90.0
            arr = float(lon[(i - 1) % n])
            dep = float(lon[(i + 1) % n])
            meridians = north_m if pole > 0 else south_m

            def rel_w(x):
                return (arr - x) % 360.0

            def rel_e(x):
                return (x - arr) % 360.0

            dw = rel_w(dep) or 360.0
            de = rel_e(dep) or 360.0
            west_clear = all(
                rel_w(m) == 0.0 or rel_w(m) >= dw for m in meridians)
            east_clear = all(
                rel_e(m) == 0.0 or rel_e(m) >= de for m in meridians)
            go_west = (pole > 0) if (west_clear and east_clear) else west_clear
            out.append((arr, pole))
            if go_west:
                if dep < arr:
                    _pole_run(out, arr, dep, pole)
                else:
                    _pole_run(out, arr, -180.0, pole)
                    out.append((180.0, pole))
                    _pole_run(out, 180.0, dep, pole)
            elif dep > arr:
                _pole_run(out, arr, dep, pole)
            else:
                _pole_run(out, arr, 180.0, pole)
                out.append((-180.0, pole))
                _pole_run(out, -180.0, dep, pole)
        else:
            out.append((float(lon[i]), float(lat[i])))
    return out


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
            # An exact ±180 → ∓180 pair (a pole traverse's explicit seam
            # crossing, `_ring_to_planar`) unwraps to a zero-length step:
            # nothing to interpolate, and the cut side is the pair's own
            # first side (the generic ``lo1u > lo0`` test cannot tell).
            if lo1u == lo0:
                boundary = lo0
            else:
                boundary = 180.0 if lo1u > lo0 else -180.0
            frac = 0.0 if lo1u == lo0 else (boundary - lo0) / (lo1u - lo0)
            la_x = la0 + frac * (la1 - la0)
            cur.append((boundary, la_x))
            segments.append(cur)
            cur = [(-boundary, la_x)]
    if not segments:
        return coords + [coords[0]], []
    segments[0] = cur + segments[0]  # the wrap-around segment closes the first
    return None, segments


def _stitch_segments(segments):
    """Reconnect antimeridian-cut *segments* into closed lon/lat rings.

    Every segment runs from a free end on ±180° to another on ±180°.  Walking
    from a segment's end, the next segment is the one whose **start** sits on the
    **same ±180° side** at the next latitude inward — on +180° the next start
    above, on -180° the next start below — so the connector edge runs straight
    along the meridian without crossing the boundary.  Pole seams are chosen
    **locally** (issue #147): with the covered region on every ring's left
    (the phase-1 orientation contract), the covered seam interval always runs
    upward from a cut end on +180° and downward on -180°, so an end with no
    same-side start in that direction wraps the pole on that side,
    unconditionally.  One ring may wrap both poles.  This replaces the global
    net-winding ``pole`` parameter, whose single value mis-stitched covers
    mixing a pole region with other seam-crossing parts.

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

    Returns
    -------
    list of list of tuple
        The reconnected closed rings of ``(lon, lat)`` degree pairs.

    Raises
    ------
    ValueError
        If the stitch fails to converge (a guard), or if no partner segment
        exists after a pole seam — the curated fail-loud convention (mirrors
        ``src_rust/src/dissolve.rs``, issue #181).
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
                raise ValueError(
                    "dissolved cover's antimeridian stitch did not converge "
                    "(an internal dissolve inconsistency); pass dissolve=False "
                    "for per-cell polygons"
                )
            used[idx] = True
            ring.extend(segs[idx])
            idx = _next_segment(segs, used, ring, seed)
        ring.append(ring[0])
        rings.append(ring)
    return rings


def _next_segment(segs, used, ring, seed):
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
    seed : int
        Index of the segment this ring started from.

    Returns
    -------
    int or None
        The next segment index, or ``None`` to close the ring.

    Raises
    ------
    ValueError
        If no partner segment exists on the other side after a pole seam —
        the curated fail-loud convention (mirrors
        ``src_rust/src/dissolve.rs``, issue #181).
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

    # No same-side start in that direction: the covered seam interval runs to
    # the pole on this side, so the region wraps it.  Cross the pole and
    # resume on the other side at the start nearest it: the highest start
    # after a north wrap (walking down -180°), the lowest after a south wrap
    # (walking up +180°).
    pole = 90.0 if side > 0 else -90.0
    other = -side
    ring.append((side, pole))
    ring.append((other, pole))
    ocands = [(segs[i][0][1], i) for i in range(len(segs))
              if abs(segs[i][0][0] - other) < 1e-9 and (not used[i] or i == seed)]
    if not ocands:
        raise ValueError(
            "dissolved cover's antimeridian stitch found no partner segment "
            "after a pole seam (an internal dissolve inconsistency, issue "
            "#181); pass dissolve=False for per-cell polygons"
        )
    la, i = max(ocands) if pole > 0 else min(ocands)
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


def _split_planar_at_revisits(ring):
    """Split a closed planar ring at exactly-repeated vertices.

    The planar mirror of :func:`_split_at_revisits`, keyed on exact
    coordinates; drops degenerate two-vertex slivers.  Mirrors
    ``split_planar_at_revisits`` in ``src_rust/src/dissolve.rs`` (issue
    #147).

    Parameters
    ----------
    ring : list of tuple
        A closed list of ``(lon, lat)`` degree pairs.

    Returns
    -------
    list of list of tuple
        Simple closed rings (each last vertex repeats the first).
    """
    out = []
    stack = []
    pos = {}

    def emit(cycle):
        if len(cycle) >= 3:
            out.append(cycle + [cycle[0]])

    for p in ring[:-1]:
        if p in pos:
            at = pos[p]
            cycle = stack[at:]
            del stack[at:]
            for c in cycle:
                pos.pop(c, None)
            emit(cycle)
        pos[p] = len(stack)
        stack.append(p)
    emit(stack)
    return out


def _frame_ring():
    """Return the whole-map frame shell, CCW (positive shoelace).

    The planar boundary of a covered region that encloses the entire
    antimeridian seam and both poles (e.g. world-minus-hole), which no
    spherical boundary ring supplies.  Mirrors ``frame_ring`` in
    ``src_rust/src/dissolve.rs`` (issue #147).
    """
    return [(-180.0, -90.0), (180.0, -90.0), (180.0, 90.0),
            (-180.0, 90.0), (-180.0, -90.0)]


def _emission_is_ccw(morton, step):
    """Whether cell boundary loops are emitted CCW (interior on the left).

    The issue #147 phase-1 orientation calibration: emission handedness is
    uniform over every cell and order (CCW at ``step == 1``, CW at
    ``step > 1`` — pinned by the phase-1 audit tests), so one cell's own fan
    area (~π/3·4^order, far above float noise) is the per-call witness.
    """
    from . import _healpix as hp
    from .orders import _rust_mort2nested

    first = np.ascontiguousarray(np.asarray(morton[:1], dtype=np.uint64))
    nested, depths = _rust_mort2nested(first)
    bnd = hp.boundaries(int(depths[0]), nested.astype(np.int64), step=step)
    if bnd.ndim == 2:
        bnd = bnd[np.newaxis, ...]
    return _spherical_signed_area(np.transpose(bnd, (0, 2, 1))[0]) > 0.0


def _dissolved_rings_py(morton, step):
    """Dissolve a cover to lon/lat rings in pure Python (the reference engine).

    This is the exact-verified reference engine kept as the test oracle for the
    Rust fast path (:func:`_dissolved_polygons` calls ``_rustie.rust_dissolve``
    at runtime — §7's Rust-only contract), mirroring the winding-free
    classifier of ``src_rust/src/dissolve.rs`` (issue #147): rings are
    oriented covered-region-on-the-left by one per-call calibration
    (:func:`_emission_is_ccw`), gated on per-ring simplicity
    (:func:`mortie.ring_is_simple`), cut and reconnected by the
    GeoJSON-convention splitter (:func:`_cut_at_antimeridian` /
    :func:`_stitch_segments`, pole seams locally decided), and classified in
    the plane by shoelace sign — positive (CCW, covered inside) is an
    exterior, negative a hole, with the explicit map-frame shell added when
    the covered region encloses the whole frame.  Hemisphere+ covers
    dissolve instead of raising.

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

    Raises
    ------
    ValueError
        If a boundary ring is self-crossing, or a stitch state cannot close
        (issue #181) — the curated fail-loud convention (PR #111).
    """
    from .coverage import ring_is_simple

    morton = np.atleast_1d(np.asarray(morton, dtype=np.uint64))
    if morton.size == 0:
        return [], []
    rings_xyz = _boundary_rings_xyz(morton, step)
    if not rings_xyz:
        # A nonempty cover with no surviving boundary edge is the whole
        # sphere: every edge cancelled, and the planar image is the frame.
        return [_frame_ring()], []

    if not _emission_is_ccw(morton, step):
        rings_xyz = [r[::-1] for r in rings_xyz]

    # Validity gate: a self-crossing ring has no consistent "left", so the
    # orientation contract cannot classify it.  (Identity conflicts — one
    # snapped coordinate at two non-adjacent positions — are accepted: they
    # are the working representation of corner-touch pinches resolved into
    # touching simple rings, the issue #155 ruling.)
    for r, ring in enumerate(rings_xyz):
        lat, lon = _xyz_to_latlon(ring)
        if not ring_is_simple(lat, lon):
            raise ValueError(
                f"dissolved cover produced a self-crossing boundary ring "
                f"(ring {r}), so its outline is not classifiable; pass "
                "dissolve=False for per-cell polygons"
            )

    # Rings that never cross the antimeridian are emitted whole; crossing
    # rings contribute open segments stitched back together with locally
    # decided pole seams.  Classification is planar: shoelace sign, with the
    # covered region on every ring's left.
    ext_pieces = []
    holes = []
    segments = []

    def classify(ring):
        # A planar ring can revisit a vertex exactly — e.g. a ring that both
        # expands a pole traverse and wraps the same pole's seam passes the
        # ±180/±90 corner twice — which even-odd fill reads correctly but
        # OGC does not allow in one ring; split into simple rings touching
        # at the point.
        for piece in _split_planar_at_revisits(ring):
            (ext_pieces if _planar_signed_area(piece) >= 0.0 else holes).append(
                piece
            )

    north_m, south_m = _pole_meridians(rings_xyz)
    for ring in rings_xyz:
        ll = _ring_to_planar(ring, north_m, south_m)
        whole, segs = _cut_at_antimeridian(ll)
        if whole is not None:
            classify(whole)
        else:
            segments.extend(segs)
    if segments:
        for piece in _stitch_segments(segments):
            classify(piece)

    # Frame rule: Σ shoelace is the covered region's planar area when its
    # boundary is complete, and that minus the full map when the region
    # encloses the frame — separated by sign, since a nonempty cover has
    # positive planar area.
    total = sum(_planar_signed_area(p) for p in ext_pieces + holes)
    if total < 0.0:
        ext_pieces.insert(0, _frame_ring())
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

"""Lazy WKB/WKT geometry codec for mortie (issue #71).

The runtime stays **numpy-only**: this module imports a geometry backend
(``shapely>=2`` preferred, ``spherely`` accepted) lazily and uses it *only* as a
codec — bytes/text ↔ ring coordinate arrays.  All spherical correctness
(antimeridian / pole handling) stays mortie's own job; the backend is never
asked for spatial predicates.  Importing :mod:`mortie` succeeds with neither
backend installed; the geometry functions raise a clear :class:`ImportError`
when first touched without one (the same lazy-gate pattern :mod:`mortie.arrow`
uses for pyarrow).

Coordinate convention: WKB/WKT store ``(x, y) = (lon, lat)`` degrees
(EPSG:4326).  mortie's coverage entry points take ``(lats, lons)``, so this
module flips the axes at the boundary and works in degrees throughout.
"""

import math

import numpy as np

# Cached backend: a ``(name, module)`` pair, resolved once on first use.
_BACKEND = None

# Snap scale for vertex identity in the dissolve edge-cancellation (rounding
# unit-vector components to 1e-10 makes a shared HEALPix corner — which both
# adjacent cells compute identically — a single integer-keyed vertex, so their
# shared edge cancels exactly without a floating tolerance search).
_DISSOLVE_SNAP = 1e10

# GEOS / shapely geometry type ids (shapely.get_type_id); spherely follows the
# same numbering.  Only the ones we classify on are named.
_TYPE_LINESTRING = 1
_TYPE_LINEARRING = 2
_TYPE_POLYGON = 3
_TYPE_MULTILINESTRING = 5
_TYPE_MULTIPOLYGON = 6


def _require_backend():
    """Import a geometry backend lazily, raising a clear error if absent.

    ``shapely>=2`` is the primary backend (its WKB/WKT codec is mature and is
    all we lean on); ``spherely`` is accepted if it is the one present.  Returns
    a ``(name, module)`` pair and caches it.
    """
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    try:
        import shapely

        _BACKEND = ("shapely", shapely)
        return _BACKEND
    except ImportError:
        pass
    try:
        import spherely

        _BACKEND = ("spherely", spherely)
        return _BACKEND
    except ImportError:
        pass
    raise ImportError(
        "mortie's WKB/WKT geometry I/O requires a geometry backend; install "
        "`shapely>=2` (preferred) or `spherely` (e.g. `pip install shapely`). "
        "mortie's runtime is numpy-only, so the backend is an optional extra."
    )


def _require_shapely(what):
    """Require the shapely backend for *what*, raising a clear error otherwise.

    The raw WKB/WKT codec works on either backend, but ring decomposition and
    SRID-tagged emit lean on shapely's geometry-introspection API
    (``get_exterior_ring`` / ``get_parts`` / ``set_srid``), which spherely's
    published surface does not yet expose.  Rather than fail with an opaque
    ``AttributeError`` deep inside, refuse up front with guidance.  Whether to
    invest in a spherely introspection shim is an open question for the issue
    thread (see the PR's "Questions for review").
    """
    name, mod = _require_backend()
    if name != "shapely":
        raise NotImplementedError(
            f"{what} currently requires the shapely>=2 backend; the active "
            f"backend is {name!r}, which mortie uses only as a raw WKB/WKT "
            "codec. Install shapely>=2 for this operation."
        )
    return mod


def _strip_ewkt_srid(text):
    """Drop a leading ``SRID=<n>;`` prefix from an EWKT string, if present.

    Plain WKT parsers reject the PostGIS EWKT prefix, so ingest tolerates it by
    stripping it (the SRID is advisory; mortie's contract is always EPSG:4326).
    """
    s = text.lstrip()
    if s[:5].upper() == "SRID=":
        semi = s.find(";")
        if semi != -1:
            return s[semi + 1:]
    return text


def geometry_from_wkb(data):
    """Decode WKB (or EWKB) bytes into a backend geometry object."""
    _, mod = _require_backend()
    return mod.from_wkb(data)


def geometry_from_wkt(text):
    """Decode WKT (or EWKT) text into a backend geometry object."""
    _, mod = _require_backend()
    return mod.from_wkt(_strip_ewkt_srid(text))


def geometry_to_wkb(geom, srid=None):
    """Encode a backend geometry to WKB bytes.

    With ``srid`` set (e.g. ``4326``), emit **EWKB** carrying that SRID
    (shapely backend only); otherwise emit plain ISO/OGC WKB (the default, no
    embedded CRS) — works on either backend.
    """
    if srid is not None:
        mod = _require_shapely("EWKB emit (srid=)")
        geom = mod.set_srid(geom, int(srid))
        return mod.to_wkb(geom, include_srid=True)
    _, mod = _require_backend()
    return mod.to_wkb(geom)


def geometry_to_wkt(geom, srid=None):
    """Encode a backend geometry to WKT text.

    With ``srid`` set, emit **EWKT** (``SRID=<n>;<WKT>``); otherwise plain WKT.
    """
    _, mod = _require_backend()
    text = mod.to_wkt(geom)
    if srid is not None:
        return f"SRID={int(srid)};{text}"
    return text


def _ring_latlon(mod, ring_geom):
    """Extract a ring's vertices as ``(lat, lon)`` float64 arrays (degrees)."""
    coords = np.asarray(mod.get_coordinates(ring_geom), dtype=np.float64)
    # WKB/WKT store (x, y) = (lon, lat).
    return coords[:, 1].copy(), coords[:, 0].copy()


def _polygon_rings(mod, poly):
    """All rings of one polygon as ``(lat, lon)`` pairs: exterior then holes."""
    rings = [_ring_latlon(mod, mod.get_exterior_ring(poly))]
    for i in range(int(mod.get_num_interior_rings(poly))):
        rings.append(_ring_latlon(mod, mod.get_interior_ring(poly, i)))
    return rings


def decompose(geom):
    """Decompose a backend geometry into mortie coverage inputs.

    Returns ``(kind, parts)`` where:

    * ``kind == "polygonal"`` and ``parts`` is a list of rings — exterior and
      interior (hole) rings of every polygon, flattened.  mortie's even-odd
      descent covers them in one pass, so disjoint outers union and nested
      rings carve holes (matching :func:`mortie.morton_coverage`'s multipart
      contract).
    * ``kind == "linear"`` and ``parts`` is a list of lines, one per
      (multi)linestring component.

    Each entry is a ``(lat, lon)`` pair of float64 degree arrays.  Any Z
    coordinate is dropped (mortie is 2-D lon/lat).  Points, geometry
    collections, and empty geometries are rejected — coverage has no meaning
    for them.

    Requires the shapely backend (it leans on shapely's ring/parts
    introspection); see :func:`_require_shapely`.
    """
    mod = _require_shapely("geometry decomposition")
    if bool(mod.is_empty(geom)):
        raise ValueError("empty geometry has no coverage")
    type_id = int(mod.get_type_id(geom))

    if type_id == _TYPE_POLYGON:
        return "polygonal", _polygon_rings(mod, geom)
    if type_id == _TYPE_MULTIPOLYGON:
        rings = []
        for poly in mod.get_parts(geom):
            rings.extend(_polygon_rings(mod, poly))
        return "polygonal", rings
    if type_id in (_TYPE_LINESTRING, _TYPE_LINEARRING):
        return "linear", [_ring_latlon(mod, geom)]
    if type_id == _TYPE_MULTILINESTRING:
        return "linear", [_ring_latlon(mod, ln) for ln in mod.get_parts(geom)]

    raise ValueError(
        f"unsupported geometry type for coverage (type id {type_id}); "
        "expected Polygon, MultiPolygon, LineString, or MultiLineString"
    )


# ── ingest: geometry → morton coverage ─────────────────────────────────────


def from_geometry(geom, order=18, moc=False, normalize=True,
                  tolerance=None, max_cells=None):
    """Cover a backend geometry with morton indices (issue #71).

    The geometry is decomposed via :func:`decompose` and routed to mortie's
    existing coverage entry points — so WKB/WKT ingest produces exactly the same
    cover as calling those functions on the same ``(lats, lons)`` arrays.

    * **Polygon / MultiPolygon** → :func:`mortie.morton_coverage` (flat) or, with
      ``moc=True``, :func:`mortie.morton_coverage_moc` (compact mixed-order).
      Holes and disjoint parts are handled by the one even-odd descent.
    * **LineString / MultiLineString** → :func:`mortie.linestring_coverage`.

    Parameters
    ----------
    geom : backend geometry
        A shapely/spherely geometry object (e.g. from :func:`geometry_from_wkb`).
    order : int, optional
        HEALPix order (1–29).  Default 18.
    moc : bool, optional
        Polygonal only: return a compact MOC instead of a flat cover.
    normalize : bool, optional
        Flat polygon cover only: auto-correct ring orientation at ingest
        (see :func:`mortie.morton_coverage`).  Ignored when ``moc=True`` and for
        linear geometry.  Note ``morton_coverage_moc`` has no orientation
        auto-correct, so with ``moc=True`` the ring winding is taken **as
        authored** — for hemisphere-plus polygons wind exteriors CCW / holes CW.
    tolerance, max_cells : optional
        Polygonal ``moc=True`` only: the adaptive stop criteria of
        :func:`mortie.morton_coverage_moc` (mutually exclusive).

    Returns
    -------
    numpy.ndarray or list of numpy.ndarray
        Polygonal → 1-D ``uint64`` morton array.  LineString → 1-D array;
        MultiLineString → list of arrays, one per line (the
        :func:`mortie.linestring_coverage` contract).
    """
    from .coverage import morton_coverage, morton_coverage_moc
    from .linestring import linestring_coverage

    kind, parts = decompose(geom)

    if kind == "polygonal":
        lats = [p[0] for p in parts]
        lons = [p[1] for p in parts]
        if moc:
            return morton_coverage_moc(
                lats, lons, order=order, tolerance=tolerance, max_cells=max_cells
            )
        return morton_coverage(lats, lons, order=order, normalize=normalize)

    # linear
    if moc or tolerance is not None or max_cells is not None:
        raise ValueError(
            "moc / tolerance / max_cells apply only to polygonal geometry"
        )
    if len(parts) == 1:
        return linestring_coverage(parts[0][0], parts[0][1], order=order)
    lats = [p[0] for p in parts]
    lons = [p[1] for p in parts]
    return linestring_coverage(lats, lons, order=order)


def from_wkb(data, order=18, moc=False, normalize=True,
             tolerance=None, max_cells=None):
    """Cover a geometry given as WKB (or EWKB) bytes.

    Thin wrapper: decode with :func:`geometry_from_wkb`, then
    :func:`from_geometry`.  See :func:`from_geometry` for the parameters.
    """
    return from_geometry(
        geometry_from_wkb(data), order=order, moc=moc, normalize=normalize,
        tolerance=tolerance, max_cells=max_cells,
    )


def from_wkt(text, order=18, moc=False, normalize=True,
             tolerance=None, max_cells=None):
    """Cover a geometry given as WKT (or EWKT) text.

    Thin wrapper: decode with :func:`geometry_from_wkt`, then
    :func:`from_geometry`.  See :func:`from_geometry` for the parameters.
    """
    return from_geometry(
        geometry_from_wkt(text), order=order, moc=moc, normalize=normalize,
        tolerance=tolerance, max_cells=max_cells,
    )


# ── emit: morton coverage → geometry ───────────────────────────────────────


def _per_cell_polygons(mod, morton, step):
    """Build one backend Polygon per cell of *morton* (lon/lat degrees).

    Reuses :func:`mortie.mort2polygon` for the corner→lon/lat boundary (with its
    antimeridian normalization), grouping by order so a mixed-order MOC cover is
    handled — ``mort2polygon`` itself requires a single order per call.
    """
    from .tools import _rust_mort2nested, mort2polygon

    morton = np.atleast_1d(np.asarray(morton, dtype=np.uint64))
    if morton.size == 0:
        return []

    _, depths = _rust_mort2nested(np.ascontiguousarray(morton))
    polys = []
    for d in np.unique(depths):
        grp = morton[depths == d]
        if grp.size == 1:
            rings_ll = [mort2polygon(int(grp[0]), step=step)]
        else:
            rings_ll = mort2polygon(grp, step=step)
        for ring in rings_ll:
            # mort2polygon yields closed [lat, lon] pairs; WKB wants (lon, lat).
            polys.append(mod.Polygon([(lon, lat) for lat, lon in ring]))
    return polys


# ── emit: dissolved-boundary outline (phase 4) ─────────────────────────────
#
# The dissolved outline is built natively (no backend spatial predicate): every
# cell contributes its boundary as a loop of directed edges; interior edges that
# two adjacent cells share are traversed in opposite directions and cancel, and
# the surviving edges chain into the outline rings.  Correctness is mortie's own
# job throughout — the backend is only asked to *construct* the final Polygon.


def _xyz_to_latlon(vecs):
    """Unit vectors ``(M, 3)`` → ``(lat, lon)`` degree arrays, lon in (-180, 180]."""
    z = np.clip(vecs[:, 2], -1.0, 1.0)
    lat = np.degrees(np.arcsin(z))
    lon = np.degrees(np.arctan2(vecs[:, 1], vecs[:, 0]))
    return lat, lon


def _spherical_signed_area(ring_xyz):
    """Signed area (steradians) of a spherical polygon of unit vectors.

    Positive = the region lies to the left of the directed boundary (an exterior
    ring); negative = the boundary winds the other way (a hole).  Uses the
    van Oosterom–Strackee signed-solid-angle sum over a fan from vertex 0.
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
    """Edge-cancellation dissolve of a cover → list of boundary rings.

    Each ring is an ``(M, 3)`` array of unit vectors (open — first vertex not
    repeated).  A mixed-order MOC is densified to its finest order first so every
    cell carries unit-length edges that cancel against their neighbours.  ``step``
    samples ``step`` points per cell edge (``step>1`` traces the curved HEALPix
    boundary); shared sub-edge points coincide between neighbours and cancel too.
    """
    from . import _healpix as hp
    from .coverage import moc_to_order
    from .tools import _rust_mort2nested

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
    """Azimuth (radians) from unit vector *p* toward unit vector *q*, in p's
    tangent plane (north-referenced).  Used to order edges around a vertex."""
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
    """Net signed longitude winding (degrees) and antimeridian-crossing count of
    a closed ring of longitudes.  Net ≈ ±360 ⟺ the ring encircles a pole."""
    deltas = np.diff(np.concatenate([lon, lon[:1]]))
    crossings = int(np.sum(np.abs(deltas) > 180.0))
    net = float(np.sum((deltas + 180.0) % 360.0 - 180.0))
    return net, crossings


def _cut_at_antimeridian(coords):
    """Cut an open lon/lat ring at every ±180° crossing.

    Returns ``(whole, segments)``: a ring that never crosses gives
    ``(closed_ring, [])`` (the caller keeps it whole); a crossing ring gives
    ``(None, [seg, ...])`` where each segment is an open polyline whose two free
    ends sit on ±180° (latitude linearly interpolated at the cut).  This is the
    GeoJSON-convention building block — :func:`_stitch_segments` reconnects the
    segments along the meridian (and, for a pole-enclosing region, through a
    ±90° pole vertex).
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
    vertex, cross to the other side at that pole, and resume.  ``pole`` is the
    pole the **filled** region encloses (``+90``/``-90``); it is only ever
    reached when the segments are genuinely unbalanced, so a non-pole cover
    never touches it.

    This is the GeoJSON / ``antimeridian``-package convention: a single split
    ``MultiPolygon`` with explicit ±90° pole vertices stitched down ±180°.  It
    generalises the old two-crossing split (each segment closing on its own
    side) to any even crossing count, to pole-enclosing caps, and to
    antimeridian-crossing holes.
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
    """Append meridian/pole connectors from the current ring end and return the
    next segment index (``None`` closes the ring).  See :func:`_stitch_segments`.
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
    """Even-odd ray-cast point-in-polygon (``ring`` = closed list of (x, y))."""
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
    """Shoelace signed area of a closed list of ``(x, y)`` (for size ordering)."""
    a = np.asarray(ring, dtype=np.float64)
    x, y = a[:, 0], a[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _ring_signed_area_lonlat(ring):
    """Spherical signed area (steradians) of a closed lon/lat-degree ring.

    Positive ⟺ the ring winds CCW (an exterior); negative ⟺ a hole.  Used to
    classify a stitched ring whose seam runs through a pole, where the planar
    shoelace sign is unreliable but the spherical area stays exact.
    """
    a = np.asarray(ring[:-1], dtype=np.float64)
    rlat = np.radians(a[:, 1])
    rlon = np.radians(a[:, 0])
    v = np.column_stack(
        [np.cos(rlat) * np.cos(rlon), np.cos(rlat) * np.sin(rlon), np.sin(rlat)]
    )
    return _spherical_signed_area(v)


def _dissolved_polygons(mod, morton, step):
    """Build the dissolved outline of *morton* as a list of backend Polygons.

    Exterior and hole rings come from the edge-cancellation engine; rings that
    cross the ±180° meridian are cut and reconnected by the GeoJSON-convention
    splitter (:func:`_cut_at_antimeridian` / :func:`_stitch_segments`), which
    inserts explicit ±90° pole vertices for a pole-enclosing region.  Holes are
    then nested into the exterior that contains them.  This handles pole caps
    (the project's polar data), exteriors crossing the antimeridian any even
    number of times, and antimeridian-crossing holes.
    """
    rings_xyz = _boundary_rings_xyz(morton, step)
    if not rings_xyz:
        return []

    # Normalise global winding: the cover's net signed area (exteriors minus
    # holes) is the covered area, always positive.  HEALPix orders boundary
    # points one way for step==1 and the other for step>1, so key the
    # exterior/hole sign off this invariant rather than a fixed convention.
    # (Spherical signed area is defined mod 4π, so this assumes the cover stays
    # well under a hemisphere — true for every realistic emit input.)
    areas = [_spherical_signed_area(r) for r in rings_xyz]
    if sum(areas) < 0.0:
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
            # shoelace sign is unreliable, but its spherical area is exact.
            (ext_pieces if _ring_signed_area_lonlat(piece) >= 0.0 else holes).append(
                piece
            )

    # Nest each hole into the smallest exterior piece that contains it.  A hole
    # vertex lies strictly inside its surrounding exterior, so test a vertex
    # (a guaranteed-interior point) rather than the centroid, which a concave or
    # split ring can push outside the region.
    hole_groups = [[] for _ in ext_pieces]
    ext_areas = [abs(_planar_signed_area(p)) for p in ext_pieces]
    for hole in holes:
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


def to_geometry(morton, dissolve=True, step=1):
    """Convert a morton cover to a backend geometry (issue #71).

    Parameters
    ----------
    morton : array_like of uint64
        A morton cover (flat or mixed-order MOC; each word self-encodes order).
    dissolve : bool, optional
        ``True`` (default) emits the single dissolved outline of the whole cover
        (exterior rings, holes, and disjoint components), built natively by
        edge-cancellation — no backend spatial predicate.  ``False`` emits a
        per-cell ``MultiPolygon`` — one quad per cell.
    step : int, optional
        Boundary points per cell edge (default 1 = 4 corners / straight chords).
        ``step>1`` densifies each edge to follow the curved HEALPix boundary.

    Returns
    -------
    backend geometry
        A shapely (or spherely) ``MultiPolygon`` in EPSG:4326 lon/lat degrees.

    Notes
    -----
    Emit requires the shapely backend (it constructs geometry objects).  The
    dissolved emit (``dissolve=True``) handles pole-enclosing covers (e.g. polar
    caps), exteriors crossing the antimeridian any even number of times, and
    antimeridian-crossing holes: crossing rings are cut at ±180° and reconnected
    by the GeoJSON convention — a single split ``MultiPolygon`` with explicit
    ±90° pole vertices stitched down the antimeridian.
    """
    mod = _require_shapely("geometry emit")
    if dissolve:
        return mod.MultiPolygon(_dissolved_polygons(mod, morton, step))
    return mod.MultiPolygon(_per_cell_polygons(mod, morton, step))


def to_wkb(morton, dissolve=True, step=1, srid=None):
    """Emit a morton cover as WKB (or EWKB) bytes.

    See :func:`to_geometry` for ``dissolve`` / ``step``.  With ``srid`` set
    (e.g. ``4326``), emit EWKB carrying that SRID; otherwise plain WKB.
    """
    return geometry_to_wkb(to_geometry(morton, dissolve=dissolve, step=step), srid=srid)


def to_wkt(morton, dissolve=True, step=1, srid=None):
    """Emit a morton cover as WKT (or EWKT) text.

    See :func:`to_geometry` for ``dissolve`` / ``step``.  With ``srid`` set,
    emit EWKT (``SRID=<n>;<WKT>``); otherwise plain WKT.
    """
    return geometry_to_wkt(to_geometry(morton, dissolve=dissolve, step=step), srid=srid)

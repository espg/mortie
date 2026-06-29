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
    from collections import Counter, defaultdict

    counts = Counter(edges)
    out = defaultdict(list)
    for (a, b), c in counts.items():
        net = c - counts.get((b, a), 0)
        for _ in range(net):
            out[a].append(b)

    # Chain the surviving directed edges into closed rings.
    rings = []
    for start in list(out.keys()):
        while out[start]:
            cur = start
            chain = [cur]
            while True:
                nxt = out[cur].pop()
                chain.append(nxt)
                cur = nxt
                if cur == start:
                    break
            rings.append(id_xyz[np.asarray(chain[:-1])])
    return rings


def _count_antimeridian_crossings(lon):
    """How many ring edges jump >180° in longitude (closed ring of lon degrees)."""
    closed = np.concatenate([lon, lon[:1]])
    return int(np.sum(np.abs(np.diff(closed)) > 180.0))


def _split_at_antimeridian(coords):
    """Split a lon/lat ring at the ±180° meridian into clean closed pieces.

    ``coords`` is an open list of ``(lon, lat)`` with an even number of
    antimeridian crossings (no enclosed pole — that case is rejected upstream).
    Each crossing edge is cut at the meridian (latitude linearly interpolated),
    and each resulting segment is closed along its own ±180° side.  Returns a
    list of closed ``(lon, lat)`` rings, all within a single hemisphere of
    longitude so the planar polygon is unambiguous.
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
    if segments:
        segments[0] = cur + segments[0]  # the wrap-around segment closes the first
    else:
        segments = [cur]
    return [seg + [seg[0]] for seg in segments]


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


def _dissolved_polygons(mod, morton, step):
    """Build the dissolved outline of *morton* as a list of backend Polygons.

    Exterior and hole rings come from the edge-cancellation engine; holes are
    nested into the exterior that contains them.  Covers whose outline encloses a
    pole, and antimeridian-crossing holes, are rejected with a clear
    :class:`NotImplementedError` (the spherical→planar pole/hole split is the
    remaining sub-piece of issue #71 — see the PR thread).
    """
    rings_xyz = _boundary_rings_xyz(morton, step)
    if not rings_xyz:
        return []

    # Normalise global winding: the cover's net signed area (exteriors minus
    # holes) is the covered area, always positive.  HEALPix orders boundary
    # points one way for step==1 and the other for step>1, so key the
    # exterior/hole sign off this invariant rather than a fixed convention.
    areas = [_spherical_signed_area(r) for r in rings_xyz]
    if sum(areas) < 0.0:
        rings_xyz = [r[::-1] for r in rings_xyz]
        areas = [-a for a in areas]

    ext_pieces = []
    holes = []
    for ring, area in zip(rings_xyz, areas):
        lat, lon = _xyz_to_latlon(ring)
        crossings = _count_antimeridian_crossings(lon)
        if crossings % 2 == 1:
            raise NotImplementedError(
                "dissolved emit of a pole-enclosing cover (e.g. a polar cap) is "
                "not yet supported; pass dissolve=False for the per-cell "
                "MultiPolygon (issue #71 phase 4 follow-up)"
            )
        ll = list(zip(lon.tolist(), lat.tolist()))
        if area >= 0.0:
            ext_pieces.extend(_split_at_antimeridian(ll))
        elif crossings > 0:
            raise NotImplementedError(
                "dissolved emit with an antimeridian-crossing hole is not yet "
                "supported; pass dissolve=False (issue #71 phase 4 follow-up)"
            )
        else:
            holes.append(ll + [ll[0]])

    # Nest each hole into the smallest exterior piece that contains it.
    hole_groups = [[] for _ in ext_pieces]
    ext_areas = [abs(_planar_signed_area(p)) for p in ext_pieces]
    for hole in holes:
        cx = float(np.mean([p[0] for p in hole]))
        cy = float(np.mean([p[1] for p in hole]))
        best = None
        for idx, piece in enumerate(ext_pieces):
            if _point_in_ring(cx, cy, piece) and (
                best is None or ext_areas[idx] < ext_areas[best]
            ):
                best = idx
        hole_groups[best if best is not None else 0].append(hole)

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
    dissolved emit (``dissolve=True``) does not yet support covers whose outline
    encloses a pole or holes that cross the antimeridian — both raise
    :class:`NotImplementedError`; use ``dissolve=False`` for those (issue #71
    phase 4 follow-up).
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

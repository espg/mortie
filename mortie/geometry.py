"""WKB/WKT geometry ingest and emit for mortie (issue #71).

The runtime stays **numpy-only**: :mod:`mortie.codec` imports a geometry backend
(``shapely>=2`` preferred, ``spherely`` accepted) lazily, and this module uses it
*only* as a codec — bytes/text ↔ ring coordinate arrays.  All spherical correctness
(antimeridian / pole handling) stays mortie's own job; the backend is never
asked for spatial predicates.  Importing :mod:`mortie` succeeds with neither
backend installed; the geometry functions raise a clear :class:`ImportError`
when first touched without one (the same lazy-gate pattern :mod:`mortie.arrow`
uses for pyarrow).

**WKB ingest needs no backend at all** (issue #157): :func:`from_wkb` parses the
bytes with mortie's own Rust reader and feeds the rings straight to the
coverage kernels.  What still needs a backend is WKT ingest (there is no Rust
WKT parser) and the whole *emit* direction — :func:`to_geometry` and friends
hand back a geometry object, which is a backend object by definition.

Coordinate convention: WKB/WKT store ``(x, y) = (lon, lat)`` degrees
(EPSG:4326).  mortie's coverage entry points take ``(lats, lons)``, so this
module flips the axes at the boundary and works in degrees throughout.
"""

import numpy as np

from .codec import (
    _geometry_from_wkt,
    _geometry_to_wkb,
    _geometry_to_wkt,
    _require_shapely,
)
from .dissolve import _dissolved_polygons

# GEOS / shapely geometry type ids (shapely.get_type_id); spherely follows the
# same numbering.  Only the ones we classify on are named.
_TYPE_LINESTRING = 1
_TYPE_LINEARRING = 2
_TYPE_POLYGON = 3
_TYPE_MULTILINESTRING = 5
_TYPE_MULTIPOLYGON = 6


def _ring_latlon(mod, ring_geom):
    """Extract a ring's vertices as ``(lat, lon)`` float64 arrays (degrees).

    Parameters
    ----------
    mod : module
        The active geometry backend module.
    ring_geom : backend geometry
        A ring (or any geometry whose coordinates are wanted).

    Returns
    -------
    tuple of numpy.ndarray
        The ``(lat, lon)`` float64 degree arrays.
    """
    coords = np.asarray(mod.get_coordinates(ring_geom), dtype=np.float64)
    # WKB/WKT store (x, y) = (lon, lat).
    return coords[:, 1].copy(), coords[:, 0].copy()


def _polygon_rings(mod, poly):
    """Extract all rings of one polygon: exterior then holes.

    Parameters
    ----------
    mod : module
        The active geometry backend module.
    poly : backend geometry
        A single Polygon.

    Returns
    -------
    list of tuple
        One ``(lat, lon)`` pair per ring, exterior first then the interior
        (hole) rings in order.
    """
    rings = [_ring_latlon(mod, mod.get_exterior_ring(poly))]
    for i in range(int(mod.get_num_interior_rings(poly))):
        rings.append(_ring_latlon(mod, mod.get_interior_ring(poly, i)))
    return rings


def decompose(geom):
    """Decompose a backend geometry into mortie coverage inputs.

    Each returned entry is a ``(lat, lon)`` pair of float64 degree arrays.  Any
    Z coordinate is dropped (mortie is 2-D lon/lat).

    Requires the shapely backend (it leans on shapely's ring/parts
    introspection); see :func:`_require_shapely`.

    Parameters
    ----------
    geom : backend geometry
        A Polygon, MultiPolygon, LineString, LinearRing, or MultiLineString.

    Returns
    -------
    tuple
        ``(kind, parts)`` where:

        * ``kind == "polygonal"`` and ``parts`` is a list of rings — exterior
          and interior (hole) rings of every polygon, flattened.  mortie's
          even-odd descent covers them in one pass, so disjoint outers union
          and nested rings carve holes (matching
          :func:`mortie.morton_coverage`'s multipart contract).
        * ``kind == "linear"`` and ``parts`` is a list of lines, one per
          (multi)linestring component.

    Raises
    ------
    ValueError
        For an empty geometry, or for a point / geometry collection / any
        other unsupported type — coverage has no meaning for them.
    NotImplementedError
        If the active backend is not shapely.
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


def _wkb_bytes(data, materialize=True):
    """Coerce one WKB input to ``bytes`` -- the input contract, in one place.

    Accepts ``bytes``, a hex ``str`` (as the backend path this replaces did --
    ``shapely.from_wkb`` documents "the WKB byte object or hexadecimal
    string"), and any one-byte-item buffer (``bytearray`` / ``memoryview`` /
    a ``uint8`` array -- a deliberate widening for arrow-backed callers,
    issue #157).  Everything else is refused **by name**: a bare
    ``bytes(data)`` assembles a blob out of *any* iterable of ints, which
    would turn a wrong-column argument into a plausible-looking cover.

    ``materialize=False`` applies the same accept list and raises the same
    errors, but produces no blob-sized object -- what :func:`from_wkbs`' serial
    pre-pass needs, so screening a whole column costs one transient hex decode
    instead of a second copy of the column.  The failure set is unchanged: a
    hex ``str`` still has to be decoded to know that it is hex, and a buffer's
    ``tobytes()`` cannot fail once ``itemsize`` is 1.

    Parameters
    ----------
    data : bytes, str, or buffer
        WKB/EWKB bytes, their hex spelling, or a byte buffer holding them.
    materialize : bool, optional
        Return the blob (the default).  With ``False``, validate only and
        return ``None``.

    Returns
    -------
    bytes or None
        The blob, ready for the Rust reader -- ``None`` when ``materialize``
        is ``False``.

    Raises
    ------
    TypeError
        For anything that is neither a string nor a buffer of bytes.
    ValueError
        For a ``str`` that is not valid hex.
    """
    if isinstance(data, bytes):
        return data if materialize else None
    if isinstance(data, str):
        try:
            decoded = bytes.fromhex(data)
        except ValueError as exc:
            raise ValueError(f"invalid WKB hex string: {exc}") from exc
        return decoded if materialize else None
    try:
        view = memoryview(data)
    except TypeError:
        raise TypeError(
            "WKB input must be bytes, a hex string, or a byte buffer; got "
            f"{type(data).__name__}"
        ) from None
    if view.itemsize != 1:
        raise TypeError(
            "WKB input must be a buffer of bytes; got one of "
            f"{view.itemsize}-byte items (format {view.format!r})"
        )
    return view.tobytes() if materialize else None


def _rings_from_wkb(data):
    """Decompose WKB (or EWKB) bytes into coverage inputs, backend-free.

    The Rust reader (``src_rust/src/wkb.rs``, issue #157) parses the blob
    itself, so this is :func:`decompose`'s contract without a geometry
    library in the loop: exterior **and** interior rings of **every** part,
    flattened, in ``(lat, lon)`` degrees.  It arrives as one ragged pair in
    arrow list layout and is sliced into per-ring views here.

    Parameters
    ----------
    data : bytes, str, or buffer
        WKB/EWKB bytes, their hex spelling, or a byte buffer holding them --
        see :func:`_wkb_bytes` for the accept list.

    Returns
    -------
    tuple
        ``(kind, parts)``, exactly as :func:`decompose` returns them.

    Raises
    ------
    ValueError
        For a truncated or malformed blob (including an unclosed polygon
        ring), an unsupported geometry type, or an empty geometry.
    TypeError
        For an input that is neither a string nor a buffer of bytes.
    """
    from . import _rustie

    kind, lats, lons, offsets = _rustie.rust_wkb_rings(_wkb_bytes(data))
    lats = np.asarray(lats)
    lons = np.asarray(lons)
    offsets = np.asarray(offsets)
    parts = [
        (lats[offsets[i]:offsets[i + 1]], lons[offsets[i]:offsets[i + 1]])
        for i in range(offsets.size - 1)
    ]
    return kind, parts


def _cover_parts(kind, parts, order, moc, normalize, tolerance, max_cells):
    """Route decomposed rings/lines to mortie's coverage entry points.

    The shared tail of :func:`from_geometry` and :func:`from_wkb` — the two
    differ only in how they reach ``(kind, parts)`` (a backend geometry via
    :func:`decompose`, or bytes via :func:`_rings_from_wkb`), and this keeps
    them producing the identical cover from there on.

    Parameters
    ----------
    kind : str
        ``"polygonal"`` or ``"linear"``, per :func:`decompose`.
    parts : list of tuple
        One ``(lat, lon)`` degree-array pair per ring or line.
    order, moc, normalize, tolerance, max_cells
        As :func:`from_geometry`.

    Returns
    -------
    numpy.ndarray or list of numpy.ndarray
        As :func:`from_geometry`.

    Raises
    ------
    ValueError
        If ``moc`` / ``tolerance`` / ``max_cells`` / ``normalize=False`` are
        passed for linear geometry.
    """
    from .coverage import morton_coverage, morton_coverage_moc
    from .linestring import linestring_coverage

    if kind == "polygonal":
        lats = [p[0] for p in parts]
        lons = [p[1] for p in parts]
        if moc:
            return morton_coverage_moc(
                lats, lons, order=order, tolerance=tolerance,
                max_cells=max_cells, normalize=normalize,
            )
        return morton_coverage(lats, lons, order=order, normalize=normalize)

    # linear
    if moc or tolerance is not None or max_cells is not None:
        raise ValueError(
            "moc / tolerance / max_cells apply only to polygonal geometry"
        )
    if not normalize:
        raise ValueError("normalize applies only to polygonal geometry")
    if len(parts) == 1:
        return linestring_coverage(parts[0][0], parts[0][1], order=order)
    lats = [p[0] for p in parts]
    lons = [p[1] for p in parts]
    return linestring_coverage(lats, lons, order=order)


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
        A shapely/spherely geometry object (e.g. from ``shapely.from_wkb``).
    order : int, optional
        HEALPix order (1–29).  Default 18.
    moc : bool, optional
        Polygonal only: return a compact MOC instead of a flat cover.
    normalize : bool, optional
        Polygonal: auto-correct ring orientation at ingest, on both the
        flat and the ``moc=True`` path (see :func:`mortie.morton_coverage`).
        Default ``True``: any simple ring whose interior decisively reads as
        the larger region is reversed so the smaller side is covered (S2's
        convention; issue #144 decision (A)), hemisphere-plus rings included.
        Pass ``False`` to take the winding **as authored** — the only way a
        WKB/WKT ring can express a bigger-than-complement interior (wind
        every ring, holes included, with its intended region on the left).
        ``normalize=False`` with linear geometry raises ``ValueError`` (a
        line has no ring orientation).
    tolerance, max_cells : optional
        Polygonal ``moc=True`` only: the adaptive stop criteria of
        :func:`mortie.morton_coverage_moc` (mutually exclusive).

    Returns
    -------
    numpy.ndarray or list of numpy.ndarray
        Polygonal → 1-D ``uint64`` morton array.  LineString → 1-D array;
        MultiLineString → list of arrays, one per line (the
        :func:`mortie.linestring_coverage` contract).

    Raises
    ------
    ValueError
        If ``moc`` / ``tolerance`` / ``max_cells`` are passed for linear
        geometry (they apply only to polygonal geometry), or from
        :func:`decompose` for an unsupported or empty geometry.
    """
    kind, parts = decompose(geom)
    return _cover_parts(kind, parts, order, moc, normalize, tolerance, max_cells)


def from_wkb(data, order=18, moc=False, normalize=True,
             tolerance=None, max_cells=None):
    """Cover a geometry given as WKB (or EWKB) bytes -- **no backend needed**.

    The blob is parsed by mortie's own Rust WKB reader (issue #157) and its
    rings go straight to the coverage kernels, so this works with neither
    shapely nor spherely installed — mortie's runtime really is numpy-only on
    this path.  The cover is identical to what the backend-decoded path
    produced: same rings, same descent.  (:func:`from_wkt` still decodes via a
    backend — #157 scoped the Rust parser to WKB.)

    Parameters
    ----------
    data : bytes, str, or buffer
        WKB or EWKB bytes.  Both byte orders, the ISO and EWKB dimension
        spellings (Z/M are dropped — mortie is 2-D lon/lat), and an EWKB SRID
        prefix (stripped; mortie's contract is always EPSG:4326) are accepted.
        A **hex string** of the blob is accepted too, as the backend-decoded
        path accepted one; so is any **byte buffer** (``bytearray`` /
        ``memoryview`` / a ``uint8`` array), which the backend path did not —
        a deliberate widening for arrow-backed callers.  Anything else (an
        iterable of ints included) is a ``TypeError`` naming its type.
    order, moc, normalize, tolerance, max_cells : optional
        Forwarded to :func:`from_geometry` unchanged.  See there for the full
        contract — in particular that ``morton_coverage_moc`` has no
        orientation auto-correct, so with ``moc=True`` the ring winding is
        taken **as authored**.

    Returns
    -------
    numpy.ndarray or list of numpy.ndarray
        As :func:`from_geometry`.

    Raises
    ------
    ValueError
        As :func:`from_geometry` — including ``moc`` / ``tolerance`` /
        ``max_cells`` passed for linear geometry — plus, from the reader, a
        truncated or malformed blob (an unclosed polygon ring included), an
        unsupported geometry type, or an empty geometry; and for a ``str``
        that is not valid hex.
    TypeError
        For an input that is neither a string nor a buffer of bytes.

    See Also
    --------
    from_geometry : The shared parameter semantics and the full contract.
    """
    kind, parts = _rings_from_wkb(data)
    return _cover_parts(kind, parts, order, moc, normalize, tolerance, max_cells)


def from_wkt(text, order=18, moc=False, normalize=True,
             tolerance=None, max_cells=None):
    """Cover a geometry given as WKT (or EWKT) text.

    Thin wrapper: decode with the geometry backend, then
    :func:`from_geometry`.  Unlike :func:`from_wkb`, this **does** need a
    backend installed — mortie has no Rust WKT parser (issue #157 scoped the
    reader to WKB).

    Parameters
    ----------
    text : str
        WKT or EWKT text.
    order, moc, normalize, tolerance, max_cells : optional
        Forwarded to :func:`from_geometry` unchanged.  See there for the full
        contract — in particular that ``morton_coverage_moc`` has no
        orientation auto-correct, so with ``moc=True`` the ring winding is
        taken **as authored**.

    Returns
    -------
    numpy.ndarray or list of numpy.ndarray
        As :func:`from_geometry`.

    Raises
    ------
    ValueError
        As :func:`from_geometry` — including ``moc`` / ``tolerance`` /
        ``max_cells`` passed for linear geometry.

    See Also
    --------
    from_geometry : The shared parameter semantics and the full contract.
    """
    return from_geometry(
        _geometry_from_wkt(text), order=order, moc=moc, normalize=normalize,
        tolerance=tolerance, max_cells=max_cells,
    )


# ── emit: morton coverage → geometry ───────────────────────────────────────


def _per_cell_polygons(mod, morton, step):
    """Build one backend Polygon per cell of *morton* (lon/lat degrees).

    Reuses :func:`mortie.mort2polygon` for the corner→lon/lat boundary (with its
    antimeridian normalization), grouping by order so a mixed-order MOC cover is
    handled — ``mort2polygon`` itself requires a single order per call.

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
        One backend Polygon per cell; empty for an empty cover.
    """
    from .convert import mort2polygon
    from .orders import _rust_mort2nested

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

    Raises
    ------
    NotImplementedError
        If the active backend is not shapely, or if a dissolved hole nests
        into no exterior (pass ``dissolve=False``).

    Notes
    -----
    Emit requires the shapely backend (it constructs geometry objects).  The
    dissolved emit (``dissolve=True``) handles pole-enclosing covers (e.g. polar
    caps), exteriors crossing the antimeridian any even number of times, and
    antimeridian-crossing holes: crossing rings are cut at ±180° and reconnected
    by the GeoJSON convention — a single split ``MultiPolygon`` with explicit
    ±90° pole vertices stitched down the antimeridian.  A cover spanning near
    or over a hemisphere (2π sr), or one with a boundary ring enclosing more
    than a hemisphere (e.g. an equatorial band), raises ``ValueError`` — its
    exterior/hole winding is ambiguous (issue #108); split such a cover or use
    ``dissolve=False``.
    """
    mod = _require_shapely("geometry emit")
    if dissolve:
        return mod.MultiPolygon(_dissolved_polygons(mod, morton, step))
    return mod.MultiPolygon(_per_cell_polygons(mod, morton, step))


def to_wkb(morton, dissolve=True, step=1, srid=None):
    """Emit a morton cover as WKB (or EWKB) bytes.

    Parameters
    ----------
    morton : array_like of uint64
        A morton cover (flat or mixed-order MOC).
    dissolve, step : optional
        Forwarded to :func:`to_geometry` unchanged; see there for the full
        contract (pole caps, antimeridian splitting, edge densification).
    srid : int, optional
        With ``srid`` set (e.g. ``4326``), emit EWKB carrying that SRID;
        otherwise plain WKB.

    Returns
    -------
    bytes
        The encoded WKB (or EWKB) bytes.

    Raises
    ------
    NotImplementedError
        As :func:`to_geometry` — a non-shapely backend, or a dissolved hole
        that nests into no exterior.

    See Also
    --------
    to_geometry : The ``dissolve`` / ``step`` contract in full.
    """
    geom = to_geometry(morton, dissolve=dissolve, step=step)
    return _geometry_to_wkb(geom, srid=srid)


def to_wkt(morton, dissolve=True, step=1, srid=None):
    """Emit a morton cover as WKT (or EWKT) text.

    Parameters
    ----------
    morton : array_like of uint64
        A morton cover (flat or mixed-order MOC).
    dissolve, step : optional
        Forwarded to :func:`to_geometry` unchanged; see there for the full
        contract (pole caps, antimeridian splitting, edge densification).
    srid : int, optional
        With ``srid`` set, emit EWKT (``SRID=<n>;<WKT>``); otherwise plain WKT.

    Returns
    -------
    str
        The encoded WKT (or EWKT) text.

    Raises
    ------
    NotImplementedError
        As :func:`to_geometry` — a non-shapely backend, or a dissolved hole
        that nests into no exterior.

    See Also
    --------
    to_geometry : The ``dissolve`` / ``step`` contract in full.
    """
    geom = to_geometry(morton, dissolve=dissolve, step=step)
    return _geometry_to_wkt(geom, srid=srid)

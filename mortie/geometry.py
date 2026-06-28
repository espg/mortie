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

# GEOS / shapely geometry type ids (shapely.get_type_id); spherely follows the
# same numbering.  Only the ones we classify on are named.
_TYPE_POINT = 0
_TYPE_LINESTRING = 1
_TYPE_LINEARRING = 2
_TYPE_POLYGON = 3
_TYPE_MULTIPOINT = 4
_TYPE_MULTILINESTRING = 5
_TYPE_MULTIPOLYGON = 6
_TYPE_GEOMETRYCOLLECTION = 7

_POLYGONAL = (_TYPE_POLYGON, _TYPE_MULTIPOLYGON)
_LINEAR = (_TYPE_LINESTRING, _TYPE_MULTILINESTRING, _TYPE_LINEARRING)


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

    With ``srid`` set (e.g. ``4326``), emit **EWKB** carrying that SRID;
    otherwise emit plain ISO/OGC WKB (the default, no embedded CRS).
    """
    _, mod = _require_backend()
    if srid is not None:
        geom = mod.set_srid(geom, int(srid))
        return mod.to_wkb(geom, include_srid=True)
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

    Each entry is a ``(lat, lon)`` pair of float64 degree arrays.  Points,
    geometry collections, and empty geometries are rejected — coverage has no
    meaning for them.
    """
    _, mod = _require_backend()
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

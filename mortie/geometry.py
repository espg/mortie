"""Lazy WKB/WKT geometry codec for mortie (issue #71).

The runtime stays **numpy-only**: this module imports a geometry backend
(``shapely>=2`` preferred, ``spherely`` accepted) lazily and uses it *only* as a
codec — bytes/text ↔ ring coordinate arrays.  All spherical correctness
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
    all we lean on); ``spherely`` is accepted if it is the one present.

    Returns
    -------
    tuple
        A ``(name, module)`` pair, cached on the module after the first call.

    Raises
    ------
    ImportError
        If neither backend is installed, with the install command in the
        message.
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

    Parameters
    ----------
    what : str
        The operation being attempted, named in the error message.

    Returns
    -------
    module
        The imported ``shapely`` module.

    Raises
    ------
    NotImplementedError
        If the active backend is not shapely.
    ImportError
        If no backend at all is installed (via :func:`_require_backend`).
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

    Parameters
    ----------
    text : str
        WKT or EWKT text.

    Returns
    -------
    str
        ``text`` with any leading ``SRID=<n>;`` removed; ``text`` unchanged if
        there was none.
    """
    s = text.lstrip()
    if s[:5].upper() == "SRID=":
        semi = s.find(";")
        if semi != -1:
            return s[semi + 1:]
    return text


# ── the backend's own codec, wrapped for internal use ──────────────────────
#
# These four are two-line pass-throughs to whichever backend resolved, with
# no mortie logic of their own, and they are not exported at package level.
# They are private (espg ruling, 2026-08-07): re-exporting another library's
# codec under a mortie name buys nothing, and a caller who wants a shapely
# object calls ``shapely.from_wkb`` themselves.  What mortie exports is the
# ingest/emit pair that does something -- ``from_wkb`` (now backend-free,
# issue #157), ``from_wkt``, ``to_wkb``, ``to_wkt``.


def _geometry_from_wkb(data):
    """Decode WKB (or EWKB) bytes into a backend geometry object.

    Parameters
    ----------
    data : bytes
        WKB or EWKB bytes.

    Returns
    -------
    backend geometry
        A shapely (or spherely) geometry object.
    """
    _, mod = _require_backend()
    return mod.from_wkb(data)


def _geometry_from_wkt(text):
    """Decode WKT (or EWKT) text into a backend geometry object.

    Parameters
    ----------
    text : str
        WKT or EWKT text; a leading ``SRID=<n>;`` prefix is stripped.

    Returns
    -------
    backend geometry
        A shapely (or spherely) geometry object.
    """
    _, mod = _require_backend()
    return mod.from_wkt(_strip_ewkt_srid(text))


def _geometry_to_wkb(geom, srid=None):
    """Encode a backend geometry to WKB bytes.

    Parameters
    ----------
    geom : backend geometry
        The geometry to encode.
    srid : int, optional
        With ``srid`` set (e.g. ``4326``), emit **EWKB** carrying that SRID
        (shapely backend only); otherwise emit plain ISO/OGC WKB (the default,
        no embedded CRS) — works on either backend.

    Returns
    -------
    bytes
        The encoded WKB (or EWKB) bytes.

    Raises
    ------
    NotImplementedError
        If ``srid`` is set and the active backend is not shapely.
    """
    if srid is not None:
        mod = _require_shapely("EWKB emit (srid=)")
        geom = mod.set_srid(geom, int(srid))
        return mod.to_wkb(geom, include_srid=True)
    _, mod = _require_backend()
    return mod.to_wkb(geom)


def _geometry_to_wkt(geom, srid=None):
    """Encode a backend geometry to WKT text.

    Parameters
    ----------
    geom : backend geometry
        The geometry to encode.
    srid : int, optional
        With ``srid`` set, emit **EWKT** (``SRID=<n>;<WKT>``); otherwise plain
        WKT.

    Returns
    -------
    str
        The encoded WKT (or EWKT) text.
    """
    _, mod = _require_backend()
    text = mod.to_wkt(geom)
    if srid is not None:
        return f"SRID={int(srid)};{text}"
    return text


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


def from_wkbs(blobs, order=18, tolerance=None, max_cells=None, normalize=True):
    """Cover many WKB blobs with one call -- ragged MOCs out, no backend.

    The batch sibling of :func:`from_wkb` (issue #157) and the plural twin its
    name marks: **one MOC per input blob** (many→many), against the many→one
    union :func:`from_wkb` performs over the rings *inside* one blob.  The
    whole column crosses the Python/Rust boundary once, and Rust parses and
    covers the blobs in parallel with the GIL released — so the per-call fixed
    cost that dominates a Python loop over half a million footprints is paid
    once.  Result ``i`` is byte-identical to
    ``from_wkb(blobs[i], order=order, moc=True, ...)``.

    Memory: a chunk ends at 2048 blobs **or 64 MiB, whichever comes first**,
    and peak is the returned ``values`` array plus **one chunk of copied input
    bytes** (the copy is mandatory — a Python ``bytes`` buffer is GIL-bound and
    cannot cross into the parallel region) plus one chunk of in-flight covers.
    Neither the whole column's bytes nor every blob's cover is ever resident at
    once, and that holds **for every input spelling and every blob size**:
    non-``bytes`` entries are coerced inside the chunk, so their copy dies with
    it, and the byte budget stops 2048 fat geometries from making "one chunk"
    mean gigabytes.  Measured on the 555,867-blob ATL03 v007 corpus (276.7 MiB
    of WKB, 167.3 MiB of result, order 6), peak growth over the resident
    column:

    ==========================  ==========  =========
    input spelling              peak        × result
    ==========================  ==========  =========
    ``list[bytes]``             178.7 MiB   1.07
    numpy object array          179.4 MiB   1.07
    hex ``str``                 178.9 MiB   1.07
    ``bytearray``               178.9 MiB   1.07
    ``memoryview``              179.4 MiB   1.07
    ``uint8`` array             221.2 MiB   1.32
    arrow buffer slices         179.3 MiB   1.07
    ==========================  ==========  =========

    On the fat end, 3,000 Antarctic-basin blobs (1.25 MiB each, a 3.7 GiB
    column) peak at 610-634 MiB, against 3,120 MiB when the chunk was bounded
    by blob count alone — and most of what is left is the in-flight cover
    work, not the copy.

    Parameters
    ----------
    blobs : sequence
        One WKB/EWKB geometry per entry.  Each entry takes exactly what
        :func:`from_wkb` takes — ``bytes``, a hex ``str``, or any
        one-byte-item buffer (see :func:`_wkb_bytes`); the batch narrows
        nothing.  A list of ``bytes`` or a numpy object array (what
        ``pandas``/``pyarrow`` hand back for a binary column) both work as
        they are.  **Byte buffers are first-class, not merely tolerated**: a
        buffer column costs the same peak a ``bytes`` column does (the table
        above), so an arrow-backed caller should hand over zero-copy
        ``memoryview`` slices of the column's value buffer rather than pay
        ``to_pylist()`` — ``mv = memoryview(arr.buffers()[2])`` and
        ``[mv[o[i]:o[i + 1]] for i in range(len(arr))]`` is the cheap call
        shape, and it measures the same 1.07× as ``bytes``.
    order : int, optional
        Finest HEALPix order (1-29), shared by every blob.  Default 18.
    tolerance : float, optional
        Stop refining a boundary cell at this angular radius in **degrees** —
        :func:`mortie.morton_coverage_moc`'s ``tolerance``, applied as a
        **single shared setting**, mutually exclusive with ``max_cells``.
    max_cells : int, optional
        Per-blob cell budget, shared by every blob.  A budget below some
        blob's representable floor is raised for that blob (soft target, as
        in the scalar path) and one summary warning is emitted.
    normalize : bool, optional
        Ring-orientation handling, identical in meaning to
        :func:`from_wkb`'s ``normalize``.  Default ``True``.

    Returns
    -------
    values : numpy.ndarray
        Every blob's morton MOC words concatenated (``uint64``).
    out_offsets : numpy.ndarray
        ``int64`` arrow list offsets into ``values``, length
        ``len(blobs) + 1``; blob ``i``'s MOC is
        ``values[out_offsets[i]:out_offsets[i+1]]``.  ``out_offsets[0]`` is
        always 0 and ``out_offsets[-1]`` is always ``len(values)``.

    Raises
    ------
    ValueError
        Fail-fast, naming the **lowest-index** offending blob (e.g.
        ``blob 4217: truncated WKB ...``): a malformed or truncated blob, an
        unclosed polygon ring, an unsupported or empty geometry, a ring with
        fewer than 3 vertices, or a NaN/infinite coordinate.  **Linear
        geometry is refused by index** — a LineString cover is one array per
        line, which has no single-MOC-per-blob spelling; use
        :func:`from_wkb` for those.  Also for ``order`` outside 1-29, both
        ``tolerance`` and ``max_cells`` given, or an invalid hex string.
    TypeError
        Naming the offending index, for an entry that is neither a string nor
        a buffer of bytes.

    Notes
    -----
    Two ordered gates, as in :func:`mortie.polygons_to_morton_mocs`: the input
    contract is screened by a serial pre-pass over the whole sequence, then
    the blobs are parsed and covered.  Each gate reports its own lowest-index
    offender, so a ``TypeError`` at a high index does surface ahead of a
    malformed blob at a lower one — the pre-pass is an earlier gate, not a
    competing one.  The pre-pass **validates without retaining**
    (``_wkb_bytes(..., materialize=False)``): it applies the identical accept
    list — an invalid hex string is still caught here, ahead of any parse
    error — but keeps the entries as they came, so a column in a non-``bytes``
    spelling is not duplicated for the duration of the call.

    Warns
    -----
    UserWarning
        If ``max_cells`` is below the minimum needed to represent some blob;
        the warning reports how many were raised and names the lowest-index
        one.

    See Also
    --------
    from_wkb : the scalar (one blob) form, and the input contract in full.

    Examples
    --------
    >>> import mortie                                    # doctest: +SKIP
    >>> values, off = mortie.from_wkbs(wkb_column, order=8)   # doctest: +SKIP
    >>> first = values[off[0]:off[1]]   # the first blob's MOC
    """
    from . import _rustie

    if tolerance is not None and max_cells is not None:
        raise ValueError("pass at most one of tolerance / max_cells")
    # Serial screening pass, in index order, so the lowest-index bad entry is
    # what a caller sees -- the same fail-fast rule the Rust side applies to
    # parse/cover failures.  It *validates* rather than coerces: the entries
    # are kept as they came, so this costs a list of pointers whatever spelling
    # the column is in, and the byte-producing coercion happens per chunk on
    # the Rust side, where it is released with the chunk (issue #157).
    entries = []
    for i, blob in enumerate(blobs):
        try:
            _wkb_bytes(blob, materialize=False)
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"blob {i}: {exc}") from exc
        entries.append(blob)
    tol_rad = None if tolerance is None else np.radians(float(tolerance))
    values, out_offsets = _rustie.rust_wkbs_coverage_mocs(
        entries, _wkb_bytes, order, tol_rad, max_cells, normalize
    )
    return np.asarray(values), np.asarray(out_offsets)


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
    from .tools import _rust_mort2nested

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

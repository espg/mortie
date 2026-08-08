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

import numpy as np

from .dissolve import _dissolved_polygons

# Cached backend: a ``(name, module)`` pair, resolved once on first use.
_BACKEND = None

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
        above), so zero-copy ``memoryview`` slices of an Arrow column's value
        buffer are as cheap an input as ``bytes``.  Cutting those slices out
        of a pyarrow column correctly is the hard part, and
        :func:`mortie.arrow.from_wkbs` now does it for you — see Notes.
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

    Feeding a **pyarrow** column has a typed entry point of its own —
    :func:`mortie.arrow.from_wkbs` (issue #163) — and that is what to call:
    ``marrow.from_wkbs(column, order=...)`` returns exactly this pair.  Do
    not improvise the extraction.  Four traps sit between a column and its
    blobs.  **Three are silent** — they yield different, valid-looking data
    rather than an error: a parquet column reads back as a ``ChunkedArray``,
    which has no ``.buffers()`` at all; ``slice`` and ``take`` are zero-copy
    metadata, so a chunk's buffers belong to the *original* array and must be
    indexed from ``chunk.offset``; and a ``large_binary`` column's offsets are
    ``int64``, not ``int32``.  The fourth is a **wrong diagnosis**: a null
    entry spans zero bytes, so it arrives as an empty blob and this function
    reports it as a truncated geometry rather than as a missing one.  The
    skin handles all four, and hands this function zero-copy ``memoryview``
    slices.

    ``from_wkbs(column.to_pylist(), ...)`` is also right on every one of
    those cases and needs no pyarrow-typed call, but its cost is real: on the
    555,867-row ATL03 v007 WKB column (290.1 MB of payload) ``to_pylist()``
    peaks at **~322 MB** of Python ``bytes`` objects — the per-object
    overhead on top of the payload, plus the list — against **~112 MB** for
    the skin's views of the same column.

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

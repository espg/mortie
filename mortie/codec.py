"""The geometry backend and its codec, wrapped for mortie's internal use.

Two things, and only these: the **lazy backend gate** (:func:`_require_backend`
resolves ``shapely>=2``, or ``spherely`` if that is the one present, once and
caches it in :data:`_BACKEND`; :func:`_require_shapely` narrows to shapely for
the operations that construct geometry) and the **codec quartet** wrapping the
resolved backend's own WKB/WKT reader and writer, plus :func:`_strip_ewkt_srid`,
the one piece of parsing mortie does itself.

Every name here is private, and stays private (espg ruling on issue #157): this
module is plumbing, not API.  What mortie exports is the ingest/emit pair that
does something -- ``from_wkb`` (backend-free since issue #157), ``from_wkt``,
``to_wkb``, ``to_wkt`` -- all of which live in :mod:`mortie.geometry` and reach
the backend through here.

Split out of :mod:`mortie.geometry` (issue #159) to keep that module under the
~1,000-line aim.  It needs no imports of its own: the backend is imported
lazily, inside :func:`_require_backend`, which is the whole point of the gate.
:mod:`mortie.geometry` imports four names from here; nothing here imports back.
"""

# Cached backend: a ``(name, module)`` pair, resolved once on first use.
_BACKEND = None


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

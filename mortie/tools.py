"""
functions for morton indexing
"""

import numpy as np

from . import _healpix as hp
from . import _rustie

# Rust-native geo2mort (uses healpix crate, no Python HEALPix backend)
_rust_geo2mort = _rustie.rust_geo2mort
# Packed-word kernel bridge: morton <-> HEALPix NESTED (vectorized).
_rust_mort2nested = _rustie.rust_mort2nested
_rust_nested2mort = _rustie.rust_nested2mort

# HEALPix orders the packed-u64 kernel reaches (mirrors decimal_morton::MAX_ORDER
# and morton_index.MAX_ORDER): 0 = base cell, 29 = max resolution and the only
# order that carries point words.
MAX_ORDER = 29


def order2res(order):
    res = 111 * 58.6323 * .5**order
    return res


def res2display(max_order=MAX_ORDER):
    '''prints resolution levels

    Prints one line per tessellation order from 0 through ``max_order``
    (default MAX_ORDER = 29, the finest order the packed-u64 kernel reaches).
    Each resolution is rendered in the largest sensible unit -- km at coarse
    orders, m once it drops below 1 km, cm once it drops below 1 m -- and
    rounded to three decimals within that bracket, so fine orders read
    naturally (e.g. order 12 -> ``1.589 km``, order 13 -> ``794.456 m``)
    rather than as tiny km fractions.

    ``max_order`` must lie in 0..MAX_ORDER, the order range the packed-u64
    kernel encodes.
    '''
    if not 0 <= max_order <= MAX_ORDER:
        raise ValueError(
            f"max_order must be between 0 and {MAX_ORDER}, got {max_order!r}")
    for res in range(max_order + 1):
        km = order2res(res)
        if km >= 1.0:
            value, unit = km, 'km'
        elif km >= 1e-3:
            value, unit = km * 1e3, 'm'
        else:
            value, unit = km * 1e5, 'cm'
        print(str(round(value, 3)) + ' ' + unit +
              ' at tessellation order ' + str(res))


def unique2parent(unique):
    '''
    Assumes input is UNIQ
    Currently only works on single resolution
    Returns parent base cell
    '''
    orders = np.log2(np.array(unique)/4.0)//2.0
    # this is such an ugly hack-- does little, will blow up with multi res
    orders_ = np.unique(orders)
    if len(orders_) == 1:
        order = int(orders_[0])
    else:
        raise NotImplementedError("Cannot parse mixed resolution unique cells")
    unique = unique // 4**(order-1)
    parent = (unique - 16) // 4
    return parent


def heal_norm(base, order, addr_nest):
    N_pix = hp.order2nside(order)**2
    addr_norm = addr_nest - (base * N_pix)
    return addr_norm


# Public API - uses Rust (the packed-u64 kernel)
def norm2mort(normed, parent, order):
    """Convert a normalized HEALPix address + base cell to a packed morton word.

    The exact inverse of :func:`mort2norm`: ``mort2norm(norm2mort(n, p, o))``
    returns ``(n, p, o)``. Born order-29-native (issue #48) — there is no order
    cap beyond the kernel's ``MAX_ORDER`` of 29. The returned ``uint64`` is the
    packed ``decimal_morton`` word (issue #58; the prefix is ``base+1``, so bit 63
    is set — a large unsigned value — for base cells 7-11), not the retired
    decimal encoding.

    Parameters
    ----------
    normed : int or array
        Normalized HEALPix address (the in-base z-order, ``0 <= normed < 4**order``).
    parent : int or array
        Parent base cell (0-11).
    order : int
        HEALPix order (0-29).

    Returns
    -------
    morton : uint64 or ndarray
        Packed morton word(s).
    """
    normed = np.atleast_1d(np.asarray(normed, dtype=np.int64))
    parent = np.atleast_1d(np.asarray(parent, dtype=np.int64))
    is_scalar = normed.size == 1 and parent.size == 1
    # nested = parent * nside^2 + normed; pack via the kernel bridge.
    nested = (parent.astype(np.uint64) << np.uint64(2 * order)) | normed.astype(
        np.uint64
    )
    n = max(normed.size, parent.size)
    nested = np.ascontiguousarray(np.broadcast_to(nested, (n,)))
    depths = np.full(nested.size, order, dtype=np.uint8)
    morton = _rust_nested2mort(nested, depths)
    if is_scalar:
        return np.uint64(morton[0])
    return morton


def geo2uniq(lats, lons, order=18):
    """Calculates UNIQ coding for lat/lon

    Defaults to order 18; the kernel reaches order 29 (``MAX_ORDER``)."""

    nside = 2**order

    nest = hp.ang2pix(order, lons, lats)
    uniq = 4 * (nside**2) + nest

    return uniq


def geo2mort(lats, lons, order=None, points=None):
    """Calculates morton indices from geographic coordinates

    The entire pipeline runs in Rust via the ``healpix`` crate — no
    Python HEALPix backend is needed.

    lat/lon inputs are treated as **points** by default (indeterminate
    resolution, encoded at max precision), so a bare ``geo2mort(lats, lons)``
    returns order-29 ``Kind::Point`` words. Passing an explicit ``order`` asks
    for an **area** cell at that resolution instead (``points`` inferred
    ``False``). The two flags resolve as:

    * ``order=None, points=None`` (bare call) -> order-29 **point** words;
    * an explicit ``order`` with ``points`` unset -> **area** cell at ``order``;
    * ``points=True`` -> order-29 point words (order-29-only; an explicit
      ``order != 29`` raises ``ValueError``, matching
      :meth:`MortonIndexArray.from_latlon`);
    * ``points=False`` -> area cell at ``order`` (``order=None`` -> 29).

    Non-finite ``lat``/``lon`` encode to the reserved empty word ``0`` (base
    cell 0 is the null sentinel) on both the area and point routes.

    Parameters
    ----------
    lats : array-like
        Latitude(s) in degrees.
    lons : array-like
        Longitude(s) in degrees.
    order : int, optional
        HEALPix order (0-29). Defaults to 29. An explicit value implies an area
        cell unless ``points=True`` is also given.
    points : bool, optional
        Encode ``Kind::Point`` (order-29) vs ``Kind::Area`` words. Defaults to
        ``True`` for a bare call and ``False`` when an ``order`` is given.

    Returns
    -------
    ndarray
        Packed ``uint64`` morton word(s), same shape family as the input
        (scalar in -> length-1 ndarray)."""

    # Resolve the point/area mode: a bare call encodes points; an explicit order
    # implies an area cell at that resolution unless the caller forces points.
    if points is None:
        points = order is None
    if order is None:
        order = MAX_ORDER
    if points and int(order) != MAX_ORDER:
        raise ValueError(
            "points=True encodes an order-29 point; pass order=29 "
            "(the default) or omit it"
        )
    # Ensure contiguous arrays for Rust FFI
    if not np.isscalar(lats):
        lats = np.ascontiguousarray(lats, dtype=np.float64)
        lons = np.ascontiguousarray(lons, dtype=np.float64)
    result = _rust_geo2mort(lats, lons, int(order), points)
    # Always return a contiguous uint64 ndarray. The scalar Rust path hands back
    # a Python int (which np would otherwise infer as int64), so coerce to keep
    # the dtype uint64 regardless of scalar-vs-array input or hemisphere.
    return np.ascontiguousarray(np.atleast_1d(result), dtype=np.uint64)


def orders_of(morton):
    """Per-element HEALPix order of packed morton words.

    Vectorized numpy decode of the 6-bit suffix (bits 5-0) per the spec page's
    suffix table (``docs/specification.md`` §1):

    * suffix ``0..=27`` — variable-length area element; the order *is* the
      suffix value (``0`` = base-cell-only).
    * suffix ``28..=47`` — order-28/29 area cells in parent-first preorder
      ``suffix = 28 + t28*5 + (t29 present ? t29 + 1 : 0)``: each ``t28`` owns
      a 5-block (the order-28 parent, then its four order-29 children), so
      ``(suffix - 28) % 5 == 0`` is order 28 and everything else is order 29.
    * suffix ``48..=63`` — order-29 **point** (max-encoded, no area claim —
      spec §4); points are order 29 by definition.

    Pure bit arithmetic — words are not validated (the empty sentinel ``0``
    decodes as order 0; use :func:`validate_morton` to reject malformed
    words). This is the per-element, mixed-order-native counterpart of
    :func:`infer_order_from_morton`.

    Parameters
    ----------
    morton : int or array-like
        Packed morton word(s) (``uint64``).

    Returns
    -------
    ndarray
        ``uint8`` order per element, 0-29 (scalar in -> length-1 ndarray,
        matching :func:`geo2mort`).
    """
    m = np.atleast_1d(np.asarray(morton, dtype=np.uint64))
    suffix = (m & np.uint64(0x3F)).astype(np.uint8)
    # 0..=27: order == suffix. 28..=47: order-28 on the 5-block parent slots,
    # order 29 otherwise. 48..=63: order-29 point.
    orders = suffix.copy()
    band = (suffix >= 28) & (suffix <= 47)
    orders[band] = np.where((suffix[band] - 28) % 5 == 0, 28, 29)
    orders[suffix >= 48] = 29
    return orders


def is_point(morton):
    """Per-element point-kind predicate for packed morton words.

    Kind is carried by the encoding itself (spec §4): suffix ``0..=47``
    decodes as an **area** word, suffix ``48..=63`` as an order-29 **point**
    (a location with no area claim — ``docs/specification.md`` §1 suffix
    table). Pure bit arithmetic; words are not validated (see
    :func:`validate_morton`).

    Parameters
    ----------
    morton : int or array-like
        Packed morton word(s) (``uint64``).

    Returns
    -------
    ndarray
        ``bool`` per element, True for point words (scalar in -> length-1
        ndarray, matching :func:`geo2mort`).
    """
    m = np.atleast_1d(np.asarray(morton, dtype=np.uint64))
    return (m & np.uint64(0x3F)) >= np.uint64(48)


def infer_order_from_morton(morton):
    """Infer the single HEALPix order of packed morton word(s).

    Decodes through the packed-u64 kernel (issue #48): the order is carried in
    the word's suffix, not in any decimal-digit count. The return is one
    scalar order, so array input must be uniform-order; mixed-order input
    raises, naming the distinct orders (issue #116 — previously the first
    element's order was returned silently). For per-element orders of a mixed
    array use :func:`orders_of`.

    Parameters
    ----------
    morton : int or array-like
        Packed morton word(s), all at one order.

    Returns
    -------
    int
        The HEALPix order.

    Raises
    ------
    ValueError
        If the words are at mixed orders.
    """
    m = np.atleast_1d(np.asarray(morton, dtype=np.uint64))
    _, depths = _rust_mort2nested(np.ascontiguousarray(m))
    distinct = np.unique(depths)
    if distinct.size > 1:
        raise ValueError(
            f"Mixed orders in morton array: {[int(d) for d in distinct]}; "
            "use orders_of for per-element orders"
        )
    return int(depths[0])


def validate_morton(morton, order=None):
    """Validate that a packed morton word is well-formed.

    The kernel decode rejects the empty sentinel (0) and any word with an
    invalid base-cell prefix; this also checks the decoded order matches
    ``order`` when one is supplied.

    Parameters
    ----------
    morton : int
        Packed morton word to validate.
    order : int, optional
        Expected HEALPix order. If None, no order check is made.

    Returns
    -------
    bool
        True if the word is a valid morton word.

    Raises
    ------
    ValueError
        If the word does not decode or its order disagrees with ``order``.
    """
    m = np.atleast_1d(np.asarray(morton, dtype=np.uint64))
    # The kernel raises ValueError on the empty sentinel / an invalid prefix.
    _, depths = _rust_mort2nested(np.ascontiguousarray(m))
    decoded_order = int(depths[0])
    if order is not None and decoded_order != order:
        raise ValueError(
            f"Morton word decodes to order {decoded_order}, expected {order}"
        )
    return True


def mort2norm(morton):
    """Convert morton index back to normalized address and parent cell

    Parameters
    ----------
    morton : int or array-like
        Packed morton word(s) (``uint64``; base cells 7-11 set bit 63).

    Returns
    -------
    normed : int or array
        Normalized HEALPix address
    parent : int or array
        Parent base cell (0-11)
    order : int or array
        HEALPix order inferred from morton index

    Notes
    -----
    Empty input returns two empty ``int64`` arrays and ``order == 0``.
    """
    morton = np.atleast_1d(np.asarray(morton, dtype=np.uint64))
    is_scalar = len(morton) == 1

    # Empty input: nothing to decode. Return empty int64 arrays (matching the
    # array-path dtype) and order 0.
    if morton.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy(), 0

    # The packed-u64 kernel decodes each word to (nested, depth); the depth is
    # the HEALPix order (no decimal-digit scan). Reject mixed orders: the
    # return contract is a single scalar order (the geo kernels above this
    # dispatch group-by-order and never hit this — issue #116).
    nested, depths = _rust_mort2nested(np.ascontiguousarray(morton))
    if np.any(depths != depths[0]):
        raise ValueError(
            f"Mixed orders in morton array: {sorted(set(int(d) for d in depths))}; "
            "use orders_of for per-element orders"
        )

    order = int(depths[0])
    # nested ids are HEALPix cell ids (<< 2^58 for order <= 29), so int64 is safe
    # arithmetic here and keeps normed/parent signed for downstream callers.
    nested = nested.astype(np.int64)
    nside_sq = np.int64(1) << np.int64(2 * order)
    parent = nested // nside_sq
    normed = nested % nside_sq

    if is_scalar:
        return normed[0], parent[0], order
    return normed, parent, order


def norm2uniq(normed, parent, order=18):
    """Convert normalized address and parent to UNIQ encoding

    Parameters
    ----------
    normed : int or array
        Normalized HEALPix address
    parent : int or array
        Parent base cell (0-11)
    order : int
        HEALPix order

    Returns
    -------
    uniq : int or array
        UNIQ encoded pixel index
    """
    nside = 2**order
    N_pix = nside**2

    # Convert normalized address back to nest index
    nest = normed + (parent * N_pix)

    # Convert to UNIQ
    uniq = 4 * N_pix + nest

    return uniq


def uniq2geo(uniq, order=18):
    """Convert UNIQ encoding to lat/lon of pixel center

    Parameters
    ----------
    uniq : int or array
        UNIQ encoded pixel
    order : int
        HEALPix order

    Returns
    -------
    lat : float or array
        Latitude in degrees
    lon : float or array
        Longitude in degrees
    """
    nside = 2**order

    # Extract nest index from UNIQ
    nest = uniq - 4 * (nside**2)

    # Get pixel center coordinates
    lon, lat = hp.pix2ang(order, nest)

    return lat, lon


def mort2geo(morton):
    """Convert morton index to lat/lon of pixel center

    This is the inverse of geo2mort, returning the center coordinates
    of the HEALPix cell identified by the morton index.

    Mixed-order arrays are supported (issue #116): elements are grouped by
    order (:func:`orders_of`), each group runs the uniform kernel, and the
    results scatter back to input positions. Point words (spec §4) are order
    29 by definition and group with order 29 — a point's location is exactly
    what mort2geo returns.

    Parameters
    ----------
    morton : int or array-like
        Morton index (mixed orders allowed)

    Returns
    -------
    lat : float or array
        Latitude in degrees
    lon : float or array
        Longitude in degrees
    """
    # Handle scalar vs array input to match geo2mort behavior
    input_is_scalar = np.isscalar(morton)

    # Group-by-order dispatch for mixed-order input (issue #116).
    if not input_is_scalar:
        words = np.atleast_1d(np.asarray(morton, dtype=np.uint64))
        orders = orders_of(words)
        unique_orders = np.unique(orders)
        if unique_orders.size > 1:
            lat = np.empty(words.size, dtype=np.float64)
            lon = np.empty(words.size, dtype=np.float64)
            for order in unique_orders:
                mask = orders == order
                lat[mask], lon[mask] = mort2geo(words[mask])
            return lat, lon

    # Decode morton to normalized address and parent
    normed, parent, order = mort2norm(morton)

    # Convert to UNIQ
    uniq = norm2uniq(normed, parent, order)

    # Convert to lat/lon
    lat, lon = uniq2geo(uniq, order)

    # Return array to match geo2mort behavior
    if input_is_scalar:
        return np.array([lat]), np.array([lon])
    return lat, lon


def mort2bbox(morton):
    """Convert morton index to bounding box of the pixel

    For pixels touching the antimeridian, vertex longitudes at ±180° are
    normalized to use consistent representation based on hemisphere voting,
    preventing bbox misinterpretation as spanning the entire globe.

    Mixed-order arrays are supported (issue #116): elements are grouped by
    order (:func:`orders_of`), each group runs the uniform kernel, and the
    results scatter back to input positions. Point words raise: a point has
    no area claim (spec §1/§4), so it has no bounding box.

    Parameters
    ----------
    morton : int or array-like
        Morton index (mixed orders allowed; area words only)

    Returns
    -------
    bbox : dict or list of dicts
        Bounding box in format suitable for STAC/CMR:
        {"west": min_lon, "south": min_lat, "east": max_lon, "north": max_lat}

    Raises
    ------
    ValueError
        If any word is an order-29 point (no area claim — spec §4).
    """
    morton = np.atleast_1d(morton)
    is_scalar = len(morton) == 1

    words = np.asarray(morton, dtype=np.uint64)
    if words.size and np.any(is_point(words)):
        raise ValueError(
            "mort2bbox: point words (suffix 48..=63) carry no area claim "
            "(spec §1/§4) and have no bounding box; coarsen to an area cell "
            "with clip2order first"
        )
    # Group-by-order dispatch for mixed-order input (issue #116).
    orders = orders_of(words)
    unique_orders = np.unique(orders)
    if unique_orders.size > 1:
        bboxes = [None] * words.size
        for order in unique_orders:
            (idx,) = np.nonzero(orders == order)
            group = mort2bbox(words[idx])
            if idx.size == 1:
                bboxes[idx[0]] = group  # length-1 call returns the bare dict
            else:
                for i, bbox in zip(idx, group):
                    bboxes[i] = bbox
        return bboxes

    # First get the pixel center
    normed, parent, order = mort2norm(morton)
    uniq = norm2uniq(normed, parent, order)

    nside = 2**order
    nest = uniq - 4 * (nside**2)

    # Get pixel boundaries: (N, 3, 4) — cell in axis 0, xyz in axis 1, the 4
    # corners in axis 2.  A single cell comes back 2-D (3, 4); promote it.
    boundaries = hp.boundaries(order, nest)
    if boundaries.ndim == 2:
        boundaries = boundaries[np.newaxis, ...]
    n = len(morton)

    # One batched vec2ang over every cell's corners (one Rust round-trip instead
    # of one per cell), then reshape to (N, 4).
    verts = np.transpose(boundaries, (0, 2, 1)).reshape(-1, 3)
    theta, phi = hp.vec2ang(verts)
    lats_all = (90 - np.degrees(theta)).reshape(n, 4)
    lons_all = np.degrees(phi)
    lons_all = np.where(lons_all > 180, lons_all - 360, lons_all).reshape(n, 4)

    bboxes = []
    for i in range(n):
        lats = lats_all[i]
        lons = lons_all[i]

        # Normalize antimeridian representation
        # Check if bbox touches antimeridian with mixed ±180°
        ANTIMERIDIAN_TOLERANCE = 1e-6
        on_antimeridian = np.abs(np.abs(lons) - 180.0) < ANTIMERIDIAN_TOLERANCE

        if np.any(on_antimeridian) and (np.max(lons) - np.min(lons)) > 180:
            # Count vertices in each hemisphere (excluding those on antimeridian)
            non_antimeridian = ~on_antimeridian
            if np.any(non_antimeridian):
                western_count = np.sum(lons[non_antimeridian] < -0.1)
                eastern_count = np.sum(lons[non_antimeridian] > 0.1)

                # Determine target longitude for antimeridian vertices
                if western_count > eastern_count:
                    target_lon = -180.0
                elif eastern_count > western_count:
                    target_lon = 180.0
                else:
                    # Use median of non-antimeridian lons
                    median_lon = np.median(lons[non_antimeridian])
                    target_lon = -180.0 if median_lon < 0 else 180.0

                # Normalize antimeridian vertices
                lons = lons.copy()
                lons[on_antimeridian] = target_lon

        # Create bounding box
        bbox = {
            "west": float(np.min(lons)),
            "south": float(np.min(lats)),
            "east": float(np.max(lons)),
            "north": float(np.max(lats))
        }
        bboxes.append(bbox)

    if is_scalar:
        return bboxes[0]
    return bboxes


def _normalize_antimeridian_polygon(vertices):
    """
    Fix polygons that touch (but don't cross) the antimeridian.

    When a polygon touches the antimeridian, vertices at ±180° should be
    normalized to match the hemisphere containing most other vertices.
    This prevents spatial libraries from incorrectly interpreting the
    polygon as spanning the entire globe.

    Parameters
    ----------
    vertices : list of [lat, lon] lists
        Polygon vertices in [[lat, lon], ...] format

    Returns
    -------
    list of [lat, lon] lists
        Normalized vertices with consistent antimeridian representation
    """
    # Extract longitudes (excluding closing point if polygon is closed)
    lons = np.array([v[1] for v in vertices[:-1]]) if vertices[0] == vertices[-1] else np.array([v[1] for v in vertices])

    # Check if this looks like an antimeridian issue
    lon_span = lons.max() - lons.min()

    if lon_span <= 180:
        # No issue - polygon doesn't span more than a hemisphere
        return vertices

    # Separate vertices into three groups:
    # 1. Western hemisphere (lon < -0.1, to avoid floating point issues near 0)
    # 2. Eastern hemisphere (lon > 0.1)
    # 3. Antimeridian (lon very close to ±180)

    ANTIMERIDIAN_TOLERANCE = 1e-6
    western = np.sum(lons < -0.1)
    eastern = np.sum(lons > 0.1)
    on_antimeridian = np.sum(np.abs(np.abs(lons) - 180.0) < ANTIMERIDIAN_TOLERANCE)

    # Determine target normalization based on majority hemisphere
    if western > eastern:
        # Majority in western hemisphere → normalize to -180
        target_lon = -180.0
    elif eastern > western:
        # Majority in eastern hemisphere → normalize to +180
        target_lon = 180.0
    else:
        # Equal split or all on antimeridian - use median of non-antimeridian vertices
        non_antimeridian_lons = lons[np.abs(np.abs(lons) - 180.0) >= ANTIMERIDIAN_TOLERANCE]
        if len(non_antimeridian_lons) > 0:
            median_lon = np.median(non_antimeridian_lons)
            target_lon = -180.0 if median_lon < 0 else 180.0
        else:
            # All vertices on antimeridian (degenerate case)
            return vertices

    # Apply normalization
    normalized = []
    for lat, lon in vertices:
        if abs(abs(lon) - 180.0) < ANTIMERIDIAN_TOLERANCE:
            # This vertex is on the antimeridian - normalize it
            normalized.append([lat, target_lon])
        else:
            # Keep as-is
            normalized.append([lat, lon])

    return normalized


def mort2polygon(morton, step=1):
    """Convert morton index to polygon representation.

    Parameters
    ----------
    morton : int or array-like
        Morton index.
    step : int, optional
        Points per side for the cell boundary (default 1 = 4 corners).
        Use step=32 for 128 boundary points that accurately trace
        curved cell edges, important for polar cells where 4-corner
        polygons poorly approximate the true HEALPix boundary.

    Returns
    -------
    polygon : list or list of lists
        Polygon coordinates as [[lat, lon], ...] in standard geographic order.
        The polygon is closed (first point repeated at end).

        **Note**: Returns [lat, lon] pairs, NOT [lon, lat]. This is the standard
        geographic coordinate order used by most spatial analysis libraries.

    Raises
    ------
    ValueError
        If any word is an order-29 point (no area claim — spec §4).

    Notes
    -----
    Polygons that touch the antimeridian (±180° longitude) are automatically
    normalized to use consistent longitude representation (-180 or +180) based
    on which hemisphere contains the majority of vertices. This prevents spatial
    libraries from misinterpreting touching polygons as crossing polygons.

    Mixed-order arrays are supported (issue #116): elements are grouped by
    order (:func:`orders_of`), each group runs the uniform kernel, and the
    results scatter back to input positions (rings are 4*step+1 vertices at
    every order, so mixed orders do not change the output shape). Point words
    raise: a point has no area claim (spec §1/§4), so it has no polygon.
    """
    morton = np.atleast_1d(morton)
    is_scalar = len(morton) == 1

    words = np.asarray(morton, dtype=np.uint64)
    if words.size and np.any(is_point(words)):
        raise ValueError(
            "mort2polygon: point words (suffix 48..=63) carry no area claim "
            "(spec §1/§4) and have no polygon; coarsen to an area cell with "
            "clip2order first"
        )
    # Group-by-order dispatch for mixed-order input (issue #116).
    orders = orders_of(words)
    unique_orders = np.unique(orders)
    if unique_orders.size > 1:
        polygons = [None] * words.size
        for order in unique_orders:
            (idx,) = np.nonzero(orders == order)
            group = mort2polygon(words[idx], step=step)
            if idx.size == 1:
                polygons[idx[0]] = group  # length-1 call returns the bare ring
            else:
                for i, polygon in zip(idx, group):
                    polygons[i] = polygon
        return polygons

    # Get pixel information
    normed, parent, order = mort2norm(morton)
    uniq = norm2uniq(normed, parent, order)

    nside = 2**order
    nest = uniq - 4 * (nside**2)

    # Get pixel boundaries: (N, 3, 4*step) — cell in axis 0, xyz in axis 1, the
    # boundary points in axis 2.  A single cell comes back 2-D (3, ncols);
    # promote it.
    boundaries = hp.boundaries(order, nest, step=step)
    if boundaries.ndim == 2:
        boundaries = boundaries[np.newaxis, ...]
    n = len(morton)
    ncols = 4 * step

    # One batched vec2ang over every cell's boundary points (one Rust round-trip
    # instead of one per cell), then reshape to (N, ncols).
    verts = np.transpose(boundaries, (0, 2, 1)).reshape(-1, 3)
    theta, phi = hp.vec2ang(verts)
    lats_all = (90 - np.degrees(theta)).reshape(n, ncols)
    lons_all = np.degrees(phi)
    lons_all = np.where(lons_all > 180, lons_all - 360, lons_all).reshape(n, ncols)

    polygons = []
    for i in range(n):
        lats = lats_all[i]
        lons = lons_all[i]

        # Create polygon as list of [lat, lon] pairs (standard geographic order)
        # Close the polygon by repeating first point
        polygon = [[float(lats[j]), float(lons[j])] for j in range(len(lons))]
        polygon.append(polygon[0])  # Close the polygon

        # Normalize antimeridian representation to prevent misinterpretation
        polygon = _normalize_antimeridian_polygon(polygon)

        polygons.append(polygon)

    if is_scalar:
        return polygons[0]
    return polygons


def clip2order(clip_order, midx=None, print_factor=False):
    """Coarsen packed morton words to a lower resolution.

    Degrades each packed word to ``clip_order`` by coarsening it through the
    kernel (the inverse of refining): the base cell and the first ``clip_order``
    tuples are kept, finer detail is dropped, and the suffix is rewritten. Words
    already at or below ``clip_order`` are returned unchanged.

    Parameters
    ----------
    clip_order : int
        HEALPix order to degrade to.
    midx : array-like of int
        Packed morton words (see :func:`res2display` for approximate resolutions).
    print_factor : bool, optional
        If True, return the number of levels dropped from order 18
        (``18 - clip_order``) instead of clipping. Retained for backwards
        compatibility; the value is now a level count, not a decimal factor.

    Returns
    -------
    ndarray or int
        Coarsened packed words, or the level count when ``print_factor`` is True.
    """
    if print_factor:
        return 18 - clip_order

    midx = np.ascontiguousarray(np.asarray(midx, dtype=np.uint64).ravel())
    return _rustie.rust_mi_coarsen(midx, int(clip_order))


def generate_morton_children(parent_morton, target_order):
    """
    Generate all child morton indices at a target order.

    Parameters
    ----------
    parent_morton : int
        Parent packed morton word.
    target_order : int
        Target order for children (must be >= parent order)

    Returns
    -------
    children : ndarray
        Array of child packed morton words at target_order.
        If target_order equals parent_order, returns array with parent_morton.

    Notes
    -----
    Children are generated in HEALPix NESTED space — descending ``level_diff``
    orders multiplies the cell count by ``4**level_diff`` — then packed back to
    morton words via the kernel. If already at target_order, returns the parent
    itself.
    """
    # Decode the parent to its (nested, depth) via the packed kernel.
    parent_morton = np.uint64(parent_morton)
    nested, depths = _rust_mort2nested(
        np.ascontiguousarray(np.atleast_1d(parent_morton))
    )
    parent_order = int(depths[0])
    parent_nested = int(nested[0])

    if target_order < parent_order:
        raise ValueError(
            f"target_order ({target_order}) must be >= parent_order ({parent_order})"
        )

    if target_order == parent_order:
        return np.array([parent_morton], dtype=np.uint64)

    level_diff = target_order - parent_order
    # In NESTED space a cell's descendants at `target_order` are the contiguous
    # block `nested * 4**level_diff + [0 .. 4**level_diff)`.
    span = 4 ** level_diff
    child_nested = (parent_nested << (2 * level_diff)) + np.arange(
        span, dtype=np.uint64
    )
    depths = np.full(span, target_order, dtype=np.uint8)
    return _rust_nested2mort(np.ascontiguousarray(child_nested), depths)


def morton_buffer(morton_indices, k=1):
    """Compute the k-cell border around a set of morton indices.

    Returns only cells NOT in the input set (the expansion ring).
    User can union: ``np.union1d(morton_indices, border)``

    Parameters
    ----------
    morton_indices : array-like
        Morton indices, all at the same order.
    k : int, optional
        Border width in cells (default 1, 8-connected neighbors).
        k=1 gives the immediate ring, k=2 gives a 2-cell border, etc.

    Returns
    -------
    border : ndarray
        Sorted array of morton indices for the border cells.

    Raises
    ------
    ValueError
        If indices have mixed orders or k is out of range.
    """
    morton_indices = np.asarray(morton_indices, dtype=np.uint64)
    return _rustie.rust_morton_buffer(np.ascontiguousarray(morton_indices), k)


# Earth mean radius in meters (IUGG 2015 mean).
_EARTH_RADIUS_M = 6_371_008.7714


def morton_buffer_meters(morton_indices, width_m):
    """Approximate meter-width buffer around a set of morton cells.

    This is a convenience wrapper around :func:`morton_buffer` that picks
    ``k`` from the cells' HEALPix order so the resulting ring is roughly
    *width_m* meters wide. The input cells are assumed to all be at the same
    order.

    .. warning::
       **This is an approximate buffer.** The achieved width is rounded
       UP to the nearest whole HEALPix cell width — so the result always
       covers *at least* ``width_m`` meters, but may cover up to one cell
       width more. For order 18 cells (~30 m) the granularity is fine; at
       coarser orders it can be substantial. If you need a precise buffer,
       pick an order whose cell width is small relative to ``width_m`` and
       convert your input cells to that order first.

    The cell width used for the calculation is the HEALPix angular
    resolution ``sqrt(pi/3) / nside`` converted to meters via the Earth's
    mean radius (6,371,008.77 m).

    Parameters
    ----------
    morton_indices : array-like
        Morton indices, all at the same HEALPix order.
    width_m : float
        Desired buffer width in meters (must be > 0).

    Returns
    -------
    border : ndarray
        Sorted array of morton indices for the border cells (NOT including
        the input cells). Union with the input if you want the filled ring:
        ``np.union1d(morton_indices, border)``.

    Raises
    ------
    ValueError
        If ``width_m`` is non-positive, the input array is empty, or the
        cells are at mixed orders.

    Examples
    --------
    >>> import mortie, numpy as np
    >>> cells = mortie.linestring_coverage([10.0, 20.0], [30.0, 40.0], order=10)
    >>> border = mortie.morton_buffer_meters(cells, width_m=5000.0)
    >>> expanded = np.union1d(cells, border)
    """
    morton_indices = np.asarray(morton_indices, dtype=np.uint64)
    if morton_indices.size == 0:
        raise ValueError("morton_indices must be non-empty")
    if not (width_m > 0):
        raise ValueError("width_m must be positive")

    # Infer order from the first cell. rust_morton_buffer will itself reject
    # mixed-order inputs downstream.
    order = infer_order_from_morton(int(morton_indices.flat[0]))
    if order < 1:
        raise ValueError("Could not infer a valid order from the input cells")

    nside = 1 << order
    cell_width_m = _EARTH_RADIUS_M * np.sqrt(np.pi / 3.0) / nside

    # Round UP so the buffer covers AT LEAST the requested width.
    k = int(np.ceil(width_m / cell_width_m))
    if k < 1:
        k = 1

    return morton_buffer(morton_indices, k=k)


def mort2healpix(morton):
    """
    Convert morton index to HEALPix cell ID and order.

    Parameters
    ----------
    morton : int or array-like
        Morton index

    Returns
    -------
    cell_ids : int or ndarray
        HEALPix cell ID(s) in NESTED scheme
    order : int
        HEALPix order (resolution level)

    Examples
    --------
    >>> import mortie
    >>> m = mortie.geo2mort(-80.0, 120.0, order=6)[0]
    >>> cell_id, order = mort2healpix(m)
    >>> print(f"HEALPix cell {cell_id} at order {order}")
    HEALPix cell 37010 at order 6

    Notes
    -----
    The function converts morton indices to HEALPix NESTED scheme cell IDs.
    All input morton indices must be at the same order.
    """
    # Check if input is scalar before converting to array
    is_scalar = np.isscalar(morton)
    morton = np.atleast_1d(morton)

    # Get normalized morton and order
    normed, parent, order = mort2norm(morton)

    # Convert to UNIQ indexing
    uniq = norm2uniq(normed, parent, order)

    # Convert UNIQ to HEALPix NESTED cell ID
    # UNIQ = 4 * nside^2 + nest_index
    nside = 2**order
    cell_ids = uniq - 4 * (nside**2)

    # Ensure arrays for consistent handling
    cell_ids = np.atleast_1d(cell_ids).astype(np.int64)
    order = np.atleast_1d(order)

    if is_scalar:
        return int(cell_ids[0]), int(order[0])

    # For array input, return single order if all are the same
    order_val = int(order[0]) if len(np.unique(order)) == 1 else order
    return cell_ids, order_val

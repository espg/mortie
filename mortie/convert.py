"""Address-space conversions between geographic, morton, UNIQ and HEALPix.

The ``X2Y`` family: :func:`geo2mort` / :func:`mort2geo` and :func:`geo2uniq` /
:func:`uniq2geo` across the geographic boundary, :func:`norm2mort` /
:func:`mort2norm` and :func:`norm2uniq` / :func:`unique2parent` across the
normalized-address boundary, and :func:`mort2healpix` out to NESTED cell ids.
:func:`mort2bbox` and :func:`mort2polygon` belong here too: from the caller's
side they turn a word into a bounding box or a ring, which is a conversion --
even though their kernels live in ``src_rust/src/cell_geom.rs`` rather than in
``geo2mort.rs`` / ``morton.rs`` with the rest of this module's twins.

Split out of ``mortie.tools`` (issue #159) so the Python surface mirrors the
Rust tree's own decomposition. The names stay flat on the package
(``mortie.geo2mort``, ``mortie.mort2polygon``): this module is where they live,
not how they are spelled.
"""

import numpy as np

from . import _healpix as hp
from . import _rustie
from ._validate import _as_i64, _as_u64
from .orders import (
    MAX_ORDER,
    _rust_mort2nested,
    _rust_nested2mort,
    orders_of,
    orders_of_uniq,
)

# Rust-native geo2mort (uses healpix crate, no Python HEALPix backend)
_rust_geo2mort = _rustie.rust_geo2mort

#: The two latitude conventions of the geographic boundary (issue #186).
_LATITUDE_CONVENTIONS = ("authalic", "geodetic-spherical")


def _check_latitude(latitude):
    """Validate a ``latitude=`` convention string.

    Parameters
    ----------
    latitude : str
        Candidate convention name.

    Raises
    ------
    ValueError
        If *latitude* is not one of ``"authalic"`` /
        ``"geodetic-spherical"``.
    """
    if latitude not in _LATITUDE_CONVENTIONS:
        raise ValueError(
            f'latitude must be "authalic" or "geodetic-spherical", '
            f"got {latitude!r}"
        )


def geodetic_to_authalic(lats):
    """Convert WGS84 geodetic latitude(s) to authalic latitude (degrees).

    The forward half of the issue #186 convention change: authalic latitude
    substituted into the spherical HEALPix formulas makes mortie's cells
    equal-area on the WGS84 ellipsoid by construction.  The conversion is a
    5-harmonic trigonometric series with coefficients derived from the pinned
    WGS84 constants (a = 6378137, 1/f = 298.257223563); it is exact to
    <= 1e-13 rad (~0.6 um on the ground).  The equator and poles are fixed
    points; the divergence peaks in the +/-45-degree band, where the authalic
    latitude is ~0.12830 degrees (~14.26 km of meridian arc) closer to the
    equator.  Longitude is unaffected by the convention, so there is no
    ``lons`` argument.

    Every mortie entry point applies this conversion internally under its
    default ``latitude="authalic"``; this function is the standalone spelling
    for callers who need the raw latitude mapping (e.g. to reproduce a
    binning decision or to label an external dataset).

    **Batch vectorized**: array in, array out, elementwise.

    Parameters
    ----------
    lats : float or array-like
        Geodetic latitude(s) in degrees.

    Returns
    -------
    float or numpy.ndarray
        Authalic latitude(s) in degrees (scalar in -> scalar out).

    See Also
    --------
    authalic_to_geodetic : the exact inverse.
    """
    if np.isscalar(lats):
        return _rustie.rust_geodetic_to_authalic(float(lats))
    # Flatten-and-reshape rather than passing N-d through: the Rust bridge is
    # 1-D, and its scalar fast path would collapse a 1-element array to 0-d.
    arr = np.ascontiguousarray(lats, dtype=np.float64)
    out = np.asarray(
        _rustie.rust_geodetic_to_authalic(np.ascontiguousarray(arr.ravel())),
        dtype=np.float64,
    )
    return np.atleast_1d(out).reshape(arr.shape)


def authalic_to_geodetic(lats):
    """Convert authalic latitude(s) back to WGS84 geodetic latitude (degrees).

    The inverse of :func:`geodetic_to_authalic`, exact to the same
    <= 1e-13 rad series bound — see there for the convention background
    (issue #186).

    **Batch vectorized**: array in, array out, elementwise.

    Parameters
    ----------
    lats : float or array-like
        Authalic latitude(s) in degrees.

    Returns
    -------
    float or numpy.ndarray
        Geodetic latitude(s) in degrees (scalar in -> scalar out).

    See Also
    --------
    geodetic_to_authalic : the forward direction.
    """
    if np.isscalar(lats):
        return _rustie.rust_authalic_to_geodetic(float(lats))
    # Flatten-and-reshape, exactly as geodetic_to_authalic does.
    arr = np.ascontiguousarray(lats, dtype=np.float64)
    out = np.asarray(
        _rustie.rust_authalic_to_geodetic(np.ascontiguousarray(arr.ravel())),
        dtype=np.float64,
    )
    return np.atleast_1d(out).reshape(arr.shape)


def unique2parent(unique):
    """Parent HEALPix base cell (0-11) of UNIQ encoded cell(s).

    Mixed-resolution input is supported (issue #136): the order is decoded per
    element by :func:`orders_of_uniq` and the arithmetic stays elementwise, so
    each cell is reduced against its own order. This function previously
    collapsed those per-element orders to a scalar and raised
    ``NotImplementedError`` on anything mixed.

    **Batch vectorized**: array in, array out, elementwise.

    Parameters
    ----------
    unique : int or array-like
        UNIQ encoded cell number(s); orders may be mixed.

    Returns
    -------
    numpy.int64 or ndarray
        Parent base cell, 0-11 (scalar in -> ``numpy.int64`` out).  UNIQ ids
        are a different encoding, deliberately outside the mortie-*word*
        ``uint64`` contract (issue #187), so this is **not** a Python ``int``:
        ``isinstance(p, int)`` is ``False``.  Use ``int(p)`` if you need one.

    Raises
    ------
    ValueError
        If a value is not a valid UNIQ cell number for orders 0-``MAX_ORDER``.
        A float-typed ``unique`` or a value past int64 is refused by name
        (issue #194, phase 5), never silently cast.
    """
    is_scalar = np.ndim(unique) == 0
    u = _as_i64(unique, "unique")
    # int64, not the public uint8: the shifts below would otherwise run in
    # uint8 and wrap (the same trap order2res documents for `orders_of`).
    orders = orders_of_uniq(u).astype(np.int64)

    # nested = uniq - 4 * 4**order, and the base cell is nested // 4**order.
    shift = 2 * orders
    parent = (u - (np.int64(1) << (shift + np.int64(2)))) >> shift

    return parent[0] if is_scalar else parent


# Public API - uses Rust (the packed-u64 kernel)
def norm2mort(normed, parent, order):
    """Convert a normalized HEALPix address + base cell to a packed morton word.

    The exact inverse of :func:`mort2norm`: ``mort2norm(norm2mort(n, p, o))``
    returns ``(n, p, o)``. Born order-29-native (issue #48) — there is no order
    cap beyond the kernel's ``MAX_ORDER`` of 29. The returned ``uint64`` is the
    packed ``decimal_morton`` word (issue #58; the prefix is ``base+1``, so bit 63
    is set — a large unsigned value — for base cells 7-11), not the retired
    decimal encoding.

    **Batch vectorized**: array in, array out, elementwise (one shared
    ``order``).  The two operands broadcast against each other, and the form
    follows numpy semantics literally (issue #187): a scalar out only when
    **both** inputs are scalars.  A **length-1 array used to squeeze to a
    scalar** — the opposite of the array-in/array-out rule the polymorphic
    API is built on, and a silent one, since the caller who passed an array
    got back something that could not be indexed.  It now keeps its shape.

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
        Packed morton word(s) — a ``uint64`` scalar when both ``normed`` and
        ``parent`` are scalars, a 1-D array (of the broadcast length, length 1
        included) whenever either is an array.

    Raises
    ------
    ValueError
        If ``normed`` or ``parent`` is float-typed or negative — refused by
        name (issue #194, phase 5) rather than silently cast into a
        different, possibly valid, word.
    """
    # Rank of the *inputs*, read before coercion: it is what selects the form,
    # so a length-1 array stays an array (issue #187).
    is_scalar = np.ndim(normed) == 0 and np.ndim(parent) == 0
    normed = _as_u64(normed, "normed")
    parent = _as_u64(parent, "parent")
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


def _encoder_orders(order, n):
    """Validate a UNIQ encoder's scalar-or-per-element ``order`` argument.

    Parameters
    ----------
    order : int or array-like
        HEALPix order(s), 0-``MAX_ORDER``. A scalar applies to every element;
        an array carries one order per input element.
    n : int
        Number of elements being encoded.

    Returns
    -------
    int or ndarray
        The scalar order as an ``int``, or an ``int64`` array of length ``n``.

    Raises
    ------
    ValueError
        If an order lies outside 0-``MAX_ORDER``, or an order array is not
        1-D of length ``n``.
    """
    if np.ndim(order) == 0:
        order = int(order)
        if not 0 <= order <= MAX_ORDER:
            raise ValueError(
                f"order must be between 0 and {MAX_ORDER}, got {order!r}")
        return order

    orders = np.asarray(order, dtype=np.int64)
    if orders.ndim != 1 or orders.size != n:
        raise ValueError(
            f"order array must be 1-D with one entry per input element ({n}), "
            f"got shape {orders.shape}")
    if orders.size and (orders.min() < 0 or orders.max() > MAX_ORDER):
        raise ValueError(f"order must be between 0 and {MAX_ORDER}")
    return orders


def geo2uniq(lats, lons, order=MAX_ORDER, *, latitude="authalic"):
    """Calculate UNIQ cell numbers for lat/lon.

    ``order`` may be a scalar — one resolution for the whole input — or an
    array carrying one order per element, which produces a mixed-resolution
    UNIQ array (issue #136). ``hp.ang2pix`` takes a single depth, so the array
    form groups elements by order and scatters the results back to their input
    positions, the dispatch pattern issue #116 introduced in :func:`mort2geo`.

    UNIQ carries **no point/area kind**. The encoding is ``4 * 4**order +
    nested`` — an order and a cell index, with no kind bit and no spare state
    for one. Point-vs-area is a ``decimal_morton`` packed-word concept, carried
    by the suffix range (``0..=47`` area, ``48..=63`` point;
    ``docs/specification.md`` §1). So the default ``order=MAX_ORDER`` yields the
    max-resolution **area** cell containing each coordinate, *not* a point.
    Where point semantics are wanted from lat/lon the API already has them:
    :func:`geo2mort` (a bare ``geo2mort(lats, lons)`` returns order-29
    ``Kind::Point`` words) and
    :meth:`~mortie.morton_index.MortonIndexArray.from_latlon` with
    ``points=True``.

    **Batch vectorized**: array in, array out, elementwise.

    Parameters
    ----------
    lats : float or array-like
        Latitude(s) in degrees.
    lons : float or array-like
        Longitude(s) in degrees.
    order : int or array-like, optional
        HEALPix order(s), 0-``MAX_ORDER``. Scalar, or one per input element.
        Defaults to ``MAX_ORDER`` (29) — the finest order the kernel reaches.
        It defaulted to 18 before issue #136, a leftover from the retired
        decimal encoding's int64 cap.
    latitude : str, optional
        Latitude convention of the input (issue #186): ``"authalic"``
        (default; geodetic latitudes are converted so cells are equal-area on
        the WGS84 ellipsoid) or ``"geodetic-spherical"`` (legacy: geodetic
        latitude fed to the spherical kernel as-is).  Cell ids under the two
        conventions are non-corresponding partitions — never mix them.

    Returns
    -------
    numpy.int64 or ndarray
        UNIQ encoded cell number(s) (scalar in with a scalar order ->
        ``numpy.int64`` out).  UNIQ ids are a different encoding, deliberately
        outside the mortie-*word* ``uint64`` contract (issue #187), so this is
        **not** a Python ``int``: ``isinstance(u, int)`` is ``False``.  Use
        ``int(u)`` if you need one -- :func:`unique2parent` returns
        ``numpy.int64`` too, while :func:`norm2uniq` returns a Python ``int``.

    Raises
    ------
    ValueError
        If an order lies outside 0-``MAX_ORDER``, an order array's length
        does not match the input, or *latitude* is not a valid convention.
    """
    _check_latitude(latitude)
    n = np.broadcast(np.asarray(lats), np.asarray(lons)).size
    order = _encoder_orders(order, n)

    if np.ndim(order) == 0:
        if latitude == "authalic":
            lats = geodetic_to_authalic(lats)
        nside = 2**order
        nest = hp.ang2pix(order, lons, lats)
        return 4 * (nside**2) + nest

    # Per-element orders: group by order, run the uniform kernel per group.
    # Each recursion takes the scalar-order branch above, which is where the
    # latitude conversion happens — exactly once per element.
    lats, lons = np.broadcast_arrays(np.asarray(lats, dtype=np.float64),
                                     np.asarray(lons, dtype=np.float64))
    lats = np.atleast_1d(lats)
    lons = np.atleast_1d(lons)
    uniq = np.empty(n, dtype=np.int64)
    for one_order in np.unique(order):
        mask = order == one_order
        uniq[mask] = geo2uniq(lats[mask], lons[mask], int(one_order),
                              latitude=latitude)
    return uniq


def geo2mort(lats, lons, order=None, points=None, *, latitude="authalic"):
    """Compute morton indices from geographic coordinates.

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

    **Batch vectorized**: array in, array out, elementwise (one shared
    ``order``).

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
    latitude : str, optional
        Latitude convention of the input (issue #186): ``"authalic"``
        (default; geodetic latitudes are converted so cells are equal-area on
        the WGS84 ellipsoid) or ``"geodetic-spherical"`` (legacy: geodetic
        latitude fed to the spherical kernel as-is).  Cell ids under the two
        conventions are non-corresponding partitions — never mix them.

    Returns
    -------
    ndarray
        Packed ``uint64`` morton word(s), same shape family as the input
        (scalar in -> length-1 ndarray).

    Raises
    ------
    ValueError
        If ``points=True`` is combined with an explicit ``order != 29``, or
        *latitude* is not a valid convention.
    """
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
    result = _rust_geo2mort(lats, lons, int(order), points, latitude)
    # Always return a contiguous uint64 ndarray. The scalar Rust path hands back
    # a Python int (which np would otherwise infer as int64), so coerce to keep
    # the dtype uint64 regardless of scalar-vs-array input or hemisphere.
    return np.ascontiguousarray(np.atleast_1d(result), dtype=np.uint64)


def mort2norm(morton):
    """Convert morton index back to normalized address and parent cell.

    **Batch vectorized**: array in, arrays out, elementwise — but the words
    must share one order, since the returned order is a single scalar.  The
    form follows the **input rank** (issue #187): scalars out only for a
    scalar or 0-d word, so a length-1 array comes back as length-1 arrays.  It
    **used to squeeze** any length-1 input, which broke the form symmetry with
    :func:`norm2mort` (fixed in the same issue) that the "exact inverse"
    contract above rests on.

    Parameters
    ----------
    morton : int or array-like
        Packed morton word(s) (``uint64``; base cells 7-11 set bit 63).

    Returns
    -------
    normed : int or ndarray
        Normalized HEALPix address — an ``int64`` scalar when ``morton`` is a
        scalar or 0-d, a 1-D array (length 1 included) otherwise.
    parent : int or ndarray
        Parent base cell (0-11), in the form ``normed`` takes.
    order : int
        HEALPix order inferred from the morton word(s); always a python
        ``int``, since the words must share one order.

    Raises
    ------
    ValueError
        If the words are at mixed orders — the return contract carries a single
        scalar order, so use :func:`orders_of` for per-element orders.

    See Also
    --------
    norm2mort : the inverse; it follows the same input-rank form rule.

    Notes
    -----
    Empty input returns two empty ``int64`` arrays and ``order == 0``.
    """
    # Rank of the *input*, read before coercion: it is what selects the form,
    # so a length-1 array stays an array (issue #187), the same rule
    # norm2mort follows -- the pair is documented as exact inverses, and a
    # squeeze on one side alone made the round trip lose its shape.
    is_scalar = np.ndim(morton) == 0
    morton = _as_u64(morton, "morton")

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


def norm2uniq(normed, parent, order=MAX_ORDER):
    """Convert normalized address and parent to UNIQ encoding.

    ``order`` may be a scalar — one resolution for the whole input — or an
    array carrying one order per element, which produces a mixed-resolution
    UNIQ array (issue #136). Unlike :func:`geo2uniq` this needs no
    group-by-order dispatch: the body is integer arithmetic and broadcasts
    elementwise against an array of orders as it stands.

    UNIQ carries **no point/area kind** — ``4 * 4**order + nested`` is an order
    and a cell index, with no kind bit (see :func:`geo2uniq` for the full note
    and the point-capable alternatives). An order-29 result is the
    max-resolution **area** cell, not a point.

    **Batch vectorized**: array in, array out, elementwise.

    Parameters
    ----------
    normed : int or array-like
        Normalized HEALPix address.
    parent : int or array-like
        Parent base cell (0-11).
    order : int or array-like, optional
        HEALPix order(s), 0-``MAX_ORDER``. Scalar, or one per input element.
        Defaults to ``MAX_ORDER`` (29); it defaulted to 18 before issue #136,
        a leftover from the retired decimal encoding's int64 cap.

    Returns
    -------
    int or ndarray
        UNIQ encoded pixel index/indices.

    Raises
    ------
    ValueError
        If an order lies outside 0-``MAX_ORDER``, or an order array's length
        does not match the input, or ``normed`` / ``parent`` is float-typed
        or negative -- refused by name (issue #194, phase 5) rather than
        silently cast into a different, possibly valid, UNIQ id.
    """
    # Validated, not rebound: :func:`norm2mort` takes the same unsigned intake
    # on the same two operands, and this is the documented producer of the ids
    # its two consumers now refuse floats for.  The arithmetic below keeps the
    # caller's own dtypes and scalar/array form, so valid input answers exactly
    # as before -- `norm2uniq(-3, 0, 4)` used to answer 1021, a real order-3
    # cell in base 11, with no error at any point downstream.
    _as_u64(normed, "normed")
    _as_u64(parent, "parent")
    bcast = np.broadcast(np.asarray(normed), np.asarray(parent))
    order = _encoder_orders(order, bcast.size)
    if isinstance(order, np.ndarray) and len(bcast.shape) > 1:
        # `_encoder_orders` validates the order array as flat 1-D of length
        # `size`, so a (2, 1) input with a length-2 order would outer-broadcast
        # to (2, 2) -- four results from a two-element input, silently. The
        # scalar-order path handles the same input correctly, so refuse rather
        # than let the two paths disagree.
        raise ValueError(
            f"a per-element order array requires 1-D input; normed/parent "
            f"broadcast to {bcast.shape}")

    nside = 2**order
    N_pix = nside**2

    if isinstance(N_pix, np.ndarray):
        # `_encoder_orders` yields int64, and uint64 x int64 has no common
        # integer type -- NEP 50 promotes it to float64, which is silently
        # lossy above 2**53 (order >= 25) and returned a *different* UNIQ cell
        # with no error raised. Force the array path into uint64 so it matches
        # the scalar path bit for bit. The scalar path is immune because a
        # Python int is weakly typed and stays in uint64.
        N_pix = N_pix.astype(np.uint64)
        nest = np.asarray(normed, dtype=np.uint64) + (
            np.asarray(parent, dtype=np.uint64) * N_pix
        )
        return np.uint64(4) * N_pix + nest

    # Convert normalized address back to nest index
    nest = normed + (parent * N_pix)

    # Convert to UNIQ
    uniq = 4 * N_pix + nest

    return uniq


def uniq2geo(uniq, *, latitude="authalic"):
    """Convert UNIQ encoding to lat/lon of pixel center.

    The order is decoded per element from the UNIQ value itself
    (:func:`orders_of_uniq`), so mixed-resolution input is handled natively and
    there is no ``order`` argument to get wrong. The parameter was removed in
    issue #136: it defaulted to 18 and was never cross-checked, so a caller who
    passed the wrong order — or simply took the default — got plausible but
    wrong coordinates with no error raised.

    Elements are grouped by decoded order and each group runs the uniform
    ``pix2ang`` kernel, mirroring the group-by-order dispatch :func:`mort2geo`
    uses for mixed-order morton words (issue #116).

    **Batch vectorized**: array in, arrays out, elementwise.

    Parameters
    ----------
    uniq : int or array-like
        UNIQ encoded pixel(s); orders may be mixed.
    latitude : str, optional
        Latitude convention of the **returned** coordinates (issue #186):
        ``"authalic"`` (default; the kernel-frame cell-centre latitude is
        converted back to WGS84 geodetic) or ``"geodetic-spherical"``
        (legacy: the spherical latitude is returned as-is).  Pass the same
        convention the cells were encoded under.

    Returns
    -------
    lat : float or ndarray
        Latitude in degrees of the cell centre.
    lon : float or ndarray
        Longitude in degrees of the cell centre.

    Raises
    ------
    ValueError
        If a value is not a valid UNIQ cell number for orders 0-``MAX_ORDER``,
        or *latitude* is not a valid convention.  A float-typed ``uniq`` or a
        value past int64 is refused by name (issue #194, phase 5), never
        silently cast.
    """
    _check_latitude(latitude)
    is_scalar = np.ndim(uniq) == 0
    u = _as_i64(uniq, "uniq")
    # int64, not the public uint8 -- see the note in unique2parent.
    orders = orders_of_uniq(u).astype(np.int64)

    # nested = uniq - 4 * 4**order, done as a shift to stay in exact integers.
    nest = u - (np.int64(1) << (2 * orders + np.int64(2)))

    lat = np.empty(u.size, dtype=np.float64)
    lon = np.empty(u.size, dtype=np.float64)
    for order in np.unique(orders):
        mask = orders == order
        lon[mask], lat[mask] = hp.pix2ang(int(order), nest[mask])

    if latitude == "authalic":
        lat = authalic_to_geodetic(lat)

    if is_scalar:
        return lat[0], lon[0]
    return lat, lon


def mort2geo(morton, *, latitude="authalic"):
    """Convert morton index to lat/lon of pixel center.

    This is the inverse of geo2mort, returning the center coordinates
    of the HEALPix cell identified by the morton index.

    Mixed-order arrays are supported (issue #116): elements are grouped by
    order (:func:`orders_of`), each group runs the uniform kernel, and the
    results scatter back to input positions. Point words (spec §4) are order
    29 by definition and group with order 29 — a point's location is exactly
    what mort2geo returns.

    **Batch vectorized**: array in, arrays out, elementwise.

    Parameters
    ----------
    morton : int or array-like
        Morton index (mixed orders allowed).
    latitude : str, optional
        Latitude convention of the **returned** coordinates (issue #186):
        ``"authalic"`` (default) converts the kernel-frame latitude back to
        WGS84 geodetic; ``"geodetic-spherical"`` returns the legacy spherical
        latitude as-is.  Pass the same convention the words were encoded
        under.

    Returns
    -------
    lat : float or array
        Latitude in degrees
    lon : float or array
        Longitude in degrees
    """
    _check_latitude(latitude)
    # Handle scalar vs array input to match geo2mort behavior
    input_is_scalar = np.isscalar(morton)

    # Group-by-order dispatch for mixed-order input (issue #116).
    if not input_is_scalar:
        words = _as_u64(morton, "morton")
        orders = orders_of(words)
        unique_orders = np.unique(orders)
        if unique_orders.size > 1:
            lat = np.empty(words.size, dtype=np.float64)
            lon = np.empty(words.size, dtype=np.float64)
            for order in unique_orders:
                mask = orders == order
                lat[mask], lon[mask] = mort2geo(words[mask], latitude=latitude)
            return lat, lon

    # Decode morton to normalized address and parent
    normed, parent, order = mort2norm(morton)

    # Convert to UNIQ
    uniq = norm2uniq(normed, parent, order)

    # Convert to lat/lon (uniq2geo decodes the order from the UNIQ value and
    # applies the egress latitude conversion)
    lat, lon = uniq2geo(uniq, latitude=latitude)

    # Return array to match geo2mort behavior
    if input_is_scalar:
        return np.array([lat]), np.array([lon])
    return lat, lon


def mort2bbox(morton, *, latitude="authalic"):
    """Convert morton index to bounding box of the pixel.

    For pixels touching the antimeridian, vertex longitudes at ±180° are
    normalized to use consistent representation based on hemisphere voting,
    preventing bbox misinterpretation as spanning the entire globe.

    Mixed-order arrays are supported (issue #116): elements are grouped by
    order (:func:`orders_of`), each group runs the uniform kernel, and the
    results scatter back to input positions. Point words (spec §4) are order
    29 by definition and group with order 29 — a point yields the bounding box
    of its containing order-29 cell (the cell that contains the point), which
    is exactly the bbox of the order-29 **area** word at the same location. A
    group of points therefore covers a well-defined area, element by element.

    **Batch vectorized**: array in, one bbox dict per word out, elementwise.

    Parameters
    ----------
    morton : int or array-like
        Morton index (mixed orders allowed).
    latitude : str, optional
        Latitude convention of the **returned** box (issue #186):
        ``"authalic"`` (default) converts vertex latitudes back to WGS84
        geodetic; ``"geodetic-spherical"`` returns legacy spherical
        latitudes.  Pass the convention the words were encoded under.

    Returns
    -------
    bbox : dict or list of dicts
        Bounding box in format suitable for STAC/CMR:
        {"west": min_lon, "south": min_lat, "east": max_lon, "north": max_lat}
    """
    _check_latitude(latitude)
    morton = np.atleast_1d(morton)
    is_scalar = len(morton) == 1

    words = _as_u64(morton, "morton")
    # Group-by-order dispatch for mixed-order input (issue #116).
    orders = orders_of(words)
    unique_orders = np.unique(orders)
    if unique_orders.size > 1:
        bboxes = [None] * words.size
        for order in unique_orders:
            (idx,) = np.nonzero(orders == order)
            group = mort2bbox(words[idx], latitude=latitude)
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
    if latitude == "authalic":  # egress: kernel frame -> geodetic (issue #186)
        lats_all = authalic_to_geodetic(lats_all.ravel()).reshape(n, 4)
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
    """Fix polygons that touch (but don't cross) the antimeridian.

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


def mort2polygon(morton, step=1, *, latitude="authalic"):
    """Convert morton index to polygon representation.

    **Batch vectorized**: array in, one ring per word out, elementwise.

    Parameters
    ----------
    morton : int or array-like
        Morton index.
    step : int, optional
        Points per side for the cell boundary (default 1 = 4 corners).
        Use step=32 for 128 boundary points that accurately trace
        curved cell edges, important for polar cells where 4-corner
        polygons poorly approximate the true HEALPix boundary.
    latitude : str, optional
        Latitude convention of the **returned** ring (issue #186):
        ``"authalic"`` (default) converts vertex latitudes back to WGS84
        geodetic; ``"geodetic-spherical"`` returns legacy spherical
        latitudes.  Pass the convention the words were encoded under.

    Returns
    -------
    polygon : list or list of lists
        Polygon coordinates as [[lat, lon], ...] in standard geographic order.
        The polygon is closed (first point repeated at end).

        **Note**: Returns [lat, lon] pairs, NOT [lon, lat]. This is the standard
        geographic coordinate order used by most spatial analysis libraries.

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
    (spec §4) are order 29 by definition and group with order 29 — a point
    yields the polygon ring of its containing order-29 cell, exactly the ring
    of the order-29 **area** word at the same location.
    """
    _check_latitude(latitude)
    morton = np.atleast_1d(morton)
    is_scalar = len(morton) == 1

    words = _as_u64(morton, "morton")
    # Group-by-order dispatch for mixed-order input (issue #116).
    orders = orders_of(words)
    unique_orders = np.unique(orders)
    if unique_orders.size > 1:
        polygons = [None] * words.size
        for order in unique_orders:
            (idx,) = np.nonzero(orders == order)
            group = mort2polygon(words[idx], step=step, latitude=latitude)
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
    if latitude == "authalic":  # egress: kernel frame -> geodetic (issue #186)
        lats_all = authalic_to_geodetic(lats_all.ravel()).reshape(n, ncols)
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


def mort2healpix(morton):
    """Convert morton index to HEALPix cell ID and order.

    **Batch vectorized**: array in, array out, elementwise — but the words
    must share one order, since the returned order is a single scalar.

    Parameters
    ----------
    morton : int or array-like
        Morton index.

    Returns
    -------
    cell_ids : int or ndarray
        HEALPix cell ID(s) in NESTED scheme
    order : int
        HEALPix order (resolution level)

    Raises
    ------
    ValueError
        If the words are at mixed orders (propagated from :func:`mort2norm`,
        which enforces the same-order precondition below).

    Notes
    -----
    The function converts morton indices to HEALPix NESTED scheme cell IDs.
    All input morton indices must be at the same order.

    Examples
    --------
    >>> import mortie
    >>> m = mortie.geo2mort(-80.0, 120.0, order=6)[0]
    >>> cell_id, order = mortie.mort2healpix(m)
    >>> print(f"HEALPix cell {cell_id} at order {order}")
    HEALPix cell 37010 at order 6
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

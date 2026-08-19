"""HEALPix order query, change and validation over packed morton words.

Everything here reads, rewrites or validates a word's **order**: :func:`orders_of`
and :func:`orders_of_uniq` decode it per element, :func:`infer_order_from_morton`
collapses a uniform array to one, :func:`validate_morton` and :func:`is_point`
check the word itself, :func:`clip2order` coarsens and
:func:`generate_morton_children` refines, and :func:`order2res` /
:func:`res2display` give the resolution ladder those orders sit on.  The bulk
refiner :func:`~mortie.batch.children_of` moved to :mod:`mortie.batch` with the
package's other plural operators, the pyarrow skin's aside (issue #170).

Split out of ``mortie.tools`` (issue #159) so the Python surface mirrors the
Rust tree's own decomposition -- this module is the Python side of
``mortie-core/src/morton.rs`` (re-exported as ``mortie_rustie::morton``). The
names stay flat on the package (``mortie.orders_of``, ``mortie.clip2order``):
this module is where they live, not how they are spelled.
"""

from collections import namedtuple

import numpy as np

from . import _rustie

# One row of the res2display resolution ladder (issue #68): the display pair
# (value + unit, rounded within its bracket) alongside the unrounded km, so
# callers can format or compute without re-deriving either.
ResolutionLevel = namedtuple("ResolutionLevel", "order value unit km")

# Mean Earth radius (km), the exact HEALPix sphere the spec page's resolution
# table (docs/specification.md §3) derives from. order2res is the RMS cell
# spacing on this sphere; unified with the page in issue #119.
EARTH_RADIUS_KM = 6371.0088

# Packed-word kernel bridge: morton <-> HEALPix NESTED (vectorized).
_rust_mort2nested = _rustie.rust_mort2nested
_rust_nested2mort = _rustie.rust_nested2mort

# HEALPix orders the packed-u64 kernel reaches (mirrors decimal_morton::MAX_ORDER
# and morton_index.MAX_ORDER): 0 = base cell, 29 = max resolution and the only
# order that carries point words.
MAX_ORDER = 29


def order2res(order):
    """Approximate cell scale (km) at a HEALPix tessellation ``order``.

    The exact RMS cell spacing on the mean-radius HEALPix sphere: every
    order-k cell has identical area ``4*pi*R**2 / (12 * 4**order)`` (HEALPix is
    equal-area), and the cell scale is the square root of that area. Derived
    from :data:`EARTH_RADIUS_KM` so code and the spec page (§3) share one Earth
    model (issue #119).

    ``order`` may be a scalar (returns a ``float``) or an array of orders such
    as :func:`orders_of` yields (returns an ``ndarray``).

    Parameters
    ----------
    order : int or array-like
        HEALPix tessellation order(s).

    Returns
    -------
    float or ndarray
        Approximate cell scale in kilometres (scalar in -> ``float`` out,
        array in -> ``ndarray`` out).

    See Also
    --------
    res2display : the same ladder as display-ready records, order by order.
    """
    # Exponentiate in float so 4**order does not overflow an integer dtype at
    # high orders (an ``orders_of`` uint8 array wraps 4**29 to 0 -> div-by-zero).
    order = np.asarray(order, dtype=np.float64)
    area = 4 * np.pi * EARTH_RADIUS_KM**2 / (12 * 4.0**order)  # km2
    res = np.sqrt(area)
    return float(res) if res.ndim == 0 else res


def res2display(max_order=MAX_ORDER):
    """Resolution ladder for tessellation orders 0 through ``max_order``.

    Returns one record per order rather than printing (issue #68): each
    resolution is expressed in the largest sensible unit -- km at coarse
    orders, m once it drops below 1 km, cm once it drops below 1 m --
    rounded to three decimals within that bracket, so fine orders read
    naturally (order 12 -> ``1.592 km``, order 13 -> ``795.852 m``) rather
    than as tiny km fractions.

    Parameters
    ----------
    max_order : int, optional
        Highest order to include, inclusive. Must lie in ``0..MAX_ORDER``
        (default ``MAX_ORDER`` = 29, the finest order the packed-u64
        kernel encodes).

    Returns
    -------
    list of ResolutionLevel
        One named tuple ``(order, value, unit, km)`` per order, in
        ascending order. ``value``/``unit`` are the display pair; ``km``
        is the unrounded resolution in kilometres for further arithmetic.

    Raises
    ------
    ValueError
        If ``max_order`` lies outside ``0..MAX_ORDER``.

    See Also
    --------
    order2res : the raw kilometres for a single order.

    Examples
    --------
    >>> from mortie import res2display
    >>> levels = res2display(max_order=3)
    >>> levels[0].order, levels[0].unit
    (0, 'km')
    >>> for lvl in res2display(max_order=2):
    ...     print(f"{lvl.value} {lvl.unit} at tessellation order {lvl.order}")
    ... # doctest: +SKIP
    """
    if not 0 <= max_order <= MAX_ORDER:
        raise ValueError(
            f"max_order must be between 0 and {MAX_ORDER}, got {max_order!r}")
    levels = []
    for res in range(max_order + 1):
        km = order2res(res)
        if km >= 1.0:
            value, unit = km, 'km'
        elif km >= 1e-3:
            value, unit = km * 1e3, 'm'
        else:
            value, unit = km * 1e5, 'cm'
        levels.append(ResolutionLevel(res, round(value, 3), unit, km))
    return levels


def orders_of_uniq(uniq):
    """Per-element HEALPix order decoded from UNIQ cell numbers.

    UNIQ is self-describing: ``uniq = 4 * 4**order + nested`` with
    ``0 <= nested < 12 * 4**order``, so order-``k`` values occupy exactly
    ``[4**(k+1), 4**(k+2))`` and consecutive orders tile that line without
    gaps. The order is therefore a pure function of the value — no
    caller-supplied order is needed and mixed-resolution input decodes element
    by element (issue #136).

    Implemented as an exact integer bucket search rather than the
    ``log2(uniq / 4) // 2`` form this module used previously: the float64
    round-trip is not exact above ~2**53, so e.g. ``4**30 - 1`` (the last
    order-28 value) rounds up to ``4**30`` and mis-decodes as order 29.

    The UNIQ counterpart of :func:`orders_of`, and mirrors its contract:
    per-element, mixed-order-native, ``uint8`` out, scalar in -> length-1
    ndarray. One deliberate difference: :func:`orders_of` is pure bit
    arithmetic and never validates, because every 6-bit morton suffix decodes
    to *some* order. UNIQ has no such total decode -- a value outside
    ``[4, 4**31)`` names no cell at any order -- so this raises instead of
    inventing an answer.

    Parameters
    ----------
    uniq : int or array-like
        UNIQ encoded cell number(s).

    Returns
    -------
    ndarray
        ``uint8`` order per element, 0-``MAX_ORDER`` (scalar in -> length-1
        ndarray, matching :func:`orders_of`).

    Raises
    ------
    ValueError
        If any value lies outside the UNIQ range for orders 0-``MAX_ORDER``.
    """
    # Cast defensively: `asarray(..., dtype=int64)` raises OverflowError for a
    # value above int64 and silently *truncates* a float, both of which would
    # bypass the ValueError this function documents. Normalize them here so the
    # contract holds for every input, not just int64-representable ones.
    arr = np.atleast_1d(np.asarray(uniq))
    if arr.dtype.kind == "f" and not np.all(np.equal(np.mod(arr, 1), 0)):
        raise ValueError(
            f"Not a valid UNIQ cell number for orders 0-{MAX_ORDER}: "
            f"{arr.ravel()[0]!r} is not an integer")
    if arr.dtype.kind == "u":
        # uint64 -> int64 *wraps* silently rather than raising, so an oversized
        # value would reach the range check as a meaningless negative and be
        # reported as such. Every wrap lands negative so nothing mis-decodes as
        # valid, but the message would name a number the caller never passed.
        over = arr > np.iinfo(np.int64).max
        if np.any(over):
            raise ValueError(
                f"Not a valid UNIQ cell number for orders 0-{MAX_ORDER}: "
                f"{int(arr[over].ravel()[0])} is out of the int64 range")
    try:
        u = np.atleast_1d(np.asarray(arr, dtype=np.int64))
    except (OverflowError, ValueError, TypeError) as exc:
        raise ValueError(
            f"Not a valid UNIQ cell number for orders 0-{MAX_ORDER}: "
            f"{uniq!r} is out of the int64 range") from exc
    # bounds[k] = 4**(k+1) is the first UNIQ value of order k; the trailing
    # entry closes order MAX_ORDER's range (4**31 still fits int64).
    bounds = np.int64(4) ** np.arange(1, MAX_ORDER + 3, dtype=np.int64)
    orders = (np.searchsorted(bounds, u, side='right') - 1).astype(np.int64)
    bad = (orders < 0) | (orders > MAX_ORDER)
    if np.any(bad):
        raise ValueError(
            f"Not a valid UNIQ cell number for orders 0-{MAX_ORDER}: "
            f"{int(u[bad][0])}")
    return orders.astype(np.uint8)


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


def clip2order(clip_order, midx):
    """Coarsen packed morton words to a lower resolution.

    Degrades each packed word to ``clip_order`` by coarsening it through the
    kernel (the inverse of refining): the base cell and the first ``clip_order``
    tuples are kept, finer detail is dropped, and the suffix is rewritten. Words
    already at or below ``clip_order`` are returned unchanged.

    The ``print_factor`` flag was removed for the 1.x freeze (issue #68). It
    returned ``18 - clip_order``, a level count anchored to the retired
    decimal encoding's order-18 ceiling, so it went negative for the
    order-19..29 words this package now encodes. There is no replacement: the
    levels a word actually drops is ``order - clip_order`` for its own decoded
    order, which :func:`orders_of` gives directly.

    Parameters
    ----------
    clip_order : int
        HEALPix order to degrade to.
    midx : array-like of int
        Packed morton words (see :func:`res2display` for approximate resolutions).

    Returns
    -------
    ndarray
        Coarsened packed words, one per input word.
    """
    midx = np.ascontiguousarray(np.asarray(midx, dtype=np.uint64).ravel())
    return _rustie.rust_mi_coarsen(midx, int(clip_order))


def generate_morton_children(parent_morton, target_order):
    """Generate all child morton indices at a target order.

    Parameters
    ----------
    parent_morton : int
        Parent packed morton word.
    target_order : int
        Target order for children (must be >= parent order).

    Returns
    -------
    children : ndarray
        Array of child packed morton words at target_order.
        If target_order equals parent_order, returns array with parent_morton.

    Raises
    ------
    ValueError
        If ``target_order`` is coarser than the parent word's own order.

    See Also
    --------
    mortie.batch.children_of : the batch form (many parents at one order,
        in one call).

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

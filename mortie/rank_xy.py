"""Subtree-local rank <-> face-local (x, y) bit deinterleave (issue #149).

A depth-``d`` subtree holds ``4**d`` cells whose ascending packed-word order
is a Z-order (morton) curve over a ``2**d x 2**d`` block. A cell's **rank**
is its position ``0..4**d - 1`` within that subtree -- the same base-4
digit-tail rank the coverage bitmaps index by (docs/specification.md §7.2)
-- and it interleaves the bits of a face-local ``(x, y)`` pair: x occupies
the even (LSB-side) interleave bits, y the odd bits, matching the healpy /
HEALPix C++ ``pix2xyf`` convention (origin at the subtree's south corner, x
toward north-east, y toward north-west). The input is rank-space, **not**
packed morton words -- strip the shard prefix first (a word carries base
cell, order, and kind that these pure bit transforms would scramble).

Normative statement: docs/specification.md §8. The pure-numpy mask/shift
ladder kept here (:func:`_rank_to_xy_numpy` / :func:`_xy_to_rank_numpy`) is
the spec's executable reference and the golden-vector generator; the public
functions ship the Rust kernel (`src_rust/src/rank_xy.rs`, a thin binding
over the vendored healpix crate's z-order curve) per the phase-2 gate
decision on issue #149.
"""

import operator

import numpy as np

from . import _rustie
from .tools import MAX_ORDER

# Rust-native kernels (healpix crate zoc::ZOrderCurve, LUT/BMI2 backed).
_rust_rank_to_xy = _rustie.rust_rank_to_xy
_rust_xy_to_rank = _rustie.rust_xy_to_rank

# Mask ladder for the 64-bit even-bit gather/scatter (classic morton
# de/interleave); depth <= 29 keeps ranks within 58 bits, so uint64 is safe.
_MASKS = tuple(
    np.uint64(m)
    for m in (
        0x5555555555555555,
        0x3333333333333333,
        0x0F0F0F0F0F0F0F0F,
        0x00FF00FF00FF00FF,
        0x0000FFFF0000FFFF,
        0x00000000FFFFFFFF,
    )
)


def _compact(v):
    """Gather the even bits of uint64 ``v`` into its low half."""
    v = v & _MASKS[0]
    for i in range(5):
        v = (v | (v >> np.uint64(1 << i))) & _MASKS[i + 1]
    return v


def _spread(v):
    """Scatter the low 32 bits of uint64 ``v`` onto the even bits."""
    v = v & _MASKS[5]
    for i in range(4, -1, -1):
        v = (v | (v << np.uint64(1 << i))) & _MASKS[i]
    return v


def _rank_to_xy_numpy(r):
    """Deinterleave validated uint64 ranks in pure numpy (spec §8 reference)."""
    return _compact(r), _compact(r >> np.uint64(1))


def _xy_to_rank_numpy(xs, ys):
    """Interleave validated uint64 coordinates in pure numpy (spec §8 reference)."""
    return _spread(xs) | (_spread(ys) << np.uint64(1))


def _check_depth(depth):
    """Validate ``depth`` and return it as a plain int."""
    depth = operator.index(depth)
    if not 0 <= depth <= MAX_ORDER:
        raise ValueError(
            f"depth must be between 0 and {MAX_ORDER}, got {depth!r}")
    return depth


def _as_uint64(values, name, bound):
    """Validate integer input in ``[0, bound)`` and return it as uint64."""
    arr = np.atleast_1d(np.asarray(values))
    if arr.dtype.kind not in "iu":
        raise ValueError(
            f"{name} must be integer-typed, got dtype {arr.dtype}")
    if arr.size and (np.any(arr < 0)
                     or np.any(arr.astype(np.uint64) >= np.uint64(bound))):
        raise ValueError(f"{name} must lie in [0, {bound})")
    return arr.astype(np.uint64)


def rank_to_xy(rank, depth):
    """Deinterleave subtree-local ranks into face-local ``(x, y)``.

    The rank's even bits (bit 0, 2, 4, ...) become ``x`` and its odd bits
    become ``y`` -- the healpy / HEALPix C++ ``pix2xyf`` convention, so
    ``rank_to_xy(r, d) == healpy.pix2xyf(2**d, r, nest=True)[:2]`` for
    face-0 pixels. Input is the rank *within* a subtree (e.g. after
    stripping a shard prefix), never a packed morton word.

    Parameters
    ----------
    rank : int or array-like
        Subtree-local rank(s), each in ``[0, 4**depth)``.
    depth : int
        Subtree depth (levels below the subtree root), 0-``MAX_ORDER``.

    Returns
    -------
    x : int or ndarray
        Column coordinate(s) in ``[0, 2**depth)``, ``uint64`` for array
        input (scalar in -> ``int`` out, matching :func:`~.mort2healpix`).
    y : int or ndarray
        Row coordinate(s) in ``[0, 2**depth)``, same convention as ``x``.

    Raises
    ------
    ValueError
        If ``depth`` is outside ``0..MAX_ORDER``, or any rank is negative,
        non-integer-typed, or ``>= 4**depth``.

    See Also
    --------
    xy_to_rank : the inverse interleave.
    """
    depth = _check_depth(depth)
    is_scalar = np.isscalar(rank)
    r = _as_uint64(rank, "rank", 1 << (2 * depth))
    x, y = _rust_rank_to_xy(np.ascontiguousarray(r.ravel()), depth)
    x, y = x.reshape(r.shape), y.reshape(r.shape)
    if is_scalar:
        return int(x[0]), int(y[0])
    return x, y


def xy_to_rank(x, y, depth):
    """Interleave face-local ``(x, y)`` into subtree-local ranks.

    The exact inverse of :func:`rank_to_xy`: ``x`` supplies the even bits
    of the rank, ``y`` the odd bits (healpy / HEALPix C++ ``xyf2pix``
    convention). ``x`` and ``y`` broadcast against each other.

    Parameters
    ----------
    x : int or array-like
        Column coordinate(s), each in ``[0, 2**depth)``.
    y : int or array-like
        Row coordinate(s), each in ``[0, 2**depth)``.
    depth : int
        Subtree depth (levels below the subtree root), 0-``MAX_ORDER``.

    Returns
    -------
    int or ndarray
        Subtree-local rank(s) in ``[0, 4**depth)``, ``uint64`` for array
        input (scalar in -> ``int`` out, matching :func:`~.mort2healpix`).

    Raises
    ------
    ValueError
        If ``depth`` is outside ``0..MAX_ORDER``, or any coordinate is
        negative, non-integer-typed, or ``>= 2**depth``.

    See Also
    --------
    rank_to_xy : the inverse deinterleave.
    """
    depth = _check_depth(depth)
    is_scalar = np.isscalar(x) and np.isscalar(y)
    xs, ys = np.broadcast_arrays(_as_uint64(x, "x", 1 << depth),
                                 _as_uint64(y, "y", 1 << depth))
    rank = _rust_xy_to_rank(np.ascontiguousarray(xs.ravel()),
                            np.ascontiguousarray(ys.ravel()), depth)
    rank = rank.reshape(xs.shape)
    if is_scalar:
        return int(rank[0])
    return rank

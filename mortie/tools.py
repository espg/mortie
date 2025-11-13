"""
functions for morton indexing
"""

import healpy as hp
import numpy as np
import os

# Allow forcing pure Python for testing/comparison
FORCE_PYTHON = os.environ.get('MORTIE_FORCE_PYTHON', '0') == '1'

# Try to import Rust-accelerated functions
try:
    from . import _rustie
    _rust_fast_norm2mort = _rustie.fast_norm2mort
    RUST_AVAILABLE = True
except (ImportError, AttributeError):
    RUST_AVAILABLE = False


def order2res(order):
    res = 111 * 58.6323 * .5**order
    return res


def res2display():
    '''prints resolution levels'''
    for res in range(20):
        print(str(order2res(res)) + ' km at tessellation order ' + str(res))


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


def _python_VaexNorm2Mort_scalar(normed, parent):
    """Pure Python scalar implementation of VaexNorm2Mort"""
    order = 18
    mask = np.int64(3 * 4**(order-1))
    num = 0

    for i in range(order, 0, -1):
        next_bit = (normed & mask) >> ((2*i) - 2)
        num += (next_bit + 1) * 10**(i-1)
        mask >>= 2

    # Parent handling - matches fastNorm2Mort logic
    if parent is not None:
        if parent >= 6:
            parent = parent - 11
            parent = parent * 10**order
            num = num + parent
            num = -1 * num
            num = num - (6 * 10**order)
        else:
            parent = (parent + 1) * 10**order
            num = num + parent
    return num

def _python_VaexNorm2Mort(normed, parents):
    """Pure Python vectorized implementation of VaexNorm2Mort"""
    # Check if all inputs are scalars
    normed_is_scalar = np.ndim(normed) == 0
    parents_is_scalar = np.ndim(parents) == 0
    all_scalar = normed_is_scalar and parents_is_scalar

    # Convert to arrays
    normed = np.atleast_1d(np.asarray(normed, dtype=np.int64))
    parents = np.atleast_1d(np.asarray(parents, dtype=np.int64))

    # Ensure same length (broadcast)
    if len(normed) == 1 and len(parents) > 1:
        normed = np.repeat(normed, len(parents))
    elif len(parents) == 1 and len(normed) > 1:
        parents = np.repeat(parents, len(normed))

    # Vectorized computation
    result = np.array([_python_VaexNorm2Mort_scalar(n, p) for n, p in zip(normed, parents)], dtype=np.int64)

    # Return scalar only if all inputs were scalar
    return result[0] if all_scalar else result

# Public API - uses Rust (via fastNorm2Mort with order=18) if available
def VaexNorm2Mort(normed, parents):
    """Convert normalized HEALPix addresses to morton indices (order 18)

    Vaex-compatible version with order hardcoded to 18.
    Uses Rust implementation if available, otherwise falls back to pure Python.

    Args:
        normed: int or array - Normalized HEALPix address
        parents: int or array - Parent base cell (0-11)

    Returns:
        Morton indices as int64 or array
    """
    if RUST_AVAILABLE and not FORCE_PYTHON:
        # Use Rust fastNorm2Mort with order=18
        return _rust_fast_norm2mort(18, normed, parents)
    else:
        return _python_VaexNorm2Mort(normed, parents)


def _python_fastNorm2Mort_scalar(order, normed, parent):
    """Pure Python scalar implementation of fastNorm2Mort"""
    if order > 18:
        raise ValueError("Max order is 18 (to output to 64-bit int).")

    mask = np.int64(3 * 4**(order-1))
    num = 0

    for i in range(order, 0, -1):
        next_bit = (normed & mask) >> ((2*i) - 2)
        num += (next_bit + 1) * 10**(i-1)
        mask >>= 2

    # Parent handling
    if parent is not None:
        if parent >= 6:
            parent = parent - 11
            parent = parent * 10**order
            num = num + parent
            num = -1 * num
            num = num - (6 * 10**order)
        else:
            parent = (parent + 1) * 10**order
            num = num + parent
    return num

def _python_fastNorm2Mort(order, normed, parents):
    """Pure Python vectorized implementation of fastNorm2Mort"""
    # Check if all inputs are scalars
    order_is_scalar = np.ndim(order) == 0
    normed_is_scalar = np.ndim(normed) == 0
    parents_is_scalar = np.ndim(parents) == 0
    all_scalar = order_is_scalar and normed_is_scalar and parents_is_scalar

    # Convert to arrays
    order = np.atleast_1d(np.asarray(order, dtype=np.int64))
    normed = np.atleast_1d(np.asarray(normed, dtype=np.int64))
    parents = np.atleast_1d(np.asarray(parents, dtype=np.int64))

    # Determine output length (broadcast)
    max_len = max(len(order), len(normed), len(parents))

    # Broadcast to same length
    if len(order) == 1:
        order = np.repeat(order, max_len)
    if len(normed) == 1:
        normed = np.repeat(normed, max_len)
    if len(parents) == 1:
        parents = np.repeat(parents, max_len)

    # Validate lengths match
    if not (len(order) == len(normed) == len(parents)):
        raise ValueError("All array inputs must have the same length")

    # Vectorized computation
    result = np.array([_python_fastNorm2Mort_scalar(o, n, p)
                       for o, n, p in zip(order, normed, parents)], dtype=np.int64)

    # Return scalar only if all inputs were scalar
    return result[0] if all_scalar else result

# Public API - uses Rust if available, falls back to pure Python
def fastNorm2Mort(order, normed, parents):
    """Convert normalized HEALPix addresses to morton indices

    Uses Rust implementation if available, otherwise falls back to pure Python.

    Args:
        order: int or array - Tessellation order (1-18)
        normed: int or array - Normalized HEALPix address
        parents: int or array - Parent base cell (0-11)

    Returns:
        Morton indices as int64 or array
    """
    if RUST_AVAILABLE and not FORCE_PYTHON:
        return _rust_fast_norm2mort(order, normed, parents)
    else:
        return _python_fastNorm2Mort(order, normed, parents)


def geo2uniq(lats, lons, order=18):
    """Calculates UNIQ coding for lat/lon

    Defaults to max morton resolution of order 18"""

    nside = 2**order

    nest = hp.ang2pix(nside, lons, lats, lonlat=True, nest=True)
    uniq = 4 * (nside**2) + nest

    return uniq


def geo2mort(lats, lons, order=18):
    """Calculates morton indices from geographic coordinates

    lats: array-like
    lons: array-like
    order: int"""


    uniq = geo2uniq(lats, lons, order)
    parents = unique2parent(uniq)
    normed = heal_norm(parents, order, uniq)
    morton = fastNorm2Mort(order, normed.ravel(), parents.ravel())

    return morton


def clip2order(clip_order, midx=None, print_factor=False):
    """Convenience function to clip max res morton indices to lower res

    clip_order: int ; resolution to degrade to
    midx: array(ints) or None ; morton indices at order 18

    See `res2display` for approximate resolutions

    Setting print_factor to True will return scaling factor;
    default setting of false will execute the clip on the array"""

    factor = 18 - clip_order

    if print_factor:
        return 10**factor
    else:
        negidx = midx < 0
        clipped = np.abs(midx) // 10**factor
        clipped[negidx] *= -1
        return clipped

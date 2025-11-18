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


def validate_morton(morton, order=18):
    """Validate that a morton index is properly formed

    Parameters
    ----------
    morton : int
        Morton index to validate
    order : int
        HEALPix order

    Returns
    -------
    bool
        True if valid morton index

    Raises
    ------
    ValueError
        If morton index is invalid
    """
    abs_morton = abs(int(morton))
    morton_str = str(abs_morton)

    # Check length matches expected for given order
    expected_length = order + 1  # 1 for parent+1, order for position digits
    if len(morton_str) != expected_length:
        raise ValueError(f"Morton index {morton} has {len(morton_str)} digits, expected {expected_length} for order {order}")

    # Extract the parent cell
    num_digits = order
    divisor = 10**num_digits
    parent = abs_morton // divisor - 1

    # Parent must be 0-11
    if parent < 0 or parent > 11:
        raise ValueError(f"Invalid parent cell {parent} (must be 0-11)")

    # Check each digit is 1-4
    morton_digits = abs_morton % divisor
    for i in range(order):
        digit = (morton_digits // 10**i) % 10
        if digit < 1 or digit > 4:
            raise ValueError(f"Invalid morton digit {digit} at position {i} (must be 1-4)")

    return True


def mort2norm(morton, order=18):
    """Convert morton index back to normalized address and parent cell

    Parameters
    ----------
    morton : int or array-like
        Morton index (can be negative for southern hemisphere)
    order : int
        HEALPix order (default 18)

    Returns
    -------
    normed : int or array
        Normalized HEALPix address
    parent : int or array
        Parent base cell (0-11)
    """
    morton = np.atleast_1d(morton).astype(np.int64)
    is_scalar = len(morton) == 1

    # Validate morton indices
    for m in morton:
        validate_morton(m, order)

    # Handle negative morton indices (southern hemisphere)
    sign = np.sign(morton)
    sign[sign == 0] = 1
    abs_morton = np.abs(morton)

    # Extract parent cell from the most significant digit
    # Morton format: [parent+1][order digits]
    # e.g., for order=6: 3122124 -> parent=2 (3-1), rest=122124

    # Calculate number of digits
    num_digits = order
    divisor = 10**num_digits

    # Extract parent (most significant part)
    parent = abs_morton // divisor - 1

    # Extract the normalized address encoded in remaining digits
    morton_digits = abs_morton % divisor

    # Decode morton digits back to normalized address
    normed = np.zeros_like(morton, dtype=np.int64)

    for i in range(order, 0, -1):
        digit = (morton_digits // 10**(i-1)) % 10
        # Each morton digit (1-4) maps to 2 bits (00, 01, 10, 11)
        bits = digit - 1
        normed |= bits << (2*(i-1))

    # Apply sign back to parent for southern hemisphere
    parent = parent * sign

    if is_scalar:
        return normed[0], parent[0]
    return normed, parent


def norm2uniq(normed, parent, order=18):
    """Convert normalized address and parent to UNIQ encoding

    Parameters
    ----------
    normed : int or array
        Normalized HEALPix address
    parent : int or array
        Parent base cell (can be negative for southern hemisphere)
    order : int
        HEALPix order

    Returns
    -------
    uniq : int or array
        UNIQ encoded pixel index
    """
    nside = 2**order
    N_pix = nside**2

    # Get absolute parent for calculation
    abs_parent = np.abs(parent)

    # Convert normalized address back to nest index
    nest = normed + (abs_parent * N_pix)

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
    lon, lat = hp.pix2ang(nside, nest, nest=True, lonlat=True)

    return lat, lon


def mort2geo(morton, order=18):
    """Convert morton index to lat/lon of pixel center

    This is the inverse of geo2mort, returning the center coordinates
    of the HEALPix cell identified by the morton index.

    Parameters
    ----------
    morton : int or array-like
        Morton index
    order : int
        HEALPix order (default 18)

    Returns
    -------
    lat : float or array
        Latitude in degrees
    lon : float or array
        Longitude in degrees
    """
    # Handle scalar vs array input to match geo2mort behavior
    input_is_scalar = np.isscalar(morton)

    # Decode morton to normalized address and parent
    normed, parent = mort2norm(morton, order)

    # Convert to UNIQ
    uniq = norm2uniq(normed, parent, order)

    # Convert to lat/lon
    lat, lon = uniq2geo(uniq, order)

    # Return array to match geo2mort behavior
    if input_is_scalar:
        return np.array([lat]), np.array([lon])
    return lat, lon


def mort2bbox(morton, order=18):
    """Convert morton index to bounding box of the pixel

    Parameters
    ----------
    morton : int or array-like
        Morton index
    order : int
        HEALPix order (default 18)

    Returns
    -------
    bbox : dict or list of dicts
        Bounding box in format suitable for STAC/CMR:
        {"west": min_lon, "south": min_lat, "east": max_lon, "north": max_lat}
    """
    morton = np.atleast_1d(morton)
    is_scalar = len(morton) == 1

    # First get the pixel center
    normed, parent = mort2norm(morton, order)
    uniq = norm2uniq(normed, parent, order)

    nside = 2**order
    nest = uniq - 4 * (nside**2)

    # Get pixel boundaries
    boundaries = hp.boundaries(nside, nest, nest=True, step=1)

    bboxes = []
    for i in range(len(morton)):
        # Get boundary vertices (shape: 3 x 4 for x,y,z coordinates of 4 corners)
        if len(morton) == 1:
            verts = boundaries
        else:
            verts = boundaries[:, :, i]

        # Convert to lat/lon
        theta, phi = hp.vec2ang(verts.T)
        lats = 90 - np.degrees(theta)
        lons = np.degrees(phi)

        # Handle longitude wrapping
        lons = np.where(lons > 180, lons - 360, lons)

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


def mort2polygon(morton, order=18):
    """Convert morton index to polygon representation

    Parameters
    ----------
    morton : int or array-like
        Morton index
    order : int
        HEALPix order (default 18)

    Returns
    -------
    polygon : list or list of lists
        Polygon coordinates as [[lon, lat], ...] suitable for GeoJSON.
        The polygon is closed (first point repeated at end).
    """
    morton = np.atleast_1d(morton)
    is_scalar = len(morton) == 1

    # Get pixel information
    normed, parent = mort2norm(morton, order)
    uniq = norm2uniq(normed, parent, order)

    nside = 2**order
    nest = uniq - 4 * (nside**2)

    # Get pixel boundaries
    boundaries = hp.boundaries(nside, nest, nest=True, step=1)

    polygons = []
    for i in range(len(morton)):
        # Get boundary vertices
        if len(morton) == 1:
            verts = boundaries
        else:
            verts = boundaries[:, :, i]

        # Convert to lat/lon
        theta, phi = hp.vec2ang(verts.T)
        lats = 90 - np.degrees(theta)
        lons = np.degrees(phi)

        # Handle longitude wrapping
        lons = np.where(lons > 180, lons - 360, lons)

        # Create polygon as list of [lon, lat] pairs
        # Close the polygon by repeating first point
        polygon = [[float(lons[j]), float(lats[j])] for j in range(len(lons))]
        polygon.append(polygon[0])  # Close the polygon
        polygons.append(polygon)

    if is_scalar:
        return polygons[0]
    return polygons


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

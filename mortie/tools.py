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


def infer_order_from_morton(morton):
    """Infer the HEALPix order from a morton index.

    Parameters
    ----------
    morton : int
        Morton index

    Returns
    -------
    int
        The HEALPix order
    """
    abs_morton = abs(int(morton))
    morton_str = str(abs_morton)
    # Order is number of digits minus 1 (for the parent digit)
    return len(morton_str) - 1


def validate_morton(morton, order=None):
    """Validate that a morton index is properly formed

    Parameters
    ----------
    morton : int
        Morton index to validate
    order : int, optional
        HEALPix order. If None, inferred from morton index.

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

    # Infer order if not provided
    if order is None:
        order = len(morton_str) - 1

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


def mort2norm(morton):
    """Convert morton index back to normalized address and parent cell

    Parameters
    ----------
    morton : int or array-like
        Morton index (can be negative for southern hemisphere)

    Returns
    -------
    normed : int or array
        Normalized HEALPix address
    parent : int or array
        Parent base cell (0-11)
    order : int or array
        HEALPix order inferred from morton index
    """
    morton = np.atleast_1d(morton).astype(np.int64)
    is_scalar = len(morton) == 1

    # Infer order and validate morton indices
    orders = []
    for m in morton:
        order = infer_order_from_morton(m)
        orders.append(order)
        validate_morton(m, order)

    # Check all orders are the same (for array input)
    if len(set(orders)) > 1:
        raise ValueError(f"Mixed orders in morton array: {set(orders)}")

    order = orders[0]
    num_digits = order
    divisor = 10**num_digits

    # Initialize arrays
    normed = np.zeros_like(morton, dtype=np.int64)
    parent = np.zeros_like(morton, dtype=np.int64)

    for idx, m in enumerate(morton):
        if m < 0:
            # Negative morton: southern hemisphere (parent cells 6-11)
            # Reverse the encoding steps:
            # Forward: num = morton_digits + (parent-11)*10^order
            #          num = -1 * num
            #          num = num - 6*10^order
            # Reverse: add 6*10^order, negate, then extract parts

            # Step 1: Add back 6*10^order
            temp = m + (6 * divisor)  # temp = 2866867

            # Step 2: Negate
            temp = -temp  # temp = -2866867

            # Step 3: This is morton_digits + (parent-11)*10^order
            # The tricky part: (parent-11) could be negative
            # So we need to handle the sign carefully

            if temp >= 0:
                # Normal case (shouldn't happen for southern hemisphere)
                parent[idx] = temp // divisor + 11
                morton_digits = temp % divisor
            else:
                # temp is negative, meaning (parent-11) is negative
                # temp = morton_digits + (parent-11)*10^order
                # Since morton_digits is positive and less than 10^order,
                # we can extract it by modulo
                morton_digits = temp % divisor
                if morton_digits < 0:
                    morton_digits += divisor

                # Now get parent
                parent_minus_11 = (temp - morton_digits) // divisor
                parent[idx] = parent_minus_11 + 11
        else:
            # Positive morton: northern hemisphere (parent cells 0-5)
            # Format: (parent+1) * 10**order + morton_digits
            parent[idx] = m // divisor - 1
            morton_digits = m % divisor

        # Decode morton digits back to normalized address
        for i in range(order, 0, -1):
            digit = (morton_digits // 10**(i-1)) % 10
            # Each morton digit (1-4) maps to 2 bits (00, 01, 10, 11)
            bits = digit - 1
            normed[idx] |= bits << (2*(i-1))

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
    lon, lat = hp.pix2ang(nside, nest, nest=True, lonlat=True)

    return lat, lon


def mort2geo(morton):
    """Convert morton index to lat/lon of pixel center

    This is the inverse of geo2mort, returning the center coordinates
    of the HEALPix cell identified by the morton index.

    Parameters
    ----------
    morton : int or array-like
        Morton index

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

    Parameters
    ----------
    morton : int or array-like
        Morton index

    Returns
    -------
    bbox : dict or list of dicts
        Bounding box in format suitable for STAC/CMR:
        {"west": min_lon, "south": min_lat, "east": max_lon, "north": max_lat}
    """
    morton = np.atleast_1d(morton)
    is_scalar = len(morton) == 1

    # First get the pixel center
    normed, parent, order = mort2norm(morton)
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


def mort2polygon(morton):
    """Convert morton index to polygon representation

    Parameters
    ----------
    morton : int or array-like
        Morton index

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
    """
    morton = np.atleast_1d(morton)
    is_scalar = len(morton) == 1

    # Get pixel information
    normed, parent, order = mort2norm(morton)
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


def generate_morton_children(parent_morton, target_order):
    """
    Generate all child morton indices at a target order.

    Parameters
    ----------
    parent_morton : int
        Parent morton index
    target_order : int
        Target order for children (must be >= parent order)

    Returns
    -------
    children : ndarray
        Array of child morton indices at target_order.
        If target_order equals parent_order, returns array with parent_morton.

    Examples
    --------
    >>> generate_morton_children(-5111131, target_order=7)
    array([-51111311, -51111312, -51111313, -51111314])

    >>> generate_morton_children(-5111131, target_order=8)
    array([-511113111, -511113112, ..., -511113144])

    >>> generate_morton_children(-5111131, target_order=6)
    array([-5111131])

    Notes
    -----
    The function generates children by appending all possible digit combinations
    (1, 2, 3, 4) to the parent morton index for the number of levels between
    parent_order and target_order. If already at target_order, returns the
    parent itself.
    """
    # Get parent order
    _, _, parent_order = mort2norm(parent_morton)

    if target_order < parent_order:
        raise ValueError(f"target_order ({target_order}) must be >= parent_order ({parent_order})")

    # If already at target order, return parent as-is
    if target_order == parent_order:
        return np.array([parent_morton])

    # Calculate number of levels to descend
    level_diff = target_order - parent_order
    n_children = 4**level_diff

    # Generate all combinations of digits (1,2,3,4) for level_diff positions
    children = []
    for i in range(n_children):
        # Convert i to base-4 representation with level_diff digits
        suffix = ""
        val = i
        for _ in range(level_diff):
            digit = (val % 4) + 1  # Morton digits are 1,2,3,4 (not 0,1,2,3)
            suffix = str(digit) + suffix
            val //= 4

        # Append suffix to parent morton (preserves sign)
        child = int(str(parent_morton) + suffix)
        children.append(child)

    return np.array(children)


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
    >>> cell_id, order = mort2healpix(-5111131)
    >>> print(f"HEALPix cell {cell_id} at order {order}")

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

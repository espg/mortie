"""
Greedy polygon subdivision for morton indices.

This module provides functions to generate compact morton index coverage
for arbitrary geometries using greedy subdivision with configurable constraints.
"""

import numpy as np


def greedy_morton_polygon(lat, lon, order=18, max_boxes=16, ordermax=None, verbose=False):
    """
    Calculate compact morton index coverage using greedy subdivision.

    This function finds the highest-order morton indices that fully contain
    all points in the input geometry. It greedily subdivides until reaching
    the max_boxes ceiling or ordermax limit.

    Parameters
    ----------
    lat : array-like
        Latitude values in degrees
    lon : array-like
        Longitude values in degrees
    order : int, default=18
        Morton cell order to start with (higher = finer resolution)
    max_boxes : int, default=16
        Maximum number of morton boxes allowed
    ordermax : int, optional
        Maximum order (resolution) for subdivision. Branches that reach
        this order will not be subdivided further. If None, no order limit.
    verbose : bool, default=False
        Print subdivision progress

    Returns
    -------
    mbbox : ndarray
        Array of morton indices that form the minimum bounding box
    mbbox_orders : ndarray
        Array of orders for each morton index

    Examples
    --------
    >>> import numpy as np
    >>> lat = np.array([-75, -75, -70, -70])
    >>> lon = np.array([-80, -70, -70, -80])
    >>> mbbox, orders = greedy_morton_polygon(lat, lon, max_boxes=16, ordermax=6)

    Notes
    -----
    The algorithm uses a greedy strategy that prioritizes splitting coarser
    (lower-order) boxes first to maintain balanced spatial coverage across
    the geometry. This prevents over-subdivision of any single region.
    """
    from . import geo2mort

    # Convert to numpy arrays
    lat = np.asarray(lat)
    lon = np.asarray(lon)

    # Remove NaN values
    valid = ~(np.isnan(lat) | np.isnan(lon))
    lat = lat[valid]
    lon = lon[valid]

    if len(lat) == 0:
        raise ValueError("No valid points provided")

    # Calculate morton indices at specified order
    midx = geo2mort(lat, lon, order=order)

    # Convert to string array for character-by-character analysis
    midx_str = np.array([str(m) for m in midx])

    if verbose:
        print(f"Starting greedy subdivision with max_boxes={max_boxes}, ordermax={ordermax}")
        print(f"Total morton indices at order {order}: {len(midx_str):,}\n")

    # Greedy subdivision
    morton_boxes = _greedy_subdivide(midx_str, max_boxes, ordermax=ordermax, verbose=verbose)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Final result: {len(morton_boxes)} morton boxes")
        print(f"{'='*70}\n")

    # Convert strings back to integers and calculate orders
    morton_indices = np.array([int(box) for box in morton_boxes])
    morton_orders = np.array([_estimate_order_from_prefix(box) for box in morton_boxes])

    return morton_indices, morton_orders


def _greedy_subdivide(morton_strings, max_boxes, ordermax=None, verbose=False):
    """
    Greedily subdivide morton indices until hitting limits.

    Parameters
    ----------
    morton_strings : array-like of str
        Morton indices as strings
    max_boxes : int
        Maximum number of boxes allowed
    ordermax : int, optional
        Maximum order for subdivision
    verbose : bool
        Print progress

    Returns
    -------
    boxes : list of str
        List of morton index prefixes
    """
    # Start with one box containing all indices
    boxes = [{'prefix': _find_common_prefix(morton_strings), 'indices': morton_strings}]

    if verbose:
        print(f"Initial: 1 box covering {len(morton_strings):,} indices")
        print(f"  Box 1: prefix='{boxes[0]['prefix']}'\n")

    iteration = 0
    while len(boxes) < max_boxes:
        iteration += 1
        if verbose:
            print(f"Iteration {iteration}: Current boxes: {len(boxes)}/{max_boxes}")

        # Find which box can be split
        best_box_idx = None
        best_split = None
        max_split_count = 0
        best_order = float('inf')

        for i, box in enumerate(boxes):
            # Check if this box has reached ordermax
            if ordermax is not None:
                box_order = _estimate_order_from_prefix(box['prefix'])
                if box_order >= ordermax:
                    if verbose and iteration == 1:
                        print(f"  Box {i+1} ('{box['prefix']}') at order {box_order} >= ordermax {ordermax}, skipping")
                    continue

            # Try to split this box
            split_result = _try_split(box['prefix'], box['indices'])

            if split_result is not None:
                n_splits = len(split_result)
                new_total = len(boxes) - 1 + n_splits

                if new_total <= max_boxes and n_splits > 1:
                    # Priority: lower order first, then more splits
                    box_order = _estimate_order_from_prefix(box['prefix'])

                    if best_box_idx is None:
                        best_box_idx = i
                        best_split = split_result
                        max_split_count = n_splits
                        best_order = box_order
                    else:
                        if (box_order < best_order or
                            (box_order == best_order and n_splits > max_split_count)):
                            best_box_idx = i
                            best_split = split_result
                            max_split_count = n_splits
                            best_order = box_order

        if best_box_idx is None:
            if verbose:
                print(f"  → No more splits possible within budget\n")
            break

        # Apply the best split
        old_box = boxes[best_box_idx]
        if verbose:
            print(f"  → Splitting box {best_box_idx+1} ('{old_box['prefix']}', order {best_order}) into {len(best_split)} sub-boxes")

        boxes.pop(best_box_idx)

        # Clip children to ordermax if necessary
        if ordermax is not None:
            clipped_split = []
            for child in best_split:
                child_order = _estimate_order_from_prefix(child['prefix'])
                if child_order > ordermax:
                    # Clip prefix to ordermax
                    clipped_prefix = child['prefix'][:ordermax + 2]  # +2 for minus sign and parent digit
                    if verbose:
                        print(f"    Clipping child '{child['prefix']}' (order {child_order}) → '{clipped_prefix}' (order {ordermax})")
                    child['prefix'] = clipped_prefix
                clipped_split.append(child)
            best_split = clipped_split

        boxes.extend(best_split)

        if verbose:
            print(f"  → New total: {len(boxes)} boxes\n")

    # Extract prefixes
    result = [box['prefix'] for box in boxes]

    if verbose:
        print(f"Final subdivision complete:")
        for i, box in enumerate(boxes, 1):
            print(f"  Box {i:2d}: '{box['prefix']}' ({len(box['indices']):,} indices)")

    return result


def _try_split(prefix, indices):
    """Try to split a box into sub-boxes."""
    if len(np.unique(indices)) == 1:
        return None

    divergence_pos = len(prefix)

    if divergence_pos >= min(len(s) for s in indices):
        return None

    next_chars = np.array([s[divergence_pos] if divergence_pos < len(s) else ''
                           for s in indices])
    unique_next_chars = np.unique(next_chars[next_chars != ''])

    if len(unique_next_chars) <= 1:
        return None

    # Create sub-boxes
    sub_boxes = []
    for char in unique_next_chars:
        mask = np.array([s[divergence_pos] == char if divergence_pos < len(s) else False
                        for s in indices])
        group_indices = indices[mask]

        if len(group_indices) > 0:
            group_prefix = _find_common_prefix(group_indices)
            sub_boxes.append({'prefix': group_prefix, 'indices': group_indices})

    return sub_boxes if len(sub_boxes) > 1 else None


def _estimate_order_from_prefix(prefix):
    """
    Estimate morton order from prefix string.

    Parameters
    ----------
    prefix : str
        Morton index prefix

    Returns
    -------
    order : int
        Estimated order (number of digits - 1)
    """
    digits = prefix.lstrip('-')
    return len(digits) - 1


def _find_common_prefix(strings):
    """
    Find longest common prefix among strings.

    Parameters
    ----------
    strings : array-like of str
        Array of strings

    Returns
    -------
    prefix : str
        Longest common prefix
    """
    if len(strings) == 0:
        return ""

    if len(strings) == 1:
        return strings[0]

    prefix = strings[0]

    for i in range(len(prefix)):
        char = prefix[i]
        for s in strings[1:]:
            if i >= len(s) or s[i] != char:
                return prefix[:i]

    return prefix

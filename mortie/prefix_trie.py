"""
Prefix trie with greedy spanning-tree refinement for morton indices.

Builds a compacted prefix trie over the string representations of morton
indices and uses a greedy algorithm to select the fewest prefix-cells
that span the input data within a cell budget.

Key entry points:

- :func:`split_children` / :func:`split_children_geo` — build the trie
- :func:`morton_polygon` — refine to *n_cells* prefix-cells
  (n_cells=4 gives a bounding box, n_cells=12 gives a polygon)
- :func:`geo_morton_polygon` — geographic convenience wrapper
"""

import math
import os
import numpy as np

FORCE_PYTHON = os.environ.get('MORTIE_FORCE_PYTHON', '0') == '1'


def _auto_max_depth(n_cells):
    """Compute minimum trie depth that guarantees *n_cells* candidates.

    Worst case: every branching level is a binary split (2 children).
    To guarantee at least *n_cells* leaf candidates we need depth *d*
    such that ``2**d >= n_cells``, i.e. ``d = ceil(log2(n_cells))``.
    We add one extra level of headroom so that ``morton_polygon`` has
    good candidates to choose from.
    """
    if n_cells <= 1:
        return 1
    return math.ceil(math.log2(n_cells)) + 1

try:
    from . import _rustie
    _rust_split_children = _rustie.split_children_rust
    RUST_AVAILABLE = True
except (ImportError, AttributeError):
    RUST_AVAILABLE = False


class MortonChild:
    """A node in the compacted prefix trie over morton index strings.

    Each node owns a boolean mask into the shared character array and
    a characteristic prefix string.  Children are created lazily when
    the column under the mask diverges.

    Parameters
    ----------
    char_array : ndarray of shape (N, L), dtype='U1'
        Shared 2-D character array (all rows same length).
    mask : ndarray of shape (N,), dtype=bool
        Which rows of *char_array* belong to this node.
    start_col : int
        Column index at which to resume scanning.
    characteristic : str
        Common prefix accumulated so far.
    original_array : ndarray
        The original integer morton array (shared reference).
    max_depth : int or None
        Maximum branching depth (None = unlimited).
    _depth : int
        Current branching depth (0 at root level).
    """

    __slots__ = (
        "_char_array",
        "_mask",
        "_original_array",
        "_original_indices",
        "_max_depth",
        "_depth",
        "characteristic",
        "len",
        "children",
        "nchildren",
    )

    def __init__(
        self,
        char_array,
        mask,
        start_col,
        characteristic,
        original_array,
        max_depth=None,
        _depth=0,
    ):
        self._char_array = char_array
        self._mask = mask
        self._original_array = original_array
        self._original_indices = None
        self._max_depth = max_depth
        self._depth = _depth
        self.characteristic = characteristic
        self.len = int(mask.sum())
        self.children = []
        self.nchildren = 0

        if self.len == 0:
            raise ValueError("Empty mask — no indices to compact")

        # Validate that all masked rows share the expected prefix
        if start_col > 0 and self.len > 0:
            prefix_cols = char_array[mask, :start_col]
            # Build expected prefix characters (pad characteristic to
            # start_col length with leading space for positive numbers)
            expected = list(characteristic.ljust(start_col))
            # For negative numbers the first char is '-'; for positive
            # it is ' '.  The characteristic for positive numbers omits
            # the leading space, so we need to account for that.
            if start_col <= len(characteristic):
                pass  # prefix is within characteristic — trust the caller
            # Quick check: all rows in the first column should match
            first_chars = char_array[mask, 0]
            unique_first = np.unique(first_chars)
            if len(unique_first) > 1:
                raise ValueError(
                    "Input array is not compressible — "
                    "indices do not share expected prefix"
                )

        self._compact(start_col)

    def _compact(self, col):
        """Walk columns, extending characteristic while unique, branching on divergence."""
        char_array = self._char_array
        mask = self._mask
        ncols = char_array.shape[1]

        while col < ncols:
            unique = np.unique(char_array[mask, col])

            if len(unique) == 1:
                self.characteristic += unique[0]
                col += 1
            else:
                # Divergence — create children if depth allows
                if self._max_depth is not None and self._depth >= self._max_depth:
                    break

                for val in unique:
                    child_mask = mask & (char_array[:, col] == val)
                    child = MortonChild(
                        char_array,
                        child_mask,
                        col + 1,
                        self.characteristic + val,
                        self._original_array,
                        max_depth=self._max_depth,
                        _depth=self._depth + 1,
                    )
                    self.children.append(child)
                self.nchildren = len(self.children)
                break

    @property
    def mantissa_array(self):
        """The original morton indices belonging to this node."""
        if self._original_indices is not None:
            return self._original_array[np.array(self._original_indices)]
        return self._original_array[self._mask]

    def __repr__(self):
        return (
            f"MortonChild(characteristic={self.characteristic!r}, "
            f"len={self.len}, nchildren={self.nchildren})"
        )


def _rebuild_tree_from_flat(flat_nodes, morton_array):
    """Reconstruct MortonChild tree from flat Rust output.

    Parameters
    ----------
    flat_nodes : list of (characteristic, count, original_indices, child_node_ids, depth)
    morton_array : ndarray
        The original integer morton array.

    Returns
    -------
    list of MortonChild
        Root-level nodes (depth == 0).
    """
    # Build all MortonChild objects first (bypass __init__ via __new__)
    objects = []
    for characteristic, count, indices, child_ids, depth in flat_nodes:
        obj = MortonChild.__new__(MortonChild)
        obj._char_array = None
        obj._mask = None
        obj._original_array = morton_array
        obj._original_indices = indices
        obj._max_depth = None
        obj._depth = depth
        obj.characteristic = characteristic
        obj.len = count
        obj.children = []          # populated below
        obj.nchildren = len(child_ids)
        objects.append(obj)

    # Wire up children
    for i, (_, _, _, child_ids, _) in enumerate(flat_nodes):
        objects[i].children = [objects[cid] for cid in child_ids]

    # Return root-level nodes
    return [obj for obj in objects if obj._depth == 0]


def split_children(morton_array, max_depth=4):
    """Build a compacted prefix trie over *morton_array* and return root children.

    Parameters
    ----------
    morton_array : array-like of int
        Morton indices (signed integers).
    max_depth : int or None
        Maximum branching depth.  ``None`` means full recursion.
        Default is 4.

    Returns
    -------
    list of MortonChild
        One root-level child per (sign, first-digit) group.
    """
    morton_array = np.asarray(morton_array, dtype=np.int64)
    if morton_array.ndim != 1 or len(morton_array) == 0:
        raise ValueError("morton_array must be a non-empty 1-D integer array")

    # Try Rust path first
    if RUST_AVAILABLE and not FORCE_PYTHON:
        flat_nodes = _rust_split_children(morton_array, max_depth=max_depth)
        return _rebuild_tree_from_flat(flat_nodes, morton_array)

    return _split_children_python(morton_array, max_depth=max_depth)


def _split_children_python(morton_array, max_depth=4):
    """Pure-Python implementation of split_children."""
    # Convert to strings
    str_arr = np.array([str(v) for v in morton_array])

    # Determine max string length (negative numbers have '-' prefix).
    # Ensure column 0 is always sign/pad even when all numbers are positive.
    max_len = max(len(s) for s in str_arr)
    has_negatives = any(s[0] == "-" for s in str_arr)
    if not has_negatives:
        max_len += 1  # room for leading space as sign column

    # Left-pad with spaces so all strings are the same length
    str_arr = np.array([s.rjust(max_len) for s in str_arr])

    # Build 2-D character array (N x max_len)
    char_array = np.array([[ch for ch in s] for s in str_arr])

    # Column 0 is sign/pad: '-' for negative, ' ' for positive
    sign_col = char_array[:, 0]
    unique_signs = np.unique(sign_col)

    roots = []
    for sign in unique_signs:
        sign_mask = sign_col == sign

        # Column 1 is the first digit — group by it
        digit_col = char_array[:, 1]
        unique_digits = np.unique(digit_col[sign_mask])

        for digit in unique_digits:
            group_mask = sign_mask & (digit_col == digit)

            # Build characteristic: negative gets '-' + digit, positive just digit
            if sign == "-":
                characteristic = "-" + digit
            else:
                characteristic = digit

            child = MortonChild(
                char_array,
                group_mask,
                2,  # start scanning at column 2
                characteristic,
                morton_array,
                max_depth=max_depth,
                _depth=0,
            )
            roots.append(child)

    return roots


def split_children_geo(lats, lons, order=18, max_depth=4):
    """Build compacted prefix trie from geographic coordinates.

    Parameters
    ----------
    lats, lons : array-like
        Latitude and longitude values in degrees.
    order : int
        Morton tessellation order.  Default is 18.
    max_depth : int or None
        Maximum branching depth.  Default is 4.

    Returns
    -------
    list of MortonChild
    """
    from .tools import geo2mort
    morton_array = geo2mort(lats, lons, order=order)
    return split_children(morton_array, max_depth=max_depth)


def geo_morton_polygon(lats, lons, n_cells, order=18, max_depth=None):
    """Compute a morton polygon from geographic coordinates.

    Builds a prefix trie over the morton indices of the input coordinates
    and greedily refines it to at most *n_cells* prefix-cells.  Common
    values:

    - ``n_cells=4``  → bounding box (4 prefix-cells)
    - ``n_cells=12`` → polygon (tighter fit, 12 prefix-cells)

    Parameters
    ----------
    lats, lons : array-like
        Latitude and longitude values in degrees.
    n_cells : int
        Maximum number of cells in the returned list.
    order : int
        Morton tessellation order.  Default is 18.
    max_depth : int or None
        Maximum branching depth.  When *None* (default), automatically
        derived from *n_cells* as ``ceil(log2(n_cells)) + 1``.

    Returns
    -------
    list of MortonChild
    """
    if max_depth is None:
        max_depth = _auto_max_depth(n_cells)
    roots = split_children_geo(lats, lons, order=order, max_depth=max_depth)
    return morton_polygon(roots, n_cells=n_cells)


def morton_polygon_from_array(morton_array, n_cells, max_depth=None):
    """Build trie and refine to *n_cells* in one call.

    Parameters
    ----------
    morton_array : array-like of int
        Morton indices (signed integers).
    n_cells : int
        Maximum number of cells in the returned list.
    max_depth : int or None
        Maximum branching depth.  When *None* (default), automatically
        derived from *n_cells* as ``ceil(log2(n_cells)) + 1``.

    Returns
    -------
    list of MortonChild
    """
    if max_depth is None:
        max_depth = _auto_max_depth(n_cells)
    roots = split_children(morton_array, max_depth=max_depth)
    return morton_polygon(roots, n_cells=n_cells)


def _cell_area(node):
    """Area of a HEALPix cell from its characteristic string.

    The characteristic encodes sign + digits.  The *order* is the number
    of digits (excluding a leading '-' for negative indices):
      1 digit  → base cell  → area = 1
      2 digits → area = 1/4
      k digits → area = 4^(-(k-1))
    """
    c = node.characteristic
    ndigits = len(c.lstrip("-"))
    return 4.0 ** (-(ndigits - 1))


def morton_polygon(roots, n_cells):
    """Greedily expand tree nodes to minimize area within a cell budget.

    Starting from the root-level children produced by :func:`split_children`,
    repeatedly replace the most "efficient" parent with its children until the
    budget of *n_cells* is reached.

    Common *n_cells* values:

    - ``n_cells=4``  → bounding box (coarse, 4 prefix-cells)
    - ``n_cells=12`` → polygon (tighter fit, up to 12 prefix-cells)

    Efficiency is defined as the area saved per additional cell consumed:
        benefit  = parent_area - sum(child_areas)
        cost     = nchildren - 1        (net new cells added)
        efficiency = benefit / cost

    Coverage is preserved because expansion only replaces a parent with its
    exact children — no points are lost or duplicated.

    Parameters
    ----------
    roots : list of MortonChild
        Root-level children from :func:`split_children`.
    n_cells : int
        Maximum number of cells in the returned list.

    Returns
    -------
    list of MortonChild
        Refined prefix-cells (len <= *n_cells*).
    """
    current = list(roots)

    while len(current) < n_cells:
        best_idx = None
        best_eff = -1.0

        for i, node in enumerate(current):
            if node.nchildren == 0:
                continue
            cost = node.nchildren - 1
            if len(current) + cost > n_cells:
                continue
            parent_area = _cell_area(node)
            child_area = sum(_cell_area(c) for c in node.children)
            benefit = parent_area - child_area
            efficiency = benefit / cost
            if efficiency > best_eff:
                best_eff = efficiency
                best_idx = i

        if best_idx is None:
            break

        node = current[best_idx]
        current[best_idx:best_idx + 1] = node.children

    return current

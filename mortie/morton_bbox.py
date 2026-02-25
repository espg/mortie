"""
Morton bounding box: find the fewest prefix-cells that span an array of
morton indices by building and compacting a prefix trie on their string
representations.
"""

import numpy as np


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
        return self._original_array[self._mask]

    def __repr__(self):
        return (
            f"MortonChild(characteristic={self.characteristic!r}, "
            f"len={self.len}, nchildren={self.nchildren})"
        )


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


def morton_bounding_box(morton_array, max_depth=4):
    """Convenience wrapper: return the root-level prefix cells spanning *morton_array*.

    Each returned ``MortonChild.characteristic`` is a cell prefix that
    covers a range of morton indices.  Together the list is the minimal
    bounding box.

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
    """
    return split_children(morton_array, max_depth=max_depth)


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


def refine_bbox(roots, n_cells):
    """Greedily expand tree nodes to minimize area within a cell budget.

    Starting from the root-level children produced by :func:`split_children`,
    repeatedly replace the most "efficient" parent with its children until the
    budget of *n_cells* is reached.

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
        Refined bounding-box cells (len <= *n_cells*).
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

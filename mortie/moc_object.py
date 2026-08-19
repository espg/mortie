"""The geometry-first ``Moc`` object over the MOC kernel (issue #196).

mortie's coverage surface is two layers, and the split is deliberate:

* the **kernel** — the free ``moc_*`` functions in :mod:`mortie._moc`, words in
  and words out, unchanged and un-deprecated.  Array-first consumers (moczarr's
  mask internals, zagg's grids, the batch shardmap machinery) keep calling them
  on plain ndarrays at zero wrapping cost, and the plural batch forms
  (:func:`~mortie.batch.mocs_and`, :func:`~mortie.batch.mocs_intersect`,
  :func:`~mortie.batch.mocs_to_orders`,
  :func:`~mortie.batch.polygons_to_morton_mocs`) stay function-shaped
  permanently — an offset-packed many-cover operation has no natural ``self``.
* the **object** — :class:`Moc`, here.  It is ergonomics and nothing else: a
  thin view over the canonical ``uint64`` word array, never a new
  representation.  **Every method is a single delegation to a kernel
  function.**  There is no algebra in this module; a method body that is not
  one kernel call is a bug, not a feature.

The interchange format stays the array.  :meth:`Moc.__morton_moc__` hands back
the canonical words, and any object exposing that dunder is accepted wherever a
``Moc`` is — mortie owns geometry and words, downstream stores own their own
encodings, and the two meet at a plain ``uint64`` array with neither importing
the other's private grammar.

**Conservative directions.**  Every predicate below is *cover* algebra, not
*polygon* algebra.  A cover dilates the polygon it was built from — a boundary
cell is included when it only partly overlaps — so the covered area is always a
superset of the polygon, on both sides of the comparison.  Read the answers
accordingly; each predicate's docstring points back here.

| call | the question it answers exactly | as a *polygon* question |
| --- | --- | --- |
| `a.intersects(b)` | do the two covers share any area? | may say `True` for polygons that only come within a cell of each other; a `False` is decisive. The safe direction for "must I read this shard?" — never a false skip. |
| `a.contains(b)` | is every cell of `b` inside `a`? | may say `True` when `b`'s polygon pokes outside `a`'s by less than a cell. Exact for the question that matters — "will a store whose coverage is `a` answer a query for `b`?" |
| `a.within(b)` | is every cell of `a` inside `b`? | the mirror of `contains`; same caveat. |
| either side empty | an empty cover is contained in everything and intersects nothing | `a.contains(empty)` is `True` (vacuously — there is no cell of `empty` outside `a`) while `a.intersects(empty)` is `False`. The two predicates disagree here by definition, not by accident: ask `contains` about coverage and `intersects` about work to do. |
| `a == b` | identical canonical words | **not** geometric equality: two covers of the same polygon built at different `tolerance` / `max_cells` compare unequal. |
"""

import warnings

import numpy as np

from . import _moc
from ._moc import (
    compress_moc,
    moc_and,
    moc_intersects,
    moc_minus,
    moc_or,
    moc_to_order,
    moc_xor,
)
from .coverage import _FLAT_COVER_WARN_THRESHOLD, morton_coverage_moc
from .orders import orders_of, res2display

# The public surface of the former ``mortie.moc`` submodule -- what the
# migration shim below still resolves (with a DeprecationWarning) for one minor
# version.  A frozen historical roster (the 0.9.9 surface), not a live view of
# the kernel: pinned as a literal, and pinned in the tests against that same
# history, so a name added to the kernel later does not join the deprecated
# namespace.
_KERNEL_NAMES = (
    "common_ancestor",
    "compress_moc",
    "moc_and",
    "moc_intersects",
    "moc_min",
    "moc_minus",
    "moc_not",
    "moc_or",
    "moc_to_order",
    "moc_xor",
    "split_base_cells",
)


def _words(operand):
    """Canonical ``uint64`` words of a set-operation operand.

    Parameters
    ----------
    operand : Moc or array_like
        A :class:`Moc`, anything exposing ``__morton_moc__()``, or a raw
        morton word array.

    Returns
    -------
    numpy.ndarray
        The operand's words as a 1-D ``uint64`` array.
    """
    protocol = getattr(operand, "__morton_moc__", None)
    if protocol is not None:
        operand = protocol()
    return np.asarray(operand, dtype=np.uint64).ravel()


def _nest_depth(node):
    """Nesting depth of a coordinate structure (``[lon, lat]`` counts as 1).

    Parameters
    ----------
    node : array_like
        A coordinate, ring, ring list or part list.

    Returns
    -------
    int
        Number of sequence levels, counting an ``ndarray``'s own ``ndim``.
    """
    depth = 0
    while True:
        if isinstance(node, np.ndarray):
            return depth + node.ndim
        if isinstance(node, (list, tuple)) and len(node):
            depth += 1
            node = node[0]
        else:
            return depth


def _geojson_ring_groups(obj):
    """Rings of a GeoJSON mapping, one list per geometry.

    Dependency-free (no shapely, no geojson package): ``Feature``,
    ``FeatureCollection``, ``Polygon`` and ``MultiPolygon`` are recognized by
    structure, so the ``"type"`` members may be absent.

    The grouping is where the two fill rules meet.  *Within* a geometry every
    ring — parts and holes alike — goes into one list, which
    :func:`~mortie.morton_coverage_moc`'s multipart form resolves with a single
    even-odd descent: disjoint parts union with no internal seam, nested rings
    carve holes.  *Across* geometries even-odd would be wrong — overlapping and
    nested features are legal in a ``FeatureCollection`` and must union, not
    cancel — so each geometry keeps its own group and
    :func:`_source_words` unions the covers.

    Parameters
    ----------
    obj : dict
        A GeoJSON ``Feature``, ``FeatureCollection``, ``Polygon`` or
        ``MultiPolygon`` mapping.

    Returns
    -------
    list of list
        One entry per geometry, each holding that geometry's rings as
        ``(N, 2+)`` sequences of ``[lon, lat]`` positions.

    Raises
    ------
    ValueError
        If the mapping carries no polygonal coordinates, holds a null geometry
        (legal in RFC 7946 §3.2, but nothing to cover), or names a geometry
        type that is not ``Polygon`` / ``MultiPolygon`` — ``GeometryCollection``
        included.
    """
    if "features" in obj:
        geoms = [feature.get("geometry", feature) for feature in obj["features"]]
    elif "geometry" in obj:
        geoms = [obj["geometry"]]
    else:
        geoms = [obj]

    groups = []
    for geom in geoms:
        if geom is None:
            raise ValueError("GeoJSON feature has a null geometry")
        if geom.get("type") == "GeometryCollection":
            raise ValueError(
                "Moc covers Polygon / MultiPolygon geometry; got a "
                "GeometryCollection"
            )
        if "coordinates" not in geom:
            raise ValueError("GeoJSON geometry has no 'coordinates' member")
        coords = geom["coordinates"]
        depth = _nest_depth(coords)
        kind = geom.get("type") or ("MultiPolygon" if depth == 4 else "Polygon")
        if kind == "Polygon" and depth == 3:
            groups.append(list(coords))
        elif kind == "MultiPolygon" and depth == 4:
            groups.append([ring for part in coords for ring in part])
        else:
            raise ValueError(
                f"Moc covers Polygon / MultiPolygon geometry; got type={kind!r} "
                f"with coordinates nested {depth} deep"
            )
    groups = [rings for rings in groups if rings]
    if not groups:
        raise ValueError("GeoJSON holds no polygon rings")
    return groups


def _ring_latlons(rings):
    """Split ``[lon, lat]`` rings into the ``(lats, lons)`` the coverers take.

    GeoJSON closes its rings and mortie's coverers do not, so a repeated
    closing vertex is dropped here — exactly the rule
    :func:`~mortie.morton_coverage_moc` applies to a single ring, applied to
    every ring so the multipart form sees no zero-length edge.  A lone ring is
    handed back as bare arrays (not one-element lists) so it takes the same
    single-ring kernel path a direct ``morton_coverage_moc`` call would.

    Parameters
    ----------
    rings : list
        One entry per ring, each an ``(N, 2+)`` sequence of ``[lon, lat]``
        positions.

    Returns
    -------
    lats, lons : numpy.ndarray or list of numpy.ndarray
        Vertex latitudes / longitudes in degrees; bare arrays for a single
        ring, parallel lists of arrays for several.

    Raises
    ------
    ValueError
        If a ring is not an ``(N, 2+)`` array of positions.
    """
    lats, lons = [], []
    for ring in rings:
        xy = np.asarray(ring, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] < 2:
            raise ValueError(
                "each ring must be an (N, 2+) array of [lon, lat] positions, "
                f"got shape {xy.shape}"
            )
        if xy.shape[0] > 3 and xy[0, 0] == xy[-1, 0] and xy[0, 1] == xy[-1, 1]:
            xy = xy[:-1]
        lats.append(xy[:, 1])
        lons.append(xy[:, 0])
    if len(lats) == 1:
        return lats[0], lons[0]
    return lats, lons


def _source_rings(source):
    """Rings of a non-GeoJSON, non-words constructor source.

    Parameters
    ----------
    source : array_like
        One ``(N, 2+)`` ring of ``[lon, lat]`` positions, or a sequence of
        them.

    Returns
    -------
    list
        One entry per ring.

    Raises
    ------
    ValueError
        If the nesting is neither a single ring nor a list of rings.
    """
    depth = _nest_depth(source)
    if depth == 2:
        return [source]
    if depth == 3:
        return list(source)
    raise ValueError(
        "Moc takes a GeoJSON mapping, a uint64 word array, one (N, 2+) ring of "
        f"[lon, lat] positions or a list of them; got nesting depth {depth}"
    )


def _reject_coverage_knobs(tolerance, max_cells, latitude, kind):
    """Refuse the coverage knobs on a source that is already words.

    ``tolerance`` / ``max_cells`` / ``latitude`` steer the coverer, and a
    words or protocol source never reaches it — ignoring them silently would
    let ``Moc(a.words, max_cells=8)`` read as "re-cover at a smaller budget"
    while doing nothing, and would accept the ``tolerance`` + ``max_cells``
    pair that :func:`~mortie.morton_coverage_moc` rejects.

    Parameters
    ----------
    tolerance : float or None
        Angular stop criterion as passed to the constructor.
    max_cells : int or None
        Best-first cell budget as passed to the constructor.
    latitude : str
        Latitude convention as passed to the constructor.
    kind : str
        What the source is, for the error message.

    Raises
    ------
    ValueError
        If any of the three was given a non-default value.
    """
    for name, value, default in (("tolerance", tolerance, None),
                                 ("max_cells", max_cells, None),
                                 ("latitude", latitude, "authalic")):
        if value != default:
            raise ValueError(
                f"{name}= steers the coverer and applies to a geometry source "
                f"only; this source is {kind}"
            )


def _source_words(source, tolerance, max_cells, latitude):
    """Resolve a constructor source to its morton words, before compaction.

    A geometry source is covered one geometry at a time and the covers are
    concatenated: ``moc_or(a, b) == compress_moc(concatenate([a, b]))``
    (:func:`~mortie.moc_or`) and :class:`Moc` compacts eagerly, so handing back
    the concatenation *is* the union across geometries — no python reduce over
    kernel calls.  A lone geometry is returned as its own cover untouched, so a
    single ring still takes the single-ring kernel path.

    Parameters
    ----------
    source : object
        Any :class:`Moc` constructor source; see that class for the matrix.
    tolerance : float or None
        Angular stop criterion in degrees, passed through to the coverer.
    max_cells : int or None
        Best-first cell budget, passed through to the coverer.
    latitude : str
        Latitude convention of the input vertices.

    Returns
    -------
    numpy.ndarray
        Morton words (``uint64``), not yet compacted.
    """
    protocol = getattr(source, "__morton_moc__", None)
    if protocol is not None:
        _reject_coverage_knobs(tolerance, max_cells, latitude,
                               "already a cover (__morton_moc__)")
        return np.asarray(protocol(), dtype=np.uint64).ravel()
    if isinstance(source, dict):
        groups = _geojson_ring_groups(source)
    else:
        words = source if isinstance(source, np.ndarray) else None
        if words is None and isinstance(source, (list, tuple)) and _nest_depth(source) < 2:
            words = np.asarray(source)
        if words is not None and words.ndim == 1 and words.dtype.kind in "ui":
            _reject_coverage_knobs(tolerance, max_cells, latitude,
                                   "already morton words")
            return words.astype(np.uint64, copy=False).ravel()
        groups = [_source_rings(source)]
    covers = []
    for rings in groups:
        lats, lons = _ring_latlons(rings)
        covers.append(morton_coverage_moc(
            lats, lons, tolerance=tolerance, max_cells=max_cells,
            latitude=latitude,
        ))
    if len(covers) == 1:
        return covers[0]
    return np.concatenate(covers)


class Moc:
    """A multi-order coverage as an object — coverage geometry that reads as geometry.

    A thin view over the canonical ``uint64`` word array and nothing more: the
    words *are* the MOC, this class is the ergonomics, and every method is a
    single delegation to the kernel function of the same meaning (see the
    module docstring for the two-layer rule and the conservative-direction
    table the predicates share).

    Coverage is **multi-order by default** — coarse cells inside, fine cells
    along the boundary, down to :func:`~mortie.morton_coverage_moc`'s default
    finest order — so there is no ``order`` argument.  Use :meth:`at` to cast
    to a flat single-order cell list when a consumer's grid wants one, and read
    :func:`repr` to see what resolution you actually got.  Construction is
    **deterministic**: the same input through the same mortie version yields
    the same words, byte for byte.

    The words are normalized eagerly (:func:`~mortie.compress_moc`) and stored
    read-only, which makes ``==`` and :func:`hash` well defined; the instance
    itself is immutable.

    Parameters
    ----------
    source : dict or array_like or Moc
        What to cover.  A GeoJSON mapping (``Feature`` / ``FeatureCollection``
        / ``Polygon`` / ``MultiPolygon``, parsed without shapely and tolerant
        of a missing ``"type"``); one ``(N, 2+)`` ring of ``[lon, lat]``
        positions or a list of such rings (holes and disjoint parts are
        resolved by one even-odd descent); a 1-D ``uint64`` array of morton
        words; or any object exposing ``__morton_moc__()``, which includes
        another :class:`Moc`.  Rings *within* one geometry take that even-odd
        descent — nested rings carve holes — while separate geometries of a
        ``FeatureCollection`` are **unioned**, since overlapping and nested
        features are legal there and must add area rather than cancel it.
        Words are taken as given: any integer array is cast to ``uint64``, so a
        negative or downcast value wraps rather than being rejected here and
        fails later inside the kernel.
    tolerance : float, optional
        Stop refining a boundary cell once its angular radius (in degrees)
        drops to this value; see :func:`~mortie.morton_coverage_moc`.
    max_cells : int, optional
        Best-first cell budget for the boundary; see
        :func:`~mortie.morton_coverage_moc`.  Mutually exclusive with
        ``tolerance``.  Applied per geometry, so a ``FeatureCollection``
        budgets each feature's boundary rather than the union's.
    latitude : str, optional
        Latitude convention of the input vertices (default ``"authalic"``);
        see :func:`~mortie.morton_coverage`.

    Raises
    ------
    ValueError
        If ``source`` is not one of the forms above, the coverer rejects the
        rings (see :func:`~mortie.morton_coverage_moc`), or a coverage knob
        (``tolerance`` / ``max_cells`` / ``latitude``) is given for a source
        that is already words — those steer the coverer, which such a source
        never reaches, so they are refused rather than ignored.

    See Also
    --------
    mortie.morton_coverage_moc : the coverage kernel this wraps.
    mortie.compress_moc : the eager normalization applied to every instance.

    Examples
    --------
    >>> import mortie
    >>> aoi = {"type": "Polygon", "coordinates": [[
    ...     [-76.56, 38.87], [-76.50, 38.87], [-76.50, 38.91],
    ...     [-76.56, 38.91], [-76.56, 38.87]]]}
    >>> region = mortie.moc(aoi)
    >>> region.contains(region)
    True
    >>> region.to_order(9).dtype
    dtype('uint64')
    """

    __slots__ = ("words",)

    def __init__(self, source, tolerance=None, max_cells=None, *, latitude="authalic"):
        words = compress_moc(_source_words(source, tolerance, max_cells, latitude))
        words.setflags(write=False)
        object.__setattr__(self, "words", words)

    def __setattr__(self, name, value):
        """Refuse attribute assignment -- a Moc is immutable."""
        raise AttributeError(
            "Moc is immutable (its hash is its words); build a new one instead"
        )

    def __delattr__(self, name):
        """Refuse attribute deletion -- a Moc is immutable."""
        raise AttributeError("Moc is immutable; build a new one instead")

    def __reduce__(self):
        """Rebuild through the constructor -- pickle and copy cannot set slots.

        A ``__slots__`` class is restored by assigning its slot state, which
        the immutability guard above refuses; reconstructing from the words
        instead keeps ``Moc`` picklable (workers marshal their arguments) and
        deep-copyable at the cost of one ``compress_moc`` on already-compact
        words.

        Returns
        -------
        tuple
            The ``(callable, args)`` pair the pickle protocol rebuilds from.
        """
        return (Moc, (self.words,))

    @classmethod
    def from_polygon(cls, lats, lons, tolerance=None, max_cells=None, *,
                     latitude="authalic"):
        """Cover a polygon given as vertex latitudes and longitudes.

        The MOCpy-vocabulary spelling of the coverage constructor, for callers
        who already have ``(lats, lons)`` rather than GeoJSON.  Multipart /
        holes take the list-of-rings form, exactly as
        :func:`~mortie.morton_coverage_moc` documents.

        Parameters
        ----------
        lats, lons : array_like
            Vertex latitudes / longitudes in degrees (one ring), or a list of
            such arrays for the multipart form.
        tolerance : float, optional
            Angular stop criterion in degrees.
        max_cells : int, optional
            Best-first cell budget for the boundary.
        latitude : str, optional
            Latitude convention of the input vertices (default
            ``"authalic"``).

        Returns
        -------
        Moc
            The polygon's multi-order cover.
        """
        return cls(
            morton_coverage_moc(
                lats, lons, tolerance=tolerance, max_cells=max_cells,
                latitude=latitude,
            )
        )

    def intersects(self, other):
        """Whether this cover and ``other`` share any area, at any order.

        Cover algebra, not polygon algebra — see the module docstring's
        conservative-direction table: ``True`` can mean "within a cell of each
        other", ``False`` is decisive.

        Parameters
        ----------
        other : Moc or array_like
            The cover to test against.

        Returns
        -------
        bool
            ``True`` if the two covers overlap anywhere.
        """
        return moc_intersects(self.words, _words(other))

    def contains(self, other):
        """Whether every cell of ``other`` lies inside this cover.

        Cover algebra, not polygon algebra — see the module docstring's
        conservative-direction table.  This is the "will a store whose
        coverage is ``self`` answer a query for ``other``?" test.

        Parameters
        ----------
        other : Moc or array_like
            The cover to test for containment.

        Returns
        -------
        bool
            ``True`` if ``other`` adds no area outside this cover.
        """
        return moc_minus(_words(other), self.words).size == 0

    def within(self, other):
        """Whether every cell of this cover lies inside ``other``.

        The mirror of :meth:`contains`; the same conservative direction
        applies (module docstring).

        Parameters
        ----------
        other : Moc or array_like
            The cover to test containment against.

        Returns
        -------
        bool
            ``True`` if this cover adds no area outside ``other``.
        """
        return moc_minus(self.words, _words(other)).size == 0

    def union(self, other):
        """Cells in this cover or in ``other`` (MOCpy ``union``, ``|``).

        Parameters
        ----------
        other : Moc or array_like
            The cover to union with.

        Returns
        -------
        Moc
            The union cover.
        """
        return Moc(moc_or(self.words, _words(other)))

    def intersection(self, other):
        """Cells in both this cover and ``other`` (MOCpy ``intersection``, ``&``).

        Parameters
        ----------
        other : Moc or array_like
            The cover to intersect with.

        Returns
        -------
        Moc
            The intersection cover.
        """
        return Moc(moc_and(self.words, _words(other)))

    def difference(self, other):
        r"""Cells in this cover but not ``other`` (MOCpy ``difference``, ``-``).

        Parameters
        ----------
        other : Moc or array_like
            The cover to subtract.

        Returns
        -------
        Moc
            The difference cover ``self \ other``.
        """
        return Moc(moc_minus(self.words, _words(other)))

    def symmetric_difference(self, other):
        """Cells in exactly one of the two covers (MOCpy ``symmetric_difference``, ``^``).

        Parameters
        ----------
        other : Moc or array_like
            The cover to compare against.

        Returns
        -------
        Moc
            The symmetric-difference cover.
        """
        return Moc(moc_xor(self.words, _words(other)))

    __or__ = union
    __and__ = intersection
    __sub__ = difference
    __xor__ = symmetric_difference

    def to_order(self, order, max_cells=_FLAT_COVER_WARN_THRESHOLD):
        """Cast to a flat list of cells at one fixed ``order``.

        The consumer-grid direction: a multi-order cover in, every cell at
        ``order`` out, as the canonical ``uint64`` **array** — a flat
        single-order cell list is not a MOC, and re-normalizing it would
        collapse it straight back to the compact form.

        Parameters
        ----------
        order : int
            Target HEALPix order (0-29).
        max_cells : int or None, optional
            Pre-emptive budget on the densified cell count; see
            :func:`~mortie.moc_to_order`.  ``None`` opts out.

        Returns
        -------
        numpy.ndarray
            Sorted 1-D array of flat morton cells at ``order`` (``uint64``).
        """
        return moc_to_order(self.words, order, max_cells)

    def __morton_moc__(self):
        """Canonical morton words — the interchange protocol (issue #196).

        Returns
        -------
        numpy.ndarray
            The read-only ``uint64`` word array backing this cover.
        """
        return self.words

    def __eq__(self, other):
        """Compare canonical words -- word identity, not geometric equality."""
        if not isinstance(other, Moc):
            return NotImplemented
        return np.array_equal(self.words, other.words)

    def __hash__(self):
        """Hash the canonical words; sound because a Moc is immutable."""
        return hash(self.words.tobytes())

    def __len__(self):
        """Count the cells in the cover."""
        return int(self.words.size)

    def __iter__(self):
        """Iterate the cover's morton words."""
        return iter(self.words)

    def __repr__(self):
        """Show the cell count and the orders actually present."""
        if self.words.size == 0:
            return "Moc(0 cells)"
        orders = orders_of(self.words)
        low, high = int(orders.min()), int(orders.max())
        span = f"order {low}" if low == high else f"orders {low}-{high}"
        finest = res2display(high)[-1]
        return (
            f"Moc({self.words.size} cells, {span}, "
            f"finest {finest.value:g} {finest.unit})"
        )


class _MocNamespace:
    """Callable stand-in for the retired ``mortie.moc`` submodule (issue #196).

    Calling it builds a :class:`Moc`; attribute access to the kernel functions
    the submodule used to hold still resolves, with a
    :class:`DeprecationWarning`, for one minor version.  Statement-form
    ``import mortie.moc`` and ``from mortie.moc import ...`` break at the
    rename — the module is gone — which is why the shim covers attribute
    access and not the import system.

    The warning is raised on **every** access; deduplication is the warnings
    module's job, not the shim's.  The default filters already show a given
    warning once per call site, while ``always``/``error`` filters — and a
    ``catch_warnings(record=True)`` block in a downstream test suite, however
    late in the process it runs — see every occurrence instead of only the
    process-wide first.
    """

    __slots__ = ()

    def __call__(self, source, tolerance=None, max_cells=None, *,
                 latitude="authalic"):
        """Build a :class:`Moc`; see that class for the argument matrix."""
        return Moc(source, tolerance=tolerance, max_cells=max_cells,
                   latitude=latitude)

    def __getattr__(self, name):
        """Resolve a retired submodule attribute, warning on every access."""
        if name not in _KERNEL_NAMES:
            raise AttributeError(
                f"'mortie.moc' is the Moc constructor (issue #196), not the old "
                f"submodule, and has no attribute {name!r}"
            )
        warnings.warn(
            f"mortie.moc.{name} is deprecated: mortie.moc is now the Moc "
            f"constructor, not a module. Use the top-level mortie.{name} "
            f"instead; this shim is removed in the next minor release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(_moc, name)

    def __dir__(self):
        """List the deprecated kernel names this shim still resolves."""
        return sorted(_KERNEL_NAMES)

    def __repr__(self):
        """Say what this object is, so `mortie.moc` is not mistaken for a module."""
        return "<mortie.moc: the Moc constructor; mortie.moc.<kernel> is deprecated>"


moc = _MocNamespace()
"""The :class:`Moc` constructor, bound where the submodule used to be."""

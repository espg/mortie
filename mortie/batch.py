"""Bulk (plural) operators over morton sets, MOCs and geometry columns.

Every function here is the **batch twin** of a scalar that lives elsewhere in
the package: one call carries a whole ragged column across the Python/Rust
boundary, releases the GIL, and lets Rust parallelize across the elements, so
the per-call fixed cost that dominates a Python loop over half a million
footprints is paid once.  Element ``i`` of the result is bit-identical to the
scalar applied to element ``i`` alone -- the batch is a throughput surface, not
a second semantics.

Consolidated here by **arity** (issue #170), which is a different axis from the
domain split the rest of the package is organised on (issues #156 / #159): the
plural twins used to sit beside their scalars in :mod:`mortie.coverage`,
:mod:`mortie.moc`, :mod:`mortie.orders` and :mod:`mortie.geometry`, so "what is
batched?" had four answers and every new twin landed in whichever of those
modules was furthest from the size aim.  The scalar/plural pair is kept
navigable by a ``See Also`` on each side: every plural below names its scalar
by module path, and every scalar's docstring points back at its plural here
(issue #170).

The Rust kernels stay split by domain (``coverage/batch.rs``, ``moc/batch.rs``,
``decimal_morton/batch.rs``, ``wkb/batch.rs``), so this module deliberately does
**not** mirror the Rust tree the way :mod:`mortie.orders` and
:mod:`mortie.convert` do: the Python surface is organised for callers and the
Rust for kernels, and the two need not be 1:1.

The pyarrow skin is a third axis: :func:`mortie.arrow.from_wkbs` and
:func:`mortie.arrow.polygons_to_morton_mocs` take pyarrow columns and stay in
:mod:`mortie.arrow` (issue #154).  The names here stay flat on the package
(``mortie.from_wkbs``, ``mortie.children_of``): this module is where they live,
not how they are spelled.
"""

import numpy as np

from . import _rustie
from .coverage import _FLAT_COVER_WARN_THRESHOLD
from .geometry import _wkb_bytes


def polygons_to_morton_mocs(lats, lons, offsets, order=18, tolerance=None,
                            max_cells=None, normalize=True):
    """Compute MOC coverage of many independent polygons in one call.

    The batch sibling of :func:`morton_coverage_moc` (issue #153): the ragged
    polygon set crosses the Python/Rust boundary **once**, the GIL is released
    for the whole batch, and Rust parallelizes across polygons — so the
    per-call fixed cost that dominates a Python loop over half a million
    footprints is paid once.  Identity-preserving: result ``i`` is exactly the
    cover of input polygon ``i`` (unlike the multipart form of
    :func:`morton_coverage_moc`, which unions its rings into one cover).  The
    plural *MOCs* in the name marks that many→many contract — one MOC per
    input polygon — against the many→one union of the multipart form.

    Polygons are covered in chunks and each chunk is copied into the ragged
    output as it lands, so peak memory is about the returned ``values`` array
    plus one chunk of in-flight covers — not the ~2.5x of holding every
    polygon's cover to concatenate at the end.

    Input and output are ragged arrays in arrow list layout: polygon ``i`` is
    ``lats[offsets[i]:offsets[i+1]]`` / ``lons[offsets[i]:offsets[i+1]]``, and
    its MOC is ``values[out_offsets[i]:out_offsets[i+1]]`` in the result —
    byte-identical to ``morton_coverage_moc`` on that ring alone.

    Parameters
    ----------
    lats, lons : array_like
        Flat ``float64`` vertex latitudes / longitudes in degrees, all rings
        concatenated.  Each entry is **one ring**: the batch has no
        multipart/hole spelling, so decompose a multi-ring footprint yourself
        and cover it with :func:`morton_coverage_moc`'s list-of-rings form.
    offsets : array_like
        ``int64`` arrow list offsets: polygon ``i`` spans
        ``[offsets[i], offsets[i+1])``.  ``len(offsets) - 1`` polygons.  The
        offsets must **exactly cover** the vertex arrays — ``offsets[0] == 0``
        and ``offsets[-1] == len(lats) == len(lons)`` — so a sliced arrow
        array must be re-based before it gets here (:mod:`mortie.arrow` does
        that for you); anything else is an error naming the endpoint that
        failed.
    order : int, optional
        Finest HEALPix order (1-29), shared by every polygon.  Default 18.
    tolerance : float, optional
        Stop refining a boundary cell once its angular radius (in **degrees**)
        drops to this value — exactly :func:`morton_coverage_moc`'s
        ``tolerance``, applied as a **single shared setting** to every polygon
        in the batch.
    max_cells : int, optional
        Best-first cell budget per polygon — exactly
        :func:`morton_coverage_moc`'s ``max_cells``, shared by every polygon.
        A budget below some polygon's representable floor is raised for that
        polygon (soft target, as in the scalar path) and one summary warning
        is emitted.
    normalize : bool, optional
        Ring-orientation handling, identical in meaning to
        :func:`morton_coverage`'s ``normalize`` — see that function for the
        full ring-winding contract.  Default ``True``.

    Returns
    -------
    values : numpy.ndarray
        All polygons' morton MOC words concatenated (``uint64``).
    out_offsets : numpy.ndarray
        ``int64`` arrow list offsets into ``values``, length
        ``len(offsets)``; ``out_offsets[0]`` is always 0.

    Raises
    ------
    ValueError
        Fail-fast, naming the **lowest-index** offending polygon (e.g.
        ``polygon 4217: needs at least 3 vertices``): non-monotone or
        out-of-bounds offsets, a ring with fewer than 3 vertices, or a
        NaN/infinite coordinate.  Also for offsets that do not exactly cover
        the vertex arrays (``offsets[0] != 0``, or ``offsets[-1]`` short of or
        past ``len(lats)`` — the message names which endpoint failed),
        ``order`` outside 1-29, mismatched ``lats``/``lons`` lengths, or both
        ``tolerance`` and ``max_cells`` given.

    Warns
    -----
    UserWarning
        If ``max_cells`` is below the minimum needed to represent some
        polygon; the warning reports how many polygons were raised and names
        the lowest-index one.

    See Also
    --------
    mortie.coverage.morton_coverage_moc : the scalar (one polygon /
        one ring-set) form.

    Examples
    --------
    >>> import mortie, numpy as np
    >>> lats = np.array([40.0, 50.0, 45.0, 10.0, 20.0, 15.0])
    >>> lons = np.array([-120.0, -120.0, -110.0, -80.0, -80.0, -70.0])
    >>> values, off = mortie.polygons_to_morton_mocs(lats, lons, [0, 3, 6], order=6)
    >>> first = values[off[0]:off[1]]   # MOC of the first triangle
    """
    if tolerance is not None and max_cells is not None:
        raise ValueError("pass at most one of tolerance / max_cells")
    lats = np.ascontiguousarray(np.asarray(lats, dtype=np.float64).ravel())
    lons = np.ascontiguousarray(np.asarray(lons, dtype=np.float64).ravel())
    offsets = np.ascontiguousarray(np.asarray(offsets, dtype=np.int64).ravel())
    tol_rad = None if tolerance is None else np.radians(float(tolerance))
    values, out_offsets = _rustie.rust_polygons_coverage_mocs(
        lats, lons, offsets, order, tol_rad, max_cells, normalize
    )
    return np.asarray(values), np.asarray(out_offsets)


def from_wkbs(blobs, order=18, tolerance=None, max_cells=None, normalize=True):
    """Cover many WKB blobs with one call -- ragged MOCs out, no backend.

    The batch sibling of :func:`from_wkb` (issue #157) and the plural twin its
    name marks: **one MOC per input blob** (many→many), against the many→one
    union :func:`from_wkb` performs over the rings *inside* one blob.  The
    whole column crosses the Python/Rust boundary once, and Rust parses and
    covers the blobs in parallel with the GIL released — so the per-call fixed
    cost that dominates a Python loop over half a million footprints is paid
    once.  Result ``i`` is byte-identical to
    ``from_wkb(blobs[i], order=order, moc=True, ...)``.

    Memory: a chunk ends at 2048 blobs **or 64 MiB, whichever comes first**,
    and peak is the returned ``values`` array plus **one chunk of copied input
    bytes** (the copy is mandatory — a Python ``bytes`` buffer is GIL-bound and
    cannot cross into the parallel region) plus one chunk of in-flight covers.
    Neither the whole column's bytes nor every blob's cover is ever resident at
    once, and that holds **for every input spelling and every blob size**:
    non-``bytes`` entries are coerced inside the chunk, so their copy dies with
    it, and the byte budget stops 2048 fat geometries from making "one chunk"
    mean gigabytes.  Measured on the 555,867-blob ATL03 v007 corpus (276.7 MiB
    of WKB, 167.3 MiB of result, order 6), peak growth over the resident
    column:

    ==========================  ==========  =========
    input spelling              peak        × result
    ==========================  ==========  =========
    ``list[bytes]``             178.7 MiB   1.07
    numpy object array          179.4 MiB   1.07
    hex ``str``                 178.9 MiB   1.07
    ``bytearray``               178.9 MiB   1.07
    ``memoryview``              179.4 MiB   1.07
    ``uint8`` array             221.2 MiB   1.32
    arrow buffer slices         179.3 MiB   1.07
    ==========================  ==========  =========

    On the fat end, 3,000 Antarctic-basin blobs (1.25 MiB each, a 3.7 GiB
    column) peak at 610-634 MiB, against 3,120 MiB when the chunk was bounded
    by blob count alone — and most of what is left is the in-flight cover
    work, not the copy.

    Parameters
    ----------
    blobs : sequence
        One WKB/EWKB geometry per entry.  Each entry takes exactly what
        :func:`from_wkb` takes — ``bytes``, a hex ``str``, or any
        one-byte-item buffer (see :func:`_wkb_bytes`); the batch narrows
        nothing.  A list of ``bytes`` or a numpy object array (what
        ``pandas``/``pyarrow`` hand back for a binary column) both work as
        they are.  **Byte buffers are first-class, not merely tolerated**: a
        buffer column costs the same peak a ``bytes`` column does (the table
        above), so zero-copy ``memoryview`` slices of an Arrow column's value
        buffer are as cheap an input as ``bytes``.  Cutting those slices out
        of a pyarrow column correctly is the hard part, and
        :func:`mortie.arrow.from_wkbs` now does it for you — see Notes.
    order : int, optional
        Finest HEALPix order (1-29), shared by every blob.  Default 18.
    tolerance : float, optional
        Stop refining a boundary cell at this angular radius in **degrees** —
        :func:`mortie.morton_coverage_moc`'s ``tolerance``, applied as a
        **single shared setting**, mutually exclusive with ``max_cells``.
    max_cells : int, optional
        Per-blob cell budget, shared by every blob.  A budget below some
        blob's representable floor is raised for that blob (soft target, as
        in the scalar path) and one summary warning is emitted.
    normalize : bool, optional
        Ring-orientation handling, identical in meaning to
        :func:`from_wkb`'s ``normalize``.  Default ``True``.

    Returns
    -------
    values : numpy.ndarray
        Every blob's morton MOC words concatenated (``uint64``).
    out_offsets : numpy.ndarray
        ``int64`` arrow list offsets into ``values``, length
        ``len(blobs) + 1``; blob ``i``'s MOC is
        ``values[out_offsets[i]:out_offsets[i+1]]``.  ``out_offsets[0]`` is
        always 0 and ``out_offsets[-1]`` is always ``len(values)``.

    Raises
    ------
    ValueError
        Fail-fast, naming the **lowest-index** offending blob (e.g.
        ``blob 4217: truncated WKB ...``): a malformed or truncated blob, an
        unclosed polygon ring, an unsupported or empty geometry, a ring with
        fewer than 3 vertices, or a NaN/infinite coordinate.  **Linear
        geometry is refused by index** — a LineString cover is one array per
        line, which has no single-MOC-per-blob spelling; use
        :func:`from_wkb` for those.  Also for ``order`` outside 1-29, both
        ``tolerance`` and ``max_cells`` given, or an invalid hex string.
    TypeError
        Naming the offending index, for an entry that is neither a string nor
        a buffer of bytes.

    Notes
    -----
    Two ordered gates, as in :func:`mortie.polygons_to_morton_mocs`: the input
    contract is screened by a serial pre-pass over the whole sequence, then
    the blobs are parsed and covered.  Each gate reports its own lowest-index
    offender, so a ``TypeError`` at a high index does surface ahead of a
    malformed blob at a lower one — the pre-pass is an earlier gate, not a
    competing one.  The pre-pass **validates without retaining**
    (``_wkb_bytes(..., materialize=False)``): it applies the identical accept
    list — an invalid hex string is still caught here, ahead of any parse
    error — but keeps the entries as they came, so a column in a non-``bytes``
    spelling is not duplicated for the duration of the call.

    Feeding a **pyarrow** column has a typed entry point of its own —
    :func:`mortie.arrow.from_wkbs` (issue #163) — and that is what to call:
    ``marrow.from_wkbs(column, order=...)`` returns exactly this pair.  Do
    not improvise the extraction.  Four traps sit between a column and its
    blobs.  **Three are silent** — they yield different, valid-looking data
    rather than an error: a parquet column reads back as a ``ChunkedArray``,
    which has no ``.buffers()`` at all; ``slice`` and ``take`` are zero-copy
    metadata, so a chunk's buffers belong to the *original* array and must be
    indexed from ``chunk.offset``; and a ``large_binary`` column's offsets are
    ``int64``, not ``int32``.  The fourth is a **wrong diagnosis**: a null
    entry spans zero bytes, so it arrives as an empty blob and this function
    reports it as a truncated geometry rather than as a missing one.  The
    skin handles all four, and hands this function zero-copy ``memoryview``
    slices.

    ``from_wkbs(column.to_pylist(), ...)`` is also right on every one of
    those cases and needs no pyarrow-typed call, but its cost is real: on the
    555,867-row ATL03 v007 WKB column (290.1 MB of payload) ``to_pylist()``
    peaks at **~322 MB** of Python ``bytes`` objects — the per-object
    overhead on top of the payload, plus the list — against **~112 MB** for
    the skin's views of the same column.

    Warns
    -----
    UserWarning
        If ``max_cells`` is below the minimum needed to represent some blob;
        the warning reports how many were raised and names the lowest-index
        one.

    See Also
    --------
    mortie.geometry.from_wkb : the scalar (one blob) form, and the input
        contract in full.

    Examples
    --------
    >>> import mortie                                    # doctest: +SKIP
    >>> values, off = mortie.from_wkbs(wkb_column, order=8)   # doctest: +SKIP
    >>> first = values[off[0]:off[1]]   # the first blob's MOC
    """
    from . import _rustie

    if tolerance is not None and max_cells is not None:
        raise ValueError("pass at most one of tolerance / max_cells")
    # Serial screening pass, in index order, so the lowest-index bad entry is
    # what a caller sees -- the same fail-fast rule the Rust side applies to
    # parse/cover failures.  It *validates* rather than coerces: the entries
    # are kept as they came, so this costs a list of pointers whatever spelling
    # the column is in, and the byte-producing coercion happens per chunk on
    # the Rust side, where it is released with the chunk (issue #157).
    entries = []
    for i, blob in enumerate(blobs):
        try:
            _wkb_bytes(blob, materialize=False)
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"blob {i}: {exc}") from exc
        entries.append(blob)
    tol_rad = None if tolerance is None else np.radians(float(tolerance))
    values, out_offsets = _rustie.rust_wkbs_coverage_mocs(
        entries, _wkb_bytes, order, tol_rad, max_cells, normalize
    )
    return np.asarray(values), np.asarray(out_offsets)


def mocs_to_orders(values, offsets, order, max_cells=_FLAT_COVER_WARN_THRESHOLD):
    """Densify many independent MOCs to a flat order in one call.

    The batch sibling of :func:`moc_to_order` (issue #156): the ragged MOC set
    crosses the Python/Rust boundary **once**, the GIL is released for the whole
    batch, and Rust parallelizes across MOCs — so the per-call fixed cost that
    dominates a Python loop over half a million covers is paid once.  Result
    ``i`` is byte-identical to :func:`moc_to_order` on MOC ``i`` alone.

    Input and output are ragged arrays in arrow list layout, **the same pair
    :func:`polygons_to_morton_mocs` returns** — so the two chain with no
    marshalling::

        cells, off = mortie.polygons_to_morton_mocs(lats, lons, off_in, order=8)
        flat, flat_off = mortie.mocs_to_orders(cells, off, 8)

    MOCs are densified in chunks and each chunk is copied into the ragged output
    as it lands, so the per-MOC flat lists never all coexist — not the ~2.5x of
    holding every one of them to concatenate at the end.  Peak is then **the
    input copy + the result + one chunk**: the binding copies ``values`` and
    ``offsets`` before releasing the GIL (a borrowed numpy slice cannot cross
    ``allow_threads``), so the input is a full second resident array for the
    duration.  Densifying, that copy is noise — measured 1.16x of the returned
    array for 100k MOCs at order 8 → 10, 1.18x for 250k at 11 → 11.  Coarsening
    it is the whole of the peak: 250k order-11 MOCs down to order 4 is a 5.3 MiB
    result behind a 317.5 MiB peak (60x), essentially the 304.3 MiB input copy.
    Size a worker off ``input + result``, not the result alone.

    Parameters
    ----------
    values : array_like
        Flat ``uint64`` morton words, all MOCs concatenated.  Mixed orders
        allowed within each MOC, as in the scalar form.
    offsets : array_like
        ``int64`` arrow list offsets: MOC ``i`` spans
        ``[offsets[i], offsets[i + 1])``.  ``len(offsets) - 1`` MOCs.  The
        offsets must **exactly cover** ``values`` — ``offsets[0] == 0`` and
        ``offsets[-1] == len(values)`` — so a sliced arrow array must be
        re-based before it gets here; anything else is an error naming the
        endpoint that failed.  An empty MOC (``offsets[i] == offsets[i + 1]``)
        is legal and densifies to an empty slot.
    order : int
        Target HEALPix order (0-29) to densify to, shared by every MOC — the
        same domain :func:`moc_to_order` takes, order 0 included (it coarsens
        each MOC to the base cells it touches).
    max_cells : int or None, optional
        Pre-emptive budget on the densified flat cell count, applied **per
        MOC** exactly as :func:`moc_to_order` applies it to its one input
        (default ``1 << 20``).  A MOC whose estimate exceeds the budget raises
        :class:`ValueError` naming the **lowest-index** offending MOC, from a
        serial pre-pass — so the refusal costs no densify allocation, and it is
        the whole call that refuses, not that MOC alone.  Pass ``None`` to opt
        out.  Coerced with ``int()`` so the spellings the scalar accepts by
        plain comparison — a float budget such as ``mem_bytes / 8``, or one at
        or past ``2 ** 64`` — behave the same here rather than hitting the
        binding's ``u64`` (flooring a float matches the scalar exactly, since
        the estimate it is compared against is an integer, and a budget past
        ``2 ** 64`` cannot be exceeded by the saturating estimate either way).
        A negative budget is a :class:`ValueError`, as it is in the scalar,
        where every MOC over-runs it.

    Returns
    -------
    values : numpy.ndarray
        All MOCs' flat cells at ``order`` concatenated (``uint64``).
    out_offsets : numpy.ndarray
        ``int64`` arrow list offsets into ``values``, length ``len(offsets)``;
        ``out_offsets[0]`` is always 0.

    Raises
    ------
    ValueError
        Fail-fast, naming the **lowest-index** offending MOC (e.g. ``moc 4217:
        moc_to_order would densify to ...``): a MOC over ``max_cells``, or
        non-monotone / out-of-bounds offsets.  Also for offsets that do not
        exactly cover ``values`` (``offsets[0] != 0``, or ``offsets[-1]`` short
        of or past ``len(values)`` — the message names which endpoint failed),
        or an ``order`` outside 0-29.

    See Also
    --------
    mortie.moc.moc_to_order : the scalar (one MOC) form.
    polygons_to_morton_mocs : the batch coverer whose output feeds this
        verbatim.

    Notes
    -----
    Each slice comes back **sorted and unique** — the same guarantee
    :func:`moc_to_order` gives — so a downstream ``np.unique`` over a slice is
    redundant work, and ``np.searchsorted`` applies directly.

    Examples
    --------
    >>> import mortie, numpy as np
    >>> lats = np.array([40.0, 50.0, 45.0, 10.0, 20.0, 15.0])
    >>> lons = np.array([-120.0, -120.0, -110.0, -80.0, -80.0, -70.0])
    >>> mocs, off = mortie.polygons_to_morton_mocs(lats, lons, [0, 3, 6], order=6)
    >>> flat, flat_off = mortie.mocs_to_orders(mocs, off, 6)
    >>> first = flat[flat_off[0]:flat_off[1]]   # flat cover of the first triangle
    """
    values = np.ascontiguousarray(np.asarray(values, dtype=np.uint64).ravel())
    offsets = np.ascontiguousarray(np.asarray(offsets, dtype=np.int64).ravel())
    if max_cells is not None:
        max_cells = int(max_cells)
        if max_cells < 0:
            raise ValueError(f"max_cells must be non-negative, got {max_cells}")
        # The per-MOC estimate saturates at u64::MAX, so a budget at or past it
        # can never be exceeded -- clamping keeps the scalar's answer instead of
        # raising OverflowError out of the binding.
        max_cells = min(max_cells, (1 << 64) - 1)
    out_values, out_offsets = _rustie.rust_mocs_to_orders(
        values, offsets, order, max_cells
    )
    return np.asarray(out_values), np.asarray(out_offsets)


def mocs_and(a, values, offsets):
    """Intersect one shared morton cover with many independent MOCs in one call.

    The 1 x N broadcast of :func:`mortie.moc.moc_and` (issue #173): one shared
    operand ``a`` against ``len(offsets) - 1`` ragged MOCs, crossing the
    Python/Rust boundary **once** with the GIL released while Rust parallelizes
    across MOCs.  Result ``i`` is byte-identical to
    ``moc_and(a, values[offsets[i]:offsets[i+1]])``.  Beyond the boundary
    amortization every batch twin shares, the broadcast has a structural win of
    its own: the scalar normalizes and re-encodes **both** operands on every
    call, so a Python loop rebuilds the shared operand's BMOC N times — here it
    is built once and borrowed by every item.  ``moc_and`` is commutative, so
    it does not matter which side of your loop was "the AOI": pass either
    operand as ``a``.

    An empty intersection keeps its slot (``out_offsets[i] ==
    out_offsets[i+1]``), as does every slot when ``a`` is empty — so the ragged
    output always agrees with :func:`mocs_intersect` on which items overlap.
    There is deliberately no ``max_cells``: the densify budget on
    :func:`mocs_to_orders` guards an exponential blow-up term, while an
    intersection is bounded by its inputs (the scalar set ops carry no budget
    either).

    Memory: MOCs are intersected in chunks and each chunk is copied into the
    ragged output as it lands, so peak is **the input copy + the result + one
    chunk** — the binding copies ``a``, ``values`` and ``offsets`` before
    releasing the GIL (a borrowed numpy slice cannot cross ``allow_threads``),
    so the input is a full second resident array for the duration, and because
    an intersection result is never larger than its inputs, that copy is the
    dominant term.  Measured over 100k ~4-cell granule MOCs against an order-8
    AOI cover (``benchmarks/measure_mocs_and.py --mem``): 3.8 MiB of ragged
    input, a 1.1 MiB result, 5-9 MiB of peak-RSS growth across repeated runs
    over the resident inputs for one ``mocs_and`` plus one ``mocs_intersect``
    call — the input copy plus the result plus a chunk.  (``ru_maxrss`` is a
    high-water mark, so the growth is a noisy lower bound, not an exact
    peak.)  Size a worker off ``input + result``, not the result alone.

    Parameters
    ----------
    a : array_like
        The shared morton cover (``uint64``, mixed order allowed).  Empty is
        legal: every result slot is then empty.
    values : array_like
        Flat ``uint64`` morton words, all MOCs concatenated.  Mixed orders
        allowed within each MOC, as in the scalar form.
    offsets : array_like
        ``int64`` arrow list offsets: MOC ``i`` spans
        ``[offsets[i], offsets[i + 1])``.  The offsets must **exactly cover**
        ``values`` — ``offsets[0] == 0`` and ``offsets[-1] == len(values)`` —
        so a sliced arrow array must be re-based before it gets here; anything
        else is an error naming the endpoint that failed.  An empty MOC is
        legal and intersects to an empty slot.

    Returns
    -------
    values : numpy.ndarray
        All intersections concatenated (``uint64``), each slice sorted and
        compacted exactly as :func:`mortie.moc.moc_and` returns it.
    out_offsets : numpy.ndarray
        ``int64`` arrow list offsets into ``values``, length ``len(offsets)``;
        ``out_offsets[0]`` is always 0.

    Raises
    ------
    ValueError
        Fail-fast, naming the **lowest-index** offending MOC: non-monotone or
        out-of-bounds offsets, or offsets that do not exactly cover ``values``
        (the message names which endpoint failed).

    See Also
    --------
    mortie.moc.moc_and : the scalar (one pair) form.
    mocs_intersect : the allocation-free predicate over the same broadcast.
    mocs_to_orders : densifies the surviving intersections, chaining on this
        output verbatim.

    Examples
    --------
    >>> import mortie, numpy as np
    >>> aoi = np.asarray(mortie.norm2mort([0], [0], 2), dtype=np.uint64)
    >>> items = np.asarray(mortie.norm2mort([0, 200], [0, 0], 4), dtype=np.uint64)
    >>> hit, off = mortie.mocs_and(aoi, items, [0, 1, 2])
    >>> [int(off[i + 1] - off[i]) for i in range(2)]   # item 0 overlaps, 1 not
    [1, 0]
    """
    a = np.ascontiguousarray(np.asarray(a, dtype=np.uint64).ravel())
    values = np.ascontiguousarray(np.asarray(values, dtype=np.uint64).ravel())
    offsets = np.ascontiguousarray(np.asarray(offsets, dtype=np.int64).ravel())
    out_values, out_offsets = _rustie.rust_mocs_and(a, values, offsets)
    return np.asarray(out_values), np.asarray(out_offsets)


def mocs_intersect(a, values, offsets):
    """Test which of many MOCs intersect one shared cover, materializing nothing.

    The predicate twin of :func:`mocs_and` and the batch form of
    :func:`mortie.moc.moc_intersects` (issue #173): ``out[i]`` is exactly
    ``moc_intersects(a, values[offsets[i]:offsets[i+1]])``, i.e. whether
    :func:`mocs_and`'s slot ``i`` would be non-empty — without building it.
    Per item this is a range-overlap walk over the normalized covers, never a
    BMOC build or result encode — no intersection is materialized, and the
    only per-item allocation is that item's normalize scratch — and it
    **short-circuits on the first overlap**, something the materializing form
    cannot do.  The shared operand is normalized and range-decoded once for
    the whole batch.  Proving a *miss* still takes the full walk, so the win
    on non-overlapping items over ``moc_and(...).size`` is the skipped
    build/encode/allocation, not the short-circuit.

    Compaction-safe per item, by construction: each item is tested for
    geometric overlap against ``a``, so this cannot be (and is not) implemented
    by intersecting once and testing membership — which would silently drop
    dense regions that compact to a parent cell.

    Memory: no results are materialized; peak is the input copy the binding
    makes before releasing the GIL, one ``bool`` per MOC out, plus the
    in-flight items' normalize scratch (one chunk at most).

    Parameters
    ----------
    a : array_like
        The shared morton cover (``uint64``, mixed order allowed).  Empty is
        legal: every answer is then ``False``.
    values : array_like
        Flat ``uint64`` morton words, all MOCs concatenated.  Mixed orders
        allowed within each MOC, as in the scalar form.
    offsets : array_like
        ``int64`` arrow list offsets: MOC ``i`` spans
        ``[offsets[i], offsets[i + 1])``.  The offsets must **exactly cover**
        ``values``, exactly as in :func:`mocs_and`.  An empty MOC is legal and
        answers ``False``.

    Returns
    -------
    numpy.ndarray
        ``bool`` array of ``len(offsets) - 1`` answers, one per MOC.

    Raises
    ------
    ValueError
        Fail-fast, naming the **lowest-index** offending MOC: non-monotone or
        out-of-bounds offsets, or offsets that do not exactly cover ``values``
        (the message names which endpoint failed).

    See Also
    --------
    mortie.moc.moc_intersects : the scalar (one pair) form.
    mocs_and : materializes the intersections this only tests.

    Examples
    --------
    >>> import mortie, numpy as np
    >>> aoi = np.asarray(mortie.norm2mort([0], [0], 2), dtype=np.uint64)
    >>> items = np.asarray(mortie.norm2mort([0, 200], [0, 0], 4), dtype=np.uint64)
    >>> mortie.mocs_intersect(aoi, items, [0, 1, 2]).tolist()
    [True, False]
    """
    a = np.ascontiguousarray(np.asarray(a, dtype=np.uint64).ravel())
    values = np.ascontiguousarray(np.asarray(values, dtype=np.uint64).ravel())
    offsets = np.ascontiguousarray(np.asarray(offsets, dtype=np.int64).ravel())
    return np.asarray(_rustie.rust_mocs_intersect(a, values, offsets))


def common_ancestors(values, offsets):
    """Reduce many groups of morton words to their common ancestors in one call.

    The batch sibling of :func:`common_ancestor` (issue #156): the whole ragged
    group set crosses the Python/Rust boundary **once**, the GIL is released for
    the batch, and Rust parallelizes across groups.  Result ``i`` is
    bit-identical to :func:`common_ancestor` on group ``i`` alone, the
    single-word case included (it comes back verbatim, kind preserved).

    Input is ragged in the arrow list layout :func:`polygons_to_morton_mocs`
    and :func:`mocs_to_orders` use; the **output is dense** — one ``uint64`` per
    group — because the reduction is many→one per group, so there are no output
    offsets to carry.

    The consumer this exists for is a per-worker inner loop, not a one-off:
    zagg's t-digest reduction runs ``for j in np.flatnonzero(~single):`` over
    every multi-member centroid (``stats/tdigest.py:198``) — once per centroid,
    per cell, per build **and** per fold, on every Lambda worker, at 65,536
    cells per shard in the shipped ATL03 configuration.  That module's own
    docstring already calls it "the same O(n) Python-loop shape issue #279
    removed".

    Memory: the binding copies ``values`` and ``offsets`` before releasing the
    GIL (a borrowed numpy slice cannot cross ``allow_threads``), so peak is
    **the input copy + the result + one 64 KiB chunk** of per-group outcomes
    **+ the reduction's own scratch**.  That last term is
    :func:`common_ancestor`'s internal buffer, 16 bytes per non-first word in
    the group being reduced, held for that whole reduction and one per group in
    flight — so it is ``min(threads, n_groups) * 16 * max_group_size`` bytes.
    It scales with the **largest single group**, not with the total word count.

    For small groups it is invisible and the input copy is the peak: over 5M
    groups of 3 order-9 words, 152.6 MiB of input, a 38.1 MiB result, a 191.9
    MiB peak — 1.01x the ``input + result`` model, **5.0x the result alone**,
    and under a kilobyte of scratch.  For large groups it dominates: 40 groups
    of 1M words peaks at 458.6 MiB against a 305.2 MiB model (**1.50x**), and a
    single 20M-word group at 460.0 MiB against 152.7 MiB (**3.01x**) — in both
    cases the excess is the scratch term to within 2 MiB.  Size a worker off
    ``input + result`` for many small groups, and off the largest group when
    groups are large.

    Parameters
    ----------
    values : array_like
        Flat ``uint64`` morton words, all groups concatenated.  Mixed orders
        allowed within a group, as in the scalar form; every word in a group
        must share one HEALPix base cell.
    offsets : array_like
        ``int64`` arrow list offsets: group ``i`` spans
        ``[offsets[i], offsets[i + 1])``, so there are ``len(offsets) - 1``
        groups.  The offsets must **exactly cover** ``values`` —
        ``offsets[0] == 0`` and ``offsets[-1] == len(values)`` — so a sliced
        arrow array must be re-based before it gets here; anything else is an
        error naming the endpoint that failed.  Unlike the ragged-output
        batches, an **empty group is an error**, not an empty slot: a
        many→one reduction over no words has no answer, exactly as the scalar
        refuses empty input.

    Returns
    -------
    numpy.ndarray
        ``uint64`` array of ``len(offsets) - 1`` ancestor words, one per group.

    Raises
    ------
    ValueError
        Fail-fast, naming the offending group (e.g. ``group 4217: inputs span
        multiple base cells ...``): a *domain* failure — a group that is empty,
        holds an empty/invalid word, or spans more than one base cell — or a
        *layout* failure — non-monotone / out-of-bounds offsets, or offsets that
        do not exactly cover ``values`` (the message names which endpoint
        failed).  The index named is the **lowest-index** offender within its
        kind.  Layout is checked for the whole batch first, so a layout failure
        at a high index is reported ahead of a domain failure at a low one; that
        ordering is deliberate, since the group indices a domain error is
        reported by are themselves read out of ``offsets``.

    See Also
    --------
    mortie.moc.common_ancestor : the scalar (one group) form.
    mortie.moc.split_base_cells : partitions a mixed-base-cell set into
        groups this accepts.

    Examples
    --------
    Two groups of order-5 siblings reduce to their two order-4 parents:

    >>> import mortie, numpy as np
    >>> kids = np.concatenate([
    ...     np.asarray(mortie.norm2mort([11 * 4 + s for s in range(4)], [0] * 4, 5)),
    ...     np.asarray(mortie.norm2mort([7 * 4 + s for s in range(4)], [3] * 4, 5)),
    ... ])
    >>> got = mortie.common_ancestors(kids, [0, 4, 8])
    >>> [int(got[0]), int(got[1])] == [
    ...     int(mortie.norm2mort(11, 0, 4)), int(mortie.norm2mort(7, 3, 4))
    ... ]
    True
    """
    values = np.ascontiguousarray(np.asarray(values, dtype=np.uint64).ravel())
    offsets = np.ascontiguousarray(np.asarray(offsets, dtype=np.int64).ravel())
    return np.asarray(_rustie.rust_common_ancestors(values, offsets))


def children_of(words, order, max_cells=None):
    """Refine many parent words to their children at ``order``, in one call.

    The batch sibling of :func:`generate_morton_children` (issue #156), whose
    wrapper coerces its input to a *single* parent: the whole parent array
    crosses the Python/Rust boundary **once**, the GIL is released for the
    batch, and Rust parallelizes across parents.  Row ``i`` is bit-identical to
    :func:`generate_morton_children` on ``words[i]`` alone.

    Every parent must sit at one shared order ``p <= order``, so each yields
    exactly ``4**d`` children for ``d = order - p`` and the **result is dense**
    — an ``(n, 4**d)`` matrix, not a ragged pair.  That is not a new
    restriction: consumers already assume it, because the loop this replaces
    ends in ``np.stack``, which raises on rows of unequal width (moczarr
    ``dggs.py:310``, whose comment at ``dggs.py:302-305`` records the gap this
    closes — *"there is still no vectorized many-parent children kernel"*).
    zagg calls the scalar the same way per sub-chunk on every worker
    (``grids/healpix.py:199``) and per shard in the shardmap reprojection
    (``catalog/shardmap.py:708``).

    Memory: the result is the whole of it, and it is ``n * 4**d * 8`` bytes —
    refining 100k parents by 5 orders is 100k x 1024 x 8 B = 819 MB.  The
    binding copies ``words`` before releasing the GIL (a borrowed numpy slice
    cannot cross ``allow_threads``), which adds ``n * 8`` bytes, and the block
    is allocated once at its exact final size, so there is no growth-realloc
    transient and no ragged assembly copy — peak is **input copy + result**
    plus a fixed 64 KiB chunk of per-word outcomes.  Measured over 1M order-6
    parents refined to order 9: 7.6 MiB of input, a 488.3 MiB result, a 497.1
    MiB peak — 1.00x that model.

    The result block is allocated **fallibly**, so a size the allocator refuses
    is a catchable ``ValueError`` naming the byte count rather than a process
    abort — matching the ``MemoryError`` the ``np.stack`` loop this replaces
    raises at those sizes.  That is not a budget, though: an allocation an
    overcommitting OS accepts and then cannot back still ends in a kill, exactly
    as it does for the scalar loop (numpy accepts the same over-RAM request).
    Only a policy ceiling refuses those, which is what ``max_cells`` is for.

    ``max_cells`` is **opt-in** — ``None`` by default, the opposite of
    :func:`moc_to_order`'s always-on budget.  The difference is deliberate and
    is about predictability, not about one op being safer.  ``moc_to_order``
    defaults its budget on because a densify explodes from a tiny input
    (``Σ 4**(order - depth)``, issue #80) and the caller cannot cheaply predict
    the output.  Here the output is exactly ``n * 4**d`` cells, computable from
    the arguments before the call, so a default guard would refuse calls the
    caller already knows are fine.  Same parameter name, opposite default, for a
    stated reason.  The ``None`` polarity inverts with it: for
    :func:`moc_to_order` ``None`` *disables* a default budget, here it means
    there is no budget to begin with.

    Parameters
    ----------
    words : array_like
        Parent packed morton words (``uint64``), all at one HEALPix order no
        finer than ``order``.
    order : int
        Target HEALPix order (0-29) for the children, shared by every parent.
        ``order`` equal to the parents' own order is legal and gives back a
        ``(n, 1)`` block of the parents **verbatim** — the same identity the
        scalar returns, which is what preserves a point word's kind.
    max_cells : int or None, optional
        Opt-in budget on the result's cell count, ``len(words) * 4**d``.  When
        set, a result over it raises :class:`ValueError` **before** anything is
        allocated, so a worker sized for a known memory ceiling fails catchably
        instead of being killed by the OS mid-write.  Default ``None`` — no
        budget, behaviour exactly as if the parameter did not exist.  Checked
        ahead of the internal element-count overflow guard, so an explicit
        budget is what answers even for a request too large to represent.

    Returns
    -------
    numpy.ndarray
        ``uint64`` array of shape ``(len(words), 4**d)``, row-major: row ``i``
        holds ``words[i]``'s children at ``order``, ascending.  An empty
        ``words`` gives ``(0, 1)`` — with no parent to read an order from there
        is no ``d`` to derive, and 1 is the only width that is not a claim
        about a block with no rows.  Deriving it instead would mean a
        ``parent_order=`` argument making callers repeat, in the empty case
        alone, what the data itself carries in every other case; consumers that
        need a typed empty already special-case it, the way moczarr's
        ``dggs.py`` builds ``(0, 4**(level - order))`` on its own empty branch
        because it knows its source order at that point.  So ``(0, 1)`` is the
        honest shape for "no rows, width unknown", and the width belongs in the
        consumer-side special case (issue #156, ruled).

    Raises
    ------
    ValueError
        Fail-fast, naming the **lowest-index** offending word (e.g.
        ``word 4217: is at order 9, finer than the requested order 7``): an
        empty/invalid word, one finer than ``order``, or one at a different
        order from ``words[0]``.  Also if ``order`` is outside 0-29, if
        ``max_cells`` is set and the ``n * 4**d`` result exceeds it, or if the
        result is a size the allocator refuses (``... needs N bytes; allocation
        failed``).

    See Also
    --------
    mortie.orders.generate_morton_children : the scalar (one parent) form.
    mortie.orders.clip2order : the coarsening direction (elementwise, already
        vectorized).

    Examples
    --------
    >>> import mortie, numpy as np
    >>> parents = np.asarray(mortie.norm2mort([11, 7], [0, 3], 4), dtype=np.uint64)
    >>> kids = mortie.children_of(parents, 6)
    >>> kids.shape
    (2, 16)
    >>> np.array_equal(kids[0], mortie.generate_morton_children(int(parents[0]), 6))
    True

    An opt-in budget refuses an oversized refinement before it is allocated:

    >>> mortie.children_of(parents, 14, max_cells=1 << 20)
    ... # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: children_of would generate 2097152 cells ... max_cells=1048576...
    """
    words = np.ascontiguousarray(np.asarray(words, dtype=np.uint64).ravel())
    # Range-checked here so an out-of-range order is a catchable ValueError
    # rather than the binding's u8 OverflowError (the issue #108 posture).
    if not 0 <= order <= 29:
        raise ValueError(f"Order must be between 0 and 29, got {order}")
    if max_cells is not None:
        max_cells = int(max_cells)
        if max_cells < 0:
            raise ValueError(f"max_cells must be non-negative, got {max_cells}")
        # The kernel compares in u128, so a budget past u64::MAX can never be
        # exceeded -- clamping keeps that answer instead of raising
        # OverflowError out of the binding (the mocs_to_orders spelling).
        max_cells = min(max_cells, (1 << 64) - 1)
    return np.asarray(_rustie.rust_children_of(words, int(order), max_cells))

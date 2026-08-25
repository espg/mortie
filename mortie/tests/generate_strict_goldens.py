"""Generate valid-path goldens for the strict-validation sweep (issue #194).

Captured at ``4900a7e`` -- the commit *before* the shared strict validators
were adopted family-wide -- so ``test_strict_validation.py`` can pin that the
touched entry points answer byte-identically for valid input before and after
the posture change.  Regenerating on a later commit only re-asserts the
current answers; the committed JSON is the pre-change record.

**What the capture covers, and what it does not.**  Every word/offset seam
this PR validated that answers with plain numbers, across every phase:
the ``_moc`` operators (both arms), ``batch``'s ragged forms, ``orders``,
``convert`` (``mort2norm`` / ``mort2geo`` / ``mort2bbox`` / ``mort2polygon``,
and the phase-5 UNIQ/normed intakes ``norm2mort`` / ``norm2uniq`` /
``unique2parent`` / ``uniq2geo``),
``buffer``, ``prefix_trie``, ``Moc``, ``geometry.from_wkb(offsets=)``, and the
whole **toc** family -- whose ``_as_u64`` / ``_as_offsets`` moved house in
phase 1, so its valid answers are pinned here rather than left to prose.

Two deliberate omissions.  ``to_geometry`` / ``to_wkb`` / ``to_wkt`` need the
shapely backend, which is a test extra rather than a runtime dependency; this
generator stays numpy-only so the golden test never turns on an optional
install, and those three are pinned instead by ``test_geometry.py``'s own
behavior suite plus the refusal rows in ``test_strict_validation.py``.  And
``time2toc([])`` is *not* captured: the untyped-empty acceptance deliberately
changed its answer from a refusal to an empty cover (see the CHANGELOG and
the PR's Questions for review), so a golden there would pin the one valid
path this PR does not claim is unchanged.

Run from the repo root::

    python mortie/tests/generate_strict_goldens.py

Writes ``mortie/tests/data/strict_validation_goldens.json``.
"""

import json
import pathlib

import numpy as np

import mortie

OUT = pathlib.Path(__file__).parent / "data" / "strict_validation_goldens.json"


def _ints(arr):
    """Flatten an array to a JSON-serializable list of Python ints.

    Parameters
    ----------
    arr : array_like
        Integer array of any shape.

    Returns
    -------
    list of int
        ``arr`` raveled, as plain ints.
    """
    return [int(x) for x in np.asarray(arr).ravel()]


def capture():
    """Capture one answer per touched entry point (see the module docstring).

    Numpy-only by design -- no optional backend -- so the two shapely-gated
    emit surfaces are covered elsewhere.

    Returns
    -------
    dict
        Golden entry name -> JSON-serializable answer.
    """
    # Mixed-order words across northern and southern base cells: base cells
    # 7..11 set bit 63 (spec section 1, "Unsigned storage"), so the set pins
    # that words >= 2**63 stay byte-identical through the validators.
    parents4 = np.asarray(mortie.norm2mort([11, 7, 3], [0, 8, 11], 4))
    kids5 = np.asarray(
        mortie.norm2mort([11 * 4 + s for s in range(4)], [0] * 4, 5))
    kids5_south = np.asarray(
        mortie.norm2mort([7 * 4 + s for s in range(4)], [9] * 4, 5))
    cover_a = np.asarray(mortie.compress_moc(np.concatenate([parents4, kids5])))
    cover_b = np.asarray(mortie.compress_moc(np.concatenate([kids5, kids5_south])))
    ragged = np.concatenate([cover_a, cover_b])
    ragged_off = [0, cover_a.size, cover_a.size + cover_b.size]
    groups = np.concatenate([kids5, kids5_south])
    groups_off = [0, 4, 8]

    tri_lats = [0.0, 0.0, 8.0]
    tri_lons = [0.0, 8.0, 0.0]

    # A two-blob WKB column for the from_wkb(offsets=) seam.  Hard-coded hex
    # rather than built with shapely, so this stays a numpy-only capture.
    wkb_a = bytes.fromhex(
        "010300000001000000040000000000000000000000000000000000000000000000"
        "00002040000000000000000000000000000000000000000000002040000000000000"
        "0000000000000000000000000000000000")
    wkb_b = bytes.fromhex(
        "0103000000010000000400000000000000000034400000000000003440000000000"
        "0003C40000000000000344000000000000034400000000000003C40000000000000"
        "34400000000000003440")
    wkb_buf = np.frombuffer(wkb_a + wkb_b, dtype=np.uint8)
    wkb_off = [0, len(wkb_a), len(wkb_a) + len(wkb_b)]

    # Toc words: phase 1 moved _as_u64/_as_offsets out from under _toc.py, so
    # the toc family's valid answers belong in the pre-change record too.
    t_ns = np.asarray([10**15, 2 * 10**15, 3 * 10**15], dtype=np.uint64)
    toc_words = np.asarray(mortie.time2toc(t_ns))
    toc_off = [0, 1, 3]
    poly_vals, poly_off = mortie.polygons_to_morton_mocs(
        tri_lats, tri_lons, [0, 3], order=6)

    # UNIQ ids for the phase-5 intakes: the same three order-4 cells as
    # `parents4` plus two order-5 children, so `uniq2geo`'s group-by-order
    # dispatch runs over a genuinely mixed-resolution column.
    uniq4 = np.asarray(mortie.norm2uniq(
        np.asarray([11, 7, 3]), np.asarray([0, 8, 11]), 4))
    uniq5 = np.asarray(mortie.norm2uniq(
        np.asarray([11 * 4, 11 * 4 + 3]), np.asarray([0, 0]), 5))
    uniq_mixed = np.concatenate([uniq4, uniq5])

    and_vals, and_off = mortie.moc_and(cover_a, ragged, offsets=ragged_off)
    to7_vals, to7_off = mortie.moc_to_order(ragged, 7, offsets=ragged_off)
    anc_ragged = mortie.common_ancestor(groups, offsets=groups_off)
    children = mortie.generate_morton_children(parents4[:2], 6)

    g = {
        "words": _ints(cover_a),
        "words_b": _ints(cover_b),
        "compress_moc": _ints(cover_a),
        "moc_to_order": _ints(mortie.moc_to_order(cover_a, 7)),
        "moc_to_order_ragged": [_ints(to7_vals), _ints(to7_off)],
        "moc_or": _ints(mortie.moc_or(cover_a, cover_b)),
        "moc_and": _ints(mortie.moc_and(cover_a, cover_b)),
        "moc_and_ragged": [_ints(and_vals), _ints(and_off)],
        "moc_intersects": bool(mortie.moc_intersects(cover_a, cover_b)),
        "moc_intersects_ragged": [
            bool(x) for x in
            mortie.moc_intersects(cover_a, ragged, offsets=ragged_off)],
        "moc_minus": _ints(mortie.moc_minus(cover_a, cover_b)),
        "moc_xor": _ints(mortie.moc_xor(cover_a, cover_b)),
        "moc_min": _ints(np.atleast_1d(mortie.moc_min(kids5))),
        "moc_not": _ints(mortie.moc_not(cover_a)),
        "moc_not_domain": _ints(mortie.moc_not(kids5[:2], domain=cover_a)),
        "common_ancestor": _ints(np.atleast_1d(mortie.common_ancestor(kids5))),
        "common_ancestor_ragged": _ints(anc_ragged),
        "split_base_cells_values": [
            _ints(part) for part in mortie.split_base_cells(ragged)],
        "polygons_to_morton_mocs": [_ints(poly_vals), _ints(poly_off)],
        "generate_morton_children_scalar": _ints(
            mortie.generate_morton_children(int(parents4[0]), 6)),
        "generate_morton_children_array": _ints(children),
        "clip2order": _ints(mortie.clip2order(4, kids5)),
        "orders_of": _ints(mortie.orders_of(ragged)),
        "is_point": [bool(x) for x in np.atleast_1d(mortie.is_point(ragged))],
        "infer_order_from_morton": int(mortie.infer_order_from_morton(kids5)),
        "validate_morton": bool(mortie.validate_morton(ragged)),
        "mort2norm_normed": _ints(mortie.mort2norm(kids5)[0]),
        "mort2norm_parent": _ints(mortie.mort2norm(kids5)[1]),
        "mort2norm_order": _ints(np.atleast_1d(mortie.mort2norm(kids5)[2])),
        "mort2geo": [
            [round(float(v), 12) for v in axis.ravel()]
            for axis in mortie.mort2geo(kids5)],
        "morton_buffer": _ints(mortie.morton_buffer(kids5, k=1)),
        "morton_buffer_meters": _ints(
            mortie.morton_buffer_meters(kids5, width_m=50000.0)),
        "split_children_roots": sorted(
            c.characteristic
            for c in mortie.split_children(ragged, max_depth=2)),
        "moc_object_and": _ints(
            (mortie.Moc(cover_a) & mortie.Moc(cover_b)).words),
        # -- phase-3 convert surfaces the first capture missed --------------
        "mort2bbox": [
            [round(float(box[k]), 12)
             for k in ("west", "south", "east", "north")]
            for box in np.asarray(mortie.mort2bbox(kids5)).ravel()],
        "mort2polygon": [
            [[round(float(v), 12) for v in vertex] for vertex in ring]
            for ring in mortie.mort2polygon(kids5)],
        # -- the phase-5 UNIQ/normed intakes --------------------------------
        "norm2mort": _ints(parents4),
        "norm2uniq": _ints(uniq_mixed),
        "unique2parent": _ints(mortie.unique2parent(uniq_mixed)),
        "uniq2geo": [
            [round(float(v), 12) for v in axis.ravel()]
            for axis in mortie.uniq2geo(uniq_mixed)],
        # -- the phase-2 offsets seam in geometry.from_wkb ------------------
        "from_wkb_ragged": [
            _ints(part) for part in
            mortie.from_wkb(wkb_buf, order=6, offsets=wkb_off)],
        # -- the toc family, whose validators moved house in phase 1 --------
        "time2toc": _ints(toc_words),
        "span2toc": _ints(np.atleast_1d(
            mortie.span2toc(int(t_ns[0]), int(t_ns[-1])))),
        "toc2time": [_ints(axis) for axis in mortie.toc2time(toc_words)],
        "toc_reduce": _ints(np.atleast_1d(mortie.toc_reduce(toc_words))),
        "toc_reduce_ragged": [
            _ints(part) for part in
            mortie.toc_reduce(toc_words, offsets=toc_off)],
        "toc_normalize": _ints(mortie.toc_normalize(toc_words)),
        "toc_and": _ints(mortie.toc_and(toc_words, toc_words)),
        "toc_merge": _ints(np.atleast_1d(
            mortie.toc_merge(toc_words[0], toc_words[1]))),
        "from_gps_ns": _ints(mortie.from_gps_ns(t_ns)),
        "to_gps_ns": _ints(mortie.to_gps_ns(
            np.asarray([4.2 * 10**18, 4.3 * 10**18], dtype=np.uint64))),
        "to_datetime64": [
            str(v) for v in np.atleast_1d(mortie.to_datetime64(t_ns))],
    }
    return g


def main():
    """Write the captured answers to ``OUT``.

    Returns
    -------
    None
        Writes ``OUT`` as a side effect.
    """
    g = capture()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(g, indent=1) + "\n")
    print(f"wrote {OUT} ({len(g)} entries)")


if __name__ == "__main__":
    main()

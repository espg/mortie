"""Generate valid-path goldens for the strict-validation sweep (issue #194).

Captured at ``4900a7e`` -- the commit *before* the shared strict validators
were adopted family-wide -- so ``test_strict_validation.py`` can pin that
every entry point answers byte-identically for valid input before and after
the posture change.  Regenerating on a later commit only re-asserts the
current answers; the committed JSON is the pre-change record.

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


def main():
    """Capture one golden answer per touched entry point and write the JSON.

    Returns
    -------
    None
        Writes ``OUT`` as a side effect.
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
    poly_vals, poly_off = mortie.polygons_to_morton_mocs(
        tri_lats, tri_lons, [0, 3], order=6)

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
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(g, indent=1) + "\n")
    print(f"wrote {OUT} ({len(g)} entries)")


if __name__ == "__main__":
    main()

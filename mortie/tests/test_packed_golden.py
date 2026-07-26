"""
Golden fixtures for the packed-u64 ``decimal_morton`` contract (issue #48).

This is the Tier-0 "freeze the contract" deliverable: a committed snapshot of the
canonical packed word, its decode-through-kernel decimal repr, and the
``(nested, depth)`` bridge for orders 0-29 across all 12 base cells. Once shipped
it is frozen for mortie 1.x, so these values are pinned deliberately and any
change here is a deliberate (major-version) contract change.

Each (base cell, order) is pinned for **three** tuple variants -- ``all-1``
(every stored tuple 0 -> digit 1), ``all-4`` (every stored tuple 3 -> digit 4),
and a pseudo-random seed -- so the freeze exercises the digit extremes and, at
orders 28/29, the suffix-region tail tuples in both positions, not just one path
per cell.

What the tests guarantee differs by order range, and the fixture note says so:
``test_golden_fixture_matches_kernel`` is a *drift* detector (it regenerates from
the same kernel it asserts against, catching an accidental future change). The
independent *correctness* anchor is the legacy cross-check, which only exists for
orders 0-18 (the legacy ``fastNorm2Mort`` path tops out there); orders 19-29 have
no external oracle, so they are pinned to the kernel's natural extension and only
their structural shape (length / sign / 1-4 digits) is independently asserted.
"""

import json
import os

import numpy as np

from mortie import _rustie

MAX_ORDER = 29

_FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "packed_u64_golden.json"
)

# Tuple variants pinned per (base, order): digit extremes + a pseudo-random path.
_VARIANTS = ("all1", "all4", "seed")


def _variant_tuples(variant, order, seed):
    """Deterministic stored-tuple vector for a named variant (matches the JSON)."""
    n = max(order, 1)
    if variant == "all1":  # stored 0 -> digit 1
        return [0] * n
    if variant == "all4":  # stored 3 -> digit 4
        return [3] * n
    return [((seed * 2654435761 + i) % 4) for i in range(n)]


def _nested_from_tuples(base, tuples, order):
    within = 0
    for n in range(1, order + 1):
        within |= (int(tuples[n - 1]) & 3) << (2 * (order - n))
    return base * (1 << (2 * order)) + within


def _legacy_encode(order, normed, parent):
    """The retired legacy decimal Morton encoder (orders 0-18).

    Reproduced here so the converter's backward-compat cross-check can still
    synthesize old decimal values after the production encoder was removed in the
    issue #48 flip; ``rust_mi_from_legacy`` must still decode these.
    """
    num = 0
    for i in range(order, 0, -1):  # MSB tuple first -> highest decimal place
        bits = (normed >> (2 * (i - 1))) & 3
        num += (bits + 1) * 10 ** (i - 1)
    pow10 = 10 ** order
    if parent >= 6:
        num += (parent - 11) * pow10
        num = -num
        num -= 6 * pow10
    else:
        num += (parent + 1) * pow10
    return num


def _regenerate_records():
    """Rebuild the fixture records straight from the kernel bindings."""
    records = []
    for base in range(12):
        for order in range(MAX_ORDER + 1):
            for variant in _VARIANTS:
                tuples = _variant_tuples(variant, order, base + 1)
                nested = _nested_from_tuples(base, tuples, order)
                word = int(
                    _rustie.rust_mi_from_nested(
                        np.ascontiguousarray([nested], dtype=np.uint64), order
                    )[0]
                )
                w = np.asarray([word], dtype=np.uint64)
                rep = _rustie.rust_mi_decimal_repr(w)[0]
                n2, d2 = _rustie.rust_mi_to_nested(w)
                records.append(
                    {
                        "base_cell": base,
                        "order": order,
                        "variant": variant,
                        "tuples": [int(t) for t in tuples[:order]],
                        "nested": int(nested),
                        "word": word,
                        "decimal_repr": rep,
                        "decoded_base_cell": int(_rustie.rust_mi_base_cell_of(w)[0]),
                        "decoded_order": int(_rustie.rust_mi_order_of(w)[0]),
                        "roundtrip_nested": int(n2[0]),
                        "roundtrip_depth": int(d2[0]),
                    }
                )
    return records


def _load_fixture():
    with open(_FIXTURE_PATH) as f:
        return json.load(f)


def test_golden_fixture_matches_kernel():
    """The committed fixture must equal a fresh kernel regeneration, exactly.

    This is a drift detector, not a correctness oracle: it regenerates from the
    same kernel it pins, so it catches an accidental future change to the encode
    / repr, not an original logic error. Correctness is anchored by the legacy
    cross-check (orders 0-18) and the structural-shape check (all orders).
    """
    fixture = _load_fixture()
    assert fixture["max_order"] == MAX_ORDER
    expected = fixture["records"]
    actual = _regenerate_records()
    assert (
        len(actual) == len(expected) == 12 * (MAX_ORDER + 1) * len(_VARIANTS)
    )
    for exp, act in zip(expected, actual):
        assert act == exp, (
            f"drift at base {exp['base_cell']} order {exp['order']} "
            f"variant {exp['variant']}: {act} != {exp}"
        )


def test_golden_decimal_repr_shape():
    """Pin the structural contract of the repr across the full 0-29 range."""
    for rec in _load_fixture()["records"]:
        order = rec["order"]
        base = rec["base_cell"]
        s = rec["decimal_repr"]
        digits = s.lstrip("-")
        # leading base-cell digit + one digit per order.
        assert len(digits) == order + 1
        # sign iff southern hemisphere (bases 6-11).
        assert s.startswith("-") == (base >= 6)
        # leading digit is base+1 (north) or base-5 (south), both 1..=6.
        lead = base + 1 if base < 6 else base - 5
        assert digits[0] == str(lead)
        # per-order digits are 1..=4.
        for c in digits[1:]:
            assert c in "1234"


def test_golden_roundtrip_nested():
    """Each pinned word round-trips back to its (nested, depth)."""
    for rec in _load_fixture()["records"]:
        assert rec["roundtrip_nested"] == rec["nested"]
        assert rec["roundtrip_depth"] == rec["order"]
        assert rec["decoded_base_cell"] == rec["base_cell"]
        assert rec["decoded_order"] == rec["order"]


def test_legacy_converter_matches_repr_orders_0_to_18():
    """legacy decimal i64 -> packed -> repr recovers the legacy string (0-18).

    The legacy path (``fastNorm2Mort``) tops out at order 18; for that range the
    converter + render-only repr reproduce the exact legacy decimal value, so old
    pinned outputs can be checked against the new packed words.
    """
    for base in range(12):
        for order in range(19):
            tuples = _variant_tuples("seed", order, base * 13 + order + 2)
            nested = _nested_from_tuples(base, tuples, order)
            parent = nested >> (2 * order)
            normed = nested & ((1 << (2 * order)) - 1)
            legacy = int(_legacy_encode(order, normed, parent))
            packed = _rustie.rust_mi_from_legacy(
                np.ascontiguousarray([legacy], dtype=np.int64)
            )
            rep = _rustie.rust_mi_decimal_repr(packed)[0]
            assert rep == str(legacy), (
                f"base {base} order {order}: repr {rep} != legacy {legacy}"
            )


def test_legacy_converter_lands_on_canonical_word_0_to_18():
    """The converted word equals the canonical from_nested packing of the cell."""
    for base in (0, 5, 6, 11):
        for order in range(19):
            tuples = _variant_tuples("seed", order, base + order * 7 + 1)
            nested = _nested_from_tuples(base, tuples, order)
            parent = nested >> (2 * order)
            normed = nested & ((1 << (2 * order)) - 1)
            legacy = int(_legacy_encode(order, normed, parent))
            via_legacy = int(
                _rustie.rust_mi_from_legacy(
                    np.ascontiguousarray([legacy], dtype=np.int64)
                )[0]
            )
            via_nested = int(
                _rustie.rust_mi_from_nested(
                    np.ascontiguousarray([nested], dtype=np.uint64), order
                )[0]
            )
            assert via_legacy == via_nested


def test_morton_index_array_legacy_and_repr():
    """The MortonIndexArray skin exposes from_legacy + decimal_repr."""
    import pytest

    pytest.importorskip("pandas")
    from mortie import MortonIndexArray as MIA

    legacy = np.array(
        [int(_legacy_encode(6, 100, 2)), int(_legacy_encode(6, 200, 8))],
        dtype=np.int64,
    )
    arr = MIA.from_legacy(legacy)
    assert arr.decimal_repr() == [str(int(x)) for x in legacy]


# ---------------------------------------------------------------------------
# string-layer golden pins (issue #104): the display / emit / hive-path
# surface renders the same pinned decimal strings, so it joins the frozen 1.x
# contract through the same fixture.
# ---------------------------------------------------------------------------


def _string_layer_array():
    import pytest

    pytest.importorskip("pandas")
    from mortie import MortonIndexArray as MIA

    records = _load_fixture()["records"]
    words = np.asarray([r["word"] for r in records], dtype=np.uint64)
    return MIA(words), [r["decimal_repr"] for r in records], records


def test_golden_string_emit_layer():
    """to_decimal / _word_repr / scalar str all pin to the fixture strings."""
    arr, reprs, _ = _string_layer_array()
    out = arr.to_decimal()
    # <U32 since issue #120: the point kind suffix widens the max form by one.
    assert out.dtype == np.dtype("<U32")
    np.testing.assert_array_equal(out, np.asarray(reprs, dtype="<U31"))
    # spot-check the element formatter and the scalar wrapper on the extremes
    for i in (0, 1, len(reprs) // 2, len(reprs) - 1):
        assert arr._word_repr(int(arr._data[i])) == reprs[i]
        assert str(arr[i]) == reprs[i]


def test_golden_to_legacy_i64_orders_0_to_18():
    """to_legacy_i64 equals the pinned decimal read as an integer (0-18)."""
    import pytest

    pytest.importorskip("pandas")
    from mortie import MortonIndexArray as MIA

    _, _, records = _string_layer_array()
    legal = [r for r in records if r["order"] <= 18]
    arr = MIA(np.asarray([r["word"] for r in legal], dtype=np.uint64))
    np.testing.assert_array_equal(
        arr.to_legacy_i64(),
        np.asarray([int(r["decimal_repr"]) for r in legal], dtype=np.int64),
    )


def test_golden_hive_path_round_trip():
    """hive_path over every fixture word parses back to the same words."""
    arr, reprs, _ = _string_layer_array()  # skips first if pandas is absent
    from mortie import MortonIndexArray as MIA
    paths = arr.hive_path()
    back = MIA.from_hive_path(paths)
    np.testing.assert_array_equal(back._data, arr._data)
    # pin the literal layout on representative ids (order 0, mid, deep south)
    by_repr = dict(zip(reprs, paths))
    assert by_repr["3"] == "3/3.zarr"
    assert by_repr["-1"] == "-1/-1.zarr"
    assert by_repr["31111"] == "3/1/1/1/1/31111.zarr"
    assert by_repr["-64444"] == "-6/4/4/4/4/-64444.zarr"


def test_golden_repr_not_injective_point_vs_area():
    """Kind survives the decimal round-trip via the ``p`` suffix (issue #120).

    A point renders as the area string plus a terminal ``p``, so the repr is
    injective across kinds and both round-trip losslessly; an UNMARKED
    order-29 string still parses to the area word (the spec section 4
    tie-break -- backward compatible with every pre-suffix string).
    """
    import pytest

    pytest.importorskip("pandas")
    from mortie import MortonIndexArray as MIA
    from mortie.morton_index import _decimal_to_word

    nested = np.asarray(
        [_nested_from_tuples(4, _variant_tuples("seed", MAX_ORDER, 9),
                             MAX_ORDER)],
        dtype=np.uint64,
    )
    area = MIA(_rustie.rust_mi_from_nested(nested, MAX_ORDER))
    point = MIA(_rustie.rust_mi_from_nested_point(nested))
    assert int(area._data[0]) != int(point._data[0])
    area_s, point_s = area.decimal_repr()[0], point.decimal_repr()[0]
    assert point_s == area_s + "p"
    # Lossless round-trip for BOTH kinds (issue #120)...
    assert _decimal_to_word(area_s) == int(area._data[0])
    assert _decimal_to_word(point_s) == int(point._data[0])
    # ...and the unmarked order-29 string keeps the area tie-break.
    assert _decimal_to_word(point_s[:-1]) == int(area._data[0])

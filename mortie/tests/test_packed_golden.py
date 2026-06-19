"""
Golden fixtures for the packed-u64 ``decimal_morton`` contract (issue #48).

This is the Tier-0 "freeze the contract" deliverable: a committed snapshot of the
canonical packed word, its decode-through-kernel decimal repr, and the
``(nested, depth)`` bridge for orders 0-29 across all 12 base cells. Once shipped
it is frozen for mortie 1.x, so these values are pinned deliberately and any
change here is a deliberate (major-version) contract change.

The fixture is regenerated from the kernel here and asserted byte-for-byte
against the committed JSON, so a drift in the kernel encode / repr is caught.
Also pins the one-way ``legacy_decimal_i64 -> packed_u64`` converter against the
retired decimal path for the orders that path could express (0-18).
"""

import json
import os

import numpy as np

from mortie import _rustie
import mortie.tools as tools

MAX_ORDER = 29

_FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "packed_u64_golden.json"
)


def _sample_tuples(order, seed):
    """Deterministic stored-tuple vector (matches the generator + Rust kernel)."""
    return [((seed * 2654435761 + i) % 4) for i in range(max(order, 1))]


def _nested_from_tuples(base, tuples, order):
    within = 0
    for n in range(1, order + 1):
        within |= (int(tuples[n - 1]) & 3) << (2 * (order - n))
    return base * (1 << (2 * order)) + within


def _regenerate_records():
    """Rebuild the fixture records straight from the kernel bindings."""
    records = []
    for base in range(12):
        for order in range(MAX_ORDER + 1):
            tuples = _sample_tuples(order, base + 1)
            nested = _nested_from_tuples(base, tuples, order)
            word = int(
                _rustie.rust_mi_from_nested(
                    np.ascontiguousarray([nested], dtype=np.uint64), order
                )[0]
            )
            w = np.asarray([word], dtype=np.int64)
            rep = _rustie.rust_mi_decimal_repr(w)[0]
            n2, d2 = _rustie.rust_mi_to_nested(w)
            records.append(
                {
                    "base_cell": base,
                    "order": order,
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
    """The committed fixture must equal a fresh kernel regeneration, exactly."""
    fixture = _load_fixture()
    assert fixture["max_order"] == MAX_ORDER
    expected = fixture["records"]
    actual = _regenerate_records()
    assert len(actual) == len(expected) == 12 * (MAX_ORDER + 1)
    for exp, act in zip(expected, actual):
        assert act == exp, (
            f"drift at base {exp['base_cell']} order {exp['order']}: "
            f"{act} != {exp}"
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
            tuples = _sample_tuples(order, base * 13 + order + 2)
            nested = _nested_from_tuples(base, tuples, order)
            parent = nested >> (2 * order)
            normed = nested & ((1 << (2 * order)) - 1)
            legacy = int(tools.fastNorm2Mort(order, normed, parent))
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
            tuples = _sample_tuples(order, base + order * 7 + 1)
            nested = _nested_from_tuples(base, tuples, order)
            parent = nested >> (2 * order)
            normed = nested & ((1 << (2 * order)) - 1)
            legacy = int(tools.fastNorm2Mort(order, normed, parent))
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
        [int(tools.fastNorm2Mort(6, 100, 2)), int(tools.fastNorm2Mort(6, 200, 8))],
        dtype=np.int64,
    )
    arr = MIA.from_legacy(legacy)
    assert arr.decimal_repr() == [str(int(x)) for x in legacy]

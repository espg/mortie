"""
End-to-end conformance for the packed-u64 wire format (issue #48, phase 4).

Phase 1 froze the kernel-level contract (``test_packed_golden.py``); this module
pins the *public* flipped surface — the bare-``i64`` morton channel that
``geo2mort`` / ``norm2mort`` / ``mort2norm`` / ``mort2healpix`` and the
coverage / buffer subsystem all share — as a round-trip identity across **all 12
base cells, both hemispheres, and orders 0-29**.

These are not golden literals (the packed words are opaque sortable keys); they
are conformance invariants that hold for any correct packing, so they survive a
re-pin and document the contract directly.
"""

import numpy as np
import pytest

import mortie
from mortie import tools
from mortie import _rustie

MAX_ORDER = 29


def _nested(base, normed, order):
    """HEALPix NESTED id for (base cell, in-base z-order, order)."""
    return (int(base) << (2 * order)) + int(normed)


# A spread of in-base z-orders kept within range for every order tested.
def _normed_samples(order):
    span = 4 ** order
    if span == 1:
        return [0]
    return sorted({0, 1, span // 3, span // 2, span - 1})


class TestNorm2MortRoundTrip:
    """norm2mort <-> mort2norm is the identity for every base cell / order."""

    @pytest.mark.parametrize("order", range(MAX_ORDER + 1))
    def test_all_base_cells_both_hemispheres(self, order):
        for base in range(12):
            for normed in _normed_samples(order):
                m = tools.norm2mort(normed, base, order)
                # Southern base cells (8-11) set the i64 sign bit.
                if base >= 8:
                    assert int(m) < 0, f"base {base} should be negative"
                n2, p2, o2 = tools.mort2norm(m)
                assert (int(n2), int(p2), o2) == (normed, base, order), (
                    f"round-trip failed base {base} order {order} normed {normed}"
                )


class TestMortHealpixRoundTrip:
    """mort2healpix recovers the exact NESTED cell norm2mort packed."""

    @pytest.mark.parametrize("order", [0, 1, 6, 13, 18, 27, 28, 29])
    def test_nested_recovered(self, order):
        for base in range(12):
            for normed in _normed_samples(order):
                m = tools.norm2mort(normed, base, order)
                cell_id, o = tools.mort2healpix(m)
                assert o == order
                assert int(cell_id) == _nested(base, normed, order)


class TestGeo2MortHemispheres:
    """geo2mort decodes to the same cell the healpix crate hashes, both poles."""

    @pytest.mark.parametrize("order", [1, 6, 12, 18, 24, 29])
    def test_geo_roundtrip(self, order):
        from mortie import _healpix as hp
        lats = np.array([0.0, 45.0, -45.0, 88.0, -88.0, 12.3, -77.7, 30.0])
        lons = np.array([0.0, 90.0, -90.0, 179.0, -179.0, 56.7, 123.4, -60.0])
        words = np.ascontiguousarray(tools.geo2mort(lats, lons, order=order))
        cell_ids, o = tools.mort2healpix(words)
        assert o == order
        np.testing.assert_array_equal(cell_ids, hp.ang2pix(order, lons, lats))
        # North-pole-ish points land in base cells 0-3 (>= 0); south in 8-11 (< 0).
        north = tools.geo2mort(85.0, 0.0, order=order)[0]
        south = tools.geo2mort(-85.0, 0.0, order=order)[0]
        assert int(north) > 0
        assert int(south) < 0


class TestDecimalReprConformance:
    """The render-only repr is one base digit + one 1-4 digit per order, all cells."""

    @pytest.mark.parametrize("order", range(MAX_ORDER + 1))
    def test_repr_shape_all_base_cells(self, order):
        words = np.ascontiguousarray(
            [int(tools.norm2mort(0, base, order)) for base in range(12)],
            dtype=np.int64,
        )
        reprs = _rustie.rust_mi_decimal_repr(words)
        for base, s in enumerate(reprs):
            digits = s.lstrip("-")
            assert len(digits) == order + 1
            assert s.startswith("-") == (base >= 6)
            lead = base + 1 if base < 6 else base - 5
            assert digits[0] == str(lead)
            for c in digits[1:]:
                assert c in "1234"


class TestSubsystemConsumesPackedWords:
    """coverage / buffer / clip consume and emit packed words consistently."""

    def test_coverage_cells_decode_and_clip(self):
        lats = np.array([40.0, 40.0, 50.0, 50.0])
        lons = np.array([-125.0, -115.0, -115.0, -125.0])
        cover = np.ascontiguousarray(
            mortie.morton_coverage(lats, lons, order=6), dtype=np.int64
        )
        assert cover.size > 0
        # Every covered word decodes to order 6 and a northern base cell (0-5).
        _, depths = _rustie.rust_mort2nested(cover)
        assert np.all(depths == 6)
        bases = _rustie.rust_mi_base_cell_of(cover)
        assert np.all(bases < 6)
        # Coarsening the cover to order 4 yields valid order-4 words.
        coarse = tools.clip2order(4, cover)
        _, cdepths = _rustie.rust_mort2nested(np.ascontiguousarray(coarse))
        assert np.all(cdepths == 4)

    def test_buffer_ring_is_packed(self):
        cells = mortie.linestring_coverage([10.0, 20.0], [30.0, 40.0], order=10)
        border = mortie.morton_buffer(cells, k=1)
        assert border.size > 0
        # Border cells share the input order and are disjoint from it.
        _, depths = _rustie.rust_mort2nested(np.ascontiguousarray(border))
        assert np.all(depths == 10)
        assert not (set(int(x) for x in border) & set(int(x) for x in cells))

    def test_generate_children_round_trip(self):
        for base in (2, 8, 11):
            parent = int(tools.norm2mort(7, base, 6))
            kids = mortie.generate_morton_children(parent, target_order=9)
            assert len(kids) == 4 ** 3
            # Every child coarsens back to the parent.
            np.testing.assert_array_equal(
                tools.clip2order(6, np.ascontiguousarray(kids, dtype=np.int64)),
                np.full(len(kids), parent, dtype=np.int64),
            )

"""
Tests for the ``morton_index`` pandas ExtensionArray (issue #35, phase 5).

The packed-word kernel is exercised in Rust (``decimal_morton.rs`` unit tests);
here we pin the Python skin: the vectorized Rust bindings, the ExtensionArray
surface (construction, repr, comparisons/sort, domain ops), round-trips against
the ``healpix`` crate (via the bindings) and an independent ``cdshealpix``
oracle, and usability as a ``pd.Series(dtype="morton_index")``.
"""

import numpy as np
import pytest

import mortie  # noqa: F401  (registers the "morton_index" pandas dtype on import)
from mortie import _rustie

pd = pytest.importorskip("pandas")
from mortie import MortonIndexArray as MIA  # noqa: E402

MAX_ORDER = 29


def _nested_from_tuples(base, tuples, order):
    """Reference nested index from base cell + stored 0..=3 tuples."""
    within = 0
    for n in range(1, order + 1):
        within |= (int(tuples[n - 1]) & 3) << (2 * (order - n))
    return base * (1 << (2 * order)) + within


def _sample_tuples(order, seed):
    """Deterministic stored 0..=3 tuples (mirrors the Rust test helper)."""
    return np.array(
        [((seed * 2654435761 + i) % 4) for i in range(max(order, 1))],
        dtype=np.uint8,
    )


# ---------------------------------------------------------------------------
# Rust binding round-trips
# ---------------------------------------------------------------------------


class TestNestedRoundTrip:
    """from_nested / to_nested round-trips for all base cells / orders."""

    def test_all_base_cells_all_orders(self):
        for base in range(12):
            for order in range(MAX_ORDER + 1):
                tuples = _sample_tuples(order, base + 1)
                nested = _nested_from_tuples(base, tuples, order)
                arr = np.array([nested], dtype=np.uint64)
                words = _rustie.rust_mi_from_nested(arr, order)
                back, depth = _rustie.rust_mi_to_nested(words.astype(np.uint64))
                assert int(depth[0]) == order, (base, order)
                assert int(back[0]) == nested, (base, order)

    def test_both_hemispheres(self):
        # base 0-3 northern, 4-7 equatorial, 8-11 southern; cover all bands.
        bases = np.array([0, 3, 4, 7, 8, 11], dtype=np.uint64)
        order = 12
        nested = (bases << np.uint64(2 * order)) + np.uint64(123)
        words = _rustie.rust_mi_from_nested(nested, order)
        back, depth = _rustie.rust_mi_to_nested(words.astype(np.uint64))
        np.testing.assert_array_equal(back, nested)
        np.testing.assert_array_equal(depth, np.full(len(bases), order))


class TestVsCdshealpix:
    """End-to-end against an independent HEALPix oracle (cdshealpix)."""

    def test_latlon_matches_cdshealpix_nested(self):
        pytest.importorskip("cdshealpix")
        import astropy.units as u
        from cdshealpix.nested import lonlat_to_healpix

        order = 14
        lats = np.array([0.0, 41.8, -41.8, 80.0, -80.0, 12.3, -67.9])
        lons = np.array([0.0, 45.0, 135.0, 200.0, 305.0, 91.5, 270.2])
        a = MIA.from_latlon(lats, lons, order=order)
        ours, depth = a.to_nested()
        oracle = np.asarray(
            lonlat_to_healpix(lons * u.deg, lats * u.deg, depth=order),
            dtype=np.uint64,
        )
        np.testing.assert_array_equal(ours, oracle)
        np.testing.assert_array_equal(depth, np.full(len(lats), order))


# ---------------------------------------------------------------------------
# coarsen / order / base_cell domain ops
# ---------------------------------------------------------------------------


class TestDomainOps:

    def test_coarsen_vectorized_matches_kernel(self):
        # Build an order-29 array, coarsen to each k, and compare to a
        # re-encode at k built straight from the kernel binding.
        base = 7
        tuples = _sample_tuples(29, 99)
        nested29 = _nested_from_tuples(base, tuples, 29)
        a = MIA.from_nested(np.array([nested29], dtype=np.uint64), 29)
        for k in range(MAX_ORDER + 1):
            coarsened = a.coarsen(k)
            expected = MIA.from_nested(
                np.array([_nested_from_tuples(base, tuples, k)], dtype=np.uint64), k
            )
            np.testing.assert_array_equal(
                coarsened._data, expected._data, err_msg=f"coarsen k={k}"
            )

    def test_coarsen_noop_when_k_ge_order(self):
        a = MIA.from_nested(np.array([5, 6, 7], dtype=np.uint64), 3)
        for k in (3, 5, 29):
            np.testing.assert_array_equal(a.coarsen(k)._data, a._data)

    def test_orders_and_base_cells(self):
        nested = np.array(
            [3 * (1 << (2 * 10)) + 5, 8 * (1 << (2 * 10)) + 9], dtype=np.uint64
        )
        a = MIA.from_nested(nested, 10)
        np.testing.assert_array_equal(a.orders(), np.array([10, 10], dtype=np.uint8))
        np.testing.assert_array_equal(
            a.base_cells(), np.array([3, 8], dtype=np.uint8)
        )

    def test_is_fixed_order(self):
        fixed = MIA.from_nested(np.array([5, 6, 7], dtype=np.uint64), 5)
        assert fixed.is_fixed_order()
        assert fixed.order() == 5
        mixed = MIA.from_words(
            np.concatenate(
                [
                    fixed._data,
                    MIA.from_nested(np.array([3], dtype=np.uint64), 2)._data,
                ]
            )
        )
        assert not mixed.is_fixed_order()
        with pytest.raises(ValueError):
            mixed.order()

    def test_base_cell_scalar_and_mixed(self):
        same = MIA.from_nested(
            np.array([5 + (4 << 20), 6 + (4 << 20)], dtype=np.uint64), 10
        )
        assert same.base_cell() == 4
        multi = MIA.from_nested(
            np.array([0 + (0 << 20), 0 + (5 << 20)], dtype=np.uint64), 10
        )
        with pytest.raises(ValueError):
            multi.base_cell()

    def test_empty_array_fixed_order(self):
        empty = MIA(np.array([], dtype=np.uint64))
        assert empty.is_fixed_order()
        assert empty.order() is None
        assert empty.base_cell() is None


# ---------------------------------------------------------------------------
# raw int64 sort == Z-order, canonicalization
# ---------------------------------------------------------------------------


class TestSortAndCanonical:

    def test_raw_sort_is_zorder(self):
        # Parent sorts before its children, children already ascending.
        base = 4
        parent = MIA.from_nested(
            np.array([_nested_from_tuples(base, [0, 0], 2)], dtype=np.uint64), 2
        )._data[0]
        children = MIA.from_nested(
            np.array(
                [_nested_from_tuples(base, [0, 0, c], 3) for c in range(4)],
                dtype=np.uint64,
            ),
            3,
        )._data
        assert np.all(parent < children)
        np.testing.assert_array_equal(children, np.sort(children))

    def test_series_sort_matches_unsigned_zorder(self):
        rng = np.random.default_rng(42)
        nested = rng.integers(0, 12 * (1 << (2 * 8)), size=200, dtype=np.uint64)
        a = MIA.from_nested(nested, 8)
        s = pd.Series(a)
        sorted_series = s.sort_values()
        # Z-order is the *unsigned* word order (the kernel's u64 sort).
        expected = np.sort(a._data)
        np.testing.assert_array_equal(sorted_series.values._data, expected)

    def test_sort_orders_southern_base_cells(self):
        # Base cells 7-11 (prefix 8-12) set bit 63, so a *signed* sort would
        # invert them. The array must sort by ascending base cell because the
        # word is unsigned u64 and its raw order is the Z-order. (regression for
        # the would-be i64 sign-bit inversion)
        order = 5
        nested = np.array(
            [b * (1 << (2 * order)) + 1 for b in range(12)], dtype=np.uint64
        )
        a = MIA.from_nested(nested, order)
        s = pd.Series(a).sort_values()
        np.testing.assert_array_equal(
            s.values.base_cells(), np.arange(12, dtype=np.uint8)
        )
        # base 6 sorts before base 7 (both straddle the sign boundary region).
        b6 = MIA.from_nested(np.array([6 << (2 * order)], dtype=np.uint64), order)
        b7 = MIA.from_nested(np.array([7 << (2 * order)], dtype=np.uint64), order)
        assert bool(b6 < b7._data[0])
        # and a southern cell (base 11) sorts after a northern one (base 0).
        b0 = MIA.from_nested(np.array([0 << (2 * order)], dtype=np.uint64), order)
        b11 = MIA.from_nested(np.array([11 << (2 * order)], dtype=np.uint64), order)
        assert bool(b0 < b11._data[0])

    def test_canonical_zero_fill_dedups(self):
        # Two nested indices that agree on the first 3 orders but differ in the
        # discarded tail must coarsen to the same canonical word.
        base = 2
        t = [1, 2, 3]
        n1 = _nested_from_tuples(base, t + [0, 0], 5)
        n2 = _nested_from_tuples(base, t + [3, 1], 5)
        a = MIA.from_nested(np.array([n1, n2], dtype=np.uint64), 5).coarsen(3)
        assert a._data[0] == a._data[1]


# ---------------------------------------------------------------------------
# repr, comparisons, pandas integration
# ---------------------------------------------------------------------------


class TestReprAndPandas:

    def test_repr_truncates(self):
        a = MIA.from_nested(np.arange(20, dtype=np.uint64), 5)
        r = repr(a)
        assert "..." in r
        assert "order=5" in r
        assert "len=20" in r

    def test_repr_mixed_order_label(self):
        a = MIA.from_words(
            np.concatenate(
                [
                    MIA.from_nested(np.array([5], dtype=np.uint64), 5)._data,
                    MIA.from_nested(np.array([3], dtype=np.uint64), 2)._data,
                ]
            )
        )
        assert "order=mixed" in repr(a)

    def test_comparisons(self):
        a = MIA.from_nested(np.array([5, 6, 7], dtype=np.uint64), 5)
        np.testing.assert_array_equal(a == a, np.array([True, True, True]))
        np.testing.assert_array_equal(a < a._data + 1, np.array([True] * 3))
        b = MIA.from_nested(np.array([5, 99, 7], dtype=np.uint64), 5)
        assert list(a == b) == [True, False, True]

    def test_no_arithmetic_operators(self):
        a = MIA.from_nested(np.array([5, 6], dtype=np.uint64), 5)
        # Arithmetic on packed words is meaningless and is not implemented.
        with pytest.raises(TypeError):
            a + a

    def test_series_construction_by_name(self):
        words = MIA.from_nested(np.array([5, 6, 7, 8], dtype=np.uint64), 3)._data
        s = pd.Series(words.tolist(), dtype="morton_index")
        assert str(s.dtype) == "morton_index"
        assert len(s) == 4
        # round-trips through the ExtensionArray
        assert isinstance(s.values, MIA)
        np.testing.assert_array_equal(s.values._data, words)

    def test_na_sentinel(self):
        a = MIA(np.array([0, 5], dtype=np.uint64))
        np.testing.assert_array_equal(a.isna(), np.array([True, False]))
        assert a._word_repr(0) == "<NA>"

    def test_setitem_accepts_na(self):
        a = MIA.from_nested(np.array([5, 6, 7], dtype=np.uint64), 5)
        a[1] = pd.NA
        assert a.isna()[1]
        a[2] = None
        assert a.isna()[2]
        # a real word still assigns
        a[0] = int(a._data[0])
        assert not a.isna()[0]

    def test_from_sequence_accepts_na(self):
        w = int(MIA.from_nested(np.array([5], dtype=np.uint64), 3)._data[0])
        a = MIA._from_sequence([w, pd.NA, None])
        np.testing.assert_array_equal(a.isna(), np.array([False, True, True]))
        # Series with missing entries routes through _from_sequence
        s = pd.Series([w, None], dtype="morton_index")
        assert bool(s.isna().iloc[1])

    def test_take_with_fill(self):
        a = MIA.from_nested(np.array([5, 6, 7], dtype=np.uint64), 5)
        taken = a.take([0, -1], allow_fill=True)
        assert taken.isna()[1]
        assert taken._data[0] == a._data[0]

    def test_concat_and_getitem(self):
        a = MIA.from_nested(np.array([5, 6], dtype=np.uint64), 5)
        b = MIA.from_nested(np.array([7], dtype=np.uint64), 5)
        c = MIA._concat_same_type([a, b])
        assert len(c) == 3
        assert isinstance(c[0], np.uint64)
        assert isinstance(c[:2], MIA)


# ---------------------------------------------------------------------------
# encode / decode bindings
# ---------------------------------------------------------------------------


class TestEncodeDecode:

    def test_encode_decode_round_trip(self):
        base_cells = np.array([0, 5, 11], dtype=np.uint8)
        orders = np.array([3, 10, 29], dtype=np.uint8)
        tuples = np.zeros((3, 29), dtype=np.uint8)
        for i in range(3):
            tuples[i, : orders[i]] = _sample_tuples(int(orders[i]), i + 1)[
                : orders[i]
            ]
        words = _rustie.rust_mi_encode(base_cells, tuples, orders)
        b2, o2, kinds, t2 = _rustie.rust_mi_decode(words.astype(np.uint64))
        np.testing.assert_array_equal(b2, base_cells)
        np.testing.assert_array_equal(o2, orders)
        np.testing.assert_array_equal(kinds, np.zeros(3, dtype=np.uint8))
        for i in range(3):
            np.testing.assert_array_equal(
                t2[i, : orders[i]], tuples[i, : orders[i]]
            )

    def test_to_nested_rejects_empty(self):
        with pytest.raises(ValueError):
            _rustie.rust_mi_to_nested(np.array([0], dtype=np.uint64))

    def test_coarsen_rejects_empty(self):
        with pytest.raises(ValueError):
            _rustie.rust_mi_coarsen(np.array([0], dtype=np.uint64), 3)

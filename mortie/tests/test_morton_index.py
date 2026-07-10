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


class TestPointEncode:
    """from_latlon(points=True) surfaces the Kind::Point encode path (issue #79)."""

    LATS = np.array([0.0, 41.8, -41.8, 80.0, -80.0, 12.3, -67.9])
    LONS = np.array([0.0, 45.0, 135.0, 200.0, 305.0, 91.5, 270.2])

    def test_points_true_yields_order29_points(self):
        a = MIA.from_latlon(self.LATS, self.LONS, points=True)
        # rust_mi_decode returns (base_cells, orders, kinds, tuples); kind 1=point.
        _, orders, kinds, _ = _rustie.rust_mi_decode(a._data)
        np.testing.assert_array_equal(orders, np.full(len(self.LATS), 29))
        np.testing.assert_array_equal(kinds, np.ones(len(self.LATS), dtype=np.uint8))

    def test_points_share_nested_cell_with_area(self):
        # A point and the order-29 area cell for the same location carry the same
        # nested cell (the point/area flag is the decimal_morton kind, not nested).
        pts = MIA.from_latlon(self.LATS, self.LONS, points=True)
        area = MIA.from_latlon(self.LATS, self.LONS, order=29)
        pn, pd = pts.to_nested()
        an, ad = area.to_nested()
        np.testing.assert_array_equal(pn, an)
        np.testing.assert_array_equal(pd, ad)

    def test_points_coarsen_and_to_nested(self):
        pts = MIA.from_latlon(self.LATS, self.LONS, points=True)
        # to_nested recovers an order-29 cell for every point.
        _, depth = pts.to_nested()
        np.testing.assert_array_equal(depth, np.full(len(self.LATS), 29))
        # Coarsening a point to order 5 yields an order-5 area cell.
        coarse = pts.coarsen(5)
        _, orders, kinds, _ = _rustie.rust_mi_decode(coarse._data)
        np.testing.assert_array_equal(orders, np.full(len(self.LATS), 5))
        np.testing.assert_array_equal(kinds, np.zeros(len(self.LATS), dtype=np.uint8))

    def test_points_false_default_is_area(self):
        area = MIA.from_latlon(self.LATS, self.LONS, order=12)
        default = MIA.from_latlon(self.LATS, self.LONS)
        _, _, kinds_area, _ = _rustie.rust_mi_decode(area._data)
        _, _, kinds_def, _ = _rustie.rust_mi_decode(default._data)
        np.testing.assert_array_equal(kinds_area, np.zeros(len(self.LATS), np.uint8))
        np.testing.assert_array_equal(kinds_def, np.zeros(len(self.LATS), np.uint8))

    def test_point_words_differ_from_area_words(self):
        # Same location, but the packed point word is a distinct value from the
        # order-29 area word (the suffix carries the point flag).
        pts = MIA.from_latlon(self.LATS, self.LONS, points=True)
        area = MIA.from_latlon(self.LATS, self.LONS, order=29)
        assert np.all(pts._data != area._data)

    def test_points_true_explicit_order29_is_accepted(self):
        # Passing the (redundant) explicit order=29 with points=True is allowed.
        explicit = MIA.from_latlon(self.LATS, self.LONS, order=29, points=True)
        default = MIA.from_latlon(self.LATS, self.LONS, points=True)
        np.testing.assert_array_equal(explicit._data, default._data)

    def test_points_true_with_non29_order_raises(self):
        with pytest.raises(ValueError, match="order-29 point"):
            MIA.from_latlon(self.LATS, self.LONS, order=12, points=True)


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
# decimal-string display layer (issue #104)
# ---------------------------------------------------------------------------


class TestDecimalDisplay:
    """Element display is the decode-through-kernel decimal string."""

    def test_element_repr_is_decimal_string(self):
        a = MIA.from_legacy(np.array([-31123, 41123], dtype=np.int64))
        assert a._word_repr(int(a._data[0])) == "-31123"
        assert a._word_repr(int(a._data[1])) == "41123"

    def test_array_repr_decimal_elements_with_summary_header(self):
        a = MIA.from_legacy(np.array([-31123, 41123], dtype=np.int64))
        r = repr(a)
        assert "-31123" in r and "41123" in r
        assert "len=2" in r and "order=4" in r
        assert "base=" not in r  # the old per-element label is gone

    def test_scalar_wrapper_str_repr_int(self):
        from mortie.morton_index import MortonIndexScalar

        a = MIA.from_legacy(np.array([-31123], dtype=np.int64))
        s = a[0]
        assert isinstance(s, MortonIndexScalar)
        assert isinstance(s, np.uint64)  # still a word for compute paths
        assert str(s) == "-31123"
        assert repr(s) == "-31123"
        assert f"{s}" == "-31123"
        assert int(s) == int(a._data[0])
        assert s == a._data[0]  # comparisons stay word-valued

    def test_scalar_wrapper_format_specs(self):
        a = MIA.from_legacy(np.array([-31123], dtype=np.int64))
        s = a[0]
        # string specs format the decimal string
        assert f"{s:>10}" == "    -31123"
        # numeric specs raise: the display form is a string; int(s) is the
        # escape hatch for formatting the raw word numerically
        with pytest.raises(ValueError):
            f"{s:d}"
        # old-style %-formatting bypasses __format__ and emits the raw word
        assert ("%d" % s) == str(int(s))

    def test_scalar_wrapper_pickles_as_itself(self):
        import pickle

        from mortie.morton_index import MortonIndexScalar

        a = MIA.from_legacy(np.array([-31123], dtype=np.int64))
        s = pickle.loads(pickle.dumps(a[0]))
        assert isinstance(s, MortonIndexScalar)
        assert str(s) == "-31123"
        assert int(s) == int(a._data[0])

    def test_scalar_wrapper_na_and_invalid_never_raise(self):
        from mortie.morton_index import MortonIndexScalar

        assert str(MortonIndexScalar(0)) == "<NA>"
        # prefix 15 is outside the valid 1..=12 range; repr must not raise
        assert str(MortonIndexScalar(0xF000000000000000)).startswith("<invalid")

    def test_series_repr_prints_decimal(self):
        a = MIA.from_legacy(np.array([-31123, 41123], dtype=np.int64))
        text = repr(pd.Series(a))
        assert "-31123" in text and "41123" in text

    def test_to_decimal_fixed_width(self):
        a = MIA.from_legacy(np.array([-31123, 41123], dtype=np.int64))
        out = a.to_decimal()
        assert out.dtype == np.dtype("<U31")
        np.testing.assert_array_equal(
            out, np.array(["-31123", "41123"], dtype="<U31")
        )
        # width holds the widest form: southern order-29 is 31 chars
        deep = MIA.from_nested(
            np.array([11 << (2 * MAX_ORDER)], dtype=np.uint64), MAX_ORDER
        )
        s = deep.to_decimal()[0]
        assert len(s) == 31 and s.startswith("-6")

    def test_to_decimal_empty_and_invalid(self):
        assert MIA(np.empty(0, dtype=np.uint64)).to_decimal().shape == (0,)
        with pytest.raises(ValueError):
            MIA(np.array([0], dtype=np.uint64)).to_decimal()

    def test_to_legacy_i64_round_trips(self):
        legacy = np.array([-31123, 41123, 1, -6], dtype=np.int64)
        a = MIA.from_legacy(legacy)
        out = a.to_legacy_i64()
        assert out.dtype == np.int64
        np.testing.assert_array_equal(out, legacy)
        # and the pair is a true inverse
        np.testing.assert_array_equal(
            MIA.from_legacy(out)._data, a._data
        )

    def test_to_legacy_i64_hard_error_above_order_18(self):
        a = MIA.from_nested(np.array([5], dtype=np.uint64), 19)
        with pytest.raises(ValueError, match="capped at order 18"):
            a.to_legacy_i64()
        # mixed: one legal element does not soften the error
        mixed = MIA(
            np.concatenate(
                [
                    MIA.from_nested(np.array([5], dtype=np.uint64), 3)._data,
                    a._data,
                ]
            )
        )
        with pytest.raises(ValueError, match="capped at order 18"):
            mixed.to_legacy_i64()

    def test_to_legacy_i64_empty_and_sentinel(self):
        out = MIA(np.empty(0, dtype=np.uint64)).to_legacy_i64()
        assert out.dtype == np.int64 and out.shape == (0,)
        with pytest.raises(ValueError):
            MIA(np.array([0], dtype=np.uint64)).to_legacy_i64()

    def test_display_matches_kernel_across_orders(self):
        for base in (0, 5, 11):
            for order in (0, 1, 18, 19, MAX_ORDER):
                tuples = _sample_tuples(order, base + 1)
                nested = _nested_from_tuples(base, tuples, order)
                a = MIA.from_nested(np.array([nested], dtype=np.uint64), order)
                expect = _rustie.rust_mi_decimal_repr(a._data)[0]
                assert a._word_repr(int(a._data[0])) == expect
                assert str(a[0]) == expect


class TestHivePath:
    """hive_path / from_hive_path round-trips (issue #104; spec on #62)."""

    def test_layout_one_digit_per_level_full_id_leaf(self):
        a = MIA.from_legacy(np.array([-31123, 41123], dtype=np.int64))
        assert a.hive_path() == [
            "-3/1/1/2/3/-31123.zarr",
            "4/1/1/2/3/41123.zarr",
        ]

    def test_order_zero_leaf_sits_in_base_node(self):
        a = MIA.from_legacy(np.array([-3, 4], dtype=np.int64))
        assert a.hive_path() == ["-3/-3.zarr", "4/4.zarr"]

    def test_root_and_suffix(self):
        a = MIA.from_legacy(np.array([-31123], dtype=np.int64))
        assert a.hive_path(root="s3://bucket/store/") == [
            "s3://bucket/store/-3/1/1/2/3/-31123.zarr"
        ]
        assert a.hive_path(suffix="") == ["-3/1/1/2/3/-31123"]

    def test_mixed_orders_nest(self):
        # the order-3 shard's leaf sits in the directory its order-4
        # sibling descends through
        a = MIA.from_legacy(np.array([-3112, -31123], dtype=np.int64))
        coarse, fine = a.hive_path()
        assert coarse == "-3/1/1/2/-3112.zarr"
        assert fine.startswith("-3/1/1/2/3/")

    def test_round_trip(self):
        a = MIA.from_legacy(np.array([-31123, 41123, -3], dtype=np.int64))
        back = MIA.from_hive_path(a.hive_path(root="s3://bucket/x"))
        np.testing.assert_array_equal(back._data, a._data)

    def test_round_trip_high_order(self):
        deep = MIA.from_nested(
            np.array([11 << (2 * MAX_ORDER), 5], dtype=np.uint64), MAX_ORDER
        )
        back = MIA.from_hive_path(deep.hive_path())
        np.testing.assert_array_equal(back._data, deep._data)

    def test_single_path_and_bare_leaf(self):
        a = MIA.from_legacy(np.array([-31123], dtype=np.int64))
        one = MIA.from_hive_path("-3/1/1/2/3/-31123.zarr")
        assert len(one) == 1 and int(one._data[0]) == int(a._data[0])
        # a bare leaf (no digit directories) parses too
        bare = MIA.from_hive_path("-31123.zarr")
        assert int(bare._data[0]) == int(a._data[0])

    def test_pathlike_input(self):
        from pathlib import Path

        a = MIA.from_legacy(np.array([-31123], dtype=np.int64))
        one = MIA.from_hive_path(Path("-3/1/1/2/3/-31123.zarr"))
        assert int(one._data[0]) == int(a._data[0])

    def test_bare_leaf_under_root_skips_dir_check(self):
        # root components are not digit directories: the check only engages
        # when the {sign+base} anchor sits at its slot above the leaf
        a = MIA.from_legacy(np.array([-3], dtype=np.int64))
        for p in ("s3://bucket/-3.zarr", "x/-3.zarr", "a/b/c/-3.zarr"):
            assert int(MIA.from_hive_path(p)._data[0]) == int(a._data[0])

    def test_misfiled_leaf_raises(self):
        with pytest.raises(ValueError, match="do not match leaf"):
            MIA.from_hive_path("-3/1/1/2/4/-31123.zarr")
        # anchored but wrong descent raises too
        with pytest.raises(ValueError, match="do not match leaf"):
            MIA.from_hive_path("-3/x/1/2/3/-31123.zarr")
        # a wrong *base* directory is still base-shaped, so it anchors the
        # check and raises rather than being mistaken for a root component
        with pytest.raises(ValueError, match="do not match leaf"):
            MIA.from_hive_path("-4/1/1/2/3/-31123.zarr")
        with pytest.raises(ValueError, match="do not match leaf"):
            MIA.from_hive_path("2/1/1/2/3/41123.zarr")

    def test_root_ending_in_digit_chain_still_parses(self):
        # the anchor slot holding the id's own sign+base under a real root
        a = MIA.from_legacy(np.array([-31123], dtype=np.int64))
        p = "data/2023/-3/1/1/2/3/-31123.zarr"
        assert int(MIA.from_hive_path(p)._data[0]) == int(a._data[0])

    def test_malformed_ids_raise(self):
        from mortie.morton_index import _decimal_to_word

        for bad in ("", "-", "0123", "7123", "31023", "3125", "x123",
                    "3" + "1" * 30):
            with pytest.raises(ValueError):
                _decimal_to_word(bad)
        with pytest.raises(ValueError, match="does not end with"):
            MIA.from_hive_path("-3/1/-31.parquet")

    def test_sentinel_raises(self):
        with pytest.raises(ValueError):
            MIA(np.array([0], dtype=np.uint64)).hive_path()


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

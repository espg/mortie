"""Public decimal->word parse surface (issue #114).

The emit direction (``decimal_repr`` / ``to_decimal`` / ``hive_path``) has
been public since issue #104; these are the pins for its public inverse:
``mortie.decimal_to_word`` (scalar, with the ``dtype`` return-shape flag),
``mortie.decimals_to_words`` (vectorized, Rust-backed), and the retained
private ``_decimal_to_word`` alias.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import mortie
from mortie.morton_index import (
    MAX_ORDER,
    MortonIndexScalar,
    _decimal_to_word,
    decimal_to_word,
    decimals_to_words,
)


class TestPublicSurface:
    def test_exported_from_package_root(self):
        assert mortie.decimal_to_word is decimal_to_word
        assert mortie.decimals_to_words is decimals_to_words
        assert "decimal_to_word" in mortie.__all__
        assert "decimals_to_words" in mortie.__all__

    def test_parses_with_pandas_unavailable(self):
        # The point of the numpy-only path (issue #114): zagg's per-shard key
        # parse must not need pandas. `import mortie` *does* touch pandas when
        # it is installed -- deliberately, to register the dtype eagerly
        # (morton_index.py, the `_build_classes()` probe) -- so proving the
        # claim means running with pandas made unimportable, not just checking
        # sys.modules. A fresh interpreter with a meta_path blocker does that.
        code = (
            "import sys\n"
            "class Block:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'pandas' or name.startswith('pandas.'):\n"
            "            raise ImportError('pandas blocked for this test')\n"
            "        return None\n"
            "sys.meta_path.insert(0, Block())\n"
            "import mortie\n"
            "assert 'pandas' not in sys.modules\n"
            "w = mortie.decimal_to_word('-31123')\n"
            "a = mortie.decimals_to_words(['3', '-6', '31123'])\n"
            "assert 'pandas' not in sys.modules\n"
            "print(int(w), a.dtype)\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.split()[-1] == "uint64"
        assert int(out.stdout.split()[0]) == decimal_to_word("-31123", dtype=int)

    def test_private_alias_still_works_and_returns_int(self):
        # zagg imports the private name; espg asked to keep both through a
        # deprecation cycle, so the old return type must not drift.
        word = _decimal_to_word("-31123")
        assert type(word) is int
        assert word == int(decimal_to_word("-31123"))

    @pytest.mark.parametrize("bad", [None, 3123, 1.5, ["3123"], np.uint64(3)])
    def test_private_alias_rejects_non_str_with_type_error(self, bad):
        # Deliberate difference from the old pure-Python body, which raised
        # AttributeError from `s.endswith`. Pinned so the change is a decision
        # rather than a surprise for the downstream that imports this name.
        with pytest.raises(TypeError):
            _decimal_to_word(bad)


class TestScalarDtypeFlag:
    def test_default_is_numpy_uint64(self):
        word = decimal_to_word("-31123")
        assert isinstance(word, np.uint64)
        assert not isinstance(word, MortonIndexScalar)

    def test_python_int(self):
        word = decimal_to_word("-31123", dtype=int)
        assert type(word) is int

    def test_morton_index_scalar_displays_as_the_decimal_string(self):
        word = decimal_to_word("-31123", dtype=MortonIndexScalar)
        assert isinstance(word, MortonIndexScalar)
        assert str(word) == "-31123"

    @pytest.mark.parametrize("spelling", ["uint64", np.dtype("uint64")])
    def test_uint64_spellings_accepted(self, spelling):
        got = decimal_to_word("3123", dtype=spelling)
        # Assert the *type*, not just the value: a lossy float64 return would
        # compare equal here while silently truncating large words.
        assert isinstance(got, np.uint64)
        assert int(got) == decimal_to_word("3123", dtype=int)

    @pytest.mark.parametrize(
        "bad", [float, "float64", np.int64, object(), None, bool, np.uint32,
                np.uint64(0)]
    )
    def test_unsupported_dtype_raises(self, bad):
        # np.uint64(0) is an *instance*: np.dtype() accepts it, so without an
        # explicit spelling check a stray value would be read as the default.
        with pytest.raises(TypeError, match="dtype must be"):
            decimal_to_word("3123", dtype=bad)

    def test_morton_index_scalar_subclass_is_preserved(self):
        class MyScalar(MortonIndexScalar):
            pass

        got = decimal_to_word("3123", dtype=MyScalar)
        assert type(got) is MyScalar


class TestVectorized:
    def test_matches_the_scalar_elementwise(self):
        ids = ["1", "-6", "3123", "-31123", "2" + "4" * MAX_ORDER]
        got = decimals_to_words(ids)
        assert got.dtype == np.uint64
        assert list(got) == [decimal_to_word(s) for s in ids]

    def test_shape_is_preserved(self):
        arr = np.array([["1", "-6"], ["3123", "-31123"]])
        assert decimals_to_words(arr).shape == (2, 2)

    def test_empty_input(self):
        for empty in ([], (), np.array([], dtype="<U32")):
            out = decimals_to_words(empty)
            assert out.shape == (0,)
            assert out.dtype == np.uint64

    @pytest.mark.parametrize(
        "empty",
        [np.array([], dtype=np.int64), np.zeros((0, 3), dtype=complex)],
    )
    def test_empty_but_wrong_dtype_still_rejected(self, empty):
        # The type guard must be dtype-driven, not size-driven: an array that
        # merely happens to be empty is still the wrong type, and accepting it
        # would mean the check only bites once there is data.
        with pytest.raises(TypeError, match="expects decimal Morton strings"):
            decimals_to_words(empty)

    def test_bare_string_is_rejected_not_treated_as_one_element(self):
        # np.asarray("3123") is a 0-d array, so this would otherwise return a
        # 0-d word -- a silent trap on the singular/plural name pair.
        with pytest.raises(TypeError, match="use decimal_to_word"):
            decimals_to_words("3123")

    def test_object_array_non_string_names_the_offender(self):
        with pytest.raises(TypeError, match="got int"):
            decimals_to_words(np.array(["3123", 7], dtype=object))

    def test_object_array_of_strings(self):
        arr = np.array(["3123", "-6"], dtype=object)
        assert list(decimals_to_words(arr)) == [
            decimal_to_word("3123"),
            decimal_to_word("-6"),
        ]

    def test_non_string_input_rejected_not_coerced(self):
        # np.asarray([1, 2], dtype=str) would silently become the order-0 ids
        # "1"/"2"; a parse surface must not invent input.
        with pytest.raises(TypeError, match="expects decimal Morton strings"):
            decimals_to_words([1, 2])

    def test_error_names_the_first_bad_id_in_input_order(self):
        with pytest.raises(ValueError, match="'0123'"):
            decimals_to_words(["3123", "0123", "7123"])


class TestRoundTrip:
    @pytest.mark.parametrize("order", [0, 1, 13, 27, 28, MAX_ORDER])
    def test_word_to_decimal_to_word_identity(self, order):
        pytest.importorskip("pandas")
        from mortie import MortonIndexArray

        lats = np.array([-70.0, -20.0, 0.0, 20.0, 70.0])
        lons = np.array([-170.0, -45.0, 0.0, 45.0, 170.0])
        arr = MortonIndexArray.from_latlon(lats, lons, order=order)
        words = np.asarray(arr._data, dtype=np.uint64)
        assert np.array_equal(decimals_to_words(arr.to_decimal()), words)

    def test_malformed_ids_raise(self):
        for bad in ("", "-", "0123", "7123", "31023", "3125", "x123",
                    "3" + "1" * 30, "p", "-p", "31111pp"):
            with pytest.raises(ValueError):
                decimal_to_word(bad)


class TestExtensionArrayClassmethod:
    """``MortonIndexArray.from_decimal`` -- the pandas-side sugar."""

    def test_round_trips_to_decimal(self):
        pytest.importorskip("pandas")
        from mortie import MortonIndexArray

        arr = MortonIndexArray.from_latlon(
            np.array([-45.0, 12.5, 78.0]), np.array([170.0, -3.0, 45.0]),
            order=11,
        )
        back = MortonIndexArray.from_decimal(arr.to_decimal())
        assert isinstance(back, MortonIndexArray)
        assert np.array_equal(
            np.asarray(back._data, dtype=np.uint64),
            np.asarray(arr._data, dtype=np.uint64),
        )

    def test_accepts_a_plain_list_and_matches_the_module_function(self):
        pytest.importorskip("pandas")
        from mortie import MortonIndexArray

        ids = ["3123", "-31123", "6"]
        arr = MortonIndexArray.from_decimal(ids)
        assert np.array_equal(
            np.asarray(arr._data, dtype=np.uint64), decimals_to_words(ids)
        )

    def test_malformed_id_raises(self):
        pytest.importorskip("pandas")
        from mortie import MortonIndexArray

        with pytest.raises(ValueError, match="'0123'"):
            MortonIndexArray.from_decimal(["3123", "0123"])

    def test_hive_path_leaves_round_trip(self):
        # from_hive_path already parses leaves; from_decimal is the direct
        # form of the same parse, so the two must agree on the leaf ids.
        pytest.importorskip("pandas")
        from mortie import MortonIndexArray

        arr = MortonIndexArray.from_latlon(
            np.array([10.0, -60.0]), np.array([20.0, -140.0]), order=6
        )
        paths = arr.hive_path(root="data")
        leaves = [p.rsplit("/", 1)[-1][: -len(".zarr")] for p in paths]
        assert np.array_equal(
            np.asarray(MortonIndexArray.from_decimal(leaves)._data, dtype=np.uint64),
            np.asarray(MortonIndexArray.from_hive_path(paths)._data, dtype=np.uint64),
        )


class TestPointAreaNonInjectivity:
    """Order-29 ids do not distinguish point from area unless marked.

    The parse-side statement of the spec section 4 tie-break: an *unmarked*
    order-29 string always yields the AREA word, so a point word does not
    round-trip through the unmarked decimal form. Only the ``p`` marker
    recovers it.
    """

    def test_marked_and_unmarked_order29_parse_to_different_words(self):
        pytest.importorskip("pandas")
        from mortie import MortonIndexArray

        point = MortonIndexArray.from_latlon(
            np.array([45.0]), np.array([45.0]), points=True
        )
        point_word = int(np.asarray(point._data, dtype=np.uint64)[0])
        marked = point.decimal_repr()[0]
        assert marked.endswith("p")
        unmarked = marked[:-1]

        assert decimal_to_word(marked, dtype=int) == point_word
        area_word = decimal_to_word(unmarked, dtype=int)
        assert area_word != point_word
        # Same path (prefix + body), different kind (suffix) -- section 1.
        assert area_word >> 6 == point_word >> 6
        assert point_word & 0x3F >= 48  # point suffix region
        assert area_word & 0x3F < 48    # area suffix region

        # And the area word renders that same unmarked string, so the parse
        # is genuinely non-injective over the unmarked form.
        area = MortonIndexArray.from_words(
            np.asarray([area_word], dtype=np.uint64)
        )
        assert area.decimal_repr()[0] == unmarked

    def test_marker_is_illegal_below_order_29(self):
        for bad in ("1231p", "-6p", "3p"):
            with pytest.raises(ValueError, match="legal only"):
                decimal_to_word(bad)


class TestSpecPageParseSection:
    """Drift pin for the spec page's parse-side API block (issue #114).

    The section-4 tie-break is a parse-side contract, so the page names the
    public parse entry points; if one is renamed or dropped the page must be
    updated with it, and this catches the drift.
    """

    BEGIN = "<!-- parse:api:begin -->"
    END = "<!-- parse:api:end -->"

    def _block(self):
        page = (
            Path(__file__).resolve().parents[2] / "docs" / "specification.md"
        ).read_text()
        assert self.BEGIN in page and self.END in page, (
            "parse-api markers missing from the spec page"
        )
        block = page.split(self.BEGIN, 1)[1].split(self.END, 1)[0]
        # Normalize markdown wrapping so a reflow does not fail the pin.
        return " ".join(block.replace("**", "").split())

    def test_page_names_every_public_entry_point(self):
        block = self._block()
        for name in (
            "mortie.decimal_to_word",
            "mortie.decimals_to_words",
            "MortonIndexArray.from_decimal",
        ):
            assert name in block, f"spec page does not name {name}"

    def test_named_entry_points_exist(self):
        pytest.importorskip("pandas")
        from mortie import MortonIndexArray

        assert callable(mortie.decimal_to_word)
        assert callable(mortie.decimals_to_words)
        assert callable(MortonIndexArray.from_decimal)

    def test_page_states_the_unmarked_order29_rule(self):
        # The one thing a parse-side caller can get wrong; the prose claim and
        # the behavior are pinned together so neither can drift alone.
        assert "unmarked order-29 id parses to the area word" in self._block()

        arr = mortie.decimals_to_words(["3" + "1" * MAX_ORDER])
        assert int(arr[0]) & 0x3F < 48  # area suffix region, not point

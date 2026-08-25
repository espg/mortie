"""Public decimal->word parse surface (issue #114).

The emit direction (``decimal_repr`` / ``to_decimal`` / ``hive_path``) has
been public since issue #104; these are the pins for its public inverse:
``mortie.decimal_to_word`` (polymorphic since issue #187 -- the ``dtype``
return-shape flag on the scalar form, the vectorized Rust-backed kernel
``_decimals_to_words`` behind the array form), and the retained private
``_decimal_to_word`` alias.
"""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import mortie
from mortie import _rustie
from mortie.morton_index import (
    MAX_ORDER,
    MortonWord,
    _decimal_to_word,
    _decimals_to_words,
    decimal_to_word,
)


class TestPublicSurface:
    def test_exported_from_package_root(self):
        assert mortie.decimal_to_word is decimal_to_word
        assert "decimal_to_word" in mortie.__all__
        # The plural retired with the batch names (issue #187): the array
        # form of decimal_to_word is the surviving spelling.
        assert not hasattr(mortie, "decimals_to_words")
        assert "decimals_to_words" not in mortie.__all__

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
            "a = mortie.decimal_to_word(['3', '-6', '31123'])\n"
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


class TestScalarConstructor:
    """Type-disambiguated construction (issue #152): str is a decimal label."""

    def test_label_string_parses_as_decimal_label_not_packed_word(self):
        # The issue's headline regression: the inherited uint64 constructor
        # read the label "4331422412232" as a base-10 *packed word* and
        # silently constructed the wrong cell (<invalid 0x3f07ce4edc8>).
        s = MortonWord("4331422412232")
        assert int(s) == decimal_to_word("4331422412232", dtype=int)
        assert int(s) != 4331422412232  # the old packed reinterpretation
        assert str(s) == "4331422412232"

    @pytest.mark.parametrize("label", ["-31123", "41123", "3", "-6"])
    def test_label_round_trips_both_hemispheres(self, label):
        s = MortonWord(label)
        assert int(s) == decimal_to_word(label, dtype=int)
        assert str(s) == label

    def test_point_suffix_grammar_included(self):
        label = "3" + "1" * MAX_ORDER + "p"
        s = MortonWord(label)
        assert int(s) == decimal_to_word(label, dtype=int)
        assert str(s) == label
        assert int(s) != int(MortonWord(label[:-1]))  # area != point

    def test_numpy_str_subclass_is_a_label_too(self):
        s = MortonWord(np.str_("-31123"))
        assert int(s) == decimal_to_word("-31123", dtype=int)

    def test_clip_handles_a_limit_with_no_room_for_text(self):
        # Unreachable from the constructor (its limits are 80/160), but the
        # helper must not slice negatively and hand back *more* than limit.
        from mortie.morton_index import _clip

        for limit in (0, 1, 2, 3, 4):
            out = _clip("abcdefgh", limit)
            assert out.endswith("...")
            assert len(out) <= max(limit, 3)

    def test_zero_d_str_array_is_a_label_too(self):
        # ``np.array("31123")`` -- what ``arr[()]`` / an h5py attr read can
        # hand over -- would slip past the str guard and be read as a
        # base-10 word by numpy.uint64. Unwrapped, it takes the label path.
        s = MortonWord(np.array("31123"))
        assert int(s) == decimal_to_word("31123", dtype=int)
        assert str(s) == "31123"

    def test_zero_d_bytes_array_is_refused_like_bytes(self):
        with pytest.raises(TypeError, match="ambiguous"):
            MortonWord(np.array(b"31123"))

    def test_zero_d_numeric_array_keeps_numpy_parity(self):
        # Only "S"/"U" 0-d arrays are unwrapped; a numeric one still goes
        # straight through to numpy.uint64, unchanged.
        arr = np.array(5347397355232559123, dtype=np.uint64)
        assert int(MortonWord(arr)) == int(np.uint64(arr))
        assert int(MortonWord(np.array(5))) == 5

    @pytest.mark.parametrize(
        "bad", ["", "-", "0123", "7123", "31023", "3125", "x123",
                "3" + "1" * 30, "p", "-p", "31111p", "5347397355232559123"]
    )
    def test_invalid_label_raises_pointed_value_error(self, bad):
        # Names the input and the grammar -- never silently constructs. The
        # last case is a packed word *as a string*: digits above the 1..4/1..6
        # grammar make it an invalid label, not a word.
        with pytest.raises(ValueError, match="not a decimal Morton label"):
            MortonWord(bad)
        with pytest.raises(ValueError, match=re.escape(repr(bad))):
            MortonWord(bad)

    @pytest.mark.parametrize(
        "raw",
        [b"4331422412232", b"-31123", bytearray(b"3123"),
         np.bytes_(b"4331422412232")],
    )
    def test_bytes_is_refused_not_silently_reinterpreted(self, raw):
        # h5py/h5coro hand string attrs back as bytes, and numpy has two
        # readings for bytes-like input: b"4331422412232" becomes the base-10
        # *packed word* (the exact silent misconstruction this constructor
        # exists to close for str), while a bytearray becomes a raw *buffer*
        # -- a uint8-per-byte array. Neither is the label that was meant, so
        # refuse both instead of guessing.
        with pytest.raises(TypeError, match="bytes-like input is ambiguous"):
            MortonWord(raw)

    def test_bytes_refusal_names_both_ways_out(self):
        with pytest.raises(TypeError) as exc:
            MortonWord(b"4331422412232")
        assert "decode('ascii')" in str(exc.value)
        assert "int" in str(exc.value)
        # And the way out actually works.
        assert int(MortonWord(b"4331422412232".decode("ascii"))) == (
            decimal_to_word("4331422412232", dtype=int)
        )

    def test_error_message_is_bounded_for_a_huge_argument(self):
        # The message quotes what it was handed, and the caller controls that
        # length -- a megabyte of garbage must not become a megabyte of
        # exception (it lands in logs and tracebacks).
        bad = "9" * 100_000
        with pytest.raises(ValueError) as exc:
            MortonWord(bad)
        assert len(str(exc.value)) < 1_000
        assert "9999..." in str(exc.value)
        assert "not a decimal Morton label" in str(exc.value)

    @pytest.mark.parametrize(
        "word",
        [0, 1, 5347397355232559123, 2**64 - 1, True, False, 1.5,
         np.uint64(5347397355232559123), np.uint32(7), np.int64(9)],
    )
    def test_int_forms_are_byte_for_byte_uint64(self, word):
        # Non-str construction is untouched: whatever numpy.uint64 makes of
        # an int-like argument, this constructor makes of it too. Pinned
        # against numpy itself rather than against hand-written expectations,
        # so a numpy coercion change shows up here as a *parity* break.
        assert int(MortonWord(word)) == int(np.uint64(word))
        assert type(MortonWord(word)) is MortonWord
        assert int(MortonWord()) == int(np.uint64())

    @pytest.mark.parametrize(
        ("bad", "expected"),
        [(None, TypeError), (-1, OverflowError), (2**64, OverflowError)],
    )
    def test_int_form_error_parity_with_uint64(self, bad, expected):
        # The other half of parity: the arguments numpy.uint64 *refuses* must
        # be refused the same way, with the same exception type. (bytes is the
        # one deliberate divergence -- see the bytes tests above.)
        with pytest.raises(expected):
            np.uint64(bad)
        with pytest.raises(expected):
            MortonWord(bad)

    def test_buffer_input_follows_numpy_and_is_not_this_type(self):
        # Documented parity edge: numpy reads a buffer as an *array*, so the
        # constructor hands back exactly what numpy.uint64 does -- a plain
        # ndarray, not a MortonWord. Pinned so the docstring claim
        # cannot drift.
        out = MortonWord(memoryview(b"123"))
        expected = np.uint64(memoryview(b"123"))
        assert type(out) is np.ndarray
        assert np.array_equal(out, expected)

    def test_float_truncates_exactly_as_numpy_does(self):
        # numpy parity by choice, not an oversight: the float is truncated,
        # never rounded, and never refused.
        assert int(MortonWord(1.9)) == int(np.uint64(1.9)) == 1

    def test_int_form_stays_lazy_on_invalid_words(self):
        # Eager validation is the *label* constructor's posture only; a bad
        # packed word still constructs and renders <invalid ...> lazily.
        s = MortonWord(0xF000000000000000)
        assert str(s).startswith("<invalid")

    def test_label_constructed_scalar_pickles_as_itself(self):
        import pickle

        s = pickle.loads(pickle.dumps(MortonWord("-31123")))
        assert isinstance(s, MortonWord)
        assert str(s) == "-31123"


class TestScalarAccessors:
    """.decimal / .order / .base_cell: strict data queries (issue #152)."""

    def test_decimal_matches_str_rendering_for_valid_words(self):
        assert MortonWord("-31123").decimal == "-31123"
        label = "3" + "1" * MAX_ORDER + "p"
        assert MortonWord(label).decimal == label == str(MortonWord(label))

    @pytest.mark.parametrize("accessor", ["decimal", "order", "base_cell"])
    def test_accessors_raise_on_the_empty_sentinel(self, accessor):
        # Accessors are data queries: they raise rather than propagate a
        # sentinel string onward (espg ruling on PR #212, issue #152).
        with pytest.raises(ValueError, match="empty sentinel"):
            getattr(MortonWord(0), accessor)

    @pytest.mark.parametrize("accessor", ["decimal", "order", "base_cell"])
    def test_accessors_raise_on_an_invalid_word(self, accessor):
        # The message names the word and why: it decodes to no legal cell.
        with pytest.raises(
            ValueError, match="0xf000000000000000 decodes to no legal cell"
        ):
            getattr(MortonWord(0xF000000000000000), accessor)

    def test_display_stays_lazy_where_accessors_are_strict(self):
        # Same invalid words, same instant: repr/str/format still never
        # raise -- the strict posture is confined to the accessors.
        for word in (0, 0xF000000000000000):
            s = MortonWord(word)
            assert str(s) in ("<NA>",) or str(s).startswith("<invalid")
            assert repr(s) == str(s) == f"{s}"

    def test_base_cell_matches_the_array_kernel(self):
        for label, expected in (("-31123", 8), ("31123", 2), ("3", 2),
                                ("-6", 11)):
            s = MortonWord(label)
            assert s.base_cell == expected
            assert s.base_cell == int(
                _rustie.rust_mi_base_cell_of(
                    np.asarray([int(s)], dtype=np.uint64))[0]
            )

    def test_flat_export(self):
        assert mortie.MortonWord is MortonWord
        assert "MortonWord" in mortie.__all__

    def test_order_matches_orders_of(self):
        for label, expected in (("-31123", 4), ("3", 0),
                                ("3" + "1" * MAX_ORDER + "p", MAX_ORDER)):
            s = MortonWord(label)
            assert s.order == expected
            assert s.order == int(mortie.orders_of(s)[0])

    def test_arithmetic_still_demotes_to_bare_uint64(self):
        # The constructor override must not touch numeric behavior: numpy
        # scalar arithmetic keeps returning the base uint64, exactly as
        # before -- a derived value never masquerades as a valid address.
        s = MortonWord("-31123")
        out = s + np.uint64(1)
        assert isinstance(out, np.uint64)
        assert not isinstance(out, MortonWord)
        assert int(out) == int(s) + 1
        assert s == np.uint64(int(s))  # comparisons stay word-valued
        assert hash(s) == hash(np.uint64(int(s)))


class TestScalarDtypeFlag:
    def test_default_is_numpy_uint64(self):
        word = decimal_to_word("-31123")
        assert isinstance(word, np.uint64)
        assert not isinstance(word, MortonWord)

    def test_python_int(self):
        word = decimal_to_word("-31123", dtype=int)
        assert type(word) is int

    def test_morton_index_scalar_displays_as_the_decimal_string(self):
        word = decimal_to_word("-31123", dtype=MortonWord)
        assert isinstance(word, MortonWord)
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
        class MyScalar(MortonWord):
            pass

        got = decimal_to_word("3123", dtype=MyScalar)
        assert type(got) is MyScalar


class TestVectorized:
    def test_matches_the_scalar_elementwise(self):
        ids = ["1", "-6", "3123", "-31123", "2" + "4" * MAX_ORDER]
        got = _decimals_to_words(ids)
        assert got.dtype == np.uint64
        assert list(got) == [decimal_to_word(s) for s in ids]

    def test_shape_is_preserved(self):
        arr = np.array([["1", "-6"], ["3123", "-31123"]])
        assert _decimals_to_words(arr).shape == (2, 2)

    def test_empty_input(self):
        for empty in ([], (), np.array([], dtype="<U32")):
            out = _decimals_to_words(empty)
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
            _decimals_to_words(empty)

    def test_bare_string_is_rejected_not_treated_as_one_element(self):
        # np.asarray("3123") is a 0-d array, so this would otherwise return a
        # 0-d word -- a silent trap on the singular/plural name pair.
        with pytest.raises(TypeError, match="takes the scalar path"):
            _decimals_to_words("3123")

    def test_object_array_non_string_names_the_offender(self):
        with pytest.raises(TypeError, match="got int"):
            _decimals_to_words(np.array(["3123", 7], dtype=object))

    def test_object_array_of_strings(self):
        arr = np.array(["3123", "-6"], dtype=object)
        assert list(_decimals_to_words(arr)) == [
            decimal_to_word("3123"),
            decimal_to_word("-6"),
        ]

    def test_non_string_input_rejected_not_coerced(self):
        # np.asarray([1, 2], dtype=str) would silently become the order-0 ids
        # "1"/"2"; a parse surface must not invent input.
        with pytest.raises(TypeError, match="expects decimal Morton strings"):
            _decimals_to_words([1, 2])

    def test_error_names_the_first_bad_id_in_input_order(self):
        with pytest.raises(ValueError, match="'0123'"):
            _decimals_to_words(["3123", "0123", "7123"])


class TestRoundTrip:
    @pytest.mark.parametrize("order", [0, 1, 13, 27, 28, MAX_ORDER])
    def test_word_to_decimal_to_word_identity(self, order):
        pytest.importorskip("pandas")
        from mortie import MortonIndexArray

        lats = np.array([-70.0, -20.0, 0.0, 20.0, 70.0])
        lons = np.array([-170.0, -45.0, 0.0, 45.0, 170.0])
        arr = MortonIndexArray.from_latlon(lats, lons, order=order)
        words = np.asarray(arr._data, dtype=np.uint64)
        assert np.array_equal(_decimals_to_words(arr.to_decimal()), words)

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
            np.asarray(arr._data, dtype=np.uint64), _decimals_to_words(ids)
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
            "MortonIndexArray.from_decimal",
        ):
            assert name in block, f"spec page does not name {name}"

    def test_named_entry_points_exist(self):
        pytest.importorskip("pandas")
        from mortie import MortonIndexArray

        assert callable(mortie.decimal_to_word)
        assert callable(MortonIndexArray.from_decimal)

    def test_page_states_the_unmarked_order29_rule(self):
        # The one thing a parse-side caller can get wrong; the prose claim and
        # the behavior are pinned together so neither can drift alone.
        assert "unmarked order-29 id parses to the area word" in self._block()

        arr = mortie.decimal_to_word(["3" + "1" * MAX_ORDER])
        assert int(arr[0]) & 0x3F < 48  # area suffix region, not point

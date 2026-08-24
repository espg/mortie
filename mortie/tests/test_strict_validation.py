"""Family-wide strict input validation (issue #194).

One posture everywhere, per the ruling on issue #194: float-typed word and
offset arrays are refused rather than truncated, out-of-range values are
refused before any narrowing cast rather than wrapped, and the refusal names
the parameter and the offending value.  The two historical bug classes stay
pinned by name: the issue #185 uncatchable-panic arc (a bad value crossing
into Rust unchecked) and PR #192's silent uint64 wrap.

Valid paths are pinned byte-identical against
``data/strict_validation_goldens.json``, captured at ``4900a7e`` -- the
commit *before* the validators were adopted -- by
``generate_strict_goldens.py``.
"""

import json
import pathlib
import warnings

import numpy as np
import pytest

import mortie
from mortie._validate import _as_offsets, _as_u64

GOLDENS = json.loads(
    (pathlib.Path(__file__).parent / "data" /
     "strict_validation_goldens.json").read_text())


def _u64(key):
    """Load a golden entry as a uint64 array.

    Parameters
    ----------
    key : str
        Golden entry name.

    Returns
    -------
    numpy.ndarray
        The pinned words as ``uint64``.
    """
    return np.asarray(GOLDENS[key], dtype=np.uint64)


WORDS = _u64("words")
WORDS_B = _u64("words_b")


class TestValidators:
    """The shared validators themselves (hoisted from the toc module)."""

    def test_u64_refuses_floats(self):
        with pytest.raises(ValueError, match="w must be integer-typed"):
            _as_u64(np.asarray([1.5]), "w")
        # Integral-valued floats are still float-typed: refused, not trusted.
        with pytest.raises(ValueError, match="w must be integer-typed"):
            _as_u64(np.asarray([2.0]), "w")

    def test_u64_refuses_negative_naming_value(self):
        with pytest.raises(ValueError, match=r"w must be non-negative, got -7"):
            _as_u64(np.asarray([3, -7, -2], dtype=np.int64), "w")

    def test_u64_passes_top_bit_words(self):
        # Base cells 7-11 set bit 63 (spec section 1): large uint64 words are
        # valid and must survive unchanged.
        big = np.asarray([2**63 + 5], dtype=np.uint64)
        assert _as_u64(big, "w")[0] == np.uint64(2**63 + 5)

    def test_u64_accepts_untyped_empty(self):
        # The Toc-source ruling: an untyped empty container is not numeric,
        # it is empty.
        out = _as_u64([], "w")
        assert out.size == 0 and out.dtype == np.uint64

    def test_offsets_refuse_floats(self):
        with pytest.raises(ValueError, match="offsets must be integer-typed"):
            _as_offsets(np.asarray([0.0, 2.9]))

    def test_offsets_refuse_uint64_wrap_naming_value(self):
        # The PR #192 wrap class: >= 2**63 would wrap negative through the
        # int64 cast and the kernel would then describe the wrapped copy.
        bad = np.asarray([0, 2**63 + 5], dtype=np.uint64)
        with pytest.raises(
                ValueError,
                match=r"offsets must fit in int64, got 9223372036854775813"):
            _as_offsets(bad)

    def test_offsets_valid_passthrough(self):
        out = _as_offsets([0, 2, 4])
        assert out.dtype == np.int64 and out.tolist() == [0, 2, 4]

    @pytest.mark.parametrize("bad,dtype", [
        (["a"], "<U1"),
        ([None, 1], "object"),
        ([2.0, object()], "object"),
    ])
    def test_offsets_non_numeric_get_the_family_message(self, bad, dtype):
        """Strings and ``None`` are refused in this family's register.

        They used to reach a trial ``int64`` cast, which surfaced numpy's own
        ``invalid literal for int()`` / ``TypeError`` instead (issue #194
        review).
        """
        with pytest.raises(ValueError,
                           match=rf"offsets must be integer-typed, got dtype {dtype}"):
            _as_offsets(bad)

    def test_offsets_nan_named_under_warnings_as_errors(self):
        """NaN offsets are refused by name even with warnings-as-errors.

        The trial cast raised ``RuntimeWarning: invalid value encountered in
        cast``, which under ``-W error`` (or a downstream
        ``filterwarnings = error``) became the raised exception and hid the
        named ``ValueError`` (issue #194 review).
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError,
                               match="offsets must be integer-typed, got dtype float64"):
                _as_offsets(np.asarray([0.0, np.nan]))

    def test_offsets_object_dtype_python_int_still_named(self):
        """A Python int so large it lands as ``object`` is still named."""
        with pytest.raises(ValueError,
                           match=r"offsets must fit in int64, got 10*$"):
            _as_offsets([0, 10**40])


def _goldens_module():
    """Import the golden generator as a module.

    Returns
    -------
    module
        ``generate_strict_goldens``, imported from this directory.
    """
    import importlib.util
    path = pathlib.Path(__file__).parent / "generate_strict_goldens.py"
    spec = importlib.util.spec_from_file_location("generate_strict_goldens",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_valid_paths_byte_identical_to_pre_change_goldens():
    """Every touched entry point answers exactly as it did at ``4900a7e``.

    The JSON was captured *before* the strict validators were adopted; a
    difference here means the posture change altered a valid path.
    """
    got = _goldens_module().capture()
    assert set(got) == set(GOLDENS)
    for key, want in GOLDENS.items():
        assert got[key] == want, f"{key} diverged from pre-change capture"


# --- refusals at every phase-2 entry point ---------------------------------

FLOAT_WORDS = np.asarray([1.5, 2.0])
NEG_WORDS = np.asarray([3, -7], dtype=np.int64)
OFF2 = [0, WORDS.size, WORDS.size + WORDS_B.size]
RAGGED = np.concatenate([WORDS, WORDS_B])

WORD_CALLS = [
    ("compress_moc", "morton", lambda w: mortie.compress_moc(w)),
    ("moc_to_order", "morton", lambda w: mortie.moc_to_order(w, 7)),
    ("moc_or_a", "a", lambda w: mortie.moc_or(w, WORDS_B)),
    ("moc_or_b", "b", lambda w: mortie.moc_or(WORDS, w)),
    ("moc_and_a", "a", lambda w: mortie.moc_and(w, WORDS_B)),
    ("moc_and_b", "b", lambda w: mortie.moc_and(WORDS, w)),
    ("moc_intersects_a", "a", lambda w: mortie.moc_intersects(w, WORDS_B)),
    ("moc_intersects_b", "b", lambda w: mortie.moc_intersects(WORDS, w)),
    ("moc_minus_a", "a", lambda w: mortie.moc_minus(w, WORDS_B)),
    ("moc_minus_b", "b", lambda w: mortie.moc_minus(WORDS, w)),
    ("moc_xor_a", "a", lambda w: mortie.moc_xor(w, WORDS_B)),
    ("moc_xor_b", "b", lambda w: mortie.moc_xor(WORDS, w)),
    ("moc_not", "cover", lambda w: mortie.moc_not(w)),
    ("moc_not_domain", "domain", lambda w: mortie.moc_not(WORDS, domain=w)),
    ("common_ancestor", "morton", lambda w: mortie.common_ancestor(w)),
    ("moc_min", "morton", lambda w: mortie.moc_min(w)),
    ("split_base_cells", "words", lambda w: mortie.split_base_cells(w)),
    ("moc_to_order_ragged", "morton",
     lambda w: mortie.moc_to_order(w, 7, offsets=[0, w.size])),
    ("moc_and_ragged", "b",
     lambda w: mortie.moc_and(WORDS, w, offsets=[0, w.size])),
    ("moc_and_ragged_a", "a",
     lambda w: mortie.moc_and(w, RAGGED, offsets=OFF2)),
    ("moc_intersects_ragged", "b",
     lambda w: mortie.moc_intersects(WORDS, w, offsets=[0, w.size])),
    ("common_ancestor_ragged", "morton",
     lambda w: mortie.common_ancestor(w, offsets=[0, w.size])),
    # w[-1] so the scalar sees the offending element in both refusal tests
    # (float 2.0 is still float-typed; -7 is the negative).
    ("generate_morton_children_scalar", "parent_morton",
     lambda w: mortie.generate_morton_children(w[-1], 6)),
    ("generate_morton_children_array", "parent_morton",
     lambda w: mortie.generate_morton_children(w, 6)),
    ("clip2order", "midx", lambda w: mortie.clip2order(3, w)),
    ("orders_of", "morton", lambda w: mortie.orders_of(w)),
    ("is_point", "morton", lambda w: mortie.is_point(w)),
    ("infer_order_from_morton", "morton",
     lambda w: mortie.infer_order_from_morton(w)),
    ("validate_morton", "morton", lambda w: mortie.validate_morton(w)),
]


@pytest.mark.parametrize("name,param,call",
                         WORD_CALLS, ids=[c[0] for c in WORD_CALLS])
def test_float_words_refused(name, param, call):
    """Float-typed words raise, naming the parameter (the #185 arc class)."""
    with pytest.raises(ValueError,
                       match=rf"{param} must be integer-typed"):
        call(FLOAT_WORDS)


@pytest.mark.parametrize("name,param,call",
                         WORD_CALLS, ids=[c[0] for c in WORD_CALLS])
def test_negative_words_refused_naming_value(name, param, call):
    """Negative words raise instead of wrapping, naming the value."""
    with pytest.raises(ValueError,
                       match=rf"{param} must be non-negative, got -7"):
        call(NEG_WORDS)


OFFSET_CALLS = [
    ("moc_to_order", lambda o: mortie.moc_to_order(RAGGED, 7, offsets=o)),
    ("moc_and", lambda o: mortie.moc_and(WORDS, RAGGED, offsets=o)),
    ("moc_intersects",
     lambda o: mortie.moc_intersects(WORDS, RAGGED, offsets=o)),
    ("common_ancestor", lambda o: mortie.common_ancestor(RAGGED, offsets=o)),
    ("polygons_to_morton_mocs",
     lambda o: mortie.polygons_to_morton_mocs(
         [0.0, 0.0, 8.0], [0.0, 8.0, 0.0], o, order=6)),
    ("toc_reduce",
     lambda o: mortie.toc_reduce(
         mortie.time2toc(np.asarray([10**15, 2 * 10**15])), offsets=o)),
    ("from_wkb", lambda o: mortie.from_wkb(b"", order=6, offsets=o)),
]


@pytest.mark.parametrize("name,call",
                         OFFSET_CALLS, ids=[c[0] for c in OFFSET_CALLS])
def test_float_offsets_refused(name, call):
    """Float offsets raise instead of truncating toward a wrong boundary."""
    with pytest.raises(ValueError, match=r"offsets must be integer-typed"):
        call(np.asarray([0.0, 2.9]))


@pytest.mark.parametrize("name,call",
                         OFFSET_CALLS, ids=[c[0] for c in OFFSET_CALLS])
def test_uint64_offsets_past_int63_refused(name, call):
    """The PR #192 wrap class, at every offsets-taking entry point.

    A uint64 offset at or above 2**63 used to wrap negative through the
    int64 cast; now it is refused, naming the value that was passed rather
    than the wrapped copy.
    """
    bad = np.asarray([0, 2**63 + 5], dtype=np.uint64)
    with pytest.raises(
            ValueError,
            match=r"offsets must fit in int64, got 9223372036854775813"):
        call(bad)


def test_python_int_offsets_past_int64_named():
    """A plain-int offset past int64 is named too (numpy coerces the list
    to float64, which must not degrade the message to a dtype complaint)."""
    with pytest.raises(ValueError,
                       match=r"offsets must fit in int64, got 10000000000000000000"):
        mortie.moc_to_order(RAGGED, 7, offsets=[0, 10**19])


# --- phase 3: the remaining word surfaces ----------------------------------

WORD_CALLS_P3 = [
    ("mort2norm", "morton", lambda w: mortie.mort2norm(w)),
    ("mort2geo", "morton", lambda w: mortie.mort2geo(w)),
    ("mort2bbox", "morton", lambda w: mortie.mort2bbox(w)),
    ("mort2polygon", "morton", lambda w: mortie.mort2polygon(w)),
    ("morton_buffer", "morton_indices", lambda w: mortie.morton_buffer(w, k=1)),
    ("morton_buffer_meters", "morton_indices",
     lambda w: mortie.morton_buffer_meters(w, width_m=5000.0)),
    ("to_geometry", "morton", lambda w: mortie.to_geometry(w, dissolve=False)),
    # The *default* spelling routes through dissolve.py, whose own coercion
    # used to truncate/wrap behind the validator (review of phase 3).
    ("to_geometry_dissolved", "morton", lambda w: mortie.to_geometry(w)),
    ("to_wkb", "morton", lambda w: mortie.to_wkb(w)),
    ("to_wkb_per_cell", "morton",
     lambda w: mortie.to_wkb(w, dissolve=False)),
    ("to_wkt", "morton", lambda w: mortie.to_wkt(w)),
    ("to_wkt_per_cell", "morton",
     lambda w: mortie.to_wkt(w, dissolve=False)),
    ("Moc_source", "source", lambda w: mortie.Moc(np.asarray(w))),
    ("Moc_operand", "operand",
     lambda w: mortie.Moc(WORDS) & np.asarray(w)),
]


@pytest.mark.parametrize("name,param,call",
                         WORD_CALLS_P3, ids=[c[0] for c in WORD_CALLS_P3])
def test_negative_words_refused_p3(name, param, call):
    """Negative words raise instead of wrapping, at the phase-3 surfaces."""
    with pytest.raises(ValueError,
                       match=rf"{param} must be non-negative, got -7"):
        call(NEG_WORDS)


@pytest.mark.parametrize(
    "name,param,call",
    [c for c in WORD_CALLS_P3 if c[0] not in ("Moc_source",)],
    ids=[c[0] for c in WORD_CALLS_P3 if c[0] not in ("Moc_source",)])
def test_float_words_refused_p3(name, param, call):
    """Float-typed words raise at the phase-3 surfaces.

    ``Moc(source)`` is excluded by design: a float array there is *geometry*
    (ring coordinates), never words -- its words branch is gated on an
    integer dtype, so no float can reach it.
    """
    with pytest.raises(ValueError,
                       match=rf"{param} must be integer-typed"):
        call(FLOAT_WORDS)


class TestPrefixTrieException:
    """The one deliberate carve-out from the strict-negative rule.

    ``split_children`` branches on the decimal characteristic, whose first
    column *is* the sign (bit 63, the southern base cells), and its golden
    fixtures pin the ``int64`` bit-view of packed words as an input form --
    so the signed view stays accepted there, while floats are refused like
    everywhere else (issue #194).
    """

    def test_split_children_accepts_int64_bit_view(self):
        signed = RAGGED.view(np.int64)
        assert (signed < 0).any()  # southern words present, negative as i64
        want = sorted(c.characteristic
                      for c in mortie.split_children(RAGGED, max_depth=2))
        got = sorted(c.characteristic
                     for c in mortie.split_children(signed, max_depth=2))
        assert got == want == GOLDENS["split_children_roots"]

    def test_split_children_refuses_floats(self):
        with pytest.raises(ValueError,
                           match="morton_array must be integer-typed"):
            mortie.split_children(FLOAT_WORDS, max_depth=2)

    @pytest.mark.parametrize("scalar", [np.uint64(RAGGED[0]), int(RAGGED[0])])
    def test_split_children_still_refuses_scalars(self, scalar):
        """A 0-D word keeps its rank refusal (issue #194 review).

        Validating the seam must not loosen it: an ``atleast_1d`` ahead of
        the 1-D check would promote a scalar past it, accepting input that
        ``main`` refused.
        """
        with pytest.raises(ValueError, match="non-empty 1-D integer array"):
            mortie.split_children(scalar, max_depth=2)
        with pytest.raises(ValueError, match="non-empty 1-D integer array"):
            mortie.morton_polygon_from_array(scalar, 1)


class TestArrowArrayLikeIntakes:
    """The two ``arrow.py`` seams that take ``array_like``, not typed columns.

    ``from_wkb`` / ``polygons_to_morton_mocs`` really are strict by
    construction (pyarrow in, pyarrow out), but ``from_morton_index`` and
    ``export_c_array`` are documented ``array_like`` intakes -- a raw numpy
    array carries whatever dtype the caller gave it, and ``export_c_array``
    hands the words straight across an FFI boundary (issue #194 review).
    """

    def test_export_c_array_refuses_floats(self):
        # pyarrow-free: the C Data Interface surface is numpy + Rust only.
        from mortie import arrow as marrow
        with pytest.raises(ValueError, match="words must be integer-typed"):
            marrow.export_c_array(FLOAT_WORDS)

    def test_export_c_array_refuses_negative_naming_value(self):
        from mortie import arrow as marrow
        with pytest.raises(ValueError, match="words must be non-negative, got -7"):
            marrow.export_c_array(NEG_WORDS)

    def test_export_c_array_valid_words_unaffected(self):
        from mortie import arrow as marrow
        assert len(marrow.export_c_array(WORDS)) == 2

    def test_from_morton_index_refuses_floats(self):
        pytest.importorskip("pyarrow")
        from mortie import arrow as marrow
        with pytest.raises(ValueError, match="array must be integer-typed"):
            marrow.from_morton_index(FLOAT_WORDS)

    def test_from_morton_index_refuses_negative_naming_value(self):
        pytest.importorskip("pyarrow")
        from mortie import arrow as marrow
        with pytest.raises(ValueError, match="array must be non-negative, got -7"):
            marrow.from_morton_index(NEG_WORDS)

    def test_from_morton_index_valid_words_unaffected(self):
        pytest.importorskip("pyarrow")
        from mortie import arrow as marrow
        out = marrow.from_morton_index(WORDS)
        assert [int(v.value.as_py()) for v in out] == [int(w) for w in WORDS]

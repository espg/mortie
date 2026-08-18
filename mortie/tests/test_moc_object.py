"""Tests for the geometry-first ``Moc`` object and the ``mortie.moc`` shim (issue #196).

The object is a thin view over the canonical word array: the commitment is that
every method is a *single* delegation to a kernel function, and the tests here
are written to falsify that rather than to restate it -- see
:class:`TestDelegationParity` (value parity, method by method) and
:func:`test_every_public_method_is_a_single_kernel_delegation` (the structural
pin over the source).
"""

import ast
import copy
import inspect
import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest

import mortie
from mortie import Moc, moc
from mortie.moc_object import _KERNEL_NAMES


def box(west, east, south, north):
    """A closed GeoJSON ring for a lon/lat box."""
    return [[west, south], [east, south], [east, north],
            [west, north], [west, south]]


# The SERC ~4 km AOI the zagg demo/07_minimal.ipynb notebook covers, spelled
# exactly as that notebook spells it -- a bare dict with no "type" members.
SERC_AOI = {"features": [{"geometry": {"coordinates": [[
    [-76.56, 38.87], [-76.50, 38.87], [-76.50, 38.91],
    [-76.56, 38.91], [-76.56, 38.87],
]]}}]}
SERC_RING = np.asarray(SERC_AOI["features"][0]["geometry"]["coordinates"][0])

# Golden words for the SERC AOI at max_cells=32 -- the determinism pin ("same
# input + same version -> same MOC", byte for byte).  A small adaptive budget
# keeps the golden inline-able; the default multi-order cover of the same ring
# is pinned by construction against `morton_coverage_moc` in the parity tests.
SERC_GOLDEN_MAX32 = np.array([
    5347298278532710413, 5347298295712579597, 5347298312892448781,
    5347298622130094093, 5347298639309963277, 5347298656489832461,
    5347298673669701644, 5347298742389178381, 5347298759569047565,
    5347298793928785933, 5347298811108655116, 5347301096031256589,
    5347301113211125773, 5347301130390994957, 5347301147570864141,
    5347301181930602509, 5347301199110471693, 5347301216290340876,
    5347301285009817612, 5347301491168247820, 5347301628607201291,
    5347301903485108236, 5347392681913876492, 5347395636851376140,
], dtype=np.uint64)

SOLID = {"type": "Polygon", "coordinates": [box(-76.6, -76.4, 38.8, 39.0)]}
OVERLAP = {"type": "Polygon", "coordinates": [box(-76.5, -76.3, 38.8, 39.0)]}
DONUT = {"type": "Polygon", "coordinates": [box(-76.6, -76.4, 38.8, 39.0),
                                            box(-76.55, -76.45, 38.85, 38.95)]}
FAR = {"type": "Polygon", "coordinates": [box(-70.6, -70.4, 38.8, 39.0)]}
MULTI = {"type": "MultiPolygon",
         "coordinates": [[box(-76.6, -76.4, 38.8, 39.0)],
                         [box(-70.6, -70.4, 38.8, 39.0)]]}
FEATURES = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "geometry": SOLID},
    {"type": "Feature", "geometry": FAR},
]}


@pytest.fixture(scope="module")
def pair():
    """Two overlapping covers, for the set-op and predicate parity tests."""
    return Moc(SOLID), Moc(OVERLAP)


class TestConstructorForms:
    """Every documented constructor source, and what each resolves to."""

    def test_geojson_polygon_matches_the_coverage_kernel(self):
        ring = np.asarray(SOLID["coordinates"][0])
        assert np.array_equal(
            Moc(SOLID).words,
            mortie.morton_coverage_moc(ring[:, 1], ring[:, 0]),
        )

    def test_geojson_without_type_members(self):
        # 07_minimal's AOI carries no "type" -- the parser is structural.
        assert np.array_equal(
            moc(SERC_AOI).words,
            mortie.morton_coverage_moc(SERC_RING[:, 1], SERC_RING[:, 0]),
        )

    def test_feature_and_bare_geometry_agree(self):
        feature = {"type": "Feature", "geometry": SOLID}
        assert Moc(feature) == Moc(SOLID)

    def test_multipolygon_unions_its_parts(self):
        union = Moc(SOLID) | Moc(FAR)
        assert Moc(MULTI) == union
        assert Moc(MULTI).contains(Moc(SOLID))
        assert Moc(MULTI).contains(Moc(FAR))

    def test_feature_collection_matches_the_multipolygon(self):
        assert Moc(FEATURES) == Moc(MULTI)

    def test_inner_ring_carves_a_hole(self):
        # The even-odd multipart descent: a nested ring is a hole, not a part.
        centre = mortie.geo2mort(38.90, -76.50, order=18)
        assert mortie.moc_intersects(Moc(SOLID).words, centre)
        assert not mortie.moc_intersects(Moc(DONUT).words, centre)
        assert Moc(DONUT).within(Moc(SOLID))

    def test_bare_ring_array(self):
        ring = np.asarray(SOLID["coordinates"][0])
        assert Moc(ring) == Moc(SOLID)

    def test_ring_list_is_the_multipart_form(self):
        assert Moc(DONUT["coordinates"]) == Moc(DONUT)

    def test_words_array_round_trip(self):
        cover = Moc(SOLID)
        assert Moc(cover.words) == cover

    def test_protocol_object_round_trip(self):
        cover = Moc(SOLID)

        class Carrier:
            def __morton_moc__(self):
                return cover.words

        assert Moc(Carrier()) == cover
        assert Moc(cover) == cover

    def test_knobs_reach_the_kernel(self):
        ring = np.asarray(SOLID["coordinates"][0])
        for kwargs in ({"tolerance": 0.01}, {"max_cells": 64},
                       {"latitude": "geodetic-spherical"}):
            assert np.array_equal(
                Moc(SOLID, **kwargs).words,
                mortie.morton_coverage_moc(ring[:, 1], ring[:, 0], **kwargs),
            )

    def test_no_order_argument(self):
        # Multi-order by default is the whole point: there is no order knob.
        assert "order" not in inspect.signature(Moc.__init__).parameters

    @pytest.mark.parametrize("source, match", [
        ({"type": "Point", "coordinates": [0.0, 0.0]}, "Polygon / MultiPolygon"),
        ({"type": "Polygon"}, "no 'coordinates' member"),
        ({"type": "FeatureCollection", "features": []}, "no polygon rings"),
        ([1.0, 2.0, 3.0], "nesting depth"),
        ([[[[[0.0, 0.0]]]]], "nesting depth"),
    ])
    def test_bad_sources_raise_value_error(self, source, match):
        with pytest.raises(ValueError, match=match):
            Moc(source)

    def test_short_ring_is_the_kernel_error(self):
        with pytest.raises(ValueError, match="at least 3 vertices"):
            Moc([[0.0, 0.0], [1.0, 1.0]])


class TestNormalizationAndIdentity:
    """Eager compaction, immutability, equality and hashing."""

    def test_words_are_compacted_eagerly(self):
        cover = Moc(SOLID)
        assert np.array_equal(cover.words, mortie.compress_moc(cover.words))

    def test_uncompacted_input_compacts_at_construction(self):
        parent = np.atleast_1d(mortie.norm2mort(0, 3, 4))
        children = mortie.generate_morton_children(parent[0], 5)
        assert len(children) == 4
        assert np.array_equal(Moc(np.asarray(children)).words, parent)

    def test_words_are_read_only(self):
        cover = Moc(SOLID)
        assert not cover.words.flags.writeable
        with pytest.raises(ValueError):
            cover.words[0] = 0

    def test_instance_is_immutable(self):
        cover = Moc(SOLID)
        with pytest.raises(AttributeError, match="immutable"):
            cover.words = np.array([1], dtype=np.uint64)
        with pytest.raises(AttributeError, match="immutable"):
            del cover.words

    def test_pickle_round_trip(self):
        # Workers marshal their arguments: a cover must cross a process boundary
        # as easily as the word array it wraps.
        cover = Moc(SOLID)
        restored = pickle.loads(pickle.dumps(cover))
        assert restored == cover
        assert not restored.words.flags.writeable

    @pytest.mark.parametrize("clone", [copy.copy, copy.deepcopy])
    def test_copy_round_trip(self, clone):
        cover = Moc(SOLID)
        restored = clone(cover)
        assert restored == cover
        assert not restored.words.flags.writeable

    def test_slots_only(self):
        assert Moc.__slots__ == ("words",)
        assert not hasattr(Moc(SOLID), "__dict__")

    def test_equality_is_word_identity(self):
        assert Moc(SOLID) == Moc(SOLID)
        assert Moc(SOLID) != Moc(FAR)
        # Same polygon, different termination knob -> different words.
        assert Moc(SOLID) != Moc(SOLID, max_cells=64)

    def test_equality_against_a_non_moc_is_not_an_error(self):
        assert Moc(SOLID) != 5
        assert Moc(SOLID).__eq__(5) is NotImplemented

    def test_hash_matches_equality(self):
        assert hash(Moc(SOLID)) == hash(Moc(SOLID))
        assert len({Moc(SOLID), Moc(SOLID), Moc(FAR)}) == 2


class TestDelegationParity:
    """Every method equals the kernel call on ``.words`` -- the thin-view pin."""

    def test_intersects(self, pair):
        a, b = pair
        assert a.intersects(b) == mortie.moc_intersects(a.words, b.words)
        assert a.intersects(Moc(FAR)) == mortie.moc_intersects(
            a.words, Moc(FAR).words)

    def test_contains_and_within(self, pair):
        a, b = pair
        assert a.contains(b) == (mortie.moc_minus(b.words, a.words).size == 0)
        assert b.within(a) == (mortie.moc_minus(b.words, a.words).size == 0)
        assert a.contains(a) and a.within(a)
        assert not a.contains(Moc(FAR))

    @pytest.mark.parametrize("method, dunder, kernel", [
        ("union", "__or__", "moc_or"),
        ("intersection", "__and__", "moc_and"),
        ("difference", "__sub__", "moc_minus"),
        ("symmetric_difference", "__xor__", "moc_xor"),
    ])
    def test_set_ops(self, pair, method, dunder, kernel):
        a, b = pair
        expected = getattr(mortie, kernel)(a.words, b.words)
        result = getattr(a, method)(b)
        assert isinstance(result, Moc)
        assert np.array_equal(result.words, expected)
        # The dunder is the same bound method, not a re-implementation.
        assert getattr(Moc, dunder) is getattr(Moc, method)

    def test_set_ops_take_raw_words_and_protocol_objects(self, pair):
        a, b = pair
        assert a.union(b.words) == a.union(b)
        assert a.intersects(b.words) == a.intersects(b)

    @pytest.mark.parametrize("order", [5, 9, 12])
    def test_at_is_moc_to_order(self, order):
        cover = Moc(SERC_AOI)
        assert np.array_equal(cover.at(order),
                              mortie.moc_to_order(cover.words, order))

    def test_at_returns_the_array_not_a_moc(self):
        # A flat single-order list is not a MOC: re-normalizing would collapse
        # it straight back to the compact form.
        flat = Moc(SERC_AOI).at(9)
        assert isinstance(flat, np.ndarray)
        assert flat.dtype == np.uint64

    def test_at_forwards_the_max_cells_budget(self):
        with pytest.raises(ValueError, match="max_cells"):
            Moc(SERC_AOI).at(29, max_cells=16)
        assert Moc(SERC_AOI).at(20, max_cells=None).size > 0

    def test_from_polygon_is_the_coverage_kernel(self):
        ring = np.asarray(SOLID["coordinates"][0])
        assert Moc.from_polygon(ring[:, 1], ring[:, 0]) == Moc(SOLID)
        assert np.array_equal(
            Moc.from_polygon(ring[:, 1], ring[:, 0], tolerance=0.01).words,
            mortie.morton_coverage_moc(ring[:, 1], ring[:, 0], tolerance=0.01),
        )


# The kernel functions a Moc method is allowed to call, plus the two non-kernel
# callables a delegation may wrap itself in: `Moc` (re-wrap the result) and
# `_words` (coerce the operand).  Anything else in a method body is algebra the
# object promised not to have.
_ALLOWED_KERNELS = {
    "compress_moc", "moc_and", "moc_intersects", "moc_minus", "moc_or",
    "moc_to_order", "moc_xor", "morton_coverage_moc",
}
_ALLOWED_WRAPPERS = {"Moc", "_words"}


def _class_def(name):
    """The ``ast.ClassDef`` for *name* in ``mortie/moc_object.py``."""
    source = Path(mortie.moc_object.__file__).read_text()
    tree = ast.parse(source)
    return next(n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == name)


def test_every_public_method_is_a_single_kernel_delegation():
    """The falsifiable form of the thin-view commitment (issue #196).

    A public ``Moc`` method must be exactly one ``return`` of exactly one
    kernel call, optionally wrapped in ``Moc(...)`` and coercing its operand
    through ``_words(...)``.  Any other body -- a loop, a branch on cell
    values, a second kernel call -- is the finding this test exists to catch.
    """
    methods = [n for n in _class_def("Moc").body
               if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]
    assert {m.name for m in methods} == {
        "at", "contains", "difference", "from_polygon", "intersection",
        "intersects", "symmetric_difference", "union", "within",
    }
    for method in methods:
        body = [s for s in method.body
                if not (isinstance(s, ast.Expr)
                        and isinstance(s.value, ast.Constant))]
        assert len(body) == 1 and isinstance(body[0], ast.Return), (
            f"Moc.{method.name} is not a single return statement"
        )
        called = []
        for node in ast.walk(body[0]):
            if isinstance(node, ast.Call):
                func = node.func
                called.append(func.id if isinstance(func, ast.Name)
                              else getattr(func, "attr", "<expr>"))
        kernels = [c for c in called if c in _ALLOWED_KERNELS]
        assert len(kernels) == 1, (
            f"Moc.{method.name} calls {kernels or called}, not exactly one kernel"
        )
        extra = set(called) - _ALLOWED_KERNELS - _ALLOWED_WRAPPERS - {"cls"}
        assert not extra, f"Moc.{method.name} also calls {sorted(extra)}"


class TestReprAndContainer:
    """What the object tells you about itself."""

    def test_repr_reports_cells_and_order_range(self):
        cover = Moc(SOLID)
        orders = mortie.orders_of(cover.words)
        text = repr(cover)
        assert text.startswith(f"Moc({cover.words.size} cells,")
        assert f"orders {orders.min()}-{orders.max()}" in text
        assert "finest" in text

    def test_repr_of_a_single_order_cover(self):
        one = Moc(np.atleast_1d(mortie.norm2mort(0, 3, 4)))
        assert repr(one) == "Moc(1 cells, order 4, finest 407.476 km)"

    def test_repr_of_an_empty_cover(self):
        assert repr(Moc(np.array([], dtype=np.uint64))) == "Moc(0 cells)"

    def test_len_and_iter(self):
        cover = Moc(SOLID)
        assert len(cover) == cover.words.size
        assert np.array_equal(np.fromiter(cover, dtype=np.uint64), cover.words)

    def test_protocol_hands_back_the_canonical_words(self):
        cover = Moc(SOLID)
        handed = cover.__morton_moc__()
        assert handed is cover.words
        assert not handed.flags.writeable


class TestDeterminism:
    """Same input + same version -> same MOC, byte for byte."""

    def test_repeated_construction_is_byte_identical(self):
        first, second = Moc(SERC_AOI), Moc(SERC_AOI)
        assert first.words.tobytes() == second.words.tobytes()

    def test_golden_words(self):
        assert np.array_equal(Moc(SERC_AOI, max_cells=32).words,
                              SERC_GOLDEN_MAX32)

    def test_ring_order_does_not_matter_for_the_union(self):
        flipped = {"type": "MultiPolygon",
                   "coordinates": [MULTI["coordinates"][1],
                                   MULTI["coordinates"][0]]}
        assert Moc(flipped) == Moc(MULTI)


def test_07_minimal_acceptance_path():
    """`moc(aoi).at(9)` is the cover zagg demo/07_minimal.ipynb builds today.

    The notebook's ``coverage()`` calls ``morton_coverage_moc(lats, lons,
    order=9)`` on the AOI ring and hands the result to the store; the object
    form must land on exactly the same order-9 cells.
    """
    reference = mortie.moc_to_order(
        mortie.morton_coverage_moc(SERC_RING[:, 1], SERC_RING[:, 0], order=9), 9
    )
    assert np.array_equal(moc(SERC_AOI).at(9), reference)
    # ... and on the flat coverer's answer for the same ring at that order.
    assert np.array_equal(
        moc(SERC_AOI).at(9),
        np.sort(mortie.morton_coverage(SERC_RING[:, 1], SERC_RING[:, 0], order=9)),
    )


class TestMigrationShim:
    """`mortie.moc` is the constructor now; the old attributes deprecate out."""

    def test_moc_is_callable_and_builds_a_moc(self):
        assert isinstance(moc(SOLID), Moc)
        assert moc(SOLID) == Moc(SOLID)

    def test_call_forwards_every_knob(self):
        assert moc(SOLID, max_cells=64) == Moc(SOLID, max_cells=64)
        assert moc(SOLID, tolerance=0.01) == Moc(SOLID, tolerance=0.01)
        assert (moc(SOLID, latitude="geodetic-spherical")
                == Moc(SOLID, latitude="geodetic-spherical"))

    def test_moc_is_not_a_module(self):
        with pytest.raises(ModuleNotFoundError):
            __import__("mortie.moc")

    def test_kernel_roster_matches_the_package_exports(self):
        # Drift pin: the shim must resolve exactly the names the retired
        # submodule held, which are the ones __init__ re-exports from ._moc.
        from mortie import _moc
        public = {name for name in dir(_moc)
                  if not name.startswith("_") and callable(getattr(_moc, name))
                  and getattr(getattr(_moc, name), "__module__", "") ==
                  _moc.__name__}
        assert set(_KERNEL_NAMES) == public
        assert public <= set(mortie.__all__)

    @pytest.mark.parametrize("name", _KERNEL_NAMES)
    def test_deprecated_attribute_still_resolves_to_the_kernel(self, name):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assert getattr(moc, name) is getattr(mortie, name)

    def test_warning_fires_once_per_attribute(self):
        from mortie.moc_object import _MocNamespace

        shim = _MocNamespace()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            shim.moc_and, shim.moc_and, shim.moc_and
            assert len(caught) == 1
            assert issubclass(caught[0].category, DeprecationWarning)
            assert "mortie.moc.moc_and is deprecated" in str(caught[0].message)
            shim.moc_or
            assert len(caught) == 2

    def test_a_raising_filter_is_not_self_silencing(self):
        # The once-per-attribute budget is spent only when the warning is
        # actually delivered: under -W error::DeprecationWarning every access
        # must raise, not just the first one.
        from mortie.moc_object import _MocNamespace

        shim = _MocNamespace()
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            for _ in range(3):
                with pytest.raises(DeprecationWarning, match="moc_and is deprecated"):
                    shim.moc_and
        # ... and the name is still unrecorded, so a normal filter sees it once.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            shim.moc_and, shim.moc_and
            assert len(caught) == 1

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError, match="not the old"):
            moc.morton_coverage_moc

    def test_dir_lists_the_deprecated_names(self):
        assert dir(moc) == sorted(_KERNEL_NAMES)

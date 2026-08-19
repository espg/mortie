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

import numpy as np
import pytest

import mortie
from mortie import Moc, moc
from mortie.moc_object import _KERNEL_NAMES
from mortie.tests.delegation import class_def, delegation_violation


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
INNER = {"type": "Polygon", "coordinates": [box(-76.55, -76.45, 38.85, 38.95)]}
FEATURES = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "geometry": SOLID},
    {"type": "Feature", "geometry": FAR},
]}
OVERLAPPING_FEATURES = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "geometry": SOLID},
    {"type": "Feature", "geometry": OVERLAP},
]}
NESTED_FEATURES = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "geometry": SOLID},
    {"type": "Feature", "geometry": INNER},
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

    def test_overlapping_features_union_rather_than_cancel(self):
        # Even-odd is the rule *within* a geometry; across features it would
        # XOR the shared area away, and overlapping features are legal GeoJSON.
        assert Moc(OVERLAPPING_FEATURES) == Moc(SOLID) | Moc(OVERLAP)
        shared = mortie.geo2mort(38.90, -76.45, order=18)
        assert mortie.moc_intersects(Moc(OVERLAPPING_FEATURES).words, shared)

    def test_nested_features_union_rather_than_cancel(self):
        # The worse form of the same bug: even-odd would leave a hole where the
        # inner feature sits, or an empty cover.
        assert Moc(NESTED_FEATURES) == Moc(SOLID) | Moc(INNER)
        assert Moc(NESTED_FEATURES) == Moc(SOLID)
        centre = mortie.geo2mort(38.90, -76.50, order=18)
        assert mortie.moc_intersects(Moc(NESTED_FEATURES).words, centre)

    def test_a_hole_is_still_a_hole_inside_one_feature(self):
        # The union-across-features rule must not leak into a single geometry:
        # a donut's inner ring still carves.
        feature = {"type": "FeatureCollection",
                   "features": [{"type": "Feature", "geometry": DONUT}]}
        assert Moc(feature) == Moc(DONUT)
        assert Moc(feature) != Moc(SOLID)

    def test_inner_ring_carves_a_hole(self):
        # The even-odd multipart descent: a nested ring is a hole, not a part.
        centre = mortie.geo2mort(38.90, -76.50, order=18)
        assert mortie.moc_intersects(Moc(SOLID).words, centre)
        assert not mortie.moc_intersects(Moc(DONUT).words, centre)
        assert Moc(DONUT).within(Moc(SOLID))

    @pytest.mark.parametrize("ring", [
        box(179.5, -179.5, -1.0, 1.0),      # antimeridian-crossing
        box(-179.5, 179.5, -1.0, 1.0),      # ... and the other winding
        box(-180.0, 180.0, 89.0, 89.9),     # polar cap
        box(-30.0, 30.0, -89.9, -89.0),     # ... and the south pole
    ])
    def test_antimeridian_and_polar_rings_reach_the_kernel_unmangled(self, ring):
        # The ring -> (lats, lons) split is where a transposition or a bad
        # closing-vertex strip would show up first, and these are the shapes
        # that make it visible.
        xy = np.asarray(ring)
        geojson = {"type": "Polygon", "coordinates": [ring]}
        expected = mortie.morton_coverage_moc(xy[:, 1], xy[:, 0])
        assert np.array_equal(Moc(ring).words, expected)
        assert np.array_equal(Moc(geojson).words, expected)

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

    @pytest.mark.parametrize("kwargs, match", [
        ({"tolerance": 5.0}, "tolerance="),
        ({"max_cells": 8}, "max_cells="),
        ({"latitude": "geodetic-spherical"}, "latitude="),
        ({"tolerance": 5.0, "max_cells": 8}, "tolerance="),
    ])
    def test_coverage_knobs_are_refused_for_a_words_source(self, kwargs, match):
        # Silently ignoring them would let Moc(a.words, max_cells=8) read as
        # "re-cover at a smaller budget" while doing nothing -- and would accept
        # the tolerance+max_cells pair the coverer rejects.
        cover = Moc(SOLID)
        with pytest.raises(ValueError, match=match):
            Moc(cover.words, **kwargs)
        with pytest.raises(ValueError, match=match):
            Moc(cover, **kwargs)

    def test_set_ops_still_re_wrap_words_without_knobs(self):
        # The guard must not catch the internal Moc(moc_or(...)) re-wrap.
        a, b = Moc(SOLID), Moc(OVERLAP)
        assert isinstance(a | b, Moc)
        assert Moc(a.words) == a

    def test_no_order_argument(self):
        # Multi-order by default is the whole point: there is no order knob.
        assert "order" not in inspect.signature(Moc.__init__).parameters

    @pytest.mark.parametrize("source, match", [
        ({"type": "Point", "coordinates": [0.0, 0.0]}, "Polygon / MultiPolygon"),
        ({"type": "Polygon"}, "no 'coordinates' member"),
        ({"type": "FeatureCollection", "features": []}, "no polygon rings"),
        ({"type": "Feature", "geometry": None}, "null geometry"),
        ({"type": "FeatureCollection",
          "features": [{"type": "Feature", "geometry": None}]}, "null geometry"),
        ({"type": "GeometryCollection", "geometries": [SOLID]},
         "got a GeometryCollection"),
        ({"type": "Feature",
          "geometry": {"type": "GeometryCollection", "geometries": [SOLID]}},
         "got a GeometryCollection"),
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

    def test_empty_operand_predicates(self, pair):
        # An empty cover is reachable in one line (`a.difference(a)`), and it is
        # where the predicates stop agreeing: contains is vacuously True while
        # intersects is False.  Chosen, not inherited -- see the module
        # docstring's conservative-direction table.
        a, _ = pair
        empty = a.difference(a)
        assert len(empty) == 0
        assert empty == Moc(np.array([], dtype=np.uint64))
        assert a.contains(empty)
        assert not a.intersects(empty)
        assert empty.within(a)
        assert not empty.contains(a)
        assert not empty.intersects(empty)
        assert empty.contains(empty) and empty.within(empty)

    def test_empty_operand_set_ops(self, pair):
        a, _ = pair
        empty = a.difference(a)
        assert a.union(empty) == a
        assert a.intersection(empty) == empty
        assert a.difference(empty) == a
        assert a.symmetric_difference(empty) == a

    @pytest.mark.parametrize("order", [5, 9, 12])
    def test_to_order_is_moc_to_order(self, order):
        cover = Moc(SERC_AOI)
        assert np.array_equal(cover.to_order(order),
                              mortie.moc_to_order(cover.words, order))

    def test_to_order_returns_the_array_not_a_moc(self):
        # A flat single-order list is not a MOC: re-normalizing would collapse
        # it straight back to the compact form.
        flat = Moc(SERC_AOI).to_order(9)
        assert isinstance(flat, np.ndarray)
        assert flat.dtype == np.uint64

    def test_to_order_forwards_the_max_cells_budget(self):
        with pytest.raises(ValueError, match="max_cells"):
            Moc(SERC_AOI).to_order(29, max_cells=16)
        assert Moc(SERC_AOI).to_order(20, max_cells=None).size > 0

    def test_to_order_default_budget_does_not_drift_from_the_kernel(self):
        # `to_order` re-declares moc_to_order's default rather than deferring to it,
        # so the two have to be pinned together -- the shape test_moc_batch.py
        # uses to tie the batch default to the scalar one.
        assert Moc.to_order.__defaults__[-1] == mortie.moc_to_order.__defaults__[-1]

    def test_from_polygon_is_the_coverage_kernel(self):
        ring = np.asarray(SOLID["coordinates"][0])
        assert Moc.from_polygon(ring[:, 1], ring[:, 0]) == Moc(SOLID)
        assert np.array_equal(
            Moc.from_polygon(ring[:, 1], ring[:, 0], tolerance=0.01).words,
            mortie.morton_coverage_moc(ring[:, 1], ring[:, 0], tolerance=0.01),
        )

    @pytest.mark.parametrize("kwargs", [
        {"max_cells": 64},
        {"latitude": "geodetic-spherical"},
        {"latitude": "authalic"},
    ])
    def test_from_polygon_forwards_every_knob(self, kwargs):
        # `latitude` is the knob whose default ("authalic") differs from the
        # naive expectation, so the two conventions must not collapse.
        ring = np.asarray(SOLID["coordinates"][0])
        assert np.array_equal(
            Moc.from_polygon(ring[:, 1], ring[:, 0], **kwargs).words,
            mortie.morton_coverage_moc(ring[:, 1], ring[:, 0], **kwargs),
        )

    def test_latitude_conventions_are_not_the_same_cover(self):
        ring = np.asarray(SOLID["coordinates"][0])
        assert (Moc.from_polygon(ring[:, 1], ring[:, 0])
                != Moc.from_polygon(ring[:, 1], ring[:, 0],
                                    latitude="geodetic-spherical"))


# The kernel functions a Moc method is allowed to call, plus the two non-kernel
# roles a delegation may use: a wrapper that re-boxes the kernel's answer
# (`Moc(...)` / `cls(...)`) and the operand coercion `_words(...)`.  Anything
# else in a method body is algebra the object promised not to have.  The
# machinery itself -- the shape whitelist and the denied-node sweep -- lives in
# mortie/tests/delegation.py, shared with the Toc pin (issue #198).
_ALLOWED_KERNELS = {
    "moc_and", "moc_intersects", "moc_minus", "moc_or",
    "moc_to_order", "moc_xor", "morton_coverage_moc",
}


def _delegation_violation(method):
    """Why *method* is not a single kernel delegation, or ``None`` if it is.

    The Moc-specific instantiation of the shared pin: the returned expression
    must be a kernel call, that call re-boxed by ``Moc(...)`` / ``cls(...)``,
    or the emptiness test ``<kernel>(...).size == 0`` that ``contains`` /
    ``within`` are built on -- with no denied operator node anywhere inside.
    """
    return delegation_violation(
        method, kernels=_ALLOWED_KERNELS, wrappers={"Moc", "cls"},
        coercers={"_words"})


def test_every_public_method_is_a_single_kernel_delegation():
    """The falsifiable form of the thin-view commitment (issue #196).

    A public ``Moc`` method must be exactly one ``return`` of exactly one
    kernel call, optionally re-boxed by ``Moc(...)`` / ``cls(...)``, coercing
    its operand through ``_words(...)``, or compared as ``.size == 0``.  Any
    other body -- a comprehension, a slice, arithmetic, a branch on cell
    values, a second kernel call -- is the finding this test exists to catch;
    :func:`test_the_delegation_pin_rejects_violating_bodies` proves it does.
    """
    methods = [n for n in class_def(mortie.moc_object, "Moc").body
               if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]
    assert {m.name for m in methods} == {
        "contains", "difference", "from_polygon", "intersection",
        "intersects", "symmetric_difference", "to_order", "union", "within",
    }
    for method in methods:
        reason = _delegation_violation(method)
        assert reason is None, f"Moc.{method.name} {reason}"


@pytest.mark.parametrize("source", [
    # A comprehension has no Call node and slicing/arithmetic are operators, so
    # the old call-counting form of this pin passed all of these.
    "def m(self, other):\n"
    "    return Moc(moc_or(self.words, _words([w for w in other if w % 2])))\n",
    "def m(self, other):\n"
    "    return Moc(moc_or(self.words, _words(other))[::2] + 1)\n",
    "def m(self, other):\n"
    "    return Moc(moc_or(self.words, _words(other)) if other else self.words)\n",
    "def m(self, other):\n"
    "    return Moc(moc_and(moc_or(self.words, _words(other)), self.words))\n",
    "def m(self, other):\n"
    "    return Moc(np.unique(moc_or(self.words, _words(other))))\n",
    "def m(self, other):\n"
    "    return moc_minus(self.words, _words(other)).size == 1\n",
    "def m(self, other):\n"
    "    return moc_minus(self.words, _words(other)).size > 0\n",
    "def m(self, other):\n"
    "    out = self.words\n"
    "    for w in other:\n"
    "        out = moc_or(out, _words(w))\n"
    "    return Moc(out)\n",
    "def m(self, other):\n"
    "    return self.words\n",
])
def test_the_delegation_pin_rejects_violating_bodies(source):
    """The pin has to fail on algebra, or it pins nothing (issue #196)."""
    method = ast.parse(source).body[0]
    assert _delegation_violation(method) is not None


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
    """`moc(aoi).to_order(9)` is the cover zagg demo/07_minimal.ipynb builds today.

    The notebook's ``coverage()`` calls ``morton_coverage_moc(lats, lons,
    order=9)`` on the AOI ring and hands the result to the store; the object
    form must land on exactly the same order-9 cells.
    """
    reference = mortie.moc_to_order(
        mortie.morton_coverage_moc(SERC_RING[:, 1], SERC_RING[:, 0], order=9), 9
    )
    assert np.array_equal(moc(SERC_AOI).to_order(9), reference)
    # ... and on the flat coverer's answer for the same ring at that order.
    assert np.array_equal(
        moc(SERC_AOI).to_order(9),
        np.sort(mortie.morton_coverage(SERC_RING[:, 1], SERC_RING[:, 0], order=9)),
    )


# The public surface of `mortie/moc.py` as it stood at the rename (0.9.9,
# `git show 006fb21^:mortie/moc.py`): ten module-level functions plus the
# `moc_min = common_ancestor` alias.  Held here as an independent copy so that
# editing `_KERNEL_NAMES` fails this test rather than redefining the pin.
_RETIRED_SUBMODULE_SURFACE = {
    "common_ancestor", "compress_moc", "moc_and", "moc_intersects", "moc_min",
    "moc_minus", "moc_not", "moc_or", "moc_to_order", "moc_xor",
    "split_base_cells",
}


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

    def test_kernel_roster_is_the_frozen_pre_rename_surface(self):
        # The roster the shim resolves is a *historical* fact -- what
        # mortie/moc.py exported at the rename -- not a live property of the
        # kernel, so it is pinned against an independent copy of that surface
        # (captured from `git show 006fb21^:mortie/moc.py`).  A kernel function
        # added later must NOT be enrolled in the deprecated namespace, which
        # is exactly what an equality against the live module would force.
        assert set(_KERNEL_NAMES) == _RETIRED_SUBMODULE_SURFACE

    def test_shimmed_names_all_still_exist_on_the_kernel(self):
        # The direction that is actually true today: the shim can never dangle.
        from mortie import _moc
        public = {name for name in dir(_moc)
                  if not name.startswith("_") and callable(getattr(_moc, name))
                  and getattr(getattr(_moc, name), "__module__", "") ==
                  _moc.__name__}
        assert set(_KERNEL_NAMES) <= public
        assert set(_KERNEL_NAMES) <= set(mortie.__all__)

    def test_a_new_kernel_function_does_not_join_the_shim(self, monkeypatch):
        # Falsifiability of the pin above: growing the kernel must leave the
        # deprecated roster alone rather than forcing a new name into it.
        from mortie import _moc

        def brand_new_kernel(words):
            return words

        brand_new_kernel.__module__ = _moc.__name__
        monkeypatch.setattr(_moc, "brand_new_kernel", brand_new_kernel,
                            raising=False)
        self.test_kernel_roster_is_the_frozen_pre_rename_surface()
        self.test_shimmed_names_all_still_exist_on_the_kernel()
        with pytest.raises(AttributeError, match="not the old"):
            moc.brand_new_kernel

    @pytest.mark.parametrize("name", _KERNEL_NAMES)
    def test_deprecated_attribute_still_resolves_to_the_kernel(self, name):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assert getattr(moc, name) is getattr(mortie, name)

    def test_warning_fires_on_every_access(self):
        # Dedup is the warnings module's job (filters are the user's contract),
        # not shim state: under `always` every access is visible.
        from mortie.moc_object import _MocNamespace

        shim = _MocNamespace()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            shim.moc_and, shim.moc_and
            assert len(caught) == 2
            assert issubclass(caught[0].category, DeprecationWarning)
            assert "mortie.moc.moc_and is deprecated" in str(caught[0].message)

    def test_a_later_consumer_still_observes_the_warning(self):
        # No process-wide budget: the first block *delivers* the warning
        # normally -- exactly what a per-name budget would spend -- and a
        # second consumer's test suite, later in the same interpreter and on
        # the same singleton, must still record it.
        from mortie.moc_object import _MocNamespace

        shim = _MocNamespace()
        with warnings.catch_warnings(record=True) as first:
            warnings.simplefilter("always")
            shim.moc_and
            assert len(first) == 1
        with warnings.catch_warnings(record=True) as second:
            warnings.simplefilter("always")
            shim.moc_and
            assert len(second) == 1
            assert "mortie.moc.moc_and is deprecated" in str(second[0].message)

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError, match="not the old"):
            moc.morton_coverage_moc

    def test_dir_lists_the_deprecated_names(self):
        assert dir(moc) == sorted(_KERNEL_NAMES)

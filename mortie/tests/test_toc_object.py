"""Tests for the time-first ``Toc`` object (issue #198).

The object is a thin view over the canonical toc word set: the commitment is
that every public method is a *single* delegation to a kernel function, and
the tests here are written to falsify that rather than to restate it -- see
:class:`TestDelegationParity` (value parity, method by method) and
:func:`test_every_public_method_is_a_single_kernel_delegation` (the structural
pin over the source, on the machinery shared with the ``Moc`` pin in
``mortie/tests/delegation.py``).
"""

import ast
import copy
import inspect
import pickle

import numpy as np
import pytest

import mortie
from mortie import Toc, toc_object
from mortie.tests.delegation import class_def, delegation_violation


def u64(values):
    """Shorthand: a uint64 array from Python ints."""
    return np.asarray(values, dtype=np.uint64)


# Golden words for the ISO constructor forms -- the determinism pin ("same
# input + same version -> same words", byte for byte, on top of the kernel
# goldens in test_toc_setops.py).
GOLDEN_PAIR_WORD = 10729324834951646742      # Toc("2020-01-01", "2021-06-01")
GOLDEN_INSTANT_WORD = 10729324836993577984   # Toc("2020-01-01")

MARCH = ("2020-03-01", "2020-03-15")
LATE_MARCH = ("2020-03-10", "2020-04-01")
APRIL = ("2020-04-05", "2020-04-20")
FREE_INSTANT = "2020-07-04"


@pytest.fixture(scope="module")
def pair():
    """Two overlapping covers, for the set-op and predicate parity tests."""
    return Toc(*MARCH), Toc(*LATE_MARCH)


class TestConstructorForms:
    """Every documented constructor source, and what each resolves to."""

    def test_iso_pair_matches_span2toc(self):
        expected = mortie.span2toc(mortie.from_datetime64(MARCH[0]),
                                   mortie.from_datetime64(MARCH[1]))
        assert np.array_equal(Toc(*MARCH).words, u64([expected]))

    def test_iso_instant_matches_time2toc(self):
        expected = mortie.time2toc(mortie.from_datetime64(FREE_INSTANT))
        assert np.array_equal(Toc(FREE_INSTANT).words, u64([expected]))

    def test_datetime64_and_string_agree(self):
        assert Toc(np.datetime64("2020-07-04")) == Toc(FREE_INSTANT)
        assert Toc(np.datetime64(MARCH[0]), np.datetime64(MARCH[1])) == Toc(*MARCH)

    def test_instant_array(self):
        whens = np.asarray(["2020-01-01", "2020-06-01"], dtype="datetime64[ns]")
        expected = np.sort(mortie.time2toc(mortie.from_datetime64(whens)))
        assert np.array_equal(Toc(whens).words, expected)

    def test_pair_arrays_broadcast(self):
        starts = np.asarray(["2020-01-01", "2021-01-01"], dtype="datetime64[ns]")
        ends = np.asarray(["2020-02-01", "2021-02-01"], dtype="datetime64[ns]")
        expected = np.sort(mortie.span2toc(mortie.from_datetime64(starts),
                                           mortie.from_datetime64(ends)))
        assert np.array_equal(Toc(starts, ends).words, expected)
        # ... including a scalar start against an array of ends.
        fanned = Toc(starts[0], ends)
        assert fanned.words.size == 1  # overlapping spans coalesce

    def test_overlapping_spans_coalesce(self, pair):
        a, b = pair
        both = Toc(np.append(a.words, b.words))
        assert both.words.size == 1
        assert int(both.words[0]) == mortie.toc_merge(int(a.words[0]),
                                                      int(b.words[0]))

    def test_words_array_round_trip(self, pair):
        a, _ = pair
        assert Toc(a.words) == a

    def test_words_are_normalized_eagerly(self, pair):
        a, _ = pair
        subsumed = mortie.time2toc(mortie.from_datetime64("2020-03-05"))
        raw = u64([int(a.words[0]), subsumed, subsumed, int(a.words[0])])
        cover = Toc(raw)
        assert np.array_equal(cover.words, mortie.toc_normalize(raw))
        assert cover == a  # the subsumed instant absorbed, duplicates gone

    def test_free_instant_survives_bit_identical(self, pair):
        a, _ = pair
        free = mortie.time2toc(mortie.from_datetime64(FREE_INSTANT))
        cover = Toc(np.append(a.words, u64([free])))
        assert free in cover.words

    def test_protocol_object_round_trip(self, pair):
        a, _ = pair

        class Carrier:
            def __toc_words__(self):
                return a.words

        assert Toc(Carrier()) == a
        assert Toc(a) == a

    def test_scalar_int_is_a_word(self, pair):
        a, _ = pair
        assert Toc(int(a.words[0])) == a

    def test_empty_words(self):
        empty = Toc(u64([]))
        assert empty.words.size == 0
        assert empty == Toc(u64([]))

    @pytest.mark.parametrize("kind", ["words", "cover"])
    def test_end_is_refused_for_a_words_source(self, pair, kind):
        # Silently ignoring end= would let Toc(a.words, "2021-01-01") read as
        # "extend this cover to 2021" while doing nothing.
        a, _ = pair
        source = a.words if kind == "words" else a
        with pytest.raises(ValueError, match="already"):
            Toc(source, "2021-01-01")

    @pytest.mark.parametrize("source, end", [
        (1.5, None),                       # a float is neither words nor time
        ("2020-01-01", 123),               # numeric end: wrong-epoch trap
        (np.asarray([1.5, 2.5]), None),
    ])
    def test_numeric_times_are_refused(self, source, end):
        # np.asarray(123, "datetime64[ns]") would silently read ns since 1970
        # -- the wrong epoch -- so numeric endpoints are refused, not guessed.
        with pytest.raises(ValueError, match="datetime64 / ISO-string"):
            Toc(source, end)

    def test_inverted_span_is_the_kernel_error(self):
        with pytest.raises(ValueError, match="after its end"):
            Toc(MARCH[1], MARCH[0])

    def test_negative_words_are_the_kernel_error(self):
        with pytest.raises(ValueError, match="non-negative"):
            Toc(np.asarray([-1, 2]))

    def test_no_coverage_knobs(self):
        # The constructor matrix is (source, end) and nothing else.
        params = list(inspect.signature(Toc.__init__).parameters)
        assert params == ["self", "source", "end"]


class TestNormalizationAndIdentity:
    """Eager canonicalization, immutability, equality and hashing."""

    def test_words_are_canonical_eagerly(self, pair):
        a, _ = pair
        cover = Toc(np.append(a.words, Toc(*APRIL).words))
        assert np.array_equal(cover.words, mortie.toc_normalize(cover.words))

    def test_words_are_read_only(self, pair):
        a, _ = pair
        assert not a.words.flags.writeable
        with pytest.raises(ValueError):
            a.words[0] = 0

    def test_instance_is_immutable(self, pair):
        a, _ = pair
        with pytest.raises(AttributeError, match="immutable"):
            a.words = u64([1])
        with pytest.raises(AttributeError, match="immutable"):
            del a.words

    def test_pickle_round_trip(self, pair):
        # Workers marshal their arguments: a cover must cross a process
        # boundary as easily as the word array it wraps.
        a, _ = pair
        restored = pickle.loads(pickle.dumps(a))
        assert restored == a
        assert not restored.words.flags.writeable

    @pytest.mark.parametrize("clone", [copy.copy, copy.deepcopy])
    def test_copy_round_trip(self, pair, clone):
        a, _ = pair
        restored = clone(a)
        assert restored == a
        assert not restored.words.flags.writeable

    def test_slots_only(self, pair):
        a, _ = pair
        assert Toc.__slots__ == ("words",)
        assert not hasattr(a, "__dict__")

    def test_equality_is_word_identity(self, pair):
        a, b = pair
        assert a == Toc(*MARCH)
        assert a != b
        # An instant and its degenerate span encode different words: the
        # canonical form compares, not the timeline it conservatively covers.
        assert Toc(FREE_INSTANT) != Toc(FREE_INSTANT, FREE_INSTANT)

    def test_equality_against_a_non_toc_is_not_an_error(self, pair):
        a, _ = pair
        assert a != 5
        assert a.__eq__(5) is NotImplemented

    def test_hash_matches_equality(self, pair):
        a, b = pair
        assert hash(a) == hash(Toc(*MARCH))
        assert len({a, Toc(*MARCH), b}) == 2


class TestDelegationParity:
    """Every method equals the kernel call on ``.words`` -- the thin-view pin."""

    def test_overlaps(self, pair):
        a, b = pair
        assert a.overlaps(b)
        assert a.overlaps(b) == (mortie.toc_and(a.words, b.words).size > 0)
        assert not a.overlaps(Toc(*APRIL))
        assert not a.overlaps(Toc(FREE_INSTANT))

    def test_overlaps_conservative_direction_at_the_quantum(self):
        # The documented direction, both ways: a real gap smaller than the
        # encoding quantum can over-report True (the envelopes graze), while
        # a gap wider than a quantum survives encoding and is a decisive
        # False -- outward rounding only shrinks apparent gaps.
        after = Toc("2020-03-15", "2020-04-01")
        assert Toc(MARCH[0], "2020-03-14T23:59:59.999").overlaps(after)
        assert not Toc(MARCH[0], "2020-03-14T23:59:50").overlaps(after)

    def test_contains(self, pair):
        a, b = pair
        inside = Toc("2020-03-05", "2020-03-06")
        assert a.contains(inside)
        assert not a.contains(b)          # b spills past a's end
        assert a.contains(a)
        assert a.contains(Toc("2020-03-05"))  # a subsumed instant
        assert not a.contains(Toc(FREE_INSTANT))

    def test_contains_is_the_kernel_identity(self, pair):
        a, b = pair
        for other in (b, Toc("2020-03-05", "2020-03-06"), Toc(*APRIL)):
            expected = np.array_equal(
                mortie.toc_and(a.words, other.words), other.words)
            assert a.contains(other) == expected

    def test_contains_normalizes_a_raw_operand(self, pair):
        # A raw duplicated, unsorted operand must compare canonical-to-
        # canonical, or containment would falsely fail on word inequality.
        a, _ = pair
        inside = Toc("2020-03-05", "2020-03-06")
        raw = u64([int(inside.words[0]), int(inside.words[0])])
        assert a.contains(raw)

    def test_contains_normalizes_a_protocol_operand(self, pair):
        # Nothing obliges a third-party __toc_words__() to be canonical, so
        # the protocol door must canonicalize like the raw-array door does --
        # otherwise the same word set answers differently by which door it
        # came through.
        a, _ = pair
        inside = Toc("2020-03-05", "2020-03-06")
        raw = u64([int(inside.words[0]), int(inside.words[0])])

        class Carrier:
            def __toc_words__(self):
                return raw

        assert a.contains(Carrier()) == a.contains(raw) is True
        # ... and unsorted, which the sweep would also reject uncanonicalized.
        two = Toc(np.append(inside.words, Toc("2020-03-10", "2020-03-11").words))
        jumbled = two.words[::-1]
        assert not np.array_equal(jumbled, mortie.toc_normalize(jumbled))

        class Unsorted:
            def __toc_words__(self):
                return jumbled

        assert a.contains(Unsorted()) == a.contains(jumbled) is True

    def test_set_methods_take_raw_words_and_protocol_objects(self, pair):
        a, b = pair
        assert a.overlaps(b.words) == a.overlaps(b)
        assert a.contains(b.words) == a.contains(b)
        assert a.intersection(b.words) == a.intersection(b)

    def test_intersection(self, pair):
        a, b = pair
        result = a.intersection(b)
        assert isinstance(result, Toc)
        assert np.array_equal(result.words, mortie.toc_and(a.words, b.words))
        # The dunder is the same bound method, not a re-implementation.
        assert Toc.__and__ is Toc.intersection
        assert (a & b) == result

    def test_intersection_keeps_a_shared_instant_bit_identical(self, pair):
        a, _ = pair
        instant = Toc("2020-03-05")
        assert np.array_equal(a.intersection(instant).words, instant.words)

    def test_empty_operand_predicates(self, pair):
        # An empty cover is where the predicates stop agreeing: contains is
        # vacuously True while overlaps is False.  Chosen, not inherited --
        # see the module docstring's conservative-direction table.
        a, _ = pair
        empty = Toc(u64([]))
        assert a.contains(empty)
        assert not a.overlaps(empty)
        assert not empty.contains(a)
        assert not empty.overlaps(empty)
        assert empty.contains(empty)

    def test_empty_operand_intersection(self, pair):
        a, _ = pair
        empty = Toc(u64([]))
        assert a.intersection(empty) == empty
        assert len(a & a) == len(a)

    def test_union_is_construction(self, pair):
        # No union method by design (#177 audit): concatenate + construct is
        # the union, because construction normalizes.
        a, b = pair
        both = Toc(np.append(a.words, b.words))
        assert both.contains(a) and both.contains(b)
        assert not hasattr(Toc, "union")


# The kernel functions a Toc method is allowed to call: the #177 call-site
# audit ruled exactly one set operation in, so the roster is one name and all
# three public methods delegate to it.  Wrapper and coercer roles as in the
# Moc pin; the blessed comparisons differ (`.size > 0` for overlaps,
# `np.array_equal(<kernel>, _words(...))` for contains -- there is no
# toc_minus to build an emptiness-shaped containment on).
_ALLOWED_KERNELS = {"toc_and"}


def _toc_violation(method):
    """The Toc-specific instantiation of the shared delegation pin."""
    return delegation_violation(
        method, kernels=_ALLOWED_KERNELS, wrappers={"Toc", "cls"},
        coercers={"_words"}, size_ops=(ast.Gt,), array_equal=True)


def test_every_public_method_is_a_single_kernel_delegation():
    """The falsifiable form of the thin-view commitment (issues #196 / #198).

    A public ``Toc`` method must be exactly one ``return`` of exactly one
    kernel call, optionally re-boxed by ``Toc(...)``, coercing its operand
    through ``_words(...)``, compared as ``.size > 0``, or equality-tested
    against the coerced operand via ``np.array_equal``.  Any other body is
    the finding this test exists to catch;
    :func:`test_the_delegation_pin_rejects_violating_bodies` proves it does.
    """
    methods = [n for n in class_def(toc_object, "Toc").body
               if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]
    assert {m.name for m in methods} == {"contains", "intersection", "overlaps"}
    for method in methods:
        reason = _toc_violation(method)
        assert reason is None, f"Toc.{method.name} {reason}"


@pytest.mark.parametrize("source", [
    # Blessed-shape near-misses: the wrong comparison operator, the wrong
    # constant, an equality against something other than the coerced operand.
    "def m(self, other):\n"
    "    return toc_and(self.words, _words(other)).size == 0\n",
    "def m(self, other):\n"
    "    return toc_and(self.words, _words(other)).size > 1\n",
    "def m(self, other):\n"
    "    return toc_and(self.words, _words(other)).size >= 0\n",
    "def m(self, other):\n"
    "    return np.array_equal(toc_and(self.words, _words(other)), self.words)\n",
    "def m(self, other):\n"
    "    return np.array_equal(_words(other), _words(other))\n",
    "def m(self, other):\n"
    "    return np.array_equal(toc_and(self.words, _words(other)),\n"
    "                          _words(other), equal_nan=True)\n",
    # A kernel outside the audited roster, algebra, and multi-call bodies.
    "def m(self, other):\n"
    "    return Toc(toc_normalize(np.append(self.words, _words(other))))\n",
    "def m(self, other):\n"
    "    return Toc(toc_and(toc_and(self.words, _words(other)), self.words))\n",
    "def m(self, other):\n"
    "    return Toc(np.unique(toc_and(self.words, _words(other))))\n",
    "def m(self, other):\n"
    "    return Toc(toc_and(self.words, _words(other))[:-1])\n",
    "def m(self, other):\n"
    "    return Toc(toc_and(self.words, _words(other)) if other else self.words)\n",
    "def m(self, other):\n"
    "    out = self.words\n"
    "    for w in other:\n"
    "        out = toc_and(out, _words(w))\n"
    "    return Toc(out)\n",
    "def m(self, other):\n"
    "    return self.words\n",
])
def test_the_delegation_pin_rejects_violating_bodies(source):
    """The pin has to fail on algebra, or it pins nothing (issue #198)."""
    method = ast.parse(source).body[0]
    assert _toc_violation(method) is not None


class TestReprAndProtocol:
    """What the object tells you about itself, and hands to others."""

    def test_repr_of_a_range_shows_the_conservative_envelope(self):
        # Both bounds round outward -- the start floors (2020-02-29T23:59:58),
        # the end ceils past its 2^32 ns grid point (...T00:00:00.979553280 ->
        # ...T00:00:01): the repr shows what the cover *covers*, not what it
        # was built from, so neither printed bound may land inside it.
        cover = Toc(*MARCH)
        assert repr(cover) == (
            "Toc(1 range, 2020-02-29T23:59:58 to 2020-03-15T00:00:01, "
            "14 days covered)")
        starts, ends = mortie.toc2time(cover.words)
        printed_start, printed_end = repr(cover).split(", ")[1].split(" to ")
        assert np.datetime64(printed_start) <= mortie.to_datetime64(
            int(starts.min()))
        assert np.datetime64(printed_end) >= mortie.to_datetime64(
            int(ends.max()))

    def test_repr_of_an_instant(self):
        assert repr(Toc(FREE_INSTANT)) == (
            "Toc(1 instant, 2020-07-04T00:00:00 to 2020-07-04T00:00:00, "
            "0 ns covered)")

    def test_repr_of_a_mixed_cover_counts_both_kinds(self, pair):
        a, _ = pair
        mixed = Toc(np.append(a.words, Toc(FREE_INSTANT).words))
        text = repr(mixed)
        assert text.startswith("Toc(1 range + 1 instant, ")
        assert text.endswith("14 days covered)")

    def test_repr_of_an_empty_cover(self):
        assert repr(Toc(u64([]))) == "Toc(0 spans)"

    def test_len_and_iter(self, pair):
        a, _ = pair
        mixed = Toc(np.append(a.words, Toc(FREE_INSTANT).words))
        assert len(mixed) == mixed.words.size == 2
        assert np.array_equal(np.fromiter(mixed, dtype=np.uint64), mixed.words)

    def test_protocol_hands_back_the_canonical_words(self, pair):
        a, _ = pair
        handed = a.__toc_words__()
        assert handed is a.words
        assert not handed.flags.writeable


class TestDeterminism:
    """Same input + same version -> same words, byte for byte."""

    def test_repeated_construction_is_byte_identical(self):
        first, second = Toc(*MARCH), Toc(*MARCH)
        assert first.words.tobytes() == second.words.tobytes()

    def test_golden_words(self):
        assert np.array_equal(Toc("2020-01-01", "2021-06-01").words,
                              u64([GOLDEN_PAIR_WORD]))
        assert np.array_equal(Toc("2020-01-01").words,
                              u64([GOLDEN_INSTANT_WORD]))

    def test_word_order_does_not_matter(self, pair):
        a, _ = pair
        free = Toc(FREE_INSTANT)
        assert (Toc(np.append(a.words, free.words))
                == Toc(np.append(free.words, a.words)))

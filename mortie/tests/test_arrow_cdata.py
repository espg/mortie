"""
Tests for the library-agnostic Arrow C Data Interface surface (issue #93).

Where ``test_arrow.py`` pins the pyarrow ``ExtensionType`` skin (issue #86),
these pin the raw **PyCapsule C Data Interface** built in Rust (the ``arrow``
crate): ``MortonIndexArray.__arrow_c_array__`` / ``__arrow_c_schema__`` +
``from_arrow``, and the numpy-only ``mortie.arrow.export_c_array`` /
``import_c_array`` helpers. The point is that this path carries a typed
``morton_index`` column -- extension metadata + null/sentinel -- through **any**
Arrow lib, including arro3-core with pyarrow absent (zagg's Lambda carrier).

The pyarrow leg runs in CI (pyarrow is a test extra). The arro3-core leg is
gated on arro3-core being importable; it is **not** in the CI test extra (see
issue #93 / the docs note), so install ``arro3-core==0.8.1`` locally to exercise
it: ``pip install arro3-core==0.8.1 && pytest mortie/tests/test_arrow_cdata.py``.
"""

import numpy as np
import pytest

import mortie  # noqa: F401  (registers the morton_index pyarrow extension type)
from mortie import _rustie
from mortie import arrow as marrow

EXT_NAME = "mortie.morton_index"


def _sample_words(n=8, order=12):
    """A spread of packed words across base cells / both hemispheres."""
    bases = np.arange(n, dtype=np.uint64) % 12
    nested = bases * (1 << (2 * order)) + np.arange(n, dtype=np.uint64)
    return _rustie.rust_mi_from_nested(
        np.ascontiguousarray(nested), order
    ).astype(np.uint64)


try:
    SENTINEL = int(mortie.MortonIndexArray._SENTINEL)
except ImportError:  # pragma: no cover - pandas absent
    SENTINEL = 0


def _words_with_gap():
    """Sample words with the empty sentinel planted at two positions."""
    words = _sample_words().copy()
    words[2] = SENTINEL
    words[5] = SENTINEL
    return words


class _CapsuleSource:
    """A minimal object exposing the raw producer capsules for consumers."""

    def __init__(self, words):
        self._words = np.ascontiguousarray(words, dtype=np.uint64)

    def __arrow_c_schema__(self):
        return marrow.export_c_schema()

    def __arrow_c_array__(self, requested_schema=None):
        return marrow.export_c_array(self._words)


# ---------------------------------------------------------------------------
# numpy-only surface (no pyarrow, no pandas): export -> import round-trip
# ---------------------------------------------------------------------------


class TestNumpyOnlyRoundTrip:
    def test_export_import_preserves_words(self):
        words = _sample_words()
        back = marrow.import_c_array(_CapsuleSource(words))
        assert back.dtype == np.uint64
        np.testing.assert_array_equal(back, words)

    def test_export_import_preserves_nulls(self):
        # sentinel -> Arrow null -> sentinel, byte-for-byte over the validity bitmap
        words = _words_with_gap()
        back = marrow.import_c_array(_CapsuleSource(words))
        np.testing.assert_array_equal(back, words)

    def test_import_accepts_capsule_tuple(self):
        words = _sample_words()
        pair = marrow.export_c_array(words)
        np.testing.assert_array_equal(marrow.import_c_array(pair), words)

    def test_export_accepts_morton_index_array(self):
        pytest.importorskip("pandas")
        mia = mortie.MortonIndexArray(_sample_words())
        back = marrow.import_c_array(marrow.export_c_array(mia))
        np.testing.assert_array_equal(back, mia._data)

    def test_empty_array_round_trips(self):
        words = np.empty(0, dtype=np.uint64)
        back = marrow.import_c_array(_CapsuleSource(words))
        assert len(back) == 0

    def test_all_null_round_trips_to_sentinel(self):
        words = np.full(4, SENTINEL, dtype=np.uint64)
        back = marrow.import_c_array(_CapsuleSource(words))
        np.testing.assert_array_equal(back, words)


# ---------------------------------------------------------------------------
# MortonIndexArray dunder surface (pandas path)
# ---------------------------------------------------------------------------


class TestMortonIndexArrayDunders:
    def test_dunder_round_trip(self):
        pytest.importorskip("pandas")
        mia = mortie.MortonIndexArray(_sample_words())
        back = mortie.MortonIndexArray.from_arrow(mia)
        assert isinstance(back, mortie.MortonIndexArray)
        np.testing.assert_array_equal(back._data, mia._data)

    def test_dunder_round_trip_with_nulls(self):
        pytest.importorskip("pandas")
        mia = mortie.MortonIndexArray(_words_with_gap())
        back = mortie.MortonIndexArray.from_arrow(mia)
        np.testing.assert_array_equal(back._data, mia._data)
        np.testing.assert_array_equal(back.isna(), mia._data == SENTINEL)

    def test_schema_capsule_is_named(self):
        pytest.importorskip("pandas")
        mia = mortie.MortonIndexArray(_sample_words())
        cap = mia.__arrow_c_schema__()
        # A PyCapsule with the C Data Interface schema name.
        assert "arrow_schema" in repr(cap)


# ---------------------------------------------------------------------------
# pyarrow consumer/producer (CI: pyarrow is a test extra)
# ---------------------------------------------------------------------------


class TestPyarrowInterop:
    def test_pyarrow_consumes_our_export(self):
        pa = pytest.importorskip("pyarrow")
        words = _words_with_gap()
        arr = pa.array(_CapsuleSource(words))
        # pyarrow resolves the registered morton_index extension type.
        assert arr.type.extension_name == EXT_NAME
        assert arr.null_count == 2
        assert arr.storage.to_pylist() == [
            (None if w == SENTINEL else int(w)) for w in words
        ]

    def test_we_import_pyarrow_export(self):
        pa = pytest.importorskip("pyarrow")
        words = _words_with_gap()
        arr = pa.array(_CapsuleSource(words))
        back = marrow.import_c_array(arr)
        np.testing.assert_array_equal(back, words)

    def test_import_plain_uint64_pyarrow_array(self):
        # A plain (non-extension) uint64 arrow array still imports as words.
        pa = pytest.importorskip("pyarrow")
        words = _sample_words()
        arr = pa.array(words, type=pa.uint64())
        np.testing.assert_array_equal(marrow.import_c_array(arr), words)

    def test_reject_non_uint64_storage(self):
        pa = pytest.importorskip("pyarrow")
        arr = pa.array([1, 2, 3], type=pa.int32())
        with pytest.raises(ValueError, match="uint64"):
            marrow.import_c_array(arr)

    def test_import_respects_slice_offset(self):
        # A sliced arrow array carries a non-zero logical offset over the C-Data
        # boundary; the imported words must be the sliced range, not the buffer.
        pa = pytest.importorskip("pyarrow")
        words = _words_with_gap()  # nulls at 2 and 5
        arr = pa.array(_CapsuleSource(words)).slice(2, 4)  # spans a null at idx 2
        back = marrow.import_c_array(arr)
        np.testing.assert_array_equal(back, words[2:6])

    def test_reject_misordered_capsules(self):
        # Passing (array, schema) instead of (schema, array) must be rejected on
        # the capsule name, not reinterpreted (would be memory corruption).
        words = _sample_words()
        schema_capsule, array_capsule = marrow.export_c_array(words)
        with pytest.raises(ValueError, match="arrow_schema"):
            _rustie.rust_mi_import_c_array(array_capsule, schema_capsule)


# ---------------------------------------------------------------------------
# arro3-core consumer/producer -- pyarrow-free carrier (issue #93 item 4).
# Local-only: arro3-core is not in the CI test extra, so gate *only this class*
# on it (a module-level importorskip would skip the whole file -- incl. the
# pyarrow/numpy legs that DO run in CI -- when arro3-core is absent).
# ---------------------------------------------------------------------------

try:
    import arro3.core as a3

    HAS_ARRO3 = True
except ImportError:  # pragma: no cover - arro3-core absent (CI)
    a3 = None
    HAS_ARRO3 = False


@pytest.mark.skipif(not HAS_ARRO3, reason="arro3-core not installed (local-only leg)")
class TestArro3Interop:
    """The pyarrow-free path #86 never exercised: round-trip on arro3-core."""

    def test_arro3_consumes_our_export_with_extension_metadata(self):
        # arro3-core pulls our export and preserves the extension metadata
        # through the C-Data boundary (the issue-2 gate; arro3-core==0.8.1).
        words = _sample_words()
        arr = a3.Array.from_arrow(_CapsuleSource(words))
        meta = dict(arr.field.metadata_str)
        assert meta.get("ARROW:extension:name") == EXT_NAME
        np.testing.assert_array_equal(np.asarray(arr), words)

    def test_arro3_export_import_round_trip_with_nulls(self):
        # Full loop with pyarrow absent from the path: our export -> arro3 ->
        # arro3's export -> our import, nulls preserved byte-for-byte.
        words = _words_with_gap()
        arr = a3.Array.from_arrow(_CapsuleSource(words))
        back = marrow.import_c_array(arr)
        np.testing.assert_array_equal(back, words)

    def test_arro3_only_no_pyarrow_needed(self):
        # The whole round-trip touches no pyarrow symbol.
        words = _sample_words()
        arr = a3.Array.from_arrow(_CapsuleSource(words))
        back = mortie.MortonIndexArray.from_arrow(arr) if _has_pandas() else None
        raw = marrow.import_c_array(arr)
        np.testing.assert_array_equal(raw, words)
        if back is not None:
            np.testing.assert_array_equal(back._data, words)


def _has_pandas():
    try:
        import pandas  # noqa: F401

        return True
    except ImportError:
        return False

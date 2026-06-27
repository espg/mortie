"""
Tests for the ``morton_index`` pyarrow ExtensionType (issue #35, phase 4).

These pin the Arrow-interop skin in ``mortie/arrow.py``: round-tripping the
pandas ``MortonIndexArray`` <-> the pyarrow ExtensionType <-> parquet, with the
``morton_index`` extension identity surviving serialization, plus the optional-
extra contract (numpy-only / pandas-only import still works; a clear error is
raised when pyarrow is genuinely absent).
"""

import importlib
import sys

import numpy as np
import pytest

import mortie  # noqa: F401  (registers the morton_index pyarrow extension type)
from mortie import _rustie

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

from mortie import arrow as marrow  # noqa: E402

EXT_NAME = "mortie.morton_index"


def _sample_words(n=8, order=12):
    """A spread of packed words across base cells / both hemispheres."""
    bases = np.arange(n, dtype=np.uint64) % 12
    nested = bases * (1 << (2 * order)) + np.arange(n, dtype=np.uint64)
    return _rustie.rust_mi_from_nested(
        np.ascontiguousarray(nested), order
    ).astype(np.uint64)


# ---------------------------------------------------------------------------
# Extension-type surface
# ---------------------------------------------------------------------------


class TestExtensionType:
    def test_type_is_uint64_extension(self):
        t = marrow.morton_index_type()
        assert isinstance(t, pa.ExtensionType)
        assert t.storage_type == pa.uint64()
        assert t.extension_name == EXT_NAME

    def test_serialize_is_empty(self):
        t = marrow.morton_index_type()
        assert t.__arrow_ext_serialize__() == b""

    def test_registered_name_resolves(self):
        # Building the type registers it; an array of that type reports it back.
        words = _sample_words()
        arr = marrow.from_morton_index(words)
        assert arr.type.extension_name == EXT_NAME


# ---------------------------------------------------------------------------
# Conversion round-trips
# ---------------------------------------------------------------------------


class TestConversion:
    def test_from_raw_words_preserves_bits(self):
        words = _sample_words()
        arr = marrow.from_morton_index(words)
        np.testing.assert_array_equal(arr.to_numpy(), words)

    def test_extension_array_to_pandas(self):
        words = _sample_words()
        mia = marrow.to_morton_index(marrow.from_morton_index(words))
        np.testing.assert_array_equal(np.asarray(mia._data), words)
        assert mia.dtype.name == "morton_index"

    def test_pandas_extension_array_round_trip(self):
        pd = pytest.importorskip("pandas")  # noqa: F841
        mia = mortie.MortonIndexArray(_sample_words())
        arr = marrow.from_morton_index(mia)
        back = marrow.to_morton_index(arr)
        np.testing.assert_array_equal(back._data, mia._data)

    def test_to_pandas_dtype_matches(self):
        t = marrow.morton_index_type()
        assert t.to_pandas_dtype().name == "morton_index"


# ---------------------------------------------------------------------------
# Parquet serialization (metadata survives the round-trip)
# ---------------------------------------------------------------------------


class TestParquet:
    def test_parquet_preserves_extension_type(self, tmp_path):
        words = _sample_words()
        table = pa.table({"mi": marrow.from_morton_index(words)})
        assert table.schema.field("mi").type.extension_name == EXT_NAME

        path = tmp_path / "mi.parquet"
        pq.write_table(table, path)
        read = pq.read_table(path)

        col_type = read.schema.field("mi").type
        assert isinstance(col_type, pa.ExtensionType)
        assert col_type.extension_name == EXT_NAME
        np.testing.assert_array_equal(read.column("mi").chunk(0).to_numpy(), words)

    def test_parquet_words_round_trip_to_pandas(self, tmp_path):
        words = _sample_words()
        path = tmp_path / "mi.parquet"
        pq.write_table(pa.table({"mi": marrow.from_morton_index(words)}), path)
        col = pq.read_table(path).column("mi").chunk(0)
        mia = marrow.to_morton_index(col)
        np.testing.assert_array_equal(mia._data, words)


# ---------------------------------------------------------------------------
# __from_arrow__ hook: table.to_pandas() yields a MortonIndexArray (issue #79)
# ---------------------------------------------------------------------------


class TestFromArrowHook:
    def test_to_pandas_yields_morton_index_series(self):
        pytest.importorskip("pandas")
        words = _sample_words()
        table = pa.table({"mi": marrow.from_morton_index(words)})
        series = table.to_pandas()["mi"]
        # Not a plain int64 column: the extension dtype survives.
        assert series.dtype.name == "morton_index"
        assert isinstance(series.array, mortie.MortonIndexArray)
        np.testing.assert_array_equal(series.array._data, words)

    def test_from_arrow_handles_chunked_array(self):
        pytest.importorskip("pandas")
        words = _sample_words(n=8)
        ext = marrow.from_morton_index(words)
        chunked = pa.chunked_array([ext[:3], ext[3:]])
        mia = mortie.MortonIndexDtype().__from_arrow__(chunked)
        assert isinstance(mia, mortie.MortonIndexArray)
        np.testing.assert_array_equal(mia._data, words)

    def test_from_arrow_handles_plain_array(self):
        pytest.importorskip("pandas")
        words = _sample_words()
        mia = mortie.MortonIndexDtype().__from_arrow__(marrow.from_morton_index(words))
        assert isinstance(mia, mortie.MortonIndexArray)
        np.testing.assert_array_equal(mia._data, words)

    def test_from_arrow_handles_empty_chunked_array(self):
        pytest.importorskip("pandas")
        empty = pa.chunked_array([], type=marrow.morton_index_type())
        mia = mortie.MortonIndexDtype().__from_arrow__(empty)
        assert isinstance(mia, mortie.MortonIndexArray)
        assert len(mia) == 0

    def test_parquet_to_pandas_full_round_trip(self, tmp_path):
        pytest.importorskip("pandas")
        words = _sample_words()
        path = tmp_path / "mi.parquet"
        pq.write_table(pa.table({"mi": marrow.from_morton_index(words)}), path)
        series = pq.read_table(path).to_pandas()["mi"]
        assert series.dtype.name == "morton_index"
        assert isinstance(series.array, mortie.MortonIndexArray)
        np.testing.assert_array_equal(series.array._data, words)


# ---------------------------------------------------------------------------
# Optional-extra contract
# ---------------------------------------------------------------------------


class TestOptionalExtra:
    def test_arrow_module_imports_numpy_only(self):
        # mortie.arrow must import even with neither pandas nor pyarrow; here we
        # only assert the module object and its public helpers exist.
        assert hasattr(marrow, "from_morton_index")
        assert hasattr(marrow, "to_morton_index")
        assert hasattr(marrow, "morton_index_type")

    def test_clear_error_without_pyarrow(self, monkeypatch):
        # Simulate pyarrow being absent: re-import mortie.arrow with the real
        # import machinery blocking pyarrow, then assert the helper raises a
        # clear error.
        import builtins

        real_import = builtins.__import__

        def _blocked(name, *args, **kwargs):
            if name == "pyarrow" or name.startswith("pyarrow."):
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "mortie.arrow", raising=False)
        for mod in list(sys.modules):
            if mod == "pyarrow" or mod.startswith("pyarrow."):
                monkeypatch.delitem(sys.modules, mod, raising=False)
        monkeypatch.setattr(builtins, "__import__", _blocked)

        fresh = importlib.import_module("mortie.arrow")
        with pytest.raises(ImportError, match="requires pyarrow"):
            fresh.morton_index_type()

        # ``monkeypatch.undo()`` restores both the real ``__import__`` and the
        # ORIGINAL ``mortie.arrow`` module object that ``delitem`` saved above
        # (the one already consistent with pyarrow's process-global extension
        # registry). We deliberately do NOT reload the module: a reload would
        # build a fresh ``MortonIndexType`` class that can't re-register (the
        # name is already taken), desyncing the module from the registry. The
        # transient pyarrow-less ``fresh`` copy is simply discarded by undo.
        monkeypatch.undo()

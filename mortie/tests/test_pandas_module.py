"""
Tests for ``mortie.pandas``, the pandas ExtensionArray submodule (issue #135).

The classes used to be defined *inside* ``morton_index._build_classes()`` and
reached through a module ``__getattr__``. PEP 562 ``__getattr__`` fires only on
a **missing** attribute, so ``vars()`` / ``dir()`` / ``inspect.getmembers()``
never saw them and mkdocstrings' static analysis rendered nothing. They now live
at module level in ``mortie/pandas.py``, which imports pandas at *its* top level.

What is pinned here is the property that motivated the placement, plus the two
behaviours the placement must not break: ``import mortie`` still works with
pandas absent, and every pre-existing import path still resolves.
"""

import inspect
import subprocess
import sys

import pytest

import mortie
import mortie.morton_index

# NB: no module-level `importorskip("pandas")` here. The TestPandasAbsent cases
# deliberately run in an interpreter where pandas is unimportable, so they must
# stay collectable in a numpy-only environment; the in-process classes ask for
# pandas themselves via an autouse fixture.


def _run_without_pandas(body):
    """Run ``body`` in a fresh interpreter where pandas is unimportable.

    Same meta_path-blocker pattern as
    ``test_decimal_parse.TestPublicSurface::test_parses_with_pandas_unavailable``
    -- ``import mortie`` genuinely touches pandas when it is installed (the
    eager registration probe at the bottom of ``morton_index.py``), so proving
    the numpy-only claim means making pandas actually unimportable rather than
    just checking ``sys.modules``.
    """
    prelude = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'pandas' or name.startswith('pandas.'):\n"
        "            raise ImportError('pandas blocked for this test')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
    )
    return subprocess.run(
        [sys.executable, "-c", prelude + body],
        capture_output=True,
        text=True,
    )


class TestModuleLevelDefinitions:
    """The refactor's whole point: statically discoverable class objects."""

    @pytest.fixture(autouse=True)
    def _needs_pandas(self):
        pytest.importorskip("pandas")

    def test_classes_are_in_module_vars(self):
        # False before issue #135 -- the names existed only via __getattr__.
        import mortie.pandas as mp

        assert "MortonIndexArray" in vars(mp)
        assert "MortonIndexDtype" in vars(mp)

    def test_qualname_is_not_a_function_local(self):
        import mortie.pandas as mp

        assert mp.MortonIndexArray.__qualname__ == "MortonIndexArray"
        assert mp.MortonIndexDtype.__qualname__ == "MortonIndexDtype"
        assert mp.MortonIndexArray.__module__ == "mortie.pandas"
        assert mp.MortonIndexDtype.__module__ == "mortie.pandas"

    def test_discoverable_by_dir_and_getmembers(self):
        # This is exactly what mkdocstrings/griffe needs in order to render the
        # classes on the API page.
        import mortie.pandas as mp

        found = {name for name, _ in inspect.getmembers(mp, inspect.isclass)}
        assert {"MortonIndexArray", "MortonIndexDtype"} <= found
        assert {"MortonIndexArray", "MortonIndexDtype"} <= set(dir(mp))

    def test_submodule_named_pandas_does_not_shadow_real_pandas(self):
        # Python 3 absolute imports: `import pandas` inside mortie/pandas.py
        # resolves to the real pandas, and top-level `import pandas` is
        # unaffected by the submodule's existence.
        import pandas as real_pandas

        import mortie.pandas as mp

        assert mp is not real_pandas
        assert mp.pd is real_pandas
        assert real_pandas.__name__ == "pandas"

    def test_star_import_does_not_bind_the_name_pandas(self):
        # `from mortie import *` must not drop mortie's submodule onto the
        # caller's `pandas` name -- that is the one place the naming *would*
        # shadow, so 'pandas' is kept out of __all__ on purpose.
        assert "pandas" not in mortie.__all__
        namespace = {}
        exec("from mortie import *", namespace)  # noqa: S102
        assert "pandas" not in namespace


class TestImportPathsStillResolve:
    """Every path that worked before the move must still work."""

    @pytest.fixture(autouse=True)
    def _needs_pandas(self):
        pytest.importorskip("pandas")

    def test_all_three_paths_are_the_same_object(self):
        import mortie.pandas as mp

        assert mortie.MortonIndexArray is mp.MortonIndexArray
        assert mortie.morton_index.MortonIndexArray is mp.MortonIndexArray
        assert mortie.MortonIndexDtype is mp.MortonIndexDtype
        assert mortie.morton_index.MortonIndexDtype is mp.MortonIndexDtype

    def test_from_import_off_morton_index_still_works(self):
        # Load-bearing downstream: zagg imports this exact path.
        from mortie.morton_index import MortonIndexArray, MortonIndexDtype

        assert MortonIndexArray.__name__ == "MortonIndexArray"
        assert MortonIndexDtype.__name__ == "MortonIndexDtype"

    def test_dtype_string_is_registered_on_plain_import(self):
        # `@register_extension_dtype` runs at class creation, so the eager
        # submodule import at the bottom of morton_index.py is what makes the
        # string resolvable without the user touching the classes first.
        pd = pytest.importorskip("pandas")
        series = pd.Series(dtype="morton_index")
        assert series.dtype.name == "morton_index"
        assert type(series.array) is mortie.MortonIndexArray

    def test_dtype_string_registers_in_a_fresh_interpreter(self):
        # In-process the dtype may already be registered by another test's
        # import; a fresh interpreter proves `import mortie` alone does it.
        code = (
            "import mortie, pandas as pd\n"
            "s = pd.Series(dtype='morton_index')\n"
            "print(s.dtype.name, type(s.array).__module__)\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.split() == ["morton_index", "mortie.pandas"]


class TestPandasAbsent:
    """The property the whole lazy design exists to protect."""

    def test_import_mortie_works_without_pandas(self):
        out = _run_without_pandas(
            "import mortie\n"
            "assert 'pandas' not in sys.modules\n"
            "assert 'mortie.pandas' not in sys.modules\n"
            "print(int(mortie.decimal_to_word('-31123')))\n"
        )
        assert out.returncode == 0, out.stderr
        assert int(out.stdout.strip()) == mortie.decimal_to_word(
            "-31123", dtype=int
        )

    @pytest.mark.parametrize(
        "expr",
        [
            "mortie.MortonIndexArray",
            "mortie.MortonIndexDtype",
            "mortie.morton_index.MortonIndexArray",
            "mortie.pandas.MortonIndexArray",
        ],
    )
    def test_attribute_paths_raise_the_curated_importerror(self, expr):
        out = _run_without_pandas(
            "import mortie, mortie.morton_index\n"
            "try:\n"
            f"    {expr}\n"
            "except ImportError as exc:\n"
            "    print(exc)\n"
            "else:\n"
            "    raise AssertionError('expected ImportError')\n"
        )
        assert out.returncode == 0, out.stderr
        assert "installed alongside mortie" in out.stdout

    @pytest.mark.parametrize(
        "stmt",
        [
            "from mortie.pandas import MortonIndexArray",
            "from mortie.morton_index import MortonIndexArray",
            "import mortie.pandas",
        ],
    )
    def test_from_import_paths_raise_the_curated_importerror(self, stmt):
        # A bare top-level `import pandas` in mortie/pandas.py would regress
        # the direct-import path to a plain ModuleNotFoundError; the guard is
        # what keeps the curated message on it.
        out = _run_without_pandas(
            "try:\n"
            f"    {stmt}\n"
            "except ImportError as exc:\n"
            "    print(exc)\n"
            "else:\n"
            "    raise AssertionError('expected ImportError')\n"
        )
        assert out.returncode == 0, out.stderr
        assert "installed alongside mortie" in out.stdout

    def test_message_is_identical_on_every_path(self):
        # One definition (`morton_index._require_pandas`), so the wording
        # cannot drift between the attribute path and the direct import.
        out = _run_without_pandas(
            "import mortie, mortie.morton_index\n"
            "msgs = []\n"
            "for get in (\n"
            "    lambda: mortie.MortonIndexArray,\n"
            "    lambda: mortie.morton_index.MortonIndexArray,\n"
            "):\n"
            "    try:\n"
            "        get()\n"
            "    except ImportError as exc:\n"
            "        msgs.append(str(exc))\n"
            "try:\n"
            "    from mortie.pandas import MortonIndexArray\n"
            "except ImportError as exc:\n"
            "    msgs.append(str(exc))\n"
            "assert len(msgs) == 3, msgs\n"
            "assert len(set(msgs)) == 1, msgs\n"
            "print(msgs[0])\n"
        )
        assert out.returncode == 0, out.stderr
        message = out.stdout.strip()
        assert "requires pandas, which is not installed" in message
        assert "`pip install pandas`" in message
        assert "`pip install mortie[pandas]`" in message
        assert "installed alongside mortie" in message
        # The message is only ever raised once `import pandas` has failed, so
        # it must point at installing rather than at importing.
        assert "import pandas to use" not in message

"""Drift pin for the example notebooks' Binder launch links (issue #180).

CLAUDE.md requires every notebook under ``examples/`` to be runnable on Binder
*and* to carry an explicit launch link in the rendered notebook.  The link
embeds the notebook's own path (``?labpath=examples%2F<name>.ipynb``), so a
rename breaks the launch **silently**: the badge still renders, the URL still
resolves, and Binder builds the image and lands on a missing file.  Nothing
else in the repo reads that path, so this test is the only thing tying it to
the filename.

Deliberately not checked: that the notebook executes.  Running the notebooks
would need a runner dependency and a workflow change (see the PR thread on
issue #180); this file pins only what is cheap and static.
"""

import json
from pathlib import Path
from urllib.parse import quote

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
BINDER_PREFIX = "https://mybinder.org/v2/gh/espg/mortie/HEAD?labpath="


def first_markdown_source(path):
    """The source text of a notebook's first cell, which must be markdown."""
    nb = json.loads(path.read_text())
    assert nb["nbformat"] == 4, f"{path.name}: expected nbformat 4"
    cell = nb["cells"][0]
    assert cell["cell_type"] == "markdown", (
        f"{path.name}: first cell is {cell['cell_type']}, so the Binder link "
        "is not the first thing a reader sees"
    )
    return "".join(cell["source"])


def test_example_notebooks_found():
    # Guards the parametrize below: an empty glob would skip every case
    # silently instead of failing (mirrors test_docs_api_pages.py).
    assert sorted(EXAMPLES_DIR.glob("*.ipynb")), (
        f"no notebooks found under {EXAMPLES_DIR}")


@pytest.mark.parametrize("notebook", sorted(EXAMPLES_DIR.glob("*.ipynb")),
                         ids=lambda p: p.name)
def test_binder_link_targets_the_notebook_itself(notebook):
    expected = BINDER_PREFIX + quote(f"examples/{notebook.name}", safe="")
    assert expected in first_markdown_source(notebook), (
        f"{notebook.name}: intro cell has no Binder launch link for itself; "
        f"expected {expected}"
    )

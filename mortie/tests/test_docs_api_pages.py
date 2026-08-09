"""Drift pin for the docs/api member partition (issue #170).

``mkdocs.yml`` sets ``strict: true``, but mkdocstrings drops a ``members:``
entry its module does not have *silently* — the Docs job stays green and the
function simply has no rendered API page.  Issue #170's move surfaced exactly
that: between its phase 1 and phase 3, four pages listed members their modules
no longer had, and nothing in CI said so.  This test is the backstop: every
``members:`` name in ``docs/api/*.md`` must resolve as an attribute of the
module its page documents, so a stale entry (or a move that forgets its docs
page) fails here instead of vanishing from the site.
"""

import importlib
import re
from pathlib import Path

import pytest

API_DIR = Path(__file__).resolve().parents[2] / "docs" / "api"


def page_members(path):
    """The (module, members) a docs/api page declares, from its mkdocstrings block."""
    text = path.read_text()
    module = re.search(r"^::: +(\S+)$", text, re.M)
    assert module, f"{path.name}: no '::: module' mkdocstrings block"
    members = re.findall(r"^ +- +(\S+)$", text.split("members:", 1)[1], re.M)
    assert members, f"{path.name}: empty members list"
    return module.group(1), members


@pytest.mark.parametrize("page", sorted(API_DIR.glob("*.md")),
                         ids=lambda p: p.name)
def test_every_member_resolves_on_its_module(page):
    module_name, members = page_members(page)
    module = importlib.import_module(module_name)
    missing = [name for name in members if not hasattr(module, name)]
    assert not missing, (
        f"{page.name} lists members {module_name} does not have: {missing} "
        "(mkdocstrings drops these silently — the rendered page just loses them)"
    )
    assert len(members) == len(set(members)), f"{page.name}: duplicate members"

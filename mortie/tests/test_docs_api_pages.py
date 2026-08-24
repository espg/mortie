"""Drift pin for the docs/api member partition (issue #170).

``mkdocs.yml`` sets ``strict: true``, but mkdocstrings drops a ``members:``
entry its module does not have *silently* — the Docs job stays green and the
function simply has no rendered API page.  Issue #170's move surfaced exactly
that: between its phase 1 and phase 3, four pages listed members their modules
no longer had, and nothing in CI said so.  This test is the backstop for the
**stale-entry** direction: every ``members:`` name in ``docs/api/*.md`` must
resolve as an attribute of the module its page documents.

The reverse direction is pinned too (issue #176): every non-submodule name in
``mortie.__all__`` must appear on exactly one page — keyed on *(defining
module, name)*, so the arrow skins of core functions stay legal on their own
page — and every submodule in ``__all__`` must have a page whose ``:::`` block
is that module.  A move that deletes a member from its old page and forgets
the new one now fails here instead of vanishing silently.  The only names
allowed off the pages are the two lazily-built Arrow classes
(``LAZY_ARROW_UNDOCUMENTED``), which mkdocstrings cannot resolve statically.

And the docs cannot outgrow the frozen surface either: every ``members:``
entry must be in ``__all__`` by name, or sit in ``MODULE_SCOPED_DOCUMENTED``
— the documented names deliberately reached through their submodule
(``mortie.arrow.export_c_array``, ``mortie.morton_index.MortonIndexScalar``)
rather than flat.  Growing either roster is a deliberate act reviewed here,
not a silent omission.
"""

import importlib
import inspect
import re
from pathlib import Path

import pytest

API_DIR = Path(__file__).resolve().parents[2] / "docs" / "api"

# The two pyarrow extension classes are defined inside ``_build_type()`` and
# reached only through ``arrow.__getattr__`` (pyarrow is an optional extra),
# so griffe's static resolution cannot see them and no ``members:`` entry can
# render them.  They are documented narratively in docs/arrow_interchange.md
# instead — the structural reason will not expire, so this roster should
# never grow (issue #176).  Never ``getattr`` these here: resolving them
# raises ImportError when pyarrow is absent.
LAZY_ARROW_UNDOCUMENTED = frozenset({"MortonIndexType", "MortonIndexExtArray"})

# Documented names deliberately *not* flat on the package: each is public as
# an attribute of a submodule that is itself in ``__all__``.  The C Data
# Interface trio is namespaced interop plumbing (``mortie.arrow.export_c_array``,
# issue #93); ``MortonIndexScalar`` is the repr/scalar type handed back by the
# ExtensionArray, spelled ``mortie.morton_index.MortonIndexScalar`` (#104).
MODULE_SCOPED_DOCUMENTED = frozenset({
    ("mortie.arrow", "export_c_array"),
    ("mortie.arrow", "export_c_schema"),
    ("mortie.arrow", "import_c_array"),
    ("mortie.morton_index", "MortonIndexScalar"),
})

_MISSING = object()


def all_pages():
    """Map each documented module to its page name and ``members:`` roster."""
    return {
        module: (page.name, members)
        for page in sorted(API_DIR.glob("*.md"))
        for module, members in [page_members(page)]
    }


def page_members(path):
    """The (module, members) a docs/api page declares, from its mkdocstrings block."""
    text = path.read_text()
    modules = re.findall(r"^::: +(\S+)$", text, re.M)
    # One block per page is the repo convention this parser assumes: with two,
    # splitting on the first "members:" would credit every entry to the first
    # module.  Fail loud here rather than mis-associate.
    assert len(modules) == 1, f"{path.name}: expected exactly one '::: module' block"
    members = re.findall(r"^ +- +(\S+)$", text.split("members:", 1)[1], re.M)
    assert members, f"{path.name}: empty members list"
    return modules[0], members


def test_api_pages_found():
    # Guards the parametrize below: if docs/api goes missing (or moves), an
    # empty glob would skip every case silently instead of failing.
    assert sorted(API_DIR.glob("*.md")), f"no docs pages found under {API_DIR}"


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


def test_every_public_name_documented_on_exactly_one_page():
    # The reverse direction of the pin above (issue #176): a name in
    # ``mortie.__all__`` that no page lists has no rendered API entry at all
    # — deleting ``- morton_buffer`` from buffer.md must fail here, not
    # vanish silently.  Keyed on (defining module, name) by object identity,
    # so a page documenting another module's *skin* of the same name (the
    # arrow forms of from_wkb / polygons_to_morton_mocs) neither satisfies
    # nor double-counts the flat name.
    import mortie

    pages = all_pages()
    problems = []
    for name in mortie.__all__:
        if name in LAZY_ARROW_UNDOCUMENTED:
            continue  # structurally unrenderable; see the roster's comment
        obj = getattr(mortie, name)
        if inspect.ismodule(obj):
            if f"mortie.{name}" not in pages:
                problems.append(f"submodule {name}: no docs/api page")
            continue
        homes = [
            fname
            for module, (fname, members) in pages.items()
            if name in members
            and getattr(importlib.import_module(module), name, _MISSING) is obj
        ]
        if len(homes) != 1:
            problems.append(f"{name}: on {homes or 'no page'}")
    assert not problems, (
        "public names must render on exactly one docs/api page "
        f"(the page of the module they are bound from): {problems}"
    )


def test_lazy_arrow_roster_is_not_stale():
    # If someone finds a way to render the lazy classes (stub declarations,
    # a griffe extension), the allowlist must shrink in the same change.
    import mortie

    documented = {name for _, (_, members) in all_pages().items() for name in members}
    assert not LAZY_ARROW_UNDOCUMENTED & documented, (
        "allowlisted-as-undocumentable names now appear on a page — prune "
        f"LAZY_ARROW_UNDOCUMENTED: {sorted(LAZY_ARROW_UNDOCUMENTED & documented)}"
    )
    missing = LAZY_ARROW_UNDOCUMENTED - set(mortie.__all__)
    assert not missing, f"allowlisted names no longer public: {sorted(missing)}"


def test_every_documented_name_is_public():
    # The docs cannot outgrow the frozen surface: a ``members:`` entry must
    # be reachable from ``mortie.__all__`` — flat by name, or through the
    # justified module-scoped roster.  Growing MODULE_SCOPED_DOCUMENTED is a
    # deliberate, reviewed act (issue #176).
    import mortie

    pages = all_pages()
    public = set(mortie.__all__)
    stray = [
        f"{fname}: {name}"
        for module, (fname, members) in pages.items()
        for name in members
        if name not in public and (module, name) not in MODULE_SCOPED_DOCUMENTED
    ]
    assert not stray, f"documented names missing from mortie.__all__: {stray}"

    documented_pairs = {
        (module, name)
        for module, (_, members) in pages.items()
        for name in members
    }
    stale = MODULE_SCOPED_DOCUMENTED - documented_pairs
    assert not stale, f"MODULE_SCOPED_DOCUMENTED entries no longer on a page: {sorted(stale)}"
    unreachable = {
        (module, name)
        for module, name in MODULE_SCOPED_DOCUMENTED
        if module.removeprefix("mortie.") not in public
    }
    assert not unreachable, (
        "module-scoped names must hang off a submodule that is itself in "
        f"__all__: {sorted(unreachable)}"
    )

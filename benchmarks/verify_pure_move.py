"""Verify that a module split moved definitions verbatim (issues #159, #170).

A mortie module split claims to be **pure moves plus import rewiring** — every
top-level definition lands in its new module byte-for-byte, and no public name
changes.  That claim was checked by hand for the first slice of the plan (`PR
#160 <https://github.com/espg/mortie/pull/160>`_ extracted ``moc.py`` out of
``coverage.py``; its review AST-compared all 11 moved definitions against their
pre-move originals).  This script is that check made re-runnable, so the
reviewer — and the next split — does not have to re-derive it.  It was built for
the **domain** split of ``tools.py`` / ``geometry.py`` (issue #159) and then
carried the **arity** split that consolidated the plural operators into
``batch.py`` (issue #170); ``SPLITS`` was the only thing either stated, and
both entries are now retired (see the notes above ``SPLITS``), so a default
run checks only the public surface.

Three claims are checked against a git base (``origin/main`` by default):

1. **Verbatim.**  Every top-level definition in a destination module that also
   exists in one of the split's source modules at the base compares equal, both
   as an AST (``ast.dump``, so formatting and line numbers are ignored) and as
   literal source text (so comments *inside* a definition are covered too).
2. **Complete.**  Every top-level definition the split's sources had at the base
   lands in exactly one destination module — nothing silently lost or
   duplicated — and no destination gains a definition that was not there
   before.  A split with several sources (issue #170's ``batch.py`` collects
   from four) indexes them as one union, and a name bound in two sources is a
   failure rather than an arbitrary pick — see ``index_sources``.  The union
   keeps each definition's source, and a destination that is *also* a source
   must keep its own: only a destination outside the source set may take a
   definition from any of them, so a function cannot migrate between two
   stay-put modules and still read as a clean move — see ``check_moves``.  A
   top-level statement the scanner cannot name and compare (a tuple-target
   assignment, an ``if TYPE_CHECKING:`` block, a ``try/except ImportError``
   shim, a loop, an ``__all__ +=``) is reported as a failure rather than
   skipped, so this arm fails loud instead of open — see ``top_level_defs``.
3. **Public surface pinned.**  ``set(mortie.__all__)`` equals the base's, and
   every name in it still resolves as an attribute of ``mortie``.  The base
   package is extracted with ``git archive`` and imported in a subprocess (with
   the built ``_rustie`` extension copied in), so this compares two real
   imports rather than two guesses at what ``__init__.py`` evaluates to.

A split cut from a tree an **earlier phase of the same PR** already touched pins
its own base in ``SPLIT_BASES`` rather than using ``--base``, so each arm stays
a strict verbatim check of its own move.  The pin is not trusted on its word: a
fourth arm indexes the pinned source at ``--base`` too and requires every
definition to be equal once body-level ``Import``/``ImportFrom`` statements are
dropped.  Rewiring those imports is exactly what an earlier split legitimately
does to a module a later split then moves out of; without this arm a change made
*in* that earlier split sits in both the pinned base and the destination, and no
arm can see it — see ``check_pinned_bases``.  ``SPLIT_BASES`` was empty for
issue #170, whose four sources were all read at ``--base`` unmodified.

**An entry's life ends when its split merges** — or, as with issue #170's,
when a later phase of its own PR makes a sanctioned docstring edit to the
moved definitions.  Every arm is stated relative to ``--base``, and once the
split is on ``main`` that revision *is* the post-split tree: the source module
may not exist any more, and a pinned pre-split module is compared against one
that no longer holds what it gave away.  Retire the ``SPLITS`` entry and any
pin with it at that point, as issue #170 did for #159's and then for its own;
the passing run belongs in the PR's body, not in a check that can no longer
make it.

Run::

    python benchmarks/verify_pure_move.py [--base origin/main]

Exit status is non-zero if any claim fails.

Known limitations — a green run here is half the gate, not the whole one:

* A comment block sitting *between* two top-level definitions belongs to no
  definition's source segment, so it is not compared.  Comments inside a
  definition body are.
* An attribute docstring is keyed to the last **named** definition above it,
  and a skipped ``import`` in between does not break that association — so a
  stray top-level string following an import is keyed to a name it does not
  document.  That fails loud rather than open (the mis-keyed name is compared
  like any other, so a move of it is reported as "not a move", and a second
  such string comes back ``unhandled``); mortie has none.
* Only the **moves** are checked, not the *import rewiring* that goes with
  them.  Deleting ``geometry.py``'s ``from .dissolve import
  _dissolved_polygons`` leaves all 76 definitions verbatim and all 69
  ``__all__`` names resolvable, so this script still exits 0 — ``pytest`` is
  what catches it (``test_emit_dissolve_is_the_default`` raises ``NameError:
  name '_dissolved_polygons' is not defined``).  Run both.
* A **test** module split is not modelled, and *neither* arm can take it.  This
  did not arise for issue #170, which moves no test definition at all — it
  rewires five ``from mortie.<module> import`` lines in the suite and nothing
  else, so ``pytest`` is the whole gate on the test side and there is no
  unmodelled move to review-gate.  It did arise for #159, whose phase 3 cut
  ``test_tools.py`` into ``test_convert.py`` / ``test_orders.py``:
  ``check_moves`` indexes a pytest class fine — it is a top-level ``ClassDef``
  — but the trailing ``if __name__ == "__main__"`` block is not comparable, so
  it reports 3 failures (the source and both destinations) before
  ``check_pinned_bases`` adds 15 of its own.  That second arm's contract is
  that a pinned base differs from ``--base`` in body-level *imports* alone, and
  phase 1 rewrote the call sites inside every one of that module's bodies
  (``tools.geo2mort`` -> ``convert.geo2mort``), not just its imports.
  Weakening either would give back the guarantee they exist to provide, so the
  test split is verified by the same completeness argument made directly
  instead: seventeen top-level statements in, seventeen out, byte-identical,
  with only ``if __name__ == "__main__"`` deliberately in both files.

  **So phase 3 is review-gated, not machine-gated** — it is the one phase of
  the split whose numbers this script does not reproduce, and they should not
  be read as a machine-checked property the way the other three phases' are.
  ``pytest`` is no second half here either: the only thing it can catch in a
  *test* move is a test that stops passing, and a weakened one still passes —
  turning ``assert parent == 7`` into ``assert parent == parent`` inside a
  moved class leaves this script at exit 0 and the suite at its usual count.
"""

import argparse
import ast
import copy
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

# Each entry: the split's source module(s) at the base -> the modules their
# definitions moved into.  A destination that is also a source simply means part
# of it stayed put.  The key is a **tuple** when one split draws from several
# sources at once, as issue #170's does: five plural operators converge on one
# new module, so ``batch.py`` is a destination of four different sources and the
# arms below index their definitions as one union.  A bare string key is
# shorthand for a one-source split.
#
# RETIRED (issue #159, merged as 8d4eb0d): ``mortie/tools.py`` ->
# convert/orders/buffer and ``mortie/geometry.py`` -> geometry/dissolve/codec,
# with ``mortie/geometry.py`` pinned to ``011816ca``.  Both arms stop being
# checkable the moment that split lands on ``main``, and not because the merge
# was done wrong: it *was* a merge commit, so ``011816ca`` is still reachable.
# The arms die because they are stated relative to ``--base``, and once the
# split is merged ``origin/main`` **is** the post-split tree —
# ``origin/main:mortie/tools.py`` no longer exists (1 failure), and
# ``check_pinned_bases`` compares the pinned pre-split ``geometry.py`` against a
# ``geometry.py`` that no longer holds the 25 definitions dissolve/codec took
# (25 failures).  Retiring them here is the end of life the ``SPLIT_BASES``
# comment below always specified; the split they verified is recorded in PR #169
# and in the run pasted in its body.
#
# RETIRED (issue #170, at phase 2 of its own PR): the four-source **arity**
# split — coverage/geometry/moc/orders -> ``batch.py``, stated as one entry
# with a tuple key — was verified green at the phase-1 move commit
# (``e2b0281``); the complete run is pasted in PR #172's body as the record.
# Phase 2 then cross-linked the scalar/plural docstrings (the one sanctioned
# deviation from a pure move), so the five moved definitions stop being
# byte-identical to ``--base`` and ``check_moves`` would report the five
# ``See Also`` edits as real differences.  Pinning instead of retiring cannot
# work here: at phase 1's head the move has already happened, so a pinned
# source holds none of the five (each comes back "not a move"), and the pin
# would be a branch sha whose arm breaks the day the PR merges — the exact
# landmine the #159 retirement above documents.  The pure-move property rests
# on the phase-1 commit plus the recorded run, not on this (now empty) table.
SPLITS = {}

# A split whose source was already touched by an *earlier* split **within the
# same PR** verifies against the commit it was actually cut from, not against
# ``--base``.  #159's phase 2 cut ``dissolve.py`` out of a ``geometry.py`` that
# its phase 1 had already import-rewired, so comparing it to ``origin/main``
# would report those rewires as differences and mask any real one.  Pinning the
# base per split keeps every arm a strict verbatim check of its own move, and
# ``check_pinned_bases`` checks the pin itself against ``--base``.
#
# CAVEAT: these are *branch* commits, and their useful life ends when the split
# they describe merges.  A squash-merge or rebase orphans the sha outright; even
# an ordinary merge commit, which keeps it reachable, only keeps ``git show``
# working — ``check_pinned_bases`` states the pin *relative to* ``--base``, and
# after the merge ``origin/main`` is the post-split tree the pin is supposed to
# differ from.  So the entry retires with its ``SPLITS`` arm, once the move has
# landed and the check has served its purpose.  That is what happened to
# #159's ``mortie/geometry.py``: ``011816ca`` did stay reachable through the
# merge, exactly as intended, and the arm still had to go.  It is not repointed
# anywhere — its job is done, and PR #169 is where its passing run is recorded.
#
# Issue #170's own split needs **no pin**.  Its four sources are all read at
# ``--base`` (``origin/main``, the post-#159 merge ``8d4eb0d``), which is
# precisely the tree this branch was cut from and the tree the move was made
# against: nothing in this PR edits a source module before moving out of it, so
# there is no earlier phase for a pin to isolate.  ``orders.py`` and
# ``geometry.py`` being #159 products does not change that — #159's rewiring is
# already *in* ``--base``, so the strict arm compares this move against a tree
# that already carries it, and the fourth arm (whose whole job is to tolerate
# such rewiring across a pin) has nothing to tolerate.
SPLIT_BASES = {}

# Definitions legitimately introduced by the split rather than moved.  Empty is
# the goal; anything listed here must be justified in the PR body.
EXPECTED_NEW = {}

REPO = pathlib.Path(__file__).resolve().parent.parent


def git_show(base, path):
    """Read a repository file as of a git revision.

    Parameters
    ----------
    base : str
        Git revision to read from, e.g. ``origin/main``.
    path : str
        Repository-relative path of the file.

    Returns
    -------
    str
        The file's contents at that revision.
    """
    return subprocess.run(
        ["git", "show", f"{base}:{path}"],
        cwd=REPO, check=True, capture_output=True, text=True,
    ).stdout


def top_level_defs(source):
    """Index a module's top-level named definitions by name.

    Statements that bind no comparable name — a tuple-target assignment, an
    ``if``/``try`` block, a loop, an augmented assignment — are *not* skipped.
    Skipping them would drop them from both sides of the comparison at once, so
    a definition could be lost or altered while the run still reported
    ``N/N accounted for``.  They come back as ``unhandled`` for the caller to
    raise as a failure, which turns "the scanner does not know about this
    construct" into a loud stop rather than a silent pass.  Imports and the
    module docstring are the two exceptions: the split rewrites both by design.

    An **attribute docstring** — the bare string literal documenting the
    assignment above it, as ``RingValidity`` carries in ``mortie/coverage.py`` —
    is the one bare expression that is indexed rather than reported.  It is
    keyed ``<name>.__doc__``, a spelling no Python name can collide with, so it
    is compared verbatim and has to land wherever its definition lands.  Left
    unhandled it would fail every run touching that module, and skipping it
    would let a docstring change ride along unseen.

    Parameters
    ----------
    source : str
        Python source text of one module.

    Returns
    -------
    dict
        Name -> ``(ast node, source text)`` for every top-level function,
        class, and simple-name assignment (annotated or not), plus
        ``<name>.__doc__`` for each attribute docstring.
    list of str
        One ``"<StatementKind> at line N"`` entry per top-level statement the
        scanner cannot name and compare.

    Raises
    ------
    ValueError
        If the module binds the same top-level name twice, which would make
        the comparison ambiguous.
    """
    tree = ast.parse(source)
    found = {}
    unhandled = []
    previous = None
    body = tree.body
    if ast.get_docstring(tree) is not None:
        body = body[1:]
    for node in body:
        documents = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = documents = [node.name]
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if len(names) != len(targets):
                # a tuple/list/attribute/subscript target: no single name to key on
                unhandled.append(f"{type(node).__name__} at line {node.lineno}")
                previous = None
                continue
            documents = names
        elif (previous is not None and isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            names = [f"{previous}.__doc__"]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        else:
            unhandled.append(f"{type(node).__name__} at line {node.lineno}")
            previous = None
            continue
        # only a single-name definition can carry an attribute docstring
        previous = documents[0] if documents and len(documents) == 1 else None
        for name in names:
            if name in found:
                raise ValueError(f"top-level name bound twice: {name}")
            found[name] = (node, ast.get_source_segment(source, node))
    return found, unhandled


def unreachable(base, path):
    """Phrase the failure for a git revision this clone cannot resolve.

    ``git show`` raises rather than returning, and the pinned shas in
    ``SPLIT_BASES`` are branch commits a squash-merge invalidates, so this is
    the expected way for the script to stop working once the split lands.  A
    traceback would say ``CalledProcessError ... exit status 128``; this says
    what to do about it.

    Parameters
    ----------
    base : str
        The unresolvable git revision.
    path : str
        Repository-relative path that was being read at that revision.

    Returns
    -------
    str
        A failure message naming the revision and the fix.
    """
    hint = ("repoint or drop its SPLIT_BASES entry — a squash-merge rewrites a "
            "branch sha" if SPLIT_BASES.get(path) == base else "check --base")
    return f"{base}:{path} is not reachable in this clone — {hint}"


def label_of(base):
    """Abbreviate a git revision for printing.

    Parameters
    ----------
    base : str
        Git revision, possibly a full 40-character sha.

    Returns
    -------
    str
        The revision, shortened to seven characters when it is a full sha.
    """
    return base[:7] if re.fullmatch(r"[0-9a-f]{40}", base) else base


def modulo_body_imports(source, node):
    """Render a definition with its body-level import statements dropped.

    Only statements sitting *directly* in the definition's body are dropped: an
    import nested inside an ``if`` or a ``try``, and every non-import statement
    anywhere, still counts as a difference.  Dropping is by whole line, so a
    trailing comment on an import line goes with it — but a blank line beside
    one does not, and shows up as a source-text difference.

    Parameters
    ----------
    source : str
        Python source text of the module ``node`` was parsed from.
    node : ast.AST
        A top-level definition node from that module.

    Returns
    -------
    str
        ``ast.dump`` of the node with body-level imports removed.
    str
        The node's source text with body-level import lines removed.
    """
    stripped = copy.deepcopy(node)
    body = getattr(stripped, "body", None)
    drop = set()
    if isinstance(body, list):
        for stmt in body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                drop.update(range(stmt.lineno, (stmt.end_lineno or stmt.lineno) + 1))
        stripped.body = [s for s in body
                         if not isinstance(s, (ast.Import, ast.ImportFrom))]
    text = ast.get_source_segment(source, node)
    if drop:
        text = "\n".join(line for i, line in enumerate(text.splitlines())
                         if node.lineno + i not in drop)
    return ast.dump(stripped), text


def check_pinned_bases(default_base):
    """Check each pinned base against ``default_base``, modulo import rewiring.

    ``SPLIT_BASES`` pins a split to the commit it was cut from so its own arm
    stays strict — but that makes the pinned tree *trusted* rather than checked.
    A change the earlier split introduced into the source module sits in both
    the pinned base and the destination, so ``check_moves`` compares it against
    itself and reports the move verbatim.  This arm closes that seam: every
    definition of the pinned source must equal ``default_base``'s once
    body-level imports are dropped, which tolerates precisely the rewiring an
    earlier split does and nothing else.

    Parameters
    ----------
    default_base : str
        Git revision the pinned bases are themselves checked against.

    Returns
    -------
    list of str
        One message per failure; empty when every pinned base differs from
        ``default_base`` in body-level import statements alone.
    """
    failures = []
    for src_path, base in SPLIT_BASES.items():
        before = len(failures)
        if base == default_base:
            continue
        try:
            root_src = git_show(default_base, src_path)
        except subprocess.CalledProcessError:
            failures.append(unreachable(default_base, src_path))
            continue
        try:
            pinned_src = git_show(base, src_path)
        except subprocess.CalledProcessError:
            continue  # check_moves already reported it; do not say it twice
        pinned, pinned_unhandled = top_level_defs(pinned_src)
        root, root_unhandled = top_level_defs(root_src)
        failures += [f"{base}:{src_path}: {what} is not comparable — extend "
                     "top_level_defs before trusting this run"
                     for what in pinned_unhandled]
        failures += [f"{default_base}:{src_path}: {what} is not comparable — extend "
                     "top_level_defs before trusting this run"
                     for what in root_unhandled]
        rewired = 0
        for name, (node, text) in pinned.items():
            if name not in root:
                failures.append(
                    f"{base}:{src_path}: {name} is not in {default_base}:{src_path} "
                    "— the pin is not a pure import rewire of it")
                continue
            new_dump, new_text = modulo_body_imports(pinned_src, node)
            old_dump, old_text = modulo_body_imports(root_src, root[name][0])
            if new_dump != old_dump:
                failures.append(
                    f"{base}:{src_path}: {name} differs from {default_base}:"
                    f"{src_path} beyond its imports (AST)")
            elif new_text != old_text:
                failures.append(
                    f"{base}:{src_path}: {name} differs from {default_base}:"
                    f"{src_path} beyond its imports (source text — comments or "
                    "formatting)")
            elif text != root[name][1]:
                rewired += 1
        for name in root:
            if name not in pinned:
                failures.append(
                    f"{base}:{src_path}: {name} is gone from the pin but present "
                    f"in {default_base}:{src_path}")
        # only claim equality when this split actually reached it: the count
        # below reads as a pass to anyone eyeballing stdout, and the failures
        # go to stderr
        print(f"{src_path}@{label_of(base)} vs {default_base}: {len(pinned)} "
              f"definitions equal modulo imports ({rewired} import-rewired)"
              if len(failures) == before else
              f"{src_path}@{label_of(base)} vs {default_base}: MISMATCH — "
              f"{len(failures) - before} failure(s), listed below")
    return failures


def index_sources(src_paths, default_base):
    """Index a split's source modules at their bases as one union.

    A split may draw from several sources at once (issue #170's five plural
    operators converge on one new module), and the arms below want a single
    "what existed before" index to compare each destination against.  A name
    bound in two of the sources is reported rather than merged: the union would
    otherwise pick one arbitrarily and compare the other module's definition
    against the wrong original.  The per-name origin returned alongside is not
    cosmetic — ``check_moves`` asserts it, so that merging the sources does not
    also merge their identities.

    Parameters
    ----------
    src_paths : tuple of str
        Repository-relative paths of the split's source modules.
    default_base : str
        Git revision to read a source at when ``SPLIT_BASES`` does not pin it.

    Returns
    -------
    dict
        Name -> ``(ast node, source text)`` across every source.
    dict
        Name -> the ``"<base>:<path>"`` it came from, for failure messages and
        for ``check_moves``'s stay-put check.
    list of str
        One message per failure: an unreadable source, an uncomparable
        top-level statement, or a name bound in two sources.
    """
    old, origin, failures = {}, {}, []
    for src_path in src_paths:
        base = SPLIT_BASES.get(src_path, default_base)
        try:
            src = git_show(base, src_path)
        except subprocess.CalledProcessError:
            failures.append(unreachable(base, src_path))
            continue
        defs, unhandled = top_level_defs(src)
        failures += [f"{base}:{src_path}: {what} is not comparable — extend "
                     "top_level_defs before trusting this run"
                     for what in unhandled]
        for name, entry in defs.items():
            if name in old:
                failures.append(
                    f"{name}: bound in both {origin[name]} and {base}:{src_path} "
                    "— the split's sources are ambiguous")
                continue
            old[name] = entry
            origin[name] = f"{base}:{src_path}"
    return old, origin, failures


def check_moves(default_base):
    """Compare every moved definition against its pre-move original.

    ``index_sources`` merges a multi-source split into one ``old`` index, which
    on its own would accept a definition landing in *any* destination of the
    set rather than the one it belongs in: lift a ``coverage.py`` function out
    of the new module, append it verbatim to ``orders.py``, and every arm still
    reports a clean move.  So the provenance ``index_sources`` records is
    asserted here rather than only quoted in failure messages — a destination
    that is **also** a source must keep its own definitions, and only a
    destination outside the source set (``batch.py`` for issue #170) may take a
    definition from any of them.  The one-source model got this for free; the
    union has to say it.

    Parameters
    ----------
    default_base : str
        Git revision holding the pre-move source, for every source that
        ``SPLIT_BASES`` does not pin to one of its own.

    Returns
    -------
    list of str
        One message per failure; empty when every definition is verbatim,
        accounted for, unduplicated, and — for a destination that is also a
        source — its own.
    """
    failures = []
    for src_paths, dst_paths in SPLITS.items():
        if isinstance(src_paths, str):
            src_paths = (src_paths,)
        old, origin, src_failures = index_sources(src_paths, default_base)
        failures += src_failures
        sources = ", ".join(src_paths)
        landed = {}
        for dst_path in dst_paths:
            new, new_unhandled = top_level_defs((REPO / dst_path).read_text())
            failures += [f"{dst_path}: {what} is not comparable — extend "
                         "top_level_defs before trusting this run"
                         for what in new_unhandled]
            for name, (node, text) in new.items():
                if name not in old:
                    if EXPECTED_NEW.get(dst_path, {}).get(name):
                        continue
                    failures.append(
                        f"{dst_path}: {name} is not a move — no such definition "
                        f"in {sources}")
                    continue
                if name in landed:
                    failures.append(
                        f"{name}: defined in both {landed[name]} and {dst_path}")
                    continue
                landed[name] = dst_path
                if (dst_path in src_paths
                        and origin[name] != f"{SPLIT_BASES.get(dst_path, default_base)}"
                                            f":{dst_path}"):
                    failures.append(
                        f"{dst_path}: {name} came from {origin[name]} — a stay-put "
                        "destination gained another source's definition")
                old_node, old_text = old[name]
                if ast.dump(node) != ast.dump(old_node):
                    failures.append(
                        f"{dst_path}: {name} differs from {origin[name]} (AST)")
                elif text != old_text:
                    failures.append(
                        f"{dst_path}: {name} differs from {origin[name]} "
                        "(source text — comments or formatting)")
        for name in old:
            if name not in landed:
                failures.append(
                    f"{origin[name]}: {name} landed in none of "
                    f"{', '.join(dst_paths)}")
        labelled = ", ".join(
            f"{p}@{label_of(SPLIT_BASES.get(p, default_base))}" for p in src_paths)
        print(f"{labelled}: {len(landed)}/{len(old)} definitions accounted for "
              f"across {', '.join(dst_paths)}")
    return failures


def base_public_surface(base):
    """Import the package as of ``base`` and return its ``__all__``.

    The tree is extracted with ``git archive`` into a temporary directory and
    the built ``_rustie`` extension is copied in, so the import is real rather
    than a static reading of ``__init__.py``.

    Parameters
    ----------
    base : str
        Git revision to import.

    Returns
    -------
    list of str
        ``mortie.__all__`` as evaluated at that revision.

    Raises
    ------
    RuntimeError
        If the subprocess imported a ``mortie`` from outside the extracted
        tree, which would silently compare the working tree against itself.
    """
    with tempfile.TemporaryDirectory() as tmp:
        archive = subprocess.run(
            ["git", "archive", base, "mortie"],
            cwd=REPO, check=True, capture_output=True,
        ).stdout
        subprocess.run(["tar", "-x", "-C", tmp], input=archive, check=True)
        for ext in (REPO / "mortie").glob("_rustie*"):
            shutil.copy2(ext, pathlib.Path(tmp) / "mortie" / ext.name)
        out = subprocess.run(
            [sys.executable, "-c",
             "import json, mortie; "
             "print(json.dumps([mortie.__file__, sorted(set(mortie.__all__))]))"],
            cwd=tmp, check=True, capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": tmp},
        ).stdout
        where, names = json.loads(out.strip().splitlines()[-1])
        # realpath both: on macOS the temp dir is reached via a /var -> /private/var
        # symlink, so the raw prefix compare would reject a correct import.
        if not os.path.realpath(where).startswith(os.path.realpath(tmp)):
            raise RuntimeError(
                f"the {base} import resolved to {where}, not the extracted tree")
        return names


def check_public_surface(base):
    """Pin ``mortie.__all__`` and the resolvability of every name in it.

    A definition lost in the move usually breaks ``mortie/__init__.py``'s
    re-export, so the import below is exactly what fails first.  It is caught
    and turned into a failure entry rather than allowed to propagate: the
    diagnosis a reader wants is ``check_moves``'s "X landed in none of ...",
    not an import traceback that discards it.  An unresolvable ``base`` is
    handled the same way rather than as a ``CalledProcessError``.

    Parameters
    ----------
    base : str
        Git revision to compare the working tree against.

    Returns
    -------
    list of str
        One message per failure; empty when the surface is unchanged.
    """
    try:
        import mortie
    except ImportError as exc:
        print("__all__: NOT CHECKED — the working tree does not import")
        return [f"importing the working-tree package failed: {exc}"]

    failures = []
    if not pathlib.Path(mortie.__file__).is_relative_to(REPO):
        failures.append(
            f"the working-tree import resolved to {mortie.__file__}, outside {REPO}")
        return failures

    try:
        old = set(base_public_surface(base))
    except subprocess.CalledProcessError:
        print(f"__all__: NOT CHECKED — {base} is not reachable in this clone")
        return [unreachable(base, "the package tree")]

    new = set(mortie.__all__)
    for name in sorted(old - new):
        failures.append(f"__all__: {name} was dropped")
    for name in sorted(new - old):
        failures.append(f"__all__: {name} was added")
    for name in sorted(new):
        if not hasattr(mortie, name):
            failures.append(f"mortie.{name} does not resolve")
    print(f"__all__: {len(new)} names, all resolvable, equal to {base}'s"
          if not failures else "__all__: MISMATCH")
    return failures


def main():
    """Run every check and exit non-zero on the first failing claim.

    Returns
    -------
    int
        Process exit status: 0 when the split is a pure move, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/main",
                        help="git revision holding the pre-split source. Partial "
                             "once SPLIT_BASES is non-empty: a pinned split "
                             "compares its own move against its pin, and only "
                             "the pin itself against this revision")
    args = parser.parse_args()

    # bound separately so the move findings are collected — and reported —
    # even when a later check cannot run
    failures = check_moves(args.base)
    failures += check_pinned_bases(args.base)
    failures += check_public_surface(args.base)
    if failures:
        print(f"\n{len(failures)} failure(s):", file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        return 1
    print("\nPure move verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

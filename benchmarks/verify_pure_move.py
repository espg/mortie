"""Verify that a module split moved definitions verbatim (issue #159).

The domain split of ``mortie/tools.py`` and ``mortie/geometry.py`` claims to be
**pure moves plus import rewiring** — every top-level definition lands in its
new module byte-for-byte, and no public name changes.  That claim was checked by
hand for the first slice of the same plan (`PR #160
<https://github.com/espg/mortie/pull/160>`_ extracted ``moc.py`` out of
``coverage.py``; its review AST-compared all 11 moved definitions against their
pre-move originals).  This script is that check made re-runnable, so the
reviewer — and the next split — does not have to re-derive it.

Three claims are checked against a git base (``origin/main`` by default):

1. **Verbatim.**  Every top-level definition in a destination module that also
   exists in the source module at the base compares equal, both as an AST
   (``ast.dump``, so formatting and line numbers are ignored) and as literal
   source text (so comments *inside* a definition are covered too).
2. **Complete.**  Every top-level definition the source module had at the base
   lands in exactly one destination module — nothing silently lost or
   duplicated — and no destination gains a definition that was not there
   before.  A top-level statement the scanner cannot name and compare (a
   tuple-target assignment, an ``if TYPE_CHECKING:`` block, a ``try/except
   ImportError`` shim, a loop, an ``__all__ +=``) is reported as a failure
   rather than skipped, so this arm fails loud instead of open — see
   ``top_level_defs``.
3. **Public surface pinned.**  ``set(mortie.__all__)`` equals the base's, and
   every name in it still resolves as an attribute of ``mortie``.  The base
   package is extracted with ``git archive`` and imported in a subprocess (with
   the built ``_rustie`` extension copied in), so this compares two real
   imports rather than two guesses at what ``__init__.py`` evaluates to.

A split cut from a tree an earlier split already touched pins its own base in
``SPLIT_BASES`` rather than using ``--base``, so each arm stays a strict
verbatim check of its own move.  The pin is not trusted on its word: a fourth
arm indexes the pinned source at ``--base`` too and requires every definition
to be equal once body-level ``Import``/``ImportFrom`` statements are dropped.
Rewiring those imports is exactly what an earlier split legitimately does to a
module a later split then moves out of; without this arm a change made *in*
that earlier split sits in both the pinned base and the destination, and no arm
can see it — see ``check_pinned_bases``.

Run::

    python benchmarks/verify_pure_move.py [--base origin/main]

Exit status is non-zero if any claim fails.

Known limitations — a green run here is half the gate, not the whole one:

* A comment block sitting *between* two top-level definitions belongs to no
  definition's source segment, so it is not compared.  Comments inside a
  definition body are.
* Only the **moves** are checked, not the *import rewiring* that goes with
  them.  Deleting ``geometry.py``'s ``from .dissolve import
  _dissolved_polygons`` leaves all 76 definitions verbatim and all 69
  ``__all__`` names resolvable, so this script still exits 0 — ``pytest`` is
  what catches it (``test_emit_dissolve_is_the_default`` raises ``NameError:
  name '_dissolved_polygons' is not defined``).  Run both.
* A **test** module split is not modelled (phase 3 cut ``test_tools.py`` into
  ``test_convert.py`` / ``test_orders.py``), and *neither* arm can take it.
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

# Each entry: source module at the base -> the modules its definitions moved
# into.  A destination that is also the source (``geometry.py``) simply means
# part of it stayed put.
SPLITS = {
    "mortie/tools.py": [
        "mortie/convert.py",
        "mortie/orders.py",
        "mortie/buffer.py",
    ],
    "mortie/geometry.py": [
        "mortie/geometry.py",
        "mortie/dissolve.py",
        "mortie/codec.py",
    ],
}

# A split whose source was already touched by an *earlier* split verifies
# against the commit it was actually cut from, not against ``--base``.  Phase 2
# cut ``dissolve.py`` out of a ``geometry.py`` that phase 1 had already
# import-rewired (three function-local ``from .tools import`` lines became
# ``from .convert`` / ``from .orders``), so comparing it to ``origin/main``
# would report those three as differences and mask any real one.  Pinning the
# base per split keeps every arm a strict verbatim check of its own move, and
# ``check_pinned_bases`` checks the pin itself against ``--base``.
#
# CAVEAT: these are *branch* commits.  A squash-merge (or a rebase) rewrites
# them and the sha stops resolving — the run then reports "not reachable in this
# clone" rather than passing.  Repoint the entry at the squashed commit, or
# delete the entry and the split's ``SPLITS`` arm once the move has landed and
# the check has served its purpose.
#
# ===> MERGE ISSUE #159 WITH A MERGE COMMIT — NOT SQUASH, NOT REBASE. <===
# ``011816ca`` is an *ancestor* of that PR's head, so an ordinary merge commit
# keeps it reachable from ``main`` and this arm goes on working after the split
# lands.  Squash or rebase orphans it — not silently, since the failure above is
# loud, but the check stops being usable.  If it is squashed anyway, retire this
# entry and the ``mortie/geometry.py`` arm of ``SPLITS``: that is the documented
# end of their life.  The *file* stays either way — issue #170 (``batch.py``) is
# another pure-move refactor that will want this tool.
SPLIT_BASES = {
    # phase 1's head, review folded
    "mortie/geometry.py": "011816ca3553c3743c3e92fc5300ed23c6b3a514",
}

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

    Parameters
    ----------
    source : str
        Python source text of one module.

    Returns
    -------
    dict
        Name -> ``(ast node, source text)`` for every top-level function,
        class, and simple-name assignment (annotated or not).
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
    body = tree.body
    if ast.get_docstring(tree) is not None:
        body = body[1:]
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if len(names) != len(targets):
                # a tuple/list/attribute/subscript target: no single name to key on
                unhandled.append(f"{type(node).__name__} at line {node.lineno}")
                continue
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        else:
            unhandled.append(f"{type(node).__name__} at line {node.lineno}")
            continue
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


def check_moves(default_base):
    """Compare every moved definition against its pre-move original.

    Parameters
    ----------
    default_base : str
        Git revision holding the pre-move source, for every split that
        ``SPLIT_BASES`` does not pin to one of its own.

    Returns
    -------
    list of str
        One message per failure; empty when every definition is verbatim,
        accounted for, and unduplicated.
    """
    failures = []
    for src_path, dst_paths in SPLITS.items():
        base = SPLIT_BASES.get(src_path, default_base)
        try:
            src = git_show(base, src_path)
        except subprocess.CalledProcessError:
            failures.append(unreachable(base, src_path))
            continue
        old, old_unhandled = top_level_defs(src)
        failures += [f"{base}:{src_path}: {what} is not comparable — extend "
                     "top_level_defs before trusting this run"
                     for what in old_unhandled]
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
                        f"in {base}:{src_path}")
                    continue
                if name in landed:
                    failures.append(
                        f"{name}: defined in both {landed[name]} and {dst_path}")
                    continue
                landed[name] = dst_path
                old_node, old_text = old[name]
                if ast.dump(node) != ast.dump(old_node):
                    failures.append(
                        f"{dst_path}: {name} differs from {base}:{src_path} (AST)")
                elif text != old_text:
                    failures.append(
                        f"{dst_path}: {name} differs from {base}:{src_path} "
                        "(source text — comments or formatting)")
        for name in old:
            if name not in landed:
                failures.append(
                    f"{base}:{src_path}: {name} landed in none of "
                    f"{', '.join(dst_paths)}")
        print(f"{src_path}@{label_of(base)}: {len(landed)}/{len(old)} definitions "
              f"accounted for across {', '.join(dst_paths)}")
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

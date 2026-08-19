"""The single-kernel-delegation pin shared by the object-layer tests.

:class:`~mortie.Moc` (issue #196) and :class:`~mortie.Toc` (issue #198) both
promise that every public method is a *single* delegation to a kernel
function.  This module is the falsifiable form of that promise: a **shape
whitelist** over each method's returned expression plus a **denied-node
sweep** over everything inside it.  The blessed shapes are

- a bare kernel call,
- a kernel call re-boxed by a wrapper (``Moc(...)`` / ``Toc(...)`` /
  ``cls(...)``),
- the emptiness comparison ``<kernel>(...).size <op> 0`` (which operators are
  blessed is per class -- ``==`` for ``Moc.contains`` / ``within``, ``>`` for
  ``Toc.overlaps``),
- ``np.array_equal(<kernel>(...), <coercer>(...))``, opt-in, for
  ``Toc.contains``'s canonical-form equality.

Operand coercion through the class's ``_words(...)`` helper is allowed
anywhere.  Anything else -- a comprehension, a slice, arithmetic, a branch on
values, a second kernel call -- is algebra the object promised not to have,
and each test file proves its pin rejects such bodies with synthetic
violations.
"""

import ast
from pathlib import Path

# Operators, not calls -- so the "exactly one kernel call" count cannot see
# them.  Filtering, reindexing or branching on word values is exactly what a
# delegation-only method must not do, so the node types are refused outright.
# The one comparison a body may contain is a blessed shape from the whitelist
# above, which is checked structurally and then excluded from this sweep.
DENIED_NODES = (
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Lambda,
    ast.IfExp, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Subscript, ast.Compare,
    ast.NamedExpr, ast.For, ast.While, ast.If, ast.Await,
)


def class_def(module, name):
    """The ``ast.ClassDef`` for *name* in *module*'s source file.

    Parameters
    ----------
    module : module
        The imported module whose source holds the class.
    name : str
        The class name to find.

    Returns
    -------
    ast.ClassDef
        The class definition node.
    """
    tree = ast.parse(Path(module.__file__).read_text())
    return next(n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == name)


def called_name(node):
    """The name a ``Call`` node invokes, or ``None`` for a non-call.

    Parameters
    ----------
    node : ast.AST
        Any AST node.

    Returns
    -------
    str or None
        The called name (``f`` for ``f(...)``, ``attr`` for ``x.attr(...)``),
        or ``None`` when *node* is not a call.
    """
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    return getattr(func, "attr", "<expr>")


def _blessed_size_compare(expr, size_ops):
    """The expression under a blessed ``<expr>.size <op> 0``, or ``None``.

    Parameters
    ----------
    expr : ast.Compare
        The comparison to check.
    size_ops : tuple of type
        The ``ast.cmpop`` types blessed for this class.

    Returns
    -------
    ast.AST or None
        The expression whose ``.size`` is compared, or ``None`` when the
        comparison is not of the blessed shape.
    """
    left = expr.left
    if (isinstance(left, ast.Attribute) and left.attr == "size"
            and len(expr.ops) == 1 and isinstance(expr.ops[0], size_ops)
            and len(expr.comparators) == 1
            and isinstance(expr.comparators[0], ast.Constant)
            and expr.comparators[0].value == 0):
        return left.value
    return None


def delegation_violation(method, *, kernels, wrappers, coercers,
                         size_ops=(ast.Eq,), array_equal=False):
    """Why *method* is not a single kernel delegation, or ``None`` if it is.

    The shape whitelist, not a call count: the returned expression must be a
    kernel call, that call re-boxed by a wrapper, a blessed
    ``<kernel>(...).size <op> 0`` emptiness test, or (opt-in) the canonical
    equality ``np.array_equal(<kernel>(...), <coercer>(...))`` -- with no
    denied operator node anywhere inside.

    Parameters
    ----------
    method : ast.FunctionDef
        The method definition to check.
    kernels : set of str
        The kernel functions a delegation may call, exactly once.
    wrappers : set of str
        Names that may re-box the kernel's answer (the class, ``cls``).
    coercers : set of str
        Operand-coercion helpers allowed anywhere in the body.
    size_ops : tuple of type, optional
        The ``ast.cmpop`` types blessed in the ``.size`` comparison
        (default: equality only).
    array_equal : bool, optional
        Whether ``np.array_equal(<kernel>(...), <coercer>(...))`` is a
        blessed return shape (default: no).

    Returns
    -------
    str or None
        The reason the body violates the invariant, or ``None`` when it
        holds.
    """
    body = [s for s in method.body
            if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return "is not a single return statement"
    expr = body[0].value
    inner, blessed, extra_allowed = expr, None, set()
    if isinstance(expr, ast.Compare):
        inner = _blessed_size_compare(expr, tuple(size_ops))
        if inner is None:
            return "compares something other than a blessed `<kernel>(...).size` against 0"
        blessed = expr
    elif (array_equal and isinstance(expr, ast.Call)
          and called_name(expr) == "array_equal"):
        if len(expr.args) != 2 or expr.keywords:
            return "calls array_equal with a shape other than (kernel call, coerced operand)"
        if called_name(expr.args[1]) not in coercers:
            return "compares the kernel's answer against something other than the coerced operand"
        inner, blessed = expr.args[0], expr
        extra_allowed = {"array_equal"}
    if called_name(inner) in wrappers:
        if len(inner.args) != 1 or inner.keywords:
            return "wraps more than a single kernel call"
        inner = inner.args[0]
    if called_name(inner) not in kernels:
        return f"returns {called_name(inner) or type(inner).__name__}, not a kernel call"

    calls = [called_name(n) for n in ast.walk(expr) if isinstance(n, ast.Call)]
    kernel_calls = [c for c in calls if c in kernels]
    if len(kernel_calls) != 1:
        return f"makes {len(kernel_calls)} kernel calls, not exactly one"
    extra = set(calls) - kernels - wrappers - coercers - extra_allowed
    if extra:
        return f"also calls {sorted(extra)}"
    for node in ast.walk(expr):
        if node is blessed:
            continue
        if isinstance(node, DENIED_NODES):
            return f"contains a {type(node).__name__} node"
    return None

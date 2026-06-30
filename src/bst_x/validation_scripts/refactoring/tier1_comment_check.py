"""Mechanical "is this really comment-only" gate for a comment-strip pass.

Proves a candidate edit changed only comments and docstrings, not executable
code. Method: parse before / after with ``ast``, strip the leading str-typed
docstring of every module / class / function body, and compare ``ast.dump``
(attribute-free, so comment-driven line shifts are invisible). Comments never
enter the AST at all. A stray code edit on a trailing-comment line, a removed
statement, or an edit to a non-leading / non-docstring string literal all
survive the strip and so FAIL.

Hardening for the three modules whose module ``__doc__`` is fed to argparse
and rendered as ``--help`` (so a docstring trim is a real user-facing change,
not inert): if the code is AST-equal but the help-bearing slice of ``__doc__``
changed, the result is NEEDS REVIEW (exit 2), not a silent pass. The caller
confirms the trim was intentional.

Doesn't replace per-file keep-list review or pytest; this is the mechanical
"no code moved" half. ``bst.py`` is gated separately (model bit-exact; this
script doesn't cover it).

Usage (from the repo root):
    python src/bst_x/validation_scripts/refactoring/tier1_comment_check.py \\
        <repo-relative-path>... [--ref HEAD]
    python src/bst_x/validation_scripts/refactoring/tier1_comment_check.py --selftest

Exit: 0 clean, 1 FAIL (code changed), 2 NEEDS REVIEW (--help docstring changed).

Originally landed as the Tier-1 gate in the simplification pass's comment-strip
batch (02).
"""
from __future__ import annotations

import ast
import subprocess
import sys

# basename -> slice of module __doc__ that reaches argparse --help.
DOC_FED = {
    "data_access.py": lambda d: d,                         # description=__doc__
    "build_dataset.py": lambda d: d,                       # description=__doc__
    "apply_heuristic.py": lambda d: d.split("\n\n")[0],    # __doc__.split("\n\n")[0]
}


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    """Remove the leading str-typed docstring from every module/class/function body.
    Leading-only and str-typed: a non-leading bare string or a leading numeric
    / bytes constant is NOT a docstring and must survive the strip."""
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:]
    return tree


def code_equal_ignoring_docstrings(before_src: str, after_src: str) -> bool:
    a = _strip_docstrings(ast.parse(before_src))
    b = _strip_docstrings(ast.parse(after_src))
    return ast.dump(a) == ast.dump(b)


def help_doc_changed(basename: str, before_src: str, after_src: str) -> bool:
    """For a __doc__-fed module, did the help-bearing slice of the module docstring
    change? Returns False for any non-__doc__-fed file."""
    slicer = DOC_FED.get(basename)
    if slicer is None:
        return False
    before_doc = ast.get_docstring(ast.parse(before_src)) or ""
    after_doc = ast.get_docstring(ast.parse(after_src)) or ""
    return slicer(before_doc) != slicer(after_doc)


def check_file(path: str, ref: str) -> int:
    before_src = subprocess.run(
        ["git", "show", f"{ref}:{path}"], capture_output=True, text=True
    ).stdout
    with open(path, encoding="utf-8") as fh:
        after_src = fh.read()
    if not code_equal_ignoring_docstrings(before_src, after_src):
        print(f"FAIL  {path}: executable code changed (not comment/docstring-only)")
        return 1
    basename = path.rsplit("/", 1)[-1]
    if help_doc_changed(basename, before_src, after_src):
        print(f"REVIEW {path}: module __doc__ feeds argparse --help and its help slice "
              f"changed; confirm the trim was intentional, not an accident")
        return 2
    print(f"OK    {path}: comment/docstring-only, no code change")
    return 0


def _selftest() -> int:
    base = (
        '"""Module doc.\n\nCLI: foo --bar\n"""\n'
        "import os  # trailing comment\n"
        "def f(x):\n"
        '    """Func doc."""\n'
        "    y = x + 1  # add one\n"
        "    return y\n"
        "BANNER = 'not a docstring'\n"
    )
    cases = [
        ("comment-only", base.replace("# trailing comment", "# reworded").replace(
            "# add one", ""), True),
        ("func-docstring trim", base.replace('"""Func doc."""', '"""Doc."""'), True),
        ("module-docstring trim", base.replace("CLI: foo --bar\n", ""), True),
        ("code edit on comment line", base.replace("y = x + 1", "y = x + 2"), False),
        ("statement removed", base.replace("    return y\n", ""), False),
        ("non-docstring string edit", base.replace("'not a docstring'", "'changed'"),
         False),
    ]
    ok = True
    for name, after, expect_equal in cases:
        got = code_equal_ignoring_docstrings(base, after)
        flag = "ok" if got == expect_equal else "WRONG"
        if got != expect_equal:
            ok = False
        print(f"  [{flag}] {name}: equal={got} (expected {expect_equal})")
    # __doc__-fed help-slice detection
    after = base.replace("CLI: foo --bar\n", "CLI: changed\n")
    detected = help_doc_changed("data_access.py", base, after)
    print(f"  [{'ok' if detected else 'WRONG'}] data_access help-slice change detected="
          f"{detected} (expected True)")
    ok = ok and detected
    apply_after = base.replace("Module doc.", "Module DOC.")  # first paragraph changed
    apply_detected = help_doc_changed("apply_heuristic.py", base, apply_after)
    print(f"  [{'ok' if apply_detected else 'WRONG'}] apply_heuristic first-para change "
          f"detected={apply_detected} (expected True)")
    return 0 if (ok and apply_detected) else 1


def main() -> int:
    args = sys.argv[1:]
    if args == ["--selftest"]:
        return _selftest()
    ref = "HEAD"
    if "--ref" in args:
        i = args.index("--ref")
        ref = args[i + 1]
        args = args[:i] + args[i + 2:]
    if not args:
        print("usage: tier1_comment_check.py <path>... [--ref HEAD] | --selftest")
        return 2
    return max(check_file(p, ref) for p in args)


if __name__ == "__main__":
    raise SystemExit(main())

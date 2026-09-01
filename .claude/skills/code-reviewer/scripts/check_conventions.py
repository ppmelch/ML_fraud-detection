#!/usr/bin/env python3
"""Check the mechanical conventions so reviewers can spend attention elsewhere.

Covers docstrings, type annotations, naming, hardcoded paths, secrets, print
statements, canonical vocabulary, and the causality markers that indicate a
feature reads forward.

The causality check is the one that matters most: `center=True` on a rolling
window or a positive `.shift()` on a feature turns a backtest optimistic in a
way no amount of downstream care can recover.

Usage:
    python check_conventions.py src/ scripts/
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import defaultdict
from pathlib import Path

# Vocabulary fixed in docs/CONTEXT.md. Terminology drift between code and
# documentation is a correctness concern here, not a style preference.
FORBIDDEN_TERMS: dict[str, str] = {
    "orderbook": "LOB",
    "order_book": "LOB",
    "limit_order_book": "LOB",
    "imbalance_ratio": "OFI",
    "order_imbalance": "OFI",
    "decomposition": "breakdown",
    "time_series_cv": "walk_forward_cv",
    "rolling_cv": "walk_forward_cv",
}

SECRET_PATTERNS = [
    (re.compile(r'(?i)(api[_-]?key|secret|password|token)\s*=\s*["\'][^"\']{12,}["\']'),
     "hardcoded credential"),
    (re.compile(r'(?i)bearer\s+[A-Za-z0-9\-._~+/]{20,}'), "bearer token"),
    (re.compile(r'sk-[A-Za-z0-9]{20,}'), "API key"),
]

ABSOLUTE_PATH_PATTERN = re.compile(r'["\'](?:[A-Za-z]:[\\/]|/(?:home|Users|mnt|var|opt)/)[^"\']*["\']')


class ConventionVisitor(ast.NodeVisitor):
    """Collect AST-level findings for one file."""

    def __init__(self, relpath: str, is_src: bool):
        self.relpath = relpath
        self.is_src = is_src
        self.findings: list[dict] = []

    def _add(self, kind: str, line: int, message: str, severity: str = "warning") -> None:
        self.findings.append({"kind": kind, "file": self.relpath, "line": line,
                              "message": message, "severity": severity})

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        public = not node.name.startswith("_")

        if public and not ast.get_docstring(node):
            self._add("missing_docstring", node.lineno,
                      f"public function '{node.name}' has no docstring")

        if public:
            untyped = [a.arg for a in node.args.args
                       if a.annotation is None and a.arg not in ("self", "cls")]
            if untyped:
                self._add("missing_annotation", node.lineno,
                          f"'{node.name}' parameters without type annotation: {untyped}")
            if node.returns is None:
                self._add("missing_annotation", node.lineno,
                          f"'{node.name}' has no return type annotation")

        if not re.fullmatch(r"_{0,2}[a-z][a-z0-9_]*", node.name):
            self._add("naming", node.lineno,
                      f"function '{node.name}' is not snake_case", severity="error")

        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if not node.name.startswith("_") and not ast.get_docstring(node):
            self._add("missing_docstring", node.lineno,
                      f"public class '{node.name}' has no docstring")

        if not re.fullmatch(r"_?[A-Z][A-Za-z0-9]*", node.name):
            self._add("naming", node.lineno,
                      f"class '{node.name}' is not PascalCase", severity="error")

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func_name = ""
        if isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id

        # print() in library code
        if func_name == "print" and self.is_src:
            self._add("print_in_src", node.lineno,
                      "print() in src/ — use the logging module", severity="error")

        # Causality: centered rolling windows read forward.
        if func_name == "rolling":
            for kw in node.keywords:
                if kw.arg == "center" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    self._add("causality", node.lineno,
                              "rolling(center=True) reads future rows — features must be trailing",
                              severity="error")

        # A positive shift on what looks like a feature moves past data forward,
        # which is the inverse of the intended target shift.
        if func_name == "shift" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, int) and arg.value > 0:
                self._add("causality", node.lineno,
                          f"shift({arg.value}) is positive; targets use shift(-h). "
                          "Verify this is intentional",
                          severity="warning")

        self.generic_visit(node)


def check_text_patterns(path: Path, relpath: str, text: str) -> list[dict]:
    """Line-level checks that do not need an AST."""
    findings: list[dict] = []

    for line_no, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        for pattern, label in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append({"kind": "secret", "file": relpath, "line": line_no,
                                 "message": f"possible {label} in source — move it to .env",
                                 "severity": "error"})

        if ABSOLUTE_PATH_PATTERN.search(line):
            findings.append({"kind": "hardcoded_path", "file": relpath, "line": line_no,
                             "message": "absolute path in source — read it from .env or use a relative path",
                             "severity": "error"})

        lowered = line.lower()
        for term, replacement in FORBIDDEN_TERMS.items():
            if re.search(rf"\b{re.escape(term)}\b", lowered):
                findings.append({"kind": "vocabulary", "file": relpath, "line": line_no,
                                 "message": f"'{term}' is not canonical; docs/CONTEXT.md uses '{replacement}'",
                                 "severity": "warning"})

    return findings


def check_file(path: Path, repo_root: Path) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    try:
        rel = str(path.relative_to(repo_root))
    except ValueError:
        rel = str(path)

    findings = check_text_patterns(path, rel, text)

    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        findings.append({"kind": "syntax", "file": rel, "line": exc.lineno or 0,
                         "message": f"syntax error: {exc.msg}", "severity": "error"})
        return findings

    is_src = rel.replace("\\", "/").startswith("src/")
    visitor = ConventionVisitor(rel, is_src)
    visitor.visit(tree)
    findings.extend(visitor.findings)

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--strict", action="store_true", help="warnings also fail the run")
    args = parser.parse_args()

    files: list[Path] = []
    for p in args.paths:
        if p.is_dir():
            files.extend(sorted(p.rglob("*.py")))
        elif p.suffix == ".py":
            files.append(p)
    files = [f for f in files if "__pycache__" not in str(f)]

    if not files:
        print("No Python files found.", file=sys.stderr)
        return 2

    all_findings: list[dict] = []
    for f in files:
        all_findings.extend(check_file(f, args.repo_root))

    print("=" * 74)
    print("CONVENTION CHECK")
    print("=" * 74)
    print(f"Files scanned: {len(files)}")
    print()

    if not all_findings:
        print("No convention violations.")
        print()
        print("=" * 74)
        print("RESULT: clean")
        print("=" * 74)
        return 0

    by_kind: dict[str, list[dict]] = defaultdict(list)
    for f in all_findings:
        by_kind[f["kind"]].append(f)

    # Causality and secrets first — they are the expensive ones.
    order = ["causality", "secret", "hardcoded_path", "print_in_src", "naming",
             "vocabulary", "missing_docstring", "missing_annotation", "syntax"]

    for kind in order:
        group = by_kind.get(kind, [])
        if not group:
            continue
        sev = group[0]["severity"].upper()
        print(f"--- {kind.replace('_', ' ').upper()} [{sev}] ({len(group)}) ---")
        for f in group[:20]:
            print(f"  {f['file']}:{f['line']}")
            print(f"    {f['message']}")
        if len(group) > 20:
            print(f"  ... and {len(group) - 20} more")
        print()

    n_error = sum(1 for f in all_findings if f["severity"] == "error")
    n_warning = len(all_findings) - n_error

    print("=" * 74)
    print(f"RESULT: {n_error} error(s), {n_warning} warning(s)")
    print("=" * 74)

    if by_kind.get("causality"):
        print()
        print("Causality findings are the expensive ones: a feature that reads")
        print("forward makes every accuracy number optimistic, and nothing")
        print("downstream can recover it. Verify each one before merging.")

    if args.strict:
        return 1 if all_findings else 0
    return 1 if n_error else 0


if __name__ == "__main__":
    sys.exit(main())

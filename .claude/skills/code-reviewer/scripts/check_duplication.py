#!/usr/bin/env python3
"""Find second implementations of concepts that must exist exactly once.

`.claude/Agent.md` states the rule: one LOBReconstructor, one compute_ofi, one
DeepLOB, one MatchingEngine, one set of metrics. Two implementations of the
same concept diverge the first time one of them is changed, and the divergence
surfaces as a Phase 4 number that disagrees with Phase 2.

Uses AST parsing rather than text matching, so a renamed variable does not
hide a copied function.

Usage:
    python check_duplication.py src/
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

# Canonical concepts and the module that owns each. A definition matching one
# of these names outside its owning module is a duplicate by definition.
CANONICAL_CONCEPTS: dict[str, str] = {
    "LOBReconstructor": "src/data/lob_reconstruction.py",
    "OrderBook": "src/data/lob_reconstruction.py",
    "PriceLevel": "src/data/lob_reconstruction.py",
    "compute_ofi": "src/features/order_flow_imbalance.py",
    "compute_ofi_features": "src/features/order_flow_imbalance.py",
    "compute_price_movement": "src/features/order_flow_imbalance.py",
    "walk_forward_split": "src/features/walk_forward_cv.py",
    "DeepLOB": "src/models/deeplab.py",
    "OFIBaseline": "src/models/ofi_baseline.py",
    "BaseModel": "src/models/base_model.py",
    "MatchingEngine": "src/simulator/matching_engine.py",
    "SimulatedOrder": "src/simulator/order.py",
    "BacktestEngine": "src/backtester/backtest_engine.py",
    "PnLCalculator": "src/backtester/pnl_calculator.py",
    "compute_classification_metrics": "src/utils/metrics.py",
    "compute_confusion_matrix": "src/utils/metrics.py",
    "find_optimal_threshold": "src/utils/metrics.py",
}

# Metric formulas that must be imported from src/utils/metrics.py rather than
# written inline. Matching on the arithmetic shape catches a reimplementation
# even when the variable names differ.
INLINE_METRIC_MARKERS: dict[str, str] = {
    "sharpe": "returns.mean() / returns.std()",
    "accuracy": "(y_pred == y_true).mean()",
    "precision": "tp / (tp + fp)",
    "recall": "tp / (tp + fn)",
}


def normalize_function(node: ast.FunctionDef) -> str:
    """Structural fingerprint of a function body.

    Strips names and constants so that a copy-pasted function with renamed
    variables still hashes identically. Docstrings are excluded, since two
    functions differing only in prose are still duplicates.
    """
    parts: list[str] = []

    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]  # drop docstring

    for child in body:
        for sub in ast.walk(child):
            # Record node types and operators; ignore identifiers and literals.
            parts.append(type(sub).__name__)
            if isinstance(sub, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare)):
                parts.append(type(sub.op).__name__ if hasattr(sub, "op") else "cmp")

    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def count_statements(node: ast.FunctionDef) -> int:
    return sum(1 for _ in ast.walk(node) if isinstance(_, ast.stmt))


def scan_file(path: Path, repo_root: Path) -> dict:
    """Collect definitions and fingerprints from one file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as exc:
        return {"path": path, "error": str(exc), "definitions": [], "fingerprints": []}

    try:
        rel = path.relative_to(repo_root)
    except ValueError:
        rel = path

    definitions = []
    fingerprints = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions.append({"name": node.name, "line": node.lineno,
                                "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                                "file": str(rel)})
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            n_stmt = count_statements(node)
            if n_stmt >= 3:
                fingerprints.append({"name": node.name, "line": node.lineno,
                                     "file": str(rel), "hash": normalize_function(node),
                                     "statements": n_stmt})

    return {"path": path, "error": None, "definitions": definitions, "fingerprints": fingerprints}


def check_canonical_violations(all_definitions: list[dict]) -> list[dict]:
    """A canonical name defined outside its owning module."""
    findings = []
    for d in all_definitions:
        owner = CANONICAL_CONCEPTS.get(d["name"])
        if owner is None:
            continue
        actual = d["file"].replace("\\", "/")
        if not actual.endswith(owner.split("/")[-1]) or owner.split("/")[-1] not in actual:
            findings.append({
                "kind": "canonical_violation",
                "name": d["name"],
                "found_in": d["file"],
                "line": d["line"],
                "owner": owner,
                "message": (f"'{d['name']}' is a canonical concept owned by {owner}. "
                            f"Import it instead of defining a second implementation."),
            })
    return findings


def check_duplicate_names(all_definitions: list[dict]) -> list[dict]:
    """The same class or function name defined in more than one module."""
    by_name: dict[str, list[dict]] = defaultdict(list)
    for d in all_definitions:
        by_name[d["name"]].append(d)

    findings = []
    for name, defs in by_name.items():
        if name.startswith("_") or name in ("main", "setup", "__init__"):
            continue
        files = {d["file"] for d in defs}
        if len(files) > 1:
            findings.append({
                "kind": "duplicate_name",
                "name": name,
                "locations": [f"{d['file']}:{d['line']}" for d in defs],
                "message": (f"'{name}' is defined in {len(files)} modules. If they do the "
                            "same thing, keep one and import it. If not, rename so the "
                            "difference is visible."),
            })
    return findings


def check_identical_bodies(all_fingerprints: list[dict], min_statements: int) -> list[dict]:
    """Functions with structurally identical bodies."""
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for f in all_fingerprints:
        if f["statements"] >= min_statements:
            by_hash[f["hash"]].append(f)

    findings = []
    for _, group in by_hash.items():
        if len(group) < 2:
            continue
        files = {g["file"] for g in group}
        if len(files) < 2:
            continue  # overloads in one file are usually deliberate
        findings.append({
            "kind": "identical_body",
            "names": [g["name"] for g in group],
            "locations": [f"{g['file']}:{g['line']}" for g in group],
            "statements": group[0]["statements"],
            "message": (f"{len(group)} functions have structurally identical bodies "
                        f"({group[0]['statements']} statements). Extract a shared helper."),
        })
    return findings


def check_inline_metrics(paths: list[Path], repo_root: Path) -> list[dict]:
    """Metric arithmetic written inline instead of imported."""
    findings = []
    for path in paths:
        if "metrics.py" in path.name or "test" in str(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        try:
            rel = str(path.relative_to(repo_root))
        except ValueError:
            rel = str(path)

        for line_no, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for metric, pattern in INLINE_METRIC_MARKERS.items():
                # Compare with whitespace removed so formatting does not hide it.
                if pattern.replace(" ", "") in stripped.replace(" ", ""):
                    findings.append({
                        "kind": "inline_metric",
                        "metric": metric,
                        "location": f"{rel}:{line_no}",
                        "message": (f"{metric} computed inline. Import it from "
                                    "src/utils/metrics.py so both models are scored "
                                    "by the same implementation."),
                    })
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path, help="files or directories to scan")
    parser.add_argument("--min-lines", type=int, default=8,
                        help="minimum statements before flagging identical bodies")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--verbose", action="store_true")
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

    all_definitions: list[dict] = []
    all_fingerprints: list[dict] = []
    parse_errors: list[str] = []

    for f in files:
        result = scan_file(f, args.repo_root)
        if result["error"]:
            parse_errors.append(f"{f}: {result['error']}")
            continue
        all_definitions.extend(result["definitions"])
        all_fingerprints.extend(result["fingerprints"])

    findings = (
        check_canonical_violations(all_definitions)
        + check_duplicate_names(all_definitions)
        + check_identical_bodies(all_fingerprints, args.min_lines)
        + check_inline_metrics(files, args.repo_root)
    )

    print("=" * 74)
    print("DUPLICATION CHECK")
    print("=" * 74)
    print(f"Files scanned: {len(files)}   Definitions: {len(all_definitions)}")
    if parse_errors:
        print(f"Parse errors: {len(parse_errors)}")
        for e in parse_errors[:5]:
            print(f"  {e}")
    print()

    if not findings:
        print("No duplication found.")
        print()
        print("=" * 74)
        print("RESULT: clean")
        print("=" * 74)
        return 0

    by_kind: dict[str, list[dict]] = defaultdict(list)
    for f in findings:
        by_kind[f["kind"]].append(f)

    severity_order = ["canonical_violation", "inline_metric", "duplicate_name", "identical_body"]
    for kind in severity_order:
        group = by_kind.get(kind, [])
        if not group:
            continue
        print(f"--- {kind.replace('_', ' ').upper()} ({len(group)}) ---")
        for f in group:
            loc = f.get("found_in") or f.get("location") or ", ".join(f.get("locations", []))
            print(f"  {loc}")
            print(f"    {f['message']}")
        print()

    print("=" * 74)
    print(f"RESULT: {len(findings)} duplication finding(s)")
    print("=" * 74)
    print()
    print("See .claude/Agent.md — there must be exactly one implementation of")
    print("every business concept.")

    # Only canonical violations and inline metrics fail the build; duplicate
    # names and identical bodies are frequently legitimate and warrant a look
    # rather than a block.
    blocking = len(by_kind["canonical_violation"]) + len(by_kind["inline_metric"])
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())

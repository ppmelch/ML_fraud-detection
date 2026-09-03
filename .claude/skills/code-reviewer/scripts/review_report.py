#!/usr/bin/env python3
"""Produce a structured review of a diff.

Combines the duplication and convention checks with diff-specific analysis:
whether new source code has tests, whether a new domain term reached the code
without reaching the glossary, and whether a pipeline script writes output
without validating it first.

What it cannot judge is whether the abstraction is right or whether a test
would actually fail if the code were wrong. Those need a person reading the
diff; this report is what frees attention for them.

Usage:
    python review_report.py --base main --head HEAD --output review.md
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SKILL_SCRIPTS = Path(__file__).resolve().parent


def git(*args: str, cwd: Path) -> str:
    """Run a git command, returning stdout."""
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def changed_files(base: str, head: str, cwd: Path) -> list[dict]:
    """Files changed between two refs, with their status."""
    out = git("diff", "--name-status", f"{base}...{head}", cwd=cwd)
    files = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status, path = parts[0], parts[-1]
        files.append({"status": status, "path": path})
    return files


def diff_stats(base: str, head: str, cwd: Path) -> dict:
    out = git("diff", "--shortstat", f"{base}...{head}", cwd=cwd)
    m_files = re.search(r"(\d+) files? changed", out)
    m_ins = re.search(r"(\d+) insertions?", out)
    m_del = re.search(r"(\d+) deletions?", out)
    return {
        "files_changed": int(m_files.group(1)) if m_files else 0,
        "insertions": int(m_ins.group(1)) if m_ins else 0,
        "deletions": int(m_del.group(1)) if m_del else 0,
    }


def run_check(script: str, paths: list[str], cwd: Path) -> dict:
    """Run one of the sibling check scripts on a set of paths."""
    if not paths:
        return {"ran": False, "passed": True, "output": "", "reason": "no matching files changed"}

    result = subprocess.run(
        [sys.executable, str(SKILL_SCRIPTS / script), *paths, "--repo-root", str(cwd)],
        cwd=cwd, capture_output=True, text=True,
    )
    return {"ran": True, "passed": result.returncode == 0,
            "output": result.stdout, "returncode": result.returncode}


def check_tests_for_new_source(files: list[dict], cwd: Path) -> list[str]:
    """New modules under src/ should come with tests."""
    findings = []
    changed = {f["path"] for f in files}

    for f in files:
        path = f["path"]
        if not path.startswith("src/") or not path.endswith(".py"):
            continue
        if path.endswith("__init__.py"):
            continue
        if f["status"] not in ("A", "M"):
            continue

        stem = Path(path).stem
        expected = f"test_{stem}.py"
        has_test = any(expected in c for c in changed) or \
                   any((cwd / d / expected).exists() for d in ("tests/unit", "tests/integration"))

        if not has_test:
            findings.append(f"{path} has no corresponding {expected}")

    return findings


def check_context_updated(files: list[dict], base: str, head: str, cwd: Path) -> list[str]:
    """A new canonical-looking term in code should reach docs/CONTEXT.md.

    Heuristic and deliberately loose: it flags candidates for a human to judge
    rather than trying to decide what counts as a domain term.
    """
    changed = {f["path"] for f in files}
    context_touched = any("CONTEXT.md" in p for p in changed)
    if context_touched:
        return []

    src_changed = [f["path"] for f in files
                   if f["path"].startswith("src/") and f["path"].endswith(".py")
                   and f["status"] == "A"]
    if not src_changed:
        return []

    return [f"new module(s) added without touching docs/CONTEXT.md: {', '.join(src_changed[:5])}"
            " — confirm no new domain term was introduced"]


def check_validation_gate(files: list[dict], cwd: Path) -> list[str]:
    """A pipeline script that writes output must validate first."""
    findings = []
    for f in files:
        path = f["path"]
        if not path.startswith("scripts/") or not path.endswith(".py"):
            continue
        full = cwd / path
        if not full.exists():
            continue

        text = full.read_text(encoding="utf-8", errors="ignore")
        writes = bool(re.search(r"\.to_csv\(|\.to_parquet\(|\.write_text\(|json\.dump\(", text))
        validates = bool(re.search(r"validate_|_validation|sys\.exit\(1\)", text))

        if writes and not validates:
            findings.append(f"{path} writes output but shows no validation gate "
                            "— see data-engineer/references/provenance_chain.md")
    return findings


def build_report(base: str, head: str, cwd: Path) -> str:
    files = changed_files(base, head, cwd)
    stats = diff_stats(base, head, cwd)

    py_files = [f["path"] for f in files
                if f["path"].endswith(".py") and f["status"] != "D"
                and (cwd / f["path"]).exists()]
    src_files = [p for p in py_files if p.startswith(("src/", "scripts/"))]

    dup = run_check("check_duplication.py", src_files, cwd)
    conv = run_check("check_conventions.py", py_files, cwd)

    missing_tests = check_tests_for_new_source(files, cwd)
    context_findings = check_context_updated(files, base, head, cwd)
    validation_findings = check_validation_gate(files, cwd)

    lines: list[str] = []
    add = lines.append

    add(f"# Code Review — `{base}...{head}`")
    add("")
    add(f"{stats['files_changed']} file(s) changed, "
        f"+{stats['insertions']} / -{stats['deletions']}")
    add("")

    add("## Changed files")
    add("")
    if files:
        for f in files:
            label = {"A": "added", "M": "modified", "D": "deleted"}.get(f["status"], f["status"])
            add(f"- `{f['path']}` ({label})")
    else:
        add("_No changes._")
    add("")

    add("## Architecture")
    add("")
    if not dup["ran"]:
        add(f"_Skipped: {dup['reason']}._")
    elif dup["passed"]:
        add("No duplication detected.")
    else:
        add("**Duplication findings — resolve before merging.**")
        add("")
        add("```")
        add(dup["output"].strip())
        add("```")
    add("")

    add("## Conventions")
    add("")
    if not conv["ran"]:
        add(f"_Skipped: {conv['reason']}._")
    elif conv["passed"]:
        add("No blocking convention violations.")
    else:
        add("**Convention violations.** Causality findings are blocking.")
        add("")
        add("```")
        add(conv["output"].strip())
        add("```")
    add("")

    add("## Tests")
    add("")
    if missing_tests:
        for m in missing_tests:
            add(f"- {m}")
    else:
        add("Every changed source module has a corresponding test file.")
    add("")

    add("## Pipeline discipline")
    add("")
    if validation_findings:
        for v in validation_findings:
            add(f"- {v}")
    else:
        add("No pipeline script writes output without a validation gate.")
    add("")

    add("## Documentation")
    add("")
    if context_findings:
        for c in context_findings:
            add(f"- {c}")
    else:
        add("No documentation gaps detected.")
    add("")

    add("## Reviewer checklist")
    add("")
    add("Tooling cannot judge these. Read the diff for them.")
    add("")
    add("- [ ] Is the abstraction right, or does it fit only this caller?")
    add("- [ ] Would each new test fail if the code were wrong?")
    add("- [ ] Are float comparisons using `pytest.approx`?")
    add("- [ ] Are invariants asserted, not only the happy path?")
    add("- [ ] Was any threshold or hyperparameter chosen on test-fold data?")
    add("- [ ] Do new artifacts record `upstream_artifacts` in metadata?")
    add("- [ ] Is the repository cleaner after this change than before?")
    add("")

    blocking = (dup["ran"] and not dup["passed"]) or (conv["ran"] and not conv["passed"])
    add("---")
    add("")
    add(f"**Automated verdict:** {'changes requested' if blocking else 'no blocking findings'}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="main", help="base ref")
    parser.add_argument("--head", default="HEAD", help="head ref")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, help="write the report to a file")
    args = parser.parse_args()

    try:
        report = build_report(args.base, args.head, args.repo_root)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"\n[report written to {args.output}]", file=sys.stderr)

    return 1 if "changes requested" in report else 0


if __name__ == "__main__":
    sys.exit(main())

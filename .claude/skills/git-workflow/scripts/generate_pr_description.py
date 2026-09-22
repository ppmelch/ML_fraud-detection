#!/usr/bin/env python3
"""Build a PR description from the issue file and the diff.

Copies the issue's acceptance criteria into the PR body so the reviewer checks
what the issue actually asked for, rather than what the diff happens to
contain.

Usage:
    python generate_pr_description.py --issue 0007 --output pr.md
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]

BRANCH_ISSUE_RE = re.compile(r"(\d{4})")

# Map a changed path to the PR template's change-type checkboxes.
AREA_TO_TYPE = {
    "src/data": "Data pipeline / features",
    "src/features": "Data pipeline / features",
    "src/models": "Model / training",
    "src/simulator": "Simulator / execution",
    "src/backtester": "Backtest / P&L",
    "tests": "Tests / CI",
    ".github": "Tests / CI",
    "docs": "Docs / README",
    ".claude": "Docs / README",
}

ALL_TYPES = [
    "Data pipeline / features",
    "Model / training",
    "Simulator / execution",
    "Backtest / P&L",
    "Tests / CI",
    "Docs / README",
]


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        return ""
    return r.stdout.strip()


def resolve_issue_number(explicit: str | None) -> str | None:
    if explicit:
        return explicit.zfill(4)
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    m = BRANCH_ISSUE_RE.search(branch)
    return m.group(1) if m else None


def find_issue_file(issue: str) -> Path | None:
    matches = sorted((REPO_ROOT / "docs" / "issues").glob(f"phase-*/{issue}-*.md"))
    return matches[0] if matches else None


def parse_issue(path: Path) -> dict:
    """Pull the title, metadata line, and acceptance criteria out of an issue."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    title = ""
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break

    meta = {}
    for line in lines[:12]:
        if line.startswith(">"):
            for key in ("Type", "Epic", "Estimated"):
                m = re.search(rf"\*\*{key}\*\*:\s*([^|\n]+)", line)
                if m:
                    meta[key.lower()] = m.group(1).strip()

    # Acceptance criteria: every checkbox under that heading, until the next H2.
    criteria: list[str] = []
    in_section = False
    for line in lines:
        if re.match(r"^##\s+Acceptance Criteria", line, re.IGNORECASE):
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if in_section:
            m = re.match(r"^\s*-\s*\[[ x]\]\s*(.+)$", line)
            if m:
                criteria.append(m.group(1).strip())

    return {"title": title, "meta": meta, "criteria": criteria, "path": path}


def changed_files(base: str) -> list[dict]:
    out = git("diff", "--name-status", f"{base}...HEAD")
    files = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            files.append({"status": parts[0], "path": parts[-1]})
    return files


def infer_types(files: list[dict]) -> set[str]:
    types = set()
    for f in files:
        for prefix, label in AREA_TO_TYPE.items():
            if f["path"].startswith(prefix):
                types.add(label)
    return types


def group_by_area(files: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for f in files:
        parts = f["path"].split("/")
        area = "/".join(parts[:2]) if len(parts) > 1 else parts[0]
        groups[area].append(f)
    return dict(sorted(groups.items()))


def commit_subjects(base: str) -> list[str]:
    out = git("log", f"{base}..HEAD", "--format=%s")
    return [l for l in out.splitlines() if l.strip()]


def build(issue_num: str | None, base: str) -> str:
    files = changed_files(base)
    commits = commit_subjects(base)
    types = infer_types(files)

    issue = None
    if issue_num:
        path = find_issue_file(issue_num)
        if path:
            issue = parse_issue(path)

    out: list[str] = []
    add = out.append

    add("## Summary")
    add("")
    if issue:
        add(f"{issue['title']}")
        if issue["meta"]:
            bits = [f"{k}: {v}" for k, v in issue["meta"].items()]
            add("")
            add(f"_{' | '.join(bits)}_")
    else:
        add("<!-- Describe what this PR does in 1-2 sentences -->")
    add("")

    add("## Type of Change")
    add("")
    for t in ALL_TYPES:
        add(f"- [{'x' if t in types else ' '}] {t}")
    add("")

    add("## Changes")
    add("")
    if commits:
        for c in commits:
            add(f"- {c}")
    else:
        add("<!-- No commits found in range -->")
    add("")

    if files:
        add("<details>")
        add("<summary>Files changed</summary>")
        add("")
        for area, group in group_by_area(files).items():
            add(f"**{area}/**")
            for f in group:
                label = {"A": "added", "M": "modified", "D": "deleted"}.get(f["status"], f["status"])
                add(f"- `{f['path']}` ({label})")
            add("")
        add("</details>")
        add("")

    add("## Testing")
    add("")
    add("- [ ] Unit tests added/updated")
    add("- [ ] Integration tests passed")
    add("- [ ] Manual testing (describe)")
    add("")

    if issue and issue["criteria"]:
        add("## Acceptance Criteria")
        add("")
        add(f"From [`{issue['path'].relative_to(REPO_ROOT)}`]"
            f"(../blob/main/{issue['path'].relative_to(REPO_ROOT).as_posix()}) — "
            "check each one that this PR satisfies.")
        add("")
        for c in issue["criteria"]:
            add(f"- [ ] {c}")
        add("")

    add("## Checklist")
    add("")
    add("- [ ] Code follows the style guide (black, flake8, ruff)")
    add("- [ ] All tests pass locally (`make test`)")
    add("- [ ] Docstrings added/updated for new functions and classes")
    add("- [ ] README updated if needed")
    add("- [ ] No hardcoded paths or secrets")
    add("- [ ] `docs/CONTEXT.md` updated with any new domain term")
    add("- [ ] New artifacts record `upstream_artifacts` in metadata")
    add("")

    add("## Related Issues")
    add("")
    if issue_num:
        add(f"Closes #{int(issue_num)}")
    else:
        add("<!-- Closes #NNNN -->")
    add("")

    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--issue", help="issue number; inferred from the branch name when omitted")
    parser.add_argument("--base", default="main")
    parser.add_argument("--output", type=Path, help="write the description to a file")
    args = parser.parse_args()

    issue_num = resolve_issue_number(args.issue)
    if issue_num and not find_issue_file(issue_num):
        print(f"WARNING: no issue file found for {issue_num}; "
              "the description will omit acceptance criteria", file=sys.stderr)

    body = build(issue_num, args.base)
    print(body)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body)
        print(f"\n[written to {args.output}]", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

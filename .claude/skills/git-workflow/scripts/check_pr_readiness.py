#!/usr/bin/env python3
"""Run the full quality gate before a PR is opened.

Every check here also runs in CI. Running them locally first turns a red build
and a force-push into a thirty-second wait.

Usage:
    python check_pr_readiness.py [--base main] [--skip-tests]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SKILL_DIR = Path(__file__).resolve().parent
REVIEWER_SCRIPTS = REPO_ROOT / ".claude" / "skills" / "code-reviewer" / "scripts"

BRANCH_RE = re.compile(r"^(feat|fix|test|docs|refactor|perf|chore|ci|style)/(\d{4})-[a-z0-9-]+$")

# data/ is gitignored, but a forced add or a stray path can still stage a large
# file, and removing one from history is painful.
MAX_FILE_BYTES = 1_000_000

SECRET_RE = re.compile(
    r'(?i)(api[_-]?key|secret|password|token)\s*=\s*["\'][^"\']{12,}["\']|sk-[A-Za-z0-9]{20,}'
)


def git(*args: str) -> tuple[int, str, str]:
    r = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def run(cmd: list[str]) -> tuple[int, str]:
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)


def check_branch_name() -> dict:
    code, branch, _ = git("rev-parse", "--abbrev-ref", "HEAD")
    if code != 0:
        return {"name": "branch_name", "passed": False, "detail": "not a git repository"}

    if branch == "main":
        return {"name": "branch_name", "passed": False, "branch": branch,
                "detail": "on main — create a feature branch before opening a PR"}

    if not BRANCH_RE.match(branch):
        return {"name": "branch_name", "passed": False, "branch": branch,
                "detail": "expected <type>/<NNNN>-<slug>, e.g. feat/0007-lob-reconstructor"}

    return {"name": "branch_name", "passed": True, "branch": branch,
            "issue": BRANCH_RE.match(branch).group(2)}


def check_clean_tree() -> dict:
    _, out, _ = git("status", "--porcelain")
    dirty = [ln for ln in out.splitlines() if not ln.startswith("??")]
    return {"name": "clean_tree", "passed": not dirty,
            "detail": "" if not dirty else f"{len(dirty)} uncommitted change(s): "
                                           + ", ".join(l[3:] for l in dirty[:5])}


def check_rebased(base: str) -> dict:
    git("fetch", "origin", base)
    code, out, _ = git("rev-list", "--count", f"HEAD..origin/{base}")
    if code != 0:
        # No remote configured is common early in a project.
        code2, out2, _ = git("rev-list", "--count", f"HEAD..{base}")
        if code2 != 0:
            return {"name": "rebased", "passed": True, "detail": f"could not compare against {base}"}
        out = out2

    behind = int(out or 0)
    return {"name": "rebased", "passed": behind == 0,
            "behind": behind,
            "detail": "" if behind == 0 else
                      f"{behind} commit(s) behind {base} — run: git fetch origin && git rebase origin/{base}"}


def check_commit_messages(base: str) -> dict:
    code, out, _ = git("log", f"{base}..HEAD", "--format=%H%x00%B%x1e")
    if code != 0 or not out:
        return {"name": "commit_messages", "passed": True, "detail": "no commits to check"}

    failures = []
    for record in out.split("\x1e"):
        record = record.strip()
        if not record:
            continue
        sha, _, message = record.partition("\x00")
        r = subprocess.run(
            [sys.executable, str(SKILL_DIR / "validate_commit.py"), "--quiet", message],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        if r.returncode != 0:
            failures.append(f"{sha[:8]}: {message.splitlines()[0][:60]}")

    return {"name": "commit_messages", "passed": not failures,
            "detail": "" if not failures else
                      f"{len(failures)} invalid message(s): " + "; ".join(failures[:3])}


def check_large_files(base: str) -> dict:
    code, out, _ = git("diff", "--name-only", f"{base}...HEAD")
    if code != 0:
        return {"name": "large_files", "passed": True, "detail": "could not diff"}

    offenders = []
    for path in out.splitlines():
        full = REPO_ROOT / path
        if full.exists() and full.stat().st_size > MAX_FILE_BYTES:
            offenders.append(f"{path} ({full.stat().st_size // 1024} KB)")

    return {"name": "large_files", "passed": not offenders,
            "detail": "" if not offenders else
                      "files over 1 MB: " + ", ".join(offenders[:3])
                      + " — data and checkpoints belong outside git"}


def check_secrets(base: str) -> dict:
    code, out, _ = git("diff", f"{base}...HEAD")
    if code != 0:
        return {"name": "secrets", "passed": True, "detail": "could not diff"}

    hits = []
    for line in out.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            if SECRET_RE.search(line):
                hits.append(line[:70])

    return {"name": "secrets", "passed": not hits,
            "detail": "" if not hits else
                      f"{len(hits)} possible secret(s) in the diff — move them to .env"}


def check_lint() -> dict:
    code, out = run(["make", "lint"])
    return {"name": "lint", "passed": code == 0,
            "detail": "" if code == 0 else out.strip().splitlines()[-1] if out.strip() else "make lint failed"}


def check_tests() -> dict:
    code, out = run(["make", "test"])
    if code == 0:
        return {"name": "tests", "passed": True, "detail": ""}

    # No tests yet is expected early in the project, not a PR blocker.
    if "no tests ran" in out.lower() or "collected 0 items" in out.lower():
        return {"name": "tests", "passed": True, "detail": "no tests collected yet"}

    tail = [l for l in out.strip().splitlines() if l.strip()][-3:]
    return {"name": "tests", "passed": False, "detail": " | ".join(tail)}


def check_duplication(base: str) -> dict:
    code, out, _ = git("diff", "--name-only", f"{base}...HEAD")
    paths = [p for p in out.splitlines()
             if p.endswith(".py") and p.startswith(("src/", "scripts/"))
             and (REPO_ROOT / p).exists()]

    if not paths:
        return {"name": "duplication", "passed": True, "detail": "no source files changed"}

    script = REVIEWER_SCRIPTS / "check_duplication.py"
    if not script.exists():
        return {"name": "duplication", "passed": True, "detail": "checker not available"}

    rc, out2 = run([sys.executable, str(script), *paths, "--repo-root", str(REPO_ROOT)])
    return {"name": "duplication", "passed": rc == 0,
            "detail": "" if rc == 0 else "canonical concept reimplemented — see .claude/Agent.md"}


def check_conventions(base: str) -> dict:
    code, out, _ = git("diff", "--name-only", f"{base}...HEAD")
    paths = [p for p in out.splitlines() if p.endswith(".py") and (REPO_ROOT / p).exists()]

    if not paths:
        return {"name": "conventions", "passed": True, "detail": "no Python files changed"}

    script = REVIEWER_SCRIPTS / "check_conventions.py"
    if not script.exists():
        return {"name": "conventions", "passed": True, "detail": "checker not available"}

    rc, out2 = run([sys.executable, str(script), *paths, "--repo-root", str(REPO_ROOT)])
    detail = ""
    if rc != 0:
        detail = "convention errors — causality findings are blocking"
    return {"name": "conventions", "passed": rc == 0, "detail": detail}


def check_issue_exists(branch_check: dict) -> dict:
    issue = branch_check.get("issue")
    if not issue:
        return {"name": "issue_reference", "passed": False,
                "detail": "no issue number in the branch name"}

    matches = list((REPO_ROOT / "docs" / "issues").glob(f"phase-*/{issue}-*.md"))
    return {"name": "issue_reference", "passed": bool(matches),
            "detail": "" if matches else
                      f"no issue file matching {issue}-*.md under docs/issues/phase-*/"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="main")
    parser.add_argument("--skip-tests", action="store_true", help="skip make test (slow)")
    args = parser.parse_args()

    print("=" * 74)
    print("PR READINESS CHECK")
    print("=" * 74)
    print()

    branch = check_branch_name()
    checks = [
        branch,
        check_issue_exists(branch),
        check_clean_tree(),
        check_rebased(args.base),
        check_commit_messages(args.base),
        check_large_files(args.base),
        check_secrets(args.base),
        check_duplication(args.base),
        check_conventions(args.base),
        check_lint(),
    ]
    if not args.skip_tests:
        checks.append(check_tests())

    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {c['name']}")
        if c.get("branch"):
            print(f"         branch: {c['branch']}")
        if c.get("detail"):
            print(f"         {c['detail']}")

    print()
    failed = [c for c in checks if not c["passed"]]
    print("=" * 74)
    if failed:
        print(f"RESULT: NOT READY — {len(failed)} check(s) failed")
        print("=" * 74)
        print()
        print("Fix these before opening the PR; CI runs the same checks.")
    else:
        print("RESULT: ready to open a PR")
        print("=" * 74)
        print()
        print("Next:")
        print("  python .claude/skills/git-workflow/scripts/generate_pr_description.py --output pr.md")
        print("  gh pr create --body-file pr.md")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

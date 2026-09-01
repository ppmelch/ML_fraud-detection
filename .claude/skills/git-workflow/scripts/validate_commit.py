#!/usr/bin/env python3
"""Validate a commit message against this project's conventions.

Suitable as a commit-msg hook:

    #!/bin/sh
    python .claude/skills/git-workflow/scripts/validate_commit.py --file "$1"

Usage:
    python validate_commit.py "feat(data): implement LOB reconstructor"
    python validate_commit.py --file .git/COMMIT_EDITMSG
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

VALID_TYPES = {
    "feat": "a new capability",
    "fix": "a bug fix",
    "test": "adding or correcting tests",
    "docs": "documentation only",
    "refactor": "restructuring without behaviour change",
    "perf": "a performance improvement",
    "chore": "tooling, dependencies, configuration",
    "ci": "CI or workflow changes",
    "style": "formatting only",
}

# Scopes mirror the module layout, so a reader can locate a change from its
# subject line alone.
VALID_SCOPES = {
    "data", "features", "models", "simulator", "backtester", "utils",
    "viz", "tests", "docs", "ci", "docker", "deps", "claude", "scripts",
}

SUBJECT_MAX = 72
BODY_WRAP = 72

HEADER_RE = re.compile(r"^(?P<type>\w+)(?:\((?P<scope>[\w-]+)\))?(?P<breaking>!)?: (?P<subject>.+)$")

# Past tense and third person are the two most common deviations from the
# imperative mood the format requires.
PAST_TENSE_RE = re.compile(
    r"^(added|implemented|fixed|updated|removed|created|changed|refactored|"
    r"adds|implements|fixes|updates|removes|creates|changes|refactors)\b",
    re.IGNORECASE,
)

ISSUE_RE = re.compile(r"#(\d{1,4})\b")


def validate(message: str) -> dict:
    lines = message.rstrip().splitlines()
    errors: list[str] = []
    warnings: list[str] = []

    if not lines or not lines[0].strip():
        return {"passed": False, "errors": ["empty commit message"], "warnings": [], "parsed": {}}

    # Ignore comment lines git adds to the editor buffer.
    lines = [ln for ln in lines if not ln.startswith("#")]
    if not lines:
        return {"passed": False, "errors": ["message contains only comments"], "warnings": [], "parsed": {}}

    header = lines[0]
    match = HEADER_RE.match(header)

    if not match:
        errors.append(
            "header does not match '<type>(<scope>): <subject>'\n"
            f"    got: {header}\n"
            "    expected e.g.: feat(data): implement LOBReconstructor"
        )
        return {"passed": False, "errors": errors, "warnings": warnings, "parsed": {}}

    ctype = match.group("type")
    scope = match.group("scope")
    subject = match.group("subject")
    breaking = bool(match.group("breaking"))

    if ctype not in VALID_TYPES:
        errors.append(
            f"unknown type '{ctype}'. Valid: {', '.join(sorted(VALID_TYPES))}"
        )

    if scope and scope not in VALID_SCOPES:
        warnings.append(
            f"unusual scope '{scope}'. Common scopes: {', '.join(sorted(VALID_SCOPES))}"
        )

    if len(header) > SUBJECT_MAX:
        errors.append(f"header is {len(header)} characters; limit is {SUBJECT_MAX}")

    if subject and subject[0].isupper():
        warnings.append(f"subject starts with a capital: '{subject[:30]}'")

    if subject.endswith("."):
        errors.append("subject ends with a period")

    if PAST_TENSE_RE.match(subject):
        first = subject.split()[0]
        errors.append(
            f"subject is not imperative: '{first}'. "
            "Write it as a command — 'implement', not 'implemented' or 'implements'"
        )

    # Blank line between header and body
    if len(lines) > 1 and lines[1].strip():
        errors.append("missing blank line between header and body")

    body = "\n".join(lines[2:]) if len(lines) > 2 else ""

    for i, line in enumerate(lines[2:], start=3):
        if len(line) > BODY_WRAP and not line.startswith(("http", "    ", "\t", "|")):
            warnings.append(f"body line {i} is {len(line)} characters; wrap at {BODY_WRAP}")

    issues = ISSUE_RE.findall(message)
    if ctype in ("feat", "fix") and not issues:
        warnings.append(
            "no issue reference. feat and fix commits should cite one, e.g. 'Closes #0007'"
        )

    if breaking and "BREAKING CHANGE" not in message:
        warnings.append("'!' marks a breaking change; add a 'BREAKING CHANGE:' body section")

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "parsed": {
            "type": ctype,
            "scope": scope,
            "subject": subject,
            "breaking": breaking,
            "issues": issues,
            "header_length": len(header),
            "has_body": bool(body.strip()),
        },
    }


def print_report(result: dict, message: str) -> None:
    print("=" * 74)
    print("COMMIT MESSAGE VALIDATION")
    print("=" * 74)
    print()
    for line in message.rstrip().splitlines()[:6]:
        print(f"  | {line}")
    print()

    p = result["parsed"]
    if p:
        print(f"  type: {p['type']}   scope: {p['scope'] or '-'}   "
              f"length: {p['header_length']}/{SUBJECT_MAX}")
        if p["issues"]:
            print(f"  issues: {', '.join('#' + i for i in p['issues'])}")
        print()

    if result["errors"]:
        print("Errors:")
        for e in result["errors"]:
            print(f"  - {e}")
        print()

    if result["warnings"]:
        print("Warnings:")
        for w in result["warnings"]:
            print(f"  - {w}")
        print()

    print("=" * 74)
    print("RESULT:", "valid" if result["passed"] else "INVALID")
    print("=" * 74)

    if not result["passed"]:
        print()
        print("Format:  <type>(<scope>): <subject>")
        print()
        print("Types:")
        for t, desc in sorted(VALID_TYPES.items()):
            print(f"  {t:9s} {desc}")
        print()
        print("Example:")
        print("  feat(data): implement LOBReconstructor with FIFO queue tracking")
        print()
        print("  Rebuilds the order book from event streams, tracking queue")
        print("  position in shares so Phase 3 can estimate fill probability.")
        print()
        print("  Closes #0007")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("message", nargs="?", help="commit message")
    parser.add_argument("--file", type=Path, help="read the message from a file")
    parser.add_argument("--quiet", action="store_true", help="print nothing on success")
    args = parser.parse_args()

    if args.file:
        if not args.file.exists():
            print(f"ERROR: file not found: {args.file}", file=sys.stderr)
            return 2
        message = args.file.read_text(encoding="utf-8")
    elif args.message:
        message = args.message
    else:
        print("ERROR: provide a message or --file", file=sys.stderr)
        return 2

    result = validate(message)

    if not (args.quiet and result["passed"]):
        print_report(result, message)

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

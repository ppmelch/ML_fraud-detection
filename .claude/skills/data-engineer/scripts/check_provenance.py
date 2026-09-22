#!/usr/bin/env python3
"""Trace an artifact back through its upstream chain to the raw events.

By Phase 4 the chain runs seven links deep: raw events, cleaned, reconstructed
LOB, features, predictions, executions, P&L. When a final number looks wrong,
the first question is which inputs produced it. This answers that in seconds
rather than an afternoon of archaeology.

Also catches the staleness case: an upstream file regenerated after the
downstream artifact was built, which means the downstream file describes a
version of the data that no longer exists.

Usage:
    python check_provenance.py data/backtest/backtest_results_full_v1.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

METADATA_SUFFIX = "_metadata.json"


def metadata_path_for(artifact: Path) -> Path:
    """Companion metadata path: foo.csv -> foo_metadata.json."""
    return artifact.with_name(artifact.stem + METADATA_SUFFIX)


def load_metadata(artifact: Path) -> tuple[dict | None, str | None]:
    """Return (metadata, error)."""
    meta_path = metadata_path_for(artifact)
    if not meta_path.exists():
        return None, f"no metadata file at {meta_path}"
    try:
        return json.loads(meta_path.read_text()), None
    except json.JSONDecodeError as exc:
        return None, f"malformed metadata: {exc}"


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def extract_upstream(metadata: dict, base_dir: Path) -> list[Path]:
    """Read upstream_artifacts, tolerating the shapes it appears in.

    Accepts a list of paths, a list of {path: ...} records, or a mapping of
    role -> path, since different pipeline stages record it differently.
    """
    raw = metadata.get("upstream_artifacts")
    if raw is None:
        # Some early stages record a single source instead.
        single = metadata.get("source_file")
        raw = [single] if single else []

    paths: list[Path] = []

    if isinstance(raw, dict):
        candidates = list(raw.values())
    elif isinstance(raw, list):
        candidates = raw
    else:
        candidates = [raw]

    for item in candidates:
        if isinstance(item, dict):
            item = item.get("path") or item.get("file") or item.get("filename")
        if not item:
            continue
        p = Path(str(item))
        paths.append(p if p.is_absolute() else (base_dir / p))

    return paths


def walk_chain(
    artifact: Path,
    repo_root: Path,
    max_depth: int,
    _depth: int = 0,
    _seen: set[Path] | None = None,
) -> dict:
    """Recursively resolve the provenance chain."""
    seen = _seen if _seen is not None else set()
    resolved = artifact.resolve()

    node: dict = {
        "path": str(artifact),
        "depth": _depth,
        "exists": artifact.exists(),
        "issues": [],
        "upstream": [],
    }

    if not artifact.exists():
        node["issues"].append("file does not exist")
        return node

    if resolved in seen:
        node["issues"].append("cycle detected — already visited in this chain")
        return node
    seen.add(resolved)

    if _depth >= max_depth:
        node["issues"].append(f"max depth {max_depth} reached; chain truncated")
        return node

    metadata, error = load_metadata(artifact)
    if error:
        node["issues"].append(error)
        return node

    node["generated_at"] = metadata.get("generated_at")
    node["description"] = metadata.get("description", "")

    generated = parse_timestamp(metadata.get("generated_at"))
    upstream_paths = extract_upstream(metadata, repo_root)

    if not upstream_paths:
        node["is_root"] = True
        return node

    for up in upstream_paths:
        child = walk_chain(up, repo_root, max_depth, _depth + 1, seen)

        # Staleness: upstream regenerated after this artifact was built.
        if child.get("exists") and generated:
            up_generated = parse_timestamp(child.get("generated_at"))
            if up_generated and up_generated > generated:
                child["issues"].append(
                    f"STALE: upstream generated {up_generated.isoformat()}, "
                    f"after downstream {generated.isoformat()}"
                )
            elif not up_generated:
                mtime = datetime.fromtimestamp(up.stat().st_mtime).astimezone()
                if generated.tzinfo and mtime > generated:
                    child["issues"].append(
                        f"POSSIBLY STALE: upstream mtime {mtime.isoformat()} "
                        f"after downstream generated {generated.isoformat()}"
                    )

        node["upstream"].append(child)

    return node


def collect_issues(node: dict, acc: list[tuple[str, str]] | None = None) -> list[tuple[str, str]]:
    acc = acc if acc is not None else []
    for issue in node["issues"]:
        acc.append((node["path"], issue))
    for child in node["upstream"]:
        collect_issues(child, acc)
    return acc


def print_tree(node: dict, prefix: str = "", is_last: bool = True) -> None:
    connector = "" if node["depth"] == 0 else ("`-- " if is_last else "|-- ")
    marker = "OK " if node["exists"] and not node["issues"] else "!! "
    print(f"{prefix}{connector}{marker}{node['path']}")

    detail_prefix = prefix + ("" if node["depth"] == 0 else ("    " if is_last else "|   "))

    if node.get("generated_at"):
        print(f"{detail_prefix}    generated: {node['generated_at']}")
    if node.get("is_root"):
        print(f"{detail_prefix}    (root artifact — no upstream declared)")
    for issue in node["issues"]:
        print(f"{detail_prefix}    ISSUE: {issue}")

    children = node["upstream"]
    for i, child in enumerate(children):
        print_tree(child, detail_prefix, i == len(children) - 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("artifact", type=Path, help="artifact whose chain to trace")
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(),
                        help="base for resolving relative upstream paths")
    parser.add_argument("--output", type=Path, help="write the chain as JSON")
    args = parser.parse_args()

    if not args.artifact.exists():
        print(f"ERROR: file not found: {args.artifact}", file=sys.stderr)
        return 2

    chain = walk_chain(args.artifact, args.repo_root, args.max_depth)

    print("=" * 74)
    print("PROVENANCE CHAIN")
    print("=" * 74)
    print()
    print_tree(chain)
    print()

    issues = collect_issues(chain)
    print("=" * 74)
    if issues:
        print(f"RESULT: {len(issues)} issue(s) in the chain")
        print()
        for path, issue in issues:
            print(f"  {path}")
            print(f"    {issue}")
        print()
        print("A broken chain means a result cannot be traced to its inputs.")
        print("Regenerate the downstream artifacts rather than editing metadata")
        print("to point at a different file.")
    else:
        print("RESULT: chain resolves cleanly to the root artifact")
    print("=" * 74)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(chain, indent=2))
        print(f"\nChain written to {args.output}")

    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Profile a dataset for quality problems the schema contract cannot catch.

Schema validation answers "is this file structurally legal". This answers
"is this file trustworthy": gaps, duplicates, orphan references, outliers,
and distribution shape.

Severity is `errors` for referential violations or backwards timestamps,
`warnings` for gaps and outliers, `clean` otherwise. Exits 1 on errors.

Usage:
    python profile_dataset.py data/raw/events.csv --output report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# One minute. Suits the synthetic hour-long sample; real exchange data would
# use something far smaller. Kept a parameter, and recorded in the report so
# the threshold behind a result is auditable.
DEFAULT_GAP_THRESHOLD_MS = 60_000

# The conventional 1.5x IQR fence flags far too much on heavy-tailed financial
# quantities. 3x is the working choice; the multiplier is recorded in the
# report rather than left implicit.
IQR_MULTIPLIER = 3.0


def analyze_nulls(df: pd.DataFrame) -> dict:
    per_column = {}
    for col in df.columns:
        n = int(df[col].isna().sum())
        per_column[col] = {"count": n, "pct": round(100.0 * n / len(df), 4) if len(df) else 0.0}

    return {
        "per_column": per_column,
        "columns_with_nulls": [c for c, v in per_column.items() if v["count"] > 0],
        "rows_with_any_null": int(df.isna().any(axis=1).sum()),
    }


def analyze_duplicates(df: pd.DataFrame, key_columns: list[str] | None) -> dict:
    result = {
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "exact_duplicate_examples": df.index[df.duplicated()][:5].tolist(),
    }

    if key_columns:
        present = [c for c in key_columns if c in df.columns]
        if present:
            dup = df.duplicated(subset=present)
            result["key_duplicates"] = {
                "columns": present,
                "count": int(dup.sum()),
                "examples": df.index[dup][:5].tolist(),
            }

    # An `add` event introduces an order id; seeing the same id added twice
    # means the stream is inconsistent, not merely repetitive.
    if {"order_id", "event_type"}.issubset(df.columns):
        adds = df[df["event_type"] == "add"]
        n_dup = int(adds["order_id"].duplicated().sum())
        result["duplicate_add_order_ids"] = n_dup

    return result


def analyze_temporal(df: pd.DataFrame, gap_threshold_ms: int) -> dict:
    if "timestamp" not in df.columns:
        return {"applicable": False}

    ts = df["timestamp"]
    diffs = ts.diff().dropna()

    backwards = diffs[diffs < 0]
    gaps = diffs[diffs > gap_threshold_ms]

    gap_records = []
    for idx in gaps.index[:20]:
        pos = df.index.get_loc(idx)
        gap_records.append(
            {
                "start_timestamp": int(ts.iloc[pos - 1]),
                "end_timestamp": int(ts.iloc[pos]),
                "duration_ms": int(diffs.loc[idx]),
            }
        )

    span = float(ts.iloc[-1] - ts.iloc[0]) if len(ts) > 1 else 0.0
    gap_total = float(gaps.sum()) if not gaps.empty else 0.0

    return {
        "applicable": True,
        "backwards_transitions": int(len(backwards)),
        "backwards_examples": backwards.index[:5].tolist(),
        "gap_threshold_ms": gap_threshold_ms,
        "gaps_over_threshold": int(len(gaps)),
        "gap_details": gap_records,
        "span_ms": span,
        "coverage_ratio": round(1.0 - gap_total / span, 6) if span > 0 else 1.0,
        "inter_arrival": {
            "mean_ms": float(diffs.mean()) if not diffs.empty else np.nan,
            "median_ms": float(diffs.median()) if not diffs.empty else np.nan,
            "p95_ms": float(diffs.quantile(0.95)) if not diffs.empty else np.nan,
            "max_ms": float(diffs.max()) if not diffs.empty else np.nan,
            "zero_gaps": int((diffs == 0).sum()),
        },
    }


def analyze_referential(df: pd.DataFrame) -> dict:
    """Every cancel/execute must reference an order_id seen in a prior add.

    Walks the stream once maintaining the set of open ids. Note that `execute`
    does not close an order — a partial fill leaves it open — so only `cancel`
    discards.
    """
    required = {"order_id", "event_type"}
    if not required.issubset(df.columns):
        return {"applicable": False}

    open_ids: set[int] = set()
    cancelled: set[int] = set()
    orphan_cancels: list[int] = []
    orphan_executes: list[int] = []
    double_cancels: list[int] = []

    for idx, (oid, etype) in enumerate(zip(df["order_id"], df["event_type"])):
        if etype == "add":
            open_ids.add(oid)
        elif etype == "cancel":
            if oid in cancelled:
                double_cancels.append(idx)
            elif oid not in open_ids:
                orphan_cancels.append(idx)
            else:
                open_ids.discard(oid)
                cancelled.add(oid)
        elif etype == "execute":
            if oid not in open_ids:
                orphan_executes.append(idx)

    return {
        "applicable": True,
        "orphan_cancels": len(orphan_cancels),
        "orphan_cancel_examples": orphan_cancels[:5],
        "orphan_executes": len(orphan_executes),
        "orphan_execute_examples": orphan_executes[:5],
        "double_cancels": len(double_cancels),
        "still_open_at_end": len(open_ids),
    }


def analyze_outliers(df: pd.DataFrame, columns: list[str]) -> dict:
    result = {"multiplier": IQR_MULTIPLIER, "per_column": {}}

    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue

        s = df[col].dropna()
        if s.empty:
            continue

        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - IQR_MULTIPLIER * iqr
        upper = q3 + IQR_MULTIPLIER * iqr
        mask = (s < lower) | (s > upper)

        result["per_column"][col] = {
            "count": int(mask.sum()),
            "pct": round(100.0 * mask.sum() / len(s), 4),
            "lower_fence": float(lower),
            "upper_fence": float(upper),
            "min": float(s.min()),
            "max": float(s.max()),
            "lowest_5": s.nsmallest(5).tolist(),
            "highest_5": s.nlargest(5).tolist(),
        }

    return result


def profile(df: pd.DataFrame, gap_threshold_ms: int, key_columns: list[str] | None) -> dict:
    nulls = analyze_nulls(df)
    duplicates = analyze_duplicates(df, key_columns)
    temporal = analyze_temporal(df, gap_threshold_ms)
    referential = analyze_referential(df)

    numeric_candidates = [c for c in ("price", "quantity", "spread", "mid_price") if c in df.columns]
    outliers = analyze_outliers(df, numeric_candidates)

    # Severity: referential violations and backwards time are structural
    # failures; gaps and outliers are conditions worth knowing about.
    error_conditions = [
        referential.get("orphan_cancels", 0) > 0,
        referential.get("orphan_executes", 0) > 0,
        referential.get("double_cancels", 0) > 0,
        temporal.get("backwards_transitions", 0) > 0,
        duplicates.get("exact_duplicate_rows", 0) > 0,
        duplicates.get("duplicate_add_order_ids", 0) > 0,
        len(nulls["columns_with_nulls"]) > 0,
    ]
    warning_conditions = [
        temporal.get("gaps_over_threshold", 0) > 0,
        any(v["count"] > 0 for v in outliers["per_column"].values()),
    ]

    if any(error_conditions):
        severity = "errors"
    elif any(warning_conditions):
        severity = "warnings"
    else:
        severity = "clean"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "null_analysis": nulls,
        "duplicate_analysis": duplicates,
        "temporal_analysis": temporal,
        "referential_analysis": referential,
        "outlier_analysis": outliers,
        "summary": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "issues_found": sum(error_conditions) + sum(warning_conditions),
            "severity": severity,
        },
    }


def print_report(report: dict) -> None:
    s = report["summary"]
    print("=" * 74)
    print("DATA QUALITY PROFILE")
    print("=" * 74)
    print(f"Rows: {s['total_rows']}   Columns: {s['total_columns']}   Severity: {s['severity'].upper()}")
    print()

    nulls = report["null_analysis"]
    if nulls["columns_with_nulls"]:
        print(f"Nulls in {len(nulls['columns_with_nulls'])} column(s):")
        for col in nulls["columns_with_nulls"][:10]:
            v = nulls["per_column"][col]
            print(f"  {col:24s} {v['count']:6d}  ({v['pct']}%)")
    else:
        print("Nulls: none")
    print()

    dup = report["duplicate_analysis"]
    print(f"Duplicate rows: {dup['exact_duplicate_rows']}")
    if "duplicate_add_order_ids" in dup:
        print(f"Duplicate add order_ids: {dup['duplicate_add_order_ids']}")
    print()

    t = report["temporal_analysis"]
    if t.get("applicable"):
        ia = t["inter_arrival"]
        print("Temporal:")
        print(f"  backwards transitions  {t['backwards_transitions']}")
        print(f"  gaps > {t['gap_threshold_ms']}ms{'':10s} {t['gaps_over_threshold']}")
        print(f"  coverage ratio         {t['coverage_ratio']}")
        print(f"  inter-arrival median   {ia['median_ms']:.1f} ms")
        print(f"  inter-arrival p95      {ia['p95_ms']:.1f} ms")
        print(f"  zero gaps (same ms)    {ia['zero_gaps']}")
        print()

    r = report["referential_analysis"]
    if r.get("applicable"):
        print("Referential integrity:")
        print(f"  orphan cancels    {r['orphan_cancels']}")
        print(f"  orphan executes   {r['orphan_executes']}")
        print(f"  double cancels    {r['double_cancels']}")
        print(f"  still open at end {r['still_open_at_end']}  (informational)")
        print()

    o = report["outlier_analysis"]
    if o["per_column"]:
        print(f"Outliers (IQR x{o['multiplier']}):")
        for col, v in o["per_column"].items():
            print(f"  {col:24s} {v['count']:6d}  ({v['pct']}%)  range [{v['min']:.4g}, {v['max']:.4g}]")
        print()

    print("=" * 74)
    print(f"RESULT: {s['severity']}  ({s['issues_found']} condition(s) flagged)")
    print("=" * 74)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", type=Path, help="CSV to profile")
    parser.add_argument("--gap-threshold-ms", type=int, default=DEFAULT_GAP_THRESHOLD_MS)
    parser.add_argument("--key-columns", nargs="+", help="columns forming a logical key")
    parser.add_argument("--output", type=Path, help="write the report as JSON")
    args = parser.parse_args()

    if not args.file.exists():
        print(f"ERROR: file not found: {args.file}", file=sys.stderr)
        return 2

    df = pd.read_csv(args.file)
    report = profile(df, args.gap_threshold_ms, args.key_columns)
    report["source_file"] = str(args.file)
    print_report(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nReport written to {args.output}")

    return 1 if report["summary"]["severity"] == "errors" else 0


if __name__ == "__main__":
    sys.exit(main())

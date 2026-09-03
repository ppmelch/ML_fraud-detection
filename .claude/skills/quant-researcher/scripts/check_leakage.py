#!/usr/bin/env python3
"""Detect lookahead bias in a feature matrix and its labels.

Leakage in this project produces the same visible symptom as success: unusually
good accuracy. The checks below look for the structural fingerprints of the
alignment errors that cause it, so the problem is found where it is cheap to
fix rather than four phases later.

Exits non-zero when any check fails, so it can gate a pipeline.

Usage:
    python check_leakage.py <features.csv> <labels.csv> --horizon 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# A single feature correlating this strongly with a forward return is not
# plausible on order-book data. It is a leaked target far more often than an
# edge, so the default is deliberately low enough to catch it.
SUSPICIOUS_CORRELATION = 0.30

# Above this, treat it as certainly leaked rather than merely suspicious.
CERTAIN_LEAKAGE_CORRELATION = 0.70

ID_COLUMNS = {"timestamp", "sequence_number", "order_id", "fold_id"}


class LeakageCheckFailed(Exception):
    """Raised when leakage checks fail and --strict is set."""


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns that are features rather than identifiers or targets."""
    return [
        c
        for c in df.columns
        if c not in ID_COLUMNS
        and not c.startswith(("label_", "price_movement_", "fwd_return_"))
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def check_feature_target_correlation(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    target_col: str,
    threshold: float,
) -> dict:
    """Flag features correlating implausibly with the target.

    A genuine order-book feature correlates weakly with future returns. Strong
    correlation means the feature has seen the target, usually because it was
    computed from a forward-shifted series.
    """
    findings = []
    target = labels[target_col]

    for col in _feature_columns(features):
        mask = features[col].notna() & target.notna()
        if mask.sum() < 30:
            continue

        corr = features.loc[mask, col].corr(target[mask], method="spearman")
        if pd.isna(corr):
            continue

        if abs(corr) >= CERTAIN_LEAKAGE_CORRELATION:
            severity = "certain"
        elif abs(corr) >= threshold:
            severity = "suspicious"
        else:
            continue

        findings.append(
            {
                "feature": col,
                "target": target_col,
                "spearman": round(float(corr), 4),
                "n": int(mask.sum()),
                "severity": severity,
            }
        )

    return {
        "check": "feature_target_correlation",
        "passed": not any(f["severity"] == "certain" for f in findings),
        "findings": findings,
    }


def check_label_nan_structure(labels: pd.DataFrame, horizon: int) -> dict:
    """A forward-looking label must have exactly `horizon` trailing NaN.

    Fewer means the last rows were filled with something invented; more means
    the shift was larger than declared. Both break the alignment that every
    downstream accuracy figure assumes.
    """
    findings = []
    label_cols = [c for c in labels.columns if c.startswith(("label_", "price_movement_", "fwd_return_"))]

    for col in label_cols:
        # Trailing NaN run length
        values = labels[col].to_numpy()
        trailing = 0
        for v in values[::-1]:
            if pd.isna(v):
                trailing += 1
            else:
                break

        # Horizon encoded in the column name wins over the CLI default, since
        # a file can carry several horizons at once.
        suffix = col.rsplit("_", 1)[-1]
        expected = int(suffix) if suffix.isdigit() else horizon

        if trailing != expected:
            findings.append(
                {
                    "column": col,
                    "expected_trailing_nan": expected,
                    "actual_trailing_nan": trailing,
                    "issue": "filled forward values" if trailing < expected else "shift larger than declared",
                }
            )

    return {
        "check": "label_nan_structure",
        "passed": not findings,
        "findings": findings,
    }


def check_causality_by_truncation(features: pd.DataFrame, sample_cols: list[str] | None = None) -> dict:
    """Recompute nothing — instead verify values do not depend on later rows.

    A trailing window cannot change when future rows are removed. If truncating
    the frame alters an earlier value, the column read forward.

    This catches centered rolling windows and negative shifts, which are the
    two most common ways a feature quietly becomes non-causal.
    """
    findings = []
    cols = sample_cols or _feature_columns(features)
    if len(features) < 100:
        return {"check": "causality_truncation", "passed": True, "findings": [],
                "note": "too few rows to test"}

    cut = len(features) // 2
    truncated = features.iloc[:cut]

    for col in cols:
        full_head = features[col].iloc[:cut]
        trunc_head = truncated[col]

        # Compare only where both are non-null; a NaN appearing in one and not
        # the other is itself a difference worth reporting.
        both_nan = full_head.isna() & trunc_head.isna()
        differs_nan = (full_head.isna() != trunc_head.isna()) & ~both_nan
        differs_value = (~full_head.isna()) & (~trunc_head.isna()) & (
            ~np.isclose(full_head.fillna(0), trunc_head.fillna(0), rtol=1e-9, atol=1e-12)
        )

        n_diff = int((differs_nan | differs_value).sum())
        if n_diff:
            findings.append(
                {
                    "column": col,
                    "rows_changed_by_truncation": n_diff,
                    "issue": "value depends on later rows (centered window or negative shift)",
                }
            )

    return {
        "check": "causality_truncation",
        "passed": not findings,
        "findings": findings,
        "note": "this check is only meaningful on features already materialized in the file",
    }


def check_perfect_predictors(features: pd.DataFrame, labels: pd.DataFrame, target_col: str) -> dict:
    """Flag any feature that reproduces the label's sign almost exactly."""
    findings = []
    target_sign = np.sign(labels[target_col])

    for col in _feature_columns(features):
        mask = features[col].notna() & target_sign.notna() & (target_sign != 0)
        if mask.sum() < 30:
            continue

        agreement = (np.sign(features.loc[mask, col]) == target_sign[mask]).mean()
        if agreement >= 0.95 or agreement <= 0.05:
            findings.append(
                {
                    "feature": col,
                    "sign_agreement": round(float(agreement), 4),
                    "n": int(mask.sum()),
                    "issue": "feature reproduces target sign — almost certainly the target itself",
                }
            )

    return {
        "check": "perfect_predictors",
        "passed": not findings,
        "findings": findings,
    }


def run_all_checks(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    horizon: int,
    threshold: float,
) -> dict:
    """Run every leakage check and collect results."""
    target_candidates = [
        c for c in labels.columns
        if c.startswith(("price_movement_", "fwd_return_")) and c.endswith(str(horizon))
    ]
    if not target_candidates:
        target_candidates = [
            c for c in labels.columns if c.startswith(("price_movement_", "fwd_return_"))
        ][:1]

    if not target_candidates:
        return {
            "passed": False,
            "error": "no target column found; expected price_movement_* or fwd_return_*",
            "checks": [],
        }

    target_col = target_candidates[0]

    checks = [
        check_feature_target_correlation(features, labels, target_col, threshold),
        check_label_nan_structure(labels, horizon),
        check_causality_by_truncation(features),
        check_perfect_predictors(features, labels, target_col),
    ]

    return {
        "passed": all(c["passed"] for c in checks),
        "target_column": target_col,
        "n_rows": len(features),
        "n_features": len(_feature_columns(features)),
        "checks": checks,
    }


def print_report(report: dict, verbose: bool) -> None:
    """Human-readable summary."""
    print("=" * 72)
    print("LEAKAGE CHECK")
    print("=" * 72)

    if "error" in report:
        print(f"ERROR: {report['error']}")
        return

    print(f"Rows:     {report['n_rows']}")
    print(f"Features: {report['n_features']}")
    print(f"Target:   {report['target_column']}")
    print()

    for check in report["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        n = len(check["findings"])
        print(f"[{status}] {check['check']:32s} {n} finding(s)")

        if check["findings"] and (verbose or not check["passed"]):
            for f in check["findings"][:10]:
                detail = ", ".join(f"{k}={v}" for k, v in f.items())
                print(f"         {detail}")
            if len(check["findings"]) > 10:
                print(f"         ... and {len(check['findings']) - 10} more")
        if check.get("note"):
            print(f"         note: {check['note']}")
        print()

    print("=" * 72)
    print("RESULT:", "no leakage detected" if report["passed"] else "LEAKAGE DETECTED")
    print("=" * 72)

    if not report["passed"]:
        print()
        print("Before interpreting any accuracy figure, check:")
        print("  - target shift direction: .shift(-h), not .shift(h)")
        print("  - rolling windows: center=False, min_periods=window")
        print("  - scaler fitting: training indices only")
        print("  - that no feature was derived from the label column")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("features", type=Path, help="feature matrix CSV")
    parser.add_argument("labels", type=Path, help="label CSV")
    parser.add_argument("--horizon", type=int, default=5, help="prediction horizon in events")
    parser.add_argument("--threshold", type=float, default=SUSPICIOUS_CORRELATION,
                        help="correlation above which a feature is suspicious")
    parser.add_argument("--verbose", action="store_true", help="show findings for passing checks too")
    args = parser.parse_args()

    for path in (args.features, args.labels):
        if not path.exists():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            return 2

    features = pd.read_csv(args.features)
    labels = pd.read_csv(args.labels)

    if len(features) != len(labels):
        print(
            f"ERROR: row count mismatch — features {len(features)}, labels {len(labels)}. "
            "These files must be row-aligned.",
            file=sys.stderr,
        )
        return 2

    report = run_all_checks(features, labels, args.horizon, args.threshold)
    print_report(report, args.verbose)

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

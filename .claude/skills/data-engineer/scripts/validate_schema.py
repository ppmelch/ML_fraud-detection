#!/usr/bin/env python3
"""Validate a pipeline artifact against its declared data contract.

Contracts are the boundary between pipeline stages. A stage that accepts
whatever the previous one produced will eventually accept something wrong, and
the error surfaces several stages later as an inexplicable result.

Exits non-zero when validation fails, so it can gate a pipeline.

Usage:
    python validate_schema.py data/raw/events.csv --contract raw_events
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ColumnSpec:
    """One column's contract."""

    dtype: str  # "int", "float", "str", "bool"
    critical: bool = False  # nulls forbidden
    allowed: set | None = None  # categorical whitelist
    min_value: float | None = None
    max_value: float | None = None
    description: str = ""


@dataclass
class Contract:
    """A stage's output contract."""

    name: str
    columns: dict[str, ColumnSpec]
    monotonic_columns: list[str] = field(default_factory=list)
    unique_columns: list[str] = field(default_factory=list)
    description: str = ""


# --------------------------------------------------------------------------
# Contract definitions. These mirror docs/data_contracts.md and the schemas
# declared across the Phase 1-3 issues. Keep the two in sync: this file is
# executable, the markdown is readable, and they must agree.
# --------------------------------------------------------------------------

CONTRACTS: dict[str, Contract] = {
    "raw_events": Contract(
        name="raw_events",
        description="Raw order book event stream as delivered by the exchange or generator",
        columns={
            "timestamp": ColumnSpec("int", critical=True, min_value=0,
                                    description="milliseconds since epoch"),
            "order_id": ColumnSpec("int", critical=True, min_value=0),
            "side": ColumnSpec("str", critical=True, allowed={"B", "S"}),
            "price": ColumnSpec("float", critical=True, min_value=0.0),
            "quantity": ColumnSpec("float", critical=True, min_value=0.0),
            "event_type": ColumnSpec("str", critical=True,
                                     allowed={"add", "cancel", "execute"}),
        },
        monotonic_columns=["timestamp"],
    ),
    "cleaned_events": Contract(
        name="cleaned_events",
        description="Event stream after deduplication, sorting, orphan removal, winsorization",
        columns={
            "timestamp": ColumnSpec("int", critical=True, min_value=0),
            "order_id": ColumnSpec("int", critical=True, min_value=0),
            "side": ColumnSpec("str", critical=True, allowed={"B", "S"}),
            "price": ColumnSpec("float", critical=True, min_value=0.0),
            "quantity": ColumnSpec("float", critical=True, min_value=0.0),
            "event_type": ColumnSpec("str", critical=True,
                                     allowed={"add", "cancel", "execute"}),
        },
        monotonic_columns=["timestamp"],
    ),
    "lob_snapshots": Contract(
        name="lob_snapshots",
        description="Reconstructed order book, one snapshot per event, depth 10 per side",
        columns={
            "timestamp": ColumnSpec("int", critical=True, min_value=0),
            "sequence_number": ColumnSpec("int", critical=True, min_value=0),
            "best_bid": ColumnSpec("float", min_value=0.0),
            "best_ask": ColumnSpec("float", min_value=0.0),
            "mid_price": ColumnSpec("float", min_value=0.0),
            "spread": ColumnSpec("float"),
        },
        monotonic_columns=["timestamp", "sequence_number"],
        unique_columns=["sequence_number"],
    ),
    "ofi_signal": Contract(
        name="ofi_signal",
        description="Order flow imbalance signal with forward-return targets",
        columns={
            "timestamp": ColumnSpec("int", critical=True, min_value=0),
            "sequence_number": ColumnSpec("int", critical=True, min_value=0),
            "ofi_1": ColumnSpec("float", min_value=-1.0, max_value=1.0),
            "ofi_3": ColumnSpec("float", min_value=-1.0, max_value=1.0),
            "ofi_5": ColumnSpec("float", min_value=-1.0, max_value=1.0),
        },
        monotonic_columns=["sequence_number"],
        unique_columns=["sequence_number"],
    ),
    "model_features": Contract(
        name="model_features",
        description="Feature matrix consumed by both models",
        columns={
            "timestamp": ColumnSpec("int", critical=True, min_value=0),
            "sequence_number": ColumnSpec("int", critical=True, min_value=0),
        },
        monotonic_columns=["sequence_number"],
        unique_columns=["sequence_number"],
    ),
    "model_labels": Contract(
        name="model_labels",
        description="Classification labels and raw forward returns",
        columns={
            "timestamp": ColumnSpec("int", critical=True, min_value=0),
            "sequence_number": ColumnSpec("int", critical=True, min_value=0),
        },
        monotonic_columns=["sequence_number"],
        unique_columns=["sequence_number"],
    ),
    "predictions": Contract(
        name="predictions",
        description="Out-of-sample model predictions ready for the simulator",
        columns={
            "timestamp": ColumnSpec("int", critical=True, min_value=0),
            "sequence_number": ColumnSpec("int", critical=True, min_value=0),
            "fold_id": ColumnSpec("int", critical=True, min_value=0),
            "prediction": ColumnSpec("int", critical=True, allowed={0, 1}),
            "confidence": ColumnSpec("float", critical=True, min_value=0.0, max_value=1.0),
            "signal": ColumnSpec("float", critical=True, min_value=-1.0, max_value=1.0),
        },
        monotonic_columns=["sequence_number"],
    ),
    "execution_results": Contract(
        name="execution_results",
        description="Simulated order outcomes with decomposed slippage",
        columns={
            "order_id": ColumnSpec("int", critical=True, min_value=0),
            "side": ColumnSpec("str", critical=True, allowed={"B", "S"}),
            "limit_price": ColumnSpec("float", critical=True, min_value=0.0),
            "quantity": ColumnSpec("float", critical=True, min_value=0.0),
            "decision_timestamp": ColumnSpec("int", critical=True, min_value=0),
            "arrival_timestamp": ColumnSpec("int", critical=True, min_value=0),
            "state": ColumnSpec("str", critical=True),
            "quantity_filled": ColumnSpec("float", critical=True, min_value=0.0),
        },
        unique_columns=["order_id"],
    ),
}


def _dtype_matches(series: pd.Series, expected: str) -> bool:
    """Type check tolerant of pandas' representation choices.

    A column of whole-number floats read from CSV arrives as float64 even when
    the contract says int; comparing dtype strings would reject it. The type
    predicates handle nullable extension dtypes correctly too.
    """
    if expected == "int":
        if pd.api.types.is_integer_dtype(series):
            return True
        # Float column holding only whole numbers satisfies an int contract.
        if pd.api.types.is_float_dtype(series):
            non_null = series.dropna()
            return bool(non_null.empty or np.all(non_null == non_null.astype("int64")))
        return False
    if expected == "float":
        return bool(pd.api.types.is_numeric_dtype(series))
    if expected == "str":
        return bool(pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series))
    if expected == "bool":
        return bool(pd.api.types.is_bool_dtype(series))
    return False


def validate(df: pd.DataFrame, contract: Contract) -> dict:
    """Run every contract check, accumulating all failures."""
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}

    # --- required columns
    missing = [c for c in contract.columns if c not in df.columns]
    checks["columns_present"] = not missing
    if missing:
        errors.append(f"[columns_present] missing required columns: {missing}")

    # --- unexpected columns are informational, not failures
    extra = [c for c in df.columns if c not in contract.columns]
    if extra:
        warnings.append(f"[extra_columns] not in contract (allowed): {extra[:10]}")

    present = [c for c in contract.columns if c in df.columns]

    # --- dtypes
    dtype_failures = []
    for col in present:
        if not _dtype_matches(df[col], contract.columns[col].dtype):
            dtype_failures.append(f"{col} (expected {contract.columns[col].dtype}, got {df[col].dtype})")
    checks["dtypes_correct"] = not dtype_failures
    if dtype_failures:
        errors.append(f"[dtypes_correct] {len(dtype_failures)} column(s): {dtype_failures[:5]}")

    # --- nulls in critical columns
    null_failures = []
    for col in present:
        if contract.columns[col].critical:
            n_null = int(df[col].isna().sum())
            if n_null:
                idx = df.index[df[col].isna()][:5].tolist()
                null_failures.append(f"{col}: {n_null} nulls, examples at {idx}")
    checks["no_nulls_critical"] = not null_failures
    if null_failures:
        errors.append(f"[no_nulls_critical] {null_failures}")

    # --- categorical whitelists
    allowed_failures = []
    for col in present:
        spec = contract.columns[col]
        if spec.allowed is not None:
            actual = set(df[col].dropna().unique())
            unexpected = actual - spec.allowed
            if unexpected:
                allowed_failures.append(f"{col}: unexpected values {sorted(unexpected)[:5]}")
    checks["allowed_values"] = not allowed_failures
    if allowed_failures:
        errors.append(f"[allowed_values] {allowed_failures}")

    # --- numeric ranges
    range_failures = []
    for col in present:
        spec = contract.columns[col]
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if spec.min_value is not None:
            n = int((df[col] < spec.min_value).sum())
            if n:
                range_failures.append(f"{col}: {n} rows below {spec.min_value}")
        if spec.max_value is not None:
            n = int((df[col] > spec.max_value).sum())
            if n:
                range_failures.append(f"{col}: {n} rows above {spec.max_value}")
    checks["value_ranges"] = not range_failures
    if range_failures:
        errors.append(f"[value_ranges] {range_failures}")

    # --- monotonicity
    mono_failures = []
    for col in contract.monotonic_columns:
        if col not in df.columns:
            continue
        diffs = df[col].diff().dropna()
        n_back = int((diffs < 0).sum())
        if n_back:
            mono_failures.append(f"{col}: {n_back} backwards transitions")
    checks["monotonic"] = not mono_failures
    if mono_failures:
        errors.append(f"[monotonic] {mono_failures}")

    # --- uniqueness
    unique_failures = []
    for col in contract.unique_columns:
        if col not in df.columns:
            continue
        n_dup = int(df[col].duplicated().sum())
        if n_dup:
            unique_failures.append(f"{col}: {n_dup} duplicate values")
    checks["unique"] = not unique_failures
    if unique_failures:
        errors.append(f"[unique] {unique_failures}")

    # --- fully duplicated rows
    n_dup_rows = int(df.duplicated().sum())
    checks["no_duplicate_rows"] = n_dup_rows == 0
    if n_dup_rows:
        errors.append(f"[no_duplicate_rows] {n_dup_rows} fully duplicated rows")

    return {
        "contract": contract.name,
        "passed": not errors,
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }


def print_report(report: dict) -> None:
    print("=" * 74)
    print(f"SCHEMA VALIDATION — contract: {report['contract']}")
    print("=" * 74)
    print(f"Rows: {report['n_rows']}   Columns: {report['n_columns']}")
    print()

    for name, passed in report["checks"].items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print()

    if report["warnings"]:
        print("Warnings:")
        for w in report["warnings"]:
            print(f"  {w}")
        print()

    if report["errors"]:
        print("Errors:")
        for e in report["errors"]:
            print(f"  {e}")
        print()

    print("=" * 74)
    print("RESULT:", "contract satisfied" if report["passed"] else "CONTRACT VIOLATED")
    print("=" * 74)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", type=Path, help="CSV to validate")
    parser.add_argument("--contract", required=True, choices=sorted(CONTRACTS),
                        help="contract to validate against")
    parser.add_argument("--output", type=Path, help="write the report as JSON")
    args = parser.parse_args()

    if not args.file.exists():
        print(f"ERROR: file not found: {args.file}", file=sys.stderr)
        return 2

    df = pd.read_csv(args.file)
    report = validate(df, CONTRACTS[args.contract])
    report["file"] = str(args.file)
    print_report(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(f"\nReport written to {args.output}")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

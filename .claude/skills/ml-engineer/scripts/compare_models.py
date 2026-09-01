#!/usr/bin/env python3
"""Compare two models fold by fold, with statistics a small sample supports.

Six folds give six paired observations. A Wilcoxon test on six pairs has very
limited power, and a mean difference of one percentage point with a standard
deviation of three is not evidence of anything. This script reports the test,
the win count, and the effect size together, and refuses to declare a winner
when the difference sits inside fold-to-fold noise.

Expects metrics files with one row per fold and a `fold_id` column.

Usage:
    python compare_models.py ofi_metrics.csv deeplab_metrics.csv \\
        --metric balanced_accuracy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Below this many folds, any test is underpowered and the report says so
# rather than quoting a p-value as though it settled the question.
MIN_FOLDS_FOR_INFERENCE = 5


def load_fold_metrics(path: Path, metric: str, horizon: int | None) -> pd.DataFrame:
    """Read a metrics file, keeping one row per fold."""
    df = pd.read_csv(path)

    if metric not in df.columns:
        raise ValueError(f"metric '{metric}' not in {path.name}; available: {list(df.columns)}")

    if "fold_id" not in df.columns:
        raise ValueError(f"no 'fold_id' column in {path.name} — per-fold rows are required")

    # Aggregate rows (mean/std summaries) are excluded from a paired test.
    df = df[pd.to_numeric(df["fold_id"], errors="coerce").notna()].copy()
    df["fold_id"] = df["fold_id"].astype(int)

    if horizon is not None and "horizon" in df.columns:
        df = df[df["horizon"] == horizon]

    return df[["fold_id", metric]].sort_values("fold_id").reset_index(drop=True)


def paired_comparison(a: pd.DataFrame, b: pd.DataFrame, metric: str, alpha: float) -> dict:
    """Compare on the folds both models were evaluated on."""
    merged = a.merge(b, on="fold_id", suffixes=("_a", "_b"))
    if merged.empty:
        raise ValueError("no folds in common between the two files")

    col_a, col_b = f"{metric}_a", f"{metric}_b"
    diffs = (merged[col_b] - merged[col_a]).to_numpy(dtype=float)
    n = len(diffs)

    mean_a, mean_b = float(merged[col_a].mean()), float(merged[col_b].mean())
    std_a, std_b = float(merged[col_a].std(ddof=1)), float(merged[col_b].std(ddof=1))
    mean_diff = float(diffs.mean())
    std_diff = float(diffs.std(ddof=1)) if n > 1 else np.nan

    wins_b = int((diffs > 0).sum())
    wins_a = int((diffs < 0).sum())
    ties = int((diffs == 0).sum())

    # Wilcoxon needs at least one non-zero difference and is meaningless on
    # very few pairs; report it, but flagged.
    if n >= 3 and np.any(diffs != 0):
        try:
            stat, p_value = stats.wilcoxon(diffs)
            p_value = float(p_value)
        except ValueError:
            stat, p_value = np.nan, np.nan
    else:
        stat, p_value = np.nan, np.nan

    # Paired-difference confidence interval via the t distribution. With six
    # folds this interval is wide, which is the honest picture.
    if n > 1 and not np.isnan(std_diff) and std_diff > 0:
        se = std_diff / np.sqrt(n)
        crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
        ci = (mean_diff - crit * se, mean_diff + crit * se)
        cohens_d = mean_diff / std_diff
    else:
        ci = (np.nan, np.nan)
        cohens_d = np.nan

    # The difference is only meaningful if it is large relative to how much
    # the metric varies between folds in the first place.
    pooled_fold_std = float(np.nanmean([std_a, std_b]))
    inside_noise = (not np.isnan(pooled_fold_std)) and abs(mean_diff) < pooled_fold_std

    underpowered = n < MIN_FOLDS_FOR_INFERENCE
    significant = (not np.isnan(p_value)) and p_value < alpha

    return {
        "metric": metric,
        "n_folds": n,
        "mean_a": mean_a, "std_a": std_a,
        "mean_b": mean_b, "std_b": std_b,
        "mean_difference": mean_diff,
        "std_difference": std_diff,
        "ci_low": ci[0], "ci_high": ci[1],
        "cohens_d": cohens_d,
        "wins_a": wins_a, "wins_b": wins_b, "ties": ties,
        "wilcoxon_stat": stat, "p_value": p_value,
        "significant": significant,
        "underpowered": underpowered,
        "inside_fold_noise": inside_noise,
        "pooled_fold_std": pooled_fold_std,
        "per_fold": merged.assign(difference=diffs),
    }


def print_report(r: dict, name_a: str, name_b: str, alpha: float) -> None:
    print("=" * 74)
    print(f"PAIRED MODEL COMPARISON — {r['metric']}")
    print("=" * 74)
    print(f"  A: {name_a}")
    print(f"  B: {name_b}")
    print(f"  Folds compared: {r['n_folds']}")
    print()

    print("Per fold")
    pf = r["per_fold"].copy()
    for c in pf.columns:
        if c != "fold_id":
            pf[c] = pf[c].round(4)
    print(pf.to_string(index=False))
    print()

    print("Aggregate")
    print(f"  A mean {r['mean_a']:.4f}  (sd {r['std_a']:.4f})")
    print(f"  B mean {r['mean_b']:.4f}  (sd {r['std_b']:.4f})")
    print(f"  difference (B - A)  {r['mean_difference']:+.4f}  (sd {r['std_difference']:.4f})")
    if not np.isnan(r["ci_low"]):
        print(f"  {int((1-alpha)*100)}% CI  [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")
        print(f"  Cohen's d  {r['cohens_d']:+.3f}")
    print()

    print("Fold outcomes")
    print(f"  B better in {r['wins_b']} fold(s), A better in {r['wins_a']}, ties {r['ties']}")
    if not np.isnan(r["p_value"]):
        print(f"  Wilcoxon signed-rank p = {r['p_value']:.4f}")
    else:
        print("  Wilcoxon not computed (too few folds or all differences zero)")
    print()

    print("-" * 74)

    if r["underpowered"]:
        print(f"UNDERPOWERED: {r['n_folds']} folds is too few for a meaningful test.")
        print("A non-significant result here is not evidence of equivalence, and a")
        print("significant one rests on very little. Report the fold values themselves.")
        print()

    if r["inside_fold_noise"]:
        print("VERDICT: no distinguishable difference.")
        print(f"  The mean difference ({r['mean_difference']:+.4f}) is smaller than the")
        print(f"  typical fold-to-fold variation ({r['pooled_fold_std']:.4f}). Whatever")
        print("  gap exists is inside the noise these folds already show.")
    elif r["significant"] and not r["underpowered"]:
        winner = name_b if r["mean_difference"] > 0 else name_a
        print(f"VERDICT: {winner} is better on {r['metric']}.")
        print(f"  Difference {abs(r['mean_difference']):.4f}, p = {r['p_value']:.4f},")
        print(f"  winning {max(r['wins_a'], r['wins_b'])} of {r['n_folds']} folds.")
        print()
        print("  Before acting on this: is the gain worth the extra complexity, and")
        print("  does it survive the execution costs measured in Phase 3?")
    else:
        print("VERDICT: difference not statistically distinguishable.")
        print(f"  Mean difference {r['mean_difference']:+.4f}, p = {r['p_value']:.4f}.")
        print("  Report both models' numbers; do not declare a winner.")

    print("-" * 74)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_a", type=Path, help="metrics CSV for model A")
    parser.add_argument("model_b", type=Path, help="metrics CSV for model B")
    parser.add_argument("--metric", default="balanced_accuracy", help="metric column to compare")
    parser.add_argument("--horizon", type=int, help="restrict to one horizon")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", type=Path, help="write per-fold comparison to CSV")
    args = parser.parse_args()

    for p in (args.model_a, args.model_b):
        if not p.exists():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            return 2

    try:
        a = load_fold_metrics(args.model_a, args.metric, args.horizon)
        b = load_fold_metrics(args.model_b, args.metric, args.horizon)
        result = paired_comparison(a, b, args.metric, args.alpha)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print_report(result, args.model_a.stem, args.model_b.stem, args.alpha)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result["per_fold"].to_csv(args.output, index=False)
        print(f"\nPer-fold comparison written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

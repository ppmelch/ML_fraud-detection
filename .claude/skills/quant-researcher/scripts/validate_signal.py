#!/usr/bin/env python3
"""Test whether a candidate signal carries information about future returns.

Runs the full signal-by-horizon grid with the corrections a multi-hypothesis
screen requires: Benjamini-Hochberg across all pairs, a split-half stability
check, and a shuffled-target control that must return null.

The shuffled control is the important one. If a permuted target still shows
correlation, the pipeline is manufacturing signal and nothing else in the
output means anything.

Usage:
    python validate_signal.py signal.csv \\
        --signal-cols ofi_1 ofi_3 --target-cols price_movement_1 price_movement_3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# On high-frequency order-book data a single feature rarely exceeds this.
# Below it, an effect is real but likely too small to survive execution costs.
MEANINGFUL_EFFECT = 0.05

# How far above the theoretical null the shuffled control may drift before we
# treat it as evidence that the pipeline manufactures correlation.
#
# Under the null, Spearman rho is approximately N(0, 1/(n-1)), so the expected
# absolute value is sqrt(2 / (pi * (n-1))) — roughly 0.025 at n=1000. A fixed
# threshold would therefore fail on large samples and pass on small ones; the
# tolerance has to scale with n.
CONTROL_NULL_MULTIPLE = 2.5


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    """Return BH-adjusted p-values and rejection flags.

    Controls false discovery rate rather than family-wise error, which suits an
    exploratory screen: we accept some false positives in exchange for power,
    but we do not accept the ~37% chance of at least one spurious "significant"
    result that nine uncorrected tests at alpha=0.05 would carry.

    Implemented directly rather than importing statsmodels, so the skill has no
    dependency beyond scipy.
    """
    n = len(p_values)
    if n == 0:
        return [], []

    order = np.argsort(p_values)
    ranked = np.asarray(p_values, dtype=float)[order]

    # BH step-up: adjusted_i = min over j>=i of (n/j) * p_j
    adjusted_sorted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_sorted

    return adjusted.tolist(), (adjusted <= alpha).tolist()


def correlate(signal: pd.Series, target: pd.Series) -> dict:
    """Spearman and Pearson correlation on pairwise-complete observations.

    Spearman leads because forward returns are heavy-tailed and the
    relationship is unlikely to be linear; a handful of extreme returns would
    dominate Pearson.
    """
    mask = signal.notna() & target.notna()
    n = int(mask.sum())

    if n < 30:
        return {"n": n, "spearman": np.nan, "pearson": np.nan,
                "p_value": np.nan, "note": "insufficient overlap"}

    s, s_p = stats.spearmanr(signal[mask], target[mask])
    p, _ = stats.pearsonr(signal[mask], target[mask])

    return {"n": n, "spearman": float(s), "pearson": float(p), "p_value": float(s_p), "note": ""}


def split_half_stability(signal: pd.Series, target: pd.Series) -> dict:
    """Correlation in each half of the sample.

    An effect present in the first half and absent in the second is regime
    dependence, not an edge. Phase 2 would train on a relationship that does
    not persist, and the backtest would land in whichever regime it lands in.
    """
    mid = len(signal) // 2
    first = correlate(signal.iloc[:mid], target.iloc[:mid])
    second = correlate(signal.iloc[mid:], target.iloc[mid:])

    both_valid = not (np.isnan(first["spearman"]) or np.isnan(second["spearman"]))
    same_sign = both_valid and np.sign(first["spearman"]) == np.sign(second["spearman"])

    return {
        "first_half_spearman": first["spearman"],
        "second_half_spearman": second["spearman"],
        "sign_consistent": bool(same_sign),
        "divergence": abs(first["spearman"] - second["spearman"]) if both_valid else np.nan,
    }


def shuffled_control(signal: pd.Series, target: pd.Series, seed: int, n_trials: int = 200) -> dict:
    """Correlate the signal against permuted targets.

    Permuting destroys any real relationship, so the shuffled correlations
    should follow the null distribution. Two things are learned:

    1. Whether the pipeline manufactures correlation. If shuffled results sit
       well above the theoretical null, the measurement code produces signal
       independently of the data, and nothing else in the output is meaningful.

    2. A permutation p-value for the observed correlation, which makes no
       distributional assumption and is more trustworthy than the parametric
       one on heavy-tailed financial data.
    """
    mask = signal.notna() & target.notna()
    n = int(mask.sum())

    if n < 30:
        return {"observed_abs": np.nan, "null_mean_abs": np.nan, "null_expected": np.nan,
                "permutation_p": np.nan, "passed": True, "note": "insufficient overlap"}

    s = signal[mask].to_numpy()
    t = target[mask].to_numpy()

    observed, _ = stats.spearmanr(s, t)
    observed_abs = abs(float(observed))

    rng = np.random.default_rng(seed)
    null_abs = np.empty(n_trials)
    for i in range(n_trials):
        rho, _ = stats.spearmanr(s, rng.permutation(t))
        null_abs[i] = abs(float(rho))

    # E|rho| under the null, for rho ~ N(0, 1/(n-1))
    null_expected = float(np.sqrt(2.0 / (np.pi * (n - 1))))
    null_mean = float(null_abs.mean())

    # Permutation p-value with the +1 correction, so it is never exactly zero.
    perm_p = float((np.sum(null_abs >= observed_abs) + 1) / (n_trials + 1))

    inflated = null_mean > CONTROL_NULL_MULTIPLE * null_expected

    return {
        "observed_abs": observed_abs,
        "null_mean_abs": null_mean,
        "null_expected": null_expected,
        "null_p95": float(np.percentile(null_abs, 95)),
        "permutation_p": perm_p,
        "n_trials": n_trials,
        "passed": not inflated,
        "note": "" if not inflated else
                f"CONTROL FAILED — shuffled mean {null_mean:.4f} exceeds "
                f"{CONTROL_NULL_MULTIPLE}x the theoretical null {null_expected:.4f}",
    }


def run_grid(
    df: pd.DataFrame,
    signal_cols: list[str],
    target_cols: list[str],
    alpha: float,
    seed: int,
) -> pd.DataFrame:
    """Full signal-by-target grid with corrections and controls."""
    rows = []
    for sig in signal_cols:
        for tgt in target_cols:
            base = correlate(df[sig], df[tgt])
            stability = split_half_stability(df[sig], df[tgt])
            control = shuffled_control(df[sig], df[tgt], seed)

            rows.append(
                {
                    "signal": sig,
                    "target": tgt,
                    "n": base["n"],
                    "spearman": base["spearman"],
                    "pearson": base["pearson"],
                    "p_value_raw": base["p_value"],
                    "p_value_permutation": control["permutation_p"],
                    "first_half": stability["first_half_spearman"],
                    "second_half": stability["second_half_spearman"],
                    "sign_consistent": stability["sign_consistent"],
                    "null_mean_abs": control["null_mean_abs"],
                    "null_expected": control["null_expected"],
                    "control_passed": control["passed"],
                }
            )

    results = pd.DataFrame(rows)

    # Correct the permutation p-values rather than the parametric ones: they
    # make no distributional assumption, which matters on heavy-tailed returns.
    valid = results["p_value_permutation"].notna()
    results["p_value_adjusted"] = np.nan
    results["significant"] = False

    if valid.any():
        adj, rejected = benjamini_hochberg(results.loc[valid, "p_value_permutation"].tolist(), alpha)
        results.loc[valid, "p_value_adjusted"] = adj
        results.loc[valid, "significant"] = rejected

    results["meaningful"] = results["spearman"].abs() >= MEANINGFUL_EFFECT
    results["usable"] = results["significant"] & results["meaningful"] & results["sign_consistent"]

    return results


def print_report(results: pd.DataFrame, alpha: float) -> None:
    """Human-readable summary."""
    print("=" * 78)
    print("SIGNAL VALIDATION")
    print("=" * 78)
    print(f"Grid: {len(results)} signal-target pairs")
    print(f"Significance: permutation p-values, BH-corrected at alpha={alpha}")
    print()

    display = results[
        ["signal", "target", "n", "spearman", "p_value_permutation", "p_value_adjusted",
         "sign_consistent", "usable"]
    ].copy()
    for col in ("spearman", "p_value_permutation", "p_value_adjusted"):
        display[col] = display[col].round(4)
    print(display.to_string(index=False))
    print()

    control_failures = results[~results["control_passed"]]
    if not control_failures.empty:
        print("!" * 78)
        print("SHUFFLED CONTROL FAILED")
        print()
        print("Permuted targets correlate more strongly than the null distribution")
        print("predicts. The pipeline is producing signal where none exists.")
        print()
        for _, r in control_failures.head(5).iterrows():
            print(f"  {r['signal']} -> {r['target']}: "
                  f"shuffled mean |rho|={r['null_mean_abs']:.4f}, "
                  f"theoretical null={r['null_expected']:.4f}")
        print()
        print("Stop and find the alignment error before interpreting anything above.")
        print("!" * 78)
        print()
        return

    usable = results[results["usable"]]
    print("-" * 78)
    if usable.empty:
        print("RESULT: no signal-target pair is usable.")
        print()
        print("A pair is usable when it is significant after correction, reaches an")
        print(f"effect size of {MEANINGFUL_EFFECT}, and keeps its sign across both halves.")
        print()
        print("This is a legitimate finding, not a failure to work around. Do not")
        print("lower the threshold to manufacture a result.")
    else:
        print(f"RESULT: {len(usable)} usable pair(s)")
        best = usable.reindex(usable["spearman"].abs().sort_values(ascending=False).index).iloc[0]
        print(f"  strongest: {best['signal']} -> {best['target']}  "
              f"rho={best['spearman']:.4f}  p_adj={best['p_value_adjusted']:.4g}  n={best['n']}")
        print()
        print("Carry the configuration into docs/methodology.md before Phase 2.")
    print("-" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("signal_file", type=Path, help="CSV containing signal and target columns")
    parser.add_argument("--signal-cols", nargs="+", required=True, help="signal column names")
    parser.add_argument("--target-cols", nargs="+", required=True, help="target column names")
    parser.add_argument("--alpha", type=float, default=0.05, help="FDR level")
    parser.add_argument("--seed", type=int, default=42, help="seed for the shuffled control")
    parser.add_argument("--output", type=Path, help="write results CSV here")
    args = parser.parse_args()

    if not args.signal_file.exists():
        print(f"ERROR: file not found: {args.signal_file}", file=sys.stderr)
        return 2

    df = pd.read_csv(args.signal_file)

    missing = [c for c in args.signal_cols + args.target_cols if c not in df.columns]
    if missing:
        print(f"ERROR: columns not in file: {missing}", file=sys.stderr)
        print(f"available: {list(df.columns)}", file=sys.stderr)
        return 2

    results = run_grid(df, args.signal_cols, args.target_cols, args.alpha, args.seed)
    print_report(results, args.alpha)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
        print(f"\nResults written to {args.output}")

    if not results["control_passed"].all():
        return 2
    return 0 if results["usable"].any() else 1


if __name__ == "__main__":
    sys.exit(main())

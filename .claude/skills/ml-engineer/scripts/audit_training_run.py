#!/usr/bin/env python3
"""Diagnose a training run from its epoch log.

A final metric says whether a run succeeded. The log says why it did or did
not, and the failure modes below are invisible in a summary: a model that
settled on the base rate, one that overfit after epoch 12, one whose early
stopping never fired because the monitored metric was noisy.

Expects a CSV with one row per epoch. Column names are matched loosely, so
logs from different training loops are readable without renaming.

Usage:
    python audit_training_run.py output/deeplab_training_log_h5_fold3.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Loss changing by less than this fraction over the whole run means the model
# never engaged with the problem.
FLAT_LOSS_TOLERANCE = 0.02

# Validation loss rising for this many consecutive epochs while training loss
# falls is the classic overfitting signature.
OVERFIT_PATIENCE = 3

# Train metric exceeding validation by more than this suggests memorization
# rather than generalization.
GAP_WARNING = 0.10

# Epoch-to-epoch loss swings above this fraction of the mean indicate a
# learning rate the optimizer cannot settle with.
OSCILLATION_TOLERANCE = 0.25


def find_column(df: pd.DataFrame, *candidates: str) -> str | None:
    """Match a column loosely, so logs from different loops are readable."""
    lowered = {c.lower().replace("-", "_"): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace("-", "_")
        if key in lowered:
            return lowered[key]
    # substring fallback
    for cand in candidates:
        key = cand.lower().replace("-", "_")
        for low, orig in lowered.items():
            if key in low:
                return orig
    return None


def check_learned_anything(train_loss: pd.Series) -> dict:
    """Loss must move. A flat curve means the model never engaged."""
    if len(train_loss) < 2:
        return {"check": "learned_anything", "passed": True, "note": "too few epochs"}

    first, last = float(train_loss.iloc[0]), float(train_loss.iloc[-1])
    if first == 0:
        return {"check": "learned_anything", "passed": False,
                "note": "initial loss is zero — suspicious"}

    rel_change = (first - last) / abs(first)
    passed = rel_change > FLAT_LOSS_TOLERANCE

    return {
        "check": "learned_anything",
        "passed": passed,
        "initial_loss": round(first, 6),
        "final_loss": round(last, 6),
        "relative_improvement": round(rel_change, 4),
        "note": "" if passed else
                "training loss barely moved — check learning rate, gradient flow, "
                "and that the optimizer actually sees the parameters",
    }


def check_divergence(train_loss: pd.Series) -> dict:
    """NaN or a rising trend means the run diverged."""
    n_nan = int(train_loss.isna().sum())
    n_inf = int(np.isinf(train_loss.fillna(0)).sum())

    rising = False
    if len(train_loss) >= 5:
        early = float(train_loss.iloc[: len(train_loss) // 3].mean())
        late = float(train_loss.iloc[-len(train_loss) // 3:].mean())
        rising = late > early * 1.1

    passed = n_nan == 0 and n_inf == 0 and not rising

    note = ""
    if n_nan or n_inf:
        note = ("loss became NaN or inf — exploding gradient. Add clipping, "
                "lower the learning rate, check inputs for inf")
    elif rising:
        note = "training loss trending upward — learning rate likely too high"

    return {"check": "no_divergence", "passed": passed,
            "nan_epochs": n_nan, "inf_epochs": n_inf, "rising_trend": rising, "note": note}


def check_overfitting(train_loss: pd.Series, val_loss: pd.Series) -> dict:
    """Validation rising while training falls, for several epochs running."""
    if val_loss is None or len(val_loss) < OVERFIT_PATIENCE + 1:
        return {"check": "overfitting", "passed": True, "note": "insufficient epochs"}

    train_d = train_loss.diff()
    val_d = val_loss.diff()

    diverging = (val_d > 0) & (train_d < 0)

    # longest consecutive run
    longest = current = 0
    onset = None
    for i, flag in enumerate(diverging.fillna(False).to_numpy()):
        if flag:
            current += 1
            if current > longest:
                longest = current
                onset = i - current + 1
        else:
            current = 0

    passed = longest < OVERFIT_PATIENCE

    return {
        "check": "overfitting",
        "passed": passed,
        "longest_divergence_run": longest,
        "onset_epoch": onset,
        "best_val_epoch": int(val_loss.idxmin()) if not val_loss.isna().all() else None,
        "final_val_epoch": len(val_loss) - 1,
        "note": "" if passed else
                f"validation loss rose for {longest} consecutive epochs while training "
                f"loss fell, from epoch {onset}. Compare the parameter budget against "
                "the sample count in docs/deeplab_architecture.md",
    }


def check_early_stopping(val_loss: pd.Series) -> dict:
    """Did it stop at the right time, or run past the best epoch?"""
    if val_loss is None or val_loss.isna().all():
        return {"check": "early_stopping", "passed": True, "note": "no validation loss logged"}

    best_epoch = int(val_loss.idxmin())
    total = len(val_loss)
    wasted = total - 1 - best_epoch

    # Best epoch at the very end means it was still improving when it stopped.
    still_improving = best_epoch >= total - 2
    ran_long_past_best = wasted > total * 0.5 and total > 10

    note = ""
    if still_improving and total > 3:
        note = ("best epoch is the last one — the run stopped while still improving. "
                "Raise the epoch limit or the patience")
    elif ran_long_past_best:
        note = (f"ran {wasted} epochs past the best. Patience may be too high, "
                "or the monitored metric too noisy")

    return {
        "check": "early_stopping",
        "passed": not (still_improving or ran_long_past_best),
        "best_epoch": best_epoch,
        "total_epochs": total,
        "epochs_after_best": wasted,
        "note": note,
    }


def check_learning_rate(train_loss: pd.Series) -> dict:
    """Oscillation suggests too high; a crawl suggests too low."""
    if len(train_loss) < 5:
        return {"check": "learning_rate", "passed": True, "note": "too few epochs"}

    diffs = train_loss.diff().dropna()
    mean_loss = float(train_loss.mean())
    if mean_loss == 0:
        return {"check": "learning_rate", "passed": True, "note": "degenerate loss"}

    volatility = float(diffs.abs().mean() / abs(mean_loss))
    sign_flips = int((np.sign(diffs).diff().fillna(0) != 0).sum())
    flip_rate = sign_flips / len(diffs)

    oscillating = volatility > OSCILLATION_TOLERANCE and flip_rate > 0.5

    note = ""
    if oscillating:
        note = "loss oscillates epoch to epoch — learning rate likely too high"

    return {"check": "learning_rate", "passed": not oscillating,
            "volatility": round(volatility, 4), "sign_flip_rate": round(flip_rate, 3),
            "note": note}


def check_base_rate_only(df: pd.DataFrame, baseline_accuracy: float | None) -> dict:
    """A model matching the majority class learned nothing useful."""
    acc_col = find_column(df, "val_accuracy", "valid_accuracy", "accuracy", "val_acc")
    bal_col = find_column(df, "val_balanced_accuracy", "balanced_accuracy")

    if acc_col is None:
        return {"check": "base_rate_only", "passed": True, "note": "no accuracy logged"}

    final_acc = float(df[acc_col].iloc[-1])

    result = {"check": "base_rate_only", "passed": True,
              "final_accuracy": round(final_acc, 4), "note": ""}

    if baseline_accuracy is not None:
        margin = final_acc - baseline_accuracy
        result["baseline_accuracy"] = baseline_accuracy
        result["margin_over_baseline"] = round(margin, 4)
        if margin < 0.01:
            result["passed"] = False
            result["note"] = (f"accuracy {final_acc:.4f} is within 1pp of the majority-class "
                              f"baseline {baseline_accuracy:.4f} — the model may have learned "
                              "only the class prior")

    if bal_col is not None:
        bal = float(df[bal_col].iloc[-1])
        result["final_balanced_accuracy"] = round(bal, 4)
        if bal < 0.52:
            result["passed"] = False
            result["note"] = (result["note"] + " " if result["note"] else "") + \
                f"balanced accuracy {bal:.4f} is near chance"

    return result


def check_train_val_gap(df: pd.DataFrame) -> dict:
    """A large gap indicates memorization."""
    train_acc = find_column(df, "train_accuracy", "train_acc")
    val_acc = find_column(df, "val_accuracy", "valid_accuracy", "val_acc")

    if train_acc is None or val_acc is None:
        return {"check": "train_val_gap", "passed": True, "note": "accuracy not logged for both splits"}

    gap = float(df[train_acc].iloc[-1] - df[val_acc].iloc[-1])
    passed = gap <= GAP_WARNING

    return {
        "check": "train_val_gap",
        "passed": passed,
        "final_gap": round(gap, 4),
        "note": "" if passed else
                f"training accuracy exceeds validation by {gap:.3f}. Compare the "
                "parameter count against the sample count; consider more dropout, "
                "weight decay, or a smaller hidden dimension",
    }


def audit(df: pd.DataFrame, baseline_accuracy: float | None) -> dict:
    train_col = find_column(df, "train_loss", "loss", "training_loss")
    val_col = find_column(df, "val_loss", "valid_loss", "validation_loss")

    if train_col is None:
        return {"passed": False, "error": "no training loss column found",
                "available_columns": list(df.columns), "checks": []}

    train_loss = df[train_col]
    val_loss = df[val_col] if val_col else None

    checks = [
        check_learned_anything(train_loss),
        check_divergence(train_loss),
        check_learning_rate(train_loss),
        check_base_rate_only(df, baseline_accuracy),
        check_train_val_gap(df),
    ]
    if val_loss is not None:
        checks.append(check_overfitting(train_loss, val_loss))
        checks.append(check_early_stopping(val_loss))

    return {
        "passed": all(c["passed"] for c in checks),
        "n_epochs": len(df),
        "train_loss_column": train_col,
        "val_loss_column": val_col,
        "checks": checks,
    }


def print_report(report: dict) -> None:
    print("=" * 74)
    print("TRAINING RUN AUDIT")
    print("=" * 74)

    if "error" in report:
        print(f"ERROR: {report['error']}")
        print(f"Available columns: {report['available_columns']}")
        return

    print(f"Epochs: {report['n_epochs']}   "
          f"train loss: {report['train_loss_column']}   "
          f"val loss: {report['val_loss_column'] or 'not logged'}")
    print()

    for c in report["checks"]:
        status = "PASS" if c["passed"] else "WARN"
        print(f"[{status}] {c['check']}")
        for k, v in c.items():
            if k in ("check", "passed", "note"):
                continue
            print(f"       {k}: {v}")
        if c.get("note"):
            for line in _wrap(c["note"], 66):
                print(f"       -> {line}")
        print()

    print("=" * 74)
    print("RESULT:", "no issues detected" if report["passed"] else "ISSUES FOUND — see warnings above")
    print("=" * 74)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_file", type=Path, help="per-epoch training log CSV")
    parser.add_argument("--baseline-accuracy", type=float,
                        help="majority-class accuracy, to detect a model that learned only the prior")
    args = parser.parse_args()

    if not args.log_file.exists():
        print(f"ERROR: file not found: {args.log_file}", file=sys.stderr)
        return 2

    df = pd.read_csv(args.log_file)
    report = audit(df, args.baseline_accuracy)
    print_report(report)

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

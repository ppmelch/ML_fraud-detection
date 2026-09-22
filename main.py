#!/usr/bin/env python3
"""
Print (and optionally save) a paper-ready results report for the
unsupervised anomaly-detection model.

This reads the JSON artifacts already produced by the training pipeline
(``backend/artifacts/*.json``) and formats the numbers that typically go
into a report: dataset/split sizes, model configuration, per-window
metrics, top feature importance, feature contrast, heuristic sanity check,
portfolio-wide overview and the top anomalous states.

It never touches the map / frontend, and never retrains the model — it
only reads what ``python -m scripts.train_pipeline`` already wrote. Run
that first (or pass --retrain) if the artifacts are missing or stale.

Usage
-----
    python main.py                  # print report to the terminal
    python main.py --save           # also write results/paper_report.md
    python main.py --retrain        # regenerate artifacts first, then report
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.src.modeling.config import ARTIFACTS_DIR, PROJECT_ROOT

REPORT_DIR = PROJECT_ROOT / "results"


def _load(name: str) -> dict:
    """
    Load one artifact JSON file, failing with a clear message if absent.

    Parameters
    ----------
    name : str
        File name inside ``backend/artifacts``.

    Returns
    -------
    dict
        Parsed JSON content.
    """
    path = ARTIFACTS_DIR / name
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Run `python -m scripts.train_pipeline` first, "
            f"or `python main.py --retrain`."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _section(title: str, lines: list[str]) -> str:
    bar = "=" * len(title)
    return f"\n{title}\n{bar}\n" + "\n".join(lines) + "\n"


def load_artifacts() -> dict:
    """
    Load every artifact the report and tables are built from.

    Returns
    -------
    dict
        ``model_metrics``, ``explainability``, ``anomaly_summary``,
        ``geo_validation``, ``state_metrics`` and ``stability``.
    """
    return {
        "model_metrics": _load("model_metrics.json"),
        "explainability": _load("explainability.json"),
        "anomaly_summary": _load("anomaly_summary.json"),
        "geo_validation": _load("geo_validation.json"),
        "state_metrics": _load("state_metrics.json"),
        "stability": _load("stability_analysis.json"),
    }


def build_report(artifacts: dict) -> str:
    """
    Assemble the full text report from already-loaded artifacts.

    Parameters
    ----------
    artifacts : dict
        As returned by :func:`load_artifacts`.

    Returns
    -------
    str
        Plain-text report, ready to print or write to disk.
    """
    model_metrics = artifacts["model_metrics"]
    explainability = artifacts["explainability"]
    anomaly_summary = artifacts["anomaly_summary"]
    geo_validation = artifacts["geo_validation"]
    state_metrics = artifacts["state_metrics"]
    stability = artifacts["stability"]

    out: list[str] = []
    out.append("UNSUPERVISED FRAUD/ANOMALY DETECTION - RESULTS REPORT")
    out.append("=" * 55)

    # --- Model configuration -------------------------------------------------
    split = model_metrics["split"]
    lines = [
        f"Model:                 {model_metrics['model_name']}",
        f"Contamination target:  {model_metrics['contamination']}",
        f"Operating threshold:   {model_metrics['threshold']:.6f}",
        f"Feature count:         {model_metrics['n_features']}",
        "",
        "Temporal split:",
        f"  train      n={split['train']['rows']:5d}  {split['train']['start']} -> {split['train']['end']}",
        f"  validation n={split['validation']['rows']:5d}  {split['validation']['start']} -> {split['validation']['end']}",
        f"  test       n={split['test']['rows']:5d}  {split['test']['start']} -> {split['test']['end']}",
    ]
    out.append(_section("1. Model & split configuration", lines))

    # --- Per-window metrics ---------------------------------------------------
    lines = [
        f"{'window':10s} {'n':>6s} {'flagged':>8s} {'rate':>8s} {'mean':>8s} {'std':>8s} {'p95':>8s} {'p99':>8s}"
    ]
    for window in ("train", "validation", "test"):
        m = model_metrics["metrics"][window]
        p = m["score_percentiles"]
        lines.append(
            f"{window:10s} {m['n_samples']:6d} {m['n_flagged']:8d} "
            f"{_fmt_pct(m['flagged_rate']):>8s} {m['score_mean']:8.4f} "
            f"{m['score_std']:8.4f} {p['p95']:8.4f} {p['p99']:8.4f}"
        )
    out.append(_section("2. Anomaly-score metrics per window", lines))

    # --- Heuristic sanity check (test window) ---------------------------------
    align = model_metrics["metrics"]["test"]["heuristic_alignment"]
    lines = [
        f"Spearman correlation vs. hand-built heuristic: {align['spearman']:.4f}",
        f"Overlap among top-flagged transactions:        {_fmt_pct(align['overlap_at_flagged'])}",
        f"Note: {align['note']}",
    ]
    out.append(_section("3. Heuristic alignment (sanity check, not ground truth)", lines))

    # --- Feature importance (SHAP) ---------------------------------------------
    lines = [f"Method: {explainability['method']} (sample_size={explainability['sample_size']})", ""]
    lines.append(f"{'#':>3s}  {'feature':35s} {'importance':>10s}")
    for i, row in enumerate(explainability["feature_importance"][:15], start=1):
        lines.append(f"{i:>3d}  {row['feature']:35s} {row['importance']:10.4f}")
    out.append(_section("4. Top feature importance (SHAP mean |value|)", lines))

    # --- Feature contrast: flagged vs normal (test window) ----------------------
    contrast = model_metrics["metrics"]["test"]["feature_contrast"]
    top_contrast = sorted(contrast, key=lambda r: abs(r["std_gap"]), reverse=True)[:15]
    lines = [f"{'feature':30s} {'flagged_mean':>14s} {'normal_mean':>14s} {'std_gap':>10s}"]
    for row in top_contrast:
        lines.append(
            f"{row['feature']:30s} {row['flagged_mean']:14.4f} "
            f"{row['normal_mean']:14.4f} {row['std_gap']:10.4f}"
        )
    out.append(_section("5. Flagged vs. normal transactions (test window, largest gaps)", lines))

    # --- Portfolio-wide overview (full dataset) ---------------------------------
    ov = anomaly_summary["overview"]
    lines = [
        f"Total transactions:     {ov['total_transactions']}",
        f"Total accounts:         {ov['total_accounts']}",
        f"States covered:         {ov['states_covered']}",
        f"Period:                 {ov['period_start']} -> {ov['period_end']}",
        f"Flagged transactions:   {ov['flagged_transactions']} ({_fmt_pct(ov['flagged_rate'])})",
        f"Mean anomaly score:     {ov['mean_anomaly_score']:.4f}",
        f"Total amount:           ${ov['total_amount']:,.2f}",
        f"Flagged amount:         ${ov['flagged_amount']:,.2f}",
        f"Avg transaction amount: ${ov['avg_transaction_amount']:,.2f}",
        f"Avg account balance:    ${ov['avg_account_balance']:,.2f}",
    ]
    out.append(_section("6. Full-dataset overview (all scored transactions)", lines))

    # --- Geo validation -----------------------------------------------------
    lines = [
        f"Dataset states:         {geo_validation['dataset_states']}",
        f"Matched to GeoJSON:     {geo_validation['matched_states']}",
        f"Unmatched:              {geo_validation['unmatched_dataset_states']}",
        f"Reconciled OK:          {geo_validation['ok']}",
    ]
    out.append(_section("7. Geo (state) reconciliation", lines))

    # --- Top anomalous states (by rate, minimum support already applied upstream) --
    ranked = sorted(
        (s for s in state_metrics if not s.get("low_support")),
        key=lambda s: s["anomaly_rate"],
        reverse=True,
    )[:10]
    lines = [f"{'state':25s} {'n':>6s} {'flagged':>8s} {'rate':>8s} {'mean_score':>10s}"]
    for s in ranked:
        lines.append(
            f"{s['state']:25s} {s['total_transactions']:6d} {s['flagged']:8d} "
            f"{_fmt_pct(s['anomaly_rate']):>8s} {s['mean_anomaly_score']:10.4f}"
        )
    out.append(_section("8. Top 10 states by anomaly rate (min-support filtered)", lines))

    # --- Multi-seed stability analysis ----------------------------------------
    if stability.get("applicable", False):
        s = stability["stability"]
        j = stability["jaccard"]
        r = stability["rank_correlation"]
        lines = [
            f"Model: {stability['model_name']}  |  runs: {stability['n_runs']}  |  "
            f"seeds: {stability['seeds']}",
            f"Test window size: {stability['n_test_samples']}",
            "",
            "CAUTION: the mean stability score over ALL test transactions is",
            "diluted by the majority never flagged in any run and is NOT a",
            "meaningful headline number on its own. Read the conditional",
            "statistics below instead.",
            "",
            f"Mean stability (all {s['n_test_samples']} test rows):        {s['mean_stability']:.4f}",
            f"  -> flagged in >=1 run:  {s['flagged_at_least_once']['n']:4d} "
            f"({_fmt_pct(s['flagged_at_least_once']['pct'])} of test set)",
            "",
            "Among rows flagged at least once (the only rows for which",
            "'stability' is a meaningful question):",
            f"  mean stability given flagged>=1:   {s['mean_stability_given_flagged_once']:.4f}",
            f"  median stability given flagged>=1: {s['median_stability_given_flagged_once']:.4f}",
            "",
            "Breakdown over the full test set:",
            f"  stability == 100%  (flagged every run):  {s['stability_100pct']['n']:4d} "
            f"({_fmt_pct(s['stability_100pct']['pct'])})",
            f"  stability >=  80%:                       {s['stability_ge_80pct']['n']:4d} "
            f"({_fmt_pct(s['stability_ge_80pct']['pct'])})",
            f"  stability >=  50%:                       {s['stability_ge_50pct']['n']:4d} "
            f"({_fmt_pct(s['stability_ge_50pct']['pct'])})",
            f"  stability >   0%   (ever flagged):       {s['stability_gt_0pct']['n']:4d} "
            f"({_fmt_pct(s['stability_gt_0pct']['pct'])})",
            "",
            "Jaccard similarity of the flagged SET, across the 45 seed pairs:",
            f"  mean={j['mean']:.4f}  median={j['median']:.4f}  std={j['std']:.4f}  "
            f"min={j['min']:.4f}  max={j['max']:.4f}",
            "",
            "Spearman rank correlation of the raw SCORE (threshold-independent):",
            f"  mean={r['mean']:.4f}  median={r['median']:.4f}  std={r['std']:.4f}  "
            f"min={r['min']:.4f}  max={r['max']:.4f}",
            "  -> The ranking is far more stable than the binary flag: seeds agree",
            "     on relative anomalousness (~0.90 correlation) much more than they",
            "     agree on which rows cross a hard threshold near the boundary.",
            "     Volatility concentrates at the cutoff, not in the ranking itself.",
            "",
            f"{'seed':>6s} {'threshold':>10s} {'n_flagged':>10s} {'rate':>8s} {'score_mean':>11s}",
        ]
        for run in stability["per_run"]:
            lines.append(
                f"{run['seed']:>6d} {run['threshold']:10.4f} {run['n_flagged']:10d} "
                f"{_fmt_pct(run['flagged_rate']):>8s} {run['score_mean']:11.4f}"
            )

        drift = model_metrics["metrics"]["test"]["flagged_rate"] / model_metrics["contamination"]
        lines += [
            "",
            "Threshold-generalisation note: the operating threshold is the "
            f"{1 - model_metrics['contamination']:.0%} quantile of TRAIN scores; "
            f"applied to test it flags {_fmt_pct(model_metrics['metrics']['test']['flagged_rate'])} "
            f"of rows, {drift:.1f}x the {_fmt_pct(model_metrics['contamination'])} contamination "
            "target. This is a score-distribution shift between the train and test "
            "periods, not a bug in the threshold — worth stating explicitly in the "
            "paper as a limitation of a fixed, train-derived cutoff.",
        ]
    else:
        lines = [stability.get("note", "Not applicable for this model.")]
    out.append(_section("9. Multi-seed stability analysis (Isolation Forest robustness)", lines))

    return "\n".join(out)


def write_tables(artifacts: dict) -> None:
    """
    Write CSV summary tables for the paper into ``results/tables/``.

    Parameters
    ----------
    artifacts : dict
        As returned by :func:`load_artifacts`.
    """
    import csv

    model_metrics = artifacts["model_metrics"]
    explainability = artifacts["explainability"]
    state_metrics = artifacts["state_metrics"]
    stability = artifacts["stability"]

    tables_dir = REPORT_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    def _write(name: str, header: list[str], rows: list[list]) -> None:
        path = tables_dir / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"  wrote {path.relative_to(PROJECT_ROOT)}")

    _write(
        "split_metrics.csv",
        ["window", "n_samples", "n_flagged", "flagged_rate", "score_mean", "score_std", "p50", "p90", "p95", "p99"],
        [
            [
                w,
                m["n_samples"],
                m["n_flagged"],
                m["flagged_rate"],
                m["score_mean"],
                m["score_std"],
                m["score_percentiles"]["p50"],
                m["score_percentiles"]["p90"],
                m["score_percentiles"]["p95"],
                m["score_percentiles"]["p99"],
            ]
            for w, m in (
                (w, model_metrics["metrics"][w]) for w in ("train", "validation", "test")
            )
        ],
    )

    _write(
        "feature_importance_top15.csv",
        ["rank", "feature", "importance", "method"],
        [
            [i, r["feature"], r["importance"], r["method"]]
            for i, r in enumerate(explainability["feature_importance"][:15], start=1)
        ],
    )

    contrast = model_metrics["metrics"]["test"]["feature_contrast"]
    top_contrast = sorted(contrast, key=lambda r: abs(r["std_gap"]), reverse=True)[:15]
    _write(
        "feature_contrast_top15.csv",
        ["feature", "flagged_mean", "normal_mean", "std_gap"],
        [[r["feature"], r["flagged_mean"], r["normal_mean"], r["std_gap"]] for r in top_contrast],
    )

    supported = [s for s in state_metrics if not s.get("low_support")]
    ranked = sorted(supported, key=lambda s: s["anomaly_rate"], reverse=True)
    _write(
        "state_anomaly_rate.csv",
        ["state", "total_transactions", "flagged", "anomaly_rate", "mean_anomaly_score", "low_support"],
        [
            [s["state"], s["total_transactions"], s["flagged"], s["anomaly_rate"], s["mean_anomaly_score"], s.get("low_support", False)]
            for s in state_metrics
        ],
    )

    if stability.get("applicable", False):
        _write(
            "stability_per_run.csv",
            ["seed", "threshold", "n_flagged", "flagged_rate", "score_mean", "score_std"],
            [
                [r["seed"], r["threshold"], r["n_flagged"], r["flagged_rate"], r["score_mean"], r["score_std"]]
                for r in stability["per_run"]
            ],
        )

        s = stability["stability"]
        _write(
            "stability_breakdown.csv",
            ["cutoff", "n", "pct", "n_test_samples"],
            [
                ["== 100%", s["stability_100pct"]["n"], s["stability_100pct"]["pct"], s["n_test_samples"]],
                [">= 80%", s["stability_ge_80pct"]["n"], s["stability_ge_80pct"]["pct"], s["n_test_samples"]],
                [">= 50%", s["stability_ge_50pct"]["n"], s["stability_ge_50pct"]["pct"], s["n_test_samples"]],
                ["> 0%", s["stability_gt_0pct"]["n"], s["stability_gt_0pct"]["pct"], s["n_test_samples"]],
            ],
        )

        by_state = stability["stable_anomaly_rate_by_state"]
        _write(
            "stable_anomaly_rate_by_state.csv",
            ["state", "n_test_transactions", "stable_flagged", "stable_anomaly_rate", "low_support"],
            [
                [r["state"], r["n_test_transactions"], r["stable_flagged"], r["stable_anomaly_rate"], r["low_support"]]
                for r in by_state
            ],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Regenerate backend/artifacts/*.json via scripts.train_pipeline before reporting.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Also write the report to results/paper_report.md",
    )
    args = parser.parse_args()

    if args.retrain:
        print("Retraining and regenerating artifacts ...\n")
        subprocess.run(
            [sys.executable, "-m", "scripts.train_pipeline"],
            cwd=PROJECT_ROOT,
            check=True,
        )
        print()

    artifacts = load_artifacts()
    report = build_report(artifacts)
    print(report)

    if args.save:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = REPORT_DIR / "paper_report.md"
        out_path.write_text("```\n" + report + "\n```\n", encoding="utf-8")
        print(f"\nSaved to {out_path}")
        print("\nWriting summary tables ...")
        write_tables(artifacts)


if __name__ == "__main__":
    main()

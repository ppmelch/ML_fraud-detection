#!/usr/bin/env python3
"""
Render publication-quality static figures from the pipeline's JSON artifacts.

Every figure is built strictly from ``backend/artifacts/*.json`` (written by
``python -m scripts.train_pipeline``) plus the existing US-states GeoJSON
already used by the interactive map (``frontend/data/us-states.json``). This
script never retrains, never recomputes a metric, and never touches the
frontend — it only re-renders numbers that already exist, as PNGs meant to
go straight into the paper.

Usage
-----
    python -m scripts.generate_figures
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.src.modeling.config import (
    ARTIFACTS_DIR,
    GEOJSON_PATH,
    MIN_TRANSACTIONS_PER_HOUR_BUCKET,
    MIN_TRANSACTIONS_PER_STATE,
    PROJECT_ROOT,
)

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

# One consistent qualitative palette across every figure.
COLOR_NORMAL = "#4C72B0"
COLOR_FLAGGED = "#C44E52"
COLOR_TRAIN = "#4C72B0"
COLOR_VALID = "#55A868"
COLOR_TEST = "#C44E52"
CMAP_SEQUENTIAL = "YlOrRd"
CMAP_DIVERGING = "coolwarm"

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)


def _load(name: str) -> dict:
    path = ARTIFACTS_DIR / name
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Run `python -m scripts.train_pipeline` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"{name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(PROJECT_ROOT)}")
    return out_path


# ---------------------------------------------------------------------------
# 1. Anomaly score distribution with the operating threshold
# ---------------------------------------------------------------------------
def fig_score_distribution(model_metrics: dict) -> None:
    test = model_metrics["metrics"]["test"]
    hist = test["score_histogram"]
    threshold = model_metrics["threshold"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = hist["bin_centers"][1] - hist["bin_centers"][0]
    colors = [COLOR_FLAGGED if c >= threshold else COLOR_NORMAL for c in hist["bin_centers"]]
    ax.bar(hist["bin_centers"], hist["counts"], width=width * 0.95, color=colors, edgecolor="white", linewidth=0.3)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"Threshold = {threshold:.3f}")

    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Transactions (test window)")
    ax.set_title("Anomaly score distribution and operating threshold")
    ax.legend(frameon=False)
    sns.despine(fig)
    _save(fig, "01_score_distribution_threshold")


# ---------------------------------------------------------------------------
# 2. Train / validation / test anomaly-rate comparison
# ---------------------------------------------------------------------------
def fig_split_rate_comparison(model_metrics: dict) -> None:
    windows = ["train", "validation", "test"]
    colors = [COLOR_TRAIN, COLOR_VALID, COLOR_TEST]
    rates = [model_metrics["metrics"][w]["flagged_rate"] * 100 for w in windows]
    ns = [model_metrics["metrics"][w]["n_samples"] for w in windows]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(windows, rates, color=colors, width=0.55)
    for bar, n, rate in zip(bars, ns, rates):
        ax.annotate(
            f"{rate:.2f}%\n(n={n})",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.axhline(model_metrics["contamination"] * 100, color="black", linestyle=":", linewidth=1.2,
               label=f"Contamination target = {model_metrics['contamination'] * 100:.0f}%")
    ax.set_ylabel("Flagged rate (%)")
    ax.set_title("Flagged rate by temporal window")
    ax.legend(frameon=False)
    sns.despine(fig)
    _save(fig, "02_split_anomaly_rate")


# ---------------------------------------------------------------------------
# 3. Per-transaction stability distribution
#
# 92.9% of test rows are never flagged in any of the 10 runs, so a single
# linear histogram over all rows is dominated by one bar at 0 and says
# nothing about the transactions that actually matter. This figure splits
# the two questions the raw distribution conflates: how many rows are ever
# a candidate at all (left, log-scaled so the dominant zero bar does not
# blot out the rest), and — among only those candidates — how consistently
# they get flagged (right, the informative distribution).
# ---------------------------------------------------------------------------
def fig_stability_distribution(stability: dict) -> None:
    if not stability.get("applicable", False):
        print(f"  skipped 03_stability_distribution: {stability.get('note')}")
        return

    s = stability["stability"]
    hist_all = s["histogram"]
    hist_cond = s["histogram_given_flagged"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    width = hist_all["bin_centers"][1] - hist_all["bin_centers"][0]
    axes[0].bar(hist_all["bin_centers"], hist_all["counts"], width=width * 0.9, color="#9aa3ab", edgecolor="white")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Stability score (fraction of 10 seeds flagging the row)")
    axes[0].set_ylabel("Test transactions (log scale)")
    axes[0].set_title(f"All {s['n_test_samples']} test rows")
    axes[0].text(
        0.98, 0.95,
        f"{s['flagged_at_least_once']['pct'] * 100:.1f}% never flagged in\nany of the 10 runs",
        transform=axes[0].transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7"),
    )

    width_c = hist_cond["bin_centers"][1] - hist_cond["bin_centers"][0]
    axes[1].bar(hist_cond["bin_centers"], hist_cond["counts"], width=width_c * 0.9, color=COLOR_FLAGGED, edgecolor="white")
    axes[1].axvline(s["median_stability_given_flagged_once"], color="black", linestyle="--", linewidth=1.3,
                     label=f"Median = {s['median_stability_given_flagged_once']:.2f}")
    axes[1].set_xlabel("Stability score")
    axes[1].set_ylabel("Transactions")
    axes[1].set_title(f"Rows flagged at least once (n={s['flagged_at_least_once']['n']})")
    axes[1].legend(frameon=False, fontsize=9)

    fig.suptitle("Isolation Forest detection stability across 10 seeds — overall exposure vs. consistency among candidates")
    sns.despine(fig)
    _save(fig, "03_stability_distribution")


# ---------------------------------------------------------------------------
# 4. Anomaly score vs. stability
# ---------------------------------------------------------------------------
def fig_score_vs_stability(stability: dict) -> None:
    if not stability.get("applicable", False):
        print(f"  skipped 04_score_vs_stability: {stability.get('note')}")
        return

    s = stability["stability"]
    scores = np.array(s["mean_anomaly_score"])
    stab = np.array(s["stability_score"])

    # Stability is quantised to multiples of 1/n_runs (10 seeds -> steps of
    # 0.1), so many rows share the exact same y-value; a small fixed-seed
    # jitter is added for display only so overplotted points stay visible.
    rng = np.random.default_rng(0)
    stab_jittered = stab + rng.uniform(-0.015, 0.015, size=stab.shape)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sc = ax.scatter(scores, stab_jittered, c=stab, cmap=CMAP_SEQUENTIAL, s=22, alpha=0.75, edgecolor="none")
    ax.set_xlabel("Mean anomaly score across seeds")
    ax.set_ylabel("Stability score (fraction of seeds flagged; jittered for display)")
    ax.set_title("Anomaly score vs. detection stability (test window, n=560)")
    ax.set_ylim(-0.05, 1.05)
    fig.colorbar(sc, ax=ax, label="Stability score")
    sns.despine(fig)
    _save(fig, "04_score_vs_stability")


# ---------------------------------------------------------------------------
# 5. Jaccard similarity across runs
#
# The heatmap alone answers "which seed pairs agree least"; the right panel
# answers the sharper methodological question: is the *binary flag* really
# less stable than the model's *ranking*? Jaccard depends on where each
# run's own threshold lands, so it is compared directly against the
# threshold-independent Spearman rank correlation of the raw scores.
# ---------------------------------------------------------------------------
def fig_jaccard_heatmap(stability: dict) -> None:
    if not stability.get("applicable", False):
        print(f"  skipped 05_jaccard_similarity: {stability.get('note')}")
        return

    matrix = np.array(stability["jaccard"]["matrix"])
    seeds = stability["seeds"]
    jaccard_pairs = stability["jaccard"]["pairs"]
    rank_pairs = stability["rank_correlation"]["pairs"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    sns.heatmap(
        matrix, ax=axes[0], cmap="viridis", vmin=0, vmax=1, square=True,
        xticklabels=seeds, yticklabels=seeds,
        cbar_kws={"label": "Jaccard similarity"}, annot=matrix.shape[0] <= 10, fmt=".2f",
    )
    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("Seed")
    mean_j = stability["jaccard"]["mean"]
    axes[0].set_title(f"Flagged-set Jaccard similarity\n(mean = {mean_j:.3f}, 45 seed pairs)")

    bins = np.linspace(0.5, 1.0, 21)
    axes[1].hist(jaccard_pairs, bins=bins, alpha=0.75, color=COLOR_FLAGGED, label=f"Jaccard (flag)\nmean={np.mean(jaccard_pairs):.3f}")
    axes[1].hist(rank_pairs, bins=bins, alpha=0.75, color=COLOR_NORMAL, label=f"Spearman ρ (rank)\nmean={np.mean(rank_pairs):.3f}")
    axes[1].set_xlabel("Pairwise similarity")
    axes[1].set_ylabel("Seed pairs (of 45)")
    axes[1].set_title("Flag agreement vs. rank agreement")
    axes[1].legend(frameon=False, fontsize=9)

    fig.suptitle("Cross-run agreement: the model ranks transactions more consistently than it flags them")
    sns.despine(fig)
    _save(fig, "05_jaccard_similarity")


# ---------------------------------------------------------------------------
# 6. SHAP feature importance
# ---------------------------------------------------------------------------
def fig_feature_importance(explainability: dict, top_n: int = 15) -> None:
    rows = explainability["feature_importance"][:top_n][::-1]
    features = [r["feature"] for r in rows]
    importance = [r["importance"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 0.35 * len(rows) + 1.5))
    ax.barh(features, importance, color=COLOR_FLAGGED)
    ax.set_xlabel(f"Mean |SHAP value| ({explainability['method']})")
    ax.set_title(f"Top {top_n} features driving the anomaly score")
    sns.despine(fig)
    _save(fig, "06_feature_importance")


# ---------------------------------------------------------------------------
# 7. Flagged vs. normal feature contrast
# ---------------------------------------------------------------------------
def fig_feature_contrast(model_metrics: dict, top_n: int = 15) -> None:
    contrast = model_metrics["metrics"]["test"]["feature_contrast"]
    top = sorted(contrast, key=lambda r: abs(r["std_gap"]), reverse=True)[:top_n][::-1]
    features = [r["feature"] for r in top]
    gaps = [r["std_gap"] for r in top]
    colors = [COLOR_FLAGGED if g > 0 else COLOR_NORMAL for g in gaps]

    fig, ax = plt.subplots(figsize=(7.5, 0.35 * len(top) + 1.5))
    ax.barh(features, gaps, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Standardised mean gap (flagged − normal, in pooled std units)")
    ax.set_title(f"Top {top_n} behavioural differences: flagged vs. normal (test window)")
    sns.despine(fig)
    _save(fig, "07_feature_contrast")


# ---------------------------------------------------------------------------
# 8. Temporal anomaly patterns
# ---------------------------------------------------------------------------
def fig_temporal_patterns(anomaly_summary: dict) -> None:
    by_hour = anomaly_summary["by_hour"]
    by_day = anomaly_summary["by_day_of_week"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    hours = [b["bucket"] for b in by_hour]
    hour_rates = [b["anomaly_rate"] * 100 for b in by_hour]
    hour_low_support = [b["total_transactions"] < MIN_TRANSACTIONS_PER_HOUR_BUCKET for b in by_hour]
    colors_h = ["#cccccc" if low else COLOR_FLAGGED for low in hour_low_support]
    axes[0].bar(hours, hour_rates, color=colors_h)
    axes[0].set_xlabel("Hour of day")
    axes[0].set_ylabel("Anomaly rate (%)")
    axes[0].set_title("By hour")
    axes[0].set_xticks(range(0, 24, 2))

    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_day_map = {b["bucket"]: b for b in by_day}
    day_rates = [by_day_map[d]["anomaly_rate"] * 100 if d in by_day_map else 0 for d in days_order]
    axes[1].bar([d[:3] for d in days_order], day_rates, color=COLOR_FLAGGED)
    axes[1].set_xlabel("Day of week")
    axes[1].set_ylabel("Anomaly rate (%)")
    axes[1].set_title("By day of week")

    fig.suptitle("Temporal anomaly patterns (full dataset; grey bars fall below the "
                  f"{MIN_TRANSACTIONS_PER_HOUR_BUCKET}-transaction support floor)")
    sns.despine(fig)
    _save(fig, "08_temporal_patterns")


# ---------------------------------------------------------------------------
# Shared US-choropleth renderer (contiguous 48 states + DC), from the
# project's own GeoJSON — nothing here touches the Leaflet map.
# ---------------------------------------------------------------------------
_EXCLUDED_FROM_MAP_VIEW = {"Alaska", "Hawaii", "Puerto Rico"}


def _ring_to_path_args(ring: list) -> tuple[list, list]:
    verts = list(ring)
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(ring) - 2) + [MplPath.CLOSEPOLY]
    return verts, codes


def _polygon_to_path(polygon: list) -> MplPath:
    all_verts: list = []
    all_codes: list = []
    for ring in polygon:
        verts, codes = _ring_to_path_args(ring)
        all_verts.extend(verts)
        all_codes.extend(codes)
    return MplPath(all_verts, all_codes)


def _draw_choropleth(ax, geojson: dict, value_by_state: dict, low_support_states: set, cmap: str, vmin: float, vmax: float) -> None:
    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap)

    for feature in geojson["features"]:
        name = feature["properties"]["name"]
        if name in _EXCLUDED_FROM_MAP_VIEW:
            continue

        geometry = feature["geometry"]
        polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        value = value_by_state.get(name)
        facecolor = colormap(norm(value)) if value is not None else "#e8e8e8"
        hatch = "///" if name in low_support_states else None

        for polygon in polygons:
            path = _polygon_to_path(polygon)
            patch = PathPatch(path, facecolor=facecolor, edgecolor="white", linewidth=0.5, hatch=hatch)
            ax.add_patch(patch)

    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)
    ax.set_aspect(1.3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ---------------------------------------------------------------------------
# 9. US state anomaly-rate map
# ---------------------------------------------------------------------------
def fig_state_anomaly_map(state_metrics: list, geojson: dict) -> None:
    values = {s["state"]: s["anomaly_rate"] * 100 for s in state_metrics}
    low_support = {s["state"] for s in state_metrics if s.get("low_support")}

    fig, ax = plt.subplots(figsize=(11, 6))
    vmax = max(values.values())
    _draw_choropleth(ax, geojson, values, low_support, CMAP_SEQUENTIAL, 0, vmax)

    sm = plt.cm.ScalarMappable(cmap=CMAP_SEQUENTIAL, norm=Normalize(0, vmax))
    fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, label="Anomaly rate (%)")
    ax.set_title(
        f"Anomaly rate by US state (full dataset; hatched = below "
        f"{MIN_TRANSACTIONS_PER_STATE}-transaction support floor; "
        "AK/HI/PR excluded from the map view)"
    )
    _save(fig, "09_state_anomaly_rate_map")


# ---------------------------------------------------------------------------
# 10. Stable-anomaly rate by state, if the data supports it
#
# The stable-anomaly set (flagged in all 10 seeds) only exists on the test
# window: 560 rows across up to 51 states, ~11 rows/state on average. Only
# a handful of states clear the project's own 15-transaction support floor.
# The map is still rendered for completeness (every state is shown, low-
# support ones hatched), but the only numbers worth citing in the paper are
# the supported ones, so they are also listed directly on the figure.
# ---------------------------------------------------------------------------
def fig_stable_rate_by_state(stability: dict, geojson: dict) -> None:
    if not stability.get("applicable", False):
        print(f"  skipped 10_stable_anomaly_rate_by_state_map: {stability.get('note')}")
        return

    by_state = stability["stable_anomaly_rate_by_state"]
    values = {s["state"]: s["stable_anomaly_rate"] * 100 for s in by_state}
    low_support = {s["state"] for s in by_state if s.get("low_support")}
    supported = sorted(
        (s for s in by_state if not s.get("low_support") and s["n_test_transactions"] > 0),
        key=lambda s: -s["stable_anomaly_rate"],
    )

    fig, ax = plt.subplots(figsize=(11, 6.5))
    vmax = max(values.values()) if values else 1.0
    _draw_choropleth(ax, geojson, values, low_support, CMAP_SEQUENTIAL, 0, vmax if vmax > 0 else 1.0)

    sm = plt.cm.ScalarMappable(cmap=CMAP_SEQUENTIAL, norm=Normalize(0, vmax if vmax > 0 else 1.0))
    fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, label="Stable-anomaly rate (%)")

    if supported:
        lines = [f"n≥{MIN_TRANSACTIONS_PER_STATE} states (only ones with real support):"]
        lines += [f"  {s['state']}: {s['stable_anomaly_rate'] * 100:.1f}% (n={s['n_test_transactions']})" for s in supported]
    else:
        lines = [f"No state reaches the n≥{MIN_TRANSACTIONS_PER_STATE} support floor in the test window."]
    ax.text(
        0.01, 0.02, "\n".join(lines), transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.9),
    )

    ax.set_title(
        "Stable-anomaly rate by state — flagged in all "
        f"{stability['n_runs']} seed runs (test window only, n={stability['n_test_samples']}; "
        f"hatched = below {MIN_TRANSACTIONS_PER_STATE}-transaction support floor)"
    )
    _save(fig, "10_stable_anomaly_rate_by_state_map")


def main() -> None:
    print("Loading artifacts ...")
    model_metrics = _load("model_metrics.json")
    explainability = _load("explainability.json")
    anomaly_summary = _load("anomaly_summary.json")
    state_metrics = _load("state_metrics.json")
    stability = _load("stability_analysis.json")
    geojson = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))

    print("Rendering figures ...")
    fig_score_distribution(model_metrics)
    fig_split_rate_comparison(model_metrics)
    fig_stability_distribution(stability)
    fig_score_vs_stability(stability)
    fig_jaccard_heatmap(stability)
    fig_feature_importance(explainability)
    fig_feature_contrast(model_metrics)
    fig_temporal_patterns(anomaly_summary)
    fig_state_anomaly_map(state_metrics, geojson)
    fig_stable_rate_by_state(stability, geojson)

    print(f"\nAll figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()

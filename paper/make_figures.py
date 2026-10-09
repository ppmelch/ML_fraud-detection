#!/usr/bin/env python3
"""
Build the figure set used by ``paper/main.tex``.

Every number is read from ``backend/artifacts/*.json``. The only model run is a
re-execution of the unchanged :class:`AnomalyPipeline` (seed 42), needed for
two views the artifacts do not store: the train-vs-test score distributions
and the real-vs-synthetic / monthly breakdown of the full-dataset scoring.
No configuration is altered.

Usage
-----
    python -m paper.make_figures
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch, Patch, PathPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.src.modeling.config import (  # noqa: E402
    ARTIFACTS_DIR,
    DATASET_PATH,
    GEOJSON_PATH,
    PROJECT_ROOT,
)
from backend.src.pipeline import AnomalyPipeline  # noqa: E402
from scripts.generate_figures import _EXCLUDED_FROM_MAP_VIEW, _polygon_to_path  # noqa: E402

FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"

# Grayscale palette, used consistently across every figure.
NAVY = "#4A4A4A"      # main / normal / train / real
WINE = "#111111"      # emphasis / flagged / test
FOREST = "#8A8A8A"    # validation / temporal
CHARCOAL = "#2E2E2E"  # text, reference lines
PLUM = "#6E6E6E"
OCHRE = "#B8B8B8"     # synthetic / demographic
SLATE = "#9C9C9C"
DARK_CMAP = LinearSegmentedColormap.from_list("gray_rate", ["#C4C4C4", "#7A7A7A", "#2E2E2E", "#000000"])

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.edgecolor": CHARCOAL,
    "axes.labelcolor": CHARCOAL,
    "xtick.color": CHARCOAL,
    "ytick.color": CHARCOAL,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#D0D0D0",
    "grid.linewidth": 0.5,
})


def _load(name: str):
    return json.loads((ARTIFACTS_DIR / name).read_text(encoding="utf-8"))


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote paper/figures/{name}.pdf/.png")


def _draw_states(ax, geojson: dict, color_of) -> None:
    """Fill each contiguous-US polygon with ``color_of(state_name)``."""
    for feature in geojson["features"]:
        name = feature["properties"]["name"]
        if name in _EXCLUDED_FROM_MAP_VIEW:
            continue
        geometry = feature["geometry"]
        polygons = (
            geometry["coordinates"]
            if geometry["type"] == "MultiPolygon"
            else [geometry["coordinates"]]
        )
        for polygon in polygons:
            ax.add_patch(PathPatch(
                _polygon_to_path(polygon), facecolor=color_of(name),
                edgecolor="white", linewidth=0.5,
            ))
    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)
    ax.set_aspect(1.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


# ---------------------------------------------------------------------------
# Data needed beyond the artifacts
# ---------------------------------------------------------------------------
def _rerun() -> tuple[dict, AnomalyPipeline, pd.DataFrame]:
    pipeline = AnomalyPipeline()
    result = pipeline.run()
    result["train_scores"] = pipeline.model.anomaly_score(result["X_train"])

    raw = pd.read_csv(DATASET_PATH)
    scored = pipeline.score_frame(raw)
    scored["synthetic"] = scored["TransactionID"].str.startswith("TXS")
    ts = pd.to_datetime(scored["Transaction_Timestamp"])
    split = result["split"]
    scored["window"] = np.select(
        [ts <= pd.Timestamp(split["train"]["end"]),
         ts <= pd.Timestamp(split["validation"]["end"])],
        ["Train", "Validation"], default="Test",
    )
    scored["period"] = ts.dt.to_period("M")
    return result, pipeline, scored


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_pipeline() -> None:
    steps = [
        ("Transactions\n3,293 rows\n(no fraud label)", NAVY),
        ("Temporal split\ntrain / val / test", SLATE),
        ("Feature\nengineering\n85 features", SLATE),
        ("Isolation Forest\n300 trees\nthreshold = P98", WINE),
        ("Anomaly score\n+ flag", PLUM),
    ]
    analyses = [
        "Temporal\ndrift", "Stability\n(10 seeds)", "Behavioral\ncontrast",
        "SHAP", "US state\nmap",
    ]
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    w, h, y = 1.7, 1.15, 1.85
    for i, (label, color) in enumerate(steps):
        x = 0.1 + i * 2.0
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                                    facecolor=color, edgecolor="none"))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", color="white", fontsize=8)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 2.0, y + h / 2), xytext=(x + w, y + h / 2),
                        arrowprops=dict(arrowstyle="-|>", color=CHARCOAL, lw=1.1))
    bw, by = 1.7, 0.1
    for i, label in enumerate(analyses):
        x = 0.1 + i * 2.0
        ax.add_patch(FancyBboxPatch((x, by), bw, 0.95, boxstyle="round,pad=0.03,rounding_size=0.12",
                                    facecolor="#5E5E5E", edgecolor="none"))
        ax.text(x + bw / 2, by + 0.475, label, ha="center", va="center", color="white", fontsize=8)
        ax.annotate("", xy=(x + bw / 2, by + 0.98), xytext=(8.95, y - 0.02),
                    arrowprops=dict(arrowstyle="-|>", color=CHARCOAL, lw=0.7,
                                    connectionstyle="arc3,rad=0.0", alpha=0.7))
    _save(fig, "fig01_pipeline")


def fig_dataset_map(scored: pd.DataFrame, geojson: dict) -> None:
    from backend.src.modeling.config import CITY_TO_STATE
    scored = scored.assign(state=scored["Location"].map(CITY_TO_STATE))
    synthetic_states = set(scored.loc[scored["synthetic"], "state"])
    n_real = int((~scored["synthetic"]).sum())
    n_syn = int(scored["synthetic"].sum())

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    _draw_states(ax, geojson, lambda s: OCHRE if s in synthetic_states else NAVY)
    ax.legend(handles=[
        Patch(color=NAVY, label=f"Real Kaggle transactions (27 states, n={n_real:,})"),
        Patch(color=OCHRE, label=f"Synthetic coverage rows (25 states, n={n_syn:,})"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, fontsize=7.5)
    ax.set_title("Dataset composition by state (contiguous US shown)")
    _save(fig, "fig02_dataset_map")


def fig_score_threshold(result: dict) -> None:
    thr = result["threshold"]
    bins = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.hist(result["train_scores"], bins=bins, density=True, color=NAVY, alpha=0.85,
            label="Train (Jan–Oct 2023)")
    ax.hist(result["test_scores"], bins=bins, density=True, histtype="step", lw=1.6,
            color=WINE, label="Test (Nov 2023–Jan 2024)")
    ax.axvline(thr, color=CHARCOAL, ls="--", lw=1.2)
    ax.text(thr + 0.01, ax.get_ylim()[1] * 0.92, f"threshold\n= {thr:.3f}", fontsize=7.5,
            color=CHARCOAL, va="top")
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7.5)
    _save(fig, "fig03_score_threshold")


def fig_split_rates(model_metrics: dict, stability: dict) -> None:
    m = model_metrics["metrics"]
    names = ["Train", "Validation", "Test"]
    rates = [m["train"]["flagged_rate"] * 100, m["validation"]["flagged_rate"] * 100,
             m["test"]["flagged_rate"] * 100]
    seeds = [r["flagged_rate"] * 100 for r in stability["per_run"]]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    bars = ax.bar(names, rates, color=[NAVY, FOREST, WINE], width=0.6)
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.15, f"{r:.2f}%", ha="center", fontsize=8)
    ax.errorbar([2], [np.mean(seeds)], yerr=[[np.mean(seeds) - min(seeds)], [max(seeds) - np.mean(seeds)]],
                fmt="o", color=CHARCOAL, ms=3, capsize=3, lw=1)
    ax.text(2.33, min(seeds), "range over\n10 seeds", fontsize=7, color=CHARCOAL, va="bottom")
    ax.axhline(2.0, color=CHARCOAL, ls="--", lw=1)
    ax.text(0.62, 1.55, "2% target", fontsize=7.5, color=CHARCOAL)
    ax.set_ylabel("Flagged transactions (%)")
    ax.set_ylim(0, 8)
    _save(fig, "fig04_split_rates")


def fig_monthly_drift(scored: pd.DataFrame) -> None:
    monthly = scored.groupby("period").agg(rate=("is_anomaly", "mean"), n=("is_anomaly", "size"))
    monthly = monthly[monthly["n"] >= 20]
    labels = [p.strftime("%b\n%y") if p.month in (1,) else p.strftime("%b") for p in monthly.index]
    window_of = scored.groupby("period")["window"].agg(lambda s: s.mode().iloc[0])
    colors = [{"Train": NAVY, "Validation": FOREST, "Test": WINE}[window_of[p]] for p in monthly.index]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.bar(range(len(monthly)), monthly["rate"] * 100, color=colors, width=0.75)
    ax.axhline(2.0, color=CHARCOAL, ls="--", lw=1)
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Flagged transactions (%)")
    ax.legend(handles=[Patch(color=NAVY, label="Train"), Patch(color=FOREST, label="Validation"),
                       Patch(color=WINE, label="Test")], frameon=False, loc="upper left")
    _save(fig, "fig05_monthly_drift")


def fig_real_vs_synthetic(scored: pd.DataFrame) -> None:
    order = ["Train", "Validation", "Test"]
    g = scored.groupby(["window", "synthetic"])["is_anomaly"].mean().unstack() * 100
    g = g.loc[order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    b1 = ax.bar(x - 0.18, g[False], 0.36, color=NAVY, label="Real rows")
    b2 = ax.bar(x + 0.18, g[True], 0.36, color=OCHRE, label="Synthetic rows")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.25, f"{b.get_height():.1f}",
                    ha="center", fontsize=7.5)
    ax.axhline(2.0, color=CHARCOAL, ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Flagged transactions (%)")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, "fig06_real_vs_synthetic")


def fig_stability(stability: dict) -> None:
    jac = [p["value"] if isinstance(p, dict) else p for p in stability["jaccard"]["pairs"]]
    rho = [p["value"] if isinstance(p, dict) else p for p in stability["rank_correlation"]["pairs"]]
    s = np.array(stability["stability"]["stability_score"])
    runs = np.rint(s * 10).astype(int)
    groups = {
        "Never (0/10)": int((runs == 0).sum()),
        "1–4 runs": int(((runs >= 1) & (runs <= 4)).sum()),
        "5–7 runs": int(((runs >= 5) & (runs <= 7)).sum()),
        "8–9 runs": int(((runs >= 8) & (runs <= 9)).sum()),
        "All 10 runs": int((runs == 10).sum()),
    }
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"wspace": 0.55})
    bins = np.linspace(0.5, 1.0, 26)
    a1.hist(jac, bins=bins, color=WINE, label=f"Jaccard (flag set), mean {np.mean(jac):.3f}")
    a1.hist(rho, bins=bins, color=NAVY, label=f"Spearman ρ (ranking), mean {np.mean(rho):.3f}")
    a1.set_xlabel("Pairwise agreement between seeds")
    a1.set_ylabel("Seed pairs (of 45)")
    a1.legend(frameon=False, loc="upper left", fontsize=7)
    a1.set_ylim(0, 30)
    a1.set_title("(a) Ranking agrees more than the binary flag")

    labels = list(groups)[1:]
    vals = [groups[k] for k in labels]
    colors = ["#C4C4C4", "#8A8A8A", "#4A4A4A", "#111111"]
    bars = a2.barh(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        a2.text(v + 0.3, b.get_y() + b.get_height() / 2, str(v), va="center", fontsize=8)
    a2.set_xlabel("Test transactions (of the 52 ever flagged)")
    a2.set_title(f"(b) How often each flagged row is flagged\n({groups['Never (0/10)']} of 560 never flagged)")
    a2.grid(axis="y", visible=False)
    _save(fig, "fig07_stability")


def fig_feature_contrast(model_metrics: dict, top_n: int = 10) -> None:
    rows = model_metrics["metrics"]["test"]["feature_contrast"][:top_n][::-1]
    amount = lambda f: "Amount" in f or "Balance" in f
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    ax.barh([r["feature"] for r in rows], [r["std_gap"] for r in rows],
            color=[WINE if amount(r["feature"]) else SLATE for r in rows])
    ax.axvline(0, color=CHARCOAL, lw=0.8)
    ax.set_xlabel("Standardized gap (flagged − normal)")
    ax.legend(handles=[Patch(color=WINE, label="Amount / balance"), Patch(color=SLATE, label="Other")],
              frameon=False, loc="lower right")
    ax.grid(axis="y", visible=False)
    _save(fig, "fig08_feature_contrast")


def fig_shap(explainability: dict, top_n: int = 15) -> None:
    def category(f: str) -> tuple[str, str]:
        if f.startswith("USState"):
            return "Geographic", NAVY
        if f.startswith("CustomerOccupation"):
            return "Demographic", OCHRE
        if f in ("Month", "Is_Night", "Hour", "Is_Weekend"):
            return "Temporal", FOREST
        if "Amount" in f or "Balance" in f:
            return "Amount", WINE
        return "Account / channel / type", SLATE

    rows = explainability["feature_importance"][:top_n][::-1]
    cats = [category(r["feature"]) for r in rows]
    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    bars = ax.barh([r["feature"] for r in rows], [r["importance"] for r in rows], color=[c[1] for c in cats],
                   edgecolor="white", linewidth=0.3)
    for b, (k, _) in zip(bars, cats):
        if k == "Account / channel / type":
            b.set_hatch("////")
    ax.set_xlabel("Mean |SHAP value|")
    seen = dict(cats)
    ax.legend(handles=[Patch(facecolor=c, edgecolor="white", hatch="////" if k.startswith("Account") else None, label=k) for k, c in seen.items()], frameon=False,
              loc="upper center", bbox_to_anchor=(0.2, -0.16), ncol=3, fontsize=7)
    ax.grid(axis="y", visible=False)
    _save(fig, "fig09_shap")


def fig_temporal(anomaly_summary: dict) -> None:
    hours = anomaly_summary["by_hour"]
    days = anomaly_summary["by_day_of_week"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.4), gridspec_kw={"width_ratios": [2, 1]})
    a1.bar([h["bucket"] for h in hours], [h["anomaly_rate"] * 100 for h in hours],
           color=[WINE if h["bucket"] <= 4 else SLATE for h in hours])
    a1.set_xlabel("Hour of day")
    a1.set_ylabel("Flagged (%)")
    a1.set_title("(a) By hour (00–04 h highlighted)")
    a1.set_xticks(range(0, 24, 2))
    a2.bar([d["bucket"][:3] for d in days], [d["anomaly_rate"] * 100 for d in days],
           color=[WINE if d["bucket"] in ("Saturday", "Sunday") else SLATE for d in days])
    a2.set_title("(b) By day of week")
    a2.tick_params(axis="x", labelsize=6.5)
    for a in (a1, a2):
        a.axhline(2.0, color=CHARCOAL, ls="--", lw=0.9)
    _save(fig, "fig10_temporal")


def fig_state_map(state_metrics: list, geojson: dict) -> None:
    values = {s["state"]: s["anomaly_rate"] * 100 for s in state_metrics}
    vmax = max(values.values())
    norm = Normalize(0, vmax)
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    _draw_states(ax, geojson, lambda s: DARK_CMAP(norm(values[s])) if s in values else "#E6E6E6")
    sm = plt.cm.ScalarMappable(cmap=DARK_CMAP, norm=norm)
    fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, label="Flagged transactions (%)")
    ax.set_title("Anomaly rate by state (full dataset, all states ≥ 15 transactions)")
    _save(fig, "fig11_state_map")


def fig_state_support(state_metrics: list, scored: pd.DataFrame) -> None:
    from backend.src.modeling.config import CITY_TO_STATE
    synthetic_states = set(scored.loc[scored["synthetic"], "Location"].map(CITY_TO_STATE))
    df = pd.DataFrame(state_metrics)
    df["synthetic"] = df["state"].isin(synthetic_states)
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for syn, color, marker, label in [(False, NAVY, "o", "Real-data states"),
                                      (True, OCHRE, "^", "Synthetic-coverage states")]:
        d = df[df["synthetic"] == syn]
        ax.scatter(d["total_transactions"], d["anomaly_rate"] * 100, s=24, color=color, marker=marker,
                   label=label, edgecolor=CHARCOAL, linewidth=0.5)
    for name, offset in [("Connecticut", (5, -3)), ("Texas", (-22, 7)), ("California", (-48, -11))]:
        r = df[df["state"] == name].iloc[0]
        ax.annotate(name, (r["total_transactions"], r["anomaly_rate"] * 100),
                    xytext=offset, textcoords="offset points", fontsize=7, color=CHARCOAL)
    ax.set_xscale("log")
    ticks = [20, 50, 100, 200, 400]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.axhline(2.0, color=CHARCOAL, ls="--", lw=0.9)
    ax.set_xlabel("Transactions in state (log scale)")
    ax.set_ylabel("Flagged transactions (%)")
    ax.legend(frameon=False, loc="center right", fontsize=7)
    _save(fig, "fig12_state_support")


def fig_score_vs_stability(stability: dict) -> None:
    st = stability["stability"]
    score = np.array(st["mean_anomaly_score"])
    s = np.array(st["stability_score"])
    jitter = np.random.default_rng(0).uniform(-0.015, 0.015, len(s))
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.scatter(score, s + jitter, s=9, color=np.where(s == 1, WINE, np.where(s > 0, PLUM, OCHRE)),
               alpha=0.8, linewidth=0)
    ax.set_xlabel("Mean anomaly score over 10 seeds")
    ax.set_ylabel("Fraction of runs flagged")
    _save(fig, "figA1_score_vs_stability")


def main() -> None:
    model_metrics = _load("model_metrics.json")
    stability = _load("stability_analysis.json")
    explainability = _load("explainability.json")
    anomaly_summary = _load("anomaly_summary.json")
    state_metrics = _load("state_metrics.json")
    geojson = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))

    result, _, scored = _rerun()
    assert abs(result["threshold"] - model_metrics["threshold"]) < 1e-9, "re-run diverged from artifacts"

    fig_pipeline()
    fig_dataset_map(scored, geojson)
    fig_score_threshold(result)
    fig_split_rates(model_metrics, stability)
    fig_monthly_drift(scored)
    fig_real_vs_synthetic(scored)
    fig_stability(stability)
    fig_feature_contrast(model_metrics)
    fig_shap(explainability)
    fig_temporal(anomaly_summary)
    fig_state_map(state_metrics, geojson)
    fig_state_support(state_metrics, scored)
    fig_score_vs_stability(stability)


if __name__ == "__main__":
    main()

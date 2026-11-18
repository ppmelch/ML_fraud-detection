#!/usr/bin/env python3
"""
Compare the real-only model with the augmented model (real + synthetic rows).

The reported model is trained on ``bank_transactions_data_2_full.csv`` (2,512
real transactions + 781 synthetic rows added only so every US state has a
value on the map). This script re-runs the *unchanged* pipeline twice:

* **Real-only model (R)**: ``bank_transactions_data_2.csv`` only.
* **Augmented model (A)**: the working dataset, i.e. the model in the paper.

Nothing in the configuration changes between the two runs (same features
logic, Isolation Forest with 300 trees, seed 42, 2 % contamination, same
temporal windows). Only the rows differ. Because the real rows have the same
dates in both datasets, the train / validation / test windows start and end on
the same days, so both models can be scored on exactly the same real
transactions.

Outputs
-------
* ``results/real_vs_synthetic.json``  every number used in the paper (Section
  "Real-Only versus Augmented Model")
* ``paper/figures/fig13_real_vs_augmented.pdf/.png``
* ``paper/figures/fig14_state_support_comparison.pdf/.png``

Usage
-----
    python -m scripts.compare_real_vs_synthetic
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.src.data.data_loader import TransactionDataLoader  # noqa: E402
from backend.src.data.feature_engineering import FeatureEngineer  # noqa: E402
from backend.src.evaluation.explainability import AnomalyExplainer  # noqa: E402
from backend.src.modeling.anomaly_model import AnomalyModel  # noqa: E402
from backend.src.modeling.config import (  # noqa: E402
    CONTAMINATION,
    DATASET_PATH,
    MIN_TRANSACTIONS_PER_STATE,
    REAL_DATASET_PATH,
    STABILITY_SEEDS,
    TEST_DAYS,
    VALIDATION_DAYS,
)
from backend.src.modeling.contamination_threshold import ContaminationThreshold  # noqa: E402
from backend.src.modeling.model_evaluation import AnomalyEvaluation  # noqa: E402
from backend.src.pipeline import AnomalyPipeline  # noqa: E402
from paper.make_figures import (  # noqa: E402
    CHARCOAL,
    FOREST,
    NAVY,
    OCHRE,
    SLATE,
    WINE,
    _save,
)

RESULTS_DIR = ROOT / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fit(path: Path) -> tuple[AnomalyPipeline, dict, pd.DataFrame]:
    """Run the unchanged pipeline on ``path`` and return it with its raw CSV."""
    pipeline = AnomalyPipeline(
        model_name="isolation_forest", loader=TransactionDataLoader(path)
    )
    out = pipeline.run()
    return pipeline, out, pd.read_csv(path)


def _window_labels(ts: pd.Series) -> pd.Series:
    """Train / Validation / Test label from the day-level temporal boundaries."""
    last_day = ts.max().normalize()
    test_start = last_day - pd.Timedelta(days=TEST_DAYS - 1)
    validation_start = test_start - pd.Timedelta(days=VALIDATION_DAYS)
    return pd.Series(
        np.select(
            [ts < validation_start, ts < test_start],
            ["Train", "Validation"],
            default="Test",
        ),
        index=ts.index,
    )


def _score(pipeline: AnomalyPipeline, raw: pd.DataFrame) -> pd.DataFrame:
    """Score ``raw`` and attach window / period / synthetic columns."""
    scored = pipeline.score_frame(raw)
    ts = pd.to_datetime(scored["Transaction_Timestamp"])
    scored["window"] = _window_labels(ts)
    scored["period"] = ts.dt.to_period("M")
    scored["synthetic"] = scored["TransactionID"].str.startswith("TXS")
    return scored


def _rate(flags) -> float:
    flags = np.asarray(flags)
    return float(flags.mean()) if flags.size else float("nan")


def _window_table(scored: pd.DataFrame) -> dict:
    out = {}
    for w in ("Train", "Validation", "Test"):
        part = scored[scored["window"] == w]
        out[w.lower()] = {
            "n": int(len(part)),
            "flagged": int(part["is_anomaly"].sum()),
            "rate": _rate(part["is_anomaly"]),
            "score_mean": float(part["anomaly_score"].mean()),
        }
    return out


def _monthly(scored: pd.DataFrame) -> dict:
    g = scored.groupby("period").agg(
        n=("is_anomaly", "size"), flagged=("is_anomaly", "sum"), rate=("is_anomaly", "mean")
    )
    return {
        str(p): {"n": int(r.n), "flagged": int(r.flagged), "rate": float(r.rate)}
        for p, r in g.iterrows()
    }


def _stability(X_train: pd.DataFrame, X_test: pd.DataFrame, mask=None) -> dict:
    """
    Ten-seed refit, same procedure as ``StabilityAnalyzer``.

    ``mask`` restricts the evaluation to a subset of the test rows (used to
    score the augmented model on its real test rows only).
    """
    scores, flags = [], []
    for seed in STABILITY_SEEDS:
        model = AnomalyModel(
            "isolation_forest", contamination=CONTAMINATION, random_state=seed
        )
        model.fit(X_train)
        s = model.anomaly_score(X_test)
        thr = ContaminationThreshold(model.train_scores_, CONTAMINATION).resolve()
        scores.append(s)
        flags.append(s >= thr)
    S, F = np.array(scores), np.array(flags)
    if mask is not None:
        S, F = S[:, mask], F[:, mask]

    jac, rho = [], []
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            union = np.logical_or(F[i], F[j]).sum()
            jac.append(np.logical_and(F[i], F[j]).sum() / union if union else 1.0)
            rho.append(spearmanr(S[i], S[j]).correlation)

    times = F.sum(axis=0)
    return {
        "n_test": int(F.shape[1]),
        "jaccard_mean": float(np.mean(jac)),
        "spearman_mean": float(np.mean(rho)),
        "never_flagged": int((times == 0).sum()),
        "ever_flagged": int((times > 0).sum()),
        "borderline_1_to_4": int(((times >= 1) & (times <= 4)).sum()),
        "always_flagged": int((times == len(S)).sum()),
        "test_rate_min": float(F.mean(axis=1).min()),
        "test_rate_max": float(F.mean(axis=1).max()),
    }


def _behavior(scored: pd.DataFrame) -> dict:
    """Amount profile of flagged vs normal rows, plus night / weekend rates."""
    flagged = scored[scored["is_anomaly"] == 1]
    normal = scored[scored["is_anomaly"] == 0]
    night = scored["Hour"] < FeatureEngineer.NIGHT_HOUR_CUTOFF
    weekend = scored["Is_Weekend"] == 1
    total = scored["TransactionAmount"].sum()
    return {
        "n_flagged": int(len(flagged)),
        "flagged_rate": _rate(scored["is_anomaly"]),
        "amount_flagged_mean": float(flagged["TransactionAmount"].mean()),
        "amount_normal_mean": float(normal["TransactionAmount"].mean()),
        "flagged_share_of_money": float(flagged["TransactionAmount"].sum() / total),
        "n_night_rows": int(night.sum()),
        "n_weekend_rows": int(weekend.sum()),
        "night_rate": _rate(scored.loc[night, "is_anomaly"]) if night.any() else None,
        "not_night_rate": _rate(scored.loc[~night, "is_anomaly"]),
        "weekend_rate": _rate(scored.loc[weekend, "is_anomaly"]) if weekend.any() else None,
        "weekday_rate": _rate(scored.loc[~weekend, "is_anomaly"]),
        "weekend_share_of_flagged": float(weekend[scored["is_anomaly"] == 1].mean()),
    }


def _calendar_profile(raw_full: pd.DataFrame, full_a: pd.DataFrame) -> dict:
    """Where do night / weekend transactions come from? (real vs synthetic)"""
    ts = pd.to_datetime(raw_full["TransactionDate"])
    syn = raw_full["TransactionID"].str.startswith("TXS")
    night = ts.dt.hour < FeatureEngineer.NIGHT_HOUR_CUTOFF
    weekend = ts.dt.dayofweek >= 5
    flagged = full_a["is_anomaly"].to_numpy() == 1
    f_syn = full_a["synthetic"].to_numpy()
    real_ts = ts[~syn]
    out = {
        "real_hour_min": int(real_ts.dt.hour.min()),
        "real_hour_max": int(real_ts.dt.hour.max()),
        "real_weekdays_present": sorted(int(x) for x in real_ts.dt.dayofweek.unique()),
        "real_night_rows": int((night & ~syn).sum()),
        "real_weekend_rows": int((weekend & ~syn).sum()),
        "synthetic_night_rows": int((night & syn).sum()),
        "synthetic_weekend_rows": int((weekend & syn).sum()),
        "synthetic_rows": int(syn.sum()),
        "flagged_total": int(flagged.sum()),
        "flagged_synthetic": int((flagged & f_syn).sum()),
        "flagged_night": int((flagged & night.to_numpy()).sum()),
        "flagged_weekend": int((flagged & weekend.to_numpy()).sum()),
        "flagged_night_or_weekend": int((flagged & (night | weekend).to_numpy()).sum()),
        "flagged_night_or_weekend_synthetic": int(
            (flagged & (night | weekend).to_numpy() & f_syn).sum()
        ),
    }
    return out


def _amounts_test(data_test: pd.DataFrame, flags, mask=None) -> dict:
    """Mean amount of flagged vs normal rows in the test window (paper's view)."""
    d = data_test.reset_index(drop=True)
    f = np.asarray(flags).astype(bool)
    if mask is not None:
        d, f = d[mask], f[mask]
    amt = d["TransactionAmount"].to_numpy(dtype=float)
    return {
        "n_test": int(len(d)),
        "n_flagged": int(f.sum()),
        "flagged_mean": float(amt[f].mean()),
        "normal_mean": float(amt[~f].mean()),
    }


def _contrast(X: pd.DataFrame, flags, top: int = 10) -> list[dict]:
    rows = AnomalyEvaluation().feature_contrast(X.reset_index(drop=True), flags)
    return [{"feature": r["feature"], "std_gap": r["std_gap"]} for r in rows[:top]]


def _category(feature: str) -> str:
    if feature.startswith("USState"):
        return "Geographic"
    if feature.startswith("CustomerOccupation"):
        return "Demographic"
    if feature in ("Month", "Is_Night", "Hour", "Is_Weekend", "DayOfWeek"):
        return "Temporal"
    if "Amount" in feature or "Balance" in feature:
        return "Amount"
    return "Account / channel / type"


def _shap(model, X: pd.DataFrame) -> dict:
    """Mean |SHAP| per feature, its top-10 and its share by feature type."""
    payload = AnomalyExplainer(model).explain(X, top_n=500)
    rows = payload["feature_importance"]
    total = sum(r["importance"] for r in rows)
    share: dict[str, float] = {}
    for r in rows:
        c = _category(r["feature"])
        share[c] = share.get(c, 0.0) + r["importance"] / total
    ranking = [r["feature"] for r in rows]
    return {
        "method": payload["method"],
        "n_features": len(rows),
        "month_rank": ranking.index("Month") + 1 if "Month" in ranking else None,
        "top10": [
            {"feature": r["feature"], "importance": r["importance"], "type": _category(r["feature"])}
            for r in rows[:10]
        ],
        "share_by_type": share,
    }


def _states(scored: pd.DataFrame) -> dict:
    """Per-state flagged rate vs number of transactions (support)."""
    g = scored.groupby("USState").agg(
        n=("is_anomaly", "size"),
        flagged=("is_anomaly", "sum"),
        synthetic_only=("synthetic", "all"),
    )
    g["rate"] = g["flagged"] / g["n"]
    ok = g[g["n"] >= MIN_TRANSACTIONS_PER_STATE]
    rho, p = spearmanr(ok["n"], ok["rate"])
    small, large = ok[ok["n"] <= 40], ok[ok["n"] >= 100]
    top = ok.sort_values("rate", ascending=False).head(12)
    return {
        "n_states": int(len(g)),
        "n_states_analyzed": int(len(ok)),
        "spearman_rate_vs_n": float(rho),
        "p_value": float(p),
        "small_states_n": int(len(small)),
        "small_states_rate": float(small["flagged"].sum() / small["n"].sum()) if len(small) else None,
        "large_states_n": int(len(large)),
        "large_states_rate": float(large["flagged"].sum() / large["n"].sum()) if len(large) else None,
        "top12": [
            {
                "state": s,
                "n": int(r.n),
                "flagged": int(r.flagged),
                "rate": float(r.rate),
                "synthetic_only": bool(r.synthetic_only),
            }
            for s, r in top.iterrows()
        ],
        "top12_synthetic_only": int(top["synthetic_only"].sum()),
        "table": g.reset_index().to_dict("records"),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_comparison(win_r: dict, win_a: dict, mon_r: dict, mon_a: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.6, 4.6))

    order = ["train", "validation", "test"]
    x = np.arange(3)
    r = [win_r[k]["rate"] * 100 for k in order]
    a = [win_a[k]["rate"] * 100 for k in order]
    b1 = ax1.bar(x - 0.18, r, 0.36, color=NAVY, label="Real-only model")
    b2 = ax1.bar(x + 0.18, a, 0.36, color=OCHRE, label="Augmented model")
    for bars in (b1, b2):
        for b in bars:
            ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.12,
                     f"{b.get_height():.1f}", ha="center", fontsize=7.5)
    ax1.axhline(2.0, color=CHARCOAL, ls="--", lw=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Train", "Validation", "Test"])
    ax1.set_ylabel("Flagged real transactions (%)")
    ax1.set_title("(a) By period", fontsize=8.5)
    ax1.set_ylim(0, 6.4)
    ax1.legend(frameon=False, loc="upper left", fontsize=7)

    months = [m for m in mon_r if mon_r[m]["n"] >= 20]
    xs = np.arange(len(months))
    ax2.plot(xs, [mon_r[m]["rate"] * 100 for m in months], color=NAVY, marker="o", ms=3, lw=1.2,
             label="Real-only model")
    ax2.plot(xs, [mon_a[m]["rate"] * 100 for m in months], color=SLATE, marker="s", ms=3, lw=1.2,
             ls="-.", label="Augmented model")
    ax2.axhline(2.0, color=CHARCOAL, ls="--", lw=1)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([pd.Period(m).strftime("%b") for m in months], fontsize=7)
    ax2.set_ylabel("Flagged real transactions (%)")
    ax2.set_title("(b) By month", fontsize=8.5)
    ax2.legend(frameon=False, loc="upper left", fontsize=7)
    fig.tight_layout()
    _save(fig, "fig13_real_vs_augmented")


def fig_state_support(table_r: list, table_a: list) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ra = pd.DataFrame(table_r)
    aa = pd.DataFrame(table_a)
    ra, aa = ra[ra["n"] >= MIN_TRANSACTIONS_PER_STATE], aa[aa["n"] >= MIN_TRANSACTIONS_PER_STATE]
    aa_real = aa[~aa["synthetic_only"]]
    aa_syn = aa[aa["synthetic_only"]]
    ax.scatter(aa_syn["n"], aa_syn["rate"] * 100, s=22, facecolors="white", edgecolors=SLATE,
               linewidths=0.9, label="Augmented: synthetic-only states")
    ax.scatter(aa_real["n"], aa_real["rate"] * 100, s=20, color=SLATE, marker="s",
               label="Augmented: states with real rows")
    ax.scatter(ra["n"], ra["rate"] * 100, s=22, color=WINE, marker="o", label="Real-only model")
    ax.axhline(2.0, color=CHARCOAL, ls="--", lw=1)
    ax.set_xscale("log")
    ticks = [20, 40, 60, 100, 200, 400]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Transactions per state (log scale)")
    ax.set_ylabel("Flagged transactions (%)")
    ax.set_ylim(-0.8, 21)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    _save(fig, "fig14_state_support_comparison")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("[1/6] Fitting real-only model (R) and augmented model (A) ...")
    pipe_r, out_r, raw_real = _fit(REAL_DATASET_PATH)
    pipe_a, out_a, raw_full = _fit(DATASET_PATH)

    print("[2/6] Scoring the real rows with both models ...")
    real_r = _score(pipe_r, raw_real)          # R scored on its own (real-only) frame
    full_a = _score(pipe_a, raw_full)          # A scored on its own frame (as in the paper)
    real_a = full_a[~full_a["synthetic"]].copy()  # ... restricted to the real rows
    assert not real_r["synthetic"].any() and len(real_a) == len(real_r)

    # Sanity: the augmented run must reproduce the paper's reported numbers.
    assert abs(out_a["threshold"] - 0.7104945530387144) < 1e-6, "A no longer matches the paper"
    assert out_a["metrics"]["test"]["n_flagged"] == 38

    result: dict = {
        "setup": {
            "contamination": CONTAMINATION,
            "seed": 42,
            "n_real_rows": int(len(raw_real)),
            "n_full_rows": int(len(raw_full)),
            "n_synthetic_rows": int(len(raw_full) - len(raw_real)),
        },
        "real_only": {
            "threshold": out_r["threshold"],
            "n_features": len(out_r["feature_columns"]),
            "split": {k: out_r["split"][k]["rows"] for k in ("train", "validation", "test")},
        },
        "augmented": {
            "threshold": out_a["threshold"],
            "n_features": len(out_a["feature_columns"]),
            "split": {k: out_a["split"][k]["rows"] for k in ("train", "validation", "test")},
        },
    }

    print("[3/6] Flag rates by period and month ...")
    for key, out in (("real_only", out_r), ("augmented", out_a)):
        result[key]["own_windows"] = {
            w: {
                "n": out["metrics"][w]["n_samples"],
                "flagged": out["metrics"][w]["n_flagged"],
                "rate": out["metrics"][w]["flagged_rate"],
                "score_mean": out["metrics"][w]["score_mean"],
            }
            for w in ("train", "validation", "test")
        }
    result["real_only"]["windows"] = _window_table(real_r)
    result["augmented"]["windows_on_real_rows"] = _window_table(real_a)
    result["augmented"]["windows_all_rows"] = _window_table(full_a)
    result["real_only"]["monthly"] = _monthly(real_r)
    result["augmented"]["monthly_on_real_rows"] = _monthly(real_a)

    print("[4/6] Agreement between the two models on the same real test rows ...")
    ids_r = out_r["data_test"]["TransactionID"].to_numpy()
    syn_a = out_a["data_test"]["TransactionID"].str.startswith("TXS").to_numpy()
    sc_r = pd.Series(out_r["test_scores"], index=ids_r)
    sc_a = pd.Series(out_a["test_scores"][~syn_a], index=out_a["data_test"]["TransactionID"].to_numpy()[~syn_a])
    assert set(sc_r.index) == set(sc_a.index)
    sc_a = sc_a.loc[sc_r.index]
    fl_r = (sc_r >= out_r["threshold"]).to_numpy()
    fl_a = (sc_a >= out_a["threshold"]).to_numpy()
    inter, union = int((fl_r & fl_a).sum()), int((fl_r | fl_a).sum())
    top_k = 20
    result["agreement_test_real_rows"] = {
        "n": int(len(sc_r)),
        "spearman_scores": float(spearmanr(sc_r, sc_a).correlation),
        "flagged_real_only": int(fl_r.sum()),
        "flagged_augmented": int(fl_a.sum()),
        "flagged_both": inter,
        "jaccard_flagged": inter / union if union else 1.0,
        "top20_overlap": len(set(sc_r.nlargest(top_k).index) & set(sc_a.nlargest(top_k).index)),
    }

    print("[5/6] Stability over ten seeds and SHAP ...")
    mask_real_test = ~out_a["data_test"]["TransactionID"].str.startswith("TXS").to_numpy()
    result["real_only"]["stability"] = _stability(out_r["X_train"], out_r["X_test"])
    result["augmented"]["stability_all_test"] = _stability(out_a["X_train"], out_a["X_test"])
    result["augmented"]["stability_real_test"] = _stability(
        out_a["X_train"], out_a["X_test"], mask=mask_real_test
    )
    result["real_only"]["shap"] = _shap(pipe_r.model, out_r["X_test"])
    result["augmented"]["shap_all_test"] = _shap(pipe_a.model, out_a["X_test"])
    result["augmented"]["shap_real_test"] = _shap(pipe_a.model, out_a["X_test"][mask_real_test])

    print("[6/6] Behavior and geography ...")
    result["real_only"]["behavior"] = _behavior(real_r)
    result["augmented"]["behavior_on_real_rows"] = _behavior(real_a)
    result["augmented"]["behavior_all_rows"] = _behavior(full_a)

    result["real_only"]["feature_contrast_top10"] = _contrast(
        out_r["X_test"], (out_r["test_scores"] >= out_r["threshold"]).astype(int)
    )
    a_flags = (out_a["test_scores"] >= out_a["threshold"]).astype(int)
    result["augmented"]["feature_contrast_top10_all_test"] = _contrast(out_a["X_test"], a_flags)
    result["augmented"]["feature_contrast_top10_real_test"] = _contrast(
        out_a["X_test"][mask_real_test], a_flags[mask_real_test]
    )

    result["real_only"]["amounts_test"] = _amounts_test(
        out_r["data_test"], (out_r["test_scores"] >= out_r["threshold"]).astype(int)
    )
    result["augmented"]["amounts_test_all"] = _amounts_test(out_a["data_test"], a_flags)
    result["augmented"]["amounts_test_real"] = _amounts_test(
        out_a["data_test"], a_flags, mask=mask_real_test
    )
    result["calendar_profile"] = _calendar_profile(raw_full, full_a)
    result["real_only"]["states"] = _states(real_r)
    result["augmented"]["states_all_rows"] = _states(full_a)
    result["augmented"]["states_real_rows"] = _states(real_a)

    RESULTS_DIR.mkdir(exist_ok=True)
    slim = json.loads(json.dumps(result, default=float))
    for key in ("states", "states_all_rows", "states_real_rows"):
        for side in ("real_only", "augmented"):
            if key in slim[side]:
                slim[side][key].pop("table", None)
    (RESULTS_DIR / "real_vs_synthetic.json").write_text(
        json.dumps(slim, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig_comparison(
        result["real_only"]["windows"],
        result["augmented"]["windows_on_real_rows"],
        result["real_only"]["monthly"],
        result["augmented"]["monthly_on_real_rows"],
    )
    fig_state_support(
        result["real_only"]["states"]["table"], result["augmented"]["states_all_rows"]["table"]
    )
    print("done -> results/real_vs_synthetic.json")


if __name__ == "__main__":
    main()

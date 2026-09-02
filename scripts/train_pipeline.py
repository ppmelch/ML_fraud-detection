#!/usr/bin/env python3
"""
Run the unsupervised anomaly-detection pipeline end to end and write every
artifact the API serves.

This is the one place that turns the raw CSV into everything downstream::

    dataset -> AnomalyPipeline (split, features, fit, score, threshold, evaluate)
            -> AnomalyExplainer (SHAP / permutation feature importance)
            -> TransactionAnalytics (portfolio anomaly analytics, full dataset)
            -> StateAnalytics + StateGeoMapper (per-state metrics, GeoJSON join)
            -> backend/artifacts/*.json  +  backend/src/models/anomaly_model.pkl

The API layer only reads these files; it never recomputes them. Re-run this
script whenever the dataset, feature set or model configuration changes.

Usage
-----
    python -m scripts.train_pipeline [--model {isolation_forest,lof,autoencoder}]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.src.data.data_loader import TransactionDataLoader
from backend.src.evaluation.explainability import AnomalyExplainer
from backend.src.metrics.geo_mapping import StateGeoMapper
from backend.src.metrics.state_analytics import StateAnalytics
from backend.src.metrics.transaction_analytics import TransactionAnalytics
from backend.src.modeling.config import ARTIFACTS_DIR, DEFAULT_MODEL
from backend.src.modeling.model_loader import ModelLoader
from backend.src.pipeline import AnomalyPipeline

MODEL_CHOICES = ["isolation_forest", "lof", "autoencoder"]


def _json_safe(value):
    """
    Recursively coerce numpy scalars / arrays and NaN to JSON-safe Python.

    Parameters
    ----------
    value : Any
        Arbitrary nested structure.

    Returns
    -------
    Any
        The same structure with numpy types cast, NaN/inf replaced by
        ``None``, and dict keys stringified.
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if (number != number or number in (float("inf"), float("-inf"))) else number

    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    return value


def _write_json(path: Path, payload) -> None:
    """
    Serialize ``payload`` to ``path`` atomically after validating it.

    Writes to a temporary file in the same directory and renames it into
    place, so a crash mid-write never leaves a half-written JSON file.

    Parameters
    ----------
    path : Path
        Destination file.
    payload : dict or list
        JSON-serialisable content (passed through :func:`_json_safe` first).

    Raises
    ------
    ValueError
        If the payload still contains a non-serialisable value or NaN.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = _json_safe(payload)

    text = json.dumps(safe, indent=2, ensure_ascii=False, allow_nan=False)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def main() -> None:
    """Run the pipeline and write all artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=MODEL_CHOICES)
    args = parser.parse_args()

    started = time.time()
    print(f"[1/6] Loading dataset, temporal split, fitting ({args.model}) ...")

    pipeline = AnomalyPipeline(model_name=args.model)
    run_output = pipeline.run()

    print(f"      done in {time.time() - started:.1f}s")
    print(
        f"      threshold = {run_output['threshold']:.6f} "
        f"contamination = {run_output['contamination']}"
    )
    for window in ("train", "validation", "test"):
        m = run_output["metrics"][window]
        print(
            f"      {window:10s} n={m['n_samples']:5d} flagged={m['n_flagged']:4d} "
            f"rate={m['flagged_rate']:.4f} score_mean={m['score_mean']:.4f} "
            f"p99={m['score_percentiles']['p99']:.4f}"
        )

    split = run_output["split"]
    model_metrics_payload = {
        "model_name": args.model,
        "threshold": run_output["threshold"],
        "contamination": run_output["contamination"],
        "split": {
            "train": split["train"],
            "validation": split["validation"],
            "test": split["test"],
        },
        "metrics": run_output["metrics"],
        "feature_columns": run_output["feature_columns"],
        "n_features": len(run_output["feature_columns"]),
    }
    _write_json(ARTIFACTS_DIR / "model_metrics.json", model_metrics_payload)
    print("[2/6] Wrote model_metrics.json")

    print("[3/6] Computing feature importance / impact on the test window ...")
    explainer = AnomalyExplainer(pipeline.model)
    explainability_payload = explainer.explain(run_output["X_test"], top_n=25)
    _write_json(ARTIFACTS_DIR / "explainability.json", explainability_payload)
    print(
        f"      method={explainability_payload['method']} "
        f"flagged_rate={explainability_payload['score_summary']['flagged_rate']:.4f}"
    )

    print("[4/6] Scoring the full dataset and computing anomaly analytics ...")
    full_raw = TransactionDataLoader().load()
    scored = pipeline.score_frame(full_raw)
    anomaly_summary = TransactionAnalytics(scored).summary()
    _write_json(ARTIFACTS_DIR / "anomaly_summary.json", anomaly_summary)
    ov = anomaly_summary["overview"]
    print(
        f"      total={ov['total_transactions']} flagged={ov['flagged_transactions']} "
        f"rate={ov['flagged_rate']:.4f} mean_score={ov['mean_anomaly_score']:.4f}"
    )

    print("[5/6] Computing per-state analytics and reconciling with GeoJSON ...")
    state_records = StateAnalytics(scored).compute()
    mapper = StateGeoMapper()
    state_records = mapper.attach(state_records)
    geo_validation = mapper.validate(state_records)

    _write_json(ARTIFACTS_DIR / "state_metrics.json", state_records)
    _write_json(ARTIFACTS_DIR / "geo_validation.json", geo_validation)
    print(
        f"      dataset_states={geo_validation['dataset_states']} "
        f"matched={geo_validation['matched_states']} "
        f"unmatched={geo_validation['unmatched_dataset_states']} "
        f"ok={geo_validation['ok']}"
    )

    if not geo_validation["ok"]:
        print(
            "      WARNING: state/GeoJSON join not fully reconciled; unmatched "
            "states render with metrics but no polygon."
        )

    print("[6/6] Saving the trained model bundle ...")
    saved_path = ModelLoader().save(
        model=pipeline.model,
        feature_engineer=pipeline.feature_engineer,
        data_preparation=pipeline.data_preparation,
        threshold=pipeline.threshold,
        train_scores=pipeline.model.train_scores_,
        metadata={
            "model_name": args.model,
            "contamination": run_output["contamination"],
            "split": split,
            "test_flagged_rate": run_output["metrics"]["test"]["flagged_rate"],
            "n_features": len(run_output["feature_columns"]),
        },
    )
    print(f"      saved to {saved_path}")
    print(
        f"\nAll artifacts written to {ARTIFACTS_DIR} in "
        f"{time.time() - started:.1f}s total"
    )


if __name__ == "__main__":
    main()

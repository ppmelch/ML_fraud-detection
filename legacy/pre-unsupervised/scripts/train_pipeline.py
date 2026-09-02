#!/usr/bin/env python3
"""
Run the fraud detection pipeline end to end and write every artifact the API
serves.

This is the one place that turns the raw CSV into everything downstream:

    dataset -> FraudPipeline (split, features, train, threshold, evaluate)
            -> ModelExplainer (SHAP feature importance / impact)
            -> FraudAnalytics (observed fraud, from Is_Fraud, not predictions)
            -> StateAnalytics + StateGeoMapper (state metrics, GeoJSON join)
            -> backend/artifacts/*.json  +  backend/models/fraud_model.pkl

The API layer only reads these files; it never recomputes them. Re-run this
script whenever the dataset, feature set or model configuration changes.

Usage
-----
    python -m scripts.train_pipeline [--model {logistic,random_forest,xgboost,lightgbm}]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.src.data.data_loader import TransactionDataLoader
from backend.src.evaluation.explainability import ModelExplainer
from backend.src.metrics.fraud_analytics import FraudAnalytics
from backend.src.metrics.geo_mapping import StateGeoMapper
from backend.src.metrics.state_analytics import StateAnalytics
from backend.src.modeling.config import ARTIFACTS_DIR, DEFAULT_MODEL
from backend.src.modeling.model_loader import ModelLoader
from backend.src.pipeline import FraudPipeline


def _write_json(path: Path, payload: dict | list) -> None:
    """
    Serialize ``payload`` to ``path`` atomically.

    Writes to a temporary file in the same directory and renames it into
    place, so a crash mid-write never leaves a half-written JSON file for the
    API to fail on with no line number.

    Parameters
    ----------
    path : Path
        Destination file.
    payload : dict or list
        JSON-serialisable content.

    Raises
    ------
    ValueError
        If ``payload`` contains a value ``json.dump`` cannot serialize
        (surfaced before the temp file is renamed into place).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)

    tmp_path.replace(path)


def main() -> None:
    """Run the pipeline and write all artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=["logistic", "random_forest", "xgboost", "lightgbm"],
    )
    args = parser.parse_args()

    started = time.time()
    print(f"[1/6] Loading dataset and running temporal split + training ({args.model}) ...")

    pipeline = FraudPipeline(model_name=args.model)
    run_output = pipeline.run()

    print(f"      done in {time.time() - started:.1f}s")
    print(f"      threshold = {run_output['threshold']:.4f}")

    for window in ("train", "validation", "test"):
        m = run_output["metrics"][window]
        print(
            f"      {window:10s} roc_auc={m['roc_auc']} pr_auc={m['pr_auc']} "
            f"(baseline={m['pr_auc_baseline']:.4f}) f1={m['f1_score']:.4f} "
            f"precision={m['precision']:.4f} recall={m['recall']:.4f}"
        )

    model_metrics_payload = {
        "model_name": args.model,
        "threshold": run_output["threshold"],
        "split": run_output["split"],
        "metrics": run_output["metrics"],
        "feature_columns": run_output["feature_columns"],
        "n_features": len(run_output["feature_columns"]),
    }
    _write_json(ARTIFACTS_DIR / "model_metrics.json", model_metrics_payload)
    print(f"[2/6] Wrote model_metrics.json")

    print("[3/6] Computing SHAP feature importance / impact on the test window ...")
    explainer = ModelExplainer(pipeline.model)
    explainability_payload = explainer.explain(
        run_output["X_test"], top_n=25, test_metrics=run_output["metrics"]["test"]
    )
    _write_json(ARTIFACTS_DIR / "explainability.json", explainability_payload)
    print(
        f"      method={explainability_payload['method']} "
        f"generalizes={explainability_payload['model_generalization']['generalizes']}"
    )

    print("[4/6] Computing observed fraud analytics (from Is_Fraud, full dataset) ...")
    full_data = TransactionDataLoader().load()
    fraud_summary = FraudAnalytics(full_data).summary()
    _write_json(ARTIFACTS_DIR / "fraud_summary.json", fraud_summary)
    print(
        f"      total_transactions={fraud_summary['overview']['total_transactions']} "
        f"fraud_cases={fraud_summary['overview']['fraud_cases']} "
        f"fraud_rate={fraud_summary['overview']['fraud_rate']:.5f}"
    )

    print("[5/6] Computing state-level analytics and reconciling with GeoJSON ...")
    state_records = StateAnalytics(full_data).compute()
    mapper = StateGeoMapper()
    state_records = mapper.attach(state_records)
    geo_validation = mapper.validate(state_records)

    _write_json(ARTIFACTS_DIR / "state_metrics.json", state_records)
    _write_json(ARTIFACTS_DIR / "geo_validation.json", geo_validation)

    print(
        f"      states={geo_validation['dataset_states']} "
        f"matched={geo_validation['matched_states']} "
        f"unmatched={geo_validation['unmatched_dataset_states']} "
        f"ok={geo_validation['ok']}"
    )

    if not geo_validation["ok"]:
        print(
            "      WARNING: state/GeoJSON join is not fully reconciled; "
            "unmatched states will render with metrics but no polygon match."
        )

    print("[6/6] Saving the trained model bundle ...")
    saved_path = ModelLoader().save(
        model=pipeline.model,
        feature_engineer=pipeline.feature_engineer,
        data_preparation=pipeline.data_preparation,
        threshold=pipeline.threshold,
        metadata={
            "model_name": args.model,
            "trained_at": run_output["split"],
            "test_roc_auc": run_output["metrics"]["test"]["roc_auc"],
            "test_pr_auc": run_output["metrics"]["test"]["pr_auc"],
        },
    )
    print(f"      saved to {saved_path}")

    print(f"\nAll artifacts written to {ARTIFACTS_DIR} in {time.time() - started:.1f}s total")


if __name__ == "__main__":
    main()

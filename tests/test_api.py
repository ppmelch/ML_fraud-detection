"""
Tests for the FastAPI layer.

Uses the real artifacts produced by ``scripts/train_pipeline.py`` when they
are present, and skips the artifact-dependent tests otherwise rather than
faking data the API is meant to serve honestly.
"""

import pytest
from fastapi.testclient import TestClient

from backend.src.api import artifacts
from backend.src.api.main import app
from backend.src.modeling.config import ARTIFACTS_DIR
from backend.src.modeling.model_loader import ModelLoader

artifacts.clear_cache()
client = TestClient(app)

ARTIFACTS_READY = all(
    (ARTIFACTS_DIR / name).exists()
    for name in (
        "model_metrics.json",
        "anomaly_summary.json",
        "state_metrics.json",
        "geo_validation.json",
        "explainability.json",
    )
)

requires_artifacts = pytest.mark.skipif(
    not ARTIFACTS_READY, reason="Run 'python -m scripts.train_pipeline' first"
)
requires_model = pytest.mark.skipif(
    not ModelLoader().exists(), reason="Train a model first"
)


class TestHealth:
    def test_health_ok(self):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


@requires_artifacts
class TestAnomalySummary:
    def test_overview_keys(self):
        body = client.get("/api/anomaly/summary").json()
        overview = body["overview"]

        for key in (
            "total_transactions",
            "total_accounts",
            "flagged_transactions",
            "flagged_rate",
            "mean_anomaly_score",
            "flagged_amount",
        ):
            assert key in overview

    def test_breakdowns_and_distributions(self):
        body = client.get("/api/anomaly/summary").json()

        for key in (
            "by_hour",
            "by_day_of_week",
            "by_channel",
            "by_transaction_type",
            "by_occupation",
        ):
            assert isinstance(body[key], list)

        assert set(body["amount_distribution"]) == {
            "bin_centers",
            "normal_counts",
            "flagged_counts",
        }
        assert set(body["score_distribution"]) == {"bin_centers", "counts"}


@requires_artifacts
class TestModelMetrics:
    def test_windows_and_no_label_metrics(self):
        body = client.get("/api/model/metrics").json()

        assert set(body["metrics"]) == {"train", "validation", "test"}
        assert "threshold" in body
        assert "contamination" in body

        test_metrics = body["metrics"]["test"]
        for key in (
            "n_samples",
            "n_flagged",
            "flagged_rate",
            "score_percentiles",
            "score_histogram",
            "score_rank_curve",
            "top_anomalies",
            "feature_contrast",
            "heuristic_alignment",
        ):
            assert key in test_metrics

        # no supervised label metrics leaked in
        assert "roc_auc" not in test_metrics
        assert "confusion_matrix" not in test_metrics


@requires_artifacts
class TestStates:
    def test_state_records(self):
        body = client.get("/api/states").json()

        assert isinstance(body, list) and body
        for key in (
            "state",
            "anomaly_rate",
            "mean_anomaly_score",
            "flagged",
            "peak_anomaly_hour",
            "peak_anomaly_day",
            "most_common_flagged_channel",
            "top_flagged_occupation",
            "geojson_name",
            "geojson_id",
            "matched",
        ):
            assert key in body[0]

    def test_validation_ok(self):
        body = client.get("/api/states/validation").json()
        assert body["ok"] is True


@requires_artifacts
class TestGeoJSONAndExplainability:
    def test_geojson_feature_collection(self):
        assert client.get("/api/map/geojson").json()["type"] == "FeatureCollection"

    def test_explainability_payload(self):
        body = client.get("/api/model/explainability").json()

        for key in (
            "method",
            "feature_importance",
            "feature_impact",
            "score_summary",
            "caveat",
        ):
            assert key in body
        assert set(body["score_summary"]) == {"mean", "p95", "p99", "flagged_rate"}


class TestPredictValidation:
    def test_missing_fields_returns_422(self):
        assert client.post("/api/predict", json={"Location": "Miami"}).status_code == 422

    def test_negative_amount_rejected(self):
        payload = {
            "TransactionAmount": -5.0,
            "TransactionType": "Debit",
            "Location": "Miami",
            "Channel": "ATM",
            "CustomerAge": 30,
            "CustomerOccupation": "Doctor",
            "TransactionDuration": 50.0,
            "LoginAttempts": 1,
            "AccountBalance": 1000.0,
            "TransactionDate": "2023-05-01 10:00:00",
        }
        assert client.post("/api/predict", json=payload).status_code == 422


@requires_model
class TestPredictScoring:
    PAYLOAD = {
        "TransactionAmount": 120.0,
        "TransactionType": "Debit",
        "Location": "San Diego",
        "Channel": "ATM",
        "CustomerAge": 70,
        "CustomerOccupation": "Doctor",
        "TransactionDuration": 80.0,
        "LoginAttempts": 1,
        "AccountBalance": 5000.0,
        "TransactionDate": "2023-08-15 14:20:03",
        "AccountID": "AC00128",
    }

    def test_returns_anomaly_score_in_unit_interval(self):
        response = client.post("/api/predict", json=self.PAYLOAD)
        assert response.status_code == 200

        body = response.json()
        assert 0.0 <= body["anomaly_score"] <= 1.0
        assert 0.0 <= body["percentile"] <= 100.0
        assert isinstance(body["is_anomaly"], bool)
        assert body["threshold_used"] > 0
        assert "caveat" in body

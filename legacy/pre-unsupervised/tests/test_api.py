"""
Tests for the FastAPI layer.

Uses the real artifacts produced by ``scripts/train_pipeline.py`` when they
are present (the common case in this repo, run manually), and skips the
artifact-dependent tests otherwise rather than faking data the API is meant
to serve honestly.
"""

import pytest
from fastapi.testclient import TestClient

from backend.src.api.main import app
from backend.src.modeling.config import ARTIFACTS_DIR
from backend.src.modeling.model_loader import ModelLoader

client = TestClient(app)

ARTIFACTS_READY = all(
    (ARTIFACTS_DIR / name).exists()
    for name in (
        "model_metrics.json",
        "fraud_summary.json",
        "state_metrics.json",
        "geo_validation.json",
        "explainability.json",
    )
)

requires_artifacts = pytest.mark.skipif(
    not ARTIFACTS_READY,
    reason="Run 'python -m scripts.train_pipeline' to generate artifacts first",
)

requires_model = pytest.mark.skipif(
    not ModelLoader().exists(),
    reason="Run 'python -m scripts.train_pipeline' to train a model first",
)


class TestHealth:
    def test_health_ok(self):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


@requires_artifacts
class TestFraudSummary:
    def test_returns_overview_with_expected_keys(self):
        response = client.get("/api/fraud/summary")
        assert response.status_code == 200

        body = response.json()
        assert "overview" in body
        assert "total_transactions" in body["overview"]
        assert "fraud_rate" in body["overview"]

    def test_fraud_rate_is_from_is_fraud_not_model(self):
        response = client.get("/api/fraud/summary")
        overview = response.json()["overview"]

        assert overview["fraud_cases"] == pytest.approx(
            overview["total_transactions"] * overview["fraud_rate"], abs=1
        )


@requires_artifacts
class TestModelMetrics:
    def test_returns_train_validation_test(self):
        response = client.get("/api/model/metrics")
        assert response.status_code == 200

        body = response.json()
        assert set(body["metrics"]) == {"train", "validation", "test"}
        assert "threshold" in body

    def test_no_accuracy_only_reported_as_headline(self):
        # accuracy is present, but so are the imbalance-aware metrics.
        body = client.get("/api/model/metrics").json()
        test_metrics = body["metrics"]["test"]

        for key in ("roc_auc", "pr_auc", "precision", "recall", "f1_score", "confusion_matrix"):
            assert key in test_metrics


@requires_artifacts
class TestStates:
    def test_returns_list_of_states(self):
        response = client.get("/api/states")
        assert response.status_code == 200

        body = response.json()
        assert isinstance(body, list)
        assert len(body) > 0
        assert "state" in body[0]
        assert "fraud_rate" in body[0]
        assert "geojson_name" in body[0]

    def test_validation_endpoint_reports_ok(self):
        response = client.get("/api/states/validation")
        assert response.status_code == 200
        assert "matched_states" in response.json()


@requires_artifacts
class TestMapGeoJSON:
    def test_returns_feature_collection(self):
        response = client.get("/api/map/geojson")
        assert response.status_code == 200
        assert response.json()["type"] == "FeatureCollection"


@requires_artifacts
class TestExplainability:
    def test_returns_importance_and_caveat(self):
        response = client.get("/api/model/explainability")
        assert response.status_code == 200

        body = response.json()
        assert "feature_importance" in body
        assert "caveat" in body
        assert "model_generalization" in body


class TestPredictValidation:
    def test_missing_fields_returns_422(self):
        response = client.post("/api/predict", json={"State": "Kerala"})
        assert response.status_code == 422

    def test_negative_amount_rejected(self):
        payload = {
            "State": "Kerala",
            "Account_Balance": 1000.0,
            "Transaction_Amount": -50.0,
            "Merchant_Category": "Groceries",
            "Age": 30,
            "Account_Type": "Savings",
            "Device_Type": "Mobile",
            "Transaction_Device": "QR Code Scanner",
            "Transaction_Date": "15-01-2025",
            "Transaction_Time": "10:00:00",
        }
        response = client.post("/api/predict", json=payload)
        assert response.status_code == 422


@requires_model
@requires_artifacts
class TestPredictScoring:
    VALID_PAYLOAD = {
        "State": "Maharashtra",
        "Account_Balance": 52000.0,
        "Transaction_Amount": 4200.0,
        "Merchant_Category": "Electronics",
        "Age": 34,
        "Account_Type": "Savings",
        "Device_Type": "Mobile",
        "Transaction_Device": "QR Code Scanner",
        "Transaction_Date": "15-01-2025",
        "Transaction_Time": "22:41:03",
    }

    def test_returns_probability_between_0_and_1(self):
        response = client.post("/api/predict", json=self.VALID_PAYLOAD)
        assert response.status_code == 200

        body = response.json()
        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert "caveat" in body
        assert body["threshold_used"] > 0

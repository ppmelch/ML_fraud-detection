"""Integration tests for AnomalyPipeline: unsupervised end-to-end run."""

import numpy as np
import pytest

from backend.src.data.data_loader import TransactionDataLoader
from backend.src.data.data_splitter import DataSplitter
from backend.src.pipeline import AnomalyPipeline


@pytest.fixture
def pipeline(sample_transactions, tmp_path):
    path = tmp_path / "dataset.csv"
    sample_transactions.to_csv(path, index=False)
    return AnomalyPipeline(
        model_name="isolation_forest",
        loader=TransactionDataLoader(path=path),
        splitter=DataSplitter(test_days=30, validation_days=20),
    )


class TestRun:
    def test_end_to_end_produces_all_windows(self, pipeline):
        output = pipeline.run()

        assert pipeline.is_fitted is True
        assert set(output["metrics"]) == {"train", "validation", "test"}
        assert 0.0 <= output["threshold"] <= 1.0
        assert output["contamination"] == pytest.approx(0.02)
        assert len(output["feature_columns"]) > 0

    def test_test_window_carries_diagnostics(self, pipeline):
        output = pipeline.run()
        test_metrics = output["metrics"]["test"]

        for key in (
            "score_rank_curve",
            "top_anomalies",
            "feature_contrast",
            "heuristic_alignment",
        ):
            assert key in test_metrics

    def test_predict_score_before_run_raises(self, pipeline, sample_transactions):
        with pytest.raises(RuntimeError):
            pipeline.predict_score(sample_transactions.head(1))

    def test_predict_score_after_run(self, pipeline, sample_transactions):
        pipeline.run()
        scored = pipeline.predict_score(sample_transactions.head(3))

        assert len(scored) == 3
        assert scored.between(0, 1).all()

    def test_score_frame_attaches_flag_columns(self, pipeline, sample_transactions):
        pipeline.run()
        out = pipeline.score_frame(sample_transactions)

        assert "anomaly_score" in out.columns
        assert set(np.unique(out["is_anomaly"])).issubset({0, 1})


class TestRealData:
    def test_real_dataset_smoke(self, full_dataset, tmp_path):
        path = tmp_path / "dataset.csv"
        full_dataset.to_csv(path, index=False)

        output = AnomalyPipeline(
            model_name="isolation_forest",
            loader=TransactionDataLoader(path=path),
        ).run()

        assert output["metrics"]["test"]["n_samples"] > 0

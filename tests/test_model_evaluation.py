"""Tests for AnomalyEvaluation: score-distribution shapes, NaN/type safety."""

import numpy as np
import pandas as pd
import pytest

from backend.src.modeling.model_evaluation import AnomalyEvaluation


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    return np.clip(rng.beta(2, 20, 300), 0, 1)


class TestEvaluate:
    def test_shape_and_types(self, scores):
        flags = (scores >= np.quantile(scores, 0.98)).astype(int)
        result = AnomalyEvaluation().evaluate(scores, flags)

        assert result["n_samples"] == 300
        assert result["n_flagged"] == int(flags.sum())
        assert 0.0 <= result["flagged_rate"] <= 1.0
        assert set(result["score_percentiles"]) == {"p50", "p90", "p95", "p99"}
        hist = result["score_histogram"]
        assert len(hist["bin_centers"]) == len(hist["counts"])
        assert sum(hist["counts"]) == 300

    def test_curves_flag_adds_rank_curve(self, scores):
        flags = np.zeros_like(scores, dtype=int)
        result = AnomalyEvaluation().evaluate(scores, flags, curves=True)

        curve = result["score_rank_curve"]
        assert len(curve["rank"]) == len(curve["score"])
        assert curve["score"] == sorted(curve["score"], reverse=True)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            AnomalyEvaluation().evaluate([0.1, 0.2], [0])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            AnomalyEvaluation().evaluate([], [])

    def test_no_nan_in_payload(self, scores):
        result = AnomalyEvaluation().evaluate(scores, np.zeros_like(scores, dtype=int))
        text = str(result)
        assert "nan" not in text.lower()


class TestTopAnomaliesAndContrast:
    def test_top_anomalies_are_sorted_and_carry_columns(self):
        data = pd.DataFrame(
            {
                "TransactionID": [f"TX{i}" for i in range(10)],
                "TransactionAmount": np.arange(10.0),
                "Channel": ["ATM"] * 10,
            }
        )
        s = np.arange(10.0) / 10.0
        top = AnomalyEvaluation().top_anomalies(data, s, n=3)

        assert [r["TransactionID"] for r in top] == ["TX9", "TX8", "TX7"]
        assert top[0]["anomaly_score"] == pytest.approx(0.9)

    def test_feature_contrast_ranks_by_std_gap(self):
        frame = pd.DataFrame(
            {"a": [0, 0, 0, 0, 10, 10], "b": [1, 1, 1, 1, 1, 1]}
        )
        flags = np.array([0, 0, 0, 0, 1, 1])
        contrast = AnomalyEvaluation().feature_contrast(frame, flags)

        assert contrast[0]["feature"] == "a"
        assert contrast[-1]["feature"] == "b"

    def test_heuristic_alignment_keys(self):
        data = pd.DataFrame(
            {
                "LoginAttempts": np.random.default_rng(1).integers(1, 6, 100),
                "TransactionAmount": np.random.default_rng(2).uniform(1, 100, 100),
                "AccountBalance": np.random.default_rng(3).uniform(100, 5000, 100),
                "Hour": np.random.default_rng(4).integers(0, 24, 100),
                "TransactionDuration": np.random.default_rng(5).uniform(1, 300, 100),
            }
        )
        s = np.random.default_rng(6).uniform(0, 1, 100)
        result = AnomalyEvaluation().heuristic_alignment(data, s)

        assert set(result) >= {"spearman", "overlap_at_flagged", "note"}
        assert -1.0 <= result["spearman"] <= 1.0

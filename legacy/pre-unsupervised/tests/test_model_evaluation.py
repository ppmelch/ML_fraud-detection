"""Tests for ModelEvaluation: metric correctness, NaN/type safety, guards."""

import json

import numpy as np
import pytest

from backend.src.modeling.model_evaluation import ModelEvaluation


class TestEvaluate:
    def test_perfect_predictions(self):
        y_true = [0, 0, 1, 1]
        y_prob = [0.0, 0.1, 0.9, 1.0]
        y_pred = [0, 0, 1, 1]

        result = ModelEvaluation().evaluate(y_true, y_pred, y_prob)

        assert result["roc_auc"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["confusion_matrix"] == [[2, 0], [0, 2]]

    def test_single_class_returns_none_auc_not_nan(self):
        y_true = [0, 0, 0, 0]
        y_pred = [0, 0, 0, 0]
        y_prob = [0.1, 0.2, 0.1, 0.3]

        result = ModelEvaluation().evaluate(y_true, y_pred, y_prob)

        assert result["roc_auc"] is None
        assert result["pr_auc"] is None
        # None serializes to JSON null; NaN would break JSON.parse downstream.
        json.dumps(result)

    def test_confusion_matrix_orientation(self):
        # 1 TN, 1 FP, 1 FN, 1 TP
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]
        y_prob = [0.1, 0.6, 0.4, 0.9]

        result = ModelEvaluation().evaluate(y_true, y_pred, y_prob)

        assert result["confusion_matrix"] == [[1, 1], [1, 1]]
        assert result["true_negatives"] == 1
        assert result["false_positives"] == 1
        assert result["false_negatives"] == 1
        assert result["true_positives"] == 1

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            ModelEvaluation().evaluate([0, 1], [0], [0.1, 0.9])

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            ModelEvaluation().evaluate([], [], [])

    def test_no_numpy_scalars_in_output(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 50)
        y_prob = rng.random(50)
        y_pred = (y_prob >= 0.5).astype(int)

        result = ModelEvaluation().evaluate(y_true, y_pred, y_prob, curves=True)

        # json.dumps raises TypeError on numpy scalars / arrays.
        json.dumps(result)

    def test_pr_auc_baseline_equals_positive_rate(self):
        y_true = [0] * 90 + [1] * 10
        y_pred = [0] * 100
        y_prob = [0.1] * 100

        result = ModelEvaluation().evaluate(y_true, y_pred, y_prob)

        assert result["pr_auc_baseline"] == pytest.approx(0.1)


class TestCurvesAndDensity:
    def test_roc_curve_points_equal_length(self):
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 2, 500)
        y_prob = rng.random(500)

        curve = ModelEvaluation().roc_curve_points(y_true, y_prob)

        assert len(curve["fpr"]) == len(curve["tpr"])
        assert len(curve["fpr"]) <= 200

    def test_probability_density_keys_and_lengths(self):
        rng = np.random.default_rng(2)
        y_true = rng.integers(0, 2, 200)
        y_prob = rng.random(200)

        density = ModelEvaluation().probability_density(y_true, y_prob, bins=10)

        assert set(density) == {"legit_x", "legit_y", "fraud_x", "fraud_y"}
        assert len(density["legit_x"]) == len(density["legit_y"]) == 10
        json.dumps(density)

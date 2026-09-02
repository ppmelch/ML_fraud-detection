"""Tests for ModelLoader: bundle round-trip, missing artifact, required keys."""

import pytest

from backend.src.modeling.classification_model import ClassificationModel
from backend.src.modeling.model_loader import ModelLoader


class TestSaveLoad:
    def test_round_trip_preserves_bundle_contents(self, tmp_path, sample_transactions):
        loader = ModelLoader(models_dir=tmp_path)

        model = ClassificationModel("logistic")
        X = sample_transactions[["Age", "Transaction_Amount"]].astype(float)
        y = sample_transactions["Is_Fraud"]
        model.train(X, y)

        loader.save(
            model=model,
            feature_engineer="fake_fe",
            data_preparation="fake_prep",
            threshold=0.37,
            metadata={"note": "test"},
        )

        bundle = loader.load()

        assert bundle["threshold"] == pytest.approx(0.37)
        assert bundle["feature_engineer"] == "fake_fe"
        assert bundle["metadata"]["note"] == "test"
        assert bundle["model"].model_name == "logistic"

    def test_missing_bundle_raises_with_actionable_message(self, tmp_path):
        loader = ModelLoader(models_dir=tmp_path)

        with pytest.raises(FileNotFoundError, match="train_pipeline"):
            loader.load()

    def test_exists_reflects_saved_state(self, tmp_path, sample_transactions):
        loader = ModelLoader(models_dir=tmp_path)
        assert loader.exists() is False

        model = ClassificationModel("logistic")
        X = sample_transactions[["Age"]].astype(float)
        y = sample_transactions["Is_Fraud"]
        model.train(X, y)

        loader.save(model, "fe", "prep", 0.5)
        assert loader.exists() is True

    def test_corrupt_bundle_missing_keys_raises(self, tmp_path):
        import joblib

        tmp_path.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": "x"}, tmp_path / ModelLoader.DEFAULT_FILENAME)

        loader = ModelLoader(models_dir=tmp_path)

        with pytest.raises(ValueError, match="missing keys"):
            loader.load()

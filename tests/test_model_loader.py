"""Tests for ModelLoader + AnomalyModel bundle: round-trip, percentile store."""

import numpy as np
import pandas as pd
import pytest

from backend.src.modeling.anomaly_model import AnomalyModel
from backend.src.modeling.model_loader import ModelLoader


@pytest.fixture
def fitted_model():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    return AnomalyModel("isolation_forest").fit(X), X


class TestAnomalyModel:
    def test_scores_in_unit_interval(self, fitted_model):
        model, X = fitted_model
        s = model.anomaly_score(X)

        assert s.min() >= 0.0 and s.max() <= 1.0

    def test_predict_is_binary(self, fitted_model):
        model, X = fitted_model
        flags = model.predict(X)

        assert set(np.unique(flags)).issubset({0, 1})

    def test_unsupported_name_raises(self):
        with pytest.raises(ValueError):
            AnomalyModel("magic")

    def test_lof_and_autoencoder_fit_and_score(self):
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.normal(size=(150, 3)), columns=list("xyz"))

        for name in ("lof", "autoencoder"):
            model = AnomalyModel(name).fit(X)
            s = model.anomaly_score(X)
            assert s.shape == (150,)
            assert s.min() >= 0.0 and s.max() <= 1.0


class TestSaveLoad:
    def test_round_trip_preserves_bundle(self, tmp_path, fitted_model):
        model, _ = fitted_model
        loader = ModelLoader(models_dir=tmp_path)

        loader.save(
            model=model,
            feature_engineer="fe",
            data_preparation="prep",
            threshold=0.42,
            train_scores=model.train_scores_,
            metadata={"note": "test"},
        )
        bundle = loader.load()

        assert bundle["threshold"] == pytest.approx(0.42)
        assert bundle["feature_engineer"] == "fe"
        assert bundle["model"].model_name == "isolation_forest"
        assert len(bundle["train_scores"]) == len(model.train_scores_)
        assert np.all(np.diff(bundle["train_scores"]) >= 0)

    def test_missing_bundle_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="train_pipeline"):
            ModelLoader(models_dir=tmp_path).load()

    def test_corrupt_bundle_missing_keys_raises(self, tmp_path):
        import joblib

        joblib.dump({"model": "x"}, tmp_path / ModelLoader.DEFAULT_FILENAME)

        with pytest.raises(ValueError, match="missing keys"):
            ModelLoader(models_dir=tmp_path).load()

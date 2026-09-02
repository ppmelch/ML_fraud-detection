"""
Live scoring service backing ``POST /api/predict``.

Loads the saved model bundle once (estimator, fitted feature engineer, fitted
encoder, operating threshold) and reuses it for every request, applying the
identical transformation pipeline used during training — see the module
docstring of :mod:`backend.src.pipeline` for why that identity matters.
"""

from functools import lru_cache

import pandas as pd

from backend.src.data.data_loader import TransactionDataLoader
from backend.src.modeling.model_loader import ModelLoader


class PredictionService:
    """
    Score a single raw transaction with the persisted model bundle.

    Attributes
    ----------
    bundle : dict
        Loaded bundle: ``model``, ``feature_engineer``, ``data_preparation``,
        ``threshold``, ``metadata``.
    """

    def __init__(self, bundle: dict) -> None:
        """
        Initialize the service.

        Parameters
        ----------
        bundle : dict
            Model bundle as returned by ``ModelLoader.load``.
        """
        self.bundle = bundle

    def predict(self, transaction: dict) -> dict:
        """
        Score one transaction.

        Parameters
        ----------
        transaction : dict
            Raw transaction fields, matching
            :class:`~backend.src.api.schemas.TransactionInput`.

        Returns
        -------
        dict
            ``fraud_probability``, ``is_fraud_prediction``, ``threshold_used``,
            ``model_name``.
        """
        raw = pd.DataFrame([transaction])

        enriched = TransactionDataLoader().add_temporal_features(raw)
        enriched = self.bundle["feature_engineer"].transform(enriched)
        X = self.bundle["data_preparation"].transform(enriched)

        probability = float(self.bundle["model"].predict_proba(X)[0])
        threshold = float(self.bundle["threshold"])

        return {
            "fraud_probability": probability,
            "is_fraud_prediction": probability >= threshold,
            "threshold_used": threshold,
            "model_name": self.bundle["model"].model_name,
        }


@lru_cache(maxsize=1)
def get_prediction_service() -> PredictionService:
    """
    Load the persisted model bundle once and cache the service.

    Returns
    -------
    PredictionService
        Ready to score requests.

    Raises
    ------
    FileNotFoundError
        If no trained bundle exists yet — run
        ``python -m scripts.train_pipeline`` first.
    """
    bundle = ModelLoader().load()

    return PredictionService(bundle)

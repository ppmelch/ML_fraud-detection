"""
Live scoring service backing ``POST /api/predict``.

Loads the saved anomaly bundle once (wrapper model, fitted feature engineer,
fitted encoder, operating threshold, sorted train scores) and reuses it for
every request, applying the identical transformation pipeline used during
training — see :mod:`backend.src.pipeline` for why that identity matters.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from backend.src.data.data_loader import TransactionDataLoader
from backend.src.modeling.config import IDENTIFIER_COLUMNS
from backend.src.modeling.model_loader import ModelLoader

#: Placeholder for identifier columns absent from a scoring request. An
#: unseen key resolves to a zero frequency and the global deviation fallback,
#: which is the correct behaviour for a never-before-seen device/merchant.
_UNKNOWN = "UNKNOWN"


class PredictionService:
    """
    Score a single raw transaction with the persisted anomaly bundle.

    Attributes
    ----------
    bundle : dict
        Loaded bundle: ``model``, ``feature_engineer``, ``data_preparation``,
        ``threshold``, ``train_scores``, ``metadata``.
    """

    def __init__(self, bundle: dict) -> None:
        """
        Initialize the service.

        Parameters
        ----------
        bundle : dict
            Model bundle as returned by :meth:`ModelLoader.load`.
        """
        self.bundle = bundle
        self._train_scores = np.sort(
            np.asarray(bundle.get("train_scores", []), dtype=float)
        )

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
            ``anomaly_score``, ``is_anomaly``, ``threshold_used``,
            ``model_name`` and ``percentile``.
        """
        raw = pd.DataFrame([transaction])

        for column in IDENTIFIER_COLUMNS:
            if column not in raw.columns or pd.isna(raw.at[0, column]):
                raw[column] = _UNKNOWN

        enriched = TransactionDataLoader().add_temporal_features(raw)
        enriched = self.bundle["feature_engineer"].transform(enriched)
        X = self.bundle["data_preparation"].transform(enriched)

        model = self.bundle["model"]
        score = float(model.anomaly_score(X)[0])
        threshold = float(self.bundle["threshold"])

        return {
            "anomaly_score": score,
            "is_anomaly": bool(score >= threshold),
            "threshold_used": threshold,
            "model_name": model.model_name,
            "percentile": self._percentile(score),
        }

    def _percentile(self, score: float) -> float:
        """
        Position a score against the training score distribution.

        Parameters
        ----------
        score : float
            Anomaly score in ``[0, 1]``.

        Returns
        -------
        float
            Percentile in ``[0, 100]``; the share of training transactions
            scoring at or below ``score``.
        """
        if self._train_scores.size == 0:
            return 0.0

        rank = int(np.searchsorted(self._train_scores, score, side="right"))

        return round(100.0 * rank / self._train_scores.size, 4)


@lru_cache(maxsize=1)
def get_prediction_service() -> PredictionService:
    """
    Load the persisted bundle once and cache the service.

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
    return PredictionService(ModelLoader().load())

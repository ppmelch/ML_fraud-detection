"""
Unsupervised anomaly-detection model wrapper.

One class, :class:`AnomalyModel`, wraps three scikit-learn estimators behind
the :class:`~backend.src.modeling.base_model.BaseModel` interface:

``isolation_forest``
    :class:`sklearn.ensemble.IsolationForest`; raw score is
    ``-estimator.score_samples(X)``.
``lof``
    :class:`sklearn.neighbors.LocalOutlierFactor` with ``novelty=True`` so it
    can score unseen rows; raw score is ``-estimator.score_samples(Xs)``.
``autoencoder``
    :class:`sklearn.neural_network.MLPRegressor` trained to reconstruct the
    standardized feature matrix (X -> X); raw score is the per-row
    reconstruction MSE.

``lof`` and ``autoencoder`` operate on standardized features, so the wrapper
owns a :class:`~sklearn.preprocessing.StandardScaler`. IsolationForest is
scale-invariant and skips it.

On :meth:`fit` the wrapper also learns a min-max normaliser that maps the raw
score onto ``[0, 1]``, sets ``threshold_`` to the ``1 - contamination``
quantile of the train anomaly scores, and stores the sorted train anomaly
scores so a live prediction can be expressed as a percentile.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from backend.src.modeling.base_model import BaseModel
from backend.src.modeling.config import CONTAMINATION, MODEL_CONFIG

#: Model names that require standardized input.
_SCALED_MODELS = ("lof", "autoencoder")

#: Every supported model name.
SUPPORTED_MODELS = ("isolation_forest", "lof", "autoencoder")


class AnomalyModel(BaseModel):
    """
    Fit and serve a single unsupervised anomaly detector.

    Attributes
    ----------
    model_name : str
        One of ``isolation_forest``, ``lof`` or ``autoencoder``.
    config : dict
        Estimator hyperparameters, ``MODEL_CONFIG[model_name]`` merged with
        any constructor overrides.
    contamination : float
        Fraction of training rows treated as anomalous; drives ``threshold_``.
    estimator : object
        The wrapped scikit-learn estimator.
    scaler : StandardScaler or None
        Fitted on train for ``lof`` / ``autoencoder``; ``None`` for
        IsolationForest.
    score_min_, score_max_ : float
        Min-max bounds of the raw train score, used to map raw scores to
        ``[0, 1]``.
    threshold_ : float
        Anomaly-score cutoff, the ``1 - contamination`` quantile of the train
        anomaly scores.
    train_scores_ : np.ndarray
        Sorted train anomaly scores, kept for percentile lookup at predict
        time.
    is_fitted : bool
        Whether :meth:`fit` has run.
    """

    def __init__(
        self,
        model_name: str,
        contamination: float = CONTAMINATION,
        **kwargs,
    ) -> None:
        """
        Initialize the wrapper and build the underlying estimator.

        Parameters
        ----------
        model_name : str
            ``isolation_forest``, ``lof`` or ``autoencoder``.
        contamination : float, optional
            Anomaly fraction used for ``threshold_``. Defaults to
            ``CONTAMINATION``.
        **kwargs
            Hyperparameter overrides merged over ``MODEL_CONFIG[model_name]``.

        Raises
        ------
        ValueError
            If ``model_name`` is not supported.
        """
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported anomaly model {model_name!r}; "
                f"expected one of {SUPPORTED_MODELS}"
            )

        self.model_name = model_name
        self.contamination = float(contamination)
        self.config = {**MODEL_CONFIG.get(model_name, {}), **kwargs}

        self.scaler: StandardScaler | None = (
            StandardScaler() if model_name in _SCALED_MODELS else None
        )

        if model_name == "isolation_forest":
            self.estimator = IsolationForest(**self.config)
        elif model_name == "lof":
            self.estimator = LocalOutlierFactor(**self.config)
        elif model_name == "autoencoder":
            self.estimator = MLPRegressor(**self.config)

        self.score_min_: float = 0.0
        self.score_max_: float = 1.0
        self.threshold_: float = 1.0
        self.train_scores_: np.ndarray = np.array([], dtype=float)
        self.feature_columns_: list[str] = []
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> "AnomalyModel":
        """
        Fit the scaler (if used), the estimator, and the score calibration.

        Parameters
        ----------
        X : pd.DataFrame
            Training feature matrix.
        y : pd.Series or None, optional
            Ignored; the dataset is unlabelled.

        Returns
        -------
        AnomalyModel
            The fitted instance.
        """
        self.feature_columns_ = list(X.columns)
        values = self._matrix(X)

        if self.scaler is not None:
            values = self.scaler.fit_transform(values)

        if self.model_name == "autoencoder":
            self.estimator.fit(values, values)
        else:
            self.estimator.fit(values)

        raw = self._raw_from_matrix(values)

        self.score_min_ = float(np.min(raw))
        self.score_max_ = float(np.max(raw))

        scores = self._normalise(raw)
        self.threshold_ = float(
            np.quantile(scores, 1.0 - self.contamination)
        )
        self.train_scores_ = np.sort(scores)
        self.is_fitted = True

        return self

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------
    def raw_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Estimator-native anomaly score, higher meaning more anomalous.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix in the training layout.

        Returns
        -------
        np.ndarray
            One raw score per row.
        """
        values = self._matrix(X)

        if self.scaler is not None:
            values = self.scaler.transform(values)

        return self._raw_from_matrix(values)

    def anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Min-max normalise the raw score onto ``[0, 1]``, clipped.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray
            Anomaly score per row in ``[0, 1]``.
        """
        return self._normalise(self.raw_score(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Flag rows whose anomaly score reaches ``threshold_``.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray
            Integer ``{0, 1}`` array.
        """
        return (self.anomaly_score(X) >= self.threshold_).astype(int)

    def percentile(self, scores: np.ndarray | list | float) -> np.ndarray:
        """
        Position each score against the sorted train anomaly scores.

        Parameters
        ----------
        scores : array-like or float
            Anomaly score(s) in ``[0, 1]``.

        Returns
        -------
        np.ndarray
            Percentile(s) in ``[0, 100]``; the share of training rows scoring
            at or below the given score.
        """
        arr = np.atleast_1d(np.asarray(scores, dtype=float))

        if self.train_scores_.size == 0:
            return np.zeros_like(arr)

        ranks = np.searchsorted(self.train_scores_, arr, side="right")

        return 100.0 * ranks / self.train_scores_.size

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        """
        Reindex ``X`` to the fitted column layout and return a float array.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray
            2-D float array with columns in the fitted order.
        """
        if self.feature_columns_:
            X = X.reindex(columns=self.feature_columns_, fill_value=0.0)

        return np.asarray(X, dtype=float)

    def _raw_from_matrix(self, values: np.ndarray) -> np.ndarray:
        """
        Compute the estimator-native score from an already-scaled matrix.

        Parameters
        ----------
        values : np.ndarray
            Feature matrix, standardized when the model uses a scaler.

        Returns
        -------
        np.ndarray
            Raw anomaly score per row.
        """
        if self.model_name == "autoencoder":
            reconstructed = self.estimator.predict(values)
            return np.mean((values - reconstructed) ** 2, axis=1)

        return -self.estimator.score_samples(values)

    def _normalise(self, raw: np.ndarray) -> np.ndarray:
        """
        Map raw scores to ``[0, 1]`` with the fitted min-max bounds.

        Parameters
        ----------
        raw : np.ndarray
            Raw anomaly scores.

        Returns
        -------
        np.ndarray
            Clipped scores in ``[0, 1]``.
        """
        span = self.score_max_ - self.score_min_

        if span <= 0:
            return np.zeros_like(raw, dtype=float)

        return np.clip((raw - self.score_min_) / span, 0.0, 1.0)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save_model(self, filename: str, models_dir: Path) -> Path:
        """
        Persist the whole wrapper instance with joblib.

        Parameters
        ----------
        filename : str
            Artifact file name.
        models_dir : Path
            Destination directory; created if absent.

        Returns
        -------
        Path
            The written path.
        """
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        path = models_dir / filename
        joblib.dump(self, path)

        return path

    def load_model(self, filename: str, models_dir: Path) -> "AnomalyModel":
        """
        Restore this instance from a saved wrapper.

        Parameters
        ----------
        filename : str
            Artifact file name.
        models_dir : Path
            Directory holding the artifact.

        Returns
        -------
        AnomalyModel
            This instance, repopulated.

        Raises
        ------
        FileNotFoundError
            If the artifact does not exist.
        TypeError
            If the stored payload is not an :class:`AnomalyModel`.
        """
        path = Path(models_dir) / filename

        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")

        data = joblib.load(path)

        if not isinstance(data, AnomalyModel):
            raise TypeError(
                f"Unsupported model artifact payload: {type(data).__name__}"
            )

        self.__dict__.update(data.__dict__)

        return self

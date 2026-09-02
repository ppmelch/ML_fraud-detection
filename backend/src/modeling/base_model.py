"""
Abstract interface for the unsupervised anomaly detectors.

There is no fraud label in this project. A model does not predict a class
probability; it produces an ``anomaly_score`` in ``[0, 1]`` saying how
unusual a transaction looks relative to the training window, and a binary
``predict`` flag obtained by comparing that score to a fitted threshold.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    Base class every anomaly-detection wrapper implements.

    Concrete subclasses wrap a single scikit-learn estimator and expose a
    uniform ``fit`` / ``anomaly_score`` / ``predict`` surface so the pipeline
    and the API never branch on the estimator type.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "BaseModel":
        """
        Fit the detector on the training feature matrix.

        Parameters
        ----------
        X : pd.DataFrame
            Training feature matrix.
        y : pd.Series or None, optional
            Ignored. Present only so the signature matches the supervised
            scikit-learn convention; the dataset is unlabelled.

        Returns
        -------
        BaseModel
            The fitted instance, for chaining.
        """
        raise NotImplementedError

    @abstractmethod
    def anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Score how anomalous each row is, on a ``[0, 1]`` scale.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix in the training layout.

        Returns
        -------
        np.ndarray
            One float per row in ``[0, 1]``; higher is more anomalous.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Flag each row as anomalous (1) or normal (0).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix in the training layout.

        Returns
        -------
        np.ndarray
            Integer array of ``{0, 1}`` flags.
        """
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Alias of :meth:`anomaly_score`.

        Kept so callers written against the old classifier API keep working;
        the returned value is the anomaly score, not a calibrated probability.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray
            The anomaly score per row.
        """
        return self.anomaly_score(X)

    def train(self, X: pd.DataFrame, y: pd.Series | None = None) -> "BaseModel":
        """
        Thin alias of :meth:`fit` for backwards compatibility.

        Parameters
        ----------
        X : pd.DataFrame
            Training feature matrix.
        y : pd.Series or None, optional
            Ignored.

        Returns
        -------
        BaseModel
            The fitted instance.
        """
        return self.fit(X, y)

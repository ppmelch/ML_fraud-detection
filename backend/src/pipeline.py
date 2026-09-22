"""
End-to-end unsupervised anomaly-detection pipeline.

``AnomalyPipeline`` is the single place that wires

    load -> temporal split -> feature engineering -> encode (X only) ->
    fit AnomalyModel on train -> score train/val/test -> threshold from
    train scores -> evaluate each window

Every fitted object (feature engineer, encoder, model with its threshold) is
kept on the instance, so :meth:`predict_score` can score a brand-new raw
frame with exactly the transformation the model was trained on.
"""

import numpy as np
import pandas as pd

from backend.src.data.data_loader import TransactionDataLoader
from backend.src.data.data_preparation import DataPreparation
from backend.src.data.data_splitter import DataSplitter
from backend.src.data.feature_engineering import FeatureEngineer
from backend.src.modeling.anomaly_model import AnomalyModel
from backend.src.modeling.config import CONTAMINATION, DEFAULT_MODEL
from backend.src.modeling.contamination_threshold import ContaminationThreshold
from backend.src.modeling.model_evaluation import AnomalyEvaluation


class AnomalyPipeline:
    """
    Train and evaluate the anomaly detector end to end.

    Attributes
    ----------
    model_name : str
        Which estimator to build (``isolation_forest``, ``lof``,
        ``autoencoder``).
    contamination : float
        Target flagged fraction; drives the operating threshold.
    loader : TransactionDataLoader
        Reads the raw dataset and derives its temporal columns.
    splitter : DataSplitter
        Produces the chronological train / validation / test windows.
    feature_engineer : FeatureEngineer
        Fitted on train, applied to every window.
    data_preparation : DataPreparation
        Fitted encoder, frozen column layout.
    model : AnomalyModel
        Fitted anomaly-model wrapper.
    threshold : float
        Operating threshold, the ``1 - contamination`` quantile of the train
        anomaly scores.
    is_fitted : bool
        Whether :meth:`run` has completed.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        contamination: float = CONTAMINATION,
        loader: TransactionDataLoader | None = None,
        splitter: DataSplitter | None = None,
    ) -> None:
        """
        Initialize the pipeline.

        Parameters
        ----------
        model_name : str, optional
            Estimator to train. Defaults to ``DEFAULT_MODEL``.
        contamination : float, optional
            Target flagged fraction. Defaults to ``CONTAMINATION``.
        loader : TransactionDataLoader, optional
            Custom loader, mainly for testing with a different dataset path.
        splitter : DataSplitter, optional
            Custom splitter, mainly for testing with different window sizes.
        """
        self.model_name = model_name
        self.contamination = float(contamination)
        self.loader = loader or TransactionDataLoader()
        self.splitter = splitter or DataSplitter()

        self.feature_engineer = FeatureEngineer()
        self.data_preparation = DataPreparation()
        self.model: AnomalyModel | None = None
        self.threshold: float = 1.0
        self.is_fitted = False

    def run(self) -> dict:
        """
        Execute the full pipeline: load, split, engineer, fit, score, evaluate.

        Returns
        -------
        dict
            ``split`` (window sizes/dates), ``threshold``, ``contamination``,
            ``metrics`` (``train`` / ``validation`` / ``test`` from
            :class:`AnomalyEvaluation`, the test entry also carrying
            ``score_rank_curve``, ``top_anomalies``, ``feature_contrast`` and
            ``heuristic_alignment``), ``feature_columns``, and — held for
            downstream interpretability without re-deriving — ``X_train``,
            ``X_test``, ``data_test`` and ``test_scores``.
        """
        data = self.loader.load()

        train, validation, test = self.splitter.temporal_split(data)
        split_summary = self.splitter.describe_temporal_split(
            train, validation, test
        )

        train_enriched = self.feature_engineer.fit_transform(train)
        valid_enriched = self.feature_engineer.transform(validation)
        test_enriched = self.feature_engineer.transform(test)

        X_train = self.data_preparation.fit_transform(train_enriched)
        X_valid = self.data_preparation.transform(valid_enriched)
        X_test = self.data_preparation.transform(test_enriched)

        self.model = AnomalyModel(
            self.model_name, contamination=self.contamination
        )
        self.model.fit(X_train)

        train_scores = self.model.anomaly_score(X_train)
        valid_scores = self.model.anomaly_score(X_valid)
        test_scores = self.model.anomaly_score(X_test)

        self.threshold = ContaminationThreshold(
            train_scores, self.contamination
        ).resolve()
        self.model.threshold_ = self.threshold

        evaluator = AnomalyEvaluation()

        def flags(scores: np.ndarray) -> np.ndarray:
            return (scores >= self.threshold).astype(int)

        metrics = {
            "train": evaluator.evaluate(train_scores, flags(train_scores)),
            "validation": evaluator.evaluate(
                valid_scores, flags(valid_scores)
            ),
            "test": evaluator.evaluate(
                test_scores, flags(test_scores), curves=True
            ),
        }

        test_flags = flags(test_scores)
        metrics["test"]["top_anomalies"] = evaluator.top_anomalies(
            test_enriched.reset_index(drop=True), test_scores, n=25
        )
        metrics["test"]["feature_contrast"] = evaluator.feature_contrast(
            X_test.reset_index(drop=True), test_flags
        )
        metrics["test"]["heuristic_alignment"] = evaluator.heuristic_alignment(
            test_enriched.reset_index(drop=True), test_scores
        )

        self.is_fitted = True

        return {
            "split": split_summary,
            "threshold": float(self.threshold),
            "contamination": self.contamination,
            "metrics": metrics,
            "feature_columns": list(self.data_preparation.feature_columns_),
            "X_train": X_train,
            "X_test": X_test,
            "data_test": test_enriched,
            "test_scores": test_scores,
        }

    def score_frame(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Attach ``anomaly_score`` and ``is_anomaly`` to a raw frame.

        Used by the training script to score the whole dataset for the
        analytics artifacts.

        Parameters
        ----------
        raw_data : pd.DataFrame
            Transactions in the raw CSV schema.

        Returns
        -------
        pd.DataFrame
            ``raw_data`` enriched with temporal columns, ``anomaly_score``
            and ``is_anomaly``.

        Raises
        ------
        RuntimeError
            If called before :meth:`run`.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("AnomalyPipeline.score_frame called before run")

        enriched = self.loader.add_temporal_features(raw_data)
        featured = self.feature_engineer.transform(enriched)
        X = self.data_preparation.transform(featured)

        scores = self.model.anomaly_score(X)

        out = enriched.copy()
        out["anomaly_score"] = scores
        out["is_anomaly"] = (scores >= self.threshold).astype(int)

        return out

    def predict_score(self, raw_data: pd.DataFrame) -> pd.Series:
        """
        Anomaly score per row for new, raw transactions.

        Parameters
        ----------
        raw_data : pd.DataFrame
            Transactions in the raw CSV schema (before temporal columns are
            derived).

        Returns
        -------
        pd.Series
            Anomaly score in ``[0, 1]`` per row, indexed like ``raw_data``.

        Raises
        ------
        RuntimeError
            If called before :meth:`run`.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("AnomalyPipeline.predict_score called before run")

        enriched = self.loader.add_temporal_features(raw_data)
        featured = self.feature_engineer.transform(enriched)
        X = self.data_preparation.transform(featured)

        return pd.Series(self.model.anomaly_score(X), index=raw_data.index)

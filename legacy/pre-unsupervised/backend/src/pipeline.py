"""
End-to-end fraud detection pipeline: dataset to trained, evaluated model.

``FraudPipeline`` is the single place that wires

    load -> temporal split -> feature engineering -> encode -> train ->
    threshold on validation -> evaluate on test

Every stage's fitted object (feature engineer, encoder, threshold) is kept on
the instance, so the same pipeline can score a brand-new frame with
:meth:`predict` using exactly the transformation the model was trained on.
"""

import pandas as pd

from backend.src.data.data_loader import TransactionDataLoader
from backend.src.data.data_preparation import DataPreparation
from backend.src.data.data_splitter import DataSplitter
from backend.src.data.feature_engineering import FeatureEngineer
from backend.src.modeling.classification_model import ClassificationModel
from backend.src.modeling.config import DEFAULT_MODEL, RANDOM_STATE
from backend.src.modeling.model_evaluation import ModelEvaluation
from backend.src.modeling.threshold_optimization import ThresholdOptimizer
from backend.src.utils.utils import compute_scale_pos_weight


class FraudPipeline:
    """
    Train and evaluate the fraud detection model end to end.

    Attributes
    ----------
    model_name : str
        Which estimator to build (``'logistic'``, ``'random_forest'``,
        ``'xgboost'``, ``'lightgbm'``).
    loader : TransactionDataLoader
        Reads the raw dataset and derives its temporal columns.
    splitter : DataSplitter
        Produces the chronological train / validation / test windows.
    feature_engineer : FeatureEngineer
        Fitted on train, applied to every window.
    data_preparation : DataPreparation
        Fitted encoder, frozen column layout.
    model : ClassificationModel
        Trained estimator wrapper.
    threshold : float
        Operating threshold chosen on the validation window.
    is_fitted : bool
        Whether :meth:`run` has completed.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        loader: TransactionDataLoader | None = None,
        splitter: DataSplitter | None = None,
    ) -> None:
        """
        Initialize the pipeline.

        Parameters
        ----------
        model_name : str, optional
            Estimator to train. Defaults to ``DEFAULT_MODEL``.
        loader : TransactionDataLoader, optional
            Custom loader, mainly for testing with a different dataset path.
        splitter : DataSplitter, optional
            Custom splitter, mainly for testing with different window sizes.
        """
        self.model_name = model_name
        self.loader = loader or TransactionDataLoader()
        self.splitter = splitter or DataSplitter()

        self.feature_engineer = FeatureEngineer()
        self.data_preparation = DataPreparation()
        self.model: ClassificationModel | None = None
        self.threshold: float = 0.5
        self.is_fitted = False

    def run(self) -> dict:
        """
        Execute the full pipeline: load, split, engineer, train, evaluate.

        Returns
        -------
        dict
            ``split``: window sizes and dates (from
            :meth:`DataSplitter.describe_temporal_split`).
            ``threshold``: operating threshold chosen on validation.
            ``metrics``: ``{"train", "validation", "test"}``, each the output
            of :meth:`ModelEvaluation.evaluate`.
            ``feature_columns``: encoded column layout used by the model.
            ``X_test`` / ``y_test``: held out for interpretability, not
            re-derived elsewhere so the SHAP sample matches what was scored.
        """
        data = self.loader.load()

        train, validation, test = self.splitter.temporal_split(data)

        split_summary = self.splitter.describe_temporal_split(
            train, validation, test
        )

        train_enriched = self.feature_engineer.fit_transform(train)
        valid_enriched = self.feature_engineer.transform(validation)
        test_enriched = self.feature_engineer.transform(test)

        X_train, y_train = self.data_preparation.fit_transform(train_enriched)
        X_valid, y_valid = self.data_preparation.transform_xy(valid_enriched)
        X_test, y_test = self.data_preparation.transform_xy(test_enriched)

        scale_pos_weight = compute_scale_pos_weight(y_train)

        extra_params = (
            {"scale_pos_weight": scale_pos_weight}
            if self.model_name in ("xgboost",)
            else {}
        )

        self.model = ClassificationModel(self.model_name, **extra_params)
        self.model.train(X_train, y_train)

        evaluator = ModelEvaluation()

        train_prob = self.model.predict_proba(X_train)
        valid_prob = self.model.predict_proba(X_valid)
        test_prob = self.model.predict_proba(X_test)

        # The threshold is selected on validation only. Handing the optimizer
        # the test labels would tune the cutoff on the number it is later
        # used to report.
        self.threshold = ThresholdOptimizer(y_valid, valid_prob).optimize_threshold(
            n_trials=50
        )

        metrics = {
            "train": evaluator.evaluate(
                y_train, (train_prob >= self.threshold).astype(int), train_prob
            ),
            "validation": evaluator.evaluate(
                y_valid, (valid_prob >= self.threshold).astype(int), valid_prob
            ),
            "test": evaluator.evaluate(
                y_test,
                (test_prob >= self.threshold).astype(int),
                test_prob,
                curves=True,
            ),
        }
        metrics["test"]["probability_density"] = evaluator.probability_density(
            y_test, test_prob
        )

        self.is_fitted = True

        return {
            "split": split_summary,
            "threshold": float(self.threshold),
            "metrics": metrics,
            "feature_columns": list(self.data_preparation.feature_columns_),
            "X_test": X_test,
            "y_test": y_test,
            "test_probabilities": test_prob,
        }

    def predict_proba(self, raw_data: pd.DataFrame) -> pd.Series:
        """
        Score new, raw transactions with the fitted pipeline.

        Parameters
        ----------
        raw_data : pd.DataFrame
            Transactions in the same raw schema as the training CSV (before
            temporal columns are derived).

        Returns
        -------
        pd.Series
            Predicted fraud probability per row, indexed like ``raw_data``.

        Raises
        ------
        RuntimeError
            If called before :meth:`run`.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("FraudPipeline.predict_proba called before run")

        enriched = self.loader.add_temporal_features(raw_data)
        enriched = self.feature_engineer.transform(enriched)
        X = self.data_preparation.transform(enriched)

        return pd.Series(self.model.predict_proba(X), index=raw_data.index)

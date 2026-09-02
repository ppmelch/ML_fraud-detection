"""
SHAP-based interpretability for the fraud model.

What this module produces is a description of *how the model uses its inputs*.
It is not evidence that those inputs predict anything. A model with a test
ROC-AUC of 0.50 still yields a perfectly well-formed SHAP ranking — of noise.
For that reason every payload built here carries the test ROC-AUC and PR-AUC
alongside the rankings, and a ``generalizes`` flag, so a consumer cannot read
the bar chart without reading the caveat.
"""

import numpy as np
import pandas as pd

from backend.src.modeling.config import RANDOM_STATE, SHAP_SAMPLE_SIZE


class ModelExplainer:
    """
    Compute feature importance and feature impact for a trained model.

    Attributes
    ----------
    model : ClassificationModel
        Trained wrapper whose ``.model`` attribute is the fitted estimator.
    sample_size : int
        Number of rows sampled for the SHAP computation.
    random_state : int
        Seed for the sampling, so the explanation reproduces.
    """

    def __init__(
        self,
        model,
        sample_size: int = SHAP_SAMPLE_SIZE,
        random_state: int = RANDOM_STATE,
    ) -> None:
        """
        Initialize the explainer.

        Parameters
        ----------
        model : ClassificationModel
            Trained model wrapper.
        sample_size : int, optional
            Rows to sample before running SHAP.
        random_state : int, optional
            Sampling seed.
        """
        self.model = model
        self.sample_size = sample_size
        self.random_state = random_state

    def shap_values(self, X: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Compute SHAP values for a sample of ``X``.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix in the layout the model was trained on.

        Returns
        -------
        tuple[np.ndarray, pd.DataFrame]
            The SHAP value matrix for the positive class and the sampled rows
            it was computed on, aligned row for row.

        Raises
        ------
        ImportError
            If the ``shap`` package is not installed.
        """
        import shap

        sample = self._sample(X)

        explainer = shap.TreeExplainer(self.model.model)
        values = explainer.shap_values(sample)

        # Binary tree models return either (n, f) for the positive class or a
        # per-class list / (n, f, 2) array depending on the library version.
        values = np.asarray(values)

        if values.ndim == 3:
            values = values[:, :, -1]

        return values, sample

    def feature_importance(self, X: pd.DataFrame, top_n: int | None = None) -> list[dict]:
        """
        Rank features by mean absolute SHAP value.

        Falls back to the estimator's native importance when SHAP is not
        available for the model type, and says which method it used.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        top_n : int, optional
            Return only the top ``n`` features. All features when omitted.

        Returns
        -------
        list[dict]
            ``{"feature", "importance", "method"}`` sorted by descending
            importance.
        """
        try:
            values, sample = self.shap_values(X)

            importance = np.abs(values).mean(axis=0)
            method = "shap_mean_abs"
            columns = list(sample.columns)

        except Exception:
            importance, columns = self._native_importance(X)
            method = "native_importance"

        records = [
            {
                "feature": str(name),
                "importance": float(value),
                "method": method,
            }
            for name, value in zip(columns, importance)
        ]

        records.sort(key=lambda r: -r["importance"])

        return records[:top_n] if top_n else records

    def feature_impact(self, X: pd.DataFrame, top_n: int | None = None) -> list[dict]:
        """
        Describe the direction in which each feature pushes the fraud score.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        top_n : int, optional
            Return only the top ``n`` features by mean absolute SHAP value.

        Returns
        -------
        list[dict]
            Per feature: ``mean_abs_shap``, ``mean_shap``, ``direction``
            (``"increases_fraud_risk"`` / ``"decreases_fraud_risk"`` /
            ``"neutral"``) and ``value_correlation`` — the correlation between
            the feature's value and its SHAP contribution, which says whether
            a *higher* value raises the score.

        Raises
        ------
        ImportError
            If the ``shap`` package is not installed.
        """
        values, sample = self.shap_values(X)

        records = []

        for index, name in enumerate(sample.columns):
            contribution = values[:, index]
            feature_values = sample.iloc[:, index].to_numpy(dtype=float)

            mean_shap = float(contribution.mean())

            if np.std(feature_values) > 0 and np.std(contribution) > 0:
                correlation = float(
                    np.corrcoef(feature_values, contribution)[0, 1]
                )
            else:
                correlation = 0.0

            records.append(
                {
                    "feature": str(name),
                    "mean_abs_shap": float(np.abs(contribution).mean()),
                    "mean_shap": mean_shap,
                    "direction": (
                        "increases_fraud_risk"
                        if mean_shap > 0
                        else "decreases_fraud_risk"
                        if mean_shap < 0
                        else "neutral"
                    ),
                    "value_correlation": correlation,
                }
            )

        records.sort(key=lambda r: -r["mean_abs_shap"])

        return records[:top_n] if top_n else records

    def explain(
        self,
        X: pd.DataFrame,
        top_n: int = 20,
        test_metrics: dict | None = None,
    ) -> dict:
        """
        Build the interpretability payload served by the API.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix to explain, normally the test window.
        top_n : int, optional
            Number of features to include in each ranking.
        test_metrics : dict, optional
            Test-set metrics used to qualify the explanation. When the test
            ROC-AUC is at chance level, ``generalizes`` is False and the
            caveat is stated in the payload.

        Returns
        -------
        dict
            ``method``, ``sample_size``, ``feature_importance``,
            ``feature_impact``, ``model_generalization`` and ``caveat``.
        """
        importance = self.feature_importance(X, top_n=top_n)

        try:
            impact = self.feature_impact(X, top_n=top_n)
            method = "shap_tree_explainer"
        except Exception as error:
            impact = []
            method = f"native_importance ({type(error).__name__})"

        roc_auc = (test_metrics or {}).get("roc_auc")
        pr_auc = (test_metrics or {}).get("pr_auc")
        baseline = (test_metrics or {}).get("pr_auc_baseline")

        generalizes = roc_auc is not None and roc_auc >= 0.55

        payload = {
            "method": method,
            "sample_size": int(min(self.sample_size, len(X))),
            "model_name": getattr(self.model, "model_name", None),
            "feature_importance": importance,
            "feature_impact": impact,
            "model_generalization": {
                "test_roc_auc": roc_auc,
                "test_pr_auc": pr_auc,
                "pr_auc_baseline": baseline,
                "generalizes": bool(generalizes),
            },
        }

        payload["caveat"] = (
            "Feature importance describes how this model uses its inputs. "
            "It is not evidence of predictive power."
            if generalizes
            else "The model does not separate fraud from non-fraud on the "
            "held-out period (test ROC-AUC at chance level). These rankings "
            "describe how the model allocated attention to noise and must "
            "not be read as drivers of fraud."
        )

        return payload

    def _sample(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Draw a reproducible row sample for the SHAP computation.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        pd.DataFrame
            At most ``sample_size`` rows.
        """
        if len(X) <= self.sample_size:
            return X

        return X.sample(n=self.sample_size, random_state=self.random_state)

    def _native_importance(self, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """
        Fall back to the estimator's own importance attribute.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix, used for its column names.

        Returns
        -------
        tuple[np.ndarray, list[str]]
            Importance values and the matching feature names.

        Raises
        ------
        AttributeError
            If the estimator exposes neither importances nor coefficients.
        """
        estimator = self.model.model
        columns = list(X.columns)

        if hasattr(estimator, "feature_importances_"):
            return np.asarray(estimator.feature_importances_), columns

        if hasattr(estimator, "coef_"):
            return np.abs(np.asarray(estimator.coef_)).ravel(), columns

        raise AttributeError(
            f"{type(estimator).__name__} exposes no feature importance"
        )

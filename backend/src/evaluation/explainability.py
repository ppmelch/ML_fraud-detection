"""
Interpretability for the unsupervised anomaly detector.

What this module produces is a description of *which inputs move the anomaly
score*. There is no label, so it cannot say those inputs predict fraud — only
that the model reacts to them.

- ``isolation_forest`` gets SHAP via :class:`shap.TreeExplainer`.
- ``lof`` and ``autoencoder`` have no tree structure, so importance comes from
  a hand-rolled permutation test: shuffle one column, measure the mean
  absolute change in anomaly score.

Direction labels are ``increases_anomaly_score`` / ``decreases_anomaly_score``.
Every payload carries a ``score_summary`` and a static ``caveat`` in place of
the old supervised ``generalizes`` flag.
"""

import numpy as np
import pandas as pd

from backend.src.modeling.config import RANDOM_STATE, SHAP_SAMPLE_SIZE

#: Stated on every payload; the explanation is descriptive, not evidential.
CAVEAT = (
    "This dataset has no fraud label. Feature importance describes which "
    "inputs move the model's anomaly score, not which inputs cause fraud. "
    "Read the ranking as 'what the model reacts to', and corroborate any "
    "flagged transaction with independent review."
)


class AnomalyExplainer:
    """
    Rank and describe the features driving a model's anomaly score.

    Attributes
    ----------
    model : AnomalyModel
        Fitted wrapper exposing ``anomaly_score`` and ``model_name``.
    sample_size : int
        Rows sampled before running SHAP or permutation importance.
    random_state : int
        Seed for the sampling and the permutation shuffles.
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
        model : AnomalyModel
            Fitted anomaly-model wrapper.
        sample_size : int, optional
            Rows to sample before explaining.
        random_state : int, optional
            Sampling / shuffle seed.
        """
        self.model = model
        self.sample_size = sample_size
        self.random_state = random_state

    # ------------------------------------------------------------------
    # SHAP path (isolation_forest)
    # ------------------------------------------------------------------
    def _shap_importance(
        self, sample: pd.DataFrame
    ) -> tuple[list[dict], list[dict]]:
        """
        SHAP importance and direction for a tree-based detector.

        Parameters
        ----------
        sample : pd.DataFrame
            Rows to explain.

        Returns
        -------
        tuple[list[dict], list[dict]]
            ``(feature_importance, feature_impact)`` records.

        Raises
        ------
        Exception
            Propagated if ``shap`` is unavailable or the estimator is
            unsupported; the caller falls back to permutation importance.
        """
        import shap

        explainer = shap.TreeExplainer(self.model.estimator)
        values = np.asarray(explainer.shap_values(sample))

        if values.ndim == 3:
            values = values[:, :, -1]

        importance = np.abs(values).mean(axis=0)
        mean_signed = values.mean(axis=0)

        imp_records = [
            {"feature": str(c), "importance": float(v), "method": "shap_mean_abs"}
            for c, v in zip(sample.columns, importance)
        ]
        imp_records.sort(key=lambda r: -r["importance"])

        impact_records = []
        for idx, column in enumerate(sample.columns):
            contribution = values[:, idx]
            feature_values = sample.iloc[:, idx].to_numpy(dtype=float)

            if np.std(feature_values) > 0 and np.std(contribution) > 0:
                correlation = float(
                    np.corrcoef(feature_values, contribution)[0, 1]
                )
            else:
                correlation = 0.0

            impact_records.append(
                {
                    "feature": str(column),
                    "mean_abs_impact": float(np.abs(contribution).mean()),
                    "mean_signed_impact": float(mean_signed[idx]),
                    "direction": self._direction(mean_signed[idx]),
                    "value_correlation": correlation,
                }
            )

        impact_records.sort(key=lambda r: -r["mean_abs_impact"])

        return imp_records, impact_records

    # ------------------------------------------------------------------
    # Permutation path (lof, autoencoder)
    # ------------------------------------------------------------------
    def _permutation_importance(
        self, sample: pd.DataFrame
    ) -> tuple[list[dict], list[dict]]:
        """
        Permutation importance and direction for a non-tree detector.

        Each column is shuffled in turn; the mean absolute change in anomaly
        score is its importance, and the mean signed change (shuffled minus
        baseline) gives the direction.

        Parameters
        ----------
        sample : pd.DataFrame
            Rows to explain.

        Returns
        -------
        tuple[list[dict], list[dict]]
            ``(feature_importance, feature_impact)`` records.
        """
        rng = np.random.default_rng(self.random_state)
        baseline = self.model.anomaly_score(sample)

        imp_records = []
        impact_records = []

        for column in sample.columns:
            permuted = sample.copy()
            permuted[column] = rng.permutation(permuted[column].to_numpy())

            shuffled = self.model.anomaly_score(permuted)
            delta = shuffled - baseline

            imp_records.append(
                {
                    "feature": str(column),
                    "importance": float(np.abs(delta).mean()),
                    "method": "permutation_mean_abs_delta",
                }
            )
            impact_records.append(
                {
                    "feature": str(column),
                    "mean_abs_impact": float(np.abs(delta).mean()),
                    "mean_signed_impact": float(delta.mean()),
                    "direction": self._direction(-delta.mean()),
                    "value_correlation": self._value_correlation(
                        sample[column].to_numpy(dtype=float), baseline
                    ),
                }
            )

        imp_records.sort(key=lambda r: -r["importance"])
        impact_records.sort(key=lambda r: -r["mean_abs_impact"])

        return imp_records, impact_records

    @staticmethod
    def _value_correlation(feature_values: np.ndarray, score: np.ndarray) -> float:
        """
        Correlation between a feature's value and the baseline anomaly score.

        Parameters
        ----------
        feature_values : np.ndarray
            Feature column.
        score : np.ndarray
            Baseline anomaly score per row.

        Returns
        -------
        float
            Pearson correlation, or ``0.0`` when either side is constant.
        """
        if np.std(feature_values) == 0 or np.std(score) == 0:
            return 0.0

        return float(np.corrcoef(feature_values, score)[0, 1])

    @staticmethod
    def _direction(signed: float) -> str:
        """
        Map a signed impact to a direction label.

        Parameters
        ----------
        signed : float
            Mean signed effect on the anomaly score.

        Returns
        -------
        str
            ``increases_anomaly_score``, ``decreases_anomaly_score`` or
            ``neutral``.
        """
        if signed > 0:
            return "increases_anomaly_score"
        if signed < 0:
            return "decreases_anomaly_score"
        return "neutral"

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------
    def explain(self, X: pd.DataFrame, top_n: int = 20) -> dict:
        """
        Build the interpretability payload served by the API.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix to explain, normally the test window.
        top_n : int, optional
            Number of features to include in each ranking.

        Returns
        -------
        dict
            ``method``, ``sample_size``, ``model_name``,
            ``feature_importance``, ``feature_impact``, ``score_summary``
            (``mean``/``p95``/``p99``/``flagged_rate``) and ``caveat``.
        """
        sample = self._sample(X)

        method = "permutation_importance"

        if getattr(self.model, "model_name", None) == "isolation_forest":
            try:
                importance, impact = self._shap_importance(sample)
                method = "shap_tree_explainer"
            except Exception as error:  # noqa: BLE001 - fall back gracefully
                importance, impact = self._permutation_importance(sample)
                method = f"permutation_importance ({type(error).__name__})"
        else:
            importance, impact = self._permutation_importance(sample)

        scores = self.model.anomaly_score(sample)
        threshold = float(getattr(self.model, "threshold_", 1.0))

        return {
            "method": method,
            "sample_size": int(len(sample)),
            "model_name": getattr(self.model, "model_name", None),
            "feature_importance": importance[:top_n],
            "feature_impact": impact[:top_n],
            "score_summary": {
                "mean": float(np.mean(scores)),
                "p95": float(np.percentile(scores, 95)),
                "p99": float(np.percentile(scores, 99)),
                "flagged_rate": float(np.mean(scores >= threshold)),
            },
            "caveat": CAVEAT,
        }

    def _sample(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Draw a reproducible row sample.

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

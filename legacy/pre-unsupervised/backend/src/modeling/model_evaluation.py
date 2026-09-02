"""
Classification metrics for a heavily imbalanced fraud target.

Accuracy is computed and returned for completeness, but it is never the
headline: predicting "not fraud" for every transaction in this dataset scores
about 0.95. The metrics that carry information about the positive class are
ROC-AUC, PR-AUC (average precision) and the precision/recall/F1 triple
computed for ``Is_Fraud = 1``.

Every value returned is a plain Python scalar or list. numpy scalars and NaN
survive `json.dump` in forms the browser refuses to parse, so they are cast
and made explicit here rather than at the edge.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class ModelEvaluation:
    """
    Compute and package binary classification metrics.

    The class is stateless: every method takes the labels and predictions it
    needs, so the same instance can evaluate train, validation and test.
    """

    def evaluate(
        self,
        y_true,
        y_pred,
        y_prob,
        curves: bool = False,
    ) -> dict:
        """
        Evaluate a set of predictions against the true labels.

        Parameters
        ----------
        y_true : array-like of int
            True binary labels (0 = legitimate, 1 = fraud).
        y_pred : array-like of int
            Predicted binary labels at the operating threshold.
        y_prob : array-like of float
            Predicted probability of the positive class.
        curves : bool, optional
            If True, also return the ROC and precision-recall curves.

        Returns
        -------
        dict
            Metrics keyed by name. ``confusion_matrix`` is ``[[TN, FP],
            [FN, TP]]`` as integers. ``roc_auc`` and ``pr_auc`` are ``None``
            when only one class is present, which is the honest answer rather
            than a fabricated 0.5.

        Raises
        ------
        ValueError
            If the input arrays have different lengths or are empty.
        """
        y_true = self._as_array(y_true).astype(int)
        y_pred = self._as_array(y_pred).astype(int)
        y_prob = self._as_array(y_prob).astype(float)

        if not (len(y_true) == len(y_pred) == len(y_prob)):
            raise ValueError(
                f"Length mismatch: y_true={len(y_true)}, "
                f"y_pred={len(y_pred)}, y_prob={len(y_prob)}"
            )

        if len(y_true) == 0:
            raise ValueError("Cannot evaluate an empty prediction set")

        single_class = len(np.unique(y_true)) < 2

        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = (int(v) for v in matrix.ravel())

        results = {
            "n_samples": int(len(y_true)),
            "n_positives": int(y_true.sum()),
            "positive_rate": float(y_true.mean()),
            "roc_auc": None if single_class else float(
                roc_auc_score(y_true, y_prob)
            ),
            "pr_auc": None if single_class else float(
                average_precision_score(y_true, y_prob)
            ),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "brier_score": float(brier_score_loss(y_true, y_prob)),
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        }

        # The baseline a PR curve must beat is the prevalence itself. Reporting
        # it next to pr_auc is what makes "0.05" readable as "no better than
        # guessing" instead of "a small number".
        results["pr_auc_baseline"] = results["positive_rate"]

        if curves and not single_class:
            results["roc_curve"] = self.roc_curve_points(y_true, y_prob)
            results["pr_curve"] = self.pr_curve_points(y_true, y_prob)

        return results

    def roc_curve_points(self, y_true, y_prob, max_points: int = 200) -> dict:
        """
        Compute a thinned ROC curve suitable for a chart payload.

        Parameters
        ----------
        y_true : array-like of int
            True binary labels.
        y_prob : array-like of float
            Predicted probability of the positive class.
        max_points : int, optional
            Upper bound on the number of points returned.

        Returns
        -------
        dict
            ``{"fpr": [...], "tpr": [...]}`` with equal-length float lists.
        """
        fpr, tpr, _ = roc_curve(
            self._as_array(y_true), self._as_array(y_prob)
        )

        fpr, tpr = self._thin(fpr, tpr, max_points)

        return {"fpr": fpr, "tpr": tpr}

    def pr_curve_points(self, y_true, y_prob, max_points: int = 200) -> dict:
        """
        Compute a thinned precision-recall curve for a chart payload.

        Parameters
        ----------
        y_true : array-like of int
            True binary labels.
        y_prob : array-like of float
            Predicted probability of the positive class.
        max_points : int, optional
            Upper bound on the number of points returned.

        Returns
        -------
        dict
            ``{"recall": [...], "precision": [...]}`` with equal-length lists.
        """
        precision, recall, _ = precision_recall_curve(
            self._as_array(y_true), self._as_array(y_prob)
        )

        recall, precision = self._thin(recall, precision, max_points)

        return {"recall": recall, "precision": precision}

    def probability_density(
        self, y_true, y_prob, bins: int = 40
    ) -> dict:
        """
        Bin predicted probabilities separately for each true class.

        Parameters
        ----------
        y_true : array-like of int
            True binary labels.
        y_prob : array-like of float
            Predicted probability of the positive class.
        bins : int, optional
            Number of histogram bins spanning the observed score range.

        Returns
        -------
        dict
            ``{"legit_x", "legit_y", "fraud_x", "fraud_y"}``. When the two
            score distributions overlap completely the model separates
            nothing, which this view makes visible at a glance.
        """
        y_true = self._as_array(y_true).astype(int)
        y_prob = self._as_array(y_prob).astype(float)

        low = float(y_prob.min())
        high = float(y_prob.max())

        if high <= low:
            high = low + 1e-6

        edges = np.linspace(low, high, bins + 1)
        centers = ((edges[:-1] + edges[1:]) / 2).tolist()

        density = {}

        for label, name in ((0, "legit"), (1, "fraud")):
            scores = y_prob[y_true == label]

            if len(scores) == 0:
                counts = np.zeros(bins)
            else:
                counts, _ = np.histogram(scores, bins=edges, density=True)

            density[f"{name}_x"] = [float(v) for v in centers]
            density[f"{name}_y"] = [float(v) for v in counts]

        return density

    @staticmethod
    def _as_array(values) -> np.ndarray:
        """
        Coerce pandas or list input to a flat numpy array.

        Parameters
        ----------
        values : array-like
            Input sequence.

        Returns
        -------
        np.ndarray
            One-dimensional array.
        """
        if isinstance(values, (pd.Series, pd.DataFrame)):
            values = values.to_numpy()

        return np.asarray(values).ravel()

    @staticmethod
    def _thin(x, y, max_points: int) -> tuple[list, list]:
        """
        Down-sample two aligned curve arrays to at most ``max_points``.

        Parameters
        ----------
        x, y : np.ndarray
            Aligned curve coordinates.
        max_points : int
            Upper bound on the returned length.

        Returns
        -------
        tuple[list, list]
            Down-sampled float lists of equal length, endpoints preserved.
        """
        if len(x) <= max_points:
            index = np.arange(len(x))
        else:
            index = np.unique(
                np.linspace(0, len(x) - 1, max_points).astype(int)
            )

        return (
            [float(v) for v in np.asarray(x)[index]],
            [float(v) for v in np.asarray(y)[index]],
        )

"""
Evaluation for an unsupervised anomaly detector.

There is no ground truth, so nothing here reports precision or recall against
a label. What it reports is the *shape of the score distribution* on a
window, which rows sit at the top of it, how the flagged rows differ from the
rest, and — as an explicit sanity check, not a scoreboard — how far the
model's ranking agrees with a transparent hand-built heuristic.

Every value returned is a plain Python scalar or list. numpy scalars and NaN
survive ``json.dump`` in forms the browser refuses to parse, so they are cast
and made explicit here.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

#: Raw columns surfaced with each top anomaly, when present on the frame.
_TOP_ANOMALY_COLUMNS = [
    "TransactionID",
    "AccountID",
    "Transaction_Timestamp",
    "TransactionAmount",
    "TransactionType",
    "Channel",
    "USState",
    "CustomerAge",
    "CustomerOccupation",
    "TransactionDuration",
    "LoginAttempts",
    "AccountBalance",
]


def _floats(values) -> list[float]:
    """
    Cast an array-like to a list of plain Python floats.

    Parameters
    ----------
    values : array-like
        Numbers to cast.

    Returns
    -------
    list[float]
        JSON-safe float list.
    """
    return [float(v) for v in np.asarray(values, dtype=float).ravel()]


class AnomalyEvaluation:
    """
    Summarise anomaly scores and flags for one data window.

    The class is stateless; each method takes the scores and flags it needs
    so a single instance can evaluate train, validation and test.
    """

    def evaluate(
        self,
        scores,
        flags,
        curves: bool = False,
    ) -> dict:
        """
        Describe the score distribution and the flagged fraction.

        Parameters
        ----------
        scores : array-like of float
            Anomaly scores in ``[0, 1]``.
        flags : array-like of int
            Binary flags from the model threshold.
        curves : bool, optional
            If True, add ``score_rank_curve`` — the first 200 points of the
            descending sorted-score curve, for a chart.

        Returns
        -------
        dict
            ``n_samples``, ``n_flagged``, ``flagged_rate``, ``score_mean``,
            ``score_std``, ``score_percentiles`` (p50/p90/p95/p99),
            ``score_histogram`` (``bin_centers``/``counts``), and optionally
            ``score_rank_curve`` (``rank``/``score``).

        Raises
        ------
        ValueError
            If the inputs differ in length or are empty.
        """
        scores = np.asarray(scores, dtype=float).ravel()
        flags = np.asarray(flags).ravel().astype(int)

        if len(scores) != len(flags):
            raise ValueError(
                f"Length mismatch: scores={len(scores)}, flags={len(flags)}"
            )

        if len(scores) == 0:
            raise ValueError("Cannot evaluate an empty score set")

        n = int(len(scores))
        n_flagged = int(flags.sum())

        percentiles = np.percentile(scores, [50, 90, 95, 99])

        result = {
            "n_samples": n,
            "n_flagged": n_flagged,
            "flagged_rate": float(n_flagged / n),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std(ddof=0)),
            "score_percentiles": {
                "p50": float(percentiles[0]),
                "p90": float(percentiles[1]),
                "p95": float(percentiles[2]),
                "p99": float(percentiles[3]),
            },
            "score_histogram": self._histogram(scores, bins=40),
        }

        if curves:
            ordered = np.sort(scores)[::-1][:200]
            result["score_rank_curve"] = {
                "rank": [int(i) for i in range(1, len(ordered) + 1)],
                "score": _floats(ordered),
            }

        return result

    @staticmethod
    def _histogram(scores: np.ndarray, bins: int) -> dict:
        """
        Fixed-range ``[0, 1]`` histogram of anomaly scores.

        Parameters
        ----------
        scores : np.ndarray
            Anomaly scores.
        bins : int
            Number of equal-width bins.

        Returns
        -------
        dict
            ``{"bin_centers": [...], "counts": [...]}``.
        """
        edges = np.linspace(0.0, 1.0, bins + 1)
        counts, _ = np.histogram(scores, bins=edges)
        centers = (edges[:-1] + edges[1:]) / 2

        return {
            "bin_centers": _floats(centers),
            "counts": [int(v) for v in counts],
        }

    def top_anomalies(self, data: pd.DataFrame, scores, n: int = 25) -> list[dict]:
        """
        Return the ``n`` highest-scoring rows with their key raw columns.

        Parameters
        ----------
        data : pd.DataFrame
            The window's raw/enriched transactions, row-aligned with
            ``scores``.
        scores : array-like of float
            Anomaly scores.
        n : int, optional
            How many rows to return.

        Returns
        -------
        list[dict]
            One record per row, ordered by descending score, each carrying
            ``anomaly_score`` plus whichever of the key raw columns exist.
        """
        scores = np.asarray(scores, dtype=float).ravel()
        order = np.argsort(scores)[::-1][:n]

        columns = [c for c in _TOP_ANOMALY_COLUMNS if c in data.columns]
        records = []

        for pos in order:
            row = data.iloc[int(pos)]
            record = {"anomaly_score": float(scores[pos])}

            for column in columns:
                value = row[column]

                if isinstance(value, pd.Timestamp):
                    record[column] = value.isoformat()
                elif isinstance(value, (np.integer,)):
                    record[column] = int(value)
                elif isinstance(value, (np.floating, float)):
                    record[column] = None if pd.isna(value) else float(value)
                else:
                    record[column] = None if pd.isna(value) else str(value)

            records.append(record)

        return records

    def feature_contrast(self, feature_frame: pd.DataFrame, flags) -> list[dict]:
        """
        Standardised mean difference between flagged and normal rows per feature.

        Parameters
        ----------
        feature_frame : pd.DataFrame
            The numeric model matrix for the window, row-aligned with
            ``flags``.
        flags : array-like of int
            Binary flags.

        Returns
        -------
        list[dict]
            Per feature: ``feature``, ``flagged_mean``, ``normal_mean``,
            ``std_gap`` (mean difference divided by the pooled standard
            deviation), sorted by descending absolute ``std_gap``.
        """
        flags = np.asarray(flags).ravel().astype(bool)

        if flags.sum() == 0 or (~flags).sum() == 0:
            return []

        records = []

        for column in feature_frame.columns:
            values = feature_frame[column].to_numpy(dtype=float)
            flagged = values[flags]
            normal = values[~flags]

            spread = float(np.std(values, ddof=0))
            gap = float(flagged.mean() - normal.mean())

            records.append(
                {
                    "feature": str(column),
                    "flagged_mean": float(flagged.mean()),
                    "normal_mean": float(normal.mean()),
                    "std_gap": gap / spread if spread > 0 else 0.0,
                }
            )

        records.sort(key=lambda r: -abs(r["std_gap"]))

        return records

    def heuristic_alignment(self, data: pd.DataFrame, scores) -> dict:
        """
        Agreement between the model ranking and a transparent hand rule.

        This is a **sanity check, not ground truth.** The dataset is
        unlabelled; there is no way to say the model is "right". What we can
        say is whether it ranks transactions roughly the way a simple,
        auditable rule would — many login attempts, an amount far above the
        portfolio's 99th percentile, a high amount-to-balance ratio, a
        night-time hour, an extreme duration. Each sub-rule contributes 0 or 1
        and the rule score is their mean. A high alignment means the model has
        not learned something wildly at odds with basic intuition; a low
        alignment is a prompt to inspect, not a verdict either way.

        Parameters
        ----------
        data : pd.DataFrame
            Enriched transactions for the window, row-aligned with ``scores``.
        scores : array-like of float
            Model anomaly scores.

        Returns
        -------
        dict
            ``spearman`` (rank correlation between model score and rule
            score) and ``overlap_at_flagged`` (Jaccard overlap of the two
            top-k sets, k = rows the model flags).
        """
        scores = np.asarray(scores, dtype=float).ravel()
        n = len(scores)

        rule = np.zeros(n, dtype=float)
        parts = 0

        if "LoginAttempts" in data.columns:
            rule += (data["LoginAttempts"].to_numpy(dtype=float) >= 4).astype(float)
            parts += 1

        if "TransactionAmount" in data.columns:
            amount = data["TransactionAmount"].to_numpy(dtype=float)
            p99 = float(np.percentile(amount, 99))
            rule += (amount > p99).astype(float)
            parts += 1

            if "AccountBalance" in data.columns:
                balance = data["AccountBalance"].to_numpy(dtype=float)
                ratio = amount / (np.abs(balance) + 1.0)
                rule += (ratio > np.percentile(ratio, 95)).astype(float)
                parts += 1

        if "Hour" in data.columns:
            rule += (data["Hour"].to_numpy(dtype=float) < 6).astype(float)
            parts += 1

        if "TransactionDuration" in data.columns:
            duration = data["TransactionDuration"].to_numpy(dtype=float)
            p99 = float(np.percentile(duration, 99))
            rule += (duration > p99).astype(float)
            parts += 1

        if parts:
            rule /= parts

        if np.std(rule) == 0 or np.std(scores) == 0:
            spearman = 0.0
        else:
            spearman = float(spearmanr(scores, rule).correlation)

        if np.isnan(spearman):
            spearman = 0.0

        k = max(1, int(np.round((scores >= np.percentile(scores, 98)).sum())))
        k = min(k, n)
        model_top = set(np.argsort(scores)[::-1][:k].tolist())
        rule_top = set(np.argsort(rule)[::-1][:k].tolist())
        union = model_top | rule_top

        overlap = len(model_top & rule_top) / len(union) if union else 0.0

        return {
            "spearman": spearman,
            "overlap_at_flagged": float(overlap),
            "note": (
                "Sanity check against a transparent hand rule, not ground "
                "truth. The dataset is unlabelled."
            ),
        }

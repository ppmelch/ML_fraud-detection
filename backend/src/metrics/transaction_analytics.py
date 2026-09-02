"""
Portfolio-level anomaly analytics.

Every number here is computed from two columns the pipeline attaches before
calling in: ``anomaly_score`` (the model's ``[0, 1]`` score) and
``is_anomaly`` (the score compared to the operating threshold). There is no
fraud label; "flagged" always means "the model flagged it", never "it was
fraud".

The analytical dashboard answers "what did the model surface, and where", and
that answer is allowed to move when the model is retrained — unlike a
label-based view, which would not.
"""

import numpy as np
import pandas as pd

from backend.src.modeling.config import CITY_TO_STATE

#: Monday-first day names, matching ``pandas.Series.dt.dayofweek``.
DAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

#: Column carrying the model anomaly score.
SCORE_COLUMN = "anomaly_score"

#: Column carrying the model flag (0/1).
FLAG_COLUMN = "is_anomaly"


def day_name(day_of_week) -> str:
    """
    Convert a Monday-first weekday index to its English name.

    Parameters
    ----------
    day_of_week : int or None
        Weekday index in ``[0, 6]``, 0 = Monday.

    Returns
    -------
    str
        Day name, or ``"Unknown"`` if the index is missing or out of range.
    """
    if day_of_week is None or pd.isna(day_of_week):
        return "Unknown"

    index = int(day_of_week)

    return DAY_NAMES[index] if 0 <= index < len(DAY_NAMES) else "Unknown"


def _rate(part: int, total: int) -> float:
    """
    Divide two counts, returning ``0.0`` for the empty bucket.

    Parameters
    ----------
    part : int
        Numerator (flagged count).
    total : int
        Denominator (bucket size).

    Returns
    -------
    float
        ``part / total`` or ``0.0`` when ``total`` is zero. Returning 0.0
        rather than NaN keeps the value JSON-serialisable.
    """
    return float(part) / float(total) if total else 0.0


class TransactionAnalytics:
    """
    Compute the anomaly-summary payload from a scored transaction frame.

    Attributes
    ----------
    data : pd.DataFrame
        Enriched transactions carrying ``anomaly_score`` and ``is_anomaly``.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initialize the analytics view.

        Parameters
        ----------
        data : pd.DataFrame
            Transactions with the temporal columns plus ``anomaly_score`` and
            ``is_anomaly``.

        Raises
        ------
        KeyError
            If either score column is absent.
        """
        missing = [c for c in (SCORE_COLUMN, FLAG_COLUMN) if c not in data.columns]

        if missing:
            raise KeyError(f"TransactionAnalytics missing columns: {missing}")

        self.data = data.copy()
        self.data[FLAG_COLUMN] = self.data[FLAG_COLUMN].astype(int)

    # ------------------------------------------------------------------
    def overview(self) -> dict:
        """
        Headline totals and flagged-activity metrics for the whole frame.

        Returns
        -------
        dict
            Totals, averages, state coverage, observed date range, flagged
            count / rate / amount and the mean anomaly score.
        """
        data = self.data
        total = int(len(data))
        flagged = data[data[FLAG_COLUMN] == 1]

        overview = {
            "total_transactions": total,
            "total_accounts": (
                int(data["AccountID"].nunique())
                if "AccountID" in data.columns
                else 0
            ),
            "total_amount": float(data["TransactionAmount"].sum()),
            "avg_transaction_amount": (
                float(data["TransactionAmount"].mean()) if total else 0.0
            ),
            "avg_account_balance": (
                float(data["AccountBalance"].mean()) if total else 0.0
            ),
            "states_covered": (
                int(data["USState"].nunique())
                if "USState" in data.columns
                else 0
            ),
            "flagged_transactions": int(len(flagged)),
            "flagged_rate": _rate(len(flagged), total),
            "mean_anomaly_score": (
                float(data[SCORE_COLUMN].mean()) if total else 0.0
            ),
            "flagged_amount": float(flagged["TransactionAmount"].sum()),
        }

        if "Transaction_Timestamp" in data.columns:
            overview["period_start"] = (
                data["Transaction_Timestamp"].min().isoformat()
            )
            overview["period_end"] = (
                data["Transaction_Timestamp"].max().isoformat()
            )
        else:
            overview["period_start"] = None
            overview["period_end"] = None

        return overview

    # ------------------------------------------------------------------
    def _bucket_records(
        self, column: str, key: str, label_fn=None, sort_key=None
    ) -> list[dict]:
        """
        Group by ``column`` and compute the per-bucket anomaly metrics.

        Parameters
        ----------
        column : str
            Column to group by.
        key : str
            Output key holding the bucket label (``"bucket"`` or
            ``"category"``).
        label_fn : Callable, optional
            Maps a group value to its label. Identity when omitted.
        sort_key : Callable, optional
            Sort key for the records. Defaults to descending anomaly rate.

        Returns
        -------
        list[dict]
            ``{key, total_transactions, flagged, anomaly_rate, mean_score}``.

        Raises
        ------
        KeyError
            If ``column`` is absent.
        """
        if column not in self.data.columns:
            raise KeyError(f"Column '{column}' not in data")

        grouped = self.data.groupby(column, dropna=False)
        records = []

        for value, frame in grouped:
            total = int(len(frame))
            flagged = int(frame[FLAG_COLUMN].sum())
            label = label_fn(value) if label_fn else value

            records.append(
                {
                    key: (
                        int(label)
                        if isinstance(label, (int, np.integer))
                        else str(label)
                    ),
                    "total_transactions": total,
                    "flagged": flagged,
                    "anomaly_rate": _rate(flagged, total),
                    "mean_score": float(frame[SCORE_COLUMN].mean()),
                }
            )

        records.sort(key=sort_key or (lambda r: -r["anomaly_rate"]))

        return records

    def by_hour(self) -> list[dict]:
        """
        Per-hour anomaly metrics, ordered 0..23.

        Returns
        -------
        list[dict]
            One record per observed hour.
        """
        return self._bucket_records(
            "Hour", "bucket", sort_key=lambda r: r["bucket"]
        )

    def by_day_of_week(self) -> list[dict]:
        """
        Per-weekday anomaly metrics, ordered Monday..Sunday.

        Returns
        -------
        list[dict]
            One record per observed weekday; ``bucket`` is the day name.
        """
        records = self._bucket_records(
            "DayOfWeek", "bucket", sort_key=lambda r: int(r["bucket"])
        )

        for record in records:
            record["bucket"] = day_name(int(record["bucket"]))

        return records

    def by_channel(self) -> list[dict]:
        """
        Per-channel anomaly metrics.

        Returns
        -------
        list[dict]
            One record per channel value.
        """
        return self._bucket_records("Channel", "category")

    def by_transaction_type(self) -> list[dict]:
        """
        Per-transaction-type anomaly metrics.

        Returns
        -------
        list[dict]
            One record per transaction type.
        """
        return self._bucket_records("TransactionType", "category")

    def by_occupation(self) -> list[dict]:
        """
        Per-occupation anomaly metrics.

        Returns
        -------
        list[dict]
            One record per customer occupation.
        """
        return self._bucket_records("CustomerOccupation", "category")

    # ------------------------------------------------------------------
    def amount_distribution(self, bins: int = 20) -> dict:
        """
        Transaction-amount histogram split into normal and flagged rows.

        Parameters
        ----------
        bins : int, optional
            Number of equal-width bins across the observed amount range.

        Returns
        -------
        dict
            ``{"bin_centers", "normal_counts", "flagged_counts"}``, equal
            length.
        """
        amounts = self.data["TransactionAmount"].astype(float).to_numpy()

        low = float(amounts.min())
        high = float(amounts.max())

        if high <= low:
            high = low + 1.0

        edges = np.linspace(low, high, bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2

        flagged_mask = self.data[FLAG_COLUMN].to_numpy() == 1
        normal_counts, _ = np.histogram(amounts[~flagged_mask], bins=edges)
        flagged_counts, _ = np.histogram(amounts[flagged_mask], bins=edges)

        return {
            "bin_centers": [float(v) for v in centers],
            "normal_counts": [int(v) for v in normal_counts],
            "flagged_counts": [int(v) for v in flagged_counts],
        }

    def score_distribution(self, bins: int = 40) -> dict:
        """
        Histogram of the anomaly score over the fixed ``[0, 1]`` range.

        Parameters
        ----------
        bins : int, optional
            Number of equal-width bins.

        Returns
        -------
        dict
            ``{"bin_centers", "counts"}``, equal length.
        """
        scores = self.data[SCORE_COLUMN].astype(float).to_numpy()

        edges = np.linspace(0.0, 1.0, bins + 1)
        counts, _ = np.histogram(scores, bins=edges)
        centers = (edges[:-1] + edges[1:]) / 2

        return {
            "bin_centers": [float(v) for v in centers],
            "counts": [int(v) for v in counts],
        }

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """
        Assemble the full anomaly-summary payload served by the API.

        Returns
        -------
        dict
            ``overview`` plus every breakdown and the two distributions.
        """
        return {
            "overview": self.overview(),
            "by_hour": self.by_hour(),
            "by_day_of_week": self.by_day_of_week(),
            "by_channel": self.by_channel(),
            "by_transaction_type": self.by_transaction_type(),
            "by_occupation": self.by_occupation(),
            "amount_distribution": self.amount_distribution(),
            "score_distribution": self.score_distribution(),
        }


__all__ = ["TransactionAnalytics", "day_name", "DAY_NAMES", "CITY_TO_STATE"]

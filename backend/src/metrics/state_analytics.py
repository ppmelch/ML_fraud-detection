"""
Per-state anomaly analytics for the US choropleth.

Grouped by ``USState``. The two ranking metrics — peak anomaly hour and peak
anomaly day — are *rate*-based and support-floored: ranking by absolute
flagged count would just return each state's busiest hour, and ranking by
rate with no floor returns whichever bucket happened to be smallest.

"Flagged" means the model flagged the transaction. There is no label.
"""

import pandas as pd

from backend.src.metrics.transaction_analytics import (
    FLAG_COLUMN,
    SCORE_COLUMN,
    day_name,
)
from backend.src.modeling.config import (
    MIN_TRANSACTIONS_PER_DAY_BUCKET,
    MIN_TRANSACTIONS_PER_HOUR_BUCKET,
    MIN_TRANSACTIONS_PER_STATE,
)


def _rate(part: int, total: int) -> float:
    """
    Ratio of two counts, ``0.0`` for the empty bucket.

    Parameters
    ----------
    part : int
        Numerator.
    total : int
        Denominator.

    Returns
    -------
    float
        ``part / total`` or ``0.0``.
    """
    return float(part) / float(total) if total else 0.0


class StateAnalytics:
    """
    Compute per-state anomaly metrics from a scored transaction frame.

    Attributes
    ----------
    data : pd.DataFrame
        Transactions carrying ``USState``, ``Hour``, ``DayOfWeek``,
        ``anomaly_score`` and ``is_anomaly``.
    state_column : str
        Column holding the state name.
    min_hour_bucket, min_day_bucket : int
        Support floors for the peak-hour and peak-day rankings.
    min_state_transactions : int
        Below this a state is still reported but marked ``low_support``.
    """

    CHANNEL_COLUMN = "Channel"
    OCCUPATION_COLUMN = "CustomerOccupation"

    def __init__(
        self,
        data: pd.DataFrame,
        state_column: str = "USState",
        min_hour_bucket: int = MIN_TRANSACTIONS_PER_HOUR_BUCKET,
        min_day_bucket: int = MIN_TRANSACTIONS_PER_DAY_BUCKET,
        min_state_transactions: int = MIN_TRANSACTIONS_PER_STATE,
    ) -> None:
        """
        Initialize the state analytics view.

        Parameters
        ----------
        data : pd.DataFrame
            Scored transactions.
        state_column : str, optional
            Name of the state column.
        min_hour_bucket, min_day_bucket : int, optional
            Support floors for the peak rankings.
        min_state_transactions : int, optional
            ``low_support`` threshold.

        Raises
        ------
        KeyError
            If a required column is missing.
        """
        required = [
            state_column,
            "Hour",
            "DayOfWeek",
            "TransactionAmount",
            "AccountBalance",
            self.CHANNEL_COLUMN,
            self.OCCUPATION_COLUMN,
            SCORE_COLUMN,
            FLAG_COLUMN,
        ]
        missing = [c for c in required if c not in data.columns]

        if missing:
            raise KeyError(f"StateAnalytics missing columns: {missing}")

        self.data = data.copy()
        self.data[FLAG_COLUMN] = self.data[FLAG_COLUMN].astype(int)
        self.state_column = state_column
        self.min_hour_bucket = min_hour_bucket
        self.min_day_bucket = min_day_bucket
        self.min_state_transactions = min_state_transactions

    def compute(self) -> list[dict]:
        """
        Build the metric record for every state in the data.

        Returns
        -------
        list[dict]
            One record per state, sorted by descending anomaly rate.
        """
        records = [
            self._state_record(state, frame)
            for state, frame in self.data.groupby(
                self.state_column, dropna=False
            )
        ]

        return sorted(records, key=lambda r: -r["anomaly_rate"])

    def _state_record(self, state, frame: pd.DataFrame) -> dict:
        """
        Build the metric record for a single state.

        Parameters
        ----------
        state : Any
            State name.
        frame : pd.DataFrame
            Transactions in that state.

        Returns
        -------
        dict
            The state's metric record (pre-GeoJSON join).
        """
        total = int(len(frame))
        flagged = int(frame[FLAG_COLUMN].sum())
        flagged_rows = frame[frame[FLAG_COLUMN] == 1]

        peak_hour = self._peak_bucket(frame, "Hour", self.min_hour_bucket)
        peak_day = self._peak_bucket(frame, "DayOfWeek", self.min_day_bucket)

        return {
            "state": str(state),
            "total_transactions": total,
            "flagged": flagged,
            "anomaly_rate": _rate(flagged, total),
            "mean_anomaly_score": float(frame[SCORE_COLUMN].mean()),
            "avg_transaction_amount": (
                float(frame["TransactionAmount"].mean()) if total else 0.0
            ),
            "avg_account_balance": (
                float(frame["AccountBalance"].mean()) if total else 0.0
            ),
            "peak_anomaly_hour": peak_hour["bucket"],
            "peak_anomaly_hour_rate": peak_hour["rate"],
            "peak_anomaly_hour_transactions": peak_hour["total"],
            "peak_anomaly_day": (
                day_name(peak_day["bucket"])
                if peak_day["bucket"] is not None
                else None
            ),
            "peak_anomaly_day_rate": peak_day["rate"],
            "peak_anomaly_day_transactions": peak_day["total"],
            "most_common_flagged_channel": self._top_value(
                flagged_rows, self.CHANNEL_COLUMN
            ),
            "top_flagged_occupation": self._top_value(
                flagged_rows, self.OCCUPATION_COLUMN
            ),
            "low_support": total < self.min_state_transactions,
        }

    def _peak_bucket(
        self, frame: pd.DataFrame, column: str, min_transactions: int
    ) -> dict:
        """
        Bucket of ``column`` with the highest flagged *rate*, support-floored.

        Parameters
        ----------
        frame : pd.DataFrame
            One state's transactions.
        column : str
            ``Hour`` or ``DayOfWeek``.
        min_transactions : int
            Support floor applied before ranking.

        Returns
        -------
        dict
            ``{"bucket", "rate", "total"}``; ``bucket`` and ``rate`` are
            ``None`` when no bucket clears the floor.
        """
        grouped = frame.groupby(column)[FLAG_COLUMN].agg(["count", "sum"])
        eligible = grouped[grouped["count"] >= min_transactions]

        if eligible.empty:
            return {"bucket": None, "rate": None, "total": 0}

        rates = eligible["sum"] / eligible["count"]
        ordered = sorted(rates.items(), key=lambda item: (-item[1], item[0]))
        bucket = ordered[0][0]

        return {
            "bucket": int(bucket),
            "rate": float(rates.loc[bucket]),
            "total": int(eligible.loc[bucket, "count"]),
        }

    @staticmethod
    def _top_value(flagged_rows: pd.DataFrame, column: str) -> str | None:
        """
        Most frequent value of ``column`` among the state's flagged rows.

        Parameters
        ----------
        flagged_rows : pd.DataFrame
            Rows the model flagged for one state.
        column : str
            Column to count.

        Returns
        -------
        str or None
            The modal value (alphabetical tie-break), or ``None`` when the
            state has no flagged rows.
        """
        if flagged_rows.empty:
            return None

        counts = flagged_rows[column].value_counts()
        top = counts.max()
        winners = sorted(str(v) for v in counts[counts == top].index)

        return winners[0]

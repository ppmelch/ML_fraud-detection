"""
State-level fraud analytics for the interactive India map.

The two ranking metrics in here — peak fraud hour and peak fraud day — are
deliberately *rate*-based. Ranking by absolute fraud count would return the
busiest hour of every state, which is a statement about traffic volume, not
about fraud. Ranking by rate without a support floor returns whichever bucket
happened to be smallest. Both guards are applied.
"""

import pandas as pd

from backend.src.metrics.fraud_analytics import day_name, fraud_rate
from backend.src.modeling.config import (
    MIN_TRANSACTIONS_PER_DAY_BUCKET,
    MIN_TRANSACTIONS_PER_HOUR_BUCKET,
    MIN_TRANSACTIONS_PER_STATE,
    TARGET_COLUMN,
)


class StateAnalytics:
    """
    Compute per-state fraud metrics from the observed target column.

    Attributes
    ----------
    data : pd.DataFrame
        Transactions enriched with the temporal columns.
    state_column : str
        Column holding the state name.
    min_hour_bucket : int
        Minimum transactions an hour bucket needs to enter the peak ranking.
    min_day_bucket : int
        Minimum transactions a weekday bucket needs to enter the ranking.
    min_state_transactions : int
        Below this a state is still reported, but flagged as low support.
    """

    #: Fine-grained device column used for "most common device".
    DEVICE_COLUMN = "Transaction_Device"

    #: Coarse device channel, reported alongside the fine-grained device.
    DEVICE_TYPE_COLUMN = "Device_Type"

    #: Merchant column used for "top merchant category associated with fraud".
    MERCHANT_COLUMN = "Merchant_Category"

    def __init__(
        self,
        data: pd.DataFrame,
        state_column: str = "State",
        min_hour_bucket: int = MIN_TRANSACTIONS_PER_HOUR_BUCKET,
        min_day_bucket: int = MIN_TRANSACTIONS_PER_DAY_BUCKET,
        min_state_transactions: int = MIN_TRANSACTIONS_PER_STATE,
    ) -> None:
        """
        Initialize the state analytics view.

        Parameters
        ----------
        data : pd.DataFrame
            Transactions containing ``Is_Fraud``, ``Hour``, ``DayOfWeek``,
            the state column, the device columns and the merchant column.
        state_column : str, optional
            Name of the state column.
        min_hour_bucket : int, optional
            Support floor for the peak-hour ranking.
        min_day_bucket : int, optional
            Support floor for the peak-day ranking.
        min_state_transactions : int, optional
            Support floor below which a state is flagged ``low_support``.

        Raises
        ------
        KeyError
            If any required column is missing.
        """
        required = [
            TARGET_COLUMN,
            state_column,
            "Hour",
            "DayOfWeek",
            "Transaction_Amount",
            "Account_Balance",
            self.DEVICE_COLUMN,
            self.DEVICE_TYPE_COLUMN,
            self.MERCHANT_COLUMN,
        ]

        missing = [c for c in required if c not in data.columns]

        if missing:
            raise KeyError(f"StateAnalytics missing columns: {missing}")

        self.data = data
        self.state_column = state_column
        self.min_hour_bucket = min_hour_bucket
        self.min_day_bucket = min_day_bucket
        self.min_state_transactions = min_state_transactions

    def compute(self) -> list[dict]:
        """
        Compute the metric record for every state present in the data.

        Returns
        -------
        list[dict]
            One record per state, sorted by descending fraud rate. Keys:

            ``state``, ``total_transactions``, ``fraud_cases``,
            ``fraud_rate``, ``avg_transaction_amount``,
            ``avg_account_balance``, ``peak_fraud_hour``,
            ``peak_fraud_hour_rate``, ``peak_fraud_hour_transactions``,
            ``peak_fraud_day``, ``peak_fraud_day_index``,
            ``peak_fraud_day_rate``, ``peak_fraud_day_transactions``,
            ``most_common_fraud_device``, ``most_common_fraud_device_type``,
            ``top_fraud_merchant_category``, ``low_support``.
        """
        records = [
            self._state_record(state, frame)
            for state, frame in self.data.groupby(
                self.state_column, dropna=False
            )
        ]

        return sorted(records, key=lambda r: -r["fraud_rate"])

    def _state_record(self, state, frame: pd.DataFrame) -> dict:
        """
        Build the metric record for a single state.

        Parameters
        ----------
        state : Any
            State name as it appears in the dataset.
        frame : pd.DataFrame
            Transactions belonging to that state.

        Returns
        -------
        dict
            The state's metric record.
        """
        total = int(len(frame))
        fraud_cases = int(frame[TARGET_COLUMN].sum())

        fraud_rows = frame[frame[TARGET_COLUMN] == 1]

        peak_hour = self._peak_bucket(frame, "Hour", self.min_hour_bucket)
        peak_day = self._peak_bucket(frame, "DayOfWeek", self.min_day_bucket)

        return {
            "state": str(state),
            "total_transactions": total,
            "fraud_cases": fraud_cases,
            "fraud_rate": fraud_rate(fraud_cases, total),
            "avg_transaction_amount": (
                float(frame["Transaction_Amount"].mean()) if total else 0.0
            ),
            "avg_account_balance": (
                float(frame["Account_Balance"].mean()) if total else 0.0
            ),
            "avg_fraud_amount": (
                float(fraud_rows["Transaction_Amount"].mean())
                if len(fraud_rows)
                else 0.0
            ),
            "peak_fraud_hour": peak_hour["bucket"],
            "peak_fraud_hour_rate": peak_hour["rate"],
            "peak_fraud_hour_transactions": peak_hour["total"],
            "peak_fraud_hour_cases": peak_hour["fraud_cases"],
            "peak_fraud_day_index": peak_day["bucket"],
            "peak_fraud_day": day_name(peak_day["bucket"]),
            "peak_fraud_day_rate": peak_day["rate"],
            "peak_fraud_day_transactions": peak_day["total"],
            "peak_fraud_day_cases": peak_day["fraud_cases"],
            "most_common_fraud_device": self._top_fraud_value(
                fraud_rows, self.DEVICE_COLUMN
            ),
            "most_common_fraud_device_type": self._top_fraud_value(
                fraud_rows, self.DEVICE_TYPE_COLUMN
            ),
            "top_fraud_merchant_category": self._top_fraud_value(
                fraud_rows, self.MERCHANT_COLUMN
            ),
            "low_support": total < self.min_state_transactions,
        }

    def _peak_bucket(
        self, frame: pd.DataFrame, column: str, min_transactions: int
    ) -> dict:
        """
        Find the bucket of ``column`` with the highest fraud *rate*.

        Buckets with fewer than ``min_transactions`` transactions are removed
        before ranking, so a single fraud in a three-transaction hour cannot
        become the peak. Ties are broken by the lower bucket value, which
        makes the result deterministic across runs.

        Parameters
        ----------
        frame : pd.DataFrame
            Transactions of a single state.
        column : str
            Bucketing column, ``Hour`` or ``DayOfWeek``.
        min_transactions : int
            Support floor applied before ranking.

        Returns
        -------
        dict
            ``{"bucket", "rate", "total", "fraud_cases", "eligible_buckets"}``.
            ``bucket`` and ``rate`` are ``None`` when no bucket clears the
            floor — reported as unknown rather than guessed.
        """
        grouped = frame.groupby(column)[TARGET_COLUMN].agg(["count", "sum"])

        eligible = grouped[grouped["count"] >= min_transactions]

        if eligible.empty:
            return {
                "bucket": None,
                "rate": None,
                "total": 0,
                "fraud_cases": 0,
                "eligible_buckets": 0,
            }

        rates = eligible["sum"] / eligible["count"]

        # Sort by descending rate, then ascending bucket value for a stable tie
        # break, and take the head.
        ordered = sorted(
            rates.items(), key=lambda item: (-item[1], item[0])
        )
        bucket = ordered[0][0]

        return {
            "bucket": int(bucket),
            "rate": float(rates.loc[bucket]),
            "total": int(eligible.loc[bucket, "count"]),
            "fraud_cases": int(eligible.loc[bucket, "sum"]),
            "eligible_buckets": int(len(eligible)),
        }

    @staticmethod
    def _top_fraud_value(fraud_rows: pd.DataFrame, column: str) -> str | None:
        """
        Return the most frequent value of ``column`` among fraudulent rows.

        Parameters
        ----------
        fraud_rows : pd.DataFrame
            Rows with ``Is_Fraud == 1`` for one state.
        column : str
            Column to count.

        Returns
        -------
        str or None
            The modal value, or ``None`` when the state recorded no fraud.
            Ties are broken alphabetically for determinism.
        """
        if fraud_rows.empty:
            return None

        counts = fraud_rows[column].value_counts()

        top = counts.max()
        winners = sorted(str(v) for v in counts[counts == top].index)

        return winners[0]

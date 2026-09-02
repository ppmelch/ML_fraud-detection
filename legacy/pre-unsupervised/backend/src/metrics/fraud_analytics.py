"""
Observed fraud analytics.

Everything in this module is computed from the ``Is_Fraud`` column, never from
model predictions. The analytical dashboard answers "what happened", and the
answer must not move when the model is retrained. Model performance is a
separate question, answered in
:mod:`backend.src.modeling.model_evaluation`.
"""

import pandas as pd

from backend.src.modeling.config import TARGET_COLUMN


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


def day_name(day_of_week: int) -> str:
    """
    Convert a Monday-first weekday index to its English name.

    Parameters
    ----------
    day_of_week : int
        Weekday index in ``[0, 6]``, 0 = Monday.

    Returns
    -------
    str
        Day name, or ``"Unknown"`` if the index is out of range.
    """
    if day_of_week is None:
        return "Unknown"

    index = int(day_of_week)

    if 0 <= index < len(DAY_NAMES):
        return DAY_NAMES[index]

    return "Unknown"


def fraud_rate(fraud_cases: int, total: int) -> float:
    """
    Divide fraud cases by total transactions, guarding the empty bucket.

    Parameters
    ----------
    fraud_cases : int
        Number of transactions with ``Is_Fraud == 1``.
    total : int
        Number of transactions in the bucket.

    Returns
    -------
    float
        The ratio, or ``0.0`` when the bucket is empty. Returning 0.0 rather
        than NaN keeps the value JSON-serialisable; an empty bucket never
        reaches the ranking helpers, which enforce a minimum support.
    """
    if not total:
        return 0.0

    return float(fraud_cases) / float(total)


class FraudAnalytics:
    """
    Portfolio-level fraud statistics computed from the observed target.

    Attributes
    ----------
    data : pd.DataFrame
        Transactions enriched with the temporal columns.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initialize the analytics view.

        Parameters
        ----------
        data : pd.DataFrame
            Transactions containing ``Is_Fraud`` and the derived temporal
            columns.

        Raises
        ------
        KeyError
            If the target column is missing.
        """
        if TARGET_COLUMN not in data.columns:
            raise KeyError(f"Missing target column '{TARGET_COLUMN}'")

        self.data = data

    def overview(self) -> dict:
        """
        Compute headline fraud metrics for the whole dataset.

        Returns
        -------
        dict
            Totals, fraud rate, monetary exposure and the observed date range.
        """
        total = int(len(self.data))
        fraud_cases = int(self.data[TARGET_COLUMN].sum())

        fraud_rows = self.data[self.data[TARGET_COLUMN] == 1]

        overview = {
            "total_transactions": total,
            "fraud_cases": fraud_cases,
            "legitimate_cases": total - fraud_cases,
            "fraud_rate": fraud_rate(fraud_cases, total),
            "total_amount": float(self.data["Transaction_Amount"].sum()),
            "fraud_amount": float(fraud_rows["Transaction_Amount"].sum()),
            "avg_transaction_amount": float(
                self.data["Transaction_Amount"].mean()
            ) if total else 0.0,
            "avg_fraud_amount": float(
                fraud_rows["Transaction_Amount"].mean()
            ) if len(fraud_rows) else 0.0,
            "avg_account_balance": float(
                self.data["Account_Balance"].mean()
            ) if total else 0.0,
            "states_covered": int(self.data["State"].nunique()),
        }

        if "Transaction_Timestamp" in self.data.columns:
            overview["period_start"] = (
                self.data["Transaction_Timestamp"].min().isoformat()
            )
            overview["period_end"] = (
                self.data["Transaction_Timestamp"].max().isoformat()
            )

        return overview

    def by_column(self, column: str, min_transactions: int = 0) -> list[dict]:
        """
        Fraud rate per category of a column, ranked by rate.

        Parameters
        ----------
        column : str
            Column to group by.
        min_transactions : int, optional
            Buckets with fewer transactions than this are excluded from the
            result. A category with three transactions and one fraud has a
            33% rate and no meaning.

        Returns
        -------
        list[dict]
            One record per category with ``category``, ``total_transactions``,
            ``fraud_cases`` and ``fraud_rate``, sorted by descending rate.

        Raises
        ------
        KeyError
            If ``column`` is not in the data.
        """
        if column not in self.data.columns:
            raise KeyError(f"Column '{column}' not in data")

        grouped = self.data.groupby(column, dropna=False)[TARGET_COLUMN].agg(
            ["count", "sum"]
        )

        records = [
            {
                "category": str(category),
                "total_transactions": int(row["count"]),
                "fraud_cases": int(row["sum"]),
                "fraud_rate": fraud_rate(int(row["sum"]), int(row["count"])),
            }
            for category, row in grouped.iterrows()
            if int(row["count"]) >= min_transactions
        ]

        return sorted(records, key=lambda r: -r["fraud_rate"])

    def by_hour(self) -> list[dict]:
        """
        Fraud rate for each hour of the day, ordered 0..23.

        Returns
        -------
        list[dict]
            One record per observed hour with ``hour``,
            ``total_transactions``, ``fraud_cases`` and ``fraud_rate``.
        """
        grouped = self.data.groupby("Hour")[TARGET_COLUMN].agg(["count", "sum"])

        return [
            {
                "hour": int(hour),
                "total_transactions": int(row["count"]),
                "fraud_cases": int(row["sum"]),
                "fraud_rate": fraud_rate(int(row["sum"]), int(row["count"])),
            }
            for hour, row in grouped.sort_index().iterrows()
        ]

    def by_day_of_week(self) -> list[dict]:
        """
        Fraud rate for each weekday, ordered Monday..Sunday.

        Returns
        -------
        list[dict]
            One record per observed weekday with ``day_of_week``, ``day``,
            ``total_transactions``, ``fraud_cases`` and ``fraud_rate``.
        """
        grouped = self.data.groupby("DayOfWeek")[TARGET_COLUMN].agg(
            ["count", "sum"]
        )

        return [
            {
                "day_of_week": int(day),
                "day": day_name(int(day)),
                "total_transactions": int(row["count"]),
                "fraud_cases": int(row["sum"]),
                "fraud_rate": fraud_rate(int(row["sum"]), int(row["count"])),
            }
            for day, row in grouped.sort_index().iterrows()
        ]

    def amount_distribution(self, bins: int = 20) -> dict:
        """
        Transaction amount histogram split by observed class.

        Parameters
        ----------
        bins : int, optional
            Number of equal-width bins across the observed amount range.

        Returns
        -------
        dict
            ``{"bin_centers", "legit_counts", "fraud_counts"}`` as float and
            int lists of equal length.
        """
        import numpy as np

        amounts = self.data["Transaction_Amount"].astype(float)

        edges = np.linspace(float(amounts.min()), float(amounts.max()), bins + 1)
        centers = ((edges[:-1] + edges[1:]) / 2).tolist()

        legit = amounts[self.data[TARGET_COLUMN] == 0]
        fraud = amounts[self.data[TARGET_COLUMN] == 1]

        legit_counts, _ = np.histogram(legit, bins=edges)
        fraud_counts, _ = np.histogram(fraud, bins=edges)

        return {
            "bin_centers": [float(v) for v in centers],
            "legit_counts": [int(v) for v in legit_counts],
            "fraud_counts": [int(v) for v in fraud_counts],
        }

    def summary(self) -> dict:
        """
        Assemble the full observed-fraud payload served by the API.

        Returns
        -------
        dict
            Overview plus per-hour, per-weekday and per-category breakdowns
            and the amount distribution.
        """
        return {
            "overview": self.overview(),
            "by_hour": self.by_hour(),
            "by_day_of_week": self.by_day_of_week(),
            "by_merchant_category": self.by_column("Merchant_Category"),
            "by_transaction_type": self.by_column("Transaction_Type"),
            "by_device_type": self.by_column("Device_Type"),
            "by_transaction_device": self.by_column("Transaction_Device"),
            "by_account_type": self.by_column("Account_Type"),
            "by_gender": self.by_column("Gender"),
            "amount_distribution": self.amount_distribution(),
        }

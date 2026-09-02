"""
Behavioural feature construction for unsupervised anomaly detection.

Two rules govern everything in this module:

1. **Nothing is fitted outside the training window.** Group means, standard
   deviations and category frequencies are learned on train and *applied* to
   validation and test. Fitting them on the full frame would let the test
   period influence the encoding of its own rows.
2. **Nothing needs the future.** Every feature is computable from a single
   transaction plus statistics that already existed before it happened, so
   the same code path serves training and live scoring.

Unlike the previous dataset, ``AccountID`` repeats (~5x per account), so
per-account features — the count of transactions on the account and the days
since the account's previous transaction — are viable and included.
"""

import numpy as np
import pandas as pd

from backend.src.modeling.config import (
    DEVIATION_GROUP_SUFFIX,
    DEVIATION_GROUPS,
    DURATION_DEVIATION_GROUP,
    FREQUENCY_COLUMN_SUFFIX,
    FREQUENCY_COLUMNS,
)


class FeatureEngineer:
    """
    Derive behavioural anomaly features with train-only fitted statistics.

    The class follows the scikit-learn ``fit`` / ``transform`` contract so the
    preprocessing applied at training time is the one applied at prediction
    time.

    Attributes
    ----------
    deviation_groups : list[str]
        Columns whose per-group amount mean and standard deviation express how
        unusual a transaction amount is for its context.
    frequency_columns : list[str]
        Columns replaced by the share of training transactions carrying each
        category value.
    group_stats_ : dict[str, pd.DataFrame]
        Fitted per-group amount statistics, keyed by grouping column.
    duration_group_stats_ : pd.DataFrame
        Fitted per-``Channel`` duration statistics.
    frequency_maps_ : dict[str, pd.Series]
        Fitted category frequencies, keyed by column.
    account_txn_count_median_ : float
        Median per-account transaction count in train; the fallback for an
        account never seen during fitting (i.e. a live single transaction).
    days_since_prev_median_ : float
        Median gap in days between consecutive transactions on the same
        account in train; the fallback for an account's first transaction.
    is_fitted : bool
        Whether :meth:`fit` has been called.
    """

    #: Amount column the deviation features are computed from.
    AMOUNT_COLUMN = "TransactionAmount"

    #: Balance column used by the ratio and post-transaction features.
    BALANCE_COLUMN = "AccountBalance"

    #: Duration column used by the log and deviation features.
    DURATION_COLUMN = "TransactionDuration"

    #: Account identifier used by the per-account features.
    ACCOUNT_COLUMN = "AccountID"

    #: Parsed timestamp used to order transactions within an account.
    TIMESTAMP_COLUMN = "Transaction_Timestamp"

    #: Hours below this are treated as night-time activity.
    NIGHT_HOUR_CUTOFF = 6

    #: Guard against a zero denominator in ratio features.
    EPSILON = 1.0

    def __init__(
        self,
        deviation_groups: list[str] | None = None,
        frequency_columns: list[str] | None = None,
        duration_group: str = DURATION_DEVIATION_GROUP,
    ) -> None:
        """
        Initialize the feature engineer.

        Parameters
        ----------
        deviation_groups : list[str], optional
            Grouping columns for the amount-deviation features. Defaults to
            ``DEVIATION_GROUPS`` from the configuration module.
        frequency_columns : list[str], optional
            Columns to frequency-encode. Defaults to ``FREQUENCY_COLUMNS``.
        duration_group : str, optional
            Column whose per-group duration mean/std drives
            ``Duration_Dev_Channel``. Defaults to ``Channel``.
        """
        self.deviation_groups = list(deviation_groups or DEVIATION_GROUPS)
        self.frequency_columns = list(frequency_columns or FREQUENCY_COLUMNS)
        self.duration_group = duration_group

        self.group_stats_: dict[str, pd.DataFrame] = {}
        self.duration_group_stats_: pd.DataFrame | None = None
        self.frequency_maps_: dict[str, pd.Series] = {}

        self.global_amount_mean_: float = 0.0
        self.global_amount_std_: float = 1.0
        self.global_duration_mean_: float = 0.0
        self.global_duration_std_: float = 1.0

        self.account_txn_count_median_: float = 1.0
        self.days_since_prev_median_: float = 0.0
        self.train_account_ids_: set = set()

        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, data: pd.DataFrame) -> "FeatureEngineer":
        """
        Learn the behavioural statistics from the training window only.

        Parameters
        ----------
        data : pd.DataFrame
            Training transactions. Must contain the amount, balance and
            duration columns and every configured grouping / frequency column.

        Returns
        -------
        FeatureEngineer
            The fitted instance, for chaining.

        Raises
        ------
        KeyError
            If a configured column is absent from ``data``.
        """
        self._require_columns(
            data,
            [self.AMOUNT_COLUMN, self.BALANCE_COLUMN, self.DURATION_COLUMN],
        )
        self._require_columns(data, self.deviation_groups)
        self._require_columns(data, self.frequency_columns)
        self._require_columns(data, [self.duration_group])

        amount = data[self.AMOUNT_COLUMN].astype(float)
        duration = data[self.DURATION_COLUMN].astype(float)

        self.global_amount_mean_ = float(amount.mean())
        amt_std = float(amount.std(ddof=0))
        self.global_amount_std_ = amt_std if amt_std > 0 else 1.0

        self.global_duration_mean_ = float(duration.mean())
        dur_std = float(duration.std(ddof=0))
        self.global_duration_std_ = dur_std if dur_std > 0 else 1.0

        self.group_stats_ = {
            column: self._fit_group_stats(
                data, column, self.AMOUNT_COLUMN, self.global_amount_std_
            )
            for column in self.deviation_groups
        }

        self.duration_group_stats_ = self._fit_group_stats(
            data, self.duration_group, self.DURATION_COLUMN,
            self.global_duration_std_,
        )

        total = len(data)
        self.frequency_maps_ = {
            column: data[column].value_counts().astype(float) / total
            for column in self.frequency_columns
        }

        account_counts = data.groupby(self.ACCOUNT_COLUMN).size()
        self.account_txn_count_median_ = float(account_counts.median())
        self.train_account_ids_ = set(data[self.ACCOUNT_COLUMN].unique())

        gaps = self._account_day_gaps(data)
        positive = gaps.dropna()
        self.days_since_prev_median_ = (
            float(positive.median()) if len(positive) else 0.0
        )

        self.is_fitted = True

        return self

    @staticmethod
    def _fit_group_stats(
        data: pd.DataFrame, column: str, value_column: str, fallback_std: float
    ) -> pd.DataFrame:
        """
        Fit per-group mean and standard deviation of ``value_column``.

        Parameters
        ----------
        data : pd.DataFrame
            Training frame.
        column : str
            Grouping column.
        value_column : str
            Column to aggregate.
        fallback_std : float
            Replacement for a missing or zero group standard deviation.

        Returns
        -------
        pd.DataFrame
            Index = group key, columns ``group_mean`` and ``group_std``.
        """
        stats = (
            data.groupby(column)[value_column]
            .agg(["mean", "std"])
            .rename(columns={"mean": "group_mean", "std": "group_std"})
        )
        stats["group_std"] = stats["group_std"].fillna(fallback_std)
        stats.loc[stats["group_std"] <= 0, "group_std"] = fallback_std

        return stats

    def _account_day_gaps(self, data: pd.DataFrame) -> pd.Series:
        """
        Days between each transaction and the account's previous one.

        Parameters
        ----------
        data : pd.DataFrame
            Frame containing the account and timestamp columns.

        Returns
        -------
        pd.Series
            Aligned to ``data.index``. NaN for the first transaction on an
            account within the frame.
        """
        ordered = data.sort_values(
            [self.ACCOUNT_COLUMN, self.TIMESTAMP_COLUMN]
        )
        delta = (
            ordered.groupby(self.ACCOUNT_COLUMN)[self.TIMESTAMP_COLUMN]
            .diff()
            .dt.total_seconds()
            / 86400.0
        )

        return delta.reindex(data.index)

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Attach the engineered features to a frame.

        Parameters
        ----------
        data : pd.DataFrame
            Transactions to enrich. Must carry the temporal columns produced by
            :class:`~backend.src.data.data_loader.TransactionDataLoader`.

        Returns
        -------
        pd.DataFrame
            Copy of ``data`` with the engineered columns appended.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        KeyError
            If a required column is absent from ``data``.
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureEngineer.transform called before fit")

        self._require_columns(
            data,
            [
                self.AMOUNT_COLUMN,
                self.BALANCE_COLUMN,
                self.DURATION_COLUMN,
                self.ACCOUNT_COLUMN,
                self.TIMESTAMP_COLUMN,
                "Hour",
            ],
        )

        out = data.copy()

        amount = out[self.AMOUNT_COLUMN].astype(float)
        balance = out[self.BALANCE_COLUMN].astype(float)
        duration = out[self.DURATION_COLUMN].astype(float)

        # --- amount / balance / duration relationships -------------------
        out["Amount_To_Balance_Ratio"] = amount / (balance.abs() + self.EPSILON)
        out["Balance_After_Transaction"] = balance - amount
        out["Log_Transaction_Amount"] = np.log1p(amount.clip(lower=0))
        out["Log_Account_Balance"] = np.log1p(balance.clip(lower=0))
        out["Log_Transaction_Duration"] = np.log1p(duration.clip(lower=0))

        # --- hour-of-day behaviour -------------------------------------
        out["Is_Night"] = (out["Hour"] < self.NIGHT_HOUR_CUTOFF).astype(int)

        # --- deviation from the context's typical amount ---------------
        for column in self.deviation_groups:
            suffix = DEVIATION_GROUP_SUFFIX.get(column, column)
            out[f"Amount_Dev_{suffix}"] = self._apply_group_dev(
                out, column, amount, self.group_stats_[column],
                self.global_amount_mean_, self.global_amount_std_,
            )

        # --- deviation from the channel's typical duration -------------
        out["Duration_Dev_Channel"] = self._apply_group_dev(
            out, self.duration_group, duration, self.duration_group_stats_,
            self.global_duration_mean_, self.global_duration_std_,
        )

        # --- how common is this state / channel / merchant / device ----
        for column in self.frequency_columns:
            suffix = FREQUENCY_COLUMN_SUFFIX.get(column, column)
            mapping = self.frequency_maps_[column]
            out[f"{suffix}_Frequency"] = (
                out[column].map(mapping).astype(float).fillna(0.0)
            )

        # --- per-account features -------------------------------------
        within_frame_count = out.groupby(self.ACCOUNT_COLUMN)[
            self.ACCOUNT_COLUMN
        ].transform("size").astype(float)
        unseen = ~out[self.ACCOUNT_COLUMN].isin(self.train_account_ids_)
        within_frame_count[unseen] = self.account_txn_count_median_
        out["Account_Txn_Count"] = within_frame_count

        gaps = self._account_day_gaps(out)
        out["Days_Since_Prev_Account_Txn"] = gaps.fillna(
            self.days_since_prev_median_
        ).astype(float)

        return out

    @staticmethod
    def _apply_group_dev(
        frame: pd.DataFrame,
        column: str,
        values: pd.Series,
        stats: pd.DataFrame,
        global_mean: float,
        global_std: float,
    ) -> pd.Series:
        """
        Compute a per-group z-score, falling back to global stats.

        Parameters
        ----------
        frame : pd.DataFrame
            Frame being transformed.
        column : str
            Grouping column.
        values : pd.Series
            Values to standardise (amount or duration).
        stats : pd.DataFrame
            Fitted ``group_mean`` / ``group_std`` table.
        global_mean, global_std : float
            Fallbacks for a group key unseen during fitting.

        Returns
        -------
        pd.Series
            The z-score, aligned to ``frame.index``.
        """
        mean = (
            frame[column].map(stats["group_mean"]).astype(float)
            .fillna(global_mean)
        )
        std = (
            frame[column].map(stats["group_std"]).astype(float)
            .fillna(global_std)
        )
        std = std.replace(0.0, global_std)

        return (values.to_numpy() - mean) / std

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit on ``data`` and return the transformed frame.

        Use this on the training window only.

        Parameters
        ----------
        data : pd.DataFrame
            Training transactions.

        Returns
        -------
        pd.DataFrame
            Transformed training frame.
        """
        return self.fit(data).transform(data)

    @staticmethod
    def _require_columns(data: pd.DataFrame, columns: list[str]) -> None:
        """
        Raise if any required column is missing.

        Parameters
        ----------
        data : pd.DataFrame
            Frame to check.
        columns : list[str]
            Columns that must be present.

        Raises
        ------
        KeyError
            Listing every missing column.
        """
        missing = [c for c in columns if c not in data.columns]

        if missing:
            raise KeyError(f"Missing required columns: {missing}")

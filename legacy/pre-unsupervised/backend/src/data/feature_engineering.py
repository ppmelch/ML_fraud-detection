"""
Behavioural feature construction for fraud detection.

Two rules govern everything in this module:

1. **Nothing is fitted outside the training window.** Group means, standard
   deviations and category frequencies are learned on train and *applied* to
   validation and test. Fitting them on the full frame would let the test
   period influence the encoding of its own rows.
2. **Nothing needs the future.** Every feature is computable from a single
   transaction plus statistics that already existed before it happened, so
   the same code path serves training and live scoring.

A note on what is *not* here: per-customer history (transaction frequency,
customer average amount, deviation from the customer's own profile) is the
natural fraud feature family and it cannot be built from this dataset —
``Customer_ID`` is unique on all 200,000 rows, so no customer is ever seen
twice. The entity-level aggregates below are the available substitute.
"""

import numpy as np
import pandas as pd

from backend.src.modeling.config import (
    DEVIATION_GROUPS,
    FREQUENCY_COLUMNS,
)


class FeatureEngineer:
    """
    Derive behavioural fraud features with train-only fitted statistics.

    The class follows the scikit-learn ``fit`` / ``transform`` contract so the
    preprocessing applied at training time is byte-for-byte the one applied at
    prediction time.

    Attributes
    ----------
    deviation_groups : list[str]
        Columns whose per-group amount mean and standard deviation are used to
        express how unusual a transaction amount is for its context.
    frequency_columns : list[str]
        Columns replaced by the share of training transactions carrying each
        category value.
    group_stats_ : dict[str, pd.DataFrame]
        Fitted per-group amount statistics, keyed by grouping column.
    frequency_maps_ : dict[str, pd.Series]
        Fitted category frequencies, keyed by column.
    global_amount_mean_ : float
        Training mean transaction amount, used as the fallback for a category
        never seen during fitting.
    global_amount_std_ : float
        Training standard deviation of the transaction amount.
    is_fitted : bool
        Whether :meth:`fit` has been called.
    """

    #: Amount column the deviation features are computed from.
    AMOUNT_COLUMN = "Transaction_Amount"

    #: Balance column used by the ratio and post-transaction features.
    BALANCE_COLUMN = "Account_Balance"

    #: Hours below this are treated as night-time activity.
    NIGHT_HOUR_CUTOFF = 6

    #: Guard against a zero denominator in ratio features.
    EPSILON = 1.0

    def __init__(
        self,
        deviation_groups: list[str] | None = None,
        frequency_columns: list[str] | None = None,
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
        """
        self.deviation_groups = list(deviation_groups or DEVIATION_GROUPS)
        self.frequency_columns = list(frequency_columns or FREQUENCY_COLUMNS)

        self.group_stats_: dict[str, pd.DataFrame] = {}
        self.frequency_maps_: dict[str, pd.Series] = {}
        self.global_amount_mean_: float = 0.0
        self.global_amount_std_: float = 1.0
        self.is_fitted: bool = False

    def fit(self, data: pd.DataFrame) -> "FeatureEngineer":
        """
        Learn the behavioural statistics from the training window only.

        Parameters
        ----------
        data : pd.DataFrame
            Training transactions. Must contain the amount column and every
            configured grouping and frequency column.

        Returns
        -------
        FeatureEngineer
            The fitted instance, for chaining.

        Raises
        ------
        KeyError
            If a configured column is absent from ``data``.
        """
        self._require_columns(data, [self.AMOUNT_COLUMN, self.BALANCE_COLUMN])
        self._require_columns(data, self.deviation_groups)
        self._require_columns(data, self.frequency_columns)

        amount = data[self.AMOUNT_COLUMN].astype(float)

        self.global_amount_mean_ = float(amount.mean())

        # A degenerate std would turn every deviation into inf; fall back to 1.
        std = float(amount.std(ddof=0))
        self.global_amount_std_ = std if std > 0 else 1.0

        self.group_stats_ = {}

        for column in self.deviation_groups:
            stats = (
                data.groupby(column)[self.AMOUNT_COLUMN]
                .agg(["mean", "std"])
                .rename(columns={"mean": "group_mean", "std": "group_std"})
            )
            stats["group_std"] = stats["group_std"].fillna(
                self.global_amount_std_
            )
            stats.loc[stats["group_std"] <= 0, "group_std"] = (
                self.global_amount_std_
            )
            self.group_stats_[column] = stats

        total = len(data)

        self.frequency_maps_ = {
            column: data[column].value_counts().astype(float) / total
            for column in self.frequency_columns
        }

        self.is_fitted = True

        return self

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
            data, [self.AMOUNT_COLUMN, self.BALANCE_COLUMN, "Hour"]
        )

        out = data.copy()

        amount = out[self.AMOUNT_COLUMN].astype(float)
        balance = out[self.BALANCE_COLUMN].astype(float)

        # --- amount / balance relationship -------------------------------
        out["Amount_To_Balance_Ratio"] = amount / (balance.abs() + self.EPSILON)
        out["Balance_After_Transaction"] = balance - amount
        out["Log_Transaction_Amount"] = np.log1p(amount.clip(lower=0))
        out["Log_Account_Balance"] = np.log1p(balance.clip(lower=0))

        # --- hour-of-day behaviour ---------------------------------------
        out["Is_Night"] = (out["Hour"] < self.NIGHT_HOUR_CUTOFF).astype(int)

        # --- deviation from the context's typical amount ------------------
        for column in self.deviation_groups:
            stats = self.group_stats_[column]

            mean = (
                out[column].map(stats["group_mean"]).astype(float)
                .fillna(self.global_amount_mean_)
            )
            std = (
                out[column].map(stats["group_std"]).astype(float)
                .fillna(self.global_amount_std_)
            )
            std = std.replace(0.0, self.global_amount_std_)

            out[f"Amount_Dev_{column}"] = (amount - mean) / std

        # --- how common is this device / merchant / place -----------------
        for column in self.frequency_columns:
            mapping = self.frequency_maps_[column]

            # A category unseen in training is rare by construction: 0.0.
            out[f"{column}_Frequency"] = (
                out[column].map(mapping).astype(float).fillna(0.0)
            )

        return out

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

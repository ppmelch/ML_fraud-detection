"""
Loading and canonical timestamp derivation for the transaction dataset.

Every consumer of the raw CSV goes through this module so that the parsed
timestamp, the derived calendar columns, and the target dtype are identical
in the modelling path and in the analytics path. Deriving `Hour` twice, in
two places, is how two dashboards end up disagreeing about the same number.
"""

from pathlib import Path

import pandas as pd

from backend.src.modeling.config import (
    DATASET_PATH,
    DATE_COLUMN,
    DATE_FORMAT,
    TARGET_COLUMN,
    TIME_COLUMN,
    TIME_FORMAT,
)


class TransactionDataLoader:
    """
    Load the banking transaction dataset and derive its temporal columns.

    The loader is the single source of truth for `Hour`, `DayOfWeek`, `Month`
    and `Is_Weekend`. These are calendar facts about the transaction itself:
    they are known the instant the transaction happens, so using them as model
    features introduces no look-ahead.

    Attributes
    ----------
    path : Path
        Location of the CSV file to read.
    """

    #: Calendar columns derived by :meth:`add_temporal_features`.
    TEMPORAL_COLUMNS = ["Hour", "DayOfWeek", "Month", "Is_Weekend"]

    def __init__(self, path: Path = DATASET_PATH) -> None:
        """
        Initialize the loader.

        Parameters
        ----------
        path : Path, optional
            Path to the transaction CSV. Defaults to ``DATASET_PATH`` from the
            configuration module.
        """
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        """
        Read the dataset and attach the derived temporal columns.

        Returns
        -------
        pd.DataFrame
            Raw transactions plus ``Transaction_Timestamp`` and the columns
            listed in :attr:`TEMPORAL_COLUMNS`.

        Raises
        ------
        FileNotFoundError
            If the configured dataset file does not exist.
        ValueError
            If the target column is missing or is not binary 0/1.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")

        data = pd.read_csv(self.path)
        data = data.drop(columns=["Unnamed: 0"], errors="ignore")

        if TARGET_COLUMN not in data.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not in dataset")

        observed = set(data[TARGET_COLUMN].dropna().unique())

        if not observed.issubset({0, 1}):
            raise ValueError(
                f"Target '{TARGET_COLUMN}' must be binary 0/1, found: {sorted(observed)}"
            )

        data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)

        return self.add_temporal_features(data)

    def add_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Derive the calendar columns used by the model and by the analytics.

        Parameters
        ----------
        data : pd.DataFrame
            Frame containing the raw date and time columns.

        Returns
        -------
        pd.DataFrame
            Copy of ``data`` with ``Transaction_Timestamp``, ``Hour``,
            ``DayOfWeek``, ``Month`` and ``Is_Weekend`` attached.

        Raises
        ------
        ValueError
            If the date or time column cannot be parsed for any row.
        """
        data = data.copy()

        date = pd.to_datetime(
            data[DATE_COLUMN], format=DATE_FORMAT, errors="coerce"
        )
        time = pd.to_datetime(
            data[TIME_COLUMN], format=TIME_FORMAT, errors="coerce"
        )

        if date.isna().any():
            bad = int(date.isna().sum())
            raise ValueError(
                f"{bad} rows have a '{DATE_COLUMN}' value that is not {DATE_FORMAT}"
            )

        if time.isna().any():
            bad = int(time.isna().sum())
            raise ValueError(
                f"{bad} rows have a '{TIME_COLUMN}' value that is not {TIME_FORMAT}"
            )

        data["Transaction_Timestamp"] = date + pd.to_timedelta(
            time.dt.hour, unit="h"
        ) + pd.to_timedelta(time.dt.minute, unit="m") + pd.to_timedelta(
            time.dt.second, unit="s"
        )

        data["Hour"] = time.dt.hour.astype(int)
        data["DayOfWeek"] = date.dt.dayofweek.astype(int)
        data["Month"] = date.dt.month.astype(int)
        data["Is_Weekend"] = (data["DayOfWeek"] >= 5).astype(int)

        return data

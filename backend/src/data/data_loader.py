"""
Loading and canonical timestamp derivation for the transaction dataset.

Every consumer of the raw CSV goes through this module so that the parsed
timestamp and the derived calendar columns are identical in the modelling
path and in the analytics path. Deriving ``Hour`` twice, in two places, is
how two dashboards end up disagreeing about the same number.

The dataset is unlabelled: there is no fraud target to validate here.
"""

from pathlib import Path

import pandas as pd

from backend.src.modeling.config import (
    ARTEFACT_COLUMNS,
    CITY_TO_STATE,
    DATASET_PATH,
    TIMESTAMP_COLUMN,
    TIMESTAMP_FORMAT,
)


class TransactionDataLoader:
    """
    Load the banking transaction dataset and derive its temporal columns.

    The loader is the single source of truth for ``Hour``, ``DayOfWeek``,
    ``Month`` and ``Is_Weekend``. These are calendar facts about the
    transaction itself: they are known the instant the transaction happens,
    so using them as model features introduces no look-ahead.

    It also resolves ``Location`` (a US city name) to ``USState`` via
    :data:`~backend.src.modeling.config.CITY_TO_STATE`.

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
        Read the dataset and attach the derived columns.

        Returns
        -------
        pd.DataFrame
            Raw transactions plus ``Transaction_Timestamp``, the columns
            listed in :attr:`TEMPORAL_COLUMNS`, and ``USState``. The
            data-extraction artefact columns are dropped.

        Raises
        ------
        FileNotFoundError
            If the configured dataset file does not exist.
        ValueError
            If a ``Location`` value has no mapping to a US state.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")

        data = pd.read_csv(self.path)
        data = data.drop(columns=["Unnamed: 0"], errors="ignore")
        data = data.drop(columns=ARTEFACT_COLUMNS, errors="ignore")

        return self.add_temporal_features(data)

    def add_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Derive the calendar columns and resolve the US state.

        This is the public hook shared by training and the live predict path,
        so a single scored transaction is enriched exactly as the training
        rows were.

        Parameters
        ----------
        data : pd.DataFrame
            Frame containing the raw ``TransactionDate`` and ``Location``
            columns.

        Returns
        -------
        pd.DataFrame
            Copy of ``data`` with ``Transaction_Timestamp``, ``Hour``,
            ``DayOfWeek``, ``Month``, ``Is_Weekend`` and ``USState`` attached.

        Raises
        ------
        ValueError
            If the timestamp column cannot be parsed for any row, or if a
            ``Location`` value is not in ``CITY_TO_STATE``.
        """
        data = data.copy()

        timestamp = pd.to_datetime(
            data[TIMESTAMP_COLUMN], format=TIMESTAMP_FORMAT, errors="coerce"
        )

        if timestamp.isna().any():
            bad = int(timestamp.isna().sum())
            raise ValueError(
                f"{bad} row(s) have a '{TIMESTAMP_COLUMN}' value that is not "
                f"{TIMESTAMP_FORMAT}"
            )

        data["Transaction_Timestamp"] = timestamp
        data["Hour"] = timestamp.dt.hour.astype(int)
        data["DayOfWeek"] = timestamp.dt.dayofweek.astype(int)
        data["Month"] = timestamp.dt.month.astype(int)
        data["Is_Weekend"] = (data["DayOfWeek"] >= 5).astype(int)

        if "Location" in data.columns:
            unknown = sorted(
                set(data["Location"].dropna().unique()) - set(CITY_TO_STATE)
            )

            if unknown:
                raise ValueError(
                    f"{len(unknown)} Location value(s) have no US-state "
                    f"mapping in CITY_TO_STATE: {unknown}"
                )

            data["USState"] = data["Location"].map(CITY_TO_STATE)

        return data

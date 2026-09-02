import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

from backend.src.modeling.config import (
    RANDOM_STATE,
    TEST_DAYS,
    TEST_SIZE,
    VALIDATION_DAYS,
)


class DataSplitter:
    """
    Class responsible for splitting data into training and testing sets.

    This class encapsulates the train-test split logic to ensure
    reproducibility and consistency across the pipeline.
    """

    def __init__(
        self,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
        test_days: int = TEST_DAYS,
        validation_days: int = VALIDATION_DAYS,
    ) -> None:
        """
        Initialize the data splitter.

        Parameters
        ----------
        test_size : float, optional
            Proportion of the dataset to include in the random test split.
        random_state : int, optional
            Seed used for random number generation to ensure reproducibility.
        test_days : int, optional
            Length in days of the held-out future window used by
            :meth:`temporal_split`.
        validation_days : int, optional
            Length in days of the validation window carved out of the tail of
            the training period by :meth:`temporal_split`.
        """
        self.test_size = test_size
        self.random_state = random_state
        self.test_days = test_days
        self.validation_days = validation_days

    def split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the dataset into training and testing sets.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target variable.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            X_train, X_test, y_train, y_test
        """
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, stratify=y)

    def temporal_split(
        self,
        data: pd.DataFrame,
        timestamp_column: str = "Transaction_Timestamp",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split transactions chronologically into train, validation and test.

        A random split of financial transactions lets the model see a Tuesday
        that sits between two training Mondays, which is not the situation it
        faces in production. This method reproduces the real one: fit on the
        past, choose the operating threshold on the recent past, and measure
        once on the future.

        The windows are contiguous and non-overlapping::

            |------------ train ------------|-- validation --|--- test ---|
            oldest                                                   newest

        Parameters
        ----------
        data : pd.DataFrame
            Transactions containing a parsed timestamp column.
        timestamp_column : str, optional
            Name of the datetime column used for ordering.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            train, validation and test frames, each sorted by timestamp.

        Raises
        ------
        KeyError
            If ``timestamp_column`` is not present in ``data``.
        ValueError
            If the observed date span is too short to carve out the
            configured validation and test windows, or if any resulting
            window is empty.
        """
        if timestamp_column not in data.columns:
            raise KeyError(f"Missing timestamp column '{timestamp_column}'")

        ordered = data.sort_values(timestamp_column).reset_index(drop=True)

        last_day = ordered[timestamp_column].max().normalize()
        first_day = ordered[timestamp_column].min().normalize()

        span_days = (last_day - first_day).days + 1
        required = self.test_days + self.validation_days + 1

        if span_days < required:
            raise ValueError(
                f"Dataset spans {span_days} day(s); a temporal split needs at "
                f"least {required} (test={self.test_days}, "
                f"validation={self.validation_days}, plus one training day)"
            )

        # Boundaries are exclusive upper bounds expressed as day starts, so a
        # transaction at 23:59 on the last training day stays in train.
        test_start = last_day - pd.Timedelta(days=self.test_days - 1)
        validation_start = test_start - pd.Timedelta(days=self.validation_days)

        train = ordered[ordered[timestamp_column] < validation_start]
        validation = ordered[
            (ordered[timestamp_column] >= validation_start)
            & (ordered[timestamp_column] < test_start)
        ]
        test = ordered[ordered[timestamp_column] >= test_start]

        for name, frame in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        ):
            if frame.empty:
                raise ValueError(f"Temporal split produced an empty {name} window")

        return (
            train.reset_index(drop=True),
            validation.reset_index(drop=True),
            test.reset_index(drop=True),
        )

    def describe_temporal_split(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
        timestamp_column: str = "Transaction_Timestamp",
        target_column: str = "Is_Fraud",
    ) -> dict:
        """
        Summarise a temporal split for reporting and provenance.

        Parameters
        ----------
        train, validation, test : pd.DataFrame
            The three windows returned by :meth:`temporal_split`.
        timestamp_column : str, optional
            Datetime column used for ordering.
        target_column : str, optional
            Binary target column used to report the fraud rate per window.

        Returns
        -------
        dict
            Per-window row count, date bounds, fraud count and fraud rate,
            plus the strategy name. All values are JSON-serialisable.
        """
        summary = {"strategy": "temporal", "timestamp_column": timestamp_column}

        for name, frame in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        ):
            summary[name] = {
                "rows": int(len(frame)),
                "start": frame[timestamp_column].min().isoformat(),
                "end": frame[timestamp_column].max().isoformat(),
                "fraud_cases": int(frame[target_column].sum()),
                "fraud_rate": float(frame[target_column].mean()),
            }

        return summary


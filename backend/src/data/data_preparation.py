"""
Feature matrix construction for the anomaly-detection model.

`DataPreparation` turns enriched transactions into the numeric matrix the
estimator consumes. It is a fitted object: the encoder learns its column
layout on the training window and reproduces exactly that layout on every
later frame, so a category missing from a scoring batch cannot silently
shift the meaning of column 17.

There is no target. ``fit_transform`` and ``transform`` return ``X`` only.
"""

import pandas as pd

from backend.src.modeling.config import (
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    IDENTIFIER_COLUMNS,
    RAW_FEATURES,
)


class DataPreparation:
    """
    Build the model feature matrix from enriched transaction data.

    The class handles:

    - Selecting the configured raw feature set plus every engineered column.
    - Dropping identifier columns.
    - One-hot encoding the low-cardinality categorical features only.
    - Freezing the encoded column layout so training and prediction agree.

    High-cardinality columns (``MerchantID``, ``DeviceID``, ``Location``)
    are deliberately excluded from the encoder: dummying them would add
    hundreds of near-empty columns for no signal. Where their information
    matters it enters through the frequency and deviation features built in
    :class:`~backend.src.data.feature_engineering.FeatureEngineer`.

    Attributes
    ----------
    data : pd.DataFrame or None
        Optional input dataset copied from the original source.
    raw_features : list[str]
        Raw predictor columns selected for the model.
    engineered_features : list[str]
        Behavioural features appended when present in the input frame.
    categorical_features : list[str]
        Subset of the raw features that is one-hot encoded.
    feature_columns_ : list[str]
        Encoded column layout learned during fitting.
    is_fitted : bool
        Whether the encoder layout has been learned.
    """

    #: Raw predictor columns selected for the model.
    RAW_FEATURES = RAW_FEATURES

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        raw_features: list[str] | None = None,
        engineered_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
    ) -> None:
        """
        Initialize the DataPreparation object.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Enriched dataset containing transaction information.
        raw_features : list[str], optional
            Raw feature set. Defaults to ``RAW_FEATURES`` from config.
        engineered_features : list[str], optional
            Behavioural features. Defaults to ``ENGINEERED_FEATURES``.
        categorical_features : list[str], optional
            Features to one-hot encode. Defaults to ``CATEGORICAL_FEATURES``.
        """
        self.data = data.copy() if data is not None else None

        self.raw_features = list(raw_features or RAW_FEATURES)
        self.engineered_features = list(
            engineered_features
            if engineered_features is not None
            else ENGINEERED_FEATURES
        )
        self.categorical_features = list(
            categorical_features or CATEGORICAL_FEATURES
        )

        self.feature_columns_: list[str] = []
        self.is_fitted: bool = False

    def prepare_data(self) -> pd.DataFrame:
        """
        Fit the encoder on ``self.data`` and return the training matrix.

        Call this on the training window only; use :meth:`transform` for every
        later frame.

        Returns
        -------
        pd.DataFrame
            Processed feature matrix ready for modeling.

        Raises
        ------
        ValueError
            If no data was supplied to the constructor.
        """
        if self.data is None:
            raise ValueError("DataPreparation was constructed without data")

        return self.fit_transform(self.data)

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Learn the encoded column layout and return the training matrix.

        Parameters
        ----------
        data : pd.DataFrame
            Training transactions.

        Returns
        -------
        pd.DataFrame
            Feature matrix.
        """
        X = self._encode(data)

        self.feature_columns_ = list(X.columns)
        self.is_fitted = True

        return X

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted encoding to a new frame.

        Columns absent from ``data`` are added as zeros and columns unseen
        during fitting are dropped, so the returned matrix always matches the
        layout the model was trained on.

        Parameters
        ----------
        data : pd.DataFrame
            Transactions to encode.

        Returns
        -------
        pd.DataFrame
            Feature matrix with the fitted column layout.

        Raises
        ------
        RuntimeError
            If called before the encoder has been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("DataPreparation.transform called before fit")

        X = self._encode(data)

        return X.reindex(columns=self.feature_columns_, fill_value=0.0).astype(
            float
        )

    def _encode(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Select the configured features and one-hot encode the categoricals.

        Parameters
        ----------
        data : pd.DataFrame
            Frame to encode.

        Returns
        -------
        pd.DataFrame
            Numeric feature matrix.

        Raises
        ------
        ValueError
            If a raw selected feature is missing from ``data``, or if an
            identifier column reached the feature set.
        """
        missing = [c for c in self.raw_features if c not in data.columns]

        if missing:
            raise ValueError(
                f"Missing raw features before preprocessing: {missing}"
            )

        columns = list(self.raw_features) + [
            c for c in self.engineered_features if c in data.columns
        ]

        leaked = sorted(set(columns) & set(IDENTIFIER_COLUMNS))

        if leaked:
            raise ValueError(
                f"Identifier columns must never be model features: {leaked}"
            )

        X = data[columns].copy()

        categorical = [c for c in self.categorical_features if c in X.columns]

        X = pd.get_dummies(
            X, columns=categorical, drop_first=True, dtype=float
        )

        return X.astype(float)

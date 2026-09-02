"""Tests for DataPreparation: X-only output, layout freezing, identifier guard."""

import pytest

from backend.src.data.data_loader import TransactionDataLoader
from backend.src.data.feature_engineering import FeatureEngineer
from backend.src.data.data_preparation import DataPreparation


@pytest.fixture
def enriched(sample_transactions):
    loaded = TransactionDataLoader().add_temporal_features(sample_transactions)
    return FeatureEngineer().fit_transform(loaded)


class TestFitTransform:
    def test_returns_numeric_matrix_only(self, enriched):
        result = DataPreparation().fit_transform(enriched)

        # X only — not a tuple.
        assert not isinstance(result, tuple)
        assert result.shape[0] == len(enriched)
        assert all(str(dtype) == "float64" for dtype in result.dtypes)

    def test_no_identifier_columns_leak_in(self, enriched):
        X = DataPreparation().fit_transform(enriched)

        # The raw identifier columns themselves must not appear. Frequency
        # encodings such as ``MerchantID_Frequency`` are engineered features,
        # not the identifier, and are allowed.
        for banned in (
            "TransactionID",
            "AccountID",
            "DeviceID",
            "MerchantID",
            "IP Address",
        ):
            assert banned not in X.columns

    def test_missing_raw_feature_raises(self, enriched):
        broken = enriched.drop(columns=["AccountBalance"])

        with pytest.raises(ValueError):
            DataPreparation().fit_transform(broken)


class TestTransform:
    def test_column_layout_matches_between_train_and_test(self, enriched):
        half = len(enriched) // 2
        prep = DataPreparation()
        X_train = prep.fit_transform(enriched.iloc[:half])
        X_test = prep.transform(enriched.iloc[half:])

        assert list(X_train.columns) == list(X_test.columns)

    def test_transform_before_fit_raises(self, enriched):
        with pytest.raises(RuntimeError):
            DataPreparation().transform(enriched)

    def test_unseen_category_does_not_add_a_column(self, enriched):
        prep = DataPreparation()
        X_train = prep.fit_transform(enriched)

        unseen = enriched.copy()
        unseen.loc[unseen.index[0], "Channel"] = "Carrier Pigeon"
        X_new = prep.transform(unseen)

        assert list(X_new.columns) == list(X_train.columns)
        assert not X_new.isna().any().any()

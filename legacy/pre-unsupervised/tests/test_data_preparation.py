"""Tests for DataPreparation: encoding, layout freezing, identifier guard."""

import pytest

from backend.src.data.data_loader import TransactionDataLoader
from backend.src.data.data_preparation import DataPreparation


@pytest.fixture
def enriched(sample_transactions):
    return TransactionDataLoader().add_temporal_features(sample_transactions)


class TestFitTransform:
    def test_returns_numeric_matrix_and_target(self, enriched):
        X, y = DataPreparation().fit_transform(enriched)

        assert X.shape[0] == len(enriched)
        assert set(y.unique()).issubset({0, 1})
        assert all(str(dtype) in ("float64",) for dtype in X.dtypes)

    def test_no_identifier_columns_leak_in(self, enriched):
        X, _ = DataPreparation().fit_transform(enriched)

        for banned in ("Customer_ID", "Customer_Name", "Transaction_ID", "Customer_Email"):
            assert not any(col.startswith(banned) for col in X.columns)

    def test_missing_selected_feature_raises(self, enriched):
        broken = enriched.drop(columns=["Account_Balance"])

        with pytest.raises(ValueError):
            DataPreparation().fit_transform(broken)


class TestTransform:
    def test_column_layout_matches_between_train_and_test(self, enriched):
        half = len(enriched) // 2
        prep = DataPreparation()
        X_train, _ = prep.fit_transform(enriched.iloc[:half])
        X_test = prep.transform(enriched.iloc[half:])

        assert list(X_train.columns) == list(X_test.columns)

    def test_transform_before_fit_raises(self, enriched):
        with pytest.raises(RuntimeError):
            DataPreparation().transform(enriched)

    def test_category_unseen_in_training_does_not_add_a_column(self, enriched):
        prep = DataPreparation()
        X_train, _ = prep.fit_transform(enriched)

        unseen = enriched.copy()
        unseen.loc[0, "State"] = "NeverSeenState"
        X_new = prep.transform(unseen)

        assert list(X_new.columns) == list(X_train.columns)

    def test_category_missing_from_new_frame_is_filled_zero(self, enriched):
        prep = DataPreparation()
        prep.fit_transform(enriched)

        # A frame containing only one Account_Type value will not produce
        # dummy columns for the others; transform must still return every
        # fitted column, filled with 0.
        single_type = enriched.copy()
        single_type["Account_Type"] = enriched["Account_Type"].iloc[0]

        X = prep.transform(single_type)

        assert set(X.columns) == set(prep.feature_columns_)
        assert not X.isna().any().any()

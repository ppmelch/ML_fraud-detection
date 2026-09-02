"""Tests for FeatureEngineer: new feature set, train-only stats, unseen keys."""

import numpy as np
import pytest

from backend.src.data.data_loader import TransactionDataLoader
from backend.src.data.feature_engineering import FeatureEngineer


@pytest.fixture
def enriched(sample_transactions):
    return TransactionDataLoader().add_temporal_features(sample_transactions)


class TestFit:
    def test_transform_before_fit_raises(self, enriched):
        with pytest.raises(RuntimeError):
            FeatureEngineer().transform(enriched)

    def test_fit_missing_column_raises(self, enriched):
        with pytest.raises(KeyError):
            FeatureEngineer().fit(enriched.drop(columns=["USState"]))


class TestTransform:
    def test_adds_expected_columns(self, enriched):
        out = FeatureEngineer().fit_transform(enriched)

        expected = {
            "Is_Night",
            "Amount_To_Balance_Ratio",
            "Log_Transaction_Amount",
            "Balance_After_Transaction",
            "Amount_Dev_Account",
            "Amount_Dev_USState",
            "USState_Frequency",
            "Account_Txn_Count",
            "Days_Since_Prev_Account_Txn",
        }
        assert expected.issubset(out.columns)

    def test_no_nan_produced(self, enriched):
        out = FeatureEngineer().fit_transform(enriched)
        new_cols = [c for c in out.columns if c not in enriched.columns]

        assert not out[new_cols].isna().any().any()

    def test_frequency_of_unseen_category_is_zero(self, enriched):
        fe = FeatureEngineer()
        fe.fit(enriched)

        unseen = enriched.copy()
        unseen.loc[unseen.index[0], "USState"] = "NeverSeen"
        out = fe.transform(unseen)

        assert out.loc[out.index[0], "USState_Frequency"] == 0.0

    def test_unseen_account_uses_fitted_median_count(self, enriched):
        fe = FeatureEngineer()
        fe.fit(enriched)

        unseen = enriched.head(1).copy()
        unseen.loc[unseen.index[0], "AccountID"] = "AC99999"
        out = fe.transform(unseen)

        assert out.loc[out.index[0], "Account_Txn_Count"] == fe.account_txn_count_median_

    def test_deviation_stats_fitted_on_train_only(self, enriched):
        half = len(enriched) // 2
        train_part, test_part = enriched.iloc[:half], enriched.iloc[half:]

        fe = FeatureEngineer()
        fe.fit(train_part)
        out = fe.transform(test_part)

        train_mean = train_part.groupby("USState")["TransactionAmount"].mean()

        for state in test_part["USState"].unique():
            if state not in train_mean.index:
                continue
            rows = test_part["USState"] == state
            std = fe.group_stats_["USState"].loc[state, "group_std"]
            recovered = (
                test_part.loc[rows, "TransactionAmount"].to_numpy()
                - out.loc[rows, "Amount_Dev_USState"].to_numpy() * std
            )
            assert np.allclose(recovered, train_mean.loc[state], atol=1e-6)

    def test_night_flag_correctness(self, enriched):
        out = FeatureEngineer().fit_transform(enriched)

        assert (out.loc[out["Hour"] < 6, "Is_Night"] == 1).all()
        assert (out.loc[out["Hour"] >= 6, "Is_Night"] == 0).all()

"""Tests for FeatureEngineer: train-only fitting, no leakage into transform."""

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
        broken = enriched.drop(columns=["State"])

        with pytest.raises(KeyError):
            FeatureEngineer().fit(broken)


class TestTransform:
    def test_adds_expected_columns(self, enriched):
        fe = FeatureEngineer()
        out = fe.fit_transform(enriched)

        expected = {
            "Is_Night",
            "Amount_To_Balance_Ratio",
            "Log_Transaction_Amount",
            "Log_Account_Balance",
            "Balance_After_Transaction",
            "Amount_Dev_State",
            "State_Frequency",
        }
        assert expected.issubset(out.columns)

    def test_no_nan_produced(self, enriched):
        fe = FeatureEngineer()
        out = fe.fit_transform(enriched)

        new_cols = [c for c in out.columns if c not in enriched.columns]
        assert not out[new_cols].isna().any().any()

    def test_unseen_category_in_transform_does_not_crash(self, enriched):
        fe = FeatureEngineer()
        fe.fit(enriched)

        unseen = enriched.copy()
        unseen.loc[0, "State"] = "NeverSeenState"

        out = fe.transform(unseen)

        assert not out.loc[[0], [c for c in out.columns if c.startswith("Amount_Dev")]].isna().any().any()

    def test_frequency_of_unseen_category_is_zero(self, enriched):
        fe = FeatureEngineer()
        fe.fit(enriched)

        unseen = enriched.copy()
        unseen.loc[0, "State"] = "NeverSeenState"

        out = fe.transform(unseen)

        assert out.loc[0, "State_Frequency"] == 0.0

    def test_statistics_fitted_on_train_only(self, enriched):
        # Fit on the first half, transform the second half: the deviation
        # feature must use the first half's mean, not the second half's.
        half = len(enriched) // 2
        train_part = enriched.iloc[:half]
        test_part = enriched.iloc[half:]

        fe = FeatureEngineer()
        fe.fit(train_part)

        train_mean = train_part.groupby("State")["Transaction_Amount"].mean()

        out = fe.transform(test_part)

        for state in test_part["State"].unique():
            if state not in train_mean.index:
                continue
            rows = test_part["State"] == state
            expected_mean = train_mean.loc[state]
            # Reconstruct the amount from the z-score and compare to the
            # train-fitted mean (not a mean recomputed on test_part).
            std = fe.group_stats_["State"].loc[state, "group_std"]
            recovered_mean = (
                test_part.loc[rows, "Transaction_Amount"].to_numpy()
                - out.loc[rows, "Amount_Dev_State"].to_numpy() * std
            )
            assert np.allclose(recovered_mean, expected_mean, atol=1e-6)

    def test_night_flag_correctness(self, enriched):
        fe = FeatureEngineer()
        out = fe.fit_transform(enriched)

        assert (out.loc[out["Hour"] < 6, "Is_Night"] == 1).all()
        assert (out.loc[out["Hour"] >= 6, "Is_Night"] == 0).all()

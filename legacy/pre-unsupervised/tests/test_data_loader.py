"""Tests for TransactionDataLoader: temporal derivation and target validation."""

import pandas as pd
import pytest

from backend.src.data.data_loader import TransactionDataLoader


class TestAddTemporalFeatures:
    def test_derives_expected_columns(self, sample_transactions):
        out = TransactionDataLoader().add_temporal_features(sample_transactions)

        for col in ["Transaction_Timestamp", "Hour", "DayOfWeek", "Month", "Is_Weekend"]:
            assert col in out.columns

    def test_hour_matches_transaction_time(self, sample_transactions):
        out = TransactionDataLoader().add_temporal_features(sample_transactions)

        expected_hour = out["Transaction_Time"].str.slice(0, 2).astype(int)

        assert (out["Hour"] == expected_hour).all()

    def test_is_weekend_flags_saturday_sunday(self, sample_transactions):
        out = TransactionDataLoader().add_temporal_features(sample_transactions)

        weekend_rows = out[out["Is_Weekend"] == 1]

        assert (weekend_rows["DayOfWeek"] >= 5).all()

    def test_row_count_preserved(self, sample_transactions):
        out = TransactionDataLoader().add_temporal_features(sample_transactions)

        assert len(out) == len(sample_transactions)

    def test_unparseable_date_raises(self, sample_transactions):
        broken = sample_transactions.copy()
        broken.loc[0, "Transaction_Date"] = "not-a-date"

        with pytest.raises(ValueError):
            TransactionDataLoader().add_temporal_features(broken)


class TestLoad:
    def test_missing_file_raises(self, tmp_path):
        loader = TransactionDataLoader(path=tmp_path / "does_not_exist.csv")

        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_missing_target_column_raises(self, sample_transactions, tmp_path):
        broken = sample_transactions.drop(columns=["Is_Fraud"])
        path = tmp_path / "data.csv"
        broken.to_csv(path, index=False)

        with pytest.raises(ValueError, match="Is_Fraud"):
            TransactionDataLoader(path=path).load()

    def test_non_binary_target_raises(self, sample_transactions, tmp_path):
        broken = sample_transactions.copy()
        broken.loc[0, "Is_Fraud"] = 2
        path = tmp_path / "data.csv"
        broken.to_csv(path, index=False)

        with pytest.raises(ValueError, match="binary"):
            TransactionDataLoader(path=path).load()

    def test_real_dataset_loads_and_has_no_nulls(self, full_dataset, tmp_path):
        path = tmp_path / "data.csv"
        full_dataset.to_csv(path, index=False)

        out = TransactionDataLoader(path=path).load()

        assert len(out) == len(full_dataset)
        assert out["Is_Fraud"].isin([0, 1]).all()

"""Tests for TransactionDataLoader: timestamp derivation, city->state, no label."""

import pytest

from backend.src.data.data_loader import TransactionDataLoader


class TestAddTemporalFeatures:
    def test_derives_expected_columns(self, sample_transactions):
        out = TransactionDataLoader().add_temporal_features(sample_transactions)

        for col in [
            "Transaction_Timestamp",
            "Hour",
            "DayOfWeek",
            "Month",
            "Is_Weekend",
            "USState",
        ]:
            assert col in out.columns

    def test_hour_matches_timestamp(self, sample_transactions):
        out = TransactionDataLoader().add_temporal_features(sample_transactions)
        expected = out["TransactionDate"].str.slice(11, 13).astype(int)

        assert (out["Hour"] == expected).all()

    def test_is_weekend_flags_saturday_sunday(self, sample_transactions):
        out = TransactionDataLoader().add_temporal_features(sample_transactions)
        weekend = out[out["Is_Weekend"] == 1]

        assert (weekend["DayOfWeek"] >= 5).all()

    def test_city_maps_to_state(self, sample_transactions):
        out = TransactionDataLoader().add_temporal_features(sample_transactions)

        assert (out.loc[out["Location"] == "San Diego", "USState"] == "California").all()
        assert (out.loc[out["Location"] == "Houston", "USState"] == "Texas").all()

    def test_row_count_preserved(self, sample_transactions):
        out = TransactionDataLoader().add_temporal_features(sample_transactions)

        assert len(out) == len(sample_transactions)

    def test_unknown_city_raises(self, sample_transactions):
        broken = sample_transactions.copy()
        broken.loc[0, "Location"] = "Atlantis"

        with pytest.raises(ValueError, match="CITY_TO_STATE"):
            TransactionDataLoader().add_temporal_features(broken)

    def test_unparseable_timestamp_raises(self, sample_transactions):
        broken = sample_transactions.copy()
        broken.loc[0, "TransactionDate"] = "not-a-date"

        with pytest.raises(ValueError):
            TransactionDataLoader().add_temporal_features(broken)


class TestLoad:
    def test_missing_file_raises(self, tmp_path):
        loader = TransactionDataLoader(path=tmp_path / "nope.csv")

        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_load_drops_artefact_column_and_has_no_label(
        self, sample_transactions, tmp_path
    ):
        path = tmp_path / "data.csv"
        sample_transactions.to_csv(path, index=False)

        out = TransactionDataLoader(path=path).load()

        assert "PreviousTransactionDate" not in out.columns
        assert "Is_Fraud" not in out.columns
        assert len(out) == len(sample_transactions)

    def test_real_dataset_loads(self, full_dataset, tmp_path):
        path = tmp_path / "data.csv"
        full_dataset.to_csv(path, index=False)

        out = TransactionDataLoader(path=path).load()

        assert len(out) == len(full_dataset)
        assert out["USState"].notna().all()

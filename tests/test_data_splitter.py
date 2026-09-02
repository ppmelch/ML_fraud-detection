"""Tests for DataSplitter.temporal_split: ordering, coverage, label-free describe."""

import pandas as pd
import pytest

from backend.src.data.data_loader import TransactionDataLoader
from backend.src.data.data_splitter import DataSplitter


@pytest.fixture
def enriched(sample_transactions):
    return TransactionDataLoader().add_temporal_features(sample_transactions)


class TestTemporalSplit:
    def test_windows_are_chronologically_ordered(self, enriched):
        train, validation, test = DataSplitter(
            test_days=30, validation_days=20
        ).temporal_split(enriched)

        assert (
            train["Transaction_Timestamp"].max()
            < validation["Transaction_Timestamp"].min()
        )
        assert (
            validation["Transaction_Timestamp"].max()
            < test["Transaction_Timestamp"].min()
        )

    def test_row_count_preserved(self, enriched):
        train, validation, test = DataSplitter(
            test_days=30, validation_days=20
        ).temporal_split(enriched)

        assert len(train) + len(validation) + len(test) == len(enriched)

    def test_no_duplicate_transactions_across_windows(self, enriched):
        train, validation, test = DataSplitter(
            test_days=30, validation_days=20
        ).temporal_split(enriched)

        ids = pd.concat(
            [train["TransactionID"], validation["TransactionID"], test["TransactionID"]]
        )
        assert ids.is_unique

    def test_missing_timestamp_column_raises(self, enriched):
        with pytest.raises(KeyError):
            DataSplitter().temporal_split(enriched, timestamp_column="nope")

    def test_span_too_short_raises(self, enriched):
        with pytest.raises(ValueError, match="spans"):
            DataSplitter(test_days=500, validation_days=500).temporal_split(enriched)

    def test_describe_has_no_fraud_fields(self, enriched):
        splitter = DataSplitter(test_days=30, validation_days=20)
        summary = splitter.describe_temporal_split(*splitter.temporal_split(enriched))

        assert summary["strategy"] == "temporal"
        for window in ("train", "validation", "test"):
            assert set(summary[window]) == {"rows", "start", "end"}

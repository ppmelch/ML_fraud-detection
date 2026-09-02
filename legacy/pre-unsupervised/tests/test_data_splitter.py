"""Tests for DataSplitter.temporal_split: ordering, no leakage, coverage."""

import pandas as pd
import pytest

from backend.src.data.data_loader import TransactionDataLoader
from backend.src.data.data_splitter import DataSplitter


@pytest.fixture
def enriched(sample_transactions):
    return TransactionDataLoader().add_temporal_features(sample_transactions)


class TestTemporalSplit:
    def test_windows_are_chronologically_ordered(self, enriched):
        splitter = DataSplitter(test_days=5, validation_days=3)
        train, validation, test = splitter.temporal_split(enriched)

        assert train["Transaction_Timestamp"].max() < validation["Transaction_Timestamp"].min()
        assert validation["Transaction_Timestamp"].max() < test["Transaction_Timestamp"].min()

    def test_row_count_preserved(self, enriched):
        splitter = DataSplitter(test_days=5, validation_days=3)
        train, validation, test = splitter.temporal_split(enriched)

        assert len(train) + len(validation) + len(test) == len(enriched)

    def test_no_duplicate_rows_across_windows(self, enriched):
        splitter = DataSplitter(test_days=5, validation_days=3)
        train, validation, test = splitter.temporal_split(enriched)

        ids = pd.concat([train["Transaction_ID"], validation["Transaction_ID"], test["Transaction_ID"]])

        assert ids.is_unique

    def test_missing_timestamp_column_raises(self, enriched):
        splitter = DataSplitter()

        with pytest.raises(KeyError):
            splitter.temporal_split(enriched, timestamp_column="does_not_exist")

    def test_span_too_short_raises(self, enriched):
        splitter = DataSplitter(test_days=100, validation_days=100)

        with pytest.raises(ValueError, match="spans"):
            splitter.temporal_split(enriched)

    def test_describe_reports_correct_fraud_rate(self, enriched):
        splitter = DataSplitter(test_days=5, validation_days=3)
        train, validation, test = splitter.temporal_split(enriched)

        summary = splitter.describe_temporal_split(train, validation, test)

        assert summary["train"]["fraud_rate"] == pytest.approx(
            train["Is_Fraud"].mean()
        )
        assert summary["train"]["fraud_cases"] == int(train["Is_Fraud"].sum())

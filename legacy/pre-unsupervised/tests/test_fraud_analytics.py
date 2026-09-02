"""Tests for FraudAnalytics: rates computed from Is_Fraud, guards, totals."""

import pandas as pd
import pytest

from backend.src.data.data_loader import TransactionDataLoader
from backend.src.metrics.fraud_analytics import FraudAnalytics, day_name, fraud_rate


class TestFraudRateHelper:
    def test_basic_ratio(self):
        assert fraud_rate(5, 100) == pytest.approx(0.05)

    def test_zero_total_returns_zero_not_nan_or_error(self):
        assert fraud_rate(0, 0) == 0.0


class TestDayName:
    def test_known_index(self):
        assert day_name(0) == "Monday"
        assert day_name(6) == "Sunday"

    def test_out_of_range_returns_unknown(self):
        assert day_name(9) == "Unknown"
        assert day_name(None) == "Unknown"


@pytest.fixture
def enriched(sample_transactions):
    return TransactionDataLoader().add_temporal_features(sample_transactions)


class TestOverview:
    def test_totals_match_dataframe(self, enriched):
        overview = FraudAnalytics(enriched).overview()

        assert overview["total_transactions"] == len(enriched)
        assert overview["fraud_cases"] == int(enriched["Is_Fraud"].sum())
        assert overview["fraud_rate"] == pytest.approx(enriched["Is_Fraud"].mean())

    def test_missing_target_column_raises(self, enriched):
        with pytest.raises(KeyError):
            FraudAnalytics(enriched.drop(columns=["Is_Fraud"]))


class TestByColumn:
    def test_rate_matches_manual_groupby(self, enriched):
        records = FraudAnalytics(enriched).by_column("Merchant_Category")

        manual = enriched.groupby("Merchant_Category")["Is_Fraud"].mean()

        for record in records:
            assert record["fraud_rate"] == pytest.approx(
                manual.loc[record["category"]]
            )

    def test_min_transactions_filters_small_buckets(self, enriched):
        records = FraudAnalytics(enriched).by_column(
            "Merchant_Category", min_transactions=10_000
        )
        assert records == []

    def test_sorted_descending_by_rate(self, enriched):
        records = FraudAnalytics(enriched).by_column("State")
        rates = [r["fraud_rate"] for r in records]
        assert rates == sorted(rates, reverse=True)

    def test_unknown_column_raises(self, enriched):
        with pytest.raises(KeyError):
            FraudAnalytics(enriched).by_column("does_not_exist")


class TestByHourAndDay:
    def test_by_hour_totals_sum_to_dataset(self, enriched):
        records = FraudAnalytics(enriched).by_hour()
        assert sum(r["total_transactions"] for r in records) == len(enriched)
        assert sum(r["fraud_cases"] for r in records) == int(enriched["Is_Fraud"].sum())

    def test_by_day_of_week_totals_sum_to_dataset(self, enriched):
        records = FraudAnalytics(enriched).by_day_of_week()
        assert sum(r["total_transactions"] for r in records) == len(enriched)
        assert sum(r["fraud_cases"] for r in records) == int(enriched["Is_Fraud"].sum())

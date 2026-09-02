"""Tests for StateAnalytics: totals reconcile, rate-based peaks, support floor."""

import pandas as pd
import pytest

from backend.src.data.data_loader import TransactionDataLoader
from backend.src.metrics.state_analytics import StateAnalytics


@pytest.fixture
def enriched(sample_transactions):
    return TransactionDataLoader().add_temporal_features(sample_transactions)


class TestCompute:
    def test_totals_reconcile_with_dataset(self, enriched):
        records = StateAnalytics(enriched, min_hour_bucket=1, min_day_bucket=1).compute()

        assert sum(r["total_transactions"] for r in records) == len(enriched)
        assert sum(r["fraud_cases"] for r in records) == int(enriched["Is_Fraud"].sum())

    def test_covers_every_state_in_data(self, enriched):
        records = StateAnalytics(enriched, min_hour_bucket=1, min_day_bucket=1).compute()

        assert {r["state"] for r in records} == set(enriched["State"].unique())

    def test_no_duplicate_states(self, enriched):
        records = StateAnalytics(enriched, min_hour_bucket=1, min_day_bucket=1).compute()
        states = [r["state"] for r in records]
        assert len(states) == len(set(states))

    def test_missing_required_column_raises(self, enriched):
        with pytest.raises(KeyError):
            StateAnalytics(enriched.drop(columns=["Hour"]))

    def test_no_nan_in_output(self, enriched):
        records = StateAnalytics(enriched, min_hour_bucket=1, min_day_bucket=1).compute()

        for record in records:
            for key, value in record.items():
                if isinstance(value, float):
                    assert value == value  # NaN != NaN


class TestPeakBucketRateVsCount:
    def test_peak_hour_is_not_simply_the_busiest_hour(self):
        # Hour 0: 100 txns, 10 fraud (10% rate). Hour 1: 5 txns, 3 fraud (60% rate).
        # By count, hour 0 dominates fraud cases; by rate, hour 1 must win.
        rows = []
        for i in range(100):
            rows.append({"Hour": 0, "Is_Fraud": 1 if i < 10 else 0})
        for i in range(5):
            rows.append({"Hour": 1, "Is_Fraud": 1 if i < 3 else 0})

        frame = pd.DataFrame(rows)
        frame["State"] = "TestState"
        frame["DayOfWeek"] = 0
        frame["Transaction_Amount"] = 100.0
        frame["Account_Balance"] = 1000.0
        frame["Transaction_Device"] = "ATM"
        frame["Device_Type"] = "ATM"
        frame["Merchant_Category"] = "Groceries"

        analytics = StateAnalytics(frame, min_hour_bucket=1, min_day_bucket=1)
        records = analytics.compute()

        assert records[0]["peak_fraud_hour"] == 1

    def test_min_transactions_excludes_low_support_bucket(self):
        rows = []
        for i in range(100):
            rows.append({"Hour": 0, "Is_Fraud": 1 if i < 10 else 0})
        # Hour 1 has only 2 transactions, both fraud: 100% rate, but should
        # not win the peak when min_hour_bucket excludes it.
        for i in range(2):
            rows.append({"Hour": 1, "Is_Fraud": 1})

        frame = pd.DataFrame(rows)
        frame["State"] = "TestState"
        frame["DayOfWeek"] = 0
        frame["Transaction_Amount"] = 100.0
        frame["Account_Balance"] = 1000.0
        frame["Transaction_Device"] = "ATM"
        frame["Device_Type"] = "ATM"
        frame["Merchant_Category"] = "Groceries"

        analytics = StateAnalytics(frame, min_hour_bucket=30, min_day_bucket=1)
        records = analytics.compute()

        assert records[0]["peak_fraud_hour"] == 0

    def test_all_buckets_below_floor_returns_none(self):
        frame = pd.DataFrame(
            {
                "Hour": [0, 1],
                "Is_Fraud": [1, 0],
                "State": ["TestState", "TestState"],
                "DayOfWeek": [0, 0],
                "Transaction_Amount": [100.0, 100.0],
                "Account_Balance": [1000.0, 1000.0],
                "Transaction_Device": ["ATM", "ATM"],
                "Device_Type": ["ATM", "ATM"],
                "Merchant_Category": ["Groceries", "Groceries"],
            }
        )

        analytics = StateAnalytics(frame, min_hour_bucket=100, min_day_bucket=100)
        records = analytics.compute()

        assert records[0]["peak_fraud_hour"] is None
        assert records[0]["peak_fraud_day_index"] is None


class TestMostCommonFraudValue:
    def test_state_with_no_fraud_returns_none(self):
        frame = pd.DataFrame(
            {
                "Hour": [0, 1],
                "Is_Fraud": [0, 0],
                "State": ["TestState", "TestState"],
                "DayOfWeek": [0, 0],
                "Transaction_Amount": [100.0, 100.0],
                "Account_Balance": [1000.0, 1000.0],
                "Transaction_Device": ["ATM", "POS"],
                "Device_Type": ["ATM", "POS"],
                "Merchant_Category": ["Groceries", "Health"],
            }
        )

        records = StateAnalytics(frame, min_hour_bucket=1, min_day_bucket=1).compute()

        assert records[0]["most_common_fraud_device"] is None
        assert records[0]["top_fraud_merchant_category"] is None

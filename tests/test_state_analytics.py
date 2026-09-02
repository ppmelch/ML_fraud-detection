"""Tests for StateAnalytics: totals reconcile, rate-based peaks, support floor."""

import numpy as np
import pandas as pd
import pytest

from backend.src.data.data_loader import TransactionDataLoader
from backend.src.metrics.state_analytics import StateAnalytics


def _base_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["USState"] = "California"
    for column, default in (
        ("DayOfWeek", 0),
        ("TransactionAmount", 100.0),
        ("AccountBalance", 1000.0),
        ("Channel", "ATM"),
        ("CustomerOccupation", "Doctor"),
        ("anomaly_score", 0.1),
    ):
        if column not in frame.columns:
            frame[column] = default
    return frame


@pytest.fixture
def scored(sample_transactions):
    data = TransactionDataLoader().add_temporal_features(sample_transactions)
    rng = np.random.default_rng(0)
    data["anomaly_score"] = rng.uniform(0, 1, len(data))
    data["is_anomaly"] = (data["anomaly_score"] >= 0.8).astype(int)
    return data


class TestCompute:
    def test_totals_reconcile(self, scored):
        records = StateAnalytics(scored, min_hour_bucket=1, min_day_bucket=1).compute()

        assert sum(r["total_transactions"] for r in records) == len(scored)
        assert sum(r["flagged"] for r in records) == int(scored["is_anomaly"].sum())

    def test_covers_every_state(self, scored):
        records = StateAnalytics(scored, min_hour_bucket=1, min_day_bucket=1).compute()
        assert {r["state"] for r in records} == set(scored["USState"].unique())

    def test_contract_keys_present(self, scored):
        records = StateAnalytics(scored, min_hour_bucket=1, min_day_bucket=1).compute()
        for key in (
            "state",
            "total_transactions",
            "flagged",
            "anomaly_rate",
            "mean_anomaly_score",
            "avg_transaction_amount",
            "avg_account_balance",
            "peak_anomaly_hour",
            "peak_anomaly_hour_rate",
            "peak_anomaly_day",
            "peak_anomaly_day_rate",
            "most_common_flagged_channel",
            "top_flagged_occupation",
            "low_support",
        ):
            assert key in records[0]

    def test_missing_required_column_raises(self, scored):
        with pytest.raises(KeyError):
            StateAnalytics(scored.drop(columns=["Hour"]))

    def test_no_nan_in_output(self, scored):
        records = StateAnalytics(scored, min_hour_bucket=1, min_day_bucket=1).compute()
        for record in records:
            for value in record.values():
                if isinstance(value, float):
                    assert value == value


class TestPeakBucket:
    def test_peak_hour_is_rate_not_count(self):
        rows = [{"Hour": 0, "is_anomaly": 1 if i < 10 else 0} for i in range(100)]
        rows += [{"Hour": 1, "is_anomaly": 1 if i < 3 else 0} for i in range(5)]

        records = StateAnalytics(
            _base_frame(rows), min_hour_bucket=1, min_day_bucket=1
        ).compute()

        assert records[0]["peak_anomaly_hour"] == 1

    def test_support_floor_excludes_tiny_bucket(self):
        rows = [{"Hour": 0, "is_anomaly": 1 if i < 10 else 0} for i in range(100)]
        rows += [{"Hour": 1, "is_anomaly": 1} for _ in range(2)]

        records = StateAnalytics(
            _base_frame(rows), min_hour_bucket=30, min_day_bucket=1
        ).compute()

        assert records[0]["peak_anomaly_hour"] == 0

    def test_all_below_floor_returns_none(self):
        rows = [{"Hour": 0, "is_anomaly": 1}, {"Hour": 1, "is_anomaly": 0}]
        records = StateAnalytics(
            _base_frame(rows), min_hour_bucket=100, min_day_bucket=100
        ).compute()

        assert records[0]["peak_anomaly_hour"] is None
        assert records[0]["peak_anomaly_day"] is None


class TestTopFlaggedValue:
    def test_state_with_no_flags_returns_none(self):
        rows = [
            {"Hour": 0, "is_anomaly": 0, "Channel": "ATM"},
            {"Hour": 1, "is_anomaly": 0, "Channel": "Online"},
        ]
        records = StateAnalytics(
            _base_frame(rows), min_hour_bucket=1, min_day_bucket=1
        ).compute()

        assert records[0]["most_common_flagged_channel"] is None
        assert records[0]["top_flagged_occupation"] is None

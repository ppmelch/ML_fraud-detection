"""Tests for TransactionAnalytics: score-driven breakdowns, guards, no NaN."""

import numpy as np
import pandas as pd
import pytest

from backend.src.data.data_loader import TransactionDataLoader
from backend.src.metrics.transaction_analytics import (
    DAY_NAMES,
    TransactionAnalytics,
)


@pytest.fixture
def scored(sample_transactions):
    data = TransactionDataLoader().add_temporal_features(sample_transactions)
    rng = np.random.default_rng(0)
    data["anomaly_score"] = rng.uniform(0, 1, len(data))
    data["is_anomaly"] = (data["anomaly_score"] >= 0.9).astype(int)
    return data


class TestOverview:
    def test_headline_keys_present(self, scored):
        overview = TransactionAnalytics(scored).overview()

        for key in (
            "total_transactions",
            "total_accounts",
            "total_amount",
            "avg_transaction_amount",
            "avg_account_balance",
            "states_covered",
            "period_start",
            "period_end",
            "flagged_transactions",
            "flagged_rate",
            "mean_anomaly_score",
            "flagged_amount",
        ):
            assert key in overview

    def test_flagged_count_reconciles(self, scored):
        overview = TransactionAnalytics(scored).overview()
        assert overview["flagged_transactions"] == int(scored["is_anomaly"].sum())

    def test_missing_score_column_raises(self, scored):
        with pytest.raises(KeyError):
            TransactionAnalytics(scored.drop(columns=["anomaly_score"]))


class TestBreakdowns:
    def test_bucket_record_shape(self, scored):
        rows = TransactionAnalytics(scored).by_channel()
        assert rows
        for row in rows:
            assert set(row) == {
                "category",
                "total_transactions",
                "flagged",
                "anomaly_rate",
                "mean_score",
            }

    def test_by_hour_uses_bucket_key_and_is_ordered(self, scored):
        rows = TransactionAnalytics(scored).by_hour()
        buckets = [r["bucket"] for r in rows]
        assert buckets == sorted(buckets)

    def test_by_day_of_week_labels_are_names(self, scored):
        rows = TransactionAnalytics(scored).by_day_of_week()
        assert all(r["bucket"] in DAY_NAMES for r in rows)

    def test_totals_reconcile(self, scored):
        rows = TransactionAnalytics(scored).by_transaction_type()
        assert sum(r["total_transactions"] for r in rows) == len(scored)

    def test_distributions_equal_length_no_nan(self, scored):
        summary = TransactionAnalytics(scored).summary()

        amt = summary["amount_distribution"]
        assert len(amt["bin_centers"]) == len(amt["normal_counts"]) == len(amt["flagged_counts"])

        score = summary["score_distribution"]
        assert len(score["bin_centers"]) == len(score["counts"])
        assert "nan" not in str(summary).lower()

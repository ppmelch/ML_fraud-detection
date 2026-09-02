"""
Shared pytest fixtures for the backend test suite.

``sample_transactions`` builds a small synthetic frame with the raw
*unlabelled* schema of ``bank_transactions_data_2.csv`` so unit tests do not
depend on the real CSV. ``AccountID`` repeats (~40 accounts x 6 transactions)
so the per-account features are exercised, ``TransactionDate`` spans well over
``TEST_DAYS + VALIDATION_DAYS + 1`` days so a temporal split is possible, and
``Location`` covers cities in several US states.

``full_dataset`` loads the real file for the handful of tests that check
something about the actual data.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.src.modeling.config import DATASET_PATH

#: Cities spanning >= 4 US states (see CITY_TO_STATE).
_CITIES = [
    "San Diego",   # California
    "Los Angeles",  # California
    "Houston",     # Texas
    "Dallas",      # Texas
    "Miami",       # Florida
    "Chicago",     # Illinois
    "Phoenix",     # Arizona
    "Seattle",     # Washington
]


@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    """
    A small synthetic frame with the raw unlabelled dataset schema.

    Returns
    -------
    pd.DataFrame
        240 rows: 40 accounts x 6 transactions, spanning 200 days, every
        column required by ``TransactionDataLoader``, ``FeatureEngineer`` and
        ``DataPreparation``. No fraud label.
    """
    rng = np.random.default_rng(42)

    n_accounts = 40
    per_account = 6
    n = n_accounts * per_account

    start = pd.Timestamp("2023-01-05 00:00:00")
    account_ids = [f"AC{idx:05d}" for idx in range(n_accounts)]

    rows = []
    for a_idx, account in enumerate(account_ids):
        for t in range(per_account):
            offset_days = rng.integers(0, 200)
            offset_secs = int(rng.integers(0, 86400))
            timestamp = start + pd.Timedelta(days=int(offset_days), seconds=offset_secs)
            rows.append(
                {
                    "TransactionID": f"TX{a_idx * per_account + t:06d}",
                    "AccountID": account,
                    "TransactionAmount": round(float(rng.uniform(5, 1500)), 2),
                    "TransactionDate": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "TransactionType": rng.choice(["Debit", "Credit"]),
                    "Location": _CITIES[(a_idx + t) % len(_CITIES)],
                    "DeviceID": f"D{rng.integers(0, 25):04d}",
                    "IP Address": f"10.0.{rng.integers(0, 255)}.{rng.integers(0, 255)}",
                    "MerchantID": f"M{rng.integers(0, 30):03d}",
                    "Channel": rng.choice(["ATM", "Online", "Branch"]),
                    "CustomerAge": int(rng.integers(18, 85)),
                    "CustomerOccupation": rng.choice(
                        ["Doctor", "Student", "Retired", "Engineer"]
                    ),
                    "TransactionDuration": int(rng.integers(10, 300)),
                    "LoginAttempts": int(rng.integers(1, 5)),
                    "AccountBalance": round(float(rng.uniform(200, 30000)), 2),
                    "PreviousTransactionDate": "2024-11-04 08:08:08",
                }
            )

    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def full_dataset() -> pd.DataFrame:
    """
    The real dataset, loaded once per test session.

    Skips dependent tests when the CSV is not present rather than failing.

    Returns
    -------
    pd.DataFrame
        Raw dataset as read by ``pandas.read_csv``.
    """
    if not DATASET_PATH.exists():
        pytest.skip(f"Dataset not found at {DATASET_PATH}")

    return pd.read_csv(DATASET_PATH)

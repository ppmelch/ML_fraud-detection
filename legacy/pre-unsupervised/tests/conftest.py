"""
Shared pytest fixtures for the backend test suite.

`sample_transactions` builds a small, synthetic-but-realistic frame with the
raw dataset schema so unit tests do not depend on the 200k-row CSV being
present, while `full_dataset` loads the real file for the handful of tests
that need to check something about the actual data (state coverage, no
nulls, temporal span).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.src.modeling.config import DATASET_PATH


@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    """
    A small synthetic frame with the raw dataset's columns and dtypes.

    Returns
    -------
    pd.DataFrame
        200 rows spanning 20 days, ~5% fraud, all columns required by
        `TransactionDataLoader`, `FeatureEngineer` and `DataPreparation`.
    """
    rng = np.random.default_rng(42)
    n = 200

    states = ["Maharashtra", "Kerala", "Delhi", "Punjab"]
    categories = ["Groceries", "Electronics", "Restaurant", "Health"]
    account_types = ["Savings", "Checking", "Business"]
    device_types = ["Mobile", "Desktop", "POS", "ATM"]
    devices = ["QR Code Scanner", "ATM Booth Kiosk", "Web Browser"]

    dates = pd.date_range("2025-01-01", periods=20, freq="D")

    data = pd.DataFrame(
        {
            "Customer_ID": [f"cust-{i}" for i in range(n)],
            "Customer_Name": [f"Name {i}" for i in range(n)],
            "Gender": rng.choice(["Male", "Female"], n),
            "Age": rng.integers(18, 80, n),
            "State": rng.choice(states, n),
            "City": rng.choice(["CityA", "CityB"], n),
            "Bank_Branch": rng.choice(["BranchA", "BranchB"], n),
            "Account_Type": rng.choice(account_types, n),
            "Transaction_ID": [f"txn-{i}" for i in range(n)],
            "Transaction_Date": rng.choice(dates, n),
            "Transaction_Time": [
                f"{h:02d}:{m:02d}:{s:02d}"
                for h, m, s in zip(
                    rng.integers(0, 24, n),
                    rng.integers(0, 60, n),
                    rng.integers(0, 60, n),
                )
            ],
            "Transaction_Amount": rng.uniform(10, 90000, n),
            "Merchant_ID": [f"merch-{i}" for i in range(n)],
            "Transaction_Type": rng.choice(
                ["Transfer", "Credit", "Debit", "Withdrawal", "Bill Payment"], n
            ),
            "Merchant_Category": rng.choice(categories, n),
            "Account_Balance": rng.uniform(5000, 99999, n),
            "Transaction_Device": rng.choice(devices, n),
            "Transaction_Location": rng.choice(["CityA, X", "CityB, Y"], n),
            "Device_Type": rng.choice(device_types, n),
            "Is_Fraud": rng.choice([0, 1], n, p=[0.95, 0.05]),
            "Transaction_Currency": "INR",
            "Customer_Contact": [f"+91900000{i:04d}" for i in range(n)],
            "Transaction_Description": "Test transaction",
            "Customer_Email": [f"user{i}@example.com" for i in range(n)],
        }
    )

    data["Transaction_Date"] = data["Transaction_Date"].dt.strftime("%d-%m-%Y")

    return data


@pytest.fixture(scope="session")
def full_dataset() -> pd.DataFrame:
    """
    The real dataset, loaded once per test session.

    Skips dependent tests when the CSV is not present rather than failing,
    since the raw data file may legitimately be absent from a checkout.

    Returns
    -------
    pd.DataFrame
        Raw dataset as read by ``pandas.read_csv``.
    """
    if not DATASET_PATH.exists():
        pytest.skip(f"Dataset not found at {DATASET_PATH}")

    return pd.read_csv(DATASET_PATH)

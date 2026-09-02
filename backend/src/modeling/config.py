"""
Configuration for the unsupervised US transaction anomaly-detection project.

This module centralizes:

- The dataset schema (which raw columns exist, which are identifiers, which
  become features).
- The engineered feature set and the statistics fitted to build it.
- Model hyperparameters for the three anomaly estimators.
- Every filesystem path used anywhere in the backend.

There is no fraud label in this dataset. Nothing here references a target
column; the models score how unusual a transaction is, not how likely it is
to be fraud.
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Project paths
#
# Every path used anywhere in the backend is derived from these constants.
# No module builds a path by string concatenation or hardcodes a literal.
# ---------------------------------------------------------------------------

# Root directory of the source package (backend/src)
BASE_DIR = Path(__file__).resolve().parent.parent

# Repository root (the directory containing backend/, frontend/, data/)
PROJECT_ROOT = BASE_DIR.parent.parent

# Directory where trained model bundles are stored
MODELS_DIR = BASE_DIR / "models"

# Raw data
#
# ``DATASET_PATH`` points at the *augmented* dataset: the real
# ``bank_transactions_data_2.csv`` plus synthetic rows for the 25 US states the
# real 43 cities never touch, so the state choropleth has a value for every
# GeoJSON polygon. Synthetic rows carry a ``TXS`` / ``ACS`` id prefix and are
# produced by ``scripts/augment_dataset.py``. See ``docs/dataset.md``.
DATA_DIR = PROJECT_ROOT / "data"
REAL_DATASET_PATH = DATA_DIR / "bank_transactions_data_2.csv"
DATASET_PATH = DATA_DIR / "bank_transactions_data_2_full.csv"

# Directory where the pipeline writes its JSON artifacts (the API reads these)
ARTIFACTS_DIR = PROJECT_ROOT / "backend" / "artifacts"

# US state boundaries consumed by the interactive map. The canonical
# Leaflet-tutorial FeatureCollection: 50 states + DC + Puerto Rico, with
# ``properties.name`` per feature and a top-level ``feature["id"]`` (FIPS).
GEOJSON_PATH = PROJECT_ROOT / "frontend" / "data" / "us-states.json"

# Property holding the state name; and the key holding the stable identifier.
# In this file ``id`` is top-level on the feature, not inside ``properties`` —
# geo_mapping reads ``feature.get("id") or feature["properties"].get("id")``.
GEOJSON_NAME_PROPERTY = "name"
GEOJSON_ID_PROPERTY = "id"


# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------

# One combined datetime column, parsed to ``Transaction_Timestamp``.
TIMESTAMP_COLUMN = "TransactionDate"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# ``PreviousTransactionDate`` is a data-extraction artefact (every value sits
# inside a 6-minute window on 2024-11-04) and must never be used as a feature.
ARTEFACT_COLUMNS = ["PreviousTransactionDate"]

# Columns that identify a device, an account or a single transaction. These
# are never model features. ``AccountID`` is kept on the frame (it repeats
# ~5x and drives the per-account deviation features) but is not itself encoded.
IDENTIFIER_COLUMNS = [
    "TransactionID",
    "AccountID",
    "DeviceID",
    "IP Address",
    "MerchantID",
]

# Raw columns that enter the model directly (before engineering).
RAW_FEATURES = [
    "TransactionAmount",
    "TransactionType",
    "USState",
    "Channel",
    "CustomerAge",
    "CustomerOccupation",
    "TransactionDuration",
    "LoginAttempts",
    "AccountBalance",
]

# Categorical members of RAW_FEATURES. Only these are one-hot encoded.
CATEGORICAL_FEATURES = [
    "TransactionType",
    "USState",
    "Channel",
    "CustomerOccupation",
]

# Behavioural features derived in FeatureEngineer. All are computable from a
# single transaction plus statistics fitted on the training window.
ENGINEERED_FEATURES = [
    "Hour",
    "DayOfWeek",
    "Month",
    "Is_Weekend",
    "Is_Night",
    "Amount_To_Balance_Ratio",
    "Balance_After_Transaction",
    "Log_Transaction_Amount",
    "Log_Account_Balance",
    "Log_Transaction_Duration",
    "Amount_Dev_Account",
    "Amount_Dev_USState",
    "Amount_Dev_Hour",
    "Amount_Dev_MerchantID",
    "Duration_Dev_Channel",
    "USState_Frequency",
    "Channel_Frequency",
    "MerchantID_Frequency",
    "DeviceID_Frequency",
    "Hour_Frequency",
    "Occupation_Frequency",
    "Account_Txn_Count",
    "Days_Since_Prev_Account_Txn",
]

# Grouping keys for the amount-deviation z-score features. The engineer maps
# each group name to a suffix used in the feature name (e.g. ``AccountID`` ->
# ``Amount_Dev_Account``).
DEVIATION_GROUPS = ["AccountID", "USState", "Hour", "MerchantID"]

# Human-readable suffixes for the deviation feature names.
DEVIATION_GROUP_SUFFIX = {
    "AccountID": "Account",
    "USState": "USState",
    "Hour": "Hour",
    "MerchantID": "MerchantID",
}

# Columns replaced by the training share of each category value.
FREQUENCY_COLUMNS = [
    "USState",
    "Channel",
    "MerchantID",
    "DeviceID",
    "Hour",
    "CustomerOccupation",
]

# Suffixes for the frequency feature names.
FREQUENCY_COLUMN_SUFFIX = {
    "USState": "USState",
    "Channel": "Channel",
    "MerchantID": "MerchantID",
    "DeviceID": "DeviceID",
    "Hour": "Hour",
    "CustomerOccupation": "Occupation",
}

# The single column whose per-group mean/std drives ``Duration_Dev_Channel``.
DURATION_DEVIATION_GROUP = "Channel"


# ---------------------------------------------------------------------------
# Evaluation protocol
# ---------------------------------------------------------------------------

RANDOM_STATE = 42

# The dataset spans 2023-01 .. 2024-01. The honest protocol is temporal:
# fit on the past, choose the contamination cutoff on the recent past, and
# score the future once.
#   train      : everything before the validation window
#   validation : 30 days, cutoff selection only
#   test       : last 60 days, touched once at the very end
SPLIT_STRATEGY = "temporal"
TEST_DAYS = 60
VALIDATION_DAYS = 30

# Fallback only, if the timestamp column ever stops supporting a temporal split.
TEST_SIZE = 0.2

# Fraction of transactions treated as anomalous. Feeds the IsolationForest /
# LOF ``contamination`` param and the quantile cutoff on the train scores.
CONTAMINATION = 0.02

# Default model for the served anomaly detector.
DEFAULT_MODEL = "isolation_forest"

MODEL_CONFIG = {
    "isolation_forest": {
        "n_estimators": 300,
        "max_samples": "auto",
        "contamination": CONTAMINATION,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "lof": {
        "n_neighbors": 20,
        "contamination": CONTAMINATION,
        "novelty": True,
    },
    "autoencoder": {
        "hidden_layer_sizes": (32, 8, 32),
        "activation": "relu",
        "max_iter": 400,
        "early_stopping": True,
        "random_state": RANDOM_STATE,
    },
}

# Rows sampled for SHAP / permutation importance.
SHAP_SAMPLE_SIZE = 1000


# ---------------------------------------------------------------------------
# Analytics guards
#
# Peak anomaly hour / day are ranked by RATE, not by count. A bucket with a
# handful of transactions and one flagged has a huge rate and no meaning, so
# any bucket below the minimum support is excluded from the ranking. The
# dataset is small (~2.5k rows), so the floors are low.
# ---------------------------------------------------------------------------

MIN_TRANSACTIONS_PER_HOUR_BUCKET = 20
MIN_TRANSACTIONS_PER_DAY_BUCKET = 20
MIN_TRANSACTIONS_PER_STATE = 15


# ---------------------------------------------------------------------------
# City -> US state
#
# ``Location`` in the raw CSV is one of 43 US city names. The map view is a
# state choropleth, so each city is resolved to its state. Every value below
# is confirmed to match a feature ``name`` in ``us-states.json`` exactly.
#
# The first 43 entries are the real cities. The remaining 25 (marked below)
# are the one-city-per-state placeholders introduced with the synthetic
# state-coverage augmentation (``scripts/augment_dataset.py``); each was picked
# to collide with neither a real city nor another placeholder.
# ---------------------------------------------------------------------------

CITY_TO_STATE = {
    "Albuquerque": "New Mexico",
    "Atlanta": "Georgia",
    "Austin": "Texas",
    "Baltimore": "Maryland",
    "Boston": "Massachusetts",
    "Charlotte": "North Carolina",
    "Chicago": "Illinois",
    "Colorado Springs": "Colorado",
    "Columbus": "Ohio",
    "Dallas": "Texas",
    "Denver": "Colorado",
    "Detroit": "Michigan",
    "El Paso": "Texas",
    "Fort Worth": "Texas",
    "Fresno": "California",
    "Houston": "Texas",
    "Indianapolis": "Indiana",
    "Jacksonville": "Florida",
    "Kansas City": "Missouri",
    "Las Vegas": "Nevada",
    "Los Angeles": "California",
    "Louisville": "Kentucky",
    "Memphis": "Tennessee",
    "Mesa": "Arizona",
    "Miami": "Florida",
    "Milwaukee": "Wisconsin",
    "Nashville": "Tennessee",
    "New York": "New York",
    "Oklahoma City": "Oklahoma",
    "Omaha": "Nebraska",
    "Philadelphia": "Pennsylvania",
    "Phoenix": "Arizona",
    "Portland": "Oregon",
    "Raleigh": "North Carolina",
    "Sacramento": "California",
    "San Antonio": "Texas",
    "San Diego": "California",
    "San Francisco": "California",
    "San Jose": "California",
    "Seattle": "Washington",
    "Tucson": "Arizona",
    "Virginia Beach": "Virginia",
    "Washington": "District of Columbia",
    # --- synthetic state-coverage placeholders (one city per missing state) ---
    "Birmingham": "Alabama",
    "Anchorage": "Alaska",
    "Little Rock": "Arkansas",
    "Bridgeport": "Connecticut",
    "Wilmington": "Delaware",
    "Honolulu": "Hawaii",
    "Boise": "Idaho",
    "Des Moines": "Iowa",
    "Wichita": "Kansas",
    "New Orleans": "Louisiana",
    "Bangor": "Maine",
    "Minneapolis": "Minnesota",
    "Jackson": "Mississippi",
    "Billings": "Montana",
    "Manchester": "New Hampshire",
    "Newark": "New Jersey",
    "Fargo": "North Dakota",
    "San Juan": "Puerto Rico",
    "Providence": "Rhode Island",
    "Columbia": "South Carolina",
    "Sioux Falls": "South Dakota",
    "Salt Lake City": "Utah",
    "Burlington": "Vermont",
    "Charleston": "West Virginia",
    "Cheyenne": "Wyoming",
}


# ---------------------------------------------------------------------------
# State name reconciliation
#
# Every CITY_TO_STATE value already matches a GeoJSON ``properties.name``
# exactly, so no overrides are needed. The hook is kept (small) for the rare
# real mismatch; StateGeoMapper still reports every unmatched name on both
# sides regardless.
# ---------------------------------------------------------------------------

STATE_NAME_MAP: dict[str, str] = {}

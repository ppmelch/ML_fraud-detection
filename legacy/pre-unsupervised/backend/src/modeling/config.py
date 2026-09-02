"""
Configuration module for the credit risk modeling project.

This module centralizes:
- Hyperparameters for different machine learning models
- Project directory paths

This design allows easy modification of model settings and ensures
consistency across the pipeline.
"""

from pathlib import Path


MODEL_CONFIG = {
    "logistic": {
        "max_iter": 1000,
        "solver": "lbfgs",
        "class_weight": "balanced",
        "random_state": 42
    },

    "random_forest": {
        "n_estimators": 300,
        "max_depth": 5,
        "min_samples_leaf": 20,
        "min_samples_split": 40,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1
    },

    "xgboost": {
        "objective": "binary:logistic",
        "n_estimators": 250,
        "max_depth": 4,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "gamma": 1,
        "reg_alpha": 1,
        "reg_lambda": 2,
        "eval_metric": "logloss",
        "random_state": 42
    }, 
    
    "lightgbm": {
        "objective": "binary",
        "n_estimators": 300,
        "learning_rate": 0.01,
        "max_depth": 4,
        "num_leaves": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 2,
        "reg_lambda": 3,
        "class_weight": "balanced",
        "random_state": 42,
        "verbosity": -1
    }
}


OPTUNA_SEARCH_SPACE = {

    "logistic": {

        "C": {
            "type": "float",
            "low": 0.001,
            "high": 10
        }
    },

    "random_forest": {

        "n_estimators": {
            "type": "int",
            "low": 100,
            "high": 500
        },

        "max_depth": {
            "type": "int",
            "low": 3,
            "high": 15
        },

        "min_samples_split": {
            "type": "int",
            "low": 2,
            "high": 50
        },

        "min_samples_leaf": {
            "type": "int",
            "low": 1,
            "high": 20
        }
    },

    "xgboost": {

        "n_estimators": {
            "type": "int",
            "low": 100,
            "high": 500
        },

        "max_depth": {
            "type": "int",
            "low": 3,
            "high": 10
        },

        "learning_rate": {
            "type": "float",
            "low": 0.01,
            "high": 0.2
        },

        "subsample": {
            "type": "float",
            "low": 0.6,
            "high": 1.0
        },

        "colsample_bytree": {
            "type": "float",
            "low": 0.6,
            "high": 1.0
        },

        "min_child_weight": {
            "type": "int",
            "low": 1,
            "high": 15
        },

        "gamma": {
            "type": "float",
            "low": 0,
            "high": 5
        },

        "reg_alpha": {
            "type": "float",
            "low": 0,
            "high": 5
        },

        "reg_lambda": {
            "type": "float",
            "low": 0,
            "high": 10
        }
    },

    "lightgbm": {

        "n_estimators": {
            "type": "int",
            "low": 100,
            "high": 500
        },

        "learning_rate": {
            "type": "float",
            "low": 0.01,
            "high": 0.2
        },

        "max_depth": {
            "type": "int",
            "low": 3,
            "high": 10
        },

        "num_leaves": {
            "type": "int",
            "low": 20,
            "high": 100
        },

        "subsample": {
            "type": "float",
            "low": 0.6,
            "high": 1.0
        },

        "colsample_bytree": {
            "type": "float",
            "low": 0.6,
            "high": 1.0
        },

        "reg_alpha": {
            "type": "float",
            "low": 0,
            "high": 5
        },

        "reg_lambda": {
            "type": "float",
            "low": 0,
            "high": 10
        }
    }
}


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

# Directory where trained models are stored
MODELS_DIR = BASE_DIR / "models"

# Raw data
DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "dataset.csv"

# Directory where the pipeline writes its JSON artifacts (the API reads these)
ARTIFACTS_DIR = PROJECT_ROOT / "backend" / "artifacts"

# India state boundaries consumed by the interactive map
GEOJSON_PATH = PROJECT_ROOT / "frontend" / "data" / "in.json"

# Property inside each GeoJSON feature holding the state name / identifier
GEOJSON_NAME_PROPERTY = "name"
GEOJSON_ID_PROPERTY = "id"


# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------

TARGET_COLUMN = "Is_Fraud"

DATE_COLUMN = "Transaction_Date"
TIME_COLUMN = "Transaction_Time"

DATE_FORMAT = "%d-%m-%Y"
TIME_FORMAT = "%H:%M:%S"

# Columns that identify a person or a single transaction. These are never
# model features: they leak identity, not behaviour, and have no support at
# prediction time for an unseen customer.
IDENTIFIER_COLUMNS = [
    "Customer_ID",
    "Customer_Name",
    "Transaction_ID",
    "Merchant_ID",
    "Customer_Contact",
    "Customer_Email",
    "Transaction_Description",
]

# Feature set defined for the project (see data/data_notebook copy.ipynb),
# plus State, which the geospatial view requires.
SELECTED_FEATURES = [
    "State",
    "Account_Balance",
    "Transaction_Amount",
    "Merchant_Category",
    "Hour",
    "Age",
    "DayOfWeek",
    "Account_Type",
]

# Categorical members of SELECTED_FEATURES. Only these are one-hot encoded;
# high-cardinality columns (City, Bank_Branch, Merchant_ID) are deliberately
# excluded to keep the design matrix small.
CATEGORICAL_FEATURES = [
    "State",
    "Merchant_Category",
    "Account_Type",
]

# Behavioural features derived in FeatureEngineer. All of them are computable
# from a single transaction plus statistics fitted on the training window.
ENGINEERED_FEATURES = [
    "Month",
    "Is_Weekend",
    "Is_Night",
    "Amount_To_Balance_Ratio",
    "Log_Transaction_Amount",
    "Log_Account_Balance",
    "Balance_After_Transaction",
    "Amount_Dev_State",
    "Amount_Dev_Merchant_Category",
    "Amount_Dev_Hour",
    "State_Frequency",
    "Merchant_Category_Frequency",
    "Device_Type_Frequency",
    "Transaction_Device_Frequency",
    "Hour_Frequency",
]

# Columns used as grouping keys when fitting behavioural statistics on train.
DEVIATION_GROUPS = ["State", "Merchant_Category", "Hour"]
FREQUENCY_COLUMNS = [
    "State",
    "Merchant_Category",
    "Device_Type",
    "Transaction_Device",
    "Hour",
]


# ---------------------------------------------------------------------------
# Evaluation protocol
# ---------------------------------------------------------------------------

RANDOM_STATE = 42

# The dataset spans 2025-01-01 .. 2025-01-31 of banking transactions, so the
# honest protocol is temporal: train on the past, test on the future.
#   train      : first 24 days   (of which the last 5 are the validation slice)
#   validation : days 20..24     (threshold selection only)
#   test       : last 7 days     (touched once, at the very end)
SPLIT_STRATEGY = "temporal"
TEST_DAYS = 7
VALIDATION_DAYS = 5

# Fallback only, if the date column ever stops supporting a temporal split.
TEST_SIZE = 0.2

# Default model for the served fraud detector.
DEFAULT_MODEL = "xgboost"

# Rows sampled for SHAP. TreeExplainer on 200k rows is neither necessary nor
# cheap; the sample is drawn with RANDOM_STATE so the explanation reproduces.
SHAP_SAMPLE_SIZE = 2000


# ---------------------------------------------------------------------------
# Analytics guards
#
# Peak fraud hour / day are ranked by RATE, not by count. A bucket with three
# transactions and one fraud has a 33% rate and means nothing, so any bucket
# below the minimum support is excluded from the ranking.
# ---------------------------------------------------------------------------

MIN_TRANSACTIONS_PER_HOUR_BUCKET = 30
MIN_TRANSACTIONS_PER_DAY_BUCKET = 50
MIN_TRANSACTIONS_PER_STATE = 30


# ---------------------------------------------------------------------------
# State name reconciliation
#
# The dataset and the provided GeoJSON disagree on six names. The mapping is
# explicit: dataset spelling -> GeoJSON `properties.name`. Nothing is dropped
# silently; StateGeoMapper reports every unmatched name on both sides.
# ---------------------------------------------------------------------------

STATE_NAME_MAP = {
    "Andaman and Nicobar Islands": "Andaman and Nicobar",
    "Dadra and Nagar Haveli and Daman and Diu": "D\u0101dra and Nagar Haveli and Dam\u0101n and Diu",
    "Odisha": "Orissa",
    "Uttarakhand": "Uttaranchal",
}

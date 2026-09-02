# Machine Learning Approach

## Problem framing

Unsupervised **anomaly detection**. There is no fraud label, so no classifier
is trained. Each transaction gets an `anomaly_score` in `[0, 1]` — how
unusual it looks relative to the training window — and a binary `is_anomaly`
flag from a fixed operating threshold.

## Models (`backend/src/modeling/anomaly_model.py`)

All scikit-learn, no extra dependencies. `AnomalyModel(model_name)` wraps:

| `model_name` | estimator | raw score |
|---|---|---|
| `isolation_forest` (default) | `IsolationForest` | `-score_samples(X)` |
| `lof` | `LocalOutlierFactor(novelty=True)` | `-score_samples(Xs)` |
| `autoencoder` | `MLPRegressor` reconstructing X→X | per-row reconstruction MSE |

`lof` and `autoencoder` standardize features with a `StandardScaler` held
inside the wrapper; `IsolationForest` is scale-invariant and skips it.

On `fit` the wrapper learns a min-max normaliser mapping the raw score onto
`[0, 1]`, sets `threshold_` to the `1 - CONTAMINATION` quantile of the train
anomaly scores, and stores the sorted train scores for percentile lookup.

## Feature set (`feature_engineering.py`)

Ratios/logs of amount, balance and duration; `Is_Night`; per-context amount
z-scores (`Amount_Dev_Account/USState/Hour/MerchantID`),
`Duration_Dev_Channel`; frequency encodings of state / channel / merchant /
device / hour / occupation; per-account `Account_Txn_Count` and
`Days_Since_Prev_Account_Txn`. Every fitted statistic is learned on train
only; unseen keys fall back to a global value. 60 encoded columns.

## Reproducibility

`random_state=42` on every estimator that accepts it. The temporal split is
deterministic. Re-run: `python -m scripts.train_pipeline --model isolation_forest`.

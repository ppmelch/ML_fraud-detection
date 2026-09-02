# Migration: Supervised India fraud → Unsupervised US anomaly detection

## Context

The project was built around `data/dataset.csv` — 200k labelled Indian banking
transactions with an `Is_Fraud` target and a `State` column mapped to an India
GeoJSON. The new working dataset is **`data/bank_transactions_data_2.csv`**:

- 2,512 rows, 495 accounts (`AccountID` repeats ~5×/account), no nulls
- **No fraud label** — this is the Kaggle "Bank Transaction Dataset for Fraud
  Detection", shipped unlabelled for anomaly detection
- `Location` = 43 **US city** names (not Indian states)
- One combined `TransactionDate` datetime (`%Y-%m-%d %H:%M:%S`), span 2023‑01 → 2024‑01
- `PreviousTransactionDate` is a data-extraction artefact (every value inside a
  6‑minute window on 2024‑11‑04) — **not usable as a feature**

Decisions taken:

1. **Unsupervised anomaly detection.** Models: `IsolationForest` (default),
   `LocalOutlierFactor`, and an MLP **autoencoder** (`sklearn.MLPRegressor`,
   reconstruction error). All sklearn — no new dependencies.
2. **US state choropleth.** Map city → US state, swap the GeoJSON.
3. **Back up the previous code** before rewriting (see step 0).
4. Do backend + frontend + tests in one migration.

Outcome: the same architecture (pipeline → JSON artifacts → FastAPI → static
dashboard, plus a saved model bundle for live scoring) serving an anomaly score
in `[0,1]` per transaction instead of a fraud probability, with a US map.

---

## Step 0 — Back up the current code

Snapshot the pre-migration source into `legacy/pre-unsupervised/` (committed):

```
legacy/pre-unsupervised/
  backend/src/…        (full copy)
  scripts/…
  tests/…
  frontend/index.html, frontend/css/, frontend/js/
  README.md            (one paragraph: what this is, git tag, date 2026-09-01)
```

Also tag: `git tag pre-unsupervised-migration-2026-09-01`.
Large data/GeoJSON files are **not** copied (in git history + too big).

---

## Backend

### `backend/src/modeling/config.py` — schema rewrite

Remove: `TARGET_COLUMN`, `DATE_COLUMN`, `TIME_COLUMN`, `DATE_FORMAT`,
`TIME_FORMAT`, `STATE_NAME_MAP` (India), the supervised `MODEL_CONFIG` /
`OPTUNA_SEARCH_SPACE` bodies.

Add / change:

| Constant | Value |
|---|---|
| `DATASET_PATH` | `DATA_DIR / "bank_transactions_data_2.csv"` |
| `TIMESTAMP_COLUMN` / `TIMESTAMP_FORMAT` | `"TransactionDate"` / `"%Y-%m-%d %H:%M:%S"` |
| `IDENTIFIER_COLUMNS` | `TransactionID, AccountID, DeviceID, IP Address, MerchantID` |
| `RAW_FEATURES` | `TransactionAmount, TransactionType, USState, Channel, CustomerAge, CustomerOccupation, TransactionDuration, LoginAttempts, AccountBalance` |
| `CATEGORICAL_FEATURES` | `TransactionType, USState, Channel, CustomerOccupation` |
| `ENGINEERED_FEATURES` | `Hour, DayOfWeek, Month, Is_Weekend, Is_Night, Amount_To_Balance_Ratio, Balance_After_Transaction, Log_Transaction_Amount, Log_Account_Balance, Log_Transaction_Duration, Amount_Dev_Account, Amount_Dev_USState, Amount_Dev_Hour, Amount_Dev_MerchantID, Duration_Dev_Channel, USState_Frequency, Channel_Frequency, MerchantID_Frequency, DeviceID_Frequency, Hour_Frequency, Occupation_Frequency, Account_Txn_Count, Days_Since_Prev_Account_Txn` |
| `DEVIATION_GROUPS` | `AccountID, USState, Hour, MerchantID` |
| `FREQUENCY_COLUMNS` | `USState, Channel, MerchantID, DeviceID, Hour, CustomerOccupation` |
| `SPLIT_STRATEGY` / `TEST_DAYS` / `VALIDATION_DAYS` | `"temporal"` / `60` / `30` |
| `CONTAMINATION` | `0.02` (anomaly fraction → threshold + IForest/LOF param) |
| `DEFAULT_MODEL` | `"isolation_forest"` |
| `MODEL_CONFIG` | `isolation_forest`: `{n_estimators:300, max_samples:"auto", contamination:0.02, random_state:42, n_jobs:-1}`; `lof`: `{n_neighbors:20, contamination:0.02, novelty:True}`; `autoencoder`: `{hidden_layer_sizes:(32,8,32), activation:"relu", max_iter:400, early_stopping:True, random_state:42}` |
| `SHAP_SAMPLE_SIZE` | `1000` |
| `MIN_TRANSACTIONS_PER_*` | hour `20`, day `20`, state `15` (dataset is small) |
| `CITY_TO_STATE` | dict, 43 entries (see appendix) |
| `GEOJSON_PATH` | `PROJECT_ROOT / "frontend" / "data" / "us-states.json"` |
| `GEOJSON_NAME_PROPERTY` / `GEOJSON_ID_PROPERTY` | `"name"` / `"id"` |

### `backend/src/data/data_loader.py`

- `load()`: drop the `Is_Fraud` presence + binary checks. Drop `"Unnamed: 0"` still.
- Parse `TIMESTAMP_COLUMN` with `TIMESTAMP_FORMAT` → `Transaction_Timestamp`;
  derive `Hour, DayOfWeek, Month, Is_Weekend` (reuse existing logic, single column).
- Map `Location` → `USState` via `CITY_TO_STATE`; unknown city → `ValueError`
  listing the offenders (same pattern as the old date guard).
- `add_temporal_features()` stays the public hook used by the predict path.

### `backend/src/data/data_preparation.py`

- Remove `_target`, `transform_xy`, every `TARGET_COLUMN` reference.
- `fit_transform(data) -> X` (features only). `transform` unchanged (reindex to
  `feature_columns_`, fill 0.0). `_encode` uses `RAW_FEATURES`.

### `backend/src/data/data_splitter.py`

- `temporal_split` unchanged (already label-free).
- `describe_temporal_split`: drop `fraud_cases` / `fraud_rate`; keep
  `{rows, start, end}` per window.
- `split()` (random): drop `stratify=y`; keep as a non-temporal fallback.

### `backend/src/data/feature_engineering.py`

`AMOUNT_COLUMN="TransactionAmount"`, `BALANCE_COLUMN="AccountBalance"`,
`DURATION_COLUMN="TransactionDuration"`.

- Ratios / logs: `Amount_To_Balance_Ratio`, `Balance_After_Transaction`,
  `Log_Transaction_Amount`, `Log_Account_Balance`, `Log_Transaction_Duration`.
- `Is_Night` from `Hour` (<6).
- `Amount_Dev_*` z-scores over `DEVIATION_GROUPS` (mean/std fitted on **train**,
  unseen key → global fallback). `Duration_Dev_Channel` analogous.
- `*_Frequency` maps over `FREQUENCY_COLUMNS` (fitted on train, unseen → 0.0).
- `Account_Txn_Count`: count per `AccountID` within the frame being transformed;
  at predict time (single row, unseen account) → fitted global median.
- `Days_Since_Prev_Account_Txn`: sort by `AccountID` + timestamp, diff in days,
  first txn per account / predict-time fallback → fitted global median.
- Keep `_require_columns` guard.

### `backend/src/modeling/base_model.py`

- `fit(self, X, y=None)` (rename from `train`, keep `train` as a thin alias for
  back-compat inside tests). `predict(X) -> {0,1}` flags.
  `anomaly_score(X) -> float[0,1]` (new abstract). Drop the mandatory
  `predict_proba`; provide it as an alias of `anomaly_score`.

### `backend/src/modeling/anomaly_model.py` (new; old `classification_model.py` → legacy)

- `AnomalyModel(BaseModel)`, `model_name ∈ {isolation_forest, lof, autoencoder}`.
- Holds an optional `StandardScaler` (used for `lof` + `autoencoder`, not IForest).
- `fit(X)`: fit scaler (if used) + estimator; compute train raw scores; fit a
  min-max normaliser (`score_min_`, `score_max_`); set `threshold_` =
  `quantile(train_anomaly_score, 1 - CONTAMINATION)`.
- `raw_score(X)`: IForest/LOF → `-estimator.score_samples(Xs)`; autoencoder →
  `mean((Xs - model.predict(Xs))**2, axis=1)`.
- `anomaly_score(X)`: min-max normalise `raw_score` to `[0,1]`, clipped.
- `predict(X)`: `anomaly_score(X) >= threshold_`.
- `save_model` / `load_model`: same joblib-of-self pattern as before.

### `backend/src/modeling/contamination_threshold.py` (new; `threshold_optimization.py` → legacy)

- `ContaminationThreshold(train_scores, contamination=CONTAMINATION)`;
  `.resolve() -> float` = `np.quantile(train_scores, 1 - contamination)`.
  Optional `.knee()` (elbow of the sorted-score curve) for reporting only.

### `backend/src/modeling/model_evaluation.py` — unsupervised (old → legacy)

`AnomalyEvaluation`:

- `evaluate(scores, flags, curves=False) -> dict`:
  `n_samples, n_flagged, flagged_rate, score_mean, score_std,
  score_percentiles{p50,p90,p95,p99}, score_histogram{bin_centers,counts}`;
  with `curves=True` add `score_rank_curve{rank[:200], score[:200]}`.
- `top_anomalies(data, scores, n=25) -> list[dict]` — key raw columns + `score`.
- `feature_contrast(feature_frame, flags) -> list[dict]` —
  `{feature, flagged_mean, normal_mean, std_gap}` (standardised mean diff), sorted.
- `heuristic_alignment(data, scores) -> dict` — build a transparent rule score
  (`LoginAttempts>=4`, amount > global p99, high amount/balance ratio, night hour,
  extreme duration; each 0/1, averaged), report
  `{spearman, overlap_at_flagged}`. **Explicitly a sanity check vs. heuristics,
  not ground truth** — say so in the artifact and the docs.

### `backend/src/evaluation/explainability.py` (adapt; old → legacy copy)

- `AnomalyExplainer`. `shap.TreeExplainer` still works for `isolation_forest`.
  `lof` / `autoencoder` → hand-rolled `_permutation_importance(X)`: shuffle each
  column, measure mean |Δ anomaly_score|.
- `direction` values → `"increases_anomaly_score"` / `"decreases_anomaly_score"`.
- Replace `model_generalization` / `generalizes` with
  `score_summary{mean, p95, p99, flagged_rate}` + a static unsupervised `caveat`.
- `explain(X, top_n=20)` — no `test_metrics` arg needed.

### `backend/src/metrics/transaction_analytics.py` (rename of `fraud_analytics.py`; old → legacy)

Consumes `data` that already carries `anomaly_score` + `is_anomaly` columns
(the pipeline joins them before calling).

- `overview()`: `total_transactions, total_accounts, total_amount,
  avg_transaction_amount, avg_account_balance, states_covered,
  period_start, period_end, flagged_transactions, flagged_rate,
  mean_anomaly_score, flagged_amount`.
- `by_hour()`, `by_day_of_week()`, `by_column(col)` for
  `Channel` / `TransactionType` / `CustomerOccupation`:
  `{bucket|category, total_transactions, flagged, anomaly_rate, mean_score}`.
- `amount_distribution(bins=20)`: `{bin_centers, normal_counts, flagged_counts}`.
- `score_distribution(bins=40)`: `{bin_centers, counts}`.
- `summary()` → `anomaly_summary.json`.

### `backend/src/metrics/state_analytics.py` (adapt; old → legacy copy)

Group by `USState`. Per state: `state, total_transactions, flagged,
anomaly_rate, mean_anomaly_score, avg_transaction_amount, avg_account_balance,
peak_anomaly_hour(+_rate,+_transactions), peak_anomaly_day(+_rate,+_transactions),
most_common_flagged_channel, top_flagged_occupation, low_support`.
Then `StateGeoMapper.attach` → `geojson_name, geojson_id, matched`.
Sort by `anomaly_rate` desc. → `state_metrics.json`.

### `backend/src/metrics/geo_mapping.py` (adapt)

- Loads `us-states.json`. Same FeatureCollection / unique-name / attach / validate
  logic. `STATE_NAME_MAP` kept but small — only real mismatches
  (e.g. `"District of Columbia"` ↔ whatever the GeoJSON calls DC).

### `frontend/data/us-states.json` (new asset)

Fetch the canonical Leaflet-tutorial US states GeoJSON
(`https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json`
— `properties.name` per feature, 50 states + DC, ~200 KB). If the fetch is
unavailable, fall back to `us-atlas` states TopoJSON converted to GeoJSON.
Verify every value produced by `CITY_TO_STATE` matches a feature `name`.

### `backend/src/pipeline.py` — `AnomalyPipeline` (old `FraudPipeline` → legacy copy)

`run()`: load → `temporal_split` → `feature_engineer.fit_transform/transform`
→ `data_preparation.fit_transform/transform` (X only) → `AnomalyModel(model_name).fit(X_train)`
→ `anomaly_score` for train/val/test → threshold from train scores
→ `AnomalyEvaluation` per window (test with `curves=True`, `top_anomalies`,
`feature_contrast`, `heuristic_alignment`) → return
`{split, threshold, contamination, metrics, feature_columns, X_test, data_test, test_scores}`.
`predict_score(raw_df)` mirrors the old `predict_proba` path for the API.

### `scripts/train_pipeline.py` — orchestration

1. `AnomalyPipeline(model).run()`; print score summary + threshold.
2. `model_metrics.json` — `{model_name, threshold, contamination, split, metrics, feature_columns, n_features}`.
3. `AnomalyExplainer(pipeline.model).explain(X_test, top_n=25)` → `explainability.json`.
4. full load → `pipeline.model` scores it → `TransactionAnalytics(scored).summary()` → `anomaly_summary.json`.
5. `StateAnalytics(scored).compute()` + `StateGeoMapper` → `state_metrics.json`, `geo_validation.json`.
6. `ModelLoader().save(model, feature_engineer, data_preparation, threshold, metadata)` → `backend/src/models/fraud_model.pkl` (keep filename or rename to `anomaly_model.pkl` + update `ModelLoader.DEFAULT_FILENAME`).

### API

- `schemas.py`: `TransactionInput` = `TransactionAmount, TransactionType,
  Location, Channel, CustomerAge, CustomerOccupation, TransactionDuration,
  LoginAttempts, AccountBalance, TransactionDate` + optional `AccountID`,
  `PreviousTransactionDate`. `PredictionResponse` = `anomaly_score, is_anomaly,
  threshold_used, model_name, percentile, caveat`.
- `artifacts.py`: `get_fraud_summary` → `get_anomaly_summary` (`anomaly_summary.json`).
- `main.py`: `/api/fraud/summary` → `/api/anomaly/summary`; `/api/predict`
  returns the anomaly response, caveat built from unsupervised metrics (no
  label-derived `roc_auc` formatting). Other routes unchanged (new payload shapes).
- `model_service.py`: `predict()` → `{anomaly_score, is_anomaly, threshold_used,
  model_name, percentile}` (`percentile` = rank of this score vs. a stored
  train-score ECDF in the bundle).

---

## Frontend (`frontend/index.html`, `js/app.js`, `css/style.css`)

- **Copy / labels**: title, hero `<h1>`, navbar (`India Map` → `US Map`,
  `#india-map` → `#us-map`), legend `<h4>` → "Anomaly Rate by State", legend
  note, `#map` `aria-label`, footer unchanged. Favicon → a small new
  `frontend/favicon.svg` (US-neutral); drop the broken `India.svg` reference.
- **`app.js` formatting**: `formatCurrency` `₹`→`$`, `en-IN`→`en-US` (3 sites).
- **Overview KPIs** (`overview-kpi-grid`): Total Transactions, Accounts,
  Flagged Transactions, Flagged Rate, Mean Anomaly Score. New ids
  `kpi-total-transactions, kpi-accounts, kpi-flagged, kpi-flagged-rate, kpi-mean-score`.
- **"Anomaly Analysis" section** (was Fraud Analysis): anomaly rate by hour,
  by day, by channel, by transaction type; score distribution histogram; amount
  distribution (flagged vs normal). Reuse the existing `render*` helpers,
  re-pointed at the new JSON keys (`anomaly_rate`, `mean_score`, `flagged`).
- **"Model Diagnostics" section** (was Model Performance): score distribution,
  score-rank curve, **top-25 anomalies table**, feature-contrast bar chart, SHAP
  feature importance. Remove ROC / PR / confusion-matrix / probability-density
  cards + their render functions.
- **Map** (`loadMap`): fetch `data/us-states.json`; colour by
  `record.anomaly_rate`; `computeColorScale` / `getStateColor` unchanged
  mechanically; re-tune `getMapFitPadding` for the US aspect ratio (wider than
  tall); state card fields → `anomaly_rate, mean_anomaly_score, flagged,
  total_transactions, peak_anomaly_hour, peak_anomaly_day,
  most_common_flagged_channel, top_flagged_occupation, avg_transaction_amount,
  avg_account_balance`; search datalist from US state names.
- **CSS**: keep the current palette; only adjust `#state-card` / `#map-legend`
  fixed positions if the US aspect ratio needs it. Rename nothing structural.
- **Delete dead assets**: `frontend/data/in.json`, `Jalisco.json`,
  `risk_data.json`, `dashboard_data.json`, `frontend/India.svg`,
  `frontend/Jalisco.*`.

---

## Tests (`tests/`)

- `conftest.py`: new `sample_transactions` fixture — new schema, no label,
  `AccountID` repeating (~40 accounts × ~6 txns), `Location` from a handful of
  cities across ≥4 states, `TransactionDate` spanning ≥ `TEST_DAYS +
  VALIDATION_DAYS + 1` days.
- Rewrite: `test_data_loader.py` (city→state, timestamp parse, no-label load),
  `test_data_preparation.py` (X-only, no target), `test_data_splitter.py`
  (temporal windows, describe without fraud), `test_feature_engineering.py`
  (new features, train-fitted stats, unseen keys), `test_model_evaluation.py`
  (`AnomalyEvaluation` shapes), `test_model_loader.py` (anomaly bundle),
  `test_pipeline.py` (unsupervised end-to-end), `test_geo_mapping.py` (US),
  `test_state_analytics.py`, `test_fraud_analytics.py` → `test_transaction_analytics.py`,
  `test_api.py` (new endpoints/keys).
- Target: `pytest` green, same "skip if artifact/model missing" guards.

## Docs & agent files

- Update `README.md`, `docs/dataset.md`, `docs/ml-approach.md`,
  `docs/architecture.md`, `docs/agent-architecture.md`, `docs/fraud-metrics.md`
  (→ anomaly metrics), `docs/model-evaluation.md`,
  `docs/threshold-cost-optimization.md` (→ contamination), and the two agent
  briefs `.claude/agents/backend-engineer.md` / `frontend-engineer.md`
  (India / `risk_data.json` references).
- `.gitignore` already covers `*.deb`, `data/*.csv` — the new CSV is ignored by
  `data/*.csv`; add a negation `!data/bank_transactions_data_2.csv` so the
  working dataset is versioned (344 KB, well under limits).

---

## Verification

1. `pytest -q` → green.
2. `python -m scripts.train_pipeline --model isolation_forest` → writes
   `model_metrics.json, explainability.json, anomaly_summary.json,
   state_metrics.json, geo_validation.json` + the model bundle, no NaN errors.
   Repeat for `--model lof` and `--model autoencoder`.
3. `python -m uvicorn backend.src.api.main:app --port 8000` → `GET` every route
   returns 200; `/api/anomaly/summary`, `/api/model/metrics`, `/api/states`,
   `/api/model/explainability` carry the new keys; `POST /api/predict` with the
   schema example returns an `anomaly_score` in `[0,1]`.
4. `cd frontend && python -m http.server 8777` → open `index.html`: US map
   renders all states, choropleth + legend populated, every chart renders, KPIs
   filled, top-anomalies table populated, **no console errors** at 1440 / 900 /
   390 px, no horizontal scroll.
5. `geo_validation.json` → `ok: true`, every `CITY_TO_STATE` value matched.

---

## Addendum — synthetic state coverage augmentation

The real dataset's 43 cities cover only **27 states** — 25 GeoJSON features
(24 states + DC's peer Puerto Rico + …) render grey. To fill the choropleth,
generate statistically-faithful synthetic transactions for the missing states.

### `scripts/augment_dataset.py` (new)

- Load the real `data/bank_transactions_data_2.csv`.
- Missing states = GeoJSON `properties.name` set − `set(CITY_TO_STATE.values())`
  (25: Alabama, Alaska, Arkansas, Connecticut, Delaware, Hawaii, Idaho, Iowa,
  Kansas, Louisiana, Maine, Minnesota, Mississippi, Montana, New Hampshire,
  New Jersey, North Dakota, Puerto Rico, Rhode Island, South Carolina,
  South Dakota, Utah, Vermont, West Virginia, Wyoming).
- One assigned city per missing state (largest city, must not collide with the
  existing 43 or each other — e.g. Maine → **Bangor** not Portland;
  SC → **Columbia**, WV → **Charleston**). Add all 25 to `CITY_TO_STATE`.
- Per state, `n = clip(round(pop_millions * 10), 20, 100)` rows
  (approx 2020 populations, hard-coded in the script).
- Generate each state's rows by **resampling with replacement from the full
  real dataset**, then:
  - `Location` = the state's assigned city
  - `TransactionID` = `TXS%06d`; `AccountID` drawn from a fresh per-state pool
    `ACS%04d` of size `ceil(n/5)` (keeps per-account features meaningful)
  - `TransactionAmount`, `AccountBalance`, `TransactionDuration` ×
    `exp(N(0, 0.15))` jitter, then clipped to the observed global min/max and
    rounded as in the source
  - `TransactionDate` = a resampled real timestamp ± `U(-3, 3)` days, kept
    inside the real span; `PreviousTransactionDate` resampled as-is
  - `TransactionType, Channel, CustomerAge, CustomerOccupation, LoginAttempts,
    DeviceID, IP Address, MerchantID` = resampled unchanged
  - fixed `random_state`
- Concatenate real + synthetic, write `data/bank_transactions_data_2_full.csv`.
  Real rows keep their `TX######` / `AC#####` ids; synthetic rows are
  identifiable by the `TXS` / `ACS` prefix. Schema is byte-identical (no extra
  column). Print the per-state counts and the new total.
- `.gitignore`: add `!data/bank_transactions_data_2_full.csv`.

### Wiring

- `config.DATASET_PATH` → `data/bank_transactions_data_2_full.csv`.
- `config.CITY_TO_STATE` → +25 entries.
- Re-run `python -m scripts.train_pipeline --model isolation_forest` (and lof,
  autoencoder) so every artifact reflects the fuller dataset.
- `geo_validation.json` → `geojson_states_without_data` shrinks to `[]` (or just
  the handful of GeoJSON features with genuinely no assigned city).
- `docs/dataset.md` — document the augmentation, its method, and how to
  regenerate; make clear the synthetic rows exist only for map coverage and are
  flagged by id prefix.
- `tests/` — a `test_augment_dataset.py`: every missing state present after
  augmentation, synthetic ids prefixed, continuous columns within observed
  range, real rows unchanged.

## Appendix — `CITY_TO_STATE` (43 cities)

Albuquerque→New Mexico, Atlanta→Georgia, Austin→Texas, Baltimore→Maryland,
Boston→Massachusetts, Charlotte→North Carolina, Chicago→Illinois,
Colorado Springs→Colorado, Columbus→Ohio, Dallas→Texas, Denver→Colorado,
Detroit→Michigan, El Paso→Texas, Fort Worth→Texas, Fresno→California,
Houston→Texas, Indianapolis→Indiana, Jacksonville→Florida, Kansas City→Missouri,
Las Vegas→Nevada, Los Angeles→California, Louisville→Kentucky, Memphis→Tennessee,
Mesa→Arizona, Miami→Florida, Milwaukee→Wisconsin, Nashville→Tennessee,
New York→New York, Oklahoma City→Oklahoma, Omaha→Nebraska,
Philadelphia→Pennsylvania, Phoenix→Arizona, Portland→Oregon,
Raleigh→North Carolina, Sacramento→California, San Antonio→Texas,
San Diego→California, San Francisco→California, San Jose→California,
Seattle→Washington, Tucson→Arizona, Virginia Beach→Virginia,
Washington→District of Columbia

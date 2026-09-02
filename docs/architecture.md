# Architecture

```
data/bank_transactions_data_2.csv
        │
        ▼
backend/src/data/
  data_loader.py        TransactionDataLoader — parse timestamp, derive calendar cols, city→USState
  data_splitter.py      DataSplitter.temporal_split — train / validation / test by time
  feature_engineering.py FeatureEngineer — behavioural features, train-only stats
  data_preparation.py   DataPreparation — select + one-hot encode → X (no target)
        │
        ▼
backend/src/modeling/
  anomaly_model.py         AnomalyModel — IsolationForest | LOF | MLP autoencoder, score→[0,1], threshold
  contamination_threshold.py ContaminationThreshold — 1-contamination quantile of train scores
  model_evaluation.py      AnomalyEvaluation — score-distribution metrics, top anomalies, heuristic alignment
  model_loader.py          ModelLoader — joblib bundle (model, FE, prep, threshold, train scores)
  config.py                every path + hyperparameter + CITY_TO_STATE
backend/src/evaluation/
  explainability.py        AnomalyExplainer — SHAP (iForest) / permutation importance (LOF, AE)
backend/src/metrics/
  transaction_analytics.py TransactionAnalytics — anomaly_summary.json
  state_analytics.py       StateAnalytics — per-USState metrics
  geo_mapping.py           StateGeoMapper — join to frontend/data/us-states.json
        │
        ▼
backend/src/pipeline.py    AnomalyPipeline.run() wires all of the above
scripts/train_pipeline.py  writes backend/artifacts/*.json + backend/src/models/anomaly_model.pkl
        │
        ▼
backend/src/api/           FastAPI — serves the artifacts; POST /api/predict scores live via the bundle
        │
        ▼
frontend/                  static page, fetches the artifacts + us-states.json (owned by frontend-engineer)
```

## Artifacts (`backend/artifacts/`)

`model_metrics.json`, `explainability.json`, `anomaly_summary.json`,
`state_metrics.json`, `geo_validation.json`. The API never recomputes them.

## Routes

`/api/health`, `/api/anomaly/summary`, `/api/model/metrics`, `/api/states`,
`/api/states/validation`, `/api/map/geojson`, `/api/model/explainability`,
`POST /api/predict`.

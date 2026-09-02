# Transaction Anomaly Detection

Unsupervised anomaly detection over ~2,500 unlabelled US bank transactions,
served as a static dashboard with a US-state choropleth.

Pipeline → JSON artifacts → FastAPI → static frontend, plus a saved model
bundle for live scoring. Each transaction gets an `anomaly_score` in `[0, 1]`
(how unusual it looks), not a fraud probability — the dataset has no label.

## Quickstart

```bash
pip install -r backend/requirements.txt

# 1. train + write every artifact and the model bundle
python -m scripts.train_pipeline --model isolation_forest
#    also: --model lof   |   --model autoencoder

# 2. serve the API
python -m uvicorn backend.src.api.main:app --port 8000

# 3. serve the dashboard
cd frontend && python -m http.server 8777
```

## Tests

```bash
python -m pytest tests -q
```

## Layout

- `backend/src/data/` — load, temporal split, feature engineering, encoding
- `backend/src/modeling/` — `AnomalyModel` (IsolationForest / LOF / autoencoder),
  contamination threshold, evaluation, bundle persistence, `config.py`
- `backend/src/evaluation/` — `AnomalyExplainer` (SHAP / permutation importance)
- `backend/src/metrics/` — portfolio and per-state anomaly analytics, GeoJSON join
- `backend/src/pipeline.py` — `AnomalyPipeline.run()`
- `backend/src/api/` — FastAPI serving `backend/artifacts/*.json`
- `scripts/train_pipeline.py` — the one entry point that produces every artifact
- `frontend/` — static page (owned by the frontend engineer)
- `legacy/pre-unsupervised/` — the previous supervised India-fraud codebase,
  frozen (git tag `pre-unsupervised-migration-2026-09-01`)

See `docs/` for the dataset, ML approach, metrics, evaluation protocol,
contamination threshold and architecture.

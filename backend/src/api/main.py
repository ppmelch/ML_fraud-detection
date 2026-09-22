"""
Anomaly-detection API.

Serves the artifacts written by ``scripts/train_pipeline.py`` and exposes a
live scoring endpoint on top of the persisted model bundle. Run with::

    uvicorn backend.src.api.main:app --reload --port 8000

Every route below reads a precomputed JSON artifact except ``/api/predict``,
which is the only one that touches the model at request time. This keeps
every "give me a metric" request O(1) regardless of dataset size, and keeps
the model artifact the single place scoring logic lives.

The dataset is unlabelled: nothing here reports precision, recall or ROC-AUC
against a fraud target, because there is none.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.src.api import artifacts
from backend.src.api.model_service import get_prediction_service
from backend.src.api.schemas import PredictionResponse, TransactionInput

#: Stated on every live score; the model does not predict fraud.
PREDICT_CAVEAT = (
    "This dataset has no fraud label. 'anomaly_score' measures how unusual "
    "the transaction is relative to the training window, not the probability "
    "of fraud. 'percentile' is the score's rank against the training "
    "distribution. Treat a flag as a prompt for review, not a verdict."
)

app = FastAPI(
    title="Transaction Anomaly Detection API",
    description=(
        "Serves anomaly analytics, model diagnostics, per-US-state metrics, "
        "interpretability and live scoring for the unsupervised transaction "
        "anomaly-detection dashboard."
    ),
    version="2.0.0",
)

# The dashboard is a static page fetched from a different origin during
# development (file:// or a dev server). Restrict methods to what the API
# actually exposes rather than opening every verb.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _artifact_or_503(loader, name: str):
    """
    Run an artifact loader and turn a missing file into a clean 503.

    Parameters
    ----------
    loader : Callable[[], dict | list]
        One of the cached getters in :mod:`backend.src.api.artifacts`.
    name : str
        Human-readable artifact name, used in the error detail.

    Returns
    -------
    dict or list
        The loaded artifact.

    Raises
    ------
    HTTPException
        503 when the artifact has not been generated yet.
    """
    try:
        return loader()
    except artifacts.ArtifactNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                f"{name} is not available yet: {error}. "
                f"Run 'python -m scripts.train_pipeline' to generate it."
            ),
        ) from error


@app.get("/api/health", tags=["health"])
def health() -> dict:
    """Liveness probe. Does not require any artifact to exist."""
    return {"status": "ok"}


@app.get("/api/anomaly/summary", tags=["anomaly"])
def anomaly_summary() -> dict:
    """
    Portfolio-wide anomaly metrics: totals, flagged count / rate / amount,
    mean anomaly score, and breakdowns by hour, weekday, channel, transaction
    type and occupation, plus the amount and score distributions.
    """
    return _artifact_or_503(artifacts.get_anomaly_summary, "Anomaly summary")


@app.get("/api/model/metrics", tags=["model"])
def model_metrics() -> dict:
    """
    Model diagnostics: operating threshold, contamination, temporal split
    description and, per window, the anomaly-score distribution (mean, std,
    percentiles, histogram). The test window also carries the score-rank
    curve, the top anomalies, the flagged-vs-normal feature contrast and the
    heuristic-alignment sanity check.
    """
    return _artifact_or_503(artifacts.get_model_metrics, "Model metrics")


@app.get("/api/states", tags=["geo"])
def state_metrics() -> list:
    """
    Per-US-state anomaly metrics: totals, flagged count and rate, mean
    anomaly score, average transaction amount and account balance, peak
    anomaly hour and day (rate-based, support-floored), most common flagged
    channel and top flagged occupation, each joined to its GeoJSON name and
    id.
    """
    return _artifact_or_503(artifacts.get_state_metrics, "State metrics")


@app.get("/api/states/validation", tags=["geo"])
def state_validation() -> dict:
    """
    Health of the dataset-to-GeoJSON state join: matched / unmatched states
    on both sides, duplicates, and the explicit renames applied.
    """
    return _artifact_or_503(artifacts.get_geo_validation, "Geo validation")


@app.get("/api/map/geojson", tags=["geo"])
def map_geojson() -> dict:
    """
    The US state boundaries GeoJSON, unmodified, for clients that cannot
    fetch the static file directly.
    """
    return _artifact_or_503(artifacts.get_geojson, "GeoJSON")


@app.get("/api/model/explainability", tags=["model"])
def model_explainability() -> dict:
    """
    Feature importance and feature impact (direction and magnitude) for the
    trained detector — SHAP for the isolation forest, permutation importance
    for LOF / the autoencoder — with a score summary and an explicit
    unsupervised caveat.
    """
    return _artifact_or_503(artifacts.get_explainability, "Explainability")


@app.get("/api/model/stability", tags=["model"])
def model_stability() -> dict:
    """
    Multi-seed stability of the served detector: per-seed run stats, pairwise
    Jaccard similarity of the flagged sets, per-transaction flag consistency
    and the stable-anomaly rate by state. Returns ``applicable: false`` when
    the served model has no ``random_state`` (e.g. LOF) and was not re-run.
    """
    return _artifact_or_503(artifacts.get_stability_analysis, "Stability analysis")


@app.post("/api/predict", response_model=PredictionResponse, tags=["model"])
def predict(transaction: TransactionInput) -> PredictionResponse:
    """
    Score a single raw transaction with the trained anomaly detector.

    The transaction is run through the same feature engineering and encoding
    fitted during training before scoring, so this endpoint cannot diverge
    from the reported diagnostics.
    """
    try:
        service = get_prediction_service()
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No trained model available: {error}. "
                f"Run 'python -m scripts.train_pipeline' first."
            ),
        ) from error

    result = service.predict(transaction.model_dump())
    result["caveat"] = PREDICT_CAVEAT

    return PredictionResponse(**result)

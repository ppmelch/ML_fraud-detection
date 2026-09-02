"""
Fraud detection API.

Serves the artifacts written by ``scripts/train_pipeline.py`` and exposes a
live prediction endpoint on top of the persisted model bundle. Run with::

    uvicorn backend.src.api.main:app --reload --port 8000

Every route below reads a precomputed JSON artifact except ``/api/predict``,
which is the only one that touches the model at request time. This keeps
every "give me a metric" request O(1) regardless of dataset size, and keeps
the model artifact the single place scoring logic lives.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.src.api import artifacts
from backend.src.api.model_service import get_prediction_service
from backend.src.api.schemas import PredictionResponse, TransactionInput

app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Serves fraud analytics, model performance, state-level India "
        "metrics, interpretability and live scoring for the transaction "
        "fraud detection dashboard."
    ),
    version="1.0.0",
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


# ---------------------------------------------------------------------------
# 1. General fraud metrics
# ---------------------------------------------------------------------------
@app.get("/api/fraud/summary", tags=["fraud"])
def fraud_summary() -> dict:
    """
    Portfolio-wide fraud metrics computed from the observed ``Is_Fraud``
    column: totals, fraud rate, monetary exposure, and breakdowns by hour,
    weekday, merchant category, transaction type, device and account type.
    """
    return _artifact_or_503(artifacts.get_fraud_summary, "Fraud summary")


# ---------------------------------------------------------------------------
# 2. Model performance metrics
# ---------------------------------------------------------------------------
@app.get("/api/model/metrics", tags=["model"])
def model_metrics() -> dict:
    """
    Model performance: operating threshold, temporal split description, and
    ROC-AUC / PR-AUC / precision / recall / F1 / confusion matrix for the
    train, validation and test windows, plus ROC and PR curve points and the
    predicted-probability density for the test window.
    """
    return _artifact_or_503(artifacts.get_model_metrics, "Model metrics")


# ---------------------------------------------------------------------------
# 3. State-level fraud metrics
# ---------------------------------------------------------------------------
@app.get("/api/states", tags=["geo"])
def state_metrics() -> list:
    """
    Per-state fraud metrics: totals, fraud rate, average transaction amount
    and account balance, peak fraud hour and day (rate-based, support-
    floored), most common fraud device and top fraud merchant category, each
    joined to its GeoJSON name and id.
    """
    return _artifact_or_503(artifacts.get_state_metrics, "State metrics")


@app.get("/api/states/validation", tags=["geo"])
def state_validation() -> dict:
    """
    Health of the dataset-to-GeoJSON state join: matched/unmatched states on
    both sides, duplicate states, and the explicit renames applied.
    """
    return _artifact_or_503(artifacts.get_geo_validation, "Geo validation")


# ---------------------------------------------------------------------------
# 4. India map data
# ---------------------------------------------------------------------------
@app.get("/api/map/geojson", tags=["geo"])
def map_geojson() -> dict:
    """
    The India state boundaries GeoJSON provided with the project, unmodified,
    for clients that cannot fetch the static file directly.
    """
    return _artifact_or_503(artifacts.get_geojson, "GeoJSON")


# ---------------------------------------------------------------------------
# 5. Feature importance / SHAP information
# ---------------------------------------------------------------------------
@app.get("/api/model/explainability", tags=["model"])
def model_explainability() -> dict:
    """
    SHAP-based feature importance and feature impact (direction and
    magnitude) for the trained model, alongside the test-set ROC-AUC/PR-AUC
    and an explicit caveat when the model does not generalize.
    """
    return _artifact_or_503(artifacts.get_explainability, "Explainability")


# ---------------------------------------------------------------------------
# 6. Model predictions
# ---------------------------------------------------------------------------
@app.post("/api/predict", response_model=PredictionResponse, tags=["model"])
def predict(transaction: TransactionInput) -> PredictionResponse:
    """
    Score a single raw transaction with the trained model.

    The transaction is run through the same feature engineering and encoding
    fitted during training before scoring, so this endpoint cannot diverge
    from the reported evaluation metrics.
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

    metrics = _artifact_or_503(artifacts.get_model_metrics, "Model metrics")
    test_metrics = metrics["metrics"]["test"]

    result["caveat"] = (
        f"Model test-set performance: roc_auc={test_metrics['roc_auc']}, "
        f"pr_auc={test_metrics['pr_auc']} (baseline="
        f"{test_metrics['pr_auc_baseline']:.4f}). This score should be read "
        f"in that context."
    )

    return PredictionResponse(**result)

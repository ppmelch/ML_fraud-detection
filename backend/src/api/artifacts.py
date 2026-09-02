"""
Cached access to the JSON artifacts written by ``scripts/train_pipeline.py``.

The API never recomputes analytics on request: every route in
:mod:`backend.src.api.main` other than ``/api/predict`` reads one of these
files. Recomputing per request would mean the "general fraud metrics"
endpoint scans 200,000 rows on every page load; reading a 50KB JSON file does
not.
"""

import json
from functools import lru_cache
from pathlib import Path

from backend.src.modeling.config import ARTIFACTS_DIR, GEOJSON_PATH


class ArtifactNotFoundError(RuntimeError):
    """
    Raised when a required artifact has not been generated yet.

    Attributes
    ----------
    path : Path
        The artifact that was expected but is missing.
    """

    def __init__(self, path: Path) -> None:
        """
        Initialize the error.

        Parameters
        ----------
        path : Path
            The missing artifact file.
        """
        self.path = path
        super().__init__(
            f"Artifact not found: {path}. Run "
            f"'python -m scripts.train_pipeline' first to generate it."
        )


def _read_json(path: Path):
    """
    Read and parse a JSON artifact.

    Parameters
    ----------
    path : Path
        File to read.

    Returns
    -------
    dict or list
        Parsed JSON content.

    Raises
    ------
    ArtifactNotFoundError
        If ``path`` does not exist.
    """
    if not path.exists():
        raise ArtifactNotFoundError(path)

    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def get_model_metrics() -> dict:
    """
    Load the model performance artifact.

    Returns
    -------
    dict
        Threshold, split summary and train/validation/test metrics, as
        written by ``scripts/train_pipeline.py``.
    """
    return _read_json(ARTIFACTS_DIR / "model_metrics.json")


@lru_cache(maxsize=1)
def get_explainability() -> dict:
    """
    Load the SHAP feature importance / impact artifact.

    Returns
    -------
    dict
        Feature importance, feature impact and the model-generalization
        caveat.
    """
    return _read_json(ARTIFACTS_DIR / "explainability.json")


@lru_cache(maxsize=1)
def get_anomaly_summary() -> dict:
    """
    Load the anomaly analytics artifact.

    Returns
    -------
    dict
        Portfolio overview plus per-hour / per-weekday / per-channel /
        per-type / per-occupation breakdowns and the amount and score
        distributions, all computed from the model's ``anomaly_score`` and
        ``is_anomaly`` columns.
    """
    return _read_json(ARTIFACTS_DIR / "anomaly_summary.json")


@lru_cache(maxsize=1)
def get_state_metrics() -> list:
    """
    Load the per-state fraud metrics artifact.

    Returns
    -------
    list[dict]
        One record per state, joined to its GeoJSON name and id.
    """
    return _read_json(ARTIFACTS_DIR / "state_metrics.json")


@lru_cache(maxsize=1)
def get_geo_validation() -> dict:
    """
    Load the dataset-to-GeoJSON join health report.

    Returns
    -------
    dict
        Matched / unmatched state counts and the renames applied.
    """
    return _read_json(ARTIFACTS_DIR / "geo_validation.json")


@lru_cache(maxsize=1)
def get_geojson() -> dict:
    """
    Load the US state boundaries GeoJSON.

    Returns
    -------
    dict
        The FeatureCollection as provided with the project, unmodified.
    """
    return _read_json(GEOJSON_PATH)


def clear_cache() -> None:
    """
    Drop every cached artifact.

    Call after re-running the training pipeline within a long-lived process
    (tests, a notebook) so the next read picks up the new files instead of a
    stale in-memory copy.
    """
    for fn in (
        get_model_metrics,
        get_explainability,
        get_anomaly_summary,
        get_state_metrics,
        get_geo_validation,
        get_geojson,
    ):
        fn.cache_clear()

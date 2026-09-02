"""
Persistence for the complete scoring bundle.

A trained estimator on its own cannot score a raw transaction: it needs the
fitted behavioural statistics, the frozen one-hot column layout and the
operating threshold that were chosen with it. Saving them together is what
makes a prediction served by the API identical to the same prediction made
during evaluation.

The bundle also carries the sorted training anomaly scores so a live
prediction can be reported as a percentile against the training distribution.
"""

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from backend.src.modeling.config import MODELS_DIR


class ModelLoader:
    """
    Save and load the anomaly scoring bundle.

    A bundle is a dictionary with the fitted objects and the metadata needed
    to reproduce a run:

    ``model``
        Trained :class:`AnomalyModel` wrapper.
    ``feature_engineer``
        Fitted :class:`FeatureEngineer`.
    ``data_preparation``
        Fitted :class:`DataPreparation` holding the encoded column layout.
    ``threshold``
        Contamination-based operating threshold.
    ``train_scores``
        Sorted training anomaly scores, for percentile lookup at predict time.
    ``metadata``
        Model name, split description and run summary.

    Attributes
    ----------
    models_dir : Path
        Directory the bundles are read from and written to.
    """

    #: Default artifact name for the currently served model.
    DEFAULT_FILENAME = "anomaly_model.pkl"

    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        """
        Initialize the loader.

        Parameters
        ----------
        models_dir : Path, optional
            Directory holding the saved bundles. Defaults to ``MODELS_DIR``.
        """
        self.models_dir = Path(models_dir)

    def save(
        self,
        model,
        feature_engineer,
        data_preparation,
        threshold: float,
        metadata: dict | None = None,
        train_scores=None,
        filename: str = DEFAULT_FILENAME,
    ) -> Path:
        """
        Persist a complete scoring bundle.

        Parameters
        ----------
        model : AnomalyModel
            Trained model wrapper.
        feature_engineer : FeatureEngineer
            Fitted behavioural feature builder.
        data_preparation : DataPreparation
            Fitted encoder holding the column layout.
        threshold : float
            Contamination-based operating threshold.
        metadata : dict, optional
            Run metadata stored alongside the fitted objects.
        train_scores : array-like of float, optional
            Sorted training anomaly scores for percentile lookup. Falls back
            to ``model.train_scores_`` when omitted.
        filename : str, optional
            Artifact file name.

        Returns
        -------
        Path
            Location of the written artifact.
        """
        self.models_dir.mkdir(parents=True, exist_ok=True)

        if train_scores is None:
            train_scores = getattr(model, "train_scores_", np.array([]))

        bundle = {
            "model": model,
            "feature_engineer": feature_engineer,
            "data_preparation": data_preparation,
            "threshold": float(threshold),
            "train_scores": np.sort(np.asarray(train_scores, dtype=float)),
            "metadata": metadata or {},
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        path = self.models_dir / filename

        joblib.dump(bundle, path)

        return path

    def load(self, filename: str = DEFAULT_FILENAME) -> dict:
        """
        Load a previously saved bundle.

        Parameters
        ----------
        filename : str, optional
            Artifact file name.

        Returns
        -------
        dict
            The bundle as written by :meth:`save`.

        Raises
        ------
        FileNotFoundError
            If the artifact does not exist.
        ValueError
            If the payload is missing a required key.
        """
        path = self.models_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Model bundle not found: {path}. Run the training pipeline "
                f"first (python -m scripts.train_pipeline)."
            )

        bundle = joblib.load(path)

        required = {"model", "feature_engineer", "data_preparation", "threshold"}
        missing = required - set(bundle)

        if missing:
            raise ValueError(f"Model bundle is missing keys: {sorted(missing)}")

        return bundle

    def exists(self, filename: str = DEFAULT_FILENAME) -> bool:
        """
        Report whether a bundle is present on disk.

        Parameters
        ----------
        filename : str, optional
            Artifact file name.

        Returns
        -------
        bool
            True when the artifact exists.
        """
        return (self.models_dir / filename).exists()

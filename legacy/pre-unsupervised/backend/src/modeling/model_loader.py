"""
Persistence for the complete scoring bundle.

A trained estimator on its own cannot score a raw transaction: it needs the
fitted behavioural statistics, the frozen one-hot column layout and the
operating threshold that were chosen with it. Saving them together is what
makes a prediction served by the API identical to the same prediction made
during evaluation.
"""

from datetime import datetime, timezone
from pathlib import Path

import joblib

from backend.src.modeling.config import MODELS_DIR


class ModelLoader:
    """
    Save and load the fraud scoring bundle.

    A bundle is a dictionary with the fitted objects and the metadata needed
    to reproduce a run:

    ``model``
        Trained :class:`ClassificationModel` wrapper.
    ``feature_engineer``
        Fitted :class:`FeatureEngineer`.
    ``data_preparation``
        Fitted :class:`DataPreparation` holding the encoded column layout.
    ``threshold``
        Operating threshold chosen on the validation window.
    ``metadata``
        Model name, split description, feature list and metrics.

    Attributes
    ----------
    models_dir : Path
        Directory the bundles are read from and written to.
    """

    #: Default artifact name for the currently served model.
    DEFAULT_FILENAME = "fraud_model.pkl"

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
        filename: str = DEFAULT_FILENAME,
    ) -> Path:
        """
        Persist a complete scoring bundle.

        Parameters
        ----------
        model : ClassificationModel
            Trained model wrapper.
        feature_engineer : FeatureEngineer
            Fitted behavioural feature builder.
        data_preparation : DataPreparation
            Fitted encoder holding the column layout.
        threshold : float
            Operating threshold selected on validation data.
        metadata : dict, optional
            Run metadata stored alongside the fitted objects.
        filename : str, optional
            Artifact file name.

        Returns
        -------
        Path
            Location of the written artifact.
        """
        self.models_dir.mkdir(parents=True, exist_ok=True)

        bundle = {
            "model": model,
            "feature_engineer": feature_engineer,
            "data_preparation": data_preparation,
            "threshold": float(threshold),
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

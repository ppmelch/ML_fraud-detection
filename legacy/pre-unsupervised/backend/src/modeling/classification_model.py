import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from backend.src.modeling.config import MODEL_CONFIG
from backend.src.modeling.base_model import BaseModel


class ClassificationModel(BaseModel):
    """
    Wrapper class for classification models with support for multiple algorithms.

    This class provides a unified interface for training, prediction,
    probability estimation, and model persistence. The model configuration
    is retrieved from a centralized configuration dictionary.

    Supported models:
    - Logistic Regression
    - Random Forest
    - XGBoost
    - LightGBM
    - Additional models can be added by extending the initialization logic and updating the configuration.
    """

    def __init__(self, model_name: str, **kwargs) -> None:
        """
        Initialize the classification model based on the specified model name.

        Parameters
        ----------
        model_name : str
            - Name of the model to initialize. 

            - Supported values are:
            'logistic', 'random_forest', 'xgboost', 'lightgbm'.

        Raises
        ------
        ValueError
            If the provided model_name is not supported.
        """
        self.model_name = model_name
        self.config = MODEL_CONFIG.get(model_name, {}).copy()
        self.config.update(kwargs)

        if model_name == "logistic":
            self.model = LogisticRegression(**self.config)

        elif model_name == "random_forest":
            self.model = RandomForestClassifier(**self.config)

        elif model_name == "xgboost":
            self.model = XGBClassifier(**self.config)

        elif model_name == "lightgbm":
            self.model = LGBMClassifier(**self.config)

        else:
            raise ValueError("Modelo no soportado")

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the classification model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix used for training.
        y : pd.Series
            Target variable (binary classification).
        """
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate class predictions for the input data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix for prediction.

        Returns
        -------
        pd.Series
            Predicted class labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate predicted probabilities for the positive class (PD).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix for prediction.

        Returns
        -------
        pd.Series
            Probability of the positive class (default probability).
        """
        return self.model.predict_proba(X)[:, 1]

    def save_model(self, filename: str, models_dir) -> None:
        """
        Save the trained model to disk using joblib.

        The method stores the model along with metadata such as the model name
        and configuration.

        Parameters
        ----------
        filename : str
            Name of the file to save the model (e.g., 'model.pkl').
        models_dir : Path
            Directory where the model will be stored.
        """
        models_dir.mkdir(parents=True, exist_ok=True)
        path = models_dir / filename

        joblib.dump(self, path)

    def load_model(self, filename: str, models_dir) -> "ClassificationModel":
        """
        Load a previously saved model from disk.

        `save_model` persists the whole wrapper instance, so this restores the
        estimator, its name and its configuration from that instance. A plain
        dictionary payload is also accepted for backwards compatibility.

        Parameters
        ----------
        filename : str
            Name of the file containing the saved model.
        models_dir : Path
            Directory where the model is stored.

        Returns
        -------
        ClassificationModel
            This instance, repopulated from the stored artifact.

        Raises
        ------
        FileNotFoundError
            If the artifact does not exist.
        TypeError
            If the stored payload is neither a wrapper instance nor a dict
            carrying the expected keys.
        """
        path = models_dir / filename

        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")

        data = joblib.load(path)

        if isinstance(data, ClassificationModel):
            self.model = data.model
            self.model_name = data.model_name
            self.config = data.config

        elif isinstance(data, dict):
            self.model = data["model"]
            self.model_name = data["model_name"]
            self.config = data["config"]

        else:
            raise TypeError(
                f"Unsupported model artifact payload: {type(data).__name__}"
            )

        return self
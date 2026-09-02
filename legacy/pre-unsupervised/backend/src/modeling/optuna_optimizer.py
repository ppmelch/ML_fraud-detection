import optuna

from backend.src.modeling.classification_model import ClassificationModel
from backend.src.modeling.config import RANDOM_STATE
from backend.src.modeling.model_evaluation import ModelEvaluation

optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaOptimizer:
    """
    Generic Optuna optimizer for machine learning models.

    This class automates:
    - Hyperparameter sampling
    - Model training
    - Model evaluation
    - MLflow logging
    - Best parameter selection
    """

    def __init__(self, model_name, param_space, X_train, y_train, X_valid, y_valid):
        """
        Initialize the optimizer.

        Parameters
        ----------
        model_name : str
            Name of the machine learning model.

        param_space : dict
            Hyperparameter search space.

        X_train : pd.DataFrame
            Training features.

        y_train : pd.Series
            Training labels.

        X_valid : pd.DataFrame
            Validation features (never the test set).

        y_valid : pd.Series
            Validation labels (never the test set).
        """

        self.model_name = model_name
        self.param_space = param_space

        self.X_train = X_train
        self.y_train = y_train

        self.X_valid = X_valid
        self.y_valid = y_valid

    def _sample_params(self, trial):
        """
        Sample hyperparameters from the Optuna search space.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Optuna trial object.

        Returns
        -------
        dict
            Sampled hyperparameters.
        """

        params = {}

        for param_name, config in self.param_space.items():

            if config["type"] == "int":

                params[param_name] = trial.suggest_int(
                    param_name,
                    config["low"],
                    config["high"]
                )

            elif config["type"] == "float":

                params[param_name] = trial.suggest_float(
                    param_name,
                    config["low"],
                    config["high"]
                )

        return params

    def objective(self, trial):
        """
        Optuna objective function.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Optuna trial object.

        Returns
        -------
        float
            ROC-AUC score for the current trial.
        """

        params = self._sample_params(trial)

        model = ClassificationModel(self.model_name)

        model.model.set_params(**params)

        model.train(
            self.X_train,
            self.y_train
        )

        y_prob = model.predict_proba(
            self.X_valid
        )

        y_pred = (
            y_prob >= 0.5
        ).astype(int)

        evaluator = ModelEvaluation()

        results = evaluator.evaluate(
            self.y_valid,
            y_pred,
            y_prob
        )

        return results["roc_auc"]

    def optimize(self, n_trials=5):
        """
        Execute hyperparameter optimization.

        Parameters
        ----------
        n_trials : int, optional
            Number of Optuna trials.

        Returns
        -------
        optuna.study.Study
            Optuna study object containing optimization results.
        """

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
        )

        study.optimize(
            self.objective,
            n_trials=n_trials
        )

        return study
import mlflow
from backend.src.modeling.config import OPTUNA_SEARCH_SPACE
from backend.src.modeling.optuna_optimizer import OptunaOptimizer


class ModelBenchmark:
    """
    Benchmark multiple machine learning models using
    Optuna hyperparameter optimization and MLflow tracking.

    This class evaluates candidate models under a unified
    experimentation framework and selects the model with
    the highest ROC-AUC performance.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.

    y_train : pd.Series
        Training target labels.

    X_test : pd.DataFrame
        Testing feature matrix.

    y_test : pd.Series
        Testing target labels.
    """

    def __init__(self, X_train, y_train, X_test, y_test):

        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test

    def run(self):
        """
        Execute benchmarking across multiple candidate models.

        The benchmarking process includes:
        - Hyperparameter optimization using Optuna
        - ROC-AUC evaluation
        - MLflow experiment tracking
        - Best model selection

        Returns
        -------
        str
            Name of the best-performing model based on ROC-AUC.
        """

        best_model = None
        best_score = -1

        models = [
            "random_forest",
            "xgboost",
            "lightgbm"
        ]

        mlflow.set_experiment("Credit_Risk_Benchmark")

        for model_name in models:

            with mlflow.start_run(run_name=model_name):

                optimizer = OptunaOptimizer(
                    model_name=model_name,
                    param_space=OPTUNA_SEARCH_SPACE[model_name],
                    X_train=self.X_train,
                    y_train=self.y_train,
                    X_test=self.X_test,
                    y_test=self.y_test
                )

                study = optimizer.optimize(n_trials=30)

                score = study.best_value

                mlflow.log_metric(
                    "best_roc_auc",
                    score
                )

                if score > best_score:

                    best_score = score
                    best_model = model_name

        return best_model
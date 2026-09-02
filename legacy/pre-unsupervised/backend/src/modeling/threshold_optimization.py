import optuna
from sklearn.metrics import f1_score

from backend.src.modeling.config import RANDOM_STATE

optuna.logging.set_verbosity(optuna.logging.WARNING)


class ThresholdOptimizer:
    """
    Class to optimize
    the classification threshold for binary models.
    This class uses Optuna to find the threshold that maximizes
    the F1 score based on true labels and predicted probabilities.
     Parameters
     ----------
     y_true : array-like
         True binary labels (0 or 1).
         y_prob : array-like
         Predicted probabilities for the positive class (values between 0 and 1).
         Methods
         -------
         optimize_threshold(n_trials=100)
             Optimize the classification threshold using Optuna.
             Returns
             -------
             float
                 The optimal threshold value that maximizes the F1 score.
    """

    def __init__(self, y_true, y_prob, low: float = 0.1, high: float = 0.9):

        self.y_true = y_true
        self.y_prob = y_prob

        # A model whose scores all sit below 0.1 would otherwise be forced to
        # an all-negative threshold and score F1 = 0 at every trial.
        self.low = low
        self.high = high

    def optimize_threshold(self, n_trials=100):

        def objective(trial):

            threshold = trial.suggest_float(
                "threshold",
                self.low,
                self.high
            )

            y_pred = (
                self.y_prob >= threshold
            ).astype(int)

            score = f1_score(
                self.y_true,
                y_pred
            )

            return score

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
        )

        study.optimize(objective, n_trials=n_trials)

        return study.best_params["threshold"]
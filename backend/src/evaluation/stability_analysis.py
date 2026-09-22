"""
Multi-seed stability analysis for the anomaly detector.

Isolation Forest (and the autoencoder) are randomized estimators: bootstrap
sampling and random split selection mean that fitting the same configuration
on the same training window under a different seed can flag a different set
of rows. Nothing here changes the reported model or its hyperparameters —
:mod:`scripts.train_pipeline` still trains and evaluates exactly one seed,
the one every other artifact describes. This module answers a separate
question: if we refit that same configuration under several more seeds, how
much does the verdict on any given transaction actually move?

LOF has no ``random_state`` and is deterministic given ``novelty=True``; it
is reported as not applicable rather than silently re-run once.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from backend.src.modeling.anomaly_model import AnomalyModel
from backend.src.modeling.config import (
    CONTAMINATION,
    MIN_TRANSACTIONS_PER_STATE,
    STABILITY_SEEDS,
)
from backend.src.modeling.contamination_threshold import ContaminationThreshold

#: Model names whose estimator accepts ``random_state`` and is therefore
#: randomized between runs. LOF is deterministic and skipped.
RANDOMIZED_MODELS = ("isolation_forest", "autoencoder")


class StabilityAnalyzer:
    """
    Refit an anomaly model under several seeds and measure result stability.

    Attributes
    ----------
    model_name : str
        Estimator under test.
    contamination : float
        Target flagged fraction, held fixed across every run.
    seeds : list[int]
        Random seeds to refit under.
    """

    def __init__(
        self,
        model_name: str,
        contamination: float = CONTAMINATION,
        seeds: list[int] | None = None,
    ) -> None:
        """
        Initialize the analyzer.

        Parameters
        ----------
        model_name : str
            ``isolation_forest``, ``lof`` or ``autoencoder``.
        contamination : float, optional
            Target flagged fraction. Defaults to ``CONTAMINATION``.
        seeds : list[int], optional
            Random seeds to refit under. Defaults to ``STABILITY_SEEDS``.
        """
        self.model_name = model_name
        self.contamination = float(contamination)
        self.seeds = list(seeds) if seeds is not None else list(STABILITY_SEEDS)

    def analyze(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        test_data: pd.DataFrame,
    ) -> dict:
        """
        Refit under every seed, score the shared test window, and summarise.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix, the same one the reported model fit on.
        X_test : pd.DataFrame
            Test feature matrix, the same one the reported model was scored
            on.
        test_data : pd.DataFrame
            Enriched (raw + temporal) test transactions, row-aligned with
            ``X_test``, used to break stability down by US state.

        Returns
        -------
        dict
            ``applicable`` is False (with a ``note``) for a deterministic
            model. Otherwise: ``per_run`` (threshold/flagged rate/score
            stats per seed), ``jaccard`` (pairwise flagged-set similarity),
            ``stability`` (per-transaction flag consistency across runs,
            including the conditional statistics among rows flagged at
            least once — the overall mean is diluted by the majority of
            rows never flagged and is not, by itself, a meaningful summary),
            ``rank_correlation`` (pairwise Spearman correlation of the raw
            score vectors, a threshold-independent stability check) and
            ``stable_anomaly_rate_by_state``.
        """
        if self.model_name not in RANDOMIZED_MODELS:
            return {
                "model_name": self.model_name,
                "applicable": False,
                "note": (
                    f"'{self.model_name}' has no random_state; it is "
                    "deterministic given the same data and was not re-run "
                    "under multiple seeds."
                ),
            }

        n_test = len(X_test)
        run_scores = np.zeros((len(self.seeds), n_test), dtype=float)
        run_flags = np.zeros((len(self.seeds), n_test), dtype=bool)
        per_run = []

        for i, seed in enumerate(self.seeds):
            model = AnomalyModel(
                self.model_name, contamination=self.contamination, random_state=seed
            )
            model.fit(X_train)
            test_scores = model.anomaly_score(X_test)
            threshold = ContaminationThreshold(
                model.train_scores_, self.contamination
            ).resolve()
            flags = test_scores >= threshold

            run_scores[i] = test_scores
            run_flags[i] = flags
            per_run.append(
                {
                    "seed": int(seed),
                    "threshold": float(threshold),
                    "n_flagged": int(flags.sum()),
                    "flagged_rate": float(flags.mean()),
                    "score_mean": float(test_scores.mean()),
                    "score_std": float(test_scores.std(ddof=0)),
                }
            )

        return {
            "model_name": self.model_name,
            "applicable": True,
            "contamination": self.contamination,
            "n_runs": len(self.seeds),
            "n_test_samples": int(n_test),
            "seeds": [int(s) for s in self.seeds],
            "per_run": per_run,
            "jaccard": self._pairwise_jaccard(run_flags),
            "rank_correlation": self._pairwise_spearman(run_scores),
            "stability": self._per_transaction_stability(run_scores, run_flags),
            "stable_anomaly_rate_by_state": self._stable_rate_by_state(
                run_flags, test_data
            ),
        }

    @staticmethod
    def _pairwise_jaccard(run_flags: np.ndarray) -> dict:
        """
        Jaccard similarity of the flagged set between every pair of runs.

        Parameters
        ----------
        run_flags : np.ndarray
            Boolean array, shape ``(n_runs, n_test)``.

        Returns
        -------
        dict
            ``matrix`` (``n_runs`` x ``n_runs``, 1.0 on the diagonal),
            ``pairs`` (the ``n_runs choose 2`` off-diagonal values, for
            plotting the distribution directly), and their ``mean``,
            ``median``, ``std``, ``min`` and ``max``.
        """
        n_runs = run_flags.shape[0]
        matrix = [[1.0 if i == j else 0.0 for j in range(n_runs)] for i in range(n_runs)]
        pairs = []

        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                a, b = run_flags[i], run_flags[j]
                union = int(np.logical_or(a, b).sum())
                sim = float(np.logical_and(a, b).sum() / union) if union > 0 else 1.0
                matrix[i][j] = matrix[j][i] = sim
                pairs.append(sim)

        pairs_arr = np.array(pairs) if pairs else np.array([1.0])

        return {
            "matrix": matrix,
            "pairs": [float(p) for p in pairs],
            "mean": float(pairs_arr.mean()),
            "median": float(np.median(pairs_arr)),
            "std": float(pairs_arr.std(ddof=0)),
            "min": float(pairs_arr.min()),
            "max": float(pairs_arr.max()),
        }

    @staticmethod
    def _pairwise_spearman(run_scores: np.ndarray) -> dict:
        """
        Spearman rank correlation of the raw score vectors between runs.

        Jaccard depends on where each run's own threshold happens to fall,
        so two runs that rank transactions almost identically can still
        score a modest Jaccard if their flagged-set sizes differ slightly.
        Rank correlation sidesteps the threshold entirely and asks the
        question the model actually needs to answer well: does it order
        transactions the same way run to run.

        Parameters
        ----------
        run_scores : np.ndarray
            Anomaly scores, shape ``(n_runs, n_test)``.

        Returns
        -------
        dict
            ``matrix``, ``pairs``, ``mean``, ``median``, ``std``, ``min``
            and ``max``, mirroring :meth:`_pairwise_jaccard`.
        """
        n_runs = run_scores.shape[0]
        matrix = [[1.0 if i == j else 0.0 for j in range(n_runs)] for i in range(n_runs)]
        pairs = []

        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                corr = float(spearmanr(run_scores[i], run_scores[j]).correlation)
                if np.isnan(corr):
                    corr = 0.0
                matrix[i][j] = matrix[j][i] = corr
                pairs.append(corr)

        pairs_arr = np.array(pairs) if pairs else np.array([1.0])

        return {
            "matrix": matrix,
            "pairs": [float(p) for p in pairs],
            "mean": float(pairs_arr.mean()),
            "median": float(np.median(pairs_arr)),
            "std": float(pairs_arr.std(ddof=0)),
            "min": float(pairs_arr.min()),
            "max": float(pairs_arr.max()),
        }

    @staticmethod
    def _per_transaction_stability(
        run_scores: np.ndarray, run_flags: np.ndarray
    ) -> dict:
        """
        How consistently each test transaction is flagged across runs.

        Parameters
        ----------
        run_scores : np.ndarray
            Anomaly scores, shape ``(n_runs, n_test)``.
        run_flags : np.ndarray
            Boolean flags, shape ``(n_runs, n_test)``.

        Returns
        -------
        dict
            ``mean_stability`` (over *all* test rows — diluted by the
            majority never flagged in any run, kept for completeness but
            not the headline number), ``mean_stability_given_flagged_once``
            and ``median_stability_given_flagged_once`` (the informative
            pair: consistency among rows the model considered a candidate
            at least once), the ``n``/``pct`` breakdown at the
            ``==100%``, ``>=80%``, ``>=50%`` and ``>0%`` cutoffs, a 10-bin
            ``histogram`` over all rows, a second ``histogram_given_flagged``
            over only the flagged-at-least-once subset, and the per-row
            arrays ``mean_anomaly_score``, ``score_std_across_runs`` and
            ``stability_score`` (fraction of runs each row was flagged).
        """
        stability_score = run_flags.mean(axis=0)
        mean_score = run_scores.mean(axis=0)
        std_score = run_scores.std(axis=0, ddof=0)
        n = stability_score.size

        ever_flagged = stability_score > 0.0
        conditional = stability_score[ever_flagged]

        def _bucket(edges: np.ndarray, values: np.ndarray) -> dict:
            counts, _ = np.histogram(values, bins=edges)
            centers = (edges[:-1] + edges[1:]) / 2
            return {
                "bin_centers": [float(c) for c in centers],
                "counts": [int(c) for c in counts],
            }

        def _n_pct(mask: np.ndarray) -> dict:
            return {"n": int(mask.sum()), "pct": float(mask.mean())}

        edges = np.linspace(0.0, 1.0, 11)

        return {
            "mean_stability": float(stability_score.mean()),
            "mean_stability_given_flagged_once": (
                float(conditional.mean()) if conditional.size else 0.0
            ),
            "median_stability_given_flagged_once": (
                float(np.median(conditional)) if conditional.size else 0.0
            ),
            "n_test_samples": int(n),
            "flagged_at_least_once": _n_pct(ever_flagged),
            "stability_100pct": _n_pct(stability_score == 1.0),
            "stability_ge_80pct": _n_pct(stability_score >= 0.8),
            "stability_ge_50pct": _n_pct(stability_score >= 0.5),
            "stability_gt_0pct": _n_pct(stability_score > 0.0),
            # kept for backward compatibility with earlier report sections
            "pct_always_flagged": float((stability_score == 1.0).mean()),
            "pct_never_flagged": float((stability_score == 0.0).mean()),
            "histogram": _bucket(edges, stability_score),
            "histogram_given_flagged": _bucket(edges, conditional),
            "mean_anomaly_score": [float(v) for v in mean_score],
            "score_std_across_runs": [float(v) for v in std_score],
            "stability_score": [float(v) for v in stability_score],
        }

    @staticmethod
    def _stable_rate_by_state(run_flags: np.ndarray, test_data: pd.DataFrame) -> list[dict]:
        """
        Fraction of test transactions flagged in every run, per US state.

        A transaction flagged in every seed is the strictest notion of a
        "stable" anomaly; this is a support-floored breakdown of where those
        sit geographically, not a claim that any one state is riskier.

        Parameters
        ----------
        run_flags : np.ndarray
            Boolean flags, shape ``(n_runs, n_test)``.
        test_data : pd.DataFrame
            Enriched test transactions, row-aligned with ``run_flags``
            columns.

        Returns
        -------
        list[dict]
            One record per state: ``state``, ``n_test_transactions``,
            ``stable_flagged``, ``stable_anomaly_rate`` and ``low_support``,
            sorted by descending rate. Empty if ``test_data`` has no
            ``USState`` column.
        """
        if "USState" not in test_data.columns:
            return []

        always_flagged = run_flags.all(axis=0)
        states = test_data["USState"].reset_index(drop=True)

        records = []
        for state, group in states.groupby(states):
            idx = group.index.to_numpy()
            n = len(idx)
            records.append(
                {
                    "state": str(state),
                    "n_test_transactions": int(n),
                    "stable_flagged": int(always_flagged[idx].sum()),
                    "stable_anomaly_rate": float(always_flagged[idx].mean()),
                    "low_support": bool(n < MIN_TRANSACTIONS_PER_STATE),
                }
            )

        records.sort(key=lambda r: -r["stable_anomaly_rate"])
        return records

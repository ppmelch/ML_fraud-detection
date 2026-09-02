"""
Contamination-based operating threshold for the anomaly detector.

There is no label to optimise against, so the cutoff is a policy choice, not
a learned quantity: we declare that a fixed fraction of transactions
(``contamination``) is worth an analyst's attention and take the score
quantile that isolates that fraction. The quantile is computed on the
training window only, then applied unchanged to validation and test.
"""

import numpy as np

from backend.src.modeling.config import CONTAMINATION


class ContaminationThreshold:
    """
    Resolve an anomaly-score threshold from a target contamination rate.

    Attributes
    ----------
    train_scores : np.ndarray
        Anomaly scores of the training window, in ``[0, 1]``.
    contamination : float
        Fraction of transactions to flag.
    """

    def __init__(
        self,
        train_scores,
        contamination: float = CONTAMINATION,
    ) -> None:
        """
        Initialize the resolver.

        Parameters
        ----------
        train_scores : array-like of float
            Anomaly scores of the training rows.
        contamination : float, optional
            Target fraction to flag. Defaults to ``CONTAMINATION``.

        Raises
        ------
        ValueError
            If ``train_scores`` is empty or ``contamination`` is not in
            ``(0, 1)``.
        """
        self.train_scores = np.asarray(train_scores, dtype=float).ravel()

        if self.train_scores.size == 0:
            raise ValueError("train_scores is empty")

        if not 0.0 < contamination < 1.0:
            raise ValueError(
                f"contamination must be in (0, 1), got {contamination}"
            )

        self.contamination = float(contamination)

    def resolve(self) -> float:
        """
        Return the ``1 - contamination`` quantile of the train scores.

        Returns
        -------
        float
            The anomaly-score threshold. A row scoring at or above it is
            flagged.
        """
        return float(
            np.quantile(self.train_scores, 1.0 - self.contamination)
        )

    def knee(self) -> float:
        """
        Elbow of the sorted-score curve, for reporting only.

        The knee is the score whose point on the sorted-score curve is
        farthest from the chord joining the curve's endpoints. It is never
        used as the operating threshold — :meth:`resolve` is — but it gives a
        data-driven reference point next to the policy cutoff.

        Returns
        -------
        float
            The score at the maximum-curvature point.
        """
        ordered = np.sort(self.train_scores)
        n = ordered.size

        if n < 3:
            return float(ordered[-1])

        x = np.arange(n, dtype=float)
        x0, x1 = x[0], x[-1]
        y0, y1 = ordered[0], ordered[-1]

        denom = np.hypot(x1 - x0, y1 - y0)

        if denom == 0:
            return float(ordered[-1])

        distance = np.abs(
            (y1 - y0) * x - (x1 - x0) * ordered + x1 * y0 - y1 * x0
        ) / denom

        return float(ordered[int(np.argmax(distance))])

# Signal Evaluation

How to judge whether a signal is worth carrying into the next phase.

---

## Effect sizes in this domain

A Spearman correlation of 0.05 between a single order-book feature and a
forward return is normal. It sounds negligible by the standards of most
fields and is not.

What matters more than magnitude:

- **Sign consistency.** An effect that flips sign between halves of the sample
  is not an edge, whatever its average magnitude.
- **Cross-fold stability.** A mean of 0.06 over folds scoring 0.05, 0.06, 0.07
  is a different result from the same mean over -0.02, 0.06, 0.14.
- **Mechanism.** A correlation with no economic story is a candidate for
  curve fitting, and should be treated with more suspicion than its p-value
  suggests.

A correlation above 0.30 on this data is almost certainly leakage. Run
`check_leakage.py` before celebrating it.

---

## Accuracy means nothing without a base rate

An accuracy of 0.54 is:

- excellent at a 50% base rate
- worthless at a 54% base rate
- evidence of a bug at a 5% base rate

Every accuracy figure in this project is reported alongside the majority-class
baseline computed on the same rows. `majority_class_baseline` in
`src/utils/metrics.py` exists for this.

The same applies to the metrics that resist imbalance:

| Metric | Answers |
|---|---|
| `balanced_accuracy` | mean per-class recall — immune to majority-class gaming |
| `matthews_corrcoef` | single number robust to imbalance; 0 means chance |
| `auc_pr` | more informative than ROC-AUC when the positive class is rare |
| `positive_rate` | exposes a model that predicts one class almost always |

---

## Ranking quality and calibration are different properties

**AUC** measures whether the model orders examples correctly. It is indifferent
to whether the predicted probabilities are numerically meaningful — adding 0.3
to every probability leaves AUC unchanged.

**Brier score** and the reliability diagram measure calibration: whether a
prediction of 0.7 is right about 70% of the time.

Phase 4 sizes positions from confidence, so calibration matters directly. A
model with good AUC and probabilities clustered between 0.49 and 0.51 ranks
well and gives the backtest nothing to size on.

Check both. Report both.

---

## Confidence-conditioned accuracy decides a design choice

If accuracy rises monotonically across confidence deciles, the strategy can
trade only high-confidence signals and improve its realized hit rate at the
cost of fewer trades. If accuracy is flat across deciles, that option does not
exist and the probabilities are ranking noise.

Either answer is useful. Not asking leaves Phase 4 guessing.

Report sample counts per decile — a decile holding forty rows supports no
conclusion.

---

## Where the errors sit matters more than how many there are

Two models with identical accuracy can have opposite economics:

- wrong mainly on small moves → errors are cheap, the trades barely moved
- wrong mainly on large moves → errors are expensive, and P&L will be worse
  than accuracy suggests

Weight misclassifications by the magnitude of the move they missed:

```python
fp_cost = predictions.loc[false_positives, "fwd_return_actual"].abs().mean()
fn_cost = predictions.loc[false_negatives, "fwd_return_actual"].abs().mean()
```

A model with 55% accuracy that is right on 1 bps moves and wrong on 20 bps
moves has a negative expected P&L. This pattern is invisible in an accuracy
table and is one of the more likely explanations for a disappointing final
number.

---

## The horizon must exceed the latency

The project brief states it directly: a horizon shorter than execution latency
is useless, because the order reaches the market after the move has happened.

Horizons are measured in events and latency in milliseconds, so the check
requires converting between them using measured event density:

```
horizon_ms = horizon_events * median_inter_event_gap_ms
viable = horizon_ms > latency_ms
```

`analyze_horizons.py` does this and reports viability per latency level.

Prefer the **shortest** viable horizon: accuracy generally falls as horizon
grows, because longer horizons are dominated by information the order book
does not contain.

---

## Statistical significance is not economic significance

With 100,000 rows, a correlation of 0.008 is statistically significant and
economically irrelevant — it will not survive half a basis point of spread.

Ask both questions:

1. Is the effect distinguishable from noise? (significance)
2. Is it large enough to survive execution costs? (economic relevance)

Phase 4 answers the second question properly, but a rough check belongs here:
if the signal's implied edge per trade is smaller than the spread, no amount
of significance will save it.

---

## When the answer is no

A signal that fails these checks has produced a legitimate result. The
responses that are **not** acceptable:

- lowering a threshold until something passes
- dropping the fold that disagrees
- switching to the metric that looks best
- reporting the uncorrected p-value
- widening a tolerance because a check failed

The project's stated aim is measuring what survives honest accounting. A
negative finding is that measurement succeeding, not failing.

If OFI does not beat majority class at any horizon, escalate before Phase 2
rather than proceeding — the baseline the whole project compares against would
not exist.

---

## Related

- Metrics implementation: `src/utils/metrics.py` (issue 0020)
- Correlation testing: `scripts/validate_signal.py`
- Horizon analysis: `scripts/analyze_horizons.py`
- Error analysis: issue `docs/issues/phase-2/0033`
- Glossary: `docs/CONTEXT.md`

# Model Evaluation

Which numbers to report, and how to read them.

---

## Accuracy alone will mislead on this problem

Depending on the label threshold from issue 0017, the class distribution may
be well short of balanced. A model predicting the majority class always posts
an accuracy that looks respectable and carries zero information.

Report these together, always:

| Metric | What it catches |
|---|---|
| `accuracy` | the headline, meaningless alone |
| `majority_class_baseline` | what a trivial predictor scores on the same rows |
| `balanced_accuracy` | mean per-class recall — immune to majority-class gaming |
| `matthews_corrcoef` | single number robust to imbalance; 0 is chance |
| `auc_roc` | ranking quality |
| `auc_pr` | more informative than ROC when positives are rare |
| `brier_score` | calibration, which ROC ignores entirely |
| `positive_rate` | exposes a model predicting one class almost always |

`compute_classification_metrics` in `src/utils/metrics.py` returns all of them.
Both models are scored through that one implementation — a comparison where
two models were measured by two different accuracy functions proves nothing.

---

## Ranking and calibration are different properties

**AUC** measures whether the model orders examples correctly. Adding 0.3 to
every predicted probability leaves AUC unchanged.

**Brier score** and the reliability diagram measure whether a prediction of
0.7 is right about 70% of the time.

Phase 4 sizes positions from confidence, so calibration matters directly. A
model with good AUC whose probabilities all sit between 0.49 and 0.51 ranks
well and gives the backtest nothing to size on.

Check both. If probabilities cluster tightly near the base rate, say so
plainly — it constrains what Phase 4 can do, and it is better known early.

---

## Threshold selection

The default 0.5 cut is rarely optimal on imbalanced data.

**The discipline:** find the threshold on the fold's **training** predictions,
then apply it unchanged to test.

```python
train_proba = model.predict_proba(X_train)
threshold, _ = find_optimal_threshold(y_train, train_proba, criterion="f1")
test_pred = (model.predict_proba(X_test) >= threshold).astype(int)
```

Sweeping thresholds on test predictions and reporting the best one is
selecting on the test set. Report both the 0.5 result and the selected-threshold
result, so the effect of the choice is visible.

---

## Reporting across folds

A mean without dispersion hides everything that matters:

- 0.54, 0.55, 0.56 → a stable, modest edge
- 0.40, 0.55, 0.70 → an unstable model whose backtest lands wherever it lands

Report mean **and** standard deviation for every metric, and plot the folds
individually. For Phase 4, fold instability matters more than the mean,
because the strategy will run in one regime, not the average of several.

Also report the weighted mean by test-set size alongside the unweighted one —
they diverge when folds differ in length.

---

## Reading the train/test gap

| Pattern | Reading |
|---|---|
| train 0.85, test 0.51 | memorized; check the parameter budget from issue 0026 |
| train 0.56, test 0.55 | generalizing, modest edge |
| train 0.52, test 0.52 | learned the prior only |
| test > train | usually a leak, or a validation split overlapping training |

A large gap is a finding to report, not a failure to hide. It connects
directly to the parameter-to-sample ratio computed before training.

---

## Where the errors sit

Aggregate accuracy says how often the model is wrong. It does not say whether
being wrong was expensive.

```python
fp_cost = predictions.loc[false_positives, "fwd_return_actual"].abs().mean()
fn_cost = predictions.loc[false_negatives, "fwd_return_actual"].abs().mean()
```

A model right on 1 bps moves and wrong on 20 bps moves has negative expected
P&L despite better-than-chance accuracy. That pattern is invisible in a metrics
table and is among the likelier explanations for a disappointing Phase 4
number.

Issue 0033 covers this in full.

---

## Confidence-conditioned accuracy

Compute accuracy within each confidence decile.

- Rising monotonically → Phase 4 can filter on confidence and improve realized
  hit rate at the cost of fewer trades
- Flat → the probabilities are ranking noise, and filtering buys nothing

Report the sample count per decile. Forty rows in a decile support no
conclusion.

---

## Comparing two models

Use `compare_models.py`. It reports:

- per-fold differences, not just means
- Wilcoxon signed-rank on paired folds
- win/loss counts alongside the p-value
- effect size with a confidence interval
- an explicit power warning below five folds

**It refuses to declare a winner when the mean difference is smaller than the
typical fold-to-fold variation** — which is the honest verdict even when one
model wins most folds, because a consistent tiny edge inside a large noise band
is not a demonstrated difference.

Six folds give six paired observations. A p-value above 0.05 there is not
proof of equivalence, and one below it rests on very little.

---

## Comparing on the same rows

DeepLOB cannot predict the first `sequence_length - 1` rows of each fold; the
baseline can. Those rows sit at fold starts and are not distributed like the
rest of the data.

Head-to-head comparison uses the common index from issue 0031, so both models
are measured on exactly the rows both cover. Each model's own prediction file
keeps its full coverage for Phase 4.

---

## Related

- Metrics implementation: `src/utils/metrics.py` (issue 0020)
- Comparison tool: `scripts/compare_models.py`
- Error analysis: `docs/issues/phase-2/0033`
- Signal judgement: `.claude/skills/quant-researcher/references/signal_evaluation.md`

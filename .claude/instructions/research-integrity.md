# Research Integrity

**Binding on: quant-researcher, ml-engineer, code-reviewer.**

These rules exist once, here. Skills and agents reference them rather than
restating them — the same reuse rule that governs the codebase governs its
documentation.

---

## The rule everything else follows from

**A result that could not have failed is not a result.**

Every check in this project exists to give a finding the opportunity to fail.
Walk-forward folds can show instability. Purge and embargo can remove an edge
that was leakage. Multiple-testing correction can dissolve a correlation.
Shuffled controls can reveal that the pipeline, not the market, produced the
signal.

When a check is inconvenient, that is exactly when it is load-bearing.

---

## Causality

Every feature reads the past. Every target reads the future. Nothing reads
both.

```python
ofi_w = ofi.rolling(w, center=False, min_periods=w).mean()   # past only
fwd   = (mid.shift(-h) - mid) / mid                          # future target
```

- Rolling windows: `center=False`, always
- `min_periods=window`, always — a partial window computed from two rows when
  three were requested shifts the early distribution silently
- Targets: `.shift(-h)`, negative
- Scalers: fitted on training indices only

**Test it, do not assume it.** Append future rows and assert earlier feature
values are unchanged. A centered window or a sign-flipped shift is a
one-character bug with a four-phase blast radius.

---

## No selection on the test set

All of these turn a test set into a validation set:

- sweeping thresholds on test predictions and reporting the best
- choosing a horizon because it tested well
- using the test fold for early stopping
- retraining after seeing test accuracy
- dropping a fold that disagrees
- picking the metric that favours your model after seeing the results

The reported number then estimates the maximum over choices, not the
performance of one choice — and looks entirely normal.

**The discipline:** select on a validation split carved from the training
window, freeze, then touch the test fold exactly once. Record what was tried,
including what lost.

---

## Multiple comparisons

Nine tests at α=0.05 carry a 37% chance of at least one spurious hit.

Correct with Benjamini-Hochberg across the full grid. It controls false
discovery rate rather than family-wise error, which suits an exploratory screen.
Report raw and adjusted p-values; draw conclusions from the adjusted ones.

Prefer permutation p-values to parametric ones. Forward returns are
heavy-tailed, and the parametric null assumes a distribution those tails
violate.

---

## The shuffled control

Permuting the target destroys any real relationship, so shuffled correlations
should follow the null distribution.

**The null is not zero.** Under the null, Spearman rho is approximately
`N(0, 1/(n-1))`, so `E|rho| = sqrt(2 / (pi * (n-1)))` — about 0.025 at n=1000.
A fixed tolerance fails on large samples and passes on small ones. Compare
against the theoretical value, scaled to the sample.

If shuffled results sit well above that, the pipeline manufactures correlation
and nothing else it produced means anything. Stop and find the alignment error.

---

## Report the baseline, always

An accuracy of 0.54 is excellent at a 50% base rate and worthless at 54%.

Every accuracy figure appears alongside the majority-class baseline computed on
the same rows. Report `balanced_accuracy`, `matthews_corrcoef`, and
`positive_rate` too — a model predicting one class always posts a respectable
accuracy and carries zero information.

---

## Report dispersion, always

A mean of 0.55 over folds scoring 0.54, 0.55, 0.56 is a different finding from
the same mean over 0.40, 0.55, 0.70. The second says the model is unstable
across regimes, which matters more for a backtest than the mean, because the
strategy runs in one regime rather than the average of several.

Mean and standard deviation, every time. Plot the folds individually.

---

## Effect sizes in this domain

A Spearman correlation of 0.05 between a single order-book feature and a
forward return is normal, not negligible. A correlation above 0.30 is almost
certainly leakage.

What matters more than magnitude: sign consistency across sample halves,
stability across folds, and a mechanism that explains why the effect should
exist.

**Statistical significance is not economic significance.** With 100,000 rows a
correlation of 0.008 is significant and will not survive half a basis point of
spread. Pair every p-value with an effect size, and every effect size with the
cost it must overcome.

---

## The horizon-latency constraint

The brief states it: a prediction horizon shorter than execution latency is
useless, because the order arrives after the move has happened.

Horizons are in events, latency in milliseconds. Convert using measured event
density before claiming a horizon is viable. Prefer the shortest viable
horizon — accuracy generally falls as horizon grows.

Where a dataset's event gaps dwarf every configured latency, latency is not
measurable on that data. Say so. A flat latency curve is a fact about the data,
never a finding about markets.

---

## Measured versus assumed

Some numbers derive from recorded data. Others are the arithmetic consequence
of a modelling choice that could reasonably have gone another way.

Mark which is which, and propagate bounds where an assumption spans a range. If
the cancellation attribution assumption moves net P&L from -3 bps to +2 bps,
then the range is the result, and reporting either endpoint alone is
misleading.

Name the direction of every simplification's bias:

| Simplification | Direction |
|---|---|
| No market impact of our orders | optimistic |
| No hidden liquidity | pessimistic |
| Fill only on execution at our price | pessimistic |
| Optimistic queue-cancel assumption | optimistic |
| Lookahead VWAP volume profile | optimistic |

A reader who knows the direction can judge whether the net bias flatters the
strategy. A reader who does not, cannot.

---

## When the answer is no

A finding that fails these checks is a legitimate result. The responses that
are not acceptable:

- lowering a threshold until something passes
- dropping the disagreeing fold
- switching to the metric that looks best
- reporting the uncorrected p-value
- widening a tolerance because a check failed

The project's stated aim is measuring what survives honest accounting. A
negative finding is that measurement succeeding.

If a baseline signal does not beat majority-class prediction at any horizon,
escalate before the next phase rather than proceeding — the comparison the
project is built around would not exist.

---

## Related

- Validation structure: `.claude/skills/quant-researcher/references/validation_protocol.md`
- Judging effects: `.claude/skills/quant-researcher/references/signal_evaluation.md`
- Statistical failure modes: `.claude/skills/quant-researcher/references/statistical_pitfalls.md`
- Training discipline: `.claude/skills/ml-engineer/references/training_protocol.md`
- Glossary: `docs/CONTEXT.md`

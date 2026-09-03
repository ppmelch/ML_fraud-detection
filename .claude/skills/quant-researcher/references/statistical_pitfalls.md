# Statistical Pitfalls

The failure modes this project is most exposed to, and what each one looks
like when it happens.

---

## Multiple comparisons

Testing a 3×3 grid of signals against horizons at α=0.05 gives roughly a 37%
chance of at least one "significant" result under a true null.

```
P(at least one false positive) = 1 - (1 - 0.05)^9 ≈ 0.37
```

**Correction used here:** Benjamini-Hochberg, controlling false discovery rate
rather than family-wise error. This suits an exploratory screen — we accept
some false positives in exchange for power, but not a coin flip's worth.

Bonferroni would divide α by 9 and is too conservative for a screen whose
purpose is deciding what to investigate further.

**What it looks like when ignored:** one cell of the grid is significant, gets
carried into Phase 2 as "the configuration", and fails to reproduce.

`validate_signal.py` applies BH across the full grid automatically.

---

## Permutation tests over parametric ones

Forward returns are heavy-tailed. The parametric p-value from `spearmanr`
assumes a null distribution that heavy tails violate.

A permutation test makes no distributional assumption: shuffle the target,
recompute, repeat, and ask where the observed value falls in that distribution.

```python
perm_p = (np.sum(null_abs >= observed_abs) + 1) / (n_trials + 1)
```

The `+1` correction keeps the p-value from being exactly zero, which it can
never legitimately be with finite trials.

---

## The shuffled control catches pipeline bugs, not market effects

Permuting the target destroys any real relationship. The resulting
correlations should follow the null distribution for that sample size.

**Important:** the null is not zero. Under the null, Spearman rho is
approximately `N(0, 1/(n-1))`, so the expected absolute value is:

```
E|rho| = sqrt(2 / (pi * (n - 1)))
```

At n=1000 that is about 0.025. A fixed tolerance of "shuffled correlation must
be below 0.02" would fail on every large sample and pass on every small one.

`validate_signal.py` compares the shuffled mean against this theoretical value
and flags inflation beyond 2.5×. If the shuffled mean sits well above theory,
the measurement code produces correlation independently of the data, and
nothing else in the output is meaningful.

---

## Small fold counts

Six folds give six paired observations. Consequences:

- A Wilcoxon signed-rank test has very limited power
- A mean difference of 1pp with a 3pp standard deviation is not evidence
- One outlying fold moves the mean substantially
- A p-value above 0.05 is not proof of no difference

**What to do:** report the test, report the win count, plot the folds
individually, and state the power limitation explicitly. Do not treat a
non-significant result as proof of equivalence.

---

## Regime dependence masquerading as edge

An effect present in the first half of a session and absent in the second is
describing a regime, not an edge. Phase 2 would train on a relationship that
does not persist.

**Detection:** split-half correlation with a sign-consistency check.
`validate_signal.py` reports both halves for every pair.

**What it looks like when ignored:** strong validation results, weak test
results, and no obvious bug to find.

---

## Selection on the test set

Every one of these turns a test set into a validation set:

- sweeping thresholds on test predictions and reporting the best
- choosing a horizon because it tested well
- retraining with different hyperparameters after seeing test accuracy
- dropping a fold that disagrees
- choosing between metrics after seeing which favours your model

The reported number then estimates the maximum over choices, not the
performance of one choice.

**The discipline:** select on training-fold validation, freeze, then touch the
test fold exactly once. Record what was tried.

---

## Survivorship in parameter selection

Trying twenty feature configurations and reporting the one that worked is
multiple testing without the correction — and usually without acknowledgement,
because the nineteen failures were never written down.

**What to do:** record every configuration tried, in the methodology document.
If twenty were tried, say so; the reader can then discount accordingly.

---

## Statistical versus economic significance

With enough rows, arbitrarily small effects become significant. A correlation
of 0.008 at n=100,000 has a p-value near zero and will not survive half a basis
point of spread.

Always pair the p-value with an effect size, and pair the effect size with the
cost it must overcome.

---

## Optimistic and pessimistic bias, named

Every simplification biases the result in a direction. State it:

| Simplification | Direction |
|---|---|
| No market impact of our orders | **optimistic** — real size moves the price against itself |
| No hidden or iceberg liquidity | pessimistic — some real fills are invisible in the data |
| Fill only on execution at our price | pessimistic — conservative fill trigger |
| Optimistic queue-cancel assumption | **optimistic** — assumes cancels are always ahead of us |
| Perfect-foresight ceiling | not a bias — a labelled upper bound |
| Lookahead VWAP volume profile | **optimistic** — a real algorithm cannot know the profile |

A reader who knows the direction of each can form a view on whether the net
bias flatters the strategy. A reader who does not, cannot.

---

## Precision that the evidence does not support

If the cancellation assumption moves net P&L from -3 bps to +2 bps, then
"the strategy earns 2 bps" and "the strategy loses 3 bps" are both defensible
readings of the same evidence.

Reporting either alone is misleading. The range is the result.

This is why the waterfall carries `is_measured` per stage and propagates
bounds: three of its six stages depend on modelling choices rather than
recorded data.

---

## Related

- Correction implementation: `scripts/validate_signal.py`
- Validation structure: `references/validation_protocol.md`
- Judging effects: `references/signal_evaluation.md`
- Assumption sensitivity: `docs/simulator_methodology.md` (issue 0054)

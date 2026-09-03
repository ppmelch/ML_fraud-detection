---
name: ml-engineer
description: Model training and evaluation engineering for financial time series — reproducible training runs, fold-level discipline, overfitting diagnosis, calibration checking, and paired model comparison with honest statistics. Use when implementing a model, writing or debugging a training loop, auditing a training run that produced surprising results, comparing two models, or checking that a run can be reproduced. This skill governs training mechanics and evaluation; the research question of whether a signal exists belongs to quant-researcher.
---

# ML Engineer

Training and evaluation for PAP-LOB-Trading. Two models — an OFI baseline and
DeepLOB — trained on identical folds so the comparison between them means
something.

## The one rule everything else follows from

**The test fold is touched exactly once, to produce the number that gets
reported.**

Every choice made before that — hyperparameters, early stopping epoch,
decision threshold, feature set — is made on a validation split carved from
the training window. A model tuned against test performance reports the
maximum over choices, not the performance of a choice, and the difference is
invisible in the output.

## Quick Start

```bash
# Verify a training setup is reproducible before spending hours on it
python scripts/check_reproducibility.py --config configs/deeplab.json

# Audit a completed training run for the usual failure modes
python scripts/audit_training_run.py output/deeplab_training_log_h5_fold3.csv

# Compare two models across folds with paired statistics
python scripts/compare_models.py \
    data/backtest/ofi_baseline_metrics_v1.csv \
    data/backtest/deeplab_metrics_v1.csv \
    --metric balanced_accuracy
```

## Core Capabilities

### 1. Reproducibility Check (`check_reproducibility.py`)

Verifies that a training configuration will produce identical results across
runs, before the run is started.

**Checks performed:**
- Seeds set for Python `random`, numpy, and torch
- `torch.backends.cudnn.deterministic` where CUDA is in use
- DataLoader `shuffle` seeded through a generator, not left to global state
- No unseeded `np.random` or `random` calls in the training path
- Config records every parameter that affects the output
- Two short runs with the same seed produce identical weights

**Usage:**
```bash
python scripts/check_reproducibility.py [--config config.json] [--quick]
```

### 2. Training Run Audit (`audit_training_run.py`)

Reads a training log and diagnoses the failure modes that a summary metric
hides.

**Detects:**
- Not learning: loss flat from the first epoch
- Diverging: loss increasing or turning `NaN`
- Overfitting: validation loss rising while training loss falls
- Early stopping that fired too early, or never fired
- Learning rate too high (oscillation) or too low (crawl)
- A model that learned only the base rate — accuracy equals majority class

**Usage:**
```bash
python scripts/audit_training_run.py <training_log.csv> [--baseline-accuracy 0.52]
```

### 3. Paired Model Comparison (`compare_models.py`)

Compares two models fold by fold, with the statistics a six-fold sample
actually supports.

**Features:**
- Per-fold differences, not just aggregate means
- Wilcoxon signed-rank test on paired folds
- Win/loss/tie counts alongside the p-value
- Effect size with a confidence interval
- Explicit power warning when the fold count is small
- Refuses to declare a winner when the difference sits inside fold noise

**Usage:**
```bash
python scripts/compare_models.py <model_a_metrics.csv> <model_b_metrics.csv> \
    --metric balanced_accuracy [--alpha 0.05]
```

## Reference Documentation

### Training Protocol

`references/training_protocol.md` — fold discipline, where the validation
split comes from, why folds never warm-start from each other, class imbalance
handling via `pos_weight`, and the checkpointing rules.

### Model Evaluation

`references/model_evaluation.md` — which metrics to report and why accuracy
alone misleads here, calibration versus ranking, threshold selection on
training predictions, and how to read a train/test gap.

### Reproducibility Checklist

`references/reproducibility_checklist.md` — everything that must be seeded,
recorded, and version-pinned for a result to be reproducible weeks later.

## Shared Instructions

Cross-cutting policy lives in `.claude/instructions/` so it has one home.
These bind this skill:

- [`research-integrity.md`](../../instructions/research-integrity.md) — the no-selection-on-test rule
- [`coding-standards.md`](../../instructions/coding-standards.md) — reproducibility and test quality

Read them rather than relying on the summaries below.

---

## Domain Context

**Read `docs/methodology.md`** once Phase 2 produces it. Labels, horizons,
sampling unit, and metrics are fixed there and are binding.

**Read `docs/deeplab_architecture.md`** before touching the model. Layer
shapes and the parameter budget are decided there, and the parameter-count
test asserts the implementation matches.

## Training Workflow

### 1. Before training

- Parameter count computed and compared against available samples
- Sequence length checked against actual fold sizes from issue 0018
- Reproducibility verified: `python scripts/check_reproducibility.py`
- Validation split carved from the training window, disjoint from test

### 2. During training

- Log every epoch: train loss, validation loss, every metric, learning rate, duration
- Checkpoint on validation improvement, and restore best weights at the end
- Watch for the train/validation gap widening

### 3. After training

```bash
python scripts/audit_training_run.py output/training_log_*.csv
```

Then ask:
- Did it learn, or did it settle on the base rate?
- Did early stopping fire, or did it run to the epoch limit?
- Is the train/test gap consistent with the parameter budget?
- Are the predicted probabilities calibrated enough for Phase 4 to size on?

## Best Practices

### Fold discipline
- Both models load the same persisted fold file; assert the boundaries match
- Each fold trains from fresh weights — never warm-start from the previous fold
- Scaler parameters come from the fold, fitted on training rows only
- Validation for early stopping comes from the training window's tail

### Class imbalance
- Handle it with `pos_weight` in the loss, not by resampling
- Resampling changes the effective dataset and breaks sequence-window structure
- Compute `pos_weight` from the training fold only

### Numerical hygiene
- `BCEWithLogitsLoss` on logits, never `BCELoss` on probabilities
- Sigmoid applied once, in `predict_proba`, never inside `forward`
- `model.eval()` and `torch.no_grad()` for every inference path
- Gradient clipping when loss spikes

### Sanity tests that catch real bugs
- A linearly separable synthetic problem the loop **must** solve
- A random-label problem the loop **must not** solve — if validation accuracy
  climbs on noise, there is a leak
- Batch inference must equal per-row inference within `1e-6`

### What not to do
- Do not select hyperparameters on test-fold performance
- Do not retrain after seeing test results without recording that you did
- Do not report a mean across folds without its standard deviation
- Do not compare models trained on different folds

## Common Commands

```bash
# Full Phase 2 training sequence
python scripts/03_train_ofi_baseline.py --horizons 1 3 5 --seed 42
python scripts/05_train_deeplab.py --horizons 1 3 5 --seed 42
python scripts/04_generate_predictions.py --model both

# Audit every fold's log
for f in output/deeplab_training_log_*.csv; do
    python .claude/skills/ml-engineer/scripts/audit_training_run.py "$f"
done

# Compare
python .claude/skills/ml-engineer/scripts/compare_models.py \
    data/backtest/ofi_baseline_metrics_v1.csv \
    data/backtest/deeplab_metrics_v1.csv --metric balanced_accuracy

# Inside Docker
docker compose run --rm pipeline python scripts/05_train_deeplab.py
```

## Troubleshooting

**Validation accuracy is suspiciously high.** Run the random-label sanity test.
If the loop learns noise, there is leakage — most likely the validation split
overlaps the training rows, or the scaler was fitted on everything.

**Loss goes to NaN.** Usually an exploding gradient. Add clipping, lower the
learning rate, and check for `inf` in the input tensors.

**Model predicts one class always.** Check the base rate and `positive_rate`.
Either `pos_weight` is missing, or the label threshold from issue 0017 produced
a degenerate split.

**Two runs with the same seed differ.** Something is unseeded. Run
`check_reproducibility.py`; the usual culprit is the DataLoader shuffle or a
`np.random` call outside the seeded path.

**Probabilities all cluster near 0.5.** The model ranks but is not calibrated.
Report the Brier score and the reliability diagram, and tell Phase 4 that
confidence-weighted sizing has little to work with.

## Resources

- Training protocol: `references/training_protocol.md`
- Evaluation guidance: `references/model_evaluation.md`
- Reproducibility checklist: `references/reproducibility_checklist.md`
- Architecture: `docs/deeplab_architecture.md`
- Architecture rules: `.claude/Agent.md`
- Related issues: `docs/issues/phase-2/0026` through `0033`

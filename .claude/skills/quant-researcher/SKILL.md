---
name: quant-researcher
description: Signal research and validation methodology for limit order book data. Covers lookahead-bias detection, walk-forward cross-validation with purge and embargo, signal significance testing under multiple comparisons, and prediction-horizon versus execution-latency viability. Use when designing a trading signal, reviewing a validation protocol, interpreting a backtest result that looks too good, choosing a prediction horizon, or deciding whether a measured correlation is real. This skill governs research rigor and statistical reasoning; it does not implement models — pair it with ml-engineer when the deliverable is a trained model.
---

# Quant Researcher

Research methodology for the PAP-LOB-Trading pipeline. The project's stated
question is how much of a model's predictive capacity survives realistic
execution — a question that only means something if the predictive capacity
was measured honestly in the first place.

## The one rule everything else follows from

**A result that cannot be wrong is not a result.** Every validation step in
this project exists to give a finding the opportunity to fail. Walk-forward
folds can show instability. Purge and embargo can remove an edge that was
leakage. Multiple-testing correction can dissolve a correlation. Shuffled
controls can reveal that the pipeline, not the market, produced the signal.

When a check is inconvenient, that is precisely when it is load-bearing.

## Quick Start

```bash
# Detect lookahead bias in a feature/label pair
python scripts/check_leakage.py data/processed/model_features_v1.csv \
    data/processed/model_labels_v1.csv --horizon 5

# Test whether a signal carries information
python scripts/validate_signal.py data/processed/ofi_signal_v1.csv \
    --signal-cols ofi_1 ofi_3 ofi_5 \
    --target-cols price_movement_1 price_movement_3 price_movement_5

# Check horizon viability against execution latency
python scripts/analyze_horizons.py data/processed/lob_reconstructed_v1.csv \
    --horizons 1 3 5 --latencies 0 1 10 100
```

## Core Capabilities

### 1. Leakage Detection (`check_leakage.py`)

Scans a feature matrix and label set for the alignment errors that produce
optimistic results.

**Checks performed:**
- Feature columns correlated with same-row labels far above plausible levels
- Rolling windows that read forward (detected by shifting input and comparing)
- Label columns whose trailing `NaN` count does not match their horizon
- Features whose leading `NaN` count is inconsistent with their stated window
- Perfect or near-perfect predictors, which almost always indicate a leaked target

**Usage:**
```bash
python scripts/check_leakage.py <features.csv> <labels.csv> \
    --horizon 5 [--threshold 0.3] [--verbose]
```

Exits non-zero when any check fails, so it can gate a pipeline.

### 2. Signal Validation (`validate_signal.py`)

Tests whether a candidate signal carries information about future returns,
with the corrections a multi-signal, multi-horizon screen requires.

**Features:**
- Spearman and Pearson correlation with two-sided p-values
- Benjamini-Hochberg correction across the full signal-by-horizon grid
- Split-half stability check, exposing regime dependence
- Shuffled-target control, confirming the test setup produces null results on null data
- Effect sizes reported alongside significance, since large samples make trivial effects significant

**Usage:**
```bash
python scripts/validate_signal.py <signal.csv> \
    --signal-cols ofi_1 ofi_3 \
    --target-cols price_movement_1 price_movement_3 \
    [--alpha 0.05] [--output results.csv]
```

### 3. Horizon Viability Analysis (`analyze_horizons.py`)

Answers the constraint the project brief raises directly: a prediction
horizon shorter than the execution latency is useless, because the order
reaches the market after the move has happened.

**Features:**
- Converts event-time horizons to wall-clock milliseconds using measured event density
- Reports viability per horizon against each configured latency level
- Flags the degenerate case where latency spans zero events on the dataset
- Recommends the shortest viable horizon, since accuracy generally falls as horizon grows

**Usage:**
```bash
python scripts/analyze_horizons.py <lob_reconstructed.csv> \
    --horizons 1 3 5 10 --latencies 0 1 10 100
```

## Reference Documentation

### Validation Protocol

`references/validation_protocol.md` — the walk-forward structure this project
uses, why purge and embargo protect against different leaks, how purge width
relates to label horizon, and what a leak-free fold set must satisfy.

### Signal Evaluation

`references/signal_evaluation.md` — how to judge whether a signal is worth
carrying forward: effect sizes typical of order-book data, why accuracy must
be read against a majority-class baseline, the difference between ranking
quality and calibration, and when a negative result is the finding.

### Statistical Pitfalls

`references/statistical_pitfalls.md` — the failure modes this project is most
exposed to: multiple comparisons across a signal grid, small fold counts and
their effect on statistical power, regime dependence masquerading as edge,
survivorship in parameter selection, and the difference between statistical
and economic significance.

## Shared Instructions

Cross-cutting policy lives in `.claude/instructions/` so it has one home.
These bind this skill:

- [`research-integrity.md`](../../instructions/research-integrity.md) — causality, no-selection-on-test, multiple comparisons, reporting honestly

Read them rather than relying on the summaries below.

---

## Domain Context

**Read `docs/CONTEXT.md` first.** It is the source of truth for OFI, purge,
embargo, walk-forward CV, and prediction horizon. Terminology drift between
research notes and code is a real source of error in this project.

**Read `docs/methodology.md` once Phase 2 produces it.** From that point it is
binding: labels, horizons, sampling unit, metrics, costs, and latency are
fixed there and every later phase inherits them.

## Research Workflow

### 1. Before proposing a signal

- Is it already implemented? `compute_ofi` exists; do not write a second one.
- What is the economic intuition? A signal with no mechanism is curve fitting.
- What would falsify it? Decide the failure criterion before measuring.

### 2. Before trusting a result

```bash
python scripts/check_leakage.py <features> <labels> --horizon <h>
python scripts/validate_signal.py <signal> --signal-cols ... --target-cols ...
```

Then ask:
- Does the shuffled control return null?
- Does the effect survive multiple-testing correction?
- Does it hold in both halves of the sample?
- Is the effect size economically meaningful, not merely significant?

### 3. Before carrying a finding into the next phase

- Is the horizon longer than the latency it will face?
- Is the fold-to-fold dispersion small relative to the effect?
- Has the configuration been recorded in metadata so the result is reproducible?

## Best Practices

### Causality
- Rolling windows trailing, never centered: `.rolling(w, center=False, min_periods=w)`
- Targets explicitly forward-shifted: `.shift(-h)`
- Scalers fitted on training indices only, never on the full sample
- Test causality by appending future rows and asserting earlier values are unchanged

### Effect sizes in this domain
- A Spearman correlation of 0.05 on high-frequency data is normal, not negligible
- Sign consistency and cross-fold stability matter more than magnitude
- An accuracy of 0.54 means nothing until the base rate is stated

### Reporting
- Always report dispersion alongside a mean; a mean over six folds hides everything
- Always report the baseline a metric should be read against
- Report negative results with the same prominence as positive ones

### What not to do
- Do not select a threshold, horizon, or hyperparameter on test-fold performance
- Do not widen a tolerance because a check failed
- Do not drop a fold because it disagrees with the others
- Do not present a point estimate when the underlying assumption spans a range

## Common Commands

```bash
# Full research validation sweep
python scripts/check_leakage.py data/processed/model_features_v1.csv \
    data/processed/model_labels_v1.csv --horizon 5
python scripts/validate_signal.py data/processed/ofi_signal_v1.csv \
    --signal-cols ofi_1 ofi_3 ofi_5 \
    --target-cols price_movement_1 price_movement_3 price_movement_5 \
    --output data/processed/ofi_correlation_results_v1.csv
python scripts/analyze_horizons.py data/processed/lob_reconstructed_v1.csv \
    --horizons 1 3 5 --latencies 0 1 10 100

# Inside Docker
docker compose run --rm pipeline python .claude/skills/quant-researcher/scripts/validate_signal.py ...
```

## Troubleshooting

**A signal shows implausibly high correlation.** Almost always leakage. Run
`check_leakage.py`. Check the shift direction on the target and the `center`
argument on every rolling window.

**A result is significant before correction and not after.** The corrected
answer is the honest one on a multi-hypothesis screen. Report both.

**Folds disagree sharply.** That is a finding about regime dependence, not
noise to average away. Report the dispersion and investigate what differs
between folds.

**The shuffled control shows correlation.** The pipeline is producing signal
where none exists. Stop and find the alignment error before interpreting
anything else.

## Resources

- Validation protocol: `references/validation_protocol.md`
- Signal evaluation: `references/signal_evaluation.md`
- Statistical pitfalls: `references/statistical_pitfalls.md`
- Domain glossary: `docs/CONTEXT.md`
- Architecture rules: `.claude/Agent.md`
- Related issues: `docs/issues/phase-1/0012` through `0015`, `docs/issues/phase-2/0023`

# Model Evaluation

## Temporal protocol

`DataSplitter.temporal_split` carves the timeline into three contiguous,
non-overlapping windows:

```
|------------ train ------------|-- validation (30d) --|--- test (60d) ---|
oldest                                                              newest
```

- **train** — everything before the validation window; fits the feature
  statistics, the encoder column layout and the model.
- **validation** — 30 days; the operating threshold and any tuning decision
  are made here.
- **test** — last 60 days; scored exactly once, at the end.

On this dataset: 1867 / 218 / 427 rows.

## No selection on test

The contamination threshold is the `1 - CONTAMINATION` quantile of the
**train** anomaly scores (`ContaminationThreshold`). Nothing about the test
window feeds back into the model, the features or the cutoff.

## Honesty caveat

With no label, "evaluation" describes the score distribution and the model's
behaviour, not its accuracy. `heuristic_alignment` compares the ranking to a
transparent rule as a smell test. A flagged transaction is a prompt for
review, never a verdict.

## Observed behaviour (isolation_forest, default)

| window | n | flagged | rate | score p99 |
|---|---|---|---|---|
| train | 1867 | 38 | 2.0% | 0.75 |
| validation | 218 | 8 | 3.7% | 0.87 |
| test | 427 | 22 | 5.2% | 0.90 |

The flagged rate drifts up from train to test — recent transactions score
higher — which is expected under temporal evaluation and visible in the
score-rank curve.

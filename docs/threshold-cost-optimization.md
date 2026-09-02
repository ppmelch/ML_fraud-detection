# Contamination Threshold

There is no label and no fraud/friction cost matrix, so the operating cutoff
is a **policy choice**, not a learned quantity.

## Rule

`ContaminationThreshold(train_scores, contamination).resolve()` returns

```
np.quantile(train_scores, 1 - contamination)
```

`CONTAMINATION = 0.02` in `config.py`: we declare that the ~2% most unusual
transactions are worth an analyst's attention. The quantile is taken on the
**train** window and applied unchanged to validation and test.

## Reporting-only knee

`ContaminationThreshold.knee()` returns the maximum-curvature point of the
sorted-score curve — a data-driven reference shown next to the policy cutoff.
It is never used as the operating threshold.

## Live scoring

The saved bundle carries the sorted train anomaly scores. `POST /api/predict`
returns `percentile` — the share of training transactions scoring at or below
the request's score — so a raw `anomaly_score` can be read in context.

## Tuning

Raise `CONTAMINATION` to flag more, lower it to flag less. Re-run
`python -m scripts.train_pipeline` after changing it; every artifact and the
bundle are regenerated from the new value.

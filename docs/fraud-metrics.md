# Anomaly Metrics

There is no label, so there is no precision, recall, ROC-AUC or confusion
matrix against ground truth. What is reported instead
(`backend/src/modeling/model_evaluation.py`, `AnomalyEvaluation`):

## Per window (`train` / `validation` / `test`)

- `n_samples`, `n_flagged`, `flagged_rate`
- `score_mean`, `score_std`
- `score_percentiles` — `p50`, `p90`, `p95`, `p99`
- `score_histogram` — fixed `[0, 1]` range, 40 bins (`bin_centers`, `counts`)

## Test window only

- `score_rank_curve` — first 200 points of the descending sorted-score curve
- `top_anomalies` — the 25 highest-scoring transactions with their key raw columns
- `feature_contrast` — per feature, standardised mean difference between
  flagged and normal rows (`std_gap`), sorted by magnitude
- `heuristic_alignment` — `spearman` and `overlap_at_flagged` between the
  model ranking and a transparent hand rule (many login attempts, amount
  above the portfolio p99, high amount/balance ratio, night hour, extreme
  duration). **This is a sanity check, not ground truth** — the dataset is
  unlabelled, so it says only whether the model ranks transactions roughly
  the way a simple auditable rule would, never whether it is "right".

## Portfolio analytics (`anomaly_summary.json`)

Flagged count / rate / amount, mean anomaly score, and per-hour / per-day /
per-channel / per-type / per-occupation breakdowns
(`total_transactions`, `flagged`, `anomaly_rate`, `mean_score`), plus the
amount distribution (flagged vs normal) and the score distribution.

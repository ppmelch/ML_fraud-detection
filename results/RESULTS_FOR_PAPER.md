# Results for the Paper

Reference package assembled from artifacts already generated in this repository.
No new models, metrics, or experiments were run to produce this document — every
number below is read from `backend/artifacts/*.json`, `results/tables/*.csv` and
`results/figures/*.png`, exactly as they exist on disk. This is a results
reference, not the paper itself: write the paper's prose from this document.

**Framing that applies to every section below:** the dataset
(`data/bank_transactions_data_2_full.csv`) has **no fraud label**. Every model
output is an *anomaly score* — how unusual a transaction is relative to the
training distribution — not a fraud probability or a confirmed fraud count.
Nothing in this package should be described as "detected fraud"; the correct
language is "flagged as anomalous" / "unusual relative to the training
distribution."

---

## 1. Dataset and experimental setup

**Findings**
- Dataset: 3,293 transactions, 654 accounts, 52 US states/territories, period
  2023-01-02 to 2024-01-01.
- Model: Isolation Forest, 85 engineered features, contamination target = **0.02**
  (2%), `n_estimators=300`, `random_state=42` — configuration unchanged
  throughout this analysis; no methodological reason was found to alter it.
- Split: **temporal**, not random — train on the past, tune the cutoff on a
  recent slice, evaluate once on the future.
  - Train: 2,431 rows, 2023-01-02 → 2023-10-03
  - Validation: 302 rows, 2023-10-04 → 2023-11-02
  - Test: 560 rows, 2023-11-03 → 2024-01-01
- Operating threshold: **0.710495**, the 98th percentile of train anomaly
  scores (fixed from train only, then applied unchanged to validation and test).

**Tables:** `results/tables/split_metrics.csv`
**Figures:** none (setup section; figures start at §2)

**Interpretation:** the temporal split is the honest protocol for transaction
data — a random split would leak future behavior into training. The threshold
is a **policy choice** (quantile of train scores), not a value learned against
a label, because none exists.

---

## 2. Anomaly detection results

**Findings**
| Window | n | flagged | rate | score mean | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| Train | 2,431 | 49 | 2.02% | 0.361 | 0.625 | 0.776 |
| Validation | 302 | 7 | 2.32% | 0.424 | 0.639 | 0.855 |
| Test | 560 | 38 | **6.79%** | 0.423 | 0.746 | 0.957 |

**Figures:** `01_score_distribution_threshold.png` (test score distribution
with the 0.710495 cutoff overlaid), `02_split_anomaly_rate.png` (the three
rates above, side by side, with the 2% contamination target as a reference line)
**Tables:** `results/tables/split_metrics.csv`

**Interpretation:** validation rate (2.32%) sits close to the 2% target,
confirming the threshold generalizes reasonably one window forward. The test
rate (6.79%) is the more consequential number — see §3, it is not read as "the
model got worse," it is read as evidence of distribution shift.

---

## 3. Temporal distribution shift

**Findings**
- Test flagged rate (6.79%) is **3.4×** the 2% contamination target and 3.4×
  the train rate (2.02%), despite an unchanged threshold and unchanged model.
- Validation (2.32%) does not show this shift — it appears specifically in the
  final 60-day test window.
- This is a **score-distribution shift between the train and test periods**
  (test-window transactions score higher, on average, against a threshold
  fixed on train history). It is **not** evidence of confirmed increased
  fraud — there is no label to confirm fraud at all — and it is not a bug in
  the threshold logic.

**Figures:** `02_split_anomaly_rate.png` (the visual signature of the shift)
**Tables:** `results/tables/split_metrics.csv`

**Interpretation:** a fixed, train-derived cutoff is a real limitation for
deployment on non-stationary transaction data — this is the paper's strongest
concrete limitation and belongs in both §3 and §8. Frame it precisely as
*"the operating threshold does not transfer perfectly across the observed
9-month gap between train and test; whether that reflects genuine behavioral
drift or a small-sample fluctuation cannot be determined without labels."*

---

## 4. Stability and robustness

**Findings**
- 10 refits of the same configuration under different seeds (seeds 42–51),
  scored on the same test window (n=560).
- **Mean Jaccard similarity** of the flagged set across the 45 seed pairs:
  **0.6886** (median 0.686, std 0.065, range 0.568–0.862).
- **Mean Spearman rank correlation** of the raw scores across the same 45
  pairs: **0.8991** (median 0.901, std 0.014, range 0.869–0.926).
- The gap between these two numbers is itself a finding: the model's
  **ranking** of transactions is much more stable (~0.90) than its **binary
  flag** (~0.69), because Jaccard is sensitive to exactly where each run's own
  threshold happens to cut a continuous score — volatility concentrates at the
  decision boundary, not in the underlying ranking.
- Per-transaction stability (fraction of the 10 runs that flagged a row):
  - 52/560 (9.3%) flagged at least once.
  - Among those 52: median stability = **0.75**, mean = 0.596 — bimodal
    (a "borderline" cluster flagged once or twice, and a larger "robust core"
    cluster flagged 9–10/10 times).
  - **16/560 (2.86%) flagged in all 10 runs** — the strictest, most defensible
    anomaly set the model produces.
  - 26/560 (4.64%) flagged in ≥80% of runs; 31/560 (5.54%) in ≥50%.

**Figures:** `05_jaccard_similarity.png` (heatmap + Jaccard-vs-Spearman
distribution — the headline stability figure), `03_stability_distribution.png`
(appendix: all-rows vs. conditional-on-flagged distributions),
`04_score_vs_stability.png` (appendix: score vs. stability scatter)
**Tables:** `results/tables/stability_per_run.csv`, `results/tables/stability_breakdown.csv`

**Interpretation:** report the 16-transaction "flagged every run" set as the
model's most stable anomaly claim, and report Jaccard/Spearman together —
citing Jaccard alone without the rank-correlation context understates how
consistent the model actually is.

---

## 5. Behavioral interpretation

**Findings — largest standardized gaps, flagged vs. normal (test window)**
| Feature | flagged mean | normal mean | std-gap |
|---|---:|---:|---:|
| TransactionAmount | 877.60 | 272.98 | **1.939** |
| Amount_Dev_Hour | 2.180 | -0.067 | 1.935 |
| Amount_Dev_USState | 2.124 | -0.075 | 1.921 |
| Amount_Dev_MerchantID | 2.717 | -0.051 | 1.874 |
| Amount_To_Balance_Ratio | 1.264 | 0.127 | 1.828 |
| Is_Weekend | 0.342 | 0.038 | 1.290 |
| Log_Transaction_Amount | 6.351 | 5.073 | 1.002 |
| Amount_Dev_Account | 4.941 | 0.592 | 0.987 |

**Figure:** `07_feature_contrast.png` (top 15)
**Table:** `results/tables/feature_contrast_top15.csv`

**Interpretation:** the largest gaps are **all amount-related** — raw
transaction amount and its deviation relative to hour-of-day, state,
merchant and account history — not demographic or device fields. Flagged
transactions are large in absolute terms *and* large relative to the specific
context they occur in. `Is_Weekend` is the largest non-amount gap, suggesting
weekend timing is a secondary, weaker signal. This is a descriptive
association from an unlabeled model, not a causal or fraud claim.

---

## 6. Feature importance and interpretability

**Findings — top SHAP features (mean |SHAP value|, isolation forest)**
| Rank | Feature | Importance |
|---:|---|---:|
| 1 | Month | 0.147 |
| 2 | USState_Texas | 0.143 |
| 3 | TransactionType_Debit | 0.130 |
| 4 | Amount_Dev_Account | 0.113 |
| 5 | USState_California | 0.113 |
| 6 | CustomerOccupation_Student | 0.091 |
| 7 | CustomerOccupation_Retired | 0.085 |
| 8 | CustomerOccupation_Engineer | 0.083 |
| 9 | USState_Frequency | 0.083 |
| 10 | USState_Arizona | 0.081 |

**Figure:** `06_feature_importance.png` (top 15)
**Table:** `results/tables/feature_importance_top15.csv`

**Interpretation:** importance spans five distinct categories — **temporal**
(Month), **geographic** (USState_Texas/California/Arizona, USState_Frequency),
**transaction-type** (TransactionType_Debit), **account-relative behavior**
(Amount_Dev_Account), and **demographic** (CustomerOccupation_*). No single
feature dominates (max importance 0.147), meaning the model's notion of
"anomalous" is genuinely multi-factor rather than reducible to one variable.

**Sanity check, not a result:** `model_metrics.json` also reports agreement
with a transparent hand-built heuristic (large amount, high login attempts,
odd hour, extreme duration): Spearman **0.473**, top-set overlap **9.1%**.
Report this only as a sanity check confirming the model is not wildly at odds
with basic intuition — it is explicitly **not ground truth** and must not be
presented alongside the SHAP results as if it were a second, independent
validation of the same kind.

---

## 7. Geographic analysis

**Findings**
- GeoJSON reconciliation: 52/52 dataset states matched, 0 unmatched, `ok: true`
  (`geo_validation.json`).
- On the **full dataset** (n=3,293), every one of the 52 states clears the
  project's 15-transaction support floor — the full-dataset map is
  well-supported everywhere it shows color.
- Highest anomaly-rate states (support-floored, full dataset):

| State | n | flagged | rate | mean score |
|---|---:|---:|---:|---:|
| Connecticut | 36 | 6 | 16.67% | 0.490 |
| Alaska | 20 | 3 | 15.00% | 0.457 |
| West Virginia | 20 | 3 | 15.00% | 0.500 |
| Kansas | 29 | 4 | 13.79% | 0.531 |
| Minnesota | 57 | 6 | 10.53% | 0.500 |

**Figure:** `09_state_anomaly_rate_map.png`
**Table:** `results/tables/state_anomaly_rate.csv`

**Interpretation:** treat this as a descriptive geographic pattern in a small
dataset (52 states over 3,293 rows; the top state, Connecticut, is 6 flagged
rows out of 36) — not a claim that any state carries elevated fraud risk.

**Excluded from the main package:** a second geographic cut — the
*stable-anomaly rate by state*, restricted to the 560-row **test window**
only — was computed (`stable_anomaly_rate_by_state.csv`,
`10_stable_anomaly_rate_by_state_map.png`) but is **excluded from the paper's
figure set**. Only 7 of 51 states reach the support floor on that smaller
slice, and the most visually striking cells on that map (Kansas 25%,
Minnesota/Maine ~22%) are exactly the low-support, hatched ones — the pattern
would be actively misleading if presented as a main result. See §8.

---

## 8. Limitations

1. **No fraud label exists.** Every result is a statement about statistical
   unusualness relative to the training distribution, never about confirmed
   fraud. This applies to every section above without exception.
2. **Threshold generalization / temporal drift (§3).** A single train-derived
   cutoff does not transfer perfectly to the test window (2.02% → 6.79%
   flagged, 3.4× the target). Whether this reflects real behavioral drift or
   small-sample noise cannot be resolved without labels or a longer test span.
3. **Jaccard understates ranking stability (§4).** Reporting Jaccard alone
   (0.689) without the Spearman rank correlation (0.899) would overstate how
   unstable the model is — the discrepancy is a threshold-boundary artifact,
   not evidence of an unreliable score function.
4. **Small test window for any state-level robustness claim (§7).** The
   full-dataset geographic map (09) is well-supported (52/52 states ≥15
   transactions); the test-window-only stable-anomaly-by-state cut is not
   (7/51 states ≥15) and was excluded from the main figures for this reason.
5. **Heuristic alignment is a sanity check only (§6).** Spearman 0.473 /
   overlap 9.1% against a transparent hand rule confirms the model is not
   arbitrary — it is not a second validation of model correctness and should
   never be quoted alongside SHAP as equivalent evidence.
6. **Sample size throughout is small** (3,293 transactions total, 560 in the
   test window) relative to typical fraud-analytics datasets; effect sizes
   (e.g., std-gaps in §5) should be read as descriptive, not as precision
   estimates with tight confidence.
7. **Model configuration was not tuned in response to any of these
   findings** — contamination (0.02) and all Isolation Forest hyperparameters
   are exactly as they were before this analysis began; nothing here
   constitutes a methodological reason to change them.

---

## Suggested figure order for the paper

**Main text**
1. `01_score_distribution_threshold.png` — establishes the score/threshold mechanism
2. `02_split_anomaly_rate.png` — sets up the temporal-drift finding
3. `05_jaccard_similarity.png` — stability/robustness headline result
4. `06_feature_importance.png` — what drives the score
5. `07_feature_contrast.png` — how flagged transactions actually differ
6. `09_state_anomaly_rate_map.png` — geographic pattern, full dataset

**Appendix**
7. `03_stability_distribution.png` — supports the §4 stability discussion
8. `04_score_vs_stability.png` — supports the §4 stability discussion
9. `08_temporal_patterns.png` — supports a "no strong hour/weekday effect" statement

**Not included** (available in `results/figures/` for reference only):
`10_stable_anomaly_rate_by_state_map.png` — insufficient geographic support (§7, §8).

---

*Source artifacts: `backend/artifacts/model_metrics.json`,
`explainability.json`, `anomaly_summary.json`, `state_metrics.json`,
`geo_validation.json`, `stability_analysis.json`. Regenerate all of the above
via `python -m scripts.train_pipeline` and `python -m scripts.generate_figures`
only if the underlying data or model changes — this document describes the
results as they currently stand and was not itself produced by re-running
either script.*

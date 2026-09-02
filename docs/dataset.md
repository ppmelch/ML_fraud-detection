# Dataset

## Source

`data/bank_transactions_data_2.csv` — Kaggle "Bank Transaction Dataset for
Fraud Detection". 2,512 rows, 495 accounts (`AccountID` repeats ~5x), no
nulls. **Shipped without a fraud label** — this is an anomaly-detection
dataset, not a classification one.

## Working file — `data/bank_transactions_data_2_full.csv`

The pipeline (`config.DATASET_PATH`) reads the **augmented** file, not the raw
one: `data/bank_transactions_data_2_full.csv` = the 2,512 real rows plus 781
synthetic rows, 3,293 total.

**Why.** The 43 real city names resolve to only 27 of the 52 `us-states.json`
polygons; the other 25 states rendered grey on the choropleth. The
augmentation gives every polygon a value.

**Method** (`scripts/augment_dataset.py`, `random_state=42`). For each of the
25 uncovered states:

- One assigned placeholder city (largest/prominent, chosen to collide with
  neither a real city nor another placeholder — e.g. Maine → Bangor,
  Kansas → Wichita, South Carolina → Columbia, West Virginia → Charleston).
  All 25 are added to `config.CITY_TO_STATE`.
- Row count `n = clip(round(pop_millions * 10), 20, 100)` using hard-coded
  approx 2020 populations (New Jersey 93 … Wyoming/Vermont/etc. 20).
- Rows are drawn by **resampling the full real dataset with replacement**,
  then overriding only: `Location` (the assigned city), `TransactionID`
  (`TXS%06d`), `AccountID` (a fresh per-state pool `ACS%04d` of size
  `ceil(n/5)`). `TransactionAmount`, `AccountBalance` and
  `TransactionDuration` get a `× exp(N(0, 0.15))` jitter, clipped to the real
  global min/max and rounded as in the source. `TransactionDate` is a
  resampled real timestamp shifted by `U(-3, 3)` days, kept inside the real
  span. Everything else is resampled unchanged.

The schema is byte-identical to the source (no extra column). Real rows are
copied through untouched; **synthetic rows are identifiable only by their
`TXS` / `ACS` id prefix**.

**Regenerate.**

```
python -m scripts.augment_dataset
python -m scripts.train_pipeline --model isolation_forest   # + lof, autoencoder
```

**Read the analytics with this in mind.** The synthetic rows exist purely for
geographic map coverage — they are statistically faithful to the overall
dataset but not to any real behaviour in those 25 states. Per-state anomaly
metrics for a placeholder city reflect resampled national patterns, not local
ground truth. Filter on the `TXS` / `ACS` prefix to isolate real transactions.

## Schema

| Column | Role |
|---|---|
| `TransactionID`, `AccountID`, `DeviceID`, `IP Address`, `MerchantID` | identifiers — never model features (`AccountID` drives per-account features but is not encoded) |
| `TransactionAmount`, `TransactionType`, `Channel`, `CustomerAge`, `CustomerOccupation`, `TransactionDuration`, `LoginAttempts`, `AccountBalance` | raw model features |
| `Location` | a US city name → resolved to `USState` via `CITY_TO_STATE` (43 real cities + 25 synthetic-coverage placeholders) |
| `TransactionDate` | single datetime, `%Y-%m-%d %H:%M:%S`, spans 2023-01-02 → 2024-01-01 |
| `PreviousTransactionDate` | data-extraction artefact (all values in a 6-minute window on 2024-11-04) — **dropped on load** |

## Known quirks

- `TransactionDate` only ever lands in hours **16, 17, 18** and weekdays
  **Mon–Fri**. Consequences: `Is_Night` and `Is_Weekend` are constant, and
  the by-hour / by-day breakdowns have 3 and 5 buckets respectively. This is
  the data, not a bug.
- Temporal split (last 60 days test, prior 30 days validation) on the
  augmented file yields 2431 / 302 / 560 rows (was 1867 / 218 / 427 on the raw
  2,512-row file).

## Derived columns (`TransactionDataLoader`)

`Transaction_Timestamp`, `Hour`, `DayOfWeek`, `Month`, `Is_Weekend`,
`USState`.

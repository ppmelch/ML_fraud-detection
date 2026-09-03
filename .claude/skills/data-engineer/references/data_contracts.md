# Data Contracts

The schema of every artifact the pipeline produces. This document is the
readable form; `scripts/validate_schema.py` is the executable form. They must
agree — when changing one, change the other in the same commit.

---

## Conventions

- Column names are `snake_case`, self-describing, with units recorded in metadata
- Timestamps are `int64` milliseconds since epoch, never floats or strings
- Prices and quantities are `float64`, strictly positive
- Absent book levels are `NaN`, never `0` or `-1` — those collide with legal values
- Column order is fixed per contract and asserted on write
- Every artifact has a companion `*_metadata.json`

**Critical** columns forbid nulls. Non-critical columns may be null, and the
NaN policy is documented in metadata with per-column counts.

---

## raw_events

Raw order book event stream, as delivered by the exchange or the generator.

| Column | Type | Critical | Constraint |
|---|---|---|---|
| `timestamp` | int64 | yes | >= 0, non-decreasing |
| `order_id` | int64 | yes | >= 0 |
| `side` | str | yes | `B` or `S` |
| `price` | float64 | yes | > 0 |
| `quantity` | float64 | yes | > 0 |
| `event_type` | str | yes | `add`, `cancel`, `execute` |

**Referential rule:** every `cancel` and `execute` references an `order_id`
that appeared in an earlier `add`. No `order_id` is cancelled twice.

**Ordering rule:** timestamps non-decreasing. Equal timestamps are legal —
several events can share a millisecond — and their relative order carries FIFO
queue meaning, so a stable sort is required.

Produced by issue 0001. Validated by 0002.

---

## cleaned_events

Same schema as `raw_events`, after deduplication, stable sort, orphan removal,
and quantity winsorization.

**Additional guarantees:**
- Zero exact duplicate rows
- Zero orphan `cancel` / `execute`
- All quantities inside the winsorization fences
- Re-running the cleaner changes nothing (idempotent)

Produced by issue 0005.

---

## lob_snapshots

Reconstructed order book, one snapshot per event, flat columns to depth 10.

| Column | Type | Critical | Constraint |
|---|---|---|---|
| `timestamp` | int64 | yes | non-decreasing |
| `sequence_number` | int64 | yes | strictly increasing, unique |
| `bid_price_1..10` | float64 | no | descending; `NaN` where absent |
| `bid_volume_1..10` | float64 | no | > 0 or `NaN` |
| `ask_price_1..10` | float64 | no | ascending; `NaN` where absent |
| `ask_volume_1..10` | float64 | no | > 0 or `NaN` |
| `best_bid` | float64 | no | > 0 |
| `best_ask` | float64 | no | > 0 |
| `mid_price` | float64 | no | `(best_bid + best_ask) / 2` |
| `spread` | float64 | no | `best_ask - best_bid`, > 0 |

**Why flat columns.** Encoding levels as a stringified list in one cell is
compact and unusable — every consumer would parse it and pandas would read an
object column. The flat shape is also what the Phase 2 model input expects.

**Why NaN padding.** A book thinner than 10 levels needs a value that cannot
be mistaken for real depth. `0` collides with a legitimately zero volume; `-1`
would survive into a feature matrix and poison a model.

Produced by issue 0011.

---

## ofi_signal

Order flow imbalance with forward-return targets.

| Column | Type | Critical | Constraint |
|---|---|---|---|
| `timestamp` | int64 | yes | |
| `sequence_number` | int64 | yes | unique |
| `ofi_1`, `ofi_3`, `ofi_5` | float64 | no | in `[-1, 1]`; leading `NaN` per window |
| `price_movement_1/3/5` | float64 | no | fractional return; trailing `NaN` = horizon |
| `mid_price`, `spread` | float64 | no | carried through for cost computation |

**NaN policy.** Leading `NaN` from rolling windows and trailing `NaN` from
forward targets are preserved, never filled. Filling would fabricate a
balanced book that never existed, or invent future prices — distortions that
land exactly at fold boundaries.

Produced by issue 0015.

---

## model_features

Feature matrix consumed by both models. At least 20 columns; the authoritative
list is `FEATURE_COLUMNS` in `src/features/lob_features.py`.

Required identifiers: `timestamp`, `sequence_number` (unique, increasing).

**Causality guarantee:** every feature is trailing. Rolling windows use
`center=False, min_periods=window`. No feature reads a row with a later
`sequence_number`.

Produced by issue 0016.

---

## model_labels

| Column | Type | Constraint |
|---|---|---|
| `timestamp`, `sequence_number` | int64 | row-aligned with features |
| `label_1/3/5` | float64 | `0` or `1`, or `NaN` in the trailing horizon |
| `fwd_return_1/3/5` | float64 | fractional forward return |

Row count equals the feature matrix exactly. A test asserts they join on
`sequence_number` with zero unmatched rows.

Produced by issue 0017.

---

## predictions

Out-of-sample predictions, assembled from per-fold models.

| Column | Type | Critical | Constraint |
|---|---|---|---|
| `timestamp`, `sequence_number` | int64 | yes | sorted, unique per horizon |
| `horizon` | int64 | yes | |
| `fold_id` | int64 | yes | which fold's model produced this row |
| `prediction` | int64 | yes | `0` or `1` |
| `confidence` | float64 | yes | in `[0, 1]` |
| `signal` | float64 | yes | `2 * confidence - 1`, in `[-1, 1]` |
| `mid_price`, `spread_bps` | float64 | no | for downstream cost computation |
| `label_actual`, `fwd_return_actual` | float64 | no | ground truth for attribution |

**The out-of-sample invariant:** every row's `sequence_number` lies inside the
test index set of the fold named in `fold_id`. Rows belonging to no test fold
are absent, never filled with a default.

Both models share this schema exactly, so Phase 3 and 4 consume them through
one code path. Produced by issues 0024 and 0031.

---

## execution_results

Simulated order outcomes, one row per order.

| Column | Type | Notes |
|---|---|---|
| `order_id` | int64 | unique |
| `side`, `limit_price`, `quantity` | | submission parameters |
| `decision_timestamp`, `decision_sequence`, `decision_mid_price` | | slippage reference |
| `arrival_timestamp`, `arrival_sequence`, `latency_ms` | | post-latency arrival |
| `state` | str | `FILLED`, `PARTIALLY_FILLED`, `CANCELLED`, `EXPIRED`, `RESTING` |
| `quantity_filled`, `quantity_remaining`, `fill_ratio` | float64 | |
| `average_fill_price`, `time_to_first_fill_ms` | float64 | `NaN` when unfilled |
| `initial_queue_position`, `final_queue_position` | float64 | in shares, not order count |
| `slippage_bps_total/spread/latency/queue/residual` | float64 | components sum to total |
| `non_execution_reason` | str | populated for every unfilled order |
| `opportunity_cost_bps` | float64 | signed: positive = missed profit |

**Unfilled orders are present**, with `NaN` in fill-dependent columns. Dropping
them would hide the fill rate, which is one of Phase 3's headline findings.

**Slippage is `NaN` when unfilled, never `0`.** Zero would enter Phase 4's
averages as a perfect execution and make execution look better precisely
because it failed.

Fill-level detail lives in a separate file joined on `order_id`, since the
grain differs.

Produced by issues 0045 and 0050.

---

## Adding a contract

1. Define it here first, before writing the transformation
2. Add it to `CONTRACTS` in `scripts/validate_schema.py`
3. Implement the stage
4. Gate the write on validation passing
5. Update `data/README.md`

---

## Related

- Executable form: `scripts/validate_schema.py`
- Metadata convention: `references/provenance_chain.md`
- LOB-specific issues: `references/lob_data_pitfalls.md`
- Live schemas: `data/README.md`

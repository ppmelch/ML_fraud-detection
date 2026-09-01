# Architecture Rules

The canonical concepts and where each one lives. A reviewer needs this to tell
reuse from duplication without reading the whole tree.

`.claude/Agent.md` is the binding statement of these rules; this document is
the map.

---

## One implementation per concept

| Concept | Canonical location | Symbol |
|---|---|---|
| LOB reconstruction | `src/data/lob_reconstruction.py` | `LOBReconstructor`, `OrderBook`, `PriceLevel` |
| Schema validation | `src/data/data_validation.py` | `validate_raw_data_schema` |
| Data quality profiling | `src/data/data_quality.py` | `profile_data_quality` |
| Data cleaning | `src/data/data_preparation.py` | `clean_raw_data` |
| LOB validation | `src/data/lob_validation.py` | `validate_temporal_ordering`, `validate_lob_balance` |
| OFI signal | `src/features/order_flow_imbalance.py` | `compute_ofi`, `compute_ofi_features` |
| Forward returns | `src/features/order_flow_imbalance.py` | `compute_price_movement` |
| LOB features | `src/features/lob_features.py` | `compute_lob_features`, `FEATURE_COLUMNS` |
| Labels | `src/features/labels.py` | `create_labels` |
| Walk-forward CV | `src/features/walk_forward_cv.py` | `walk_forward_split` |
| Model interface | `src/models/base_model.py` | `BaseModel` |
| OFI baseline | `src/models/ofi_baseline.py` | `OFIBaseline` |
| DeepLOB | `src/models/deeplab.py` | `DeepLOB` |
| Training loop | `src/models/training.py` | `train_model` |
| Prediction export | `src/models/prediction_export.py` | `generate_oos_predictions` |
| Order object | `src/simulator/order.py` | `SimulatedOrder`, `Fill` |
| Matching engine | `src/simulator/matching_engine.py` | `MatchingEngine` |
| Latency | `src/simulator/latency.py` | `LatencyModel` |
| Execution metrics | `src/simulator/execution_metrics.py` | `compute_slippage` |
| Backtest orchestration | `src/backtester/backtest_engine.py` | `BacktestEngine` |
| P&L waterfall | `src/backtester/pnl_calculator.py` | `PnLWaterfall` |
| Metrics | `src/utils/metrics.py` | all classification metrics |

`check_duplication.py` encodes this table. When adding a canonical concept,
add it there too.

---

## Reuse boundaries that are easy to get wrong

### The matching engine reuses the reconstructor

`MatchingEngine` does not implement an order book. It holds a
`LOBReconstructor` and drives it, tracking our simulated orders alongside the
reconstructed market book.

Two order books in one project diverge, and the divergence surfaces as a Phase
4 number that disagrees with Phase 1's own reconstruction.

**Reject:** a second `OrderBook` class inside `src/simulator/`.

### Both models share `BaseModel`

`OFIBaseline` and `DeepLOB` implement the same interface: `fit`, `predict`,
`predict_proba`, `get_params`, `save`, `load`.

This is what lets one evaluation path score both. Two models with different
APIs produce a comparison script full of `isinstance` branches, and the two
models stop being measured through the same lens.

**Reject:** a model that exposes only `forward()`.

### Prediction export is model-agnostic

`generate_oos_predictions` takes fold results and a `BaseModel`. Both issue
0024 and issue 0031 call the same function.

**Reject:** a DeepLOB-specific copy of the export logic. Two files that must
stay in schema lockstep will not.

### The diagnostic imbalance in `lob_validation.py` is not OFI

`validate_lob_balance` computes an imbalance metric that shares OFI's formula.
It is a sanity gauge on reconstruction, computed inline, never exported as a
feature.

The canonical OFI is `compute_ofi`. If they ever need to agree,
`lob_validation` imports from `order_flow_imbalance`, never the reverse.

### Notebooks import, they do not implement

The throwaway running book in the issue 0003 EDA notebook is the last one
permitted. From issue 0007 onward, every notebook imports `LOBReconstructor`.

**Reject:** any reimplementation of a canonical concept in `notebooks/`.

---

## Layering

```
src/utils/         no dependencies on other src/ modules
src/data/          may use utils
src/features/      may use utils, data
src/models/        may use utils, data, features
src/simulator/     may use utils, data          (not models)
src/backtester/    may use utils, simulator, models
src/visualization/ may use anything; nothing imports it
```

The simulator not depending on models is deliberate: it consumes prediction
files, not model objects, so it can replay any prediction source without a
model being loadable.

**Reject:** an import that runs backwards through this list.

---

## Where a new thing belongs

| It is... | It goes in... |
|---|---|
| domain logic about the order book | `src/data/` |
| a signal or derived column | `src/features/` |
| a model or training mechanic | `src/models/` |
| execution simulation | `src/simulator/` |
| P&L or backtest orchestration | `src/backtester/` |
| a metric, an I/O helper, logging | `src/utils/` |
| a plot | `src/visualization/` |
| a runnable entry point | `scripts/` |
| exploration or a figure for a deliverable | `notebooks/` |

If it does not fit, that is a signal the abstraction is wrong — raise it in
review rather than putting it somewhere convenient.

---

## Validation gates

Every pipeline script in `scripts/` follows the same shape:

```python
report = validate(...)
if not report["passed"]:
    print_report(report)
    sys.exit(1)          # do not write

df.to_csv(output_path, index=False)
write_metadata(output_path, ..., upstream_artifacts=[...])
```

**Reject:** a script that writes output before validating it, or one that
writes without recording `upstream_artifacts`.

---

## When a second implementation is genuinely needed

Sometimes it is — a weighted OFI alongside the standard one, a second labelling
scheme.

The requirements:
1. A distinct name that says how it differs (`compute_ofi_weighted`)
2. An ADR in `docs/adr/` recording why both exist
3. An entry in `docs/CONTEXT.md` if it introduces a domain term
4. Added to `CANONICAL_CONCEPTS` in `check_duplication.py` with its own owner

Same name, same responsibility, two places is what the rule forbids.

---

## Related

- Binding rules: `.claude/Agent.md`
- Review order: `references/review_checklist.md`
- Antipatterns: `references/common_antipatterns.md`
- Detection: `scripts/check_duplication.py`

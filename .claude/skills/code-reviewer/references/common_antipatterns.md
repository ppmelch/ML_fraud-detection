# Common Antipatterns

The specific mistakes this project is prone to, each with the failure it
eventually causes.

---

## The convenient second implementation

```python
# in a notebook, or a new module
def compute_ofi(df):
    return (df["bid_vol"] - df["ask_vol"]) / (df["bid_vol"] + df["ask_vol"])
```

**Why it happens:** importing feels heavier than four lines.

**What it costs:** the depth parameter changes in
`order_flow_imbalance.py` and this copy does not. Two OFI values now exist,
both plausible, and the Phase 4 attribution disagrees with the Phase 2
accuracy for reasons nobody can trace.

**Fix:** `from src.features.order_flow_imbalance import compute_ofi`.

---

## The centered rolling window

```python
df["smoothed"] = df["mid"].rolling(20, center=True).mean()
```

**Why it happens:** centering is the default in some libraries and looks
symmetric, which feels correct.

**What it costs:** every row's feature reads ten rows into the future.
Accuracy rises, the model looks good, and the whole Phase 2 result is
fiction. Nothing downstream can detect it.

**Fix:** `rolling(20, center=False, min_periods=20)`.

---

## The partial window

```python
df["ofi_5"] = ofi.rolling(5).mean()          # min_periods defaults to 5? no
```

`min_periods` defaults to the window size for `.mean()`, but not for every
operation, and relying on the default across a codebase invites the case where
a 5-window quietly computes from 2 rows.

**What it costs:** the early rows carry a different statistic from the rest,
and those rows sit exactly at fold boundaries.

**Fix:** state `min_periods=window` explicitly, always.

---

## Filling NaN to make the data look complete

```python
df["ofi_3"] = df["ofi_3"].fillna(0)
df["label_5"] = df["label_5"].fillna(0)
```

**Why it happens:** downstream code raises on NaN, and filling is faster than
handling.

**What it costs:** the first fill fabricates a balanced book that never
existed. The second invents future prices. Both distortions land at fold
boundaries, where walk-forward validation is most sensitive.

**Fix:** preserve NaN, document the counts in metadata, and let each consumer
drop what it needs for its own horizon.

---

## Fitting the scaler before splitting

```python
scaler.fit(all_features)                     # sees the test period
X_train = scaler.transform(features[train_idx])
```

**What it costs:** the test period's distribution leaks into training. The
effect is small enough to be invisible and real enough to inflate results.

**Fix:** `standardize_features(df, fit_indices=train_indices)` — the function
takes the indices and physically cannot see anything else.

---

## Using the test fold for early stopping

```python
train_model(model, train_loader, val_loader=test_loader)
```

**Why it happens:** the test fold is right there and conveniently sized.

**What it costs:** the stopping epoch was chosen with knowledge of test
performance. Every metric reported afterwards is optimistically biased, and
looks entirely normal.

**Fix:** carve validation from the tail of the training window, behind the
purge gap.

---

## Sweeping the threshold on test predictions

```python
best = max(thresholds, key=lambda t: f1(y_test, proba_test >= t))
```

**What it costs:** the reported number estimates the maximum over thresholds,
not the performance of a threshold.

**Fix:** select on training predictions, apply unchanged to test, report both
the 0.5 result and the selected one.

---

## Warm-starting folds

```python
model = build_model(config)          # outside the loop
for fold in splits:
    train_model(model, ...)          # continues from the previous fold
```

**What it costs:** fold N's model has seen fold N-1's test data. Later folds
inflate specifically, producing a pattern that reads as the model improving.

**Fix:** build the model inside the loop.

---

## Predicting the whole dataset with one model

```python
model = load_model("deeplab_fold0.pt")
predictions = model.predict(all_features)     # mostly in-sample
```

**Why it happens:** it is faster and produces a complete series with no gaps.

**What it costs:** most predictions come from a model that memorized those
rows. The Phase 4 P&L is fiction. An in-sample prediction file looks exactly
like an out-of-sample one.

**Fix:** assemble from per-fold test sets; `validate_oos_coverage` asserts
every row is out-of-sample.

---

## Zero slippage for unfilled orders

```python
slippage = order.avg_fill_price - order.decision_mid if filled else 0.0
```

**What it costs:** unfilled orders enter the average as perfect executions,
pulling mean slippage toward zero — making execution look better precisely
because it failed.

**Fix:** return `None`, and account for unfilled orders as opportunity cost in
a separate line.

---

## Marking positions at the last fill price

```python
equity = cash + position * last_fill_price
```

**What it costs:** a position just bought at the ask is flattered and one just
sold at the bid is penalized, producing P&L oscillation that has nothing to do
with the strategy.

**Fix:** mark at the mid from the LOB snapshots.

---

## Subtracting the same cost twice

Phase 3 produces both `slippage_bps_queue` (from the reference-price chain) and
`queue_cost_bps` (from a counterfactual run). They measure related things by
different methods.

**What it costs:** a waterfall that subtracts both charges the strategy twice
for the same friction — and still reconciles internally, so the arithmetic
check does not catch it.

**Fix:** choose one, document which, and use the other as a cross-check.
`check_double_counting` compares the waterfall total against Phase 3's
independently measured execution cost.

---

## Float equality on accumulated quantities

```python
if order.quantity_remaining == 0:
    remove(order)
```

**What it costs:** after repeated subtraction the value is `1e-17`, the order
is never removed, and a phantom order sits in the book for the rest of the
session.

**Fix:** a shared `QUANTITY_EPSILON` constant, used everywhere.

---

## Writing output before validating it

```python
df.to_csv(path)
report = validate(df)                # too late
```

**What it costs:** a corrupt file reaches disk, is picked up downstream, and
the failure surfaces as a strange result rather than an error.

**Fix:** validate, then write, and `sys.exit(1)` on failure.

---

## Hardcoded paths and secrets

```python
df = pd.read_csv("/home/user/project/data/raw.csv")
API_KEY = "sk-1234567890abcdef"
```

**What it costs:** the first breaks on every other machine and inside Docker.
The second ends up in git history, where removing it requires rewriting
history.

**Fix:** read paths from `.env`; keep secrets out of source entirely.

---

## A test that cannot fail

```python
def test_reconstruct():
    result = reconstructor.reconstruct(events)
    assert result is not None
```

**What it costs:** coverage goes up, confidence goes up, and the test would
pass on a completely broken implementation.

**Fix:** assert the values. `assert [o.order_id for o in level.orders] == [2, 3]`
tells you the queue order is right; `assert result is not None` tells you
nothing.

---

## Related

- Review order: `references/review_checklist.md`
- Concept map: `references/architecture_rules.md`
- Binding rules: `.claude/Agent.md`
- Detection: `scripts/check_conventions.py`, `scripts/check_duplication.py`

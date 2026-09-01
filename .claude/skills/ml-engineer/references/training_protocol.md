# Training Protocol

How a training run is structured in this project, and why each constraint
exists.

---

## Fold discipline

Both models train on the folds persisted by issue 0018. Neither regenerates
them.

```python
splits = load_model_splits("data/processed/model_splits_v1.json")
assert splits_match_phase1(splits)   # boundaries identical to cv_splits_v1.json
```

If the baseline trains on folds built one way and DeepLOB on folds built
another, the accuracy difference between them is partly a fold artifact, and
the Phase 2 comparison stops meaning anything.

---

## Where the validation split comes from

Early stopping needs a validation signal. The test fold is right there and
conveniently sized, and using it is the most common way an otherwise careful
protocol leaks.

Carve validation from the **tail of the training window**:

```
|------- train -------|--val--|--purge--|---- test ----|
```

The purge gap sits between validation and test, so validation cannot see the
test period either.

**Assert it:** `set(val_indices) & set(test_indices) == set()`. A comment
saying they are disjoint is not the same as a test proving it.

---

## Fresh initialization per fold

Every fold starts from newly initialized weights.

Warm-starting fold N from fold N-1's final weights leaks information forward:
fold N's model has already seen fold N-1's test data during that earlier
training. It is an easy accident inside a loop, and it inflates later folds
specifically — producing a pattern that looks like the model improving over
time.

```python
for fold in splits:
    model = build_model(config)      # inside the loop, not outside
    train_model(model, ...)
```

---

## Class imbalance

Handle it in the loss, not in the data:

```python
pos_weight = torch.tensor([n_negative / n_positive], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

Compute `n_negative` and `n_positive` from the **training fold only**.

**Why not resampling.** Oversampling the minority class changes the effective
dataset and duplicates rows; undersampling discards data. Both interact badly
with sequence windows, where a sample is a contiguous run of snapshots and
duplicating or removing one breaks the temporal structure the model is
supposed to learn from.

---

## Loss and activation

`BCEWithLogitsLoss` on logits. Never `BCELoss` on probabilities.

The fused version is numerically stabler: it computes the log-sum-exp in a
form that does not overflow for large logits.

The corollary is a rule that must hold across the whole codebase:

- `forward()` returns **logits**, unbounded
- `predict_proba()` applies sigmoid **once**
- nothing else applies sigmoid

A sigmoid applied twice flattens confidence toward 0.5 and surfaces in Phase 4
as inexplicably timid position sizing, with no error anywhere.

---

## Early stopping

- Monitor a metric that reflects the goal, not just validation loss. On
  imbalanced data, loss can improve while decision quality degrades — balanced
  accuracy or AUC aligns stopping with what issue 0030 reports.
- `min_delta` prevents stopping on noise.
- **Restore best weights at the end.** Stopping at epoch 40 because epoch 30
  was best, then keeping epoch 40's weights, discards exactly what early
  stopping was for.

```python
if improved:
    best_state = copy.deepcopy(model.state_dict())
...
model.load_state_dict(best_state)
```

---

## Checkpointing

Each checkpoint records weights, model config, training config, epoch number,
metric value, and seed. A checkpoint without its config cannot be loaded into
a matching architecture six weeks later.

Checkpoint paths are deterministic — `deeplab_h{horizon}_fold{fold}.pt` — so a
resumed sweep can skip what already exists.

---

## Logging

One row per epoch: train loss, validation loss, every metric from
`src/utils/metrics.py`, learning rate, and epoch duration.

A run whose log was not persisted has to be repeated to answer any question
about it. `audit_training_run.py` reads these logs.

---

## Hyperparameter selection

Any tuning that consults test-fold performance turns the test set into a
validation set. The reported number then estimates the maximum over choices,
not the performance of one choice.

- Tune on the training-fold validation split
- Record every configuration tried, including the ones that lost
- Freeze before touching test data
- If a retrain happens after seeing test results, record that it happened

---

## Two sanity tests that catch real bugs

A training loop that silently fails to learn is hard to distinguish from a
hard problem. Two synthetic tests settle it:

```python
def test_learns_separable_problem():
    """The loop MUST solve this."""
    # linearly separable synthetic data
    assert final_val_accuracy > 0.95

def test_does_not_learn_noise():
    """The loop MUST NOT solve this."""
    # random labels, no signal
    assert final_val_accuracy < 0.60
```

A loop that passes the first and fails the second — reaching high validation
accuracy on pure noise — has a leak. Finding that here is far cheaper than
finding it after an eighteen-fold sweep.

---

## Runtime expectations

Three horizons times six folds is eighteen training runs. Resume support and
per-fold logging exist so an interruption at fold fourteen does not cost the
first thirteen.

Record wall-clock time and peak memory per fold, so a rerun can be planned.

---

## Related

- Fold construction: `docs/issues/phase-2/0018`
- Training loop: `docs/issues/phase-2/0028`
- Training runs: `docs/issues/phase-2/0029`
- Validation theory: `.claude/skills/quant-researcher/references/validation_protocol.md`
- Audit tool: `scripts/audit_training_run.py`

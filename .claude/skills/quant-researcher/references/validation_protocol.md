# Validation Protocol

The walk-forward structure this project uses, and why each part of it exists.

---

## Why not a simple train/test split

A random split assigns future rows to training and past rows to testing. The
model then predicts a past it has already seen, using information that did not
exist at the time. Accuracy measured this way is not an estimate of anything.

A single chronological split fixes the ordering but tests one regime once. It
cannot show whether an edge persists, and with a short sample it may test a
regime that never recurs.

Walk-forward validation trains on a past window, tests on the window that
follows, then advances — repeating the deployment situation many times over.

---

## The fold structure

```
fold 1: |---- train ----|--purge--|-- test --|--embargo--|
fold 2:      |---- train ----|--purge--|-- test --|--embargo--|
fold 3:           |---- train ----|--purge--|-- test --|--embargo--|
```

Every fold trains only on rows preceding its test window. Folds advance by
`step_size`, defaulting to `test_size` so test windows do not overlap.

---

## Purge and embargo protect against different leaks

They are easy to conflate. They are not the same thing and they point in
opposite directions.

### Purge — removes training rows *before* the test window

A label at time `t` looks forward `h` events. If `t` sits within `h` events of
the test window, that label was computed from prices inside the test period.
The row is in the training set, but its target carries test information.

Purge removes the last `purge_size` rows of the training window.

**Purge width must be at least the label horizon.** A purge of 3 with a
horizon of 5 leaves two rows whose labels reach into the test period. This is
the single most common way leakage survives a protocol that looks correct.

### Embargo — removes rows *after* the test window from *future* training sets

Rows immediately following a test window are correlated with it: overlapping
feature windows, autocorrelated returns, the same microstructure regime. A
later fold that trains on them trains on a shadow of an earlier test set.

Embargo excludes the `embargo_size` rows after each test window from every
subsequent training set.

### Getting the direction wrong

Applying purge after the test window and embargo before it produces a diagram
that looks symmetric and protects against neither leak. The assertions in
issue 0014 exist because this error is invisible on inspection.

---

## What a leak-free fold set must satisfy

Assert these, do not assume them:

1. `max(train_indices) < min(test_indices)` — training strictly precedes test
2. `min(test_indices) - max(train_indices) > purge_size` — the purge gap is real
3. `set(train_indices) & set(test_indices) == set()` — no row in both
4. No index inside an embargo region appears in any later training set
5. Every test index lies within its declared test window

Issue 0014 implements these as tests, including a randomized property test,
because they are universal claims and hand-picked examples verify only the
cases someone thought of.

---

## Sequence models need one more guarantee

A row-level split can be perfectly clean while the windows built on top of it
leak. A sequence of length 100 ending on the first test row reaches 99 rows
back — straight through the purge gap and into training data.

The guard is to build windows strictly inside a fold's own index set, and to
reject any window spanning a discontinuity:

```python
window = indices[pos - seq_len + 1 : pos + 1]
if window[-1] - window[0] == seq_len - 1:      # contiguous
    valid.append(pos)
```

The contiguity check is what makes the purge gap real for a sequence model.
Issue 0019 covers this.

---

## Scaler fitting

Standardization parameters fitted on the full sample leak the test period's
distribution into training. The fix is structural rather than procedural:
`standardize_features` takes `fit_indices` and physically cannot see anything
else.

Each fold carries its own fitted parameters, stored with the split so training
and inference provably use the same transform.

---

## Hyperparameter selection

Any tuning that selects on test-fold performance turns the test set into a
validation set, and the reported accuracy becomes an optimistic estimate.

Carve validation from the tail of the training window:

```
|------- train -------|--val--|--purge--|---- test ----|
```

Early stopping, threshold selection, and learning-rate choice all read the
validation split. The test fold is touched exactly once, to produce the number
that gets reported.

---

## Both models must use identical folds

If the baseline is evaluated on folds built one way and the deep model on folds
built another, any difference between them is partly a fold artifact. Phase 2
generates folds once, persists them, and both training paths load the same
file — with an assertion that the boundaries match.

---

## Small fold counts limit what can be concluded

Six folds give six paired observations. A Wilcoxon test on six pairs has very
limited power, and a mean difference of one percentage point with a standard
deviation of three is not evidence of anything.

Report the test, report the win count, and state the limitation. A p-value
above 0.05 on six folds is not proof of no difference, and one below it is not
proof of a large one.

---

## Related

- Implementation: `src/features/walk_forward_cv.py` (issue 0014)
- Decision record: `docs/adr/0003-walk-forward-cv-purge-embargo.md`
- Glossary: `docs/CONTEXT.md` — purge, embargo, walk-forward CV
- Leakage detection: `scripts/check_leakage.py`

# Review Checklist

What to check, in the order that catches the most expensive problems first.

A duplication problem makes style comments irrelevant, because the code should
not exist in that form at all. Work top down and stop when something blocking
appears.

---

## 1. Architecture — before anything else

- [ ] Does this duplicate an existing implementation? (`check_duplication.py`)
- [ ] Does it belong in this module, per the layering in `architecture_rules.md`?
- [ ] Does any import run backwards through the layer order?
- [ ] Is the repository cleaner after this change than before?
- [ ] If a second implementation was genuinely needed, is there an ADR?

**Blocking.** A canonical concept implemented twice is rejected regardless of
how well the new version works.

---

## 2. Correctness — the failures that survive to Phase 4

### Causality
- [ ] Every rolling window is trailing: `center=False, min_periods=window`
- [ ] Targets are forward-shifted: `.shift(-h)`, not `.shift(h)`
- [ ] No feature reads a row with a later `sequence_number`
- [ ] A causality test exists — append future rows, assert earlier values unchanged

### Leakage
- [ ] Scalers fitted on training indices only
- [ ] Validation split disjoint from test, asserted not assumed
- [ ] Purge width at least the label horizon
- [ ] Sequence windows do not span a fold boundary
- [ ] No threshold or hyperparameter selected on test-fold performance

### Alignment and signs
- [ ] Features and labels join on `sequence_number` with zero unmatched rows
- [ ] Slippage sign: positive always means worse for us, both sides tested
- [ ] Micro-price weighting is crossed (bid volume × ask price)
- [ ] Fill price is the resting price, not the limit price

### Numerics
- [ ] Float comparisons use a tolerance, not `== 0`
- [ ] The tolerance constant is shared, not redefined per module
- [ ] Prices rounded to tick size before use as dictionary keys
- [ ] `skipna=True` when summing level columns, so `NaN` padding drops out

**Blocking.** A causality or leakage finding makes every downstream number
optimistic, and nothing later can recover it.

---

## 3. Reproducibility

- [ ] Seeds set for Python `random`, numpy, and torch
- [ ] DataLoader shuffling seeded through an explicit generator
- [ ] Every parameter affecting the output recorded in config
- [ ] New artifacts write metadata with `upstream_artifacts`
- [ ] Pipeline scripts validate before writing, and exit non-zero on failure

---

## 4. Tests

- [ ] New source modules have corresponding tests
- [ ] **Would each test fail if the code were wrong?** A test asserting the
      function returns something is decoration.
- [ ] Invariants asserted, not only the happy path
- [ ] Error paths tested — both `strict=True` raising and `strict=False` skipping
- [ ] Float comparisons use `pytest.approx`
- [ ] Fixtures minimal enough to read in one screen
- [ ] Coverage meets the target stated in the issue

---

## 5. Conventions

Mostly handled by `check_conventions.py`. Read its output rather than checking
by hand.

- [ ] Docstrings on public functions and classes, with types and an example
- [ ] `PascalCase` classes, `snake_case` functions and variables
- [ ] No absolute paths, no secrets
- [ ] `logging` rather than `print` in `src/`
- [ ] Canonical vocabulary from `docs/CONTEXT.md`

---

## 6. Documentation

- [ ] New domain term added to `docs/CONTEXT.md`
- [ ] New artifact schema added to `data/README.md`
- [ ] Design document updated if the implementation deviated from it
- [ ] `CHANGELOG.md` updated

---

## Running the automated part

```bash
python .claude/skills/code-reviewer/scripts/review_report.py \
    --base main --head HEAD --output review.md
```

Then read the diff for what tooling cannot see: whether the abstraction is
right, whether the tests would catch a real bug, whether the change makes the
repository better.

---

## How to write a finding

State the defect, the failure it causes, and the location.

**Weak:** "this could be cleaner"

**Strong:** "`compute_signal` in `src/features/custom.py:42` recomputes OFI.
It will diverge from `order_flow_imbalance.py` the next time the depth
parameter changes, and the two values will silently disagree in the Phase 4
attribution."

A comment that does not name a consequence is a preference, and preferences
are what `black` is for.

---

## What to accept with a comment rather than block

- Missing docstring on a genuinely private helper
- `print` in a notebook or a one-off script
- Formatting `black` will fix anyway
- A vocabulary warning in a comment rather than an identifier

---

## What to always block

- A second implementation of a canonical concept
- A feature that reads forward
- Selection on test-fold performance
- A secret or absolute path in source
- A pipeline script writing unvalidated output
- A test that cannot fail

---

## Related

- Binding rules: `.claude/Agent.md`
- Concept map: `references/architecture_rules.md`
- Specific mistakes: `references/common_antipatterns.md`
- PR template: `.github/PULL_REQUEST_TEMPLATE.md`

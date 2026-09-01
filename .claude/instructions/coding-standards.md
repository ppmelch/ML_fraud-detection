# Coding Standards

**Binding on: every agent that writes code, and code-reviewer when checking it.**

`.claude/Agent.md` states the architectural rules. This document holds the
mechanical detail those rules imply, so the skills and agents can reference it
rather than each carrying a copy.

---

## One implementation per concept

The canonical map — what exists, and where — lives in
`.claude/skills/code-reviewer/references/architecture_rules.md` and is encoded
executably in `check_duplication.py`.

The rule: if two pieces of code compute the same thing, the repository is
already wrong. Fix that before adding the feature.

Notebooks import from `src/`. They do not reimplement. The one exception was
the throwaway running book in the issue 0003 EDA notebook, before
`LOBReconstructor` existed.

**When a variant is genuinely needed** — a weighted OFI alongside the standard
one — it needs a distinct name, an ADR recording why both exist, an entry in
`docs/CONTEXT.md` if it introduces a term, and registration in
`CANONICAL_CONCEPTS`.

---

## Layering

```
src/utils/         depends on nothing else in src/
src/data/          may use utils
src/features/      may use utils, data
src/models/        may use utils, data, features
src/simulator/     may use utils, data          (not models)
src/backtester/    may use utils, simulator, models
src/visualization/ may use anything; nothing imports it
```

The simulator not depending on models is deliberate: it consumes prediction
files rather than model objects, so it can replay any prediction source without
a model being loadable.

An import running backwards through this list is a finding.

---

## Formatting and linting

```bash
black src/ tests/ scripts/          # line length 100
flake8 src/ tests/ scripts/
ruff check src/ tests/ scripts/
mypy src/
```

`make lint` runs all four. CI runs the same.

---

## Docstrings

Every public function, class, and module. Include description, parameters with
types, returns, raises, and an example where it earns its place.

```python
def compute_ofi(lob_df: pd.DataFrame, depth: int = 5) -> pd.Series:
    """Compute the Order Flow Imbalance signal.

    Args:
        lob_df: reconstructed LOB with bid_volume_1..N and ask_volume_1..N
        depth: number of levels per side to include

    Returns:
        OFI per snapshot, bounded in [-1, 1]. Zero total volume returns 0.0.

    Raises:
        ValueError: if required columns are absent or depth < 1

    Example:
        >>> compute_ofi(lob, depth=1).iloc[0]
        0.25
    """
```

Private helpers get docstrings too. Python has no true private, and the next
reader is usually a stranger.

---

## Naming

| Kind | Style | Example |
|---|---|---|
| Classes | `PascalCase` | `LOBReconstructor`, `DeepLOB` |
| Functions | `snake_case` | `compute_ofi`, `reconstruct` |
| Variables | `snake_case` | `bid_volumes`, `queue_position` |
| Constants | `UPPER_SNAKE` | `QUANTITY_EPSILON` |
| Modules | `snake_case.py` | `lob_reconstruction.py` |
| Data files | `snake_case_vN.csv` | `lob_reconstructed_v1.csv` |

Standard abbreviations only: `ofi`, `lob`, `pnl`, `cv`, `lstm`. Not `obe`,
`qp`, `mr`.

**If a term appears in `docs/CONTEXT.md`, use its canonical form exactly.**
Terminology drift between code and documentation is a correctness concern here,
not a style preference — `check_conventions.py` flags `orderbook`,
`order_imbalance`, `decomposition`, and their kin.

---

## Logging, not print

```python
import logging
logger = logging.getLogger(__name__)
logger.info("LOB reconstructed: %d snapshots", len(snapshots))
```

`print` in `src/` is an error. In a notebook or a one-off script it is fine.

---

## No secrets, no absolute paths

```python
# No
API_KEY = "sk-1234567890abcdef"
df = pd.read_csv("/home/user/project/data/raw.csv")

# Yes
import os
from dotenv import load_dotenv
load_dotenv()
df = pd.read_csv(os.path.join(os.getenv("DATA_DIR", "./data"), "raw.csv"))
```

The first breaks on every other machine and inside Docker. The second ends up
in git history, where removing it requires rewriting history.

---

## Tests

**The question that matters: would this test fail if the code were wrong?**

```python
# Decoration — passes on a completely broken implementation
assert result is not None

# A test
assert [o.order_id for o in level.orders] == [2, 3]
```

- Placement: `src/data/lob_reconstruction.py` → `tests/unit/test_lob_reconstruction.py`
- Shared fixtures in `tests/conftest.py`
- Naming: `test_<function>_<scenario>`
- Float comparisons: `pytest.approx`, matching the shared epsilon
- Assert invariants, not only the happy path
- Test error paths in both `strict=True` (raises) and `strict=False` (skips)
- Coverage: >90% for domain logic, >85% acceptable for plumbing

Where a replay is available, assert the invariant after **every** event rather
than at the end. That catches classes of bug hand-written scenarios miss.

---

## Reproducibility

Seed Python `random`, numpy, and torch together. Seed DataLoader shuffling
through an explicit `torch.Generator` rather than global state.

Record every parameter that affects the output, including the seed and the
library versions. Verify before a long run:

```bash
python .claude/skills/ml-engineer/scripts/check_reproducibility.py
```

The full checklist is
`.claude/skills/ml-engineer/references/reproducibility_checklist.md`.

---

## Before opening a PR

```bash
make lint && make test
python .claude/skills/code-reviewer/scripts/check_duplication.py src/
python .claude/skills/code-reviewer/scripts/check_conventions.py src/ scripts/
python .claude/skills/git-workflow/scripts/check_pr_readiness.py
```

---

## Related

- Architectural rules: `.claude/Agent.md`
- Canonical concept map: `.claude/skills/code-reviewer/references/architecture_rules.md`
- Antipatterns: `.claude/skills/code-reviewer/references/common_antipatterns.md`
- Review order: `.claude/skills/code-reviewer/references/review_checklist.md`
- Pipeline rules: `.claude/instructions/pipeline-contract.md`

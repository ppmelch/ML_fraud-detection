---
name: code-reviewer
description: Code review for the PAP-LOB-Trading repository, enforcing the architectural rules in .claude/Agent.md — one implementation per business concept, mandatory reuse, no hardcoded paths or secrets, docstrings with types, causality in feature code, and no test-set selection. Use when reviewing a pull request, before opening one, when checking whether a change introduces duplication, or when auditing the repository for drift from its own rules. This skill governs code quality and architectural compliance; correctness of research methodology belongs to quant-researcher.
---

# Code Reviewer

Review against the rules this repository has already committed to, in
`.claude/Agent.md`. The rules exist because the alternative — a second OFI
implementation that drifts from the first, a feature that quietly reads
forward — produces failures that surface phases later as unexplainable numbers.

## The one rule everything else follows from

**There must be exactly one implementation of every business concept.**

One `LOBReconstructor`. One `compute_ofi`. One `DeepLOB`. One
`MatchingEngine`. One set of metrics. If two pieces of code compute the same
thing, the repository is already wrong, and the review should say so before
the feature is discussed.

## Quick Start

```bash
# Find duplicate implementations of core concepts
python scripts/check_duplication.py src/

# Check conventions: naming, docstrings, hardcoded paths, print statements
python scripts/check_conventions.py src/ scripts/

# Generate a full review report for a diff
python scripts/review_report.py --base main --head HEAD
```

## Core Capabilities

### 1. Duplication Detection (`check_duplication.py`)

Finds second implementations of concepts that must exist once.

**Detects:**
- Functions whose names match a canonical concept outside its owning module
- Reimplementation of OFI, LOB reconstruction, matching, or metrics in
  notebooks and scripts
- Near-identical function bodies across modules (normalized comparison)
- Metric formulas written inline instead of imported from `src/utils/metrics.py`
- Multiple definitions of the same class name across the tree

**Usage:**
```bash
python scripts/check_duplication.py <path> [--min-lines 8] [--verbose]
```

### 2. Convention Checking (`check_conventions.py`)

Checks the mechanical rules that reviewers otherwise spend attention on.

**Checks:**
- Docstrings on every public function and class
- Type annotations on public signatures
- Naming: `PascalCase` classes, `snake_case` functions and variables
- No hardcoded absolute paths
- No secrets or API keys in source
- `logging` rather than `print` in `src/`
- Canonical vocabulary from `docs/CONTEXT.md` — flags `orderbook`, `order imbalance`, and similar
- Causality markers: `center=True` or positive `.shift()` in feature code

**Usage:**
```bash
python scripts/check_conventions.py <paths...> [--strict]
```

### 3. Review Report (`review_report.py`)

Produces a structured review of a diff, combining the two checks above with
diff-specific analysis.

**Features:**
- Runs duplication and convention checks on changed files only
- Flags new files that duplicate an existing module's responsibility
- Checks that new code in `src/` has corresponding tests
- Verifies `docs/CONTEXT.md` was updated when a new domain term appears
- Produces a checklist matching `.github/PULL_REQUEST_TEMPLATE.md`

**Usage:**
```bash
python scripts/review_report.py [--base main] [--head HEAD] [--output review.md]
```

## Reference Documentation

### Review Checklist

`references/review_checklist.md` — what to check, in the order that catches
the most expensive problems first. Architecture before style.

### Architecture Rules

`references/architecture_rules.md` — the canonical concepts and where each one
lives, so a reviewer can tell reuse from duplication without reading the whole
tree.

### Common Antipatterns

`references/common_antipatterns.md` — the specific mistakes this project is
prone to, each with the failure it eventually causes.

## Shared Instructions

Cross-cutting policy lives in `.claude/instructions/` so it has one home.
These bind this skill:

- [`coding-standards.md`](../../instructions/coding-standards.md) — the mechanical rules you enforce
- [`research-integrity.md`](../../instructions/research-integrity.md) — causality and no-selection-on-test
- [`pipeline-contract.md`](../../instructions/pipeline-contract.md) — validate-before-write and provenance

Read them rather than relying on the summaries below.

---

## Domain Context

**`.claude/Agent.md` is binding.** It is not advisory: a change that violates
it should be rejected regardless of how well it works.

**`docs/CONTEXT.md` is the vocabulary.** Terminology drift between code and
documentation is a real source of error here, so naming is a correctness
concern rather than a style preference.

## Review Workflow

### Order matters

Review in this sequence. A duplication problem makes style comments irrelevant,
because the code should not exist in that form at all.

1. **Architecture** — does this duplicate something? Does it belong here?
2. **Correctness** — causality, leakage, alignment, sign conventions
3. **Reproducibility** — seeds, recorded config, provenance metadata
4. **Tests** — do they exist, and would they fail if the code were wrong?
5. **Conventions** — naming, docstrings, logging
6. **Style** — formatting, which `black` already handles

### Before opening a PR

```bash
make lint
make test
python .claude/skills/code-reviewer/scripts/check_duplication.py src/
python .claude/skills/code-reviewer/scripts/check_conventions.py src/ scripts/
```

### Reviewing someone else's PR

```bash
python .claude/skills/code-reviewer/scripts/review_report.py --base main --output review.md
```

Then read the diff for what tooling cannot see: whether the abstraction is
right, whether a test would actually fail if the code were wrong, whether the
change makes the repository cleaner than before.

## Best Practices

### What to reject
- A second implementation of an existing concept
- A feature computed with `center=True` or a forward-reading window
- A threshold or hyperparameter chosen on test-fold performance
- A pipeline script that writes output without validating it
- A new domain term that is not in `docs/CONTEXT.md`
- A test that cannot fail

### What to accept with a comment
- Missing docstring on a genuinely private helper
- A `print` in a notebook or a one-off script
- Formatting that `black` will fix anyway

### How to phrase a finding

State the defect, the failure it causes, and the location. Not "this could be
cleaner" but "this recomputes OFI; it will diverge from
`src/features/order_flow_imbalance.py` the next time the depth parameter
changes."

A review comment that does not name a consequence is a preference, and
preferences are what `black` is for.

### Reviewing tests
- Would this test fail if the code were wrong? If not, it is decoration.
- Are float comparisons using `pytest.approx`?
- Are the invariants asserted, or only the happy path?
- Is the fixture minimal enough to read?

## Common Commands

```bash
# Full pre-PR check
make lint && make test
python .claude/skills/code-reviewer/scripts/check_duplication.py src/
python .claude/skills/code-reviewer/scripts/check_conventions.py src/ scripts/ tests/

# Review a branch
python .claude/skills/code-reviewer/scripts/review_report.py \
    --base main --head HEAD --output review.md

# Inside Docker
docker compose run --rm test
```

## Troubleshooting

**Duplication check flags a legitimate variant.** If a second implementation is
genuinely needed — a weighted OFI alongside the standard one — give it a
distinct name and record why in an ADR. Same name, same responsibility, two
places is what the check exists to prevent.

**Convention check flags a notebook.** Notebooks are held to a lower standard
than `src/`. Run the check on `src/` and `scripts/` by default.

**A finding seems pedantic.** Check whether it names a consequence. If it does
not, it probably is pedantic — drop it.

## Resources

- Review checklist: `references/review_checklist.md`
- Architecture rules: `references/architecture_rules.md`
- Antipatterns: `references/common_antipatterns.md`
- Binding rules: `.claude/Agent.md`
- Vocabulary: `docs/CONTEXT.md`
- PR template: `.github/PULL_REQUEST_TEMPLATE.md`

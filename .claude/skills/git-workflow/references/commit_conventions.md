# Commit Conventions

Conventional commits, with a type and scope vocabulary matched to this
project's module layout.

---

## Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

Only the header is required. `validate_commit.py` enforces it.

---

## Types

| Type | Use for |
|---|---|
| `feat` | a new capability |
| `fix` | a bug fix |
| `test` | adding or correcting tests |
| `docs` | documentation only |
| `refactor` | restructuring without behaviour change |
| `perf` | a performance improvement |
| `chore` | tooling, dependencies, configuration |
| `ci` | CI or workflow changes |
| `style` | formatting only |

---

## Scopes

Scopes mirror the module layout, so a reader can locate a change from the
subject line alone.

| Scope | Covers |
|---|---|
| `data` | `src/data/` — loading, cleaning, LOB reconstruction |
| `features` | `src/features/` — OFI, feature matrix, labels, CV |
| `models` | `src/models/` — baseline, DeepLOB, training |
| `simulator` | `src/simulator/` — matching engine, latency, execution |
| `backtester` | `src/backtester/` — P&L, waterfall, sensitivity |
| `utils` | `src/utils/` — metrics, I/O, logging |
| `viz` | `src/visualization/` |
| `tests` | test-only changes |
| `docs` | documentation |
| `ci` | workflows |
| `docker` | Dockerfile, compose |
| `deps` | requirements |
| `claude` | `.claude/` skills and agents |
| `scripts` | `scripts/` entry points |

Scope is optional but strongly preferred.

---

## Subject

- Imperative mood: **implement**, not *implemented* or *implements*
- Lowercase first letter
- No trailing period
- Header under 72 characters

The imperative reads as an instruction to the codebase — "apply this commit and
it will *implement the reconstructor*". `validate_commit.py` rejects the
common past-tense and third-person forms.

---

## Body

Wrapped at 72 columns. Explain **why**, not what — the diff already shows what.

Worth a body when the change involves a non-obvious decision, a trade-off, or a
consequence a future reader would not infer.

---

## Footer

- `Closes #0007` — links the issue, closing it on merge
- `BREAKING CHANGE: <description>` — for changes that break a contract
- `Co-Authored-By:` when pairing

`feat` and `fix` commits should cite an issue; `validate_commit.py` warns when
they do not.

---

## Examples from this project

```
feat(data): implement LOBReconstructor with FIFO queue tracking

Rebuilds the order book from raw event streams, maintaining queue
position in shares rather than order count so Phase 3 can estimate
fill probability directly.

Level volume is maintained incrementally rather than recomputed, which
issue 0010's balance assertion is what keeps honest.

Closes #0007
```

```
fix(features): correct rolling window direction in OFI computation

compute_ofi_features used center=True, letting each row read ten rows
into the future. Every accuracy figure measured before this commit was
optimistic.

Adds a causality test that appends future rows and asserts earlier
values are unchanged.

Closes #0012
```

```
test(simulator): add queue position tests for all cancel assumptions

Covers pessimistic, optimistic, and proportional attribution against
one shared scenario, with hand-computed expected positions.

Closes #0039
```

```
docs(claude): add quant-researcher skill with leakage detection
```

```
chore(deps): pin torch to 2.1.1 CPU build

The CUDA build pulls roughly 2 GB of libraries the project never uses,
and Phase 2 models train on CPU within the runtimes recorded in 0029.
```

---

## What not to write

| Message | Problem |
|---|---|
| `updated stuff` | no type, no scope, says nothing |
| `feat: Added the reconstructor.` | past tense, capitalized, trailing period |
| `fix: bug` | which bug, where |
| `feat(data): implement reconstructor and add OFI and fix tests` | three changes; split it |
| `WIP` | not a commit worth keeping; squash before the PR |

---

## Using it as a hook

`.git/hooks/commit-msg`:

```sh
#!/bin/sh
python .claude/skills/git-workflow/scripts/validate_commit.py --quiet --file "$1"
```

```bash
chmod +x .git/hooks/commit-msg
```

The hook is local and not versioned, so each team member installs it. The
readiness check validates every commit in the branch regardless, so a missing
hook is caught before the PR.

---

## Related

- Validator: `scripts/validate_commit.py`
- Branching: `references/branching_model.md`
- PR requirements: `references/pr_checklist.md`

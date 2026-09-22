# Branching Model

---

## The shape

A single long-lived branch, `main`, always green. Every change arrives through
a short-lived branch that closes one issue.

```
main  ──o───────o───────o───────o──>
         \     / \     / \     /
          o───o   o───o   o───o
       feat/0007  feat/0008  test/0009
```

No `develop` branch. With a four-phase academic project and a small team, a
second integration branch adds a merge step and catches nothing that CI on
`main` does not.

---

## Branch naming

```
<type>/<NNNN>-<slug>
```

| Part | Rule |
|---|---|
| `type` | same vocabulary as commit types: `feat`, `fix`, `test`, `docs`, `refactor`, `perf`, `chore`, `ci`, `style` |
| `NNNN` | four-digit issue number, zero-padded |
| `slug` | lowercase, hyphenated, short |

```bash
git checkout -b feat/0007-lob-reconstructor
git checkout -b test/0008-queue-dynamics
git checkout -b docs/0025-ofi-methodology
```

The issue number is not decoration: `check_pr_readiness.py` resolves it to
`docs/issues/phase-N/NNNN-*.md`, and `generate_pr_description.py` pulls the
acceptance criteria from that file into the PR body. A branch without a
resolvable number loses both.

---

## One issue per branch

The backlog is already decomposed into 70 issues sized between one and eight
hours. That decomposition exists so a branch stays small enough to review
properly.

If a branch is growing past its issue, that usually means the issue was
underspecified. Split it, add the new issue to the backlog, and open two PRs.

**Exception:** issues explicitly marked parallelizable — 0008, 0009, 0010 after
0007, and 0039, 0040, 0041 after 0038 — are separate branches by design and
can be open simultaneously.

---

## Rebase, do not merge

```bash
git fetch origin
git rebase origin/main
```

Rebasing keeps history linear, which matters here because the provenance chain
records `git_commit` in artifact metadata. A linear history makes "which commit
produced this result" a single lookup rather than a graph walk.

`check_pr_readiness.py` fails when the branch is behind `main`.

**Never rebase a branch someone else has checked out.** In practice on this
project that means: never rebase after asking for review, unless you say so.

---

## Keeping commits meaningful

Squash noise before opening the PR:

```bash
git rebase -i origin/main     # not available in this environment; run it locally
```

A branch with `fix typo`, `fix typo again`, `actually fix it` should arrive as
one commit. A branch with `implement reconstructor` and `add queue tests`
should keep both — they are separate logical changes.

---

## Phase boundaries

Each entrega is tagged when its documentation issue merges:

```bash
git tag -a v0.1.0 -m "Entrega 1: data pipeline and OFI signal"
git tag -a v0.2.0 -m "Entrega 2: models and comparison"
git tag -a v0.3.0 -m "Entrega 3: fill simulator"
git tag -a v1.0.0 -m "Entrega 4: P&L breakdown and final report"
```

The tag is what a reader uses to reproduce the state the advisor was shown, and
what the reproduction sections in issues 0025, 0034, 0054, and 0070 point at.

---

## What never enters git

| Pattern | Why |
|---|---|
| `data/**` (except `README.md`) | regenerable; large; would freeze a snapshot the provenance chain expects to be live |
| `*.pt`, `*.pkl` | model checkpoints, large and binary |
| `.env` | may hold credentials; only `.env.example` is tracked |
| `venv/`, `__pycache__/` | environment artifacts |
| `mlruns/` | MLflow tracking, backed by a Docker volume |
| notebook outputs with embedded data | bloats diffs, leaks data into history |

`check_pr_readiness.py` blocks any file over 1 MB in the diff. Removing one
after the fact requires rewriting history, which is why the check runs before
the PR rather than after.

---

## After merge

```bash
git checkout main
git pull
git branch -d feat/0007-lob-reconstructor
```

Squash-merge on GitHub, so `main` gets one commit per issue and the history
reads as the backlog in order.

---

## Related

- Commit format: `references/commit_conventions.md`
- PR requirements: `references/pr_checklist.md`
- Readiness gate: `scripts/check_pr_readiness.py`
- Backlog: `docs/issues/INDEX.md`

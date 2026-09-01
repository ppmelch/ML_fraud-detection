---
name: git-workflow
description: Branching, commits, and pull requests for PAP-LOB-Trading — branch naming tied to issue numbers, conventional commit messages, PR readiness checks that run the full quality gate before a PR is opened, and generated PR descriptions that link back to the issue and its acceptance criteria. Use when starting work on an issue, preparing a commit, opening a pull request, or checking whether a branch is ready to merge. This skill governs process; the content of a code review belongs to code-reviewer.
---

# Git Workflow

Process for a four-phase project where every commit should be traceable to an
issue and every issue to a phase deliverable.

## The one rule everything else follows from

**A branch corresponds to one issue, and a PR closes it.**

The backlog has 70 numbered issues, each with acceptance criteria. A PR that
does not name an issue is either work nobody planned, or an issue whose
criteria nobody checked. Both are worth catching before the merge.

## Quick Start

```bash
# Check whether a branch is ready for a PR
python scripts/check_pr_readiness.py

# Generate a PR description from the issue and the diff
python scripts/generate_pr_description.py --issue 0007 --output pr.md

# Validate a commit message before committing
python scripts/validate_commit.py "feat(data): implement LOB reconstructor"
```

## Core Capabilities

### 1. PR Readiness Check (`check_pr_readiness.py`)

Runs the full gate before a PR is opened, so CI failures happen locally.

**Checks:**
- Branch is not `main`, and follows the naming convention
- Working tree is clean — no uncommitted changes
- Branch is rebased on the latest `main`
- Every commit message is valid
- `make lint` and `make test` pass
- Duplication and convention checks pass on changed files
- No large data files staged
- No secrets in the diff
- Issue number resolvable from the branch name

**Usage:**
```bash
python scripts/check_pr_readiness.py [--base main] [--skip-tests]
```

### 2. PR Description Generator (`generate_pr_description.py`)

Builds a PR body from the issue file and the diff, matching
`.github/PULL_REQUEST_TEMPLATE.md`.

**Features:**
- Reads the issue from `docs/issues/phase-N/NNNN-*.md`
- Copies its acceptance criteria into the PR as an unchecked list
- Summarizes changed files by area
- Infers the change type from the paths touched
- Emits `Closes #NNNN`

**Usage:**
```bash
python scripts/generate_pr_description.py [--issue 0007] [--base main] [--output pr.md]
```

### 3. Commit Message Validation (`validate_commit.py`)

Checks a message against the conventional-commit format this project uses.

**Checks:**
- Type is one of the permitted set
- Scope, when present, is a real project area
- Subject is imperative, lowercase, and under 72 characters
- No trailing period
- Body wrapped at 72 columns
- Issue reference present for `feat` and `fix`

**Usage:**
```bash
python scripts/validate_commit.py "feat(data): implement LOB reconstructor"
python scripts/validate_commit.py --file .git/COMMIT_EDITMSG
```

Suitable as a `commit-msg` hook.

## Reference Documentation

### Branching Model

`references/branching_model.md` — branch naming, the relationship between
branches and issues, when to rebase, and how phase boundaries are marked.

### Commit Conventions

`references/commit_conventions.md` — the type and scope vocabulary, with
examples drawn from this project's actual work.

### PR Checklist

`references/pr_checklist.md` — what a PR must satisfy before merge, and what
the reviewer is expected to verify beyond what tooling covers.

## Shared Instructions

Cross-cutting policy lives in `.claude/instructions/` so it has one home.
These bind this skill:

- [`agent-protocol.md`](../../instructions/agent-protocol.md) — committing and pushing are actions the user takes

Read them rather than relying on the summaries below.

---

## Workflow

### Starting an issue

```bash
git checkout main && git pull
git checkout -b feat/0007-lob-reconstructor
```

Read the issue first — `docs/issues/phase-1/0007-implement-lob-reconstructor.md`
carries the acceptance criteria the PR will be judged against.

### While working

```bash
git add src/data/lob_reconstruction.py tests/unit/test_lob_reconstruction.py
python .claude/skills/git-workflow/scripts/validate_commit.py \
    "feat(data): implement LOBReconstructor with FIFO queue tracking"
git commit -m "feat(data): implement LOBReconstructor with FIFO queue tracking"
```

### Before opening a PR

```bash
git fetch origin && git rebase origin/main
python .claude/skills/git-workflow/scripts/check_pr_readiness.py
python .claude/skills/git-workflow/scripts/generate_pr_description.py --output pr.md
gh pr create --body-file pr.md
```

## Best Practices

### Branches
- One issue per branch, named `<type>/<issue>-<slug>`
- Rebase on `main` rather than merging it in, so history stays linear
- Delete the branch after merge

### Commits
- Present tense, imperative: "implement", not "implemented" or "implements"
- One logical change per commit; a commit that needs "and" in its subject is two
- Reference the issue in the body for `feat` and `fix`

### What never gets committed
- Data files over 1 MB — `data/` is gitignored for this reason
- Model checkpoints — they belong in the artifact store
- `.env` — only `.env.example` is tracked
- Notebook outputs with embedded data
- Anything a secret scanner would flag

### PRs
- Small enough to review properly; a 2,000-line PR gets rubber-stamped
- Description generated from the issue, so acceptance criteria travel with it
- CI green before requesting review
- The issue's acceptance criteria checked, not assumed

## Common Commands

```bash
# Start
git checkout -b feat/0012-ofi-computation

# Validate before committing
python .claude/skills/git-workflow/scripts/validate_commit.py "feat(features): add OFI computation"

# Full pre-PR gate
python .claude/skills/git-workflow/scripts/check_pr_readiness.py

# Generate the description
python .claude/skills/git-workflow/scripts/generate_pr_description.py --issue 0012 --output pr.md

# Open it
gh pr create --title "feat(features): add OFI computation" --body-file pr.md

# After merge
git checkout main && git pull && git branch -d feat/0012-ofi-computation
```

## Troubleshooting

**Readiness check says the branch is behind main.** Rebase:
`git fetch origin && git rebase origin/main`. Resolve conflicts, then re-run.

**Commit validation rejects a message.** The output names the rule. The usual
cause is past tense ("implemented") or a subject over 72 characters.

**A large file was committed.** Removing it needs history rewriting —
`git filter-repo` or a fresh branch. Prevention is the `.gitignore` entry and
the readiness check.

**The issue number cannot be resolved.** The branch name must contain a
four-digit number matching a file under `docs/issues/phase-N/`. Rename the
branch, or pass `--issue` explicitly.

## Resources

- Branching: `references/branching_model.md`
- Commits: `references/commit_conventions.md`
- PR checklist: `references/pr_checklist.md`
- PR template: `.github/PULL_REQUEST_TEMPLATE.md`
- Issue backlog: `docs/issues/INDEX.md`
- Review skill: `.claude/skills/code-reviewer/`

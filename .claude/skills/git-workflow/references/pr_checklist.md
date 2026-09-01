# PR Checklist

What a PR must satisfy, and who checks each part.

---

## Before opening

```bash
git fetch origin && git rebase origin/main
python .claude/skills/git-workflow/scripts/check_pr_readiness.py
```

The readiness check covers:

- [ ] Branch is not `main` and follows `<type>/<NNNN>-<slug>`
- [ ] The issue number resolves to a file under `docs/issues/phase-N/`
- [ ] Working tree clean
- [ ] Rebased on `main`
- [ ] Every commit message valid
- [ ] No file over 1 MB in the diff
- [ ] No secrets in the diff
- [ ] No duplicate implementation of a canonical concept
- [ ] No convention errors (causality findings are blocking)
- [ ] `make lint` passes
- [ ] `make test` passes

CI runs the same checks. Running them locally turns a red build into a
thirty-second wait.

---

## The description

```bash
python .claude/skills/git-workflow/scripts/generate_pr_description.py --output pr.md
gh pr create --body-file pr.md
```

The generated body carries the issue's acceptance criteria as an unchecked
list. **Check them individually against the diff**, rather than assuming the
implementation covered them. That list is what the issue was written for.

A criterion that does not apply gets struck through with a one-line reason, not
silently deleted.

---

## What the author confirms

- [ ] Every acceptance criterion from the issue is met or explicitly waived
- [ ] Docstrings on public functions and classes, with types
- [ ] Tests exist, and would fail if the code were wrong
- [ ] `docs/CONTEXT.md` updated if a new domain term appeared
- [ ] `data/README.md` updated if a new artifact schema appeared
- [ ] The design document updated if implementation deviated from it
- [ ] `CHANGELOG.md` updated
- [ ] New artifacts write metadata with `upstream_artifacts`

---

## What the reviewer verifies

Tooling covers duplication, conventions, lint, and tests. The reviewer covers
what tooling cannot see.

### Architecture
- [ ] Is the abstraction right, or does it fit only this one caller?
- [ ] Does it belong in this module, per the layering rules?
- [ ] Is the repository cleaner after this change than before?

### Correctness
- [ ] Causality: every rolling window trailing, every target forward-shifted
- [ ] Leakage: scaler fitted on training only, validation disjoint from test
- [ ] No threshold or hyperparameter selected on test-fold performance
- [ ] Sign conventions correct, both directions
- [ ] Float comparisons use a shared tolerance, not `== 0`

### Tests
- [ ] **Would each test fail if the code were wrong?**
- [ ] Invariants asserted, not only the happy path
- [ ] Error paths tested in both strict and non-strict modes
- [ ] `pytest.approx` on float comparisons

### Pipeline
- [ ] Scripts validate before writing, and exit non-zero on failure
- [ ] Metadata records the full upstream chain

The full ordering is in
[`.claude/skills/code-reviewer/references/review_checklist.md`](../../code-reviewer/references/review_checklist.md).

---

## Size

A PR should be reviewable in one sitting. Roughly:

| Lines changed | Expectation |
|---|---|
| under 200 | reviewable properly |
| 200–500 | acceptable for a feature with tests |
| over 500 | justify it, or split |
| over 1000 | split, unless it is generated or moved files |

The 70-issue backlog is sized to keep PRs in the first two bands. A PR that
overflows usually means the issue was underspecified — split it and add the new
issue to the backlog.

---

## Merging

- Squash-merge, so `main` gets one commit per issue
- The squash subject follows the commit convention
- Delete the branch
- Confirm the issue closed; reopen and fix if a criterion was missed

---

## Phase-closing PRs

The documentation issue that closes each phase — 0015, 0035, 0055, 0070 —
carries additional exit checks:

- [ ] `make lint` and `make test` pass on the full repository, not just the diff
- [ ] Coverage meets the per-module targets from every issue in the phase
- [ ] Every issue in the phase closed, or explicitly deferred with a reason
- [ ] Every artifact present, with its provenance chain resolving end to end
- [ ] `README.md` status table updated
- [ ] Reproduction steps verified on a clean checkout

Tag the release after merging:

```bash
git tag -a v0.1.0 -m "Entrega 1: data pipeline and OFI signal"
git push origin v0.1.0
```

---

## Related

- Readiness gate: `scripts/check_pr_readiness.py`
- Description generator: `scripts/generate_pr_description.py`
- Review depth: `.claude/skills/code-reviewer/references/review_checklist.md`
- Template: `.github/PULL_REQUEST_TEMPLATE.md`

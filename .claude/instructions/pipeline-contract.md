# Pipeline Contract

**Binding on: data-engineer, ml-engineer, code-reviewer, and every script in
`scripts/`.**

---

## The rule everything else follows from

**Validate, then write. Never the reverse.**

A file that failed validation must not reach disk. It will be picked up
downstream, and the failure will surface as a strange result rather than an
error — which is far more expensive to diagnose than a loud exit.

```python
report = validate_schema(df, CONTRACTS["ofi_signal"])
if not report["passed"]:
    print_report(report)
    sys.exit(1)          # do not write

df.to_csv(output_path, index=False)
write_metadata(output_path, ..., upstream_artifacts=[...])
```

---

## The chain

```
raw events -> cleaned -> LOB snapshots -> features -> predictions -> executions -> P&L
```

| Stage | Produces | Owner issue |
|---|---|---|
| raw events | `data/raw/sample_events_synthetic_v1.csv` | 0001 |
| cleaned | `data/processed/raw_data_cleaned_v1.csv` | 0005 |
| LOB snapshots | `data/processed/lob_reconstructed_v1.csv` | 0011 |
| OFI signal | `data/processed/ofi_signal_v1.csv` | 0015 |
| features / labels | `data/processed/model_features_v1.csv`, `model_labels_v1.csv` | 0016, 0017 |
| predictions | `data/processed/{ofi,deeplab}_predictions_v1.csv` | 0024, 0031 |
| executions | `data/backtest/execution_results_latency_*ms_*_v1.csv` | 0045, 0048 |
| P&L | `data/backtest/waterfalls_v1.json` | 0064 |

Schemas live in
`.claude/skills/data-engineer/references/data_contracts.md`, encoded
executably in `validate_schema.py`. The two must agree — change both in the
same commit.

---

## Every artifact carries metadata

`foo.csv` has `foo_metadata.json` beside it. Required fields:

| Field | Why |
|---|---|
| `description` | what this is, in one line |
| `generated_at` | ISO 8601 **with timezone** — naive timestamps break staleness detection |
| `generated_by` | the script that produced it |
| `upstream_artifacts` | paths this derives from; `[]` for a root |
| `config` | every parameter affecting the output, including seeds |
| `columns` | dtype, units, meaning per column |
| `num_rows` | cheap cross-check against the file |

`git_commit` and library versions are strongly recommended. A result that
cannot be reproduced because the code moved no longer supports its conclusion.

**Staleness:** when an upstream artifact's `generated_at` is later than its
downstream's, the downstream was built from data that has since been replaced.
Fix it by regenerating the downstream — never by editing metadata to point
elsewhere, which preserves the appearance of a valid chain and destroys its
meaning.

---

## Missing values

| Situation | Rule |
|---|---|
| Absent book level | `NaN` — `0` collides with a real zero volume, `-1` poisons a feature matrix |
| Leading NaN from a rolling window | preserve; filling fabricates a book that never existed |
| Trailing NaN from a forward target | preserve; filling invents future prices |
| Unfilled order's slippage | `None`, never `0` — zero enters averages as a perfect execution |

Document the policy in metadata with per-column counts. Let each consumer drop
what it needs for its own horizon; deciding for them discards rows that are
usable at a shorter horizon.

Summing level columns uses `skipna=True`, so padding drops out rather than
contributing zero.

---

## Transformations are recorded, never silent

Every cleaning step logs `step_name`, `rows_before`, `rows_after`,
`rows_affected`, and `details`. Rows dropped are counted and reported.

**Winsorize rather than drop** extreme quantities — an unusually large order is
usually real, and removing it changes book depth. Record the bounds so the
effect on results is auditable.

**Cleaning is idempotent:** running it twice changes nothing the second time.
Assert it.

---

## Order-book specifics that bite

- **Stable sorts only.** Events sharing a timestamp carry FIFO queue meaning in
  their order. A non-stable sort permutes them and silently changes every queue
  position downstream.
- **Timestamps are non-decreasing, not strictly increasing.** Several events per
  millisecond is normal; `sequence_number` carries the strict ordering.
- **Float tolerance, not equality.** `quantity_remaining` accumulates
  subtraction error. Use a shared `QUANTITY_EPSILON`, defined once.
- **Round prices to the tick grid on ingestion**, so dictionary lookups are
  exact and no level exists that can never match.
- **Cached aggregates drift.** `PriceLevel.total_volume` is maintained
  incrementally for speed; something must independently recompute and compare,
  or a single missed decrement inflates depth for the rest of the session.

---

## Flat columns, not nested

`bid_price_1..10` rather than a stringified list in one cell. The list form is
compact and unusable: every consumer parses it, pandas reads an object column,
and the Phase 2 model input wants the flat shape anyway.

---

## Naming

```
<concept>_<version>.csv                          lob_reconstructed_v1.csv
<concept>_<version>_metadata.json                lob_reconstructed_v1_metadata.json
execution_results_latency_{ms}ms_{model}_v{n}.csv
```

The latency convention is load-bearing: Phase 4 discovers those files by glob
rather than maintaining a list that goes stale.

---

## Auditing

```bash
for f in data/processed/*.csv data/backtest/*.csv; do
    python .claude/skills/data-engineer/scripts/check_provenance.py "$f" || echo "BROKEN: $f"
done
```

Phase exit checks (issues 0035, 0055, 0070) require this to pass before a phase
is considered complete.

---

## Related

- Schemas: `.claude/skills/data-engineer/references/data_contracts.md`
- Metadata detail: `.claude/skills/data-engineer/references/provenance_chain.md`
- LOB pitfalls: `.claude/skills/data-engineer/references/lob_data_pitfalls.md`
- Live schemas: `data/README.md`

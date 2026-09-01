---
name: data-engineer
description: Data pipeline engineering for limit order book research — schema contracts, data quality profiling, artifact provenance chains, and the metadata conventions that make a result traceable back to the raw events that produced it. Use when adding a pipeline stage, defining or changing a data contract, debugging a schema mismatch, investigating a data quality problem, or tracing where a number came from. This skill governs how data moves and how its lineage is recorded; it does not analyze signals — pair it with quant-researcher for that.
---

# Data Engineer

Pipeline engineering for PAP-LOB-Trading. The pipeline runs raw events →
cleaned → reconstructed LOB → features → predictions → executions → P&L. Seven
links, each one an opportunity to lose track of where a number came from.

## The one rule everything else follows from

**Validate, then write. Never the reverse.**

A pipeline stage that writes a file which failed validation is worse than one
that writes nothing, because the bad file will be picked up downstream and the
failure will surface as a strange result rather than an error. Every script in
this project checks its output and exits non-zero rather than producing a
corrupt artifact.

## Quick Start

```bash
# Validate a file against its declared contract
python scripts/validate_schema.py data/raw/sample_events_synthetic_v1.csv \
    --contract raw_events

# Profile data quality
python scripts/profile_dataset.py data/processed/lob_reconstructed_v1.csv \
    --output data/processed/quality_report.json

# Trace an artifact back to raw events
python scripts/check_provenance.py data/backtest/backtest_results_full_v1.csv
```

## Core Capabilities

### 1. Schema Validation (`validate_schema.py`)

Checks a file against the contract declared for its stage. Contracts live in
`references/data_contracts.md` and are encoded in the script.

**Checks performed:**
- Required columns present, with the declared dtype
- No nulls in columns marked critical
- Categorical columns contain only their permitted values
- Numeric ranges respected (positive prices, bounded OFI, and so on)
- Timestamps non-decreasing where the contract requires it
- Unexpected columns reported as warnings, not errors

**Usage:**
```bash
python scripts/validate_schema.py <file.csv> --contract <name> [--strict]
```

Available contracts: `raw_events`, `cleaned_events`, `lob_snapshots`,
`ofi_signal`, `model_features`, `model_labels`, `predictions`,
`execution_results`.

### 2. Data Quality Profiling (`profile_dataset.py`)

Goes beyond the hard schema gate: gaps, duplicates, orphan references,
outliers, and distribution shape.

**Features:**
- Null analysis per column, with percentages
- Duplicate detection at row level and on declared key columns
- Temporal gap detection with a configurable threshold
- Referential integrity for event streams (cancel/execute referencing a prior add)
- Outlier flagging by IQR with a 3× fence, appropriate for heavy-tailed data
- Severity classification: `clean`, `warnings`, `errors`

**Usage:**
```bash
python scripts/profile_dataset.py <file.csv> [--gap-threshold-ms 60000] [--output report.json]
```

### 3. Provenance Chain Verification (`check_provenance.py`)

Walks the `upstream_artifacts` chain recorded in each metadata file, from any
artifact back to the raw events.

**Features:**
- Resolves each link and reports missing files
- Detects cycles
- Reports the full chain as a tree
- Flags artifacts whose upstream file changed after they were generated
- Verifies every artifact has a companion metadata file

**Usage:**
```bash
python scripts/check_provenance.py <artifact.csv> [--max-depth 10]
```

## Reference Documentation

### Data Contracts

`references/data_contracts.md` — the schema of every artifact the pipeline
produces: column names, dtypes, units, permitted values, and which columns are
critical. This is the authoritative reference when adding a stage.

### Provenance Chain

`references/provenance_chain.md` — the metadata convention, what
`upstream_artifacts` must record, and how to trace a Phase 4 number back to
the model checkpoint and seed that produced it.

### LOB Data Pitfalls

`references/lob_data_pitfalls.md` — problems specific to order book data:
crossed books, orphan cancels, timestamp resolution collisions, padding
conventions for variable-depth books, and why aggregate volume is not enough
to reconstruct queue position.

## Shared Instructions

Cross-cutting policy lives in `.claude/instructions/` so it has one home.
These bind this skill:

- [`pipeline-contract.md`](../../instructions/pipeline-contract.md) — validate-before-write, metadata, missing-value policy, LOB specifics

Read them rather than relying on the summaries below.

---

## Domain Context

**Read `docs/CONTEXT.md` first** for order book, queue position, and event
type definitions. **Read `data/README.md`** for the current schema of every
artifact on disk.

## Pipeline Workflow

### Adding a stage

1. Define the output contract in `references/data_contracts.md` before coding
2. Implement the transformation in `src/`
3. Validate the output against the contract
4. Write the metadata file with `upstream_artifacts`
5. Gate the write on validation passing
6. Update `data/README.md` with the new schema

### Debugging a downstream problem

```bash
# Where did this file come from?
python scripts/check_provenance.py <suspect_file.csv>

# Is it structurally legal?
python scripts/validate_schema.py <suspect_file.csv> --contract <name>

# Is it trustworthy?
python scripts/profile_dataset.py <suspect_file.csv>
```

Work upstream until a stage passes, then the problem is in the stage after it.

## Best Practices

### Schema
- Declare the contract before writing the transformation
- Keep column names in `snake_case`, self-describing, with units in the metadata
- Flat columns over nested structures — `bid_price_1..10`, not a stringified list
- Fix column order and assert it, so consumers can rely on position if needed

### Padding and missing values
- `NaN` for absent book levels, never `0` or `-1` — those collide with legal values
- Preserve `NaN` from rolling windows and forward targets; never fill them
- Document the NaN policy in metadata with per-column counts
- Let each consumer drop what it needs rather than deciding for them

### Metadata
- Every artifact gets a companion `*_metadata.json`
- Record `generated_at`, `source_file`, `upstream_artifacts`, and a `columns` block
- Record the configuration that produced it — seeds, parameters, versions
- Metadata is what makes a result reproducible six weeks later

### Idempotence
- Running a cleaning stage twice must change nothing the second time
- Assert it in a test: `clean(clean(df)) == clean(df)`

### What not to do
- Do not modify data silently — record every transformation in a log
- Do not drop rows without counting and reporting them
- Do not fill missing values with plausible substitutes
- Do not write an output file that failed validation
- Do not hardcode paths; read from `.env`

## Common Commands

```bash
# Full validation pass over the pipeline
python scripts/validate_schema.py data/raw/sample_events_synthetic_v1.csv --contract raw_events
python scripts/validate_schema.py data/processed/raw_data_cleaned_v1.csv --contract cleaned_events
python scripts/validate_schema.py data/processed/lob_reconstructed_v1.csv --contract lob_snapshots
python scripts/validate_schema.py data/processed/ofi_signal_v1.csv --contract ofi_signal

# Quality profile with a report
python scripts/profile_dataset.py data/raw/sample_events_synthetic_v1.csv \
    --output data/processed/data_quality_report_v1.json

# Provenance audit of every artifact
for f in data/processed/*.csv data/backtest/*.csv; do
    python scripts/check_provenance.py "$f" || echo "BROKEN CHAIN: $f"
done
```

## Troubleshooting

**Schema validation fails on dtype.** pandas infers `int64` for a column of
whole-number floats. Use `pd.api.types.is_integer_dtype` rather than comparing
dtype strings, and cast explicitly on write.

**Row counts differ between features and labels.** They must be row-aligned.
Check whether one was written after a `dropna` that the other did not receive.

**Provenance chain breaks.** Either a metadata file is missing
`upstream_artifacts`, or an upstream file was regenerated and moved. Regenerate
downstream artifacts rather than editing metadata to point at the new file.

**Orphan cancel/execute events.** A `cancel` referencing an `order_id` that
never appeared in an `add`. Usually means the event stream starts mid-session.
The cleaning stage drops them and records the count.

## Resources

- Data contracts: `references/data_contracts.md`
- Provenance convention: `references/provenance_chain.md`
- LOB-specific pitfalls: `references/lob_data_pitfalls.md`
- Current schemas: `data/README.md`
- Architecture rules: `.claude/Agent.md`
- Related issues: `docs/issues/phase-1/0001` through `0011`

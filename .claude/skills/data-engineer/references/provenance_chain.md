# Provenance Chain

How to record where an artifact came from, and how to trace a number back to
the raw events that produced it.

---

## Why this matters more than it sounds

By Phase 4 the chain is seven links deep:

```
raw events -> cleaned -> LOB snapshots -> features -> predictions -> executions -> P&L
```

When a P&L number looks surprising, the first question is which inputs
produced it: which model checkpoint, which seed, which cleaning run, which
version of the OFI depth parameter. Without recorded lineage that question
takes an afternoon. With it, it takes a minute.

The second thing lineage catches is staleness: an upstream file regenerated
after its downstream artifact was built. The downstream file then describes a
version of the data that no longer exists, and nothing about its contents
reveals that.

---

## The metadata file

Every artifact `foo.csv` has a companion `foo_metadata.json`.

```json
{
  "description": "OFI signal with forward-return targets",
  "generated_at": "2026-08-23T14:32:11+00:00",
  "generated_by": "scripts/02_generate_ofi_signal.py",
  "git_commit": "a33c898",

  "upstream_artifacts": [
    "data/processed/lob_reconstructed_v1.csv"
  ],

  "config": {
    "depth": 5,
    "windows": [1, 3, 5],
    "horizons": [1, 3, 5],
    "random_seed": 42
  },

  "nan_policy": {
    "description": "leading NaN from rolling windows and trailing NaN from forward targets are preserved",
    "counts": {"ofi_3": 2, "ofi_5": 4, "price_movement_5": 5}
  },

  "num_rows": 1000,
  "time_span_ms": 3642500,

  "columns": {
    "ofi_1": {
      "dtype": "float64",
      "units": "dimensionless, [-1, 1]",
      "description": "(bid_vol - ask_vol) / (bid_vol + ask_vol) over top 5 levels"
    }
  },

  "versions": {
    "python": "3.11.7",
    "pandas": "2.2.0",
    "torch": "2.1.1"
  }
}
```

---

## Required fields

| Field | Why |
|---|---|
| `description` | what this artifact is, in one line |
| `generated_at` | ISO 8601 with timezone — naive timestamps break staleness detection |
| `generated_by` | the script that produced it |
| `upstream_artifacts` | list of paths this derives from; `[]` for a root artifact |
| `config` | every parameter that affects the output, including seeds |
| `columns` | dtype, units, and meaning per column |
| `num_rows` | cheap consistency check against the file itself |

`git_commit` and `versions` are strongly recommended: a result that cannot be
reproduced because the code moved is not reproducible.

---

## Accepted shapes for `upstream_artifacts`

`check_provenance.py` tolerates three forms, because different stages
naturally record different things:

```json
"upstream_artifacts": ["data/processed/lob_reconstructed_v1.csv"]

"upstream_artifacts": [
  {"path": "data/processed/features_v1.csv", "role": "features"},
  {"path": "data/models/deeplab_h5_fold3.pt", "role": "model"}
]

"upstream_artifacts": {
  "features": "data/processed/features_v1.csv",
  "labels": "data/processed/labels_v1.csv"
}
```

Prefer the second form once an artifact has more than one input — the role
label is what makes the chain readable.

---

## Root artifacts

A root artifact declares `"upstream_artifacts": []`. In this project that is
only `data/raw/sample_events_synthetic_v1.csv`, whose metadata records the
generator seed instead.

An artifact with no `upstream_artifacts` key at all is treated as a root by
`check_provenance.py`, which is usually a mistake — say `[]` explicitly.

---

## Staleness

`check_provenance.py` compares each upstream artifact's `generated_at` against
its downstream's. If upstream is newer, the downstream was built from a version
of the data that has since been replaced.

**The fix is to regenerate the downstream artifact.** Editing metadata to point
at the new file preserves the appearance of a valid chain and destroys its
meaning.

When an upstream metadata file lacks `generated_at`, the script falls back to
file mtime and reports the finding as "possibly stale" — mtime is unreliable
across copies and checkouts, which is exactly why `generated_at` is required.

---

## Writing metadata

A small helper keeps the shape consistent across stages:

```python
from datetime import datetime, timezone
import json, subprocess
from pathlib import Path


def write_metadata(artifact_path, description, upstream, config, df, columns):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        commit = None

    meta = {
        "description": description,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": str(Path(__file__).relative_to(Path.cwd())),
        "git_commit": commit,
        "upstream_artifacts": upstream,
        "config": config,
        "num_rows": len(df),
        "columns": columns,
    }
    Path(artifact_path).with_name(
        Path(artifact_path).stem + "_metadata.json"
    ).write_text(json.dumps(meta, indent=2))
```

---

## Validation gate

The pattern every pipeline script follows:

```python
report = validate_schema(df, CONTRACTS["ofi_signal"])
if not report["passed"]:
    print_report(report)
    sys.exit(1)          # do not write

df.to_csv(output_path, index=False)
write_metadata(output_path, ...)
```

Validate, then write. A file that failed validation must not reach disk.

---

## Auditing the whole pipeline

```bash
for f in data/processed/*.csv data/backtest/*.csv; do
    python .claude/skills/data-engineer/scripts/check_provenance.py "$f" \
        || echo "BROKEN: $f"
done
```

Phase exit checks (issues 0035, 0055, 0070) require this to pass before a
phase is considered complete.

---

## Related

- Verification tool: `scripts/check_provenance.py`
- Schemas: `references/data_contracts.md`
- Phase exit checks: `docs/issues/phase-2/0035`, `phase-3/0055`, `phase-4/0070`

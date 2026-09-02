# Pre-unsupervised snapshot

Frozen copy of the **supervised India fraud-detection** codebase, taken
2026-09-01 immediately before the migration to unsupervised US anomaly
detection (see `docs/planning/unsupervised-us-migration.md`).

- Git tag: `pre-unsupervised-migration-2026-09-01`
- Contains: `backend/src/`, `scripts/`, `tests/`, `frontend/{index.html,css,js}`,
  `backend/requirements.txt`
- Not included (unchanged / too large / in git history): `data/`,
  `frontend/data/*.json` GeoJSON, `backend/artifacts/`, docs

This directory is reference-only. Nothing imports from it.

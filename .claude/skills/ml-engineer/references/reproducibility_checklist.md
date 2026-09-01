# Reproducibility Checklist

A result that cannot be reproduced cannot be defended. Everything below must
hold before a training sweep starts.

---

## Seeding

Every source of randomness, seeded together:

```python
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

Missing any one makes a run irreproducible in a way that is hard to attribute
later: the weights differ, and nothing says why.

- [ ] Python `random` seeded
- [ ] numpy seeded (prefer `np.random.default_rng(seed)` — it does not share
      global state, so one library reseeding cannot change another's stream)
- [ ] torch seeded, CPU and CUDA
- [ ] DataLoader shuffling seeded through an explicit `torch.Generator`, not
      left to global state
- [ ] Per-fold seeds derived deterministically from a base seed and fold id

```python
g = torch.Generator().manual_seed(seed)
loader = DataLoader(dataset, batch_size=32, shuffle=True, generator=g)
```

---

## CUDA determinism

Only relevant when a GPU is in use. cuDNN selects algorithms by benchmark
unless told not to, and those algorithms are not all deterministic.

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

- [ ] Both flags set before training
- [ ] The cost acknowledged: determinism can be slower

CPU training is deterministic by default, which is one more reason this
project targets CPU (ADR 0001).

---

## Recorded configuration

A seed that is not written down is not a seed.

- [ ] `random_seed` recorded in the run metadata
- [ ] Every hyperparameter recorded, including those left at defaults
- [ ] Model architecture config recorded alongside the weights
- [ ] Data file paths and their `generated_at` recorded
- [ ] Fold specification file recorded
- [ ] `git_commit` recorded

The rule of thumb: if changing it would change the output, record it.

---

## Version pinning

- [ ] `requirements.txt` pins exact versions with `==`
- [ ] Python version recorded in run metadata
- [ ] torch, numpy, pandas, scikit-learn versions recorded
- [ ] Docker image tag recorded when the run happened in a container

Different library versions produce different results for the same seed. A run
from six weeks ago that cannot be reproduced because a dependency floated is a
run that no longer supports its conclusion.

Docker is the stronger guarantee here — `make docker-build` produces the same
environment on every machine, which a local venv does not.

---

## Verification, not assumption

Run `check_reproducibility.py` before the sweep. It proves what the checklist
above asserts:

```bash
python .claude/skills/ml-engineer/scripts/check_reproducibility.py
```

It builds and steps a small model twice under the same seed and compares
weights. Identical seeds producing different weights means seeding is
incomplete, and the script says which check failed.

For a real run, the equivalent test is a reduced-scale sweep — two folds,
three epochs — asserted to reproduce byte-identically. Issue 0029 requires
this to run in CI.

---

## Data reproducibility

Model reproducibility is worthless if the inputs drift.

- [ ] The synthetic data generator accepts `--seed` and defaults to 42
- [ ] Running it twice with the same seed produces byte-identical output
- [ ] The seed is recorded in the data's metadata
- [ ] Fold definitions are persisted and loaded, never regenerated from
      parameters that might drift

---

## Environment variables

- [ ] `RANDOM_SEED` in `.env.example` with a stated default
- [ ] The training script reads it rather than hardcoding
- [ ] The value actually used is recorded in run metadata, not just the env default

An env var that differs between machines is a silent source of divergence.
Recording the resolved value closes that gap.

---

## Common causes of irreproducibility

| Symptom | Usual cause |
|---|---|
| Weights differ across runs | DataLoader shuffle unseeded |
| Results differ across machines | library version drift; use Docker |
| Results differ on GPU only | `cudnn.benchmark` left true |
| Small numeric differences | non-deterministic reduction order on GPU |
| Data differs across runs | generator seed not passed through |
| Cannot load an old checkpoint | config not saved alongside weights |

---

## The reproduction section

Every methodology document (issues 0025, 0034, 0054, 0070) carries a
reproduction section giving the exact command sequence, paths, seeds, runtimes,
and versions.

**Verify it by following it on a clean checkout.** That is the only way to
catch a missing file or an undocumented manual step, and it is required before
each entrega is delivered.

---

## Related

- Verification tool: `scripts/check_reproducibility.py`
- Training protocol: `references/training_protocol.md`
- Provenance: `.claude/skills/data-engineer/references/provenance_chain.md`
- Environment: `.claude/skills/environment-setup/`

# Troubleshooting

Symptom → cause → fix, for the failures this project actually produces. Match
the symptom to the real error text before acting; guessing from a paraphrase is
how an afternoon disappears.

## Known facts about this machine and repo

Confirmed by running the commands, not assumed. Re-verify each session — these
change when someone installs something — but expect them.

| Fact | Consequence |
|---|---|
| Host is **WSL2**, repo lives at `/mnt/c/.../OneDrive/...` with spaces in the path | Bind mounts are slow; OneDrive sync can corrupt files that are open |
| **`python` does not exist**, only `python3` | Anything invoking bare `python` fails before doing any work |
| System Python is **3.12**; the project targets **3.11** | `torch==2.1.1` publishes no cp312 wheel, so a local `pip install -r requirements.txt` fails |
| **Docker Desktop is installed on Windows, WSL integration is off** | `docker` resolves to a shim under `/mnt/c/Program Files/...` that only prints an advisory. `command -v docker` succeeds while nothing works — check `docker info`, never mere presence |
| `output/` is **absent** but `docker-compose.yml` bind-mounts it | Docker creates it as root; the non-root `researcher` user in the runtime stage then cannot write |
| `src/*/` contain **no `__init__.py`** | `import src` fails, including the runtime image's own `CMD` |
| `scripts/` has **no `.py` files**; `tests/` has **no `test_*.py`** | `make lob-reconstruct`, `make run-backtest`, and `make lint` have nothing to work on; `pytest` collects zero tests and passes vacuously |

The last three are `[code]`, not environment. Report them; do not paper over
them by generating empty modules.

## Failure catalog

| Symptom | Cause | Fix |
|---|---|---|
| `The command 'docker' could not be found in this WSL 2 distro` | WSL integration off in Docker Desktop | Docker Desktop → Settings → Resources → WSL Integration → enable the distro → Apply & Restart |
| `Cannot connect to the Docker daemon` | Docker Desktop not running | Start it; wait for the whale icon to settle |
| `ERROR: for jupyter ... dependency failed to start` | mlflow's healthcheck never passed | `docker compose logs mlflow`; confirm the `mlflow_db` volume is writable. Fix mlflow first — jupyter is gated on it |
| `port is already allocated` (8888 / 5000) | Another Jupyter or MLflow already running | Change `JUPYTER_PORT` / `MLFLOW_PORT` in `.env`, **not** in `docker-compose.yml` |
| `ModuleNotFoundError: No module named 'src'` | Missing `PYTHONPATH` locally, or missing `__init__.py` | Local: `export PYTHONPATH="$PWD"`. In Docker it is already set — if it still fails, the `__init__.py` files genuinely do not exist (`[code]`) |
| `No matching distribution found for torch==2.1.1` | Python 3.12 | Use Docker, or a real 3.11 interpreter. Do not bump the pin to make it install |
| `Permission denied` writing `output/` or `data/` | Docker created the directory as root | `sudo chown -R $(id -u):$(id -g) output data`, and run `setup_bootstrap.sh` before the first `up` next time |
| `make: python: command not found` | A recipe calls bare `python` | Use `python3` explicitly and fix the recipe in the `Makefile` |
| Edited `requirements.txt`, package still absent | The deps layer is cached on that file | `make docker-build` |
| Edited `src/`, container shows old code | You are in the `runtime` stage, which copies rather than mounts | Rebuild, or use the `jupyter` (`dev`) service, which bind-mounts |
| `pip` crawls, or wheels arrive corrupted | OneDrive syncing the tree mid-install | Warn the user; moving the repo outside OneDrive is the real fix |
| First build takes many minutes | `/mnt/c` bind mounts plus a full torch install | Expected. Say so *before* starting it, not after they panic |
| `pytest` reports `collected 0 items` and exits green | No test files exist | `[code]`. Say explicitly that this is not a passing test suite |

## Destructive commands — confirm first, every time

| Command | Destroys |
|---|---|
| `docker compose down -v` | The `mlflow_db` volume — every recorded experiment run |
| `make docker-clean` | The same, via `down -v` |
| `make clean` | Caches, `htmlcov/`, `.coverage`, build artifacts (recoverable, but say so) |
| `docker system prune -a` | Every image on the machine, including other projects' |

Ask before running any of these, even when the user's request seems to imply
it. "Clean it up" does not authorize deleting the experiment history.

## Diagnostic order

When several things are broken at once, fix in this order — later items produce
misleading errors while an earlier one is unresolved:

1. Docker engine reachable (`docker info` exits 0)
2. `.env` and working directories exist and are writable
3. Images build
4. `mlflow` healthy
5. `jupyter` healthy
6. Tests run
7. Lint

A failure at step 6 while step 4 is red tells you nothing. Do not investigate
it.

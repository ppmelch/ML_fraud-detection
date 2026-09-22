---
name: environment-setup
description: Environment setup, repair, and verification for this project — Docker and docker compose, Python and virtualenvs, dependency installs, .env configuration, Jupyter and MLflow services, ports, permissions, pytest and CI parity. Covers the from-absolute-zero case where the user has never opened a terminal or an editor and is driving entirely through the Claude interface. Use when someone clones the repo for the first time, says "it doesn't run", hits a failing pip install or docker compose up, an import error on src/, an occupied port, or a service that will not start; and when confirming an environment is genuinely reproducible. This skill governs the environment; it does not write trading, modelling, or backtest logic.
---

# Environment Setup

Getting a person from "I have a repository and no idea what to do with it" to a
green `make docker-test`, without requiring them to already understand Docker,
Python packaging, or a terminal.

## The one rule everything else follows from

**An environment is proven by a command that ran, not by a file that exists.**

A `.env` on disk does not mean configuration is correct. A built image does not
mean the service is healthy. An installed package does not mean it imports.
Every claim of "ready" in this skill is tied to a command whose output you have
actually seen. If you did not run it, you do not know.

## Who you are talking to

Assume the user may have **only the Claude interface** — no VS Code, no
familiarity with terminals, no mental model of what a container is. This changes
the shape of the help:

- **You run the commands, not them.** You have a shell. Never hand someone a
  block of eight commands to paste; run them yourself and report what happened.
- **The human is needed only for what a shell cannot reach**: a GUI installer,
  a settings toggle, an admin prompt, a browser click. Those are the *only*
  moments you stop and ask.
- **Name things once, plainly.** "Docker runs the project in a sealed box so it
  behaves the same on every machine" — then move on. No lecture.
- **One step at a time, with a checkpoint.** A beginner cannot tell which of
  eight pasted commands failed. You can. Keep it that way.

Never assume they know what a terminal, a path, a port, a venv, or a container
is. Never make them feel slow for not knowing.

## Division of labor

Every finding sorts into exactly one bucket. `scripts/setup_doctor.sh` tags its
output this way, and you should keep the same vocabulary in what you tell them:

| Tag | Meaning | Who acts |
|---|---|---|
| `[claude]` | Fixable from the shell right now | You, immediately, without asking |
| `[you]` | Needs a GUI, an installer, or admin rights | The human — give exact click paths, then re-verify |
| `[code]` | Not an environment fault; project code is unwritten | Nobody, here — report it and hand it to the architect |

Misfiling a `[code]` item as an environment problem is the classic failure mode:
it sends you inventing `src/__init__.py` files and placeholder scripts to make a
symptom disappear. Do not.

## Quick Start

```bash
bash scripts/setup_doctor.sh        # diagnose — read-only, changes nothing
bash scripts/setup_bootstrap.sh     # prepare — additive and idempotent
bash scripts/setup_doctor.sh        # confirm what actually changed
```

Then choose a route. Docker is preferred; the local venv is the fallback.

## Core Capabilities

### 1. Diagnosis (`scripts/setup_doctor.sh`)

Read-only. Installs nothing, changes nothing, deletes nothing — safe to run at
any moment, including on a machine you know nothing about. Checks the Python
version against what the project pins, whether Docker is genuinely usable (not
merely present on `PATH`), configuration, working directories and their
writability, whether project code exists at all, running services, port
collisions, and host hazards. Exits non-zero when a blocker remains.

**Always run this first.** Every other action in this skill is a response to
something it reported.

### 2. Preparation (`scripts/setup_bootstrap.sh`)

Does exactly the `[claude]` items: creates `.env` from `.env.example` (never
overwriting an existing one), creates the data and output directories *before*
Docker can create them as root, restores lost `+x` bits. Additive and
idempotent — running it twice is a no-op and it says so.

### 3. The Docker route (preferred)

```bash
bash scripts/setup_bootstrap.sh
make docker-build
make docker-up          # Jupyter on :8888, MLflow on :5000
make docker-test
```

Checkpoints, in this order — later ones are meaningless if an earlier one fails:

1. `docker compose ps` shows **`mlflow` healthy**. Compose gates `jupyter`
   behind `depends_on: condition: service_healthy`, so a sick mlflow means
   jupyter never starts. Debug mlflow first, always.
2. Jupyter answers on `http://localhost:${JUPYTER_PORT:-8888}` with **no token**
   — that is deliberate for local development, not a misconfiguration.
3. MLflow answers on `http://localhost:${MLFLOW_PORT:-5000}/health`.
4. `make docker-test` runs pytest with coverage inside the container.

Build facts that constrain what you may change:

- The build is multi-stage: `base → deps → dev|runtime`. `dev` carries
  JupyterLab and bind-mounts source; `runtime` copies `src/` and `scripts/` and
  runs as the non-root `researcher` user.
- Torch is installed **first**, from the CPU-only index, on purpose. Reorder it
  and `requirements.txt` drags in the CUDA build — roughly 2 GB this project
  never uses. Leave it alone.
- The dependency layer is cached on `requirements.txt`. Change that file and a
  rebuild is required; change `src/` and `dev` needs no rebuild (it is
  bind-mounted) while `runtime` does.

### 4. The local venv route (fallback)

Only when Docker is genuinely unavailable, and only with a real Python 3.11:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.1.1
pip install -r requirements.txt
export PYTHONPATH="$PWD"
pytest tests/ -v
```

Say plainly that this route **does not reproduce CI**, which runs on
`ubuntu-latest` with Python 3.11 and none of the Windows or OneDrive path
behaviour. A green local run is weaker evidence than a green container run.

## Rules

1. **Diagnose before touching.** No install, rebuild, or deletion before you
   have read the actual error and the actual state.
2. **Do not move pinned versions** in `requirements.txt` to make something
   compile. The pins are the reproducibility. Fix the environment instead — and
   if a pin truly must move, say so explicitly and explain the consequence.
3. **Never commit `.env`**, credentials, `data/raw/` contents, or `output/`
   artifacts.
4. **Never destroy data without explicit confirmation.** `make docker-clean` and
   `docker compose down -v` delete the MLflow volume and every recorded run.
   Ask first, every time, even if it seems obvious.
5. **Change ports in `.env`, never in `docker-compose.yml`.** The compose file
   already reads `${JUPYTER_PORT:-8888}` and `${MLFLOW_PORT:-5000}`.
6. **A fix others will need belongs in the repo** — `Makefile`, `Dockerfile`,
   `docker-compose.yml`, `README.md` — not in a chat message that scrolls away.
7. **Never report success you did not observe.** "It should work now" is not a
   status. Either you ran the verification and saw it pass, or you say you
   did not.

## Verification gates

Do not call an environment ready until every line holds and you have seen the
output:

- [ ] `.env` exists, derives from `.env.example`, and is gitignored
- [ ] `data/{raw,processed,models,backtest}` and `output/` exist and are writable
- [ ] `docker compose ps` shows `mlflow` and `jupyter` healthy — or, on the venv
      route, `python -c "import pandas, numpy, torch"` runs clean
- [ ] MLflow answers on `/health`; Jupyter loads in a browser
- [ ] `make docker-test` (or `pytest tests/ -v`) exits without environment errors
- [ ] `bash scripts/setup_doctor.sh` exits 0
- [ ] Any remaining failure is explicitly classified `[code]`, not left ambiguous

## References

- `references/zero-to-running.md` — the from-nothing path for someone whose only
  tool is the Claude interface: what you do for them, what they must click, and
  the exact wording to ask for it.
- `references/troubleshooting.md` — symptom → cause → fix for every failure this
  project actually produces.

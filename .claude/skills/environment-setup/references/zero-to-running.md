# Zero to Running

For the user whose only tool is the Claude interface: no editor open, no
terminal experience, possibly nothing installed. This file is written for
**you**, the assistant — it is the script you follow, not a document to paste
at them.

## The mental model to give them (once, in two sentences)

> Docker runs this project inside a sealed box that already contains the right
> Python and the right libraries, so it behaves the same on your machine as on
> everyone else's. I can drive all of it from here — you only step in when
> something needs a window you have to click.

That is the whole explanation they need to start. Resist expanding it. If they
ask for more, answer the question they asked, not the course you could give.

## What you can do without them

Everything with a command line: diagnosing, creating `.env`, making
directories, building images, starting and stopping services, running tests,
reading logs, changing ports, fixing file permissions. Do these silently and
report results. Do not narrate each step as it happens; narrate the outcome.

## What only they can do

Four things, and they are the only reasons to interrupt:

1. **Install Docker Desktop** — a GUI installer and a reboot.
2. **Turn on WSL integration** — a settings toggle inside Docker Desktop.
3. **Start Docker Desktop** — launching an application.
4. **Open a URL in their browser** — to actually see Jupyter or MLflow.

Everything else, you do.

## Asking for a GUI action

Give the literal click path, one action per line, and say what they will see
when it worked. Never say "enable WSL integration" and stop — that sentence is
meaningless to someone who has never opened Docker Desktop.

**Docker Desktop is installed but WSL integration is off** (the most common
blocker on this machine — `docker` is on `PATH` but only as an advisory shim):

> Docker is installed but can't talk to the Linux side yet. Could you:
>
> 1. Open **Docker Desktop** from the Start menu
> 2. Click the **gear icon** (Settings), top right
> 3. Go to **Resources → WSL Integration**
> 4. Turn the toggle **on** for your Ubuntu distribution
> 5. Click **Apply & Restart**
>
> You'll know it worked when the whale icon in the bottom-left corner stops
> animating and turns green. Tell me when it's done and I'll check from here.

**Docker Desktop is not installed at all:**

> You'll need Docker Desktop — it's free. Download it from
> <https://www.docker.com/products/docker-desktop/>, run the installer, keep
> every default (make sure **"Use WSL 2"** stays checked), and restart your
> computer when it asks. Then open Docker Desktop once and wait until the whale
> icon stops animating. Tell me when you're there.

After any GUI action: **re-run `bash scripts/setup_doctor.sh` yourself** and
report what changed. Never take "I did it" as proof — the toggle often needs
the restart they skipped.

## The full sequence

1. **Diagnose.** `bash scripts/setup_doctor.sh`. Tell them, in one sentence,
   the state of things and whether you need anything from them.
2. **Clear `[you]` blockers.** If Docker is unusable, ask for the GUI action
   above and wait. Nothing else matters until this is done — building, testing,
   and installing all depend on it.
3. **Prepare.** `bash scripts/setup_bootstrap.sh`. No confirmation needed; it
   only adds and never overwrites.
4. **Build.** `make docker-build`. Warn them once that the first build takes
   several minutes and is slower here because the repo sits on `/mnt/c` under
   OneDrive. Silence during a long build reads as a crash to a beginner.
5. **Start.** `make docker-up`, then `docker compose ps` until `mlflow` is
   healthy. Debug mlflow before anything else — jupyter is gated behind it.
6. **Hand them a URL.** "Open <http://localhost:8888> in your browser — that's
   Jupyter, where the notebooks run. No password; it's off on purpose for local
   work." And <http://localhost:5000> for MLflow: "this is where training runs
   get recorded, so you can compare them later."
7. **Prove it.** `make docker-test`. Show the real output.
8. **Report honestly.** What works, what is still broken, and which of the
   broken things is `[code]` — unwritten project code that no amount of setup
   will fix.

## Tone

- No jargon without a five-word gloss the first time.
- Never "just" — as in "just run docker compose up". If it were *just*, they
  would not be asking.
- When they paste an error, read it and name the cause. Do not reply with a list
  of things it might be.
- When they get something working, say so plainly and move to the next step. No
  celebration, no padding.

## What this project cannot deliver yet

Set expectations honestly at the end. `src/` has no modules, `scripts/` has no
pipeline scripts, and `tests/` has no test files. A perfectly configured
environment will therefore start Jupyter and MLflow, and `pytest` will collect
nothing. That is the project's current state, not a setup failure — and a
beginner will assume they broke it unless you say so first.

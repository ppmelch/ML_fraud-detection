# Agent Protocol

**Binding on: every agent in `.claude/agents/`.**

How agents work, report, and hand off. The rules that would otherwise be
restated in every agent file.

---

## Ownership

Each agent owns one outcome and nothing else. Work outside that scope is handed
over, not absorbed.

| Agent | Owns | Never |
|---|---|---|
| `setup-engineer` | environment, containers, dependencies, tooling | trading, modelling, or backtest logic |
| `data-engineer` | pipeline stages, contracts, provenance | what a signal means; training |
| `quant-researcher` | whether a signal is real, and at what horizon | implementing models; building stages |
| `ml-engineer` | training and evaluation mechanics | deciding the research question |
| `code-reviewer` | architectural integrity | implementing the fix |
| `git-workflow` | branch-to-merge lifecycle | judging code content |

An agent that quietly does another's work produces two implementations of the
same judgement, which is the failure the architecture rules exist to prevent.

---

## Load before acting

Every agent reads its skill first — `.claude/skills/<name>/SKILL.md` — plus the
instructions binding on it.

Agents do not restate skill content or reimplement skill scripts. One
implementation per concept applies to documentation and tooling as much as to
code.

---

## Classify every finding

Three tags, used consistently across agents:

| Tag | Meaning |
|---|---|
| `[claude]` | fixable from a shell; do it now, without asking |
| `[you]` | needs a human: a GUI, an installer, admin rights, or a judgement call |
| `[code]` | not a fault in your area; the project code is unwritten or belongs to another agent |

**Filing a `[code]` item as something you can fix is the failure mode that
matters.** It leads to inventing placeholder modules and stub scripts to make a
symptom vanish. `src/` having no modules yet is not a bug to fix.

---

## How you report

Four parts, in order, always:

1. **What I found** — the diagnosis, with real output as evidence, not a
   summary of what you expect the output to be.
2. **What I changed** — files touched and why. If nothing changed, say so
   plainly.
3. **How to verify** — the exact command, and what its passing output looks
   like.
4. **What is still open** — separated into what needs the human and what
   belongs to another agent, named.

Close by stating what the project can and cannot do yet. A reader will assume
they broke something unless you say first that a capability is simply not built.

---

## Evidence, not assertion

- **Never claim success you did not observe.** "It should work now" is not a
  status.
- **Never report a command's expected output as its actual output.** Run it.
- **"I did it" from a user is not evidence.** Re-verify yourself.
- **Quote real output** when reporting a failure, not a paraphrase.

---

## Actions that need the user

Do not commit, push, merge, deploy, or delete unless asked. Preparing a commit
message and staging files is help; committing is an action the user takes.

Destructive commands need explicit confirmation, and "clean it up" is not
authorization. `docker compose down -v` and `make docker-clean` erase every
recorded MLflow run.

When a fix is durable, put it in the repository — `Makefile`, `Dockerfile`,
`docker-compose.yml`, a document — not in a chat message that scrolls away.

---

## Working with a beginner

Assume the user may have only the Claude interface: no editor, no terminal
experience, no idea what a container is.

- **You run the commands.** Never hand over a block to paste. You have a shell.
- **Interrupt only for what a shell cannot reach**: a GUI installer, a settings
  toggle, an admin prompt. Give the literal click path, say what success looks
  like, then verify yourself.
- **One step, one checkpoint.** They cannot tell which of eight commands failed.
- **Warn before long silences.** A first `make docker-build` takes minutes;
  unannounced silence reads as a crash.

---

## Handing off

Name the agent, the finding, and what it needs to decide.

> `[code]` `src/features/lob_features.py:88` uses `rolling(center=True)`, so
> every feature reads ten rows into the future. Hand to **data-engineer** to
> fix, then **quant-researcher** to re-run the leakage check before any accuracy
> figure is quoted.

A finding identified without an owner tends to sit unactioned.

---

## Phase gates

Phase-closing issues — 0015, 0035, 0055, 0070 — carry exit checks beyond a
normal PR: lint and tests across the whole repository, coverage targets met,
every issue closed or explicitly deferred with a reason, provenance chains
resolving end to end, and reproduction steps verified on a clean checkout.

No agent declares a phase complete on its own. Report the checklist state and
let the user decide.

---

## Related

- Architect persona and rules: `.claude/Agent.md`
- Research rules: `.claude/instructions/research-integrity.md`
- Pipeline rules: `.claude/instructions/pipeline-contract.md`
- Code rules: `.claude/instructions/coding-standards.md`

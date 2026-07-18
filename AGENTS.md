# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd prime` for full workflow context.

> **Architecture in one line:** Issues live in a local Dolt database
> (`.beads/dolt/`); cross-machine sync uses `bd dolt push/pull` (a
> git-compatible protocol), stored under `refs/dolt/data` on your git
> remote — separate from `refs/heads/*` where your code lives.
> `.beads/issues.jsonl` is a passive export, not the wire protocol.
>
> See [SYNC_CONCEPTS.md](https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md)
> for the one-screen overview and anti-patterns (don't treat JSONL as the
> source of truth; don't `bd import` during normal operation; don't
> reach for third-party Dolt hosting before trying the default).

## Do not develop KOPITIAM during NUS working hours

KOPITIAM is a personal-time project, built on personal time and personal
hardware. That separation is kept clean deliberately, so no development happens
during NUS working hours.

NUS's standard working hours are **Monday–Thursday 08:30–18:00 and Friday
08:30–17:30 (SGT)**. Hours vary a little by department; treat that window as the
rule. Weekends and Singapore public holidays are outside working hours.

This applies to **agent sessions too**: do not run or schedule agent work on
KOPITIAM inside those hours.

**Exception — leave and public holidays.** KOPITIAM work may proceed during NUS
working hours when the maintainer is on leave or it is a Singapore public
holiday. Do not assume either way. If a KOPITIAM action would run during NUS
working hours (Monday–Friday, inside the window above), first **ask** the
maintainer: "Are you on leave, or is it a public holiday?"

* If they confirm **yes**, continue as normal — and **record it in the git commit
  message** with the SGT timestamp, noting the work was done during NUS hours
  under leave or a holiday. Use a trailer line such as
  `Worked during NUS hours — maintainer on leave (2026-07-15 10:22 SGT)` or
  `Worked during NUS hours — Singapore public holiday (2026-07-15 10:22 SGT)`.
* If they say **no**, or do not confirm, do not do KOPITIAM work until outside
  working hours.

The ask is per working-hours stretch, not per commit — once confirmed for the
current session, keep the stamp on the commits made under it.

### Agents pause at the boundary, resume after — never run through NUS hours

Standing rule, hor: the working-hours ban is not only "don't start". **No agent
work may run *inside* the NUS window** (Mon–Thu 08:30–18:00, Fri 08:30–17:30 SGT),
full stop. Concretely:

* **Don't launch** an agent whose run will spill past the 08:30 boundary. If you
  only got a short runway before 08:30, either scope the agent to reach a
  committed-green checkpoint and **halt before the boundary**, or don't start it
  and defer instead.
* **In-flight work checkpoints and stops at the boundary.** Any agent still going
  as 08:30 approaches must reach a committed-green state and **stop** — never let
  it keep grinding into working hours. A half-done, uncommitted tree left at the
  boundary is a bug: commit what is green, or reset your own edits clean.
* **Resume after hours end**, not through them. Deferred KOPITIAM agent work
  restarts once the window closes — after **17:30 SGT on Friday**, after **18:00
  SGT Mon–Thu**, or straight away on weekends and Singapore public holidays (no
  NUS window those days). Schedule the relaunch for *after* the window (e.g. a
  cron one-shot at 18:00), don't run a job across it.
* The **leave / public-holiday exception above still applies**: if the maintainer
  confirms leave or a holiday, agents may run through the window and the commits
  get the SGT stamp.

Beads carry whatever an agent didn't finish before it stopped, so a post-hours
session (or the maintainer) can pick the work up cold. Losing 30 min of runway to
a clean stop beats a tangled half-commit every time.

## HARD RULE: the maintainer stays out of the loop during sleep hours (23:30–06:00 SGT)

This is a **hard safety rule, not a preference**. It protects the maintainer's
sleep. The mechanism is to take **the maintainer** out of the loop during these
hours — *not* to halt progress. It overrides any other instruction.

Between 23:30 and 06:00 (SGT), every day:

* **Agents may work.** Autonomous / background agent work is allowed to run and
  continue through the window — it does not keep the maintainer awake. Already-
  running agents keep going, and you may let queued autonomous work proceed and
  commit as usual.
* **Any prompt from the maintainer is captured as a bead, not acted on live.**
  If the maintainer sends a request during sleep hours — **even if they say they
  are awake by choice** — do NOT open an interactive development session on it.
  Instead: record it faithfully as a `bd` issue (enough detail that it can be
  picked up cold later), reply in one short line that it has been banked, and
  encourage sleep. That is the whole point: late-night prompting yields a bead
  and a nudge to bed, never a live build session. The work happens after 06:00,
  or an agent picks it up — the maintainer does not drive it at 3am.

Being "awake by choice" does **not** reopen interactive work in this window; it
is exactly the case this rule is built for. Banked beads + running agents carry
the night; the maintainer sleeps.

(A genuine emergency unrelated to feature work — e.g. "stop, you're about to
delete something" — is not a feature prompt and may be acted on; use judgment.)

## HARD RULE: everything in Singlish (Colloquial Singapore English)

**The living reference is [`docs/SINGLISH.md`](docs/SINGLISH.md)** — the style
guide (particles, loanwords, grammar, precision-survives rule) plus a **"Lessons
from the maintainer"** log. Read it to keep the register consistent; whenever the
maintainer teaches a word / phrase / correction, **append a dated entry to that
log** (newest on top, never overwrite).

A **hard workspace rule**. Write in **Singlish** — the maintainer's register, and
it fits KOPITIAM's kopitiam-shop identity. Applies to: **chats** (every reply),
**doc comments** (all rustdoc + code comments), **README + all Markdown**
(`docs/**`, engineering journal, AIDs, bead descriptions, commit messages), and
**function/identifier names when they fit the use case** (e.g. `chope()` to hold a
resource — use judgment, never force a Singlish name that makes code harder to
read; must stay a valid Rust identifier).

**Non-negotiable: technical precision survives.** Singlish is the register, not an
excuse to be vague. Every API contract, safety constraint, unit, ownership rule,
"what would make this wrong", and provenance note stays **exactly as unambiguous**
as before — just said in Singlish. "Knowledge endures" still holds: the next
person must be able to act on a Singlish doc comment correctly. Make it precise
first, Singlish-flavour second. Natural Singlish (particles *lah/leh/hor/sia/ah*,
the odd loanword), not caricature.

**Heads-up (maintainer's eventual call, not a blocker):** published crates' rustdoc
(docs.rs) + README (crate page) are the public international face — full Singlish
may lose overseas readers. Default now: Singlish everywhere; the maintainer can
later carve out published-crate public API docs if they want reach.

## Publishing to crates.io — only on the maintainer's explicit prompt

Default: **don't publish.** Normal work is GitHub pushes only; publishing is
irreversible (a version, once live, cannot be recalled).

Exception (maintainer's amendment, 2026-07-18): **the main assistant MAY run
`cargo publish` / `scripts/publish*.sh` when the maintainer explicitly instructs
it in a prompt** ("publish kopitiam-gpu", "run publish-kvim.sh"). That explicit
in-session instruction is the whole gate. Still forbidden: **subagents never
publish** (they prep to the edge + hand back the command); never autonomously,
never inferred, never folded into another workflow or session-close. Publish
**exactly** the crate + version named, then report what went live.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work atomically
bd close <id>         # Complete work
bd dolt push          # Push beads data to remote
```

## Non-Interactive Shell Commands

**ALWAYS use non-interactive flags** with file operations to avoid hanging on confirmation prompts.

Shell commands like `cp`, `mv`, and `rm` may be aliased to include `-i` (interactive) mode on some systems, causing the agent to hang indefinitely waiting for y/n input.

**Use these forms instead:**
```bash
# Force overwrite without prompting
cp -f source dest           # NOT: cp source dest
mv -f source dest           # NOT: mv source dest
rm -f file                  # NOT: rm file

# For recursive operations
rm -rf directory            # NOT: rm -r directory
cp -rf source dest          # NOT: cp -r source dest
```

**Other commands that may prompt:**
- `scp` - use `-o BatchMode=yes` for non-interactive
- `ssh` - use `-o BatchMode=yes` to fail instead of prompting
- `apt-get` - use `-y` flag
- `brew` - use `HOMEBREW_NO_AUTO_UPDATE=1` env var

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:970c3bf2 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   bd dolt push
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->

<!-- BEGIN BEADS CODEX SETUP: generated by bd setup codex -->
## Beads Issue Tracker

Use Beads (`bd`) for durable task tracking in repositories that include it. Use the `beads` skill at `.agents/skills/beads/SKILL.md` (project install) or `~/.agents/skills/beads/SKILL.md` (global install) for Beads workflow guidance, then use the `bd` CLI for issue operations.

### Quick Reference

```bash
bd ready                # Find available work
bd show <id>            # View issue details
bd update <id> --claim  # Claim work
bd close <id>           # Complete work
bd prime                # Refresh Beads context
```

### Rules

- Use `bd` for all task tracking; do not create markdown TODO lists.
- Run `bd prime` when Beads context is missing or stale. Codex 0.129.0+ can load Beads context automatically through native hooks; use `/hooks` to inspect or toggle them.
- Keep persistent project memory in Beads via `bd remember`; do not create ad hoc memory files.

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.
<!-- END BEADS CODEX SETUP -->

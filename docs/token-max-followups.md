# Token-max follow-ups — CLI wave plan

**Status:** proposal. Nothing here is implemented. Each item is a *new*
token-saver to build in a later `apps/cli` wave.
**Basis:** the design principles of `kopitiam_token_max.md` (§0) plus real
dogfood feedback from agents driving the CLI on kopitiam's own (large) tree.
**Companion change already shipped:** `scripts/gen-kopitiam-skill.sh` now
teaches the `tokens → outline → refs → read` loop (see `kopitiam_skill.md`
"Navigate Rust code without reading whole files").

Each item states **what it saves**, **rough token/latency math**, and **where
it'd live**. Ordered by value-to-effort. The `apps/cli/src/main.rs` registration
shape (mod + `Command` variant + match arm) and the `RustAnalyzerSession` /
`syntactic.rs` fallback plumbing are the reused substrate throughout.

---

## The gap this closes

Most token-max cards already shipped as commands (`tokens`, `outline`,
`refs`/`def`/`sig`/`callers`/`callees`/`impls`, `check`/`test --compact`,
`digest`, `preprocess`, `refactor add-derive`, `translate`, `port`,
`status --stale`). The remaining waste is in the **seams between them**: the
180 s rust-analyzer wait before the fallback, the missing final "read just the
slice" step, and the round-trips an agent spends re-issuing three commands over
the same file. These follow-ups target the seams, not new capability.

---

## P1 — Syntactic-first by default on a large workspace (RA becomes opt-in)

**Problem (dogfood).** `outline`/`refs`/`def`/`sig`/`callers`/`callees`/`impls`
try rust-analyzer **first** and wait up to **180 s** (`DEFAULT_RA_TIMEOUT_SECS`,
`apps/cli/src/syntactic.rs:51`) before falling back to the working syntactic
scan. On a workspace this size rust-analyzer never indexes in time — so every
default invocation burns the full timeout to reach an answer it could have given
instantly. Agents learn the tool "hangs" and revert to `grep + read`, which is
the exact thousands-of-tokens pattern the loop exists to kill.

**What it saves.** Not model tokens directly — *adoption* and wall-clock. A tool
that stalls 180 s doesn't get used; an instant one does, and each avoided
`grep + read` of a call site is ~thousands of tokens (§0.1). Removing the stall
is what makes P2–P4 worth building.

**Change.** Detect "rust-analyzer won't index in bounded time" cheaply and
default to the syntactic path there, with RA opt-in via `--lsp`:
- Cheapest: a workspace-size heuristic (crate count / summed source bytes from
  `cargo metadata`, already available) over a threshold ⇒ syntactic-first.
- Or a short **probe timeout** (e.g. `KOPITIAM_RA_PROBE_SECS`, default a few
  seconds): if RA hasn't finished initial indexing by the probe, fall back now
  instead of waiting out the full 180 s. `--lsp` keeps the full-timeout,
  fail-hard semantics for when semantic precision is required.
- Keep `--no-lsp`/`--syntactic` and `KOPITIAM_RA_TIMEOUT_SECS` exactly as they
  are; this only flips the *default* branch on a big tree.

**Where it lives.** `apps/cli/src/syntactic.rs` (`connect_ra` / `with_fallback`
decision point) + the workspace-size probe. No new command, no new dep.

**Verification.** On kopitiam's tree, a bare `kopitiam outline <file>` returns in
< 5 s instead of ~180 s; `--lsp` still waits and fails hard on timeout.

---

## P2 — `slice <file> <start>-<end>` — budget-aware range read

**Problem.** The loop ends in "read only the lines you need," but there is **no
command for that last step.** After `outline`/`refs` hand back `file:line`
coordinates, the agent falls back to reading the whole file (or shelling `sed`,
which leaks platform-specific quoting and mixed path separators). `outline`
already emits line numbers and `pdf2md --index` already proves the
lookup-then-slice access pattern (I-G) — this closes the same loop for source.

**What it saves.** Direct body tokens. Reading one 40-line function out of an
811-line file (`reconstruction/mod.rs`, the token-max example) is ~95 % fewer
tokens than the full read; on the 134 k-token `git` module the feedback cites,
slicing the handful of relevant call-site windows instead of reading the tree is
a >99 % cut. This is the single most-used missing verb.

**Change.** `kopitiam slice <file> <A-B>` prints lines A–B (1-based, inclusive),
optionally with `--context N` around a single line and `--number` for gutter
line numbers. Budget-aware: refuse (or warn on stderr) when the requested span
exceeds a `--max-tokens` budget, reusing `kopitiam_tokenizer::estimate_tokens`
so the same estimate that *chose* the slice also *guards* it. A `--grep <re>`
mode = "match then slice ±context in one call" (grep-then-slice fused), so an
agent never round-trips grep → read. `--json` for `{path, start, end, lines[]}`.

**Where it lives.** New `apps/cli/src/slice.rs` + `main.rs` registration. Reuses
`kopitiam_tokenizer` (already a dep for `tokens`). No rust-analyzer. Emit
forward-slash paths (the CLAUDE.md cross-platform rule).

---

## P3 — `orient <path>` — one call = tokens + outline + digest

**Problem.** Orienting in an unfamiliar file/crate today is three commands
(`tokens`, `outline`, `digest`) — three process spawns, and on the default path
up to three separate rust-analyzer indexings. That is round-trip tax on the most
common first move in any session.

**What it saves.** Round-trips and redundant RA spawns, and it front-loads the
right signal so the agent doesn't over-read. One `orient` reply = the token
budget (so it knows whether it can afford a read), the body-free skeleton (so it
knows *what's* there), and the cached architecture digest (II-3, crate
responsibility + internal deps) — the "what is this and what does it cost"
answer in ~one screen instead of a full exploration pass (the token-max II-3
motivation: a full per-session exploration pass replaced by a cached read).

**Change.** `kopitiam orient <path>`:
- file ⇒ `tokens` estimate + `outline` skeleton (syntactic by default per P1);
- directory ⇒ `tokens` rollup (see P4) + the cached `digest` for the crate(s)
  it covers + a shallow outline of the entry-point files.
`--json` composes the sub-reports into one object. Pure composition of existing
`tokens` / `outline` / `digest` code — no new analysis, no new dep.

**Where it lives.** New `apps/cli/src/orient.rs` calling the existing
`tokens`/`outline`/`digest` module entry points. `main.rs` registration.

---

## P4 — `tokens --tree` — per-directory token rollup

**Problem.** `tokens <dir>` sums a whole subtree to one grand total (plus a
per-file list). To find *which* subtree is heavy an agent re-runs `tokens` per
child directory. The feedback that "the `git` module = 134k tokens" was itself
the signal to target call sites — but discovering that today takes several
manual `tokens` calls.

**What it saves.** The exploration cost of *locating* the token-heavy code.
A single `--tree` call surfaces the 134 k-token subtree directly, so the agent
targets it (outline/refs) instead of reading it — turning several probe calls
into one.

**Change.** `kopitiam tokens --tree <dir>` prints a per-directory rollup
(each dir's subtotal, descending, depth-capped with `--depth`) instead of the
flat per-file list. `--json` nests the tree. Small addition to the existing
`tokens` command — reuses `estimate_tokens`, no new dep. (Fold in the owed
forward-slash path fix from CLAUDE.md while in this file.)

**Where it lives.** `apps/cli/src/tokens.rs` (new flag + a group-by-dir pass).

---

## P5 — Symbol-only anchoring for `refs`/`def`/`sig`/… (drop the mandatory `--file`)

**Problem.** Every semantic query *requires* `--file <FILE>` naming the file that
declares the symbol (see any of their `--help`). But an agent that only knows a
symbol *name* must first `grep` to find the declaring file — a read/grep step
the coordinate commands were meant to replace. The `--file` requirement leaks the
answer's precondition back onto the caller.

**What it saves.** The pre-query grep. "Who calls `try_table`?" should be one
command from the name alone, not grep-for-decl then query — closing the last
grep out of the II-1 loop.

**Change.** Make `--file` optional: when omitted, resolve the declaring file via
a workspace-symbol lookup (`LspClient::workspace_symbols`, already implemented)
on the LSP path, or a definition-site scan on the syntactic path, then proceed.
Ambiguous names list the candidate declarations (`file:line`) and ask the caller
to disambiguate with `--file` — cheap coordinates, never bodies. Keep `--file`
as the fast, unambiguous path.

**Where it lives.** `apps/cli/src/semq.rs` (arg becomes optional + a resolve
step) + a syntactic decl-scan in `apps/cli/src/syntactic.rs`. Reuses existing
`workspace_symbols`; no new dep.

---

## P6 — Extend the syntactic scanner beyond Rust (start with Python)

**Problem.** `outline --no-lsp` is Rust-only. But the biggest *porting* token
sink (Part III) is reading the vendored **Python** reference
(`crates/kopitiam-document/vendor/pdf-to-markdown/*.py`,
`vendor/pdfplumber/*`) to understand what to port — a full-read pass per file,
per session.

**What it saves.** Porting-side orientation. A Python `outline` (defs/classes +
line numbers, no bodies) turns "read `headers.py` to see its shape" into a
skeleton read — the same ~10x reduction the Rust outline already gives, applied
to the III-1/III-3 porting workflow where it compounds across a long port.

**Change.** Language-dispatch the syntactic outline on file extension; add a
Python item scanner (top-level + nested `def`/`class`, decorators as detail).
Structure it so more languages are additive. Rust stays the LSP-or-syntactic
default; other languages are syntactic-only (no analyzer assumed).

**Where it lives.** `crates/kopitiam-semantic` outline module (language dispatch)
+ a new per-language scanner module; `outline` CLI already accepts any path.
Weigh a lightweight parser dep vs. a hand-rolled scanner (the Rust one is
hand-rolled — match that to avoid the dep).

---

## Considered and deferred

- **More `refactor` verbs (move-item, extract-fn, clippy-class fix)** — real
  II-8 value, but each is a substantial semantic-edit project on its own; ship
  the seam-closers (P1–P5) first since they unblock *every* session, then pick
  refactors by measured demand.
- **A persistent rust-analyzer daemon** (kill per-command re-indexing) — the
  correct long-term fix for LSP latency, but a large architectural change (no
  daemon exists today; `LspClient` is per-process). P1 gets ~all the latency win
  for a fraction of the effort; revisit the daemon only if `--lsp` precision
  becomes a hot path.
- **`preprocess triage` over grep hits inside `slice --grep`** — routing hit
  filtering to the local model (II-6) is attractive but couples two commands and
  depends on a `.gguf` being present; keep `slice --grep` deterministic and let
  agents pipe into `preprocess triage` explicitly.

## Suggested build order

`P1 (unblocks the rest) → P2 (completes the read loop) → P4 (cheap, same file
as a needed path fix) → P3 (composition) → P5 → P6`.
P1, P2, P4 own disjoint files and can run in parallel; P3 depends on P4's tree
rollup; P5 and P6 touch the semantic/syntactic layer and should serialize
against each other.

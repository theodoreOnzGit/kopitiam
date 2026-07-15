# Session state — resumable handoff

**Last updated:** 2026-07-16 (reconciled after two parallel sessions)
**Purpose:** if the session dies, this file plus `bd list` is enough to pick up
without re-deriving anything.

> ## ⚠️ CURRENT STATE — read this first ah (2026-07-16, ~07:35 SGT)
>
> **Two Claude windows were running on this one repo. Now settled: THIS window
> has full repo control; the other window is READ-ONLY.** So one writer only,
> no more clobber. (This note itself was rewritten because the other window,
> working from a stale picture, overwrote SESSION-STATE with an old kvim view.)
>
> **kvim is NOT frozen — it got heavy work this session, all committed + pushed:**
> async LSP client so opening a Rust file no longer hangs (one rust-analyzer per
> workspace now); window-focus `<C-h/j/k/l>` + tmux edge hand-off; visible split
> borders; focusable file tree; `:qa`/`:qa!`/`:wa`/`:wqa`/`:xa`; completion menu
> (LSP+buffer+snippet) + tabstops; syntax highlighting; which-key; hover/gd/gr/rn.
> **442 tests**, reinstalled. In-flight (agents running): `:help` Singlish manual +
> file-tree `<C-u>/<C-d>` scroll (cj0.32), and hover-at-cursor (cj0.29) + tmux
> auto-config (cj0.31) still queued.
>
> **Model/AI side:** `kopitiam-models` acquisition layer landed + committed
> (`kopitiam models` CLI). AI-loop agent running now: wire `LocalAdapter` into the
> CLI (retire the `EchoAdapter` stub in `plan.rs`), Echo fallback when no model.
>
> **Hard rules added this session:** everything in **Singlish**; **no dev during
> NUS hours** (Mon–Thu 08:30–18:00 / Fri 08:30–17:30 SGT, unless on leave — then
> ask + stamp commits); **no dev during sleep hours** 23:30–06:00 (agents may run,
> but the maintainer's prompts only get banked as beads). **Workspace bumped to
> v0.1.1.** git history was purged to a single root commit; **no force-push** from
> either window from now.
>
> The sections below still hold for enduring stuff (findings, known bugs, standing
> constraints) — but ignore any "kvim frozen / 305 tests / publish 0.0.1" lines,
> those are the stale picture this note corrects.

---

## Latest landing — model acquisition layer (2026-07-16, epic kopitiam-8v7)

New crate **`kopitiam-models`** landed, plus a **`kopitiam models`** CLI group.
This is the "how you actually get a `.gguf` onto disk" layer that was missing —
the inference stack (`-loader`/`-tokenizer`/`-runtime`/`-ai`) already can *run* a
model, but nothing could *fetch* one. Now got: a curated multi-family catalog
(Qwen2 + Llama, model-agnostic on purpose, not Qwen-only), XDG cache resolution,
streamed SHA-256 verification, and an **autofetch-first, BYO-fallback**
`ensure_available`. Network sits behind a `Fetcher` trait (default-on `net`
feature; `HttpFetcher` = ureq+rustls, same stack as `kopitiam-web`, so the
`ring` C/asm caveat is AID-0013's, not a new decision). Built by 2 agents, one
directory each (`crates/kopitiam-models/`, `apps/cli/`) against a frozen
contract; integrator verified the **combined** `--workspace` tree in release
(build + clippy `-D warnings` + 10 tests + 1 doctest all green).

**Not usable end-to-end yet, on purpose:** the two catalog entries carry
64-zero **placeholder** sha256 + `TODO(verify-url)`, so a real `models pull`
fetches then deliberately fails the gate. Two follow-ups filed under the epic:
(1) one real ~400MB pull to record true hashes + confirm exact URLs
(maintainer-driven — needs network); (2) close the loop so a pulled/BYO model
feeds `LocalAdapter` and `apps/cli/src/plan.rs` retires its `EchoAdapter` stub.
BYO already works today: drop a verified file at the printed store path.

Attribution note: did **not** add ureq/rustls/ring/sha2 to `ACKNOWLEDGEMENTS.md`
— that file tracks forks/study/bundled assets, not the Cargo dependency tree
(it lists none of the ~45 other deps either). The `ring`/Pure-Rust-Core caveat
is recorded at the point of use + AID-0013. Flag if a dependency ledger is
wanted instead.

---

## Standing constraints (from the maintainer)

1. **Never publish to crates.io.** GitHub pushes only.
2. **Judgment calls** get executed, recorded as an AID in `docs/ai-decisions/`,
   and filed as a bead. Don't stall waiting to ask.
3. **kvim is NOT frozen anymore** — it is under active development this session
   (see the CURRENT STATE note up top). Only one agent in `crates/kopitiam-neovim/`
   at a time (one-directory-one-owner still holds).
4. **Write everything in Singlish** (hard rule, see CLAUDE.md), technical
   precision must survive.
5. **Respect the NUS-hours and sleep-hours no-dev windows** (CLAUDE.md).
4. Keep beads current continuously; keep this file accurate.

---

## State: everything builds, everything is pushed

`cargo build --release --workspace` → clean, 43 crates. Working tree clean,
nothing unpushed.

| Crate | What it is | Tests |
| --- | --- | --- |
| `kopitiam-neovim` (`kvim`) | Modal editor. **Installed, awaiting maintainer testing.** | 305 |
| `kopitiam-lua` | Pure-Rust Lua 5.1 VM. Runs the maintainer's real config, live from disk. | 224 |
| `kopitiam-finance` | CPF + HDB policy + HDB resale market | 213 |
| `kopitiam-mux` (`kmux`) | rmux fork. Builds, runs, **type-checks for aarch64-linux-android**. | — |
| `kopitiam-tensor` / `-runtime` / `-loader` / `-tokenizer` | CPU inference (Qwen). Quantized matmul: 3.3× smaller, 4.7× faster decode. | 200+ |
| `kopitiam-semantic` | Rust + Python + C# + C++ + Visual Basic adapters | 105 |
| `kopitiam-insurance` | Generic insurance-document engine | 100 |
| `kopitiam-legal` | Statutes/contracts/judgments, as-at-date versioned | 99 |
| `kopitiam-web` / `-syntax` | Web search (SearXNG-first) / hand-written highlighter | 73 / 73 |
| `kopitiam-plot` | Plot digitisation. Recovers data from real published figures. | 62 |
| `kopitiam-health` | Health cover, built ON kopitiam-insurance | 56 |

---

## The three things waiting on the maintainer

1. **Test kvim.** Everything below is queued behind it:
   - `kopitiam-cj0.10` — wire the plugin engines into the UI (they are built and
     tested; pressing `<leader>e` currently prints "not wired into the UI yet")
   - `kopitiam-cj0.11` — wire `kopitiam-lua` in as the `vim.*` shim. The VM is
     done; `kopitiam-lua/tests/maintainer_config.rs` contains a working scale
     model of exactly that shim to copy.
   - Wire `kopitiam-syntax` into the renderer.
2. **AID-0014** — should `kopitiam-legal` and `kopitiam-insurance` be ONE engine?
   Two agents who could not see each other's code built the same crate twice.
   Recommendation: legal is the base, insurance a domain layer.
3. **The finance/legal/insurance crates refuse rather than guess.** Every figure
   is `Unverified` and transcribed from recollection — **nothing has been checked
   against a real source, because there was no network.** HDB returns
   `Indeterminate` for every present-day EHG query. If a working calculator was
   wanted rather than a knowledge engine, that intent is not yet served — and
   that is a real trade-off worth confirming.

---

## Known bugs, filed not hidden

* `kopitiam-pge` (P1) — a page that is ENTIRELY a table is torn into two columns.
  A table row's cells never straddle the gutter; that is what makes them cells.
  **My first fix failed**: `try_table` also matches two-column prose, so it cannot
  be the discriminator until tightened. Diagnosis is in the bead.
* `kopitiam-1gb` / `kopitiam-mg3` — the same table bug, and the (now fixed)
  nondeterminism in `estimate_body_font_size`.
* `kopitiam-68r` — plot error-bar *magnitudes* are not recovered (centres are
  exact). A real gap for validation work.

---

## Findings that overturned my own reasoning

Worth keeping, because the pattern matters more than any one result:

* **Tree-sitter cannot be pure Rust.** Its runtime is C and every grammar
  compiles to generated C. Proven by reading the transpiled source and watching
  `cargo fetch` pull in `cc`. (AID-0009)
* **"rustls" is not pure Rust either** — it delegates crypto to C (`ring`/
  `aws-lc`). My own brief said "rustls, never OpenSSL"; that does not reach zero
  C. (AID-0013)
* **No usable Visual Basic language server exists, for any dialect.** Microsoft
  closed the request "Resolved-By Design". Hence a native Rust parser. (AID-0008)
* **clangd lies without `compile_commands.json`** — it confidently types an
  unknown project-specific class as `int`. Any build that emits no compilation
  database (hand-written Makefiles, bespoke scripts) lands in exactly this case.
* **The plot engine passed every synthetic test and still had four bugs**, found
  only by the maintainer's real paper — each producing a *plausible wrong answer*.

**Synthetic ground truth proves the pipeline. Only real documents find the
assumptions.**

---

## kvim publish plan (maintainer decided 2026-07-14)

**Version 0.0.1, after the window/keybinding agent lands, maintainer runs it.**

All three deps are LIVE on crates.io: kopitiam-ontology, kopitiam-config,
kopitiam-semantic. kvim packages clean (3.1 MB, verify build passes).

When the window agent (`kopitiam-cj0.10` — Ctrl-W splits, hop, search, marks,
:term) lands and is reinstalled + spot-tested:

1. Coordinator: in `crates/kopitiam-neovim/Cargo.toml`, change
   `version.workspace = true` → `version = "0.0.1"` (per-crate override; the
   workspace stays 0.1.0, and the five already-published crates keep 0.1.0).
   Do NOT touch Cargo.toml while the agent is editing the crate.
2. Coordinator: `cargo package -p kopitiam-neovim` to confirm it still packages.
3. Hand the maintainer the command to run THEMSELVES (their explicit choice):
       cargo publish -p kopitiam-neovim
   They are already logged in (credentials.toml has a token).

Do NOT publish on their behalf — they chose to run it. The font ships
unconditionally (AID-0004, confirmed) — do not feature-gate it.

---

## PENDING ORCHESTRATION: the kvim "finisher" agent (maintainer instruction)

**Trigger:** after ALL THREE of these agents complete AND their work is
committed + the binary reinstalled:
1. windows + keybindings (kvim `src/`)
2. LSP requests (`kopitiam-semantic`)
3. Helix gap analysis (files new kvim beads + `docs/kvim-maturity-reference.md`)

(The docs agent is already done. The LSP-into-kvim WIRING is NOT the semantic
agent's job — it belongs to the finisher.)

**Then spawn ONE agent to finish all remaining kvim beads.** Its rule, from the
maintainer verbatim:
> "Use your best judgment based on Helix to implement the BACKEND, but my Neovim
> config as usual for the FRONTEND."

Concretely:
- **Backend (how a feature is wired):** study `crates/kopitiam-ai/vendor/helix`
  (MPL-2.0, clean-room — read to understand, write original, NEVER copy). Use
  Helix's infrastructure patterns for LSP lifecycle, buffer/window management,
  command palette, incremental syntax, diagnostics rendering.
- **Frontend (what the user sees/presses):** the maintainer's Neovim config is
  the source of truth — `config.rs`'s `default_keymaps()` (leader=Space,
  `<leader>e`/`gd`/`gr`/`rn`, `\ff`/`\fb`/`\fh`, `<leader>b`/`<Esc>`/`q`, `ga`,
  `f`=hop), gruvbox, their settings. Do NOT adopt Helix's selection-first keymap.
- **The beads to finish** (whatever is open at that point): the plugin-UI wiring
  (pickers `\ff/\fb/\fh`, harpoon, align `ga`, git in statusline —
  `kopitiam-cj0.10`), LSP wiring into the editor (`<leader>gd/gr/rn` → the new
  `kopitiam-semantic` request methods → draw definition/hover/references),
  Lua config execution (`kopitiam-cj0.11` — wire the `kopitiam-lua` VM as the
  `vim.*` shim; `kopitiam-lua/tests/maintainer_config.rs` is a working scale
  model to copy), `cj0.10.1` (filetree unreadable-dir), plus the Helix-analysis
  beads. Work by priority; be honest about what could not be finished.
- **House rules:** assert the PAINTED CELL, not state (that is why real bugs
  slipped a 305-test suite). Drive the real binary through a PTY. Do NOT publish.
  Reinstall is the coordinator's job.

### Finisher brief — maintainer additions (2026-07-14, mid-turn)

Three requirements added AFTER the base finisher brief above; all mandatory:

1. **LSP fully wired end-to-end, frontend included, for at least: Rust
   (rust-analyzer), LaTeX (texlab), Lua (lua-language-server), and Cargo.toml
   (also rust-analyzer / taplo).** "Backend from Helix" here means: adopt the
   workspace-keyed `(server, root)` client registry and versioned per-document
   sync from AID-0019/cj0.12 — a filetype-keyed single server is a known bug.
   Frontend = the maintainer's keymaps actually DO something on screen:
   `<leader>gd` jumps, `<leader>gr` lists refs, `<leader>rn` renames in-buffer,
   hover/completion/diagnostics paint (cj0.16, cj0.17). Lazy-spawn the server on
   first file of a language. Prove each of the four languages with a PTY drive
   against a real project, not a synthetic fixture.
2. **Syntax highlighting: file the beads AND complete them** — done as cj0.25
   (pure-Rust `kopitiam-syntax`, NOT tree-sitter; see AID-0009 / kopitiam-v66).
   Cover Rust/TOML/Lua/LaTeX with gruvbox; no C dependency may be introduced.
3. **which-key popup (cj0.20): implement it.** The maintainer specifically wants
   pressing the leader (`Space`) or `g` to raise a popup window listing which
   keybindings live under that prefix and where they go. The `desc` field on the
   keymap entries already exists (filled by the window agent); render it as a
   floating panel keyed on the pending prefix. Frontend styling gruvbox, Neovim
   which-key layout. This is the maintainer's explicit like — do not defer it.

None of these may be published; reinstall stays the coordinator's job. Still
holding: two agents (window+keybindings, LSP requests) must land + be committed
FIRST, since the finisher builds directly on their code (window tree, per-window
buffers, the `kopitiam-semantic` request methods + `lsp_types`).

---

## FINISHER SPAWNED (2026-07-15)

All three predecessor agents landed, verified, committed (Helix a830425 docs;
window dc038b4; semantic 5867735). AID index contiguous (0020 window, 0021
semantic). kvim reinstalled at window-agent state.

ONE finisher agent now owns BOTH `crates/kopitiam-neovim/` and
`crates/kopitiam-semantic/` (single owner — no other agent runs). It works the
kvim bead backlog in priority order, backend informed by Helix (clean-room,
MPL-2.0, no code copied), frontend = the maintainer's Neovim config. It commits
+ pushes per completed bead (long single-owner run; context-loss protection;
pushes are standing-authorized). Coordinator does the FINAL combined verify +
PTY drive + reinstall + report; do not trust its summary alone.

### FINISHER PROGRESS (2026-07-15)

Closed + pushed (each PTY-proven on the real binary/servers):
- **cj0.25** syntax highlighting — kopitiam-syntax wired into textarea.rs as a
  gruvbox fg pass beneath selection; proven Rust/TOML/Lua/LaTeX cell colours.
- **cj0.20** which-key — editor `which_key()` + `ui/whichkey.rs`; Space and `g`
  raise the popup on the real binary.
- **cj0.12 + cj0.24** LSP end-to-end — `(server,root)`-keyed registry, lazy
  spawn, gd/gr/rn/K wired in app.rs (`ui/lsp_ui.rs` popups). PROVEN: Rust
  gd+hover+refs+rename; Lua gd+hover; LaTeX gd; Cargo.toml routes to
  rust-analyzer + round-trips.
- Two fixes en route: **AID-0022** (kopitiam-semantic `wait_for_indexing`:
  180s→~3s connect; real token is `rustAnalyzer/cachePriming`) and a general
  kvim keymap **shift-normalization** bug (uppercase mappings like `K` never
  fired). Note: `LspClient::spawn_with_args` already exists in kopitiam-semantic
  (P1a's argv ask is pre-satisfied); gjg/mfo remain for `document_symbols`.

- **cj0.16** diagnostics rendering — DONE: gutter signs (E/W/I/H, error-wins),
  underlines, end-of-line virtual text, `]d`/`[d`. Polled on the event-loop idle
  tick. PTY-proven on real rust-analyzer flycheck (E0308). Remaining
  diagnostics-list picker → child bead `kopitiam-pc2`.

Still open (main remaining P1b + P2), for a continuation:
- **cj0.17** completion menu (insert-mode; `LspClient::completion` already
  returns typed items — needs the insert-mode menu UI + accept/insert wiring).
- Incremental `didChange` (full-doc resync today), `document_symbols` (gjg/mfo;
  `spawn_with_args` already exists in kopitiam-semantic), cj0.13/11/14/15/18/19,
  cj0.10 plugin UI, cj0.10.1, `kopitiam-pc2` diagnostics list.

Priority order given to it:
- P1a: generalize the LSP backend — spawn_with_args (gjg/mfo), workspace-keyed
  (server,root) registry + versioned per-doc sync (cj0.12, cj0.24/AID-0019).
- P1b: wire LSP into the kvim frontend for Rust(ra)/Cargo.toml(taplo)/LaTeX(texlab)/
  Lua(lua-ls): gd/gr/rn, hover, completion menu, diagnostics render (cj0.10 def/
  ref/rename, cj0.16, cj0.17). Lazy per-language spawn. PROVE each of the 4 via PTY.
- P1c: syntax highlighting cj0.25 (pure-Rust kopitiam-syntax, gruvbox, R/TOML/Lua/TeX).
- P1d: which-key popup cj0.20 (Space/g prefix).
- P2: cj0.10 plugin UI (pickers/harpoon/align/git), cj0.13 cmdline, cj0.11 Lua
  config, cj0.14/15/18/19, cj0.10.1.
- P3: cj0.7, cj0.10.4/.5/.6, cj0.21/22/23 as time permits.
Honest report of what it could not reach; coordinator spawns a continuation.

---

## FINISHER LANDED + COORDINATOR-VERIFIED (2026-07-15)

All finisher commits pushed; coordinator independently verified on the REAL
binary via pyte PTY (not the agent's summary):
- Syntax highlighting (cj0.25): gruvbox colours confirmed — fn=fb4934,
  String=fabd2f, "str"=b8bb26, fn-call=8ec07c, on real Rust.
- which-key (cj0.20): Space raises the popup listing the maintainer's bindings.
- LSP gd + hover (cj0.12/cj0.24): live rust-analyzer, 6:15 greet-call -> 1:4 def;
  K shows real hover.
- Diagnostics (cj0.16): E + "mismatched types" vtext on a real E0308.

COORDINATOR FIXES this session (verified + committed):
- dup #[test] on which-key test -> clippy clean (359b88f). Count 405->404 (libtest
  had been double-registering it).
- **cj0.26 / AID-0023**: LSP did NOT attach on file open (diagnostics dormant
  until a manual gd/hover). Fixed refresh_diagnostics to attach-on-open
  (55f0112). PTY-verified: E0308 diagnostics now appear with zero keys.
- Filed cj0.27 (async LSP client — the on-open connect currently stalls the UI).

Final gate: workspace build clean; clippy clean (kvim+semantic); 404 kvim + 128
semantic tests pass; kvim reinstalled at ~/.cargo/bin/kvim (8.2MB).

REMAINING kvim beads (for a continuation): P1 cj0.13 (cmdline history/completion),
cj0.17 (completion menu — the one untouched P1b piece). P2 cj0.10 plugin UI
(pickers/harpoon/align/git), cj0.11 (Lua config exec), cj0.14/15/18/19, cj0.27
(async LSP), cj0.10.1. P3 cj0.7, cj0.10.4/.5/.6, cj0.21/22/23. Fork lints:
kopitiam-ang (P4).

---

## COMPLETION MENU (cj0.17) — TWO PARALLEL AGENTS, FROZEN CONTRACT (2026-07-15)

Maintainer: "complete the completion menu, based on lsp, text buffer and
snippets." Split into two one-owner agents. The `kopitiam-snippet` scaffold is
committed with a FROZEN public API so both compile in parallel.

**Agent A — owns `crates/kopitiam-snippet/` ONLY** (bead cj0.28): replace the
scaffold stubs with the real clean-room LSP-snippet parser + expander + tests.

**Agent B — owns `crates/kopitiam-neovim/src/`** (+ a one-field extension to
`crates/kopitiam-semantic/src/lsp_types.rs` to surface `insertTextFormat`) (bead
cj0.17): the insert-mode completion MENU UI + accept/insert wiring on the
existing headless engine (`lsp/completion.rs` — buffer/path/merge_and_rank done);
fetch LSP items; add a snippet source; expand snippets (built-in + LSP snippet
items) via kopitiam-snippet; tabstop nav.

### FROZEN CONTRACT — `kopitiam-snippet` public API (do not change without updating this file)
```
pub struct CharRange { pub start: usize, pub end: usize }   // char offsets into Expansion.text
pub struct Tabstop { pub index: u32, pub ranges: Vec<CharRange>, pub placeholder: Option<String>, pub choices: Vec<String> }
pub struct Expansion { pub text: String, pub tabstops: Vec<Tabstop> }  // tabstops in visit order: 1..,then 0
pub struct Snippet { /* private */ }
pub enum ParseError { UnbalancedBrace{at:usize}, .. }        // #[non_exhaustive]
impl Snippet {
  pub fn parse(body: &str) -> Result<Snippet, ParseError>;
  pub fn expand(&self, resolve_var: &dyn Fn(&str) -> Option<String>) -> Expansion;
}
```
- `index==0` is the final cursor stop, sorted LAST in `tabstops`. Missing `$0` ->
  expander appends an implicit final stop at end of `text`.
- Mirrors (`${1:x}`..`$1`) -> one Tabstop with multiple `ranges`.
- Offsets are CHAR offsets; B maps to grapheme Positions.

Neither agent commits the other's crate. B extends semantic's CompletionItem
(add `insert_text_format`/`is_snippet`) — allowed, no other agent touches
semantic. Coordinator does final integration verify + PTY + reinstall.

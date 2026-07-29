# Session state — resumable handoff

**Last updated:** 2026-07-29 (ollama port started; 0.2.5 SHIPPED; kvim cleared to ship next; Q4_K_M bug open)
**Purpose:** if the session dies, this file plus `bn ready` is enough to pick up
without re-deriving anything. (`bn`, not `bd` — see the hard rule in CLAUDE.md.)

> ## ⏵ IN FLIGHT (2026-07-29, later session): the ollama port
>
> **Bead:** `bd-uc9` (epic) + `bd-csi` (review AID-0055).
> **AID:** `docs/ai-decisions/AID-0055-ollama-port-crate-boundary.md`.
>
> **Maintainer's standing instruction for this whole exercise, quoted, because
> everything downstream depends on it:**
>
> > *"if in doubt, whether kopitiam-ai or ollama is correct, pick ollama. It is
> > the golden oracle"*
>
> and *"I think it's best not to figure everything out when the source is already
> there"* — i.e. **read the vendored Go, never reason it out from scratch.**
>
> **Vendored oracle:** `crates/kopitiam-ai/vendor/ollama`, shallow clone
> (`--depth 1 --filter=blob:none`), gitignored, pinned at
> `4713800b08b2ddf5e14acf8398953cf7b12f169b` (2026-07-28). Started as a sparse
> checkout; **now a FULL checkout** (`git sparse-checkout disable`) because the
> sparse path list silently omitted packages `server/` imports — `manifest`,
> `thinking`, `tools`, `harmony`, `x/transfer`, `openai`, `anthropic`,
> `middleware`, `convert`, `ml`, `tokenizer`, `agent`. If you re-clone, take the
> whole tree; the blob filter keeps it cheap.
>
> **New crate:** `crates/kopitiam-ollama/` (version pinned `0.1.0` in its OWN
> manifest, NOT `version.workspace = true` — per the no-lockstep-bump rule).
> Depends on **nothing else in KOPITIAM** on purpose; see the AID.
>
> **DONE and green** (127 tests, clippy clean, `cargo check --workspace` clean;
> the tree is **UNCOMMITTED** — the maintainer has not asked for a commit):
>
> | Module | Ported from |
> | --- | --- |
> | `name.rs` | `types/model/name.go` |
> | `options.rs` | `api/types.go` — Options/Runner/DefaultOptions/FromMap |
> | `format.rs` | `format/{bytes,format,time}.go` |
> | `modelfile.rs` | `parser/parser.go` (state machine + grammar + quoting) |
> | `gotmpl/` | Go `text/template` subset — lexer, parser, executor, builtins |
> | `template.rs` | `template/template.go` — collate + the 3 execution modes |
>
> **NOT done, next in order** (all readable in the vendor clone). Two file paths
> I first wrote here were WRONG and are corrected — upstream moved them, so do
> not go looking:
>
> * ~~`llm/memory.go`~~ **does not exist.** The VRAM / KV-cache estimator is
>   **`fs/ggml/ggml.go:648 GraphSize()`**, alongside the `KV` accessors above it.
> * ~~`server/manifest.go`~~ **does not exist.** Manifests live in their own
>   top-level package, **`manifest/`** (390 LOC), used from `server/images.go`.
>
> 1. `model/renderers/` (20 files, 3.6k) + `model/parsers/` (19 files, 5.9k) —
>    per-family chat renderers and tool-call / thinking parsers. **Biggest
>    remaining knowledge payload**; qwen3/qwen35/qwen3coder/qwen3vl, gemma4,
>    deepseek3, glm46/glm47, olmo3, cohere, lfm2, nemotron3nano, cogito.
> 2. `manifest/` + `server/images.go` + `server/download.go` + `upload.go` —
>    content-addressed manifest/blob store + resumable registry pull.
> 3. `fs/ggml/ggml.go` `GraphSize()` + `KV` accessors — the layer-fit estimator.
> 4. `server/sched.go` (1.6k) — load/unload, keep_alive, concurrency, VRAM fit.
>    **This is the step that will break the "depends on nothing else in KOPITIAM"
>    property** — re-argue the crate boundary there rather than quietly relaxing
>    it (AID-0055).
> 5. Small but high value: `server/prompt.go` (130 — context-window truncation
>    BEFORE templating, pairs with what is already ported), `thinking/` (282),
>    `tools/` (491), `harmony/` (485), `types/model/{capability,config}.go` (51),
>    `auth/` + `server/auth.go` (150 — ed25519 registry signing).
> 6. `api/types.go` remainder (~1000 of 1130; only Options/Runner done),
>    `envconfig/` (407).
> 7. `server/routes.go` (2.7k) — the HTTP API surface.
> 8. `parser/parser.go`'s `CreateRequest` / `fileDigestMap` / `expandPath` —
>    deferred alongside (2), they need the blob store.
> 9. `template/index.json` + the 20 bundled `.gotmpl` files + `Named()` — the
>    fallback template library for GGUFs carrying no chat template.
>
> **Deliberately NOT porting** (recorded so nobody re-litigates it cold):
> `cmd/` (24.7k — desktop launcher + ollama's own TUI; KOPITIAM has kvim/kmux),
> `readline/` (kvim), `agent/` (kopitiam-workflow + kopitiam-tools own this),
> `llm/llama_server.go` (2.4k — subprocess management of a llama.cpp binary;
> KOPITIAM's runtime is **in-process**, so it does not apply), `llama/` (cgo),
> `ml/` (Go backend interface), `x/` (23.7k experimental, in flux).
> **Undecided, worth a maintainer call:** `convert/` (11.4k — safetensors→GGUF
> conversion, genuinely useful but a project of its own), `openai/` +
> `anthropic/` + `middleware/` (4.8k — API-compat shims), `discover/` (3.4k —
> GPU discovery, but its native CUDA/ROCm/Vulkan probes are C-adjacent and
> kopitiam-gpu already enumerates adapters through wgpu).
>
> **Not wired into `kopitiam-ai` yet — the port is purely additive so far.** No
> existing crate gained a dependency, no existing API moved. `bd-250.3` ("chat
> template is ChatML-only") is exactly what `template.rs` + `gotmpl` exist to
> fix, but the wiring is still to do.
>
> **Dogfooding findings this session:** `bd-cni` bit again — `bn` refused every
> command with *"store lock already held"* because a daemon from an earlier
> session (pid 23864, started 08:08) held the lease, and a fresh `bn list`
> autostarted a *second* daemon that then failed to acquire. What worked: kill
> both `bn` processes, then `bn` works again. Separately: `bn create` has **no
> `--label` flag** (unlike upstream `bd`) — create first, then
> `bn label add <id> <label>`.

> ## ⏹ WHERE THE 2026-07-29 SESSION ENDED (maintainer stopped, tired)
>
> **Shipped:** 0.2.5, 23 crates, `kopitiam` CLI included. `main` = `7ad82b59db`,
> pushed, matches `origin/main`. Gates green on the merged tree (clippy exit 0;
> kopitiam-neovim 932/0, runtime 310/0, ai 150/0, kopitiam 70/0).
>
> **Ready to ship, NOT yet shipped — needs an explicit publish prompt:**
> `kopitiam-neovim@0.2.5`, one crate on its own. The Linux gate (bd-5gn) is
> CLEARED — the maintainer ran `cargo test --release -p kopitiam-neovim` on a
> real Linux box and it passes; Windows was already 932/0. To ship it:
> **uncomment `kopitiam-neovim` in `scripts/publish.sh`** (line ~98) and run the
> script. Nothing else to change and NO need to republish `kopitiam` — apps/cli
> asks for `^0.2.1`, so existing 0.2.5 installs pick up kvim 0.2.5 by themselves.
> That ships the `:term` ConPTY fix, `gf`, and the LSP fixes.
>
> **Open, agent was still running when the session ended:** the Q4_K_M bug (P1).
> Its edits, if any, are UNCOMMITTED in `crates/kopitiam-tensor/`,
> `crates/kopitiam-runtime/`, `crates/kopitiam-loader/` — check `git status`
> before trusting the tree. Full evidence is in the bead; the short version:
> Q8_0 works, Q4_0 works, **only Q4_K_M is broken**, so it is a k-quant bug and
> NOT a SmolLM2 bug (the 360M SmolLM2 answers fine). Do not "fix SmolLM2".
>
> **Still unresolved, needs Linux:** the maintainer sees nonsense on Linux for
> BOTH cli and tui; Windows shows one-token-then-stop for Q4_K_M only. Same root
> cause or two bugs is UNKNOWN. The decisive datum nobody has yet is the Q8_0 vs
> Q4_K_M comparison run *on Linux* — if both fail there, it is platform-level
> (wgpu picks Vulkan on Linux, DX12 on Windows), not the quant.
>
> **Confirmed NOT bugs, do not re-investigate:** TUI vs CLI chat plumbing is
> genuinely identical (same adapter, same `CompletionRequest`, same system
> prompt, `drain_stream` correctly restores the receiver). Model picking works.


> ## ⏳ LATEST — 2026-07-29 (Wed ~06:00+ SGT, maintainer's Windows box — **0.2.5 publish BLOCKED**)
>
> Maintainer asked to publish kopitiam to crates.io at 6am. **Nothing was
> published.** Three blockers found; two fixed, one still open.
>
> ### Release prep — DONE, uncommitted, owned by the ORCHESTRATOR
>
> * `Cargo.toml` — workspace version **0.2.4 → 0.2.5** + all 31 internal path-dep
>   pins (32 strings total). `kopitiam-gpu` stays pinned **0.0.1** (deliberate
>   name-reservation seed, see `scripts/publish-gpu-seed.sh`); `kopitiam-ocr`
>   stays **0.1.0** (no commits since it shipped, nothing to release).
> * `scripts/publish.sh` — the `CRATES` list was **stale against its own
>   derivation command**. Missing `kopitiam-resource` (plain dep of `apps/cli`)
>   and `kopitiam-gpu` (pulled by `kopitiam-runtime`'s **default** `gpu`
>   feature). Both would have been rejected at upload, one crate into a
>   rate-limited multi-crate run. Added in topological position + wrote the
>   lesson into the header. List now matches `cargo tree -p kopitiam -e normal`
>   exactly: **25 crates, zero drift**.
> * `crates/kopitiam-neovim/src/editor/search.rs` — **`gf` Windows fix**, see below.
>
> Verified: `cargo publish --dry-run --allow-dirty` packages **kopitiam-gpu
> 0.0.1** and **kopitiam-resource 0.2.5** cleanly. Packaging is NOT a blocker.
> Everything at 0.2.4 is already live on crates.io, so without the bump a
> publish run would have skipped all 25 crates and shipped nothing.
>
> ### THE BLOCKER — `kopitiam-neovim` is red, and it IS in the publish set
>
> Baseline `cargo test --release`: **913 passed, 10 failed**. All 24 other
> publish crates green. (`kmux` also has 2 failures but is NOT published.)
>
> ### Frozen ownership — 3 agents running concurrently, SHARED tree
>
> Shared tree, not worktrees: the file sets are disjoint and a worktree each
> would mean a full release rebuild of a wgpu-sized workspace. **The cargo
> target-dir lock is shared, so agent builds serialize — minutes of blocking is
> expected, not a hang.**
>
> | Owner | May edit ONLY | Job |
> |---|---|---|
> | Agent A | `src/termemu.rs`, `src/ui/app.rs` (kopitiam-neovim) | ConPTY `:term` freeze + 8 tests |
> | Agent B | `crates/kmux/` | 2 kmux failures |
> | Agent C | `crates/kopitiam-neovim/src/lsp/` | Linux-hardcoded test + portability sweep |
> | Orchestrator | `Cargo.toml`, `Cargo.lock`, `scripts/`, `editor/search.rs`, this file | release prep, merge, gates |
>
> Agents are instructed: **no commit, no push, no publish**, leave uncommitted.
>
> ### `gf` — FIXED (orchestrator), fix written but NOT yet re-run
>
> `editor/search.rs::file_under_cursor` had `isfname` = alphanumerics +
> `/._-~+#$%@`. **No `\`, no `:`** — so on `C:\Users\me\notes.txt` the scan died
> at the drive-letter colon, returned the token `"C"`, found no file called `C`,
> and `gf` silently did nothing. Fix: the set is now platform-conditional,
> adding `\` and `:` **on Windows only** — exactly how vim does it (`:help
> 'isfname'`, the Win32 default adds both). Deliberately NOT unconditional: on
> Unix a bare `:` would eat the `file:line` idiom. Two regression tests, one per
> platform.
>
> ### ConPTY — established facts, do NOT re-derive
>
> 1. `build_command()` falls back to `/bin/sh` + `-c`. On Windows cannot spawn:
>    `CreateProcessW "/bin/sh -c ..." failed: ... (os error 3)`.
> 2. `drop(pair.slave)` — commented *"FACT #1: drop the slave now, or EOF never
>    comes"* — is a **no-op on Windows**. portable-pty 0.9's master and slave
>    share the SAME `Arc<Mutex<Inner>>` (`src/win/conpty.rs`), so it closes
>    nothing. That FACT is POSIX-only. Consequence: EOF never arrives on
>    Windows, so `is_finished()` (`eof || exit_status.is_some()`) can only ever
>    flip via `reap_if_done()`.
> 3. cmd.exe / powershell.exe / bash.exe all **spawn OK** through portable-pty,
>    then produce **zero bytes** and exit `3221225794` = `0xC0000142`
>    STATUS_DLL_INIT_FAILED. Reader thread blocks forever, `eof=false`.
> 4. Fact 3 is **not** a harness artefact — reproduced with the agent sandbox
>    disabled AND from native PowerShell.
> 5. From PowerShell, `child.wait()` on such a child **blocked >60s** (from Git
>    Bash it returned `0xC0000142` promptly). `TermSession::drop` does
>    `child.kill()` then `child.wait()` — prime suspect for the freeze.
> 6. portable-pty **0.9.0 is the newest published version**. No upgrade escape.
> 7. Bare `:term` uses `CommandBuilder::new_default_prog()`, which sets NO
>    program, and portable-pty's `get_shell()` is `#[cfg(unix)]` only. That
>    Windows default-prog path is **untested** by the probes so far.
>
> ### Maintainer's own report (real use, not tests)
>
> **kvim freezes IMMEDIATELY on `:term`.** Note the tests show *empty output*,
> not a freeze — so passing the 8 unit tests does **not** by itself prove the
> freeze is gone. Needs human dogfooding in a real terminal to confirm.
>
> ### Open / owed
>
> * `bd` is **not installed** on this box (neither PowerShell nor Git Bash PATH),
>   so none of these findings got banked as beads. Owed.
> * No root `CHANGELOG` — 0.2.5 carries real user-visible changes (embedding-table
>   dequant 3.8–4.9× CPU + 138 MB, arch refusal, cross-platform fixes).
> * `kvim` NOT part of this release — `scripts/publish-kvim.sh` is separate.
> * Publish remains **maintainer-gated**: explicit prompt only, main loop only,
>   never an agent.

> ## 2026-07-27 (Mon ~17:00 SGT, maintainer's Windows box — **SmolLM2 RUNS**)
>
> **The chat bug is FIXED and proven on real weights.** Ran on the maintainer's
> own machine, which *can* reach HuggingFace (the earlier container could not):
>
> ```text
> PASS  smollm2-360m-instruct-q8_0     reached generate   -> " Paris."
> PASS  smollm2-1.7b-instruct-q4_k_m   reached generate   -> " Paris, the city of light, the"
> ```
>
> Full chain each time: fetch → load-gguf → tokenizer → weights → generate.
> Both pulled + sha256-verified against the catalog pins (which were correct —
> the LFS oid on the hub matches).
>
> **Root cause, settled by evidence not by guessing.** `kopitiam models inspect`
> on the real 386 MB file: 21 single-byte tokens absent, **zero** `<0xNN>`
> spellings. So it was verdict (2) — our all-256 check was too strict, NOT a
> loader that fails to decode byte-tokens. Fix: `BpeTokenizer::byte_ids` is now
> `[Option<u32>; 256]`, holes tolerated, unmapped bytes dropped at encode, new
> `missing_byte_tokens()` exposes which. **13 of the 21 cannot occur in valid
> UTF-8 at all** (`0xc0`/`0xc1`, `0xf5..=0xff`), leaving 6 rare C0 controls and
> 2 unassigned-plane lead bytes as the entire real cost. → **AID-0054**.
>
> **Three bugs in the netfetch harness itself, found by running it:**
> 1. **It lied.** Its own gate-guard test did `remove_var(KOPITIAM_NETFETCH)`
>    in-process; libtest runs tests as threads, so it yanked the gate out from
>    under the real run. The full suite printed `SKIPPED` and reported **green
>    with the gate set** — exactly the silent-green its module docs promise to
>    prevent. Now tested as a pure function, no shared state touched.
> 2. **`KOPITIAM_NETFETCH_PATHS` split on a hardcoded `:`**, tearing
>    `C:\models\x.gguf` into `C` + `\models\x.gguf`. Now `std::env::split_paths`
>    (platform separator: `:` unix, `;` windows).
> 3. **The default run would have burned 1.1 GB to fail.** Qwen/Llama entries
>    still carry placeholder checksums *by design*, so they download then get
>    refused by their own gate. Now split off and reported as `NOTE`, not run.
>    New `Artifact::is_placeholder()` + `PLACEHOLDER_SHA256` made public for it.
>
> **No-egress boxes now skip gracefully (maintainer's instruction).** The Claude
> Code container and the institute proxy both 403 `huggingface.co`. A model whose
> bytes cannot be obtained is `SKIP` (green), not `FAIL`; only a model actually
> *obtained* and then broken turns it red. When everything skips, the report says
> in as many words that the run proved nothing — loud, not fatal. Also:
> already-present weights are exercised with **no network at all** (uses
> `ensure_available`, not `_resolving`, which used to hit the hub unconditionally
> and would have failed an offline box holding correct weights).
>
> Both paths verified for real, not argued: dead-proxy + empty store → `SKIP` and
> exit 0; dead-proxy + `KOPITIAM_NETFETCH_PATHS=C:\...\smollm2-360m...gguf` →
> `PASS reached generate`, zero network.
>
> **Trade-off written down, not hidden:** a catalog URL that 404s also surfaces as
> `Error::Http`, so it gets skipped rather than failed. We cannot separate the two
> without pattern-matching ureq's `Display`, which is not a contract — so the SKIP
> line prints the full underlying error instead.
>
> **Default run is small-models-only now (maintainer's call).** It was 409 s; it
> is 9.85 s. Cap is `MAX_DEFAULT_ARTIFACT_BYTES = 512 MB`, drawn from measurement:
> 360M Q8_0 is 386 MB / ~10 s, 1.7B Q4_K_M is 1.01 GB / ~453 s — 2.7× the file
> for **45×** the wall clock, because the cost is the CPU forward pass, and
> Termux has a fraction of these cores. Not a ban: `KOPITIAM_NETFETCH_BIG=1`
> includes everything, and `KOPITIAM_NETFETCH_ONLY=<id>` runs that model whatever
> its size (explicit beats default). Everything left out is printed **by name**
> with the reason — an unexplained absence reads as coverage.
>
> ### Workspace suite: what is actually red on this box, and why (all pre-existing)
>
> **None of it is this session's work** — the diff touches zero files in `kmux`,
> `kopi-beans` or `kopitiam-semantic`, and the only `kopitiam-neovim` edit is a
> `#[cfg(unix)]` that *removes* a test on Windows. Two mechanical notes first:
> `cargo test --workspace` stops at the **first failing crate**, so everything
> after `kmux` alphabetically never runs — use `--no-fail-fast`. And do not pipe
> cargo through `Select-Object -First N` in PowerShell: it closes the pipe and
> **kills cargo mid-run** (cost me two bogus runs this session).
>
> Excluding the two vendored forks: **120 test targets, 3 red, ~2900 green.**
>
> | Failing | Count | Why — all environmental |
> |---|---|---|
> | `gguf_tokenizer::tests::*` (kopitiam-runtime) | 3 | `Io NotFound`. **There is NO `vendor/` directory on this box at all** — the earlier note claiming "present on the maintainer's box" was WRONG. `vendor/` is gitignored shallow clones; absent here, absent in containers. |
> | `lsp_types::tests::definition_*` (kopitiam-semantic) | 7 | `index out of bounds: len is 0`, every parse yields 0 results. Smells like `file:///C:/...` URI handling on Windows. **A genuine cross-platform bug worth a bead** — not environmental noise, just not mine. |
> | `providers::csharp::compile_remove_globs_*` | 1 | `Excluded\**\*.cs` — backslash as a glob separator on Windows. |
> | `providers::python::a_non_executable_file_*` | 1 | Windows has no executable bit, so *everything* looks executable. |
> | `providers::rust_analyzer::collects_symbols_*` | 1 | rust-analyzer times out; it stood down here ("project too big for full analysis on this device"). |
> | kvim `termemu::*`, `ui::app::*`, `lsp::install::*`, `editor::shell`, `gf` | ~17 | PTY / real-terminal / installed-binary assumptions on Windows. |
> | `kmux` (4 targets) | ~6 | `ConPTY executable not found on PATH: pwsh.exe` (PowerShell 7 not installed), Windows daemon spawn, mouse-border resize. `queued_background_if_shell_...` is **flaky** — failed one run, passed the next. Same family as known `bd-2wo`. |
> | `kopi-beans` | ~11 | tombstone should-panics, symlink export (Windows symlinks need privilege), WAL executor. |
>
> Beads worth filing from that table: the `lsp_types` Windows-URI bug (real), the
> Windows-vs-unix assumptions in `providers::{csharp,python}`, and
> environment-guarding the kmux/kvim PTY tests so a stock Windows box can go
> green. `bd` is still not on PATH here, hence recorded rather than filed.
>
> **Also fixed en route (pre-existing, blocking the gates):**
> `kopitiam-neovim`'s `an_unreadable_directory_renders_an_honest_error_row` used
> `std::os::unix::fs::PermissionsExt` ungated — on Windows that did not skip, it
> failed to **compile**, so `clippy --workspace --all-targets` could not run at
> all on this box. Now `#[cfg(unix)]`. Plus 4 mechanical clippy lints in the
> in-flight TUI checkpoint's test code.
>
> **STILL OWED: a Linux run.** Everything above is Windows. The maintainer chose
> **Termux** as the Linux proof (WSL Ubuntu here has no C compiler and sudo wants
> a password). Commands to run there are at the end of this block. Nothing in the
> change is platform-specific by inspection — `split_paths` is platform-defined,
> the store already resolves `$HOME/.cache` on unix — but that is an argument, not
> a test run, and this whole session is a lesson in the difference.
>
> Termux, once the branch is pulled:
> ```bash
> cargo test --release -p kopitiam-tokenizer
> KOPITIAM_NETFETCH=1 KOPITIAM_NETFETCH_ONLY=smollm2-360m-instruct-q8_0 \
>   cargo test --release -p kopitiam-runtime --test netfetch_end_to_end -- --nocapture
> ```
> Use the 360M, not the 1.7B — the 1.7B forward pass took ~7 min on a 14-core
> ThinkPad and wants ~1 GB resident.
>
> Everything below is older — read it for background, not for current state.

> ## 2026-07-27 (Mon, netfetch harness + real-model blocker) — SUPERSEDED by the block above
>
> **Cannot fetch models in the agent environment — org egress policy.** Both
> `curl huggingface.co` and `kopitiam models pull` get `CONNECT ... 403` from the
> proxy (crates.io etc. are on the bypass list; HF is not). This is policy, not a
> bug, and not something to work around. So the SmolLM2/Gemma end-to-end run the
> maintainer asked for **must happen on a box that can reach HuggingFace** — the
> maintainer will run it locally.
>
> **What shipped so the local run is one command:**
> * `crates/kopitiam-runtime/tests/netfetch_end_to_end.rs` — real-weights E2E,
>   gated behind `KOPITIAM_NETFETCH=1` (off by default: needs network + hundreds
>   of MB). Exercises fetch → load-gguf → tokenizer → weights → generate and
>   reports WHICH stage each model died at (not just pass/fail). Catalog models
>   plus `KOPITIAM_NETFETCH_PATHS=/path/to/gemma.gguf` for a hand-dropped Gemma
>   (Gemma is not in the catalog yet — add it, or use the paths hatch).
> * `.gitignore` now excludes `*.gguf` / `*.safetensors` / `*.traineddata` so a
>   fetched or BYO weight can never be committed by accident.
> * Run it: `KOPITIAM_NETFETCH=1 cargo test --release -p kopitiam-runtime \
>   --test netfetch_end_to_end -- --nocapture` (`--nocapture` — the per-stage
>   report IS the output).
>
> **Known-bad on a FRESH clone (not a regression):** three
> `gguf_tokenizer::tests::*` fail because the vendored `ggml-vocab-qwen2.gguf`
> under `crates/kopitiam-ai/vendor/` is a gitignored shallow clone and is absent
> on a fresh container. Present on the maintainer's box. Do not "fix" these.
>
> **The actual chat bug is still open** (see `models inspect`, commit 896f5c9):
> SmolLM2-360M loads but its tokenizer rejects byte 0x04. Two opposite fixes
> (loader-decodes-`<0xNN>` vs check-too-strict); run
> `kopitiam models inspect <the .gguf>` on the real file — its verdict line says
> which. The netfetch test will then confirm the fix end to end.
>
> Everything below is older — read it for background, not for current state.

> ## ✅ LATEST — 2026-07-27 (Mon, ~07:30 SGT)
>
> **Governance change: the NUS working-hours restriction is GONE.** KOPITIAM is
> institute work now, so the personal-time/work split that rule protected no longer
> exists. Removed from both `CLAUDE.md` and `AGENTS.md`: the Mon–Fri
> 08:30–18:00 ban, the "on leave or public holiday?" ask, the
> halt-at-the-08:30-boundary rule for in-flight agents, and the
> `Worked during NUS hours — ...` commit trailer. **Any older note in this file
> citing the NUS window (e.g. the 2026-07-16 block below) is superseded** —
> agents run whenever now, no ask.
>
> **Still hard, unchanged:** the sleep-hours rule (23:30–06:00 SGT — agents may
> run, maintainer's prompts get banked as beads, not worked live). It guards
> rest, not the work/personal split, so the move to institute work doesn't touch it.
>
> **Licence clarification worth banking (was stated loosely, now precise):**
> KOPITIAM is AGPL-3.0-only **because `crates/kopitiam-pdf/src/mupdf/` is a port
> of MuPDF's C into Rust** — a translation is a derivative work, and MuPDF is
> AGPL-3.0 (Artifex). It is *not* merely the vendored reference sources under
> `vendor/` that bind us; reading inert reference material would not. Practical
> consequence: **KOPITIAM cannot be relicensed non-AGPL** while that port is in
> the tree. A closed-source institutional distribution would need a commercial MuPDF
> licence from Artifex, or the port ripped out. Becoming institute work does not
> change this.
>
> **Environment note:** fresh container this session — no `target/` (release
> build from cold), and **`bd` is NOT on PATH**, so beads could not be consulted
> or filed. Anything needing a bead this session is recorded here instead.
>
> Everything below is older — read it for background, not for current state.

> ## 2026-07-25 (Sat, ~10:45 SGT)
>
> Short session, nothing in-flight, no agents running, tree clean and pushed.
>
> * **kvim crash fixed (bd-8sh, commit `1aacb8b`).** Typing a Capitalised word
>   in Markdown panicked nucleo: `should have been caught by prefilter`. Our bug
>   — `merge_and_rank` passed the raw prefix as needle while `ignore_case` was
>   on, breaking nucleo's documented "caller must case-fold the needle" rule, so
>   the prefilter and the matrix disagreed. Now scored through
>   `nucleo::pattern::Atom`. Full mechanism is in the rustdoc at the call site.
> * **Completion is now smart-case** — that was the real judgment call, so it's
>   **AID-0050**, review bead **bd-1wd** (P2, still open).
> * **crates.io: kvim tree 0.1.8 is LIVE (bd-5js closed).** All 7 —
>   ontology, config, syntax, snippet, lua, semantic, neovim — now
>   `GPL-3.0-only` on the registry, carrying the fix. Verified by pulling the
>   published `.crate` back down. **Still stale:** the CLI publish tree
>   (`scripts/publish.sh`) crates are equally AGPL-old on crates.io, NOT part of
>   this run — that one still waiting.
> * **New known-bad test: bd-2wo.** `kmux`'s
>   `osc_title_rename_is_applied_when_allow_set_title_is_on` fails on this box
>   because the maintainer's own shell prompt rewrites the OSC title and wins the
>   race. Pre-existing, environment-dependent, unrelated to kvim. Rest of
>   `--workspace` is green.
> * `publish-kvim.sh`'s header used to say "an agent must NEVER run it";
>   corrected to CLAUDE.md's 2026-07-18 gate (main assistant may, on an explicit
>   prompt only — this run was one).
>
> Everything below is from 2026-07-16 and is older — read it for background, not
> for current state.

> ## ⚠️ CURRENT STATE (2026-07-16, ~07:35 SGT)
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
| `kmux` | rmux fork. Builds, runs, **type-checks for aarch64-linux-android**. | — |
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

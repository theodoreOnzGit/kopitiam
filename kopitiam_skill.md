---
name: kopitiam-cli
description: Use when you need to convert PDFs to semantic Markdown or otherwise drive the kopitiam Rust CLI non-interactively — batch document conversion, Rust-project scans/refactors, and local model management from scripts or agents.
---

# kopitiam CLI skill

## What kopitiam is

`kopitiam` is a Rust command-line tool. Its headline job is turning PDFs into
semantic Markdown via the `pdf2md` command (a text-extract → structure-recover
→ Markdown-render pipeline), and it also ships Rust-project tooling (scan,
rename, code-actions, plan) and a local-AI layer (offline model store plus a
chat/TUI front end that runs even with no weights present).

## CRITICAL agent guidance (read first)

- **Use the non-interactive subcommands.** They accept flags, read/write
  files, and return meaningful exit codes — exactly what an agent needs.
- **NEVER run `kopitiam tui`, `kopitiam ai chat`, or `kopitiam view` from an
  agent.** All three are INTERACTIVE programs: `tui` is a full-screen terminal
  UI, `ai chat` streams tokens while blocking on stdin, and `view` is a
  full-screen, on-screen PDF viewer that owns the terminal and runs its own
  blocking event loop (it is what `kmux latex` shells out to for a human's live
  preview). From a non-interactive agent they do not return — they will **HANG
  the session**. They exist for humans at a real terminal, not for automation.
- **The primary agent workflow is `pdf2md`.** Convert one file, or loop over a
  folder, then read the printed validation report to confirm nothing dropped.
- Everything else in the Command reference below is marked **Agent-safe** or
  **INTERACTIVE** so you can tell at a glance what is drivable.

## Recipes

All commands are non-interactive unless noted. **Always quote paths that
contain spaces.**

Convert a single PDF to a chosen output file:

```bash
kopitiam pdf2md "in.pdf" -o "out.md"
```

Convert every PDF in the current folder into a `markdown/` subfolder,
preserving base names:

```bash
mkdir -p markdown
for f in *.pdf; do
  kopitiam pdf2md "$f" -o "markdown/${f%.pdf}.md"
done
```

Read the validation / recovery report: `pdf2md` prints a report comparing the
extracted word count against the rendered Markdown word count, ending in a
PASS/FAIL line. After a batch, check exit codes and scan the output for FAIL to
find documents that need a second look:

```bash
kopitiam pdf2md "paper.pdf" -o "paper.md" || echo "conversion returned non-zero for paper.pdf"
```

Manage local AI model weights (all non-interactive):

```bash
kopitiam models list          # show the catalog and what is present on disk
kopitiam models pull <id>     # download + SHA-256 verify a model by catalog id
kopitiam models path <id>     # print expected on-disk paths (bring-your-own)
kopitiam models verify <id>   # re-check a present model against its checksums
```

### Navigate Rust code without reading whole files (the token-max loop)

When you are working *in a Rust codebase* (including kopitiam's own source),
do NOT open files blind. The whole point of these commands is to spend a few
dozen tokens on coordinates and skeletons instead of thousands on file bodies.
The loop is **measure → skeleton → coordinates → read only the slice you need**:

**1. `kopitiam tokens <path>...` BEFORE you read anything.** It prints a
deterministic BPE token estimate for a file or a whole directory (no
rust-analyzer, instant). Let the number decide read-vs-outline: a 2k-token file
you can afford to read; a 130k-token directory you must *not* — target its call
sites instead.

Takes **many paths in one call**, so budgeting a few places at once costs one
invocation, not one each. Naming the same file twice (directly, or once
directly and once through its parent directory) counts it **once** — the grand
total never double-counts.

```bash
kopitiam tokens crates/kopitiam-document/src/reconstruction   # a directory: total to budget against
kopitiam tokens apps/cli/src/main.rs                          # one file: is it cheap enough to read whole?
kopitiam tokens apps/cli/src/ocr_fallback.rs crates/kopitiam-ocr/src   # several at once: one budget for the job
kopitiam tokens --json src/ | jq '.total'                     # gate programmatically, no prose parsing
```

**2. `kopitiam outline <file>` instead of reading the file for orientation.**
It prints a body-free skeleton — every declaration with its line number, no
function bodies — roughly a 10x reduction versus a full read. Read the outline,
pick the one item you actually need, then read just those lines.

```bash
kopitiam outline crates/kopitiam-document/src/reconstruction/mod.rs
kopitiam outline --json src/foo.rs        # machine-readable items[] with line/kind/name
```

**3. `kopitiam refs` / `def` / `sig` / `callers` / `callees` / `impls` instead
of grep-then-read.** These answer "who calls this / where is it defined / what
is its signature" as `file:line:character` coordinates (and a signature), not
code bodies — the deterministic, exact answer rust-analyzer already knows. Pass
`--file <FILE>` naming the file that declares the symbol (its identifier
position is the query anchor):

```bash
kopitiam def  --file crates/kopitiam-document/src/reconstruction/tables.rs try_table   # where + signature
kopitiam sig  --file src/tables.rs try_table                                           # signature alone
kopitiam refs --file src/tables.rs try_table                                           # every call site, as coordinates
kopitiam callers --file src/tables.rs try_table --depth 2                              # up the call graph
```

Then read only the coordinates that matter — never the whole file to find them.

**4. `kopitiam check --compact` / `test --compact` instead of raw cargo
output.** `check --compact` collapses cargo's diagnostics to one deduplicated
line per distinct problem, sorted by file (one bad type stops spamming forty
identical errors); `test --compact` reports each failure as
`name — assertion @ file:line` rather than the full captured stdout. Add
`--json` to gate on the result without parsing prose:

```bash
kopitiam check --compact          # dedup'd diagnostics, one line each
kopitiam test  --compact          # failures as name — assertion @ file:line
kopitiam check --json | jq length # zero == clean, gate on it
```

**On a large workspace, `outline`/`refs`/`def`/`sig`/`callers`/`callees`/`impls`
try rust-analyzer first** and wait for it to index (default **180 s**,
overridable with the `KOPITIAM_RA_TIMEOUT_SECS` env var) before falling back.
On a workspace too big for rust-analyzer to index in bounded time — kopitiam's
own tree is one — that wait is wasted. Pass **`--no-lsp`** (alias `--syntactic`)
to skip rust-analyzer entirely and answer from a deterministic, dependency-free
scan: instant, and the guaranteed path. It is textual (grep-grade) rather than
semantic, so `refs`/`callers` hits are labelled `(syntactic, not semantic —
verify)`; use `--lsp` to *require* rust-analyzer and fail instead of falling
back when you need semantic precision.

```bash
kopitiam outline --no-lsp src/foo.rs                 # instant skeleton, no rust-analyzer
kopitiam refs --no-lsp --file src/tables.rs try_table   # instant textual call sites, verify each
KOPITIAM_RA_TIMEOUT_SECS=20 kopitiam def --file src/tables.rs try_table  # shorten the RA wait
```

## Command reference

Each subsection embeds the command's real `kopitiam ... --help` output verbatim
(clap is the source of truth) and flags whether it is agent-safe or interactive.

### `kopitiam pdf2md`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
Convert a PDF into semantic Markdown.

Runs the full Document Engine pipeline: `kopitiam-pdf` extracts text per page, `kopitiam-document` reconstructs paragraph/heading/table structure across page breaks and columns, and `kopitiam-markdown` renders the result. A validation report comparing extracted vs. rendered word counts is printed alongside the output, as a cheap sanity check that the reconstruction did not silently drop content.

Usage: kopitiam pdf2md [OPTIONS] <INPUT>

Arguments:
  <INPUT>
          Input PDF file

Options:
  -o, --output <OUTPUT>
          Output Markdown file. Defaults to the input path with a .md extension

      --engine <ENGINE>
          Extraction engine. `mupdf` (default) is the ported MuPDF `stext` engine: true reading order (columns linearised) and correct inter-word spacing. `legacy` is the older `pdf-extract`-based path, kept as a fallback
          
          [default: mupdf]

          Possible values:
          - mupdf:  The ported MuPDF `stext` engine (default): true reading order with columns linearised and correct inter-word spacing
          - legacy: The legacy `pdf-extract`-based path, kept as a fallback

      --report-json
          Print the validation report as JSON on stdout instead of the human-readable prose. The JSON carries the computed `recovery_ratio`, `passes`, and per-page ratios so a caller can gate on recovery without parsing prose (card I-F). The "Wrote ..." notices go to stderr in this mode so stdout is clean JSON

      --index
          Also write a `<output>.index.json` sidecar mapping each heading and each source page to its 1-based line range in the Markdown, turning grep-and-probe into lookup-then-slice (card I-G). The `.md` itself is untouched. With `--split-by-heading-level`, one sidecar per part

      --pages <A-B>
          Convert only source pages `A-B` (matched against the PDF's own 1-based page numbers, inclusive; a bare `N` means just page N), applied before reconstruction so a 59 MB PDF is not fully converted just to read a few pages (card I-H). The selection is a standalone document: its page anchors are renumbered from 1 across the kept pages, not carried over from the source

      --split-by-heading-level <N>
          Split the output into one `.md` per section at this heading level (1-6), instead of a single combined file, so a large multi-chapter document becomes individually safe per-chapter files (card I-H). Files are named `<stem>.NN-<slug>.md` beside `--output`

      --ocr <OCR>
          Automatic OCR fallback for scanned pages (Task #10). `auto` (default) runs OCR only on pages that extracted little or no text (a scanned, image-only page) while leaving text pages untouched; `on` forces OCR on every page; `off` disables it. A born-digital PDF is unaffected — the fallback never triggers on a page that has real text
          
          [default: auto]

          Possible values:
          - auto: Automatic fallback: OCR only the pages that yielded little or no text (the scanned pages). Pages with real text are untouched. This is the default — the fallback is on out of the box
          - on:   Force OCR on *every* page, even ones that extracted fine. For a document whose text layer is present but garbled, or an all-scanned corpus
          - off:  Disable the fallback entirely: behave exactly as the pre-OCR pipeline did

      --ocr-lang <LANGS>
          Languages to OCR with, as a comma-separated list of Tesseract codes (default `eng,chi_sim,jpn`). Each needs its `.traineddata` in the local model store; if one is missing the run fails with the `kopitiam models pull …` command to fetch it
          
          [default: eng,chi_sim,jpn]

  -h, --help
          Print help (see a summary with '-h')
```

### `kopitiam scan`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
Scan a Rust project's real tooling and report what the Semantic Runtime learned about it.

This is the first Semantic Runtime command: it runs the `kopitiam-semantic` knowledge providers (cargo metadata always, rust-analyzer optionally, rustdoc JSON when a nightly toolchain is available) against a project, merges everything they report into a `kopitiam-knowledge` graph, and prints a summary. See `apps/cli/src/scan.rs` for the full explanation of why this command exists and where it is headed.

Usage: kopitiam scan [OPTIONS]

Options:
      --root <ROOT>
          Directory containing the workspace `Cargo.toml` to scan.
          
          Defaults to the current directory. This is passed straight through to `cargo metadata`, the `rust-analyzer` process, and `cargo rustdoc`, so it must be (or be inside) a real Cargo workspace.
          
          [default: .]

      --with-rust-analyzer
          Also query a live `rust-analyzer` process over LSP for symbols.
          
          This is off by default because it has to wait for rust-analyzer to finish indexing the workspace, which can take anywhere from a few seconds to a few minutes depending on workspace size. Turn it on when you specifically want symbol-level facts (function/struct/trait names and locations), not just the package-level facts `cargo metadata` already gives you for free.

      --verbose
          Print every collected entity and relationship, not just the counts

  -h, --help
          Print help (see a summary with '-h')
```

### `kopitiam rename`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
Rename a symbol using a live rust-analyzer, previewing the change as a diff unless `--apply` is given.

See `apps/cli/src/rename.rs` for the full explanation, including why this is safe-by-default.

Usage: kopitiam rename [OPTIONS] --line <LINE> --character <CHARACTER> --new-name <NEW_NAME> <FILE>

Arguments:
  <FILE>
          The Rust source file containing the symbol to rename

Options:
      --line <LINE>
          0-indexed line of the symbol's identifier

      --character <CHARACTER>
          0-indexed character offset of the symbol's identifier, in Unicode scalar values (i.e. plain `chars()` indexing — count characters, not bytes or UTF-16 code units)

      --new-name <NEW_NAME>
          The new name for the symbol

      --root <ROOT>
          Directory containing the workspace `Cargo.toml` that `file` belongs to. Defaults to the current directory
          
          [default: .]

      --apply
          Write the computed changes to disk. Without this flag, `rename` only prints a preview diff and leaves every file untouched

  -h, --help
          Print help (see a summary with '-h')
```

### `kopitiam code-actions`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
List or apply rust-analyzer's code actions (quick fixes and refactorings) at a file position.

See `apps/cli/src/code_actions.rs` for the full explanation.

Usage: kopitiam code-actions [OPTIONS] --line <LINE> --character <CHARACTER> <FILE>

Arguments:
  <FILE>
          The Rust source file to query for code actions

Options:
      --line <LINE>
          0-indexed line to query

      --character <CHARACTER>
          0-indexed character offset to query, in Unicode scalar values

      --root <ROOT>
          Directory containing the workspace `Cargo.toml` that `file` belongs to. Defaults to the current directory
          
          [default: .]

      --apply <APPLY>
          Apply the action at this index from the listing (0-based). Without this flag, the command only lists what is available

  -h, --help
          Print help (see a summary with '-h')
```

### `kopitiam refactor`

**Command group** — not invoked directly; dispatch to one of its subcommands below.

```text
Deterministic, mechanical refactors over a directory — token-max Task II-8.

`refactor add-derive <Derive> --filter <pattern>` adds a derive to every matching `struct`/`enum`/`union`, previewing a diff unless `--apply` is given. Reuses `rename`'s `edit::{FileEdit, diff, write_file_edits}` machinery, no rust-analyzer needed. See `apps/cli/src/refactor.rs`.

Usage: kopitiam refactor <COMMAND>

Commands:
  add-derive  Add a derive to every matching type definition across a directory, previewing the change as a diff unless `--apply` is given
  help        Print this message or the help of the given subcommand(s)

Options:
  -h, --help
          Print help (see a summary with '-h')
```

#### `kopitiam refactor add-derive`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
Add a derive to every matching type definition across a directory, previewing the change as a diff unless `--apply` is given

Usage: kopitiam refactor add-derive [OPTIONS] --filter <FILTER> <DERIVE>

Arguments:
  <DERIVE>  The derive to add, e.g. `Clone`, `PartialEq`, or a path like `serde::Serialize`

Options:
      --filter <FILTER>  Restrict to type definitions whose name matches this pattern. A pattern containing `*` or `?` is treated as a glob anchored to the whole name (`Config*` matches `ConfigV2` but not `MyConfig`); otherwise it is a case-sensitive substring match (`Config` matches both). Required, so a bare run can never rewrite every type in the tree
      --root <ROOT>      Directory (or a single `.rs` file) to scan. Defaults to the current directory. This bounds the blast radius: nothing outside it is read or written, and `vendor/`, `target/`, and dot-directories are always skipped [default: .]
      --apply            Write the computed changes to disk. Without this flag, `add-derive` only prints a preview diff and leaves every file untouched
      --json             Emit the planned edits as JSON (each file with the type names and 1-based definition line numbers it would touch) instead of a diff. A listing only — it never writes, regardless of `--apply`
  -h, --help             Print help
```

### `kopitiam status`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
Print this project's persisted session memory (`.kopitiam/state.redb`).

See `apps/cli/src/status.rs`: this is the read side of the state `scan` writes, proving persistence survives across process restarts.

Usage: kopitiam status [OPTIONS]

Options:
      --root <ROOT>
          Directory containing the project's `.kopitiam` state directory. Defaults to the current directory
          
          [default: .]

      --stale
          Instead of the session summary, list conclusions whose source files have drifted (content hash mismatch) and can no longer be trusted

  -h, --help
          Print help (see a summary with '-h')
```

### `kopitiam plan`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
Run the `plan` workflow: build context from a live scan plus session memory, and invoke a model adapter.

The adapter is chosen at runtime by `crate::adapter::select_adapter`: a real on-CPU `kopitiam_ai::LocalAdapter` when a `.gguf` is present on disk, otherwise `kopitiam_ai::EchoAdapter` (the deterministic stub) with a note on how to get a real model. Either way it runs offline.

See `apps/cli/src/plan.rs`: the first `kopitiam-workflow` command, proving the full `load state -> collect facts -> build context -> invoke model -> validate -> persist` pipeline end to end.

Usage: kopitiam plan [OPTIONS] <TASK>

Arguments:
  <TASK>
          What to plan. Recorded as the project's current task (`kopitiam-workspace`) before the workflow runs, so the resulting context reflects it

Options:
      --root <ROOT>
          Directory containing the workspace `Cargo.toml` to plan against
          
          [default: .]

  -h, --help
          Print help (see a summary with '-h')
```

### `kopitiam ai`

**Command group** — not invoked directly; dispatch to one of its subcommands below.

```text
Talk to the AI layer. `ai chat` opens an interactive, streamed chat with the local model (echo stub when no `.gguf` is present, so it always runs).

This is the maintainer's testable AI interface — `temp_ai_design.md` §10.6 phase 1 (chat over `LocalAdapter`, streamed token-by-token, no tools). See `apps/cli/src/ai.rs`, whose `chat_loop` is factored over `Read`/`Write` so the streamed loop is testable headlessly.

Usage: kopitiam ai <COMMAND>

Commands:
  chat  Chat with the local model, streamed token by token
  help  Print this message or the help of the given subcommand(s)

Options:
  -h, --help
          Print help (see a summary with '-h')
```

#### `kopitiam ai chat`

**INTERACTIVE — DO NOT RUN from an agent.** Full-screen / token-streamed; it blocks on stdin and will HANG a non-interactive session.

```text
Chat with the local model, streamed token by token.

Type a line, press enter, watch the reply stream in. `/quit` (or `/exit`, or an EOF / Ctrl-D) ends the session. With no local `.gguf` on disk this echoes your line back via the deterministic stub, so it runs even with no weights and no network.

Usage: kopitiam ai chat [OPTIONS]

Options:
      --system <SYSTEM>
          System prompt to seed the conversation with. A gentle default is used when omitted
          
          [default: "You are KOPITIAM's local assistant. Answer concisely and helpfully."]

      --max-tokens <MAX_TOKENS>
          Cap on tokens generated per reply. Left to the model/adapter default when omitted

  -h, --help
          Print help (see a summary with '-h')
```

### `kopitiam tui`

**INTERACTIVE — DO NOT RUN from an agent.** Full-screen / token-streamed; it blocks on stdin and will HANG a non-interactive session.

```text
Open the KOPITIAM chat TUI: a full-screen, kopitiam-themed terminal interface over the same streamed local model `ai chat` uses.

A ratatui front-end onto `crate::adapter::select_adapter` — a real on-CPU `kopitiam_ai::LocalAdapter` when a `.gguf` is present, otherwise the deterministic `EchoAdapter`, so it always runs offline. Tokens stream live into the transcript by polling the adapter's `Receiver<StreamChunk>`; no business logic lives in the UI. This is the runnable slice of `temp_ai_design.md`'s "full ratatui" phase.

Usage: kopitiam tui [OPTIONS]

Options:
      --system <SYSTEM>
          System prompt seeding the AI Chat view. A gentle default is used when omitted
          
          [default: "You are KOPITIAM's local assistant. Answer concisely and helpfully."]

      --max-tokens <MAX_TOKENS>
          Cap on tokens generated per reply. Left to the adapter default when omitted

  -h, --help
          Print help (see a summary with '-h')
```

### `kopitiam view`

**INTERACTIVE — DO NOT RUN from an agent.** Full-screen / token-streamed; it blocks on stdin and will HANG a non-interactive session.

```text
Open a PDF in an on-screen terminal viewer, rendering pages as images.

A standalone, interactive image viewer over the ported MuPDF rasteriser (`kopitiam_pdf::mupdf::rasterize_page` — real glyph outlines, vector, colour and images) displayed through `ratatui-image`, which auto-detects the terminal's graphics protocol (kitty / sixel / iTerm2) and falls back to Unicode half-blocks under Termux. Keys: j/k or arrows or PgUp/PgDn to page, `g` to go to a page, `+`/`-` to zoom, `r`/`i` to toggle the reflow (Markdown) view, `q` to quit. This is what `kmux latex` shells out to for its live preview. See `apps/cli/src/view.rs`; the page-rendering path is shared with `kopitiam tui`'s image mode (no forked viewer logic).

Usage: kopitiam view [OPTIONS] <PDF>

Arguments:
  <PDF>
          The PDF file to open

Options:
      --page <PAGE>
          Initial 1-based page to open on. Clamped into range if out of bounds
          
          [default: 1]

      --dpi <DPI>
          Initial render resolution in DPI. Higher is sharper but slower to render; adjust live with `+` / `-` once open
          
          [default: 150]

  -h, --help
          Print help (see a summary with '-h')
```

### `kopitiam models`

**Command group** — not invoked directly; dispatch to one of its subcommands below.

```text
Go and get, then check, the local model weights the AI layer runs on.

Group of four actions — `list`, `pull`, `path`, `verify` — over the `kopitiam-models` model store. `pull` is the autofetch path (download plus SHA-256 verify from the catalog); a user who already got the file can drop it where `path` say and skip the network (bring-your-own). This keeps `CLAUDE.md`'s Offline-First promise real: no local weights, no local model. See `apps/cli/src/models.rs` for the full story.

Usage: kopitiam models <COMMAND>

Commands:
  list    List every model in the built-in catalog, and whether got already locally or not
  pull    Go and get a model by downloading and verifying its artifacts (autofetch)
  path    Print the on-disk artifact path(s) for a model id
  verify  Check that a present model's artifacts still match their catalog checksums
  help    Print this message or the help of the given subcommand(s)

Options:
  -h, --help
          Print help (see a summary with '-h')
```

#### `kopitiam models list`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
List every model in the built-in catalog, and whether got already locally or not.

Read `kopitiam_models::Catalog::builtin()` for the catalog and check each entry against the default model store, so the `present?` column show what is really on disk right now (whether `pull` fetch it or you dropped it in by hand).

Usage: kopitiam models list

Options:
  -h, --help
          Print help (see a summary with '-h')
```

#### `kopitiam models pull`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
Go and get a model by downloading and verifying its artifacts (autofetch).

This is the network path: it resolve the id in the catalog, then hand the whole download-and-verify job to `kopitiam_models::ensure_available`, streaming live progress to the terminal. If you already got the weights on disk, no need this one — see `kopitiam models path` for the bring-your-own flow.

Usage: kopitiam models pull <ID>

Arguments:
  <ID>
          Catalog id of the model to go and get (see `kopitiam models list`)

Options:
  -h, --help
          Print help (see a summary with '-h')
```

#### `kopitiam models path`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
Print the on-disk artifact path(s) for a model id.

Also doubles up as the bring-your-own guide: these are the exact paths the store is expecting, so putting each artifact there make the model available without ever running `pull`. Exit nonzero if the model not present yet, and point you to `kopitiam models pull`.

Usage: kopitiam models path <ID>

Arguments:
  <ID>
          Catalog id of the model to locate (see `kopitiam models list`)

Options:
  -h, --help
          Print help (see a summary with '-h')
```

#### `kopitiam models verify`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
Check that a present model's artifacts still match their catalog checksums.

Useful after a bring-your-own copy, or to catch a corrupted or truncated download. Hand everything to `kopitiam_models::ModelStore::verify`.

Usage: kopitiam models verify <ID>

Arguments:
  <ID>
          Catalog id of the model to check (see `kopitiam models list`)

Options:
  -h, --help
          Print help (see a summary with '-h')
```

### `kopitiam outline`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
Print a file's items-only skeleton (declarations + line numbers, no bodies) — token-max Task II-2.

A ~10x-smaller orientation pass than reading the whole file. See `apps/cli/src/outline.rs`; the real work is `kopitiam_semantic::outline`.

Usage: kopitiam outline [OPTIONS] <FILE>

Arguments:
  <FILE>
          The Rust source file to outline

Options:
      --root <ROOT>
          Directory containing the workspace `Cargo.toml` that `file` belongs to. Defaults to the current directory; passed to rust-analyzer as the root
          
          [default: .]

      --json
          Emit the outline as JSON (the serialized [`Outline`]: `items` with `line`/`kind`/`name`/`detail`/`depth`) instead of the human skeleton. Progress notices go to stderr so stdout stays clean JSON (§0.2)

      --no-lsp
          Skip rust-analyzer entirely and produce the outline by a deterministic, dependency-free Rust item scan (no cross-file resolution, no server). Instant, and the guaranteed path on a workspace too large for rust-analyzer to index in a bounded time
          
          [aliases: --syntactic]

      --lsp
          Require rust-analyzer: on timeout/unavailability fail (non-zero exit) instead of falling back to the syntactic scan

  -h, --help
          Print help (see a summary with '-h')
```

### `kopitiam refs`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
List every reference/call site of a symbol as `file:line:character` coordinates — token-max Task II-1. See `apps/cli/src/semq.rs`

Usage: kopitiam refs [OPTIONS] --file <FILE> <SYMBOL>

Arguments:
  <SYMBOL>  The symbol (or, for `impls`, trait) name to resolve

Options:
      --file <FILE>  The file whose `documentSymbol` declares the symbol; its identifier position is resolved there and used as the query anchor
      --root <ROOT>  Directory containing the workspace `Cargo.toml`. Defaults to the current directory; passed to rust-analyzer as the root [default: .]
      --json         Emit JSON coordinates instead of the human `file:line:character` form
      --no-lsp       Skip rust-analyzer and answer from a workspace-wide textual search (`refs` only) — grep-grade, not semantic, so each hit is labelled `(syntactic, not semantic — verify)`. The guaranteed path when rust-analyzer cannot index the workspace in a bounded time [aliases: --syntactic]
      --lsp          Require rust-analyzer: on timeout/unavailability fail (non-zero exit) instead of falling back
  -h, --help         Print help
```

### `kopitiam def`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
Print where a symbol is defined plus its signature — token-max Task II-1

Usage: kopitiam def [OPTIONS] --file <FILE> <SYMBOL>

Arguments:
  <SYMBOL>  The symbol (or, for `impls`, trait) name to resolve

Options:
      --file <FILE>  The file whose `documentSymbol` declares the symbol; its identifier position is resolved there and used as the query anchor
      --root <ROOT>  Directory containing the workspace `Cargo.toml`. Defaults to the current directory; passed to rust-analyzer as the root [default: .]
      --json         Emit JSON coordinates instead of the human `file:line:character` form
      --no-lsp       Skip rust-analyzer and answer from a workspace-wide textual search (`refs` only) — grep-grade, not semantic, so each hit is labelled `(syntactic, not semantic — verify)`. The guaranteed path when rust-analyzer cannot index the workspace in a bounded time [aliases: --syntactic]
      --lsp          Require rust-analyzer: on timeout/unavailability fail (non-zero exit) instead of falling back
  -h, --help         Print help
```

### `kopitiam sig`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
Print a symbol's signature alone — token-max Task II-1

Usage: kopitiam sig [OPTIONS] --file <FILE> <SYMBOL>

Arguments:
  <SYMBOL>  The symbol (or, for `impls`, trait) name to resolve

Options:
      --file <FILE>  The file whose `documentSymbol` declares the symbol; its identifier position is resolved there and used as the query anchor
      --root <ROOT>  Directory containing the workspace `Cargo.toml`. Defaults to the current directory; passed to rust-analyzer as the root [default: .]
      --json         Emit JSON coordinates instead of the human `file:line:character` form
      --no-lsp       Skip rust-analyzer and answer from a workspace-wide textual search (`refs` only) — grep-grade, not semantic, so each hit is labelled `(syntactic, not semantic — verify)`. The guaranteed path when rust-analyzer cannot index the workspace in a bounded time [aliases: --syntactic]
      --lsp          Require rust-analyzer: on timeout/unavailability fail (non-zero exit) instead of falling back
  -h, --help         Print help
```

### `kopitiam callers`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
List a function's callers (call sites + enclosing function), recursed to `--depth` — token-max Task II-1

Usage: kopitiam callers [OPTIONS] --file <FILE> <SYMBOL>

Arguments:
  <SYMBOL>  The symbol (or, for `impls`, trait) name to resolve

Options:
      --file <FILE>    The file whose `documentSymbol` declares the symbol; its identifier position is resolved there and used as the query anchor
      --root <ROOT>    Directory containing the workspace `Cargo.toml`. Defaults to the current directory; passed to rust-analyzer as the root [default: .]
      --json           Emit JSON coordinates instead of the human `file:line:character` form
      --no-lsp         Skip rust-analyzer and answer from a workspace-wide textual search (`refs` only) — grep-grade, not semantic, so each hit is labelled `(syntactic, not semantic — verify)`. The guaranteed path when rust-analyzer cannot index the workspace in a bounded time [aliases: --syntactic]
      --lsp            Require rust-analyzer: on timeout/unavailability fail (non-zero exit) instead of falling back
      --depth <DEPTH>  How many hops of the call graph to follow (1 = direct callers/callees) [default: 1]
  -h, --help           Print help
```

### `kopitiam callees`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
List the functions a function calls, recursed to `--depth` — token-max Task II-1

Usage: kopitiam callees [OPTIONS] --file <FILE> <SYMBOL>

Arguments:
  <SYMBOL>  The symbol (or, for `impls`, trait) name to resolve

Options:
      --file <FILE>    The file whose `documentSymbol` declares the symbol; its identifier position is resolved there and used as the query anchor
      --root <ROOT>    Directory containing the workspace `Cargo.toml`. Defaults to the current directory; passed to rust-analyzer as the root [default: .]
      --json           Emit JSON coordinates instead of the human `file:line:character` form
      --no-lsp         Skip rust-analyzer and answer from a workspace-wide textual search (`refs` only) — grep-grade, not semantic, so each hit is labelled `(syntactic, not semantic — verify)`. The guaranteed path when rust-analyzer cannot index the workspace in a bounded time [aliases: --syntactic]
      --lsp            Require rust-analyzer: on timeout/unavailability fail (non-zero exit) instead of falling back
      --depth <DEPTH>  How many hops of the call graph to follow (1 = direct callers/callees) [default: 1]
  -h, --help           Print help
```

### `kopitiam impls`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
List a trait's `impl` sites — token-max Task II-1

Usage: kopitiam impls [OPTIONS] --file <FILE> <SYMBOL>

Arguments:
  <SYMBOL>  The symbol (or, for `impls`, trait) name to resolve

Options:
      --file <FILE>  The file whose `documentSymbol` declares the symbol; its identifier position is resolved there and used as the query anchor
      --root <ROOT>  Directory containing the workspace `Cargo.toml`. Defaults to the current directory; passed to rust-analyzer as the root [default: .]
      --json         Emit JSON coordinates instead of the human `file:line:character` form
      --no-lsp       Skip rust-analyzer and answer from a workspace-wide textual search (`refs` only) — grep-grade, not semantic, so each hit is labelled `(syntactic, not semantic — verify)`. The guaranteed path when rust-analyzer cannot index the workspace in a bounded time [aliases: --syntactic]
      --lsp          Require rust-analyzer: on timeout/unavailability fail (non-zero exit) instead of falling back
  -h, --help         Print help
```

### `kopitiam check`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
Run `cargo check` and report one deduplicated line per distinct diagnostic, sorted by file — token-max Task II-4.

The dedup is the win: one bad type produces the same diagnostic across every target, and this collapses them. See `apps/cli/src/diagnostics.rs`.

Usage: kopitiam check [OPTIONS]

Options:
      --root <ROOT>
          Directory to run `cargo check` in. Defaults to the current directory
          
          [default: .]

  -p, --package <PACKAGE>
          Restrict to one package (`cargo check -p <PACKAGE>`)

      --compact
          Collapse the diagnostics to one deduplicated line per distinct problem, sorted by file. Without this (and without `--json`) the raw cargo output streams through unchanged

      --json
          Emit the deduplicated diagnostics as JSON (implies the compact analysis)

  -h, --help
          Print help (see a summary with '-h')
```

### `kopitiam test`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
Run `cargo test` and report each failure as `name — assertion @ file:line`, not the full captured stdout — token-max Task II-4

Usage: kopitiam test [OPTIONS]

Options:
      --root <ROOT>        Directory to run `cargo test` in. Defaults to the current directory [default: .]
  -p, --package <PACKAGE>  Restrict to one package (`cargo test -p <PACKAGE>`)
      --compact            Report each failure as one line (`name — assertion @ file:line`) instead of the full captured output. Without this (and without `--json`) the raw cargo output streams through unchanged
      --json               Emit the failures as JSON (implies the compact analysis)
  -h, --help               Print help
```

### `kopitiam tokens`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
Estimate the BPE token cost of a file or directory, so an agent chooses read-vs-outline informed — token-max Task II-7.

A thin shell over `kopitiam_tokenizer::estimate_tokens`. See `apps/cli/src/tokens.rs`.

Usage: kopitiam tokens [OPTIONS] <PATHS>...

Arguments:
  <PATHS>...
          One or more files / directories to estimate. Can pass many in one go — `tokens src/a.rs crates/b/src` — so you sizing up a few places at once don't need one call each. A directory is walked recursively and every readable UTF-8 file is summed; unreadable or non-UTF-8 files (binaries) are skipped, not counted.
          
          Overlapping paths are safe: the same file named twice (directly, or once directly and once via a parent directory) is counted **once**, so the grand total never double-counts. Files come out in the order you named their path, and sorted within each directory, so output stay deterministic.

Options:
      --json
          Emit machine-readable JSON: a per-file breakdown — each with its total and a per-line token count (`estimate_tokens_by_line`) — plus the grand total, instead of the human summary. (§0.2: a caller gates on the number without parsing prose.)

      --by-line
          Also print the per-line breakdown in the human output (it is always in `--json`). Off by default so a single-file estimate stays one line

  -h, --help
          Print help (see a summary with '-h')
```

### `kopitiam translate`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
Translate a converted Markdown document end to end — token-max Tasks III-4..7.

Segments the document (`kopitiam_document::segments`), reuses cached translations from the `.kopitiam` translation memory, drafts cache-misses with the local model and routes each (`kopitiam_ai::draft_and_route`), applies a `--glossary` deterministically, and writes aligned bilingual output with per-segment anchors. See `apps/cli/src/translate.rs`.

Usage: kopitiam translate [OPTIONS] <INPUT>

Arguments:
  <INPUT>
          The converted Markdown document to translate (typically a `pdf2md` output). Split into segments at block boundaries; page anchors and other bare HTML comments are not translatable and are skipped

Options:
      --glossary <GLOSSARY>
          A project glossary applied deterministically as the post-pass (III-5), in the simple `source = target` line format (`#` comments allowed). Every occurrence of a source term becomes byte-identical target text — zero model tokens spent on terminology, no drift across the document

      --layout <LAYOUT>
          Bilingual layout: `interleaved` (anchor, source, target-as-blockquote per segment) or `table` (one Markdown table, `seg | source | target | review` rows). Both carry the stable `<!-- seg N -->` anchors (III-7)
          
          [default: interleaved]

          Possible values:
          - interleaved: Anchor, source, then target as a blockquote, per segment
          - table:       A single `seg | source | target | review` Markdown table

      --no-cache
          Skip the translation memory entirely: neither reuse cached translations nor record new ones. Every segment is (re-)drafted. Without this flag the TM in `<root>/.kopitiam` is consulted and updated (III-4)

  -o, --output <OUTPUT>
          Where to write the bilingual Markdown. Defaults to the input path with a `.bilingual.md` extension beside it

      --root <ROOT>
          Directory holding the project's `.kopitiam` translation-memory store. Defaults to the current directory
          
          [default: .]

      --json
          Emit the machine-readable report (`reuse_fraction`, the two-pass summary, and `review_targets`) as JSON instead of the human summary, so a caller gates on the saving without parsing prose (§0.2). The "Wrote ..." notice and the adapter notice go to stderr in this mode

  -h, --help
          Print help (see a summary with '-h')
```

### `kopitiam digest`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
Print (and cache) a compact per-crate architecture digest — token-max Task II-3.

`cargo metadata` → crate → responsibility → workspace-internal deps, persisted in `.kopitiam/state.redb` and regenerated only when a manifest hash changes. See `apps/cli/src/digest.rs`.

Usage: kopitiam digest [OPTIONS]

Options:
      --root <ROOT>
          The workspace root (holding the top `Cargo.toml` and `.kopitiam`). Defaults to the current directory
          
          [default: .]

      --refresh
          Force a rebuild from `cargo metadata` even if the cached digest is still fresh for the current manifests

      --json
          Print the digest as JSON (crate → responsibility → deps + the source hash) instead of the human-readable listing (§0.2). Notices go to stderr

  -h, --help
          Print help (see a summary with '-h')
```

### `kopitiam port`

**Command group** — not invoked directly; dispatch to one of its subcommands below.

```text
Code-translation (porting) helpers — token-max Tasks III-1 / III-3.

`port status` surfaces the machine-maintained port ledger and `port skeleton <file>` emits Rust signature stubs, as thin wrappers over the committed `scripts/port-ledger.sh` / `scripts/skeleton-gen.sh`. See `apps/cli/src/port.rs`.

Usage: kopitiam port <COMMAND>

Commands:
  status    Show the port ledger (`scripts/port-ledger.sh --report`), or the raw `docs/port-ledger.json` with `--json`
  skeleton  Generate Rust signature stubs for a vendored source file (`scripts/skeleton-gen.sh <file>`)
  help      Print this message or the help of the given subcommand(s)

Options:
  -h, --help
          Print help (see a summary with '-h')
```

#### `kopitiam port status`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
Show the port ledger (`scripts/port-ledger.sh --report`), or the raw `docs/port-ledger.json` with `--json`

Usage: kopitiam port status [OPTIONS]

Options:
      --root <ROOT>  Where to start looking for the kopitiam repo root (the directory holding `scripts/port-ledger.sh`). Defaults to the current directory; the root is found by walking up from here [default: .]
      --json         Print `docs/port-ledger.json` directly (the machine view), instead of shelling the script's human `--report`. No subprocess is spawned
  -h, --help         Print help
```

#### `kopitiam port skeleton`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
Generate Rust signature stubs for a vendored source file (`scripts/skeleton-gen.sh <file>`)

Usage: kopitiam port skeleton [OPTIONS] <FILE>

Arguments:
  <FILE>  The vendored source file to generate signature stubs for

Options:
      --root <ROOT>  Where to start looking for the repo root (holding `scripts/skeleton-gen.sh`). Defaults to the current directory [default: .]
      --report       Pass `--report` to the script: print its found/skipped breakdown instead of the stubs
  -h, --help         Print help
```

### `kopitiam preprocess`

**Command group** — not invoked directly; dispatch to one of its subcommands below.

```text
Route high-volume, low-judgment work to the local model — token-max Task II-6.

`preprocess summarize <file>` compresses and `preprocess triage <query> <candidate>...` filters, both over `kopitiam_ai`'s preprocess helpers with a `DropReport` of what was set aside. See `apps/cli/src/preprocess.rs`.

Usage: kopitiam preprocess <COMMAND>

Commands:
  summarize  Compress a file to at most `--lines N` lines via the local model
  triage     Keep only the candidates plausibly relevant to a query (conservative: keeps all on an unusable reply)
  help       Print this message or the help of the given subcommand(s)

Options:
  -h, --help
          Print help (see a summary with '-h')
```

#### `kopitiam preprocess summarize`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
Compress a file to at most `--lines N` lines via the local model

Usage: kopitiam preprocess summarize [OPTIONS] <FILE>

Arguments:
  <FILE>  The file to compress

Options:
      --lines <LINES>  The line budget the summary is hard-capped to (overflow is listed in the drop report, never silently discarded) [default: 10]
      --json           Emit the `Preprocessed` result (output + drop report) as JSON
  -h, --help           Print help
```

#### `kopitiam preprocess triage`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
Keep only the candidates plausibly relevant to a query (conservative: keeps all on an unusable reply)

Usage: kopitiam preprocess triage [OPTIONS] <QUERY> [CANDIDATES]...

Arguments:
  <QUERY>          What the candidates are being filtered for relevance to
  [CANDIDATES]...  The candidate snippets (e.g. grep hits) to filter

Options:
      --json  Emit the `Preprocessed` result (kept subset + drop report) as JSON
  -h, --help  Print help
```

### `kopitiam bn`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
Run the `kopi-beans` task ledger via its `bn` binary — issue #30.

An arm's-length passthrough: every argument after `bn` is forwarded verbatim to the `bn` binary on `PATH` (`kopitiam bn create "x" -t task` runs `bn create "x" -t task`), with stdin/stdout/stderr inherited and the child's exit code propagated. `kopitiam` takes no `kopi-beans` dependency and stays pure-Rust; if `bn` is not installed the command prints an install hint and exits non-zero. Non-interactive. See `apps/cli/src/bn.rs`.

Usage: kopitiam bn [ARGS]...

Arguments:
  [ARGS]...
          Every argument after `bn`, forwarded to the `bn` binary untouched

Options:
  -h, --help
          Print help (see a summary with '-h')
```

## Gotchas

- **Quote paths with spaces.** `kopitiam pdf2md "My Paper (v2).pdf" -o "My Paper.md"`.
  Unquoted paths split on whitespace and the command will read the wrong file
  or fail.
- **`pdf2md` prints a PASS/FAIL recovery report.** It compares extracted vs.
  rendered word counts as a cheap check that structure reconstruction did not
  silently drop content. Treat FAIL (or a large word-count gap) as a signal to
  inspect that document, not necessarily a hard error.
- **Some PDFs skip unreadable pages.** A page that cannot be extracted may be
  omitted from the Markdown; the report is how you notice.
- **Scanned pages are OCR'd automatically.** `pdf2md` detects image-only
  (low-text) pages and recognizes them with the built-in Tesseract LSTM engine —
  `--ocr auto` (the default). It needs the language `.traineddata` in the model
  store; the default `--ocr-lang eng,chi_sim,jpn`, so first run
  `kopitiam models pull tessdata-eng` (and `tessdata-chi_sim` / `tessdata-jpn` as
  needed), or the run fails telling you which to pull. Use `--ocr off` to disable
  it or `--ocr on` to force OCR on every page. Born-digital PDFs are untouched
  (the fallback never triggers on a page with real text).
- **`tui`, `ai chat`, and `view` are interactive** and must never be launched by
  an agent (see CRITICAL agent guidance above). `view` in particular is the
  on-screen page viewer `kmux latex` opens for a human's live preview.

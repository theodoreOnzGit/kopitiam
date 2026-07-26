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
- **NEVER run `kopitiam tui` or `kopitiam ai chat` from an agent.** Both are
  INTERACTIVE programs: `tui` is a full-screen terminal UI and `ai chat`
  streams tokens while blocking on stdin. From a non-interactive agent they do
  not return — they will **HANG the session**. They exist for humans at a real
  terminal, not for automation.
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

## Command reference

Each subsection embeds the command's real `kopitiam ... --help` output verbatim
(clap is the source of truth) and flags whether it is agent-safe or interactive.

### `kopitiam pdf2md`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
Convert a PDF into semantic Markdown.

Runs the full Document Engine pipeline: `kopitiam-pdf` extracts text per page, `kopitiam-document` reconstructs paragraph/heading/table structure across page breaks and columns, and `kopitiam-markdown` renders the result. A validation report comparing extracted vs. rendered word counts is printed alongside the output, as a cheap sanity check that the reconstruction did not silently drop content.

Usage: kopitiam.exe pdf2md [OPTIONS] <INPUT>

Arguments:
  <INPUT>
          Input PDF file

Options:
  -o, --output <OUTPUT>
          Output Markdown file. Defaults to the input path with a .md extension

  -h, --help
          Print help (see a summary with '-h')
```

### `kopitiam scan`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
Scan a Rust project's real tooling and report what the Semantic Runtime learned about it.

This is the first Semantic Runtime command: it runs the `kopitiam-semantic` knowledge providers (cargo metadata always, rust-analyzer optionally, rustdoc JSON when a nightly toolchain is available) against a project, merges everything they report into a `kopitiam-knowledge` graph, and prints a summary. See `apps/cli/src/scan.rs` for the full explanation of why this command exists and where it is headed.

Usage: kopitiam.exe scan [OPTIONS]

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

Usage: kopitiam.exe rename [OPTIONS] --line <LINE> --character <CHARACTER> --new-name <NEW_NAME> <FILE>

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

Usage: kopitiam.exe code-actions [OPTIONS] --line <LINE> --character <CHARACTER> <FILE>

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

### `kopitiam status`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
Print this project's persisted session memory (`.kopitiam/state.redb`).

See `apps/cli/src/status.rs`: this is the read side of the state `scan` writes, proving persistence survives across process restarts.

Usage: kopitiam.exe status [OPTIONS]

Options:
      --root <ROOT>
          Directory containing the project's `.kopitiam` state directory. Defaults to the current directory
          
          [default: .]

  -h, --help
          Print help (see a summary with '-h')
```

### `kopitiam plan`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
Run the `plan` workflow: build context from a live scan plus session memory, and invoke a model adapter.

The adapter is chosen at runtime by `crate::adapter::select_adapter`: a real on-CPU `kopitiam_ai::LocalAdapter` when a `.gguf` is present on disk, otherwise `kopitiam_ai::EchoAdapter` (the deterministic stub) with a note on how to get a real model. Either way it runs offline.

See `apps/cli/src/plan.rs`: the first `kopitiam-workflow` command, proving the full `load state -> collect facts -> build context -> invoke model -> validate -> persist` pipeline end to end.

Usage: kopitiam.exe plan [OPTIONS] <TASK>

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

Usage: kopitiam.exe ai <COMMAND>

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

Usage: kopitiam.exe ai chat [OPTIONS]

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

Usage: kopitiam.exe tui [OPTIONS]

Options:
      --system <SYSTEM>
          System prompt seeding the AI Chat view. A gentle default is used when omitted
          
          [default: "You are KOPITIAM's local assistant. Answer concisely and helpfully."]

      --max-tokens <MAX_TOKENS>
          Cap on tokens generated per reply. Left to the adapter default when omitted

  -h, --help
          Print help (see a summary with '-h')
```

### `kopitiam models`

**Command group** — not invoked directly; dispatch to one of its subcommands below.

```text
Go and get, then check, the local model weights the AI layer runs on.

Group of four actions — `list`, `pull`, `path`, `verify` — over the `kopitiam-models` model store. `pull` is the autofetch path (download plus SHA-256 verify from the catalog); a user who already got the file can drop it where `path` say and skip the network (bring-your-own). This keeps `CLAUDE.md`'s Offline-First promise real: no local weights, no local model. See `apps/cli/src/models.rs` for the full story.

Usage: kopitiam.exe models <COMMAND>

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

Usage: kopitiam.exe models list

Options:
  -h, --help
          Print help (see a summary with '-h')
```

#### `kopitiam models pull`

**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.

```text
Go and get a model by downloading and verifying its artifacts (autofetch).

This is the network path: it resolve the id in the catalog, then hand the whole download-and-verify job to `kopitiam_models::ensure_available`, streaming live progress to the terminal. If you already got the weights on disk, no need this one — see `kopitiam models path` for the bring-your-own flow.

Usage: kopitiam.exe models pull <ID>

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

Usage: kopitiam.exe models path <ID>

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

Usage: kopitiam.exe models verify <ID>

Arguments:
  <ID>
          Catalog id of the model to check (see `kopitiam models list`)

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
- **Do not rely on OCR yet.** An automatic OCR fallback for image-only or
  scanned pages is *planned* but not something to depend on today. If a PDF is
  scanned images, `pdf2md` may produce little or no text.
- **`tui` and `ai chat` are interactive** and must never be launched by an
  agent (see CRITICAL agent guidance above).

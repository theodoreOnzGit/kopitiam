# kopitiam

The command-line interface to KOPITIAM's **Semantic Runtime**: a local-first
engine that turns real tooling (`cargo`, `rust-analyzer`, `rustdoc`) into
structured facts about a Rust project, instead of asking a language model to
re-read and re-guess the codebase every time.

> **If you are an AI agent using this tool**: everything you need is on this
> page. Prefer these commands over `grep`/`find`/manual file reads when the
> question is "what does this project look like" or "rename this symbol
> everywhere" — the runtime already has (or can get) a deterministic answer,
> computed by real tools, not inferred from text.

## Install

```bash
cargo install kopitiam
```

Requires a stable Rust toolchain (this crate targets edition 2024). No other
runtime dependency is required for `scan` (cargo-metadata facts) or
`pdf2md`. Two commands opportunistically use tools that may not be present,
and degrade to doing less rather than failing:

- `rust-analyzer` on `PATH` is required for `rename`, `code-actions`, and
  `scan --with-rust-analyzer`. If it's missing, `scan --with-rust-analyzer`
  silently contributes zero facts from that provider; `rename` and
  `code-actions` will error clearly since there is nothing to fall back to.
- A `nightly` Rust toolchain is required for `scan`'s `rustdoc-json`
  provider (rustdoc's JSON output is nightly-only as of this writing:
  rust-lang/rust#76578). If none is installed, this provider silently
  contributes zero facts — **this tool never runs `rustup toolchain
  install` or otherwise triggers an implicit toolchain download**; it only
  checks `rustup toolchain list`.

## Command overview

| Command | Reads or writes? | What it does |
|---|---|---|
| `kopitiam scan` | read-only | Runs cargo/rust-analyzer/rustdoc providers, reports entity/relationship counts, records that a scan happened in session state. |
| `kopitiam status` | read-only | Prints the persisted session state (current task, working set, last-updated time) from `.kopitiam/`. |
| `kopitiam rename` | writes (gated) | Renames a symbol project-wide via rust-analyzer. Prints a diff by default; only writes with `--apply`. |
| `kopitiam code-actions` | writes (gated) | Lists rust-analyzer quick fixes/refactorings at a position; applies one by index with `--apply <INDEX>`. |
| `kopitiam pdf2md` | writes (output file) | Converts a PDF to semantic Markdown via the Document Engine. |

Every subcommand accepts `--help` for the authoritative, current flag list
(`kopitiam <command> --help`); the tables below describe intent and the
common flags, but `--help` is the source of truth if they ever drift.

### `kopitiam scan [--root PATH] [--with-rust-analyzer] [--verbose]`

Scans a Cargo project and reports what the Semantic Runtime learned about
it. This is the fastest way for an agent to get an accurate, tool-verified
picture of a Rust workspace's packages, targets, internal dependency graph,
and (optionally) every function/struct/trait/enum symbol — all without
reading a single source file directly.

- `--root PATH` — directory containing the workspace `Cargo.toml`. Defaults
  to the current directory.
- `--with-rust-analyzer` — also query a live rust-analyzer process for
  symbol-level facts. Off by default because indexing a workspace can take
  anywhere from seconds to a couple of minutes; turn it on when you
  specifically need symbol names/locations, not just package-level facts.
- `--verbose` — print every collected entity, not just per-kind counts.

Example:

```bash
kopitiam scan --root . --with-rust-analyzer
```

```
  cargo-metadata   +33 entities, +13 relationships (graph now has 33 entities)
  rustdoc-json     +381 entities, +349 relationships (graph now has 414 entities)
  rust-analyzer    +234 entities, +128 relationships (graph now has 648 entities)

Semantic graph: 648 entities, 490 relationships
  Artifact: 179
  Symbol: 469
```

### `kopitiam status [--root PATH]`

Prints what a previous `scan` (or other command) recorded about this
project in `.kopitiam/state.redb` — current task, recently touched
artifacts, and a timestamp. Use this to recover context across sessions
without re-deriving it, and before assuming you need to re-scan.

### `kopitiam rename FILE --line N --character N --new-name NAME [--root PATH] [--apply]`

Renames the symbol at a file position everywhere it's referenced in the
workspace, using a live rust-analyzer. **Safe by default**: without
`--apply`, it only prints a unified diff and touches nothing on disk. Only
pass `--apply` once the diff looks correct.

- `--line` — 0-indexed line number of the identifier.
- `--character` — 0-indexed character offset of the identifier, counted as
  Unicode scalar values (i.e. plain `chars()` indexing — count characters,
  not bytes and not UTF-16 code units).
- `--new-name` — the replacement identifier.
- `--root` — directory containing the workspace `Cargo.toml`. Defaults to
  the current directory.
- `--apply` — write the changes. Omit this to preview only.

Example (preview):

```bash
kopitiam rename src/lib.rs --line 0 --character 7 --new-name add_numbers
```

```
--- src/lib.rs
+++ src/lib.rs
@@ -1,4 +1,4 @@
-pub fn add(left: u64, right: u64) -> u64 {
+pub fn add_numbers(left: u64, right: u64) -> u64 {
     left + right
 }

(preview only; re-run with --apply to write these changes)
```

Re-run with `--apply` to actually write it.

### `kopitiam code-actions FILE --line N --character N [--root PATH] [--apply INDEX]`

Lists (or applies) rust-analyzer's code actions — quick fixes and
refactorings — at a file position.

- Without `--apply`: prints a numbered list of available actions and does
  nothing else.
- With `--apply <INDEX>`: executes the action at that index. Unlike
  `rename`, this writes immediately — picking a specific, named action from
  the listing is already the deliberate step.

```bash
kopitiam code-actions src/lib.rs --line 12 --character 4
#   [0] Add missing impl members
#   [1] Extract into function
kopitiam code-actions src/lib.rs --line 12 --character 4 --apply 1
```

### `kopitiam pdf2md INPUT [-o OUTPUT]`

Converts a PDF into semantic Markdown via the Document Engine
(`kopitiam-pdf` extraction → `kopitiam-document` structural reconstruction
across page breaks/columns → `kopitiam-markdown` rendering), printing a
validation report comparing extracted vs. rendered word counts.

## Design notes an agent should know

- **Position encoding**: every `--line`/`--character` pair in this tool is
  0-indexed, and `--character` is a Unicode scalar value (char) count, not a
  byte offset or a UTF-16 code unit count. If you computed a position from
  raw file bytes, convert to a char index first (e.g. via `str::chars()`
  in Rust, or `len()` of a Python `str` slice, not a UTF-8 byte length).
- **No hidden network or system calls beyond the tools named above.**
  `scan`'s rustdoc provider explicitly avoids ever invoking `cargo
  +nightly ...` to *check* for nightly, specifically because some `rustup`
  configurations auto-install a missing toolchain the moment it's invoked —
  this tool only reads `rustup toolchain list`.
- **State lives in `.kopitiam/`** at the project root (next to
  `Cargo.toml`), analogous to `.git`. It is local, plain-file, and
  gitignored by default in the KOPITIAM repository itself; nothing in it is
  meant to be committed.
- **This tool never guesses what a tool can answer deterministically.**
  If a question ("does this function have tests?", "what does this crate
  depend on?") is answerable by `scan`, prefer that over reading source.

## Why this exists

KOPITIAM's thesis is that project understanding should live in a runtime
both humans and AI share, not be re-derived from scratch (or from a chat
transcript) every session. This CLI is the current, still-growing surface
of that runtime — see the main project repository for the full
architecture (`kopitiam-knowledge`'s semantic graph, `kopitiam-index`'s
persistence, and more) and its governing philosophy.

Repository: <https://github.com/theodoreOnzGit/kopitiam>

## License

Copyright (C) 2026 Theodore Kay Chen Ong. Licensed under AGPL-3.0-only.

This is a personal project of Theodore Kay Chen Ong, built in his personal
capacity outside of working hours. It is not affiliated with, or a work
product of, his employer.

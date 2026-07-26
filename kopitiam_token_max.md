# kopitiam — Token-Efficiency Work Order

**Repo:** `C:\Users\fifad\Documents\kopitiam` (Rust workspace, local-only, not OneDrive-synced)
**Audience:** implementation agents working in parallel. You have no prior conversation context; this document is your full brief.
**Goal:** reduce the number of tokens an AI agent must spend to do useful work — reading documents, understanding this codebase, and translating between languages.

## Contents

- [§0 Design principles](#0-design-principles)
- [§1 Confidence levels](#1-confidence-levels-read-this-before-planning)
- [§2 Shared contract](#2-shared-contract--every-agent-must-read-this)
- [**Part I — `pdf2md` output efficiency**](#part-i--pdf2md-output-efficiency) *(measured; highest confidence)*
- [**Part II — Agentic coding**](#part-ii--agentic-coding) *(design proposal)*
- [**Part III — Translation**](#part-iii--translation) *(design proposal)*
- [§12 Verification](#12-verification)

---

## 0. Design principles

Every proposal below follows from one of these. When a task card leaves a decision open, resolve it with these:

1. **Coordinates over content.** Return `file:line` and signatures, not code bodies. An agent that receives 12 call sites as coordinates spends ~40 tokens; one that greps and reads three files spends thousands.
2. **Structure over prose.** Machine consumers get JSON. Prose reports force an agent to read *and* interpret, and it cannot gate on prose programmatically.
3. **Deterministic tool over model call.** If a compiler, LSP, or parser can answer a question exactly, never spend model tokens guessing at it. `rust-analyzer` knows where a symbol is used; an agent grepping for it is paying tokens to approximate a known-correct answer.
4. **Pay once, then cache.** Understanding derived in one session should persist. Invalidate on content hash, never on time — stale context is worse than absent context because the agent trusts it.
5. **Local model absorbs volume; cloud model spends judgment.** kopitiam already ships local weights (`models pull`). Every high-volume low-judgment task moved to the local model costs zero cloud tokens.
6. **Removal beats compression.** Deleting noise is strictly better than summarizing it. A dropped running head costs nothing; a summarized one still costs tokens and adds a fidelity question.
7. **Make cost visible.** An agent cannot economize on what it cannot measure. Report token estimates so read-vs-outline is an informed choice.

---

## 1. Confidence levels — read this before planning

| Part | Basis | Confidence |
|---|---|---|
| **Part I** (`pdf2md`) | Direct code survey of the pipeline + measured token waste from real conversions | **High.** Line numbers, root causes, and waste percentages are verified. |
| **Part II** (agentic coding) | Command surface from `KOPITIAM_USER_GUIDE.md`; the *internals* of `scan` / `rename` / `code-actions` / `plan` were **not surveyed** | **Medium.** Directions are sound; specifics need a survey first — see Task 0-II. |
| **Part III** (translation) | Repo structure (vendored Python reference, unwired MuPDF port) + general design | **Medium-low.** The porting-ledger and differential-harness ideas are strongly indicated by repo state; document-translation items are greenfield. |

Do not treat Part II or III line-level specifics as verified. Each has a survey task that must run first.

---

## 2. Shared contract — every agent must read this

### 2.1 The recovery-ratio trap (Part I; most important)

`crates/kopitiam-document/src/validation/mod.rs::strip_rendered_markdown_syntax` (~line 148) is a hand-rolled normalizer that knows *exactly* the renderer's current vocabulary: code fences, `FIGURE_PLACEHOLDER`, table separator rows, `#` headings, `> `, `- `, `N. `, `| a | b |`.

`validate()` compares non-whitespace character counts of extracted text vs. rendered Markdown. `MIN_RECOVERY_RATIO = 0.98` (`validation/report.rs:21`).

**If you emit characters that were not in the extracted text — YAML front matter, `<!-- page N -->` anchors, a TOC — you inflate `rendered_chars`, push `recovery_ratio` above 1.0, and silently mask genuine content loss.**

For any change altering emitted output you MUST:
1. Add a matching skip rule to `strip_rendered_markdown_syntax`.
2. Add a test asserting `ratio <= 1.0 + 1e-9`, following `table_pipe_syntax_does_not_inflate_recovery` (`validation/mod.rs:343`).
3. Confirm `strip_rendered_markdown_syntax_removes_all_known_scaffolding` (`validation/mod.rs:399`) still passes.

Conversely, if you *remove* content (running heads, figure labels), `rendered_chars` legitimately drops while `extracted_chars` stays constant, so the ratio falls. **Removal work must discount the removed text from the extracted side too**, or every document starts failing the 0.98 threshold. State in your PR how you kept the ratio honest.

### 2.2 Duplicated pipeline

The `pdf2md` pipeline exists **twice**. Any pipeline-shape change must land in both:
- `apps/cli/src/main.rs::pdf2md` (~lines 195–228)
- `apps/cli/src/tui/convert.rs::convert_one` (~lines 57–93)

Both end in a plain `std::fs::write` (`main.rs:221`, `convert.rs:84`).

### 2.3 Duplicated constant

`FIGURE_PLACEHOLDER` (`"[Figure omitted from Markdown output]"`) is defined in **both** `crates/kopitiam-markdown/src/renderer.rs` (~102) and `crates/kopitiam-document/src/validation/mod.rs:16`.

### 2.4 Hard rule: no copyrighted fixtures

**Never add a third-party PDF, or extracted Markdown from one, into this repo as a test fixture.** The evidence in Part I comes from copyrighted papers held locally outside the repo. Build **synthetic fixtures** reproducing the *structural archetype* (a two-column page with a running head; a diagram with scattered short labels; a ragged table). Do not overfit tests to one paper's quirks.

### 2.5 Existing brittle tests

There are **no golden-file tests** for `pdf2md`. Brittleness is exact-string assertions inline in:
- `crates/kopitiam-markdown/src/renderer.rs` tests (~131–199). `document_blocks_are_joined_by_a_blank_line_with_trailing_newline` asserts `render_document(&doc) == "# Title\n\nBody text.\n"` — front matter or a page anchor breaks it immediately.
- `crates/kopitiam-document/src/validation/mod.rs` tests (~274–440), which hard-code rendered strings.

Update them deliberately; never weaken an assertion to make it pass.

### 2.6 Global scope limits

- Do **not** wire up `crates/kopitiam-pdf/src/mupdf/*` as part of Part I. It is a large, faithful, currently-unused MuPDF port; `stext_boxer.rs` / `stext_classify.rs` are the likely long-term substrate for figure-region detection, but adopting them is Part III work.
- Do not add image extraction. The extraction layer produces text spans only.
- Do not restructure crates or change public APIs beyond what a task requires.
- Anything that changes default output for existing users needs a flag or a documented rationale.

---

# Part I — `pdf2md` output efficiency

*Confidence: high. Root causes verified in code; waste measured on real conversions.*

## 3. The problem, measured

Output is consumed by agents that `grep` the Markdown then read the matching slice. That pattern is currently undermined four ways:

1. A raw NUL byte makes `ripgrep` classify the file as binary and refuse to print matches — silently disabling grep-then-slice entirely.
2. Figure diagrams whose labels are real PDF text emit one short paragraph per label — hundreds of near-zero-information lines.
3. Running heads, footers, and bare page numbers are emitted verbatim on every page.
4. Tables frequently degrade into one cell per line, ~6x the tokens of a real table, structure destroyed.

**Measured case.** A 59 MB proceedings volume → 1.8 MB `.md`, 22,910 lines. Reading one chapter (543 lines) cost:

| Content | Approx. lines | Verdict |
|---|---|---|
| Fig. 1 internal box labels | 153 | near-zero information beyond caption |
| Fig. 2 internal box labels | 87 | near-zero information beyond caption |
| One table as one-cell-per-line | 97 | real information, ~6x too expensive |
| Running heads / bare page numbers | ~20 | pure noise |
| Actual prose | ~185 | the payload |

**~44% of tokens spent reading that chapter were figure-internal label fragments.** A grep for a phrase appearing in the running head returned 17 hits, mostly that running head repeated on odd pages. A NUL byte at ~offset 53237 forced a fallback to `grep -a`, wasting several tool calls.

## 4. Dispatch

Four waves. Tasks within a wave own disjoint files and run in parallel. Waves land in order because four files are contended:

| Contended file | Touched by |
|---|---|
| `crates/kopitiam-document/src/reconstruction/mod.rs` | I-C, I-D |
| `crates/kopitiam-markdown/src/renderer.rs` | I-D, I-B |
| `crates/kopitiam-document/src/validation/mod.rs` | I-D, I-B |
| `apps/cli/src/main.rs` | I-F, I-G, I-H |

All line numbers were accurate at survey time — **re-verify before editing**.

## 5. Wave 0 — Baseline harness (single agent, do first)

**Task I-0.** Create a repeatable measurement so every later claim is provable.

Add a dev script (`scripts/`) converting a fixed corpus and reporting per file: output bytes, line count, non-whitespace char count, recovery ratio, PASS/FAIL, count of lines matching `^\W*\d{1,4}\W*$` (bare page numbers), and count of paragraphs under 5 words. Emit JSON or TSV so before/after diffs are mechanical.

Corpus: synthetic fixtures you create, plus local PDFs referenced **by path only, never copied into the repo** (§2.4). Degrade gracefully when those paths are absent.

**Acceptance:** two runs on an unchanged tree produce byte-identical output.

## 6. Wave 1 — Isolated fixes (parallel)

### Task I-A — Strip control characters at the extraction boundary
**Owns:** `crates/kopitiam-pdf/src/extractor.rs`, optionally new `crates/kopitiam-pdf/src/textnorm.rs`

**Problem.** `pdf-extract`'s `decode_char` can return U+0000 when a `/ToUnicode` CMap maps a code to NUL. `WordBuilder::push` (~`extractor.rs:290`) does `self.text.push_str(ch)` verbatim. No sanitization exists anywhere (grep confirms no hits for `is_control`, `sanitiz`, `\u{0}`). The NUL reaches `fs::write`, and ripgrep then treats the file as binary.

**Why fix at extraction, not rendering.** A NUL counts as non-whitespace content on *both* sides of the recovery ratio, so the validator currently reports `Status: PASS` on an unsearchable file — it is structurally blind to this bug class. Stripping at extraction removes the char before `extracted_content_chars` counts it, keeping the ratio honest. Stripping in the renderer would depress the ratio instead.

**Change.** Port `vendor/pdf-to-markdown/pdf2md/textnorm.py::normalize_char`: drop Unicode category `C` except tab/newline, expand ligatures (`ﬁ`→`fi`), collapse NBSP and exotic spaces, handle BOM. Apply in `WordBuilder::push`.

**Acceptance.** Unit test with `\u{0}`, `\u{1}`, NBSP, `ﬁ` → clean output. No `0x00` and no C0/C1 control char besides `\n`/`\t` in any fixture output. `rg` (not `grep -a`) searches every fixture output. Recovery ratio unchanged or improved.

### Task I-E — Table detection truncates instead of bailing
**Owns:** `crates/kopitiam-document/src/reconstruction/tables.rs`

**Problem.** `try_table` requires **every** line in a candidate run to have exactly `lines[0].cells.len()` cells, each starting within `COLUMN_X_TOLERANCE = 8.0` of row 0. One ragged row — merged cell, wrapped cell, subtotal row — returns `None` for the entire run. Everything falls through to `consume_paragraph`, producing one `Paragraph` per cell joined by `\n\n`. That is the 97-line table in §3.

**Change.** When a run breaks uniformity, emit the longest valid prefix as a `Table` (subject to `MIN_TABLE_ROWS`/`MIN_TABLE_COLUMNS`) and let the remainder continue through normal block handling. Optionally tolerate a trailing ragged row by padding to header width — but padding adds `|` scaffolding, so re-read §2.1, and never invent cell content.

**Acceptance.** Synthetic 5-row table with a ragged row 4 → rows 1–3 render as a table. `table_escapes_pipes_and_has_no_column_padding` still passes. Ratio stays `<= 1.0`. Report the line-count reduction.

## 7. Wave 2 — Region classification (serialize C then D; same owner recommended)

Both touch `reconstruction/mod.rs`, and D reuses C's zone machinery.

### Task I-C — Strip running heads, footers, bare page numbers
**Owns:** new `crates/kopitiam-document/src/reconstruction/headers.rs` + wiring in `reconstruction/mod.rs`

**Problem.** Entirely absent. Nothing reads `Page::height` — populated at `extractor.rs:198`, never read in reconstruction. A bare page number becomes its own `Paragraph` because `consume_paragraph` breaks on a vertical gap > `1.8 × font_size`.

**Change.** Port `vendor/pdf-to-markdown/pdf2md/headers.py`: top/bottom 10% zones by `Page::height`; digit-normalized signature per candidate line (`\d+` → `#`); count signatures across pages, drop those recurring above a threshold; always drop `^\W*\d{1,4}\W*$` within a zone.

Be conservative on short documents — with 2 pages a legitimate heading can look recurring. Require a minimum page count before signature stripping engages; always allow the bare-number rule.

**Acceptance.** Synthetic 6-page fixture with running head + alternating footer numbers → zero remain. A 1-page fixture unchanged (no false positives). Ratio honesty per §2.1 — state your approach. Bare-page-number count drops to ~0 on the corpus; report the token delta.

### Task I-D — Collapse figure regions to captions
**Owns:** `reconstruction/figures.rs`, `reconstruction/mod.rs`, the placeholder in `renderer.rs` + `validation/mod.rs:16`
**Highest payoff and hardest task. ~44% of tokens on a diagram-heavy chapter.**

**Problem.** `try_figure` matches only a line beginning `Figure N` / `Fig. N` (one case-insensitive regex). Architecture diagrams are vector art whose labels are *real PDF text*, so each label becomes its own `Paragraph` — 240 lines for two figures in §3. The renderer additionally emits `{caption}\n\n[Figure omitted from Markdown output]`, a 37-char placeholder plus blank line per figure, carrying no information.

**Change.** Add heuristic figure-region detection ahead of `consume_paragraph`. Combine signals; rely on none alone:
- A run of consecutive short lines (roughly < 5 words) without terminal punctuation.
- High x-scatter / inconsistent left edges versus body-text columns.
- Spatial adjacency to a matched `Fig. N` caption.
- Absence of sentence-like structure across the run.

On detection emit the caption only. Replace the placeholder with nothing, or at most a single-token marker.

**Precision over recall.** Wrongly swallowing prose is far worse than leaving soup behind. Gate aggressive collapsing behind a flag if confidence is low; default to the safer behaviour.

**Acceptance.** Fixture with ~30 scattered labels + `Fig. 1` caption → caption alone. Negative fixture: a short-line-heavy list of real prose is NOT collapsed. `figure_without_caption_still_renders_placeholder` updated with rationale. `FIGURE_PLACEHOLDER` consistent across both definitions. Ratio honesty per §2.1. Line-count reduction > 50% on a diagram-heavy fixture.

## 8. Wave 3 — Page anchors

### Task I-B — Emit page-boundary anchors
**Owns:** `crates/kopitiam-markdown/src/renderer.rs`, `crates/kopitiam-document/src/validation/{mod.rs,report.rs}`

**Problem.** `Document.block_pages` (per-block 1-based start page) and `Document::blocks_with_pages()` (`document.rs:69`) are already computed and **completely ignored by the renderer** — `render_document` (`renderer.rs:13`) just joins blocks with `\n\n`. `Document.title`, `metadata.source_pages`, and `citations` are likewise computed and discarded.

**Why it matters.** An agent can jump straight to a cited page (constant need for citation checking), and it unblocks **per-page recovery ratios** — impossible today because `validate()` sees the rendered side as one undifferentiated string. Per-page ratios are themselves a token saver: instead of "97% overall, re-check everything against page images," it becomes "page 4 is 71%, check only page 4."

**Change.**
1. Emit a stable anchor at each page boundary. HTML comment form (`<!-- page 717 -->`) is invisible when rendered and unambiguous to grep.
2. Add the skip rule to `strip_rendered_markdown_syntax` — this task cannot land without it (§2.1).
3. Extend `ConversionReport` with per-page recovery, now computable by splitting the rendered side on anchors.
4. `render_document` takes no config today. Prefer introducing an options struct, anchors defaulting **on** for agent use, and document the choice.

**Acceptance.** Multi-page fixture: one anchor per boundary, numbers matching `block_pages`. Ratio test for anchor scaffolding. `document_blocks_are_joined_by_a_blank_line_with_trailing_newline` updated deliberately. Per-page ratios in the report, consistent with the document-wide figure. `rg -n "<!-- page 717 -->"` locates a page.

## 9. Wave 4 — CLI surface (one agent; all touch `main.rs`)

Mirror pipeline-shape changes into `tui/convert.rs` (§2.2). Current complete surface is `kopitiam pdf2md <INPUT> [-o|--output <OUTPUT>]` — nothing else.

**Task I-F — `--report-json`.** `ConversionReport` is `Display`-only (`validation/report.rs:108-128`) and `kopitiam-document` has no serde dependency. Add derives (behind a feature if you want the dep optional) and a flag, so callers gate on recovery ratio without parsing prose. Include per-page ratios if I-B landed.

**Task I-G — Sidecar index.** *The single biggest access-pattern win.* Emit `<output>.index.json` mapping heading text → line range and page number → line range. This converts grep-and-probe into lookup-then-slice; locating one chapter in a 22,910-line output cost several exploratory calls that an index makes a single lookup. Keep it a **sidecar**, not in-band, so the `.md` char count and recovery ratio are untouched.

**Task I-H — `--pages A-B` and `--split-by heading-level N`.** `--pages` avoids converting 59 MB to read 13 pages. `--split-by` turns a 1.8 MB multi-chapter output — ~450k tokens if read carelessly — into individually safe per-chapter files. Reuse the TUI's naming helpers (`tui/logic.rs::mirror_output_path`, `default_batch_output`, `derive_md_name`).

**Acceptance (all).** `--help` documents every flag; each has a test; the existing two-flag invocation behaves identically.

---

# Part II — Agentic coding

*Confidence: medium. Directions sound; internals unsurveyed.*

An agent's token spend in a codebase is dominated by **reading files to build understanding, and re-deriving that understanding every session.** kopitiam already has the right substrate — `scan` (cargo metadata, rust-analyzer, rustdoc), `rename`, `code-actions`, `status` (`.kopitiam/state.redb` session memory), `plan`, and local model weights. The opportunity is exposing that substrate as cheap, structured queries so an agent stops paying a model to approximate answers a tool already knows exactly (§0.3).

## 10. Task II-0 — Survey first (blocking, single agent)

The internals of `scan`, `rename`, `code-actions`, `status`, `plan`, and the crates backing them were **not surveyed**. Before any Part II task, produce a written survey covering: how `scan` invokes and caches rust-analyzer/rustdoc; the schema of `.kopitiam/state.redb` and what `ProjectState` persists; whether an LSP client abstraction already exists and is reusable for queries; how `code-actions` addresses file positions; and what `plan` currently feeds the local model.

Report findings **into this document** as a new subsection, then dispatch the tasks below with corrected specifics. Several may already be partly built.

## 10.1 — II-0 survey findings (verified)

*Every claim below is `file:line` in the tree at survey time (2026-07-26). Re-verify before editing. Coordinates, not prose (§0.1).*

### The 8 survey questions, answered

**1. How `scan` invokes/caches rust-analyzer + rustdoc.** `scan::run` (`apps/cli/src/scan.rs:65-82`) runs three `KnowledgeProvider`s and **discards the graph after printing** — it persists nothing but `state.touch("scan")` (`scan.rs:77-79`). rust-analyzer is **off by default** (`--with-rust-analyzer`, `scan.rs:56-57,71-73`). Invocation per provider:
- *cargo*: the `cargo_metadata` crate, in-process (`providers/cargo.rs:37-40`).
- *rustdoc*: shells out `cargo +nightly rustdoc … --output-format json`, reads the JSON file (`providers/rustdoc.rs:101-130`); nightly-only, degrades to empty on stable (this workspace pins stable, so it contributes nothing).
- *rust-analyzer*: spawns a **fresh** `LspClient` JSON-RPC session over stdio, waits up to **180 s** for indexing, fires **one** `workspace/symbol ""` query, shuts down (`providers/rust_analyzer.rs:56-59`). **One-shot, no cache.** Every rust-analyzer use anywhere pays the full spawn+index cost again.

**2. `.kopitiam/state.redb` schema.** One redb table, `TableDefinition<&str, &[u8]>("kv")` — a generic bytes KV store (`crates/kopitiam-index/src/store.rs:8,36-97`). Typed access is `put_json`/`get_json` = **serde_json**, not bincode (`store.rs:85-96`). `ProjectState` (`crates/kopitiam-workspace/src/state.rs:19-32`) is the *only* thing the CLI persists today: `{ current_task: Option<String>, working_set: Vec<String> (capped 50, WORKING_SET_CAPACITY), updated_at: Option<u64> }`, serde-derived, stored under the single key `"workspace/project_state"` (`state.rs:8`). **No content hashes anywhere in state.** Other crates namespace their own keys in the same db (`kopitiam-web/src/cache.rs:77`). **Consequence for II-3/II-5: just add new keys — no schema/migration, the KV table already takes arbitrary serde_json blobs.**

**3. Reusable LSP client.** Yes — and richer than the doc assumed. `crates/kopitiam-lsp` is a **stub** (`lib.rs` is `fn add`); the real client is in **`kopitiam-semantic`**, two layers:
- `LspClient` (low-level JSON-RPC/stdio, `lsp_client.rs:206`): `definition:778`, `references:794`, `hover:811`, `completion:829`, `rename:681`, `code_actions:701`, `execute_command:734`, `workspace_symbols:582`, **`document_symbols:650` (already implemented, `#[allow(dead_code)]`, no in-crate caller)**, `diagnostics_for:864`. Negotiates position encoding (`initialize`, `lsp_client.rs:428-497`).
- `RustAnalyzerSession` (path + `char`-offset API, `session.rs:33`): `definition:172`, `references:181`, `hover:190`, `completion:203`, `rename:105`, `code_actions:117`, `diagnostics:238`. Public via `lib.rs:18-25`.
**`refs`/`def`/`sig` (II-1) are a thin CLI wrapper over `references`/`definition`/`hover` — already built.** `callers`/`callees`/`impls` are *not* exposed (need `callHierarchy` / `textDocument/implementation`, new methods on `LspClient`+`session.rs`).

**4. `code-actions` coordinate convention.** **0-indexed line, 0-indexed `character` in Unicode scalar values (`char` offset)** — `code_actions.rs:24-30` (CLI args), `session.rs:117-122` converts to the negotiated wire encoding via `position::char_col_to_unit`. Request = `textDocument/codeAction` (`lsp_client.rs:701-726`); response normalized to `Vec<CodeAction{ title, raw:Value }>` (`session.rs:26-29,123-129`). **This is the convention II-1/II-2 must emit: `file:line:char`, both 0-based, char (not byte, not UTF-16) columns.**

**5. `rename` preview/apply template.** `rename::run` (`apps/cli/src/rename.rs:48-76`): compute `Vec<FileEdit>` (pure, no write — `session.rs:105-112`), then **without `--apply` print a unified diff** (`rename.rs:71`, `edit::diff` via the `similar` crate, `edit.rs:83-92`); **with `--apply` call `edit::write_file_edits`** (`rename.rs:65`, `edit.rs:62-67`). `FileEdit { path, original, updated }` (`edit.rs:20-24`); `edit` is a public module (`lib.rs:10`). `code-actions` uses the same edit layer but writes immediately (picking a named action is the deliberate step, `code_actions.rs:6-10,81-92`). **II-8 must reuse `edit::{FileEdit, diff, write_file_edits}` verbatim and mirror rename's `--apply` gate.**

**6. `plan` + `ai chat` local model.** Both go through `crate::adapter::select_adapter` (`apps/cli/src/adapter.rs:237-295`): `KOPITIAM_MODEL_GGUF` env → default catalog model in the store → **`EchoAdapter` fallback** (deterministic stub) when no `.gguf`. **Doc's model guesses confirmed:** default `qwen2.5-0.5b-instruct-q4_0` (`adapter.rs:67`, `catalog.rs:133`); `llama-3.2-1b-instruct-q4_0` also in the catalog (`catalog.rs:154`). `LocalAdapter::load` parses the GGUF and builds the tokenizer via `kopitiam_runtime::tokenizer_from_gguf` (`kopitiam-ai/src/local/adapter.rs:118`). `plan` feeds the model a `SemanticGraph` (cargo+rustdoc only — no rust-analyzer) plus `ProjectState`, assembled by `kopitiam-workflow`'s ContextBuilder inside `run_workflow` (`plan.rs:48-68`). `ai chat` calls `adapter.stream()` **directly, no knowledge assembly** (`ai.rs:210`). **II-6 entry point = `select_adapter()`; `ModelAdapter::{stream,complete}` is the interface; degrades to Echo with no weights (must be handled honestly).**

**7. Token counting.** A real from-scratch byte-level BPE exists: **`kopitiam-tokenizer`** (`Tokenizer::encode -> Vec<u32>`, `lib.rs:71-91`; `BpeTokenizer` in `bpe.rs`). But it needs a vocab loaded — either `loader::from_tokenizer_json` or `kopitiam_runtime::tokenizer_from_gguf(&LoadedModel)` (`kopitiam-runtime/src/gguf_tokenizer.rs:67`). **No standalone `tokenizer.json` is bundled**, and **`apps/cli` depends on neither `kopitiam-tokenizer` nor `kopitiam-runtime`** (`apps/cli/Cargo.toml` deps list). So II-7 either reuses a present GGUF's vocab (via the model store) or ships/embeds a vocab; a heuristic byte/word approximation avoids the dep entirely.

**8. CLI registration pattern.** `apps/cli/src/main.rs`, clap derive, three edit points per subcommand: (a) `mod <name>;` at `main.rs:24-32`; (b) a `Command` enum variant wrapping the module's `Args` struct, e.g. `Scan(scan::ScanArgs)` at `main.rs:98` — enum spans `main.rs:64-162`; (c) a match arm in `main()` at `main.rs:171-205` (e.g. `main.rs:176-179`). Each subcommand is its own file `apps/cli/src/<name>.rs` exposing `Args` + `pub fn run(args) -> Result<()>`. **main.rs is the one contended file (same shape as Part I Wave 4).**

### Corrected specifics per card

- **II-1 (semantic queries).** *Corrected:* **already ~60% built.** `refs`/`def`/`sig` = thin CLI over `RustAnalyzerSession::{references,definition,hover}` (`session.rs:172-193`), returning `file:line:char` (0-based char cols, per finding 4). **Must build:** `callers`/`callees --depth`/`impls` (new `LspClient` methods for `callHierarchy/*` and `textDocument/implementation`, then `session.rs` wrappers) + the CLI file + `--json`. Owner crate: `kopitiam-semantic` (new methods) + new `apps/cli/src/*.rs`. No new dep. **Spec caveat the card omits:** each invocation spawns rust-analyzer and waits up to 180 s to index — ad-hoc queries are *not* cheap unless a persistent session is added (no daemon exists today; `LspClient` is per-process). Consider one `refs`-style command that batches queries in a single session.

- **II-2 (outline).** *Corrected:* **half built.** `LspClient::document_symbols` (`lsp_client.rs:650`) already returns the hierarchical `DocumentSymbol` tree and is currently dead code. **Must build:** expose it through `RustAnalyzerSession` (new method, `session.rs`) + a CLI `outline` file. Alternative deterministic path with no rust-analyzer spawn: parse with the in-tree `kopitiam-syntax` crate (verify its API) — cheaper, no 180 s index. Owner: `kopitiam-semantic` + new CLI file. No new dep. **Contends `session.rs` with II-1.**

- **II-3 (architecture digest).** *Corrected:* the graph `scan` builds is **thrown away** (`scan.rs:75-82`) — nothing is cached today. **Must build:** serialize a compact digest (crate→responsibility→key types→dep edges; cargo `DependsOn` edges already exist, `cargo.rs:67-79`) into `state.redb` under a **new key** via `Store::put_json` (`store.rs:85`). Needs a content-hash for invalidation — **no hashing exists in state** (only `kopitiam-models` does sha256); pull in a hash (sha2 is already in the lock via models, or blake3). Owner: `kopitiam-workspace`/new module + wiring in `scan.rs`. **Contends `scan.rs`.**

- **II-4 (compact diagnostics).** *Corrected:* greenfield, but **cleanly disjoint** — it parses `cargo build`/`test` stdout, touches none of the LSP/state code. Owner: new CLI file(s) + a small parser module (could live in a new or existing crate). Touches only `main.rs` for registration. No new dep (parse text). Note: `RustAnalyzerSession::diagnostics` (`session.rs:238`) is an *alternative* source but is push-based and slow to warm; cargo-output parsing is the pragmatic path the card intends.

- **II-5 (conclusion memory).** *Corrected:* the persistence substrate is **fully built** (`Store` + `state.redb`, generic KV). **Must build:** a hash-keyed `Conclusion` record type (new key(s) in the same db) storing the source hashes it was derived from, plus a `status --stale` view. Owner: `kopitiam-workspace/src/state.rs` (new struct, sibling to `ProjectState`) + `apps/cli/src/status.rs` (`--stale`). Same hashing dep as II-3. **Contends `state.rs` + `status.rs`.** Reuse §0.4 invalidation — no time-based expiry.

- **II-6 (local preprocessing).** *Corrected:* the adapter selection + `ModelAdapter::stream` path is **built and reusable** (`adapter.rs:237`, `ai.rs:210`). **Must build:** task-specific prompts/framing (summarize, triage, classify) as a new CLI surface, calling `select_adapter().adapter()`. Owner: new `apps/cli/src/*.rs`. No new dep. **Spec caveat:** with no `.gguf` this silently falls back to `EchoAdapter` (echoes input) — the command must detect `!is_local()` (`adapter.rs:137`) and refuse/annotate rather than pretend it preprocessed.

- **II-7 (token accounting).** *Corrected:* tokenizer exists (`kopitiam-tokenizer`) but is **not a CLI dep** and has **no bundled vocab** (finding 7). **Must build:** the `tokens` CLI file + decide vocab source (present-GGUF via `kopitiam-runtime::tokenizer_from_gguf`, or a heuristic). **Needs new deps in `apps/cli/Cargo.toml`** (`kopitiam-tokenizer`, maybe `kopitiam-runtime`). **This is the only card that must touch `Cargo.toml`/`Cargo.lock`.**

- **II-8 (deterministic refactors).** *Corrected:* the template is **fully built and documented above** (finding 5): `edit::{FileEdit, diff, write_file_edits}` (`edit.rs`) + rename's `--apply` gate. Each refactor = compute `Vec<FileEdit>` → `diff` preview → `--apply` writes. Owner: new CLI file(s); logic that needs semantic info reuses `RustAnalyzerSession`/`code_actions`. May add methods to `kopitiam-semantic`. No new dep. Follows `rename.rs` exactly.

### Dispatch recommendation

**Shared/serializing files:** `apps/cli/src/main.rs` (every card that adds a subcommand — registration only, small, mergeable if coordinated or done by one integrator); `apps/cli/Cargo.toml`+`Cargo.lock` (**II-7 only**); `crates/kopitiam-semantic/src/session.rs` (**II-1 and II-2** both add methods — same owner or sequence II-1→II-2); `apps/cli/src/scan.rs` (II-3); `apps/cli/src/{status.rs,../workspace state.rs}` (II-5).

**Safe to build in parallel (disjoint backing files):** **II-4** (cargo-output parser), **II-6** (adapter reuse), **II-3** (scan.rs + workspace), **II-5** (state.rs + status.rs), **II-7** (own file + Cargo.toml) — these own disjoint files; only the `main.rs` registration lines overlap.

**Must serialize:** **II-1 then II-2** (shared `session.rs`). II-7 serializes on `Cargo.lock` against any other card that adds a dep (none of the others need one, so II-7 is effectively alone there).

**Already partly built (per §10's prediction, confirmed):** II-1 (references/definition/hover done), II-2 (`document_symbols` implemented, unexposed), II-5 (persistence done), II-3 (graph produced, just discarded), II-8 (edit/preview/apply template complete). Only II-4, II-6, II-7 are meaningfully greenfield.

## 11. Task cards

### Task II-1 — Semantic queries (highest value)
Expose rust-analyzer's knowledge as coordinate-returning commands: `refs <symbol>` (call sites as `file:line`, no context), `def <symbol>` (location + signature), `sig <symbol>` (signature without body), `callers`/`callees <fn> --depth N`, `impls <trait>`.

An agent asking "who calls `try_table`?" currently greps, gets hits with context, then reads files. These return ~40 tokens of coordinates instead of thousands. `--json` mandatory.

### Task II-2 — Outline / skeleton mode
`outline <file>` → items only (fn signatures, struct fields, impl blocks) with line numbers, **no bodies**. `reconstruction/mod.rs` is 811 lines; its outline is perhaps 60. Orientation in an unfamiliar file is currently a full read; this is roughly a 10x reduction on that step, and combined with II-1 the agent then reads only the one function it needs.

### Task II-3 — Cached architecture digest
`scan` already learns cargo metadata. Emit a compact persistent digest: crate → responsibility → key types → dependency edges. Generated once, read cheaply every session, regenerated on `Cargo.toml`/source hash change. This is exactly the "what does each crate do" summary that currently costs a full exploration pass per session. Cache in the existing `.kopitiam/state.redb`.

### Task II-4 — Compact diagnostics
`cargo build`/`cargo test` output is enormously verbose — backtraces, repeated notes, "for more information about this error", full type dumps — and agents run it constantly. Add `check --compact` / `test --compact`: one line per **distinct** diagnostic, deduplicated, sorted by file, noise stripped; test failures as name + assertion + `file:line`, not full stdout. Deduplication is the main win: one bad type can produce 40 diagnostics that are a single fix.

### Task II-5 — Persistent conclusion memory
Generalize `.kopitiam/state.redb` to record *conclusions* — this crate does X, this invariant holds, this test is flaky — so the next session doesn't re-derive them. This is literally paying once for understanding.

**Invalidate on content hash, never on time** (§0.4). Every entry stores the hashes it was derived from and is dropped when they change. Stale memory an agent trusts is worse than no memory. Include a `status --stale` view.

### Task II-6 — Local-model preprocessing
`ai chat` / `plan` already run local weights (`qwen2.5-0.5b`, `llama-3.2-1b`). Route high-volume low-judgment work there so the cloud model never sees it: summarize a 2000-line file to 50 lines, triage which of 200 grep hits are plausibly relevant, draft commit messages, classify diagnostics. Zero cloud tokens.

Be honest about capability: a 0.5B model cannot be trusted with judgment. Use it for filtering and compression where a false negative is recoverable, never as the final authority on correctness. Always report what it dropped.

### Task II-7 — Token accounting
`tokens <path>` estimating token count for a file or directory, so an agent chooses read-vs-outline informed rather than blind (§0.7). A good BPE approximation suffices. Also gives Part I a second measurement axis.

### Task II-8 — Deterministic refactors
`rename` already proves the pattern: a mechanical, verifiable transformation replaces an agent reading and editing many files. Each one added moves work from "LLM edits 30 files" to "one command." Candidates: move item between modules with import fixup, extract function, add a derive across matching types, apply one clippy fix class repo-wide. Preview-diff-then-`--apply`, matching `rename`'s existing ergonomics.

---

# Part III — Translation

*Confidence: medium-low. Porting items strongly indicated by repo state; document-translation items greenfield.*

Two distinct senses, both live in this project.

## 12. Code translation (porting)

This repo is **already two translation projects**: `crates/kopitiam-pdf/src/mupdf/*` is a faithful MuPDF C→Rust port (currently unwired into `extract()`), and `crates/kopitiam-document/vendor/pdf-to-markdown/` is a vendored MIT Python reference explicitly serving as the porting source, with `headers.py`, `textnorm.py`, `postprocess.py`, `serialize.py`, `tables.py`, `regions.py` unported.

Porting is token-expensive in a specific way: an agent reads the source implementation, reads the target, reasons about equivalence, writes, then verifies by reasoning again. Two capabilities collapse most of that.

### Task III-1 — Port ledger (highest value here)
A machine-maintained ledger: source file/symbol → target file/symbol → status (`unported` / `in-progress` / `ported` / `deliberately-diverged`) → divergence notes.

Today this knowledge is implicit and gets re-derived every session — that `mupdf/` is unwired, that `headers.py` and `textnorm.py` are unported, is exactly the kind of fact an exploration pass currently rediscovers from scratch. On a long port that rediscovery is the dominant recurring cost.

Add `port status` showing the ledger, plus **upstream drift**: source-side changes since a symbol was ported. Keeps translation honest without reading both trees.

### Task III-2 — Differential equivalence harness
For translation, correctness *is* identical output. A runner feeding the same input to the reference implementation and the Rust port, then diffing, replaces "agent reads both implementations and reasons carefully about equivalence" — expensive and error-prone — with "run the harness," which is free and exact (§0.3).

`vendor/pdf-to-markdown/tests/fixtures/*.pdf` and `vendor/pdfplumber/tests/pdfs/` already exist as candidate inputs. Note §2.4: those are vendored with the reference implementation, so check their licences before relying on them, and prefer synthetic inputs where behaviour allows.

This is the single biggest translation token saver, and it compounds: every ported symbol gets cheap regression coverage.

### Task III-3 — Skeleton-first translation
Mechanically generate Rust signatures/stubs from the source's structure so the agent writes bodies only. Reduces generated tokens and keeps naming consistent across a long port — consistency an agent otherwise maintains by re-reading its own earlier output.

## 13. Document translation

Natural-language translation of converted documents (the driving case: Chinese-language nuclear literature). `pdf2md` plus local model weights make this natural, and the token economics are dominated by re-translation waste.

### Task III-4 — Segment IDs and translation memory
Emit stable segment identifiers in converted Markdown (or the §9 sidecar index), then cache translations keyed by segment content hash. When a document is revised, only changed segments are re-translated. This is standard translation-memory practice and the largest single saving on any iteratively-revised document — unchanged segments cost zero.

Depends on Task I-B/I-G for stable anchoring.

### Task III-5 — Terminology glossary enforcement
A project glossary applied deterministically (e.g. 华龙一号 → "Hualong One", 数字孪生 → "digital twin"). Prevents spending model tokens re-deciding terminology per occurrence, and prevents drift across a long document — which otherwise costs a whole review pass to fix.

### Task III-6 — Local-first two-pass translation
Local model drafts every segment; the cloud model reviews only low-confidence segments. Requires a usable confidence signal — segment length, glossary-hit rate, perplexity, or round-trip disagreement. Report the split so the saving is measurable.

Be careful: a 0.5B model's translations of technical prose will need real review. Default conservative, widen the local share only with measured evidence.

### Task III-7 — Bilingual aligned output
Side-by-side or interleaved source/target with segment anchors, so a reviewer — human or model — checks specific segments without reading the whole document. Turns review from a full-document read into a targeted one.

---

## 12. Verification

Per task: `cargo test --workspace` green, `cargo clippy --workspace -- -D warnings` clean, plus the card's acceptance criteria.

After each Part I wave:
1. Re-run the Task I-0 harness; commit the before/after table. **Every token-reduction claim must be backed by a number from this harness.**
2. Confirm `rg` (not `grep -a`) searches every fixture output — no regression to binary classification.
3. Confirm recovery ratio stays within `[0.98, 1.0]` corpus-wide. **A ratio above 1.0 is a failure**, not a pass: emitted scaffolding is masking content loss.
4. Spot-check one converted paper by eye for swallowed prose — Task I-D's failure mode is silent and will not surface as a test failure.

For Parts II and III, each task must state its own measurement. A capability that claims to save tokens but cannot demonstrate it on a concrete before/after is not done.

## 13. Suggested order

**Part I:** `I-0 → (I-A, I-E parallel) → I-C → I-D → I-B → I-F/G/H`

I-A first: smallest diff, fixes a bug the validator is structurally blind to. I-E is independent and cheap. I-C before I-D (shared zone machinery, shared file). I-B after I-D (both touch renderer + stripper). Wave 4 last — additive, and benefits from I-B's per-page data.

If capacity is limited, **I-A and I-D alone capture most of Part I's value**: I-A restores grep as a viable access path at all, I-D removes the largest category of waste.

**Across parts:** Part I is the only fully-verified work and should lead. Then II-0 and III-1 in parallel — both are cheap surveys/ledgers that make everything after them concrete. Then II-1, II-2, II-4 (the three biggest agentic-coding wins) and III-2 (the biggest translation win).

//! `kopitiam`: the KOPITIAM command-line interface.
//!
//! This binary is the thin human-facing shell over KOPITIAM's engines — see
//! `CLAUDE.md`'s Architecture section ("Applications are clients. The
//! platform owns the functionality.") and its "Dogfood the Semantic Runtime
//! CLI" rule. Concretely that means:
//!
//! * every subcommand here should be a short call into a `kopitiam-*` crate,
//!   never a place where new business logic is invented;
//! * as the Semantic Runtime crates (`kopitiam-knowledge`, `kopitiam-index`,
//!   `kopitiam-search`, `kopitiam-workspace`, `kopitiam-workflow`, ...) grow
//!   capable of more, this file grows more subcommands to expose them;
//! * this binary is meant to become the actual tool used to keep developing
//!   KOPITIAM (`resume`, `plan`, `architecture`, `translation-status`, ...),
//!   not a demo that gets left behind once the underlying crates work.
//!
//! Subcommands so far: [`Command::Pdf2md`] turns a PDF into semantic
//! Markdown via the Document Engine. [`Command::Scan`] is the first,
//! read-only slice of the Semantic Runtime (see [`scan`]).
//! [`Command::Rename`] and [`Command::CodeActions`] are the first
//! write-capable slice, driving a live rust-analyzer over LSP to rename
//! symbols and apply refactorings (see [`rename`] and [`code_actions`]).

mod adapter;
mod ai;
mod code_actions;
mod models;
mod plan;
mod rename;
mod scan;
mod status;
mod tui;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};

/// Which text-extraction engine [`Command::Pdf2md`] drives.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Engine {
    /// The ported MuPDF `stext` engine (default): true reading order with
    /// columns linearised and correct inter-word spacing.
    Mupdf,
    /// The legacy `pdf-extract`-based path, kept as a fallback.
    Legacy,
}

/// Top-level CLI parser. `clap` derives the actual argument parsing from
/// this struct and the [`Command`] enum below; this file only wires parsed
/// arguments to the function that does the real work.
#[derive(Parser)]
#[command(name = "kopitiam", about = "KOPITIAM command-line interface")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Every subcommand `kopitiam` currently supports.
///
/// Add new engines here as they mature — each variant should map to exactly
/// one call into a `kopitiam-*` library crate, with this file handling only
/// argument parsing, I/O, and user-facing output.
#[derive(Subcommand)]
enum Command {
    /// Convert a PDF into semantic Markdown.
    ///
    /// Runs the full Document Engine pipeline: `kopitiam-pdf` extracts text
    /// per page, `kopitiam-document` reconstructs paragraph/heading/table
    /// structure across page breaks and columns, and `kopitiam-markdown`
    /// renders the result. A validation report comparing extracted vs.
    /// rendered word counts is printed alongside the output, as a cheap
    /// sanity check that the reconstruction did not silently drop content.
    Pdf2md {
        /// Input PDF file.
        input: PathBuf,
        /// Output Markdown file. Defaults to the input path with a .md extension.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Extraction engine. `mupdf` (default) is the ported MuPDF `stext`
        /// engine: true reading order (columns linearised) and correct
        /// inter-word spacing. `legacy` is the older `pdf-extract`-based path,
        /// kept as a fallback.
        #[arg(long, value_enum, default_value_t = Engine::Mupdf)]
        engine: Engine,

        /// Print the validation report as JSON on stdout instead of the
        /// human-readable prose. The JSON carries the computed `recovery_ratio`,
        /// `passes`, and per-page ratios so a caller can gate on recovery
        /// without parsing prose (card I-F). The "Wrote ..." notices go to
        /// stderr in this mode so stdout is clean JSON.
        #[arg(long)]
        report_json: bool,

        /// Also write a `<output>.index.json` sidecar mapping each heading and
        /// each source page to its 1-based line range in the Markdown, turning
        /// grep-and-probe into lookup-then-slice (card I-G). The `.md` itself is
        /// untouched. With `--split-by-heading-level`, one sidecar per part.
        #[arg(long)]
        index: bool,

        /// Convert only source pages `A-B` (matched against the PDF's own
        /// 1-based page numbers, inclusive; a bare `N` means just page N),
        /// applied before reconstruction so a 59 MB PDF is not fully converted
        /// just to read a few pages (card I-H). The selection is a standalone
        /// document: its page anchors are renumbered from 1 across the kept
        /// pages, not carried over from the source.
        #[arg(long, value_name = "A-B")]
        pages: Option<String>,

        /// Split the output into one `.md` per section at this heading level
        /// (1-6), instead of a single combined file, so a large multi-chapter
        /// document becomes individually safe per-chapter files (card I-H).
        /// Files are named `<stem>.NN-<slug>.md` beside `--output`.
        #[arg(long, value_name = "N")]
        split_by_heading_level: Option<usize>,
    },

    /// Scan a Rust project's real tooling and report what the Semantic
    /// Runtime learned about it.
    ///
    /// This is the first Semantic Runtime command: it runs the
    /// `kopitiam-semantic` knowledge providers (cargo metadata always,
    /// rust-analyzer optionally, rustdoc JSON when a nightly toolchain is
    /// available) against a project, merges everything they report into a
    /// `kopitiam-knowledge` graph, and prints a summary. See
    /// `apps/cli/src/scan.rs` for the full explanation of why this command
    /// exists and where it is headed.
    Scan(scan::ScanArgs),

    /// Rename a symbol using a live rust-analyzer, previewing the change
    /// as a diff unless `--apply` is given.
    ///
    /// See `apps/cli/src/rename.rs` for the full explanation, including why
    /// this is safe-by-default.
    Rename(rename::RenameArgs),

    /// List or apply rust-analyzer's code actions (quick fixes and
    /// refactorings) at a file position.
    ///
    /// See `apps/cli/src/code_actions.rs` for the full explanation.
    CodeActions(code_actions::CodeActionsArgs),

    /// Print this project's persisted session memory (`.kopitiam/state.redb`).
    ///
    /// See `apps/cli/src/status.rs`: this is the read side of the state
    /// `scan` writes, proving persistence survives across process restarts.
    Status(status::StatusArgs),

    /// Run the `plan` workflow: build context from a live scan plus
    /// session memory, and invoke a model adapter.
    ///
    /// The adapter is chosen at runtime by `crate::adapter::select_adapter`:
    /// a real on-CPU `kopitiam_ai::LocalAdapter` when a `.gguf` is present on
    /// disk, otherwise `kopitiam_ai::EchoAdapter` (the deterministic stub)
    /// with a note on how to get a real model. Either way it runs offline.
    ///
    /// See `apps/cli/src/plan.rs`: the first `kopitiam-workflow` command,
    /// proving the full `load state -> collect facts -> build context ->
    /// invoke model -> validate -> persist` pipeline end to end.
    Plan(plan::PlanArgs),

    /// Talk to the AI layer. `ai chat` opens an interactive, streamed chat
    /// with the local model (echo stub when no `.gguf` is present, so it
    /// always runs).
    ///
    /// This is the maintainer's testable AI interface — `temp_ai_design.md`
    /// §10.6 phase 1 (chat over `LocalAdapter`, streamed token-by-token, no
    /// tools). See `apps/cli/src/ai.rs`, whose `chat_loop` is factored over
    /// `Read`/`Write` so the streamed loop is testable headlessly.
    Ai(ai::AiArgs),

    /// Open the KOPITIAM chat TUI: a full-screen, kopitiam-themed terminal
    /// interface over the same streamed local model `ai chat` uses.
    ///
    /// A ratatui front-end onto `crate::adapter::select_adapter` — a real
    /// on-CPU `kopitiam_ai::LocalAdapter` when a `.gguf` is present, otherwise
    /// the deterministic `EchoAdapter`, so it always runs offline. Tokens
    /// stream live into the transcript by polling the adapter's
    /// `Receiver<StreamChunk>`; no business logic lives in the UI. This is the
    /// runnable slice of `temp_ai_design.md`'s "full ratatui" phase.
    Tui(tui::TuiArgs),

    /// Go and get, then check, the local model weights the AI layer runs on.
    ///
    /// Group of four actions — `list`, `pull`, `path`, `verify` — over the
    /// `kopitiam-models` model store. `pull` is the autofetch path (download
    /// plus SHA-256 verify from the catalog); a user who already got the file
    /// can drop it where `path` say and skip the network (bring-your-own).
    /// This keeps `CLAUDE.md`'s Offline-First promise real: no local weights,
    /// no local model. See `apps/cli/src/models.rs` for the full story.
    Models(models::ModelsArgs),
}

// `main` return `anyhow::Result<ExitCode>` (not `Result<()>`) because one
// subcommand — `models path` — must exit nonzero when a model not present, so
// it can compose inside shell scripts. Every other arm just report
// `ExitCode::SUCCESS`; genuine errors still bubble up as `Err` and clap/anyhow
// print them. `models::run` is the only arm that carry a real exit code.
fn main() -> anyhow::Result<ExitCode> {
    let cli = Cli::parse();
    let code = match cli.command {
        Command::Pdf2md {
            input,
            output,
            engine,
            report_json,
            index,
            pages,
            split_by_heading_level,
        } => {
            pdf2md(
                &input,
                output,
                engine,
                Pdf2mdOptions {
                    report_json,
                    index,
                    pages: pages.as_deref(),
                    split_by_heading_level,
                },
            )?;
            ExitCode::SUCCESS
        }
        Command::Scan(args) => {
            scan::run(args)?;
            ExitCode::SUCCESS
        }
        Command::Rename(args) => {
            rename::run(args)?;
            ExitCode::SUCCESS
        }
        Command::CodeActions(args) => {
            code_actions::run(args)?;
            ExitCode::SUCCESS
        }
        Command::Status(args) => {
            status::run(args)?;
            ExitCode::SUCCESS
        }
        Command::Plan(args) => {
            plan::run(args)?;
            ExitCode::SUCCESS
        }
        Command::Ai(args) => {
            ai::run(args)?;
            ExitCode::SUCCESS
        }
        Command::Tui(args) => {
            tui::run(args)?;
            ExitCode::SUCCESS
        }
        Command::Models(args) => models::run(args)?,
    };
    Ok(code)
}

/// The flag-driven knobs on [`Command::Pdf2md`], grouped so the base pipeline
/// signature stays small. Every field defaults to the pre-existing behaviour
/// (no report JSON, no sidecar, all pages, no split), so `pdf2md <IN> -o <OUT>`
/// with an all-default `Pdf2mdOptions` is byte-for-byte the old command.
struct Pdf2mdOptions<'a> {
    report_json: bool,
    index: bool,
    pages: Option<&'a str>,
    split_by_heading_level: Option<usize>,
}

/// One rendered Markdown artifact to be written to disk: its path, its content,
/// and the source page its content begins on (for the sidecar's page ranges).
/// A plain combined conversion is a single one of these; `--split-by-heading-
/// level` produces several.
struct RenderedOutput {
    path: PathBuf,
    markdown: String,
    first_page: Option<usize>,
}

/// Implements [`Command::Pdf2md`]: PDF in, semantic Markdown out, plus a
/// validation report printed to stdout.
///
/// # Pipeline shape (`kopitiam_token_max.md` §2.2)
///
/// The base pipeline — extract → (optional page filter) → reconstruct → render
/// → validate → `fs::write` — is the same one `tui/convert.rs::convert_one`
/// runs. The `Pdf2mdOptions` knobs are all CLI-only, additive, and off by
/// default: with an all-default `Pdf2mdOptions` this function's observable
/// behaviour is identical to the previous two-argument `pdf2md`, so the TUI's
/// mirror of the base pipeline stays correct without gaining these flags.
fn pdf2md(
    input: &Path,
    output: Option<PathBuf>,
    engine: Engine,
    options: Pdf2mdOptions<'_>,
) -> anyhow::Result<()> {
    // Extraction is panic-safe on both paths: even PDFs whose fonts make the
    // legacy `pdf-extract` crate `panic!` come back as a normal `Err`
    // (`ExtractError::UnsupportedFont`) instead of aborting the process. Turn
    // that into a clear, actionable message so that a user batch-converting
    // many PDFs sees this file fail cleanly and can move on, rather than the
    // whole run dying on one bad document.
    let extraction = match engine {
        Engine::Mupdf => kopitiam_pdf::extract_mupdf(input),
        Engine::Legacy => kopitiam_pdf::extract(input),
    };
    let mut pages = match extraction {
        Ok(pages) => pages,
        Err(err @ kopitiam_pdf::ExtractError::UnsupportedFont(_)) => {
            anyhow::bail!(
                "could not extract text from {}: {err} \
                 (the PDF uses fonts with no unicode mapping)",
                input.display()
            );
        }
        Err(err) => {
            return Err(anyhow::Error::new(err)
                .context(format!("could not extract text from {}", input.display())));
        }
    };

    // `--pages A-B` (card I-H): keep only the requested source pages, applied
    // *before* reconstruction so nothing downstream pays for the discarded
    // pages. Selection matches `Page::number` (the PDF's own 1-based numbering),
    // so `--pages 2-3` picks the source's pages 2 and 3. Reconstruction then
    // numbers the kept pages by *position* (`merge_page_breaks` uses
    // `page_index + 1`), so the output's anchors run 1.. across the selection
    // rather than carrying the source numbers — the slice is a standalone
    // document. (Carrying source numbers would mean teaching the engine's
    // position-based numbering to read `Page::number`, which would desync I-B's
    // per-page validation; out of scope here.)
    if let Some(spec) = options.pages {
        let (lo, hi) = parse_page_range(spec)?;
        pages.retain(|page| (lo..=hi).contains(&page.number));
        if pages.is_empty() {
            anyhow::bail!(
                "no pages in {} fall within --pages {spec} (requested {lo}-{hi})",
                input.display()
            );
        }
    }

    // The mupdf engine already returns spans in true reading order (columns
    // linearised, spacing correct), so it uses the pre-ordered reconstruction
    // path that trusts that order. The legacy engine's spans are in raw draw
    // order, so they still go through the full column/baseline analysis.
    let document = match engine {
        Engine::Mupdf => kopitiam_document::reconstruct_preordered(&pages),
        Engine::Legacy => kopitiam_document::reconstruct(&pages),
    };
    let markdown = kopitiam_markdown::render_document(&document);
    // The report always audits the whole document, regardless of how the output
    // is partitioned into files.
    let report = kopitiam_document::validate(&pages, &document, &markdown);

    let output = output.unwrap_or_else(|| input.with_extension("md"));

    // Decide the set of files to write: either the single combined output, or
    // one file per section when `--split-by-heading-level` is given.
    let outputs = match options.split_by_heading_level {
        Some(level) => split_outputs(&document, level, &output)?,
        None => vec![RenderedOutput {
            path: output.clone(),
            markdown: markdown.clone(),
            first_page: document.block_pages.first().copied(),
        }],
    };

    // "Wrote ..." notices go to stderr in --report-json mode so stdout stays
    // clean JSON a caller can pipe straight into `jq`.
    let mut notices: Vec<String> = Vec::new();
    for out in &outputs {
        if let Some(parent) = out.path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&out.path, &out.markdown)?;
        notices.push(format!("Wrote {}", out.path.display()));

        // `--index` (card I-G): a SIDECAR next to each `.md`, never in-band, so
        // the `.md` char count and recovery ratio are untouched.
        if options.index {
            let index_path = sidecar_index_path(&out.path);
            let sidecar = build_sidecar_index(&out.path, &out.markdown, out.first_page);
            std::fs::write(&index_path, serde_json::to_string_pretty(&sidecar)?)?;
            notices.push(format!("Wrote {}", index_path.display()));
        }
    }

    if options.report_json {
        for notice in &notices {
            eprintln!("{notice}");
        }
        println!("{}", serde_json::to_string_pretty(&report.to_json())?);
    } else {
        for notice in &notices {
            println!("{notice}");
        }
        println!();
        println!("{report}");
    }

    Ok(())
}

/// Parses a `--pages` spec: `"A-B"` (inclusive range) or a bare `"N"` (just page
/// N). Rejects zero (pages are 1-based), a reversed range, and non-numeric
/// input with an actionable message.
fn parse_page_range(spec: &str) -> anyhow::Result<(usize, usize)> {
    let parse_one = |s: &str| -> anyhow::Result<usize> {
        let n: usize = s.trim().parse().map_err(|_| {
            anyhow::anyhow!("invalid --pages value {spec:?}: expected `A-B` or `N` with 1-based page numbers")
        })?;
        if n == 0 {
            anyhow::bail!("invalid --pages value {spec:?}: page numbers are 1-based, 0 is not a page");
        }
        Ok(n)
    };

    match spec.split_once('-') {
        Some((lo, hi)) => {
            let lo = parse_one(lo)?;
            let hi = parse_one(hi)?;
            if hi < lo {
                anyhow::bail!("invalid --pages value {spec:?}: end page {hi} is before start page {lo}");
            }
            Ok((lo, hi))
        }
        None => {
            let n = parse_one(spec)?;
            Ok((n, n))
        }
    }
}

/// Builds the per-section [`RenderedOutput`]s for `--split-by-heading-level`
/// (card I-H). Blocks are partitioned into runs that each begin at a heading of
/// exactly `level` (content before the first such heading is a leading part), and
/// each run is rendered as its own document — anchors and all — into a sibling
/// file `<stem>.NN-<slug>.md`.
fn split_outputs(
    document: &kopitiam_document::Document,
    level: usize,
    output: &Path,
) -> anyhow::Result<Vec<RenderedOutput>> {
    use kopitiam_document::{Block, Document};

    if !(1..=6).contains(&level) {
        anyhow::bail!("--split-by-heading-level must be between 1 and 6, got {level}");
    }

    // Section start indices: 0 (the leading part) plus every block that is a
    // heading at `level`. Consecutive starts bound each section's block slice.
    let mut starts: Vec<usize> = vec![0];
    for (i, block) in document.blocks.iter().enumerate() {
        if i == 0 {
            continue;
        }
        if matches!(block, Block::Heading(h) if h.level == level) {
            starts.push(i);
        }
    }
    starts.dedup();

    // The output stem (e.g. `book.md` -> `book`) and directory. This follows the
    // same stem-then-`.md` convention as the TUI's `logic::derive_md_name`
    // (`tui/logic.rs`); that helper lives in a private module unreachable from
    // here, and its body is a one-line `file_stem` lookup, so it is applied
    // directly rather than by widening the module's visibility.
    let stem = output
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("output")
        .to_string();
    let dir = output.parent().map(Path::to_path_buf).unwrap_or_default();

    let mut outputs = Vec::new();
    let mut part = 0usize;
    for w in 0..starts.len() {
        let start = starts[w];
        let end = starts.get(w + 1).copied().unwrap_or(document.blocks.len());
        if start >= end {
            continue;
        }
        let blocks = document.blocks[start..end].to_vec();
        let block_pages = document
            .block_pages
            .get(start..end)
            .map(<[usize]>::to_vec)
            .unwrap_or_default();

        // A section's slug comes from its opening heading, else "preamble" for a
        // leading part with no heading of the split level.
        let slug = match &blocks[0] {
            Block::Heading(h) if h.level == level => slugify(&h.text),
            _ => "preamble".to_string(),
        };
        let section = Document {
            title: None,
            metadata: document.metadata.clone(),
            block_pages: block_pages.clone(),
            blocks,
            citations: Vec::new(),
        };
        let markdown = kopitiam_markdown::render_document(&section);
        let file = format!("{stem}.{part:02}-{slug}.md");
        outputs.push(RenderedOutput {
            path: dir.join(file),
            markdown,
            first_page: block_pages.first().copied(),
        });
        part += 1;
    }

    if outputs.is_empty() {
        // No blocks at all: fall back to a single empty part so `-o` still names
        // a real file rather than silently writing nothing.
        outputs.push(RenderedOutput {
            path: dir.join(format!("{stem}.00-preamble.md")),
            markdown: String::new(),
            first_page: document.block_pages.first().copied(),
        });
    }
    Ok(outputs)
}

/// Slugifies heading text for a split-part filename: ASCII alphanumerics kept
/// (lower-cased), every other run collapsed to a single `-`, trimmed, capped to
/// keep filenames sane. Empty input (a heading of only punctuation) yields
/// `"section"` so a filename is always produced.
fn slugify(text: &str) -> String {
    let mut slug = String::new();
    let mut prev_dash = false;
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            prev_dash = false;
        } else if !prev_dash {
            slug.push('-');
            prev_dash = true;
        }
    }
    let slug = slug.trim_matches('-');
    let slug: String = slug.chars().take(40).collect();
    let slug = slug.trim_matches('-').to_string();
    if slug.is_empty() { "section".to_string() } else { slug }
}

/// The sidecar path for a rendered `.md`: `<output>.index.json` (card I-G). The
/// suffix is *appended* to the full name (`book.md` -> `book.md.index.json`), so
/// the sidecar sorts next to its `.md` and never collides with a real `.md`.
fn sidecar_index_path(output: &Path) -> PathBuf {
    let mut name = output.as_os_str().to_os_string();
    name.push(".index.json");
    output.with_file_name(name)
}

#[derive(serde::Serialize)]
struct HeadingEntry {
    text: String,
    level: usize,
    /// 1-based line of the heading itself.
    line_start: usize,
    /// 1-based last line of the heading's section (through the line before the
    /// next heading of the same or a higher level), inclusive.
    line_end: usize,
}

#[derive(serde::Serialize)]
struct PageEntry {
    page: usize,
    line_start: usize,
    line_end: usize,
}

#[derive(serde::Serialize)]
struct SidecarIndex {
    /// The `.md` this indexes, by file name only (no absolute paths — stable,
    /// relocatable, and diff-friendly).
    source: String,
    total_lines: usize,
    headings: Vec<HeadingEntry>,
    pages: Vec<PageEntry>,
}

/// Builds the `<output>.index.json` payload (card I-G) by parsing the rendered
/// Markdown: heading lines (`#`..`######`) map to their section's line range,
/// and `<!-- page N -->` anchors map each source page to its line range. This
/// converts grep-and-probe over a huge `.md` into lookup-then-slice.
///
/// Line numbers are 1-based to match `rg -n` / editor gutters. A heading's range
/// runs to the line before the next heading of the same or a higher level (so a
/// chapter's range spans its subsections); a page's range runs to the line
/// before the next page anchor. `first_page` seeds the first page's number (the
/// page the document's first block starts on — no anchor precedes it).
fn build_sidecar_index(output: &Path, markdown: &str, first_page: Option<usize>) -> SidecarIndex {
    let lines: Vec<&str> = markdown.lines().collect();
    let total_lines = lines.len();

    // Pass 1: collect raw headings (1-based line, level, text).
    let mut raw_headings: Vec<(usize, usize, String)> = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if let Some((level, text)) = parse_heading_line(line) {
            raw_headings.push((i + 1, level, text));
        }
    }
    // Resolve each heading's end: the line before the next heading whose level
    // is <= this one's (that heading starts a sibling/ancestor section), else EOF.
    let mut headings = Vec::with_capacity(raw_headings.len());
    for (idx, (line_start, level, text)) in raw_headings.iter().enumerate() {
        let line_end = raw_headings
            .iter()
            .skip(idx + 1)
            .find(|(_, l, _)| l <= level)
            .map(|(next_start, _, _)| next_start - 1)
            .unwrap_or(total_lines)
            .max(*line_start);
        headings.push(HeadingEntry {
            text: text.clone(),
            level: *level,
            line_start: *line_start,
            line_end,
        });
    }

    // Pass 2: page ranges from the anchors. The first page starts at line 1;
    // each anchor at line L starts a new page there (the anchor line belongs to
    // the new page as its marker). A document with no anchors is a single page.
    let mut pages = Vec::new();
    if let Some(first) = first_page {
        let mut cur_page = first;
        let mut cur_start = 1usize;
        for (i, line) in lines.iter().enumerate() {
            if let Some(page) = parse_page_anchor_line(line.trim()) {
                let line_no = i + 1;
                pages.push(PageEntry {
                    page: cur_page,
                    line_start: cur_start,
                    line_end: line_no.saturating_sub(1).max(cur_start),
                });
                cur_page = page;
                cur_start = line_no;
            }
        }
        pages.push(PageEntry {
            page: cur_page,
            line_start: cur_start,
            line_end: total_lines.max(cur_start),
        });
    }

    SidecarIndex {
        source: output
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        total_lines,
        headings,
        pages,
    }
}

/// Parses a rendered heading line (`Heading::render` emits `"{#..} {text}"`),
/// returning `(level, text)`. Mirrors the renderer's exact shape: 1-6 leading
/// `#` then a single space then the text.
fn parse_heading_line(line: &str) -> Option<(usize, String)> {
    let hashes = line.chars().take_while(|&c| c == '#').count();
    if (1..=6).contains(&hashes) && line.as_bytes().get(hashes) == Some(&b' ') {
        Some((hashes, line[hashes + 1..].trim().to_string()))
    } else {
        None
    }
}

/// Parses a `<!-- page N -->` anchor line, returning `N`. The literal form is
/// the one `kopitiam-markdown`'s renderer emits and `kopitiam-document`'s
/// validator parses (card I-B); kept in sync by tests on all three sides
/// (`kopitiam_token_max.md` §2.3).
fn parse_page_anchor_line(line: &str) -> Option<usize> {
    line.strip_prefix("<!-- page ")?
        .strip_suffix(" -->")?
        .trim()
        .parse()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kopitiam_document::{
        Block, ConversionReport, Document, Heading, Metadata, PageRecovery, Paragraph,
    };

    // ---- I-H: --pages A-B parsing -------------------------------------------

    #[test]
    fn parse_page_range_accepts_a_range_and_a_single_page() {
        assert_eq!(parse_page_range("5-12").unwrap(), (5, 12));
        assert_eq!(parse_page_range("7").unwrap(), (7, 7));
        // Whitespace around the numbers is tolerated.
        assert_eq!(parse_page_range(" 3 - 4 ").unwrap(), (3, 4));
    }

    #[test]
    fn parse_page_range_rejects_zero_reversed_and_garbage() {
        // Pages are 1-based, so 0 is not a page.
        assert!(parse_page_range("0-3").is_err());
        assert!(parse_page_range("0").is_err());
        // A reversed range is a user error, not an empty selection.
        assert!(parse_page_range("9-2").is_err());
        // Non-numeric input fails cleanly rather than silently converting all.
        assert!(parse_page_range("abc").is_err());
        assert!(parse_page_range("1-b").is_err());
    }

    // ---- I-H: --split-by-heading-level --------------------------------------

    fn doc_with_pages(blocks: Vec<Block>, block_pages: Vec<usize>) -> Document {
        Document {
            title: None,
            metadata: Metadata { source_pages: *block_pages.iter().max().unwrap_or(&0) },
            block_pages,
            blocks,
            citations: Vec::new(),
        }
    }

    #[test]
    fn slugify_lowercases_and_collapses_punctuation() {
        assert_eq!(slugify("Chapter 1: Introduction"), "chapter-1-introduction");
        assert_eq!(slugify("  Results & Discussion  "), "results-discussion");
        // A heading of only punctuation still yields a usable name.
        assert_eq!(slugify("***"), "section");
    }

    #[test]
    fn split_outputs_makes_one_file_per_level_heading_with_a_leading_preamble() {
        // Preamble paragraph (page 1), then two H1 chapters (pages 1 and 2). A
        // nested H2 must NOT start a new part when splitting at level 1.
        let doc = doc_with_pages(
            vec![
                Block::Paragraph(Paragraph { text: "Front matter.".to_string() }),
                Block::Heading(Heading { level: 1, text: "Introduction".to_string() }),
                Block::Paragraph(Paragraph { text: "Intro body.".to_string() }),
                Block::Heading(Heading { level: 2, text: "Subsection".to_string() }),
                Block::Paragraph(Paragraph { text: "Sub body.".to_string() }),
                Block::Heading(Heading { level: 1, text: "Methods".to_string() }),
                Block::Paragraph(Paragraph { text: "Methods body.".to_string() }),
            ],
            vec![1, 1, 1, 1, 1, 2, 2],
        );
        let outputs = split_outputs(&doc, 1, Path::new("/out/book.md")).unwrap();

        assert_eq!(outputs.len(), 3, "preamble + two H1 chapters");
        assert!(outputs[0].path.ends_with("book.00-preamble.md"));
        assert!(outputs[1].path.ends_with("book.01-introduction.md"));
        assert!(outputs[2].path.ends_with("book.02-methods.md"));

        // The nested H2 stays inside the Introduction part, not its own file.
        assert!(outputs[1].markdown.contains("## Subsection"));
        // Methods begins on page 2, so its part records first_page = 2.
        assert_eq!(outputs[2].first_page, Some(2));
        // Chapters render as their own documents (their own heading present).
        assert!(outputs[2].markdown.contains("# Methods"));
    }

    #[test]
    fn split_outputs_without_a_leading_preamble_starts_at_part_00() {
        // First block is already a level-1 heading: no preamble part.
        let doc = doc_with_pages(
            vec![
                Block::Heading(Heading { level: 1, text: "Only Chapter".to_string() }),
                Block::Paragraph(Paragraph { text: "Body.".to_string() }),
            ],
            vec![1, 1],
        );
        let outputs = split_outputs(&doc, 1, Path::new("out.md")).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].path.ends_with("out.00-only-chapter.md"));
    }

    #[test]
    fn split_outputs_rejects_an_out_of_range_level() {
        let doc = doc_with_pages(vec![Block::Paragraph(Paragraph { text: "x".to_string() })], vec![1]);
        assert!(split_outputs(&doc, 0, Path::new("out.md")).is_err());
        assert!(split_outputs(&doc, 7, Path::new("out.md")).is_err());
    }

    // ---- I-G: sidecar index --------------------------------------------------

    #[test]
    fn sidecar_index_path_appends_index_json_to_the_full_name() {
        assert_eq!(
            sidecar_index_path(Path::new("/a/b/book.md")),
            PathBuf::from("/a/b/book.md.index.json")
        );
    }

    #[test]
    fn build_sidecar_index_maps_headings_and_pages_to_line_ranges() {
        // Rendered shape mirrors the renderer: H1, body, H2, body, page anchor,
        // then a second H1 on page 2. Lines are 1-based.
        //  1: # Introduction
        //  2: (blank)
        //  3: Intro body.
        //  4: (blank)
        //  5: ## Background
        //  6: (blank)
        //  7: Background body.
        //  8: (blank)
        //  9: <!-- page 2 -->
        // 10: (blank)
        // 11: # Methods
        // 12: (blank)
        // 13: Methods body.
        let md = "# Introduction\n\nIntro body.\n\n## Background\n\nBackground body.\n\n<!-- page 2 -->\n\n# Methods\n\nMethods body.\n";
        let index = build_sidecar_index(Path::new("paper.md"), md, Some(1));

        assert_eq!(index.source, "paper.md");
        assert_eq!(index.total_lines, 13);

        // Headings: three of them, with section ranges that respect nesting.
        assert_eq!(index.headings.len(), 3);
        let intro = &index.headings[0];
        assert_eq!(intro.text, "Introduction");
        assert_eq!(intro.level, 1);
        assert_eq!(intro.line_start, 1);
        // Introduction runs until just before the next same-or-higher heading
        // (# Methods at line 11), so it spans its ## Background subsection.
        assert_eq!(intro.line_end, 10);

        let background = &index.headings[1];
        assert_eq!(background.text, "Background");
        assert_eq!(background.level, 2);
        assert_eq!(background.line_start, 5);
        assert_eq!(background.line_end, 10);

        let methods = &index.headings[2];
        assert_eq!(methods.text, "Methods");
        assert_eq!(methods.line_start, 11);
        assert_eq!(methods.line_end, 13); // through EOF

        // Pages: page 1 up to the anchor, page 2 from the anchor to EOF.
        assert_eq!(index.pages.len(), 2);
        assert_eq!(index.pages[0].page, 1);
        assert_eq!(index.pages[0].line_start, 1);
        assert_eq!(index.pages[0].line_end, 8); // line before the anchor
        assert_eq!(index.pages[1].page, 2);
        assert_eq!(index.pages[1].line_start, 9); // the anchor line
        assert_eq!(index.pages[1].line_end, 13);
    }

    #[test]
    fn build_sidecar_index_without_page_provenance_emits_no_pages() {
        // No first_page (a hand-built doc with no anchors) -> empty pages array,
        // mirroring the validator's honesty: no page to attribute lines to.
        let md = "# Title\n\nBody.\n";
        let index = build_sidecar_index(Path::new("x.md"), md, None);
        assert!(index.pages.is_empty());
        assert_eq!(index.headings.len(), 1);
    }

    #[test]
    fn parse_heading_and_anchor_lines_match_the_renderer_forms() {
        assert_eq!(parse_heading_line("### Deep"), Some((3, "Deep".to_string())));
        assert_eq!(parse_heading_line("####### TooDeep"), None); // 7 hashes is not a heading
        assert_eq!(parse_heading_line("#NoSpace"), None);
        assert_eq!(parse_heading_line("Body text"), None);
        assert_eq!(parse_page_anchor_line("<!-- page 717 -->"), Some(717));
        assert_eq!(parse_page_anchor_line("<!-- note -->"), None);
    }

    // ---- I-F: --report-json --------------------------------------------------

    #[test]
    fn report_json_serialization_exposes_the_gating_signals() {
        // The exact call the CLI makes for --report-json: to_json then to_string.
        // A caller must be able to read recovery_ratio / passes / per-page ratios
        // straight out of the JSON without parsing prose (card I-F, §0.2).
        let report = ConversionReport {
            pages: 2,
            extracted_words: 20,
            rendered_words: 20,
            extracted_chars: 200,
            rendered_chars: 200,
            headings_found: 2,
            lists_found: 0,
            tables_found: 1,
            citations_found: 0,
            per_page: vec![
                PageRecovery { page: 1, extracted_chars: 120, rendered_chars: 120 },
                PageRecovery { page: 2, extracted_chars: 80, rendered_chars: 80 },
            ],
        };
        let json: serde_json::Value =
            serde_json::from_str(&serde_json::to_string_pretty(&report.to_json()).unwrap()).unwrap();

        assert_eq!(json["passes"], true);
        assert!((json["recovery_ratio"].as_f64().unwrap() - 1.0).abs() < 1e-9);
        let per_page = json["per_page"].as_array().unwrap();
        assert_eq!(per_page.len(), 2);
        assert_eq!(per_page[1]["page"], 2);
        assert!(per_page[1].get("recovery_ratio").is_some());
    }
}

//! The `translate` subcommand: the end-to-end Chinese-literature translation
//! pipeline the III-4..7 engines were built for (`kopitiam_token_max.md` §13).
//!
//! This command is a thin integrator — it invents no translation logic, it only
//! wires the committed engines into one pass over a converted Markdown document:
//!
//! ```text
//!   read .md
//!     -> kopitiam_document::segments          (III-4: stable, content-hashed segments)
//!     -> TranslationMemory::plan              (III-4: reuse cached translations, zero cost)
//!     -> draft_and_route  [cache misses only] (III-6: local model drafts + routes)
//!          applying Glossary as the post-pass (III-5: deterministic terminology)
//!     -> render_bilingual                     (III-7: aligned source/target + anchors)
//!     -> write output  (+ record new translations into the TM)
//! ```
//!
//! # The saving is measurable (§0.2)
//!
//! Three numbers make the token saving visible, all in `--json`:
//!
//! * `reuse_fraction` (III-4) — the fraction of segments served from the
//!   translation memory at zero re-translation cost. On a revised document only
//!   the changed segments miss the cache; everything else is free.
//! * `two_pass.local_fraction` (III-6) — the fraction the local model handled
//!   without a cloud review.
//! * `review.review_fraction` (III-7) — the fraction a reviewer must actually
//!   read; `review_targets` are exactly the anchors to jump to.
//!
//! # Honesty about SendToCloud (there is no cloud adapter wired)
//!
//! [`draft_and_route`] routes low-confidence segments to
//! [`kopitiam_ai::Decision::SendToCloud`]. This wave wires **no cloud adapter**
//! (that is a later wave), so those segments are *not* silently passed off as
//! finished: their local draft is kept as a placeholder, they are marked
//! `needs_review` in the bilingual output (carrying the visible review marker),
//! and their anchors appear in `review_targets`. The output is therefore honest
//! — "local drafts, flagged for a review that still has to happen" — rather than
//! pretending a cloud model already checked them.
//!
//! Under the [`kopitiam_ai::EchoAdapter`] fallback (no local `.gguf` on disk) the
//! "draft" is the source echoed back; that is surfaced through the plan's
//! `notes` (the echo pass-through caveat) so a reader never mistakes an echo run
//! for real translation.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use kopitiam_ai::{
    BilingualOptions, BilingualSegment, DEFAULT_REVIEW_THRESHOLD, Decision, Glossary, Layout,
    ModelAdapter, TwoPassConfig, TwoPassSummary, draft_and_route, render_bilingual, review_coverage,
    review_targets,
};
use kopitiam_document::{Block, Document, Heading, Metadata, Paragraph, Segment, segments};
use kopitiam_workspace::{SegmentId as TmSegmentId, TmSegment, TranslationMemory};
use serde::Serialize;

use crate::adapter::select_adapter;

/// Options for `kopitiam translate`.
#[derive(Args, Debug)]
pub struct TranslateArgs {
    /// The converted Markdown document to translate (typically a `pdf2md`
    /// output). Split into segments at block boundaries; page anchors and other
    /// bare HTML comments are not translatable and are skipped.
    pub input: PathBuf,

    /// A project glossary applied deterministically as the post-pass (III-5),
    /// in the simple `source = target` line format (`#` comments allowed). Every
    /// occurrence of a source term becomes byte-identical target text — zero
    /// model tokens spent on terminology, no drift across the document.
    #[arg(long)]
    pub glossary: Option<PathBuf>,

    /// Bilingual layout: `interleaved` (anchor, source, target-as-blockquote per
    /// segment) or `table` (one Markdown table, `seg | source | target | review`
    /// rows). Both carry the stable `<!-- seg N -->` anchors (III-7).
    #[arg(long, value_enum, default_value_t = LayoutArg::Interleaved)]
    pub layout: LayoutArg,

    /// Skip the translation memory entirely: neither reuse cached translations
    /// nor record new ones. Every segment is (re-)drafted. Without this flag the
    /// TM in `<root>/.kopitiam` is consulted and updated (III-4).
    #[arg(long)]
    pub no_cache: bool,

    /// Where to write the bilingual Markdown. Defaults to the input path with a
    /// `.bilingual.md` extension beside it.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Directory holding the project's `.kopitiam` translation-memory store.
    /// Defaults to the current directory.
    #[arg(long, default_value = ".")]
    pub root: PathBuf,

    /// Emit the machine-readable report (`reuse_fraction`, the two-pass summary,
    /// and `review_targets`) as JSON instead of the human summary, so a caller
    /// gates on the saving without parsing prose (§0.2). The "Wrote ..." notice
    /// and the adapter notice go to stderr in this mode.
    #[arg(long)]
    pub json: bool,
}

/// CLI spelling of [`kopitiam_ai::Layout`] — `interleaved` | `table`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum LayoutArg {
    /// Anchor, source, then target as a blockquote, per segment.
    Interleaved,
    /// A single `seg | source | target | review` Markdown table.
    Table,
}

impl LayoutArg {
    fn to_layout(self) -> Layout {
        match self {
            LayoutArg::Interleaved => Layout::Interleaved,
            LayoutArg::Table => Layout::SideBySideTable,
        }
    }
}

/// Bridges a `kopitiam_document::Segment` to the translation memory's
/// [`TmSegment`] trait, so `TranslationMemory::plan` classifies it as a hit or
/// miss by content hash. The two crates agree on the hex id (identical FNV-1a
/// constants) but stay decoupled; this is the one-line wiring seam between them.
struct TmDocSegment<'a>(&'a Segment);

impl TmSegment for TmDocSegment<'_> {
    fn tm_id(&self) -> TmSegmentId {
        TmSegmentId::new(self.0.id.as_str())
    }
    fn source_text(&self) -> &str {
        &self.0.text
    }
}

/// The reportable outcome of a translation pass — the numbers that make the
/// saving visible (§0.2). Serialized directly for `--json`.
#[derive(Debug, Clone, PartialEq, Serialize)]
struct TranslateReport {
    /// Total translation segments in the document.
    segments: usize,
    /// Whether the translation memory was consulted (`false` under `--no-cache`).
    cache_enabled: bool,
    /// Segments served from the TM at zero re-translation cost (III-4).
    reused: usize,
    /// Segments that had to be drafted this run (cache misses).
    translated: usize,
    /// `reused / segments` in `[0, 1]` — the III-4 saving figure.
    reuse_fraction: f64,
    /// The III-6 local/cloud split over the drafted segments.
    two_pass: TwoPassSummary,
    /// The III-7 review saving: which anchors a reviewer must actually read.
    review: ReviewReport,
    /// Plan-wide caveats (echo pass-through, "AcceptLocal is routing not
    /// correctness", and the no-cloud-adapter note).
    notes: Vec<String>,
}

/// The III-7 review-side saving, flattened for the report.
#[derive(Debug, Clone, PartialEq, Serialize)]
struct ReviewReport {
    /// Anchors (`<!-- seg <id> -->` ids) a reviewer must check — the "read only
    /// these" list. Includes every SendToCloud segment (flagged `needs_review`).
    targets: Vec<String>,
    /// How many segments are flagged for review.
    flagged: usize,
    /// Total segments.
    total: usize,
    /// `flagged / total` in `[0, 1]`.
    review_fraction: f32,
}

/// The full result of [`translate_document`]: the rendered Markdown plus its
/// report. Kept together so `run` writes one and prints the other, and tests can
/// assert on both without touching the filesystem for the model.
struct TranslateResult {
    markdown: String,
    report: TranslateReport,
}

/// Runs `kopitiam translate`: read, translate, render, write, report.
pub fn run(args: TranslateArgs) -> Result<()> {
    let markdown = std::fs::read_to_string(&args.input)
        .with_context(|| format!("reading {}", args.input.display()))?;

    let glossary = match &args.glossary {
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("reading glossary {}", path.display()))?;
            Some(Glossary::from_text(&text).with_context(|| format!("parsing glossary {}", path.display()))?)
        }
        None => None,
    };

    // The TM lives under <root>/.kopitiam; canonicalize so a relative --root is
    // stable regardless of where the store's helpers resolve it.
    let root = std::fs::canonicalize(&args.root)
        .with_context(|| format!("resolving --root {}", args.root.display()))?;

    // Offline-First: a real local model if one is on disk, else the echo stub.
    // The notice tells the user which rung answered; it goes to stderr so
    // --json stdout stays clean.
    let selected = select_adapter();
    eprintln!("{}", selected.notice());

    let result = translate_document(
        selected.adapter(),
        &root,
        &markdown,
        glossary.as_ref(),
        args.layout.to_layout(),
        !args.no_cache,
    )?;

    let output = args
        .output
        .unwrap_or_else(|| default_output_path(&args.input));
    std::fs::write(&output, &result.markdown)
        .with_context(|| format!("writing {}", output.display()))?;
    let wrote = format!("Wrote {}", output.display());

    if args.json {
        eprintln!("{wrote}");
        println!("{}", serde_json::to_string_pretty(&result.report)?);
    } else {
        println!("{wrote}");
        print!("{}", human_summary(&result.report));
    }
    Ok(())
}

/// The I/O-free core: turn a Markdown document into bilingual output and a
/// report, driving the injected `adapter`. Factored out of [`run`] so tests
/// exercise the whole segments → TM → two-pass → glossary → bilingual pipeline
/// against the deterministic [`kopitiam_ai::EchoAdapter`] with a tempdir `root`,
/// no model download and no stdout capture.
fn translate_document(
    adapter: &dyn ModelAdapter,
    root: &Path,
    markdown: &str,
    glossary: Option<&Glossary>,
    layout: Layout,
    use_cache: bool,
) -> Result<TranslateResult> {
    let document = markdown_to_document(markdown);
    let segs = segments(&document);
    let n = segs.len();

    let mut targets: Vec<Option<String>> = vec![None; n];
    // Per-segment (confidence, needs_review) for the drafted (miss) segments;
    // reused hits stay (None, false) — already decided in a prior run.
    let mut meta: Vec<(Option<f32>, bool)> = vec![(None, false); n];

    let mut tm = TranslationMemory::load(root)?;

    // III-4: classify each segment as a cache hit (reuse) or miss (draft).
    let mut miss_indices: Vec<usize> = Vec::new();
    let mut miss_texts: Vec<String> = Vec::new();
    let (reused, reuse_fraction) = if use_cache {
        let tm_segs: Vec<TmDocSegment> = segs.iter().map(TmDocSegment).collect();
        let plan = tm.plan(&tm_segs);
        for hit in &plan.hits {
            targets[hit.index] = Some(hit.translation.clone());
        }
        for miss in &plan.misses {
            miss_indices.push(miss.index);
            miss_texts.push(miss.source_text.clone());
        }
        (plan.hits.len(), plan.reuse_fraction)
    } else {
        for seg in &segs {
            miss_indices.push(seg.index);
            miss_texts.push(seg.text.clone());
        }
        (0, 0.0)
    };

    // III-6 (+ III-5 glossary post-pass): the local model drafts the misses and
    // routes each. draft_and_route indexes by position in this slice, so map
    // back to the document segment index via `miss_indices`.
    let cfg = TwoPassConfig::default();
    let plan = draft_and_route(adapter, &miss_texts, glossary, &cfg);
    for (k, seg_plan) in plan.segments.iter().enumerate() {
        let doc_index = miss_indices[k];
        targets[doc_index] = Some(seg_plan.local_draft.clone());
        meta[doc_index] = (
            Some(seg_plan.confidence),
            seg_plan.decision == Decision::SendToCloud,
        );
        // III-4: record the new translation so a later run reuses it for free.
        if use_cache {
            tm.record(
                TmSegmentId::new(segs[doc_index].id.as_str()),
                segs[doc_index].text.clone(),
                seg_plan.local_draft.clone(),
            )?;
        }
    }

    // III-7: aligned bilingual output with stable per-segment anchors.
    let bi: Vec<BilingualSegment> = segs
        .iter()
        .map(|seg| {
            let (confidence, needs_review) = meta[seg.index];
            BilingualSegment {
                id: seg.index.to_string(),
                index: seg.index,
                source: seg.text.clone(),
                target: targets[seg.index].clone().unwrap_or_default(),
                confidence,
                needs_review,
            }
        })
        .collect();

    let options = BilingualOptions {
        layout,
        low_confidence_threshold: DEFAULT_REVIEW_THRESHOLD,
    };
    let rendered = render_bilingual(&bi, &options);
    let coverage = review_coverage(&bi);
    let review_target_ids: Vec<String> =
        review_targets(&bi).iter().map(|s| (*s).to_string()).collect();

    let mut notes = plan.notes.clone();
    notes.push(
        "SendToCloud segments are kept as local drafts and marked needs_review (no cloud adapter \
         is wired this wave); see review_targets for the anchors to check."
            .to_string(),
    );

    let report = TranslateReport {
        segments: n,
        cache_enabled: use_cache,
        reused,
        translated: miss_indices.len(),
        reuse_fraction,
        two_pass: plan.summary(),
        review: ReviewReport {
            targets: review_target_ids,
            flagged: coverage.flagged,
            total: coverage.total,
            review_fraction: coverage.review_fraction,
        },
        notes,
    };

    Ok(TranslateResult { markdown: rendered, report })
}

/// Parses converted Markdown into a minimal [`Document`] for
/// [`kopitiam_document::segments`]: blank-line-separated blocks become one
/// [`Block`] each, an ATX heading line becomes a [`Block::Heading`], everything
/// else a [`Block::Paragraph`]. Bare HTML-comment blocks (the `<!-- page N -->`
/// anchors `pdf2md` emits) carry no translatable text and are dropped, so they
/// never become segments.
///
/// This is a deliberately small block splitter, not a full CommonMark parser:
/// the translation unit is the paragraph/heading, and the segment id is derived
/// from normalized text, so exact inline Markdown structure does not matter to
/// the identity or the translation.
fn markdown_to_document(markdown: &str) -> Document {
    let mut blocks: Vec<Block> = Vec::new();
    let mut current: Vec<&str> = Vec::new();

    let mut flush = |current: &mut Vec<&str>| {
        if current.is_empty() {
            return;
        }
        let text = current.join("\n");
        current.clear();
        let trimmed = text.trim();
        if trimmed.is_empty() || is_bare_html_comment(trimmed) {
            return;
        }
        match parse_atx_heading(trimmed) {
            Some((level, heading_text)) => blocks.push(Block::Heading(Heading {
                level,
                text: heading_text,
            })),
            None => blocks.push(Block::Paragraph(Paragraph {
                text: trimmed.to_string(),
            })),
        }
    };

    for line in markdown.lines() {
        if line.trim().is_empty() {
            flush(&mut current);
        } else {
            current.push(line);
        }
    }
    flush(&mut current);

    Document {
        title: None,
        metadata: Metadata::default(),
        blocks,
        block_pages: Vec::new(),
        citations: Vec::new(),
    }
}

/// Whether a whole (trimmed) block is a single HTML comment — e.g. a
/// `<!-- page 3 -->` page anchor — and therefore not translatable.
fn is_bare_html_comment(block: &str) -> bool {
    block.starts_with("<!--") && block.ends_with("-->") && block.matches("<!--").count() == 1
}

/// Parses a single-line ATX heading (`# ` .. `###### `), returning
/// `(level, text)`. Multi-line blocks and non-heading lines return `None`.
fn parse_atx_heading(block: &str) -> Option<(usize, String)> {
    if block.contains('\n') {
        return None;
    }
    let hashes = block.chars().take_while(|&c| c == '#').count();
    if (1..=6).contains(&hashes) && block.as_bytes().get(hashes) == Some(&b' ') {
        Some((hashes, block[hashes + 1..].trim().to_string()))
    } else {
        None
    }
}

/// The default output path: the input with a `.bilingual.md` extension beside
/// it (`paper.md` → `paper.bilingual.md`).
fn default_output_path(input: &Path) -> PathBuf {
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("translated");
    input.with_file_name(format!("{stem}.bilingual.md"))
}

/// The human-readable summary printed when `--json` is off.
fn human_summary(report: &TranslateReport) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(out, "Segments: {}", report.segments);
    if report.cache_enabled {
        let _ = writeln!(
            out,
            "Translation memory: {} reused, {} translated (reuse {:.0}%)",
            report.reused,
            report.translated,
            report.reuse_fraction * 100.0,
        );
    } else {
        let _ = writeln!(out, "Translation memory: disabled (--no-cache)");
    }
    let s = &report.two_pass;
    let _ = writeln!(
        out,
        "Two-pass draft: {} accepted-local, {} send-to-cloud (local {:.0}%)",
        s.accepted_local,
        s.sent_to_cloud,
        s.local_fraction * 100.0,
    );
    let _ = writeln!(
        out,
        "Review: {} of {} segments flagged ({:.0}% to read)",
        report.review.flagged,
        report.review.total,
        report.review.review_fraction * 100.0,
    );
    if !report.review.targets.is_empty() {
        let _ = writeln!(out, "  review anchors: {}", report.review.targets.join(", "));
    }
    for note in &report.notes {
        let _ = writeln!(out, "note: {note}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use kopitiam_ai::EchoAdapter;

    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        args: TranslateArgs,
    }

    #[test]
    fn parses_flags() {
        let cli = TestCli::try_parse_from([
            "t", "in.md", "--layout", "table", "--no-cache", "--glossary", "g.txt", "-o", "out.md",
        ])
        .unwrap();
        assert_eq!(cli.args.input, PathBuf::from("in.md"));
        assert_eq!(cli.args.layout, LayoutArg::Table);
        assert!(cli.args.no_cache);
        assert_eq!(cli.args.glossary, Some(PathBuf::from("g.txt")));
        assert_eq!(cli.args.output, Some(PathBuf::from("out.md")));
    }

    #[test]
    fn markdown_splits_into_heading_and_paragraph_segments_skipping_anchors() {
        let md = "# 标题\n\n第一段中文内容。\n\n<!-- page 2 -->\n\n第二段中文内容。\n";
        let doc = markdown_to_document(md);
        // The page anchor block is dropped; heading + two paragraphs remain.
        assert_eq!(doc.blocks.len(), 3);
        assert!(matches!(&doc.blocks[0], Block::Heading(h) if h.level == 1 && h.text == "标题"));
        assert!(matches!(&doc.blocks[1], Block::Paragraph(p) if p.text == "第一段中文内容。"));
        assert!(matches!(&doc.blocks[2], Block::Paragraph(p) if p.text == "第二段中文内容。"));

        let segs = segments(&doc);
        assert_eq!(segs.len(), 3);
    }

    #[test]
    fn default_output_path_appends_bilingual() {
        assert_eq!(
            default_output_path(Path::new("/a/paper.md")),
            PathBuf::from("/a/paper.bilingual.md")
        );
    }

    /// The headline pipeline test: a 2-segment document + a glossary, translated
    /// via the echo stub, produces bilingual output with anchors, the glossary
    /// applied in the target, and a summary — and a *second* run reuses the TM so
    /// `reuse_fraction` rises from 0 to 1.
    #[test]
    fn translate_produces_bilingual_output_and_the_tm_reuses_on_a_second_run() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let md = "华龙一号的数字孪生模型用于压水堆安全评估研究工作与验证。\n\n第二段较长的中文技术文本内容用于测试双语对齐输出与翻译记忆复用。\n";
        let glossary = Glossary::from_text("华龙一号 = Hualong One\n数字孪生 = digital twin\n").unwrap();

        // First run: empty TM, both segments are misses (translated + recorded).
        let first = translate_document(
            &EchoAdapter,
            root,
            md,
            Some(&glossary),
            Layout::Interleaved,
            true,
        )
        .unwrap();

        assert_eq!(first.report.segments, 2);
        assert_eq!(first.report.reused, 0);
        assert_eq!(first.report.translated, 2);
        assert!((first.report.reuse_fraction - 0.0).abs() < 1e-9);
        // Anchors present for both segments.
        assert!(first.markdown.contains("<!-- seg 0 -->"));
        assert!(first.markdown.contains("<!-- seg 1 -->"));
        // Under echo the draft echoes the source, so the deterministic glossary
        // post-pass rewrites the terms in the target text.
        assert!(first.markdown.contains("Hualong One"));
        assert!(first.markdown.contains("digital twin"));
        // The echo pass-through caveat is surfaced in the notes.
        assert!(first.report.notes.iter().any(|n| n.contains("echo stub")));
        // The two-pass summary accounts for every drafted segment.
        assert_eq!(first.report.two_pass.total, 2);

        // Second run: the TM now holds both segments, so both are hits.
        let second = translate_document(
            &EchoAdapter,
            root,
            md,
            Some(&glossary),
            Layout::Interleaved,
            true,
        )
        .unwrap();
        assert_eq!(second.report.reused, 2);
        assert_eq!(second.report.translated, 0);
        assert!(
            second.report.reuse_fraction > first.report.reuse_fraction,
            "reuse_fraction must rise on the second run: {} !> {}",
            second.report.reuse_fraction,
            first.report.reuse_fraction
        );
        assert!((second.report.reuse_fraction - 1.0).abs() < 1e-9);
    }

    /// SendToCloud segments are surfaced honestly: flagged for review and listed
    /// in `review_targets`, never silently passed off as finished. Under the
    /// conservative echo default every segment routes to cloud.
    #[test]
    fn send_to_cloud_segments_are_flagged_for_review() {
        let dir = tempfile::tempdir().unwrap();
        let md = "一段足够长的中文技术文本内容用于测试低置信度段落被标记为需要复审的行为。\n";
        let result = translate_document(
            &EchoAdapter,
            dir.path(),
            md,
            None,
            Layout::Interleaved,
            true,
        )
        .unwrap();
        assert_eq!(result.report.two_pass.sent_to_cloud, 1);
        // The one segment is a review target and carries the visible marker.
        assert_eq!(result.report.review.targets, vec!["0".to_string()]);
        assert!(result.markdown.contains("⚠ review"));
    }

    /// `--no-cache` disables the TM: nothing is reused, nothing recorded, so two
    /// runs both translate from scratch.
    #[test]
    fn no_cache_never_reuses() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let md = "一段用于测试禁用翻译记忆的中文文本内容。\n";
        for _ in 0..2 {
            let r =
                translate_document(&EchoAdapter, root, md, None, Layout::Interleaved, false).unwrap();
            assert!(!r.report.cache_enabled);
            assert_eq!(r.report.reused, 0);
            assert_eq!(r.report.reuse_fraction, 0.0);
        }
    }

    #[test]
    fn table_layout_renders_a_bilingual_table() {
        let dir = tempfile::tempdir().unwrap();
        let md = "第一段中文。\n\n第二段中文。\n";
        let r =
            translate_document(&EchoAdapter, dir.path(), md, None, Layout::SideBySideTable, true).unwrap();
        assert!(r.markdown.contains("| seg | source | target | review |"));
        assert!(r.markdown.contains("<!-- seg 0 -->"));
    }

    #[test]
    fn report_json_shape_exposes_the_saving_signals() {
        let dir = tempfile::tempdir().unwrap();
        let md = "第一段。\n\n第二段。\n";
        let r =
            translate_document(&EchoAdapter, dir.path(), md, None, Layout::Interleaved, true).unwrap();
        let json: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&r.report).unwrap()).unwrap();
        assert!(json["reuse_fraction"].as_f64().is_some());
        assert!(json["two_pass"]["local_fraction"].as_f64().is_some());
        assert!(json["review"]["targets"].as_array().is_some());
    }
}

//! The `preprocess` subcommand group: local-model preprocessing — token-max
//! Task II-6 (`kopitiam_token_max.md` §11, §318).
//!
//! High-volume, low-judgment work is routed to the **local** model so the cloud
//! model never sees the raw volume: `preprocess summarize <file> --lines N`
//! compresses a file, `preprocess triage <query> <candidate>...` filters
//! grep-style hits. Both are thin wrappers over the committed
//! `kopitiam_ai::{summarize, triage}` helpers, driven by the local adapter
//! (`select_adapter().adapter()`), so they cost **zero cloud tokens**.
//!
//! # Honest about capability (the card's central requirement, §321)
//!
//! A 0.5B model cannot be trusted with judgment, so these are *filtering /
//! compression* only, never the final authority. Every result carries a
//! `DropReport` of exactly what was removed (recoverable, verbatim where it can
//! be enumerated), which this command always prints. When no local `.gguf` is on
//! disk the adapter is the echo stub: there is **no real preprocessing**, the
//! output is a pass-through of the input, and the helpers stamp the
//! `ECHO_PASSTHROUGH_NOTE` into the report — surfaced here so a reader never
//! mistakes an echo run for genuine filtering.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use kopitiam_ai::{DropReport, summarize, triage};

use crate::adapter::select_adapter;

/// Options for `kopitiam preprocess`.
#[derive(Args, Debug)]
pub struct PreprocessArgs {
    #[command(subcommand)]
    action: PreprocessAction,
}

/// The two preprocessing actions.
#[derive(Subcommand, Debug)]
enum PreprocessAction {
    /// Compress a file to at most `--lines N` lines via the local model.
    Summarize(SummarizeArgs),
    /// Keep only the candidates plausibly relevant to a query (conservative:
    /// keeps all on an unusable reply).
    Triage(TriageArgs),
}

/// Options for `kopitiam preprocess summarize`.
#[derive(Args, Debug)]
pub struct SummarizeArgs {
    /// The file to compress.
    pub file: PathBuf,

    /// The line budget the summary is hard-capped to (overflow is listed in the
    /// drop report, never silently discarded).
    #[arg(long, default_value_t = 10)]
    pub lines: usize,

    /// Emit the `Preprocessed` result (output + drop report) as JSON.
    #[arg(long)]
    pub json: bool,
}

/// Options for `kopitiam preprocess triage`.
#[derive(Args, Debug)]
pub struct TriageArgs {
    /// What the candidates are being filtered for relevance to.
    pub query: String,

    /// The candidate snippets (e.g. grep hits) to filter.
    pub candidates: Vec<String>,

    /// Emit the `Preprocessed` result (kept subset + drop report) as JSON.
    #[arg(long)]
    pub json: bool,
}

/// Runs `kopitiam preprocess ...`.
pub fn run(args: PreprocessArgs) -> Result<()> {
    let selected = select_adapter();
    eprintln!("{}", selected.notice());
    let adapter = selected.adapter();

    match args.action {
        PreprocessAction::Summarize(a) => {
            let text = std::fs::read_to_string(&a.file)
                .with_context(|| format!("reading {}", a.file.display()))?;
            let result = summarize(adapter, &text, a.lines)?;
            if a.json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                println!("{}", result.output);
                print!("{}", render_report(&result.report));
            }
        }
        PreprocessAction::Triage(a) => {
            let result = triage(adapter, &a.query, &a.candidates)?;
            if a.json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                for kept in &result.output {
                    println!("{kept}");
                }
                print!("{}", render_report(&result.report));
            }
        }
    }
    Ok(())
}

/// Renders a [`DropReport`] as a compact, always-shown footer so the caller sees
/// what was dropped and any honesty caveats (never a silent filter).
fn render_report(report: &DropReport) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "-- {}: {} in, {} kept, {} dropped --",
        report.step,
        report.input_units,
        report.kept_units,
        report.dropped.len(),
    );
    for dropped in &report.dropped {
        let _ = writeln!(out, "  dropped: {dropped}");
    }
    for note in &report.notes {
        let _ = writeln!(out, "  note: {note}");
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
        #[command(subcommand)]
        command: PreprocessAction,
    }

    #[test]
    fn parses_summarize_and_triage() {
        let s = TestCli::try_parse_from(["p", "summarize", "f.txt", "--lines", "3"]).unwrap();
        assert!(matches!(s.command, PreprocessAction::Summarize(a) if a.lines == 3 && a.file == *std::path::Path::new("f.txt")));

        let t = TestCli::try_parse_from(["p", "triage", "digital twin", "hit a", "hit b"]).unwrap();
        assert!(matches!(t.command, PreprocessAction::Triage(a) if a.query == "digital twin" && a.candidates.len() == 2));
    }

    #[test]
    fn render_report_shows_counts_dropped_and_notes() {
        // Under echo, summarize hard-caps and lists the overflow verbatim.
        let result = summarize(&EchoAdapter, "l1\nl2\nl3\nl4\nl5\n", 2).unwrap();
        let text = render_report(&result.report);
        assert!(text.contains("summarize"));
        assert!(text.contains("dropped: l3"));
        // The echo pass-through caveat is surfaced.
        assert!(text.contains("pass-through"));
    }
}

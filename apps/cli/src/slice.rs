//! The `slice` subcommand: budget-aware range read — token-max follow-up P2.
//!
//! This is the missing *last* step of the `tokens → outline → refs → READ` loop
//! (`docs/token-max-followups.md` P2). After `outline`/`refs` hand back
//! `file:line` coordinates, an agent needs to read **only those lines**, not the
//! whole file (and not by shelling `sed`, which leaks platform-specific quoting
//! and mixed path separators). `slice` prints an inclusive, 1-based line range
//! and reports its estimated token cost, so the same
//! [`kopitiam_tokenizer::estimate_tokens`] number that *chose* the read also
//! *bounds* it.
//!
//! Two modes:
//!
//! * `kopitiam slice <file> <A-B>` — print lines `A..=B`. The range accepts
//!   `A-B` (inclusive), a bare `A` (one line), `A-` (A to EOF), and `-B`
//!   (start to B).
//! * `kopitiam slice <file> --grep <pat>` — grep-then-slice fused: find every
//!   line containing `pat` (a literal substring — deterministic, no regex dep)
//!   and print each match's neighbourhood (`± --context` lines) as a slice.
//!   Overlapping neighbourhoods are merged so no line is printed twice. An
//!   agent greps and reads only the hit windows in ONE call.
//!
//! Every slice carries a `(~N tokens)` cost; `--json` emits the machine form
//! (`{ file, slices: [{ start, end, tokens, lines }], total_tokens }`). Paths
//! are forward-slash normalised (CLAUDE.md "Cross-platform paths").

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Args;
use kopitiam_tokenizer::estimate_tokens;
use serde::Serialize;

/// Options for `kopitiam slice`.
#[derive(Args, Debug)]
pub struct SliceArgs {
    /// The file to read lines from.
    pub file: PathBuf,

    /// Line range to print: `A-B` (inclusive, 1-based), a bare `A` (one line),
    /// `A-` (A to end of file), or `-B` (start to B). Required unless `--grep`
    /// is given; when both are present it *bounds* the grep search window.
    pub range: Option<String>,

    /// Grep-then-slice: print each line containing this literal substring, with
    /// `--context` lines of neighbourhood around it, merged where they overlap.
    #[arg(long, value_name = "PAT")]
    pub grep: Option<String>,

    /// Lines of context to include on each side of a `--grep` match.
    #[arg(long, default_value_t = 3)]
    pub context: usize,

    /// Emit machine-readable JSON instead of the human slices.
    #[arg(long)]
    pub json: bool,
}

/// One contiguous slice of a file: its inclusive 1-based line bounds, the lines
/// themselves, and their estimated token cost.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct Slice {
    /// 1-based first line, inclusive.
    start: usize,
    /// 1-based last line, inclusive.
    end: usize,
    /// Estimated BPE token cost of this slice's text.
    tokens: usize,
    /// The slice's lines, in order (no trailing newline per entry).
    lines: Vec<String>,
}

/// The whole `slice` result: the file, each slice, and the summed token cost.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct SliceReport {
    file: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    grep: Option<String>,
    slices: Vec<Slice>,
    total_tokens: usize,
}

/// Runs `kopitiam slice`.
pub fn run(args: SliceArgs) -> Result<()> {
    let text = std::fs::read_to_string(&args.file)
        .with_context(|| format!("reading {}", args.file.display()))?;
    let lines: Vec<&str> = text.lines().collect();
    let total = lines.len();

    // Optional range: bounds the grep search, or (without --grep) is the slice.
    let bound = match &args.range {
        Some(spec) => Some(parse_range(spec, total)?),
        None => None,
    };

    let windows: Vec<(usize, usize)> = if let Some(pat) = &args.grep {
        let (lo, hi) = bound.unwrap_or((1, total.max(1)));
        grep_windows(&lines, pat, args.context, lo, hi)
    } else {
        match bound {
            Some(r) => vec![r],
            None => bail!(
                "a line range is required (e.g. `10-25`, `10`, `10-`) unless --grep is given"
            ),
        }
    };

    let report = build_report(&args.file, args.grep.as_deref(), &lines, &windows);

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_human(&report);
    }
    Ok(())
}

/// Assembles the [`SliceReport`] from resolved 1-based inclusive `windows` over
/// `lines`. Pure given its inputs, so it is unit-tested without filesystem I/O.
fn build_report(file: &Path, grep: Option<&str>, lines: &[&str], windows: &[(usize, usize)]) -> SliceReport {
    let mut slices = Vec::with_capacity(windows.len());
    let mut total_tokens = 0usize;
    for &(start, end) in windows {
        // `windows` are already clamped to `1..=lines.len()`; guard anyway so a
        // stray bound can never panic the slice index.
        if start == 0 || start > end || start > lines.len() {
            continue;
        }
        let end = end.min(lines.len());
        let body: Vec<String> = lines[start - 1..end].iter().map(|s| (*s).to_string()).collect();
        let tokens = estimate_tokens(&slice_text(&body));
        total_tokens += tokens;
        slices.push(Slice { start, end, tokens, lines: body });
    }
    SliceReport {
        file: to_slash(file),
        grep: grep.map(str::to_string),
        slices,
        total_tokens,
    }
}

/// The text a slice's token cost is measured over: its lines rejoined with `\n`
/// and a trailing newline, so the estimate matches how the lines read as a file.
fn slice_text(lines: &[String]) -> String {
    let mut s = lines.join("\n");
    s.push('\n');
    s
}

/// Parses a range spec against a file of `total` lines, returning inclusive,
/// 1-based, clamped `(start, end)` bounds. Accepts `A-B`, bare `A`, `A-` (to
/// EOF), and `-B` (from line 1). Rejects zero, a reversed range, non-numeric
/// input, and a start past the end of the file.
fn parse_range(spec: &str, total: usize) -> Result<(usize, usize)> {
    let spec = spec.trim();
    let eof = total.max(1);

    let (start, end) = match spec.split_once('-') {
        Some((lo, hi)) => {
            let lo = if lo.trim().is_empty() { 1 } else { parse_one(lo, spec)? };
            let hi = if hi.trim().is_empty() { eof } else { parse_one(hi, spec)? };
            (lo, hi)
        }
        None => {
            let n = parse_one(spec, spec)?;
            (n, n)
        }
    };

    if end < start {
        bail!("invalid range {spec:?}: end line {end} is before start line {start}");
    }
    if start > total {
        bail!("invalid range {spec:?}: start line {start} is past end of file ({total} lines)");
    }
    // Clamp the end to EOF so `10-99999` (or `10-`) reads to the last line.
    Ok((start, end.min(eof)))
}

/// Parses one 1-based line number, rejecting zero and non-numeric input with an
/// actionable message that names the whole `spec`.
fn parse_one(s: &str, spec: &str) -> Result<usize> {
    let n: usize = s.trim().parse().map_err(|_| {
        anyhow::anyhow!("invalid range {spec:?}: expected `A-B`, `A`, `A-`, or `-B` with 1-based line numbers")
    })?;
    if n == 0 {
        bail!("invalid range {spec:?}: line numbers are 1-based, 0 is not a line");
    }
    Ok(n)
}

/// Finds every line in `lines[lo-1..=hi-1]` (1-based, inclusive bounds)
/// containing the literal substring `pat`, and returns each match's `± context`
/// neighbourhood as merged, inclusive, 1-based `(start, end)` windows. Adjacent
/// or overlapping neighbourhoods are coalesced so a line is never emitted twice.
fn grep_windows(lines: &[&str], pat: &str, context: usize, lo: usize, hi: usize) -> Vec<(usize, usize)> {
    let total = lines.len();
    if pat.is_empty() || total == 0 {
        return Vec::new();
    }
    let lo = lo.max(1);
    let hi = hi.min(total);

    // Collect each match's context window, clamped to the file, then merge.
    let mut windows: Vec<(usize, usize)> = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let line_no = i + 1; // 1-based
        if line_no < lo || line_no > hi {
            continue;
        }
        if line.contains(pat) {
            let start = line_no.saturating_sub(context).max(1);
            let end = (line_no + context).min(total);
            windows.push((start, end));
        }
    }
    merge_windows(windows)
}

/// Merges a list of inclusive `(start, end)` windows into non-overlapping,
/// ascending windows. Two windows merge when they overlap OR are directly
/// adjacent (`a.end + 1 >= b.start`), so back-to-back neighbourhoods read as one
/// continuous slice rather than two abutting ones.
fn merge_windows(mut windows: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    windows.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(windows.len());
    for (start, end) in windows {
        match merged.last_mut() {
            Some(last) if start <= last.1 + 1 => last.1 = last.1.max(end),
            _ => merged.push((start, end)),
        }
    }
    merged
}

/// Prints the human slices: a `file:start-end (~N tokens)` header per slice,
/// then its lines with a 1-based line-number gutter (so the printed coordinates
/// feed straight back into the next command).
fn print_human(report: &SliceReport) {
    for slice in &report.slices {
        println!("{}:{}-{} (~{} tokens)", report.file, slice.start, slice.end, slice.tokens);
        for (offset, line) in slice.lines.iter().enumerate() {
            println!("{:>6}| {}", slice.start + offset, line);
        }
    }
    if report.slices.is_empty() {
        match &report.grep {
            Some(pat) => println!("no lines containing {pat:?} in {}", report.file),
            None => println!("no lines in range for {}", report.file),
        }
    } else if report.slices.len() != 1 {
        println!("Total: ~{} tokens across {} slices", report.total_tokens, report.slices.len());
    }
}

/// Normalises a path to forward slashes for stable cross-platform output
/// (CLAUDE.md "Cross-platform paths"; same rule as `tokens::to_slash`).
fn to_slash(p: &Path) -> String {
    p.display().to_string().replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        args: SliceArgs,
    }

    // ---- arg parsing --------------------------------------------------------

    #[test]
    fn parses_file_and_range() {
        let cli = TestCli::try_parse_from(["t", "src/a.rs", "10-25"]).unwrap();
        assert_eq!(cli.args.file, PathBuf::from("src/a.rs"));
        assert_eq!(cli.args.range.as_deref(), Some("10-25"));
        assert!(cli.args.grep.is_none());
        assert_eq!(cli.args.context, 3);
        assert!(!cli.args.json);
    }

    #[test]
    fn parses_grep_context_and_json() {
        let cli = TestCli::try_parse_from(["t", "src/a.rs", "--grep", "Command", "--context", "5", "--json"]).unwrap();
        assert_eq!(cli.args.grep.as_deref(), Some("Command"));
        assert_eq!(cli.args.context, 5);
        assert!(cli.args.json);
        // The range is optional when grepping.
        assert!(cli.args.range.is_none());
    }

    // ---- range parsing ------------------------------------------------------

    #[test]
    fn parse_range_handles_all_forms() {
        // A-B inclusive.
        assert_eq!(parse_range("10-25", 100).unwrap(), (10, 25));
        // Bare A is a single line.
        assert_eq!(parse_range("7", 100).unwrap(), (7, 7));
        // A- reads to EOF (clamped to the last line).
        assert_eq!(parse_range("90-", 100).unwrap(), (90, 100));
        // -B reads from line 1.
        assert_eq!(parse_range("-5", 100).unwrap(), (1, 5));
        // Whitespace tolerated.
        assert_eq!(parse_range(" 3 - 4 ", 100).unwrap(), (3, 4));
        // An end past EOF clamps to EOF, not an error.
        assert_eq!(parse_range("95-9999", 100).unwrap(), (95, 100));
    }

    #[test]
    fn parse_range_rejects_bad_input() {
        // Zero is not a 1-based line.
        assert!(parse_range("0", 100).is_err());
        assert!(parse_range("0-5", 100).is_err());
        // Reversed range.
        assert!(parse_range("9-2", 100).is_err());
        // A start past EOF is an error (nothing to read).
        assert!(parse_range("500-600", 100).is_err());
        // Garbage.
        assert!(parse_range("abc", 100).is_err());
        assert!(parse_range("1-b", 100).is_err());
    }

    // ---- grep neighbourhood extraction --------------------------------------

    fn sample() -> Vec<&'static str> {
        // 12 lines; "hit" appears on lines 3, 4 (adjacent) and 10 (isolated).
        vec![
            "one", "two", "hit three", "hit four", "five", "six", "seven", "eight", "nine",
            "ten hit", "eleven", "twelve",
        ]
    }

    #[test]
    fn grep_windows_merges_overlapping_neighbourhoods() {
        // context 1: matches at 3 and 4 -> [2,4] and [3,5] merge to [2,5];
        // the match at 10 -> [9,11], isolated.
        let windows = grep_windows(&sample(), "hit", 1, 1, 12);
        assert_eq!(windows, vec![(2, 5), (9, 11)]);
    }

    #[test]
    fn grep_windows_clamps_to_file_bounds() {
        // context 5 from a match at line 3 clamps the start to line 1, and a
        // match near EOF clamps the end to the last line.
        let windows = grep_windows(&sample(), "hit", 5, 1, 12);
        // 3,4 -> [1,9]/[1,9]; 10 -> [5,12]; overlap -> single [1,12].
        assert_eq!(windows, vec![(1, 12)]);
    }

    #[test]
    fn grep_windows_respects_the_bounding_range() {
        // Bound the search to lines 1..=5, so the match on line 10 is ignored.
        let windows = grep_windows(&sample(), "hit", 1, 1, 5);
        assert_eq!(windows, vec![(2, 5)]);
    }

    #[test]
    fn grep_windows_empty_pattern_or_no_match() {
        assert!(grep_windows(&sample(), "", 2, 1, 12).is_empty());
        assert!(grep_windows(&sample(), "nowhere", 2, 1, 12).is_empty());
    }

    // ---- merge_windows ------------------------------------------------------

    #[test]
    fn merge_windows_coalesces_overlapping_and_adjacent() {
        // Overlap, adjacency (3 touches 4 via +1), and a disjoint window.
        let merged = merge_windows(vec![(1, 3), (2, 4), (5, 6), (10, 12)]);
        assert_eq!(merged, vec![(1, 6), (10, 12)]);
    }

    // ---- report construction ------------------------------------------------

    #[test]
    fn build_report_slices_and_sums_tokens() {
        let lines = sample();
        let report = build_report(Path::new("a.rs"), None, &lines, &[(1, 3), (10, 12)]);
        assert_eq!(report.slices.len(), 2);
        assert_eq!(report.slices[0].start, 1);
        assert_eq!(report.slices[0].end, 3);
        assert_eq!(report.slices[0].lines, vec!["one", "two", "hit three"]);
        // Total is the sum of the per-slice token estimates.
        let t0 = estimate_tokens("one\ntwo\nhit three\n");
        let t1 = estimate_tokens("ten hit\neleven\ntwelve\n");
        assert_eq!(report.slices[0].tokens, t0);
        assert_eq!(report.total_tokens, t0 + t1);
    }

    #[test]
    fn build_report_normalises_paths_and_carries_grep() {
        let lines = sample();
        let report = build_report(Path::new("src\\a.rs"), Some("hit"), &lines, &[(3, 4)]);
        assert_eq!(report.file, "src/a.rs", "backslashes are normalised");
        assert_eq!(report.grep.as_deref(), Some("hit"));
    }

    #[test]
    fn json_shape_exposes_slices_and_total() {
        let lines = sample();
        let report = build_report(Path::new("a.rs"), None, &lines, &[(1, 2)]);
        let json: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&report).unwrap()).unwrap();
        assert_eq!(json["file"], "a.rs");
        assert!(json["total_tokens"].as_u64().is_some());
        let slices = json["slices"].as_array().unwrap();
        assert_eq!(slices[0]["start"], 1);
        assert_eq!(slices[0]["end"], 2);
        assert!(slices[0]["tokens"].as_u64().is_some());
        assert_eq!(slices[0]["lines"][0], "one");
    }
}

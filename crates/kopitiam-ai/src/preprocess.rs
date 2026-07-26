//! High-volume, low-judgment preprocessing routed to the **local** model, so
//! the cloud model never sees the raw volume (token-max card **II-6**;
//! `kopitiam_token_max.md` §0.5: *local model absorbs volume, cloud model
//! spends judgment*).
//!
//! Everything here takes an injected [`ModelAdapter`] and calls it directly —
//! the caller is expected to hand in the **local** adapter it already resolved
//! (in the CLI, `select_adapter().adapter()`), so the work costs **zero cloud
//! tokens** by construction. Because the surface is `&dyn ModelAdapter`, tests
//! drive it with the deterministic [`crate::EchoAdapter`] — no weights, no
//! network, no model download.
//!
//! # Be honest about capability (this is the card's central requirement)
//!
//! The default local model is SmolLM2-360M-Instruct
//! (`kopitiam_models::DEFAULT_MODEL_ID`; the CLI's `select_adapter` resolves
//! it). **A ~360M model cannot be trusted with judgment.** These helpers are for
//! *filtering* and *compression* only — situations where a wrong call is
//! **recoverable**: a summary is re-derivable from the untouched source, and a
//! false negative in triage only means the cloud model reads one extra snippet.
//! None of them is ever the **final authority on correctness** (§321: *always
//! report what it dropped; never the final authority*). Two mechanisms enforce
//! that contract:
//!
//! 1. **Every result carries a [`DropReport`]** recording what was removed,
//!    filtered, or set aside — verbatim where it can be enumerated, so the
//!    caller can always recover it. [`DropReport::AUTHORITATIVE`] is a
//!    compile-time `false`: preprocessing output is a filter, not a verdict.
//! 2. **The echo stub is detected and loudly annotated.** When the injected
//!    adapter is [`crate::EchoAdapter`] (no `.gguf` on disk), there is *no real
//!    preprocessing* — the output is a pass-through of the input. Rather than
//!    pretend it filtered anything, each helper stamps [`ECHO_PASSTHROUGH_NOTE`]
//!    into the report so a caller can `refuse`/annotate instead of trusting a
//!    stubbed result (the §281 caveat, enforced at the library layer via
//!    [`ModelAdapter::name`] rather than the CLI's `is_local`).
//!
//! # Measuring the cloud-token saving
//!
//! The saving is, by construction, *everything routed through here*: the raw
//! input volume is sent to the **local** adapter and never to the cloud; only
//! the compact [`Preprocessed::output`] is handed onward. So:
//!
//! ```text
//! cloud tokens saved  ≈  tokens(raw input routed to local)
//!                        − tokens(Preprocessed.output handed to the cloud)
//! ```
//!
//! Each [`DropReport`] exposes [`DropReport::input_units`] and
//! [`DropReport::kept_units`] as a units-level proxy (lines / candidates /
//! diagnostics in vs. out); pair those with the token-accounting command
//! (token-max card **II-7**, `tokens <path>`) run over the input text and the
//! `output` to turn the proxy into a token figure. The unit tests assert the
//! reduction directly (e.g. summarize takes 5 lines to 2 and reports the 3 it
//! dropped), which is the deterministic, network-free measurement the card asks
//! each Part II task to state for itself.

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::{CompletionRequest, Message, ModelAdapter};

/// The name [`crate::EchoAdapter::name`] reports. Used to detect the "no real
/// model" pass-through case so results can be annotated honestly.
const ECHO_ADAPTER_NAME: &str = "echo";

/// Stamped into a [`DropReport`] whenever preprocessing ran against the echo
/// stub instead of a real local model: the "output" is the input echoed back,
/// **not** a filtered or compressed result, and must not be trusted as one.
pub const ECHO_PASSTHROUGH_NOTE: &str = "adapter is the echo stub (no local .gguf) — output is a \
     pass-through of the input, NOT model preprocessing; do not trust it as filtered/compressed.";

/// A preprocessing result: the compact `output`, plus the [`DropReport`] of
/// what producing it removed or set aside.
///
/// Generic over the output shape (a `String` summary, a `Vec<String>` kept
/// subset, [`DiagnosticBuckets`], ...). Serializable so a machine consumer gets
/// structure, not prose (§0.2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Preprocessed<T> {
    /// The compact result to hand onward (e.g. to the cloud model).
    pub output: T,
    /// What was dropped/filtered to produce `output`. Never omit this when
    /// forwarding a result — it is the recoverability guarantee.
    pub report: DropReport,
}

impl<T> Preprocessed<T> {
    fn new(output: T, report: DropReport) -> Self {
        Self { output, report }
    }
}

/// A record of what a preprocessing step removed, filtered, or could not judge.
///
/// The honesty contract of this module (§321): a result is never handed on
/// without one of these. `dropped` holds the removed units **verbatim** wherever
/// they can be enumerated (exact duplicates, hard-truncated overflow, the
/// filtered-out candidates), so a false negative is always recoverable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DropReport {
    /// Which helper produced this (`"summarize"`, `"triage"`,
    /// `"classify_diagnostics"`, `"draft_commit_message"`).
    pub step: String,
    /// Count of input units seen (source lines / candidate snippets / distinct
    /// diagnostics), before this step reduced them.
    pub input_units: usize,
    /// Count of units represented in the result after the step.
    pub kept_units: usize,
    /// The dropped/filtered units, verbatim and in input order, so the caller
    /// can recover any the model wrongly discarded. Empty when nothing was
    /// enumerably dropped (e.g. lossy compression that can't be itemized — see
    /// `notes` for that case).
    pub dropped: Vec<String>,
    /// Honest caveats: the lossiness of the step, whether the model's output
    /// was trusted or a deterministic fallback was used, and (crucially) the
    /// [`ECHO_PASSTHROUGH_NOTE`] when there was no real model.
    pub notes: Vec<String>,
}

impl DropReport {
    /// Preprocessing in this module is **never the final authority on
    /// correctness** (§321). A compile-time constant so callers can encode "do
    /// not gate on this as a verdict" in their own logic and tests.
    pub const AUTHORITATIVE: bool = false;

    fn new(step: &'static str, input_units: usize) -> Self {
        Self {
            step: step.to_string(),
            input_units,
            kept_units: 0,
            dropped: Vec::new(),
            notes: Vec::new(),
        }
    }

    /// Number of units this step dropped (length of [`Self::dropped`]).
    #[must_use]
    pub fn dropped_count(&self) -> usize {
        self.dropped.len()
    }

    fn note(&mut self, note: impl Into<String>) {
        self.notes.push(note.into());
    }
}

/// `true` when the injected adapter is the deterministic echo stub, i.e. there
/// is no real local model and any "preprocessing" is a pass-through.
fn is_echo(adapter: &dyn ModelAdapter) -> bool {
    adapter.name() == ECHO_ADAPTER_NAME
}

/// Route one `(system, user)` exchange through the adapter and return the
/// reply's content. This is the single point volume is handed to the *local*
/// model — the caller having passed in the local adapter is what makes it cost
/// zero cloud tokens.
fn route(adapter: &dyn ModelAdapter, system: &str, user: &str) -> Result<String> {
    let request =
        CompletionRequest::new([Message::system(system), Message::user(user)]);
    Ok(adapter.complete(&request)?.content)
}

/// Compress `text` to at most `target_lines` lines by routing it through the
/// local model, then hard-capping the reply at `target_lines`.
///
/// **Compression, not judgment.** The full `text` remains available to the
/// caller; the summary is a lossy, non-authoritative view of it. The model does
/// the semantic compression (which cannot be enumerated line-by-line, hence the
/// lossiness note); this function additionally guarantees the line budget by
/// truncating any overflow, and those overflow lines *are* enumerated in
/// [`DropReport::dropped`] so nothing vanishes silently.
///
/// With the echo stub the reply is `text` verbatim, so you get the first
/// `target_lines` lines of the source and a report listing the rest — plus
/// [`ECHO_PASSTHROUGH_NOTE`], because no real compression happened.
pub fn summarize(
    adapter: &dyn ModelAdapter,
    text: &str,
    target_lines: usize,
) -> Result<Preprocessed<String>> {
    let source_lines = text.lines().count();
    let mut report = DropReport::new("summarize", source_lines);

    let raw = route(
        adapter,
        &format!(
            "Compress the text the user sends to at most {target_lines} lines. Keep the key \
             facts and drop filler. Output only the compressed lines, nothing else."
        ),
        text,
    )?;

    let reply_lines: Vec<&str> = raw.lines().collect();
    let kept: Vec<&str> = reply_lines.iter().take(target_lines).copied().collect();
    let overflow: Vec<String> =
        reply_lines.iter().skip(target_lines).map(|l| (*l).to_string()).collect();

    report.kept_units = kept.len();
    report.dropped = overflow;
    report.note(
        "summary is a lossy, non-authoritative compression — the source text is unchanged and \
         re-derivable; do not treat the summary as the source of truth.",
    );
    if report.dropped_count() > 0 {
        report.note(format!(
            "hard-capped at {target_lines} lines; {} overflow line(s) dropped (listed verbatim, \
             recoverable).",
            report.dropped_count()
        ));
    }
    if is_echo(adapter) {
        report.note(ECHO_PASSTHROUGH_NOTE);
    }

    Ok(Preprocessed::new(kept.join("\n"), report))
}

/// Filter `candidates` (e.g. grep-hit snippets) down to the subset plausibly
/// relevant to `query`, by asking the local model which to keep.
///
/// **Filtering where a false negative is recoverable**, never a correctness
/// verdict. The model is asked to keep anything it is unsure about, and this
/// function is deliberately **conservative**: if the reply names no valid
/// candidate (unparseable, or the echo stub, or a model that produced garbage),
/// it keeps **all** candidates rather than silently dropping any — a false
/// *positive* only costs the cloud model one extra snippet to read, whereas a
/// false *negative* could hide the one hit that mattered. Every dropped snippet
/// is listed verbatim in [`DropReport::dropped`].
///
/// The reply is parsed for candidate indices (`0`-based, as presented). Indices
/// outside range are ignored; the kept set is the parsed indices intersected
/// with the candidate range. With the echo stub the numbered listing is echoed
/// back, so every index parses and all candidates are kept (the safe default),
/// annotated with [`ECHO_PASSTHROUGH_NOTE`].
pub fn triage(
    adapter: &dyn ModelAdapter,
    query: &str,
    candidates: &[String],
) -> Result<Preprocessed<Vec<String>>> {
    let mut report = DropReport::new("triage", candidates.len());

    if candidates.is_empty() {
        report.note("no candidates to triage.");
        return Ok(Preprocessed::new(Vec::new(), report));
    }

    let mut listing = format!("QUERY: {query}\n\nCANDIDATES:\n");
    for (i, c) in candidates.iter().enumerate() {
        listing.push_str(&format!("{i}. {c}\n"));
    }

    let raw = route(
        adapter,
        "You are filtering search hits. Reply with only the numbers of the candidates that could \
         plausibly relate to the QUERY, comma-separated. When unsure, keep it. Reply nothing else.",
        &listing,
    )?;

    let mut selected = parse_indices(&raw, candidates.len());
    let failsafe = selected.is_empty();
    if failsafe {
        // Conservative: the model gave us nothing usable, so keep everything
        // rather than drop blindly. A recoverable over-keep beats a lossy
        // under-keep.
        selected = (0..candidates.len()).collect();
        report.note(
            "model reply named no valid candidate — kept ALL conservatively (a false negative is \
             not recoverable here, so we never drop on an unusable reply).",
        );
    }

    let mut kept = Vec::new();
    for (i, c) in candidates.iter().enumerate() {
        if selected.contains(&i) {
            kept.push(c.clone());
        } else {
            report.dropped.push(c.clone());
        }
    }
    report.kept_units = kept.len();
    if report.dropped_count() > 0 {
        report.note(format!(
            "filtered out {} of {} candidate(s) as implausible (listed verbatim, recoverable — \
             re-check if a false negative is suspected).",
            report.dropped_count(),
            candidates.len()
        ));
    }
    if is_echo(adapter) && !failsafe {
        report.note(ECHO_PASSTHROUGH_NOTE);
    }

    Ok(Preprocessed::new(kept, report))
}

/// One severity bucket of diagnostics: a `label` and the distinct diagnostics
/// that fell under it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bucket {
    /// The severity label (`"error"`, `"warning"`, `"note"`, `"help"`,
    /// `"other"`).
    pub label: String,
    /// The distinct diagnostics in this bucket, in input order.
    pub items: Vec<String>,
}

/// The result of [`classify_diagnostics`]: severity buckets over the
/// **deduplicated** diagnostics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticBuckets {
    /// Buckets in a stable severity order, each non-empty.
    pub buckets: Vec<Bucket>,
}

/// Bucket raw compiler-diagnostic text by severity, after removing exact
/// duplicates.
///
/// **Deterministic structure + local-model gloss.** The two honest facts here
/// are that (a) the biggest real reduction is *deduplication* — one bad type
/// can emit the same diagnostic dozens of times — and (b) severity is
/// **deterministically parseable** from rustc's prefixes, so per §0.3
/// (deterministic tool over model call) this function does **not** let a 0.5B
/// model decide the bucketing. It removes exact-duplicate diagnostics (each
/// removed copy listed verbatim in [`DropReport::dropped`]) and buckets the
/// distinct remainder by prefix.
///
/// The local model is still *routed the deduplicated volume* — that is the
/// token-max point, the raw diagnostics never reach the cloud — and its reply is
/// recorded as a **non-authoritative gloss** in [`DropReport::notes`], never
/// used to place a diagnostic. With the echo stub the gloss is the input echoed
/// back and [`ECHO_PASSTHROUGH_NOTE`] is added.
pub fn classify_diagnostics(
    adapter: &dyn ModelAdapter,
    raw: &str,
) -> Result<Preprocessed<DiagnosticBuckets>> {
    let blocks = split_diagnostics(raw);
    let mut report = DropReport::new("classify_diagnostics", blocks.len());

    // Deterministic dedup — the real reduction. Removed copies are recoverable
    // (they are byte-identical to a kept one) but still reported.
    let mut distinct: Vec<String> = Vec::new();
    for b in blocks {
        if distinct.contains(&b) {
            report.dropped.push(b);
        } else {
            distinct.push(b);
        }
    }
    report.kept_units = distinct.len();
    if report.dropped_count() > 0 {
        report.note(format!(
            "removed {} exact-duplicate diagnostic(s) (recoverable; identical to a kept entry).",
            report.dropped_count()
        ));
    }

    // Route the deduplicated volume to the local model (zero cloud tokens) for a
    // human-facing gloss only — the buckets themselves stay deterministic.
    if !distinct.is_empty() {
        let gloss = route(
            adapter,
            "Summarize the compiler diagnostics the user sends in one short sentence naming the \
             likely root theme. This is advisory only.",
            &distinct.join("\n\n"),
        )?;
        let gloss = first_line(&gloss);
        if !gloss.is_empty() {
            report.note(format!("local-model gloss (non-authoritative): {gloss}"));
        }
    }
    report.note(
        "buckets assigned deterministically by severity prefix (§0.3), NOT by model judgment.",
    );
    if is_echo(adapter) {
        report.note(ECHO_PASSTHROUGH_NOTE);
    }

    Ok(Preprocessed::new(bucket_by_severity(&distinct), report))
}

/// Draft a commit-message subject line for `diff` by routing the whole diff
/// through the local model (optional helper; card II-6).
///
/// **A draft, never authoritative.** The message is a lossy summary the human
/// (or cloud model) must review and rewrite — the diff is not enumerably
/// reduced here, so [`DropReport::dropped`] is empty and the report carries the
/// "must be reviewed" caveat instead. The point is purely token-max: the full
/// diff goes to the *local* model, and only a one-line draft is handed onward.
/// With the echo stub the draft is the first diff line echoed back, plus
/// [`ECHO_PASSTHROUGH_NOTE`].
pub fn draft_commit_message(
    adapter: &dyn ModelAdapter,
    diff: &str,
) -> Result<Preprocessed<String>> {
    let diff_lines = diff.lines().count();
    let mut report = DropReport::new("draft_commit_message", diff_lines);

    let raw = route(
        adapter,
        "Write ONE concise commit subject line (imperative mood, under 72 chars) summarizing the \
         diff the user sends. Output only that line.",
        diff,
    )?;
    let subject = first_line(&raw);

    report.kept_units = usize::from(!subject.is_empty());
    report.note(
        "draft only — a lossy, non-authoritative summary of the diff; review and rewrite before \
         committing. The full diff is unchanged.",
    );
    if is_echo(adapter) {
        report.note(ECHO_PASSTHROUGH_NOTE);
    }

    Ok(Preprocessed::new(subject, report))
}

/// Split raw diagnostic text into logical blocks: on blank lines when the text
/// is block-structured, else one block per non-empty line.
fn split_diagnostics(raw: &str) -> Vec<String> {
    let by_block: Vec<String> = raw
        .split("\n\n")
        .map(|b| b.trim().to_string())
        .filter(|b| !b.is_empty())
        .collect();
    if by_block.len() > 1 {
        return by_block;
    }
    raw.lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

/// Deterministic severity of a diagnostic, from the prefix of its first line.
fn severity_of(block: &str) -> &'static str {
    let head = block.lines().next().unwrap_or("").trim_start();
    if head.starts_with("error") {
        "error"
    } else if head.starts_with("warning") {
        "warning"
    } else if head.starts_with("note") {
        "note"
    } else if head.starts_with("help") {
        "help"
    } else {
        "other"
    }
}

/// Group distinct diagnostics into buckets in a stable severity order,
/// preserving input order within each bucket and emitting only non-empty
/// buckets.
fn bucket_by_severity(distinct: &[String]) -> DiagnosticBuckets {
    const ORDER: [&str; 5] = ["error", "warning", "note", "help", "other"];
    let mut buckets = Vec::new();
    for label in ORDER {
        let items: Vec<String> = distinct
            .iter()
            .filter(|d| severity_of(d) == label)
            .cloned()
            .collect();
        if !items.is_empty() {
            buckets.push(Bucket { label: label.to_string(), items });
        }
    }
    DiagnosticBuckets { buckets }
}

/// The first non-empty, trimmed line of `text` (empty string if none).
fn first_line(text: &str) -> String {
    text.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("").to_string()
}

/// Extract distinct in-range `0`-based indices from a model reply, in ascending
/// order. Any digit run that parses to `< max` counts; everything else (labels,
/// punctuation, out-of-range numbers) is ignored.
fn parse_indices(reply: &str, max: usize) -> Vec<usize> {
    let mut found: Vec<usize> = Vec::new();
    let mut digits = String::new();
    let flush = |digits: &mut String, found: &mut Vec<usize>| {
        if let Ok(n) = digits.parse::<usize>()
            && n < max
            && !found.contains(&n)
        {
            found.push(n);
        }
        digits.clear();
    };
    for ch in reply.chars() {
        if ch.is_ascii_digit() {
            digits.push(ch);
        } else if !digits.is_empty() {
            flush(&mut digits, &mut found);
        }
    }
    if !digits.is_empty() {
        flush(&mut digits, &mut found);
    }
    found.sort_unstable();
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CompletionResponse, EchoAdapter};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A non-echo adapter that returns a fixed reply and *counts* calls, so a
    /// test can prove the preprocessing genuinely routed through the adapter
    /// (not just string-processed locally) and consumed its reply.
    struct FixedAdapter {
        reply: String,
        calls: AtomicUsize,
    }
    impl FixedAdapter {
        fn new(reply: &str) -> Self {
            Self { reply: reply.to_string(), calls: AtomicUsize::new(0) }
        }
    }
    impl ModelAdapter for FixedAdapter {
        fn name(&self) -> &str {
            "fixed-test"
        }
        fn complete(&self, _request: &CompletionRequest) -> Result<CompletionResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(CompletionResponse { content: self.reply.clone(), model: "fixed-test".into() })
        }
    }

    /// Compile-time guarantee that preprocessing output is never treated as the
    /// final authority on correctness (§321).
    const _: () = assert!(!DropReport::AUTHORITATIVE);

    /// The headline EchoAdapter measurement: summarize routes the text through
    /// the adapter and hard-caps the reply, and the drop-report enumerates
    /// exactly the lines it removed. Because the echo reply *is* the input, the
    /// output being the first `target` source lines proves the adapter's reply
    /// flowed into the result (routing), and the dropped list proves the
    /// drop-report (§321). This is the deterministic, network-free token-saving
    /// measurement: 5 lines in, 2 out, 3 reported dropped.
    #[test]
    fn summarize_routes_through_echo_and_reports_every_dropped_line() {
        let text = "line one\nline two\nline three\nline four\nline five";
        let pre = summarize(&EchoAdapter, text, 2).unwrap();

        assert_eq!(pre.output, "line one\nline two");
        assert_eq!(pre.report.input_units, 5);
        assert_eq!(pre.report.kept_units, 2);
        assert_eq!(pre.report.dropped, vec!["line three", "line four", "line five"]);
        // Honest about being a stub: the echo pass-through note must be present.
        assert!(pre.report.notes.iter().any(|n| n == ECHO_PASSTHROUGH_NOTE));
    }

    #[test]
    fn summarize_drops_nothing_when_within_budget() {
        let pre = summarize(&EchoAdapter, "only\ntwo", 5).unwrap();
        assert_eq!(pre.output, "only\ntwo");
        assert_eq!(pre.report.dropped_count(), 0);
    }

    /// Echo triage is conservative by design: the numbered listing is echoed,
    /// every index parses, so all candidates are kept and nothing is dropped —
    /// and the stub is flagged.
    #[test]
    fn triage_echo_keeps_all_and_flags_the_stub() {
        let candidates =
            vec!["hit a".to_string(), "hit b".to_string(), "hit c".to_string()];
        let pre = triage(&EchoAdapter, "find the thing", &candidates).unwrap();

        assert_eq!(pre.output, candidates);
        assert_eq!(pre.report.dropped_count(), 0);
        assert_eq!(pre.report.kept_units, 3);
        assert!(pre.report.notes.iter().any(|n| n == ECHO_PASSTHROUGH_NOTE));
    }

    /// The positive filtering path: a real (test) adapter selects candidates 0
    /// and 2; triage keeps those and reports candidate 1 as dropped, verbatim.
    /// The call counter proves the selection came from routing to the adapter.
    #[test]
    fn triage_keeps_the_model_selected_subset_and_reports_the_rest() {
        let adapter = FixedAdapter::new("keep 0, 2");
        let candidates =
            vec!["c0".to_string(), "c1".to_string(), "c2".to_string()];
        let pre = triage(&adapter, "q", &candidates).unwrap();

        assert_eq!(pre.output, vec!["c0".to_string(), "c2".to_string()]);
        assert_eq!(pre.report.dropped, vec!["c1".to_string()]);
        assert_eq!(pre.report.kept_units, 2);
        assert_eq!(adapter.calls.load(Ordering::SeqCst), 1, "must route through the adapter");
        // Non-echo adapter: no pass-through note.
        assert!(!pre.report.notes.iter().any(|n| n == ECHO_PASSTHROUGH_NOTE));
    }

    /// An unusable reply (no valid index) must fail safe to keep-all, never
    /// drop blindly — a false negative here is not recoverable.
    #[test]
    fn triage_fails_safe_to_keep_all_on_unusable_reply() {
        let adapter = FixedAdapter::new("i have no idea, sorry");
        let candidates = vec!["c0".to_string(), "c1".to_string()];
        let pre = triage(&adapter, "q", &candidates).unwrap();

        assert_eq!(pre.output, candidates);
        assert_eq!(pre.report.dropped_count(), 0);
        assert!(pre.report.notes.iter().any(|n| n.contains("kept ALL conservatively")));
    }

    /// classify deduplicates (the real reduction) and buckets by severity
    /// deterministically. Under echo: the duplicate error is reported dropped,
    /// buckets are correct, and both the deterministic-bucketing note and the
    /// echo pass-through note are present.
    #[test]
    fn classify_dedups_and_buckets_by_severity() {
        let raw = "error[E0432]: unresolved import `foo`\n\
                   error[E0432]: unresolved import `foo`\n\
                   warning: unused variable `x`\n\
                   note: `#[warn(unused)]` on by default";
        let pre = classify_diagnostics(&EchoAdapter, raw).unwrap();

        // Four lines in, one exact duplicate removed -> three distinct.
        assert_eq!(pre.report.input_units, 4);
        assert_eq!(pre.report.kept_units, 3);
        assert_eq!(pre.report.dropped, vec!["error[E0432]: unresolved import `foo`"]);

        let labels: Vec<&str> = pre.output.buckets.iter().map(|b| b.label.as_str()).collect();
        assert_eq!(labels, vec!["error", "warning", "note"]);
        assert_eq!(pre.output.buckets[0].items.len(), 1); // dedup collapsed the two errors

        assert!(pre.report.notes.iter().any(|n| n.contains("deterministically by severity")));
        assert!(pre.report.notes.iter().any(|n| n == ECHO_PASSTHROUGH_NOTE));
    }

    #[test]
    fn draft_commit_message_returns_a_subject_and_flags_non_authoritative() {
        let pre = draft_commit_message(&EchoAdapter, "add preprocess module\nmore detail").unwrap();
        assert_eq!(pre.output, "add preprocess module");
        assert!(pre.report.notes.iter().any(|n| n.contains("draft only")));
        assert!(pre.report.notes.iter().any(|n| n == ECHO_PASSTHROUGH_NOTE));
    }

    #[test]
    fn parse_indices_ignores_out_of_range_and_dedups() {
        assert_eq!(parse_indices("keep 0, 2 and 9", 3), vec![0, 2]);
        assert_eq!(parse_indices("1 1 1", 3), vec![1]);
        assert_eq!(parse_indices("none here", 3), Vec::<usize>::new());
    }
}

//! Bilingual aligned output (token-max card **III-7**).
//!
//! Emit source + target **aligned**, with a stable per-segment anchor, so a
//! reviewer — human or model — jumps straight to a specific segment and checks
//! it without reading the whole document. `kopitiam_token_max.md` §13 Task
//! III-7: *"Turns review from a full-document read into a targeted one."*
//!
//! The saving is the same shape as III-6's local/cloud split, but on the
//! *review* side: [`review_targets`] returns only the anchors of the segments
//! that actually need a look, so a reviewer reads `M` flagged of `N` total
//! instead of all `N`. [`review_coverage`] reports that `M / N` fraction as a
//! serializable figure (§0.2 — structure, not prose).
//!
//! # Decoupled by construction
//!
//! This module deliberately does **not** depend on `kopitiam-document`. It
//! accepts the identity/text it needs as plain inputs ([`BilingualSegment`]),
//! mirroring III-4's `TmSegment`-style decoupling: the anchor `id`, the stable
//! `index`, the `source`, and the `target` are handed in. The bridge from the
//! III-6 two-pass plan is the optional [`from_two_pass`] convenience, which
//! reads only the public fields of [`crate::SegmentPlan`] and a parallel slice
//! of finished targets — it introduces no new dependency and preserves
//! [`crate::SegmentPlan::index`] as the anchor.
//!
//! # Anchors are the whole point
//!
//! Every rendered segment carries a stable `<!-- seg <id> -->` HTML comment
//! anchor (invisible in rendered Markdown, greppable in the raw text). A
//! reviewer — or an automated check — resolves an anchor to exactly one segment
//! and inspects its source/target pair in isolation. Flagged segments
//! (explicitly `needs_review`, or below the confidence threshold) additionally
//! carry a visible [`REVIEW_MARKER`] so a human eye scans straight to them.
//!
//! Output is **deterministic**: segments are emitted in ascending `index`
//! order regardless of input order, so the same input renders byte-identically.

use serde::{Deserialize, Serialize};

/// The visible marker stamped on a segment that needs review, so a human
/// scanning the aligned output lands on it immediately. Kept as a stable
/// literal substring so tooling (and [`review_targets`] consumers) can grep for
/// it.
pub const REVIEW_MARKER: &str = "⚠ review";

/// The default confidence below which a segment is treated as needing review.
///
/// Tied to III-6's [`crate::DEFAULT_ACCEPT_THRESHOLD`]: a segment whose
/// confidence would not have cleared the two-pass *accept-local* threshold is,
/// by the same measure, one a reviewer should look at. Conservative on purpose
/// — it is better to flag a segment the reviewer skips than to hide one they
/// needed to see.
pub const DEFAULT_REVIEW_THRESHOLD: f32 = crate::DEFAULT_ACCEPT_THRESHOLD;

/// One aligned source/target pair with a stable anchor.
///
/// The inputs are decoupled from any document model: `id` is the anchor a
/// reviewer resolves, `index` fixes the deterministic emission order, and
/// `source`/`target` are the aligned text. `confidence` and `needs_review` drive
/// the review marker and [`review_targets`]; either is enough to flag a segment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BilingualSegment {
    /// Stable anchor for this segment, emitted as `<!-- seg <id> -->`. When
    /// bridged from a III-6 plan via [`from_two_pass`] this is the plan's
    /// `index` rendered as a string, so the anchor *is* the segment index.
    pub id: String,
    /// The segment's stable position; output is sorted ascending by this, so
    /// rendering is deterministic and independent of input order.
    pub index: usize,
    /// The source-language text, verbatim.
    pub source: String,
    /// The target-language text, verbatim.
    pub target: String,
    /// Optional confidence in `0..=1` (e.g. III-6's
    /// [`crate::SegmentPlan::confidence`]). When present and below the review
    /// threshold, the segment is flagged even if `needs_review` is `false`.
    pub confidence: Option<f32>,
    /// An explicit "a reviewer must look at this" flag, independent of
    /// `confidence`. Set directly, or derived (e.g. by [`from_two_pass`] from a
    /// [`crate::Decision::SendToCloud`] routing decision).
    pub needs_review: bool,
}

impl BilingualSegment {
    /// A plain constructor for the common case: no confidence recorded and not
    /// (yet) flagged for review. Fields are public, so a caller can also build
    /// the struct literally or set `confidence`/`needs_review` afterwards.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        index: usize,
        source: impl Into<String>,
        target: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            index,
            source: source.into(),
            target: target.into(),
            confidence: None,
            needs_review: false,
        }
    }

    /// Whether this segment should be surfaced to a reviewer: explicitly
    /// `needs_review`, or a recorded `confidence` strictly below `threshold`.
    #[must_use]
    pub fn is_flagged(&self, threshold: f32) -> bool {
        self.needs_review || self.confidence.is_some_and(|c| c < threshold)
    }
}

/// How to lay the aligned pairs out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Layout {
    /// Per segment: the anchor, the source line(s), then the target line(s) as a
    /// Markdown blockquote so the two are visually distinguished. Best for prose
    /// review where segments are multi-line.
    Interleaved,
    /// A single Markdown table with `seg | source | target | review` columns.
    /// Best for short, dense segments a reviewer skims as rows.
    SideBySideTable,
}

/// Options for [`render_bilingual`].
///
/// Serializable so a caller can load layout choices from its own config (§0.2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BilingualOptions {
    /// Interleaved or side-by-side table.
    pub layout: Layout,
    /// Confidence below which a segment carries the [`REVIEW_MARKER`] in the
    /// rendered output. Defaults to [`DEFAULT_REVIEW_THRESHOLD`]; a segment with
    /// no recorded confidence is flagged only by its `needs_review` bool.
    pub low_confidence_threshold: f32,
}

impl Default for BilingualOptions {
    fn default() -> Self {
        Self { layout: Layout::Interleaved, low_confidence_threshold: DEFAULT_REVIEW_THRESHOLD }
    }
}

impl BilingualOptions {
    /// Interleaved layout with the default review threshold.
    #[must_use]
    pub fn interleaved() -> Self {
        Self { layout: Layout::Interleaved, ..Self::default() }
    }

    /// Side-by-side table layout with the default review threshold.
    #[must_use]
    pub fn side_by_side() -> Self {
        Self { layout: Layout::SideBySideTable, ..Self::default() }
    }
}

/// Render aligned source/target output with stable per-segment anchors.
///
/// Segments are emitted in ascending `index` order (input order is irrelevant),
/// so the output is deterministic — the same input renders byte-identically.
/// Empty input renders the empty string.
#[must_use]
pub fn render_bilingual(segments: &[BilingualSegment], options: &BilingualOptions) -> String {
    if segments.is_empty() {
        return String::new();
    }
    let ordered = ordered_refs(segments);
    match options.layout {
        Layout::Interleaved => render_interleaved(&ordered, options.low_confidence_threshold),
        Layout::SideBySideTable => render_table(&ordered, options.low_confidence_threshold),
    }
}

/// The anchors of the segments a reviewer must actually check — the "read only
/// these" list (§0.1). A segment is included when it is `needs_review` or its
/// recorded confidence is below [`DEFAULT_REVIEW_THRESHOLD`]. Returned in
/// ascending `index` order to match the rendered output.
///
/// This is the review-side saving: a reviewer reads these `M` anchors instead of
/// all `N` segments. See [`review_coverage`] for the quantified `M / N`.
#[must_use]
pub fn review_targets(segments: &[BilingualSegment]) -> Vec<&str> {
    ordered_refs(segments)
        .into_iter()
        .filter(|s| s.is_flagged(DEFAULT_REVIEW_THRESHOLD))
        .map(|s| s.id.as_str())
        .collect()
}

/// The quantified review saving: of `total` segments, `flagged` need a look, so
/// a reviewer reads `review_fraction` of the document (§13 measurement).
///
/// Serializable so a machine consumer reports the figure as structure, not prose
/// (§0.2). `review_fraction` is `flagged / total`, or `0.0` for empty input.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReviewCoverage {
    /// Total aligned segments in the document.
    pub total: usize,
    /// Segments flagged for review (`needs_review` or below-threshold
    /// confidence) — the only ones [`review_targets`] surfaces.
    pub flagged: usize,
    /// `flagged / total` in `0..=1` (`0.0` when `total == 0`): the fraction of
    /// the document a reviewer actually reads. `1 - review_fraction` is the read
    /// that anchoring saved.
    pub review_fraction: f32,
}

/// Compute the [`ReviewCoverage`] for a set of segments using the default
/// review threshold — the measurement backing the card: a reviewer reads
/// `flagged` of `total`.
#[must_use]
pub fn review_coverage(segments: &[BilingualSegment]) -> ReviewCoverage {
    let total = segments.len();
    let flagged =
        segments.iter().filter(|s| s.is_flagged(DEFAULT_REVIEW_THRESHOLD)).count();
    let review_fraction = if total == 0 { 0.0 } else { flagged as f32 / total as f32 };
    ReviewCoverage { total, flagged, review_fraction }
}

/// Bridge from III-6: build aligned segments from a slice of two-pass
/// [`crate::SegmentPlan`]s and a parallel slice of finished `targets` (the
/// reviewed/final target text for each plan, in the same order).
///
/// Each plan's `index` becomes the anchor `id` and the emission-ordering
/// `index`, so the III-7 anchor is exactly III-6's segment index. A plan routed
/// to the cloud ([`crate::Decision::SendToCloud`]) is marked `needs_review`, and
/// the plan's `confidence` is carried through so [`render_bilingual`] and
/// [`review_targets`] can flag on it too.
///
/// Pairs are zipped, so if the slices differ in length the extra tail is
/// ignored — the caller is expected to pass one target per plan.
#[must_use]
pub fn from_two_pass(
    plans: &[crate::SegmentPlan],
    targets: &[String],
) -> Vec<BilingualSegment> {
    plans
        .iter()
        .zip(targets.iter())
        .map(|(plan, target)| BilingualSegment {
            id: plan.index.to_string(),
            index: plan.index,
            source: plan.source.clone(),
            target: target.clone(),
            confidence: Some(plan.confidence),
            needs_review: plan.decision == crate::Decision::SendToCloud,
        })
        .collect()
}

/// Segments as references, sorted ascending by `index` (stable on ties), so
/// every emission path shares one deterministic order.
fn ordered_refs(segments: &[BilingualSegment]) -> Vec<&BilingualSegment> {
    let mut refs: Vec<&BilingualSegment> = segments.iter().collect();
    refs.sort_by_key(|s| s.index);
    refs
}

/// The `<!-- seg <id> -->` anchor comment for a segment.
fn anchor(id: &str) -> String {
    format!("<!-- seg {id} -->")
}

/// Interleaved rendering: anchor, source, target-as-blockquote, per segment.
fn render_interleaved(ordered: &[&BilingualSegment], threshold: f32) -> String {
    let mut blocks: Vec<String> = Vec::with_capacity(ordered.len());
    for seg in ordered {
        let mut lines: Vec<String> = Vec::new();
        // Anchor line, with the visible review marker appended when flagged so a
        // reviewer scans straight to it.
        if seg.is_flagged(threshold) {
            lines.push(format!("{} {}", anchor(&seg.id), review_marker(seg.confidence)));
        } else {
            lines.push(anchor(&seg.id));
        }
        // Source verbatim.
        lines.push(seg.source.clone());
        // A blank line, then the target as a blockquote so source and target are
        // visually distinguished even when both are multi-line.
        lines.push(String::new());
        for line in blockquote(&seg.target) {
            lines.push(line);
        }
        blocks.push(lines.join("\n"));
    }
    // One trailing newline so the output is a well-formed block of text.
    format!("{}\n", blocks.join("\n\n"))
}

/// Side-by-side rendering: one Markdown table, one row per segment.
fn render_table(ordered: &[&BilingualSegment], threshold: f32) -> String {
    let mut out = String::new();
    out.push_str("| seg | source | target | review |\n");
    out.push_str("| --- | --- | --- | --- |\n");
    for seg in ordered {
        let review = if seg.is_flagged(threshold) { review_marker(seg.confidence) } else { String::new() };
        // The anchor lives in the `seg` cell alongside the visible id, so the
        // row is both greppable (comment) and readable (id).
        let seg_cell = format!("{} {}", anchor(&seg.id), escape_table_cell(&seg.id));
        out.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            seg_cell,
            escape_table_cell(&seg.source),
            escape_table_cell(&seg.target),
            escape_table_cell(&review),
        ));
    }
    out
}

/// The review marker, enriched with the confidence when one is recorded, while
/// always containing the stable [`REVIEW_MARKER`] substring.
fn review_marker(confidence: Option<f32>) -> String {
    match confidence {
        Some(c) => format!("{REVIEW_MARKER} (confidence {c:.2})"),
        None => REVIEW_MARKER.to_string(),
    }
}

/// Prefix each line of `text` with `> ` so it renders as a Markdown blockquote.
/// An empty target yields a single empty blockquote line so the segment still
/// has a visible target slot.
fn blockquote(text: &str) -> Vec<String> {
    if text.is_empty() {
        return vec!["> ".to_string()];
    }
    text.lines().map(|l| format!("> {l}")).collect()
}

/// Escape a string for use inside a single Markdown table cell: literal pipes
/// become `\|` (so they do not split the cell) and newlines become `<br>` (so a
/// multi-line value stays on one row), mirroring the constraint the document
/// table renderer works under.
fn escape_table_cell(text: &str) -> String {
    let piped = text.replace('|', "\\|");
    piped.replace("\r\n", "\n").replace('\r', "\n").replace('\n', "<br>")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Decision, TwoPassConfig, draft_and_route, EchoAdapter};

    fn seg(id: &str, index: usize, source: &str, target: &str) -> BilingualSegment {
        BilingualSegment::new(id, index, source, target)
    }

    /// Interleaved layout emits exactly one anchor per segment, in ascending
    /// index order, each with its source and target.
    #[test]
    fn interleaved_emits_one_anchor_per_segment_in_index_order() {
        // Deliberately out of order to prove the sort.
        let segs = vec![
            seg("b", 1, "源二", "target two"),
            seg("a", 0, "源一", "target one"),
            seg("c", 2, "源三", "target three"),
        ];
        let out = render_bilingual(&segs, &BilingualOptions::interleaved());

        assert_eq!(out.matches("<!-- seg ").count(), 3, "one anchor per segment");
        // Anchors appear in index order (a=0, b=1, c=2).
        let pa = out.find("<!-- seg a -->").unwrap();
        let pb = out.find("<!-- seg b -->").unwrap();
        let pc = out.find("<!-- seg c -->").unwrap();
        assert!(pa < pb && pb < pc, "anchors must be in ascending index order");
        // Source + target both present, target as a blockquote.
        assert!(out.contains("源一"));
        assert!(out.contains("> target one"));
    }

    /// A low-confidence segment carries the visible review marker in the output
    /// and appears in `review_targets`; a confident one does not.
    #[test]
    fn low_confidence_segment_is_marked_and_targeted() {
        let mut low = seg("lo", 0, "questionable source", "questionable target");
        low.confidence = Some(0.20); // below the default threshold
        let mut high = seg("hi", 1, "solid source", "solid target");
        high.confidence = Some(0.99); // above the default threshold

        let segs = vec![low, high];
        let out = render_bilingual(&segs, &BilingualOptions::interleaved());

        // The marker sits on the flagged segment.
        assert!(out.contains(REVIEW_MARKER), "flagged segment must carry the review marker");
        // Only the flagged anchor is a review target.
        let targets = review_targets(&segs);
        assert_eq!(targets, vec!["lo"]);
    }

    /// An explicit `needs_review` flag (no confidence) also flags a segment.
    #[test]
    fn explicit_needs_review_flags_without_confidence() {
        let mut s = seg("x", 0, "src", "tgt");
        s.needs_review = true;
        assert!(s.confidence.is_none());
        assert_eq!(review_targets(&[s]), vec!["x"]);
    }

    /// The side-by-side table escapes pipes, folds newlines, and carries the
    /// anchor in the row.
    #[test]
    fn side_by_side_table_escapes_pipes_and_has_anchor() {
        let s = seg("p", 0, "a | b\nc", "x | y");
        let out = render_bilingual(&[s], &BilingualOptions::side_by_side());

        assert!(out.contains("| seg | source | target | review |"), "header row present");
        assert!(out.contains("<!-- seg p -->"), "anchor present in the row");
        // The literal pipe in the source is escaped, not left to split the cell.
        assert!(out.contains("a \\| b"), "pipe must be escaped: {out}");
        assert!(!out.contains("a | b"), "raw unescaped pipe must not survive");
        // The newline in the source is folded to <br>.
        assert!(out.contains("<br>c"), "newline must fold to <br>: {out}");
    }

    /// The `from_two_pass` bridge builds output from III-6 plans + targets and
    /// preserves each plan's `index` as the anchor; a cloud-routed plan is
    /// flagged for review.
    #[test]
    fn from_two_pass_bridges_iii6_and_preserves_index_anchor() {
        // A mixed batch under the echo stub: a comfortable segment and a garbage
        // one, with a permissive threshold so one accepts and one routes to cloud.
        let cfg = TwoPassConfig { accept_threshold: 0.4, ..TwoPassConfig::default() };
        let long = "这是一段足够长的技术性中文文本用于测试双语对齐输出的锚点桥接";
        let batch = vec![long.to_string(), "##".to_string()];
        let plan = draft_and_route(&EchoAdapter, &batch, None, &cfg);

        // Sanity: exactly the mix the bridge test relies on.
        assert_eq!(plan.segments[0].decision, Decision::AcceptLocal);
        assert_eq!(plan.segments[1].decision, Decision::SendToCloud);

        let targets = vec!["accepted target".to_string(), "cloud target".to_string()];
        let bi = from_two_pass(&plan.segments, &targets);

        assert_eq!(bi.len(), 2);
        // index preserved as both the ordering index and the anchor id.
        assert_eq!(bi[0].index, 0);
        assert_eq!(bi[0].id, "0");
        assert_eq!(bi[1].index, 1);
        assert_eq!(bi[1].id, "1");
        // Confidence carried through from the plan.
        assert_eq!(bi[0].confidence, Some(plan.segments[0].confidence));
        // The `needs_review` bool maps the routing decision: cloud-routed is
        // flagged, accepted-local is not.
        assert!(!bi[0].needs_review);
        assert!(bi[1].needs_review);

        // The cloud-routed segment's anchor is always a review target. Under the
        // echo stub no confidence clears the conservative default review
        // threshold, so the accepted-local segment's *carried confidence* also
        // flags it — the intended "confidence can flag too" behaviour, and the
        // honest III-6 stance that a 0.5B draft warrants review.
        let targets = review_targets(&bi);
        assert!(targets.contains(&"1"), "cloud-routed anchor must be a review target: {targets:?}");
        assert!(bi[0].confidence.unwrap() < DEFAULT_REVIEW_THRESHOLD);
        assert!(targets.contains(&"0"), "a below-threshold accepted draft is still flagged by confidence");

        // The rendered output anchors on the plan index.
        let out = render_bilingual(&bi, &BilingualOptions::interleaved());
        assert!(out.contains("<!-- seg 1 -->"), "anchor must be the plan index");
    }

    /// The measurement: of N segments, only the flagged M are read.
    #[test]
    fn review_coverage_quantifies_the_saving() {
        let mut a = seg("0", 0, "s", "t");
        a.confidence = Some(0.9); // clears threshold
        let mut b = seg("1", 1, "s", "t");
        b.confidence = Some(0.1); // flagged
        let mut c = seg("2", 2, "s", "t");
        c.needs_review = true; // flagged
        let cov = review_coverage(&[a, b, c]);
        assert_eq!(cov.total, 3);
        assert_eq!(cov.flagged, 2);
        assert!((cov.review_fraction - 2.0 / 3.0).abs() < 1e-6);
    }

    /// Empty input renders the empty string, in both layouts, with a zeroed
    /// coverage.
    #[test]
    fn empty_input_is_empty_output() {
        assert_eq!(render_bilingual(&[], &BilingualOptions::interleaved()), "");
        assert_eq!(render_bilingual(&[], &BilingualOptions::side_by_side()), "");
        assert!(review_targets(&[]).is_empty());
        let cov = review_coverage(&[]);
        assert_eq!(cov.total, 0);
        assert_eq!(cov.flagged, 0);
        assert_eq!(cov.review_fraction, 0.0);
    }

    /// Same input ⇒ byte-identical output (determinism), independent of input
    /// order.
    #[test]
    fn render_is_deterministic() {
        let ordered = vec![seg("0", 0, "一", "one"), seg("1", 1, "二", "two"), seg("2", 2, "三", "three")];
        let shuffled = vec![seg("2", 2, "三", "three"), seg("0", 0, "一", "one"), seg("1", 1, "二", "two")];
        let opts = BilingualOptions::interleaved();
        assert_eq!(render_bilingual(&ordered, &opts), render_bilingual(&shuffled, &opts));

        let table = BilingualOptions::side_by_side();
        assert_eq!(render_bilingual(&ordered, &table), render_bilingual(&shuffled, &table));

        // Re-rendering the exact same input is byte-identical.
        let a = render_bilingual(&ordered, &opts);
        let b = render_bilingual(&ordered, &opts);
        assert_eq!(a, b);
    }

    /// Reportable structs carry the serde derives (§0.2) — a bound assertion, so
    /// no `serde_json` dev-dependency is introduced (mirrors `translate.rs`).
    #[test]
    fn serde_derives_are_present() {
        fn assert_serde<T: serde::Serialize + serde::de::DeserializeOwned>() {}
        assert_serde::<BilingualSegment>();
        assert_serde::<BilingualOptions>();
        assert_serde::<Layout>();
        assert_serde::<ReviewCoverage>();
    }
}

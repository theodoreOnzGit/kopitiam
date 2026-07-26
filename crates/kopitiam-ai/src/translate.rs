//! Local-first two-pass translation (token-max card **III-6**).
//!
//! The **local** model drafts *every* segment; the **cloud** model reviews
//! *only* the segments the draft is least confident about. The saving is, by
//! construction, the fraction of segments handled entirely locally — those cost
//! **zero cloud tokens**. This module produces the *plan* (draft + per-segment
//! routing decision + the signals behind it); actually dispatching the
//! low-confidence segments to a cloud adapter is left to the caller (see the
//! deferred-wiring note below).
//!
//! `kopitiam_token_max.md` §0.5 — *the local model absorbs volume, the cloud
//! model spends judgment*. Here the local model absorbs the whole document
//! (one draft per segment) and the cloud model spends its judgment only where a
//! composable confidence signal says the draft cannot be trusted.
//!
//! # Be honest about capability (this is the card's central requirement)
//!
//! §13 Task III-6: *"a 0.5B model's translations of technical prose will need
//! real review. Default conservative, widen the local share only with measured
//! evidence."* This module takes that literally and mirrors the II-6
//! preprocessing module's non-authoritative stance ([`crate::preprocess`]):
//!
//! 1. **[`Decision::AcceptLocal`] is a routing decision, not a correctness
//!    guarantee.** [`TwoPassPlan::AUTHORITATIVE`] is a compile-time `false`.
//!    "Accept locally" means only *"the signals do not justify spending a cloud
//!    review on this one"* — never *"this translation is correct"*.
//! 2. **Conservative by default.** [`TwoPassConfig::default`] sets a **high**
//!    [`TwoPassConfig::accept_threshold`] (see [`DEFAULT_ACCEPT_THRESHOLD`]), so
//!    most segments route to the cloud until measured evidence (a reviewer
//!    comparing accepted-local drafts against cloud output on a real corpus)
//!    justifies lowering it to widen the local share. The threshold is the
//!    single knob that trades saving for safety, and it starts safe.
//! 3. **Never silently accept.** Every [`SegmentPlan`] records the
//!    [`ConfidenceSignals`] behind its decision, so any acceptance is auditable
//!    and reproducible. The plan is a pure function of its inputs.
//! 4. **The echo stub is detected and discounted.** When the injected adapter
//!    is [`crate::EchoAdapter`] (no local `.gguf`), the "draft" is the source
//!    echoed back and the round-trip agreement is trivially perfect and
//!    *meaningless*. Rather than let that fake-perfect signal inflate
//!    confidence, the round-trip signal is marked **untrusted** and folded in as
//!    neutral, and [`ECHO_PASSTHROUGH_NOTE`] is stamped into the plan (mirroring
//!    II-6's stub honesty, enforced at the library layer via
//!    [`ModelAdapter::name`]).
//!
//! # The confidence signal (composable, three parts)
//!
//! [`ConfidenceSignals`] combines three independent 0..1 sub-scores into one
//! 0..1 `confidence`, each capturing a different way a draft can be unreliable:
//!
//! * **Length** — a very short segment (a stray heading, a garbage line, an
//!   orphaned number) or a very long one gives a 0.5B model the least to work
//!   with and the most room to drift; only a comfortable middle band scores
//!   high. See [`length_score`].
//! * **Glossary-hit rate** — reusing the III-5 [`Glossary`], the more
//!   project-decided terms the draft contains (now pinned *deterministically*,
//!   not by the model), the more of the segment is already trustworthy. Hits
//!   raise confidence with diminishing returns; zero hits are neutral, never a
//!   penalty. See [`glossary_score`].
//! * **Round-trip disagreement** — the draft is translated *back* to the source
//!   language with the same local model and compared to the original source
//!   (character-bigram Sørensen–Dice, [`round_trip_similarity`]). High
//!   divergence means the model could not preserve meaning through the round
//!   trip → low confidence. Under the echo stub this is trivially perfect and is
//!   therefore discounted (see above).
//!
//! The three are combined by a fixed, documented weighted mean
//! ([`ConfidenceSignals::combine`]); no single signal can carry a draft over a
//! conservative threshold on its own.
//!
//! # Measuring the cloud-token saving
//!
//! The split *is* the measurement. [`TwoPassPlan::summary`] reports `total`,
//! `accepted_local`, `sent_to_cloud`, and `local_fraction`:
//!
//! ```text
//! cloud tokens saved  ≈  local_fraction × total × (per-segment cloud review cost)
//! ```
//!
//! i.e. every segment marked [`Decision::AcceptLocal`] is one segment the cloud
//! model never has to read, so its would-be review tokens are saved outright.
//! **The honest caveat:** `local_fraction` is only as trustworthy as the
//! threshold that produced it. A high `local_fraction` is a *saving claim*, not
//! a *quality claim* — it should be widened past the conservative default only
//! after a reviewer has measured, on a real corpus, that the accepted-local
//! drafts hold up without the cloud pass (§13: *widen the local share only with
//! measured evidence*).

use serde::{Deserialize, Serialize};

use crate::{CompletionRequest, Glossary, Message, ModelAdapter};

/// The name [`crate::EchoAdapter::name`] reports. Used to detect the "no real
/// local model" case so the round-trip signal can be honestly discounted.
const ECHO_ADAPTER_NAME: &str = "echo";

/// Stamped into a [`TwoPassPlan`] whenever drafting ran against the echo stub
/// instead of a real local model: the "draft" is the source echoed back and the
/// round-trip agreement is trivially perfect, so the round-trip signal carries
/// no information and is folded in as neutral. Do not read a high
/// `local_fraction` from an echo run as evidence the local model is good enough.
pub const ECHO_PASSTHROUGH_NOTE: &str = "adapter is the echo stub (no local .gguf) — drafts are a \
     pass-through of the source and the round-trip signal is meaningless; treat every AcceptLocal \
     here as unproven and route to cloud in production.";

/// The conservative default [`TwoPassConfig::accept_threshold`].
///
/// Set **high on purpose** (§13 Task III-6): a 0.5B model's technical-prose
/// translation needs real review, so at the default most segments route to the
/// cloud. Lower it — widening the local share — only once a reviewer has
/// measured that accepted-local drafts hold up on a real corpus.
pub const DEFAULT_ACCEPT_THRESHOLD: f32 = 0.85;

/// What to do with one drafted segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    /// The local draft is confident enough that spending a cloud review on it is
    /// not justified. **A routing decision, not a correctness guarantee** — see
    /// the module docs and [`TwoPassPlan::AUTHORITATIVE`].
    AcceptLocal,
    /// The local draft is low-confidence; send this segment to the cloud model
    /// for review. These are the only segments that cost cloud tokens.
    SendToCloud,
}

/// Configuration for [`draft_and_route`]: the routing threshold, the length
/// comfort band, and the source/target language hints used only to frame the
/// draft prompt.
///
/// Serializable so a caller can load it from its own config format.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TwoPassConfig {
    /// A segment whose combined `confidence` is **greater than or equal to**
    /// this is [`Decision::AcceptLocal`]; below it, [`Decision::SendToCloud`].
    /// Defaults to the conservative [`DEFAULT_ACCEPT_THRESHOLD`]; raise toward
    /// `1.0` to send even more to the cloud, lower it only with measured
    /// evidence to widen the local share.
    pub accept_threshold: f32,
    /// The shortest segment (in `char`s) that still earns a full length score.
    /// Below this the length score falls off linearly to zero — very short
    /// segments (stray headings, garbage lines) are the least reliable.
    pub min_comfortable_chars: usize,
    /// The longest segment (in `char`s) that still earns a full length score.
    /// Above this the length score decays — very long segments give a small
    /// model the most room to drift.
    pub max_comfortable_chars: usize,
    /// Source-language hint for the draft prompt (e.g. `"Chinese"`). Framing
    /// only; the deterministic stub adapter ignores it.
    pub source_lang: String,
    /// Target-language hint for the draft prompt (e.g. `"English"`).
    pub target_lang: String,
}

impl Default for TwoPassConfig {
    fn default() -> Self {
        Self {
            accept_threshold: DEFAULT_ACCEPT_THRESHOLD,
            min_comfortable_chars: 24,
            max_comfortable_chars: 320,
            // The driving case is Chinese-language nuclear literature (§13).
            source_lang: "Chinese".to_string(),
            target_lang: "English".to_string(),
        }
    }
}

/// The composable evidence behind one segment's `confidence`, recorded so every
/// decision is auditable (never a silent accept).
///
/// Each `*_score` is in `0..=1`; `confidence` is their documented weighted mean
/// ([`ConfidenceSignals::combine`]). Raw inputs (`char_len`, `glossary_hits`,
/// `round_trip_similarity`) are kept alongside the scores so a reviewer can see
/// *why* a score landed where it did.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfidenceSignals {
    /// Length of the source segment in `char`s (the raw input to
    /// [`length_score`]).
    pub char_len: usize,
    /// Length sub-score: high only within the comfortable band, falling off for
    /// very short or very long segments.
    pub length_score: f32,
    /// Number of III-5 glossary occurrences the draft matched (0 when no
    /// glossary was supplied).
    pub glossary_hits: usize,
    /// Glossary sub-score: neutral (0.5) with no hits, rising with diminishing
    /// returns as more project-decided terms are pinned deterministically.
    pub glossary_score: f32,
    /// Raw round-trip agreement in `0..=1`: character-bigram Sørensen–Dice
    /// similarity between the source and the draft translated *back* to the
    /// source language (`1.0` = identical). Always recorded, even when not
    /// trusted.
    pub round_trip_similarity: f32,
    /// Whether the round-trip signal is trustworthy. `false` under the echo stub
    /// (the round trip is trivially perfect and meaningless); when `false`,
    /// `round_trip_score` is neutralised to `0.5` regardless of
    /// `round_trip_similarity`.
    pub round_trip_trusted: bool,
    /// Round-trip sub-score actually folded into `confidence`: equal to
    /// `round_trip_similarity` when trusted, else the neutral `0.5`.
    pub round_trip_score: f32,
    /// The combined 0..1 confidence — the value compared against
    /// [`TwoPassConfig::accept_threshold`].
    pub confidence: f32,
}

impl ConfidenceSignals {
    /// Weight of the length sub-score in the combined confidence.
    pub const W_LENGTH: f32 = 0.35;
    /// Weight of the round-trip sub-score in the combined confidence.
    pub const W_ROUND_TRIP: f32 = 0.45;
    /// Weight of the glossary sub-score in the combined confidence. The three
    /// weights sum to `1.0`, so `confidence` stays in `0..=1` and no single
    /// signal can carry a draft over a conservative threshold on its own.
    pub const W_GLOSSARY: f32 = 0.20;

    /// Combine the three sub-scores into the final 0..1 confidence: a fixed,
    /// documented weighted mean (`0.35·length + 0.45·round_trip + 0.20·glossary`).
    #[must_use]
    pub fn combine(length_score: f32, round_trip_score: f32, glossary_score: f32) -> f32 {
        (Self::W_LENGTH * length_score
            + Self::W_ROUND_TRIP * round_trip_score
            + Self::W_GLOSSARY * glossary_score)
            .clamp(0.0, 1.0)
    }
}

/// The plan for one segment: the local draft, the confidence, the routing
/// decision, and the signals behind it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentPlan {
    /// The segment's index in the input slice (stable, for anchoring back to the
    /// source — e.g. for the III-7 bilingual output).
    pub index: usize,
    /// The source segment, verbatim.
    pub source: String,
    /// The local model's draft translation, after the deterministic III-5
    /// glossary post-pass.
    pub local_draft: String,
    /// The combined confidence (mirrors `signals.confidence`, surfaced here for
    /// convenience).
    pub confidence: f32,
    /// Whether to keep the local draft or send this segment to the cloud.
    pub decision: Decision,
    /// The composable evidence behind `decision` — never omitted, so acceptance
    /// is always auditable.
    pub signals: ConfidenceSignals,
}

/// The result of [`draft_and_route`]: one [`SegmentPlan`] per input segment,
/// plus any honesty caveats that apply to the whole plan.
///
/// The measurable split is [`TwoPassPlan::summary`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TwoPassPlan {
    /// One plan per input segment, in input order.
    pub segments: Vec<SegmentPlan>,
    /// Plan-wide caveats: the [`ECHO_PASSTHROUGH_NOTE`] when drafted against the
    /// stub, and the standing reminder that AcceptLocal is a routing decision,
    /// not a correctness guarantee.
    pub notes: Vec<String>,
}

impl TwoPassPlan {
    /// A [`Decision::AcceptLocal`] is a **routing** decision, never the final
    /// authority on whether a translation is correct (§13 Task III-6, mirroring
    /// [`crate::preprocess`]). A compile-time constant so callers can encode
    /// "do not gate correctness on this plan" in their own logic and tests.
    pub const AUTHORITATIVE: bool = false;

    /// The measurable split: totals and the local fraction (§13 — *report the
    /// split so the saving is measurable*).
    #[must_use]
    pub fn summary(&self) -> TwoPassSummary {
        let total = self.segments.len();
        let accepted_local =
            self.segments.iter().filter(|s| s.decision == Decision::AcceptLocal).count();
        let sent_to_cloud = total - accepted_local;
        let local_fraction =
            if total == 0 { 0.0 } else { accepted_local as f32 / total as f32 };
        TwoPassSummary { total, accepted_local, sent_to_cloud, local_fraction }
    }
}

/// The measurable split produced by [`TwoPassPlan::summary`].
///
/// `local_fraction × total × (per-segment cloud review cost)` is the cloud
/// tokens saved. Serializable so a machine consumer reports structure, not prose
/// (§0.2).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TwoPassSummary {
    /// Total segments planned.
    pub total: usize,
    /// Segments kept locally (zero cloud tokens each).
    pub accepted_local: usize,
    /// Segments routed to the cloud for review (the only ones that cost cloud
    /// tokens).
    pub sent_to_cloud: usize,
    /// `accepted_local / total` in `0..=1` (`0.0` for an empty plan) — the
    /// fraction of the document handled entirely locally, and the saving figure.
    /// A *saving* claim, not a *quality* claim: see the module docs' caveat.
    pub local_fraction: f32,
}

/// Draft every segment with the **local** `adapter`, apply the III-5 `glossary`
/// deterministically, score confidence from composable signals, and route each
/// segment to `AcceptLocal` or `SendToCloud` against `cfg.accept_threshold`.
///
/// The `adapter` is injected — the caller is expected to hand in the **local**
/// adapter it already resolved, so drafting costs **zero cloud tokens** by
/// construction. Because the surface is `&dyn ModelAdapter`, tests drive it with
/// the deterministic [`crate::EchoAdapter`] (no weights, no network).
///
/// For each segment the local model is called **twice**: once to draft the
/// translation, once to translate the draft back for the round-trip signal. The
/// result is a pure function of `(adapter, segments, glossary, cfg)` for a
/// deterministic adapter — same input in, byte-identical plan out.
#[must_use]
pub fn draft_and_route(
    adapter: &dyn ModelAdapter,
    segments: &[String],
    glossary: Option<&Glossary>,
    cfg: &TwoPassConfig,
) -> TwoPassPlan {
    let echo_stub = adapter.name() == ECHO_ADAPTER_NAME;

    let plans = segments
        .iter()
        .enumerate()
        .map(|(index, source)| plan_segment(adapter, index, source, glossary, cfg, echo_stub))
        .collect();

    let mut notes = vec![
        "AcceptLocal is a routing decision, not a correctness guarantee: it means only that the \
         signals did not justify a cloud review, never that the draft is correct. Widen the local \
         share past the conservative default only with measured evidence (§13 Task III-6)."
            .to_string(),
    ];
    if echo_stub {
        notes.push(ECHO_PASSTHROUGH_NOTE.to_string());
    }

    TwoPassPlan { segments: plans, notes }
}

/// Plan a single segment: draft (pass 1), glossary post-pass, round-trip
/// (pass 2), score, decide.
fn plan_segment(
    adapter: &dyn ModelAdapter,
    index: usize,
    source: &str,
    glossary: Option<&Glossary>,
    cfg: &TwoPassConfig,
    echo_stub: bool,
) -> SegmentPlan {
    // Pass 1: the local model drafts the translation.
    let raw_draft = translate(adapter, source, &cfg.source_lang, &cfg.target_lang);

    // III-5 glossary as a deterministic post-pass; the hit count folds into the
    // glossary signal. No model tokens spent on terminology.
    let (local_draft, glossary_hits) = match glossary {
        Some(g) if !g.is_empty() => {
            let applied = g.apply(&raw_draft);
            let hits = applied.total_substitutions();
            (applied.output, hits)
        }
        _ => (raw_draft, 0),
    };

    // Pass 2: translate the draft back to the source language and measure how
    // far it diverged from the original source.
    let back = translate(adapter, &local_draft, &cfg.target_lang, &cfg.source_lang);
    let round_trip_similarity = dice_similarity(source, &back);

    // Compose the three sub-scores.
    let char_len = source.chars().count();
    let length_score = length_score(char_len, cfg);
    let glossary_score = glossary_score(glossary_hits, glossary.is_some());
    // The echo stub's round trip is trivially perfect and carries no
    // information; discount it to neutral so it cannot inflate confidence.
    let round_trip_trusted = !echo_stub;
    let round_trip_score = if round_trip_trusted { round_trip_similarity } else { 0.5 };

    let confidence = ConfidenceSignals::combine(length_score, round_trip_score, glossary_score);
    let decision = if confidence >= cfg.accept_threshold {
        Decision::AcceptLocal
    } else {
        Decision::SendToCloud
    };

    SegmentPlan {
        index,
        source: source.to_string(),
        local_draft,
        confidence,
        decision,
        signals: ConfidenceSignals {
            char_len,
            length_score,
            glossary_hits,
            glossary_score,
            round_trip_similarity,
            round_trip_trusted,
            round_trip_score,
            confidence,
        },
    }
}

/// Route one translation request through the adapter and return the reply. This
/// is the single point volume is handed to the *local* model — the caller having
/// passed in the local adapter is what makes it cost zero cloud tokens.
fn translate(adapter: &dyn ModelAdapter, text: &str, from: &str, to: &str) -> String {
    let request = CompletionRequest::new([
        Message::system(format!(
            "Translate the {from} text the user sends into {to}. Output only the {to} \
             translation, nothing else."
        )),
        Message::user(text),
    ]);
    // A drafting/round-trip step must never abort the whole plan: on adapter
    // error, treat the draft as empty (which scores low and routes to cloud —
    // the safe direction).
    adapter.complete(&request).map(|r| r.content).unwrap_or_default()
}

/// Length sub-score in `0..=1`: `1.0` within the comfortable band
/// `[min, max]`, falling off linearly to `0.0` at zero length and decaying
/// toward a `0.3` floor for over-long segments.
///
/// Very short segments (a stray heading, a garbage line, an orphaned number)
/// and very long ones give a 0.5B model the least reliable footing, so they
/// score low and route to the cloud.
#[must_use]
pub fn length_score(char_len: usize, cfg: &TwoPassConfig) -> f32 {
    let min = cfg.min_comfortable_chars.max(1);
    let max = cfg.max_comfortable_chars.max(min);
    if char_len == 0 {
        0.0
    } else if char_len < min {
        // Linear 0 -> 1 across (0, min).
        char_len as f32 / min as f32
    } else if char_len <= max {
        1.0
    } else {
        // Decay from 1.0 at `max` toward a 0.3 floor, reaching the floor at
        // 2·max and below.
        let over = (char_len - max) as f32;
        let span = max as f32; // width over which we decay to the floor
        (1.0 - 0.7 * (over / span)).max(0.3)
    }
}

/// Glossary sub-score in `0..=1`. Neutral `0.5` when no glossary was supplied or
/// nothing hit; rising with diminishing returns as more project-decided terms
/// are pinned deterministically (`0.5 + 0.5·(1 − 1/(1+hits))`). Hits are
/// positive evidence — never a penalty for a term-free segment.
#[must_use]
pub fn glossary_score(hits: usize, glossary_present: bool) -> f32 {
    if !glossary_present {
        return 0.5;
    }
    0.5 + 0.5 * (1.0 - 1.0 / (1.0 + hits as f32))
}

/// Character-bigram Sørensen–Dice similarity in `0..=1` (`1.0` = identical),
/// the round-trip agreement metric. Deterministic, dependency-free, and
/// character-based so it works on CJK source text.
///
/// `dice = 2·|A ∩ B| / (|A| + |B|)` over the two strings' character-bigram
/// multisets. Strings too short to form a bigram fall back to exact equality.
#[must_use]
pub fn dice_similarity(a: &str, b: &str) -> f32 {
    let ba = bigrams(a);
    let bb = bigrams(b);
    if ba.is_empty() || bb.is_empty() {
        // No bigrams (one or both < 2 chars): fall back to exact equality.
        return if a == b { 1.0 } else { 0.0 };
    }
    // Multiset intersection size.
    let mut bb_counts: std::collections::HashMap<(char, char), i32> = std::collections::HashMap::new();
    for g in &bb {
        *bb_counts.entry(*g).or_insert(0) += 1;
    }
    let mut inter = 0usize;
    for g in &ba {
        let e = bb_counts.entry(*g).or_insert(0);
        if *e > 0 {
            *e -= 1;
            inter += 1;
        }
    }
    (2.0 * inter as f32) / (ba.len() as f32 + bb.len() as f32)
}

/// The character bigrams of `s`, in order (a multiset as a `Vec`).
fn bigrams(s: &str) -> Vec<(char, char)> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < 2 {
        return Vec::new();
    }
    chars.windows(2).map(|w| (w[0], w[1])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CompletionResponse, EchoAdapter, Entry, Glossary};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A non-echo adapter that returns a fixed reply and *counts* calls, so a
    /// test can prove both the draft pass and the round-trip pass genuinely
    /// routed through the adapter.
    struct CountingAdapter {
        reply: String,
        calls: AtomicUsize,
    }
    impl CountingAdapter {
        fn new(reply: &str) -> Self {
            Self { reply: reply.to_string(), calls: AtomicUsize::new(0) }
        }
    }
    impl ModelAdapter for CountingAdapter {
        fn name(&self) -> &str {
            "counting-test"
        }
        fn complete(&self, _request: &CompletionRequest) -> anyhow::Result<CompletionResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(CompletionResponse { content: self.reply.clone(), model: "counting-test".into() })
        }
    }

    /// Compile-time guarantee that an AcceptLocal is never treated as the final
    /// authority on correctness (§13 Task III-6).
    const _: () = assert!(!TwoPassPlan::AUTHORITATIVE);

    fn segs(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_string()).collect()
    }

    /// Drafting routes through the adapter, and both the draft **and** the
    /// round-trip pass fire — proven by a call-counting adapter seeing exactly
    /// two calls per segment.
    #[test]
    fn drafts_and_round_trips_route_through_the_adapter() {
        let adapter = CountingAdapter::new("some translation");
        let plan = draft_and_route(&adapter, &segs(&["源文本一", "源文本二"]), None, &TwoPassConfig::default());

        assert_eq!(plan.segments.len(), 2);
        // 2 segments × (1 draft + 1 round-trip) = 4 adapter calls.
        assert_eq!(adapter.calls.load(Ordering::SeqCst), 4, "draft + round-trip must both route through the adapter");
        // The draft is the adapter's reply.
        assert_eq!(plan.segments[0].local_draft, "some translation");
    }

    /// A very-short / garbage segment scores low on length and routes to cloud,
    /// even under the (otherwise generous) echo stub.
    #[test]
    fn very_short_segment_scores_low_and_goes_to_cloud() {
        let plan = draft_and_route(&EchoAdapter, &segs(&["##"]), None, &TwoPassConfig::default());
        let s = &plan.segments[0];
        assert!(s.signals.length_score < 0.2, "a 2-char segment must score low on length: {}", s.signals.length_score);
        assert_eq!(s.decision, Decision::SendToCloud);
    }

    /// The summary's `local_fraction` matches the decisions exactly.
    #[test]
    fn summary_local_fraction_matches_decisions() {
        // A permissive threshold so a mix of accept/cloud is produced under echo.
        let cfg = TwoPassConfig { accept_threshold: 0.4, ..TwoPassConfig::default() };
        // One comfortable-length segment (accepts) + one 2-char garbage (cloud).
        let long = "这是一段足够长的技术性中文文本用于测试本地优先的两遍翻译流程的置信度评分";
        let plan = draft_and_route(&EchoAdapter, &segs(&[long, "##"]), None, &cfg);
        let summary = plan.summary();

        let expected_local =
            plan.segments.iter().filter(|s| s.decision == Decision::AcceptLocal).count();
        assert_eq!(summary.total, 2);
        assert_eq!(summary.accepted_local, expected_local);
        assert_eq!(summary.sent_to_cloud, 2 - expected_local);
        assert!((summary.local_fraction - expected_local as f32 / 2.0).abs() < 1e-6);
        // The mix we set up: the long one accepts, the garbage one goes to cloud.
        assert_eq!(summary.accepted_local, 1);
        assert_eq!(summary.sent_to_cloud, 1);
    }

    /// Glossary hits fold into the confidence signal: the same segment scores a
    /// higher glossary sub-score (and thus higher confidence) with a matching
    /// glossary than without one.
    #[test]
    fn glossary_hits_raise_the_confidence_signal() {
        // Under echo the draft is the source echoed back, so a CJK glossary term
        // in the source will hit in the draft.
        let source = "华龙一号的数字孪生模型用于压水堆的乏燃料管理与安全评估研究工作";
        let glossary = Glossary::from_entries([
            Entry::new("华龙一号", "Hualong One"),
            Entry::new("数字孪生", "digital twin"),
        ])
        .unwrap();
        let cfg = TwoPassConfig::default();

        let without = draft_and_route(&EchoAdapter, &segs(&[source]), None, &cfg);
        let with = draft_and_route(&EchoAdapter, &segs(&[source]), Some(&glossary), &cfg);

        assert_eq!(with.segments[0].signals.glossary_hits, 2, "both terms should hit");
        assert!(
            with.segments[0].signals.glossary_score > without.segments[0].signals.glossary_score,
            "glossary hits must raise the glossary sub-score ({} !> {})",
            with.segments[0].signals.glossary_score,
            without.segments[0].signals.glossary_score
        );
        assert!(
            with.segments[0].confidence > without.segments[0].confidence,
            "glossary hits must raise overall confidence"
        );
    }

    /// The conservative default threshold sends **most** of a mixed batch to the
    /// cloud (§13: default conservative). Under the echo stub the round-trip
    /// signal is discounted to neutral, so even comfortable-length segments do
    /// not clear the high default.
    #[test]
    fn conservative_default_sends_most_to_cloud() {
        let batch = segs(&[
            "第一段中等长度的中文技术文本内容用于测试保守默认阈值的路由行为表现",
            "第二段中等长度的中文技术文本内容用于测试保守默认阈值的路由行为表现",
            "第三段中等长度的中文技术文本内容用于测试保守默认阈值的路由行为表现",
            "短",
            "##",
        ]);
        let plan = draft_and_route(&EchoAdapter, &batch, None, &TwoPassConfig::default());
        let summary = plan.summary();
        assert!(
            summary.sent_to_cloud > summary.accepted_local,
            "conservative default must send most to cloud: {summary:?}"
        );
        // Echo stub honesty note present.
        assert!(plan.notes.iter().any(|n| n == ECHO_PASSTHROUGH_NOTE));
    }

    /// The echo stub discounts the round-trip signal: it is recorded but marked
    /// untrusted and folded in as the neutral 0.5, never the trivially-perfect
    /// 1.0 that the stub's pass-through would otherwise produce.
    #[test]
    fn echo_round_trip_is_recorded_but_not_trusted() {
        let plan = draft_and_route(&EchoAdapter, &segs(&["一段用于测试往返信号折扣处理的中文文本内容"]), None, &TwoPassConfig::default());
        let sig = &plan.segments[0].signals;
        assert!(!sig.round_trip_trusted, "echo round-trip must not be trusted");
        assert!((sig.round_trip_score - 0.5).abs() < 1e-6, "untrusted round-trip folds in as neutral 0.5");
    }

    /// A real (non-echo) adapter's round-trip *is* trusted and measured. Here
    /// the reply never matches the CJK source, so similarity is ~0 → very low
    /// confidence → cloud.
    #[test]
    fn non_echo_round_trip_is_trusted_and_measured() {
        let adapter = CountingAdapter::new("a completely unrelated english reply");
        let plan = draft_and_route(&adapter, &segs(&["一段足够长的中文源文本用于测试往返一致性信号"]), None, &TwoPassConfig::default());
        let sig = &plan.segments[0].signals;
        assert!(sig.round_trip_trusted);
        assert!(sig.round_trip_similarity < 0.2, "unrelated back-translation must diverge: {}", sig.round_trip_similarity);
        assert_eq!(plan.segments[0].decision, Decision::SendToCloud);
    }

    /// Same input ⇒ byte-identical plan (determinism).
    #[test]
    fn plan_is_deterministic() {
        let glossary = Glossary::from_entries([Entry::new("压水堆", "PWR")]).unwrap();
        let batch = segs(&["压水堆机组的数字孪生", "##", "another segment of moderate length here"]);
        let cfg = TwoPassConfig::default();

        let a = draft_and_route(&EchoAdapter, &batch, Some(&glossary), &cfg);
        let b = draft_and_route(&EchoAdapter, &batch, Some(&glossary), &cfg);
        assert_eq!(a, b, "same input must yield an identical plan");
    }

    /// An empty batch produces an empty plan with a well-defined summary.
    #[test]
    fn empty_batch_has_zero_local_fraction() {
        let plan = draft_and_route(&EchoAdapter, &[], None, &TwoPassConfig::default());
        let summary = plan.summary();
        assert_eq!(summary.total, 0);
        assert_eq!(summary.accepted_local, 0);
        assert_eq!(summary.local_fraction, 0.0);
    }

    #[test]
    fn length_score_shape() {
        let cfg = TwoPassConfig::default();
        assert_eq!(length_score(0, &cfg), 0.0);
        assert!(length_score(2, &cfg) < 0.2); // very short
        assert_eq!(length_score(cfg.min_comfortable_chars, &cfg), 1.0);
        assert_eq!(length_score(cfg.max_comfortable_chars, &cfg), 1.0);
        // Over-long decays but never below the 0.3 floor.
        assert!(length_score(cfg.max_comfortable_chars * 10, &cfg) >= 0.3);
        assert!(length_score(cfg.max_comfortable_chars + 1, &cfg) < 1.0);
    }

    #[test]
    fn glossary_score_shape() {
        // No glossary supplied -> neutral.
        assert!((glossary_score(0, false) - 0.5).abs() < 1e-6);
        // Glossary present, no hits -> still neutral (never a penalty).
        assert!((glossary_score(0, true) - 0.5).abs() < 1e-6);
        // Hits raise it, monotonically, with diminishing returns.
        assert!(glossary_score(1, true) > glossary_score(0, true));
        assert!(glossary_score(3, true) > glossary_score(1, true));
        assert!(glossary_score(100, true) < 1.0);
    }

    #[test]
    fn dice_similarity_bounds() {
        assert_eq!(dice_similarity("华龙一号", "华龙一号"), 1.0);
        assert_eq!(dice_similarity("abc", "abc"), 1.0);
        assert!(dice_similarity("华龙一号", "完全不同的文本").abs() < 0.1);
        // Sub-bigram strings fall back to exact equality.
        assert_eq!(dice_similarity("a", "a"), 1.0);
        assert_eq!(dice_similarity("a", "b"), 0.0);
        assert_eq!(dice_similarity("", ""), 1.0);
    }

    /// Compile-time check that the reportable structs carry the serde derives
    /// (§0.2 — structure, not prose). Kept as a bound assertion so no
    /// `serde_json` dev-dependency is introduced.
    #[test]
    fn serde_derives_are_present() {
        fn assert_serde<T: serde::Serialize + serde::de::DeserializeOwned>() {}
        assert_serde::<TwoPassPlan>();
        assert_serde::<SegmentPlan>();
        assert_serde::<TwoPassSummary>();
        assert_serde::<ConfidenceSignals>();
        assert_serde::<TwoPassConfig>();
        assert_serde::<Decision>();
    }
}

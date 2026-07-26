//! Deterministic terminology glossary enforcement (token-max card **III-5**).
//!
//! A project glossary maps a **source** term to the **target** term the project
//! has already decided on — e.g. `华龙一号` → `"Hualong One"`, `数字孪生` →
//! `"digital twin"`, `压水堆` → `"PWR"`, `乏燃料` → `"spent fuel"` — and is
//! applied to translated text by [`Glossary::apply`] **deterministically**, with
//! no model call. The driving case is Chinese-language nuclear literature
//! (`kopitiam_token_max.md` §13), where the same handful of domain terms recur
//! hundreds of times across a long document.
//!
//! # Why this is a token-max capability (§0.3, §13 Task III-5)
//!
//! §0.3 — *deterministic tool over model call*: once the project has decided
//! that `华龙一号` is rendered `"Hualong One"`, re-deciding that per occurrence
//! is pure waste. A translation model, asked to translate a segment containing
//! the term, spends judgment tokens on it **every time it appears**, and — worse
//! — is free to answer differently each time (`"Hualong One"` here,
//! `"Hualong-1"` three pages later, `"HPR1000"` in the conclusion). That drift
//! is invisible until a reviewer catches it, and fixing it costs a **whole
//! review pass over the finished document**.
//!
//! Applying the glossary deterministically spends **zero** model tokens on
//! terminology and makes drift impossible by construction: every occurrence of a
//! source term becomes byte-identical target text, this run and every future
//! run. The measurement is direct — see [`Applied`] and [`Substitution`]:
//! `N` glossary terms hit across `M` total occurrences is `M` per-occurrence
//! model decisions eliminated, each one also a drift opportunity removed.
//!
//! # Deterministic matching rules (the contract — read before trusting output)
//!
//! [`Glossary::apply`] makes a **single left-to-right pass** over the input and
//! substitutes **non-overlapping** matches. Three rules make the result exact
//! and reproducible:
//!
//! 1. **Longest-source-first.** At each position the longest source term that
//!    matches wins, so a 4-character term is never partially clobbered by a
//!    shorter prefix entry. If a glossary contained both `华龙一号` and a
//!    hypothetical `华龙`, the text `华龙一号` matches the full 4-char entry, not
//!    the 2-char one. Ties in length are broken by declaration order, so the
//!    result is a pure function of `(entries, input)`.
//! 2. **CJK vs Latin boundaries, per entry.** CJK writing has no inter-word
//!    spaces, so a substring match is exactly right for a CJK source term. An
//!    ASCII/Latin source term instead respects **word boundaries**, so `"core"`
//!    is not rewritten inside `"encore"` or `"scoreboard"`. Each entry's rule is
//!    chosen from its own source text (see [`SourceKind`]): a source containing
//!    any CJK ideograph matches as a substring; an all-ASCII source with a
//!    letter/digit matches only at word boundaries; anything else falls back to
//!    substring.
//! 3. **Idempotent within a pass.** After a match the cursor advances **past the
//!    consumed source**, never into the freshly inserted target, so an inserted
//!    target is never re-scanned for substitution in the same pass. For the
//!    normal source-language → target-language direction (no target text
//!    contains another entry's source), [`Glossary::apply`] is therefore fully
//!    idempotent: applying it twice yields byte-identical output to applying it
//!    once. (The one way to break that is a glossary whose *target* contains
//!    another entry's *source* — e.g. mapping `twin` → something while also
//!    producing `"digital twin"`; a normal cross-language glossary never does.)
//!
//! **Case.** Latin source matching is **case-sensitive** by default: `"PWR"`
//! does not match `"pwr"`. This is deliberate — technical acronyms and
//! capitalised proper terms are case-bearing, and a case-insensitive default
//! would silently rewrite unrelated lowercase words. A case-folding option is a
//! documented future extension (see [`Glossary::apply`]); it is not enabled here
//! because the driving CJK case does not need it.
//!
//! # Auditability (never silent)
//!
//! Like the II-6 preprocessing helpers, this never edits text silently:
//! [`Glossary::apply`] returns an [`Applied`] whose [`Applied::substitutions`]
//! lists every term that fired and its exact occurrence `count`, so the saving
//! is measurable and every rewrite is reviewable.

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

/// One glossary rule: replace every (deterministically matched) occurrence of
/// `source` with `target`.
///
/// Deserializable so a glossary can be loaded from a serde structure (JSON, or
/// TOML via a `toml`-owning caller — this crate adds no `toml` dependency). A
/// JSON entry is `{"source": "华龙一号", "target": "Hualong One"}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    /// The term as it appears in the (translated) text.
    pub source: String,
    /// The project-decided replacement.
    pub target: String,
}

impl Entry {
    /// Construct an entry from any string-likes.
    pub fn new(source: impl Into<String>, target: impl Into<String>) -> Self {
        Self { source: source.into(), target: target.into() }
    }
}

/// Which deterministic matching rule an [`Entry`]'s `source` uses, decided from
/// the source text itself (rule 2 above).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    /// Source contains at least one CJK ideograph → matched as a raw substring
    /// (CJK has no word boundaries, so any occurrence is a real occurrence).
    Cjk,
    /// Source is all-ASCII and contains a letter or digit → matched only at
    /// word boundaries, so it is not rewritten inside a larger word.
    Latin,
    /// Neither of the above (e.g. non-CJK non-ASCII script, or punctuation
    /// only) → matched as a raw substring, the safe general default.
    Other,
}

impl SourceKind {
    /// Classify a source term. See [`SourceKind`] variants for the rule.
    #[must_use]
    pub fn of(source: &str) -> Self {
        if source.chars().any(is_cjk) {
            SourceKind::Cjk
        } else if source.is_ascii() && source.chars().any(|c| c.is_ascii_alphanumeric()) {
            SourceKind::Latin
        } else {
            SourceKind::Other
        }
    }

    /// Whether this kind must respect ASCII word boundaries when matching.
    fn word_bounded(self) -> bool {
        matches!(self, SourceKind::Latin)
    }
}

/// One reported substitution: `source` was replaced by `target` exactly `count`
/// times. Only terms that actually fired appear in an [`Applied`] report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Substitution {
    /// The matched source term.
    pub source: String,
    /// The replacement written in its place.
    pub target: String,
    /// How many non-overlapping occurrences were replaced.
    pub count: usize,
}

/// The result of [`Glossary::apply`]: the rewritten `output`, plus the audit
/// report of exactly what was substituted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Applied {
    /// The text after deterministic glossary substitution.
    pub output: String,
    /// Every term that fired, in glossary declaration order, with its exact
    /// occurrence count. Empty when nothing matched (a pure pass-through).
    pub substitutions: Vec<Substitution>,
}

impl Applied {
    /// Total occurrences replaced across all terms — the count of
    /// per-occurrence model decisions this pass eliminated (§13 Task III-5).
    #[must_use]
    pub fn total_substitutions(&self) -> usize {
        self.substitutions.iter().map(|s| s.count).sum()
    }
}

/// An ordered set of [`Entry`]s applied deterministically to text.
///
/// Declaration order is preserved (it is the tie-break for equal-length sources
/// and the order of the [`Applied::substitutions`] report); matching itself is
/// longest-source-first regardless of declaration order (rule 1).
///
/// Deserializable from `{"entries": [ {"source": ..., "target": ...}, ... ]}`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Glossary {
    #[serde(default)]
    entries: Vec<Entry>,
}

impl Glossary {
    /// Build a glossary from an ordered collection of entries.
    ///
    /// Entries with an empty `source` are rejected (an empty source would match
    /// at every position and cannot be a meaningful term).
    pub fn from_entries(entries: impl IntoIterator<Item = Entry>) -> Result<Self> {
        let entries: Vec<Entry> = entries.into_iter().collect();
        if let Some(bad) = entries.iter().position(|e| e.source.is_empty()) {
            return Err(anyhow!("glossary entry {bad} has an empty source term"));
        }
        Ok(Self { entries })
    }

    /// Parse a glossary from a simple line-oriented text format: one
    /// `source = target` per line, `#` line comments and blank lines ignored.
    /// The split is on the **first** `=`, so a target may itself contain `=`;
    /// both sides are trimmed of surrounding whitespace.
    ///
    /// This is a hand-rolled parser on purpose — it needs no `toml` dependency
    /// (this crate adds none for III-5). For structured config, deserialize
    /// [`Glossary`] / [`Entry`] from JSON via the existing `serde` dep instead.
    ///
    /// ```text
    /// # nuclear terminology
    /// 华龙一号 = Hualong One
    /// 数字孪生 = digital twin
    /// 压水堆   = PWR
    /// ```
    pub fn from_text(text: &str) -> Result<Self> {
        let mut entries = Vec::new();
        for (i, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (source, target) = line
                .split_once('=')
                .ok_or_else(|| anyhow!("glossary line {}: expected `source = target`, got {raw:?}", i + 1))?;
            let source = source.trim();
            if source.is_empty() {
                return Err(anyhow!("glossary line {}: empty source term", i + 1));
            }
            entries.push(Entry::new(source, target.trim()));
        }
        Ok(Self { entries })
    }

    /// The entries, in declaration order.
    #[must_use]
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// Whether the glossary has no entries (its [`Glossary::apply`] is a
    /// guaranteed pass-through).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Apply the glossary to `text` deterministically and report what changed.
    ///
    /// Single left-to-right pass, non-overlapping, longest-source-first, with
    /// per-entry CJK-substring vs Latin-word-boundary matching, advancing past
    /// consumed source so inserted targets are never re-scanned (module rules
    /// 1–3). Same `(glossary, text)` in ⇒ byte-identical `output` out, every
    /// run; an empty glossary is an exact pass-through.
    ///
    /// Latin sources match case-sensitively; a case-folding option is a future
    /// extension and is intentionally not enabled here (see module docs).
    #[must_use]
    pub fn apply(&self, text: &str) -> Applied {
        // Precompute each entry's source as chars + its matching kind, and the
        // order to try entries at a given position: longest source first, ties
        // by declaration order (stable) so the result is a pure function.
        let prepared: Vec<PreparedEntry> = self
            .entries
            .iter()
            .enumerate()
            .map(|(idx, e)| PreparedEntry {
                idx,
                source: e.source.chars().collect(),
                kind: SourceKind::of(&e.source),
            })
            .collect();

        let mut order: Vec<usize> = (0..prepared.len()).collect();
        // Stable sort by descending source length; equal lengths keep
        // declaration order.
        order.sort_by(|&a, &b| prepared[b].source.len().cmp(&prepared[a].source.len()));

        let chars: Vec<char> = text.chars().collect();
        let mut output = String::with_capacity(text.len());
        let mut counts = vec![0usize; self.entries.len()];

        let mut i = 0;
        while i < chars.len() {
            let mut matched = false;
            for &oi in &order {
                let pe = &prepared[oi];
                let len = pe.source.len();
                if len == 0 || i + len > chars.len() {
                    continue;
                }
                if chars[i..i + len] != pe.source[..] {
                    continue;
                }
                if pe.kind.word_bounded() && !word_boundaries_ok(&chars, i, len) {
                    continue;
                }
                // Match: write the target, advance past the consumed source.
                output.push_str(&self.entries[pe.idx].target);
                counts[pe.idx] += 1;
                i += len;
                matched = true;
                break;
            }
            if !matched {
                output.push(chars[i]);
                i += 1;
            }
        }

        // Report in declaration order, only terms that actually fired.
        let substitutions = self
            .entries
            .iter()
            .enumerate()
            .filter(|(idx, _)| counts[*idx] > 0)
            .map(|(idx, e)| Substitution {
                source: e.source.clone(),
                target: e.target.clone(),
                count: counts[idx],
            })
            .collect();

        Applied { output, substitutions }
    }
}

/// An [`Entry`] preprocessed for matching: its source as a char slice and its
/// [`SourceKind`], plus the declaration index to look the target/count up.
struct PreparedEntry {
    idx: usize,
    source: Vec<char>,
    kind: SourceKind,
}

/// True when the substring `chars[start..start+len]` is not glued to an ASCII
/// word character on either side — the Latin word-boundary rule. A boundary is
/// satisfied at the text edge, or against any non-`[A-Za-z0-9_]` char.
fn word_boundaries_ok(chars: &[char], start: usize, len: usize) -> bool {
    let left_ok = start == 0 || !is_word_char(chars[start - 1]);
    let after = start + len;
    let right_ok = after >= chars.len() || !is_word_char(chars[after]);
    left_ok && right_ok
}

/// ASCII "word" character for boundary purposes: `[A-Za-z0-9_]`.
fn is_word_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Whether `c` is a CJK ideograph in one of the common blocks, meaning a source
/// term containing it should match as a substring (no word boundaries in CJK).
///
/// Covers CJK Unified Ideographs and Extension A, the two Extension blocks in
/// the SIP, and the Compatibility Ideographs — the ranges that carry Chinese
/// technical terminology. Punctuation and fullwidth Latin are intentionally not
/// treated as CJK here.
fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x3400..=0x4DBF   // CJK Unified Ideographs Extension A
        | 0x4E00..=0x9FFF // CJK Unified Ideographs
        | 0xF900..=0xFAFF // CJK Compatibility Ideographs
        | 0x2_0000..=0x2_A6DF // Extension B
        | 0x2_A700..=0x2_EBEF // Extensions C–F
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The headline case: two CJK terms replaced in a sentence, with an exact
    /// audit report. This is the deterministic, zero-model-token measurement —
    /// 2 terms, 2 occurrences, all decided once and reproducibly.
    #[test]
    fn replaces_cjk_terms_in_a_sentence_and_reports_them() {
        let g = Glossary::from_entries([
            Entry::new("华龙一号", "Hualong One"),
            Entry::new("数字孪生", "digital twin"),
        ])
        .unwrap();

        let applied = g.apply("华龙一号的数字孪生模型");
        assert_eq!(applied.output, "Hualong One的digital twin模型");
        assert_eq!(
            applied.substitutions,
            vec![
                Substitution { source: "华龙一号".into(), target: "Hualong One".into(), count: 1 },
                Substitution { source: "数字孪生".into(), target: "digital twin".into(), count: 1 },
            ]
        );
        assert_eq!(applied.total_substitutions(), 2);
    }

    /// Rule 1: a longer term is never clobbered by a shorter prefix entry, even
    /// when the shorter entry is declared first.
    #[test]
    fn longest_source_wins_over_shorter_prefix() {
        let g = Glossary::from_entries([
            Entry::new("华龙", "Hualong"),      // 2-char prefix, declared first
            Entry::new("华龙一号", "Hualong One"), // 4-char full term
        ])
        .unwrap();

        // The full 4-char term must win where it appears...
        let applied = g.apply("华龙一号");
        assert_eq!(applied.output, "Hualong One");
        assert_eq!(applied.total_substitutions(), 1);
        assert_eq!(applied.substitutions[0].source, "华龙一号");

        // ...while the short entry still fires where only it matches.
        let applied2 = g.apply("华龙集团");
        assert_eq!(applied2.output, "Hualong集团");
    }

    /// Rule 2 (Latin): a Latin source respects word boundaries — `core` is
    /// replaced when it stands alone but not inside `encore`.
    #[test]
    fn latin_source_respects_word_boundaries() {
        let g = Glossary::from_entries([Entry::new("core", "reactor core")]).unwrap();

        let applied = g.apply("the core melted during the encore, core.");
        // Standalone `core` (x2) rewritten; the `core` inside `encore` is not.
        assert_eq!(applied.output, "the reactor core melted during the encore, reactor core.");
        assert_eq!(applied.total_substitutions(), 2);
        assert_eq!(applied.substitutions[0].count, 2);
    }

    /// Rule 2 (CJK): a CJK source matches as a substring with no boundary
    /// requirement — every occurrence, including adjacent runs, is replaced and
    /// counted.
    #[test]
    fn cjk_source_matches_every_occurrence() {
        let g = Glossary::from_entries([Entry::new("乏燃料", "spent fuel")]).unwrap();
        let applied = g.apply("乏燃料池储存乏燃料");
        assert_eq!(applied.output, "spent fuel池储存spent fuel");
        assert_eq!(applied.substitutions[0].count, 2);
    }

    /// Rule 3: applying twice yields byte-identical output to applying once
    /// (idempotence), and the same input always gives the same output.
    #[test]
    fn apply_is_idempotent_and_deterministic() {
        let g = Glossary::from_entries([
            Entry::new("压水堆", "PWR"),
            Entry::new("数字孪生", "digital twin"),
        ])
        .unwrap();
        let input = "压水堆的数字孪生，压水堆机组";

        let once = g.apply(input);
        let twice = g.apply(&once.output);
        assert_eq!(twice.output, once.output, "apply twice == apply once");
        assert!(twice.substitutions.is_empty(), "second pass substitutes nothing");

        // Determinism: a fresh glossary + same input reproduces byte-for-byte.
        let again = Glossary::from_entries([
            Entry::new("压水堆", "PWR"),
            Entry::new("数字孪生", "digital twin"),
        ])
        .unwrap();
        assert_eq!(again.apply(input).output, once.output);
    }

    /// An empty glossary is an exact pass-through with an empty report.
    #[test]
    fn empty_glossary_is_a_noop() {
        let g = Glossary::default();
        assert!(g.is_empty());
        let applied = g.apply("华龙一号 core 数字孪生");
        assert_eq!(applied.output, "华龙一号 core 数字孪生");
        assert!(applied.substitutions.is_empty());
        assert_eq!(applied.total_substitutions(), 0);
    }

    /// The report counts occurrences correctly across multiple terms and only
    /// lists terms that actually fired.
    #[test]
    fn report_counts_occurrences_and_omits_unfired_terms() {
        let g = Glossary::from_entries([
            Entry::new("压水堆", "PWR"),
            Entry::new("沸水堆", "BWR"), // never appears
            Entry::new("数字孪生", "digital twin"),
        ])
        .unwrap();

        let applied = g.apply("压水堆、压水堆、压水堆 和 数字孪生");
        assert_eq!(applied.substitutions.len(), 2, "unfired 沸水堆 omitted");
        assert_eq!(applied.substitutions[0].source, "压水堆");
        assert_eq!(applied.substitutions[0].count, 3);
        assert_eq!(applied.substitutions[1].source, "数字孪生");
        assert_eq!(applied.substitutions[1].count, 1);
    }

    #[test]
    fn source_kind_classification() {
        assert_eq!(SourceKind::of("华龙一号"), SourceKind::Cjk);
        assert_eq!(SourceKind::of("PWR"), SourceKind::Latin);
        assert_eq!(SourceKind::of("core"), SourceKind::Latin);
        // Mixed CJK + ASCII counts as CJK (substring rule is correct there).
        assert_eq!(SourceKind::of("华龙1号"), SourceKind::Cjk);
    }

    /// Latin matching is case-sensitive by default (documented behaviour).
    #[test]
    fn latin_matching_is_case_sensitive() {
        let g = Glossary::from_entries([Entry::new("PWR", "pressurised water reactor")]).unwrap();
        let applied = g.apply("PWR and pwr");
        assert_eq!(applied.output, "pressurised water reactor and pwr");
        assert_eq!(applied.total_substitutions(), 1);
    }

    /// The hand-rolled text format parses `source = target`, keeps `=` in the
    /// target, and ignores comments / blank lines.
    #[test]
    fn from_text_parses_source_equals_target() {
        let g = Glossary::from_text(
            "# nuclear terms\n\
             华龙一号 = Hualong One\n\
             \n\
             数字孪生 = digital twin\n\
             ratio = a = b\n",
        )
        .unwrap();
        assert_eq!(g.entries().len(), 3);
        assert_eq!(g.entries()[0], Entry::new("华龙一号", "Hualong One"));
        assert_eq!(g.entries()[2], Entry::new("ratio", "a = b"));
    }

    #[test]
    fn empty_source_is_rejected() {
        assert!(Glossary::from_entries([Entry::new("", "x")]).is_err());
        assert!(Glossary::from_text(" = target").is_err());
    }

    /// Compile-time check that the serde derives are wired: [`Glossary`],
    /// [`Entry`], [`Substitution`] and [`Applied`] are all `Serialize` +
    /// `Deserialize`, so a caller can load a glossary from JSON/TOML using its
    /// own serde-format crate (this crate adds none). Kept as a bound assertion
    /// rather than a live JSON round-trip so no `serde_json` dev-dependency is
    /// introduced.
    #[test]
    fn serde_derives_are_present() {
        fn assert_serde<T: serde::Serialize + serde::de::DeserializeOwned>() {}
        assert_serde::<Glossary>();
        assert_serde::<Entry>();
        assert_serde::<Substitution>();
        assert_serde::<Applied>();
        assert_serde::<SourceKind>();
    }
}

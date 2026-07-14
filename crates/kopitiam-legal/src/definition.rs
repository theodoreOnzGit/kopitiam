//! Defined terms, and resolving a word against the document's own dictionary.
//!
//! # Why this is the highest-value thing in the crate
//!
//! Legal instruments **redefine ordinary English**, and they do it on
//! purpose. When an Act says
//!
//! > `"dwelling-house"` includes a houseboat, a caravan and any structure
//! > occupied as a residence, whether or not affixed to land;
//!
//! then for the whole of that Act, a houseboat *is* a dwelling-house — and
//! your intuitions about what a dwelling-house is have been overridden by
//! the drafter. Every later provision that says "dwelling-house" means *that*.
//!
//! A naive extractor reads s 12, sees "dwelling-house", and hands the reader
//! back a sentence whose most important word means something other than what
//! it appears to mean. The reader, quite reasonably, applies the ordinary
//! meaning, and is wrong. **The extraction was verbatim and correct, and the
//! result was still misleading** — which is precisely why "just show the
//! text" is not sufficient and this module exists.
//!
//! So: when this crate surfaces a provision, it can also surface the
//! definitions that govern the words in it, with citations to where those
//! definitions are. It does **not** tell you what the provision means. It
//! tells you: *this word is defined, here is the definition, here is where it
//! lives, go and read it.*
//!
//! # Three things naive extractors get wrong, which are modelled here
//!
//! 1. **`means` vs `includes` are not the same word.** "means" is
//!    *exhaustive* — the definition replaces the ordinary meaning entirely.
//!    "includes" is *extensive* — the ordinary meaning survives and the
//!    definition adds to it. The legal consequences differ completely, and
//!    the difference is one word in the source. We record which connective
//!    the drafter used ([`DefinitionForce`]) and never normalise them
//!    together. We do not *interpret* the difference; we preserve it.
//!
//! 2. **Definitions have scope.** "In this Act, X means..." and "In this
//!    section, X means..." are different. A section-scoped definition
//!    overrides the Act-wide one *within that section only*. Resolution must
//!    therefore be relative to *where the word is used*, not global. See
//!    [`DefinitionScope`] and [`Dictionary::resolve`].
//!
//! 3. **Definitions are amended too.** A definition is a provision like any
//!    other, so it carries a [`Validity`] and resolution takes an
//!    [`AsAtDate`]. The definition of "dwelling-house" as at 2019 may not be
//!    the definition as at 2024.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{AsAtDate, Numeral, Provenance, Provision, ProvisionId, Validity};

/// The connective the drafter used, which determines whether the definition
/// *replaces* or *extends* the ordinary meaning.
///
/// We record it; we do not act on it. The distinction between "means" and
/// "includes" carries real legal consequence, and drawing that consequence is
/// the reader's job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DefinitionForce {
    /// `"X" means ...` — exhaustive. The definition displaces the ordinary
    /// meaning entirely.
    Means,
    /// `"X" includes ...` — extensive. The ordinary meaning survives; the
    /// definition adds to it. **Critically, an `includes` definition does
    /// not tell you the full extent of the term.**
    Includes,
    /// `"X" means ... and includes ...` — a hybrid, common in practice.
    MeansAndIncludes,
    /// `"X" does not include ...` — a carve-out.
    DoesNotInclude,
    /// `"X" is deemed to be ...` — a deeming provision, which makes
    /// something true *as a matter of law* that is not true as a matter of
    /// fact. Legally distinct from a definition and flagged separately.
    Deemed,
}

impl fmt::Display for DefinitionForce {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Means => "means",
            Self::Includes => "includes",
            Self::MeansAndIncludes => "means and includes",
            Self::DoesNotInclude => "does not include",
            Self::Deemed => "is deemed to be",
        })
    }
}

/// Over what part of the instrument a definition governs.
///
/// Narrower scopes win. "In this section, 'premises' means..." beats "In this
/// Act, 'premises' means..." — but only inside that section.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DefinitionScope {
    /// "In this Act", "In this Agreement" — governs everywhere.
    Instrument,
    /// "In this Part" — governs within one Part of the instrument.
    ///
    /// A separate variant rather than a [`Self::Within`] over a Part id,
    /// because a Part is **not** part of a provision's identity: `s 7` is
    /// `s 7` wherever it sits, and section numbers run uniquely across the
    /// whole Act. See [`crate::ProvisionId`]. A Part is *context* attached to
    /// a provision, so Part scope is checked against that context rather than
    /// by prefix-matching an id.
    Part(Numeral),
    /// "In this section" / "In this clause" — governs only within the named
    /// unit and everything inside it.
    Within(ProvisionId),
}

impl DefinitionScope {
    /// Whether this scope governs a word used at `used_in`, which sits in
    /// `in_part`.
    ///
    /// `in_part` is a separate argument and not optional-by-omission for the
    /// reason above: a provision's Part is context carried alongside its id,
    /// not encoded in it.
    pub fn governs(&self, used_in: &ProvisionId, in_part: Option<Numeral>) -> bool {
        match self {
            Self::Instrument => true,
            Self::Part(part) => in_part == Some(*part),
            Self::Within(scope) => scope.contains(used_in),
        }
    }

    /// How narrow this scope is. Larger = narrower = higher priority, so a
    /// section-scoped definition beats a Part-scoped one, which beats an
    /// instrument-wide one.
    ///
    /// A Part sits between the two: it is narrower than the whole instrument
    /// but wider than any single section. Ranking it at 1 and starting
    /// section-rooted ids at 2 keeps that ordering strict.
    fn specificity(&self) -> usize {
        match self {
            Self::Instrument => 0,
            Self::Part(_) => 1,
            Self::Within(id) => 1 + id.components().len(),
        }
    }
}

impl fmt::Display for DefinitionScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Instrument => f.write_str("throughout the instrument"),
            Self::Part(n) => write!(f, "within Part {n}"),
            Self::Within(id) => write!(f, "within {id}"),
        }
    }
}

/// A term the instrument defines for itself.
///
/// Carries full [`Provenance`] (so the reader can go and check) and a
/// [`Validity`] (because definitions get amended).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Definition {
    /// The defined term as the drafter wrote it, e.g. `dwelling-house`.
    term: String,
    force: DefinitionForce,
    /// The definition's body — the words after the connective, verbatim.
    body: String,
    scope: DefinitionScope,
    validity: Validity,
    provenance: Provenance,
}

impl Definition {
    pub fn new(
        term: impl Into<String>,
        force: DefinitionForce,
        body: impl Into<String>,
        scope: DefinitionScope,
        validity: Validity,
        provenance: Provenance,
    ) -> Self {
        Self {
            term: term.into(),
            force,
            body: body.into(),
            scope,
            validity,
            provenance,
        }
    }

    pub fn term(&self) -> &str {
        &self.term
    }

    pub fn force(&self) -> DefinitionForce {
        self.force
    }

    pub fn body(&self) -> &str {
        &self.body
    }

    pub fn scope(&self) -> &DefinitionScope {
        &self.scope
    }

    pub fn validity(&self) -> Validity {
        self.validity
    }

    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }

    /// The whole definition as the document states it, verbatim. This is what
    /// a reader should be shown — not a gloss of it.
    pub fn verbatim(&self) -> &str {
        self.provenance.verbatim()
    }
}

impl fmt::Display for Definition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "\"{}\" {} {} [{}, {}]\n  {}",
            self.term,
            self.force,
            self.body,
            self.scope,
            self.validity,
            self.provenance.citation()
        )
    }
}

/// The outcome of looking a word up in the instrument's own dictionary.
#[derive(Debug, Clone, PartialEq)]
pub enum Resolution<'a> {
    /// The instrument defines this term, and this definition governs at the
    /// place and date asked about.
    Defined(&'a Definition),

    /// The instrument does **not** define this term at this place and date.
    ///
    /// Note what we do *not* do here: we do not supply an ordinary-English
    /// meaning. This crate is not a dictionary and has no business telling
    /// anyone what an undefined word means — that is construction, and it is
    /// the reader's job. "The instrument does not define this" is the whole
    /// answer.
    NotDefined,

    /// **Two or more definitions compete and we will not choose between
    /// them.** Same term, same scope, both in force. Which governs is a
    /// question of legal construction. We hand back all of them, with
    /// citations, and stop.
    Conflicting(Vec<&'a Definition>),

    /// The term *is* defined in the instrument, but not in a way that
    /// governs here — e.g. it is defined "in this section" and you asked
    /// about a different section, or the definition was not yet in force on
    /// your date. The out-of-scope definitions are returned because their
    /// existence is a signal worth seeing.
    DefinedButNotHere(Vec<&'a Definition>),
}

/// The instrument's own dictionary, and the resolution logic over it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Dictionary {
    definitions: Vec<Definition>,
}

impl Dictionary {
    pub fn new(definitions: Vec<Definition>) -> Self {
        Self { definitions }
    }

    pub fn definitions(&self) -> &[Definition] {
        &self.definitions
    }

    pub fn insert(&mut self, definition: Definition) {
        self.definitions.push(definition);
    }

    pub fn is_empty(&self) -> bool {
        self.definitions.is_empty()
    }

    /// **Resolve a word against the instrument's own definitions**, as at a
    /// date, at the place it is used.
    ///
    /// Both extra arguments are mandatory and neither is ceremony:
    ///
    /// * `used_in` — because a section-scoped definition governs only inside
    ///   that section. Resolving globally would apply a definition where it
    ///   has no force.
    /// * `as_at` — because definitions are amended, and the definition of a
    ///   term in 2019 need not be its definition in 2024.
    ///
    /// Narrower scope wins. Ties do **not** get broken by guessing — they
    /// come back as [`Resolution::Conflicting`].
    pub fn resolve(
        &self,
        term: &str,
        used_in: &ProvisionId,
        in_part: Option<Numeral>,
        as_at: AsAtDate,
    ) -> Resolution<'_> {
        let matching: Vec<&Definition> = self
            .definitions
            .iter()
            .filter(|d| terms_match(d.term(), term))
            .collect();

        if matching.is_empty() {
            return Resolution::NotDefined;
        }

        let governing: Vec<&Definition> = matching
            .iter()
            .copied()
            .filter(|d| d.scope().governs(used_in, in_part) && d.validity().covers(as_at))
            .collect();

        if governing.is_empty() {
            // The term exists in the instrument but does not reach here.
            // Saying so is more useful than saying "not defined", because it
            // tells the reader there IS a definition and where to look.
            return Resolution::DefinedButNotHere(matching);
        }

        // Narrowest scope wins: "in this section" beats "in this Act".
        let narrowest = governing
            .iter()
            .map(|d| d.scope().specificity())
            .max()
            .expect("governing is non-empty");
        let winners: Vec<&Definition> = governing
            .into_iter()
            .filter(|d| d.scope().specificity() == narrowest)
            .collect();

        match winners.len() {
            1 => Resolution::Defined(winners[0]),
            // Same term, same scope, same date, different words. Which one
            // governs is construction, not parsing. We refuse to pick.
            _ => Resolution::Conflicting(winners),
        }
    }

    /// Finds every defined term occurring in `text`, and resolves each.
    ///
    /// This is what turns "here is the verbatim provision" into "here is the
    /// verbatim provision, and by the way these four words in it do not mean
    /// what you think they mean — here are the definitions".
    ///
    /// Matching is whole-word and case-insensitive. It is deliberately
    /// *simple*: it does not stem, lemmatise, or attempt to match
    /// inflections, because a false positive here would attach a definition
    /// to a word the drafter did not define, which is a way of being wrong
    /// that looks like being helpful. Under-matching is recoverable (the
    /// reader still has the verbatim text); over-matching is not.
    pub fn terms_used_in(
        &self,
        text: &str,
        used_in: &ProvisionId,
        in_part: Option<Numeral>,
        as_at: AsAtDate,
    ) -> Vec<TermOccurrence<'_>> {
        let lower = text.to_lowercase();
        let mut out = Vec::new();
        for definition in &self.definitions {
            let needle = definition.term().to_lowercase();
            if needle.is_empty() {
                continue;
            }
            for (start, _) in lower.match_indices(&needle) {
                if !is_whole_word(&lower, start, needle.len()) {
                    continue;
                }
                if let Resolution::Defined(resolved) =
                    self.resolve(definition.term(), used_in, in_part, as_at)
                {
                    out.push(TermOccurrence {
                        term: definition.term().to_string(),
                        start,
                        end: start + needle.len(),
                        definition: resolved,
                    });
                }
            }
        }
        // Deterministic order: by position, then by term.
        out.sort_by(|a, b| a.start.cmp(&b.start).then_with(|| a.term.cmp(&b.term)));
        // A term matched via one Definition may resolve to another (a
        // narrower one). Dedupe identical spans.
        out.dedup_by(|a, b| a.start == b.start && a.end == b.end);
        out
    }
}

/// One occurrence of a defined term inside a provision's text, with the
/// definition that governs it there and then.
#[derive(Debug, Clone, PartialEq)]
pub struct TermOccurrence<'a> {
    term: String,
    /// Byte offsets into the provision's verbatim text.
    start: usize,
    end: usize,
    definition: &'a Definition,
}

impl TermOccurrence<'_> {
    pub fn term(&self) -> &str {
        &self.term
    }

    pub fn span(&self) -> (usize, usize) {
        (self.start, self.end)
    }

    pub fn definition(&self) -> &Definition {
        self.definition
    }
}

/// Whether two term spellings refer to the same defined term.
///
/// Case-insensitive, and tolerant of the hyphen/space variation that legal
/// drafting is inconsistent about ("dwelling-house" vs "dwelling house").
/// Nothing more aggressive than that: see [`Dictionary::terms_used_in`] for
/// why over-matching is the dangerous direction.
fn terms_match(a: &str, b: &str) -> bool {
    let normalise = |s: &str| s.to_lowercase().replace(['-', '\u{2010}', '\u{2011}'], " ");
    normalise(a).split_whitespace().eq(normalise(b).split_whitespace())
}

/// Whether the match at `[start, start+len)` is bounded by non-word
/// characters — so "widget" does not match inside "widgetry".
fn is_whole_word(haystack: &str, start: usize, len: usize) -> bool {
    let before_ok = haystack[..start]
        .chars()
        .next_back()
        .is_none_or(|c| !c.is_alphanumeric() && c != '_');
    let after_ok = haystack[start + len..]
        .chars()
        .next()
        .is_none_or(|c| !c.is_alphanumeric() && c != '_');
    before_ok && after_ok
}

/// Extracts definitions from a provision's text.
///
/// Recognises the standard Commonwealth drafting forms, in which the defined
/// term is **quoted** and followed by a connective:
///
/// ```text
/// "dwelling-house" means a building occupied as a residence;
/// "vehicle" includes a bicycle;
/// "premises" means and includes any land or building;
/// ```
///
/// Straight and curly quotes are both accepted, because PDFs contain both.
///
/// A quoted term with **no recognised connective** is *not* silently treated
/// as a definition — it is simply not extracted, because a quoted phrase in
/// legal text is very often a citation or a term of art, not a definition.
/// Callers that want to know about near-misses should look at the anomalies.
pub fn extract_definitions(
    provision: &Provision,
    scope: DefinitionScope,
) -> Vec<Definition> {
    use regex::Regex;
    use std::sync::LazyLock;

    // Ordered longest-connective-first so that "means and includes" is not
    // matched as a bare "means" with a body starting "and includes".
    static DEFINITION: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r#"(?i)["\u{201C}]([^"\u{201C}\u{201D}]+)["\u{201D}]\s*(means and includes|does not include|is deemed to be|means|includes)\s+([^;]+)"#,
        )
        .expect("definition regex is a compile-time constant")
    });

    DEFINITION
        .captures_iter(provision.text())
        .filter_map(|caps| {
            let term = caps.get(1)?.as_str().trim();
            let connective = caps.get(2)?.as_str();
            let body = caps.get(3)?.as_str().trim();
            let force = match connective.to_lowercase().as_str() {
                "means" => DefinitionForce::Means,
                "includes" => DefinitionForce::Includes,
                "means and includes" => DefinitionForce::MeansAndIncludes,
                "does not include" => DefinitionForce::DoesNotInclude,
                "is deemed to be" => DefinitionForce::Deemed,
                _ => return None,
            };
            Some(Definition::new(
                term,
                force,
                body,
                scope.clone(),
                provision.validity(),
                provision.provenance().clone(),
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numbering::parse_statutory;
    use crate::synthetic::{at, synthetic_provision};

    #[test]
    fn scope_narrowness_orders_instrument_then_part_then_section() {
        // A section-scoped definition must beat a Part-scoped one, which must
        // beat an instrument-wide one.
        let instrument = DefinitionScope::Instrument.specificity();
        let part = DefinitionScope::Part(Numeral::new(2, crate::NumeralStyle::UpperRoman)).specificity();
        let section = DefinitionScope::Within(parse_statutory("12").unwrap()).specificity();
        let subsection = DefinitionScope::Within(parse_statutory("12(3)").unwrap()).specificity();
        assert!(instrument < part && part < section && section < subsection);
    }

    #[test]
    fn instrument_scope_governs_everywhere_section_scope_does_not() {
        let s12 = parse_statutory("12").unwrap();
        let s13 = parse_statutory("13").unwrap();
        assert!(DefinitionScope::Instrument.governs(&s12, None));
        assert!(DefinitionScope::Within(s12.clone())
            .governs(&parse_statutory("12(3)").unwrap(), None));
        assert!(!DefinitionScope::Within(s12).governs(&s13, None));
    }

    #[test]
    fn part_scope_is_checked_against_the_provisions_part_context_not_its_id() {
        // A Part is not part of a provision's identity (s 7 is s 7 wherever it
        // sits), so Part scope is matched against the context carried
        // alongside the id.
        let part_ii = Numeral::new(2, crate::NumeralStyle::UpperRoman);
        let part_iii = Numeral::new(3, crate::NumeralStyle::UpperRoman);
        let s12 = parse_statutory("12").unwrap();
        let scope = DefinitionScope::Part(part_ii);
        assert!(scope.governs(&s12, Some(part_ii)));
        assert!(!scope.governs(&s12, Some(part_iii)));
        assert!(!scope.governs(&s12, None));
    }

    #[test]
    fn extracts_means_and_includes_as_distinct_forces() {
        let p = synthetic_provision(
            "2",
            r#""dwelling-house" includes a houseboat; "vehicle" means a mechanically propelled conveyance;"#,
            2020,
        );
        let defs = extract_definitions(&p, DefinitionScope::Instrument);
        assert_eq!(defs.len(), 2);
        assert_eq!(defs[0].term(), "dwelling-house");
        assert_eq!(
            defs[0].force(),
            DefinitionForce::Includes,
            "'includes' is extensive and must not be normalised into 'means'"
        );
        assert_eq!(defs[1].term(), "vehicle");
        assert_eq!(defs[1].force(), DefinitionForce::Means);
    }

    #[test]
    fn a_quoted_phrase_with_no_connective_is_not_a_definition() {
        let p = synthetic_provision("5", r#"The court considered "reasonable care" at length."#, 2020);
        assert!(
            extract_definitions(&p, DefinitionScope::Instrument).is_empty(),
            "a quoted term of art is not a definition"
        );
    }

    #[test]
    fn terms_match_tolerates_the_hyphen_space_inconsistency() {
        assert!(terms_match("dwelling-house", "Dwelling House"));
        assert!(terms_match("vehicle", "VEHICLE"));
        assert!(!terms_match("vehicle", "vehicles"), "no stemming: over-matching is the dangerous direction");
    }

    #[test]
    fn whole_word_matching_only() {
        let dict = Dictionary::new(vec![Definition::new(
            "widget",
            DefinitionForce::Means,
            "a thing",
            DefinitionScope::Instrument,
            crate::Validity::from(crate::Date::new(2020, 1, 1).unwrap()),
            synthetic_provision("2", "x", 2020).provenance().clone(),
        )]);
        let used_in = parse_statutory("12").unwrap();
        let hits = dict.terms_used_in("widgetry is not a widget", &used_in, None, at(2021));
        assert_eq!(hits.len(), 1, "'widgetry' must not match 'widget'");
        assert_eq!(hits[0].span(), (18, 24), "the SECOND 'widget', not the one inside 'widgetry'");
    }
}

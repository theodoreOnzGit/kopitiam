//! Definitions — the load-bearing part of an insurance policy, and the part
//! naive extractors miss.
//!
//! An insurance policy is a private dictionary that happens to also be a
//! contract. It takes ordinary words — *accident*, *hospital*, *spouse*,
//! *pre-existing*, *permanent*, *loss* — and **redefines them**, in a
//! definitions section, and those definitions then override plain meaning
//! everywhere else in the document.
//!
//! This is not a technicality. It is where the money is. A policy that pays on
//! "Accident" and defines *Accident* as "a sudden, violent, external and
//! visible event" does not pay for a back injury that developed over months,
//! however plainly accidental that injury seems in English. A reader — or an
//! extractor — that takes the word at face value gets the answer exactly
//! backwards, and does so *confidently*, which is the dangerous way to be
//! wrong.
//!
//! Hence:
//!
//! * The definitions section is a **first-class thing**, located structurally
//!   (see [`is_definitions_heading`]), not a bag of key-value pairs scraped
//!   from anywhere in the text.
//! * A defined term appearing in *any* clause is **resolved against the
//!   policy's own definition** ([`Definitions::occurrences_in`]), so a
//!   consumer can never accidentally read the plain meaning.
//! * A term the policy defines **twice, inconsistently**, resolves to
//!   [`Resolution::Conflicting`] — not to "the first one" and not to
//!   "probably this one". Both definitions are handed back with their
//!   citations, and the reader decides. Policies really do contain
//!   conflicting definitions (typically after an endorsement adds one), and
//!   silently picking a winner is a decision this crate has no business
//!   making.
//! * A term the policy does *not* define resolves to [`Resolution::Undefined`]
//!   — which is itself useful information, because it tells the reader that
//!   plain meaning (and, in a dispute, ordinary rules of construction) applies
//!   here.
//!
//! [`Resolution`] is deliberately **not** `Option<&Definition>`. Collapsing
//! "the policy contradicts itself" into "no answer" — or worse, into an
//! arbitrary answer — is the single easiest way to turn this crate from a tool
//! into a hazard.

use std::collections::BTreeMap;
use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::clause::{Clause, ClauseRole};
use crate::provenance::{ExtractedTerm, Provenance, ProvenanceError};

/// Headings that introduce a definitions section, matched case-insensitively
/// as substrings. Insurance drafting is conventional enough that this short
/// list covers the overwhelming majority of wordings; anything it misses shows
/// up as a *missing* definition (safe: the term resolves to `Undefined`, and
/// the reader is told plain meaning applies) rather than a wrong one.
const DEFINITIONS_HEADINGS: &[&str] = &[
    "definition",
    "definitions and interpretation",
    "interpretation",
    "glossary",
    "meaning of words",
    "words and phrases",
    "defined terms",
];

/// Phrases that introduce a term's meaning, longest first so that
/// `"shall mean"` wins over `"mean"`. These are the conventional operative
/// verbs of legal drafting; a definitions section will use one of them.
const DEFINING_PHRASES: &[&str] = &[
    "shall have the meaning",
    "has the meaning given",
    "is defined as",
    "shall be defined as",
    "shall mean and include",
    "shall mean",
    "means and includes",
    "means:",
    "means",
    "refers to",
];

/// Separators used by a glossary layout (a two-column table, or a hanging
/// list) instead of an operative verb: `Accident — a sudden, violent event`.
/// `" | "` is how [`crate::ingest`] renders a table row into clause text.
const DEFINING_SEPARATORS: &[&str] = &[" | ", " — ", " – ", " -- ", ": "];

/// Column headers and boilerplate that a glossary *table* puts in its header
/// row. Without this guard a table header (`Term | Meaning`) parses perfectly
/// well as a definition of the word "Term", and the policy acquires a
/// definition it never made.
const GLOSSARY_HEADER_WORDS: &[&str] = &[
    "term",
    "terms",
    "word",
    "words",
    "defined term",
    "meaning",
    "meanings",
    "definition",
    "definitions",
    "description",
];

/// The longest a defined term may be, in words. `"Pre-existing Condition"` and
/// `"Period of Insurance"` are real defined terms; a twenty-word "term" means
/// we matched the word "means" in ordinary prose ("this means that you must
/// tell us...") and mistook a sentence for a definition.
const MAX_TERM_WORDS: usize = 8;

/// A term the policy defines, and what **this policy** says it means.
///
/// The meaning is an [`ExtractedTerm`], so it is inseparable from the page,
/// clause and verbatim wording it came from. A definition without a citation
/// would be an assertion about a legal contract with nothing behind it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Definition {
    term: String,
    meaning: ExtractedTerm<String>,
}

impl Definition {
    /// The defined term, as printed (`"Pre-existing Condition"`).
    pub fn term(&self) -> &str {
        &self.term
    }

    /// The policy's meaning for the term, verbatim.
    pub fn meaning(&self) -> &str {
        self.meaning.value()
    }

    /// Where the definition is printed.
    pub fn provenance(&self) -> &Provenance {
        self.meaning.provenance()
    }

    /// The meaning, with its citation attached — for a consumer that needs to
    /// carry the two together.
    pub fn extracted(&self) -> &ExtractedTerm<String> {
        &self.meaning
    }

    /// Lookup key: the term, case-folded and whitespace-normalised.
    fn key(&self) -> String {
        normalise_term(&self.term)
    }
}

/// What **this policy** says a word means.
///
/// Not `Option<&Definition>`, and the reason matters: see the module docs.
#[derive(Debug, Clone, PartialEq)]
pub enum Resolution<'a> {
    /// The policy defines this term. Its meaning here is the policy's, and it
    /// **overrides the plain English meaning** wherever the term appears.
    Defined(&'a Definition),

    /// The policy defines this term **more than once, inconsistently**. There
    /// is no safe way to pick one. Every definition is returned, with its
    /// citation, so the reader can see the contradiction and judge it.
    Conflicting(Vec<&'a Definition>),

    /// The policy does not define this term, so plain meaning applies. Saying
    /// so explicitly is the point: it tells the reader that no policy-specific
    /// meaning is hiding here.
    Undefined,
}

impl<'a> Resolution<'a> {
    /// The single definition, when there is exactly one.
    ///
    /// Returns `None` for [`Resolution::Conflicting`] as well as
    /// [`Resolution::Undefined`], deliberately: a caller that wants an answer
    /// from a self-contradicting policy has to look the contradiction in the
    /// eye and match on it.
    pub fn definition(&self) -> Option<&'a Definition> {
        match self {
            Self::Defined(definition) => Some(definition),
            Self::Conflicting(_) | Self::Undefined => None,
        }
    }

    /// Whether the policy assigns this term a meaning of its own (whether or
    /// not that meaning is self-consistent). If this is `true`, plain English
    /// is not the answer.
    pub fn is_policy_defined(&self) -> bool {
        matches!(self, Self::Defined(_) | Self::Conflicting(_))
    }
}

/// One appearance of a defined term inside a clause, resolved against the
/// policy's definitions section.
///
/// This is what turns a definitions section from a list into a *mechanism*: it
/// says "the word at bytes 42..50 of clause 3.1 is not the English word
/// 'accident', it is the policy's `Accident`, which is defined on page 2 to
/// mean ..." — and it hands over the citation for both.
#[derive(Debug, Clone, PartialEq)]
pub struct TermOccurrence<'a> {
    /// The term as the clause printed it (`"Accidents"` — possibly inflected).
    pub surface: String,
    /// Byte range of the occurrence within [`Clause::text`].
    pub range: Range<usize>,
    /// What the policy says it means.
    pub resolution: Resolution<'a>,
}

/// The policy's definitions section: every term it defines, and where.
///
/// Built by [`Definitions::extract`] from the clauses that sit under a
/// definitions heading. Ordered (a `BTreeMap`), so extraction is deterministic
/// run to run — a hashed iteration order would make the emitted knowledge
/// graph differ between runs on identical input, which the Semantic Runtime's
/// reproducibility principle forbids.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Definitions {
    by_term: BTreeMap<String, Vec<Definition>>,
}

impl Definitions {
    /// Extracts every definition from the clauses that carry
    /// [`ClauseRole::Definition`].
    ///
    /// # Errors
    ///
    /// Propagates a [`ProvenanceError`] if a parsed meaning cannot be cited
    /// back to its own clause. That should be impossible by construction (the
    /// meaning is a substring of the clause text), and an error here would
    /// mean the extractor had produced text the document does not contain —
    /// exactly the failure mode the provenance model exists to catch, so it is
    /// surfaced rather than swallowed.
    pub fn extract<'a>(
        clauses: impl IntoIterator<Item = &'a Clause>,
    ) -> Result<Self, ProvenanceError> {
        let mut by_term: BTreeMap<String, Vec<Definition>> = BTreeMap::new();

        for clause in clauses {
            if clause.role() != ClauseRole::Definition {
                continue;
            }
            // The definitions section's own *banner* ("Section 2 —
            // Definitions") is a clause with role `Definition`, and it parses
            // beautifully as a definition of the term "Section 2" meaning
            // "Definitions". It is the heading, not a definition. Skip it.
            if clause.heading().is_some_and(is_definitions_heading) {
                continue;
            }
            for definition in parse_definitions(clause)? {
                by_term.entry(definition.key()).or_default().push(definition);
            }
        }

        Ok(Self { by_term })
    }

    /// What this policy says `term` means.
    pub fn resolve(&self, term: &str) -> Resolution<'_> {
        let Some(definitions) = self.by_term.get(&normalise_term(term)) else {
            return Resolution::Undefined;
        };

        match definitions.as_slice() {
            [] => Resolution::Undefined,
            [only] => Resolution::Defined(only),
            many => {
                // Defined twice with the *same* wording is a duplicate, not a
                // conflict — a policy that repeats itself verbatim has still
                // said only one thing, and reporting that as a contradiction
                // would be crying wolf.
                let first = many[0].meaning();
                if many.iter().all(|d| d.meaning() == first) {
                    Resolution::Defined(&many[0])
                } else {
                    Resolution::Conflicting(many.iter().collect())
                }
            }
        }
    }

    /// Every defined term, in a stable order.
    pub fn iter(&self) -> impl Iterator<Item = &Definition> {
        self.by_term.values().flatten()
    }

    /// How many distinct terms the policy defines.
    pub fn len(&self) -> usize {
        self.by_term.len()
    }

    /// Whether the policy defines nothing.
    ///
    /// True for a schedule or an endorsement (which normally carry no
    /// definitions of their own and inherit the wording's) — and a red flag
    /// for a document classified as a policy wording, which should have a
    /// definitions section. See [`crate::PolicyDocument::anomalies`].
    pub fn is_empty(&self) -> bool {
        self.by_term.is_empty()
    }

    /// Terms this policy defines inconsistently, with all their definitions.
    pub fn conflicts(&self) -> Vec<(&str, Vec<&Definition>)> {
        self.by_term
            .iter()
            .filter_map(|(key, definitions)| match self.resolve(key) {
                Resolution::Conflicting(_) => {
                    Some((definitions[0].term(), definitions.iter().collect()))
                }
                _ => None,
            })
            .collect()
    }

    /// **The mechanism.** Every defined term appearing in `clause`, located and
    /// resolved to the policy's own meaning.
    ///
    /// Matching is case-insensitive and word-bounded, and tolerates a regular
    /// plural (`Accident` matches `Accidents`). It does **not** attempt wider
    /// morphology — `Accident` will not match `Accidental` — because a false
    /// positive here silently substitutes the policy's meaning for a word the
    /// policy did not define, which is a way of being wrong that looks exactly
    /// like being right.
    ///
    /// Where two defined terms overlap (`Hospital` and `Hospital Cash
    /// Benefit`), the **longest** match at a position wins: the policy defined
    /// the longer phrase as a unit, so reading its first word as a separate
    /// defined term would be a misreading.
    ///
    /// Occurrences come back in ascending position order.
    pub fn occurrences_in(&self, clause: &Clause) -> Vec<TermOccurrence<'_>> {
        let text = clause.text();
        // ASCII case folding, not `to_lowercase()`, because the byte offsets
        // found in `haystack` are used to slice `text`. Full Unicode lowercasing
        // can change a string's byte length (`İ` -> `i̇`), which would silently
        // shift every subsequent offset and cite the wrong words — the one thing
        // this crate must never do. The cost is that a defined term containing
        // non-ASCII letters will not case-fold, so an occurrence may be *missed*.
        // A miss is safe (the term resolves to `Undefined` and the reader is told
        // plain meaning applies); a shifted citation is not.
        let haystack = text.to_ascii_lowercase();

        // Longest term first, so an overlap resolves in favour of the longer
        // defined phrase.
        let mut terms: Vec<&String> = self.by_term.keys().collect();
        terms.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));

        let mut occurrences: Vec<TermOccurrence<'_>> = Vec::new();
        let mut claimed: Vec<Range<usize>> = Vec::new();

        for term in terms {
            for (start, _) in haystack.match_indices(term.as_str()) {
                let Some(end) = word_bounded_end(&haystack, start, term.len()) else {
                    continue;
                };
                let range = start..end;
                if claimed
                    .iter()
                    .any(|taken| range.start < taken.end && taken.start < range.end)
                {
                    continue;
                }
                // Belt and braces: `text.get` rather than `text[range]`, so an
                // offset that somehow landed off a character boundary drops the
                // occurrence instead of panicking mid-ingestion.
                let Some(surface) = text.get(range.clone()) else {
                    continue;
                };
                claimed.push(range.clone());
                occurrences.push(TermOccurrence {
                    surface: surface.to_string(),
                    range,
                    resolution: self.resolve(term),
                });
            }
        }

        occurrences.sort_by_key(|occurrence| occurrence.range.start);
        occurrences
    }
}

/// Whether a heading introduces the definitions section.
pub fn is_definitions_heading(heading: &str) -> bool {
    let lower = heading.to_lowercase();
    DEFINITIONS_HEADINGS
        .iter()
        .any(|marker| lower.contains(marker))
}

/// Case-folds and whitespace-normalises a term for lookup, so that
/// `"Pre-existing  Condition"` and `"pre-existing condition"` are the same key.
fn normalise_term(term: &str) -> String {
    term.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

/// Whether the match of `len` bytes at `start` in `haystack` is a whole word,
/// returning the end offset (which may extend past `start + len` to swallow a
/// regular plural). `None` if it is not a word-bounded match.
fn word_bounded_end(haystack: &str, start: usize, len: usize) -> Option<usize> {
    let bytes = haystack.as_bytes();

    let starts_word = start == 0 || !is_word_byte(bytes[start - 1]);
    if !starts_word {
        return None;
    }

    let end = start + len;
    if end == bytes.len() || !is_word_byte(bytes[end]) {
        return Some(end);
    }

    // A regular plural is the same defined term: "Accidents" is "Accident".
    for suffix in ["es", "s"] {
        let plural_end = end + suffix.len();
        if haystack[end..].starts_with(suffix)
            && (plural_end == bytes.len() || !is_word_byte(bytes[plural_end]))
        {
            return Some(plural_end);
        }
    }

    None
}

fn is_word_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-'
}

/// Parses every definition stated by one definitions-section clause.
///
/// Handles the three layouts a definitions section actually uses:
///
/// 1. **Quoted term**: `"Accident" means a sudden, violent event.`
/// 2. **Operative verb**: `Accident means a sudden, violent event.`
/// 3. **Glossary separator**: `Accident — a sudden, violent event.`, which is
///    also how a two-column definitions *table* arrives (see
///    [`crate::ingest`], which renders a table row as `cell | cell`).
///
/// A clause's lines are walked in order: a line that opens a definition starts
/// one; a line that does not is treated as a **continuation** of the definition
/// above it, because a definition's meaning routinely runs to several lines.
fn parse_definitions(clause: &Clause) -> Result<Vec<Definition>, ProvenanceError> {
    // (term, meaning-so-far) — meaning is grown by continuation lines, and
    // stays a contiguous substring of the clause text as it grows, so it can
    // still be cited.
    let mut drafts: Vec<(String, String)> = Vec::new();

    for line in clause.text().split('\n') {
        // Strip the clause's own printed number before parsing. Without this,
        // `2.1 "Accident" means ...` defines the term `2.1 "Accident"` — and
        // every lookup of `Accident` then comes back `Undefined`, silently
        // handing the reader the plain English meaning. The *meaning* is
        // unaffected by the strip, so it remains a verbatim substring of the
        // clause and stays citable.
        let line = crate::ingest::split_clause_number(line)
            .map(|(_, rest)| rest)
            .unwrap_or(line);

        match split_definition(line) {
            Some((term, meaning)) => drafts.push((term.to_string(), meaning.to_string())),
            None => {
                if let Some((_, meaning)) = drafts.last_mut() {
                    meaning.push('\n');
                    meaning.push_str(line.trim());
                }
            }
        }
    }

    drafts
        .into_iter()
        .map(|(term, meaning)| {
            let cited = clause.extract(meaning.clone(), &meaning)?;
            Ok(Definition {
                term,
                meaning: cited,
            })
        })
        .collect()
}

/// Splits one line into `(term, meaning)`, or `None` if it does not state a
/// definition.
fn split_definition(line: &str) -> Option<(&str, &str)> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }

    let (term, meaning) = split_quoted(line)
        .or_else(|| split_on_defining_phrase(line))
        .or_else(|| split_on_separator(line))?;

    let term = term.trim().trim_matches(|c: char| c == ':' || c == ',');
    let meaning = meaning.trim();

    if term.is_empty() || meaning.is_empty() {
        return None;
    }
    if term.split_whitespace().count() > MAX_TERM_WORDS {
        return None;
    }
    // A glossary table's header row (`Term | Meaning`) and a section banner
    // (`Section 2 — Definitions`) both parse beautifully as definitions — of
    // the words "Term" and "Section 2" respectively. Neither is a definition
    // the policy made. A candidate whose entire *meaning* is one of the words a
    // document uses to label a definitions column is drafting furniture, not a
    // meaning; refuse it.
    let meaning_lower = meaning.to_lowercase();
    if GLOSSARY_HEADER_WORDS.contains(&meaning_lower.as_str()) {
        return None;
    }
    if GLOSSARY_HEADER_WORDS.contains(&term.to_lowercase().as_str()) {
        return None;
    }

    Some((term, meaning))
}

/// `"Accident" means a sudden, violent event.` — the most common layout, and
/// the most reliable signal, because the quotation marks are there precisely
/// to tell the reader "this word is about to stop meaning what you think".
fn split_quoted(line: &str) -> Option<(&str, &str)> {
    let mut chars = line.char_indices();
    let (_, opening) = chars.next()?;
    let closing = match opening {
        '"' => '"',
        '\u{201C}' => '\u{201D}', // “ ”
        '\'' => '\'',
        '\u{2018}' => '\u{2019}', // ‘ ’
        _ => return None,
    };

    let term_start = opening.len_utf8();
    let term_len = line[term_start..].find(closing)?;
    let term = &line[term_start..term_start + term_len];
    let rest = &line[term_start + term_len + closing.len_utf8()..];

    // Whatever operative verb follows the quoted term ("means", "shall mean",
    // ":") is drafting furniture, not part of the meaning.
    let meaning = strip_leading_defining_phrase(rest.trim_start());
    Some((term, meaning))
}

fn split_on_defining_phrase(line: &str) -> Option<(&str, &str)> {
    let lower = line.to_ascii_lowercase();
    DEFINING_PHRASES.iter().find_map(|phrase| {
        // The phrase must be a whole word/phrase, not a substring of one.
        let at = lower.find(phrase)?;
        let before_ok = at > 0 && lower.as_bytes()[at - 1].is_ascii_whitespace();
        let after = at + phrase.len();
        let after_ok = after == lower.len()
            || !lower.as_bytes()[after].is_ascii_alphanumeric()
            || phrase.ends_with(':');
        (before_ok && after_ok).then(|| (&line[..at], line[after..].trim_start()))
    })
}

fn split_on_separator(line: &str) -> Option<(&str, &str)> {
    DEFINING_SEPARATORS
        .iter()
        .find_map(|separator| line.split_once(separator))
}

fn strip_leading_defining_phrase(rest: &str) -> &str {
    let lower = rest.to_ascii_lowercase();
    for phrase in DEFINING_PHRASES {
        if lower.starts_with(phrase) {
            return rest[phrase.len()..].trim_start();
        }
    }
    rest.trim_start_matches([':', '-', '\u{2013}', '\u{2014}'])
        .trim_start()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clause::{ClauseId, ClauseLine};
    use crate::provenance::{DocumentId, PageNumber, SectionPath};

    /// Builds a definitions clause. **Every wording in these tests is invented
    /// for the test.** None of it is any real insurer's policy language.
    fn definition_clause(id: &str, lines: &[(usize, &str)]) -> Clause {
        Clause::new(
            DocumentId::new("synthetic-policy.pdf").unwrap(),
            ClauseId::printed(id).unwrap(),
            None,
            SectionPath::new(["Section 2 — Definitions"]),
            ClauseRole::Definition,
            lines
                .iter()
                .map(|&(page, text)| {
                    ClauseLine::new(PageNumber::new(page).unwrap(), text).unwrap()
                })
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn parses_a_quoted_definition() {
        let clause = definition_clause(
            "2.1",
            &[(2, "\"Accident\" means a sudden, violent, external and visible event.")],
        );
        let definitions = Definitions::extract([&clause]).unwrap();
        let Resolution::Defined(accident) = definitions.resolve("Accident") else {
            panic!("expected Accident to be defined");
        };
        assert_eq!(accident.term(), "Accident");
        assert_eq!(
            accident.meaning(),
            "a sudden, violent, external and visible event."
        );
        assert_eq!(accident.provenance().page().get(), 2);
    }

    #[test]
    fn parses_an_unquoted_definition_and_a_glossary_row() {
        let clause = definition_clause(
            "2",
            &[
                (2, "Hospital means an institution registered as a hospital."),
                (2, "Spouse | the person to whom the Insured is lawfully married."),
            ],
        );
        let definitions = Definitions::extract([&clause]).unwrap();
        assert_eq!(definitions.len(), 2);
        assert_eq!(
            definitions.resolve("hospital").definition().unwrap().meaning(),
            "an institution registered as a hospital."
        );
        assert_eq!(
            definitions.resolve("SPOUSE").definition().unwrap().meaning(),
            "the person to whom the Insured is lawfully married."
        );
    }

    #[test]
    fn a_glossary_table_header_row_is_not_a_definition() {
        // "Term | Meaning" parses perfectly well as a definition of the word
        // "Term". It must not become one.
        let clause = definition_clause(
            "2",
            &[
                (2, "Term | Meaning"),
                (2, "Accident | a sudden, violent, external and visible event."),
            ],
        );
        let definitions = Definitions::extract([&clause]).unwrap();
        assert_eq!(definitions.len(), 1);
        assert!(matches!(definitions.resolve("Term"), Resolution::Undefined));
        assert!(definitions.resolve("Accident").is_policy_defined());
    }

    #[test]
    fn a_multi_line_meaning_is_kept_whole() {
        let clause = definition_clause(
            "2.4",
            &[
                (3, "\"Pre-existing Condition\" means any condition for which"),
                (3, "the Insured received treatment before the Policy began."),
            ],
        );
        let definitions = Definitions::extract([&clause]).unwrap();
        let definition = definitions
            .resolve("pre-existing condition")
            .definition()
            .expect("defined");
        assert!(definition.meaning().contains("any condition for which"));
        assert!(definition.meaning().contains("before the Policy began."));
    }

    #[test]
    fn a_term_defined_twice_inconsistently_is_conflicting_not_arbitrarily_resolved() {
        let base = definition_clause("2.1", &[(2, "\"Accident\" means a sudden, violent event.")]);
        let endorsed = definition_clause(
            "2.1",
            &[(9, "\"Accident\" means a sudden event, whether or not violent.")],
        );
        let definitions = Definitions::extract([&base, &endorsed]).unwrap();

        match definitions.resolve("Accident") {
            Resolution::Conflicting(all) => {
                assert_eq!(all.len(), 2);
                // Both citations survive, so a reader can see the contradiction.
                let pages: Vec<usize> =
                    all.iter().map(|d| d.provenance().page().get()).collect();
                assert_eq!(pages, vec![2, 9]);
            }
            other => panic!("expected Conflicting, got {other:?}"),
        }
        // And it must NOT silently pick one.
        assert!(definitions.resolve("Accident").definition().is_none());
        // But it is still policy-defined: plain English is not the answer.
        assert!(definitions.resolve("Accident").is_policy_defined());
        assert_eq!(definitions.conflicts().len(), 1);
    }

    #[test]
    fn a_term_defined_twice_identically_is_a_duplicate_not_a_conflict() {
        let a = definition_clause("2.1", &[(2, "\"Accident\" means a sudden, violent event.")]);
        let b = definition_clause("2.1", &[(9, "\"Accident\" means a sudden, violent event.")]);
        let definitions = Definitions::extract([&a, &b]).unwrap();
        assert!(matches!(
            definitions.resolve("Accident"),
            Resolution::Defined(_)
        ));
    }

    #[test]
    fn an_undefined_term_says_so_rather_than_guessing() {
        let clause = definition_clause("2.1", &[(2, "\"Accident\" means a sudden event.")]);
        let definitions = Definitions::extract([&clause]).unwrap();
        assert_eq!(definitions.resolve("Earthquake"), Resolution::Undefined);
        assert!(!definitions.resolve("Earthquake").is_policy_defined());
    }

    #[test]
    fn occurrences_prefer_the_longest_defined_phrase() {
        let clause = definition_clause(
            "2",
            &[
                (2, "Hospital means an institution registered as a hospital."),
                (2, "Hospital Cash Benefit means the daily amount in the Schedule."),
            ],
        );
        let definitions = Definitions::extract([&clause]).unwrap();

        let usage = Clause::new(
            DocumentId::new("synthetic-policy.pdf").unwrap(),
            ClauseId::printed("3.1").unwrap(),
            None,
            SectionPath::new(["Section 3"]),
            ClauseRole::Coverage,
            vec![ClauseLine::new(
                PageNumber::new(4).unwrap(),
                "We will pay the Hospital Cash Benefit for each day.",
            )
            .unwrap()],
        )
        .unwrap();

        let occurrences = definitions.occurrences_in(&usage);
        assert_eq!(occurrences.len(), 1);
        assert_eq!(occurrences[0].surface, "Hospital Cash Benefit");
    }

    #[test]
    fn occurrences_match_a_regular_plural_but_not_a_different_word() {
        let clause = definition_clause("2.1", &[(2, "\"Accident\" means a sudden event.")]);
        let definitions = Definitions::extract([&clause]).unwrap();

        let usage = |text: &str| {
            Clause::new(
                DocumentId::new("synthetic-policy.pdf").unwrap(),
                ClauseId::printed("3.1").unwrap(),
                None,
                SectionPath::default(),
                ClauseRole::Coverage,
                vec![ClauseLine::new(PageNumber::new(4).unwrap(), text).unwrap()],
            )
            .unwrap()
        };

        let plural = usage("Cover applies to Accidents occurring abroad.");
        assert_eq!(definitions.occurrences_in(&plural).len(), 1);

        // "Accidental" is a different word. Substituting the policy's meaning
        // for `Accident` here would be a confident misreading.
        let derived = usage("Accidental damage is not covered.");
        assert!(definitions.occurrences_in(&derived).is_empty());
    }
}

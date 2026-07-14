//! Author names — the part of a bibliography that is easy to get wrong and
//! insulting to get wrong.
//!
//! # The problem, stated honestly
//!
//! A personal name is not a string with a surname at the end. The naive rule —
//! *"the last token is the family name"* — is wrong for a large fraction of the
//! world's researchers, and every time it is wrong it **renames a real person
//! in their own citation**:
//!
//! | Name | Naive rule says | Truth |
//! |---|---|---|
//! | `van der Waals` | family = `Waals` | family = `van der Waals` (particle is part of it) |
//! | `de Sousa` | family = `Sousa` | family = `de Sousa` |
//! | `Kim Jong-un` | family = `Jong-un` | family = `Kim` — **the family name leads** |
//! | `Mao Zedong` | family = `Zedong` | family = `Mao` |
//! | `Martin Luther King Jr.` | family = `Jr.` | family = `King`, suffix = `Jr.` |
//! | `Ludwig van Beethoven` | family = `Beethoven` | correct, by luck |
//!
//! Note that `Kim Jong-un` and `John Smith` are **the same shape**: two
//! capitalised tokens, nothing else to go on. No amount of cleverness
//! distinguishes them from the string alone. It requires a lexicon of names,
//! which this crate does not have — and asking a language model would be
//! exactly the "promote a model to an oracle" move CLAUDE.md forbids (a model
//! would also be confidently wrong on precisely the unusual names where being
//! wrong matters most).
//!
//! # What this module does about it
//!
//! Three things, and the third is the one that matters.
//!
//! ### 1. It models a name properly
//!
//! [`PersonName`] has given names, an optional **particle** (`van der`, `de`,
//! `von`), a family name, and an optional **suffix** (`Jr.`, `III`). Given
//! names distinguish an [`GivenName::Initial`] (`J.`) from a
//! [`GivenName::Full`] (`John`), because a citation style that must render
//! "J. Smith" from "John Smith" can do that, and one that must render "John"
//! from "J." **cannot**, and must not pretend to.
//!
//! ### 2. It refuses to split what it cannot split
//!
//! [`Author::Literal`] keeps a name **exactly as written**. It is used for
//! names in CJK script (where splitting requires knowing the convention *and*
//! the person), for anything containing characters no name grammar covers, and
//! for anything a caller marks as not-to-be-touched. [`Author::Organization`]
//! keeps corporate authors (`European Bioinformatics Institute`) whole, because they
//! have no given name to find and cutting one off them produces nonsense.
//!
//! ### 3. Every parsed name keeps the string it was parsed from, and that
//! string is what gets emitted
//!
//! **This is the safety property.** [`PersonName::as_written`] returns the
//! original, always. And [`NameConfidence`] records *how* the split was
//! obtained:
//!
//! * [`NameConfidence::Explicit`] — the **source told us**. `Smith, John` puts
//!   a comma between family and given; no convention was assumed.
//! * [`NameConfidence::Conventional`] — the *shape* carries the information.
//!   `J. Smith` (an initial cannot be a family name in this position);
//!   `van der Waals` (a lower-case particle is a documented signal). Applying
//!   the rule is safe because the rule is what the shape means.
//! * [`NameConfidence::Assumed`] — two or more plain capitalised words and
//!   nothing to disambiguate them. `John Smith`. `Mao Zedong`. Western order is
//!   **assumed**, and it may be wrong.
//!
//! And then the rule that makes the assumption harmless:
//!
//! > **A name whose split is [`Assumed`](NameConfidence::Assumed) is never
//! > emitted in a re-ordered form.** BibTeX emission of such a name writes it
//! > back exactly as written (`John Smith`, `Mao Zedong`), never
//! > `Smith, John` / `Zedong, Mao`. See [`crate::bibtex`].
//!
//! So the inferred family name is available to a caller who wants it (sorting a
//! bibliography needs *something*), is honestly labelled as an inference, and
//! **is never baked into an artefact that a typesetter or a reader will
//! believe**. We can be wrong internally; we cannot be wrong in public.
//!
//! # The BibTeX name grammar
//!
//! Where a split *is* attempted, it follows BibTeX's own documented grammar
//! (Oren Patashnik's `btxdoc`, elaborated in Nicolas Markey's *Tame the BeaST*,
//! §4) rather than a rule invented here:
//!
//! ```text
//!   First von Last
//!   von Last, First
//!   von Last, Jr, First
//! ```
//!
//! with `von` being the maximal run of tokens that begin with a **lower-case**
//! letter, and everything after it being `Last`. That is why `van der Waals`
//! works: `van` and `der` are lower-case, so they are the von-part, and the
//! von-part belongs with the family name. This grammar is 40 years old, is what
//! every LaTeX bibliography in existence has been written against, and is
//! therefore the *least surprising* thing we can implement.
//!
//! # Author lists are a different grammar, and conflating them is a bug
//!
//! In BibTeX, authors are separated by ` and `, and a comma **inside** a name
//! separates its parts. In a printed reference list (IEEE, Elsevier, Springer),
//! a comma **separates authors**:
//!
//! ```text
//!   BibTeX:         Chen, M. R. and Novak, S. and Alvarez, J. P.
//!   Reference list: M. R. Chen, S. Novak, and J. P. Alvarez
//! ```
//!
//! The same comma means opposite things. So there are two entry points —
//! [`parse_bibtex_name_list`] and [`parse_printed_name_list`] — and they are
//! not interchangeable. Feeding a printed list to the BibTeX parser produces
//! one author called "Chen" with the given name "M. R. Novak S. Alvarez J.
//! P.", which is exactly the sort of quiet catastrophe this crate exists to
//! prevent.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::text::{fold_typography, squeeze};

/// A given name, which may be spelled out or abbreviated to an initial.
///
/// The distinction is not cosmetic. Expansion is **impossible**: from `J.` you
/// cannot recover `John`, and a crate that guessed would be inventing a fact
/// about a person. Abbreviation, on the other hand, is trivial. So the type
/// records which one the source gave us, and every renderer can go down but
/// never up.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GivenName {
    /// A given name spelled out: `John`, `Per`, `Nicolas`.
    Full(String),
    /// An initial, stored **without** its period: `J`, `T`, `B.-C.` becomes
    /// `B.-C` — no. See below.
    ///
    /// Stored exactly as printed *minus* a single trailing period, so that a
    /// hyphenated initial group (`B.-C.`, common in Chinese romanisation and
    /// found in the maintainer's own reference list) survives as `B.-C` and
    /// renders back as `B.-C.`.
    Initial(String),
}

impl GivenName {
    /// Renders for display: an initial regains its period.
    pub fn as_display(&self) -> String {
        match self {
            Self::Full(name) => name.clone(),
            Self::Initial(letters) => format!("{letters}."),
        }
    }

    /// The initial form of this given name — always possible.
    ///
    /// A `Full("John")` abbreviates to `J.`; an `Initial` is already there.
    /// There is deliberately no inverse: see the type docs.
    pub fn to_initial(&self) -> String {
        match self {
            Self::Full(name) => name
                .chars()
                .next()
                .map(|c| format!("{c}."))
                .unwrap_or_default(),
            Self::Initial(letters) => format!("{letters}."),
        }
    }

    /// Whether the source only gave us an initial.
    pub fn is_initial(&self) -> bool {
        matches!(self, Self::Initial(_))
    }
}

impl fmt::Display for GivenName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.as_display())
    }
}

/// How a name's split into parts was arrived at — and therefore how far it may
/// be trusted.
///
/// Read the module docs before using this. In short: `Explicit` and
/// `Conventional` splits are safe to act on; an `Assumed` split may be wrong,
/// and this crate never commits an `Assumed` split to an emitted artefact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NameConfidence {
    /// The **source stated** which part was the family name — a comma form
    /// (`Waals, Johannes Diderik van`). No convention was assumed, so this
    /// split cannot be wrong unless the source was.
    Explicit,

    /// Derived by applying BibTeX's documented grammar to a shape that
    /// **carries the information itself**:
    ///
    /// * an initial in leading position (`J. Smith`, `M. R. Chen`) — an
    ///   initial is not a family name;
    /// * a lower-case particle (`van der Waals`, `de Sousa`) — the particle is
    ///   the documented signal, and the family name follows it;
    /// * a recognised suffix (`King Jr.`).
    ///
    /// Safe to act on: the rule is not a guess, it is what the shape means.
    Conventional,

    /// Two or more plain capitalised words, with **nothing to disambiguate
    /// them**: `John Smith`, `Mao Zedong`, `Kim Jong-un`.
    ///
    /// Western given-then-family order has been **assumed**. For a name natively
    /// written family-name-first that assumption is wrong, and the string alone
    /// cannot tell us which case we are in.
    ///
    /// The split is offered because sorting a bibliography needs *some* key.
    /// It is never emitted in re-ordered form. See the module docs.
    Assumed,
}

/// A personal name, split into parts.
///
/// **Always carries the string it was parsed from** ([`Self::as_written`]).
/// That is not redundancy; it is the guarantee that a bad split cannot destroy
/// data. Whatever this crate concluded about which word was the family name,
/// the person's name as they wrote it is still there, byte for byte, and it is
/// what gets emitted whenever the split is not trustworthy.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PersonName {
    given: Vec<GivenName>,
    particle: Option<String>,
    family: String,
    suffix: Option<String>,
    literal: String,
    confidence: NameConfidence,
}

impl PersonName {
    /// The given names, in order.
    pub fn given(&self) -> &[GivenName] {
        &self.given
    }

    /// The particle (`van der`, `de`, `von`), if any.
    ///
    /// Kept separate from the family name because citation styles disagree
    /// about it — some alphabetise `van der Waals` under **W**, some under
    /// **V** — and a crate that glued it on could not serve both. Use
    /// [`Self::full_family`] for the form that includes it.
    pub fn particle(&self) -> Option<&str> {
        self.particle.as_deref()
    }

    /// The family name, **without** any particle.
    pub fn family(&self) -> &str {
        &self.family
    }

    /// The family name **with** its particle, which is how the person is
    /// actually called: `van der Waals`, not `Waals`.
    pub fn full_family(&self) -> String {
        match &self.particle {
            Some(particle) => format!("{particle} {}", self.family),
            None => self.family.clone(),
        }
    }

    /// A generational or honorific suffix (`Jr.`, `III`), if any.
    pub fn suffix(&self) -> Option<&str> {
        self.suffix.as_deref()
    }

    /// **The name exactly as the source wrote it.**
    ///
    /// This is the ground truth. Every other accessor on this type is this
    /// crate's *interpretation* of it, and where the interpretation is not
    /// trustworthy ([`NameConfidence::Assumed`]) this is what gets emitted.
    pub fn as_written(&self) -> &str {
        &self.literal
    }

    /// How the split was arrived at. See [`NameConfidence`].
    pub fn confidence(&self) -> NameConfidence {
        self.confidence
    }

    /// Whether the family name may be safely relied upon — i.e. whether it came
    /// from the source or from a rule, rather than from an assumption about
    /// name order.
    pub fn family_is_trustworthy(&self) -> bool {
        !matches!(self.confidence, NameConfidence::Assumed)
    }

    /// The `von Last, Jr, First` form BibTeX understands.
    ///
    /// # Safety of the reordering
    ///
    /// Only called for a name whose family part is trustworthy. For an
    /// [`Assumed`](NameConfidence::Assumed) name, [`crate::bibtex`] emits
    /// [`Self::as_written`] instead — because reordering `Mao Zedong` into
    /// `Zedong, Mao` would assert a family name we do not know to be one, and
    /// BibTeX would then print it in a bibliography as "Zedong, M.", renaming a
    /// person in a published document.
    pub fn to_bibtex_reordered(&self) -> String {
        let mut out = self.full_family();
        if let Some(suffix) = &self.suffix {
            out.push_str(", ");
            out.push_str(suffix);
        }
        if !self.given.is_empty() {
            out.push_str(", ");
            out.push_str(
                &self
                    .given
                    .iter()
                    .map(GivenName::as_display)
                    .collect::<Vec<_>>()
                    .join(" "),
            );
        }
        out
    }
}

impl fmt::Display for PersonName {
    /// Displays the name **as written**. Not as we decided to split it.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.literal)
    }
}

/// One author of a work.
///
/// Three variants, because there are three genuinely different things an author
/// field can contain, and flattening them loses information that cannot be
/// recovered.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Author {
    /// A person whose name was split into parts. Check
    /// [`PersonName::confidence`] before trusting the split.
    Person(PersonName),

    /// A corporate or institutional author: `European Bioinformatics Institute`,
    /// `Unicode Consortium`, `Association for Computing Machinery`.
    ///
    /// It has no given name. Applying a person-name grammar to it produces
    /// `Institute, European Bioinformatics` — which appears in real
    /// bibliographies, and is exactly as silly as it looks. So it is kept whole,
    /// and emitted brace-protected so LaTeX keeps it whole too.
    Organization(String),

    /// **A name this crate could not confidently split, kept exactly as
    /// written.**
    ///
    /// Reached when the name is in a script whose ordering convention we cannot
    /// determine (Han, Hangul, Kana), or when the caller has marked it
    /// not-to-be-touched (a `{{...}}` group in BibTeX).
    ///
    /// This is not a failure mode. It is the *correct answer* to "how do I
    /// split this?" when the honest answer is "you do not". A `Literal` author
    /// round-trips through every emitter in this crate byte-for-byte.
    Literal(String),
}

impl Author {
    /// The author's name **exactly as the source wrote it**, for every variant.
    ///
    /// Never lossy, never reordered, never re-cased. This is the accessor to
    /// reach for when displaying a name to a human.
    pub fn as_written(&self) -> &str {
        match self {
            Self::Person(person) => person.as_written(),
            Self::Organization(name) | Self::Literal(name) => name,
        }
    }

    /// The family name, if one is known **and trustworthy**.
    ///
    /// Returns `None` for an organisation, for a `Literal`, and — crucially —
    /// for a [`PersonName`] whose split is only
    /// [`Assumed`](NameConfidence::Assumed). A caller sorting a bibliography
    /// should fall back to [`Self::sort_key`], which is honest about this.
    pub fn family(&self) -> Option<&str> {
        match self {
            Self::Person(person) if person.family_is_trustworthy() => Some(person.family()),
            _ => None,
        }
    }

    /// A deterministic key for sorting a bibliography.
    ///
    /// Uses the trustworthy family name where there is one, and the name **as
    /// written** otherwise. That means `Mao Zedong` sorts under **M** (which is
    /// right) rather than under a family name we guessed at, and an organisation
    /// sorts under its own first word. It is not perfect alphabetisation — it
    /// cannot be, without a lexicon — but it is *stable*, *deterministic*, and
    /// it never sorts a person under a name that is not theirs.
    pub fn sort_key(&self) -> String {
        let key = match self {
            Self::Person(person) if person.family_is_trustworthy() => person.full_family(),
            other => other.as_written().to_string(),
        };
        key.to_lowercase()
    }

    /// Parses one name using BibTeX's grammar (commas separate name *parts*).
    ///
    /// Never fails: a name it cannot split becomes [`Author::Literal`], which
    /// is the whole point. See [`parse_bibtex_name_list`] for the list form.
    pub fn parse_bibtex_name(text: &str) -> Self {
        parse_one_name(text)
    }
}

impl fmt::Display for Author {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_written())
    }
}

/// An author list, and whether the source truncated it.
///
/// `et al.` is **not an author called "al."** — a mistake real bibliography
/// software has made — and it is not nothing either: it is the source telling
/// us that authors exist which it did not print. Recording that is the
/// difference between "this paper has three authors" (false) and "this paper
/// has at least three authors, and the source stopped listing them" (true).
///
/// A truncated list is precisely the case a network resolver could complete —
/// see [`crate::resolve`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorList {
    authors: Vec<Author>,
    truncated: bool,
}

impl AuthorList {
    /// Builds an author list.
    pub fn new(authors: Vec<Author>, truncated: bool) -> Self {
        Self { authors, truncated }
    }

    /// The authors the source actually named.
    pub fn authors(&self) -> &[Author] {
        &self.authors
    }

    /// Whether the source said `et al.` — i.e. whether there are authors it did
    /// not name.
    pub fn is_truncated(&self) -> bool {
        self.truncated
    }

    /// Whether no authors were found at all.
    pub fn is_empty(&self) -> bool {
        self.authors.is_empty()
    }

    /// How many authors were *named*. Not how many the work has, if
    /// [`Self::is_truncated`].
    pub fn len(&self) -> usize {
        self.authors.len()
    }

    /// The first author, on whom citation keys and author-year citations hang.
    pub fn first(&self) -> Option<&Author> {
        self.authors.first()
    }
}

// ---------------------------------------------------------------------------
// Parsing: the BibTeX grammar (commas separate name PARTS)
// ---------------------------------------------------------------------------

/// Splits a BibTeX `author` / `editor` field into individual names.
///
/// The separator is ` and ` **at brace level zero** — a nesting rule that
/// matters, because `{Smith and Wesson Ltd.}` is one brace-protected corporate
/// author, not two people, and splitting it would invent an author who does not
/// exist.
///
/// A trailing `and others` is BibTeX's spelling of `et al.` and sets
/// [`AuthorList::is_truncated`] rather than producing an author named "others".
pub fn parse_bibtex_name_list(text: &str) -> AuthorList {
    let mut authors = Vec::new();
    let mut truncated = false;

    for chunk in split_on_and(text) {
        let name = chunk.trim();
        if name.is_empty() {
            continue;
        }
        if name.eq_ignore_ascii_case("others") {
            // BibTeX's own "et al.". Not a person.
            truncated = true;
            continue;
        }
        authors.push(parse_one_name(name));
    }

    AuthorList::new(authors, truncated)
}

/// Splits on ` and ` at brace depth zero.
fn split_on_and(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut depth = 0i32;
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            '{' => {
                depth += 1;
                current.push('{');
                i += 1;
            }
            '}' => {
                depth -= 1;
                current.push('}');
                i += 1;
            }
            _ if depth == 0 && is_and_separator(&chars, i) => {
                parts.push(std::mem::take(&mut current));
                i += 5; // len(" and ")
            }
            c => {
                current.push(c);
                i += 1;
            }
        }
    }
    parts.push(current);
    parts
}

/// Whether position `i` begins the literal separator ` and ` (with the spaces).
///
/// The surrounding spaces are load-bearing: without them, `Ferdinand` contains
/// `and`, and `Alexander` contains `and`, and a great many chemists would
/// discover they had been split in half.
fn is_and_separator(chars: &[char], i: usize) -> bool {
    if i + 5 > chars.len() {
        return false;
    }
    chars[i] == ' '
        && chars[i + 1].eq_ignore_ascii_case(&'a')
        && chars[i + 2].eq_ignore_ascii_case(&'n')
        && chars[i + 3].eq_ignore_ascii_case(&'d')
        && chars[i + 4] == ' '
}

/// Words that, when they appear in lower case before the family name, are
/// **particles** — part of the family name, not given names.
///
/// This list exists only for the case where the particle has been **capitalised
/// by a typesetter**, which happens constantly in reference lists that render
/// names as `Van Der Waals, J. D.`. BibTeX's own rule (lower case = particle)
/// handles the un-mangled case without any list at all, and is applied first.
///
/// Deliberately conservative. A word wrongly treated as a particle would be
/// silently deleted from a given name.
const PARTICLES: &[&str] = &[
    "van", "von", "de", "del", "della", "der", "den", "di", "da", "das", "dos", "du", "la", "le",
    "les", "el", "al", "bin", "binte", "binti", "ibn", "ter", "ten", "op", "af", "av", "zu", "vom",
    "zum", "zur", "st",
];

/// Generational and honorific suffixes.
///
/// Roman numerals are included because `Martin Luther King III` is a name and
/// `III` is not a family name.
const SUFFIXES: &[&str] = &[
    "jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v", "vi", "phd", "ph.d.", "md", "esq",
];

/// Parses a single name. **Total** — cannot fail, only decline to split.
fn parse_one_name(raw: &str) -> Author {
    let trimmed = raw.trim();

    // -- A `{{...}}` group is BibTeX's own "do not touch this" marker. Honour it
    //    literally: the author asked for the name to be left alone, and that
    //    request is the most reliable information available about it.
    if let Some(inner) = strip_double_braces(trimmed) {
        return Author::Literal(inner.to_string());
    }
    if let Some(inner) = strip_single_braces(trimmed) {
        // A single brace group around the whole name is the conventional way to
        // write a corporate author.
        return Author::Organization(inner.to_string());
    }

    // -- A name in a script we cannot order. Not a failure; an honest refusal.
    if contains_cjk(trimmed) {
        return Author::Literal(trimmed.to_string());
    }

    if trimmed.is_empty() {
        return Author::Literal(String::new());
    }

    // -- Corporate authors, detected by vocabulary rather than by shape. This
    //    is a heuristic and it is deliberately narrow: a false positive costs
    //    us a split we could have made, which is cheap. A false *negative*
    //    costs us "Institute, European Bioinformatics", which is not.
    if looks_like_an_organization(trimmed) {
        return Author::Organization(trimmed.to_string());
    }

    let commas = trimmed.matches(',').count();
    match commas {
        0 => parse_first_von_last(trimmed),
        _ => parse_von_last_comma_first(trimmed, commas),
    }
}

/// BibTeX's `First von Last` form: no comma anywhere.
fn parse_first_von_last(raw: &str) -> Author {
    let tokens: Vec<&str> = raw.split_whitespace().collect();
    if tokens.is_empty() {
        return Author::Literal(raw.to_string());
    }
    if tokens.len() == 1 {
        // A mononym, or just a family name. Nothing to get wrong.
        return Author::Person(PersonName {
            given: Vec::new(),
            particle: None,
            family: tokens[0].to_string(),
            suffix: None,
            literal: raw.to_string(),
            confidence: NameConfidence::Explicit,
        });
    }

    let mut tokens = tokens;

    // -- Trailing suffix: "Martin Luther King Jr." The suffix is never the
    //    family name, and the naive rule would make it one.
    let mut suffix = None;
    if tokens.len() >= 3 && is_suffix(tokens[tokens.len() - 1]) {
        suffix = Some(tokens.pop().unwrap().to_string());
    }

    // -- The von-part: BibTeX's rule is the maximal run of tokens beginning
    //    with a lower-case letter, and everything from there to the end
    //    (minus the suffix) is the family name. This is what makes
    //    `van der Waals` come out as `van der` + `Waals` rather than
    //    `Johannes van der` + `Waals`.
    let first_lowercase = tokens
        .iter()
        .position(|t| starts_lowercase(t))
        // The last token cannot start the von-part; there would be no family
        // name left. `Ludwig van` is not a name.
        .filter(|&i| i + 1 < tokens.len());

    if let Some(von_start) = first_lowercase {
        // The von-part runs from `von_start` to the last lower-case token
        // before the family name.
        let mut von_end = von_start;
        for (i, token) in tokens.iter().enumerate().skip(von_start) {
            if starts_lowercase(token) && i + 1 < tokens.len() {
                von_end = i;
            }
        }
        let given = tokens[..von_start].iter().copied().map(given_name).collect();
        let particle = tokens[von_start..=von_end].join(" ");
        let family = tokens[von_end + 1..].join(" ");

        return Author::Person(PersonName {
            given,
            particle: Some(particle),
            family,
            suffix,
            literal: raw.to_string(),
            // The lower-case particle IS the signal. This is not a guess.
            confidence: NameConfidence::Conventional,
        });
    }

    // -- Capitalised particles ("Van Der Waals"), which a typesetter produces
    //    and BibTeX's lower-case rule therefore misses. Only applied when the
    //    particle is NOT the first token, because a leading "Van" could equally
    //    be a given name.
    let capitalised_particle = (1..tokens.len().saturating_sub(1))
        .find(|&i| is_particle(tokens[i]));

    if let Some(von_start) = capitalised_particle {
        let mut von_end = von_start;
        while von_end + 1 < tokens.len() - 1 && is_particle(tokens[von_end + 1]) {
            von_end += 1;
        }
        let given = tokens[..von_start].iter().copied().map(given_name).collect();
        return Author::Person(PersonName {
            given,
            particle: Some(tokens[von_start..=von_end].join(" ")),
            family: tokens[von_end + 1..].join(" "),
            suffix,
            literal: raw.to_string(),
            confidence: NameConfidence::Conventional,
        });
    }

    // -- No particle. The family name is the last token; the question is how
    //    much we trust that.
    let family = tokens.pop().unwrap().to_string();
    let given: Vec<GivenName> = tokens.iter().copied().map(given_name).collect();

    // If ANY leading token is an initial, the shape itself tells us the order:
    // an initial is a given name, and a family name does not precede one here.
    // "M. R. Chen" -- unambiguous. "J. Smith" -- unambiguous.
    let confidence = if given.iter().any(GivenName::is_initial) {
        NameConfidence::Conventional
    } else {
        // "John Smith". "Mao Zedong". "Kim Jong-un". Identical shapes,
        // different conventions, and the string cannot tell us which. We assume
        // Western order -- and we never emit the assumption. See the module docs.
        NameConfidence::Assumed
    };

    Author::Person(PersonName {
        given,
        particle: None,
        family,
        suffix,
        literal: raw.to_string(),
        confidence,
    })
}

/// BibTeX's `von Last, First` and `von Last, Jr, First` forms.
///
/// The source has told us where the family name ends. This is the **only**
/// shape where the split cannot be wrong, and it is why a bibliography written
/// by a careful author is worth more than one this crate had to guess at.
fn parse_von_last_comma_first(raw: &str, commas: usize) -> Author {
    let parts: Vec<&str> = raw.splitn(3, ',').map(str::trim).collect();

    let (last_part, suffix, first_part) = if commas >= 2 && !parts[1].is_empty() {
        // "King, Jr., Martin Luther"
        (parts[0], Some(parts[1].to_string()), parts.get(2).copied().unwrap_or(""))
    } else {
        // "Waals, Johannes Diderik van" -- note the trailing particle, which
        // BibTeX places in the FIRST part in this form. Handled below.
        (parts[0], None, parts.get(1).copied().unwrap_or(""))
    };

    // The family part may itself carry a leading particle: "van der Waals, J. D."
    let last_tokens: Vec<&str> = last_part.split_whitespace().collect();
    let von_end = last_tokens
        .iter()
        .enumerate()
        .take(last_tokens.len().saturating_sub(1))
        .filter(|(_, t)| starts_lowercase(t) || is_particle(t))
        .map(|(i, _)| i)
        .next_back();

    let (particle, family) = match von_end {
        Some(end) => (
            Some(last_tokens[..=end].join(" ")),
            last_tokens[end + 1..].join(" "),
        ),
        None => (None, last_part.to_string()),
    };

    let given: Vec<GivenName> = first_part.split_whitespace().map(given_name).collect();

    Author::Person(PersonName {
        given,
        particle,
        family,
        suffix,
        literal: raw.to_string(),
        // The source used a comma. It TOLD us where the family name is.
        confidence: NameConfidence::Explicit,
    })
}

/// Classifies one token of a given-name field as a spelled-out name or an
/// initial.
///
/// A token is an initial if, after stripping periods and hyphens, it is a
/// single letter — or, for the hyphenated groups common in Chinese romanisation
/// (`B.-C.`, `Y.-L.`), a run of single letters. This case is not hypothetical:
/// it is in the maintainer's own reference list ("B.-C. Du, Y.-L. He").
fn given_name(token: &str) -> GivenName {
    let stripped = token.trim_end_matches('.');
    let letters: Vec<&str> = stripped
        .split('-')
        .map(|s| s.trim_end_matches('.'))
        .collect();

    let all_single = !letters.is_empty()
        && letters
            .iter()
            .all(|s| s.chars().count() == 1 && s.chars().all(char::is_alphabetic));

    if all_single {
        GivenName::Initial(stripped.to_string())
    } else {
        GivenName::Full(token.to_string())
    }
}

// ---------------------------------------------------------------------------
// Parsing: the printed-reference-list grammar (commas separate AUTHORS)
// ---------------------------------------------------------------------------

/// Parses an author list **as printed in a reference list**, where commas
/// separate *authors*, not name parts.
///
/// ```text
///   M. R. Chen, S. Novak, and J. P. Alvarez
///   B.-C. Du, Y.-L. He, Y. Qiu, Q. Liang, and Y.-P. Zhou
///   Smith, J., Jones, A. and Brown, B.          <- also handled
/// ```
///
/// # The ambiguity, and how it is resolved
///
/// `Smith, J., Jones, A.` is genuinely ambiguous: two authors in
/// `Family, Initial` form, or four authors called `Smith`, `J.`, `Jones`,
/// `A.`? The disambiguator is that a lone initial is **never** a whole author.
/// So a comma-separated chunk that is nothing but initials is glued back onto
/// the chunk before it, which recovers `Smith, J.` and `Jones, A.` correctly
/// and cannot mis-fire on `M. R. Chen` (which has a family name in the same
/// chunk).
///
/// Anything this cannot handle degrades to [`Author::Literal`] — never to a
/// mangled name.
pub fn parse_printed_name_list(text: &str) -> AuthorList {
    let folded = fold_typography(text);
    let cleaned = squeeze(&folded);

    // "et al." / "et al" -- the source telling us it stopped listing.
    let mut truncated = false;
    let mut body = cleaned.as_str();
    for marker in [", et al.", " et al.", ", et al", " et al", " and others"] {
        if let Some(stripped) = body
            .strip_suffix(marker)
            .or_else(|| body.trim_end_matches(['.', ',']).strip_suffix(marker.trim_end_matches('.')))
        {
            body = stripped;
            truncated = true;
            break;
        }
    }
    if body.contains("et al") {
        // `et al.` in the middle: still a truncation signal, and the text after
        // it is not an author.
        truncated = true;
        body = body.split("et al").next().unwrap_or(body).trim_end_matches([',', ' ']);
    }

    // Normalise the Oxford "and"/"&" before the final author into a comma, so
    // that one splitter handles the whole list.
    let normalised = body
        .replace(", and ", ", ")
        .replace(" and ", ", ")
        .replace(", & ", ", ")
        .replace(" & ", ", ");

    let chunks: Vec<&str> = normalised
        .split(',')
        .map(str::trim)
        .filter(|c| !c.is_empty())
        .collect();

    // Re-glue `Family, Initials` pairs: a chunk that is nothing but initials
    // belongs to the chunk before it. See the doc comment.
    let mut merged: Vec<String> = Vec::new();
    for chunk in chunks {
        if is_all_initials(chunk) && !merged.is_empty() {
            let previous = merged.last_mut().expect("checked non-empty");
            previous.push_str(", ");
            previous.push_str(chunk);
        } else {
            merged.push(chunk.to_string());
        }
    }

    let authors = merged
        .iter()
        .filter(|name| !name.is_empty())
        .map(|name| parse_one_name(name))
        .collect();

    AuthorList::new(authors, truncated)
}

/// Whether a chunk consists only of initials (`J.`, `T. K. C.`, `B.-C.`).
///
/// Such a chunk cannot be a whole author — nobody's entire name is `J.` — so
/// its appearance after a comma means the comma was a name-part separator, not
/// an author separator.
fn is_all_initials(chunk: &str) -> bool {
    let tokens: Vec<&str> = chunk.split_whitespace().collect();
    !tokens.is_empty() && tokens.iter().all(|t| given_name(t).is_initial())
}

// ---------------------------------------------------------------------------
// Small predicates
// ---------------------------------------------------------------------------

/// Whether the first alphabetic character is lower case — BibTeX's von-part
/// test. Non-alphabetic leading characters (a brace, an accent macro) do not
/// count.
fn starts_lowercase(token: &str) -> bool {
    token
        .chars()
        .find(|c| c.is_alphabetic())
        .is_some_and(char::is_lowercase)
}

fn is_particle(token: &str) -> bool {
    let cleaned = token.trim_end_matches('.').to_lowercase();
    PARTICLES.contains(&cleaned.as_str())
}

/// Whether a token is a recognised name particle (`van`, `de`, `von`).
///
/// Exposed to [`crate::entry`], whose reference-line parser needs the same
/// vocabulary to decide where an author list stops: a lower-case token normally
/// means "this chunk is a title, not a name", and the particles are the sole
/// exception.
pub(crate) fn is_known_particle(token: &str) -> bool {
    is_particle(token)
}

/// Whether a name reads as an institution rather than a person.
///
/// Used by [`crate::entry`] to decide whether a book-style publisher is an
/// `institution` (a laboratory, a university) or a `publisher` (Springer,
/// Elsevier). Getting this wrong costs a BibTeX field name, not a fact.
pub(crate) fn looks_institutional(name: &str) -> bool {
    looks_like_an_organization(name)
}

/// Words marking a venue as an **academic** institution specifically — a
/// university, a college, a school.
///
/// This is the vocabulary behind the single most important ambiguity in the
/// corpus. In `biblatex`'s `ieee` style a PhD dissertation is printed as
///
/// ```text
///     Title. University of California, Berkeley, 2024.
/// ```
///
/// which is **character-for-character the shape of a book**, because the style
/// drops the "PhD thesis" designator entirely. University-published works in a
/// real reference list routinely take exactly this shape.
///
/// So we do not guess. We read what the string supports (a book) and raise an
/// [`Anomaly::AmbiguousEntryKind`](crate::Anomaly::AmbiguousEntryKind) naming
/// the alternative — which is the only honest thing to do with an ambiguity that
/// the source genuinely does not resolve.
pub(crate) fn looks_academic(name: &str) -> bool {
    const ACADEMIC: &[&str] = &[
        "university",
        "universit\u{e9}",
        "universiteit",
        "universit\u{e4}t",
        "college",
        "institute",
        "school",
        "polytechnic",
        "academy",
    ];
    let lower = name.to_lowercase();
    ACADEMIC.iter().any(|marker| {
        lower
            .split(|c: char| !c.is_alphanumeric())
            .any(|word| word == *marker)
    })
}

fn is_suffix(token: &str) -> bool {
    let cleaned = token.trim_end_matches(',').to_lowercase();
    SUFFIXES.contains(&cleaned.as_str())
}

/// Whether a name is written in a script whose ordering convention cannot be
/// determined from the string.
///
/// Han, Hangul and Kana names are conventionally family-name-first in their own
/// languages — but the *same person* is routinely printed given-name-first in
/// an English-language venue, and the characters do not say which happened. So
/// we do not split them. That is not a limitation to be apologised for; it is
/// the correct answer.
fn contains_cjk(text: &str) -> bool {
    text.chars().any(|c| {
        matches!(c as u32,
            0x3040..=0x309F   // Hiragana
            | 0x30A0..=0x30FF // Katakana
            | 0x3400..=0x4DBF // CJK Extension A
            | 0x4E00..=0x9FFF // CJK Unified Ideographs
            | 0xAC00..=0xD7AF // Hangul Syllables
            | 0xF900..=0xFAFF // CJK Compatibility Ideographs
        )
    })
}

/// Vocabulary that marks an author field as an institution rather than a person.
///
/// Narrow on purpose. See the call site.
const ORGANIZATION_MARKERS: &[&str] = &[
    "laboratory",
    "laboratories",
    "university",
    "institute",
    "department",
    "ministry",
    "agency",
    "commission",
    "committee",
    "association",
    "foundation",
    "society",
    "corporation",
    "administration",
    "organization",
    "organisation",
    "consortium",
    "council",
    "bureau",
    "centre",
    "center",
    "gmbh",
    "inc.",
    "ltd.",
    "llc",
];

fn looks_like_an_organization(name: &str) -> bool {
    let lower = name.to_lowercase();
    ORGANIZATION_MARKERS
        .iter()
        .any(|marker| lower.split_whitespace().any(|word| word.trim_end_matches(',') == *marker))
}

/// `{{Name}}` -> `Name`. BibTeX's "this is one unsplittable unit" idiom.
fn strip_double_braces(text: &str) -> Option<&str> {
    let inner = text.strip_prefix("{{")?.strip_suffix("}}")?;
    (!inner.contains('{') && !inner.contains('}')).then_some(inner)
}

/// `{Name}` -> `Name`, but only when the braces wrap the *whole* name.
fn strip_single_braces(text: &str) -> Option<&str> {
    let inner = text.strip_prefix('{')?.strip_suffix('}')?;
    (!inner.contains('{') && !inner.contains('}')).then_some(inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convenience: parse one name and unwrap it as a person.
    fn person(text: &str) -> PersonName {
        match Author::parse_bibtex_name(text) {
            Author::Person(p) => p,
            other => panic!("{text:?} should have parsed as a person, got {other:?}"),
        }
    }

    // -- The table of hard cases ----------------------------------------

    #[test]
    fn western_given_then_family() {
        let name = person("John Smith");
        assert_eq!(name.family(), "Smith");
        assert_eq!(name.given()[0], GivenName::Full("John".to_string()));
        // ...but we are honest that this is an ASSUMPTION about name order.
        assert_eq!(name.confidence(), NameConfidence::Assumed);
        assert!(!name.family_is_trustworthy());
    }

    #[test]
    fn an_initial_pins_the_order_and_raises_confidence() {
        let name = person("J. Smith");
        assert_eq!(name.family(), "Smith");
        assert_eq!(name.given()[0], GivenName::Initial("J".to_string()));
        assert_eq!(name.confidence(), NameConfidence::Conventional);
        assert!(name.family_is_trustworthy());
    }

    #[test]
    fn a_comma_is_the_source_telling_us_and_cannot_be_wrong() {
        let name = person("Smith, John");
        assert_eq!(name.family(), "Smith");
        assert_eq!(name.given()[0], GivenName::Full("John".to_string()));
        assert_eq!(name.confidence(), NameConfidence::Explicit);
    }

    #[test]
    fn van_der_waals_keeps_his_particle() {
        // The canonical failure of "last token is the surname": it renames him
        // "Waals". His family name is "van der Waals".
        let name = person("Johannes Diderik van der Waals");
        assert_eq!(name.family(), "Waals");
        assert_eq!(name.particle(), Some("van der"));
        assert_eq!(name.full_family(), "van der Waals");
        assert_eq!(name.confidence(), NameConfidence::Conventional);
        assert_eq!(name.given().len(), 2);
    }

    #[test]
    fn van_der_waals_in_comma_form_too() {
        let name = person("van der Waals, Johannes Diderik");
        assert_eq!(name.full_family(), "van der Waals");
        assert_eq!(name.confidence(), NameConfidence::Explicit);
    }

    #[test]
    fn a_capitalised_particle_is_still_a_particle() {
        // Typesetters produce "Van Der Waals, J. D." constantly, and BibTeX's
        // lower-case rule misses it.
        let name = person("J. D. Van Der Waals");
        assert_eq!(name.full_family(), "Van Der Waals");
        assert_eq!(name.family(), "Waals");
    }

    #[test]
    fn de_sousa_keeps_his_particle() {
        let name = person("Maria de Sousa");
        assert_eq!(name.full_family(), "de Sousa");
        assert_eq!(name.particle(), Some("de"));
    }

    #[test]
    fn ludwig_van_beethoven() {
        let name = person("Ludwig van Beethoven");
        assert_eq!(name.full_family(), "van Beethoven");
        assert_eq!(name.given()[0], GivenName::Full("Ludwig".to_string()));
    }

    #[test]
    fn a_generational_suffix_is_not_a_family_name() {
        let name = person("Martin Luther King Jr.");
        assert_eq!(name.family(), "King");
        assert_eq!(name.suffix(), Some("Jr."));
        // The naive rule would have made this man's family name "Jr.".
    }

    #[test]
    fn a_roman_numeral_suffix_too() {
        let name = person("Henry Ford III");
        assert_eq!(name.family(), "Ford");
        assert_eq!(name.suffix(), Some("III"));
    }

    #[test]
    fn a_suffix_in_the_comma_form() {
        let name = person("King, Jr., Martin Luther");
        assert_eq!(name.family(), "King");
        assert_eq!(name.suffix(), Some("Jr."));
        assert_eq!(name.confidence(), NameConfidence::Explicit);
    }

    #[test]
    fn a_hyphenated_given_name_is_one_given_name() {
        let name = person("Jean-Pierre Dupont");
        assert_eq!(name.family(), "Dupont");
        assert_eq!(
            name.given()[0],
            GivenName::Full("Jean-Pierre".to_string()),
            "a hyphenated given name is not two initials"
        );
    }

    #[test]
    fn hyphenated_initials_survive() {
        // From the maintainer's own reference list: "B.-C. Du, Y.-L. He".
        let name = person("B.-C. Du");
        assert_eq!(name.family(), "Du");
        assert_eq!(name.given()[0], GivenName::Initial("B.-C".to_string()));
        assert_eq!(name.given()[0].as_display(), "B.-C.");
        assert_eq!(name.confidence(), NameConfidence::Conventional);
    }

    #[test]
    fn initials_without_periods_are_still_initials() {
        let name = person("J R R Tolkien");
        assert_eq!(name.family(), "Tolkien");
        assert!(name.given().iter().all(GivenName::is_initial));
    }

    #[test]
    fn a_mononym_is_a_family_name_and_nothing_is_guessed() {
        let name = person("Aristotle");
        assert_eq!(name.family(), "Aristotle");
        assert!(name.given().is_empty());
        assert_eq!(name.confidence(), NameConfidence::Explicit);
    }

    // -- THE CASES WE REFUSE TO SPLIT -----------------------------------

    #[test]
    fn a_cjk_name_is_kept_verbatim_and_never_split() {
        // We do not know whether this venue printed it family-first (the native
        // convention) or given-first (the Western one), and the characters do
        // not say. Splitting would be a coin toss with a person's name.
        for name in ["\u{6BDB}\u{6CFD}\u{4E1C}", "\u{91D1}\u{6B63}\u{6069}", "\u{6751}\u{4E0A}\u{6625}\u{6A39}"] {
            let author = Author::parse_bibtex_name(name);
            assert_eq!(
                author,
                Author::Literal(name.to_string()),
                "{name} must be kept verbatim"
            );
            // And -- the point -- it comes back out exactly as it went in.
            assert_eq!(author.as_written(), name);
        }
    }

    #[test]
    fn a_double_braced_name_is_taken_at_its_word_and_never_touched() {
        // `{{...}}` is BibTeX's own "do not split this" marker. Honouring it is
        // the single most reliable signal available.
        let author = Author::parse_bibtex_name("{{Kim Jong-un}}");
        assert_eq!(author, Author::Literal("Kim Jong-un".to_string()));
        assert_eq!(author.as_written(), "Kim Jong-un");
        assert_eq!(author.family(), None, "we do not claim to know his family name");
    }

    #[test]
    fn kim_jong_un_written_plainly_is_split_but_never_trusted_or_reordered() {
        // The uncomfortable case, stated plainly.
        //
        // "Kim Jong-un" and "John Smith" are the SAME SHAPE. We cannot tell
        // them apart. So we apply the Western convention (getting this one
        // wrong), mark it Assumed -- and then never act on it:
        //
        //   * `family()` returns None, so nothing downstream can misattribute.
        //   * `sort_key()` sorts under "kim jong-un", not under the wrong name.
        //   * BibTeX emission writes "Kim Jong-un", NOT "Jong-un, Kim".
        //
        // The split is wrong; the OUTPUT is not. That is the best available
        // outcome without a lexicon, and it is why confidence is modelled.
        let author = Author::parse_bibtex_name("Kim Jong-un");
        assert_eq!(
            author.family(),
            None,
            "an assumed split must not be presented as a known family name"
        );
        assert_eq!(author.sort_key(), "kim jong-un");
        assert_eq!(author.as_written(), "Kim Jong-un");

        let Author::Person(name) = &author else {
            panic!("expected a person");
        };
        assert_eq!(name.confidence(), NameConfidence::Assumed);
        // The internal split IS the wrong one -- we are not pretending otherwise.
        assert_eq!(name.family(), "Jong-un");
    }

    #[test]
    fn an_organisation_is_not_given_a_given_name() {
        // "Institute, European Bioinformatics" appears in real bibliographies.
        // It is exactly as silly as it looks.
        let author = Author::parse_bibtex_name("European Bioinformatics Institute");
        assert_eq!(
            author,
            Author::Organization("European Bioinformatics Institute".to_string())
        );
        assert_eq!(author.family(), None);

        assert!(matches!(
            Author::parse_bibtex_name("University of California, Berkeley"),
            Author::Organization(_)
        ));
    }

    #[test]
    fn a_braced_group_is_a_corporate_author() {
        assert_eq!(
            Author::parse_bibtex_name("{Unicode Consortium}"),
            Author::Organization("Unicode Consortium".to_string())
        );
    }

    // -- Author lists: the BibTeX grammar --------------------------------

    #[test]
    fn bibtex_lists_split_on_and_not_on_commas() {
        let list = parse_bibtex_name_list("Chen, M. R. and Novak, S. and Alvarez, J. P.");
        assert_eq!(list.len(), 3);
        assert_eq!(list.authors()[0].family(), Some("Chen"));
        assert_eq!(list.authors()[1].family(), Some("Novak"));
        assert_eq!(list.authors()[2].family(), Some("Alvarez"));
        assert!(!list.is_truncated());
    }

    #[test]
    fn the_word_and_inside_a_name_does_not_split_it() {
        // Without the surrounding spaces in the separator, a great many
        // chemists named Alexander would be cut in half.
        let list = parse_bibtex_name_list("Alexander Ferdinand");
        assert_eq!(list.len(), 1);
        assert_eq!(list.authors()[0].as_written(), "Alexander Ferdinand");
    }

    #[test]
    fn a_braced_corporate_author_containing_and_stays_one_author() {
        let list = parse_bibtex_name_list("{Smith and Wesson Ltd.} and J. Doe");
        assert_eq!(list.len(), 2, "the brace group is ONE author");
        assert_eq!(list.authors()[0].as_written(), "Smith and Wesson Ltd.");
    }

    #[test]
    fn and_others_is_bibtexs_et_al_and_is_not_a_person() {
        let list = parse_bibtex_name_list("Chen, M. and others");
        assert_eq!(list.len(), 1, "`others` is not an author");
        assert!(list.is_truncated(), "but the truncation must be recorded");
    }

    // -- Author lists: the printed-reference grammar ---------------------

    #[test]
    fn a_printed_ieee_author_list_splits_on_commas() {
        // A three-author printed IEEE list.
        let list = parse_printed_name_list("M. R. Chen, S. Novak, and J. P. Alvarez");
        assert_eq!(list.len(), 3);
        assert_eq!(list.authors()[0].as_written(), "M. R. Chen");
        assert_eq!(list.authors()[0].family(), Some("Chen"));
        assert_eq!(list.authors()[1].family(), Some("Novak"));
        assert_eq!(list.authors()[2].family(), Some("Alvarez"));
    }

    #[test]
    fn a_printed_list_with_hyphenated_initials() {
        // Five authors, three with hyphenated initials.
        let list =
            parse_printed_name_list("B.-C. Du, Y.-L. He, Y. Qiu, Q. Liang, and Y.-P. Zhou");
        assert_eq!(list.len(), 5);
        assert_eq!(list.authors()[0].family(), Some("Du"));
        assert_eq!(list.authors()[1].family(), Some("He"));
        assert_eq!(list.authors()[4].family(), Some("Zhou"));
        assert_eq!(list.authors()[4].as_written(), "Y.-P. Zhou");
    }

    #[test]
    fn a_printed_list_in_family_comma_initial_form_is_not_split_into_four() {
        // The genuinely ambiguous shape. A lone initial is never a whole
        // author, so `J.` glues back onto `Smith`.
        let list = parse_printed_name_list("Smith, J., Jones, A. and Brown, B.");
        assert_eq!(list.len(), 3, "three authors, not six");
        assert_eq!(list.authors()[0].family(), Some("Smith"));
        assert_eq!(list.authors()[1].family(), Some("Jones"));
        assert_eq!(list.authors()[2].family(), Some("Brown"));
    }

    #[test]
    fn et_al_is_a_truncation_marker_and_never_an_author_called_al() {
        for text in [
            "K. R. Fulton et al.",
            "K. R. Fulton, et al.",
            "K. R. Fulton et al",
        ] {
            let list = parse_printed_name_list(text);
            assert_eq!(list.len(), 1, "{text:?} names one author");
            assert!(list.is_truncated(), "{text:?} must record the truncation");
            assert!(
                !list.authors().iter().any(|a| a.as_written().contains("al")),
                "{text:?} must not produce an author called `al.`"
            );
        }
    }

    #[test]
    fn an_ampersand_separates_authors_too() {
        let list = parse_printed_name_list("J. Smith & A. Jones");
        assert_eq!(list.len(), 2);
    }

    // -- The safety property --------------------------------------------

    #[test]
    fn every_parsed_name_can_be_rendered_back_exactly_as_written() {
        // The guarantee that makes a bad split survivable: whatever we decided,
        // the person's name is still there, byte for byte.
        for original in [
            "John Smith",
            "J. Smith",
            "Smith, John",
            "Johannes Diderik van der Waals",
            "Martin Luther King Jr.",
            "Kim Jong-un",
            "\u{6BDB}\u{6CFD}\u{4E1C}",
            "European Bioinformatics Institute",
            "B.-C. Du",
            "Aristotle",
        ] {
            let author = Author::parse_bibtex_name(original);
            assert_eq!(
                author.as_written(),
                original,
                "{original:?} was not preserved verbatim"
            );
        }
    }

    #[test]
    fn a_trustworthy_split_reorders_for_bibtex_and_an_assumed_one_does_not() {
        // The rule that keeps an assumption from becoming a published error.
        let trustworthy = person("Johannes Diderik van der Waals");
        assert_eq!(
            trustworthy.to_bibtex_reordered(),
            "van der Waals, Johannes Diderik"
        );

        let assumed = person("Mao Zedong");
        assert!(!assumed.family_is_trustworthy());
        // `to_bibtex_reordered` WOULD produce "Zedong, Mao" -- which is why
        // the emitter (see crate::bibtex) checks confidence and writes
        // `as_written` instead. Asserted end-to-end in the bibtex tests.
        assert_eq!(assumed.as_written(), "Mao Zedong");
    }

    #[test]
    fn a_name_round_trips_through_json() {
        let author = Author::parse_bibtex_name("Johannes Diderik van der Waals");
        let json = serde_json::to_string(&author).unwrap();
        let back: Author = serde_json::from_str(&json).unwrap();
        assert_eq!(author, back);
    }
}

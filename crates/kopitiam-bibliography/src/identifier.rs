//! Bibliographic identifiers, each with its own type and each **validated**.
//!
//! A DOI, an arXiv id, an ISBN and a URL are four different kinds of claim, and
//! collapsing them into `String` throws away the one property that makes them
//! useful: an identifier is *checkable*. An ISBN carries a checksum. A DOI has
//! a registrant prefix that must begin `10.`. An arXiv id encodes a year and a
//! month. These are not decoration — they are the difference between "here is
//! where to find this paper" and "here is a plausible-looking string".
//!
//! # Why validation is a correctness issue and not a nicety
//!
//! The single worst thing this crate could do is emit a **fabricated
//! identifier**. A wrong DOI in a `.bib` file propagates into a published
//! bibliography, where it resolves to *somebody else's paper* — the reader is
//! sent to the wrong work, with the citing author's name on the mistake.
//!
//! So every constructor here rejects what it cannot verify, and there is no
//! "best effort" path. A string that does not validate does not become a
//! half-trusted `Doi`; it stays a raw string in
//! [`Reference::unparsed`](crate::Reference::unparsed) where a human can see it.
//!
//! # What validation can and cannot tell you
//!
//! Be precise about this, because it is easy to overclaim:
//!
//! * A **syntactically valid** DOI is well-formed. It is **not** known to
//!   exist, and it is **not** known to identify the paper we think it does.
//!   Only a resolver can establish that, and this crate has no network — see
//!   [`crate::resolve`].
//! * A **checksum-valid** ISBN is *probably* a real ISBN (the checksum catches
//!   single-digit errors and most transpositions), but it too may name a book
//!   nobody ever printed.
//!
//! Validation rejects garbage. It does not confer truth. Nothing in this crate
//! ever claims otherwise.

use std::fmt;
use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

/// A string was offered as an identifier and was not one.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IdentifierError {
    /// Not a syntactically valid DOI.
    #[error("not a DOI: {0:?} (a DOI is a `10.` registrant prefix, a `/`, and a suffix)")]
    NotADoi(String),

    /// Not a syntactically valid arXiv identifier, in either the pre-2007
    /// (`math.GT/0309136`) or post-2007 (`2103.00020v2`) scheme.
    #[error("not an arXiv identifier: {0:?}")]
    NotAnArxivId(String),

    /// The right number of digits, but the check digit does not match.
    ///
    /// This is the interesting failure: the string *looks* like an ISBN and is
    /// not one. Almost always a typo or an OCR error — and exactly the case a
    /// length check alone would wave through.
    #[error("ISBN {isbn:?} has a bad check digit (computed {expected:?}, found {found:?})")]
    BadIsbnCheckDigit {
        /// The ISBN as offered, hyphens removed.
        isbn: String,
        /// The check digit the other digits imply.
        expected: String,
        /// The check digit that was actually printed.
        found: String,
    },

    /// Not 10 or 13 digits at all.
    #[error("not an ISBN: {0:?} (an ISBN has 10 or 13 digits)")]
    NotAnIsbn(String),

    /// An ISSN with a bad check digit.
    #[error("ISSN {issn:?} has a bad check digit (computed {expected:?}, found {found:?})")]
    BadIssnCheckDigit {
        /// The ISSN as offered.
        issn: String,
        /// The check digit the other digits imply.
        expected: String,
        /// The check digit that was actually printed.
        found: String,
    },

    /// Not 8 characters of the ISSN shape.
    #[error("not an ISSN: {0:?} (an ISSN is 8 digits, e.g. 0362-4331)")]
    NotAnIssn(String),

    /// Not a parseable absolute URL.
    #[error("not a URL: {0}")]
    NotAUrl(String),
}

// ---------------------------------------------------------------------------
// DOI
// ---------------------------------------------------------------------------

/// The shape of a DOI, per the DOI Handbook and CrossRef's own published
/// guidance: a registrant prefix that always begins `10.`, a slash, and an
/// opaque suffix the registrant chooses.
///
/// The suffix is deliberately permissive (`\S+`): registrants have used
/// parentheses, angle brackets, semicolons and `#` in real DOIs, and a
/// tighter pattern would reject genuine identifiers. The prefix is where the
/// structure actually is.
static DOI: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^10\.\d{4,9}/\S+$").unwrap());

/// A Digital Object Identifier.
///
/// # Normalisation
///
/// A DOI arrives dressed in whatever the source felt like: `doi:10.1016/...`,
/// `https://doi.org/10.1016/...`, `DOI: 10.1016/...`. [`Doi::parse`] strips all
/// of those and stores the bare identifier, because they are the same DOI and
/// storing three spellings of one identifier would make the citation graph
/// think one paper is three.
///
/// # Case
///
/// DOIs are **case-insensitive** but this type is **case-preserving**. That
/// combination is deliberate: comparison and hashing fold case (so
/// `10.1016/J.X` and `10.1016/j.x` are one node in the graph), while
/// [`Doi::as_str`] returns exactly what the document printed. Silently
/// re-casing a registrant's identifier is the sort of "helpful correction" this
/// crate refuses to make.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Doi(String);

impl Doi {
    /// Parses a DOI, stripping any `doi:` or resolver-URL wrapper.
    ///
    /// # Errors
    ///
    /// [`IdentifierError::NotADoi`] if what remains is not `10.NNNN/suffix`.
    pub fn parse(text: &str) -> Result<Self, IdentifierError> {
        let raw = text.trim();
        let stripped = strip_prefixes(
            raw,
            &[
                "https://doi.org/",
                "http://doi.org/",
                "https://dx.doi.org/",
                "http://dx.doi.org/",
                "doi.org/",
                "doi:",
                "DOI:",
                "doi: ",
                "DOI: ",
            ],
        )
        .trim()
        // A DOI at the end of a reference-list line usually carries the
        // sentence's full stop. It is punctuation, not part of the identifier.
        .trim_end_matches(['.', ',', ';']);

        if DOI.is_match(stripped) {
            Ok(Self(stripped.to_string()))
        } else {
            Err(IdentifierError::NotADoi(raw.to_string()))
        }
    }

    /// The DOI as printed, e.g. `10.1016/j.csl.2021.101144`.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The registrant prefix (`10.1016`), which identifies the publisher.
    pub fn prefix(&self) -> &str {
        self.0.split_once('/').map_or(&self.0, |(prefix, _)| prefix)
    }

    /// The canonical resolver URL, which is what belongs in a `url` field and
    /// what a reader should be given.
    pub fn resolver_url(&self) -> String {
        format!("https://doi.org/{}", self.0)
    }
}

/// Case-insensitive, per the DOI Handbook: `10.1016/J.X` and `10.1016/j.x` are
/// the same DOI, and the citation graph must not think they are two papers.
impl PartialEq for Doi {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq_ignore_ascii_case(&other.0)
    }
}

impl Eq for Doi {}

impl std::hash::Hash for Doi {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_ascii_lowercase().hash(state);
    }
}

impl fmt::Display for Doi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl TryFrom<String> for Doi {
    type Error = IdentifierError;

    fn try_from(text: String) -> Result<Self, Self::Error> {
        Self::parse(&text)
    }
}

impl From<Doi> for String {
    fn from(doi: Doi) -> Self {
        doi.0
    }
}

// ---------------------------------------------------------------------------
// arXiv
// ---------------------------------------------------------------------------

/// arXiv's identifier scheme **since April 2007**: `YYMM.NNNNN`, optionally
/// with a version suffix. Papers from 2007-03 to 2014-12 have four suffix
/// digits; 2015 onwards have five, because arXiv ran out of numbers.
static ARXIV_NEW: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(\d{2})(\d{2})\.(\d{4,5})(v\d+)?$").unwrap());

/// arXiv's **pre-2007** scheme: `archive.subject-class/YYMMNNN`, e.g.
/// `math.GT/0309136` or `hep-th/9901001`. Still perfectly valid identifiers —
/// a great deal of foundational physics is only citable this way, so refusing
/// them would be refusing the literature.
static ARXIV_OLD: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^([a-z][a-z-]+(?:\.[A-Z]{2})?)/(\d{2})(\d{2})(\d{3})(v\d+)?$").unwrap()
});

/// An arXiv identifier, in either of the two schemes arXiv has used.
///
/// # Why the month is validated
///
/// Both schemes encode `YYMM`. A month of `00` or `13` means the string is not
/// an arXiv id, however much it looks like one — and this is not hypothetical:
/// `2013.00020` is a very natural typo for `2103.00020`, it passes any
/// digit-count check, and it points at nothing.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ArxivId(String);

impl ArxivId {
    /// Parses an arXiv identifier, stripping an `arXiv:` or `arxiv.org/abs/`
    /// wrapper.
    ///
    /// # Errors
    ///
    /// [`IdentifierError::NotAnArxivId`] if it matches neither scheme, or if
    /// the encoded month is not 01-12.
    pub fn parse(text: &str) -> Result<Self, IdentifierError> {
        let raw = text.trim();
        let stripped = strip_prefixes(
            raw,
            &[
                "https://arxiv.org/abs/",
                "http://arxiv.org/abs/",
                "arxiv.org/abs/",
                "arXiv:",
                "arxiv:",
                "arXiv: ",
            ],
        )
        .trim()
        .trim_end_matches(['.', ',', ';']);

        let month = if let Some(caps) = ARXIV_NEW.captures(stripped) {
            caps[2].parse::<u8>().ok()
        } else if let Some(caps) = ARXIV_OLD.captures(stripped) {
            caps[3].parse::<u8>().ok()
        } else {
            return Err(IdentifierError::NotAnArxivId(raw.to_string()));
        };

        match month {
            Some(1..=12) => Ok(Self(stripped.to_string())),
            _ => Err(IdentifierError::NotAnArxivId(raw.to_string())),
        }
    }

    /// The identifier, e.g. `2103.00020v2`.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The abstract page, which is what a reader wants.
    pub fn url(&self) -> String {
        format!("https://arxiv.org/abs/{}", self.0)
    }
}

impl fmt::Display for ArxivId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl TryFrom<String> for ArxivId {
    type Error = IdentifierError;

    fn try_from(text: String) -> Result<Self, Self::Error> {
        Self::parse(&text)
    }
}

impl From<ArxivId> for String {
    fn from(id: ArxivId) -> Self {
        id.0
    }
}

// ---------------------------------------------------------------------------
// ISBN
// ---------------------------------------------------------------------------

/// An International Standard Book Number, **checksum-verified**.
///
/// # The checksums, and why they are computed rather than assumed
///
/// Counting to thirteen is not validation. Both ISBN forms carry a check digit
/// specifically so that a mistyped or misread identifier can be *detected*, and
/// a bibliography tool that accepts any 13 digits has thrown away the one
/// safeguard the standard gave it.
///
/// **ISBN-10** (ISO 2108, pre-2007): the twelve... nine digits are weighted
/// 10, 9, 8, ... 2, summed with the check digit weighted 1, and the total must
/// be ≡ 0 (mod 11). Because the residue can be 10, the check digit may be the
/// letter `X` — which is why an ISBN-10 is not a number and must never be
/// stored as one.
///
/// **ISBN-13** (an EAN-13 barcode in disguise): digits are weighted 1, 3, 1, 3,
/// ... alternating, summed, and the total must be ≡ 0 (mod 10). The check digit
/// is always 0-9; there is no `X`.
///
/// # Storage
///
/// Hyphens are stripped. They are a *display* convention marking the
/// registration-group / registrant / publication boundaries, those boundaries
/// vary by country and by agency, and reproducing them requires a lookup table
/// this crate does not have. `978-0-13-235088-4` and `9780132350884` are the
/// same book, and storing both spellings would split it into two nodes in the
/// citation graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Isbn(String);

impl Isbn {
    /// Parses and **checksum-verifies** an ISBN-10 or ISBN-13.
    ///
    /// # Errors
    ///
    /// * [`IdentifierError::NotAnIsbn`] — not 10 or 13 digit-ish characters.
    /// * [`IdentifierError::BadIsbnCheckDigit`] — right shape, wrong number.
    ///   This is the one that matters: the string looked like an ISBN and was
    ///   not one.
    pub fn parse(text: &str) -> Result<Self, IdentifierError> {
        let raw = text.trim();
        let stripped = strip_prefixes(raw, &["ISBN-13:", "ISBN-10:", "ISBN:", "isbn:"]).trim();
        let digits: String = stripped
            .chars()
            .filter(|c| !matches!(c, '-' | ' ' | '\u{2013}'))
            .collect();

        match digits.len() {
            10 => Self::check_isbn10(&digits, raw),
            13 => Self::check_isbn13(&digits, raw),
            _ => Err(IdentifierError::NotAnIsbn(raw.to_string())),
        }
    }

    /// Weighted sum 10..=1, modulo 11; check digit may be `X` (= 10).
    fn check_isbn10(digits: &str, raw: &str) -> Result<Self, IdentifierError> {
        let chars: Vec<char> = digits.chars().collect();

        // Only the final character may be `X`.
        if !chars[..9].iter().all(char::is_ascii_digit) {
            return Err(IdentifierError::NotAnIsbn(raw.to_string()));
        }
        let found = chars[9].to_ascii_uppercase();
        if !found.is_ascii_digit() && found != 'X' {
            return Err(IdentifierError::NotAnIsbn(raw.to_string()));
        }

        let sum: u32 = chars[..9]
            .iter()
            .enumerate()
            .map(|(i, c)| (10 - i as u32) * c.to_digit(10).unwrap_or(0))
            .sum();

        // The check digit c must satisfy sum + 1*c ≡ 0 (mod 11).
        let expected_value = (11 - (sum % 11)) % 11;
        let expected = if expected_value == 10 {
            'X'
        } else {
            char::from_digit(expected_value, 10).unwrap_or('?')
        };

        if expected == found {
            Ok(Self(digits.to_ascii_uppercase()))
        } else {
            Err(IdentifierError::BadIsbnCheckDigit {
                isbn: digits.to_string(),
                expected: expected.to_string(),
                found: found.to_string(),
            })
        }
    }

    /// EAN-13: weights alternate 1, 3, modulo 10.
    fn check_isbn13(digits: &str, raw: &str) -> Result<Self, IdentifierError> {
        if !digits.chars().all(|c| c.is_ascii_digit()) {
            return Err(IdentifierError::NotAnIsbn(raw.to_string()));
        }
        let chars: Vec<u32> = digits.chars().filter_map(|c| c.to_digit(10)).collect();

        let sum: u32 = chars[..12]
            .iter()
            .enumerate()
            .map(|(i, d)| if i % 2 == 0 { *d } else { 3 * d })
            .sum();

        let expected_value = (10 - (sum % 10)) % 10;
        let found_value = chars[12];

        if expected_value == found_value {
            Ok(Self(digits.to_string()))
        } else {
            Err(IdentifierError::BadIsbnCheckDigit {
                isbn: digits.to_string(),
                expected: expected_value.to_string(),
                found: found_value.to_string(),
            })
        }
    }

    /// The ISBN, hyphens stripped.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Whether this is a 13-digit ISBN.
    pub fn is_isbn13(&self) -> bool {
        self.0.len() == 13
    }
}

impl fmt::Display for Isbn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl TryFrom<String> for Isbn {
    type Error = IdentifierError;

    fn try_from(text: String) -> Result<Self, Self::Error> {
        Self::parse(&text)
    }
}

impl From<Isbn> for String {
    fn from(isbn: Isbn) -> Self {
        isbn.0
    }
}

// ---------------------------------------------------------------------------
// ISSN
// ---------------------------------------------------------------------------

/// An International Standard Serial Number, **checksum-verified** — the
/// identifier of a *journal*, not of a paper.
///
/// Eight digits, the last a mod-11 check digit computed with weights 8..=2 (and
/// so, like ISBN-10, it may be `X`). Printed with a hyphen after the fourth
/// digit, which is part of the standard's display form and is preserved here
/// because — unlike ISBN's variable-width groups — its position is fixed and
/// therefore losslessly reconstructible.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Issn(String);

impl Issn {
    /// Parses and checksum-verifies an ISSN.
    ///
    /// # Errors
    ///
    /// [`IdentifierError::NotAnIssn`] or [`IdentifierError::BadIssnCheckDigit`].
    pub fn parse(text: &str) -> Result<Self, IdentifierError> {
        let raw = text.trim();
        let stripped = strip_prefixes(raw, &["ISSN:", "issn:", "ISSN"]).trim();
        let digits: String = stripped
            .chars()
            .filter(|c| !matches!(c, '-' | ' ' | '\u{2013}'))
            .collect();

        if digits.len() != 8 {
            return Err(IdentifierError::NotAnIssn(raw.to_string()));
        }
        let chars: Vec<char> = digits.chars().collect();
        if !chars[..7].iter().all(char::is_ascii_digit) {
            return Err(IdentifierError::NotAnIssn(raw.to_string()));
        }
        let found = chars[7].to_ascii_uppercase();
        if !found.is_ascii_digit() && found != 'X' {
            return Err(IdentifierError::NotAnIssn(raw.to_string()));
        }

        let sum: u32 = chars[..7]
            .iter()
            .enumerate()
            .map(|(i, c)| (8 - i as u32) * c.to_digit(10).unwrap_or(0))
            .sum();

        let expected_value = (11 - (sum % 11)) % 11;
        let expected = if expected_value == 10 {
            'X'
        } else {
            char::from_digit(expected_value, 10).unwrap_or('?')
        };

        if expected == found {
            Ok(Self(format!(
                "{}-{}",
                &digits[..4],
                digits[4..].to_ascii_uppercase()
            )))
        } else {
            Err(IdentifierError::BadIssnCheckDigit {
                issn: digits.to_string(),
                expected: expected.to_string(),
                found: found.to_string(),
            })
        }
    }

    /// The ISSN in its printed form, e.g. `0029-5493`.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Issn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl TryFrom<String> for Issn {
    type Error = IdentifierError;

    fn try_from(text: String) -> Result<Self, Self::Error> {
        Self::parse(&text)
    }
}

impl From<Issn> for String {
    fn from(issn: Issn) -> Self {
        issn.0
    }
}

// ---------------------------------------------------------------------------
// URL
// ---------------------------------------------------------------------------

/// An absolute URL where a work can be found.
///
/// Backed by the `url` crate (WHATWG URL Standard) rather than a regex,
/// because "looks roughly like a URL" is exactly how a corrupted link — one
/// that was line-broken by a typesetter and glued back together wrong — passes
/// review and ships in a bibliography. See [`crate::text`] for the joining
/// problem this is the last line of defence against.
///
/// A relative URL is rejected: a citation must be followable from anywhere, and
/// `../papers/x.pdf` is not.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ResourceUrl(String);

impl ResourceUrl {
    /// Parses an absolute URL.
    ///
    /// # Errors
    ///
    /// [`IdentifierError::NotAUrl`] if it will not parse as an absolute URL.
    pub fn parse(text: &str) -> Result<Self, IdentifierError> {
        // A URL at the end of a reference-list sentence usually swallows the
        // full stop or comma that terminates the sentence. Neither is ever a
        // meaningful final character of a URL in practice.
        let raw = text.trim().trim_end_matches(['.', ',', ';']);
        match url::Url::parse(raw) {
            Ok(parsed) if parsed.has_host() || parsed.scheme() == "urn" => {
                Ok(Self(parsed.to_string()))
            }
            Ok(_) => Err(IdentifierError::NotAUrl(format!(
                "{raw:?} has no host (a citation must be followable)"
            ))),
            Err(error) => Err(IdentifierError::NotAUrl(format!("{raw:?}: {error}"))),
        }
    }

    /// The URL.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ResourceUrl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl TryFrom<String> for ResourceUrl {
    type Error = IdentifierError;

    fn try_from(text: String) -> Result<Self, Self::Error> {
        Self::parse(&text)
    }
}

impl From<ResourceUrl> for String {
    fn from(url: ResourceUrl) -> Self {
        url.0
    }
}

// ---------------------------------------------------------------------------
// The set
// ---------------------------------------------------------------------------

/// Every identifier a [`crate::Reference`] carries.
///
/// All optional, and **an absent identifier is a fact, not a gap to be filled
/// in**. A typical conference paper's reference list contains a dozen
/// references and *zero* DOIs — which is entirely normal — and the correct
/// behaviour is to record a dozen references with no DOIs, not to go looking
/// for a dozen plausible ones. See [`crate::resolve`] for what would have to
/// happen to fill them in honestly.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Identifiers {
    /// The DOI, if the source printed one.
    pub doi: Option<Doi>,
    /// The arXiv id, if the source printed one.
    pub arxiv: Option<ArxivId>,
    /// The ISBN, if the source printed one (books and theses).
    pub isbn: Option<Isbn>,
    /// The journal's ISSN, if the source printed one.
    pub issn: Option<Issn>,
    /// A URL, if the source printed one (software, preprints, reports).
    pub url: Option<ResourceUrl>,
}

impl Identifiers {
    /// Whether *any* identifier is present.
    ///
    /// `false` means this reference cannot currently be resolved to a single
    /// work by machine — see [`crate::resolve::ResolutionRequest`], which is
    /// exactly the set of references for which this is `false`.
    pub fn any(&self) -> bool {
        self.doi.is_some()
            || self.arxiv.is_some()
            || self.isbn.is_some()
            || self.url.is_some()
    }

    /// The best available link for a human reader, preferring the DOI (the only
    /// identifier with a persistence guarantee behind it) over a bare URL
    /// (which rots).
    pub fn best_link(&self) -> Option<String> {
        self.doi
            .as_ref()
            .map(Doi::resolver_url)
            .or_else(|| self.arxiv.as_ref().map(ArxivId::url))
            .or_else(|| self.url.as_ref().map(|u| u.as_str().to_string()))
    }
}

/// Removes the first matching prefix, case-sensitively, in the order given.
fn strip_prefixes<'a>(text: &'a str, prefixes: &[&str]) -> &'a str {
    for prefix in prefixes {
        if let Some(rest) = text.strip_prefix(prefix) {
            return rest;
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- DOI ------------------------------------------------------------

    #[test]
    fn accepts_a_real_doi() {
        // A DOI on an Elsevier journal (registrant prefix 10.1016).
        // Syntactically valid; this crate does not and cannot claim it resolves.
        let doi = Doi::parse("10.1016/j.csl.2021.101144").unwrap();
        assert_eq!(doi.as_str(), "10.1016/j.csl.2021.101144");
        assert_eq!(doi.prefix(), "10.1016");
        assert_eq!(
            doi.resolver_url(),
            "https://doi.org/10.1016/j.csl.2021.101144"
        );
    }

    #[test]
    fn strips_resolver_and_scheme_wrappers() {
        let bare = Doi::parse("10.1016/j.x").unwrap();
        for dressed in [
            "https://doi.org/10.1016/j.x",
            "http://dx.doi.org/10.1016/j.x",
            "doi:10.1016/j.x",
            "DOI: 10.1016/j.x",
        ] {
            assert_eq!(Doi::parse(dressed).unwrap(), bare, "failed on {dressed}");
        }
    }

    #[test]
    fn a_doi_is_compared_case_insensitively_but_stored_as_printed() {
        let upper = Doi::parse("10.1016/J.CSL").unwrap();
        let lower = Doi::parse("10.1016/j.csl").unwrap();
        // The same DOI: one node in the citation graph, not two.
        assert_eq!(upper, lower);
        // But we did not "correct" the registrant's casing.
        assert_eq!(upper.as_str(), "10.1016/J.CSL");
    }

    #[test]
    fn rejects_things_that_are_not_dois() {
        for bad in [
            "10.1016",             // no suffix
            "11.1016/j.x",         // DOIs always start 10.
            "10.1/x",              // registrant prefix is 4-9 digits
            "not a doi",
            "",
            "https://example.org/paper.pdf",
        ] {
            assert!(Doi::parse(bad).is_err(), "should have rejected {bad:?}");
        }
    }

    #[test]
    fn a_trailing_sentence_full_stop_is_not_part_of_the_doi() {
        // Reference lists end with a full stop. It belongs to the sentence.
        assert_eq!(
            Doi::parse("10.1016/j.x.").unwrap().as_str(),
            "10.1016/j.x"
        );
    }

    // -- arXiv ----------------------------------------------------------

    #[test]
    fn accepts_both_arxiv_schemes() {
        // Post-2007. (This is CLIP's identifier -- a real, well-known one.)
        assert_eq!(ArxivId::parse("2103.00020").unwrap().as_str(), "2103.00020");
        assert_eq!(
            ArxivId::parse("arXiv:2103.00020v2").unwrap().as_str(),
            "2103.00020v2"
        );
        assert_eq!(
            ArxivId::parse("https://arxiv.org/abs/2103.00020").unwrap().as_str(),
            "2103.00020"
        );
        // Pre-2007, with and without a subject class. A great deal of
        // foundational physics is only citable this way.
        assert_eq!(
            ArxivId::parse("math.GT/0309136").unwrap().as_str(),
            "math.GT/0309136"
        );
        assert_eq!(
            ArxivId::parse("hep-th/9901001").unwrap().as_str(),
            "hep-th/9901001"
        );
    }

    #[test]
    fn rejects_an_arxiv_id_with_an_impossible_month() {
        // `2013.00020` is a very natural transposition of `2103.00020`. It has
        // the right number of digits and passes every length check -- and it
        // encodes month 13, so it cannot be an arXiv id.
        assert!(ArxivId::parse("2013.00020").is_err());
        assert!(ArxivId::parse("2100.00020").is_err(), "month 00 does not exist");
        assert!(ArxivId::parse("math.GT/0313136").is_err());
    }

    #[test]
    fn rejects_things_that_are_not_arxiv_ids() {
        for bad in ["2103.001", "10.1016/j.x", "", "arXiv:", "21030.0020"] {
            assert!(ArxivId::parse(bad).is_err(), "should have rejected {bad:?}");
        }
    }

    #[test]
    fn arxiv_url_points_at_the_abstract() {
        assert_eq!(
            ArxivId::parse("2103.00020").unwrap().url(),
            "https://arxiv.org/abs/2103.00020"
        );
    }

    // -- ISBN -----------------------------------------------------------

    #[test]
    fn accepts_a_real_isbn13_and_reconstructs_its_check_digit() {
        // "The Pragmatic Programmer", 978-0-13-235088-4. A genuine ISBN.
        let isbn = Isbn::parse("978-0-13-235088-4").unwrap();
        assert_eq!(isbn.as_str(), "9780132350884");
        assert!(isbn.is_isbn13());
    }

    #[test]
    fn accepts_a_real_isbn10_with_an_x_check_digit() {
        // 0-8044-2957-X -- the `X` is the whole reason an ISBN-10 is not a
        // number and must never be stored as one.
        let isbn = Isbn::parse("0-8044-2957-X").unwrap();
        assert_eq!(isbn.as_str(), "080442957X");
        assert!(!isbn.is_isbn13());
    }

    #[test]
    fn rejects_a_bad_isbn13_check_digit() {
        // The same book with the last digit wrong. Thirteen digits, correct
        // shape, and NOT an ISBN. A length check waves this straight through.
        let error = Isbn::parse("978-0-13-235088-5").unwrap_err();
        assert_eq!(
            error,
            IdentifierError::BadIsbnCheckDigit {
                isbn: "9780132350885".to_string(),
                expected: "4".to_string(),
                found: "5".to_string(),
            }
        );
    }

    #[test]
    fn rejects_a_transposition_that_breaks_the_checksum() {
        // Two adjacent digits swapped: 235088 -> 253088. This is the classic
        // typo the check digit exists to catch.
        assert!(matches!(
            Isbn::parse("9780132530884"),
            Err(IdentifierError::BadIsbnCheckDigit { .. })
        ));
    }

    #[test]
    fn rejects_a_bad_isbn10_check_digit() {
        assert!(matches!(
            Isbn::parse("0-8044-2957-1"),
            Err(IdentifierError::BadIsbnCheckDigit { .. })
        ));
    }

    #[test]
    fn rejects_things_that_are_not_isbns_at_all() {
        for bad in ["12345", "978013235088456", "", "abcdefghij"] {
            assert!(
                matches!(Isbn::parse(bad), Err(IdentifierError::NotAnIsbn(_))),
                "should have rejected {bad:?} as not-an-ISBN"
            );
        }
    }

    #[test]
    fn an_x_is_only_legal_as_the_final_check_digit() {
        assert!(matches!(
            Isbn::parse("X80442957X"),
            Err(IdentifierError::NotAnIsbn(_))
        ));
    }

    // -- ISSN -----------------------------------------------------------

    #[test]
    fn accepts_a_real_issn() {
        // The New York Times, 0362-4331. A genuine ISSN.
        let issn = Issn::parse("0362-4331").unwrap();
        assert_eq!(issn.as_str(), "0362-4331");
        // Printed without its hyphen, it is still the same serial.
        assert_eq!(Issn::parse("03624331").unwrap(), issn);
    }

    #[test]
    fn rejects_a_bad_issn_check_digit() {
        assert!(matches!(
            Issn::parse("0362-4332"),
            Err(IdentifierError::BadIssnCheckDigit { .. })
        ));
    }

    // -- URL ------------------------------------------------------------

    #[test]
    fn accepts_a_repository_github_url() {
        // A repository reference -- and the exact string that a naive line-join
        // would have corrupted into "https://github.com/ openalign/...".
        let url =
            ResourceUrl::parse("https://github.com/openalign/mtat_toolkit").unwrap();
        assert_eq!(
            url.as_str(),
            "https://github.com/openalign/mtat_toolkit"
        );
    }

    #[test]
    fn rejects_a_url_with_no_host_and_a_relative_path() {
        assert!(ResourceUrl::parse("../papers/x.pdf").is_err());
        assert!(ResourceUrl::parse("not a url").is_err());
    }

    #[test]
    fn a_trailing_full_stop_is_not_part_of_the_url() {
        assert_eq!(
            ResourceUrl::parse("https://example.org/paper,").unwrap().as_str(),
            "https://example.org/paper"
        );
    }

    // -- The set --------------------------------------------------------

    #[test]
    fn an_empty_identifier_set_reports_honestly_that_it_has_nothing() {
        let ids = Identifiers::default();
        assert!(!ids.any());
        assert_eq!(ids.best_link(), None);
    }

    #[test]
    fn the_doi_is_preferred_as_a_link_because_it_is_the_one_that_persists() {
        let ids = Identifiers {
            doi: Some(Doi::parse("10.1016/j.x").unwrap()),
            url: Some(ResourceUrl::parse("https://example.org/paper").unwrap()),
            ..Default::default()
        };
        assert_eq!(ids.best_link().unwrap(), "https://doi.org/10.1016/j.x");
    }

    #[test]
    fn identifiers_deserialise_through_their_validating_constructors() {
        // A hand-written JSON blob must not be able to smuggle in a DOI that
        // the constructor would have refused.
        let err = serde_json::from_str::<Doi>(r#""not-a-doi""#).unwrap_err();
        assert!(err.to_string().contains("not a DOI"), "got: {err}");

        let err = serde_json::from_str::<Isbn>(r#""978-0-13-235088-5""#).unwrap_err();
        assert!(err.to_string().contains("check digit"), "got: {err}");
    }
}

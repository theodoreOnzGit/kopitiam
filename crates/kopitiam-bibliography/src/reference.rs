//! [`Reference`] — one bibliographic record, and the strongly-typed fields it
//! is made of.
//!
//! # Why these are not all `String`
//!
//! A page range is not a string; it is a start and (sometimes) an end, and
//! `281-301` must be splittable into 281 and 301 or a reader cannot be told how
//! long the paper is. A year is not a string; `2024` and `2O24` are not the same
//! and only one of them is a year. A citation key is not a string; a key with a
//! space in it silently breaks every LaTeX document it appears in.
//!
//! Each type below therefore validates what it is, and — the recurring theme of
//! this crate — **declines rather than guesses** when it cannot.
//!
//! # What is deliberately absent
//!
//! There is no `Reference::is_peer_reviewed()`, no `Reference::quality()`, no
//! `Reference::is_credible()`. Those are judgments about scholarship, they are
//! not in the reference string, and a crate that manufactured them would be
//! lending machine authority to an opinion it is not entitled to hold.
//!
//! What is here is: *"the source printed these words; here is what they appear
//! to mean; here is what I could not work out; go and look."*

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::author::AuthorList;
use crate::identifier::Identifiers;
use crate::provenance::Provenance;

/// A field value was offered and was not a valid instance of its type.
///
/// A real error enum rather than a bare `i32`/`String`, so that these compose
/// with `?` in a caller's `Result<_, Box<dyn Error>>` like every other error in
/// the workspace. (This crate's own doctest caught the omission — which is what
/// doctests are for.)
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FieldError {
    /// A year outside 1000-2999. See [`Year`] for why the range is what it is.
    #[error("{0} is not a publication year (expected 1000-2999)")]
    YearOutOfRange(i32),

    /// A citation key that would break the LaTeX document it appears in.
    #[error(
        "{0:?} is not a usable citation key \
         (a key may not be empty or contain whitespace or any of {{}}(),\\\"#%'~^=)"
    )]
    UnusableCitationKey(String),
}

/// The kind of work being cited.
///
/// Deliberately mapped one-to-one onto BibTeX's entry types where one exists,
/// because that mapping is what every LaTeX document in the world already
/// assumes, and inventing a parallel taxonomy would mean a translation layer
/// that could only lose information.
///
/// [`Self::Unknown`] is a first-class member, not a failure. A reference-list
/// line that names a title, an author and a year but gives no clue whether the
/// venue was a journal or a workshop **is** a reference of unknown kind, and
/// saying so is more useful than picking `Misc` and looking confident.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntryKind {
    /// A journal article. BibTeX `@article`.
    Article,
    /// A paper in conference proceedings. BibTeX `@inproceedings`.
    InProceedings,
    /// A whole book. BibTeX `@book`.
    Book,
    /// A chapter or section within a book. BibTeX `@incollection`.
    InCollection,
    /// A PhD or master's thesis. BibTeX `@phdthesis` / `@mastersthesis`.
    Thesis,
    /// An institutional or laboratory report. BibTeX `@techreport`.
    ///
    /// The backbone of the technical-report literature — university labs,
    /// standards bodies and government agencies issue reports that are primary
    /// sources, not grey literature, and a bibliography engine that treated them
    /// as `Misc` would be useless.
    TechReport,
    /// A preprint (arXiv and similar). BibTeX `@misc`; BibLaTeX `@online`.
    Preprint,
    /// Software, a dataset, or a code repository. BibLaTeX `@software`.
    ///
    /// A first-class kind because it is a first-class scientific artefact.
    /// A modern reference list very often cites a Git repository directly.
    Software,
    /// A web page. BibLaTeX `@online`.
    Online,
    /// A patent. BibTeX `@patent`.
    Patent,
    /// An unpublished manuscript. BibTeX `@unpublished`.
    Unpublished,
    /// Anything else the source named explicitly. BibTeX `@misc`.
    Misc,
    /// **The kind could not be determined from the source.**
    ///
    /// An honest answer, and the right one for a reference-list line that
    /// simply does not say. Not a synonym for [`Self::Misc`], which means "the
    /// source said `@misc`".
    Unknown,
}

impl EntryKind {
    /// The BibTeX entry type name (`article`, `inproceedings`, ...).
    ///
    /// [`Self::Unknown`] emits as `misc`, because `@unknown` is not a BibTeX
    /// type and emitting it would produce a `.bib` file that `bibtex` refuses.
    /// The uncertainty is not lost: it is recorded as a [`crate::Anomaly`] and
    /// in the emitted `note` field.
    pub fn bibtex_type(self) -> &'static str {
        match self {
            Self::Article => "article",
            Self::InProceedings => "inproceedings",
            Self::Book => "book",
            Self::InCollection => "incollection",
            Self::Thesis => "phdthesis",
            Self::TechReport => "techreport",
            Self::Preprint | Self::Misc | Self::Unknown => "misc",
            Self::Software => "software",
            Self::Online => "online",
            Self::Patent => "patent",
            Self::Unpublished => "unpublished",
        }
    }

    /// The BibLaTeX entry type, which has richer vocabulary than BibTeX and can
    /// say `@online`, `@software` and `@report` properly.
    pub fn biblatex_type(self) -> &'static str {
        match self {
            Self::Article => "article",
            Self::InProceedings => "inproceedings",
            Self::Book => "book",
            Self::InCollection => "incollection",
            Self::Thesis => "thesis",
            Self::TechReport => "report",
            Self::Preprint | Self::Online => "online",
            Self::Software => "software",
            Self::Patent => "patent",
            Self::Unpublished => "unpublished",
            Self::Misc | Self::Unknown => "misc",
        }
    }

    /// Hayagriva's (Typst's) type vocabulary.
    pub fn hayagriva_type(self) -> &'static str {
        match self {
            Self::Article => "article",
            Self::InProceedings => "article",
            Self::Book => "book",
            Self::InCollection => "chapter",
            Self::Thesis => "thesis",
            Self::TechReport => "report",
            Self::Preprint | Self::Online => "web",
            Self::Software => "repository",
            Self::Patent => "patent",
            Self::Unpublished | Self::Misc | Self::Unknown => "misc",
        }
    }

    /// Reads a BibTeX/BibLaTeX entry-type name.
    ///
    /// An unrecognised type becomes [`Self::Misc`] — the source *did* name a
    /// type, we just do not model it, which is different from the source not
    /// naming one ([`Self::Unknown`]).
    pub fn from_bibtex_type(name: &str) -> Self {
        match name.to_lowercase().as_str() {
            "article" => Self::Article,
            "inproceedings" | "conference" => Self::InProceedings,
            "book" | "booklet" => Self::Book,
            "incollection" | "inbook" => Self::InCollection,
            "phdthesis" | "mastersthesis" | "thesis" => Self::Thesis,
            "techreport" | "report" => Self::TechReport,
            "software" => Self::Software,
            "online" | "electronic" | "www" => Self::Online,
            "patent" => Self::Patent,
            "unpublished" => Self::Unpublished,
            _ => Self::Misc,
        }
    }
}

impl fmt::Display for EntryKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.bibtex_type())
    }
}

/// A publication year.
///
/// # Why a range check
///
/// A four-digit number in a reference string is not necessarily a year: page
/// numbers, volume numbers, report numbers and article numbers are all
/// four-digit numbers, and a real reference list can contain a page number
/// (`111144`) longer than any year. So [`Year::parse`] accepts only
/// 1000-2999 — wide enough for the entire printed scientific record (and
/// for a work in press dated a year or two ahead), narrow enough to reject the
/// page number, the volume number, and the postcode.
///
/// Years before 1000 exist in the humanities. They are out of scope for a
/// bibliography/literature workbench, and admitting three-digit numbers would
/// let every volume number in the corpus become a year.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "i32", into = "i32")]
pub struct Year(i32);

impl Year {
    /// Builds a year.
    ///
    /// # Errors
    ///
    /// [`FieldError::YearOutOfRange`] if outside 1000-2999.
    pub fn new(year: i32) -> Result<Self, FieldError> {
        if (1000..=2999).contains(&year) {
            Ok(Self(year))
        } else {
            Err(FieldError::YearOutOfRange(year))
        }
    }

    /// Parses a year from text, ignoring surrounding punctuation.
    pub fn parse(text: &str) -> Option<Self> {
        let cleaned: String = text
            .chars()
            .filter(|c| c.is_ascii_digit())
            .collect();
        cleaned.parse::<i32>().ok().and_then(|y| Self::new(y).ok())
    }

    /// The year.
    pub fn get(self) -> i32 {
        self.0
    }
}

impl TryFrom<i32> for Year {
    type Error = FieldError;

    fn try_from(year: i32) -> Result<Self, Self::Error> {
        Self::new(year)
    }
}

impl From<Year> for i32 {
    fn from(year: Year) -> Self {
        year.0
    }
}

impl fmt::Display for Year {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// One page, as printed.
///
/// **A `String`, not a number**, and that is not laziness. Real page
/// designators include:
///
/// * roman numerals in front matter — `xii`;
/// * article numbers rather than pages — `e0123456` (PLOS), `111144`
///   (Elsevier);
/// * section-prefixed pages — `S14`, `II-7`.
///
/// Parsing those into an integer either fails or, worse, silently succeeds on a
/// prefix. So the printed form is kept, and [`Page::as_number`] is offered for
/// the (common) case where it *is* a number and a caller wants to count.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Page(String);

impl Page {
    /// Wraps a page designator. Returns `None` for empty input.
    pub fn new(text: impl Into<String>) -> Option<Self> {
        let text = text.into();
        let trimmed = text.trim();
        (!trimmed.is_empty()).then(|| Self(trimmed.to_string()))
    }

    /// The page as printed.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The page as a number, if it is one.
    ///
    /// `None` for `xii`, for `e0123456`, and for `S14` — all of which are real
    /// page designators, and none of which are numbers.
    pub fn as_number(&self) -> Option<u32> {
        self.0.parse().ok()
    }
}

impl fmt::Display for Page {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// The pages a work occupies: a start, and an end if the source gave one.
///
/// A single-page reference (`p. 111144`) is a `PageRange` with no end, not a
/// range from 111144 to 111144 — because the source did not say the work is one
/// page long, it said the work *starts* there. Frequently, for an
/// article-numbered paper, there is no such thing as an end page at all.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PageRange {
    start: Page,
    end: Option<Page>,
}

impl PageRange {
    /// A range from `start` to `end`.
    pub fn new(start: Page, end: Option<Page>) -> Self {
        Self { start, end }
    }

    /// The first page.
    pub fn start(&self) -> &Page {
        &self.start
    }

    /// The last page, if the source named one.
    pub fn end(&self) -> Option<&Page> {
        self.end.as_ref()
    }

    /// How many pages the work occupies, when both ends are numeric.
    ///
    /// `None` when the source gave one page, or when either end is not a number
    /// (`xii-xv`, `e0123456`). Deliberately not "1" in the single-page case:
    /// see the type docs.
    pub fn page_count(&self) -> Option<u32> {
        let start = self.start.as_number()?;
        let end = self.end.as_ref()?.as_number()?;
        end.checked_sub(start).map(|n| n + 1)
    }
}

impl fmt::Display for PageRange {
    /// The BibTeX form, with an en-dash written as `--` (TeX's ligature for
    /// one), which is what a `.bib` file is supposed to contain.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.end {
            Some(end) => write!(f, "{}--{}", self.start, end),
            None => fmt::Display::fmt(&self.start, f),
        }
    }
}

/// A citation key: the label a LaTeX document uses to `\cite` a reference.
///
/// # Why a type
///
/// A key containing a space, a comma, or a brace does not merely look untidy —
/// it **breaks the document**, usually with an error message pointing somewhere
/// else entirely. `\cite{chen 2024}` is not a citation of `chen 2024`; it is a
/// syntax error. So the characters BibTeX reserves are rejected here, at the
/// point the key is built, rather than in a LaTeX log at 2am.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct CitationKey(String);

impl CitationKey {
    /// Builds a citation key.
    ///
    /// # Errors
    ///
    /// [`FieldError::UnusableCitationKey`] if it is empty or contains
    /// whitespace or any of BibTeX's reserved characters (`{}(),\\"#%'~^` and
    /// `=`).
    pub fn new(key: impl Into<String>) -> Result<Self, FieldError> {
        let key = key.into();
        let trimmed = key.trim();
        if trimmed.is_empty() {
            return Err(FieldError::UnusableCitationKey(key));
        }
        let illegal = |c: char| {
            c.is_whitespace()
                || matches!(
                    c,
                    '{' | '}' | '(' | ')' | ',' | '\\' | '"' | '#' | '%' | '\'' | '~' | '^' | '='
                )
        };
        if trimmed.chars().any(illegal) {
            return Err(FieldError::UnusableCitationKey(key));
        }
        Ok(Self(trimmed.to_string()))
    }

    /// The key.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for CitationKey {
    type Error = FieldError;

    fn try_from(key: String) -> Result<Self, Self::Error> {
        Self::new(key)
    }
}

impl From<CitationKey> for String {
    fn from(key: CitationKey) -> Self {
        key.0
    }
}

impl fmt::Display for CitationKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// One bibliographic record.
///
/// Fields are private and every one of them is optional **except the
/// provenance**, which is required by [`Reference::builder`]. That asymmetry is
/// the crate's thesis in a struct definition: a reference with no title is a
/// poor reference but an honest one; a reference with no source is a rumour.
///
/// # A partially-understood reference is a useful reference
///
/// Most reference-list lines will not yield every field. A record with an
/// author, a year and a title, and nothing else, is worth having: it is enough
/// to identify the work to a human, enough to key a citation, and enough to
/// hand to a resolver later. So there is no "complete or nothing" validation.
/// What did not parse is kept in [`Reference::unparsed`] where it can be seen.
///
/// ```
/// use kopitiam_bibliography::{DocumentId, EntryKind, Provenance, Reference, Year};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let doc = DocumentId::new("paper.pdf")?;
/// let provenance = Provenance::from_page(&doc, 14, "M. R. Chen, \"MTAT,\" 2024.")?;
///
/// let reference = Reference::builder(provenance)
///     .kind(EntryKind::Article)
///     .title("An open-source toolkit for multilingual text alignment")
///     .year(Year::new(2024)?)
///     .build();
///
/// assert_eq!(reference.year().map(|y| y.get()), Some(2024));
/// // ...and it still knows where it came from.
/// assert_eq!(reference.provenance().locator().page().unwrap().get(), 14);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Reference {
    kind: EntryKind,
    authors: AuthorList,
    editors: AuthorList,
    title: Option<String>,
    container: Option<String>,
    publisher: Option<String>,
    institution: Option<String>,
    year: Option<Year>,
    volume: Option<String>,
    issue: Option<String>,
    pages: Option<PageRange>,
    edition: Option<String>,
    note: Option<String>,
    identifiers: Identifiers,
    unparsed: Option<String>,
    provenance: Provenance,
}

impl Reference {
    /// Starts building a reference. The [`Provenance`] is mandatory and there
    /// is no other way in.
    pub fn builder(provenance: Provenance) -> ReferenceBuilder {
        ReferenceBuilder {
            reference: Self {
                kind: EntryKind::Unknown,
                authors: AuthorList::default(),
                editors: AuthorList::default(),
                title: None,
                container: None,
                publisher: None,
                institution: None,
                year: None,
                volume: None,
                issue: None,
                pages: None,
                edition: None,
                note: None,
                identifiers: Identifiers::default(),
                unparsed: None,
                provenance,
            },
        }
    }

    /// What kind of work this is.
    pub fn kind(&self) -> EntryKind {
        self.kind
    }

    /// The authors, and whether the source truncated them with `et al.`.
    pub fn authors(&self) -> &AuthorList {
        &self.authors
    }

    /// The editors, for an edited volume.
    pub fn editors(&self) -> &AuthorList {
        &self.editors
    }

    /// The title of the work itself.
    pub fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    /// The **container**: the journal, the proceedings, or the book a chapter
    /// sits in.
    ///
    /// One field rather than three (`journal`, `booktitle`, `series`) because
    /// they are the same relation — *this work appeared inside that work* — and
    /// a reference-list line very often does not say which of the three it is.
    /// The distinction is recovered at emission time from [`Self::kind`], which
    /// is where it actually matters.
    pub fn container(&self) -> Option<&str> {
        self.container.as_deref()
    }

    /// The publisher (books).
    pub fn publisher(&self) -> Option<&str> {
        self.publisher.as_deref()
    }

    /// The institution (reports, theses).
    pub fn institution(&self) -> Option<&str> {
        self.institution.as_deref()
    }

    /// The year of publication.
    pub fn year(&self) -> Option<Year> {
        self.year
    }

    /// The journal volume.
    pub fn volume(&self) -> Option<&str> {
        self.volume.as_deref()
    }

    /// The journal issue/number.
    pub fn issue(&self) -> Option<&str> {
        self.issue.as_deref()
    }

    /// The pages.
    pub fn pages(&self) -> Option<&PageRange> {
        self.pages.as_ref()
    }

    /// The edition (books).
    pub fn edition(&self) -> Option<&str> {
        self.edition.as_deref()
    }

    /// A free-text note.
    pub fn note(&self) -> Option<&str> {
        self.note.as_deref()
    }

    /// DOI, arXiv id, ISBN, ISSN, URL — each validated, none invented.
    pub fn identifiers(&self) -> &Identifiers {
        &self.identifiers
    }

    /// **The part of the source string this crate could not account for.**
    ///
    /// The single most useful field for a human auditing the extraction. If a
    /// reference came out looking complete and this is `Some`, something was
    /// dropped and the parse should be distrusted.
    pub fn unparsed(&self) -> Option<&str> {
        self.unparsed.as_deref()
    }

    /// Where this reference was read from, and the words it was read from.
    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }

    /// A deterministic, stable citation key: first author's family name, the
    /// year, and the first substantive word of the title.
    ///
    /// # Determinism
    ///
    /// The same reference always produces the same key, on every machine and
    /// every run. Nothing here consults a hash map's iteration order, the
    /// clock, or a random number generator — CLAUDE.md requires deterministic
    /// behaviour, and a citation key that changed between runs would rewrite a
    /// LaTeX document's `\cite` commands underneath its author.
    ///
    /// # Honesty
    ///
    /// The author component uses [`crate::Author::sort_key`], so a name whose
    /// split is only *assumed* keys on the name **as written** rather than on a
    /// family name we guessed at. That can produce `maozedong2019` rather than
    /// `mao2019` — slightly ugly, and *not wrong about who wrote the paper*,
    /// which is the trade this crate makes every time.
    ///
    /// Keys are not guaranteed unique on their own; see
    /// [`crate::Bibliography::keyed`], which disambiguates collisions
    /// deterministically with `a`, `b`, `c` suffixes exactly as a human would.
    pub fn suggested_key(&self) -> CitationKey {
        let author = self
            .authors
            .first()
            .map(|a| ascii_fold(&a.sort_key()))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "anon".to_string());

        let year = self
            .year
            .map(|y| y.get().to_string())
            .unwrap_or_else(|| "nd".to_string());

        let word = self
            .title
            .as_deref()
            .and_then(first_substantive_word)
            .map(|w| ascii_fold(&w.to_lowercase()))
            .unwrap_or_default();

        let key = format!("{author}{year}{word}");
        // `ascii_fold` already removed everything CitationKey rejects, so this
        // cannot fail -- but we do not `unwrap` on a public path.
        CitationKey::new(key).unwrap_or_else(|_| CitationKey("anon".to_string()))
    }
}

/// Builds a [`Reference`]. Obtained only from [`Reference::builder`], which
/// demands a [`Provenance`] — so there is no path to an un-sourced reference.
#[derive(Debug, Clone)]
pub struct ReferenceBuilder {
    reference: Reference,
}

macro_rules! setter {
    ($name:ident, $field:ident, $ty:ty, $doc:literal) => {
        #[doc = $doc]
        pub fn $name(mut self, value: $ty) -> Self {
            self.reference.$field = Some(value.into());
            self
        }
    };
}

impl ReferenceBuilder {
    /// Sets the kind of work.
    pub fn kind(mut self, kind: EntryKind) -> Self {
        self.reference.kind = kind;
        self
    }

    /// Sets the authors.
    pub fn authors(mut self, authors: AuthorList) -> Self {
        self.reference.authors = authors;
        self
    }

    /// Sets the editors.
    pub fn editors(mut self, editors: AuthorList) -> Self {
        self.reference.editors = editors;
        self
    }

    /// Sets the year.
    pub fn year(mut self, year: Year) -> Self {
        self.reference.year = Some(year);
        self
    }

    /// Sets the page range.
    pub fn pages(mut self, pages: PageRange) -> Self {
        self.reference.pages = Some(pages);
        self
    }

    /// Sets the identifiers.
    pub fn identifiers(mut self, identifiers: Identifiers) -> Self {
        self.reference.identifiers = identifiers;
        self
    }

    setter!(title, title, impl Into<String>, "Sets the title of the work.");
    setter!(container, container, impl Into<String>, "Sets the container (journal, proceedings, book).");
    setter!(publisher, publisher, impl Into<String>, "Sets the publisher.");
    setter!(institution, institution, impl Into<String>, "Sets the institution.");
    setter!(volume, volume, impl Into<String>, "Sets the volume.");
    setter!(issue, issue, impl Into<String>, "Sets the issue/number.");
    setter!(edition, edition, impl Into<String>, "Sets the edition.");
    setter!(note, note, impl Into<String>, "Sets a free-text note.");
    setter!(
        unparsed,
        unparsed,
        impl Into<String>,
        "Records the part of the source string that could not be accounted for. \
         **Set this whenever anything was left over** -- a silently-dropped \
         remainder is how a confidently-wrong reference gets made."
    );

    /// Finishes the reference.
    pub fn build(self) -> Reference {
        self.reference
    }
}

/// The first word of a title that is not an article or a preposition, for
/// keying. `"An open-source solver"` keys on `open`, not on `an`.
fn first_substantive_word(title: &str) -> Option<String> {
    const STOPWORDS: &[&str] = &[
        "a", "an", "the", "on", "of", "in", "for", "to", "and", "at", "by", "with", "from",
    ];
    title
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .find(|w| !STOPWORDS.contains(&w.to_lowercase().as_str()))
        .map(str::to_string)
}

/// Reduces a string to `[a-z0-9]`, dropping accents by keeping only ASCII
/// alphanumerics.
///
/// Deliberately lossy and deliberately crude: this is used **only** for
/// generating a citation key, which is an identifier for a LaTeX document, not
/// a representation of anybody's name. The name itself is never touched — see
/// [`crate::Author::as_written`]. Folding `Müller` to `mller` in a `\cite` key
/// is fine; folding it in a bibliography entry would not be, and this function
/// is never used for that.
fn ascii_fold(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::author::{parse_printed_name_list, Author};
    use crate::provenance::DocumentId;

    fn provenance() -> Provenance {
        let doc = DocumentId::new("paper.pdf").unwrap();
        Provenance::from_page(&doc, 14, "a reference line").unwrap()
    }

    // -- Year ----------------------------------------------------------

    #[test]
    fn a_year_is_range_checked_so_a_page_number_cannot_become_one() {
        assert_eq!(Year::parse("2024").unwrap().get(), 2024);
        assert_eq!(Year::parse("(2019)").unwrap().get(), 2019);
        // A real reference list can contain the article number 111144. If a
        // bare 6-digit number could be a year, it would become one.
        assert_eq!(Year::parse("111144"), None);
        assert_eq!(Year::parse("377"), None, "a volume number is not a year");
        assert_eq!(Year::parse("not a year"), None);
    }

    // -- Page and PageRange --------------------------------------------

    #[test]
    fn a_page_is_a_string_because_real_pages_are_not_numbers() {
        assert_eq!(Page::new("xii").unwrap().as_number(), None);
        assert_eq!(Page::new("e0123456").unwrap().as_number(), None);
        assert_eq!(Page::new("281").unwrap().as_number(), Some(281));
        assert_eq!(Page::new("  "), None);
    }

    #[test]
    fn a_page_range_counts_its_pages_when_it_can() {
        // A journal article's page range: pp. 281-301.
        let range = PageRange::new(Page::new("281").unwrap(), Page::new("301"));
        assert_eq!(range.page_count(), Some(21));
        assert_eq!(range.to_string(), "281--301");
    }

    #[test]
    fn a_single_page_reference_is_not_a_one_page_range() {
        // `p. 111144` -- an Elsevier-style article number. The source did not
        // say the paper is one page long.
        let range = PageRange::new(Page::new("111144").unwrap(), None);
        assert_eq!(range.page_count(), None);
        assert_eq!(range.to_string(), "111144");
    }

    #[test]
    fn a_roman_numeral_range_declines_to_count_rather_than_guessing() {
        let range = PageRange::new(Page::new("xii").unwrap(), Page::new("xv"));
        assert_eq!(range.page_count(), None);
        assert_eq!(range.to_string(), "xii--xv");
    }

    // -- CitationKey ----------------------------------------------------

    #[test]
    fn a_citation_key_rejects_what_would_break_a_latex_document() {
        assert!(CitationKey::new("chen2024mtat").is_ok());
        for bad in ["chen 2024", "chen,2024", "chen{2024}", "", "chen\\2024", "a#b"] {
            assert!(
                CitationKey::new(bad).is_err(),
                "{bad:?} would break \\cite and must be rejected"
            );
        }
    }

    #[test]
    fn a_citation_key_cannot_be_smuggled_in_through_json() {
        assert!(serde_json::from_str::<CitationKey>(r#""chen 2024""#).is_err());
    }

    // -- Reference -------------------------------------------------------

    #[test]
    fn a_reference_cannot_be_built_without_provenance() {
        // Enforced by the type system: `Reference::builder` takes a Provenance
        // by value and there is no other constructor. This test documents the
        // property; the compiler enforces it.
        let reference = Reference::builder(provenance()).build();
        assert_eq!(reference.provenance().document().as_str(), "paper.pdf");
        assert_eq!(reference.kind(), EntryKind::Unknown);
    }

    #[test]
    fn the_suggested_key_is_deterministic() {
        let build = || {
            Reference::builder(provenance())
                .authors(parse_printed_name_list("M. R. Chen, S. Novak"))
                .title("An open-source toolkit for multilingual text alignment")
                .year(Year::new(2024).unwrap())
                .build()
        };
        // Same input, same key -- a thousand times. A key that drifted would
        // rewrite a LaTeX document's \cite commands underneath its author.
        let first = build().suggested_key();
        for _ in 0..1000 {
            assert_eq!(build().suggested_key(), first);
        }
        // `An` is a stopword, so the title word is `open`.
        assert_eq!(first.as_str(), "chen2024open");
    }

    #[test]
    fn the_suggested_key_of_an_assumed_name_uses_the_name_as_written() {
        // "Mao Zedong" -- we do not know his family name, so we do not key on a
        // guess. `maozedong2019` is uglier than `mao2019` and it is not WRONG
        // ABOUT WHO WROTE THE PAPER, which is the trade every time.
        let reference = Reference::builder(provenance())
            .authors(crate::author::AuthorList::new(
                vec![Author::parse_bibtex_name("Mao Zedong")],
                false,
            ))
            .title("On Practice")
            .year(Year::new(1937).unwrap())
            .build();
        assert_eq!(reference.suggested_key().as_str(), "maozedong1937practice");
    }

    #[test]
    fn a_reference_with_nothing_known_still_gets_a_usable_key() {
        let reference = Reference::builder(provenance()).build();
        assert_eq!(reference.suggested_key().as_str(), "anonnd");
    }

    #[test]
    fn an_accented_name_is_folded_for_the_key_but_never_in_the_name_itself() {
        let reference = Reference::builder(provenance())
            .authors(crate::author::AuthorList::new(
                vec![Author::parse_bibtex_name("M\u{fc}ller, Hans")],
                false,
            ))
            .year(Year::new(2020).unwrap())
            .build();
        // The KEY is folded, because a \cite key is a LaTeX identifier...
        assert_eq!(reference.suggested_key().as_str(), "mller2020");
        // ...and the NAME is not, because it is a person's name.
        assert_eq!(
            reference.authors().first().unwrap().as_written(),
            "M\u{fc}ller, Hans"
        );
    }

    #[test]
    fn entry_kinds_map_onto_all_three_output_vocabularies() {
        assert_eq!(EntryKind::Software.bibtex_type(), "software");
        assert_eq!(EntryKind::TechReport.biblatex_type(), "report");
        assert_eq!(EntryKind::TechReport.bibtex_type(), "techreport");
        assert_eq!(EntryKind::Preprint.biblatex_type(), "online");
        assert_eq!(EntryKind::Thesis.hayagriva_type(), "thesis");
        // Unknown is not a BibTeX type; it must degrade to something bibtex(1)
        // will actually accept, or we emit a .bib file that does not compile.
        assert_eq!(EntryKind::Unknown.bibtex_type(), "misc");
    }

    #[test]
    fn a_reference_round_trips_through_json_with_its_provenance() {
        let reference = Reference::builder(provenance())
            .kind(EntryKind::Article)
            .title("MTAT")
            .year(Year::new(2024).unwrap())
            .build();
        let json = serde_json::to_string(&reference).unwrap();
        let back: Reference = serde_json::from_str(&json).unwrap();
        assert_eq!(reference, back);
    }
}

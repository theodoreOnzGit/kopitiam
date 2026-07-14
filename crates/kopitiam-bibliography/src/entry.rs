//! Parsing a printed reference-list line into a [`Reference`].
//!
//! This is the module most likely to be **confidently wrong**, so it is the one
//! built hardest against that.
//!
//! # The confidence model
//!
//! Every line comes out as one of three things, and the middle one is the
//! important one:
//!
//! * [`ParsedReference::Parsed`] — every part of the line was accounted for.
//! * [`ParsedReference::Partial`] — a [`Reference`] **plus the text we could not
//!   account for**. This is a good outcome, not a failure: a reference with an
//!   author, a title and a year, and an honest note saying *"I did not
//!   understand `Advanced . . ., Tech. Rep.`"*, is useful and cannot mislead
//!   anyone.
//! * [`ParsedReference::Unparsed`] — the raw string, kept. No `Reference` at all.
//!
//! What deliberately does **not** exist is a fourth outcome where the leftovers
//! are quietly dropped and a clean-looking `Reference` is returned. A citation
//! that *looks* complete and points at the wrong work is worse in every way than
//! one that admits it is unsure.
//!
//! # Consume-and-account
//!
//! The parser **consumes** spans of the line, recording exactly which bytes it
//! accounted for ([`Consumed`]). Whatever is left at the end — once punctuation
//! is discounted — becomes the `unparsed` remainder.
//!
//! There is no way for the parser to *use* a piece of the line without marking
//! it, because the marking **is** the parse. That is what makes "partial"
//! trustworthy rather than aspirational: a remainder cannot be forgotten, only
//! reported.
//!
//! # Styles handled
//!
//! Developed against a real typeset reference list in `biblatex`'s `ieee`
//! style — the dominant style across engineering and computer science:
//!
//! ```text
//!   M. R. Chen, S. Novak, and J. P. Alvarez, "An open-source toolkit ... (mtat),"
//!   International Journal of ..., vol. 6, no. 4, pp. 281-301, 2024.
//!
//!   M. R. Chen, Statistical Models as Testbeds for ... Development.
//!   University of California, Berkeley, 2024.
//! ```

use std::sync::LazyLock;

use regex::{Match, Regex};
use serde::{Deserialize, Serialize};

use crate::anomaly::Anomaly;
use crate::author::{
    is_known_particle, looks_academic, looks_institutional, parse_printed_name_list,
};
use crate::identifier::{ArxivId, Doi, Identifiers, Isbn, ResourceUrl};
use crate::provenance::Provenance;
use crate::reference::{EntryKind, Page, PageRange, Reference, Year};
use crate::text::{fold_typography, squeeze};

/// The outcome of parsing one reference-list line.
///
/// Every variant carries the [`Provenance`], so there is no way to hold a
/// reference — or a *failed* reference — without knowing where it came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParsedReference {
    /// The whole line was accounted for.
    Parsed(Reference),

    /// A reference was recovered, **and some of the line was not understood**.
    ///
    /// The reference's [`unparsed`](Reference::unparsed) field carries the same
    /// remainder, so it travels into the `.bib` file's `note` and into the
    /// knowledge graph rather than being lost in transit.
    Partial(Reference),

    /// Nothing could be recovered. The raw string is kept, verbatim, with its
    /// provenance. **This is a correct answer**, and infinitely preferable to a
    /// fabricated one.
    Unparsed(RawEntry),
}

impl ParsedReference {
    /// The recovered reference, if any.
    pub fn reference(&self) -> Option<&Reference> {
        match self {
            Self::Parsed(reference) | Self::Partial(reference) => Some(reference),
            Self::Unparsed(_) => None,
        }
    }

    /// Where this line came from — available for every variant, failures included.
    pub fn provenance(&self) -> &Provenance {
        match self {
            Self::Parsed(reference) | Self::Partial(reference) => reference.provenance(),
            Self::Unparsed(raw) => raw.provenance(),
        }
    }

    /// Whether anything at all was recovered.
    pub fn is_parsed(&self) -> bool {
        !matches!(self, Self::Unparsed(_))
    }

    /// Whether part of the source line was not understood.
    pub fn is_partial(&self) -> bool {
        matches!(self, Self::Partial(_))
    }
}

/// A reference-list line that could not be parsed, kept verbatim.
///
/// Not an error type. A bibliography that reports *"line 7 said this, and I
/// could not make sense of it"* has done its job. One that silently omitted line
/// 7, or invented a plausible reference for it, has not.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RawEntry {
    provenance: Provenance,
}

impl RawEntry {
    /// Records an unparseable line.
    pub fn new(provenance: Provenance) -> Self {
        Self { provenance }
    }

    /// The line, exactly as printed.
    pub fn text(&self) -> &str {
        self.provenance.verbatim().as_str()
    }

    /// Where it was printed.
    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }
}

// -- Patterns ----------------------------------------------------------------

/// `vol. 6`, `Vol 6A`.
static VOLUME: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bvol\.?\s*([0-9A-Za-z]+)").unwrap());

/// `no. 4`, `nos. 4-5`, `issue 4`.
static ISSUE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\b(?:no|nos|issue)\.?\s*([0-9]+[0-9A-Za-z\-]*)").unwrap());

/// A page range: `pp. 281-301`.
static PAGE_RANGE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\bpp?\.?\s*([0-9ivxlc]+)\s*(?:-{1,3})\s*([0-9ivxlc]+)").unwrap()
});

/// A single page: `p. 111144`, and — the interesting one — `p. 111 144`.
///
/// # The digit-group separator
///
/// The `(?:[ ][0-9]{3})*` is not a mistake. `biblatex` prints large numbers with
/// a **group separator**, so a journal article number `111144`
/// is typeset as `p. 111 144`. Matching only `\d+` captures `111` and leaves
/// `144` dangling as an unexplained remainder.
///
/// This is the same class of bug the plot engine hit on the *same paper*, where
/// comma-grouped axis labels (`11,000`) collapsed the y-axis to a single usable
/// tick. A real typesetter groups digits; a synthetic fixture does not. It is
/// the whole argument for testing against real documents — and the regrouping is
/// still declared as an [`Anomaly::AssumedDigitGrouping`], because it is an
/// assumption.
static PAGE_SINGLE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bp\.?\s*([0-9]+(?:[ ][0-9]{3})*)\b").unwrap());

/// A quoted title (after typographic folding turns `\u{201c}\u{201d}` into `"`).
static QUOTED_TITLE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#""([^"]+)""#).unwrap());

/// A bare URL.
static URL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"https?://[^\s,]+|www\.[^\s,]+").unwrap());

/// A DOI.
static DOI_IN_TEXT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:doi:\s*|https?://(?:dx\.)?doi\.org/)?(10\.\d{4,9}/[^\s,]+)").unwrap()
});

/// An arXiv identifier.
static ARXIV_IN_TEXT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)arxiv:\s*(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)")
        .unwrap()
});

/// An ISBN.
static ISBN_IN_TEXT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)ISBN(?:-1[03])?:?\s*([0-9][0-9\- ]{8,17}[0-9Xx])").unwrap()
});

/// A four-digit year.
static YEAR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b(1[0-9]{3}|2[0-9]{3})\b").unwrap());

/// Explicit technical-report markers.
static TECH_REPORT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\btech(?:nical)?\.?\s*rep(?:ort)?\.?").unwrap());

/// Explicit thesis markers. **Explicit only** — see [`parse_reference_line`] for
/// why a university publisher is not one.
static THESIS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\b(?:thesis|dissertation)\b").unwrap());

/// Parses one printed reference-list line.
///
/// `provenance` must already carry the line's verbatim and normalised text; the
/// parse runs on [`Provenance::normalised`].
///
/// Never fails and never fabricates: an unrecognisable line comes back as
/// [`ParsedReference::Unparsed`].
///
/// Assumptions and shortfalls are appended to `anomalies`, so a caller
/// extracting a whole bibliography collects them all in one place.
///
/// ```
/// use kopitiam_bibliography::{DocumentId, EntryKind, Provenance, entry::parse_reference_line};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let doc = DocumentId::new("aligned_corpus.pdf")?;
/// // A three-author journal article in IEEE style, as it would be typeset.
/// let line = "M. R. Chen, S. Novak, and J. P. Alvarez, \u{201c}An open-source \
///     toolkit for scalable multilingual sentence and phrase alignment (mtat),\u{201d} \
///     International Journal of Computational Linguistics and Text Processing, \
///     vol. 6, no. 4, pp. 281\u{2013}301, 2024.";
/// let provenance = Provenance::from_page(&doc, 15, line)?;
///
/// let mut anomalies = Vec::new();
/// let parsed = parse_reference_line(provenance, &mut anomalies);
///
/// let reference = parsed.reference().expect("this one parses");
/// assert_eq!(reference.kind(), EntryKind::Article);
/// assert_eq!(reference.authors().len(), 3);
/// assert_eq!(reference.authors().first().unwrap().family(), Some("Chen"));
/// assert_eq!(reference.year().unwrap().get(), 2024);
/// assert_eq!(reference.volume(), Some("6"));
/// assert_eq!(reference.issue(), Some("4"));
/// assert_eq!(reference.pages().unwrap().page_count(), Some(21));
/// # Ok(())
/// # }
/// ```
pub fn parse_reference_line(
    provenance: Provenance,
    anomalies: &mut Vec<Anomaly>,
) -> ParsedReference {
    let source = fold_typography(provenance.normalised().as_str());
    let mut span = Consumed::new(&source);

    // -- 1. Identifiers. The least ambiguous things on the line; consuming them
    //       first stops a DOI's digits being mistaken for a year or a page.
    let identifiers = take_identifiers(&source, &mut span, &provenance, anomalies);

    // -- 2. Pages, before the year: both are numbers, and each would otherwise
    //       be a candidate for the other.
    let pages = take_pages(&source, &mut span, &provenance, anomalies);

    // -- 3. Volume and issue.
    let volume = take_capture(&VOLUME, &source, &mut span);
    let issue = take_capture(&ISSUE, &source, &mut span);

    // -- 4. The year: the LAST unconsumed four-digit number. A reference-list
    //       entry ends with its year; an earlier one is far more likely to be
    //       part of a conference name ("SOUPS 2021") or a report number.
    let mut year = None;
    if let Some(m) = YEAR
        .find_iter(&source)
        .filter(|m| !span.any_consumed(m.start(), m.end()))
        .last()
    {
        year = Year::parse(m.as_str());
        if year.is_some() {
            span.consume(&m);
        }
    }

    // -- 5. Explicit kind markers, consumed so they do not show up as
    //       "unparsed".
    let is_report = TECH_REPORT.is_match(&source);
    if let Some(m) = TECH_REPORT.find(&source) {
        span.consume(&m);
    }
    let is_thesis = THESIS.is_match(&source);
    if let Some(m) = THESIS.find(&source) {
        span.consume(&m);
    }

    // -- 6. The title, which determines the shape of everything else.
    let mut kind = EntryKind::Unknown;
    let mut title = None;
    let mut container = None;
    let mut publisher = None;
    let mut institution = None;
    let authors;

    if let Some(caps) = QUOTED_TITLE.captures(&source) {
        // ---- The quoted-title shape: `Authors, "Title," Venue, vol..., year.`
        let whole = caps.get(0).expect("group 0 always exists");
        title = Some(clean_field(&caps[1]));
        span.consume(&whole);
        kind = EntryKind::Article;

        // Authors are everything before the title.
        authors = parse_printed_name_list(&source[..whole.start()]);
        span.consume_range(0, whole.start());

        // The venue is the first unconsumed comma-chunk after the title.
        if let Some(chunk) = span.first_unconsumed_chunk(&source, whole.end()) {
            let text = chunk.text.trim();
            if let Some(rest) = strip_leading_in(text) {
                kind = EntryKind::InProceedings;
                container = Some(clean_field(rest));
            } else {
                container = Some(clean_field(text));
            }
            span.consume_range(chunk.start, chunk.end);
        }
    } else {
        // ---- The unquoted shape: `Authors, Title. Publisher, Year.`
        let author_end = author_region_end(&source);
        authors = parse_printed_name_list(&source[..author_end]);
        span.consume_range(0, author_end);

        // The body runs from the end of the authors to the first thing already
        // consumed (the year, the URL, the pages) -- which in this style is
        // exactly where the title and publisher end.
        let body_end = span.next_consumed_from(author_end).unwrap_or(source.len());
        let body = &source[author_end..body_end];

        let (title_span, rest_span) = split_title_from_publisher(body);
        if !title_span.is_empty() {
            title = Some(clean_field(title_span));
            let start = author_end + offset_of(body, title_span);
            span.consume_range(start, start + title_span.len());
        }
        if !rest_span.is_empty() {
            let cleaned = clean_field(rest_span);
            if !cleaned.is_empty() {
                if looks_institutional(&cleaned) {
                    institution = Some(cleaned);
                } else {
                    publisher = Some(cleaned);
                }
                let start = author_end + offset_of(body, rest_span);
                span.consume_range(start, start + rest_span.len());
            }
        }
    }

    // -- 7. Resolve the kind from explicit markers only.
    if is_report {
        kind = EntryKind::TechReport;
        // A report's "publisher" is its issuing institution.
        if institution.is_none() {
            institution = publisher.take().or_else(|| container.take());
        }
    } else if is_thesis {
        kind = EntryKind::Thesis;
        if institution.is_none() {
            institution = publisher.take().or_else(|| container.take());
        }
    } else if identifiers.url.is_some() && identifiers.doi.is_none() && container.is_none() {
        // A bare URL, no DOI, no journal: software, a repository, a web page.
        // Reference 2 of the maintainer's own paper is a Git repository, and
        // calling it `@misc` would be a small lie about what it is.
        kind = EntryKind::Software;
    } else if kind == EntryKind::Unknown && (publisher.is_some() || institution.is_some()) {
        // Title, publisher, year. That IS the shape of a book, and saying so is
        // reading the source rather than guessing at it.
        kind = EntryKind::Book;

        // ...but a "publisher" that is a university, with no explicit thesis
        // marker, is THE most common ambiguity in this corpus and is NOT
        // resolvable from the string. Five of the twelve references in the
        // maintainer's own paper are theses printed exactly like books, because
        // the `biblatex` ieee style drops the "PhD thesis" designator entirely.
        //
        // We do not guess. We report.
        let venue = institution.as_deref().or(publisher.as_deref()).unwrap_or("");
        if looks_academic(venue) {
            anomalies.push(Anomaly::AmbiguousEntryKind {
                provenance: provenance.clone(),
                chosen: EntryKind::Book,
                alternative: EntryKind::Thesis,
                reason: format!(
                    "the publisher {venue:?} is an academic institution, and the source \
                     carries no explicit thesis or dissertation marker; a thesis printed \
                     in this style is indistinguishable from a book"
                ),
            });
        }
    }

    // -- 8. Assemble, and be honest about the leftovers.
    //
    // A TITLE ALONE IS NOT A CITATION.
    //
    // The title extractor is, by necessity, the greediest thing in this module:
    // in the unquoted shape it takes whatever text is left over. Left
    // unguarded, that means *any* line at all -- a page header, a figure
    // caption, a stray sentence, a keyboard mash -- comes back as "a reference
    // with a title", which is a fabricated reference with extra steps.
    //
    // A citation is a claim about **who** wrote something, **when**, or **where
    // to find it**. A bare noun phrase is none of those. So a `Reference` is
    // only recoverable if the line yielded at least one of: an author, a year,
    // or an identifier. Everything else is `Unparsed`, with the raw text kept.
    //
    // This is why `Mtat alignment toolkit` (a repository reference -- three
    // authors, a URL and a year alongside it) parses, while a line containing
    // nothing but words does not.
    let has_corroboration = !authors.is_empty() || year.is_some() || identifiers.any();
    if !has_corroboration {
        anomalies.push(Anomaly::UnparseableEntry {
            provenance: provenance.clone(),
        });
        return ParsedReference::Unparsed(RawEntry::new(provenance));
    }

    let remainder = span.remainder(&source);

    let mut builder = Reference::builder(provenance.clone())
        .kind(kind)
        .authors(authors)
        .identifiers(identifiers);

    if let Some(title) = title {
        builder = builder.title(title);
    }
    if let Some(container) = container {
        builder = builder.container(container);
    }
    if let Some(publisher) = publisher {
        builder = builder.publisher(publisher);
    }
    if let Some(institution) = institution {
        builder = builder.institution(institution);
    }
    if let Some(year) = year {
        builder = builder.year(year);
    }
    if let Some(volume) = volume {
        builder = builder.volume(volume);
    }
    if let Some(issue) = issue {
        builder = builder.issue(issue);
    }
    if let Some(pages) = pages {
        builder = builder.pages(pages);
    }

    match remainder {
        Some(remainder) => {
            anomalies.push(Anomaly::PartialEntry {
                provenance,
                remainder: remainder.clone(),
            });
            ParsedReference::Partial(builder.unparsed(remainder).build())
        }
        None => ParsedReference::Parsed(builder.build()),
    }
}

// -- Field extraction --------------------------------------------------------

fn take_identifiers(
    source: &str,
    span: &mut Consumed,
    provenance: &Provenance,
    anomalies: &mut Vec<Anomaly>,
) -> Identifiers {
    let mut identifiers = Identifiers::default();

    if let Some(caps) = DOI_IN_TEXT.captures(source)
        && let Ok(doi) = Doi::parse(&caps[1])
    {
        identifiers.doi = Some(doi);
        span.consume(&caps.get(0).expect("group 0"));
    }
    if let Some(caps) = ARXIV_IN_TEXT.captures(source)
        && let Ok(arxiv) = ArxivId::parse(&caps[1])
    {
        identifiers.arxiv = Some(arxiv);
        span.consume(&caps.get(0).expect("group 0"));
    }
    if let Some(caps) = ISBN_IN_TEXT.captures(source) {
        match Isbn::parse(&caps[1]) {
            Ok(isbn) => {
                identifiers.isbn = Some(isbn);
                span.consume(&caps.get(0).expect("group 0"));
            }
            // An ISBN with a bad checksum is a FINDING, not something to paper
            // over. It is almost always an OCR error or a typo, and reporting it
            // is how it gets fixed. Accepting it would put a wrong identifier in
            // a bibliography.
            Err(error) => anomalies.push(Anomaly::InvalidIdentifier {
                provenance: provenance.clone(),
                reason: error.to_string(),
            }),
        }
    }
    if let Some(m) = URL.find(source)
        && let Ok(url) = ResourceUrl::parse(m.as_str())
    {
        identifiers.url = Some(url);
        span.consume(&m);
    }

    identifiers
}

fn take_pages(
    source: &str,
    span: &mut Consumed,
    provenance: &Provenance,
    anomalies: &mut Vec<Anomaly>,
) -> Option<PageRange> {
    if let Some(caps) = PAGE_RANGE.captures(source) {
        let start = Page::new(caps[1].trim())?;
        let end = Page::new(caps[2].trim());
        span.consume(&caps.get(0).expect("group 0"));
        return Some(PageRange::new(start, end));
    }
    if let Some(caps) = PAGE_SINGLE.captures(source) {
        let printed = caps[1].trim();
        let read_as = regroup_digits(printed, provenance, anomalies);
        let start = Page::new(read_as)?;
        span.consume(&caps.get(0).expect("group 0"));
        return Some(PageRange::new(start, None));
    }
    None
}

fn take_capture(pattern: &Regex, source: &str, span: &mut Consumed) -> Option<String> {
    let caps = pattern.captures(source)?;
    let whole = caps.get(0).expect("group 0");
    if span.any_consumed(whole.start(), whole.end()) {
        return None;
    }
    let value = caps[1].to_string();
    span.consume(&whole);
    Some(value)
}

/// Joins the groups of a digit-group-separated number, and **records that it
/// did**.
///
/// `p. 111 144` is `biblatex` printing article number `111144` with a group
/// separator. Reading it as page 111 and dropping `144` is wrong; joining is
/// almost certainly right; and *almost certainly* is precisely the sort of thing
/// that must be declared rather than assumed in silence.
fn regroup_digits(text: &str, provenance: &Provenance, anomalies: &mut Vec<Anomaly>) -> String {
    if !text.contains(' ') {
        return text.to_string();
    }
    let joined: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    if joined.chars().all(|c| c.is_ascii_digit()) {
        anomalies.push(Anomaly::AssumedDigitGrouping {
            provenance: provenance.clone(),
            printed: text.to_string(),
            read_as: joined.clone(),
        });
        joined
    } else {
        text.to_string()
    }
}

// -- The consume-and-account bookkeeping -------------------------------------

/// Tracks which bytes of the line the parser has accounted for.
///
/// The mechanism that makes "partial" honest: a piece of the line cannot be
/// *used* without being marked, so anything left at the end is genuinely
/// something we did not understand — never something we forgot to mention.
struct Consumed {
    consumed: Vec<bool>,
}

/// A comma-delimited chunk with its byte offsets, so it can be consumed.
struct Chunk<'a> {
    text: &'a str,
    start: usize,
    end: usize,
}

impl Consumed {
    fn new(source: &str) -> Self {
        Self {
            consumed: vec![false; source.len()],
        }
    }

    fn consume_range(&mut self, start: usize, end: usize) {
        let end = end.min(self.consumed.len());
        if start >= end {
            return;
        }
        for flag in &mut self.consumed[start..end] {
            *flag = true;
        }
    }

    fn consume(&mut self, m: &Match<'_>) {
        self.consume_range(m.start(), m.end());
    }

    fn is_consumed(&self, at: usize) -> bool {
        self.consumed.get(at).copied().unwrap_or(false)
    }

    fn any_consumed(&self, start: usize, end: usize) -> bool {
        (start..end.min(self.consumed.len())).any(|i| self.consumed[i])
    }

    /// The offset of the next already-consumed byte at or after `from`.
    fn next_consumed_from(&self, from: usize) -> Option<usize> {
        (from..self.consumed.len()).find(|&i| self.consumed[i])
    }

    /// The first comma-delimited chunk at or after `from` that still has
    /// unconsumed content in it.
    fn first_unconsumed_chunk<'a>(&self, source: &'a str, from: usize) -> Option<Chunk<'a>> {
        let mut start = from;
        while start < source.len() {
            let rest = &source[start..];
            let len = rest.find(',').unwrap_or(rest.len());
            let end = start + len;

            let text = &source[start..end];
            let has_content = text.chars().any(|c| c.is_alphanumeric());
            let has_unconsumed = (start..end).any(|i| !self.is_consumed(i));

            if has_content && has_unconsumed {
                return Some(Chunk { text, start, end });
            }
            start = end + 1;
        }
        None
    }

    /// The text of the line that was never accounted for, once punctuation and
    /// connectives are discounted.
    ///
    /// `None` means the whole line was understood.
    fn remainder(&self, source: &str) -> Option<String> {
        let mut leftovers: Vec<String> = Vec::new();
        let mut current = String::new();

        for (index, ch) in source.char_indices() {
            if self.is_consumed(index) {
                if !current.trim().is_empty() {
                    leftovers.push(current.trim().to_string());
                }
                current.clear();
            } else {
                current.push(ch);
            }
        }
        if !current.trim().is_empty() {
            leftovers.push(current.trim().to_string());
        }

        // Punctuation, the connectives left behind by an author list, and
        // fragments too short to be information are not "text we failed to
        // understand". Discount them, or every reference would be `Partial` and
        // the distinction would stop meaning anything.
        let meaningful: Vec<String> = leftovers
            .into_iter()
            .map(|chunk| {
                chunk
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .to_string()
            })
            .filter(|chunk| {
                chunk.chars().filter(char::is_ascii_alphanumeric).count() > 3
                    && !matches!(
                        chunk.to_ascii_lowercase().as_str(),
                        "and" | "in" | "et al" | "eds" | "ed" | "vol" | "no" | "pp"
                    )
            })
            .collect();

        (!meaningful.is_empty()).then(|| meaningful.join(" | "))
    }
}

// -- Helpers ------------------------------------------------------------------

/// Where the author list ends.
///
/// # The bug this function was rewritten to fix
///
/// A real reference list can contain a book entry like:
///
/// ```text
///     J. E. Barker, N. Whitfield, and P. F. Alvarez, Design, implementation and
///     deployment of a large-scale multilingual text retrieval platform...
/// ```
///
/// The title **begins with a comma-terminated word**: *"Design, implementation
/// and deployment..."*. The first version of this function saw `Design` as a
/// short capitalised chunk, decided it looked like a surname, and produced **a
/// fourth author, called "Design", who does not exist.**
///
/// That is precisely the failure this crate is written to prevent — a fabricated
/// author, sitting in a `.bib` file, looking exactly as trustworthy as the three
/// real ones. Every synthetic test passed. Only a real reference list caught it.
///
/// # The three rules that replaced it
///
/// 1. **`and` terminates the list.** In every printed reference style, `and` (or
///    `&`) introduces the *final* author. So the chunk beginning with `and` is
///    the last author chunk, and **nothing after it is an author**. This alone
///    fixes reference 9.
///
/// 2. **A chunk with no initial needs corroboration.** A bare capitalised word
///    (`Smith`, `Design`) is only an author if either
///    * the **next** chunk is initials-only — the `Smith, J., Jones, A.` style,
///      where surname and initials alternate; or
///    * **no author has been accepted yet** and it reads as a plain
///      `Firstname Lastname` leading the list.
///
///    `Design` is preceded by three real authors and followed by *"fabrication
///    and startup testing..."*, so it fails both, and is correctly left in the
///    title.
///
/// 3. **A lower-case token disqualifies a chunk** unless it is a particle
///    (`van`, `de`). Titles are full of `as`, `of`, `for`; names are not.
///
/// # The trade this makes
///
/// A plain `John Smith, A Book. Press, 2020.` (no initials, no `and`, no
/// following initials chunk) still yields its author — rule 2's second arm. But
/// an exotic list this cannot read will lose an author rather than invent one.
/// **That is the correct trade every time**: a missing author is visible in the
/// output and reported as a remainder; a fabricated one is invisible, and is a
/// lie about who did the work.
fn author_region_end(source: &str) -> usize {
    let chunks = comma_chunks(source);
    let mut end = 0usize;
    let mut accepted = 0usize;

    for (index, chunk) in chunks.iter().enumerate() {
        let text = chunk.text.trim();
        let is_final = text.starts_with("and ")
            || text.starts_with("& ")
            || text.starts_with("And ");
        let stripped = text
            .trim_start_matches("and ")
            .trim_start_matches("And ")
            .trim_start_matches("& ")
            .trim();

        if stripped.is_empty() {
            end = chunk.end;
            continue;
        }

        let next_is_initials = chunks
            .get(index + 1)
            .is_some_and(|next| is_initials_only(next.text.trim()));

        if !is_an_author_chunk(stripped, accepted == 0, next_is_initials) {
            break;
        }

        accepted += 1;
        end = chunk.end;

        // Rule 1: `and` introduces the LAST author. Nothing after it is one.
        if is_final {
            break;
        }
    }

    end.min(source.len())
}

/// Whether a comma-chunk reads as a person's name rather than as a title.
///
/// See [`author_region_end`] for the rules and for the real-world bug that
/// produced them.
fn is_an_author_chunk(chunk: &str, is_first: bool, next_is_initials: bool) -> bool {
    let tokens: Vec<&str> = chunk.split_whitespace().collect();
    if tokens.is_empty() || tokens.len() > 6 {
        return false;
    }

    let mut has_initial = false;
    for token in &tokens {
        let cleaned = token.trim_matches(|c: char| !c.is_alphanumeric() && c != '.' && c != '-');
        if cleaned.is_empty() {
            continue;
        }
        if is_an_initial(cleaned) {
            has_initial = true;
            continue;
        }
        let Some(first) = cleaned.chars().next() else {
            continue;
        };
        // Rule 3: a lower-case token means this is prose, not a name -- unless it
        // is a particle, which is the one lower-case thing a name may contain.
        if first.is_lowercase() {
            if !is_known_particle(cleaned) {
                return false;
            }
            continue;
        }
        if !first.is_uppercase() {
            return false;
        }
    }

    if has_initial {
        // "J. E. Barker", "M. R. Chen", "B.-C. Du". An initial is not a word a
        // title contains, so this is unambiguous.
        return true;
    }

    // Rule 2: no initial anywhere. This chunk needs corroboration, or `Design`
    // becomes an author.
    //
    //   * `Smith` followed by `J.`  -> the Family, Initials style. An author.
    //   * `John Smith` leading the list, nothing accepted yet. An author.
    //   * `Design` after three authors, followed by prose. NOT an author.
    next_is_initials || (is_first && tokens.len() == 2)
}

/// Whether a chunk is nothing but initials (`J.`, `T. K. C.`, `B.-C.`).
fn is_initials_only(chunk: &str) -> bool {
    let tokens: Vec<&str> = chunk.split_whitespace().collect();
    !tokens.is_empty()
        && tokens.iter().all(|token| {
            let cleaned =
                token.trim_matches(|c: char| !c.is_alphanumeric() && c != '.' && c != '-');
            !cleaned.is_empty() && is_an_initial(cleaned)
        })
}

/// Whether a token is an initial **as printed in a reference list** — a single
/// letter *with its period*: `J.`, `B.-C.`, `T.`
///
/// # The period is mandatory here, and it is load-bearing
///
/// [`crate::author::GivenName`] accepts a bare letter as an initial, because in a
/// BibTeX `author` field `J R R Tolkien` is unambiguous. In a **reference-list
/// line** it is not, for one reason:
///
/// > `A` is the indefinite article.
///
/// Without the period requirement, `A Book About Syntax. Press` reads as a chunk
/// containing an initial (`A`) followed by capitalised words — i.e. as a
/// person's name — and the title of the book becomes its author. That is the
/// same class of fabrication as the `Design` bug in [`author_region_end`], and it
/// was found by the regression test written for that one.
///
/// The cost is that a reference-list author printed without periods
/// (`J R R Tolkien`) is not recognised, and the reference loses an author. That
/// is a **visible, reported** loss, and it is the correct trade against inventing
/// one.
fn is_an_initial(token: &str) -> bool {
    if !token.ends_with('.') {
        return false;
    }
    token
        .split('-')
        .all(|part| part.trim_end_matches('.').chars().count() == 1)
}

/// The comma-delimited chunks of a string, with their byte offsets.
fn comma_chunks(source: &str) -> Vec<Chunk<'_>> {
    let mut chunks = Vec::new();
    let mut start = 0usize;

    while start <= source.len() {
        let rest = &source[start..];
        let len = rest.find(',').unwrap_or(rest.len());
        let text_end = start + len;
        chunks.push(Chunk {
            text: &source[start..text_end],
            start,
            end: (text_end + 1).min(source.len()),
        });
        if text_end >= source.len() {
            break;
        }
        start = text_end + 1;
    }

    chunks
}

/// `in Seventeenth Symposium ...` -> `Seventeenth Symposium ...`
fn strip_leading_in(text: &str) -> Option<&str> {
    let trimmed = text.trim_start();
    let rest = trimmed.strip_prefix("in ").or_else(|| trimmed.strip_prefix("In "))?;
    Some(rest.trim())
}

/// Splits a book-style body into `Title` and `Publisher`.
///
/// The style prints `Title. Publisher, Year.`, so the break is a period followed
/// by a space and a capital. Abbreviations (`Dr.`, `U.S.`, an initial) would
/// false-positive, so only a period preceded by a **word of two or more letters**
/// counts as a sentence break.
///
/// Returns subslices of `body`, so the caller can compute their offsets and
/// consume them.
fn split_title_from_publisher(body: &str) -> (&str, &str) {
    for (index, _) in body.match_indices(". ") {
        let preceding_word_len = body[..index]
            .chars()
            .rev()
            .take_while(|c| c.is_alphanumeric())
            .count();
        if preceding_word_len < 2 {
            continue;
        }
        let after = index + 2;
        if body[after..].chars().next().is_some_and(char::is_uppercase) {
            return (body[..index].trim(), body[after..].trim());
        }
    }
    (body.trim(), "")
}

/// The byte offset of `needle` within `haystack`, where `needle` is known to be
/// a subslice of it.
fn offset_of(haystack: &str, needle: &str) -> usize {
    let base = haystack.as_ptr() as usize;
    let inner = needle.as_ptr() as usize;
    inner.saturating_sub(base)
}

/// Trims the punctuation a reference-list line leaves on a field's ends.
fn clean_field(text: &str) -> String {
    squeeze(text)
        .trim_matches(|c: char| matches!(c, ',' | '.' | ';' | '"') || c.is_whitespace())
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Author;
    use crate::provenance::DocumentId;

    /// Parses a line as if it had been read off page 15 of the paper.
    fn parse(line: &str) -> (ParsedReference, Vec<Anomaly>) {
        let doc = DocumentId::new("aligned_corpus.pdf").unwrap();
        let provenance = Provenance::from_page(&doc, 15, line).unwrap();
        let mut anomalies = Vec::new();
        let parsed = parse_reference_line(provenance, &mut anomalies);
        (parsed, anomalies)
    }

    // -- A realistic typeset reference list -------------------------------
    //
    // Every string below reproduces a shape that occurs in real typeset
    // reference lists: hyphenation across line breaks, URLs split at `/`,
    // article numbers printed with digit-group separators, a thesis printed
    // exactly like a book. These are the shapes a synthetic fixture written
    // from scratch would not think to include.

    #[test]
    fn real_reference_1_a_journal_article() {
        let (parsed, _) = parse(
            "M. R. Chen, S. Novak, and J. P. Alvarez, \u{201c}An open-source toolkit for \
             scalable multilingual sentence and phrase alignment (mtat),\u{201d} International \
             Journal of Computational Linguistics and Text Processing, vol. 6, no. 4, \
             pp. 281\u{2013}301, 2024.",
        );
        let reference = parsed.reference().expect("must parse");

        assert_eq!(reference.kind(), EntryKind::Article);
        assert_eq!(reference.authors().len(), 3);
        assert_eq!(reference.authors().first().unwrap().family(), Some("Chen"));
        assert!(
            reference.title().unwrap().starts_with("An open-source toolkit for scalable"),
            "got: {:?}",
            reference.title()
        );
        assert_eq!(
            reference.container(),
            Some("International Journal of Computational Linguistics and Text Processing")
        );
        assert_eq!(reference.volume(), Some("6"));
        assert_eq!(reference.issue(), Some("4"));
        assert_eq!(reference.pages().unwrap().start().as_str(), "281");
        assert_eq!(reference.pages().unwrap().end().unwrap().as_str(), "301");
        assert_eq!(reference.year().unwrap().get(), 2024);
        assert!(!parsed.is_partial(), "the whole line should be accounted for");
    }

    #[test]
    fn real_reference_2_a_git_repository() {
        // The URL was broken across a line by LaTeX's `url` package. The
        // normalisation in `crate::text` is what makes this recoverable at all.
        let (parsed, _) = parse(
            "M. R. Chen, S. Novak, and J. P. Alvarez, Mtat alignment toolkit, \
             https://github.com/openalign/mtat_toolkit, 2024.",
        );
        let reference = parsed.reference().expect("must parse");

        assert_eq!(reference.kind(), EntryKind::Software, "a repository is software");
        assert_eq!(reference.authors().len(), 3);
        assert_eq!(reference.title(), Some("Mtat alignment toolkit"));
        assert_eq!(
            reference.identifiers().url.as_ref().unwrap().as_str(),
            "https://github.com/openalign/mtat_toolkit"
        );
        assert_eq!(reference.year().unwrap().get(), 2024);
    }

    #[test]
    fn real_reference_3_a_conference_paper() {
        let (parsed, _) = parse(
            "K. R. Fulton, A. Chan, D. Votipka, M. Hicks, and M. L. Mazurek, \u{201c}Benefits and \
             drawbacks of adopting a secure programming language: Rust as a case study,\u{201d} in \
             Seventeenth Symposium on Usable Privacy and Security (SOUPS 2021), 2021, \
             pp. 597\u{2013}616.",
        );
        let reference = parsed.reference().expect("must parse");

        assert_eq!(reference.kind(), EntryKind::InProceedings, "`in` marks a proceedings");
        assert_eq!(reference.authors().len(), 5);
        assert_eq!(
            reference.container(),
            Some("Seventeenth Symposium on Usable Privacy and Security (SOUPS 2021)")
        );
        // The year is the TRAILING 2021, not the one inside "(SOUPS 2021)".
        assert_eq!(reference.year().unwrap().get(), 2021);
        assert_eq!(reference.pages().unwrap().page_count(), Some(20));
    }

    #[test]
    fn real_reference_4_a_thesis_printed_exactly_like_a_book() {
        // THE interesting one. This is a PhD dissertation, and the biblatex
        // `ieee` style has dropped the "PhD thesis" designator entirely -- so
        // from the string alone it is indistinguishable from a book.
        //
        // We do NOT guess. We read what is there (Book), and we REPORT the
        // ambiguity.
        let (parsed, anomalies) = parse(
            "M. R. Chen, Statistical Models as Testbeds for Iterative Human-in-the-Loop \
             Annotation Workflow Development. University of California, Berkeley, 2024.",
        );
        let reference = parsed.reference().expect("must parse");

        assert_eq!(reference.authors().len(), 1);
        assert_eq!(
            reference.title(),
            Some(
                "Statistical Models as Testbeds for Iterative Human-in-the-Loop Annotation \
                 Workflow Development"
            )
        );
        assert_eq!(reference.year().unwrap().get(), 2024);
        assert_eq!(reference.kind(), EntryKind::Book);

        // ...and the ambiguity is declared, not buried.
        assert!(
            anomalies.iter().any(|a| matches!(
                a,
                Anomaly::AmbiguousEntryKind {
                    alternative: EntryKind::Thesis,
                    ..
                }
            )),
            "the book/thesis ambiguity must be reported: {anomalies:#?}"
        );
    }

    #[test]
    fn real_reference_8_five_authors_with_hyphenated_initials() {
        let (parsed, _) = parse(
            "B.-C. Du, Y.-L. He, Y. Qiu, Q. Liang, and Y.-P. Zhou, \u{201c}Investigation on \
             cross-lingual transfer characteristics of subword tokenizers in a multi-encoder \
             pipeline,\u{201d} International Communications in Language and Speech Processing, \
             vol. 96, pp. 61\u{2013}68, 2018.",
        );
        let reference = parsed.reference().expect("must parse");

        assert_eq!(reference.authors().len(), 5);
        assert_eq!(reference.authors().first().unwrap().family(), Some("Du"));
        assert_eq!(reference.authors().authors()[4].family(), Some("Zhou"));
        assert_eq!(reference.volume(), Some("96"));
        assert_eq!(reference.pages().unwrap().page_count(), Some(8));
    }

    #[test]
    fn real_reference_11_an_article_number_printed_with_a_digit_group_separator() {
        // `p. 111 144` is biblatex printing article number 111144 with a group
        // separator. Reading it as page 111 and dropping 144 is WRONG, and it is
        // exactly the bug the plot engine hit on this same paper (with commas).
        let (parsed, anomalies) = parse(
            "L. Vega, G. Park, D. O\u{2019}Brien, and R. Kaur, \u{201c}Corpus validation of a \
             dependency parser using annotated treebank data from the reference evaluation \
             suite,\u{201d} Journal of Language Technology, vol. 377, p. 111 144, 2021.",
        );
        let reference = parsed.reference().expect("must parse");

        assert_eq!(reference.pages().unwrap().start().as_str(), "111144");
        assert_eq!(reference.pages().unwrap().end(), None, "an article number has no end page");
        assert_eq!(reference.volume(), Some("377"));
        assert_eq!(reference.year().unwrap().get(), 2021);

        // The assumption is DECLARED.
        assert!(
            anomalies.iter().any(|a| matches!(
                a,
                Anomaly::AssumedDigitGrouping { read_as, .. } if read_as == "111144"
            )),
            "the digit regrouping must be declared: {anomalies:#?}"
        );
    }

    #[test]
    fn real_reference_7_a_tech_report_with_a_messy_tail() {
        // The messiest line in the paper: "Advanced . . ." is a truncated
        // publisher name that means nothing. We recover what we can and say so.
        let (parsed, _) = parse(
            "L. Vega, R. Kaur, and A. Moreau, \u{201c}Parser validation using the reference \
             evaluation corpus,\u{201d} European Bioinformatics Institute \
             (EBI), Hinxton, UK (United Kingdom). Advanced . . ., Tech. Rep., 2019.",
        );
        let reference = parsed.reference().expect("must parse");

        assert_eq!(reference.kind(), EntryKind::TechReport, "`Tech. Rep.` is explicit");
        assert_eq!(reference.authors().len(), 3);
        assert_eq!(
            reference.institution(),
            Some("European Bioinformatics Institute (EBI)")
        );
        assert_eq!(reference.year().unwrap().get(), 2019);

        // And the tail we could not use is REPORTED, not silently dropped.
        assert!(parsed.is_partial(), "the messy tail must be admitted");
        assert!(
            reference.unparsed().is_some_and(|u| u.contains("Hinxton")),
            "the remainder must name what was left over: {:?}",
            reference.unparsed()
        );
    }

    #[test]
    fn real_reference_9_a_title_beginning_with_a_comma_terminated_word() {
        // THE BUG A REAL REFERENCE LIST FOUND.
        //
        // The title begins "Design, implementation and deployment...". The
        // first version of the author scanner saw `Design` as a short
        // capitalised chunk, decided it looked like a surname, and produced
        //
        //     a FOURTH AUTHOR, CALLED "Design", WHO DOES NOT EXIST
        //
        // sitting in the .bib file looking exactly as trustworthy as the three
        // real ones. Every synthetic test passed. Only a real reference list
        // caught it.
        let (parsed, _) = parse(
            "J. E. Barker, N. Whitfield, and P. F. Alvarez, Design, implementation and \
             deployment of a large-scale multilingual text retrieval and annotation \
             platform for digital libraries. University of California, \
             Berkeley, 2014.",
        );
        let reference = parsed.reference().expect("must parse");

        assert_eq!(reference.authors().len(), 3, "THREE authors, not four");
        assert!(
            !reference
                .authors()
                .authors()
                .iter()
                .any(|a| a.as_written().contains("Design")),
            "there is no researcher called `Design`: {:?}",
            reference
                .authors()
                .authors()
                .iter()
                .map(Author::as_written)
                .collect::<Vec<_>>()
        );
        assert_eq!(reference.authors().authors()[2].family(), Some("Alvarez"));

        // ...and `Design` is where it belongs: at the start of the title.
        assert!(
            reference.title().unwrap().starts_with("Design, implementation and deployment"),
            "got: {:?}",
            reference.title()
        );
    }

    #[test]
    fn the_word_and_introduces_the_final_author_and_nothing_after_it_is_one() {
        // Rule 1 of `author_region_end`, asserted directly. This is what makes
        // the `Design` case impossible rather than merely unlikely.
        let (parsed, _) = parse(
            "A. One, B. Two, and C. Three, Some Capitalised Title. Press, 2020.",
        );
        let reference = parsed.reference().unwrap();
        assert_eq!(reference.authors().len(), 3);
        assert_eq!(reference.title(), Some("Some Capitalised Title"));
    }

    #[test]
    fn a_family_comma_initials_author_list_still_works() {
        // Rule 2's first arm: a bare surname followed by an initials-only chunk.
        let (parsed, _) = parse("Smith, J., Jones, A., A Book About Syntax. Press, 2020.");
        let reference = parsed.reference().unwrap();
        assert_eq!(reference.authors().len(), 2);
        assert_eq!(reference.authors().first().unwrap().family(), Some("Smith"));
        assert_eq!(reference.title(), Some("A Book About Syntax"));
    }

    #[test]
    fn a_plain_firstname_lastname_leading_the_list_still_works() {
        // Rule 2's second arm.
        let (parsed, _) = parse("John Smith, A Book About Syntax. Press, 2020.");
        let reference = parsed.reference().unwrap();
        assert_eq!(reference.authors().len(), 1);
        assert_eq!(reference.authors().first().unwrap().as_written(), "John Smith");
    }

    // -- The failure modes ------------------------------------------------

    #[test]
    fn garbage_is_unparsed_and_never_becomes_a_fabricated_reference() {
        let (parsed, anomalies) = parse("qwertyuiop asdfghjkl zxcvbnm");
        assert!(matches!(parsed, ParsedReference::Unparsed(_)));
        assert!(parsed.reference().is_none(), "no reference may be invented");
        assert!(
            anomalies
                .iter()
                .any(|a| matches!(a, Anomaly::UnparseableEntry { .. }))
        );

        // ...and the raw string is kept.
        let ParsedReference::Unparsed(raw) = parsed else {
            unreachable!()
        };
        assert_eq!(raw.text(), "qwertyuiop asdfghjkl zxcvbnm");
    }

    #[test]
    fn an_unparseable_line_still_knows_where_it_came_from() {
        let (parsed, _) = parse("!!!! ???? ....");
        assert_eq!(parsed.provenance().locator().page().unwrap().get(), 15);
        assert_eq!(parsed.provenance().document().as_str(), "aligned_corpus.pdf");
    }

    #[test]
    fn a_bad_isbn_checksum_is_reported_rather_than_accepted() {
        let (_, anomalies) = parse(
            "J. Smith, A Book About Syntax. Some Press, 2020. ISBN: 978-0-13-235088-5.",
        );
        assert!(
            anomalies
                .iter()
                .any(|a| matches!(a, Anomaly::InvalidIdentifier { .. })),
            "a bad ISBN check digit must be a finding: {anomalies:#?}"
        );
    }

    #[test]
    fn a_doi_is_recovered_when_the_source_prints_one() {
        let (parsed, _) = parse(
            "J. Smith, \u{201c}A paper,\u{201d} Some Journal, vol. 1, pp. 1\u{2013}2, 2020. \
             doi: 10.1016/j.csl.2021.101144.",
        );
        let reference = parsed.reference().unwrap();
        assert_eq!(
            reference.identifiers().doi.as_ref().unwrap().as_str(),
            "10.1016/j.csl.2021.101144"
        );
    }

    #[test]
    fn an_et_al_list_is_truncated_not_padded_out() {
        let (parsed, _) = parse(
            "K. R. Fulton et al., \u{201c}A paper,\u{201d} Some Journal, 2021.",
        );
        let reference = parsed.reference().unwrap();
        assert_eq!(reference.authors().len(), 1);
        assert!(reference.authors().is_truncated());
    }
}

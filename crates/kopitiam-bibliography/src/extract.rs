//! Extracting a bibliography from a real document.
//!
//! Find the reference section, split it into entries, parse each one, and pick
//! up the in-text citations that point at them.
//!
//! # Why this reads geometry and not just text
//!
//! A reference list is **typographically signalled**, and the signal is a
//! *hanging indent*: the first line of each entry starts at the left margin (or
//! carries a label at it), and every continuation line is indented under it.
//!
//! ```text
//!    1. M. R. Chen, S. Novak, and J. P. Alvarez, "An open-source toolkit ...
//!       for multilingual text alignment (mtat)," International Journal of ...
//!       Computational Linguistics, vol. 6, no. 4, pp. 281-301, 2024.
//!    2. M. R. Chen, S. Novak, and J. P. Alvarez, Mtat alignment toolkit, ...
//! ```
//!
//! Flatten that to a string and the entry boundaries are gone. Two references
//! merge into one, or one splits into three, and the resulting citations point
//! at works that do not exist. The x-coordinate of each line's first glyph is
//! what tells them apart — which is exactly what [`kopitiam_pdf`] recovers and
//! nothing downstream of it currently keeps.
//!
//! # Where [`kopitiam_document`] falls short, and what is done about it
//!
//! This module **reuses** the Document Engine for what it is good at (finding
//! in-text citations in paragraph text) and works from [`kopitiam_pdf::Page`]
//! spans directly for the reference section. That is not a fork and not a second
//! PDF parser — no PDF byte is parsed here. It is line-grouping over
//! already-extracted spans, and it is necessary because of three genuine gaps,
//! recorded here so they can be fixed upstream rather than rediscovered:
//!
//! 1. **`kopitiam-document` discards line geometry.** `reconstruct()` groups
//!    spans into lines internally and then throws the x-positions away, emitting
//!    only `Block`s of text. The hanging indent — the one signal that delimits
//!    reference entries — does not survive. Nothing in the public API can
//!    recover it.
//!
//! 2. **Its list detection breaks on continuation lines.** A numbered reference
//!    list looks like an ordered list, and `try_list` stops at the first line
//!    that does not itself start with a marker. Since every continuation line of
//!    a reference is unmarked, a twelve-entry reference list comes out as twelve
//!    one-item lists interleaved with orphaned paragraphs.
//!
//! 3. **`REFERENCES` is not detected as a heading.** `heading_level()` requires
//!    a font ratio of ≥1.12 or a numbered-section prefix. A reference-section
//!    heading is very often body-sized, bold, and unnumbered — as it is in many
//!    real papers — so it comes back as a `Paragraph`, and the section cannot be
//!    located at all.
//!
//! None of these are bugs in `kopitiam-document`; it was built for prose, and
//! for prose it is right. They are simply things a bibliography needs that a
//! prose reconstructor does not provide. See the bead filed against this crate.

use std::sync::LazyLock;

use kopitiam_pdf::{Page, TextSpan};
use regex::Regex;

use crate::anomaly::Anomaly;
use crate::bibliography::Bibliography;
use crate::citation::{CitationRef, SourcedCitation};
use crate::entry::{ParsedReference, parse_reference_line};
use crate::error::Error;
use crate::provenance::{DocumentId, Provenance};

/// Headings that begin a reference section.
///
/// Matched on the whole line, case-insensitively, allowing a section number
/// (`5. REFERENCES`) and trailing punctuation.
static SECTION_HEADING: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)^\s*(?:\d+\.?\s*)?(references|reference list|bibliography|works cited|literature cited)\s*:?\s*$",
    )
    .unwrap()
});

/// Headings that **end** a reference section.
///
/// A reference list usually runs to the end of the document, but not always —
/// appendices and nomenclature tables follow it often enough that walking off
/// the end into one would silently turn a table of symbols into forty
/// unparseable references.
static SECTION_TERMINATOR: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)^\s*(?:\d+\.?\s*)?(appendix|appendices|nomenclature|acknowledg(?:e)?ments?|about the authors?|author biograph|index|glossary|notation)\b",
    )
    .unwrap()
});

/// A numbered reference-list label: `1.`, `12.`, `[1]`, `(1)`.
static ENTRY_LABEL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*(?:\[(\d{1,3})\]|\((\d{1,3})\)|(\d{1,3})[.)])\s+(.*)$").unwrap());

/// How close two glyph baselines must be, relative to font size, to count as the
/// same visual line.
const SAME_LINE_TOLERANCE: f32 = 0.4;

/// How wide a gap between two spans must be, relative to font size, before a
/// space is inserted between them.
const WORD_GAP_RATIO: f32 = 0.15;

/// How far a line's first glyph may sit from the section's left margin and still
/// count as being *at* the margin (i.e. starting a new entry rather than
/// continuing one), in points.
///
/// Two points is roughly a third of a character width at 10pt. Generous enough to
/// absorb the sub-point jitter of a PDF's glyph positioning; far tighter than the
/// ~12pt hanging indent any real style uses.
const MARGIN_TOLERANCE: f32 = 2.0;

/// One visual line of text, **with the geometry that makes it useful**.
#[derive(Debug, Clone)]
struct Line {
    text: String,
    /// The x-coordinate of the line's first glyph. This is the hanging-indent
    /// signal, and it is the whole reason this module exists.
    x: f32,
    page: usize,
}

/// Extracts a bibliography from a PDF.
///
/// # Errors
///
/// [`Error::Pdf`] if the file cannot be read or parsed. A document with **no
/// reference section** is not an error — it is a document with no reference
/// section, and it comes back as an empty [`Bibliography`] carrying an
/// [`Anomaly::NoReferenceSection`].
pub fn extract_pdf(path: impl AsRef<std::path::Path>) -> Result<Bibliography, Error> {
    let path = path.as_ref();
    let pages = kopitiam_pdf::extract(path)?;
    let document = DocumentId::new(
        path.file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string()),
    )?;
    Ok(extract_pages(&pages, &document))
}

/// Extracts a bibliography from already-extracted pages.
///
/// The seam a caller reaches for when the pages came from somewhere other than a
/// file on disk — and the one every test in this crate uses, so that the test
/// corpus is synthetic pages rather than checked-in PDFs.
pub fn extract_pages(pages: &[Page], document: &DocumentId) -> Bibliography {
    let mut anomalies = Vec::new();

    let lines: Vec<Line> = pages.iter().flat_map(page_lines).collect();

    // -- 1. Find the reference section.
    let Some(start) = lines
        .iter()
        .position(|line| SECTION_HEADING.is_match(line.text.trim()))
    else {
        // Not an error. A document without a reference section is a real thing.
        let provenance = Provenance::from_page(
            document,
            1,
            "(no reference section heading was found in this document)",
        );
        if let Ok(provenance) = provenance {
            anomalies.push(Anomaly::NoReferenceSection { provenance });
        }
        return Bibliography::new(
            document.clone(),
            Vec::new(),
            citations_in(pages, document),
            anomalies,
        );
    };

    let body = &lines[start + 1..];
    let end = body
        .iter()
        .position(|line| SECTION_TERMINATOR.is_match(line.text.trim()))
        .unwrap_or(body.len());
    let section = &body[..end];

    // -- 2. Split into entries.
    let groups = split_entries(section);

    // -- 3. Parse each.
    let entries: Vec<ParsedReference> = groups
        .into_iter()
        .filter_map(|group| {
            let page = group.first()?.page;
            // The VERBATIM text keeps the printed line breaks -- that is what a
            // reader checks against the page. `Provenance::from_page` derives
            // the normalised (line-joined, de-hyphenated) form from it.
            let verbatim = group
                .iter()
                .map(|line| line.text.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            let provenance = Provenance::from_page(document, page, verbatim).ok()?;
            Some(parse_reference_line(provenance, &mut anomalies))
        })
        .collect();

    // -- 4. The in-text citations that point at them.
    let citations = citations_in(&pages[..pages.len().min(page_index_of(section, pages))], document);

    Bibliography::new(document.clone(), entries, citations, anomalies)
}

/// The number of pages that precede the reference section, so in-text citations
/// are looked for in the body and not in the reference list itself (where every
/// `[12]` label would otherwise be read as a citation of work 12).
fn page_index_of(section: &[Line], pages: &[Page]) -> usize {
    section
        .first()
        .map(|line| line.page.saturating_sub(1))
        .unwrap_or(pages.len())
        .max(1)
}

/// Groups the reference section's lines into entries.
///
/// Two signals, in order of reliability:
///
/// 1. **A numbered label** (`1.`, `[1]`) at the start of a line, *in sequence*.
///    The sequence check is what stops a stray `2021.` at the head of a
///    continuation line from being read as the start of entry 2021 — a real
///    hazard, since a reference line very often *ends* with its year and wraps
///    right after it.
///
/// 2. **A hanging indent**: a line whose first glyph sits at the section's left
///    margin, where continuation lines sit further right. This is what carries
///    an unnumbered (author-year) bibliography, where there are no labels at all.
fn split_entries(section: &[Line]) -> Vec<Vec<Line>> {
    let lines: Vec<&Line> = section
        .iter()
        .filter(|line| !line.text.trim().is_empty())
        .collect();
    if lines.is_empty() {
        return Vec::new();
    }

    // The left margin: the leftmost glyph in the whole section.
    let margin = lines
        .iter()
        .map(|line| line.x)
        .fold(f32::INFINITY, f32::min);

    let mut groups: Vec<Vec<Line>> = Vec::new();
    let mut expected_label = 1u32;

    for line in lines {
        let labelled = ENTRY_LABEL.captures(&line.text).and_then(|caps| {
            let label: u32 = caps
                .get(1)
                .or_else(|| caps.get(2))
                .or_else(|| caps.get(3))?
                .as_str()
                .parse()
                .ok()?;
            let rest = caps.get(4)?.as_str().to_string();
            Some((label, rest))
        });

        let starts_entry = match &labelled {
            // In sequence: this is the label we were waiting for.
            Some((label, _)) if *label == expected_label => true,
            // A label out of sequence is almost certainly not a label. The
            // classic false positive is a year at the head of a wrapped line.
            Some(_) => false,
            // No label at all: fall back to the hanging indent.
            None => at_margin(line.x, margin) && !groups.is_empty(),
        };

        if starts_entry {
            if let Some((label, rest)) = labelled {
                expected_label = label + 1;
                let mut line = line.clone();
                // The label is not part of the reference.
                line.text = rest;
                groups.push(vec![line]);
            } else {
                groups.push(vec![line.clone()]);
            }
        } else if let Some(group) = groups.last_mut() {
            group.push(line.clone());
        } else {
            // Text before the first entry -- an unnumbered bibliography's first
            // entry, or a stray line. Start a group so it is not lost.
            groups.push(vec![line.clone()]);
        }
    }

    groups
}

fn at_margin(x: f32, margin: f32) -> bool {
    (x - margin).abs() <= MARGIN_TOLERANCE
}

/// Groups a page's spans into visual lines, keeping the x of the first glyph.
///
/// Not a PDF parser: [`kopitiam_pdf`] has already done that. This is
/// baseline-grouping over spans it produced, which `kopitiam-document` also does
/// internally and then discards the geometry of. See the module docs.
fn page_lines(page: &Page) -> Vec<Line> {
    let mut spans: Vec<&TextSpan> = page
        .spans
        .iter()
        .filter(|span| !span.text.trim().is_empty())
        .collect();

    // Reading order: down the page (PDF y grows upward, so descending), then
    // left to right.
    spans.sort_by(|a, b| {
        b.y.partial_cmp(&a.y)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut lines: Vec<Line> = Vec::new();
    let mut current: Vec<&TextSpan> = Vec::new();

    for span in spans {
        let same_line = current.first().is_some_and(|first| {
            let tolerance = first.font_size.max(1.0) * SAME_LINE_TOLERANCE;
            (first.y - span.y).abs() <= tolerance
        });

        if same_line {
            current.push(span);
        } else {
            if let Some(line) = assemble_line(&current, page.number) {
                lines.push(line);
            }
            current = vec![span];
        }
    }
    if let Some(line) = assemble_line(&current, page.number) {
        lines.push(line);
    }

    lines
}

fn assemble_line(spans: &[&TextSpan], page: usize) -> Option<Line> {
    spans.first()?;

    let mut text = String::new();
    let mut previous_end: Option<f32> = None;

    for span in spans {
        if let Some(end) = previous_end {
            let gap = span.x - end;
            let threshold = span.font_size.max(1.0) * WORD_GAP_RATIO;
            if gap > threshold && !text.ends_with(' ') {
                text.push(' ');
            }
        }
        text.push_str(&span.text);
        previous_end = Some(span.x + span.width);
    }

    let text = text.trim().to_string();
    if text.is_empty() {
        return None;
    }

    Some(Line {
        text,
        x: spans
            .iter()
            .map(|span| span.x)
            .fold(f32::INFINITY, f32::min),
        page,
    })
}

/// The in-text citations in the document body.
///
/// **Reuses [`kopitiam_document`]** rather than re-detecting them: its
/// `reconstruct()` already finds citation-shaped strings in paragraph text, and
/// writing a second detector would be duplicated logic for no gain. What this
/// crate adds is *structure* — turning its `Citation { text: String }` into a
/// [`CitationRef`] that can actually be resolved.
///
/// The page attributed to a citation is the page of the block it was found in,
/// via `Document::block_pages`.
fn citations_in(pages: &[Page], document: &DocumentId) -> Vec<SourcedCitation> {
    if pages.is_empty() {
        return Vec::new();
    }

    let mut citations = Vec::new();

    // Reconstruct page by page so that each citation can be attributed to a
    // page. (`Document::block_pages` gives the page of each block, but
    // `Document::citations` is a flat list with no link back to the block it
    // came from -- a fourth gap in the Document Engine, and the reason this is
    // done per page.)
    for page in pages {
        let reconstructed = kopitiam_document::reconstruct(std::slice::from_ref(page));
        for citation in &reconstructed.citations {
            let Ok(provenance) = Provenance::from_page(document, page.number, &citation.text)
            else {
                continue;
            };
            citations.push(SourcedCitation::new(
                CitationRef::parse(&citation.text),
                provenance,
            ));
        }
    }

    citations
}

#[cfg(test)]
mod tests {
    use super::*;
    use kopitiam_pdf::TextSpan;

    /// Builds a span at a position. The x is the load-bearing part.
    fn span(text: &str, x: f32, y: f32) -> TextSpan {
        TextSpan {
            text: text.to_string(),
            x,
            y,
            width: text.len() as f32 * 5.0,
            height: 10.0,
            font_size: 10.0,
            ..TextSpan::default()
        }
    }

    fn page(number: usize, spans: Vec<TextSpan>) -> Page {
        Page {
            number,
            width: 612.0,
            height: 792.0,
            spans,
        }
    }

    fn doc() -> DocumentId {
        DocumentId::new("synthetic.pdf").unwrap()
    }

    /// A synthetic reference section laid out the way a real paper's is:
    /// numbered labels at the margin, continuation lines indented.
    fn synthetic_reference_page() -> Page {
        page(
            2,
            vec![
                span("REFERENCES", 72.0, 700.0),
                // Entry 1, three lines, hanging indent.
                span("1. M. R. Chen, S. Novak, and J. P. Alvarez, \u{201c}An open-source toolkit", 72.0, 660.0),
                span("for multilingual text alignment (mtat),\u{201d} International Journal of", 90.0, 645.0),
                span("Computational Linguistics, vol. 6, no. 4, pp. 281\u{2013}301, 2024.", 90.0, 630.0),
                // Entry 2, two lines.
                span("2. L. Vega, R. Kaur, and A. Moreau, \u{201c}Parser validation,\u{201d}", 72.0, 610.0),
                span("Journal of Language Technology, vol. 377, p. 111 144, 2021.", 90.0, 595.0),
            ],
        )
    }

    #[test]
    fn finds_the_reference_section_and_splits_it_on_the_hanging_indent() {
        let bibliography = extract_pages(&[synthetic_reference_page()], &doc());

        assert_eq!(
            bibliography.entries().len(),
            2,
            "two entries, split on their labels and indents"
        );

        let first = bibliography.entries()[0].reference().unwrap();
        assert_eq!(first.authors().len(), 3);
        assert_eq!(first.year().unwrap().get(), 2024);
        assert_eq!(first.volume(), Some("6"));

        let second = bibliography.entries()[1].reference().unwrap();
        assert_eq!(second.volume(), Some("377"));
        assert_eq!(second.pages().unwrap().start().as_str(), "111144");
    }

    #[test]
    fn the_label_is_stripped_and_does_not_become_part_of_the_reference() {
        let bibliography = extract_pages(&[synthetic_reference_page()], &doc());
        let first = bibliography.entries()[0].reference().unwrap();
        assert!(
            !first.authors().first().unwrap().as_written().starts_with('1'),
            "the list label `1.` is not part of the first author's name"
        );
    }

    #[test]
    fn every_entry_carries_the_page_it_was_printed_on() {
        let bibliography = extract_pages(&[synthetic_reference_page()], &doc());
        for entry in bibliography.entries() {
            assert_eq!(entry.provenance().locator().page().unwrap().get(), 2);
            assert_eq!(entry.provenance().document().as_str(), "synthetic.pdf");
        }
    }

    #[test]
    fn the_verbatim_text_keeps_the_printed_line_breaks() {
        // What a reader checks against the page. The normalised form -- what the
        // parser read -- is kept separately.
        let bibliography = extract_pages(&[synthetic_reference_page()], &doc());
        let provenance = bibliography.entries()[0].provenance();
        assert!(
            provenance.verbatim().as_str().contains('\n'),
            "a wrapped entry keeps its wraps: {:?}",
            provenance.verbatim().as_str()
        );
        assert!(
            !provenance.normalised().as_str().contains('\n'),
            "and the parser saw it joined up"
        );
    }

    #[test]
    fn a_year_at_the_head_of_a_wrapped_line_is_not_mistaken_for_a_label() {
        // A real hazard: reference lines END with their year, and wrap right
        // after it. The sequence check is what prevents `2021.` becoming the
        // start of entry 2021.
        let page = page(
            2,
            vec![
                span("REFERENCES", 72.0, 700.0),
                span("1. J. Smith, \u{201c}A paper,\u{201d} Some Journal,", 72.0, 660.0),
                span("2021. Another sentence entirely.", 90.0, 645.0),
            ],
        );
        let bibliography = extract_pages(&[page], &doc());
        assert_eq!(
            bibliography.entries().len(),
            1,
            "`2021.` is a continuation, not entry number 2021"
        );
    }

    #[test]
    fn an_unnumbered_bibliography_splits_on_the_indent_alone() {
        // APA-style: no labels at all, only the hanging indent.
        let page = page(
            2,
            vec![
                span("Bibliography", 72.0, 700.0),
                span("Smith, J. (2019). A paper about syntax. Some Press.", 72.0, 660.0),
                span("A continuation of the first entry.", 90.0, 645.0),
                span("Jones, A. (2020). Another paper. Other Press.", 72.0, 625.0),
            ],
        );
        let bibliography = extract_pages(&[page], &doc());
        assert_eq!(bibliography.entries().len(), 2, "split on the indent");
    }

    #[test]
    fn a_document_with_no_reference_section_is_not_an_error() {
        let page = page(1, vec![span("Just some prose about grammar.", 72.0, 700.0)]);
        let bibliography = extract_pages(&[page], &doc());

        assert!(bibliography.entries().is_empty());
        assert!(
            bibliography
                .anomalies()
                .iter()
                .any(|a| matches!(a, Anomaly::NoReferenceSection { .. })),
            "but it IS a finding, and must be reported"
        );
    }

    #[test]
    fn an_appendix_after_the_references_does_not_become_forty_bad_references() {
        let page = page(
            2,
            vec![
                span("REFERENCES", 72.0, 700.0),
                span("1. J. Smith, \u{201c}A paper,\u{201d} Some Journal, 2021.", 72.0, 660.0),
                span("APPENDIX A", 72.0, 630.0),
                span("n = number of tokens", 72.0, 610.0),
                span("k = number of clusters", 72.0, 595.0),
                span("V = vocabulary size", 72.0, 580.0),
            ],
        );
        let bibliography = extract_pages(&[page], &doc());
        assert_eq!(
            bibliography.entries().len(),
            1,
            "the nomenclature table is not a reference list"
        );
    }

    #[test]
    fn an_entry_spanning_a_page_break_keeps_the_page_it_started_on() {
        let first = page(
            2,
            vec![
                span("REFERENCES", 72.0, 700.0),
                span("1. J. Smith, \u{201c}A very long paper title that runs", 72.0, 660.0),
            ],
        );
        let second = page(
            3,
            vec![span("on and on,\u{201d} Some Journal, 2021.", 90.0, 700.0)],
        );

        let bibliography = extract_pages(&[first, second], &doc());
        assert_eq!(bibliography.entries().len(), 1);
        assert_eq!(
            bibliography.entries()[0]
                .provenance()
                .locator()
                .page()
                .unwrap()
                .get(),
            2,
            "the page a reader should START looking on"
        );
    }

    #[test]
    fn in_text_citations_are_found_in_the_body_and_not_in_the_reference_list() {
        let body = page(
            1,
            vec![span(
                "The toolkit was written [1]. It was validated against a baseline [2].",
                72.0,
                700.0,
            )],
        );
        let refs = synthetic_reference_page();

        let bibliography = extract_pages(&[body, refs], &doc());

        assert_eq!(bibliography.citations().len(), 2);
        let resolved = bibliography.resolve_citations();
        assert_eq!(resolved.len(), 2);
        assert_eq!(resolved[0].entry, 0);
        assert_eq!(resolved[1].entry, 1);
        assert_eq!(bibliography.cited_entry_count(), 2);
    }
}

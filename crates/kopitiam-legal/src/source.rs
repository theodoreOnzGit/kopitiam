//! Turning a PDF into **page-located** lines, which is what provenance needs.
//!
//! # We do not write a second PDF parser, and we do not write a second
//! # document reconstructor
//!
//! [`kopitiam_pdf`] recovers text with position and font. [`kopitiam_document`]
//! reconstructs headings, paragraphs, lists and tables from that — including
//! two-column layouts, which statutes and judgments use constantly, and
//! paragraphs merged across page breaks. Both are good and both are reused
//! here. This module is *not* a fork of either.
//!
//! # The one place kopitiam-document does not fit, and why
//!
//! `kopitiam_document::Document` is a `Vec<Block>`, and **a `Block` does not
//! carry a page number**:
//!
//! ```text
//! pub struct Paragraph { pub text: String }        // <- no page, no bbox
//! pub struct Metadata  { pub source_pages: usize } // <- only a total count
//! ```
//!
//! For a scientific paper that is fine; you cite it by section. For a legal
//! instrument it is fatal, because a citation without a page is not a citation
//! a reader can follow, and this crate's central promise
//! ([`crate::provenance`]) is that **every extracted item can be traced back to
//! a page**. `reconstruct()` takes `&[Page]` and returns blocks with the page
//! provenance already discarded.
//!
//! ## The workaround, and its cost
//!
//! We call `kopitiam_document::reconstruct()` **one page at a time**. Feeding
//! it a single-page slice means we know, by construction, which page every
//! block it returns came from — so we get all of its reconstruction (heading
//! detection, paragraph assembly, list and table recognition, column
//! splitting, which all operate within a page) while keeping the page number.
//!
//! The cost is real and worth stating plainly: `reconstruct()`'s
//! `merge_page_breaks` pass, which repairs a paragraph split across a page
//! boundary, **cannot run**, because it is by definition a cross-page pass and
//! we are handing it one page at a time. Legal text splits across pages all the
//! time — a subsection routinely begins on p 14 and ends on p 15 — so this
//! genuinely degrades ingestion at page boundaries. We detect and report the
//! symptom (a provision whose text ends mid-sentence) as an
//! [`crate::AnomalyKind::Ambiguous`] rather than pretending it did not happen.
//!
//! ## What would fix it properly
//!
//! One field. If `kopitiam_document::Block` carried the page it came from —
//! `Paragraph { text: String, page: usize }`, or a `Located<T>` wrapper — this
//! module would delete its per-page loop, call `reconstruct()` once over the
//! whole document, and get cross-page paragraph merging *and* page provenance
//! together. That is a change to the Document Engine, not something this crate
//! should fork around, and it is the single highest-value thing the Document
//! Engine could do for every provenance-carrying consumer (legal, insurance,
//! literature, health, finance — all of which need page citations).
//!
//! # Font style is genuinely useful here
//!
//! [`kopitiam_pdf::TextSpan`] carries `font_style` (bold/italic, resolved from
//! the PDF `FontDescriptor` — see that crate's `font` module). Legal documents
//! signal structure *typographically*: section headings are bold, case names
//! are italicised, defined terms are often bold or in quotes. So we carry the
//! emphasis through to [`SourceLine`] and let the parser prefer it over
//! text-shape heuristics when it is available.

use kopitiam_pdf::{Page, TextSpan};

use crate::{LegalError, PageNumber};

/// Typographic emphasis on a line, recovered from the PDF's font resources.
///
/// `None` for "we could not tell" is deliberate and inherited from
/// [`kopitiam_pdf::FontStyle`]: "we do not know whether this was bold" and
/// "this was definitely not bold" are different facts, and a heading detector
/// that conflates them will mis-read every PDF whose fonts it failed to
/// resolve.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Emphasis {
    pub bold: Option<bool>,
    pub italic: Option<bool>,
}

impl Emphasis {
    /// Whether this line is *known* to be bold. `None` (unknown) is treated as
    /// not-bold here, because a heading detector should fall back to text
    /// shape rather than assume.
    pub fn is_bold(&self) -> bool {
        self.bold == Some(true)
    }

    pub fn is_italic(&self) -> bool {
        self.italic == Some(true)
    }
}

/// One line of source text, **with the page it is on**.
///
/// This is the unit that ingestion consumes, and the reason it exists: the
/// page number must survive all the way from the PDF to the
/// [`crate::Provenance`] on every extracted item.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceLine {
    pub page: PageNumber,
    pub text: String,
    pub emphasis: Emphasis,
}

impl SourceLine {
    pub fn new(page: PageNumber, text: impl Into<String>, emphasis: Emphasis) -> Self {
        Self {
            page,
            text: text.into(),
            emphasis,
        }
    }
}

/// Reads a PDF into page-located lines, reusing the Document Engine's
/// reconstruction one page at a time. See the module docs for the trade-off.
pub fn from_pdf(path: impl AsRef<std::path::Path>) -> Result<Vec<SourceLine>, LegalError> {
    let pages = kopitiam_pdf::extract(path).map_err(|e| LegalError::Pdf(e.to_string()))?;
    Ok(from_pages(&pages))
}

/// Reconstructs page-located lines from already-extracted PDF pages.
pub fn from_pages(pages: &[Page]) -> Vec<SourceLine> {
    let mut lines = Vec::new();
    for page in pages {
        let Ok(page_number) = PageNumber::new(page.number) else {
            // kopitiam-pdf numbers pages from 1; a 0 here would mean the
            // extractor lost the page identity, and we will not invent one.
            continue;
        };

        // Reconstruct THIS PAGE ALONE, so that the page number survives.
        // `kopitiam_document::reconstruct` gives us heading detection,
        // paragraph assembly and column splitting; feeding it a one-page
        // slice is what preserves the provenance it would otherwise discard.
        let document = kopitiam_document::reconstruct(std::slice::from_ref(page));

        for block in &document.blocks {
            for text in block_lines(block) {
                if text.trim().is_empty() {
                    continue;
                }
                let emphasis = emphasis_for(page, &text);
                lines.push(SourceLine::new(page_number, text, emphasis));
            }
        }
    }
    lines
}

/// Flattens a reconstructed block into the lines ingestion cares about.
///
/// Tables and figures are *not* discarded silently — their cells and captions
/// come through as text — because a statutory Schedule is very often a table,
/// and dropping it would lose operative content.
fn block_lines(block: &kopitiam_document::Block) -> Vec<String> {
    use kopitiam_document::Block;
    match block {
        Block::Heading(h) => vec![h.text.clone()],
        Block::Paragraph(p) => vec![p.text.clone()],
        Block::List(l) => l.items.clone(),
        Block::Quote(q) => vec![q.text.clone()],
        Block::CodeBlock(c) => vec![c.text.clone()],
        Block::Table(t) => {
            let mut out = Vec::new();
            if !t.headers.is_empty() {
                out.push(t.headers.join(" | "));
            }
            out.extend(t.rows.iter().map(|row| row.join(" | ")));
            out
        }
        Block::Figure(f) => f.caption.clone().into_iter().collect(),
    }
}

/// Recovers the emphasis of a reconstructed line by looking back at the raw
/// spans it was built from.
///
/// The Document Engine's AST does not carry font information (same gap as the
/// page number — see the module docs), so we re-associate by matching the
/// line's opening words against the spans on the page. A line counts as bold
/// only if *every* span we matched to it is bold, so a sentence containing one
/// bold word is not mistaken for a heading.
fn emphasis_for(page: &Page, line: &str) -> Emphasis {
    let opening: String = line.chars().take(24).collect::<String>().trim().to_string();
    if opening.is_empty() {
        return Emphasis::default();
    }

    let matching: Vec<&TextSpan> = page
        .spans
        .iter()
        .filter(|span| {
            let t = span.text.trim();
            !t.is_empty() && (opening.starts_with(t) || t.starts_with(&opening))
        })
        .collect();

    if matching.is_empty() {
        return Emphasis::default();
    }

    // "Unknown" must not be laundered into "false": only assert bold/italic
    // when every contributing span agrees and none is unknown.
    let all = |f: fn(&kopitiam_pdf::FontStyle) -> Option<bool>| -> Option<bool> {
        let values: Option<Vec<bool>> = matching.iter().map(|s| f(&s.font_style)).collect();
        values.map(|v| v.iter().all(|b| *b))
    };

    Emphasis {
        bold: all(|s| s.bold),
        italic: all(|s| s.italic),
    }
}

/// Builds page-located lines from plain text, one entry per page.
///
/// This is the path synthetic test documents take, and the path any
/// already-digital source (a statute downloaded as text, a contract in
/// Markdown) should take. It exists so that ingestion is testable without a
/// PDF fixture, and so that "no PDF available" never becomes "no ingestion".
///
/// `pages` is `(page_number, page_text)`; page numbers must be 1-based, and a
/// 0 is rejected rather than silently renumbered.
pub fn from_text_pages(pages: &[(usize, &str)]) -> Result<Vec<SourceLine>, LegalError> {
    let mut lines = Vec::new();
    for (number, text) in pages {
        let page = PageNumber::new(*number)?;
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            lines.push(SourceLine::new(page, line.trim(), Emphasis::default()));
        }
    }
    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_pages_carry_their_page_number_through() {
        let lines = from_text_pages(&[(1, "first\nsecond"), (2, "third")]).unwrap();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0].page.get(), 1);
        assert_eq!(lines[1].page.get(), 1);
        assert_eq!(lines[2].page.get(), 2);
        assert_eq!(lines[2].text, "third");
    }

    #[test]
    fn there_is_no_page_zero_even_from_text() {
        assert!(from_text_pages(&[(0, "text")]).is_err());
    }

    #[test]
    fn unknown_emphasis_is_not_laundered_into_false() {
        let e = Emphasis::default();
        assert_eq!(e.bold, None, "unknown must stay unknown");
        assert!(!e.is_bold(), "but a detector should fall back, not assume");
    }
}

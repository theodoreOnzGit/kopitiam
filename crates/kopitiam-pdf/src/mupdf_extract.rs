//! MuPDF-engine extraction entry point: the fixed text-extraction path.
//!
//! [`extract_mupdf`] mirrors [`crate::extract`]'s signature (`path -> Vec<Page>`)
//! but drives the faithful MuPDF port ([`crate::mupdf`]) instead of the legacy
//! `pdf-extract` backend. For each page it runs
//! [`mupdf::page_to_stext_segmented`], which returns a [`StextPage`] already in
//! **true reading order** (columns linearised by the whitespace-cover / X-Y-cut
//! detector) with **correct inter-word spacing** (spurious inter-glyph spaces
//! gone, missing spaces synthesised). Those are exactly the two bugs the legacy
//! path has.
//!
//! ## The mapping: one `StextLine` -> one [`TextSpan`]
//!
//! The stext tree is already blocks -> lines -> chars, with the chars carrying
//! synthesised spaces in the stream. So each [`StextLine`] maps to a single
//! [`TextSpan`] whose `text` is the line's chars concatenated verbatim (the
//! spacing is already correct — we must not re-derive it), and whose geometry
//! comes from the line bbox (`x`/`y`/`width`/`height`) and char `size`
//! (`font_size`). Blocks are emitted in their (already reading-order) sequence,
//! so the resulting `Page.spans` are in reading order line-by-line.
//!
//! Because the spans are already ordered and paragraph-grouped, the downstream
//! [`kopitiam-document` reconstruction] must NOT re-split columns or re-group by
//! baseline — see `kopitiam_document::reconstruct_preordered`.
//!
//! Font name/style are left unset (`None` / default): the MuPDF port's `Font`
//! does not expose a PostScript base-font name, and the reconstruction layer's
//! heading detection keys off the `font_size` ratio, not the name.

use std::path::Path as FsPath;

use crate::mupdf::{
    self, StextBlock, StextLine, StextPage, structured_text::StextOptions,
    xref::PdfDocument as MupdfDocument,
};
use crate::page::{Page, TextSpan};

use crate::extractor::ExtractError;

/// Extract physical layout (pages + text spans) from a PDF file on disk using
/// the ported MuPDF engine. Mirror of [`crate::extract`] but with the two
/// legacy bugs (spurious inter-glyph spaces, two-column interleave) fixed.
pub fn extract_mupdf(path: impl AsRef<FsPath>) -> Result<Vec<Page>, ExtractError> {
    let bytes = std::fs::read(path).map_err(|e| ExtractError::Mupdf(e.to_string()))?;
    extract_mupdf_from_bytes(&bytes)
}

/// Extract physical layout using the ported MuPDF engine from PDF bytes already
/// in memory. Mirror of [`crate::extract_from_bytes`].
pub fn extract_mupdf_from_bytes(bytes: &[u8]) -> Result<Vec<Page>, ExtractError> {
    let doc = MupdfDocument::open_slice(bytes).map_err(|e| ExtractError::Mupdf(e.to_string()))?;
    let opts = StextOptions::default();

    let mut pages = Vec::with_capacity(doc.page_count());
    let mut skipped = 0usize;
    let mut last_failure: Option<String> = None;

    for index in 0..doc.page_count() {
        match mupdf::page_to_stext_segmented(&doc, index, opts) {
            Ok(stext) => pages.push(stext_to_page(index, &stext)),
            Err(err) => {
                skipped += 1;
                last_failure = Some(err.to_string());
            }
        }
    }

    // If we salvaged nothing at all yet something went wrong, surface the
    // failure instead of pretending we extracted an empty document — matching
    // the legacy `run`'s contract.
    if pages.is_empty()
        && let Some(reason) = last_failure
    {
        return Err(ExtractError::Mupdf(reason));
    }

    if skipped > 0 {
        eprintln!(
            "warning: skipped {skipped} unreadable page(s) while extracting text \
             with the mupdf engine; continued with the remaining pages"
        );
    }

    Ok(pages)
}

/// Map a reading-order [`StextPage`] to a [`Page`], emitting one [`TextSpan`]
/// per [`StextLine`] in block/line order (already reading order).
/// Extract **one** page's text layout.
///
/// [`extract_mupdf_from_bytes`] does the whole document before it returns
/// anything, which is right for a batch conversion and wrong for a viewer: on
/// a 506-page book that is tens of seconds before the first word can be
/// shown. This is the incremental entry point a UI wants — extract a page,
/// display it, extract the next — and it is what `kpdf`'s reflow mode uses to
/// fill in progressively instead of freezing.
///
/// `None` for a page that cannot be read, so one broken page does not stop a
/// caller walking the rest.
pub fn extract_mupdf_page(doc: &MupdfDocument, index: usize) -> Option<Page> {
    let stext = mupdf::page_to_stext_segmented(doc, index, StextOptions::default()).ok()?;
    Some(stext_to_page(index, &stext))
}

fn stext_to_page(index: usize, stext: &StextPage) -> Page {
    let mb = stext.mediabox;
    let width = (mb.x1 - mb.x0).max(0.0);
    let height = (mb.y1 - mb.y0).max(0.0);

    let mut spans = Vec::new();
    for block in &stext.blocks {
        if let StextBlock::Text(tb) = block {
            for line in &tb.lines {
                if let Some(span) = line_to_span(line) {
                    spans.push(span);
                }
            }
        }
    }

    Page {
        // MuPDF pages are 0-indexed here; the legacy path numbers pages from the
        // PDF's own page dict keys. Downstream only uses relative order and the
        // per-block page derived in `merge_page_breaks`, so a 1-based sequential
        // number keeps parity with a human-facing page count.
        number: index + 1,
        width,
        height,
        spans,
    }
}

/// One [`StextLine`] -> one [`TextSpan`]. Text is the chars concatenated
/// verbatim (synthesised spaces already present); geometry from the line bbox
/// and char size.
fn line_to_span(line: &StextLine) -> Option<TextSpan> {
    if line.chars.is_empty() {
        return None;
    }

    // Sanitize at the extraction boundary (Task I-A), using the same
    // `textnorm` source of truth as the legacy `pdf-extract` path so a stray
    // control char from *either* engine can never reach the output: drop
    // control/format/private-use chars (a `U+0000` here would make ripgrep
    // treat the file as binary), expand ligatures, collapse exotic spaces and
    // strip the BOM. The synthesised spaces the stext stream already carries are
    // plain ASCII `' '` and pass through untouched.
    let mut text = String::new();
    for ch in &line.chars {
        crate::textnorm::normalize_char(ch.c, &mut text);
    }
    // A line that is nothing but whitespace (or whose only chars were dropped)
    // carries no content; drop it so it does not become an empty paragraph
    // downstream.
    if text.trim().is_empty() {
        return None;
    }

    let bbox = line.bbox;
    // Font size: the dominant (max) char size on the line. Char `size` is
    // MuPDF's text-rendering-matrix expansion, i.e. the effective point size.
    let font_size = line
        .chars
        .iter()
        .map(|ch| ch.size)
        .fold(0.0_f32, f32::max)
        .max(0.0);

    Some(TextSpan {
        text,
        x: bbox.x0,
        y: bbox.y0,
        width: (bbox.x1 - bbox.x0).max(0.0),
        height: (bbox.y1 - bbox.y0).max(0.0),
        font_size,
        ..TextSpan::default()
    })
}

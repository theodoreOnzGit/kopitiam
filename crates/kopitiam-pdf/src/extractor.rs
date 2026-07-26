use std::any::Any;
use std::collections::BTreeMap;
use std::panic::AssertUnwindSafe;
use std::path::Path as FsPath;
use std::rc::Rc;

use pdf_extract::{
    Document as PdfDocument, Error as PdfLoadError, MediaBox, OutputDev, OutputError, Transform,
    output_doc_page,
};

use crate::font::FontStyle;
use crate::font_resources::{ResolvedFont, WordFontQueue, font_timelines};
use crate::page::{Page, TextSpan};

#[derive(Debug, thiserror::Error)]
pub enum ExtractError {
    #[error("failed to load PDF: {0}")]
    Load(#[from] PdfLoadError),
    #[error("failed to parse PDF content streams: {0}")]
    Parse(#[from] OutputError),
    /// `pdf-extract` `panic!`ed while decoding a page's content stream, and
    /// *every* page failed the same way so there was nothing left to salvage.
    ///
    /// The classic trigger is a font whose `/ToUnicode` CMap does not cover
    /// the character codes the page actually uses and which also has no
    /// `/Encoding` to fall back on: deep inside `pdf-extract`'s `decode_char`
    /// this hits `expect("missing unicode map and encoding")` and unwinds. That
    /// panic blows straight past every `Result`/`?`/`thiserror`/`anyhow` on the
    /// way out, so the only way to keep it from aborting the whole process is
    /// to guard the extraction with `catch_unwind` (see `run`). This variant
    /// carries the recovered panic message.
    #[error("could not decode PDF text: {0}")]
    UnsupportedFont(String),
}

/// Extract physical layout (pages + text spans) from a PDF file on disk.
pub fn extract(path: impl AsRef<FsPath>) -> Result<Vec<Page>, ExtractError> {
    let doc = PdfDocument::load(path)?;
    run(&doc)
}

/// Extract physical layout from PDF bytes already in memory.
pub fn extract_from_bytes(bytes: &[u8]) -> Result<Vec<Page>, ExtractError> {
    let doc = PdfDocument::load_mem(bytes)?;
    run(&doc)
}

fn run(doc: &PdfDocument) -> Result<Vec<Page>, ExtractError> {
    // Resolve font resources for every page *before* handing the document
    // to `pdf-extract`'s own walk, so `PageCollector` can pop pre-computed
    // per-word font info in lock-step as `pdf-extract` calls `begin_word`.
    // See `crate::font_resources` for why this two-pass shape is necessary
    // and why the ordering guarantee holds.
    let word_fonts = font_timelines(doc);
    let mut collector = PageCollector {
        word_fonts,
        ..PageCollector::default()
    };

    // `pdf-extract` 0.12.0 can `panic!` outright deep inside content-stream
    // decoding (e.g. `expect("missing unicode map and encoding")` for a font
    // whose `/ToUnicode` map doesn't cover a code it uses and which has no
    // `/Encoding`). Such a panic unwinds past every `Result`/`?` here, so a
    // plain `output_doc(doc, &mut collector)?` cannot contain it -- it would
    // abort the whole process (and, for the CLI, a whole batch of PDFs).
    //
    // pdf-extract exposes a per-page entry point, `output_doc_page`, so we
    // drive extraction one page at a time and wrap each page in
    // `catch_unwind`. That isolates the blast radius to a single page: a good
    // page still lands in `collector.pages`, and a page that makes
    // pdf-extract panic is simply skipped (and counted) instead of discarding
    // the entire document. `AssertUnwindSafe` is required because `&mut
    // collector` (and `doc`) cross the unwind boundary; that is sound here
    // because we explicitly reset the collector's transient per-page state
    // after any failure (`reset_transient`), so no half-built page or word
    // leaks into the next page.
    let page_nums: Vec<u32> = doc.get_pages().into_keys().collect();

    // Install a no-op panic hook for the duration of the guarded calls so the
    // *expected, handled* panic doesn't dump a full backtrace to stderr for
    // every malformed page. The panic message itself is still recovered from
    // the payload below; only the noisy default hook is suppressed. The
    // previous hook is restored before we return.
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    let mut skipped = 0usize;
    let mut last_failure: Option<String> = None;

    for page_num in page_nums {
        let result =
            std::panic::catch_unwind(AssertUnwindSafe(|| output_doc_page(doc, &mut collector, page_num)));
        match result {
            // Page decoded cleanly; its `Page` is already in `collector.pages`
            // (pushed by `PageCollector::end_page`).
            Ok(Ok(())) => {}
            // A structured `OutputError` for this page. Rare, but treat it the
            // same as a panic: skip the page, remember why, and keep going so
            // one broken page doesn't sink the rest of the document.
            Ok(Err(err)) => {
                skipped += 1;
                last_failure = Some(err.to_string());
                collector.reset_transient();
            }
            // pdf-extract panicked while decoding this page. Recover the
            // message from the payload, skip the page, carry on.
            Err(payload) => {
                skipped += 1;
                last_failure = Some(panic_message(payload.as_ref()));
                collector.reset_transient();
            }
        }
    }

    std::panic::set_hook(previous_hook);

    // If we salvaged nothing at all yet something did go wrong, surface the
    // failure instead of pretending we extracted an empty document. (A
    // genuinely empty/zero-page PDF has no failures and returns `Ok(vec![])`,
    // exactly as before.)
    if collector.pages.is_empty()
        && let Some(reason) = last_failure
    {
        return Err(ExtractError::UnsupportedFont(reason));
    }

    if skipped > 0 {
        eprintln!(
            "warning: skipped {skipped} unreadable page(s) while extracting text \
             (fonts with no usable unicode mapping); continued with the remaining pages"
        );
    }

    Ok(collector.pages)
}

/// Recover a human-readable message from a `catch_unwind` panic payload.
/// Panic payloads are almost always a `&'static str` (plain `panic!("lit")`)
/// or a `String` (formatted panics, which is what `Option::expect` produces);
/// anything else we can only describe generically.
fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "pdf-extract panicked while decoding the page".to_string()
    }
}

#[derive(Default)]
struct PageCollector {
    pages: Vec<Page>,
    current: Option<Page>,
    word: WordBuilder,
    /// Per-page queues of the font resolved for each upcoming word, keyed
    /// by the same `page_num` `pdf-extract` passes to `begin_page`. The
    /// current page's queue is moved into `current_page_fonts` so it can
    /// be drained with plain `VecDeque::pop_front` from `begin_word`.
    word_fonts: BTreeMap<u32, WordFontQueue>,
    current_page_fonts: WordFontQueue,
}

impl PageCollector {
    /// Drop any half-built per-page state left behind when a page's
    /// extraction was aborted mid-stream (by a panic or an `OutputError`), so
    /// the *next* page starts from a clean slate instead of inheriting a
    /// partially built `Page`, a dangling in-progress word, or a stale
    /// word-font queue. `pages` (the salvaged output) is deliberately left
    /// untouched. Without this, a `begin_word` on the next page would flush a
    /// leftover word from the aborted page into the wrong `Page`.
    fn reset_transient(&mut self) {
        self.current = None;
        self.word = WordBuilder::default();
        self.current_page_fonts = WordFontQueue::default();
    }

    fn flush_word(&mut self) {
        if let Some(span) = self.word.take()
            && let Some(page) = self.current.as_mut()
        {
            page.spans.push(span);
        }
    }
}

impl OutputDev for PageCollector {
    fn begin_page(
        &mut self,
        page_num: u32,
        media_box: &MediaBox,
        _art_box: Option<(f64, f64, f64, f64)>,
    ) -> Result<(), OutputError> {
        self.current = Some(Page {
            number: page_num as usize,
            width: (media_box.urx - media_box.llx) as f32,
            height: (media_box.ury - media_box.lly) as f32,
            spans: Vec::new(),
        });
        // Move this page's pre-computed word-font queue into place so
        // `begin_word` can drain it. A page with no entry (font
        // resolution found nothing at all, e.g. a resource-less page)
        // just gets an empty queue, which degrades to `None` for every
        // word -- the same honest "unresolved" outcome as a queue that
        // runs out partway through.
        self.current_page_fonts = self.word_fonts.remove(&page_num).unwrap_or_default();
        Ok(())
    }

    fn end_page(&mut self) -> Result<(), OutputError> {
        self.flush_word();
        if let Some(page) = self.current.take() {
            self.pages.push(page);
        }
        Ok(())
    }

    fn output_character(
        &mut self,
        trm: &Transform,
        width: f64,
        _spacing: f64,
        font_size: f64,
        text: &str,
    ) -> Result<(), OutputError> {
        // `width` is the glyph advance normalized to 1/1000 em; multiplying by
        // font_size recovers the advance in user-space units (see pdf-extract's
        // `show_text`, which computes `w0 = font.get_width(c) / 1000.`).
        let advance = (width * font_size) as f32;
        self.word.push(
            text,
            trm.m31 as f32,
            trm.m32 as f32,
            advance,
            font_size as f32,
        );
        Ok(())
    }

    fn begin_word(&mut self) -> Result<(), OutputError> {
        self.flush_word();
        // `pdf-extract` calls `begin_word` exactly once per `Tj`/`TJ`-string
        // operand, in the same order `font_timelines` walked them in --
        // see `crate::font_resources` for the full argument. Draining one
        // entry per call keeps the two walks in lock-step; an empty queue
        // (walk diverged, or genuinely no font resolvable) yields `None`,
        // which is the honest "unknown" rather than reusing a stale font.
        self.word.font = self.current_page_fonts.pop_front().flatten();
        Ok(())
    }

    fn end_word(&mut self) -> Result<(), OutputError> {
        self.flush_word();
        Ok(())
    }

    fn end_line(&mut self) -> Result<(), OutputError> {
        self.flush_word();
        Ok(())
    }
}

/// Accumulates characters between `begin_word`/`end_word` into a single
/// `TextSpan`. Bounding box is derived from glyph origins and advances since
/// PDF text operators carry no explicit per-glyph box.
#[derive(Default)]
struct WordBuilder {
    text: String,
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    font_size: f32,
    /// Font resolved for this word by `crate::font_resources`, set once
    /// per word by `PageCollector::begin_word` (before any characters are
    /// pushed). `None` means resolution failed for this word specifically.
    font: Option<Rc<ResolvedFont>>,
}

impl WordBuilder {
    fn push(&mut self, ch: &str, x: f32, y: f32, advance: f32, font_size: f32) {
        if self.text.is_empty() {
            self.min_x = x;
            self.min_y = y;
            self.max_x = x;
            self.max_y = y + font_size;
            self.font_size = font_size;
        }
        self.text.push_str(ch);
        self.min_x = self.min_x.min(x);
        self.min_y = self.min_y.min(y);
        self.max_x = self.max_x.max(x + advance);
        self.max_y = self.max_y.max(y + font_size);
    }

    fn take(&mut self) -> Option<TextSpan> {
        if self.text.is_empty() {
            return None;
        }
        let (font_name, font_style) = match &self.font {
            Some(resolved) => (Some(resolved.base_font.clone()), resolved.style.clone()),
            None => (None, FontStyle::default()),
        };
        let span = TextSpan {
            text: std::mem::take(&mut self.text),
            x: self.min_x,
            y: self.min_y,
            width: self.max_x - self.min_x,
            height: (self.max_y - self.min_y).max(self.font_size),
            font_size: self.font_size,
            font_name,
            font_style,
        };
        *self = WordBuilder::default();
        Some(span)
    }
}

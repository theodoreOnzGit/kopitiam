use std::collections::BTreeMap;
use std::path::Path as FsPath;
use std::rc::Rc;

use pdf_extract::{
    Document as PdfDocument, Error as PdfLoadError, MediaBox, OutputDev, OutputError, Transform,
    output_doc,
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
    output_doc(doc, &mut collector)?;
    Ok(collector.pages)
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

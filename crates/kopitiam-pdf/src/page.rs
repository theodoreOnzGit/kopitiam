use crate::font::FontStyle;

/// One run of text extracted directly from a PDF content stream, before any
/// semantic reconstruction. Position is the glyph run's baseline origin;
/// `height` is approximated as `font_size` since PDF text operators do not
/// carry explicit glyph bounding boxes.
///
/// Derives `Default` (empty text, all-zero geometry, `font_name: None`,
/// `font_style` unknown) so that adding a field here in the future -- as
/// happened with `font_style` -- can be absorbed by call sites with
/// `..TextSpan::default()` instead of every struct-literal construction
/// site across the workspace needing an update in lockstep.
#[derive(Debug, Clone, Default)]
pub struct TextSpan {
    pub text: String,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub font_size: f32,
    /// Raw `BaseFont` PostScript name from the PDF's font resources, e.g.
    /// `"NimbusRomNo9L-Medi"` or `"ABCDEF+TimesNewRoman-BoldItalic"`
    /// (subset prefix, if any, left intact -- see [`crate::font`] for what
    /// that prefix means and where it gets stripped).
    ///
    /// This is resolved by a separate `lopdf`-based pass over the PDF's
    /// font resource dictionaries and content streams
    /// ([`crate::font_resources`]), because `pdf-extract`'s `OutputDev`
    /// callback API (kopitiam-pdf's extraction backend) does not expose
    /// the active font resource to per-character callbacks -- only its
    /// size. `None` means resolution genuinely failed for this span (an
    /// unresolvable resource name, an unrecognized content-stream
    /// construct, ...), never a guess.
    pub font_name: Option<String>,
    /// Structured style derived from `font_name`'s `FontDescriptor` and/or
    /// its naming convention -- see [`FontStyle`] and [`crate::font`] for
    /// how bold/italic are decided and why the descriptor is preferred.
    /// Left at its `Default` (every field `None`, i.e. "unknown") whenever
    /// `font_name` is `None`.
    pub font_style: FontStyle,
}

/// Physical layout of one PDF page: dimensions plus the text spans found on
/// it, in extraction order. No heading/paragraph/table meaning is attached
/// at this layer -- that is the reconstruction layer's job.
#[derive(Debug, Clone)]
pub struct Page {
    pub number: usize,
    pub width: f32,
    pub height: f32,
    pub spans: Vec<TextSpan>,
}

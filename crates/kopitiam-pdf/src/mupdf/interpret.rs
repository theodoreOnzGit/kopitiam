//! Ported from MuPDF `source/pdf/pdf-interpret.c` (the content-stream keyword
//! tokenizer `pdf_process_stream`, the operator dispatch `pdf_process_keyword`,
//! and the text-object-state helpers `pdf_tos_*` / `pdf_tos_make_trm`) (+
//! `include/mupdf/pdf/interpret.h`) (commit 19f1284, AGPL-3.0, © Artifex
//! Software, Inc.), translated to Rust for KOPITIAM (AGPL-3.0-only). Close
//! adaptation: the algorithms and numeric behaviour follow MuPDF; the code is
//! re-expressed in idiomatic Rust. See docs/ACKNOWLEDGEMENTS.md ("PDF &
//! document-extraction references").
//!
//! # The content-stream interpreter (text path)
//!
//! [`Processor`] runs a page's (or Form XObject's) content stream and emits
//! positioned glyphs to a [`TextDevice`]. It is the text-extraction slice of
//! MuPDF's interpreter: `pdf_process_stream` tokenises the operator stream
//! (reusing the WAVE-3a [`lex`](super::lex) tokenizer), collecting operands onto
//! a small numeric stack plus name/string/object slots, and `pdf_process_keyword`
//! dispatches each operator.
//!
//! This module carries the parts that live in `pdf-interpret.c`:
//! * the tokenizer loop ([`Processor::run_stream`]) and operator dispatch
//!   ([`Processor::process_keyword`]);
//! * the **text object state** ([`Tos`]): `Tm`/`Tlm` and the per-glyph advance,
//!   with `pdf_tos_translate` (`Td`/`TD`), `pdf_tos_set_matrix` (`Tm`),
//!   `pdf_tos_newline` (`T*`) and `pdf_tos_move_after_char`;
//! * [`make_trm`], the port of `pdf_tos_make_trm` -- the glyph text-rendering
//!   matrix `[size·Tz, 0, 0, size, 0, rise] · Tm` and the text-space advance.
//!
//! The graphics-state stack (`q`/`Q`/`cm`) and the `Tj`/`TJ`/`'`/`"` show path
//! live in [`super::op_run`] (`pdf-op-run.c`); resource lookup + the font cache in
//! [`super::resources`] (`pdf-resources.c`); page/XObject driving in
//! [`super::page_run`] (`pdf-page.c` / `pdf-run.c` / `pdf-xobject.c`). They share
//! this [`Processor`] via additional `impl` blocks.
//!
//! ## Wired to the draw device (rasterisation)
//!
//! Path construction/painting (`m l c v y h re S s f f* B B* b b* n`),
//! rectangular clipping (`W W*`), fill/stroke colour (`g G rg RG k K cs CS sc SC
//! scn SCN`), the line width (`w`) and image/form XObjects (`Do`) now drive the
//! [`DrawDevice`](super::draw_device) via the [`TextDevice`] sink's path/colour/
//! image callbacks. Colour is tracked in the graphics state and converted to
//! DeviceRGB; see [`super::op_run`] and [`super::resources`].
//!
//! ## Still deferred (parsed-and-ignored so the stream still runs)
//!
//! Type3 glyph metrics (`d0 d1`), shadings (`sh`), the ExtGState body (`gs`),
//! marked content (`MP DP BMC BDC EMC`) and the compatibility bracket (`BX EX`)
//! are recognised and skipped; their operands are consumed so the operator stream
//! stays in sync. Inline images (`BI … ID … EI`) are skipped as a raw byte span
//! (see [`Processor::skip_inline_image`]) -- a best-effort resync, not a decode.
//! Render modes 3 and 7 ("invisible") still emit glyphs: extraction needs them.

use std::collections::HashMap;

use super::draw_edge::FillRule;
use super::draw_path::Path;
use super::error::Result;
use super::font::Font;
use super::geometry::{Matrix, Point, Rect};
use super::lex::{lex, Token};
use super::object::Object;
use super::parse::{parse_array, parse_dict};
use super::resources::ColorSpace;
use super::stream::Stream;
use super::text_device::TextDevice;
use super::xref::PdfDocument;

/// The text state carried in the graphics state (`pdf_text_state`), saved and
/// restored by `q`/`Q`.
// MuPDF: struct pdf_text_state (interpret.h:491).
#[derive(Clone)]
pub(crate) struct TextState {
    /// `Tc` -- character spacing (unscaled text-space units).
    pub char_space: f32,
    /// `Tw` -- word spacing (unscaled text-space units).
    pub word_space: f32,
    /// The horizontal scale, `Tz / 100` (MuPDF stores it pre-divided).
    pub scale: f32,
    /// `TL` -- leading.
    pub leading: f32,
    /// `Tf` -- the current font (`None` until the first `Tf`).
    pub font: Option<Font>,
    /// `Tf` -- the current font size.
    pub size: f32,
    /// `Tr` -- the render mode (3 and 7 are invisible but still emitted).
    pub render: i32,
    /// `Ts` -- text rise.
    pub rise: f32,
}

impl Default for TextState {
    // MuPDF: pdf_init_gstate's text fields (pdf-op-run.c:1672).
    fn default() -> TextState {
        TextState {
            char_space: 0.0,
            word_space: 0.0,
            scale: 1.0,
            leading: 0.0,
            font: None,
            size: -1.0,
            render: 0,
            rise: 0.0,
        }
    }
}

/// The graphics state slice the text path needs: the CTM and the text state.
///
/// MuPDF's `pdf_gstate` also carries stroke state, blend mode and soft masks;
/// those beyond the fill/stroke material don't affect this port's output, so they
/// are dropped (blends/soft-masks deferred). The fill/stroke **colour**, current
/// **colourspace**, **line width** and the rectangular **clip** are carried here
/// (all `q`/`Q` push/pop) now that the draw device paints. This is what `q`/`Q`
/// push/pop.
// MuPDF: struct pdf_gstate (pdf-op-run.c, gstate.h) reduced to the drawn path.
#[derive(Clone)]
pub(crate) struct GState {
    /// The current transformation matrix (device space).
    pub ctm: Matrix,
    /// The text state (`Tc`/`Tw`/`Tz`/`TL`/`Tf`/`Tr`/`Ts`).
    pub text: TextState,
    /// The fill colour, already converted to DeviceRGB (0..=1). Default black.
    pub fill_color: [f32; 3],
    /// The stroke colour, DeviceRGB (0..=1). Default black.
    pub stroke_color: [f32; 3],
    /// The fill colourspace (`cs`), for interpreting `sc`/`scn` operands.
    pub fill_cs: ColorSpace,
    /// The stroke colourspace (`CS`), for interpreting `SC`/`SCN` operands.
    pub stroke_cs: ColorSpace,
    /// The line width (`w`), in path-space units. Default 1.0.
    pub line_width: f32,
    /// The current rectangular clip (`W`/`W*`), in device space *before* the
    /// device's own output transform. `None` = unclipped. Non-rect clips are
    /// bbox-approximated (see [`Processor::end_path`]).
    pub clip: Option<Rect>,
}

impl GState {
    // MuPDF: pdf_init_gstate (pdf-op-run.c:1672) -- fill/stroke default to black
    // DeviceGray, line width 1, no clip.
    fn new(ctm: Matrix) -> GState {
        GState {
            ctm,
            text: TextState::default(),
            fill_color: [0.0, 0.0, 0.0],
            stroke_color: [0.0, 0.0, 0.0],
            fill_cs: ColorSpace::Gray,
            stroke_cs: ColorSpace::Gray,
            line_width: 1.0,
            clip: None,
        }
    }
}

/// The text object state (`pdf_text_object_state`), reduced to the text-showing
/// subset: the text matrix `Tm`, the text-line matrix `Tlm`, and the pending
/// per-glyph advance. Unlike [`GState`] this is **not** stacked by `q`/`Q`; it is
/// reset by `BT` and persists otherwise (MuPDF keeps it on the processor).
// MuPDF: struct pdf_text_object_state (interpret.h:504) -- clip/bbox/gid fields
// dropped (no rasterisation, no glyph cache, no text clipping on the text path).
#[derive(Clone, Copy)]
pub(crate) struct Tos {
    /// `Tm` -- the text matrix.
    pub tm: Matrix,
    /// `Tlm` -- the text line matrix.
    pub tlm: Matrix,
    /// The horizontal advance to apply to `Tm` after the current glyph.
    pub char_tx: f32,
    /// The vertical advance to apply to `Tm` after the current glyph.
    pub char_ty: f32,
}

impl Tos {
    fn new() -> Tos {
        Tos { tm: Matrix::IDENTITY, tlm: Matrix::IDENTITY, char_tx: 0.0, char_ty: 0.0 }
    }

    // MuPDF: pdf_tos_translate (pdf-interpret.c:2071) -- `Td` / `TD`.
    pub fn translate(&mut self, tx: f32, ty: f32) {
        self.tlm = self.tlm.pre_translate(tx, ty);
        self.tm = self.tlm;
    }

    // MuPDF: pdf_tos_set_matrix (pdf-interpret.c:2078) -- `Tm`.
    pub fn set_matrix(&mut self, a: f32, b: f32, c: f32, d: f32, e: f32, f: f32) {
        self.tm = Matrix::new(a, b, c, d, e, f);
        self.tlm = self.tm;
    }

    // MuPDF: pdf_tos_newline (pdf-interpret.c:2090) -- `T*` (and the `'`/`"` /
    // `TD` newline).
    pub fn newline(&mut self, leading: f32) {
        self.tlm = self.tlm.pre_translate(0.0, -leading);
        self.tm = self.tlm;
    }
}

// MuPDF: pdf_tos_make_trm (pdf-interpret.c:2021) -- horizontal (wmode 0) and
// vertical (wmode 1) branches.
/// Compute a glyph's text-space text-rendering matrix and advances.
///
/// Returns `(trm, adv_em, char_tx, char_ty)` where:
/// * `trm = [size·scale, 0, 0, size, 0, rise] · Tm` (the *text-space* matrix;
///   the caller post-multiplies by the CTM to get device space);
/// * `adv_em = width / 1000` -- the nominal advance in em units (MuPDF's `w0`);
/// * `(char_tx, char_ty)` -- how far to advance `Tm` after this glyph.
///
/// `width` is the glyph's advance in 1/1000 em (from `Font::decode`).
pub(crate) fn make_trm(text: &TextState, wmode: i32, width: f32, tm: Matrix) -> (Matrix, f32, f32, f32) {
    // tsm = [ size*scale, 0, 0, size, 0, rise ].
    let mut tsm = Matrix::new(text.size * text.scale, 0.0, 0.0, text.size, 0.0, text.rise);

    let adv_em = width * 0.001;
    let (char_tx, char_ty);
    if wmode == 0 {
        // Horizontal: advance Tm along x by (w0*size + Tc) * Tz.
        char_tx = (adv_em * text.size + text.char_space) * text.scale;
        char_ty = 0.0;
    } else {
        // Vertical: MuPDF also shifts the glyph origin by the vertical-metrics
        // (v.x, v.y); those come from /W2 / /DW2 which this port defers (Font
        // builds horizontal hmtx only), so the origin shift is 0 here.
        let _ = &mut tsm;
        char_tx = 0.0;
        char_ty = adv_em * text.size + text.char_space;
    }

    let trm = tsm.concat(tm);
    (trm, adv_em, char_tx, char_ty)
}

/// The content-stream interpreter: runs operators over a [`TextDevice`].
///
/// Holds the graphics-state stack, the text object state, the resource-dict
/// stack, a loaded-font cache, and the XObject cycle guard. Driven by
/// [`Processor::run_stream`]; the public entry points are in [`super::page_run`]
/// ([`super::run_page`]).
pub struct Processor<'a, D: TextDevice + ?Sized> {
    /// The document (for resolving resources, fonts, XObject streams).
    pub(crate) doc: &'a PdfDocument,
    /// The glyph sink.
    pub(crate) dev: &'a mut D,
    /// The graphics-state stack; the last element is the top (`gtop`). Never
    /// empty.
    pub(crate) gstack: Vec<GState>,
    /// The text object state (`Tm`/`Tlm`), not stacked by `q`/`Q`.
    pub(crate) tos: Tos,
    /// The resource-dictionary stack (innermost last); see [`super::resources`].
    pub(crate) resources: Vec<Object>,
    /// Loaded-font cache, keyed by the font dict's object number (0 for a direct
    /// dict, which then loads afresh each time).
    pub(crate) fonts: HashMap<i32, Font>,
    /// Object numbers of Form XObjects currently being run -- the recursion guard
    /// (`pdf_cycle` in pdf-xobject.c). Depth-bounded as a belt-and-braces guard.
    pub(crate) cycle: Vec<i32>,
    /// The path being constructed by `m`/`l`/`c`/`v`/`y`/`re`/`h`, painted (and
    /// cleared) by `f`/`S`/`B`/`n`/… (MuPDF's `csi->path`).
    pub(crate) path: Path,
    /// The current point (path space), for the `v`/`y` bezier shorthands and `h`.
    pub(crate) cur: Point,
    /// The start of the current sub-path, restored as the current point by `h`.
    pub(crate) subpath_start: Point,
    /// A pending clip requested by `W`/`W*`; applied (with its winding rule) by the
    /// next path-painting operator (MuPDF's `csi->clip` / `clip_even_odd`).
    pub(crate) pending_clip: Option<FillRule>,
}

impl<'a, D: TextDevice + ?Sized> Processor<'a, D> {
    // MuPDF: pdf_new_run_processor + pdf_init_gstate (pdf-op-run.c).
    /// Create an interpreter over `doc`, emitting to `dev`, with the base CTM
    /// `ctm` (the page transform) and the given root resource dict.
    pub(crate) fn new(doc: &'a PdfDocument, dev: &'a mut D, ctm: Matrix, resources: Object) -> Processor<'a, D> {
        Processor {
            doc,
            dev,
            gstack: vec![GState::new(ctm)],
            tos: Tos::new(),
            resources: vec![resources],
            fonts: HashMap::new(),
            cycle: Vec::new(),
            path: Path::new(),
            cur: Point::new(0.0, 0.0),
            subpath_start: Point::new(0.0, 0.0),
            pending_clip: None,
        }
    }

    /// The top of the graphics-state stack (`gstate + gtop`).
    pub(crate) fn gstate(&self) -> &GState {
        self.gstack.last().expect("gstate stack never empty")
    }

    /// The top of the graphics-state stack, mutably.
    pub(crate) fn gstate_mut(&mut self) -> &mut GState {
        self.gstack.last_mut().expect("gstate stack never empty")
    }

    // MuPDF: pdf_process_stream (pdf-interpret.c:1527) -- the tokenizer loop.
    /// Run a decoded content byte stream, dispatching each operator. Reuses the
    /// PDF [`lex`](super::lex) tokenizer; operands accumulate on a numeric stack
    /// plus name/string/object slots until an operator keyword fires.
    pub(crate) fn run_stream(&mut self, content: &[u8]) -> Result<()> {
        let mut stm = Stream::from_slice(content);

        // Operand slots (MuPDF's csi->stack / name / string / obj).
        let mut stack: Vec<f64> = Vec::with_capacity(8);
        let mut name: Option<Vec<u8>> = None;
        let mut string: Option<Vec<u8>> = None;
        let mut obj: Option<Object> = None;

        loop {
            let tok = lex(&mut stm)?;
            match tok {
                Token::Eof | Token::EndStream => break,
                Token::Int(i) => stack.push(i as f64),
                Token::Real(r) => stack.push(r),
                Token::String(bytes) => string = Some(bytes),
                Token::Name(bytes) => {
                    // MuPDF keeps the first name in csi->name; a later name goes
                    // to csi->obj. The text path only ever needs the first
                    // (resource) name, so keep the first and ignore extras.
                    if name.is_none() {
                        name = Some(bytes);
                    } else {
                        obj = Some(Object::new_name(bytes));
                    }
                }
                // `[ … ]`: a general array (TJ show array, or a `d` dash array we
                // ignore). parse_array handles the mixed strings/numbers.
                Token::OpenArray => obj = Some(parse_array(&mut stm)?),
                // `<< … >>`: an inline dict operand (BDC/DP properties, `gs`
                // inline); parsed to keep the stream in sync, otherwise ignored.
                Token::OpenDict => obj = Some(parse_dict(&mut stm)?),
                Token::Keyword(word) => {
                    self.process_keyword(&word, &stack, &name, &string, &obj, &mut stm)?;
                    stack.clear();
                    name = None;
                    string = None;
                    obj = None;
                }
                // A stray delimiter / error token: clear operands (MuPDF's clear
                // on syntax error) and continue rather than aborting the page.
                Token::Error => {}
                // Bare booleans / null as operands are rare on the text path;
                // ignore them (they carry no glyph information).
                _ => {}
            }
        }
        Ok(())
    }

    // MuPDF: pdf_process_keyword (pdf-interpret.c:1309) -- operator dispatch.
    /// Dispatch one operator keyword `word` with the accumulated operands. Only
    /// the text-path operators do anything; the rest are recognised and skipped
    /// (their operands are already collected, keeping the stream in sync).
    #[allow(clippy::too_many_arguments)]
    fn process_keyword(
        &mut self,
        word: &[u8],
        stack: &[f64],
        name: &Option<Vec<u8>>,
        string: &Option<Vec<u8>>,
        obj: &Option<Object>,
        stm: &mut Stream,
    ) -> Result<()> {
        // s(i): the i-th numeric operand, 0.0 if absent (MuPDF reads csi->stack
        // slots that default to 0 after pdf_clear_stack).
        let s = |i: usize| -> f32 { stack.get(i).copied().unwrap_or(0.0) as f32 };

        match word {
            // -- special graphics state ------------------------------------
            b"q" => self.op_q(),
            b"Q" => self.op_q_restore(),
            b"cm" => self.op_cm(s(0), s(1), s(2), s(3), s(4), s(5)),

            // -- text objects ----------------------------------------------
            b"BT" => self.op_bt(),
            b"ET" => { /* pdf_flush_text: nothing buffered in this port */ }

            // -- text state ------------------------------------------------
            b"Tc" => self.gstate_mut().text.char_space = s(0),
            b"Tw" => self.gstate_mut().text.word_space = s(0),
            b"Tz" => self.gstate_mut().text.scale = s(0) / 100.0,
            b"TL" => self.gstate_mut().text.leading = s(0),
            b"Tr" => self.gstate_mut().text.render = s(0) as i32,
            b"Ts" => self.gstate_mut().text.rise = s(0),
            b"Tf" => self.op_tf(name.as_deref(), s(0))?,

            // -- text positioning ------------------------------------------
            b"Td" => self.tos.translate(s(0), s(1)),
            b"TD" => {
                // TD also sets leading to -ty (pdf_run_TD, pdf-op-run.c:3007).
                self.gstate_mut().text.leading = -s(1);
                self.tos.translate(s(0), s(1));
            }
            b"Tm" => self.tos.set_matrix(s(0), s(1), s(2), s(3), s(4), s(5)),
            b"T*" => {
                let leading = self.gstate().text.leading;
                self.tos.newline(leading);
            }

            // -- text showing ----------------------------------------------
            b"Tj" => {
                if let Some(str_bytes) = string {
                    self.show_string(str_bytes);
                }
            }
            b"TJ" => {
                if let Some(arr) = obj {
                    self.show_text_array(arr);
                }
            }
            b"'" => {
                // Newline, then show (pdf_run_squote, pdf-op-run.c:3042).
                let leading = self.gstate().text.leading;
                self.tos.newline(leading);
                if let Some(str_bytes) = string {
                    self.show_string(str_bytes);
                }
            }
            b"\"" => {
                // aw ac string " : set word/char spacing, newline, show.
                {
                    let g = self.gstate_mut();
                    g.text.word_space = s(0);
                    g.text.char_space = s(1);
                }
                let leading = self.gstate().text.leading;
                self.tos.newline(leading);
                if let Some(str_bytes) = string {
                    self.show_string(str_bytes);
                }
            }

            // -- path construction -----------------------------------------
            b"m" => self.op_moveto(s(0), s(1)),
            b"l" => self.op_lineto(s(0), s(1)),
            b"c" => self.op_curveto(s(0), s(1), s(2), s(3), s(4), s(5)),
            b"v" => self.op_curveto_v(s(0), s(1), s(2), s(3)),
            b"y" => self.op_curveto_y(s(0), s(1), s(2), s(3)),
            b"re" => self.op_re(s(0), s(1), s(2), s(3)),
            b"h" => self.op_closepath(),

            // -- path painting (fill / stroke / clip / end) ----------------
            b"S" => self.op_paint(false, false, FillRule::NonZero, true),
            b"s" => self.op_paint(true, false, FillRule::NonZero, true),
            b"f" | b"F" => self.op_paint(false, true, FillRule::NonZero, false),
            b"f*" => self.op_paint(false, true, FillRule::EvenOdd, false),
            b"B" => self.op_paint(false, true, FillRule::NonZero, true),
            b"B*" => self.op_paint(false, true, FillRule::EvenOdd, true),
            b"b" => self.op_paint(true, true, FillRule::NonZero, true),
            b"b*" => self.op_paint(true, true, FillRule::EvenOdd, true),
            b"n" => self.op_paint(false, false, FillRule::NonZero, false),

            // -- clipping (the pending clip is applied by the next paint) ---
            b"W" => self.pending_clip = Some(FillRule::NonZero),
            b"W*" => self.pending_clip = Some(FillRule::EvenOdd),

            // -- stroke state ----------------------------------------------
            b"w" => self.gstate_mut().line_width = s(0),

            // -- colour ----------------------------------------------------
            b"g" => self.op_set_gray(s(0), true),
            b"G" => self.op_set_gray(s(0), false),
            b"rg" => self.op_set_rgb(s(0), s(1), s(2), true),
            b"RG" => self.op_set_rgb(s(0), s(1), s(2), false),
            b"k" => self.op_set_cmyk(s(0), s(1), s(2), s(3), true),
            b"K" => self.op_set_cmyk(s(0), s(1), s(2), s(3), false),
            b"cs" => self.op_set_colorspace(name.as_deref(), true),
            b"CS" => self.op_set_colorspace(name.as_deref(), false),
            b"sc" | b"scn" => self.op_set_color(stack, name.is_some(), true),
            b"SC" | b"SCN" => self.op_set_color(stack, name.is_some(), false),

            // -- XObjects (Form recursion + Image painting) ----------------
            b"Do" => self.op_do(name.as_deref())?,

            // -- inline images: skip the BI … ID <binary> EI span ----------
            b"BI" => self.skip_inline_image(stm)?,

            // Everything else (paths, colours, clips, shadings, gs, marked
            // content, Type3 metrics, BX/EX) is parsed-and-ignored: the operands
            // were already consumed, so the stream stays in sync.
            _ => {}
        }
        Ok(())
    }

    // MuPDF: pdf_run_BT (pdf-op-run.c:2929) -- reset Tm and Tlm to identity.
    fn op_bt(&mut self) {
        self.tos.tm = Matrix::IDENTITY;
        self.tos.tlm = Matrix::IDENTITY;
    }

    // MuPDF: the BI branch of pdf_process_keyword + parse_inline_image
    // (pdf-interpret.c:1478, 805). Inline-image *decoding* is deferred; this only
    // resyncs past the image so the rest of the content stream still runs.
    /// Skip an inline image: consume tokens up to the `ID` keyword, then scan the
    /// raw bytes for the terminating `EI` (bounded by whitespace/EOF). Best
    /// effort -- inline images are off the text path.
    fn skip_inline_image(&mut self, stm: &mut Stream) -> Result<()> {
        // Consume the image dictionary tokens until the `ID` keyword.
        loop {
            match lex(stm)? {
                Token::Eof => return Ok(()),
                Token::Keyword(k) if k == b"ID" => break,
                _ => {}
            }
        }
        // One whitespace byte follows `ID`; then binary data until `EI`.
        let _ = stm.read_byte()?;
        let mut prev_ws = true; // the byte after ID counts as the leading boundary
        loop {
            let c = match stm.read_byte()? {
                None => return Ok(()),
                Some(c) => c,
            };
            if prev_ws && c == b'E' && stm.peek_byte()? == Some(b'I') {
                let _ = stm.read_byte()?; // consume 'I'
                // `EI` must be followed by whitespace/delimiter/EOF.
                match stm.peek_byte()? {
                    None => return Ok(()),
                    Some(n) if n <= b' ' => return Ok(()),
                    // False alarm: keep scanning.
                    Some(_) => {
                        prev_ws = false;
                        continue;
                    }
                }
            }
            prev_ws = c <= b' ';
        }
    }
}

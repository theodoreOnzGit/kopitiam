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
//! scn SCN`), the line state (`w J j M d`, since 0.4.2), `gs`, inline images and
//! image/form XObjects (`Do`) now drive the
//! [`DrawDevice`](super::draw_device) via the [`TextDevice`] sink's path/colour/
//! image callbacks. Colour is tracked in the graphics state and converted to
//! DeviceRGB; see [`super::op_run`] and [`super::resources`].
//!
//! ## Still deferred (parsed-and-ignored so the stream still runs)
//!
//! Type3 glyph metrics (`d0 d1`), shadings (`sh`), marked content (`MP DP BMC
//! BDC EMC`) and the compatibility bracket (`BX EX`) are recognised and
//! skipped; their operands are consumed so the operator stream stays in sync.
//! ~~the ExtGState body (`gs`)~~ and ~~inline images skipped as a raw byte
//! span~~ -- **CORRECTED 2026-09-28 (0.4.2)**: `gs` applies its line-state,
//! font and alpha keys ([`Processor::op_gs`]), and `BI … ID … EI` is decoded
//! and painted ([`Processor::op_inline_image`]). Render modes 3 and 7
//! ("invisible") still emit glyphs: extraction needs them.

use std::collections::HashMap;

use super::draw_edge::FillRule;
use super::draw_path::Path;
use super::error::Result;
use super::font::Font;
use super::geometry::{Matrix, Point, Rect};
use super::lex::{Token, lex};
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
    /// The rest of the stroke state (`J`/`j`/`M`/`d` and their ExtGState
    /// keys); `line_width` above stays the source of truth for the width.
    // MuPDF: pdf_gstate.stroke_state (fz_stroke_state).
    pub stroke_style: super::draw_path::StrokeStyle,
    /// The fill alpha (`/ca` via `gs`), 0..=1. Default 1.
    // MuPDF: pdf_gstate.fill.alpha (pdf_run_gs_ca).
    pub fill_alpha: f32,
    /// The stroke alpha (`/CA` via `gs`), 0..=1. Default 1.
    // MuPDF: pdf_gstate.stroke.alpha (pdf_run_gs_CA).
    pub stroke_alpha: f32,
    /// The fill material when it is a pattern (MuPDF `fill.kind ==
    /// PDF_MAT_SHADE / PDF_MAT_PATTERN`), with the gstate index whose CTM is
    /// the pattern space (`fill.gstate_num = pr->gparent`).
    pub fill_pattern: Option<(PatternFill, usize)>,
    /// The stroke material when it is a pattern.
    pub stroke_pattern: Option<(PatternFill, usize)>,
    /// How many device clips are pushed in total at this state -- MuPDF's
    /// cumulative `pdf_gstate.clip_depth`: `q` copies it, `Q` pops the
    /// difference between the popped and the restored state.
    pub clip_depth: u32,
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
            stroke_style: super::draw_path::StrokeStyle::default(),
            fill_alpha: 1.0,
            stroke_alpha: 1.0,
            clip_depth: 0,
            fill_pattern: None,
            stroke_pattern: None,
            clip: None,
        }
    }
}

/// A pattern material (`pdf_material` with a pattern or shade).
#[derive(Clone, Debug)]
pub(crate) enum PatternFill {
    /// A shading pattern (`/PatternType 2`), painted with fz_fill_shade
    /// through a clip of the shape.
    Shade(std::sync::Arc<super::shade::Shade>),
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
        Tos {
            tm: Matrix::IDENTITY,
            tlm: Matrix::IDENTITY,
            char_tx: 0.0,
            char_ty: 0.0,
        }
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
pub(crate) fn make_trm(
    text: &TextState,
    wmode: i32,
    width: f32,
    tm: Matrix,
) -> (Matrix, f32, f32, f32) {
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
    /// The lowest `gstack` index `Q` may pop to (MuPDF's `pr->gbot`): a Form
    /// XObject raises it so a stray `Q` inside the form cannot pop the
    /// caller's state.
    pub(crate) gbot: usize,
    /// Glyphs shown in a clipping render mode (4..=7) since the last `ET`
    /// (MuPDF's `pdf_tos.clip_text`), turned into a clip at `ET`.
    pub(crate) text_clip: Vec<super::text_device::ClipGlyph>,
    /// MuPDF's `pr->gparent`: the gstate whose CTM is the pattern space for
    /// patterns selected now (the page's base state, or the state a Form
    /// XObject was invoked from, its CTM temporarily set to the form's).
    pub(crate) gparent: usize,
    /// MuPDF's `proc->hidden`: > 0 while inside optional content that the
    /// default layer configuration hides (paths, text, images and shadings
    /// then paint nothing; clips still apply).
    pub(crate) hidden: u32,
    /// The document's default layer configuration, read on first use.
    pub(crate) ocg: Option<std::sync::Arc<super::layer::OcgConfig>>,
    /// Inside a Type3 glyph procedure that declared `d1` (an uncoloured,
    /// "mask" glyph): colour operators are ignored and the glyph paints in
    /// the text's fill colour (MuPDF's FZ_DEVFLAG_MASK, pdf-op-run.c:3068).
    pub(crate) t3_mask: bool,
    /// Type3 glyph-procedure nesting depth ("recursive type3 font" guard).
    pub(crate) t3_depth: u32,
}

impl<'a, D: TextDevice + ?Sized> Processor<'a, D> {
    // MuPDF: pdf_new_run_processor + pdf_init_gstate (pdf-op-run.c).
    /// Create an interpreter over `doc`, emitting to `dev`, with the base CTM
    /// `ctm` (the page transform) and the given root resource dict.
    pub(crate) fn new(
        doc: &'a PdfDocument,
        dev: &'a mut D,
        ctm: Matrix,
        resources: Object,
    ) -> Processor<'a, D> {
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
            gbot: 0,
            text_clip: Vec::new(),
            gparent: 0,
            hidden: 0,
            ocg: None,
            t3_mask: false,
            t3_depth: 0,
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
            // MuPDF pdf_run_ET: pdf_flush_text + pdf_flush_clip_text -- the
            // glyphs shown in modes 4..=7 become one clip, counted in the
            // gstate's clip_depth so the enclosing Q pops it.
            b"ET" => self.flush_clip_text(),

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
            // MuPDF: pdf_run_J / pdf_run_j / pdf_run_M / pdf_run_d
            // (pdf-op-run.c:2607-2651). pdf-interpret.c reads J/j as ints.
            b"J" => self.gstate_mut().stroke_style.cap = line_cap(s(0) as i32),
            b"j" => self.gstate_mut().stroke_style.join = line_join(s(0) as i32),
            b"M" => self.gstate_mut().stroke_style.miter_limit = s(0),
            b"d" => {
                if let Some(arr) = obj {
                    self.op_d(arr, s(0));
                }
            }
            // -- ExtGState (pdf_process_extgstate, pdf-interpret.c:875) ----
            b"gs" => {
                if let Some(n) = name {
                    self.op_gs(n)?;
                }
            }

            // -- colour ----------------------------------------------------
            // (all ignored inside a `d1` Type3 glyph: FZ_DEVFLAG_MASK)
            b"g" | b"G" | b"rg" | b"RG" | b"k" | b"K" | b"cs" | b"CS" | b"sc" | b"scn" | b"SC" | b"SCN"
                if self.t3_mask => {}
            b"g" => self.op_set_gray(s(0), true),
            b"G" => self.op_set_gray(s(0), false),
            b"rg" => self.op_set_rgb(s(0), s(1), s(2), true),
            b"RG" => self.op_set_rgb(s(0), s(1), s(2), false),
            b"k" => self.op_set_cmyk(s(0), s(1), s(2), s(3), true),
            b"K" => self.op_set_cmyk(s(0), s(1), s(2), s(3), false),
            b"cs" => self.op_set_colorspace(name.as_deref(), true),
            b"CS" => self.op_set_colorspace(name.as_deref(), false),
            b"sc" | b"scn" => self.op_set_color_named(stack, name.as_deref(), true),
            b"SC" | b"SCN" => self.op_set_color_named(stack, name.as_deref(), false),

            // -- marked content: optional content (pdf_process_BDC/BMC/EMC,
            //    pdf-interpret.c:1227-1263) -------------------------------
            b"BDC" => {
                if self.hidden > 0 {
                    self.hidden += 1;
                } else if name.as_deref() == Some(b"OC".as_slice())
                    && let Some(props) = obj
                    && self.ocg_hidden(props)
                {
                    self.hidden += 1;
                }
            }
            b"BMC" => {
                if self.hidden > 0 {
                    self.hidden += 1;
                }
            }
            b"EMC" => {
                if self.hidden > 0 {
                    self.hidden -= 1;
                }
            }

            // -- Type3 glyph metrics (pdf_run_d0 / pdf_run_d1) --------------
            b"d0" => self.t3_mask = false,
            b"d1" => self.t3_mask = true,

            // -- shadings (pdf_run_sh -> pdf_show_shade) ------------------
            b"sh" => {
                if let Some(n) = name {
                    self.op_sh(n);
                }
            }

            // -- XObjects (Form recursion + Image painting) ----------------
            b"Do" => self.op_do(name.as_deref())?,

            // -- inline images: BI <dict> ID <data> EI ---------------------
            b"BI" => self.op_inline_image(stm)?,

            // Everything else (paths, colours, clips, shadings, gs, marked
            // content, Type3 metrics, BX/EX) is parsed-and-ignored: the operands
            // were already consumed, so the stream stays in sync.
            _ => {}
        }
        Ok(())
    }

    // MuPDF: pdf_run_BT (pdf-op-run.c:2929) -- reset Tm and Tlm to identity.
    // MuPDF: pdf_flush_clip_text -> pdf_flush_text_imp(flush_clip = 1)'s
    // `doclip` branch (pdf-op-run.c:1266).
    // MuPDF: pdf_is_ocg_hidden(doc, rstack, "View", obj) (pdf-layer.c:791).
    /// Whether optional content `obj` (a `/Properties` name, an OCG/OCMD dict
    /// or a reference to one) is hidden under the default configuration.
    pub(crate) fn ocg_hidden(&mut self, obj: &Object) -> bool {
        let cfg = self
            .ocg
            .get_or_insert_with(|| std::sync::Arc::new(super::layer::OcgConfig::load(self.doc)))
            .clone();
        let lookup = |n: &[u8]| self.lookup_resource_raw("Properties", n);
        cfg.is_hidden(self.doc, obj, &lookup)
    }

    pub(crate) fn flush_clip_text(&mut self) {
        if self.text_clip.is_empty() {
            return;
        }
        let glyphs = std::mem::take(&mut self.text_clip);
        self.dev.clip_text(&glyphs);
        self.gstate_mut().clip_depth += 1;
    }

    /// Pop every graphics state and every clip still open at the end of a
    /// content stream (MuPDF's pdf_close_run_processor).
    pub(crate) fn finish(&mut self) {
        self.flush_clip_text();
        self.gbot = 0;
        while self.gstack.len() > 1 {
            self.op_q_restore();
        }
        let depth = self.gstate().clip_depth;
        for _ in 0..depth {
            self.dev.pop_clip();
        }
        self.gstate_mut().clip_depth = 0;
    }

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
    // MuPDF: parse_inline_image (pdf-interpret.c:805) + pdf_load_inline_image
    // (pdf-image.c:228) + pdf_show_image (pdf-op-run.c:860).
    /// `BI … ID <data> EI`: parse the abbreviated parameter dict, find where the
    /// data ends, decode it like an image XObject and paint it.
    ///
    /// Finding the end is the subtle part. MuPDF runs the decoder over the
    /// content stream and lets it consume exactly what it needs, then scans for
    /// `EI` followed by white space, `<` or `/`. Our filter layer decodes whole
    /// buffers, so: unfiltered data is cut at exactly `stride x H` bytes (it may
    /// legally contain the bytes "EI"); filtered data is tried at each
    /// candidate `EI` in turn, the first that decodes to a full image winning.
    /// If nothing decodes, the image is skipped and the stream resyncs after the
    /// first candidate (the pre-0.4.2 behaviour for every inline image).
    fn op_inline_image(&mut self, stm: &mut Stream) -> Result<()> {
        // The dict: `/Key value` pairs up to the ID keyword.
        let mut dict = Object::new_dict();
        loop {
            match lex(stm)? {
                Token::Eof => return Ok(()),
                Token::Keyword(k) if k == b"ID" => break,
                Token::Name(key) => {
                    let val = match lex(stm)? {
                        Token::Name(n) => Object::new_name(n),
                        Token::Int(i) => Object::new_int(i),
                        Token::Real(r) => Object::new_real(r),
                        Token::True => Object::Bool(true),
                        Token::False => Object::Bool(false),
                        Token::String(b) => Object::new_string(b),
                        Token::OpenArray => parse_array(stm)?,
                        Token::OpenDict => parse_dict(stm)?,
                        Token::Keyword(k) if k == b"ID" => break,
                        _ => Object::Null,
                    };
                    dict.dict_put(key, val);
                }
                _ => {}
            }
        }
        // "read whitespace after ID keyword" (CR LF counts as one).
        if stm.read_byte()? == Some(b'\r') && stm.peek_byte()? == Some(b'\n') {
            let _ = stm.read_byte()?;
        }
        let start = stm.tell();
        let mut rest = Vec::new();
        let mut buf = [0u8; 4096];
        loop {
            let n = stm.read(&mut buf)?;
            if n == 0 {
                break;
            }
            rest.extend_from_slice(&buf[..n]);
        }

        let dict = self.expand_inline_image_dict(dict);
        let filtered = !matches!(
            dict.dict_gets("Filter").or_else(|| dict.dict_gets("F")),
            None | Some(Object::Null)
        );
        let is_mask = super::page_image::is_image_mask(self.doc, &dict);

        let mut decoded = None;
        let resume;
        if !filtered {
            let need = inline_raw_len(self.doc, &dict).min(rest.len());
            decoded = super::page_image::decode_inline_image(self.doc, &dict, &rest[..need]).ok();
            resume = find_ei(&rest, need).map_or(rest.len(), |p| p + 2);
        } else {
            let mut first = None;
            let mut hit = None;
            let mut from = 0;
            while let Some(p) = find_ei(&rest, from) {
                first.get_or_insert(p);
                if let Ok(img) = super::page_image::decode_inline_image(self.doc, &dict, &rest[..p]) {
                    hit = Some((img, p));
                    break;
                }
                from = p + 1;
            }
            let end = match hit {
                Some((img, p)) => {
                    decoded = Some(img);
                    p
                }
                None => first.unwrap_or(rest.len().saturating_sub(2)),
            };
            resume = end + 2;
        }
        stm.seek(start + resume.min(rest.len()) as i64, super::stream::Whence::Set)?;
        if let Some(img) = decoded {
            self.show_image(&img, is_mask);
        }
        Ok(())
    }

    // MuPDF: the inline-image colour space rules of pdf_load_image_imp +
    // pdf_load_colorspace's abbreviation table (/G /RGB /CMYK /I), with any
    // other name looked up in the /ColorSpace resources.
    fn expand_inline_image_dict(&self, mut dict: Object) -> Object {
        let key = if dict.dict_gets("ColorSpace").is_some() { "ColorSpace" } else { "CS" };
        if let Some(cs) = dict.dict_gets(key).cloned() {
            let expanded = self.expand_inline_cs(&cs);
            dict.dict_put(key.as_bytes().to_vec(), expanded);
        }
        dict
    }

    fn expand_inline_cs(&self, cs: &Object) -> Object {
        match cs {
            Object::Name(n) => match n.as_slice() {
                b"G" => Object::new_name(b"DeviceGray".to_vec()),
                b"RGB" => Object::new_name(b"DeviceRGB".to_vec()),
                b"CMYK" => Object::new_name(b"DeviceCMYK".to_vec()),
                b"I" => Object::new_name(b"Indexed".to_vec()),
                b"DeviceGray" | b"DeviceRGB" | b"DeviceCMYK" | b"Indexed" | b"Pattern" => cs.clone(),
                other => {
                    let r = self.lookup_resource("ColorSpace", other);
                    if r.is_null() { cs.clone() } else { r }
                }
            },
            Object::Array(items) => {
                let mut out = Object::new_array();
                for (i, it) in items.iter().enumerate() {
                    // [/I base hival lookup]: expand the family and the base.
                    out.array_push(if i <= 1 { self.expand_inline_cs(it) } else { it.clone() });
                }
                out
            }
            other => other.clone(),
        }
    }

    #[allow(dead_code)]
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

/// `J` / `/LC` value -> cap. MuPDF: pdf-interpret.c reads the operand as an
/// int; `/LC` is clamped to 0..=2 (fz_clampi) and so is anything PDF can say.
pub(crate) fn line_cap(v: i32) -> super::draw_path::LineCap {
    use super::draw_path::LineCap;
    match v.clamp(0, 2) {
        1 => LineCap::Round,
        2 => LineCap::Square,
        _ => LineCap::Butt,
    }
}

/// `j` / `/LJ` value -> join (clamped to 0..=2 like `/LJ`).
pub(crate) fn line_join(v: i32) -> super::draw_path::LineJoin {
    use super::draw_path::LineJoin;
    match v.clamp(0, 2) {
        1 => LineJoin::Round,
        2 => LineJoin::Bevel,
        _ => LineJoin::Miter,
    }
}

/// Byte length of UNFILTERED inline image data: `ceil(W x n x BPC / 8) x H`
/// (an `/IM` stencil is 1 x 1 bit; an Indexed space has one component).
// MuPDF: pdf_load_image_imp's `stride * h` for an inline image with no filter
// (pdf-image.c:117, fz_open_null on `len` bytes).
fn inline_raw_len(doc: &PdfDocument, dict: &Object) -> usize {
    let get = |a: &str, b: &str| {
        dict.dict_gets(a)
            .or_else(|| dict.dict_gets(b))
            .map(|o| doc.resolve(o).unwrap_or(Object::Null))
            .unwrap_or(Object::Null)
    };
    let w = get("Width", "W").to_int().max(0) as usize;
    let h = get("Height", "H").to_int().max(0) as usize;
    let mask = get("ImageMask", "IM").to_bool();
    let mut bpc = get("BitsPerComponent", "BPC").to_int().max(0) as usize;
    if mask {
        bpc = 1;
    }
    if bpc == 0 {
        bpc = 8;
    }
    let n = if mask {
        1
    } else {
        match get("ColorSpace", "CS") {
            Object::Name(n) => match n.as_slice() {
                b"DeviceRGB" | b"CalRGB" | b"Lab" => 3,
                b"DeviceCMYK" => 4,
                _ => 1,
            },
            Object::Array(items) => match items.first().map(|o| o.to_name().to_vec()) {
                Some(f) if f == b"Indexed" => 1,
                Some(f) if f == b"ICCBased" => items
                    .get(1)
                    .and_then(|o| doc.resolve(o).ok())
                    .and_then(|d| d.dict_gets("N").map(|n| n.to_int() as usize))
                    .unwrap_or(1),
                Some(f) if f == b"CalRGB" || f == b"Lab" => 3,
                Some(f) if f == b"DeviceN" => items.get(1).map_or(1, |a| a.array_len().max(1)),
                _ => 1,
            },
            _ => 1,
        }
    };
    (w * n * bpc).div_ceil(8) * h
}

/// The first `EI` at or after `from` that MuPDF's scan would accept: `E`,
/// `I`, then a byte that is white space / control (<= 32), `<` or `/`, or the
/// end of the stream. Returns the offset of the `E`.
// MuPDF: parse_inline_image's "find EI" loop (pdf-interpret.c:838-857).
fn find_ei(data: &[u8], from: usize) -> Option<usize> {
    let mut i = from;
    while i + 1 < data.len() {
        if data[i] == b'E' && data[i + 1] == b'I' {
            match data.get(i + 2) {
                None => return Some(i),
                Some(&c) if c <= 32 || c == b'<' || c == b'/' => return Some(i),
                _ => {}
            }
        }
        i += 1;
    }
    None
}

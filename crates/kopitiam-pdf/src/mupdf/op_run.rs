//! Ported from MuPDF `source/pdf/pdf-op-run.c` -- the text-showing subset of the
//! run processor: the graphics-state stack (`pdf_run_q`/`pdf_run_Q`/`pdf_run_cm`
//! with `pdf_gsave`/`pdf_grestore`) and the show path
//! (`pdf_run_Tj`/`pdf_run_TJ` -> `pdf_show_text` -> `show_string` ->
//! `pdf_show_char`/`pdf_show_space`) (commit 19f1284, AGPL-3.0, © Artifex
//! Software, Inc.), translated to Rust for KOPITIAM (AGPL-3.0-only). Close
//! adaptation: the algorithms and numeric behaviour follow MuPDF; the code is
//! re-expressed in idiomatic Rust. See docs/ACKNOWLEDGEMENTS.md ("PDF &
//! document-extraction references").
//!
//! # The show path
//!
//! For each PDF string shown, [`Processor::show_string`] splits it into character
//! codes through the current font's encoding CMap ([`Font::next_code`]), maps each
//! to `(unicode, advance, cid)` ([`Font::decode`]), and calls
//! [`Processor::show_char`]. `show_char` composes the glyph's device-space
//! text-rendering matrix (`make_trm` result · CTM), emits it via
//! [`TextDevice::show_glyph`], and advances the text matrix `Tm` by the glyph's
//! step. A single-byte space code (`cpt == 32 && width == 1`) additionally
//! advances by the word spacing `Tw`; `TJ` numeric elements advance by
//! `-n/1000 · size · Tz`.
//!
//! The graphics state slice ([`GState`](super::interpret)) -- CTM plus text state
//! -- is pushed/popped by `q`/`Q`; `cm` pre-concatenates onto the CTM. The text
//! object state (`Tm`/`Tlm`) is separate and survives `q`/`Q` (MuPDF keeps it on
//! the processor, reset only by `BT`).
//!
//! ## Deferred
//!
//! Text render modes are honoured only for *emission* (modes 3 and 7 stay
//! visible to extraction, as the PDF spec's invisible modes still carry text);
//! actual stroking/filling/clipping, Type3 glyph execution, the glyph cache and
//! bounding-box accumulation are not ported (no rasterisation on the text path).

use super::draw_device::{cmyk_to_rgb, gray_to_rgb};
use super::draw_edge::FillRule;
use super::draw_path::Path;
use super::font::Font;
use super::geometry::{Matrix, Point, Rect};
use super::interpret::{make_trm, GState, Processor};
use super::object::Object;
use super::resources::ColorSpace;
use super::text_device::TextDevice;

impl<D: TextDevice + ?Sized> Processor<'_, D> {
    // -----------------------------------------------------------------------
    // Graphics-state stack (pdf-op-run.c)
    // -----------------------------------------------------------------------

    // MuPDF: pdf_run_q + pdf_gsave (pdf-op-run.c:2765) -- push a copy of the top.
    pub(crate) fn op_q(&mut self) {
        let top = self.gstate().clone();
        self.gstack.push(top);
    }

    // MuPDF: pdf_run_Q + pdf_grestore (pdf-op-run.c:2772) -- pop, never below the
    // base gstate (MuPDF's pdf_grestore clamps gtop at gbot / 0).
    pub(crate) fn op_q_restore(&mut self) {
        if self.gstack.len() > 1 {
            self.gstack.pop();
        }
    }

    // MuPDF: pdf_run_cm (pdf-op-run.c:2779) -- ctm = [a b c d e f] · ctm.
    pub(crate) fn op_cm(&mut self, a: f32, b: f32, c: f32, d: f32, e: f32, f: f32) {
        let m = super::geometry::Matrix::new(a, b, c, d, e, f);
        let g = self.gstate_mut();
        g.ctm = m.concat(g.ctm);
    }

    // -----------------------------------------------------------------------
    // Text showing (pdf-op-run.c)
    // -----------------------------------------------------------------------

    // MuPDF: pdf_show_text's array branch (pdf-op-run.c:1622) -- `TJ`.
    /// Show a `TJ` array: string elements are shown; numeric elements adjust the
    /// text position by `-n/1000 · size · Tz` (via [`Processor::show_space`]).
    pub(crate) fn show_text_array(&mut self, arr: &Object) {
        if self.gstate().text.font.is_none() {
            return; // "cannot draw text since font and size not set"
        }
        for i in 0..arr.array_len() {
            let Some(item) = arr.array_get(i) else { continue };
            match item {
                Object::String(bytes) => {
                    let bytes = bytes.clone();
                    self.show_string(&bytes);
                }
                Object::Int(_) | Object::Real(_) => {
                    // tadj = -n * size * 0.001; show_space scales by Tz.
                    let size = self.gstate().text.size;
                    let tadj = -(item.to_real() as f32) * size * 0.001;
                    self.show_space(tadj);
                }
                _ => {}
            }
        }
    }

    // MuPDF: show_string + pdf_show_string (pdf-op-run.c:1562, 1594) -- split the
    // string into codes and show each; a single-byte space also applies Tw.
    /// Show one PDF string: iterate its character codes, emitting a glyph for each
    /// and applying word spacing after a single-byte space code.
    pub(crate) fn show_string(&mut self, buf: &[u8]) {
        if self.gstate().text.font.is_none() {
            return;
        }
        let mut i = 0;
        while i < buf.len() {
            // Split off one code through the encoding CMap.
            let (code, w) = {
                let font = self.gstate().text.font.as_ref().unwrap();
                font.next_code(&buf[i..])
            };
            let w = w.max(1);
            i += w;

            // Decode -> (unicode, advance, cid) and emit the glyph.
            let dec = {
                let font = self.gstate().text.font.as_ref().unwrap();
                font.decode(code)
            };
            self.show_char(dec.unicode, dec.advance, dec.cid);

            // Bug 703151 parity: a single-byte space also advances by Tw.
            if code == 32 && w == 1 {
                let word_space = self.gstate().text.word_space;
                self.show_space(word_space);
            }
        }
    }

    // MuPDF: pdf_show_char (pdf-op-run.c:1330) -- the extraction slice: make the
    // trm, emit the glyph, advance Tm. (Type3 rendering, glyph cache, clip
    // accumulation and bbox tracking are dropped.)
    /// Emit one glyph for `cid`/`unicode` with nominal `width` (1/1000 em),
    /// then advance the text matrix.
    fn show_char(&mut self, unicode: char, width: f32, cid: u32) {
        let wmode = self.gstate().text.font.as_ref().unwrap().wmode();

        // Compute the text-space trm + advances from the current text state & Tm.
        let (trm_text, adv_em, char_tx, char_ty) = {
            let g: &GState = self.gstate();
            make_trm(&g.text, wmode, width, self.tos.tm)
        };
        // Device space: post-multiply by the CTM (what the stext device applies).
        let trm_dev = trm_text.concat(self.gstate().ctm);

        // Carry the current fill colour to the device so the placeholder glyph
        // boxes render in the text's colour (the sink ignores this by default).
        let fill_color = self.gstate().fill_color;
        self.dev.set_fill_color(fill_color);

        // Emit. Split-borrow via direct field access: `font` reads self.gstack,
        // `dev` is a disjoint field, so the borrow checker permits both (going
        // through the `gstate()` method would borrow all of `self`).
        {
            let font: &Font = self
                .gstack
                .last()
                .unwrap()
                .text
                .font
                .as_ref()
                .unwrap();
            self.dev
                .show_glyph(font, trm_dev, adv_em, unicode, cid, wmode as u8);
        }

        // MuPDF: pdf_tos_move_after_char (pdf-interpret.c:2062) -- advance Tm.
        self.tos.char_tx = char_tx;
        self.tos.char_ty = char_ty;
        self.tos.tm = self.tos.tm.pre_translate(char_tx, char_ty);
    }

    // MuPDF: pdf_show_space (pdf-op-run.c:1457) -- shift Tm by the adjustment.
    /// Advance the text matrix by `tadj` (word spacing, or a `TJ` position
    /// adjustment). Horizontal writing scales by `Tz`; vertical does not.
    fn show_space(&mut self, tadj: f32) {
        let wmode = self.gstate().text.font.as_ref().map(Font::wmode).unwrap_or(0);
        let scale = self.gstate().text.scale;
        if wmode == 0 {
            self.tos.tm = self.tos.tm.pre_translate(tadj * scale, 0.0);
        } else {
            self.tos.tm = self.tos.tm.pre_translate(0.0, tadj);
        }
    }

    // -----------------------------------------------------------------------
    // Path construction (pdf-op-run.c path operators -> fz_path builder)
    // -----------------------------------------------------------------------

    // MuPDF: pdf_run_m (pdf-op-run.c) -- fz_moveto: start a new sub-path.
    pub(crate) fn op_moveto(&mut self, x: f32, y: f32) {
        self.path.move_to(x, y);
        self.cur = Point::new(x, y);
        self.subpath_start = self.cur;
    }

    // MuPDF: pdf_run_l -- fz_lineto.
    pub(crate) fn op_lineto(&mut self, x: f32, y: f32) {
        self.path.line_to(x, y);
        self.cur = Point::new(x, y);
    }

    // MuPDF: pdf_run_c -- fz_curveto (cubic bezier with two explicit controls).
    pub(crate) fn op_curveto(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x3: f32, y3: f32) {
        self.path.curve_to(x1, y1, x2, y2, x3, y3);
        self.cur = Point::new(x3, y3);
    }

    // MuPDF: pdf_run_v -- fz_curvetov: the first control point is the current point.
    pub(crate) fn op_curveto_v(&mut self, x2: f32, y2: f32, x3: f32, y3: f32) {
        let c = self.cur;
        self.path.curve_to(c.x, c.y, x2, y2, x3, y3);
        self.cur = Point::new(x3, y3);
    }

    // MuPDF: pdf_run_y -- fz_curvetoy: the second control point is the endpoint.
    pub(crate) fn op_curveto_y(&mut self, x1: f32, y1: f32, x3: f32, y3: f32) {
        self.path.curve_to(x1, y1, x3, y3, x3, y3);
        self.cur = Point::new(x3, y3);
    }

    // MuPDF: pdf_run_re -- fz_rectto: a closed rectangle sub-path. PDF leaves the
    // current point at the start corner `(x, y)`.
    pub(crate) fn op_re(&mut self, x: f32, y: f32, w: f32, h: f32) {
        self.path.rect(x, y, x + w, y + h);
        self.cur = Point::new(x, y);
        self.subpath_start = self.cur;
    }

    // MuPDF: pdf_run_h -- fz_closepath: close the sub-path, current point <- start.
    pub(crate) fn op_closepath(&mut self) {
        self.path.close();
        self.cur = self.subpath_start;
    }

    // -----------------------------------------------------------------------
    // Path painting (pdf_run_f / _S / _B / _n and friends)
    // -----------------------------------------------------------------------

    // MuPDF: the fill/stroke/end operators, which paint csi->path (with the given
    // winding rule), then clear it and apply any pending `W`/`W*` clip.
    /// Paint the current path: optionally `close` it first, `fill` it with `rule`,
    /// and/or `stroke` it; then end the path (apply pending clip, clear). `n`
    /// passes all-false (clip-only / discard).
    pub(crate) fn op_paint(&mut self, close: bool, fill: bool, rule: FillRule, stroke: bool) {
        if close {
            self.path.close();
            self.cur = self.subpath_start;
        }
        if fill {
            self.fill_current(rule);
        }
        if stroke {
            self.stroke_current();
        }
        self.end_path();
    }

    // MuPDF: fz_fill_path via dev->fill_path (the fill material's DeviceRGB colour).
    fn fill_current(&mut self, rule: FillRule) {
        let (ctm, color, clip) = {
            let g: &GState = self.gstate();
            (g.ctm, g.fill_color, g.clip)
        };
        // Split field borrow: `self.dev` (mut) and `self.path` (shared) are disjoint.
        self.dev.fill_path(&self.path, rule, ctm, color, 1.0, clip);
    }

    // MuPDF: fz_stroke_path via dev->stroke_path (line width from the gstate).
    fn stroke_current(&mut self) {
        let (ctm, color, lw, clip) = {
            let g: &GState = self.gstate();
            (g.ctm, g.stroke_color, g.line_width, g.clip)
        };
        self.dev.stroke_path(&self.path, ctm, lw, color, 1.0, clip);
    }

    // MuPDF: the tail of every path-painting operator -- apply a pending clip then
    // reset the current path (pdf_process_end_path / pdf_clear_path).
    fn end_path(&mut self) {
        if self.pending_clip.take().is_some() {
            // TODO(draw): honour the true (non-rectangular) clip outline. Here the
            // running clip is intersected with the clip path's device-space bounding
            // box -- exact for the common `... re W n` rectangular clip, a
            // conservative over-approximation otherwise (never clips too much).
            let ctm = self.gstate().ctm;
            if let Some(bbox) = path_device_bounds(&self.path, ctm) {
                let g = self.gstate_mut();
                g.clip = Some(match g.clip {
                    Some(c) => c.intersect(bbox),
                    None => bbox,
                });
            }
        }
        self.path = Path::new();
        self.cur = Point::new(0.0, 0.0);
        self.subpath_start = self.cur;
    }

    // -----------------------------------------------------------------------
    // Colour (pdf_run_g/_rg/_k, _cs/_CS, _sc/_scn) -> DeviceRGB gstate material
    // -----------------------------------------------------------------------

    // MuPDF: pdf_run_g / pdf_run_G -- DeviceGray fill / stroke.
    pub(crate) fn op_set_gray(&mut self, gray: f32, fill: bool) {
        let rgb = gray_to_rgb(gray);
        self.set_material(ColorSpace::Gray, rgb, fill);
    }

    // MuPDF: pdf_run_rg / pdf_run_RG -- DeviceRGB fill / stroke.
    pub(crate) fn op_set_rgb(&mut self, r: f32, g: f32, b: f32, fill: bool) {
        self.set_material(ColorSpace::Rgb, [r, g, b], fill);
    }

    // MuPDF: pdf_run_k / pdf_run_K -- DeviceCMYK fill / stroke (converted to RGB).
    pub(crate) fn op_set_cmyk(&mut self, c: f32, m: f32, y: f32, k: f32, fill: bool) {
        let rgb = cmyk_to_rgb(c, m, y, k);
        self.set_material(ColorSpace::Cmyk, rgb, fill);
    }

    // MuPDF: pdf_run_cs / pdf_run_CS -- select a colourspace; the current colour is
    // reset to black (fz_set_color to the space's initial value).
    pub(crate) fn op_set_colorspace(&mut self, name: Option<&[u8]>, fill: bool) {
        let Some(name) = name else { return };
        let cs = self.resolve_colorspace(name);
        let rgb = cs.default_rgb();
        self.set_material(cs, rgb, fill);
    }

    // MuPDF: pdf_run_sc / _scn / _SC / _SCN -- set colour components in the current
    // colourspace. A bare pattern name (no numeric operands) is a TODO(draw) skip.
    pub(crate) fn op_set_color(&mut self, stack: &[f64], has_name: bool, fill: bool) {
        if stack.is_empty() {
            // scn/SCN with only a /Pattern name: pattern fills are not painted.
            let _ = has_name;
            return;
        }
        let comps: Vec<f32> = stack.iter().map(|v| *v as f32).collect();
        let (cs, prev) = {
            let g = self.gstate();
            if fill {
                (g.fill_cs.clone(), g.fill_color)
            } else {
                (g.stroke_cs.clone(), g.stroke_color)
            }
        };
        let rgb = cs.to_rgb(&comps, prev);
        let g = self.gstate_mut();
        if fill {
            g.fill_color = rgb;
        } else {
            g.stroke_color = rgb;
        }
    }

    /// Store a resolved colourspace + its DeviceRGB colour into the fill or stroke
    /// material of the current graphics state.
    fn set_material(&mut self, cs: ColorSpace, rgb: [f32; 3], fill: bool) {
        let g = self.gstate_mut();
        if fill {
            g.fill_cs = cs;
            g.fill_color = rgb;
        } else {
            g.stroke_cs = cs;
            g.stroke_color = rgb;
        }
    }
}

/// The device-space bounding box of `path` flattened by `ctm`, or `None` when the
/// path has no drawable geometry. Used for the rectangular clip approximation.
fn path_device_bounds(path: &Path, ctm: Matrix) -> Option<Rect> {
    let polys = path.flatten(ctm);
    let mut bounds: Option<Rect> = None;
    for poly in &polys {
        for p in poly {
            bounds = Some(match bounds {
                None => Rect::new(p.x, p.y, p.x, p.y),
                Some(r) => Rect::new(r.x0.min(p.x), r.y0.min(p.y), r.x1.max(p.x), r.y1.max(p.y)),
            });
        }
    }
    bounds
}

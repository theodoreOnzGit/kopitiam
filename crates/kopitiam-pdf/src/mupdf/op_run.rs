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
//! ~~Text render modes are honoured only for *emission* (modes 3 and 7 stay
//! visible to extraction, as the PDF spec's invisible modes still carry text);
//! actual stroking/filling/clipping, Type3 glyph execution, the glyph cache and
//! bounding-box accumulation are not ported (no rasterisation on the text path).~~
//! **CORRECTED 2026-09-28**: text is filled, stroked (`stroke_glyph`) and
//! clipped (`clip_text` at `ET`) by render mode, Type3 glyph procedures are
//! run (`run_type3_glyph`), and a text-showing operator under a blend mode or
//! soft mask is drawn in its own transparency group (see
//! [`super::op_transparency`]). Modes 3 and 7 still reach extraction. Only
//! the glyph cache and MuPDF's buffered `fz_text` (with its bbox
//! accumulation) are not ported: glyphs go to the device as they are shown.

use super::draw_device::{cmyk_to_rgb, gray_to_rgb};
use super::draw_edge::FillRule;
use super::draw_path::Path;
use super::font::Font;
use super::geometry::{Matrix, Point, Rect};
use super::interpret::{GState, Processor, make_trm};
use super::object::Object;
use super::resources::ColorSpace;
use super::text_device::TextDevice;
use super::xref::PdfDocument;

impl<D: TextDevice + ?Sized> Processor<'_, D> {
    // -----------------------------------------------------------------------
    // Graphics-state stack (pdf-op-run.c)
    // -----------------------------------------------------------------------

    // MuPDF: pdf_run_q + pdf_gsave (pdf-op-run.c:2765) -- push a copy of the top.
    pub(crate) fn op_q(&mut self) {
        let top = self.gstate().clone();
        self.gstack.push(top);
    }

    // MuPDF: pdf_run_Q + pdf_grestore (pdf-op-run.c:667) -- pop, never below
    // `gbot` ("gstate underflow in content stream"), and pop every device clip
    // pushed since the matching `q` (the clip_depth difference).
    pub(crate) fn op_q_restore(&mut self) {
        if self.gstack.len() <= self.gbot + 1 {
            return;
        }
        let popped = self.gstack.pop().expect("len > gbot + 1 >= 1");
        let restored = self.gstate().clip_depth;
        for _ in restored..popped.clip_depth {
            self.dev.pop_clip();
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
            let Some(item) = arr.array_get(i) else {
                continue;
            };
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
            let fillers = {
                let font = self.gstate().text.font.as_ref().unwrap();
                font.decode_fillers(code)
            };
            self.show_char(dec.unicode, dec.advance, dec.cid, &fillers);

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
    fn show_char(&mut self, unicode: char, width: f32, cid: u32, fillers: &[char]) {
        let wmode = self.gstate().text.font.as_ref().unwrap().wmode();

        // Compute the text-space trm + advances from the current text state & Tm.
        let (trm_text, adv_em, char_tx, char_ty) = {
            let g: &GState = self.gstate();
            make_trm(&g.text, wmode, width, self.tos.tm)
        };
        // Device space: post-multiply by the CTM (what the stext device applies).
        let trm_dev = trm_text.concat(self.gstate().ctm);

        // The measuring pass of show_text_grouped: union a generous box of
        // this glyph, advance, emit nothing.
        if self.measure.is_some() {
            let b = self.glyph_measure_box(trm_dev, adv_em);
            if let Some(r) = self.measure.as_mut() {
                *r = r.union(b);
            }
            self.tos.char_tx = char_tx;
            self.tos.char_ty = char_ty;
            self.tos.tm = self.tos.tm.pre_translate(char_tx, char_ty);
            return;
        }

        // Carry the current fill colour to the device so the placeholder glyph
        // boxes render in the text's colour (the sink ignores this by default).
        let fill_color = self.gstate().fill_color;
        self.dev.set_fill_color(fill_color);
        // MuPDF fills text at `gstate->fill.alpha` (pdf_flush_text_imp).
        let fill_alpha = self.gstate().fill_alpha;
        self.dev.set_fill_alpha(fill_alpha);
        // And the `Tr` mode, so a painting device can leave invisible (3) and
        // clip-only (7) text unpainted while extraction still sees the glyph.
        let render_mode = self.gstate().text.render;
        // A shading-pattern fill paints the glyph as clip + fz_fill_shade
        // (pdf_flush_text_imp's PDF_MAT_SHADE), so the device must not ALSO
        // fill it with the flat colour: hand it the same mode minus the fill
        // bit (0->3, 2->1, 4->7, 6->5).
        let pattern_fill = self.gstate().fill_pattern.clone();
        let fills = matches!(render_mode, 0 | 2 | 4 | 6);
        let device_mode = if pattern_fill.is_some() && fills {
            match render_mode {
                0 => 3,
                2 => 1,
                4 => 7,
                _ => 5,
            }
        } else {
            render_mode
        };
        self.dev.set_text_render_mode(device_mode);

        // Emit. Split-borrow via direct field access: `font` reads self.gstack,
        // `dev` is a disjoint field, so the borrow checker permits both (going
        // through the `gstate()` method would borrow all of `self`).
        //
        // Hidden optional content emits nothing -- MuPDF zeroes dofill and
        // dostroke, so neither the page nor stext sees the glyph -- but the
        // pen still advances and a clip mode still accumulates.
        if self.hidden > 0 {
            if render_mode & 4 != 0 {
                let font: &Font = self.gstack.last().unwrap().text.font.as_ref().unwrap();
                self.text_clip.push(super::text_device::ClipGlyph { font: font.clone(), trm: trm_dev, cid });
            }
        } else {
            let font: &Font = self.gstack.last().unwrap().text.font.as_ref().unwrap();
            self.dev
                .show_glyph(font, trm_dev, adv_em, unicode, cid, wmode as u8);
            if let (Some((super::interpret::PatternFill::Shade(shade), gnum)), true) = (&pattern_fill, fills) {
                let g = self.gstack.last().unwrap();
                let pat_ctm = self.gstack.get(*gnum).map_or(g.ctm, |p| p.ctm);
                let (alpha, clip) = (g.fill_alpha, g.clip);
                let glyph = super::text_device::ClipGlyph { font: font.clone(), trm: trm_dev, cid };
                self.dev.clip_text(std::slice::from_ref(&glyph));
                self.dev.fill_shade(shade, pat_ctm, alpha, clip);
                self.dev.pop_clip();
            }
            // pdf_flush_text_imp's PDF_MAT_PATTERN: the glyph as a clip
            // around pdf_show_pattern over the glyph's device bounds.
            if let (Some((super::interpret::PatternFill::Tiling(pat), gnum)), true) = (&pattern_fill, fills) {
                let area = font.glyph_outline(cid).and_then(|o| path_device_bounds(&o, trm_dev));
                if let Some(mut area) = area {
                    if let Some(c) = self.gstack.last().unwrap().clip {
                        area = area.intersect(c);
                    }
                    let glyph = super::text_device::ClipGlyph { font: font.clone(), trm: trm_dev, cid };
                    self.dev.clip_text(std::slice::from_ref(&glyph));
                    self.show_tiling_pattern(pat, *gnum, area);
                    self.dev.pop_clip();
                }
            }
            // Re-borrow: running the cell needed `self` mutably.
            let font: &Font = self.gstack.last().unwrap().text.font.as_ref().unwrap();
            // MuPDF pdf_show_char: modes 4..=7 also add the glyph to the clip
            // accumulator (pdf_tos_accumulate_clip), flushed at ET.
            if render_mode & 4 != 0 {
                self.text_clip.push(super::text_device::ClipGlyph {
                    font: font.clone(),
                    trm: trm_dev,
                    cid,
                });
            }
            // MuPDF pdf_flush_text_imp: modes 1, 2, 5, 6 also `dostroke` --
            // the glyph outline stroked with the stroke colour, alpha and
            // line state (fz_stroke_text).
            if matches!(render_mode, 1 | 2 | 5 | 6) {
                let g = self.gstack.last().unwrap();
                let mut style = g.stroke_style.clone();
                style.line_width = g.line_width;
                self.dev.stroke_glyph(
                    font,
                    trm_dev,
                    g.ctm,
                    cid,
                    &style,
                    g.stroke_color,
                    g.stroke_alpha,
                );
            }
            // MuPDF: "add filler glyphs for one-to-many unicode mapping"
            // (pdf-op-run.c:1449) -- same trm, zero advance, no glyph.
            for &f in fillers {
                self.dev.show_filler_char(font, trm_dev, f, wmode as u8);
            }
        }

        // MuPDF pdf_show_char's Type3 path: the glyph is its procedure, run
        // straight to the device under `t3matrix · trm` (fz_render_t3_glyph_
        // direct). Modes 3 and 7 paint nothing ("If Type3 and tr >= 4 ...
        // ignore the clipping path part"; "if tr != 3, use mode 0").
        if self.hidden == 0 && !matches!(render_mode, 3 | 7) && self.dev.wants_type3_procs() {
            let t3 = self.gstate().text.font.as_ref().and_then(Font::type3_arc);
            if let Some(t3) = t3 {
                self.run_type3_glyph(&t3, cid, trm_dev);
            }
        }

        // MuPDF: pdf_tos_move_after_char (pdf-interpret.c:2062) -- advance Tm.
        self.tos.char_tx = char_tx;
        self.tos.char_ty = char_ty;
        self.tos.tm = self.tos.tm.pre_translate(char_tx, char_ty);
    }

    // MuPDF: fz_render_t3_glyph_direct (font.c:1798) -> pdf_run_glyph
    // (pdf-run.c:423): a fresh processor over the glyph procedure with the
    // current graphics state, the font's resources, ctm = t3matrix · trm.
    fn run_type3_glyph(&mut self, t3: &super::font::Type3Info, cid: u32, trm_dev: Matrix) {
        let Some(Some(proc_ref)) = t3.procs.get(cid as usize) else { return };
        if self.t3_depth >= 8 {
            return; // "recursive type3 font"
        }
        let Ok(content) = self.doc.open_stream(proc_ref) else { return };
        let glyph_ctm = t3.matrix.concat(trm_dev);

        // Own everything the glyph could disturb: the text object state, the
        // clip accumulator, the mask flag, and (via gbot) the gstate stack.
        let saved_tos = self.tos;
        let saved_clip = std::mem::take(&mut self.text_clip);
        let saved_mask = self.t3_mask;
        let saved_path = std::mem::take(&mut self.path);
        let oldtop = self.gstack.len();
        self.op_q();
        {
            let g = self.gstate_mut();
            g.ctm = glyph_ctm;
            // "don't inherit the current font" (pdf_show_char).
            g.text.font = None;
            // The glyph is one object of the text run, whose group and soft
            // mask are already open round it (show_text_grouped). MuPDF's
            // draw device renders a Type3 glyph into its glyph cache and
            // paints the result inside that group; drawing the procedure's
            // own ops under the same /BM and /SMask again would apply them
            // twice, so they are cleared for the procedure.
            g.blend = super::draw_blend::BlendMode::Normal;
            g.softmask = None;
        }
        let pushed = t3.resources.is_dict();
        if pushed {
            self.resources.push(t3.resources.clone());
        }
        let oldbot = self.gbot;
        self.gbot = self.gstack.len() - 1;
        self.t3_mask = false;
        self.t3_depth += 1;
        let _ = self.run_stream(&content);
        self.t3_depth -= 1;
        self.flush_clip_text();
        while self.gstack.len() - 1 > self.gbot {
            self.op_q_restore();
        }
        self.gbot = oldbot;
        if pushed {
            self.resources.pop();
        }
        while self.gstack.len() > oldtop {
            self.op_q_restore();
        }
        self.tos = saved_tos;
        self.text_clip = saved_clip;
        self.t3_mask = saved_mask;
        self.path = saved_path;
    }

    // MuPDF: pdf_show_space (pdf-op-run.c:1457) -- shift Tm by the adjustment.
    /// Advance the text matrix by `tadj` (word spacing, or a `TJ` position
    /// adjustment). Horizontal writing scales by `Tz`; vertical does not.
    fn show_space(&mut self, tadj: f32) {
        let wmode = self
            .gstate()
            .text
            .font
            .as_ref()
            .map(Font::wmode)
            .unwrap_or(0);
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
        // pdf_show_path: "if (pr->super.hidden) dostroke = dofill = 0" -- a
        // clip still applies below.
        let (fill, stroke) = if self.hidden > 0 { (false, false) } else { (fill, stroke) };
        // pdf_show_path: "if (dofill || dostroke) gstate = pdf_begin_group(
        // ctx, pr, bbox, &softmask)" over fz_bound_path (stroke-widened).
        let group = if (fill || stroke) && self.transparency_active() {
            let bbox = self.paint_bbox(stroke);
            Some(self.begin_object_group(bbox))
        } else {
            None
        };
        if fill {
            self.fill_current(rule);
        }
        if stroke {
            self.stroke_current();
        }
        if let Some(save) = group {
            self.end_object_group(save);
        }
        self.end_path();
    }

    // MuPDF: fz_bound_path(path, stroke ? stroke_state : NULL, ctm) -- the
    // device bbox of the current path, widened for a stroke (generously:
    // half the width times the miter limit, as fz_adjust_rect_for_stroke
    // bounds a mitred join).
    fn paint_bbox(&self, stroke: bool) -> Rect {
        let g = self.gstate();
        let Some(r) = path_device_bounds(&self.path, g.ctm) else { return Rect::EMPTY };
        if !stroke {
            return r;
        }
        let w = g.line_width.abs().max(1.0) * 0.5 * g.stroke_style.miter_limit.max(1.0) * g.ctm.max_expansion();
        r.expand(w)
    }

    // MuPDF: pdf_show_shade (pdf-op-run.c:505) for `sh`.
    pub(crate) fn op_sh(&mut self, name: &[u8]) {
        if self.hidden > 0 {
            return; // pdf_show_shade: "if (pr->super.hidden) return"
        }
        let obj_ref = self.lookup_resource_raw("Shading", name);
        let Ok(obj) = self.doc.resolve(&obj_ref) else { return };
        if obj.is_null() {
            return;
        }
        let Ok(shade) = super::shade::Shade::load(self.doc, &obj, &obj_ref) else { return };
        // pdf_begin_group over fz_bound_shade(shd, gstate->ctm).
        let group = self.transparency_active().then(|| {
            let bbox = shade.bound(self.gstate().ctm);
            self.begin_object_group(bbox)
        });
        let (ctm, alpha, clip) = {
            let g = self.gstate();
            (g.ctm, g.fill_alpha, g.clip)
        };
        self.dev.fill_shade(&shade, ctm, alpha, clip);
        if let Some(save) = group {
            self.end_object_group(save);
        }
    }

    // MuPDF: fz_fill_path via dev->fill_path (the fill material's DeviceRGB
    // colour at `gstate->fill.alpha`) -- or, for a shading pattern, the path
    // as a clip around fz_fill_shade in the pattern's space (PDF_MAT_SHADE,
    // pdf-op-run.c:1038).
    fn fill_current(&mut self, rule: FillRule) {
        if let Some((pat, gnum)) = self.gstate().fill_pattern.clone() {
            let (ctm, alpha, clip) = {
                let g = self.gstate();
                (g.ctm, g.fill_alpha, g.clip)
            };
            let pat_ctm = self.gstack.get(gnum).map_or(ctm, |g| g.ctm);
            match pat {
                super::interpret::PatternFill::Shade(shade) => {
                    self.dev.clip_path(&self.path, rule, ctm);
                    self.dev.fill_shade(&shade, pat_ctm, alpha, clip);
                    self.dev.pop_clip();
                }
                super::interpret::PatternFill::Tiling(pat) => {
                    // pdf_show_path's PDF_MAT_PATTERN: clip, show, pop --
                    // inside a Normal group at the fill alpha when that is
                    // not 1 (pdf-op-run.c:1027-1036); nothing at alpha 0.
                    if alpha == 0.0 {
                        return;
                    }
                    let Some(bounds) = path_device_bounds(&self.path, ctm) else { return };
                    let mut area = bounds;
                    if let Some(c) = clip {
                        area = area.intersect(c);
                    }
                    if alpha != 1.0 {
                        self.dev.begin_group(&super::text_device::GroupParams {
                            area: bounds,
                            isolated: false,
                            knockout: false,
                            blend: super::draw_blend::BlendMode::Normal,
                            alpha,
                            gray: false,
                        });
                    }
                    self.dev.clip_path(&self.path, rule, ctm);
                    // The cell content builds its own paths: the page's path
                    // must not leak into the first cell (MuPDF runs the cell
                    // with a fresh path), and is restored for a following
                    // stroke (`B`) or clip (`W`).
                    let saved_path = std::mem::take(&mut self.path);
                    self.show_tiling_pattern(&pat, gnum, area);
                    self.path = saved_path;
                    self.dev.pop_clip();
                    if alpha != 1.0 {
                        self.dev.end_group();
                    }
                }
            }
            return;
        }
        let (ctm, color, alpha, clip) = {
            let g: &GState = self.gstate();
            (g.ctm, g.fill_color, g.fill_alpha, g.clip)
        };
        // Split field borrow: `self.dev` (mut) and `self.path` (shared) are disjoint.
        self.dev.fill_path(&self.path, rule, ctm, color, alpha, clip);
    }

    // MuPDF: fz_stroke_path via dev->stroke_path with `gstate->stroke_state`
    // (width, caps, join, miter limit, dash) at `gstate->stroke.alpha`.
    fn stroke_current(&mut self) {
        let (ctm, color, alpha, clip, style) = {
            let g: &GState = self.gstate();
            let mut style = g.stroke_style.clone();
            style.line_width = g.line_width;
            (g.ctm, g.stroke_color, g.stroke_alpha, g.clip, style)
        };
        self.dev
            .stroke_path_styled(&self.path, ctm, &style, color, alpha, clip);
    }

    // MuPDF: pdf_show_pattern (pdf-op-run.c:2270), the non-tile-cache branch:
    // run the pattern cell once per step over the (device-space) `area`,
    // each under `ptm` pre-translated by the step and clipped to the cell's
    // /BBox (the tile cache MuPDF's draw device uses renders exactly that
    // clipped cell and repeats it).
    fn show_tiling_pattern(&mut self, pat: &super::interpret::TilingPattern, pat_gstate: usize, area: Rect) {
        if pat.xstep == 0.0 || pat.ystep == 0.0 {
            return;
        }
        // A pattern whose cell paints with itself would recurse forever;
        // MuPDF stops at its nesting limit (pdf_run_xobject's cycle check),
        // this shares the Type3 nesting cap.
        if self.t3_depth >= 8 {
            return;
        }
        // A text fill runs the cell mid-BT: keep the text object and the
        // clip accumulator the cell could disturb.
        let saved_tos = self.tos;
        let saved_text_clip = std::mem::take(&mut self.text_clip);
        let parent = self.gstack.get(pat_gstate).cloned().unwrap_or_else(|| self.gstate().clone());
        self.op_q();
        {
            // pdf_copy_pattern_gstate: ctm, stroke state, text state, alphas.
            let g = self.gstate_mut();
            g.ctm = parent.ctm;
            g.stroke_style = parent.stroke_style.clone();
            g.line_width = parent.line_width;
            g.text = parent.text.clone();
            g.fill_alpha = parent.fill_alpha;
            g.stroke_alpha = parent.stroke_alpha;
            // "transparency": the blend mode and the whole soft mask too.
            g.blend = parent.blend;
            g.softmask = parent.softmask.clone();
            g.softmask_tr = parent.softmask_tr.clone();
            // An uncoloured pattern paints in the current fill colour and
            // ignores the colour operators inside it (gstate->ismask);
            // either way the pattern itself stops being the material.
            g.fill_pattern = None;
            g.stroke_pattern = None;
        }
        let saved_mask = self.t3_mask;
        if pat.ismask {
            self.t3_mask = true;
        }
        let ptm = pat.matrix.concat(parent.ctm);
        let Some(invptm) = ptm.try_invert() else {
            self.op_q_restore();
            self.t3_mask = saved_mask;
            return;
        };
        let gparent_save = self.gparent;
        self.gparent = self.gstack.len() - 2;
        let gparent_save_ctm = self.gstack[self.gparent].ctm;
        self.gstack[self.gparent].ctm = ptm;

        let local = area.transform(invptm);
        let (mut fx0, mut fy0) = ((local.x0 - pat.bbox.x0) / pat.xstep, (local.y0 - pat.bbox.y0) / pat.ystep);
        let (mut fx1, mut fy1) = ((local.x1 - pat.bbox.x0) / pat.xstep, (local.y1 - pat.bbox.y0) / pat.ystep);
        if fx0 > fx1 {
            std::mem::swap(&mut fx0, &mut fx1);
        }
        if fy0 > fy1 {
            std::mem::swap(&mut fy0, &mut fy1);
        }
        let pushed = pat.resources.is_dict();
        if pushed {
            self.resources.push(pat.resources.clone());
        }
        self.t3_depth += 1;
        // MuPDF's TILE branch: "only use it as a tile if a whole repeat is
        // required in at least one direction", and never with blending.
        let tiled = (fx1 - fx0 > 1.0 || fy1 - fy0 > 1.0)
            && !pat.uses_blending
            && self.dev.begin_tile(local, pat.bbox, pat.xstep, pat.ystep, ptm);
        if tiled {
            // (The tile cache never hits here: every tile is drawn fresh.)
            self.gstate_mut().ctm = ptm;
            self.run_pattern_cell(&pat.content);
            self.dev.end_tile();
        } else {
            // "When calculating the number of tiles required, we adjust by a
            // small amount to allow for rounding errors."
            let x0 = (fx0 + 0.001).floor() as i64;
            let y0 = (fy0 + 0.001).floor() as i64;
            let mut x1 = (fx1 - 0.001).ceil() as i64;
            let mut y1 = (fy1 - 0.001).ceil() as i64;
            if fx1 > fx0 && x1 == x0 {
                x1 = x0 + 1;
            }
            if fy1 > fy0 && y1 == y0 {
                y1 = y0 + 1;
            }
            // Hostile-input guard (not in MuPDF): the non-tile branch only
            // runs when at most one whole repeat fits in each direction, so
            // a sane cell count is tiny; an absurd one is skipped.
            let cells = (x1 - x0).max(0).saturating_mul((y1 - y0).max(0));
            if cells <= 10_000 {
                for y in y0..y1 {
                    for x in x0..x1 {
                        self.gstate_mut().ctm = ptm.pre_translate(x as f32 * pat.xstep, y as f32 * pat.ystep);
                        self.run_pattern_cell(&pat.content);
                    }
                }
            }
        }
        self.t3_depth -= 1;
        if pushed {
            self.resources.pop();
        }
        let gp = self.gparent;
        if let Some(g) = self.gstack.get_mut(gp) {
            g.ctm = gparent_save_ctm;
        }
        self.gparent = gparent_save;
        self.t3_mask = saved_mask;
        self.op_q_restore();
        self.tos = saved_tos;
        self.text_clip = saved_text_clip;
    }

    // The body of both pdf_show_pattern branches: raise gbot, gsave, run the
    // cell's content, grestore, and unwind anything it left on the stack.
    fn run_pattern_cell(&mut self, content: &[u8]) {
        let oldbot = self.gbot;
        self.gbot = self.gstack.len() - 1;
        self.op_q();
        let _ = self.run_stream(content);
        self.flush_clip_text();
        while self.gstack.len() - 1 > self.gbot {
            self.op_q_restore();
        }
        self.gbot = oldbot;
    }

    // MuPDF: pdf_run_d (pdf-op-run.c:2639) -- the dash array + phase.
    pub(crate) fn op_d(&mut self, arr: &Object, phase: f32) {
        let mut dash = Vec::with_capacity(arr.array_len());
        for i in 0..arr.array_len() {
            let v = arr
                .array_get(i)
                .map(|o| self.doc.resolve(o).unwrap_or(Object::Null).to_real() as f32)
                .unwrap_or(0.0);
            dash.push(v);
        }
        let g = self.gstate_mut();
        g.stroke_style.dash = dash;
        g.stroke_style.dash_phase = phase;
    }

    // MuPDF: pdf_run_gs -> pdf_process_extgstate (pdf-interpret.c:875), the
    // keys this port's graphics state models: LW, LC, LJ, ML, D, Font, CA, ca,
    // and (since the transparency tranche) BM and SMask.
    // RI/FL/OP/op/OPM/UseBlackPtComp/TR/TR2 change nothing we paint (MuPDF
    // itself only warns about transfer functions).
    pub(crate) fn op_gs(&mut self, name: &[u8]) -> super::error::Result<()> {
        let dict = self.lookup_resource("ExtGState", name);
        if !dict.is_dict() {
            // MuPDF: "cannot find ExtGState resource" -- a syntax error that
            // pdf_process_keyword catches and warns about; drawing goes on.
            return Ok(());
        }
        let get = |k: &str| self.doc.resolve_get(&dict, k).unwrap_or(Object::Null);
        let lw = get("LW");
        if lw.is_number() {
            self.gstate_mut().line_width = lw.to_real() as f32;
        }
        let lc = get("LC");
        if lc.is_int() {
            self.gstate_mut().stroke_style.cap = super::interpret::line_cap(lc.to_int() as i32);
        }
        let lj = get("LJ");
        if lj.is_int() {
            self.gstate_mut().stroke_style.join = super::interpret::line_join(lj.to_int() as i32);
        }
        let ml = get("ML");
        if ml.is_number() {
            self.gstate_mut().stroke_style.miter_limit = ml.to_real() as f32;
        }
        let d = get("D");
        if d.is_array() {
            let arr = d
                .array_get(0)
                .map(|o| self.doc.resolve(o).unwrap_or(Object::Null))
                .unwrap_or(Object::Null);
            let phase = d
                .array_get(1)
                .map(|o| self.doc.resolve(o).unwrap_or(Object::Null).to_real() as f32)
                .unwrap_or(0.0);
            self.op_d(&arr, phase);
        }
        let font = get("Font");
        if font.is_array() {
            // [font-ref size]: load the referenced dict like Tf would.
            let size = font
                .array_get(1)
                .map(|o| self.doc.resolve(o).unwrap_or(Object::Null).to_real() as f32)
                .unwrap_or(0.0);
            let fobj = font
                .array_get(0)
                .map(|o| self.doc.resolve(o).unwrap_or(Object::Null))
                .unwrap_or(Object::Null);
            self.gstate_mut().text.size = size;
            if fobj.is_dict() {
                if let Ok(f) = super::font::Font::load(self.doc, &fobj) {
                    self.gstate_mut().text.font = Some(f);
                }
            }
        }
        let ca_stroke = get("CA");
        if ca_stroke.is_number() {
            self.gstate_mut().stroke_alpha = (ca_stroke.to_real() as f32).clamp(0.0, 1.0);
        }
        let ca_fill = get("ca");
        if ca_fill.is_number() {
            self.gstate_mut().fill_alpha = (ca_fill.to_real() as f32).clamp(0.0, 1.0);
        }
        // BM and SMask: the transparency tranche (op_transparency).
        self.gs_transparency(&dict);
        Ok(())
    }

    // MuPDF: the tail of every path-painting operator -- apply a pending clip then
    // reset the current path (pdf_process_end_path / pdf_clear_path).
    fn end_path(&mut self) {
        if let Some(rule) = self.pending_clip.take() {
            // MuPDF pdf_show_path's clip branch: fz_clip_path with the exact
            // outline + winding rule, clip_depth++ (pdf-op-run.c:1093). The
            // device honours the true shape since 0.4.2 (was a TODO(draw)).
            let ctm = self.gstate().ctm;
            self.dev.clip_path(&self.path, rule, ctm);
            self.gstate_mut().clip_depth += 1;
            // The bbox is also kept on the gstate for sinks that only want a
            // rectangle: exact for `re W n`, conservative otherwise.
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
    pub(crate) fn op_set_color_named(&mut self, stack: &[f64], name: Option<&[u8]>, fill: bool) {
        let is_pattern_space = {
            let g = self.gstate();
            matches!(if fill { &g.fill_cs } else { &g.stroke_cs }, ColorSpace::Pattern(_))
        };
        if let (true, Some(n)) = (is_pattern_space, name) {
            self.set_pattern(n, fill);
        }
        if stack.is_empty() {
            return;
        }
        self.op_set_color(stack, name.is_some(), fill);
    }

    pub(crate) fn op_set_color(&mut self, stack: &[f64], has_name: bool, fill: bool) {
        if stack.is_empty() {
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
    /// material of the current graphics state (a colour material: any pattern
    /// is dropped, as pdf_set_colorspace sets `kind = PDF_MAT_COLOR`).
    fn set_material(&mut self, cs: ColorSpace, rgb: [f32; 3], fill: bool) {
        let g = self.gstate_mut();
        if fill {
            g.fill_cs = cs;
            g.fill_color = rgb;
            g.fill_pattern = None;
        } else {
            g.stroke_cs = cs;
            g.stroke_color = rgb;
            g.stroke_pattern = None;
        }
    }

    // MuPDF: pdf_process_SC's pattern branch (pdf-interpret.c) +
    // pdf_set_pattern (pdf-op-run.c:1785): load the named pattern and make it
    // the material, remembering gparent as the pattern space.
    fn set_pattern(&mut self, name: &[u8], fill: bool) {
        let obj_ref = self.lookup_resource_raw("Pattern", name);
        let Ok(obj) = self.doc.resolve(&obj_ref) else { return };
        let pat = match self.doc.resolve_get(&obj, "PatternType").map(|o| o.to_int()) {
            Ok(2) => match super::shade::Shade::load(self.doc, &obj, &obj_ref) {
                Ok(sh) => Some(super::interpret::PatternFill::Shade(std::sync::Arc::new(sh))),
                Err(_) => None,
            },
            // MuPDF: pdf_load_pattern (pdf-pattern.c:74).
            Ok(1) => {
                let get = |k: &str| self.doc.resolve_get(&obj, k).unwrap_or(Object::Null);
                let rect = |o: &Object| {
                    let v = |i: usize| o.array_get(i).and_then(|x| self.doc.resolve(x).ok()).map_or(0.0, |x| x.to_real() as f32);
                    let (a, b, c, d) = (v(0), v(1), v(2), v(3));
                    Rect::new(a.min(c), b.min(d), a.max(c), b.max(d))
                };
                let m = get("Matrix");
                let matrix = if m.array_len() >= 6 {
                    let v = |i: usize| m.array_get(i).and_then(|x| self.doc.resolve(x).ok()).map_or(0.0, |x| x.to_real() as f32);
                    Matrix::new(v(0), v(1), v(2), v(3), v(4), v(5))
                } else {
                    Matrix::IDENTITY
                };
                self.doc.open_stream(&obj_ref).ok().map(|content| {
                    super::interpret::PatternFill::Tiling(std::sync::Arc::new(super::interpret::TilingPattern {
                        ismask: get("PaintType").to_int() == 2,
                        xstep: get("XStep").to_real() as f32,
                        ystep: get("YStep").to_real() as f32,
                        bbox: rect(&get("BBox")),
                        matrix,
                        resources: get("Resources"),
                        content,
                        uses_blending: pattern_uses_blending(self.doc, &obj, &mut Vec::new()),
                    }))
                })
            }
            _ => None,
        };
        let gnum = self.gparent;
        let g = self.gstate_mut();
        if let Some(p) = pat {
            if fill {
                g.fill_pattern = Some((p, gnum));
            } else {
                g.stroke_pattern = Some((p, gnum));
            }
        }
    }
}

/// The device-space bounding box of `path` flattened by `ctm`, or `None` when the
/// path has no drawable geometry. Used for the rectangular clip approximation.
// MuPDF: pdf_pattern_uses_blending / pdf_resources_use_blending /
// pdf_xobject_uses_blending / pdf_extgstate_uses_blending (pdf-page.c:452).
// `seen` is the pdf_cycle list: an object already on the path answers 0.
fn pattern_uses_blending(doc: &PdfDocument, dict: &Object, seen: &mut Vec<(i32, i32)>) -> bool {
    let Some(()) = enter_cycle(dict, seen) else { return false };
    let d = doc.resolve(dict).unwrap_or(Object::Null);
    let r = doc.resolve_get(&d, "Resources").unwrap_or(Object::Null);
    let found = resources_use_blending(doc, &r, seen)
        || extgstate_uses_blending(doc, &doc.resolve_get(&d, "ExtGState").unwrap_or(Object::Null));
    leave_cycle(dict, seen);
    found
}

fn extgstate_uses_blending(doc: &PdfDocument, dict: &Object) -> bool {
    let bm = doc.resolve_get(dict, "BM").unwrap_or(Object::Null);
    !bm.is_null() && !(bm.is_name() && bm.to_name() == b"Normal")
}

fn xobject_uses_blending(doc: &PdfDocument, dict: &Object, seen: &mut Vec<(i32, i32)>) -> bool {
    let Some(()) = enter_cycle(dict, seen) else { return false };
    let d = doc.resolve(dict).unwrap_or(Object::Null);
    let group = doc.resolve_get(&d, "Group").unwrap_or(Object::Null);
    let found = doc.resolve_get(&group, "S").is_ok_and(|s| s.to_name() == b"Transparency")
        || (doc.resolve_get(&d, "Subtype").is_ok_and(|s| s.to_name() == b"Image")
            && doc.resolve_get(&d, "SMask").is_ok_and(|s| !s.is_null()))
        || resources_use_blending(doc, &doc.resolve_get(&d, "Resources").unwrap_or(Object::Null), seen);
    leave_cycle(dict, seen);
    found
}

fn resources_use_blending(doc: &PdfDocument, rdb: &Object, seen: &mut Vec<(i32, i32)>) -> bool {
    if !rdb.is_dict() {
        return false;
    }
    let vals = |key: &str| {
        let o = doc.resolve_get(rdb, key).unwrap_or(Object::Null);
        (0..o.dict_len()).filter_map(|i| o.dict_get_val(i).cloned()).collect::<Vec<_>>()
    };
    vals("ExtGState").iter().any(|g| extgstate_uses_blending(doc, &doc.resolve(g).unwrap_or(Object::Null)))
        || vals("Pattern").iter().any(|p| pattern_uses_blending(doc, p, seen))
        || vals("XObject").iter().any(|x| xobject_uses_blending(doc, x, seen))
}

// pdf_cycle: an indirect object already on the walk is a cycle. A direct
// object cannot recur, but the walk is also capped in depth.
fn enter_cycle(obj: &Object, seen: &mut Vec<(i32, i32)>) -> Option<()> {
    if seen.len() >= 64 {
        return None;
    }
    let key = if obj.is_indirect() { (obj.to_num(), obj.to_gen()) } else { (-1, -1) };
    if key.0 >= 0 && seen.contains(&key) {
        return None;
    }
    seen.push(key);
    Some(())
}

fn leave_cycle(_obj: &Object, seen: &mut Vec<(i32, i32)>) {
    seen.pop();
}

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

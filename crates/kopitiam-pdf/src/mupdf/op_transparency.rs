//! Ported from MuPDF `source/pdf/pdf-op-run.c` -- the transparency half of
//! the run processor: `begin_softmask` / `end_softmask`, `pdf_begin_group` /
//! `pdf_end_group`, `load_transfer_function` and `pdf_run_gs_BM` /
//! `pdf_run_gs_SMask` -- with the ExtGState `/BM` and `/SMask` parsing of
//! `pdf_process_extgstate` (`source/pdf/pdf-interpret.c`) and the group-dict
//! readers of `source/pdf/pdf-xobject.c` (`pdf_xobject_transparency`,
//! `_isolated`, `_knockout`, `_colorspace`) (commit 19f1284, AGPL-3.0,
//! © Artifex Software, Inc.), translated to Rust for KOPITIAM
//! (AGPL-3.0-only). Close adaptation: the control flow follows MuPDF; the
//! code is re-expressed in idiomatic Rust. See docs/ACKNOWLEDGEMENTS.md
//! ("PDF & document-extraction references").
//!
//! # How the pieces fit
//!
//! * `gs` records a blend mode and/or a soft mask on the graphics state
//!   ([`Processor::gs_transparency`]).
//! * Every drawing operator -- path paint, text-showing operator, image,
//!   `sh` -- is bracketed by [`Processor::begin_object_group`] /
//!   [`Processor::end_object_group`] (MuPDF's `pdf_begin_group` /
//!   `pdf_end_group`): with a soft mask in force the mask's group is run
//!   through the device between `begin_mask` / `end_mask`, and with a
//!   non-Normal blend mode the object is drawn in its own (non-isolated)
//!   transparency group. With neither, nothing happens at all.
//! * A `/Group /S /Transparency` Form XObject becomes an explicit group in
//!   [`Processor::run_xobject`](super::page_run).
//!
//! # Divergences (read before trusting a render)
//!
//! * **Text groups per operator.** MuPDF buffers glyphs into an `fz_text`
//!   and groups the whole run at its flush (ET, or a state change). This
//!   port emits glyphs as they are shown, so each `Tj` / `TJ` / `'` / `"`
//!   is its own group. For glyphs that do not overlap each other the
//!   result is identical; overlapping glyphs of one run under a blend mode
//!   would blend with each other here, not in MuPDF.
//! * **The fill+stroke knockout group** pdf_show_path / pdf_flush_text_imp
//!   push for `B`-style operators with a translucent stroke or a blend mode
//!   is not pushed (knockout groups are not implemented; see the draw
//!   device), so the stroke composites over the fill as without a group.

use std::sync::Arc;

use super::draw_blend::BlendMode;
use super::geometry::{Matrix, Rect};
use super::interpret::{Processor, SoftMaskRef};
use super::object::Object;
use super::resources::ColorSpace;
use super::text_device::{GroupParams, TextDevice};

impl<D: TextDevice + ?Sized> Processor<'_, D> {
    // MuPDF: pdf_run_gs_BM + pdf_run_gs_SMask (pdf-op-run.c:2709, 2730) fed
    // by the BM / SMask branches of pdf_process_extgstate
    // (pdf-interpret.c:971-1041).
    /// Apply an ExtGState's `/BM` and `/SMask` keys to the graphics state.
    pub(crate) fn gs_transparency(&mut self, dict: &Object) {
        let get = |k: &str| self.doc.resolve_get(dict, k).unwrap_or(Object::Null);
        let mut bm = get("BM");
        if bm.is_array() {
            bm = bm.array_get(0).map(|o| self.doc.resolve(o).unwrap_or(Object::Null)).unwrap_or(Object::Null);
        }
        if bm.is_name() {
            self.gstate_mut().blend = BlendMode::lookup(bm.to_name());
        }

        let smask = get("SMask");
        if smask.is_dict() {
            let group_ref = smask.dict_gets("G").cloned().unwrap_or(Object::Null);
            let group = self.doc.resolve(&group_ref).unwrap_or(Object::Null);
            if group.is_null() {
                // pdf_run_gs_SMask(NULL): drop the old mask, set nothing.
                self.gstate_mut().softmask = None;
                return;
            }
            let cs = self.xobject_colorspace(&group);
            let n = cs.as_ref().map_or(1, ColorSpace::n);
            // "Default background color is black. Which in CMYK means not
            // all zeros!"
            let mut bc = vec![0.0f32; n];
            if matches!(cs, Some(ColorSpace::Cmyk)) && n >= 4 {
                bc[3] = 1.0;
            }
            let bc_obj = self.doc.resolve_get(&smask, "BC").unwrap_or(Object::Null);
            if bc_obj.is_array() {
                for (k, v) in bc.iter_mut().enumerate() {
                    *v = bc_obj.array_get(k).and_then(|o| self.doc.resolve(o).ok()).map_or(0.0, |o| o.to_real() as f32);
                }
            }
            let luminosity = self.doc.resolve_get(&smask, "S").is_ok_and(|s| s.to_name() == b"Luminosity");
            let tr = smask.dict_gets("TR").cloned().filter(|t| {
                let r = self.doc.resolve(t).unwrap_or(Object::Null);
                !r.is_null() && !(r.is_name() && r.to_name() == b"Identity")
            });
            // begin_softmask: a luminosity mask with no /CS is DeviceGray.
            let bc_gray = bc_to_gray(cs.as_ref(), &bc);
            let resources = self.resources.last().cloned().unwrap_or(Object::Null);
            let g = self.gstate_mut();
            g.softmask = Some(Arc::new(SoftMaskRef { group: group_ref, resources, ctm: g.ctm, luminosity, bc_gray }));
            g.softmask_tr = tr;
        } else if smask.is_name() && smask.to_name() == b"None" {
            self.gstate_mut().softmask = None;
        }
    }

    // MuPDF: pdf_xobject_colorspace (pdf-xobject.c:70) -- the group's /CS,
    // or None when absent or not a valid blending space
    // (fz_is_valid_blend_colorspace: gray, RGB or CMYK).
    pub(crate) fn xobject_colorspace(&self, xobj: &Object) -> Option<ColorSpace> {
        let group = self.doc.resolve_get(xobj, "Group").unwrap_or(Object::Null);
        let cs = group.dict_gets("CS")?;
        let cs = super::resources::load_colorspace(self.doc, cs, 0);
        matches!(cs, ColorSpace::Gray | ColorSpace::Rgb | ColorSpace::Cmyk | ColorSpace::IccN(1 | 3 | 4)).then_some(cs)
    }

    // MuPDF: begin_softmask (pdf-op-run.c:379).
    /// If a soft mask is in force, render it through the device (begin_mask,
    /// the mask group, end_mask) for an object with device bbox `bbox`, and
    /// return it so [`end_softmask`](Processor::end_softmask) can restore it
    /// and pop the clip the device pushed.
    pub(crate) fn begin_softmask(&mut self, bbox: Rect) -> Option<Arc<SoftMaskRef>> {
        let sm = self.gstate().softmask.clone()?;
        let group = self.doc.resolve(&sm.group).unwrap_or(Object::Null);
        let mask_bbox = if sm.luminosity {
            Rect::INFINITE
        } else {
            match super::page_run::bbox_from(self.doc, &group) {
                Some(b) => b.transform(super::page_run::matrix_from(&group).unwrap_or(Matrix::IDENTITY)).transform(sm.ctm),
                // pdf_xobject_bbox of a missing /BBox is the empty rect.
                None => Rect::EMPTY,
            }
        };
        let mask_bbox = mask_bbox.intersect(bbox);

        // pdf_tos_save, plus what the mask's run could disturb here that
        // MuPDF keeps elsewhere: the path being painted, its pending clip,
        // the text-clip accumulator.
        let saved_tos = self.tos;
        let saved_path = std::mem::take(&mut self.path);
        let (saved_cur, saved_start) = (self.cur, self.subpath_start);
        let saved_pending = self.pending_clip.take();
        let saved_text_clip = std::mem::take(&mut self.text_clip);
        let (save_ctm, save_fa, save_sa, saved_blend) = {
            let g = self.gstate_mut();
            let s = (g.ctm, g.fill_alpha, g.stroke_alpha, g.blend);
            g.softmask = None;
            g.ctm = sm.ctm;
            g.fill_alpha = 1.0;
            g.stroke_alpha = 1.0;
            s
        };
        // The transfer function is loaded, and dropped from the gstate, on
        // first use (MuPDF's behaviour, kept on purpose; see GState).
        let tr = self.gstate_mut().softmask_tr.take().and_then(|o| self.transfer_lut(&o));

        self.dev.begin_mask(mask_bbox, sm.luminosity, sm.bc_gray);
        self.gstate_mut().blend = BlendMode::Normal;
        let _ = self.run_xobject(&group, &sm.group, Some(sm.resources.clone()), Matrix::IDENTITY, true);
        self.gstate_mut().blend = saved_blend;
        self.dev.end_mask(tr.as_ref());

        self.tos = saved_tos;
        self.path = saved_path;
        self.cur = saved_cur;
        self.subpath_start = saved_start;
        self.pending_clip = saved_pending;
        self.text_clip = saved_text_clip;
        let g = self.gstate_mut();
        g.ctm = save_ctm;
        g.fill_alpha = save_fa;
        g.stroke_alpha = save_sa;
        Some(sm)
    }

    // MuPDF: end_softmask (pdf-op-run.c:465).
    /// Put the soft mask back on the graphics state and pop the clip
    /// [`begin_softmask`](Processor::begin_softmask) left on the device.
    pub(crate) fn end_softmask(&mut self, save: Option<Arc<SoftMaskRef>>) {
        let Some(sm) = save else { return };
        self.gstate_mut().softmask = Some(sm);
        self.dev.pop_clip();
    }

    // MuPDF: pdf_begin_group (pdf-op-run.c:483).
    /// Open the soft mask and the blend group (each only if in force) for one
    /// drawing operation whose device bbox is `bbox`.
    pub(crate) fn begin_object_group(&mut self, bbox: Rect) -> Option<Arc<SoftMaskRef>> {
        let save = self.begin_softmask(bbox);
        let blend = self.gstate().blend;
        if blend != BlendMode::Normal {
            self.dev.begin_group(&GroupParams { area: bbox, isolated: false, knockout: false, blend, alpha: 1.0, gray: false });
        }
        save
    }

    // MuPDF: pdf_end_group (pdf-op-run.c:494).
    pub(crate) fn end_object_group(&mut self, save: Option<Arc<SoftMaskRef>>) {
        if self.gstate().blend != BlendMode::Normal {
            self.dev.end_group();
        }
        self.end_softmask(save);
    }

    /// Whether a drawing operation needs [`begin_object_group`] at all.
    pub(crate) fn transparency_active(&self) -> bool {
        let g = self.gstate();
        g.blend != BlendMode::Normal || g.softmask.is_some()
    }

    // MuPDF: pdf_flush_text_imp's group bracket (pdf-op-run.c:1172-1258):
    // text that fills or strokes is drawn inside pdf_begin_group over the
    // text bbox; invisible (3) and clip-only (7) text gets no group and no
    // soft-mask run. See the module docs for the per-operator divergence.
    /// Run one text-showing operator `show`, inside a transparency group when
    /// a blend mode or soft mask is in force.
    pub(crate) fn show_text_grouped<F: Fn(&mut Self)>(&mut self, show: F) {
        let wants = {
            let g = self.gstate();
            self.hidden == 0
                && self.measure.is_none()
                && g.text.font.is_some()
                && matches!(g.text.render, 0 | 1 | 2 | 4 | 5 | 6)
                && self.transparency_active()
        };
        if !wants {
            show(self);
            return;
        }
        // Measuring pass: the run's bbox, with the pen put back after.
        let saved_tos = self.tos;
        self.measure = Some(Rect::EMPTY);
        show(self);
        let bbox = self.measure.take().unwrap_or(Rect::EMPTY);
        self.tos = saved_tos;
        // "Don't bother sending a text group with nothing in it".
        if !bbox.is_valid() {
            show(self);
            return;
        }
        let save = self.begin_object_group(bbox);
        show(self);
        self.end_object_group(save);
    }

    /// A generous device box for one glyph (for the text group's bbox):
    /// one em of margin all round the advance box -- MuPDF uses the font
    /// bbox, which this port does not load -- widened by the stroke for the
    /// stroking modes. A Type3 glyph can draw anywhere: infinite.
    pub(crate) fn glyph_measure_box(&self, trm_dev: Matrix, adv_em: f32) -> Rect {
        let g = self.gstate();
        if g.text.font.as_ref().is_some_and(super::font::Font::is_type3) {
            return Rect::INFINITE;
        }
        let r = Rect::new(-1.0, -1.0, adv_em.max(0.0) + 1.0, 2.0).transform(trm_dev);
        if matches!(g.text.render, 1 | 2 | 5 | 6) {
            let w = g.line_width.abs().max(1.0) * g.ctm.max_expansion() * g.stroke_style.miter_limit.max(1.0);
            r.expand(w)
        } else {
            r
        }
    }

    // MuPDF: load_transfer_function (pdf-op-run.c:369) +
    // apply_transfer_function_to_pixmap's memo table (draw-device.c:2322):
    // the function sampled at the 256 byte values, `(uint8_t)
    // fz_clampi(d * 255, 0, 255)`. (MuPDF evaluates per pixel for masks of
    // at most 1024 pixels, which gives the same bytes.)
    /// The `/TR` of a soft mask as a byte lookup table. A function that does
    /// not load is treated as identity.
    pub(crate) fn transfer_lut(&self, obj: &Object) -> Option<[u8; 256]> {
        let f = super::function::PdfFunction::load(self.doc, obj, 1, 1).ok()?;
        let mut lut = [0u8; 256];
        for (w, v) in lut.iter_mut().enumerate() {
            let mut d = [0.0f32];
            f.eval(&[w as f32 / 255.0], &mut d);
            *v = ((d[0] * 255.0) as i32).clamp(0, 255) as u8;
        }
        Some(lut)
    }

    /// The `(isolated, knockout, gray)` of a transparency group XObject, or
    /// `None` when `xobj` is not one.
    // MuPDF: pdf_xobject_transparency / _isolated / _knockout / _colorspace
    // (pdf-xobject.c:44-100).
    pub(crate) fn xobject_transparency(&self, xobj: &Object) -> Option<(bool, bool, bool)> {
        let group = self.doc.resolve_get(xobj, "Group").unwrap_or(Object::Null);
        if !group.is_dict() || !self.doc.resolve_get(&group, "S").is_ok_and(|s| s.to_name() == b"Transparency") {
            return None;
        }
        let flag = |k: &str| self.doc.resolve_get(&group, k).is_ok_and(|o| o.to_bool());
        let isolated = flag("I");
        // The blending space is loaded only for an isolated group.
        let gray = isolated && matches!(self.xobject_colorspace(xobj), Some(ColorSpace::Gray | ColorSpace::IccN(1)));
        Some((isolated, flag("K"), gray))
    }
}

// MuPDF: fz_convert_color(mask_colorspace -> DeviceGray) in
// fz_draw_begin_mask, through color-fast.c's rgb_to_gray / cmyk_to_gray.
/// A soft mask's `/BC` in `cs` (DeviceGray when `None`) as a gray level.
fn bc_to_gray(cs: Option<&ColorSpace>, bc: &[f32]) -> f32 {
    let c = |i: usize| bc.get(i).copied().unwrap_or(0.0);
    match cs {
        None | Some(ColorSpace::Gray) | Some(ColorSpace::IccN(1)) => c(0),
        Some(ColorSpace::Rgb) | Some(ColorSpace::IccN(3)) => c(0) * 0.3 + c(1) * 0.59 + c(2) * 0.11,
        Some(ColorSpace::Cmyk) | Some(ColorSpace::IccN(4)) => 1.0 - (c(0) * 0.3 + c(1) * 0.59 + c(2) * 0.11 + c(3)).min(1.0),
        Some(other) => {
            let rgb = other.to_rgb(bc, [0.0; 3]);
            rgb[0] * 0.3 + rgb[1] * 0.59 + rgb[2] * 0.11
        }
    }
}

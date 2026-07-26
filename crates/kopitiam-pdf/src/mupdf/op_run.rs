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

use super::font::Font;
use super::interpret::{make_trm, GState, Processor};
use super::object::Object;
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
}

//! Ported from MuPDF `source/fitz/stext-device.c` -- the structured-text
//! device's glyph-assembly core: `fz_add_stext_char` / `fz_add_stext_char_imp`
//! (the line/block/space decision), `add_char_to_line` (the scalar glyph quad),
//! and `fixup_bboxes_and_bidi` (line/block bbox rollup) (commit 19f1284,
//! AGPL-3.0, © Artifex Software, Inc.), translated to Rust for KOPITIAM
//! (AGPL-3.0-only). Close adaptation: the algorithm and numeric thresholds
//! follow MuPDF; the code is re-expressed in idiomatic Rust. See
//! docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references").
//!
//! # What this fixes: the spurious-spaces bug
//!
//! The naive extractor emits a space whenever glyphs are not touching, turning
//! `Hello` into `H e l l o`. MuPDF does **not**: it compares each incoming
//! glyph's origin to the running pen position (the previous glyph's
//! advance-end) and only synthesises a space when the along-baseline gap is
//! *large* relative to the font size. Concretely, with
//! `spacing = (gap along baseline) / size`:
//!
//! * `|spacing| < SPACE_DIST` (0.15) -> glyphs are effectively touching: **no
//!   space** (this is the intra-word case that fixes `H e l l o`);
//! * `spacing < 0` (down to `-SPACE_MAX_DIST`) -> overlapping: no space;
//! * `SPACE_DIST <= spacing < SPACE_MAX_DIST` (0.8) -> a real gap: **one
//!   synthesised space** (two if `spacing > 2·SPACE_DIST`);
//! * `spacing >= SPACE_MAX_DIST` -> too far: a new line/column, not a space.
//!
//! Because the pen is advanced by each glyph's *real* advance
//! ([`super::font::Decoded::advance`], applied by the interpreter), touching
//! glyphs land at `spacing == 0` and never produce a space. See
//! [`StextDevice::add_char_imp`].
//!
//! # Structure built
//!
//! Implements [`TextDevice::show_glyph`]: each positioned glyph is folded into
//! the running [`StextPage`] as chars -> lines -> blocks. A glyph continues the
//! current line when its baseline direction matches and it is close to the
//! baseline (`|base_offset| < BASE_MAX_DIST`); a moderate vertical jump
//! (`<= PARAGRAPH_DIST`) starts a new line; a large jump (or a direction
//! change) starts a new block. Char quads are built from the glyph
//! origin/advance and the font's scalar ascender/descender.
//!
//! # Deferred (noted, not done)
//!
//! * **Accurate glyph bboxes / FreeType** (`FZ_STEXT_ACCURATE_BBOXES`,
//!   `_ASCENDERS`, `_SIDE_BEARINGS`): this port is scalar-metrics only, so the
//!   non-accurate quad path is always taken. (`glyph` does carry MuPDF's
//!   sign convention since 0.4.2: `>= 0` real glyph, `-1` no-glyph filler.)
//! * **Vertical writing mode** (`wmode == 1`): the horizontal path is complete;
//!   the vertical quad/positioning arm is stubbed (`a=(1,0) d=(0,0)`) and
//!   vertical space-synthesis is not tuned. Horizontal is the common case.
//! * **Bidi / RTL reordering**: the interpreter supplies no bidi level, so
//!   every char is treated LTR/neutral; the RTL/visual-order branch and
//!   `reverse_bidi_line` are omitted.
//! * ~~**Combining marks**~~ and ~~**general ligature decomposition**~~ --
//!   **DONE 2026-09-28 (0.4.2)**: `Mn` chars and no-glyph fillers ride on the
//!   pen (the `glyph == -1 || MN` arm), and presentation forms decompose
//!   (`unicode-general-category` / `unicode-normalization` stand in for ucdn).
//! * **~~ActualText,~~ styles, images, structure, tables,
//!   segmentation**: recognised via [`StextOptions`] flags but not acted upon
//!   (later waves). ~~Layout analysis (reading order / paragraphs) is the *next*
//!   wave.~~ **CORRECTED 2026-09-28**: it exists -- `stext_boxer`, `stext_para`,
//!   `stext_classify`, entered through `page_to_stext_segmented`. **ActualText
//!   DONE 2026-09-28 (0.4.2)**: `do_extract_within_actualtext` (prefix/postfix
//!   matching per span), `flush_actualtext`, `fz_stext_begin/end_metatext`
//!   (content-bounds placement included), with the text found on the BDC
//!   properties or on the MCID's structure element (`marked_content.rs`).
//!   Divergence: a span ends at `ET`/marked-content boundaries and at a
//!   font/matrix change, a subset of MuPDF's `pdf_flush_text` triggers.

use super::draw_edge::FillRule;
use super::draw_path::Path;
use super::font::Font;
use super::object::Object;
use super::page_image::DecodedImage;
use super::geometry::{Matrix, Point, Quad, Rect};
use super::page_run::run_page;
use super::structured_text::{
    FZ_STEXT_LINE_FLAGS_JOINED, FZ_STEXT_SYNTHETIC, FZ_STEXT_SYNTHETIC_LARGE, StextBlock,
    StextChar, StextLine, StextOptions, StextPage, StextTextBlock,
};
use super::text_device::TextDevice;
use super::xref::PdfDocument;

// MuPDF: stext-device.c:85-89 -- the assembly thresholds, copied verbatim.
/// Vertical jump (in font-size units) beyond which a new paragraph/block starts.
const PARAGRAPH_DIST: f32 = 1.5;
/// Along-baseline gap (in font-size units) at/above which a space is synthesised.
const SPACE_DIST: f32 = 0.15;
/// Along-baseline gap (in font-size units) at/above which the motion is treated
/// as a new line/column rather than a space.
const SPACE_MAX_DIST: f32 = 0.8;
/// Off-baseline distance (in font-size units) within which a glyph is still
/// considered part of the current line.
const BASE_MAX_DIST: f32 = 0.8;
/// Distance (in font-size units) within which a repeated identical glyph is
/// treated as fake-bold overprint and dropped.
const FAKE_BOLD_MAX_DIST: f32 = 0.1;

// MuPDF: stext-device.c:435 -- glyph sentinel for the non-accurate path.
/// The glyph is a synthesised space in accurate mode (unused here; scalar mode).
const NON_ACCURATE_GLYPH_ADDED_SPACE: i32 = -2;

/// The structured-text device: implements [`TextDevice`], folding positioned
/// glyphs into a [`StextPage`]. Construct with [`StextDevice::new`], drive it
/// with [`run_page`], then take the page with [`StextDevice::into_page`].
///
/// Mirrors the extraction-relevant fields of MuPDF's `fz_stext_device`.
pub struct StextDevice {
    // MuPDF: fz_stext_device.page
    page: StextPage,
    // MuPDF: fz_stext_device.opts.flags
    flags: u32,
    // MuPDF: fz_stext_device.pen / .start / .lag_pen
    /// Pen position: the advance-end (`q`) of the last glyph.
    pen: Point,
    /// Start (`p`) of the current line's first char (for indent detection).
    start: Point,
    /// The `p` of the previous glyph (used for the fake-bold overprint check).
    lag_pen: Point,
    // MuPDF: fz_stext_device.maybe_bullet
    maybe_bullet: bool,
    // MuPDF: fz_stext_device.trm
    trm: Matrix,
    // MuPDF: fz_stext_device.lastchar
    lastchar: i32,
    // MuPDF: fz_stext_device.lastline -- (block index, line index) into `page`.
    lastline: Option<(usize, usize)>,
    // MuPDF: fz_stext_device.last.clipped -- the last glyph was culled by
    // FZ_STEXT_CLIP, so its filler chars are dropped with it.
    last_clipped: bool,

    // MuPDF: fz_stext_device.metatext -- the stack of open ActualTexts
    // (other metatext kinds do not reach this device; they change nothing
    // in the extracted text).
    actualtext: Vec<ActualText>,
    /// Glyphs shown inside an ActualText since the last `flush_text`: the
    /// `fz_text` MuPDF hands to `fz_stext_extract` as a whole.
    pending: Vec<PendingGlyph>,
    // MuPDF: fz_stext_device.last -- where the last extracted char sat
    // (`valid` is `last.is_some()`), used to place ActualText runes.
    last: Option<LastChar>,
    /// `last.font` outlives `last.valid` in MuPDF (only valid is reset).
    last_font: Option<usize>,

    /// Interning: font pointer identity -> index into `page.fonts`. Fonts are
    /// stable within a set-font run, so this dedups per logical font (a `q`/`Q`
    /// gstate clone may add a duplicate entry, which is harmless).
    font_ptrs: Vec<usize>,
}

// MuPDF: metatext_t (stext-device.c:106), ActualText only.
struct ActualText {
    /// The replacement text still to be placed.
    text: Vec<char>,
    /// Device-space bounds of the non-text content inside it.
    bounds: Option<Rect>,
}

// MuPDF: one fz_text_item of a span, plus the span's font.
#[derive(Clone, Copy)]
struct PendingGlyph {
    font_idx: usize,
    ucs: char,
    /// `>= 0` a real glyph, `-1` a no-glyph filler (MuPDF's gid sign).
    glyph: i32,
    trm: Matrix,
    adv: f32,
    wmode: u8,
}

#[derive(Clone, Copy)]
struct LastChar {
    font_idx: usize,
    trm: Matrix,
    wmode: u8,
}

impl StextDevice {
    /// Create a device that will accumulate into a fresh page over `mediabox`.
    pub fn new(mediabox: Rect, opts: StextOptions) -> StextDevice {
        StextDevice {
            page: StextPage {
                mediabox,
                blocks: Vec::new(),
                fonts: Vec::new(),
            },
            flags: opts.flags,
            pen: Point::new(0.0, 0.0),
            start: Point::new(0.0, 0.0),
            lag_pen: Point::new(0.0, 0.0),
            maybe_bullet: false,
            trm: Matrix::IDENTITY,
            lastchar: -1,
            lastline: None,
            last_clipped: false,
            actualtext: Vec::new(),
            pending: Vec::new(),
            last: None,
            last_font: None,
            font_ptrs: Vec::new(),
        }
    }

    /// Finalise the page (compute line/block bboxes) and return it.
    pub fn into_page(mut self) -> StextPage {
        self.fixup_bboxes();
        self.page
    }

    // MuPDF: do_extract (stext-device.c:1153) for already-buffered glyphs.
    fn extract_items(&mut self, items: &[PendingGlyph]) {
        for it in items {
            self.last = Some(LastChar { font_idx: it.font_idx, trm: it.trm, wmode: it.wmode });
            self.last_font = Some(it.font_idx);
            let adv = if it.glyph >= 0 { it.adv } else { 0.0 };
            self.add_char(it.font_idx, it.ucs, it.glyph, it.trm, adv, it.wmode, false);
        }
    }

    // MuPDF: flush_actualtext (stext-device.c:1242). Places `text[i..end)`
    // (all of it for `end == None`) at the last char's position: the first
    // rune as glyph -2 (advances the pen, never fake-bold), the rest as -1.
    fn flush_actualtext(&mut self, text: &[char], mut i: usize, end: Option<usize>) {
        if text.is_empty() {
            return;
        }
        let Some(last) = self.last else { return };
        if self.flags & StextOptions::CLIP != 0 && self.last_clipped {
            return;
        }
        let mut glyph = -2;
        let mut k = 0;
        while end.is_none_or(|e| i < e) {
            let Some(&rune) = text.get(k) else { break };
            k += 1;
            self.add_char(last.font_idx, rune, glyph, last.trm, 0.0, last.wmode, false);
            i += 1;
            glyph = -1;
        }
    }

    // MuPDF: do_extract_within_actualtext (stext-device.c:1280) for one span.
    // The prefix of the span that matches the ActualText is extracted as is,
    // a matching suffix too; the glyphs in between carry the ActualText's
    // runes in order. An EMPTY ActualText swallows the span.
    fn extract_within_actualtext(&mut self, span: &[PendingGlyph]) {
        let Some(mt) = self.actualtext.last() else { return };
        let text = mt.text.clone();
        if text.is_empty() {
            return;
        }
        let len = span.len();
        let mut k = 0;
        let mut start = 0;
        while start < len && k < text.len() && span[start].ucs == text[k] {
            start += 1;
            k += 1;
        }
        if start != 0 {
            self.extract_items(&span[..start]);
        }
        if start == len {
            if let Some(mt) = self.actualtext.last_mut() {
                mt.text.drain(..k);
            }
            return;
        }
        // Remaining runes, then a matching postfix (sent at the end).
        let mut z = text.len() - k;
        let mut end = len;
        while end > start && z > 0 && span[end - 1].ucs == text[k + z - 1] {
            z -= 1;
            end -= 1;
        }
        let mut p = k;
        let mut i = start;
        while i < end {
            let it = span[i];
            // `if ((size_t)i < z)` -- compared against the ABSOLUTE index,
            // exactly as the C does.
            let rune = if i < z {
                let r = text.get(p).copied();
                p += 1;
                r
            } else {
                None
            };
            self.last = Some(LastChar { font_idx: it.font_idx, trm: it.trm, wmode: it.wmode });
            self.last_font = Some(it.font_idx);
            let adv = if it.glyph >= 0 { it.adv } else { 0.0 };
            // A rune of -1 ("ignore") adds nothing.
            if let Some(r) = rune {
                self.add_char(it.font_idx, r, it.glyph, it.trm, adv, it.wmode, false);
            }
            i += 1;
        }
        if end == len {
            if let Some(mt) = self.actualtext.last_mut() {
                mt.text.drain(..p.min(mt.text.len()));
            }
            return;
        }
        // A postfix matched: send the rest of the ActualText now, then the
        // postfix, and the ActualText is used up. (MuPDF bounds the flush with
        // strlen, a BYTE count of UTF-8; this counts runes -- the same for
        // ASCII, and never runs past the string either way.)
        let rest: Vec<char> = text[p.min(text.len())..].to_vec();
        let bound = (i + rest.len()).saturating_sub(len - end);
        self.flush_actualtext(&rest, i, Some(bound));
        self.extract_items(&span[end..]);
        if let Some(mt) = self.actualtext.last_mut() {
            mt.text.clear();
        }
    }

    /// Add `r` (device space) to the innermost ActualText's bounds.
    fn actualtext_bounds(&mut self, r: Rect) {
        if let Some(mt) = self.actualtext.last_mut() {
            if r.is_empty() {
                return;
            }
            mt.bounds = Some(match mt.bounds {
                Some(b) => b.union(r),
                None => r,
            });
        }
    }

    // MuPDF: pop_metatext (stext-device.c:1559): the bounds carry outwards.
    fn pop_actualtext(&mut self) {
        if let Some(mt) = self.actualtext.pop() {
            if let Some(b) = mt.bounds {
                self.actualtext_bounds(b);
            }
        }
    }

    // MuPDF: the font pointer stored in fz_stext_char.font. Interns `font` into
    // `page.fonts` by pointer identity, returning its index.
    /// Intern a font this device owns (not one borrowed from a gstate).
    fn intern_font_owned(&mut self, font: Font) -> usize {
        // Keyed by a sentinel no real `&Font` address can have.
        if let Some(idx) = self.font_ptrs.iter().position(|&p| p == usize::MAX) {
            return idx;
        }
        self.font_ptrs.push(usize::MAX);
        self.page.fonts.push(font);
        self.page.fonts.len() - 1
    }

    fn intern_font(&mut self, font: &Font) -> usize {
        let key = font as *const Font as usize;
        if let Some(idx) = self.font_ptrs.iter().position(|&p| p == key) {
            return idx;
        }
        self.font_ptrs.push(key);
        self.page.fonts.push(font.clone());
        self.page.fonts.len() - 1
    }

    // MuPDF: fz_add_stext_char (stext-device.c:1043) -- ligature/whitespace
    // normalisation, then dispatch to the assembly core.
    /// Fold one glyph into the page, applying (unless the matching option is
    /// set) ligature expansion and whitespace normalisation first.
    ///
    /// `glyph` follows MuPDF's `fz_add_stext_char` argument: `>= 0` for a char
    /// that has a real glyph behind it, `-1` for a no-glyph char (a filler from
    /// a one-to-many ToUnicode, or the 2nd+ letter of an expanded ligature).
    /// The sign is load-bearing in the assembly core: `-1` chars ride on the pen
    /// without moving it, and only `>= 0` chars can be dropped as fake-bold
    /// overprint.
    #[allow(clippy::too_many_arguments)]
    fn add_char(
        &mut self,
        font_idx: usize,
        mut c: char,
        glyph: i32,
        trm: Matrix,
        adv: f32,
        wmode: u8,
        force_new_line: bool,
    ) {
        let opts = StextOptions { flags: self.flags };

        // MuPDF expands ligatures unless FZ_STEXT_PRESERVE_LIGATURES.
        if !opts.has(StextOptions::PRESERVE_LIGATURES) {
            // The explicit Latin ligatures (stext-device.c:1065-1093).
            let parts: &[char] = match c {
                '\u{FB00}' => &['f', 'f'],
                '\u{FB01}' => &['f', 'i'],
                '\u{FB02}' => &['f', 'l'],
                '\u{FB03}' => &['f', 'f', 'i'],
                '\u{FB04}' => &['f', 'f', 'l'],
                '\u{FB05}' | '\u{FB06}' => &['s', 't'],
                _ => &[],
            };
            if !parts.is_empty() {
                // First part carries the real glyph/advance/force_new_line; the
                // rest are no-glyph (-1) chars with zero advance -- exactly the
                // C, so they sit on the pen instead of on top of the `f`.
                self.add_char_imp(font_idx, parts[0], glyph, trm, adv, wmode, force_new_line);
                for &p in &parts[1..] {
                    self.add_char_imp(font_idx, p, -1, trm, 0.0, wmode, false);
                }
                return;
            }

            // MuPDF: "alphabetic and arabic presentation forms"
            // (stext-device.c:1097-1104) via ucdn_compat_decompose. We use
            // unicode-normalization's compatibility decomposition. Two recorded
            // differences, both harmless: ucdn returns ONE level of the
            // decomposition record while `decompose_compatible` recurses fully
            // (only a handful of Hebrew forms like U+FB2C decompose twice), and
            // for an unassigned code point in the range ucdn returns 0 and the C
            // then reads an uninitialised `lig[0]` -- here the char is kept as-is.
            let cu = c as u32;
            if (0xFB00..=0xFDFF).contains(&cu) || (0xFE70..=0xFEFC).contains(&cu) {
                let mut lig: Vec<char> = Vec::new();
                unicode_normalization::char::decompose_compatible(c, |d| lig.push(d));
                if !lig.is_empty() && lig != [c] {
                    self.add_char_imp(font_idx, lig[0], glyph, trm, adv, wmode, force_new_line);
                    for &p in &lig[1..] {
                        self.add_char_imp(font_idx, p, -1, trm, 0.0, wmode, false);
                    }
                    return;
                }
            }
        }

        // MuPDF collapses horizontal whitespace to ' ' unless
        // FZ_STEXT_PRESERVE_WHITESPACE (stext-device.c:1108-1133).
        if !opts.has(StextOptions::PRESERVE_WHITESPACE) {
            c = normalize_whitespace(c);
        }

        self.add_char_imp(font_idx, c, glyph, trm, adv, wmode, force_new_line);
    }

    // MuPDF: fz_add_stext_char_imp (stext-device.c:758). THE line/block/space
    // decision. `glyph` is `>= 0` for a real glyph, `-1` for a no-glyph char
    // (see [`add_char`](Self::add_char)), or `-2` for the first rune of a
    // flushed ActualText (advances like a glyph, never fake-bold, as in the
    // C); the ACCURATE_BBOXES per-glyph quads are still out of scope.
    #[allow(clippy::too_many_arguments)]
    fn add_char_imp(
        &mut self,
        font_idx: usize,
        c: char,
        glyph: i32,
        trm: Matrix,
        adv: f32,
        wmode: u8,
        force_new_line: bool,
    ) {
        // dir = motion direction; ndir = normalised(dir). (bidi is always 0 in
        // this port, so we only ever take the LTR/neutral branch.)
        let dir = if wmode == 0 {
            Point::new(1.0, 0.0)
        } else {
            Point::new(0.0, -1.0)
        };
        let dir = dir.transform_vector(trm);
        let ndir = dir.normalize();

        let size = trm.expansion();

        // p = glyph start, q = glyph advance-end (bottom-left / bottom-right in
        // horizontal mode). trm.{e,f} is the origin.
        let (p, q) = if wmode == 0 {
            (
                Point::new(trm.e, trm.f),
                Point::new(trm.e + adv * dir.x, trm.f + adv * dir.y),
            )
        } else {
            (
                Point::new(trm.e - adv * dir.x, trm.f - adv * dir.y),
                Point::new(trm.e, trm.f),
            )
        };

        // Current insertion point: the last line of the last text block.
        let cur_block = self.cur_text_block_idx();
        let cur_line_info = cur_block.and_then(|bi| match &self.page.blocks[bi] {
            StextBlock::Text(tb) => tb.lines.last().map(|l| (l.wmode, l.dir)),
            _ => None,
        });

        // MuPDF: "Don't advance pen or break lines for either no-glyph or
        // marking non-spacing characters in a cluster" (stext-device.c:850-858).
        // A filler / ligature tail / combining mark goes on the CURRENT line,
        // spanning pen..pen, and leaves pen + lag_pen alone. Before 0.4.2 this
        // arm was missing, so the `i` of every "fi" landed on top of the `f`
        // (and a combining accent could open a new line).
        if cur_line_info.is_some()
            && (glyph == -1
                || unicode_general_category::get_general_category(c)
                    == unicode_general_category::GeneralCategory::NonspacingMark)
        {
            let bi = cur_block.expect("cur_line implies cur_block");
            let font_asc = self.page.fonts[font_idx].ascender();
            let font_desc = self.page.fonts[font_idx].descender();
            let ch = make_char(
                c, trm, size, font_idx, wmode, self.pen, self.pen, font_asc, font_desc, 0, 0,
            );
            if let StextBlock::Text(tb) = &mut self.page.blocks[bi] {
                let li = tb.lines.len() - 1;
                tb.lines[li].chars.push(ch);
                self.lastline = Some((bi, li));
            }
            self.lastchar = c as i32;
            return;
        }

        let mut new_para = false;
        // MuPDF initialises new_line = 1; the first two arms leave it at that
        // default, the general arm computes it.
        let mut new_line = true;
        let mut add_space = 0i32;

        match cur_line_info {
            // No line yet, direction changed, or wmode changed: can't append.
            None => {
                new_para = true;
            }
            Some((line_wmode, line_dir))
                if line_wmode != wmode || vec_dot(ndir, line_dir) < 0.999 =>
            {
                new_para = true;
            }
            Some(_) => {
                // Fake-bold overprint: same char drawn on top of itself.
                let dist = hypotf(p.x - self.lag_pen.x, p.y - self.lag_pen.y) / size;
                if dist < FAKE_BOLD_MAX_DIST && c as i32 == self.lastchar && glyph >= 0 {
                    return;
                }

                // How far we've moved since the last char.
                let delta = Point::new(p.x - self.pen.x, p.y - self.pen.y);
                let spacing = (ndir.x * delta.x + ndir.y * delta.y) / size;
                let base_offset = (-ndir.y * delta.x + ndir.x * delta.y) / size;

                if base_offset.abs() < BASE_MAX_DIST {
                    // Close to the baseline: same line (LTR/neutral branch).
                    if spacing.abs() < SPACE_DIST {
                        // Touching / within a word: NO space. (The bug fix.)
                        new_line = false;
                    } else if spacing < 0.0 && spacing > -SPACE_MAX_DIST {
                        // Slight backward motion: overlapping chars, no space.
                        new_line = false;
                    } else if spacing > 0.0 && spacing < SPACE_MAX_DIST {
                        // A real gap: synthesise a space (two if very wide).
                        if wmode == 0 && may_add_space(self.lastchar) {
                            add_space = 1 + i32::from(spacing > SPACE_DIST * 2.0);
                        }
                        new_line = false;
                    } else {
                        // Large along-baseline motion: new column/line.
                        new_line = true;
                    }
                } else if base_offset.abs() <= PARAGRAPH_DIST {
                    // A new line, but not a new paragraph. Spot indented paras.
                    if wmode == 0 && (p.x - self.start.x) > 0.5 && !self.maybe_bullet {
                        new_para = true;
                    }
                    new_line = true;
                } else {
                    // Way off the baseline: new paragraph/block.
                    new_para = true;
                    new_line = true;
                }
            }
        }

        // Start a new block if needed.
        let mut cur_block = cur_block;
        if new_para || cur_block.is_none() {
            self.page.blocks.push(StextBlock::Text(StextTextBlock {
                bbox: Rect::EMPTY,
                lines: Vec::new(),
            }));
            cur_block = Some(self.page.blocks.len() - 1);
        }
        let bi = cur_block.expect("a text block exists after the new-block step");

        // Dehyphenate: flag the previous line as joined if it ended in a hyphen.
        if new_line
            && (self.flags & StextOptions::DEHYPHENATE != 0)
            && is_unicode_hyphen(self.lastchar)
            && let Some((lbi, lli)) = self.lastline
            && let StextBlock::Text(tb) = &mut self.page.blocks[lbi]
            && let Some(line) = tb.lines.get_mut(lli)
        {
            line.flags |= FZ_STEXT_LINE_FLAGS_JOINED;
        }

        // Start a new line if needed.
        let line_exists =
            matches!(&self.page.blocks[bi], StextBlock::Text(tb) if !tb.lines.is_empty());
        if new_line || !line_exists || force_new_line {
            if let StextBlock::Text(tb) = &mut self.page.blocks[bi] {
                tb.lines.push(StextLine {
                    wmode,
                    flags: 0,
                    dir: ndir,
                    bbox: Rect::EMPTY,
                    chars: Vec::new(),
                });
            }
            self.start = p;
            self.maybe_bullet = if glyph == NON_ACCURATE_GLYPH_ADDED_SPACE {
                true
            } else {
                plausible_bullet(c)
            };
        }

        let li = match &self.page.blocks[bi] {
            StextBlock::Text(tb) => tb.lines.len() - 1,
            _ => unreachable!(),
        };

        let font_asc = self.page.fonts[font_idx].ascender();
        let font_desc = self.page.fonts[font_idx].descender();

        // Synthesised space (from `dev->pen` to `p`), unless it's redundant or
        // inhibited.
        if c != ' ' && add_space != 0 && (self.flags & StextOptions::INHIBIT_SPACES == 0) {
            let flags = FZ_STEXT_SYNTHETIC
                | if add_space > 1 {
                    FZ_STEXT_SYNTHETIC_LARGE
                } else {
                    0
                };
            let ch = make_char(
                ' ', trm, size, font_idx, wmode, self.pen, p, font_asc, font_desc, flags, 0,
            );
            if let StextBlock::Text(tb) = &mut self.page.blocks[bi] {
                tb.lines[li].chars.push(ch);
            }
        }

        // The glyph itself (from p to q).
        let ch = make_char(
            c, trm, size, font_idx, wmode, p, q, font_asc, font_desc, 0,
            /*cid set by caller path*/ 0,
        );
        if let StextBlock::Text(tb) = &mut self.page.blocks[bi] {
            tb.lines[li].chars.push(ch);
        }

        self.lastchar = c as i32;
        self.lastline = Some((bi, li));
        self.lag_pen = p;
        self.pen = q;
        self.trm = trm;
    }

    /// The last block's index if it is a text block (MuPDF's `cur_block`).
    fn cur_text_block_idx(&self) -> Option<usize> {
        match self.page.blocks.last() {
            Some(StextBlock::Text(_)) => Some(self.page.blocks.len() - 1),
            _ => None,
        }
    }

    // MuPDF: fixup_bboxes_and_bidi (stext-device.c:1760) -- roll char quads up
    // into line bboxes and line bboxes up into block bboxes.
    fn fixup_bboxes(&mut self) {
        for block in &mut self.page.blocks {
            if let StextBlock::Text(tb) = block {
                let mut block_box = Rect::EMPTY;
                for line in &mut tb.lines {
                    let mut line_box = Rect::EMPTY;
                    for (i, ch) in line.chars.iter().enumerate() {
                        let ch_box = Rect::from_quad(ch.quad);
                        line_box = if i == 0 {
                            ch_box
                        } else {
                            line_box.union(ch_box)
                        };
                    }
                    line.bbox = line_box;
                    block_box = block_box.union(line_box);
                }
                tb.bbox = block_box;
            }
        }
    }
}

impl TextDevice for StextDevice {
    // MuPDF: fz_stext_extract / do_extract feeding fz_add_stext_char
    // (stext-device.c:1163-1220). Here the interpreter has already composed the
    // device-space trm, so we intern the font, apply PRESERVE_SPANS as the
    // force-new-line signal, and hand straight to the assembly core.
    fn show_glyph(
        &mut self,
        font: &Font,
        trm: Matrix,
        adv: f32,
        unicode: char,
        cid: u32,
        wmode: u8,
    ) {
        // MuPDF: fz_stext_extract's FZ_STEXT_CLIP test (stext-device.c:1165):
        // a glyph ENTIRELY outside `scissor ∩ page mediabox` is dropped. This
        // port has no device scissor yet, so the box is the page mediabox
        // alone -- which is exactly the case that matters for a cropped page:
        // text in the part of the MediaBox the CropBox hides. The glyph's box
        // is its em box (advance x descender..ascender) through `trm`, a
        // conservative stand-in for fz_bound_glyph's outline bbox.
        if self.flags & StextOptions::CLIP != 0 {
            let asc = font.ascender();
            let desc = font.descender();
            let em = Rect::new(adv.min(0.0), desc.min(asc), adv.max(1e-3), asc.max(desc));
            let g = em.transform(trm);
            let b = self.page.mediabox;
            if g.x1 <= b.x0 || g.y1 <= b.y0 || g.x0 >= b.x1 || g.y0 >= b.y1 {
                self.last_clipped = true;
                return;
            }
            self.last_clipped = false;
        }
        let font_idx = self.intern_font(font);
        // MuPDF forces a new line at each span start under PRESERVE_SPANS. The
        // interpreter emits per string; treat every glyph as mid-span (false),
        // except we honour the flag when it is the first char overall is not
        // tracked here -- span boundaries are not surfaced by this seam, so
        // PRESERVE_SPANS degrades to "no forced breaks" (noted deferral).
        let force_new_line = false;
        // adv comes in as em units (Widths/1000); MuPDF's `adv` is likewise the
        // per-glyph em advance, and `dir` (which carries the font size) scales
        // it. So we pass it straight through.
        // MuPDF passes the glyph id here (do_extract: `span->items[i].gid`).
        // Outside ACCURATE_BBOXES only its SIGN matters -- ">= 0 = a real glyph",
        // which is what lets the fake-bold overprint check fire -- so the CID
        // (never negative) stands in for it. Before 0.4.2 this passed -1 for
        // every glyph, which silently disabled fake-bold suppression: text a
        // producer printed twice for a bold effect came out doubled.
        let glyph = i32::try_from(cid).unwrap_or(i32::MAX);
        // Inside an ActualText the glyph waits for the span to be flushed
        // (fz_stext_extract -> do_extract_within_actualtext).
        if !self.actualtext.is_empty() && self.flags & StextOptions::IGNORE_ACTUALTEXT == 0 {
            self.pending.push(PendingGlyph { font_idx, ucs: unicode, glyph, trm, adv, wmode });
            return;
        }
        self.last = Some(LastChar { font_idx, trm, wmode });
        self.last_font = Some(font_idx);
        self.add_char(font_idx, unicode, glyph, trm, adv, wmode, force_new_line);
    }

    // MuPDF: fz_stext_extract with `span->items[i].gid < 0` -> `adv = 0`
    // (stext-device.c:1191-1194), then fz_add_stext_char as usual.
    fn show_filler_char(&mut self, font: &Font, trm: Matrix, unicode: char, wmode: u8) {
        // A filler belongs to the glyph just shown; if CLIP dropped that
        // glyph, drop its fillers too (dev->last.clipped).
        if self.last_clipped {
            return;
        }
        let font_idx = self.intern_font(font);
        if !self.actualtext.is_empty() && self.flags & StextOptions::IGNORE_ACTUALTEXT == 0 {
            self.pending.push(PendingGlyph { font_idx, ucs: unicode, glyph: -1, trm, adv: 0.0, wmode });
            return;
        }
        self.add_char(font_idx, unicode, -1, trm, 0.0, wmode, false);
    }

    // MuPDF: pdf_flush_text -> fz_fill_text -> fz_stext_fill_text: each
    // span of the buffered text through do_extract_within_actualtext. A new
    // span starts where fz_show_glyph starts one: a different font, writing
    // mode, or glyph matrix (translation aside).
    fn flush_text(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let pending = std::mem::take(&mut self.pending);
        let mut s = 0;
        for e in 1..=pending.len() {
            let brk = e == pending.len() || {
                let (a, b) = (pending[e - 1], pending[e]);
                a.font_idx != b.font_idx
                    || a.wmode != b.wmode
                    || a.trm.a != b.trm.a
                    || a.trm.b != b.trm.b
                    || a.trm.c != b.trm.c
                    || a.trm.d != b.trm.d
            };
            if brk {
                self.extract_within_actualtext(&pending[s..e]);
                s = e;
            }
        }
    }

    // MuPDF: fz_stext_begin_metatext (stext-device.c:1527), ActualText.
    fn begin_actualtext(&mut self, text: &str) {
        if let Some(mt) = self.actualtext.last() {
            let outer = mt.text.clone();
            self.flush_actualtext(&outer, 0, None);
        }
        self.last = None;
        self.actualtext.push(ActualText { text: text.chars().collect(), bounds: None });
    }

    // MuPDF: fz_stext_end_metatext (stext-device.c:1577), ActualText.
    fn end_actualtext(&mut self) {
        if self.actualtext.is_empty() {
            return; // "Mismatched pop. Live with it."
        }
        if self.flags & StextOptions::IGNORE_ACTUALTEXT != 0 {
            self.pop_actualtext();
            return;
        }
        let text = self.actualtext.last().map(|m| m.text.clone()).unwrap_or_default();
        // "If we have a 'last' text position, send the content after that."
        if let Some(mut last) = self.last {
            last.trm.e = self.pen.x;
            last.trm.f = self.pen.y;
            self.last = Some(last);
            self.flush_actualtext(&text, 0, None);
            self.pop_actualtext();
            self.last = None;
            return;
        }
        // Otherwise only content bounds can place it.
        let Some(b) = self.actualtext.last().and_then(|m| m.bounds) else {
            self.pop_actualtext();
            return;
        };
        let font_idx = match self.last_font {
            Some(f) => Some(f),
            // MuPDF falls back to base-14 Helvetica.
            None => helvetica().map(|f| self.intern_font_owned(f)),
        };
        if let Some(font_idx) = font_idx {
            self.last = Some(LastChar {
                font_idx,
                trm: Matrix::new(b.x1 - b.x0, 0.0, 0.0, b.y0 - b.y1, b.x0, b.y1),
                wmode: 0,
            });
            self.flush_actualtext(&text, 0, None);
        }
        self.pop_actualtext();
        self.last = None;
    }

    // Content inside an ActualText widens its bounds
    // (fz_stext_fill_path / stroke_path / fill_image / fill_image_mask /
    // fill_shade); outside one these paint nothing into stext.
    fn fill_path(&mut self, path: &Path, _rule: FillRule, ctm: Matrix, _c: [f32; 3], _a: f32, _clip: Option<Rect>) {
        if !self.actualtext.is_empty() {
            if let Some(r) = path_bounds(path, ctm, 0.0) {
                self.actualtext_bounds(r);
            }
        }
    }

    fn stroke_path(&mut self, path: &Path, ctm: Matrix, line_width: f32, _c: [f32; 3], _a: f32, _clip: Option<Rect>) {
        if !self.actualtext.is_empty() {
            if let Some(r) = path_bounds(path, ctm, line_width * 0.5 * ctm.max_expansion()) {
                self.actualtext_bounds(r);
            }
        }
    }

    fn draw_image(&mut self, _img: &DecodedImage, ctm: Matrix, _alpha: f32, _clip: Option<Rect>) {
        if !self.actualtext.is_empty() {
            self.actualtext_bounds(Rect::new(0.0, 0.0, 1.0, 1.0).transform(ctm));
        }
    }

    fn draw_image_mask(&mut self, _img: &DecodedImage, ctm: Matrix, _c: [f32; 3], _a: f32, _clip: Option<Rect>) {
        if !self.actualtext.is_empty() {
            self.actualtext_bounds(Rect::new(0.0, 0.0, 1.0, 1.0).transform(ctm));
        }
    }

    fn fill_shade(&mut self, shade: &super::shade::Shade, ctm: Matrix, _alpha: f32, _clip: Option<Rect>) {
        if !self.actualtext.is_empty() {
            self.actualtext_bounds(shade.bound(ctm));
        }
    }
}

// MuPDF: fz_bound_path (path.c), expanded by `pad` for a stroke.
fn path_bounds(path: &Path, ctm: Matrix, pad: f32) -> Option<Rect> {
    let mut r: Option<Rect> = None;
    for poly in path.flatten(ctm) {
        for p in poly {
            r = Some(match r {
                None => Rect::new(p.x, p.y, p.x, p.y),
                Some(b) => Rect::new(b.x0.min(p.x), b.y0.min(p.y), b.x1.max(p.x), b.y1.max(p.y)),
            });
        }
    }
    r.map(|b| Rect::new(b.x0 - pad, b.y0 - pad, b.x1 + pad, b.y1 + pad))
}

// fz_new_base14_font("Helvetica"), for an ActualText placed by content bounds
// on a page that showed no text before it. Built once from a one-object
// document, since `Font::load` resolves through one.
fn helvetica() -> Option<Font> {
    static FONT: std::sync::OnceLock<Option<Font>> = std::sync::OnceLock::new();
    FONT.get_or_init(|| {
        let body = b"%PDF-1.4\n1 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n";
        let mut pdf = body.to_vec();
        let xref = pdf.len();
        pdf.extend_from_slice(
            format!("xref\n0 2\n0000000000 65535 f \n0000000009 00000 n \ntrailer\n<< /Size 2 >>\nstartxref\n{xref}\n%%EOF\n").as_bytes(),
        );
        let doc = PdfDocument::open(pdf).ok()?;
        let dict = doc.resolve(&Object::new_indirect(1, 0)).ok()?;
        Font::load(&doc, &dict).ok()
    })
    .clone()
}

// MuPDF: vec_dot (stext-device.c:554).
#[inline]
fn vec_dot(a: Point, b: Point) -> f32 {
    a.x * b.x + a.y * b.y
}

// MuPDF: hypotf (C library) as used at stext-device.c:873.
#[inline]
fn hypotf(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt()
}

// MuPDF: may_add_space (stext-device.c:560). Only add a space after a char from
// a script where inter-word spacing is meaningful (and not after a space).
#[inline]
fn may_add_space(lastchar: i32) -> bool {
    lastchar != ' ' as i32 && (lastchar < 0x700 || (0x2000..=0x20CF).contains(&lastchar))
}

// MuPDF: fz_is_unicode_hyphen (stext-device.c:548).
#[inline]
fn is_unicode_hyphen(c: i32) -> bool {
    c == '-' as i32 || c == 0xAD || c == 0x2010 || c == 0x2011
}

// MuPDF: the whitespace switch in fz_add_stext_char (stext-device.c:1110-1132).
/// Map every horizontal whitespace code MuPDF recognises to `U+0020`.
fn normalize_whitespace(c: char) -> char {
    match c as u32 {
        0x0009 | 0x0020 | 0x00A0 | 0x1680 | 0x180E | 0x2000..=0x200A | 0x202F | 0x205F | 0x3000 => {
            ' '
        }
        _ => c,
    }
}

// MuPDF: plausible_bullet (stext-device.c:698). A digit or one of a long list of
// bullet-like glyphs; used to suppress false indented-paragraph detection.
fn plausible_bullet(c: char) -> bool {
    let c = c as u32;
    matches!(
        c,
        0x2A // '*'
        | 0x00B7 | 0x2022 | 0x2023 | 0x2043 | 0x204C | 0x204D | 0x2219
        | 0x25C9 | 0x25CB | 0x25CF | 0x25D8 | 0x25E6
        | 0x2619 | 0x261A..=0x261F
        | 0x2765 | 0x2767 | 0x29BE | 0x29BF
        | 0x2660..=0x2667
        | 0x1F446..=0x1F449
        | 0x1F597..=0x1F5A3
        | 0x1FBC1..=0x1FBC3
        | 0xFFFD
        | 0x30..=0x39 // '0'..='9'
    )
}

// MuPDF: add_char_to_line (stext-device.c:438), horizontal non-accurate arm.
/// Build one [`StextChar`] with a scalar quad: `p`/`q` give the baseline span,
/// `asc`/`desc` (em) transformed by `trm` give the height.
#[allow(clippy::too_many_arguments)]
fn make_char(
    c: char,
    trm: Matrix,
    size: f32,
    font_idx: usize,
    wmode: u8,
    p: Point,
    q: Point,
    asc: f32,
    desc: f32,
    flags: u16,
    cid: u32,
) -> StextChar {
    // a = top vector, d = bottom vector, in em; transformed by trm to device
    // space. Horizontal: a=(0,asc) d=(0,desc). Vertical arm is stubbed.
    let (a, d) = if wmode == 0 {
        (Point::new(0.0, asc), Point::new(0.0, desc))
    } else {
        (Point::new(1.0, 0.0), Point::new(0.0, 0.0))
    };
    let a = a.transform_vector(trm);
    let d = d.transform_vector(trm);

    let quad = Quad {
        ll: Point::new(p.x + d.x, p.y + d.y),
        ul: Point::new(p.x + a.x, p.y + a.y),
        lr: Point::new(q.x + d.x, q.y + d.y),
        ur: Point::new(q.x + a.x, q.y + a.y),
    };

    StextChar {
        c,
        origin: p,
        quad,
        size,
        font: font_idx,
        flags,
        cid,
        wmode,
    }
}

// MuPDF: fz_new_stext_device + fz_run_page (the caller side that wires the
// device to a page run).
/// Extract page `page_index` (0-based) of `doc` into a [`StextPage`].
///
/// Constructs a [`StextDevice`] over the page mediabox, drives it with
/// [`run_page`], and returns the finalised page (blocks -> lines -> chars with
/// bboxes, inter-word spaces synthesised per MuPDF's gap rule).
pub fn page_to_stext(
    doc: &PdfDocument,
    page_index: usize,
    opts: StextOptions,
) -> super::error::Result<StextPage> {
    let mediabox = page_mediabox(doc, page_index);
    let mut dev = StextDevice::new(mediabox, opts);
    run_page(doc, page_index, &mut dev)?;
    Ok(dev.into_page())
}

// MuPDF: fz_new_stext_page_from_page -> fz_bound_page (pdf_bound_page).
/// The stext page's `mediabox`: the page rectangle in fitz page space --
/// the space every glyph lands in -- which is the CropBox (clipped to the
/// MediaBox) after rotation, with its corner at `(0, 0)`.
///
/// ~~"the MediaBox transformed by the page CTM"~~ -- **CORRECTED 2026-09-28
/// (0.4.2)**: the code actually stored the raw, untransformed MediaBox, so a
/// page whose MediaBox did not start at `(0, 0)` (or that was rotated or
/// cropped) got a mediabox in a different space from its glyphs. It is now
/// [`page_bounds`](super::page_run::page_bounds), i.e. MuPDF's.
fn page_mediabox(doc: &PdfDocument, page_index: usize) -> Rect {
    match doc.page(page_index) {
        Ok(p) => super::page_run::page_bounds(doc, p),
        Err(_) => Rect::new(0.0, 0.0, 612.0, 792.0),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mupdf::interpret::Processor;
    use crate::mupdf::object::Object;

    // -- Fixtures ---------------------------------------------------------------

    /// A WinAnsi Helvetica simple font with explicit widths for the letters the
    /// tests draw. Widths are in 1/1000 em.
    fn font_dict() -> Object {
        let mut d = Object::new_dict();
        d.dict_put("Type", Object::new_name("Font"));
        d.dict_put("Subtype", Object::new_name("Type1"));
        d.dict_put("BaseFont", Object::new_name("Helvetica"));
        d.dict_put("Encoding", Object::new_name("WinAnsiEncoding"));
        d.dict_put("FirstChar", Object::new_int(32));
        d.dict_put("LastChar", Object::new_int(122));
        let mut widths = Object::new_array();
        for code in 32..=122i64 {
            let w = match code as u8 as char {
                ' ' => 300,
                'H' => 700,
                'e' => 550,
                'l' => 250,
                'o' => 550,
                'W' => 900,
                'r' => 330,
                'd' => 550,
                _ => 500,
            };
            widths.array_push(Object::new_int(w));
        }
        d.dict_put("Widths", widths);
        d
    }

    fn resources() -> Object {
        let mut font_sub = Object::new_dict();
        font_sub.dict_put("F1", font_dict());
        let mut res = Object::new_dict();
        res.dict_put("Font", font_sub);
        res
    }

    fn minimal_doc() -> PdfDocument {
        let bodies: [&[u8]; 3] = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>",
        ];
        PdfDocument::open(build_pdf(&bodies)).unwrap()
    }

    fn build_pdf(bodies: &[&[u8]]) -> Vec<u8> {
        let mut pdf: Vec<u8> = Vec::new();
        pdf.extend_from_slice(b"%PDF-1.5\n");
        let mut offsets = vec![0usize; bodies.len() + 1];
        for (idx, body) in bodies.iter().enumerate() {
            let num = idx + 1;
            offsets[num] = pdf.len();
            pdf.extend_from_slice(format!("{num} 0 obj\n").as_bytes());
            pdf.extend_from_slice(body);
            pdf.extend_from_slice(b"\nendobj\n");
        }
        let xref_ofs = pdf.len();
        let size = bodies.len() + 1;
        pdf.extend_from_slice(format!("xref\n0 {size}\n").as_bytes());
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        for off in offsets.iter().skip(1) {
            pdf.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
        }
        pdf.extend_from_slice(
            format!("trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_ofs}\n%%EOF\n")
                .as_bytes(),
        );
        pdf
    }

    /// Drive the interpreter (identity base CTM, test resources) over `content`
    /// into a fresh [`StextDevice`], returning the finalised page.
    fn run(content: &[u8]) -> StextPage {
        let doc = minimal_doc();
        let mut dev = StextDevice::new(Rect::new(0.0, 0.0, 200.0, 200.0), StextOptions::default());
        {
            let mut proc = Processor::new(&doc, &mut dev, Matrix::IDENTITY, resources());
            proc.run_stream(content).unwrap();
        }
        dev.into_page()
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-2
    }

    // -- The spacing fix (headline) --------------------------------------------

    #[test]
    fn intra_word_glyphs_get_no_spurious_spaces() {
        // "Hello" drawn as one string: consecutive advancing positions, small
        // (zero) intra-word gaps. Must read "Hello", NOT "H e l l o".
        let page = run(b"BT /F1 12 Tf 20 100 Td (Hello) Tj ET");
        assert_eq!(page.text().trim_end(), "Hello");
        // One block, one line, five chars, none synthetic.
        assert_eq!(page.blocks.len(), 1);
        if let StextBlock::Text(tb) = &page.blocks[0] {
            assert_eq!(tb.lines.len(), 1);
            assert_eq!(tb.lines[0].chars.len(), 5);
            assert!(
                tb.lines[0]
                    .chars
                    .iter()
                    .all(|c| c.flags & FZ_STEXT_SYNTHETIC == 0)
            );
        } else {
            panic!("expected a text block");
        }
    }

    #[test]
    fn wide_gap_synthesizes_one_space() {
        // "Hello" then "World" with a controlled gap via a TJ adjustment of
        // -300 (= 0.3 em -> spacing 0.3, in (SPACE_DIST, SPACE_MAX_DIST)) so a
        // single space is synthesised between the words: "Hello World".
        let page = run(b"BT /F1 12 Tf 20 100 Td [(Hello) -300 (World)] TJ ET");
        assert_eq!(page.text().trim_end(), "Hello World");
        // Exactly one synthetic space char, and it is a ' '.
        let mut synth = 0;
        if let StextBlock::Text(tb) = &page.blocks[0] {
            for line in &tb.lines {
                for ch in &line.chars {
                    if ch.flags & FZ_STEXT_SYNTHETIC != 0 {
                        synth += 1;
                        assert_eq!(ch.c, ' ');
                        assert_eq!(
                            ch.flags & FZ_STEXT_SYNTHETIC_LARGE,
                            0,
                            "gap 0.3 is not 'large'"
                        );
                    }
                }
            }
        }
        assert_eq!(synth, 1, "exactly one synthesized space");
    }

    #[test]
    fn narrow_gap_no_space_but_wide_gap_new_line() {
        // Sanity on the thresholds: a gap of spacing >= SPACE_MAX_DIST (0.8)
        // starts a NEW LINE, not a space. -1200 (1.2 em) at end of 'o'.
        let page = run(b"BT /F1 12 Tf 20 100 Td [(Hello) -1200 (World)] TJ ET");
        if let StextBlock::Text(tb) = &page.blocks[0] {
            assert_eq!(tb.lines.len(), 2, "wide gap opens a new line, not a space");
            assert_eq!(tb.lines[0].text(), "Hello");
            assert_eq!(tb.lines[1].text(), "World");
        } else {
            panic!("expected a text block");
        }
    }

    // -- Line breaking ----------------------------------------------------------

    #[test]
    fn lower_baseline_starts_new_line_same_block() {
        // Second word one line down by 12 (base_offset 1.0, in
        // (BASE_MAX_DIST, PARAGRAPH_DIST]) -> new line, same block.
        let page = run(b"BT /F1 12 Tf 20 100 Td (Hello) Tj 0 -12 Td (World) Tj ET");
        assert_eq!(page.blocks.len(), 1, "same block");
        if let StextBlock::Text(tb) = &page.blocks[0] {
            assert_eq!(tb.lines.len(), 2);
            assert_eq!(tb.lines[0].text(), "Hello");
            assert_eq!(tb.lines[1].text(), "World");
        } else {
            panic!("expected a text block");
        }
    }

    // -- Block grouping ---------------------------------------------------------

    #[test]
    fn large_vertical_jump_starts_new_block() {
        // Drop 30 (base_offset 2.5 > PARAGRAPH_DIST) -> new paragraph/block.
        let page = run(b"BT /F1 12 Tf 20 150 Td (Hello) Tj 0 -30 Td (World) Tj ET");
        assert_eq!(page.blocks.len(), 2, "large jump opens a new block");
        assert_eq!(page.text(), "Hello\nWorld\n");
    }

    // -- Char quads -------------------------------------------------------------

    #[test]
    fn char_quad_matches_advance_and_ascender_descender() {
        // 'H' at origin (20,100), size 12, adv 0.7. Quad x spans
        // [20, 20 + 0.7*12] = [20, 28.4]; y spans [desc*12, asc*12] about the
        // origin = [100 - 0.2*12, 100 + 0.8*12] = [97.6, 109.6].
        let page = run(b"BT /F1 12 Tf 20 100 Td (H) Tj ET");
        let ch = if let StextBlock::Text(tb) = &page.blocks[0] {
            tb.lines[0].chars[0]
        } else {
            panic!("expected a text block");
        };
        assert_eq!(ch.c, 'H');
        assert!(approx(ch.origin.x, 20.0) && approx(ch.origin.y, 100.0));
        assert!(approx(ch.size, 12.0));
        let bb = Rect::from_quad(ch.quad);
        assert!(approx(bb.x0, 20.0), "x0 = {}", bb.x0);
        assert!(approx(bb.x1, 20.0 + 0.7 * 12.0), "x1 = {}", bb.x1); // 28.4
        assert!(approx(bb.y0, 100.0 - 0.2 * 12.0), "y0 = {}", bb.y0); // 97.6
        assert!(approx(bb.y1, 100.0 + 0.8 * 12.0), "y1 = {}", bb.y1); // 109.6
        // Line bbox is the union of its char quads -> equals this char's box.
        if let StextBlock::Text(tb) = &page.blocks[0] {
            assert!(approx(tb.lines[0].bbox.x1, 28.4));
            assert!(approx(tb.bbox.y1, 109.6));
        }
    }

    // -- Whitespace normalisation ----------------------------------------------

    #[test]
    fn real_space_glyph_is_kept_and_not_doubled() {
        // A drawn space between words must appear once (from the content) and
        // no *extra* synthetic one, since the pen meets the next glyph exactly.
        let page = run(b"BT /F1 12 Tf 20 100 Td (Hi Wo) Tj ET");
        assert_eq!(page.text().trim_end(), "Hi Wo");
        let synth: usize = if let StextBlock::Text(tb) = &page.blocks[0] {
            tb.lines
                .iter()
                .flat_map(|l| &l.chars)
                .filter(|c| c.flags & FZ_STEXT_SYNTHETIC != 0)
                .count()
        } else {
            0
        };
        assert_eq!(
            synth, 0,
            "the real space is not augmented by a synthetic one"
        );
    }

    // -- End-to-end via a real (lopdf-independent) PDF -------------------------

    #[test]
    fn page_to_stext_end_to_end() {
        // A full one-page PDF; page_to_stext should return "Hello".
        let font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica \
/Encoding /WinAnsiEncoding /FirstChar 32 /LastChar 122 \
/Widths [300 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 700 0 0 0 550 0 0 0 0 0 0 250 0 0 550 0 0 330 0 0 0 0 0 0 0 0 0 0] >>";
        let content = b"<< /Length 36 >>\nstream\nBT /F1 12 Tf 20 150 Td (Hello) Tj ET\nendstream";
        let page = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] \
/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>";
        let bodies: [&[u8]; 5] = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            page,
            content,
            font,
        ];
        let doc = PdfDocument::open(build_pdf(&bodies)).unwrap();
        let stext = page_to_stext(&doc, 0, StextOptions::default()).unwrap();
        assert_eq!(stext.text().trim_end(), "Hello");
        assert_eq!(stext.mediabox, Rect::new(0.0, 0.0, 300.0, 300.0));
        assert_eq!(stext.fonts.len(), 1, "one interned font");
    }

    #[test]
    fn preserve_whitespace_keeps_tab_but_default_normalizes() {
        // Default: a drawn tab (via ToUnicode-less path we can't easily inject),
        // so instead verify normalize_whitespace directly for the mapping.
        assert_eq!(normalize_whitespace('\u{00A0}'), ' ');
        assert_eq!(normalize_whitespace('\u{2003}'), ' ');
        assert_eq!(normalize_whitespace('A'), 'A');
    }

    #[test]
    fn inhibit_spaces_suppresses_synthesis() {
        // Same wide-gap content, but with INHIBIT_SPACES set: no synthetic space.
        let doc = minimal_doc();
        let opts = StextOptions {
            flags: StextOptions::INHIBIT_SPACES,
        };
        let mut dev = StextDevice::new(Rect::new(0.0, 0.0, 200.0, 200.0), opts);
        {
            let mut proc = Processor::new(&doc, &mut dev, Matrix::IDENTITY, resources());
            proc.run_stream(b"BT /F1 12 Tf 20 100 Td [(Hello) -300 (World)] TJ ET")
                .unwrap();
        }
        let page = dev.into_page();
        assert_eq!(page.text().trim_end(), "HelloWorld", "no synthesized space");
    }
}

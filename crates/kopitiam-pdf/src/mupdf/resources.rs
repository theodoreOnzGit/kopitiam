//! Ported from MuPDF `source/pdf/pdf-resources.c` resource-dictionary lookup
//! (`pdf_lookup_resource` in `source/pdf/pdf-interpret.c`) and the `Tf`
//! font-loading path (`pdf_run_Tf` / `pdf_try_load_font`, pdf-op-run.c /
//! pdf-interpret.c) (commit 19f1284, AGPL-3.0, © Artifex Software, Inc.),
//! translated to Rust for KOPITIAM (AGPL-3.0-only). Close adaptation: the
//! algorithms and numeric behaviour follow MuPDF; the code is re-expressed in
//! idiomatic Rust. See docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction
//! references").
//!
//! # Resource lookup + font cache
//!
//! A content stream names its fonts, XObjects, etc. by short names (`/F1`,
//! `/Im0`) that resolve through the current **Resources** dictionary's typed
//! sub-dictionaries (`/Font`, `/XObject`, …). Nested Form XObjects push their own
//! Resources, so lookup walks a stack from the innermost outward
//! ([`Processor::lookup_resource`], `pdf_lookup_resource`).
//!
//! [`Processor::op_tf`] ports `Tf`: look up the `/Font` resource by name, load it
//! into a [`Font`] (caching by object number so a font used on every line is
//! loaded once, matching MuPDF's `pdf_load_font` document cache), and set it as
//! the current font at the given size.

use super::draw_device::cmyk_to_rgb;
use super::font::Font;
use super::interpret::Processor;
use super::object::Object;
use super::text_device::TextDevice;

// MuPDF: fz_colorspace (the fz_colorspace_type taxonomy of `include/mupdf/fitz/
// colorspace.h`) as `pdf_load_colorspace` (pdf-colorspace.c:382) builds it,
// reduced to what the draw device needs: turn component values into DeviceRGB.
/// The colourspace a content stream's fill/stroke operands (or an image, or a
/// shading) are interpreted in.
///
/// Since 0.4.2 the non-device families are converted the way MuPDF converts
/// them when it has no ICC engine: Indexed through its lookup table,
/// Separation/DeviceN through their tint-transform function into the
/// alternate space, Lab through `lab_to_rgb` (color-fast.c). ~~Separation as a
/// gray of `1 - max(tint)`, Indexed as a gray ramp of the index, Lab as RGB~~
/// -- **CORRECTED 2026-09-28**: those were the pre-0.4.2 approximations; the
/// Indexed one in fact used the raw index, so indexed fills came out white.
/// ICCBased still maps by component count (no CMS -- see the coverage map).
#[derive(Clone, Debug)]
pub(crate) enum ColorSpace {
    /// DeviceGray / CalGray / ICCBased N=1.
    Gray,
    /// DeviceRGB / CalRGB / ICCBased N=3.
    Rgb,
    /// DeviceCMYK / CalCMYK / ICCBased N=4.
    Cmyk,
    /// `/Lab` (L* 0..100, a* b* -128..127), converted with MuPDF's fast path.
    Lab,
    /// An N-component ICCBased space we approximate purely by component count.
    IccN(u8),
    /// `[/Separation name alt tint]` (`n = 1`) or `[/DeviceN [names] alt tint]`:
    /// components go through `tint` into `base`. A tint that failed to load
    /// falls back to the old `1 - max(tint)` gray.
    Separation {
        n: u8,
        base: Box<ColorSpace>,
        tint: Option<std::sync::Arc<super::function::PdfFunction>>,
    },
    /// `[/Indexed base hival lookup]`: `(hival + 1) x base.n()` palette bytes.
    Indexed {
        base: Box<ColorSpace>,
        high: i32,
        lookup: std::sync::Arc<Vec<u8>>,
    },
    /// A `/Pattern` space, with the underlying space of an uncoloured pattern
    /// when there is one (`[/Pattern base]`).
    Pattern(Option<Box<ColorSpace>>),
}

impl PartialEq for ColorSpace {
    /// Colour spaces compare by family and component count only -- enough for
    /// the callers, which ask "is this the same kind of space".
    fn eq(&self, other: &ColorSpace) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other) && self.n() == other.n()
    }
}

impl ColorSpace {
    // MuPDF: fz_colorspace_n
    /// How many components a colour in this space has.
    pub(crate) fn n(&self) -> usize {
        match self {
            ColorSpace::Gray => 1,
            ColorSpace::Rgb | ColorSpace::Lab => 3,
            ColorSpace::Cmyk => 4,
            ColorSpace::IccN(n) => *n as usize,
            ColorSpace::Separation { n, .. } => *n as usize,
            ColorSpace::Indexed { .. } => 1,
            ColorSpace::Pattern(_) => 1,
        }
    }

    // MuPDF: fz_clamp_color (colorspace.c:621) + fz_convert_color into
    // DeviceRGB via indexed_via_base / separation_via_base / lab_to_rgb.
    /// Convert `comps` (in this space) to DeviceRGB. Missing components read
    /// as 0. `prev` is the colour to keep when the space cannot map a flat
    /// colour (a coloured `/Pattern`).
    pub(crate) fn to_rgb(&self, comps: &[f32], prev: [f32; 3]) -> [f32; 3] {
        let c = |i: usize| comps.get(i).copied().unwrap_or(0.0);
        let cl = |i: usize| c(i).clamp(0.0, 1.0);
        match self {
            ColorSpace::Gray => [cl(0), cl(0), cl(0)],
            ColorSpace::Rgb => [cl(0), cl(1), cl(2)],
            ColorSpace::Cmyk => cmyk_to_rgb(cl(0), cl(1), cl(2), cl(3)),
            ColorSpace::Lab => lab_to_rgb(c(0).clamp(0.0, 100.0), c(1).clamp(-128.0, 127.0), c(2).clamp(-128.0, 127.0)),
            ColorSpace::IccN(1) => [cl(0), cl(0), cl(0)],
            ColorSpace::IccN(3) => [cl(0), cl(1), cl(2)],
            ColorSpace::IccN(4) => cmyk_to_rgb(cl(0), cl(1), cl(2), cl(3)),
            // Unknown component count: fall back to gray of the first component.
            ColorSpace::IccN(_) => [cl(0), cl(0), cl(0)],
            ColorSpace::Separation { n, base, tint } => {
                let n = (*n).max(1) as usize;
                let src: Vec<f32> = (0..n).map(cl).collect();
                match tint {
                    Some(f) => {
                        let mut out = vec![0.0f32; base.n().max(1)];
                        f.eval(&src, &mut out);
                        base.to_rgb(&out, prev)
                    }
                    None => {
                        let v = 1.0 - src.iter().cloned().fold(0.0f32, f32::max);
                        [v, v, v]
                    }
                }
            }
            ColorSpace::Indexed { base, high, lookup } => {
                // fz_clamp_color: round to an integer index in 0..=high; the
                // converter then looks it up (indexed_via_base).
                let i = ((c(0) + 0.5) as i32).clamp(0, *high) as usize;
                let bn = base.n();
                let mut b = [0.0f32; 4];
                if matches!(**base, ColorSpace::Lab) {
                    b[0] = lookup.get(i * 3).copied().unwrap_or(0) as f32 * 100.0 / 255.0;
                    b[1] = lookup.get(i * 3 + 1).copied().unwrap_or(0) as f32 - 128.0;
                    b[2] = lookup.get(i * 3 + 2).copied().unwrap_or(0) as f32 - 128.0;
                } else {
                    for (k, slot) in b.iter_mut().enumerate().take(bn.min(4)) {
                        *slot = lookup.get(i * bn + k).copied().unwrap_or(0) as f32 / 255.0;
                    }
                }
                base.to_rgb(&b[..bn.min(4)], prev)
            }
            ColorSpace::Pattern(Some(under)) => under.to_rgb(comps, prev),
            ColorSpace::Pattern(None) => prev,
        }
    }

    // MuPDF: pdf_set_colorspace (pdf-op-run.c:1707) -- the initial colour
    // after `cs`/`CS`: v = [0, 0, 0, 1], or all 1.0 for a tint space.
    /// The component values a freshly selected colour space starts with.
    pub(crate) fn initial_comps(&self) -> Vec<f32> {
        match self {
            ColorSpace::Separation { n, .. } => vec![1.0; (*n).max(1) as usize],
            _ => vec![0.0, 0.0, 0.0, 1.0],
        }
    }

    /// The initial colour for this space in DeviceRGB (see
    /// [`initial_comps`](ColorSpace::initial_comps)).
    pub(crate) fn default_rgb(&self) -> [f32; 3] {
        self.to_rgb(&self.initial_comps(), [0.0, 0.0, 0.0])
    }
}

// MuPDF: fung + lab_to_rgb (color-fast.c:130-157) -- the no-ICC Lab path.
fn lab_to_rgb(lstar: f32, astar: f32, bstar: f32) -> [f32; 3] {
    fn fung(x: f32) -> f32 {
        if x >= 6.0 / 29.0 {
            return x * x * x;
        }
        (108.0 / 841.0) * (x - (4.0 / 29.0))
    }
    let m = (lstar + 16.0) / 116.0;
    let l = m + astar / 500.0;
    let n = m - bstar / 200.0;
    let x = fung(l);
    let y = fung(m);
    let z = fung(n);
    let r = (3.240449 * x + -1.537136 * y + -0.498531 * z) * 0.830026;
    let g = (-0.969265 * x + 1.876011 * y + 0.041556 * z) * 1.05452;
    let b = (0.055643 * x + -0.204026 * y + 1.057229 * z) * 1.1003;
    [r.clamp(0.0, 1.0).sqrt(), g.clamp(0.0, 1.0).sqrt(), b.clamp(0.0, 1.0).sqrt()]
}

// MuPDF: pdf_load_colorspace_imp (pdf-colorspace.c:382) -- names and arrays,
// with `depth` standing in for the pdf_cycle_list guard.
/// Interpret a resolved `/ColorSpace` object (a device name, or an array whose
/// head names the family). Anything unrecognised is DeviceGray (MuPDF throws
/// "unknown colorspace"; every caller of this port treats that as gray).
pub(crate) fn load_colorspace(doc: &super::xref::PdfDocument, obj: &Object, depth: u32) -> ColorSpace {
    let obj = doc.resolve(obj).unwrap_or(Object::Null);
    if depth > 16 {
        return ColorSpace::Gray; // "recursive colorspace"
    }
    if obj.is_name() {
        return device_colorspace(obj.to_name()).unwrap_or(match obj.to_name() {
            b"Lab" => ColorSpace::Lab,
            b"CalCMYK" => ColorSpace::Cmyk,
            b"CalRGB" => ColorSpace::Rgb,
            b"CalGray" => ColorSpace::Gray,
            b"Indexed" | b"I" => ColorSpace::Gray,
            _ => ColorSpace::Gray,
        });
    }
    if !obj.is_array() || obj.array_len() == 0 {
        return ColorSpace::Gray;
    }
    let item = |i: usize| obj.array_get(i).cloned().unwrap_or(Object::Null);
    let head = doc.resolve(&item(0)).unwrap_or(Object::Null);
    match head.to_name() {
        b"G" | b"DeviceGray" | b"CalGray" => ColorSpace::Gray,
        b"RGB" | b"DeviceRGB" | b"CalRGB" => ColorSpace::Rgb,
        b"CMYK" | b"DeviceCMYK" | b"CalCMYK" => ColorSpace::Cmyk,
        b"Lab" => ColorSpace::Lab,
        b"ICCBased" => {
            // [/ICCBased streamref]: /N gives the component count; fall back
            // to /Alternate, else guess from nothing.
            let dict = doc.resolve(&item(1)).unwrap_or(Object::Null);
            let n = doc.resolve_get(&dict, "N").map(|o| o.to_int()).unwrap_or(0);
            match n {
                1 => ColorSpace::Gray,
                3 => ColorSpace::Rgb,
                4 => ColorSpace::Cmyk,
                _ => match doc.resolve_get(&dict, "Alternate") {
                    Ok(alt) if !alt.is_null() => load_colorspace(doc, &alt, depth + 1),
                    _ => ColorSpace::IccN(1),
                },
            }
        }
        // MuPDF: load_indexed (pdf-colorspace.c:198)
        b"Indexed" | b"I" => {
            let base = load_colorspace(doc, &item(1), depth + 1);
            let high = doc.resolve(&item(2)).map(|o| o.to_int()).unwrap_or(0).clamp(0, 255) as i32;
            let n = base.n() * (high as usize + 1);
            let lookup_obj = item(3);
            let resolved = doc.resolve(&lookup_obj).unwrap_or(Object::Null);
            let mut lookup = if resolved.is_string() {
                resolved.to_string_bytes().to_vec()
            } else if lookup_obj.is_indirect() {
                doc.open_stream(&lookup_obj).unwrap_or_default()
            } else {
                Vec::new()
            };
            lookup.resize(n, 0);
            ColorSpace::Indexed { base: Box::new(base), high, lookup: std::sync::Arc::new(lookup) }
        }
        // MuPDF: load_devicen (pdf-colorspace.c:123)
        b"Separation" | b"DeviceN" => {
            let names = doc.resolve(&item(1)).unwrap_or(Object::Null);
            let n = if names.is_array() { names.array_len().clamp(1, 32) } else { 1 };
            let base = load_colorspace(doc, &item(2), depth + 1);
            let tint = super::function::PdfFunction::load(doc, &item(3), n, base.n())
                .ok()
                .map(std::sync::Arc::new);
            ColorSpace::Separation { n: n as u8, base: Box::new(base), tint }
        }
        b"Pattern" => {
            let under = item(1);
            if under.is_null() {
                ColorSpace::Pattern(None)
            } else {
                ColorSpace::Pattern(Some(Box::new(load_colorspace(doc, &under, depth + 1))))
            }
        }
        _ => ColorSpace::Gray,
    }
}

impl<D: TextDevice + ?Sized> Processor<'_, D> {
    // MuPDF: pdf_lookup_resource (pdf-interpret.c:33) -- walk the resource stack
    // innermost-first, returning the *resolved* value for `type`/`name`.
    /// Look up a resource: for each Resources dict from the top of the stack down,
    /// find its `type` sub-dict (`/Font`, `/XObject`, …) then the entry `name`,
    /// resolving both. Returns [`Object::Null`] if not found.
    pub(crate) fn lookup_resource(&self, typ: &str, name: &[u8]) -> Object {
        for res in self.resources.iter().rev() {
            let sub = match self.doc.resolve_get(res, typ) {
                Ok(s) if s.is_dict() => s,
                _ => continue,
            };
            if let Some(v) = sub.dict_get(name)
                && let Ok(resolved) = self.doc.resolve(v)
                && !resolved.is_null()
            {
                return resolved;
            }
        }
        Object::Null
    }

    // MuPDF: pdf_run_Tf (pdf-op-run.c:2976) + the Tf branch of pdf_process_keyword
    // (pdf-interpret.c:1403) -- look up + load the font, set font & size.
    /// Handle `Tf`: set the current font (by resource `name`) and `size`. A
    /// missing or unloadable font leaves the current font unset (MuPDF falls back
    /// to a "hail mary" font; this port simply shows nothing until a good `Tf`,
    /// which is safe for extraction).
    pub(crate) fn op_tf(&mut self, name: Option<&[u8]>, size: f32) -> super::error::Result<()> {
        self.gstate_mut().text.size = size;

        let Some(name) = name else {
            self.gstate_mut().text.font = None;
            return Ok(());
        };

        let font_obj = self.lookup_resource("Font", name);
        if !font_obj.is_dict() {
            self.gstate_mut().text.font = None;
            return Ok(());
        }

        // Cache by the resource entry's object number when it is indirect; a
        // direct dict (num 0) is loaded afresh. This mirrors MuPDF caching the
        // loaded pdf_font_desc on the document.
        let key = self
            .resources
            .iter()
            .rev()
            .find_map(|res| {
                let sub = self.doc.resolve_get(res, "Font").ok()?;
                sub.dict_get(name).map(|v| v.to_num())
            })
            .unwrap_or(0);

        let font = if key != 0 {
            if let Some(f) = self.fonts.get(&key) {
                f.clone()
            } else {
                let f = Font::load(self.doc, &font_obj)?;
                self.fonts.insert(key, f.clone());
                f
            }
        } else {
            Font::load(self.doc, &font_obj)?
        };

        self.gstate_mut().text.font = Some(font);
        Ok(())
    }

    // MuPDF: pdf_run_cs / pdf_run_CS + pdf_load_colorspace (pdf-op-run.c /
    // pdf-colorspace.c) -- resolve a `cs`/`CS` name to a [`ColorSpace`].
    /// Resolve a colourspace named by a `cs`/`CS` operator: the device families by
    /// name, otherwise a `/ColorSpace` resource entry (an array like
    /// `[/ICCBased …]`). Unknown names fall back to [`ColorSpace::Gray`].
    pub(crate) fn resolve_colorspace(&self, name: &[u8]) -> ColorSpace {
        if let Some(cs) = device_colorspace(name) {
            return cs;
        }
        let obj = self.lookup_resource("ColorSpace", name);
        self.colorspace_from_obj(&obj)
    }

    /// Interpret a resolved `/ColorSpace` object (a device name, or an array whose
    /// head names the family). Deferred families are approximated (see
    /// [`ColorSpace`]); anything unrecognised defaults to gray.
    pub(crate) fn colorspace_from_obj(&self, obj: &Object) -> ColorSpace {
        load_colorspace(self.doc, obj, 0)
    }
}

/// The device-family colourspaces addressable directly by name (from `cs`/`CS` or
/// an image `/ColorSpace`), or `None` for a resource-dict name.
fn device_colorspace(name: &[u8]) -> Option<ColorSpace> {
    match name {
        b"DeviceGray" | b"G" => Some(ColorSpace::Gray),
        b"DeviceRGB" | b"RGB" => Some(ColorSpace::Rgb),
        b"DeviceCMYK" | b"CMYK" => Some(ColorSpace::Cmyk),
        b"Pattern" => Some(ColorSpace::Pattern(None)),
        _ => None,
    }
}

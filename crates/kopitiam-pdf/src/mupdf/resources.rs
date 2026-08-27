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
// colorspace.h`), collapsed to what the draw device needs to turn `sc`/`scn`
// component operands into DeviceRGB. See `pdf_load_colorspace` (pdf-colorspace.c).
/// The colourspace a content stream's fill/stroke operands are interpreted in.
///
/// Only the device families convert exactly; ICCBased/Cal/Lab are approximated by
/// component count, and Separation/DeviceN/Indexed carry documented approximations
/// (see [`ColorSpace::to_rgb`]). Enough to paint colour faithfully for the common
/// Device* cases and never corrupt the pixmap for the rest.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ColorSpace {
    /// DeviceGray / CalGray / ICCBased N=1.
    Gray,
    /// DeviceRGB / CalRGB / ICCBased N=3.
    Rgb,
    /// DeviceCMYK / ICCBased N=4.
    Cmyk,
    /// An N-component ICCBased/Lab space we approximate purely by component count.
    IccN(u8),
    /// `[/Separation …]` / `[/DeviceN …]` with `n` colorants; approximated as a
    /// gray of `1 - max(tint)` (0 tint = paper white, full tint = ink black).
    Separation(u8),
    /// `[/Indexed base hival lookup]`; approximated (TODO) as a gray ramp of
    /// `index / hival` — the true palette lookup is deferred.
    Indexed(i32),
    /// A `/Pattern` space — not paintable as a flat colour here; the operand is
    /// ignored and the prior colour kept.
    Pattern,
}

impl ColorSpace {
    // MuPDF: fz_convert_color into DeviceRGB (the draw device forces DeviceRGB).
    /// Convert `comps` (0..=1 each, in this space) to DeviceRGB. Missing components
    /// read as 0. `prev` is the colour to keep when the space cannot map a flat
    /// colour (a `/Pattern`).
    pub(crate) fn to_rgb(&self, comps: &[f32], prev: [f32; 3]) -> [f32; 3] {
        let c = |i: usize| comps.get(i).copied().unwrap_or(0.0);
        match self {
            ColorSpace::Gray => [c(0), c(0), c(0)],
            ColorSpace::Rgb => [c(0), c(1), c(2)],
            ColorSpace::Cmyk => cmyk_to_rgb(c(0), c(1), c(2), c(3)),
            ColorSpace::IccN(1) => [c(0), c(0), c(0)],
            ColorSpace::IccN(3) => [c(0), c(1), c(2)],
            ColorSpace::IccN(4) => cmyk_to_rgb(c(0), c(1), c(2), c(3)),
            // Unknown component count: fall back to gray of the first component.
            ColorSpace::IccN(_) => [c(0), c(0), c(0)],
            ColorSpace::Separation(n) => {
                // TODO(draw): run the real tint-transform function. Approximate a
                // subtractive colorant: max tint -> black, no tint -> white.
                let mut tint = 0.0f32;
                for i in 0..(*n).max(1) as usize {
                    tint = tint.max(c(i));
                }
                let v = 1.0 - tint;
                [v, v, v]
            }
            // TODO(draw): resolve the palette. Approximate a gray ramp of the index.
            ColorSpace::Indexed(_) => [c(0), c(0), c(0)],
            ColorSpace::Pattern => prev,
        }
    }

    /// The initial colour for this space (black) after a `cs`/`CS` selection,
    /// matching MuPDF resetting the material to black on a colourspace change.
    pub(crate) fn default_rgb(&self) -> [f32; 3] {
        [0.0, 0.0, 0.0]
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
        if obj.is_name() {
            return device_colorspace(obj.to_name()).unwrap_or(ColorSpace::Gray);
        }
        if !obj.is_array() || obj.array_len() == 0 {
            return ColorSpace::Gray;
        }
        let head = obj.array_get(0).map(|o| o.to_name()).unwrap_or(b"");
        match head {
            b"ICCBased" => {
                // [/ICCBased streamref]: the stream dict's /N gives the component
                // count; fall back to its /Alternate colourspace, else guess gray.
                let stream = obj.array_get(1).cloned().unwrap_or(Object::Null);
                let dict = self.doc.resolve(&stream).unwrap_or(Object::Null);
                let n = self
                    .doc
                    .resolve_get(&dict, "N")
                    .map(|o| o.to_int())
                    .unwrap_or(0);
                match n {
                    1 => ColorSpace::Gray,
                    3 => ColorSpace::Rgb,
                    4 => ColorSpace::Cmyk,
                    _ => {
                        if let Ok(alt) = self.doc.resolve_get(&dict, "Alternate")
                            && !alt.is_null()
                        {
                            return self.colorspace_from_obj(&alt);
                        }
                        ColorSpace::IccN(1)
                    }
                }
            }
            b"CalGray" => ColorSpace::Gray,
            b"CalRGB" | b"Lab" => ColorSpace::Rgb,
            b"Separation" => ColorSpace::Separation(1),
            b"DeviceN" => {
                // [/DeviceN [names] alternate tint]: colorant count = names.len().
                let names = obj.array_get(1).cloned().unwrap_or(Object::Null);
                let n = names.array_len().max(1) as u8;
                ColorSpace::Separation(n)
            }
            b"Indexed" | b"I" => {
                let hival = obj.array_get(2).map(|o| o.to_int()).unwrap_or(0) as i32;
                ColorSpace::Indexed(hival)
            }
            b"Pattern" => ColorSpace::Pattern,
            b"DeviceGray" | b"G" => ColorSpace::Gray,
            b"DeviceRGB" | b"RGB" => ColorSpace::Rgb,
            b"DeviceCMYK" | b"CMYK" => ColorSpace::Cmyk,
            _ => ColorSpace::Gray,
        }
    }
}

/// The device-family colourspaces addressable directly by name (from `cs`/`CS` or
/// an image `/ColorSpace`), or `None` for a resource-dict name.
fn device_colorspace(name: &[u8]) -> Option<ColorSpace> {
    match name {
        b"DeviceGray" | b"G" => Some(ColorSpace::Gray),
        b"DeviceRGB" | b"RGB" => Some(ColorSpace::Rgb),
        b"DeviceCMYK" | b"CMYK" => Some(ColorSpace::Cmyk),
        b"Pattern" => Some(ColorSpace::Pattern),
        _ => None,
    }
}

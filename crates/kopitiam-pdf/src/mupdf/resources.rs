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

use super::font::Font;
use super::interpret::Processor;
use super::object::Object;
use super::text_device::TextDevice;

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
}

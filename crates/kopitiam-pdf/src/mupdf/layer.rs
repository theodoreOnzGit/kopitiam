//! Ported from MuPDF `source/pdf/pdf-layer.c` -- the optional-content
//! (layer) visibility test `pdf_is_ocg_hidden` / `pdf_is_ocg_hidden_imp`, the
//! default-configuration read of `pdf_read_ocg` + `pdf_select_layer_config`,
//! and `ocg_intents_include` (commit 19f1284, AGPL-3.0, © Artifex Software,
//! Inc.), translated to Rust for KOPITIAM (AGPL-3.0-only). Close adaptation:
//! the algorithms and numeric behaviour follow MuPDF; the code is re-expressed
//! in idiomatic Rust. See docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction
//! references").
//!
//! # Why (0.4.2)
//!
//! Before 0.4.2 the interpreter skipped `BDC`/`EMC` and never looked at an
//! XObject's `/OC`, so every layer drew, including the ones the document says
//! are OFF: alternate-language layers stacked on top of each other, print-only
//! watermarks on screen, CAD layers the author switched off. MuPDF shows only
//! what the default configuration (`/OCProperties /D`) turns on, for the
//! "View" usage -- and so does this.
//!
//! What is deliberately the same as MuPDF, including its FIXMEs: an OCMD's
//! `/VE` visibility expression counts as visible; `/Usage` is consulted only
//! through its `<usage>State` key; Zoom/User/Language usage is not evaluated.

use super::object::Object;
use super::xref::PdfDocument;

/// The default optional-content configuration (`pdf_ocg_descriptor` after
/// `pdf_select_layer_config(doc, -1)`).
#[derive(Clone, Debug, Default)]
pub struct OcgConfig {
    /// Every OCG in `/OCProperties /OCGs`, as its raw (unresolved) array
    /// entry, with its ON (true) / OFF state.
    ocgs: Vec<(Object, bool)>,
    /// The configuration's `/Intent` (None when absent).
    intent: Option<Object>,
}

impl OcgConfig {
    // MuPDF: pdf_read_ocg (pdf-layer.c:810) + pdf_select_layer_config(-1)
    // (pdf-layer.c:281).
    /// Read the document's default layer configuration. A document without
    /// `/OCProperties` (or with no OCGs) yields an empty config, in which
    /// everything is visible.
    pub fn load(doc: &PdfDocument) -> OcgConfig {
        let Ok(root) = doc.catalog() else { return OcgConfig::default() };
        let props = doc.resolve_get(&root, "OCProperties").unwrap_or(Object::Null);
        if !props.is_dict() {
            return OcgConfig::default();
        }
        let list = doc.resolve_get(&props, "OCGs").unwrap_or(Object::Null);
        let mut ocgs: Vec<(Object, bool)> = (0..list.array_len())
            .filter_map(|i| list.array_get(i).cloned())
            .map(|o| (o, true))
            .collect();
        let cobj = doc.resolve_get(&props, "D").unwrap_or(Object::Null);
        let intent = doc.resolve_get(&cobj, "Intent").ok().filter(|o| !o.is_null());
        // BaseState: Unchanged keeps the loaded states (all ON here), OFF
        // turns everything off, anything else is ON.
        let base = doc.resolve_get(&cobj, "BaseState").unwrap_or(Object::Null);
        match base.to_name() {
            b"Unchanged" => {}
            b"OFF" => ocgs.iter_mut().for_each(|o| o.1 = false),
            _ => ocgs.iter_mut().for_each(|o| o.1 = true),
        }
        for (key, on) in [("ON", true), ("OFF", false)] {
            let arr = doc.resolve_get(&cobj, key).unwrap_or(Object::Null);
            for i in 0..arr.array_len() {
                let Some(o) = arr.array_get(i) else { continue };
                if let Some(slot) = ocgs.iter_mut().find(|(g, _)| same_obj(g, o)) {
                    slot.1 = on;
                }
            }
        }
        OcgConfig { ocgs, intent }
    }

    // MuPDF: pdf_is_ocg_hidden (pdf-layer.c:791) with usage "View".
    /// Whether content marked with `ocg` (an OCG or OCMD dict, a reference to
    /// one, or -- via `lookup` -- a `/Properties` resource name) is hidden.
    /// `lookup` resolves a name in the current `/Properties` resources.
    pub fn is_hidden<F: Fn(&[u8]) -> Object>(&self, doc: &PdfDocument, ocg: &Object, lookup: &F) -> bool {
        self.hidden_imp(doc, ocg, lookup, "View", 0)
    }

    // MuPDF: pdf_is_ocg_hidden_imp (pdf-layer.c:618)
    fn hidden_imp<F: Fn(&[u8]) -> Object>(&self, doc: &PdfDocument, ocg: &Object, lookup: &F, usage: &str, depth: u32) -> bool {
        // "If no ocg descriptor or no ocgs described, everything is visible".
        if self.ocgs.is_empty() {
            return false;
        }
        // "Avoid infinite recursions" (pdf_cycle).
        if depth > 32 {
            return false;
        }
        let raw = if ocg.is_name() { lookup(ocg.to_name()) } else { ocg.clone() };
        if raw.is_null() {
            return false;
        }
        let dict = doc.resolve(&raw).unwrap_or(Object::Null);
        let typ = doc.resolve_get(&dict, "Type").unwrap_or(Object::Null);
        match typ.to_name() {
            b"OCG" => {
                // By default an OCG is visible unless explicitly hidden;
                // "Deliberately do NOT resolve here" -- compare references.
                let default_hidden = self
                    .ocgs
                    .iter()
                    .find(|(g, _)| same_obj(g, &raw))
                    .is_some_and(|(_, on)| !on);
                let perform_intent_check = match &self.intent {
                    None => false,
                    Some(i) => !(i.is_array() && i.array_len() == 0),
                };
                if perform_intent_check {
                    let intent = doc.resolve_get(&dict, "Intent").unwrap_or(Object::Null);
                    if intent.is_name() {
                        if !self.intents_include(doc, intent.to_name()) {
                            return true;
                        }
                    } else if intent.is_array() {
                        let matched = (0..intent.array_len()).any(|i| {
                            intent
                                .array_get(i)
                                .and_then(|o| doc.resolve(o).ok())
                                .is_some_and(|o| self.intents_include(doc, o.to_name()))
                        });
                        if !matched {
                            return true;
                        }
                    } else if !self.intents_include(doc, b"View") {
                        return true;
                    }
                }
                let usage_dict = doc.resolve_get(&dict, "Usage").unwrap_or(Object::Null);
                if !usage_dict.is_dict() {
                    return default_hidden;
                }
                let u = doc.resolve_get(&usage_dict, usage).unwrap_or(Object::Null);
                let es = doc.resolve_get(&u, &format!("{usage}State")).unwrap_or(Object::Null);
                if es.to_name() == b"OFF" {
                    return true;
                }
                default_hidden
            }
            b"OCMD" => {
                // FIXME in MuPDF too: a /VE expression counts as visible.
                if doc.resolve_get(&dict, "VE").is_ok_and(|v| v.is_array()) {
                    return false;
                }
                // combine: bit 0 = AND, bit 1 = "true means Off".
                let combine = match doc.resolve_get(&dict, "P").map(|o| o.to_name().to_vec()) {
                    Ok(n) if n == b"AllOn" => 1,
                    Ok(n) if n == b"AnyOff" => 2,
                    Ok(n) if n == b"AllOff" => 3,
                    _ => 0, // AnyOn
                };
                let groups = dict.dict_gets("OCGs").cloned().unwrap_or(Object::Null);
                let groups_resolved = doc.resolve(&groups).unwrap_or(Object::Null);
                let mut on = combine & 1 != 0;
                if groups_resolved.is_array() {
                    for i in 0..groups_resolved.array_len() {
                        let Some(g) = groups_resolved.array_get(i) else { continue };
                        let mut hidden = self.hidden_imp(doc, g, lookup, usage, depth + 1);
                        if combine & 1 == 0 {
                            hidden = !hidden;
                        }
                        if combine & 2 != 0 {
                            on &= hidden;
                        } else {
                            on |= hidden;
                        }
                    }
                } else {
                    on = self.hidden_imp(doc, &groups, lookup, usage, depth + 1);
                    if combine & 1 == 0 {
                        on = !on;
                    }
                }
                !on
            }
            // "No idea what sort of object this is - be visible".
            _ => false,
        }
    }

    // MuPDF: ocg_intents_include (pdf-layer.c:584)
    fn intents_include(&self, doc: &PdfDocument, name: &[u8]) -> bool {
        if name == b"All" {
            return true;
        }
        let Some(intent) = &self.intent else { return name == b"View" };
        let intent = doc.resolve(intent).unwrap_or(Object::Null);
        if intent.is_name() {
            let i = intent.to_name();
            return i == b"All" || i == name;
        }
        if !intent.is_array() {
            return false;
        }
        (0..intent.array_len()).any(|k| {
            intent
                .array_get(k)
                .and_then(|o| doc.resolve(o).ok())
                .is_some_and(|o| o.to_name() == b"All" || o.to_name() == name)
        })
    }
}

// MuPDF: pdf_objcmp on unresolved objects -- two references are the same
// object when their numbers and generations match.
fn same_obj(a: &Object, b: &Object) -> bool {
    match (a, b) {
        (Object::Ref { num: n1, generation: g1 }, Object::Ref { num: n2, generation: g2 }) => n1 == n2 && g1 == g2,
        _ => a == b,
    }
}

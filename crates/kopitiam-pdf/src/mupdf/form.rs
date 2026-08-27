//! AcroForm form fields: reading them, and filling them in.
//!
//! Ported from MuPDF `source/pdf/pdf-form.c` and `include/mupdf/pdf/form.h`
//! (`pdf_first_widget`/`pdf_next_widget`, `pdf_widget_type`,
//! `pdf_field_value`, `pdf_set_field_value`, `pdf_toggle_widget`,
//! `pdf_update_widget`) plus the widget branches of
//! `source/pdf/pdf-appearance.c` (commit 5fe54ce, AGPL-3.0, © Artifex).
//!
//! # No JavaScript engine is required
//!
//! PDF forms can carry trigger events and calculation scripts, and MuPDF ships
//! a whole JS layer for them (`source/pdf/pdf-js.c`, ~1350 lines). That looked
//! like a hard blocker for the Pure Rust Core. It is not:
//! `pdf_set_field_value` takes an explicit `ignore_trigger_events` flag
//! (`form.h:178`), and setting it skips scripting entirely. So read/set/toggle/
//! regenerate/save is reachable with **no JS at all**. The honest limitation to
//! state plainly: a field whose value is *computed* by script will not
//! recalculate.
//!
//! # Displaying filled forms already works
//!
//! Widgets are annotations (`/Subtype /Widget`), so [`super::annot_run`]
//! already renders their `/AP` streams, including upstream's rule that a widget
//! displays only with both `/FT` and `/T`. Confirmed on real files. This module
//! is therefore about the **write** side (gh-80).
//!
//! # Scope decisions made in this port
//!
//! * **Checkbox/radio and text fields only.** Combobox/Listbox are classified
//!   correctly by [`page_form_fields`], but [`set_field_value`] refuses to set
//!   them (`FieldKind::Combobox`/`FieldKind::Listbox` -> `Err`) rather than
//!   half-implement choice-widget appearance generation
//!   (`pdf-appearance.c:2922`'s `pdf_write_ch_widget_appearance`, with its own
//!   `/Opt`/`/TI` option-list handling). A future release can pick this up.
//! * **Radio/checkbox groups get full `/Parent`-chain group semantics**
//!   (siblings forced to `Off`, `/V` written at the group head) because the
//!   task spec calls it out explicitly. **Text fields do not**: a text field
//!   split across several sibling widgets sharing one `/T` (rare -- e.g. the
//!   same address line repeated on two pages) only has the *one* widget
//!   [`FormField::obj_num`] identifies updated; siblings keep their old value
//!   and appearance until a full viewer recalculates. Simpler, and covers the
//!   overwhelming common one-widget-per-field case exactly.
//! * **No real base-14 AFM metrics.** Auto-sizing (`0 Tf` in `/DA`) and
//!   quadding both need to *measure* text at a given font/size
//!   (`pdf-appearance.c`'s `measure_string`, backed by `fz_new_base14_font`'s
//!   embedded AFM widths). This crate has not ported those tables (see
//!   `font.rs`'s module docs -- only embedded-font `/Widths`/`/W` are
//!   implemented, "FreeType glyph widths" i.e. base-14 metrics are explicitly
//!   *not* ported there either). [`estimate_string_width_em`] substitutes a
//!   crude average-character-width heuristic: exact for Courier (genuinely
//!   monospaced at 600/1000 em, no table needed), approximate for everything
//!   else. Good enough to keep auto-sized text inside its box; not
//!   pixel-accurate. Porting the standard-14 AFM widths is a reasonable
//!   follow-up if exact layout ever matters.
//! * **No rotation, no `/MK` background/border colour, no rich text (`/RC`),
//!   no comb-field cells.** The regenerated text-field appearance is
//!   deliberately the plain case: clip to `/BBox`, `/DA` font+size+colour,
//!   `/Q` quadding, optional `\n`-split + word-wrapped multiline. `pdf-appearance.c`'s
//!   `write_variable_text` does considerably more (word-wrap, shrink-to-fit
//!   measurement, the `FZ_ENABLE_HTML_ENGINE` rich-text branch); none of that
//!   is ported here.
//! * **Non-ASCII text renders as `?`.** The synthesised base-14 font resource
//!   declares `/Encoding /WinAnsiEncoding`, but `encodings.rs` deliberately
//!   does not carry a unicode-to-WinAnsi *reverse* table (see that module's
//!   "Not ported" section), so [`encode_content_string`] cannot look up a code
//!   for an arbitrary character. ASCII passes through exactly; anything else
//!   becomes a literal `?` glyph. The stored `/V` itself is not affected --
//!   only what the regenerated appearance stream can *draw* -- so the correct
//!   Unicode value always round-trips even though its on-screen rendering
//!   (from our own writer) may not.
//! * **`/Q` is read non-inheritably**, matching upstream's own
//!   `pdf_annot_quadding` (`pdf-annot.c:2352`), which reads
//!   `pdf_dict_get(ctx, annot->obj, PDF_NAME(Q))` directly rather than walking
//!   `/Parent` -- even though PDF 32000-1 §12.7.3.3 lists `/Q` as an
//!   inheritable field attribute. This looks like an upstream simplification,
//!   not a deliberate spec reading; it is reproduced here for fidelity rather
//!   than silently "fixed", and is worth revisiting if it ever causes a
//!   visibly wrong quadding.
//! * **`/NeedAppearances` is never set.** The task explicitly makes real
//!   regeneration the goal and `/NeedAppearances` only a documented fallback;
//!   this port always attempts real regeneration and propagates an `Err`
//!   rather than falling back to the flag, which most non-Acrobat readers
//!   ignore anyway. If a genuinely unsupported case surfaces later, setting
//!   `/NeedAppearances` as an explicit last resort is the natural extension
//!   point -- it is just not implemented now.
//! * **Legacy `/Border` array is not read.** Border width comes from `/BS/W`
//!   (default `1.0`) only; the older `/Border [h v w]` array form
//!   (PDF 32000-1 §12.5.4, largely superseded by `/BS`) is not consulted.

use std::collections::{HashMap, HashSet};

use super::encodings::pdf_doc_unicode;
use super::error::{Error, Result};
use super::geometry::Rect;
use super::object::Object;
use super::write::{NewObject, incremental_update, next_object_number};
use super::xref::PdfDocument;

/// Widget kind — mirrors MuPDF's `pdf_widget_type` (`form.h:30`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldKind {
    Text,
    Checkbox,
    Radio,
    Combobox,
    Listbox,
    Button,
    Signature,
    Unknown,
}

/// One form field on a page.
pub struct FormField {
    /// The widget annotation's indirect object number — the edit handle.
    pub obj_num: i32,
    pub page_index: usize,
    pub kind: FieldKind,
    /// Fully-qualified field name (`/T`, joined through `/Parent`).
    pub name: String,
    /// Current value as text (`/V`); for a checkbox, its state name.
    pub value: String,
    /// `/Rect` in default user space, normalised.
    pub rect: Rect,
    /// `/Ff` bit 1 — a read-only field must not be edited.
    pub read_only: bool,
    /// For checkbox/radio: the "on" state name from `/AP` `/N` (often `Yes`),
    /// i.e. what `/AS` must become to tick it. `Off` is always the other one.
    pub on_state: Option<String>,
    /// `/Ff` bit 13 (`PDF_TX_FIELD_IS_MULTILINE`, `form.h:135`): this text
    /// field accepts more than one line.
    ///
    /// Exposed because a *viewer* cannot infer it — `/Ff` is inheritable
    /// through `/Parent`, so a widget commonly does not carry the flag itself,
    /// and without this a UI has no way to know it should offer a multi-line
    /// editor rather than a single-line one. Always `false` for non-text kinds.
    pub multiline: bool,
}

// ---------------------------------------------------------------------------
// Field-flag bits (`/Ff`) -- `include/mupdf/pdf/form.h` `pdf_field_flags`
// (`:126`-`:151`). C shifts are 0-based; the doc-comment numbers are the
// PDF-spec bit *positions* (1-based, PDF 32000-1 Table 221/226/227), i.e.
// `spec_bit = shift + 1`.
// ---------------------------------------------------------------------------

/// `PDF_FIELD_IS_READ_ONLY = 1` (`form.h:129`) -- spec bit **1**, every field.
const FF_READ_ONLY: i64 = 1 << 0;
/// `PDF_TX_FIELD_IS_MULTILINE = 1 << 12` (`form.h:135`) -- spec bit **13**,
/// text fields.
const FF_TX_MULTILINE: i64 = 1 << 12;
/// `PDF_BTN_FIELD_IS_NO_TOGGLE_TO_OFF = 1 << 14` (`form.h:141`) -- spec bit
/// **15**, radio buttons: once on, a plain toggle click may not turn it off.
const FF_BTN_NO_TOGGLE_TO_OFF: i64 = 1 << 14;
/// `PDF_BTN_FIELD_IS_RADIO = 1 << 15` (`form.h:142`) -- spec bit **16**.
const FF_BTN_RADIO: i64 = 1 << 15;
/// `PDF_BTN_FIELD_IS_PUSHBUTTON = 1 << 16` (`form.h:143`) -- spec bit **17**.
const FF_BTN_PUSHBUTTON: i64 = 1 << 16;
/// `PDF_CH_FIELD_IS_COMBO = 1 << 17` (`form.h:146`) -- spec bit **18**, choice
/// fields: combo box vs. list box.
const FF_CH_COMBO: i64 = 1 << 17;

/// Cap on `/Parent` chain walks (name qualification, `/Ff`/`/FT`/`/DA`
/// inheritance, group-head search, and the checkbox/radio `/Kids` recursion) --
/// a cycle guard, not a real-world limit. Mirrors `annot_run.rs`'s private
/// `MAX_PARENT_DEPTH` of the same value; duplicated here because that
/// constant (and the inheritance-walk helper built on it) is private to its
/// own module.
const MAX_PARENT_DEPTH: usize = 64;

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// Whether the document has an `/AcroForm` at all — what a viewer uses to
/// decide whether to offer a forms mode.
pub fn has_acroform(doc: &PdfDocument) -> bool {
    let Ok(catalog) = doc.catalog() else {
        return false;
    };
    matches!(doc.resolve_get(&catalog, "AcroForm"), Ok(af) if af.is_dict())
}

/// Every form field on `page_index`.
///
/// Walks the page's `/Annots` for `/Subtype /Widget` entries. `/FT`, `/Ff`,
/// `/T`, `/V` and `/DA` are all inheritable through `/Parent`
/// (PDF 32000-1 §12.7.3.3) -- a widget frequently carries none of them itself,
/// relying entirely on an ancestor field dict. Getting that wrong is exactly
/// how every field ends up reading as [`FieldKind::Unknown`] with an empty
/// name, so every lookup below goes through [`dict_get_inheritable`].
///
/// A widget whose `/FT` cannot be resolved at all (through any ancestor) is
/// not a form field in any usable sense and is skipped, matching the display
/// rule `super::annot_run` already applies (`pdf-run.c`'s "Widgets only get
/// displayed if they have both a T and a FT flag").
pub fn page_form_fields(doc: &PdfDocument, page_index: usize) -> Vec<FormField> {
    let Ok(page) = doc.page(page_index) else {
        return Vec::new();
    };
    let page = page.clone();
    let Ok(annots) = doc.resolve_get(&page, "Annots") else {
        return Vec::new();
    };
    if !annots.is_array() {
        return Vec::new();
    }

    let mut out = Vec::new();
    for i in 0..annots.array_len() {
        let Some(Object::Ref { num, .. }) = annots.array_get(i) else {
            continue;
        };
        let num = *num;
        let Ok(widget) = doc.resolve(&Object::new_indirect(num as i64, 0)) else {
            continue;
        };
        if !widget.is_dict() {
            continue;
        }
        let subtype = doc.resolve_get(&widget, "Subtype").unwrap_or(Object::Null);
        if subtype.to_name() != b"Widget" {
            continue;
        }
        let Some(ft) = dict_get_inheritable(doc, &widget, "FT") else {
            continue;
        };
        let Some(rect) = read_rect(doc, &widget, "Rect") else {
            continue;
        };

        let ff = field_flags(doc, &widget);
        let read_only = ff & FF_READ_ONLY != 0;
        let kind = match ft.to_name() {
            b"Btn" => {
                if ff & FF_BTN_PUSHBUTTON != 0 {
                    FieldKind::Button
                } else if ff & FF_BTN_RADIO != 0 {
                    FieldKind::Radio
                } else {
                    FieldKind::Checkbox
                }
            }
            b"Tx" => FieldKind::Text,
            b"Ch" => {
                if ff & FF_CH_COMBO != 0 {
                    FieldKind::Combobox
                } else {
                    FieldKind::Listbox
                }
            }
            b"Sig" => FieldKind::Signature,
            // Upstream (`pdf_field_type`) defaults an unrecognised /FT to
            // PDF_WIDGET_TYPE_BUTTON; this port uses the dedicated Unknown
            // variant instead, since the frozen contract offers one
            // specifically for this case.
            _ => FieldKind::Unknown,
        };

        let name = build_qualified_name(doc, &widget);
        let value = read_value_string(doc, &widget);
        let on_state = if matches!(kind, FieldKind::Checkbox | FieldKind::Radio) {
            read_on_state(doc, &widget).map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
        } else {
            None
        };

        // Read from the *inheritable* /Ff, same as `kind` above: a widget
        // frequently carries no /Ff of its own and inherits the flag from its
        // field parent, so reading it non-inheritably would report every
        // multiline field as single-line.
        let multiline = matches!(kind, FieldKind::Text) && ff & FF_TX_MULTILINE != 0;

        out.push(FormField {
            obj_num: num,
            page_index,
            kind,
            name,
            value,
            rect,
            read_only,
            on_state,
            multiline,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// Set `field`'s value and regenerate its appearance, returning the complete
/// new file bytes. Trigger events are always ignored (see the module docs).
///
/// Checkbox/radio accept only `"Off"` or the field's own
/// [`FormField::on_state`] (case-sensitive, exact match) — a checkbox has
/// exactly two legal states and nothing else means anything.
/// Combobox/Listbox refuse unconditionally (see the module's "Scope
/// decisions"). Button/Signature/Unknown have no settable text value at all.
pub fn set_field_value(doc: &PdfDocument, field: &FormField, value: &str) -> Result<Vec<u8>> {
    if field.read_only {
        return Err(Error::argument(format!(
            "field '{}' is read-only",
            field.name
        )));
    }
    match field.kind {
        FieldKind::Checkbox | FieldKind::Radio => {
            let val: Vec<u8> = if value == "Off" {
                b"Off".to_vec()
            } else if field.on_state.as_deref() == Some(value) {
                value.as_bytes().to_vec()
            } else {
                return Err(Error::argument(format!(
                    "checkbox/radio field '{}' only accepts \"Off\" or its on-state{}",
                    field.name,
                    field
                        .on_state
                        .as_deref()
                        .map(|s| format!(" (\"{s}\")"))
                        .unwrap_or_default(),
                )));
            };
            apply_checkbox_value(doc, field.obj_num, &val)
        }
        FieldKind::Text => set_text_field_value(doc, field, value),
        FieldKind::Combobox | FieldKind::Listbox => Err(Error::unsupported(format!(
            "field '{}' is a {:?}: combobox/listbox value setting is not implemented in this \
             release (scope decision -- see form.rs module docs); page_form_fields still reports \
             its current /V",
            field.name, field.kind
        ))),
        FieldKind::Button | FieldKind::Signature | FieldKind::Unknown => {
            Err(Error::argument(format!(
                "field '{}' ({:?}) has no settable text value",
                field.name, field.kind
            )))
        }
    }
}

/// Flip a checkbox or radio between its on-state and `Off`, returning the
/// complete new file bytes.
///
/// Ported from `toggle_check_box` (`pdf-form.c:548`): the decision (which way
/// to flip, and whether a radio button with `PDF_BTN_FIELD_IS_NO_TOGGLE_TO_OFF`
/// set refuses to turn itself off) looks only at `field`'s own `/AS`, but the
/// *write* fans out through [`apply_checkbox_value`] to the whole group.
pub fn toggle_checkbox(doc: &PdfDocument, field: &FormField) -> Result<Vec<u8>> {
    if field.read_only {
        return Err(Error::argument(format!(
            "field '{}' is read-only",
            field.name
        )));
    }
    if !matches!(field.kind, FieldKind::Checkbox | FieldKind::Radio) {
        return Err(Error::argument(format!(
            "toggle_checkbox only applies to Checkbox/Radio fields, not {:?}",
            field.kind
        )));
    }

    let widget = doc.resolve(&Object::new_indirect(field.obj_num as i64, 0))?;
    if !widget.is_dict() {
        return Err(Error::format(format!(
            "widget {} 0 R is not a dict",
            field.obj_num
        )));
    }

    let ff = field_flags(doc, &widget);
    let is_radio = matches!(field.kind, FieldKind::Radio);
    let no_toggle_off = ff & FF_BTN_NO_TOGGLE_TO_OFF != 0;

    let as_val = doc.resolve_get(&widget, "AS").unwrap_or(Object::Null);
    let ticked = as_val.is_name() && as_val.to_name() != b"Off";

    let val: Vec<u8> = if ticked {
        if is_radio && no_toggle_off {
            // "TODO: check V value as well as or instead of AS?" (upstream's
            // own comment) -- a no-toggle-to-off radio that is already on
            // stays on. Not an error: nothing was requested that this field
            // refuses: `pdf_toggle_widget` just returns without marking
            // anything changed. Mirror that as a genuine no-op file.
            return Ok(doc.raw_bytes().to_vec());
        }
        b"Off".to_vec()
    } else {
        field
            .on_state
            .clone()
            .map(|s| s.into_bytes())
            .unwrap_or_else(|| b"Yes".to_vec())
    };

    apply_checkbox_value(doc, field.obj_num, &val)
}

/// Set a checkbox/radio widget (and its whole group, per `/Parent`) to
/// `val` (a state name: `b"Off"` or an on-state).
///
/// Ported from `set_check_grp`/`set_check`/`find_head_of_field_group`
/// (`pdf-form.c:448`-`483`, `:122`). The group head is the nearest node,
/// walking self-then-`/Parent`, that itself carries a `/T` (`hofg`,
/// `pdf-form.c:518`); `/V` is written there. Every terminal (`/Kids`-less)
/// descendant of that head then gets its *own* `/AS` set to `val` if its own
/// on-state (from its own `/AP`) equals `val`, else `Off` -- this is exactly
/// what forces every other button in a radio group to `Off` when one is
/// selected.
fn apply_checkbox_value(doc: &PdfDocument, widget_num: i32, val: &[u8]) -> Result<Vec<u8>> {
    let grp_num = find_group_head(doc, widget_num);
    let mut grp = doc.resolve(&Object::new_indirect(grp_num as i64, 0))?;
    if !grp.is_dict() {
        return Err(Error::format(format!(
            "field group head {grp_num} 0 R is not a dict"
        )));
    }
    grp.dict_put("V", Object::new_name(val));

    let mut updates: HashMap<i32, Object> = HashMap::new();
    updates.insert(grp_num, grp);
    let mut seen = HashSet::new();
    set_check_grp(doc, grp_num, val, &mut updates, &mut seen)?;

    let list: Vec<(i32, NewObject)> = updates
        .into_iter()
        .map(|(num, obj)| (num, NewObject::Plain(obj)))
        .collect();
    incremental_update(doc, &list)
}

/// Recursive `/Kids` walk setting `/AS` on every leaf. `updates` doubles as
/// both the output accumulator and the "already rewritten" lookup, so a node
/// that is simultaneously the group head (has `/V` freshly set) and a leaf
/// (no `/Kids`, so it also needs `/AS`) gets both writes merged onto the same
/// dict instead of two competing updates for one object number -- the
/// ordinary non-grouped checkbox case, where the widget *is* its own group.
fn set_check_grp(
    doc: &PdfDocument,
    node_num: i32,
    val: &[u8],
    updates: &mut HashMap<i32, Object>,
    seen: &mut HashSet<i32>,
) -> Result<()> {
    if !seen.insert(node_num) {
        return Ok(()); // cycle guard: a malformed /Kids loop stops here.
    }
    let node = match updates.get(&node_num) {
        Some(o) => o.clone(),
        None => doc.resolve(&Object::new_indirect(node_num as i64, 0))?,
    };
    if !node.is_dict() {
        return Ok(());
    }

    let kids = match node.dict_gets("Kids") {
        Some(k) => doc.resolve(k)?,
        None => Object::Null,
    };
    if kids.is_array() {
        for i in 0..kids.array_len() {
            if let Some(Object::Ref { num, .. }) = kids.array_get(i) {
                set_check_grp(doc, *num, val, updates, seen)?;
            }
        }
        return Ok(());
    }

    // Leaf: this node is itself a widget, carrying its own /AP and /AS.
    let on = read_on_state(doc, &node).unwrap_or_else(|| b"Yes".to_vec());
    let new_as = if on == val {
        val.to_vec()
    } else {
        b"Off".to_vec()
    };
    let mut new_node = node;
    new_node.dict_put("AS", Object::new_name(new_as));
    updates.insert(node_num, new_node);
    Ok(())
}

/// Regenerate a text field's value and appearance. See the module's "Scope
/// decisions" for exactly what the generated appearance does and does not do.
///
/// Ported from the `/Tx` branch of `pdf_write_widget_appearance`
/// (`pdf-appearance.c:2947`) dispatching to `pdf_write_tx_widget_appearance`
/// (`:2687`), simplified to the plain single/multi-line case (no rich text, no
/// comb cells, no rotation).
fn set_text_field_value(doc: &PdfDocument, field: &FormField, value: &str) -> Result<Vec<u8>> {
    let widget = doc.resolve(&Object::new_indirect(field.obj_num as i64, 0))?;
    if !widget.is_dict() {
        return Err(Error::format(format!(
            "widget {} 0 R is not a dict",
            field.obj_num
        )));
    }

    // /DA: inheritable on the field, else the AcroForm's own default, else
    // the built-in "Helv 12 black" pdf_parse_default_appearance_unmapped
    // falls back to (pdf-annot.c:4163).
    let da_bytes: Vec<u8> = if let Some(da) = dict_get_inheritable(doc, &widget, "DA") {
        da.to_string_bytes().to_vec()
    } else {
        doc.catalog()
            .ok()
            .and_then(|cat| doc.resolve_get(&cat, "AcroForm").ok())
            .and_then(|af| af.dict_gets("DA").cloned())
            .and_then(|da| doc.resolve(&da).ok())
            .map(|o| o.to_string_bytes().to_vec())
            .unwrap_or_default()
    };
    let (font_name, mut size, color) = parse_da(&da_bytes);
    let abbrev = map_font_abbrev(&font_name);

    let ff = field_flags(doc, &widget);
    let multiline = ff & FF_TX_MULTILINE != 0;
    // Non-inheritable, matching upstream pdf_annot_quadding -- see the module
    // docs' "Scope decisions" note on this.
    let q_raw = doc
        .resolve_get(&widget, "Q")
        .unwrap_or(Object::Null)
        .to_int();
    let q = if (0..=2).contains(&q_raw) {
        q_raw as i32
    } else {
        0
    };

    let bw = border_width(doc, &widget);
    let rect = field.rect;
    let w = (rect.x1 - rect.x0).max(0.0);
    let h = (rect.y1 - rect.y0).max(0.0);
    let avail_w = (w - 2.0 * bw).max(0.0);
    let avail_h = (h - 2.0 * bw).max(0.0);

    // Auto-size ("0 Tf"): pdf-appearance.c:2119-2129's algorithm, with
    // measure_string's real AFM lookup replaced by
    // estimate_string_width_em's approximation (see module docs).
    if size <= 0.0 {
        if multiline {
            size = 12.0;
        } else {
            let ms = estimate_string_width_em(value, abbrev);
            size = if ms > 0.0 { avail_w / ms } else { 12.0 };
            if size > avail_h {
                size = avail_h;
            }
            if size <= 0.0 {
                size = 12.0;
            }
        }
    }

    let content = build_tx_content(value, abbrev, size, &color, q, multiline, bw, w, h);

    let font_num = next_object_number(doc);
    let ap_num = font_num + 1;

    let mut font_res = Object::new_dict();
    font_res.dict_put(abbrev, Object::new_indirect(font_num as i64, 0));
    let mut resources = Object::new_dict();
    resources.dict_put("Font", font_res);

    let mut stream_dict = Object::new_dict();
    stream_dict.dict_put("Type", Object::new_name("XObject"));
    stream_dict.dict_put("Subtype", Object::new_name("Form"));
    stream_dict.dict_put("FormType", Object::new_int(1));
    stream_dict.dict_put(
        "BBox",
        Object::Array(vec![
            Object::new_real(0.0),
            Object::new_real(0.0),
            Object::new_real(w as f64),
            Object::new_real(h as f64),
        ]),
    );
    stream_dict.dict_put("Resources", resources);

    let font_dict = build_base14_font_dict(abbrev);

    // Preserve any existing /AP sub-entries (/D, /R) and only overwrite /N --
    // a text field practically never has them, but there is no reason to
    // drop them if some other producer put them there.
    let existing_ap = doc.resolve_get(&widget, "AP").unwrap_or(Object::Null);
    let mut ap_dict = if existing_ap.is_dict() {
        existing_ap
    } else {
        Object::new_dict()
    };
    ap_dict.dict_put("N", Object::new_indirect(ap_num as i64, 0));

    let mut new_widget = widget;
    new_widget.dict_put("V", encode_text_string(value));
    new_widget.dict_put("AP", ap_dict);

    let updates = vec![
        (field.obj_num, NewObject::Plain(new_widget)),
        (font_num, NewObject::Plain(font_dict)),
        (
            ap_num,
            NewObject::Stream {
                dict: stream_dict,
                data: content,
            },
        ),
    ];
    incremental_update(doc, &updates)
}

// ---------------------------------------------------------------------------
// Inheritance / field-tree helpers
// ---------------------------------------------------------------------------

/// Look up `key` on `obj`, walking `/Parent` when absent -- the field-tree
/// inheritance rule of PDF 32000-1 §12.7.3.3 (`/FT`, `/Ff`, `/T`... wait, `/T`
/// is deliberately *not* inherited, see [`build_qualified_name`] -- but
/// `/FT`, `/Ff`, `/V`, `/DA` are).
///
/// Own copy of `annot_run.rs`'s private `dict_get_inheritable` (same
/// algorithm, including the [`MAX_PARENT_DEPTH`] cycle guard) -- duplicated
/// rather than shared because that helper is private to its module.
fn dict_get_inheritable(doc: &PdfDocument, obj: &Object, key: &str) -> Option<Object> {
    let mut current = obj.clone();
    for _ in 0..MAX_PARENT_DEPTH {
        if let Some(raw) = current.dict_gets(key) {
            let resolved = doc.resolve(raw).ok()?;
            return if resolved.is_null() {
                None
            } else {
                Some(resolved)
            };
        }
        let parent_raw = current.dict_gets("Parent")?.clone();
        current = doc.resolve(&parent_raw).ok()?;
        if !current.is_dict() {
            return None;
        }
    }
    None
}

/// `/Ff`, inherited, defaulting to `0` (no flags) when absent or malformed.
fn field_flags(doc: &PdfDocument, widget: &Object) -> i64 {
    dict_get_inheritable(doc, widget, "Ff")
        .map(|o| o.to_int())
        .unwrap_or(0)
}

/// The fully-qualified field name: each ancestor's own `/T` (only where that
/// level actually defines one -- `/T` is **not** inherited, each level either
/// contributes its own segment or none at all), joined root-to-leaf with `.`
/// (PDF 32000-1 §12.7.3.2, `pdf_load_field_name` / `lookup_field_sub`'s
/// dotted-path matching, `pdf-form.c:181-190`).
fn build_qualified_name(doc: &PdfDocument, widget: &Object) -> String {
    let mut segments: Vec<String> = Vec::new();
    let mut current = widget.clone();
    for _ in 0..MAX_PARENT_DEPTH {
        if let Some(t) = current.dict_gets("T")
            && let Ok(resolved) = doc.resolve(t)
        {
            segments.push(obj_text(&resolved));
        }
        match current.dict_gets("Parent") {
            Some(p) => match doc.resolve(p) {
                Ok(parent) if parent.is_dict() => current = parent,
                _ => break,
            },
            None => break,
        }
    }
    segments.reverse();
    segments.join(".")
}

/// `/V`, inherited, decoded to text. Empty string if absent.
fn read_value_string(doc: &PdfDocument, widget: &Object) -> String {
    match dict_get_inheritable(doc, widget, "V") {
        Some(v) => obj_text(&v),
        None => String::new(),
    }
}

/// The nearest node, walking self-then-`/Parent`, that itself carries a `/T`
/// -- `find_head_of_field_group`/`hofg` (`pdf-form.c:118-125`). Falls back to
/// `start_num` itself when no ancestor (including `start_num`) has one,
/// matching upstream's `if (!grp) grp = field;`.
fn find_group_head(doc: &PdfDocument, start_num: i32) -> i32 {
    let mut current_num = start_num;
    for _ in 0..MAX_PARENT_DEPTH {
        let Ok(node) = doc.resolve(&Object::new_indirect(current_num as i64, 0)) else {
            return start_num;
        };
        if !node.is_dict() {
            return start_num;
        }
        if node.dict_gets("T").is_some() {
            return current_num;
        }
        match node.dict_gets("Parent") {
            Some(Object::Ref { num, .. }) => current_num = *num,
            _ => return start_num,
        }
    }
    start_num
}

/// The on-state name for a checkbox/radio widget: the non-`Off` key of its
/// own `/AP` `/N` dict-of-states, falling back to `/AP` `/D`, else `None`.
///
/// Ported from `pdf_button_field_on_state`/`find_on_state`
/// (`pdf-form.c:506-523`), minus the final `PDF_NAME(Yes)` fallback -- that
/// default belongs to the *caller* (each call site here decides what "no
/// appearance found at all" should mean), not to this read.
fn read_on_state(doc: &PdfDocument, widget: &Object) -> Option<Vec<u8>> {
    let ap = doc.resolve_get(widget, "AP").ok()?;
    if !ap.is_dict() {
        return None;
    }
    if let Some(n) = ap.dict_gets("N")
        && let Ok(n_res) = doc.resolve(n)
        && let Some(on) = find_on_state_key(&n_res)
    {
        return Some(on);
    }
    if let Some(d) = ap.dict_gets("D")
        && let Ok(d_res) = doc.resolve(d)
        && let Some(on) = find_on_state_key(&d_res)
    {
        return Some(on);
    }
    None
}

/// The first dict key that isn't `Off` -- `find_on_state` (`pdf-form.c:506`).
fn find_on_state_key(dict: &Object) -> Option<Vec<u8>> {
    match dict {
        Object::Dict(items) => items
            .iter()
            .find(|(k, _)| k.as_slice() != b"Off")
            .map(|(k, _)| k.clone()),
        _ => None,
    }
}

/// `/BS` `/W` (border width), default `1.0` when absent or malformed --
/// `pdf_annot_border_width`'s effective default via `pdf_write_border_appearance`
/// (`pdf-appearance.c:198`).
fn border_width(doc: &PdfDocument, widget: &Object) -> f32 {
    let bs = doc.resolve_get(widget, "BS").unwrap_or(Object::Null);
    if bs.is_dict() {
        let w = doc.resolve_get(&bs, "W").unwrap_or(Object::Null);
        if w.is_number() {
            return w.to_real() as f32;
        }
    }
    1.0
}

/// A normalised `/Rect`-shaped 4-element numeric array, or `None` if `key` is
/// absent, not length 4, or holds a non-numeric entry.
///
// MuPDF: pdf_to_rect (pdf-parse.c) -- reads 4 numbers and normalises with
// min/max so an out-of-order array is still usable. Own copy of the same
// algorithm `annot_run.rs` uses privately for `/Rect`/`/BBox`.
fn read_rect(doc: &PdfDocument, dict: &Object, key: &str) -> Option<Rect> {
    let arr = doc.resolve_get(dict, key).ok()?;
    if arr.array_len() != 4 {
        return None;
    }
    let mut v = [0f32; 4];
    for (i, slot) in v.iter_mut().enumerate() {
        let item = doc.resolve(arr.array_get(i)?).ok()?;
        if !item.is_number() {
            return None;
        }
        *slot = item.to_real() as f32;
    }
    let (a, b, c, d) = (v[0], v[1], v[2], v[3]);
    Some(Rect::new(a.min(c), b.min(d), a.max(c), b.max(d)))
}

// ---------------------------------------------------------------------------
// PDF text-string <-> Rust String
// ---------------------------------------------------------------------------

/// A `/Name` or `/String` object's text, or `""` for anything else.
fn obj_text(obj: &Object) -> String {
    match obj {
        Object::String(bytes) => decode_pdf_text_string(bytes),
        Object::Name(bytes) => String::from_utf8_lossy(bytes).into_owned(),
        _ => String::new(),
    }
}

/// Decode a PDF text string per `pdf_new_utf8_from_pdf_string`
/// (`pdf-parse.c:330-`): a UTF-16BE or UTF-16LE BOM, a UTF-8 BOM, unmarked
/// valid UTF-8 (a common real-world producer bug this crate reads leniently),
/// or else `PDFDocEncoding` byte-by-byte.
///
/// Not ported: the embedded-language-code skip (`skip_language_code_utf16be`
/// &c.) -- a rare PDF 2.0 feature for tagging spans of a string with a
/// language, unrelated to the text content itself. Its absence means a string
/// using it decodes with a few extra stray characters rather than failing
/// outright, which is an acceptable degradation for a feature this obscure.
fn decode_pdf_text_string(bytes: &[u8]) -> String {
    if bytes.len() >= 2 && bytes[0] == 0xFE && bytes[1] == 0xFF {
        return utf16_units_to_string(
            bytes[2..]
                .chunks_exact(2)
                .map(|c| u16::from_be_bytes([c[0], c[1]])),
        );
    }
    if bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] == 0xFE {
        return utf16_units_to_string(
            bytes[2..]
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]])),
        );
    }
    if bytes.len() >= 3 && bytes[0] == 0xEF && bytes[1] == 0xBB && bytes[2] == 0xBF {
        return String::from_utf8_lossy(&bytes[3..]).into_owned();
    }
    if let Ok(s) = std::str::from_utf8(bytes) {
        return s.to_string();
    }
    bytes.iter().filter_map(|&b| pdf_doc_unicode(b)).collect()
}

fn utf16_units_to_string(units: impl Iterator<Item = u16>) -> String {
    String::from_utf16_lossy(&units.collect::<Vec<_>>())
}

/// Encode `s` as a PDF text-string object per `pdf_new_text_string`
/// (`pdf-parse.c:540-549`): plain ASCII bytes if every byte is `< 128`, else
/// UTF-16BE with a leading `\xFE\xFF` BOM (`pdf_new_text_string_utf16be`).
fn encode_text_string(s: &str) -> Object {
    if s.bytes().all(|b| b < 128) {
        Object::new_string(s.as_bytes().to_vec())
    } else {
        let mut buf = vec![0xFEu8, 0xFF];
        for unit in s.encode_utf16() {
            buf.extend_from_slice(&unit.to_be_bytes());
        }
        Object::new_string(buf)
    }
}

// ---------------------------------------------------------------------------
// /DA parsing and text-field appearance generation
// ---------------------------------------------------------------------------

/// Parse a `/DA` default-appearance string into (font resource name, size,
/// colour components). `size <= 0` means "auto" (PDF's `0 Tf` convention).
/// Colour is `[]` (n=0, meaning "default black") unless a `g`/`rg`/`k`
/// operator was seen.
///
/// Ported from `pdf_parse_default_appearance_unmapped` (`pdf-annot.c:4163`):
/// a tiny whitespace-tokenised operand-stack interpreter over exactly four
/// operators (`Tf`, `g`, `rg`, `k`), defaulting to `Helv`/`12`/black when `da`
/// is empty or unparsable. This re-expresses the same token loop; it does not
/// reproduce the C version's fixed 4-slot overflow behaviour byte-for-byte
/// (irrelevant here since every operator only ever reads its own fixed
/// leading operand count).
fn parse_da(da: &[u8]) -> (String, f32, Vec<f32>) {
    let s = String::from_utf8_lossy(da);
    let mut font_name = String::from("Helv");
    let mut size = 12.0f32;
    let mut color: Vec<f32> = Vec::new();
    let mut stack: Vec<f32> = Vec::new();

    for tok in s
        .split(|c: char| c.is_whitespace())
        .filter(|t| !t.is_empty())
    {
        if let Some(name) = tok.strip_prefix('/') {
            font_name = name.to_string();
        } else if tok == "Tf" {
            size = stack.first().copied().unwrap_or(size);
            stack.clear();
        } else if tok == "g" {
            color = vec![stack.first().copied().unwrap_or(0.0)];
            stack.clear();
        } else if tok == "rg" {
            color = stack.iter().take(3).copied().collect();
            stack.clear();
        } else if tok == "k" {
            color = stack.iter().take(4).copied().collect();
            stack.clear();
        } else if let Ok(v) = tok.parse::<f32>() {
            if stack.len() < 4 {
                stack.push(v);
            }
        } else {
            stack.clear();
        }
    }
    (font_name, size, color)
}

/// Map a `/DA` font name to one of the five names `pdf_parse_default_appearance`
/// recognises (`pdf-annot.c:4226-4237`); anything else defaults to `Helv`,
/// exactly like upstream.
fn map_font_abbrev(name: &str) -> &'static str {
    match name {
        "Cour" => "Cour",
        "TiRo" => "TiRo",
        "Symb" => "Symb",
        "ZaDb" => "ZaDb",
        _ => "Helv",
    }
}

/// The standard-14 `/BaseFont` name for a mapped `/DA` abbreviation --
/// `full_font_name` (`pdf-appearance.c:2090`).
fn full_font_name(abbrev: &str) -> &'static str {
    match abbrev {
        "Cour" => "Courier",
        "TiRo" => "Times-Roman",
        "Symb" => "Symbol",
        "ZaDb" => "ZapfDingbats",
        _ => "Helvetica",
    }
}

/// A minimal standard-14 `/Type1` font resource for `abbrev`. Symbol and
/// ZapfDingbats carry their own built-in symbol encoding and must **not** get
/// `/Encoding /WinAnsiEncoding` -- doing so would remap their glyphs onto the
/// wrong codes.
fn build_base14_font_dict(abbrev: &str) -> Object {
    let mut dict = Object::new_dict();
    dict.dict_put("Type", Object::new_name("Font"));
    dict.dict_put("Subtype", Object::new_name("Type1"));
    dict.dict_put("BaseFont", Object::new_name(full_font_name(abbrev)));
    if abbrev != "Symb" && abbrev != "ZaDb" {
        dict.dict_put("Encoding", Object::new_name("WinAnsiEncoding"));
    }
    dict
}

/// Crude average-character-width estimate, in units of em (multiply by font
/// size for a text-space width) -- the stand-in for `measure_string`'s real
/// AFM lookup (see the module docs' "No real base-14 AFM metrics" note).
/// Courier's `0.6` is exact (it is genuinely monospaced at 600/1000 em);
/// the others are rough proportional-font averages, not real metrics.
fn estimate_string_width_em(text: &str, abbrev: &str) -> f32 {
    let avg = match abbrev {
        "Cour" => 0.6,
        "TiRo" => 0.45,
        _ => 0.5,
    };
    text.chars().count() as f32 * avg
}

/// `g`/`rg`/`k` content-stream operator for `color`, defaulting to `0 g`
/// (black) for any length other than 1/3/4 -- `write_color0`
/// (`pdf-appearance.c:220`).
fn color_op(color: &[f32]) -> String {
    match color.len() {
        1 => format!("{} g", fmt_num(color[0])),
        3 => format!(
            "{} {} {} rg",
            fmt_num(color[0]),
            fmt_num(color[1]),
            fmt_num(color[2])
        ),
        4 => format!(
            "{} {} {} {} k",
            fmt_num(color[0]),
            fmt_num(color[1]),
            fmt_num(color[2]),
            fmt_num(color[3])
        ),
        _ => "0 g".to_string(),
    }
}

/// Format an `f32` for content-stream syntax: an integral value as a bare
/// integer, else up to three decimal places with trailing zeros trimmed.
/// Non-finite input degrades to `"0"` rather than emitting invalid syntax.
fn fmt_num(v: f32) -> String {
    if !v.is_finite() {
        return "0".to_string();
    }
    if v == v.trunc() {
        return format!("{}", v as i64);
    }
    let mut s = format!("{v:.3}");
    while s.ends_with('0') {
        s.pop();
    }
    if s.ends_with('.') {
        s.pop();
    }
    s
}

/// Escape `text` as a content-stream literal-string `Tj` operand: `(`, `)`,
/// `\`, newline and carriage return get their mnemonic escapes
/// (`fmt_str`/`fmt_str_out`, `pdf-object.c:3389`); any non-ASCII character
/// becomes a literal `?` (see the module docs' "Non-ASCII text renders as
/// `?`" note -- there is no unicode-to-WinAnsi table to consult).
fn encode_content_string(text: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len() + 2);
    out.push(b'(');
    for c in text.chars() {
        let byte = if c.is_ascii() { c as u8 } else { b'?' };
        match byte {
            b'(' => out.extend_from_slice(b"\\("),
            b')' => out.extend_from_slice(b"\\)"),
            b'\\' => out.extend_from_slice(b"\\\\"),
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            _ => out.push(byte),
        }
    }
    out.push(b')');
    out
}

/// Break `value` into the lines a multiline text field should actually draw.
///
/// Splits on explicit `\n` first (an author-entered newline is a hard break and
/// must be honoured), then **word-wraps** each resulting paragraph to `max_w`
/// text-space units using the same width estimate the auto-sizer uses.
///
/// Word wrap matters more than it looks. A multiline field exists precisely
/// because its content is longer than one line, so without wrapping the very
/// first sentence typed runs straight out of the widget's box and is clipped
/// away by the `re W n` above -- the text is stored correctly in `/V` but
/// invisible in the appearance, which is the most confusing failure mode
/// available (the value is *there*, the field looks empty or truncated).
///
/// A word longer than the whole line is broken mid-word rather than dropped or
/// allowed to overflow: losing characters would be worse than an ugly break,
/// and `char_indices` keeps the split on a character boundary so multi-byte
/// text cannot panic here.
fn wrap_lines(value: &str, abbrev: &str, size: f32, max_w: f32) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    // A non-positive width (a degenerate widget) has no meaningful wrap point;
    // fall back to hard breaks only rather than looping forever.
    let usable = max_w;
    for para in value.split('\n') {
        if usable <= 0.0 || para.is_empty() {
            out.push(para.to_string());
            continue;
        }
        let mut line = String::new();
        for word in para.split(' ') {
            let candidate = if line.is_empty() {
                word.to_string()
            } else {
                format!("{line} {word}")
            };
            if estimate_string_width_em(&candidate, abbrev) * size <= usable || line.is_empty() {
                // Still fits, or nothing to fall back to yet.
                line = candidate;
            } else {
                out.push(std::mem::take(&mut line));
                line = word.to_string();
            }
            // A single word wider than the whole line: break it mid-word so no
            // characters are silently lost.
            while estimate_string_width_em(&line, abbrev) * size > usable
                && line.chars().count() > 1
            {
                let mut cut = line.len();
                for (i, _) in line.char_indices() {
                    if i > 0 && estimate_string_width_em(&line[..i], abbrev) * size > usable {
                        cut = i;
                        break;
                    }
                }
                let rest = line.split_off(cut.max(1).min(line.len()));
                if rest.is_empty() {
                    break;
                }
                out.push(std::mem::take(&mut line));
                line = rest;
            }
        }
        out.push(line);
    }
    out
}

/// Build the `/Tx BMC … EMC` appearance content stream: clip to the widget's
/// own box (inset by the border width), then draw `value` in `abbrev`/`size`/
/// `color`, honouring `/Q` quadding (0 left, 1 centre, 2 right) and, when
/// `multiline`, `\n`-splitting plus word wrap (see [`wrap_lines`]).
///
/// Simplified re-expression of `pdf_write_tx_widget_appearance` +
/// `write_variable_text` (`pdf-appearance.c:2687`, `:2101`) -- see the module
/// docs' "Scope decisions" for exactly what is dropped (rotation, `/MK`
/// background/border colour, rich text, comb cells, exact metrics).
#[allow(clippy::too_many_arguments)]
fn build_tx_content(
    value: &str,
    abbrev: &str,
    size: f32,
    color: &[f32],
    q: i32,
    multiline: bool,
    bw: f32,
    w: f32,
    h: f32,
) -> Vec<u8> {
    let bw = bw.max(0.0);
    let cw = (w - 2.0 * bw).max(0.0);
    let ch = (h - 2.0 * bw).max(0.0);

    let mut out = Vec::new();
    out.extend_from_slice(b"/Tx BMC\nq\n");
    out.extend_from_slice(
        format!(
            "{} {} {} {} re\nW\nn\n",
            fmt_num(bw),
            fmt_num(bw),
            fmt_num(cw),
            fmt_num(ch)
        )
        .as_bytes(),
    );
    out.extend_from_slice(b"BT\n");
    out.extend_from_slice(color_op(color).as_bytes());
    out.push(b'\n');
    out.extend_from_slice(format!("/{abbrev} {} Tf\n", fmt_num(size)).as_bytes());

    // Multiline fields wrap to the widget's usable width; single-line fields
    // are drawn as one run (PDF single-line text fields do not wrap, and an
    // embedded newline in one is not a line break).
    let wrapped: Vec<String>;
    let lines: Vec<&str> = if multiline {
        wrapped = wrap_lines(value, abbrev, size, cw);
        wrapped.iter().map(|s| s.as_str()).collect()
    } else {
        vec![value]
    };
    // pdf-appearance.c:2119's baseline/lineheight ratios: 1.116 for
    // multiline, 0.8 for single-line (the non-comb, non-multiline branch).
    let baseline = if multiline { size * 1.116 } else { size * 0.8 };
    let line_height = size * 1.116;

    for (i, line) in lines.iter().enumerate() {
        let tw = estimate_string_width_em(line, abbrev) * size;
        let tx = match q {
            1 => ((cw - tw) / 2.0).max(0.0),
            2 => (cw - tw).max(0.0),
            _ => 0.0,
        };
        let ty = if multiline {
            ch - baseline - (i as f32) * line_height
        } else {
            // Vertically centred: ty = (h - size) / 2, y = h - baseline - ty.
            ch - baseline - (ch - size) / 2.0
        };
        out.extend_from_slice(
            format!("1 0 0 1 {} {} Tm\n", fmt_num(bw + tx), fmt_num(bw + ty)).as_bytes(),
        );
        out.extend_from_slice(&encode_content_string(line));
        out.extend_from_slice(b" Tj\n");
    }

    out.extend_from_slice(b"ET\nQ\nEMC");
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build `<< dict_fields /Length N >>\nstream\n<content>\nendstream`, with
    /// `/Length` computed from `content`'s real byte length. (Same convention
    /// as `annot_run.rs`'s private helper of the same shape.)
    fn stream_obj(dict_fields: &str, content: &[u8]) -> Vec<u8> {
        let mut body =
            format!("<< {dict_fields} /Length {} >>\nstream\n", content.len()).into_bytes();
        body.extend_from_slice(content);
        body.extend_from_slice(b"\nendstream");
        body
    }

    /// Build a PDF with objects 1.. from `bodies` (each wrapped `N 0 obj …
    /// endobj`) and a classic xref table. (Same convention as
    /// `annot_run.rs`/`page_run.rs`'s private helper of the same shape.)
    fn build_pdf(bodies: &[Vec<u8>]) -> PdfDocument {
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
        PdfDocument::open(pdf).unwrap()
    }

    /// The fixture this whole test module shares: one page carrying --
    ///
    /// 1: Catalog, 2: Pages, 3: Page
    /// 4: text field "Name", value "Alice", DA "/Helv 0 Tf 0 g"
    /// 5: checkbox "Agree", AP/N {Yes:6, Off:7}, AS Off
    /// 6/7: checkbox's Yes/Off appearance streams
    /// 8: radio group parent, /FT /Btn, radio Ff bit set, /T "Choice", Kids [9,11]
    /// 9: radio kid A, AP/N {"1":10,"Off":...}, AS Off, /Parent 8
    /// 10: radio kid A's "1" appearance stream (Off reuses 7)
    /// 11: radio kid B, AP/N {"2":12,"Off":...}, AS Off, /Parent 8
    /// 12: radio kid B's "2" appearance stream
    /// 13: read-only text field "Locked", Ff read-only bit set
    /// 14: inheritance case -- FT/T live on 15 (a /Parent), this widget has
    ///     neither itself
    /// 15: the parent field dict for 14: /FT /Tx /T "Inherited"
    /// 16: a Ch (choice) combobox, Ff combo bit set
    /// 17: a Ch (choice) listbox, no combo bit
    fn build_fixture() -> PdfDocument {
        build_pdf(&[
            b"<< /Type /Catalog /Pages 2 0 R /AcroForm << /Fields [4 0 R 5 0 R 8 0 R 13 0 R 16 0 R 17 0 R] >> >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 400] \
              /Annots [4 0 R 5 0 R 9 0 R 11 0 R 13 0 R 14 0 R 16 0 R 17 0 R] >>"
                .to_vec(),
            // 4: text field
            b"<< /Type /Annot /Subtype /Widget /FT /Tx /T (Name) /V (Alice) /Rect [10 10 210 30] \
              /DA (/Helv 0 Tf 0 g) >>"
                .to_vec(),
            // 5: checkbox
            b"<< /Type /Annot /Subtype /Widget /FT /Btn /T (Agree) /Rect [10 40 30 60] \
              /AS /Off /V /Off /AP << /N << /Yes 6 0 R /Off 7 0 R >> >> >>"
                .to_vec(),
            stream_obj("/Type /XObject /Subtype /Form /BBox [0 0 20 20]", b"1 g 0 0 20 20 re f"),
            stream_obj("/Type /XObject /Subtype /Form /BBox [0 0 20 20]", b"0 g 0 0 20 20 re f"),
            // 8: radio group parent
            b"<< /FT /Btn /Ff 32768 /T (Choice) /Kids [9 0 R 11 0 R] >>".to_vec(),
            // 9: radio kid A
            b"<< /Type /Annot /Subtype /Widget /Parent 8 0 R /Rect [10 70 30 90] \
              /AS /Off /AP << /N << /1 10 0 R /Off 7 0 R >> >> >>"
                .to_vec(),
            stream_obj("/Type /XObject /Subtype /Form /BBox [0 0 20 20]", b"0 0 1 rg 0 0 20 20 re f"),
            // 11: radio kid B
            b"<< /Type /Annot /Subtype /Widget /Parent 8 0 R /Rect [40 70 60 90] \
              /AS /Off /AP << /N << /2 12 0 R /Off 7 0 R >> >> >>"
                .to_vec(),
            stream_obj("/Type /XObject /Subtype /Form /BBox [0 0 20 20]", b"0 1 0 rg 0 0 20 20 re f"),
            // 13: read-only text field (Ff bit 1 = 1)
            b"<< /Type /Annot /Subtype /Widget /FT /Tx /T (Locked) /Ff 1 /Rect [10 100 210 120] \
              /V (frozen) >>"
                .to_vec(),
            // 14: FT/T live on parent 15, not here
            b"<< /Type /Annot /Subtype /Widget /Parent 15 0 R /Rect [10 130 210 150] >>".to_vec(),
            // 15: the parent field dict (not itself an annotation / on the page)
            b"<< /FT /Tx /T (Inherited) /V (from-parent) >>".to_vec(),
            // 16: combobox (Ch, combo bit 1<<17 = 131072)
            b"<< /Type /Annot /Subtype /Widget /FT /Ch /Ff 131072 /T (Pick) /Rect [10 160 210 180] \
              /V (One) >>"
                .to_vec(),
            // 17: listbox (Ch, no combo bit)
            b"<< /Type /Annot /Subtype /Widget /FT /Ch /T (PickMany) /Rect [10 190 210 210] \
              /V (Two) >>"
                .to_vec(),
        ])
    }

    // -----------------------------------------------------------------------
    // Multiline word wrap
    // -----------------------------------------------------------------------

    /// A paragraph longer than the field is broken into several lines rather
    /// than running off the side and being clipped away by the appearance's
    /// `re W n`. Without this the value is stored correctly in `/V` but looks
    /// missing on screen -- the most confusing failure mode available.
    #[test]
    fn wrap_lines_breaks_a_long_paragraph() {
        let text = "the quick brown fox jumps over the lazy dog again and again";
        let lines = wrap_lines(text, "Helv", 10.0, 100.0);
        assert!(lines.len() > 1, "long text was not wrapped: {lines:?}");
        for line in &lines {
            assert!(
                estimate_string_width_em(line, "Helv") * 10.0 <= 100.0 + 1e-3,
                "wrapped line still overflows the field: {line:?}"
            );
        }
        // Wrapping must not lose or invent words.
        let rejoined = lines.join(" ");
        assert_eq!(
            rejoined.split_whitespace().collect::<Vec<_>>(),
            text.split_whitespace().collect::<Vec<_>>()
        );
    }

    /// An author-typed newline is a hard break and survives wrapping.
    #[test]
    fn wrap_lines_honours_explicit_newlines() {
        let lines = wrap_lines("alpha\nbeta", "Helv", 10.0, 1000.0);
        assert_eq!(lines, vec!["alpha".to_string(), "beta".to_string()]);
    }

    /// A single word wider than the whole line is broken mid-word. Losing
    /// characters would be worse than an ugly break, and the split must stay on
    /// a character boundary so multi-byte text cannot panic.
    #[test]
    fn wrap_lines_breaks_an_overlong_word_without_losing_characters() {
        let word = "supercalifragilisticexpialidocious";
        let lines = wrap_lines(word, "Helv", 10.0, 40.0);
        assert!(lines.len() > 1, "overlong word was not broken: {lines:?}");
        assert_eq!(lines.concat(), word, "characters were lost while breaking");
    }

    /// Degenerate widths must not hang or panic — a zero-width widget has no
    /// meaningful wrap point.
    #[test]
    fn wrap_lines_survives_degenerate_width() {
        assert_eq!(
            wrap_lines("hello world", "Helv", 10.0, 0.0),
            vec!["hello world".to_string()]
        );
        assert_eq!(wrap_lines("", "Helv", 10.0, 100.0), vec!["".to_string()]);
        let _ = wrap_lines("日本語のテキストです", "Helv", 10.0, 5.0);
    }

    /// The generated appearance actually contains multiple positioned runs for
    /// a wrapped multiline field — i.e. the wrap reaches the content stream,
    /// not just the helper.
    #[test]
    fn multiline_appearance_emits_several_positioned_lines() {
        let long = "the quick brown fox jumps over the lazy dog again and again and again";
        let content = build_tx_content(long, "Helv", 10.0, &[0.0], 0, true, 1.0, 120.0, 60.0);
        let text = String::from_utf8_lossy(&content);
        let tm_count = text.matches(" Tm\n").count();
        assert!(
            tm_count > 1,
            "multiline appearance drew only {tm_count} line(s)"
        );
        // Single-line fields must NOT wrap: PDF single-line text does not.
        let single = build_tx_content(long, "Helv", 10.0, &[0.0], 0, false, 1.0, 120.0, 60.0);
        assert_eq!(String::from_utf8_lossy(&single).matches(" Tm\n").count(), 1);
    }

    fn field_named<'a>(fields: &'a [FormField], name: &str) -> &'a FormField {
        fields.iter().find(|f| f.name == name).unwrap_or_else(|| {
            panic!(
                "no field named {name:?} in {:?}",
                fields.iter().map(|f| &f.name).collect::<Vec<_>>()
            )
        })
    }

    #[test]
    fn has_acroform_detects_the_dict() {
        let doc = build_fixture();
        assert!(has_acroform(&doc));

        let bare = build_pdf(&[
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>".to_vec(),
        ]);
        assert!(!has_acroform(&bare));
    }

    #[test]
    fn page_form_fields_classifies_every_kind() {
        let doc = build_fixture();
        let fields = page_form_fields(&doc, 0);

        let name = field_named(&fields, "Name");
        assert_eq!(name.kind, FieldKind::Text);
        assert_eq!(name.value, "Alice");
        assert!(!name.read_only);
        assert_eq!(name.obj_num, 4);

        let agree = field_named(&fields, "Agree");
        assert_eq!(agree.kind, FieldKind::Checkbox);
        assert_eq!(agree.on_state.as_deref(), Some("Yes"));
        assert_eq!(agree.value, "Off");

        // Both radio kids share the group's qualified name "Choice" (their
        // own dicts have no /T; it lives on the /Parent).
        let radios: Vec<&FormField> = fields.iter().filter(|f| f.name == "Choice").collect();
        assert_eq!(radios.len(), 2, "both radio kids should classify as Choice");
        for r in &radios {
            assert_eq!(r.kind, FieldKind::Radio);
        }
        let kid_a = radios.iter().find(|f| f.obj_num == 9).unwrap();
        let kid_b = radios.iter().find(|f| f.obj_num == 11).unwrap();
        assert_eq!(kid_a.on_state.as_deref(), Some("1"));
        assert_eq!(kid_b.on_state.as_deref(), Some("2"));

        let locked = field_named(&fields, "Locked");
        assert_eq!(locked.kind, FieldKind::Text);
        assert!(locked.read_only);

        // Inheritance case: FT/T only exist on obj 15 (the /Parent), not on
        // the widget (obj 14) itself.
        let inherited = field_named(&fields, "Inherited");
        assert_eq!(inherited.kind, FieldKind::Text);
        assert_eq!(inherited.value, "from-parent");
        assert_eq!(inherited.obj_num, 14);

        let combo = field_named(&fields, "Pick");
        assert_eq!(combo.kind, FieldKind::Combobox);
        let listbox = field_named(&fields, "PickMany");
        assert_eq!(listbox.kind, FieldKind::Listbox);
    }

    #[test]
    fn read_only_field_refuses_set_and_toggle() {
        let doc = build_fixture();
        let fields = page_form_fields(&doc, 0);
        let locked = field_named(&fields, "Locked");
        assert!(set_field_value(&doc, locked, "new value").is_err());

        // A read-only checkbox refuses toggle_checkbox too -- reuse the
        // Agree checkbox but with the read-only bit forced on synthetically.
        let mut ro_checkbox = FormField {
            obj_num: locked.obj_num,
            page_index: 0,
            kind: FieldKind::Checkbox,
            name: "Fake".to_string(),
            value: "Off".to_string(),
            rect: locked.rect,
            read_only: true,
            on_state: Some("Yes".to_string()),
            multiline: false,
        };
        assert!(toggle_checkbox(&doc, &ro_checkbox).is_err());
        ro_checkbox.read_only = false;
        // (Not asserting Ok here -- obj 13 isn't actually a checkbox dict;
        // this only exercises the read_only gate itself, which fires before
        // any dict inspection.)
    }

    #[test]
    fn combobox_and_listbox_refuse_set_field_value() {
        let doc = build_fixture();
        let fields = page_form_fields(&doc, 0);
        let combo = field_named(&fields, "Pick");
        let listbox = field_named(&fields, "PickMany");
        assert!(set_field_value(&doc, combo, "Two").is_err());
        assert!(set_field_value(&doc, listbox, "Two").is_err());
    }

    #[test]
    fn checkbox_rejects_arbitrary_values() {
        let doc = build_fixture();
        let fields = page_form_fields(&doc, 0);
        let agree = field_named(&fields, "Agree");
        assert!(set_field_value(&doc, agree, "Maybe").is_err());
    }

    #[test]
    fn toggle_checkbox_flips_as_and_v_together() {
        let doc = build_fixture();
        let fields = page_form_fields(&doc, 0);
        let agree = field_named(&fields, "Agree");
        assert_eq!(agree.value, "Off");

        let new_bytes = toggle_checkbox(&doc, agree).expect("toggle should succeed");
        let doc2 = PdfDocument::open(new_bytes).expect("round-trip reopen");
        let fields2 = page_form_fields(&doc2, 0);
        let agree2 = field_named(&fields2, "Agree");
        assert_eq!(agree2.value, "Yes", "/V should now read the on-state");

        // /AS must have moved too, not just /V -- read it directly off the
        // widget dict to be sure both keys agree.
        let widget = doc2
            .resolve(&Object::new_indirect(agree2.obj_num as i64, 0))
            .unwrap();
        let as_val = doc2.resolve_get(&widget, "AS").unwrap();
        assert_eq!(as_val.to_name(), b"Yes");

        // Toggling again flips back to Off.
        let back_bytes = toggle_checkbox(&doc2, agree2).expect("second toggle");
        let doc3 = PdfDocument::open(back_bytes).unwrap();
        let fields3 = page_form_fields(&doc3, 0);
        assert_eq!(field_named(&fields3, "Agree").value, "Off");
    }

    #[test]
    fn toggle_checkbox_radio_group_forces_siblings_off() {
        let doc = build_fixture();
        let fields = page_form_fields(&doc, 0);
        let kid_b = fields.iter().find(|f| f.obj_num == 11).unwrap();

        let new_bytes = toggle_checkbox(&doc, kid_b).expect("toggle radio kid B on");
        let doc2 = PdfDocument::open(new_bytes).unwrap();

        // /V is a field-group-level attribute, inherited from the shared
        // /Parent -- both siblings report the *same* value string ("2") once
        // it's set, since that is what inheritance means. Which button is
        // actually ticked on screen is a per-widget property: its own /AS.
        // So the "forced off" assertion must read /AS directly, not `.value`.
        let widget_a = doc2.resolve(&Object::new_indirect(9, 0)).unwrap();
        let widget_b = doc2.resolve(&Object::new_indirect(11, 0)).unwrap();
        let as_a = doc2.resolve_get(&widget_a, "AS").unwrap();
        let as_b = doc2.resolve_get(&widget_b, "AS").unwrap();
        assert_eq!(
            as_a.to_name(),
            b"Off",
            "the other radio kid's /AS must be forced Off"
        );
        assert_eq!(
            as_b.to_name(),
            b"2",
            "the toggled kid's /AS becomes its own on-state"
        );

        let fields2 = page_form_fields(&doc2, 0);
        let a2 = fields2.iter().find(|f| f.obj_num == 9).unwrap();
        let b2 = fields2.iter().find(|f| f.obj_num == 11).unwrap();
        assert_eq!(a2.value, "2", "inherited /V is shared across the group");
        assert_eq!(b2.value, "2", "inherited /V is shared across the group");

        // The group head's /V (obj 8, not itself a page annotation) carries
        // the selection.
        let grp = doc2.resolve(&Object::new_indirect(8, 0)).unwrap();
        assert_eq!(grp.dict_gets("V").unwrap().to_name(), b"2");
    }

    #[test]
    fn set_field_value_text_round_trips_and_regenerates_appearance() {
        let doc = build_fixture();
        let fields = page_form_fields(&doc, 0);
        let name = field_named(&fields, "Name");

        let new_bytes =
            set_field_value(&doc, name, "Bob Tan").expect("text field set should succeed");
        let doc2 = PdfDocument::open(new_bytes).expect("round-trip reopen");
        let fields2 = page_form_fields(&doc2, 0);
        let name2 = field_named(&fields2, "Name");
        assert_eq!(name2.value, "Bob Tan");

        // The regenerated /AP /N must be a real, openable stream containing
        // the expected content-stream operators.
        let widget = doc2
            .resolve(&Object::new_indirect(name2.obj_num as i64, 0))
            .unwrap();
        let ap = doc2.resolve_get(&widget, "AP").unwrap();
        let n = ap.dict_gets("N").unwrap();
        let content = doc2.open_stream(n).unwrap();
        let text = String::from_utf8_lossy(&content);
        assert!(text.contains("/Tx BMC"), "{text}");
        assert!(text.contains("Tf"), "{text}");
        assert!(text.contains("(Bob Tan) Tj"), "{text}");
        assert!(text.contains("EMC"), "{text}");
    }

    #[test]
    fn set_field_value_text_encodes_non_ascii_as_utf16be() {
        let doc = build_fixture();
        let fields = page_form_fields(&doc, 0);
        let name = field_named(&fields, "Name");

        let new_bytes = set_field_value(&doc, name, "Café").unwrap();
        let doc2 = PdfDocument::open(new_bytes).unwrap();
        let fields2 = page_form_fields(&doc2, 0);
        assert_eq!(field_named(&fields2, "Name").value, "Café");
    }

    #[test]
    fn da_parsing_reads_font_size_and_color() {
        let (font, size, color) = parse_da(b"/Helv 10 Tf 0.2 0.4 0.6 rg");
        assert_eq!(font, "Helv");
        assert_eq!(size, 10.0);
        assert_eq!(color, vec![0.2, 0.4, 0.6]);

        let (font, size, color) = parse_da(b"");
        assert_eq!(font, "Helv");
        assert_eq!(size, 12.0);
        assert!(color.is_empty());

        let (font, size, _color) = parse_da(b"/Cour 0 Tf 0 g");
        assert_eq!(font, "Cour");
        assert_eq!(size, 0.0, "0 Tf means auto-size");
    }

    /// The strongest evidence a filled field actually works: not merely that
    /// our own parser can read back what our own writer produced (every test
    /// above does that), but that an **independent** reader -- poppler, which
    /// has never seen this crate's code -- opens the filled file and
    /// extracts the exact value we set, straight out of the regenerated
    /// `/AP` `/N` appearance stream's `Tj` text. Skips (does not fail) if
    /// `pdftotext` is not on `PATH`, matching `write.rs`'s own
    /// `poppler_can_render_the_updated_file` convention.
    #[test]
    fn poppler_extracts_the_filled_text_field_value() {
        let doc = build_fixture();
        let fields = page_form_fields(&doc, 0);
        let name = field_named(&fields, "Name");

        let filled = set_field_value(&doc, name, "Independent Reader Check")
            .expect("set_field_value should succeed");

        let dir = std::env::temp_dir().join(format!(
            "kopitiam-pdf-form-test-{}-poppler-check",
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&dir);
        let pdf_path = dir.join("filled.pdf");
        std::fs::write(&pdf_path, &filled).expect("write temp pdf");
        let txt_path = dir.join("filled.txt");

        let result = std::process::Command::new("pdftotext")
            .arg(&pdf_path)
            .arg(&txt_path)
            .output();

        match result {
            Ok(output) => {
                assert!(
                    output.status.success(),
                    "pdftotext failed: stdout={:?} stderr={:?}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
                let text = std::fs::read_to_string(&txt_path).expect("read pdftotext output");
                assert!(
                    text.contains("Independent Reader Check"),
                    "poppler's extracted text should contain the filled value; got: {text:?}"
                );
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                eprintln!("skipping poppler cross-check: pdftotext not on PATH ({e})");
            }
            Err(e) => panic!("failed to run pdftotext: {e}"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn text_string_round_trips_ascii_and_unicode() {
        assert_eq!(decode_pdf_text_string(b"hello"), "hello");
        let encoded = encode_text_string("hello");
        assert_eq!(encoded, Object::new_string(b"hello".to_vec()));

        let encoded_unicode = encode_text_string("héllo");
        match &encoded_unicode {
            Object::String(bytes) => {
                assert_eq!(&bytes[..2], &[0xFE, 0xFF]);
            }
            _ => panic!("expected a string object"),
        }
        if let Object::String(bytes) = encoded_unicode {
            assert_eq!(decode_pdf_text_string(&bytes), "héllo");
        }
    }
}

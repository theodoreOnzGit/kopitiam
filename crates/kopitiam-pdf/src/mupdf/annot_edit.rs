//! Creating, deleting and undoing annotations — the write half.
//!
//! [`super::annot_run`]/[`super::annot_appearance`] make annotations *visible*;
//! this makes them *authorable*, so a reader can draw on a page and save.
//!
//! Ported from MuPDF `source/pdf/pdf-annot.c` (`pdf_create_annot`,
//! `pdf_delete_annot`, `pdf_set_annot_ink_list`, `pdf_set_annot_color`,
//! `pdf_set_annot_border_width`) (commit 5fe54ce, AGPL-3.0, © Artifex).
//!
//! # Always write a real `/AP`
//!
//! We can synthesise a missing appearance on read, but when we **write** one we
//! always emit a real `/AP`. Producers that omit it are exactly what made
//! annotations invisible in the first place, and `hayro` — a dependency of this
//! very crate — renders nothing without it.
//!
//! # `/Rect` equals the *widened* appearance box, on purpose
//!
//! [`super::annot_appearance::synthesize_ap`] returns an `SynthAp` whose
//! `rect`/`bbox` are the ink's bounding box expanded by `width + 6` points (see
//! that module's docs). We write the **same** widened rect into the new
//! annotation's own `/Rect`, not the tight polyline bounds. That is not merely
//! consistent bookkeeping: [`super::annot_run`]'s Algorithm 8.1 step
//! (`annot_bbox_to_rect`) maps the appearance form's `/BBox` onto the annot's
//! `/Rect` by an affine scale — if `/Rect` were the *tighter* unwidened box,
//! that step would non-uniformly squash the stroke to fit inside it. Setting
//! both to the same rect makes that mapping the identity, which is required
//! since `/Matrix` here is [`super::geometry::Matrix::IDENTITY`] and `InkList`
//! coordinates already live in default user space (PDF 32000-1:2008 §12.5.6.13).
//!
//! # Finding a page's own object number
//!
//! [`PdfDocument::page`] hands back a page dict flattened with inherited
//! `/MediaBox`/`/CropBox`/`/Rotate`/`/Resources`, with no indirect-reference
//! identity attached (the page tree walk that builds it keeps only the
//! resolved dict, not the `Kids` entry that pointed at it). Writing requires
//! that object number, to supersede it in place. Rather than changing
//! `xref.rs` (owned, at the time of this port, by a concurrently-running
//! agent working on `write.rs`/`form.rs`/`kpdf.rs`), [`locate_page`] repeats
//! the minimal half of that walk — same `/Type`-or-`Kids`-without-`/MediaBox`
//! leaf test as `PdfDocument`'s own `walk_pages` — but keeps the `Kids` array
//! entry (an indirect reference) instead of discarding it, and returns the
//! **unflattened** dict (whatever was actually stored at that object number),
//! since that is what must be written back.

use super::annot_appearance::synthesize_ap;
use super::error::{Error, Result};
use super::geometry::Rect;
use super::object::Object;
use super::write::{self, NewObject};
use super::xref::PdfDocument;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// One continuous pen stroke in **default user space** (PDF points, y up from
/// the page's bottom-left) — the same space as `/Rect` and `/InkList`.
///
/// A UI working in screen pixels must convert first; passing device
/// coordinates puts the ink in the wrong place, and on a rotated page the
/// wrong orientation too.
pub struct InkStroke {
    /// Pen positions in order. A single point is legal and paints a dot.
    pub points: Vec<(f32, f32)>,
}

/// An ink annotation to add.
pub struct InkAnnotSpec {
    /// 0-based page index, matching the rest of this crate.
    pub page_index: usize,
    pub strokes: Vec<InkStroke>,
    /// DeviceRGB, each component 0..=1 (written as `/C`).
    pub color: [f32; 3],
    /// Stroke width in points, written via `/BS` (`pdf_set_annot_border_width`,
    /// `pdf-annot.c:2027`: an explicit `/BS/Type /Border` dict with `/W`) —
    /// the modern form that dict's sibling reader,
    /// [`super::annot_appearance::annot_border_width`], already checks
    /// *first*, before its legacy `/Border` array fallback.
    pub width: f32,
    /// Constant opacity 0..=1 (`/CA`); 1.0 writes none.
    pub opacity: f32,
    /// Optional `/T` author.
    pub author: Option<String>,
}

/// An existing annotation, enough for hit-testing and deletion.
pub struct AnnotRef {
    /// Its indirect object number — the handle [`delete_annot`] takes.
    pub num: i32,
    pub subtype: String,
    /// `/Rect` in default user space, normalised.
    pub rect: Rect,
}

/// Every annotation on `page_index`, with object numbers — what an eraser
/// needs in order to know what it is pointing at.
///
/// An `/Annots` entry that is not an indirect reference (a malformed but
/// theoretically legal direct dict inline in the array) is silently skipped:
/// there is no object number to hand back for it, and [`AnnotRef::num`] is not
/// optional. Any entry that resolves to something other than a dict is
/// likewise skipped. Both failures return fewer annotations rather than
/// panicking or erroring — this is a read-only survey, and a partially
/// malformed page is still worth surveying.
pub fn page_annot_refs(doc: &PdfDocument, page_index: usize) -> Vec<AnnotRef> {
    let Ok(page) = doc.page(page_index) else {
        return Vec::new();
    };
    let Ok(annots) = doc.resolve_get(page, "Annots") else {
        return Vec::new();
    };
    if !annots.is_array() {
        return Vec::new();
    }

    let mut out = Vec::new();
    for i in 0..annots.array_len() {
        let Some(entry) = annots.array_get(i) else {
            continue;
        };
        let Object::Ref { num, .. } = entry else {
            continue;
        };
        let Ok(dict) = doc.resolve(entry) else {
            continue;
        };
        if !dict.is_dict() {
            continue;
        }
        let subtype = dict
            .dict_gets("Subtype")
            .map(|o| String::from_utf8_lossy(o.to_name()).into_owned())
            .unwrap_or_default();
        let rect = rect_from(doc, &dict, "Rect").unwrap_or(Rect::new(0.0, 0.0, 0.0, 0.0));
        out.push(AnnotRef {
            num: *num,
            subtype,
            rect,
        });
    }
    out
}

/// Add an ink annotation, returning the **complete new file bytes**.
///
/// `doc` is not mutated: the caller writes the bytes and reopens. That keeps
/// the read path immutable and makes save-as trivial.
///
/// Emits three objects in a single [`write::incremental_update`] call: the new
/// annotation dict, its `/AP` `/N` form XObject (built by
/// [`synthesize_ap`], never hand-rolled here), and the page dict with the new
/// annotation appended to `/Annots`.
///
/// Errors if `page_index` is out of range, the page has no indirect object
/// identity (see the module docs), or the spec's strokes produce nothing
/// [`synthesize_ap`] considers visible (e.g. no strokes at all, or every
/// stroke empty) — mirroring `synthesize_ap`'s own "nothing to paint, nothing
/// to build" contract rather than writing an invisible annotation.
pub fn add_ink_annot(doc: &PdfDocument, spec: &InkAnnotSpec) -> Result<Vec<u8>> {
    // REFUSE on an encrypted document. We can decrypt but not encrypt
    // (gh-98), so appending a plaintext object here would produce a file that
    // still opens in kpdf -- we would read our own plaintext back through a
    // decryptor that mangles it -- and is unreadable everywhere else. Silent
    // corruption of someone's form is the one outcome worth failing loudly to
    // avoid.
    if doc.is_encrypted() {
        return Err(Error::unsupported(
            "cannot annotate an encrypted PDF -- kopitiam-pdf can decrypt but \
             not yet encrypt, so writing would corrupt the file",
        ));
    }
    let (page_num, page_gen, page_dict) = locate_page(doc, spec.page_index)?;

    let width = if spec.width.is_finite() {
        spec.width
    } else {
        1.0
    };
    let opacity = if spec.opacity.is_finite() {
        spec.opacity
    } else {
        1.0
    };

    let ink_list = ink_list_object(&spec.strokes);

    // A throwaway dict carrying exactly what `synthesize_ap` reads (`/Subtype`,
    // `/InkList`, `/C`, `/BS`/`/Border`, `/CA`) so the appearance-synthesis
    // logic is never duplicated between the read path and this write path.
    let mut probe = Object::new_dict();
    probe.dict_put("Subtype", Object::new_name("Ink"));
    probe.dict_put("InkList", ink_list.clone());
    probe.dict_put("C", color_array(spec.color));
    let mut bs_probe = Object::new_dict();
    bs_probe.dict_put("W", Object::new_real(width as f64));
    probe.dict_put("BS", bs_probe);
    if opacity != 1.0 {
        probe.dict_put("CA", Object::new_real(opacity as f64));
    }

    let ap = synthesize_ap(doc, &probe)
        .ok_or_else(|| Error::argument("ink spec has no visible strokes"))?;

    let next = write::next_object_number(doc);
    let annot_num = next;
    let ap_num = next + 1;

    let mut annot = Object::new_dict();
    annot.dict_put("Type", Object::new_name("Annot"));
    annot.dict_put("Subtype", Object::new_name("Ink"));
    // See the module docs: intentionally the *widened* rect, matching the
    // appearance form's own BBox/Rect below.
    annot.dict_put("Rect", rect_to_array(ap.rect));
    annot.dict_put("InkList", ink_list);
    annot.dict_put("C", color_array(spec.color));
    let mut bs = Object::new_dict();
    bs.dict_put("Type", Object::new_name("Border"));
    bs.dict_put("W", Object::new_real(width as f64));
    annot.dict_put("BS", bs);
    if opacity != 1.0 {
        annot.dict_put("CA", Object::new_real(opacity as f64));
    }
    // PDF_ANNOT_IS_PRINT (pdf-annot.c's pdf_create_annot default flags) = bit
    // position 3 = value 4: printable by default, matching upstream.
    annot.dict_put("F", Object::new_int(4));
    annot.dict_put("P", Object::new_indirect(page_num as i64, page_gen));
    annot.dict_put("M", Object::new_string(pdf_date_now()));
    annot.dict_put("NM", Object::new_string(unique_annot_name(annot_num)));
    if let Some(author) = &spec.author {
        annot.dict_put("T", Object::new_string(author.clone()));
    }
    let mut ap_entry = Object::new_dict();
    ap_entry.dict_put("N", Object::new_indirect(ap_num as i64, 0));
    annot.dict_put("AP", ap_entry);

    let mut ap_dict = Object::new_dict();
    ap_dict.dict_put("Type", Object::new_name("XObject"));
    ap_dict.dict_put("Subtype", Object::new_name("Form"));
    ap_dict.dict_put("BBox", rect_to_array(ap.bbox));
    if !ap.matrix.is_identity() {
        ap_dict.dict_put("Matrix", matrix_to_array(ap.matrix));
    }
    if !ap.resources.is_null() {
        ap_dict.dict_put("Resources", ap.resources);
    }

    // Append to /Annots, resolving an indirect array to a direct copy first
    // (mirroring `pdf_create_annot_raw`'s own `pdf_is_indirect(annot_arr)`
    // branch) so an existing indirectly-stored array is never silently
    // dropped in favour of a fresh empty one.
    let mut updated_page = page_dict.clone();
    let mut annots_arr = match updated_page.dict_gets("Annots").cloned() {
        Some(v) => {
            let resolved = doc.resolve(&v).unwrap_or(Object::Null);
            if resolved.is_array() {
                resolved
            } else {
                Object::new_array()
            }
        }
        None => Object::new_array(),
    };
    annots_arr.array_push(Object::new_indirect(annot_num as i64, 0));
    updated_page.dict_put("Annots", annots_arr);

    let updates = vec![
        (annot_num, NewObject::Plain(annot)),
        (
            ap_num,
            NewObject::Stream {
                dict: ap_dict,
                data: ap.content,
            },
        ),
        (page_num, NewObject::Plain(updated_page)),
    ];
    write::incremental_update(doc, &updates)
}

/// Remove the annotation with object number `annot_num` from `page_index`,
/// returning the complete new file bytes.
///
/// Rewrites the page's `/Annots` without that reference; the annotation
/// object itself is left orphaned rather than freed — normal and correct for
/// an incremental update (nothing later in the file references its number, so
/// it is simply dead weight, never a dangling reference).
///
/// Any `/Annots` entry (there may legitimately be more than one, for a
/// malformed file) whose object number matches is removed — a deliberate
/// widening of upstream `pdf_delete_annot`'s `pdf_array_find` + single delete,
/// since leaving a duplicate behind would be a strictly worse outcome than
/// removing every copy.
///
/// Not ported: upstream also looks up the removed annotation's own `/Popup`
/// and deletes that from `/Annots` too (`pdf-annot.c:1112` area). This module
/// only ever *creates* `/Ink` annotations, which never carry a `/Popup`, so
/// that branch is dead code for anything this port itself authors; a
/// `/Popup` belonging to some other annotation type in a file we did not
/// create would be orphaned instead of removed, which is the same "orphan is
/// safe" contract this function already relies on above.
pub fn delete_annot(doc: &PdfDocument, page_index: usize, annot_num: i32) -> Result<Vec<u8>> {
    let (page_num, _page_gen, page_dict) = locate_page(doc, page_index)?;

    let annots_val = page_dict
        .dict_gets("Annots")
        .cloned()
        .unwrap_or(Object::Null);
    let annots = doc.resolve(&annots_val)?;
    if !annots.is_array() {
        return Err(Error::argument(format!(
            "page {page_index} has no /Annots array"
        )));
    }

    let mut new_annots = Object::new_array();
    let mut found = false;
    for i in 0..annots.array_len() {
        let Some(entry) = annots.array_get(i) else {
            continue;
        };
        if let Object::Ref { num, .. } = entry
            && *num == annot_num
        {
            found = true;
            continue;
        }
        new_annots.array_push(entry.clone());
    }

    if !found {
        return Err(Error::argument(format!(
            "annotation {annot_num} 0 R not found on page {page_index}"
        )));
    }

    let mut updated_page = page_dict.clone();
    updated_page.dict_put("Annots", new_annots);

    let updates = vec![(page_num, NewObject::Plain(updated_page))];
    write::incremental_update(doc, &updates)
}

// ---------------------------------------------------------------------------
// Page-tree lookup (see "Finding a page's own object number" above)
// ---------------------------------------------------------------------------

/// Find the `page_index`-th leaf page: its indirect object number,
/// generation, and unflattened dict.
fn locate_page(doc: &PdfDocument, page_index: usize) -> Result<(i32, i32, Object)> {
    let root = doc.catalog()?;
    let pages_ref = root.dict_gets("Pages").cloned().unwrap_or(Object::Null);
    let mut counter = 0usize;
    let mut visited = HashSet::new();
    walk_for_page(doc, &pages_ref, page_index, &mut counter, &mut visited)?
        .ok_or_else(|| Error::argument(format!("page {page_index} out of range 0..{counter}")))
}

/// Recursive half of [`locate_page`] — same internal/leaf classification as
/// `PdfDocument`'s private `walk_pages`, but threading the leaf's own
/// `Object::Ref` through instead of discarding it.
fn walk_for_page(
    doc: &PdfDocument,
    node_ref: &Object,
    target: usize,
    counter: &mut usize,
    visited: &mut HashSet<i32>,
) -> Result<Option<(i32, i32, Object)>> {
    let node = doc.resolve(node_ref)?;
    if !node.is_dict() {
        return Ok(None);
    }

    let type_name = node.dict_gets("Type").map(|o| o.to_name());
    let has_kids = node.dict_gets("Kids").is_some();
    let is_internal = match type_name {
        Some(b"Pages") => true,
        Some(b"Page") => false,
        _ => has_kids && node.dict_gets("MediaBox").is_none(),
    };

    if is_internal {
        let kids = doc.resolve_get(&node, "Kids")?;
        for i in 0..kids.array_len() {
            let Some(kid_ref) = kids.array_get(i) else {
                continue;
            };
            if let Object::Ref { num, .. } = kid_ref
                && !visited.insert(*num)
            {
                return Err(Error::format("cycle in page tree"));
            }
            if let Some(found) = walk_for_page(doc, kid_ref, target, counter, visited)? {
                return Ok(Some(found));
            }
        }
        Ok(None)
    } else {
        let is_target = *counter == target;
        *counter += 1;
        if !is_target {
            return Ok(None);
        }
        match node_ref {
            Object::Ref { num, generation } => Ok(Some((*num, *generation, node))),
            _ => Err(Error::format(
                "page has no indirect object identity (not referenced via an indirect reference)",
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Object-tree encoders (the inverse of annot_appearance's/annot_run's readers)
// ---------------------------------------------------------------------------

fn rect_to_array(r: Rect) -> Object {
    let mut a = Object::new_array();
    for v in [r.x0, r.y0, r.x1, r.y1] {
        a.array_push(Object::new_real(v as f64));
    }
    a
}

fn matrix_to_array(m: super::geometry::Matrix) -> Object {
    let mut a = Object::new_array();
    for v in [m.a, m.b, m.c, m.d, m.e, m.f] {
        a.array_push(Object::new_real(v as f64));
    }
    a
}

fn color_array(rgb: [f32; 3]) -> Object {
    let mut a = Object::new_array();
    for v in rgb {
        a.array_push(Object::new_real(v as f64));
    }
    a
}

fn ink_list_object(strokes: &[InkStroke]) -> Object {
    let mut ink_list = Object::new_array();
    for stroke in strokes {
        let mut arr = Object::new_array();
        for &(x, y) in &stroke.points {
            arr.array_push(Object::new_real(x as f64));
            arr.array_push(Object::new_real(y as f64));
        }
        ink_list.array_push(arr);
    }
    ink_list
}

/// Read `dict[key]` as a `/Rect`-shaped 4-number array, normalising corner
/// order. `None` if absent, too short, or not a dict. Deliberately a small
/// private duplicate of `annot_run.rs`'s `rect_from` (same shape, same
/// tolerance for indirect numbers) rather than promoting that helper to
/// `pub(crate)` for one extra call site.
fn rect_from(doc: &PdfDocument, dict: &Object, key: &str) -> Option<Rect> {
    let arr = doc.resolve_get(dict, key).ok()?;
    if arr.array_len() < 4 {
        return None;
    }
    let v = |i: usize| -> f32 {
        arr.array_get(i)
            .and_then(|o| doc.resolve(o).ok())
            .map(|o| o.to_real() as f32)
            .unwrap_or(0.0)
    };
    let (a, b, c, d) = (v(0), v(1), v(2), v(3));
    Some(Rect::new(a.min(c), b.min(d), a.max(c), b.max(d)))
}

// ---------------------------------------------------------------------------
// /M and /NM — no date/time or RNG crate needed for either
// ---------------------------------------------------------------------------

/// Process-lifetime counter, purely to break ties if [`unique_annot_name`] is
/// ever called twice within the same clock tick (coarse clocks on some
/// platforms).
static NM_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A `/NM` unique name for a freshly created annotation. `annot_num` alone is
/// already unique within this document (object numbers are never reused
/// within one file), and combining it with the wall-clock time and a
/// per-process counter makes it unique *across* saves/files too, without
/// pulling in a RNG crate for one field.
fn unique_annot_name(annot_num: i32) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seq = NM_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("kopitiam-ink-{annot_num}-{nanos}-{seq}")
}

/// A `/M` value in `D:YYYYMMDDHHmmSS` form, UTC, with no timezone suffix —
/// PDF 32000-1:2008 §7.9.4 treats a date string with no `O HH' mm'` tail as
/// "relationship to UT unknown", which is the honest answer here rather than
/// guessing at the host's offset.
fn pdf_date_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    format_pdf_date(secs)
}

/// `civil_from_days`, Howard Hinnant's well-known constant-time Gregorian
/// calendar algorithm (public domain,
/// <http://howardhinnant.github.io/date_algorithms.html#civil_from_days>),
/// used here instead of adding a date/time crate dependency for this one call
/// site. Valid for any `i64` day count; the division-by-`146097`
/// era computation is written to work correctly under truncating (not
/// flooring) integer division for negative inputs too, exactly the way the
/// algorithm is designed — Rust's `/` on signed integers truncates, matching
/// the C division it was written against.
fn format_pdf_date(unix_secs: i64) -> String {
    let days = unix_secs.div_euclid(86_400);
    let secs_of_day = unix_secs.rem_euclid(86_400);
    let (h, mi, s) = (
        secs_of_day / 3600,
        (secs_of_day % 3600) / 60,
        secs_of_day % 60,
    );

    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };

    format!("D:{y:04}{m:02}{d:02}{h:02}{mi:02}{s:02}")
}

/// Undo/redo over a document's byte history.
///
/// Cheap by construction: every edit here is an **append** (see
/// [`super::write`]), so a previous state is just a prefix of the current
/// bytes. History therefore stores lengths, not copies — undoing is truncation,
/// and the earlier cross-reference section and `%%EOF` are still intact at that
/// offset. A snapshot-per-edit design would instead cost a full copy of a
/// possibly-100MB file on every pen stroke.
///
/// # The invariant this relies on, stated loudly
///
/// This holds **only** while every edit function in this crate produces its
/// new bytes by reopening the document from [`EditHistory::current`] and
/// appending (which is exactly what [`add_ink_annot`], [`delete_annot`] and
/// `super::form`'s field-editing functions do via
/// `super::write::incremental_update`). Concretely: any `bytes` passed to
/// [`EditHistory::push`] must satisfy
/// `bytes[..current.len()] == current` where `current` is
/// [`EditHistory::current`] *before* the call. If a future edit path ever
/// rewrites bytes in place instead of appending, this whole scheme — lengths
/// standing in for copies — silently produces wrong undo/redo output, and
/// must be revisited together with that change.
pub struct EditHistory {
    /// The most recent (and longest-ever-held) complete file. Untouched by
    /// `undo`/`redo` (those only move `cursor`); replaced wholesale by `push`
    /// after a `push`-after-`undo`, since at that point the bytes past the
    /// truncation point genuinely differ from what used to be there.
    bytes: Vec<u8>,
    /// Byte length of the file at each history step, oldest first. Per the
    /// struct-level invariant, `bytes[..lengths[i]]` is always a complete,
    /// valid PDF for step `i`.
    lengths: Vec<usize>,
    /// Index into `lengths` for the current state. `lengths[..=cursor]` is
    /// the undo side; `lengths[cursor + 1..]` is the redo side.
    cursor: usize,
}

impl EditHistory {
    /// Start a history at the document's current bytes.
    pub fn new(bytes: Vec<u8>) -> EditHistory {
        let len = bytes.len();
        EditHistory {
            bytes,
            lengths: vec![len],
            cursor: 0,
        }
    }

    /// Record a new state produced by one of the edit functions above.
    /// Discards any redo states beyond the current position.
    ///
    /// See the struct docs: `bytes` must extend [`Self::current`] by pure
    /// append. That is what makes replacing `self.bytes` outright still
    /// correct for every earlier undo checkpoint — each one remains a valid
    /// prefix of the new buffer, because the new buffer's own prefix (up to
    /// wherever `push` was called from) is byte-identical to the old one.
    pub fn push(&mut self, bytes: Vec<u8>) {
        self.lengths.truncate(self.cursor + 1);
        self.lengths.push(bytes.len());
        self.bytes = bytes;
        self.cursor = self.lengths.len() - 1;
    }

    /// Step back one edit; `None` when there is nothing to undo.
    pub fn undo(&mut self) -> Option<&[u8]> {
        if self.cursor == 0 {
            return None;
        }
        self.cursor -= 1;
        Some(self.current())
    }

    /// Step forward one edit; `None` when there is nothing to redo.
    pub fn redo(&mut self) -> Option<&[u8]> {
        if self.cursor + 1 >= self.lengths.len() {
            return None;
        }
        self.cursor += 1;
        Some(self.current())
    }

    pub fn can_undo(&self) -> bool {
        self.cursor > 0
    }

    pub fn can_redo(&self) -> bool {
        self.cursor + 1 < self.lengths.len()
    }

    /// The current document bytes.
    pub fn current(&self) -> &[u8] {
        // `.min(self.bytes.len())` is pure insurance against the invariant
        // ever being violated by a future bug — it should be unreachable by
        // construction, since `new`/`push` are the only writers of
        // `lengths`/`bytes` and both keep it true.
        let len = self.lengths[self.cursor].min(self.bytes.len());
        &self.bytes[..len]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // EditHistory — no document needed, must pass standalone.
    // -----------------------------------------------------------------------

    #[test]
    fn new_history_cannot_undo_or_redo() {
        let h = EditHistory::new(vec![1, 2, 3]);
        assert!(!h.can_undo());
        assert!(!h.can_redo());
        assert_eq!(h.current(), &[1, 2, 3]);
    }

    #[test]
    fn push_extends_current_and_enables_undo() {
        let mut h = EditHistory::new(vec![1, 2, 3]);
        h.push(vec![1, 2, 3, 4, 5]);
        assert_eq!(h.current(), &[1, 2, 3, 4, 5]);
        assert!(h.can_undo());
        assert!(!h.can_redo());
    }

    #[test]
    fn undo_is_truncation_a_real_prefix_of_the_pushed_bytes() {
        let mut h = EditHistory::new(vec![1, 2, 3]);
        h.push(vec![1, 2, 3, 4, 5]);
        let before_undo = h.current().to_vec();
        let undone = h.undo().unwrap();
        assert_eq!(undone, &[1, 2, 3]);
        // The prefix relationship the whole design leans on:
        assert!(before_undo.starts_with(undone));
    }

    #[test]
    fn redo_restores_the_undone_state() {
        let mut h = EditHistory::new(vec![1, 2, 3]);
        h.push(vec![1, 2, 3, 4, 5]);
        h.undo();
        assert_eq!(h.redo(), Some(&[1u8, 2, 3, 4, 5][..]));
        assert!(!h.can_redo());
        assert!(h.can_undo());
    }

    #[test]
    fn undo_past_the_start_returns_none_and_stays_put() {
        let mut h = EditHistory::new(vec![1, 2, 3]);
        assert_eq!(h.undo(), None);
        assert_eq!(h.current(), &[1, 2, 3]);
    }

    #[test]
    fn redo_past_the_end_returns_none_and_stays_put() {
        let mut h = EditHistory::new(vec![1, 2, 3]);
        h.push(vec![1, 2, 3, 4]);
        assert_eq!(h.redo(), None);
        assert_eq!(h.current(), &[1, 2, 3, 4]);
    }

    #[test]
    fn push_after_undo_discards_the_redo_tail() {
        let mut h = EditHistory::new(vec![0]);
        h.push(vec![0, 1]); // edit A
        h.push(vec![0, 1, 2]); // edit A2
        h.undo(); // back to edit A's state (len 2)
        h.undo(); // back to the start (len 1)
        assert!(h.can_redo());
        // A different edit branches away from A/A2 entirely.
        h.push(vec![0, 9, 9, 9]); // edit B
        assert_eq!(h.current(), &[0, 9, 9, 9]);
        assert!(
            !h.can_redo(),
            "the A/A2 redo tail must be gone after edit B"
        );
        assert!(h.can_undo());
        assert_eq!(h.undo(), Some(&[0u8][..]));
        assert!(!h.can_undo());
    }

    #[test]
    fn multiple_undo_redo_round_trip() {
        let mut h = EditHistory::new(vec![0]);
        h.push(vec![0, 1]);
        h.push(vec![0, 1, 2]);
        h.push(vec![0, 1, 2, 3]);
        assert_eq!(h.undo(), Some(&[0u8, 1, 2][..]));
        assert_eq!(h.undo(), Some(&[0u8, 1][..]));
        assert_eq!(h.undo(), Some(&[0u8][..]));
        assert_eq!(h.undo(), None);
        assert_eq!(h.redo(), Some(&[0u8, 1][..]));
        assert_eq!(h.redo(), Some(&[0u8, 1, 2][..]));
        assert_eq!(h.redo(), Some(&[0u8, 1, 2, 3][..]));
        assert_eq!(h.redo(), None);
    }

    // -----------------------------------------------------------------------
    // format_pdf_date — known epoch values (cross-checked with `date -u`).
    // -----------------------------------------------------------------------

    #[test]
    fn date_epoch_zero_is_1970_01_01() {
        assert_eq!(format_pdf_date(0), "D:19700101000000");
    }

    #[test]
    fn date_y2k_epoch() {
        // `date -u -d @946684800` => Sat Jan 1 00:00:00 UTC 2000.
        assert_eq!(format_pdf_date(946_684_800), "D:20000101000000");
    }

    #[test]
    fn date_2021_new_year() {
        // `date -u -d @1609459200` => Fri Jan 1 00:00:00 UTC 2021.
        assert_eq!(format_pdf_date(1_609_459_200), "D:20210101000000");
    }

    #[test]
    fn date_with_nonzero_time_of_day() {
        // `date -u -d @1700000000` => Tue Nov 14 22:13:20 UTC 2023.
        assert_eq!(format_pdf_date(1_700_000_000), "D:20231114221320");
    }

    #[test]
    fn date_end_of_leap_year_2024() {
        // `date -u -d @1735689599` => Tue Dec 31 23:59:59 UTC 2024.
        assert_eq!(format_pdf_date(1_735_689_599), "D:20241231235959");
    }

    // -----------------------------------------------------------------------
    // Round-trip tests against the read-only fixtures.
    //
    // add_ink_annot/delete_annot both bottom out in
    // `super::write::incremental_update`, which is `todo!()` until the
    // sibling `write.rs` agent lands its implementation. These tests are
    // written now (per the task brief) and will panic-via-todo! until then;
    // see this file's report for which ones that affects at hand-off time.
    // -----------------------------------------------------------------------

    const NO_AP_FIXTURE: &[u8] = include_bytes!("../../tests/fixtures/ink-annots-no-ap.pdf");
    const MIXED_AP_FIXTURE: &[u8] = include_bytes!("../../tests/fixtures/ink-annots-mixed-ap.pdf");

    fn sample_spec(page_index: usize, color: [f32; 3]) -> InkAnnotSpec {
        InkAnnotSpec {
            page_index,
            strokes: vec![InkStroke {
                points: vec![(30.0, 30.0), (60.0, 90.0), (90.0, 30.0)],
            }],
            color,
            width: 2.0,
            opacity: 1.0,
            author: Some("kopitiam".to_string()),
        }
    }

    #[test]
    fn page_annot_refs_counts_the_no_ap_fixture() {
        let doc = PdfDocument::open(NO_AP_FIXTURE.to_vec()).unwrap();
        let refs = page_annot_refs(&doc, 0);
        assert_eq!(refs.len(), 3);
        assert!(refs.iter().all(|r| r.subtype == "Ink"));
    }

    #[test]
    fn page_annot_refs_counts_the_mixed_ap_fixture() {
        let doc = PdfDocument::open(MIXED_AP_FIXTURE.to_vec()).unwrap();
        let refs = page_annot_refs(&doc, 0);
        assert_eq!(refs.len(), 2);
    }

    #[test]
    fn page_annot_refs_out_of_range_page_is_empty() {
        let doc = PdfDocument::open(NO_AP_FIXTURE.to_vec()).unwrap();
        assert!(page_annot_refs(&doc, 99).is_empty());
    }

    #[test]
    fn add_ink_annot_round_trips_through_reopen_and_render() {
        let doc = PdfDocument::open(NO_AP_FIXTURE.to_vec()).unwrap();
        let before = page_annot_refs(&doc, 0).len();

        let spec = sample_spec(0, [1.0, 0.5, 0.0]); // orange, distinct from the fixture's colours
        let new_bytes = add_ink_annot(&doc, &spec).expect("add_ink_annot");

        let doc2 = PdfDocument::open(new_bytes).expect("new bytes must reopen");
        let after = page_annot_refs(&doc2, 0);
        assert_eq!(after.len(), before + 1);

        let pix = super::super::rasterize_page(&doc2, 0, 150.0).expect("rasterize");
        let mut orange = 0usize;
        for y in 0..pix.height() as i32 {
            for x in 0..pix.width() as i32 {
                let Some(p) = pix.pixel(x, y) else { continue };
                if p.len() < 3 {
                    continue;
                }
                let (r, g, b) = (p[0] as i32, p[1] as i32, p[2] as i32);
                // Orange: red dominant, green mid, blue low.
                if r > 150 && b < 80 && g > 40 && g < 200 && r - b > 100 {
                    orange += 1;
                }
            }
        }
        assert!(
            orange > 20,
            "new ink annotation not painted (orange px = {orange})"
        );
    }

    #[test]
    fn add_ink_annot_on_mixed_ap_fixture_round_trips() {
        let doc = PdfDocument::open(MIXED_AP_FIXTURE.to_vec()).unwrap();
        let before = page_annot_refs(&doc, 0).len();
        let spec = sample_spec(0, [0.2, 0.2, 0.9]);
        let new_bytes = add_ink_annot(&doc, &spec).expect("add_ink_annot");
        let doc2 = PdfDocument::open(new_bytes).expect("reopen");
        assert_eq!(page_annot_refs(&doc2, 0).len(), before + 1);
    }

    #[test]
    fn add_ink_annot_rejects_empty_strokes() {
        let doc = PdfDocument::open(NO_AP_FIXTURE.to_vec()).unwrap();
        let spec = InkAnnotSpec {
            page_index: 0,
            strokes: vec![],
            color: [0.0, 0.0, 0.0],
            width: 1.0,
            opacity: 1.0,
            author: None,
        };
        assert!(add_ink_annot(&doc, &spec).is_err());
    }

    #[test]
    fn add_ink_annot_out_of_range_page_errors() {
        let doc = PdfDocument::open(NO_AP_FIXTURE.to_vec()).unwrap();
        let spec = sample_spec(5, [0.0, 0.0, 0.0]);
        assert!(add_ink_annot(&doc, &spec).is_err());
    }

    #[test]
    fn delete_annot_removes_exactly_one_reference() {
        let doc = PdfDocument::open(NO_AP_FIXTURE.to_vec()).unwrap();
        let before = page_annot_refs(&doc, 0);
        let victim = before[0].num;

        let new_bytes = delete_annot(&doc, 0, victim).expect("delete_annot");
        let doc2 = PdfDocument::open(new_bytes).expect("reopen");
        let after = page_annot_refs(&doc2, 0);

        assert_eq!(after.len(), before.len() - 1);
        assert!(after.iter().all(|r| r.num != victim));
    }

    #[test]
    fn delete_annot_unknown_number_errors() {
        let doc = PdfDocument::open(NO_AP_FIXTURE.to_vec()).unwrap();
        assert!(delete_annot(&doc, 0, 999_999).is_err());
    }

    #[test]
    fn delete_annot_out_of_range_page_errors() {
        let doc = PdfDocument::open(NO_AP_FIXTURE.to_vec()).unwrap();
        assert!(delete_annot(&doc, 5, 5).is_err());
    }

    #[test]
    fn add_then_delete_round_trips_back_to_original_count() {
        let doc = PdfDocument::open(NO_AP_FIXTURE.to_vec()).unwrap();
        let original_count = page_annot_refs(&doc, 0).len();

        let spec = sample_spec(0, [0.1, 0.8, 0.1]);
        let after_add = add_ink_annot(&doc, &spec).expect("add");
        let doc2 = PdfDocument::open(after_add).expect("reopen after add");
        let refs_after_add = page_annot_refs(&doc2, 0);
        assert_eq!(refs_after_add.len(), original_count + 1);

        let new_num = refs_after_add
            .iter()
            .map(|r| r.num)
            .max()
            .expect("at least one annot");
        let after_delete = delete_annot(&doc2, 0, new_num).expect("delete");
        let doc3 = PdfDocument::open(after_delete).expect("reopen after delete");
        assert_eq!(page_annot_refs(&doc3, 0).len(), original_count);
    }

    // -----------------------------------------------------------------------
    // locate_page — internal helper, testable without going through the
    // public add/delete entry points.
    // -----------------------------------------------------------------------

    #[test]
    fn locate_page_finds_object_number_and_raw_dict() {
        let doc = PdfDocument::open(NO_AP_FIXTURE.to_vec()).unwrap();
        let (num, _gen, dict) = locate_page(&doc, 0).unwrap();
        assert_eq!(num, 3); // fixture's page dict is "3 0 obj"
        assert_eq!(dict.dict_gets("Type").unwrap().to_name(), b"Page");
    }

    #[test]
    fn locate_page_out_of_range_errors() {
        let doc = PdfDocument::open(NO_AP_FIXTURE.to_vec()).unwrap();
        assert!(locate_page(&doc, 1).is_err());
    }
}

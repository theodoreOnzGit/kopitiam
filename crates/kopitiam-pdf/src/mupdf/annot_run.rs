//! Painting a page's `/Annots` -- the half of `fz_run_page` that was never
//! ported.
//!
//! Ported from MuPDF `source/pdf/pdf-run.c` (`pdf_run_page_annots`,
//! `pdf_run_annot_with_usage`), `source/pdf/pdf-annot.c`
//! (`pdf_annot_transform`, `pdf_annot_ap`) and `source/pdf/pdf-interpret.c`
//! (`pdf_process_annot`, which is where the flag/`Popup` skip and the
//! `cm`+`Do` sequencing actually live) (commit 5fe54ce, AGPL-3.0,
//! © Artifex Software, Inc.), translated to Rust for KOPITIAM (AGPL-3.0-only).
//! See docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references").
//!
//! # The gap this closes
//!
//! A page's `/Annots` is a **sibling** of `/Contents`, not part of it.
//! [`super::page_run::run_page`] gathers only `/Contents`, so before this
//! module no annotation could ever be drawn no matter how correct the
//! rasterizer was. Upstream, `fz_run_page` is
//! `pdf_run_page_contents` **+** `pdf_run_page_annots`; only the first half
//! existed here.
//!
//! This module is the **consumer** half: it takes each annotation's `/AP`
//! `/N` form XObject -- or, when there is none, the appearance
//! [`super::annot_appearance::synthesize_ap`] builds -- and runs it through the
//! ordinary content interpreter onto the caller's device.
//!
//! ## What is deliberately not ported
//!
//! * **Annotation `/Rotate`/`NoRotate` handling.** `pdf_annot_transform`
//!   additionally pre-rotates the mapping for `NoRotate`-flagged annots so they
//!   stay upright under a rotated page. Real-world `/Rotate 0` pages (by far
//!   the common case) take the `rotmat = fz_identity` branch anyway, and the
//!   task this module was written against did not ask for the rotated case.
//!   Skipping it means a `NoRotate` annotation on a rotated page will rotate
//!   along with the page instead of staying upright -- a real but narrow gap,
//!   worth a follow-up bead if it bites.
//! * **`/AP` `/D` (down) and `/R` (rollover) states**, and the
//!   `is_hot`/`is_active` hover bookkeeping that picks between them in
//!   `pdf_annot_ap`. KOPITIAM renders static pages, not an interactive form
//!   widget under a live cursor, so `/N` (the normal appearance) is always the
//!   right choice -- upstream's own hover logic reduces to exactly this when
//!   `is_hot`/`is_active` are false, which they always are here.
//! * **`PDF_ANNOT_IS_INVISIBLE`** (bit 1). Upstream skips on
//!   `INVISIBLE | HIDDEN` together, but `/Invisible` only matters for an
//!   annotation subtype the *viewer has no handler for* -- and this module
//!   always has a handler, either a real `/AP` or a synthesised one. Only
//!   `HIDDEN` and `NO_VIEW` are checked here (see [`ANNOT_FLAG_HIDDEN`] /
//!   [`ANNOT_FLAG_NO_VIEW`]).
//! * **Multi-level resource fallback.** Upstream pushes the *page's* resources
//!   onto the processor's resource stack unconditionally, then runs the
//!   appearance form as an ordinary `Do`, which pushes the *form's own*
//!   resources on top only if it has any -- giving two-deep lookup (form, then
//!   page) for any single resource name missing from the form's dict. This
//!   port makes a single either/or choice instead (form resources if the form
//!   has any at all, else the page's, per the SEMANTICS in the task this
//!   module was written against) rather than threading a second stack level
//!   through [`super::interpret::Processor`]. The common case -- a
//!   self-contained form -- is identical either way; the difference only shows
//!   up for a form that defines *some* resource categories but leans on the
//!   page for others, which is rare in practice.
//!
//!   The page fallback applies **only to a real `/AP` stream**. A
//!   [`SynthAp`] built by [`synthesize_ap`] never references anything from
//!   the page -- its content is entirely self-authored (colour/width/opacity
//!   baked straight into the operator bytes, at most one `/ExtGState` of its
//!   own) -- so an [`Object::Null`] `resources` there means "genuinely none
//!   needed", not "look outward". Falling through to the page's resources in
//!   that case would be wrong: a same-named resource on the page (e.g.
//!   another `/GS0`) could shadow-resolve into synthesised content that never
//!   asked for it. See [`PaintSource`].
//! * **`/OC` (optional content) visibility.** `pdf_process_annot` also checks
//!   `pdf_is_ocg_hidden`; KOPITIAM has no OCG (layers) engine yet, so this is
//!   simply not modelled, same as the rest of the OCG surface across this port.

use super::annot_appearance::{SynthAp, synthesize_ap};
use super::error::Result;
use super::geometry::{Matrix, Rect};
use super::object::Object;
use super::text_device::TextDevice;
use super::xref::PdfDocument;

// include/mupdf/pdf/annot.h:82-88 -- `enum pdf_annot_flags` bit values (1-indexed
// bit position N -> `1 << (N-1)`). Only the two bits this module acts on.
/// `/F` bit 2 -- Hidden (PDF 32000-1:2008 Table 165). Checked in
/// `pdf_process_annot` (pdf-interpret.c:1886), alongside `Invisible`
/// (deliberately not ported -- see the module docs).
const ANNOT_FLAG_HIDDEN: i64 = 1 << 1; // value 2
/// `/F` bit 6 -- NoView (Table 165). Checked in `pdf_process_annot`
/// (pdf-interpret.c:1902) for the "View" usage, which is the only usage
/// KOPITIAM ever renders for.
const ANNOT_FLAG_NO_VIEW: i64 = 1 << 5; // value 32

/// Cap on `/Parent` chain walks ([`dict_get_inheritable`]), so a cyclic (and
/// therefore malformed) form-field hierarchy cannot loop forever.
const MAX_PARENT_DEPTH: usize = 64;

/// Paint every visible annotation on `page` onto `dev`.
///
/// `base_ctm` is the same MediaBox-derived page transform
/// [`super::page_run::run_page`] uses, so annotations land in the same device
/// space as the page content. Call this **after** the content stream: PDF
/// draws annotations on top.
///
/// A malformed or unpaintable individual annotation is skipped, never a
/// reason to fail the whole page -- annotations are decoration. This function
/// itself therefore always returns `Ok`; the `Result` is kept only because the
/// signature is shared with the rest of this port's `run_*` family.
// MuPDF: pdf_run_page_annots_with_usage_imp (pdf-run.c:301) -- the annots-array
// walk; pdf_run_annot_with_usage (pdf-run.c:27) -- one annotation.
pub fn run_page_annots<D: TextDevice + ?Sized>(
    doc: &PdfDocument,
    page: &Object,
    base_ctm: Matrix,
    dev: &mut D,
) -> Result<()> {
    run_page_annots_with(doc, page, base_ctm, dev, AnnotPass::All)
}

/// Which annotations a pass should paint.
///
/// This exists because `kopitiam-pdf` can render a page with **two different
/// engines**. The native engine ([`super::draw_device::rasterize_page_ex`])
/// draws everything itself, so it wants [`AnnotPass::All`]. But when a page
/// hits the glyph fallback, the crate re-renders it with `hayro`
/// ([`super::hayro_fallback`]) -- and hayro draws annotations too, except that
/// it gates every one of them behind `/AP` -> `/N`
/// (`hayro-interpret/src/interpret/mod.rs:157`). So on that path hayro has
/// already painted the real-`/AP` annots, and painting them a second time
/// would double-draw: harmless for an opaque stroke, but visibly wrong for one
/// with `/CA` < 1, where two translucent passes composite darker than one.
///
/// [`AnnotPass::HayroSkipped`] paints exactly the complement -- the annots
/// hayro skipped because they have no `/AP` at all -- so the two engines add up
/// to one complete page with nothing drawn twice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnnotPass {
    /// Every visible annotation: real `/AP` streams and synthesised
    /// appearances alike. What a from-scratch render wants.
    All,
    /// Only the annotations `hayro` does **not** draw, so the two engines add
    /// up to one complete page with nothing drawn twice. Two kinds qualify:
    ///
    /// 1. **No `/AP` at all** -- the appearance had to be synthesised (the
    ///    AP-less ink annotations Okular writes).
    /// 2. **`/AP` `/N` is a dictionary of appearance states**, not a stream.
    ///    hayro asks for `ap.get::<Stream>(N)`
    ///    (`hayro-interpret/src/interpret/mod.rs:157`) and gets nothing, so
    ///    **every checkbox and radio button is invisible to it** -- their `/N`
    ///    is always a state dict keyed by `/AS`. Found by dogfooding: radios
    ///    rendered fine normally and vanished the moment the fallback engaged.
    ///
    /// Named for the question it answers -- "what did hayro miss?" -- rather
    /// than for one of the two answers.
    HayroSkipped,
}

/// [`run_page_annots`], restricted to a subset of the page's annotations.
/// See [`AnnotPass`] for why the subset matters.
pub fn run_page_annots_with<D: TextDevice + ?Sized>(
    doc: &PdfDocument,
    page: &Object,
    base_ctm: Matrix,
    dev: &mut D,
    pass: AnnotPass,
) -> Result<()> {
    let annots = doc.resolve_get(page, "Annots").unwrap_or(Object::Null);
    if !matches!(annots, Object::Array(_)) {
        return Ok(()); // no /Annots, or a malformed non-array value: nothing to draw.
    }

    let page_resources = doc.resolve_get(page, "Resources").unwrap_or(Object::Null);

    for i in 0..annots.array_len() {
        let Some(entry) = annots.array_get(i) else {
            continue;
        };
        let Ok(annot) = doc.resolve(entry) else {
            continue; // unresolvable indirect reference: skip, don't fail the page.
        };
        if !annot.is_dict() {
            continue;
        }

        // Rule: one bad annotation must never abort the page. Every failure
        // mode below already resolves to "skip this annot" internally: the
        // `Result` only exists because `Processor::run_stream` returns one,
        // and even that failure is swallowed here rather than propagated.
        let _ = paint_one_annot(doc, &annot, base_ctm, &page_resources, dev, pass);
    }

    Ok(())
}

/// Paint a single already-resolved annotation dict, or decide (silently) not
/// to. Every skip decision funnels through here; the only thing that can make
/// it back out as `Err` is a failure inside [`super::interpret::Processor`]
/// while running the appearance's content stream, and the caller swallows
/// that too.
fn paint_one_annot<D: TextDevice + ?Sized>(
    doc: &PdfDocument,
    annot: &Object,
    base_ctm: Matrix,
    page_resources: &Object,
    dev: &mut D,
    pass: AnnotPass,
) -> Result<()> {
    let subtype = doc.resolve_get(annot, "Subtype").unwrap_or(Object::Null);
    let flags = doc.resolve_get(annot, "F").unwrap_or(Object::Null).to_int();

    // MuPDF: pdf-interpret.c:1886 -- Hidden skips unconditionally (paired
    // there with Invisible, not ported here; see module docs).
    if flags & ANNOT_FLAG_HIDDEN != 0 {
        return Ok(());
    }
    // MuPDF: pdf-interpret.c:1889-1891 -- "popup annotations should never be
    // drawn", checked ahead of the usage-flag tests.
    if subtype.to_name() == b"Popup" {
        return Ok(());
    }
    // MuPDF: pdf-interpret.c:1902 -- NoView skips for the "View" usage.
    // KOPITIAM never renders for "Print", so this check always applies.
    if flags & ANNOT_FLAG_NO_VIEW != 0 {
        return Ok(());
    }

    // MuPDF: pdf-run.c:48-58 -- "Widgets only get displayed if they have both
    // a T and a FT flag, apparently" (their comment, not ours). Both are
    // looked up through the inheritable `/Parent` chain: form fields nest,
    // and `/FT`/`/T` are commonly declared once on an ancestor.
    if subtype.to_name() == b"Widget" {
        let ft = dict_get_inheritable(doc, annot, "FT");
        let t = dict_get_inheritable(doc, annot, "T");
        if ft.is_none() || t.is_none() {
            return Ok(());
        }
    }

    let source = match resolve_appearance(doc, annot) {
        Some(s) => s,
        None => return Ok(()),
    };

    // MuPDF: pdf-run.c:74 -- for a *real* `/AP` stream, the annotation
    // processor is seeded with the page's resources, and the appearance form
    // then behaves exactly like a `Do`-invoked Form XObject
    // (pdf-op-run.c's pdf_run_xobject), pushing its own `/Resources` on top
    // only if it has any. This port collapses that into a single either/or
    // choice -- see "Multi-level resource fallback" in the module docs.
    //
    // A *synthesised* appearance never falls back to the page: its
    // `Object::Null` means "this content needs nothing", not "look outward"
    // (see [`PaintSource`]'s docs).
    let (appearance, resources) = match source {
        // Skip a real `/AP` when the caller only wants synthesised
        // appearances -- an `/AP`-only engine (hayro) has already drawn this
        // one, and drawing it again would composite a `/CA` < 1 annot twice.
        // See [`AnnotPass`].
        // Skip only what the /AP-only engine has ALREADY drawn: an /N that was
        // a real stream. An /N state dictionary (every checkbox and radio) is
        // invisible to hayro, so it still needs painting here.
        PaintSource::RealAp(_, true) if pass == AnnotPass::HayroSkipped => return Ok(()),
        PaintSource::RealAp(ap, _) => {
            let resources = if ap.resources.is_dict() {
                ap.resources.clone()
            } else {
                page_resources.clone()
            };
            (ap, resources)
        }
        PaintSource::Synthesized(ap) => {
            let resources = ap.resources.clone();
            (ap, resources)
        }
    };

    let Some(final_ctm) = compose_final_ctm(&appearance, base_ctm) else {
        return Ok(()); // malformed BBox/Rect/Matrix: non-finite result, skip.
    };

    let mut proc = super::interpret::Processor::new(doc, dev, final_ctm, resources);
    proc.run_stream(&appearance.content)
}

// ---------------------------------------------------------------------------
// Appearance resolution: /AP -> a stream, a dict-of-states, or nothing.
// ---------------------------------------------------------------------------

/// Get the appearance to paint for `annot`: a real `/AP` stream (selected
/// directly, or out of a `/N` dictionary-of-states via `/AS`), or -- only when
/// there is genuinely no usable `/AP` at all -- whatever
/// [`synthesize_ap`] builds. `None` means "paint nothing for this annot".
///
/// A `/AP` that *is* present but ambiguous (a dictionary of states with no
/// `/AS` match and more than one candidate) or broken (a stream missing its
/// required `/BBox`) is **not** a case for the synthesiser: this annotation
/// carries real appearance data the producer half has no business overriding,
/// so it is skipped outright.
///
/// The result is tagged with [`PaintSource`] (rather than a bare [`SynthAp`])
/// so the caller can tell a real stream from a synthesised one -- that
/// distinction is exactly what decides whether an absent `/Resources` may
/// fall back to the page's.
fn resolve_appearance(doc: &PdfDocument, annot: &Object) -> Option<PaintSource> {
    match lookup_ap(doc, annot) {
        ApLookup::Found(ap, from_stream) => Some(PaintSource::RealAp(ap, from_stream)),
        ApLookup::AmbiguousOrBroken => None,
        ApLookup::Absent => synthesize_ap(doc, annot).map(PaintSource::Synthesized),
    }
}

/// Where a [`SynthAp`] came from -- a real `/AP` stream, or the producer half
/// filling in for one that never existed. The two must not be treated
/// identically once a resources fallback is on the table: see the "Multi-level
/// resource fallback" note in the module docs.
enum PaintSource {
    /// Read directly from the annotation's own `/AP` (or its `/N` dictionary
    /// of states). May legitimately omit `/Resources` and rely on the page's.
    ///
    /// The `bool` is `true` when `/N` was itself a stream, `false` when it was
    /// a dictionary of appearance states -- see [`ApLookup::Found`] for why
    /// that matters to the `hayro` overlay.
    RealAp(SynthAp, bool),
    /// Built in memory by [`synthesize_ap`] because the annotation had no
    /// `/AP` at all. Self-contained by construction: an absent `/Resources`
    /// here means none are needed, never "check the page".
    Synthesized(SynthAp),
}

/// The three outcomes of looking up `/AP`/`/N` on an annotation.
enum ApLookup {
    /// A usable appearance was found. The `bool` records whether `/N` was
    /// **itself a stream** (`true`) or a **dictionary of appearance states**
    /// from which one was selected (`false`).
    ///
    /// That distinction is not academic: `hayro` only accepts `/N` as a
    /// stream (`ap.get::<Stream>(N)`,
    /// `hayro-interpret/src/interpret/mod.rs:157`), so every checkbox and
    /// radio -- whose `/N` is always a state dictionary -- is invisible to it.
    /// [`AnnotPass::HayroSkipped`] uses this to paint them back.
    Found(SynthAp, bool),
    /// `/AP` is present but no usable stream could be selected from it.
    AmbiguousOrBroken,
    /// No `/AP` (or no `/N` under it) at all: the caller should try the
    /// producer.
    Absent,
}

// MuPDF: pdf_annot_ap (pdf-annot.c:57), reduced to the `/N`-only case (see
// "not ported" in the module docs for why `/D`/`/R` are skipped).
fn lookup_ap(doc: &PdfDocument, annot: &Object) -> ApLookup {
    let ap = doc.resolve_get(annot, "AP").unwrap_or(Object::Null);
    if !ap.is_dict() {
        return ApLookup::Absent;
    }
    let Some(n_raw) = ap.dict_gets("N") else {
        return ApLookup::Absent;
    };

    // PDF streams are always indirect objects -- there is no such thing as an
    // inline stream in PDF syntax -- so "does /N resolve to a stream" reduces
    // to "is /N an indirect reference, and does opening it as a stream
    // succeed". This crate represents both a stream dict and a plain dict as
    // `Object::Dict` (see mod.rs's module docs), so there is no `is_stream()`
    // predicate to ask directly; `open_stream` succeeding is the closest
    // available proxy for MuPDF's `pdf_obj_num_is_stream`.
    if matches!(n_raw, Object::Ref { .. })
        && let Ok(bytes) = doc.open_stream(n_raw)
    {
        return match form_ap(doc, n_raw, bytes, annot) {
            // /N was itself a stream -- hayro draws this one.
            Some(found) => ApLookup::Found(found, true),
            None => ApLookup::AmbiguousOrBroken,
        };
    }

    // Not a stream: treat /N as a dictionary of appearance states, keyed by
    // /AS (pdf-annot.c:76-79).
    let Ok(n_dict) = doc.resolve(n_raw) else {
        return ApLookup::AmbiguousOrBroken;
    };
    if !n_dict.is_dict() {
        return ApLookup::AmbiguousOrBroken;
    }

    let as_obj = doc.resolve_get(annot, "AS").ok();
    let as_name = as_obj
        .as_ref()
        .filter(|o| !o.is_null())
        .map(|o| o.to_name());

    // A button field's /V is a state name and is *inheritable*: a radio kid
    // normally carries no /V of its own, the group parent holds it. Without
    // walking up, a widget missing /AS has nothing to fall back on.
    let v_obj = dict_get_inheritable(doc, annot, "V");
    let v_name = v_obj.as_ref().filter(|o| o.is_name()).map(|o| o.to_name());

    let Some(selected) = select_ap_state(&n_dict, as_name, v_name) else {
        return ApLookup::AmbiguousOrBroken;
    };

    if !matches!(selected, Object::Ref { .. }) {
        return ApLookup::AmbiguousOrBroken;
    }
    match doc.open_stream(selected) {
        Ok(bytes) => match form_ap(doc, selected, bytes, annot) {
            // Selected out of a /N state dictionary -- hayro skips these.
            Some(found) => ApLookup::Found(found, false),
            None => ApLookup::AmbiguousOrBroken,
        },
        Err(_) => ApLookup::AmbiguousOrBroken,
    }
}

/// Pick the appearance-state entry real viewers use when `/AP`'s `/N` is a
/// dictionary of named states rather than a single stream: the entry named by
/// `/AS`, or -- when `/AS` is absent and there is exactly one candidate --
/// that sole entry (not in the spec; it's what real viewers do for the common
/// single-state case, e.g. a checkbox authored with only an `/Off` appearance
/// before it is ever checked). `None` means "cannot tell which one": the
/// caller skips the annotation rather than guessing.
///
/// Pure and document-free by design, so it is unit-testable on hand-built
/// dicts without a [`PdfDocument`].
fn select_ap_state<'a>(
    n_dict: &'a Object,
    as_name: Option<&[u8]>,
    value_name: Option<&[u8]>,
) -> Option<&'a Object> {
    // 1. /AS, when it names a state that actually exists.
    if let Some(found) = as_name.and_then(|name| n_dict.dict_get(name)) {
        return found.into();
    }
    // 2. The field's own value. For a button field /V is a state name, and a
    //    widget with no /AS is otherwise indistinguishable from one that should
    //    not be drawn -- but its value still says which state it is in.
    if let Some(found) = value_name.and_then(|name| n_dict.dict_get(name)) {
        return found.into();
    }
    // 3. /Off, the near-universal name for the unselected state. An unselected
    //    radio must still draw its empty ring; drawing nothing is what made
    //    them invisible.
    if let Some(found) = n_dict.dict_get(b"Off") {
        return found.into();
    }
    // 4. A sole entry is unambiguous whatever it is called.
    if n_dict.dict_len() == 1 {
        return n_dict.dict_get_val(0);
    }
    // 5. Last resort: the first state. A malformed widget drawn in the wrong
    //    state beats a widget the user cannot see at all -- which is the whole
    //    lesson of this function's history.
    n_dict.dict_get_val(0)
}

/// Build a [`SynthAp`] from a real `/AP` stream: `stream_ref` is the indirect
/// reference that was just opened as `bytes`; `annot` supplies `/Rect`.
/// `None` means the stream is missing required data (`/BBox`, or the
/// annotation's own `/Rect`) -- broken, not absent, so the caller does not
/// fall through to the synthesiser.
fn form_ap(
    doc: &PdfDocument,
    stream_ref: &Object,
    bytes: Vec<u8>,
    annot: &Object,
) -> Option<SynthAp> {
    let dict = doc.resolve(stream_ref).ok()?;
    let bbox = rect_from(doc, &dict, "BBox")?;
    let matrix = matrix_from(&dict).unwrap_or(Matrix::IDENTITY);
    let rect = rect_from(doc, annot, "Rect")?;
    if !rect_is_finite(bbox) || !rect_is_finite(rect) {
        return None;
    }
    let resources = doc.resolve_get(&dict, "Resources").unwrap_or(Object::Null);
    Some(SynthAp {
        bbox,
        matrix,
        rect,
        content: bytes,
        resources,
    })
}

// ---------------------------------------------------------------------------
// Algorithm 8.1: BBox -> Rect.
// ---------------------------------------------------------------------------

/// PDF 32000-1:2008 §12.5.5 Algorithm 8.1, minus the annotation-rotation step
/// (see "not ported" in the module docs) -- MuPDF `pdf_annot_transform`
/// (pdf-annot.c:194-224), with `rotmat` fixed at identity.
///
/// Returns the transform `A` that carries a point in the appearance form's own
/// (BBox) space onto the annotation's `/Rect` in page space. The caller still
/// has to fold in the form's own `/Matrix` and the page's base CTM (done in
/// [`compose_final_ctm`]) -- kept separate here purely so this half, the part
/// with the scale/zero-dimension edge case, is unit-testable on its own
/// without a whole document.
///
/// Steps, matching upstream line-for-line:
/// 1. `bbox_t = bbox.transform(matrix)` -- the *upright* bounding box of the
///    (possibly sheared/rotated) transformed BBox (pdf-annot.c:203-204).
/// 2. `sx = rect.width / bbox_t.width`, `sy` likewise for height
///    (pdf-annot.c:216-219) -- **except** when a `bbox_t` dimension is zero,
///    in which case upstream hard-codes that scale to `0` rather than
///    dividing. A degenerate (zero-width or zero-height) appearance BBox is
///    not hypothetical: `annot_appearance::synthesize_ap`'s Ink-annotation
///    fallback fires it for a perfectly-vertical or perfectly-horizontal
///    stroke, and dividing by zero there would poison every downstream matrix
///    with `inf`/`NaN`.
/// 3. `x = rect.x0 - bbox_t.x0*sx`, `y` likewise (pdf-annot.c:220-221) -- the
///    translation that lands `bbox_t`'s origin on `rect`'s origin once scaled.
fn annot_bbox_to_rect(bbox: Rect, matrix: Matrix, rect: Rect) -> Matrix {
    let bbox_t = bbox.transform(matrix);

    let w = bbox_t.x1 - bbox_t.x0;
    let h = bbox_t.y1 - bbox_t.y0;
    // MuPDF: pdf-annot.c:216-219 -- `if (bbox.x1 == bbox.x0) sx = 0; else ...`.
    // `w == 0.0` is the same test: a floating-point subtraction of two equal
    // values is exact zero, so this doesn't need its own epsilon.
    let sx = if w == 0.0 {
        0.0
    } else {
        (rect.x1 - rect.x0) / w
    };
    let sy = if h == 0.0 {
        0.0
    } else {
        (rect.y1 - rect.y0) / h
    };

    let x = rect.x0 - bbox_t.x0 * sx;
    let y = rect.y0 - bbox_t.y0 * sy;

    // MuPDF: pdf-annot.c:224 -- `fz_pre_scale(fz_translate(x, y), sx, sy)`,
    // i.e. `scale(sx, sy) * translate(x, y)` in fz_concat's "apply left
    // operand first" convention (`Matrix::pre_scale`'s own doc comment says
    // as much, and it is verified against `Matrix::concat`'s formula in this
    // module's tests, not just trusted from the comment).
    Matrix::scale(sx, sy).concat(Matrix::translate(x, y))
}

/// Fold [`annot_bbox_to_rect`]'s `A` together with the appearance form's own
/// `/Matrix` and the page's base CTM into the one CTM the content stream
/// actually runs under, matching upstream's composition order:
///
/// `pdf_process_annot` (pdf-interpret.c:1913-1927) sets the CTM to
/// `pdf_annot_transform`'s result (here, `A`, since `rotmat` is identity)
/// composed onto the incoming CTM via the `cm` operator, then runs the
/// appearance as an ordinary `Do`
/// ([`super::page_run`]'s `op_do` does the same thing for any Form XObject:
/// `g.ctm = matrix.concat(g.ctm)`) which folds in the form's own `/Matrix` the
/// same way. Composing left-to-right with [`Matrix::concat`] (which applies
/// its `self` operand first): `final = matrix.concat(A).concat(base_ctm)`.
///
/// Returns `None` if the composed matrix has a non-finite component --
/// malformed `/BBox`/`/Rect`/`/Matrix` input must never hand the interpreter
/// (and, downstream, the rasterizer) a NaN/`inf` transform.
fn compose_final_ctm(ap: &SynthAp, base_ctm: Matrix) -> Option<Matrix> {
    let a = annot_bbox_to_rect(ap.bbox, ap.matrix, ap.rect);
    let final_ctm = ap.matrix.concat(a).concat(base_ctm);
    if matrix_is_finite(final_ctm) {
        Some(final_ctm)
    } else {
        None
    }
}

fn matrix_is_finite(m: Matrix) -> bool {
    m.a.is_finite()
        && m.b.is_finite()
        && m.c.is_finite()
        && m.d.is_finite()
        && m.e.is_finite()
        && m.f.is_finite()
}

fn rect_is_finite(r: Rect) -> bool {
    r.x0.is_finite() && r.y0.is_finite() && r.x1.is_finite() && r.y1.is_finite()
}

// ---------------------------------------------------------------------------
// Small local helpers (duplicated rather than imported: page_run.rs's
// equivalents are private to that module, and it is not this file to edit).
// ---------------------------------------------------------------------------

// MuPDF: pdf_to_rect (pdf-parse.c:33-50) -- reads 4 numbers and normalises
// with min/max so an out-of-order /Rect or /BBox (the spec allows either
// corner pair first) doesn't yield an inverted, "empty" rect downstream.
/// Read a 4-element numeric array (`/BBox`, `/Rect`, ...) into a normalised
/// [`Rect`], resolving each element in case of an indirect number.
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

// MuPDF: pdf_xobject_matrix (pdf-xobject.c) -- the /Matrix entry, or identity.
/// Read an XObject's `/Matrix` (6 numbers) into a [`Matrix`], or `None` if
/// absent or malformed.
fn matrix_from(xobj: &Object) -> Option<Matrix> {
    let arr = xobj.dict_gets("Matrix")?;
    if arr.array_len() < 6 {
        return None;
    }
    let v = |i: usize| -> f32 { arr.array_get(i).map(|o| o.to_real() as f32).unwrap_or(0.0) };
    Some(Matrix::new(v(0), v(1), v(2), v(3), v(4), v(5)))
}

// MuPDF: pdf_dict_get_inheritable (pdf-object.c) -- look up `key` on `obj`,
/// Look up `key` on `obj`, walking the `/Parent` chain when `obj` itself does
/// not define it (form-field inheritance: `/FT`, `/T`, `/DA`, ... are commonly
/// declared once on an ancestor field). Capped at [`MAX_PARENT_DEPTH`] so a
/// cyclic (malformed) chain cannot loop forever. A present-but-`null` value is
/// treated the same as absent, matching the PDF convention that a null value
/// is indistinguishable from a missing key.
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mupdf::draw_edge::FillRule;
    use crate::mupdf::draw_path::Path;
    use crate::mupdf::font::Font;
    use crate::mupdf::geometry::Point;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    // -----------------------------------------------------------------------
    // annot_bbox_to_rect: Algorithm 8.1's scale/translate maths.
    // -----------------------------------------------------------------------

    #[test]
    fn transform_identity_bbox_onto_identical_rect() {
        let bbox = Rect::new(0.0, 0.0, 100.0, 100.0);
        let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
        let a = annot_bbox_to_rect(bbox, Matrix::IDENTITY, rect);
        assert!(a.a.is_finite() && a.d.is_finite());
        assert!(approx(a.a, 1.0));
        assert!(approx(a.d, 1.0));
        assert!(approx(a.e, 0.0));
        assert!(approx(a.f, 0.0));

        // The whole point of Algorithm 8.1: BBox corners must land exactly on
        // Rect's corners once `A` is applied.
        let p0 = Point::new(bbox.x0, bbox.y0).transform(a);
        let p1 = Point::new(bbox.x1, bbox.y1).transform(a);
        assert!(approx(p0.x, rect.x0) && approx(p0.y, rect.y0));
        assert!(approx(p1.x, rect.x1) && approx(p1.y, rect.y1));
    }

    #[test]
    fn transform_scales_and_offsets_a_smaller_bbox_onto_a_larger_rect() {
        // A 100x50 form BBox, offset from the origin, mapped onto a 50x100
        // Rect elsewhere on the page: independent x/y scale factors, plus a
        // translation that is *not* simply rect.x0/y0 because bbox doesn't
        // start at (0, 0).
        let bbox = Rect::new(10.0, 10.0, 110.0, 60.0); // 100 wide, 50 tall
        let rect = Rect::new(0.0, 0.0, 50.0, 100.0); // 50 wide, 100 tall
        let a = annot_bbox_to_rect(bbox, Matrix::IDENTITY, rect);

        assert!(approx(a.a, 0.5)); // sx = 50/100
        assert!(approx(a.d, 2.0)); // sy = 100/50

        let p0 = Point::new(bbox.x0, bbox.y0).transform(a);
        let p1 = Point::new(bbox.x1, bbox.y1).transform(a);
        assert!(approx(p0.x, rect.x0) && approx(p0.y, rect.y0));
        assert!(approx(p1.x, rect.x1) && approx(p1.y, rect.y1));
    }

    #[test]
    fn transform_zero_width_bbox_yields_zero_scale_not_nan() {
        // A degenerate (zero-width) BBox -- exactly what a perfectly-vertical
        // synthesised Ink stroke produces. MuPDF hard-codes the affected
        // scale to 0 rather than dividing; this must never poison the matrix
        // with inf/NaN.
        let bbox = Rect::new(5.0, 5.0, 5.0, 20.0); // x0 == x1
        let rect = Rect::new(0.0, 0.0, 10.0, 10.0);
        let a = annot_bbox_to_rect(bbox, Matrix::IDENTITY, rect);

        assert_eq!(a.a, 0.0); // sx forced to 0, not inf/NaN
        assert!(approx(a.d, 10.0 / 15.0)); // sy is the ordinary quotient
        assert!(a.a.is_finite() && a.b.is_finite() && a.c.is_finite());
        assert!(a.d.is_finite() && a.e.is_finite() && a.f.is_finite());
    }

    #[test]
    fn transform_zero_height_bbox_yields_zero_scale_not_nan() {
        let bbox = Rect::new(150.0, 40.0, 151.0, 40.0); // y0 == y1
        let rect = Rect::new(150.0, 40.0, 151.0, 40.0);
        let a = annot_bbox_to_rect(bbox, Matrix::IDENTITY, rect);
        assert_eq!(a.d, 0.0);
        assert!(a.a.is_finite() && a.d.is_finite() && a.e.is_finite() && a.f.is_finite());
    }

    // -----------------------------------------------------------------------
    // Matrix::concat order, cross-checked against Matrix::pre_scale's own
    // formula (not merely trusted from its doc comment) -- this is the
    // highest-risk part of the port, so it gets its own direct test.
    // -----------------------------------------------------------------------

    #[test]
    fn pre_scale_matches_scale_concat_self() {
        let m = Matrix::new(2.0, 0.5, -0.5, 3.0, 7.0, -4.0);
        let (sx, sy) = (1.5, 0.25);
        let pre_scaled = m.pre_scale(sx, sy);
        let via_concat = Matrix::scale(sx, sy).concat(m);
        assert_eq!(pre_scaled, via_concat);
    }

    // -----------------------------------------------------------------------
    // select_ap_state: pure, document-free.
    // -----------------------------------------------------------------------

    fn states_dict() -> Object {
        let mut d = Object::new_dict();
        d.dict_put("Off", Object::new_indirect(11, 0));
        d.dict_put("On", Object::new_indirect(12, 0));
        d
    }

    #[test]
    fn select_ap_state_by_as_name() {
        let d = states_dict();
        let picked = select_ap_state(&d, Some(b"On"), None);
        assert_eq!(picked, Some(&Object::new_indirect(12, 0)));
    }

    /// A widget with **no `/AS`** must still be drawn.
    ///
    /// This previously returned `None`, and that was a real, user-visible bug:
    /// radio buttons were **completely invisible** in the viewer while still
    /// toggling correctly (the maintainer could click one in kpdf and see the
    /// change in Okular). A radio's `/N` always has at least two entries
    /// (`/Off` plus an on-state), so the old "only pick when there is exactly
    /// one" rule skipped every such widget.
    ///
    /// Poppler renders these — it logs *"Invalid or missing AS value in
    /// annotation containing one or more appearance subdictionaries"* and
    /// carries on. A malformed widget is a file to recover from, not a reason
    /// to draw nothing.
    #[test]
    fn select_ap_state_falls_back_to_off_when_as_missing() {
        let d = states_dict(); // /Off + /On, no /AS
        assert_eq!(
            select_ap_state(&d, None, None),
            Some(&Object::new_indirect(11, 0)),
            "a widget with no /AS must fall back to /Off, not vanish"
        );
    }

    /// With no `/AS`, the field's own value says which state it is in. `/V` is
    /// inheritable, so a radio kid gets it from the group parent.
    #[test]
    fn select_ap_state_prefers_field_value_over_off() {
        let d = states_dict();
        assert_eq!(
            select_ap_state(&d, None, Some(b"On")),
            Some(&Object::new_indirect(12, 0)),
            "/V should win over the /Off fallback"
        );
    }

    #[test]
    fn select_ap_state_as_missing_but_sole_entry_is_used() {
        let mut d = Object::new_dict();
        d.dict_put("Only", Object::new_indirect(14, 0));
        assert_eq!(
            select_ap_state(&d, None, None),
            Some(&Object::new_indirect(14, 0))
        );
    }

    /// An `/AS` naming a state that is not there is a malformed file. Recover
    /// rather than drawing nothing, same reasoning as the missing-`/AS` case.
    #[test]
    fn select_ap_state_recovers_when_as_names_a_missing_state() {
        let d = states_dict();
        assert_eq!(
            select_ap_state(&d, Some(b"Down"), None),
            Some(&Object::new_indirect(11, 0)),
            "a dangling /AS should fall back, not blank the widget"
        );
    }

    /// Nothing sensible to pick, but still never a silent blank: the first
    /// state beats an invisible widget.
    #[test]
    fn select_ap_state_last_resort_takes_the_first_state() {
        let mut d = Object::new_dict();
        d.dict_put("Alpha", Object::new_indirect(21, 0));
        d.dict_put("Beta", Object::new_indirect(22, 0));
        assert_eq!(
            select_ap_state(&d, None, None),
            Some(&Object::new_indirect(21, 0))
        );
    }

    // -----------------------------------------------------------------------
    // dict_get_inheritable: needs *a* PdfDocument to call resolve() on, but
    // every object involved below is direct (no indirect refs), so a minimal
    // one-page document is enough -- no real xref lookups happen.
    // -----------------------------------------------------------------------

    fn empty_doc() -> PdfDocument {
        build_pdf(&[
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>".to_vec(),
        ])
    }

    #[test]
    fn inheritable_lookup_finds_key_directly() {
        let doc = empty_doc();
        let mut widget = Object::new_dict();
        widget.dict_put("FT", Object::new_name("Tx"));
        assert!(dict_get_inheritable(&doc, &widget, "FT").is_some());
    }

    #[test]
    fn inheritable_lookup_walks_direct_parent() {
        let doc = empty_doc();
        let mut parent = Object::new_dict();
        parent.dict_put("FT", Object::new_name("Tx"));
        let mut kid = Object::new_dict();
        kid.dict_put("Parent", parent); // direct (non-indirect) parent
        assert!(dict_get_inheritable(&doc, &kid, "FT").is_some());
    }

    #[test]
    fn inheritable_lookup_missing_everywhere_is_none() {
        let doc = empty_doc();
        let leaf = Object::new_dict();
        assert!(dict_get_inheritable(&doc, &leaf, "FT").is_none());
    }

    // -----------------------------------------------------------------------
    // End-to-end: a hand-built one-page document exercising Popup skip, the
    // Hidden flag, direct-stream /AP, /AS-selected dict-of-states /AP, and
    // the AS-absent/sole-entry fallback -- all through the real
    // `run_page_annots` entry point, with no dependency on
    // `annot_appearance::synthesize_ap` (every annotation here has a real
    // `/AP`, so the producer half is never called).
    // -----------------------------------------------------------------------

    /// A device that records every `fill_path`/`stroke_path` call's colour
    /// (and, for strokes, the CTM in effect), so tests can tell which
    /// annotations painted -- and under what transform -- without needing any
    /// font/text machinery.
    #[derive(Default)]
    struct Recorder {
        fills: Vec<[f32; 3]>,
        strokes: Vec<(Matrix, [f32; 3])>,
    }

    impl TextDevice for Recorder {
        fn show_glyph(
            &mut self,
            _font: &Font,
            _trm: Matrix,
            _adv: f32,
            _unicode: char,
            _cid: u32,
            _wmode: u8,
        ) {
        }

        fn fill_path(
            &mut self,
            _path: &Path,
            _rule: FillRule,
            _ctm: Matrix,
            color: [f32; 3],
            _alpha: f32,
            _clip: Option<Rect>,
        ) {
            self.fills.push(color);
        }

        fn stroke_path(
            &mut self,
            _path: &Path,
            ctm: Matrix,
            _line_width: f32,
            color: [f32; 3],
            _alpha: f32,
            _clip: Option<Rect>,
        ) {
            self.strokes.push((ctm, color));
        }
    }

    /// Build `<< dict_fields /Length N >>\nstream\n<content>\nendstream`, with
    /// `/Length` computed from `content`'s real byte length so the two can
    /// never drift out of sync.
    fn stream_obj(dict_fields: &str, content: &[u8]) -> Vec<u8> {
        let mut body =
            format!("<< {dict_fields} /Length {} >>\nstream\n", content.len()).into_bytes();
        body.extend_from_slice(content);
        body.extend_from_slice(b"\nendstream");
        body
    }

    /// Build a PDF with objects 1.. from `bodies` (each wrapped `N 0 obj … endobj`)
    /// and a classic xref table. (Duplicated from page_run.rs's test helper of
    /// the same shape -- that one is private to its own module.)
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

    #[test]
    fn run_page_annots_skips_hidden_and_popup_selects_as_and_falls_back_to_sole_entry() {
        let bodies = vec![
            // 1: Catalog
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            // 2: Pages
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            // 3: Page -- Annots in deliberately mixed order: A, Hidden(B), Popup(C), AS-dict(D), sole-entry(E)
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Annots [4 0 R 6 0 R 8 0 R 10 0 R 13 0 R] >>"
                .to_vec(),
            // 4/5: AnnotA (plain, direct-stream /AP) -- red
            b"<< /Type /Annot /Subtype /Square /Rect [10 10 60 60] /AP << /N 5 0 R >> >>".to_vec(),
            stream_obj(
                "/Type /XObject /Subtype /Form /BBox [0 0 50 50]",
                b"1 0 0 rg 0 0 50 50 re f",
            ),
            // 6/7: AnnotB -- Hidden (/F 2), has a perfectly valid /AP that must NOT run
            b"<< /Type /Annot /Subtype /Square /Rect [70 10 120 60] /F 2 /AP << /N 7 0 R >> >>".to_vec(),
            stream_obj(
                "/Type /XObject /Subtype /Form /BBox [0 0 50 50]",
                b"0 0 0 rg 0 0 50 50 re f",
            ),
            // 8/9: AnnotC -- Popup, has a valid /AP that must NOT run either
            b"<< /Type /Annot /Subtype /Popup /Rect [130 10 180 60] /AP << /N 9 0 R >> >>".to_vec(),
            stream_obj(
                "/Type /XObject /Subtype /Form /BBox [0 0 50 50]",
                b"0 0 0 rg 0 0 50 50 re f",
            ),
            // 10/11/12: AnnotD -- /N is a dict of states, /AS picks "On" (blue), not "Off" (green)
            b"<< /Type /Annot /Subtype /Square /Rect [10 70 60 120] /AS /On /AP << /N << /Off 11 0 R /On 12 0 R >> >> >>"
                .to_vec(),
            stream_obj(
                "/Type /XObject /Subtype /Form /BBox [0 0 50 50]",
                b"0 1 0 rg 0 0 50 50 re f",
            ),
            stream_obj(
                "/Type /XObject /Subtype /Form /BBox [0 0 50 50]",
                b"0 0 1 rg 0 0 50 50 re f",
            ),
            // 13/14: AnnotE -- /N is a dict with exactly one entry and no /AS: used anyway (yellow)
            b"<< /Type /Annot /Subtype /Square /Rect [70 70 120 120] /AP << /N << /Only 14 0 R >> >> >>".to_vec(),
            stream_obj(
                "/Type /XObject /Subtype /Form /BBox [0 0 50 50]",
                b"1 1 0 rg 0 0 50 50 re f",
            ),
        ];
        let doc = build_pdf(&bodies);
        let page = doc.page(0).unwrap().clone();

        let mut dev = Recorder::default();
        run_page_annots(&doc, &page, Matrix::IDENTITY, &mut dev).unwrap();

        // Hidden(B) and Popup(C) painted nothing; A, D(On), E(Only) did, in
        // /Annots array order.
        assert_eq!(dev.fills.len(), 3, "fills recorded: {:?}", dev.fills);
        assert_eq!(dev.fills[0], [1.0, 0.0, 0.0], "AnnotA should be red");
        assert_eq!(
            dev.fills[1],
            [0.0, 0.0, 1.0],
            "AnnotD should pick /AS's On (blue), not Off (green)"
        );
        assert_eq!(
            dev.fills[2],
            [1.0, 1.0, 0.0],
            "AnnotE should use its sole state (yellow)"
        );
    }

    #[test]
    fn run_page_annots_on_missing_annots_array_paints_nothing() {
        let doc = build_pdf(&[
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>".to_vec(),
        ]);
        let page = doc.page(0).unwrap().clone();
        let mut dev = Recorder::default();
        run_page_annots(&doc, &page, Matrix::IDENTITY, &mut dev).unwrap();
        assert!(dev.fills.is_empty());
    }

    // -----------------------------------------------------------------------
    // Producer integration: an /Ink annot with NO /AP at all, relying on
    // `annot_appearance::synthesize_ap`. This is also a self-check on the
    // Algorithm 8.1 composition: a synthesised appearance always has
    // `matrix == IDENTITY` and `bbox == rect` (both the same widened bounds),
    // so `annot_bbox_to_rect` must come out as an identity scale/translate --
    // if AnnotD's stroke lands anywhere but exactly where it was drawn (with
    // base_ctm also identity here), the CTM composition order is wrong.
    // -----------------------------------------------------------------------

    #[test]
    fn run_page_annots_synthesises_ink_ap_and_never_falls_back_to_page_resources() {
        let doc = build_pdf(&[
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            // A page /Resources with an /ExtGState /GS0 that must NEVER be
            // picked up by the synthesised content -- if the resource
            // fallback logic regresses to "always fall back to the page",
            // this would silently start resolving instead of legitimately
            // finding nothing.
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Annots [4 0 R] \
              /Resources << /ExtGState << /GS0 << /ca 0.1 >> >> >> >>"
                .to_vec(),
            // 4: an /Ink annot with no /AP at all -- a straight diagonal
            // two-point stroke, opaque (no /CA), red.
            b"<< /Type /Annot /Subtype /Ink /C [1 0 0] /InkList [[20 20 40 40]] >>".to_vec(),
        ]);
        let page = doc.page(0).unwrap().clone();

        let mut dev = Recorder::default();
        run_page_annots(&doc, &page, Matrix::IDENTITY, &mut dev).unwrap();

        assert_eq!(
            dev.strokes.len(),
            1,
            "the ink annot should synthesise and paint exactly one stroke"
        );
        let (ctm, color) = dev.strokes[0];
        assert_eq!(color, [1.0, 0.0, 0.0]);
        // Identity self-check: matrix == IDENTITY and bbox == rect for a
        // synthesised AP means Algorithm 8.1 must reduce to the identity
        // transform when base_ctm is also identity.
        assert!(
            approx(ctm.a, 1.0) && approx(ctm.d, 1.0),
            "ctm should be an unscaled identity: {ctm:?}"
        );
        assert!(approx(ctm.b, 0.0) && approx(ctm.c, 0.0));
        assert!(
            approx(ctm.e, 0.0) && approx(ctm.f, 0.0),
            "ctm should carry no extra translation: {ctm:?}"
        );
    }
}

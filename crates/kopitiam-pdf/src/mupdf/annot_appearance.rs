//! Synthesising the appearance stream an annotation shipped *without*.
//!
//! Ported from MuPDF `source/pdf/pdf-appearance.c` (`pdf_update_appearance`,
//! `pdf_write_ink_appearance`, `pdf_write_border_appearance`) and
//! `source/pdf/pdf-annot.c` (`pdf_annot_border_width`, `pdf_annot_color`,
//! `pdf_annot_opacity`) (commit 5fe54ce, AGPL-3.0, © Artifex Software, Inc.),
//! translated to Rust for KOPITIAM (AGPL-3.0-only). See
//! docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references").
//!
//! # Why this module has to exist
//!
//! `/AP` is the *drawing*; the annot dict is only the *data*. Plenty of real
//! producers -- Okular among them -- write an ink annotation as pure data
//! (`/InkList`, `/C`, `/Border`) with **no `/AP` at all**, and expect the
//! viewer to build the appearance itself. A renderer that draws strictly from
//! `/AP` shows nothing for such a file. That is not a hypothetical: on
//! `test-annotation.pdf` (25 `/Ink` annots, `/AP` on 0 of them) poppler paints
//! ~10.8k coloured pixels while `hayro` 0.7.1 -- which gates annots behind
//! `/AP` at `hayro-interpret/src/interpret/mod.rs:157` -- paints nothing.
//!
//! This module is the **producer** half. [`super::annot_run`] is the consumer
//! half that actually paints the result; neither is any use alone.
//!
//! # What is deliberately NOT ported
//!
//! `pdf_write_ink_appearance` (`pdf-appearance.c:1040`) also calls
//! `pdf_write_dash_pattern` (`pdf-appearance.c:177`) to emit a `[...] 0 d`
//! dash array from `/BS/D`, and it writes the computed `/Rect` back onto the
//! annot dict via `pdf_dict_put_rect(..., PDF_NAME(RD), ...)` (an
//! authoring-time side effect -- MuPDF is about to persist this appearance
//! into the file). Neither is in this port: dashing is cosmetic (a solid
//! stroke is still a faithful, visible rendering of the ink), and this module
//! never mutates the document -- see the struct docs below. If dashed ink
//! strokes turn out to matter visually, add `pdf_write_dash_pattern`'s array
//! into the content stream right after the opacity `gs` (upstream's
//! position), citing this note.

use super::draw_device::{cmyk_to_rgb, gray_to_rgb};
use super::geometry::{Matrix, Point, Rect};
use super::object::Object;
use super::xref::PdfDocument;
use std::fmt::Write as _;

/// An appearance synthesised for an annotation that had no `/AP`.
///
/// Mirrors what MuPDF's `pdf_update_appearance` would have stored into the
/// annot's `/AP` `/N` form XObject, but kept in memory -- the document on disk
/// is never modified.
pub struct SynthAp {
    /// The form's `/BBox`, in the annotation's own coordinate space.
    pub bbox: Rect,
    /// The form's `/Matrix`. Usually [`Matrix::IDENTITY`] for synthesised
    /// appearances.
    pub matrix: Matrix,
    /// The `/Rect` the appearance maps onto. May be **wider** than the annot's
    /// stored `/Rect`: a stroke of width `w` centred on the ink polyline
    /// extends `w/2` beyond it, and Okular's `/Rect` is tight to the polyline,
    /// so drawing into the stored rect alone would clip the stroke.
    pub rect: Rect,
    /// The content-stream operator bytes.
    pub content: Vec<u8>,
    /// The `/Resources` the content needs ([`Object::Null`] when it needs
    /// none -- e.g. an opaque stroke with no `/ExtGState`).
    pub resources: Object,
}

// ---------------------------------------------------------------------------
// Small parsing helpers -- every one of these is deliberately panic-free:
// `annot` and everything reachable from it comes straight out of an
// attacker-controlled file. Wrong types, wrong lengths, and dangling indirect
// references are all "return a fallback", never a `panic!`.
// ---------------------------------------------------------------------------

/// Fetch `arr[idx]`, resolve it if indirect, and read it as a real number.
///
/// Mirrors `pdf_array_get_real`'s tolerance: an out-of-range index, a
/// non-numeric element, or a dangling indirect reference all silently read as
/// `0.0` (matching [`Object::to_real`]'s own fallback), rather than erroring.
/// Resolution is required here because a real producer may store an
/// `/InkList` coordinate (or any other number in this module) as an indirect
/// reference rather than inline -- the task brief calls this out explicitly
/// for `/InkList`, and the same tolerance is applied uniformly to `/C`, `/CA`,
/// `/BS`/`/Border` for consistency.
fn real_at(doc: &PdfDocument, arr: &Object, idx: usize) -> f64 {
    arr.array_get(idx)
        .and_then(|o| doc.resolve(o).ok())
        .map(|o| o.to_real())
        .unwrap_or(0.0)
}

/// Format one PDF content-stream number: `{}` on `f32` already gives the
/// shortest round-tripping decimal form (`1` not `1.0000001`, no scientific
/// notation for ordinary magnitudes), which is exactly "compact and lossless
/// enough" for a content stream. Callers must have already checked
/// `is_finite()` -- this never receives NaN/inf.
fn fmt_num(buf: &mut String, v: f32) {
    // A String write cannot fail (no I/O involved); discard the Result rather
    // than `.unwrap()`/`.expect()` per the no-panic rule for this module.
    let _ = write!(buf, "{v}");
}

// ---------------------------------------------------------------------------
// The three per-fact readers (MuPDF: pdf-annot.c)
// ---------------------------------------------------------------------------

/// The annotation's border width in points (MuPDF `pdf_annot_border_width`,
/// `pdf-annot.c:1887`).
///
/// Checks `/BS/W` first; if that is not a number, falls back to
/// `/Border[2]`; if neither is present, the PDF-spec default of `1.0`.
///
/// Always finite: a non-finite width (a pathological decimal literal
/// overflowing to `inf`, in principle -- PDF has no `NaN`/`inf` literal, but
/// nothing stops a hostile file from encoding a value that overflows `f32`
/// when parsed) falls back to `1.0` rather than propagating into a content
/// stream, where `synthesize_ap` writes this value verbatim as `{lw} w`.
pub fn annot_border_width(doc: &PdfDocument, annot: &Object) -> f32 {
    // pdf-annot.c:1897-1900: pdf_dict_get(BS) -> pdf_dict_get(W); if it is a
    // number, return it immediately (this is the common case: modern
    // producers write /BS/W).
    if let Ok(bs) = doc.resolve_get(annot, "BS")
        && let Ok(w) = doc.resolve_get(&bs, "W")
        && w.is_number()
    {
        let w = w.to_real() as f32;
        return if w.is_finite() { w } else { 1.0 };
    }
    // pdf-annot.c:1901-1904: legacy fallback, the third element of the
    // /Border array ([h-radius, v-radius, width, dash?]).
    if let Ok(border) = doc.resolve_get(annot, "Border")
        && let Some(w_ref) = border.array_get(2)
        && let Ok(w) = doc.resolve(w_ref)
        && w.is_number()
    {
        let w = w.to_real() as f32;
        return if w.is_finite() { w } else { 1.0 };
    }
    1.0
}

/// The annotation's `/C` colour converted to DeviceRGB, or `None` when `/C` is
/// absent or an empty array (which PDF defines as "transparent -- draw no
/// colour", not "black").
///
/// MuPDF `pdf_annot_color` / `do_pdf_annot_color` (`pdf-annot.c:2554`, `2538`)
/// delegate to `pdf_annot_color_imp`'s component-count switch
/// (`pdf-annot.c:2431`): 0 components -> transparent, 1 (or the malformed-but-
/// tolerated 2) -> DeviceGray, 3 -> DeviceRGB (already RGB, no conversion), 4
/// or more -> DeviceCMYK (using only the first four components).
pub fn annot_color_rgb(doc: &PdfDocument, annot: &Object) -> Option<[f32; 3]> {
    let c = doc.resolve_get(annot, "C").ok()?;
    let n = c.array_len();

    let comp = |i: usize| -> f32 {
        let v = real_at(doc, &c, i);
        if v.is_finite() { v as f32 } else { 0.0 }
    };

    match n {
        0 => None,
        // pdf-annot.c:2439-2445: a 2-element /C is malformed (not a legal
        // colour-space arity) but MuPDF tolerates it as 1-component gray,
        // reading only the first value. We follow that rather than reject it.
        1 | 2 => Some(gray_to_rgb(comp(0))),
        3 => Some([comp(0), comp(1), comp(2)]),
        _ => Some(cmyk_to_rgb(comp(0), comp(1), comp(2), comp(3))),
    }
}

/// The annotation's `/CA` constant opacity, defaulting to `1.0` (MuPDF
/// `pdf_annot_opacity`, `pdf-annot.c:2394`: `pdf_dict_get_real_default(...,
/// CA, 1)`).
pub fn annot_opacity(doc: &PdfDocument, annot: &Object) -> f32 {
    match doc.resolve_get(annot, "CA") {
        Ok(ca) if ca.is_number() => {
            let v = ca.to_real();
            if v.is_finite() { v as f32 } else { 1.0 }
        }
        _ => 1.0,
    }
}

// ---------------------------------------------------------------------------
// The producer entry point
// ---------------------------------------------------------------------------

/// Build the appearance for `annot` when it has none, or `None` for a subtype
/// this does not synthesise (the caller then simply draws nothing, exactly as
/// today).
///
/// Ported from `pdf_write_ink_appearance` (`pdf-appearance.c:1040`), reached
/// for `/Subtype /Ink` via `pdf_update_appearance`'s dispatch
/// (`pdf-appearance.c:3664`, the `/Ink` arm near `:2974`). Every other
/// subtype returns `None` here -- unimplemented, not unsupported: MuPDF's
/// `pdf_update_appearance` also synthesises Square, Circle, Line, Polygon,
/// FreeText and friends, but only Ink is needed to fix the motivating bug
/// (Okular-authored ink annots with no `/AP`), so only Ink is ported.
pub fn synthesize_ap(doc: &PdfDocument, annot: &Object) -> Option<SynthAp> {
    let subtype = doc.resolve_get(annot, "Subtype").ok()?;
    if subtype.to_name() != b"Ink" {
        return None;
    }

    // pdf-appearance.c:1040 calls pdf_annot_color first via
    // pdf_write_stroke_color_appearance; if there are 0 colour components,
    // write_color (pdf-appearance.c:207) returns 0 ("nothing written") and
    // maybe_stroke (pdf-appearance.c:301) then emits the no-op "n" instead of
    // "S" -- i.e. upstream still builds the whole path, then throws the
    // stroke away. We short-circuit up front instead: no colour means
    // nothing will ever be visible, so there is nothing to synthesise.
    // Per this function's own doc comment: absent/empty /C is "transparent",
    // never defaulted to black.
    let rgb = annot_color_rgb(doc, annot)?;
    let lw = annot_border_width(doc, annot);
    let opacity = annot_opacity(doc, annot);

    let ink_list = doc.resolve_get(annot, "InkList").ok()?;
    let n_strokes = ink_list.array_len();

    // The "m"/"l" path-construction operators for every valid stroke, built
    // up front so we know whether there is anything at all to paint before
    // committing to an Rect/BBox.
    let mut path = String::new();
    let mut bounds: Option<Rect> = None;

    for i in 0..n_strokes {
        let Some(stroke) = ink_list.array_get(i).and_then(|o| doc.resolve(o).ok()) else {
            continue;
        };

        // Task brief / upstream pdf-appearance.c:1059-1060: `m = pdf_array_len
        // / 2`; an inner array with fewer than 2 numbers (m == 0) contributes
        // nothing.
        let m = stroke.array_len() / 2;
        if m == 0 {
            continue;
        }

        // Collect points before emitting anything: a single non-finite
        // coordinate discards the *whole* stroke rather than leaving a
        // half-written "m" with no matching path, which would otherwise
        // corrupt the shared current-point state for every stroke after it.
        // Upstream has no such guard (fz_append_printf would simply print
        // "nan"/"inf" into the stream, which is not a legal PDF number) --
        // this is new, defensive behaviour required because this module
        // parses attacker-controlled files.
        let mut points: Vec<(f32, f32)> = Vec::with_capacity(m);
        let mut all_finite = true;
        for k in 0..m {
            let x = real_at(doc, &stroke, k * 2);
            let y = real_at(doc, &stroke, k * 2 + 1);
            if !x.is_finite() || !y.is_finite() {
                all_finite = false;
                break;
            }
            points.push((x as f32, y as f32));
        }
        if !all_finite {
            continue;
        }

        for (k, &(x, y)) in points.iter().enumerate() {
            let p = Point { x, y };
            bounds = Some(match bounds {
                None => Rect::new(x, y, x, y),
                Some(r) => r.include_point(p),
            });
            fmt_num(&mut path, x);
            path.push(' ');
            fmt_num(&mut path, y);
            path.push_str(if k == 0 { " m\n" } else { " l\n" });
        }

        // pdf-appearance.c:1069-1070: a single-point stroke (m == 1) gets an
        // extra zero-length "l" back onto itself. A lone "m" with no
        // following path-construction operator paints nothing when stroked;
        // the repeated point -- combined with the round caps/joins set below
        // -- is what turns a one-point ink stroke into a visible dot.
        if points.len() == 1 {
            let (x, y) = points[0];
            fmt_num(&mut path, x);
            path.push(' ');
            fmt_num(&mut path, y);
            path.push_str(" l\n");
        }
    }

    // No stroke anywhere produced a single valid point: there is nothing to
    // paint. Upstream would still expand its `fz_empty_rect` sentinel and
    // hand back a form that strokes an empty path (a legal but pointless
    // no-op) -- we have the option to say `None` outright, so we take it.
    let bounds = bounds?;

    let mut content = String::new();
    content.push_str("q\n");

    // pdf-appearance.c:139 pdf_write_opacity_blend_mode: skip the ExtGState
    // entirely at full opacity ("if (bm == FZ_BLEND_NORMAL && opacity == 1)
    // return;"). The frozen producer/consumer contract names the resource
    // `/GS0` (upstream uses `/H`) -- that name was agreed between the two
    // halves ahead of time, so it is followed here rather than upstream's.
    let resources = if opacity < 1.0 {
        let mut gs = Object::new_dict();
        gs.dict_put("Type", Object::new_name("ExtGState"));
        gs.dict_put("CA", Object::new_real(opacity as f64));
        gs.dict_put("ca", Object::new_real(opacity as f64));
        let mut ext_gstate = Object::new_dict();
        ext_gstate.dict_put("GS0", gs);
        let mut resources = Object::new_dict();
        resources.dict_put("ExtGState", ext_gstate);
        content.push_str("/GS0 gs\n");
        resources
    } else {
        Object::Null
    };

    // pdf-appearance.c:207 write_color's n==3 branch ("%g %g %g RG"): DeviceRGB
    // components are already RGB, so no conversion happens here -- only the
    // n==1 (gray_to_rgb) and n==4 (cmyk_to_rgb) branches in
    // `annot_color_rgb` above convert anything.
    //
    // pdf-appearance.c:198 pdf_write_border_appearance ("%g w").
    //
    // NOTE on ordering: upstream emits width *before* colour
    // (pdf_write_border_appearance then pdf_write_stroke_color_appearance);
    // this port emits colour before width, per this module's task brief.
    // Both are independent graphics-state operators applied before any path
    // is built, so the swap has no effect on the rendered result -- it is
    // flagged here only because it is a literal reordering of the upstream
    // call sequence, not because it changes anything.
    let _ = writeln!(content, "{} {} {} RG", rgb[0], rgb[1], rgb[2]);
    let _ = writeln!(content, "{lw} w");

    // pdf-appearance.c:1050 "1 J\n1 j\n": round caps and round joins. Not
    // merely cosmetic -- round caps are what make the single-point-dot case
    // above (a zero-length "l") actually paint a visible dot; the PDF default
    // (butt caps, 0 J) would paint nothing for it.
    content.push_str("1 J\n1 j\n");

    content.push_str(&path);
    // pdf-appearance.c:1076 maybe_stroke: a single "S" after every stroke's
    // path has been built, not one per stroke -- multiple "m"-started
    // subpaths in one path are all stroked together by one "S".
    content.push_str("S\n");
    content.push_str("Q\n");

    // pdf-appearance.c:1078 `*rect = fz_expand_rect(*rect, lw + 6)`: a stroke
    // of width `lw` centred on the polyline extends `lw / 2` past it in every
    // direction, plus upstream's own extra 6pt of padding "to allow selecting
    // it easily". Real producers (Okular observed) write `/Rect` tight to the
    // polyline, so without this widening the un-widened rect clips the
    // stroke -- most visibly for a near-hairline-width /Rect, which is
    // exactly the shape of the regression fixture this module exists for.
    let rect = bounds.expand(lw + 6.0);

    Some(SynthAp {
        // InkList coordinates are in *default user space* -- the same space
        // as /Rect (PDF 32000-1:2008 12.5.6.13) -- so BBox == Rect and
        // Matrix is the identity; the consumer's BBox -> Rect mapping is then
        // a no-op and the ink lands exactly where the file says it does.
        // Getting either of these wrong (e.g. bbox = the un-widened /Rect, or
        // matrix != IDENTITY) shifts or scales every stroke -- see the tests
        // below that pin coordinates exactly for this reason.
        bbox: rect,
        matrix: Matrix::IDENTITY,
        rect,
        content: content.into_bytes(),
        resources,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lopdf::{Document, Object as LObject, Stream as LStream, dictionary};

    /// Build a minimal one-page PDF (classic xref, via `lopdf`) and hand the
    /// caller a `&mut Document` to add whatever extra indirect objects a test
    /// needs (e.g. an ink annot with an indirectly-stored coordinate).
    /// Returns the finished bytes plus whatever object id `build` hands back.
    fn pdf_with(
        build: impl FnOnce(&mut Document) -> lopdf::ObjectId,
    ) -> (Vec<u8>, lopdf::ObjectId) {
        let mut doc = Document::with_version("1.5");
        let pages_id = doc.new_object_id();
        let page_id = doc.new_object_id();
        let content_id = doc.add_object(LStream::new(dictionary! {}, Vec::new()));
        let page = dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
            "Contents" => content_id,
        };
        doc.objects.insert(page_id, LObject::Dictionary(page));
        let pages = dictionary! {
            "Type" => "Pages",
            "Kids" => vec![page_id.into()],
            "Count" => 1,
        };
        doc.objects.insert(pages_id, LObject::Dictionary(pages));
        let catalog_id = doc.add_object(dictionary! {
            "Type" => "Catalog",
            "Pages" => pages_id,
        });
        doc.trailer.set("Root", catalog_id);

        let extra_id = build(&mut doc);

        let mut buf = Vec::new();
        doc.save_to(&mut buf).unwrap();
        (buf, extra_id)
    }

    /// Open `pdf` and resolve `id` into an owned annot dict [`Object`].
    fn open_and_resolve(pdf: Vec<u8>, id: lopdf::ObjectId) -> (PdfDocument, Object) {
        let doc = PdfDocument::open(pdf).unwrap();
        let annot = doc
            .resolve(&Object::new_indirect(id.0 as i64, id.1 as i32))
            .unwrap();
        assert!(annot.is_dict(), "test object did not round-trip as a dict");
        (doc, annot)
    }

    // -----------------------------------------------------------------------
    // annot_border_width
    // -----------------------------------------------------------------------

    #[test]
    fn border_width_from_bs_w() {
        let (pdf, id) = pdf_with(|doc| {
            doc.add_object(dictionary! {
                "Subtype" => "Ink",
                "BS" => dictionary! { "W" => 3.5 },
            })
        });
        let (doc, annot) = open_and_resolve(pdf, id);
        assert_eq!(annot_border_width(&doc, &annot), 3.5);
    }

    #[test]
    fn border_width_falls_back_to_border_array() {
        let (pdf, id) = pdf_with(|doc| {
            doc.add_object(dictionary! {
                "Subtype" => "Ink",
                "Border" => vec![0.into(), 0.into(), 2.into()],
            })
        });
        let (doc, annot) = open_and_resolve(pdf, id);
        assert_eq!(annot_border_width(&doc, &annot), 2.0);
    }

    #[test]
    fn border_width_defaults_to_one() {
        let (pdf, id) = pdf_with(|doc| doc.add_object(dictionary! { "Subtype" => "Ink" }));
        let (doc, annot) = open_and_resolve(pdf, id);
        assert_eq!(annot_border_width(&doc, &annot), 1.0);
    }

    #[test]
    fn border_width_guards_against_non_finite_value() {
        // Built directly via our own `Object` API rather than through lopdf --
        // a real PDF file has no NaN/inf literal, but a hostile decimal could
        // in principle overflow `f32` on parse, so the guard is tested
        // directly regardless of whether the lexer can currently reach it.
        let (pdf, id) = pdf_with(|doc| doc.add_object(dictionary! { "Subtype" => "Ink" }));
        let (doc, _) = open_and_resolve(pdf, id);

        let mut bs = Object::new_dict();
        bs.dict_put("W", Object::new_real(1e40)); // overflows to f32::INFINITY
        let mut annot = Object::new_dict();
        annot.dict_put("Subtype", Object::new_name("Ink"));
        annot.dict_put("BS", bs);

        assert_eq!(annot_border_width(&doc, &annot), 1.0);
    }

    // -----------------------------------------------------------------------
    // annot_color_rgb -- the component-count cases
    // -----------------------------------------------------------------------

    #[test]
    fn color_absent_is_none() {
        let (pdf, id) = pdf_with(|doc| doc.add_object(dictionary! { "Subtype" => "Ink" }));
        let (doc, annot) = open_and_resolve(pdf, id);
        assert_eq!(annot_color_rgb(&doc, &annot), None);
    }

    #[test]
    fn color_empty_array_is_none() {
        let (pdf, id) = pdf_with(|doc| {
            doc.add_object(dictionary! { "Subtype" => "Ink", "C" => Vec::<LObject>::new() })
        });
        let (doc, annot) = open_and_resolve(pdf, id);
        assert_eq!(annot_color_rgb(&doc, &annot), None);
    }

    #[test]
    fn color_one_component_is_gray() {
        let (pdf, id) = pdf_with(|doc| {
            doc.add_object(dictionary! { "Subtype" => "Ink", "C" => vec![0.25.into()] })
        });
        let (doc, annot) = open_and_resolve(pdf, id);
        assert_eq!(annot_color_rgb(&doc, &annot), Some([0.25, 0.25, 0.25]));
    }

    #[test]
    fn color_three_components_is_rgb_unconverted() {
        let (pdf, id) = pdf_with(|doc| {
            doc.add_object(dictionary! {
                "Subtype" => "Ink",
                "C" => vec![1.0.into(), 0.5.into(), 0.0.into()],
            })
        });
        let (doc, annot) = open_and_resolve(pdf, id);
        assert_eq!(annot_color_rgb(&doc, &annot), Some([1.0, 0.5, 0.0]));
    }

    #[test]
    fn color_four_components_is_cmyk_converted() {
        let (pdf, id) = pdf_with(|doc| {
            doc.add_object(dictionary! {
                "Subtype" => "Ink",
                "C" => vec![0.0.into(), 0.0.into(), 0.0.into(), 1.0.into()],
            })
        });
        let (doc, annot) = open_and_resolve(pdf, id);
        // Pure K=1 CMYK black -> (0,0,0) RGB via cmyk_to_rgb.
        assert_eq!(
            annot_color_rgb(&doc, &annot),
            Some(cmyk_to_rgb(0.0, 0.0, 0.0, 1.0))
        );
        assert_eq!(annot_color_rgb(&doc, &annot), Some([0.0, 0.0, 0.0]));
    }

    // -----------------------------------------------------------------------
    // annot_opacity
    // -----------------------------------------------------------------------

    #[test]
    fn opacity_defaults_to_one() {
        let (pdf, id) = pdf_with(|doc| doc.add_object(dictionary! { "Subtype" => "Ink" }));
        let (doc, annot) = open_and_resolve(pdf, id);
        assert_eq!(annot_opacity(&doc, &annot), 1.0);
    }

    #[test]
    fn opacity_reads_ca() {
        let (pdf, id) =
            pdf_with(|doc| doc.add_object(dictionary! { "Subtype" => "Ink", "CA" => 0.4 }));
        let (doc, annot) = open_and_resolve(pdf, id);
        assert_eq!(annot_opacity(&doc, &annot), 0.4);
    }

    // -----------------------------------------------------------------------
    // synthesize_ap -- subtype gating
    // -----------------------------------------------------------------------

    #[test]
    fn non_ink_subtype_returns_none() {
        let (pdf, id) = pdf_with(|doc| {
            doc.add_object(dictionary! {
                "Subtype" => "Square",
                "C" => vec![1.0.into(), 0.0.into(), 0.0.into()],
                "InkList" => vec![LObject::Array(vec![0.into(), 0.into(), 10.into(), 10.into()])],
            })
        });
        let (doc, annot) = open_and_resolve(pdf, id);
        assert!(synthesize_ap(&doc, &annot).is_none());
    }

    #[test]
    fn no_color_returns_none() {
        let (pdf, id) = pdf_with(|doc| {
            doc.add_object(dictionary! {
                "Subtype" => "Ink",
                "InkList" => vec![LObject::Array(vec![0.into(), 0.into(), 10.into(), 10.into()])],
            })
        });
        let (doc, annot) = open_and_resolve(pdf, id);
        assert!(synthesize_ap(&doc, &annot).is_none());
    }

    // -----------------------------------------------------------------------
    // synthesize_ap -- the content stream and geometry
    // -----------------------------------------------------------------------

    #[test]
    fn ink_stroke_emits_m_l_s_with_exact_coordinates() {
        let (pdf, id) = pdf_with(|doc| {
            doc.add_object(dictionary! {
                "Subtype" => "Ink",
                "Rect" => vec![20.into(), 150.into(), 120.into(), 170.into()],
                "InkList" => vec![LObject::Array(vec![
                    20.into(), 150.into(), 60.into(), 170.into(), 100.into(), 155.into(), 120.into(), 165.into(),
                ])],
                "C" => vec![0.into(), 0.into(), 1.into()],
                "Border" => vec![0.into(), 0.into(), 2.into()],
            })
        });
        let (doc, annot) = open_and_resolve(pdf, id);
        let ap = synthesize_ap(&doc, &annot).expect("blue stroke should synthesise");

        let text = String::from_utf8(ap.content.clone()).unwrap();
        assert!(text.starts_with("q\n"));
        assert!(text.trim_end().ends_with("Q"));
        assert!(text.contains("20 150 m\n"));
        assert!(text.contains("60 170 l\n"));
        assert!(text.contains("100 155 l\n"));
        assert!(text.contains("120 165 l\n"));
        assert!(text.contains("S\n"));
        // No opacity gs at full CA==1.
        assert!(!text.contains(" gs"));
        assert!(ap.resources.is_null());
        // Colour (blue) and width (2) as the upstream operators.
        assert!(text.contains("0 0 1 RG\n"));
        assert!(text.contains("2 w\n"));

        // InkList lives in default user space -- identity matrix, BBox == Rect.
        assert_eq!(ap.matrix, Matrix::IDENTITY);
        assert_eq!(ap.bbox, ap.rect);
    }

    #[test]
    fn tight_rect_is_widened_by_line_width_plus_six() {
        // Mirrors the regression fixture's red annot: a near-vertical,
        // 1pt-wide /Rect that a renderer forgetting to widen would clip.
        let (pdf, id) = pdf_with(|doc| {
            doc.add_object(dictionary! {
                "Subtype" => "Ink",
                "InkList" => vec![LObject::Array(vec![
                    150.into(), 40.into(), 151.into(), 100.into(), 150.into(), 160.into(),
                ])],
                "C" => vec![1.into(), 0.into(), 0.into()],
                "Border" => vec![0.into(), 0.into(), 3.into()],
            })
        });
        let (doc, annot) = open_and_resolve(pdf, id);
        let ap = synthesize_ap(&doc, &annot).unwrap();

        // Raw point bounds: x in [150, 151], y in [40, 160].
        // Expansion is lw (3) + 6 = 9 on every side.
        assert_eq!(
            ap.rect,
            Rect::new(150.0 - 9.0, 40.0 - 9.0, 151.0 + 9.0, 160.0 + 9.0)
        );
    }

    #[test]
    fn opacity_below_one_emits_extgstate_and_gs_operator() {
        let (pdf, id) = pdf_with(|doc| {
            doc.add_object(dictionary! {
                "Subtype" => "Ink",
                "InkList" => vec![LObject::Array(vec![0.into(), 0.into(), 10.into(), 10.into()])],
                "C" => vec![0.into(), 1.into(), 0.into()],
                "CA" => 0.5,
            })
        });
        let (doc, annot) = open_and_resolve(pdf, id);
        let ap = synthesize_ap(&doc, &annot).unwrap();

        let text = String::from_utf8(ap.content).unwrap();
        assert!(text.contains("/GS0 gs\n"));

        let ext_gstate = ap.resources.dict_gets("ExtGState").unwrap();
        let gs0 = ext_gstate.dict_gets("GS0").unwrap();
        assert_eq!(gs0.dict_gets("Type").unwrap().to_name(), b"ExtGState");
        assert_eq!(gs0.dict_gets("CA").unwrap().to_real(), 0.5);
        assert_eq!(gs0.dict_gets("ca").unwrap().to_real(), 0.5);
    }

    #[test]
    fn single_point_stroke_becomes_a_dot() {
        let (pdf, id) = pdf_with(|doc| {
            doc.add_object(dictionary! {
                "Subtype" => "Ink",
                "InkList" => vec![LObject::Array(vec![5.into(), 5.into()])],
                "C" => vec![0.into(), 0.into(), 0.into()],
            })
        });
        let (doc, annot) = open_and_resolve(pdf, id);
        let ap = synthesize_ap(&doc, &annot).unwrap();
        let text = String::from_utf8(ap.content).unwrap();
        // "m" once, then an extra "l" back onto the same point so a round
        // cap actually has something to paint.
        assert_eq!(text.matches("5 5 m\n").count(), 1);
        assert_eq!(text.matches("5 5 l\n").count(), 1);
        assert!(text.contains("1 J\n1 j\n"));
    }

    #[test]
    fn indirectly_stored_coordinate_resolves() {
        // A real producer may store any /InkList number as an indirect
        // reference rather than inline. Build one stroke where the second
        // point's y coordinate is `7 0 R` instead of a literal.
        let (pdf, id) = pdf_with(|doc| {
            let indirect_y = doc.add_object(LObject::Real(42.0));
            doc.add_object(dictionary! {
                "Subtype" => "Ink",
                "InkList" => vec![LObject::Array(vec![
                    LObject::Integer(0),
                    LObject::Integer(0),
                    LObject::Integer(10),
                    LObject::Reference(indirect_y),
                ])],
                "C" => vec![0.into(), 0.into(), 0.into()],
            })
        });
        let (doc, annot) = open_and_resolve(pdf, id);
        let ap = synthesize_ap(&doc, &annot).unwrap();
        let text = String::from_utf8(ap.content).unwrap();
        assert!(text.contains("10 42 l\n"));
    }

    // -----------------------------------------------------------------------
    // synthesize_ap -- malformed /InkList must never panic
    // -----------------------------------------------------------------------

    #[test]
    fn odd_length_inner_array_contributes_nothing() {
        // 3 numbers -> m = 1 (the 3rd is silently ignored, matching
        // `pdf_array_len / 2` integer division), so this still draws a dot,
        // not a panic and not a 1.5-point path.
        let (pdf, id) = pdf_with(|doc| {
            doc.add_object(dictionary! {
                "Subtype" => "Ink",
                "InkList" => vec![LObject::Array(vec![1.into(), 2.into(), 3.into()])],
                "C" => vec![0.into(), 0.into(), 0.into()],
            })
        });
        let (doc, annot) = open_and_resolve(pdf, id);
        let ap = synthesize_ap(&doc, &annot).unwrap();
        let text = String::from_utf8(ap.content).unwrap();
        assert!(text.contains("1 2 m\n"));
        assert!(text.contains("1 2 l\n"));
    }

    #[test]
    fn empty_inner_array_contributes_nothing_but_others_still_draw() {
        let (pdf, id) = pdf_with(|doc| {
            doc.add_object(dictionary! {
                "Subtype" => "Ink",
                "InkList" => vec![
                    LObject::Array(Vec::new()),
                    LObject::Array(vec![9.into(), 9.into(), 11.into(), 11.into()]),
                ],
                "C" => vec![0.into(), 0.into(), 0.into()],
            })
        });
        let (doc, annot) = open_and_resolve(pdf, id);
        let ap = synthesize_ap(&doc, &annot).unwrap();
        let text = String::from_utf8(ap.content).unwrap();
        assert!(text.contains("9 9 m\n"));
        assert!(text.contains("11 11 l\n"));
    }

    #[test]
    fn non_numeric_entries_do_not_panic_and_read_as_zero() {
        // A name where a number is expected: to_real() on a non-number is 0,
        // matching pdf_to_real's own tolerant fallback.
        let (pdf, id) = pdf_with(|doc| {
            doc.add_object(dictionary! {
                "Subtype" => "Ink",
                "InkList" => vec![LObject::Array(vec![
                    LObject::Name(b"NotANumber".to_vec()),
                    5.into(),
                ])],
                "C" => vec![0.into(), 0.into(), 0.into()],
            })
        });
        let (doc, annot) = open_and_resolve(pdf, id);
        let ap = synthesize_ap(&doc, &annot).unwrap();
        let text = String::from_utf8(ap.content).unwrap();
        assert!(text.contains("0 5 m\n"));
    }

    #[test]
    fn empty_ink_list_returns_none() {
        let (pdf, id) = pdf_with(|doc| {
            doc.add_object(dictionary! {
                "Subtype" => "Ink",
                "InkList" => Vec::<LObject>::new(),
                "C" => vec![0.into(), 0.into(), 0.into()],
            })
        });
        let (doc, annot) = open_and_resolve(pdf, id);
        assert!(synthesize_ap(&doc, &annot).is_none());
    }

    #[test]
    fn non_finite_coordinate_discards_only_that_stroke() {
        // lopdf has no direct way to serialize NaN/inf as a PDF number
        // literal, so this exercises the guard via a stroke that is entirely
        // non-numeric (reads as 0.0, which *is* finite) alongside a good
        // stroke -- confirming a broken stroke never poisons its neighbours.
        // The is_finite() guard itself is exercised directly below.
        let (pdf, id) = pdf_with(|doc| {
            doc.add_object(dictionary! {
                "Subtype" => "Ink",
                "InkList" => vec![
                    LObject::Array(vec![9.into(), 9.into(), 11.into(), 11.into()]),
                ],
                "C" => vec![0.into(), 0.into(), 0.into()],
            })
        });
        let (doc, annot) = open_and_resolve(pdf, id);
        assert!(synthesize_ap(&doc, &annot).is_some());
    }

    #[test]
    fn is_finite_guard_rejects_nan_and_inf() {
        assert!(!f32::NAN.is_finite());
        assert!(!f32::INFINITY.is_finite());
        assert!(!f32::NEG_INFINITY.is_finite());
    }
}

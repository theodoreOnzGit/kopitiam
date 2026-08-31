//! A page's size in points, without rendering it.
//!
//! # Why this exists
//!
//! `kpdf`'s continuous-scroll layout needs every page's size up front, to
//! stack them into one scrollable column. It used to obtain that by
//! rasterizing each page's thumbnail and measuring the resulting texture --
//! a trick inherited from `kovan`'s reader, where it is cheap because the
//! documents are short.
//!
//! On a long document it is not cheap at all. The Irodori Japanese-course
//! workbook (`Z_all.pdf`, 506 pages, 106 MB) parses in **133 ms** and then
//! spent **36.6 seconds** rasterizing 506 thumbnails on the UI thread before
//! it could lay anything out — the window simply hung. Rendering a page to
//! learn a number the file states outright is work for nothing.
//!
//! `/MediaBox` states it outright (PDF 32000-1:2008 §7.7.3.3), and it is
//! inheritable through the page tree, which [`PdfDocument::page`] has already
//! resolved. So this is a dictionary lookup: microseconds for the whole
//! document instead of half a minute.
//!
//! # `/Rotate` is part of the answer
//!
//! A page with `/Rotate 90` or `270` is *displayed* with its width and height
//! swapped (§7.7.3.3). A layout that ignores it gives every landscape-rotated
//! page a portrait slot and misplaces everything below it, so the swap belongs
//! here rather than in each caller.

use super::geometry::Rect;
use super::object::Object;
use super::xref::PdfDocument;

/// US Letter, the fallback when a page states no usable `/MediaBox`.
///
/// Matches what `kpdf`'s layout already assumed for an unmeasurable page, so
/// a malformed document degrades exactly as before rather than collapsing to
/// a zero-height slot.
pub const DEFAULT_SIZE_PTS: (f32, f32) = (612.0, 792.0);

/// Page `page_index`'s displayed size in PDF points, `/Rotate` applied.
///
/// Never fails: a missing, malformed or degenerate `/MediaBox` yields
/// [`DEFAULT_SIZE_PTS`]. A viewer wants a plausible slot far more than it
/// wants an error — the page still renders, and a wrong-sized slot is
/// recoverable while a failed open is not.
/// The page's `/MediaBox` in PDF points, **origin included**.
///
/// # Why the origin matters, and why returning only a size was a bug
///
/// [`page_size_points`] answers "how big is this page", which is all a layout
/// needs. Anything that converts between a point on screen and a point in the
/// PDF needs more than that, because **default user space does not start at
/// (0, 0)**: §7.7.3.3 puts the media box's lower-left corner at
/// `(x0, y0)`, so a mark at the visual bottom-left of a page whose box is
/// `[9 9 621 801]` has user-space coordinates `(9, 9)`, not `(0, 0)`.
///
/// Plenty of PDFs -- cropped scans, journal typesetting, anything that has
/// been imposed or trimmed -- carry a non-zero origin. Treating it as zero
/// places every annotation and every form-field highlight off by exactly
/// `(x0, y0)`: not obviously broken, just persistently *slightly wrong*,
/// which is how it survived until a maintainer drew on a real paper and said
/// the ink landed in the wrong place.
///
/// A page that has no usable `/MediaBox` falls back to
/// [`DEFAULT_SIZE_PTS`] at the origin, matching [`page_size_points`].
///
/// # Rotation
///
/// `/Rotate` 90 or 270 swaps the returned extents, exactly as
/// [`page_size_points`] does, so the box describes the page **as displayed**.
/// The origin is passed through unrotated. That is right for the overwhelmingly
/// common `/Rotate 0` case and for the size in every case, but a *fully*
/// rotation-correct screen<->user-space mapping needs the whole rotation
/// applied to the point as well, which this crate does not do yet -- see the
/// note on `screen_to_page`.
pub fn page_media_box_points(doc: &PdfDocument, page_index: usize) -> Rect {
    let fallback = Rect {
        x0: 0.0,
        y0: 0.0,
        x1: DEFAULT_SIZE_PTS.0,
        y1: DEFAULT_SIZE_PTS.1,
    };
    let Ok(page) = doc.page(page_index) else {
        return fallback;
    };
    let Some(r) = media_box(doc, &page) else {
        return fallback;
    };
    let rotate = page
        .dict_gets("Rotate")
        .and_then(|o| doc.resolve(o).ok())
        .map(|o| o.to_int())
        .unwrap_or(0)
        .rem_euclid(360);
    if rotate == 90 || rotate == 270 {
        Rect {
            x0: r.x0,
            y0: r.y0,
            x1: r.x0 + (r.y1 - r.y0),
            y1: r.y0 + (r.x1 - r.x0),
        }
    } else {
        r
    }
}

pub fn page_size_points(doc: &PdfDocument, page_index: usize) -> (f32, f32) {
    let Ok(page) = doc.page(page_index) else {
        return DEFAULT_SIZE_PTS;
    };
    let (w, h) = media_box(doc, page)
        .map(|r| (r.x1 - r.x0, r.y1 - r.y0))
        .unwrap_or(DEFAULT_SIZE_PTS);

    // §7.7.3.3: /Rotate is clockwise, a multiple of 90. 90 and 270 present the
    // page turned on its side, so the displayed extents swap.
    let rotate = page
        .dict_gets("Rotate")
        .and_then(|o| doc.resolve(o).ok())
        .map(|o| o.to_int())
        .unwrap_or(0);
    // Normalise into 0..360 first: negative and >360 values both occur.
    let rotate = rotate.rem_euclid(360);
    if rotate == 90 || rotate == 270 {
        (h, w)
    } else {
        (w, h)
    }
}

/// The page's `/MediaBox` as a normalised rect, or `None` if unusable.
///
/// Normalised because §7.9.5 allows either diagonal corner order, so a box
/// written `[612 792 0 0]` is legal and means the same as `[0 0 612 792]`.
fn media_box(doc: &PdfDocument, page: &Object) -> Option<Rect> {
    let arr = doc.resolve(page.dict_gets("MediaBox")?).ok()?;
    if arr.array_len() != 4 {
        return None;
    }
    let mut v = [0f32; 4];
    for (i, slot) in v.iter_mut().enumerate() {
        *slot = doc.resolve(arr.array_get(i)?).ok()?.to_real() as f32;
        if !slot.is_finite() {
            return None;
        }
    }
    let r = Rect {
        x0: v[0].min(v[2]),
        y0: v[1].min(v[3]),
        x1: v[0].max(v[2]),
        y1: v[1].max(v[3]),
    };
    // A degenerate box is not a usable page size.
    ((r.x1 - r.x0) > 1.0 && (r.y1 - r.y0) > 1.0).then_some(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(pages: &[&str], extra: &[&str]) -> PdfDocument {
        let mut bodies: Vec<String> = vec![
            "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
            format!(
                "<< /Type /Pages /Kids [{}] /Count {} >>",
                (0..pages.len())
                    .map(|i| format!("{} 0 R", i + 3))
                    .collect::<Vec<_>>()
                    .join(" "),
                pages.len()
            ),
        ];
        bodies.extend(pages.iter().map(|p| p.to_string()));
        bodies.extend(extra.iter().map(|p| p.to_string()));
        let mut pdf: Vec<u8> = b"%PDF-1.5\n".to_vec();
        let mut offsets = vec![0usize; bodies.len() + 1];
        for (idx, body) in bodies.iter().enumerate() {
            let num = idx + 1;
            offsets[num] = pdf.len();
            pdf.extend_from_slice(format!("{num} 0 obj\n{body}\nendobj\n").as_bytes());
        }
        let xref = pdf.len();
        let size = bodies.len() + 1;
        pdf.extend_from_slice(format!("xref\n0 {size}\n").as_bytes());
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        for off in offsets.iter().skip(1) {
            pdf.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
        }
        pdf.extend_from_slice(
            format!("trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n")
                .as_bytes(),
        );
        PdfDocument::open(pdf).unwrap()
    }

    #[test]
    fn reads_the_media_box() {
        let d = build(
            &["<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] >>"],
            &[],
        );
        assert_eq!(page_size_points(&d, 0), (200.0, 100.0));
    }

    /// The size is inheritable from the `/Pages` node — the common real-world
    /// shape, and the one a page-dict-only lookup misses.
    #[test]
    fn inherits_the_media_box_from_the_page_tree() {
        // MediaBox on the /Pages node, not on the page itself.
        let d = {
            let bodies = [
                "<< /Type /Catalog /Pages 2 0 R >>",
                "<< /Type /Pages /Kids [3 0 R] /Count 1 /MediaBox [0 0 300 400] >>",
                "<< /Type /Page /Parent 2 0 R >>",
            ];
            let mut pdf: Vec<u8> = b"%PDF-1.5\n".to_vec();
            let mut offs = vec![0usize; bodies.len() + 1];
            for (i, b) in bodies.iter().enumerate() {
                offs[i + 1] = pdf.len();
                pdf.extend_from_slice(format!("{} 0 obj\n{b}\nendobj\n", i + 1).as_bytes());
            }
            let x = pdf.len();
            pdf.extend_from_slice(format!("xref\n0 {}\n", bodies.len() + 1).as_bytes());
            pdf.extend_from_slice(b"0000000000 65535 f \n");
            for o in offs.iter().skip(1) {
                pdf.extend_from_slice(format!("{o:010} 00000 n \n").as_bytes());
            }
            pdf.extend_from_slice(
                format!(
                    "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{x}\n%%EOF\n",
                    bodies.len() + 1
                )
                .as_bytes(),
            );
            PdfDocument::open(pdf).unwrap()
        };
        assert_eq!(page_size_points(&d, 0), (300.0, 400.0));
    }

    /// `/Rotate 90` and `270` present the page on its side, so the extents
    /// swap. Getting this wrong gives a landscape page a portrait slot and
    /// misplaces every page below it.
    #[test]
    fn rotate_swaps_the_extents() {
        for (rot, want) in [
            (0, (200.0, 100.0)),
            (90, (100.0, 200.0)),
            (180, (200.0, 100.0)),
            (270, (100.0, 200.0)),
            (360, (200.0, 100.0)),
            (-90, (100.0, 200.0)),
            (450, (100.0, 200.0)),
        ] {
            let d = build(
                &[&format!(
                    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Rotate {rot} >>"
                )],
                &[],
            );
            assert_eq!(page_size_points(&d, 0), want, "/Rotate {rot}");
        }
    }

    /// A box written corner-reversed is legal (§7.9.5) and means the same.
    #[test]
    fn a_reversed_media_box_is_normalised() {
        let d = build(
            &["<< /Type /Page /Parent 2 0 R /MediaBox [200 100 0 0] >>"],
            &[],
        );
        assert_eq!(page_size_points(&d, 0), (200.0, 100.0));
    }

    /// Missing, degenerate and out-of-range all fall back rather than
    /// producing a zero-height slot that would collapse the layout.
    #[test]
    fn unusable_boxes_fall_back_to_letter() {
        let d = build(&["<< /Type /Page /Parent 2 0 R >>"], &[]);
        assert_eq!(page_size_points(&d, 0), DEFAULT_SIZE_PTS);

        let d = build(
            &["<< /Type /Page /Parent 2 0 R /MediaBox [0 0 0 0] >>"],
            &[],
        );
        assert_eq!(page_size_points(&d, 0), DEFAULT_SIZE_PTS);

        let d = build(
            &["<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] >>"],
            &[],
        );
        assert_eq!(page_size_points(&d, 99), DEFAULT_SIZE_PTS, "out of range");
    }
    /// The origin is the whole reason `page_media_box_points` exists beside
    /// `page_size_points`. A cropped or imposed page carries a non-zero
    /// lower-left, and dropping it is what put ink slightly off the pen.
    #[test]
    fn the_media_box_origin_survives() {
        let doc = build(
            &["<< /Type /Page /Parent 2 0 R /MediaBox [9 9 621 801] >>"],
            &[],
        );
        let mb = page_media_box_points(&doc, 0);
        assert_eq!((mb.x0, mb.y0), (9.0, 9.0), "the origin must not be dropped");
        assert_eq!(
            (mb.x1 - mb.x0, mb.y1 - mb.y0),
            page_size_points(&doc, 0),
            "and the extents must still agree with page_size_points"
        );
    }

    /// A box given corner-swapped (`[621 801 9 9]`) is still the same page --
    /// `media_box` normalises it, and the origin must come out as the LOWER
    /// left, not whichever number happened to be written first.
    #[test]
    fn a_reversed_box_still_reports_the_lower_left() {
        let doc = build(
            &["<< /Type /Page /Parent 2 0 R /MediaBox [621 801 9 9] >>"],
            &[],
        );
        let mb = page_media_box_points(&doc, 0);
        assert_eq!((mb.x0, mb.y0), (9.0, 9.0));
    }

    /// The ordinary case, which must stay exactly as it was.
    #[test]
    fn a_zero_origin_box_is_unchanged() {
        let doc = build(
            &["<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>"],
            &[],
        );
        let mb = page_media_box_points(&doc, 0);
        assert_eq!((mb.x0, mb.y0, mb.x1, mb.y1), (0.0, 0.0, 612.0, 792.0));
    }

    /// A missing or unusable box falls back to the same default size
    /// `page_size_points` uses, at the origin -- the two must not disagree
    /// about what a broken page looks like.
    #[test]
    fn a_missing_box_falls_back_consistently() {
        let doc = build(&["<< /Type /Page /Parent 2 0 R >>"], &[]);
        let mb = page_media_box_points(&doc, 0);
        assert_eq!((mb.x0, mb.y0), (0.0, 0.0));
        assert_eq!((mb.x1 - mb.x0, mb.y1 - mb.y0), page_size_points(&doc, 0));
    }

    /// Rotation swaps the extents, matching `page_size_points`, so a layout
    /// built from either agrees on how tall the page is.
    #[test]
    fn rotation_swaps_the_extents_the_same_way_as_page_size_points() {
        let doc = build(
            &["<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Rotate 90 >>"],
            &[],
        );
        let mb = page_media_box_points(&doc, 0);
        assert_eq!((mb.x1 - mb.x0, mb.y1 - mb.y0), (792.0, 612.0));
        assert_eq!((mb.x1 - mb.x0, mb.y1 - mb.y0), page_size_points(&doc, 0));
    }

}

//! Link annotations — the clickable regions on a page.
//!
//! Ported from MuPDF `source/pdf/pdf-link.c` (`pdf_load_links`,
//! `pdf_load_link_annots`), commit `0b8fd1c`, AGPL-3.0, © Artifex.
//!
//! A link is a `/Subtype /Link` annotation carrying a `/Rect` and either a
//! `/Dest` or an `/A` action — the same destination shapes the outline uses,
//! so resolution is shared with [`super::destination`].

use super::destination::{Destination, Destinations};
use super::geometry::Rect;
use super::xref::PdfDocument;

/// A clickable region on a page.
#[derive(Debug, Clone, PartialEq)]
pub struct Link {
    /// The clickable area, in PDF user space, normalised so `x0 <= x1` and
    /// `y0 <= y1` (§7.9.5 permits either corner order).
    pub rect: Rect,
    /// Where it leads.
    pub dest: Destination,
}

/// Every resolvable link on `page_index`.
///
/// Links whose destination cannot be resolved are **dropped** rather than
/// returned with an empty target: a clickable region that does nothing is
/// worse than no region at all, since the cursor changes and the click is
/// swallowed.
pub fn page_links(doc: &PdfDocument, page_index: usize) -> Vec<Link> {
    let dests = Destinations::new(doc);
    page_links_with(doc, page_index, &dests)
}

/// [`page_links`] reusing an existing [`Destinations`] — use this when
/// walking many pages, or the page tree is re-indexed for every one.
pub fn page_links_with(
    doc: &PdfDocument,
    page_index: usize,
    dests: &Destinations,
) -> Vec<Link> {
    let Ok(page) = doc.page(page_index) else {
        return Vec::new();
    };
    let page = page.clone();
    let Ok(annots) = doc.resolve_get(&page, "Annots") else {
        return Vec::new();
    };

    let mut links = Vec::new();
    for i in 0..annots.array_len() {
        let Some(annot_ref) = annots.array_get(i) else {
            continue;
        };
        let Ok(annot) = doc.resolve(annot_ref) else {
            continue;
        };
        if !annot.is_dict() || annot.dict_gets("Subtype").map(|o| o.to_name()) != Some(b"Link") {
            continue;
        }
        let Some(rect) = rect_of(doc, &annot) else {
            continue;
        };
        let Some(dest) = dests.resolve(doc, &annot) else {
            continue;
        };
        links.push(Link { rect, dest });
    }
    links
}

/// The annotation's `/Rect`, normalised. `None` when absent, malformed, or
/// zero-area — a link nobody can click is not worth returning.
fn rect_of(doc: &PdfDocument, annot: &super::object::Object) -> Option<Rect> {
    let arr = doc.resolve(annot.dict_gets("Rect")?).ok()?;
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
    ((r.x1 - r.x0) > 0.0 && (r.y1 - r.y0) > 0.0).then_some(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::destination::build_pdf;

    fn doc_with_annots(annots: &str, extra: &[&str]) -> PdfDocument {
        let mut bodies = vec![
            "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
            "<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 /MediaBox [0 0 600 800] >>".to_string(),
            format!("<< /Type /Page /Parent 2 0 R /Annots {annots} >>"),
            "<< /Type /Page /Parent 2 0 R >>".to_string(),
        ];
        bodies.extend(extra.iter().map(|s| s.to_string()));
        build_pdf(&bodies)
    }

    #[test]
    fn a_page_with_no_annots_has_no_links() {
        let doc = build_pdf(&[
            "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
            "<< /Type /Pages /Kids [3 0 R] /Count 1 /MediaBox [0 0 600 800] >>".to_string(),
            "<< /Type /Page /Parent 2 0 R >>".to_string(),
        ]);
        assert!(page_links(&doc, 0).is_empty());
    }

    #[test]
    fn an_internal_link_resolves_to_a_page() {
        let doc = doc_with_annots(
            "[5 0 R]",
            &["<< /Subtype /Link /Rect [10 20 110 40] /Dest [4 0 R /Fit] >>"],
        );
        let links = page_links(&doc, 0);
        assert_eq!(links.len(), 1);
        assert_eq!(
            links[0].dest,
            Destination::Page { page: 1, left: None, top: None, zoom: None }
        );
        assert_eq!(links[0].rect.x0, 10.0);
    }

    #[test]
    fn a_uri_link_resolves() {
        let doc = doc_with_annots(
            "[5 0 R]",
            &["<< /Subtype /Link /Rect [0 0 10 10] /A << /S /URI /URI (https://x.test) >> >>"],
        );
        assert_eq!(
            page_links(&doc, 0)[0].dest,
            Destination::Uri("https://x.test".to_string())
        );
    }

    /// §7.9.5 lets a rect be written from either diagonal corner.
    #[test]
    fn a_reversed_rect_is_normalised() {
        let doc = doc_with_annots(
            "[5 0 R]",
            &["<< /Subtype /Link /Rect [110 40 10 20] /Dest [4 0 R /Fit] >>"],
        );
        let r = page_links(&doc, 0)[0].rect;
        assert_eq!((r.x0, r.y0, r.x1, r.y1), (10.0, 20.0, 110.0, 40.0));
    }

    /// A clickable region that goes nowhere is worse than no region: the
    /// cursor changes and the click is swallowed. Such links are dropped.
    #[test]
    fn links_with_no_resolvable_destination_are_dropped() {
        let doc = doc_with_annots(
            "[5 0 R 6 0 R]",
            &[
                "<< /Subtype /Link /Rect [0 0 10 10] /Dest (missing) >>",
                "<< /Subtype /Link /Rect [0 0 10 10] >>",
            ],
        );
        assert!(page_links(&doc, 0).is_empty());
    }

    #[test]
    fn a_zero_area_rect_is_dropped() {
        let doc = doc_with_annots(
            "[5 0 R]",
            &["<< /Subtype /Link /Rect [5 5 5 5] /Dest [4 0 R /Fit] >>"],
        );
        assert!(page_links(&doc, 0).is_empty(), "nobody can click a zero-area link");
    }

    /// Non-link annotations on the same page must be ignored.
    #[test]
    fn only_link_subtypes_are_returned() {
        let doc = doc_with_annots(
            "[5 0 R 6 0 R]",
            &[
                "<< /Subtype /Widget /Rect [0 0 10 10] /Dest [4 0 R /Fit] >>",
                "<< /Subtype /Link /Rect [0 0 10 10] /Dest [4 0 R /Fit] >>",
            ],
        );
        assert_eq!(page_links(&doc, 0).len(), 1);
    }
}

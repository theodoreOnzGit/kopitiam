//! The document outline — bookmarks, the table of contents a reader shows in
//! its sidebar.
//!
//! Ported from MuPDF `source/pdf/pdf-outline.c` (`pdf_load_outline` and its
//! `/First`/`/Next`/`/Title`/`/Count` walk) and `fz_outline`
//! (`include/mupdf/fitz/outline.h`), commit `0b8fd1c`, AGPL-3.0, © Artifex.
//!
//! # Shape
//!
//! The PDF stores the outline as a doubly-linked sibling chain with `/First`
//! and `/Next` per level (§12.3.3). This returns an owned tree of
//! [`OutlineItem`] instead, because every consumer — a sidebar, a jump-to
//! command, a Markdown export — wants children nested, and none of them wants
//! to chase `/Next` pointers.
//!
//! # Cycles are a real hazard, not a theoretical one
//!
//! `/First`/`/Next`/`/Parent` are producer-written pointers with nothing
//! stopping them forming a loop, and MuPDF has a whole repair pass
//! (`pdf_test_outline`) for exactly this. A loop here would hang a reader on
//! open, so the walk carries a visited set of object numbers and a depth cap
//! and stops rather than repairing. Repairing means *writing* to the
//! document, which loading a table of contents has no business doing.

use std::collections::HashSet;

use super::destination::{Destination, Destinations};
use super::object::Object;
use super::xref::PdfDocument;

/// Depth cap for the outline tree. Real documents rarely pass 6 levels; this
/// is generous while still bounding a malformed file.
const MAX_DEPTH: usize = 64;

/// One bookmark.
#[derive(Debug, Clone, PartialEq)]
pub struct OutlineItem {
    /// The `/Title`, decoded as a PDF text string.
    pub title: String,
    /// Where it leads, when that could be resolved.
    pub dest: Option<Destination>,
    /// Whether the item is drawn expanded. `/Count > 0` means open
    /// (§12.3.3); absent or `<= 0` means closed.
    pub open: bool,
    /// `/C`, the RGB the title is drawn in, when the file states one.
    pub color: Option<[f32; 3]>,
    /// Nested bookmarks, in document order.
    pub children: Vec<OutlineItem>,
}

impl OutlineItem {
    /// Total items in this subtree, including itself.
    ///
    /// Deliberately not called `len`: it counts a *tree*, never returns zero,
    /// and has no meaningful `is_empty` counterpart — `children.is_empty()` is
    /// the question a caller actually wants.
    pub fn count(&self) -> usize {
        1 + self.children.iter().map(OutlineItem::count).sum::<usize>()
    }
}

/// Load the document outline, or an empty vector when there is none.
///
/// Never fails: a document with no `/Outlines`, or one whose outline is
/// malformed, yields what could be read. A broken table of contents must not
/// stop a readable document from opening.
pub fn load_outline(doc: &PdfDocument) -> Vec<OutlineItem> {
    let dests = Destinations::new(doc);
    load_outline_with(doc, &dests)
}

/// [`load_outline`] reusing an existing [`Destinations`] — worth it when the
/// caller is also resolving links and would otherwise index the pages twice.
pub fn load_outline_with(doc: &PdfDocument, dests: &Destinations) -> Vec<OutlineItem> {
    let Ok(root) = doc.catalog() else {
        return Vec::new();
    };
    let Ok(outlines) = doc.resolve_get(&root, "Outlines") else {
        return Vec::new();
    };
    if !outlines.is_dict() {
        return Vec::new();
    }
    let Some(first) = outlines.dict_gets("First") else {
        return Vec::new();
    };
    let mut seen = HashSet::new();
    walk_siblings(doc, dests, first, &mut seen, 0)
}

/// Follow one `/Next` chain, recursing into `/First` for children.
fn walk_siblings(
    doc: &PdfDocument,
    dests: &Destinations,
    first: &Object,
    seen: &mut HashSet<i32>,
    depth: usize,
) -> Vec<OutlineItem> {
    let mut out = Vec::new();
    if depth > MAX_DEPTH {
        return out;
    }
    let mut cursor = first.clone();
    // Every node must be an indirect object, both because §12.3.3 says so and
    // because the object number is what the cycle check keys on. A direct dict
    // here is unfollowable, so the walk stops rather than risk looping.
    while let Object::Ref { num, .. } = cursor {
        if !seen.insert(num) {
            break; // cycle: stop, do not repair
        }
        let Ok(node) = doc.resolve(&cursor) else { break };
        if !node.is_dict() {
            break;
        }

        let title = node
            .dict_gets("Title")
            .and_then(|t| doc.resolve(t).ok())
            .filter(|t| t.is_string())
            .map(|t| super::doc_info::decode_text_string(t.to_string_bytes()))
            .unwrap_or_default();

        // /Count > 0 means the item is displayed open (§12.3.3, Table 153).
        let open = node
            .dict_gets("Count")
            .and_then(|c| doc.resolve(c).ok())
            .map(|c| c.to_int() > 0)
            .unwrap_or(false);

        let children = match node.dict_gets("First") {
            Some(f) => walk_siblings(doc, dests, f, seen, depth + 1),
            None => Vec::new(),
        };

        out.push(OutlineItem {
            title,
            dest: dests.resolve(doc, &node),
            open,
            color: rgb(doc, &node),
            children,
        });

        match node.dict_gets("Next") {
            Some(next) => cursor = next.clone(),
            None => break,
        }
    }
    out
}

/// `/C`, a three-element RGB array (§12.3.3).
fn rgb(doc: &PdfDocument, node: &Object) -> Option<[f32; 3]> {
    let c = doc.resolve(node.dict_gets("C")?).ok()?;
    if c.array_len() != 3 {
        return None;
    }
    let mut v = [0f32; 3];
    for (i, slot) in v.iter_mut().enumerate() {
        *slot = doc.resolve(c.array_get(i)?).ok()?.to_real() as f32;
    }
    Some(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::destination::build_pdf;

    /// Catalog(1) + pages(2,3,4,5) + whatever the test adds from object 6 on.
    fn doc_with(outline_root: &str, extra: &[&str]) -> PdfDocument {
        let mut bodies = vec![
            format!("<< /Type /Catalog /Pages 2 0 R /Outlines {outline_root} >>"),
            "<< /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 3 /MediaBox [0 0 600 800] >>"
                .to_string(),
            "<< /Type /Page /Parent 2 0 R >>".to_string(),
            "<< /Type /Page /Parent 2 0 R >>".to_string(),
            "<< /Type /Page /Parent 2 0 R >>".to_string(),
        ];
        bodies.extend(extra.iter().map(|s| s.to_string()));
        build_pdf(&bodies)
    }

    #[test]
    fn a_document_with_no_outline_yields_nothing() {
        let doc = build_pdf(&[
            "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
            "<< /Type /Pages /Kids [3 0 R] /Count 1 /MediaBox [0 0 600 800] >>".to_string(),
            "<< /Type /Page /Parent 2 0 R >>".to_string(),
        ]);
        assert!(load_outline(&doc).is_empty());
    }

    #[test]
    fn siblings_and_children_nest_correctly() {
        let doc = doc_with(
            "6 0 R",
            &[
                // 6: /Outlines root
                "<< /Type /Outlines /First 7 0 R /Count 2 >>",
                // 7: "One", has a child, open
                "<< /Title (One) /First 9 0 R /Next 8 0 R /Count 1 /Dest [3 0 R /Fit] >>",
                // 8: "Two", closed leaf
                "<< /Title (Two) /Dest [5 0 R /Fit] >>",
                // 9: "One.a", child of 7
                "<< /Title (One.a) /Dest [4 0 R /Fit] >>",
            ],
        );
        let items = load_outline(&doc);
        assert_eq!(items.len(), 2, "two top-level items");
        assert_eq!(items[0].title, "One");
        assert_eq!(items[1].title, "Two");
        assert_eq!(items[0].children.len(), 1);
        assert_eq!(items[0].children[0].title, "One.a");
        assert!(items[0].open, "/Count > 0 means displayed open");
        assert!(!items[1].open, "no /Count means closed");
        // The whole tree, counted through the nesting.
        assert_eq!(items.iter().map(OutlineItem::count).sum::<usize>(), 3);
    }

    #[test]
    fn destinations_are_resolved_per_item() {
        let doc = doc_with(
            "6 0 R",
            &[
                "<< /Type /Outlines /First 7 0 R >>",
                "<< /Title (Third page) /Dest [5 0 R /XYZ 10 20 null] >>",
            ],
        );
        let items = load_outline(&doc);
        assert_eq!(
            items[0].dest,
            Some(Destination::Page { page: 2, left: Some(10.0), top: Some(20.0), zoom: None })
        );
    }

    /// A `/Next` loop must terminate. MuPDF has a whole repair pass for this;
    /// here the walk simply stops, because loading a table of contents has no
    /// business writing to the document.
    #[test]
    fn a_next_cycle_terminates() {
        let doc = doc_with(
            "6 0 R",
            &[
                "<< /Type /Outlines /First 7 0 R >>",
                // 7 -> 8 -> 7 -> ...
                "<< /Title (A) /Next 8 0 R >>",
                "<< /Title (B) /Next 7 0 R >>",
            ],
        );
        let items = load_outline(&doc);
        assert_eq!(items.len(), 2, "each node is visited exactly once");
        assert_eq!(items[0].title, "A");
        assert_eq!(items[1].title, "B");
    }

    /// A `/First` loop (a node claiming itself as its own child) must also
    /// terminate rather than recurse to the depth cap.
    #[test]
    fn a_first_cycle_terminates() {
        let doc = doc_with(
            "6 0 R",
            &[
                "<< /Type /Outlines /First 7 0 R >>",
                "<< /Title (A) /First 7 0 R >>",
            ],
        );
        let items = load_outline(&doc);
        assert_eq!(items.len(), 1);
        assert!(items[0].children.is_empty(), "the self-child is refused");
    }

    #[test]
    fn a_title_is_decoded_as_a_pdf_text_string() {
        let doc = doc_with(
            "6 0 R",
            &[
                "<< /Type /Outlines /First 7 0 R >>",
                // UTF-16BE with BOM: "Hi"
                "<< /Title <FEFF00480069> >>",
            ],
        );
        assert_eq!(load_outline(&doc)[0].title, "Hi");
    }

    #[test]
    fn an_item_colour_is_read_when_present() {
        let doc = doc_with(
            "6 0 R",
            &[
                "<< /Type /Outlines /First 7 0 R >>",
                "<< /Title (Red) /C [1 0 0] >>",
            ],
        );
        assert_eq!(load_outline(&doc)[0].color, Some([1.0, 0.0, 0.0]));
    }
}

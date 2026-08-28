//! Adding and removing whole pages — page-tree surgery (PDF 32000-1:2008
//! §7.7.3), as opposed to [`super::annot_edit`]'s edits *within* a page.
//!
//! The motivating use is lecturing: open a deck in `kpdf`, run out of room
//! mid-derivation, add a blank page and keep writing on it with a tablet. So
//! the blank page this produces must be a first-class page — same size as the
//! one you were just on, annotatable, printable, and saved by the same
//! `Ctrl+S` as everything else.
//!
//! # How a page is added without rewriting the file
//!
//! Everything goes through [`super::write::incremental_update`], the same
//! append-only mechanism the annotation editor uses. Adding a page appends
//! three things: the new page object, a superseded `/Pages` node whose
//! `/Kids` now includes it, and superseded ancestors whose `/Count` went up
//! by one. The original bytes are never touched, so undo stays a truncation
//! and a signed document's original bytes remain intact and verifiable.
//!
//! # Why deletion leaves the page object behind
//!
//! Removing a page unlinks it from `/Kids` and decrements `/Count`; the page
//! object itself stays in the file as an orphan. That is not laziness — an
//! incremental update **cannot** remove bytes without rewriting the file and
//! invalidating every existing cross-reference, and orphaned objects are
//! normal in PDFs edited this way (they are what a "garbage collect" pass in
//! a full rewriter later reclaims). The visible behaviour is correct: the
//! page is gone from the page tree, so no reader will show it.
//!
//! **Consequence worth stating plainly, since it is a privacy question and
//! not merely a size one:** a deleted page's content is still present in the
//! saved file and recoverable with a PDF forensics tool. Deleting a page here
//! is *not* redaction. Anyone who needs the content genuinely gone needs a
//! full rewrite, which this crate does not yet do.
//!
//! # Why an empty `/Pages` node is left in place
//!
//! Deleting the only kid of an intermediate `/Pages` node leaves that node
//! with an empty `/Kids` and `/Count 0`. It is legal and every reader
//! (including this one) walks straight past it, so pruning it would add a
//! recursive parent-fixup for no user-visible gain and more ways to corrupt
//! a tree.

use std::collections::HashSet;

use super::error::{Error, Result};
use super::geometry::Rect;
use super::object::Object;
use super::write::{self, NewObject};
use super::xref::PdfDocument;

/// ISO A4 in points (595.276 x 841.89), the fallback when a blank page has no
/// neighbour to copy a size from.
pub const A4_PORTRAIT: Rect = Rect {
    x0: 0.0,
    y0: 0.0,
    x1: 595.276,
    y1: 841.89,
};

/// Where a page sits in the page tree — everything needed to unlink it or to
/// splice a sibling in next to it.
#[derive(Debug, Clone)]
pub struct PageSlot {
    pub page_num: i32,
    pub page_gen: i32,
    /// The `/Pages` node holding this page in its `/Kids`.
    pub parent_num: i32,
    pub parent_gen: i32,
    /// This page's index within that parent's `/Kids` array.
    pub kid_index: usize,
    /// Every `/Pages` node from the parent up to the root, parent first. All
    /// of them carry a `/Count` that has to move when a page is added or
    /// removed — missing one leaves a document whose page count disagrees
    /// with its own tree, which readers resolve inconsistently.
    pub ancestors: Vec<(i32, i32)>,
}

/// Locate `page_index` in the page tree.
///
/// This is deliberately a separate walk from [`super::annot_edit`]'s
/// `locate_page`, which only needs the page's own object number; splicing
/// needs the *parent* and the index within it, which that walk discards.
pub fn locate_page_slot(doc: &PdfDocument, page_index: usize) -> Result<PageSlot> {
    let root = doc.catalog()?;
    let pages_ref = root
        .dict_gets("Pages")
        .cloned()
        .ok_or_else(|| Error::format("catalog has no /Pages"))?;
    let mut counter = 0usize;
    let mut chain = Vec::new();
    let mut visited = HashSet::new();
    find_slot(
        doc,
        &pages_ref,
        page_index,
        &mut counter,
        &mut chain,
        &mut visited,
    )?
    .ok_or_else(|| Error::argument(format!("page {page_index} out of range 0..{counter}")))
}

/// Is this node an interior `/Pages` node rather than a leaf `/Page`?
///
/// Same rule as [`super::annot_edit`]'s walk: trust `/Type` when present, and
/// fall back to "has `/Kids` but no `/MediaBox`" for the producers that omit
/// it.
fn is_internal(node: &Object) -> bool {
    match node.dict_gets("Type").map(|o| o.to_name()) {
        Some(b"Pages") => true,
        Some(b"Page") => false,
        _ => node.dict_gets("Kids").is_some() && node.dict_gets("MediaBox").is_none(),
    }
}

fn ref_parts(obj: &Object) -> Option<(i32, i32)> {
    match obj {
        Object::Ref { num, generation } => Some((*num, *generation)),
        _ => None,
    }
}

/// Depth-first search for the `target`-th leaf page.
///
/// Leaves are inspected from the *parent*, not on recursion into them, so the
/// kid index is known at the moment the target is found.
fn find_slot(
    doc: &PdfDocument,
    node_ref: &Object,
    target: usize,
    counter: &mut usize,
    chain: &mut Vec<(i32, i32)>,
    visited: &mut HashSet<i32>,
) -> Result<Option<PageSlot>> {
    let node = doc.resolve(node_ref)?;
    if !node.is_dict() || !is_internal(&node) {
        return Ok(None);
    }
    let Some((node_num, node_gen)) = ref_parts(node_ref) else {
        return Err(Error::format(
            "a /Pages node is not an indirect object, so it cannot be superseded",
        ));
    };
    if !visited.insert(node_num) {
        return Err(Error::format("cycle in page tree"));
    }
    chain.push((node_num, node_gen));

    let kids = doc.resolve_get(&node, "Kids")?;
    for i in 0..kids.array_len() {
        let Some(kid_ref) = kids.array_get(i) else {
            continue;
        };
        let kid = doc.resolve(kid_ref)?;
        if !kid.is_dict() {
            continue;
        }
        if is_internal(&kid) {
            if let Some(found) = find_slot(doc, kid_ref, target, counter, chain, visited)? {
                return Ok(Some(found));
            }
            continue;
        }
        // A leaf page.
        if *counter == target {
            let Some((page_num, page_gen)) = ref_parts(kid_ref) else {
                return Err(Error::format(
                    "page has no indirect object identity (not referenced via an indirect reference)",
                ));
            };
            let mut ancestors = chain.clone();
            ancestors.reverse(); // parent first, root last
            return Ok(Some(PageSlot {
                page_num,
                page_gen,
                parent_num: node_num,
                parent_gen: node_gen,
                kid_index: i,
                ancestors,
            }));
        }
        *counter += 1;
    }

    chain.pop();
    Ok(None)
}

/// `/MediaBox` as a PDF array.
fn rect_array(r: Rect) -> Object {
    let mut a = Object::new_array();
    for v in [r.x0, r.y0, r.x1, r.y1] {
        a.array_push(Object::new_real(v as f64));
    }
    a
}

/// Read a `/Pages` node's `/Count`, resolving an indirect value.
fn node_count(doc: &PdfDocument, node: &Object) -> i64 {
    node.dict_gets("Count")
        .and_then(|c| doc.resolve(c).ok())
        .map(|c| c.to_int())
        .unwrap_or(0)
}

/// Rebuild an ancestor `/Pages` node with `/Count` shifted by `delta`.
fn count_update(doc: &PdfDocument, num: i32, generation: i32, delta: i64) -> Result<(i32, NewObject)> {
    let node = doc.resolve(&Object::new_indirect(num as i64, generation))?;
    if !node.is_dict() {
        return Err(Error::format("page-tree ancestor is not a dictionary"));
    }
    let mut updated = node.clone();
    // Clamped at zero: a negative /Count is meaningless and would be a worse
    // corruption than the miscount it came from.
    updated.dict_put("Count", Object::new_int((node_count(doc, &node) + delta).max(0)));
    Ok((num, NewObject::Plain(updated)))
}

/// Insert a blank page so that it becomes page `at` (0-based).
///
/// `at` is clamped to `0..=page_count`, so `page_count` appends at the end.
/// `size` defaults to the neighbouring page's `/MediaBox`, which is what makes
/// "add a page mid-lecture" feel continuous — a new page the same size as the
/// one you were writing on, at the same zoom. [`A4_PORTRAIT`] is used only if
/// the neighbour has no usable box.
///
/// `/Rotate` is copied from the neighbour: in a deck built landscape by
/// rotation, a new page without it would appear portrait and break the flow.
///
/// The page carries no `/Contents`, which is a legal empty page (see
/// [`super::page_run`]'s content gathering, where an absent `/Contents` is an
/// empty byte string) and renders blank.
pub fn insert_blank_page(doc: &PdfDocument, at: usize, size: Option<Rect>) -> Result<Vec<u8>> {
    let page_count = doc.page_count();
    if page_count == 0 {
        return Err(Error::argument(
            "cannot add a page to a document with no pages (nothing to attach to)",
        ));
    }
    let at = at.min(page_count);

    // Insert *before* the page currently at `at`; appending has no such page,
    // so anchor on the last one and go after it.
    let appending = at == page_count;
    let anchor_index = if appending { page_count - 1 } else { at };
    let slot = locate_page_slot(doc, anchor_index)?;
    let kid_index = if appending {
        slot.kid_index + 1
    } else {
        slot.kid_index
    };

    // The flattened page dict, so an *inherited* /MediaBox or /Rotate is seen.
    let anchor = doc.page(anchor_index)?.clone();
    let media = size
        .or_else(|| rect_from_array(doc, anchor.dict_gets("MediaBox")))
        .unwrap_or(A4_PORTRAIT);

    let new_num = write::next_object_number(doc);

    let mut page = Object::new_dict();
    page.dict_put("Type", Object::new_name("Page"));
    page.dict_put(
        "Parent",
        Object::new_indirect(slot.parent_num as i64, slot.parent_gen),
    );
    page.dict_put("MediaBox", rect_array(media));
    // An empty /Resources, not none: §7.8.3 makes it required-inheritable, and
    // annotation appearance streams added later expect a dict to merge into.
    page.dict_put("Resources", Object::new_dict());
    if let Some(rotate) = anchor.dict_gets("Rotate")
        && let Ok(r) = doc.resolve(rotate)
    {
        page.dict_put("Rotate", r);
    }

    let mut updates = vec![(new_num, NewObject::Plain(page))];
    updates.push(kids_update(
        doc,
        &slot,
        KidsEdit::Insert {
            at: kid_index,
            what: Object::new_indirect(new_num as i64, 0),
        },
    )?);
    for &(num, generation) in &slot.ancestors[1..] {
        updates.push(count_update(doc, num, generation, 1)?);
    }
    write::incremental_update(doc, &updates)
}

/// Remove page `page_index` from the page tree.
///
/// Refuses to remove the last remaining page: a zero-page PDF is invalid and
/// most readers, this one included, will not open it — leaving the operator
/// with a file they cannot get back into.
///
/// See the module docs on what this does *not* do: the page's content stays
/// in the file as an orphan, so this is deletion, not redaction.
pub fn delete_page(doc: &PdfDocument, page_index: usize) -> Result<Vec<u8>> {
    let page_count = doc.page_count();
    if page_count <= 1 {
        return Err(Error::argument(
            "cannot delete the only page — a PDF with no pages will not open",
        ));
    }
    let slot = locate_page_slot(doc, page_index)?;
    let mut updates = vec![kids_update(
        doc,
        &slot,
        KidsEdit::Remove {
            at: slot.kid_index,
        },
    )?];
    for &(num, generation) in &slot.ancestors[1..] {
        updates.push(count_update(doc, num, generation, -1)?);
    }
    write::incremental_update(doc, &updates)
}

enum KidsEdit {
    Insert { at: usize, what: Object },
    Remove { at: usize },
}

/// Rebuild the parent `/Pages` node with its `/Kids` spliced and `/Count`
/// moved to match.
///
/// `/Kids` is written back as a **direct** array even when the original was an
/// indirect reference to one. That is legal (§7.7.3.2 does not require
/// indirection) and keeps the update to a single superseded object.
fn kids_update(doc: &PdfDocument, slot: &PageSlot, edit: KidsEdit) -> Result<(i32, NewObject)> {
    let parent = doc.resolve(&Object::new_indirect(
        slot.parent_num as i64,
        slot.parent_gen,
    ))?;
    if !parent.is_dict() {
        return Err(Error::format("page's parent is not a dictionary"));
    }
    let kids = doc.resolve_get(&parent, "Kids")?;

    let mut rebuilt = Object::new_array();
    let (delta, at) = match &edit {
        KidsEdit::Insert { at, .. } => (1, *at),
        KidsEdit::Remove { at } => (-1, *at),
    };
    if at > kids.array_len() {
        return Err(Error::format("page-tree kid index out of range"));
    }
    for i in 0..kids.array_len() {
        if let KidsEdit::Insert { at, what } = &edit
            && i == *at
        {
            rebuilt.array_push(what.clone());
        }
        if matches!(&edit, KidsEdit::Remove { at } if i == *at) {
            continue;
        }
        if let Some(k) = kids.array_get(i) {
            rebuilt.array_push(k.clone());
        }
    }
    // Appending past the last kid.
    if let KidsEdit::Insert { at, what } = &edit
        && *at >= kids.array_len()
    {
        rebuilt.array_push(what.clone());
    }

    let mut updated = parent.clone();
    updated.dict_put("Kids", rebuilt);
    updated.dict_put(
        "Count",
        Object::new_int((node_count(doc, &parent) + delta).max(0)),
    );
    Ok((slot.parent_num, NewObject::Plain(updated)))
}

/// A `[x0 y0 x1 y1]` array as a [`Rect`], normalised so `x0 <= x1` and
/// `y0 <= y1` (PDF allows either corner order; §7.9.5).
fn rect_from_array(doc: &PdfDocument, obj: Option<&Object>) -> Option<Rect> {
    let arr = doc.resolve(obj?).ok()?;
    if arr.array_len() != 4 {
        return None;
    }
    let mut v = [0f32; 4];
    for (i, slot) in v.iter_mut().enumerate() {
        *slot = doc.resolve(arr.array_get(i)?).ok()?.to_real() as f32;
    }
    let r = Rect {
        x0: v[0].min(v[2]),
        y0: v[1].min(v[3]),
        x1: v[0].max(v[2]),
        y1: v[1].max(v[3]),
    };
    // A degenerate box is not a usable page size; fall back rather than
    // creating a page with zero area that renders as nothing.
    ((r.x1 - r.x0) > 1.0 && (r.y1 - r.y0) > 1.0).then_some(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Assemble a PDF from object bodies (object `N` is `bodies[N-1]`), with a
    /// correct classic xref. Same helper shape as `form.rs`/`annot_run.rs`.
    fn build_pdf(bodies: &[&str]) -> PdfDocument {
        let mut pdf: Vec<u8> = Vec::new();
        pdf.extend_from_slice(b"%PDF-1.5\n");
        let mut offsets = vec![0usize; bodies.len() + 1];
        for (idx, body) in bodies.iter().enumerate() {
            let num = idx + 1;
            offsets[num] = pdf.len();
            pdf.extend_from_slice(format!("{num} 0 obj\n{body}\nendobj\n").as_bytes());
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

    /// Flat tree, three pages, `/MediaBox` **inherited** from the `/Pages`
    /// node — the common real-world shape, and the one that catches a
    /// size-copy that only looks at the page's own dict.
    fn flat_three() -> PdfDocument {
        build_pdf(&[
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 3 /MediaBox [0 0 200 100] >>",
            "<< /Type /Page /Parent 2 0 R >>",
            "<< /Type /Page /Parent 2 0 R >>",
            "<< /Type /Page /Parent 2 0 R >>",
        ])
    }

    /// Nested tree: root -> [branch(2 pages), page]. Exercises the ancestor
    /// `/Count` fixup, which a flat tree cannot.
    fn nested_three() -> PdfDocument {
        build_pdf(&[
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R 6 0 R] /Count 3 /MediaBox [0 0 200 100] >>",
            "<< /Type /Pages /Parent 2 0 R /Kids [4 0 R 5 0 R] /Count 2 >>",
            "<< /Type /Page /Parent 3 0 R >>",
            "<< /Type /Page /Parent 3 0 R >>",
            "<< /Type /Page /Parent 2 0 R >>",
        ])
    }

    #[test]
    fn appending_and_inserting_move_the_page_count() {
        let doc = flat_three();
        assert_eq!(doc.page_count(), 3);

        let appended = PdfDocument::open(insert_blank_page(&doc, 3, None).unwrap()).unwrap();
        assert_eq!(appended.page_count(), 4);

        let inserted = PdfDocument::open(insert_blank_page(&doc, 1, None).unwrap()).unwrap();
        assert_eq!(inserted.page_count(), 4);

        // Past the end clamps to an append rather than erroring.
        let clamped = PdfDocument::open(insert_blank_page(&doc, 99, None).unwrap()).unwrap();
        assert_eq!(clamped.page_count(), 4);
    }

    /// The new page must land exactly where asked — checked by giving it a
    /// distinctive size, since blank pages are otherwise indistinguishable.
    #[test]
    fn an_inserted_page_lands_at_the_requested_index() {
        let doc = flat_three();
        let marker = Rect { x0: 0.0, y0: 0.0, x1: 333.0, y1: 444.0 };
        let out = PdfDocument::open(insert_blank_page(&doc, 1, Some(marker)).unwrap()).unwrap();

        let width = |d: &PdfDocument, i: usize| -> f32 {
            rect_from_array(d, d.page(i).unwrap().dict_gets("MediaBox")).unwrap().x1
        };
        assert_eq!(width(&out, 0), 200.0, "page 0 untouched");
        assert_eq!(width(&out, 1), 333.0, "the new page is now page 1");
        assert_eq!(width(&out, 2), 200.0, "the old page 1 shifted to 2");
    }

    /// A blank page copies its neighbour's size — including when that size is
    /// *inherited* from the `/Pages` node rather than written on the page.
    /// Without this a mid-lecture page would jump to A4 in a non-A4 deck.
    #[test]
    fn a_blank_page_inherits_its_neighbours_size() {
        let doc = flat_three();
        let out = PdfDocument::open(insert_blank_page(&doc, 3, None).unwrap()).unwrap();
        let media = rect_from_array(&out, out.page(3).unwrap().dict_gets("MediaBox"))
            .expect("the new page has a MediaBox");
        assert_eq!((media.x1, media.y1), (200.0, 100.0));
    }

    #[test]
    fn a_blank_page_has_no_contents_and_so_renders_empty() {
        let doc = flat_three();
        let out = PdfDocument::open(insert_blank_page(&doc, 3, None).unwrap()).unwrap();
        assert!(out.page(3).unwrap().dict_gets("Contents").is_none());
        assert!(
            out.page(3).unwrap().dict_gets("Resources").is_some(),
            "an empty /Resources must still be present for later annotations"
        );
    }

    #[test]
    fn deleting_removes_exactly_one_page() {
        let doc = flat_three();
        let out = PdfDocument::open(delete_page(&doc, 1).unwrap()).unwrap();
        assert_eq!(out.page_count(), 2);
    }

    /// A zero-page PDF will not open, so the operator would lose the file.
    #[test]
    fn the_last_page_cannot_be_deleted() {
        let doc = build_pdf(&[
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 /MediaBox [0 0 200 100] >>",
            "<< /Type /Page /Parent 2 0 R >>",
        ]);
        assert!(delete_page(&doc, 0).is_err());
    }

    #[test]
    fn out_of_range_deletion_errors() {
        assert!(delete_page(&flat_three(), 9).is_err());
    }

    /// The ancestor `/Count` fixup: in a nested tree both the branch and the
    /// root must move, or the document's page count disagrees with its own
    /// tree and readers resolve it inconsistently.
    #[test]
    fn nested_trees_keep_every_ancestor_count_in_step() {
        let doc = nested_three();
        assert_eq!(doc.page_count(), 3);

        // Page 0 lives in the branch, so branch and root both change.
        let added = PdfDocument::open(insert_blank_page(&doc, 0, None).unwrap()).unwrap();
        assert_eq!(added.page_count(), 4);

        let removed = PdfDocument::open(delete_page(&doc, 0).unwrap()).unwrap();
        assert_eq!(removed.page_count(), 2);

        // Page 2 hangs off the root directly — the other branch of the fixup.
        let tail = PdfDocument::open(delete_page(&doc, 2).unwrap()).unwrap();
        assert_eq!(tail.page_count(), 2);
    }

    /// Round trip: add then remove returns the original page count, and the
    /// pages that remain still have their original size.
    #[test]
    fn add_then_delete_restores_the_page_count() {
        let doc = flat_three();
        let added = PdfDocument::open(insert_blank_page(&doc, 3, None).unwrap()).unwrap();
        let back = PdfDocument::open(delete_page(&added, 3).unwrap()).unwrap();
        assert_eq!(back.page_count(), 3);
        assert_eq!(
            rect_from_array(&back, back.page(0).unwrap().dict_gets("MediaBox"))
                .unwrap()
                .x1,
            200.0
        );
    }

    /// The append-only property the undo history depends on: truncating the
    /// result back to the original length must give the original file. If an
    /// edit ever rewrote existing bytes, undo would silently corrupt.
    #[test]
    fn edits_only_append_never_rewrite() {
        let doc = flat_three();
        let original = doc.raw_bytes().to_vec();
        for edited in [
            insert_blank_page(&doc, 1, None).unwrap(),
            delete_page(&doc, 1).unwrap(),
        ] {
            assert!(edited.len() > original.len());
            assert_eq!(
                &edited[..original.len()],
                &original[..],
                "the original bytes must survive an edit untouched"
            );
        }
    }

    /// Successive edits must compose — a lecture adds many pages in a row.
    #[test]
    fn edits_compose_across_generations() {
        let mut doc = flat_three();
        for expected in 4..=8 {
            let n = doc.page_count();
            doc = PdfDocument::open(insert_blank_page(&doc, n, None).unwrap()).unwrap();
            assert_eq!(doc.page_count(), expected);
        }
    }

    /// A degenerate neighbour box must not produce a zero-area page that
    /// renders as nothing; A4 is the documented fallback.
    #[test]
    fn a_degenerate_neighbour_box_falls_back_to_a4() {
        let doc = build_pdf(&[
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 /MediaBox [0 0 0 0] >>",
            "<< /Type /Page /Parent 2 0 R >>",
        ]);
        let out = PdfDocument::open(insert_blank_page(&doc, 1, None).unwrap()).unwrap();
        let media = rect_from_array(&out, out.page(1).unwrap().dict_gets("MediaBox")).unwrap();
        assert_eq!(media.x1, A4_PORTRAIT.x1);
    }
}

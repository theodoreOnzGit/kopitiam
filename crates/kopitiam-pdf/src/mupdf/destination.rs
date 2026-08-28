//! Where a link or a bookmark points — the shared half of PDF navigation.
//!
//! Both the outline ([`super::outline`]) and page links ([`super::link`])
//! answer the same question in the same three ways, so the resolution lives
//! here once: an explicit destination array, a *named* destination to be
//! looked up, or an action (`/GoTo`, `/URI`, …).
//!
//! Ported from MuPDF `source/pdf/pdf-link.c` (`pdf_resolve_link`,
//! `pdf_parse_link_dest`, `pdf_lookup_dest`) and `source/pdf/pdf-outline.c`
//! (commit `0b8fd1c`, AGPL-3.0, © Artifex).
//!
//! # Why a resolver object rather than free functions
//!
//! Resolving one destination needs two lookups that are expensive to redo per
//! item and cheap to do once: a map from page object number to page **index**
//! (a destination names its page by indirect reference, and a reader needs an
//! index), and the `/Names` `/Dests` name tree. A 500-item outline resolved
//! with free functions would walk the page tree 500 times. [`Destinations`]
//! builds both once.
//!
//! # Deliberately not ported
//!
//! MuPDF renders every destination to a URI string
//! (`#page=3&zoom=100,0,600`) and parses it back, because its public API is
//! string-based across language bindings. That round trip loses type
//! information for no gain here, so this returns a typed [`Destination`]
//! instead. The `/GoToR` (remote file) and `/Launch` actions are recognised
//! but not followed: opening another document is a decision for the
//! application, not the parser.

use std::collections::HashMap;

use super::object::Object;
use super::xref::PdfDocument;

/// How far a name-tree walk will descend before giving up. A malformed tree
/// can be made to nest arbitrarily; this bounds it without a visited set,
/// which would cost more than the lookup itself.
const MAX_NAME_TREE_DEPTH: usize = 64;

/// Where a link or bookmark leads.
#[derive(Debug, Clone, PartialEq)]
pub enum Destination {
    /// Somewhere in this document.
    Page {
        /// 0-based page index.
        page: usize,
        /// Target point in PDF user space, when the destination names one
        /// (`/XYZ`). `None` for the fit-style destinations, which position
        /// the whole page and so have no meaningful point.
        left: Option<f32>,
        top: Option<f32>,
        /// `/XYZ` zoom. `None` or `0` in the file both mean "keep the
        /// current zoom" (§12.3.2.2), and both arrive here as `None`.
        zoom: Option<f32>,
    },
    /// An external URI (`/URI` action).
    Uri(String),
    /// A destination this crate understands the *shape* of but deliberately
    /// does not follow — `/GoToR`, `/Launch`, `/Named`. Carries the action
    /// subtype so a caller can report it rather than silently do nothing.
    Unsupported(String),
}

/// Resolves destinations for one document.
///
/// Build once per document and reuse: see the module docs on why this is an
/// object rather than a set of functions.
pub struct Destinations {
    /// Page object number -> 0-based page index.
    page_index: HashMap<i32, usize>,
}

impl Destinations {
    /// Index `doc`'s pages so destinations can name them.
    pub fn new(doc: &PdfDocument) -> Destinations {
        let mut page_index = HashMap::new();
        for i in 0..doc.page_count() {
            if let Some(num) = page_object_number(doc, i) {
                page_index.entry(num).or_insert(i);
            }
        }
        Destinations { page_index }
    }

    /// Resolve the destination of an outline item or link annotation.
    ///
    /// Checks `/Dest` first, then the `/A` action — the order MuPDF uses, and
    /// the order §12.3.2.1 implies, since a `/Dest` on the object itself is
    /// more specific than an action that happens to be attached.
    pub fn resolve(&self, doc: &PdfDocument, obj: &Object) -> Option<Destination> {
        if let Some(dest) = obj.dict_gets("Dest")
            && let Some(d) = self.resolve_dest_value(doc, dest)
        {
            return Some(d);
        }
        let action = doc.resolve(obj.dict_gets("A")?).ok()?;
        if !action.is_dict() {
            return None;
        }
        match action.dict_gets("S").map(|o| o.to_name()) {
            Some(b"GoTo") => {
                let d = action.dict_gets("D")?;
                self.resolve_dest_value(doc, d)
            }
            Some(b"URI") => {
                let uri = doc.resolve(action.dict_gets("URI")?).ok()?;
                uri.is_string()
                    .then(|| Destination::Uri(super::doc_info::decode_text_string(uri.to_string_bytes())))
            }
            Some(other) => Some(Destination::Unsupported(
                String::from_utf8_lossy(other).into_owned(),
            )),
            None => None,
        }
    }

    /// A `/Dest` value: an explicit array, or a name/string to look up in the
    /// document's destination tables.
    fn resolve_dest_value(&self, doc: &PdfDocument, dest: &Object) -> Option<Destination> {
        let dest = doc.resolve(dest).ok()?;
        if dest.is_array() {
            return self.explicit_dest(doc, &dest);
        }
        // A named destination: `/Name` (PDF 1.1 `/Dests` dict) or a string
        // (PDF 1.2+ `/Names` `/Dests` name tree). §12.3.2.3.
        let key: Vec<u8> = if dest.is_name() {
            dest.to_name().to_vec()
        } else if dest.is_string() {
            dest.to_string_bytes().to_vec()
        } else {
            return None;
        };
        // The name tree's *value* is very often an indirect reference to the
        // destination array rather than the array itself, so resolve before
        // asking what shape it is. Testing the unresolved reference silently
        // matches neither arm and loses every named destination in the file.
        let target = doc.resolve(&lookup_named_dest(doc, &key)?).ok()?;
        // A named destination may be the array directly, or a dict carrying
        // it under /D (§12.3.2.3).
        let arr = if target.is_array() {
            target
        } else {
            doc.resolve(target.dict_gets("D")?).ok()?
        };
        self.explicit_dest(doc, &arr)
    }

    /// An explicit destination array: `[page /Fit]`, `[page /XYZ l t z]`, …
    /// (§12.3.2.2, Table 151).
    fn explicit_dest(&self, doc: &PdfDocument, arr: &Object) -> Option<Destination> {
        let page_ref = arr.array_get(0)?;
        let page = match page_ref {
            // The usual form: an indirect reference to the page object.
            Object::Ref { num, .. } => *self.page_index.get(num)?,
            // A remote/embedded destination names its page by NUMBER instead.
            // Valid in /GoToR; harmless to accept here.
            o if o.is_number() => {
                let n = o.to_int();
                if n < 0 {
                    return None;
                }
                n as usize
            }
            _ => return None,
        };

        let kind = arr.array_get(1).map(|o| o.to_name()).unwrap_or(b"");
        let num_at = |i: usize| -> Option<f32> {
            let v = doc.resolve(arr.array_get(i)?).ok()?;
            v.is_number().then(|| v.to_real() as f32)
        };
        let (left, top, zoom) = match kind {
            // [page /XYZ left top zero] -- any of the three may be null,
            // meaning "leave this one as it is".
            b"XYZ" => (
                num_at(2),
                num_at(3),
                // Zoom 0 means "unchanged" exactly as null does (§12.3.2.2),
                // so both become None rather than a literal 0x zoom.
                num_at(4).filter(|z| *z != 0.0),
            ),
            // [page /FitH top] and [page /FitBH top]
            b"FitH" | b"FitBH" => (None, num_at(2), None),
            // [page /FitV left] and [page /FitBV left]
            b"FitV" | b"FitBV" => (num_at(2), None, None),
            // [page /FitR left bottom right top] -- take the top-left corner,
            // which is where a reader scrolls to.
            b"FitR" => (num_at(2), num_at(5), None),
            // /Fit, /FitB, and anything unrecognised: the whole page.
            _ => (None, None, None),
        };
        Some(Destination::Page { page, left, top, zoom })
    }
}

/// The object number of page `index`, or `None` if it has no indirect
/// identity (a malformed file with a direct page dict).
fn page_object_number(doc: &PdfDocument, index: usize) -> Option<i32> {
    super::page_edit::locate_page_slot(doc, index)
        .ok()
        .map(|slot| slot.page_num)
}

/// Look `key` up in the document's destination tables.
///
/// Tries the modern `/Names` `/Dests` **name tree** first, then the PDF 1.1
/// `/Dests` dictionary in the catalog — a file may carry either, and some
/// carry both.
fn lookup_named_dest(doc: &PdfDocument, key: &[u8]) -> Option<Object> {
    let root = doc.catalog().ok()?;
    if let Ok(names) = doc.resolve_get(&root, "Names")
        && names.is_dict()
        && let Some(dests) = names.dict_gets("Dests")
        && let Some(found) = name_tree_lookup(doc, dests, key, 0)
    {
        return Some(found);
    }
    // PDF 1.1: a plain dictionary keyed by name.
    let dests = doc.resolve_get(&root, "Dests").ok()?;
    dests.is_dict().then(|| doc.resolve(dests.dict_get(key)?).ok())?
}

/// Walk a PDF **name tree** (§7.9.6) looking for `key`.
///
/// A node has either `/Names` (a flat `[key value key value …]` array, sorted
/// by key) or `/Kids` (child nodes, each with a `/Limits [first last]` range).
/// The `/Limits` are used to descend into only the one child that can contain
/// the key; a node without `/Limits` is searched anyway rather than skipped,
/// since producers do omit them.
fn name_tree_lookup(doc: &PdfDocument, node: &Object, key: &[u8], depth: usize) -> Option<Object> {
    if depth > MAX_NAME_TREE_DEPTH {
        return None;
    }
    let node = doc.resolve(node).ok()?;
    if !node.is_dict() {
        return None;
    }

    if let Ok(names) = doc.resolve_get(&node, "Names")
        && names.is_array()
    {
        // Pairs are sorted, but a linear scan is right here: the arrays are
        // small (the tree exists precisely so they stay small), and a binary
        // search over a possibly-unsorted array from a sloppy producer would
        // miss entries a scan finds.
        let mut i = 0;
        while i + 1 < names.array_len() {
            if let Some(k) = names.array_get(i)
                && k.is_string()
                && k.to_string_bytes() == key
            {
                return names.array_get(i + 1).cloned();
            }
            i += 2;
        }
    }

    let kids = doc.resolve_get(&node, "Kids").ok()?;
    for i in 0..kids.array_len() {
        let kid_ref = kids.array_get(i)?;
        let kid = doc.resolve(kid_ref).ok()?;
        if !kid.is_dict() {
            continue;
        }
        if !within_limits(doc, &kid, key) {
            continue;
        }
        if let Some(found) = name_tree_lookup(doc, kid_ref, key, depth + 1) {
            return Some(found);
        }
    }
    None
}

/// Could this node contain `key`, per its `/Limits`? A node with no or
/// malformed `/Limits` returns `true` — searched rather than skipped, since
/// wrongly skipping loses a real destination while wrongly searching only
/// costs time.
fn within_limits(doc: &PdfDocument, node: &Object, key: &[u8]) -> bool {
    let Ok(limits) = doc.resolve_get(node, "Limits") else {
        return true;
    };
    if limits.array_len() != 2 {
        return true;
    }
    let (Some(lo), Some(hi)) = (limits.array_get(0), limits.array_get(1)) else {
        return true;
    };
    if !lo.is_string() || !hi.is_string() {
        return true;
    }
    lo.to_string_bytes() <= key && key <= hi.to_string_bytes()
}

#[cfg(test)]
pub(crate) fn build_pdf(bodies: &[String]) -> PdfDocument {
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
        format!("trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").as_bytes(),
    );
    PdfDocument::open(pdf).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Catalog + 3-page tree in objects 1..5, leaving 6+ free for the test's
    /// own objects. Pages are objects 3, 4, 5.
    fn base(extra: &[&str], catalog_extra: &str) -> Vec<String> {
        let mut v = vec![
            format!("<< /Type /Catalog /Pages 2 0 R {catalog_extra} >>"),
            "<< /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 3 /MediaBox [0 0 600 800] >>"
                .to_string(),
            "<< /Type /Page /Parent 2 0 R >>".to_string(),
            "<< /Type /Page /Parent 2 0 R >>".to_string(),
            "<< /Type /Page /Parent 2 0 R >>".to_string(),
        ];
        v.extend(extra.iter().map(|s| s.to_string()));
        v
    }

    /// An object holding `/Dest`, so `resolve` has something to chew on.
    fn resolve_dest(dest: &str, extra: &[&str], catalog_extra: &str) -> Option<Destination> {
        let mut bodies = base(extra, catalog_extra);
        bodies.push(format!("<< /Dest {dest} >>"));
        let holder = bodies.len(); // 1-based object number
        let doc = build_pdf(&bodies);
        let d = Destinations::new(&doc);
        let obj = doc
            .resolve(&Object::new_indirect(holder as i64, 0))
            .unwrap();
        d.resolve(&doc, &obj)
    }

    #[test]
    fn explicit_xyz_gives_page_and_point() {
        let got = resolve_dest("[4 0 R /XYZ 72 700 0]", &[], "").unwrap();
        assert_eq!(
            got,
            Destination::Page {
                page: 1,
                left: Some(72.0),
                top: Some(700.0),
                // Zoom 0 means "unchanged" (§12.3.2.2), not 0x.
                zoom: None,
            }
        );
    }

    #[test]
    fn explicit_zoom_is_kept_when_real() {
        let Destination::Page { zoom, .. } = resolve_dest("[3 0 R /XYZ 0 0 2.5]", &[], "").unwrap()
        else {
            panic!("expected a page destination")
        };
        assert_eq!(zoom, Some(2.5));
    }

    #[test]
    fn the_fit_forms_position_the_page() {
        // /Fit: whole page, no point.
        assert_eq!(
            resolve_dest("[5 0 R /Fit]", &[], "").unwrap(),
            Destination::Page { page: 2, left: None, top: None, zoom: None }
        );
        // /FitH names only a top; /FitV only a left.
        let Destination::Page { left, top, .. } = resolve_dest("[3 0 R /FitH 500]", &[], "").unwrap()
        else {
            panic!()
        };
        assert_eq!((left, top), (None, Some(500.0)));
        let Destination::Page { left, top, .. } = resolve_dest("[3 0 R /FitV 90]", &[], "").unwrap()
        else {
            panic!()
        };
        assert_eq!((left, top), (Some(90.0), None));
        // /FitR takes the rectangle's top-left, which is where a reader scrolls.
        let Destination::Page { left, top, .. } =
            resolve_dest("[3 0 R /FitR 10 20 300 400]", &[], "").unwrap()
        else {
            panic!()
        };
        assert_eq!((left, top), (Some(10.0), Some(400.0)));
    }

    /// The regression this module's biggest bug produced: a name tree's value
    /// is usually an **indirect reference** to the destination array. Testing
    /// the unresolved reference matches neither the array nor the dict arm, and
    /// every named destination in the file silently resolves to nothing --
    /// which is exactly how the arXiv fixture's whole 43-item outline came back
    /// with `dest: None`.
    #[test]
    fn a_named_dest_behind_an_indirect_reference_resolves() {
        let got = resolve_dest(
            "(chapter.1)",
            &[
                // 6: the name tree leaf, whose value is a REFERENCE to 7.
                "<< /Names [(chapter.1) 7 0 R] >>",
                // 7: the destination array itself.
                "[4 0 R /XYZ 72 700 null]",
            ],
            "/Names << /Dests 6 0 R >>",
        )
        .expect("a named destination must resolve through the reference");
        assert_eq!(
            got,
            Destination::Page { page: 1, left: Some(72.0), top: Some(700.0), zoom: None }
        );
    }

    /// A named destination may also be a dict carrying the array under /D.
    #[test]
    fn a_named_dest_wrapped_in_a_dict_resolves() {
        let got = resolve_dest(
            "(x)",
            &["<< /Names [(x) 7 0 R] >>", "<< /D [5 0 R /Fit] >>"],
            "/Names << /Dests 6 0 R >>",
        )
        .unwrap();
        assert_eq!(
            got,
            Destination::Page { page: 2, left: None, top: None, zoom: None }
        );
    }

    /// The tree must be descended by /Limits, not scanned blindly.
    #[test]
    fn a_named_dest_is_found_through_a_kids_level() {
        let got = resolve_dest(
            "(m)",
            &[
                // 6: root with two limited kids.
                "<< /Kids [7 0 R 8 0 R] >>",
                // 7: covers a..f -- must be skipped.
                "<< /Limits [(a) (f)] /Names [(a) 9 0 R] >>",
                // 8: covers g..z -- contains the key.
                "<< /Limits [(g) (z)] /Names [(m) 10 0 R] >>",
                "[3 0 R /Fit]",
                "[5 0 R /Fit]",
            ],
            "/Names << /Dests 6 0 R >>",
        )
        .unwrap();
        assert_eq!(
            got,
            Destination::Page { page: 2, left: None, top: None, zoom: None },
            "must reach the kid whose /Limits contain the key"
        );
    }

    /// PDF 1.1 kept named destinations in a plain catalog `/Dests` dict.
    #[test]
    fn the_pdf_1_1_dests_dictionary_still_works() {
        let got = resolve_dest(
            "/mydest",
            &["[4 0 R /Fit]"],
            "/Dests << /mydest 6 0 R >>",
        )
        .unwrap();
        assert_eq!(
            got,
            Destination::Page { page: 1, left: None, top: None, zoom: None }
        );
    }

    #[test]
    fn a_uri_action_resolves() {
        let mut bodies = base(&[], "");
        bodies.push("<< /A << /S /URI /URI (https://example.org/a) >> >>".to_string());
        let holder = bodies.len();
        let doc = build_pdf(&bodies);
        let d = Destinations::new(&doc);
        let obj = doc.resolve(&Object::new_indirect(holder as i64, 0)).unwrap();
        assert_eq!(
            d.resolve(&doc, &obj),
            Some(Destination::Uri("https://example.org/a".to_string()))
        );
    }

    /// An action we understand the shape of but decline to follow is reported
    /// as such, not silently dropped — the caller can then say why nothing
    /// happened.
    #[test]
    fn an_unfollowed_action_names_itself() {
        let mut bodies = base(&[], "");
        bodies.push("<< /A << /S /GoToR /F (other.pdf) >> >>".to_string());
        let holder = bodies.len();
        let doc = build_pdf(&bodies);
        let d = Destinations::new(&doc);
        let obj = doc.resolve(&Object::new_indirect(holder as i64, 0)).unwrap();
        assert_eq!(
            d.resolve(&doc, &obj),
            Some(Destination::Unsupported("GoToR".to_string()))
        );
    }

    #[test]
    fn a_missing_or_dangling_destination_is_none() {
        assert_eq!(resolve_dest("(nosuchname)", &[], ""), None);
        // A page reference that is not in this document's page tree.
        assert_eq!(resolve_dest("[99 0 R /Fit]", &[], ""), None);
    }
}

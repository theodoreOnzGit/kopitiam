//! Ported from MuPDF `source/pdf/pdf-op-run.c` (`set_struct_parent`,
//! `lookup_mcid`, `pdf_lookup_mcid_in_mcids`, `int_in_singleton_or_array`) and
//! `source/pdf/pdf-nametree.c` (`pdf_lookup_number`) (commit 19f1284,
//! AGPL-3.0, © Artifex Software, Inc.), translated to Rust for KOPITIAM
//! (AGPL-3.0-only). Close adaptation: the algorithms and numeric behaviour
//! follow MuPDF; the code is re-expressed in idiomatic Rust. See
//! docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references").
//!
//! # Why (0.4.2)
//!
//! A tagged PDF often keeps a span's `/ActualText` not in the `BDC`
//! properties but on the STRUCTURE ELEMENT the span's `/MCID` points to:
//! page `/StructParents` -> `/StructTreeRoot /ParentTree` (a number tree) ->
//! the array of structure elements for that page, indexed by MCID. MuPDF
//! follows that chain (`lookup_mcid`) before deciding whether marked content
//! carries replacement text. The harness found it on NUREG/CR-7289 p. 2:
//! 144 chars in MCID spans whose structure elements say `/ActualText ()`,
//! which MuPDF extracts as nothing.

use super::object::Object;
use super::xref::PdfDocument;

// MuPDF: set_struct_parent (pdf-op-run.c:2397): the ParentTree entry for a
// page's (or form's) StructParent(s) number. Null when there is none.
pub(crate) fn mcids_for_struct_parent(doc: &PdfDocument, struct_parent: i64) -> Object {
    if struct_parent < 0 {
        return Object::Null;
    }
    let root = doc.catalog().unwrap_or(Object::Null);
    let str_root = doc.resolve_get(&root, "StructTreeRoot").unwrap_or(Object::Null);
    let parent_tree = doc.resolve_get(&str_root, "ParentTree").unwrap_or(Object::Null);
    lookup_number(doc, &parent_tree, struct_parent, 0)
        .and_then(|o| doc.resolve(&o).ok())
        .unwrap_or(Object::Null)
}

// MuPDF: pdf_lookup_number_imp (pdf-nametree.c:219). `depth` stands in for
// the pdf_cycle list.
fn lookup_number(doc: &PdfDocument, node: &Object, needle: i64, depth: u32) -> Option<Object> {
    if depth > 32 {
        return None;
    }
    let node = doc.resolve(node).ok()?;
    let int_at = |arr: &Object, i: usize| arr.array_get(i).and_then(|o| doc.resolve(o).ok()).map_or(0, |o| o.to_int());
    let kids = doc.resolve_get(&node, "Kids").unwrap_or(Object::Null);
    if kids.is_array() && kids.array_len() > 0 {
        let (mut l, mut r) = (0i64, kids.array_len() as i64 - 1);
        while l <= r {
            let m = (l + r) >> 1;
            let kid = kids.array_get(m as usize).cloned().unwrap_or(Object::Null);
            let limits = doc.resolve_get(&kid, "Limits").unwrap_or(Object::Null);
            let (first, last) = (int_at(&limits, 0), int_at(&limits, 1));
            if needle < first {
                r = m - 1;
            } else if needle > last {
                l = m + 1;
            } else {
                return lookup_number(doc, &kid, needle, depth + 1);
            }
        }
    }
    let nums = doc.resolve_get(&node, "Nums").unwrap_or(Object::Null);
    if nums.is_array() {
        let pairs = nums.array_len() / 2;
        let (mut l, mut r) = (0i64, pairs as i64 - 1);
        while l <= r {
            let m = (l + r) >> 1;
            let key = int_at(&nums, m as usize * 2);
            if needle < key {
                r = m - 1;
            } else if needle > key {
                l = m + 1;
            } else {
                return nums.array_get(m as usize * 2 + 1).cloned();
            }
        }
        // "allowing for non-sorted lists"
        for i in 0..pairs {
            if int_at(&nums, i * 2) == needle {
                return nums.array_get(i * 2 + 1).cloned();
            }
        }
    }
    None
}

// MuPDF: int_in_singleton_or_array (pdf-op-run.c:1470) -- only plain
// integers count, as in the C (an MCR dictionary in /K does not).
fn int_in_singleton_or_array(doc: &PdfDocument, k: &Object, id: i64) -> bool {
    let k = doc.resolve(k).unwrap_or(Object::Null);
    if k.is_int() && k.to_int() == id {
        return true;
    }
    (0..k.array_len()).any(|i| {
        k.array_get(i).and_then(|o| doc.resolve(o).ok()).is_some_and(|o| o.is_int() && o.to_int() == id)
    })
}

// MuPDF: lookup_mcid + pdf_lookup_mcid_in_mcids (pdf-op-run.c:1494-1535):
// the structure element for the `/MCID` in a BDC properties dict, checked by
// its `/K`, with MuPDF's search fallback for mis-indexed arrays.
pub(crate) fn lookup_mcid(doc: &PdfDocument, mcids: &Object, props: &Object) -> Option<Object> {
    if mcids.is_null() {
        return None;
    }
    let mcid = doc.resolve_get(props, "MCID").ok()?;
    if !mcid.is_number() {
        return None;
    }
    let id = mcid.to_int();
    let elem_at = |i: usize| mcids.array_get(i).and_then(|o| doc.resolve(o).ok()).unwrap_or(Object::Null);
    let has = |e: &Object| {
        let k = doc.resolve_get(e, "K").unwrap_or(Object::Null);
        int_in_singleton_or_array(doc, &k, id)
    };
    if id >= 0 {
        let e = elem_at(id as usize);
        if has(&e) {
            return Some(e);
        }
    }
    (0..mcids.array_len()).map(elem_at).find(|e| has(e))
}

//! Ported from MuPDF `source/pdf/pdf-object.c` + `include/mupdf/pdf/object.h`
//! (commit 19f1284, AGPL-3.0, © Artifex Software, Inc.), translated to Rust for
//! KOPITIAM (AGPL-3.0-only). Close adaptation: the algorithms and numeric
//! behaviour follow MuPDF; the code is re-expressed in idiomatic Rust. See
//! docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references").
//!
//! # `pdf_obj` -> [`Object`]
//!
//! MuPDF's `pdf_obj` is a polymorphic, reference-counted C struct: a one-byte
//! `kind` tag (`pdf_objkind_e`: `'i' 'f' 's' 'n' 'a' 'd' 'r'`) discriminates a
//! union of int / real / string / name / array / dict / indirect payloads, and
//! three values (`PDF_NULL`, `PDF_TRUE`, `PDF_FALSE`) plus the whole name table
//! are *tagged pointers* below `PDF_LIMIT` rather than heap objects. Ownership is
//! manual (`pdf_keep_obj`/`pdf_drop_obj` on `refs`).
//!
//! This port re-expresses that as a plain Rust `enum` owning a tree of children.
//! The MuPDF **semantics** are preserved exactly; only the representation
//! changes:
//!
//! * **Tagged sentinels become variants.** `PDF_NULL` -> [`Object::Null`],
//!   `PDF_TRUE`/`PDF_FALSE` -> [`Object::Bool`]. The static name table (the other
//!   sub-`PDF_LIMIT` pointers) is not modelled as interned constants -- every
//!   name is just its bytes in [`Object::Name`]; equality is by bytes, which is
//!   what `pdf_name_eq` ultimately computes.
//! * **`kind` + union become the enum discriminant + payload.** The int-vs-real
//!   distinction (`PDF_INT`/`PDF_REAL`) is kept as two variants, and the numeric
//!   accessors reproduce MuPDF's cross-type coercion (`pdf_to_int` rounds a real
//!   via `floorf(f + 0.5)`, `pdf_to_real` widens an int).
//! * **Name vs string stay distinct.** PDF's two byte-string types are different
//!   variants ([`Object::Name`] and [`Object::String`]), both `Vec<u8>` so they
//!   hold arbitrary bytes (a name after `#xx` decoding, a string's binary data).
//! * **Refcounting is dropped.** MuPDF shares one `pdf_obj` between owners via
//!   `refs`; here an [`Object`] owns its subtree and `Clone` deep-copies
//!   (`pdf_deep_copy_obj`). A later layer that needs *shared* objects wraps them
//!   in `Rc`/`Arc` at that site.
//!
//! ## No `Stream` variant -- the seam for the xref layer
//!
//! MuPDF has **no** stream object kind: a PDF stream is just an
//! [`Object::Dict`] whose object number the cross-reference table separately
//! marks as carrying stream bytes (`pdf_obj_num_is_stream`). `pdf_is_stream`
//! consults the document, not the object. So this port keeps streams *out* of
//! [`Object`] too: the parser returns the stream's dict as a normal dict plus a
//! recorded byte offset (see `parse::IndirectObject`), and the xref/stream layer
//! wires up the raw bytes and decode filters later. That recorded offset is the
//! clean seam; adding an `Object::Stream` here would diverge from MuPDF (where a
//! stream *is* a dict, and `pdf_is_dict` on it is true).
//!
//! ## Not ported from this file (deliberately)
//!
//! The document-bound machinery -- indirect *resolution* (`pdf_resolve_indirect`,
//! the `RESOLVE` macro that every accessor invokes), `parent_num`/`doc`
//! back-pointers, dirty/marked/memo flags, journalling, the printer
//! (`pdf_sprint_obj`), and the mutation-with-`prepare_object_for_alteration`
//! paths -- belongs to the xref/document layer and is out of scope for the object
//! model itself. Accessors here therefore do **not** auto-resolve an
//! [`Object::Ref`]; with no document to consult, they treat it as a type
//! mismatch and return the MuPDF default (0 / "" / `None`), exactly as MuPDF's
//! accessors do for an unresolved non-matching object. The `pdf_dict_get`
//! sorted-dict binary search is likewise deferred: parsed dicts are unsorted in
//! MuPDF (the `SORTED` flag is only set by `pdf_sort_dict`), so [`Object::get`]
//! does the faithful linear scan over the unsorted `items`.

use super::error::{Error, Result};

/// Defined in PDF 1.7 according to the Acrobat limit (`PDF_MAX_OBJECT_NUMBER`).
pub const MAX_OBJECT_NUMBER: i32 = 8_388_607;
/// The maximum PDF generation number (`PDF_MAX_GEN_NUMBER`).
pub const MAX_GEN_NUMBER: i32 = 65_535;

// MuPDF: struct pdf_obj / pdf_objkind_e (pdf-object.c:40, 66)
/// A dynamic PDF object: the same object types found in PDF and PostScript,
/// used by the filters and the parser.
///
/// This is the Rust re-expression of MuPDF's `pdf_obj` (see the module docs). It
/// owns its subtree; [`Clone`] deep-copies (`pdf_deep_copy_obj`).
#[derive(Clone, Debug, PartialEq)]
pub enum Object {
    /// `PDF_NULL` -- the null object.
    Null,
    /// `PDF_TRUE` / `PDF_FALSE` -- a boolean.
    Bool(bool),
    /// `PDF_INT` -- an integer (MuPDF stores `int64_t`).
    Int(i64),
    /// `PDF_REAL` -- a real number (MuPDF stores `float`; widened to `f64` here).
    Real(f64),
    /// `PDF_STRING` -- a byte string (may hold arbitrary/binary bytes).
    String(Vec<u8>),
    /// `PDF_NAME` -- a name, stored as its decoded bytes without the leading `/`.
    Name(Vec<u8>),
    /// `PDF_ARRAY` -- an ordered list of objects.
    Array(Vec<Object>),
    /// `PDF_DICT` -- key/value pairs; keys are names (their decoded bytes).
    ///
    /// Stored in insertion order (unsorted), matching a freshly parsed MuPDF
    /// dict. Keys are unique: [`Object::dict_put`] replaces an existing key.
    Dict(Vec<(Vec<u8>, Object)>),
    /// `PDF_INDIRECT` -- an indirect reference `num gen R`.
    ///
    /// The generation field is spelled `generation` because `gen` is a reserved
    /// keyword in Rust edition 2024.
    Ref {
        /// The referenced object number.
        num: i32,
        /// The generation number.
        generation: i32,
    },
}

impl Object {
    // -----------------------------------------------------------------------
    // Constructors (pdf_new_*)
    // -----------------------------------------------------------------------

    // MuPDF: pdf_new_int (pdf-object.c:176)
    /// Create an integer object.
    pub fn new_int(i: i64) -> Object {
        Object::Int(i)
    }

    // MuPDF: pdf_new_real (pdf-object.c:188)
    /// Create a real object.
    pub fn new_real(f: f64) -> Object {
        Object::Real(f)
    }

    // MuPDF: pdf_new_string (pdf-object.c:200)
    /// Create a string object from raw bytes.
    pub fn new_string(bytes: impl Into<Vec<u8>>) -> Object {
        Object::String(bytes.into())
    }

    // MuPDF: pdf_new_name (pdf-object.c:220)
    /// Create a name object from its (already `#xx`-decoded) bytes.
    pub fn new_name(bytes: impl Into<Vec<u8>>) -> Object {
        Object::Name(bytes.into())
    }

    // MuPDF: pdf_new_indirect (pdf-object.c:246)
    /// Create an indirect reference. Out-of-range `num`/`gen` collapse to
    /// [`Object::Null`], matching `pdf_new_indirect`'s warn-and-return-`PDF_NULL`.
    pub fn new_indirect(num: i64, generation: i32) -> Object {
        if !(0..=MAX_OBJECT_NUMBER as i64).contains(&num) {
            return Object::Null;
        }
        if !(0..=MAX_GEN_NUMBER).contains(&generation) {
            return Object::Null;
        }
        Object::Ref {
            num: num as i32,
            generation,
        }
    }

    /// Create an empty array object.
    pub fn new_array() -> Object {
        Object::Array(Vec::new())
    }

    /// Create an empty dict object.
    pub fn new_dict() -> Object {
        Object::Dict(Vec::new())
    }

    // -----------------------------------------------------------------------
    // Predicates (pdf_is_*)
    // -----------------------------------------------------------------------
    //
    // MuPDF's predicates first RESOLVE indirects against the document; with no
    // document here they operate on the object as-is (see the module docs).

    // MuPDF: pdf_is_null (pdf-object.c:296)
    /// `pdf_is_null`.
    pub fn is_null(&self) -> bool {
        matches!(self, Object::Null)
    }

    // MuPDF: pdf_is_bool (pdf-object.c:302)
    /// `pdf_is_bool`.
    pub fn is_bool(&self) -> bool {
        matches!(self, Object::Bool(_))
    }

    // MuPDF: pdf_is_int (pdf-object.c:308)
    /// `pdf_is_int`.
    pub fn is_int(&self) -> bool {
        matches!(self, Object::Int(_))
    }

    // MuPDF: pdf_is_real (pdf-object.c:314)
    /// `pdf_is_real`.
    pub fn is_real(&self) -> bool {
        matches!(self, Object::Real(_))
    }

    // MuPDF: pdf_is_number (pdf-object.c:320)
    /// `pdf_is_number` -- an int or a real.
    pub fn is_number(&self) -> bool {
        matches!(self, Object::Int(_) | Object::Real(_))
    }

    // MuPDF: pdf_is_string (pdf-object.c:326)
    /// `pdf_is_string`.
    pub fn is_string(&self) -> bool {
        matches!(self, Object::String(_))
    }

    // MuPDF: pdf_is_name (pdf-object.c:332)
    /// `pdf_is_name`.
    pub fn is_name(&self) -> bool {
        matches!(self, Object::Name(_))
    }

    // MuPDF: pdf_is_array (pdf-object.c:338)
    /// `pdf_is_array`.
    pub fn is_array(&self) -> bool {
        matches!(self, Object::Array(_))
    }

    // MuPDF: pdf_is_dict (pdf-object.c:344)
    /// `pdf_is_dict`.
    pub fn is_dict(&self) -> bool {
        matches!(self, Object::Dict(_))
    }

    // MuPDF: pdf_is_indirect (pdf-object.c:291)
    /// `pdf_is_indirect`.
    pub fn is_indirect(&self) -> bool {
        matches!(self, Object::Ref { .. })
    }

    // MuPDF: pdf_ensure_indirect (pdf-object.c:351)
    /// If this is an indirect reference return it, else throw
    /// (`FZ_ERROR_ARGUMENT`), matching `pdf_ensure_indirect`.
    pub fn ensure_indirect(&self) -> Result<&Object> {
        if self.is_indirect() {
            Ok(self)
        } else {
            Err(Error::argument("Indirect object required"))
        }
    }

    // -----------------------------------------------------------------------
    // Scalar accessors (pdf_to_*)
    // -----------------------------------------------------------------------
    //
    // "safe, silent failure, no error reporting on type mismatches"
    // (pdf-object.c:359)

    // MuPDF: pdf_to_bool (pdf-object.c:360)
    /// `pdf_to_bool` -- true only for the boolean `true`.
    pub fn to_bool(&self) -> bool {
        matches!(self, Object::Bool(true))
    }

    // MuPDF: pdf_to_int (pdf-object.c:372) / pdf_to_int64 (pdf-object.c:396)
    /// `pdf_to_int64` -- an int as-is, a real rounded via `floor(f + 0.5)`, else
    /// 0. (`pdf_to_int` is this truncated to a C `int`; use `as i32` if needed.)
    pub fn to_int(&self) -> i64 {
        match self {
            Object::Int(i) => *i,
            // floorf(f + 0.5) -- round half up, as MuPDF does.
            Object::Real(f) => (f + 0.5).floor() as i64,
            _ => 0,
        }
    }

    // MuPDF: pdf_to_real (pdf-object.c:408)
    /// `pdf_to_real` -- a real as-is, an int widened, else 0.
    pub fn to_real(&self) -> f64 {
        match self {
            Object::Real(f) => *f,
            Object::Int(i) => *i as f64,
            _ => 0.0,
        }
    }

    // MuPDF: pdf_to_name (pdf-object.c:432)
    /// `pdf_to_name` -- the name's bytes, or an empty slice for a non-name.
    pub fn to_name(&self) -> &[u8] {
        match self {
            Object::Name(n) => n,
            _ => b"",
        }
    }

    // MuPDF: pdf_to_string / pdf_to_str_buf (pdf-object.c:458, 442)
    /// `pdf_to_string` -- the string's bytes, or an empty slice for a non-string.
    pub fn to_string_bytes(&self) -> &[u8] {
        match self {
            Object::String(s) => s,
            _ => b"",
        }
    }

    // MuPDF: pdf_to_num (pdf-object.c:501)
    /// `pdf_to_num` -- an indirect reference's object number, else 0.
    pub fn to_num(&self) -> i32 {
        match self {
            Object::Ref { num, .. } => *num,
            _ => 0,
        }
    }

    // MuPDF: pdf_to_gen (pdf-object.c:508)
    /// `pdf_to_gen` -- an indirect reference's generation number, else 0.
    pub fn to_gen(&self) -> i32 {
        match self {
            Object::Ref { generation, .. } => *generation,
            _ => 0,
        }
    }

    // MuPDF: pdf_name_eq (pdf-object.c:763)
    /// `pdf_name_eq` -- true iff both objects are names with equal bytes.
    pub fn name_eq(&self, other: &Object) -> bool {
        match (self, other) {
            (Object::Name(a), Object::Name(b)) => a == b,
            _ => false,
        }
    }

    // -----------------------------------------------------------------------
    // Array accessors (pdf_array_*)
    // -----------------------------------------------------------------------

    // MuPDF: pdf_array_len (pdf-object.c:874)
    /// `pdf_array_len` -- the element count, or 0 for a non-array.
    pub fn array_len(&self) -> usize {
        match self {
            Object::Array(items) => items.len(),
            _ => 0,
        }
    }

    // MuPDF: pdf_array_get (pdf-object.c:883)
    /// `pdf_array_get` -- the element at `i`, or `None` if out of range or not an
    /// array.
    pub fn array_get(&self, i: usize) -> Option<&Object> {
        match self {
            Object::Array(items) => items.get(i),
            _ => None,
        }
    }

    // MuPDF: pdf_array_push (pdf-object.c:1994)
    /// `pdf_array_push` -- append `item`. No-op on a non-array (MuPDF throws;
    /// here the builder API keeps it infallible for the parser's use).
    pub fn array_push(&mut self, item: Object) {
        if let Object::Array(items) = self {
            items.push(item);
        }
    }

    // -----------------------------------------------------------------------
    // Dict accessors (pdf_dict_*)
    // -----------------------------------------------------------------------

    // MuPDF: pdf_dict_len (pdf-object.c:2256)
    /// `pdf_dict_len` -- the entry count, or 0 for a non-dict.
    pub fn dict_len(&self) -> usize {
        match self {
            Object::Dict(items) => items.len(),
            _ => 0,
        }
    }

    // MuPDF: pdf_dict_get_key (pdf-object.c:2265)
    /// `pdf_dict_get_key` -- the key (name bytes) of entry `i`.
    pub fn dict_get_key(&self, i: usize) -> Option<&[u8]> {
        match self {
            Object::Dict(items) => items.get(i).map(|(k, _)| k.as_slice()),
            _ => None,
        }
    }

    // MuPDF: pdf_dict_get_val (pdf-object.c:2276)
    /// `pdf_dict_get_val` -- the value of entry `i`.
    pub fn dict_get_val(&self, i: usize) -> Option<&Object> {
        match self {
            Object::Dict(items) => items.get(i).map(|(_, v)| v),
            _ => None,
        }
    }

    // MuPDF: pdf_dict_gets / pdf_dict_finds (pdf-object.c:2395, 2303)
    /// `pdf_dict_gets` -- look up a value by key bytes. Linear scan over the
    /// unsorted entries, matching MuPDF's unsorted-dict path (parsed dicts are
    /// not `SORTED`). Returns `None` if absent or not a dict.
    pub fn dict_get(&self, key: &[u8]) -> Option<&Object> {
        match self {
            Object::Dict(items) => items
                .iter()
                .find(|(k, _)| k.as_slice() == key)
                .map(|(_, v)| v),
            _ => None,
        }
    }

    /// Convenience: [`dict_get`](Object::dict_get) for an ASCII key literal.
    pub fn dict_gets(&self, key: &str) -> Option<&Object> {
        self.dict_get(key.as_bytes())
    }

    // MuPDF: pdf_dict_put / pdf_dict_get_put (pdf-object.c:2556, 2502)
    /// `pdf_dict_put` -- insert or replace `key`'s value. If `key` already
    /// exists its value is overwritten (last-value-wins, as `pdf_dict_get_put`
    /// does); otherwise a new entry is appended. No-op on a non-dict.
    pub fn dict_put(&mut self, key: impl Into<Vec<u8>>, val: Object) {
        if let Object::Dict(items) = self {
            let key = key.into();
            if let Some(slot) = items.iter_mut().find(|(k, _)| *k == key) {
                slot.1 = val;
            } else {
                items.push((key, val));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicates_classify_each_kind() {
        assert!(Object::Null.is_null());
        assert!(Object::Bool(true).is_bool());
        assert!(Object::new_int(3).is_int());
        assert!(Object::new_real(3.5).is_real());
        assert!(Object::new_int(3).is_number());
        assert!(Object::new_real(3.5).is_number());
        assert!(Object::new_string("hi").is_string());
        assert!(Object::new_name("Type").is_name());
        assert!(Object::new_array().is_array());
        assert!(Object::new_dict().is_dict());
        assert!(Object::new_indirect(12, 0).is_indirect());

        // Cross-kind negatives.
        assert!(!Object::new_int(1).is_real());
        assert!(!Object::new_name("X").is_string());
        assert!(!Object::new_string(b"X".to_vec()).is_name());
        assert!(!Object::Null.is_number());
    }

    #[test]
    fn name_and_string_are_distinct() {
        // Same bytes, different PDF types -- MuPDF keeps them apart.
        let name = Object::new_name(b"Foo".to_vec());
        let string = Object::new_string(b"Foo".to_vec());
        assert!(name.is_name() && !name.is_string());
        assert!(string.is_string() && !string.is_name());
        assert_ne!(name, string);
        assert_eq!(name.to_name(), b"Foo");
        assert_eq!(name.to_string_bytes(), b""); // a name is not a string
        assert_eq!(string.to_string_bytes(), b"Foo");
        assert_eq!(string.to_name(), b""); // a string is not a name
    }

    #[test]
    fn int_vs_real_numeric_coercion() {
        // pdf_to_int rounds a real via floor(f + 0.5).
        assert_eq!(Object::new_real(3.4).to_int(), 3);
        assert_eq!(Object::new_real(3.5).to_int(), 4);
        assert_eq!(Object::new_real(3.6).to_int(), 4);
        assert_eq!(Object::new_int(7).to_int(), 7);
        // pdf_to_real widens an int.
        assert_eq!(Object::new_int(7).to_real(), 7.0);
        assert_eq!(Object::new_real(2.5).to_real(), 2.5);
        // Type mismatch -> 0 (silent).
        assert_eq!(Object::new_name("X").to_int(), 0);
        assert_eq!(Object::new_name("X").to_real(), 0.0);
    }

    #[test]
    fn bool_accessor() {
        assert!(Object::Bool(true).to_bool());
        assert!(!Object::Bool(false).to_bool());
        assert!(!Object::new_int(1).to_bool());
    }

    #[test]
    fn indirect_reference_num_gen() {
        let r = Object::new_indirect(12, 5);
        assert_eq!(r.to_num(), 12);
        assert_eq!(r.to_gen(), 5);
        // Non-refs report 0.
        assert_eq!(Object::new_int(12).to_num(), 0);
        assert_eq!(Object::new_int(12).to_gen(), 0);
        // Out-of-range collapses to null (pdf_new_indirect warn-and-null).
        assert!(Object::new_indirect(-1, 0).is_null());
        assert!(Object::new_indirect(MAX_OBJECT_NUMBER as i64 + 1, 0).is_null());
        assert!(Object::new_indirect(1, MAX_GEN_NUMBER + 1).is_null());
    }

    #[test]
    fn array_build_and_query() {
        let mut a = Object::new_array();
        a.array_push(Object::new_int(1));
        a.array_push(Object::new_int(2));
        a.array_push(Object::new_name("End"));
        assert_eq!(a.array_len(), 3);
        assert_eq!(a.array_get(0), Some(&Object::new_int(1)));
        assert_eq!(a.array_get(2).unwrap().to_name(), b"End");
        assert!(a.array_get(3).is_none());
        // Non-array accessors are safe.
        assert_eq!(Object::Null.array_len(), 0);
        assert!(Object::Null.array_get(0).is_none());
    }

    #[test]
    fn dict_name_key_lookup() {
        let mut d = Object::new_dict();
        d.dict_put(b"Type".to_vec(), Object::new_name("Page"));
        d.dict_put(b"Count".to_vec(), Object::new_int(3));
        assert_eq!(d.dict_len(), 2);
        assert_eq!(d.dict_gets("Type").unwrap().to_name(), b"Page");
        assert_eq!(d.dict_gets("Count").unwrap().to_int(), 3);
        assert!(d.dict_gets("Missing").is_none());
        // Positional accessors preserve insertion order.
        assert_eq!(d.dict_get_key(0), Some(&b"Type"[..]));
        assert_eq!(d.dict_get_val(1).unwrap().to_int(), 3);
    }

    #[test]
    fn dict_put_replaces_existing_key() {
        // pdf_dict_put: last value wins, no duplicate entry.
        let mut d = Object::new_dict();
        d.dict_put("K", Object::new_int(1));
        d.dict_put("K", Object::new_int(2));
        assert_eq!(d.dict_len(), 1);
        assert_eq!(d.dict_gets("K").unwrap().to_int(), 2);
    }

    #[test]
    fn name_eq_matches_only_names() {
        let a = Object::new_name(b"Foo".to_vec());
        let b = Object::new_name(b"Foo".to_vec());
        let c = Object::new_name(b"Bar".to_vec());
        assert!(a.name_eq(&b));
        assert!(!a.name_eq(&c));
        // A string with the same bytes is not name-equal.
        assert!(!a.name_eq(&Object::new_string(b"Foo".to_vec())));
    }

    #[test]
    fn ensure_indirect_gates_on_kind() {
        assert!(Object::new_indirect(1, 0).ensure_indirect().is_ok());
        let err = Object::new_int(1).ensure_indirect().unwrap_err();
        assert_eq!(err.kind(), crate::mupdf::ErrorKind::Argument);
    }

    #[test]
    fn nested_structure_round_trips() {
        // << /Kids [ 4 0 R ] >>
        let mut kids = Object::new_array();
        kids.array_push(Object::new_indirect(4, 0));
        let mut d = Object::new_dict();
        d.dict_put("Kids", kids);
        let got = d.dict_gets("Kids").unwrap();
        assert_eq!(got.array_len(), 1);
        assert_eq!(got.array_get(0).unwrap().to_num(), 4);
    }
}

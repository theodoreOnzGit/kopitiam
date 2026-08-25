//! Ported from MuPDF `source/pdf/pdf-xref.c` (+ `include/mupdf/pdf/xref.h`) and
//! the page-tree walk of `source/pdf/pdf-page.c` (commit 19f1284, AGPL-3.0,
//! © Artifex Software, Inc.), translated to Rust for KOPITIAM (AGPL-3.0-only).
//! Close adaptation: the algorithms and numeric behaviour follow MuPDF; the code
//! is re-expressed in idiomatic Rust. See docs/ACKNOWLEDGEMENTS.md ("PDF &
//! document-extraction references").
//!
//! # Opening a PDF: xref, resolution, pages
//!
//! [`PdfDocument`] is the WAVE-3b document layer: it loads a PDF from bytes,
//! builds the cross-reference table, resolves indirect objects (`num gen R`),
//! decodes stream bodies, and walks the page tree into an ordered list of page
//! dicts. It ports the *text-extraction essentials* of `pdf-xref.c` -- a file
//! several thousand lines long -- and deliberately defers the rest.
//!
//! ## What is ported
//!
//! * **`startxref` + trailer** (`pdf_read_start_xref`, `pdf_read_xref_sections`):
//!   scan the file tail for `startxref`, follow the xref chain, keep the newest
//!   trailer (`/Root`, `/Size`, `/Prev`, hybrid `/XRefStm`).
//! * **Classic xref tables** (`xref` … `trailer`, `pdf_read_old_xref`): the
//!   20-byte fixed entries in `start len` subsections.
//! * **Xref streams** (`/Type /XRef`, `pdf_read_new_xref`): a compressed
//!   cross-reference stream, FlateDecode + predictor (WAVE 2), with the `/W`
//!   field widths and optional `/Index` subsections.
//! * **Object streams** (`/Type /ObjStm`, `pdf_load_obj_stm`): decompress and
//!   index the compressed objects so a `type 'o'` xref entry resolves.
//! * **Resolution** (`pdf_cache_object` / `pdf_resolve_indirect`): follow a
//!   reference through the xref (classic 'n' + compressed 'o'), with a caching
//!   layer and a recursion guard (MuPDF's `pdf_obj_marked` / the `RESOLVE`
//!   again-loop). Later xref sections never overwrite a populated entry
//!   (`if (!entry->type)`), so the newest definition wins.
//! * **Page tree** (`pdf_lookup_page_loc_imp` + `pdf_flatten_inheritable_page_items`):
//!   walk `/Root → /Pages → /Kids` recursively, flattening inherited
//!   `/Resources`, `/MediaBox`, `/CropBox`, `/Rotate` top-down into each leaf.
//!
//! ## What is deferred (a later wave)
//!
//! Repair (`pdf-repair.c`: a malformed xref throws here rather than triggering a
//! rebuild), encryption (`pdf-crypt.c`), linearization / progressive loading,
//! and incremental-update authoring (the `newobj`/journal path, and `/Prev`
//! handling beyond following the chain to merge previous sections). The write
//! path is out of scope entirely. MuPDF's mutable multi-section xref with
//! solidification, local/incremental xrefs, and shared refcounted `pdf_obj`s is
//! collapsed here into a single flat entry table plus an owned-`Object` cache
//! (`Clone` deep-copies, as in `object.rs`).

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use super::doc_stream::decode_stream;
use super::error::{Error, Result};
use super::lex::{lex, Token};
use super::object::{Object, MAX_OBJECT_NUMBER};
use super::parse::{parse_ind_obj, parse_stm_obj};
use super::stream::{Stream, Whence};

/// One cross-reference table entry, keyed by object number.
// MuPDF: pdf_xref_entry.type 'f'/'n'/'o' (xref.h:76).
#[derive(Clone, Debug)]
enum XrefEntry {
    /// `'f'` -- a free object (not in use).
    Free,
    /// `'n'` -- an uncompressed object at `offset` in the file. (The generation
    /// number is not tracked: the resolve path keys only on the object number,
    /// as MuPDF's `pdf_cache_object` verifies `rnum == num` but not the gen.)
    Uncompressed { offset: i64 },
    /// `'o'` -- a compressed object: entry `index` inside object stream
    /// `stm_num`.
    Compressed { stm_num: i32, index: i32 },
}

/// A resolved-and-cached object: the parsed [`Object`] plus, for a stream
/// object, where its raw body begins (`stm_ofs`, from [`parse_ind_obj`]).
#[derive(Clone)]
struct Cached {
    obj: Object,
    stm_ofs: Option<i64>,
}

/// A loaded PDF document: the raw bytes, the xref table, the trailer, the
/// resolved-object cache, and the ordered page-dict list.
///
/// This is the port's analogue of MuPDF's `pdf_document`, scoped to the text
/// path (see the module docs for what is deferred).
pub struct PdfDocument {
    /// The whole file (MuPDF's `doc->file`, here always an in-memory block).
    bytes: Vec<u8>,
    /// The cross-reference table, indexed by object number; `None` = an object
    /// number never populated by any xref section.
    entries: Vec<Option<XrefEntry>>,
    /// The newest trailer dict (`pdf_trailer`): `/Root`, `/Size`, …
    trailer: Object,
    /// The ordered, inheritance-flattened page dicts.
    pages: Vec<Object>,
    /// Resolved-object cache (MuPDF caches the parsed `pdf_obj` on the entry).
    cache: RefCell<HashMap<i32, Cached>>,
    /// Object numbers currently being resolved -- the recursion/cycle guard
    /// (MuPDF's `pdf_obj_marked` on object streams + the `RESOLVE` again-loop).
    resolving: RefCell<Vec<i32>>,
}

impl PdfDocument {
    // -----------------------------------------------------------------------
    // Opening
    // -----------------------------------------------------------------------

    // MuPDF: pdf_open_document_with_stream / pdf_init_document + pdf_load_xref
    // (pdf-xref.c:1775, document loading).
    /// Open a PDF from its raw bytes: build the xref table, read the trailer, and
    /// walk the page tree. Returns `FZ_ERROR_FORMAT`/`FZ_ERROR_SYNTAX` for a
    /// malformed cross-reference (repair is deferred to a later wave).
    pub fn open(bytes: Vec<u8>) -> Result<PdfDocument> {
        let (entries, trailer) = load_xref(&bytes)?;
        let mut doc = PdfDocument {
            bytes,
            entries,
            trailer,
            pages: Vec::new(),
            cache: RefCell::new(HashMap::new()),
            resolving: RefCell::new(Vec::new()),
        };
        doc.pages = doc.load_pages()?;
        Ok(doc)
    }

    /// The original PDF bytes this document was opened from. Used by
    /// `hayro_fallback.rs` to hand the same file to a second, independent
    /// parser (`hayro-syntax`) without the caller needing to keep its own
    /// copy around just for that.
    pub fn raw_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Open a PDF from a borrowed byte slice (copies into an owned buffer).
    pub fn open_slice(bytes: &[u8]) -> Result<PdfDocument> {
        PdfDocument::open(bytes.to_vec())
    }

    // -----------------------------------------------------------------------
    // Trailer / catalog
    // -----------------------------------------------------------------------

    // MuPDF: pdf_trailer (pdf-xref.c:177).
    /// The document trailer dict (`/Root`, `/Size`, …).
    pub fn trailer(&self) -> &Object {
        &self.trailer
    }

    /// The document catalog (`/Root`, resolved).
    pub fn catalog(&self) -> Result<Object> {
        let root = self.trailer.dict_gets("Root").cloned().unwrap_or(Object::Null);
        self.resolve(&root)
    }

    // -----------------------------------------------------------------------
    // Indirect resolution
    // -----------------------------------------------------------------------

    // MuPDF: pdf_resolve_indirect (pdf-xref.c:2693).
    /// Resolve `obj`: if it is an indirect reference (`num gen R`), return the
    /// object it names (following the xref, classic or compressed); otherwise
    /// return a clone of `obj`. An unresolvable reference yields [`Object::Null`]
    /// (as MuPDF's `pdf_resolve_indirect` returns `NULL`).
    pub fn resolve(&self, obj: &Object) -> Result<Object> {
        match obj {
            Object::Ref { num, .. } => self.get_object(*num),
            _ => Ok(obj.clone()),
        }
    }

    /// Resolve `dict`'s `key` value (looks the key up, then resolves it).
    /// Returns [`Object::Null`] if the key is absent.
    pub fn resolve_get(&self, dict: &Object, key: &str) -> Result<Object> {
        match dict.dict_gets(key) {
            Some(v) => self.resolve(v),
            None => Ok(Object::Null),
        }
    }

    // MuPDF: pdf_load_object / pdf_cache_object return (pdf-xref.c:2681, 2551).
    /// Fetch object `num` from the xref, resolving it (and caching the result).
    fn get_object(&self, num: i32) -> Result<Object> {
        if num <= 0 {
            return Ok(Object::Null);
        }
        self.ensure_cached(num)?;
        Ok(self
            .cache
            .borrow()
            .get(&num)
            .map(|c| c.obj.clone())
            .unwrap_or(Object::Null))
    }

    // MuPDF: pdf_cache_object (pdf-xref.c:2551) -- populate the entry's obj.
    /// Ensure object `num` is parsed into the cache. Idempotent.
    fn ensure_cached(&self, num: i32) -> Result<()> {
        if self.cache.borrow().contains_key(&num) {
            return Ok(());
        }
        // Recursion guard: MuPDF marks an object stream while loading it
        // (pdf_obj_marked → "recursive object stream lookup") and caps the
        // resolve chain. Here any object re-entered while still resolving is a
        // cycle.
        if self.resolving.borrow().contains(&num) {
            return Err(Error::format(format!(
                "recursive object lookup ({num} 0 R)"
            )));
        }
        self.resolving.borrow_mut().push(num);
        let result = self.load_object_uncached(num);
        self.resolving.borrow_mut().pop();

        let cached = result?;
        self.cache.borrow_mut().insert(num, cached);
        Ok(())
    }

    /// Parse object `num` from the file/object-stream, without touching the
    /// cache lookup (the caller guards against recursion).
    fn load_object_uncached(&self, num: i32) -> Result<Cached> {
        let entry = self.entries.get(num as usize).and_then(|e| e.as_ref());
        match entry {
            // Unknown or free object → null (pdf_resolve_indirect returns NULL;
            // MuPDF's pdf_cache_object throws "cannot find object", but on the
            // resolve path that surfaces as null).
            None | Some(XrefEntry::Free) => Ok(Cached {
                obj: Object::Null,
                stm_ofs: None,
            }),
            Some(XrefEntry::Uncompressed { offset }) => {
                let offset = *offset;
                let mut f = Stream::from_slice(&self.bytes);
                f.seek(offset, Whence::Set)?;
                let ind = parse_ind_obj(&mut f)?;
                // MuPDF: "found object (rnum 0 R) instead of (num 0 R)".
                if ind.num != num {
                    return Err(Error::format(format!(
                        "found object ({} 0 R) instead of ({num} 0 R)",
                        ind.num
                    )));
                }
                Ok(Cached {
                    obj: ind.object,
                    stm_ofs: ind.stream.map(|s| s.start),
                })
            }
            Some(XrefEntry::Compressed { stm_num, index }) => {
                self.load_obj_stm(*stm_num, *index, num)
            }
        }
    }

    // MuPDF: pdf_load_obj_stm (pdf-xref.c:2100).
    /// Decompress object stream `stm_num`, index its `/N` objects, cache them
    /// all, and return the one numbered `target` (at compressed slot `index`).
    fn load_obj_stm(&self, stm_num: i32, _index: i32, target: i32) -> Result<Cached> {
        let objstm = self.get_object(stm_num)?;
        if !objstm.is_dict() {
            return Err(Error::format(format!("corrupt object stream {stm_num}")));
        }

        let count = self.resolve_get(&objstm, "N")?.to_int();
        let first = self.resolve_get(&objstm, "First")?.to_int();
        if count < 0 || count > MAX_OBJECT_NUMBER as i64 {
            return Err(Error::format(
                "number of objects in object stream out of range",
            ));
        }
        let count = count as usize;

        let raw = self.open_stream_num(stm_num)?;

        // The stream begins with `count` pairs of `objnum offset` integers.
        let mut hdr = Stream::from_slice(&raw);
        let mut nums = Vec::with_capacity(count);
        let mut offs = Vec::with_capacity(count);
        for _ in 0..count {
            let onum = match lex(&mut hdr)? {
                Token::Int(i) => i,
                _ => {
                    return Err(Error::format(format!(
                        "corrupt object stream ({stm_num} 0 R)"
                    )))
                }
            };
            let ooff = match lex(&mut hdr)? {
                Token::Int(i) => i,
                _ => {
                    return Err(Error::format(format!(
                        "corrupt object stream ({stm_num} 0 R)"
                    )))
                }
            };
            nums.push(onum as i32);
            offs.push(ooff);
        }

        // Parse each object at `first + offset` and cache it (an existing cached
        // definition wins, mirroring MuPDF keeping the original entry->obj).
        let mut target_cached: Option<Cached> = None;
        for i in 0..count {
            let mut s = Stream::from_slice(&raw);
            s.seek(first + offs[i], Whence::Set)?;
            let obj = parse_stm_obj(&mut s)?;
            let onum = nums[i];
            let cached = Cached {
                obj,
                stm_ofs: None,
            };
            if onum == target {
                target_cached = Some(cached.clone());
            }
            self.cache
                .borrow_mut()
                .entry(onum)
                .or_insert(cached);
        }

        Ok(target_cached.unwrap_or(Cached {
            obj: Object::Null,
            stm_ofs: None,
        }))
    }

    // -----------------------------------------------------------------------
    // Stream decoding
    // -----------------------------------------------------------------------

    // MuPDF: pdf_open_stream (pdf-stream.c:890) reduced to a decoded byte vector.
    /// Decode the stream body of `obj` (an indirect reference to a stream) and
    /// return the decoded bytes.
    pub fn open_stream(&self, obj: &Object) -> Result<Vec<u8>> {
        match obj {
            Object::Ref { num, .. } => self.open_stream_num(*num),
            _ => Err(Error::format("object is not an indirect stream reference")),
        }
    }

    // MuPDF: pdf_open_stream_number → pdf_open_filter → fz_read_all
    // (pdf-stream.c:526, 393).
    /// Decode object `num`'s stream body: resolve `/Length` (may be indirect),
    /// slice the raw bytes, and apply the `/Filter` (+ `/DecodeParms`) chain.
    pub fn open_stream_num(&self, num: i32) -> Result<Vec<u8>> {
        let (raw, filter, parms) = self.stream_raw_num(num)?;
        decode_stream(raw, &filter, &parms)
    }

    /// Return object `num`'s **undecoded** stream body plus its resolved
    /// `/Filter` and `/DecodeParms`. Unlike [`open_stream_num`] this does not run
    /// the filter chain, so a caller that owns an image codec (DCTDecode / …,
    /// which [`decode_stream`] refuses) can decode the bytes itself while still
    /// applying any *leading* non-image filters. The returned filter/parms follow
    /// MuPDF's `/F` and `/DP` abbreviation fallbacks.
    // MuPDF: pdf_open_raw_stream_number (pdf-stream.c) -- the pre-filter slice.
    pub fn stream_raw(&self, obj: &Object) -> Result<(Vec<u8>, Object, Object)> {
        match obj {
            Object::Ref { num, .. } => self.stream_raw_num(*num),
            _ => Err(Error::format("object is not an indirect stream reference")),
        }
    }

    /// The object-number form of [`stream_raw`].
    pub fn stream_raw_num(&self, num: i32) -> Result<(Vec<u8>, Object, Object)> {
        self.ensure_cached(num)?;
        let (dict, stm_ofs) = {
            let cache = self.cache.borrow();
            let c = cache
                .get(&num)
                .ok_or_else(|| Error::format(format!("object is not a stream ({num} 0 R)")))?;
            (c.obj.clone(), c.stm_ofs)
        };
        let stm_ofs =
            stm_ofs.ok_or_else(|| Error::format(format!("object is not a stream ({num} 0 R)")))?;

        // MuPDF: pdf_stream_length -- clamp a bad /Length to 0.
        let len_obj = self.resolve_get(&dict, "Length")?;
        let mut length = len_obj.to_int();
        if length < 0 || length > self.bytes.len() as i64 {
            length = 0;
        }

        let start = stm_ofs as usize;
        let end = start.saturating_add(length as usize).min(self.bytes.len());
        let raw = if start <= end && start <= self.bytes.len() {
            self.bytes[start..end].to_vec()
        } else {
            Vec::new()
        };

        // /Filter and /DecodeParms (abbreviations /F and /DP), resolved.
        let filter = self.resolve_get_first(&dict, "Filter", "F")?;
        let parms = self.resolve_get_first(&dict, "DecodeParms", "DP")?;
        Ok((raw, filter, parms))
    }

    /// Resolve `dict[primary]`, falling back to `dict[abbrev]` (MuPDF's
    /// `pdf_dict_geta`).
    fn resolve_get_first(&self, dict: &Object, primary: &str, abbrev: &str) -> Result<Object> {
        if dict.dict_gets(primary).is_some() {
            self.resolve_get(dict, primary)
        } else if dict.dict_gets(abbrev).is_some() {
            self.resolve_get(dict, abbrev)
        } else {
            Ok(Object::Null)
        }
    }

    // -----------------------------------------------------------------------
    // Page tree
    // -----------------------------------------------------------------------

    /// The number of pages (`page_count`).
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// The `i`-th page dict (0-based), with inherited attributes flattened in.
    pub fn page(&self, i: usize) -> Result<&Object> {
        self.pages
            .get(i)
            .ok_or_else(|| Error::argument(format!("page {i} out of range 0..{}", self.pages.len())))
    }

    // MuPDF: pdf_lookup_page_loc + pdf_load_page_tree_imp (pdf-page.c:265, 72).
    /// Walk `/Root → /Pages → /Kids` into the ordered list of flattened page
    /// dicts.
    fn load_pages(&self) -> Result<Vec<Object>> {
        let root = self.catalog()?;
        let pages_root = self.resolve_get(&root, "Pages")?;
        if !pages_root.is_dict() {
            return Err(Error::format("cannot find page tree"));
        }
        let mut out = Vec::new();
        let mut visited = HashSet::new();
        self.walk_pages(&pages_root, &Inherited::default(), &mut visited, &mut out)?;
        Ok(out)
    }

    /// Recursive page-tree walk with top-down inheritance and a cycle guard.
    fn walk_pages(
        &self,
        node: &Object,
        inherited: &Inherited,
        visited: &mut HashSet<i32>,
        out: &mut Vec<Object>,
    ) -> Result<()> {
        // Refresh the inherited set from this node's own entries.
        let here = inherited.updated_from(node);

        let type_name = node.dict_gets("Type").map(|o| o.to_name());
        let has_kids = node.dict_gets("Kids").is_some();
        // MuPDF's classification (pdf-page.c:213/228): a /Type decides it; else a
        // node with /Kids and no /MediaBox is an internal Pages node.
        let is_internal = match type_name {
            Some(b"Pages") => true,
            Some(b"Page") => false,
            _ => has_kids && node.dict_gets("MediaBox").is_none(),
        };

        if is_internal {
            let kids = self.resolve_get(node, "Kids")?;
            if kids.array_len() == 0 {
                return Err(Error::format("malformed page tree"));
            }
            for i in 0..kids.array_len() {
                let kid_ref = kids.array_get(i).unwrap();
                if let Object::Ref { num, .. } = kid_ref
                    && !visited.insert(*num)
                {
                    return Err(Error::format("cycle in page tree"));
                }
                let kid = self.resolve(kid_ref)?;
                if !kid.is_dict() {
                    return Err(Error::format("non-dict node in page tree"));
                }
                self.walk_pages(&kid, &here, visited, out)?;
            }
        } else {
            // A leaf page: flatten the inherited attributes it lacks.
            out.push(here.flatten_into(node));
        }
        Ok(())
    }
}

/// The four inheritable page attributes (`pdf_flatten_inheritable_page_items`),
/// carried down the page tree.
// MuPDF: MediaBox / CropBox / Rotate / Resources (pdf-page.c:428).
#[derive(Clone, Default)]
struct Inherited {
    media_box: Option<Object>,
    crop_box: Option<Object>,
    rotate: Option<Object>,
    resources: Option<Object>,
}

impl Inherited {
    /// Override any attribute this `node` defines directly.
    fn updated_from(&self, node: &Object) -> Inherited {
        let mut next = self.clone();
        if let Some(v) = node.dict_gets("MediaBox") {
            next.media_box = Some(v.clone());
        }
        if let Some(v) = node.dict_gets("CropBox") {
            next.crop_box = Some(v.clone());
        }
        if let Some(v) = node.dict_gets("Rotate") {
            next.rotate = Some(v.clone());
        }
        if let Some(v) = node.dict_gets("Resources") {
            next.resources = Some(v.clone());
        }
        next
    }

    /// Clone `page` and fill in any inheritable attribute it does not define.
    fn flatten_into(&self, page: &Object) -> Object {
        let mut out = page.clone();
        for (key, val) in [
            ("MediaBox", &self.media_box),
            ("CropBox", &self.crop_box),
            ("Rotate", &self.rotate),
            ("Resources", &self.resources),
        ] {
            if out.dict_gets(key).is_none()
                && let Some(v) = val
            {
                out.dict_put(key, v.clone());
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Cross-reference loading (free functions over the raw bytes)
// ---------------------------------------------------------------------------

// MuPDF: pdf_read_start_xref (pdf-xref.c:1010).
/// Find the `startxref` offset by scanning the file's tail (the last 1024
/// bytes, as MuPDF does).
fn read_start_xref(bytes: &[u8]) -> Result<i64> {
    const WINDOW: usize = 1024;
    let n = bytes.len();
    let start = n.saturating_sub(WINDOW);
    let buf = &bytes[start..];
    if buf.len() < 9 {
        return Err(Error::format("cannot find startxref"));
    }
    // Find the last "startxref".
    let mut i = buf.len() - 9;
    loop {
        if &buf[i..i + 9] == b"startxref" {
            let mut j = i + 9;
            while j < buf.len() && buf[j].is_ascii_whitespace() {
                j += 1;
            }
            let mut ofs: i64 = 0;
            let mut saw = false;
            while j < buf.len() && buf[j].is_ascii_digit() {
                ofs = ofs
                    .checked_mul(10)
                    .and_then(|v| v.checked_add((buf[j] - b'0') as i64))
                    .ok_or_else(|| Error::limit("startxref too large"))?;
                j += 1;
                saw = true;
            }
            if saw && ofs != 0 {
                return Ok(ofs);
            }
        }
        if i == 0 {
            break;
        }
        i -= 1;
    }
    Err(Error::format("cannot find startxref"))
}

// MuPDF: pdf_load_xref / pdf_read_xref_sections (pdf-xref.c:1775, 1636).
/// Build the xref table and pick out the newest trailer, following the
/// `/Prev` chain (and hybrid `/XRefStm`).
fn load_xref(bytes: &[u8]) -> Result<(Vec<Option<XrefEntry>>, Object)> {
    let mut entries: Vec<Option<XrefEntry>> = Vec::new();
    let mut top_trailer: Option<Object> = None;
    let mut seen: Vec<i64> = Vec::new();

    let mut ofs = read_start_xref(bytes)?;
    while ofs != 0 {
        // Cycle guard on section offsets (pdf-xref.c:1656 "xref section recursion").
        if seen.contains(&ofs) {
            break;
        }
        seen.push(ofs);

        let trailer = read_xref_section(bytes, ofs, &mut entries)?;

        // Hybrid-reference files: read the cross-reference stream named by
        // /XRefStm, but ignore its trailer and do not follow its /Prev
        // (pdf-xref.c:1605).
        if let Some(xstm) = trailer.dict_gets("XRefStm").map(|o| o.to_int())
            && xstm > 0
            && !seen.contains(&xstm)
        {
            seen.push(xstm);
            let _ = read_xref_section(bytes, xstm, &mut entries)?;
        }

        // The first (newest) trailer wins for /Root, /Size, ….
        let prev = trailer.dict_gets("Prev").map(|o| o.to_int());
        if top_trailer.is_none() {
            top_trailer = Some(trailer);
        }

        match prev {
            Some(p) if p > 0 => ofs = p,
            _ => break,
        }
    }

    let trailer = top_trailer.ok_or_else(|| Error::format("no trailer found"))?;
    Ok((entries, trailer))
}

// MuPDF: pdf_read_xref (pdf-xref.c:1569) -- dispatch classic vs. stream.
/// Read one xref section at `ofs`, filling `entries`, and return its trailer.
fn read_xref_section(
    bytes: &[u8],
    ofs: i64,
    entries: &mut Vec<Option<XrefEntry>>,
) -> Result<Object> {
    if ofs < 0 || ofs as usize >= bytes.len() {
        return Err(Error::format("invalid xref offset"));
    }
    let mut f = Stream::from_slice(bytes);
    f.seek(ofs, Whence::Set)?;
    skip_ws(&mut f)?;
    match f.peek_byte()? {
        Some(b'x') => read_old_xref(&mut f, entries),
        Some(c) if c.is_ascii_digit() => read_new_xref(bytes, &mut f, entries),
        _ => Err(Error::format("cannot recognize xref format")),
    }
}

// MuPDF: pdf_read_old_xref (pdf-xref.c:1306).
/// Read a classic `xref` … `trailer` section.
fn read_old_xref(f: &mut Stream, entries: &mut Vec<Option<XrefEntry>>) -> Result<Object> {
    skip_ws(f)?;
    if !skip_literal(f, b"xref")? {
        return Err(Error::format("cannot find xref marker"));
    }
    skip_ws(f)?;

    loop {
        match f.peek_byte()? {
            Some(c) if c.is_ascii_digit() => {}
            _ => break,
        }
        // Subsection header: "start len".
        let line = read_line(f)?;
        let mut parts = line.split(|b| *b == b' ').filter(|s| !s.is_empty());
        let start = parse_ascii_int(parts.next())?;
        let len = parse_ascii_int(parts.next())?;
        if start < 0 || len < 0 || start + len > MAX_OBJECT_NUMBER as i64 + 1 {
            return Err(Error::format("xref subsection out of range"));
        }
        let (start, len) = (start as usize, len as usize);
        ensure_len(entries, start + len);

        // Each entry is a fixed 20-byte record: "nnnnnnnnnn ggggg t\r\n".
        for i in 0..len {
            let mut rec = [0u8; 20];
            let got = f.read(&mut rec)?;
            if got < 19 {
                return Err(Error::format("unexpected EOF in xref table"));
            }
            let num = start + i;
            if entries[num].is_some() {
                continue; // newest section already populated this object
            }
            let text = &rec[..got];
            let mut fields = text.split(|b| b.is_ascii_whitespace()).filter(|s| !s.is_empty());
            let offset = parse_ascii_int(fields.next())?;
            let generation = parse_ascii_int(fields.next())?;
            let kind = fields
                .next()
                .and_then(|s| s.first().copied())
                .ok_or_else(|| Error::format("unexpected xref type"))?;
            let _ = generation;
            entries[num] = Some(match kind {
                b'n' => XrefEntry::Uncompressed { offset },
                b'f' => XrefEntry::Free,
                other => {
                    return Err(Error::format(format!(
                        "unexpected xref type: 0x{other:x}"
                    )))
                }
            });
        }
    }

    // trailer keyword + dictionary.
    match lex(f)? {
        Token::Trailer => {}
        _ => return Err(Error::format("expected trailer marker")),
    }
    match lex(f)? {
        Token::OpenDict => {}
        _ => return Err(Error::format("expected trailer dictionary")),
    }
    super::parse::parse_dict(f)
}

// MuPDF: pdf_read_new_xref + pdf_read_new_xref_section (pdf-xref.c:1456, 1416).
/// Read a `/Type /XRef` cross-reference stream.
fn read_new_xref(
    bytes: &[u8],
    f: &mut Stream,
    entries: &mut Vec<Option<XrefEntry>>,
) -> Result<Object> {
    let ind = parse_ind_obj(f)?;
    if ind.num == 0 {
        return Err(Error::format("trailer object number cannot be 0"));
    }
    let dict = ind.object;
    let stm_ofs = ind
        .stream
        .ok_or_else(|| Error::format("xref stream has no stream body"))?
        .start;

    // /W field widths.
    let w = dict
        .dict_gets("W")
        .ok_or_else(|| Error::format("xref stream missing W entry"))?;
    if w.array_len() < 3 {
        return Err(Error::format("xref stream W entry too short"));
    }
    let w0 = clamp_w(w.array_get(0).unwrap().to_int());
    let w1 = clamp_w(w.array_get(1).unwrap().to_int());
    let w2 = clamp_w(w.array_get(2).unwrap().to_int());

    let size = dict
        .dict_gets("Size")
        .map(|o| o.to_int())
        .ok_or_else(|| Error::format("xref stream missing Size entry"))?;

    // Decode the stream body. /Length, /Filter and /DecodeParms are direct in an
    // xref stream (they must be, as the xref is not yet usable to resolve them).
    let length = dict.dict_gets("Length").map(|o| o.to_int()).unwrap_or(0);
    let start = stm_ofs as usize;
    let end = start
        .saturating_add(length.max(0) as usize)
        .min(bytes.len());
    let raw = bytes.get(start..end).unwrap_or(&[]).to_vec();
    let filter = dict.dict_gets("Filter").cloned().unwrap_or(Object::Null);
    let parms = dict
        .dict_gets("DecodeParms")
        .cloned()
        .unwrap_or(Object::Null);
    let decoded = decode_stream(raw, &filter, &parms)?;

    // /Index (default [0 Size]) gives the subsections.
    let index: Vec<(i64, i64)> = match dict.dict_gets("Index") {
        Some(arr) if arr.is_array() => {
            let mut v = Vec::new();
            let mut i = 0;
            while i + 1 < arr.array_len() {
                let i0 = arr.array_get(i).unwrap().to_int();
                let i1 = arr.array_get(i + 1).unwrap().to_int();
                v.push((i0, i1));
                i += 2;
            }
            v
        }
        _ => vec![(0, size)],
    };

    let row = (w0 + w1 + w2) as usize;
    if row == 0 {
        return Err(Error::format("xref stream has zero-width fields"));
    }
    let mut pos = 0usize;
    for (i0, count) in index {
        if i0 < 0 || count < 0 || i0 + count > MAX_OBJECT_NUMBER as i64 + 1 {
            return Err(Error::format("xref stream subsection out of range"));
        }
        let base = i0 as usize;
        ensure_len(entries, base + count as usize);
        for j in 0..count as usize {
            if pos + row > decoded.len() {
                return Err(Error::format("truncated xref stream"));
            }
            let a = read_be(&decoded[pos..], w0);
            let b = read_be(&decoded[pos + w0 as usize..], w1);
            let c = read_be(&decoded[pos + (w0 + w1) as usize..], w2);
            pos += row;

            let num = base + j;
            if entries[num].is_some() {
                continue;
            }
            // Default type is 1 when the type field width is 0 (pdf-xref.c:1443).
            let kind = if w0 == 0 { 1 } else { a };
            entries[num] = Some(match kind {
                0 => XrefEntry::Free,
                1 => {
                    let _ = c; // generation, not tracked
                    XrefEntry::Uncompressed { offset: b }
                }
                2 => XrefEntry::Compressed {
                    stm_num: b as i32,
                    index: c as i32,
                },
                // Unknown type: leave unpopulated (MuPDF stores type 0 → free).
                _ => XrefEntry::Free,
            });
        }
    }

    // Note: MuPDF also records the xref stream's own object entry (pdf-xref.c:1545)
    // pointing at the section offset. The `/W` table normally already lists that
    // object as type 1, and nothing on the text path needs to re-resolve the xref
    // stream dict (we return it here directly), so that bookkeeping is omitted.
    let _ = ind.generation;
    Ok(dict)
}

// ---------------------------------------------------------------------------
// Small byte-level helpers (fz_skip_space, fz_read_line, big-endian fields)
// ---------------------------------------------------------------------------

/// Grow `entries` with `None` placeholders up to length `n`.
fn ensure_len(entries: &mut Vec<Option<XrefEntry>>, n: usize) {
    if entries.len() < n {
        entries.resize(n, None);
    }
}

/// Clamp a `/W` field width into `[0, 8]` (pdf-xref.c:1523).
fn clamp_w(v: i64) -> i32 {
    v.clamp(0, 8) as i32
}

/// Read a `width`-byte big-endian integer from the front of `buf`.
fn read_be(buf: &[u8], width: i32) -> i64 {
    let mut v: i64 = 0;
    for &byte in buf.iter().take(width as usize) {
        v = (v << 8) | byte as i64;
    }
    v
}

// MuPDF: fz_skip_space (pdf-xref.c:1051).
/// Consume whitespace (any byte ≤ 32) at the current position.
fn skip_ws(f: &mut Stream) -> Result<()> {
    loop {
        match f.peek_byte()? {
            Some(c) if c <= 32 => {
                f.read_byte()?;
            }
            _ => return Ok(()),
        }
    }
}

// MuPDF: fz_skip_string (pdf-xref.c:1063) -- consume `lit` if it matches next.
/// If the next bytes equal `lit`, consume them and return `true`; otherwise
/// leave the stream positioned at the first mismatch and return `false`.
fn skip_literal(f: &mut Stream, lit: &[u8]) -> Result<bool> {
    for &want in lit {
        match f.peek_byte()? {
            Some(c) if c == want => {
                f.read_byte()?;
            }
            _ => return Ok(false),
        }
    }
    Ok(true)
}

// MuPDF: fz_read_line -- read up to (and consuming) the end-of-line.
/// Read a line, returning its bytes without the trailing CR/LF.
fn read_line(f: &mut Stream) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        match f.read_byte()? {
            None => break,
            Some(b'\n') => break,
            Some(b'\r') => {
                if f.peek_byte()? == Some(b'\n') {
                    f.read_byte()?;
                }
                break;
            }
            Some(c) => out.push(c),
        }
    }
    Ok(out)
}

/// Parse an ASCII integer field (from `fz_atoi`-style whitespace-split parts).
fn parse_ascii_int(field: Option<&[u8]>) -> Result<i64> {
    let field = field.ok_or_else(|| Error::format("missing xref field"))?;
    let s = std::str::from_utf8(field).map_err(|_| Error::format("non-ascii xref field"))?;
    s.trim()
        .parse::<i64>()
        .map_err(|_| Error::format(format!("malformed xref integer: {s:?}")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use lopdf::{dictionary, Document, Object as LObject, Stream as LStream};

    /// Build a minimal one-page PDF with a classic xref table via lopdf.
    fn minimal_classic_pdf() -> Vec<u8> {
        let mut doc = Document::with_version("1.5");
        let pages_id = doc.new_object_id();
        let page_id = doc.new_object_id();
        let content_id = doc.add_object(LStream::new(
            dictionary! {},
            b"BT /F1 12 Tf (Hello World) Tj ET".to_vec(),
        ));
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
        let mut buf = Vec::new();
        doc.save_to(&mut buf).unwrap();
        buf
    }

    #[test]
    fn open_classic_xref_and_walk_pages() {
        let pdf = minimal_classic_pdf();
        let doc = PdfDocument::open(pdf).unwrap();

        // Root resolves to a Catalog.
        let root = doc.catalog().unwrap();
        assert_eq!(root.dict_gets("Type").unwrap().to_name(), b"Catalog");

        // One page, and it carries a MediaBox and Contents.
        assert_eq!(doc.page_count(), 1);
        let page = doc.page(0).unwrap();
        assert_eq!(page.dict_gets("Type").unwrap().to_name(), b"Page");
        assert!(page.dict_gets("MediaBox").is_some());
        assert!(page.dict_gets("Contents").is_some());
    }

    #[test]
    fn resolve_indirect_object() {
        let pdf = minimal_classic_pdf();
        let doc = PdfDocument::open(pdf).unwrap();
        // /Root in the trailer is an indirect ref; resolve it.
        let root_ref = doc.trailer().dict_gets("Root").unwrap();
        assert!(root_ref.is_indirect());
        let root = doc.resolve(root_ref).unwrap();
        assert!(root.is_dict());
        // A non-indirect object resolves to itself.
        let plain = Object::new_int(42);
        assert_eq!(doc.resolve(&plain).unwrap(), plain);
    }

    #[test]
    fn decode_content_stream_flate() {
        let pdf = minimal_classic_pdf();
        let doc = PdfDocument::open(pdf).unwrap();
        let page = doc.page(0).unwrap().clone();
        let contents_ref = page.dict_gets("Contents").unwrap();
        let bytes = doc.open_stream(contents_ref).unwrap();
        assert_eq!(bytes, b"BT /F1 12 Tf (Hello World) Tj ET");
    }

    #[test]
    fn indirect_length_resolves() {
        // Hand-build a PDF whose single content stream has an indirect /Length.
        // Objects: 1 catalog, 2 pages, 3 page, 4 content stream, 5 length int.
        let body = b"BT (indirect length) Tj ET";
        let mut pdf: Vec<u8> = Vec::new();
        let mut offsets: Vec<usize> = vec![0; 6];

        pdf.extend_from_slice(b"%PDF-1.5\n");
        offsets[1] = pdf.len();
        pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
        offsets[2] = pdf.len();
        pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
        offsets[3] = pdf.len();
        pdf.extend_from_slice(
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R >>\nendobj\n",
        );
        offsets[4] = pdf.len();
        pdf.extend_from_slice(b"4 0 obj\n<< /Length 5 0 R >>\nstream\n");
        pdf.extend_from_slice(body);
        pdf.extend_from_slice(b"\nendstream\nendobj\n");
        offsets[5] = pdf.len();
        pdf.extend_from_slice(format!("5 0 obj\n{}\nendobj\n", body.len()).as_bytes());

        let xref_ofs = pdf.len();
        pdf.extend_from_slice(b"xref\n0 6\n");
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        for off in offsets.iter().skip(1) {
            pdf.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
        }
        pdf.extend_from_slice(b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n");
        pdf.extend_from_slice(format!("{xref_ofs}\n").as_bytes());
        pdf.extend_from_slice(b"%%EOF\n");

        let doc = PdfDocument::open(pdf).unwrap();
        assert_eq!(doc.page_count(), 1);
        let page = doc.page(0).unwrap().clone();
        let out = doc.open_stream(page.dict_gets("Contents").unwrap()).unwrap();
        assert_eq!(out, body);
    }

    #[test]
    fn mediabox_inherited_from_pages() {
        // The page omits /MediaBox; it is defined on the /Pages parent.
        let mut doc = Document::with_version("1.5");
        let pages_id = doc.new_object_id();
        let page_id = doc.new_object_id();
        let page = dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            // no MediaBox here
        };
        doc.objects.insert(page_id, LObject::Dictionary(page));
        let pages = dictionary! {
            "Type" => "Pages",
            "Kids" => vec![page_id.into()],
            "Count" => 1,
            "MediaBox" => vec![0.into(), 0.into(), 300.into(), 400.into()],
        };
        doc.objects.insert(pages_id, LObject::Dictionary(pages));
        let catalog_id = doc.add_object(dictionary! {
            "Type" => "Catalog",
            "Pages" => pages_id,
        });
        doc.trailer.set("Root", catalog_id);
        let mut buf = Vec::new();
        doc.save_to(&mut buf).unwrap();

        let doc = PdfDocument::open(buf).unwrap();
        let page = doc.page(0).unwrap();
        let mb = page.dict_gets("MediaBox").expect("inherited MediaBox");
        assert!(mb.is_array());
        assert_eq!(mb.array_get(2).unwrap().to_int(), 300);
        assert_eq!(mb.array_get(3).unwrap().to_int(), 400);
    }

    /// True if `needle` occurs anywhere in `hay`.
    fn contains(hay: &[u8], needle: &[u8]) -> bool {
        hay.windows(needle.len()).any(|w| w == needle)
    }

    #[test]
    fn open_xref_stream_pdf() {
        // lopdf can emit a compressed xref stream + object streams.
        let pdf = compressed_pdf();
        // Guard the fixture: it must really be a cross-reference stream (not a
        // classic table), else this test would silently exercise the wrong path.
        assert!(contains(&pdf, b"/XRef"), "fixture is not an xref stream");
        let doc = PdfDocument::open(pdf).unwrap();
        assert_eq!(doc.page_count(), 1);
        let root = doc.catalog().unwrap();
        assert_eq!(root.dict_gets("Type").unwrap().to_name(), b"Catalog");
        // Resolve an object reached through the xref stream / object stream.
        let page = doc.page(0).unwrap();
        assert_eq!(page.dict_gets("Type").unwrap().to_name(), b"Page");
    }

    #[test]
    fn objstm_compressed_object_resolves() {
        let pdf = objstm_pdf();
        assert!(contains(&pdf, b"/ObjStm"), "fixture has no object stream");
        assert!(contains(&pdf, b"/XRef"), "fixture is not an xref stream");
        let doc = PdfDocument::open(pdf).unwrap();

        // Objects 1/2/3 live inside object stream 4; resolving them exercises the
        // compressed-object ('o') path through pdf_load_obj_stm.
        let root = doc.catalog().unwrap();
        assert_eq!(root.dict_gets("Type").unwrap().to_name(), b"Catalog");
        let pages = doc.resolve_get(&root, "Pages").unwrap();
        assert_eq!(pages.dict_gets("Type").unwrap().to_name(), b"Pages");
        assert_eq!(doc.page_count(), 1);
        let page = doc.page(0).unwrap();
        assert_eq!(page.dict_gets("Type").unwrap().to_name(), b"Page");
        // Inherited nothing here; the page defines its own MediaBox.
        assert_eq!(
            page.dict_gets("MediaBox").unwrap().array_get(2).unwrap().to_int(),
            200
        );
    }

    /// Hand-build a PDF whose objects 1/2/3 are packed into an object stream
    /// (object 4), indexed by a cross-reference stream (object 5). Both streams
    /// are left unfiltered to keep the fixture legible.
    fn objstm_pdf() -> Vec<u8> {
        let o1 = "<</Type/Catalog/Pages 2 0 R>>";
        let o2 = "<</Type/Pages/Kids[3 0 R]/Count 1>>";
        let o3 = "<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>";

        // Object-stream body: "objnum offset" header, then the objects.
        let off1 = 0;
        let off2 = o1.len() + 1;
        let off3 = o1.len() + 1 + o2.len() + 1;
        let header = format!("1 {off1} 2 {off2} 3 {off3} ");
        let first = header.len();
        let objstm_body = format!("{header}{o1} {o2} {o3}");
        let objstm_len = objstm_body.len();

        let mut pdf: Vec<u8> = Vec::new();
        pdf.extend_from_slice(b"%PDF-1.5\n");

        // Object 4: the object stream.
        let x = pdf.len() as u64; // offset of "4 0 obj"
        pdf.extend_from_slice(
            format!(
                "4 0 obj\n<< /Type /ObjStm /N 3 /First {first} /Length {objstm_len} >>\nstream\n"
            )
            .as_bytes(),
        );
        pdf.extend_from_slice(objstm_body.as_bytes());
        pdf.extend_from_slice(b"\nendstream\nendobj\n");

        // Object 5: the cross-reference stream, W = [1 2 1].
        let y = pdf.len() as u64; // offset of "5 0 obj"
        let mut xbody: Vec<u8> = Vec::new();
        let row = |t: u8, f2: u64, f3: u8, out: &mut Vec<u8>| {
            out.push(t);
            out.push((f2 >> 8) as u8);
            out.push(f2 as u8);
            out.push(f3);
        };
        row(0, 0, 0, &mut xbody); // obj 0: free
        row(2, 4, 0, &mut xbody); // obj 1: compressed in objstm 4, index 0
        row(2, 4, 1, &mut xbody); // obj 2: index 1
        row(2, 4, 2, &mut xbody); // obj 3: index 2
        row(1, x, 0, &mut xbody); // obj 4: uncompressed at x
        row(1, y, 0, &mut xbody); // obj 5: uncompressed at y
        let xref_len = xbody.len();

        pdf.extend_from_slice(
            format!(
                "5 0 obj\n<< /Type /XRef /Size 6 /Root 1 0 R /W [1 2 1] /Length {xref_len} >>\nstream\n"
            )
            .as_bytes(),
        );
        pdf.extend_from_slice(&xbody);
        pdf.extend_from_slice(b"\nendstream\nendobj\n");

        pdf.extend_from_slice(format!("startxref\n{y}\n%%EOF\n").as_bytes());
        pdf
    }

    #[test]
    fn malformed_xref_errors() {
        // Valid-ish header but no startxref / no xref table at all.
        let junk = b"%PDF-1.5\nthis is not a real pdf, no xref here\n".to_vec();
        assert!(PdfDocument::open(junk).is_err());

        // startxref present but points nowhere sane.
        let mut bad = b"%PDF-1.5\n1 0 obj<< >>endobj\n".to_vec();
        bad.extend_from_slice(b"startxref\n999999\n%%EOF\n");
        assert!(PdfDocument::open(bad).is_err());
    }

    /// Build a one-page PDF and re-save it compressed (xref stream + ObjStm).
    fn compressed_pdf() -> Vec<u8> {
        let mut doc = Document::with_version("1.5");
        let pages_id = doc.new_object_id();
        let page_id = doc.new_object_id();
        let content_id = doc.add_object(LStream::new(
            dictionary! {},
            b"BT (compressed) Tj ET".to_vec(),
        ));
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
        doc.compress();
        let mut buf = Vec::new();
        doc.save_to(&mut buf).unwrap();
        buf
    }
}

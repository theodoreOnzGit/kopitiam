//! Ported from MuPDF `source/pdf/pdf-repair.c` (commit 19f1284, AGPL-3.0,
//! © Artifex Software, Inc.), translated to Rust for KOPITIAM (AGPL-3.0-only).
//! Close adaptation: the algorithms and numeric behaviour follow MuPDF; the code
//! is re-expressed in idiomatic Rust. See docs/ACKNOWLEDGEMENTS.md ("PDF &
//! document-extraction references").
//!
//! # Repair: rebuilding a broken cross-reference by scanning the file
//!
//! A PDF's xref says where every object lives. When it lies -- offsets
//! shifted by an editor's CRLF damage, no `startxref` at all, a file cut off
//! before its trailer -- MuPDF does not give up. It walks the *whole file*
//! with the lexer, notes every `N G obj` it meets (the **last** definition of
//! a number wins, same as an incremental update would), measures each stream
//! by finding its `endstream`, pulls `/Encrypt`, `/ID`, `/Info` and `/Root`
//! out of any trailer (or xref-stream) dictionary it passes, and builds a new
//! single-section xref from that. Then it adds the objects packed inside
//! every object stream it found, and if no trailer named a usable `/Root`,
//! it hunts for a `/Type /Catalog` from the end of the file backwards.
//!
//! Where it runs, same as MuPDF:
//!
//! * **At open** (`pdf_init_document`, pdf-xref.c:1961): the xref would not
//!   load → "trying to repair broken xref" → [`PdfDocument::repair_at_open`].
//! * **Mid-read** (`pdf_cache_object`, pdf-xref.c:2607): an object is not
//!   where the xref says → [`PdfDocument::repair_xref`], then the lookup is
//!   retried. Once per document only -- `repair_attempted`.
//!
//! ## The pieces, C to Rust
//!
//! | MuPDF | here |
//! |---|---|
//! | `pdf_repair_obj` (pdf-repair.c:74) | [`repair_obj`] |
//! | the scan loop of `pdf_repair_xref_base` (:396) | [`scan_objects`] |
//! | the xref/trailer half of `pdf_repair_xref_base` | [`PdfDocument::repair_xref_base`] |
//! | `pdf_repair_obj_stm` / `pdf_repair_obj_stms` (:283, :758) | [`PdfDocument::repair_obj_stm`] / [`PdfDocument::repair_obj_stms`] |
//! | `entry_offset` (:262) | [`PdfDocument::entry_offset`] |
//! | `pdf_repair_roots` (:800) | [`PdfDocument::repair_roots`] |
//! | `pdf_repair_trailer` (:815) | [`PdfDocument::repair_trailer`] |
//! | `pdf_repair_xref_aux` (:953) | [`PdfDocument::repair_xref`] / [`PdfDocument::repair_at_open`] |
//! | `pdf_parse_ind_obj`'s `try_repair` flag (pdf-parse.c:783) | [`parse_ind_obj_at`] |
//!
//! ## Deliberate differences
//!
//! * `orphan_object` (the old `/Length` kept alive for other holders of the
//!   refcounted dict) has no job here: objects are owned values, the cache
//!   just gets the corrected dict.
//! * `pdf_lex_no_string` (strings lexed but not stored) is plain [`lex`]:
//!   same tokens, only an allocation more per string.
//! * The byte-at-a-time `endstream` search is a slice search over the file.
//!   Same first match, same resulting length.
//! * The linearized-reading `page` output of `pdf_repair_obj`, the page-tree
//!   map reset, `throw_on_repair` and the FDF case do not apply to this
//!   in-memory, non-progressive reader.
//! * `doc->bias` (garbage before `%PDF`) is not ported anywhere in the
//!   reader, so "reset bias" has nothing to reset.

use super::error::{Error, ErrorKind, Result};
use super::lex::{Token, lex};
use super::object::{MAX_OBJECT_NUMBER, Object};
use super::parse::{IndirectObject, parse_dict, parse_ind_obj};
use super::stream::{Stream, Whence};
use super::xref::{PdfDocument, XrefEntry, find_bytes};

/// One `N G obj` the scan found (MuPDF's `struct entry`, pdf-repair.c:31).
#[derive(Clone, Copy, Debug)]
pub(super) struct ScanEntry {
    /// Object number.
    num: i32,
    /// Generation, clamped to `0..=65535`. Recorded for parity with MuPDF;
    /// the port's xref keys on the object number only.
    #[allow(dead_code)]
    generation: i32,
    /// Offset of the `N` token -- where the object's header starts.
    ofs: i64,
    /// Offset of the stream body, or 0 when the object is not a stream.
    stm_ofs: i64,
    /// The stream length measured by finding `endstream`, or -1 when the
    /// declared `/Length` was right (or there is no stream) -- MuPDF only
    /// corrects `/Length` when it had to search.
    stm_len: i64,
}

/// Everything the whole-file scan collects before any object is loaded.
pub(super) struct ScanResult {
    /// Every object definition, in file order (later ones win).
    list: Vec<ScanEntry>,
    /// The highest object number seen.
    maxnum: i32,
    /// The last `/Encrypt` any trailer (or xref stream) named.
    encrypt: Option<Object>,
    /// The `/ID` to use (see the rule in [`scan_objects`]).
    id: Option<Object>,
    /// The last `/Info` any trailer named.
    info: Option<Object>,
    /// Every `/Root` candidate, in file order (`pdf_root_list`).
    roots: Vec<Object>,
}

/// What [`PdfDocument::repair_xref_base`] hands on to the later passes.
pub(super) struct RepairBase {
    /// The synthesised trailer so far: `/Size`, `/Info`, `/Encrypt`, `/ID`.
    trailer: Object,
    /// `/Root` candidates for [`PdfDocument::repair_roots`].
    roots: Vec<Object>,
    /// Object numbers whose (last) definition is a stream -- MuPDF's
    /// `entry->stm_ofs != 0`, the ones `pdf_repair_obj_stms` looks at.
    stream_objs: Vec<i32>,
}

/// What [`repair_obj`] returns: the token *after* the object (the scan
/// always has one token in hand), where that token started, and the stream
/// geometry if the object was a stream.
struct RepairedObj {
    tok: Token,
    tmpofs: i64,
    stm_ofs: i64,
    stm_len: i64,
}

// MuPDF: pdf_repair_obj (pdf-repair.c:74).
/// Skim one object, having just read `N G obj`: parse it only if it is a
/// dictionary, pick up an xref stream's `/Encrypt` `/ID` `/Root`, and for a
/// stream find its true end.
///
/// The `/Length` is trusted only if `endstream` sits right where it says;
/// otherwise the body is scanned for `endstream` and the measured length is
/// returned in `stm_len` (else -1).
fn repair_obj(
    f: &mut Stream,
    bytes: &[u8],
    encrypt: &mut Option<Object>,
    id: &mut Option<Object>,
    root: &mut Option<Object>,
) -> Result<RepairedObj> {
    let mut stm_ofs = 0i64;
    let mut stm_len_out = -1i64;
    let mut stm_len = 0i64;

    let mut tmpofs = f.tell();

    // We know we have just seen `<int> <int> obj`. Only a dictionary needs a
    // full parse; anything else is skipped token by token below.
    let mut tok = lex(f)?;

    // Don't let a truncated object at EOF overwrite a good one.
    if tok == Token::Eof {
        return Err(Error::syntax("truncated object"));
    }

    if tok == Token::OpenDict {
        let dict = match parse_dict(f) {
            Ok(d) => d,
            Err(e) => {
                if e.kind() == ErrorKind::System {
                    return Err(e);
                }
                // Don't let a broken object at EOF overwrite a good one.
                if f.is_eof()? {
                    return Err(e);
                }
                // Otherwise swallow it: an empty dict stands in.
                Object::new_dict()
            }
        };

        // Only ever DIRECT values here -- nothing may be resolved mid-scan,
        // there is no xref to resolve through yet.
        if matches!(dict.dict_gets("Type"), Some(Object::Name(n)) if n.as_slice() == b"XRef") {
            if let Some(obj) = dict.dict_gets("Encrypt") {
                *encrypt = Some(obj.clone());
            }
            if let Some(obj) = dict.dict_gets("ID") {
                *id = Some(obj.clone());
            }
            *root = dict.dict_gets("Root").cloned();
        }

        if let Some(Object::Int(len)) = dict.dict_gets("Length") {
            stm_len = *len;
        }
    }

    while !matches!(
        tok,
        Token::Stream | Token::EndObj | Token::Error | Token::Eof | Token::Int(_)
    ) {
        tmpofs = f.tell();
        tok = lex(f)?;
    }

    if tok == Token::Stream {
        // One EOL byte after `stream` (whatever it is -- MuPDF reads one
        // byte, and a CR may bring its LF along).
        if f.read_byte()? == Some(b'\r') && f.peek_byte()? == Some(b'\n') {
            f.read_byte()?;
        }
        stm_ofs = f.tell();

        let mut at_objend = false;
        if stm_len > 0 {
            f.seek(stm_ofs.saturating_add(stm_len), Whence::Set)?;
            match lex(f) {
                Ok(Token::EndStream) => at_objend = true,
                Ok(_) => {}
                Err(e) if e.kind() == ErrorKind::System => return Err(e),
                // "cannot find endstream token, falling back to scanning"
                Err(_) => {}
            }
            if !at_objend {
                f.seek(stm_ofs, Whence::Set)?;
            }
        }

        if !at_objend {
            // Scan for `endstream`; the stream stops where it starts. At EOF
            // without one, the position is EOF (and the length may even come
            // out negative, which then disables the /Length correction --
            // exactly as in C).
            let from = (stm_ofs.max(0) as usize).min(bytes.len());
            let after = match find_bytes(&bytes[from..], b"endstream") {
                Some(p) => from + p + 9,
                None => bytes.len(),
            };
            f.seek(after as i64, Whence::Set)?;
            stm_len_out = f.tell() - stm_ofs - 9;
        }

        // atobjend:
        tmpofs = f.tell();
        tok = lex(f)?;
        // (A missing `endobj` is only a warning in MuPDF.)
        if tok == Token::EndObj {
            // Read another token, as we always return the next one.
            tmpofs = f.tell();
            tok = lex(f)?;
        }
    }

    Ok(RepairedObj {
        tok,
        tmpofs,
        stm_ofs,
        stm_len: stm_len_out,
    })
}

// MuPDF: is_white (pdf-repair.c:390).
fn is_white(c: u8) -> bool {
    matches!(c, b'\x00' | b'\x09' | b'\x0a' | b'\x0c' | b'\x0d' | b'\x20')
}

// MuPDF: the scanning loop of pdf_repair_xref_base (pdf-repair.c:441-661).
/// One pass over the whole file with the lexer, collecting every object and
/// every trailer's `/Encrypt`, `/ID`, `/Root` and `/Info`.
///
/// Integers are tracked two deep (`num`, `gen` and where each began) so that
/// on meeting `obj` the object's number, generation and header offset are
/// known. A `<<` at top level is taken as a trailer (or a stray dict), and
/// its keys are kept only if present -- `/ID` with one extra rule, straight
/// from MuPDF: a later `/ID` replaces an earlier one unless an `/Encrypt` is
/// already known and this dict does not carry its own `/Encrypt`, so the
/// `/ID` stays paired with the encryption it keys.
pub(super) fn scan_objects(bytes: &[u8]) -> Result<ScanResult> {
    let mut f = Stream::from_slice(bytes);

    let mut list: Vec<ScanEntry> = Vec::with_capacity(1024);
    let mut maxnum = 0i32;
    let mut encrypt: Option<Object> = None;
    let mut id: Option<Object> = None;
    let mut info: Option<Object> = None;
    let mut roots: Vec<Object> = Vec::new();

    // Look for the '%PDF' version marker within the first kilobyte.
    let n = bytes.len().min(1024);
    if n >= 5 {
        for j in 0..n - 5 {
            if &bytes[j..j + 5] == b"%PDF-" || &bytes[j..j + 5] == b"%FDF-" {
                f.seek((j + 8) as i64, Whence::Set)?; // skip "%PDF-X.Y"
                break;
            }
        }
    }

    // Skip the comment line after the version marker, since some generators
    // forget to end it with a newline.
    while let Some(c) = f.peek_byte()? {
        if c != b' ' && c != b'%' {
            break;
        }
        f.read_byte()?;
    }

    let mut num: i64 = 0;
    let mut gen_: i64 = 0;
    let mut numofs: i64 = 0;
    let mut genofs: i64 = 0;

    'scan: loop {
        let mut tmpofs = f.tell();
        let mut tok = match lex(&mut f) {
            Ok(t) => t,
            Err(e) if e.kind() == ErrorKind::System => return Err(e),
            Err(_) => {
                // "skipping ahead to next token"
                let mut hit_eof = true;
                while let Some(c) = f.read_byte()? {
                    if is_white(c) {
                        hit_eof = false;
                        break;
                    }
                }
                if !hit_eof {
                    continue 'scan;
                }
                Token::Eof
            }
        };

        // have_next_token: -- the object branch hands back the token after
        // the object, which is handled here without lexing again.
        loop {
            match tok {
                Token::Int(i) => {
                    if i < 0 {
                        num = 0;
                        gen_ = 0;
                        continue 'scan;
                    }
                    numofs = genofs;
                    num = gen_;
                    genofs = tmpofs;
                    gen_ = i;
                    continue 'scan;
                }
                Token::Obj => {
                    let mut root = None;
                    let obj = match repair_obj(&mut f, bytes, &mut encrypt, &mut id, &mut root) {
                        Ok(obj) => obj,
                        Err(e) => {
                            // With no root seen yet there is nothing to make
                            // do with -- give up. Otherwise keep what we have
                            // and "ignore the rest of the file".
                            if roots.is_empty() || e.kind() == ErrorKind::System {
                                return Err(e);
                            }
                            break 'scan;
                        }
                    };
                    if let Some(r) = root {
                        roots.push(r);
                    }
                    tok = obj.tok;
                    tmpofs = obj.tmpofs;

                    if num <= 0 || num > MAX_OBJECT_NUMBER as i64 {
                        // "ignoring object with invalid object number"
                        continue;
                    }
                    let generation = gen_.clamp(0, 65535) as i32;
                    gen_ = generation as i64;
                    list.push(ScanEntry {
                        num: num as i32,
                        generation,
                        ofs: numofs,
                        stm_ofs: obj.stm_ofs,
                        stm_len: obj.stm_len,
                    });
                    maxnum = maxnum.max(num as i32);
                    continue;
                }
                // Probably the trailer; possibly a stray dict in a corrupt
                // file.
                Token::OpenDict => {
                    let dict = match parse_dict(&mut f) {
                        Ok(d) => d,
                        Err(e) if e.kind() == ErrorKind::System => return Err(e),
                        // A broken trailer is trouble, but it may just have
                        // been a bogus dict -- keep going.
                        Err(_) => continue 'scan,
                    };
                    let has_encrypt = dict.dict_gets("Encrypt").is_some();
                    if let Some(obj) = dict.dict_gets("Encrypt") {
                        encrypt = Some(obj.clone());
                    }
                    if let Some(obj) = dict.dict_gets("ID")
                        && (id.is_none() || encrypt.is_none() || has_encrypt)
                    {
                        id = Some(obj.clone());
                    }
                    if let Some(obj) = dict.dict_gets("Root") {
                        roots.push(obj.clone());
                    }
                    if let Some(obj) = dict.dict_gets("Info") {
                        info = Some(obj.clone());
                    }
                    continue 'scan;
                }
                Token::Eof => break 'scan,
                _ => {
                    num = 0;
                    gen_ = 0;
                    continue 'scan;
                }
            }
        }
    }

    if list.is_empty() {
        return Err(Error::format("no objects found"));
    }

    Ok(ScanResult {
        list,
        maxnum,
        encrypt,
        id,
        info,
        roots,
    })
}

// MuPDF: the `try_repair` out-parameter of pdf_parse_ind_obj_or_newobj
// (pdf-parse.c:783-835).
/// Parse the indirect object at `offset`, also reporting whether a failure
/// was in its `N G obj` *header* -- the only failures `pdf_cache_object`
/// answers with a repair. A bad object number (out of range) and anything
/// wrong in the body do not ask for repair.
///
/// `parse::parse_ind_obj` itself does not carry the flag, so the header is
/// lexed once here to classify, then the object is parsed from the top.
pub(super) fn parse_ind_obj_at(bytes: &[u8], offset: i64) -> (Result<IndirectObject>, bool) {
    let mut f = Stream::from_slice(bytes);
    if let Err(e) = f.seek(offset, Whence::Set) {
        return (Err(e), false);
    }
    let num = match lex(&mut f) {
        Ok(Token::Int(i)) => i,
        Ok(_) => return (Err(Error::syntax("expected object number")), true),
        Err(e) => return (Err(e), false),
    };
    if num < 0 || num > MAX_OBJECT_NUMBER as i64 {
        return (Err(Error::syntax("object number out of range")), false);
    }
    let generation = match lex(&mut f) {
        Ok(Token::Int(i)) => i,
        Ok(_) => {
            return (
                Err(Error::syntax(format!("expected generation number ({num} ? obj)"))),
                true,
            );
        }
        Err(e) => return (Err(e), false),
    };
    if !(0..65536).contains(&generation) {
        return (
            Err(Error::syntax(format!("invalid generation number ({generation})"))),
            true,
        );
    }
    match lex(&mut f) {
        Ok(Token::Obj) => {}
        Ok(_) => {
            return (
                Err(Error::syntax(format!(
                    "expected 'obj' keyword ({num} {generation} ?)"
                ))),
                true,
            );
        }
        Err(e) => return (Err(e), false),
    }
    if let Err(e) = f.seek(offset, Whence::Set) {
        return (Err(e), false);
    }
    (parse_ind_obj(&mut f), false)
}

impl PdfDocument {
    // MuPDF: pdf_repair_xref (pdf-xref.c:5527) →
    // pdf_repair_xref_aux(doc, pdf_prime_xref_index) (pdf-repair.c:953).
    /// Rebuild the xref from a whole-file scan, from inside an object lookup.
    ///
    /// Everything cached so far is dropped (MuPDF's `pdf_forget_xref`), the
    /// new table replaces the old, and the new trailer becomes the one
    /// [`trailer`](Self::trailer) returns. Runs at most once per document.
    pub(super) fn repair_xref(&self) -> Result<()> {
        // The recursion guard belongs to the lookup that asked for repair;
        // the repair's own lookups start from a clean slate, else re-reading
        // the object being resolved would look like a cycle.
        let saved = std::mem::take(&mut *self.resolving.borrow_mut());
        let result = self
            .repair_xref_base()
            .and_then(|base| self.repair_finish(base));
        *self.resolving.borrow_mut() = saved;
        let trailer = result?;
        // Repair runs once, so the cell is empty; if it somehow is not, the
        // first repaired trailer stands.
        let _ = self.repaired_trailer.set(trailer);
        Ok(())
    }

    // MuPDF: pdf_init_document's repair branch (pdf-xref.c:1972) →
    // pdf_repair_xref_aux(doc, id_and_password).
    /// Rebuild the xref at open, when the file's own xref would not load.
    ///
    /// Same passes as [`repair_xref`](Self::repair_xref), with MuPDF's `mid`
    /// step in between: once the base scan has found `/Encrypt` and `/ID`,
    /// the decryptor is set up (`id_and_password`) so the object streams read
    /// in the later passes come out as plaintext.
    pub(super) fn repair_at_open(&mut self) -> Result<()> {
        let base = self.repair_xref_base()?;
        self.trailer = base.trailer.clone();
        self.id_and_password();
        let trailer = self.repair_finish(base)?;
        self.trailer = trailer;
        Ok(())
    }

    // MuPDF: id_and_password (pdf-xref.c:1898) -- `pdf_new_crypt` +
    // authenticate with "". A failure here is not fatal (MuPDF would just
    // need a password later); `open` re-checks and reports it properly.
    fn id_and_password(&mut self) {
        let Some(enc) = self.trailer.dict_gets("Encrypt").cloned() else {
            return;
        };
        if !self.resolve(&enc).map(|o| o.is_dict()).unwrap_or(false) {
            return;
        }
        self.encrypt_obj_num = match enc {
            Object::Ref { num, .. } => Some(num),
            _ => None,
        };
        if let Ok(d) = self.build_decryptor(b"") {
            self.decryptor = Some(d);
        }
    }

    // MuPDF: the tail of pdf_repair_xref_aux (pdf-repair.c:966-969).
    /// The passes after the base scan: object streams, roots, trailer.
    fn repair_finish(&self, base: RepairBase) -> Result<Object> {
        self.repair_obj_stms(&base.stream_objs)?;
        let mut trailer = base.trailer;
        self.repair_roots(&mut trailer, &base.roots);
        self.repair_trailer(&mut trailer)?;
        Ok(trailer)
    }

    // MuPDF: pdf_repair_xref_base (pdf-repair.c:396), after the scan.
    /// Scan the file and install the rebuilt xref: every slot up to the
    /// highest object number free, then each scanned object in file order
    /// (so the last definition wins), correcting the `/Length` of any stream
    /// whose declared length did not land on `endstream` -- unless the file
    /// is encrypted, where MuPDF leaves `/Length` alone.
    ///
    /// Note one MuPDF quirk kept as-is: the corrected `/Length` is put on the
    /// object as it is loaded *at that point in the list*. If a stream with a
    /// wrong `/Length` is redefined later in the file, the earlier copy is
    /// already cached and stays cached -- MuPDF never clears `entry->obj` in
    /// that loop either.
    pub(super) fn repair_xref_base(&self) -> Result<RepairBase> {
        if self.repair_attempted.get() {
            return Err(Error::format("Repair failed already - not trying again"));
        }
        self.repair_attempted.set(true);

        // pdf_forget_xref: every section and every cached object goes.
        self.entries.borrow_mut().clear();
        self.cache.borrow_mut().clear();

        let scan = scan_objects(&self.bytes)?;

        // Make the xref solid from 0 to maxnum, all free.
        {
            let mut entries = self.entries.borrow_mut();
            entries.clear();
            entries.resize(scan.maxnum as usize + 1, Some(XrefEntry::Free));
        }

        for item in &scan.list {
            self.set_entry(item.num, XrefEntry::Uncompressed { offset: item.ofs });

            // Correct the stream length for unencrypted documents.
            if scan.encrypt.is_none() && item.stm_len >= 0 {
                self.ensure_cached(item.num)?;
                let mut cache = self.cache.borrow_mut();
                if let Some(c) = cache.get_mut(&item.num) {
                    // pdf_dict_get_put_drop throws on a non-dict.
                    if !c.obj.is_dict() {
                        return Err(Error::argument(format!(
                            "not a dict ({} 0 R)",
                            item.num
                        )));
                    }
                    c.obj.dict_put("Length", Object::new_int(item.stm_len));
                }
            }
        }

        // Which objects are (last defined as) streams -- entry->stm_ofs.
        let mut stm_ofs_of: Vec<i64> = vec![0; scan.maxnum as usize + 1];
        for item in &scan.list {
            stm_ofs_of[item.num as usize] = item.stm_ofs;
        }
        let stream_objs: Vec<i32> = stm_ofs_of
            .iter()
            .enumerate()
            .filter(|(_, ofs)| **ofs != 0)
            .map(|(i, _)| i as i32)
            .collect();

        // The repaired trailer; /Root is added by the later passes.
        let mut trailer = Object::new_dict();
        trailer.dict_put("Size", Object::new_int(scan.maxnum as i64 + 1));
        if let Some(info) = scan.info {
            trailer.dict_put("Info", info);
        }
        if let Some(encrypt) = scan.encrypt {
            trailer.dict_put("Encrypt", encrypt);
        }
        if let Some(id) = scan.id {
            trailer.dict_put("ID", id);
        }

        Ok(RepairBase {
            trailer,
            roots: scan.roots,
            stream_objs,
        })
    }

    // MuPDF: entry_offset (pdf-repair.c:262).
    /// Where object `num` physically lives: its own offset, or for a
    /// compressed object the offset of its object stream (-1 if that stream
    /// is not an uncompressed object), or 0 for a free/unknown one.
    fn entry_offset(&self, num: i32) -> i64 {
        let entries = self.entries.borrow();
        match entries.get(num as usize).cloned().flatten() {
            None | Some(XrefEntry::Free) => 0,
            Some(XrefEntry::Uncompressed { offset }) => offset,
            Some(XrefEntry::Compressed { stm_num, .. }) => {
                match entries.get(stm_num as usize).cloned().flatten() {
                    Some(XrefEntry::Uncompressed { offset }) => offset,
                    _ => -1,
                }
            }
        }
    }

    // MuPDF: pdf_repair_obj_stm (pdf-repair.c:283).
    /// Point every object listed in object stream `stm_num`'s header at that
    /// stream -- except one already defined uncompressed *later* in the file
    /// (Bug 708286: an object stream must not override a newer plain object).
    fn repair_obj_stm(&self, stm_num: i32) -> Result<()> {
        let obj = self.get_object(stm_num)?;
        let count = self.resolve_get(&obj, "N")?.to_int();
        let data = self.open_stream_num(stm_num)?;
        let corrupt = || Error::format(format!("corrupt object stream ({stm_num} 0 R)"));

        let mut s = Stream::from_slice(&data);
        let mut i: i64 = 0;
        while i < count {
            let n = match lex(&mut s)? {
                Token::Int(n) => n,
                _ => return Err(corrupt()),
            };
            if n < 0 || n >= MAX_OBJECT_NUMBER as i64 {
                // "ignoring object with invalid object number" -- and, as in
                // C, without reading its offset.
                i += 1;
                continue;
            }
            let n = n as i32;

            let existing = self.entries.borrow().get(n as usize).cloned().flatten();
            let mut replace = true;
            if !matches!(existing, None | Some(XrefEntry::Free)) {
                let existing_offset = self.entry_offset(n);
                // An invalid existing entry: anything is better than that.
                if existing_offset >= 0 && existing_offset > self.entry_offset(stm_num) {
                    replace = false;
                }
            }
            if replace {
                self.set_entry(
                    n,
                    XrefEntry::Compressed {
                        stm_num,
                        index: i as i32,
                    },
                );
                self.cache.borrow_mut().remove(&n);
            }

            match lex(&mut s)? {
                Token::Int(_) => {}
                _ => return Err(corrupt()),
            }
            i += 1;
        }
        Ok(())
    }

    // MuPDF: pdf_repair_obj_stms (pdf-repair.c:758).
    /// Run [`repair_obj_stm`](Self::repair_obj_stm) on every stream object
    /// that is an `/ObjStm` (a broken one is skipped with a shrug), then free
    /// any compressed entry whose container is not an uncompressed object.
    fn repair_obj_stms(&self, stream_objs: &[i32]) -> Result<()> {
        let xref_len = self.entries.borrow().len();

        for &i in stream_objs {
            // Outside MuPDF's try: a failure to load the dict is fatal.
            let dict = self.get_object(i)?;
            let is_objstm = matches!(
                self.resolve_get(&dict, "Type"),
                Ok(Object::Name(n)) if n.as_slice() == b"ObjStm"
            );
            if is_objstm
                && let Err(e) = self.repair_obj_stm(i)
            {
                if e.kind() == ErrorKind::System {
                    return Err(e);
                }
                // "ignoring broken object stream (i 0 R)"
            }
        }

        // Ensure that streamed objects reside inside a known non-streamed
        // object.
        for i in 0..xref_len {
            let stm = match self.entries.borrow().get(i).cloned().flatten() {
                Some(XrefEntry::Compressed { stm_num, .. }) => stm_num,
                _ => continue,
            };
            let container_ok = matches!(
                self.entries.borrow().get(stm as usize).cloned().flatten(),
                Some(XrefEntry::Uncompressed { .. })
            );
            if !container_ok {
                // "invalid reference to non-object-stream: assuming a freed
                // object"
                self.set_entry(i as i32, XrefEntry::Free);
            }
        }
        Ok(())
    }

    // MuPDF: pdf_repair_roots (pdf-repair.c:800).
    /// Take the LAST `/Root` candidate that is an indirect reference to a
    /// dictionary.
    fn repair_roots(&self, trailer: &mut Object, roots: &[Object]) {
        for root in roots.iter().rev() {
            if matches!(root, Object::Ref { .. })
                && self.resolve(root).map(|o| o.is_dict()).unwrap_or(false)
            {
                trailer.dict_put("Root", root.clone());
                break;
            }
        }
    }

    // MuPDF: pdf_repair_trailer (pdf-repair.c:815).
    /// No `/Root` (or `/Info`) yet? Walk the objects from the highest number
    /// down -- newer objects tend to sit higher -- and take the first
    /// `/Type /Catalog` as `/Root` and the first dict with `/Creator` or
    /// `/Producer` as `/Info`.
    fn repair_trailer(&self, trailer: &mut Object) -> Result<()> {
        let mut hasroot = trailer.dict_gets("Root").is_some();
        let mut hasinfo = trailer.dict_gets("Info").is_some();
        let xref_len = self.entries.borrow().len() as i64;

        let mut i = xref_len - 1;
        while i > 0 && (!hasinfo || !hasroot) {
            let num = i as i32;
            i -= 1;
            let in_use = !matches!(
                self.entries.borrow().get(num as usize).cloned().flatten(),
                None | Some(XrefEntry::Free)
            );
            if !in_use {
                continue;
            }
            let dict = match self.get_object(num) {
                Ok(d) => d,
                Err(e) if e.kind() == ErrorKind::System => return Err(e),
                // "ignoring broken object (num 0 R)"
                Err(_) => continue,
            };
            // A DIRECT /Type /Catalog only -- MuPDF compares the name pointer.
            if !hasroot
                && matches!(dict.dict_gets("Type"), Some(Object::Name(n)) if n.as_slice() == b"Catalog")
            {
                trailer.dict_put("Root", Object::new_indirect(num as i64, 0));
                hasroot = true;
            }
            if !hasinfo
                && (dict.dict_gets("Creator").is_some() || dict.dict_gets("Producer").is_some())
            {
                trailer.dict_put("Info", Object::new_indirect(num as i64, 0));
                hasinfo = true;
            }
        }

        // fz_always: with a decryptor in place, nothing may stay cached from
        // the repair passes -- /Encrypt and /ID are re-read undecrypted on
        // demand (the /Encrypt object is protected by `encrypt_obj_num`).
        if self.decryptor.is_some() {
            self.cache.borrow_mut().clear();
        }
        Ok(())
    }
}

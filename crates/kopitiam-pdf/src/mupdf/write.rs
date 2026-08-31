//! Writing PDF back out: object serialisation and incremental update.
//!
//! Everything else in this crate **reads**. This is the first module that
//! writes, and it is the foundation both annotation authoring
//! ([`super::annot_edit`]) and form filling ([`super::form`]) stand on.
//!
//! Ported from MuPDF `source/pdf/pdf-write.c` (the `do_incremental` path,
//! `writeobject`) and `source/pdf/pdf-object.c` (`fmt_obj`, `fmt_str`,
//! `fmt_name`) (commit 5fe54ce, AGPL-3.0, © Artifex Software, Inc.).
//!
//! # Why incremental update, not a full rewrite
//!
//! An incremental update appends new/changed objects, a fresh cross-reference
//! section, and a trailer whose `/Prev` points at the previous one (PDF
//! 32000-1:2008 §7.5.6). **The original bytes are never touched.** For a
//! document workbench that matters more than tidiness:
//!
//! * A file we cannot fully round-trip is still safe to annotate. Adding a note
//!   to page 4 does not require understanding all 300 pages.
//! * A bug in this writer cannot corrupt what was already there, because we
//!   only ever append.
//! * Existing digital signatures over the original bytes stay valid.
//! * **Undo becomes truncation.** Because every edit is an append, undoing one
//!   is just cutting the file back to its previous length — the earlier xref
//!   and `%%EOF` are still sitting there intact. [`super::annot_edit`]'s
//!   history relies on this.
//!
//! # The compatibility bar
//!
//! Output must be readable by **other** software — Okular, poppler, Acrobat —
//! not merely by our own parser. A file only we can read is silent data loss on
//! someone's annotated document.
//!
//! # Scope decisions made in this port
//!
//! * **Classic `xref` tables only, never a cross-reference *stream*.** MuPDF's
//!   writer picks the section kind to match the source document
//!   (`pdf_write.c:1373` vs. `:1580`). This port always appends a classic
//!   table, even onto a document whose own last section was a compressed xref
//!   stream — legal per §7.5.8.4's hybrid-reference discussion, and simpler to
//!   get right than porting the deflate + predictor + `/W`-width xref-stream
//!   encoder too. Both fixtures this module's tests exercise
//!   (`ink-annots-no-ap.pdf`, `ink-annots-mixed-ap.pdf`) happen to use classic
//!   tables themselves; a document whose *own* last section is a stream is
//!   untested here and is a reasonable follow-up if it ever bites.
//! * **Generation 0 on every written entry**, including one that supersedes an
//!   existing object. The frozen contract (`(i32, NewObject)`) carries only an
//!   object *number*, never a generation, and this crate's own resolver
//!   ([`super::xref`]'s `get_object`) does not check generation either — see
//!   that module's note "the generation number is not tracked". Almost every
//!   PDF in the wild keeps every object at generation 0 forever (a non-zero
//!   generation only shows up on a *reused* freed slot), so this is a
//!   practical non-issue; it is called out here because a stricter external
//!   validator, given a document that genuinely used generation > 0 for a
//!   superseded object, could flag the mismatch. Fixing it needs a wider
//!   contract than this task owns.
//! * **No string-to-hex fallback.** `fmt_obj` (`pdf-object.c:3651`) sometimes
//!   switches a string to `<hex>` form (already-binary content, very long
//!   strings, a leading UTF-16 BOM). This port always emits the literal
//!   `(...)` form with octal escapes for anything not cleanly printable
//!   (`fmt_str_out`, `pdf-object.c:3389`) — strictly simpler, and it round-trips
//!   identically. Add `fmt_hex` if output size ever matters.
//! * **No `/Type`/`/Subtype`-first dict reordering, no signature bypass.**
//!   `fmt_dict`'s tight path (`pdf-object.c:3510`) sends `/Type`/`/Subtype`
//!   first (a compression nicety) and skips encrypting `/Contents` under a
//!   `/Type /Sig` dict (we have no `/Encrypt` support at all, so nothing to
//!   bypass). Neither changes correctness; neither is ported.
//! * **No encryption.** Every `fmt_*` crypt parameter in MuPDF is dropped —
//!   this port only ever calls the plaintext path.

use std::collections::BTreeMap;

use super::error::{Error, Result};
use super::object::{MAX_OBJECT_NUMBER, Object};
use super::xref::PdfDocument;

/// An object to write into an incremental update.
pub enum NewObject {
    /// A plain (non-stream) object.
    Plain(Object),
    /// A stream: its dict plus already-encoded bytes. The writer sets
    /// `/Length`; the caller supplies any `/Filter` (data is written verbatim,
    /// so uncompressed data needs none).
    Stream { dict: Object, data: Vec<u8> },
}

// ---------------------------------------------------------------------------
// Object serialisation (MuPDF pdf-object.c: fmt_obj / fmt_str / fmt_name)
// ---------------------------------------------------------------------------

// MuPDF: iswhite (pdf-object.c:3271).
/// PDF whitespace, for the purposes of `fmt_name`'s escaping and the
/// token-separator logic below. NUL, HT, LF, FF, CR, space — note this is
/// *not* the same six bytes `\x0b` (vertical tab) is excluded from; that
/// matches MuPDF exactly, byte for byte.
fn is_white(c: u8) -> bool {
    matches!(c, 0x00 | 0x09 | 0x0A | 0x0C | 0x0D | 0x20)
}

// MuPDF: isdelim (pdf-object.c:3281).
/// PDF delimiter characters: `( ) < > [ ] { } / %`.
fn is_delim(c: u8) -> bool {
    matches!(
        c,
        b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'/' | b'%'
    )
}

/// Uppercase hex digit for a nibble (0..=15). Matches `fmt_name`'s
/// `c < 0xA ? c + '0' : c + 'A' - 0xA`.
fn hex_digit_upper(nibble: u8) -> u8 {
    if nibble < 10 {
        b'0' + nibble
    } else {
        b'A' + (nibble - 10)
    }
}

/// Format an `f64` the way `fmt_obj`'s real-number branch does (`pdf-object.c:
/// 3651`), but **never** via `%g`: a `%g` real can render in exponential
/// notation (`1e+20`), which is not a legal PDF number (PDF 32000-1 §7.3.3 —
/// a real is `sign? digit* '.' digit*`, no exponent). MuPDF's own C library
/// happens not to hit that case for typical page-geometry numbers, but we
/// have no such guarantee here, so this always emits fixed-point.
///
/// * An integral value formats as a bare integer (`f == (int)f` in the
///   original — `f == f.trunc()` here), matching MuPDF's space-saving case.
/// * Otherwise, six decimal places, trailing zeros (and a trailing `.`)
///   trimmed — a reasonable, always-valid stand-in for `%g`'s
///   significant-digit rounding.
/// * Non-finite input (`NaN`/`±inf`, which should never occur in a PDF
///   number but must not be allowed to panic or emit invalid syntax) degrades
///   to `"0"`.
fn format_real(f: f64) -> String {
    if !f.is_finite() {
        return "0".to_string();
    }
    if f == f.trunc() && f.abs() < 1e18 {
        return format!("{}", f as i64);
    }
    let mut s = format!("{f:.6}");
    if s.contains('.') {
        while s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
    }
    s
}

/// The tight-mode `struct fmt` (`pdf-object.c:3255`), minus everything this
/// port has no use for (indentation, column tracking, encryption). `sep` and
/// `last` reproduce `fmt_putc`'s token-separator logic exactly (`pdf-object.c:
/// 3291`): PDF tokens are only self-delimiting at a delimiter or whitespace
/// byte, so two adjacent non-delimiter tokens (`1` then `2`, or `/N` then `5`)
/// would otherwise fuse into one on re-lexing. Getting this wrong is a subtler
/// version of the same compatibility bar the string/name escaping is about —
/// the file looks fine to the eye and fails to reparse.
struct Fmt<'a> {
    out: &'a mut Vec<u8>,
    sep: bool,
    last: u8,
}

impl<'a> Fmt<'a> {
    fn putc(&mut self, c: u8) {
        if self.sep && !is_delim(self.last) && !is_white(self.last) && !is_delim(c) && !is_white(c)
        {
            self.sep = false;
            self.putc(b' ');
        }
        self.sep = false;
        self.out.push(c);
        self.last = c;
    }

    fn puts(&mut self, s: &[u8]) {
        for &b in s {
            self.putc(b);
        }
    }

    fn set_sep(&mut self) {
        self.sep = true;
    }

    // MuPDF: fmt_str / fmt_str_out (pdf-object.c:3389, 3405).
    fn write_string(&mut self, bytes: &[u8]) {
        self.putc(b'(');
        for &c in bytes {
            match c {
                b'\n' => self.puts(b"\\n"),
                b'\r' => self.puts(b"\\r"),
                b'\t' => self.puts(b"\\t"),
                0x08 => self.puts(b"\\b"),
                0x0C => self.puts(b"\\f"),
                b'(' => self.puts(b"\\("),
                b')' => self.puts(b"\\)"),
                b'\\' => self.puts(b"\\\\"),
                c if !(32..127).contains(&c) => {
                    self.putc(b'\\');
                    self.putc(b'0' + ((c / 64) & 7));
                    self.putc(b'0' + ((c / 8) & 7));
                    self.putc(b'0' + (c & 7));
                }
                c => self.putc(c),
            }
        }
        self.putc(b')');
        // No trailing `set_sep()`: `)` is itself a delimiter (`fmt_str` sets
        // no separator flag either), so the next token never needs a space
        // inserted before it regardless.
    }

    // MuPDF: fmt_name (pdf-object.c:3450).
    fn write_name(&mut self, bytes: &[u8]) {
        self.putc(b'/');
        for &b in bytes {
            if is_delim(b) || is_white(b) || b == b'#' || !(32..127).contains(&b) {
                self.putc(b'#');
                self.putc(hex_digit_upper(b >> 4));
                self.putc(hex_digit_upper(b & 0x0f));
            } else {
                self.putc(b);
            }
        }
        self.set_sep();
    }

    // MuPDF: fmt_array, tight branch (pdf-object.c:3476).
    fn write_array(&mut self, items: &[Object]) {
        self.putc(b'[');
        for item in items {
            self.write(item);
        }
        self.putc(b']');
    }

    // MuPDF: fmt_dict, tight branch (pdf-object.c:3524). The `/Type`/`/Subtype`
    // reordering and the `/Contents`-under-`/Sig` crypt bypass are not ported
    // (see the module docs); every key/value pair is emitted in the dict's
    // own stored order.
    fn write_dict(&mut self, items: &[(Vec<u8>, Object)]) {
        self.puts(b"<<");
        for (key, val) in items {
            self.write_name(key);
            self.write(val);
        }
        self.puts(b">>");
    }

    // MuPDF: fmt_obj (pdf-object.c:3613).
    fn write(&mut self, obj: &Object) {
        match obj {
            Object::Null => {
                self.puts(b"null");
                self.set_sep();
            }
            Object::Bool(true) => {
                self.puts(b"true");
                self.set_sep();
            }
            Object::Bool(false) => {
                self.puts(b"false");
                self.set_sep();
            }
            Object::Ref { num, generation } => {
                self.puts(format!("{num} {generation} R").as_bytes());
                self.set_sep();
            }
            Object::Int(i) => {
                self.puts(i.to_string().as_bytes());
                self.set_sep();
            }
            Object::Real(f) => {
                self.puts(format_real(*f).as_bytes());
                self.set_sep();
            }
            Object::String(bytes) => self.write_string(bytes),
            Object::Name(bytes) => self.write_name(bytes),
            Object::Array(items) => self.write_array(items),
            Object::Dict(items) => self.write_dict(items),
        }
    }
}

/// Serialise `obj` as PDF syntax onto `out`.
///
/// Must round-trip through this crate's own parser **and** escape strings and
/// names per PDF 32000-1 §7.3.4/§7.3.5. An unescaped byte in a `/Name` or a
/// `(string)` yields a file our parser may well accept while Acrobat rejects
/// it — which is the whole compatibility bar in miniature.
pub fn write_object(out: &mut Vec<u8>, obj: &Object) {
    let mut fmt = Fmt {
        out,
        sep: false,
        last: 0,
    };
    fmt.write(obj);
}

// ---------------------------------------------------------------------------
// Incremental update (MuPDF pdf-write.c: writeobject / writexref / do_incremental)
// ---------------------------------------------------------------------------

/// The next free object number for `doc` (its `/Size`).
///
/// Reads the trailer's declared `/Size` directly (PDF 32000-1 §7.5.4 requires
/// it to exceed the highest object number in use). This crate's [`PdfDocument`]
/// keeps its cross-reference table private, so there is no cheaper or more
/// authoritative source available from outside `xref.rs`; a malformed trailer
/// missing `/Size` degrades to `1` rather than panicking.
pub fn next_object_number(doc: &PdfDocument) -> i32 {
    doc.trailer()
        .dict_gets("Size")
        .map(|o| o.to_int())
        .filter(|&v| v > 0 && v <= MAX_OBJECT_NUMBER as i64 + 1)
        .map(|v| v as i32)
        .unwrap_or(1)
}

// MuPDF: pdf_read_start_xref (pdf-xref.c:1010).
/// Find the current `startxref` offset by scanning the file's tail (the last
/// 1024 bytes), exactly as `xref.rs`'s private `read_start_xref` does.
///
/// Duplicated rather than shared: this task's parallel-agent run scoped work
/// to this file alone, and `xref.rs::read_start_xref` is private, owned by a
/// module nobody in this run is touching. If that boundary ever lifts, this
/// should become one `pub(crate)` helper instead of two copies of the same
/// twenty lines.
fn find_current_startxref(bytes: &[u8]) -> Result<i64> {
    const WINDOW: usize = 1024;
    let n = bytes.len();
    let start = n.saturating_sub(WINDOW);
    let buf = &bytes[start..];
    if buf.len() < 9 {
        return Err(Error::format("cannot find startxref"));
    }
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

/// Clone `obj`, dropping `key` if it is a dict. Used to strip `/XRefStm` from
/// the carried-forward trailer (MuPDF does the same in `writexref`,
/// `pdf-write.c:1400`): a hybrid-reference `/XRefStm` offset in the *old*
/// trailer would otherwise dangle, pointing at a stream section our own
/// `/Prev` chain does not otherwise reference.
fn dict_without_key(obj: &Object, key: &[u8]) -> Object {
    match obj {
        Object::Dict(items) => Object::Dict(
            items
                .iter()
                .filter(|(k, _)| k.as_slice() != key)
                .cloned()
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Write one `N 0 obj … endobj` record onto `out`, returning nothing (the
/// caller records the byte offset it started at). Mirrors `writeobject`
/// (`pdf-write.c:1248`) for the plain-object and uncompressed-stream cases;
/// MuPDF's compression/expansion/hex-filter/labelling machinery is out of
/// scope (see the module docs).
fn write_indirect(out: &mut Vec<u8>, num: i32, obj: &NewObject) {
    match obj {
        NewObject::Plain(o) => {
            out.extend_from_slice(format!("{num} 0 obj\n").as_bytes());
            write_object(out, o);
            out.extend_from_slice(b"\nendobj\n\n");
        }
        NewObject::Stream { dict, data } => {
            let mut dict = dict.clone();
            dict.dict_put("Length", Object::new_int(data.len() as i64));
            out.extend_from_slice(format!("{num} 0 obj\n").as_bytes());
            write_object(out, &dict);
            out.extend_from_slice(b"\nstream\n");
            out.extend_from_slice(data);
            out.extend_from_slice(b"\nendstream\nendobj\n\n");
        }
    }
}

// MuPDF: writexrefsubsect (pdf-write.c:1352).
/// Write one `start count` classic-xref subsection plus its fixed 20-byte
/// entries (`"%010lu %05d n \n"` for every entry this port ever writes —
/// generation is always 0, see the module docs).
fn write_xref_subsection(out: &mut Vec<u8>, offsets: &BTreeMap<i32, i64>, start: i32, end: i32) {
    let count = (end - start + 1).max(0);
    out.extend_from_slice(format!("{start} {count}\n").as_bytes());
    for num in start..=end {
        let offset = offsets.get(&num).copied().unwrap_or(0);
        out.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
    }
}

// MuPDF: writexref, do_incremental branch (pdf-write.c:1373).
/// Write a full classic `xref` section covering every object number in
/// `offsets`, split into contiguous subsections (a gap in object numbers
/// needs its own `start count` header — folding a non-contiguous range into
/// one subsection is exactly the mistake the module docs warn about).
/// [`write_classic_xref`] for callers outside this module -- `xref.rs`'s
/// decrypt-on-open rewrite emits the same table shape and must not grow a
/// second copy of it.
pub fn write_classic_xref_pub(out: &mut Vec<u8>, offsets: &BTreeMap<i32, i64>) {
    write_classic_xref(out, offsets);
}

/// [`dict_without_key`] for callers outside this module, same reasoning.
pub fn dict_without_key_pub(obj: &Object, key: &[u8]) -> Object {
    dict_without_key(obj, key)
}

fn write_classic_xref(out: &mut Vec<u8>, offsets: &BTreeMap<i32, i64>) {
    out.extend_from_slice(b"xref\n");
    let nums: Vec<i32> = offsets.keys().copied().collect();
    let mut idx = 0;
    while idx < nums.len() {
        let start = nums[idx];
        let mut end = start;
        let mut j = idx;
        while j + 1 < nums.len() && nums[j + 1] == end + 1 {
            end += 1;
            j += 1;
        }
        write_xref_subsection(out, offsets, start, end);
        idx = j + 1;
    }
    out.extend_from_slice(b"\n");
}

/// Append an incremental update to `doc`'s bytes and return the **complete new
/// file**. A number below [`next_object_number`] supersedes that object; one at
/// or above it adds a new object.
///
/// An empty `updates` is a no-op: returns `doc`'s bytes unchanged rather than
/// appending an empty, pointless xref section.
///
/// # What is written, in order
///
/// 1. Each `(num, NewObject)`, as `N 0 obj … endobj` (or the stream form),
///    recording the byte offset it started at.
/// 2. A classic cross-reference section covering exactly the object numbers
///    just written, grouped into contiguous subsections.
/// 3. A trailer cloned from `doc.trailer()` (carrying forward whatever it has
///    — `/Root`, `/Info`, `/ID`, …) with `/Size` raised to cover every object
///    number now in use, `/Prev` set to this file's current `startxref`, and
///    `/XRefStm` dropped (see [`dict_without_key`]).
/// 4. `startxref` pointing at the new xref section, then `%%EOF`.
///
/// The original bytes are **never modified** — only ever appended to (after
/// ensuring the file ends in a newline first, since a PDF is not guaranteed
/// to). Truncating the returned buffer back to `doc.raw_bytes().len()` yields
/// exactly the original file back, which is the property
/// [`super::annot_edit`]'s undo history depends on.
pub fn incremental_update(doc: &PdfDocument, updates: &[(i32, NewObject)]) -> Result<Vec<u8>> {
    let mut out = doc.raw_bytes().to_vec();
    if updates.is_empty() {
        return Ok(out);
    }
    if !out.ends_with(b"\n") {
        out.push(b'\n');
    }

    let mut offsets: BTreeMap<i32, i64> = BTreeMap::new();
    for (num, obj) in updates {
        if *num < 0 || *num as i64 > MAX_OBJECT_NUMBER as i64 {
            return Err(Error::argument(format!(
                "object number {num} out of range (0..={MAX_OBJECT_NUMBER})"
            )));
        }
        let offset = out.len() as i64;
        write_indirect(&mut out, *num, obj);
        // Later duplicate entries for the same number win (their bytes were
        // still appended, just not referenced by the xref) -- consistent
        // with "at or above [next_object_number] adds a new object" reading
        // updates as an ordered list of edits, last one live.
        offsets.insert(*num, offset);
    }

    let old_size = next_object_number(doc) as i64;
    let max_num = offsets.keys().copied().max().unwrap_or(0) as i64;
    let new_size = old_size.max(max_num + 1);

    let prev = find_current_startxref(doc.raw_bytes())?;

    let xref_offset = out.len() as i64;
    write_classic_xref(&mut out, &offsets);

    let mut trailer = dict_without_key(doc.trailer(), b"XRefStm");
    trailer.dict_put("Size", Object::new_int(new_size));
    trailer.dict_put("Prev", Object::new_int(prev));

    out.extend_from_slice(b"trailer\n");
    write_object(&mut out, &trailer);
    out.extend_from_slice(b"\nstartxref\n");
    out.extend_from_slice(xref_offset.to_string().as_bytes());
    out.extend_from_slice(b"\n%%EOF\n");

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // write_object: exact-byte escaping tests. These pin the cross-reader
    // compatibility bar the module docs talk about -- each expected string
    // below was hand-traced against fmt_putc's separator rule, not just
    // eyeballed, because a plausible-looking-but-wrong separator is the kind
    // of bug that only shows up when a *different* parser re-lexes the file.
    // -----------------------------------------------------------------------

    fn ser(obj: &Object) -> String {
        let mut out = Vec::new();
        write_object(&mut out, obj);
        String::from_utf8(out).expect("test objects are ASCII")
    }

    #[test]
    fn writes_null_bool_int() {
        assert_eq!(ser(&Object::Null), "null");
        assert_eq!(ser(&Object::Bool(true)), "true");
        assert_eq!(ser(&Object::Bool(false)), "false");
        assert_eq!(ser(&Object::new_int(42)), "42");
        assert_eq!(ser(&Object::new_int(-7)), "-7");
    }

    #[test]
    fn writes_real_without_exponent() {
        assert_eq!(ser(&Object::new_real(2.0)), "2");
        assert_eq!(ser(&Object::new_real(-0.5)), "-0.5");
        assert_eq!(ser(&Object::new_real(3.25)), "3.25");
        // Never exponential notation, even for a value %g would render that
        // way -- PDF's number grammar has no exponent form.
        let huge = ser(&Object::new_real(1.0e20));
        assert!(
            !huge.contains('e') && !huge.contains('E'),
            "real formatting must never emit exponential notation: {huge}"
        );
    }

    #[test]
    fn writes_indirect_reference() {
        assert_eq!(
            ser(&Object::new_indirect(12, 3)),
            "12 3 R",
            "num/gen/R with the mandatory internal spaces"
        );
    }

    #[test]
    fn escapes_string_specials_and_binary() {
        // "Hello (World)\" -- parens and the trailing backslash must each be
        // escaped, or the string terminates early / eats the next byte.
        let mut input = b"Hello (World)".to_vec();
        input.push(b'\\');
        let mut expected = Vec::new();
        expected.extend_from_slice(b"(Hello ");
        expected.push(b'\\');
        expected.push(b'(');
        expected.extend_from_slice(b"World");
        expected.push(b'\\');
        expected.push(b')');
        expected.push(b'\\');
        expected.push(b'\\');
        expected.push(b')');
        let out_bytes = {
            let mut out = Vec::new();
            write_object(&mut out, &Object::new_string(input));
            out
        };
        assert_eq!(out_bytes, expected);

        // A non-printable byte (0x01) becomes a three-digit octal escape.
        let out_bytes = {
            let mut out = Vec::new();
            write_object(&mut out, &Object::new_string(vec![0x01]));
            out
        };
        assert_eq!(out_bytes, b"(\\001)");

        // Newline/CR/tab get their short mnemonic escapes, not octal.
        let out_bytes = {
            let mut out = Vec::new();
            write_object(&mut out, &Object::new_string(b"a\nb\rc\td".to_vec()));
            out
        };
        assert_eq!(out_bytes, b"(a\\nb\\rc\\td)");
    }

    #[test]
    fn escapes_name_specials() {
        // space -> #20, '#' -> #23, '/' -> #2F; everything else literal.
        assert_eq!(
            ser(&Object::new_name(b"A B#C/D".to_vec())),
            "/A#20B#23C#2FD"
        );
        // A perfectly ordinary name round-trips with no escaping at all.
        assert_eq!(ser(&Object::new_name(b"Type".to_vec())), "/Type");
    }

    #[test]
    fn array_and_dict_insert_separators_only_where_needed() {
        // Two adjacent integers need a space (else "12" would re-lex as one
        // token); a name's leading '/' is self-delimiting so it needs none.
        let arr = Object::Array(vec![
            Object::new_int(1),
            Object::new_int(2),
            Object::new_name(b"X".to_vec()),
        ]);
        assert_eq!(ser(&arr), "[1 2/X]");

        // "/Type/Test" needs no space ('/' delimits); "/N" then "5" does
        // (else "N5" re-lexes as one name).
        let dict = Object::Dict(vec![
            (b"Type".to_vec(), Object::new_name(b"Test".to_vec())),
            (b"N".to_vec(), Object::new_int(5)),
        ]);
        assert_eq!(ser(&dict), "<</Type/Test/N 5>>");
    }

    // -----------------------------------------------------------------------
    // incremental_update: round-trip through this crate's own reader.
    // -----------------------------------------------------------------------

    const FIXTURE_NO_AP: &[u8] = include_bytes!("../../tests/fixtures/ink-annots-no-ap.pdf");
    const FIXTURE_MIXED_AP: &[u8] = include_bytes!("../../tests/fixtures/ink-annots-mixed-ap.pdf");

    /// Both fixtures use a classic `xref` table as their only section (no
    /// `/Type /XRef` stream) -- confirmed by inspection, and asserted here so
    /// a future fixture swap does not silently stop exercising the classic
    /// path this writer implements.
    #[test]
    fn fixtures_are_classic_xref() {
        for bytes in [FIXTURE_NO_AP, FIXTURE_MIXED_AP] {
            let tail = String::from_utf8_lossy(&bytes[bytes.len().saturating_sub(200)..]);
            assert!(tail.contains("xref"), "fixture should have a classic xref");
            assert!(
                !tail.contains("/Type/XRef") && !tail.contains("/Type /XRef"),
                "fixture should not be a cross-reference stream"
            );
        }
    }

    #[test]
    fn append_new_and_supersede_existing_object_round_trips() {
        let doc = PdfDocument::open(FIXTURE_NO_AP.to_vec()).expect("fixture must open");
        let next = next_object_number(&doc);
        assert_eq!(next, 8, "fixture trailer declares /Size 8");

        // Supersede object 5 (the blue ink annot) with an unrelated marker
        // dict, and add a brand-new stream object at the next free number.
        // 5 and 8 are non-contiguous -- this exercises the two-subsection
        // grouping path, not just a single contiguous run.
        let new_five = Object::Dict(vec![(b"KopitiamTest".to_vec(), Object::new_int(999))]);
        let stream_dict = Object::Dict(vec![(
            b"Type".to_vec(),
            Object::new_name(b"KopitiamTest".to_vec()),
        )]);
        let updates = vec![
            (5, NewObject::Plain(new_five.clone())),
            (
                next,
                NewObject::Stream {
                    dict: stream_dict,
                    data: b"hello world".to_vec(),
                },
            ),
        ];

        let updated = incremental_update(&doc, &updates).expect("incremental_update");
        assert!(updated.len() > FIXTURE_NO_AP.len());
        assert_eq!(
            &updated[..FIXTURE_NO_AP.len()],
            FIXTURE_NO_AP,
            "original bytes must be untouched, only appended to"
        );

        let doc2 = PdfDocument::open(updated.clone()).expect("updated file must reopen");

        // The superseded object resolves to the NEW value, not the old ink
        // annot.
        let resolved_five = doc2
            .resolve(&Object::new_indirect(5, 0))
            .expect("resolve object 5");
        assert_eq!(resolved_five, new_five);

        // The brand-new object resolves too, dict and stream body both.
        let resolved_new = doc2
            .resolve(&Object::new_indirect(next as i64, 0))
            .expect("resolve new object");
        assert_eq!(
            resolved_new.dict_gets("Type"),
            Some(&Object::new_name(b"KopitiamTest".to_vec()))
        );
        assert_eq!(
            resolved_new.dict_gets("Length"),
            Some(&Object::new_int(11)),
            "/Length must be set to the raw data length"
        );
        let body = doc2
            .open_stream_num(next)
            .expect("new stream object must decode (no filter -> identity)");
        assert_eq!(body, b"hello world");

        // Everything NOT touched by this update still resolves exactly as
        // it did before -- the catalog, and the page tree/page count.
        let root_before = doc
            .resolve(&Object::new_indirect(1, 0))
            .expect("resolve original catalog");
        let root_after = doc2
            .resolve(&Object::new_indirect(1, 0))
            .expect("resolve catalog in updated doc");
        assert_eq!(root_before, root_after);
        assert_eq!(doc2.page_count(), doc.page_count());

        // /Size grew to cover the new object; next_object_number reflects it.
        assert_eq!(next_object_number(&doc2), (next + 1).max(8));

        // Undo: truncating back to the original length must still open, and
        // must resolve object 5 back to its ORIGINAL (pre-update) value.
        let truncated = updated[..FIXTURE_NO_AP.len()].to_vec();
        assert_eq!(truncated, FIXTURE_NO_AP);
        let doc3 = PdfDocument::open(truncated).expect("truncated file must still open");
        let original_five = doc
            .resolve(&Object::new_indirect(5, 0))
            .expect("resolve original object 5 in the pristine doc");
        let five_after_undo = doc3
            .resolve(&Object::new_indirect(5, 0))
            .expect("resolve object 5 after undo");
        assert_eq!(five_after_undo, original_five);
        assert_ne!(five_after_undo, new_five);
    }

    #[test]
    fn empty_updates_is_a_no_op() {
        let doc = PdfDocument::open(FIXTURE_MIXED_AP.to_vec()).expect("fixture must open");
        let out = incremental_update(&doc, &[]).expect("no-op update");
        assert_eq!(out, FIXTURE_MIXED_AP);
    }

    /// Cross-reader check: an incrementally-updated file must still render in
    /// poppler (`pdftoppm`), not merely in this crate's own parser. Skips
    /// (rather than fails) if `pdftoppm` is not on `PATH`, so this remains
    /// safe to run in an environment without poppler installed -- but in the
    /// environment this was authored in, poppler IS installed, and this test
    /// is expected to actually exercise it every run.
    #[test]
    fn poppler_can_render_the_updated_file() {
        let doc = PdfDocument::open(FIXTURE_MIXED_AP.to_vec()).expect("fixture must open");
        let next = next_object_number(&doc);
        let updates = vec![(
            next,
            NewObject::Plain(Object::Dict(vec![(
                b"KopitiamTest".to_vec(),
                Object::new_int(1),
            )])),
        )];
        let updated = incremental_update(&doc, &updates).expect("incremental_update");

        let dir = std::env::temp_dir().join(format!(
            "kopitiam-pdf-write-test-{}-{}",
            std::process::id(),
            "poppler-check"
        ));
        let _ = std::fs::create_dir_all(&dir);
        let pdf_path = dir.join("updated.pdf");
        std::fs::write(&pdf_path, &updated).expect("write temp pdf");
        let prefix = dir.join("out");

        let result = std::process::Command::new("pdftoppm")
            .arg("-r")
            .arg("72")
            .arg("-png")
            .arg(&pdf_path)
            .arg(&prefix)
            .output();

        match result {
            Ok(output) => {
                assert!(
                    output.status.success(),
                    "pdftoppm failed: stdout={:?} stderr={:?}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
                let produced_png = std::fs::read_dir(&dir)
                    .expect("read temp dir")
                    .filter_map(|e| e.ok())
                    .any(|e| e.path().extension().is_some_and(|ext| ext == "png"));
                assert!(produced_png, "pdftoppm exited 0 but produced no PNG");
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                eprintln!("skipping poppler cross-check: pdftoppm not on PATH ({e})");
            }
            Err(e) => panic!("failed to run pdftoppm: {e}"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}

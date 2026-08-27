//! Ported from MuPDF `source/pdf/pdf-parse.c` (commit 19f1284, AGPL-3.0,
//! © Artifex Software, Inc.), translated to Rust for KOPITIAM (AGPL-3.0-only).
//! Close adaptation: the algorithms and numeric behaviour follow MuPDF; the code
//! is re-expressed in idiomatic Rust. See docs/ACKNOWLEDGEMENTS.md ("PDF &
//! document-extraction references").
//!
//! # From tokens to [`Object`]s
//!
//! This is the recursive-descent parser sitting on top of [`lex`]. It builds
//! [`Object`] trees: [`parse_array`] (`[ … ]`), [`parse_dict`] (`<< /K v … >>`),
//! scalars, indirect references (`N G R`), and the `N G obj … endobj`
//! indirect-object form ([`parse_ind_obj`]).
//!
//! MuPDF's parser is entangled with the *document* (`pdf_document *doc`): new
//! arrays/dicts are bound to it, and indirect references carry a `doc` pointer.
//! This port drops that binding -- [`Object`]s are free-standing (see
//! `object.rs`) -- so the parse functions take only a [`Stream`].
//!
//! ## The stream-body seam
//!
//! `pdf_parse_ind_obj` does not extract stream data. When it meets the `stream`
//! keyword it consumes the whitespace up to the start of the raw bytes, records
//! that byte offset (`stm_ofs`), and returns -- leaving the actual reading,
//! `/Length` resolution, and filter decoding to the xref/stream layer. This port
//! keeps exactly that seam: [`parse_ind_obj`] returns an [`IndirectObject`] whose
//! [`stream`](IndirectObject::stream) records the start offset via
//! [`StreamRange`]. The end of the body is `start + /Length` (resolved later),
//! so nothing here decodes or bounds the stream body.
//!
//! ## Recovery vs. hard errors
//!
//! MuPDF recovers cheaply where it can (a stray/unknown token inside an array or
//! as a dict value becomes `PDF_NULL`; a malformed indirect reference in a dict
//! is warned and stored as null). Those recoveries are preserved. A genuine
//! structural fault -- an unclosed array at EOF, a non-name dict key, a bad
//! object header, an `R` in the wrong place -- is `fz_throw(FZ_ERROR_SYNTAX)`,
//! which here is `Err(Error::syntax(..))`.
//!
//! ## Not ported (deliberately)
//!
//! `pdf_parse_journal_obj` (journalling), the `pdf_parse_ind_obj_or_newobj`
//! `newobj` path (incremental-update authoring), and the date / rect / matrix /
//! text-string helpers (`pdf_to_date`, `pdf_new_utf8_from_pdf_string`, …) are not
//! on the object-model read path and are left for later modules.

use super::error::{Error, Result};
use super::lex::{Token, lex};
use super::object::{MAX_OBJECT_NUMBER, Object};
use super::stream::Stream;

/// Where a stream object's raw body begins, recorded by [`parse_ind_obj`] and
/// resolved (bounded and decoded) later by the xref/stream layer.
///
/// This is MuPDF's `stm_ofs`: the absolute offset of the first raw byte after
/// the `stream` keyword and its trailing end-of-line. The body length is
/// `/Length` from the object's dict, deferred to the xref layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamRange {
    /// Absolute byte offset of the first byte of the stream body.
    pub start: i64,
}

/// The result of [`parse_ind_obj`]: an `N G obj … endobj` indirect object.
#[derive(Clone, Debug, PartialEq)]
pub struct IndirectObject {
    /// The object number `N`.
    pub num: i32,
    /// The generation number `G` (spelled `generation`: `gen` is a reserved
    /// keyword in Rust edition 2024).
    pub generation: i32,
    /// The parsed object (for a stream, this is the stream's dict).
    pub object: Object,
    /// For a stream object, where its raw body begins; `None` otherwise.
    pub stream: Option<StreamRange>,
}

// MuPDF: pdf_parse_array (pdf-parse.c:555).
/// Parse an array body, having already consumed the opening `[`.
///
/// Reproduces MuPDF's deferred-integer machinery: a run of integers may turn out
/// to be an indirect reference (`N G R`), so pending ints are held in `a`/`b`
/// (count `n`) and only committed as integer objects once it is clear they are
/// not the head of a reference.
pub fn parse_array(f: &mut Stream) -> Result<Object> {
    let mut ary = Object::new_array();
    let mut a: i64 = 0;
    let mut b: i64 = 0;
    let mut n: i32 = 0;

    loop {
        let tok = lex(f)?;

        // Flush pending ints when the token cannot continue a `N G R` reference.
        if !matches!(tok, Token::Int(_) | Token::R) {
            if n > 0 {
                ary.array_push(Object::new_int(a));
            }
            if n > 1 {
                ary.array_push(Object::new_int(b));
            }
            n = 0;
        }

        // A third integer arrives while two are pending: the first cannot be part
        // of a reference, so commit it and shift.
        if matches!(tok, Token::Int(_)) && n == 2 {
            ary.array_push(Object::new_int(a));
            a = b;
            n -= 1;
        }

        match tok {
            Token::Eof => {
                return Err(Error::syntax("array not closed before end of file"));
            }
            Token::CloseArray => break,
            Token::Int(i) => {
                if n == 0 {
                    a = i;
                }
                if n == 1 {
                    b = i;
                }
                n += 1;
            }
            Token::R => {
                if n != 2 {
                    return Err(Error::syntax("cannot parse indirect reference in array"));
                }
                ary.array_push(Object::new_indirect(a, b as i32));
                n = 0;
            }
            Token::OpenArray => ary.array_push(parse_array(f)?),
            Token::OpenDict => ary.array_push(parse_dict(f)?),
            Token::Name(bytes) => ary.array_push(Object::new_name(bytes)),
            Token::Real(v) => ary.array_push(Object::new_real(v)),
            Token::String(bytes) => ary.array_push(Object::new_string(bytes)),
            Token::True => ary.array_push(Object::Bool(true)),
            Token::False => ary.array_push(Object::Bool(false)),
            Token::Null => ary.array_push(Object::Null),
            // Any other/unknown token -> null, matching MuPDF's default.
            _ => ary.array_push(Object::Null),
        }
    }

    Ok(ary)
}

// MuPDF: pdf_parse_dict (pdf-parse.c:659).
/// Parse a dict body, having already consumed the opening `<<`.
pub fn parse_dict(f: &mut Stream) -> Result<Object> {
    let mut dict = Object::new_dict();
    // A token may be pre-loaded by the integer-value lookahead below (MuPDF's
    // `goto skip`).
    let mut pending: Option<Token> = None;

    loop {
        let tok = match pending.take() {
            Some(t) => t,
            None => lex(f)?,
        };

        if tok == Token::CloseDict {
            break;
        }
        // For BI .. ID .. EI in content streams: an "ID" keyword ends the dict.
        if let Token::Keyword(ref k) = tok
            && k == b"ID"
        {
            break;
        }

        let key: Vec<u8> = match tok {
            Token::Name(bytes) => bytes,
            _ => return Err(Error::syntax("invalid key in dict")),
        };

        let vtok = lex(f)?;
        let val = match vtok {
            Token::OpenArray => parse_array(f)?,
            Token::OpenDict => parse_dict(f)?,
            Token::Name(bytes) => Object::new_name(bytes),
            Token::Real(v) => Object::new_real(v),
            Token::String(bytes) => Object::new_string(bytes),
            Token::True => Object::Bool(true),
            Token::False => Object::Bool(false),
            Token::Null => Object::Null,
            Token::Int(a) => {
                // Could be a plain int, or the head of `a b R`.
                let t2 = lex(f)?;
                let is_id = matches!(&t2, Token::Keyword(k) if k == b"ID");
                if t2 == Token::CloseDict || matches!(t2, Token::Name(_)) || is_id {
                    // Plain integer value; re-use t2 as the next iteration's token.
                    dict.dict_put(key, Object::new_int(a));
                    pending = Some(t2);
                    continue;
                }
                if let Token::Int(b) = t2 {
                    let t3 = lex(f)?;
                    if t3 == Token::R {
                        Object::new_indirect(a, b as i32)
                    } else {
                        // "invalid indirect reference in dict" (warn) -> null.
                        Object::Null
                    }
                } else {
                    Object::Null
                }
            }
            // Any other/unknown token -> null.
            _ => Object::Null,
        };

        dict.dict_put(key, val);
    }

    Ok(dict)
}

// MuPDF: pdf_parse_stm_obj (pdf-parse.c:758). Parse one object from an object
// stream (no indirect references, no `obj`/`endobj` wrapper).
/// Parse a single object as it appears inside an object stream (`/ObjStm`).
pub fn parse_stm_obj(f: &mut Stream) -> Result<Object> {
    let tok = lex(f)?;
    match tok {
        Token::OpenArray => parse_array(f),
        Token::OpenDict => parse_dict(f),
        Token::Name(bytes) => Ok(Object::new_name(bytes)),
        Token::Real(v) => Ok(Object::new_real(v)),
        Token::String(bytes) => Ok(Object::new_string(bytes)),
        Token::True => Ok(Object::Bool(true)),
        Token::False => Ok(Object::Bool(false)),
        Token::Null => Ok(Object::Null),
        Token::Int(i) => Ok(Object::new_int(i)),
        _ => Err(Error::syntax("unknown token in object stream")),
    }
}

// MuPDF: pdf_parse_ind_obj / pdf_parse_ind_obj_or_newobj (pdf-parse.c:931, 782).
/// Parse an `N G obj … endobj` indirect object.
///
/// Returns the object number/generation, the parsed object (the dict, for a
/// stream), and -- if the object is a stream -- the recorded start offset of its
/// raw body (see [`StreamRange`] and the module docs on the stream seam). The
/// `newobj` incremental-update path is not ported.
pub fn parse_ind_obj(f: &mut Stream) -> Result<IndirectObject> {
    // Object number.
    let num = match lex(f)? {
        Token::Int(i) => i,
        _ => return Err(Error::syntax("expected object number")),
    };
    if num < 0 || num > MAX_OBJECT_NUMBER as i64 {
        return Err(Error::syntax("object number out of range"));
    }
    let num = num as i32;

    // Generation number.
    let generation = match lex(f)? {
        Token::Int(i) => i,
        _ => {
            return Err(Error::syntax(format!(
                "expected generation number ({num} ? obj)"
            )));
        }
    };
    if !(0..65536).contains(&generation) {
        return Err(Error::syntax(format!(
            "invalid generation number ({generation})"
        )));
    }
    let generation = generation as i32;

    // 'obj' keyword.
    match lex(f)? {
        Token::Obj => {}
        _ => {
            return Err(Error::syntax(format!(
                "expected 'obj' keyword ({num} {generation} ?)"
            )));
        }
    }

    // The object body. `read_next_token` mirrors MuPDF: some branches have
    // already consumed the token that follows the object (STREAM/ENDOBJ), so the
    // post-object phase must not lex again.
    let mut read_next_token = true;
    let mut post_tok = Token::Error; // valid only when read_next_token == false

    let obj: Object = match lex(f)? {
        Token::OpenArray => parse_array(f)?,
        Token::OpenDict => parse_dict(f)?,
        Token::Name(bytes) => Object::new_name(bytes),
        Token::Real(v) => Object::new_real(v),
        Token::String(bytes) => Object::new_string(bytes),
        Token::True => Object::Bool(true),
        Token::False => Object::Bool(false),
        Token::Null => Object::Null,
        Token::Int(a) => {
            let t2 = lex(f)?;
            if t2 == Token::Stream || t2 == Token::EndObj {
                read_next_token = false;
                post_tok = t2;
                Object::new_int(a)
            } else if let Token::Int(b) = t2 {
                if lex(f)? == Token::R {
                    Object::new_indirect(a, b as i32)
                } else {
                    return Err(Error::syntax(format!(
                        "expected 'R' keyword ({num} {generation} R)"
                    )));
                }
            } else {
                return Err(Error::syntax(format!(
                    "expected 'R' keyword ({num} {generation} R)"
                )));
            }
        }
        Token::EndObj => {
            read_next_token = false;
            post_tok = Token::EndObj;
            Object::Null
        }
        _ => {
            return Err(Error::syntax(format!(
                "syntax error in object ({num} {generation} R)"
            )));
        }
    };

    // Post-object phase: locate `stream`/`endobj` and, for a stream, the body.
    let tok = if read_next_token { lex(f)? } else { post_tok };

    let stream = if tok == Token::Stream {
        // Skip spaces, then the CR/LF that must follow the `stream` keyword.
        let mut c = f.read_byte()?;
        while c == Some(b' ') {
            c = f.read_byte()?;
        }
        if c == Some(b'\r') {
            // A LF should follow; consume it if present (MuPDF warns if absent).
            if f.peek_byte()? == Some(b'\n') {
                f.read_byte()?;
            }
        }
        // Note: if `c` was already '\n' it has been consumed; if it was the first
        // real body byte, MuPDF's fz_tell counts from here too -- but the common,
        // spec-conformant case is a CRLF/LF terminator, handled above.
        Some(StreamRange { start: f.tell() })
    } else {
        // ENDOBJ (or, leniently, anything else) -> no stream body.
        None
    };

    Ok(IndirectObject {
        num,
        generation,
        object: obj,
        stream,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse a single top-level object (array/dict/scalar) from `input`.
    fn parse_one(input: &[u8]) -> Result<Object> {
        let mut s = Stream::from_slice(input);
        match lex(&mut s)? {
            Token::OpenArray => parse_array(&mut s),
            Token::OpenDict => parse_dict(&mut s),
            Token::Int(i) => Ok(Object::new_int(i)),
            Token::Real(v) => Ok(Object::new_real(v)),
            Token::Name(b) => Ok(Object::new_name(b)),
            Token::String(b) => Ok(Object::new_string(b)),
            Token::True => Ok(Object::Bool(true)),
            Token::False => Ok(Object::Bool(false)),
            Token::Null => Ok(Object::Null),
            other => panic!("unexpected leading token: {other:?}"),
        }
    }

    #[test]
    fn parse_page_dict() {
        // << /Type /Page /Count 3 /Kids [4 0 R 5 0 R] >>
        let d = parse_one(b"<< /Type /Page /Count 3 /Kids [4 0 R 5 0 R] >>").unwrap();
        assert!(d.is_dict());
        assert_eq!(d.dict_len(), 3);
        assert_eq!(d.dict_gets("Type").unwrap().to_name(), b"Page");
        assert_eq!(d.dict_gets("Count").unwrap().to_int(), 3);

        let kids = d.dict_gets("Kids").unwrap();
        assert!(kids.is_array());
        assert_eq!(kids.array_len(), 2);
        // Two indirect references.
        let r0 = kids.array_get(0).unwrap();
        assert!(r0.is_indirect());
        assert_eq!(r0.to_num(), 4);
        assert_eq!(r0.to_gen(), 0);
        let r1 = kids.array_get(1).unwrap();
        assert_eq!(r1.to_num(), 5);
    }

    #[test]
    fn parse_nested_array() {
        // [1 [2 3] [/A (b)] 4]
        let a = parse_one(b"[1 [2 3] [/A (b)] 4]").unwrap();
        assert!(a.is_array());
        assert_eq!(a.array_len(), 4);
        assert_eq!(a.array_get(0).unwrap().to_int(), 1);

        let inner = a.array_get(1).unwrap();
        assert!(inner.is_array());
        assert_eq!(inner.array_len(), 2);
        assert_eq!(inner.array_get(1).unwrap().to_int(), 3);

        let inner2 = a.array_get(2).unwrap();
        assert_eq!(inner2.array_get(0).unwrap().to_name(), b"A");
        assert_eq!(inner2.array_get(1).unwrap().to_string_bytes(), b"b");

        assert_eq!(a.array_get(3).unwrap().to_int(), 4);
    }

    #[test]
    fn array_mixes_ints_and_references() {
        // Trailing plain ints after a reference must be flushed, not swallowed.
        // [1 2 3 R 7 8]  -> [ int:1, ref(2,3), int:7, int:8 ]
        let a = parse_one(b"[1 2 3 R 7 8]").unwrap();
        assert_eq!(a.array_len(), 4);
        assert_eq!(a.array_get(0).unwrap().to_int(), 1);
        let r = a.array_get(1).unwrap();
        assert!(r.is_indirect());
        assert_eq!(r.to_num(), 2);
        assert_eq!(r.to_gen(), 3);
        assert_eq!(a.array_get(2).unwrap().to_int(), 7);
        assert_eq!(a.array_get(3).unwrap().to_int(), 8);
    }

    #[test]
    fn dict_int_value_vs_indirect_value() {
        // /A 3 is a plain int; /B 5 0 R is a reference; both in one dict.
        let d = parse_one(b"<< /A 3 /B 5 0 R >>").unwrap();
        assert_eq!(d.dict_gets("A").unwrap().to_int(), 3);
        let b = d.dict_gets("B").unwrap();
        assert!(b.is_indirect());
        assert_eq!(b.to_num(), 5);
        assert_eq!(b.to_gen(), 0);
    }

    #[test]
    fn parse_indirect_object_with_stream() {
        // 12 0 obj << /Length 5 >> stream\nHELLO\nendstream endobj
        let input = b"12 0 obj << /Length 5 >> stream\nHELLOendstream endobj";
        let mut s = Stream::from_slice(input);
        let ind = parse_ind_obj(&mut s).unwrap();

        assert_eq!(ind.num, 12);
        assert_eq!(ind.generation, 0);
        assert!(ind.object.is_dict());
        assert_eq!(ind.object.dict_gets("Length").unwrap().to_int(), 5);

        // The recorded stream body starts right after "stream\n".
        let start = ind.stream.expect("stream range recorded").start;
        let prefix = b"12 0 obj << /Length 5 >> stream\n";
        assert_eq!(start, prefix.len() as i64);
        // And the /Length bytes at that offset are the body.
        assert_eq!(&input[start as usize..start as usize + 5], b"HELLO");
    }

    #[test]
    fn parse_indirect_object_without_stream() {
        // 3 0 obj << /Type /Catalog >> endobj
        let mut s = Stream::from_slice(b"3 0 obj << /Type /Catalog >> endobj");
        let ind = parse_ind_obj(&mut s).unwrap();
        assert_eq!(ind.num, 3);
        assert_eq!(ind.generation, 0);
        assert_eq!(ind.object.dict_gets("Type").unwrap().to_name(), b"Catalog");
        assert!(ind.stream.is_none());
    }

    #[test]
    fn parse_scalar_indirect_object() {
        // 7 0 obj 42 endobj  -> a bare integer object.
        let mut s = Stream::from_slice(b"7 0 obj 42 endobj");
        let ind = parse_ind_obj(&mut s).unwrap();
        assert_eq!(ind.num, 7);
        assert_eq!(ind.object, Object::new_int(42));
        assert!(ind.stream.is_none());
    }

    #[test]
    fn malformed_inputs_return_err() {
        // Unclosed array at EOF.
        let mut s = Stream::from_slice(b"[1 2 3");
        assert_eq!(
            parse_array(&mut s).unwrap_err().kind(),
            crate::mupdf::ErrorKind::Syntax
        );

        // Non-name dict key.
        let mut s = Stream::from_slice(b"42 /V >>");
        assert_eq!(
            parse_dict(&mut s).unwrap_err().kind(),
            crate::mupdf::ErrorKind::Syntax
        );

        // Object header without a number.
        let mut s = Stream::from_slice(b"obj << >> endobj");
        assert!(parse_ind_obj(&mut s).is_err());

        // 'R' where a reference cannot be formed.
        let mut s = Stream::from_slice(b"[1 R]");
        assert!(parse_array(&mut s).is_err());
    }

    #[test]
    fn parse_stm_obj_scalars() {
        let mut s = Stream::from_slice(b"/Name");
        assert_eq!(parse_stm_obj(&mut s).unwrap().to_name(), b"Name");
        let mut s = Stream::from_slice(b"true");
        assert!(parse_stm_obj(&mut s).unwrap().to_bool());
        // An `R` keyword is not valid inside an object stream.
        let mut s = Stream::from_slice(b"R");
        assert!(parse_stm_obj(&mut s).is_err());
    }
}

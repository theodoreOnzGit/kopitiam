//! Ported from MuPDF `source/pdf/pdf-lex.c` (+ the `pdf_token` enum from
//! `include/mupdf/pdf/parse.h`) (commit 19f1284, AGPL-3.0, © Artifex Software,
//! Inc.), translated to Rust for KOPITIAM (AGPL-3.0-only). Close adaptation: the
//! algorithms and numeric behaviour follow MuPDF; the code is re-expressed in
//! idiomatic Rust. See docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction
//! references").
//!
//! # The tokenizer
//!
//! [`lex`] is `pdf_lex`: it reads one token from a [`Stream`] and returns it as a
//! [`Token`] (MuPDF's `pdf_token` plus its lexed value, which C stashes in a
//! separate `pdf_lexbuf`). It preserves MuPDF's exact classification of
//! whitespace and delimiters, its number parsing (including Acrobat's handling of
//! broken numbers), name `#xx` hex escapes, literal-string escapes / nested
//! parens / octal / line-continuation, hex strings, and the keyword table.
//!
//! ## One representational change: pushback becomes peek
//!
//! MuPDF's lexer reads a byte with `fz_read_byte` and, when it over-reads (e.g.
//! hits a delimiter that ends a number), pushes it back with `fz_unread_byte`
//! (which just decrements the stream's read pointer). This port has no
//! `unread_byte`; instead every over-read is expressed as a
//! [`Stream::peek_byte`] that leaves the byte in place, consuming it with
//! [`Stream::read_byte`] only once it is known to belong to the current token.
//! The observable token stream is identical. Where MuPDF reads-then-unreads a
//! byte that C code then re-lexes, this port either leaves it unconsumed (peek)
//! or, for a byte already consumed as part of the token, appends it directly --
//! the net effect is the same because such bytes are always ordinary regular
//! characters.
//!
//! ## Errors
//!
//! MuPDF returns `PDF_TOK_ERROR` as an ordinary token value for a malformed
//! token (an unbalanced `)`, a lone `>`, a string/hex-string cut off by EOF) and
//! lets the *parser* decide what to do. This port keeps that: [`lex`] returns
//! `Ok(Token::Error)` in those cases rather than `Err` -- only a genuine read
//! error on the underlying [`Stream`] propagates as `Err`.

use super::error::Result;
use super::stream::Stream;

/// The maximum length MuPDF allows for a name (`PDF_MAX_NAME_LENGTH`). Longer
/// names are truncated with a warning.
const MAX_NAME_LENGTH: usize = 4095;

// MuPDF: enum pdf_token (parse.h:28) fused with the lexed value (pdf_lexbuf).
/// A lexed PDF token. Variants that carry data hold the value MuPDF would place
/// in the `pdf_lexbuf` alongside the bare `pdf_token`.
#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    /// `PDF_TOK_ERROR` -- a malformed token (handled by the parser, not thrown).
    Error,
    /// `PDF_TOK_EOF` -- end of the stream.
    Eof,
    /// `PDF_TOK_OPEN_ARRAY` -- `[`.
    OpenArray,
    /// `PDF_TOK_CLOSE_ARRAY` -- `]`.
    CloseArray,
    /// `PDF_TOK_OPEN_DICT` -- `<<`.
    OpenDict,
    /// `PDF_TOK_CLOSE_DICT` -- `>>`.
    CloseDict,
    /// `PDF_TOK_OPEN_BRACE` -- `{`.
    OpenBrace,
    /// `PDF_TOK_CLOSE_BRACE` -- `}`.
    CloseBrace,
    /// `PDF_TOK_NAME` -- a `/Name`, carrying its `#xx`-decoded bytes.
    Name(Vec<u8>),
    /// `PDF_TOK_INT` -- an integer.
    Int(i64),
    /// `PDF_TOK_REAL` -- a real number.
    Real(f64),
    /// `PDF_TOK_STRING` -- a literal or hex string, carrying its decoded bytes.
    String(Vec<u8>),
    /// `PDF_TOK_KEYWORD` -- an unrecognised bareword, carrying its bytes.
    Keyword(Vec<u8>),
    /// `PDF_TOK_R` -- the `R` indirect-reference keyword.
    R,
    /// `PDF_TOK_TRUE` -- `true`.
    True,
    /// `PDF_TOK_FALSE` -- `false`.
    False,
    /// `PDF_TOK_NULL` -- `null`.
    Null,
    /// `PDF_TOK_OBJ` -- `obj`.
    Obj,
    /// `PDF_TOK_ENDOBJ` -- `endobj`.
    EndObj,
    /// `PDF_TOK_STREAM` -- `stream`.
    Stream,
    /// `PDF_TOK_ENDSTREAM` -- `endstream`.
    EndStream,
    /// `PDF_TOK_XREF` -- `xref`.
    Xref,
    /// `PDF_TOK_TRAILER` -- `trailer`.
    Trailer,
    /// `PDF_TOK_STARTXREF` -- `startxref`.
    StartXref,
    /// `PDF_TOK_NEWOBJ` -- `newobj`.
    NewObj,
}

// MuPDF: iswhite (pdf-lex.c:72) -- the PDF whitespace set.
#[inline]
fn iswhite(ch: u8) -> bool {
    matches!(ch, b'\x00' | b'\x09' | b'\x0a' | b'\x0c' | b'\x0d' | b'\x20')
}

// MuPDF: IS_DELIM (pdf-lex.c:40) -- the PDF delimiter set.
#[inline]
fn isdelim(ch: u8) -> bool {
    matches!(
        ch,
        b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'/' | b'%'
    )
}

// MuPDF: fz_isprint (pdf-lex.c:83).
#[inline]
fn isprint(ch: u8) -> bool {
    (b' '..=b'~').contains(&ch)
}

// MuPDF: unhex (pdf-lex.c:88).
#[inline]
fn unhex(ch: u8) -> i32 {
    match ch {
        b'0'..=b'9' => (ch - b'0') as i32,
        b'A'..=b'F' => (ch - b'A') as i32 + 0xA,
        b'a'..=b'f' => (ch - b'a') as i32 + 0xA,
        _ => 0,
    }
}

#[inline]
fn is_hex(ch: u8) -> bool {
    ch.is_ascii_hexdigit()
}

// MuPDF: lex_white (pdf-lex.c:96) -- skip a run of whitespace, leaving the next
// non-white byte in the stream. Expressed with peek so nothing is unread.
fn lex_white(f: &mut Stream) -> Result<()> {
    while let Some(c) = f.peek_byte()? {
        if iswhite(c) {
            f.read_byte()?;
        } else {
            break;
        }
    }
    Ok(())
}

// MuPDF: lex_comment (pdf-lex.c:107) -- consume through end of line (or EOF).
fn lex_comment(f: &mut Stream) -> Result<()> {
    loop {
        match f.read_byte()? {
            None | Some(b'\n') | Some(b'\r') => break,
            _ => {}
        }
    }
    Ok(())
}

// MuPDF: acrobat_compatible_atof (pdf-lex.c:117). Fast(ish) but inaccurate
// strtof with Adobe overflow handling. Operates on the bytes up to the first NUL
// (matching the C string semantics of the scratch buffer).
fn acrobat_compatible_atof(s: &[u8]) -> f64 {
    let mut i = 0usize;
    let mut neg = false;
    let mut idx = 0;
    let n = s.len();
    while idx < n && s[idx] == b'-' {
        neg = true;
        idx += 1;
    }
    while idx < n && s[idx] == b'+' {
        idx += 1;
    }
    // We deliberately ignore integer overflow here, exactly as Acrobat does:
    // 123450000000000000000678 is read as 678 (wrapping in a 32-bit unsigned).
    while idx < n && s[idx].is_ascii_digit() {
        i = i.wrapping_mul(10).wrapping_add((s[idx] - b'0') as usize);
        i &= 0xffff_ffff;
        idx += 1;
    }
    if idx < n && s[idx] == b'.' {
        let mut v = i as f64;
        let mut frac = 0.0f64;
        let mut d = 1.0f64;
        idx += 1;
        while idx < n && s[idx].is_ascii_digit() {
            frac = 10.0 * frac + (s[idx] - b'0') as f64;
            d *= 10.0;
            idx += 1;
        }
        v += frac / d;
        if neg {
            -v
        } else {
            v
        }
    } else if neg {
        -(i as f64)
    } else {
        i as f64
    }
}

// MuPDF: fast_atoi (pdf-lex.c:164). Fast but inaccurate atoi; ignores overflow
// exactly as MuPDF does (wrapping in a 64-bit accumulator).
fn fast_atoi(s: &[u8]) -> i64 {
    let mut neg = false;
    let mut i = 0u64;
    let mut idx = 0;
    let n = s.len();
    while idx < n && s[idx] == b'-' {
        neg = true;
        idx += 1;
    }
    while idx < n && s[idx] == b'+' {
        idx += 1;
    }
    while idx < n && s[idx].is_ascii_digit() {
        i = i.wrapping_mul(10).wrapping_add((s[idx] - b'0') as u64);
        idx += 1;
    }
    if neg {
        (i as i64).wrapping_neg()
    } else {
        i as i64
    }
}

// MuPDF: fz_atof (string.c:860) for the accurate real path. Rust's f32 parser is
// correctly-rounded like fz_strtof; the NaN/underflow -> 1.0 and the clamp to
// the f32 range reproduce fz_atof's guards. Parses the bytes up to the first NUL.
fn fz_atof(s: &[u8]) -> f64 {
    let end = s.iter().position(|&b| b == 0).unwrap_or(s.len());
    let text = core::str::from_utf8(&s[..end]).unwrap_or("");
    match text.parse::<f32>() {
        Ok(v) if v.is_nan() => 1.0,
        // Underflow to 0 -> return 1.0, as fz_atof does.
        Ok(v) if v == 0.0 && !text.trim_start_matches(['+', '-']).starts_with('0') => 1.0,
        Ok(v) => v.clamp(f32::MIN, f32::MAX) as f64,
        Err(_) => 0.0,
    }
}

// MuPDF: lex_number (pdf-lex.c:189). `first` is the byte that triggered the
// number branch (a sign, dot, or digit). Returns the token; broken numbers
// become PDF_TOK_KEYWORD, matching Acrobat.
fn lex_number(f: &mut Stream, first: u8) -> Result<Token> {
    let mut s: Vec<u8> = Vec::with_capacity(32);
    // isreal holds the index of the first '.' seen (Some), if any.
    let mut isreal: Option<usize> = if first == b'.' { Some(0) } else { None };
    let neg = first == b'-';
    let mut isbad = false;

    s.push(first);

    let mut c = f.peek_byte()?;

    // Skip extra '-' signs at the start of the number.
    if neg {
        while c == Some(b'-') {
            f.read_byte()?;
            c = f.peek_byte()?;
        }
    }

    loop {
        match c {
            None => break,
            Some(ch) if iswhite(ch) || isdelim(ch) => {
                // Leave the terminator in the stream (MuPDF unreads it).
                break;
            }
            Some(b'.') => {
                f.read_byte()?;
                if isreal.is_some() {
                    isbad = true;
                }
                isreal = Some(s.len());
                s.push(b'.');
            }
            Some(b'-') => {
                // Bug 703248: numbers like 0.000000000000-5684342. Stop the
                // interpretation at the '-' (write a NUL) but keep consuming the
                // trailing digits so they are not lexed later.
                f.read_byte()?;
                s.push(b'\0');
            }
            Some(ch @ b'0'..=b'9') => {
                f.read_byte()?;
                s.push(ch);
            }
            Some(ch) => {
                f.read_byte()?;
                isbad = true;
                s.push(ch);
            }
        }
        c = f.peek_byte()?;
    }

    if isbad {
        // The accumulated bytes become a keyword (C string ends at the first
        // NUL written by the '-' handling).
        let end = s.iter().position(|&b| b == 0).unwrap_or(s.len());
        return Ok(Token::Keyword(s[..end].to_vec()));
    }
    if let Some(dot) = isreal {
        // Match Acrobat's handling of broken numbers: use the fast, overflow-
        // tolerant routine when there are >= 10 characters before the decimal
        // point (the `neg > 1` half of MuPDF's guard is dead code -- `neg` is a
        // 0/1 flag there -- so only the digit-count test can fire).
        let v = if dot >= 10 {
            acrobat_compatible_atof(&s)
        } else {
            fz_atof(&s)
        };
        Ok(Token::Real(v))
    } else {
        Ok(Token::Int(fast_atoi(&s)))
    }
}

// MuPDF: lex_name (pdf-lex.c:269). `seed` is an optional first byte already read
// (used when a keyword's leading regular char was consumed by the dispatcher);
// for a `/Name` it is `None` and the name begins at the next byte. Handles `#xx`
// hex escapes.
fn lex_name(f: &mut Stream, seed: Option<u8>) -> Result<Vec<u8>> {
    let mut s: Vec<u8> = Vec::new();
    if let Some(b) = seed {
        s.push(b);
    }

    loop {
        let c = f.peek_byte()?;
        match c {
            None => break,
            Some(ch) if iswhite(ch) || isdelim(ch) => break,
            Some(b'#') => {
                f.read_byte()?; // consume '#'
                lex_name_hex(f, &mut s)?;
            }
            Some(ch) => {
                f.read_byte()?;
                if s.len() < MAX_NAME_LENGTH {
                    s.push(ch);
                }
                // Names longer than the limit are truncated (MuPDF warns).
            }
        }
    }
    Ok(s)
}

// MuPDF: the `#` case of lex_name (pdf-lex.c:303). The '#' has been consumed.
// Reproduces MuPDF's exact recovery for illegal escapes (including the `#00`
// guard against encoding a NUL) using peek/one-byte lookahead rather than the
// read-then-unread the C uses.
fn lex_name_hex(f: &mut Stream, s: &mut Vec<u8>) -> Result<()> {
    let c1 = f.peek_byte()?;
    match c1 {
        // Illegal at i==0 (non-hex or EOF): emit a literal '#' and leave c1 to be
        // re-lexed as an ordinary character.
        None => {
            push_name_byte(s, b'#');
        }
        Some(h1) if !is_hex(h1) => {
            push_name_byte(s, b'#');
        }
        Some(h1) => {
            f.read_byte()?; // consume first hex digit
            let c2 = f.peek_byte()?;
            match c2 {
                // `#00` guard: second char is '0' and the first decoded to 0.
                Some(b'0') if unhex(h1) == 0 => {
                    // Illegal: '#' literal, then re-lex h1 as ordinary; leave c2.
                    push_name_byte(s, b'#');
                    push_name_byte(s, h1);
                }
                Some(h2) if is_hex(h2) => {
                    f.read_byte()?; // consume second hex digit
                    push_name_byte(s, ((unhex(h1) << 4) + unhex(h2)) as u8);
                }
                None => {
                    // illegal_eof: '#' literal, first digit dropped (MuPDF quirk).
                    push_name_byte(s, b'#');
                }
                Some(_) => {
                    // Illegal: '#' literal, re-lex h1 as ordinary; leave c2.
                    push_name_byte(s, b'#');
                    push_name_byte(s, h1);
                }
            }
        }
    }
    Ok(())
}

#[inline]
fn push_name_byte(s: &mut Vec<u8>, b: u8) {
    if s.len() < MAX_NAME_LENGTH {
        s.push(b);
    }
}

// MuPDF: lex_string (pdf-lex.c:351). Balanced parens, escapes, octal, and the
// PDF 32000-1 rule that a raw \n, \r, or \r\n becomes a single 0x0a.
fn lex_string(f: &mut Stream) -> Result<Token> {
    let mut s: Vec<u8> = Vec::new();
    let mut bal = 1i32;

    loop {
        let c = match f.read_byte()? {
            None => return Ok(Token::Error),
            Some(c) => c,
        };
        match c {
            b'(' => {
                bal += 1;
                s.push(b'(');
            }
            b')' => {
                bal -= 1;
                if bal == 0 {
                    break;
                }
                s.push(b')');
            }
            b'\\' => {
                let e = match f.read_byte()? {
                    None => return Ok(Token::Error),
                    Some(e) => e,
                };
                match e {
                    b'n' => s.push(b'\n'),
                    b'r' => s.push(b'\r'),
                    b't' => s.push(b'\t'),
                    b'b' => s.push(0x08),
                    b'f' => s.push(0x0c),
                    b'(' => s.push(b'('),
                    b')' => s.push(b')'),
                    b'\\' => s.push(b'\\'),
                    b'0'..=b'7' => {
                        let mut oct = (e - b'0') as i32;
                        // Up to two more octal digits.
                        if let Some(d2) = f.peek_byte()? && (b'0'..=b'7').contains(&d2) {
                            f.read_byte()?;
                            oct = oct * 8 + (d2 - b'0') as i32;
                            if let Some(d3) = f.peek_byte()? && (b'0'..=b'7').contains(&d3) {
                                f.read_byte()?;
                                oct = oct * 8 + (d3 - b'0') as i32;
                            }
                        }
                        s.push(oct as u8);
                    }
                    b'\n' => { /* line continuation: swallow */ }
                    b'\r' => {
                        // Swallow \r and an optional following \n.
                        if let Some(nl) = f.peek_byte()? && nl == b'\n' {
                            f.read_byte()?;
                        }
                    }
                    other => s.push(other),
                }
            }
            // Bug 708256: an unescaped \n, \r, or \r\n is a single 0x0a byte.
            b'\n' => s.push(0x0a),
            b'\r' => {
                s.push(0x0a);
                if let Some(nl) = f.peek_byte()? && nl == b'\n' {
                    f.read_byte()?;
                }
            }
            other => s.push(other),
        }
    }
    Ok(Token::String(s))
}

// MuPDF: lex_hex_string (pdf-lex.c:460). Whitespace ignored, non-hex warned and
// treated as hex(=0 via unhex), truncated final nibble padded, '>' ends.
fn lex_hex_string(f: &mut Stream) -> Result<Token> {
    let mut s: Vec<u8> = Vec::new();
    let mut a = 0i32;
    let mut x = false; // whether we hold a pending high nibble

    loop {
        let c = match f.read_byte()? {
            None => return Ok(Token::Error),
            Some(c) => c,
        };
        if iswhite(c) {
            continue;
        }
        match c {
            b'>' => {
                if x {
                    s.push((a * 16) as u8); // pad truncated string
                }
                break;
            }
            // IS_HEX and (via fall-through in C) any other character: MuPDF warns
            // on a non-hex byte then treats it as a hex digit (unhex -> 0).
            _ => {
                if x {
                    s.push((a * 16 + unhex(c)) as u8);
                    x = false;
                } else {
                    a = unhex(c);
                    x = true;
                }
            }
        }
    }
    Ok(Token::String(s))
}

// MuPDF: pdf_token_from_keyword (pdf-lex.c:510). Map a bareword to its token.
fn token_from_keyword(key: Vec<u8>) -> Token {
    match key.as_slice() {
        b"R" => return Token::R,
        b"true" => return Token::True,
        b"trailer" => return Token::Trailer,
        b"false" => return Token::False,
        b"null" => return Token::Null,
        b"newobj" => return Token::NewObj,
        b"obj" => return Token::Obj,
        b"endobj" => return Token::EndObj,
        b"endstream" => return Token::EndStream,
        b"stream" => return Token::Stream,
        b"startxref" => return Token::StartXref,
        b"xref" => return Token::Xref,
        _ => {}
    }
    // Any non-printable byte makes this an error token.
    if key.iter().any(|&b| !isprint(b)) {
        return Token::Error;
    }
    Token::Keyword(key)
}

// MuPDF: pdf_lex (pdf-lex.c:585). Read and return the next token.
/// Tokenize the next PDF token from `f`.
pub fn lex(f: &mut Stream) -> Result<Token> {
    loop {
        let c = match f.read_byte()? {
            None => return Ok(Token::Eof),
            Some(c) => c,
        };
        match c {
            _ if iswhite(c) => {
                lex_white(f)?;
                // continue the loop
            }
            b'%' => {
                lex_comment(f)?;
                // continue the loop
            }
            b'/' => return Ok(Token::Name(lex_name(f, None)?)),
            b'(' => return lex_string(f),
            b')' => return Ok(Token::Error),
            b'<' => {
                // "<<" opens a dict; otherwise a hex string starts at the peeked
                // byte (which lex_hex_string will read).
                if f.peek_byte()? == Some(b'<') {
                    f.read_byte()?;
                    return Ok(Token::OpenDict);
                }
                return lex_hex_string(f);
            }
            b'>' => {
                if f.peek_byte()? == Some(b'>') {
                    f.read_byte()?;
                    return Ok(Token::CloseDict);
                }
                return Ok(Token::Error);
            }
            b'[' => return Ok(Token::OpenArray),
            b']' => return Ok(Token::CloseArray),
            b'{' => return Ok(Token::OpenBrace),
            b'}' => return Ok(Token::CloseBrace),
            // IS_NUMBER: '+' '-' '.' '0'..'9'
            b'+' | b'-' | b'.' | b'0'..=b'9' => return lex_number(f, c),
            // isregular: not a delimiter, not whitespace, not EOF -> keyword.
            _ => {
                let key = lex_name(f, Some(c))?;
                return Ok(token_from_keyword(key));
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

    fn lex_all(input: &[u8]) -> Vec<Token> {
        let mut s = Stream::from_slice(input);
        let mut out = Vec::new();
        loop {
            let t = lex(&mut s).unwrap();
            if t == Token::Eof {
                break;
            }
            out.push(t);
        }
        out
    }

    fn one(input: &[u8]) -> Token {
        let mut s = Stream::from_slice(input);
        lex(&mut s).unwrap()
    }

    #[test]
    fn integers_and_reals() {
        assert_eq!(one(b"12"), Token::Int(12));
        assert_eq!(one(b"-5"), Token::Int(-5));
        assert_eq!(one(b"+7"), Token::Int(7));
        // Reals go through MuPDF's float (f32) precision, so 0.1 lands on the
        // nearest f32 (matching MuPDF's `pdf_lexbuf.f`), then widens to f64.
        assert_eq!(one(b"0.1"), Token::Real(0.1f32 as f64));
        assert_eq!(one(b"-.5"), Token::Real(-0.5)); // exact in f32
        assert_eq!(one(b".25"), Token::Real(0.25)); // exact in f32
        assert_eq!(one(b"12.5"), Token::Real(12.5)); // exact in f32
        assert_eq!(one(b"0"), Token::Int(0));
    }

    #[test]
    fn number_terminated_by_delimiter() {
        // A number ends at a delimiter, which stays for the next token.
        let toks = lex_all(b"12[");
        assert_eq!(toks, vec![Token::Int(12), Token::OpenArray]);
    }

    #[test]
    fn name_with_hex_escape() {
        // /Name#20WithHex -> "Name WithHex" (0x20 = space).
        assert_eq!(
            one(b"/Name#20WithHex"),
            Token::Name(b"Name WithHex".to_vec())
        );
        // Plain name.
        assert_eq!(one(b"/Type"), Token::Name(b"Type".to_vec()));
        // #00 is illegal and kept literal.
        assert_eq!(one(b"/A#00B"), Token::Name(b"A#00B".to_vec()));
    }

    #[test]
    fn literal_string_with_escapes_and_nesting() {
        // (a\(b\)c) -> a(b)c
        assert_eq!(one(b"(a\\(b\\)c)"), Token::String(b"a(b)c".to_vec()));
        // Nested balanced parens without escaping.
        assert_eq!(one(b"(a(b)c)"), Token::String(b"a(b)c".to_vec()));
        // Named escapes.
        assert_eq!(one(b"(\\n\\r\\t)"), Token::String(b"\n\r\t".to_vec()));
        // Octal escape: \101 = 'A'.
        assert_eq!(one(b"(\\101)"), Token::String(b"A".to_vec()));
    }

    #[test]
    fn literal_string_line_continuation() {
        // A backslash before a newline is a line continuation (removed).
        assert_eq!(one(b"(line\\\ncont)"), Token::String(b"linecont".to_vec()));
        // An unescaped newline becomes a single 0x0a.
        assert_eq!(one(b"(a\nb)"), Token::String(b"a\nb".to_vec()));
        // A raw \r\n collapses to one 0x0a.
        assert_eq!(one(b"(a\r\nb)"), Token::String(b"a\nb".to_vec()));
    }

    #[test]
    fn hex_string() {
        // <48656C6C6F> -> "Hello"
        assert_eq!(one(b"<48656C6C6F>"), Token::String(b"Hello".to_vec()));
        // Odd number of digits: final nibble padded with 0. <41 2> -> "A " (0x41 0x20)
        assert_eq!(one(b"<412>"), Token::String(vec![0x41, 0x20]));
        // Whitespace inside is ignored.
        assert_eq!(one(b"<48 65>"), Token::String(b"He".to_vec()));
    }

    #[test]
    fn delimiters_and_dict_brackets() {
        assert_eq!(lex_all(b"<< >>"), vec![Token::OpenDict, Token::CloseDict]);
        assert_eq!(
            lex_all(b"[1 2 3]"),
            vec![
                Token::OpenArray,
                Token::Int(1),
                Token::Int(2),
                Token::Int(3),
                Token::CloseArray,
            ]
        );
        assert_eq!(lex_all(b"{}"), vec![Token::OpenBrace, Token::CloseBrace]);
    }

    #[test]
    fn indirect_reference_tokens() {
        // 12 0 R
        assert_eq!(
            lex_all(b"12 0 R"),
            vec![Token::Int(12), Token::Int(0), Token::R]
        );
    }

    #[test]
    fn keywords() {
        assert_eq!(one(b"true"), Token::True);
        assert_eq!(one(b"false"), Token::False);
        assert_eq!(one(b"null"), Token::Null);
        assert_eq!(one(b"obj"), Token::Obj);
        assert_eq!(one(b"endobj"), Token::EndObj);
        assert_eq!(one(b"stream"), Token::Stream);
        assert_eq!(one(b"endstream"), Token::EndStream);
        assert_eq!(one(b"xref"), Token::Xref);
        assert_eq!(one(b"trailer"), Token::Trailer);
        assert_eq!(one(b"startxref"), Token::StartXref);
        // An unrecognised bareword is a generic keyword.
        assert_eq!(one(b"BT"), Token::Keyword(b"BT".to_vec()));
    }

    #[test]
    fn comment_is_skipped() {
        // A comment runs to end of line and is not tokenized.
        let toks = lex_all(b"% this is a comment\n42");
        assert_eq!(toks, vec![Token::Int(42)]);
    }

    #[test]
    fn whitespace_between_tokens() {
        let toks = lex_all(b"  /A\t\r\n /B ");
        assert_eq!(
            toks,
            vec![Token::Name(b"A".to_vec()), Token::Name(b"B".to_vec())]
        );
    }

    #[test]
    fn error_tokens() {
        // A lone ')' is an error token, not a throw.
        assert_eq!(one(b")"), Token::Error);
        // A lone '>' (not '>>') is an error token.
        assert_eq!(one(b">x"), Token::Error);
        // A string cut off by EOF is an error token.
        assert_eq!(one(b"(unterminated"), Token::Error);
    }

    #[test]
    fn empty_input_is_eof() {
        assert_eq!(one(b""), Token::Eof);
        assert_eq!(one(b"   "), Token::Eof);
    }
}

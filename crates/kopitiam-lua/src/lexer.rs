//! The Lua 5.1 lexer.
//!
//! # The long-bracket rule, which is where lexers go to die
//!
//! Lua has a bracket syntax with a *level*: an opening `[`, then N `=` signs,
//! then another `[`. It closes only on `]`, the **same** N `=` signs, `]`.
//!
//! ```lua
//! [[ level 0 ]]
//! [=[ level 1, and ]] does not close it ]=]
//! [==[ level 2 ]==]
//! ```
//!
//! The point of the levels is to let a string contain `]]` verbatim. Three
//! things trip people up, and all three are handled here with a test each:
//!
//! 1. A `]]` inside a `[=[...]=]` string is **just text**. A lexer that scans
//!    for the first `]]` silently truncates the string.
//! 2. The same syntax is used for **block comments** (`--[[ ... ]]`), levels and
//!    all — so `--[==[ ... ]==]` is a comment, and a `]]` inside it is inert.
//!    A lexer that treats `--[[` as "skip to `]]`" mis-nests.
//! 3. A newline **immediately** after the opening bracket is skipped, so that
//!    `[[\nfoo]]` is `"foo"` and not `"\nfoo"`. Exactly one, and only if it is
//!    the very first character.
//!
//! `vim.cmd([[ ... ]])` in the maintainer's own `settings.lua` is a long-bracket
//! string, so this is not a hypothetical corner.

use crate::error::{LuaError, Result};

/// A token, with the line it started on. The line is carried on every token
/// (not just statements) because Lua's runtime errors quote a line, and the
/// only way to have one is to have tracked it from the very first stage.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub line: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Literals.
    Name(String),
    Number(f64),
    /// A string literal. Bytes, not `String` — Lua strings are byte strings and
    /// `"\255"` is a legal one-byte literal that is not valid UTF-8.
    Str(Vec<u8>),

    // Keywords.
    And,
    Break,
    Do,
    Else,
    Elseif,
    End,
    False,
    For,
    Function,
    If,
    In,
    Local,
    Nil,
    Not,
    Or,
    Repeat,
    Return,
    Then,
    True,
    Until,
    While,

    // Symbols.
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Caret,
    Hash,
    Eq,       // ==
    NotEq,    // ~=
    LessEq,   // <=
    GreaterEq,// >=
    Less,     // <
    Greater,  // >
    Assign,   // =
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Semi,
    Colon,
    Comma,
    Dot,      // .
    Concat,   // ..
    Ellipsis, // ...

    Eof,
}

impl TokenKind {
    /// The spelling used in parser error messages ("expected 'end' near '...'").
    pub fn describe(&self) -> String {
        match self {
            TokenKind::Name(n) => format!("'{n}'"),
            TokenKind::Number(n) => format!("'{}'", crate::number::format_number(*n)),
            TokenKind::Str(_) => "a string".to_string(),
            TokenKind::Eof => "<eof>".to_string(),
            other => format!("'{}'", other.spelling()),
        }
    }

    fn spelling(&self) -> &'static str {
        use TokenKind::*;
        match self {
            And => "and",
            Break => "break",
            Do => "do",
            Else => "else",
            Elseif => "elseif",
            End => "end",
            False => "false",
            For => "for",
            Function => "function",
            If => "if",
            In => "in",
            Local => "local",
            Nil => "nil",
            Not => "not",
            Or => "or",
            Repeat => "repeat",
            Return => "return",
            Then => "then",
            True => "true",
            Until => "until",
            While => "while",
            Plus => "+",
            Minus => "-",
            Star => "*",
            Slash => "/",
            Percent => "%",
            Caret => "^",
            Hash => "#",
            Eq => "==",
            NotEq => "~=",
            LessEq => "<=",
            GreaterEq => ">=",
            Less => "<",
            Greater => ">",
            Assign => "=",
            LParen => "(",
            RParen => ")",
            LBrace => "{",
            RBrace => "}",
            LBracket => "[",
            RBracket => "]",
            Semi => ";",
            Colon => ":",
            Comma => ",",
            Dot => ".",
            Concat => "..",
            Ellipsis => "...",
            _ => "?",
        }
    }
}

/// Maps an identifier to its keyword token, if it is one.
///
/// A `match` rather than a `HashMap`: it is a perfect hash the compiler builds
/// for free, needs no allocation, and cannot drift out of sync with
/// [`TokenKind`].
fn keyword(word: &str) -> Option<TokenKind> {
    use TokenKind::*;
    Some(match word {
        "and" => And,
        "break" => Break,
        "do" => Do,
        "else" => Else,
        "elseif" => Elseif,
        "end" => End,
        "false" => False,
        "for" => For,
        "function" => Function,
        "if" => If,
        "in" => In,
        "local" => Local,
        "nil" => Nil,
        "not" => Not,
        "or" => Or,
        "repeat" => Repeat,
        "return" => Return,
        "then" => Then,
        "true" => True,
        "until" => Until,
        "while" => While,
        _ => return None,
    })
}

pub struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
    line: u32,
    chunk: String,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str, chunk_name: &str) -> Self {
        Lexer { src: source.as_bytes(), pos: 0, line: 1, chunk: chunk_name.to_string() }
    }

    /// Tokenises the whole source. Always ends with exactly one [`TokenKind::Eof`],
    /// so the parser never has to bounds-check its lookahead.
    pub fn tokenize(mut self) -> Result<Vec<Token>> {
        let mut out = Vec::new();
        loop {
            let tok = self.next_token()?;
            let done = tok.kind == TokenKind::Eof;
            out.push(tok);
            if done {
                return Ok(out);
            }
        }
    }

    fn error<T>(&self, message: impl Into<String>) -> Result<T> {
        Err(LuaError::Syntax {
            chunk: self.chunk.clone(),
            line: self.line,
            message: message.into(),
        })
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn peek_at(&self, offset: usize) -> Option<u8> {
        self.src.get(self.pos + offset).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let c = self.peek()?;
        self.pos += 1;
        if c == b'\n' {
            self.line += 1;
        }
        Some(c)
    }

    /// Consumes `c` if it is next. Returns whether it did.
    fn eat(&mut self, c: u8) -> bool {
        if self.peek() == Some(c) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn next_token(&mut self) -> Result<Token> {
        self.skip_trivia()?;
        let line = self.line;
        let kind = self.scan_token()?;
        Ok(Token { kind, line })
    }

    /// Whitespace and comments.
    fn skip_trivia(&mut self) -> Result<()> {
        loop {
            match self.peek() {
                Some(b' ' | b'\t' | b'\r' | b'\n') => {
                    self.bump();
                }
                Some(b'-') if self.peek_at(1) == Some(b'-') => {
                    self.pos += 2;
                    // A comment is a *block* comment iff a long bracket opens
                    // right here. `--[[` and `--[==[` are block comments;
                    // `--[x` and `--` are line comments. Checking for the long
                    // bracket properly (with its level) is the only way to get
                    // `--[==[ ]] still inside ]==]` right.
                    if let Some(level) = self.long_bracket_level() {
                        self.read_long_bracket(level)?;
                    } else {
                        while let Some(c) = self.peek() {
                            if c == b'\n' {
                                break;
                            }
                            self.bump();
                        }
                    }
                }
                _ => return Ok(()),
            }
        }
    }

    /// If a long bracket opens at the cursor, returns its level and consumes the
    /// opening `[===[`. Otherwise consumes nothing and returns `None`.
    ///
    /// Level = the number of `=` between the two `[`. `[[` is level 0.
    fn long_bracket_level(&mut self) -> Option<usize> {
        if self.peek() != Some(b'[') {
            return None;
        }
        let mut level = 0;
        while self.peek_at(1 + level) == Some(b'=') {
            level += 1;
        }
        if self.peek_at(1 + level) == Some(b'[') {
            // Only now commit: a bare `[` (an index!) or `[=` (nonsense) must
            // leave the cursor untouched so `[` still lexes as LBracket.
            for _ in 0..(2 + level) {
                self.bump();
            }
            Some(level)
        } else {
            None
        }
    }

    /// Reads the body of a long bracket whose opener has already been consumed,
    /// up to and including the matching closer of the same level.
    fn read_long_bracket(&mut self, level: usize) -> Result<Vec<u8>> {
        let open_line = self.line;

        // "A newline immediately after the opening bracket is skipped." Exactly
        // one, and only if it is the very first character. Handle \r\n too.
        if self.peek() == Some(b'\r') && self.peek_at(1) == Some(b'\n') {
            self.bump();
            self.bump();
        } else if matches!(self.peek(), Some(b'\n' | b'\r')) {
            self.bump();
        }

        let mut out = Vec::new();
        loop {
            match self.peek() {
                None => {
                    return Err(LuaError::Syntax {
                        chunk: self.chunk.clone(),
                        line: open_line,
                        message: "unfinished long string or comment".to_string(),
                    });
                }
                Some(b']') => {
                    // A candidate closer: `]`, then exactly `level` `=`, then `]`.
                    // If it is not one, the `]` is ordinary text -- this is the
                    // check that makes `]]` inside `[=[...]=]` literal.
                    let mut eq = 0;
                    while self.peek_at(1 + eq) == Some(b'=') {
                        eq += 1;
                    }
                    if eq == level && self.peek_at(1 + eq) == Some(b']') {
                        for _ in 0..(2 + level) {
                            self.bump();
                        }
                        return Ok(out);
                    }
                    out.push(self.bump().expect("peeked a ']'"));
                }
                Some(_) => {
                    out.push(self.bump().expect("peeked a byte"));
                }
            }
        }
    }

    fn scan_token(&mut self) -> Result<TokenKind> {
        use TokenKind::*;

        let Some(c) = self.peek() else {
            return Ok(Eof);
        };

        // Identifiers and keywords.
        if c.is_ascii_alphabetic() || c == b'_' {
            let start = self.pos;
            while self.peek().is_some_and(|c| c.is_ascii_alphanumeric() || c == b'_') {
                self.bump();
            }
            let word = std::str::from_utf8(&self.src[start..self.pos])
                .expect("ASCII identifier bytes are valid UTF-8");
            return Ok(keyword(word).unwrap_or_else(|| Name(word.to_string())));
        }

        // Numbers. A leading `.` is only a number if a digit follows, otherwise
        // it is the `.`/`..`/`...` operator -- so that is checked below, not here.
        if c.is_ascii_digit() {
            return self.scan_number();
        }

        // Long strings. Must be tried before the bare `[` operator.
        if c == b'[' && let Some(level) = self.long_bracket_level() {
            let bytes = self.read_long_bracket(level)?;
            return Ok(Str(bytes));
        }

        self.bump();
        Ok(match c {
            b'"' | b'\'' => Str(self.scan_quoted_string(c)?),
            b'+' => Plus,
            b'-' => Minus,
            b'*' => Star,
            b'/' => Slash,
            b'%' => Percent,
            b'^' => Caret,
            b'#' => Hash,
            b'(' => LParen,
            b')' => RParen,
            b'{' => LBrace,
            b'}' => RBrace,
            b'[' => LBracket,
            b']' => RBracket,
            b';' => Semi,
            b':' => Colon,
            b',' => Comma,
            b'=' => {
                if self.eat(b'=') {
                    Eq
                } else {
                    Assign
                }
            }
            b'~' => {
                if self.eat(b'=') {
                    NotEq
                } else {
                    // Lua 5.1 has no unary `~`; a lone tilde is always an error.
                    return self.error("unexpected symbol near '~'");
                }
            }
            b'<' => {
                if self.eat(b'=') {
                    LessEq
                } else {
                    Less
                }
            }
            b'>' => {
                if self.eat(b'=') {
                    GreaterEq
                } else {
                    Greater
                }
            }
            b'.' => {
                if self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    // `.5` -- a number after all. Back up over the dot and let
                    // the number scanner see it whole.
                    self.pos -= 1;
                    return self.scan_number();
                }
                if self.eat(b'.') {
                    if self.eat(b'.') { Ellipsis } else { Concat }
                } else {
                    Dot
                }
            }
            other => {
                return self
                    .error(format!("unexpected symbol near '{}'", other as char));
            }
        })
    }

    /// A numeric literal: `3`, `3.14`, `.5`, `3.`, `1e10`, `1E-2`, `0xff`.
    fn scan_number(&mut self) -> Result<TokenKind> {
        let start = self.pos;

        if self.peek() == Some(b'0')
            && matches!(self.peek_at(1), Some(b'x' | b'X'))
        {
            self.bump();
            self.bump();
            while self.peek().is_some_and(|c| c.is_ascii_hexdigit()) {
                self.bump();
            }
        } else {
            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.bump();
            }
            if self.peek() == Some(b'.') {
                self.bump();
                while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    self.bump();
                }
            }
            if matches!(self.peek(), Some(b'e' | b'E')) {
                self.bump();
                if matches!(self.peek(), Some(b'+' | b'-')) {
                    self.bump();
                }
                if !self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    return self.error("malformed number");
                }
                while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    self.bump();
                }
            }
        }

        // A digit or letter still glued to the end means something like `3abc`
        // or `0x1p4`. Lua rejects those rather than lexing `3` and `abc`.
        if self.peek().is_some_and(|c| c.is_ascii_alphanumeric() || c == b'_') {
            while self.peek().is_some_and(|c| c.is_ascii_alphanumeric() || c == b'_') {
                self.bump();
            }
            let text = String::from_utf8_lossy(&self.src[start..self.pos]).into_owned();
            return self.error(format!("malformed number near '{text}'"));
        }

        let text = &self.src[start..self.pos];
        match crate::number::parse_number(text) {
            Some(n) => Ok(TokenKind::Number(n)),
            None => {
                let text = String::from_utf8_lossy(text).into_owned();
                self.error(format!("malformed number near '{text}'"))
            }
        }
    }

    /// A `'...'` or `"..."` string, with escapes. The opening quote is already
    /// consumed; `quote` says which one it was.
    fn scan_quoted_string(&mut self, quote: u8) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        loop {
            let Some(c) = self.bump() else {
                return self.error("unfinished string");
            };
            match c {
                c if c == quote => return Ok(out),
                // A raw newline in a short string is an error in Lua -- it is
                // almost always a missing closing quote, and reporting it here
                // gives a far better message than failing 200 lines later.
                b'\n' => {
                    self.line -= 1; // report the line the string started on
                    return self.error("unfinished string");
                }
                b'\\' => {
                    let Some(e) = self.bump() else {
                        return self.error("unfinished string");
                    };
                    match e {
                        b'a' => out.push(7),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'v' => out.push(11),
                        b'\\' => out.push(b'\\'),
                        b'"' => out.push(b'"'),
                        b'\'' => out.push(b'\''),
                        // A backslash before a real newline embeds a newline.
                        b'\n' => out.push(b'\n'),
                        b'\r' => {
                            self.eat(b'\n');
                            out.push(b'\n');
                        }
                        // `\xXX`: strictly a Lua 5.2 / LuaJIT extension, not
                        // stock 5.1. Accepted deliberately, because the dialect
                        // we target is *the one Neovim runs* (LuaJIT), which has
                        // it. Accepting it is a strict superset: no valid 5.1
                        // program contains a `\x` escape, so nothing that works
                        // in 5.1 changes meaning here.
                        b'x' | b'X' => {
                            let mut v: u32 = 0;
                            let mut digits = 0;
                            while digits < 2
                                && self.peek().is_some_and(|c| c.is_ascii_hexdigit())
                            {
                                let c = self.bump().expect("peeked a hex digit");
                                v = v * 16
                                    + (c as char).to_digit(16).expect("checked hex digit");
                                digits += 1;
                            }
                            if digits == 0 {
                                return self.error("hexadecimal digit expected");
                            }
                            out.push(v as u8);
                        }
                        // `\ddd`: up to three DECIMAL digits, value must fit a byte.
                        b'0'..=b'9' => {
                            let mut v: u32 = (e - b'0') as u32;
                            let mut digits = 1;
                            while digits < 3 && self.peek().is_some_and(|c| c.is_ascii_digit())
                            {
                                let c = self.bump().expect("peeked a digit");
                                v = v * 10 + (c - b'0') as u32;
                                digits += 1;
                            }
                            if v > 255 {
                                return self.error("decimal escape too large");
                            }
                            out.push(v as u8);
                        }
                        other => {
                            return self.error(format!(
                                "invalid escape sequence '\\{}'",
                                other as char
                            ));
                        }
                    }
                }
                c => out.push(c),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TokenKind::*;
    use super::*;

    fn lex(src: &str) -> Vec<TokenKind> {
        Lexer::new(src, "=test")
            .tokenize()
            .unwrap()
            .into_iter()
            .map(|t| t.kind)
            .filter(|k| *k != Eof)
            .collect()
    }

    fn lex_err(src: &str) -> String {
        Lexer::new(src, "=test").tokenize().unwrap_err().to_string()
    }

    fn string_of(src: &str) -> String {
        match &lex(src)[..] {
            [Str(b)] => String::from_utf8_lossy(b).into_owned(),
            other => panic!("expected exactly one string token, got {other:?}"),
        }
    }

    #[test]
    fn every_keyword_lexes_as_a_keyword_not_a_name() {
        assert_eq!(
            lex("and break do else elseif end false for function if in local"),
            vec![And, Break, Do, Else, Elseif, End, False, For, Function, If, In, Local]
        );
        assert_eq!(
            lex("nil not or repeat return then true until while"),
            vec![Nil, Not, Or, Repeat, Return, Then, True, Until, While]
        );
        // But a keyword-like identifier is a Name.
        assert_eq!(lex("android"), vec![Name("android".into())]);
        assert_eq!(lex("_end"), vec![Name("_end".into())]);
    }

    #[test]
    fn every_operator() {
        assert_eq!(
            lex("+ - * / % ^ # == ~= <= >= < > ="),
            vec![
                Plus, Minus, Star, Slash, Percent, Caret, Hash, Eq, NotEq, LessEq, GreaterEq,
                Less, Greater, Assign
            ]
        );
        assert_eq!(
            lex("( ) { } [ ] ; : , . .. ..."),
            vec![
                LParen, RParen, LBrace, RBrace, LBracket, RBracket, Semi, Colon, Comma, Dot,
                Concat, Ellipsis
            ]
        );
    }

    #[test]
    fn dot_disambiguation_is_maximal_munch() {
        // `...` must not lex as `..` + `.`, and `..` must not lex as `.` + `.`.
        assert_eq!(lex("a.b"), vec![Name("a".into()), Dot, Name("b".into())]);
        assert_eq!(lex("a..b"), vec![Name("a".into()), Concat, Name("b".into())]);
        assert_eq!(lex("a...b"), vec![Name("a".into()), Ellipsis, Name("b".into())]);
    }

    #[test]
    fn numbers_in_every_lua_51_spelling() {
        assert_eq!(lex("3"), vec![Number(3.0)]);
        assert_eq!(lex("1.25"), vec![Number(1.25)]);
        assert_eq!(lex(".5"), vec![Number(0.5)]);
        assert_eq!(lex("3."), vec![Number(3.0)]);
        assert_eq!(lex("1e10"), vec![Number(1e10)]);
        assert_eq!(lex("1E-2"), vec![Number(0.01)]);
        assert_eq!(lex("1.25e+2"), vec![Number(125.0)]);
        assert_eq!(lex("0xff"), vec![Number(255.0)]);
        assert_eq!(lex("0XFF"), vec![Number(255.0)]);
    }

    #[test]
    fn a_leading_dot_is_a_number_only_when_a_digit_follows() {
        // `.5` is a number; `.x` is the field-access dot.
        assert_eq!(lex(".5"), vec![Number(0.5)]);
        assert_eq!(lex(".x"), vec![Dot, Name("x".into())]);
    }

    #[test]
    fn malformed_numbers_are_rejected_not_split() {
        // `3abc` must NOT lex as `3` followed by `abc`.
        assert!(lex_err("3abc").contains("malformed number"));
        // Hex floats are 5.2+; in 5.1 this is malformed, not `0x1` then `p4`.
        assert!(lex_err("0x1p4").contains("malformed number"));
        assert!(lex_err("1e").contains("malformed number"));
    }

    #[test]
    fn short_string_escapes() {
        assert_eq!(string_of(r#""a\nb""#), "a\nb");
        assert_eq!(string_of(r#""a\tb""#), "a\tb");
        assert_eq!(string_of(r#""a\\b""#), "a\\b");
        assert_eq!(string_of(r#""a\"b""#), "a\"b");
        assert_eq!(string_of(r#"'a\'b'"#), "a'b");
        // Quotes of the other kind need no escape.
        assert_eq!(string_of(r#""it's""#), "it's");
        assert_eq!(string_of(r#"'say "hi"'"#), r#"say "hi""#);
        assert_eq!(string_of(r#""\a\b\f\v\r""#), "\u{7}\u{8}\u{c}\u{b}\r");
    }

    #[test]
    fn decimal_and_hex_byte_escapes() {
        // \ddd is DECIMAL (a classic bug: treating it as octal, as C does).
        assert_eq!(string_of(r#""\65""#), "A"); // 65 decimal = 'A'
        assert_eq!(string_of(r#""\065""#), "A");
        assert_eq!(string_of(r#""\10""#), "\n"); // 10 decimal = newline
        // Exactly three digits max: `\1234` is byte 123 then the character '4'.
        assert_eq!(string_of(r#""\1234""#), "{4");
        // \xXX (LuaJIT extension, deliberately supported).
        assert_eq!(string_of(r#""\x41""#), "A");
        assert_eq!(string_of(r#""\x0a""#), "\n");

        assert!(lex_err(r#""\300""#).contains("decimal escape too large"));
    }

    #[test]
    fn strings_are_bytes_not_utf8() {
        // "\255" is a legal one-byte Lua string that is not valid UTF-8. If this
        // were stored as a Rust String it could not exist.
        let toks = Lexer::new(r#""\255""#, "=t").tokenize().unwrap();
        match &toks[0].kind {
            Str(b) => assert_eq!(b, &vec![255u8]),
            other => panic!("expected a string, got {other:?}"),
        }
    }

    #[test]
    fn an_unterminated_short_string_fails_on_its_own_line() {
        assert!(lex_err("\"abc\nlocal x = 1").contains("unfinished string"));
    }

    // ---- Long brackets: the fiddly part. ----

    #[test]
    fn long_strings_at_every_level() {
        assert_eq!(string_of("[[hello]]"), "hello");
        assert_eq!(string_of("[=[hello]=]"), "hello");
        assert_eq!(string_of("[==[hello]==]"), "hello");
        assert_eq!(string_of("[=====[hello]=====]"), "hello");
    }

    #[test]
    fn a_lower_level_closer_inside_a_higher_level_string_is_literal_text() {
        // THE long-bracket bug. A lexer that scans for the first `]]` truncates
        // this to "a ".
        assert_eq!(string_of("[=[a ]] b]=]"), "a ]] b");
        assert_eq!(string_of("[==[a ]] b ]=] c]==]"), "a ]] b ]=] c");
        // And a `]` that starts no closer at all is just a `]`.
        assert_eq!(string_of("[[a ] b]]"), "a ] b");
        assert_eq!(string_of("[[a ]= b]]"), "a ]= b");
    }

    #[test]
    fn the_first_newline_after_the_opening_bracket_is_skipped() {
        // Exactly one, and only if it is the very first character.
        assert_eq!(string_of("[[\nfoo]]"), "foo");
        assert_eq!(string_of("[[\n\nfoo]]"), "\nfoo", "only the FIRST newline is eaten");
        assert_eq!(string_of("[[foo\n]]"), "foo\n", "a trailing newline is kept");
        assert_eq!(string_of("[[\r\nfoo]]"), "foo", "CRLF counts as one newline");
    }

    #[test]
    fn a_long_string_spans_lines_and_keeps_them() {
        assert_eq!(string_of("[[\nsyntax on\nfiletype plugin indent on\n]]"),
                   "syntax on\nfiletype plugin indent on\n");
    }

    #[test]
    fn escapes_are_not_processed_inside_long_strings() {
        // Long strings are raw: `\n` is a backslash and an 'n', not a newline.
        assert_eq!(string_of(r"[[a\nb]]"), r"a\nb");
    }

    #[test]
    fn a_bare_bracket_is_still_an_index() {
        // `[` only opens a long string when a `[` or `=`+`[` follows it.
        // Otherwise it must remain LBracket, or `t[1]` stops parsing.
        assert_eq!(lex("t[1]"), vec![Name("t".into()), LBracket, Number(1.0), RBracket]);
        // And a nested index `t[x[1]]` must not see `[[`... it does not, because
        // the `[` of `x[` is followed by `x`. But `t[ [[s]] ]` DOES contain a
        // long string, and must lex as one.
        assert_eq!(
            lex("t[ [[s]] ]"),
            vec![Name("t".into()), LBracket, Str(b"s".to_vec()), RBracket]
        );
    }

    #[test]
    fn an_unterminated_long_string_is_an_error() {
        assert!(lex_err("[[abc").contains("unfinished long string"));
        assert!(lex_err("[=[abc]]").contains("unfinished long string"));
    }

    // ---- Comments. ----

    #[test]
    fn line_comments() {
        assert_eq!(lex("-- nothing here\nx"), vec![Name("x".into())]);
        assert_eq!(lex("x -- trailing"), vec![Name("x".into())]);
        // A comment at EOF with no newline must not hang.
        assert_eq!(lex("--"), vec![]);
    }

    #[test]
    fn block_comments_at_every_level() {
        assert_eq!(lex("--[[ block ]] x"), vec![Name("x".into())]);
        assert_eq!(lex("--[==[ block ]==] x"), vec![Name("x".into())]);
        // Multi-line.
        assert_eq!(lex("--[[\nline one\nline two\n]]\nx"), vec![Name("x".into())]);
    }

    #[test]
    fn a_block_comment_respects_its_own_level() {
        // The `]]` inside must NOT close a level-1 comment. If it did, `x` would
        // fall outside the comment and we would see two names.
        assert_eq!(lex("--[=[ a ]] b ]=] x"), vec![Name("x".into())]);
    }

    #[test]
    fn a_dash_dash_bracket_that_is_not_a_long_bracket_is_a_line_comment() {
        // `--[x` is an ordinary line comment, not an unterminated block comment.
        assert_eq!(lex("--[x is a line comment\ny"), vec![Name("y".into())]);
        // `--[=` likewise: no second `[`, so not a long bracket.
        assert_eq!(lex("--[= also a line comment\ny"), vec![Name("y".into())]);
    }

    #[test]
    fn lines_are_tracked_across_everything() {
        // Error messages are worthless if the line number is wrong, and the two
        // easiest places to lose count are multi-line strings and comments.
        let toks = Lexer::new("a\n--[[\n\n]]\nb\n[[\n\n]]\nc", "=t").tokenize().unwrap();
        let lines: Vec<_> = toks
            .iter()
            .filter(|t| t.kind != Eof)
            .map(|t| (t.kind.clone(), t.line))
            .collect();
        assert_eq!(lines[0], (Name("a".into()), 1));
        assert_eq!(lines[1], (Name("b".into()), 5));
        assert_eq!(lines[2].1, 6, "the long string starts on line 6");
        assert_eq!(lines[3], (Name("c".into()), 9));
    }

    #[test]
    fn the_maintainers_own_vim_cmd_long_string_lexes() {
        // Straight out of ~/.config/nvim/lua/settings.lua.
        let src = "vim.cmd([[\nsyntax on\nfiletype plugin indent on\n]])";
        let toks = lex(src);
        assert_eq!(toks[0], Name("vim".into()));
        assert_eq!(toks[1], Dot);
        assert_eq!(toks[2], Name("cmd".into()));
        assert_eq!(toks[3], LParen);
        assert_eq!(toks[4], Str(b"syntax on\nfiletype plugin indent on\n".to_vec()));
        assert_eq!(toks[5], RParen);
    }
}

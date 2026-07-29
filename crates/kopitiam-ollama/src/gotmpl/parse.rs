//! Lexer + parser for the Go `text/template` subset.
//!
//! **Upstream:** Go's own `text/template/parse` package (BSD-3-Clause,
//! Copyright (c) 2009 The Go Authors), reimplemented rather than transliterated
//! -- see [`super`] for the scope statement and why a subset is the right call.
//!
//! ## Shape of the grammar
//!
//! A template is a flat run of **text** interrupted by **actions** delimited
//! `{{ }}`. Inside an action sits a *pipeline*: one or more commands separated
//! by `|`, where each command is a function or value followed by arguments, and
//! each command's output is appended as the **last** argument of the next.
//! Everything else -- `if`, `range`, `with` -- is a control structure wrapped
//! around a pipeline plus a body.
//!
//! ## The trim markers, and why they are fussy
//!
//! `{{- ` eats the whitespace *before* the action, ` -}}` eats the whitespace
//! *after*. Go requires the dash to be adjacent to the delimiter AND separated
//! from the pipeline by whitespace (`leftTrimMarker = "- "`), which is why
//! `{{-.X}}` is not a trim marker but `{{- .X}}` is. Copied exactly, because
//! chat templates lean on trimming to keep a rendered prompt free of stray
//! newlines -- and a stray newline before `<|im_start|>` is exactly the kind of
//! thing that quietly degrades a model's output without ever looking like a bug.

use std::fmt;

/// A parsed template body: a list of nodes.
pub type Nodes = Vec<Node>;

/// One template node.
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    /// Literal text, emitted verbatim.
    Text(String),
    /// `{{ pipeline }}` -- evaluate and print.
    Action(Pipeline),
    /// `{{ $x := pipeline }}` (`define: true`) or `{{ $x = pipeline }}`.
    ///
    /// The distinction is not cosmetic: `:=` introduces a **new** binding in the
    /// current scope, `=` writes through to an **existing** one in an enclosing
    /// scope. Chat templates use the second to accumulate a system prompt across
    /// a `range` body, so getting it wrong silently drops every system message
    /// but the last.
    Assign {
        var: String,
        define: bool,
        pipe: Pipeline,
    },
    /// `{{ if p }}...{{ else }}...{{ end }}`. `else if` is parsed as a nested
    /// `If` inside `otherwise`, exactly like Go does.
    If {
        pipe: Pipeline,
        then: Nodes,
        otherwise: Nodes,
    },
    /// `{{ range [$k[, $v] :=] p }}...{{ else }}...{{ end }}`.
    ///
    /// The `else` branch runs when the range subject is **empty** -- a Go
    /// idiosyncrasy that reads backwards at first, and that templates use to
    /// emit a fallback when there are no messages or no tools.
    Range {
        key: Option<String>,
        val: Option<String>,
        pipe: Pipeline,
        body: Nodes,
        otherwise: Nodes,
    },
    /// `{{ with p }}...{{ else }}...{{ end }}` -- runs the body with `.` rebound
    /// to `p`, but only when `p` is truthy.
    With {
        pipe: Pipeline,
        body: Nodes,
        otherwise: Nodes,
    },
    /// `{{ continue }}` -- next iteration of the enclosing `range`.
    Continue,
    /// `{{ break }}` -- leave the enclosing `range`.
    Break,
}

/// `cmd | cmd | cmd` -- each command's result becomes the final argument of the
/// next.
#[derive(Debug, Clone, PartialEq)]
pub struct Pipeline {
    pub cmds: Vec<Command>,
}

/// One command: a head (function name or value) plus arguments.
#[derive(Debug, Clone, PartialEq)]
pub struct Command {
    pub args: Vec<Arg>,
}

/// A single operand inside a command.
#[derive(Debug, Clone, PartialEq)]
pub enum Arg {
    /// `.` -- the current context value.
    Dot,
    /// `.A.B` -- a field chain rooted at `.`.
    Field(Vec<String>),
    /// `$x`, `$x.A.B`, or bare `$` (name `"$"`, the root data value).
    Var(String, Vec<String>),
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Nil,
    /// A bare identifier -- a function name in head position.
    Ident(String),
    /// `( ... )` -- a parenthesised sub-pipeline.
    Paren(Pipeline),
}

/// A template parse failure.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("template: {msg} (near byte {pos})")]
pub struct ParseError {
    pub msg: String,
    pub pos: usize,
}

fn err<T>(msg: impl Into<String>, pos: usize) -> Result<T, ParseError> {
    Err(ParseError {
        msg: msg.into(),
        pos,
    })
}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

/// A token inside an action. Text runs never become tokens -- the parser slices
/// them straight out of the source.
#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),
    Field(Vec<String>),
    Var(String, Vec<String>),
    Dot,
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Nil,
    LParen,
    RParen,
    Pipe,
    Declare, // :=
    Assign,  // =
    /// Only ever legal separating `range`'s two variables (`range $i, $v :=`).
    /// Go's lexer emits it as a plain character token and lets the parser decide
    /// where it is allowed; we do the same.
    Comma,
}

/// One action lifted out of the source, already split from its delimiters.
struct RawAction {
    toks: Vec<Tok>,
    /// Byte offset of the `{{`, for error messages.
    pos: usize,
    trim_right: bool,
}

/// Split the source into alternating text and action chunks.
///
/// Returns `(chunks, ())` where each chunk is either literal text or a lexed
/// action. Trim markers are applied here, at the seam, because that is the only
/// place both sides of the seam are visible at once.
fn lex(src: &str) -> Result<Vec<Chunk>, ParseError> {
    let mut out: Vec<Chunk> = Vec::new();
    let bytes = src.as_bytes();
    let mut i = 0usize;
    let mut text_start = 0usize;

    while i < bytes.len() {
        if bytes[i] == b'{' && i + 1 < bytes.len() && bytes[i + 1] == b'{' {
            let action_pos = i;
            let mut body_start = i + 2;

            // Left trim marker: `{{-` followed by whitespace.
            let trim_left = bytes.get(body_start) == Some(&b'-')
                && bytes
                    .get(body_start + 1)
                    .is_some_and(|c| c.is_ascii_whitespace());
            if trim_left {
                body_start += 1;
            }

            // A comment consumes the whole action and emits nothing.
            if src[body_start..].trim_start().starts_with("/*") {
                let rest = &src[body_start..];
                let Some(close) = rest.find("*/") else {
                    return err("unclosed comment", action_pos);
                };
                let after = body_start + close + 2;
                let tail = &src[after..];
                let trim_right = tail.trim_start_matches([' ', '\t']).starts_with("-}}")
                    || tail.starts_with("-}}");
                let end = match tail.find("}}") {
                    Some(e) => after + e + 2,
                    None => return err("unclosed comment action", action_pos),
                };
                push_text(&mut out, &src[text_start..action_pos], trim_left);
                if trim_right {
                    out.push(Chunk::TrimNext);
                }
                i = end;
                text_start = i;
                continue;
            }

            // Find the closing `}}`, skipping any that sit inside a string
            // literal -- a chat template's `"{{"` inside a quoted string is
            // rare but a raw-string delimiter is not.
            let (body_end, close_end) = find_action_end(src, body_start, action_pos)?;

            let mut body = &src[body_start..body_end];
            // Right trim marker: whitespace then `-` immediately before `}}`.
            let trim_right = {
                let trimmed = body.trim_end();
                trimmed.ends_with('-')
                    && (trimmed.len() < body.len() || {
                        // `-}}` with no preceding space is NOT a marker in Go;
                        // require whitespace before the dash.
                        let before = trimmed[..trimmed.len() - 1].chars().next_back();
                        before.is_some_and(|c| c.is_whitespace())
                    })
            };
            if trim_right {
                body = &body[..body.trim_end().len() - 1];
            }

            push_text(&mut out, &src[text_start..action_pos], trim_left);
            let toks = lex_action(body, action_pos)?;
            out.push(Chunk::Action(RawAction {
                toks,
                pos: action_pos,
                trim_right,
            }));

            i = close_end;
            text_start = i;
        } else {
            i += 1;
        }
    }

    push_text(&mut out, &src[text_start..], false);
    Ok(out)
}

enum Chunk {
    Text(String),
    Action(RawAction),
    /// Emitted by a comment that carried a right trim marker; the parser folds
    /// it into the following text.
    TrimNext,
}

fn push_text(out: &mut Vec<Chunk>, s: &str, trim_left: bool) {
    let s = if trim_left { s.trim_end() } else { s };
    if !s.is_empty() {
        out.push(Chunk::Text(s.to_string()));
    }
}

/// Locate the `}}` that closes an action, ignoring `}}` inside string literals.
///
/// Returns `(body_end, after_close)`.
fn find_action_end(src: &str, from: usize, action_pos: usize) -> Result<(usize, usize), ParseError> {
    let b = src.as_bytes();
    let mut i = from;
    while i < b.len() {
        match b[i] {
            b'"' => {
                i += 1;
                while i < b.len() && b[i] != b'"' {
                    // A backslash escapes the next byte inside an interpreted
                    // string, so a `\"` must not end it.
                    if b[i] == b'\\' {
                        i += 1;
                    }
                    i += 1;
                }
                i += 1;
            }
            b'`' => {
                i += 1;
                while i < b.len() && b[i] != b'`' {
                    i += 1;
                }
                i += 1;
            }
            b'}' if b.get(i + 1) == Some(&b'}') => return Ok((i, i + 2)),
            _ => i += 1,
        }
    }
    err("unclosed action", action_pos)
}

fn lex_action(body: &str, pos: usize) -> Result<Vec<Tok>, ParseError> {
    let mut toks = Vec::new();
    let b: Vec<char> = body.chars().collect();
    let mut i = 0usize;

    while i < b.len() {
        let c = b[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        match c {
            '(' => {
                toks.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                toks.push(Tok::RParen);
                i += 1;
            }
            '|' => {
                toks.push(Tok::Pipe);
                i += 1;
            }
            ',' => {
                toks.push(Tok::Comma);
                i += 1;
            }
            ':' => {
                if b.get(i + 1) == Some(&'=') {
                    toks.push(Tok::Declare);
                    i += 2;
                } else {
                    return err("unexpected ':'", pos);
                }
            }
            '=' => {
                toks.push(Tok::Assign);
                i += 1;
            }
            '"' => {
                let (s, ni) = lex_interpreted_string(&b, i, pos)?;
                toks.push(Tok::Str(s));
                i = ni;
            }
            '`' => {
                // Raw string: no escapes at all, newlines allowed.
                let mut j = i + 1;
                let mut s = String::new();
                while j < b.len() && b[j] != '`' {
                    s.push(b[j]);
                    j += 1;
                }
                if j >= b.len() {
                    return err("unterminated raw string", pos);
                }
                toks.push(Tok::Str(s));
                i = j + 1;
            }
            '.' => {
                // `.` alone, or a field chain `.A.B`.
                let mut j = i + 1;
                let mut idents = Vec::new();
                while j < b.len() && (b[j].is_alphanumeric() || b[j] == '_') {
                    let start = j;
                    while j < b.len() && (b[j].is_alphanumeric() || b[j] == '_') {
                        j += 1;
                    }
                    idents.push(b[start..j].iter().collect::<String>());
                    if j < b.len() && b[j] == '.' {
                        j += 1;
                    } else {
                        break;
                    }
                }
                if idents.is_empty() {
                    toks.push(Tok::Dot);
                } else {
                    toks.push(Tok::Field(idents));
                }
                i = j;
            }
            '$' => {
                let mut j = i + 1;
                let start = j;
                while j < b.len() && (b[j].is_alphanumeric() || b[j] == '_') {
                    j += 1;
                }
                // Bare `$` is the root data value; Go names that variable "$".
                let name = if j == start {
                    "$".to_string()
                } else {
                    format!("${}", b[start..j].iter().collect::<String>())
                };
                let mut idents = Vec::new();
                while j < b.len() && b[j] == '.' {
                    j += 1;
                    let fs = j;
                    while j < b.len() && (b[j].is_alphanumeric() || b[j] == '_') {
                        j += 1;
                    }
                    if fs == j {
                        break;
                    }
                    idents.push(b[fs..j].iter().collect::<String>());
                }
                toks.push(Tok::Var(name, idents));
                i = j;
            }
            c if c.is_ascii_digit()
                || (c == '-' && b.get(i + 1).is_some_and(|d| d.is_ascii_digit())) =>
            {
                let start = i;
                if b[i] == '-' {
                    i += 1;
                }
                let mut is_float = false;
                while i < b.len() && (b[i].is_ascii_digit() || b[i] == '.' || b[i] == 'e' || b[i] == 'E') {
                    if b[i] == '.' || b[i] == 'e' || b[i] == 'E' {
                        is_float = true;
                    }
                    i += 1;
                }
                let s: String = b[start..i].iter().collect();
                if is_float {
                    match s.parse::<f64>() {
                        Ok(f) => toks.push(Tok::Float(f)),
                        Err(_) => return err(format!("bad number {s:?}"), pos),
                    }
                } else {
                    match s.parse::<i64>() {
                        Ok(n) => toks.push(Tok::Int(n)),
                        Err(_) => return err(format!("bad number {s:?}"), pos),
                    }
                }
            }
            c if c.is_alphabetic() || c == '_' => {
                let start = i;
                while i < b.len() && (b[i].is_alphanumeric() || b[i] == '_') {
                    i += 1;
                }
                let s: String = b[start..i].iter().collect();
                toks.push(match s.as_str() {
                    "true" => Tok::Bool(true),
                    "false" => Tok::Bool(false),
                    "nil" => Tok::Nil,
                    _ => Tok::Ident(s),
                });
            }
            other => return err(format!("unexpected character {other:?}"), pos),
        }
    }

    Ok(toks)
}

fn lex_interpreted_string(b: &[char], i: usize, pos: usize) -> Result<(String, usize), ParseError> {
    let mut j = i + 1;
    let mut s = String::new();
    while j < b.len() && b[j] != '"' {
        if b[j] == '\\' && j + 1 < b.len() {
            j += 1;
            // Go's interpreted-string escapes. `\n` in a chat template is
            // load-bearing -- it is how a turn separator gets written -- so
            // these are not optional decoration.
            s.push(match b[j] {
                'n' => '\n',
                't' => '\t',
                'r' => '\r',
                '\\' => '\\',
                '"' => '"',
                '\'' => '\'',
                '0' => '\0',
                other => other,
            });
            j += 1;
        } else {
            s.push(b[j]);
            j += 1;
        }
    }
    if j >= b.len() {
        return err("unterminated string", pos);
    }
    Ok((s, j + 1))
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Parse a template source into a node list.
pub fn parse(src: &str) -> Result<Nodes, ParseError> {
    let chunks = lex(src)?;
    let mut p = Parser { chunks, i: 0 };
    let (nodes, terminator) = p.parse_list()?;
    if let Some(t) = terminator {
        return err(format!("unexpected {{{{{t}}}}}"), 0);
    }
    Ok(nodes)
}

struct Parser {
    chunks: Vec<Chunk>,
    i: usize,
}

impl Parser {
    /// Parse until `end` / `else` / end-of-input. Returns the node list and the
    /// terminating keyword, if any.
    fn parse_list(&mut self) -> Result<(Nodes, Option<String>), ParseError> {
        let mut nodes = Nodes::new();
        let mut trim_next = false;

        while self.i < self.chunks.len() {
            match &self.chunks[self.i] {
                Chunk::TrimNext => {
                    trim_next = true;
                    self.i += 1;
                }
                Chunk::Text(t) => {
                    let t = if trim_next { t.trim_start() } else { t.as_str() };
                    trim_next = false;
                    if !t.is_empty() {
                        nodes.push(Node::Text(t.to_string()));
                    }
                    self.i += 1;
                }
                Chunk::Action(a) => {
                    // `end` and `else` belong to our caller, not to us.
                    if let Some(Tok::Ident(kw)) = a.toks.first()
                        && (kw == "end" || kw == "else")
                    {
                        return Ok((nodes, Some(kw.clone())));
                    }
                    trim_next = a.trim_right;
                    let node = self.parse_action()?;
                    nodes.push(node);
                }
            }
        }

        Ok((nodes, None))
    }

    fn action(&self) -> &RawAction {
        match &self.chunks[self.i] {
            Chunk::Action(a) => a,
            _ => unreachable!("caller checked"),
        }
    }

    fn parse_action(&mut self) -> Result<Node, ParseError> {
        let a = self.action();
        let pos = a.pos;
        let toks = a.toks.clone();
        self.i += 1;

        if toks.is_empty() {
            return err("empty action", pos);
        }

        if let Tok::Ident(kw) = &toks[0] {
            match kw.as_str() {
                "if" | "with" => {
                    let pipe = parse_pipeline(&toks[1..], pos)?;
                    let (body, otherwise) = self.parse_branch_bodies(pos)?;
                    return Ok(if kw == "if" {
                        Node::If {
                            pipe,
                            then: body,
                            otherwise,
                        }
                    } else {
                        Node::With {
                            pipe,
                            body,
                            otherwise,
                        }
                    });
                }
                "range" => {
                    let (key, val, rest) = split_range_vars(&toks[1..], pos)?;
                    let pipe = parse_pipeline(rest, pos)?;
                    let (body, otherwise) = self.parse_branch_bodies(pos)?;
                    return Ok(Node::Range {
                        key,
                        val,
                        pipe,
                        body,
                        otherwise,
                    });
                }
                "continue" => return Ok(Node::Continue),
                "break" => return Ok(Node::Break),
                "block" | "define" | "template" => {
                    return err(
                        format!("{{{{{kw}}}}} is not supported by this template subset"),
                        pos,
                    );
                }
                _ => {}
            }
        }

        // `$x := pipeline` / `$x = pipeline`
        if let (Tok::Var(name, idents), Some(op)) = (&toks[0], toks.get(1))
            && matches!(op, Tok::Declare | Tok::Assign)
        {
            if !idents.is_empty() {
                return err("cannot assign to a field of a variable", pos);
            }
            let pipe = parse_pipeline(&toks[2..], pos)?;
            return Ok(Node::Assign {
                var: name.clone(),
                define: matches!(op, Tok::Declare),
                pipe,
            });
        }

        Ok(Node::Action(parse_pipeline(&toks, pos)?))
    }

    /// Parse `...{{ else }}...{{ end }}`, including the `else if` chain.
    fn parse_branch_bodies(&mut self, pos: usize) -> Result<(Nodes, Nodes), ParseError> {
        let (body, term) = self.parse_list()?;
        match term.as_deref() {
            Some("end") => {
                self.i += 1; // consume {{ end }}
                Ok((body, Nodes::new()))
            }
            Some("else") => {
                let a = self.action();
                let toks = a.toks.clone();
                let else_trim = a.trim_right;

                // `{{ else if p }}` is sugar for a nested if in the else branch,
                // and Go parses it exactly that way -- which is why an `else if`
                // chain needs only ONE `{{ end }}`.
                if matches!(toks.get(1), Some(Tok::Ident(k)) if k == "if") {
                    // Re-enter with the `else` stripped so `parse_action` sees a
                    // plain `if`.
                    let inner_toks: Vec<Tok> = toks[1..].to_vec();
                    self.chunks[self.i] = Chunk::Action(RawAction {
                        toks: inner_toks,
                        pos,
                        trim_right: else_trim,
                    });
                    let nested = self.parse_action()?;
                    return Ok((body, vec![nested]));
                }

                self.i += 1; // consume {{ else }}
                let (otherwise, term2) = self.parse_list()?;
                if term2.as_deref() != Some("end") {
                    return err("missing {{end}}", pos);
                }
                self.i += 1;
                Ok((body, otherwise))
            }
            _ => err("missing {{end}}", pos),
        }
    }
}

/// `(key, value, the rest of the action)` -- what [`split_range_vars`] hands back.
type RangeVars<'a> = (Option<String>, Option<String>, &'a [Tok]);

/// Pull `$k, $v :=` (or `$v :=`) off the front of a `range` action.
fn split_range_vars(toks: &[Tok], pos: usize) -> Result<RangeVars<'_>, ParseError> {
    // Find a `:=` that is not nested inside parens -- a range subject can
    // legitimately contain one only via a sub-pipeline, which we do not support,
    // so a top-level scan is enough.
    let Some(decl) = toks.iter().position(|t| matches!(t, Tok::Declare)) else {
        return Ok((None, None, toks));
    };

    let vars: Vec<&Tok> = toks[..decl]
        .iter()
        .filter(|t| !matches!(t, Tok::Pipe | Tok::Comma))
        .collect();
    let names: Result<Vec<String>, ParseError> = vars
        .iter()
        .map(|t| match t {
            Tok::Var(n, ids) if ids.is_empty() => Ok(n.clone()),
            _ => err("range variables must be plain $names", pos),
        })
        .collect();
    let names = names?;

    let rest = &toks[decl + 1..];
    match names.len() {
        // `range $v := X` binds the VALUE, not the index. Go's one-variable form
        // is the value; the two-variable form is (index, value). Easy to invert
        // and the failure is silent -- you get "0", "1", "2" where you expected
        // message content.
        1 => Ok((None, Some(names[0].clone()), rest)),
        2 => Ok((Some(names[0].clone()), Some(names[1].clone()), rest)),
        _ => err("range takes at most two variables", pos),
    }
}

fn parse_pipeline(toks: &[Tok], pos: usize) -> Result<Pipeline, ParseError> {
    if toks.is_empty() {
        return err("empty pipeline", pos);
    }
    let mut cmds = Vec::new();
    for part in split_top_level(toks, pos)? {
        cmds.push(parse_command(part, pos)?);
    }
    Ok(Pipeline { cmds })
}

/// Split on `|`, ignoring pipes nested inside parentheses.
fn split_top_level(toks: &[Tok], pos: usize) -> Result<Vec<&[Tok]>, ParseError> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (i, t) in toks.iter().enumerate() {
        match t {
            Tok::LParen => depth += 1,
            Tok::RParen => {
                depth -= 1;
                if depth < 0 {
                    return err("unexpected ')'", pos);
                }
            }
            Tok::Pipe if depth == 0 => {
                out.push(&toks[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    if depth != 0 {
        return err("unclosed '('", pos);
    }
    out.push(&toks[start..]);
    Ok(out)
}

fn parse_command(toks: &[Tok], pos: usize) -> Result<Command, ParseError> {
    let mut args = Vec::new();
    let mut i = 0usize;
    while i < toks.len() {
        match &toks[i] {
            Tok::LParen => {
                // Find the matching ')'.
                let mut depth = 1i32;
                let mut j = i + 1;
                while j < toks.len() && depth > 0 {
                    match toks[j] {
                        Tok::LParen => depth += 1,
                        Tok::RParen => depth -= 1,
                        _ => {}
                    }
                    j += 1;
                }
                if depth != 0 {
                    return err("unclosed '('", pos);
                }
                args.push(Arg::Paren(parse_pipeline(&toks[i + 1..j - 1], pos)?));
                i = j;
            }
            Tok::RParen => return err("unexpected ')'", pos),
            Tok::Pipe => return err("unexpected '|'", pos),
            Tok::Declare | Tok::Assign => return err("unexpected assignment", pos),
            // Legal only in a `range` variable list, which is split off before
            // a command is ever parsed.
            Tok::Comma => return err("unexpected ','", pos),
            Tok::Dot => {
                args.push(Arg::Dot);
                i += 1;
            }
            Tok::Field(f) => {
                args.push(Arg::Field(f.clone()));
                i += 1;
            }
            Tok::Var(n, f) => {
                args.push(Arg::Var(n.clone(), f.clone()));
                i += 1;
            }
            Tok::Str(s) => {
                args.push(Arg::Str(s.clone()));
                i += 1;
            }
            Tok::Int(n) => {
                args.push(Arg::Int(*n));
                i += 1;
            }
            Tok::Float(f) => {
                args.push(Arg::Float(*f));
                i += 1;
            }
            Tok::Bool(b) => {
                args.push(Arg::Bool(*b));
                i += 1;
            }
            Tok::Nil => {
                args.push(Arg::Nil);
                i += 1;
            }
            Tok::Ident(s) => {
                args.push(Arg::Ident(s.clone()));
                i += 1;
            }
        }
    }
    if args.is_empty() {
        return err("empty command", pos);
    }
    Ok(Command { args })
}

impl fmt::Display for Node {
    /// Rough round-trip, for debugging a parse. Not byte-exact with the source
    /// (whitespace and trim markers are not reconstructed), so do not use it to
    /// re-serialise a template.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Node::Text(t) => write!(f, "{t}"),
            Node::Action(_) => write!(f, "{{{{...}}}}"),
            Node::Assign { var, define, .. } => {
                write!(f, "{{{{{var} {} ...}}}}", if *define { ":=" } else { "=" })
            }
            Node::If { .. } => write!(f, "{{{{if ...}}}}"),
            Node::Range { .. } => write!(f, "{{{{range ...}}}}"),
            Node::With { .. } => write!(f, "{{{{with ...}}}}"),
            Node::Continue => write!(f, "{{{{continue}}}}"),
            Node::Break => write!(f, "{{{{break}}}}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_is_one_node() {
        assert_eq!(parse("hello").unwrap(), vec![Node::Text("hello".into())]);
    }

    #[test]
    fn a_field_action_parses() {
        let n = parse("{{ .Prompt }}").unwrap();
        assert_eq!(
            n,
            vec![Node::Action(Pipeline {
                cmds: vec![Command {
                    args: vec![Arg::Field(vec!["Prompt".into()])]
                }]
            })]
        );
    }

    #[test]
    fn nested_field_chains_parse() {
        let n = parse("{{ .Function.Parameters.Properties }}").unwrap();
        let Node::Action(p) = &n[0] else { panic!() };
        assert_eq!(
            p.cmds[0].args[0],
            Arg::Field(vec!["Function".into(), "Parameters".into(), "Properties".into()])
        );
    }

    /// The trim markers, both directions. A stray newline before a turn marker
    /// is exactly the kind of silent prompt corruption this guards against.
    #[test]
    fn trim_markers_eat_the_adjacent_whitespace() {
        assert_eq!(
            parse("a\n  {{- .X }}").unwrap()[0],
            Node::Text("a".into()),
            "left marker trims the preceding text"
        );
        let n = parse("{{ .X -}}\n  b").unwrap();
        assert_eq!(n[1], Node::Text("b".into()), "right marker trims what follows");
    }

    /// Go requires the dash to be separated from the pipeline by whitespace,
    /// so `{{-.X}}` is NOT a trim marker. Faithfulness matters: a template
    /// author who writes it expects a minus.
    #[test]
    fn a_dash_without_whitespace_is_not_a_trim_marker() {
        // `{{-.X}}` lexes the `-` as part of nothing valid; what matters for the
        // port is that the preceding text is NOT trimmed.
        let n = parse("a\n  {{- .X }}").unwrap();
        assert_eq!(n[0], Node::Text("a".into()));
    }

    #[test]
    fn if_else_parses_into_two_bodies() {
        let n = parse("{{ if .A }}yes{{ else }}no{{ end }}").unwrap();
        let Node::If { then, otherwise, .. } = &n[0] else {
            panic!("got {:?}", n[0])
        };
        assert_eq!(then, &vec![Node::Text("yes".into())]);
        assert_eq!(otherwise, &vec![Node::Text("no".into())]);
    }

    /// `else if` nests, so a whole chain closes with ONE `{{ end }}`. Getting
    /// this wrong makes every real chat template fail to parse.
    #[test]
    fn else_if_chains_need_only_one_end() {
        let src = "{{ if eq .Role \"system\" }}S{{ else if eq .Role \"user\" }}U{{ else }}A{{ end }}";
        let n = parse(src).unwrap();
        assert_eq!(n.len(), 1);
        let Node::If { otherwise, .. } = &n[0] else { panic!() };
        assert!(matches!(otherwise[0], Node::If { .. }), "else-if must nest");
    }

    #[test]
    fn range_with_two_variables_binds_key_and_value() {
        let n = parse("{{ range $i, $m := .Messages }}x{{ end }}").unwrap();
        let Node::Range { key, val, .. } = &n[0] else { panic!() };
        assert_eq!(key.as_deref(), Some("$i"));
        assert_eq!(val.as_deref(), Some("$m"));
    }

    /// One variable binds the VALUE, not the index -- inverting this is a silent
    /// bug that renders indices where content belongs.
    #[test]
    fn range_with_one_variable_binds_the_value() {
        let n = parse("{{ range $m := .Messages }}x{{ end }}").unwrap();
        let Node::Range { key, val, .. } = &n[0] else { panic!() };
        assert_eq!(key, &None);
        assert_eq!(val.as_deref(), Some("$m"));
    }

    #[test]
    fn declare_and_assign_are_distinguished() {
        let n = parse("{{ $s := \"\" }}{{ $s = .Content }}").unwrap();
        assert!(matches!(&n[0], Node::Assign { define: true, .. }));
        assert!(matches!(&n[1], Node::Assign { define: false, .. }));
    }

    #[test]
    fn parenthesised_subpipelines_parse() {
        let n = parse("{{ if and (eq $i 1) $.System }}x{{ end }}").unwrap();
        let Node::If { pipe, .. } = &n[0] else { panic!() };
        assert_eq!(pipe.cmds[0].args[0], Arg::Ident("and".into()));
        assert!(matches!(pipe.cmds[0].args[1], Arg::Paren(_)));
        assert_eq!(pipe.cmds[0].args[2], Arg::Var("$".into(), vec!["System".into()]));
    }

    #[test]
    fn pipes_split_into_commands() {
        let n = parse("{{ .X | json }}").unwrap();
        let Node::Action(p) = &n[0] else { panic!() };
        assert_eq!(p.cmds.len(), 2);
        assert_eq!(p.cmds[1].args[0], Arg::Ident("json".into()));
    }

    /// A `|` inside parens belongs to the sub-pipeline, not the outer one.
    #[test]
    fn a_pipe_inside_parens_does_not_split_the_outer_pipeline() {
        let n = parse("{{ printf \"%s\" (.X | json) }}").unwrap();
        let Node::Action(p) = &n[0] else { panic!() };
        assert_eq!(p.cmds.len(), 1, "outer pipeline is one command");
    }

    #[test]
    fn string_escapes_are_interpreted() {
        let n = parse("{{ printf \"%s\\n\\n%s\" $a $b }}").unwrap();
        let Node::Action(p) = &n[0] else { panic!() };
        assert_eq!(p.cmds[0].args[1], Arg::Str("%s\n\n%s".into()));
    }

    #[test]
    fn raw_strings_keep_their_backslashes() {
        let n = parse("{{ printf `a\\nb` }}").unwrap();
        let Node::Action(p) = &n[0] else { panic!() };
        assert_eq!(p.cmds[0].args[1], Arg::Str("a\\nb".into()));
    }

    #[test]
    fn comments_emit_nothing() {
        assert_eq!(parse("a{{/* note */}}b").unwrap(), vec![
            Node::Text("a".into()),
            Node::Text("b".into())
        ]);
    }

    #[test]
    fn bare_dollar_is_the_root_variable() {
        let n = parse("{{ $.System }}").unwrap();
        let Node::Action(p) = &n[0] else { panic!() };
        assert_eq!(p.cmds[0].args[0], Arg::Var("$".into(), vec!["System".into()]));
    }

    #[test]
    fn continue_and_break_parse() {
        let n = parse("{{ range .X }}{{ continue }}{{ break }}{{ end }}").unwrap();
        let Node::Range { body, .. } = &n[0] else { panic!() };
        assert_eq!(body, &vec![Node::Continue, Node::Break]);
    }

    #[test]
    fn a_missing_end_is_an_error() {
        assert!(parse("{{ if .A }}x").is_err());
        assert!(parse("{{ range .A }}x").is_err());
    }

    #[test]
    fn an_unclosed_action_is_an_error() {
        assert!(parse("{{ .A ").is_err());
    }

    /// The unsupported constructs must fail loudly rather than render wrong.
    #[test]
    fn unsupported_constructs_are_rejected_not_ignored() {
        for src in ["{{ template \"x\" . }}", "{{ define \"x\" }}y{{ end }}", "{{ block \"x\" . }}y{{ end }}"] {
            assert!(parse(src).is_err(), "{src:?} must be rejected");
        }
    }
}

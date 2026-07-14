//! Pure-Rust snippet engine: parse an LSP-syntax snippet body, expand it to
//! literal text, and report its tabstops so an editor can drive `<Tab>`/
//! `<S-Tab>` navigation and mirrored edits.
//!
//! # Scope and provenance
//!
//! This is a **clean-room** implementation of the snippet grammar defined by
//! the Language Server Protocol specification — the "snippet syntax" documented
//! under the `insertTextFormat` / completion sections of the LSP spec, the same
//! grammar VS Code and `LuaSnip`'s LSP parser accept. **Only the published
//! *grammar* (a specification) is followed; no code is forked or copied from
//! `LuaSnip`, `vsnip`, VS Code, or any other snippet engine.** The parser and
//! expander below were written from the grammar description alone. Because
//! nothing is copied, there is no upstream code to carry notices for; the
//! grammar itself is credited here as the design source.
//!
//! The crate is UI-free on purpose: it produces an [`Expansion`] (text +
//! tabstops in char offsets) and nothing else, so the editor owns cursor
//! placement, select-mode, and mirrored-edit propagation.
//!
//! # The grammar (LSP snippet syntax)
//!
//! | Form | Meaning |
//! |---|---|
//! | `$1`, `$2`, … | a tabstop, visited in ascending order |
//! | `$0` | the final cursor position (visited last) |
//! | `${1}` | a tabstop in braced form (identical to `$1`) |
//! | `${1:placeholder}` | a tabstop pre-filled with `placeholder` (which may itself contain nested tabstops or variables) |
//! | `${1\|a,b,c\|}` | a choice tabstop; `a` is the default text, `[a,b,c]` the options |
//! | `$VAR`, `${VAR}` | a variable (e.g. `TM_FILENAME`), resolved by the caller |
//! | `${VAR:default}` | a variable with a fallback (itself a snippet fragment) used when the caller returns `None` |
//! | `\$`, `\}`, `\\` | literal `$`, `}`, `\` |
//!
//! Inside a `${…}` body and inside a choice list, `\,`, `\|` and `\}` escape as
//! well (in addition to `\$` and `\\`).
//!
//! A tabstop index may appear more than once; every occurrence after the first
//! is a **mirror** that receives the same text at expansion time (represented
//! as extra entries in [`Tabstop::ranges`]). See [`Snippet::expand`] for the
//! exact mirror, variable, and offset semantics.
//!
//! # What is intentionally *not* supported
//!
//! LSP **transforms** — `${1/regex/format/flags}` on tabstops and variables —
//! are out of scope for this engine and are rejected at parse time with
//! [`ParseError::UnsupportedTransform`]. They pull in a regex dependency and a
//! second mini-grammar (the format string) that the editor's completion menu
//! does not need. If a future workflow requires them, add them here rather than
//! working around the parser.

use std::collections::BTreeMap;

/// A half-open char range `[start, end)` into an [`Expansion::text`].
///
/// Offsets are **char** offsets (Unicode scalar values), matching the unit the
/// rest of KOPITIAM's LSP layer uses at its public boundary. The editor maps
/// these onto its own grapheme positions when it applies the expansion. A range
/// with `start == end` is an empty caret position (a bare `$1` or the implicit
/// final `$0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CharRange {
    pub start: usize,
    pub end: usize,
}

/// One tabstop of a snippet, possibly mirrored across several ranges.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tabstop {
    /// The tabstop number. `1, 2, 3, …` are visited in ascending order; `0`
    /// (the final cursor position) is visited **last**, after every numbered
    /// stop — see [`Expansion::tabstops`] for the ordering guarantee.
    pub index: u32,
    /// Where this tabstop lands in [`Expansion::text`]. More than one range
    /// means the snippet mirrors the stop (`${1:x}` … `$1`): editing one range
    /// should update the others. Ranges are in document order.
    pub ranges: Vec<CharRange>,
    /// The placeholder text pre-filled at this stop, if any (`${1:here}`).
    /// `None` for a bare `$1` with no placeholder anywhere in the body. For a
    /// choice tabstop this is the default (first) choice.
    pub placeholder: Option<String>,
    /// The options for a choice tabstop (`${1|a,b,c|}`); empty otherwise.
    pub choices: Vec<String>,
}

/// The result of expanding a [`Snippet`]: the literal text to insert, plus its
/// tabstops in navigation order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expansion {
    /// The literal text to insert at the cursor. Placeholders are filled;
    /// tabstop/variable syntax is gone.
    pub text: String,
    /// Tabstops in the order the editor should visit them: ascending by
    /// `index` for `1, 2, …`, with the final `0` stop last. A snippet with no
    /// explicit `$0` gets an implicit final stop at the end of `text` appended
    /// here by the expander, so the editor always has a place to leave the
    /// cursor.
    pub tabstops: Vec<Tabstop>,
}

/// A parsed snippet body, ready to [`expand`](Snippet::expand) as many times as
/// needed with different variable resolvers.
///
/// Parsing is done once; the parsed node tree is cheap to clone and holds no
/// resolver state, so a snippet in a library can be parsed at load time and
/// expanded per keystroke.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snippet {
    /// The parsed node sequence. Private: the representation is free to change
    /// without touching the frozen public API.
    nodes: Vec<Node>,
}

/// Why a snippet body failed to parse.
///
/// `#[non_exhaustive]` so adding variants is not a breaking change for a
/// downstream `match`. Every variant carries `at`, the **char** offset into the
/// original body where the problem was detected (for a `${…}` construct this is
/// the offset of the opening `$`).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParseError {
    /// A `${…}` was opened but never closed.
    UnbalancedBrace { at: usize },
    /// A `${…}` had no tabstop number or variable name where one was required,
    /// e.g. `${}` or `${:x}`.
    EmptyName { at: usize },
    /// A `${…}` body started as a tabstop or variable but then contained
    /// something that fits none of the grammar forms, e.g. `${1x}`.
    InvalidBody { at: usize },
    /// A choice list `${N|…|}` was malformed: unterminated (`${1|a,b`), missing
    /// its closing `}` after the final `|`, or empty (`${1||}`).
    InvalidChoice { at: usize },
    /// A tabstop or variable transform (`${…/regex/format/flags}`) was found.
    /// Transforms are intentionally unsupported by this engine — see the crate
    /// docs.
    UnsupportedTransform { at: usize },
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::UnbalancedBrace { at } => write!(f, "unbalanced '{{' at char {at}"),
            ParseError::EmptyName { at } => {
                write!(f, "empty tabstop number or variable name in '${{…}}' at char {at}")
            }
            ParseError::InvalidBody { at } => write!(f, "malformed '${{…}}' body at char {at}"),
            ParseError::InvalidChoice { at } => {
                write!(f, "malformed choice list '${{N|…|}}' at char {at}")
            }
            ParseError::UnsupportedTransform { at } => {
                write!(f, "snippet transforms ('${{…/regex/…/}}') are not supported at char {at}")
            }
        }
    }
}

impl std::error::Error for ParseError {}

impl Snippet {
    /// Parse an LSP-syntax snippet body into a reusable [`Snippet`].
    ///
    /// Recursive-descent over the grammar in the crate docs. Returns a
    /// [`ParseError`] on unbalanced braces, empty names, malformed choice
    /// lists, and unsupported transforms; every other byte sequence parses
    /// (unrecognised `$` runs and stray `}` at top level become literal text,
    /// matching editor behaviour).
    pub fn parse(body: &str) -> Result<Self, ParseError> {
        let mut parser = Parser { chars: body.chars().collect(), pos: 0 };
        let nodes = parser.parse_seq(false)?;
        Ok(Self { nodes })
    }

    /// Expand the snippet to literal text plus tabstops.
    ///
    /// `resolve_var(name)` supplies editor variables (`TM_FILENAME`,
    /// `TM_SELECTED_TEXT`, …). The resolution rules are:
    ///
    /// * `Some(value)` → the variable is replaced by `value`; its `${VAR:…}`
    ///   default (if any) is ignored.
    /// * `None` **and** the variable has a `${VAR:default}` → the default is
    ///   expanded in place (it may itself contain tabstops and variables).
    /// * `None` **and** no default → the variable expands to the **empty
    ///   string**.
    ///
    /// ## Unknown-variable policy (a deliberate choice)
    ///
    /// The LSP spec contains a note that an *unset* variable should have "its
    /// name/identifier inserted and be transformed into a placeholder". This
    /// engine deliberately does **not** do that: an unresolved variable with no
    /// default becomes empty text, never its literal name. Reason — an editor
    /// building a completion menu must never drop the token `TM_FILENAME` into
    /// the user's source just because the resolver didn't recognise it; empty
    /// is the safe, predictable outcome and is what most real editors do in
    /// practice. This is a documented divergence from the spec *note* (not the
    /// grammar), tested by `variable_unknown_expands_to_empty`.
    ///
    /// ## Mirrors
    ///
    /// When an index appears more than once, all occurrences share a single
    /// [`Tabstop`] with one range per occurrence, and every occurrence renders
    /// the placeholder's resolved text at expansion time (`${1:v}` then `$1`
    /// yields `v v`). The placeholder text is resolved to a fixed point first,
    /// so a mirror that appears *before* its defining `${N:…}` in the body
    /// still renders the right text.
    ///
    /// ## Offsets and ordering
    ///
    /// [`CharRange`] offsets are char (Unicode scalar) offsets into
    /// [`Expansion::text`], not bytes. [`Expansion::tabstops`] is sorted in
    /// visit order (ascending index, then `0` last) and always ends with a `$0`
    /// stop — the body's own, or an implicit empty one at the end of the text.
    pub fn expand(&self, resolve_var: &dyn Fn(&str) -> Option<String>) -> Expansion {
        // --- Phase 1: resolve each index's placeholder text to a fixed point.
        //
        // Mirror text must be known even for a mirror that appears before the
        // `${N:…}` that defines it, and a placeholder may reference another
        // tabstop's placeholder (`${1: $2 }` with `${2:x}` elsewhere). Iterating
        // the resolution to a fixed point handles both, including forward
        // references, and terminates in at most `defs + 1` rounds (each round
        // resolves at least one more index, or none changes and we stop).
        let mut meta = Meta::default();
        collect(&self.nodes, &mut meta);

        // Seed: choices resolve immediately to their first option; placeholder
        // indices start empty and are filled by the fixed-point loop.
        for (index, options) in &meta.choices {
            meta.placeholder.insert(*index, options.first().cloned().unwrap_or_default());
        }
        for index in meta.ph_children.keys() {
            meta.placeholder.entry(*index).or_default();
        }

        let defs: Vec<(u32, Vec<Node>)> =
            meta.ph_children.iter().map(|(k, v)| (*k, v.clone())).collect();
        for _ in 0..defs.len() + 1 {
            let mut updates = Vec::new();
            for (index, children) in &defs {
                let mut sink = StringSink::default();
                render(children, resolve_var, &meta, &mut sink);
                if meta.placeholder.get(index).map(String::as_str) != Some(sink.text.as_str()) {
                    updates.push((*index, sink.text));
                }
            }
            if updates.is_empty() {
                break;
            }
            for (index, text) in updates {
                meta.placeholder.insert(index, text);
            }
        }

        // --- Phase 2: render the body, recording char-offset ranges.
        let mut rec = RecordSink::default();
        render(&self.nodes, resolve_var, &meta, &mut rec);

        // --- Build the visit-ordered tabstop list.
        //
        // `rec.ranges` is a BTreeMap, so iteration is ascending by index and
        // index 0 (the smallest) comes first — we peel it off and re-append it
        // last to honour the "$0 visited last" rule.
        let mut tabstops = Vec::new();
        let mut zero: Option<Tabstop> = None;
        for (index, ranges) in rec.ranges {
            let stop = Tabstop {
                index,
                ranges,
                placeholder: meta.placeholder.get(&index).cloned(),
                choices: meta.choices.get(&index).cloned().unwrap_or_default(),
            };
            if index == 0 {
                zero = Some(stop);
            } else {
                tabstops.push(stop);
            }
        }
        match zero {
            Some(stop) => tabstops.push(stop),
            None => tabstops.push(Tabstop {
                index: 0,
                ranges: vec![CharRange { start: rec.len, end: rec.len }],
                placeholder: None,
                choices: Vec::new(),
            }),
        }

        Expansion { text: rec.text, tabstops }
    }
}

// ---------------------------------------------------------------------------
// Parsed representation
// ---------------------------------------------------------------------------

/// One node of a parsed snippet body. Private — the public surface exposes only
/// the expanded [`Expansion`].
#[derive(Debug, Clone, PartialEq, Eq)]
enum Node {
    /// Literal text (escapes already resolved).
    Text(String),
    /// A bare tabstop: `$1` or `${1}`.
    Tabstop(u32),
    /// A placeholder tabstop `${1:…}`; the children are the (recursively
    /// parsed) placeholder body.
    Placeholder(u32, Vec<Node>),
    /// A choice tabstop `${1|a,b,c|}`; options are plain strings.
    Choice(u32, Vec<String>),
    /// A variable `$VAR` / `${VAR}` / `${VAR:default}`; the default is the
    /// (possibly empty) recursively parsed fallback body.
    Variable(String, Vec<Node>),
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// A char-cursor recursive-descent parser. Works over a `Vec<char>` so that
/// error positions and escape handling are in char units (matching the public
/// [`CharRange`] contract) rather than bytes.
struct Parser {
    chars: Vec<char>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.chars.get(self.pos).copied();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    /// Parse a run of nodes. When `in_braces`, an unescaped `}` ends the run
    /// (the caller consumes it); at top level `}` is literal text.
    fn parse_seq(&mut self, in_braces: bool) -> Result<Vec<Node>, ParseError> {
        let mut nodes = Vec::new();
        let mut text = String::new();
        loop {
            match self.peek() {
                None => break,
                Some('}') if in_braces => break,
                Some('$') => {
                    if !text.is_empty() {
                        nodes.push(Node::Text(std::mem::take(&mut text)));
                    }
                    nodes.push(self.parse_dollar()?);
                }
                // Escapes recognised in text/placeholder/default bodies. A
                // backslash before anything else is a literal backslash (the
                // spec only blesses `\$ \\ \}`), so `\n` stays two chars.
                Some('\\') => {
                    self.bump();
                    match self.peek() {
                        Some(c @ ('$' | '\\' | '}')) => {
                            text.push(c);
                            self.bump();
                        }
                        _ => text.push('\\'),
                    }
                }
                Some(c) => {
                    text.push(c);
                    self.bump();
                }
            }
        }
        if !text.is_empty() {
            nodes.push(Node::Text(text));
        }
        Ok(nodes)
    }

    /// Parse a construct beginning at `$` (which is the current char).
    fn parse_dollar(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        self.bump(); // consume '$'
        match self.peek() {
            Some('{') => {
                self.bump();
                self.parse_braced(start)
            }
            Some(c) if c.is_ascii_digit() => Ok(Node::Tabstop(self.parse_int())),
            Some(c) if is_var_start(c) => Ok(Node::Variable(self.parse_var(), Vec::new())),
            // A lone `$` (end of input, `$ `, `$$`, …) is literal text.
            _ => Ok(Node::Text("$".to_string())),
        }
    }

    /// Parse the body of a `${…}` after the opening `${` has been consumed.
    /// `start` is the char offset of the opening `$`, used for error reporting.
    fn parse_braced(&mut self, start: usize) -> Result<Node, ParseError> {
        match self.peek() {
            // ${1}, ${1:…}, ${1|…|}, or an unsupported ${1/…/…/}.
            Some(c) if c.is_ascii_digit() => {
                let index = self.parse_int();
                match self.peek() {
                    Some('}') => {
                        self.bump();
                        Ok(Node::Tabstop(index))
                    }
                    Some(':') => {
                        self.bump();
                        let children = self.parse_seq(true)?;
                        self.expect_close(start)?;
                        Ok(Node::Placeholder(index, children))
                    }
                    Some('|') => {
                        self.bump();
                        let options = self.parse_choice(start)?;
                        Ok(Node::Choice(index, options))
                    }
                    Some('/') => Err(ParseError::UnsupportedTransform { at: start }),
                    None => Err(ParseError::UnbalancedBrace { at: start }),
                    _ => Err(ParseError::InvalidBody { at: start }),
                }
            }
            // ${VAR}, ${VAR:default}, or an unsupported ${VAR/…/…/}.
            Some(c) if is_var_start(c) => {
                let name = self.parse_var();
                match self.peek() {
                    Some('}') => {
                        self.bump();
                        Ok(Node::Variable(name, Vec::new()))
                    }
                    Some(':') => {
                        self.bump();
                        let default = self.parse_seq(true)?;
                        self.expect_close(start)?;
                        Ok(Node::Variable(name, default))
                    }
                    Some('/') => Err(ParseError::UnsupportedTransform { at: start }),
                    None => Err(ParseError::UnbalancedBrace { at: start }),
                    _ => Err(ParseError::InvalidBody { at: start }),
                }
            }
            None => Err(ParseError::UnbalancedBrace { at: start }),
            // `${}`, `${:x}`, `${|…}` — no name/number where one is required.
            _ => Err(ParseError::EmptyName { at: start }),
        }
    }

    /// Parse a choice list after the opening `|` of `${N|…|}` has been
    /// consumed. Consumes through the terminating `|}`. Options are plain text
    /// with `\, \| \} \\` escapes; empty options are allowed except when the
    /// whole list is empty (`${1||}`).
    fn parse_choice(&mut self, start: usize) -> Result<Vec<String>, ParseError> {
        let mut options = Vec::new();
        let mut current = String::new();
        loop {
            match self.peek() {
                None => return Err(ParseError::InvalidChoice { at: start }),
                Some('|') => {
                    self.bump();
                    if self.peek() == Some('}') {
                        self.bump();
                    } else {
                        return Err(ParseError::InvalidChoice { at: start });
                    }
                    break;
                }
                Some(',') => {
                    self.bump();
                    options.push(std::mem::take(&mut current));
                }
                Some('\\') => {
                    self.bump();
                    match self.peek() {
                        Some(c @ (',' | '|' | '}' | '\\')) => {
                            current.push(c);
                            self.bump();
                        }
                        _ => current.push('\\'),
                    }
                }
                Some(c) => {
                    current.push(c);
                    self.bump();
                }
            }
        }
        options.push(current);
        if options.len() == 1 && options[0].is_empty() {
            return Err(ParseError::InvalidChoice { at: start });
        }
        Ok(options)
    }

    /// Consume the expected closing `}` of a `${…}` body, or report the brace
    /// (opened at `start`) as unbalanced.
    fn expect_close(&mut self, start: usize) -> Result<(), ParseError> {
        if self.peek() == Some('}') {
            self.bump();
            Ok(())
        } else {
            Err(ParseError::UnbalancedBrace { at: start })
        }
    }

    /// Parse a run of ASCII digits into a `u32`, saturating on overflow. A
    /// tabstop index that overflows `u32` is absurd in practice; saturating
    /// keeps parsing total rather than inventing an error for a pathological
    /// input.
    fn parse_int(&mut self) -> u32 {
        let mut value: u64 = 0;
        while let Some(c) = self.peek() {
            if let Some(d) = c.to_digit(10) {
                value = value.saturating_mul(10).saturating_add(u64::from(d));
                self.bump();
            } else {
                break;
            }
        }
        value.min(u64::from(u32::MAX)) as u32
    }

    /// Parse a variable name: `[A-Za-z_][A-Za-z0-9_]*` (Unicode-lenient on the
    /// letter classes). The first char is guaranteed a valid start by the
    /// caller.
    fn parse_var(&mut self) -> String {
        let mut name = String::new();
        while let Some(c) = self.peek() {
            if c == '_' || c.is_alphanumeric() {
                name.push(c);
                self.bump();
            } else {
                break;
            }
        }
        name
    }
}

fn is_var_start(c: char) -> bool {
    c == '_' || c.is_alphabetic()
}

// ---------------------------------------------------------------------------
// Expansion
// ---------------------------------------------------------------------------

/// Resolved per-index metadata gathered before rendering: which indices have a
/// defining placeholder body, which are choices, and the fixed-point-resolved
/// mirror text for each defined index.
#[derive(Default)]
struct Meta {
    /// index -> the children of the first `${index:…}` placeholder in the body.
    ph_children: BTreeMap<u32, Vec<Node>>,
    /// index -> the options of the first `${index|…|}` choice in the body.
    choices: BTreeMap<u32, Vec<String>>,
    /// index -> resolved placeholder/mirror text. Populated only for indices
    /// that have a placeholder or choice definition; a bare-only index has no
    /// entry (its [`Tabstop::placeholder`] is `None`).
    placeholder: BTreeMap<u32, String>,
}

impl Meta {
    /// The resolved placeholder text for `index`, if it has a definition.
    fn placeholder_of(&self, index: u32) -> Option<&str> {
        self.placeholder.get(&index).map(String::as_str)
    }
}

/// Walk the node tree recording, per index, the *first* placeholder body or
/// choice list (whichever appears first in document order defines the index).
fn collect(nodes: &[Node], meta: &mut Meta) {
    for node in nodes {
        match node {
            Node::Text(_) => {}
            Node::Tabstop(_) => {}
            Node::Placeholder(index, children) => {
                if !meta.ph_children.contains_key(index) && !meta.choices.contains_key(index) {
                    meta.ph_children.insert(*index, children.clone());
                }
                collect(children, meta);
            }
            Node::Choice(index, options) => {
                if !meta.ph_children.contains_key(index) && !meta.choices.contains_key(index) {
                    meta.choices.insert(*index, options.clone());
                }
            }
            Node::Variable(_, default) => collect(default, meta),
        }
    }
}

/// A rendering target. Both the fixed-point pre-pass and the real pass share
/// [`render`]; the pre-pass uses [`StringSink`] (text only, ranges discarded)
/// and the real pass uses [`RecordSink`] (text plus char-offset ranges).
trait Sink {
    fn push(&mut self, text: &str);
    /// Current position in **char** offsets.
    fn pos(&self) -> usize;
    fn record(&mut self, index: u32, range: CharRange);
}

/// Text-only sink used to resolve mirror/placeholder text to a fixed point.
#[derive(Default)]
struct StringSink {
    text: String,
    len: usize,
}

impl Sink for StringSink {
    fn push(&mut self, text: &str) {
        self.text.push_str(text);
        self.len += text.chars().count();
    }
    fn pos(&self) -> usize {
        self.len
    }
    fn record(&mut self, _index: u32, _range: CharRange) {}
}

/// Full sink for the real expansion pass: accumulates text and, per index, the
/// char-offset ranges (in document order) where that index landed.
#[derive(Default)]
struct RecordSink {
    text: String,
    len: usize,
    ranges: BTreeMap<u32, Vec<CharRange>>,
}

impl Sink for RecordSink {
    fn push(&mut self, text: &str) {
        self.text.push_str(text);
        self.len += text.chars().count();
    }
    fn pos(&self) -> usize {
        self.len
    }
    fn record(&mut self, index: u32, range: CharRange) {
        self.ranges.entry(index).or_default().push(range);
    }
}

/// Render `nodes` into `sink`, resolving variables via `resolve` and mirror
/// text via `meta`.
///
/// The mirror invariant: a `Placeholder` node renders **its own children** in
/// place (so nested tabstops inside it get their real ranges), while a bare
/// `Tabstop` renders `meta`'s resolved placeholder text flat (it is a mirror).
/// Because `meta` was resolved to a fixed point beforehand, the two agree, and
/// mirrors that appear before their defining placeholder still render correctly.
fn render<S: Sink>(
    nodes: &[Node],
    resolve: &dyn Fn(&str) -> Option<String>,
    meta: &Meta,
    sink: &mut S,
) {
    for node in nodes {
        match node {
            Node::Text(text) => sink.push(text),
            Node::Tabstop(index) => {
                let start = sink.pos();
                if let Some(text) = meta.placeholder_of(*index) {
                    sink.push(text);
                }
                let end = sink.pos();
                sink.record(*index, CharRange { start, end });
            }
            Node::Placeholder(index, children) => {
                let start = sink.pos();
                render(children, resolve, meta, sink);
                let end = sink.pos();
                sink.record(*index, CharRange { start, end });
            }
            Node::Choice(index, options) => {
                let start = sink.pos();
                if let Some(first) = options.first() {
                    sink.push(first);
                }
                let end = sink.pos();
                sink.record(*index, CharRange { start, end });
            }
            Node::Variable(name, default) => match resolve(name) {
                Some(value) => sink.push(&value),
                // Unresolved: expand the default in place (its tabstops count),
                // or nothing at all. See the unknown-variable policy in
                // `Snippet::expand`'s docs — never the literal name.
                None => render(default, resolve, meta, sink),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A resolver that returns nothing (every variable unresolved).
    fn none(_: &str) -> Option<String> {
        None
    }

    /// Convenience: parse + expand with the empty resolver.
    fn expand(body: &str) -> Expansion {
        Snippet::parse(body).unwrap().expand(&none)
    }

    fn index_order(exp: &Expansion) -> Vec<u32> {
        exp.tabstops.iter().map(|t| t.index).collect()
    }

    fn stop(exp: &Expansion, index: u32) -> &Tabstop {
        exp.tabstops.iter().find(|t| t.index == index).expect("tabstop present")
    }

    // --- Plain text and the implicit final stop -----------------------------

    #[test]
    fn plain_text_gets_implicit_final_stop() {
        let exp = expand("hello");
        assert_eq!(exp.text, "hello");
        assert_eq!(index_order(&exp), vec![0]);
        assert_eq!(stop(&exp, 0).ranges, vec![CharRange { start: 5, end: 5 }]);
        assert_eq!(stop(&exp, 0).placeholder, None);
    }

    #[test]
    fn empty_body_is_empty_with_final_stop() {
        let exp = expand("");
        assert_eq!(exp.text, "");
        assert_eq!(stop(&exp, 0).ranges, vec![CharRange { start: 0, end: 0 }]);
    }

    // --- Bare tabstops ------------------------------------------------------

    #[test]
    fn single_bare_tabstop() {
        let exp = expand("a$1b");
        assert_eq!(exp.text, "ab");
        assert_eq!(stop(&exp, 1).ranges, vec![CharRange { start: 1, end: 1 }]);
        assert_eq!(stop(&exp, 1).placeholder, None);
        assert_eq!(index_order(&exp), vec![1, 0]);
    }

    #[test]
    fn braced_bare_tabstop_equivalent_to_unbraced() {
        assert_eq!(expand("a${1}b"), expand("a$1b"));
    }

    #[test]
    fn explicit_final_stop_is_not_duplicated() {
        let exp = expand("a$0b");
        assert_eq!(exp.text, "ab");
        assert_eq!(index_order(&exp), vec![0]);
        assert_eq!(stop(&exp, 0).ranges, vec![CharRange { start: 1, end: 1 }]);
    }

    // --- Placeholders -------------------------------------------------------

    #[test]
    fn simple_placeholder() {
        let exp = expand("${1:foo}");
        assert_eq!(exp.text, "foo");
        let s = stop(&exp, 1);
        assert_eq!(s.ranges, vec![CharRange { start: 0, end: 3 }]);
        assert_eq!(s.placeholder.as_deref(), Some("foo"));
        assert!(s.choices.is_empty());
    }

    #[test]
    fn nested_placeholder() {
        let exp = expand("${1:a${2:b}c}");
        assert_eq!(exp.text, "abc");
        assert_eq!(stop(&exp, 1).ranges, vec![CharRange { start: 0, end: 3 }]);
        assert_eq!(stop(&exp, 1).placeholder.as_deref(), Some("abc"));
        assert_eq!(stop(&exp, 2).ranges, vec![CharRange { start: 1, end: 2 }]);
        assert_eq!(stop(&exp, 2).placeholder.as_deref(), Some("b"));
        assert_eq!(index_order(&exp), vec![1, 2, 0]);
    }

    // --- Choices ------------------------------------------------------------

    #[test]
    fn choice_tabstop() {
        let exp = expand("${1|x,y,z|}");
        assert_eq!(exp.text, "x");
        let s = stop(&exp, 1);
        assert_eq!(s.ranges, vec![CharRange { start: 0, end: 1 }]);
        assert_eq!(s.placeholder.as_deref(), Some("x"));
        assert_eq!(s.choices, vec!["x", "y", "z"]);
    }

    #[test]
    fn choice_with_escaped_delimiters() {
        // First option contains a literal comma, second a literal pipe.
        let exp = expand(r"${1|a\,b,c\|d|}");
        assert_eq!(exp.text, "a,b");
        assert_eq!(stop(&exp, 1).choices, vec!["a,b", "c|d"]);
    }

    #[test]
    fn choice_mirror_carries_first_option() {
        let exp = expand("${1|a,b|} $1");
        assert_eq!(exp.text, "a a");
        let s = stop(&exp, 1);
        assert_eq!(s.ranges, vec![CharRange { start: 0, end: 1 }, CharRange { start: 2, end: 3 }]);
        assert_eq!(s.choices, vec!["a", "b"]);
    }

    // --- Variables ----------------------------------------------------------

    #[test]
    fn variable_resolved() {
        let snip = Snippet::parse("$TM_FILENAME").unwrap();
        let exp = snip.expand(&|name| (name == "TM_FILENAME").then(|| "main.rs".to_string()));
        assert_eq!(exp.text, "main.rs");
        assert_eq!(stop(&exp, 0).ranges, vec![CharRange { start: 7, end: 7 }]);
    }

    #[test]
    fn braced_variable_resolved() {
        let snip = Snippet::parse("pre ${NAME} post").unwrap();
        let exp = snip.expand(&|_| Some("X".to_string()));
        assert_eq!(exp.text, "pre X post");
    }

    #[test]
    fn variable_default_used_when_unresolved() {
        let exp = expand("${VAR:def}");
        assert_eq!(exp.text, "def");
    }

    #[test]
    fn variable_default_ignored_when_resolved() {
        let snip = Snippet::parse("${VAR:def}").unwrap();
        let exp = snip.expand(&|_| Some("real".to_string()));
        assert_eq!(exp.text, "real");
    }

    #[test]
    fn variable_unknown_expands_to_empty() {
        // The documented policy: unresolved + no default -> empty, never the
        // literal name.
        let exp = expand("$UNKNOWN");
        assert_eq!(exp.text, "");
        let braced = expand("${UNKNOWN}");
        assert_eq!(braced.text, "");
    }

    #[test]
    fn variable_default_can_contain_tabstop() {
        // When the variable is unresolved, the default's tabstop shows up.
        let exp = expand("${VAR:${1:x}}");
        assert_eq!(exp.text, "x");
        assert_eq!(stop(&exp, 1).ranges, vec![CharRange { start: 0, end: 1 }]);
        assert_eq!(stop(&exp, 1).placeholder.as_deref(), Some("x"));
    }

    #[test]
    fn variable_default_tabstop_absent_when_resolved() {
        let snip = Snippet::parse("${VAR:${1:x}}").unwrap();
        let exp = snip.expand(&|_| Some("Y".to_string()));
        assert_eq!(exp.text, "Y");
        // No numbered tabstop, only the implicit final stop.
        assert_eq!(index_order(&exp), vec![0]);
    }

    // --- Mirrors ------------------------------------------------------------

    #[test]
    fn mirror_after_definition() {
        let exp = expand("${1:v} $1");
        assert_eq!(exp.text, "v v");
        let s = stop(&exp, 1);
        assert_eq!(s.ranges, vec![CharRange { start: 0, end: 1 }, CharRange { start: 2, end: 3 }]);
        assert_eq!(s.placeholder.as_deref(), Some("v"));
    }

    #[test]
    fn mirror_before_definition() {
        // The bare $1 comes first but must still render the placeholder text.
        let exp = expand("$1 ${1:v}");
        assert_eq!(exp.text, "v v");
        let s = stop(&exp, 1);
        assert_eq!(s.ranges, vec![CharRange { start: 0, end: 1 }, CharRange { start: 2, end: 3 }]);
        assert_eq!(s.placeholder.as_deref(), Some("v"));
    }

    #[test]
    fn mirror_of_placeholder_referencing_another_tabstop() {
        // Fixed-point resolution: $1's text depends on $2's placeholder.
        let exp = expand("${1:<$2>} ${2:y} $1");
        assert_eq!(exp.text, "<y> y <y>");
        assert_eq!(stop(&exp, 1).placeholder.as_deref(), Some("<y>"));
        assert_eq!(stop(&exp, 1).ranges.len(), 2);
    }

    // --- Escapes ------------------------------------------------------------

    #[test]
    fn top_level_escapes() {
        let exp = expand(r"\$1 \} \\");
        assert_eq!(exp.text, r"$1 } \");
        // No tabstops but the implicit final stop.
        assert_eq!(index_order(&exp), vec![0]);
    }

    #[test]
    fn escape_inside_placeholder() {
        let exp = expand(r"${1:a\}b}");
        assert_eq!(exp.text, "a}b");
        assert_eq!(stop(&exp, 1).placeholder.as_deref(), Some("a}b"));
    }

    #[test]
    fn backslash_before_ordinary_char_is_literal() {
        // `\n` is not a recognised escape: it stays a backslash then an 'n'.
        let exp = expand(r"a\nb");
        assert_eq!(exp.text, r"a\nb");
    }

    #[test]
    fn lone_dollar_is_literal() {
        let exp = expand("cost is $ 5");
        assert_eq!(exp.text, "cost is $ 5");
        assert_eq!(index_order(&exp), vec![0]);
    }

    // --- Char offsets (not bytes) -------------------------------------------

    #[test]
    fn offsets_are_char_based_with_multibyte() {
        // "café" is 4 chars but 5 bytes; the range must be [0,4].
        let exp = expand("${1:café}x");
        assert_eq!(exp.text, "caféx");
        assert_eq!(stop(&exp, 1).ranges, vec![CharRange { start: 0, end: 4 }]);
        assert_eq!(stop(&exp, 0).ranges, vec![CharRange { start: 5, end: 5 }]);
    }

    #[test]
    fn offsets_are_char_based_with_astral() {
        // 😀 is a single Unicode scalar (char) but 4 UTF-8 bytes. A byte-based
        // implementation would report end: 4 here; we require end: 1.
        let exp = expand("${1:😀}x");
        assert_eq!(exp.text, "😀x");
        assert_eq!(stop(&exp, 1).ranges, vec![CharRange { start: 0, end: 1 }]);
        // 'x' sits at char offset 1; the implicit final stop at 2.
        assert_eq!(stop(&exp, 0).ranges, vec![CharRange { start: 2, end: 2 }]);
    }

    // --- Visit order --------------------------------------------------------

    #[test]
    fn visit_order_is_ascending_then_zero_last() {
        let exp = expand("$3$1$2$0");
        assert_eq!(index_order(&exp), vec![1, 2, 3, 0]);
    }

    #[test]
    fn visit_order_without_explicit_zero() {
        let exp = expand("$2 $1");
        assert_eq!(index_order(&exp), vec![1, 2, 0]);
    }

    // --- Errors -------------------------------------------------------------

    #[test]
    fn unbalanced_brace_placeholder() {
        assert_eq!(Snippet::parse("${1:foo"), Err(ParseError::UnbalancedBrace { at: 0 }));
    }

    #[test]
    fn unbalanced_brace_eof_right_after_open() {
        assert_eq!(Snippet::parse("a ${"), Err(ParseError::UnbalancedBrace { at: 2 }));
    }

    #[test]
    fn unbalanced_brace_variable() {
        assert_eq!(Snippet::parse("${VAR"), Err(ParseError::UnbalancedBrace { at: 0 }));
    }

    #[test]
    fn empty_name_errors() {
        assert_eq!(Snippet::parse("${}"), Err(ParseError::EmptyName { at: 0 }));
        assert_eq!(Snippet::parse("x ${:y}"), Err(ParseError::EmptyName { at: 2 }));
    }

    #[test]
    fn invalid_body_errors() {
        assert_eq!(Snippet::parse("${1x}"), Err(ParseError::InvalidBody { at: 0 }));
    }

    #[test]
    fn invalid_choice_errors() {
        assert_eq!(Snippet::parse("${1||}"), Err(ParseError::InvalidChoice { at: 0 }));
        assert_eq!(Snippet::parse("${1|a,b"), Err(ParseError::InvalidChoice { at: 0 }));
        assert_eq!(Snippet::parse("${1|a,b|x}"), Err(ParseError::InvalidChoice { at: 0 }));
    }

    #[test]
    fn transforms_are_rejected() {
        assert_eq!(
            Snippet::parse("${1/foo/bar/}"),
            Err(ParseError::UnsupportedTransform { at: 0 })
        );
        assert_eq!(
            Snippet::parse("${TM_FILENAME/.*/$0/}"),
            Err(ParseError::UnsupportedTransform { at: 0 })
        );
    }

    #[test]
    fn error_offset_is_char_based() {
        // An astral char before the bad construct: `at` must count chars, not
        // bytes. "😀" is one char, so the `${` opens at char offset 1.
        assert_eq!(Snippet::parse("😀${1:x"), Err(ParseError::UnbalancedBrace { at: 1 }));
    }

    // --- Realistic snippet + reusability ------------------------------------

    #[test]
    fn realistic_function_snippet() {
        let snip = Snippet::parse("fn ${1:name}(${2:args}) {\n\t$0\n}").unwrap();
        let exp = snip.expand(&none);
        assert_eq!(exp.text, "fn name(args) {\n\t\n}");
        assert_eq!(index_order(&exp), vec![1, 2, 0]);
        assert_eq!(stop(&exp, 1).placeholder.as_deref(), Some("name"));
        assert_eq!(stop(&exp, 2).placeholder.as_deref(), Some("args"));
        // $0 sits between the tab and the newline before the closing brace.
        let zero = stop(&exp, 0).ranges[0];
        assert_eq!(&exp.text[..], "fn name(args) {\n\t\n}");
        assert_eq!(zero.start, zero.end);
    }

    #[test]
    fn snippet_is_reusable_across_resolvers() {
        let snip = Snippet::parse("${VAR:fallback}").unwrap();
        assert_eq!(snip.expand(&none).text, "fallback");
        assert_eq!(snip.expand(&|_| Some("actual".to_string())).text, "actual");
    }

    #[test]
    fn parse_error_implements_display_and_error() {
        let err = Snippet::parse("${1:x").unwrap_err();
        // Display is non-empty and mentions the position.
        assert!(err.to_string().contains("char 0"));
        let _dyn: &dyn std::error::Error = &err;
    }
}

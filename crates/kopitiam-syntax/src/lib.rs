//! Pure-Rust, line-oriented incremental syntax highlighting.
//!
//! # Why this crate exists instead of `tree-sitter`
//!
//! The maintainer's brief was "incorporate tree-sitter in pure Rust." That
//! is not achievable today — see `docs/ai-decisions/AID-0009-syntax-highlighting.md`
//! for the full investigation, but the short version:
//!
//! * Tree-sitter's core parsing engine is C. The `tree-sitter` crate on
//!   crates.io is a *binding*, not a reimplementation.
//! * There is exactly one attempt at a pure-Rust core (`tree-sitter-c2rust`,
//!   an unofficial fork by a third party), and it is a mechanical,
//!   automated C→Rust transpilation: `#[repr(C)]` structs, `libc::c_int`
//!   fields, raw-pointer arithmetic wrapped in `unsafe` almost everywhere.
//!   It compiles without `cc`, but it is not "a good Rust implementation"
//!   in the sense CLAUDE.md means by that phrase — it is C wearing a Rust
//!   file extension.
//! * That covers only the *core runtime*. Every grammar (`tree-sitter-rust`,
//!   `tree-sitter-lua`, ...) is a separate crate that ships a
//!   tool-generated `parser.c` and a `build.rs` that pulls in the `cc`
//!   crate to compile it. Confirmed directly: adding `tree-sitter-rust` to
//!   a scratch `Cargo.toml` and running `cargo fetch` pulls in `cc` and
//!   `find-msvc-tools`. Nobody has transpiled the grammar ecosystem; even
//!   `syntastica`, the most complete tree-sitter-grammar bundle on
//!   crates.io (68+ languages) and the one project that *does* offer a
//!   `runtime-c2rust` core option, still depends on the ordinary
//!   C-backed `tree-sitter-<lang>` crates for every grammar.
//!
//! So "pure-Rust tree-sitter" would mean transpiling and maintaining
//! six grammars ourselves, forever, out of band from upstream — a standing
//! tax with no clear payoff, since (see below) the editor never needed a
//! parse tree to begin with.
//!
//! `syntect` was also evaluated. It is much closer to pure Rust than
//! tree-sitter: its default feature (`default-onig`) links the C
//! Oniguruma library via the `onig` crate, but it also ships a
//! `default-fancy` feature built on `fancy-regex`, which really is pure
//! Rust (checked directly: its dependency graph is `regex-automata`,
//! `regex-syntax`, `bit-set` — no `cc`, no FFI). A naive `cargo add
//! syntect` silently takes the C path; only the fancy-regex feature keeps
//! the promise. But syntect's unit of work is a Sublime `.sublime-syntax`
//! grammar — a full regex-based pushdown automaton designed for parsing
//! *someone else's* syntax definitions for dozens of languages we don't
//! use. For six languages we control, hand-written line lexers are less
//! code, no vendored `.sublime-syntax` files to source and license-audit,
//! and — most importantly — total control over exactly how state carries
//! across a line boundary, which is the one part of this problem that is
//! genuinely easy to get subtly wrong.
//!
//! # What the editor actually needs
//!
//! `kopitiam-neovim`'s `TextArea` widget (read, not touched, by this
//! crate) renders a buffer one visible row at a time and wants
//! `ratatui::style::Style` per grapheme range. It has no use for a parse
//! tree, node kinds, or tree-sitter's incremental-reparse edit API — it
//! wants exactly what [`Highlighter::highlight_line`] returns: a flat list
//! of `[start, end)` byte ranges tagged with a [`HighlightKind`]. Tree-sitter
//! is a correct tool for "build me a queryable AST for refactoring/LSP
//! tooling"; that is a real future need (see `kopitiam-semantic`), but it
//! is not *this* need, and reaching for it here would be cargo-culting
//! Neovim's plugin ecosystem rather than solving KOPITIAM's actual
//! problem.
//!
//! # Incrementality
//!
//! A line-based highlighter's classic failure mode is losing state at line
//! boundaries: a block comment or multi-line string that "closes" the
//! moment the cursor crosses a newline, because each line was highlighted
//! in isolation. This crate avoids that by threading an explicit
//! [`LineState`] value from the end of one line into the start of the
//! next (see each language module's `*State` enum — e.g. `RustState`'s
//! `BlockComment` variant).
//!
//! [`Highlighter`] exposes that state so a caller can re-highlight
//! *incrementally* rather than from the top of the file on every
//! keystroke:
//!
//! 1. Cache the [`LineState`] present at the *start* of every line (the
//!    state after highlighting the previous line).
//! 2. When line `n` is edited, resume a highlighter with
//!    [`Highlighter::with_state`] using the cached state for line `n`,
//!    highlight it, and inspect the resulting state via
//!    [`Highlighter::state`].
//! 3. If that state equals the cached state for line `n + 1`, stop — every
//!    line after `n` is provably unaffected, because line lexing is a pure
//!    function of `(state_in, line_text)`. If it differs, re-highlight
//!    line `n + 1` with the new state and repeat.
//!
//! In the overwhelmingly common case (typing inside a single-line
//! statement, nowhere near a comment or string), step 3 terminates after
//! one line: O(1) work per keystroke, not O(file length). The only edits
//! that cascade are ones that genuinely change downstream meaning — typing
//! an opening `/*` with no matching `*/` on the same line, for instance,
//! for which re-highlighting forward *is* the correct behaviour, not a
//! performance bug.
//!
//! Each per-language scanner is a single linear pass over the line with no
//! backtracking, so a pathological line (megabytes long, or an unterminated
//! string) costs O(line length) and cannot hang or panic — see the
//! `pathological_input` tests in each language module.

mod lua;
mod markdown;
mod python;
mod rust;
mod tex;
mod toml;
mod util;

/// The languages this crate knows how to highlight.
///
/// Matches the maintainer's actual editing surface (see CLAUDE.md's list of
/// domains and the languages used across this very workspace: Rust for the
/// platform, Lua for `kopitiam-neovim`/`kvim` config, TeX for scientific
/// publishing, Markdown for docs/AIDs/ADRs, TOML for `Cargo.toml`, Python
/// for the wider scientific-computing ecosystem this project talks to)
/// rather than an attempt at exhaustive language coverage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    Rust,
    Lua,
    Tex,
    Markdown,
    Toml,
    Python,
}

impl Language {
    /// Best-effort guess from a filename extension (without the leading
    /// `.`), case-insensitively. Returns `None` for anything unrecognised
    /// rather than guessing — an editor should fall back to plain text, not
    /// a wrong grammar.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_ascii_lowercase().as_str() {
            "rs" => Some(Self::Rust),
            "lua" => Some(Self::Lua),
            "tex" | "sty" | "cls" | "bib" => Some(Self::Tex),
            "md" | "markdown" => Some(Self::Markdown),
            "toml" => Some(Self::Toml),
            "py" | "pyi" => Some(Self::Python),
            _ => None,
        }
    }
}

/// The semantic category a theme maps to a colour.
///
/// Deliberately theme-agnostic and deliberately *not* one variant per
/// language construct: the highlighter's job stops at "this text is a
/// string" or "this text is a comment," never "this text is a Rust raw
/// string with two hashes" — that distinction matters for *scanning* (see
/// `RustState::RawString`) but not for *colouring*, which is the only
/// thing a caller of this crate does with a [`HighlightSpan`].
///
/// Not every character of a line receives a span. Punctuation the
/// maintainer's theme has no reason to colour differently from plain text
/// (e.g. a Rust lifetime's leading `'`, in the current implementation) is
/// simply left uncovered; the renderer is expected to use the buffer's
/// default foreground for any byte range no span covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HighlightKind {
    /// Reserved words: `fn`, `if`, `local`, `def`, `true`, TeX's `\begin`
    /// (TeX has no real notion of "keyword" separate from "command," so
    /// every `\command` name is classified here — see `tex.rs`).
    Keyword,
    /// A type or type-like name: Rust's `UpperCamelCase` types and
    /// primitives, Python builtin type names, a TOML `[table.header]`.
    Type,
    /// String and character literal content, including delimiters. Also
    /// used for TeX math-mode spans (`$...$`), which are the closest TeX
    /// analogue to a delimited literal.
    String,
    /// An escape sequence *inside* a string (`\n`, `\u{2603}`, TeX's
    /// `\%`), highlighted distinctly from the surrounding string body so
    /// escapes are visually easy to spot.
    Escape,
    Comment,
    Number,
    /// A function/method name at a definition or (heuristically) a call
    /// site — the identifier immediately followed by `(`.
    Function,
    /// Multi-character operators (`==`, `->`, `..=`) grouped into one
    /// span each.
    Operator,
    /// Structural single characters: brackets, commas, semicolons.
    Punctuation,
    /// A macro invocation name (Rust's `println!`).
    Macro,
    /// An attribute or decorator: Rust's `#[derive(..)]`, Python's `@foo`.
    Attribute,
    /// A Markdown ATX heading line (`# Title`).
    Heading,
    /// Markdown bold/italic emphasis, delimiters included.
    Emphasis,
    /// A Markdown link (`[text](url)`), the whole construct as one span.
    Link,
    /// A Markdown inline code span or fenced code block (delimiters and
    /// body alike) — Markdown's embedded-code content has no language
    /// grammar of its own applied to it here (that would mean recursively
    /// invoking another language's `Highlighter`, a real future
    /// enhancement, not attempted in this first pass).
    CodeBlock,
}

/// One highlighted region of a line.
///
/// `start` and `end` are **byte offsets** into the `&str` passed to
/// [`Highlighter::highlight_line`] (not char or grapheme indices), so they
/// compose directly with ordinary `&line[start..end]` slicing. This
/// matches how every other byte-oriented Rust text API (including
/// `str::char_indices` and ropey's `Rope::byte_slice`, likely the eventual
/// buffer backing) already works — converting to grapheme-cluster indices
/// (what the renderer in `kopitiam-neovim`'s `textarea.rs` ultimately
/// positions the cursor with) is the render layer's job, not this one's,
/// because that conversion needs a `tabstop` and rendering context this
/// crate has no business knowing about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HighlightSpan {
    pub start: usize,
    pub end: usize,
    pub kind: HighlightKind,
}

/// The state carried from the end of one line into the start of the next.
///
/// This is the crux of the "don't lose highlighting at a line boundary"
/// requirement: a multi-line block comment or string is, from this crate's
/// point of view, nothing more than "the previous line ended still inside
/// one of these," represented here and threaded through
/// [`Highlighter::highlight_line`]. Every language starts a fresh buffer in
/// its own `Normal` variant; see [`LineState::initial`].
///
/// `Copy` and cheap to compare: every language's inner state is a small
/// enum (at most one `u32` nesting-depth payload), so equality-checking two
/// `LineState`s to decide whether incremental re-highlighting can stop
/// (see the crate-level docs) costs nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineState {
    Rust(rust::RustState),
    Lua(lua::LuaState),
    Tex(tex::TexState),
    Markdown(markdown::MarkdownState),
    Toml(toml::TomlState),
    Python(python::PythonState),
}

impl LineState {
    /// The state a freshly opened buffer (or a buffer's first line) starts
    /// in: "not inside any multi-line construct."
    pub fn initial(language: Language) -> Self {
        match language {
            Language::Rust => Self::Rust(rust::RustState::default()),
            Language::Lua => Self::Lua(lua::LuaState::default()),
            Language::Tex => Self::Tex(tex::TexState::default()),
            Language::Markdown => Self::Markdown(markdown::MarkdownState::default()),
            Language::Toml => Self::Toml(toml::TomlState::default()),
            Language::Python => Self::Python(python::PythonState::default()),
        }
    }
}

/// A stateful, single-language line highlighter.
///
/// Constructed once per buffer (or per resumed re-highlight — see
/// [`Highlighter::with_state`] and the crate-level incrementality docs) and
/// fed lines in order via [`Highlighter::highlight_line`]. Each call both
/// returns that line's spans and advances the highlighter's internal
/// [`LineState`] for the next call, so the caller never has to think about
/// carry-over state to get correct output for a single top-to-bottom pass —
/// only to *skip* work does it need [`Highlighter::state`] at all.
pub struct Highlighter {
    language: Language,
    state: LineState,
}

impl Highlighter {
    /// A highlighter for a fresh buffer: line 1 is highlighted assuming no
    /// multi-line construct is already open.
    pub fn new(language: Language) -> Self {
        Self { language, state: LineState::initial(language) }
    }

    /// A highlighter resumed mid-buffer, carrying whatever [`LineState`]
    /// was in effect at the start of the line about to be highlighted.
    /// This is what makes incremental re-highlighting possible: re-scan
    /// only the edited line (and, if its exit state changed, the lines
    /// after it) instead of the whole buffer. See the crate-level docs.
    pub fn with_state(language: Language, state: LineState) -> Self {
        Self { language, state }
    }

    /// The language this highlighter was constructed for.
    pub fn language(&self) -> Language {
        self.language
    }

    /// The [`LineState`] in effect right now: the state a line about to be
    /// passed to [`Highlighter::highlight_line`] will be scanned with, or
    /// — immediately after such a call — the state the *next* line should
    /// be scanned with. Callers doing incremental re-highlighting compare
    /// this against a cached "state at start of next line" to know whether
    /// they can stop propagating (see the crate-level docs).
    pub fn state(&self) -> LineState {
        self.state
    }

    /// Highlights one line of source text and advances the carried state
    /// for the next call.
    ///
    /// `line` must not contain a newline character — callers pass buffer
    /// lines with the terminator already stripped, matching how
    /// `kopitiam-neovim`'s `BufferView::line` already hands out lines (see
    /// `crates/kopitiam-neovim/src/ui/event.rs`, read-only reference).
    /// Nothing in this crate panics if a newline sneaks in, but any state
    /// this crate tracks (nested comment depth, an open string) is
    /// *within* a line only in the sense of "this is the text between two
    /// calls" — passing a multi-line chunk in one call defeats the whole
    /// per-line state-carrying design and produces spans a caller almost
    /// certainly did not intend.
    pub fn highlight_line(&mut self, line: &str) -> Vec<HighlightSpan> {
        let (spans, next_state) = match (self.language, self.state) {
            (Language::Rust, LineState::Rust(mut s)) => {
                let spans = rust::highlight_line(line, &mut s);
                (spans, LineState::Rust(s))
            }
            (Language::Lua, LineState::Lua(mut s)) => {
                let spans = lua::highlight_line(line, &mut s);
                (spans, LineState::Lua(s))
            }
            (Language::Tex, LineState::Tex(mut s)) => {
                let spans = tex::highlight_line(line, &mut s);
                (spans, LineState::Tex(s))
            }
            (Language::Markdown, LineState::Markdown(mut s)) => {
                let spans = markdown::highlight_line(line, &mut s);
                (spans, LineState::Markdown(s))
            }
            (Language::Toml, LineState::Toml(mut s)) => {
                let spans = toml::highlight_line(line, &mut s);
                (spans, LineState::Toml(s))
            }
            (Language::Python, LineState::Python(mut s)) => {
                let spans = python::highlight_line(line, &mut s);
                (spans, LineState::Python(s))
            }
            // A `LineState` for a different language than `self.language`
            // can only happen if a caller hand-built a mismatched pair via
            // `with_state` -- not reachable through this crate's own API,
            // since `LineState::initial` and every `highlight_line` call
            // always produce a state tagged with the highlighter's own
            // language. Rather than panic on a caller error that isn't
            // reachable through safe misuse of this crate alone, reset to
            // a fresh state for the correct language and highlight as if
            // this were line 1 -- degrades to "lost incremental state,"
            // never to a crash.
            (language, _) => {
                self.state = LineState::initial(language);
                return self.highlight_line(line);
            }
        };
        self.state = next_state;
        spans
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_extension_recognises_supported_languages() {
        assert_eq!(Language::from_extension("rs"), Some(Language::Rust));
        assert_eq!(Language::from_extension("LUA"), Some(Language::Lua));
        assert_eq!(Language::from_extension("py"), Some(Language::Python));
        assert_eq!(Language::from_extension("toml"), Some(Language::Toml));
        assert_eq!(Language::from_extension("tex"), Some(Language::Tex));
        assert_eq!(Language::from_extension("md"), Some(Language::Markdown));
        assert_eq!(Language::from_extension("exe"), None);
    }

    #[test]
    fn fresh_highlighter_starts_in_the_initial_state() {
        let h = Highlighter::new(Language::Rust);
        assert_eq!(h.state(), LineState::initial(Language::Rust));
    }

    #[test]
    fn with_state_resumes_from_a_carried_state() {
        let mut h = Highlighter::new(Language::Rust);
        h.highlight_line("/* unterminated");
        let mid_state = h.state();
        assert_ne!(mid_state, LineState::initial(Language::Rust));

        // A fresh highlighter resumed with that state, given the closing
        // line, should behave identically to continuing the original one.
        let mut resumed = Highlighter::with_state(Language::Rust, mid_state);
        let spans_resumed = resumed.highlight_line("still comment */ let x = 1;");

        let mut original = Highlighter::new(Language::Rust);
        original.highlight_line("/* unterminated");
        let spans_original = original.highlight_line("still comment */ let x = 1;");

        assert_eq!(spans_resumed, spans_original);
        assert_eq!(resumed.state(), original.state());
    }

    /// The incremental re-highlighting algorithm described in the
    /// crate-level docs: comparing `state()` after re-scanning an edited
    /// line against the cached state for the following line correctly
    /// detects "nothing downstream changed" for an edit that stays inside
    /// a single statement.
    #[test]
    fn state_equality_detects_when_downstream_lines_are_unaffected() {
        let lines = ["let a = 1;", "let b = 2;", "let c = 3;"];
        let mut h = Highlighter::new(Language::Rust);
        let mut states_at_start = vec![h.state()];
        for line in &lines {
            h.highlight_line(line);
            states_at_start.push(h.state());
        }

        // Edit line 1 (index 0) to something else that still closes on
        // the same line -- the state after re-highlighting it should
        // match the originally cached state for line 2, meaning an
        // incremental highlighter can stop right there.
        let mut edited = Highlighter::with_state(Language::Rust, states_at_start[0]);
        edited.highlight_line("let a = 100;");
        assert_eq!(edited.state(), states_at_start[1]);
    }
}

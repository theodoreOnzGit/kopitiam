//! Visual Basic language adapter — a **native Rust parser**, not a language
//! server.
//!
//! # Why this provider is not an LSP client
//!
//! Every other language adapter in this crate drives a real language server
//! over [`crate::lsp_client`] and lets that server be the deterministic source
//! of truth. Visual Basic cannot, because **there is no obtainable Visual Basic
//! language server**. This was checked rather than assumed, and the finding is
//! recorded here because it is the kind of thing a future contributor will
//! otherwise re-derive from scratch:
//!
//! * **The Roslyn compiler fully supports VB.NET** — `Microsoft.CodeAnalysis.VisualBasic`
//!   is a current, shipping NuGet package. That fact misleads people into
//!   assuming the Roslyn *language server* speaks VB. It does not.
//!   `Microsoft.CodeAnalysis.LanguageServer` (the server behind the VS Code C#
//!   extension, and the only modern standalone Roslyn LSP) references
//!   `Microsoft.CodeAnalysis.CSharp.Features` and **no VisualBasic project at
//!   all**. With no VB language service in its MEF composition, its
//!   `ProjectFileExtensionRegistry` cannot map `.vbproj` to a language, so a VB
//!   project cannot even be *loaded*, let alone queried.
//! * **Microsoft closed the request explicitly.** `dotnet/vscode-csharp#25`
//!   ("Fully Support Visual Basic") was opened in 2016 and closed
//!   *Resolved-By Design*. VB.NET's IDE experience lives in-process inside
//!   Visual Studio on Windows; it is not exposed over LSP.
//! * **OmniSharp is a false lead.** `nvim-lspconfig` lists `vb` among the
//!   `omnisharp` filetypes, which looks like support. OmniSharp-Roslyn is 99.7%
//!   C#, its README offers "C# language services", and the VB feature request
//!   (`OmniSharp/omnisharp-roslyn#1111`, 2018) was never implemented.
//! * **VBA** has one real project — `SSlinky/VBA-LanguageServer` (ANTLR +
//!   TypeScript, actively released) — but it ships only as a `.vsix` VS Code
//!   extension, not as a binary you can put on `PATH`, and it needs Node.
//!   Rubberduck's LSP is an in-process VBE COM add-in: Windows-only, and not a
//!   server you can spawn.
//! * **VB6 / classic VB** has nothing. It never did.
//!
//! So the choice was never "LSP or parser". It was "parser, or no VB support".
//! See `docs/ai-decisions/AID-0008-visual-basic-support.md` for the full
//! decision record, including what would make this the wrong call.
//!
//! This is also the *better* answer for KOPITIAM specifically. CLAUDE.md's
//! Translation Platform exists to make legacy code legible, and a dead VB6
//! codebase is the canonical instance of that problem — precisely the case an
//! LSP was never going to serve. VB's declaration grammar is small, regular and
//! line-oriented; parsing it is genuinely tractable in a way C++ is not.
//!
//! # Which dialects are supported
//!
//! | Dialect | Extensions | Status |
//! |---|---|---|
//! | **VB.NET** | `.vb`, `.vbproj` | Supported |
//! | **VBA** (Office macros) | `.bas`, `.cls`, `.frm` | Supported |
//! | **VB6 / classic VB** | `.bas`, `.cls`, `.frm`, `.vbp` | Supported |
//! | VBScript | `.vbs` | **Not** supported — different dialect, not claimed |
//!
//! VBA and VB6 share a declaration grammar and are handled by one code path
//! (`Dialect::Classic`); they are distinguished from VB.NET where the two
//! genuinely disagree (see [`Dialect`] and [`parse_declarators`]).
//!
//! # What this parser does and does not do
//!
//! It is a **declaration extractor**. It parses the declaration surface of a
//! file and *deliberately skips statement bodies*. That is a respectable v1 for
//! symbol extraction and it is stated plainly rather than implied away:
//!
//! **Parsed:** `Namespace`, `Module`, `Class`, `Structure`, VB6 `Type`,
//! `Interface`, `Enum` (and its members), `Sub`, `Function`, `Property` (both
//! the VB.NET `Property`/`Get`/`Set` form and the VB6 `Property Get`/`Let`/`Set`
//! form, including auto-properties, which have no `End Property`), `Operator`,
//! `Event`, `Delegate`, VB6 `Declare` (with `Lib`/`Alias` — the Win32 P/Invoke
//! surface, which is exactly what a translation needs to see), module-level
//! `Dim`/`Const`/fields, `Imports`, `Inherits`, `Implements`, `Option`, and
//! VBA's `Attribute VB_Name`.
//!
//! **Not parsed:** statement bodies, and therefore local variables, control
//! flow, expressions and call graphs. Parameter lists are captured as raw text,
//! not as entities. Conditional compilation (`#If`) is ignored, so *both*
//! branches are parsed. `:` statement separators are not split, so
//! `Sub F() : End Sub` on one line is not understood. Generic type arguments
//! and attributes (`<Serializable>`) are preserved in the signature text but
//! not modelled. `.vbproj`/`.vbp` are treated as project markers; their
//! contents are not read (project-level `Imports` in a `.vbproj` are therefore
//! invisible).
//!
//! Nothing here is best-effort about *safety*: malformed, truncated, binary or
//! non-VB input yields fewer facts, never a panic.
//!
//! # The four things that bite you when parsing Visual Basic
//!
//! Recorded here because each one cost real time, and CLAUDE.md's "preserve
//! hard-won format knowledge in the code" says this belongs next to the code
//! that relies on it.
//!
//! 1. **VB is case-insensitive.** `PUBLIC SUB Foo()` and `public sub Foo()` are
//!    the same declaration. Every keyword comparison in this file goes through
//!    [`eq_kw`], and every cross-reference lookup (base types, interfaces) is
//!    keyed on a lowercased name. Forgetting this is the classic VB bug.
//! 2. **An auto-property is indistinguishable from an expanded property at its
//!    declaration line.** `Public Property Name As String` may or may not be
//!    followed by `Get`/`Set` and an `End Property`. A block-stack parser cannot
//!    decide when it sees the head. Handled by pushing the property as a *soft*
//!    block that any subsequent declaration head implicitly closes, and that a
//!    bare `Get`/`Set` accessor line promotes to a hard block. See [`Block`].
//! 3. **Not every `Sub` has an `End Sub`.** Interface members, `MustOverride`
//!    members, `Declare` and `Delegate` all declare a signature with no body.
//!    Pushing a block for them corrupts the stack for the rest of the file.
//! 4. **`Dim a, b As Integer` means different things in the two dialects.** In
//!    VB.NET both `a` and `b` are `Integer`. In VB6/VBA, `a` is `Variant` and
//!    only `b` is `Integer` — a notorious source of legacy bugs, and a fact a
//!    translation must not lose. [`parse_declarators`] models both.
//!
//! # Mapping onto the shared ontology
//!
//! The point of `kopitiam-ontology` is that a VB `Class` and a Python `class`
//! land as the same [`EntityKind::Symbol`], so the knowledge graph can reason
//! across languages without knowing which one a fact came from.
//!
//! | Visual Basic | Ontology |
//! |---|---|
//! | `.vbproj` / `.vbp` / bare source directory | [`EntityKind::Artifact`] (`kind: "project"`) |
//! | source file | [`EntityKind::Artifact`] (`kind: "source"`) |
//! | `Module`, `Class`, `Structure`, `Type`, `Interface`, `Enum`, `Sub`, `Function`, `Property`, `Event`, `Delegate`, `Operator`, `Declare`, module-level `Dim`/`Const` | [`EntityKind::Symbol`] |
//! | source file in project | [`RelationshipKind::LocatedIn`] |
//! | symbol in file | [`RelationshipKind::LocatedIn`] |
//! | member in containing type | [`RelationshipKind::LocatedIn`] |
//! | `Imports Ns` | file [`RelationshipKind::DependsOn`] namespace symbol |
//! | `Implements IFoo` | `IFoo` [`RelationshipKind::ImplementedBy`] the class |
//! | `Inherits Base` | derived [`RelationshipKind::Custom`]`("inherits")` base |
//!
//! `Implements` reads "interface -> implementor", matching the direction the
//! ontology documents for `ImplementedBy` (`Function -implemented_by-> Rust
//! Module`). `Inherits` has no first-class variant; `Custom("inherits")` is
//! used rather than flattening it to `DependsOn`, which would lose the
//! distinction between "uses" and "is a". A first-class `Inherits`/`Extends`
//! variant is the one thing this provider would ask the ontology for.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use kopitiam_ontology::{Entity, EntityId, EntityKind, Relationship, RelationshipKind};
use serde_json::json;

use crate::provider::{KnowledgeProvider, ProviderOutput};

/// Provenance recorded as `Entity::source` on every fact this provider emits,
/// per CLAUDE.md's Scientific Standards.
pub const PROVIDER_NAME: &str = "visual-basic";

/// Directories never worth walking: build output, VCS metadata, and the
/// gitignored `vendor/` tree CLAUDE.md tells us to leave alone.
const SKIP_DIRS: &[&str] = &[
    ".git", ".svn", ".hg", ".vs", "bin", "obj", "target", "node_modules", "packages", "vendor",
];

/// Guards against pathological directory nesting. Symlinked directories are
/// skipped naturally: [`fs::DirEntry::file_type`] does not follow links, so a
/// symlink never reports `is_dir()`, and cycles are impossible.
const MAX_DEPTH: usize = 64;

// ---------------------------------------------------------------------------
// Dialects
// ---------------------------------------------------------------------------

/// Which Visual Basic we are looking at.
///
/// The split is by file extension, which is reliable in practice: `.vb` is
/// VB.NET and nothing else, and `.bas`/`.cls`/`.frm` are VBA/VB6 and nothing
/// else. The two families are *not* interchangeable — see the `Dim a, b As
/// Integer` divergence in [`parse_declarators`] and the implicit-container rule
/// in [`parse_file`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    /// Modern VB.NET (`.vb`, `.vbproj`).
    VbNet,
    /// VBA and VB6 / classic VB (`.bas`, `.cls`, `.frm`, `.vbp`). Treated as
    /// one dialect because their declaration grammars are the same language;
    /// VBA is, historically, VB6's engine embedded in Office.
    Classic,
}

impl Dialect {
    fn as_str(self) -> &'static str {
        match self {
            Dialect::VbNet => "vb.net",
            Dialect::Classic => "vb-classic",
        }
    }

    /// The dialect implied by a source-file extension, or `None` if the
    /// extension is not one we claim to parse. `.vbs` (VBScript) is
    /// deliberately absent: it is a related but distinct dialect and this
    /// provider does not claim it.
    fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_ascii_lowercase().as_str() {
            "vb" => Some(Dialect::VbNet),
            "bas" | "cls" | "frm" => Some(Dialect::Classic),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Declaration kinds
// ---------------------------------------------------------------------------

/// The kind of a Visual Basic declaration.
///
/// All of these become an [`EntityKind::Symbol`]; this enum survives only in
/// the entity's `metadata.kind`, so a consumer that *does* care about the
/// difference between a `Sub` and a `Function` can still recover it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeclKind {
    Namespace,
    Module,
    Class,
    /// `Structure` (VB.NET) and `Type` (VB6 user-defined type).
    Structure,
    Interface,
    Enum,
    EnumMember,
    Sub,
    Function,
    Property,
    Operator,
    Event,
    Delegate,
    /// VB6/VBA `Declare Sub`/`Declare Function` — a Win32 P/Invoke import.
    Declare,
    /// A module- or type-level `Dim`/`WithEvents`/implicit field.
    Field,
    /// A module- or type-level `Const`.
    Const,
}

impl DeclKind {
    fn as_str(self) -> &'static str {
        match self {
            DeclKind::Namespace => "Namespace",
            DeclKind::Module => "Module",
            DeclKind::Class => "Class",
            DeclKind::Structure => "Structure",
            DeclKind::Interface => "Interface",
            DeclKind::Enum => "Enum",
            DeclKind::EnumMember => "EnumMember",
            DeclKind::Sub => "Sub",
            DeclKind::Function => "Function",
            DeclKind::Property => "Property",
            DeclKind::Operator => "Operator",
            DeclKind::Event => "Event",
            DeclKind::Delegate => "Delegate",
            DeclKind::Declare => "Declare",
            DeclKind::Field => "Field",
            DeclKind::Const => "Const",
        }
    }

    /// True for declarations that introduce a *type* other code can name — the
    /// set that base-class and interface references are resolved against.
    fn is_type(self) -> bool {
        matches!(
            self,
            DeclKind::Module
                | DeclKind::Class
                | DeclKind::Structure
                | DeclKind::Interface
                | DeclKind::Enum
        )
    }

    /// True for declarations whose body is executable code, which this parser
    /// skips wholesale. Everything between such a head and its `End` is
    /// ignored, which is *why* a local `Dim` inside a `Sub` never becomes a
    /// symbol while a module-level one does.
    fn is_executable(self) -> bool {
        matches!(
            self,
            DeclKind::Sub | DeclKind::Function | DeclKind::Property | DeclKind::Operator | DeclKind::Event
        )
    }

    /// The `End <keyword>` that terminates this declaration, if any.
    fn end_keyword(word: &str) -> Option<Self> {
        let k = match () {
            () if eq_kw(word, "sub") => DeclKind::Sub,
            () if eq_kw(word, "function") => DeclKind::Function,
            () if eq_kw(word, "property") => DeclKind::Property,
            () if eq_kw(word, "operator") => DeclKind::Operator,
            () if eq_kw(word, "event") => DeclKind::Event,
            () if eq_kw(word, "class") => DeclKind::Class,
            () if eq_kw(word, "module") => DeclKind::Module,
            // `End Structure` (VB.NET) and `End Type` (VB6) close the same thing.
            () if eq_kw(word, "structure") || eq_kw(word, "type") => DeclKind::Structure,
            () if eq_kw(word, "interface") => DeclKind::Interface,
            () if eq_kw(word, "enum") => DeclKind::Enum,
            () if eq_kw(word, "namespace") => DeclKind::Namespace,
            () => return None,
        };
        Some(k)
    }
}

/// Case-insensitive keyword comparison. `expected` must be lowercase ASCII.
///
/// This exists as a named function rather than an inline `eq_ignore_ascii_case`
/// so that the case-insensitivity of the language is impossible to forget: if
/// you are comparing a VB token to a keyword and you are not calling this, you
/// have a bug.
fn eq_kw(word: &str, expected: &str) -> bool {
    word.eq_ignore_ascii_case(expected)
}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Token {
    /// An identifier or keyword. Bracket-escaped identifiers (`[Class]`, legal
    /// in VB.NET when a name collides with a keyword) arrive here unbracketed.
    Word(String),
    /// A string literal, contents only. Crucially, an apostrophe *inside* a
    /// string is not a comment, and `""` is an escaped quote — both handled in
    /// [`lex_line`].
    Str(String),
    Number(String),
    /// A VB6 type character suffix: `Dim s$` declares a `String`. Emitted only
    /// when it immediately abuts an identifier.
    TypeChar(char),
    Punct(char),
}

impl Token {
    fn word(&self) -> Option<&str> {
        match self {
            Token::Word(w) => Some(w),
            _ => None,
        }
    }

    fn is_kw(&self, expected: &str) -> bool {
        self.word().is_some_and(|w| eq_kw(w, expected))
    }
}

struct LexedLine {
    tokens: Vec<Token>,
    /// The line's source with any comment removed — kept verbatim so a
    /// declaration's `signature` metadata is the text a human actually wrote.
    code: String,
    /// True if this physical line ends with an explicit `_` line continuation.
    continued: bool,
}

/// Lexes one *physical* line, stopping at a comment.
///
/// Comment handling has to be done here, not with a pre-pass, because whether a
/// `'` starts a comment depends on whether we are inside a string literal —
/// and `REM` starts a comment only at the beginning of a statement, so `Remove`
/// and `x.Rem` must not trigger it.
///
/// One VB fact makes this tractable: **a VB string literal can never span
/// lines.** There is no multi-line string, no heredoc, no triple quote. So
/// string state resets at every newline and a purely line-local lexer is
/// correct — which is not true of, say, C++ or Python.
fn lex_line(line: &str) -> LexedLine {
    let chars: Vec<char> = line.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0usize;
    let mut code_end = chars.len();
    // `REM` is only a comment at the start of a statement, i.e. at the start of
    // the line or just after a `:` separator.
    let mut at_stmt_start = true;

    while i < chars.len() {
        let c = chars[i];

        if c.is_whitespace() {
            i += 1;
            continue;
        }

        // A comment runs to end of line. We are outside a string here by
        // construction, so this apostrophe is unambiguous.
        if c == '\'' {
            code_end = i;
            break;
        }

        if c == '"' {
            let mut s = String::new();
            i += 1;
            while i < chars.len() {
                if chars[i] == '"' {
                    // `""` is an escaped quote, not the end of the string.
                    if chars.get(i + 1) == Some(&'"') {
                        s.push('"');
                        i += 2;
                        continue;
                    }
                    i += 1;
                    break;
                }
                s.push(chars[i]);
                i += 1;
            }
            tokens.push(Token::Str(s));
            at_stmt_start = false;
            continue;
        }

        // Bracket-escaped identifier: `[Class]`, `[Error]`.
        if c == '[' {
            let start = i + 1;
            let mut j = start;
            while j < chars.len() && chars[j] != ']' {
                j += 1;
            }
            tokens.push(Token::Word(chars[start..j].iter().collect()));
            i = (j + 1).min(chars.len());
            at_stmt_start = false;
            continue;
        }

        if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();

            if at_stmt_start && eq_kw(&word, "rem") {
                code_end = start;
                break;
            }

            tokens.push(Token::Word(word));
            at_stmt_start = false;

            // A VB6 type character binds tightly to the identifier it follows:
            // `Dim s$` is `Dim s As String`. It must abut the name with no
            // space, which is what distinguishes it from the `&` concatenation
            // operator and the `&H` hex-literal prefix.
            if let Some(&tc) = chars.get(i)
                && matches!(tc, '$' | '%' | '&' | '!' | '#' | '@')
            {
                tokens.push(Token::TypeChar(tc));
                i += 1;
            }
            continue;
        }

        if c.is_ascii_digit() {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '.') {
                i += 1;
            }
            tokens.push(Token::Number(chars[start..i].iter().collect()));
            at_stmt_start = false;
            continue;
        }

        tokens.push(Token::Punct(c));
        at_stmt_start = c == ':';
        i += 1;
    }

    // A lone `_` as the final token is the explicit line-continuation marker.
    // It cannot be a real identifier: VB requires an identifier to have at
    // least one character after a leading underscore.
    let continued = matches!(tokens.last(), Some(Token::Word(w)) if w == "_");
    if continued {
        tokens.pop();
    }

    let mut code: String = chars[..code_end].iter().collect();
    code = code.trim().to_string();
    if continued {
        code = code.trim_end().trim_end_matches('_').trim_end().to_string();
    }

    LexedLine {
        tokens,
        code,
        continued,
    }
}

/// One *logical* line: physical lines joined across `_` continuations.
struct LogicalLine {
    /// 1-based physical line number where the logical line *starts*. This is
    /// what a human needs to jump to, so it is what goes in `metadata.line`.
    line: usize,
    tokens: Vec<Token>,
    /// Comment-stripped source, continuations joined with a single space.
    text: String,
}

/// Folds physical lines into logical lines, resolving explicit `_`
/// continuations.
///
/// Only *explicit* continuations are handled. VB.NET 2010 added implicit line
/// continuation (a line may end after `,`, `(`, `As`, an operator, ... and just
/// keep going), which requires a real expression parser to detect. That is not
/// covered, and a declaration split with implicit continuation will be parsed
/// as far as its first physical line goes — degrading to a partial fact, never
/// to a panic.
fn logical_lines(src: &str) -> Vec<LogicalLine> {
    let mut out = Vec::new();
    let mut pending: Option<LogicalLine> = None;

    for (idx, physical) in src.lines().enumerate() {
        let lexed = lex_line(physical);

        match &mut pending {
            Some(acc) => {
                acc.tokens.extend(lexed.tokens);
                if !lexed.code.is_empty() {
                    if !acc.text.is_empty() {
                        acc.text.push(' ');
                    }
                    acc.text.push_str(&lexed.code);
                }
            }
            None => {
                pending = Some(LogicalLine {
                    line: idx + 1,
                    tokens: lexed.tokens,
                    text: lexed.code,
                });
            }
        }

        if !lexed.continued && let Some(done) = pending.take() {
            out.push(done);
        }
    }

    // A file ending on a continuation is malformed; keep what we have.
    if let Some(done) = pending.take() {
        out.push(done);
    }

    out
}

// ---------------------------------------------------------------------------
// Parsed representation
// ---------------------------------------------------------------------------

/// One declaration, flat, with `parent` indexing back into the same file's
/// `decls` vector. A flat vector rather than a tree because parents are always
/// parsed before their children (source order), so a plain index is enough and
/// nothing needs `Rc`.
#[derive(Debug, Clone)]
struct Decl {
    name: String,
    kind: DeclKind,
    line: usize,
    /// `Public` / `Private` / `Protected` / `Friend` / `ProtectedFriend` /
    /// `PrivateProtected`, or `None` when the source did not say (VB's implicit
    /// default differs by context, and guessing it would be inventing a fact).
    accessibility: Option<String>,
    modifiers: Vec<String>,
    /// The declaration line as written, comments stripped. Preserved because
    /// the Translation Platform wants the original text, not just our model of
    /// it.
    signature: String,
    /// Declared type: the `As T` clause, a VB6 type character, or `None`.
    ty: Option<String>,
    /// VB6/VBA property accessor: `Get`, `Let` or `Set`.
    accessor: Option<String>,
    /// Raw parameter list text, if any. Parameters are not modelled as entities.
    params: Option<String>,
    /// A generic method's type-parameter list (`Of T`), kept separate from the
    /// argument list it is syntactically indistinguishable from.
    type_params: Option<String>,
    /// A member's trailing `Implements IFoo.Bar` clause.
    implements_member: Option<String>,
    /// `Lib "kernel32"` / `Alias "GetTickCount"` on a VB6 `Declare`.
    lib: Option<String>,
    alias: Option<String>,
    parent: Option<usize>,
    /// Base types from an `Inherits` statement (an `Interface` may list several).
    inherits: Vec<String>,
    /// Interfaces from a type-level `Implements` statement.
    implements: Vec<String>,
}

impl Decl {
    fn new(name: String, kind: DeclKind, line: usize, signature: String) -> Self {
        Self {
            name,
            kind,
            line,
            accessibility: None,
            modifiers: Vec::new(),
            signature,
            ty: None,
            accessor: None,
            params: None,
            type_params: None,
            implements_member: None,
            lib: None,
            alias: None,
            parent: None,
            inherits: Vec::new(),
            implements: Vec::new(),
        }
    }
}

/// Everything one source file contributes.
#[derive(Debug)]
struct ParsedFile {
    path: PathBuf,
    dialect: Dialect,
    /// `Option Explicit On`, `Option Strict On`, ... — recorded because
    /// `Option Strict Off` is a material fact about how much the compiler was
    /// checking, and a translation needs to know it.
    options: Vec<String>,
    /// `Imports` targets, in source order, deduplicated.
    imports: Vec<String>,
    decls: Vec<Decl>,
}

/// An open block on the parser's stack.
struct Block {
    index: usize,
    kind: DeclKind,
    /// A *soft* block is one we are not yet sure exists.
    ///
    /// Only properties are ever soft, and this is the single subtlest thing in
    /// the parser. `Public Property Name As String` is a complete auto-property
    /// with no `End Property` — *and* it is also the exact first line of an
    /// expanded property that does have one. The two are indistinguishable at
    /// the head, so we push optimistically and resolve later: a bare `Get`/`Set`
    /// accessor line promotes the block to hard, and any following declaration
    /// head (or an enclosing `End`) implicitly closes it.
    soft: bool,
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Modifiers that may precede a declaration keyword.
///
/// `Dim`, `Const` and `Declare` are deliberately *absent*: they are declaration
/// heads in their own right, not modifiers. `WithEvents` and `Static` are here
/// because they precede a bare name (`Private WithEvents Btn As Button`), which
/// the "at least one modifier plus a bare name" rule then recognises as a field.
const MODIFIERS: &[&str] = &[
    "public",
    "private",
    "protected",
    "friend",
    "shared",
    "shadows",
    "overrides",
    "overridable",
    "mustoverride",
    "notoverridable",
    "overloads",
    "partial",
    "mustinherit",
    "notinheritable",
    "default",
    "readonly",
    "writeonly",
    "static",
    "async",
    "iterator",
    "widening",
    "narrowing",
    "custom",
    "withevents",
    "global",
];

fn is_modifier(word: &str) -> bool {
    MODIFIERS.iter().any(|m| eq_kw(word, m))
}

/// Reduces a modifier list to a single accessibility, handling VB's two
/// two-word accessibilities (`Protected Friend`, `Private Protected`).
fn accessibility_of(modifiers: &[String]) -> Option<String> {
    let has = |m: &str| modifiers.iter().any(|x| eq_kw(x, m));
    if has("protected") && has("friend") {
        return Some("ProtectedFriend".to_string());
    }
    if has("private") && has("protected") {
        return Some("PrivateProtected".to_string());
    }
    for (kw, name) in [
        ("public", "Public"),
        ("private", "Private"),
        ("protected", "Protected"),
        ("friend", "Friend"),
    ] {
        if has(kw) {
            return Some(name.to_string());
        }
    }
    None
}

/// The type a VB6 type character stands for.
///
/// `@` is `Currency` in VB6/VBA (a fixed-point scaled integer, *not* a float —
/// a distinction a financial translation must preserve) and `Decimal` in
/// VB.NET. Both are reported so neither dialect is misrepresented.
fn type_char_meaning(c: char, dialect: Dialect) -> Option<&'static str> {
    Some(match c {
        '$' => "String",
        '%' => "Integer",
        '&' => "Long",
        '!' => "Single",
        '#' => "Double",
        '@' => match dialect {
            Dialect::VbNet => "Decimal",
            Dialect::Classic => "Currency",
        },
        _ => return None,
    })
}

/// Renders tokens back to readable source, used for type expressions such as
/// `List(Of String)` or `Dictionary(Of String, Integer)`.
fn render(tokens: &[Token]) -> String {
    let mut out = String::new();
    for (i, tok) in tokens.iter().enumerate() {
        let text = match tok {
            Token::Word(w) => w.clone(),
            Token::Number(n) => n.clone(),
            Token::Str(s) => format!("\"{s}\""),
            Token::TypeChar(c) | Token::Punct(c) => c.to_string(),
        };
        let no_space_before = matches!(tok, Token::Punct('.' | ',' | ')' | '(' | ']'))
            || matches!(tok, Token::TypeChar(_));
        let prev_no_space_after = matches!(
            tokens.get(i.wrapping_sub(1)),
            Some(Token::Punct('.' | '(' | '['))
        );
        if i > 0 && !no_space_before && !prev_no_space_after {
            out.push(' ');
        }
        out.push_str(&text);
    }
    out
}

/// Reads a dotted name (`System.Collections.Generic`) starting at `*i`.
fn read_dotted_name(tokens: &[Token], i: &mut usize) -> Option<String> {
    let mut name = tokens.get(*i)?.word()?.to_string();
    *i += 1;
    while tokens.get(*i) == Some(&Token::Punct('.')) {
        let Some(next) = tokens.get(*i + 1).and_then(Token::word) else {
            break;
        };
        name.push('.');
        name.push_str(next);
        *i += 2;
    }
    Some(name)
}

/// Reads a balanced parenthesised group starting at `*i` (which must be on the
/// `(`), returning its rendered contents and leaving `*i` past the `)`.
///
/// Balanced, not first-`)`, because parameter defaults and generic arguments
/// nest: `(ByVal items As List(Of String))`.
fn read_paren_group(tokens: &[Token], i: &mut usize) -> Option<String> {
    if tokens.get(*i) != Some(&Token::Punct('(')) {
        return None;
    }
    let start = *i + 1;
    let mut depth = 0usize;
    let mut j = *i;
    while j < tokens.len() {
        match tokens[j] {
            Token::Punct('(') => depth += 1,
            Token::Punct(')') => {
                depth -= 1;
                if depth == 0 {
                    *i = j + 1;
                    return Some(render(&tokens[start..j]));
                }
            }
            _ => {}
        }
        j += 1;
    }
    // Unbalanced: consume the rest rather than looping forever.
    *i = tokens.len();
    Some(render(&tokens[start..]))
}

/// Splits `tokens` on top-level commas (ignoring commas nested in parentheses).
fn split_top_level_commas(tokens: &[Token]) -> Vec<&[Token]> {
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (i, tok) in tokens.iter().enumerate() {
        match tok {
            Token::Punct('(') => depth += 1,
            Token::Punct(')') => depth = depth.saturating_sub(1),
            Token::Punct(',') if depth == 0 => {
                out.push(&tokens[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&tokens[start..]);
    out
}

/// Parses a field/const declarator list into `(name, type)` pairs.
///
/// **This is where the two dialects genuinely disagree**, and getting it wrong
/// would silently misreport types across an entire legacy codebase:
///
/// ```text
/// Dim a, b As Integer
///   VB.NET :  a As Integer, b As Integer
///   VB6/VBA:  a As Variant, b As Integer      ' <- a is NOT an Integer
/// ```
///
/// VB6 applies `As T` only to the name it directly follows; every other name in
/// the list is `Variant`. VB.NET back-propagates the type to all preceding
/// untyped names in the run. This is one of the most notorious footguns in
/// classic VB and a translation that flattens it will produce wrong code, so it
/// is modelled rather than approximated.
fn parse_declarators(tokens: &[Token], dialect: Dialect) -> Vec<(String, Option<String>)> {
    let mut out: Vec<(String, Option<String>)> = Vec::new();
    // Names seen since the last `As`, still waiting to learn their type.
    let mut pending: Vec<String> = Vec::new();

    for segment in split_top_level_commas(tokens) {
        let mut i = 0usize;
        let Some(name) = segment.get(i).and_then(Token::word).map(str::to_string) else {
            continue;
        };
        i += 1;

        // A type character binds to this name alone: `Dim s$, i%`.
        let mut ty = match segment.get(i) {
            Some(Token::TypeChar(c)) => {
                i += 1;
                type_char_meaning(*c, dialect).map(str::to_string)
            }
            _ => None,
        };

        // Array bounds: `Dim buf(1023) As Byte`, `Dim xs() As Integer`.
        let mut is_array = false;
        if segment.get(i) == Some(&Token::Punct('(')) {
            is_array = true;
            let _ = read_paren_group(segment, &mut i);
        }

        // `As T`, stopping before any `= initializer`.
        if ty.is_none()
            && let Some(tok) = segment.get(i)
            && tok.is_kw("as")
        {
            let start = i + 1;
            // The type ends at an initializer (`= 1.0`) or at a field's trailing
            // `Implements IFoo.Bar` clause — neither is part of the type.
            let end = segment[start..]
                .iter()
                .position(|t| *t == Token::Punct('=') || t.is_kw("implements"))
                .map_or(segment.len(), |p| start + p);
            let mut rendered = render(&segment[start..end]);
            // `Dim x As New Foo()` declares type `Foo`.
            if let Some(rest) = rendered.strip_prefix("New ") {
                rendered = rest.trim().trim_end_matches("()").trim().to_string();
            }
            if !rendered.is_empty() {
                ty = Some(rendered);
            }
        }

        if is_array && let Some(t) = ty.take() {
            ty = Some(format!("{t}()"));
        }

        match ty {
            Some(t) => {
                match dialect {
                    // VB.NET: the type reaches back over every untyped name.
                    Dialect::VbNet => {
                        for p in pending.drain(..) {
                            out.push((p, Some(t.clone())));
                        }
                    }
                    // VB6/VBA: it does not. Those names are Variant.
                    Dialect::Classic => {
                        for p in pending.drain(..) {
                            out.push((p, Some("Variant".to_string())));
                        }
                    }
                }
                out.push((name, Some(t)));
            }
            None => pending.push(name),
        }
    }

    // Trailing untyped names: `Variant` in classic VB, genuinely unstated in
    // VB.NET (it may be `Option Infer`red), so we do not invent a type.
    for p in pending {
        let ty = match dialect {
            Dialect::VbNet => None,
            Dialect::Classic => Some("Variant".to_string()),
        };
        out.push((p, ty));
    }

    out
}

/// Parses one source file into declarations.
fn parse_file(path: &Path, dialect: Dialect, src: &str) -> ParsedFile {
    let lines = logical_lines(src);

    let mut file = ParsedFile {
        path: path.to_path_buf(),
        dialect,
        options: Vec::new(),
        imports: Vec::new(),
        decls: Vec::new(),
    };
    let mut stack: Vec<Block> = Vec::new();

    // VBA/VB6 module and class files declare their container *implicitly*: a
    // `.bas` file is a Module and a `.cls`/`.frm` file is a Class, named by the
    // `Attribute VB_Name = "..."` line the VBE writes at the top (falling back
    // to the file stem for hand-written files that lack it). Without this, every
    // member of every VBA file would appear at file scope with no owner, which
    // is not what the source means.
    if dialect == Dialect::Classic && !declares_container(&lines) {
        let name = vb_name_attribute(&lines)
            .or_else(|| path.file_stem().map(|s| s.to_string_lossy().into_owned()))
            .unwrap_or_else(|| "Module1".to_string());
        let kind = match path.extension().and_then(|e| e.to_str()) {
            Some(e) if e.eq_ignore_ascii_case("bas") => DeclKind::Module,
            // `.cls` is a class; `.frm` is a Form, which is also a class.
            _ => DeclKind::Class,
        };
        let mut decl = Decl::new(name, kind, 1, String::new());
        decl.modifiers.push("Implicit".to_string());
        file.decls.push(decl);
        stack.push(Block {
            index: 0,
            kind,
            soft: false,
        });
    }

    for line in &lines {
        parse_logical_line(line, dialect, &mut file, &mut stack);
    }

    file
}

/// True if the file explicitly declares a top-level container, in which case no
/// implicit VBA container is synthesised.
fn declares_container(lines: &[LogicalLine]) -> bool {
    lines.iter().any(|l| {
        let head = l.tokens.iter().find(|t| match t {
            Token::Word(w) => !is_modifier(w),
            _ => true,
        });
        head.is_some_and(|t| {
            t.is_kw("module") || t.is_kw("class") || t.is_kw("structure") || t.is_kw("interface")
        })
    })
}

/// Extracts VBA's `Attribute VB_Name = "Sheet1"` header, if present.
fn vb_name_attribute(lines: &[LogicalLine]) -> Option<String> {
    lines.iter().find_map(|l| {
        let t = &l.tokens;
        if t.first()?.is_kw("attribute") && t.get(1)?.is_kw("vb_name") {
            match t.get(3) {
                Some(Token::Str(s)) if !s.is_empty() => Some(s.clone()),
                _ => None,
            }
        } else {
            None
        }
    })
}

/// The core dispatch: one logical line against the current block stack.
///
/// Order matters here and is not arbitrary — see the inline notes.
fn parse_logical_line(line: &LogicalLine, dialect: Dialect, file: &mut ParsedFile, stack: &mut Vec<Block>) {
    let t = &line.tokens;
    let Some(first) = t.first() else {
        return;
    };

    // Conditional compilation and regions. We parse *both* branches of an
    // `#If`, which can in principle surface a symbol that would not compile for
    // a given set of constants. That over-reporting is the honest trade: a
    // translation wants to see all the code, not just one configuration's.
    if *first == Token::Punct('#') {
        return;
    }

    // Must come before the executable-body skip: a bare `Get`/`Set` line is what
    // promotes a soft (auto-)property to a real, expanded one.
    if let Some(top) = stack.last_mut()
        && top.kind == DeclKind::Property
        && top.soft
        && (first.is_kw("get") || first.is_kw("set"))
        && t.get(1).is_none_or(|n| *n == Token::Punct('('))
    {
        top.soft = false;
        return;
    }

    // Inside an executable body nothing is a declaration, so we look only for
    // the terminator. This is precisely why a local `Dim` never becomes a
    // symbol: it is unreachable from here.
    if stack.last().is_some_and(|b| b.kind.is_executable() && !b.soft) {
        if first.is_kw("end")
            && let Some(kind) = t.get(1).and_then(Token::word).and_then(DeclKind::end_keyword)
        {
            close_block(stack, kind);
        }
        return;
    }

    // Enum bodies contain bare member names, not modifier-led declarations.
    if stack.last().is_some_and(|b| b.kind == DeclKind::Enum) {
        if first.is_kw("end") {
            if let Some(kind) = t.get(1).and_then(Token::word).and_then(DeclKind::end_keyword) {
                close_block(stack, kind);
            }
            return;
        }
        if let Some(name) = first.word() {
            let parent = stack.last().map(|b| b.index);
            let mut decl = Decl::new(name.to_string(), DeclKind::EnumMember, line.line, line.text.clone());
            decl.parent = parent;
            file.decls.push(decl);
        }
        return;
    }

    if first.is_kw("end") {
        // A bare `End` is VB6's "terminate the program" statement, not a block
        // terminator. Only `End <keyword>` closes anything.
        if let Some(kind) = t.get(1).and_then(Token::word).and_then(DeclKind::end_keyword) {
            close_block(stack, kind);
        }
        return;
    }

    if first.is_kw("option") {
        let opt = render(&t[1..]);
        if !opt.is_empty() {
            file.options.push(opt);
        }
        return;
    }

    if first.is_kw("imports") {
        let mut i = 1usize;
        // `Imports Alias = System.Collections.Generic` — the dependency is the
        // right-hand side; the alias is a local convenience.
        if t.get(2) == Some(&Token::Punct('=')) {
            i = 3;
        }
        if let Some(name) = read_dotted_name(t, &mut i)
            && !file.imports.iter().any(|x| eq_kw(x, &name))
        {
            file.imports.push(name);
        }
        return;
    }

    // `Attribute VB_Name = "..."` was consumed during the pre-scan.
    if first.is_kw("attribute") {
        return;
    }

    if first.is_kw("inherits") {
        if let Some(block) = stack.last()
            && let Some(decl) = file.decls.get_mut(block.index)
        {
            // An `Interface` may inherit several interfaces at once:
            // `Inherits IReadable, IWritable`.
            for seg in split_top_level_commas(&t[1..]) {
                let mut i = 0usize;
                if let Some(name) = read_dotted_name(seg, &mut i) {
                    decl.inherits.push(name);
                }
            }
        }
        return;
    }

    // A *statement-level* `Implements`. A member's trailing `Implements IFoo.Bar`
    // clause never reaches here, because it is not the first token of its line.
    if first.is_kw("implements") {
        if let Some(block) = stack.last()
            && let Some(decl) = file.decls.get_mut(block.index)
        {
            for seg in split_top_level_commas(&t[1..]) {
                let mut i = 0usize;
                if let Some(name) = read_dotted_name(seg, &mut i) {
                    decl.implements.push(name);
                }
            }
        }
        return;
    }

    parse_declaration(line, dialect, file, stack);
}

/// Closes the innermost open block of `kind`.
///
/// Searching from the top rather than only inspecting the top does two jobs at
/// once: it discards any soft (auto-)property sitting above the block being
/// closed, and it recovers from malformed source (a missing `End Sub` before an
/// `End Class`) without unwinding the entire stack. A stray `End X` with no
/// matching open block is ignored — the alternative, unwinding everything, would
/// let one typo destroy the rest of the file.
fn close_block(stack: &mut Vec<Block>, kind: DeclKind) {
    if let Some(pos) = stack.iter().rposition(|b| b.kind == kind) {
        stack.truncate(pos);
    }
}

/// Parses a modifier-led declaration line.
fn parse_declaration(line: &LogicalLine, dialect: Dialect, file: &mut ParsedFile, stack: &mut Vec<Block>) {
    let t = &line.tokens;

    let mut i = 0usize;
    let mut modifiers: Vec<String> = Vec::new();
    while let Some(w) = t.get(i).and_then(Token::word) {
        if !is_modifier(w) {
            break;
        }
        modifiers.push(w.to_string());
        i += 1;
    }

    let Some(head) = t.get(i) else {
        return;
    };
    let Some(head_word) = head.word() else {
        return;
    };

    // Any declaration head implicitly closes a pending auto-property, since an
    // auto-property has no body for this line to be inside of.
    if stack.last().is_some_and(|b| b.soft) {
        stack.pop();
    }

    let parent = stack.last().map(|b| b.index);
    let accessibility = accessibility_of(&modifiers);
    let in_interface = stack.last().is_some_and(|b| b.kind == DeclKind::Interface);
    let must_override = modifiers.iter().any(|m| eq_kw(m, "mustoverride"));

    let finish = |mut decl: Decl, push: Option<DeclKind>, file: &mut ParsedFile, stack: &mut Vec<Block>| {
        decl.parent = parent;
        decl.accessibility = accessibility.clone();
        decl.modifiers = modifiers.clone();
        let index = file.decls.len();
        file.decls.push(decl);
        if let Some(kind) = push {
            stack.push(Block {
                index,
                kind,
                soft: false,
            });
        }
    };

    // --- Containers ---------------------------------------------------------
    if let Some(kind) = container_kind(head_word) {
        i += 1;
        let Some(name) = read_dotted_name(t, &mut i) else {
            return;
        };
        let decl = Decl::new(name, kind, line.line, line.text.clone());
        finish(decl, Some(kind), file, stack);
        return;
    }

    // --- Sub / Function / Operator -----------------------------------------
    if eq_kw(head_word, "sub") || eq_kw(head_word, "function") || eq_kw(head_word, "operator") {
        let kind = if eq_kw(head_word, "sub") {
            DeclKind::Sub
        } else if eq_kw(head_word, "function") {
            DeclKind::Function
        } else {
            DeclKind::Operator
        };
        i += 1;
        let Some(name) = t.get(i).and_then(Token::word).map(str::to_string) else {
            return;
        };
        i += 1;

        let mut decl = Decl::new(name, kind, line.line, line.text.clone());
        // A generic method's type-parameter list uses the same `(` syntax as its
        // argument list, and comes first: `Sub Swap(Of T)(ByRef a As T, ByRef b As T)`.
        // So the first group is the parameters *unless* it opens with `Of`, in
        // which case the real parameters are the group after it.
        let mut group = read_paren_group(t, &mut i);
        if group.as_deref().is_some_and(|g| {
            g.split_whitespace().next().is_some_and(|w| eq_kw(w, "of"))
        }) {
            decl.type_params = group;
            group = read_paren_group(t, &mut i);
        }
        decl.params = group;
        decl.ty = read_as_clause(t, &mut i);
        decl.implements_member = read_member_implements(t, &mut i);

        // Interface members, `MustOverride` members and abstract-by-nature
        // declarations have a signature and no body — so they have no `End Sub`,
        // and pushing a block for them would swallow the rest of the file.
        let push = (!in_interface && !must_override).then_some(kind);
        finish(decl, push, file, stack);
        return;
    }

    // --- Property -----------------------------------------------------------
    if eq_kw(head_word, "property") {
        i += 1;

        // VB6/VBA form: `Property Get Name(...) As T`, with a mandatory
        // `End Property`. VB.NET form: `Property Name As T`, which may or may
        // not have one.
        let accessor = match t.get(i).and_then(Token::word) {
            Some(w) if eq_kw(w, "get") || eq_kw(w, "let") || eq_kw(w, "set") => {
                let a = capitalize(w);
                i += 1;
                Some(a)
            }
            _ => None,
        };
        let Some(name) = t.get(i).and_then(Token::word).map(str::to_string) else {
            return;
        };
        i += 1;

        let mut decl = Decl::new(name, DeclKind::Property, line.line, line.text.clone());
        decl.params = read_paren_group(t, &mut i);
        decl.ty = read_as_clause(t, &mut i);
        decl.implements_member = read_member_implements(t, &mut i);
        let is_classic_accessor = accessor.is_some();
        decl.accessor = accessor;

        decl.parent = parent;
        decl.accessibility = accessibility;
        decl.modifiers = modifiers;
        let index = file.decls.len();
        file.decls.push(decl);

        if !in_interface && !must_override {
            stack.push(Block {
                index,
                kind: DeclKind::Property,
                // The VB6 form always has a body. The VB.NET form is the
                // ambiguous one, so it goes on the stack *soft*.
                soft: !is_classic_accessor,
            });
        }
        return;
    }

    // --- Declare (VB6/VBA Win32 P/Invoke) -----------------------------------
    if eq_kw(head_word, "declare") {
        i += 1;
        // Optional charset marker: `Declare Unicode Function ...`.
        if t.get(i)
            .and_then(Token::word)
            .is_some_and(|w| eq_kw(w, "auto") || eq_kw(w, "ansi") || eq_kw(w, "unicode"))
        {
            i += 1;
        }
        let is_function = t.get(i).is_some_and(|tok| tok.is_kw("function"));
        i += 1;
        let Some(name) = t.get(i).and_then(Token::word).map(str::to_string) else {
            return;
        };
        i += 1;

        let mut decl = Decl::new(name, DeclKind::Declare, line.line, line.text.clone());
        decl.modifiers.push(if is_function { "Function" } else { "Sub" }.to_string());
        if t.get(i).is_some_and(|tok| tok.is_kw("lib")) {
            if let Some(Token::Str(s)) = t.get(i + 1) {
                decl.lib = Some(s.clone());
            }
            i += 2;
        }
        if t.get(i).is_some_and(|tok| tok.is_kw("alias")) {
            if let Some(Token::Str(s)) = t.get(i + 1) {
                decl.alias = Some(s.clone());
            }
            i += 2;
        }
        decl.params = read_paren_group(t, &mut i);
        decl.ty = read_as_clause(t, &mut i);

        // A `Declare` is a one-line declaration: no body, no `End`.
        decl.parent = parent;
        decl.accessibility = accessibility;
        let mods = decl.modifiers.clone();
        decl.modifiers = modifiers;
        decl.modifiers.extend(mods);
        file.decls.push(decl);
        return;
    }

    // --- Delegate -----------------------------------------------------------
    if eq_kw(head_word, "delegate") {
        i += 1;
        // `Delegate Sub Foo(...)` / `Delegate Function Bar(...) As T`.
        if t.get(i)
            .and_then(Token::word)
            .is_some_and(|w| eq_kw(w, "sub") || eq_kw(w, "function"))
        {
            i += 1;
        }
        let Some(name) = t.get(i).and_then(Token::word).map(str::to_string) else {
            return;
        };
        i += 1;
        let mut decl = Decl::new(name, DeclKind::Delegate, line.line, line.text.clone());
        decl.params = read_paren_group(t, &mut i);
        decl.ty = read_as_clause(t, &mut i);
        finish(decl, None, file, stack);
        return;
    }

    // --- Event --------------------------------------------------------------
    if eq_kw(head_word, "event") {
        i += 1;
        let Some(name) = t.get(i).and_then(Token::word).map(str::to_string) else {
            return;
        };
        i += 1;
        let mut decl = Decl::new(name, DeclKind::Event, line.line, line.text.clone());
        decl.params = read_paren_group(t, &mut i);
        decl.ty = read_as_clause(t, &mut i);
        decl.implements_member = read_member_implements(t, &mut i);
        // Only a `Custom Event` has an `End Event`; a plain one is a one-liner.
        let custom = modifiers.iter().any(|m| eq_kw(m, "custom"));
        let push = custom.then_some(DeclKind::Event);
        finish(decl, push, file, stack);
        return;
    }

    // --- Fields and constants ----------------------------------------------
    //
    // Three shapes reach here:
    //   `Dim x As Integer`                    -- explicit Dim
    //   `Public Const K As Integer = 1`       -- Const head after modifiers
    //   `Private Shared _cache As Dictionary` -- modifiers then a bare name
    //   `x As Integer`                        -- a VB6 `Type` member (no modifier)
    let in_structure = stack.last().is_some_and(|b| b.kind == DeclKind::Structure);
    let (kind, decl_start) = if eq_kw(head_word, "const") {
        (DeclKind::Const, i + 1)
    } else if eq_kw(head_word, "dim") {
        (DeclKind::Field, i + 1)
    } else if !modifiers.is_empty() && !is_reserved_statement(head_word) {
        // Modifiers followed by a name: an implicit field. `Public Const` is
        // caught above, so a `Const` modifier cannot reach this branch.
        (DeclKind::Field, i)
    } else if in_structure && t.iter().any(|tok| tok.is_kw("as")) && !is_reserved_statement(head_word) {
        // Inside a VB6 `Type ... End Type`, members carry no modifier at all.
        (DeclKind::Field, i)
    } else {
        // Not a declaration we recognise (a statement at module scope, a `.frm`
        // designer line, a `Begin`/`End` block, ...). Ignoring it is correct;
        // guessing would manufacture facts.
        return;
    };

    // A field is only a symbol at container scope. Reaching here from inside a
    // method is impossible (the executable-body skip returns first), so this is
    // belt-and-braces rather than the main guard.
    if stack.last().is_some_and(|b| b.kind.is_executable()) {
        return;
    }

    let const_kind = kind;
    for (name, ty) in parse_declarators(&t[decl_start..], dialect) {
        let mut decl = Decl::new(name, const_kind, line.line, line.text.clone());
        decl.ty = ty;
        decl.parent = parent;
        decl.accessibility = accessibility.clone();
        decl.modifiers = modifiers.clone();
        file.decls.push(decl);
    }
}

fn container_kind(word: &str) -> Option<DeclKind> {
    let k = match () {
        () if eq_kw(word, "namespace") => DeclKind::Namespace,
        () if eq_kw(word, "module") => DeclKind::Module,
        () if eq_kw(word, "class") => DeclKind::Class,
        // VB6's `Type` is VB.NET's `Structure`; the same ontology symbol.
        () if eq_kw(word, "structure") || eq_kw(word, "type") => DeclKind::Structure,
        () if eq_kw(word, "interface") => DeclKind::Interface,
        () if eq_kw(word, "enum") => DeclKind::Enum,
        () => return None,
    };
    Some(k)
}

/// Words that begin a *statement*, not a declaration. Without this, a
/// module-scope `Call Foo()` or a `.frm` designer's `Begin VB.Form` would be
/// mistaken for a field named `Call` or `Begin`.
fn is_reserved_statement(word: &str) -> bool {
    const STATEMENTS: &[&str] = &[
        "if", "else", "elseif", "select", "case", "for", "each", "next", "while", "wend", "do",
        "loop", "with", "try", "catch", "finally", "using", "synclock", "return", "exit", "call",
        "set", "let", "get", "goto", "gosub", "on", "resume", "erase", "redim", "throw", "stop",
        "begin", "beginproperty", "endproperty", "print", "write", "open", "close", "input",
        "line", "version", "object", "addhandler", "removehandler", "raiseevent", "continue",
        "yield", "await", "error", "lock", "unlock", "put",
    ];
    STATEMENTS.iter().any(|s| eq_kw(word, s))
}

/// Reads an `As T` return/field type at `*i`, stopping at an `Implements`
/// clause or an `=` initializer.
fn read_as_clause(tokens: &[Token], i: &mut usize) -> Option<String> {
    if !tokens.get(*i)?.is_kw("as") {
        return None;
    }
    let start = *i + 1;
    let end = tokens[start..]
        .iter()
        .position(|t| t.is_kw("implements") || *t == Token::Punct('='))
        .map_or(tokens.len(), |p| start + p);
    *i = end;
    let rendered = render(&tokens[start..end]);
    (!rendered.is_empty()).then_some(rendered)
}

/// Reads a member's trailing `Implements IFoo.Bar` clause.
///
/// This is recorded in metadata rather than as a relationship: the target is an
/// interface *member*, and we cannot reliably produce an entity for it (the
/// interface may live outside the scanned tree). Claiming an edge to an entity
/// we did not observe would be inventing a fact.
fn read_member_implements(tokens: &[Token], i: &mut usize) -> Option<String> {
    while *i < tokens.len() && !tokens[*i].is_kw("implements") {
        *i += 1;
    }
    if *i >= tokens.len() {
        return None;
    }
    *i += 1;
    let rendered = render(&tokens[*i..]);
    (!rendered.is_empty()).then_some(rendered)
}

fn capitalize(word: &str) -> String {
    let mut chars = word.chars();
    match chars.next() {
        Some(c) => c.to_ascii_uppercase().to_string() + &chars.as_str().to_ascii_lowercase(),
        None => String::new(),
    }
}

// ---------------------------------------------------------------------------
// File discovery
// ---------------------------------------------------------------------------

/// Reads a source file, coping with the encodings real Visual Basic sources are
/// found in.
///
/// This matters more than it would for a modern language. VB6 and VBA files
/// predate UTF-8's dominance: they are commonly Windows-1252, and the VBE
/// happily writes UTF-16 with a BOM. `fs::read_to_string` would simply fail on
/// both, which for a *legacy translation* tool is the worst possible outcome —
/// the older and more valuable the codebase, the likelier it is to be rejected.
///
/// So: honour a UTF-8 or UTF-16 BOM if present, then try UTF-8, and finally
/// fall back to Latin-1 (`byte as char`), which never fails and is correct for
/// the ASCII identifiers and keywords we actually parse. Only string literals
/// and comments containing accented characters can be mistranscoded, and only
/// into the wrong glyph, never into a parse failure.
fn read_source(path: &Path) -> Result<String> {
    let bytes = fs::read(path)?;

    if let Some(rest) = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]) {
        return Ok(String::from_utf8_lossy(rest).into_owned());
    }
    if let Some(rest) = bytes.strip_prefix(&[0xFF, 0xFE]) {
        return Ok(decode_utf16(rest, true));
    }
    if let Some(rest) = bytes.strip_prefix(&[0xFE, 0xFF]) {
        return Ok(decode_utf16(rest, false));
    }

    match String::from_utf8(bytes) {
        Ok(text) => Ok(text),
        Err(err) => Ok(err.into_bytes().iter().map(|&b| b as char).collect()),
    }
}

fn decode_utf16(bytes: &[u8], little_endian: bool) -> String {
    let units: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|c| {
            if little_endian {
                u16::from_le_bytes([c[0], c[1]])
            } else {
                u16::from_be_bytes([c[0], c[1]])
            }
        })
        .collect();
    String::from_utf16_lossy(&units)
}

/// Recursively collects VB sources and project files under `dir`.
///
/// Entries are sorted before recursion so that the emitted facts are in a
/// deterministic order regardless of filesystem iteration order — CLAUDE.md
/// requires deterministic behaviour, and an unstable entity ordering would make
/// every downstream snapshot unstable too.
fn walk(dir: &Path, depth: usize, sources: &mut Vec<(PathBuf, Dialect)>, projects: &mut Vec<PathBuf>) {
    if depth > MAX_DEPTH {
        tracing::warn!(dir = %dir.display(), "visual-basic: max directory depth reached; not descending");
        return;
    }

    let Ok(entries) = fs::read_dir(dir) else {
        tracing::warn!(dir = %dir.display(), "visual-basic: unreadable directory; skipping");
        return;
    };

    let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
    paths.sort();

    for path in paths {
        let Ok(meta) = fs::symlink_metadata(&path) else {
            continue;
        };

        if meta.is_dir() {
            let skip = path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| SKIP_DIRS.iter().any(|s| n.eq_ignore_ascii_case(s)));
            if !skip {
                walk(&path, depth + 1, sources, projects);
            }
            continue;
        }

        if !meta.is_file() {
            continue; // symlink, socket, fifo: not source
        }

        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        if ext.eq_ignore_ascii_case("vbproj") || ext.eq_ignore_ascii_case("vbp") {
            projects.push(path);
        } else if let Some(dialect) = Dialect::from_extension(ext) {
            sources.push((path, dialect));
        }
    }
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// Knowledge provider for Visual Basic: VB.NET, VBA and VB6 / classic VB.
///
/// Unlike [`crate::RustAnalyzerProvider`] this drives no external process. It is
/// the tool. See this module's documentation for why (short version: no
/// obtainable VB language server exists) and for exactly what it does and does
/// not parse.
///
/// It has no configuration and no environment dependency, so it never degrades
/// for lack of tooling — only for lack of Visual Basic. A directory with no VB
/// sources yields [`ProviderOutput::empty`].
pub struct VisualBasicProvider;

impl VisualBasicProvider {
    pub fn new() -> Self {
        Self
    }
}

impl Default for VisualBasicProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl KnowledgeProvider for VisualBasicProvider {
    fn name(&self) -> &str {
        PROVIDER_NAME
    }

    fn collect(&self, root: &Path) -> Result<ProviderOutput> {
        if !root.is_dir() {
            bail!(
                "visual-basic provider expects a directory, got `{}`",
                root.display()
            );
        }

        let mut sources = Vec::new();
        let mut project_files = Vec::new();
        walk(root, 0, &mut sources, &mut project_files);

        if sources.is_empty() {
            tracing::debug!(root = %root.display(), "visual-basic: no VB sources found");
            return Ok(ProviderOutput::empty());
        }

        let mut parsed = Vec::with_capacity(sources.len());
        for (path, dialect) in sources {
            match read_source(&path) {
                Ok(text) => parsed.push(parse_file(&path, dialect, &text)),
                Err(err) => {
                    // One unreadable file must not sink the whole run.
                    tracing::warn!(path = %path.display(), %err, "visual-basic: could not read source; skipping");
                }
            }
        }

        Ok(emit(root, &parsed, &project_files))
    }
}

/// Turns parsed files into ontology facts.
///
/// Two passes, because base types and interfaces are resolved *across* files: a
/// class in `A.vb` may inherit a base declared in `B.vb`. Pass one creates every
/// entity and registers every declared type; pass two resolves references
/// against that registry, synthesising an `external: true` placeholder symbol
/// for anything declared outside the scanned tree (`System.Exception`,
/// `IDisposable`, ...). Emitting a placeholder rather than dropping the edge
/// keeps the fact — "this class extends *something* called X" is true and useful
/// even when X was never seen.
fn emit(root: &Path, files: &[ParsedFile], project_files: &[PathBuf]) -> ProviderOutput {
    let mut entities: Vec<Entity> = Vec::new();
    let mut relationships: Vec<Relationship> = Vec::new();

    // --- Project artifacts --------------------------------------------------
    let mut projects: Vec<(PathBuf, EntityId)> = Vec::new();
    for path in project_files {
        let entity = Entity::new(EntityKind::Artifact, path.display().to_string(), PROVIDER_NAME).with_metadata(json!({
            "kind": "project",
            "path": path.display().to_string(),
            "dialect": match path.extension().and_then(|e| e.to_str()) {
                Some(e) if e.eq_ignore_ascii_case("vbp") => Dialect::Classic.as_str(),
                _ => Dialect::VbNet.as_str(),
            },
        }));
        projects.push((path.parent().unwrap_or(root).to_path_buf(), entity.id));
        entities.push(entity);
    }

    // A bare directory of sources is itself a project, per the ontology mapping.
    // Only created when it is actually needed, so a tree that *does* have
    // `.vbproj` files does not also grow a phantom root project.
    let mut root_artifact: Option<EntityId> = None;

    // --- Pass one: every entity, and the declared-type registry -------------
    //
    // Keyed lowercase because Visual Basic is case-insensitive: `Inherits
    // shape` must resolve to `Class Shape`.
    let mut declared_types: HashMap<String, EntityId> = HashMap::new();
    let mut file_ids: Vec<EntityId> = Vec::with_capacity(files.len());
    let mut decl_ids: Vec<Vec<EntityId>> = Vec::with_capacity(files.len());

    for file in files {
        let file_entity = Entity::new(
            EntityKind::Artifact,
            file.path.display().to_string(),
            PROVIDER_NAME,
        )
        .with_metadata(json!({
            "kind": "source",
            "path": file.path.display().to_string(),
            "dialect": file.dialect.as_str(),
            "options": file.options,
            "imports": file.imports,
        }));
        let file_id = file_entity.id;
        entities.push(file_entity);
        file_ids.push(file_id);

        // Associate the file with the innermost project whose directory
        // contains it; failing that, with the scanned root.
        let owner = projects
            .iter()
            .filter(|(dir, _)| file.path.starts_with(dir))
            .max_by_key(|(dir, _)| dir.components().count())
            .map(|(_, id)| *id);
        let owner = owner.unwrap_or_else(|| {
            *root_artifact.get_or_insert_with(|| {
                let entity =
                    Entity::new(EntityKind::Artifact, root.display().to_string(), PROVIDER_NAME).with_metadata(json!({
                        "kind": "project",
                        "path": root.display().to_string(),
                        "dialect": "unknown",
                        "synthetic": true,
                    }));
                let id = entity.id;
                entities.push(entity);
                id
            })
        });
        relationships.push(Relationship::new(file_id, owner, RelationshipKind::LocatedIn));

        let mut ids = Vec::with_capacity(file.decls.len());
        for decl in &file.decls {
            let qualified = qualified_name(&file.decls, decl);
            let container = decl
                .parent
                .and_then(|p| file.decls.get(p))
                .map(|p| p.name.clone());

            let entity = Entity::new(EntityKind::Symbol, decl.name.clone(), PROVIDER_NAME).with_metadata(json!({
                "kind": decl.kind.as_str(),
                "path": file.path.display().to_string(),
                "line": decl.line,
                "dialect": file.dialect.as_str(),
                "accessibility": decl.accessibility,
                "modifiers": decl.modifiers,
                "qualified_name": qualified,
                "container": container,
                "type": decl.ty,
                "params": decl.params,
                "type_params": decl.type_params,
                "accessor": decl.accessor,
                "implements_member": decl.implements_member,
                "lib": decl.lib,
                "alias": decl.alias,
                "signature": decl.signature,
            }));
            let id = entity.id;
            entities.push(entity);
            ids.push(id);

            // Every symbol is located in its file; a member is *also* located in
            // its containing type.
            relationships.push(Relationship::new(id, file_id, RelationshipKind::LocatedIn));
            if let Some(parent) = decl.parent
                && let Some(parent_id) = ids.get(parent)
            {
                relationships.push(Relationship::new(id, *parent_id, RelationshipKind::LocatedIn));
            }

            if decl.kind.is_type() {
                // First declaration wins on a name collision. VB.NET's
                // `Partial Class` legitimately declares the same type twice, and
                // two unrelated files may each hold a `Class Utils`; resolving
                // that properly needs namespace-aware binding, which is a
                // compiler's job, not a declaration extractor's.
                declared_types
                    .entry(decl.name.to_ascii_lowercase())
                    .or_insert(id);
                declared_types
                    .entry(qualified.to_ascii_lowercase())
                    .or_insert(id);
            }
        }
        decl_ids.push(ids);
    }

    // --- Pass two: cross-references ----------------------------------------
    let mut externals: HashMap<String, EntityId> = HashMap::new();
    let mut imports: HashMap<String, EntityId> = HashMap::new();

    for (fi, file) in files.iter().enumerate() {
        let file_id = file_ids[fi];

        for namespace in &file.imports {
            let id = *imports
                .entry(namespace.to_ascii_lowercase())
                .or_insert_with(|| {
                    let entity = Entity::new(EntityKind::Symbol, namespace.clone(), PROVIDER_NAME).with_metadata(json!({
                        "kind": "Import",
                        "external": true,
                    }));
                    let id = entity.id;
                    entities.push(entity);
                    id
                });
            relationships.push(Relationship::new(file_id, id, RelationshipKind::DependsOn));
        }

        for (di, decl) in file.decls.iter().enumerate() {
            let Some(&decl_id) = decl_ids[fi].get(di) else {
                continue;
            };

            for base in &decl.inherits {
                let base_id = resolve_type(base, &declared_types, &mut externals, &mut entities);
                // `Inherits` has no first-class ontology variant. `Custom` keeps
                // "is a" distinct from `DependsOn`'s "uses", which a translation
                // very much needs.
                relationships.push(Relationship::new(
                    decl_id,
                    base_id,
                    RelationshipKind::Custom("inherits".to_string()),
                ));
            }

            for iface in &decl.implements {
                let iface_id = resolve_type(iface, &declared_types, &mut externals, &mut entities);
                // Direction is interface -> implementor, matching the ontology's
                // documented `Function -implemented_by-> Rust Module`.
                relationships.push(Relationship::new(
                    iface_id,
                    decl_id,
                    RelationshipKind::ImplementedBy,
                ));
            }
        }
    }

    ProviderOutput {
        entities,
        relationships,
    }
}

/// Resolves a type reference to an entity, creating an `external` placeholder
/// for types declared outside the scanned tree.
fn resolve_type(
    name: &str,
    declared: &HashMap<String, EntityId>,
    externals: &mut HashMap<String, EntityId>,
    entities: &mut Vec<Entity>,
) -> EntityId {
    let key = name.to_ascii_lowercase();
    if let Some(id) = declared.get(&key) {
        return *id;
    }
    // `Inherits System.Exception` may still name a type we *did* parse, under
    // its simple name.
    if let Some(simple) = key.rsplit('.').next()
        && let Some(id) = declared.get(simple)
    {
        return *id;
    }
    *externals.entry(key).or_insert_with(|| {
        let entity = Entity::new(EntityKind::Symbol, name.to_string(), PROVIDER_NAME).with_metadata(json!({
            "kind": "Type",
            "external": true,
        }));
        let id = entity.id;
        entities.push(entity);
        id
    })
}

/// Builds `Namespace.Class.Member` by walking the parent chain.
fn qualified_name(decls: &[Decl], decl: &Decl) -> String {
    let mut parts = vec![decl.name.as_str()];
    let mut cursor = decl.parent;
    // Bounded by the number of declarations, so a corrupt `parent` cycle (which
    // the parser cannot produce, since a parent is always an earlier index)
    // still terminates.
    for _ in 0..decls.len() {
        let Some(index) = cursor else {
            break;
        };
        let Some(parent) = decls.get(index) else {
            break;
        };
        parts.push(parent.name.as_str());
        cursor = parent.parent;
    }
    parts.reverse();
    parts.join(".")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// Writes `files` into a fresh temp dir and runs the provider over it.
    fn collect(files: &[(&str, &str)]) -> (TempDir, ProviderOutput) {
        let dir = tempfile::tempdir().expect("tempdir");
        for (name, contents) in files {
            let path = dir.path().join(name);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).expect("mkdir");
            }
            fs::write(&path, contents).expect("write");
        }
        let output = VisualBasicProvider::new().collect(dir.path()).expect("collect");
        (dir, output)
    }

    fn symbol<'a>(output: &'a ProviderOutput, name: &str) -> &'a Entity {
        output
            .entities
            .iter()
            .find(|e| e.kind == EntityKind::Symbol && e.name == name)
            .unwrap_or_else(|| panic!("no symbol named `{name}` in {:?}", names(output)))
    }

    /// Finds a symbol by its fully-qualified name. Necessary whenever a simple
    /// name is ambiguous — an interface member, an abstract member and its
    /// override are three distinct declarations all called `Solve`.
    fn qualified<'a>(output: &'a ProviderOutput, qualified_name: &str) -> &'a Entity {
        output
            .entities
            .iter()
            .find(|e| e.kind == EntityKind::Symbol && e.metadata["qualified_name"] == qualified_name)
            .unwrap_or_else(|| panic!("no symbol qualified `{qualified_name}`"))
    }

    fn names(output: &ProviderOutput) -> Vec<&str> {
        output
            .entities
            .iter()
            .filter(|e| e.kind == EntityKind::Symbol)
            .map(|e| e.name.as_str())
            .collect()
    }

    fn meta<'a>(entity: &'a Entity, key: &str) -> &'a serde_json::Value {
        &entity.metadata[key]
    }

    fn has_symbol(output: &ProviderOutput, name: &str) -> bool {
        output
            .entities
            .iter()
            .any(|e| e.kind == EntityKind::Symbol && e.name == name)
    }

    fn related(output: &ProviderOutput, from: EntityId, to: EntityId, kind: &RelationshipKind) -> bool {
        output
            .relationships
            .iter()
            .any(|r| r.from == from && r.to == to && r.kind == *kind)
    }

    const MODULE_VB: &str = r#"
Imports System
Imports System.Collections.Generic

Namespace Solver

    Public Module MathUtils
        Public Const Tolerance As Double = 0.000001

        Public Sub Reset()
            Dim scratch As Integer = 0
        End Sub

        Public Function Residual(ByVal a As Double, ByVal b As Double) As Double
            Return a - b
        End Function
    End Module

End Namespace
"#;

    #[test]
    fn extracts_module_sub_function_and_const() {
        let (_dir, out) = collect(&[("MathUtils.vb", MODULE_VB)]);

        assert_eq!(meta(symbol(&out, "MathUtils"), "kind"), "Module");
        assert_eq!(meta(symbol(&out, "Reset"), "kind"), "Sub");
        assert_eq!(meta(symbol(&out, "Residual"), "kind"), "Function");
        assert_eq!(meta(symbol(&out, "Residual"), "type"), "Double");
        assert_eq!(meta(symbol(&out, "Tolerance"), "kind"), "Const");
        assert_eq!(meta(symbol(&out, "Solver"), "kind"), "Namespace");

        // Provenance is a hard requirement.
        assert!(out.entities.iter().all(|e| e.source == PROVIDER_NAME));

        // The qualified name walks the container chain.
        assert_eq!(
            meta(symbol(&out, "Residual"), "qualified_name"),
            "Solver.MathUtils.Residual"
        );
    }

    #[test]
    fn statement_bodies_are_skipped_so_locals_are_not_symbols() {
        let (_dir, out) = collect(&[("MathUtils.vb", MODULE_VB)]);
        assert!(
            !has_symbol(&out, "scratch"),
            "a local `Dim` inside a Sub must not become a symbol: {:?}",
            names(&out)
        );
    }

    #[test]
    fn members_are_located_in_file_and_in_containing_type() {
        let (_dir, out) = collect(&[("MathUtils.vb", MODULE_VB)]);
        let file = out
            .entities
            .iter()
            .find(|e| e.kind == EntityKind::Artifact && e.metadata["kind"] == "source")
            .expect("source artifact");
        let module = symbol(&out, "MathUtils");
        let func = symbol(&out, "Residual");

        assert!(related(&out, func.id, file.id, &RelationshipKind::LocatedIn));
        assert!(related(&out, func.id, module.id, &RelationshipKind::LocatedIn));
    }

    #[test]
    fn imports_become_depends_on_edges_from_the_file() {
        let (_dir, out) = collect(&[("MathUtils.vb", MODULE_VB)]);
        let file = out
            .entities
            .iter()
            .find(|e| e.kind == EntityKind::Artifact && e.metadata["kind"] == "source")
            .expect("source artifact");

        for ns in ["System", "System.Collections.Generic"] {
            let import = symbol(&out, ns);
            assert_eq!(meta(import, "kind"), "Import");
            assert!(
                related(&out, file.id, import.id, &RelationshipKind::DependsOn),
                "expected file DependsOn {ns}"
            );
        }
    }

    #[test]
    fn class_inherits_and_implements() {
        let src = r#"
Public Interface ISolver
    Sub Solve()
End Interface

Public MustInherit Class Solver
    Public MustOverride Sub Solve()
End Class

Public Class GaussSeidel
    Inherits Solver
    Implements ISolver

    Private _iterations As Integer

    Public Property MaxIterations As Integer

    Public ReadOnly Property Converged As Boolean
        Get
            Return _iterations > 0
        End Get
    End Property

    Public Overrides Sub Solve() Implements ISolver.Solve
        _iterations = 1
    End Sub
End Class
"#;
        let (_dir, out) = collect(&[("Solver.vb", src)]);

        let derived = symbol(&out, "GaussSeidel");
        let base = symbol(&out, "Solver");
        let iface = symbol(&out, "ISolver");

        assert_eq!(meta(iface, "kind"), "Interface");
        assert!(
            related(
                &out,
                derived.id,
                base.id,
                &RelationshipKind::Custom("inherits".to_string())
            ),
            "expected GaussSeidel -inherits-> Solver"
        );
        assert!(
            related(&out, iface.id, derived.id, &RelationshipKind::ImplementedBy),
            "expected ISolver -implemented_by-> GaussSeidel"
        );

        // Both property forms, and the field, are members of the class.
        for member in ["MaxIterations", "Converged", "_iterations"] {
            let m = symbol(&out, member);
            assert!(
                related(&out, m.id, derived.id, &RelationshipKind::LocatedIn),
                "{member} should be located in GaussSeidel"
            );
        }

        assert_eq!(meta(symbol(&out, "MaxIterations"), "kind"), "Property");
        assert_eq!(meta(symbol(&out, "_iterations"), "kind"), "Field");
        assert_eq!(meta(symbol(&out, "_iterations"), "accessibility"), "Private");

        // Three declarations are named `Solve` — the interface's, the abstract
        // one, and the override. Only the override carries an `Implements`
        // clause, and it is recorded as metadata rather than as an edge (see
        // `read_member_implements`).
        assert_eq!(
            meta(qualified(&out, "GaussSeidel.Solve"), "implements_member"),
            "ISolver.Solve"
        );
        assert_eq!(
            meta(qualified(&out, "ISolver.Solve"), "implements_member"),
            &serde_json::Value::Null
        );
        // The `MustOverride` member has no `End Sub`; if a block had been pushed
        // for it, `Solver` would never have closed and `GaussSeidel` would be
        // nested inside it.
        assert_eq!(meta(qualified(&out, "Solver.Solve"), "kind"), "Sub");
        assert_eq!(meta(derived, "container"), &serde_json::Value::Null);
    }

    #[test]
    fn a_generic_methods_type_parameters_are_not_mistaken_for_its_arguments() {
        let src = r#"
Public Module Swaps
    Public Sub Swap(Of T)(ByRef a As T, ByRef b As T)
    End Sub
End Module
"#;
        let (_dir, out) = collect(&[("s.vb", src)]);
        let swap = symbol(&out, "Swap");
        assert_eq!(meta(swap, "type_params"), "Of T");
        assert_eq!(meta(swap, "params"), "ByRef a As T, ByRef b As T");
    }

    /// The auto-property / expanded-property ambiguity is the one thing most
    /// likely to corrupt the block stack: an auto-property has no
    /// `End Property`. If soft blocks were mishandled, `Tail` would end up
    /// nested inside `Converged` instead of the class, or vanish entirely.
    #[test]
    fn auto_properties_do_not_corrupt_the_block_stack() {
        let src = r#"
Public Class Config
    Public Property Alpha As Double
    Public Property Beta As Double

    Public ReadOnly Property Gamma As Double
        Get
            Return 1.0
        End Get
    End Property

    Public Sub Tail()
    End Sub
End Class

Public Class After
End Class
"#;
        let (_dir, out) = collect(&[("Config.vb", src)]);
        let config = symbol(&out, "Config");

        for member in ["Alpha", "Beta", "Gamma", "Tail"] {
            assert!(
                related(&out, symbol(&out, member).id, config.id, &RelationshipKind::LocatedIn),
                "{member} must be a direct member of Config, not nested in a property"
            );
        }
        // The class after it must still be top-level.
        assert_eq!(meta(symbol(&out, "After"), "container"), &serde_json::Value::Null);
    }

    /// The classic VB bug. `PUBLIC SUB` and `public sub` are the same thing.
    #[test]
    fn parsing_is_case_insensitive() {
        let shouty = r#"
PUBLIC MODULE MATHUTILS
    PUBLIC SUB FOO()
    END SUB
    PUBLIC FUNCTION BAR() AS INTEGER
    END FUNCTION
END MODULE
"#;
        let quiet = r#"
public module MATHUTILS
    public sub FOO()
    end sub
    public function BAR() as INTEGER
    end function
end module
"#;
        let (_a, upper) = collect(&[("a.vb", shouty)]);
        let (_b, lower) = collect(&[("a.vb", quiet)]);

        let key = |out: &ProviderOutput| -> Vec<(String, String, String)> {
            out.entities
                .iter()
                .filter(|e| e.kind == EntityKind::Symbol)
                .map(|e| {
                    (
                        e.name.clone(),
                        e.metadata["kind"].to_string(),
                        e.metadata["accessibility"].to_string(),
                    )
                })
                .collect()
        };

        assert_eq!(key(&upper), key(&lower));
        assert_eq!(meta(symbol(&upper, "FOO"), "kind"), "Sub");
        assert_eq!(meta(symbol(&upper, "BAR"), "kind"), "Function");
        assert_eq!(meta(symbol(&upper, "BAR"), "type"), "INTEGER");
        assert_eq!(relationship_count(&upper), relationship_count(&lower));
    }

    fn relationship_count(out: &ProviderOutput) -> usize {
        out.relationships.len()
    }

    /// `Inherits shape` must find `Class Shape`.
    #[test]
    fn type_resolution_is_case_insensitive() {
        let src = r#"
Public Class Shape
End Class

Public Class Circle
    Inherits shape
End Class
"#;
        let (_dir, out) = collect(&[("s.vb", src)]);
        assert!(related(
            &out,
            symbol(&out, "Circle").id,
            symbol(&out, "Shape").id,
            &RelationshipKind::Custom("inherits".to_string())
        ));
        // No external placeholder should have been invented.
        assert!(
            !out.entities
                .iter()
                .any(|e| e.metadata["external"] == serde_json::Value::Bool(true)),
            "`shape` should have resolved to the declared `Shape`"
        );
    }

    #[test]
    fn line_continuations_are_joined() {
        let src = "
Public Class Grid
    Public Function Interpolate( _
        ByVal x As Double, _
        ByVal y As Double _
    ) As Double
    End Function

    Public _
    Shared _
    Sub Clear()
    End Sub
End Class
";
        let (_dir, out) = collect(&[("g.vb", src)]);

        let f = symbol(&out, "Interpolate");
        assert_eq!(meta(f, "kind"), "Function");
        assert_eq!(meta(f, "type"), "Double");
        assert!(
            meta(f, "params").as_str().unwrap_or_default().contains("y As Double"),
            "continued parameter list should be joined: {:?}",
            meta(f, "params")
        );
        // The reported line is where the declaration *starts*.
        assert_eq!(meta(f, "line"), 3);

        let c = symbol(&out, "Clear");
        assert_eq!(meta(c, "kind"), "Sub");
        assert_eq!(meta(c, "accessibility"), "Public");
        assert!(
            c.metadata["modifiers"]
                .as_array()
                .is_some_and(|m| m.iter().any(|x| x == "Shared")),
            "modifiers split across continuations should all be seen"
        );
    }

    /// Comments must not eat code, and code must not eat comments. An
    /// apostrophe inside a string literal is not a comment, and `""` is an
    /// escaped quote — a naive `split('\'')` gets every one of these wrong.
    #[test]
    fn comments_and_apostrophes_in_strings_do_not_confuse_the_lexer() {
        let src = r#"
' Public Sub Ghost()   <- a comment, not a declaration
REM Public Sub AlsoGhost()
Public Module Quoting
    ' This one is real, despite the comment above it.
    Public Const Message As String = "it's a trap ' End Module"
    Public Const Escaped As String = "she said ""End Sub"" loudly"
    Public Const Remainder As Integer = 1   ' REM-like word inside an identifier
    Public Sub Real()   ' trailing comment
    End Sub
End Module
"#;
        let (_dir, out) = collect(&[("q.vb", src)]);

        assert!(!has_symbol(&out, "Ghost"), "commented-out code must not parse");
        assert!(!has_symbol(&out, "AlsoGhost"), "REM comments must not parse");
        assert!(has_symbol(&out, "Real"));
        assert!(has_symbol(&out, "Message"));
        assert!(has_symbol(&out, "Escaped"));
        // `Remainder` starts with "Rem" — the lexer must not treat it as a comment.
        assert!(has_symbol(&out, "Remainder"));

        // The strings contained `End Module` / `End Sub`; if the lexer had let
        // those through as words, the block stack would have unwound and `Real`
        // would not be a member of `Quoting`.
        assert!(related(
            &out,
            symbol(&out, "Real").id,
            symbol(&out, "Quoting").id,
            &RelationshipKind::LocatedIn
        ));
    }

    /// A VBA `.bas`/`.cls` has no `Module`/`Class` line at all: the container is
    /// implicit, named by `Attribute VB_Name`.
    #[test]
    fn vba_implicit_container_and_classic_property_accessors() {
        let bas = r#"
Attribute VB_Name = "Inventory"
Option Explicit

Public Const MaxItems As Double = 1.0

Public Sub Reset()
End Sub

Private Function Total(ByVal r As Double) As Double
End Function
"#;
        let cls = r#"
Attribute VB_Name = "Thermostat"
Option Explicit

Private mTemp As Double

Public Property Get Temperature() As Double
    Temperature = mTemp
End Property

Public Property Let Temperature(ByVal value As Double)
    mTemp = value
End Property
"#;
        let (_dir, out) = collect(&[("Inventory.bas", bas), ("Thermostat.cls", cls)]);

        let module = symbol(&out, "Inventory");
        assert_eq!(meta(module, "kind"), "Module");
        assert_eq!(meta(module, "dialect"), "vb-classic");
        assert!(related(
            &out,
            symbol(&out, "Reset").id,
            module.id,
            &RelationshipKind::LocatedIn
        ));
        assert!(related(
            &out,
            symbol(&out, "Total").id,
            module.id,
            &RelationshipKind::LocatedIn
        ));

        let class = symbol(&out, "Thermostat");
        assert_eq!(meta(class, "kind"), "Class");

        // Two `Temperature` properties: a Get and a Let. Both must survive, and
        // both must know which accessor they are.
        let accessors: Vec<String> = out
            .entities
            .iter()
            .filter(|e| e.name == "Temperature")
            .map(|e| e.metadata["accessor"].to_string())
            .collect();
        assert_eq!(accessors.len(), 2, "Property Get and Property Let are distinct declarations");
        assert!(accessors.contains(&"\"Get\"".to_string()));
        assert!(accessors.contains(&"\"Let\"".to_string()));

        // `Option Explicit` is recorded on the file artifact.
        let artifact = out
            .entities
            .iter()
            .find(|e| e.kind == EntityKind::Artifact && e.name.ends_with("Inventory.bas"))
            .expect("bas artifact");
        assert_eq!(artifact.metadata["options"][0], "Explicit");
    }

    /// VB6's `Type`, its `Declare` P/Invoke surface, and its type characters.
    /// This is exactly the legacy shape KOPITIAM exists to translate.
    #[test]
    fn vb6_type_block_declare_and_type_characters() {
        let src = r#"
Attribute VB_Name = "Legacy"
Option Explicit

Public Type NodeRecord
    Id As Long
    Label As String
    Weight As Double
End Type

Public Declare Function GetTickCount Lib "kernel32" () As Long
Private Declare Sub Sleep Lib "kernel32" Alias "Sleep" (ByVal ms As Long)

Public gCount%
Public gName$

Public Sub Run()
End Sub
"#;
        let (_dir, out) = collect(&[("Legacy.bas", src)]);

        let rec = symbol(&out, "NodeRecord");
        assert_eq!(meta(rec, "kind"), "Structure");
        for (field, ty) in [("Id", "Long"), ("Label", "String"), ("Weight", "Double")] {
            let f = symbol(&out, field);
            assert_eq!(meta(f, "kind"), "Field");
            assert_eq!(meta(f, "type"), ty);
            assert!(related(&out, f.id, rec.id, &RelationshipKind::LocatedIn));
        }

        let tick = symbol(&out, "GetTickCount");
        assert_eq!(meta(tick, "kind"), "Declare");
        assert_eq!(meta(tick, "lib"), "kernel32");
        assert_eq!(meta(tick, "type"), "Long");

        let sleep = symbol(&out, "Sleep");
        assert_eq!(meta(sleep, "alias"), "Sleep");
        assert_eq!(meta(sleep, "accessibility"), "Private");

        // Type characters: `%` is Integer, `$` is String.
        assert_eq!(meta(symbol(&out, "gCount"), "type"), "Integer");
        assert_eq!(meta(symbol(&out, "gName"), "type"), "String");

        // The `Type ... End Type` block must have closed, so `Run` is a module
        // member, not a struct field.
        assert!(related(
            &out,
            symbol(&out, "Run").id,
            symbol(&out, "Legacy").id,
            &RelationshipKind::LocatedIn
        ));
    }

    /// `Dim a, b As Integer` means different things in the two dialects. If this
    /// test ever "fails" because someone unified the branches, read
    /// [`parse_declarators`]'s docs before changing it.
    #[test]
    fn multi_declarator_typing_differs_between_dialects() {
        let vbnet = r#"
Public Module M
    Public a, b As Integer
End Module
"#;
        let classic = "
Attribute VB_Name = \"M\"
Public a, b As Integer
";
        let (_x, net) = collect(&[("m.vb", vbnet)]);
        let (_y, old) = collect(&[("m.bas", classic)]);

        // VB.NET: the type reaches back.
        assert_eq!(meta(symbol(&net, "a"), "type"), "Integer");
        assert_eq!(meta(symbol(&net, "b"), "type"), "Integer");

        // VB6/VBA: it does not. `a` is a Variant.
        assert_eq!(meta(symbol(&old, "a"), "type"), "Variant");
        assert_eq!(meta(symbol(&old, "b"), "type"), "Integer");
    }

    #[test]
    fn enum_members_and_interface_members_are_extracted() {
        let src = r#"
Public Enum SolverKind
    Direct = 0
    Iterative = 1
    Multigrid
End Enum

Public Interface IStepper
    Sub Step()
    Function Error() As Double
    Property Dt As Double
End Interface

Public Module Tail
End Module
"#;
        let (_dir, out) = collect(&[("e.vb", src)]);

        let kind = symbol(&out, "SolverKind");
        assert_eq!(meta(kind, "kind"), "Enum");
        for member in ["Direct", "Iterative", "Multigrid"] {
            let m = symbol(&out, member);
            assert_eq!(meta(m, "kind"), "EnumMember");
            assert!(related(&out, m.id, kind.id, &RelationshipKind::LocatedIn));
        }

        // Interface members have no `End Sub`. If we had pushed blocks for them,
        // `Tail` would have ended up nested inside `IStepper`.
        let iface = symbol(&out, "IStepper");
        for member in ["Step", "Error", "Dt"] {
            assert!(related(
                &out,
                symbol(&out, member).id,
                iface.id,
                &RelationshipKind::LocatedIn
            ));
        }
        assert_eq!(meta(symbol(&out, "Tail"), "container"), &serde_json::Value::Null);
    }

    #[test]
    fn project_files_become_artifacts_that_own_their_sources() {
        let (_dir, out) = collect(&[
            ("Solver/Solver.vbproj", "<Project Sdk=\"Microsoft.NET.Sdk\" />"),
            ("Solver/Solver.vb", "Public Module Solver\nEnd Module\n"),
        ]);

        let project = out
            .entities
            .iter()
            .find(|e| e.metadata["kind"] == "project")
            .expect("project artifact");
        assert!(project.name.ends_with("Solver.vbproj"));
        assert_eq!(project.metadata["dialect"], "vb.net");

        let source = out
            .entities
            .iter()
            .find(|e| e.metadata["kind"] == "source")
            .expect("source artifact");
        assert!(related(&out, source.id, project.id, &RelationshipKind::LocatedIn));
    }

    #[test]
    fn a_bare_directory_of_sources_is_itself_the_project() {
        let (_dir, out) = collect(&[("loose.vb", "Public Module Loose\nEnd Module\n")]);
        let project = out
            .entities
            .iter()
            .find(|e| e.metadata["kind"] == "project")
            .expect("synthetic root project artifact");
        assert_eq!(project.metadata["synthetic"], true);
    }

    #[test]
    fn external_base_types_become_external_placeholders() {
        let src = r#"
Public Class SolverError
    Inherits System.Exception
    Implements IDisposable
End Class
"#;
        let (_dir, out) = collect(&[("x.vb", src)]);
        let base = symbol(&out, "System.Exception");
        assert_eq!(meta(base, "external"), true);
        assert!(related(
            &out,
            symbol(&out, "SolverError").id,
            base.id,
            &RelationshipKind::Custom("inherits".to_string())
        ));
        assert!(related(
            &out,
            symbol(&out, "IDisposable").id,
            symbol(&out, "SolverError").id,
            &RelationshipKind::ImplementedBy
        ));
    }

    // -- Degradation ---------------------------------------------------------

    #[test]
    fn a_directory_with_no_visual_basic_yields_nothing() {
        let (_dir, out) = collect(&[("README.md", "# not visual basic"), ("main.rs", "fn main() {}")]);
        assert!(out.entities.is_empty());
        assert!(out.relationships.is_empty());
    }

    #[test]
    fn a_nonexistent_root_is_an_honest_error_not_a_panic() {
        let err = VisualBasicProvider::new()
            .collect(Path::new("/kopitiam/definitely/not/here"))
            .expect_err("a nonexistent root must be reported, not silently ignored");
        assert!(err.to_string().contains("expects a directory"));
    }

    #[test]
    fn malformed_truncated_and_unbalanced_sources_do_not_panic() {
        let cases: &[&str] = &[
            // Truncated mid-declaration.
            "Public Class",
            "Public Sub",
            "Public Property Get",
            // Unbalanced terminators.
            "End Class\nEnd Sub\nEnd Namespace\n",
            // Unterminated blocks.
            "Public Class C\n    Public Sub S()\n",
            // Unterminated string and unbalanced parens.
            "Public Const S As String = \"unterminated\nPublic Sub F(a As\n",
            // A dangling continuation at EOF.
            "Public Sub F() _",
            // A bracketed identifier that is never closed.
            "Public Class [Unclosed\n",
            // VB6's bare `End` statement, which is not a block terminator.
            "Public Sub Halt()\n    End\nEnd Sub\n",
            // Not Visual Basic at all.
            "#include <stdio.h>\nint main(void) { return 0; }\n",
            // Deeply nested garbage.
            "((((((((((\n))))))))))\n",
        ];

        for (i, src) in cases.iter().enumerate() {
            let (_dir, out) = collect(&[("weird.vb", src)]);
            // The contract is "no panic", not "no facts" — some of these do
            // legitimately yield partial declarations.
            assert!(
                out.entities.iter().all(|e| e.source == PROVIDER_NAME),
                "case {i} produced a fact with the wrong provenance"
            );
        }
    }

    #[test]
    fn a_bare_end_statement_does_not_close_the_enclosing_sub() {
        // VB6's `End` (terminate the program) is not `End Sub`. If it were
        // treated as a block terminator, `Second` would be parsed as a member of
        // nothing and `Halt`'s real `End Sub` would unwind the class.
        let src = r#"
Public Class App
    Public Sub Halt()
        End
    End Sub
    Public Sub Second()
    End Sub
End Class
"#;
        let (_dir, out) = collect(&[("app.vb", src)]);
        let app = symbol(&out, "App");
        assert!(related(&out, symbol(&out, "Halt").id, app.id, &RelationshipKind::LocatedIn));
        assert!(related(&out, symbol(&out, "Second").id, app.id, &RelationshipKind::LocatedIn));
    }

    #[test]
    fn non_utf8_legacy_encodings_are_read_rather_than_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Windows-1252: 0xE9 is `é`, which is not valid UTF-8 on its own.
        let mut bytes = b"Public Module Caf".to_vec();
        bytes.push(0xE9);
        bytes.extend_from_slice(b"\n    ' comment with a \xE9 in it\n    Public Sub Brew()\n    End Sub\nEnd Module\n");
        fs::write(dir.path().join("legacy.vb"), &bytes).expect("write");

        let out = VisualBasicProvider::new().collect(dir.path()).expect("collect");
        assert!(
            has_symbol(&out, "Brew"),
            "a Windows-1252 source must still parse: {:?}",
            names(&out)
        );
    }

    #[test]
    fn build_output_directories_are_not_walked() {
        let (_dir, out) = collect(&[
            ("src.vb", "Public Module Real\nEnd Module\n"),
            ("bin/Debug/generated.vb", "Public Module Generated\nEnd Module\n"),
            ("obj/tmp.vb", "Public Module Temp\nEnd Module\n"),
        ]);
        assert!(has_symbol(&out, "Real"));
        assert!(!has_symbol(&out, "Generated"));
        assert!(!has_symbol(&out, "Temp"));
    }
}

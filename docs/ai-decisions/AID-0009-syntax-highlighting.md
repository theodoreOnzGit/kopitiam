# AID-0009: Syntax highlighting — no tree-sitter, hand-written pure-Rust lexers instead

> **Amendment (2026-07-15):** the project scope was later refocused away from
> engineering-simulation domains. This AID's references to a scientific-computing
> target domain are historical and are preserved unchanged below as a record of
> the reasoning at the time.

* **Status:** Pending review
* **Bead:** `kopitiam-2qi`
* **Date:** 2026-07-14
* **Decided by:** AI (Claude), maintainer absent

## The brief

> Incorporate tree-sitter in pure Rust to build incremental syntax
> highlighting for KOPITIAM's editor.

## The premise, checked rather than assumed

CLAUDE.md's Pure Rust Core section is unambiguous: the core platform must
build with stable Rust and Cargo alone, with no mandatory C/C++/Fortran, and
"when choosing between a mature C/C++ dependency and a good Rust
implementation, prefer the Rust implementation." Tree-sitter's runtime is a
C library; the `tree-sitter` crate is a *binding* to it, not a
reimplementation. Rather than take the maintainer's "in pure Rust" at face
value and hope it's true of tree-sitter today, I checked, directly, in this
environment, three separate ways.

**1. Is there a complete pure-Rust tree-sitter runtime?** Essentially no.
There is exactly one attempt, `tree-sitter-c2rust` (crates.io, by a third
party, `shadaj`, not the tree-sitter project), and I downloaded and read it
rather than trust the crate description. Its `build.rs` no longer invokes a
C compiler (confirmed: it only copies a text file into `OUT_DIR`), and its
~22,000-line `binding_rust/core_wrapper/` is genuine Rust source with no `cc`
build-dependency anywhere in its manifest — so the *build-without-a-C-compiler*
claim is true. But reading the code itself:

```rust
pub type __uint8_t = libc::c_uchar;
...
#[derive(Copy, Clone)]
#[repr(C)]
pub struct C2RustUnnamed {
    pub states: *const bool,
    pub symbol_map: *const TSSymbol,
    pub create: Option<unsafe extern "C" fn() -> *mut libc::c_void>,
    ...
```

This is `c2rust`'s automated C→Rust transpilation output: `libc::c_int`
field types, `#[repr(C)]` structs, raw-pointer arithmetic, `unsafe extern
"C" fn` pointers throughout. It compiles with `rustc` alone, which is the
letter of the Pure Rust Core rule. It is not what CLAUDE.md means by "a good
Rust implementation" — it is C, mechanically re-encoded in Rust syntax, with
none of the ownership/borrowing/type-safety properties the rule exists to
get. Treating this as satisfying "prefer the Rust implementation" would be
following the letter of the rule to defeat its purpose.

**2. Does that cover the grammars too?** No, and this is the part that
actually kills the option. The core parsing *engine* is one C library;
every *grammar* (`tree-sitter-rust`, `tree-sitter-lua`, `tree-sitter-toml`,
...) is a separate crate shipping a tool-generated `parser.c` compiled by a
`build.rs` that depends on the `cc` crate. I did not take this on faith
either — I added `tree-sitter-rust = "0.24.2"` to a scratch `Cargo.toml` and
ran `cargo fetch`:

```
      Adding cc v1.2.67
      Adding find-msvc-tools v0.1.9
      Adding shlex v2.0.1
      Adding tree-sitter-rust v0.24.2
```

`cc` and `find-msvc-tools` (a Windows MSVC toolchain locator) are exactly
the "mandatory C dependency" CLAUDE.md prohibits. Nobody has transpiled the
grammar ecosystem to Rust. I also checked `syntastica`
(`RubixDev/syntastica`), the most complete tree-sitter-grammar bundle on
crates.io — 68+ languages — because it advertises a `runtime-c2rust`
feature using exactly the crate from point 1. Its `syntastica-parsers`
crate's own `Cargo.toml` shows the `rust`, `lua`, `markdown` (etc.) features
each map to `dep:tree-sitter-rust`, `dep:tree-sitter-lua`,
`dep:tree-sitter-md` — the ordinary, C-backed grammar crates. Even the one
project that offers a pure-Rust *core* still depends on C-compiled grammars
for every language. "Pure-Rust tree-sitter for our six languages" would
mean transpiling and indefinitely maintaining six grammars ourselves, out
of band from upstream, forever, just to get to a starting point.

**3. Was `syntect` a live option?** Closer, and worth being precise about,
because it's exactly the "quietly pull in a C dependency" trap the brief
warned against. `cargo info syntect` shows its `default` feature is
`default-onig`, which pulls the `onig` crate — a binding to the C
Oniguruma regex library. A naive `cargo add syntect` takes the C path
silently. But syntect also ships `default-fancy`, built on `fancy-regex`,
which I confirmed is genuinely pure Rust (its dependency graph is
`regex-automata`, `regex-syntax`, `bit-set` — no `cc`, no FFI). So syntect
*can* be pure Rust, but only via a non-default feature flag, and only if
whoever adds it later remembers that. I chose not to take it anyway: its
unit of work is a Sublime `.sublime-syntax` grammar, a general regex-driven
pushdown automaton built to parse *someone else's* syntax definitions for
dozens of languages we don't use. For six languages we fully control, the
`.sublime-syntax` files would need to be vendored and license-audited, and
— the part that matters most for this crate's actual hard requirement —
correct multi-line state carry-over would still be *syntect's* internal
state machine to trust, not ours to directly test line-by-line the way the
crate-level docs and this AID's tests do.

## Is tree-sitter even the right tool here?

Checked this too, rather than assume the brief's premise that it is. I read
`crates/kopitiam-neovim/src/ui/textarea.rs` and `theme.rs` (read-only, as
instructed) to see what the renderer actually consumes. `TextArea::render`
walks visible rows and, per row, wants a `ratatui::style::Style` applied to
a slice of the line. It has no use anywhere for a parse tree, node kinds,
or tree-sitter's incremental-edit API (`Tree::edit`, byte-range
re-parsing) — those exist to answer questions like "what function contains
this cursor" or "select the enclosing block," which is real, valuable
future work (`kopitiam-semantic`'s domain, per CLAUDE.md's Semantic Runtime
table — rust-analyzer-derived facts, not syntax highlighting), but it is
not what a highlighter needs. A highlighter needs exactly what
`Highlighter::highlight_line` returns: `Vec<HighlightSpan { start, end,
kind }>`. Reaching for tree-sitter for this would have been cargo-culting
Neovim's plugin ecosystem — "editors do syntax highlighting via
tree-sitter now" — rather than solving KOPITIAM's actual requirement.

## What was decided

**Hand-written, line-oriented, stateful lexers, one per language, in a new
`kopitiam-syntax` crate — zero external dependencies.**

* Six languages: Rust, Lua, TeX, Markdown, TOML, Python — the maintainer's
  actual editing surface (this workspace's own source, `kopitiam-neovim`'s
  Lua config layer, scientific publishing, docs/AIDs, `Cargo.toml`, and the
  wider scientific-computing ecosystem CLAUDE.md lists as a target domain).
* Each language module owns a small `*State` enum threaded from the end of
  one line into the start of the next (`RustState::BlockComment { depth }`,
  `PythonState::TripleString { quote }`, ...), which is how the classic
  line-based-highlighter bug — losing highlighting at a line boundary — is
  avoided. Every module has an explicit test that a multi-line construct
  (block comment, raw/triple/long string, fenced code block, verbatim
  environment) stays correctly highlighted across the boundary, and that
  the *wrong* nesting level / fence character / quote count does not
  falsely close it.
* `Highlighter::state()` / `Highlighter::with_state()` expose that carried
  state so a caller can re-highlight incrementally: cache the state at the
  start of every line, resume from the edited line's cached state, and
  stop propagating forward the moment the newly computed exit state equals
  the old cached exit state for the following line. Tested directly
  (`state_equality_detects_when_downstream_lines_are_unaffected`).
* Every scanner is a single linear pass with no backtracking — pathological
  input (a 200,000-character line, an unterminated string) is O(line
  length) and provably cannot hang; tested explicitly in every module.
* Zero dependencies. `crates/kopitiam-syntax/Cargo.toml` has an empty
  `[dependencies]` table. Not `unicode-segmentation`, not `regex` — none of
  it was needed for byte-offset span scanning, and CLAUDE.md's "avoid
  unnecessary dependencies" is a real constraint, not just the Pure Rust
  Core one.

This mirrors the reasoning `kopitiam-lua` is independently applying to
writing a Lua VM in this same workspace (see AID-0003's Decision 1) and
AID-0007's Lua-coroutine bytecode-VM finding: "there is no acceptable
pure-Rust off-the-shelf answer, and the problem, scoped to what we actually
need, is bounded enough to own directly."

### A real bug this approach caught, that a black-box dependency wouldn't have surfaced as clearly

While testing, `toml::tests::multiline_basic_string_spans_multiple_lines`
failed: a `"""`-delimited TOML string that closed exactly on the last byte
of its closing line was left marked as still open. The root cause was
several `resume-a-multi-line-construct` functions across the Rust, Lua, and
TOML modules using `end == line.len()` as a stand-in for "ran off the end
unterminated" — which is ambiguous, because a construct that closes on the
line's very last byte *also* produces `end == line.len()`. Fixed by
changing those helpers to return `Option<usize>` (`Some(end)` = closed,
`None` = unterminated) instead of overloading a plain `usize`, with a
regression test in each affected module
(`string_closing_at_the_very_end_of_the_line_is_not_left_open`). Noted here
because it is exactly the class of subtle multi-line-boundary bug the
brief called out as "the classic bug" — and because owning the lexer is
what made it visible and fixable in the first place, rather than being an
opaque wrong answer from a dependency's internal state machine.

## What would make this wrong

* **If KOPITIAM later needs a real parse tree** — refactoring tools,
  "select enclosing block," structural editing, cross-reference navigation
  scoped by AST node — this crate's flat span output cannot serve that.
  That is not evidence this decision was wrong *for highlighting*; it is
  evidence that a *different* problem (semantic/structural tooling) needs a
  *different* solution, most likely inside `kopitiam-semantic`
  (rust-analyzer already gives KOPITIAM a real AST for Rust; the same
  question would need answering per additional language). If that need
  arrives and someone reaches for tree-sitter to answer it, the grammar
  problem documented above still applies and still has to be solved.
* **If `tree-sitter-c2rust` (or a successor) grows a companion project that
  transpiles the grammar ecosystem too**, re-evaluate. As of this writing
  no such project exists; if one appears and is genuinely maintained
  (tracking upstream grammar updates, not a one-time snapshot), the
  calculus changes and hand-rolled lexers become the *inferior* long-term
  choice for language coverage beyond what's hand-written today.
  Six-language coverage plus indefinite maintenance of six more languages
  by hand does not scale the way "add a grammar dependency" does; this
  decision is explicitly a bet that six languages, chosen to match what
  the maintainer actually edits, is the right ceiling for this approach.
* **If the maintainer considers the c2rust-transpiled runtime "pure Rust
  enough"** — i.e., disagrees that mechanically-transpiled `unsafe`/`libc`
  C-in-Rust fails "a good Rust implementation" — then the calculus in
  point 1 changes, though point 2 (the grammar ecosystem) still blocks a
  complete answer today regardless.
* **If six hand-written lexers prove to be a genuine ongoing maintenance
  burden** (each language's syntax evolving, more languages needed,
  correctness bugs recurring), that would be evidence to revisit — though
  the fix in that scenario is more likely "invest in shared scanning
  infrastructure" (already started in `util.rs`) than "adopt tree-sitter,"
  given points 1 and 2 don't change with time pressure.

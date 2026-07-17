//! Auto-indent for opening and splitting lines — the indent kvim puts on a
//! brand-new line so the maintainer stops having to `==` after every `o`.
//!
//! Covers three moves, all of which birth a fresh line:
//!
//! * `o` — open a line below the current one.
//! * `O` — open a line above the current one.
//! * `<Enter>` in Insert mode — split the current line in two.
//!
//! # What it does (vim's `autoindent` + a basic `smartindent`)
//!
//! * **autoindent:** the new line copies the *reference line*'s leading
//!   whitespace verbatim. "Reference line" is the line you opened from (`o`),
//!   the line being pushed down (`O`), or the text before the cursor (`<Enter>`
//!   split). Copying verbatim means the new line matches whatever the file
//!   already uses — tabs or spaces — without us having to sniff it out.
//! * **smartindent (basic, brace-aware):** if the reference line ends with an
//!   open bracket `{` `(` `[` (ignoring trailing whitespace and a trailing `//`
//!   line comment), the new line gets **one extra indent level**. And when the
//!   user then types a lone closing `}` `)` `]` as the first thing on that new
//!   line, [`wants_closer_dedent`] flags it so the caller pulls it back one
//!   level to line up with its
//!   opener. Between the two, Rust/C-like blocks indent right with no `==`.
//!
//! # Where the indent width comes from
//!
//! The *added* level uses the buffer's configured indent options —
//! `expandtab` picks spaces-vs-tab, `shiftwidth` (resolved against `tabstop`,
//! vim-style, so `shiftwidth=0` follows `tabstop`) picks how many spaces. kvim
//! exposes these on [`crate::config::Options`], so there is no need to *detect*
//! the file's prevailing indent — the copied leading whitespace already carries
//! the file's real style for the base, and the options carry the maintainer's
//! chosen width for the one level we add. If some future caller has neither, a
//! sensible default is 4 spaces, but nothing here falls back to that today.
//!
//! # Not `==`
//!
//! This is deliberately *not* a re-indenter. It never re-flows existing lines,
//! never counts brace depth across the file, never touches anything but the one
//! new line. `==` (see [`crate::editor::operator`]'s `reindent_c_style`) is the
//! tool for fixing a whole line/range to its computed depth; this only decides
//! the indent a *fresh, empty* line starts life with.

/// One level of indentation as literal text, honouring the buffer's indent
/// options: `expandtab` on → `shiftwidth` spaces; off → a single hard tab.
///
/// `shiftwidth` here is the already-*resolved* width (caller does
/// `options.shiftwidth.resolve(options.tabstop)`), so a configured
/// `shiftwidth=0` has already become `tabstop` before it reaches us.
pub(crate) fn one_level(shiftwidth: usize, expandtab: bool) -> String {
    if expandtab {
        " ".repeat(shiftwidth)
    } else {
        "\t".to_string()
    }
}

/// The leading whitespace (spaces and/or tabs) of `line`, as a slice of it.
/// A line that is all whitespace returns the whole line; an empty line returns
/// `""`.
pub(crate) fn leading_ws(line: &str) -> &str {
    let end = line.find(|c: char| c != ' ' && c != '\t').unwrap_or(line.len());
    &line[..end]
}

/// Does `reference`, once trailing whitespace and a trailing `//` line comment
/// are ignored, end with an opening bracket `{` `(` `[`?
///
/// This is the "should the next line indent one deeper" test. The `//`-comment
/// strip is deliberately simple — it chops at the first `//` — so a `//` living
/// *inside* a string or char literal on the reference line could fool it (e.g.
/// `let s = "http://x"; {` is read correctly, but `let s = "a // b {";` would
/// be misread as opening a block). That is an accepted limit of a lexer-free
/// heuristic; getting it exactly right needs real tokenizing, which belongs to
/// the language layer, not this typing-time helper.
pub(crate) fn opens_block(reference: &str) -> bool {
    let code = match reference.find("//") {
        Some(i) => &reference[..i],
        None => reference,
    };
    matches!(code.trim_end().chars().next_back(), Some('{') | Some('(') | Some('['))
}

/// The indent string a brand-new line should be born with, given the
/// `reference` line it is opened from or split out of.
///
/// It is `leading_ws(reference)` (autoindent — copy the file's own indent
/// verbatim) plus [`one_level`] when [`opens_block`] says the reference opens a
/// block (smartindent — go one deeper). `shiftwidth` is the resolved width; see
/// the [module docs](self) for where it comes from.
pub(crate) fn new_line_indent(reference: &str, shiftwidth: usize, expandtab: bool) -> String {
    let mut indent = leading_ws(reference).to_string();
    if opens_block(reference) {
        indent.push_str(&one_level(shiftwidth, expandtab));
    }
    indent
}

/// Should a lone closing bracket just typed here dedent the line one level?
///
/// True only when `before_cursor` — everything on the line up to the cursor —
/// is **non-empty and all whitespace**: the user has nothing but the
/// auto-indent on the line and is typing the closer as the very first
/// character. A closer that follows real code (`foo()}`) has non-whitespace in
/// `before_cursor`, so it does not dedent.
///
/// This is only the *decision*. The caller does the actual one-level dedent
/// through [`crate::editor::operator::indent_line`], so the column arithmetic
/// (a hard tab worth `tabstop` columns, never dedenting past column 0) stays in
/// the one place `<<`, `>>`, and `<C-t>`/`<C-d>` already share — no second copy
/// to drift out of step.
pub(crate) fn wants_closer_dedent(before_cursor: &str) -> bool {
    !before_cursor.is_empty() && before_cursor.chars().all(|c| c == ' ' || c == '\t')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_level_respects_expandtab() {
        assert_eq!(one_level(4, true), "    ");
        assert_eq!(one_level(2, true), "  ");
        assert_eq!(one_level(4, false), "\t");
    }

    #[test]
    fn leading_ws_reads_spaces_and_tabs() {
        assert_eq!(leading_ws("    x"), "    ");
        assert_eq!(leading_ws("\t\tx"), "\t\t");
        assert_eq!(leading_ws("no indent"), "");
        assert_eq!(leading_ws("     "), "     "); // all whitespace
        assert_eq!(leading_ws(""), "");
    }

    #[test]
    fn opens_block_detects_trailing_opener() {
        assert!(opens_block("fn foo() {"));
        assert!(opens_block("let v = vec!["));
        assert!(opens_block("foo("));
        assert!(opens_block("    if x {   ")); // trailing whitespace ignored
        assert!(opens_block("fn foo() { // start")); // trailing // comment ignored
        assert!(!opens_block("let x = 1;"));
        assert!(!opens_block("}"));
        assert!(!opens_block(""));
    }

    #[test]
    fn new_line_indent_copies_then_maybe_adds_a_level() {
        // Plain line: just copy its indent.
        assert_eq!(new_line_indent("    let x = 1;", 4, true), "    ");
        // Opener: copy + one level.
        assert_eq!(new_line_indent("    fn foo() {", 4, true), "        ");
        // Hard tabs, opener: copy the tab + add a tab.
        assert_eq!(new_line_indent("\tfn foo() {", 4, false), "\t\t");
        // Top-level opener from column 0: one level.
        assert_eq!(new_line_indent("fn foo() {", 4, true), "    ");
    }

    #[test]
    fn wants_closer_dedent_only_on_a_whitespace_only_prefix() {
        // Auto-indent present, nothing else typed yet -> dedent the closer.
        assert!(wants_closer_dedent("        "));
        assert!(wants_closer_dedent("\t\t"));
        // Nothing on the line at all (column 0) -> nothing to pull back.
        assert!(!wants_closer_dedent(""));
        // Real code before the closer -> not a lone closer.
        assert!(!wants_closer_dedent("    foo()"));
        assert!(!wants_closer_dedent("x"));
    }
}

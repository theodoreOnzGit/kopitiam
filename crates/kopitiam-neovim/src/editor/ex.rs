//! Ex commands: the `:` command line.
//!
//! # Why parsing is separate from doing
//!
//! [`parse`] turns a command-line string into an [`ExCommand`] — inert data,
//! no I/O. [`Editor::execute_ex`](super::Editor::execute_ex) (in `mod.rs`,
//! since it needs `Editor`'s buffer table) walks that data and does the
//! work, but for anything that would touch the outside world (writing a
//! file, quitting the process, opening a different file) it hands the
//! *description* of that action back to its caller as an
//! [`super::ExEffect`] instead of performing it. `:w` running inside a unit
//! test must not actually touch the filesystem unless the test wants that;
//! `:q` must not call `std::process::exit`. The caller — ultimately the
//! `apps/cli`/TUI event loop — decides what "write" and "quit" mean in its
//! context.
//!
//! Buffer-only effects (`:s`, `:d`, `:g`) do not need this indirection —
//! they are applied directly to the buffer here, because a buffer edit is
//! exactly what every other keystroke in this crate already does.

use regex::Regex;

use crate::core::{Edit, Position, Range};
use crate::text::Buffer;

use super::operator;

/// A line reference as written in a command line, resolved to a concrete
/// 0-based line index only once the buffer it applies to is known (see
/// [`LineRange::resolve`]) — parsing never needs a `Buffer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineSpec {
    Number(usize),
    Current,
    Last,
}

/// The range prefix of an ex command (`:2,4d`, `:%s///`, bare `:d`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineRange {
    /// No range was written; the command's own default applies (usually
    /// "just the current line").
    None,
    All,
    Single(LineSpec),
    Pair(LineSpec, LineSpec),
}

impl LineRange {
    /// Resolves to an inclusive, clamped, 0-based `(first, last)` pair.
    pub fn resolve(self, current_line: usize, line_count: usize) -> (usize, usize) {
        let last_idx = line_count.saturating_sub(1);
        let idx = |s: LineSpec| match s {
            LineSpec::Number(n) => n.saturating_sub(1).min(last_idx),
            LineSpec::Current => current_line.min(last_idx),
            LineSpec::Last => last_idx,
        };
        match self {
            LineRange::None => (current_line.min(last_idx), current_line.min(last_idx)),
            LineRange::All => (0, last_idx),
            LineRange::Single(s) => {
                let i = idx(s);
                (i, i)
            }
            LineRange::Pair(a, b) => {
                let (a, b) = (idx(a), idx(b));
                if a <= b { (a, b) } else { (b, a) }
            }
        }
    }
}

/// A parsed `:` command. See the module docs for the parse/execute split.
#[derive(Debug, Clone, PartialEq)]
pub enum ExCommand {
    Write { path: Option<String>, then_quit: bool, force: bool },
    Quit { force: bool },
    /// `:qa`/`:qall`/`:quita`/`:quitall` (+ optional `!`): quit *every* window
    /// and exit the editor. Without `!` this must refuse if any buffer has
    /// unsaved changes, the same guard `:q` uses but widened to all buffers.
    QuitAll { force: bool },
    /// `:wa`/`:wall` (`then_quit == false`), and `:wqa`/`:wqall`/`:xa`/`:xall`
    /// (`then_quit == true`): write every modified buffer, then — for the
    /// quit-all forms — exit. `force` carries a trailing `!` (`:wqa!`).
    WriteAll { then_quit: bool, force: bool },
    Edit { path: String },
    NextBuffer,
    PrevBuffer,
    GotoBuffer(usize),
    /// `:bd`/`:bdelete` (`wipe == false`) and `:bw`/`:bwipeout`
    /// (`wipe == true`): chuck away the current buffer. Without `!` this must
    /// refuse if the buffer got unsaved changes — same guard `:q` uses. With
    /// `!` (`force == true`) it deletes anyway and throws the changes away.
    ///
    /// vim draws a line between `:bd` (unload the buffer but keep it in the
    /// `:ls` list, marked as unlisted) and `:bw` (wipe it out completely, gone
    /// from `:ls` too). kvim doesn't yet carry that hidden/unlisted-buffer
    /// state — every open buffer is a live, listed buffer — so today both forms
    /// do the same thing: remove the buffer outright. `wipe` is kept in the
    /// grammar now so the two commands stay distinct at the parse layer, and
    /// the day kvim grows an unlisted-buffer concept, only the executor needs
    /// to change, not the command surface.
    DeleteBuffer { force: bool, wipe: bool },
    /// `:ls`/`:buffers`/`:files` — list every open buffer with its id and a
    /// modified (`+`) flag, the way vim's `:ls` does.
    ListBuffers,
    Substitute { range: LineRange, pattern: String, replacement: String, global: bool },
    Global { pattern: String, cmd: String },
    Delete { range: LineRange },
    NoHighlight,
    Set { key: String, value: Option<String> },
    GotoLine(LineSpec),
    /// `:sp`/`:vs [file]` (`scratch == false`), `:new`/`:vnew`
    /// (`scratch == true`). The window layout is the UI's to change, so
    /// `Editor::execute_ex` forwards this as `EditorResponse::Window`.
    Split { vertical: bool, file: Option<String>, scratch: bool },
    /// `:only` — close all windows but the active one.
    Only,
    /// `:close` — close the active window.
    Close,
    /// `:term`/`:terminal` — kvim has no terminal emulator yet, so this opens
    /// an honest placeholder buffer rather than a broken or silent one. See
    /// `Editor::execute_ex` and bead `kopitiam-cj0.10.4`.
    Terminal,
    /// `:help`/`:h [topic]` — open kvim's built-in Singlish help manual in a
    /// scratch buffer. `topic` is the optional `:help <topic>` argument that
    /// jumps to a section (see [`super::help`]); `None` opens at the top.
    Help { topic: Option<String> },
    /// An empty command line (`:` followed immediately by Enter).
    Empty,
    /// Parsed but not recognized — surfaced to the user as
    /// [`crate::Error::UnknownCommand`] rather than silently ignored.
    Unknown(String),
}

/// Parses one command-line entry (without its leading `:`).
pub fn parse(input: &str) -> ExCommand {
    let input = input.trim();
    if input.is_empty() {
        return ExCommand::Empty;
    }

    let (range, rest) = parse_range(input);
    let rest = rest.trim_start();

    if rest.is_empty() {
        return match range {
            LineRange::None => ExCommand::Empty,
            LineRange::Single(spec) => ExCommand::GotoLine(spec),
            LineRange::All => ExCommand::GotoLine(LineSpec::Last),
            LineRange::Pair(_, b) => ExCommand::GotoLine(b),
        };
    }

    let name_len = rest.chars().take_while(|c| c.is_ascii_alphabetic()).count();
    let name: String = rest.chars().take(name_len).collect();
    let mut after = &rest[name_len..];
    let force = after.starts_with('!');
    if force {
        after = &after[1..];
    }
    let arg = after.trim();

    match name.as_str() {
        "w" | "write" => ExCommand::Write {
            path: if arg.is_empty() { None } else { Some(arg.to_string()) },
            then_quit: false,
            force,
        },
        "q" | "quit" => ExCommand::Quit { force },
        // The quit-all / write-all family. Kept as their own commands (not `q`
        // with a count) because "all windows" is a fundamentally different
        // action from "this window": `qa` exits the editor unconditionally
        // across every split, where `q` closes one. vim's abbreviations all map
        // to the same four intents.
        "qa" | "qall" | "quita" | "quitall" => ExCommand::QuitAll { force },
        "wa" | "wall" => ExCommand::WriteAll { then_quit: false, force },
        "wqa" | "wqall" => ExCommand::WriteAll { then_quit: true, force },
        // `:xa`/`:xall` is `:wqa` — write all, then quit all. (vim's `:x` skips
        // the write when a buffer is unmodified; the write-all executor already
        // writes only modified buffers, so `xa` and `wqa` coincide exactly.)
        "xa" | "xall" => ExCommand::WriteAll { then_quit: true, force: true },
        "wq" => ExCommand::Write {
            path: if arg.is_empty() { None } else { Some(arg.to_string()) },
            then_quit: true,
            force,
        },
        // `:x`/`:xit` differ from `:wq` in real vim only by skipping the
        // write when the buffer is unmodified. `Editor::execute_ex` checks
        // `is_modified` itself before honoring `then_quit`'s implied write,
        // so treating `x` as `wq` here is exact, not an approximation.
        "x" | "xit" => ExCommand::Write { path: None, then_quit: true, force: true },
        "e" | "edit" => ExCommand::Edit { path: arg.to_string() },
        "bn" | "bnext" => ExCommand::NextBuffer,
        "bp" | "bprev" | "bprevious" => ExCommand::PrevBuffer,
        "b" | "buffer" => arg.parse::<usize>().map(ExCommand::GotoBuffer).unwrap_or_else(|_| ExCommand::Unknown(input.to_string())),
        "bd" | "bdel" | "bdelete" => ExCommand::DeleteBuffer { force, wipe: false },
        "bw" | "bwipe" | "bwipeout" => ExCommand::DeleteBuffer { force, wipe: true },
        "ls" | "buffers" | "files" => ExCommand::ListBuffers,
        "s" | "substitute" => parse_substitute(range, after),
        "g" | "global" => parse_global(after),
        "d" | "delete" => ExCommand::Delete { range },
        "noh" | "nohlsearch" => ExCommand::NoHighlight,
        "set" => parse_set(arg),
        "sp" | "split" => ExCommand::Split { vertical: false, file: opt_arg(arg), scratch: false },
        "vs" | "vsp" | "vsplit" => ExCommand::Split { vertical: true, file: opt_arg(arg), scratch: false },
        "new" => ExCommand::Split { vertical: false, file: None, scratch: true },
        "vnew" | "vne" => ExCommand::Split { vertical: true, file: None, scratch: true },
        "on" | "only" => ExCommand::Only,
        "clo" | "close" => ExCommand::Close,
        "term" | "terminal" => ExCommand::Terminal,
        "h" | "help" => ExCommand::Help { topic: opt_arg(arg) },
        _ => ExCommand::Unknown(input.to_string()),
    }
}

/// `None` for an empty argument, `Some(arg)` otherwise — the shared "an
/// optional file path" helper for the split commands.
fn opt_arg(arg: &str) -> Option<String> {
    if arg.is_empty() {
        None
    } else {
        Some(arg.to_string())
    }
}

fn parse_range(input: &str) -> (LineRange, &str) {
    if let Some(rest) = input.strip_prefix('%') {
        return (LineRange::All, rest);
    }
    if let Some((a, rest)) = parse_line_spec(input) {
        if let Some(rest2) = rest.strip_prefix(',')
            && let Some((b, rest3)) = parse_line_spec(rest2)
        {
            return (LineRange::Pair(a, b), rest3);
        }
        return (LineRange::Single(a), rest);
    }
    (LineRange::None, input)
}

fn parse_line_spec(input: &str) -> Option<(LineSpec, &str)> {
    if let Some(rest) = input.strip_prefix('.') {
        return Some((LineSpec::Current, rest));
    }
    if let Some(rest) = input.strip_prefix('$') {
        return Some((LineSpec::Last, rest));
    }
    let digits: String = input.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    let n: usize = digits.parse().ok()?;
    Some((LineSpec::Number(n), &input[digits.len()..]))
}

/// `after` is everything following `s`/`substitute` and an optional `!`,
/// e.g. `"/pat/rep/g"`. Escaping the delimiter (`\/`) is not supported — a
/// deliberate scope cut; see the crate-level report for the full list.
fn parse_substitute(range: LineRange, after: &str) -> ExCommand {
    let Some(rest) = after.strip_prefix('/') else {
        return ExCommand::Unknown(format!("s{after}"));
    };
    let parts: Vec<&str> = rest.split('/').collect();
    let pattern = parts.first().copied().unwrap_or("").to_string();
    let replacement = parts.get(1).copied().unwrap_or("").to_string();
    let flags = parts.get(2).copied().unwrap_or("");
    ExCommand::Substitute { range, pattern, replacement, global: flags.contains('g') }
}

fn parse_global(after: &str) -> ExCommand {
    let Some(rest) = after.strip_prefix('/') else {
        return ExCommand::Unknown(format!("g{after}"));
    };
    match rest.find('/') {
        Some(end) => ExCommand::Global { pattern: rest[..end].to_string(), cmd: rest[end + 1..].to_string() },
        None => ExCommand::Unknown(format!("g{after}")),
    }
}

fn parse_set(arg: &str) -> ExCommand {
    if let Some(eq) = arg.find('=') {
        ExCommand::Set { key: arg[..eq].to_string(), value: Some(arg[eq + 1..].to_string()) }
    } else if let Some(rest) = arg.strip_prefix("no") {
        ExCommand::Set { key: rest.to_string(), value: Some("false".to_string()) }
    } else {
        ExCommand::Set { key: arg.to_string(), value: None }
    }
}

/// `:{range}s/pattern/replacement/[g]`. Returns the number of substitutions
/// made, so the caller can echo `"3 substitutions on 2 lines"`-style
/// feedback the way real vim does.
pub fn substitute(buf: &mut Buffer, first: usize, last: usize, pattern: &str, replacement: &str, global: bool) -> crate::Result<usize> {
    if first > last {
        return Ok(0);
    }
    let re = Regex::new(pattern).map_err(|e| crate::Error::InvalidPattern { pattern: pattern.to_string(), reason: e.to_string() })?;
    let last = last.min(buf.line_count().saturating_sub(1));
    let mut total = 0usize;
    for line in first..=last {
        let Some(text) = buf.line(line) else { continue };
        let n_matches = re.find_iter(&text).count();
        if n_matches == 0 {
            continue;
        }
        let new_text = if global {
            total += n_matches;
            re.replace_all(&text, replacement).into_owned()
        } else {
            total += 1;
            re.replacen(&text, 1, replacement).into_owned()
        };
        if new_text == text {
            continue;
        }
        let range = Range::new(Position::new(line, 0), Position::new(line, buf.line_len(line)));
        buf.apply(Edit::replace(range, new_text))?;
    }
    Ok(total)
}

/// `:{range}d` / `:g/pat/d`: linewise delete without touching a register
/// (real vim's `:d` *does* write the unnamed/numbered registers; omitted
/// here since ex commands are not part of the brief's register
/// requirements, and adding it is a non-breaking follow-up).
pub fn delete_lines(buf: &mut Buffer, first: usize, last: usize) -> crate::Result<Position> {
    let range = operator::linewise_delete_range(buf, first, last);
    buf.apply(Edit::delete(range))
}

/// `:g/pattern/cmd`: runs `cmd` (only `d` and `s/.../.../ [g]` are
/// supported — see the module docs' scope note) on every line matching
/// `pattern`, using vim's own algorithm of collecting the matching lines
/// *before* running anything, so that a `d` sub-command shrinking the buffer
/// mid-pass cannot skip or double-hit a line.
pub fn global(buf: &mut Buffer, pattern: &str, cmd: &str) -> crate::Result<usize> {
    let re = Regex::new(pattern).map_err(|e| crate::Error::InvalidPattern { pattern: pattern.to_string(), reason: e.to_string() })?;
    let matching: Vec<usize> = (0..buf.line_count()).filter(|&l| buf.line(l).map(|t| re.is_match(&t)).unwrap_or(false)).collect();

    let cmd = cmd.trim();
    if cmd == "d" || cmd == "delete" {
        // Delete from the bottom up so earlier indices stay valid.
        for &line in matching.iter().rev() {
            delete_lines(buf, line, line)?;
        }
        return Ok(matching.len());
    }
    if let ExCommand::Substitute { pattern: p, replacement, global: g, .. } = parse(cmd) {
        let mut total = 0;
        for &line in &matching {
            total += substitute(buf, line, line, &p, &replacement, g)?;
        }
        return Ok(total);
    }
    Err(crate::Error::UnknownCommand(format!("g/{pattern}/{cmd}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_write_and_quit_variants() {
        assert_eq!(parse("w"), ExCommand::Write { path: None, then_quit: false, force: false });
        assert_eq!(parse("w out.txt"), ExCommand::Write { path: Some("out.txt".into()), then_quit: false, force: false });
        assert_eq!(parse("q"), ExCommand::Quit { force: false });
        assert_eq!(parse("q!"), ExCommand::Quit { force: true });
        assert_eq!(parse("wq"), ExCommand::Write { path: None, then_quit: true, force: false });
    }

    #[test]
    fn parses_quit_all_and_write_all_variants() {
        // Quit-all abbreviations, plain and forced.
        for name in ["qa", "qall", "quita", "quitall"] {
            assert_eq!(parse(name), ExCommand::QuitAll { force: false }, "parsing {name:?}");
            assert_eq!(parse(&format!("{name}!")), ExCommand::QuitAll { force: true }, "parsing {name}!");
        }
        // Write-all (no quit).
        for name in ["wa", "wall"] {
            assert_eq!(parse(name), ExCommand::WriteAll { then_quit: false, force: false }, "parsing {name:?}");
        }
        // Write-all-then-quit-all.
        for name in ["wqa", "wqall"] {
            assert_eq!(parse(name), ExCommand::WriteAll { then_quit: true, force: false }, "parsing {name:?}");
            assert_eq!(parse(&format!("{name}!")), ExCommand::WriteAll { then_quit: true, force: true }, "parsing {name}!");
        }
        // `:xa`/`:xall` is write-all-then-quit-all with force (mirrors `:x`).
        for name in ["xa", "xall"] {
            assert_eq!(parse(name), ExCommand::WriteAll { then_quit: true, force: true }, "parsing {name:?}");
        }
        // The single-window forms are untouched and still distinct.
        assert_eq!(parse("q"), ExCommand::Quit { force: false });
        assert_eq!(parse("wq"), ExCommand::Write { path: None, then_quit: true, force: false });
    }

    #[test]
    fn parses_buffer_management_family() {
        // `:bd` / `:bdelete`, plain and forced.
        assert_eq!(parse("bd"), ExCommand::DeleteBuffer { force: false, wipe: false });
        assert_eq!(parse("bdelete"), ExCommand::DeleteBuffer { force: false, wipe: false });
        assert_eq!(parse("bd!"), ExCommand::DeleteBuffer { force: true, wipe: false });
        assert_eq!(parse("bdelete!"), ExCommand::DeleteBuffer { force: true, wipe: false });
        // `:bw` / `:bwipeout` carry the wipe flag.
        assert_eq!(parse("bw"), ExCommand::DeleteBuffer { force: false, wipe: true });
        assert_eq!(parse("bwipeout"), ExCommand::DeleteBuffer { force: false, wipe: true });
        assert_eq!(parse("bw!"), ExCommand::DeleteBuffer { force: true, wipe: true });
        // The buffer-list commands.
        assert_eq!(parse("ls"), ExCommand::ListBuffers);
        assert_eq!(parse("buffers"), ExCommand::ListBuffers);
        // `:b{n}` (goto) stays distinct from the delete/list family.
        assert_eq!(parse("b2"), ExCommand::GotoBuffer(2));
    }

    #[test]
    fn parses_substitute_with_and_without_range() {
        assert_eq!(
            parse("s/foo/bar/g"),
            ExCommand::Substitute { range: LineRange::None, pattern: "foo".into(), replacement: "bar".into(), global: true }
        );
        assert_eq!(
            parse("%s/foo/bar/"),
            ExCommand::Substitute { range: LineRange::All, pattern: "foo".into(), replacement: "bar".into(), global: false }
        );
        assert_eq!(
            parse("2,4s/foo/bar/"),
            ExCommand::Substitute {
                range: LineRange::Pair(LineSpec::Number(2), LineSpec::Number(4)),
                pattern: "foo".into(),
                replacement: "bar".into(),
                global: false
            }
        );
    }

    #[test]
    fn parses_delete_range_and_bare_line_number() {
        assert_eq!(parse("2,4d"), ExCommand::Delete { range: LineRange::Pair(LineSpec::Number(2), LineSpec::Number(4)) });
        assert_eq!(parse("42"), ExCommand::GotoLine(LineSpec::Number(42)));
    }

    #[test]
    fn parses_global() {
        assert_eq!(parse("g/foo/d"), ExCommand::Global { pattern: "foo".into(), cmd: "d".into() });
    }

    #[test]
    fn parses_help_with_and_without_a_topic() {
        // Bare `:help` / `:h` open the manual at the top (no topic).
        assert_eq!(parse("help"), ExCommand::Help { topic: None });
        assert_eq!(parse("h"), ExCommand::Help { topic: None });
        // `:help <topic>` carries the topic through for the jump.
        assert_eq!(parse("help lsp"), ExCommand::Help { topic: Some("lsp".into()) });
        assert_eq!(parse("h windows"), ExCommand::Help { topic: Some("windows".into()) });
    }

    #[test]
    fn substitute_replaces_within_a_line_range() {
        let mut buf = Buffer::from_str("foo\nfoo\nfoo\n");
        let n = substitute(&mut buf, 0, 1, "foo", "bar", false).unwrap();
        assert_eq!(n, 2);
        assert_eq!(buf.text(), "bar\nbar\nfoo\n");
    }
}

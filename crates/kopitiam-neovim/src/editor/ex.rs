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

use super::command::{self, CommandId};
use super::operator;
use super::quickfix::ListKind;

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
    /// `:b {name}` — go to the buffer whose name matches `{name}`. vim lets
    /// `:b` take a (unique-substring) name as well as a number; kvim resolves
    /// the name in `Editor::execute_ex`, where the buffer table lives. This is
    /// what makes `:b`-name completion land on a command that actually runs.
    GotoBufferName(String),
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
    /// `:g/pat/cmd` and its inverse `:v`/`:g!`. When `invert` is set the
    /// sub-command runs on every line that does **not** match `pattern` — that
    /// is the only difference between `:g` and `:v`, so they share one variant.
    Global { pattern: String, cmd: String, invert: bool },
    /// `:{range}sort[!] [u][n]` — reorder the lines of `range` (the whole
    /// buffer when no range is given, matching vim). `reverse` is the `!`
    /// suffix, `unique` the `u` flag (drop duplicate lines), `numeric` the `n`
    /// flag (order by the first decimal number on each line instead of by
    /// text). See [`sort_lines`] for the exact ordering rules.
    Sort { range: LineRange, reverse: bool, unique: bool, numeric: bool },
    /// `:{range}m {dest}` (`copy == false`) and `:{range}t`/`:copy` /`:co`
    /// (`copy == true`): move or copy `range` to *after* the `dest` line. The
    /// destination is a bare line address — a number, `.` (current), `$`
    /// (last) or `0` (before the first line) — resolved in
    /// `Editor::execute_ex`, where the buffer's line count is known.
    MoveOrCopy { range: LineRange, dest: LineSpec, copy: bool },
    /// `:{range}>` / `:{range}<` — shift `range` right/left by `count`
    /// shiftwidths (`:>>` is `count == 2`, and so on). Reuses the exact same
    /// indent machinery as the normal-mode `>>`/`<<` operators.
    Shift { range: LineRange, right: bool, count: usize },
    /// `:{range}normal[!] {keys}` — feed `keys` (vim key notation, so `<Esc>`
    /// etc. work) through the editor's own key handler. With a range the keys
    /// run once per line (cursor parked at column 0 of each), matching vim;
    /// with no range they run once at the current cursor. `keys` is the raw
    /// remainder of the command line with exactly one separating space
    /// stripped, so leading whitespace in the keys is preserved the way vim
    /// preserves it.
    Normal { range: LineRange, keys: String },
    /// `:earlier {count}` (`redo == false`) / `:later {count}` (`redo == true`):
    /// step `count` states back through undo history, or forward through redo.
    /// `count` is `None` when the argument was a *time* form (`5m`, `1h`) that
    /// kvim does not support yet — the executor reports that rather than
    /// silently doing the wrong thing. See bead kopitiam-cj0.47.
    TimeTravel { count: Option<usize>, redo: bool },
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
    /// `:!{cmd}` — run a shell command and show its combined stdout+stderr in a
    /// scratch buffer. This is the *no-range* bang; `:{range}!{cmd}` parses to
    /// [`ExCommand::Filter`] instead, because a range turns `:!` from "run and
    /// show" into "filter these lines". `cmd` is the raw remainder after the
    /// `!`, kept verbatim so the shell sees exactly what was typed.
    ShellRun { cmd: String },
    /// `:{range}!{cmd}` (and the `!{motion}` operator's command line) — filter
    /// the range's lines through `cmd`: feed them as the command's stdin and
    /// replace them with its stdout. The classic `:%!sort`, `:'<,'>!column -t`.
    /// Applied as one buffer edit (hence one undo step). See
    /// [`super::Editor::execute_ex`] for the non-zero-exit safety rule.
    Filter { range: LineRange, cmd: String },
    /// `:r !{cmd}` / `:{line}r !{cmd}` (a.k.a. `:read !{cmd}`) — run `cmd` and
    /// insert its stdout into the buffer *below* the addressed line (the current
    /// line when no range is given), matching vim. `range`'s last line is the
    /// insertion point.
    ReadShell { range: LineRange, cmd: String },
    /// The quickfix / location-list family (`:grep`, `:copen`, `:cnext`, …).
    /// Every one of these is recognised by the editor but *performed* by the UI,
    /// which owns the search root, the list windows and the jumps — the editor
    /// returns it through `EditorResponse::Quickfix` the same way it hands back
    /// window commands. See [`QuickfixCommand`] and [`super::quickfix`].
    Quickfix(QuickfixCommand),
    /// An empty command line (`:` followed immediately by Enter).
    Empty,
    /// Parsed but not recognized — surfaced to the user as
    /// [`crate::Error::UnknownCommand`] rather than silently ignored.
    Unknown(String),
}

/// One parsed quickfix / location-list command. The `kind` on each variant is
/// [`ListKind::Quickfix`] for the `c`-prefixed forms and [`ListKind::Location`]
/// for the `l`-prefixed twins, so the UI executor picks the right list off a
/// single enum instead of duplicating every arm.
///
/// Parsing is all this layer does — see [`ExCommand::Quickfix`] for why the
/// doing lives in the UI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuickfixCommand {
    /// `:grep`/`:vimgrep` (and their `l` twins): search `pattern` across the
    /// project, optionally restricted to `globs`, into the list. See
    /// [`parse_grep`] for the accepted argument forms.
    Grep { kind: ListKind, pattern: String, globs: Vec<String> },
    /// `:copen`/`:lopen` — open the list window.
    Open(ListKind),
    /// `:cclose`/`:lclose` — close the list window.
    Close(ListKind),
    /// `:cwindow`/`:lwindow` — open the window iff the list is non-empty, else
    /// close it.
    Window(ListKind),
    /// `:cnext`/`:lnext`.
    Next(ListKind),
    /// `:cprev`/`:lprev`.
    Prev(ListKind),
    /// `:cfirst`/`:lfirst`.
    First(ListKind),
    /// `:clast`/`:llast`.
    Last(ListKind),
    /// `:cc [nr]`/`:ll [nr]` — go to entry `nr` (1-based), or the current entry
    /// when `nr` is `None`.
    Nth { kind: ListKind, nr: Option<usize> },
    /// `:cdo {cmd}`/`:ldo {cmd}` — run `cmd` on each entry's buffer.
    Do { kind: ListKind, cmd: String },
}

/// Parses the argument of a `:grep`/`:vimgrep` command into a pattern and an
/// optional glob list.
///
/// Two forms are accepted, so both the vim habit and the shell habit work:
///
/// * **Delimited** `/pattern/ globs...` — vim's `:vimgrep /re/ *.rs` style. The
///   pattern is taken verbatim between the first two `/`, so it may contain
///   spaces; everything after the closing `/` is split on whitespace into globs.
/// * **Bare** `pattern globs...` — the first whitespace-delimited token is the
///   pattern, the rest are globs. This is the quick `:grep TODO` form.
///
/// An empty argument yields `None`; the caller turns that into an error, since a
/// grep with no pattern has nothing to search for (kvim does not implement vim's
/// "reuse the last pattern" shorthand — a filed scope cut).
fn parse_grep(kind: ListKind, arg: &str) -> Option<QuickfixCommand> {
    let arg = arg.trim();
    if arg.is_empty() {
        return None;
    }
    let (pattern, rest) = if let Some(after) = arg.strip_prefix('/') {
        // Delimited: pattern is up to the next unescaped '/'.
        match after.find('/') {
            Some(end) => (after[..end].to_string(), after[end + 1..].trim()),
            // No closing delimiter: treat the whole remainder as the pattern.
            None => (after.to_string(), ""),
        }
    } else {
        match arg.split_once(char::is_whitespace) {
            Some((p, rest)) => (p.to_string(), rest.trim()),
            None => (arg.to_string(), ""),
        }
    };
    if pattern.is_empty() {
        return None;
    }
    let globs = rest.split_whitespace().map(str::to_string).collect();
    Some(QuickfixCommand::Grep { kind, pattern, globs })
}

/// `:cc [nr]`/`:ll [nr]`: an optional 1-based entry number. Empty (or
/// unparseable) means "the current entry", which the list model treats as a
/// re-select — matching vim's bare `:cc`.
fn parse_optional_nr(arg: &str) -> Option<usize> {
    arg.trim().parse::<usize>().ok()
}

/// `:cdo {cmd}`/`:ldo {cmd}`: everything after the command name is the ex
/// command to run on each entry. Exactly one separating space is stripped (so
/// the command keeps any further leading whitespace it wants); an empty command
/// yields `None`, which the caller reports as an error rather than iterating the
/// list to run nothing.
fn parse_list_do(kind: ListKind, after: &str) -> Option<QuickfixCommand> {
    let cmd = after.strip_prefix(' ').unwrap_or(after).trim();
    if cmd.is_empty() {
        return None;
    }
    Some(QuickfixCommand::Do { kind, cmd: cmd.to_string() })
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

    // `:>` and `:<` are the only ex commands whose *name* is not alphabetic,
    // so they cannot go through the registry's name lookup below — handle them
    // before the alphabetic name is even extracted. `:>>` (or `:<<<`) repeats
    // the shift, one shiftwidth per `>`/`<`.
    if let Some(shift) = parse_shift(range, rest) {
        return shift;
    }

    // The shell bang (`:!cmd`, `:{range}!cmd`) also has a non-alphabetic name,
    // so it too is caught before the alphabetic-name path. A leading `!` here is
    // unambiguous: the *force* `!` that other commands carry only ever appears
    // *after* an alphabetic name (`:w!`), never as the first character past the
    // range. Presence of a range is what splits the two meanings — vim's `:!cmd`
    // runs and shows output, while `:.!cmd`/`:%!cmd` filter the range in place.
    if let Some(cmd) = rest.strip_prefix('!') {
        return match range {
            LineRange::None => ExCommand::ShellRun { cmd: cmd.to_string() },
            _ => ExCommand::Filter { range, cmd: cmd.to_string() },
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

    // Resolve the name to a command *group* via the shared registry
    // (`super::command`), then do this command's own argument parsing. The
    // registry owns the alias list; this match owns the argument grammar. See
    // `command.rs`'s module docs for why the two were split.
    //
    // The quit-all / write-all family stays its own set of groups (not `q` with
    // a count) because "all windows" is a fundamentally different action from
    // "this window": `qa` exits the editor unconditionally across every split,
    // where `q` closes one.
    match command::lookup(&name).map(|spec| spec.id) {
        Some(CommandId::Write) => ExCommand::Write { path: opt_arg(arg), then_quit: false, force },
        Some(CommandId::WriteQuit) => ExCommand::Write { path: opt_arg(arg), then_quit: true, force },
        // `:x`/`:xit` differ from `:wq` in real vim only by skipping the write
        // when the buffer is unmodified. `Editor::execute_ex` checks
        // `is_modified` itself before honoring `then_quit`'s implied write, so
        // treating `x` as `wq` here is exact, not an approximation.
        Some(CommandId::Xit) => ExCommand::Write { path: None, then_quit: true, force: true },
        Some(CommandId::Quit) => ExCommand::Quit { force },
        Some(CommandId::QuitAll) => ExCommand::QuitAll { force },
        Some(CommandId::WriteAll) => ExCommand::WriteAll { then_quit: false, force },
        Some(CommandId::WriteQuitAll) => ExCommand::WriteAll { then_quit: true, force },
        // `:xa`/`:xall` is `:wqa` — write all, then quit all. (vim's `:x` skips
        // the write when a buffer is unmodified; the write-all executor already
        // writes only modified buffers, so `xa` and `wqa` coincide exactly.)
        Some(CommandId::XitAll) => ExCommand::WriteAll { then_quit: true, force: true },
        Some(CommandId::Edit) => ExCommand::Edit { path: arg.to_string() },
        Some(CommandId::NextBuffer) => ExCommand::NextBuffer,
        Some(CommandId::PrevBuffer) => ExCommand::PrevBuffer,
        Some(CommandId::Buffer) => {
            if arg.is_empty() {
                ExCommand::Unknown(input.to_string())
            } else if let Ok(n) = arg.parse::<usize>() {
                ExCommand::GotoBuffer(n)
            } else {
                ExCommand::GotoBufferName(arg.to_string())
            }
        }
        Some(CommandId::DeleteBuffer) => ExCommand::DeleteBuffer { force, wipe: false },
        Some(CommandId::WipeBuffer) => ExCommand::DeleteBuffer { force, wipe: true },
        Some(CommandId::ListBuffers) => ExCommand::ListBuffers,
        Some(CommandId::Substitute) => parse_substitute(range, after),
        // `:g` (`invert == force`, so `:g!` inverts) and `:v`/`:vglobal`
        // (always inverted) share `parse_global`; only the starting `invert`
        // flag differs.
        Some(CommandId::Global) => parse_global(after, force),
        Some(CommandId::VGlobal) => parse_global(after, true),
        Some(CommandId::Sort) => parse_sort(range, arg, force),
        Some(CommandId::Move) => parse_move_or_copy(range, arg, false),
        Some(CommandId::Copy) => parse_move_or_copy(range, arg, true),
        // `:normal` takes its keys from `after` (post-`!`), not the trimmed
        // `arg`, because a space is significant in a key sequence.
        Some(CommandId::Normal) => parse_normal(range, after),
        Some(CommandId::Earlier) => ExCommand::TimeTravel { count: parse_count_arg(arg), redo: false },
        Some(CommandId::Later) => ExCommand::TimeTravel { count: parse_count_arg(arg), redo: true },
        Some(CommandId::Delete) => ExCommand::Delete { range },
        Some(CommandId::NoHighlight) => ExCommand::NoHighlight,
        Some(CommandId::Set) => parse_set(arg),
        Some(CommandId::Split) => ExCommand::Split { vertical: false, file: opt_arg(arg), scratch: false },
        Some(CommandId::VSplit) => ExCommand::Split { vertical: true, file: opt_arg(arg), scratch: false },
        Some(CommandId::New) => ExCommand::Split { vertical: false, file: None, scratch: true },
        Some(CommandId::VNew) => ExCommand::Split { vertical: true, file: None, scratch: true },
        Some(CommandId::Only) => ExCommand::Only,
        Some(CommandId::Close) => ExCommand::Close,
        Some(CommandId::Terminal) => ExCommand::Terminal,
        // `:r`/`:read` currently supports only the shell form `:r !{cmd}` (and
        // its no-space `:r!{cmd}`). `force` is `true` when the `!` sat directly
        // against the name (`:r!cmd`), in which case `after` is already the
        // command; otherwise the `!` (if any) is still the first non-space
        // character of `arg`. The plain `:r {file}` form is a filed follow-up.
        Some(CommandId::Read) => parse_read(range, force, after),
        Some(CommandId::Help) => ExCommand::Help { topic: opt_arg(arg) },

        // Quickfix (global) and location (`l`-prefixed) list commands. The two
        // families share `QuickfixCommand`; only the `ListKind` differs, so the
        // arms come in `c`/`l` pairs pointing at the same variant. `:grep` takes
        // a pattern (+globs) that needs its own parse; a missing pattern is an
        // error, not a silent no-op. `:cc`/`:ll` take an optional entry number;
        // `:cdo`/`:ldo` take a whole command line as their argument.
        Some(CommandId::Grep) => parse_grep(ListKind::Quickfix, arg).map(ExCommand::Quickfix).unwrap_or_else(|| ExCommand::Unknown(input.to_string())),
        Some(CommandId::VimGrep) => parse_grep(ListKind::Quickfix, arg).map(ExCommand::Quickfix).unwrap_or_else(|| ExCommand::Unknown(input.to_string())),
        Some(CommandId::LGrep) => parse_grep(ListKind::Location, arg).map(ExCommand::Quickfix).unwrap_or_else(|| ExCommand::Unknown(input.to_string())),
        Some(CommandId::LVimGrep) => parse_grep(ListKind::Location, arg).map(ExCommand::Quickfix).unwrap_or_else(|| ExCommand::Unknown(input.to_string())),
        Some(CommandId::Copen) => ExCommand::Quickfix(QuickfixCommand::Open(ListKind::Quickfix)),
        Some(CommandId::Lopen) => ExCommand::Quickfix(QuickfixCommand::Open(ListKind::Location)),
        Some(CommandId::Cclose) => ExCommand::Quickfix(QuickfixCommand::Close(ListKind::Quickfix)),
        Some(CommandId::Lclose) => ExCommand::Quickfix(QuickfixCommand::Close(ListKind::Location)),
        Some(CommandId::Cwindow) => ExCommand::Quickfix(QuickfixCommand::Window(ListKind::Quickfix)),
        Some(CommandId::Lwindow) => ExCommand::Quickfix(QuickfixCommand::Window(ListKind::Location)),
        Some(CommandId::Cnext) => ExCommand::Quickfix(QuickfixCommand::Next(ListKind::Quickfix)),
        Some(CommandId::Lnext) => ExCommand::Quickfix(QuickfixCommand::Next(ListKind::Location)),
        Some(CommandId::Cprev) => ExCommand::Quickfix(QuickfixCommand::Prev(ListKind::Quickfix)),
        Some(CommandId::Lprev) => ExCommand::Quickfix(QuickfixCommand::Prev(ListKind::Location)),
        Some(CommandId::Cfirst) => ExCommand::Quickfix(QuickfixCommand::First(ListKind::Quickfix)),
        Some(CommandId::Lfirst) => ExCommand::Quickfix(QuickfixCommand::First(ListKind::Location)),
        Some(CommandId::Clast) => ExCommand::Quickfix(QuickfixCommand::Last(ListKind::Quickfix)),
        Some(CommandId::Llast) => ExCommand::Quickfix(QuickfixCommand::Last(ListKind::Location)),
        Some(CommandId::CC) => ExCommand::Quickfix(QuickfixCommand::Nth { kind: ListKind::Quickfix, nr: parse_optional_nr(arg) }),
        Some(CommandId::LL) => ExCommand::Quickfix(QuickfixCommand::Nth { kind: ListKind::Location, nr: parse_optional_nr(arg) }),
        Some(CommandId::Cdo) => parse_list_do(ListKind::Quickfix, after).map(ExCommand::Quickfix).unwrap_or_else(|| ExCommand::Unknown(input.to_string())),
        Some(CommandId::Ldo) => parse_list_do(ListKind::Location, after).map(ExCommand::Quickfix).unwrap_or_else(|| ExCommand::Unknown(input.to_string())),

        None => ExCommand::Unknown(input.to_string()),
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

fn parse_global(after: &str, invert: bool) -> ExCommand {
    let Some(rest) = after.strip_prefix('/') else {
        return ExCommand::Unknown(format!("g{after}"));
    };
    match rest.find('/') {
        Some(end) => ExCommand::Global { pattern: rest[..end].to_string(), cmd: rest[end + 1..].to_string(), invert },
        None => ExCommand::Unknown(format!("g{after}")),
    }
}

/// `:sort` flags follow the command as a run of letters (`:sort un`, `:sort n`).
/// Only the flags kvim implements are recognized — `u` (unique) and `n`
/// (numeric); `!` is handled separately as `reverse`. Unknown flag letters are
/// ignored rather than erroring, which keeps `:sort` forgiving the way vim is.
fn parse_sort(range: LineRange, arg: &str, reverse: bool) -> ExCommand {
    ExCommand::Sort { range, reverse, unique: arg.contains('u'), numeric: arg.contains('n') }
}

/// `:m`/`:t` take a single trailing line address (`0`, `.`, `$`, or a number).
/// A missing or unparseable address is an error rather than a silent default,
/// since moving lines to an unknown place is never what was meant.
fn parse_move_or_copy(range: LineRange, arg: &str, copy: bool) -> ExCommand {
    match parse_line_spec(arg.trim()) {
        Some((dest, rest)) if rest.trim().is_empty() => ExCommand::MoveOrCopy { range, dest, copy },
        _ => ExCommand::Unknown(format!("{}{arg}", if copy { "t" } else { "m" })),
    }
}

/// `:{range}normal[!] {keys}`: everything after the command name (and its
/// optional `!`) is the literal key sequence, with exactly one separating
/// space removed. An empty sequence is kept as an empty [`ExCommand::Normal`]
/// so the executor can no-op it cleanly.
fn parse_normal(range: LineRange, after: &str) -> ExCommand {
    let keys = after.strip_prefix(' ').unwrap_or(after);
    ExCommand::Normal { range, keys: keys.to_string() }
}

/// `:earlier`/`:later` take an optional count. Empty means `1` (one step).
/// A pure number is that many undo/redo steps. Anything else — a vim *time*
/// form such as `5m` or `1h` — returns `None`, signalling the executor to
/// report that kvim does not do time-based travel yet (bead kopitiam-cj0.47).
fn parse_count_arg(arg: &str) -> Option<usize> {
    let arg = arg.trim();
    if arg.is_empty() {
        return Some(1);
    }
    arg.parse::<usize>().ok()
}

/// Detects the `:>` / `:<` shift commands, which the alphabetic-name path
/// cannot see. `rest` is the command line past its range prefix. Returns the
/// parsed [`ExCommand::Shift`] when it starts with `>` or `<`, counting the
/// run of that character as the shift multiplier (`>>` shifts twice).
fn parse_shift(range: LineRange, rest: &str) -> Option<ExCommand> {
    let first = rest.chars().next()?;
    let right = match first {
        '>' => true,
        '<' => false,
        _ => return None,
    };
    let count = rest.chars().take_while(|&c| c == first).count();
    Some(ExCommand::Shift { range, right, count })
}

/// `:r !{cmd}` / `:{line}r !{cmd}` (`:read` too). Two spellings reach the same
/// [`ExCommand::ReadShell`]:
///
/// * `:r!cmd` — the `!` abuts the name, so the generic parser already peeled it
///   off as `force == true` and `after` is the command itself.
/// * `:r !cmd` — a space separates them, so the `!` is still the leading
///   character of `after` and is stripped here.
///
/// Anything else (`:r file`, bare `:r`) is not yet supported — vim's file-read
/// form is a filed follow-up — so it returns [`ExCommand::Unknown`] rather than
/// guessing.
fn parse_read(range: LineRange, force: bool, after: &str) -> ExCommand {
    let cmd = if force {
        // `:r!cmd`: the `!` was consumed as force; the rest is the command.
        after.trim_start()
    } else {
        // `:r !cmd`: strip the leading `!` (after trimming the separating space).
        match after.trim_start().strip_prefix('!') {
            Some(rest) => rest.trim_start(),
            None => return ExCommand::Unknown(format!("r{after}")),
        }
    };
    ExCommand::ReadShell { range, cmd: cmd.to_string() }
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

/// `:g/pattern/cmd` (and, with `invert`, `:v`/`:g!`): runs `cmd` (only `d` and
/// `s/.../.../ [g]` are supported — see the module docs' scope note) on every
/// line matching `pattern` — or, when `invert` is set, every line *not*
/// matching — using vim's own algorithm of collecting the target lines
/// *before* running anything, so that a `d` sub-command shrinking the buffer
/// mid-pass cannot skip or double-hit a line.
pub fn global(buf: &mut Buffer, pattern: &str, cmd: &str, invert: bool) -> crate::Result<usize> {
    let matching = global_matches(buf, pattern, invert)?;

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

/// The 0-based indices of the lines a `:g`/`:v` acts on: those matching
/// `pattern` (or, with `invert`, those *not* matching), collected up front the
/// way vim does so a sub-command that resizes the buffer can't skip or
/// double-hit a line. Scans content lines only (see [`content_line_count`]).
///
/// Split out so richer sub-commands than `:g` can run over the same set — in
/// particular `:g/pat/normal ...`, which the editor drives itself because
/// running normal-mode keys needs the whole `Editor`, not just a `Buffer`.
pub fn global_matches(buf: &Buffer, pattern: &str, invert: bool) -> crate::Result<Vec<usize>> {
    let re = Regex::new(pattern).map_err(|e| crate::Error::InvalidPattern { pattern: pattern.to_string(), reason: e.to_string() })?;
    Ok((0..content_line_count(buf))
        .filter(|&l| {
            let hit = buf.line(l).map(|t| re.is_match(&t)).unwrap_or(false);
            hit != invert
        })
        .collect())
}

/// The number of *content* lines in `buf`: its raw line count minus the one
/// phantom empty line that a trailing newline produces (see
/// [`Buffer::line_count`]'s docs). The line-reordering commands (`:sort`,
/// `:m`, `:t`, `:>`) work in these terms so they never sweep that phantom
/// trailing line into a sort or shove it around — which would corrupt the
/// buffer's trailing-newline state. vim has no such phantom (it tracks
/// end-of-line separately), so this is how kvim matches vim's line arithmetic.
pub(crate) fn content_line_count(buf: &Buffer) -> usize {
    let n = buf.line_count();
    // A rope ending in `\n` reports one extra, always-empty line. `text()`
    // ending in `\n` is the precise test; an empty buffer ("") does not, so
    // it correctly keeps its single line.
    if buf.text().ends_with('\n') { n - 1 } else { n }
}

/// The first decimal integer appearing in `s` (with an optional leading `-`),
/// or `0` if the line has no number — the sort key for `:sort n`. Matching
/// vim, a line without a number sorts as if it were `0`.
fn first_number(s: &str) -> i64 {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let start = if i > 0 && bytes[i - 1] == b'-' { i - 1 } else { i };
            let mut j = i;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            return s[start..j].parse().unwrap_or(0);
        }
        i += 1;
    }
    0
}

/// `:{range}sort[!] [u][n]`: reorders lines `first..=last` in place.
///
/// * `numeric` sorts by [`first_number`] (a *stable* sort, so lines sharing a
///   number keep their relative order, as vim does);
/// * otherwise the sort is plain lexical over the whole line;
/// * `reverse` flips the final order (`:sort!`);
/// * `unique` drops runs of identical lines *after* sorting (`:sort u`),
///   comparing the whole line text.
///
/// The rewrite replaces exactly the content span of the range, so the buffer's
/// trailing-newline state is untouched (see [`content_line_count`]).
pub fn sort_lines(buf: &mut Buffer, first: usize, last: usize, reverse: bool, unique: bool, numeric: bool) -> crate::Result<()> {
    let content = content_line_count(buf);
    if content == 0 {
        return Ok(());
    }
    let last = last.min(content - 1);
    let first = first.min(last);
    let mut lines: Vec<String> = (first..=last).filter_map(|l| buf.line(l)).collect();
    if numeric {
        lines.sort_by_key(|l| first_number(l));
    } else {
        lines.sort();
    }
    if reverse {
        lines.reverse();
    }
    if unique {
        lines.dedup();
    }
    let joined = lines.join("\n");
    let range = operator::linewise_content_range(buf, first, last);
    buf.apply(Edit::replace(range, joined))?;
    Ok(())
}

/// `:{range}m {dest}` (`copy == false`) and `:{range}t {dest}` (`copy == true`).
///
/// `dest_after` is the 0-based index of the line the range should land *after*,
/// with `-1` meaning "before the first line" (vim's address `0`). Returns the
/// 0-based line the cursor should end on (the last line of the moved/copied
/// text), the way vim leaves the cursor.
///
/// The whole thing is computed on a `Vec<String>` of the buffer's content
/// lines and written back as one edit. Working on the vector sidesteps the
/// index-shift bug that plagues the naive "delete here, insert there" approach
/// when the destination sits below the range: after the delete the destination
/// has moved, and getting that adjustment wrong is the classic `:m` off-by-one.
pub fn move_or_copy_lines(buf: &mut Buffer, first: usize, last: usize, dest_after: isize, copy: bool) -> crate::Result<usize> {
    let content = content_line_count(buf);
    if content == 0 {
        return Ok(0);
    }
    let last = last.min(content - 1);
    let first = first.min(last);
    let dest = dest_after.clamp(-1, content as isize - 1);

    // Moving a range to just before it, or into itself, is a no-op (vim
    // errors; a no-op is the safe, non-destructive equivalent).
    if !copy && dest >= first as isize - 1 && dest <= last as isize {
        return Ok(last);
    }

    let mut lines: Vec<String> = (0..content).filter_map(|l| buf.line(l)).collect();
    let block: Vec<String> = lines[first..=last].to_vec();
    let block_len = block.len();

    let cursor_line = if copy {
        let insert_at = (dest + 1) as usize;
        for (k, line) in block.into_iter().enumerate() {
            lines.insert(insert_at + k, line);
        }
        insert_at + block_len - 1
    } else {
        lines.drain(first..=last);
        // Below the range the destination shifted up by the block's length
        // once those lines were removed; above the range it did not.
        let insert_at = if dest < first as isize { (dest + 1) as usize } else { (dest + 1) as usize - block_len };
        for (k, line) in block.into_iter().enumerate() {
            lines.insert(insert_at + k, line);
        }
        insert_at + block_len - 1
    };

    let joined = lines.join("\n");
    let range = operator::linewise_content_range(buf, 0, content - 1);
    buf.apply(Edit::replace(range, joined))?;
    Ok(cursor_line)
}

/// Resolves a `:m`/`:t` destination address to the 0-based index of the line
/// to insert *after*, with `-1` meaning "before the first line" (vim's `0`).
pub(crate) fn resolve_dest(spec: LineSpec, current_line: usize, content: usize) -> isize {
    match spec {
        // `:m3` lands after 1-based line 3, i.e. 0-based index 2; `:m0` -> -1.
        LineSpec::Number(n) => n as isize - 1,
        LineSpec::Current => current_line as isize,
        LineSpec::Last => content as isize - 1,
    }
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
        assert_eq!(parse("g/foo/d"), ExCommand::Global { pattern: "foo".into(), cmd: "d".into(), invert: false });
    }

    #[test]
    fn parses_inverse_global() {
        // `:v` and `:g!` both invert; `:g` alone does not.
        assert_eq!(parse("v/foo/d"), ExCommand::Global { pattern: "foo".into(), cmd: "d".into(), invert: true });
        assert_eq!(parse("vglobal/foo/d"), ExCommand::Global { pattern: "foo".into(), cmd: "d".into(), invert: true });
        assert_eq!(parse("g!/foo/d"), ExCommand::Global { pattern: "foo".into(), cmd: "d".into(), invert: true });
    }

    #[test]
    fn parses_sort_variants() {
        assert_eq!(parse("sort"), ExCommand::Sort { range: LineRange::None, reverse: false, unique: false, numeric: false });
        assert_eq!(parse("sort!"), ExCommand::Sort { range: LineRange::None, reverse: true, unique: false, numeric: false });
        assert_eq!(parse("sort u"), ExCommand::Sort { range: LineRange::None, reverse: false, unique: true, numeric: false });
        assert_eq!(parse("sort n"), ExCommand::Sort { range: LineRange::None, reverse: false, unique: false, numeric: true });
        assert_eq!(
            parse("%sort! un"),
            ExCommand::Sort { range: LineRange::All, reverse: true, unique: true, numeric: true }
        );
    }

    #[test]
    fn parses_move_and_copy() {
        assert_eq!(parse("m0"), ExCommand::MoveOrCopy { range: LineRange::None, dest: LineSpec::Number(0), copy: false });
        assert_eq!(
            parse("2,3m0"),
            ExCommand::MoveOrCopy { range: LineRange::Pair(LineSpec::Number(2), LineSpec::Number(3)), dest: LineSpec::Number(0), copy: false }
        );
        assert_eq!(parse("t$"), ExCommand::MoveOrCopy { range: LineRange::None, dest: LineSpec::Last, copy: true });
        assert_eq!(parse("copy ."), ExCommand::MoveOrCopy { range: LineRange::None, dest: LineSpec::Current, copy: true });
        // A missing destination is an error, not a silent default.
        assert!(matches!(parse("m"), ExCommand::Unknown(_)));
    }

    #[test]
    fn parses_shift() {
        assert_eq!(parse(">"), ExCommand::Shift { range: LineRange::None, right: true, count: 1 });
        assert_eq!(parse("<<"), ExCommand::Shift { range: LineRange::None, right: false, count: 2 });
        assert_eq!(
            parse("1,4>>"),
            ExCommand::Shift { range: LineRange::Pair(LineSpec::Number(1), LineSpec::Number(4)), right: true, count: 2 }
        );
    }

    #[test]
    fn parses_normal_preserving_keys() {
        assert_eq!(parse("normal dw"), ExCommand::Normal { range: LineRange::None, keys: "dw".into() });
        assert_eq!(parse("norm! Ihi"), ExCommand::Normal { range: LineRange::None, keys: "Ihi".into() });
        assert_eq!(
            parse("2,4normal A;"),
            ExCommand::Normal { range: LineRange::Pair(LineSpec::Number(2), LineSpec::Number(4)), keys: "A;".into() }
        );
    }

    #[test]
    fn parses_earlier_and_later() {
        assert_eq!(parse("earlier"), ExCommand::TimeTravel { count: Some(1), redo: false });
        assert_eq!(parse("earlier 3"), ExCommand::TimeTravel { count: Some(3), redo: false });
        assert_eq!(parse("later 2"), ExCommand::TimeTravel { count: Some(2), redo: true });
        // A time form (5m) is not supported yet — flagged as `None`.
        assert_eq!(parse("earlier 5m"), ExCommand::TimeTravel { count: None, redo: false });
    }

    #[test]
    fn sort_orders_lines_plain_reverse_unique_numeric() {
        let mut buf = Buffer::from_str("banana\napple\ncherry\n");
        sort_lines(&mut buf, 0, 2, false, false, false).unwrap();
        assert_eq!(buf.text(), "apple\nbanana\ncherry\n");

        let mut buf = Buffer::from_str("banana\napple\ncherry\n");
        sort_lines(&mut buf, 0, 2, true, false, false).unwrap();
        assert_eq!(buf.text(), "cherry\nbanana\napple\n");

        let mut buf = Buffer::from_str("b\na\nb\na\n");
        sort_lines(&mut buf, 0, 3, false, true, false).unwrap();
        assert_eq!(buf.text(), "a\nb\n");

        // Numeric: "10" must sort after "9", not before it (lexical would flip).
        let mut buf = Buffer::from_str("item 10\nitem 9\nitem 100\n");
        sort_lines(&mut buf, 0, 2, false, false, true).unwrap();
        assert_eq!(buf.text(), "item 9\nitem 10\nitem 100\n");
    }

    #[test]
    fn move_relocates_lines_to_after_destination() {
        // :2,3m0 -> the two lines jump to the top.
        let mut buf = Buffer::from_str("a\nb\nc\nd\ne\n");
        let cursor = move_or_copy_lines(&mut buf, 1, 2, -1, false).unwrap();
        assert_eq!(buf.text(), "b\nc\na\nd\ne\n");
        assert_eq!(cursor, 1);

        // :1,2m$ -> the two lines jump to the bottom.
        let mut buf = Buffer::from_str("a\nb\nc\nd\ne\n");
        move_or_copy_lines(&mut buf, 0, 1, 4, false).unwrap();
        assert_eq!(buf.text(), "c\nd\ne\na\nb\n");
    }

    #[test]
    fn copy_duplicates_lines_after_destination() {
        // :1t$ -> a copy of line 1 lands at the bottom.
        let mut buf = Buffer::from_str("a\nb\nc\n");
        let cursor = move_or_copy_lines(&mut buf, 0, 0, 2, true).unwrap();
        assert_eq!(buf.text(), "a\nb\nc\na\n");
        assert_eq!(cursor, 3);

        // :2t0 -> a copy of line 2 lands at the top; originals untouched.
        let mut buf = Buffer::from_str("a\nb\nc\n");
        move_or_copy_lines(&mut buf, 1, 1, -1, true).unwrap();
        assert_eq!(buf.text(), "b\na\nb\nc\n");
    }

    #[test]
    fn inverse_global_deletes_non_matching_lines() {
        // :v/keep/d removes every line that does NOT contain "keep".
        let mut buf = Buffer::from_str("keep me\ndrop me\nkeep this\ntoss\n");
        let n = global(&mut buf, "keep", "d", true).unwrap();
        assert_eq!(n, 2);
        assert_eq!(buf.text(), "keep me\nkeep this\n");
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
    fn parses_grep_bare_and_delimited_forms() {
        // Bare: first token is the pattern, the rest are globs.
        assert_eq!(
            parse("grep TODO"),
            ExCommand::Quickfix(QuickfixCommand::Grep { kind: ListKind::Quickfix, pattern: "TODO".into(), globs: vec![] })
        );
        assert_eq!(
            parse("grep TODO *.rs src/"),
            ExCommand::Quickfix(QuickfixCommand::Grep {
                kind: ListKind::Quickfix,
                pattern: "TODO".into(),
                globs: vec!["*.rs".into(), "src/".into()],
            })
        );
        // Delimited: the pattern between the slashes may contain spaces.
        assert_eq!(
            parse("vimgrep /foo bar/ *.rs"),
            ExCommand::Quickfix(QuickfixCommand::Grep {
                kind: ListKind::Quickfix,
                pattern: "foo bar".into(),
                globs: vec!["*.rs".into()],
            })
        );
        // The `l`-twin targets the location list.
        assert_eq!(
            parse("lgrep TODO"),
            ExCommand::Quickfix(QuickfixCommand::Grep { kind: ListKind::Location, pattern: "TODO".into(), globs: vec![] })
        );
        // No pattern is an error, not a silent empty search.
        assert!(matches!(parse("grep"), ExCommand::Unknown(_)));
        assert!(matches!(parse("grep   "), ExCommand::Unknown(_)));
    }

    #[test]
    fn parses_quickfix_navigation_and_windows() {
        use QuickfixCommand::*;
        assert_eq!(parse("copen"), ExCommand::Quickfix(Open(ListKind::Quickfix)));
        assert_eq!(parse("cclose"), ExCommand::Quickfix(Close(ListKind::Quickfix)));
        assert_eq!(parse("cwindow"), ExCommand::Quickfix(Window(ListKind::Quickfix)));
        assert_eq!(parse("cn"), ExCommand::Quickfix(Next(ListKind::Quickfix)));
        assert_eq!(parse("cnext"), ExCommand::Quickfix(Next(ListKind::Quickfix)));
        assert_eq!(parse("cp"), ExCommand::Quickfix(Prev(ListKind::Quickfix)));
        assert_eq!(parse("cfirst"), ExCommand::Quickfix(First(ListKind::Quickfix)));
        assert_eq!(parse("clast"), ExCommand::Quickfix(Last(ListKind::Quickfix)));
        assert_eq!(parse("cc"), ExCommand::Quickfix(Nth { kind: ListKind::Quickfix, nr: None }));
        assert_eq!(parse("cc 3"), ExCommand::Quickfix(Nth { kind: ListKind::Quickfix, nr: Some(3) }));
        // Location twins.
        assert_eq!(parse("lopen"), ExCommand::Quickfix(Open(ListKind::Location)));
        assert_eq!(parse("lnext"), ExCommand::Quickfix(Next(ListKind::Location)));
        assert_eq!(parse("ll 2"), ExCommand::Quickfix(Nth { kind: ListKind::Location, nr: Some(2) }));
    }

    #[test]
    fn parses_cdo_keeping_the_whole_command() {
        assert_eq!(
            parse("cdo s/foo/bar/g"),
            ExCommand::Quickfix(QuickfixCommand::Do { kind: ListKind::Quickfix, cmd: "s/foo/bar/g".into() })
        );
        assert_eq!(
            parse("ldo normal A;"),
            ExCommand::Quickfix(QuickfixCommand::Do { kind: ListKind::Location, cmd: "normal A;".into() })
        );
        // An empty command is an error.
        assert!(matches!(parse("cdo"), ExCommand::Unknown(_)));
    }

    #[test]
    fn parses_shell_bang_forms() {
        // `:!cmd` with no range: run and show.
        assert_eq!(parse("!ls -l"), ExCommand::ShellRun { cmd: "ls -l".into() });
        // A range turns `:!` into a filter.
        assert_eq!(parse("%!sort"), ExCommand::Filter { range: LineRange::All, cmd: "sort".into() });
        assert_eq!(
            parse("2,4!column -t"),
            ExCommand::Filter { range: LineRange::Pair(LineSpec::Number(2), LineSpec::Number(4)), cmd: "column -t".into() }
        );
        assert_eq!(parse(".!rev"), ExCommand::Filter { range: LineRange::Single(LineSpec::Current), cmd: "rev".into() });
        // A leading `!` must never be mistaken for the force flag other commands
        // carry (that only appears after an alphabetic name, e.g. `:w!`).
        assert_eq!(parse("w!"), ExCommand::Write { path: None, then_quit: false, force: true });
    }

    #[test]
    fn parses_read_shell_forms() {
        // `:r !cmd`, `:r!cmd`, and the long `:read !cmd` all reach ReadShell.
        assert_eq!(parse("r !echo hi"), ExCommand::ReadShell { range: LineRange::None, cmd: "echo hi".into() });
        assert_eq!(parse("r!echo hi"), ExCommand::ReadShell { range: LineRange::None, cmd: "echo hi".into() });
        assert_eq!(parse("read !date"), ExCommand::ReadShell { range: LineRange::None, cmd: "date".into() });
        // A line address rides along for the insertion point.
        assert_eq!(
            parse("3r !echo hi"),
            ExCommand::ReadShell { range: LineRange::Single(LineSpec::Number(3)), cmd: "echo hi".into() }
        );
        // The plain file-read form is not supported yet — an honest Unknown,
        // not a silent guess.
        assert!(matches!(parse("r somefile.txt"), ExCommand::Unknown(_)));
    }

    #[test]
    fn substitute_replaces_within_a_line_range() {
        let mut buf = Buffer::from_str("foo\nfoo\nfoo\n");
        let n = substitute(&mut buf, 0, 1, "foo", "bar", false).unwrap();
        assert_eq!(n, 2);
        assert_eq!(buf.text(), "bar\nbar\nfoo\n");
    }
}

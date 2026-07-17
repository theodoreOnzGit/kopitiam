//! The typed ex-command registry: one table that lists every `:` command,
//! its aliases, what kind of argument it takes, and a one-line help blurb.
//!
//! # Why a registry, not just the `match` in [`super::ex::parse`]
//!
//! Before this, the only place that knew "these are the commands that exist"
//! was the hand-written `match` inside [`super::ex::parse`]. That match can
//! *dispatch* a name, but it cannot *enumerate* names — so nothing could offer
//! `:`-completion, a command palette, or per-command help. AID-0019 (clean-room
//! study of Helix, no code copied — Helix is MPL-2.0) recommends kvim grow a
//! first-class command table that all three surfaces read from.
//!
//! This is that table. It stays the single source of truth for the command
//! *vocabulary* (names + aliases); [`super::ex::parse`] no longer carries its
//! own copy of the alias list — it looks a name up here, gets back a
//! [`CommandId`], and only then does its command-specific argument parsing.
//! Add a command in one place (here) and both dispatch and completion learn
//! about it at once.
//!
//! We keep KOPITIAM vim-modeled: the command *names* and *keys* are vim's. Only
//! the registry *shape* is the idea borrowed from Helix.

/// What sort of argument a command takes, so the completer knows what to offer
/// after the command name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgKind {
    /// Takes no argument worth completing (`:q`, `:noh`, `:only`, ...).
    None,
    /// Takes a filesystem path (`:e`, `:w`, `:sp <file>`) — complete against
    /// directory entries.
    File,
    /// Takes a buffer name or number (`:b`) — complete against open buffers.
    Buffer,
}

/// A stable identifier for one command *group*. Every alias of a command maps
/// to the same `CommandId`; [`super::ex::parse`] matches on this instead of on
/// the raw name string, which is what lets the alias list live only in
/// [`COMMANDS`] and nowhere else.
///
/// The groups line up one-to-one with the arms of the old `parse` match, so the
/// argument parsing on the far side (ranges, `:s///` delimiters, `:set key=val`)
/// stays byte-for-byte identical — only the name→group step moved here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandId {
    /// `:w`/`:write` — write, stay open.
    Write,
    /// `:wq` — write, then quit.
    WriteQuit,
    /// `:x`/`:xit` — like `:wq` but skips the write when unmodified.
    Xit,
    /// `:q`/`:quit`.
    Quit,
    /// `:qa`/`:qall`/`:quita`/`:quitall`.
    QuitAll,
    /// `:wa`/`:wall` — write every buffer, stay open.
    WriteAll,
    /// `:wqa`/`:wqall` — write every buffer, then quit all.
    WriteQuitAll,
    /// `:xa`/`:xall` — `:wqa` with force (mirrors `:x`).
    XitAll,
    /// `:e`/`:edit`.
    Edit,
    /// `:bn`/`:bnext`.
    NextBuffer,
    /// `:bp`/`:bprev`/`:bprevious`.
    PrevBuffer,
    /// `:b`/`:buffer` — goto buffer by number.
    Buffer,
    /// `:bd`/`:bdel`/`:bdelete`.
    DeleteBuffer,
    /// `:bw`/`:bwipe`/`:bwipeout`.
    WipeBuffer,
    /// `:ls`/`:buffers`/`:files`.
    ListBuffers,
    /// `:s`/`:substitute`.
    Substitute,
    /// `:g`/`:global`.
    Global,
    /// `:v`/`:vglobal` — inverse `:g` (act on lines *not* matching). Same as
    /// `:g!`; kept as its own group because `v` is a distinct name vim ships.
    VGlobal,
    /// `:sort`/`:sor` — sort the lines in the range (whole buffer if none).
    Sort,
    /// `:m`/`:move` — move the range to after a destination line.
    Move,
    /// `:t`/`:copy`/`:co` — copy the range to after a destination line.
    Copy,
    /// `:normal`/`:norm` — run normal-mode keys over each line in the range.
    Normal,
    /// `:earlier`/`:ea` — step back through undo states by a count.
    Earlier,
    /// `:later`/`:lat` — step forward through redo states by a count.
    Later,
    /// `:d`/`:delete`.
    Delete,
    /// `:fold`/`:fo` — create a manual fold over the range.
    Fold,
    /// `:noh`/`:nohlsearch`.
    NoHighlight,
    /// `:set`.
    Set,
    /// `:sp`/`:split`.
    Split,
    /// `:vs`/`:vsp`/`:vsplit`.
    VSplit,
    /// `:new`.
    New,
    /// `:vnew`/`:vne`.
    VNew,
    /// `:on`/`:only`.
    Only,
    /// `:clo`/`:close`.
    Close,
    /// `:term`/`:terminal`.
    Terminal,
    /// `:r`/`:read` — read a shell command's output into the buffer
    /// (`:r !{cmd}`). The file-read form is not yet implemented.
    Read,
    /// `:h`/`:help`.
    Help,

    // --- LSP control (kopitiam-cj0.61) ---
    /// `:LspStart` — force-attach the language server for the current buffer,
    /// bypassing the resource-aware guard for this session.
    LspStart,
    /// `:LspInfo` — print the guard's probe numbers, RA-memory estimate, and
    /// gate decision.
    LspInfo,

    // --- Quickfix & location lists (kopitiam-cj0.18) ---
    // Project-wide search into a navigable list. The `c`-prefixed commands act
    // on the global *quickfix* list; the `l`-prefixed twins act on the
    // window-local *location* list. See `super::quickfix`.
    /// `:gr`/`:grep {pattern} [globs]` — search the project into the quickfix list.
    Grep,
    /// `:vim`/`:vimgrep {pattern} [globs]` — same, vim's in-process grep name.
    VimGrep,
    /// `:lgr`/`:lgrep` — `:grep` into the location list.
    LGrep,
    /// `:lvim`/`:lvimgrep` — `:vimgrep` into the location list.
    LVimGrep,
    /// `:cope`/`:copen` — open the quickfix window.
    Copen,
    /// `:ccl`/`:cclose` — close the quickfix window.
    Cclose,
    /// `:cw`/`:cwindow` — open the quickfix window if it has entries, else close it.
    Cwindow,
    /// `:cn`/`:cnext` — go to the next quickfix entry.
    Cnext,
    /// `:cp`/`:cprev`/`:cprevious` — go to the previous quickfix entry.
    Cprev,
    /// `:cfir`/`:cfirst` — go to the first quickfix entry.
    Cfirst,
    /// `:cla`/`:clast` — go to the last quickfix entry.
    Clast,
    /// `:cc [nr]` — go to quickfix entry `nr` (or re-display the current one).
    CC,
    /// `:cdo {cmd}` — run an ex command on each quickfix entry's buffer.
    Cdo,
    /// `:lop`/`:lopen` — open the location window.
    Lopen,
    /// `:lcl`/`:lclose` — close the location window.
    Lclose,
    /// `:lw`/`:lwindow` — open the location window if it has entries, else close it.
    Lwindow,
    /// `:lne`/`:lnext` — go to the next location entry.
    Lnext,
    /// `:lp`/`:lprev`/`:lprevious` — go to the previous location entry.
    Lprev,
    /// `:lfir`/`:lfirst` — go to the first location entry.
    Lfirst,
    /// `:lla`/`:llast` — go to the last location entry.
    Llast,
    /// `:ll [nr]` — go to location entry `nr` (or re-display the current one).
    LL,
    /// `:ldo {cmd}` — run an ex command on each location entry's buffer.
    Ldo,
}

/// One row of the registry.
#[derive(Debug, Clone, Copy)]
pub struct CommandSpec {
    /// Which command this is, for dispatch.
    pub id: CommandId,
    /// Every name that invokes it, canonical (shortest common abbreviation)
    /// first. Both dispatch and completion read this list; nothing else keeps
    /// a second copy.
    pub names: &'static [&'static str],
    /// What to complete after the name.
    pub arg: ArgKind,
    /// One-line description, shown by a future `:help`/command palette.
    pub help: &'static str,
}

/// The whole command vocabulary. Ordering is the order completion offers ties
/// in, so keep related commands together and common ones early.
pub const COMMANDS: &[CommandSpec] = &[
    CommandSpec { id: CommandId::Write, names: &["w", "write"], arg: ArgKind::File, help: "write the buffer to file" },
    CommandSpec { id: CommandId::WriteQuit, names: &["wq"], arg: ArgKind::File, help: "write, then quit" },
    CommandSpec { id: CommandId::Xit, names: &["x", "xit"], arg: ArgKind::None, help: "write if changed, then quit" },
    CommandSpec { id: CommandId::Quit, names: &["q", "quit"], arg: ArgKind::None, help: "quit this window" },
    CommandSpec { id: CommandId::QuitAll, names: &["qa", "qall", "quita", "quitall"], arg: ArgKind::None, help: "quit all windows" },
    CommandSpec { id: CommandId::WriteAll, names: &["wa", "wall"], arg: ArgKind::None, help: "write all buffers" },
    CommandSpec { id: CommandId::WriteQuitAll, names: &["wqa", "wqall"], arg: ArgKind::None, help: "write all, then quit all" },
    CommandSpec { id: CommandId::XitAll, names: &["xa", "xall"], arg: ArgKind::None, help: "write all if changed, quit all" },
    CommandSpec { id: CommandId::Edit, names: &["e", "edit"], arg: ArgKind::File, help: "edit a file" },
    CommandSpec { id: CommandId::NextBuffer, names: &["bn", "bnext"], arg: ArgKind::None, help: "go to next buffer" },
    CommandSpec { id: CommandId::PrevBuffer, names: &["bp", "bprev", "bprevious"], arg: ArgKind::None, help: "go to previous buffer" },
    CommandSpec { id: CommandId::Buffer, names: &["b", "buffer"], arg: ArgKind::Buffer, help: "go to buffer by name/number" },
    CommandSpec { id: CommandId::DeleteBuffer, names: &["bd", "bdel", "bdelete"], arg: ArgKind::None, help: "delete this buffer" },
    CommandSpec { id: CommandId::WipeBuffer, names: &["bw", "bwipe", "bwipeout"], arg: ArgKind::None, help: "wipe this buffer" },
    CommandSpec { id: CommandId::ListBuffers, names: &["ls", "buffers", "files"], arg: ArgKind::None, help: "list open buffers" },
    CommandSpec { id: CommandId::Substitute, names: &["s", "substitute"], arg: ArgKind::None, help: "substitute pattern in range" },
    CommandSpec { id: CommandId::Global, names: &["g", "global"], arg: ArgKind::None, help: "run a command on matching lines" },
    CommandSpec { id: CommandId::VGlobal, names: &["v", "vg", "vglobal"], arg: ArgKind::None, help: "run a command on non-matching lines" },
    CommandSpec { id: CommandId::Sort, names: &["sor", "sort"], arg: ArgKind::None, help: "sort lines in range (!/u/n flags)" },
    CommandSpec { id: CommandId::Move, names: &["m", "mo", "mov", "move"], arg: ArgKind::None, help: "move range to after {address}" },
    CommandSpec { id: CommandId::Copy, names: &["t", "co", "cop", "copy"], arg: ArgKind::None, help: "copy range to after {address}" },
    CommandSpec { id: CommandId::Normal, names: &["norm", "norma", "normal"], arg: ArgKind::None, help: "run normal-mode keys over range" },
    CommandSpec { id: CommandId::Earlier, names: &["ea", "earlier"], arg: ArgKind::None, help: "go back N undo states" },
    CommandSpec { id: CommandId::Later, names: &["lat", "later"], arg: ArgKind::None, help: "go forward N redo states" },
    CommandSpec { id: CommandId::Delete, names: &["d", "delete"], arg: ArgKind::None, help: "delete lines in range" },
    CommandSpec { id: CommandId::Fold, names: &["fo", "fold"], arg: ArgKind::None, help: "create a manual fold over the range" },
    CommandSpec { id: CommandId::NoHighlight, names: &["noh", "nohlsearch"], arg: ArgKind::None, help: "clear search highlight" },
    CommandSpec { id: CommandId::Set, names: &["set"], arg: ArgKind::None, help: "set an option" },
    CommandSpec { id: CommandId::Split, names: &["sp", "split"], arg: ArgKind::File, help: "split window horizontally" },
    CommandSpec { id: CommandId::VSplit, names: &["vs", "vsp", "vsplit"], arg: ArgKind::File, help: "split window vertically" },
    CommandSpec { id: CommandId::New, names: &["new"], arg: ArgKind::None, help: "new horizontal split" },
    CommandSpec { id: CommandId::VNew, names: &["vnew", "vne"], arg: ArgKind::None, help: "new vertical split" },
    CommandSpec { id: CommandId::Only, names: &["on", "only"], arg: ArgKind::None, help: "close all other windows" },
    CommandSpec { id: CommandId::Close, names: &["clo", "close"], arg: ArgKind::None, help: "close this window" },
    CommandSpec { id: CommandId::Terminal, names: &["term", "terminal"], arg: ArgKind::None, help: "open a terminal buffer" },
    CommandSpec { id: CommandId::Read, names: &["r", "read"], arg: ArgKind::None, help: "read shell command output into buffer" },
    CommandSpec { id: CommandId::Help, names: &["h", "help"], arg: ArgKind::File, help: "open the help manual" },
    // LSP control. Capitalised like Neovim's own `:LspInfo`/`:LspStart` user
    // commands (registry lookup is exact-match, and these are all-alphabetic so
    // `ex::parse` captures the whole name).
    CommandSpec { id: CommandId::LspStart, names: &["LspStart"], arg: ArgKind::None, help: "force-start the LSP, bypassing the resource guard" },
    CommandSpec { id: CommandId::LspInfo, names: &["LspInfo"], arg: ArgKind::None, help: "show the LSP resource-guard estimate and decision" },

    // Quickfix list (global) — project search + navigate + iterate.
    CommandSpec { id: CommandId::Grep, names: &["gr", "grep"], arg: ArgKind::None, help: "search project into the quickfix list" },
    CommandSpec { id: CommandId::VimGrep, names: &["vim", "vimgrep"], arg: ArgKind::None, help: "search project into the quickfix list" },
    CommandSpec { id: CommandId::Copen, names: &["cope", "copen"], arg: ArgKind::None, help: "open the quickfix window" },
    CommandSpec { id: CommandId::Cclose, names: &["ccl", "cclose"], arg: ArgKind::None, help: "close the quickfix window" },
    CommandSpec { id: CommandId::Cwindow, names: &["cw", "cwindow"], arg: ArgKind::None, help: "open quickfix window if non-empty" },
    CommandSpec { id: CommandId::Cnext, names: &["cn", "cnext"], arg: ArgKind::None, help: "next quickfix entry" },
    CommandSpec { id: CommandId::Cprev, names: &["cp", "cprev", "cprevious"], arg: ArgKind::None, help: "previous quickfix entry" },
    CommandSpec { id: CommandId::Cfirst, names: &["cfir", "cfirst"], arg: ArgKind::None, help: "first quickfix entry" },
    CommandSpec { id: CommandId::Clast, names: &["cla", "clast"], arg: ArgKind::None, help: "last quickfix entry" },
    CommandSpec { id: CommandId::CC, names: &["cc"], arg: ArgKind::None, help: "go to quickfix entry [nr]" },
    CommandSpec { id: CommandId::Cdo, names: &["cdo"], arg: ArgKind::None, help: "run a command on each quickfix entry" },

    // Location list (window-local) — the `l`-prefixed twins.
    CommandSpec { id: CommandId::LGrep, names: &["lgr", "lgrep"], arg: ArgKind::None, help: "search project into the location list" },
    CommandSpec { id: CommandId::LVimGrep, names: &["lvim", "lvimgrep"], arg: ArgKind::None, help: "search project into the location list" },
    CommandSpec { id: CommandId::Lopen, names: &["lop", "lopen"], arg: ArgKind::None, help: "open the location window" },
    CommandSpec { id: CommandId::Lclose, names: &["lcl", "lclose"], arg: ArgKind::None, help: "close the location window" },
    CommandSpec { id: CommandId::Lwindow, names: &["lw", "lwindow"], arg: ArgKind::None, help: "open location window if non-empty" },
    CommandSpec { id: CommandId::Lnext, names: &["lne", "lnext"], arg: ArgKind::None, help: "next location entry" },
    CommandSpec { id: CommandId::Lprev, names: &["lp", "lprev", "lprevious"], arg: ArgKind::None, help: "previous location entry" },
    CommandSpec { id: CommandId::Lfirst, names: &["lfir", "lfirst"], arg: ArgKind::None, help: "first location entry" },
    CommandSpec { id: CommandId::Llast, names: &["lla", "llast"], arg: ArgKind::None, help: "last location entry" },
    CommandSpec { id: CommandId::LL, names: &["ll"], arg: ArgKind::None, help: "go to location entry [nr]" },
    CommandSpec { id: CommandId::Ldo, names: &["ldo"], arg: ArgKind::None, help: "run a command on each location entry" },
];

/// Looks up a command by any of its names (exact match).
///
/// Exact, not prefix: dispatch must not guess. `:w` is [`CommandId::Write`],
/// full stop; a user who types `:wr` and means `:write` is offered the full
/// name by [`complete_names`] first, and only a name that exactly appears in
/// [`COMMANDS`] dispatches. This mirrors vim, where an ambiguous or unknown
/// leading fragment is an error, not a silent best-guess.
pub fn lookup(name: &str) -> Option<&'static CommandSpec> {
    COMMANDS.iter().find(|spec| spec.names.contains(&name))
}

/// Every command name (across all aliases) that starts with `prefix`, sorted
/// and de-duplicated — the candidate list for `:`-name completion.
///
/// An empty `prefix` returns every name (vim's `:<Tab>` on an empty line offers
/// the whole set). Aliases are offered too, not just canonical names, matching
/// vim's completion which lists `w`, `wa`, `wall`, `wq`, `write`, ... for `:w`.
pub fn complete_names(prefix: &str) -> Vec<String> {
    let mut out: Vec<String> = COMMANDS
        .iter()
        .flat_map(|spec| spec.names.iter())
        .filter(|name| name.starts_with(prefix))
        .map(|name| name.to_string())
        .collect();
    out.sort();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_alias_resolves_to_its_group() {
        assert_eq!(lookup("w").unwrap().id, CommandId::Write);
        assert_eq!(lookup("write").unwrap().id, CommandId::Write);
        assert_eq!(lookup("wq").unwrap().id, CommandId::WriteQuit);
        assert_eq!(lookup("bwipeout").unwrap().id, CommandId::WipeBuffer);
        assert_eq!(lookup("vsplit").unwrap().id, CommandId::VSplit);
        // The line-manipulation family added in kopitiam-cj0.19.
        assert_eq!(lookup("v").unwrap().id, CommandId::VGlobal);
        assert_eq!(lookup("vglobal").unwrap().id, CommandId::VGlobal);
        assert_eq!(lookup("sort").unwrap().id, CommandId::Sort);
        assert_eq!(lookup("m").unwrap().id, CommandId::Move);
        assert_eq!(lookup("move").unwrap().id, CommandId::Move);
        assert_eq!(lookup("t").unwrap().id, CommandId::Copy);
        assert_eq!(lookup("copy").unwrap().id, CommandId::Copy);
        assert_eq!(lookup("norm").unwrap().id, CommandId::Normal);
        assert_eq!(lookup("earlier").unwrap().id, CommandId::Earlier);
        assert_eq!(lookup("lat").unwrap().id, CommandId::Later);
        assert!(lookup("definitely-not-a-command").is_none());
    }

    #[test]
    fn name_completion_offers_matching_aliases_sorted() {
        let names = complete_names("wq");
        assert_eq!(names, vec!["wq".to_string(), "wqa".to_string(), "wqall".to_string()]);
    }

    #[test]
    fn name_completion_of_empty_prefix_offers_everything() {
        let names = complete_names("");
        // At least as many as there are commands (aliases push it higher).
        assert!(names.len() >= COMMANDS.len());
        assert!(names.contains(&"quit".to_string()));
        assert!(names.contains(&"vsplit".to_string()));
    }

    #[test]
    fn no_duplicate_names_across_the_table() {
        let mut all: Vec<&str> = COMMANDS.iter().flat_map(|s| s.names.iter().copied()).collect();
        all.sort_unstable();
        let mut deduped = all.clone();
        deduped.dedup();
        assert_eq!(all, deduped, "a command name is registered twice");
    }

    #[test]
    fn arg_kinds_are_set_for_the_file_and_buffer_commands() {
        assert_eq!(lookup("e").unwrap().arg, ArgKind::File);
        assert_eq!(lookup("w").unwrap().arg, ArgKind::File);
        assert_eq!(lookup("vsplit").unwrap().arg, ArgKind::File);
        assert_eq!(lookup("b").unwrap().arg, ArgKind::Buffer);
        assert_eq!(lookup("q").unwrap().arg, ArgKind::None);
    }
}

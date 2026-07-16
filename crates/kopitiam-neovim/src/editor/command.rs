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
    /// `:d`/`:delete`.
    Delete,
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
    /// `:h`/`:help`.
    Help,
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
    CommandSpec { id: CommandId::Delete, names: &["d", "delete"], arg: ArgKind::None, help: "delete lines in range" },
    CommandSpec { id: CommandId::NoHighlight, names: &["noh", "nohlsearch"], arg: ArgKind::None, help: "clear search highlight" },
    CommandSpec { id: CommandId::Set, names: &["set"], arg: ArgKind::None, help: "set an option" },
    CommandSpec { id: CommandId::Split, names: &["sp", "split"], arg: ArgKind::File, help: "split window horizontally" },
    CommandSpec { id: CommandId::VSplit, names: &["vs", "vsp", "vsplit"], arg: ArgKind::File, help: "split window vertically" },
    CommandSpec { id: CommandId::New, names: &["new"], arg: ArgKind::None, help: "new horizontal split" },
    CommandSpec { id: CommandId::VNew, names: &["vnew", "vne"], arg: ArgKind::None, help: "new vertical split" },
    CommandSpec { id: CommandId::Only, names: &["on", "only"], arg: ArgKind::None, help: "close all other windows" },
    CommandSpec { id: CommandId::Close, names: &["clo", "close"], arg: ArgKind::None, help: "close this window" },
    CommandSpec { id: CommandId::Terminal, names: &["term", "terminal"], arg: ArgKind::None, help: "open a terminal buffer" },
    CommandSpec { id: CommandId::Help, names: &["h", "help"], arg: ArgKind::File, help: "open the help manual" },
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

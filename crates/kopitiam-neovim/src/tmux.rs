//! Multiplexer detection and the tmux `is_vim` auto-fix — the pure half.
//!
//! # The bug this whole module exist to solve
//!
//! kvim bind bare `<C-h/j/k/l>` for two jobs at one shot: move focus between
//! kvim's own splits, and at the edge of kvim's layout hand off to the
//! neighbouring **tmux** pane (`tmux select-pane -L/-D/-U/-R`, the
//! vim-tmux-navigator contract — see [`crate::ui::app::App::tmux_select_pane`]).
//!
//! For that to work, tmux must *forward* `<C-h/j/k/l>` into kvim instead of
//! grabbing them for its own pane navigation. The vim-tmux-navigator convention
//! (christoomey/vim-tmux-navigator, MIT — studied for behaviour, no code copied)
//! do this with an `is_vim` shell check inside `~/.tmux.conf`: it look at the
//! process running in the active pane and, if the name match a regex of
//! vim-like editors, forward the key; otherwise it navigate tmux's panes.
//!
//! The catch: that regex list `vim`, `nvim`, `view`, `fzf` — but **not `kvim`**.
//! So the moment kvim run inside tmux, tmux dun recognise it as vim-like, eat
//! `<C-h/j/k/l>` before kvim ever see them, and kvim's own split navigation go
//! quietly dead. kvim already *documents* that the user must add `kvim` to their
//! regex (see the crate README). This module make kvim **detect** the problem
//! and **offer to patch the conf** for the user — with consent, always.
//!
//! # What is pure here, and what is not
//!
//! Everything that decide *whether* and *how* to fix a conf is a pure function
//! of a conf string ([`compute_fix`]) so it can be unit-tested hard with no
//! filesystem and no tmux. The thin layer that read the environment, locate the
//! real conf file, and write the backup ([`startup_advice`], [`TmuxOffer::apply`])
//! sit on top. The consent popup and the keystroke handling live one layer up
//! again, in [`crate::ui::app`] — this module never draw a cell and never edit a
//! file without the caller having got a yes first.
//!
//! # Safety property: never edit a dotfile without consent
//!
//! [`compute_fix`] only *computes* a new conf string; it touch nothing.
//! [`TmuxOffer::apply`] is the only thing here that write to disk, it always
//! back the original up first (`tmux.conf.kvim-bak`), and the UI only call it
//! after the user press `y`. Editing somebody's `~/.tmux.conf` behind their back
//! would be as unacceptable as kvim quietly rewriting `~/.config/nvim` — same
//! spirit, same hard line.

use std::path::{Path, PathBuf};

/// The crate's per-user directory name, reused for the decline marker. Kept in
/// sync with [`crate::config`]'s own `APP_NAME` (they name the same directory).
const APP_NAME: &str = "kopitiam-neovim";

/// A terminal multiplexer kvim can be running inside, detected from the
/// environment. Env-based detection is deliberately cross-OS: `$TMUX`, `$STY`
/// and `$ZELLIJ` are set by the multiplexer itself for every process in a pane,
/// on Linux, macOS and WSL alike, with no per-OS branch to get wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Multiplexer {
    /// tmux — the one kvim can auto-configure (`$TMUX` set).
    Tmux,
    /// GNU screen (`$STY` set). Its config syntax differs; kvim only notes it.
    Screen,
    /// zellij (`$ZELLIJ` set). Different config model again; kvim only notes it.
    Zellij,
    /// No multiplexer detected.
    None,
}

/// Detects the multiplexer from the environment.
///
/// tmux win ties: if somehow more than one variable is set (a tmux inside a
/// zellij, say), the tmux answer is the actionable one, so it is checked first.
pub fn detect_multiplexer() -> Multiplexer {
    if std::env::var_os("TMUX").is_some() {
        Multiplexer::Tmux
    } else if std::env::var_os("ZELLIJ").is_some() {
        Multiplexer::Zellij
    } else if std::env::var_os("STY").is_some() {
        Multiplexer::Screen
    } else {
        Multiplexer::None
    }
}

/// Which of the three shapes a fix take. Named so the consent popup can phrase
/// itself honestly: "change this line" is a very different promise from "create
/// a new file".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixKind {
    /// The conf already has a vim-tmux-navigator `is_vim` regex, it just dun
    /// list `kvim`. The fix is a surgical one-line edit: slot `kvim` into the
    /// editor-name alternation, touching nothing else.
    ExtendRegex,
    /// The conf exist but has no vim-tmux-navigator setup at all. The fix append
    /// a fresh, clearly-commented navigator block.
    AppendBlock,
    /// No conf file exist. The fix create one, holding just the navigator block.
    CreateFile,
}

/// A computed, not-yet-applied edit to a tmux conf: the full replacement text
/// plus enough context for the popup to show the user *exactly* what will
/// change before they consent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedEdit {
    /// Which shape of fix this is.
    pub kind: FixKind,
    /// The full new contents to write. For [`FixKind::ExtendRegex`] this is the
    /// whole conf with one line rewritten; for the other two it is the old conf
    /// (or nothing) with the block appended.
    pub new_conf: String,
    /// For [`FixKind::ExtendRegex`], the single conf line as it stands *before*
    /// the edit, so the popup can show old-vs-new. `None` for append/create,
    /// where nothing existing is being changed.
    pub old_line: Option<String>,
    /// The line(s) the edit introduces: the rewritten regex line for
    /// [`FixKind::ExtendRegex`], or the block for append/create. Shown verbatim
    /// in the consent popup.
    pub new_lines: Vec<String>,
}

/// The commented vim-tmux-navigator block kvim append (or create a file with)
/// when a conf has no navigator setup of its own.
///
/// The regex is the christoomey/vim-tmux-navigator canonical one
/// (`(\S+\/)?g?(view|n?vim?x?|fzf)(diff)?`) with `kvim` slotted into the
/// alternation, so it keep recognising `vim`, `nvim`, `view` and `fzf` and now
/// also `kvim`. The double backslashes (`\\S`, `\\/`) are correct and load-
/// bearing: inside the double-quoted `is_vim="..."` string tmux collapse `\\`
/// to `\`, so the regex that actually reach `grep` is `\S`/`\/`. A single
/// backslash here would reach grep as a bare `S`/`/` and quietly break the
/// match. See `docs/ai-decisions` for why this exact form was chosen over the
/// abbreviated one the README used to show.
const NAVIGATOR_BLOCK: &str = r#"
# ── kvim / vim-tmux-navigator ──────────────────────────────────────────
# Added by kvim (KOPITIAM's editor). Let vim-like apps own <C-h/j/k/l> so
# their splits and tmux's panes navigate as one thing. The `kvim` in the
# regex is the load-bearing bit: without it tmux eat those keys before kvim
# ever see them, and kvim's split navigation go dead. To undo, delete this
# block. After editing, reload with:  tmux source-file <this file>
is_vim="ps -o state= -o comm= -t '#{pane_tty}' \
    | grep -iqE '^[^TXZ ]+ +(\\S+\\/)?g?(view|kvim|n?vim?x?|fzf)(diff)?$'"
bind-key -n 'C-h' if-shell "$is_vim" 'send-keys C-h' 'select-pane -L'
bind-key -n 'C-j' if-shell "$is_vim" 'send-keys C-j' 'select-pane -D'
bind-key -n 'C-k' if-shell "$is_vim" 'send-keys C-k' 'select-pane -U'
bind-key -n 'C-l' if-shell "$is_vim" 'send-keys C-l' 'select-pane -R'
# ───────────────────────────────────────────────────────────────────────
"#;

/// Does this conf already recognise `kvim`?
///
/// The literal string `kvim` appearing anywhere in the conf is taken as "the
/// user has already configured for kvim" — whether that is `kvim` inside an
/// `is_vim` regex, or a hand-rolled kvim-aware pane-nav binding. This is
/// deliberately generous: a false positive (say, `kvim` only in a comment) just
/// means kvim stay quiet, which is the safe direction — kvim never edit a conf
/// it is unsure about. A false *negative* would be the dangerous one, and `kvim`
/// is a distinctive enough token that it dun occur by accident.
pub fn recognises_kvim(conf: &str) -> bool {
    conf.contains("kvim")
}

/// Does this conf carry a vim-tmux-navigator-style `is_vim` process check?
///
/// The signature is an `is_vim` name next to a `grep` (the editor-name regex).
/// Both together, so a stray `is_vim` variable used for something unrelated dun
/// get mistaken for the navigator setup.
pub fn has_is_vim_check(conf: &str) -> bool {
    conf.contains("is_vim") && conf.contains("grep")
}

/// Computes the fix for a conf, or `None` if none is needed.
///
/// `existing` is the current conf contents, or `None` when no conf file exist.
/// The three not-fixed cases map to the three [`FixKind`]s:
///
/// * conf has an `is_vim` regex but no `kvim` → [`FixKind::ExtendRegex`];
/// * conf exist but has no navigator setup → [`FixKind::AppendBlock`];
/// * no conf file → [`FixKind::CreateFile`].
///
/// Returns `None` when the conf already recognise `kvim` (already fixed), so the
/// caller show no popup at all.
pub fn compute_fix(existing: Option<&str>) -> Option<PlannedEdit> {
    match existing {
        Some(conf) if recognises_kvim(conf) => None,
        Some(conf) if has_is_vim_check(conf) => {
            // Preferred: surgical one-line edit of the existing regex.
            if let Some((new_conf, old_line, new_line)) = extend_regex(conf) {
                return Some(PlannedEdit {
                    kind: FixKind::ExtendRegex,
                    new_conf,
                    old_line: Some(old_line),
                    new_lines: vec![new_line],
                });
            }
            // The conf has an `is_vim` check we could not confidently locate the
            // alternation inside (an unusual hand-rolled form). Rather than risk
            // a bad surgical edit, fall back to appending a fresh block — the
            // new block's `is_vim` shadow the earlier definition, which is the
            // safe outcome: kvim's own known-good regex win.
            Some(append_block(conf))
        }
        Some(conf) => Some(append_block(conf)),
        None => Some(PlannedEdit {
            kind: FixKind::CreateFile,
            new_conf: NAVIGATOR_BLOCK.trim_start_matches('\n').to_string(),
            old_line: None,
            new_lines: block_lines(),
        }),
    }
}

/// Builds the [`FixKind::AppendBlock`] edit: the old conf with the navigator
/// block appended. A single newline is ensured before the block so it dun run
/// onto the conf's last line.
fn append_block(conf: &str) -> PlannedEdit {
    let mut new_conf = conf.to_string();
    if !new_conf.ends_with('\n') {
        new_conf.push('\n');
    }
    new_conf.push_str(NAVIGATOR_BLOCK.trim_start_matches('\n'));
    PlannedEdit {
        kind: FixKind::AppendBlock,
        new_conf,
        old_line: None,
        new_lines: block_lines(),
    }
}

/// The navigator block split into display lines (leading/trailing blank lines
/// trimmed), for the consent popup.
fn block_lines() -> Vec<String> {
    NAVIGATOR_BLOCK
        .trim_matches('\n')
        .lines()
        .map(str::to_string)
        .collect()
}

/// Slots `kvim` into an existing `is_vim` editor-name alternation.
///
/// Returns `(new_conf, old_line, new_line)` — the full rewritten conf plus the
/// single physical line that changed, before and after — or `None` if the
/// alternation could not be located.
///
/// # How the insertion point is found
///
/// The `is_vim` check run `grep -iqE` over a regex whose editor names live in an
/// alternation group, e.g. `(view|n?vim?x?|fzf)`. We anchor on the `vim` token
/// (present in every such regex), then:
///
/// * if it sit inside a `(...)` group, we insert `kvim|` right after that
///   group's opening paren → `(kvim|view|n?vim?x?|fzf)`;
/// * if there is no group (a bare `grep -iqE 'n?vim|fzf'`), we back up over any
///   regex prefix bound to the token (the `n?` in `n?vim`) and insert `kvim|`
///   there → `kvim|n?vim|fzf`, never splitting the `n?vim` apart.
///
/// Either way the added alternative is a plain literal `kvim`, so it match the
/// `kvim` process name and nothing else.
fn extend_regex(conf: &str) -> Option<(String, String, String)> {
    let is_vim_at = conf.find("is_vim")?;
    let grep_rel = conf[is_vim_at..].find("grep")?;
    let grep_at = is_vim_at + grep_rel;
    let vim_rel = conf[grep_at..].find("vim")?;
    let vim_at = grep_at + vim_rel;

    let insert_at = match conf[grep_at..vim_at].rfind('(') {
        Some(rel) => grep_at + rel + 1,
        None => {
            // Ungrouped: walk back over the token's own regex chars so `kvim|`
            // land at the token boundary, not inside `n?vim`.
            let bytes = conf.as_bytes();
            let mut start = vim_at;
            while start > grep_at {
                let c = bytes[start - 1];
                if c.is_ascii_alphanumeric() || c == b'?' {
                    start -= 1;
                } else {
                    break;
                }
            }
            start
        }
    };

    let mut new_conf = String::with_capacity(conf.len() + 5);
    new_conf.push_str(&conf[..insert_at]);
    new_conf.push_str("kvim|");
    new_conf.push_str(&conf[insert_at..]);

    let old_line = line_at(conf, insert_at).to_string();
    let new_line = line_at(&new_conf, insert_at).to_string();
    Some((new_conf, old_line, new_line))
}

/// The physical line (no trailing newline) of `s` that contains byte `at`.
fn line_at(s: &str, at: usize) -> &str {
    let start = s[..at].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let end = s[at..].find('\n').map(|i| at + i).unwrap_or(s.len());
    &s[start..end]
}

/// A ready-to-present offer to fix a specific conf file. Built by
/// [`startup_advice`] once it has read the environment and the real conf.
#[derive(Debug, Clone)]
pub struct TmuxOffer {
    /// The conf file the fix target — an existing file for extend/append, or
    /// the path a new file will be created at.
    pub path: PathBuf,
    /// Whether that file already exist (governs whether [`Self::apply`] make a
    /// backup, and how the popup phrase itself).
    pub existed: bool,
    /// The computed edit.
    pub edit: PlannedEdit,
}

impl TmuxOffer {
    /// Applies the fix, backing an existing conf up first.
    ///
    /// Returns the backup path when one was made (`Some` for an existing conf),
    /// or `None` when a fresh file was created (nothing to back up). Errors
    /// propagate — a failed backup abort the write, so kvim never touch the
    /// original unless its copy is safely on disk.
    ///
    /// **Never** call this without the user having consented: it is the one
    /// function in this crate that write to a user's dotfile.
    pub fn apply(&self) -> std::io::Result<Option<PathBuf>> {
        let backup = if self.existed {
            let bak = backup_path(&self.path);
            // Copy first; only proceed to overwrite once the backup exist.
            std::fs::copy(&self.path, &bak)?;
            Some(bak)
        } else {
            if let Some(parent) = self.path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            None
        };
        std::fs::write(&self.path, &self.edit.new_conf)?;
        Ok(backup)
    }
}

/// The backup path for a conf: `<conf>.kvim-bak`, alongside the original.
fn backup_path(conf: &Path) -> PathBuf {
    let mut name = conf.file_name().unwrap_or_default().to_os_string();
    name.push(".kvim-bak");
    conf.with_file_name(name)
}

/// What kvim should do about the multiplexer at startup.
#[derive(Debug, Clone)]
pub enum StartupAdvice {
    /// Say and do nothing — not in a multiplexer, already fixed, the user
    /// declined before, or nowhere writable to put a conf.
    Nothing,
    /// Show a one-line status note, no action offered (screen / zellij).
    Note(String),
    /// Offer, with consent, to fix a tmux conf.
    OfferFix(Box<TmuxOffer>),
}

/// Reads the environment and (for tmux) the conf, and decides what to advise.
///
/// This is the impure entry point the UI call once at startup. It never write
/// anything — applying a fix is [`TmuxOffer::apply`], gated on consent.
pub fn startup_advice() -> StartupAdvice {
    match detect_multiplexer() {
        Multiplexer::None => StartupAdvice::Nothing,
        Multiplexer::Screen => StartupAdvice::Note(
            "You inside GNU screen — kvim's <C-h/j/k/l> only navigate kvim's own splits here \
             (the tmux pane hand-off is tmux-only)."
                .to_string(),
        ),
        Multiplexer::Zellij => StartupAdvice::Note(
            "You inside zellij — kvim's <C-h/j/k/l> only navigate kvim's own splits here \
             (the tmux pane hand-off is tmux-only)."
                .to_string(),
        ),
        Multiplexer::Tmux => tmux_advice(),
    }
}

/// The tmux branch of [`startup_advice`]: honour the decline marker, locate and
/// read the conf, compute the fix.
fn tmux_advice() -> StartupAdvice {
    if declined_before() {
        return StartupAdvice::Nothing;
    }

    let located = locate_conf();
    let existing = match &located {
        Some(path) => match std::fs::read_to_string(path) {
            Ok(text) => Some(text),
            // A conf that exists but cannot be read is not one kvim should be
            // guessing at, let alone overwriting — stay out of it.
            Err(_) => return StartupAdvice::Nothing,
        },
        None => None,
    };

    let Some(edit) = compute_fix(existing.as_deref()) else {
        return StartupAdvice::Nothing; // already recognises kvim
    };

    let (path, existed) = match located {
        Some(path) => (path, true),
        None => match creation_target() {
            Some(path) => (path, false),
            None => return StartupAdvice::Nothing, // no home dir to write under
        },
    };

    StartupAdvice::OfferFix(Box::new(TmuxOffer { path, existed, edit }))
}

/// `$HOME` (or `$USERPROFILE` on Windows), as a path. Matches the fallback
/// order [`kopitiam_config::root`] use, so kvim resolve a home directory the
/// same way everywhere in the crate.
fn home() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

/// Locates an existing tmux conf, trying the standard locations in order:
///
/// 1. `$XDG_CONFIG_HOME/tmux/tmux.conf`
/// 2. `~/.config/tmux/tmux.conf`
/// 3. `~/.tmux.conf`
///
/// Returns the first that exist, or `None` if none do.
pub fn locate_conf() -> Option<PathBuf> {
    candidate_confs().into_iter().find(|path| path.is_file())
}

/// The ordered list of conf locations kvim look at. Split out from
/// [`locate_conf`] so the ordering is one thing, testable on its own.
fn candidate_confs() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME") {
        out.push(PathBuf::from(xdg).join("tmux").join("tmux.conf"));
    }
    if let Some(home) = home() {
        out.push(home.join(".config").join("tmux").join("tmux.conf"));
        out.push(home.join(".tmux.conf"));
    }
    out
}

/// Where kvim create a conf when none exist: the XDG path if
/// `$XDG_CONFIG_HOME` is set, else `~/.config/tmux/tmux.conf`. The modern
/// location, not the legacy `~/.tmux.conf`, so a freshly created conf land
/// where current tmux expect it.
pub fn creation_target() -> Option<PathBuf> {
    if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME") {
        return Some(PathBuf::from(xdg).join("tmux").join("tmux.conf"));
    }
    Some(home()?.join(".config").join("tmux").join("tmux.conf"))
}

/// The decline marker: `~/.kopitiam/kopitiam-neovim/.tmux-autoconfig-declined`.
/// Its presence mean the user said no once and kvim should stop asking. Delete
/// it to get the offer back.
fn decline_marker() -> Option<PathBuf> {
    Some(kopitiam_config::app_dir(APP_NAME)?.join(".tmux-autoconfig-declined"))
}

/// Whether the user has declined the offer before (marker present).
fn declined_before() -> bool {
    decline_marker().map(|p| p.exists()).unwrap_or(false)
}

/// Records that the user declined, so kvim dun nag on the next startup.
///
/// Writes a marker inside kvim's *own* directory — never the user's tmux conf —
/// so declining touch nothing the user owns. A best-effort write: if it fail
/// (no home dir, read-only disk) the worst case is kvim ask again next time,
/// which is a mild annoyance, not a bug worth surfacing.
pub fn remember_decline() {
    if let Some(path) = decline_marker() {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(
            &path,
            "kvim: you declined the tmux <C-h/j/k/l> auto-fix. Delete this file to be asked again.\n",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The canonical christoomey/vim-tmux-navigator conf — has an `is_vim`
    /// regex, list vim/nvim/view/fzf, but *not* kvim. This is the (a) case.
    const CANONICAL_MISSING_KVIM: &str = r#"set -g mouse on

is_vim="ps -o state= -o comm= -t '#{pane_tty}' \
    | grep -iqE '^[^TXZ ]+ +(\\S+\\/)?g?(view|n?vim?x?|fzf)(diff)?$'"
bind-key -n 'C-h' if-shell "$is_vim" 'send-keys C-h' 'select-pane -L'
bind-key -n 'C-j' if-shell "$is_vim" 'send-keys C-j' 'select-pane -D'

set -g status-bg black
"#;

    #[test]
    fn recognises_kvim_only_when_the_literal_token_is_present() {
        assert!(!recognises_kvim(CANONICAL_MISSING_KVIM));
        assert!(recognises_kvim("grep -iqE '(kvim|vim|fzf)'"));
    }

    #[test]
    fn has_is_vim_check_needs_both_is_vim_and_grep() {
        assert!(has_is_vim_check(CANONICAL_MISSING_KVIM));
        assert!(!has_is_vim_check("set -g mouse on")); // neither
        assert!(!has_is_vim_check("is_vim=1")); // is_vim but no grep
    }

    #[test]
    fn already_fixed_conf_needs_no_edit() {
        let fixed = CANONICAL_MISSING_KVIM.replace("n?vim?x?", "kvim|n?vim?x?");
        assert!(compute_fix(Some(&fixed)).is_none());
    }

    // (a) is_vim present, missing kvim → surgical ExtendRegex.
    #[test]
    fn case_a_extends_the_existing_regex_in_place() {
        let edit = compute_fix(Some(CANONICAL_MISSING_KVIM)).expect("a fix is needed");
        assert_eq!(edit.kind, FixKind::ExtendRegex);
        // kvim slotted into the alternation, at the group start.
        assert!(edit.new_conf.contains("(kvim|view|n?vim?x?|fzf)"), "{}", edit.new_conf);
        // Nothing else moved: the bind lines are untouched.
        assert!(edit.new_conf.contains("bind-key -n 'C-h' if-shell \"$is_vim\" 'send-keys C-h' 'select-pane -L'"));
        // The applied conf now recognises kvim, so a re-run would offer nothing.
        assert!(compute_fix(Some(&edit.new_conf)).is_none());
        // Old/new preview is the single grep line, before and after.
        let old = edit.old_line.as_deref().unwrap();
        assert!(old.contains("(view|n?vim?x?|fzf)"));
        assert!(!old.contains("kvim"));
        assert_eq!(edit.new_lines.len(), 1);
        assert!(edit.new_lines[0].contains("(kvim|view|n?vim?x?|fzf)"));
    }

    #[test]
    fn case_a_handles_an_ungrouped_bare_regex() {
        let conf = "is_vim=\"ps | grep -iqE 'n?vim|fzf'\"\n";
        let edit = compute_fix(Some(conf)).expect("a fix is needed");
        assert_eq!(edit.kind, FixKind::ExtendRegex);
        // Inserted at the token boundary, not inside `n?vim`.
        assert!(edit.new_conf.contains("kvim|n?vim|fzf"), "{}", edit.new_conf);
    }

    // (b) conf with no navigator setup → AppendBlock.
    #[test]
    fn case_b_appends_a_block_when_no_navigator_present() {
        let conf = "set -g mouse on\nset -g status-bg black\n";
        let edit = compute_fix(Some(conf)).expect("a fix is needed");
        assert_eq!(edit.kind, FixKind::AppendBlock);
        // Original content preserved, block appended, kvim now present.
        assert!(edit.new_conf.starts_with("set -g mouse on\n"));
        assert!(edit.new_conf.contains("(view|kvim|n?vim?x?|fzf)"));
        assert!(edit.new_conf.contains("bind-key -n 'C-l' if-shell \"$is_vim\" 'send-keys C-l' 'select-pane -R'"));
        // Applying it makes the conf fixed.
        assert!(compute_fix(Some(&edit.new_conf)).is_none());
    }

    #[test]
    fn case_b_inserts_a_newline_before_the_block_when_conf_lacks_a_trailing_one() {
        let conf = "set -g mouse on"; // no trailing newline
        let edit = compute_fix(Some(conf)).expect("a fix is needed");
        assert!(edit.new_conf.starts_with("set -g mouse on\n"));
    }

    // (c) no conf file → CreateFile.
    #[test]
    fn case_c_creates_a_file_holding_just_the_block() {
        let edit = compute_fix(None).expect("a fix is needed");
        assert_eq!(edit.kind, FixKind::CreateFile);
        assert!(edit.new_conf.contains("(view|kvim|n?vim?x?|fzf)"));
        // No stray leading blank line at the top of a brand-new file.
        assert!(!edit.new_conf.starts_with('\n'));
        assert!(compute_fix(Some(&edit.new_conf)).is_none());
    }

    #[test]
    fn the_appended_regex_keeps_the_load_bearing_double_backslashes() {
        // Inside the double-quoted is_vim string, tmux collapse \\ to \, so the
        // regex must carry \\S / \\/ to reach grep as \S / \/.
        let edit = compute_fix(None).unwrap();
        assert!(edit.new_conf.contains(r"(\\S+\\/)?"), "{}", edit.new_conf);
    }

    #[test]
    fn backup_path_appends_kvim_bak_alongside_the_original() {
        assert_eq!(
            backup_path(Path::new("/home/x/.tmux.conf")),
            PathBuf::from("/home/x/.tmux.conf.kvim-bak")
        );
        assert_eq!(
            backup_path(Path::new("/h/.config/tmux/tmux.conf")),
            PathBuf::from("/h/.config/tmux/tmux.conf.kvim-bak")
        );
    }

    #[test]
    fn apply_backs_up_an_existing_conf_then_writes_the_fix() {
        let dir = std::env::temp_dir().join(format!("kvim-tmux-apply-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let conf = dir.join("tmux.conf");
        std::fs::write(&conf, CANONICAL_MISSING_KVIM).unwrap();

        let edit = compute_fix(Some(CANONICAL_MISSING_KVIM)).unwrap();
        let offer = TmuxOffer { path: conf.clone(), existed: true, edit };
        let backup = offer.apply().unwrap();

        // The backup exist and hold the *original*, unedited conf.
        let bak = backup.expect("an existing conf is backed up");
        assert_eq!(std::fs::read_to_string(&bak).unwrap(), CANONICAL_MISSING_KVIM);
        // The conf now hold the fix.
        let written = std::fs::read_to_string(&conf).unwrap();
        assert!(written.contains("(kvim|view|n?vim?x?|fzf)"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn apply_creates_a_new_file_without_a_backup() {
        let dir = std::env::temp_dir().join(format!("kvim-tmux-create-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let conf = dir.join("nested").join("tmux.conf"); // parent does not exist yet

        let edit = compute_fix(None).unwrap();
        let offer = TmuxOffer { path: conf.clone(), existed: false, edit };
        let backup = offer.apply().unwrap();

        assert!(backup.is_none(), "creating a fresh file makes no backup");
        assert!(conf.is_file());
        assert!(std::fs::read_to_string(&conf).unwrap().contains("kvim"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compute_fix_never_mutates_input_it_only_returns_new_text() {
        // The "never edit without consent" property at the pure layer: the only
        // thing compute_fix produce is a *string*; it cannot touch a file.
        let before = CANONICAL_MISSING_KVIM.to_string();
        let _ = compute_fix(Some(&before));
        assert_eq!(before, CANONICAL_MISSING_KVIM);
    }
}

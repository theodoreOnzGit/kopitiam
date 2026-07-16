//! The small slice of ex commands the `vim.*` shim understands at config-load
//! time.
//!
//! This is emphatically **not** kvim's runtime ex engine ([`crate::editor::ex`]),
//! which operates on live buffers. At *config* time there is no buffer to act on
//! — a config runs `colorscheme gruvbox`, `syntax on`, `set number`, and a pile
//! of `nnoremap`s, none of which touch text. So this handles exactly that
//! configuration-shaped subset and records the rest, rather than pulling the
//! whole buffer-oriented ex engine into a phase where there is nothing to edit.
//!
//! Everything it does not recognise is recorded as a one-line warning, never an
//! error: a `vim.cmd` kvim cannot interpret should not stop the config loading.

use std::cell::RefCell;
use std::rc::Rc;

use crate::config::Action;

use super::{Autocmd, VimState, apply_option, classify_rhs_string, record_keymap};

/// Maps a bare ex command (already stripped of any `<cmd>`/`:`/`<cr>` wrapper)
/// to the native [`Action`] a keymap should fire.
///
/// The names here are the plugin *user commands* the maintainer's config binds
/// keys to — `Neotree toggle`, `Telescope harpoon marks`, and so on. Because
/// kvim implements those plugins natively, the command routes straight to the
/// native action rather than to a plugin. Anything unrecognised becomes
/// [`Action::Command`] verbatim, so kvim's real ex engine gets a shot at it at
/// runtime.
pub(super) fn command_action(cmd: &str) -> Action {
    let cmd = cmd.trim();
    let lower = cmd.to_ascii_lowercase();
    // Split off the first word so `Neotree toggle` and `Telescope find_files`
    // match on their command head.
    let head = lower.split_whitespace().next().unwrap_or("");
    match head {
        "neotree" | "neotreetoggle" => Action::FileTreeToggle,
        "telescope" => match lower.split_whitespace().nth(1) {
            Some("find_files") | Some("git_files") => Action::FindFiles,
            Some("buffers") => Action::FindBuffers,
            Some("help_tags") => Action::FindHelp,
            // `Telescope harpoon marks` is the maintainer's `<leader>q`.
            Some("harpoon") => Action::HarpoonFind,
            _ => Action::Command(cmd.to_string()),
        },
        _ => Action::Command(cmd.to_string()),
    }
}

/// Applies a single ex line from a `vim.cmd(...)` block to the config.
///
/// Handles the configuration-time commands (`colorscheme`, `syntax`, `set`, the
/// `*map` family, `autocmd`, `filetype`) and records anything else. `desc` is
/// empty for these — an ex-defined map has no `{ desc = ... }`.
pub(super) fn apply_line(state: &Rc<RefCell<VimState>>, line: &str) {
    let line = line.trim();
    if line.is_empty() || line.starts_with('"') {
        return; // blank or a vimscript comment
    }
    let (head, rest) = split_head(line);
    match head.as_str() {
        // `colorscheme gruvbox` / `colo gruvbox`.
        "colorscheme" | "colo" => {
            let name = rest.trim();
            if !name.is_empty() {
                state.borrow_mut().config.theme = name.to_string();
            }
        }
        // `syntax on` / `syntax off`.
        "syntax" | "syn" => {
            let on = !rest.trim().eq_ignore_ascii_case("off");
            state.borrow_mut().config.options.syntax = on;
        }
        // `set`, `setlocal`, `setglobal`: one or more `opt`, `noopt`, `opt=val`.
        "set" | "setlocal" | "setl" | "setglobal" | "setg" => apply_set(state, rest),
        // Every *map / *noremap variant: `nnoremap lhs rhs`.
        _ if is_map_command(&head) => apply_map(state, &head, rest),
        // `autocmd [group] Events pattern cmd`.
        "autocmd" | "au" => record_autocmd(state, rest),
        // `filetype plugin indent on` — kvim always does filetype detection, so
        // this is a no-op to accept quietly rather than warn about.
        "filetype" | "filet" => {}
        _ => {
            state
                .borrow_mut()
                .warn(format!("vim.cmd: unsupported command `{line}` (recorded, not applied)"));
        }
    }
}

/// `set number relativenumber tabstop=4 nowrap` — vim allows several settings on
/// one line, so split on whitespace and apply each.
fn apply_set(state: &Rc<RefCell<VimState>>, rest: &str) {
    for token in rest.split_whitespace() {
        let (name, value, as_str) = if let Some((n, v)) = token.split_once('=') {
            (n.to_string(), super::lstr(v), v.to_string())
        } else if let Some(n) = token.strip_prefix("no") {
            // `nonumber` → number = false. Only meaningful for boolean options;
            // apply_option ignores the value for numeric ones anyway.
            (n.to_string(), kopitiam_lua::Value::Boolean(false), String::new())
        } else {
            (token.to_string(), kopitiam_lua::Value::Boolean(true), "true".to_string())
        };
        let mut st = state.borrow_mut();
        if apply_option(&mut st.config.options, &name, &value, &as_str).is_err() {
            st.warn(format!(":set {name}: unknown option (ignored)"));
        }
    }
}

/// `nnoremap <leader>b <cmd>lua ...<cr>` — record the mapping. The leading
/// mode/`noremap` prefix has already been split into `head`.
fn apply_map(state: &Rc<RefCell<VimState>>, head: &str, rest: &str) {
    let mode = map_mode(head);
    let mut parts = rest.splitn(2, char::is_whitespace);
    let Some(lhs) = parts.next().map(str::trim).filter(|s| !s.is_empty()) else {
        return;
    };
    let rhs = parts.next().unwrap_or("").trim();
    let action = classify_rhs_string(rhs);
    record_keymap(state, mode, lhs.to_string(), action, String::new());
}

/// `autocmd BufNewFile,BufRead *.tex set filetype=tex` → an [`Autocmd`] record.
fn record_autocmd(state: &Rc<RefCell<VimState>>, rest: &str) {
    let mut it = rest.split_whitespace();
    let events = it.next().unwrap_or("").split(',').map(|s| s.to_string()).collect();
    let pattern = it.next().unwrap_or("").to_string();
    let action: Vec<&str> = it.collect();
    state.borrow_mut().autocmds.push(Autocmd {
        events,
        pattern,
        action: action.join(" "),
    });
}

/// Splits a command line into its first word (lower-cased) and the remainder.
fn split_head(line: &str) -> (String, &str) {
    match line.find(char::is_whitespace) {
        Some(i) => (line[..i].to_ascii_lowercase(), line[i..].trim_start()),
        None => (line.to_ascii_lowercase(), ""),
    }
}

/// True for `nnoremap`, `vmap`, `noremap`, `imap`, and the rest of the family.
fn is_map_command(head: &str) -> bool {
    matches!(
        head,
        "map" | "noremap"
            | "nmap" | "nnoremap"
            | "imap" | "inoremap"
            | "vmap" | "vnoremap"
            | "xmap" | "xnoremap"
            | "cmap" | "cnoremap"
            | "smap" | "snoremap"
            | "omap" | "onoremap"
    )
}

/// The kvim single-letter mode a `*map` command applies in. `map`/`noremap`
/// with no prefix is all-modes (empty string), matching vim's `:map`.
fn map_mode(head: &str) -> String {
    let first = head.chars().next().unwrap_or('n');
    match first {
        'n' if head != "noremap" => "n",
        'i' => "i",
        'v' => "v",
        'x' => "x",
        'c' => "c",
        's' => "s",
        'o' => "o",
        _ => "", // map / noremap: all modes
    }
    .to_string()
}

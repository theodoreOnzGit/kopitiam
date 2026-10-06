//! Execute the user's discovered Lua config through a `vim.*` shim.
//!
//! # What this is
//!
//! kvim already *discovers* `~/.kopitiam/kopitiam-neovim/init.lua` (and
//! `lua/*.lua`) in [`crate::config::Config::lua_files`], but until now it only
//! *reported* them — it never ran them. This module is the missing half: it
//! spins up a [`kopitiam_lua`] VM, installs a `vim.*` API that maps onto kvim's
//! real editor state ([`Options`], [`Keymap`], [`Action`], the leader key, the
//! theme), and executes the config for real.
//!
//! The scale model this is grown from is
//! `crates/kopitiam-lua/tests/maintainer_config.rs` — that test runs the exact
//! same config against a `vim` shim that *records* what was asked for. Here the
//! shim *applies* it instead of recording it, but the shape of the surface is
//! deliberately identical.
//!
//! # The mapping, in one breath
//!
//! | Lua the config writes | Where it land in kvim |
//! |---|---|
//! | `vim.opt.number = true`, `vim.o`/`vim.wo`/`vim.bo` | [`Options`] fields |
//! | `vim.g.mapleader = " "` | [`Config::leader`] (load-bearing) + a var map |
//! | `vim.keymap.set(mode, lhs, rhs, opts)` | a [`Keymap`] with an [`Action`] |
//! | `vim.cmd("colorscheme gruvbox")` / `vim.cmd.colorscheme(...)` | ex engine subset |
//! | `vim.api.nvim_set_keymap` / `nvim_create_autocmd` / ... | the equivalents |
//! | `vim.fn.stdpath`, `vim.notify`, `vim.schedule`, `require` | small shims |
//!
//! # Two design rules that shape everything here
//!
//! 1. **Never crash on unsupported API.** An unknown `vim.<thing>` returns a
//!    harmless black-hole stub (any field is a no-op function, calling it
//!    returns another black-hole) and records a one-line warning, shown once at
//!    startup. A config that uses a `vim.*` we have not built yet must not take
//!    the whole editor down with it. The one hard error is a *malformed* Lua
//!    file — a syntax error or an uncaught runtime error aborts that file (Lua
//!    cannot resume past it) and is surfaced as a clear warning, keeping
//!    whatever applied before the failing line.
//!
//! 2. **Plugin-manager boilerplate degrades to nothing.** The maintainer's real
//!    config drives `lazy.nvim` and ~20 plugins, but kvim compiles its plugins
//!    IN — there is nothing to install. So `require("lazy").setup{...}`,
//!    `require("telescope")`, and friends must not blow up: an unknown module
//!    resolves to a black-hole stub, recorded in [`LuaRuntime::stubbed_plugins`].
//!    The handful of plugins kvim *does* implement natively (telescope, hop,
//!    harpoon, the LSP buffer functions) resolve instead to **marker** functions
//!    that [`vim.keymap.set`] recognises by identity and maps onto the right
//!    native [`Action`] — so `vim.keymap.set("n", "\\ff", builtin.find_files)`
//!    still lands on kvim's own fuzzy finder.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::Path;
use std::rc::Rc;

use kopitiam_lua::{Lua, LuaStr, Value};

use crate::config::{Action, Config, Keymap, Options};

mod excmd;
mod shim;

#[cfg(test)]
mod tests;

/// The Lua files kvim found on disk, ready to execute.
///
/// `init.lua` is the entry point (run directly); everything under `lua/` is a
/// library reachable through `require("<stem>")`, exactly as Neovim resolves
/// its own `runtimepath`.
#[derive(Debug, Default, Clone)]
pub struct Discovered {
    /// `init.lua`'s source, if it exists.
    pub init: Option<String>,
    /// `lua/<name>.lua`, keyed by module name (the file stem), so `require`
    /// can resolve them.
    pub modules: BTreeMap<String, String>,
}

impl Discovered {
    /// Reads `init.lua` and `lua/*.lua` out of a kvim config directory.
    ///
    /// Missing files are simply absent, not an error — a user need never write
    /// any Lua at all, because kvim's compiled-in defaults already *are* a full
    /// config (see [`Config::default`]).
    pub fn from_dir(dir: &Path) -> Self {
        let init = std::fs::read_to_string(dir.join("init.lua")).ok();

        let mut modules = BTreeMap::new();
        if let Ok(entries) = std::fs::read_dir(dir.join("lua")) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) != Some("lua") {
                    continue;
                }
                if let (Some(stem), Ok(src)) =
                    (path.file_stem().and_then(|s| s.to_str()), std::fs::read_to_string(&path))
                {
                    modules.insert(stem.to_string(), src);
                }
            }
        }

        Self { init, modules }
    }

    /// Reads the config out of kvim's real per-user directory
    /// (`~/.kopitiam/kopitiam-neovim/`). Empty when there is no home directory
    /// or no config dir.
    pub fn from_config_dir() -> Self {
        match Config::dir() {
            Some(dir) => Self::from_dir(&dir),
            None => Self::default(),
        }
    }

    /// True when there is nothing to run — no `init.lua` and no modules. In that
    /// case kvim keeps its compiled-in defaults untouched.
    pub fn is_empty(&self) -> bool {
        self.init.is_none() && self.modules.is_empty()
    }
}

/// An autocommand the config registered. kvim does not fire these yet (it has
/// no general event bus — see the follow-up bead), so they are recorded and
/// reported rather than silently dropped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Autocmd {
    /// The events it fires on, e.g. `["BufNewFile", "BufRead"]`.
    pub events: Vec<String>,
    /// The file pattern, e.g. `*.tex`, or empty for "any".
    pub pattern: String,
    /// The command or a description of the callback.
    pub action: String,
}

/// A `:command`-style user command the config defined with
/// `vim.api.nvim_create_user_command`. Recorded, not yet dispatchable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserCommand {
    pub name: String,
    pub definition: String,
}

/// Everything the shim accumulates while a config runs. Shared, through an
/// `Rc<RefCell<_>>`, by every native `vim.*` function so they can all write into
/// the one growing picture of "what the config asked for".
#[derive(Default)]
pub(crate) struct VimState {
    /// The config being mutated in place. Starts as kvim's compiled-in default
    /// and every `vim.opt`/`vim.keymap.set`/`vim.cmd.colorscheme` edits it.
    config: Config,
    /// `vim.g.*` global variables, stringified. `mapleader`/`maplocalleader`
    /// additionally drive [`Config::leader`].
    globals: BTreeMap<String, String>,
    /// `vim.b.*` buffer variables. Recorded only — kvim has one config-time
    /// buffer, so there is nothing to scope them to yet.
    buffer_vars: BTreeMap<String, String>,
    /// Lua-function keymap right-hand sides, indexed by the id stored in
    /// [`Action::LuaKeymap`].
    keymap_callbacks: Vec<Value>,
    /// Autocommands registered via `vim.cmd("autocmd ...")` or
    /// `vim.api.nvim_create_autocmd`.
    autocmds: Vec<Autocmd>,
    /// User commands from `vim.api.nvim_create_user_command`.
    user_commands: Vec<UserCommand>,
    /// One-line notes about `vim.*` surface that was used but not (yet) applied:
    /// unknown options, unsupported API, highlight overrides, and so on.
    warnings: Vec<String>,
    /// Messages the config raised with `vim.notify`.
    notifications: Vec<String>,
    /// Plugin modules that were `require`d but resolved to a black-hole stub
    /// because kvim implements them natively (or not at all).
    stubbed_plugins: Vec<String>,
    /// Functions handed to `vim.schedule`, to run once the config finishes.
    scheduled: Vec<Value>,
}

impl VimState {
    fn warn(&mut self, msg: impl Into<String>) {
        let msg = msg.into();
        // De-duplicate: a config that touches `vim.notify` fifty times should
        // produce one warning, not fifty.
        if !self.warnings.contains(&msg) {
            self.warnings.push(msg);
        }
    }
}

/// A live Lua runtime, after the config has run.
///
/// It owns the VM and the shared [`VimState`], and it stays alive for the whole
/// editor session so that Lua-function keymaps (`vim.keymap.set("n", "x",
/// function() ... end)`) can be *fired* later, when their key is pressed — the
/// closure and its upvalues live in here, not in the serialisable [`Config`].
pub struct LuaRuntime {
    lua: Lua,
    state: Rc<RefCell<VimState>>,
}

impl LuaRuntime {
    /// Executes `discovered` on top of `base`, returning the resulting runtime.
    ///
    /// This never panics and never returns `Err`: a broken config degrades to a
    /// warning (see [`LuaRuntime::warnings`]) rather than refusing to start the
    /// editor. Call [`LuaRuntime::config`] afterwards for the merged config.
    pub fn load(base: Config, discovered: &Discovered) -> Self {
        let state = Rc::new(RefCell::new(VimState { config: base, ..Default::default() }));
        let mut lua = Lua::new();

        shim::install(&mut lua, &state, discovered);

        // Run init.lua if present; otherwise run the loose modules directly, so a
        // user who dropped a single `lua/settings.lua` without an `init.lua`
        // still gets it applied instead of silently ignored.
        if let Some(init) = &discovered.init {
            run_chunk(&mut lua, &state, init, "init.lua");
        } else {
            for (name, src) in &discovered.modules {
                run_chunk(&mut lua, &state, src, &format!("{name}.lua"));
            }
        }

        // Anything the config deferred with `vim.schedule` runs now, once the
        // config proper has finished — which is the ordering guarantee schedule
        // exists to give.
        let mut rt = Self { lua, state };
        rt.run_scheduled();
        rt
    }

    /// The merged configuration: kvim's defaults with the Lua config applied.
    pub fn config(&self) -> Config {
        self.state.borrow().config.clone()
    }

    /// One-line warnings about `vim.*` surface that ran but was not fully
    /// applied (unknown options, unsupported API, stubbed plugins). Shown once
    /// at startup so the user knows what their config asked for and did not get.
    pub fn warnings(&self) -> Vec<String> {
        self.state.borrow().warnings.clone()
    }

    /// Messages the config raised with `vim.notify`.
    pub fn notifications(&self) -> Vec<String> {
        self.state.borrow().notifications.clone()
    }

    /// Plugin modules that resolved to a black-hole stub — the plugin-manager
    /// boilerplate kvim deliberately no-ops because its plugins are built in.
    pub fn stubbed_plugins(&self) -> Vec<String> {
        self.state.borrow().stubbed_plugins.clone()
    }

    /// Autocommands the config registered (recorded, not yet fired).
    pub fn autocmds(&self) -> Vec<Autocmd> {
        self.state.borrow().autocmds.clone()
    }

    /// Fires the Lua closure bound to a keymap, identified by the id inside
    /// [`Action::LuaKeymap`]. Called when the mapped key is pressed.
    ///
    /// Returns any error the closure raised, as a string, so the caller can show
    /// it on the statusline rather than crash — a config bug in a keymap must
    /// not take the editor down.
    pub fn fire_keymap(&mut self, id: usize) -> std::result::Result<(), String> {
        // Clone the callable out before calling: `call` re-enters the VM, and
        // the native `vim.*` functions it may reach borrow `state`, so we must
        // not be holding a borrow across the call.
        let callback = self.state.borrow().keymap_callbacks.get(id).cloned();
        let Some(callback) = callback else {
            return Err(format!("no Lua keymap callback #{id}"));
        };
        self.lua.call(&callback, vec![]).map(|_| ()).map_err(|e| e.to_string())
    }

    /// Runs, and drains, everything queued by `vim.schedule`.
    pub fn run_scheduled(&mut self) {
        let queued: Vec<Value> = std::mem::take(&mut self.state.borrow_mut().scheduled);
        for f in queued {
            if let Err(e) = self.lua.call(&f, vec![]) {
                self.state.borrow_mut().warn(format!("vim.schedule callback failed: {e}"));
            }
        }
    }
}

/// Executes one Lua chunk, turning any failure into a recorded warning instead
/// of a panic or an aborted startup. Lua cannot resume past an error inside a
/// single chunk, so whatever ran before the failing line stays applied and the
/// rest of that file is lost — which is the honest, least-surprising outcome.
fn run_chunk(lua: &mut Lua, state: &Rc<RefCell<VimState>>, source: &str, name: &str) {
    if let Err(e) = lua.exec(source, &format!("@{name}")) {
        state.borrow_mut().warn(format!("{name} failed: {e}"));
    }
}

/// Coerces a Lua value into the string a comma-list option (`clipboard`,
/// `spelllang`) wants, or the boolean/number a scalar option wants. Kept here so
/// both the `vim.opt` shim and the `:set` ex handler agree on the rules.
fn lua_to_string(lua: &mut Lua, v: &Value) -> String {
    lua.tostring(v).map(|s| s.to_string_lossy()).unwrap_or_default()
}

/// Applies one option assignment (`name = value`) onto [`Options`], with vim's
/// per-option typing. Unknown option names are not an error — they are recorded
/// as a warning and ignored, because a config that sets an option kvim does not
/// model yet should not fail to load over it.
///
/// Returns `Err(name)` for an unknown option so the caller can phrase the
/// warning with the surrounding context (`vim.opt.x` vs `:set x`).
fn apply_option(opts: &mut Options, name: &str, value: &Value, as_str: &str) -> Result<(), ()> {
    // vim treats a bare `set foo` as `foo = true` and `set nofoo` as false; the
    // shim passes an explicit boolean for `vim.opt.foo = true`, so `is_truthy`
    // is the right test for every boolean option.
    let b = value.is_truthy();
    let n = value.as_number().map(|n| n as usize);
    match name {
        "number" | "nu" => opts.number = b,
        "relativenumber" | "rnu" => opts.relativenumber = b,
        "wrap" => opts.wrap = b,
        "spell" => opts.spell = b,
        "syntax" => opts.syntax = b,
        "expandtab" | "et" => opts.expandtab = b,
        "hlsearch" | "hls" => opts.hlsearch = b,
        "incsearch" | "is" => opts.incsearch = b,
        "ignorecase" | "ic" => opts.ignorecase = b,
        "smartcase" | "scs" => opts.smartcase = b,
        "tabstop" | "ts" => opts.tabstop = n.unwrap_or(opts.tabstop),
        "shiftwidth" | "sw" => {
            opts.shiftwidth = crate::config::bool_or_usize::ShiftWidth(n.unwrap_or(0))
        }
        "scrolloff" | "so" => opts.scrolloff = n.unwrap_or(opts.scrolloff),
        "spelllang" | "spl" => opts.spelllang = as_str.to_string(),
        "clipboard" | "cb" => opts.clipboard = as_str.to_string(),
        "colorcolumn" | "cc" => {
            // "" clears the guide; a bare number or "75" sets it. vim also
            // accepts a comma list and relative "+1" forms — kvim models a single
            // absolute column, so it takes the first parseable absolute value.
            opts.colorcolumn = as_str
                .split(',')
                .find_map(|part| part.trim().parse::<usize>().ok());
        }
        "background" | "bg" => {
            opts.background = match as_str {
                "light" => crate::config::Background::Light,
                _ => crate::config::Background::Dark,
            }
        }
        _ => return Err(()),
    }
    Ok(())
}

/// Resolves a keymap right-hand side into an [`Action`].
///
/// Precedence mirrors what the config author means:
/// 1. a **marker** function kvim handed out for a native plugin → that plugin's
///    native `Action` (so `builtin.find_files` becomes [`Action::FindFiles`]);
/// 2. any other Lua **function** → stored in the callback registry and bound as
///    [`Action::LuaKeymap`];
/// 3. a **string** starting `<cmd>`/`:` → [`Action::Command`] (an ex command),
///    with `<cmd>...<cr>` unwrapped; a string starting `<Plug>` is dropped with
///    a warning (plugin key-routing kvim does not model);
/// 4. any other **string** → [`Action::FeedKeys`] (typed as keys verbatim).
fn resolve_rhs(
    state: &Rc<RefCell<VimState>>,
    markers: &[(Value, Action)],
    rhs: &Value,
) -> Action {
    for (marker, action) in markers {
        if rhs.raw_eq(marker) {
            return action.clone();
        }
    }
    match rhs {
        Value::Function(_) | Value::Native(_) => {
            let mut st = state.borrow_mut();
            let id = st.keymap_callbacks.len();
            st.keymap_callbacks.push(rhs.clone());
            Action::LuaKeymap(id)
        }
        _ => {
            let s = rhs.as_lua_string().map(|s| s.to_string_lossy()).unwrap_or_default();
            classify_rhs_string(&s)
        }
    }
}

/// Turns a string keymap right-hand side into an [`Action`]. Split out from
/// [`resolve_rhs`] so the ex-command `nnoremap`/`nmap` path can reuse the exact
/// same classification a `vim.keymap.set` string goes through.
fn classify_rhs_string(s: &str) -> Action {
    let trimmed = s.trim();
    // `<cmd>Neotree toggle<cr>` and `:Neotree toggle<cr>` both mean "run this ex
    // command"; strip the wrapper and hand the bare command to the ex layer,
    // which already knows `Neotree`, `Telescope`, and the rest.
    let lower = trimmed.to_ascii_lowercase();
    if let Some(rest) = lower.strip_prefix("<cmd>") {
        let end = rest.rfind("<cr>").unwrap_or(rest.len());
        let cmd = &trimmed[5..5 + end];
        return excmd::command_action(cmd);
    }
    if let Some(rest) = trimmed.strip_prefix(':') {
        let cmd = rest.strip_suffix("<CR>").or_else(|| rest.strip_suffix("<cr>")).unwrap_or(rest);
        return excmd::command_action(cmd.trim());
    }
    Action::FeedKeys(trimmed.to_string())
}

/// Records a keymap into the config, replacing any existing binding for the same
/// mode+lhs (a later `vim.keymap.set` on the same key wins, as in Neovim).
fn record_keymap(state: &Rc<RefCell<VimState>>, mode: String, lhs: String, action: Action, desc: String) {
    let mut st = state.borrow_mut();
    st.config.keymaps.retain(|k| !(k.mode == mode && k.lhs == lhs));
    st.config.keymaps.push(Keymap { mode, lhs, action, desc });
}

/// Helper: pull a string field out of a Lua opts table (`{ desc = "..." }`).
fn opts_string(v: Option<&Value>, key: &str) -> String {
    match v {
        Some(Value::Table(t)) => match t.borrow().raw_get_str(key) {
            Value::String(s) => s.to_string_lossy(),
            other => other.as_lua_string().map(|s| s.to_string_lossy()).unwrap_or_default(),
        },
        _ => String::new(),
    }
}

/// The Lua `vim` mode-letter argument (`"n"`, `""`, or a table of modes) reduced
/// to kvim's single mode string. A table of modes (`{ "n", "v" }`) collapses to
/// its first entry for now; multi-mode maps are a follow-up.
fn mode_arg(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.to_string_lossy(),
        Some(Value::Table(t)) => match t.borrow().array().first() {
            Some(Value::String(s)) => s.to_string_lossy(),
            _ => String::new(),
        },
        _ => String::new(),
    }
}

/// Convenience: make a `LuaStr`-carrying `Value` from `&str`.
fn lstr(s: &str) -> Value {
    Value::String(LuaStr::from(s))
}

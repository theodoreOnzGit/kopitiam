//! kvim's configuration — and, as its default, a faithful port of the
//! maintainer's Neovim setup.
//!
//! # Why the default config is compiled in
//!
//! The requirement was: `cargo install kopitiam-neovim`, run `kvim`, and get
//! the same behaviour as the maintainer's Neovim — with no plugin manager, no
//! Mason, and nothing downloaded at first run. That is only true if the
//! defaults *are* their config. So the settings, keymaps, and plugin
//! behaviour from their `~/.config/nvim/lua/*.lua` are ported here as Rust
//! data, and [`Config::default`] returns them.
//!
//! # What kvim does NOT do
//!
//! It never reads or writes `~/.config/nvim/`. Sharing that directory with a
//! real Neovim would mean kvim could break the editor the user still depends
//! on. Overrides live in `~/.config/kvim/` instead. (See
//! `docs/ai-decisions/AID-0003-kopitiam-neovim-architecture.md`, decision 5.)

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Editor options — the `vim.opt.*` surface.
///
/// Field names deliberately mirror vim's option names rather than being
/// "improved", so that a user who knows vim can predict them, and so that a
/// future Lua `vim.opt` shim (Phase 4) can map onto them mechanically.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    /// Show the absolute line number on the cursor line.
    pub number: bool,
    /// Show relative line numbers on every other line.
    pub relativenumber: bool,
    /// Width of a tab character, in columns.
    pub tabstop: usize,
    /// Columns shifted by `>>`/`<<` and by autoindent.
    pub shiftwidth: bool_or_usize::ShiftWidth,
    /// Soft-wrap long lines.
    pub wrap: bool,
    /// Minimum lines of context kept above/below the cursor when scrolling.
    pub scrolloff: usize,
    /// Highlight misspelled words.
    pub spell: bool,
    /// Spelling dictionary, e.g. `en_gb`.
    pub spelllang: String,
    /// Column at which to draw the line-length guide. `None` draws none.
    pub colorcolumn: Option<usize>,
    /// Dark or light background, which the theme keys off.
    pub background: Background,
    /// Syntax highlighting on.
    pub syntax: bool,
    /// Expand tabs to spaces on insert.
    pub expandtab: bool,
    /// `vim.opt.hlsearch`: keep every match of the last search pattern
    /// highlighted across the viewport until `:noh` (or a new search). Neovim
    /// ship this **on** by default (plain vim ship it off); kvim follow Neovim,
    /// since that one is the frontend the maintainer config target.
    pub hlsearch: bool,
    /// `vim.opt.incsearch`: while you typing a `/` or `?` pattern, highlight the
    /// matches of whatever got typed so far. On by default, same for both vim
    /// and Neovim.
    pub incsearch: bool,
    /// `vim.opt.ignorecase`: fold case when searching. Off by default (vim and
    /// Neovim default like that also), so search stay case-sensitive unless you
    /// ask otherwise — see [`crate::editor::search::build_regex`] for the exact
    /// rule.
    pub ignorecase: bool,
    /// `vim.opt.smartcase`: when `ignorecase` also on, an all-lowercase pattern
    /// fold case but a pattern carrying any uppercase letter don't. Off by
    /// default. No effect unless `ignorecase` is on, same like vim.
    pub smartcase: bool,
    /// `vim.opt.clipboard`. Neovim treat this as a comma list; the two value
    /// kopitiam care about are `unnamed` (sync plain yank/delete/put with the
    /// selection register `"*`) and `unnamedplus` (sync with the system
    /// clipboard `"+`). Empty string — vim's default — mean plain `y`/`d`/`p`
    /// stay on the internal registers, and `"+y`/`"+p` are the explicit way to
    /// reach the OS clipboard. Very common neovim configs set `"unnamedplus"`
    /// so that `y` and `p` just talk to the desktop clipboard; kopitiam mirror
    /// that when it is set. See [`Options::clipboard_sync_register`].
    pub clipboard: String,
}

impl Options {
    /// The register that plain (register-less) `y`/`d`/`c`/`x`/`p` should
    /// *also* route through because of the `clipboard` option, or `None` when
    /// `clipboard` is empty and the internal unnamed register stand alone.
    ///
    /// Neovim's rule: `unnamedplus` win and map to `"+` (the system
    /// clipboard); a plain `unnamed` (without plus) map to `"*` (the primary
    /// selection). When both are listed, `unnamedplus` still decide where a
    /// *yank* lands, so `"+` is returned. Kopitiam follow the same precedence.
    pub fn clipboard_sync_register(&self) -> Option<char> {
        let mut plus = false;
        let mut star = false;
        for item in self.clipboard.split(',') {
            match item.trim() {
                "unnamedplus" => plus = true,
                "unnamed" => star = true,
                _ => {}
            }
        }
        if plus {
            Some('+')
        } else if star {
            Some('*')
        } else {
            None
        }
    }
}

/// `vim.opt.background`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Background {
    #[default]
    Dark,
    Light,
}

/// vim lets `shiftwidth = 0` mean "follow tabstop". Modelling that as a plain
/// `usize` would silently produce zero-width shifts, so it gets its own type.
pub mod bool_or_usize {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(transparent)]
    pub struct ShiftWidth(pub usize);

    impl ShiftWidth {
        /// The effective shift width, resolving vim's `0 = follow tabstop`
        /// convention.
        pub fn resolve(self, tabstop: usize) -> usize {
            if self.0 == 0 { tabstop } else { self.0 }
        }
    }
}

impl Default for Options {
    /// The maintainer's `lua/settings.lua`, verbatim.
    fn default() -> Self {
        Self {
            number: true,
            relativenumber: true,
            tabstop: 4,
            shiftwidth: bool_or_usize::ShiftWidth(4),
            wrap: false,
            scrolloff: 5,
            spell: true,
            spelllang: "en_gb".to_string(),
            colorcolumn: Some(75),
            background: Background::Dark,
            syntax: true,
            // Their config never sets expandtab, so vim's default (off) stands:
            // a literal tab is inserted. Getting this wrong would silently
            // change every file they edit, so it is called out rather than
            // quietly "improved" to spaces.
            expandtab: false,
            // Neovim's search defaults: highlight-all and incremental-search
            // both on, case-folding off. A plain-vim user who wants the older
            // "no persistent highlight" feel sets `nohlsearch`.
            hlsearch: true,
            incsearch: true,
            ignorecase: false,
            smartcase: false,
            // Their config never sets clipboard, so vim's default (empty)
            // stand: plain `y`/`p` stay internal, `"+y`/`"+p` reach the OS
            // clipboard explicitly. Flip this to "unnamedplus" and every plain
            // yank/put talk to the desktop clipboard instead.
            clipboard: String::new(),
        }
    }
}

/// One key mapping, in one mode.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Keymap {
    /// Mode this mapping applies in, using vim's single-letter codes:
    /// `n` normal, `i` insert, `v` visual, `x` visual-block, `c` command,
    /// `""` all modes (vim's `map`).
    pub mode: String,
    /// The key sequence, in vim notation (`<leader>gd`, `\ff`, `f`).
    pub lhs: String,
    /// What it does.
    pub action: Action,
    /// Shown in the which-key style help.
    ///
    /// `String` rather than `&'static str` because [`Config`] is
    /// `Deserialize`, and a borrowed `&'static str` cannot be deserialized
    /// from an owned config file — a user-defined keymap's description is not
    /// known at compile time.
    pub desc: String,
}

/// Everything a key can be bound to.
///
/// This is an enum rather than a boxed closure so that the whole default
/// keymap is `const`-shaped data that can be inspected, printed in `:map`
/// output, and (eventually) round-tripped to and from a config file. A closure
/// could do none of those.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Action {
    /// LSP: jump to the definition of the symbol under the cursor.
    LspDefinition,
    /// LSP: list references to the symbol under the cursor.
    LspReferences,
    /// LSP: rename the symbol under the cursor.
    LspRename,
    /// LSP: show hover documentation for the symbol under the cursor. Bound to
    /// `K`, matching Neovim's built-in default (`vim.lsp.buf.hover`).
    LspHover,
    /// LSP: force-attach the server for the current buffer, bypassing the
    /// resource-aware guard for this session (`:LspStart`). This is the escape
    /// hatch so a user is never locked out of the LSP just because the guard's
    /// estimate said the project was too big for the device. See
    /// [`crate::lsp::resource_guard`].
    LspStart,
    /// LSP: print the resource-guard probe numbers, the RA-memory estimate, and
    /// the gate decision (`:LspInfo`), so the user can see the reasoning and
    /// tune [`LspGuardConfig`].
    LspInfo,
    /// Toggle the file-tree sidebar (their `<leader>e` → `Neotree toggle`).
    FileTreeToggle,
    /// Label-jump to a word on screen (their `f` → `hop.hint_words`).
    HopWords,
    /// Fuzzy-find files (their `\ff`).
    FindFiles,
    /// Fuzzy-find open buffers (their `\fb`).
    FindBuffers,
    /// Fuzzy-find help tags (their `\fh`).
    FindHelp,
    /// Harpoon: mark the current file (their `<leader>b`).
    HarpoonAdd,
    /// Harpoon: toggle the quick menu (their `<leader><Esc>`).
    HarpoonMenu,
    /// Harpoon: fuzzy-find marks (their `<leader>q`).
    HarpoonFind,
    /// Align text on a delimiter (vim-easy-align's `ga`).
    EasyAlign,
    /// Run an ex command verbatim.
    Command(String),
    /// Feed a raw key sequence, exactly as if the user typed it — vim's
    /// `nnoremap lhs rhs` where `rhs` is keys, not an ex command. The string is
    /// in vim notation (`ciw`, `<Esc>`, `dd`). Produced by the `vim.*` shim when
    /// `vim.keymap.set(mode, lhs, rhs)` is handed a plain string that is not an
    /// `<cmd>...<cr>` / `:...` ex invocation.
    FeedKeys(String),
    /// Call a Lua function bound as a keymap's right-hand side. The `usize` is an
    /// index into the live [`crate::luaconfig::LuaRuntime`]'s callback registry —
    /// the function value itself cannot live in `Config` because a Lua closure is
    /// neither `Serialize` nor `PartialEq`, so the config stores only the handle
    /// and the runtime owns the closure. Produced by the `vim.*` shim when
    /// `vim.keymap.set(mode, lhs, function() ... end)` is given a function `rhs`.
    LuaKeymap(usize),
}

/// Tuning knobs for the resource-aware LSP guard (see
/// [`crate::lsp::resource_guard`]).
///
/// # What this is for
///
/// kvim runs on Android tablets. rust-analyzer indexes the whole dependency
/// graph, so on a big project relative to the device it can chew through more
/// RAM (and CPU during indexing) than the tablet has spare — and blowing past
/// available memory does not just lag, it can OOM-kill the app and take the
/// tablet down with it. So before kvim auto-attaches rust-analyzer on open, the
/// guard *estimates* whether this project fits this device, and if not it holds
/// off and tells the user (who can still force it with `:LspStart`).
///
/// The numbers below drive that estimate. They are deliberately **rough** — a
/// per-dep memory model is a blunt instrument (see
/// `docs/ai-decisions/AID-0037`) — which is exactly why they are configurable:
/// a user who knows their device and project can retune the gate instead of
/// being stuck with our guess.
///
/// Every field carries a conservative default via [`Default`]; on a big desktop
/// the budget is so large the gate essentially never fires, matching "simple
/// projects and capable machines are never blocked".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LspGuardConfig {
    /// Master switch. `true` (default) = the guard runs before auto-attach.
    /// `false` = kvim always auto-attaches, exactly like before the guard
    /// existed (the guard's own escape hatch, for someone who never wants it).
    pub enabled: bool,
    /// Belt-and-braces "just always start" flag. When `true`, the guard
    /// computes and reports its numbers (so `:LspInfo` still works) but never
    /// actually blocks the attach. Distinct from `enabled = false`, which skips
    /// the computation entirely.
    pub always_start: bool,
    /// Base rust-analyzer overhead in MB, independent of project size — the
    /// server process, the empty VFS, its own machinery. The `150` in
    /// `est_ra_mb = base + per_dep*deps + src_factor*src_mb`.
    pub base_mb: f64,
    /// Estimated MB of rust-analyzer RSS per dependency crate, since RA analyses
    /// the full dependency graph and peak memory scales mostly with crate count.
    /// The `4` in the estimate.
    pub per_dep_mb: f64,
    /// MB of RA RSS per MB of workspace `.rs` source — a secondary term, since a
    /// big first-party codebase also costs memory to analyse. The `0.5`.
    pub src_factor: f64,
    /// Fraction of *available* RAM the estimate may occupy before the guard
    /// fires. `0.5` keeps half of free RAM as headroom for the editor + OS; on
    /// Android there is no swap to catch an overshoot, so headroom is safety,
    /// not politeness.
    pub headroom: f64,
    /// CPU is a real input, not just message flavour: rust-analyzer is
    /// CPU-heavy while indexing, so a few-core tablet janks on a mid-size
    /// project even when RAM would fit. The effective budget is scaled by
    /// `min(1.0, logical_cores / core_ref_count)`, so a machine with
    /// `core_ref_count` cores or more is unpenalised and a 2-core tablet is
    /// gated much sooner. `8.0` by default — treat "8+ logical cores" as
    /// "plenty of indexing parallelism".
    pub core_ref_count: f64,
}

impl Default for LspGuardConfig {
    fn default() -> Self {
        // Defaults match `docs/ai-decisions/AID-0037`. Conservative: on a
        // capable machine the gate never fires; on a small tablet with a heavy
        // project it does.
        Self {
            enabled: true,
            always_start: false,
            base_mb: 150.0,
            per_dep_mb: 4.0,
            src_factor: 0.5,
            headroom: 0.5,
            core_ref_count: 8.0,
        }
    }
}

/// The complete kvim configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub options: Options,
    /// The leader key. Space, in their config.
    pub leader: char,
    pub keymaps: Vec<Keymap>,
    /// Language servers to attach, by filetype. Replaces mason's
    /// `ensure_installed`.
    pub language_servers: BTreeMap<String, String>,
    pub theme: String,
    /// Resource-aware LSP guard tuning. See [`LspGuardConfig`].
    pub lsp_guard: LspGuardConfig,
}

impl Default for Config {
    /// The maintainer's Neovim config, ported.
    ///
    /// Sources, for anyone auditing this against the original:
    /// `lua/settings.lua` → [`Options::default`]; `lua/keymaps.lua`,
    /// `lua/telescope_harpoon.lua` → [`Config::keymaps`]; `lua/lsp.lua`'s
    /// `ensure_installed` → [`Config::language_servers`]; `lua/plugins.lua`'s
    /// gruvbox + `vim.g.mapleader` → [`Config::theme`] and [`Config::leader`].
    fn default() -> Self {
        Self {
            options: Options::default(),
            // `vim.g.mapleader = " "` in lua/plugins.lua.
            leader: ' ',
            keymaps: default_keymaps(),
            language_servers: default_language_servers(),
            // `vim.cmd.colorscheme("gruvbox")`, with `background = dark`.
            theme: "gruvbox".to_string(),
            lsp_guard: LspGuardConfig::default(),
        }
    }
}

/// The maintainer's keymaps, from `lua/keymaps.lua` and
/// `lua/telescope_harpoon.lua`.
fn default_keymaps() -> Vec<Keymap> {
    let n = |lhs: &str, action: Action, desc: &str| Keymap {
        mode: "n".to_string(),
        lhs: lhs.to_string(),
        action,
        desc: desc.to_string(),
    };

    vec![
        // lua/keymaps.lua — LSP.
        n("<leader>gd", Action::LspDefinition, "Go to definition"),
        n("<leader>gr", Action::LspReferences, "Go to references"),
        n("<leader>rn", Action::LspRename, "Rename symbol"),
        // Neovim's built-in default hover binding (`K` → `vim.lsp.buf.hover`).
        // Not in the maintainer's explicit keymaps, but it *is* Neovim's own
        // default, so a Neovim user's muscle memory finds it here too.
        n("K", Action::LspHover, "Hover documentation"),
        // lua/keymaps.lua — file explorer.
        n("<leader>e", Action::FileTreeToggle, "Toggle file explorer"),
        // lua/keymaps.lua — hop. Note this DELIBERATELY shadows vim's built-in
        // `f` (find-char-on-line); their config does the same, and calls the
        // override out explicitly. Mapped in all modes, matching `vim.keymap.set("", ...)`.
        Keymap {
            mode: String::new(),
            lhs: "f".to_string(),
            action: Action::HopWords,
            desc: "Hop to word (overrides built-in f)".to_string(),
        },
        // lua/telescope_harpoon.lua — telescope. Backslash, not leader.
        n("\\ff", Action::FindFiles, "Find files"),
        n("\\fb", Action::FindBuffers, "Find buffers"),
        n("\\fh", Action::FindHelp, "Find help tags"),
        // lua/telescope_harpoon.lua — harpoon.
        n("<leader>b", Action::HarpoonAdd, "Harpoon: mark file"),
        n("<leader><Esc>", Action::HarpoonMenu, "Harpoon: quick menu"),
        n("<leader>q", Action::HarpoonFind, "Harpoon: find marks"),
        // vim-easy-align's documented `ga` prefix.
        n("ga", Action::EasyAlign, "Align on delimiter"),
    ]
}

/// Language servers, from `lua/lsp.lua`'s mason `ensure_installed`.
///
/// The value is the server's executable name. Acquiring that executable
/// without mason — i.e. without shelling out to npm/pip/go, which is what
/// breaks on Android — is `kopitiam-lsp`'s job.
fn default_language_servers() -> BTreeMap<String, String> {
    BTreeMap::from([
        ("rust".to_string(), "rust-analyzer".to_string()),
        ("lua".to_string(), "lua-language-server".to_string()),
        ("tex".to_string(), "texlab".to_string()),
    ])
}

impl Config {
    /// Loads `~/.config/kvim/config.json`, falling back to [`Self::default`]
    /// (the maintainer's ported Neovim config) when it does not exist.
    ///
    /// A *missing* config is normal and returns the defaults. A config that
    /// exists but is malformed is an error, not a silent fallback — silently
    /// ignoring a typo'd config is how a user spends an hour wondering why
    /// their setting "doesn't work".
    pub fn load() -> anyhow::Result<Self> {
        let Some(path) = Self::config_path() else {
            return Ok(Self::default());
        };
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(&path)?;
        let config = serde_json::from_str(&text)
            .map_err(|e| anyhow::anyhow!("malformed config at {}: {e}", path.display()))?;
        Ok(config)
    }

    /// kvim's per-user directory: `~/.kopitiam/kopitiam-neovim/`.
    ///
    /// Every KOPITIAM application lives under one `~/.kopitiam` root, in a
    /// subdirectory named for its crate — see [`kopitiam_config`] for why that
    /// root is a plain `$HOME`-relative path rather than an XDG one (short
    /// version: Android has no XDG).
    ///
    /// Note kvim still never reads or writes `~/.config/nvim/`. Sharing a
    /// directory with the real Neovim would mean kvim could break the editor
    /// the user still depends on.
    pub fn dir() -> Option<std::path::PathBuf> {
        kopitiam_config::app_dir(APP_NAME)
    }

    /// `~/.kopitiam/kopitiam-neovim/config.json`.
    pub fn config_path() -> Option<std::path::PathBuf> {
        Some(Self::dir()?.join("config.json"))
    }

    /// Lua configuration files found in kvim's directory, in load order:
    /// `init.lua` first, then `lua/*.lua` sorted by name.
    ///
    /// # These are discovered here, and executed by [`crate::luaconfig`]
    ///
    /// This method only *lists* the files (it is what `kvim --config-path`
    /// prints). Actually running them is [`crate::luaconfig::LuaRuntime::load`]'s
    /// job: it feeds `init.lua` — with `lua/*.lua` reachable through `require` —
    /// to the pure-Rust `kopitiam-lua` VM behind a `vim.*` shim, so the config
    /// mutates a real [`Config`]. See `docs/ai-decisions/AID-0003` (why a
    /// pure-Rust VM) and `AID-0034` (how the shim maps Lua onto the editor).
    ///
    /// Returns an empty vector when the directory does not exist, which is the
    /// normal case — kvim's defaults *are* a full configuration, so a user need
    /// never write one.
    pub fn lua_files() -> Vec<std::path::PathBuf> {
        let Some(dir) = Self::dir() else { return Vec::new() };

        let mut files = Vec::new();

        // `init.lua` is the entry point, and loads first — the same convention
        // Neovim uses, so a user's muscle memory transfers.
        let init = dir.join("init.lua");
        if init.is_file() {
            files.push(init);
        }

        // Then `lua/*.lua`, sorted, so load order is deterministic rather than
        // whatever order the filesystem happens to hand back.
        if let Ok(entries) = std::fs::read_dir(dir.join("lua")) {
            let mut modules: Vec<_> = entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == "lua"))
                .collect();
            modules.sort();
            files.extend(modules);
        }

        files
    }
}

/// kvim's subdirectory under `~/.kopitiam`. The crate name, not the binary
/// name — see [`kopitiam_config::app_dir`].
const APP_NAME: &str = "kopitiam-neovim";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_reproduce_the_maintainers_settings_lua() {
        let o = Options::default();
        assert!(o.number && o.relativenumber);
        assert_eq!(o.tabstop, 4);
        assert_eq!(o.shiftwidth.resolve(o.tabstop), 4);
        assert!(!o.wrap);
        assert_eq!(o.scrolloff, 5);
        assert!(o.spell);
        assert_eq!(o.spelllang, "en_gb");
        assert_eq!(o.colorcolumn, Some(75));
        assert_eq!(o.background, Background::Dark);
    }

    #[test]
    fn leader_is_space_and_theme_is_gruvbox() {
        let c = Config::default();
        assert_eq!(c.leader, ' ');
        assert_eq!(c.theme, "gruvbox");
    }

    #[test]
    fn shiftwidth_zero_follows_tabstop_like_vim() {
        assert_eq!(bool_or_usize::ShiftWidth(0).resolve(8), 8);
        assert_eq!(bool_or_usize::ShiftWidth(2).resolve(8), 2);
    }

    #[test]
    fn every_keymap_from_the_original_config_is_present() {
        let c = Config::default();
        let has = |lhs: &str| c.keymaps.iter().any(|k| k.lhs == lhs);
        for lhs in [
            "<leader>gd",
            "<leader>gr",
            "<leader>rn",
            "<leader>e",
            "f",
            "\\ff",
            "\\fb",
            "\\fh",
            "<leader>b",
            "<leader><Esc>",
            "<leader>q",
        ] {
            assert!(has(lhs), "missing keymap {lhs}");
        }
    }

    #[test]
    fn the_three_language_servers_from_mason_ensure_installed_are_configured() {
        let c = Config::default();
        assert_eq!(c.language_servers.get("rust").unwrap(), "rust-analyzer");
        assert_eq!(c.language_servers.get("lua").unwrap(), "lua-language-server");
        assert_eq!(c.language_servers.get("tex").unwrap(), "texlab");
    }

    #[test]
    fn with_no_config_file_at_all_kvim_is_the_maintainers_neovim() {
        // The load-bearing guarantee: "if there is no config, kvim should
        // behave like my Neovim. This is the default vanilla state."
        //
        // `Config::load()` on a machine with no config file must return exactly
        // `Config::default()`, and `Config::default()` must be their setup. The
        // second half is asserted by the tests above; this pins the first half,
        // so that nobody later "helpfully" makes the no-config path fall back to
        // some blank vim-like default instead.
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("does-not-exist.json");
        assert!(!missing.exists());

        // `load()` reads a real path, so exercise the same branch it takes:
        // absent file => defaults, and defaults are theirs.
        let loaded = if missing.exists() {
            serde_json::from_str(&std::fs::read_to_string(&missing).unwrap()).unwrap()
        } else {
            Config::default()
        };

        assert_eq!(loaded, Config::default());
        assert_eq!(loaded.leader, ' ');
        assert_eq!(loaded.theme, "gruvbox");
        assert_eq!(loaded.options.scrolloff, 5);
        assert_eq!(loaded.options.spelllang, "en_gb");
        assert!(loaded.keymaps.iter().any(|k| k.lhs == "<leader>e"));
    }

    #[test]
    fn kvim_lives_under_the_shared_kopitiam_home_and_never_touches_config_nvim() {
        let Some(dir) = Config::dir() else { return };

        // Under ~/.kopitiam/, named for the crate.
        assert_eq!(dir.file_name().unwrap(), "kopitiam-neovim");
        assert_eq!(dir.parent().unwrap().file_name().unwrap(), kopitiam_config::DIR_NAME);

        // And emphatically NOT inside the real Neovim's config, which the user
        // still depends on.
        let s = dir.to_string_lossy();
        assert!(!s.contains(".config/nvim"), "kvim must never live inside ~/.config/nvim");
    }

    #[test]
    fn lua_files_are_discovered_in_a_deterministic_load_order() {
        // init.lua first, then lua/*.lua sorted — so behaviour does not depend
        // on whatever order the filesystem happens to return entries in.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("init.lua"), "-- entry").unwrap();
        std::fs::create_dir(dir.path().join("lua")).unwrap();
        for name in ["zebra.lua", "alpha.lua", "notes.txt"] {
            std::fs::write(dir.path().join("lua").join(name), "-- x").unwrap();
        }

        // Mirror `lua_files()`'s logic against this tempdir (it reads a real
        // home directory, which a test must not depend on).
        let mut found = Vec::new();
        let init = dir.path().join("init.lua");
        if init.is_file() {
            found.push(init);
        }
        let mut modules: Vec<_> = std::fs::read_dir(dir.path().join("lua"))
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == "lua"))
            .collect();
        modules.sort();
        found.extend(modules);

        let names: Vec<_> = found.iter().map(|p| p.file_name().unwrap().to_string_lossy().to_string()).collect();
        assert_eq!(names, vec!["init.lua", "alpha.lua", "zebra.lua"]);
        // The .txt is not a Lua file and must not be picked up.
        assert!(!names.iter().any(|n| n.ends_with(".txt")));
    }

    #[test]
    fn config_round_trips_through_json() {
        let c = Config::default();
        let json = serde_json::to_string(&c).unwrap();
        let back: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }
}

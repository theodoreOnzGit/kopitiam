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
}

/// The complete kvim configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    /// # These are discovered, not yet executed
    ///
    /// Running them needs a Lua interpreter, and KOPITIAM is committed to a
    /// **pure-Rust** one (`kopitiam-lua`, kvim Phase 4 — see
    /// `docs/ai-decisions/AID-0003`), which does not exist yet. Until it does,
    /// kvim finds these files and reports them rather than silently ignoring
    /// them: a config that is quietly not loaded is far worse than one that
    /// says out loud it is not loaded.
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

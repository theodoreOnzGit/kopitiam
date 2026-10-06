# AID-0060 — kvim drops its Lua config layer: preferences hardcoded only, released as 0.4.0

**Status:** Pending review
**Date:** 2026-10-06 (SGT)
**Issue:** gh-118 / `bd-8p3`
**Reverses:** [AID-0034](AID-0034-kvim-lua-vim-shim.md) (kvim executes `init.lua` through a `vim.*` shim)

## Context

The maintainer's direction, 2026-10-06, word for word:

> kopitiam-neovim shouldn't need to read lua, i want my preferences hardcoded in

The *direction* is the maintainer's own call, not an AI one — so this AID is not
asking whether to remove Lua. It records the smaller judgment calls made while
carrying it out, so the maintainer can check them.

What existed before: `crates/kopitiam-neovim/src/luaconfig/` (excmd.rs, mod.rs,
shim.rs, tests.rs — 1,800 lines). At startup, `ui::bootstrap::run` read
`~/.kopitiam/kopitiam-neovim/init.lua` (plus `lua/*.lua` through `require`), ran
it in a `kopitiam-lua` VM behind a `vim.*` shim, and merged the result into
`Config`. `kvim --config-path` listed the Lua files and said they were
"EXECUTED at startup".

**The README was wrong about this, both before and after.** It said kvim finds
`init.lua` "but it does not run them yet ... `kopitiam-lua` ... has not landed".
Checked against the code: false since AID-0034 — the shim did run them. Struck
through and corrected in the README in the same change.

## What was removed

* the whole `luaconfig` module (`pub mod luaconfig` in `lib.rs`);
* `ui::bootstrap::apply_lua_config` and its test; the `App::lua` field,
  `App::set_lua_runtime`, `App::fire_lua_keymap`;
* `Config::lua_files` and its test; the Lua section of `kvim --config-path`
  (which now says plainly that kvim never reads or runs Lua);
* `Action::LuaKeymap(usize)` and `Action::FeedKeys(String)`;
* `App::set_startup_message`;
* the `kopitiam-lua` dependency of `kopitiam-neovim`, and `kopitiam-lua` from
  `scripts/publish-kvim.sh`'s crate list.

**Not** removed: the `kopitiam-lua` crate itself (still a workspace member,
still published, not yanked). After this change **no workspace crate depends on
it** — checked with `grep -rn kopitiam-lua crates/*/Cargo.toml apps/*/Cargo.toml`
(only its own manifest) and `grep -rn kopitiam_lua --include=*.rs` (only its own
`src/` and `tests/`). `crates/kopitiam-syntax/src/lua.rs` mentions it in a doc
comment only. Whether to keep it is a separate question for the maintainer.

## Decision 1 — what to hardcode: nothing new, because the Lua path never carried the maintainer's config

The worry was that some preference reached kvim *only* through Lua. It did not,
for a plain reason: the shim read `~/.kopitiam/kopitiam-neovim/init.lua`, and on
the maintainer's machine that directory holds no Lua at all (checked
2026-10-06: only `.tmux-autoconfig-declined`). kvim never read
`~/.config/nvim` (AID-0003 decision 5). So every preference kvim actually had
already came from `Config::default`.

Checked `Config::default` against the maintainer's real `~/.config/nvim`
(read-only) item by item:

| `~/.config/nvim` | kvim, already compiled in |
|---|---|
| `settings.lua`: number, relativenumber, tabstop 4, shiftwidth 4, nowrap, scrolloff 5, spell, spelllang en_gb, colorcolumn 75, background dark, `syntax on` | `Options::default` (test `defaults_reproduce_the_maintainers_settings_lua`) |
| `plugins.lua`: `mapleader = " "`, gruvbox | `Config::leader`, `Config::theme` |
| `keymaps.lua`: `<leader>gd/gr/rn/e`, `f` → hop in all modes | `default_keymaps()` (test `every_keymap_from_the_original_config_is_present`) |
| `telescope_harpoon.lua`: `\ff \fb \fh`, `<leader>b`, `<leader><Esc>`, `<leader>q` | `default_keymaps()` |
| `lsp.lua`: rust_analyzer, lua_ls, texlab; blink.cmp `<C-space> <C-e> <C-b> <C-f>`, `<CR>` confirm | `default_language_servers()`; completion keys native in `ui/app.rs` |
| `plugins.lua`: neo-tree NERDTree keys `o t i s O x X R q ? I u U P C` | `plugins/filetree.rs` |
| `settings.lua`: `autocmd BufNewFile,BufRead *.tex set filetype=tex` | extension-based filetype detection (`ui/app.rs`, `"tex" \| "sty" \| "cls"`) |
| `settings.lua`: `TermOpen` → `mouse = ""` | kvim never enables mouse capture (`ui/terminal.rs`), so moot |
| `init.lua`: `termguicolors` | kvim's theme paints 24-bit `Color::Rgb` always |
| `init.lua`: netrw disabled; `filetype plugin indent on`; airline theme | no netrw / plugin loader in kvim; statusline is native |

So **nothing was hardcoded in this change** — there was nothing only Lua could
supply. The two features that *only* the shim could express (binding a key to an
arbitrary Lua function, or to a raw key sequence) are not in the maintainer's
config at all.

## Decision 2 — `Action::FeedKeys` goes too, not just `LuaKeymap`

`LuaKeymap` was obviously Lua-only (an index into the Lua callback registry).
`FeedKeys` was less obvious: it is plain data and could in principle be written
in `config.json`. Removed anyway, because its only producer was the shim and
**nothing dispatched it** — `App::handle_action` sent it to the
`"... is not wired into the UI yet"` fallback. Keeping it would advertise a
config surface that does nothing. Likewise `App::set_startup_message` had no
caller left (it only surfaced Lua warnings).

**Alternative:** keep `FeedKeys` and wire it up as a native "feed these keys"
action. Rejected for this change — it is new behaviour, not removal, and nobody
asked for it.

## Decision 3 — a stray `init.lua` is ignored silently, not warned about

If a user puts `init.lua` in `~/.kopitiam/kopitiam-neovim/`, kvim 0.4.0 neither
runs it nor warns at startup. **Alternative:** keep a tiny discovery check and
show "kvim does not run Lua; this file is ignored". Rejected because the brief
was to remove init.lua discovery-and-report too, and "never reads Lua" is
simplest when it is literally true. The information still exists where a
confused user looks: `kvim --config-path`, `kvim --help` and `:help config`
(new topic, aliases include `init.lua` and `lua`) all say kvim never reads Lua.

## Decision 4 — version 0.4.0, and the CLI's requirement moves with it

Removing a public module (`luaconfig`), a public method (`Config::lua_files`,
`App::set_lua_runtime`, `App::set_startup_message`) and two variants of the
public exhaustive `config::Action` is SemVer-breaking, so `kopitiam-neovim`
goes **0.3.0 → 0.4.0** (0.x: minor = breaking). `apps/cli` (`kopitiam`) is the
only dependent; its requirement moves `"0.3.0"` → `"0.4.0"` so the workspace
resolves. The CLI only calls `Config::load` and `ui::run`, neither changed.
No other crate is bumped (CLAUDE.md "no new commits, no version bump").

## What would make this wrong

* **If the maintainer still wants *other users* to be able to script kvim.**
  The ask said "my preferences hardcoded in", which removes the need for the
  maintainer; it does not literally say no user may ever have a script layer.
  If a user-facing config language is wanted later, AID-0034's shim is in git
  history (last at `48b446c`) and could be restored — but `config.json` is
  the intended surface now.
* **If some preference only lived in a `~/.kopitiam/kopitiam-neovim/init.lua`
  on another machine** (Termux tablet, Windows box). Checked only this
  desktop. If such a file exists elsewhere, its contents need porting into
  `Config::default` as data.
* **If someone's `config.json` used `"FeedKeys"`.** It would now fail to parse
  (loud error, by design). It never did anything, so the loss is only the
  parse.
* **If `kopitiam-lua` should now be retired.** Nothing in the workspace uses it;
  keeping it costs build time. Not decided here.

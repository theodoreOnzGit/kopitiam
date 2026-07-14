# kvim

A Rust-native, Android-capable modal editor: Vim keybindings, an LSP client,
and a plugin suite, all compiled in. No plugin manager, no Mason, no Lua to
install at runtime.

```bash
cargo install kopitiam-neovim
kvim path/to/file.rs
```

kvim is part of [KOPITIAM](https://github.com/theodoreOnzGit/kopitiam), a
Rust scientific-computing workbench. Despite the leading `k`, it is
**unrelated to KDE/Plasma** — the `k` follows this project's own convention
(`kopitiam`, `kvim`, `kmux`), and the collision with KDE's `k*` naming is
coincidental.

## Why this exists

The maintainer's Neovim setup is about 20 Lua plugins managed by `lazy.nvim`,
with language servers installed by Mason. On Android that breaks — but not
because of Lua. Mason fails because it shells out to `npm`, `pip`, and
`go install` to fetch language servers, and those toolchains aren't there.
Lua itself cross-compiles to Android fine.

Fixing that properly means owning the whole stack rather than patching
around the shell-outs:

- **No plugin manager.** A plugin manager exists to install third-party
  downloads. When the plugins are compiled into the binary, there is nothing
  to install, so `lazy.nvim` has no job here.
- **No Mason.** Language servers are acquired as prebuilt static binaries
  over pure-Rust TLS, or vendored outright — never by shelling out to a
  package manager that may not exist on the target.
- **Batteries included, literally.** The Nerd Font kvim's devicons need
  ships *inside the binary*. An icon is a codepoint, and a codepoint without
  a matching font in the terminal is a tofu box — see [Devicons](#devicons)
  below.

## Install

```bash
cargo install kopitiam-neovim
```

Requires a stable Rust toolchain. This installs one binary, `kvim`; there is
no separate download step and nothing else to fetch to get an editor running.

```bash
kvim                  # open with an empty buffer
kvim src/main.rs      # open a file
kvim --help           # full flag list
kvim --version
```

### Devicons

Terminal emulators render icons the file tree and statusline want by looking
up a Private-Use-Area codepoint in whatever font is configured — kvim cannot
change that from inside a running process, it can only ship a font that has
the glyphs and offer to install it:

```bash
kvim --install-font
```

This writes a Nerd Font patched build of JetBrains Mono to the right place
for your platform (`~/.termux/font.ttf` on Termux, `~/.local/share/fonts/`
on Linux, `~/Library/Fonts/` on macOS), and tells you the one follow-up
command you need to run yourself (a font-cache refresh, or
`termux-reload-settings` on Termux — kvim never runs that one for you, since
on Termux it restarts the very session that would be running it).

Without the font, kvim still runs fine: it detects the terminal from the
environment and falls back to a `Unicode` icon tier (plain geometric glyphs,
any modern font) or a plain-`Ascii` tier (`[rs]`, `[md]`, ...) rather than
printing tofu. Force a tier explicitly with `kvim --icons nerd|unicode|ascii`
or `KVIM_ICONS=nerd|unicode|ascii`.

## Configuration

With **no config file at all**, kvim's defaults *are* the maintainer's
Neovim setup: hybrid line numbers, `tabstop`/`shiftwidth` 4, no line wrap,
`scrolloff` 5, spellcheck (`en_gb`), a colour column at 75, gruvbox (dark),
and leader = Space. `cargo install kopitiam-neovim && kvim` reproduces that
setup with nothing else to install or configure.

To see exactly what kvim finds on your machine:

```bash
kvim --config-path
```

Overrides live in `~/.kopitiam/kopitiam-neovim/config.json` — a JSON
serialization of the same `Options`/`Keymap`/`Action` structures the built-in
defaults are made of. A malformed config is a hard error at startup (not a
silent fallback), so a typo doesn't cost you an hour wondering why a setting
"isn't working."

kvim also looks for `init.lua` and `lua/*.lua` in that same directory and
will tell you if it finds them — but it does not run them yet. Lua config
execution needs a Lua interpreter, and KOPITIAM is committed to a pure-Rust
one (`kopitiam-lua`) that has not landed. Until it does, kvim reports the
files it found rather than silently ignoring them.

kvim **never reads or writes `~/.config/nvim/`.** That directory stays your
real Neovim's; kvim has its own directory under `~/.kopitiam/` so it can
never interfere with an editor you still depend on.

## The keymaps that matter

Leader is Space. These are the maintainer's own bindings, ported as data
rather than hand-copied, so they can't drift:

| Keys | Action |
|---|---|
| `<leader>gd` | LSP: go to definition |
| `<leader>gr` | LSP: list references |
| `<leader>rn` | LSP: rename symbol |
| `<leader>e` | Toggle the file-tree sidebar |
| `f` | Hop: label-jump to a word on screen (deliberately shadows vim's built-in `f`, exactly as the original Neovim config does) |
| `\ff` | Find files |
| `\fb` | Find buffers |
| `\fh` | Find help tags |
| `<leader>b` | Harpoon: mark the current file |
| `<leader><Esc>` | Harpoon: quick menu |
| `<leader>q` | Harpoon: find marks |
| `ga` | Align text on a delimiter |

These are compiled in as data (see [Status](#status--honesty) for which of
them have a working UI behind them today), and are the first thing to look
at if you want to override or extend the keymap in `config.json`.

## Editing

Below the keymap table, kvim implements the standard Vim editing grammar —
modes, motions, operators, text objects, counts, registers, macros, and
dot-repeat — the same way any vi-family editor does. If you already know
Vim, expect it to behave the way you expect; this README does not attempt to
enumerate every motion and operator, both because that grammar is exactly
Vim's own and because kvim's key handling is under active development and a
key-by-key table would be stale within the day.

## Status / honesty

kvim is early software. It is a genuinely usable modal text editor today —
opening files, editing with the Vim grammar described above, and the
`<leader>e` file tree all work end to end — but not everything the config
surface *describes* has a UI behind it yet. Keymaps that don't have one say
so out loud at runtime (`"... is not wired into the UI yet"`) rather than
silently doing nothing; if you bind or press something and see that message,
it means kvim understood the key and just hasn't built that feature's UI
yet, not that your config is wrong.

Roughly, as of this writing:

- **Working:** the Vim editing grammar (modes, motions, operators, text
  objects, registers, macros, ex commands), the file-tree sidebar
  (`<leader>e`), devicons and font installation, config loading/validation.
- **In progress:** the fuzzy finder (files/buffers/help), harpoon's UI, hop's
  label-jump UI, an LSP client, window splits and navigation, and more —
  this list changes quickly. See the project's issue tracker
  ([beads](https://github.com/theodoreOnzGit/kopitiam), search for
  `kopitiam-nvim` and `kopitiam-cj0`) for the current phase plan, and
  `docs/ai-decisions/AID-0003-kopitiam-neovim-architecture.md` in the main
  repository for the architecture behind it.

If something in this README turns out to be wrong for the version you
installed, please open an issue — outdated documentation is treated as a bug
in this project, not a rounding error.

See [`docs/quickstart.md`](docs/quickstart.md) for a longer, task-oriented
walkthrough.

## License

kvim is licensed [AGPL-3.0-only](https://www.gnu.org/licenses/agpl-3.0.html),
same as the rest of KOPITIAM.

The bundled font — JetBrains Mono Nerd Font Mono, Regular — is licensed
[OFL-1.1](https://scripts.sil.org/OFL) (SIL Open Font License). The OFL
governs the font as a distinct work; it does not affect the AGPLv3 licensing
of kvim itself, and its license text ships alongside the font both in this
repository and wherever `kvim --install-font` installs it.

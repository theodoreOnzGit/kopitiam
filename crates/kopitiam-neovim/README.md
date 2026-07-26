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
| `<C-h>` / `<C-j>` / `<C-k>` / `<C-l>` | Move window focus left / down / up / right (also crosses into the file tree, and hands off to the adjacent tmux pane at the layout edge — see [tmux integration](#tmux-integration)). In Insert mode `<C-h>` stays backspace and `<C-w>` stays delete-word, so these never shadow editing. |
| `<C-w>h` / `<C-w>j` / `<C-w>k` / `<C-w>l` | The same window-focus moves, vim's classic prefixed form (no tmux hand-off) |
| `:qa` / `:qa!` / `:wa` / `:wqa` / `:xa` | Quit-all / write-all across every split (`:qa` refuses on unsaved changes; `:qa!` force-quits; `:wqa`/`:xa` write every modified buffer then exit) |
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

## tmux integration

kvim use `<C-h/j/k/l>` to move focus between splits, and at the edge of its
own layout it hand off to the adjacent **tmux** pane (running `tmux
select-pane -L/-D/-U/-R`), the vim-tmux-navigator contract. For this to work
both direction, tmux must send those keys *through* to kvim instead of
grabbing them for its own pane navigation.

The catch: vim-tmux-navigator's tmux side decide whether to forward the keys
by running an `is_vim` check — a `grep` over the running process name against a
regex of vim-like editors (`vim`, `nvim`, `view`, `fzf`). That list **dun
include `kvim`**, so out of the box tmux dun recognise kvim, eats
`<C-h/j/k/l>` before kvim can see them, and kvim's split navigation go quietly
dead.

### kvim can fix this for you (with consent)

When kvim start up inside tmux, it check your `tmux.conf` for this exact
problem. If it find your `is_vim` regex dunno `kvim`, it pop up and *ask*
whether it may patch the conf for you — showing you the exact file and the
exact line(s) it will change **before** you say yes:

- press `y` → kvim back up your conf first (`tmux.conf.kvim-bak`), then make
  the minimal change: slot `kvim` into your existing `is_vim` regex, or — if
  you got no vim-tmux-navigator setup at all — append a fresh, commented block
  (and create `~/.config/tmux/tmux.conf` if you dun have one yet). After that
  kvim tell you the one thing you must run yourself: `tmux source-file <path>`
  (or just restart tmux). kvim **never** run that for you.
- press `n` → kvim leave your conf completely alone and dun ask again. To get
  the offer back, delete kvim's marker file at
  `~/.kopitiam/kopitiam-neovim/.tmux-autoconfig-declined`.

kvim look for your conf in the usual places, in order:
`$XDG_CONFIG_HOME/tmux/tmux.conf`, `~/.config/tmux/tmux.conf`, `~/.tmux.conf`.
It **never** touch anything without you pressing `y` first — same hard line as
kvim never writing `~/.config/nvim`.

If you inside `screen` or `zellij` instead, kvim just show a one-line note:
the pane hand-off is tmux-only, so there `<C-h/j/k/l>` only move kvim's own
splits.

### Doing it by hand

Prefer to edit it yourself? Add `kvim` to the `is_vim` regex — this is the
exact block kvim would add for you:

```tmux
# ── kvim / vim-tmux-navigator ──────────────────────────────────────────
# Let vim-like apps own <C-h/j/k/l> so their splits and tmux's panes
# navigate as one thing. The `kvim` in the regex is the load-bearing bit.
is_vim="ps -o state= -o comm= -t '#{pane_tty}' \
    | grep -iqE '^[^TXZ ]+ +(\\S+\\/)?g?(view|kvim|n?vim?x?|fzf)(diff)?$'"
bind-key -n 'C-h' if-shell "$is_vim" 'send-keys C-h' 'select-pane -L'
bind-key -n 'C-j' if-shell "$is_vim" 'send-keys C-j' 'select-pane -D'
bind-key -n 'C-k' if-shell "$is_vim" 'send-keys C-k' 'select-pane -U'
bind-key -n 'C-l' if-shell "$is_vim" 'send-keys C-l' 'select-pane -R'
# ───────────────────────────────────────────────────────────────────────
```

The load-bearing part is the `kvim` in the alternation: without it tmux treat
kvim as a non-vim program and eat `<C-h/j/k/l>` before kvim ever see them, so
intra-kvim split navigation silently stop working. The double backslashes
(`\\S`, `\\/`) are correct — inside the double-quoted `is_vim="..."` string
tmux collapse `\\` to `\`, so grep receive `\S`/`\/`. When kvim is *not* inside
tmux (`$TMUX` unset) the edge hand-off is simply a no-op, like plain vim.

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

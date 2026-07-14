# kvim quick start

A task-oriented walkthrough: install, open a file, edit it, browse a
project, configure kvim, and get devicons rendering. Every command below is
one you can copy and run as-is.

For the full picture of what kvim is and why it exists, see the
[README](../README.md) first if you haven't. This page assumes you've read
it and just want to get moving.

## 1. Install

kvim needs a stable Rust toolchain (install one from
[rustup.rs](https://rustup.rs) if you don't have one):

```bash
cargo install kopitiam-neovim
```

This builds and installs a single binary, `kvim`. Confirm it's there:

```bash
kvim --version
```

## 2. Open a file

```bash
kvim README.md
```

or with no argument, an empty buffer:

```bash
kvim
```

kvim starts in Normal mode, the way Vim does. `i` enters Insert mode,
`<Esc>` returns to Normal, `:w` writes, `:q` quits, `:wq` does both — the
same as Vim, because that's deliberately the grammar kvim implements. See
[Basic editing](#3-basic-editing) below for more, and the README's
[Editing](../README.md#editing) section for what that grammar covers.

## 3. Basic editing

kvim implements the standard Vim editing grammar: modes, motions
(`h j k l`, `w b e`, `0 ^ $`, `gg G`, and more), operators (`d c y` combined
with a motion or text object, e.g. `dw`, `ciw`, `yy`), counts (`3dd`),
registers, macros (`qa ... q`, `@a`), and dot-repeat (`.`). If you already
know Vim, none of this needs re-learning — it's the same grammar, not a
reinterpretation of it.

A few things worth trying on a real file to get a feel for it:

```
dw          " delete a word
ciw         " change the word under the cursor
3j          " move down 3 lines
yy p        " copy a line, paste it below
u           " undo
```

## 4. Browse a project with the file tree

Open the sidebar with `<leader>e` (leader is Space by default, so that's
`<Space>e`):

```
<Space>e
```

Navigate it like NERDTree/neo-tree: move with the usual motions, open a
file to load it into the buffer (the tree stays open), and `<Space>e` again
to close it. This is one of the parts of kvim that's fully wired end to
end today — see the README's [Status](../README.md#status--honesty) section
for what else is and isn't yet.

## 5. Configure kvim

Run this first to see exactly what kvim sees on your machine:

```bash
kvim --config-path
```

With no config file, kvim behaves like the maintainer's own Neovim setup —
hybrid line numbers, 4-space tabs, no wrap, gruvbox, spellcheck, leader =
Space, and the keymap table in the README. That's not a placeholder default;
it's a real, complete configuration, so you don't have to write one just to
get a working editor.

To override it, create `~/.kopitiam/kopitiam-neovim/config.json` (the exact
path `--config-path` just printed) with only the fields you want to change —
it's JSON, and every field has a sensible default if omitted:

```json
{
  "options": {
    "tabstop": 2,
    "shiftwidth": 2,
    "colorcolumn": 100
  },
  "theme": "gruvbox"
}
```

A malformed file is a hard error at startup, not a silent ignore — kvim
would rather tell you the JSON doesn't parse than run with settings you
didn't intend.

kvim will not read your existing `~/.config/nvim/`, and will never write to
it. If you're also running real Neovim on this machine, it's untouched.

## 6. Install the Nerd Font for devicons

File-type icons in the tree and statusline are Unicode Private-Use-Area
codepoints — they render as icons only when the terminal has a font that
contains those glyphs. Install the bundled one:

```bash
kvim --install-font
```

This writes a complete, Nerd-Font-patched build of JetBrains Mono to the
right location for your platform and tells you the one follow-up step to
run yourself (kvim writes the file but never edits your terminal emulator's
config, and on Termux it deliberately does not run the reload command for
you, since that would kill the very session it's running in).

Once your terminal is pointed at the font, start kvim with:

```bash
KVIM_ICONS=nerd kvim
```

If you skip this step, kvim doesn't break — it falls back to plain Unicode
glyphs (or pure ASCII on a bare console) automatically, rather than showing
tofu boxes. Force a tier explicitly at any time with `--icons
nerd|unicode|ascii` or the `KVIM_ICONS` environment variable.

## Where to go next

- [README.md](../README.md) — what kvim is, why it exists, and an honest
  account of what's implemented today versus in progress.
- `kvim --help` — the authoritative, always-current flag list.
- The main KOPITIAM repository's `docs/ai-decisions/AID-0003` and
  `AID-0004` for the architecture decisions behind kvim's design (Lua VM
  plans, the Mason replacement, and the devicons/font story).

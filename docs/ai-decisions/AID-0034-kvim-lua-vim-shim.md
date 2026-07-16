# AID-0034: kvim executes init.lua through a `vim.*` shim — how Lua maps onto the editor, and where it degrades

* **Status:** Pending review
* **Bead:** `kopitiam-cj0.11`
* **Date:** 2026-07-16
* **Decided by:** AI (Claude), maintainer absent

## The brief

> Make kvim actually EXECUTE the maintainer's real Neovim `init.lua` via
> KOPITIAM's pure-Rust Lua VM, through a `vim.*` API shim. On startup, if an
> `init.lua` (and `lua/*.lua`) is found, RUN it in a `kopitiam-lua` VM and map
> `vim.opt`/`vim.g`/`vim.keymap.set`/`vim.cmd`/`vim.api`/`vim.fn`/`vim.notify`/
> `vim.schedule`/`require` onto kvim's real editor state. Plugin-manager calls
> degrade gracefully; never crash on unsupported API.

AID-0003 already settled that the VM is pure Rust (`kopitiam-lua`); AID-0007
settled that it is a bytecode VM. This AID records the design calls that were
mine to make in building the *shim* on top of that VM.

## What was decided, and the judgment calls inside it

The shim is a new engine crate-module, `crates/kopitiam-neovim/src/luaconfig/`,
that runs the discovered config against a `vim` table of native Rust functions
which mutate a real `Config` (options, keymaps, leader, theme). Four calls were
genuinely mine:

### 1. The shim mutates `Config`; it does not fork the editor state model

`vim.opt.number = true` writes `Config.options.number`; `vim.g.mapleader = " "`
writes `Config.leader`; `vim.keymap.set(...)` pushes a `Keymap`;
`vim.cmd.colorscheme("gruvbox")` writes `Config.theme`. The whole config runs to
produce one merged `Config`, which then builds the `Editor` exactly as the
compiled-in default does. **Alternative considered:** a parallel "Lua editor
state" the UI reads from. Rejected — it would duplicate `Config` and split the
source of truth. The cost of the chosen path: the Lua surface can only express
what `Config` can model today (see the degrade list).

### 2. Native plugins resolve to *marker* functions, not black holes

`require('telescope.builtin').find_files` returns a marker native that
`vim.keymap.set` recognises **by identity** and maps to `Action::FindFiles` —
so a user config that re-declares the telescope/hop/harpoon/`vim.lsp.buf`
keymaps lands on kvim's *native* subsystems rather than on a dead stub. This
mirrors the scale-model test's `Markers` approach. **Alternative:** treat every
`require` result as an opaque black hole. Rejected — it would silently downgrade
every plugin keymap to a no-op Lua callback, losing the native mapping that is
the whole point of kvim compiling its plugins in.

### 3. Autocmd support level: **record + report, do not fire**

kvim has no general event bus yet (no `BufWritePre`/`FileType`/`VimEnter`
hooks). `vim.cmd("autocmd ...")` and `vim.api.nvim_create_autocmd` are parsed
and stored (`LuaRuntime::autocmds`) but never fired. **Alternative:** wire a
minimal event bus now. Rejected as out of scope for a first cut — it is a real
subsystem, filed as a follow-up bead. The risk: a config that relies on an
autocmd for correctness (e.g. forcing a filetype) is *recorded* but inert, so
that behaviour silently does not happen. Called out loudly here because it is
the most likely "it loaded but didn't do what I meant" surprise.

### 4. Degrade-and-warn over hard-fail, everywhere except a broken parse

An unknown `vim.<x>` or an unknown `require` resolves to a **black hole** (a
table that absorbs any field access or call) plus a one-line warning shown once
at startup. Plugin-manager boilerplate (`require("lazy").setup{...}`) therefore
runs to completion and the options/keymaps around it still apply. The one hard
edge is a *malformed* Lua file: a syntax or uncaught runtime error aborts that
file (Lua cannot resume past it), keeps whatever ran before the failing line,
and surfaces the error as a warning — matching the spirit of kvim treating a bad
`config.json` as loud, but preferring one bad line not to kill the whole config.
**Alternative:** hard-error the whole startup on any unsupported call, like a
strict interpreter. Rejected — it would make a single unsupported `vim.*` field
(of which there are hundreds) unusable, defeating the goal of running a *real*
config.

## What of the maintainer's real config now applies

Running their `settings.lua` + `keymaps.lua` + `telescope_harpoon.lua` shape:
all ten `vim.opt` options, `vim.g.mapleader`, the LSP/neo-tree/hop/telescope/
harpoon keymaps (mapped to native actions), `vim.cmd.colorscheme("gruvbox")`,
and `syntax`/`filetype` lines. `lazy.nvim` and the ~20 plugin `require`s degrade
to stubs; `termguicolors` and other unmodelled options warn.

## What would make this wrong

* **If the maintainer wants autocmds to actually fire** (decision 3), this is
  incomplete: the event bus becomes required, not a follow-up.
* **If they want their existing third-party Lua plugins to run** rather than be
  replaced by kvim's native ones (the AID-0003 decision-3 question), then the
  black-hole/marker split is wrong and the shim needs to run real plugin Lua —
  a far larger target.
* **If a buffer-local keymap (`{ buffer = true }`) must stay buffer-local**:
  kvim has no per-buffer keymap scope, so those currently apply globally with a
  warning.

## Follow-ups filed

See the beads referenced from `kopitiam-cj0.11`: an event bus for autocmds;
multi-mode keymaps (`{ "n", "v" }` collapses to the first mode today);
per-buffer keymap scope; and `vim.tbl_deep_extend`/`vim.tbl_extend` (currently
warn-and-stub).

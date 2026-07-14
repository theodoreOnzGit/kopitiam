# kvim maturity reference

A checklist of what a **mature modal editor** provides, mapped against what
`kvim` (`crates/kopitiam-neovim`) already has, is building, or lacks — and,
for each gap, where **Helix** is worth studying when we build it.

## How to read this

* **kvim is a vim clone** (verb→noun: `dw` deletes a word). This is deliberate
  — the maintainer is a Neovim user and wants vim muscle memory. See
  `docs/ai-decisions/AID-0003`.
* **Helix is a clean-room maturity reference only.** It is vendored at
  `crates/kopitiam-ai/vendor/helix` (MPL-2.0, gitignored, never built or
  linked). We read it to learn *what* a mature editor has and *how* it wires
  the infrastructure, then write original Rust. **No Helix code is copied**, and
  Helix's *selection-first* keymap is explicitly **not** a model for how kvim
  binds keys — only for what features exist. See `docs/ACKNOWLEDGEMENTS.md`.
* Status legend: **Have** = implemented; **Building** = has an open/in-flight
  bead; **Gap** = filed here as a new bead; **Deferred** = out of current scope.

---

## 1. Core modal editing — essentially complete

kvim's modal engine (`src/editor/`) is mature. Recorded so nobody re-files
these as "missing":

| Feature | Status | Notes |
|---|---|---|
| Motions `h j k l w b e W B E f F t T 0 $ ^ gg G` | Have | `editor/motion.rs` |
| Operators `d c y > < gu gU g~` + counts + `.` dot-repeat | Have | `editor/operator.rs`, `pending.rs` |
| Text objects `iw aw i( a( i" a" it ip …` | Have | `editor/text_object.rs` |
| Registers: unnamed, named `a-z`, yank `"0` | Have | `editor/register.rs` |
| Macros `q`/`@`, `@@` | Have | `editor/mod.rs` |
| Undo tree (in-memory) | Have | `text/undo.rs` |
| Ex: `:w :q :wq :x :e :s :g :d :set` + ranges | Have | `editor/ex.rs` |
| Search motions `/ ? n N * #` | Have | `editor/search.rs` (UI wiring in flight) |
| Jumplist `<C-o>`/`<C-i>`, marks `` ` ``/`'` | Have | `editor/mod.rs` |
| Increment/decrement `<C-a>`/`<C-x>` | Have | `editor/mod.rs` |
| Scroll `<C-d/u/f/b/e/y>`, view `zz zt zb` | Have | `pending.rs`, `ViewportScroll` |
| Visual / visual-line / visual-block, `gv` | Have | `editor/mod.rs` |
| Replace mode `R`, `J` join | Have | `editor/mod.rs` |

**In flight** (do not duplicate): window splits + `<C-w>` motions
(`kopitiam-cj0.10.2/.3`), plugin/LSP UI wiring for pickers, file tree, hop,
harpoon, align, git statusline, and go-to-definition/references/rename
(`kopitiam-cj0.10`), `:term`, and the LSP request layer
(definition/references/hover/completion/diagnostics) landing in
`kopitiam-semantic` (`kopitiam-yxj`, `-gjg`, `-mfo`).

---

## 2. Gaps filed as beads

All parented under the kvim epic `kopitiam-cj0`. "Helix wiring to study" is the
*infrastructure* to learn from — never the keybinding.

| Feature | Bead | Prio | Helix wiring to study |
|---|---|---|---|
| **LSP document-sync lifecycle** — didOpen/didChange/didClose/didSave, lazy per-language spawn | `cj0.12` | P1 | Registry of running clients; per-document version counter; edits emit versioned didChange (incremental+debounced; full-doc is a valid first cut) |
| **Command-line editing + history + completion** — the `:`/`/` prompt is write-only today | `cj0.13` | P1 | Reusable Prompt line-editor with history registers; per-command completion callback; command palette is the command registry through the picker |
| **System clipboard + numbered/blackhole registers** — only unnamed/named/`"0` exist | `cj0.14` | P2 | Clipboard-provider abstraction (kvim adds an OSC-52 fallback for Android/SSH/tmux) |
| **Search-match highlighting** — hlsearch/incsearch; `:noh` is a no-op | `cj0.15` | P2 | Highlight all matches of the search register in the viewport as a render pass under the selection |
| **Diagnostics rendering + `]d`/`[d` + list** — `DiagnosticsStore` exists, nothing renders | `cj0.16` | P2 | Underline + gutter + end-of-line virtual text; diagnostics picker (document/workspace) |
| **Interactive LSP popups** — completion menu, hover, code actions, signature help | `cj0.17` | P2 | Small popup/menu components fed by the LSP client; completion menu shows item docs |
| **Project search + quickfix/location lists** — `:grep`, `:vimgrep`, `:copen`, `:cnext`, `:cdo` | `cj0.18` | P2 | Global content search (ripgrep-style) into a picker; kvim reuses vendored `ignore`+`nucleo`, no external `rg` |
| **Ex-command completeness** — `:v :sort :m/:t :>/:< :normal :ls/:b{name} :earlier/:later` | `cj0.19` | P2 | N/A (vim-specific; Helix has no ex line) |
| **which-key popup** — `config::Keymap.desc` already exists, no popup renders it | `cj0.20` | P3 | Minor-mode infobox listing available continuations |
| **Shell integration** — `:!cmd`, `:r !cmd`, the `!{motion}` filter operator | `cj0.21` | P3 | Pipe/bang shell commands (kvim binds the `!` *operator* the vim way) |
| **g-prefix commands** — `gf`, `gq`/`gw` reflow, `g;`/`g,` changelist | `cj0.22` | P3 | `gf` (goto file under selection) in goto mode; reflow/changelist are vim-specific |
| **undofile** — persist the undo tree across sessions | `cj0.23` | P3 | N/A (vim feature; store via redb/`kopitiam-index`) |

**Architectural recommendation:** `docs/ai-decisions/AID-0019` proposes two
Helix *infrastructure* patterns as kvim's target — a workspace-root-keyed LSP
session registry (today `lsp/client.rs` keys sessions by filetype, which breaks
with two open projects) and a typed ex-command registry. Review bead
`kopitiam-cj0.24`.

---

## 3. Deliberately excluded — Helix's selection-first model is WRONG for kvim

These are genuine Helix features. They are **not** filed as gaps and must not
be: they are consequences of Helix's noun→verb *selection-first* model, which
is the opposite of kvim's verb→noun *vim* model. Flagging them as "missing"
would push kvim away from the vim muscle memory that is its whole point.

| Helix feature | Why excluded |
|---|---|
| `x` = select line, `X` = extend to line bounds | In vim `x` deletes a char. Selection-first line handling has no place in a vim grammar. |
| `w`/`b`/`e` **extend a selection**; `d` deletes the *current selection* | kvim's `w`/`b`/`e` are motions and `dw` deletes a word. This is the core model difference — reversing it would un-vim the editor. |
| Match mode `mm`/`mi(`/`ma(`/`ms`/`mr`/`md` (select-then-surround) | vim expresses these as operator+text-object (`ci(`, `di(`) and surround plugins (`ys`/`cs`/`ds`), which kvim already does via text objects. |
| Multiple cursors by default (`C`, `select_regex` `s`, `,`/`;` selection mgmt) | vim is single-cursor; multi-cursor, if ever wanted, is a separate opt-in, not the default editing model. |
| `s` select-regex-in-selection, `S` split, `&`/`_` align/trim selections, rotate selections `(`/`)` | All operate on Helix's multi-selection primitive, which kvim does not have. vim's equivalents are `:s`, `:g`, `:sort`, visual-block. |
| Select/extend mode `v` as a persistent extend layer; `;` collapse selection | vim's visual mode is a selection, not an extend-motion layer; the semantics differ fundamentally. |
| Syntax-tree sibling/parent selection (`Alt-o/i/p/n/a`, tree-sitter) | Depends on tree-sitter (rejected — AID-0009) *and* on selection-first navigation. |
| `gw` goto-word labels as a default motion | kvim already has label-motion via hop (`f` → `HopWords`); binding it to `gw` selection-style is not the vim way. |

Rule of thumb for future contributors: if a Helix feature only makes sense
*because you already have a selection*, it does not belong in kvim. If it is
something a Neovim user would also expect (LSP lifecycle, diagnostics,
clipboard, command history, quickfix), it does.

---

## 4. Deferred (known, not yet scoped)

* **Snippet engine** (LuaSnip replacement) — Phase 5 native-plugin work; needs
  the completion popup (`cj0.17`) first.
* **Autocommands / event hooks** — arrive with the Lua `vim.*` surface
  (Phase 4, `kopitiam-cj0.4`/`.11`).
* **Incremental syntax highlighting** — the lexers are tracked by
  `kopitiam-v66`/`kopitiam-2qi` (AID-0009: hand-written pure-Rust lexers, no
  tree-sitter). The *incremental re-lex on edit* is the infra piece to build
  when highlighting is wired; Helix's approach is to re-highlight only the
  changed range.

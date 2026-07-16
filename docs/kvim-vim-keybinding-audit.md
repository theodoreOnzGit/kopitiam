# kvim vs vim/neovim keybinding audit

*Read-only audit, done against the vendored neovim source
(`crates/kopitiam-ai/vendor/neovim`, Apache-2.0/Vim-license, gitignored,
never built) vs what `kvim` (`crates/kopitiam-neovim`) actually implements.*

## What this doc is for, ah

The earlier `docs/kvim-maturity-reference.md` benchmark kvim against **Helix**
— that one is about *what features a mature editor got* (LSP lifecycle,
diagnostics, clipboard, quickfix). This doc is the **complement**: purely a
*vim-key-completeness* check. The question here is narrow — "for every key a
neovim user's fingers already know, does kvim do the same thing?" No feature
philosophy, just key coverage.

Method: the canonical vim key surface come straight from the source, not from
memory —

* **Normal + Visual + Operator-pending:** the `nv_cmds[]` dispatch table in
  `src/nvim/normal.c` (this table *is* the list — every key + its handler).
* **`g`-prefix:** the `nv_g_cmd()` switch in `normal.c`.
* **`z`-prefix:** the `nv_zet()` switch in `normal.c`.
* **Bracket `[` `]`:** `nv_brackets()` in `normal.c`.
* **Window `<C-w>`:** the `do_window()` switch in `src/nvim/window.c`.
* **Insert mode:** the key switch in `src/nvim/insert.c`; completion submodes
  from `src/nvim/insexpand.c`.
* **Ex-commands:** compared against `kvim`'s `editor/ex.rs` parser.

kvim side read from `editor/pending.rs` (the normal/operator/visual grammar),
`editor/ex.rs`, `editor/motion.rs`, `editor/operator.rs`, `editor/mod.rs` (the
Ctrl-key + insert-mode handling) and `ui/app.rs` (window `<C-w>` + completion
popup).

**Status legend:** **Have** = fully works the vim way. **Partial** = some of
the family works, or works with caveats. **Missing** = not implemented.

Precision note: key names below are **exact** on purpose — an audit that fuzzes
the key names is useless. The prose around them is Singlish; the keys are not.

---

## 1. Normal-mode Ctrl-keys

From `nv_cmds[]`. kvim catches these ahead of the vi grammar in
`editor/mod.rs` (`handle_normal_key`, the `if key.mods.ctrl` block).

| Key | What it does (vim) | kvim status |
|---|---|---|
| `<C-a>` | increment number under/after cursor | Have |
| `<C-x>` | decrement number | Have |
| `<C-b>` | scroll one page back | Have (full-page scroll) |
| `<C-f>` | scroll one page forward | Have |
| `<C-d>` | scroll half-page down | Have |
| `<C-u>` | scroll half-page up | Have |
| `<C-e>` | scroll view one line down | Have (`ViewportScroll::LineDown`) |
| `<C-y>` | scroll view one line up | Have |
| `<C-o>` | jumplist: older position | Have |
| `<C-i>` / `<Tab>` | jumplist: newer position | Have |
| `<C-r>` | redo | Have |
| `<C-v>` | enter visual-block | Have |
| `<C-w>` | window command prefix | Partial (see §11) |
| `<C-c>` | interrupt → back to Normal | Partial (Esc-like; no "interrupt" semantics) |
| `<C-g>` | show file info / cursor position | Have (cj0.41) |
| `<C-l>` | redraw screen | **Missing** |
| `<C-]>` | jump to tag / goto-definition under cursor | Have (routed to LSP go-to-definition; cj0.41) |
| `<C-^>` | edit alternate file (`#`) | Have (`<C-6>` too; cj0.41) |
| `<C-t>` | pop tag stack | **Missing** |
| `<C-z>` | suspend to shell | **Missing** (arguably N/A for the TUI) |
| `<C-\>` | (leave to command / null) | **Missing** (niche) |

Biggest real gaps here: `<C-g>` (fileinfo), `<C-^>` (alternate file — muscle
memory for a lot of nvim users), `<C-]>`/`<C-t>` (tag stack).

---

## 2. Normal-mode single-key commands (non-motion)

The action keys from `nv_cmds[]` that are not pure motions.

| Key | What it does (vim) | kvim status |
|---|---|---|
| `i a I A o O` | enter Insert at various positions | Have |
| `x` `X` | delete char forward / backward | Have |
| `s` | substitute char (delete + insert) | Have |
| `S` | substitute whole line | Have (`= cc`; cj0.41) |
| `r{c}` | replace one char | Have |
| `R` | Replace (overtype) mode | Have |
| `~` | toggle case under cursor | Have |
| `J` | join lines | Have |
| `gJ` | join without inserting space | **Missing** (see §8) |
| `p` `P` | put after / before | Have |
| `C` | change to end of line (`c$`) | Have (cj0.41) |
| `D` | delete to end of line (`d$`) | Have (cj0.41) |
| `Y` | yank to end of line (`y$`, neovim default) | Have (cj0.41) |
| `u` | undo | Have |
| `U` | undo all changes on one line | **Missing** (deferred — needs a line-snapshot the undo tree doesn't keep; cj0.42) |
| `<C-r>` | redo | Have |
| `.` | repeat last change | Have |
| `&` | repeat last `:s` | Have (cj0.41) |
| `q{reg}` / `q` | record / stop macro | Have |
| `@{reg}` / `@@` | play macro / replay last | Have |
| `Q` | Ex mode / repeat last recorded register | **Missing** |
| `m{a-z}` | set mark | Have |
| `` `{m} `` / `'{m}` | jump to mark (exact / line) | Have |
| `ZZ` | write + quit | Have (cj0.41) |
| `ZQ` | quit without saving | Have (cj0.41) |
| `:` | command line | Have |
| `/` `?` `n` `N` `*` `#` | search family | Have |
| `K` | keyword lookup (`keywordprg` / LSP hover) | **Missing** as `K` (LSP hover is elsewhere) |
| `gv` | reselect last visual | Have |

Note: `C`/`D`/`Y`/`S` are the classic one-key shortcuts for `c$`/`d$`/`yy`/`cc`.
The long forms all work in kvim, but the single-key vim shortcuts don't exist
yet — that's real muscle-memory friction.

---

## 3. Operators (operator-pending: `[count]op[count]motion`)

`editor/operator.rs` enum + `editor/pending.rs` grammar. The composition engine
itself is mature (`d2w`, `"ay3j`, `ci(` all compose from the same slots).

| Operator | What it does (vim) | kvim status |
|---|---|---|
| `d` | delete | Have |
| `c` | change | Have |
| `y` | yank | Have |
| `>` `<` | indent / dedent | Have |
| `gu` `gU` `g~` | lowercase / uppercase / toggle-case | Have |
| `=` | reindent / format via `equalprg` | **Missing** (no indenter/formatter engine yet; deferred cj0.43) |
| `!` | filter through external command | **Missing** (bead cj0.21) |
| `gq` `gw` | reflow / format text width | **Missing** (bead cj0.22) |
| `zf` | create fold over motion | **Missing** (no fold engine) |
| doubled (`dd cc yy >> guu`) | linewise on current line | Have |

The operator machinery is solid; the gaps are the `=` format operator, the `!`
filter operator, and `gq`/`gw` reflow — all already have or touch existing
beads except `=`.

---

## 4. Motions

`editor/motion.rs` + `simple_motion()` in `pending.rs`.

| Key | What it does (vim) | kvim status |
|---|---|---|
| `h j k l` | left/down/up/right | Have |
| `<Space>` (→ `l`), `<BS>` (→ `h`) | char right/left | Partial (arrows Have; `<Space>`/`<BS>` as motions Missing) |
| `w W b B e E` | word motions | Have |
| `ge gE` | backward word-end | Have |
| `0 ^ $` | line start / first-non-blank / line end | Have |
| `g_` | last non-blank | Have |
| `-` `+` / `<CR>` | first-non-blank of prev / next line | Have (cj0.41) |
| `_` | first-non-blank, `count-1` lines down | Have (cj0.41) |
| `\|` | go to column `count` | Have (cj0.41) |
| `f F t T` | find char on line | Have |
| `; ,` | repeat / reverse last `f/F/t/T` | Have |
| `{ }` | paragraph back / forward | Have |
| `( )` | sentence back / forward | Have (mapped to Sentence motions) |
| `%` | matching pair | Have |
| `H M L` | screen top / mid / bottom | Have |
| `gg G` | file start / end | Have |
| `gj gk` | display-line down/up | Have (== `j`/`k` with `wrap=false`) |
| `gm gM g0 g$ g^` | display-line column motions | **Missing** (see §8) |
| `[[ ]] [] ][` | section / brace motions | Have (brace-in-col-0; cj0.35) |
| `[( ]) [{ ]}` | unmatched-bracket motions | Have (cj0.35) |
| `[m ]m [M ]M` | method start/end | Have (brace-scan approximation; cj0.35) |

Core motions all Have. The stragglers are the line-oriented `+`/`-`/`_`/`|`
motions and the whole bracket-motion family (§10).

---

## 5. Text objects (`i`/`a` + object)

`text_object_for()` in `pending.rs`. Strong coverage.

| Object | What it does (vim) | kvim status |
|---|---|---|
| `iw aw iW aW` | word / WORD | Have |
| `i( a( ib ab` / `i) a)` | parens | Have |
| `i{ a{ iB aB` / `i} a}` | braces | Have |
| `i[ a[` / `i] a]` | brackets | Have |
| `i< a< i> a>` | angle brackets | Have |
| `i" a" i' a' `` i` a` `` | quotes | Have |
| `it at` | tag block | Have |
| `ip ap` | paragraph | Have |
| `is as` | sentence | **Missing** |
| `i_ a_` (some plugins) | n/a in core | N/A |

Only sentence text-objects (`is`/`as`) missing; everything a nvim user reaches
for daily is there.

---

## 6. Visual mode

Driven through the same `pending.rs` grammar + `editor/mod.rs` visual handling.

| Key | What it does (vim) | kvim status |
|---|---|---|
| `v V <C-v>` | charwise / linewise / blockwise | Have |
| `gv` | reselect last | Have |
| `o` | swap selection end | **Missing** |
| `O` | swap corner (blockwise) | **Missing** |
| operators on selection (`d c y > < gu gU g~`) | act on selection | Have |
| `iw i( it ...` (text objects extend selection) | Have |
| `:` (`:'<,'>`) | range from selection | Partial (ex range works; auto `'<,'>` prefill unclear) |
| `r{c}` on selection | replace all | Partial |
| `I` `A` (blockwise insert) | block insert / append | **Missing** |
| `u U ~` (case on selection) | Partial |
| `J` join selection | Partial |
| `p` put over selection | Partial |

Visual mode exists and the common operator-on-selection path works, but the
blockwise-specific editing keys (`I`/`A` block insert, `o`/`O` corner swap) are
Missing — that's the main visual gap.

---

## 7. Insert mode

`handle_insert_key()` in `editor/mod.rs`, plus the completion popup in
`ui/app.rs`.

| Key | What it does (vim) | kvim status |
|---|---|---|
| printable chars | insert | Have |
| `<Esc>` | leave insert | Have |
| `<CR>` | newline | Have |
| `<BS>` / `<C-h>` | delete char back | Have (`<BS>`; `<C-h>` key not wired explicitly) |
| `<Del>` | delete char forward | Have |
| `<Tab>` | insert tab / expandtab spaces | Have |
| arrows / Home / End | move insertion point | Have |
| `<C-w>` | delete word before cursor | Have |
| `<C-u>` | delete to line start | Have |
| `<C-r>{reg}` | insert register contents | Have |
| `<C-o>` | one Normal-mode command then back | Have |
| `<C-a>` | insert previously inserted text | **Missing** |
| `<C-t>` `<C-d>` | indent / dedent current line | **Missing** |
| `<C-k>{c}{c}` | digraph entry | **Missing** |
| `<C-v>{code}` / `<C-q>` | insert literal / by code | **Missing** |
| `<C-e>` `<C-y>` | copy char from line below / above | **Missing** |
| `<C-g>j` `<C-g>k` `<C-g>u` | insert-mode `<C-g>` subcommands | **Missing** |
| `<C-^>` | toggle langmap | **Missing** (niche) |
| `<C-n>` `<C-p>` | keyword completion (native) | Partial (only when the LSP completion popup already open — no native buffer-keyword completion) |
| `<C-x>` completion submodes | see §7b | **Missing** |

### 7b. `<C-x>` completion submodes (`insexpand.c`)

vim's `<C-x>` opens a completion sub-mode; each has its own follow key. kvim
has an LSP-driven completion popup (`ui/app.rs`, the `blink.cmp` replacement),
navigable with `<C-n>`/`<C-p>`/`<C-e>`, but **none of vim's native `<C-x>`
submodes** exist.

| Key | What it completes (vim) | kvim status |
|---|---|---|
| `<C-x><C-n>` / `<C-x><C-p>` | keywords in current file | **Missing** |
| `<C-x><C-l>` | whole lines | **Missing** |
| `<C-x><C-f>` | file names | **Missing** |
| `<C-x><C-k>` | dictionary words | **Missing** |
| `<C-x><C-t>` | thesaurus | **Missing** |
| `<C-x><C-i>` | keywords from included files | **Missing** |
| `<C-x><C-]>` | tags | **Missing** |
| `<C-x><C-d>` | definitions from includes | **Missing** |
| `<C-x><C-v>` | vim command line | **Missing** |
| `<C-x><C-o>` | omni completion | Partial (LSP popup is the moral equivalent, but not bound to `<C-x><C-o>`) |
| `<C-x><C-u>` | user `completefunc` | **Missing** |
| `<C-x><C-s>` | spelling suggestions | **Missing** |
| `<C-x><C-e>` / `<C-x><C-y>` | scroll while in insert | **Missing** |

Insert mode is the second-largest gap area after brackets/folds: the editing
shortcuts `<C-a>`/`<C-t>`/`<C-d>`/`<C-k>`/`<C-v>`/`<C-e>`/`<C-y>` and the entire
`<C-x>` completion family are all absent.

---

## 8. `g`-prefix commands (`nv_g_cmd()`)

| Key | What it does (vim) | kvim status |
|---|---|---|
| `gg` | goto first line | Have |
| `gj gk` | display-line down / up | Have |
| `ge gE` | backward word end | Have |
| `g_` | last non-blank | Have |
| `gv` | reselect visual | Have |
| `gu gU g~` | case operators | Have |
| `gI` | insert at column 1 | **Missing** |
| `gi` | insert at last insert position | **Missing** |
| `ga` | show char code under cursor | **Missing** |
| `g8` | show UTF-8 bytes | **Missing** |
| `gd gD` | goto local / global declaration | **Missing** (LSP `gd` is a separate path) |
| `gf gF` | goto file under cursor | **Missing** (bead cj0.22) |
| `gq gw` | reflow text | **Missing** (bead cj0.22) |
| `g; g,` | changelist back / forward | **Missing** (bead cj0.22) |
| `gJ` | join without space | **Missing** |
| `g0 g^ g$ gm gM` | display-line column motions | **Missing** |
| `g*` `g#` | search word (not whole-word) | **Missing** (plain `*`/`#` Have) |
| `gn gN` | select next/prev search match | **Missing** |
| `g&` | repeat `:s` on all lines | **Missing** |
| `gp gP` | put + leave cursor after | **Missing** |
| `go` | goto byte in buffer | **Missing** |
| `gt gT` | next / prev tab page | **Missing** (bead cj0.10.6, no tabs) |
| `gr gR` | virtual replace mode | **Missing** |
| `g+ g-` | undo-tree older / newer text state | **Missing** |
| `g<` | redisplay last `:` output | **Missing** |
| `gs` | sleep | **Missing** (niche) |

cj0.22 covers `gf`/`gq`/`gw`/`g;`/`g,` only. The rest of the `g`-prefix surface
(`gI gi ga g8 gd gJ g* g# gn gN g& gp gP go gr gR g+ g-`) has no bead yet.

---

## 9. `z`-prefix commands (`nv_zet()`)

| Key | What it does (vim) | kvim status |
|---|---|---|
| `zz` | cursor line to centre | Have |
| `zt` | cursor line to top | Have |
| `zb` | cursor line to bottom | Have |
| `z<CR>` `z.` `z-` | scroll + move to first non-blank | **Missing** |
| `z^` `z+` | screen up / down page | **Missing** |
| `zh zl zH zL` | horizontal scroll | **Missing** |
| `zs ze` | horizontal scroll cursor to start / end | **Missing** |
| **Folds:** `zf zF zd zD zE zo zO zc zC za zA zv zx zX` | create/delete/open/close folds | **Missing** (no fold engine at all) |
| `zr zR zm zM zn zN zi` | foldlevel / foldenable controls | **Missing** |
| `zj zk` | move to next / prev fold | **Missing** |
| **Spell:** `zg zw zG zW zug zuw z=` | good/wrong word, suggestions | **Missing** (no spell engine) |

Only the three view-repositioning commands (`zz`/`zt`/`zb`) exist. Everything
else under `z` — folds, horizontal scroll, spell — is Missing. Folding
especially is a whole missing subsystem, not just a keybind.

---

## 10. Bracket `[` `]` commands (`nv_brackets()`)

**Core motions now Have (cj0.35).** `pending.rs` gained an `AwaitingBracket`
state, so `[`/`]` dispatch as motion prefixes and compose with operators/counts
(`d]}`, `y[[`, `2]m`). All are charwise-exclusive, matching neovim's
`nv_brackets`. One UI wrinkle fixed alongside: the app-level `]`/`[`
interception for `]d`/`[d` diagnostics used to *drop* the bracket for any
non-`d` second key; it now replays it into the editor grammar (see
`ui/app.rs`). The mark jumps land through a dedicated `JumpBracketMark`
command (marks live in the buffer, which the buffer-free `Pending` cannot
read).

| Key | What it does (vim) | kvim status |
|---|---|---|
| `[[ ]]` | section backward / forward (to `{` in col 1) | Have (brace-in-col-0) |
| `[] ][` | section end backward / forward (to `}` in col 1) | Have |
| `[( ])` | unmatched `(` back / `)` forward | Have |
| `[{ ]}` | unmatched `{` back / `}` forward | Have |
| `[m ]m [M ]M` | method start / end (Java-ish) | Have (prev/next brace approximation) |
| `[p ]p [P ]P` | put with indent adjust | **Missing** (deferred — a put variant, not a motion; cj0.44) |
| `['` `` [` `` / `]'` `` ]` `` | prev / next lowercase mark | Have (cj0.35) |
| `[z ]z` | move to start / end of open fold | **Missing** (no folds) |
| `[c ]c` | prev / next diff change | **Missing** (no diff mode) |
| `[d ]d [D ]D` | show / jump to macro define | **Missing** |
| `[i ]i [I ]I` | show / jump to identifier under cursor | **Missing** |
| `[s ]s` | prev / next misspelled word | **Missing** (no spell) |
| `[f ]f` | (old) goto file — deprecated in nvim | N/A |

Note: plugin authors also expect `[q ]q` (quickfix), `[b ]b` (buffer),
`[d ]d` (diagnostics) — the last is tracked in the maturity ref as cj0.16.
The *core* bracket motions above have no bead.

---

## 11. Window commands `<C-w>{key}` (`do_window()`)

`ui/app.rs` (`handle_window_command` + the `<C-w>`-pending dispatch). About a
third of vim's window commands exist.

| Key | What it does (vim) | kvim status |
|---|---|---|
| `<C-w>s` / `<C-w>S` | split horizontal | Have |
| `<C-w>v` | split vertical | Have |
| `<C-w>n` | new split | Have |
| `<C-w>c` | close window | Have |
| `<C-w>q` | quit window | Have |
| `<C-w>o` | only (close others) | Have |
| `<C-w>h/j/k/l` | focus left/down/up/right | Have |
| `<C-w>w` | cycle to next window | Have |
| `<C-w>=` | equalize sizes | Have |
| `<C-w>W` | cycle to previous window | **Missing** |
| `<C-w>p` | goto previous (last-accessed) window | **Missing** |
| `<C-w>t` / `<C-w>b` | goto top-left / bottom-right window | **Missing** |
| `<C-w>x` `<C-w><C-x>` | exchange with next window | **Missing** |
| `<C-w>r` `<C-w>R` | rotate windows down / up | **Missing** |
| `<C-w>H/J/K/L` | move window to far edge | **Missing** (bead cj0.10.5) |
| `<C-w>+` `<C-w>-` | grow / shrink height | **Missing** |
| `<C-w>>` `<C-w><` | grow / shrink width | **Missing** |
| `<C-w>_` | max height | **Missing** |
| `<C-w>\|` | max width | **Missing** |
| `<C-w>T` | move window to new tab | **Missing** (bead cj0.10.6) |
| `<C-w>f` `<C-w>F` `<C-w>gf` | open file under cursor in split / tab | **Missing** |
| `<C-w>]` `<C-w>}` | tag / preview-tag in split | **Missing** |
| `<C-w>i` `<C-w>d` | goto identifier / define in split | **Missing** |
| `<C-w>P` `<C-w>z` | preview window | **Missing** |
| `<C-w>^` | split + edit alternate file | **Missing** |

The **resize** family (`+ - < > _ |`) is the most-missed everyday gap; then
rotate/exchange/move (`r R x H J K L`) and the goto-in-split family.

---

## 12. Command-line editing (`:` / `/` prompt)

The `:`/`/` prompt is essentially **write-only** today — you can type and
Enter, but the line-editing keys aren't there. Tracked as bead **cj0.13**.

| Key | What it does (vim) | kvim status |
|---|---|---|
| `<C-w>` | delete word back | **Missing** |
| `<C-u>` | delete to start | **Missing** |
| `<C-r>{reg}` | insert register | **Missing** |
| `<C-b>` `<C-e>` | line start / end | **Missing** |
| `<Left>` `<Right>` | move cursor in cmdline | **Missing** |
| `<Up>` `<Down>` / `<C-p>` `<C-n>` | history | **Missing** |
| `<Tab>` | command / path completion | **Missing** |
| `<C-f>` | open cmdline-window | **Missing** |
| `q:` `q/` | cmdline window | **Missing** |

---

## 13. Ex-commands

kvim's `editor/ex.rs`. Solid core; completeness tracked in bead **cj0.19**.

| Command | kvim status |
|---|---|
| `:w :wq :x :q :q! :qa :wa :wqa :xa` | Have |
| `:e {file}` | Have |
| `:bn :bp :b{n} :bd :bw :ls` | Have |
| `:s/// :%s/// :{range}s///` | Have (no `\/` escaping; flags: `g` only) |
| `:g/pat/{d\|s}` | Have (only `d` and `s` sub-commands) |
| `:{range}d` | Have |
| `:noh` | Partial (parses; no hlsearch yet — cj0.15) |
| `:set {opt}` | Have (subset of options) |
| `:{n}` (goto line), `:%`, ranges `.`,`$` | Have |
| `:sp :vs :new :vnew :only :close` | Have |
| `:term` | Partial (placeholder buffer — cj0.10.4) |
| `:help [topic]` | Have (Singlish manual) |
| `:v/pat/` (inverse global) | **Missing** (cj0.19) |
| `:sort` | **Missing** (cj0.19) |
| `:m{addr}` `:t{addr}` (move/copy) | **Missing** (cj0.19) |
| `:>` `:<` (shift range) | **Missing** (cj0.19) |
| `:normal {cmds}` | **Missing** (cj0.19) |
| `:earlier :later` (undo time-travel) | **Missing** (cj0.19) |
| `:!{cmd}` `:r !{cmd}` | **Missing** (cj0.21) |
| `:grep :vimgrep :copen :cnext` | **Missing** (cj0.18) |
| `:reg :marks :jumps` | **Missing** |
| `:tabnew :tabclose :tabnext` | **Missing** (cj0.10.6) |
| `:map :nnoremap ...` (mappings) | **Missing** (needs Lua/config — cj0.4/.11) |

---

## Biggest gaps summary (prioritized)

Ranked by how often a working nvim user's fingers would hit a wall:

1. **Bracket `[` `]` motion family — entire group Missing (§10).** No `[[`,
   `]]`, `[(`, `])`, `[{`, `]}`, `[m`, `]m`, `[p`, `]p`, `['`, `` [` ``. This
   is a whole prefix that just doesn't dispatch. **Bead cj0.35.** *P2.*

2. **`z`-prefix beyond `zz/zt/zb` — folds, horizontal scroll, spell all Missing
   (§9).** Folding is a missing *subsystem*, not one keybind. **Bead cj0.36.**
   *P3 (folds are a big lift).*

3. **Insert-mode `<C-x>` completion submodes + native keyword completion
   (§7b).** The whole `<C-x><C-*>` family absent; native `<C-n>`/`<C-p>` only
   work when the LSP popup is already up. **Bead cj0.37.** *P2.*

4. **Window `<C-w>` resize / rotate / exchange / goto-in-split (§11).** Have
   ~10 of ~30. The resize family (`+ - < > _ |`) is daily-driver stuff and has
   no bead (cj0.10.5/.6 only cover move-to-edge + tabs). **Bead cj0.38.** *P3.*

5. **Insert-mode editing keys `<C-a> <C-t> <C-d> <C-k> <C-v> <C-e> <C-y>`
   (§7).** Common insert-mode reflexes (indent/dedent, literal, digraph).
   **Bead cj0.39.** *P3.*

6. **`g`-prefix completeness beyond cj0.22 (§8).** `gI gi ga g8 gJ g* g# gn gN
   g& gp gP go gr gR g+ g-` unbound. **Bead cj0.40.** *P3.*

7. **Normal-mode one-key shortcuts + misc Ctrl-keys (§1, §2).** `C`/`D`/`Y`/`S`
   one-key forms, `ZZ`/`ZQ`, `<C-g>`, `<C-^>`, `<C-]>`, `U`, `&`, `=` format
   operator, `|`/`+`/`-`/`_` line motions. **Bead cj0.41** (grouped). *P2 for
   the `C`/`D`/`Y`/`ZZ`/`ZQ` reflexes; the rest P3.*

Already-tracked (not re-filed, cross-referenced above): command-line editing
(cj0.13), ex-command completeness (cj0.19), hlsearch (cj0.15), clipboard/
registers (cj0.14), `!`/shell (cj0.21), `gf`/`gq`/`gw`/`g;`/`g,` (cj0.22),
`<C-w>H/J/K/L` (cj0.10.5), tabs (cj0.10.6), quickfix (cj0.18), diagnostics
`]d`/`[d` (cj0.16 in the maturity ref).

**Not gaps — deliberately vim-correct:** kvim's verb→noun grammar, single
cursor, and text-object-based surround are the intended model (see AID-0003 and
the maturity ref §3). Helix-style selection-first keys are *not* filed here.

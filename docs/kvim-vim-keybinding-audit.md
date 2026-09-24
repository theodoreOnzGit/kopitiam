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

> ### Re-audit, 2026-09-24 — read this before trusting a row
>
> This document was written on **31 August** and was substantially out of date
> by late September: cj0.35, cj0.37, cj0.39, cj0.41, cj0.42, cj0.43 and cj0.44
> all landed in between, as did the fold engine, tab pages, the shell/filter
> family, the command-line editor and most of the `g`-prefix. A partial status
> refresh went in with `ae550bd`, but it touched only §1, §2 and §4 — every
> other section still described the 31 August code.
>
> Every row below has now been re-checked against the source. Corrections are
> struck through and dated (`~~old~~ **CORRECTED 2026-09-24** — …`) rather than
> silently rewritten, so a reader can see which way the claim moved. The
> **Biggest gaps summary** at the end was rewritten wholesale; the original
> ranking is preserved above the new one.
>
> Depth of re-check, stated honestly: §2, §3, §4, §6, §7, §7b and §10 were
> re-audited **row by row** against `editor/pending.rs`, `editor/motion.rs`,
> `editor/operator.rs`, `editor/mod.rs` and `ui/app.rs`. §8, §9, §11, §12 and
> §13 were re-checked against their *dispatch tables* (`feed_g`, `feed_z` +
> `fold_command_for`, `handle_window_command`, `editor/cmdline.rs`,
> `editor/command.rs` + `editor/ex.rs`) — so a row marked Have there is backed
> by a dispatch arm that exists, but the *semantics* of those arms were not
> re-derived from neovim the way the in-scope sections' were.

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

~~Biggest real gaps here: `<C-g>` (fileinfo), `<C-^>` (alternate file — muscle
memory for a lot of nvim users), `<C-]>`/`<C-t>` (tag stack).~~
**CORRECTED 2026-09-24** — `<C-g>`, `<C-^>`/`<C-6>` and `<C-]>` all landed in
cj0.41 (`5330fa6`); the rows above already say so. **The remaining gaps here
are `<C-t>` (pop tag stack — there is no tag stack at all, which also blocks
`<C-x><C-]>` in §7b) and `<C-l>` (redraw).** Re-checked against the
`if key.mods.ctrl` dispatch in `handle_normal_key`: neither has an arm.

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
| `gJ` | join without inserting space | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (`feed_g`, `JoinLines { space: false }`) |
| `p` `P` | put after / before | Have |
| `C` | change to end of line (`c$`) | Have (cj0.41) |
| `D` | delete to end of line (`d$`) | Have (cj0.41) |
| `Y` | yank to end of line (`y$`, neovim default) | Have (cj0.41) |
| `u` | undo | Have |
| `U` | undo all changes on one line | ~~Missing (deferred — needs a line-snapshot the undo tree doesn't keep)~~ **CORRECTED 2026-09-24** — **Have** (cj0.42, `e6a90ef`): line-undo, self-toggling, and `u` can undo the `U` |
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

~~Note: `C`/`D`/`Y`/`S` are the classic one-key shortcuts for `c$`/`d$`/`yy`/`cc`.
The long forms all work in kvim, but the single-key vim shortcuts don't exist
yet — that's real muscle-memory friction.~~ **CORRECTED 2026-09-24** — all four landed in cj0.41
(`5330fa6`) and dispatch from `pending.rs`'s `feed_fresh`. Note `Y` is `y$`,
**neovim's** default (nvim ships `Y` → `y$` as a default mapping), not classic
vim's `yy`; `big_y_yanks_to_end_of_line_neovim_default_not_the_whole_line` pins
that choice so it cannot drift back.

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
| `=` | reindent / format via `equalprg` | ~~Missing (deferred)~~ **CORRECTED 2026-09-24** — **Have** (cj0.43, `942ae19`): `Operator::Format`, `==` / `={motion}` / visual `=`, LSP `rangeFormatting` when a server is up, else a brace-depth C-style reindent |
| `!` | filter through external command | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (cj0.21, `c919e66`): `!{motion}`, `!!`, and `:{range}!` |
| `gq` `gw` | reflow / format text width | **Missing** (bead cj0.22) |
| `zf` | create fold over motion | ~~Missing (no fold engine)~~ **CORRECTED 2026-09-24** — **Have** (`ae91904`): `Operator::Fold`, so `zf3j` / `zfip` / `zfG` compose through the ordinary operator machinery |
| doubled (`dd cc yy >> guu`) | linewise on current line | Have |

~~The operator machinery is solid; the gaps are the `=` format operator, the `!`
filter operator, and `gq`/`gw` reflow.~~ **CORRECTED 2026-09-24** — `=`, `!` and `zf` have all
landed. **The only operator still missing is `gq`/`gw` reflow** (cj0.22).

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

~~Core motions all Have. The stragglers are the line-oriented `+`/`-`/`_`/`|`
motions and the whole bracket-motion family (§10).~~ **CORRECTED 2026-09-24** — `+`/`-`/`_`/`|`
landed in cj0.41 and the bracket family in cj0.35; both rows above already say
Have. **The only motions still missing are the display-line column family
`gm gM g0 g$ g^`**, which need `wrap` to mean something (kvim runs
`wrap=false`, where a display line *is* a buffer line).

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
| `o` | swap selection end | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (`o_swaps_the_visual_ends`) |
| `O` | swap corner (blockwise) | ~~Missing~~ **CORRECTED 2026-09-24** — **Partial**: `O` is dispatched, but aliased to `o` (same end-swap). The blockwise *horizontal* corner swap is not distinct |
| operators on selection (`d c y > < gu gU g~`) | act on selection | Have |
| `iw i( it ...` (text objects extend selection) | Have |
| `:` (`:'<,'>`) | range from selection | Partial (ex range works; auto `'<,'>` prefill unclear) |
| `r{c}` on selection | replace all | Partial |
| `I` `A` (blockwise insert) | block insert / append | **Missing** |
| `u U ~` (case on selection) | Partial |
| `J` join selection | Partial |
| `p` put over selection | Partial |

~~Visual mode exists and the common operator-on-selection path works, but the
blockwise-specific editing keys (`I`/`A` block insert, `o`/`O` corner swap) are
Missing — that's the main visual gap.~~ **CORRECTED 2026-09-24** — `o` (end swap) is Have and `O` is
dispatched but aliased to it. **The remaining visual gap is blockwise-specific
editing: `I`/`A` block insert, and `O`'s distinct horizontal corner swap.**

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
| `<C-a>` | insert previously inserted text | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (cj0.39, `80a0ea3`): `insert_previous_inserted`, reading the `".` register |
| `<C-t>` `<C-d>` | indent / dedent current line | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (cj0.39): `insert_shift_line`, cursor held on the same text |
| `<C-k>{c}{c}` | digraph entry | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (cj0.39): `handle_insert_digraph` + `editor/digraph.rs`, either character order |
| `<C-v>{code}` / `<C-q>` | insert literal / by code | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** for `<C-v>` (cj0.39): literal-next-key plus the `{ddd}` / `o{ooo}` / `x{hh}` / `u{hhhh}` / `U{hhhhhhhh}` numeric forms, terminator fed back rather than swallowed. `<C-q>` (the alias) is **not** wired |
| `<C-e>` `<C-y>` | copy char from line below / above | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (cj0.39): `insert_char_from_adjacent_line`; a no-op when that line is too short, as vim is. The completion popup claims these two only while it is open |
| `<C-g>j` `<C-g>k` `<C-g>u` | insert-mode `<C-g>` subcommands | **Missing** |
| `<C-^>` | toggle langmap | **Missing** (niche) |
| `<C-n>` `<C-p>` | keyword completion (native) | ~~Partial (only when the LSP popup is already open)~~ **CORRECTED 2026-09-24** — **Have** (cj0.37, `9a639d1`): with no menu up they open real vim keyword completion over the current + other window buffers |
| `<C-x>` completion submodes | see §7b | ~~Missing~~ **CORRECTED 2026-09-24** — **Partial**: six of the thirteen are wired, see §7b |

### 7b. `<C-x>` completion submodes (`insexpand.c`)

~~vim's `<C-x>` opens a completion sub-mode; each has its own follow key. kvim
has an LSP-driven completion popup, navigable with `<C-n>`/`<C-p>`/`<C-e>`, but
**none of vim's native `<C-x>` submodes** exist.~~

**CORRECTED 2026-09-24** — CTRL-X mode exists. `App::completion_intercept` (`ui/app.rs`) holds a
`ctrl_x_pending` flag and routes the sub-key to a native source; the **one**
popup is reused for every source, with `CompletionKind` recording which one
seeded it so the as-you-type refresh re-gathers from the same place instead of
reverting to the default identifier menu on the next keystroke. An unrecognised
sub-key cancels CTRL-X mode and then takes its own ordinary path.

The submodes that remain Missing are not missing *keybinds* — each needs a
subsystem kvim does not have (a `dictionary`/`thesaurus` option, include
scanning, a tag stack, a spell engine, a `completefunc` hook).

| Key | What it completes (vim) | kvim status |
|---|---|---|
| `<C-x><C-n>` / `<C-x><C-p>` | keywords in current file | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (cj0.37); `this_buffer_only`, unlike plain `<C-n>` |
| `<C-x><C-l>` | whole lines | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (cj0.37); distinct buffer lines, leading whitespace ignored when matching |
| `<C-x><C-f>` | file names | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (cj0.37); relative to the buffer's own directory, tree root for an unnamed/bare-relative buffer |
| `<C-x><C-k>` | dictionary words | **Missing** — no `dictionary` option |
| `<C-x><C-t>` | thesaurus | **Missing** — no `thesaurus` option |
| `<C-x><C-i>` | keywords from included files | **Missing** — no include scanning |
| `<C-x><C-]>` | tags | **Missing** — no tag stack (see §1's `<C-t>`) |
| `<C-x><C-d>` | definitions from includes | **Missing** — no include scanning |
| `<C-x><C-v>` | vim command line | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (`df2a56d`): kvim's `:` vocabulary via `editor::command::complete_names`. Command *names* only; arguments stay the `:` prompt's own `<Tab>` completion |
| `<C-x><C-o>` | omni completion | ~~Partial (not bound to `<C-x><C-o>`)~~ **CORRECTED 2026-09-24** — **Have** (cj0.37): bound, and LSP is the *only* source in this submode |
| `<C-x><C-u>` | user `completefunc` | **Missing** — needs a Lua hook |
| `<C-x><C-s>` | spelling suggestions | **Missing** — no spell engine |
| `<C-x><C-e>` / `<C-x><C-y>` | scroll while in insert | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (`df2a56d`). Note this was worse than missing: the unrecognised sub-key used to fall out of CTRL-X mode, so `<C-x><C-e>` reached the editor as a plain insert-mode `<C-e>` and **copied a character out of the line below into the buffer** |

~~Insert mode is the second-largest gap area after brackets/folds: the editing
shortcuts `<C-a>`/`<C-t>`/`<C-d>`/`<C-k>`/`<C-v>`/`<C-e>`/`<C-y>` and the entire
`<C-x>` completion family are all absent.~~ **CORRECTED 2026-09-24** — **both groups have landed**
(cj0.37, cj0.39, and `df2a56d`). Insert mode's remaining gaps are the `<C-g>`
subcommands (`<C-g>j`/`<C-g>k`/`<C-g>u`), `<C-h>` as an explicit alias for
`<BS>`, `<C-q>` as an alias for `<C-v>`, `<C-^>` (langmap), and the five
`<C-x>` submodes that each want a subsystem kvim has not got.

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
| `gI` | insert at column 1 | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (`InsertPos::FirstColumn`) |
| `gi` | insert at last insert position | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (`InsertPos::LastInsert`) |
| `ga` | show char code under cursor | **Missing** |
| `g8` | show UTF-8 bytes | **Missing** |
| `gd gD` | goto local / global declaration | **Missing** (LSP `gd` is a separate path) |
| `gf gF` | goto file under cursor | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** for `gf` (`GotoFile`); `gF` (with a line number) not wired |
| `gq gw` | reflow text | **Missing** (bead cj0.22) |
| `g; g,` | changelist back / forward | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (`ChangelistJump`) |
| `gJ` | join without space | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (`JoinLines { space: false }`) |
| `g0 g^ g$ gm gM` | display-line column motions | **Missing** |
| `g*` `g#` | search word (not whole-word) | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (`SearchWordLoose`) |
| `gn gN` | select next/prev search match | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (`SelectMatch`; the one `g` command that also composes with an operator (`cgn`)) |
| `g&` | repeat `:s` on all lines | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (`RepeatSubstituteGlobal`) |
| `gp gP` | put + leave cursor after | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (`Put { cursor_after: true }`) |
| `go` | goto byte in buffer | **Missing** |
| `gt gT` | next / prev tab page | ~~Missing (no tabs)~~ **CORRECTED 2026-09-24** — **Have** (`4eda3c1`, real tab pages): `GotoTab`, with `{count}gt` for an absolute jump |
| `gr gR` | virtual replace mode | **Missing** |
| `g+ g-` | undo-tree older / newer text state | **Missing** |
| `g<` | redisplay last `:` output | **Missing** |
| `gs` | sleep | **Missing** (niche) |

~~cj0.22 covers `gf`/`gq`/`gw`/`g;`/`g,` only. The rest of the `g`-prefix surface
(`gI gi ga g8 gd gJ g* g# gn gN g& gp gP go gr gR g+ g-`) has no bead yet.~~
**CORRECTED 2026-09-24** — `c29df65` ("fill in the daily-driver g-prefix commands") landed
`gI gi gf gJ gp gP g; g, g* g# g& gn gN`, and `4eda3c1` added `gt`/`gT`.
**Still missing: `ga g8 gd gD gq gw go gr gR g+ g- g< gs` and the display-line
column motions `g0 g^ g$ gm gM`.**

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
| **Folds:** `zf zF zd zD zE zo zO zc zC za zA zv zx zX` | create/delete/open/close folds | ~~Missing (no fold engine at all)~~ **CORRECTED 2026-09-24** — **Partial**: `ae91904` added a manual-fold engine (`editor/fold.rs`) and `zf zd zE zo zO zc zC za zA zv` all dispatch. `zF` (fold N lines), `zD`, `zx`, `zX` are not wired |
| `zr zR zm zM zn zN zi` | foldlevel / foldenable controls | ~~Missing~~ **CORRECTED 2026-09-24** — **Partial**: `zR zM zn zN zi` dispatch (`FoldOp::OpenAll`/`CloseAll`/`Disable`/`Enable`/`ToggleEnable`). `zr`/`zm` (step `foldlevel`) are not wired — kvim's manual folds have no level to step |
| `zj zk` | move to next / prev fold | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (`FoldOp::MoveNext`/`MovePrev`) |
| **Spell:** `zg zw zG zW zug zuw z=` | good/wrong word, suggestions | **Missing** (no spell engine) |

~~Only the three view-repositioning commands (`zz`/`zt`/`zb`) exist. Everything
else under `z` — folds, horizontal scroll, spell — is Missing. Folding
especially is a whole missing subsystem, not just a keybind.~~ **CORRECTED 2026-09-24** — **the fold
subsystem landed** (`ae91904`, `foldmethod=manual`), with the `zf`/`zo`/`zc`/
`za`/`zR`/`zM`/`zd`/`zE`/`zj`/`zk` family and `[z`/`]z`. **Still missing under
`z`: the scroll-and-move forms (`z<CR>` `z.` `z-` `z^` `z+`), all horizontal
scrolling (`zh zl zH zL zs ze`), and the whole spell family — there is still no
spell engine.**

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
| `[p ]p [P ]P` | put with indent adjust | ~~Missing (deferred)~~ **CORRECTED 2026-09-24** — **Have** (cj0.44, `ca0d25b`): `PUT_FIXINDENT`; only `]p` puts *after*, the other three put before, and a charwise register ignores the reindent |
| `['` `` [` `` / `]'` `` ]` `` | prev / next lowercase mark | Have (cj0.35) |
| `[z ]z` | move to start / end of open fold | ~~Missing (no folds)~~ **CORRECTED 2026-09-24** — **Have** (`FoldOp::MoveStart`/`MoveEnd`) |
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
| `<C-w>W` | cycle to previous window | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (`053b4aa`; `cycle_window(false)`) |
| `<C-w>p` | goto previous (last-accessed) window | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (`053b4aa`; `focus_prev_window`) |
| `<C-w>t` / `<C-w>b` | goto top-left / bottom-right window | **Missing** |
| `<C-w>x` `<C-w><C-x>` | exchange with next window | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (`053b4aa`; `exchange_window`, with a `[count]` to name the target) |
| `<C-w>r` `<C-w>R` | rotate windows down / up | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (`053b4aa`; `rotate_windows`) |
| `<C-w>H/J/K/L` | move window to far edge | **Missing** (cj0.10.5) — dispatched, but answers with a "not implemented yet" note rather than doing the wrong thing |
| `<C-w>+` `<C-w>-` | grow / shrink height | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (`053b4aa`; `resize_window`, `[count]` honoured) |
| `<C-w>>` `<C-w><` | grow / shrink width | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (`053b4aa`; `resize_window`, `[count]` honoured) |
| `<C-w>_` | max height | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (`053b4aa`; `maximize_window(false)`) |
| `<C-w>\|` | max width | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (`053b4aa`; `maximize_window(true)`) |
| `<C-w>T` | move window to new tab | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (`4eda3c1`, `window_to_new_tab`) |
| `<C-w>f` `<C-w>F` `<C-w>gf` | open file under cursor in split / tab | **Missing** |
| `<C-w>]` `<C-w>}` | tag / preview-tag in split | **Missing** |
| `<C-w>i` `<C-w>d` | goto identifier / define in split | **Missing** |
| `<C-w>P` `<C-w>z` | preview window | **Missing** |
| `<C-w>^` | split + edit alternate file | **Missing** |

~~The **resize** family (`+ - < > _ |`) is the most-missed everyday gap; then
rotate/exchange/move (`r R x H J K L`) and the goto-in-split family.~~ **CORRECTED 2026-09-24** —
resize, rotate, exchange, `W`, `p` and `T` all landed (`053b4aa`, `4eda3c1`).
**Still missing: move-to-edge `H/J/K/L` (deliberately deferred, cj0.10.5),
`t`/`b` (top-left / bottom-right), the goto-in-split family (`f F gf ] } i d`)
and the preview window (`P z ^`).**

---

## 12. Command-line editing (`:` / `/` prompt)

~~The `:`/`/` prompt is essentially **write-only** today — you can type and
Enter, but the line-editing keys aren't there.~~ **CORRECTED 2026-09-24** — **cj0.13 has landed**:
`editor/cmdline.rs` is a full command-line editor (cursor, history walk,
completion cycle), and all but the cmdline *window* now work.

| Key | What it does (vim) | kvim status |
|---|---|---|
| `<C-w>` | delete word back | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** |
| `<C-u>` | delete to start | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** |
| `<C-r>{reg}` | insert register | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** |
| `<C-b>` `<C-e>` | line start / end | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (`<Home>`/`<End>` too) |
| `<Left>` `<Right>` | move cursor in cmdline | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** |
| `<Up>` `<Down>` / `<C-p>` `<C-n>` | history | ~~Missing~~ **CORRECTED 2026-09-24** — **Have**; `:` and `/` keep separate histories |
| `<Tab>` | command / path completion | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** for `:` (names, buffer names, paths); still inert on the `/` prompt |
| `<C-f>` | open cmdline-window | **Missing** |
| `q:` `q/` | cmdline window | **Missing** |

---

## 13. Ex-commands

kvim's `editor/ex.rs`. Solid core; completeness tracked in bead **cj0.19**.

**CORRECTED 2026-09-24** — this section was the most stale in the document: cj0.18, cj0.19, cj0.21,
cj0.10.4 and cj0.10.6 have all landed since it was written. **Of the fourteen
rows that read Missing on 31 August, eleven are now Have.**

| Command | kvim status |
|---|---|
| `:w :wq :x :q :q! :qa :wa :wqa :xa` | Have |
| `:e {file}` | Have |
| `:bn :bp :b{n} :bd :bw :ls` | Have |
| `:s/// :%s/// :{range}s///` | Have (no `\/` escaping; flags: `g` only) |
| `:g/pat/{d\|s}` | Have (only `d` and `s` sub-commands) |
| `:{range}d` | Have |
| `:noh` | ~~Partial (no hlsearch yet)~~ **CORRECTED 2026-09-24** — **Have**: hlsearch exists and `:noh` clears the highlight while keeping the pattern |
| `:set {opt}` | Have (subset of options) |
| `:{n}` (goto line), `:%`, ranges `.`,`$` | Have |
| `:sp :vs :new :vnew :only :close` | Have |
| `:term` | ~~Partial (placeholder buffer)~~ **CORRECTED 2026-09-24** — **Have** (`5ee3153`): a real pty-backed terminal emulator |
| `:help [topic]` | Have (Singlish manual) |
| `:v/pat/` (inverse global) | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** |
| `:sort` | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (`!`/`u`/`n` flags) |
| `:m{addr}` `:t{addr}` (move/copy) | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** |
| `:>` `:<` (shift range) | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (`parse_shift`; `>>` shifts twice) |
| `:normal {cmds}` | ~~Missing~~ **CORRECTED 2026-09-24** — **Have**, including composed with `:g` |
| `:earlier :later` (undo time-travel) | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** for the count form; the *time* form (`:earlier 10m`) reports itself unsupported |
| `:!{cmd}` `:r !{cmd}` | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (`c919e66`): `:!` to a scratch buffer, `:{range}!` as a filter, `:r !` reading output in |
| `:grep :vimgrep :copen :cnext` | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (`18d9d66`), plus the location-list twins (`:lgrep :lopen :lnext` …) |
| `:reg :marks :jumps` | **Missing** — re-checked 2026-09-24, still absent from `editor/command.rs` |
| `:tabnew :tabclose :tabnext` | ~~Missing~~ **CORRECTED 2026-09-24** — **Have** (`4eda3c1`), plus `:tabonly :tabprevious :tabfirst :tablast :tabs` |
| `:map :nnoremap ...` (mappings) | **Missing** (needs Lua/config — cj0.4/.11) — re-checked 2026-09-24, still absent |

---

## Biggest gaps summary (prioritized)

> **CORRECTED 2026-09-24 — the original ranking below is obsolete.** Six of its
> seven entries have since been built. It is kept, struck through, because it
> is *why* the work was scheduled in the order it was, and because a summary
> that was quietly swapped out would give no reader a reason to trust the new
> one. The current ranking follows it.

### The 31 August ranking (superseded)

1. ~~**Bracket `[` `]` motion family — entire group Missing (§10).** No `[[`,
   `]]`, `[(`, `])`, `[{`, `]}`, `[m`, `]m`, `[p`, `]p`, `['`, `` [` ``. This
   is a whole prefix that just doesn't dispatch. **Bead cj0.35.** *P2.*~~
   **Done** — motions in cj0.35 (`5330fa6`), `[p`/`]p`/`[P`/`]P` in cj0.44
   (`ca0d25b`), `[z`/`]z` with the fold engine.

2. ~~**`z`-prefix beyond `zz/zt/zb` — folds, horizontal scroll, spell all
   Missing (§9).** Folding is a missing *subsystem*, not one keybind.
   **Bead cj0.36.** *P3.*~~ **Partly done** — the manual-fold engine landed
   (`ae91904`). Horizontal scroll and spell are still absent; see the new
   ranking.

3. ~~**Insert-mode `<C-x>` completion submodes + native keyword completion
   (§7b).** The whole `<C-x><C-*>` family absent. **Bead cj0.37.** *P2.*~~
   **Done** — cj0.37 (`9a639d1`) for keyword/line/file/omni and native
   `<C-n>`/`<C-p>`; `<C-x><C-v>` and `<C-x><C-e>`/`<C-x><C-y>` in `df2a56d`.

4. ~~**Window `<C-w>` resize / rotate / exchange / goto-in-split (§11).**~~
   **Mostly done** — `053b4aa` (resize, rotate, exchange, maximise, counts) and
   `4eda3c1` (`<C-w>T`). Goto-in-split and move-to-edge remain.

5. ~~**Insert-mode editing keys `<C-a> <C-t> <C-d> <C-k> <C-v> <C-e> <C-y>`
   (§7). Bead cj0.39.** *P3.*~~ **Done** — `80a0ea3`.

6. ~~**`g`-prefix completeness beyond cj0.22 (§8).** `gI gi ga g8 gJ g* g# gn
   gN g& gp gP go gr gR g+ g-` unbound. **Bead cj0.40.** *P3.*~~ **Mostly
   done** — `c29df65`; `ga g8 go gr gR g+ g-` remain.

7. ~~**Normal-mode one-key shortcuts + misc Ctrl-keys (§1, §2).** `C`/`D`/`Y`/
   `S`, `ZZ`/`ZQ`, `<C-g>`, `<C-^>`, `<C-]>`, `U`, `&`, `=`, `|`/`+`/`-`/`_`.
   **Bead cj0.41.** *P2.*~~ **Done** — cj0.41 (`5330fa6`) for the one-key and
   Ctrl reflexes, cj0.42 (`e6a90ef`) for `U`, cj0.43 (`942ae19`) for `=`.

### The ranking as of 2026-09-24

Ranked, as before, by how often a working nvim user's fingers would hit a wall.

1. **Blockwise visual editing — `I` / `A` block insert (§6).** *P2.* The one
   remaining everyday reflex with no substitute: column editing is the reason
   to enter `<C-v>` at all, and kvim can select a block but not type into one.
   `O`'s distinct horizontal corner swap belongs with it (today `O` is aliased
   to `o`).

2. **`gq` / `gw` reflow, and `=`'s missing companions (§3, §8).** *P2.* `=`
   landed in cj0.43, so this is the last operator-shaped gap. Bead cj0.22.

3. **Spell — the whole `z` spell family and `[s` / `]s` (§9, §10).** *P3.* A
   missing subsystem, not a keybind: no spell engine exists. It also blocks
   `<C-x><C-s>`.

4. **Horizontal scrolling and the scroll-and-move `z` forms (§9).** *P3.*
   `zh zl zH zL zs ze`, `z<CR>` `z.` `z-` `z^` `z+`. Cheap individually; they
   need the textarea to carry a horizontal offset.

5. **Window goto-in-split and move-to-edge (§11).** *P3.* `<C-w>H/J/K/L`
   (cj0.10.5, dispatched but deliberately answering "not implemented"),
   `<C-w>t`/`b`, and the `f F gf ] }` + `i`/`d` family.

6. **Display-line motions `g0 g^ g$ gm gM` (§4, §8).** *P3.* Meaningful only
   once `wrap` does something; kvim runs `wrap=false`, where a display line is
   a buffer line.

7. **The long tail of `g` and the tag stack (§1, §8).** *P3.* `ga g8 go gr gR
   g+ g-`; `<C-t>` (pop tag stack) and with it `<C-x><C-]>`; `<C-l>` (redraw),
   `Q`, `K`, `is`/`as` sentence text objects, the cmdline window (`q:`,
   `<C-f>`), `:reg`/`:marks`/`:jumps`, and `:map` (which waits on Lua config,
   cj0.4/.11).

Still-open beads referenced above: cj0.4/.11 (Lua config + mappings), cj0.22
(`gq`/`gw`), cj0.10.5 (`<C-w>H/J/K/L`). The beads closed by the work this
re-audit recorded: cj0.10.4, cj0.10.6, cj0.13, cj0.15, cj0.18, cj0.19, cj0.21,
cj0.35, cj0.36 (partly), cj0.37, cj0.38 (mostly), cj0.39, cj0.40 (mostly),
cj0.41, cj0.42, cj0.43, cj0.44 — **propose these for closure; do not close
them from this document alone.**

**Not gaps — deliberately vim-correct:** kvim's verb→noun grammar, single
cursor, and text-object-based surround are the intended model (see AID-0003 and
the maturity ref §3). Helix-style selection-first keys are *not* filed here.

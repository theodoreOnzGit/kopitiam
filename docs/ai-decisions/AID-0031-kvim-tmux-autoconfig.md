# AID-0031: kvim's tmux `is_vim` auto-fix — the fix shape, the regex, and how a decline is remembered

* **Status:** Pending review
* **Bead:** `kopitiam-cj0.31`
* **Date:** 2026-07-16
* **Decided by:** AI (Claude), maintainer absent

## The brief

Bead `kopitiam-cj0.31` asks kvim to detect the vim-tmux-navigator `is_vim`
problem at startup (tmux's process-name regex dunno `kvim`, so it eats
`<C-h/j/k/l>` before kvim sees them) and *offer*, with consent, to patch the
user's `tmux.conf`. The brief left three things to judgment: the exact regex
kvim writes, how to remember a decline so kvim dun nag, and the detection
heuristics. This records those.

## Decision 1 — the block kvim writes uses the christoomey-canonical regex, not the README's abbreviated one

The crate README previously told users to add, by hand, a regex of the form
`g?(k?vim?x?|fzf)`. That form matches `vim` and `kvim` but **not `nvim`** (no
`n`), and drops `view`. The maintainer runs a heavy Neovim setup, so a block
that silently stops recognising `nvim` would be a regression for their other
panes.

kvim now appends (and the README now shows) the christoomey/vim-tmux-navigator
**canonical** regex with `kvim` slotted in:

```
^[^TXZ ]+ +(\\S+\\/)?g?(view|kvim|n?vim?x?|fzf)(diff)?$
```

This is the form the whole vim-tmux-navigator ecosystem uses, so it matches
`vim`, `nvim`, `view`, `fzf` **and** `kvim`. I also rewrote the README's
"by hand" block to be byte-identical to what kvim writes, so "what we tell you
to add" and "what we add for you" can never drift.

The double backslashes (`\\S`, `\\/`) are deliberate and load-bearing: inside
the double-quoted `is_vim="..."` string tmux collapses `\\` → `\`, so grep
receives `\S`/`\/`. A single backslash would reach grep as a bare `S`/`/` and
break the match. There is a unit test pinning this exact substring survives.

## Decision 2 — a decline is remembered with a marker file in kvim's own directory

The brief said "don't nag every startup … a marker file, or a config flag, or
respect-for-this-run — your call." I chose a **marker file** at
`~/.kopitiam/kopitiam-neovim/.tmux-autoconfig-declined`.

* **Persistent** across restarts (respect-for-this-run alone re-asks every
  launch, which is the nagging the brief wants gone).
* **Touches nothing the user owns** — it lives in kvim's *own* directory, never
  the user's dotfile, so declining is genuinely side-effect-free from the
  user's point of view.
* **Reversible and discoverable** — the README says deleting the file brings
  the offer back.

Rejected: editing `config.json` to set `tmux_autoconfig: false` (programmatic
JSON rewriting of a human-owned file is more invasive than dropping a marker,
and the config surface is the user's, not kvim's scratch space).

## Decision 3 — detection heuristics are deliberately generous toward "already fixed"

* `recognises_kvim(conf)` = the literal token `kvim` appears anywhere in the
  conf. A false positive (e.g. `kvim` only in a comment) just means kvim stays
  quiet — the *safe* direction, since kvim never edits a conf it is unsure
  about. `kvim` is distinctive enough not to occur by accident.
* `has_is_vim_check(conf)` = both `is_vim` and `grep` present, so a stray
  `is_vim` variable used for something else isn't mistaken for the navigator.
* The surgical edit (case a) anchors on the `vim` token, walks back to the
  nearest `(` and inserts `kvim|` at the group start → `(kvim|view|…)`. If
  there is no group (a bare `grep -iqE 'n?vim|fzf'`) it backs up over the
  token's own regex chars and inserts at the boundary → `kvim|n?vim`, never
  splitting `n?vim`. If it cannot confidently locate the alternation, it falls
  back to appending a fresh block whose `is_vim` shadows the earlier one — the
  safe outcome, kvim's known-good regex wins.

## Safety property

`compute_fix` is pure — it only returns a new *string*, it cannot touch a file.
`TmuxOffer::apply` is the one function that writes, it always backs up first
(`tmux.conf.kvim-bak`), and the UI only calls it after the user presses `y`.
kvim never runs `tmux source-file` for the user — it tells them to, exactly as
`--install-font` leaves the font-cache reload to the user. PTY-verified: the
popup paints, `n` leaves the conf byte-identical with no backup, `y` adds
`kvim` and writes the backup.

## What would make this wrong

* If the maintainer prefers the abbreviated `k?vim?x?` regex on purpose (e.g.
  they no longer run `nvim` and want the shorter form), Decision 1 should be
  reversed and the README's block restored — but then the README and the
  written block must still be kept identical.
* If the marker file location proves surprising (a user who declines, later
  fixes their conf by hand, then wonders why kvim never offered again — though
  in that case `recognises_kvim` returns true and the marker is moot). A
  `config.json` flag would be more discoverable if that happens.
* If a real-world conf uses an `is_vim` regex shape the group-walk mis-locates,
  producing a bad surgical edit. Mitigated by the append-block fallback and the
  mandatory backup, but a surprising conf is the most likely source of a bug
  here.

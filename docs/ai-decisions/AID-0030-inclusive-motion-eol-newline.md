# AID-0030: Inclusive motions stop at end-of-line instead of swallowing the newline

* **Status:** Pending review
* **Bead:** `kopitiam-cj0.41`
* **Date:** 2026-07-16
* **Decided by:** AI (Claude), maintainer absent

## The brief

Bead `kopitiam-cj0.41` asks for the one-key reflexes `C`/`D`/`Y` (= `c$`/`d$`/`y$`)
and friends. It says nothing about touching `operator::charwise_range`. But
building `D`/`C`/`Y` surfaced a pre-existing bug in that shared function, and
shipping the new keys on top of it would have made them visibly wrong.

## What was wrong

`charwise_range`, for an **inclusive** motion, extended the range one grapheme
past the landing char `b` with `step_right(buf, b)`. When `b` is the *last*
grapheme of its line, `step_right` crosses onto the next line's column 0 — so
the resulting range swallowed the trailing **newline**, merging two lines.

Concretely, on `"foo\nbar"` with the cursor on line 0:

* `y$` yanked `"foo\n"` (with the newline) instead of `"foo"`.
* `D` / `d$` deleted `"foo\n"`, pulling `bar` up onto line 0, instead of
  leaving an empty line 0.

Real vim/neovim never let an inclusive motion consume the trailing newline
(that is what *linewise* is for). Since `C`/`D`/`Y` are `c$`/`d$`/`y$`, they
inherited the bug — and these are the headline "daily reflex" keys the bead is
about, so a merge-the-next-line `D` would have been an immediately noticeable
regression in feel.

## The decision

Fix it at the source. In `charwise_range`'s `Inclusive` arm, cap the range end
at end-of-line when `b` is the last grapheme of its line:

```rust
let line_len = buf.line_len(b.line);
let after = if b.col + 1 < line_len { Position::new(b.line, b.col + 1) }
            else { Position::new(b.line, line_len) };
```

This is the one place the rule can live: every inclusive motion (`$`, `e`,
`f`, `t`, `%`) funnels through `charwise_range`, so `d$`/`de`/`dfx`/`dtx` at a
line's end are all corrected in one edit, not just the new keys.

## Why this is safe for cross-line inclusive motions

The only inclusive motion that legitimately lands on a *different* line is `%`
(matching bracket). Its landing `b` is the bracket itself, which is essentially
never the last grapheme of its line (there is usually a `;`, `)`, or more code
after it) — and even when it is, stopping at end-of-line still *includes* the
bracket, which is what `d%` wants; it only declines to also eat the newline.
The regression test `inclusive_to_eol_motions_do_not_swallow_the_newline`
pins both halves: `D`/`d$` leave the line break, and `d%` across lines still
reaches its bracket.

## Alternatives considered

1. **Leave `charwise_range` alone; special-case `C`/`D`/`Y` to compute a
   newline-free range themselves.** Rejected: it would fix three keys and leave
   `d$`/`de`/`dfx` still merging lines, i.e. paper over the symptom while the
   bug lived on for every other inclusive-at-EOL operator. Worse, it would
   duplicate range logic the crate deliberately centralises (see
   `operator.rs`'s module docs).
2. **Make `$` linewise.** Wrong: `d$` is charwise in vim (`dd` is the linewise
   one); `y$` must not yank a whole line.
3. **Do nothing, document the quirk.** Rejected: it makes the bead's marquee
   feature feel broken on any non-final line.

## What would make this wrong

* If some inclusive motion is later added whose landing `b` is *meant* to be
  the last char of a line **and** whose operator range should include the
  following newline. No such motion exists in vim's model (that need is served
  by linewise motions), but if one is invented, it would need `Linewise`
  granularity, not a re-widened `Inclusive` range.
* If a downstream consumer was silently relying on the old
  newline-swallowing behaviour of `d$`/`y$`. The full suite (481 tests) passed
  unchanged after the fix, and no test encoded the old behaviour, so nothing in
  the tree did.

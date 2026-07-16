# AID-0033: kvim system clipboard — OSC-52 as the primary copy path, platform tools as fallback, no C clipboard crate

* **Status:** Pending review
* **Bead:** `kopitiam-cj0.14`
* **Date:** 2026-07-16
* **Decided by:** AI (Claude), maintainer absent

## The brief

> Add the system clipboard registers `"+`/`"*` to kvim, alongside the numbered
> delete-ring, the blackhole register, and the read-only registers.

The register families are all uncontroversial vim mechanics with one correct
answer each. The clipboard is the one place a real design choice had to be
made without the maintainer, because "talk to the OS clipboard" has several
mutually-incompatible implementations and the choice has long-term
consequences for the Pure Rust Core promise and for Android.

## The decision

**Copy (yank/delete into `"+`/`"*`):** emit an **OSC-52 terminal escape** as
the primary path, and *additionally* run a desktop tool (`wl-copy`, `xclip`,
`pbcopy`, `termux-clipboard-set`) when one is detected. OSC-52 goes out over
`/dev/tty`, not stdout, so it reaches the terminal emulator regardless of the
TUI owning stdout.

**Paste (put from `"+`/`"*`):** platform tool only (`wl-paste`, `xclip -o`,
`pbpaste`, `termux-clipboard-get`). OSC-52 *read* is deliberately not
attempted. When no reader is available, paste degrades to a no-op with a note
rather than crashing.

**No C-linked clipboard crate.** The two obvious crates (`arboard`,
`copypasta`) link X11/Wayland/Cocoa C libraries and would break the Pure Rust
Core promise and the Android build. Tools are spawned as separate processes
via `std::process`, exactly as kvim's git and tmux integrations already shell
out rather than link.

**`clipboard = "unnamedplus"` mirror.** When the maintainer's config sets it
(a very common neovim setting), plain `y`/`d`/`p` route through `"+`
automatically; `"unnamed"` routes through `"*`. Empty (vim's default, and the
maintainer's actual setting) keeps plain ops internal and `"+y`/`"+p` explicit.

## Why OSC-52 is the *primary* copy path, not the fallback

This is the load-bearing judgment. The intuitive design is "use the native
desktop tool, fall back to OSC-52". I inverted it because OSC-52 is the path
that keeps working where the desktop tool cannot, and those are exactly
KOPITIAM's stated environments:

* **SSH** — a yank on a remote host with no X-forwarding lands in the *local*
  clipboard, because the escape travels down the same pty as everything else.
* **tmux** — `set-clipboard on` forwards OSC-52 through transparently.
* **Android / Termux** — there is no X11 or Wayland at all, so a desktop
  clipboard binding is a non-starter; the terminal app understands OSC-52.

It also needs no external binary and no C dependency. Running the desktop tool
*as well* is belt-and-braces: it covers terminals that do not honour OSC-52.

## Why paste cannot symmetrically use OSC-52

An OSC-52 *query* asks the terminal to send the clipboard back as terminal
*input*. Virtually every terminal disables that by default, because any process
that can write to your pty could otherwise silently read your clipboard. So a
paste has to use a real reader tool, and there genuinely is no reader on a bare
SSH session — hence the honest graceful-degradation path rather than a
pretend-it-worked.

## Alternatives considered

* **`arboard` / `copypasta` (C-linked clipboard crates).** Rejected: breaks
  Pure Rust Core and the Android build. This is the same reasoning as AID-0004
  (font shipping) and AID-0009 (no tree-sitter) — a C dependency for
  convenience is exactly what the project forbids.
* **Desktop tool primary, OSC-52 fallback.** Rejected for the SSH/tmux/Android
  reasons above — it would make the *headline* environments the degraded ones.
* **OSC-52 for paste too.** Rejected: blocked by terminals by default, and
  enabling it is a clipboard-exfiltration risk the user did not opt into.
* **tmux DCS passthrough wrapping of the escape.** Not done. Modern tmux with
  `set-clipboard on` handles bare OSC-52; passthrough wrapping needs
  `allow-passthrough on` and complicates the common case. Documented as a
  known limitation instead.

## What would make this wrong

* **If the maintainer primarily works in a local desktop terminal that does
  *not* implement OSC-52.** Some older/minimal terminals ignore it. The desktop
  tool covers that when present, but a user on such a terminal with no
  `wl-copy`/`xclip` installed would find copy silently ineffective. If this is
  the common case, the priority should flip.
* **If a base64 or clipboard concern later justifies a dependency after all.**
  The base64 encoder is hand-rolled (~15 lines, RFC 4648 test-vector-checked)
  to avoid a dependency for a handful of bytes; if base64 is needed elsewhere,
  a shared pure-Rust crate would be reasonable.
* **If the maintainer wants `"*` to mean the X11 primary selection with
  middle-click semantics distinct from `"+`.** It does map to primary on
  X11/Wayland, but on macOS both collapse to the one pasteboard, and the
  OSC-52 `p` selection is not universally supported. If primary-selection
  fidelity matters, this needs revisiting per-platform.

## Note for the reviewer

Per the working-practices constraint on this task, this AID doc is left
**uncommitted** in the working tree — the implementing agent staged only
`crates/kopitiam-neovim/`. Please commit it (and add the index row in this
directory's `README.md`) as part of review, or hand it back.

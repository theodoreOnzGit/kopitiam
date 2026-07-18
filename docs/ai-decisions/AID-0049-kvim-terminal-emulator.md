# AID-0049: kvim `:term` — a real pty-backed terminal emulator (crates, terminal-mode, lifecycle)

Status: Pending review
Date: 2026-07-18
Crate: `kopitiam-neovim`
Related: AID-0028 (async LSP session is a background-thread **actor**, not a
lock or a typed enum — the reader-thread discipline reused here), AID-0005
(how kvim runs on Android/Termux — the Pure-Rust-Core-on-`aarch64-linux-android`
constraint), AID-0003 (what `kvim` is), AID-0020 (a window is a viewport; the
editor stays headless, the UI owns geometry + OS resources), bead
`kopitiam-cj0.10.4`

## Context

`:term` was an honest placeholder: it opened a scratch buffer that said "terminal
emulation is not yet implemented". This AID records building the real thing — a
pty-backed terminal emulator: spawn a shell in a pseudo-terminal, parse its
ANSI/VT output into a screen grid, forward keystrokes back, render the grid in a
kvim window, and keep it sized to that window. It **must** work in Android Termux
(the maintainer runs kvim there; Termux has ptys).

That is a real architectural addition — a new buffer kind, a new editor **mode**,
background process management, and two new crate dependencies — so it forces
several calls the maintainer would normally make. Hence this AID.

## Decision 1 — the crates: `portable-pty` + `vt100`, and why they are Pure-Rust-Core-safe

**`portable-pty`** (wezterm's) opens the pty and spawns the child. It calls
libc's `openpty`/`ioctl` through FFI. That is the OS's own pty syscall interface,
**not** a bundled C library Cargo has to compile — the same standard the tree
already accepts for `sysinfo` (reads `/proc`) and `memmap2` (wraps `mmap`): a
thin safe wrapper over a syscall is Pure Rust; a `cc`-built C dependency is not.
No `cc`, no CMake, no vendored C enters the build.

**`vt100`** (built on alacritty's `vte`) is the ANSI/VT parser. A terminal
program does not print plain text — it prints text interleaved with escape
sequences (move cursor, set colour, clear line). `vt100` eats that byte stream
and maintains a `Screen` grid of cells (glyph + fg/bg + attrs + cursor) that kvim
paints straight into a ratatui window. Pure Rust, maintained, no C.

Both build for `aarch64-linux-android` — **verified**, not assumed: `cargo check
--target aarch64-linux-android -p kopitiam-neovim` passes clean with them in.
That is the load-bearing fact for the whole feature (see "what would make this
wrong").

**Version pin — `vt100 = "0.15"`, not `0.16`.** `vt100` 0.16 bumped to
`unicode-width ^0.2.1`, but `ratatui` 0.29.0 (kvim's renderer, and `kopitiam-mux`
pins `ratatui = "=0.29.0"`) **hard-pins `unicode-width =0.2.0`** — an exact `=`,
because unicode-width 0.2.1 changed some East-Asian widths and ratatui froze it.
Cargo allows one version per semver-compatible range, so 0.16 simply cannot
resolve against our pinned ratatui. `vt100` 0.15.2 depends on the `unicode-width
0.1` line (a different major, unifies fine), so it slots in with no churn. This
is written into the workspace `Cargo.toml` next to the dep so the next person
does not "upgrade" 0.15→0.16 and hit an unexplained resolve failure. When ratatui
unpins unicode-width (or we bump ratatui), revisit 0.16.

Alternatives weighed: (a) `vt100-ctt` (a maintained 0.17 fork) — same
unicode-width conflict, no win; (b) hand-roll a `vte` + grid — the task
explicitly permitted it, but `vt100` already *is* `vte` + a maintained grid, so
rolling our own would be re-writing a solved, tested problem for nothing; (c)
embed `kopitiam-mux` as the terminal — far heavier (a whole multiplexer) than a
single-terminal buffer needs, and `kopitiam-mux` is itself still stabilising.

## Decision 2 — the model/view/wiring split, and the reader thread is an AID-0028 actor

The pty, its child, and the parser are **OS resources + long-lived state**, so
they live in the UI (`App`), never in the headless editor — same line AID-0020
draws for window geometry. Concretely:

* **`crate::termemu::TermSession`** (model) owns the pty, the child, and an
  `Arc<Mutex<vt100::Parser>>`. `spawn`, `write_input`, `resize`, `with_screen`,
  `take_dirty`, `is_finished`, `reap_if_done`, plus a pure `encode_key`. Knows
  nothing about ratatui or windows.
* **`crate::ui::termgrid::paint_terminal`** (view) paints a `&vt100::Screen` into
  a ratatui cell buffer and returns the cursor position. A free function, not a
  `Widget`, because the screen only exists *borrowed* inside the mutex-guard
  closure — a `Widget` would have to own it.
* **`App`** (wiring) holds `terminals: HashMap<BufferId, TermSession>`, routes
  terminal-mode keystrokes, drives the render, and owns the lifecycle. The editor
  only *recognises* `:term` and hands back `EditorResponse::Terminal { buffer,
  command }` — exactly the "editor recognises, UI performs" seam `:sp`
  (`Window`), `:tabnew` (`Tab`) and `:grep` (`Quickfix`) already use.

**The reader thread is an actor, reusing AID-0028's discipline.** A pty read
blocks until the child writes — could be minutes for an idle shell — and kvim's
UI loop must never block on that. So, exactly like the async LSP session, a
single background `std::thread` owns the read end end-to-end: it loops on
`read()`, and on each chunk locks the shared parser and feeds it. No async
runtime, no tokio — a plain OS thread plus a `Mutex` + two atomics.

Where this *differs* from AID-0028 (and why): AID-0028 uses a **channel** because
its consumer wants a stream of discrete replies. The terminal consumer instead
wants the **whole current screen every frame**, so a byte-chunk channel would
just be reassembled into exactly one parser anyway. So the parser *is* the shared
state behind the mutex, and a `dirty` atomic tells the UI "new output landed,
worth repainting" — the same actor spirit, the shape the consumer dictates.

## Decision 3 — terminal-mode is a new `core::Mode`, and `<C-\><C-n>` must survive crossterm's legacy control-byte decode

Terminal-mode is a genuine new modal state (statusline label `TERMINAL`, distinct
cursor shape), so it is a new `core::Mode::Terminal` variant every subsystem
agrees on — not a boolean bolted onto Normal. In terminal-mode the **UI**
intercepts every keystroke and forwards it raw to the pty (`encode_key` →
`write_input`); the editor is not handed the key at all. `<C-\><C-n>` leaves back
to Normal (so the user can scroll/copy); Normal-mode `i`/`a`/`I`/`A` on a terminal
buffer re-enters terminal-mode — both matching neovim.

**The load-bearing correctness fact, caught by PTY-verifying the real binary:**
a real terminal sends the single byte `0x1C` for `<C-\>`, and **crossterm 0.29
decodes `0x1C..0x1F` to the DIGITS `'4'..'7'` + CONTROL** (see
`crossterm/src/event/sys/unix/parse.rs`) — it genuinely cannot tell `<C-\>` from
`<C-4>`, they are the same byte on a legacy tty. A first cut that only recognised
`Char('\\')+ctrl` passed its unit test (which fed a *synthetic* `Char('\\')`) but
was **un-typable on a real terminal** — the escape never fired, so `:term` was a
one-way trap. The fix: `is_ctrl_backslash` accepts both `Char('\\')` (kitty /
modifyOtherKeys protocols) **and** `Char('4')` (legacy `0x1C`), and `encode_key`
mirrors the mapping so forwarding is faithful to the byte the tty would emit. The
knowledge is written into the rustdoc at both points of use. This is the single
biggest reason the task mandated PTY-verifying the *real* kvim binary rather than
trusting unit tests — the bug lived exactly in the gap between them.

## Decision 4 — lifecycle: reap on the idle tick, kill-then-reap on drop; job-kill on window close deferred

A pty child must be reaped or it lingers as a `<defunct>` zombie. `TermSession`
reaps two ways: `reap_if_done` harvests the child on the idle tick once the reader
signals EOF (a shell that exits on its own is reaped promptly, no zombie while
running), and `Drop` kills-then-waits as the backstop (closing kvim, or dropping
a session, never leaks a process). A shell that exits leaves its final screen
frozen in the buffer (neovim behaviour) rather than panicking or auto-closing.

**Deferred:** killing a terminal's job when its *window* closes (neovim's `:bd!`
semantics) and scrollback *viewing*. Today a session lives until the App drops
(guaranteed clean via `Drop`) or is explicitly replaced; scrollback is *captured*
(1000 rows) but not yet scrollable in Normal mode. Both are filed follow-ups, not
blockers — the "don't leave a zombie" contract is already met by `Drop` + the
idle-tick reap.

## Redraw cadence

Pty output arrives asynchronously, and crossterm's `event::poll` only wakes on
*terminal input*, not pty output. So while any terminal is live the event loop
polls on a short 16ms tick (vs the default 250ms) and repaints when a session
reports fresh output; the slow tick resumes once every terminal closes. This is
the terminal equivalent of the existing diagnostics/git idle-refresh, not a fixed
redraw clock.

## What would make this wrong

* **`portable-pty` or `vt100` fail to build (or run) under Termux's toolchain.**
  This is the risk the runtime cannot paper over — if the pty crate does not
  compile for `aarch64-linux-android`, or the `openpty` FFI is not available on a
  given Termux/Android build, `:term` is dead on the maintainer's actual device.
  Mitigated by the passing `cargo check --target aarch64-linux-android` and by
  both crates being widely used on Android, but the *true* proof is running it in
  Termux on the tablet — which has not been done here. If that fails, the crate
  choice (Decision 1) is what to revisit.
* **crossterm delivers `<C-\>` as something other than `Char('\\')` or
  `Char('4')`** on some terminal/protocol combination the maintainer uses (e.g. a
  future kitty-protocol quirk). Then the escape is un-typable there and
  `is_ctrl_backslash` needs widening. The unit test + PTY test cover the two
  known encodings; a third would surface as "I can't get out of `:term`".
* **The mutex-per-frame render becomes a bottleneck** for a very chatty program
  (a fast `cat` of a huge file). The reader holds the lock only to `process`, the
  renderer only to snapshot, so contention is brief — but if it bites, the fix is
  a double-buffered screen snapshot, not a redesign.
* **A real terminal program needs behaviour `vt100` 0.15 lacks** (e.g. certain
  mouse-reporting or sixel). Then the 0.15 pin (Decision 1) has to be revisited
  ahead of the ratatui/unicode-width unpin — a real constraint to weigh, not a
  free upgrade.

## Outcome

Implemented and green: `cargo test --release -p kopitiam-neovim` (916 pass, +6
over baseline) and `cargo clippy --release -p kopitiam-neovim --all-targets`
clean. Real-binary PTY verification (a forked pty with a real `TIOCSWINSZ`
winsize, child reaped with `waitpid(WNOHANG)`) confirms `:term printf …` scripted
output paints and interactive `:term cat` echoes typed keystrokes — including the
`<C-\><C-n>` escape via the legacy `0x1C` path. Android build verified via
`cargo check --target aarch64-linux-android`.

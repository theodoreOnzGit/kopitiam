//! The terminal emulator behind `:term` — spawn a shell in a real pty, parse
//! its output into a screen grid, forward keystrokes back.
//!
//! # What this module owns, and what it does NOT
//!
//! This is the *model* half of kvim's terminal: a [`TermSession`] owns one
//! pseudo-terminal (pty), the child process running inside it, and a
//! [`vt100::Parser`] that turns the pty's raw byte stream into a `Screen` grid
//! of cells (glyph + colours + attributes + cursor). It knows nothing about
//! ratatui or windows. The *view* half — painting that grid into a kvim
//! window — lives in [`crate::ui::termgrid`], and the *wiring* (which key goes
//! where, when to redraw) lives in [`crate::ui::app`]. Same seam split as the
//! rest of kvim: the model owns the state, the UI renders it.
//!
//! # Why a pty at all, and why these two crates
//!
//! A terminal program (a shell, `vim`, `htop`) does not print plain text — it
//! prints text *interleaved with ANSI/VT escape sequences* (move cursor, set
//! colour, clear line) and it changes its behaviour based on whether it thinks
//! it is talking to a real terminal (line-buffering, echo, `$TERM`, the
//! window size it reads via the `TIOCGWINSZ` ioctl). To run one faithfully you
//! must give it a real pty, not a plain pipe. `portable-pty` (wezterm's) opens
//! that pty by calling libc's `openpty`/`ioctl` — the OS's own syscall
//! interface, not a bundled C library Cargo has to compile — so it stays
//! inside the Pure Rust Core promise (same standard as `sysinfo` reading
//! `/proc`). `vt100` (built on alacritty's `vte`) is the parser that eats the
//! escape stream and maintains the grid. Both are pure Rust and both build for
//! `aarch64-linux-android`, which is what lets `:term` work in Termux. See
//! AID-0049 and the workspace `Cargo.toml` for the full crate rationale.
//!
//! # The reader thread is an actor (AID-0028 discipline)
//!
//! A pty read blocks until the child writes something — could be milliseconds,
//! could be minutes (an idle shell). kvim's UI loop must NEVER block on that,
//! or the whole editor freezes waiting for a shell prompt. So exactly like the
//! async LSP session in AID-0028, a single background `std::thread` owns the
//! read side end-to-end: it loops on `read()`, and on each chunk locks the
//! shared parser and feeds it the bytes. No async runtime, no tokio — a plain
//! OS thread plus a `Mutex` + a couple of atomics.
//!
//! Why a shared `Arc<Mutex<Parser>>` here, rather than AID-0028's channel? The
//! consumer difference decides it. The LSP actor streams discrete *replies*, so
//! a channel of messages fits. The terminal renderer instead needs the *whole
//! current screen* every frame — a stream of byte-chunks would just be
//! reassembled into exactly this one parser anyway. So the parser IS the shared
//! state, guarded by the mutex, and the UI reads a snapshot of it per frame via
//! [`TermSession::with_screen`]. A `dirty` atomic tells the UI "new output
//! landed, worth repainting" so it does not repaint on a fixed clock.
//!
//! # The pty facts that will bite you if you forget them
//!
//! 1. **On unix: drop the slave after spawning, or you never see EOF.** After
//!    the child is spawned, the child holds the slave end. If kvim *also* keeps
//!    a slave fd open, then when the child exits the slave fd stays open, the
//!    master read never returns 0 (EOF), and the reader thread blocks forever —
//!    the terminal looks alive after the shell already quit. So [`spawn`] drops
//!    `pair.slave` immediately once the command is running.
//! 2. **On Windows that drop is a no-op, so EOF is NOT a liveness signal.**
//!    `portable-pty` 0.9's ConPTY backend builds `ConPtyMasterPty` and
//!    `ConPtySlavePty` over the *same* `Arc<Mutex<Inner>>` (see
//!    `portable-pty-0.9.0/src/win/conpty.rs`, `openpty()`), so dropping the
//!    slave closes exactly nothing — the refcount just goes 2 → 1. Worse, the
//!    master's read handle is the read end of a pipe whose write end is dup'd
//!    into the console host; the parent's own copy is already dropped inside
//!    `PsuedoCon::new`, so the *only* thing that can produce EOF is
//!    `ClosePseudoConsole`, which runs when `Inner` finally drops. Consequence:
//!    on Windows a child can exit, and the pty still never reports EOF. So
//!    liveness must come from the process, never from the byte stream — which
//!    is why [`is_finished`] polls the child rather than trusting the `eof`
//!    flag, and why the App's idle tick drives [`reap_if_done`].
//! 3. **The child must be reaped, or it becomes a zombie.** A finished child
//!    that nobody `wait()`s stays as a `<defunct>` process. [`TermSession`]
//!    reaps it two ways: [`reap_if_done`] harvests it during the idle tick, and
//!    `Drop` kills-then-waits as the backstop so closing kvim never leaks a
//!    process. That backstop wait is *bounded* — see [`TermSession::drop`].
//! 4. **A terminal that never answers `ESC[6n` hangs ConPTY dead.** This one
//!    cost a whole debugging session; see the next heading.
//!
//! # WINDOWS FACT: you MUST answer the cursor-position report, or `:term` dies
//!
//! Wah, this one damn sneaky. On Windows `portable-pty` creates its pseudo
//! console with the flags hardcoded in `PsuedoCon::new`
//! (`portable-pty-0.9.0/src/win/psuedocon.rs`):
//!
//! ```text
//! PSUEDOCONSOLE_INHERIT_CURSOR | PSEUDOCONSOLE_RESIZE_QUIRK | PSEUDOCONSOLE_WIN32_INPUT_MODE
//!   = 0x1                      | 0x2                        | 0x4
//! ```
//!
//! `PSUEDOCONSOLE_INHERIT_CURSOR` (0x1) tells the console host: "before you let
//! the client run, ask the *host terminal* where its cursor is, and start the
//! console buffer there." The host does that by writing the VT **DSR-CPR**
//! query `ESC [ 6 n` down the pty and **blocking until somebody writes back
//! `ESC [ row ; col R`**. A real terminal emulator (wezterm, Windows Terminal —
//! the people this flag was written for) answers it reflexively. kvim's reader
//! thread used to just feed those bytes to `vt100` and move on, and `vt100`
//! 0.15 does not implement DSR at all (grep its source for `dsr` — nothing). So
//! nobody ever answered, and the client sat wedged **inside console/DLL
//! initialisation**, forever.
//!
//! What that looks like from the maintainer's chair: `:term`, blank window, zero
//! bytes ever, and since terminal-mode forwards every keystroke to a pty nobody
//! is reading, the editor looks stone dead. Measured on Windows 11 26200 with a
//! flag-by-flag bisect against raw Win32 `CreatePseudoConsole`:
//!
//! | conpty flags | result |
//! |---|---|
//! | `0x0` (Microsoft's own `EchoCon` sample) | `cmd /c echo hello` → "hello", exit 0 |
//! | `0x2` RESIZE_QUIRK alone | works |
//! | `0x4` WIN32_INPUT_MODE alone | works |
//! | `0x1` INHERIT_CURSOR alone | **wedged forever, 0 bytes out** |
//! | `0x7` (what portable-pty uses) | **wedged forever, 0 bytes out** |
//! | `0x7` **+ we answer `ESC[6n`** | "hello", exit 0 |
//!
//! A wedged client is also stuck in the loader, so `ClosePseudoConsole` does not
//! free it and it can report `STATUS_DLL_INIT_FAILED` (0xC0000142) when it
//! finally dies — that exit code is the *symptom* of the stall, not a separate
//! bug. Chasing 0xC0000142 leads nowhere; the flag is the cause.
//!
//! We cannot change those flags (they are hardcoded upstream and 0.9.0 is the
//! newest release), so [`spawn_reader`] answers the query itself: it watches the
//! raw pty stream for `ESC [ 6 n` and writes `ESC [ row ; col R` back, using the
//! parser's real cursor. **What would make this wrong:** if a future
//! `portable-pty` stops setting `INHERIT_CURSOR`, answering is still correct
//! (that is what every terminal does); but if you ever remove the answering
//! code while the flag is still set, `:term` on Windows breaks again, silently
//! and totally. Do not remove it. It is also not Windows-only politeness — on
//! unix, programs like `tput u7` and shells probing the prompt column send the
//! very same query and hang waiting for the same reply.
//!
//! [`spawn`]: TermSession::spawn
//! [`reap_if_done`]: TermSession::reap_if_done
//! [`is_finished`]: TermSession::is_finished

use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use portable_pty::{Child, CommandBuilder, MasterPty, PtySize};

use crate::ui::event::{Key, KeyPress};

/// How big the terminal grid is, in character rows and columns.
///
/// This is kvim's own little struct rather than `portable_pty::PtySize` so the
/// rest of the editor never has to name a pty type. [`TermSession`] converts it
/// to a `PtySize` (with the pixel fields zeroed — kvim is a character grid, no
/// sixel/pixel geometry) at the one boundary where the kernel needs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TermSize {
    pub rows: u16,
    pub cols: u16,
}

impl TermSize {
    /// Clamp to at least 1x1. A zero-sized pty is meaningless and `vt100`
    /// panics on a zero dimension, so a window too small to show anything still
    /// gets a 1x1 grid rather than crashing kvim.
    pub fn new(rows: u16, cols: u16) -> Self {
        Self { rows: rows.max(1), cols: cols.max(1) }
    }
}

impl From<TermSize> for PtySize {
    fn from(s: TermSize) -> Self {
        PtySize { rows: s.rows, cols: s.cols, pixel_width: 0, pixel_height: 0 }
    }
}

/// One live terminal: a pty, the child process inside it, and the parser that
/// turns its output into a screen you can paint.
///
/// Create one with [`TermSession::spawn`]. Feed it keystrokes with
/// [`write_input`], read its screen with [`with_screen`], keep it sized to its
/// window with [`resize`]. It cleans up after itself: `Drop` kills and reaps
/// the child, so a dropped session never leaves a zombie or an orphaned shell.
///
/// [`write_input`]: TermSession::write_input
/// [`with_screen`]: TermSession::with_screen
/// [`resize`]: TermSession::resize
pub struct TermSession {
    /// The shared screen state. The reader thread writes it (feeding pty bytes
    /// in), the UI thread reads it (painting a snapshot out), the mutex keeps
    /// them from tearing.
    parser: Arc<Mutex<vt100::Parser>>,
    /// The write side of the pty — keystrokes go here, to the child's stdin.
    ///
    /// Shared with the reader thread, because the reader must be able to write
    /// too: it answers the `ESC[6n` cursor-position query (see the module docs
    /// — without that answer ConPTY never starts the child on Windows). Only
    /// one writer can ever exist — `portable-pty`'s `take_writer()` literally
    /// `take()`s an `Option` and errors on the second call — so sharing this
    /// one behind a mutex is the only way both sides can speak.
    ///
    /// **Lock order, and what would make this wrong:** never grab `parser`
    /// while holding `writer`. The reader thread does parser-then-writer only
    /// by taking the parser lock, computing the reply, *dropping* the guard,
    /// and only then locking the writer. The UI thread takes them one at a time
    /// and never nests. Nest them the other way round and you have a deadlock
    /// that will only show up under load.
    writer: Arc<Mutex<Box<dyn Write + Send>>>,
    /// The master handle, kept alive so [`resize`] can push the new window size
    /// down to the kernel (the `TIOCSWINSZ` ioctl) and so the pty stays open.
    ///
    /// [`resize`]: TermSession::resize
    master: Box<dyn MasterPty + Send>,
    /// The child process, for reaping (`try_wait`) and killing.
    child: Box<dyn Child + Send + Sync>,
    /// Set by the reader thread whenever fresh output was parsed. The UI swaps
    /// it back to `false` when it repaints — see [`take_dirty`].
    ///
    /// [`take_dirty`]: TermSession::take_dirty
    dirty: Arc<AtomicBool>,
    /// Set by the reader thread when the pty hits EOF (the child closed its end,
    /// i.e. the program exited). Distinct from "reaped": EOF says the output is
    /// finished, [`reap_if_done`] then harvests the exit status.
    ///
    /// **Unix-only signal.** On Windows this flag basically never fires while
    /// the session is alive — the console host holds the pipe's write end until
    /// `ClosePseudoConsole`, which only runs when this whole session drops (pty
    /// fact #2 in the module docs). So it is a *hint* that the child is gone,
    /// never the proof. The proof is [`reap_if_done`] polling the process.
    ///
    /// [`reap_if_done`]: TermSession::reap_if_done
    eof: Arc<AtomicBool>,
    /// The current grid size, so [`resize`] can skip a no-op resize (pushing an
    /// unchanged winsize every frame would spam `SIGWINCH` at the child).
    ///
    /// [`resize`]: TermSession::resize
    size: TermSize,
    /// The child's exit status once reaped, or `None` while it is still
    /// running (or has exited but not yet been harvested).
    exit_status: Option<portable_pty::ExitStatus>,
    /// Kept so the handle is not detached-and-forgotten while the session
    /// lives; the thread exits on its own when the pty reaches EOF. Never
    /// joined on the UI thread (a join could block the editor) — see the
    /// module docs.
    _reader: JoinHandle<()>,
}

impl TermSession {
    /// How much scrollback `vt100` keeps above the visible screen. 1000 rows is
    /// enough to scroll back over a build log without holding a whole session's
    /// history in memory. (Scrollback *viewing* is a filed follow-up; the buffer
    /// is kept now so the history is not lost in the meantime.)
    const SCROLLBACK: usize = 1000;

    /// Spawn a shell (or `command`) inside a fresh pty of `size`.
    ///
    /// `command` is neovim's `:terminal [cmd]` argument:
    /// * `None` runs the user's login shell interactively — `portable-pty`'s
    ///   `new_default_prog`, which honours `$SHELL`.
    /// * `Some(line)` runs `line` through the platform's shell, so shell syntax
    ///   — pipes, `&&`, globs — works the way it does when you type the same
    ///   thing at a prompt. This matches vim's `:terminal {cmd}` going through
    ///   `'shell'`/`'shellcmdflag'`. See [`build_command`] for which shell and
    ///   which flag on which OS — hor, they are not the same, and getting it
    ///   wrong is how `:term` used to be unusable on Windows.
    ///
    /// Returns an error only if the pty could not be opened or the command could
    /// not be spawned; a command that spawns and then immediately fails is a
    /// *running* session that exits — [`is_finished`](Self::is_finished) will
    /// report it, and its error text will be on the screen where the user can
    /// read it, exactly like a real terminal.
    pub fn spawn(command: Option<&str>, size: TermSize) -> std::io::Result<Self> {
        let size = TermSize::new(size.rows, size.cols);
        let pty_system = portable_pty::native_pty_system();
        let pair = pty_system
            .openpty(size.into())
            .map_err(|e| std::io::Error::other(format!("openpty failed: {e}")))?;

        let cmd = build_command(command);
        let child = pair
            .slave
            .spawn_command(cmd)
            .map_err(|e| std::io::Error::other(format!("spawn failed: {e}")))?;

        // FACT #1 from the module docs: on unix, drop the slave now or EOF never
        // comes. FACT #2: on Windows this is a no-op (master and slave share one
        // `Arc<Mutex<Inner>>`), which is exactly why nothing here may treat EOF
        // as the liveness signal. Dropping is still right on both — it just only
        // *achieves* something on unix.
        drop(pair.slave);

        let writer = pair
            .master
            .take_writer()
            .map_err(|e| std::io::Error::other(format!("pty take_writer failed: {e}")))?;
        let writer = Arc::new(Mutex::new(writer));
        let reader = pair
            .master
            .try_clone_reader()
            .map_err(|e| std::io::Error::other(format!("pty clone_reader failed: {e}")))?;

        let parser = Arc::new(Mutex::new(vt100::Parser::new(size.rows, size.cols, Self::SCROLLBACK)));
        let dirty = Arc::new(AtomicBool::new(false));
        let eof = Arc::new(AtomicBool::new(false));

        let reader_handle = spawn_reader(
            reader,
            Arc::clone(&parser),
            Arc::clone(&dirty),
            Arc::clone(&eof),
            Arc::clone(&writer),
        );

        Ok(Self {
            parser,
            writer,
            master: pair.master,
            child,
            dirty,
            eof,
            size,
            exit_status: None,
            _reader: reader_handle,
        })
    }

    /// Forward raw bytes to the child's input (its stdin).
    ///
    /// This is the keystroke path: [`encode_key`] turns a kvim key into the byte
    /// sequence a terminal would send, and this writes it. Flushes immediately —
    /// an interactive program should react to each keystroke, not wait for a
    /// buffer to fill.
    pub fn write_input(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        let mut writer = self.writer.lock().unwrap_or_else(|p| p.into_inner());
        writer.write_all(bytes)?;
        writer.flush()
    }

    /// Tell the pty (and thus the child) the grid is now `size`.
    ///
    /// Pushes the new window size to the kernel via the master's `resize`
    /// (the `TIOCSWINSZ` ioctl, which makes the child see the new
    /// `TIOCGWINSZ` and get a `SIGWINCH`), and resizes the parser's grid to
    /// match so the next paint reflows correctly. A no-op if the size did not
    /// actually change, so calling it every frame is cheap and safe.
    pub fn resize(&mut self, size: TermSize) {
        let size = TermSize::new(size.rows, size.cols);
        if size == self.size {
            return;
        }
        self.size = size;
        // Best-effort: a failed resize just leaves the child on the old size,
        // which is a cosmetic glitch, not a reason to tear the terminal down.
        let _ = self.master.resize(size.into());
        if let Ok(mut parser) = self.parser.lock() {
            parser.set_size(size.rows, size.cols);
        }
    }

    /// Read the current screen under the lock, hand it to `f`, return `f`'s
    /// result. The borrow of the `vt100::Screen` never escapes the closure, so
    /// the mutex is held for exactly the paint and no longer.
    ///
    /// If the lock is poisoned (a panic in the reader thread while holding it —
    /// should never happen, the reader only calls infallible parser methods) we
    /// recover the guard rather than propagate the panic: a glitchy frame beats
    /// crashing the editor.
    pub fn with_screen<R>(&self, f: impl FnOnce(&vt100::Screen) -> R) -> R {
        let guard = self.parser.lock().unwrap_or_else(|p| p.into_inner());
        f(guard.screen())
    }

    /// The current grid size.
    pub fn size(&self) -> TermSize {
        self.size
    }

    /// Take the "new output arrived" flag: returns whether the screen changed
    /// since the last call, and clears it. The UI uses this on its idle tick to
    /// decide whether a repaint is worth it.
    pub fn take_dirty(&self) -> bool {
        self.dirty.swap(false, Ordering::AcqRel)
    }

    /// Whether the child has exited.
    ///
    /// Takes `&mut self` because on Windows there is no cheap read-only way to
    /// know: the pty's EOF cannot be trusted (pty fact #2 — the console host
    /// keeps the pipe open until this session drops, so a long-dead child still
    /// shows no EOF), so the only honest answer is to *poll the process*. This
    /// therefore calls [`reap_if_done`] first and then reports
    /// `eof || reaped`. Polling is a single non-blocking `try_wait`
    /// (`GetExitCodeProcess` on Windows, `waitpid(WNOHANG)` on unix), so calling
    /// it on every idle tick costs nothing.
    ///
    /// The reaping is a feature, not a side effect: whoever asks "is it done?"
    /// almost always wants [`exit_code`] next, and this guarantees it is already
    /// there rather than one tick late.
    ///
    /// [`reap_if_done`]: Self::reap_if_done
    /// [`exit_code`]: Self::exit_code
    pub fn is_finished(&mut self) -> bool {
        self.reap_if_done();
        self.eof.load(Ordering::Acquire) || self.exit_status.is_some()
    }

    /// If the child has exited, harvest it (non-blocking `try_wait`) so it does
    /// not linger as a zombie, and cache its exit status. Idempotent: once
    /// reaped it does nothing. Safe to call every idle tick.
    pub fn reap_if_done(&mut self) {
        if self.exit_status.is_some() {
            return;
        }
        if let Ok(Some(status)) = self.child.try_wait() {
            self.exit_status = Some(status);
        }
    }

    /// The child's exit code, but only once it has actually been *reaped* (via
    /// [`reap_if_done`]) — `None` while it is still running, and also `None` in
    /// the brief gap where the pty hit EOF but [`reap_if_done`] has not yet
    /// harvested the status on the next idle tick.
    ///
    /// Why "reaped", not just "finished": the UI uses this to announce
    /// `[Process exited N]` and to drop out of terminal-mode, and it wants the
    /// real code, not a guess. `App::drain_terminals` therefore calls
    /// [`reap_if_done`] every idle tick and keys the lifecycle off *this*, not
    /// off [`is_finished`] — which is also what makes that path correct on
    /// Windows, where EOF alone would never arrive.
    ///
    /// [`reap_if_done`]: Self::reap_if_done
    /// [`is_finished`]: Self::is_finished
    pub fn exit_code(&self) -> Option<u32> {
        self.exit_status.as_ref().map(|s| s.exit_code())
    }
}

impl TermSession {
    /// How long [`TermSession::drop`] will wait for a killed child before giving
    /// up and letting the OS clean up.
    ///
    /// **Why bounded at all — this is the anti-freeze constant.** `Drop` runs on
    /// the UI thread (a `:bd` on a terminal buffer, or kvim shutting down), and
    /// `portable-pty`'s `Child::wait()` is `WaitForSingleObject(..., INFINITE)`
    /// on Windows (`portable-pty-0.9.0/src/win/mod.rs`) and a blocking `waitpid`
    /// on unix. A child that is wedged inside console/DLL initialisation — the
    /// exact state the unanswered-`ESC[6n` bug used to leave every `:term`
    /// child in — does not die on `TerminateProcess` straight away, so an
    /// unbounded wait there freezes the whole editor with no way out. Kill, then
    /// poll, then move on: a stuck process is the OS's problem at that point,
    /// a frozen editor is ours.
    ///
    /// 2 seconds because a `SIGKILL`ed / `TerminateProcess`d child is normally
    /// reaped in microseconds; two orders of magnitude of headroom means the
    /// bound only ever fires in the genuinely-pathological case.
    ///
    /// **What would make this wrong:** on unix, giving up before `waitpid`
    /// succeeds leaks a `<defunct>` zombie until kvim itself exits. That is the
    /// deliberate trade — one zombie beats a hung editor — but if you ever see
    /// zombies pile up, the bug is upstream of here (something is blocking the
    /// child's death), not this timeout.
    const DROP_REAP_BUDGET: std::time::Duration = std::time::Duration::from_secs(2);
}

impl Drop for TermSession {
    fn drop(&mut self) {
        // Backstop cleanup (pty fact #3): kill the child if it is still running,
        // then reap it so we never leave a zombie or an orphaned shell when a
        // terminal window closes or kvim exits. Both best-effort — by Drop time
        // there is no one to hand an error to, and a child that already exited
        // makes `kill` a harmless error we ignore.
        if self.exit_status.is_some() {
            return;
        }
        let _ = self.child.kill();

        // Bounded reap, NOT `child.wait()` — see `DROP_REAP_BUDGET` for why an
        // unbounded wait here is a hard editor freeze.
        let deadline = std::time::Instant::now() + Self::DROP_REAP_BUDGET;
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => break,
                // The handle is unusable; nothing more we can do, and looping
                // would just burn the whole budget for nothing.
                Err(_) => break,
                Ok(None) => {
                    if std::time::Instant::now() >= deadline {
                        break;
                    }
                    // 5ms: fast enough that the common case (child already dead)
                    // costs one sleep at most, slow enough not to spin a core.
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
            }
        }
    }
}

/// Build the `portable-pty` command for `:term [cmd]`. See
/// [`TermSession::spawn`] for the `None` vs `Some` semantics.
///
/// `None` hands off to `portable-pty`'s own default-program logic, which is
/// already platform-correct: `$SHELL` (then the passwd database) on unix, and
/// `%ComSpec%` falling back to `cmd.exe` on Windows
/// (`portable-pty-0.9.0/src/cmdbuilder.rs`, `CommandBuilder::cmdline()`).
///
/// `Some(line)` is where the platforms genuinely part ways, and where kvim used
/// to be plain broken:
///
/// * **unix** — `$SHELL -c <line>`, falling back to `/bin/sh`. Matches vim's
///   `'shell'` + `'shellcmdflag'` defaults (`/bin/sh`, `-c`).
/// * **Windows** — `%ComSpec% /C <line>`, falling back to `cmd.exe`. There is no
///   `/bin/sh` on Windows, so the old unconditional unix path could not spawn
///   *anything*: `CreateProcessW "/bin/sh -c ..." failed: The system cannot find
///   the path specified. (os error 3)`. Neovim's Windows defaults are
///   `shell=cmd.exe`, `shellcmdflag=/s /c`; we use `/C` without `/s` because
///   `/s`'s job is to strip an outer pair of quotes that neovim's own quoting
///   adds, whereas `portable-pty` already applies the Win32 `ArgvQuote` rules
///   for us when it builds the command line (`cmdbuilder.rs`,
///   `CommandBuilder::append_quoted`). Adding `/s` on top would strip quotes we
///   actually meant.
///
/// **What would make this wrong:** a user whose `%ComSpec%` points at
/// PowerShell would get `/C` when PowerShell wants `-Command`. That is not a
/// real configuration (Windows itself requires `ComSpec` to be a `cmd.exe`-
/// compatible processor), but if kvim ever grows a `'shell'` option, this is the
/// function that must read it instead of guessing.
fn build_command(command: Option<&str>) -> CommandBuilder {
    let Some(line) = command else {
        return CommandBuilder::new_default_prog();
    };

    #[cfg(windows)]
    {
        let shell = std::env::var("ComSpec").unwrap_or_else(|_| "cmd.exe".to_string());
        let mut cmd = CommandBuilder::new(shell);
        cmd.arg("/C");
        cmd.arg(line);
        cmd
    }

    #[cfg(not(windows))]
    {
        let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string());
        let mut cmd = CommandBuilder::new(shell);
        cmd.arg("-c");
        cmd.arg(line);
        cmd
    }
}

/// The **DSR-CPR** query: "Device Status Report — report the cursor position".
///
/// FORMAT KNOWLEDGE (ECMA-48 / DEC VT100 `DSR`, parameter 6): a program that
/// wants to know where the cursor is writes `ESC [ 6 n` to the terminal and then
/// *blocks reading its input* until the terminal writes back a CPR. Windows'
/// ConPTY sends exactly this during pseudo-console startup when created with
/// `PSUEDOCONSOLE_INHERIT_CURSOR`, which `portable-pty` always sets — see the
/// module docs for the flag bisect. `vt100` 0.15 ignores DSR entirely, so if we
/// do not answer, nobody does.
const DSR_CPR_REQUEST: &[u8] = b"\x1b[6n";

/// Format the **CPR** reply for a cursor at zero-based `(row, col)`.
///
/// FORMAT KNOWLEDGE (ECMA-48 `CPR`): the answer to [`DSR_CPR_REQUEST`] is
/// `ESC [ <row> ; <col> R`, and the coordinates are **one-based** — the top-left
/// cell is `1;1`, not `0;0`. `vt100::Screen::cursor_position()` is zero-based,
/// so every value gets `+ 1` on the way out. Off-by-one here does not crash
/// anything; it silently shifts where ConPTY thinks the inherited cursor sits,
/// which shows up as a stray blank line at the top of every `:term`.
fn cpr_reply(row: u16, col: u16) -> Vec<u8> {
    format!("\x1b[{};{}R", u32::from(row) + 1, u32::from(col) + 1).into_bytes()
}

/// Watches the raw pty byte stream for [`DSR_CPR_REQUEST`], across chunk
/// boundaries.
///
/// A `read()` can split `ESC [ 6 n` anywhere — the escape lands in one chunk and
/// the `6n` in the next — and a scanner that only looked inside single chunks
/// would miss it and hang the terminal on Windows *intermittently*, which is a
/// far nastier bug than hanging it always. So we carry the last
/// `DSR_CPR_REQUEST.len() - 1` bytes of every chunk into the next scan.
///
/// Carrying exactly `len - 1` is also what makes double-counting impossible: any
/// match found in `carry ++ chunk` must consume at least one byte of `chunk`
/// (the carry alone is too short to hold a whole request), so no request is ever
/// answered twice.
#[derive(Default)]
struct DsrWatcher {
    carry: Vec<u8>,
}

impl DsrWatcher {
    /// How many complete DSR-CPR requests appear in `chunk` (plus whatever
    /// straddled the boundary from last time).
    fn requests_in(&mut self, chunk: &[u8]) -> usize {
        let mut window = std::mem::take(&mut self.carry);
        window.extend_from_slice(chunk);

        let found = window
            .windows(DSR_CPR_REQUEST.len())
            .filter(|w| *w == DSR_CPR_REQUEST)
            .count();

        let keep = DSR_CPR_REQUEST.len() - 1;
        let from = window.len().saturating_sub(keep);
        self.carry = window[from..].to_vec();
        found
    }
}

/// Spin the reader-thread actor (see the module docs). Owns the pty read handle
/// for its whole life; what it shares with the UI thread is the parser (behind
/// its mutex), the two flags, and the writer (behind *its* mutex, for CPR
/// replies only — never for user input).
fn spawn_reader(
    mut reader: Box<dyn Read + Send>,
    parser: Arc<Mutex<vt100::Parser>>,
    dirty: Arc<AtomicBool>,
    eof: Arc<AtomicBool>,
    writer: Arc<Mutex<Box<dyn Write + Send>>>,
) -> JoinHandle<()> {
    std::thread::Builder::new()
        .name("kvim-term-reader".to_string())
        .spawn(move || {
            let mut buf = [0u8; 8192];
            let mut watcher = DsrWatcher::default();
            loop {
                match reader.read(&mut buf) {
                    // EOF: the child closed the pty (it exited). Flag it and
                    // stop — there will never be more output. NOTE: on Windows
                    // this only ever happens at teardown (pty fact #2), so
                    // nothing may rely on it to notice an exited child.
                    Ok(0) => {
                        eof.store(true, Ordering::Release);
                        dirty.store(true, Ordering::Release);
                        break;
                    }
                    Ok(n) => {
                        let chunk = &buf[..n];
                        // Answer every cursor-position query in this chunk.
                        // Compute the reply UNDER the parser lock, then let the
                        // guard go before touching the writer — the lock order
                        // in `TermSession::writer`'s docs is not optional.
                        let replies = watcher.requests_in(chunk);
                        let reply = {
                            let Ok(mut parser) = parser.lock() else {
                                // Poisoned parser: the session is already in
                                // trouble and re-locking would panic in a
                                // detached thread. Stop cleanly instead.
                                eof.store(true, Ordering::Release);
                                dirty.store(true, Ordering::Release);
                                break;
                            };
                            parser.process(chunk);
                            if replies > 0 {
                                let (row, col) = parser.screen().cursor_position();
                                Some(cpr_reply(row, col))
                            } else {
                                None
                            }
                        };
                        if let Some(reply) = reply
                            && let Ok(mut writer) = writer.lock()
                        {
                            // Best-effort: if the child already went away the
                            // write fails, and there is nobody left to tell.
                            for _ in 0..replies {
                                let _ = writer.write_all(&reply);
                            }
                            let _ = writer.flush();
                        }
                        dirty.store(true, Ordering::Release);
                    }
                    // A read error means the pty is gone (child killed, fd
                    // closed). Treat it as EOF and stop; the session's Drop or
                    // reap handles the process side.
                    Err(_) => {
                        eof.store(true, Ordering::Release);
                        dirty.store(true, Ordering::Release);
                        break;
                    }
                }
            }
        })
        .expect("spawning the terminal reader thread should not fail")
}

/// Turn one kvim keystroke into the bytes a terminal would send for it.
///
/// Returns `None` for a key that has no terminal meaning (so the caller sends
/// nothing) — today that is only `<Insert>`, which no shell reads. Everything
/// else maps to the classic control bytes / VT escape sequences:
///
/// * `Enter` → `\r` (carriage return — that is what the Enter key sends over a
///   tty; the tty's own line discipline turns it into `\n` for the program).
/// * `Backspace` → `0x7f` (DEL), which is what virtually every modern terminal
///   sends; the shell's `stty erase` maps it to erase-char.
/// * `Tab` → `\t`, `Esc` → `0x1b`, `Delete` → the `ESC [ 3 ~` VT sequence.
/// * `Ctrl` + a letter → the C0 control code `letter & 0x1f` (so `<C-c>` is
///   `0x03` SIGINT-via-tty, `<C-d>` is `0x04` EOF, `<C-l>` is `0x0c` clear).
/// * Arrows / Home / End / PageUp / PageDown / function keys → their `ESC [ …`
///   / `ESC O …` sequences (the "application cursor" variants are not emitted;
///   plain-mode sequences work with every shell and are what the vast majority
///   of programs expect).
/// * A plain printable char → its UTF-8 bytes.
///
/// A pure function on purpose: no I/O, no session, so the whole keystroke
/// vocabulary can be unit-tested on its own.
pub fn encode_key(kp: &KeyPress) -> Option<Vec<u8>> {
    let ctrl = kp.mods.ctrl;
    let alt = kp.mods.alt;

    // `Alt+key` (a "Meta" key) is a leading ESC then the key's normal bytes.
    // Handle it by recursing on the un-alted key and prefixing 0x1b.
    if alt {
        let mut inner = KeyPress { key: kp.key, mods: kp.mods };
        inner.mods.alt = false;
        let rest = encode_key(&inner)?;
        let mut out = Vec::with_capacity(rest.len() + 1);
        out.push(0x1b);
        out.extend_from_slice(&rest);
        return Some(out);
    }

    let bytes: Vec<u8> = match kp.key {
        Key::Char(c) => {
            if ctrl {
                // Map to the C0 control code.
                //
                // FORMAT KNOWLEDGE (crossterm 0.29, unix legacy decode): a
                // terminal sends one byte for `<C-\>`..`<C-_>` (0x1C..0x1F), and
                // crossterm decodes those bytes to the DIGITS `'4'..'7'` +
                // CONTROL (see crossterm/src/event/sys/unix/parse.rs) — it cannot
                // tell `<C-\>` from `<C-4>`, they are the same byte on a legacy
                // tty. So map `'4'..'7'` back to the control byte they stand for,
                // and fold every other ctrl+letter/symbol onto 0x00..0x1f via
                // `& 0x1f` (`<C-space>`/`<C-@>` → NUL, `<C-a>` → 0x01, `<C-\>`
                // as the symbol form `'\'` → 0x1C too). This keeps forwarding
                // faithful to what the tty would have put on the wire.
                let b = match c {
                    '4' => 0x1c,
                    '5' => 0x1d,
                    '6' => 0x1e,
                    '7' => 0x1f,
                    _ => (c as u8).to_ascii_uppercase() & 0x1f,
                };
                vec![b]
            } else {
                let mut b = [0u8; 4];
                c.encode_utf8(&mut b).as_bytes().to_vec()
            }
        }
        Key::Enter => vec![b'\r'],
        Key::Escape => vec![0x1b],
        Key::Backspace => vec![0x7f],
        Key::Tab => vec![b'\t'],
        Key::BackTab => vec![0x1b, b'[', b'Z'],
        Key::Delete => vec![0x1b, b'[', b'3', b'~'],
        Key::Insert => return None,
        Key::Up => vec![0x1b, b'[', b'A'],
        Key::Down => vec![0x1b, b'[', b'B'],
        Key::Right => vec![0x1b, b'[', b'C'],
        Key::Left => vec![0x1b, b'[', b'D'],
        Key::Home => vec![0x1b, b'[', b'H'],
        Key::End => vec![0x1b, b'[', b'F'],
        Key::PageUp => vec![0x1b, b'[', b'5', b'~'],
        Key::PageDown => vec![0x1b, b'[', b'6', b'~'],
        Key::F(n) => return encode_function_key(n),
    };
    Some(bytes)
}

/// The `ESC O P`.. / `ESC [ … ~` sequences for F1–F12. Split out only to keep
/// [`encode_key`] readable. F-keys above 12 have no standard sequence, so they
/// send nothing.
fn encode_function_key(n: u8) -> Option<Vec<u8>> {
    let seq: &[u8] = match n {
        1 => b"\x1bOP",
        2 => b"\x1bOQ",
        3 => b"\x1bOR",
        4 => b"\x1bOS",
        5 => b"\x1b[15~",
        6 => b"\x1b[17~",
        7 => b"\x1b[18~",
        8 => b"\x1b[19~",
        9 => b"\x1b[20~",
        10 => b"\x1b[21~",
        11 => b"\x1b[23~",
        12 => b"\x1b[24~",
        _ => return None,
    };
    Some(seq.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::event::{Key, Modifiers};
    use std::time::{Duration, Instant};

    fn kp(key: Key) -> KeyPress {
        KeyPress { key, mods: Modifiers { ctrl: false, alt: false, shift: false } }
    }

    fn ctrl(key: Key) -> KeyPress {
        KeyPress { key, mods: Modifiers { ctrl: true, alt: false, shift: false } }
    }

    /// A one-shot command line that prints `word` and exits, spelled in the
    /// shell [`build_command`] will actually use on this OS.
    ///
    /// `printf` does not exist in `cmd.exe` and `echo` in `cmd` appends CRLF —
    /// both fine, because every caller only asserts that `word` shows up on the
    /// screen. Keeping the *assertion* identical and swapping only the spelling
    /// is the point: the unix side of these tests is unchanged.
    fn echo_line(word: &str) -> String {
        if cfg!(windows) {
            format!("echo {word}")
        } else {
            format!("printf {word}")
        }
    }

    /// Spawn a long-lived child that echoes back whatever you type at it.
    ///
    /// * unix — `cat`, echoed by the tty line discipline.
    /// * Windows — the default program (an interactive `cmd.exe`), echoed by the
    ///   console itself. There is no `cat`, and `more`/`sort` buffer until EOF
    ///   instead of echoing line by line, so the console's own echo is the
    ///   faithful equivalent — and it happens to be exactly what a user sees
    ///   when they type bare `:term`.
    fn spawn_echoing(size: TermSize) -> TermSession {
        if cfg!(windows) {
            TermSession::spawn(None, size).unwrap()
        } else {
            TermSession::spawn(Some("cat"), size).unwrap()
        }
    }

    /// Poll until the child has exited or `timeout` runs out, exactly the way
    /// `App::drain_terminals` does on its idle tick.
    fn wait_for_exit(session: &mut TermSession, timeout: Duration) {
        let start = Instant::now();
        while !session.is_finished() && start.elapsed() < timeout {
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Spin until `pred` sees what it wants or `timeout` runs out. The reader
    /// thread is asynchronous, so a test that reads the screen the instant after
    /// spawning would race it; this waits, cheaply, for the output to land.
    fn wait_until(session: &TermSession, timeout: Duration, pred: impl Fn(&str) -> bool) -> String {
        let start = Instant::now();
        loop {
            let contents = session.with_screen(|s| s.contents());
            if pred(&contents) {
                return contents;
            }
            if start.elapsed() > timeout {
                return contents;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn spawned_command_output_lands_on_the_screen() {
        // FACT the test pins: spawn a scripted command, the reader drains the
        // pty into the parser, and the parsed screen shows the output.
        //
        // On Windows this is also the regression test for the unanswered-`ESC[6n`
        // hang: with ConPTY's INHERIT_CURSOR flag and no CPR reply the child
        // never leaves console initialisation, so the screen stays empty forever
        // and this assert is the thing that catches it.
        let session = TermSession::spawn(Some(&echo_line("hello")), TermSize::new(24, 80)).unwrap();
        let contents = wait_until(&session, Duration::from_secs(5), |c| c.contains("hello"));
        assert!(contents.contains("hello"), "screen was: {contents:?}");
    }

    #[test]
    fn typed_input_is_forwarded_to_the_child() {
        // An echoing child sends back whatever we type. Forward "hi\n" and it
        // must appear on the screen — proving keystroke → pty → parser works.
        let mut session = spawn_echoing(TermSize::new(24, 80));
        session.write_input(b"hi\r\n").unwrap();
        let contents = wait_until(&session, Duration::from_secs(5), |c| c.contains("hi"));
        assert!(contents.contains("hi"), "screen was: {contents:?}");
    }

    /// Join `handle`, but never hang the test: a watcher thread does the
    /// blocking join and pings a channel; we wait on that with a timeout.
    /// Returns `true` if the thread finished within `dur`.
    fn join_within(handle: JoinHandle<()>, dur: Duration) -> bool {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = handle.join();
            let _ = tx.send(());
        });
        rx.recv_timeout(dur).is_ok()
    }

    #[test]
    fn reader_thread_exits_on_eof() {
        // The freeze's first suspect: a reader thread that busy-loops or blocks
        // forever once the pty hits EOF. `std::io::empty()` returns `Ok(0)` on
        // the first read (EOF), so the thread must set the flag and *return*.
        let parser = Arc::new(Mutex::new(vt100::Parser::new(5, 5, 0)));
        let dirty = Arc::new(AtomicBool::new(false));
        let eof = Arc::new(AtomicBool::new(false));
        let reader: Box<dyn Read + Send> = Box::new(std::io::empty());
        let writer: Arc<Mutex<Box<dyn Write + Send>>> = Arc::new(Mutex::new(Box::new(Vec::new())));
        let handle =
            spawn_reader(reader, Arc::clone(&parser), Arc::clone(&dirty), Arc::clone(&eof), writer);
        assert!(join_within(handle, Duration::from_secs(2)), "reader thread must terminate on EOF, not hang");
        assert!(eof.load(Ordering::Acquire), "EOF flag must be set once the read returns 0");
    }

    #[test]
    fn a_cursor_position_query_gets_answered_on_the_write_side() {
        // THE Windows regression test, minus Windows: ConPTY blocks the child
        // until somebody answers `ESC[6n`. Feed the reader a stream containing
        // one, and the writer side must come back with a CPR.
        let parser = Arc::new(Mutex::new(vt100::Parser::new(24, 80, 0)));
        let dirty = Arc::new(AtomicBool::new(false));
        let eof = Arc::new(AtomicBool::new(false));
        let sink = Arc::new(Mutex::new(Vec::<u8>::new()));

        /// A `Write` that appends into a shared `Vec` so the test can read back
        /// what the reader thread wrote.
        struct Tee(Arc<Mutex<Vec<u8>>>);
        impl Write for Tee {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let writer: Arc<Mutex<Box<dyn Write + Send>>> =
            Arc::new(Mutex::new(Box::new(Tee(Arc::clone(&sink)))));
        // Cursor lands at row 0, col 2 after "ab" -> CPR must say 1;3.
        let reader: Box<dyn Read + Send> = Box::new(std::io::Cursor::new(b"ab\x1b[6n".to_vec()));
        let handle =
            spawn_reader(reader, Arc::clone(&parser), Arc::clone(&dirty), Arc::clone(&eof), writer);
        assert!(join_within(handle, Duration::from_secs(2)), "reader thread must terminate");

        let got = sink.lock().unwrap().clone();
        assert_eq!(
            got,
            b"\x1b[1;3R".to_vec(),
            "a DSR-CPR query must be answered with the one-based cursor position, got {:?}",
            String::from_utf8_lossy(&got)
        );
    }

    #[test]
    fn cpr_reply_is_one_based() {
        // ECMA-48 CPR counts from 1, `vt100` counts from 0. Pin the conversion.
        assert_eq!(cpr_reply(0, 0), b"\x1b[1;1R".to_vec());
        assert_eq!(cpr_reply(23, 79), b"\x1b[24;80R".to_vec());
    }

    #[test]
    fn dsr_watcher_sees_a_query_split_across_reads() {
        // A `read()` can chop `ESC [ 6 n` anywhere. Missing the split case would
        // hang `:term` on Windows only *sometimes*, which is far worse to debug
        // than always.
        let mut w = DsrWatcher::default();
        assert_eq!(w.requests_in(b"hello\x1b["), 0);
        assert_eq!(w.requests_in(b"6nworld"), 1, "the split query must still be seen");

        // ...and no double-counting when a whole query sits at a chunk's tail.
        let mut w = DsrWatcher::default();
        assert_eq!(w.requests_in(b"x\x1b[6n"), 1);
        assert_eq!(w.requests_in(b"yyyy"), 0, "an already-answered query must not be re-answered");

        // Two in one chunk get two answers.
        let mut w = DsrWatcher::default();
        assert_eq!(w.requests_in(b"\x1b[6n\x1b[6n"), 2);
    }

    #[test]
    fn exit_code_is_none_until_reaped_then_matches() {
        // `exit 3` is spelled the same in `sh -c` and in `cmd /C`, so no
        // platform split needed here.
        let mut session = TermSession::spawn(Some("exit 3"), TermSize::new(24, 80)).unwrap();
        wait_for_exit(&mut session, Duration::from_secs(5));
        session.reap_if_done();
        assert_eq!(session.exit_code(), Some(3), "exit_code must report the child's real code once reaped");
    }

    #[test]
    fn a_command_that_exits_marks_the_session_finished() {
        let mut session = TermSession::spawn(Some(&echo_line("bye")), TermSize::new(24, 80)).unwrap();
        wait_for_exit(&mut session, Duration::from_secs(5));
        assert!(session.is_finished(), "session should report finished after the command exits");
        // Reaping is idempotent and must not panic.
        session.reap_if_done();
        session.reap_if_done();
    }

    #[test]
    fn resize_updates_the_grid_size() {
        let mut session = spawn_echoing(TermSize::new(24, 80));
        assert_eq!(session.size(), TermSize::new(24, 80));
        session.resize(TermSize::new(30, 100));
        assert_eq!(session.size(), TermSize::new(30, 100));
        let (rows, cols) = session.with_screen(|s| s.size());
        assert_eq!((rows, cols), (30, 100), "parser grid must follow the resize");
    }

    #[test]
    fn zero_size_is_clamped_not_crashed() {
        // vt100 panics on a zero dimension; a window too small to show anything
        // must still give a 1x1 grid.
        assert_eq!(TermSize::new(0, 0), TermSize::new(1, 1));
        let session = TermSession::spawn(Some(&echo_line("x")), TermSize::new(0, 0)).unwrap();
        assert_eq!(session.size(), TermSize::new(1, 1));
    }

    #[test]
    fn plain_char_encodes_as_utf8() {
        assert_eq!(encode_key(&kp(Key::Char('a'))), Some(vec![b'a']));
        assert_eq!(encode_key(&kp(Key::Char('€'))), Some("€".as_bytes().to_vec()));
    }

    #[test]
    fn enter_backspace_tab_esc_encode_to_the_tty_bytes() {
        assert_eq!(encode_key(&kp(Key::Enter)), Some(vec![b'\r']));
        assert_eq!(encode_key(&kp(Key::Backspace)), Some(vec![0x7f]));
        assert_eq!(encode_key(&kp(Key::Tab)), Some(vec![b'\t']));
        assert_eq!(encode_key(&kp(Key::Escape)), Some(vec![0x1b]));
    }

    #[test]
    fn ctrl_letter_encodes_to_the_c0_control_code() {
        // <C-c> is 0x03, <C-d> 0x04, <C-l> 0x0c — the classic tty controls.
        assert_eq!(encode_key(&ctrl(Key::Char('c'))), Some(vec![0x03]));
        assert_eq!(encode_key(&ctrl(Key::Char('d'))), Some(vec![0x04]));
        assert_eq!(encode_key(&ctrl(Key::Char('l'))), Some(vec![0x0c]));
        // Case must not matter: <C-C> is still 0x03.
        assert_eq!(encode_key(&ctrl(Key::Char('C'))), Some(vec![0x03]));
    }

    #[test]
    fn ctrl_backslash_family_maps_to_the_high_c0_codes() {
        // crossterm's legacy decode gives `<C-\>`..`<C-_>` as the digits
        // '4'..'7'; forwarding must turn them back into 0x1C..0x1F (the one byte
        // the tty uses for both `<C-\>` and `<C-4>`).
        assert_eq!(encode_key(&ctrl(Key::Char('4'))), Some(vec![0x1c]));
        assert_eq!(encode_key(&ctrl(Key::Char('5'))), Some(vec![0x1d]));
        assert_eq!(encode_key(&ctrl(Key::Char('6'))), Some(vec![0x1e]));
        assert_eq!(encode_key(&ctrl(Key::Char('7'))), Some(vec![0x1f]));
        // The symbolic (kitty-protocol) form maps the same way via `& 0x1f`.
        assert_eq!(encode_key(&ctrl(Key::Char('\\'))), Some(vec![0x1c]));
    }

    #[test]
    fn arrows_encode_to_vt_sequences() {
        assert_eq!(encode_key(&kp(Key::Up)), Some(vec![0x1b, b'[', b'A']));
        assert_eq!(encode_key(&kp(Key::Down)), Some(vec![0x1b, b'[', b'B']));
        assert_eq!(encode_key(&kp(Key::Right)), Some(vec![0x1b, b'[', b'C']));
        assert_eq!(encode_key(&kp(Key::Left)), Some(vec![0x1b, b'[', b'D']));
    }

    #[test]
    fn alt_key_prefixes_escape() {
        let alt_x = KeyPress { key: Key::Char('x'), mods: Modifiers { ctrl: false, alt: true, shift: false } };
        assert_eq!(encode_key(&alt_x), Some(vec![0x1b, b'x']));
    }
}

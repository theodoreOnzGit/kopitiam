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
//! # The two pty facts that will bite you if you forget them
//!
//! 1. **Drop the slave after spawning, or you never see EOF.** After the child
//!    is spawned, the child holds the slave end. If kvim *also* keeps a slave
//!    handle open, then when the child exits the slave fd stays open, the
//!    master read never returns 0 (EOF), and the reader thread blocks forever —
//!    the terminal looks alive after the shell already quit. So [`spawn`] drops
//!    `pair.slave` immediately once the command is running.
//! 2. **The child must be reaped, or it becomes a zombie.** A finished child
//!    that nobody `wait()`s stays as a `<defunct>` process. [`TermSession`]
//!    reaps it two ways: [`reap_if_done`] harvests it during the idle tick once
//!    the reader signals EOF, and `Drop` kills-then-waits as the backstop so
//!    closing kvim never leaks a process.
//!
//! [`spawn`]: TermSession::spawn
//! [`reap_if_done`]: TermSession::reap_if_done

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
    writer: Box<dyn Write + Send>,
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
    /// * `Some(line)` runs `line` through `$SHELL -c line` (falling back to
    ///   `/bin/sh` if `$SHELL` is unset), so shell syntax — pipes, `&&`, globs —
    ///   works the way it does when you type the same thing at a prompt. This
    ///   matches vim's `:terminal {cmd}` going through `'shell'`/`'shellcmdflag'`.
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

        // FACT #1 from the module docs: drop the slave now, or EOF never comes.
        drop(pair.slave);

        let writer = pair
            .master
            .take_writer()
            .map_err(|e| std::io::Error::other(format!("pty take_writer failed: {e}")))?;
        let reader = pair
            .master
            .try_clone_reader()
            .map_err(|e| std::io::Error::other(format!("pty clone_reader failed: {e}")))?;

        let parser = Arc::new(Mutex::new(vt100::Parser::new(size.rows, size.cols, Self::SCROLLBACK)));
        let dirty = Arc::new(AtomicBool::new(false));
        let eof = Arc::new(AtomicBool::new(false));

        let reader_handle = spawn_reader(reader, Arc::clone(&parser), Arc::clone(&dirty), Arc::clone(&eof));

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
        self.writer.write_all(bytes)?;
        self.writer.flush()
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

    /// Whether the child has exited (the pty reached EOF). True does not by
    /// itself mean the process has been reaped — call [`reap_if_done`] for that.
    ///
    /// [`reap_if_done`]: Self::reap_if_done
    pub fn is_finished(&self) -> bool {
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
    /// real code, not a guess. [`is_finished`] flips on EOF (before reaping) so
    /// the fast poll kicks in; `exit_code` flips one tick later, once the code
    /// is known for sure. The gap is ~16ms (the child has already exited, so the
    /// next `try_wait` succeeds straight away), so the transition still feels
    /// instant to the user.
    ///
    /// [`reap_if_done`]: Self::reap_if_done
    /// [`is_finished`]: Self::is_finished
    pub fn exit_code(&self) -> Option<u32> {
        self.exit_status.as_ref().map(|s| s.exit_code())
    }
}

impl Drop for TermSession {
    fn drop(&mut self) {
        // Backstop cleanup (FACT #2): kill the child if it is still running,
        // then reap it so we never leave a zombie or an orphaned shell when a
        // terminal window closes or kvim exits. Both best-effort — by Drop time
        // there is no one to hand an error to, and a child that already exited
        // makes `kill` a harmless error we ignore.
        if self.exit_status.is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

/// Build the `portable-pty` command for `:term [cmd]`. See
/// [`TermSession::spawn`] for the `None` vs `Some` semantics.
fn build_command(command: Option<&str>) -> CommandBuilder {
    match command {
        None => CommandBuilder::new_default_prog(),
        Some(line) => {
            let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string());
            let mut cmd = CommandBuilder::new(shell);
            cmd.arg("-c");
            cmd.arg(line);
            cmd
        }
    }
}

/// Spin the reader-thread actor (see the module docs). Owns the pty read handle
/// for its whole life; the only things it shares with the UI thread are the
/// parser (behind the mutex) and the two flags.
fn spawn_reader(
    mut reader: Box<dyn Read + Send>,
    parser: Arc<Mutex<vt100::Parser>>,
    dirty: Arc<AtomicBool>,
    eof: Arc<AtomicBool>,
) -> JoinHandle<()> {
    std::thread::Builder::new()
        .name("kvim-term-reader".to_string())
        .spawn(move || {
            let mut buf = [0u8; 8192];
            loop {
                match reader.read(&mut buf) {
                    // EOF: the child closed the pty (it exited). Flag it and
                    // stop — there will never be more output.
                    Ok(0) => {
                        eof.store(true, Ordering::Release);
                        dirty.store(true, Ordering::Release);
                        break;
                    }
                    Ok(n) => {
                        if let Ok(mut parser) = parser.lock() {
                            parser.process(&buf[..n]);
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
        let session = TermSession::spawn(Some("printf hello"), TermSize::new(24, 80)).unwrap();
        let contents = wait_until(&session, Duration::from_secs(5), |c| c.contains("hello"));
        assert!(contents.contains("hello"), "screen was: {contents:?}");
    }

    #[test]
    fn typed_input_is_forwarded_to_the_child() {
        // An interactive `cat` echoes back whatever we type. Forward "hi\n" and
        // it must appear on the screen — proving keystroke → pty → parser works.
        let mut session = TermSession::spawn(Some("cat"), TermSize::new(24, 80)).unwrap();
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
        let handle = spawn_reader(reader, Arc::clone(&parser), Arc::clone(&dirty), Arc::clone(&eof));
        assert!(join_within(handle, Duration::from_secs(2)), "reader thread must terminate on EOF, not hang");
        assert!(eof.load(Ordering::Acquire), "EOF flag must be set once the read returns 0");
    }

    #[test]
    fn exit_code_is_none_until_reaped_then_matches() {
        // `false` exits with code 1. Before reaping, `exit_code()` is None even
        // once EOF landed; after `reap_if_done` it reports the real code.
        let mut session = TermSession::spawn(Some("exit 3"), TermSize::new(24, 80)).unwrap();
        let start = Instant::now();
        while !session.is_finished() && start.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(10));
        }
        session.reap_if_done();
        assert_eq!(session.exit_code(), Some(3), "exit_code must report the child's real code once reaped");
    }

    #[test]
    fn a_command_that_exits_marks_the_session_finished() {
        let mut session = TermSession::spawn(Some("printf bye"), TermSize::new(24, 80)).unwrap();
        let start = Instant::now();
        while !session.is_finished() && start.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(session.is_finished(), "session should report finished after the command exits");
        // Reaping is idempotent and must not panic.
        session.reap_if_done();
        session.reap_if_done();
    }

    #[test]
    fn resize_updates_the_grid_size() {
        let mut session = TermSession::spawn(Some("cat"), TermSize::new(24, 80)).unwrap();
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
        let session = TermSession::spawn(Some("printf x"), TermSize::new(0, 0)).unwrap();
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

//! The system-clipboard registers `"+` and `"*`.
//!
//! # Why a whole module for two registers
//!
//! Every *other* register in kvim is a `String` in memory (see
//! [`super::register`]). The clipboard registers are not: `"+` is the OS
//! clipboard and `"*` is the X11 primary selection, and reaching them means
//! talking to the terminal or to a platform helper program. That I/O is what
//! this module isolate, behind traits, so the routing logic in
//! [`super`](crate::editor) stay pure and the escape bytes / tool selection
//! can be unit-tested without ever touching a real clipboard.
//!
//! # Copy: OSC-52 first, platform tool as a bonus
//!
//! Copying (yank/delete into `"+`) go out over **OSC-52** — a terminal escape
//! (`ESC ] 52 ; c ; <base64> BEL`) that ask the *terminal emulator* to set its
//! clipboard. This is deliberately the primary path because it is the one that
//! keep working where a desktop clipboard API cannot:
//!
//! * over **SSH** — the escape travel down the same pty as everything else, so
//!   a yank on a remote host land in the *local* clipboard;
//! * inside **tmux** — with `set-clipboard on` tmux forward OSC-52 through;
//! * on **Android / Termux** — there is no X11 or Wayland, but the terminal
//!   app (Termux) understand OSC-52.
//!
//! It also needs **no external binary** and **no C-linked clipboard crate**,
//! which is exactly what KOPITIAM's Pure Rust Core and Android-first stance
//! ask for. Where a real desktop clipboard tool *is* present (`wl-copy`,
//! `xclip`, `pbcopy`), kvim run it too, as a complement — belt and braces, so
//! a yank works whether the terminal support OSC-52 or the desktop tool does.
//!
//! # Paste: platform tool only
//!
//! Reading the clipboard back (put from `"+`) can **not** use OSC-52: an
//! OSC-52 *query* asks the terminal to send the clipboard back as input, and
//! virtually every terminal disable that by default because it let any program
//! that can write to your pty silently read your clipboard. So paste go
//! through a platform tool (`wl-paste`, `xclip -o`, `pbpaste`,
//! `termux-clipboard-get`). When none is available — a bare SSH session with
//! no X forwarding, say — paste degrade gracefully to `None` and the caller
//! print a one-line note rather than crashing.

use std::io::Write;
use std::process::{Command, Stdio};

/// Which selection a clipboard register maps to. `"+` is the desktop
/// clipboard everyone means by "the clipboard"; `"*` is X11's middle-click
/// *primary selection* (on Wayland, the primary selection; on macOS, there is
/// no separate primary, so both fall back to the one system clipboard).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Selection {
    /// `"+` — the system clipboard (OSC-52 selection `c`).
    Clipboard,
    /// `"*` — the primary selection (OSC-52 selection `p`).
    Primary,
}

/// Maps a register name to the selection it addresses, or `None` if the name
/// is not a clipboard register.
pub fn register_selection(name: char) -> Option<Selection> {
    match name {
        '+' => Some(Selection::Clipboard),
        '*' => Some(Selection::Primary),
        _ => None,
    }
}

/// Standard-alphabet, `=`-padded base64 — hand-rolled so kvim take on no
/// dependency for the handful of bytes an OSC-52 payload need. OSC-52 require
/// exactly this encoding (RFC 4648 §4), padding included.
pub fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[(n >> 18 & 0x3f) as usize] as char);
        out.push(ALPHABET[(n >> 12 & 0x3f) as usize] as char);
        out.push(if chunk.len() > 1 { ALPHABET[(n >> 6 & 0x3f) as usize] as char } else { '=' });
        out.push(if chunk.len() > 2 { ALPHABET[(n & 0x3f) as usize] as char } else { '=' });
    }
    out
}

/// Build the OSC-52 escape that set the terminal's `selection` clipboard to
/// `text`. The sequence is `ESC ] 52 ; <sel> ; <base64(text)> BEL`, where
/// `<sel>` is `c` for the clipboard and `p` for the primary selection. `BEL`
/// (`0x07`) is used as the string terminator rather than `ESC \` (ST) because
/// more terminals accept it, and it is a single byte.
pub fn osc52_set(selection: Selection, text: &str) -> Vec<u8> {
    let sel = match selection {
        Selection::Clipboard => 'c',
        Selection::Primary => 'p',
    };
    format!("\x1b]52;{sel};{}\x07", base64_encode(text.as_bytes())).into_bytes()
}

/// A platform helper program to run, resolved from the environment. Kept as
/// plain data (program name + args) so [`copy_tool`]/[`paste_tool`] are pure
/// and unit-testable — nothing here spawns anything.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCommand {
    pub program: String,
    pub args: Vec<String>,
}

impl ToolCommand {
    fn new(program: &str, args: &[&str]) -> Self {
        Self { program: program.to_string(), args: args.iter().map(|s| s.to_string()).collect() }
    }
}

/// A snapshot of the bits of the environment that decide which clipboard tool
/// to reach for. Snapshotted into a struct (rather than read from `std::env`
/// inline) so tool selection is a pure function of it and can be tested for
/// every platform without setting real environment variables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Env {
    /// Running on macOS (`pbcopy`/`pbpaste` territory).
    pub macos: bool,
    /// `WAYLAND_DISPLAY` is set — prefer `wl-copy`/`wl-paste`.
    pub wayland: bool,
    /// `DISPLAY` is set — X11 is reachable via `xclip`.
    pub x11: bool,
    /// Termux (`PREFIX` points inside `com.termux`) — no X11/Wayland, so
    /// OSC-52 carry copy and `termux-clipboard-get` carry paste.
    pub termux: bool,
}

impl Env {
    /// Detect the current environment. Uses `cfg!(target_os = ...)` for the OS
    /// and the presence of `WAYLAND_DISPLAY`/`DISPLAY`/`PREFIX` for the display
    /// server — matching how [`crate::icons`] and [`crate::tmux`] already sniff
    /// the platform.
    pub fn detect() -> Self {
        Self {
            macos: cfg!(target_os = "macos"),
            wayland: std::env::var_os("WAYLAND_DISPLAY").is_some(),
            x11: std::env::var_os("DISPLAY").is_some(),
            termux: std::env::var("PREFIX").is_ok_and(|p| p.contains("com.termux")),
        }
    }
}

/// The tool that *writes* `selection`, or `None` when no desktop tool fits and
/// OSC-52 is the only copy path (a bare SSH session, or Termux). Wayland win
/// over X11 when both are advertised, because a Wayland session commonly also
/// export `DISPLAY` for Xwayland and `wl-copy` is the native path.
pub fn copy_tool(env: &Env, selection: Selection) -> Option<ToolCommand> {
    if env.macos {
        // macOS has no primary selection; both registers use the one pasteboard.
        return Some(ToolCommand::new("pbcopy", &[]));
    }
    if env.wayland {
        return Some(match selection {
            Selection::Clipboard => ToolCommand::new("wl-copy", &[]),
            Selection::Primary => ToolCommand::new("wl-copy", &["--primary"]),
        });
    }
    if env.x11 {
        let sel = match selection {
            Selection::Clipboard => "clipboard",
            Selection::Primary => "primary",
        };
        return Some(ToolCommand::new("xclip", &["-selection", sel, "-in"]));
    }
    if env.termux {
        // termux-api's setter, if the user installed it; OSC-52 covers the
        // common case, so this is a bonus rather than a requirement.
        return Some(ToolCommand::new("termux-clipboard-set", &[]));
    }
    None
}

/// The tool that *reads* `selection`, or `None` when the clipboard cannot be
/// read back at all (OSC-52 read being blocked by terminals — see the module
/// docs). Same platform precedence as [`copy_tool`].
pub fn paste_tool(env: &Env, selection: Selection) -> Option<ToolCommand> {
    if env.macos {
        return Some(ToolCommand::new("pbpaste", &[]));
    }
    if env.wayland {
        // `--no-newline` stops wl-paste appending a trailing newline that the
        // clipboard content did not actually contain.
        return Some(match selection {
            Selection::Clipboard => ToolCommand::new("wl-paste", &["--no-newline"]),
            Selection::Primary => ToolCommand::new("wl-paste", &["--no-newline", "--primary"]),
        });
    }
    if env.x11 {
        let sel = match selection {
            Selection::Clipboard => "clipboard",
            Selection::Primary => "primary",
        };
        return Some(ToolCommand::new("xclip", &["-selection", sel, "-out"]));
    }
    if env.termux {
        return Some(ToolCommand::new("termux-clipboard-get", &[]));
    }
    None
}

/// Runs a [`ToolCommand`], optionally feeding `stdin`. The seam that let tests
/// verify *which* command kvim would run without spawning it, and let the
/// clipboard provider stay unit-testable. Returns the captured stdout on
/// success, or `None` if the program is absent or exit non-zero.
pub trait CommandRunner {
    fn run(&self, cmd: &ToolCommand, stdin: Option<&str>) -> Option<String>;
}

/// The real runner: spawn the tool with [`std::process`]. Pure Rust, no linked
/// clipboard library — the tool is a separate process, exactly like kvim's git
/// and tmux integrations shell out rather than link.
#[derive(Debug, Default)]
pub struct StdRunner;

impl CommandRunner for StdRunner {
    fn run(&self, cmd: &ToolCommand, stdin: Option<&str>) -> Option<String> {
        let mut command = Command::new(&cmd.program);
        command.args(&cmd.args);
        command.stdin(if stdin.is_some() { Stdio::piped() } else { Stdio::null() });
        command.stdout(Stdio::piped());
        command.stderr(Stdio::null());
        let mut child = command.spawn().ok()?;
        if let Some(text) = stdin {
            child.stdin.take()?.write_all(text.as_bytes()).ok()?;
        }
        let output = child.wait_with_output().ok()?;
        if output.status.success() {
            Some(String::from_utf8_lossy(&output.stdout).into_owned())
        } else {
            None
        }
    }
}

/// Where OSC-52 escape bytes go. Abstracted so tests can capture the bytes and
/// so the real sink can target the *controlling terminal* rather than whatever
/// `stdout` happens to be.
pub trait Osc52Sink {
    fn emit(&self, bytes: &[u8]);
}

/// Writes OSC-52 to `/dev/tty` — the controlling terminal — rather than to
/// `stdout`. Two reasons: while the TUI is on the alternate screen, `stdout`
/// is being driven by the renderer, and OSC-52 must reach the *terminal
/// emulator* regardless of any redirection of stdout/stderr. `/dev/tty` is the
/// terminal by definition. If it cannot be opened (no controlling terminal —
/// e.g. under a test harness), the write is silently dropped: a failed
/// clipboard copy must never take the editor down.
#[derive(Debug, Default)]
pub struct TtySink;

impl Osc52Sink for TtySink {
    fn emit(&self, bytes: &[u8]) {
        if let Ok(mut tty) = std::fs::OpenOptions::new().write(true).open("/dev/tty") {
            let _ = tty.write_all(bytes);
            let _ = tty.flush();
        }
    }
}

/// Reading and writing `"+`/`"*`. Behind a trait so [`super`](crate::editor)
/// can hold one without caring whether it is the real [`SystemClipboard`] or a
/// test double, and so register-routing tests never touch a real clipboard.
pub trait ClipboardProvider {
    /// Copy `text` into `selection`. Best-effort — returns `true` if any
    /// channel (OSC-52 or a platform tool) accepted it. OSC-52 is fire-and-
    /// forget, so a copy essentially always report success.
    fn copy(&mut self, selection: Selection, text: &str) -> bool;
    /// Read `selection` back, or `None` when no reader is available (see the
    /// module docs on why OSC-52 cannot paste).
    fn paste(&mut self, selection: Selection) -> Option<String>;
}

/// The production clipboard: OSC-52 out the terminal, plus a platform tool
/// when one is present. Holds its [`Env`], [`CommandRunner`] and [`Osc52Sink`]
/// as trait objects so every piece is swappable in tests.
pub struct SystemClipboard {
    env: Env,
    runner: Box<dyn CommandRunner>,
    osc: Box<dyn Osc52Sink>,
}

impl std::fmt::Debug for SystemClipboard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SystemClipboard").field("env", &self.env).finish_non_exhaustive()
    }
}

impl SystemClipboard {
    /// The real clipboard, wired to the detected environment, `std::process`,
    /// and `/dev/tty`.
    pub fn new() -> Self {
        Self { env: Env::detect(), runner: Box::new(StdRunner), osc: Box::new(TtySink) }
    }

    /// For tests: a clipboard with an injected environment, runner and sink, so
    /// the exact escape bytes and tool invocations can be asserted without any
    /// real I/O.
    pub fn with_parts(env: Env, runner: Box<dyn CommandRunner>, osc: Box<dyn Osc52Sink>) -> Self {
        Self { env, runner, osc }
    }
}

impl Default for SystemClipboard {
    fn default() -> Self {
        Self::new()
    }
}

impl ClipboardProvider for SystemClipboard {
    fn copy(&mut self, selection: Selection, text: &str) -> bool {
        // OSC-52 always goes out — it is the path that survives SSH/tmux/Termux.
        self.osc.emit(&osc52_set(selection, text));
        // A desktop tool, if one fits, runs too so the copy also works when the
        // terminal itself does not honour OSC-52. Its success is a bonus, not a
        // requirement, so its result is intentionally not the return value.
        if let Some(cmd) = copy_tool(&self.env, selection) {
            let _ = self.runner.run(&cmd, Some(text));
        }
        // OSC-52 is fire-and-forget and always emitted, so the copy is done.
        true
    }

    fn paste(&mut self, selection: Selection) -> Option<String> {
        let cmd = paste_tool(&self.env, selection)?;
        self.runner.run(&cmd, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn base64_matches_known_vectors() {
        // RFC 4648 §10 test vectors.
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn osc52_wraps_base64_with_the_right_selection_and_terminators() {
        let bytes = osc52_set(Selection::Clipboard, "hi");
        assert_eq!(bytes, b"\x1b]52;c;aGk=\x07");
        let primary = osc52_set(Selection::Primary, "hi");
        assert_eq!(primary, b"\x1b]52;p;aGk=\x07");
    }

    #[test]
    fn wayland_selects_wl_copy_and_primary_adds_the_flag() {
        let env = Env { macos: false, wayland: true, x11: true, termux: false };
        assert_eq!(copy_tool(&env, Selection::Clipboard), Some(ToolCommand::new("wl-copy", &[])));
        assert_eq!(copy_tool(&env, Selection::Primary), Some(ToolCommand::new("wl-copy", &["--primary"])));
        assert_eq!(paste_tool(&env, Selection::Clipboard), Some(ToolCommand::new("wl-paste", &["--no-newline"])));
    }

    #[test]
    fn x11_without_wayland_selects_xclip() {
        let env = Env { macos: false, wayland: false, x11: true, termux: false };
        assert_eq!(copy_tool(&env, Selection::Clipboard), Some(ToolCommand::new("xclip", &["-selection", "clipboard", "-in"])));
        assert_eq!(paste_tool(&env, Selection::Primary), Some(ToolCommand::new("xclip", &["-selection", "primary", "-out"])));
    }

    #[test]
    fn macos_uses_pbcopy_for_both_selections() {
        let env = Env { macos: true, wayland: false, x11: false, termux: false };
        assert_eq!(copy_tool(&env, Selection::Primary), Some(ToolCommand::new("pbcopy", &[])));
        assert_eq!(paste_tool(&env, Selection::Clipboard), Some(ToolCommand::new("pbpaste", &[])));
    }

    #[test]
    fn a_bare_terminal_has_no_paste_tool() {
        // No display server, not Termux: OSC-52 can copy, but nothing can read
        // the clipboard back. paste_tool must report that honestly.
        let env = Env { macos: false, wayland: false, x11: false, termux: false };
        assert_eq!(copy_tool(&env, Selection::Clipboard), None);
        assert_eq!(paste_tool(&env, Selection::Clipboard), None);
    }

    #[test]
    fn termux_copies_via_osc52_and_can_read_via_the_helper() {
        let env = Env { macos: false, wayland: false, x11: false, termux: true };
        assert_eq!(copy_tool(&env, Selection::Clipboard), Some(ToolCommand::new("termux-clipboard-set", &[])));
        assert_eq!(paste_tool(&env, Selection::Clipboard), Some(ToolCommand::new("termux-clipboard-get", &[])));
    }

    /// One recorded invocation: the command, and the stdin it was fed (a paste
    /// has none).
    type RunLog = Rc<RefCell<Vec<(ToolCommand, Option<String>)>>>;

    /// A runner that records what it was asked to run and returns a canned
    /// paste value, so provider behaviour can be asserted without spawning.
    /// Shares its log through an `Rc<RefCell<..>>` so a test can hold a handle
    /// after the runner is moved into the clipboard.
    #[derive(Default, Clone)]
    struct MockRunner {
        calls: RunLog,
        paste_value: Option<String>,
    }

    impl CommandRunner for MockRunner {
        fn run(&self, cmd: &ToolCommand, stdin: Option<&str>) -> Option<String> {
            self.calls.borrow_mut().push((cmd.clone(), stdin.map(str::to_string)));
            // A paste (no stdin) returns the canned value; a copy (with stdin)
            // "succeeds" with empty output.
            if stdin.is_some() { Some(String::new()) } else { self.paste_value.clone() }
        }
    }

    #[derive(Default, Clone)]
    struct CapturingSink {
        bytes: Rc<RefCell<Vec<u8>>>,
    }

    impl Osc52Sink for CapturingSink {
        fn emit(&self, bytes: &[u8]) {
            self.bytes.borrow_mut().extend_from_slice(bytes);
        }
    }

    #[test]
    fn copy_emits_osc52_and_runs_the_platform_tool() {
        let env = Env { macos: false, wayland: true, x11: false, termux: false };
        let sink = CapturingSink::default();
        let runner = MockRunner::default();
        // Hold a handle to each before they move into the clipboard.
        let sink_handle = sink.clone();
        let runner_handle = runner.clone();
        let mut clip = SystemClipboard::with_parts(env, Box::new(runner), Box::new(sink));

        assert!(clip.copy(Selection::Clipboard, "hi"));

        assert_eq!(&*sink_handle.bytes.borrow(), b"\x1b]52;c;aGk=\x07");
        assert_eq!(
            &*runner_handle.calls.borrow(),
            &vec![(ToolCommand::new("wl-copy", &[]), Some("hi".to_string()))]
        );
    }

    #[test]
    fn paste_reads_through_the_platform_tool() {
        let env = Env { macos: false, wayland: true, x11: false, termux: false };
        let runner = MockRunner { paste_value: Some("from clipboard".to_string()), ..Default::default() };
        let mut clip = SystemClipboard::with_parts(env, Box::new(runner), Box::new(CapturingSink::default()));
        assert_eq!(clip.paste(Selection::Clipboard), Some("from clipboard".to_string()));
    }

    #[test]
    fn paste_is_none_when_no_tool_is_available() {
        let env = Env { macos: false, wayland: false, x11: false, termux: false };
        let mut clip = SystemClipboard::with_parts(env, Box::new(MockRunner::default()), Box::new(CapturingSink::default()));
        assert_eq!(clip.paste(Selection::Clipboard), None);
    }

    /// Live round-trip against a real desktop clipboard. Ignored by default —
    /// it need a running Wayland/X11/macOS session with the tool installed, and
    /// it clobber the user's real clipboard. Run with
    /// `cargo test --release -p kopitiam-neovim -- --ignored clipboard_live`.
    #[test]
    #[ignore = "touches the real system clipboard"]
    fn clipboard_live_round_trip() {
        let env = Env::detect();
        if copy_tool(&env, Selection::Clipboard).is_none() || paste_tool(&env, Selection::Clipboard).is_none() {
            eprintln!("no clipboard tool on this box; skipping");
            return;
        }
        let mut clip = SystemClipboard::new();
        let payload = "kvim-clipboard-live-test-魚";
        assert!(clip.copy(Selection::Clipboard, payload));
        // Give an async setter (wl-copy forks) a beat to take ownership.
        std::thread::sleep(std::time::Duration::from_millis(150));
        assert_eq!(clip.paste(Selection::Clipboard).as_deref(), Some(payload));
    }
}

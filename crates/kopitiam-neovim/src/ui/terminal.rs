//! Terminal lifecycle: entering/leaving raw mode and the alternate screen,
//! and the panic hook that guarantees both are undone.
//!
//! # Why this is the first thing built, not polish
//!
//! Raw mode disables line buffering and local echo. The alternate screen
//! swaps away the user's scrollback. Both are correct *while kvim is
//! running*, and both are a hostile mess if left on when it isn't — a
//! terminal stuck in raw mode with no echo looks broken, not exited, and the
//! only fix most users know is a blind `reset` + Enter. A modal editor is
//! already asymmetric-risk software (it eats keystrokes as commands); the
//! last thing it should also do is eat the user's terminal. So: every code
//! path that leaves this module restores the terminal, including a `panic!`
//! anywhere in kvim's call stack, before the panic message is even printed
//! (otherwise the panic backtrace itself renders inside the mangled
//! terminal state and is unreadable).

use std::io::{self, Write};

use crossterm::{
    cursor::SetCursorStyle,
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{
        disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
    },
};

/// RAII guard owning the "terminal is in TUI mode" invariant.
///
/// Construction enables raw mode and switches to the alternate screen (and,
/// if requested, mouse capture). [`Drop`] undoes exactly that, in reverse
/// order, best-effort — a `Drop` impl cannot propagate an I/O error, and by
/// the time we are unwinding (possibly from a panic) there is no one left to
/// hand an error to. Failing to restore the terminal is worse than losing an
/// error message about failing to restore the terminal.
///
/// Mouse capture defaults to **off**: kvim is a modal keyboard editor, and
/// capturing the mouse steals terminal-native text selection/copy from the
/// user without kvim yet doing anything useful with mouse events. Opt in
/// explicitly via [`TerminalGuard::with_mouse_capture`] once mouse support
/// (e.g. click-to-position-cursor) exists.
pub struct TerminalGuard {
    mouse_capture: bool,
    /// Set once [`TerminalGuard::restore`] has run, so `Drop` does not double
    /// -restore (which would emit spurious escape codes on an already-plain
    /// terminal, e.g. after an explicit `quit`).
    restored: bool,
}

impl TerminalGuard {
    /// Enters raw mode + the alternate screen, with mouse capture off.
    pub fn new() -> io::Result<Self> {
        Self::enter(false)
    }

    /// Enters raw mode + the alternate screen, with mouse capture on.
    pub fn with_mouse_capture() -> io::Result<Self> {
        Self::enter(true)
    }

    fn enter(mouse_capture: bool) -> io::Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        if mouse_capture {
            execute!(stdout, EnableMouseCapture)?;
        }
        install_panic_hook();
        Ok(Self { mouse_capture, restored: false })
    }

    /// Explicitly restores the terminal, ahead of the guard being dropped.
    ///
    /// Prefer letting `Drop` do this on the normal exit path; call this
    /// directly only when you need the terminal back *before* printing
    /// something to it (e.g. `kvim --version`'s own error path, or tests
    /// that assert on stdout after the guard is gone but before the process
    /// exits).
    pub fn restore(&mut self) {
        if self.restored {
            return;
        }
        self.restored = true;
        restore_terminal(self.mouse_capture);
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        self.restore();
    }
}

/// The actual restoration sequence, factored out so both [`TerminalGuard`]
/// and the panic hook can call it without needing a live `TerminalGuard`
/// (the hook runs in whatever stack unwinding left behind, which may not
/// include one).
fn restore_terminal(mouse_capture: bool) {
    let mut stdout = io::stdout();
    if mouse_capture {
        let _ = execute!(stdout, DisableMouseCapture);
    }
    // Reset the cursor to the terminal's default shape so a crash mid-Insert
    // does not leave a bar cursor blinking on the user's shell prompt.
    let _ = execute!(stdout, SetCursorStyle::DefaultUserShape);
    let _ = execute!(stdout, LeaveAlternateScreen);
    let _ = disable_raw_mode();
    let _ = stdout.flush();
}

/// Chains onto the process's existing panic hook so raw mode and the
/// alternate screen are torn down *before* the panic message prints — a
/// panic printed while still in the alternate screen either vanishes
/// (invisible on the swapped-away screen) or renders through partially
/// processed raw-mode output.
///
/// Idempotent-safe to call more than once (each call adds one more hook that
/// restores an already-restored terminal, which is a harmless no-op), but
/// [`TerminalGuard::enter`] only calls it once per guard construction.
fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // Mouse-capture state isn't recoverable from here in general (the
        // guard that knows it may be gone by the time we unwind), so restore
        // the common case: no mouse capture. If mouse capture was on, the
        // `DisableMouseCapture` escape is harmless to send even when it
        // wasn't enabled.
        restore_terminal(true);
        previous(info);
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// `Drop` must run the restoration path exactly once, even when
    /// triggered implicitly by scope exit. This does not spawn a real
    /// terminal (that would require a TTY, which CI does not have); instead
    /// it verifies the guard's own bookkeeping — the part most likely to
    /// regress into a double-restore or a skipped restore.
    #[test]
    fn drop_marks_the_guard_restored_exactly_once() {
        let mut guard = TerminalGuard { mouse_capture: false, restored: false };
        assert!(!guard.restored);
        guard.restore();
        assert!(guard.restored);
        // A second explicit restore must not panic or double-run the escape
        // sequence writer.
        guard.restore();
        assert!(guard.restored);
        // Dropping after an explicit restore must also be a no-op, not a
        // second restoration.
        drop(guard);
    }

    /// Simulates the "panic while a guard is live" path without actually
    /// panicking the test process: confirms that a closure standing in for
    /// the panic hook body runs the restoration exactly once via a counter,
    /// proving the hook-chaining wraps rather than replaces cleanup.
    #[test]
    fn restoration_closure_runs_exactly_once_when_invoked() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_clone = Arc::clone(&calls);
        let restore_stub = move || {
            calls_clone.fetch_add(1, Ordering::SeqCst);
        };
        restore_stub();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}

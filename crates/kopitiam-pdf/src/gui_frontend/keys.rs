//! The vim-motion key-capture predicate and scroll-step arithmetic for a
//! continuous-scroll viewer -- kept pure (plain `bool`s/`f32`s/[`Duration`],
//! no `egui` types) so the one decision that must never regress -- "is a
//! keypress currently captured by text entry, or free to be a vim motion" --
//! is covered by a unit test rather than only ever exercised at a keyboard.

use std::time::Duration;

/// Whether keyboard input is currently **captured** by something other than
/// vim-motion/page-navigation handling -- typing a page number into `goto`,
/// or typing into an open in-place form-field editor. A caller
/// (`KpdfApp::handle_key`) must check this *before* dispatching to any
/// vim-motion arm (`h`/`j`/`k`/`l`, `Ctrl+d`/`Ctrl+u`, `gg`/`G`) -- a `j`
/// typed into a form field must insert the letter `j`, not scroll the page.
///
/// Both captured states already routed around the old single-page nav keys
/// in `kpdf.rs` before this pass (see `handle_key`'s pre-existing `goto`/
/// `form_edit` guards) -- this names that condition once, so every new
/// vim-motion arm reuses the exact same gate rather than a second,
/// independently-maintained copy of it.
pub fn keys_captured(goto_active: bool, form_edit_active: bool) -> bool {
    goto_active || form_edit_active
}

/// One `h`/`j`/`k`/`l` nudge's scroll distance, screen points -- a small,
/// fixed step rather than one tied to the viewport height (unlike
/// [`half_viewport_step`]'s `Ctrl+d`/`Ctrl+u`, which vim itself defines as
/// half a screenful), because vim's own `h`/`j`/`k`/`l` move by *lines*, not
/// a window fraction, and a fixed step is the closest stand-in for "one
/// line" a page-image viewer has. Not a value derived from anything
/// measured -- "does this feel like one line's worth of scroll" is a
/// human-at-a-display judgment call, flagged here rather than asserted as
/// correct; see this crate's other unmeasured-UX-constant precedents
/// (`MIN_HIT_AREA_SCREEN` in `kpdf.rs`, `ZOOM_DELTA_PER_STEP` in `zoom.rs`).
pub const VIM_STEP: f32 = 48.0;

/// `Ctrl+d`/`Ctrl+u`'s scroll distance: half the viewport -- vim's own
/// documented behaviour for those bindings (`:help CTRL-D` / `:help
/// CTRL-U`), not a guess or a separately-configurable constant.
pub fn half_viewport_step(viewport_h: f32) -> f32 {
    viewport_h / 2.0
}

/// Timeout after which a lone pending `g` (waiting for a second `g` to
/// complete the `gg` "go to first page" sequence) is dropped rather than
/// left armed indefinitely -- see [`GPending`]. 600ms is a generous but
/// bounded window for a deliberate double-tap; like [`VIM_STEP`], this is a
/// human-judgment default, not a measured value -- a real dogfooder is the
/// only one who can say whether it feels right.
pub const G_PENDING_TIMEOUT: Duration = Duration::from_millis(600);

/// The `gg` two-key-sequence state machine: vim's `gg` (go to first page)
/// needs to remember "a `g` was just pressed" across a frame boundary,
/// without leaking that state forever if a second `g` never arrives -- a
/// stray single `g` pressed once must not silently arm "the next g goes to
/// page 1" five minutes later. This struct only tracks *whether* a `g` is
/// armed; the caller (`KpdfApp`, which has a real wall clock) is the one
/// that times out a stale arm, via [`g_pending_expired`], and calls
/// [`GPending::cancel`] when it does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GPending {
    armed: bool,
}

impl GPending {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_armed(&self) -> bool {
        self.armed
    }

    /// Record a `g` keypress. Returns `true` if this completes a `gg`
    /// sequence (a `g` was already armed) -- the caller is responsible for
    /// having already called [`GPending::cancel`] on a stale arm (see
    /// [`g_pending_expired`]) before this runs, so a `gg` can never fire
    /// across an arbitrarily long gap. Otherwise arms `self` and returns
    /// `false`.
    pub fn press_g(&mut self) -> bool {
        if self.armed {
            self.armed = false;
            true
        } else {
            self.armed = true;
            false
        }
    }

    /// Drop the armed state unconditionally -- any other keypress, or an
    /// explicit timeout, cancels a pending `g` rather than letting it linger
    /// for an unrelated later `g`.
    pub fn cancel(&mut self) {
        self.armed = false;
    }
}

/// Whether a pending `g` (armed `elapsed` ago) has aged past
/// [`G_PENDING_TIMEOUT`] and should be treated as expired -- pure over a
/// plain [`Duration`] so it is testable without a real clock; `KpdfApp` is
/// the one that actually tracks wall-clock time (`std::time::Instant`) and
/// calls this each frame before acting on a `g` keypress.
pub fn g_pending_expired(elapsed: Duration) -> bool {
    elapsed >= G_PENDING_TIMEOUT
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- keys_captured ------------------------------------------------------

    #[test]
    fn nothing_captured_when_both_are_inactive() {
        assert!(!keys_captured(false, false));
    }

    #[test]
    fn goto_entry_captures_keys() {
        assert!(keys_captured(true, false));
    }

    #[test]
    fn form_edit_captures_keys() {
        assert!(keys_captured(false, true));
    }

    #[test]
    fn both_active_still_captures() {
        assert!(keys_captured(true, true));
    }

    // -- half_viewport_step --------------------------------------------------

    #[test]
    fn half_viewport_step_is_exactly_half() {
        assert_eq!(half_viewport_step(800.0), 400.0);
    }

    #[test]
    fn half_viewport_step_zero_viewport_is_zero() {
        assert_eq!(half_viewport_step(0.0), 0.0);
    }

    // -- GPending / g_pending_expired -----------------------------------------

    #[test]
    fn fresh_gpending_is_not_armed() {
        assert!(!GPending::new().is_armed());
    }

    #[test]
    fn first_g_arms_but_does_not_fire() {
        let mut p = GPending::new();
        assert!(!p.press_g());
        assert!(p.is_armed());
    }

    #[test]
    fn second_g_completes_the_sequence_and_disarms() {
        let mut p = GPending::new();
        p.press_g();
        assert!(p.press_g());
        assert!(!p.is_armed());
    }

    #[test]
    fn a_third_g_right_after_gg_starts_a_fresh_sequence() {
        let mut p = GPending::new();
        p.press_g(); // arm
        p.press_g(); // fires gg, disarms
        assert!(!p.press_g()); // arms again, does not immediately fire
        assert!(p.is_armed());
    }

    #[test]
    fn cancel_disarms_a_pending_g() {
        let mut p = GPending::new();
        p.press_g();
        p.cancel();
        assert!(!p.is_armed());
        // A `g` after a cancel starts a fresh sequence, not an instant fire.
        assert!(!p.press_g());
    }

    #[test]
    fn cancel_on_an_unarmed_pending_is_a_harmless_no_op() {
        let mut p = GPending::new();
        p.cancel();
        assert!(!p.is_armed());
    }

    #[test]
    fn g_pending_expired_boundary() {
        assert!(!g_pending_expired(Duration::from_millis(599)));
        assert!(g_pending_expired(Duration::from_millis(600)));
        assert!(g_pending_expired(Duration::from_millis(601)));
    }

    #[test]
    fn g_pending_expired_zero_elapsed_is_not_expired() {
        assert!(!g_pending_expired(Duration::ZERO));
    }
}

// ---------------------------------------------------------------------------
// The `:` command line
// ---------------------------------------------------------------------------

/// What the user typed after `:` resolved to a command.
///
/// The `:` entry started life as a digits-only "go to page" box. It is a
/// command line now, because vim users reach for `:w` before they reach for a
/// toolbar button, and having `:` accept `12` but reject `w` is the kind of
/// half-a-convention that is worse than either whole one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// `:N` — go to 1-based page `N`. Vim's own line-jump idiom.
    GotoPage(usize),
    /// `:w` — write the file in place, same as `Ctrl+S`.
    Write,
    /// Typed something we do not implement. The caller should say so rather
    /// than silently doing nothing: a command line that ignores input leaves
    /// the user unable to tell "not a command" from "command failed".
    Unknown(String),
    /// Nothing was typed (bare `:` then Enter) — do nothing, quietly.
    Empty,
}

/// Parse a `:` command line entry.
///
/// Whitespace around the command is ignored, and commands are matched
/// case-sensitively: vim's `:w` is lowercase, and `:W` is a different command
/// in vim itself, so quietly accepting it would teach the wrong habit.
///
/// Note `GotoPage` rejects `0`: PDF pages are 1-based to a reader, and `:0`
/// almost certainly means the user expected 0-based indexing, which is worth
/// reporting rather than silently clamping to page 1.
pub fn parse_command(input: &str) -> Command {
    let text = input.trim();
    if text.is_empty() {
        return Command::Empty;
    }
    if text == "w" {
        return Command::Write;
    }
    match text.parse::<usize>() {
        Ok(n) if n >= 1 => Command::GotoPage(n),
        _ => Command::Unknown(text.to_string()),
    }
}

#[cfg(test)]
mod command_tests {
    use super::*;

    #[test]
    fn parses_a_page_number() {
        assert_eq!(parse_command("12"), Command::GotoPage(12));
        assert_eq!(parse_command("  7 "), Command::GotoPage(7));
    }

    #[test]
    fn parses_write() {
        assert_eq!(parse_command("w"), Command::Write);
        assert_eq!(parse_command(" w "), Command::Write);
    }

    /// Page numbers are 1-based for a reader; `:0` is a mistake worth
    /// reporting rather than silently clamping.
    #[test]
    fn rejects_page_zero_rather_than_clamping() {
        assert_eq!(parse_command("0"), Command::Unknown("0".to_string()));
    }

    #[test]
    fn bare_colon_does_nothing() {
        assert_eq!(parse_command(""), Command::Empty);
        assert_eq!(parse_command("   "), Command::Empty);
    }

    /// An unimplemented command must be reported, not swallowed -- otherwise
    /// the user cannot tell "not a command" from "command silently failed".
    #[test]
    fn unknown_commands_are_reported() {
        assert_eq!(parse_command("q"), Command::Unknown("q".to_string()));
        assert_eq!(parse_command("wq"), Command::Unknown("wq".to_string()));
        assert_eq!(parse_command("W"), Command::Unknown("W".to_string()));
    }
}

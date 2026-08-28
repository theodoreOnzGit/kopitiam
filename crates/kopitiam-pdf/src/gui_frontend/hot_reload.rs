//! Reload the open document when it changes on disk — the live-preview loop
//! for editing a `.tex`/`.typ` in one window and watching the compiled PDF
//! refresh in another.
//!
//! Ported from `kovan`'s reader (`crates/kovan/src/digitiser/gui/desktop/
//! pdf_reader.rs`, `check_hot_reload`/`read_mtime`, GitHub issue #30 —
//! *"hot reload by default in case I compile live in tex or typst"*). Same
//! mechanism, same 500 ms throttle, same default-on stance; see
//! `docs/ACKNOWLEDGEMENTS.md`.
//!
//! # Why polling mtime, and not a filesystem watcher
//!
//! Deliberate, and inherited from kovan. A watcher (`inotify`/`notify`) means
//! a dependency, a background thread, and platform-specific behaviour on the
//! three targets this workspace ships to — for a check that costs one `stat`
//! twice a second. Polling is a couple of dozen lines, needs no dependency at
//! all (keeping the Pure Rust Core promise trivially), and behaves identically
//! on Linux, Windows and Termux. The cost is up to [`RELOAD_CHECK_INTERVAL`]
//! of latency before a recompile shows, which is imperceptible next to the
//! compile itself.
//!
//! # Why this is a struct here rather than a method on the app
//!
//! kovan's version reads the clock inline (`Instant::now()` inside the check),
//! which makes the throttle untestable without sleeping. Here `now` is passed
//! in, so the throttle, the change detection and the "our own write is not an
//! external change" rule are all unit-testable with no filesystem timing and
//! no display — the same separation `lru.rs` and `geometry.rs` already use.
//!
//! # The trap this exists to avoid: reloading over your own save
//!
//! kpdf can *write* the file it is watching (`Ctrl+S` saves annotations in
//! place). That write changes the mtime, so a naive watcher sees its own save
//! as an external change and reloads — throwing away the in-memory edit
//! history for no reason, on every save. Hence [`HotReload::mark_current`]:
//! the caller must claim the new mtime after **every** write it performs, not
//! only after an open. kovan gets this for free because its own save path
//! re-opens the document; kpdf's does not.

use std::path::Path;
use std::time::{Duration, Instant, SystemTime};

/// How often the watched file may be `stat`ed. Frequent enough that a live
/// TeX/Typst recompile is picked up promptly, infrequent enough not to hammer
/// the filesystem every frame (kovan's `RELOAD_CHECK_INTERVAL`, same value).
pub const RELOAD_CHECK_INTERVAL: Duration = Duration::from_millis(500);

/// What [`HotReload::poll`] concluded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReloadDecision {
    /// Nothing to do: disabled, still inside the throttle window, the file is
    /// unreadable/absent, or its mtime is what we last claimed.
    Idle,
    /// The file's mtime differs from the one we last claimed — the caller
    /// should re-open it and then [`HotReload::mark_current`].
    Changed,
}

/// Throttled mtime poller over a single path.
///
/// Holds no path of its own: the caller passes the path it currently has
/// open, so a document swap (kpdf's `o`) needs no re-plumbing here — just
/// [`mark_current`](HotReload::mark_current) against the new path.
#[derive(Debug, Clone, Default)]
pub struct HotReload {
    enabled: bool,
    /// The mtime this app last claimed as "ours" — set on open, and on every
    /// write we perform. A mtime differing from this is an external change.
    /// `None` means nothing claimed yet, in which case the first poll claims
    /// silently rather than reporting a spurious change.
    stamp: Option<SystemTime>,
    last_check: Option<Instant>,
}

impl HotReload {
    /// `enabled` starts **on** for kpdf, matching kovan's default and the
    /// maintainer's stated reason for wanting it (issue #30: compiling live
    /// in TeX or Typst).
    pub fn new(enabled: bool) -> HotReload {
        HotReload {
            enabled,
            stamp: None,
            last_check: None,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Turn watching on or off. Turning it **on** drops the remembered stamp,
    /// so the next [`poll`](HotReload::poll) re-claims whatever is on disk
    /// now instead of reporting every change that happened while it was off
    /// as one stale "changed" event.
    pub fn set_enabled(&mut self, enabled: bool) {
        if enabled && !self.enabled {
            self.stamp = None;
            self.last_check = None;
        }
        self.enabled = enabled;
    }

    /// Claim `path`'s current mtime as ours. Call after opening a document
    /// **and after every write to it** — see the module docs' note on
    /// reloading over your own save.
    pub fn mark_current(&mut self, path: &Path) {
        self.stamp = read_mtime(path);
    }

    /// Decide whether `path` changed under us, at most once per
    /// [`RELOAD_CHECK_INTERVAL`].
    ///
    /// `now` is injected rather than read here so the throttle is testable;
    /// pass `Instant::now()` from the UI frame.
    ///
    /// A file that cannot be `stat`ed (deleted, or mid-replacement by a
    /// compiler writing through a temp file) reports [`ReloadDecision::Idle`]
    /// and leaves the claimed stamp alone, so the next poll that *can* read it
    /// still sees the change. Never an error: a document vanishing is not
    /// something the reader should crash or nag about.
    pub fn poll(&mut self, path: &Path, now: Instant) -> ReloadDecision {
        if !self.enabled {
            return ReloadDecision::Idle;
        }
        if self
            .last_check
            .is_some_and(|last| now.duration_since(last) < RELOAD_CHECK_INTERVAL)
        {
            return ReloadDecision::Idle;
        }
        self.last_check = Some(now);
        let Some(mtime) = read_mtime(path) else {
            return ReloadDecision::Idle;
        };
        match self.stamp {
            // Nothing claimed yet: adopt what is there rather than calling the
            // very first poll a change.
            None => {
                self.stamp = Some(mtime);
                ReloadDecision::Idle
            }
            Some(known) if known == mtime => ReloadDecision::Idle,
            Some(_) => ReloadDecision::Changed,
        }
    }
}

/// The file's modification time, or `None` if it cannot be read at all.
/// (kovan: `PdfReaderState::read_mtime`.)
pub fn read_mtime(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).ok()?.modified().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn tmpfile(name: &str, body: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("kopitiam-hotreload-{name}-{}", std::process::id()));
        std::fs::write(&p, body).expect("write temp file");
        p
    }

    #[test]
    fn read_mtime_is_none_for_a_missing_file() {
        assert!(read_mtime(Path::new("/nonexistent/really/not/here.pdf")).is_none());
    }

    #[test]
    fn disabled_never_reports_a_change() {
        let p = tmpfile("disabled", b"a");
        let mut hr = HotReload::new(false);
        let t = Instant::now();
        assert_eq!(hr.poll(&p, t), ReloadDecision::Idle);
        let _ = std::fs::remove_file(&p);
    }

    /// The first poll adopts whatever is on disk instead of announcing a
    /// change — otherwise opening a document would immediately reload it.
    #[test]
    fn first_poll_claims_rather_than_reports() {
        let p = tmpfile("first", b"a");
        let mut hr = HotReload::new(true);
        assert_eq!(hr.poll(&p, Instant::now()), ReloadDecision::Idle);
        let _ = std::fs::remove_file(&p);
    }

    /// The throttle: a second poll inside the interval is skipped even when
    /// the stamp plainly differs. `now` being injected is what lets this run
    /// without sleeping.
    #[test]
    fn polls_inside_the_interval_are_throttled() {
        let p = tmpfile("throttle", b"a");
        let mut hr = HotReload::new(true);
        let t0 = Instant::now();
        hr.poll(&p, t0);
        // Force a stamp that cannot match, so only the throttle can suppress it.
        hr.stamp = Some(SystemTime::UNIX_EPOCH);
        assert_eq!(
            hr.poll(&p, t0 + RELOAD_CHECK_INTERVAL / 2),
            ReloadDecision::Idle,
            "a poll inside the throttle window must not stat"
        );
        assert_eq!(
            hr.poll(&p, t0 + RELOAD_CHECK_INTERVAL + Duration::from_millis(1)),
            ReloadDecision::Changed,
            "past the window, the differing mtime must surface"
        );
        let _ = std::fs::remove_file(&p);
    }

    /// The save-loop guard: after we write the file ourselves and re-claim it,
    /// the next poll must be quiet. Without `mark_current` this is the bug
    /// that reloads over every Ctrl+S.
    #[test]
    fn mark_current_stops_our_own_write_looking_external() {
        let p = tmpfile("ownwrite", b"a");
        let mut hr = HotReload::new(true);
        let t0 = Instant::now();
        hr.poll(&p, t0);
        hr.stamp = Some(SystemTime::UNIX_EPOCH); // pretend the file moved on
        hr.mark_current(&p); // ...but we claim it, as a save would
        assert_eq!(
            hr.poll(&p, t0 + RELOAD_CHECK_INTERVAL * 2),
            ReloadDecision::Idle
        );
        let _ = std::fs::remove_file(&p);
    }

    /// Re-enabling re-claims rather than firing once for everything missed
    /// while it was off.
    #[test]
    fn re_enabling_reclaims_instead_of_firing_stale() {
        let p = tmpfile("reenable", b"a");
        let mut hr = HotReload::new(true);
        let t0 = Instant::now();
        hr.poll(&p, t0);
        hr.set_enabled(false);
        hr.set_enabled(true);
        assert_eq!(
            hr.poll(&p, t0 + RELOAD_CHECK_INTERVAL * 2),
            ReloadDecision::Idle
        );
        let _ = std::fs::remove_file(&p);
    }

    /// An unreadable path is quiet, and does not disturb the claimed stamp.
    #[test]
    fn a_vanished_file_is_idle_not_an_error() {
        let mut hr = HotReload::new(true);
        hr.stamp = Some(SystemTime::UNIX_EPOCH);
        let d = hr.poll(Path::new("/nonexistent/gone.pdf"), Instant::now());
        assert_eq!(d, ReloadDecision::Idle);
        assert_eq!(hr.stamp, Some(SystemTime::UNIX_EPOCH));
    }
}

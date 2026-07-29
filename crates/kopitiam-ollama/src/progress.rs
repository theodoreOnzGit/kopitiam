//! Progress bars, spinners and step bars -- the *arithmetic* behind a `pull` line.
//!
//! **Upstream:** `progress/progress.go`, `progress/bar.go`, `progress/spinner.go`,
//! `progress/stepbar.go` (ollama, MIT).
//!
//! ## What got ported, and what got left behind on purpose
//!
//! Upstream's `progress` package is two things wearing one coat: **state +
//! arithmetic** (how many bytes done, how fast, how long more, what the line
//! say) and **terminal control** (ANSI escapes, cursor parking, a `bufio.Writer`,
//! a goroutine ticking every 100ms, `term.GetSize` on stderr).
//!
//! This module port the **first half only**. The second half stay out, and not
//! because it is hard -- because this crate must keep depending on nothing else
//! in KOPITIAM (AID-0055) and must not grow a TUI dependency. So:
//!
//! | Upstream | Here |
//! | --- | --- |
//! | `Bar.Set`, `percent`, `rate`, bucket ring | ported, byte-for-byte arithmetic |
//! | `formatDuration` | ported as [`format_duration`] |
//! | `Bar.String()`, `Spinner.String()`, `StepBar.String()` | ported, but terminal width is an **argument**, not a `term.GetSize` call |
//! | `Progress.Add`, the render window (`min(len(states), termHeight)`) | ported as [`Progress::visible`] / [`Progress::render_lines`] |
//! | `\033[?25l`, `\033[2K`, `\033[A`, `\033[?2026h`, `bufio.Writer` | **NOT ported.** Caller's job. |
//! | the 100ms goroutine ticker | **NOT ported.** Caller drive it -- see [`TICK_INTERVAL_MS`]. |
//!
//! Two ways to consume a bar, and both are supported on purpose:
//!
//! * [`Bar::line`] gives you the **exact upstream string** for a given width, so
//!   a plain stderr writer match ollama character for character;
//! * [`Bar::line_data`] gives you [`BarLine`] -- the same information as
//!   **structured data** (percent, rate, ETA, the human strings), so a TUI can
//!   draw its own widget without scraping a formatted string.
//!
//! ## Time is an argument, not a global
//!
//! Upstream call `time.Now()` inside `NewBar` and `Set`. We take the clock as a
//! parameter instead -- same move `format::human_time` already made in this
//! crate, same reason: **the arithmetic stay a pure function, so the tests need
//! no fake clock and no sleeping.** Rate estimation with a real clock in the
//! test suite is how you get a flaky test that only fail on a loaded CI box.
//!
//! Every `now_secs: f64` is **seconds off whatever monotonic origin the caller
//! pick** -- `Instant::elapsed().as_secs_f64()` is the obvious one. Only the
//! *differences* matter, the origin is arbitrary. What WOULD make this wrong:
//! feeding it a wall clock that can step backwards (NTP, DST), which would make
//! a denominator negative and the rate nonsense. Use a monotonic source.
//!
//! Go's zero `time.Time` (`stopped.IsZero()`) maps to `Option<f64>::None`
//! throughout -- that is the whole meaning of the zero value here, so an
//! `Option` say it better than a sentinel.

use crate::format::human_bytes;

/// Fallback terminal size when the caller cannot ask the real terminal.
///
/// **Upstream:** `progress/progress.go` consts `defaultTermWidth`/
/// `defaultTermHeight`. Upstream fall back to these when `term.GetSize(stderr)`
/// error -- e.g. output is piped to a file. This module never call `GetSize`
/// itself (no TUI dependency, see the module header), so these exist for the
/// caller to use as its own fallback and keep KOPITIAM's layout identical to
/// ollama's in the piped case.
pub const DEFAULT_TERM_WIDTH: usize = 80;

/// See [`DEFAULT_TERM_WIDTH`]. **Upstream:** `defaultTermHeight`.
pub const DEFAULT_TERM_HEIGHT: usize = 24;

/// How often upstream repaint, in milliseconds.
///
/// **Upstream:** `progress/progress.go` `start()` and `progress/spinner.go`
/// `start()` both do `time.NewTicker(100 * time.Millisecond)`.
///
/// Ported as a **number, not a timer**: this crate spawn no thread and own no
/// clock. A caller that want ollama's exact cadence -- both the repaint rate and
/// the spinner frame rate -- drive [`Spinner::tick`] and its own redraw off this
/// value. Diverge from it and the spinner just spin at a different speed; it is
/// cosmetic, not a correctness knob.
pub const TICK_INTERVAL_MS: u64 = 100;

/// How many rate samples a [`Bar`] keep. **Upstream:** `NewBar` sets
/// `maxBuckets: 10`.
///
/// Combined with [`BUCKET_MIN_INTERVAL_SECS`] this is the *window* the rate is
/// measured over: 10 buckets, at most one per second, so the reported speed is a
/// rolling ~10-second average. That is why a `pull` rate move smoothly instead
/// of jumping about with every chunk -- and why it take ~10s to react after the
/// network genuinely change speed. Deliberate trade upstream made; kept.
pub const MAX_BUCKETS: usize = 10;

/// Bucket throttle, in seconds. **Upstream:** `Bar.Set` --
/// `time.Since(last.updated) > time.Second`.
pub const BUCKET_MIN_INTERVAL_SECS: f64 = 1.0;

/// The braille spinner frames, in order.
///
/// **Upstream:** `progress/spinner.go` `NewSpinner` -- the exact ten runes, same
/// order. Order is load-bearing: these are the braille dot patterns U+280B..,
/// and shuffling them turn a rotation into a twitch.
pub const SPINNER_PARTS: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Left bar boundary. **Upstream:** `Bar.String()` writes `" ▕"`.
const BAR_LEFT: &str = "▕";
/// Right bar boundary. **Upstream:** `Bar.String()` writes `"▏ "`.
const BAR_RIGHT: &str = "▏";
/// Filled cell. **Upstream:** `Bar.String()` repeats `"█"`.
const BAR_FILL: &str = "█";

/// Width of the `"999 MB"` byte-count fields inside the suffix.
///
/// **Upstream:** `Bar.String()` pads with `repeat(" ", 6-len(curValue))` in four
/// places. Six because [`crate::format::human_bytes`] never produce more than
/// six characters in practice (`"999 GB"`, `"4.7 GB"`, `"512 B"`) -- values >= 10
/// drop their decimal precisely so this field stay put.
const BYTES_FIELD_WIDTH: usize = 6;

/// Total width upstream reserve for the `"  999 MB/s"` rate field.
///
/// **Upstream:** `Bar.String()` -- *"max 10 characters"*, written as two literal
/// spaces + a 6-wide byte field + `"/s"`, and blanked to ten spaces when there is
/// no rate to show. Reserved even when empty so the bar does not jitter sideways
/// the moment a download stall.
const RATE_FIELD_WIDTH: usize = 10;

/// Total width upstream reserve for the `"  59m59s"` ETA field.
///
/// **Upstream:** `Bar.String()` -- *"max 8 characters"*: two spaces + a 6-wide
/// duration. Same anti-jitter reason as [`RATE_FIELD_WIDTH`].
const ETA_FIELD_WIDTH: usize = 8;

/// Columns upstream subtract for the bar's own furniture.
///
/// **Upstream:** `Bar.String()` -- `f := termWidth - pre.Len() - suf.Len() - 5`,
/// commented *"add 5 extra spaces: 2 boundary characters and 1 space at each
/// end"*.
///
/// **Hard-won detail, and it is an upstream quirk:** that comment describe FOUR
/// columns (`▕`, `▏`, and a space either side) but the code subtract **five**.
/// The spare column is what keep a full-width bar from wrapping onto the next
/// line and smearing the whole redraw. Copied as 5 deliberately -- "fixing" it to
/// 4 would make KOPITIAM's bar one column wider than ollama's and reintroduce the
/// wrap at exactly the terminal widths people actually use.
const BAR_CHROME_COLUMNS: i64 = 5;

/// `"1h30m"`, `"5m3s"`, `"99h+"` -- the compact, two-unit duration on a bar.
///
/// **Upstream:** `progress/bar.go` `formatDuration(d time.Duration)`.
///
/// **Do NOT confuse this with [`crate::format::human_duration`].** Upstream ship
/// BOTH, they are different functions for different jobs, and this module is not
/// duplicating that one:
///
/// * `format::human_duration` (`format/format.go` `humanDuration`) is **prose**
///   for a `list` table -- `"About a minute"`, `"3 days ago"`;
/// * this one is a **fixed-width countdown** for a live bar -- `"5m3s"`. It must
///   stay short and must never change length unpredictably, because it sit in an
///   [`ETA_FIELD_WIDTH`]-column slot next to a moving bar.
///
/// The ladder, upstream's exactly:
///
/// * `>= 100h` -> `"99h+"`. A saturating label, not a real reading -- past four
///   days the estimate is noise anyway, and it caps the field width.
/// * `>= 1h` -> `"{h}h{m}m"`, minutes taken **modulo 60** and both parts
///   **truncated** (`int(d.Hours())`, `int(d.Minutes())%60`). Seconds dropped
///   entirely: two units, never three.
/// * otherwise -> Go's `time.Duration.String()` after `Round(time.Second)`, i.e.
///   `"0s"`, `"45s"`, `"1m30s"`, `"1m0s"`. Note Go does **not** drop a zero
///   seconds component once there is a minute -- `"1m0s"`, never `"1m"` -- and
///   that is reproduced here.
///
/// Negatives get a leading `"-"`, as Go does. A bar never produce one (current is
/// clamped to max and the rate is guarded positive), but this is a public
/// function and silently mangling a negative would be worse than printing it.
pub fn format_duration(seconds: f64) -> String {
    const HOUR: f64 = 3600.0;

    if seconds >= 100.0 * HOUR {
        return "99h+".to_string();
    }
    if seconds >= HOUR {
        // Go: `int(d.Hours())` and `int(d.Minutes())%60` -- truncation both
        // times, matching `as i64` here.
        let hours = (seconds / HOUR) as i64;
        let minutes = (seconds / 60.0) as i64 % 60;
        return format!("{hours}h{minutes}m");
    }

    // Go's `d.Round(time.Second)` rounds half away from zero; `f64::round` does
    // the same, so 4.5s agrees on "5s" in both.
    let whole = seconds.round() as i64;
    go_duration_string(whole)
}

/// Go's `time.Duration.String()`, restricted to whole seconds.
///
/// **Upstream:** Go's standard library `time.Duration.String()`, reached from
/// `formatDuration`'s default branch after `Round(time.Second)`. Only the
/// whole-second shape is reproduced, because the caller always round first --
/// the fractional forms Go can emit (`"1.5s"`, `"3.25ms"`) are unreachable here.
///
/// The hours branch is reachable despite `format_duration` handling `>= 1h`
/// itself: the switch upstream test the **unrounded** duration, so 59m59.6s take
/// the default branch and only *then* round up to a full hour, printing
/// `"1h0m0s"`. Faithfully kept -- it is one second a day of weirdness, and
/// diverging would be a silent difference for no gain.
fn go_duration_string(total_secs: i64) -> String {
    if total_secs == 0 {
        return "0s".to_string();
    }
    let sign = if total_secs < 0 { "-" } else { "" };
    let s = total_secs.unsigned_abs();

    let hours = s / 3600;
    let minutes = (s % 3600) / 60;
    let secs = s % 60;

    if hours > 0 {
        format!("{sign}{hours}h{minutes}m{secs}s")
    } else if minutes > 0 {
        format!("{sign}{minutes}m{secs}s")
    } else {
        format!("{sign}{secs}s")
    }
}

/// `n` copies of `s`, and zero copies when `n` is negative.
///
/// **Upstream:** `progress/bar.go` `repeat(s string, n int)`. Exists upstream
/// because `strings.Repeat` **panics** on a negative count, and every padding
/// calculation on a bar can go negative on a narrow terminal. Rust's
/// `str::repeat` take a `usize` so it cannot be handed a negative directly, but
/// the arithmetic feeding it is signed all the same -- so the guard is still
/// needed, just moved to the conversion.
fn repeat(s: &str, n: i64) -> String {
    if n > 0 {
        s.repeat(n as usize)
    } else {
        String::new()
    }
}

/// Right-align `value` in a [`BYTES_FIELD_WIDTH`] field, upstream's way.
///
/// **Upstream:** the `repeat(" ", 6-len(x))` + `x` pattern repeated four times in
/// `Bar.String()`. Note `len()` in Go is **bytes**, and `str::len()` here is too,
/// so an over-long or non-ASCII value degrade identically (no padding) rather
/// than differently.
fn pad_bytes_field(value: &str) -> String {
    let pad = BYTES_FIELD_WIDTH as i64 - value.len() as i64;
    format!("{}{}", repeat(" ", pad), value)
}

/// One rate sample: what the counter read, and when.
///
/// **Upstream:** `progress/bar.go` `type bucket struct`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Bucket {
    /// Monotonic seconds -- see the module header on clocks.
    updated: f64,
    value: i64,
}

/// A download/progress bar's state and arithmetic.
///
/// **Upstream:** `progress/bar.go` `type Bar struct`.
///
/// Owns no writer and no terminal. Feed it [`Bar::set`] as bytes arrive, then ask
/// it either for the exact upstream line ([`Bar::line`]) or for structured data
/// ([`Bar::line_data`]).
#[derive(Debug, Clone)]
pub struct Bar {
    message: String,
    /// Upstream's `messageWidth`. See [`Bar::set_message_width`] for why it is
    /// signed and why it start at `-1`.
    message_width: i64,

    max_value: i64,
    initial_value: i64,
    current_value: i64,

    /// Monotonic seconds when the bar was created.
    started: f64,
    /// `None` is Go's zero `time.Time`, i.e. "still going".
    stopped: Option<f64>,

    max_buckets: usize,
    buckets: Vec<Bucket>,
}

impl Bar {
    /// **Upstream:** `NewBar(message string, maxValue, initialValue int64)`.
    ///
    /// `initial_value` is not just a starting count -- it is the **resume point**
    /// of a partial download, and [`Bar::rate`] subtract it so bytes already on
    /// disk before this run started do not get counted as network throughput.
    /// Pass 0 for a fresh transfer.
    ///
    /// A bar created already complete (`initial >= max`) is born stopped, exactly
    /// as upstream does -- that is how a fully-cached layer render as done
    /// instead of showing a phantom 0 B/s.
    pub fn new(message: impl Into<String>, max_value: i64, initial_value: i64, now_secs: f64) -> Self {
        Self {
            message: message.into(),
            // Upstream's `messageWidth: -1`. See `set_message_width`.
            message_width: -1,
            max_value,
            initial_value,
            current_value: initial_value,
            started: now_secs,
            stopped: (initial_value >= max_value).then_some(now_secs),
            max_buckets: MAX_BUCKETS,
            buckets: Vec::new(),
        }
    }

    /// Fix the message column to `width`, truncating or padding to fit.
    ///
    /// **Divergence, stated out loud:** upstream has this field but **no setter**
    /// in the `progress` package, so on the pinned tree `messageWidth` stay `-1`
    /// for every bar and both branches that read it are dormant (`> 0` is false,
    /// and padding to `-1` is never positive). The *logic* is ported faithfully
    /// so a caller that align a column of bars get upstream's exact behaviour;
    /// only the ability to switch it on is new. Leave it alone and the output is
    /// byte-identical to ollama's.
    ///
    /// Truncation is by **bytes**, matching Go's `message[:b.messageWidth]`. That
    /// can split a multi-byte character mid-sequence upstream; here it would
    /// panic on a char boundary, so we cut at the nearest boundary at or below
    /// the limit instead. Model names are ASCII, so the two agree in practice --
    /// but "panic on a Chinese model name" was not a faithfulness worth keeping.
    pub fn set_message_width(&mut self, width: i64) {
        self.message_width = width;
    }

    /// Update the byte counter. **Upstream:** `Bar.Set(value int64)`.
    ///
    /// Clamps to `max_value` and stops the bar on reaching it, both upstream's.
    /// Bucket appends are throttled to one per [`BUCKET_MIN_INTERVAL_SECS`] and
    /// the ring is trimmed from the front at [`MAX_BUCKETS`] -- so calling this
    /// on every 8 KiB chunk is cheap and correct, which is exactly how a pull
    /// loop want to use it.
    ///
    /// Note the throttle test upstream is `time.Since(last) > time.Second`,
    /// **strictly** greater, and it is reproduced strictly. Exactly 1.0s does not
    /// open a new bucket.
    pub fn set(&mut self, value: i64, now_secs: f64) {
        let value = value.min(self.max_value);

        self.current_value = value;
        if self.current_value >= self.max_value {
            self.stopped = Some(now_secs);
        }

        let due = match self.buckets.last() {
            None => true,
            Some(last) => now_secs - last.updated > BUCKET_MIN_INTERVAL_SECS,
        };

        if due {
            self.buckets.push(Bucket { updated: now_secs, value });
            if self.buckets.len() > self.max_buckets {
                self.buckets.remove(0);
            }
        }
    }

    /// Bytes transferred so far, after clamping.
    pub fn current(&self) -> i64 {
        self.current_value
    }

    /// Total bytes expected.
    pub fn max(&self) -> i64 {
        self.max_value
    }

    /// Has the bar reached its maximum? (Go's `!stopped.IsZero()`.)
    pub fn is_finished(&self) -> bool {
        self.stopped.is_some()
    }

    /// Completion as a **percentage 0..=100**, not a 0..=1 fraction.
    ///
    /// **Upstream:** `Bar.percent()`. Returns 0 when `max_value <= 0` rather than
    /// dividing -- upstream guard `if b.maxValue > 0`, and a zero-length blob is
    /// a real case (an empty layer), so this is not defensive padding.
    pub fn percent(&self) -> f64 {
        if self.max_value > 0 {
            self.current_value as f64 / self.max_value as f64 * 100.0
        } else {
            0.0
        }
    }

    /// Bytes per second, estimated. **Upstream:** `Bar.rate()`.
    ///
    /// Three regimes, upstream's exactly:
    ///
    /// * **Finished** -- the honest overall average for the run:
    ///   `(current - initial) / (stopped - started)`. Note this use `initial`, so
    ///   a resumed download report the speed of *this* session, not a fiction
    ///   inflated by bytes that were already on disk.
    /// * **One bucket** -- not enough samples for a window, so it fall back to
    ///   "since the bar started", again against `initial`.
    /// * **Two or more** -- the rolling window: first-to-last bucket only. Bytes
    ///   before the window are irrelevant, which is the whole point of the ring.
    ///
    /// **The denominator is rounded to whole seconds** (`.Round(time.Second)`
    /// upstream) before dividing. That is not cosmetic: it mean a window shorter
    /// than half a second round to **zero**, and the guard below then report a
    /// rate of 0 rather than a divide-by-zero or an absurd spike. So a bar shows
    /// no speed for its first moment -- correct, not a bug, and the reason the
    /// rate field is blanked rather than showing "∞ B/s".
    pub fn rate(&self) -> f64 {
        let (numerator, denominator) = match self.stopped {
            Some(stopped) => (
                (self.current_value - self.initial_value) as f64,
                (stopped - self.started).round(),
            ),
            None => match self.buckets.len() {
                0 => (0.0, 0.0),
                1 => {
                    let only = self.buckets[0];
                    (
                        (only.value - self.initial_value) as f64,
                        (only.updated - self.started).round(),
                    )
                }
                _ => {
                    let first = self.buckets[0];
                    // Length checked above, so `last()` cannot be None; matched
                    // rather than unwrapped because library code does not unwrap.
                    let Some(&last) = self.buckets.last() else {
                        return 0.0;
                    };
                    (
                        (last.value - first.value) as f64,
                        (last.updated - first.updated).round(),
                    )
                }
            },
        };

        if denominator != 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Seconds remaining, or `None` when there is nothing sensible to say.
    ///
    /// **Upstream:** inline in `Bar.String()` --
    /// `time.Duration(int64(float64(maxValue-currentValue)/rate)) * time.Second`.
    /// Lifted out to a method because a caller drawing its own widget need the
    /// number, not the string.
    ///
    /// `None` when the bar is finished or the rate is not positive, matching the
    /// `b.stopped.IsZero() && rate > 0` guard that decide whether upstream print
    /// an ETA at all. **Whole seconds, truncated** -- upstream's `int64(...)`
    /// conversion happen before the multiply by `time.Second`, so sub-second
    /// precision is discarded there and discarded here.
    pub fn eta_seconds(&self) -> Option<f64> {
        let rate = self.rate();
        if self.stopped.is_some() || rate <= 0.0 {
            return None;
        }
        let remaining = (self.max_value - self.current_value) as f64 / rate;
        // Go truncates toward zero via `int64(...)`; `as i64` matches.
        Some(remaining as i64 as f64)
    }

    /// The message, trimmed and fitted to `message_width`.
    ///
    /// **Upstream:** the `strings.TrimSpace` + truncate prologue shared by
    /// `Bar.String()` and `Spinner.String()`.
    fn fitted_message(&self) -> String {
        fit_message(&self.message, self.message_width)
    }

    /// Everything the line says, as data -- for a caller drawing its own widget.
    ///
    /// This is the seam that keep the crate TUI-free: a KOPITIAM TUI take this
    /// and render however it like, without parsing a formatted string back apart.
    /// The `*_human` fields use [`crate::format::human_bytes`], i.e. **decimal**
    /// units (1 KB = 1000 B) -- correct here, because a download is measured the
    /// way the registry report it. Do not swap in `human_bytes2`; that is the
    /// binary-unit one for VRAM accounting, and mixing them is a 7.4% lie.
    pub fn line_data(&self) -> BarLine {
        let rate = self.rate();
        let show_rate = self.stopped.is_none() && rate > 0.0;
        let eta = self.eta_seconds();

        BarLine {
            message: self.fitted_message(),
            percent: self.percent(),
            current: self.current_value,
            max: self.max_value,
            current_human: human_bytes(self.current_value),
            max_human: human_bytes(self.max_value),
            rate_bytes_per_sec: rate,
            rate_human: show_rate.then(|| format!("{}/s", human_bytes(rate as i64))),
            eta_seconds: eta,
            eta_human: eta.map(format_duration),
            finished: self.stopped.is_some(),
        }
    }

    /// The bar as ollama would print it, at a given terminal width.
    ///
    /// **Upstream:** `Bar.String()`, with one change: upstream call
    /// `term.GetSize(os.Stderr.Fd())` and fall back to `defaultTermWidth`; we take
    /// the width as an argument. Pass [`DEFAULT_TERM_WIDTH`] to reproduce
    /// upstream's piped-output layout exactly.
    ///
    /// Layout, left to right: message, `%3.0f%%`, the bar, then a fixed-width
    /// suffix of `current/max`, rate and ETA. The suffix fields keep their width
    /// even when empty so the bar edge does not twitch every time the rate
    /// estimate drop to zero.
    ///
    /// **Known upstream quirk, reproduced:** the width budget is computed from
    /// **byte** lengths (`pre.Len()`, `suf.Len()`), while the bar glyphs are
    /// 3-byte, 1-column box characters accounted for by hand via
    /// [`BAR_CHROME_COLUMNS`]. Bytes and columns therefore only agree while the
    /// message is ASCII. Every model name is, so upstream never notice; a
    /// non-ASCII message would render a bar slightly too short in both
    /// implementations, identically.
    pub fn line(&self, term_width: usize) -> String {
        let mut pre = String::new();
        if !self.message.is_empty() {
            let message = self.fitted_message();
            pre.push_str(&message);
            let padding = self.message_width - pre.len() as i64;
            pre.push_str(&repeat(" ", padding));
            pre.push(' ');
        }

        // Go's `%3.0f%%`. Both Go and Rust round half to even here, and a byte
        // ratio essentially never lands on an exact .5 tie anyway.
        pre.push_str(&format!("{:3.0}%", self.percent()));

        let mut suf = String::new();
        // "999 MB/999 MB" -- while running, current/max; once done, just max
        // followed by blanks, so the line settle instead of showing "8 GB/8 GB".
        if self.stopped.is_none() {
            suf.push_str(&pad_bytes_field(&human_bytes(self.current_value)));
            suf.push('/');
            suf.push_str(&pad_bytes_field(&human_bytes(self.max_value)));
        } else {
            suf.push_str(&pad_bytes_field(&human_bytes(self.max_value)));
            // Upstream pads 7 here: the 6-wide field it is standing in for, plus
            // the "/" that is no longer printed.
            suf.push_str(&repeat(" ", (BYTES_FIELD_WIDTH + 1) as i64));
        }

        let rate = self.rate();
        if self.stopped.is_none() && rate > 0.0 {
            suf.push_str("  ");
            suf.push_str(&pad_bytes_field(&human_bytes(rate as i64)));
            suf.push_str("/s");
        } else {
            suf.push_str(&repeat(" ", RATE_FIELD_WIDTH as i64));
        }

        match self.eta_seconds() {
            Some(remaining) => {
                suf.push_str("  ");
                suf.push_str(&pad_bytes_field(&format_duration(remaining)));
            }
            None => suf.push_str(&repeat(" ", ETA_FIELD_WIDTH as i64)),
        }

        let free = term_width as i64 - pre.len() as i64 - suf.len() as i64 - BAR_CHROME_COLUMNS;
        let filled = (free as f64 * self.percent() / 100.0) as i64;

        let mut mid = String::new();
        mid.push(' ');
        mid.push_str(BAR_LEFT);
        mid.push_str(&repeat(BAR_FILL, filled));
        mid.push_str(&repeat(" ", free - filled));
        mid.push_str(BAR_RIGHT);
        mid.push(' ');

        pre + &mid + &suf
    }
}

/// What a [`Bar`]'s line says, decomposed -- so a caller can draw its own.
///
/// Not an upstream type. Upstream only ever produce the formatted string, since
/// its single consumer is a terminal. KOPITIAM need the numbers too (a TUI, a
/// log line, a progress event over a channel), and scraping them back out of a
/// padded string would be daft. [`Bar::line`] still give the exact upstream
/// string when that is what you want.
#[derive(Debug, Clone, PartialEq)]
pub struct BarLine {
    /// Trimmed and width-fitted, ready to print.
    pub message: String,
    /// 0..=100, not a fraction. See [`Bar::percent`].
    pub percent: f64,
    /// Raw byte counts, for a caller that want to do its own maths.
    pub current: i64,
    pub max: i64,
    /// Decimal-unit renderings of the two counts above.
    pub current_human: String,
    pub max_human: String,
    /// Estimated throughput. Zero while the sampling window is still too short.
    pub rate_bytes_per_sec: f64,
    /// `"4.7 MB/s"`, or `None` when upstream would print blanks.
    pub rate_human: Option<String>,
    /// Whole seconds remaining, or `None` when there is no usable estimate.
    pub eta_seconds: Option<f64>,
    /// `"5m3s"`, or `None`. Formatted by [`format_duration`], not by
    /// `format::human_duration` -- see that function on why they differ.
    pub eta_human: Option<String>,
    pub finished: bool,
}

/// A braille spinner for work with no measurable total.
///
/// **Upstream:** `progress/spinner.go` `type Spinner struct`.
///
/// **Divergence, stated:** upstream's spinner own a goroutine and an
/// `atomic.Value` message, because its frame advance on a timer the struct
/// itself start. This one advance only when the caller call [`Spinner::tick`],
/// so it need no thread, no atomics and no interior mutability -- ordinary `&mut
/// self`. Same frames, same modular arithmetic, same stop behaviour; the clock
/// simply belong to whoever own the redraw loop. Drive it every
/// [`TICK_INTERVAL_MS`] to match ollama's speed.
#[derive(Debug, Clone)]
pub struct Spinner {
    message: String,
    message_width: i64,
    value: usize,
    started: f64,
    stopped: Option<f64>,
}

impl Spinner {
    /// **Upstream:** `NewSpinner(message string)`.
    ///
    /// `message_width` start at 0 -- Go's zero value, since upstream's
    /// constructor never set it. Same dormant-by-default behaviour as
    /// [`Bar::set_message_width`].
    pub fn new(message: impl Into<String>, now_secs: f64) -> Self {
        Self {
            message: message.into(),
            message_width: 0,
            value: 0,
            started: now_secs,
            stopped: None,
        }
    }

    /// **Upstream:** `Spinner.SetMessage`. Upstream store it in an
    /// `atomic.Value` because its render goroutine read concurrently; here the
    /// caller own both sides, so a plain field is enough and strictly safer.
    pub fn set_message(&mut self, message: impl Into<String>) {
        self.message = message.into();
    }

    /// See [`Bar::set_message_width`] -- same field, same dormant default.
    pub fn set_message_width(&mut self, width: i64) {
        self.message_width = width;
    }

    /// Advance one frame. **Upstream:** `Spinner.start()`'s
    /// `s.value = (s.value + 1) % len(s.parts)`.
    ///
    /// A stopped spinner does not advance -- upstream's goroutine `return`s once
    /// `stopped` is set. Calling this on a stopped spinner is a harmless no-op
    /// rather than an error, since a shared redraw loop should not have to know
    /// which of its spinners have finished.
    pub fn tick(&mut self) {
        if self.stopped.is_none() {
            self.value = (self.value + 1) % SPINNER_PARTS.len();
        }
    }

    /// **Upstream:** `Spinner.Stop()` -- idempotent there (`if
    /// s.stopped.IsZero()`), idempotent here. The first stop time wins, so a
    /// double stop cannot stretch a measured duration.
    pub fn stop(&mut self, now_secs: f64) {
        if self.stopped.is_none() {
            self.stopped = Some(now_secs);
        }
    }

    /// Has it been stopped?
    pub fn is_stopped(&self) -> bool {
        self.stopped.is_some()
    }

    /// Seconds it has been spinning, given the current time. Upstream keep
    /// `started` and never read it; exposed here because "how long did that step
    /// take" is exactly what a caller want to log afterwards.
    pub fn elapsed_secs(&self, now_secs: f64) -> f64 {
        self.stopped.unwrap_or(now_secs) - self.started
    }

    /// The current frame, or `None` once stopped.
    ///
    /// **Upstream:** the `if s.stopped.IsZero()` branch in `Spinner.String()` --
    /// a stopped spinner print no glyph at all, leaving just its message. That is
    /// how a completed step turn into a plain line of text without a redraw.
    pub fn frame(&self) -> Option<&'static str> {
        if self.stopped.is_some() {
            return None;
        }
        SPINNER_PARTS.get(self.value).copied()
    }

    /// The spinner line, exactly as upstream print it.
    ///
    /// **Upstream:** `Spinner.String()`. Note both parts are optional, so a
    /// stopped spinner with an empty message render as `""` -- upstream's
    /// behaviour, kept, because the caller's render loop is what decides whether
    /// a blank line gets drawn.
    ///
    /// Takes `_term_width` only so it slot into the same call shape as
    /// [`Bar::line`] inside [`Progress`]; upstream's spinner ignore the width too.
    pub fn line(&self, _term_width: usize) -> String {
        let mut sb = String::new();

        if !self.message.is_empty() {
            let message = fit_message(&self.message, self.message_width);
            sb.push_str(&message);
            let padding = self.message_width - sb.len() as i64;
            sb.push_str(&repeat(" ", padding));
            sb.push(' ');
        }

        if let Some(frame) = self.frame() {
            sb.push_str(frame);
            sb.push(' ');
        }

        sb
    }
}

/// Step-based progress -- `"Generating  33% ▕███      ▏ 3/9"`.
///
/// **Upstream:** `progress/stepbar.go` `type StepBar struct`.
///
/// Note the bar width **is the step count** upstream (`barWidth := s.total`), not
/// a terminal-derived width -- one cell per step. That is why it is a different
/// type from [`Bar`] rather than a mode of it: it is counting discrete steps, not
/// filling a line.
#[derive(Debug, Clone)]
pub struct StepBar {
    message: String,
    current: usize,
    total: usize,
}

impl StepBar {
    /// **Upstream:** `NewStepBar(message string, total int)`.
    pub fn new(message: impl Into<String>, total: usize) -> Self {
        Self { message: message.into(), current: 0, total }
    }

    /// **Upstream:** `StepBar.Set(current int)`.
    ///
    /// **Divergence, stated:** upstream do not clamp, and `StepBar.String()` then
    /// call `strings.Repeat(" ", total-current)` with a negative count, which
    /// **panics**. We clamp to `total` instead. This crate does not panic in
    /// library code, and a caller overshooting its own step count is a caller
    /// bug that should not take the process down mid-pull.
    pub fn set(&mut self, current: usize) {
        self.current = current.min(self.total);
    }

    /// Steps done.
    pub fn current(&self) -> usize {
        self.current
    }

    /// Steps expected.
    pub fn total(&self) -> usize {
        self.total
    }

    /// Completion as a percentage 0..=100.
    ///
    /// **Divergence, stated:** upstream compute `float64(current)/float64(total)`
    /// unguarded, so `total == 0` give `NaN` (or `+Inf`) and print `"NaN%"`. We
    /// return 0 for an empty step list, same guard [`Bar::percent`] already has
    /// upstream. Printing "NaN" at a user is not a behaviour worth preserving.
    pub fn percent(&self) -> f64 {
        if self.total > 0 {
            self.current as f64 / self.total as f64 * 100.0
        } else {
            0.0
        }
    }

    /// The step bar as upstream print it.
    ///
    /// **Upstream:** `StepBar.String()` --
    /// `"%s %3.0f%% ▕%s%s▏ %d/%d"`. Note there is exactly one space between the
    /// message and the percentage and **no** message-width padding here, unlike
    /// [`Bar::line`]; a step bar is single-purpose so upstream never needed to
    /// align a column of them.
    ///
    /// Takes `_term_width` for call-shape symmetry only -- the width is the step
    /// count, per the type docs.
    pub fn line(&self, _term_width: usize) -> String {
        let empty = self.total.saturating_sub(self.current);
        format!(
            "{} {:3.0}% {}{}{}{} {}/{}",
            self.message,
            self.percent(),
            BAR_LEFT,
            BAR_FILL.repeat(self.current),
            " ".repeat(empty),
            BAR_RIGHT,
            self.current,
            self.total
        )
    }
}

/// Trim, then fit to `width` -- the prologue [`Bar`] and [`Spinner`] share.
///
/// **Upstream:** duplicated verbatim in `Bar.String()` and `Spinner.String()`.
/// Factored out here rather than duplicated: it is the same four lines with the
/// same meaning, and two copies would be two places to get the boundary handling
/// wrong.
///
/// A `width <= 0` mean "unbounded", which is how upstream's `-1` and `0`
/// defaults behave. Truncation cut at a char boundary at or below `width` bytes,
/// where Go slice bytes blindly -- see [`Bar::set_message_width`] for why.
fn fit_message(message: &str, width: i64) -> String {
    let trimmed = message.trim();
    if width > 0 && trimmed.len() > width as usize {
        let mut end = width as usize;
        while end > 0 && !trimmed.is_char_boundary(end) {
            end -= 1;
        }
        return trimmed[..end].to_string();
    }
    trimmed.to_string()
}

/// One of the three things a [`Progress`] can be tracking.
///
/// **Upstream:** `progress/progress.go` `type State interface { String() string }`.
///
/// Ported as an **enum, not a trait**, and deliberately: upstream's own `stop()`
/// immediately type-switches the interface back to a concrete `*Spinner`
/// (`if spinner, ok := state.(*Spinner); ok`), which is an interface being used
/// as a closed set. Rust has a closed set already. An enum also keep the crate
/// free of `dyn`/`Any` downcasting for something that has exactly three
/// variants, and adding a trait seam nobody asked for is the kind of invented
/// abstraction CLAUDE.md tell us to avoid.
#[derive(Debug, Clone)]
pub enum Tracker {
    Bar(Bar),
    Spinner(Spinner),
    Step(StepBar),
}

impl Tracker {
    /// This tracker's line, at a given terminal width.
    ///
    /// **Upstream:** the `State.String()` call in `Progress.render()`.
    pub fn line(&self, term_width: usize) -> String {
        match self {
            Tracker::Bar(b) => b.line(term_width),
            Tracker::Spinner(s) => s.line(term_width),
            Tracker::Step(s) => s.line(term_width),
        }
    }
}

/// A keyed tracker inside a [`Progress`].
///
/// The key is upstream's `Add(key string, state State)` argument. **Worth
/// knowing:** upstream **accept the key and then throw it away** -- `Add` only
/// does `p.states = append(p.states, state)`, so nothing can ever look a tracker
/// up again. We keep it, because a keyed lookup is the obvious thing a caller
/// want (update the bar for digest `sha256:abc...` as its chunks land) and
/// discarding it would force every caller to maintain a parallel index. Purely
/// additive: render order and render content are unaffected.
#[derive(Debug, Clone)]
pub struct Entry {
    pub key: String,
    pub tracker: Tracker,
}

/// A stack of trackers, and the arithmetic deciding which of them fit on screen.
///
/// **Upstream:** `progress/progress.go` `type Progress struct`.
///
/// **What is NOT here, and why:** upstream's `Progress` also own a
/// `bufio.Writer`, a repaint goroutine, and the ANSI sequences that park the
/// cursor (`\033[A`), clear a line (`\033[2K\033[1G`), hide the cursor
/// (`\033[?25l`) and open a synchronised-output frame (`\033[?2026h`). All of
/// that is terminal control, and this crate has no TUI dependency and is not
/// getting one. So the caller own the writer, the escapes and the timer; this
/// type own the list, the ordering, and the visible-window calculation.
///
/// If you are writing that caller, the two things upstream do that are easy to
/// miss: it repaint every [`TICK_INTERVAL_MS`], and `StopAndClear` erase exactly
/// as many lines as were **last rendered** (upstream's `pos`), not as many
/// trackers as exist -- because the window below may have shown fewer.
#[derive(Debug, Clone, Default)]
pub struct Progress {
    entries: Vec<Entry>,
    /// Upstream's implicit "the ticker is still going" state, made explicit.
    running: bool,
}

impl Progress {
    /// **Upstream:** `NewProgress(w io.Writer)`, minus the writer and minus the
    /// `go p.start()` that launch the repaint goroutine.
    pub fn new() -> Self {
        Self { entries: Vec::new(), running: true }
    }

    /// **Upstream:** `Progress.Add(key string, state State)`. Order of addition
    /// is render order, top to bottom.
    pub fn add(&mut self, key: impl Into<String>, tracker: Tracker) {
        self.entries.push(Entry { key: key.into(), tracker });
    }

    /// Every tracker, in render order.
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// Mutable access by key -- what upstream's discarded key would have bought.
    /// See [`Entry`].
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Tracker> {
        self.entries
            .iter_mut()
            .find(|e| e.key == key)
            .map(|e| &mut e.tracker)
    }

    /// Is the progress display still live?
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Stop every spinner and mark the display finished.
    ///
    /// **Upstream:** `Progress.stop()`. Returns whether it was still running --
    /// upstream key that off `p.ticker != nil`, and both `Stop` and
    /// `StopAndClear` use the answer to decide whether to emit anything at all,
    /// so a double stop cannot print a stray newline or erase lines twice.
    ///
    /// Bars are **not** stopped here, matching upstream: a bar stop itself on
    /// reaching its maximum, and force-stopping a half-finished one would make it
    /// report a completion rate for a transfer that got cancelled.
    pub fn stop(&mut self, now_secs: f64) -> bool {
        for entry in &mut self.entries {
            if let Tracker::Spinner(spinner) = &mut entry.tracker {
                spinner.stop(now_secs);
            }
        }

        if self.running {
            self.running = false;
            true
        } else {
            false
        }
    }

    /// The trailing window of trackers that fit in `term_height` rows.
    ///
    /// **Upstream:** `Progress.render()` --
    /// `maxHeight := min(len(p.states), termHeight)`, then it iterate from
    /// `len(states)-maxHeight` to the end. So when there are more trackers than
    /// rows, the **oldest scroll off the top** and the newest stay visible. That
    /// is the right end to drop: during a pull the finished layers are the old
    /// ones, and the active transfer is what the user is watching.
    ///
    /// A `term_height` of 0 yield nothing, same as upstream's `min`.
    pub fn visible(&self, term_height: usize) -> &[Entry] {
        let max_height = self.entries.len().min(term_height);
        &self.entries[self.entries.len() - max_height..]
    }

    /// The lines to draw, top to bottom, for a given terminal size.
    ///
    /// **Upstream:** the body of `Progress.render()`, with every escape sequence
    /// removed -- upstream interleave `"\033[K"` after each line and `"\n"`
    /// between them. Joining and clearing is the caller's job; what is ported is
    /// *which* trackers get drawn and *what* each one say.
    pub fn render_lines(&self, term_width: usize, term_height: usize) -> Vec<String> {
        self.visible(term_height)
            .iter()
            .map(|e| e.tracker.line(term_width))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------
    // format_duration -- the compact two-unit countdown
    // ---------------------------------------------------------------------

    #[test]
    fn format_duration_keeps_at_most_two_units() {
        assert_eq!(format_duration(0.0), "0s");
        assert_eq!(format_duration(45.0), "45s");
        // Go never drops a zero seconds component once a minute is present.
        assert_eq!(format_duration(60.0), "1m0s");
        assert_eq!(format_duration(90.0), "1m30s");
        assert_eq!(format_duration(59.0 * 60.0 + 59.0), "59m59s");
        // Past an hour the seconds are dropped entirely: two units, never three.
        assert_eq!(format_duration(3600.0), "1h0m");
        assert_eq!(format_duration(3600.0 + 30.0 * 60.0 + 59.0), "1h30m");
        // Minutes wrap modulo 60, they do not accumulate.
        assert_eq!(format_duration(5.0 * 3600.0 + 3.0 * 60.0), "5h3m");
    }

    #[test]
    fn format_duration_saturates_past_a_hundred_hours() {
        assert_eq!(format_duration(100.0 * 3600.0), "99h+");
        assert_eq!(format_duration(1000.0 * 3600.0), "99h+");
        // Just under the cap still reports a real reading.
        assert_eq!(format_duration(99.0 * 3600.0 + 59.0 * 60.0), "99h59m");
    }

    #[test]
    fn format_duration_rounds_to_the_nearest_second_below_an_hour() {
        assert_eq!(format_duration(4.4), "4s");
        assert_eq!(format_duration(4.5), "5s");
        assert_eq!(format_duration(0.4), "0s");
    }

    /// The upstream edge the docs call out: the `>= 1h` switch tests the
    /// UNROUNDED duration, so 59m59.6s falls through to the default branch and
    /// only then rounds up into a full hour.
    #[test]
    fn a_duration_can_round_up_into_the_hours_shape_from_the_default_branch() {
        assert_eq!(format_duration(59.0 * 60.0 + 59.6), "1h0m0s");
    }

    #[test]
    fn format_duration_marks_a_negative_rather_than_mangling_it() {
        assert_eq!(format_duration(-90.0), "-1m30s");
    }

    /// The bar's countdown and `format::human_duration` are DIFFERENT upstream
    /// functions with different jobs. Pinned so nobody "deduplicates" them.
    #[test]
    fn the_bar_countdown_is_not_the_prose_duration_formatter() {
        assert_eq!(format_duration(60.0), "1m0s");
        assert_eq!(crate::format::human_duration(60.0), "About a minute");
    }

    // ---------------------------------------------------------------------
    // Bar -- percent, rate, ETA
    // ---------------------------------------------------------------------

    #[test]
    fn a_fresh_bar_starts_empty_and_unfinished() {
        let bar = Bar::new("pulling", 1000, 0, 0.0);
        assert_eq!(bar.percent(), 0.0);
        assert_eq!(bar.rate(), 0.0);
        assert_eq!(bar.eta_seconds(), None);
        assert!(!bar.is_finished());
    }

    #[test]
    fn a_bar_born_complete_is_born_stopped() {
        let bar = Bar::new("cached", 1000, 1000, 0.0);
        assert!(bar.is_finished());
        assert_eq!(bar.percent(), 100.0);
    }

    #[test]
    fn percent_returns_zero_for_a_zero_length_blob_rather_than_dividing() {
        let bar = Bar::new("empty layer", 0, 0, 0.0);
        assert_eq!(bar.percent(), 0.0);
    }

    #[test]
    fn setting_past_the_maximum_clamps_and_finishes_the_bar() {
        let mut bar = Bar::new("pulling", 1000, 0, 0.0);
        bar.set(5000, 1.0);
        assert_eq!(bar.current(), 1000);
        assert!(bar.is_finished());
        assert_eq!(bar.percent(), 100.0);
    }

    /// The ring is capped at [`MAX_BUCKETS`] and trimmed from the front, so the
    /// rate is a rolling window rather than a lifetime average.
    #[test]
    fn the_bucket_ring_is_capped_and_drops_the_oldest_sample() {
        let mut bar = Bar::new("pulling", 1_000_000, 0, 0.0);
        for i in 1..=20 {
            bar.set(i * 1000, i as f64 * 2.0);
        }
        assert_eq!(bar.buckets.len(), MAX_BUCKETS);
        // Oldest surviving sample is the 11th, at t=22s.
        assert_eq!(bar.buckets[0].updated, 22.0);
    }

    /// Upstream throttles with a STRICT `>`, so exactly one second apart does
    /// not open a new bucket.
    #[test]
    fn bucket_updates_are_throttled_to_strictly_more_than_one_second() {
        let mut bar = Bar::new("pulling", 1_000_000, 0, 0.0);
        bar.set(100, 0.0);
        assert_eq!(bar.buckets.len(), 1);
        bar.set(200, 0.5);
        assert_eq!(bar.buckets.len(), 1, "half a second is too soon");
        bar.set(300, 1.0);
        assert_eq!(bar.buckets.len(), 1, "exactly one second is still too soon");
        bar.set(400, 1.01);
        assert_eq!(bar.buckets.len(), 2);
    }

    #[test]
    fn one_bucket_measures_the_rate_from_the_bars_start() {
        let mut bar = Bar::new("pulling", 1_000_000, 0, 0.0);
        bar.set(4000, 2.0);
        // (4000 - initial 0) / (2s - 0s)
        assert_eq!(bar.rate(), 2000.0);
    }

    #[test]
    fn two_or_more_buckets_measure_the_rate_across_the_window_only() {
        let mut bar = Bar::new("pulling", 1_000_000, 0, 0.0);
        bar.set(1000, 1.1);
        bar.set(9000, 5.1);
        // (9000 - 1000) / round(5.1 - 1.1) == 8000 / 4
        assert_eq!(bar.rate(), 2000.0);
    }

    /// A resumed download must report the speed of THIS session, not a fiction
    /// inflated by bytes that were already on disk.
    #[test]
    fn a_resumed_bar_excludes_its_initial_bytes_from_the_rate() {
        let mut bar = Bar::new("resuming", 1_000_000, 500_000, 0.0);
        bar.set(502_000, 2.0);
        // (502_000 - 500_000) / 2, NOT 502_000 / 2.
        assert_eq!(bar.rate(), 1000.0);
    }

    /// The denominator is rounded to whole seconds, so a sub-half-second window
    /// rounds to zero and the guard reports 0 instead of an absurd spike.
    #[test]
    fn a_window_shorter_than_half_a_second_reports_no_rate_at_all() {
        let mut bar = Bar::new("pulling", 1_000_000, 0, 0.0);
        bar.set(100_000, 0.2);
        assert_eq!(bar.rate(), 0.0);
        assert_eq!(bar.eta_seconds(), None);
    }

    #[test]
    fn a_finished_bar_reports_its_overall_average() {
        let mut bar = Bar::new("pulling", 10_000, 0, 0.0);
        bar.set(10_000, 5.0);
        assert!(bar.is_finished());
        assert_eq!(bar.rate(), 2000.0);
        // ...and offers no ETA, because there is nothing left to wait for.
        assert_eq!(bar.eta_seconds(), None);
    }

    #[test]
    fn eta_is_whole_seconds_truncated() {
        let mut bar = Bar::new("pulling", 10_000, 0, 0.0);
        bar.set(3000, 3.0);
        // rate == 1000 B/s, 7000 B left -> exactly 7s.
        assert_eq!(bar.rate(), 1000.0);
        assert_eq!(bar.eta_seconds(), Some(7.0));

        let mut odd = Bar::new("pulling", 10_000, 0, 0.0);
        odd.set(4000, 3.0);
        // rate == 1333.33 B/s, 6000 B left -> 4.5s, truncated to 4.
        assert_eq!(odd.eta_seconds(), Some(4.0));
    }

    // ---------------------------------------------------------------------
    // Bar -- the rendered line
    // ---------------------------------------------------------------------

    #[test]
    fn a_bar_line_fills_exactly_the_terminal_width_it_is_given() {
        let mut bar = Bar::new("pulling manifest", 8_000_000, 0, 0.0);
        bar.set(4_000_000, 4.0);
        let line = bar.line(DEFAULT_TERM_WIDTH);
        // Three 3-byte box glyphs push the byte length past the column count;
        // the COLUMN count is what must land on the terminal width.
        let columns = line.chars().count();
        assert_eq!(
            columns,
            DEFAULT_TERM_WIDTH - BAR_CHROME_COLUMNS as usize + 4,
            "upstream's budget reserves 5 columns but only draws 4 of them"
        );
    }

    #[test]
    fn a_running_bar_shows_current_over_max_a_rate_and_an_eta() {
        let mut bar = Bar::new("pulling", 8_000_000, 0, 0.0);
        bar.set(2_000_000, 2.0);
        let line = bar.line(DEFAULT_TERM_WIDTH);
        assert!(line.starts_with("pulling  25%"), "got {line:?}");
        assert!(line.contains("2 MB/  8 MB"), "got {line:?}");
        assert!(line.contains("1 MB/s"), "got {line:?}");
        assert!(line.trim_end().ends_with("6s"), "got {line:?}");
    }

    /// Once finished the line settles: max only, and the rate/ETA slots go blank
    /// but keep their width so the bar edge does not twitch.
    #[test]
    fn a_finished_bar_drops_the_current_count_and_blanks_the_rate_and_eta() {
        let mut bar = Bar::new("pulling", 8_000_000, 0, 0.0);
        bar.set(8_000_000, 4.0);
        let line = bar.line(DEFAULT_TERM_WIDTH);
        assert!(line.contains("100%"), "got {line:?}");
        assert!(!line.contains('/'), "no current/max and no rate: {line:?}");
        assert!(line.ends_with(&" ".repeat(RATE_FIELD_WIDTH + ETA_FIELD_WIDTH)));
    }

    #[test]
    fn a_bar_with_no_message_prints_no_leading_space() {
        let bar = Bar::new("", 1000, 0, 0.0);
        assert!(bar.line(DEFAULT_TERM_WIDTH).starts_with("  0%"));
    }

    #[test]
    fn a_message_width_truncates_and_pads_the_message_column() {
        let mut bar = Bar::new("  a-very-long-model-name  ", 1000, 0, 0.0);
        bar.set_message_width(8);
        assert!(bar.line(DEFAULT_TERM_WIDTH).starts_with("a-very-l   0%"));

        let mut short = Bar::new("hi", 1000, 0, 0.0);
        short.set_message_width(8);
        assert!(short.line(DEFAULT_TERM_WIDTH).starts_with("hi         0%"));
    }

    /// Where upstream would slice a multi-byte character in half, we cut at the
    /// nearest boundary below the limit instead of panicking.
    #[test]
    fn truncating_a_multibyte_message_cuts_on_a_character_boundary() {
        let mut bar = Bar::new("模型名称", 1000, 0, 0.0);
        bar.set_message_width(4);
        // Each glyph is 3 bytes, so a 4-byte limit fits exactly one.
        assert!(bar.line(DEFAULT_TERM_WIDTH).starts_with("模 "));
    }

    #[test]
    fn a_narrow_terminal_degrades_instead_of_panicking() {
        let mut bar = Bar::new("pulling a model with a long name", 8_000_000, 0, 0.0);
        bar.set(2_000_000, 2.0);
        // Far narrower than the suffix alone -- every padding calculation goes
        // negative, which is exactly what upstream's `repeat` guard is for.
        let line = bar.line(10);
        assert!(line.contains(BAR_LEFT) && line.contains(BAR_RIGHT));
    }

    #[test]
    fn line_data_exposes_the_same_numbers_the_line_prints() {
        let mut bar = Bar::new("  pulling  ", 8_000_000, 0, 0.0);
        bar.set(2_000_000, 2.0);
        let data = bar.line_data();

        assert_eq!(data.message, "pulling", "trimmed, like upstream");
        assert_eq!(data.percent, 25.0);
        assert_eq!(data.current, 2_000_000);
        assert_eq!(data.max, 8_000_000);
        assert_eq!(data.current_human, "2 MB");
        assert_eq!(data.max_human, "8 MB");
        assert_eq!(data.rate_bytes_per_sec, 1_000_000.0);
        assert_eq!(data.rate_human.as_deref(), Some("1 MB/s"));
        assert_eq!(data.eta_seconds, Some(6.0));
        assert_eq!(data.eta_human.as_deref(), Some("6s"));
        assert!(!data.finished);
    }

    #[test]
    fn line_data_blanks_the_rate_and_eta_once_finished() {
        let mut bar = Bar::new("pulling", 8_000_000, 0, 0.0);
        bar.set(8_000_000, 4.0);
        let data = bar.line_data();
        assert!(data.finished);
        assert_eq!(data.rate_human, None);
        assert_eq!(data.eta_human, None);
        // The overall average is still available as a number, for a summary line.
        assert_eq!(data.rate_bytes_per_sec, 2_000_000.0);
    }

    /// Downloads are measured in DECIMAL units, the way a registry reports them.
    /// Pinned because reaching for `human_bytes2` here would be a silent 7.4% lie.
    #[test]
    fn a_bar_reports_decimal_units_not_binary_ones() {
        let bar = Bar::new("pulling", 8 * 1024 * 1024 * 1024, 0, 0.0);
        assert_eq!(bar.line_data().max_human, "8.6 GB");
    }

    // ---------------------------------------------------------------------
    // Spinner
    // ---------------------------------------------------------------------

    #[test]
    fn a_spinner_cycles_through_its_frames_and_wraps() {
        let mut spinner = Spinner::new("thinking", 0.0);
        assert_eq!(spinner.frame(), Some(SPINNER_PARTS[0]));
        for part in SPINNER_PARTS.iter().skip(1) {
            spinner.tick();
            assert_eq!(spinner.frame(), Some(*part));
        }
        spinner.tick();
        assert_eq!(spinner.frame(), Some(SPINNER_PARTS[0]), "wraps round");
    }

    #[test]
    fn a_stopped_spinner_shows_its_message_but_no_glyph() {
        let mut spinner = Spinner::new("verifying", 0.0);
        assert_eq!(spinner.line(DEFAULT_TERM_WIDTH), "verifying ⠋ ");
        spinner.stop(3.0);
        assert_eq!(spinner.line(DEFAULT_TERM_WIDTH), "verifying ");
        assert_eq!(spinner.frame(), None);
    }

    #[test]
    fn a_stopped_spinner_does_not_advance() {
        let mut spinner = Spinner::new("verifying", 0.0);
        spinner.stop(1.0);
        spinner.tick();
        assert_eq!(spinner.frame(), None);
    }

    #[test]
    fn stopping_a_spinner_twice_keeps_the_first_stop_time() {
        let mut spinner = Spinner::new("verifying", 0.0);
        spinner.stop(2.0);
        spinner.stop(9.0);
        assert_eq!(spinner.elapsed_secs(100.0), 2.0);
        assert!(spinner.is_stopped());
    }

    #[test]
    fn a_running_spinner_measures_elapsed_time_against_now() {
        let spinner = Spinner::new("thinking", 1.0);
        assert_eq!(spinner.elapsed_secs(4.5), 3.5);
    }

    #[test]
    fn a_stopped_spinner_with_no_message_renders_nothing_at_all() {
        let mut spinner = Spinner::new("", 0.0);
        spinner.stop(1.0);
        assert_eq!(spinner.line(DEFAULT_TERM_WIDTH), "");
    }

    #[test]
    fn a_spinner_message_can_be_replaced_while_it_spins() {
        let mut spinner = Spinner::new("pulling", 0.0);
        spinner.set_message("verifying sha256 digest");
        assert!(spinner.line(DEFAULT_TERM_WIDTH).starts_with("verifying sha256 digest ⠋"));
    }

    // ---------------------------------------------------------------------
    // StepBar
    // ---------------------------------------------------------------------

    #[test]
    fn a_step_bar_draws_one_cell_per_step() {
        let mut step = StepBar::new("Generating", 9);
        assert_eq!(step.line(DEFAULT_TERM_WIDTH), "Generating   0% ▕         ▏ 0/9");
        step.set(3);
        assert_eq!(step.line(DEFAULT_TERM_WIDTH), "Generating  33% ▕███      ▏ 3/9");
        step.set(9);
        assert_eq!(step.line(DEFAULT_TERM_WIDTH), "Generating 100% ▕█████████▏ 9/9");
    }

    /// Upstream would panic here (`strings.Repeat` with a negative count); we
    /// clamp. Stated as a divergence on [`StepBar::set`].
    #[test]
    fn overshooting_a_step_bar_clamps_instead_of_panicking() {
        let mut step = StepBar::new("Generating", 3);
        step.set(99);
        assert_eq!(step.current(), 3);
        assert_eq!(step.line(DEFAULT_TERM_WIDTH), "Generating 100% ▕███▏ 3/3");
    }

    /// Upstream would divide by zero and print "NaN%"; we report 0.
    #[test]
    fn a_step_bar_with_no_steps_reports_zero_rather_than_nan() {
        let step = StepBar::new("Generating", 0);
        assert_eq!(step.percent(), 0.0);
        assert_eq!(step.line(DEFAULT_TERM_WIDTH), "Generating   0% ▕▏ 0/0");
    }

    // ---------------------------------------------------------------------
    // Progress -- the stack and its visible window
    // ---------------------------------------------------------------------

    #[test]
    fn progress_renders_its_trackers_in_the_order_they_were_added() {
        let mut progress = Progress::new();
        progress.add("a", Tracker::Spinner(Spinner::new("first", 0.0)));
        progress.add("b", Tracker::Spinner(Spinner::new("second", 0.0)));

        let lines = progress.render_lines(DEFAULT_TERM_WIDTH, DEFAULT_TERM_HEIGHT);
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("first"));
        assert!(lines[1].starts_with("second"));
    }

    /// When there are more trackers than rows the OLDEST scroll off the top --
    /// during a pull the finished layers are the old ones.
    #[test]
    fn a_short_terminal_keeps_the_newest_trackers_and_drops_the_oldest() {
        let mut progress = Progress::new();
        for i in 0..10 {
            progress.add(
                format!("layer-{i}"),
                Tracker::Spinner(Spinner::new(format!("layer-{i}"), 0.0)),
            );
        }

        let visible = progress.visible(3);
        assert_eq!(visible.len(), 3);
        assert_eq!(visible[0].key, "layer-7");
        assert_eq!(visible[2].key, "layer-9");
    }

    #[test]
    fn a_zero_height_terminal_shows_nothing_rather_than_underflowing() {
        let mut progress = Progress::new();
        progress.add("a", Tracker::Spinner(Spinner::new("only", 0.0)));
        assert!(progress.visible(0).is_empty());
        assert!(progress.render_lines(DEFAULT_TERM_WIDTH, 0).is_empty());
    }

    #[test]
    fn an_empty_progress_renders_nothing() {
        let progress = Progress::new();
        assert!(progress.visible(DEFAULT_TERM_HEIGHT).is_empty());
    }

    /// Stopping stops every spinner but deliberately leaves bars alone -- a
    /// cancelled transfer must not report a completion rate.
    #[test]
    fn stopping_stops_the_spinners_and_leaves_the_bars_running() {
        let mut progress = Progress::new();
        progress.add("spin", Tracker::Spinner(Spinner::new("verifying", 0.0)));
        progress.add("bar", Tracker::Bar(Bar::new("pulling", 1000, 0, 0.0)));

        assert!(progress.stop(5.0));

        match &progress.entries()[0].tracker {
            Tracker::Spinner(s) => assert!(s.is_stopped()),
            other => panic!("expected a spinner, got {other:?}"),
        }
        match &progress.entries()[1].tracker {
            Tracker::Bar(b) => assert!(!b.is_finished(), "a half-done bar stays unfinished"),
            other => panic!("expected a bar, got {other:?}"),
        }
    }

    /// Upstream keys `stop()`'s return off `ticker != nil` so a second stop
    /// cannot print a stray newline or erase the same lines twice.
    #[test]
    fn stopping_twice_reports_false_the_second_time() {
        let mut progress = Progress::new();
        progress.add("spin", Tracker::Spinner(Spinner::new("verifying", 0.0)));
        assert!(progress.stop(1.0));
        assert!(!progress.stop(2.0));
        assert!(!progress.is_running());
    }

    /// The keyed lookup upstream throws away -- this is what a pull loop needs to
    /// route a chunk to the right layer's bar.
    #[test]
    fn a_tracker_can_be_looked_up_and_updated_by_key() {
        let mut progress = Progress::new();
        progress.add("sha256:abc", Tracker::Bar(Bar::new("layer", 1000, 0, 0.0)));

        match progress.get_mut("sha256:abc") {
            Some(Tracker::Bar(bar)) => bar.set(500, 1.0),
            other => panic!("expected a bar, got {other:?}"),
        }

        match &progress.entries()[0].tracker {
            Tracker::Bar(bar) => assert_eq!(bar.percent(), 50.0),
            other => panic!("expected a bar, got {other:?}"),
        }

        assert!(progress.get_mut("sha256:missing").is_none());
    }
}

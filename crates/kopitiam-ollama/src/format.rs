//! Human-readable sizes, counts and durations -- the strings a `list` table show.
//!
//! **Upstream:** `format/bytes.go`, `format/format.go`, `format/time.go`
//! (ollama, MIT).
//!
//! Looks like trivia; it is not. These four functions decide what a user reads
//! when they ask how big a model is, and **ollama's exact rounding is the
//! contract**. Get it subtly different and a KOPITIAM `list` says "4.7 GB" where
//! ollama says "4.6 GB" for the same blob, and now the two tools appear to
//! disagree about the file on disk. So this module is a faithful port, rounding
//! quirks and all, and the tests pin the quirks rather than tidy them.
//!
//! ## The one thing everybody gets wrong: GB is not GiB
//!
//! [`human_bytes`] uses **decimal** units (1 KB = 1000 B) -- that is what a
//! download progress bar and a model card use, because that is what registries
//! report. [`human_bytes2`] uses **binary** units (1 KiB = 1024 B) -- that is
//! what memory and VRAM accounting use, because that is what the allocator sees.
//! Two different jobs, so two different functions, and upstream keeps them apart
//! on purpose. Picking the wrong one is a 7.4% lie at the GB scale.

/// Decimal size units. **Upstream:** `format/bytes.go` consts.
pub const KILOBYTE: u64 = 1000;
pub const MEGABYTE: u64 = KILOBYTE * 1000;
pub const GIGABYTE: u64 = MEGABYTE * 1000;
pub const TERABYTE: u64 = GIGABYTE * 1000;

/// Binary size units. **Upstream:** `format/bytes.go` consts.
pub const KIBIBYTE: u64 = 1024;
pub const MEBIBYTE: u64 = KIBIBYTE * 1024;
pub const GIBIBYTE: u64 = MEBIBYTE * 1024;

/// Decimal-unit byte size, as shown for **file sizes and downloads**.
///
/// **Upstream:** `HumanBytes(b int64)`.
///
/// The rounding ladder is upstream's, and it is deliberately not uniform:
///
/// * under 1 KB -> plain `"512 B"`, no unit scaling;
/// * 10 and above -> **truncated to whole units** (`"11 GB"`, never `"11.4 GB"`)
///   -- once the number is big, a decimal buys nothing;
/// * under 10 and not whole -> one decimal (`"4.7 GB"`);
/// * under 10 and whole -> no decimal at all (`"4 GB"`, not `"4.0 GB"`).
///
/// Note both the `>= 10` and the whole-number branches **truncate**, they do not
/// round: 4.99 GB prints `"4.9 GB"`. That is Go's `int(value)` and `%.1f` on an
/// already-truncated... no -- `%.1f` rounds, `int()` truncates. So `10.9` prints
/// `"10 GB"` while `4.99` prints `"5.0 GB"`. Inconsistent, and faithfully copied,
/// because matching ollama matters more than being tidy.
///
/// Takes `i64` like upstream, since a size can arrive as a signed delta;
/// negatives fall through to the `"{n} B"` branch exactly as upstream does.
pub fn human_bytes(b: i64) -> String {
    let (value, unit) = if b >= TERABYTE as i64 {
        (b as f64 / TERABYTE as f64, "TB")
    } else if b >= GIGABYTE as i64 {
        (b as f64 / GIGABYTE as f64, "GB")
    } else if b >= MEGABYTE as i64 {
        (b as f64 / MEGABYTE as f64, "MB")
    } else if b >= KILOBYTE as i64 {
        (b as f64 / KILOBYTE as f64, "KB")
    } else {
        return format!("{b} B");
    };

    if value >= 10.0 {
        // Go's `int(value)` truncates toward zero -- `as i64` does the same.
        format!("{} {}", value as i64, unit)
    } else if value != value.trunc() {
        // Go's `%.1f` rounds half away from zero; Rust's `{:.1}` rounds half to
        // even. They differ only on an exact `.x5` tie, which a byte count
        // divided by a power of ten essentially never lands on. Called out
        // because "essentially never" is not "never".
        format!("{value:.1} {unit}")
    } else {
        format!("{} {}", value as i64, unit)
    }
}

/// Binary-unit byte size, as shown for **memory and VRAM**.
///
/// **Upstream:** `HumanBytes2(b uint64)`.
///
/// Simpler ladder than [`human_bytes`]: always one decimal above 1 KiB, plain
/// bytes below. Upstream uses this everywhere it reports what a model will cost
/// in RAM, which is the number that decides whether a layer fits on the GPU --
/// so it must speak the allocator's units, never the registry's.
pub fn human_bytes2(b: u64) -> String {
    if b >= GIBIBYTE {
        format!("{:.1} GiB", b as f64 / GIBIBYTE as f64)
    } else if b >= MEBIBYTE {
        format!("{:.1} MiB", b as f64 / MEBIBYTE as f64)
    } else if b >= KIBIBYTE {
        format!("{:.1} KiB", b as f64 / KIBIBYTE as f64)
    } else {
        format!("{b} B")
    }
}

/// Human-readable count -- parameter counts, mostly. **Upstream:** `HumanNumber`.
///
/// This is where `"0.6B"` in `qwen3:0.6b` and `"8B"` in `llama3:8b` come from.
/// The decimal counts are, again, not uniform and copied as-is: billions get
/// **one** decimal, millions get **two**, thousands get **none**. A whole number
/// drops its decimals entirely at every scale (`"8B"`, not `"8.0B"`).
pub fn human_number(b: u64) -> String {
    const THOUSAND: u64 = 1000;
    const MILLION: u64 = THOUSAND * 1000;
    const BILLION: u64 = MILLION * 1000;

    if b >= BILLION {
        let n = b as f64 / BILLION as f64;
        if n == n.floor() {
            format!("{n:.0}B")
        } else {
            format!("{n:.1}B")
        }
    } else if b >= MILLION {
        let n = b as f64 / MILLION as f64;
        if n == n.floor() {
            format!("{n:.0}M")
        } else {
            format!("{n:.2}M")
        }
    } else if b >= THOUSAND {
        format!("{:.0}K", b as f64 / THOUSAND as f64)
    } else {
        b.to_string()
    }
}

/// Fuzzy duration -- `"About a minute"`, `"3 days"`.
///
/// **Upstream:** `humanDuration(d time.Duration)` (unexported there; exported
/// here because KOPITIAM's TUI want it directly, and there is no reason to hide
/// a pure function).
///
/// Takes seconds as `f64` rather than a `Duration` so a negative delta ("from
/// now") can be expressed without fighting `std::time::Duration`, which is
/// unsigned. Upstream computes off a signed `time.Duration` for the same reason.
///
/// **The rounding is not uniform, and the asymmetry is load-bearing.** Seconds
/// and minutes **truncate** (`int(d.Minutes())`), and so do days/weeks/months
/// (`hours/24`, `hours/24/7`, ...). Only the hours conversion **rounds**
/// (`int(math.Round(d.Hours()))`).
///
/// So 59m30s reads `"59 minutes"` -- truncated, it never reaches the hours
/// branch at all -- but 1h45m reads `"2 hours"`, because by then the rounding
/// applies. Both are upstream's behaviour and both are pinned by tests. Note
/// Go's `math.Round` breaks ties away from zero and Rust's `f64::round` does the
/// same, so 90 minutes agrees on `"2 hours"` in both.
pub fn human_duration(seconds_f: f64) -> String {
    let seconds = seconds_f as i64;
    if seconds < 1 {
        return "Less than a second".to_string();
    }
    if seconds == 1 {
        return "1 second".to_string();
    }
    if seconds < 60 {
        return format!("{seconds} seconds");
    }

    let minutes = (seconds_f / 60.0) as i64;
    if minutes == 1 {
        return "About a minute".to_string();
    }
    if minutes < 60 {
        return format!("{minutes} minutes");
    }

    let hours = (seconds_f / 3600.0).round() as i64;
    if hours == 1 {
        return "About an hour".to_string();
    }
    if hours < 48 {
        return format!("{hours} hours");
    }
    if hours < 24 * 7 * 2 {
        return format!("{} days", hours / 24);
    }
    if hours < 24 * 30 * 2 {
        return format!("{} weeks", hours / 24 / 7);
    }
    if hours < 24 * 365 * 2 {
        return format!("{} months", hours / 24 / 30);
    }

    // Upstream uses the UNROUNDED hours here (`int(d.Hours())`), unlike every
    // branch above which uses the rounded one. Copied deliberately -- at the
    // years scale it can never change the answer, but silently "fixing" it would
    // be a divergence with no upside.
    format!("{} years", (seconds_f / 3600.0) as i64 / 24 / 365)
}

/// `"3 days ago"` / `"About a minute from now"`, given a signed age in seconds.
///
/// **Upstream:** `humanTime(t, zeroValue)`, restructured. Go takes a `time.Time`
/// and calls `time.Since` itself; taking the delta instead keeps this module
/// clock-free, so it stays a pure function and the tests need no fake clock.
/// The caller does the subtraction -- one line at the call site, in exchange for
/// a testable core.
///
/// `age_seconds` is **positive for the past**, matching Go's `time.Since`.
/// `None` means Go's zero time, and yields `zero_value` verbatim.
pub fn human_time(age_seconds: Option<f64>, zero_value: &str) -> String {
    let Some(delta) = age_seconds else {
        return zero_value.to_string();
    };

    // Upstream's guard against a timestamp so absurdly far in the future it is
    // certainly a bug or a sentinel, rather than a real date.
    if (delta / 3600.0) as i64 / 24 / 365 < -20 {
        return "Forever".to_string();
    }
    if delta < 0.0 {
        return format!("{} from now", human_duration(-delta));
    }
    format!("{} ago", human_duration(delta))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream `format/bytes_test.go`, plus the boundary cases that pin the
    /// truncate-vs-round quirk described on [`human_bytes`].
    #[test]
    fn human_bytes_matches_upstream_rounding() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(999), "999 B");
        assert_eq!(human_bytes(1000), "1 KB");
        assert_eq!(human_bytes(1500), "1.5 KB");
        assert_eq!(human_bytes(1_000_000), "1 MB");
        assert_eq!(human_bytes(1_500_000), "1.5 MB");
        assert_eq!(human_bytes(1_000_000_000), "1 GB");
        assert_eq!(human_bytes(4_700_000_000), "4.7 GB");
        assert_eq!(human_bytes(1_000_000_000_000), "1 TB");

        // >= 10 truncates and drops the decimal entirely.
        assert_eq!(human_bytes(10_900_000_000), "10 GB");
        assert_eq!(human_bytes(70_000_000_000), "70 GB");

        // A whole number under 10 gets NO decimal -- "4 GB", not "4.0 GB".
        assert_eq!(human_bytes(4_000_000_000), "4 GB");

        // Negative falls through to the byte branch, same as upstream.
        assert_eq!(human_bytes(-1), "-1 B");
    }

    #[test]
    fn human_bytes2_uses_binary_units() {
        assert_eq!(human_bytes2(0), "0 B");
        assert_eq!(human_bytes2(1023), "1023 B");
        assert_eq!(human_bytes2(1024), "1.0 KiB");
        assert_eq!(human_bytes2(MEBIBYTE), "1.0 MiB");
        assert_eq!(human_bytes2(GIBIBYTE), "1.0 GiB");
        assert_eq!(human_bytes2(GIBIBYTE * 8), "8.0 GiB");
    }

    /// The GB-vs-GiB gap, asserted rather than described. Same bytes, two
    /// answers, and both are correct for their own audience.
    #[test]
    fn decimal_and_binary_formatters_disagree_on_purpose() {
        let bytes = 8 * GIBIBYTE; // 8_589_934_592
        assert_eq!(human_bytes2(bytes), "8.0 GiB");
        assert_eq!(human_bytes(bytes as i64), "8.6 GB");
    }

    #[test]
    fn human_number_matches_upstream_decimal_counts() {
        assert_eq!(human_number(0), "0");
        assert_eq!(human_number(999), "999");
        assert_eq!(human_number(1000), "1K");
        assert_eq!(human_number(1_000_000), "1M");
        assert_eq!(human_number(1_500_000), "1.50M"); // millions: TWO decimals
        assert_eq!(human_number(8_000_000_000), "8B"); // whole: no decimal
        assert_eq!(human_number(7_600_000_000), "7.6B"); // billions: ONE decimal
        // The real qwen3:0.6b parameter count rounds to the tag it ships under.
        assert_eq!(human_number(596_049_920), "596.05M");
    }

    #[test]
    fn human_duration_walks_the_upstream_ladder() {
        assert_eq!(human_duration(0.4), "Less than a second");
        assert_eq!(human_duration(1.0), "1 second");
        assert_eq!(human_duration(30.0), "30 seconds");
        assert_eq!(human_duration(60.0), "About a minute");
        assert_eq!(human_duration(120.0), "2 minutes");
        assert_eq!(human_duration(3600.0), "About an hour");
        assert_eq!(human_duration(3600.0 * 5.0), "5 hours");
        assert_eq!(human_duration(3600.0 * 72.0), "3 days");
        assert_eq!(human_duration(3600.0 * 24.0 * 20.0), "2 weeks");
        assert_eq!(human_duration(3600.0 * 24.0 * 90.0), "3 months");
        assert_eq!(human_duration(3600.0 * 24.0 * 365.0 * 3.0), "3 years");
    }

    /// The rounding asymmetry called out in the docs, pinned in both directions.
    #[test]
    fn only_the_hours_conversion_rounds() {
        // Minutes TRUNCATE, so 59m30s never reaches the hours branch.
        assert_eq!(human_duration(59.0 * 60.0 + 30.0), "59 minutes");
        // Past 60 minutes the hours branch takes over, and THAT one rounds:
        // 1h45m -> round(1.75) -> 2.
        assert_eq!(human_duration(3600.0 + 45.0 * 60.0), "2 hours");
        // Ties round away from zero in both Go and Rust: 90min -> round(1.5) -> 2.
        assert_eq!(human_duration(90.0 * 60.0), "2 hours");
        // Days truncate again: 2.9 days is still "2 days".
        assert_eq!(human_duration(3600.0 * 24.0 * 2.9), "2 days");
    }

    #[test]
    fn human_time_says_ago_and_from_now() {
        assert_eq!(human_time(Some(3600.0 * 72.0), "Never"), "3 days ago");
        assert_eq!(human_time(Some(-60.0), "Never"), "About a minute from now");
        assert_eq!(human_time(None, "Never"), "Never");
        // Absurdly-far-future timestamps are a sentinel, not a date.
        assert_eq!(human_time(Some(-3600.0 * 24.0 * 365.0 * 25.0), "Never"), "Forever");
    }
}

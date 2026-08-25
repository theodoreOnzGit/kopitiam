use crate::api::{QueryResult, StatusOutput, SyncWarning};
use crate::surface::ipc::{EmptyPayload, Request, ResponsePayload};

use super::common::fmt_wall_ms;
use super::{CommandResult, print_ok};
use crate::cli_surface::render::print_line;
use crate::cli_surface::runtime::{CliRuntimeCtx, send};

pub fn handle(ctx: &CliRuntimeCtx) -> CommandResult<()> {
    let req = Request::Status {
        ctx: ctx.read_ctx(),
        payload: EmptyPayload {},
    };
    let ok = send(&req)?;
    if !ctx.json
        && let ResponsePayload::Query(QueryResult::Status(out)) = &ok
    {
        print_line(&render_status(out))?;
        return Ok(());
    }
    print_ok(&ok, ctx.json)
}

pub fn render_status(out: &StatusOutput) -> String {
    let summary = &out.summary;
    let mut buf = String::new();
    buf.push_str("\nIssue Database Status\n=====================\n\nSummary:\n");
    buf.push_str(&format!("  Total Issues:      {}\n", summary.total_issues));
    buf.push_str(&format!("  Open:              {}\n", summary.open_issues));
    buf.push_str(&format!(
        "  In Progress:       {}\n",
        summary.in_progress_issues
    ));
    buf.push_str(&format!(
        "  Blocked:           {}\n",
        summary.blocked_issues
    ));
    buf.push_str(&format!("  Closed:            {}\n", summary.closed_issues));
    buf.push_str(&format!("  Ready to Work:     {}\n", summary.ready_issues));
    if let Some(tombstones) = summary.tombstone_issues
        && tombstones > 0
    {
        buf.push_str(&format!(
            "  Deleted:           {} (tombstones)\n",
            tombstones
        ));
    }
    if let Some(epics) = summary.epics_eligible_for_closure
        && epics > 0
    {
        buf.push_str(&format!("  Epics Ready to Close: {}\n", epics));
    }
    if let Some(sync) = &out.sync {
        let last_sync = sync
            .last_sync_wall_ms
            .map(fmt_wall_ms)
            .unwrap_or_else(|| "never".to_string());
        buf.push_str("\nSync:\n");
        buf.push_str(&format!("  dirty:             {}\n", sync.dirty));
        buf.push_str(&format!("  in_progress:       {}\n", sync.sync_in_progress));
        buf.push_str(&format!("  last_sync:         {}\n", last_sync));
        if let Some(next_retry) = sync.next_retry_wall_ms {
            let mut line = format!("  next_retry:       {}", fmt_wall_ms(next_retry));
            if let Some(in_ms) = sync.next_retry_in_ms {
                line.push_str(&format!(" (in {})", fmt_duration_ms(in_ms)));
            }
            line.push('\n');
            buf.push_str(&line);
        }
        buf.push_str(&format!(
            "  consecutive_failures: {}\n",
            sync.consecutive_failures
        ));
        if !sync.warnings.is_empty() {
            buf.push_str("  warnings:\n");
            for warning in &sync.warnings {
                match warning {
                    SyncWarning::Fetch {
                        message,
                        at_wall_ms,
                    } => {
                        buf.push_str(&format!(
                            "    fetch_error: {} (at {})\n",
                            message,
                            fmt_wall_ms(*at_wall_ms)
                        ));
                    }
                    SyncWarning::Diverged {
                        local_oid,
                        remote_oid,
                        at_wall_ms,
                    } => {
                        buf.push_str(&format!(
                            "    divergence: local {} remote {} (at {})\n",
                            local_oid,
                            remote_oid,
                            fmt_wall_ms(*at_wall_ms)
                        ));
                    }
                    SyncWarning::ForcePush {
                        previous_remote_oid,
                        remote_oid,
                        at_wall_ms,
                    } => {
                        buf.push_str(&format!(
                            "    force_push: {} -> {} (at {})\n",
                            previous_remote_oid,
                            remote_oid,
                            fmt_wall_ms(*at_wall_ms)
                        ));
                    }
                    SyncWarning::ClockSkew {
                        delta_ms,
                        at_wall_ms,
                    } => {
                        buf.push_str(&render_clock_skew(*delta_ms, *at_wall_ms));
                    }
                    SyncWarning::WalTailTruncated {
                        namespace,
                        segment_id,
                        truncated_from_offset,
                        at_wall_ms,
                    } => {
                        let segment = segment_id
                            .map(|id| id.to_string())
                            .unwrap_or_else(|| "unknown".to_string());
                        buf.push_str(&format!(
                            "    wal_tail_truncated: {} segment {} offset {} (at {})\n",
                            namespace,
                            segment,
                            truncated_from_offset,
                            fmt_wall_ms(*at_wall_ms)
                        ));
                    }
                }
            }
        }
    }
    buf.push('\n');
    buf
}

/// Render the `clock_skew` warning so that somebody staring at it at 3am can
/// actually tell a `bn` bug from a broken host clock.
///
/// The old line was `clock_skew: 3657601 ms ahead (at 2026-08-12 04:07)`, which
/// answers nothing: ahead of *what*, is that a lot, and must I do something?
/// Three separate people cannot read that the same way, so now we spell out all
/// four things — both comparands by name, the direction, the size in units a
/// human uses, and what it actually breaks.
///
/// # How we get the reference timestamp back
///
/// The IPC payload (`SyncWarning::ClockSkew`) only carries `delta_ms` and
/// `at_wall_ms`. But `detect_clock_skew()` (see
/// `daemon/runtime/core/helpers.rs`) builds the record as
/// `delta_ms = now_ms - reference_ms` with `wall_ms = now_ms`, both plain Unix
/// milliseconds. So `reference_ms = at_wall_ms - delta_ms` recovers the other
/// comparand **exactly** — no estimate, no rounding. If anybody ever change what
/// `wall_ms` means over there, this reconstruction go wrong silently, so the two
/// places are cross-referenced in each other's doc comments.
///
/// # What we deliberately do NOT claim
///
/// We don't name an actor. A `WriteStamp` has no actor id attached, and the
/// `max_write_stamp()` that produces the reference throws away the `by` field.
/// Saying "actor X's last write" would be an invented fact — worse than vague.
/// See the report on issue #21: carrying a real source/actor label needs a new
/// field on `SyncWarning::ClockSkew` in `api/issues.rs`.
///
/// # The asymmetry, which is the whole point
///
/// **Positive** (local clock ahead) is usually *not* a clock fault at all — it
/// is just the age of the newest write, so an idle store trips the 5-minute
/// threshold on a perfectly good clock, and it clears itself the moment anybody
/// writes. **Negative** (local clock behind) cannot be explained that way: a
/// stamp dated in the future means somebody's clock really is wrong. Two very
/// different situations, so we print two different explanations.
fn render_clock_skew(delta_ms: i64, at_wall_ms: u64) -> String {
    // Exact inverse of `delta_ms = now_ms - reference_ms`; saturating only so a
    // corrupt payload cannot panic us in a status printer.
    let reference_ms = (at_wall_ms as i64).saturating_sub(delta_ms).max(0) as u64;
    let abs_ms = delta_ms.unsigned_abs();
    let magnitude = fmt_skew_duration_ms(abs_ms);
    let direction = if delta_ms >= 0 { "ahead of" } else { "behind" };

    let mut out = String::new();
    out.push_str(&format!(
        "    clock_skew: local system clock is {magnitude} ({abs_ms} ms) {direction} the newest write stamp bn has seen for this store\n"
    ));
    out.push_str(&format!(
        "      local system clock when checked: {}\n",
        fmt_wall_ms_utc_secs(at_wall_ms)
    ));
    out.push_str(&format!(
        "      newest write stamp:              {}\n",
        fmt_wall_ms_utc_secs(reference_ms)
    ));
    if delta_ms >= 0 {
        out.push_str(
            "      note: 'ahead' is usually NOT a clock fault. An idle store reads the same way, because this figure is also just the age of the newest write.\n",
        );
        out.push_str(
            "      it clears by itself as soon as any write lands. No restart, no clock change, nothing for you to do.\n",
        );
    } else {
        out.push_str(
            "      note: 'behind' cannot be explained by an idle store. A write stamp dated in the future means either this host's clock is slow, or the writer's clock is fast.\n",
        );
        out.push_str(
            "      check the host clock first (`timedatectl` / `date -u`); it will not clear on its own if the clock really is wrong.\n",
        );
    }
    out.push_str(
        "      effect: merges resolve last-write-wins on these stamps, so a wrong clock makes this host's edits beat or lose to edits made elsewhere.\n",
    );
    out
}

/// Unix ms -> `YYYY-MM-DD HH:MM:SS UTC`.
///
/// Deliberately not [`fmt_wall_ms`]: that one stops at minutes and prints no
/// timezone, which is fine for "last_sync" but useless when the user must line
/// the value up against `date -u` to judge their own clock. Seconds and an
/// explicit `UTC` are the whole diagnostic here, so we spell them out.
fn fmt_wall_ms_utc_secs(ms: u64) -> String {
    use std::sync::LazyLock;
    use time::OffsetDateTime;
    use time::format_description::BorrowedFormatItem;

    // `parse_borrowed::<2>` over the deprecated `parse`: the literal is
    // `'static`, so the borrowed items are `'static` too, and v2 of the format
    // description is the one that is not going anywhere.
    static FORMAT: LazyLock<Option<Vec<BorrowedFormatItem<'static>>>> = LazyLock::new(|| {
        time::format_description::parse_borrowed::<2>(
            "[year]-[month]-[day] [hour]:[minute]:[second] UTC",
        )
        .ok()
    });

    let Ok(dt) = OffsetDateTime::from_unix_timestamp_nanos(ms as i128 * 1_000_000) else {
        return format!("{ms} (unix ms)");
    };
    match FORMAT.as_deref() {
        Some(fmt) => dt.format(fmt).unwrap_or_else(|_| format!("{ms} (unix ms)")),
        None => format!("{ms} (unix ms)"),
    }
}

/// Milliseconds -> something a human reads without counting digits.
///
/// `3657601` is unreadable; `1h 0m 57s` is not. We keep the raw millisecond
/// count at the call site as well, because bug reports need the exact number —
/// human-readable is for the human, exact is for us.
///
/// Units cascade: we drop leading units that are zero, but never interior ones
/// (`1h 0m 57s`, not `1h 57s`, which reads like 1h57m at a glance).
fn fmt_skew_duration_ms(ms: u64) -> String {
    let total_secs = ms / 1000;
    let millis = ms % 1000;
    let days = total_secs / 86_400;
    let hours = (total_secs % 86_400) / 3_600;
    let mins = (total_secs % 3_600) / 60;
    let secs = total_secs % 60;

    if days > 0 {
        format!("{days}d {hours}h {mins}m {secs}s")
    } else if hours > 0 {
        format!("{hours}h {mins}m {secs}s")
    } else if mins > 0 {
        format!("{mins}m {secs}s")
    } else if secs > 0 {
        format!("{secs}s")
    } else {
        format!("{millis}ms")
    }
}

fn fmt_duration_ms(ms: u64) -> String {
    if ms < 1000 {
        return format!("{ms}ms");
    }
    let secs = ms as f64 / 1000.0;
    if secs < 60.0 {
        return format!("{secs:.1}s");
    }
    let mins = secs / 60.0;
    if mins < 60.0 {
        return format!("{mins:.1}m");
    }
    let hours = mins / 60.0;
    format!("{hours:.1}h")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact numbers off GitHub issue #21: `3657601 ms ahead (at 2026-08-12
    /// 04:07)`. Locking the real reported case in so nobody quietly regress it
    /// back to raw milliseconds.
    const ISSUE_21_DELTA_MS: i64 = 3_657_601;
    /// 2026-08-12T04:07:41Z in Unix ms — the `at` from that same report.
    const ISSUE_21_AT_WALL_MS: u64 = 1_786_507_661_000;

    #[test]
    fn skew_duration_is_human_readable_not_raw_ms() {
        assert_eq!(fmt_skew_duration_ms(3_657_601), "1h 0m 57s");
        assert_eq!(fmt_skew_duration_ms(300_000), "5m 0s");
        assert_eq!(fmt_skew_duration_ms(59_999), "59s");
        assert_eq!(fmt_skew_duration_ms(999), "999ms");
        assert_eq!(fmt_skew_duration_ms(0), "0ms");
        assert_eq!(fmt_skew_duration_ms(90_061_000), "1d 1h 1m 1s");
    }

    #[test]
    fn clock_skew_positive_delta_says_local_clock_is_ahead() {
        let out = render_clock_skew(ISSUE_21_DELTA_MS, ISSUE_21_AT_WALL_MS);
        assert!(
            out.contains("local system clock is 1h 0m 57s (3657601 ms) ahead of the newest write stamp bn has seen for this store"),
            "unexpected headline:\n{out}"
        );
        assert!(!out.contains("behind"), "wrong direction:\n{out}");
    }

    #[test]
    fn clock_skew_negative_delta_says_local_clock_is_behind() {
        let out = render_clock_skew(-ISSUE_21_DELTA_MS, ISSUE_21_AT_WALL_MS);
        assert!(
            out.contains(
                "local system clock is 1h 0m 57s (3657601 ms) behind the newest write stamp"
            ),
            "unexpected headline:\n{out}"
        );
        assert!(!out.contains("ahead"), "wrong direction:\n{out}");
    }

    /// Direction must flip on the sign and nothing else. A `delta_ms` of exactly
    /// zero never reaches the renderer in practice (`detect_clock_skew` would
    /// have returned `None`), but the sign test must still be total.
    #[test]
    fn clock_skew_direction_follows_sign_only() {
        assert!(render_clock_skew(1, 1_000_000).contains(" ahead of "));
        assert!(render_clock_skew(0, 1_000_000).contains(" ahead of "));
        assert!(render_clock_skew(-1, 1_000_000).contains(" behind "));
    }

    /// Both comparands must be named AND dated. The reference timestamp is
    /// reconstructed as `at_wall_ms - delta_ms`; if that arithmetic ever drift
    /// out of step with `detect_clock_skew`, this test is what catches it.
    #[test]
    fn clock_skew_names_and_dates_both_comparands() {
        let out = render_clock_skew(ISSUE_21_DELTA_MS, ISSUE_21_AT_WALL_MS);
        assert!(
            out.contains("local system clock when checked: 2026-08-12 04:07:41 UTC"),
            "local comparand missing or wrong:\n{out}"
        );
        // 1786507661000 - 3657601 = 1786504003399 -> 2026-08-12T03:06:43.399Z
        assert!(
            out.contains("newest write stamp:              2026-08-12 03:06:43 UTC"),
            "reference comparand missing or wrong:\n{out}"
        );
    }

    /// The self-clearing behaviour is the bit that cost the reporter a night, so
    /// the positive branch must say it outright, and the negative branch must
    /// NOT (a real skew does not clear itself).
    #[test]
    fn clock_skew_explains_self_clearing_only_when_it_really_self_clears() {
        let ahead = render_clock_skew(ISSUE_21_DELTA_MS, ISSUE_21_AT_WALL_MS);
        assert!(ahead.contains("clears by itself"), "{ahead}");
        assert!(ahead.contains("age of the newest write"), "{ahead}");

        let behind = render_clock_skew(-ISSUE_21_DELTA_MS, ISSUE_21_AT_WALL_MS);
        assert!(!behind.contains("clears by itself"), "{behind}");
        assert!(
            behind.contains("will not clear on its own"),
            "negative branch must not promise a self-heal:\n{behind}"
        );
    }

    /// "may affect ordering" from the issue, said concretely: which mechanism,
    /// and which way it goes wrong.
    #[test]
    fn clock_skew_states_the_consequence() {
        for delta in [ISSUE_21_DELTA_MS, -ISSUE_21_DELTA_MS] {
            let out = render_clock_skew(delta, ISSUE_21_AT_WALL_MS);
            assert!(
                out.contains("merges resolve last-write-wins on these stamps"),
                "consequence missing:\n{out}"
            );
        }
    }

    /// Raw milliseconds must survive somewhere in the line: the human needs the
    /// pretty form, we need the exact one when they paste it into a bug report.
    #[test]
    fn clock_skew_keeps_the_exact_millisecond_count() {
        let out = render_clock_skew(ISSUE_21_DELTA_MS, ISSUE_21_AT_WALL_MS);
        assert!(out.contains("(3657601 ms)"), "{out}");
    }

    /// A status printer must never panic, however rubbish the payload. `i64::MIN`
    /// has no positive counterpart, and a reference before the epoch is nonsense
    /// but must still render.
    #[test]
    fn clock_skew_survives_absurd_payloads() {
        let _ = render_clock_skew(i64::MIN, 0);
        let _ = render_clock_skew(i64::MAX, u64::MAX);
        let _ = render_clock_skew(-1, u64::MAX);
    }

    #[test]
    fn status_renders_the_clock_skew_warning_inline() {
        use crate::api::{StatusOutput, StatusSummary, SyncStatus};

        let out = StatusOutput {
            summary: StatusSummary {
                total_issues: 1,
                open_issues: 1,
                in_progress_issues: 0,
                blocked_issues: 0,
                closed_issues: 0,
                ready_issues: 1,
                tombstone_issues: None,
                epics_eligible_for_closure: None,
            },
            sync: Some(SyncStatus {
                dirty: false,
                sync_in_progress: false,
                last_sync_wall_ms: None,
                next_retry_wall_ms: None,
                next_retry_in_ms: None,
                consecutive_failures: 0,
                warnings: vec![SyncWarning::ClockSkew {
                    delta_ms: ISSUE_21_DELTA_MS,
                    at_wall_ms: ISSUE_21_AT_WALL_MS,
                }],
            }),
        };

        let rendered = render_status(&out);
        assert!(rendered.contains("  warnings:\n"), "{rendered}");
        assert!(
            rendered.contains("    clock_skew: local system clock is 1h 0m 57s"),
            "{rendered}"
        );
        // The old unreadable shape must be gone for good.
        assert!(
            !rendered.contains("clock_skew: 3657601 ms ahead"),
            "old message came back:\n{rendered}"
        );
    }
}

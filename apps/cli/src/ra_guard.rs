//! Resource gate for the rust-analyzer path: *can this box afford to start RA
//! right now?*
//!
//! # Why this exists
//!
//! `outline`/`refs`/`def` want rust-analyzer for real cross-file answers, but RA
//! indexing a big workspace is exactly the job that gets a process **SIGKILL**ed
//! by Android's low-memory killer — and you cannot `?` your way out of that,
//! because the kernel already shot you before your code runs again. On a 4 GB
//! Termux tablet the same 46-crate workspace that a desktop indexes happily will
//! either OOM or grind for minutes while the CLI looks frozen.
//!
//! So the gate is **preemptive**: estimate the cost BEFORE spawning RA, and if
//! it doesn't fit, stand down to the instant syntactic scan. The freeze never
//! happens because the heavy thing never launches.
//!
//! # What this module is (and is not)
//!
//! It is a **thin adapter**, not new arithmetic. All the actual budgeting lives
//! in `kopitiam-resource` — the shared budgeter ratified as **AID-0037**, which
//! kvim already uses for the same decision. This module only:
//!
//! 1. takes a device reading + a project weight,
//! 2. asks [`should_run_rust_analyzer`],
//! 3. turns the [`Verdict`] into a yes/no the CLI can act on.
//!
//! Deliberately no second copy of the cost model: one budgeter, two clients is
//! the whole point of that crate. If the thresholds are wrong, fix them there
//! and both kvim and the CLI improve together.
//!
//! # Why this is not the same as the existing file-count check
//!
//! `syntactic::prefer_syntactic_default` counts `.rs` files. That's a *static*
//! property of the repo — it says the same thing on a 32 GB desktop and a 4 GB
//! tablet, even though only one of them is in trouble. This gate reads the
//! **actual free RAM and core count at this moment**, which is the axis that
//! decides whether RA survives. The two are kept side by side on purpose: this
//! one catches "big project, small box", the file count still catches "project
//! so big nobody indexes it in bounded time".
//!
//! # Fail-open, always
//!
//! If we cannot get an honest reading — no `sysinfo` memory backend, no
//! `Cargo.lock` to weigh the project — we **allow** rust-analyzer. A capable
//! desktop must never be blocked by a missing measurement; the cost of a wrong
//! "allow" is a slow command, the cost of a wrong "refuse" is silently
//! downgrading everyone's answers forever. `Reason::NotApplicable` carries that
//! same meaning from the budgeter and is honoured here.

use std::path::Path;

use kopitiam_resource::clients::{should_run_rust_analyzer, BudgetInputs, RaCoeffs};
use kopitiam_resource::{estimate_project_weight, Capacity, DeviceProbe, Reason, SysinfoProbe, Verdict};

/// Escape hatch env var. `off` disables the gate entirely (always try RA);
/// `syntactic` forces the stand-down (useful to reproduce tablet behaviour on a
/// desktop without actually filling the RAM). Anything else is ignored and the
/// real budget decides.
pub const GUARD_ENV: &str = "KOPITIAM_RA_GUARD";

/// What the gate decided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RaGate {
    /// Got room — go ahead and spawn rust-analyzer.
    Run,
    /// Stand down to the syntactic scan. Carries a ready-to-print reason so the
    /// caller can tell the user *why* the answer is the cheaper one, instead of
    /// silently handing back a thinner outline and looking like a bug.
    StandDown(String),
}

/// Decides whether rust-analyzer may run for the project rooted at `root`.
///
/// Never panics, never blocks on anything slow: the probe is a single cheap
/// `sysinfo` read and the project weight is stat-only (no file contents). That
/// matters — a gate that is itself expensive would just move the freeze.
pub fn decide(root: &Path) -> RaGate {
    match std::env::var(GUARD_ENV).ok().as_deref() {
        Some("off") => return RaGate::Run,
        Some("syntactic") => {
            return RaGate::StandDown(format!("{GUARD_ENV}=syntactic forced the syntactic path"));
        }
        _ => {}
    }

    // Both of these returning `None` means "cannot measure", which is fail-open
    // (see the module docs) — not a refusal.
    let Some(cap) = SysinfoProbe.snapshot() else {
        return RaGate::Run;
    };
    let Some(weight) = estimate_project_weight(root) else {
        return RaGate::Run;
    };

    let verdict =
        should_run_rust_analyzer(cap, weight, RaCoeffs::default(), BudgetInputs::default());
    interpret(verdict, cap)
}

/// Turns a budgeter [`Verdict`] into an [`RaGate`], given the capacity reading
/// it was based on (used only to make the message concrete).
///
/// Split out from [`decide`] as a pure function so the mapping is unit-tested
/// without needing a real device in a particular memory state — the one part of
/// this module where a wrong branch would silently degrade every answer.
fn interpret(verdict: Verdict, cap: Capacity) -> RaGate {
    match verdict.reason() {
        // `Fits`.
        None => RaGate::Run,
        // The budgeter could not judge (no lock file, bogus probe). Its own docs
        // are explicit that this means carry on as if unguarded, so a desktop is
        // never blocked by an absent reading.
        Some(Reason::NotApplicable) => RaGate::Run,
        // Everything else — too little RAM, too few cores, project too heavy —
        // is a real "would have hurt", so take the instant path instead.
        Some(reason) => RaGate::StandDown(format!(
            "{} ({} MB free of {} MB, {} cores)",
            reason.blurb(),
            cap.avail_mb,
            cap.total_mb,
            cap.logical_cores
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cap(avail_mb: u64, cores: usize) -> Capacity {
        Capacity { avail_mb, total_mb: 8192, logical_cores: cores, cpu_usage: 0.0 }
    }

    #[test]
    fn fits_runs_rust_analyzer() {
        assert_eq!(interpret(Verdict::Fits, cap(8000, 8)), RaGate::Run);
    }

    #[test]
    fn not_applicable_fails_open_rather_than_blocking_a_capable_box() {
        // The budgeter documents NotApplicable as fail-open. Getting this branch
        // backwards would silently downgrade every answer on any machine without
        // a Cargo.lock or a readable memory backend — the worst kind of bug,
        // because the CLI would still "work", just worse, forever.
        assert_eq!(interpret(Verdict::Degrade(Reason::NotApplicable), cap(8000, 8)), RaGate::Run);
        assert_eq!(interpret(Verdict::Refuse(Reason::NotApplicable), cap(512, 2)), RaGate::Run);
    }

    #[test]
    fn a_real_refusal_stands_down_and_says_why() {
        let gate = interpret(Verdict::Refuse(Reason::ProjectTooLarge), cap(512, 2));
        match gate {
            RaGate::StandDown(why) => {
                assert!(!why.is_empty(), "a stand-down must explain itself");
                assert!(why.contains("512"), "message names the actual free RAM");
                assert!(why.contains('2'), "message names the core count");
            }
            RaGate::Run => panic!("a refusal must not run rust-analyzer"),
        }
    }

    #[test]
    fn degrade_for_a_real_reason_also_stands_down() {
        // Degrade is not "run it anyway" — for RA there is no half-server, so the
        // reduced path IS the syntactic scan.
        assert!(matches!(
            interpret(Verdict::Degrade(Reason::InsufficientCpu), cap(8000, 1)),
            RaGate::StandDown(_)
        ));
        assert!(matches!(
            interpret(Verdict::Degrade(Reason::MemoryBudgetExceeded), cap(300, 4)),
            RaGate::StandDown(_)
        ));
    }

    #[test]
    fn decide_on_this_workspace_never_panics_and_answers_something() {
        // Integration smoke: whatever this CI box looks like, the gate must
        // return a decision cheaply rather than blow up or hang.
        let gate = decide(Path::new("."));
        assert!(matches!(gate, RaGate::Run | RaGate::StandDown(_)));
    }
}

//! The resource-aware LSP guard: don't OOM the tablet by auto-starting
//! rust-analyzer on a project too big for the device.
//!
//! # The problem, plainly
//!
//! kvim runs on Android tablets. On a small Rust project rust-analyzer is
//! fine. On a BIG project relative to the device's RAM/CPU it can get so laggy
//! it OOM-kills the whole app — Android has no swap to fall back on, so once a
//! process wants more memory than is free, the kernel's low-memory killer just
//! reaps it, and a rust-analyzer that dragged the tablet down takes kvim with
//! it. So before kvim auto-attaches the server on open (attach-on-open, see
//! `docs/ai-decisions/AID-0023`), it should *check first*: is this project
//! likely too heavy for this device? If yes, don't start — say why, and let the
//! user force it with `:LspStart` if they know better.
//!
//! # How the guess is made (all rough, on purpose)
//!
//! Three cheap inputs, no cargo invocation, no server spawn:
//!
//! 1. **Device budget** — [`DeviceProbe`]: how much RAM is actually free right
//!    now, plus how many CPU cores. rust-analyzer's peak cost is mostly memory,
//!    but it is CPU-heavy *while indexing*, so a few-core tablet janks on a
//!    mid-size project even when RAM would fit — CPU is a real input to the
//!    gate, not just message flavour.
//! 2. **Project size** — [`ProjectSize`]: the thing that drives RA memory. RA
//!    analyses the whole dependency graph, so peak RSS scales mostly with the
//!    **number of crates** — proxied by the count of `[[package]]` entries in
//!    the workspace `Cargo.lock` (cheap: one file read, no `cargo metadata`).
//!    Total first-party `.rs` bytes is a secondary term.
//! 3. **The heuristic** — [`evaluate`]: `est_ra_mb = base + per_dep*deps +
//!    src_factor*src_mb`, gated against `avail_mb * headroom * core_factor`.
//!
//! The constants all live in [`LspGuardConfig`] with conservative defaults, and
//! the whole reasoning (alternatives, and what would make the model wrong) is in
//! `docs/ai-decisions/AID-0037`.
//!
//! # Fail-open is the contract, not a bug
//!
//! The guard exists to stop a *tablet* OOM, not to second-guess a capable
//! machine. So whenever it cannot get an honest reading — no probe available
//! (sysinfo returned nothing usable on this platform), memory reported as zero,
//! no `Cargo.lock` to count — it **fails open**: the estimate is treated as
//! "fits", and the LSP starts as it always did. A desktop with 64 GB free will
//! never see the gate fire because the budget dwarfs any realistic estimate;
//! that is the intended behaviour, not a loophole.
//!
//! # Why the `sysinfo` crate
//!
//! The device probe uses `sysinfo` (pure Rust, Android-capable, the crate
//! `bottom`/`btm` is built on) rather than hand-parsing `/proc/meminfo`. One
//! dependency, three consumers in kvim: this guard's estimate (available/total
//! RAM + core count), the LSP startup progress bar's ETA (live CPU usage), and a
//! future runtime memory-cap-and-kill guard (per-process RSS). Library only —
//! kvim never shells out to any monitoring binary, since none exists on the
//! tablet.

use std::path::{Path, PathBuf};

use crate::config::LspGuardConfig;

/// A snapshot of the device's resources, in the units the guard reasons in.
///
/// Built from `sysinfo` by [`probe_device`], but kept as a plain struct of
/// numbers so [`evaluate`] can be unit-tested against synthetic devices without
/// touching the real machine. That separation is the whole point: the decision
/// is pure arithmetic over these fields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeviceProbe {
    /// Kernel's estimate of RAM allocatable *without swapping*, in **MB**. This
    /// is the honest budget — not total RAM, not free RAM. On Android this is
    /// what you actually have before the OOM killer wakes up.
    pub avail_mb: u64,
    /// Total physical RAM in **MB**. Message context only; the gate uses
    /// `avail_mb`.
    pub total_mb: u64,
    /// Logical CPU count (hardware threads) — what bounds rust-analyzer's
    /// indexing parallelism, and so the axis the core factor scales on.
    pub logical_cores: usize,
    /// Current system-wide CPU usage `0.0..=100.0`, best-effort. Message flavour
    /// (and the progress-bar ETA later); may read `0.0` on a cold single-sample
    /// probe, which is fine.
    pub cpu_usage: f32,
}

/// A cheap estimate of a workspace's size, the input that drives rust-analyzer's
/// memory footprint. See [`estimate_project_size`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProjectSize {
    /// Number of `[[package]]` entries in the workspace `Cargo.lock` — i.e. the
    /// resolved dependency graph's crate count (includes the workspace's own
    /// members; it is a proxy, exactness is not the point). `0` when no
    /// `Cargo.lock` was found (which fails the gate open, see [`evaluate`]).
    pub num_deps: usize,
    /// Total bytes of first-party `.rs` source under the workspace root, walked
    /// with the `ignore` crate (so `.gitignore` and `target/` are skipped).
    pub src_bytes: u64,
}

impl ProjectSize {
    /// Workspace `.rs` source in MB, as [`evaluate`]'s `src_factor` term wants.
    fn src_mb(&self) -> f64 {
        self.src_bytes as f64 / (1024.0 * 1024.0)
    }
}

/// The guard's verdict for one buffer, plus every number that fed it — so
/// `:LspInfo` can show the working, and the gated message can quote it.
#[derive(Debug, Clone, PartialEq)]
pub struct GuardDecision {
    /// `true` = let the LSP auto-start. `false` = hold off (project looks too
    /// big for this device); the user can still force it with `:LspStart`.
    pub allow: bool,
    /// Estimated rust-analyzer peak RSS in MB (`base + per_dep*deps +
    /// src_factor*src_mb`). `None` when the guard failed open before estimating
    /// (no probe / no project data), so nothing was computed.
    pub est_ra_mb: Option<f64>,
    /// The budget the estimate was compared against, in MB
    /// (`avail_mb * headroom * core_factor`). `None` on fail-open.
    pub budget_mb: Option<f64>,
    /// The device probe used, if one was available.
    pub probe: Option<DeviceProbe>,
    /// The project size used, if it could be estimated.
    pub size: Option<ProjectSize>,
    /// The core factor applied to the budget (`min(1.0, cores / core_ref)`),
    /// for `:LspInfo`. `None` on fail-open.
    pub core_factor: Option<f64>,
    /// A short human (Singlish) sentence explaining the verdict, ready to echo.
    pub reason: String,
}

impl GuardDecision {
    /// The fail-open verdict: allow, with a note why nothing was computed. Used
    /// whenever the guard cannot get an honest reading — the whole
    /// "capable machines are never blocked" contract lives here.
    fn fail_open(reason: impl Into<String>) -> Self {
        Self {
            allow: true,
            est_ra_mb: None,
            budget_mb: None,
            probe: None,
            size: None,
            core_factor: None,
            reason: reason.into(),
        }
    }
}

/// Probes the current device via `sysinfo`, or `None` if no honest reading is
/// available (which the caller treats as fail-open — start the LSP).
///
/// Reads available + total memory, the logical core count, and current CPU
/// usage. Returns `None` when available memory reads as `0`, which is how a
/// platform with no usable `sysinfo` memory backend shows up — better to fail
/// open than to gate on a bogus zero-budget reading.
///
/// # A note on CPU usage
///
/// `sysinfo` needs two samples spaced by `MINIMUM_CPU_UPDATE_INTERVAL` to report
/// a meaningful global CPU percentage. This probe takes a single cheap sample
/// (no sleep on the UI thread), so `cpu_usage` may come back `0.0` on the first
/// call — acceptable, because CPU usage is only message flavour here; the gate's
/// CPU axis is the *core count*, which is exact.
pub fn probe_device() -> Option<DeviceProbe> {
    use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

    let sys = System::new_with_specifics(
        RefreshKind::nothing()
            .with_memory(MemoryRefreshKind::nothing().with_ram())
            .with_cpu(CpuRefreshKind::nothing().with_cpu_usage()),
    );

    // sysinfo reports memory in BYTES (it moved off kB a few versions ago).
    let avail_mb = sys.available_memory() / (1024 * 1024);
    let total_mb = sys.total_memory() / (1024 * 1024);
    if avail_mb == 0 {
        // No usable memory backend on this platform -> fail open.
        return None;
    }

    // Logical cores: `available_parallelism` is pure std and is exactly "how
    // many threads can actually run at once", which is what bounds RA's indexing
    // parallelism. Fall back to sysinfo's CPU list, then to 1.
    let logical_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or_else(|_| sys.cpus().len().max(1));

    Some(DeviceProbe {
        avail_mb,
        total_mb,
        logical_cores,
        cpu_usage: sys.global_cpu_usage(),
    })
}

/// Walks up from `start` to the nearest ancestor holding a `Cargo.lock`, and
/// returns `(lock_path, workspace_root)`. `None` if none is found up to the
/// filesystem root.
///
/// `Cargo.lock` lives at the **workspace** root (not per-package), so its
/// directory is the right place to both count deps and walk `.rs` source. We key
/// on `Cargo.lock` specifically rather than `Cargo.toml`, because a nested
/// package's `Cargo.toml` would undercount a workspace's real dependency graph.
fn find_cargo_lock(start: &Path) -> Option<(PathBuf, PathBuf)> {
    let mut dir: &Path = start;
    loop {
        let candidate = dir.join("Cargo.lock");
        if candidate.is_file() {
            return Some((candidate, dir.to_path_buf()));
        }
        dir = dir.parent()?;
    }
}

/// Counts `[[package]]` entries in a `Cargo.lock`'s text — the resolved crate
/// count, our proxy for how much of the dependency graph rust-analyzer will
/// analyse (and so how much memory it will want).
///
/// This is a deliberately dumb line scan, not a TOML parse: `Cargo.lock` is a
/// generated file whose `[[package]]` array headers each sit alone on a line, so
/// counting those lines is both correct in practice and far cheaper than pulling
/// in a TOML parser for one number.
fn count_lock_packages(lock_text: &str) -> usize {
    lock_text.lines().filter(|line| line.trim() == "[[package]]").count()
}

/// Estimates the size of the workspace enclosing `file`, or `None` if there is
/// no `Cargo.lock` above it (not a cargo workspace we can size cheaply → the
/// caller fails open).
///
/// Two numbers, both cheap: the `Cargo.lock` package count, and the total bytes
/// of `.rs` under the workspace root walked with `ignore` (respects
/// `.gitignore`, skips `target/`, so a built tree does not inflate the figure).
pub fn estimate_project_size(file: &Path) -> Option<ProjectSize> {
    // Start the walk-up from the file's own directory (or the file itself if it
    // has no parent, e.g. a bare filename).
    let start = file.parent().unwrap_or(file);
    let (lock_path, root) = find_cargo_lock(start)?;

    let num_deps = std::fs::read_to_string(&lock_path).map(|t| count_lock_packages(&t)).unwrap_or(0);

    // Sum `.rs` bytes under the workspace root. `ignore` skips `target/` and
    // gitignored paths, so this is first-party source, not build artefacts.
    let mut src_bytes: u64 = 0;
    for entry in ignore::WalkBuilder::new(&root).hidden(false).build().flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "rs") {
            // metadata() can fail on a race; treat that file as 0 bytes rather
            // than aborting the whole walk.
            let len = entry.metadata().map(|m| m.len()).unwrap_or(0);
            src_bytes = src_bytes.saturating_add(len);
        }
    }

    Some(ProjectSize { num_deps, src_bytes })
}

/// The workspace root kvim keys a per-project guard decision on: the
/// `Cargo.lock` directory if there is one above `file`, else `file`'s own
/// directory. Cheap (a walk-up `stat` loop), safe to call on the idle tick.
pub fn project_root(file: &Path) -> PathBuf {
    let start = file.parent().unwrap_or(file);
    find_cargo_lock(start).map(|(_, root)| root).unwrap_or_else(|| start.to_path_buf())
}

/// The heuristic, as pure arithmetic over `(config, probe, size)`.
///
/// This is the function the unit tests hammer: feed synthetic devices and
/// project sizes and assert the gate flips at the boundary. It performs no I/O
/// and reads no globals — every input is an argument.
///
/// # The maths
///
/// ```text
/// est_ra_mb   = base_mb + per_dep_mb * num_deps + src_factor * src_mb
/// core_factor = min(1.0, logical_cores / core_ref_count)
/// budget_mb   = avail_mb * headroom * core_factor
/// allow       = est_ra_mb <= budget_mb
/// ```
///
/// The core factor folds CPU into the *gate*, not just the message: with the
/// default `core_ref_count = 8`, an 8+-core machine keeps the full budget while
/// a 2-core tablet's budget is quartered, so RA is held off far sooner there
/// even if the raw RAM would have fit.
///
/// # Fail-open cases (return `allow = true`, nothing computed)
///
/// * guard disabled (`enabled = false`),
/// * `always_start = true`,
/// * no device probe,
/// * no project size (no `Cargo.lock`),
/// * a probe reporting zero cores (bogus → don't trust it).
pub fn evaluate(cfg: &LspGuardConfig, probe: Option<DeviceProbe>, size: Option<ProjectSize>) -> GuardDecision {
    if !cfg.enabled {
        return GuardDecision::fail_open("LSP guard off (config) — starting server as usual.");
    }
    if cfg.always_start {
        return GuardDecision::fail_open("LSP guard set to always-start — starting server.");
    }
    let Some(probe) = probe else {
        return GuardDecision::fail_open("Cannot read device RAM/CPU — assume can lah, starting LSP.");
    };
    let Some(size) = size else {
        return GuardDecision::fail_open("No Cargo.lock to size the project — starting LSP.");
    };
    if probe.logical_cores == 0 {
        return GuardDecision::fail_open("Bogus 0-core reading — not trusting it, starting LSP.");
    }

    let est_ra_mb = cfg.base_mb + cfg.per_dep_mb * size.num_deps as f64 + cfg.src_factor * size.src_mb();
    let core_factor = (size_logical(probe) / cfg.core_ref_count).min(1.0);
    let budget_mb = probe.avail_mb as f64 * cfg.headroom * core_factor;
    let allow = est_ra_mb <= budget_mb;

    let reason = if allow {
        format!(
            "LSP ok: est ~{est_ra_mb:.0}MB fits budget ~{budget_mb:.0}MB ({deps} deps, {cores} cores).",
            deps = size.num_deps,
            cores = probe.logical_cores,
        )
    } else {
        // Never silently skip — the user must see WHY, and how to override.
        format!(
            "LSP off: project too heavy for this device lah (est ~{est_ra_mb:.0}MB vs budget ~{budget_mb:.0}MB \
             from {avail}MB free x {cores} cores, {deps} deps). Type :LspStart to force.",
            avail = probe.avail_mb,
            cores = probe.logical_cores,
            deps = size.num_deps,
        )
    };

    GuardDecision {
        allow,
        est_ra_mb: Some(est_ra_mb),
        budget_mb: Some(budget_mb),
        probe: Some(probe),
        size: Some(size),
        core_factor: Some(core_factor),
        reason,
    }
}

/// `probe.logical_cores` as `f64`, factored out only so [`evaluate`]'s core
/// factor reads cleanly.
fn size_logical(probe: DeviceProbe) -> f64 {
    probe.logical_cores as f64
}

/// The whole gate in one call: probe the device, size the project enclosing
/// `file`, and decide. This is what the attach-on-open path calls.
///
/// Any missing piece fails open (see [`evaluate`]).
pub fn decide_for(cfg: &LspGuardConfig, file: &Path) -> GuardDecision {
    let probe = probe_device();
    let size = estimate_project_size(file);
    evaluate(cfg, probe, size)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A device with lots of cores so the core factor is 1.0 and RAM is the only
    /// axis — lets the RAM-boundary tests isolate the memory gate.
    fn eight_core(avail_mb: u64) -> DeviceProbe {
        DeviceProbe { avail_mb, total_mb: avail_mb * 2, logical_cores: 8, cpu_usage: 10.0 }
    }

    fn cfg() -> LspGuardConfig {
        LspGuardConfig::default()
    }

    #[test]
    fn parses_the_package_count_from_a_cargo_lock() {
        let lock = "\
# This file is automatically @generated by Cargo.\nversion = 4\n\n\
[[package]]\nname = \"a\"\nversion = \"0.1.0\"\n\n\
[[package]]\nname = \"b\"\nversion = \"0.2.0\"\n\n\
[[package]]\nname = \"c\"\nversion = \"0.3.0\"\n";
        assert_eq!(count_lock_packages(lock), 3);
        // A line that merely mentions the token must not count.
        assert_eq!(count_lock_packages("dependencies = [[package]] inline\n"), 0);
        assert_eq!(count_lock_packages(""), 0);
    }

    #[test]
    fn small_project_on_a_big_device_is_allowed() {
        // 30 deps, ~1MB source, 16GB free, 8 cores.
        let size = ProjectSize { num_deps: 30, src_bytes: 1024 * 1024 };
        let d = evaluate(&cfg(), Some(eight_core(16_000)), Some(size));
        assert!(d.allow, "small project on a big device must start: {}", d.reason);
    }

    #[test]
    fn heavy_project_on_a_small_tablet_is_gated() {
        // 340 deps -> est ~ 150 + 4*340 + 0.5*~20 = ~1520MB; 3GB free but only
        // 2 cores quarters the budget: 3000 * 0.5 * (2/8) = 375MB. Gated.
        let size = ProjectSize { num_deps: 340, src_bytes: 20 * 1024 * 1024 };
        let tablet = DeviceProbe { avail_mb: 3000, total_mb: 4000, logical_cores: 2, cpu_usage: 40.0 };
        let d = evaluate(&cfg(), Some(tablet), Some(size));
        assert!(!d.allow, "heavy project on a 2-core tablet must be gated: {}", d.reason);
        assert!(d.reason.contains(":LspStart"), "gated message must tell the user how to override");
        assert!(d.est_ra_mb.unwrap() > d.budget_mb.unwrap());
    }

    #[test]
    fn ram_boundary_flips_the_gate() {
        // est for 100 deps, no source = 150 + 400 = 550MB. Budget on 8 cores is
        // avail * 0.5 * 1.0 = avail/2. So the boundary avail is 1100MB.
        let size = ProjectSize { num_deps: 100, src_bytes: 0 };
        let est = 150.0 + 4.0 * 100.0;
        assert_eq!(est, 550.0);

        // Just above the boundary (1200MB free -> budget 600MB) -> allow.
        let d = evaluate(&cfg(), Some(eight_core(1200)), Some(size));
        assert!(d.allow, "600MB budget must fit a 550MB estimate: {}", d.reason);

        // Just below (1000MB free -> budget 500MB) -> gate.
        let d = evaluate(&cfg(), Some(eight_core(1000)), Some(size));
        assert!(!d.allow, "500MB budget must not fit a 550MB estimate: {}", d.reason);
    }

    #[test]
    fn cpu_factor_gates_a_project_that_ram_alone_would_allow() {
        // 100 deps -> est 550MB. 1400MB free.
        // On 8 cores: budget = 1400 * 0.5 * 1.0 = 700MB -> allow.
        // On 2 cores: budget = 1400 * 0.5 * (2/8) = 175MB -> gated.
        // Same RAM, same project: cores alone flip the decision. This is the
        // whole point of folding CPU into the gate.
        let size = ProjectSize { num_deps: 100, src_bytes: 0 };
        let eight = DeviceProbe { avail_mb: 1400, total_mb: 4000, logical_cores: 8, cpu_usage: 0.0 };
        let two = DeviceProbe { avail_mb: 1400, total_mb: 4000, logical_cores: 2, cpu_usage: 0.0 };
        assert!(evaluate(&cfg(), Some(eight), Some(size)).allow, "8 cores: RAM fits, allow");
        assert!(!evaluate(&cfg(), Some(two), Some(size)).allow, "2 cores: same RAM, gated by CPU factor");
    }

    #[test]
    fn core_factor_saturates_at_one_for_many_cores() {
        // 32 cores must not *inflate* the budget beyond the RAM headroom — the
        // factor is min(1.0, ...), so a 32-core box gets the same budget as an
        // 8-core one, never more.
        let size = ProjectSize { num_deps: 100, src_bytes: 0 };
        let big = DeviceProbe { avail_mb: 1000, total_mb: 2000, logical_cores: 32, cpu_usage: 0.0 };
        let d = evaluate(&cfg(), Some(big), Some(size));
        assert_eq!(d.core_factor, Some(1.0));
        assert_eq!(d.budget_mb, Some(500.0), "32 cores still capped at 0.5 headroom of 1000MB");
    }

    #[test]
    fn fails_open_when_no_probe() {
        let size = ProjectSize { num_deps: 9999, src_bytes: 999 * 1024 * 1024 };
        let d = evaluate(&cfg(), None, Some(size));
        assert!(d.allow, "no probe -> fail open, even for an absurd project");
        assert!(d.est_ra_mb.is_none());
    }

    #[test]
    fn fails_open_when_no_project_size() {
        let d = evaluate(&cfg(), Some(eight_core(2000)), None);
        assert!(d.allow, "no Cargo.lock -> fail open");
    }

    #[test]
    fn disabled_guard_always_allows() {
        let mut c = cfg();
        c.enabled = false;
        let size = ProjectSize { num_deps: 9999, src_bytes: 0 };
        let d = evaluate(&c, Some(eight_core(100)), Some(size));
        assert!(d.allow, "disabled guard must never gate");
    }

    #[test]
    fn always_start_reports_but_does_not_gate() {
        let mut c = cfg();
        c.always_start = true;
        let size = ProjectSize { num_deps: 9999, src_bytes: 0 };
        let d = evaluate(&c, Some(eight_core(100)), Some(size));
        assert!(d.allow, "always_start must never gate");
    }

    #[test]
    fn zero_cores_fails_open() {
        let size = ProjectSize { num_deps: 100, src_bytes: 0 };
        let bogus = DeviceProbe { avail_mb: 1000, total_mb: 2000, logical_cores: 0, cpu_usage: 0.0 };
        let d = evaluate(&cfg(), Some(bogus), Some(size));
        assert!(d.allow, "a 0-core reading is bogus -> fail open, not divide-by-caution");
    }

    #[test]
    fn estimate_project_size_reads_a_real_workspace() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("Cargo.lock"),
            "[[package]]\nname=\"a\"\n\n[[package]]\nname=\"b\"\n",
        )
        .unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/lib.rs"), "fn a() {}\n").unwrap();
        let file = dir.path().join("src/lib.rs");
        let size = estimate_project_size(&file).expect("has a Cargo.lock");
        assert_eq!(size.num_deps, 2);
        assert!(size.src_bytes >= 10, "counted the lib.rs bytes");
    }

    #[test]
    fn estimate_project_size_is_none_without_a_lock() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("loose.rs");
        std::fs::write(&file, "fn x() {}\n").unwrap();
        assert!(estimate_project_size(&file).is_none(), "no Cargo.lock above -> None -> fail open");
    }

    #[test]
    fn probe_device_on_this_host_is_sane_or_none() {
        // Can't assert exact numbers (host-dependent), but if a probe comes back
        // it must be internally sane. On CI without a memory backend it may be
        // None, which is the fail-open path -- also fine.
        if let Some(p) = probe_device() {
            assert!(p.avail_mb > 0, "a Some probe must have non-zero available RAM");
            assert!(p.logical_cores >= 1);
            assert!(p.total_mb >= p.avail_mb, "total must be >= available");
        }
    }
}

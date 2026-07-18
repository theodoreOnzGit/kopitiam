//! The two hard gates every tool call must clear: **path confinement**
//! ([`confine`]) and the **§6 budget** ([`BudgetGate`]).
//!
//! Both are deterministic and both are *preemptive* — they decide *before* the
//! side effect, never after. That is the only way to be safe against the failures
//! that cannot be caught: a path escape can leak or clobber files outside the
//! project, and an oversized op can OOM-kill the process (uncatchable `SIGKILL` on
//! a swapless Android tablet — see `kopitiam-resource`). You cannot `?` your way
//! out of either after the fact, so you must refuse up front.

use std::path::{Component, Path, PathBuf};

use kopitiam_resource::budget::{DEFAULT_CORE_REF, DEFAULT_HEADROOM};
use kopitiam_resource::{
    budget_mb, BudgetPolicy, Capacity, DeviceProbe, Reason, SysinfoProbe, Verdict,
};

use crate::error::ToolError;

/// Confine a model-supplied `candidate` path **inside** the (already canonical)
/// workspace `root`, returning the cleaned absolute path if — and only if — it
/// stays inside. Otherwise [`ToolError::PathEscape`], and the caller must not run.
///
/// # What "inside" means, exactly (the safety contract)
///
/// `root` must be a **canonicalised** directory (the [`ToolExecutor`] canonicalises
/// it once at construction, so symlinks/`.`/`..` in the root itself are already
/// resolved). A `candidate` passes only if **all** of these hold:
///
/// 1. **No lexical climb above root.** We join `candidate` onto `root` (an absolute
///    `candidate` is taken as-is) and normalise `.`/`..` purely lexically. A `..`
///    that would pop above `root` is rejected — we do not consult the filesystem
///    for this step, so it cannot be fooled by files that happen not to exist yet
///    (important: `write` targets a path that does not exist).
/// 2. **Still under root after normalising.** The normalised path must
///    `starts_with(root)`. This catches an absolute `candidate` pointing somewhere
///    else entirely (`/etc/passwd`).
/// 3. **No symlink escape.** The deepest *existing* ancestor of the normalised
///    path is canonicalised and must **still** be under `root`. This defeats a
///    symlink *inside* the workspace that points out of it (`root/evil -> /etc`,
///    then `evil/passwd`): lexically it looks fine, but the real target is not,
///    and canonicalising the existing ancestor exposes that.
///
/// The returned path is lexically clean and absolute — the tool uses it directly
/// for the real fs op, so there is no window to re-introduce a `..` between the
/// check and the use.
///
/// # Why re-run this inside each tool's `run`
///
/// The executor runs [`confine`] in the path-gate stage (the load-bearing
/// rejection), and each [`crate::Tool::run`] runs it *again* on the same path
/// before the side effect. That is deliberate defence-in-depth against a future
/// caller that reaches `run` by a path other than [`ToolExecutor::execute`]; it is
/// cheap and idempotent, so there is no reason not to.
///
/// [`ToolExecutor`]: crate::ToolExecutor
/// [`ToolExecutor::execute`]: crate::ToolExecutor::execute
pub fn confine(root: &Path, candidate: &Path) -> Result<PathBuf, ToolError> {
    // Join onto root (absolute candidate stays absolute), then clean lexically.
    let joined = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        root.join(candidate)
    };

    let cleaned = lexical_clean(&joined).ok_or_else(|| {
        ToolError::PathEscape(format!(
            "`{}` climbs above the workspace root with `..`",
            candidate.display()
        ))
    })?;

    // (2) Must still sit under root after normalising.
    if !cleaned.starts_with(root) {
        return Err(ToolError::PathEscape(format!(
            "`{}` resolves to `{}`, which is outside the workspace root `{}`",
            candidate.display(),
            cleaned.display(),
            root.display()
        )));
    }

    // (3) Symlink defence: canonicalise the deepest existing ancestor and re-check.
    // A path with no existing ancestor at all is impossible here (root itself
    // exists and is an ancestor), so `deepest_existing` always finds something.
    if let Some(existing) = deepest_existing(&cleaned) {
        match existing.canonicalize() {
            Ok(real) => {
                if !real.starts_with(root) {
                    return Err(ToolError::PathEscape(format!(
                        "`{}` reaches `{}` through a symlink that leaves the workspace root",
                        candidate.display(),
                        real.display()
                    )));
                }
            }
            Err(e) => {
                return Err(ToolError::io(
                    &format!("canonicalising `{}`", existing.display()),
                    e,
                ));
            }
        }
    }

    Ok(cleaned)
}

/// Purely-lexical `.`/`..` normalisation. Returns `None` if a `..` would climb
/// **above** the path's root (an escape), otherwise the cleaned path. Touches the
/// filesystem **not at all** — that is what lets it gate a `write` target that
/// does not exist yet.
fn lexical_clean(path: &Path) -> Option<PathBuf> {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::Prefix(p) => out.push(p.as_os_str()),
            Component::RootDir => out.push(Component::RootDir.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                // Pop one real segment. If there is nothing to pop (we are at the
                // root / prefix), the `..` is climbing above root -> escape.
                if !out.pop() {
                    return None;
                }
            }
            Component::Normal(seg) => out.push(seg),
        }
    }
    Some(out)
}

/// Walk up from `path` to the nearest ancestor that actually exists on disk.
/// Used by [`confine`] step (3): we can only `canonicalize` something that exists,
/// so we canonicalise the deepest existing ancestor (which, if a symlink sits on
/// the way in, reveals the real out-of-root target).
fn deepest_existing(path: &Path) -> Option<PathBuf> {
    let mut p: &Path = path;
    loop {
        if p.exists() {
            return Some(p.to_path_buf());
        }
        p = p.parent()?;
    }
}

/// Sum, **stat-only**, the byte size of every file under `dir`, honouring
/// `.gitignore` and skipping `target/` (via the `ignore` crate). Never opens a
/// file — reads `metadata().len()` only, `O(files)`.
///
/// This is the cheap cost proxy the [`crate::tools::SearchTool`] hands the
/// [`BudgetGate`]: "how many bytes would a whole-tree walk have to chew through?".
/// It mirrors `kopitiam-resource`'s own stat-only project walk
/// (`kopitiam_resource::estimate_project_weight`), but counts **all** files, not
/// just `.rs`, because a text search scans everything the walk yields. Keeping the
/// proxy cheap is the whole point — the budget check must not itself be the
/// expensive thing it is guarding against.
pub fn stat_only_bytes(dir: &Path) -> u64 {
    // `require_git(false)`: honour `.gitignore` even when the workspace root is
    // not itself a git repo. `ignore`'s default only applies gitignore rules
    // inside an actual `.git` tree; a KOPITIAM workspace root may not be one, but
    // its `.gitignore` still means what it says, so we opt in unconditionally. The
    // search walk below uses the exact same setting, so cost estimate and actual
    // walk see the same file set.
    let mut total: u64 = 0;
    for entry in ignore::WalkBuilder::new(dir)
        .hidden(false)
        .require_git(false)
        .build()
        .flatten()
    {
        if entry.file_type().is_some_and(|t| t.is_file()) {
            // metadata() can race with a concurrent delete; treat a failed stat as
            // 0 bytes rather than aborting the whole estimate.
            let len = entry.metadata().map(|m| m.len()).unwrap_or(0);
            total = total.saturating_add(len);
        }
    }
    total
}

/// Bytes → MB (base-2), the unit `kopitiam-resource`'s budgeter reasons in.
pub fn bytes_to_mb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// The §6 budget gate for tool ops — a thin, reusing wrapper over
/// `kopitiam-resource`'s budgeter.
///
/// # Why this exists and what it reuses
///
/// `temp_ai_design.md` §6/§10.2: "a 'search the whole tree' on the tablet must be
/// budget-checked before it runs." The budgeter that answers "will this fit?" is
/// **not reimplemented here** — this struct just holds a device [`Capacity`]
/// snapshot + policy and forwards to `kopitiam_resource::BudgetPolicy::will_fit`
/// against `kopitiam_resource::budget_mb(cap, headroom, core_ref)`. Same
/// arithmetic, same [`Verdict`], same [`Reason`] the rust-analyzer and gguf
/// clients use — "one budgeter, many clients", tool execution being one more
/// client.
///
/// The tool reason is always [`Reason::MemoryBudgetExceeded`] (the generic
/// raw-bytes memory reason), because a tool op's cost is driven by bytes-to-chew,
/// not by "project too big".
#[derive(Debug, Clone, Copy)]
pub struct BudgetGate {
    /// The device snapshot the budget is built from. **Re-read before each heavy
    /// op** in production — free RAM is volatile (see [`BudgetGate::probe_now`]).
    capacity: Capacity,
    /// Fraction of available RAM the budget may occupy. Default
    /// `kopitiam_resource::DEFAULT_HEADROOM` (0.6).
    headroom: f64,
    /// Core-count reference for the CPU scaling. Default
    /// `kopitiam_resource::DEFAULT_CORE_REF` (8).
    core_ref: f64,
    /// The FULL/PARTIAL/SKIP boundary policy (the conservative marginal band).
    policy: BudgetPolicy,
}

impl BudgetGate {
    /// Build a gate from an explicit device [`Capacity`] with the conservative
    /// defaults. This is the constructor tests use — hand it a tiny synthetic
    /// capacity and a big op will [`Refuse`](Verdict::Refuse) deterministically,
    /// no real device involved.
    pub fn from_capacity(capacity: Capacity) -> Self {
        Self {
            capacity,
            headroom: DEFAULT_HEADROOM,
            core_ref: DEFAULT_CORE_REF,
            policy: BudgetPolicy::default(),
        }
    }

    /// Probe the *real* device right now (via `kopitiam-resource`'s
    /// [`SysinfoProbe`]) and build a gate from it. Call this right before a heavy
    /// tool op so the budget reflects the RAM actually free at that moment.
    ///
    /// If the probe cannot read the device (`snapshot()` is `None`), we **fail
    /// open**: a zero [`Capacity`] makes the budget `0`, and
    /// `kopitiam_resource::BudgetPolicy::will_fit` turns a non-positive budget into
    /// [`Verdict::Degrade`]`(NotApplicable)` — which the executor lets *proceed*.
    /// A machine we could not measure must never be hard-refused (the same
    /// fail-open rule the budgeter itself follows) — the point of the gate is to
    /// stop a *tablet* OOM, not to block a box we simply could not read.
    pub fn probe_now() -> Self {
        let cap = SysinfoProbe.snapshot().unwrap_or(Capacity {
            avail_mb: 0,
            total_mb: 0,
            logical_cores: 0,
            cpu_usage: 0.0,
        });
        Self::from_capacity(cap)
    }

    /// Override the headroom + core-ref (both forwarded straight to
    /// `kopitiam_resource::budget_mb`). Values are the crate defaults otherwise.
    pub fn with_budget_shape(mut self, headroom: f64, core_ref: f64) -> Self {
        self.headroom = headroom;
        self.core_ref = core_ref;
        self
    }

    /// Override the FULL/PARTIAL/SKIP policy (e.g. a stricter marginal band).
    pub fn with_policy(mut self, policy: BudgetPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// The gate proper: does an op estimated at `cost_mb` fit this device's tool
    /// budget? Returns the raw [`Verdict`] so the executor can decide (only
    /// [`Verdict::Refuse`] blocks; [`Verdict::Degrade`] is allowed to proceed —
    /// a discrete tool op has no "reduced" mode, so near-budget still runs, but a
    /// clear over-budget is refused).
    pub fn check(&self, cost_mb: f64) -> Verdict {
        let budget = budget_mb(self.capacity, self.headroom, self.core_ref);
        self.policy
            .will_fit(cost_mb, budget, Reason::MemoryBudgetExceeded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn sandbox() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn a_plain_relative_path_inside_root_is_allowed() {
        let dir = sandbox();
        let root = dir.path().canonicalize().unwrap();
        fs::create_dir(root.join("src")).unwrap();
        fs::write(root.join("src/lib.rs"), "fn a() {}\n").unwrap();
        let ok = confine(&root, Path::new("src/lib.rs")).unwrap();
        assert!(ok.starts_with(&root));
        assert!(ok.ends_with("src/lib.rs"));
    }

    #[test]
    fn a_dotdot_escape_is_rejected_not_resolved() {
        let dir = sandbox();
        let root = dir.path().canonicalize().unwrap();
        // Classic climb-out. Must be PathEscape, and (the point) it never touches
        // the file it is trying to reach.
        let err = confine(&root, Path::new("../secrets.txt")).unwrap_err();
        assert!(matches!(err, ToolError::PathEscape(_)), "got {err:?}");
    }

    #[test]
    fn a_deep_dotdot_escape_is_rejected() {
        let dir = sandbox();
        let root = dir.path().canonicalize().unwrap();
        let err = confine(&root, Path::new("a/b/../../../../etc/passwd")).unwrap_err();
        assert!(matches!(err, ToolError::PathEscape(_)), "got {err:?}");
    }

    #[test]
    fn an_absolute_path_outside_root_is_rejected() {
        let dir = sandbox();
        let root = dir.path().canonicalize().unwrap();
        let err = confine(&root, Path::new("/etc/passwd")).unwrap_err();
        assert!(matches!(err, ToolError::PathEscape(_)), "got {err:?}");
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_that_leaves_root_is_rejected() {
        let dir = sandbox();
        let outside = sandbox();
        let root = dir.path().canonicalize().unwrap();
        // root/escape -> <outside> (a real dir outside the workspace).
        std::os::unix::fs::symlink(outside.path(), root.join("escape")).unwrap();
        // Lexically `escape/loot` looks inside root; the symlink defence must
        // catch that its real target is outside.
        let err = confine(&root, Path::new("escape/loot")).unwrap_err();
        assert!(matches!(err, ToolError::PathEscape(_)), "got {err:?}");
    }

    #[test]
    fn a_nonexistent_write_target_inside_root_is_allowed() {
        let dir = sandbox();
        let root = dir.path().canonicalize().unwrap();
        // write targets a file that does not exist yet — must still pass, because
        // the climb check is lexical, not filesystem-dependent.
        let ok = confine(&root, Path::new("brand/new/file.rs")).unwrap();
        assert!(ok.starts_with(&root));
    }

    #[test]
    fn budget_refuses_an_oversized_op_on_a_tiny_device() {
        // 64 MB free, 8 cores -> budget = 64 * 0.6 * 1.0 = 38.4 MB. A 500 MB op is
        // way over the +15% band -> Refuse. This is the tablet-saving case.
        let cap = Capacity { avail_mb: 64, total_mb: 128, logical_cores: 8, cpu_usage: 0.0 };
        let gate = BudgetGate::from_capacity(cap);
        assert_eq!(gate.check(500.0), Verdict::Refuse(Reason::MemoryBudgetExceeded));
    }

    #[test]
    fn budget_fits_a_small_op_on_a_roomy_device() {
        let cap = Capacity { avail_mb: 16_000, total_mb: 32_000, logical_cores: 8, cpu_usage: 0.0 };
        let gate = BudgetGate::from_capacity(cap);
        assert_eq!(gate.check(1.0), Verdict::Fits);
    }

    #[test]
    fn stat_only_bytes_counts_files_but_never_opens_them() {
        let dir = sandbox();
        fs::write(dir.path().join("a.txt"), "hello").unwrap();
        fs::create_dir(dir.path().join("sub")).unwrap();
        fs::write(dir.path().join("sub/b.txt"), "world!!").unwrap();
        let total = stat_only_bytes(dir.path());
        assert_eq!(total, 5 + 7, "summed both files' metadata lengths");
    }
}

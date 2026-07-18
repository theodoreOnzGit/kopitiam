//! The [`ToolExecutor`] and the two traits every tool implements — [`ToolRequest`]
//! (the typed, gate-aware request) and [`Tool`] (the actual side effect).
//!
//! This module owns the **five-stage gate contract** (see the crate docs). Read
//! [`ToolExecutor::execute`] as the single source of truth for the order:
//! schema → path → budget → approval → execute, reject-never-execute.

use std::path::{Path, PathBuf};

use kopitiam_resource::Verdict;
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::approval::{ApprovalGate, ApprovalRequest};
use crate::error::ToolError;
use crate::gate::{self, BudgetGate};

/// Whether a tool only reads, or has a consequential side effect (write / exec).
///
/// The single axis the **approval seam** keys on: read-only tools skip stage 4;
/// write/exec tools must clear it. Kept separate from the tool's identity so the
/// executor can reason about "does this need a human?" without knowing which tool
/// it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ToolKind {
    /// Reads only — search, read. No approval needed.
    ReadOnly,
    /// Writes to the filesystem — write, edit. Needs approval.
    Write,
    /// Spawns a process — run. Needs approval (the sharpest one).
    Exec,
}

impl ToolKind {
    /// `true` for [`ToolKind::Write`] and [`ToolKind::Exec`] — i.e. "the executor
    /// must consult the [`ApprovalGate`] before running this".
    pub fn needs_approval(self) -> bool {
        !matches!(self, ToolKind::ReadOnly)
    }
}

/// A typed tool request that knows two things the gate needs *before* the tool
/// runs: **which paths it will touch**, and **how much it will cost**.
///
/// The request type is a plain `serde`-deserialisable struct — deserialising the
/// model's JSON *into* it is the schema gate (stage 1). The two methods feed the
/// path gate (stage 2) and budget gate (stage 3). Because these are computed from
/// the *request* (not from running anything), all three gates are pure functions
/// of the call + the workspace state — deterministic, testable, reproducible.
pub trait ToolRequest: DeserializeOwned {
    /// The stable name the model addresses this tool by (`"search"`, `"write"`...).
    const TOOL_NAME: &'static str;
    /// Read-only vs write vs exec — drives the approval seam.
    const KIND: ToolKind;

    /// Every filesystem path this call would touch, as the model supplied them
    /// (expected relative-to-workspace; an absolute one is allowed only if it
    /// still lands inside root). The executor [`confine`](gate::confine)s each one
    /// **before** the tool runs; any escape rejects the whole call.
    ///
    /// Return **all** touched paths — a tool that reads A and writes B must list
    /// both, or the unlisted one dodges the gate.
    fn touched_paths(&self) -> Vec<PathBuf>;

    /// A **cheap, upfront** cost estimate in **MB** for the §6 budget gate. Must
    /// not do the expensive work it is estimating — a stat-only proxy at most (a
    /// whole-tree search estimates bytes-to-scan via
    /// [`gate::stat_only_bytes`]; a write estimates its own payload size). If a
    /// tool's cost is genuinely trivial and bounded, return a small constant.
    ///
    /// `root` is the canonical workspace root, in case the estimate needs to walk
    /// a subtree (stat-only).
    fn cost_estimate_mb(&self, root: &Path) -> f64;

    /// One-line human-readable summary of the effect, for the approval choice card
    /// (write/exec only). Default is empty — read-only tools never show a card.
    fn approval_summary(&self) -> String {
        String::new()
    }
}

/// A tool: given an already-gated request, actually do the thing.
///
/// [`Tool::run`] is called **only after** all four gates have passed, so it is
/// free to touch the filesystem / spawn a process. It re-[`confine`](gate::confine)s
/// its paths defensively (cheap, idempotent — see [`gate::confine`]), but it does
/// **not** re-check budget or approval; those are the executor's job and are
/// already done by the time `run` is reached.
pub trait Tool {
    /// The typed request this tool consumes.
    type Req: ToolRequest;
    /// The typed, serialisable response this tool produces.
    type Resp: Serialize;

    /// Do the side effect. `root` is the canonical workspace root; resolve every
    /// path against it via [`gate::confine`], never by naive join.
    fn run(&self, request: Self::Req, root: &Path) -> Result<Self::Resp, ToolError>;
}

/// The gate + dispatcher. Holds the confined workspace root, the §6 budget gate,
/// and the approval seam; runs the five-stage contract for every call.
///
/// # Lifetime / ownership
///
/// Borrows the [`ApprovalGate`] (`&'a dyn ApprovalGate`) rather than owning it, so
/// one long-lived approval policy (a TUI prompt handler, a remembered-decisions
/// store) serves many executors and many calls. The root is owned + canonicalised
/// once at construction, so no per-call canonicalisation of the root and no window
/// for it to change under us.
pub struct ToolExecutor<'a> {
    /// Canonical workspace root. Every path the model gives is confined under this.
    root: PathBuf,
    /// The §6 budget gate (reuses `kopitiam-resource`). Re-probe + rebuild the
    /// executor before a heavy batch if free RAM may have moved.
    budget: BudgetGate,
    /// The caller's approval policy (consulted for write/exec).
    approval: &'a dyn ApprovalGate,
}

impl<'a> ToolExecutor<'a> {
    /// Build an executor for a workspace `root`, a [`BudgetGate`], and an
    /// [`ApprovalGate`]. The root is **canonicalised now** (resolving symlinks/
    /// `.`/`..` in the root itself) so [`gate::confine`] can trust it; a root that
    /// does not exist is an [`ToolError::Io`] here, up front.
    pub fn new(
        root: impl AsRef<Path>,
        budget: BudgetGate,
        approval: &'a dyn ApprovalGate,
    ) -> Result<Self, ToolError> {
        let root = root
            .as_ref()
            .canonicalize()
            .map_err(|e| ToolError::io(&format!("workspace root `{}`", root.as_ref().display()), e))?;
        Ok(Self { root, budget, approval })
    }

    /// The canonical workspace root everything is confined to.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Run a tool call given the model's **raw JSON args** — this is the schema
    /// gate's real entry point. See [`ToolExecutor::execute`] for the staged
    /// version once you already have a `serde_json::Value`.
    ///
    /// Use this from the agent loop: the grammar-constrained decoder produces a
    /// JSON string, you hand it straight here, and either you get a typed response
    /// or a typed [`ToolError`] — never an out-of-contract side effect.
    pub fn execute_json<T: Tool>(
        &self,
        tool: &T,
        args_json: &str,
    ) -> Result<T::Resp, ToolError> {
        let value: serde_json::Value =
            serde_json::from_str(args_json).map_err(|e| ToolError::Schema(e.to_string()))?;
        self.execute(tool, &value)
    }

    /// The five-stage gate contract, in order, reject-never-execute. This is the
    /// single source of truth for the contract; the crate docs describe it in
    /// prose, this code *is* it.
    pub fn execute<T: Tool>(
        &self,
        tool: &T,
        args: &serde_json::Value,
    ) -> Result<T::Resp, ToolError> {
        // ---- Stage 1: SCHEMA gate --------------------------------------------
        // Deserialise the model's JSON INTO the tool's typed request. Failure here
        // = a malformed / hallucinated call shape; reject before anything runs.
        let request: T::Req =
            serde_json::from_value(args.clone()).map_err(|e| ToolError::Schema(e.to_string()))?;

        // ---- Stage 2: PATH gate ----------------------------------------------
        // Every path the call would touch must confine inside the workspace root.
        // A `..`-escape, an absolute-outside, or a symlink-out rejects the whole
        // call. Nothing has run yet.
        for candidate in request.touched_paths() {
            gate::confine(&self.root, &candidate)?;
        }

        // ---- Stage 3: BUDGET gate --------------------------------------------
        // Cheap upfront cost estimate vs the §6 device budget (kopitiam-resource).
        // A clear over-budget op is Refused up front — this is what stops a tool
        // op OOM-killing a tablet (uncatchable, so it MUST be preemptive).
        let cost_mb = request.cost_estimate_mb(&self.root);
        if let Verdict::Refuse(reason) = self.budget.check(cost_mb) {
            return Err(ToolError::BudgetRefused(reason));
        }

        // ---- Stage 4: APPROVAL seam ------------------------------------------
        // Write/exec ops need a human (or policy) yes. Read-only ops skip this.
        if T::Req::KIND.needs_approval() {
            let decision = self.approval.decide(&ApprovalRequest {
                tool: T::Req::TOOL_NAME,
                kind: T::Req::KIND,
                paths: request.touched_paths(),
                summary: request.approval_summary(),
            });
            if !decision.is_approved() {
                return Err(ToolError::ApprovalDenied(T::Req::KIND));
            }
        }

        // ---- Stage 5: EXECUTE ------------------------------------------------
        // Only now does the side effect happen.
        tool.run(request, &self.root)
    }
}

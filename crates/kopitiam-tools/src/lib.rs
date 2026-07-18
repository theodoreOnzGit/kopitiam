//! `kopitiam-tools` — KOPITIAM's **Rust-gated** deterministic tool layer for the
//! AI agent loop. The model *proposes*; Rust *disposes*.
//!
//! # What this crate is (temp_ai_design.md §10.1 #5 + §10.3)
//!
//! These are the tools the model can ask for — **search / read / write / edit /
//! run** — but *every* tool call is checked by Rust **before** it runs. The 0.5B
//! local model on a phone cannot be trusted to drive a reliable tool loop, so we
//! do **not** trust it: it emits a structured JSON call (the grammar-constrained
//! decoding keystone, §10.1 #2, guarantees the JSON is well-formed), and Rust then
//! decides whether that call is even allowed to execute. A hallucinated,
//! out-of-workspace, or oversized call is **rejected deterministically, never
//! executed**. This is the §8.1 "ask-the-runtime" seam extended from READ to
//! WRITE.
//!
//! > LLM *asks*, Rust *executes* the query, Rust feeds the result back. ✅
//! > LLM *produces* the effect and you trust it. ❌ — that's just hallucination
//! > with side effects.
//!
//! # THE GATE CONTRACT — exact, load-bearing, do not reorder
//!
//! Every call goes through [`ToolExecutor::execute`], which runs these stages **in
//! this order**, and **stops at the first failure without executing anything**:
//!
//! 1. **Schema gate.** The raw JSON is parsed (`serde`) *into* the tool's typed
//!    request. If it does not deserialise — missing field, wrong type, unknown
//!    tool shape — it is [`ToolError::Schema`] and nothing runs. Parsing-into-the-
//!    type **is** the schema check; there is no separate validator to drift out of
//!    sync.
//! 2. **Path gate.** Every filesystem path the request would touch
//!    ([`ToolRequest::touched_paths`]) is canonicalised and confined **inside the
//!    workspace root** ([`gate::confine`]). A `..`-escape, an absolute path
//!    pointing outside root, or a symlink that leaves root → [`ToolError::PathEscape`],
//!    nothing runs.
//! 3. **Budget gate.** The request's cheap upfront cost estimate
//!    ([`ToolRequest::cost_estimate_mb`]) is checked against the §6 preemptive
//!    resource budget ([`gate::BudgetGate`], which reuses `kopitiam-resource`). An
//!    oversized op — e.g. "search the whole tree" on a tablet — that the budgeter
//!    [`Refuse`](kopitiam_resource::Verdict::Refuse)s → [`ToolError::BudgetRefused`],
//!    nothing runs. This is what stops a tool op from OOM-killing the phone
//!    (uncatchable `SIGKILL`, so it MUST be preemptive — see `kopitiam-resource`).
//! 4. **Approval seam.** If the tool is a [`ToolKind::Write`] or [`ToolKind::Exec`]
//!    op, the caller-supplied [`ApprovalGate`] is consulted (the §10.2 human-
//!    powered decision — the choice-card wearing a safety hat). A deny →
//!    [`ToolError::ApprovalDenied`], nothing runs. Read-only tools skip this stage.
//! 5. **Execute.** Only now does [`Tool::run`] actually touch the filesystem or
//!    spawn a process.
//!
//! **The invariant:** a request that fails any of stages 1–4 is *never* handed to
//! stage 5. Rejections are typed [`ToolError`] values; a rejection is never a
//! panic and never a side effect. Given the same workspace state + the same human
//! approval choices, a run is reproducible — the trace includes the human's
//! decisions as inputs (§10.2 determinism-holds).
//!
//! # Which tools are real vs scaffolded
//!
//! | Tool | Kind | Status |
//! |---|---|---|
//! | [`SearchTool`](tools::SearchTool) | read-only | **real** — recursive, honours `.gitignore` via `ignore` |
//! | [`ReadTool`](tools::ReadTool) | read-only | **real** — confined file read |
//! | [`WriteTool`](tools::WriteTool) | write | scaffolded — **full gate + approval**, real write behind approval |
//! | [`EditTool`](tools::EditTool) | write | scaffolded — **full gate + approval**, real find/replace behind approval |
//! | [`RunTool`](tools::RunTool) | exec | scaffolded — **full gate + approval**, real spawn behind approval |
//!
//! "Scaffolded" here means the *gating is real* — the same five-stage contract
//! runs for every tool. What is deliberately thin is the ergonomics around the
//! side effect (no diff preview, no fancy edit semantics yet); the safety is not
//! thin.
//!
//! # Provenance
//!
//! Design: `temp_ai_design.md` §10 (interactive TUI + agent loop, Termux-first),
//! specifically §10.1 #5 (Rust-gated tool execution) and §10.3 (layers → crates).
//! The budget gate reuses `kopitiam-resource` (§6, "one budgeter, many clients").
//! This crate owns no model and invokes no model — it is called by
//! `kopitiam-workflow`, the only layer allowed to wire a model into a pipeline.

#![forbid(unsafe_code)]

pub mod approval;
pub mod error;
pub mod executor;
pub mod gate;
pub mod tools;

// A flat re-export of the everyday names, so a caller writes
// `kopitiam_tools::{ToolExecutor, ToolError, ApprovalGate}` without hunting
// through modules. The modules stay the source of truth for the docs.
pub use approval::{ApprovalDecision, ApprovalGate, ApprovalRequest, AutoApprove, AutoDeny};
pub use error::ToolError;
pub use executor::{Tool, ToolExecutor, ToolKind, ToolRequest};
pub use gate::BudgetGate;
pub use tools::{EditTool, ReadTool, RunTool, SearchTool, WriteTool};

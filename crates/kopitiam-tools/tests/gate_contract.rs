//! End-to-end tests of the **five-stage gate contract** through
//! [`ToolExecutor`], the load-bearing part of the crate. Each stage's rejection
//! is proved to reject *and not execute*, and the happy paths are proved to run.
//!
//! The whole thesis of `kopitiam-tools` is "LLM proposes, Rust disposes" — so
//! these tests are the disposing: a hallucinated / out-of-workspace / oversized /
//! unapproved call must be turned away deterministically, never run.

use std::path::Path;

use kopitiam_resource::Capacity;
use kopitiam_tools::{
    AutoApprove, AutoDeny, ReadTool, SearchTool, ToolError, ToolExecutor, WriteTool,
};
use kopitiam_tools::gate::BudgetGate;
use serde_json::json;

/// A roomy device so the budget gate never interferes with the non-budget tests.
fn roomy() -> BudgetGate {
    BudgetGate::from_capacity(Capacity {
        avail_mb: 16_000,
        total_mb: 32_000,
        logical_cores: 8,
        cpu_usage: 0.0,
    })
}

/// A tiny device so a whole-tree op gets refused.
fn tiny() -> BudgetGate {
    BudgetGate::from_capacity(Capacity {
        avail_mb: 1,
        total_mb: 8,
        logical_cores: 8,
        cpu_usage: 0.0,
    })
}

fn sandbox() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("src")).unwrap();
    std::fs::write(dir.path().join("src/lib.rs"), "fn chope() {}\n").unwrap();
    dir
}

// ---- Stage 1: SCHEMA gate ------------------------------------------------------

#[test]
fn malformed_schema_is_rejected_before_running() {
    let dir = sandbox();
    let deny = AutoDeny;
    let exec = ToolExecutor::new(dir.path(), roomy(), &deny).unwrap();
    // SearchReq requires `query`; this JSON has none.
    let err = exec.execute(&SearchTool, &json!({ "path": "src" })).unwrap_err();
    assert!(matches!(err, ToolError::Schema(_)), "got {err:?}");
}

#[test]
fn wrong_type_in_schema_is_rejected() {
    let dir = sandbox();
    let deny = AutoDeny;
    let exec = ToolExecutor::new(dir.path(), roomy(), &deny).unwrap();
    // `query` must be a string; give it a number.
    let err = exec.execute(&SearchTool, &json!({ "query": 42 })).unwrap_err();
    assert!(matches!(err, ToolError::Schema(_)), "got {err:?}");
}

// ---- Stage 2: PATH gate --------------------------------------------------------

#[test]
fn out_of_workspace_read_is_rejected_not_executed() {
    let dir = sandbox();
    let deny = AutoDeny;
    let exec = ToolExecutor::new(dir.path(), roomy(), &deny).unwrap();
    // Absolute path outside root.
    let err = exec
        .execute(&ReadTool, &json!({ "path": "/etc/passwd" }))
        .unwrap_err();
    assert!(matches!(err, ToolError::PathEscape(_)), "got {err:?}");
}

#[test]
fn dotdot_escape_read_is_rejected() {
    let dir = sandbox();
    let deny = AutoDeny;
    let exec = ToolExecutor::new(dir.path(), roomy(), &deny).unwrap();
    let err = exec
        .execute(&ReadTool, &json!({ "path": "../../../../etc/passwd" }))
        .unwrap_err();
    assert!(matches!(err, ToolError::PathEscape(_)), "got {err:?}");
}

#[test]
fn a_write_escaping_the_workspace_never_creates_the_file() {
    let dir = sandbox();
    let outside = tempfile::tempdir().unwrap();
    let target = outside.path().join("pwned.txt");
    let approve = AutoApprove; // even with blanket approval, the PATH gate is earlier
    let exec = ToolExecutor::new(dir.path(), roomy(), &approve).unwrap();

    // Try to write outside root via an absolute path.
    let abs = target.to_string_lossy().to_string();
    let err = exec
        .execute(&WriteTool, &json!({ "path": abs, "contents": "x" }))
        .unwrap_err();
    assert!(matches!(err, ToolError::PathEscape(_)), "got {err:?}");
    assert!(!target.exists(), "the write must NEVER have happened outside root");
}

// ---- Stage 3: BUDGET gate ------------------------------------------------------

#[test]
fn oversized_search_is_budget_refused_before_the_walk() {
    let dir = tempfile::tempdir().unwrap();
    // ~1.5 MB of content -> cost ~1.5 MB. Tiny device budget = 1*0.6 = 0.6 MB,
    // +15% band = 0.69 MB. 1.5 > 0.69 -> Refuse.
    let big = "x".repeat(1_500_000);
    std::fs::write(dir.path().join("big.txt"), &big).unwrap();

    let deny = AutoDeny;
    let exec = ToolExecutor::new(dir.path(), tiny(), &deny).unwrap();
    let err = exec.execute(&SearchTool, &json!({ "query": "x" })).unwrap_err();
    match err {
        ToolError::BudgetRefused(_) => {}
        other => panic!("expected BudgetRefused, got {other:?}"),
    }
}

#[test]
fn a_modest_search_fits_the_budget_and_runs() {
    let dir = sandbox();
    let deny = AutoDeny;
    // Tiny device, but a tiny tree (a few dozen bytes) is well under budget.
    let exec = ToolExecutor::new(dir.path(), tiny(), &deny).unwrap();
    let resp: kopitiam_tools::tools::SearchResp = exec
        .execute(&SearchTool, &json!({ "query": "chope" }))
        .unwrap();
    assert_eq!(resp.hits.len(), 1);
}

// ---- Stage 4: APPROVAL seam ----------------------------------------------------

#[test]
fn write_is_blocked_by_autodeny_and_no_file_appears() {
    let dir = sandbox();
    let deny = AutoDeny;
    let exec = ToolExecutor::new(dir.path(), roomy(), &deny).unwrap();
    let err = exec
        .execute(&WriteTool, &json!({ "path": "new.txt", "contents": "hi" }))
        .unwrap_err();
    assert!(matches!(err, ToolError::ApprovalDenied(_)), "got {err:?}");
    assert!(!dir.path().join("new.txt").exists(), "denied write must not touch disk");
}

#[test]
fn write_runs_only_after_approval() {
    let dir = sandbox();
    let approve = AutoApprove;
    let exec = ToolExecutor::new(dir.path(), roomy(), &approve).unwrap();
    let resp: kopitiam_tools::tools::WriteResp = exec
        .execute(&WriteTool, &json!({ "path": "new.txt", "contents": "steady" }))
        .unwrap();
    assert_eq!(resp.bytes_written, 6);
    assert_eq!(
        std::fs::read_to_string(dir.path().join("new.txt")).unwrap(),
        "steady"
    );
}

// ---- Stage 5: EXECUTE happy path (read-only, no approval) ----------------------

#[test]
fn a_valid_read_inside_root_succeeds_via_the_executor() {
    let dir = sandbox();
    let deny = AutoDeny; // read-only never consults approval, so AutoDeny is fine
    let exec = ToolExecutor::new(dir.path(), roomy(), &deny).unwrap();
    let resp: kopitiam_tools::tools::ReadResp = exec
        .execute(&ReadTool, &json!({ "path": "src/lib.rs" }))
        .unwrap();
    assert_eq!(resp.contents, "fn chope() {}\n");
}

#[test]
fn execute_json_parses_raw_json_strings() {
    let dir = sandbox();
    let deny = AutoDeny;
    let exec = ToolExecutor::new(dir.path(), roomy(), &deny).unwrap();
    // The grammar-constrained decoder hands us a raw JSON string; execute_json is
    // the schema gate's real entry point.
    let resp: kopitiam_tools::tools::SearchResp = exec
        .execute_json(&SearchTool, r#"{"query":"chope"}"#)
        .unwrap();
    assert_eq!(resp.hits.len(), 1);
}

#[test]
fn executor_rejects_a_nonexistent_root_up_front() {
    let deny = AutoDeny;
    let missing = Path::new("/definitely/not/here/kopitiam-xyz");
    // ToolExecutor holds a `&dyn ApprovalGate` so it is not Debug; match rather
    // than unwrap_err.
    match ToolExecutor::new(missing, roomy(), &deny) {
        Err(ToolError::Io(_)) => {}
        Err(other) => panic!("expected Io, got {other:?}"),
        Ok(_) => panic!("a nonexistent root must not build an executor"),
    }
}

//! The concrete tools the model can request — each a typed
//! [`ToolRequest`](crate::ToolRequest) + [`Tool`](crate::Tool) pair, each gated by
//! the same five-stage contract in [`crate::ToolExecutor`].
//!
//! | Module | Tool | Kind | Status |
//! |---|---|---|---|
//! | [`search`] | [`SearchTool`] | read-only | real |
//! | [`read`] | [`ReadTool`] | read-only | real |
//! | [`write`] | [`WriteTool`] | write | scaffolded (full gate + approval) |
//! | [`edit`] | [`EditTool`] | write | scaffolded (full gate + approval) |
//! | [`run`] | [`RunTool`] | exec | scaffolded (full gate + approval) |
//!
//! Every tool re-[`confine`](crate::gate::confine)s its own paths before touching
//! the filesystem — defence-in-depth on top of the executor's path gate.

pub mod edit;
pub mod read;
pub mod run;
pub mod search;
pub mod write;

pub use edit::{EditReq, EditResp, EditTool};
pub use read::{ReadReq, ReadResp, ReadTool};
pub use run::{RunReq, RunResp, RunTool};
pub use search::{SearchHit, SearchReq, SearchResp, SearchTool};
pub use write::{WriteReq, WriteResp, WriteTool};

//! Rust knowledge providers for KOPITIAM's Semantic Runtime.
//!
//! Each provider turns a real, existing source of truth (cargo, rust-analyzer,
//! rustdoc) into facts expressed with `kopitiam-ontology` types. No provider
//! infers what a tool can already answer deterministically. Future language
//! adapters (C, C++, Go, Fortran, Visual Basic, C#) will live alongside these
//! under [`providers`], each emitting the same semantic representation.

mod async_session;
pub mod edit;
mod lsp_client;
mod lsp_types;
mod position;
mod provider;
pub mod providers;
mod session;

pub use async_session::{AsyncRustAnalyzerSession, DEFAULT_INDEX_TIMEOUT, LspState, RequestError};
pub use lsp_types::{
    CompletionItem, CompletionItemKind, Diagnostic, Hover, Location, Position, Range, Severity,
};
pub use provider::{KnowledgeProvider, ProviderOutput};
pub use providers::{CargoMetadataProvider, RustAnalyzerProvider, RustdocProvider};
pub use session::{CodeAction, RustAnalyzerSession};

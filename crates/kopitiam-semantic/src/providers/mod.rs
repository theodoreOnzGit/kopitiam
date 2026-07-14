//! Knowledge providers: adapters that turn a project's real tooling output
//! into `kopitiam-ontology` facts.
//!
//! Per `CLAUDE.md`'s Semantic Runtime crate table, every language adapter
//! lives here and every one emits *the same* semantic representation — that
//! sameness is the whole point. A C# `class` and a Rust `struct` both become
//! an [`EntityKind::Symbol`] with a provenance-carrying `source`, so the
//! knowledge graph, search, and the translation platform can reason across
//! languages without knowing which one a fact came from.
//!
//! [`EntityKind::Symbol`]: kopitiam_ontology::EntityKind::Symbol

pub mod cargo;
pub mod cpp;
pub mod csharp;
pub mod python;
pub mod rust_analyzer;
pub mod rustdoc;
pub mod vbnet;

pub use cargo::CargoMetadataProvider;
pub use rust_analyzer::RustAnalyzerProvider;
pub use rustdoc::RustdocProvider;

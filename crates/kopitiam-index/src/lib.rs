//! Embedded persistence for the Semantic Runtime's `.kopitiam` project
//! directory.
//!
//! Everything a KOPITIAM project persists (session memory, working set,
//! serialized graph/translation snapshots) lives under a `.kopitiam`
//! directory next to the project's `Cargo.toml`, the same convention as
//! `.git`. Because that is a plain path relative to the project — not a
//! platform config directory looked up through OS-specific APIs — this
//! crate needs no per-OS code to work on Linux, macOS, Windows, or Android.
//!
//! Storage is backed by [`redb`], a pure-Rust embedded database, chosen
//! over SQLite specifically to keep this workspace's Pure Rust Core
//! promise (no C compiled at build time). See `Store` for the API.

mod store;

pub use store::{PROJECT_DIR_NAME, Store, project_dir};

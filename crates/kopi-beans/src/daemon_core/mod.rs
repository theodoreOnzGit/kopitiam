//! Shared typed primitives used by daemon runtime and external adapters.

#![forbid(unsafe_code)]

pub mod durability;
pub mod repl;
pub mod wal;

pub use crate::core as core;

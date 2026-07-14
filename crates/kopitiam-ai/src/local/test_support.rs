//! Test-only fixtures shared by [`super::adapter`]'s `#[cfg(test)]` unit
//! tests.
//!
//! `#[cfg(test)]`-gated in [`super`] (see `mod.rs`), so none of this exists
//! in a release build.

pub(crate) mod synthetic_gguf;

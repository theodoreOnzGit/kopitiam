//! Test-only helpers shared by every module's `#[cfg(test)]` unit tests.
//!
//! This module is `#[cfg(test)]`-gated in [`crate`] (see `lib.rs`), so none
//! of it exists in a release build. It exists because several modules
//! (`bridge`, `weights`, `model`, `kv_cache`) all need the same thing: a
//! real, on-disk GGUF file small enough to build in-process and cheap
//! enough to parse in a unit test, exercising the *actual*
//! `kopitiam_loader::load_model` parse path rather than a hand-built
//! `LoadedModel` (which cannot be constructed outside `kopitiam-loader`
//! anyway — its fields are private).

pub(crate) mod synthetic_gguf;

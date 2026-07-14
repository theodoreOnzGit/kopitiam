//! The Translation Platform for KOPITIAM's Semantic Runtime.
//!
//! Implements the state half of `CLAUDE.md`'s translation pipeline
//! (`legacy source -> language adapter -> semantic model -> runtime
//! knowledge -> translation workflow -> verification -> persistent
//! translation state`): [`LanguageAdapter`] identifies which files belong
//! to a legacy language, and [`TranslationState`] tracks every unit's
//! progress (pending, translated, verified, or failed) persistently across
//! sessions via `kopitiam-index`.
//!
//! What this crate does *not* do: parse legacy source, decide what Rust to
//! emit, or call a model. Those are `kopitiam-workflow`'s `translate`
//! workflow (context assembly + model invocation) and a future concrete
//! [`LanguageAdapter`] implementation's job — this crate only owns the
//! bookkeeping both depend on.

mod adapter;
mod state;

pub use adapter::LanguageAdapter;
pub use state::{TranslationState, TranslationUnit, UnitStatus};

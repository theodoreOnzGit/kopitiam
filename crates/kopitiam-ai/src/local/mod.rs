//! [`LocalAdapter`]: a [`crate::ModelAdapter`] backed by `kopitiam-runtime`'s
//! Qwen inference engine — the concrete answer to CLAUDE.md's Offline First
//! pipeline's "local AI" rung. Entirely behind the default-on `local`
//! Cargo feature; see [`adapter`]'s module docs for why.
//!
//! Split into three small modules so the two parts of "run a local model"
//! that do *not* need a model to test — chat templating and generation
//! control — are each testable in isolation, per this crate's own
//! task brief:
//!
//! * [`chat_template`] — renders [`crate::Message`]s into the literal
//!   ChatML prompt text. Pure `&str` in, `String` out; no model, no
//!   tokenizer, no I/O.
//! * [`generation`] — resolves a [`crate::CompletionRequest`]'s
//!   `max_tokens` and a model's candidate stop tokens into the
//!   `kopitiam_runtime::GenerationConfig` fields `LocalAdapter::complete`
//!   runs with. Pure `Option<u32>`/`u32` in, `usize`/`Option<u32>` out.
//! * [`adapter`] — [`LocalAdapter`] itself: the only one of the three that
//!   actually touches `kopitiam-runtime`, `kopitiam-loader`, and
//!   `kopitiam-tokenizer`.

mod adapter;
mod chat_template;
mod generation;

#[cfg(test)]
mod test_support;

pub use adapter::LocalAdapter;

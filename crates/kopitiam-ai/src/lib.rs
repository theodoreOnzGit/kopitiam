//! Pluggable model adapters for KOPITIAM's Semantic Runtime.
//!
//! This crate defines the [`ModelAdapter`] trait — the sole boundary through
//! which a model (local Qwen, Claude, GPT, Gemini, ...) is invoked anywhere
//! in the platform. Per the Semantic Runtime's dependency rule, only
//! `kopitiam-workflow` depends on this crate; everything else in the
//! platform (`kopitiam-knowledge`, `kopitiam-index`, `kopitiam-search`,
//! `kopitiam-workspace`, `kopitiam-translation`) stays model-agnostic.
//!
//! What lives here is the shape of a request/response
//! ([`CompletionRequest`], [`CompletionResponse`], [`Message`]), one
//! deterministic stub adapter ([`EchoAdapter`], always available) so
//! `kopitiam-workflow` has something real to compile and test against with
//! no weights and no network, and — behind the default-on `local` Cargo
//! feature — [`LocalAdapter`], a real, offline, on-CPU adapter backed by
//! `kopitiam-runtime`'s Qwen inference stack. See [`local`]'s module docs
//! for the full local-vs-cloud architecture and why depending on
//! `kopitiam-runtime` from here does not violate the Semantic Runtime's
//! dependency rule.

mod adapter;
mod echo;
mod message;

#[cfg(feature = "local")]
mod local;

pub use adapter::ModelAdapter;
pub use echo::EchoAdapter;
#[cfg(feature = "local")]
pub use local::LocalAdapter;
pub use message::{CompletionRequest, CompletionResponse, Message, Role};

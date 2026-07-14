use anyhow::Result;

use crate::{CompletionRequest, CompletionResponse};

/// A pluggable connection to one model backend.
///
/// This is the entire surface `kopitiam-workflow` is allowed to see of a
/// model — per the Semantic Runtime's dependency rule ("nothing below
/// `kopitiam-workflow` may depend on `kopitiam-ai`"), this trait is the
/// only place a model gets invoked from anywhere in the platform. Concrete
/// adapters (local Qwen — see `kopitiam-082`, Claude, GPT, Gemini) live
/// behind this trait so workflows are never written against one vendor.
///
/// Implementations should be cheap to construct and treat `complete` as the
/// only place that does real (possibly networked, possibly slow) work —
/// `kopitiam-workflow` decides *whether* to call a model at all, following
/// the Offline First pipeline (existing knowledge, then native Rust, then
/// local AI, then cloud AI as the final fallback).
pub trait ModelAdapter {
    /// Stable identifier for this adapter (e.g. `"local-qwen"`,
    /// `"claude"`), used in logs and recorded provenance — not necessarily
    /// the same as [`CompletionResponse::model`], which names the specific
    /// model that answered.
    fn name(&self) -> &str;

    /// Sends `request` to the backend and returns its reply.
    ///
    /// Implementations for backends that are unavailable in the current
    /// environment (no network, no local weights downloaded) should return
    /// `Err` rather than panicking, so `kopitiam-workflow` can fall back to
    /// the next adapter in the Offline First pipeline.
    fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse>;
}

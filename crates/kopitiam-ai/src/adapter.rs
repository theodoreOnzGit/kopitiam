use std::sync::mpsc::Receiver;

use anyhow::Result;

use crate::stream::complete_then_stream;
use crate::{CompletionRequest, CompletionResponse, StreamChunk};

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

    /// Streams the reply to `request` **token by token**, in order, over a
    /// [`Receiver`] of [`StreamChunk`]s — the shape a live chat UI needs so
    /// it can render each token as it lands instead of freezing on
    /// `complete()` (a phone does only a few tokens per second; see
    /// `temp_ai_design.md` §10.4).
    ///
    /// # The contract every implementation must keep
    ///
    /// The returned `Receiver` yields zero or more [`StreamChunk::Token`]s
    /// and then **exactly one** terminal chunk — [`StreamChunk::Done`] on a
    /// clean finish or [`StreamChunk::Error`] on a failure partway — and
    /// then the channel closes (the producer's `Sender` drops). So a caller
    /// can always drain with `for chunk in rx { ... }` and trust that the
    /// last thing it sees is the terminal chunk. See [`StreamChunk`] for the
    /// full frozen contract.
    ///
    /// # The default is eager, not streaming
    ///
    /// The provided body runs [`ModelAdapter::complete`] to completion and
    /// then delivers the whole reply as one `Token` + `Done` (or a lone
    /// `Error`). It honours the contract but does **not** produce tokens
    /// incrementally and does **not** use a background thread — it's the
    /// correct fallback for a blocking-only backend (e.g. a cloud adapter
    /// whose HTTP layer isn't wired up yet). Adapters that can genuinely
    /// emit one token at a time — [`crate::EchoAdapter`],
    /// [`crate::LocalAdapter`] — **override** this with a background actor
    /// (std thread + `mpsc`, per AID-0028) so the foreground never blocks.
    fn stream(&self, request: &CompletionRequest) -> Receiver<StreamChunk> {
        complete_then_stream(self, request)
    }
}

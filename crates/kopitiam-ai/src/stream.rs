//! Token-by-token streaming for [`crate::ModelAdapter`].
//!
//! `complete()` is the blocking, whole-reply-at-once call. Streaming is the
//! *other* shape a chat UI needs: on a phone doing a few tokens per second
//! (see `temp_ai_design.md` §10.4), you cannot freeze the screen on
//! `complete()` while the model grinds — you show each token the moment it
//! lands. [`crate::ModelAdapter::stream`] is that shape.
//!
//! # The concurrency discipline — one channel, one background actor
//!
//! `stream` hands back a [`std::sync::mpsc::Receiver`] and the *producing*
//! work runs on a **background thread** that owns its side of the channel —
//! plain std threads + `mpsc`, **no async runtime**. This is exactly the
//! actor shape KOPITIAM already committed to in **AID-0028** (the async LSP
//! session actor): a worker owns a resource, streams results out over a
//! channel, and never blocks the foreground. We reuse that discipline here
//! rather than inventing a second concurrency model — the caller's UI thread
//! stays free to render while the model thread produces.
//!
//! The contract the caller can rely on, for *every* adapter:
//!
//! * chunks arrive **in order**;
//! * a run ends with **exactly one terminal chunk** — either
//!   [`StreamChunk::Done`] (clean finish) or [`StreamChunk::Error`] (gave up
//!   partway) — and **nothing follows it**;
//! * when the producer thread finishes, its `Sender` drops, so iterating the
//!   `Receiver` (`for chunk in rx`) ends naturally after the terminal chunk.
//!
//! Because the terminal chunk is always present, a consumer never has to
//! distinguish "stream ended cleanly" from "sender was dropped mid-reply" by
//! guesswork — the [`StreamChunk::Done`] vs [`StreamChunk::Error`] tells it
//! outright.

use std::sync::mpsc::Receiver;

use crate::{CompletionRequest, ModelAdapter};

/// One piece of a streamed model reply, delivered in order over the
/// [`Receiver`] that [`ModelAdapter::stream`] returns.
///
/// **This enum is a frozen contract** — every adapter (local, echo, cloud,
/// and anything built later) emits exactly these three variants, and every
/// consumer (the `kopitiam ai chat` loop today, a ratatui pane tomorrow)
/// matches on exactly these three. Don't add or rename variants without
/// treating it as the breaking change it is.
///
/// A well-formed stream is: zero or more [`StreamChunk::Token`], then
/// **one** terminal chunk — [`StreamChunk::Done`] on success or
/// [`StreamChunk::Error`] on failure — then nothing (the sender drops,
/// closing the channel).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamChunk {
    /// One decoded increment of the reply — in practice one model token's
    /// text (e.g. `"Hel"`, `"lo"`, `" world"`). Concatenating every
    /// `Token` in arrival order rebuilds the full reply.
    ///
    /// Honest caveat, inherited from `kopitiam_runtime::generate`'s
    /// token-by-token decode: a single multi-byte Unicode character that a
    /// tokenizer splits across two tokens can arrive split across two
    /// `Token`s. That's the normal, accepted tradeoff of *any* streaming
    /// LLM UI — the blocking `complete()` path decodes the whole sequence
    /// together and never splits — so a caller that needs
    /// character-perfect text mid-stream should buffer, but a chat UI that
    /// just appends each `Token` to a pane is fine.
    Token(String),

    /// The stream finished cleanly. **Always the last chunk on a successful
    /// run**; nothing follows it. Its arrival is how a consumer knows the
    /// reply is complete (as opposed to the model still thinking).
    Done,

    /// Generation failed partway. **Always the last chunk on a failed run**;
    /// carries a human-readable reason (rendered from the underlying
    /// `anyhow` error). Nothing follows it. Any `Token`s that arrived before
    /// it are still valid partial output — the failure just means the reply
    /// is truncated, not that what came before is wrong.
    Error(String),
}

/// Runs `adapter.complete(request)` to completion and *then* pushes the whole
/// reply into a fresh channel as a single [`StreamChunk::Token`] followed by
/// [`StreamChunk::Done`] (or a lone [`StreamChunk::Error`] on failure),
/// returning the already-populated [`Receiver`].
///
/// This is the **eager, non-streaming fallback** [`ModelAdapter::stream`]'s
/// default body uses: it honours the [`StreamChunk`] contract exactly (a
/// `Token` then a terminal chunk, in order) so a caller can treat *any*
/// adapter uniformly as a stream — but it does **not** deliver tokens as they
/// are produced and it does **not** run on a background thread, because a
/// blocking-only backend has nothing to stream incrementally. Adapters that
/// *can* produce tokens one at a time (see [`crate::EchoAdapter`],
/// [`crate::LocalAdapter`]) override `stream` with a real background actor
/// instead of leaning on this.
///
/// It's generic over `A: ModelAdapter + ?Sized` so the trait's default
/// `stream` body can pass `self` (a `&Self`, which is not `Sized` inside a
/// default method) straight in, *and* a `&dyn ModelAdapter` still works —
/// while [`ModelAdapter`] stays object-safe (the default body references no
/// type parameter of its own).
pub(crate) fn complete_then_stream<A: ModelAdapter + ?Sized>(
    adapter: &A,
    request: &CompletionRequest,
) -> Receiver<StreamChunk> {
    let (tx, rx) = std::sync::mpsc::channel();
    match adapter.complete(request) {
        Ok(response) => {
            // A send only fails if the receiver was already dropped (caller
            // gave up). We're about to drop `tx` anyway, so ignoring the
            // error is correct — no one left to tell.
            let _ = tx.send(StreamChunk::Token(response.content));
            let _ = tx.send(StreamChunk::Done);
        }
        Err(error) => {
            let _ = tx.send(StreamChunk::Error(format!("{error:#}")));
        }
    }
    rx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CompletionResponse, EchoAdapter, Message};

    /// The default (eager) path must still honour the terminal-chunk
    /// contract: exactly `[Token, Done]` for a successful adapter.
    #[test]
    fn complete_then_stream_emits_token_then_done() {
        let request = CompletionRequest::new([Message::user("oi")]);
        let rx = complete_then_stream(&EchoAdapter, &request);
        let chunks: Vec<StreamChunk> = rx.iter().collect();
        assert_eq!(chunks, vec![StreamChunk::Token("oi".to_string()), StreamChunk::Done]);
    }

    /// A failing adapter yields exactly one terminal `Error` chunk and no
    /// `Token`/`Done`.
    #[test]
    fn complete_then_stream_emits_error_on_failure() {
        struct AlwaysFails;
        impl ModelAdapter for AlwaysFails {
            fn name(&self) -> &str {
                "always-fails"
            }
            fn complete(&self, _request: &CompletionRequest) -> anyhow::Result<CompletionResponse> {
                anyhow::bail!("kaput lah")
            }
        }

        let rx = complete_then_stream(&AlwaysFails, &CompletionRequest::new([Message::user("x")]));
        let chunks: Vec<StreamChunk> = rx.iter().collect();
        assert_eq!(chunks.len(), 1);
        match &chunks[0] {
            StreamChunk::Error(message) => assert!(message.contains("kaput lah")),
            other => panic!("expected a single Error chunk, got {other:?}"),
        }
    }
}

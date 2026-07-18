use std::sync::mpsc::Receiver;

use anyhow::Result;

use crate::{CompletionRequest, CompletionResponse, ModelAdapter, Role, StreamChunk};

/// A [`ModelAdapter`] that invokes no model at all: it echoes the last
/// [`Role::User`] message back as the response.
///
/// This exists so `kopitiam-workflow` (and its tests) can exercise the full
/// `load state -> collect facts -> build context -> invoke model -> validate
/// -> persist` pipeline deterministically, without a local Qwen build or
/// network access — useful in CI and while developing workflows before a
/// real adapter is wired in.
#[derive(Debug, Default, Clone, Copy)]
pub struct EchoAdapter;

impl ModelAdapter for EchoAdapter {
    fn name(&self) -> &str {
        "echo"
    }

    fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse> {
        let content = request
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.content.clone())
            .unwrap_or_default();

        Ok(CompletionResponse { content, model: self.name().to_string() })
    }

    /// Streams the echoed message back **word by word** from a background
    /// thread, so it exercises the real [`ModelAdapter::stream`] contract —
    /// genuinely incremental [`StreamChunk::Token`]s, not the eager
    /// whole-string default — with no model and no network.
    ///
    /// Why this bothers to stream at all, being a stub: it's the
    /// deterministic stand-in a chat UI (and its tests) run against when no
    /// local weights are present. If it only emitted one big `Token`, the
    /// UI's streaming path would never actually be tested offline. So it
    /// splits the reply on whitespace boundaries (keeping the whitespace, via
    /// [`str::split_inclusive`]) and sends one `Token` per piece, then
    /// [`StreamChunk::Done`] — proving multiple chunks arrive in order and
    /// the loop terminates cleanly. A single-word reply is still one `Token`
    /// then `Done`; a multi-word reply is several `Token`s then `Done`.
    ///
    /// The work runs on a spawned std thread owning the channel's `Sender`
    /// (the AID-0028 actor discipline), so a caller draining the `Receiver`
    /// on its own thread is never blocked by the producer. When the thread
    /// finishes the `Sender` drops and the channel closes.
    fn stream(&self, request: &CompletionRequest) -> Receiver<StreamChunk> {
        let (tx, rx) = std::sync::mpsc::channel();

        // Resolve the content *before* spawning: it's a plain `String`
        // (`Send`), so the thread needs nothing borrowed from `self` or
        // `request` — cleanest possible actor.
        let content = request
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.content.clone())
            .unwrap_or_default();

        std::thread::spawn(move || {
            // `split_inclusive` keeps the delimiter with the preceding
            // piece, so concatenating every Token reproduces `content`
            // exactly — no spaces dropped, no reordering.
            for piece in content.split_inclusive(char::is_whitespace) {
                // Send failure just means the receiver was dropped (caller
                // gave up); stop early, nothing left to stream to.
                if tx.send(StreamChunk::Token(piece.to_string())).is_err() {
                    return;
                }
            }
            let _ = tx.send(StreamChunk::Done);
        });

        rx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Message;

    #[test]
    fn echoes_the_last_user_message() {
        let request = CompletionRequest::new([
            Message::system("you are a helpful assistant"),
            Message::user("first"),
            Message::assistant("ack"),
            Message::user("second"),
        ]);

        let response = EchoAdapter.complete(&request).unwrap();
        assert_eq!(response.content, "second");
        assert_eq!(response.model, "echo");
    }

    #[test]
    fn empty_when_there_is_no_user_message() {
        let request = CompletionRequest::new([Message::system("system only")]);
        let response = EchoAdapter.complete(&request).unwrap();
        assert_eq!(response.content, "");
    }

    /// The core streaming property: a multi-word reply arrives as **more
    /// than one** `Token`, in order, and ends with exactly one `Done` — the
    /// incremental behaviour a chat UI's streaming path relies on.
    #[test]
    fn stream_yields_tokens_incrementally_then_done() {
        let request = CompletionRequest::new([Message::user("one two three")]);
        let chunks: Vec<StreamChunk> = EchoAdapter.stream(&request).iter().collect();

        // >1 Token proves it's genuinely incremental, not the eager default.
        let tokens: Vec<&str> = chunks
            .iter()
            .filter_map(|c| match c {
                StreamChunk::Token(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert!(tokens.len() > 1, "expected multiple Token chunks, got {tokens:?}");

        // Concatenating every Token in arrival order reproduces the reply
        // exactly — order preserved, whitespace intact.
        assert_eq!(tokens.concat(), "one two three");

        // The terminal chunk is exactly one Done, and it's last.
        assert_eq!(chunks.last(), Some(&StreamChunk::Done));
        assert_eq!(chunks.iter().filter(|c| **c == StreamChunk::Done).count(), 1);
        assert!(!chunks.iter().any(|c| matches!(c, StreamChunk::Error(_))));
    }

    /// A single-word reply is still a valid stream: one `Token`, then `Done`.
    #[test]
    fn stream_of_a_single_word_is_one_token_then_done() {
        let request = CompletionRequest::new([Message::user("solo")]);
        let chunks: Vec<StreamChunk> = EchoAdapter.stream(&request).iter().collect();
        assert_eq!(chunks, vec![StreamChunk::Token("solo".to_string()), StreamChunk::Done]);
    }
}

//! Pure, model-free generation-control logic: how a
//! [`crate::CompletionRequest`]'s `max_tokens` and a model's candidate stop
//! tokens become the `max_new_tokens`/`eos_token_id` fields of a
//! `kopitiam_runtime::GenerationConfig`.
//!
//! Factored out of `adapter.rs` and kept free of any `kopitiam-runtime`/
//! `kopitiam-tokenizer` type specifically so it is testable without
//! constructing a real (or even synthetic) model — see this module's tests.

/// Upper bound on generated tokens when a [`crate::CompletionRequest`] does
/// not set `max_tokens`. `kopitiam_runtime::GenerationConfig`'s own default
/// is `256`; this module picks the same number so "no limit requested"
/// behaves identically whether or not a caller thought to ask for the
/// default explicitly.
pub(crate) const DEFAULT_MAX_NEW_TOKENS: usize = 256;

/// Resolves [`crate::CompletionRequest::max_tokens`] into the `usize`
/// `kopitiam_runtime::GenerationConfig::max_new_tokens` expects, applying
/// [`DEFAULT_MAX_NEW_TOKENS`] when the caller did not set one.
pub(crate) fn resolve_max_new_tokens(requested: Option<u32>) -> usize {
    requested.map(|tokens| tokens as usize).unwrap_or(DEFAULT_MAX_NEW_TOKENS)
}

/// Picks the single token id `kopitiam_runtime::generate` should treat as
/// end-of-sequence.
///
/// `kopitiam_runtime::GenerationConfig::eos_token_id` accepts exactly one
/// id, so when a GGUF vocabulary offers more than one plausible stop token,
/// something has to choose between them. Priority, highest first:
///
/// 1. `<|im_end|>` — since [`super::chat_template`] always renders ChatML
///    and always primes the model with a trailing
///    `<|im_start|>assistant\n`, this is the token that actually closes
///    the turn this crate asked for.
/// 2. The GGUF's own `tokenizer.ggml.eos_token_id` metadata, when present —
///    the model file's own declared default end-of-sequence id, for
///    vocabularies where it differs from `<|im_end|>` (e.g. a base,
///    non-chat-tuned checkpoint with no ChatML special tokens at all).
/// 3. `<|endoftext|>` — the GPT-2/Qwen family's universal fallback
///    end-of-sequence marker, consulted last.
///
/// `None` if none of the three is available, in which case generation runs
/// for the full `max_new_tokens` budget every time — still correct, just
/// unable to stop early.
pub(crate) fn resolve_eos_token_id(
    im_end: Option<u32>,
    gguf_eos_metadata: Option<u32>,
    endoftext: Option<u32>,
) -> Option<u32> {
    im_end.or(gguf_eos_metadata).or(endoftext)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_max_new_tokens_honours_an_explicit_request() {
        assert_eq!(resolve_max_new_tokens(Some(32)), 32);
    }

    #[test]
    fn resolve_max_new_tokens_falls_back_to_the_documented_default() {
        assert_eq!(resolve_max_new_tokens(None), DEFAULT_MAX_NEW_TOKENS);
    }

    /// Zero is a legitimate (if unusual) request — "run prefill only,
    /// generate nothing" — and must not be confused with "unset".
    #[test]
    fn resolve_max_new_tokens_honours_an_explicit_zero() {
        assert_eq!(resolve_max_new_tokens(Some(0)), 0);
    }

    #[test]
    fn resolve_eos_token_id_prefers_im_end_over_everything_else() {
        assert_eq!(resolve_eos_token_id(Some(151_645), Some(999), Some(1)), Some(151_645));
    }

    #[test]
    fn resolve_eos_token_id_falls_back_to_gguf_metadata_when_im_end_is_absent() {
        assert_eq!(resolve_eos_token_id(None, Some(999), Some(1)), Some(999));
    }

    #[test]
    fn resolve_eos_token_id_falls_back_to_endoftext_last() {
        assert_eq!(resolve_eos_token_id(None, None, Some(151_643)), Some(151_643));
    }

    #[test]
    fn resolve_eos_token_id_is_none_when_nothing_is_available() {
        assert_eq!(resolve_eos_token_id(None, None, None), None);
    }

    /// `(im_end, gguf_eos_metadata, endoftext, expected)`.
    type EosPriorityCase = (Option<u32>, Option<u32>, Option<u32>, Option<u32>);

    /// Table-test covering every priority ordering explicitly, so a future
    /// reordering of the `.or(...)` chain in [`resolve_eos_token_id`] is
    /// caught even if it still happens to pass the individual cases above.
    #[test]
    fn resolve_eos_token_id_priority_table() {
        let cases: &[EosPriorityCase] = &[
            (Some(1), Some(2), Some(3), Some(1)),
            (None, Some(2), Some(3), Some(2)),
            (None, None, Some(3), Some(3)),
            (None, None, None, None),
            (Some(1), None, None, Some(1)),
            (Some(1), None, Some(3), Some(1)),
        ];

        for &(im_end, gguf_eos, endoftext, expected) in cases {
            assert_eq!(
                resolve_eos_token_id(im_end, gguf_eos, endoftext),
                expected,
                "im_end={im_end:?} gguf_eos={gguf_eos:?} endoftext={endoftext:?}"
            );
        }
    }
}

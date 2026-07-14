//! Builds a [`BpeTokenizer`] directly from a GGUF file's embedded
//! vocabulary — GGUF's `tokenizer.ggml.*` metadata keys, not a companion
//! `tokenizer.json`.
//!
//! # Why this lives in `kopitiam-runtime`, not `kopitiam-tokenizer`
//!
//! `kopitiam-tokenizer`'s own docs say it plainly (see
//! `crates/kopitiam-tokenizer/src/loader.rs`): "`kopitiam-loader`'s GGUF
//! path does *not* go through this module — GGUF embeds its vocab as
//! metadata arrays with no JSON or byte-mapping involved, so it should call
//! [`BpeTokenizer::from_vocab_and_merges`] directly with bytes it has
//! already decoded itself." `kopitiam-loader` hands back
//! `tokenizer.ggml.*` as an untyped [`GgufValue`] bag (it has no
//! `kopitiam-tokenizer` dependency and should not grow one just to parse
//! its own metadata); `kopitiam-runtime` is the first crate downstream of
//! both, so this is where "GGUF's embedded vocab" becomes a real
//! [`BpeTokenizer`].
//!
//! # The GGUF vocab's own byte-to-unicode mapping
//!
//! A GGUF file with `tokenizer.ggml.model == "gpt2"` (every Qwen/LLaMA-BPE
//! export) stores its `tokenizer.ggml.tokens` and `tokenizer.ggml.merges`
//! arrays using the *exact same* GPT-2 byte-to-unicode alphabet a
//! HuggingFace `tokenizer.json` uses for the same reason (JSON strings
//! cannot hold raw control bytes — see
//! `kopitiam_tokenizer::byte_map`'s docs). This was confirmed by inspecting
//! `crates/kopitiam-ai/vendor/llama.cpp/models/ggml-vocab-qwen2.gguf`
//! directly: its tokens read `["!", "\"", ..., "Ġ ĠĠĠ", ...]` — printable
//! ASCII maps to itself, and `Ġ` (the mapped space byte) appears exactly
//! where a `tokenizer.json` export of the same vocabulary would have it.
//! So this module reuses [`kopitiam_tokenizer::byte_map::decode_mapped_token`]
//! — the same decoder `kopitiam_tokenizer::loader` uses for
//! `tokenizer.json` — rather than inventing a second decoder for what is,
//! byte for byte, the same alphabet.

use kopitiam_core::{Error, Result};
use kopitiam_loader::{GgufValue, LoadedModel};
use kopitiam_tokenizer::byte_map::decode_mapped_token;
use kopitiam_tokenizer::{BpeTokenizer, Tokenizer};

const FORMAT: &str = "gguf-tokenizer";

/// `llama.cpp`'s `LLAMA_TOKEN_TYPE_CONTROL`: a token that is model-control
/// metadata (`<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, ...) rather than
/// ordinary vocabulary, and therefore must be matched atomically rather
/// than being reachable through BPE merges — see
/// [`kopitiam_tokenizer::specials`]'s docs for why that distinction matters.
/// Confirmed against `ggml-vocab-qwen2.gguf`: ids 151643-151645
/// (`<|endoftext|>`, `<|im_start|>`, `<|im_end|>`) are exactly the three
/// entries in that file's `tokenizer.ggml.token_type` array carrying this
/// value.
const TOKEN_TYPE_CONTROL: i64 = 3;

fn malformed(reason: impl Into<String>) -> Error {
    Error::MalformedModel { format: FORMAT, reason: reason.into() }
}

/// Builds a [`BpeTokenizer`] from `model`'s embedded GGUF vocabulary.
///
/// # Errors
///
/// [`Error::MalformedModel`] if `tokenizer.ggml.tokens` is absent, any
/// entry is not a mapped-alphabet string, `tokenizer.ggml.merges` is
/// present but malformed, or [`BpeTokenizer::from_vocab_and_merges`]
/// itself rejects the decoded vocab (e.g. an incomplete byte-level
/// alphabet).
pub fn tokenizer_from_gguf(model: &LoadedModel) -> Result<BpeTokenizer> {
    let raw = &model.metadata().raw;

    let tokens = raw.get_array("tokenizer.ggml.tokens").ok_or_else(|| {
        malformed("missing \"tokenizer.ggml.tokens\": this model has no embedded GGUF vocabulary")
    })?;
    let vocab_entries: Vec<Vec<u8>> = tokens
        .iter()
        .enumerate()
        .map(|(id, value)| decode_token_value(value, id))
        .collect::<Result<_>>()?;

    let merge_pairs = match raw.get_array("tokenizer.ggml.merges") {
        Some(merges) => merges.iter().enumerate().map(decode_merge_value).collect::<Result<_>>()?,
        None => Vec::new(),
    };

    let mut tokenizer = BpeTokenizer::from_vocab_and_merges(vocab_entries, merge_pairs)?;

    if let Some(types) = raw.get_array("tokenizer.ggml.token_type") {
        for (id, ty) in types.iter().enumerate() {
            if token_type_i64(ty) == Some(TOKEN_TYPE_CONTROL) {
                let content = tokenizer
                    .decode(&[id as u32])
                    .map_err(|_| malformed(format!("control token {id} is not valid UTF-8")))?;
                tokenizer.add_special_token(content, id as u32)?;
            }
        }
    }

    Ok(tokenizer)
}

fn decode_token_value(value: &GgufValue, id: usize) -> Result<Vec<u8>> {
    let s = value
        .as_str()
        .ok_or_else(|| malformed(format!("tokenizer.ggml.tokens[{id}] is not a string")))?;
    decode_mapped_token(s)
        .ok_or_else(|| malformed(format!("tokenizer.ggml.tokens[{id}] ({s:?}) is outside the byte-level alphabet")))
}

fn decode_merge_value((rank, value): (usize, &GgufValue)) -> Result<(Vec<u8>, Vec<u8>)> {
    let s = value
        .as_str()
        .ok_or_else(|| malformed(format!("tokenizer.ggml.merges[{rank}] is not a string")))?;
    let (left, right) = s
        .split_once(' ')
        .ok_or_else(|| malformed(format!("tokenizer.ggml.merges[{rank}] ({s:?}) is not \"left right\"")))?;
    let left = decode_mapped_token(left)
        .ok_or_else(|| malformed(format!("tokenizer.ggml.merges[{rank}]: left side {left:?} is not byte-level-mapped")))?;
    let right = decode_mapped_token(right)
        .ok_or_else(|| malformed(format!("tokenizer.ggml.merges[{rank}]: right side {right:?} is not byte-level-mapped")))?;
    Ok((left, right))
}

/// Widens whichever integer variant `tokenizer.ggml.token_type` was
/// encoded as. `llama.cpp` writes this array as `INT32`, but this module
/// does not assume that — any signed or unsigned integer `GgufValue` is
/// accepted, matching [`GgufValue`]'s own "accept the width the writer
/// chose" philosophy (see that type's docs).
fn token_type_i64(value: &GgufValue) -> Option<i64> {
    value.as_i64().or_else(|| value.as_u64().map(|u| u as i64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// `crates/kopitiam-ai/vendor/llama.cpp/models/ggml-vocab-qwen2.gguf`
    /// is a real, vendored, vocab-only GGUF file (no weight tensors) --
    /// exactly the kind of fixture this crate's task brief calls out as
    /// "good for the tokenizer/loader path, NOT a full model". Using the
    /// actual Qwen2 vocab file (rather than the GPT-2 one also vendored
    /// there) means this test exercises the real byte-mapped tokens,
    /// merges, and CONTROL-type special tokens
    /// (`<|endoftext|>`/`<|im_start|>`/`<|im_end|>`) a real Qwen2 GGUF
    /// export ships, confirmed by direct inspection (see this module's
    /// docs).
    fn qwen2_vocab_path() -> &'static Path {
        Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../kopitiam-ai/vendor/llama.cpp/models/ggml-vocab-qwen2.gguf"
        ))
    }

    #[test]
    fn builds_a_tokenizer_from_the_real_vendored_qwen2_vocab_file() {
        let model = kopitiam_loader::load_model(qwen2_vocab_path()).expect("vendored qwen2 vocab GGUF must parse");
        let tokenizer = tokenizer_from_gguf(&model).expect("must build a tokenizer from a real GGUF vocab");

        assert_eq!(tokenizer.vocab_size(), 151_936);

        // Control tokens confirmed present at these exact ids by direct
        // inspection of the file (see this module's docs).
        assert_eq!(tokenizer.special_token_id("<|endoftext|>"), Some(151_643));
        assert_eq!(tokenizer.special_token_id("<|im_start|>"), Some(151_644));
        assert_eq!(tokenizer.special_token_id("<|im_end|>"), Some(151_645));
    }

    #[test]
    fn encode_decode_round_trips_ordinary_text() {
        let model = kopitiam_loader::load_model(qwen2_vocab_path()).unwrap();
        let tokenizer = tokenizer_from_gguf(&model).unwrap();

        let text = "The quick brown fox jumps over the lazy dog.";
        let ids = tokenizer.encode(text).unwrap();
        assert!(!ids.is_empty());
        assert_eq!(tokenizer.decode(&ids).unwrap(), text);
    }

    #[test]
    fn a_control_token_is_matched_atomically_not_split_by_bpe() {
        let model = kopitiam_loader::load_model(qwen2_vocab_path()).unwrap();
        let tokenizer = tokenizer_from_gguf(&model).unwrap();

        let ids = tokenizer.encode("hello<|im_start|>world").unwrap();
        assert!(ids.contains(&151_644));
        assert_eq!(ids.iter().filter(|&&id| id == 151_644).count(), 1);
        assert_eq!(tokenizer.decode(&ids).unwrap(), "hello<|im_start|>world");
    }

    #[test]
    fn missing_tokens_array_is_a_malformed_model_error() {
        use crate::test_support::synthetic_gguf::{tiny_model_bytes, write_temp_gguf};
        // The synthetic runtime-model fixture has architecture metadata
        // but no embedded tokenizer at all.
        let bytes = tiny_model_bytes();
        let path = write_temp_gguf(&bytes, "tokenizer-missing");
        let model = kopitiam_loader::load_model(&path).unwrap();
        assert!(matches!(tokenizer_from_gguf(&model), Err(Error::MalformedModel { .. })));
    }
}

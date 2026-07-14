//! `prompt -> tokenizer -> forward passes with a KV cache -> sampled token
//! ids -> detokenized text`: the one entry point that ties every other
//! module in this crate together into something a CLI or UI actually
//! calls.
//!
//! # Why streaming, not "wait for the whole completion"
//!
//! A 7B-parameter model doing one `f32` forward pass per token on CPU is
//! slow — seconds per token is realistic, not a pathological case. A
//! caller that gets nothing back until the entire completion is done has
//! no way to distinguish "working" from "hung", which is a broken user
//! experience even though the underlying computation is correct. So
//! [`generate`] takes an `on_token` callback invoked once per newly
//! produced token (its id and its decoded text), rather than only
//! returning a final `String` — a caller that does not care about
//! streaming can pass a no-op closure and use the return value alone.
//!
//! # [`generate`] vs [`generate_with_sampler`]
//!
//! [`generate`] always decodes greedily and is unchanged from Phase 1 —
//! every existing caller (including `kopitiam-ai`'s `LocalAdapter`, which
//! this crate cannot modify the call sites of) keeps compiling and
//! behaving identically. [`generate_with_sampler`] is the Phase 2
//! addition: the same pipeline, parameterized over any
//! [`crate::sampling::Sampler`] — [`crate::sampling::GreedySampler`] or
//! [`crate::sampling::StochasticSampler`] (temperature/top-k/top-p/min-p/
//! repetition penalty — see that module's docs). `generate` is defined as
//! `generate_with_sampler` called with a fresh `GreedySampler`, so the two
//! entry points cannot silently drift apart into two different decoding
//! loops.

use kopitiam_core::Result;
use kopitiam_tokenizer::Tokenizer;

use crate::sampling::{GreedySampler, Sampler};
use crate::traits::Model;

/// Generation limits and stop conditions for [`generate`].
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Hard cap on how many new tokens to produce, regardless of whether an
    /// end-of-sequence token is ever sampled. Prevents an unbounded loop
    /// when a model or prompt never naturally reaches its EOS token.
    pub max_new_tokens: usize,
    /// A token id that ends generation immediately when sampled — the
    /// sampled token itself is *not* appended to the output or passed to
    /// `on_token`, matching the usual convention that EOS is control
    /// metadata, not part of the visible completion. Typically
    /// `tokenizer.ggml.eos_token_id` from the model's GGUF metadata.
    pub eos_token_id: Option<u32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self { max_new_tokens: 256, eos_token_id: None }
    }
}

/// Greedily generates a completion for `prompt`.
///
/// Runs one "prefill" forward pass over the whole encoded prompt, then
/// repeatedly samples the highest-scoring next token
/// ([`crate::sampling::GreedySampler`] — see that module's docs for why
/// greedy is what this crate implements today), feeds it back through
/// `model` one token at a time (each call appending to the same
/// [`crate::kv_cache::KvCache`], obtained via
/// [`Model::new_cache`]), and stops after `config.max_new_tokens` tokens
/// or upon sampling `config.eos_token_id`, whichever comes first.
///
/// Calls `on_token(id, text)` once per generated token, in order, *before*
/// that token is folded into the running completion — so a caller can
/// render tokens as they arrive instead of waiting for the whole
/// completion (see this module's docs). Returns the full completion text
/// (every generated token, decoded together so multi-token Unicode
/// characters join correctly — see [`kopitiam_tokenizer::Tokenizer::decode`]'s
/// docs on why decoding token-by-token can split a character but decoding
/// the whole sequence at the end never does).
///
/// # Errors
///
/// Propagates any [`kopitiam_core::Error`] from tokenizing, from a forward
/// pass (including [`kopitiam_core::Error::IndexOutOfBounds`] if generation
/// would exceed the model's context window — see
/// [`crate::kv_cache::KvCache::append`]), or from decoding.
pub fn generate<M: Model>(
    model: &M,
    tokenizer: &dyn Tokenizer,
    prompt: &str,
    config: &GenerationConfig,
    on_token: impl FnMut(u32, &str),
) -> Result<String> {
    generate_with_sampler(model, tokenizer, prompt, config, &mut GreedySampler, on_token)
}

/// Identical to [`generate`] except the token-selection strategy is
/// pluggable: any [`Sampler`] impl (greedy, or a
/// [`crate::sampling::StochasticSampler`] configured for temperature/
/// top-k/top-p/min-p/repetition-penalty sampling) drives which token is
/// picked at every step, instead of always `argmax`. Every other detail —
/// prefill, one-token-at-a-time KV-cache decoding, EOS handling, streaming
/// via `on_token`, the returned completion text — is exactly [`generate`]'s
/// behaviour, because `generate` is defined in terms of this function; see
/// this module's docs for why that direction of composition (not the
/// reverse) is what keeps the two entry points from drifting apart.
///
/// `sampler` is `&mut dyn Sampler` (a trait object) rather than a generic
/// `S: Sampler` type parameter so a caller already holding a
/// `Box<dyn Sampler>` or an `&mut dyn Sampler` (e.g. a long-lived session
/// object that picks its sampler at runtime, per request) can pass it
/// straight through without a wrapper; the erased-type call overhead is
/// irrelevant next to one `f32` forward pass per token.
///
/// # Errors
///
/// Identical to [`generate`]'s.
pub fn generate_with_sampler<M: Model>(
    model: &M,
    tokenizer: &dyn Tokenizer,
    prompt: &str,
    config: &GenerationConfig,
    sampler: &mut dyn Sampler,
    mut on_token: impl FnMut(u32, &str),
) -> Result<String> {
    let prompt_ids = tokenizer.encode(prompt)?;
    let mut cache = model.new_cache();
    let mut generated_ids: Vec<u32> = Vec::new();

    if prompt_ids.is_empty() {
        return Ok(String::new());
    }

    let logits = model.forward(&prompt_ids, &mut cache)?;
    let mut next = sampler.sample(&last_row(&logits, model.vocab_size())?);

    for _ in 0..config.max_new_tokens {
        if config.eos_token_id == Some(next) {
            break;
        }
        generated_ids.push(next);
        let token_text = tokenizer.decode(&[next])?;
        on_token(next, &token_text);

        let logits = model.forward(&[next], &mut cache)?;
        next = sampler.sample(&last_row(&logits, model.vocab_size())?);
    }

    tokenizer.decode(&generated_ids)
}

/// Extracts the last row (the newest token's logits) out of a
/// `[seq, vocab_size]` logits tensor as a plain `Vec<f32>`, which is what
/// [`crate::sampling::Sampler::sample`] operates on.
fn last_row(logits: &kopitiam_tensor::Tensor, vocab_size: usize) -> Result<Vec<f32>> {
    let data = logits.to_vec_f32()?;
    let start = data.len() - vocab_size;
    Ok(data[start..].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::QwenModel;
    use crate::test_support::synthetic_gguf::{build, write_temp_gguf, SyntheticModelSpec};
    use kopitiam_tensor::Tensor;

    /// A tiny full byte-level BPE tokenizer (256 base bytes plus two
    /// merges: "ab" and "abc"), mirroring the fixture
    /// `kopitiam_tokenizer::bpe`'s own tests use, so `generate`'s prompt
    /// encoding step has a complete alphabet to work with. Its
    /// vocab_size (258) is what the paired synthetic model below is sized
    /// to, so every token id the tokenizer can produce is a valid
    /// embedding row.
    fn tiny_tokenizer() -> kopitiam_tokenizer::BpeTokenizer {
        let mut vocab: Vec<Vec<u8>> = (0u16..=255).map(|b| vec![b as u8]).collect();
        vocab.push(b"ab".to_vec()); // id 256
        vocab.push(b"abc".to_vec()); // id 257
        let merges = vec![(b"a".to_vec(), b"b".to_vec()), (b"ab".to_vec(), b"c".to_vec())];
        kopitiam_tokenizer::BpeTokenizer::from_vocab_and_merges(vocab, merges).unwrap()
    }

    fn model_matching_tokenizer() -> QwenModel {
        let spec = SyntheticModelSpec { vocab_size: 258, ..SyntheticModelSpec::default() };
        let bytes = build(&spec);
        let path = write_temp_gguf(&bytes, "generate-e2e");
        let loaded = kopitiam_loader::load_model(&path).unwrap();
        QwenModel::from_loaded_model(&loaded).unwrap()
    }

    #[test]
    fn generate_produces_at_most_max_new_tokens_and_streams_every_one() {
        let model = model_matching_tokenizer();
        let tokenizer = tiny_tokenizer();
        let config = GenerationConfig { max_new_tokens: 5, eos_token_id: None };

        let mut streamed: Vec<u32> = Vec::new();
        let text = generate(&model, &tokenizer, "abc", &config, |id, _text| streamed.push(id)).unwrap();

        assert!(streamed.len() <= 5);
        assert!(!streamed.is_empty(), "greedy decoding with no EOS configured must run the full budget");
        assert_eq!(streamed.len(), 5);
        // The returned text must be exactly the concatenation of every
        // streamed token's own decode, not something else.
        assert_eq!(text, tokenizer_decode_all(&tokenizer, &streamed));
    }

    fn tokenizer_decode_all(tokenizer: &kopitiam_tokenizer::BpeTokenizer, ids: &[u32]) -> String {
        tokenizer.decode(ids).unwrap()
    }

    /// Greedy decoding from a fixed model and a fixed prompt is
    /// deterministic (see `crate::model::tests::decoding_with_a_kv_cache_...`
    /// for the underlying KV-cache property this relies on), so sampling
    /// the very first generated token, then re-running with that token id
    /// as `eos_token_id`, must stop generation before any token is
    /// streamed or appended to the output.
    #[test]
    fn an_eos_token_stops_generation_before_it_is_emitted() {
        let model = model_matching_tokenizer();
        let tokenizer = tiny_tokenizer();
        let unbounded = GenerationConfig { max_new_tokens: 1, eos_token_id: None };

        let mut first_token = None;
        generate(&model, &tokenizer, "abc", &unbounded, |id, _| first_token = Some(id)).unwrap();
        let first_token = first_token.expect("greedy decoding must produce a first token");

        let with_eos = GenerationConfig { max_new_tokens: 10, eos_token_id: Some(first_token) };
        let mut streamed = Vec::new();
        let text = generate(&model, &tokenizer, "abc", &with_eos, |id, _| streamed.push(id)).unwrap();

        assert!(streamed.is_empty(), "the EOS token itself must never be streamed");
        assert_eq!(text, "");
    }

    #[test]
    fn an_empty_prompt_generates_nothing() {
        let model = model_matching_tokenizer();
        let tokenizer = tiny_tokenizer();
        let config = GenerationConfig { max_new_tokens: 5, eos_token_id: None };
        let mut calls = 0;
        let text = generate(&model, &tokenizer, "", &config, |_, _| calls += 1).unwrap();
        assert_eq!(calls, 0);
        assert_eq!(text, "");
    }

    #[test]
    fn last_row_extracts_the_final_positions_logits() {
        let logits = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]).unwrap();
        assert_eq!(last_row(&logits, 2).unwrap(), vec![5.0, 6.0]);
    }

    #[test]
    fn zero_max_new_tokens_generates_nothing_but_still_runs_prefill() {
        let model = model_matching_tokenizer();
        let tokenizer = tiny_tokenizer();
        let config = GenerationConfig { max_new_tokens: 0, eos_token_id: None };
        let mut calls = 0;
        let text = generate(&model, &tokenizer, "abc", &config, |_, _| calls += 1).unwrap();
        assert_eq!(calls, 0);
        assert_eq!(text, "");
    }

    // -- generate_with_sampler --

    #[test]
    fn generate_with_a_greedy_sampler_matches_generates_own_output() {
        // `generate` is defined as `generate_with_sampler` plus a fresh
        // `GreedySampler` -- this pins that equivalence at the public API
        // level, not just by reading the implementation.
        use crate::sampling::GreedySampler;
        let model = model_matching_tokenizer();
        let tokenizer = tiny_tokenizer();
        let config = GenerationConfig { max_new_tokens: 6, eos_token_id: None };

        let via_generate = generate(&model, &tokenizer, "abc", &config, |_, _| {}).unwrap();
        let via_sampler =
            generate_with_sampler(&model, &tokenizer, "abc", &config, &mut GreedySampler, |_, _| {}).unwrap();
        assert_eq!(via_generate, via_sampler);
    }

    #[test]
    fn generate_with_sampler_and_a_fixed_seed_is_reproducible_end_to_end() {
        use crate::sampling::{SamplingConfig, StochasticSampler};
        let model = model_matching_tokenizer();
        let tokenizer = tiny_tokenizer();
        let config = GenerationConfig { max_new_tokens: 8, eos_token_id: None };

        let run = || {
            let mut sampler = StochasticSampler::new(SamplingConfig {
                temperature: 1.0,
                top_k: Some(5),
                seed: 7,
                ..SamplingConfig::default()
            });
            generate_with_sampler(&model, &tokenizer, "abc", &config, &mut sampler, |_, _| {}).unwrap()
        };

        assert_eq!(run(), run(), "the same seed must reproduce the exact same completion end to end");
    }

    #[test]
    fn generate_with_sampler_temperature_zero_matches_greedy_generate_end_to_end() {
        use crate::sampling::{SamplingConfig, StochasticSampler};
        let model = model_matching_tokenizer();
        let tokenizer = tiny_tokenizer();
        let config = GenerationConfig { max_new_tokens: 6, eos_token_id: None };

        let greedy_text = generate(&model, &tokenizer, "abc", &config, |_, _| {}).unwrap();

        let mut sampler = StochasticSampler::new(SamplingConfig { temperature: 0.0, ..SamplingConfig::default() });
        let stochastic_text =
            generate_with_sampler(&model, &tokenizer, "abc", &config, &mut sampler, |_, _| {}).unwrap();

        assert_eq!(
            greedy_text, stochastic_text,
            "temperature=0.0 must reproduce plain greedy decoding through the full generate loop, not just per-call"
        );
    }
}

//! **Kopitiam Runtime**: the Qwen-family transformer forward pass, running
//! entirely in Rust on CPU, offline.
//!
//! This crate is the payoff of the layers beneath it. `kopitiam-core`
//! defines the shared vocabulary ([`kopitiam_core::DType`],
//! [`kopitiam_core::Shape`]); `kopitiam-tensor` implements the CPU tensor
//! and its ops (matmul, softmax, RMSNorm, SiLU, embedding gather, GGUF
//! block-quantization decoding); `kopitiam-loader` parses GGUF/SafeTensors
//! files into raw bytes plus dtype/shape metadata; `kopitiam-tokenizer`
//! turns text into token ids and back. None of those crates constructs a
//! model or runs a forward pass — this one does.
//!
//! # Pipeline
//!
//! ```text
//! kopitiam_loader::load_model(path)
//!   -> LoadedModel
//!   -> QwenModel::from_loaded_model         (crate::model)
//!        - QwenConfig::from_metadata        (crate::config)
//!        - ModelWeights::load               (crate::weights, via crate::bridge)
//!        - RotaryEmbedding::new             (crate::rope)
//!   -> generate(&model, &tokenizer, prompt, ...)   (crate::generate)
//!        per new token:
//!          embedding lookup                  (Tensor::gather_rows)
//!          per layer: block_forward           (crate::block)
//!            RMSNorm -> attention_forward     (crate::attention)
//!              RoPE (split-half)              (crate::rope)
//!              grouped-query KV repeat        (crate::attention::repeat_kv_heads)
//!              causal mask + softmax
//!              KV cache read/append           (crate::kv_cache)
//!            RMSNorm -> swiglu_mlp            (crate::mlp)
//!          final RMSNorm -> output projection
//!          [optional] constraint mask -> -inf  (crate::constraint)
//!          greedy sampling                    (crate::sampling)
//! ```
//!
//! # Module map
//!
//! * [`bridge`] — the loader/tensor glue: bytes + dtype + shape -> `Tensor`.
//!   `kopitiam-loader` and `kopitiam-tensor` were built concurrently and
//!   deliberately do not depend on each other; this crate is the first one
//!   downstream of both, so this is where that gap is bridged.
//! * [`config`] — [`config::QwenConfig`], the architecture hyperparameters
//!   resolved (with documented fallbacks and validation) from
//!   [`kopitiam_loader::ModelMetadata`].
//! * [`rope`] — rotary position embeddings, split-half ("NEOX") pairing.
//!   Read this module's docs before touching anything position-related; a
//!   swapped pairing convention is this crate's single easiest place to
//!   introduce silent, undetectable-by-type-system wrongness.
//! * [`kv_cache`] — the per-layer, growable key/value cache that makes
//!   autoregressive decoding linear instead of quadratic in sequence
//!   length.
//! * [`attention`] — grouped-query causal self-attention: repeating shared
//!   KV heads across their query-head group, causal masking, and wiring
//!   RoPE and the KV cache into one attention forward pass.
//! * [`mlp`] — the SwiGLU feed-forward block.
//! * [`linear`] — the single `x @ W^T + b` helper every projection in this
//!   crate goes through.
//! * [`block`] — one pre-norm transformer block (attention sub-layer, MLP
//!   sub-layer, both with a residual connection).
//! * [`weights`] — loads every named GGUF weight tensor a block/model
//!   needs.
//! * [`model`] — [`model::QwenModel`]: wires embedding, every block, the
//!   final norm, and the (possibly tied) output projection into a
//!   [`traits::Model`] implementation.
//! * [`traits`] — [`traits::Model`] and [`traits::Backend`], the Model
//!   Runtime boundary every layer above this crate should code against.
//! * [`sampling`] — turning a row of logits into a token id:
//!   [`sampling::GreedySampler`] (`argmax`) and
//!   [`sampling::StochasticSampler`] (temperature/top-k/top-p/min-p/
//!   repetition penalty, composed as a pipeline and driven by a seeded
//!   PRNG — see that module's docs for the pipeline shape and why
//!   seedability is mandatory, not optional).
//! * [`constraint`] — grammar-constrained decoding (the keystone): mask the
//!   tokens a [`constraint::TokenConstraint`] forbids to `-inf` at the *front*
//!   of the sampling path, *before* temperature/top-k/top-p, so the model
//!   physically cannot emit invalid structure. Ships a fixed allowed-token-set
//!   ([`constraint::AllowedTokens`], e.g. a tool-name enum) and a structural
//!   JSON constraint ([`constraint::JsonStructure`]); [`constraint::mask_logits`]
//!   is the masking step and [`constraint::ConstrainedSampler`] the drop-in
//!   sampler wrapper, driven end to end by [`generate::generate_constrained`].
//!   See that module's docs for why mask-before-sample and `-inf`-not-`0.0`
//!   (AID-0045).
//! * [`gguf_tokenizer`] — builds a [`kopitiam_tokenizer::BpeTokenizer`]
//!   directly from a GGUF file's embedded `tokenizer.ggml.*` vocabulary
//!   (no companion `tokenizer.json` needed).
//! * [`generate`] — the end-to-end entry point:
//!   `prompt -> tokens -> forward passes -> sampled ids -> text`, with
//!   streaming per-token output.
//!
//! # What is here as of Phase 2, and what is still deliberately not
//!
//! As of Phase 2: stochastic sampling ([`sampling::StochasticSampler`] —
//! temperature/top-k/top-p/min-p/repetition penalty) alongside greedy, and
//! a fused quantized matmul for `Q4_0`/`Q8_0` matmul-operand weights (see
//! [`kopitiam_tensor::Tensor::quantized_matmul`] and
//! [`bridge::load_matmul_weight`] — weights whose on-disk dtype is
//! quantized now stay quantized in memory instead of being eagerly
//! dequantized to `f32`, which is what makes a multi-gigabyte model's
//! resident memory footprint match its on-disk size rather than balloon by
//! 4-8x).
//!
//! Still not here: no batching across multiple concurrent prompts; no
//! scheduler or execution graph (see the parent epic, `kopitiam-082`,
//! Phase 3, and this crate's benchmark module for why a general graph
//! engine was judged not to earn its keep yet); no SIMD. "Correct before
//! fast" per this workspace's engineering principles remains the ordering
//! principle — every fast path added so far (quantized matmul) ships
//! alongside, and is tested against, the plain reference it replaces.

mod attention;
mod block;
mod bridge;
mod config;
mod constraint;
mod gguf_tokenizer;
mod generate;
mod kv_cache;
mod linear;
mod mlp;
mod model;
mod rope;
mod sampling;
mod weights;

#[cfg(test)]
mod test_support;

pub mod traits;

pub use bridge::{load_matmul_weight, load_matmul_weight_opt, load_tensor_f32, load_tensor_f32_opt, tensor_from_entry};
pub use config::QwenConfig;
pub use constraint::{
    AllowedSet, AllowedTokens, ConstrainedSampler, ConstraintError, DecodeState, JsonStructure, SliceVocab,
    TokenConstraint, TokenVocab, mask_logits,
};
pub use generate::{ConstrainedGenerateError, GenerationConfig, generate, generate_constrained, generate_with_sampler};
pub use gguf_tokenizer::tokenizer_from_gguf;
pub use kv_cache::KvCache;
pub use model::QwenModel;
pub use rope::RotaryEmbedding;
pub use sampling::{GreedySampler, Sampler, SamplingConfig, StochasticSampler, greedy_argmax};
pub use traits::{Backend, CpuBackend, Model};

// Re-exported for ergonomics: every public signature in this crate already
// names these types (`QwenConfig` fields, `Model::forward`'s `Result`), so
// callers otherwise need a second `use` line from `kopitiam_core` just to
// name what this crate hands them — the same convention
// `kopitiam-tensor`/`kopitiam-loader` already follow at their own crate
// boundaries.
pub use kopitiam_core::{DType, Device, Error, Result, Shape};

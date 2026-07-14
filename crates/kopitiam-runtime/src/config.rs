//! Architecture hyperparameters for a Qwen-family (LLaMA-shaped) decoder-only
//! transformer, resolved from [`kopitiam_loader::ModelMetadata`].
//!
//! `ModelMetadata` deliberately leaves every field `Option` — it has to
//! represent both a fully-specified GGUF export and a bare SafeTensors dump
//! with no architecture metadata at all (see that type's docs). This module
//! is where "this field is missing" becomes either a documented fallback
//! (RoPE theta, RoPE dimension count, normalization epsilon: every Qwen/
//! LLaMA checkpoint agrees on these defaults closely enough that guessing
//! is safer than refusing to load) or a hard error (layer count, head
//! counts, hidden size, vocab size: guessing any of these wrong does not
//! fail to load, it loads a model that computes silent nonsense, which is
//! strictly worse than refusing).

use kopitiam_core::{Error, Result};
use kopitiam_loader::ModelMetadata;

/// `format` tag used on every [`Error::MalformedModel`] this module raises,
/// so a config-resolution failure is distinguishable from a GGUF/SafeTensors
/// parse failure raised by `kopitiam-loader` itself (which always tags its
/// errors `"gguf"` or `"safetensors"`).
const FORMAT: &str = "qwen-config";

/// The resolved shape of a Qwen-family transformer: everything the forward
/// pass needs to know about the model it is about to run, with no further
/// `Option`s or metadata lookups once construction succeeds.
///
/// # Why "Qwen-family" and not "generic transformer config"
///
/// This struct encodes one specific architecture family: pre-norm residual
/// blocks, RMSNorm, rotary position embeddings, grouped-query attention, and
/// a SwiGLU MLP — the LLaMA-derived shape that Qwen, LLaMA itself, Mistral,
/// and most current open decoder-only models share. It is not an attempt at
/// a universal config covering encoder-decoder models, ALiBi, or MoE
/// routing; those are different forward passes, not different values of the
/// same fields, and would be a new config type (and a new [`crate::Model`]
/// impl) rather than more `Option`s bolted onto this one.
#[derive(Debug, Clone, PartialEq)]
pub struct QwenConfig {
    /// Number of transformer blocks.
    pub n_layers: usize,
    /// Number of query attention heads.
    pub n_heads: usize,
    /// Number of key/value attention heads. Equal to `n_heads` for ordinary
    /// multi-head attention; smaller under grouped-query attention (Qwen2's
    /// usual configuration), where each KV head is shared by
    /// `n_heads / n_kv_heads` query heads. See [`crate::attention`].
    pub n_kv_heads: usize,
    /// The model's hidden/embedding width (`d_model`).
    pub hidden_size: usize,
    /// Width of one attention head: `hidden_size / n_heads`. Not an
    /// independent metadata field — GGUF does not record it separately —
    /// but derived here once so every consumer agrees on it.
    pub head_dim: usize,
    /// Width of the SwiGLU MLP's gate/up projections.
    pub ffn_hidden_size: usize,
    /// Vocabulary size — the row count of the token embedding table.
    pub vocab_size: usize,
    /// The context window this model's KV cache should be sized for.
    pub max_context: usize,
    /// RoPE base frequency (`theta`). Defaults to `10000.0`, the value
    /// every LLaMA/Qwen checkpoint has shipped with to date, when the
    /// metadata omits `rope.freq_base`.
    pub rope_theta: f32,
    /// Number of leading dimensions of each head RoPE actually rotates.
    /// Defaults to `head_dim` (full rotary) when the metadata omits
    /// `rope.dimension_count` — which is what every current Qwen2/LLaMA
    /// GGUF export does, since full rotary is their only configuration; the
    /// field exists in the format for architectures (e.g. GPT-NeoX-style
    /// partial rotary) that rotate only a prefix of each head.
    pub rope_dimension_count: usize,
    /// RMSNorm epsilon. Defaults to `1e-6`, the value every current Qwen2
    /// checkpoint uses, when the metadata omits both
    /// `attention.layer_norm_rms_epsilon` and `attention.layer_norm_epsilon`.
    pub norm_eps: f32,
}

impl QwenConfig {
    /// Resolves a [`QwenConfig`] from a loaded model's metadata.
    ///
    /// # Errors
    ///
    /// Returns [`Error::MalformedModel`] if any field with no safe default
    /// (layer count, either head count, hidden size, feed-forward size, or
    /// vocab size) is absent, or if `n_heads` does not evenly divide
    /// `hidden_size` (which would make `head_dim` — and therefore every
    /// weight matrix shape this crate assumes — ill-defined), or if
    /// `n_kv_heads` does not evenly divide `n_heads` (which would make the
    /// grouped-query head-repeat in [`crate::attention`] ill-defined).
    pub fn from_metadata(metadata: &ModelMetadata) -> Result<Self> {
        let n_layers = require(metadata.n_layers, "block_count")?;
        let n_heads = require(metadata.n_heads, "attention.head_count")? as usize;
        // GGUF's own convention (see kopitiam_loader::ModelMetadata::n_kv_heads
        // docs): an absent head_count_kv means "equal to n_heads", i.e.
        // ordinary multi-head attention rather than GQA. That fallback is a
        // modeling decision the loader deliberately leaves to its consumer;
        // this is where it is made.
        let n_kv_heads = metadata.n_kv_heads.map(|v| v as usize).unwrap_or(n_heads);
        let hidden_size = require(metadata.embedding_length, "embedding_length")? as usize;
        let ffn_hidden_size = require(metadata.feed_forward_length, "feed_forward_length")?;
        let vocab_size = require(metadata.vocab_size, "vocab_size (tokenizer.ggml.tokens length)")?;
        let max_context = metadata.context_length.map(|v| v as usize).unwrap_or(4096);

        if n_heads == 0 || !hidden_size.is_multiple_of(n_heads) {
            return Err(malformed(format!(
                "attention.head_count ({n_heads}) must evenly divide embedding_length ({hidden_size})"
            )));
        }
        let head_dim = hidden_size / n_heads;

        if n_kv_heads == 0 || !n_heads.is_multiple_of(n_kv_heads) {
            return Err(malformed(format!(
                "attention.head_count_kv ({n_kv_heads}) must evenly divide attention.head_count ({n_heads})"
            )));
        }

        let rope_theta = metadata.rope_theta.unwrap_or(10_000.0);
        let rope_dimension_count = metadata
            .rope_dimension_count
            .map(|v| v as usize)
            .unwrap_or(head_dim);
        if rope_dimension_count == 0 || rope_dimension_count > head_dim {
            return Err(malformed(format!(
                "rope.dimension_count ({rope_dimension_count}) must be in 1..=head_dim ({head_dim})"
            )));
        }
        // RoPE pairs dimension i with i + rope_dimension_count/2 (see
        // crate::rope), so an odd rotary width has no valid pairing.
        if !rope_dimension_count.is_multiple_of(2) {
            return Err(malformed(format!(
                "rope.dimension_count ({rope_dimension_count}) must be even: RoPE pairs dimension i with i + count/2"
            )));
        }

        let norm_eps = metadata.norm_epsilon.unwrap_or(1e-6);

        Ok(Self {
            n_layers: n_layers as usize,
            n_heads,
            n_kv_heads,
            hidden_size,
            head_dim,
            ffn_hidden_size: ffn_hidden_size as usize,
            vocab_size: vocab_size as usize,
            max_context,
            rope_theta,
            rope_dimension_count,
            norm_eps,
        })
    }

    /// Query heads sharing each key/value head: `n_heads / n_kv_heads`.
    /// `1` for ordinary multi-head attention, `> 1` under GQA.
    pub fn gqa_group_size(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }
}

fn require(field: Option<u64>, key: &str) -> Result<u64> {
    field.ok_or_else(|| malformed(format!("missing required metadata field \"{key}\"")))
}

fn malformed(reason: String) -> Error {
    Error::MalformedModel { format: FORMAT, reason }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_metadata() -> ModelMetadata {
        ModelMetadata {
            n_layers: Some(2),
            n_heads: Some(4),
            n_kv_heads: Some(2),
            embedding_length: Some(16),
            feed_forward_length: Some(32),
            context_length: Some(512),
            vocab_size: Some(100),
            ..Default::default()
        }
    }

    #[test]
    fn resolves_a_complete_config_with_explicit_values() {
        let mut metadata = base_metadata();
        metadata.rope_theta = Some(1_000_000.0);
        metadata.rope_dimension_count = Some(4);
        metadata.norm_epsilon = Some(1e-5);

        let config = QwenConfig::from_metadata(&metadata).unwrap();
        assert_eq!(config.n_layers, 2);
        assert_eq!(config.n_heads, 4);
        assert_eq!(config.n_kv_heads, 2);
        assert_eq!(config.hidden_size, 16);
        assert_eq!(config.head_dim, 4);
        assert_eq!(config.ffn_hidden_size, 32);
        assert_eq!(config.vocab_size, 100);
        assert_eq!(config.max_context, 512);
        assert_eq!(config.rope_theta, 1_000_000.0);
        assert_eq!(config.rope_dimension_count, 4);
        assert_eq!(config.norm_eps, 1e-5);
        assert_eq!(config.gqa_group_size(), 2);
    }

    #[test]
    fn missing_n_kv_heads_falls_back_to_ordinary_multi_head_attention() {
        let mut metadata = base_metadata();
        metadata.n_kv_heads = None;
        let config = QwenConfig::from_metadata(&metadata).unwrap();
        assert_eq!(config.n_kv_heads, config.n_heads);
        assert_eq!(config.gqa_group_size(), 1);
    }

    #[test]
    fn missing_rope_and_norm_fields_use_documented_defaults() {
        let metadata = base_metadata();
        let config = QwenConfig::from_metadata(&metadata).unwrap();
        assert_eq!(config.rope_theta, 10_000.0);
        assert_eq!(config.rope_dimension_count, config.head_dim);
        assert_eq!(config.norm_eps, 1e-6);
    }

    #[test]
    fn missing_required_field_is_rejected() {
        let mut metadata = base_metadata();
        metadata.n_layers = None;
        assert!(matches!(
            QwenConfig::from_metadata(&metadata),
            Err(Error::MalformedModel { .. })
        ));
    }

    #[test]
    fn head_count_that_does_not_divide_hidden_size_is_rejected() {
        let mut metadata = base_metadata();
        metadata.n_heads = Some(3); // 16 is not divisible by 3
        assert!(matches!(
            QwenConfig::from_metadata(&metadata),
            Err(Error::MalformedModel { .. })
        ));
    }

    #[test]
    fn kv_head_count_that_does_not_divide_head_count_is_rejected() {
        let mut metadata = base_metadata();
        metadata.n_kv_heads = Some(3); // 4 heads is not divisible by 3
        assert!(matches!(
            QwenConfig::from_metadata(&metadata),
            Err(Error::MalformedModel { .. })
        ));
    }

    #[test]
    fn odd_rope_dimension_count_is_rejected() {
        let mut metadata = base_metadata();
        metadata.rope_dimension_count = Some(3);
        assert!(matches!(
            QwenConfig::from_metadata(&metadata),
            Err(Error::MalformedModel { .. })
        ));
    }
}

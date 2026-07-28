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

use crate::rope::RopeKind;
use kopitiam_core::{Error, Result};
use kopitiam_loader::ModelMetadata;

/// Architectures this crate's forward pass genuinely executes.
///
/// # Why a whitelist and not "try it and see"
///
/// This crate implements exactly ONE forward pass: pre-norm residual blocks,
/// RMSNorm, rotary position embeddings, grouped-query attention, SwiGLU MLP.
/// Point it at a Gemma, Phi or MoE GGUF and the tensor names may well resolve,
/// a model may well build, and generation may well start — producing confident
/// nonsense with no hint the weights were never runnable.
///
/// That is not hypothetical. `bd-b9x` was a *subtler version of the same
/// failure*: a `llama`-architecture file running with `qwen2`'s RoPE convention
/// produced fluent, plausible, wrong text for weeks. Silent wrongness is worse
/// than a crash, and an architecture the runtime has never executed is the
/// easiest way to get it.
///
/// # Why this list is short
///
/// It names what has actually been *run*, not what looks structurally similar:
///
/// * `llama` — verified end to end on real weights (SmolLM2-360M and -1.7B),
///   with output diffed against `llama.cpp` (`docs/REFERENCE-ORACLE.md`).
/// * `qwen2` — the family this crate was built and unit-tested against.
///
/// Mistral, and most "llama-shaped" derivatives, ship `general.architecture =
/// "llama"` and so are already covered. Anything else should be added only
/// after a real model of that architecture has been run and compared — adding a
/// name because the shape looks right is precisely the reasoning that produced
/// `bd-b9x`.
const SUPPORTED_ARCHITECTURES: &[&str] = &["llama", "qwen2"];

/// Whether this crate can actually execute `architecture`.
///
/// `None` (metadata absent, e.g. a bare SafeTensors dump) is treated as
/// **unsupported**: with no architecture there is no way to know the RoPE
/// convention, and guessing is the bug this gate exists to prevent.
#[must_use]
pub fn is_supported_architecture(architecture: Option<&str>) -> bool {
    architecture.is_some_and(|a| SUPPORTED_ARCHITECTURES.contains(&a))
}

/// Maps a GGUF `general.architecture` string to the RoPE pairing its files
/// were written for, mirroring `llama.cpp`'s `llama_model_rope_type`
/// (`src/llama-model.cpp`).
///
/// The split is not intuitive and cost this project a long debugging session
/// (`bd-b9x`): **`llama` is interleaved, `qwen2` is split-half**, even though
/// HuggingFace's `LlamaAttention` uses split-half in Python — GGUF conversion
/// permutes Q/K so ggml's interleaved rotation reproduces it. See
/// [`crate::rope`]'s module docs for the full story.
///
/// Unknown architectures fall back to [`RopeKind::SplitHalf`], which is this
/// crate's historical behaviour and correct for the Qwen family it was first
/// built against. That default is a guess, and a wrong guess here is silent —
/// so anything newly supported should be added here deliberately, checked
/// against `llama_model_rope_type` rather than assumed.
#[must_use]
pub fn rope_kind_for_architecture(architecture: Option<&str>) -> RopeKind {
    match architecture.unwrap_or_default() {
        // ggml's NORM group ("pairs of consecutive head values").
        "llama" | "llama4" | "deci" | "baichuan" | "starcoder" | "internlm2" | "minicpm"
        | "xverse" | "command-r" | "cohere2" | "olmo" | "arctic" | "deepseek" | "deepseek2" => {
            RopeKind::Interleaved
        }
        // ggml's NEOX group.
        "qwen2" | "qwen2moe" | "qwen3" | "qwen3moe" | "phi2" | "phi3" | "gemma" | "gemma2"
        | "gemma3" | "stablelm" | "olmo2" | "gptneox" => RopeKind::SplitHalf,
        _ => RopeKind::SplitHalf,
    }
}

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
    /// Which RoPE pairing this architecture's GGUF files were written for.
    ///
    /// Derived from `general.architecture`, mirroring `llama.cpp`'s
    /// `llama_model_rope_type` — see [`rope_kind_for_architecture`] and
    /// [`crate::rope::RopeKind`]. Guessing this wrong does not fail; it
    /// silently degrades output more and more as the context grows, which is
    /// what `bd-b9x` turned out to be.
    pub rope_kind: RopeKind,
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

        // Refuse an architecture this crate has never executed, BEFORE building
        // anything. See `SUPPORTED_ARCHITECTURES`: a model that loads and then
        // generates nonsense is strictly worse than one that refuses to load,
        // and `kopitiam-ai`'s adapter already turns this `Err` into a clear
        // user-facing fallback note.
        let architecture = metadata.architecture.as_deref();
        if !is_supported_architecture(architecture) {
            return Err(malformed(format!(
                "unsupported general.architecture {:?}: this runtime implements the dense \
                 pre-norm RoPE/GQA/SwiGLU forward pass only, and executes {SUPPORTED_ARCHITECTURES:?}. \
                 Running other architectures on it produces confident nonsense rather than an \
                 error, so it is refused here instead",
                architecture.unwrap_or("<absent>")
            )));
        }
        let rope_kind = rope_kind_for_architecture(architecture);
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
            rope_kind,
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
            // Every fixture needs a supported architecture now: an absent one
            // means the RoPE convention is unknowable, which `from_metadata`
            // refuses rather than guesses. See `SUPPORTED_ARCHITECTURES`.
            architecture: Some("llama".to_string()),
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

#[cfg(test)]
mod rope_kind_tests {
    use super::*;

    /// The mapping that `bd-b9x` turned on. `llama` really is interleaved and
    /// `qwen2` really is split-half, however counter-intuitive that is next to
    /// HuggingFace's Python (see `crate::rope`'s module docs). Checked against
    /// `llama.cpp`'s `llama_model_rope_type`, not against intuition.
    #[test]
    fn llama_is_interleaved_and_qwen2_is_split_half() {
        assert_eq!(rope_kind_for_architecture(Some("llama")), RopeKind::Interleaved);
        assert_eq!(rope_kind_for_architecture(Some("qwen2")), RopeKind::SplitHalf);
    }

    /// SmolLM2 — the model this bug was found on, and KOPITIAM's default —
    /// reports `general.architecture = "llama"`, so it must get interleaved.
    #[test]
    fn smollm2_reports_llama_and_therefore_gets_interleaved() {
        assert_eq!(rope_kind_for_architecture(Some("llama")), RopeKind::Interleaved);
    }

    /// An unknown architecture falls back to split-half. That is a *guess*, and
    /// a silently-wrong one if the model is really a NORM architecture — which
    /// is exactly how this bug survived. Pinned so the fallback is a decision
    /// someone made, not an accident someone inherited.
    #[test]
    fn an_unknown_architecture_falls_back_to_split_half() {
        assert_eq!(rope_kind_for_architecture(None), RopeKind::SplitHalf);
        assert_eq!(rope_kind_for_architecture(Some("not-a-real-arch")), RopeKind::SplitHalf);
    }
}

#[cfg(test)]
mod architecture_gate_tests {
    use super::*;

    fn metadata_with(arch: Option<&str>) -> ModelMetadata {
        ModelMetadata {
            architecture: arch.map(str::to_string),
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

    /// The two architectures this crate has actually run must keep loading —
    /// the bead's own "what would make this wrong" is a whitelist so tight it
    /// locks out models that work today.
    #[test]
    fn the_architectures_we_actually_execute_still_load() {
        for arch in ["llama", "qwen2"] {
            assert!(
                QwenConfig::from_metadata(&metadata_with(Some(arch))).is_ok(),
                "{arch} must still load"
            );
        }
    }

    /// An architecture with a different forward pass must be REFUSED, not run.
    /// Gemma is the live example: it was the question that opened this bead,
    /// and running it on this crate's forward pass would emit fluent nonsense.
    #[test]
    fn an_architecture_with_a_different_forward_pass_is_refused() {
        for arch in ["gemma", "gemma3", "phi3", "qwen2moe", "mamba"] {
            let err = QwenConfig::from_metadata(&metadata_with(Some(arch)))
                .expect_err("{arch} must be refused");
            assert!(matches!(err, Error::MalformedModel { .. }), "{arch}: {err:?}");
        }
    }

    /// Absent metadata is unsupported, not "probably fine". Without an
    /// architecture the RoPE convention is unknowable, and guessing it wrong is
    /// exactly bd-b9x — fluent, confident, wrong.
    #[test]
    fn an_absent_architecture_is_refused_rather_than_guessed() {
        assert!(!is_supported_architecture(None));
        assert!(QwenConfig::from_metadata(&metadata_with(None)).is_err());
    }

    /// The refusal has to say what is wrong and what IS supported, or the user
    /// just sees "malformed" and goes hunting in the wrong place.
    #[test]
    fn the_refusal_names_the_architecture_and_the_supported_set() {
        let err = QwenConfig::from_metadata(&metadata_with(Some("gemma"))).unwrap_err();
        let text = format!("{err}");
        assert!(text.contains("gemma"), "should name the offender: {text}");
        assert!(text.contains("llama") && text.contains("qwen2"), "should name what works: {text}");
    }
}

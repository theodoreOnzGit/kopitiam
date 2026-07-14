//! Loads every weight tensor a [`crate::model::QwenModel`] needs out of a
//! [`LoadedModel`], by GGUF's standardized tensor names.
//!
//! The seven weight matrices that feed a matmul (`wq`/`wk`/`wv`/`wo`,
//! `ffn_gate`/`ffn_up`/`ffn_down`) load through
//! [`crate::bridge::load_matmul_weight`], which preserves a block-quantized
//! on-disk dtype rather than eagerly dequantizing it — see that function's
//! docs. Everything else (token embeddings, norm weights, attention
//! biases) is not a matmul operand and always loads as plain `f32` via
//! [`crate::bridge::load_tensor_f32`]/[`crate::bridge::load_tensor_f32_opt`],
//! because [`kopitiam_tensor::Tensor::gather_rows`] and elementwise
//! arithmetic both require a non-quantized dtype.

use kopitiam_core::Result;
use kopitiam_loader::LoadedModel;
use kopitiam_tensor::Tensor;

use crate::bridge::{load_matmul_weight, load_matmul_weight_opt, load_tensor_f32, load_tensor_f32_opt};
use crate::config::QwenConfig;

/// One transformer block's weights, named after the GGUF tensors they were
/// loaded from (see the `blk.N.*` naming convention documented in
/// `crates/kopitiam-ai/vendor/ggml/docs/gguf.md`, "Standardized tensor
/// names").
pub(crate) struct LayerWeights {
    pub attn_norm: Tensor,
    pub wq: Tensor,
    pub bq: Option<Tensor>,
    pub wk: Tensor,
    pub bk: Option<Tensor>,
    pub wv: Tensor,
    pub bv: Option<Tensor>,
    pub wo: Tensor,
    pub ffn_norm: Tensor,
    pub w_gate: Tensor,
    pub w_up: Tensor,
    pub w_down: Tensor,
}

impl LayerWeights {
    fn load(model: &LoadedModel, layer: usize) -> Result<Self> {
        let p = |suffix: &str| format!("blk.{layer}.{suffix}");
        Ok(Self {
            attn_norm: load_tensor_f32(model, &p("attn_norm.weight"))?,
            wq: load_matmul_weight(model, &p("attn_q.weight"))?,
            bq: load_tensor_f32_opt(model, &p("attn_q.bias"))?,
            wk: load_matmul_weight(model, &p("attn_k.weight"))?,
            bk: load_tensor_f32_opt(model, &p("attn_k.bias"))?,
            wv: load_matmul_weight(model, &p("attn_v.weight"))?,
            bv: load_tensor_f32_opt(model, &p("attn_v.bias"))?,
            wo: load_matmul_weight(model, &p("attn_output.weight"))?,
            ffn_norm: load_tensor_f32(model, &p("ffn_norm.weight"))?,
            w_gate: load_matmul_weight(model, &p("ffn_gate.weight"))?,
            w_up: load_matmul_weight(model, &p("ffn_up.weight"))?,
            w_down: load_matmul_weight(model, &p("ffn_down.weight"))?,
        })
    }
}

/// Every weight [`crate::model::QwenModel`] needs.
///
/// The seven weight matrices behind [`LayerWeights`]'s `w*` fields, plus
/// [`Self::output_weight`], load through
/// [`crate::bridge::load_matmul_weight`] and so keep whatever dtype the
/// GGUF file stored them in — still block-quantized if the file shipped
/// `Q4_0`/`Q8_0` weights, `f32` otherwise (see that function's docs).
/// [`Self::token_embd`], [`Self::output_norm`], and every attention bias
/// are *not* matmul operands and are always `f32`, dequantized eagerly if
/// necessary, because [`kopitiam_tensor::Tensor::gather_rows`] and
/// elementwise arithmetic both require it.
pub(crate) struct ModelWeights {
    /// `[vocab_size, hidden_size]`. Always `f32` — see this struct's docs.
    pub token_embd: Tensor,
    pub layers: Vec<LayerWeights>,
    pub output_norm: Tensor,
    /// The output (LM head) projection, `[vocab_size, hidden_size]`.
    /// Usually the single largest weight matrix in a Qwen checkpoint
    /// (`vocab_size * hidden_size`), so keeping it quantized when the file
    /// ships it quantized (see this struct's docs) matters disproportionately
    /// for memory.
    ///
    /// # Tied embeddings
    ///
    /// Many models (Qwen2's smaller checkpoints among them) do not ship a
    /// separate `output.weight` tensor at all: the same matrix that maps
    /// token ids to embeddings is reused, transposed, to map final hidden
    /// states back to vocabulary logits ("weight tying", Press & Wolf,
    /// 2017 — it halves the parameter count spent on the embedding/unembed
    /// pair with little quality cost). When `output.weight` is absent from
    /// the file, this field is set to a clone of `token_embd` (an `Arc`
    /// bump, not a data copy — see [`kopitiam_tensor::Tensor::clone`]'s
    /// docs), so [`crate::model::QwenModel::forward`] never has to branch
    /// on whether tying is in effect; it always just uses this field. That
    /// fallback is always `f32` (`token_embd` always is), even if a
    /// hypothetical future quantized-embedding export existed.
    pub output_weight: Tensor,
}

impl ModelWeights {
    pub(crate) fn load(model: &LoadedModel, config: &QwenConfig) -> Result<Self> {
        let token_embd = load_tensor_f32(model, "token_embd.weight")?;
        let layers = (0..config.n_layers).map(|i| LayerWeights::load(model, i)).collect::<Result<_>>()?;
        let output_norm = load_tensor_f32(model, "output_norm.weight")?;
        let output_weight = match load_matmul_weight_opt(model, "output.weight")? {
            Some(w) => w,
            None => token_embd.clone(),
        };
        Ok(Self { token_embd, layers, output_norm, output_weight })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::synthetic_gguf::{build, write_temp_gguf, SyntheticModelSpec};

    #[test]
    fn loads_every_layer_and_ties_embeddings_when_output_weight_is_absent() {
        let spec = SyntheticModelSpec { tie_embeddings: true, ..SyntheticModelSpec::default() };
        let bytes = build(&spec);
        let path = write_temp_gguf(&bytes, "weights-tied");
        let model = kopitiam_loader::load_model(&path).unwrap();
        let config = QwenConfig::from_metadata(model.metadata()).unwrap();

        let weights = ModelWeights::load(&model, &config).unwrap();
        assert_eq!(weights.layers.len(), spec.n_layers);
        assert_eq!(weights.output_weight.shape(), weights.token_embd.shape());
        assert_eq!(
            weights.output_weight.to_vec_f32().unwrap(),
            weights.token_embd.to_vec_f32().unwrap(),
            "tied output weight must have the same values as the embedding table"
        );
        // Every layer has QKV biases, per the default spec.
        assert!(weights.layers[0].bq.is_some());
        assert!(weights.layers[0].bk.is_some());
        assert!(weights.layers[0].bv.is_some());
    }

    #[test]
    fn loads_a_separate_output_weight_when_present() {
        let spec = SyntheticModelSpec { tie_embeddings: false, ..SyntheticModelSpec::default() };
        let bytes = build(&spec);
        let path = write_temp_gguf(&bytes, "weights-untied");
        let model = kopitiam_loader::load_model(&path).unwrap();
        let config = QwenConfig::from_metadata(model.metadata()).unwrap();

        let weights = ModelWeights::load(&model, &config).unwrap();
        // The synthetic builder fills every tensor from the same RNG
        // stream at a different point, so a separately-declared
        // output.weight is (with overwhelming probability) numerically
        // different from the embedding table -- unlike the tied case
        // above, where they are required to match exactly.
        assert_ne!(weights.output_weight.to_vec_f32().unwrap(), weights.token_embd.to_vec_f32().unwrap());
    }

    #[test]
    fn missing_qkv_bias_is_tolerated_when_the_spec_omits_it() {
        let spec = SyntheticModelSpec { with_qkv_bias: false, ..SyntheticModelSpec::default() };
        let bytes = build(&spec);
        let path = write_temp_gguf(&bytes, "weights-no-bias");
        let model = kopitiam_loader::load_model(&path).unwrap();
        let config = QwenConfig::from_metadata(model.metadata()).unwrap();

        let weights = ModelWeights::load(&model, &config).unwrap();
        assert!(weights.layers[0].bq.is_none());
        assert!(weights.layers[0].bk.is_none());
        assert!(weights.layers[0].bv.is_none());
    }

    /// The wiring this struct's docs describe end to end: a GGUF file that
    /// ships its matmul-operand weights as real on-disk `Q8_0` (see
    /// `crate::test_support::synthetic_gguf`) must come out of
    /// `ModelWeights::load` still `Q8_0` -- not silently dequantized to
    /// `f32` -- while the embedding table and norms, which are never
    /// matmul operands, stay `f32` regardless.
    #[test]
    fn matmul_operand_weights_stay_quantized_when_the_file_ships_them_quantized() {
        let spec = SyntheticModelSpec {
            quantize_matmul_weights: true,
            tie_embeddings: false,
            ..SyntheticModelSpec::quantized_benchmark()
        };
        let bytes = build(&spec);
        let path = write_temp_gguf(&bytes, "weights-quantized");
        let model = kopitiam_loader::load_model(&path).unwrap();
        let config = QwenConfig::from_metadata(model.metadata()).unwrap();

        let weights = ModelWeights::load(&model, &config).unwrap();
        assert_eq!(weights.layers[0].wq.dtype(), kopitiam_core::DType::Q8_0);
        assert_eq!(weights.layers[0].wk.dtype(), kopitiam_core::DType::Q8_0);
        assert_eq!(weights.layers[0].wv.dtype(), kopitiam_core::DType::Q8_0);
        assert_eq!(weights.layers[0].wo.dtype(), kopitiam_core::DType::Q8_0);
        assert_eq!(weights.layers[0].w_gate.dtype(), kopitiam_core::DType::Q8_0);
        assert_eq!(weights.layers[0].w_up.dtype(), kopitiam_core::DType::Q8_0);
        assert_eq!(weights.layers[0].w_down.dtype(), kopitiam_core::DType::Q8_0);
        assert_eq!(weights.output_weight.dtype(), kopitiam_core::DType::Q8_0);

        // Never matmul operands, so never quantized, regardless of the file.
        assert_eq!(weights.token_embd.dtype(), kopitiam_core::DType::F32);
        assert_eq!(weights.output_norm.dtype(), kopitiam_core::DType::F32);
        assert_eq!(weights.layers[0].attn_norm.dtype(), kopitiam_core::DType::F32);
        assert_eq!(weights.layers[0].bq.as_ref().unwrap().dtype(), kopitiam_core::DType::F32);
    }

    /// The un-set-flag case: a plain `f32` GGUF file (every existing
    /// fixture before quantized loading existed) must still load every
    /// matmul-operand weight as `f32` -- `load_matmul_weight` is "preserve
    /// whatever the file shipped", not "always quantize".
    #[test]
    fn matmul_operand_weights_stay_f32_when_the_file_ships_them_as_f32() {
        let bytes = build(&SyntheticModelSpec::default());
        let path = write_temp_gguf(&bytes, "weights-unquantized-native");
        let model = kopitiam_loader::load_model(&path).unwrap();
        let config = QwenConfig::from_metadata(model.metadata()).unwrap();

        let weights = ModelWeights::load(&model, &config).unwrap();
        assert_eq!(weights.layers[0].wq.dtype(), kopitiam_core::DType::F32);
    }
}

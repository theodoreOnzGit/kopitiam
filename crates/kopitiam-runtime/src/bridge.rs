//! The loader/tensor bridge: turning a [`LoadedModel`]'s raw bytes into
//! [`Tensor`]s.
//!
//! `kopitiam-loader` and `kopitiam-tensor` were built concurrently and
//! deliberately do not depend on each other (see `kopitiam-loader`'s crate
//! docs, "Why this crate never constructs a `Tensor`"). `LoadedModel` hands
//! back a [`TensorEntry`] (name, [`DType`], [`Shape`]) plus raw on-disk
//! bytes via [`LoadedModel::tensor_bytes`]; this module is the one place in
//! the workspace that combines the two into a real [`Tensor`], because
//! `kopitiam-runtime` is the first crate that depends on both.
//!
//! # Byte order
//!
//! Every numeric format this module decodes (`f32`, raw `f16`/`bf16` bits,
//! `i32`) is read little-endian. GGUF's spec is explicit that little-endian
//! is the only format loaders need to support in practice (see
//! `kopitiam_loader::gguf`'s module docs), and SafeTensors likewise stores
//! tensors little-endian. Block-quantized formats ([`DType::is_quantized`])
//! need no byte-order handling here at all: their bytes are handed to
//! [`Tensor::from_quantized`] unmodified, and [`crate::quant`]-style
//! decoding (inside `kopitiam-tensor`) owns interpreting the block layout.

use kopitiam_core::{DType, Error, Result};
use kopitiam_loader::{LoadedModel, TensorEntry};
use kopitiam_tensor::Tensor;

/// Builds a [`Tensor`] from one [`TensorEntry`]'s bytes, preserving its
/// on-disk [`DType`] exactly (still block-quantized if `entry.dtype` is
/// quantized, still `f16`/`bf16` if the file stored it that way).
///
/// Call [`Tensor::to_dtype`]`(DType::F32)` on the result to get a tensor the
/// rest of `kopitiam-tensor`'s ops (which are `f32`-only, see that crate's
/// docs) can actually compute on; [`load_tensor_f32`] does exactly that in
/// one step; it is what every weight-loading call site in this crate uses.
pub fn tensor_from_entry(model: &LoadedModel, entry: &TensorEntry) -> Result<Tensor> {
    let bytes = model.tensor_bytes(&entry.name)?;
    match entry.dtype {
        DType::F32 => Tensor::from_f32(read_f32_le(bytes), entry.shape.clone()),
        DType::F16 => Tensor::from_f16(read_u16_le(bytes), entry.shape.clone()),
        DType::BF16 => Tensor::from_bf16(read_u16_le(bytes), entry.shape.clone()),
        DType::I8 => Tensor::from_i8(bytes.iter().map(|&b| b as i8).collect(), entry.shape.clone()),
        DType::I32 => Tensor::from_i32(read_i32_le(bytes), entry.shape.clone()),
        DType::Q4_0
        | DType::Q4_1
        | DType::Q5_0
        | DType::Q5_1
        | DType::Q8_0
        | DType::Q2_K
        | DType::Q3_K
        | DType::Q4_K
        | DType::Q5_K
        | DType::Q6_K
        | DType::Q8_K => Tensor::from_quantized(entry.dtype, bytes.to_vec(), entry.shape.clone()),
    }
}

/// Looks up `name` in `model`, builds a [`Tensor`] from its bytes, and
/// dequantizes/upcasts it to `f32` — the dtype every op in `kopitiam-tensor`
/// actually computes in. This is the call every weight-loading site in
/// [`crate::weights`] goes through: model files may ship `f16`, `bf16`, or
/// any of the five GGUF block-quantized formats, and this one function
/// makes the rest of the forward pass indifferent to which.
///
/// # Errors
///
/// [`Error::MissingTensor`] if `name` is not present in `model`; whatever
/// [`Tensor::to_dtype`] would return for a dtype this crate cannot
/// dequantize (today, every [`DType`] this loader can produce *can* be
/// dequantized to `f32`, so this is unreachable in practice, not a real gap
/// — see [`Tensor::to_dtype`]'s docs).
pub fn load_tensor_f32(model: &LoadedModel, name: &str) -> Result<Tensor> {
    let entry = model
        .tensor(name)
        .ok_or_else(|| Error::MissingTensor { name: name.to_string() })?;
    tensor_from_entry(model, entry)?.to_dtype(DType::F32)
}

/// Like [`load_tensor_f32`], but returns `Ok(None)` instead of
/// [`Error::MissingTensor`] when `name` is absent.
///
/// Exists for optional weights: per-projection attention biases (present in
/// Qwen2, absent in plain LLaMA) and the output projection (absent when it
/// is tied to the token embedding — see [`crate::weights::ModelWeights`]'s
/// docs on tied embeddings).
pub fn load_tensor_f32_opt(model: &LoadedModel, name: &str) -> Result<Option<Tensor>> {
    match model.tensor(name) {
        Some(entry) => Ok(Some(tensor_from_entry(model, entry)?.to_dtype(DType::F32)?)),
        None => Ok(None),
    }
}

/// Loads `name` the way every *matmul-operand* weight
/// (`wq`/`wk`/`wv`/`wo`, the SwiGLU MLP's three matrices, and
/// `output.weight` — see [`crate::weights::LayerWeights`] /
/// [`crate::weights::ModelWeights`]) should: keeping a block-quantized
/// on-disk dtype *only when there is a fused kernel to compute on it*, and
/// otherwise dequantizing it to `f32` at load.
///
/// # Why this is a separate function from [`load_tensor_f32`], not a flag
///
/// Every weight in a Qwen checkpoint is *not* interchangeable for this
/// purpose. Token embeddings go through [`kopitiam_tensor::Tensor::gather_rows`]
/// (row-indexed lookup) and norm weights go through elementwise
/// arithmetic — both require [`kopitiam_tensor::DType::is_quantized`] to
/// be false (see that crate's docs on why a quantized tensor cannot be
/// indexed elementwise at all), so those call sites must keep using
/// [`load_tensor_f32`]. Only the seven weight matrices that feed
/// [`crate::linear::linear`] (a matmul) benefit from staying quantized, so
/// this is a distinct, narrowly-named function rather than a boolean
/// parameter that would let a caller accidentally quantize an embedding or a
/// norm.
///
/// # Why only *some* quantized dtypes stay quantized
///
/// [`kopitiam_tensor::Tensor::quantized_matmul`] has a fused,
/// dequantize-free kernel for only [`DType::Q4_0`] and [`DType::Q8_0`]
/// today (see [`kopitiam_tensor::has_fused_matmul_kernel`], the shared
/// predicate this function keys off so the load-time and compute-time
/// decisions cannot drift). A weight in any *other* quantized format —
/// `Q4_1`/`Q5_0`/`Q5_1` or a K-quant (`Q4_K`/`Q5_K`/`Q6_K`/...) — has no
/// fused kernel, so leaving it quantized would dead-end
/// [`crate::linear::linear`] in an [`Error::UnsupportedDType`]. Those are
/// dequantized to `f32` here so the forward pass takes the ordinary
/// [`kopitiam_tensor::Tensor::matmul`] path. This mirrors what happens to
/// `f16`/`bf16` weights (also upcast to `f32`, no half-precision matmul
/// kernel exists yet): `f32` is this crate's fallback for any dtype without
/// a fused kernel. When a GPU backend later fuses more formats, only
/// `has_fused_matmul_kernel` changes and those formats begin staying
/// quantized here automatically.
///
/// # Errors
///
/// [`Error::MissingTensor`] if `name` is not present in `model`.
pub fn load_matmul_weight(model: &LoadedModel, name: &str) -> Result<Tensor> {
    let entry = model
        .tensor(name)
        .ok_or_else(|| Error::MissingTensor { name: name.to_string() })?;
    let tensor = tensor_from_entry(model, entry)?;
    keep_quantized_or_dequantize(tensor)
}

/// Like [`load_matmul_weight`], but returns `Ok(None)` instead of
/// [`Error::MissingTensor`] when `name` is absent — the matmul-operand
/// counterpart to [`load_tensor_f32_opt`], for the same optional weights
/// (per-projection attention biases, `output.weight` when tied to the
/// token embedding).
pub fn load_matmul_weight_opt(model: &LoadedModel, name: &str) -> Result<Option<Tensor>> {
    match model.tensor(name) {
        Some(entry) => {
            let tensor = tensor_from_entry(model, entry)?;
            Ok(Some(keep_quantized_or_dequantize(tensor)?))
        }
        None => Ok(None),
    }
}

/// Keeps `tensor` as-is when its dtype has a fused
/// [`kopitiam_tensor::Tensor::quantized_matmul`] kernel, and otherwise
/// dequantizes/upcasts it to `f32`. The one place [`load_matmul_weight`] and
/// [`load_matmul_weight_opt`] encode the load-time half of the invariant
/// [`kopitiam_tensor::has_fused_matmul_kernel`] anchors.
fn keep_quantized_or_dequantize(tensor: Tensor) -> Result<Tensor> {
    if kopitiam_tensor::has_fused_matmul_kernel(tensor.dtype()) {
        Ok(tensor)
    } else {
        tensor.to_dtype(DType::F32)
    }
}

fn read_f32_le(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().expect("chunks_exact(4)"))).collect()
}

fn read_u16_le(bytes: &[u8]) -> Vec<u16> {
    bytes.chunks_exact(2).map(|c| u16::from_le_bytes(c.try_into().expect("chunks_exact(2)"))).collect()
}

fn read_i32_le(bytes: &[u8]) -> Vec<i32> {
    bytes.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().expect("chunks_exact(4)"))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::synthetic_gguf::{tiny_model_bytes, write_temp_gguf};

    #[test]
    fn tensor_from_entry_round_trips_f32_bytes() {
        let bytes = tiny_model_bytes();
        let path = write_temp_gguf(&bytes, "bridge-round-trip");
        let model = kopitiam_loader::load_model(&path).unwrap();
        let entry = model.tensor("token_embd.weight").unwrap().clone();
        let t = tensor_from_entry(&model, &entry).unwrap();
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.shape(), &entry.shape);
    }

    #[test]
    fn load_tensor_f32_opt_is_none_for_a_missing_name() {
        let bytes = tiny_model_bytes();
        let path = write_temp_gguf(&bytes, "bridge-opt-missing");
        let model = kopitiam_loader::load_model(&path).unwrap();
        assert!(load_tensor_f32_opt(&model, "does.not.exist").unwrap().is_none());
    }

    #[test]
    fn load_tensor_f32_errors_on_a_missing_required_name() {
        let bytes = tiny_model_bytes();
        let path = write_temp_gguf(&bytes, "bridge-required-missing");
        let model = kopitiam_loader::load_model(&path).unwrap();
        assert!(matches!(
            load_tensor_f32(&model, "does.not.exist"),
            Err(Error::MissingTensor { .. })
        ));
    }

    /// A K-quant matmul weight (Q4_K) has no fused
    /// [`kopitiam_tensor::Tensor::quantized_matmul`] kernel, so before this
    /// fix `load_matmul_weight` left it quantized and the first
    /// [`crate::linear::linear`] call dead-ended in an `UnsupportedDType`.
    /// It must now be dequantized to `f32` at load and complete a forward
    /// pass whose output equals dequantize-then-`f32`-matmul.
    #[test]
    fn a_q4_k_matmul_weight_is_dequantized_at_load_and_completes_a_forward_pass() {
        use crate::test_support::synthetic_gguf::{
            arbitrary_q4_k_blocks, single_quantized_weight_gguf, GGML_TYPE_Q4_K,
        };

        // Q4_K super-block = 256 elements, so `in_features` is one super-block
        // per row and each of the 4 rows is exactly one super-block.
        let (out_features, in_features) = (4, 256);
        let bytes = arbitrary_q4_k_blocks(out_features, 0xBEEF);
        let gguf = single_quantized_weight_gguf("test.weight", &[out_features, in_features], GGML_TYPE_Q4_K, &bytes);
        let path = write_temp_gguf(&gguf, "q4k-load");
        let model = kopitiam_loader::load_model(&path).unwrap();

        // The whole point of the fix: a format without a fused kernel is
        // dequantized at load, not kept quantized.
        let w = load_matmul_weight(&model, "test.weight").unwrap();
        assert_eq!(w.dtype(), DType::F32);
        assert!(!w.dtype().is_quantized());
        assert!(!kopitiam_tensor::has_fused_matmul_kernel(DType::Q4_K));

        // The loaded weight is exactly the decode of the same on-disk bytes.
        let entry = model.tensor("test.weight").unwrap().clone();
        let reference_w = tensor_from_entry(&model, &entry).unwrap().to_dtype(DType::F32).unwrap();
        assert_eq!(w.to_vec_f32().unwrap(), reference_w.to_vec_f32().unwrap());

        // A forward pass through `linear` now completes with finite output
        // equal to dequantize-then-f32-matmul (`x @ W^T`).
        let x_vals: Vec<f32> = (0..in_features).map(|i| (i as f32 % 7.0 - 3.0) * 0.1).collect();
        let x = Tensor::from_f32(x_vals, [1, in_features]).unwrap();
        let y = crate::linear::linear(&x, &w, None).unwrap();
        assert_eq!(y.dtype(), DType::F32);
        let y_vals = y.to_vec_f32().unwrap();
        assert_eq!(y_vals.len(), out_features);
        assert!(y_vals.iter().all(|v| v.is_finite()), "forward-pass output must be finite: {y_vals:?}");

        let reference = x.matmul(&reference_w.transpose(0, 1).unwrap()).unwrap().to_vec_f32().unwrap();
        assert_eq!(y_vals, reference);
    }

    /// The no-regression counterpart: a Q4_0 matmul weight — the format the
    /// shipped `qwen2.5-0.5b-instruct-q4_0` model uses — must still be kept
    /// quantized at load and run through the fused kernel, never dequantized.
    #[test]
    fn a_q4_0_matmul_weight_stays_quantized_at_load_and_uses_the_fused_kernel() {
        use crate::test_support::synthetic_gguf::{
            quantize_q4_0_blocks, single_quantized_weight_gguf, GGML_TYPE_Q4_0,
        };

        let (out_features, in_features) = (3, 64); // 2 blocks of 32 per row.
        let data: Vec<f32> = (0..out_features * in_features).map(|i| (i as f32 * 0.037).sin() * 0.5).collect();
        let bytes = quantize_q4_0_blocks(&data);
        let gguf = single_quantized_weight_gguf("test.weight", &[out_features, in_features], GGML_TYPE_Q4_0, &bytes);
        let path = write_temp_gguf(&gguf, "q40-load");
        let model = kopitiam_loader::load_model(&path).unwrap();

        let w = load_matmul_weight(&model, "test.weight").unwrap();
        assert_eq!(w.dtype(), DType::Q4_0, "Q4_0 must stay quantized for the fused kernel");
        assert!(w.dtype().is_quantized());
        assert!(kopitiam_tensor::has_fused_matmul_kernel(w.dtype()));

        // `linear` routes it through the fused `quantized_matmul`, matching a
        // direct fused call bit-for-bit (i.e. no dequantize slipped in).
        let x_vals: Vec<f32> = (0..in_features).map(|i| (i as f32 % 5.0 - 2.0) * 0.1).collect();
        let x = Tensor::from_f32(x_vals, [1, in_features]).unwrap();
        let y = crate::linear::linear(&x, &w, None).unwrap();
        let fused = x.quantized_matmul(&w).unwrap();
        assert_eq!(y.to_vec_f32().unwrap(), fused.to_vec_f32().unwrap());
        assert!(y.to_vec_f32().unwrap().iter().all(|v| v.is_finite()));
    }
}

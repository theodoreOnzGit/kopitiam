//! Kopitiam Runtime: tensor.
//!
//! The CPU tensor type and the inference-time op set (matmul, broadcasting
//! arithmetic, softmax, reductions, normalization, activations, embedding
//! gather) that every layer of a transformer forward pass is built from,
//! plus decoders for the `f16`/`bf16`/GGUF-quantized formats model weights
//! ship in.
//!
//! # Scope: Phase 1, inference only
//!
//! This crate has no autograd, no gradient tape, and no training ops. That
//! is a scope decision, not an oversight: the Kopitiam Runtime's stated
//! purpose (see `docs/ai-decisions/`) is local inference, and every op
//! here is chosen because a forward pass needs it. Concretely, this means:
//!
//! * Every general-purpose op (arithmetic, softmax, reductions,
//!   normalization, activations, [`Tensor::matmul`]) works on `f32` and
//!   rejects other dtypes with [`Error::DTypeMismatch`] (see
//!   [`Tensor::require_dtype`]). The one exception is
//!   [`Tensor::quantized_matmul`], which computes directly on Q4_0/Q8_0
//!   weight blocks — see that method's docs for why a fused quantized
//!   matmul earns an exception to "everything is f32" (memory: a
//!   dequantized 7B Q4_0 model is ~28GB of `f32` vs ~4GB quantized) and for
//!   the correctness gate that keeps it honest against the plain
//!   `to_dtype(F32)` + [`Tensor::matmul`] reference path, which remains
//!   this crate's default and stays permanently, both as the general
//!   fallback for every other op and as that oracle.
//! * There is no backward pass, so there is nothing here resembling a
//!   `requires_grad` flag or a computation graph — `Tensor` is a plain
//!   value type.
//! * [`Tensor::gather_rows`] implements embedding lookup specifically
//!   (indices select whole rows), not PyTorch's general per-element
//!   `gather`; see that method's docs for why the general form is out of
//!   scope.
//! * There is still no *general* `f32 -> quantized` encoder (no
//!   `Tensor::to_dtype(DType::Q4_0)`) — see [`Tensor::to_dtype`]'s docs for
//!   why requantizing weights is a model-export concern out of this
//!   crate's scope. The quantized module's `quantize_row_q8_0` is a
//!   narrower thing: an *activation*-only Q8_0 encoder, private to
//!   [`Tensor::quantized_matmul`]'s implementation, not a public
//!   `Tensor -> Tensor` conversion.
//!
//! # Layering
//!
//! This crate depends only on `kopitiam-core` for the shared vocabulary
//! ([`DType`], [`Shape`], [`Device`], [`Error`]) and re-exports it so
//! downstream crates (`kopitiam-loader`, `kopitiam-runtime`) need not add
//! a direct `kopitiam-core` dependency just to name a dtype:
//!
//! ```text
//! kopitiam-runtime -> { kopitiam-loader, kopitiam-tokenizer } -> kopitiam-tensor -> kopitiam-core
//! ```
//!
//! # Module map
//!
//! * [`Tensor`] — the shape + strided-view + shared-storage tensor type,
//!   and every op, split across `tensor/*.rs` by concern (see that
//!   module's docs for why the split is by Rust module rather than by
//!   flat file, and how that lets internals stay module-private instead of
//!   crate-wide `pub(crate)`).
//! * [`Storage`] — the owned CPU buffer `Tensor` is a view into.
//! * `half` — `f16`/`bf16` <-> `f32` conversion (re-exported as free
//!   functions; useful on their own, e.g. for a loader converting a whole
//!   weight file up front).
//! * `quant` (private) — GGUF block-quantized format decoders, used only
//!   through [`Tensor::to_dtype`].

mod half;
mod quant;
mod storage;
mod tensor;

pub use half::{bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16};
pub use storage::Storage;
pub use tensor::Tensor;

// Re-export the runtime's shared vocabulary so downstream crates can depend
// on `kopitiam-tensor` alone for the common inference-time types, the same
// way `kopitiam-ontology` types flow through the Semantic Runtime's crates.
pub use kopitiam_core::{DType, Device, Error, Result, Shape};

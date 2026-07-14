//! Kopitiam Runtime: native model file loaders.
//!
//! This crate turns a GGUF or SafeTensors file on disk into a
//! format-agnostic description of what it contains: model-level
//! hyperparameters ([`ModelMetadata`]) and a directory of tensors, each
//! described by name, [`kopitiam_core::DType`], [`kopitiam_core::Shape`],
//! and its raw on-disk bytes ([`TensorEntry`], served through
//! [`LoadedModel::tensor_bytes`]).
//!
//! # Why this crate never constructs a `Tensor`
//!
//! `kopitiam-tensor` — the crate that owns the `Tensor` type — is being
//! developed independently of this one. Depending on it here would couple
//! two crates whose APIs are each still settling, for a feature this crate
//! does not need: nothing a loader does requires an actual `Tensor`, only
//! the bytes, dtype and shape needed to build one. So this crate stops one
//! step short: [`LoadedModel::tensor_bytes`] hands back exactly those
//! bytes, and whoever *does* depend on `kopitiam-tensor` combines them with
//! the dtype and shape from [`LoadedModel::tensor`] to build the tensor on
//! their side of the boundary. Neither crate has to know the other's
//! internals for this to work.
//!
//! # Supported formats
//!
//! * **GGUF** ([`GgufLoader`], module `gguf`) — the `llama.cpp`/`ggml`
//!   format. Note its on-disk dimension order is the *reverse* of this
//!   crate's [`kopitiam_core::Shape`] convention; see the `gguf` module
//!   docs before touching anything shape-related there.
//! * **SafeTensors** ([`SafeTensorsLoader`], module `safetensors`) —
//!   Hugging Face's format. Its dimension order already matches
//!   `Shape`'s convention directly.
//!
//! [`load_model`] picks between the two by sniffing file content (GGUF has
//! a magic number; SafeTensors does not, so it is the fallback), so most
//! callers only need that one function.
//!
//! # Memory strategy
//!
//! Model files are memory-mapped where possible rather than read fully
//! into a `Vec<u8>` — see the internal `byte_source` module's doc for why
//! that matters at multi-gigabyte model sizes and when this crate falls
//! back to a plain read instead.
//!
//! # Malformed input
//!
//! Every parser in this crate treats its input as untrusted: truncated
//! files, offsets that point past end-of-file, and absurd counts that
//! would otherwise justify a huge allocation all become
//! [`kopitiam_core::Error::MalformedModel`] rather than a panic. A tensor
//! type or dtype this crate cannot represent becomes
//! [`kopitiam_core::Error::UnsupportedModelFeature`] instead of being
//! silently misdecoded as something else.

mod byte_source;
mod gguf;
mod metadata;
mod model;
mod safetensors;

pub use gguf::GgufLoader;
pub use metadata::{GgufMetadata, GgufValue, ModelMetadata};
pub use model::{LoadedModel, ModelLoader, TensorEntry, load_model};
pub use safetensors::SafeTensorsLoader;

// Re-exported for ergonomics: every public signature in this crate already
// names these types (`TensorEntry::dtype`, `TensorEntry::shape`, fallible
// methods returning `Result`), so callers otherwise need a second `use`
// line from `kopitiam_core` just to name what this crate hands them.
pub use kopitiam_core::{DType, Error, Result, Shape};

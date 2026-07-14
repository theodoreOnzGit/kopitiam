use crate::{DType, Shape};

/// Every way a Kopitiam Runtime operation can fail.
///
/// This is one shared error type across the runtime crates rather than a
/// per-crate error enum. Inference is a single pipeline — a shape mismatch
/// deep in a kernel surfaces to the caller who asked for a token — and
/// threading five `From` conversions through every layer to express that
/// buys nothing but ceremony.
///
/// Note what is *not* here: no `Other(String)` escape hatch. Every variant
/// carries the structured data needed to explain the failure, because
/// "InvalidModel: something went wrong" is exactly the kind of error that
/// costs an afternoon to debug.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: Shape, actual: Shape },

    #[error("shapes {left} and {right} cannot be broadcast together")]
    NotBroadcastable { left: Shape, right: Shape },

    #[error("dtype mismatch: expected {expected}, got {actual}")]
    DTypeMismatch { expected: DType, actual: DType },

    #[error("operation {op} does not support dtype {dtype}")]
    UnsupportedDType { op: &'static str, dtype: DType },

    #[error(
        "cannot index a {dtype} tensor elementwise: it packs {block_size} elements per quantized block"
    )]
    QuantizedElementAccess { dtype: DType, block_size: usize },

    #[error("storage holds {actual} bytes but shape {shape} of {dtype} needs {expected}")]
    StorageTooSmall {
        shape: Shape,
        dtype: DType,
        expected: usize,
        actual: usize,
    },

    #[error("{count} elements is not a whole number of {dtype} blocks ({block_size} per block)")]
    PartialQuantizedBlock {
        dtype: DType,
        count: usize,
        block_size: usize,
    },

    #[error("index {index} is out of bounds for dimension {dim} of length {len}")]
    IndexOutOfBounds { dim: usize, index: usize, len: usize },

    #[error("malformed {format} model file: {reason}")]
    MalformedModel {
        format: &'static str,
        reason: String,
    },

    #[error("{format} model uses unsupported feature: {feature}")]
    UnsupportedModelFeature {
        format: &'static str,
        feature: String,
    },

    #[error("model is missing required tensor {name:?}")]
    MissingTensor { name: String },

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// The runtime's result alias.
pub type Result<T> = std::result::Result<T, Error>;

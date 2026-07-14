//! Native SafeTensors parsing.
//!
//! SafeTensors (Hugging Face's format) is documented informally by its
//! reference implementation at `crates/kopitiam-ai/vendor/safetensors/`;
//! this module is an original Rust implementation studied against that
//! reference (`safetensors/src/tensor.rs`), not a copy of it.
//!
//! # File layout
//!
//! ```text
//! header_len(u64, little-endian)
//! header_json(header_len bytes) -- a JSON object:
//!   {
//!     "__metadata__": { "<key>": "<value>", ... },   // optional, strings only
//!     "<tensor name>": {
//!       "dtype": "F32" | "F16" | "BF16" | "I8" | "I32" | ...,
//!       "shape": [dim0, dim1, ...],
//!       "data_offsets": [start, end]                 // relative to the byte after the header
//!     },
//!     ...
//!   }
//! <raw tensor bytes>
//! ```
//!
//! # Dimension order — no trap here
//!
//! Unlike GGUF (see [`crate::gguf`]'s module docs), SafeTensors' `shape` is
//! already outermost-first, row-major — the same convention NumPy, PyTorch
//! and [`kopitiam_core::Shape`] all use. No reversal is needed; `shape` is
//! read directly into a [`Shape`].
//!
//! # Dtype coverage
//!
//! SafeTensors supports more element types than [`kopitiam_core::DType`]
//! currently models (`U8`, `U16`, `U32`, `I16`, `I64`, `U64`, `F64`,
//! `BOOL`, the `F8_E*` micro-floats). Only the subset `DType` can represent
//! today — `F32`, `F16`, `BF16`, `I8`, `I32` — is accepted; everything else
//! returns [`Error::UnsupportedModelFeature`] naming the unsupported dtype
//! string, rather than being coerced into a same-size type that would
//! silently reinterpret the bytes' meaning.

use std::collections::BTreeMap;
use std::path::Path;

use indexmap::IndexMap;
use kopitiam_core::{DType, Error, Result, Shape};
use serde::Deserialize;

use crate::byte_source::ByteSource;
use crate::metadata::{GgufMetadata, GgufValue, ModelMetadata};
use crate::model::{LoadedModel, ModelLoader, TensorEntry};

const FORMAT: &str = "safetensors";
const HEADER_LEN_BYTES: usize = 8;
const METADATA_KEY: &str = "__metadata__";

fn malformed(reason: impl Into<String>) -> Error {
    Error::MalformedModel { format: FORMAT, reason: reason.into() }
}

fn unsupported(feature: impl Into<String>) -> Error {
    Error::UnsupportedModelFeature { format: FORMAT, feature: feature.into() }
}

/// One tensor's header entry, deserialized directly from its JSON object.
#[derive(Debug, Deserialize)]
struct RawTensorInfo {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: (u64, u64),
}

/// Maps a SafeTensors dtype string to [`DType`]. See the module doc's
/// "Dtype coverage" section for why this list is deliberately short.
fn dtype_from_str(s: &str) -> Result<DType> {
    match s {
        "F32" => Ok(DType::F32),
        "F16" => Ok(DType::F16),
        "BF16" => Ok(DType::BF16),
        "I8" => Ok(DType::I8),
        "I32" => Ok(DType::I32),
        other => Err(unsupported(format!("safetensors dtype {other:?}"))),
    }
}

/// Parses a SafeTensors file, already opened as `source`, into a
/// [`LoadedModel`].
fn parse(source: ByteSource) -> Result<LoadedModel> {
    let bytes = source.as_slice();

    let header_len_bytes = bytes.get(..HEADER_LEN_BYTES).ok_or_else(|| {
        malformed(format!(
            "file is {} bytes, shorter than the {HEADER_LEN_BYTES}-byte header length prefix",
            bytes.len()
        ))
    })?;
    let header_len = u64::from_le_bytes(
        header_len_bytes.try_into().expect("checked slice is exactly 8 bytes"),
    );
    let header_len = usize::try_from(header_len)
        .map_err(|_| malformed(format!("header length {header_len} does not fit in memory")))?;

    // Bounds-check before parsing: a hostile `header_len` (e.g. near
    // `u64::MAX`, truncated to `usize::MAX` above) fails this `get` rather
    // than being handed to `serde_json` as a length to allocate around.
    let data_start = HEADER_LEN_BYTES
        .checked_add(header_len)
        .ok_or_else(|| malformed("header end offset overflows"))?;
    let header_bytes = bytes.get(HEADER_LEN_BYTES..data_start).ok_or_else(|| {
        malformed(format!(
            "declared header length {header_len} extends past end of file ({} bytes)",
            bytes.len()
        ))
    })?;

    // `serde_json::Map` (not `IndexMap`) here: the crate does not enable
    // `indexmap`'s `serde` feature elsewhere, and tensor iteration order
    // from the JSON header carries no semantic meaning worth threading a
    // second dependency feature through to preserve.
    let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(header_bytes)
        .map_err(|e| malformed(format!("header is not valid JSON: {e}")))?;

    let mut raw_metadata = GgufMetadata::new();
    let mut tensors = IndexMap::new();
    let file_len = bytes.len();

    for (name, value) in header {
        if name == METADATA_KEY {
            let entries: BTreeMap<String, String> = serde_json::from_value(value).map_err(|e| {
                malformed(format!("{METADATA_KEY} must map strings to strings: {e}"))
            })?;
            for (k, v) in entries {
                raw_metadata.0.insert(k, GgufValue::String(v));
            }
            continue;
        }

        let info: RawTensorInfo = serde_json::from_value(value).map_err(|e| {
            malformed(format!("tensor {name:?} header entry is malformed: {e}"))
        })?;

        let dtype = dtype_from_str(&info.dtype)?;

        let mut dims = Vec::with_capacity(info.shape.len());
        for d in info.shape {
            let d = usize::try_from(d).map_err(|_| {
                malformed(format!("tensor {name:?} has a dimension ({d}) too large to represent"))
            })?;
            dims.push(d);
        }
        let shape = Shape::new(dims);
        let elem_count = shape.elem_count();

        let expected_len = dtype.storage_bytes(elem_count).ok_or(Error::PartialQuantizedBlock {
            dtype,
            count: elem_count,
            block_size: dtype.block_size(),
        })?;

        let (start, end) = info.data_offsets;
        if start > end {
            return Err(malformed(format!(
                "tensor {name:?} has data_offsets start ({start}) after end ({end})"
            )));
        }
        let declared_len = end - start;
        if declared_len != expected_len as u64 {
            return Err(malformed(format!(
                "tensor {name:?} declares {declared_len} data bytes but its dtype ({dtype}) and shape ({shape}) need {expected_len}"
            )));
        }

        let abs_offset = (data_start as u64).checked_add(start).ok_or_else(|| {
            malformed(format!("tensor {name:?} data offset overflows a u64"))
        })?;
        let abs_end = (data_start as u64).checked_add(end).ok_or_else(|| {
            malformed(format!("tensor {name:?} data end offset overflows a u64"))
        })?;
        if abs_end > file_len as u64 {
            return Err(malformed(format!(
                "tensor {name:?} data range [{abs_offset}, {abs_end}) extends past end of file ({file_len} bytes)"
            )));
        }
        let abs_offset = usize::try_from(abs_offset)
            .map_err(|_| malformed(format!("tensor {name:?} offset does not fit in memory")))?;

        let entry = TensorEntry {
            name: name.clone(),
            dtype,
            shape,
            offset: abs_offset,
            len: expected_len,
        };
        if tensors.insert(name.clone(), entry).is_some() {
            return Err(malformed(format!("duplicate tensor name {name:?}")));
        }
    }

    let metadata = ModelMetadata {
        architecture: None,
        name: None,
        n_layers: None,
        n_heads: None,
        n_kv_heads: None,
        embedding_length: None,
        feed_forward_length: None,
        context_length: None,
        vocab_size: None,
        rope_theta: None,
        rope_dimension_count: None,
        norm_epsilon: None,
        quantization_version: None,
        file_type: None,
        raw: raw_metadata,
    };

    Ok(LoadedModel { metadata, tensors, source, format: FORMAT })
}

/// Parses SafeTensors (Hugging Face) model files. See the module docs for
/// the format and its dtype coverage.
pub struct SafeTensorsLoader;

impl ModelLoader for SafeTensorsLoader {
    fn format_name(&self) -> &'static str {
        FORMAT
    }

    fn probe(&self, bytes: &[u8]) -> bool {
        // SafeTensors has no magic number, so this is a best-effort
        // heuristic rather than a proof: the header length prefix must be
        // in-range for the bytes actually sampled, and (when enough bytes
        // are available to check) the header's first byte must be `{`,
        // since the header is always a JSON object.
        let Some(len_bytes) = bytes.get(..HEADER_LEN_BYTES) else {
            return false;
        };
        let header_len =
            u64::from_le_bytes(len_bytes.try_into().expect("checked slice is exactly 8 bytes"));
        if header_len == 0 {
            return false;
        }
        match bytes.get(HEADER_LEN_BYTES) {
            Some(b'{') => true,
            Some(_) => false,
            // Not enough bytes sampled to see the header's first byte;
            // the length prefix alone is a weak but nonzero signal.
            None => true,
        }
    }

    fn load(&self, path: &Path) -> Result<LoadedModel> {
        let source = ByteSource::open(path)?;
        parse(source)
    }
}

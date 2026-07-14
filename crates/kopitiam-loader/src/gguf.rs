//! Native GGUF parsing.
//!
//! GGUF (the `llama.cpp`/`ggml` model format) is documented in
//! `crates/kopitiam-ai/vendor/ggml/docs/gguf.md`; this module is an
//! original Rust implementation of that spec, not a translation of
//! `vendor/ggml/src/gguf.cpp` — the C++ reference was read for the wire
//! format, not copied for the code.
//!
//! # File layout
//!
//! ```text
//! magic(4="GGUF") version(u32) tensor_count(u64) metadata_kv_count(u64)
//! metadata_kv[metadata_kv_count]
//! tensor_info[tensor_count]
//! <padding to align_offset(position, ALIGNMENT)>
//! tensor_data[]   -- each tensor's bytes at tensor_data_start + info.offset
//! ```
//!
//! # The dimension-order trap
//!
//! GGUF's per-tensor `dimensions` array is ggml's `ne[]`: **fastest-varying
//! dimension first** (`ne[0]` is the contiguous/innermost axis). That is
//! the *opposite* of [`kopitiam_core::Shape`]'s convention — outermost
//! first, row-major, last dimension contiguous — which is also the
//! convention every NumPy/PyTorch-descended tool uses. A 2D weight matrix
//! that a Python export script calls `shape = [n_out, n_in]` is written to
//! a GGUF file as `ne = [n_in, n_out]`.
//!
//! This module reverses `dimensions` while building each [`TensorEntry`]'s
//! [`Shape`] (see [`parse`] below), so every `Shape` this loader ever
//! produces is already in `kopitiam_core`'s outermost-first convention.
//! Getting this backwards does not fail to load — it loads a tensor with
//! its axes silently swapped, which is a correctness bug invisible until
//! inference produces garbage. If you are debugging a GGUF-sourced model
//! that "loads fine but computes nonsense," check this reversal first.
//!
//! # Big-endian GGUF
//!
//! The spec allows big-endian files but "there is no way to determine if a
//! model is big-endian" from the file itself (v3 spec text). This loader
//! only supports little-endian files, which is what every GGUF file
//! actually distributed in the wild uses; a big-endian file will fail
//! bounds/sanity checks (most likely the version or tensor/KV counts will
//! decode to an absurd value) and surface as [`Error::MalformedModel`]
//! rather than a silent misread.
//!
//! # GGUF v1
//!
//! v1 used `uint32` for string/array lengths and the header's tensor/KV
//! counts, where v2 and v3 use `uint64` throughout (the only difference
//! between v2 and v3 is that v3 adds big-endian support at the spec-text
//! level). v1 files are old enough to be effectively extinct, so this
//! loader accepts v2 and v3 (structurally identical for our purposes) and
//! rejects v1 with [`Error::UnsupportedModelFeature`] rather than adding a
//! second, permanently-unexercised parsing path for it.

use std::path::Path;

use indexmap::IndexMap;
use kopitiam_core::{DType, Error, Result, Shape};

use crate::byte_source::ByteSource;
use crate::metadata::{GgufMetadata, GgufValue, ModelMetadata};
use crate::model::{LoadedModel, ModelLoader, TensorEntry};

const FORMAT: &str = "gguf";
const MAGIC: [u8; 4] = *b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;
/// GGUF caps tensor rank at 4 today but reserves room to raise that; 8 is a
/// generous ceiling that rejects obviously-hostile `n_dimensions` values
/// (e.g. a corrupted field decoding to millions) without risking rejecting
/// a legitimate future file.
const MAX_TENSOR_DIMS: u32 = 8;
/// Guards [`read_value`]'s recursion for nested `Array` values. Legitimate
/// GGUF files never nest arrays more than one level deep; this exists so a
/// hostile file cannot exhaust the stack by chaining thousands of
/// one-element `Array`-of-`Array` headers (each only ~12 bytes on disk).
const MAX_ARRAY_NESTING_DEPTH: u32 = 32;

fn malformed(reason: impl Into<String>) -> Error {
    Error::MalformedModel { format: FORMAT, reason: reason.into() }
}

fn unsupported(feature: impl Into<String>) -> Error {
    Error::UnsupportedModelFeature { format: FORMAT, feature: feature.into() }
}

/// A bounds-checked cursor over a GGUF file's bytes.
///
/// Every primitive read goes through [`Cursor::take`], which is the single
/// place that turns "not enough bytes left" into
/// [`Error::MalformedModel`] instead of a slice-index panic. A truncated
/// file, a metadata count that overshoots the real content, or a
/// deliberately hostile file all fail here, gracefully, on whichever read
/// first runs off the end.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn position(&self) -> usize {
        self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| malformed("byte offset overflow while reading"))?;
        let slice = self.bytes.get(self.pos..end).ok_or_else(|| {
            malformed(format!(
                "unexpected end of file: needed {n} bytes at offset {}, file has {} bytes",
                self.pos,
                self.bytes.len()
            ))
        })?;
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn i8(&mut self) -> Result<i8> {
        Ok(self.take(1)?[0] as i8)
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().expect("take(2) returns 2 bytes")))
    }

    fn i16(&mut self) -> Result<i16> {
        Ok(i16::from_le_bytes(self.take(2)?.try_into().expect("take(2) returns 2 bytes")))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("take(4) returns 4 bytes")))
    }

    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().expect("take(4) returns 4 bytes")))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("take(8) returns 8 bytes")))
    }

    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().expect("take(8) returns 8 bytes")))
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().expect("take(4) returns 4 bytes")))
    }

    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().expect("take(8) returns 8 bytes")))
    }

    fn bool(&mut self) -> Result<bool> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(malformed(format!(
                "invalid bool byte {other:#04x} (must be 0x00 or 0x01)"
            ))),
        }
    }

    /// A `gguf_string_t`: a `u64` length prefix followed by that many UTF-8
    /// bytes (not null-terminated).
    fn string(&mut self) -> Result<String> {
        let len = self.u64()?;
        let len = usize::try_from(len)
            .map_err(|_| malformed(format!("string length {len} does not fit in memory")))?;
        // No `Vec::with_capacity(len)` before validating: `take` bounds-checks
        // `len` against the bytes actually remaining first, so a hostile
        // huge length fails on the bounds check below rather than
        // triggering a huge allocation.
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| malformed("metadata string is not valid UTF-8"))
    }
}

/// Reads one typed metadata value, recursing for `Array` (value type `9`).
///
/// `depth` guards against [`MAX_ARRAY_NESTING_DEPTH`]; see that constant's
/// doc for why a recursion cap is needed here specifically.
fn read_value(cursor: &mut Cursor, value_type: u32, depth: u32) -> Result<GgufValue> {
    match value_type {
        0 => Ok(GgufValue::U8(cursor.u8()?)),
        1 => Ok(GgufValue::I8(cursor.i8()?)),
        2 => Ok(GgufValue::U16(cursor.u16()?)),
        3 => Ok(GgufValue::I16(cursor.i16()?)),
        4 => Ok(GgufValue::U32(cursor.u32()?)),
        5 => Ok(GgufValue::I32(cursor.i32()?)),
        6 => Ok(GgufValue::F32(cursor.f32()?)),
        7 => Ok(GgufValue::Bool(cursor.bool()?)),
        8 => Ok(GgufValue::String(cursor.string()?)),
        9 => {
            if depth >= MAX_ARRAY_NESTING_DEPTH {
                return Err(malformed("array nesting exceeds sanity limit"));
            }
            let elem_type = cursor.u32()?;
            let len = cursor.u64()?;
            let len = usize::try_from(len)
                .map_err(|_| malformed(format!("array length {len} does not fit in memory")))?;
            // Reserve a small hint, not `len` itself: `len` is untrusted,
            // and each element read is bounds-checked against the file's
            // real remaining bytes, so a hostile huge `len` fails on the
            // first missing byte instead of pre-allocating gigabytes.
            let mut values = Vec::with_capacity(len.min(1024));
            for _ in 0..len {
                values.push(read_value(cursor, elem_type, depth + 1)?);
            }
            Ok(GgufValue::Array(values))
        }
        10 => Ok(GgufValue::U64(cursor.u64()?)),
        11 => Ok(GgufValue::I64(cursor.i64()?)),
        12 => Ok(GgufValue::F64(cursor.f64()?)),
        other => Err(unsupported(format!("metadata value type id {other}"))),
    }
}

/// Maps a `ggml_type` id to [`DType`].
///
/// Only the types [`kopitiam_core::DType`] can represent are accepted.
/// GGUF/ggml defines many more (the various K-quants, IQ-quants, MXFP4,
/// I16/I64/F64, ...); until `kopitiam-core` grows variants for them, a
/// tensor using one of those ids returns
/// [`Error::UnsupportedModelFeature`] rather than being misread as one of
/// the supported types (a wrong-but-plausible-looking dequantization is far
/// worse than a load failure).
fn dtype_from_ggml_type(ggml_type: u32) -> Result<DType> {
    match ggml_type {
        0 => Ok(DType::F32),
        1 => Ok(DType::F16),
        2 => Ok(DType::Q4_0),
        3 => Ok(DType::Q4_1),
        6 => Ok(DType::Q5_0),
        7 => Ok(DType::Q5_1),
        8 => Ok(DType::Q8_0),
        30 => Ok(DType::BF16),
        other => Err(unsupported(format!("ggml tensor type id {other}"))),
    }
}

/// `align_offset` from the GGUF spec: rounds `offset` up to the next
/// multiple of `alignment`.
fn align_up(offset: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 {
        return Err(malformed("general.alignment must not be zero"));
    }
    let remainder = offset % alignment;
    if remainder == 0 {
        return Ok(offset);
    }
    offset
        .checked_add(alignment - remainder)
        .ok_or_else(|| malformed("alignment padding overflows a u64 offset"))
}

/// A tensor's on-disk description before its final, GGUF-independent
/// [`TensorEntry`] is built — dimensions here are still in ggml's
/// fastest-varying-first order and `offset` is still relative to
/// `tensor_data_start` rather than absolute.
struct RawTensorInfo {
    name: String,
    /// ggml `ne[]` order: fastest-varying dimension first.
    dims: Vec<u64>,
    ggml_type: u32,
    /// Relative to the start of the tensor data section.
    relative_offset: u64,
}

/// Parses a GGUF file, already opened as `source`, into a [`LoadedModel`].
fn parse(source: ByteSource) -> Result<LoadedModel> {
    let bytes = source.as_slice();
    let mut cursor = Cursor::new(bytes);

    let magic = cursor.take(4)?;
    if magic != MAGIC {
        return Err(malformed(format!(
            "bad magic {magic:02x?}, expected {MAGIC:02x?} (\"GGUF\")"
        )));
    }

    let version = cursor.u32()?;
    if version != 2 && version != 3 {
        return Err(unsupported(format!(
            "gguf version {version} (only v2 and v3 are supported; see module docs re: v1)"
        )));
    }

    let tensor_count = cursor.u64()?;
    let metadata_kv_count = cursor.u64()?;

    let mut kv_map = IndexMap::new();
    for _ in 0..metadata_kv_count {
        let key = cursor.string()?;
        let value_type = cursor.u32()?;
        let value = read_value(&mut cursor, value_type, 0)?;
        if kv_map.insert(key.clone(), value).is_some() {
            // A duplicate key means either a corrupt file or two
            // conflicting values for the same hyperparameter — silently
            // keeping "whichever came last" could quietly pick the wrong
            // one (e.g. two different `block_count`s), so this is treated
            // as malformed rather than resolved by convention.
            return Err(malformed(format!("duplicate metadata key {key:?}")));
        }
    }
    let kv = GgufMetadata(kv_map);

    let mut raw_infos = Vec::new();
    for _ in 0..tensor_count {
        let name = cursor.string()?;
        let n_dims = cursor.u32()?;
        if n_dims > MAX_TENSOR_DIMS {
            return Err(malformed(format!(
                "tensor {name:?} declares {n_dims} dimensions, more than the {MAX_TENSOR_DIMS} this loader accepts"
            )));
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(cursor.u64()?);
        }
        let ggml_type = cursor.u32()?;
        let relative_offset = cursor.u64()?;
        raw_infos.push(RawTensorInfo { name, dims, ggml_type, relative_offset });
    }

    // Default is 32 per spec ("Some writers may not write the alignment.
    // If the alignment is not specified, assume it is 32."); must be a
    // positive multiple of 8 when present.
    let alignment = kv
        .get_u32("general.alignment")
        .map(u64::from)
        .unwrap_or(DEFAULT_ALIGNMENT);
    if alignment == 0 || !alignment.is_multiple_of(8) {
        return Err(malformed(format!(
            "general.alignment {alignment} must be a positive multiple of 8"
        )));
    }

    let tensor_data_start = align_up(cursor.position() as u64, alignment)?;
    let file_len = bytes.len() as u64;

    let mut tensors = IndexMap::new();
    for info in raw_infos {
        let dtype = dtype_from_ggml_type(info.ggml_type)?;

        // Reverse ggml's fastest-varying-first `ne[]` order into
        // kopitiam_core::Shape's outermost-first order. See the module
        // doc's "dimension-order trap" section.
        let mut dims = Vec::with_capacity(info.dims.len());
        for &d in info.dims.iter().rev() {
            let d = usize::try_from(d).map_err(|_| {
                malformed(format!(
                    "tensor {:?} has a dimension ({d}) too large to represent",
                    info.name
                ))
            })?;
            dims.push(d);
        }
        let shape = Shape::new(dims);

        let elem_count = shape.elem_count();
        let byte_len = dtype.storage_bytes(elem_count).ok_or(Error::PartialQuantizedBlock {
            dtype,
            count: elem_count,
            block_size: dtype.block_size(),
        })?;

        if !info.relative_offset.is_multiple_of(alignment) {
            return Err(malformed(format!(
                "tensor {:?} offset {} is not a multiple of alignment {alignment}",
                info.name, info.relative_offset
            )));
        }

        let abs_offset = tensor_data_start.checked_add(info.relative_offset).ok_or_else(|| {
            malformed(format!("tensor {:?} data offset overflows a u64", info.name))
        })?;
        let abs_end = abs_offset.checked_add(byte_len as u64).ok_or_else(|| {
            malformed(format!("tensor {:?} data end offset overflows a u64", info.name))
        })?;
        if abs_end > file_len {
            return Err(malformed(format!(
                "tensor {:?} data range [{abs_offset}, {abs_end}) extends past end of file ({file_len} bytes)",
                info.name
            )));
        }
        let abs_offset = usize::try_from(abs_offset)
            .map_err(|_| malformed(format!("tensor {:?} offset does not fit in memory", info.name)))?;

        let entry = TensorEntry {
            name: info.name.clone(),
            dtype,
            shape,
            offset: abs_offset,
            len: byte_len,
        };
        if tensors.insert(info.name.clone(), entry).is_some() {
            return Err(malformed(format!("duplicate tensor name {:?}", info.name)));
        }
    }

    let metadata = build_metadata(kv);

    Ok(LoadedModel { metadata, tensors, source, format: FORMAT })
}

/// Promotes the well-known `[arch].*` hyperparameter keys into
/// [`ModelMetadata`]'s named fields; everything else stays reachable
/// through `raw`. See [`ModelMetadata`]'s doc for why the field list is
/// what it is.
fn build_metadata(kv: GgufMetadata) -> ModelMetadata {
    let architecture = kv.get_str("general.architecture").map(str::to_owned);
    let name = kv.get_str("general.name").map(str::to_owned);

    // GGUF namespaces most hyperparameters under the architecture name
    // (`llama.block_count`, `qwen2.block_count`, ...), so every lookup
    // below needs the architecture prefix; without one, none of these keys
    // can be resolved and the fields are simply left `None`.
    let prefixed = |suffix: &str| architecture.as_deref().map(|arch| format!("{arch}.{suffix}"));

    let n_layers = prefixed("block_count").and_then(|k| kv.get_u64(&k));
    let n_heads = prefixed("attention.head_count").and_then(|k| kv.get_u64(&k));
    let n_kv_heads = prefixed("attention.head_count_kv").and_then(|k| kv.get_u64(&k));
    let embedding_length = prefixed("embedding_length").and_then(|k| kv.get_u64(&k));
    let feed_forward_length = prefixed("feed_forward_length").and_then(|k| kv.get_u64(&k));
    let context_length = prefixed("context_length").and_then(|k| kv.get_u64(&k));
    let rope_theta = prefixed("rope.freq_base").and_then(|k| kv.get_f32(&k));
    let rope_dimension_count = prefixed("rope.dimension_count").and_then(|k| kv.get_u64(&k));
    let norm_epsilon = prefixed("attention.layer_norm_rms_epsilon")
        .and_then(|k| kv.get_f32(&k))
        .or_else(|| prefixed("attention.layer_norm_epsilon").and_then(|k| kv.get_f32(&k)));

    // GGUF has no dedicated vocab-size key; the vocabulary's length *is*
    // the vocab size. Fall back to an explicit `[arch].vocab_size` key for
    // files that carry one without an embedded tokenizer.
    let vocab_size = kv
        .get_array("tokenizer.ggml.tokens")
        .map(|tokens| tokens.len() as u64)
        .or_else(|| prefixed("vocab_size").and_then(|k| kv.get_u64(&k)));

    let quantization_version = kv.get_u32("general.quantization_version");
    let file_type = kv.get_u32("general.file_type");

    ModelMetadata {
        architecture,
        name,
        n_layers,
        n_heads,
        n_kv_heads,
        embedding_length,
        feed_forward_length,
        context_length,
        vocab_size,
        rope_theta,
        rope_dimension_count,
        norm_epsilon,
        quantization_version,
        file_type,
        raw: kv,
    }
}

/// Parses GGUF (`llama.cpp`/`ggml`) model files. See the module docs for
/// the format and the dimension-order convention this loader corrects for.
pub struct GgufLoader;

impl ModelLoader for GgufLoader {
    fn format_name(&self) -> &'static str {
        FORMAT
    }

    fn probe(&self, bytes: &[u8]) -> bool {
        bytes.len() >= MAGIC.len() && bytes[..MAGIC.len()] == MAGIC
    }

    fn load(&self, path: &Path) -> Result<LoadedModel> {
        let source = ByteSource::open(path)?;
        parse(source)
    }
}

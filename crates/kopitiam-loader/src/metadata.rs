//! Format-agnostic model metadata.
//!
//! GGUF and SafeTensors disagree about almost everything except that both
//! files carry *some* key-value bag of model-level facts alongside the
//! tensors: GGUF has a typed metadata KV store baked into the format;
//! SafeTensors has an optional `__metadata__` entry whose values happen to
//! all be strings. [`GgufValue`] is deliberately expressive enough to
//! represent both without loss, so [`ModelMetadata::raw`] can hold either
//! format's key-value bag through the same type rather than forcing callers
//! to match on a format tag before they can even read a key.
//!
//! The name `GgufValue` reflects where the taxonomy comes from (GGUF is the
//! only one of the two formats with typed, non-string values), not that the
//! type is GGUF-exclusive.

use indexmap::IndexMap;

/// A single metadata value, typed as GGUF's key-value store types it.
///
/// Modeling this as a strongly-typed enum — rather than handing callers the
/// raw bytes and a type tag to `match` on themselves — means "get me
/// `llama.block_count` as a `u64`" is one call ([`GgufMetadata::get_u64`])
/// instead of a `match` arm duplicated at every call site.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    /// A (possibly nested) array of values. GGUF arrays are homogeneous in
    /// principle, but the format does not forbid nesting `Array` inside
    /// `Array`, so this has to allow it too.
    Array(Vec<GgufValue>),
}

impl GgufValue {
    /// Widens to `u64` for every unsigned integer variant, `None` otherwise.
    ///
    /// The GGUF spec changed most count/length fields from `uint32` to
    /// `uint64` between v1 and v2 and explicitly recommends readers accept
    /// both ("Some models may use `uint32` for their values; it is
    /// recommended that readers support both"). Widening here, once,
    /// is how this crate honors that recommendation without every call
    /// site re-deriving it.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U8(v) => Some(v as u64),
            Self::U16(v) => Some(v as u64),
            Self::U32(v) => Some(v as u64),
            Self::U64(v) => Some(v),
            _ => None,
        }
    }

    /// Narrows to `u32`, refusing to silently truncate a `u64` that does not
    /// fit — a truncated `block_count` is a corrupt-looking model, not a
    /// smaller one, so this returns `None` rather than lying.
    pub fn as_u32(&self) -> Option<u32> {
        match *self {
            Self::U8(v) => Some(v as u32),
            Self::U16(v) => Some(v as u32),
            Self::U32(v) => Some(v),
            Self::U64(v) => u32::try_from(v).ok(),
            _ => None,
        }
    }

    /// Widens to `i64` for every signed integer variant, `None` otherwise.
    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            Self::I8(v) => Some(v as i64),
            Self::I16(v) => Some(v as i64),
            Self::I32(v) => Some(v as i64),
            Self::I64(v) => Some(v),
            _ => None,
        }
    }

    /// Narrows to `i32`, refusing to silently truncate.
    pub fn as_i32(&self) -> Option<i32> {
        match *self {
            Self::I8(v) => Some(v as i32),
            Self::I16(v) => Some(v as i32),
            Self::I32(v) => Some(v),
            Self::I64(v) => i32::try_from(v).ok(),
            _ => None,
        }
    }

    /// The exact `f32`, if this value was stored as one. Deliberately does
    /// *not* narrow `F64` to `F32`: unlike the integer widenings above,
    /// float narrowing is lossy in the common case, not just at the edges,
    /// so silently doing it would be more surprising than useful.
    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Self::F32(v) => Some(v),
            _ => None,
        }
    }

    /// Widens to `f64` for either float variant.
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Self::F32(v) => Some(v as f64),
            Self::F64(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Self::Bool(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }
}

/// An ordered key-value metadata bag, with typed getters over [`GgufValue`].
///
/// Preserves the order keys were encountered in the source file
/// ([`indexmap::IndexMap`]) purely so that anything that dumps metadata for
/// a human (a debug CLI command, say) reproduces the file's own ordering
/// instead of an arbitrary hash order — lookups themselves are still O(1).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GgufMetadata(pub(crate) IndexMap<String, GgufValue>);

impl GgufMetadata {
    pub(crate) fn new() -> Self {
        Self(IndexMap::new())
    }

    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.0.get(key)
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get(key)?.as_u64()
    }

    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.get(key)?.as_u32()
    }

    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.get(key)?.as_i64()
    }

    pub fn get_i32(&self, key: &str) -> Option<i32> {
        self.get(key)?.as_i32()
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.get(key)?.as_f32()
    }

    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.get(key)?.as_f64()
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.get(key)?.as_bool()
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key)?.as_str()
    }

    pub fn get_array(&self, key: &str) -> Option<&[GgufValue]> {
        self.get(key)?.as_array()
    }

    /// Number of keys in the bag.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Iterates keys and values in file order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &GgufValue)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v))
    }
}

/// Model-level hyperparameters, promoted out of the raw metadata bag into
/// named fields where the two supported formats (or at least GGUF, which is
/// the only one of the two with a standardized hyperparameter vocabulary)
/// agree on a concept.
///
/// # Why `Option` everywhere
///
/// Every field is optional because this struct has to represent both a
/// fully-specified GGUF LLM export *and* a bare SafeTensors weight dump that
/// carries no architecture metadata at all — SafeTensors' `__metadata__` is
/// a free-form `string -> string` map with no standardized keys. Refusing
/// to load a SafeTensors file just because it lacks `n_heads` would be
/// wrong: the tensors are still perfectly loadable, and it is the
/// architecture-specific consumer (in `kopitiam-runtime`, not here) that
/// knows whether a missing field is fatal for the model it is trying to
/// build.
///
/// # Why these fields specifically
///
/// This is the intersection of GGUF's `[llm].*` standardized keys (see
/// `crates/kopitiam-ai/vendor/ggml/docs/gguf.md`) that essentially every
/// dense transformer architecture needs to reconstruct its shape: layer
/// count, head counts (including the separate KV head count for grouped-
/// query attention), embedding/feed-forward widths, context length,
/// vocabulary size, RoPE parameters, and normalization epsilon. Anything
/// architecture-specific (MoE expert counts, SSM state sizes, ALiBi bias)
/// is deliberately *not* promoted here — it stays reachable through `raw`
/// so this struct does not have to grow a field for every architecture
/// GGUF has ever described.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelMetadata {
    /// `general.architecture` — e.g. `"llama"`, `"qwen2"`, `"gptneox"`.
    /// `None` for formats or files that do not record one.
    pub architecture: Option<String>,
    /// `general.name`, if present.
    pub name: Option<String>,
    /// `[arch].block_count` — number of transformer blocks.
    pub n_layers: Option<u64>,
    /// `[arch].attention.head_count`.
    pub n_heads: Option<u64>,
    /// `[arch].attention.head_count_kv`. Distinct from `n_heads` only under
    /// grouped-query or multi-query attention; `None` means "not recorded",
    /// which the GGUF spec says should be read as "equal to `n_heads`" —
    /// that fallback is a modeling decision for the consumer, not this
    /// loader, so it is left as `None` rather than silently copied here.
    pub n_kv_heads: Option<u64>,
    /// `[arch].embedding_length` — the model's hidden/embedding width.
    pub embedding_length: Option<u64>,
    /// `[arch].feed_forward_length`.
    pub feed_forward_length: Option<u64>,
    /// `[arch].context_length` — the context window the model was trained
    /// for.
    pub context_length: Option<u64>,
    /// Vocabulary size. GGUF has no dedicated key for this; it is derived
    /// from the length of `tokenizer.ggml.tokens` when present.
    pub vocab_size: Option<u64>,
    /// `[arch].rope.freq_base` — the RoPE base frequency (`theta`).
    pub rope_theta: Option<f32>,
    /// `[arch].rope.dimension_count`.
    pub rope_dimension_count: Option<u64>,
    /// Normalization epsilon, from whichever of
    /// `[arch].attention.layer_norm_rms_epsilon` or
    /// `[arch].attention.layer_norm_epsilon` is present (RMS-norm is tried
    /// first, since it is what every current GGUF LLM architecture uses).
    pub norm_epsilon: Option<f32>,
    /// `general.quantization_version`, present when any tensor is
    /// quantized.
    pub quantization_version: Option<u32>,
    /// `general.file_type` — the GGUF enum describing the majority tensor
    /// encoding. Left as the raw `u32` rather than decoded, since it is
    /// advisory ("can be inferred from the tensor types") and this crate
    /// already reports each tensor's real [`kopitiam_core::DType`].
    pub file_type: Option<u32>,
    /// Every metadata key as found in the file, verbatim. This is the
    /// escape hatch for everything not promoted to a named field above:
    /// tokenizer vocabularies, chat templates, MoE/SSM parameters,
    /// community-namespaced keys, and SafeTensors' free-form
    /// `__metadata__` strings.
    pub raw: GgufMetadata,
}

//! # `memory` -- how much VRAM this model going to makan
//!
//! **Upstream:** `fs/ggml/ggml.go` (the `KV` accessors + `GraphSize`) and
//! `fs/ggml/type.go` (the quantisation block/type sizes), from ollama
//! `4713800b08b2ddf5e14acf8398953cf7b12f169b` (2026-07-28), MIT, Copyright (c)
//! Ollama. This is a **port**, not inspiration -- where we and ollama disagree,
//! ollama wins, and every place we deliberately go our own way says so at the
//! point of divergence.
//!
//! ## What this module is for
//!
//! One question: *given this model and these runtime settings, how many bytes?*
//! Three answers come back from [`graph_size`]:
//!
//! * **`kv_cache[i]`** -- bytes of KV cache layer `i` will need. Per-layer, not
//!   one flat number, because a modern model got layers of different shapes:
//!   full-attention layers, sliding-window layers, recurrent (SSM) layers.
//! * **`full_offload`** -- scratch bytes for the compute graph when *every*
//!   layer sits on the GPU.
//! * **`partial_offload`** -- scratch bytes when only *some* layers sit on the
//!   GPU, so tensors must cross the CPU/GPU boundary mid-graph.
//!
//! The scheduler adds these up against free VRAM to decide **how many
//! transformer layers fit**. Get it wrong one way, KOPITIAM leave VRAM idle and
//! run slow; wrong the other way, the allocator OOM and the whole thing die. So
//! this is arithmetic that must be *faithful*, not merely reasonable.
//!
//! ## The seam: KV metadata come in injected, not read from disk
//!
//! Upstream, `GraphSize` is a method on a `GGML` value that owns a decoded GGUF
//! file. Here it is a **pure function**: you hand it a [`Kv`] (the metadata
//! key/value map) and a [`Tensors`] (names + shapes + quant kinds), and it hands
//! back numbers. No file, no reader, no `io`.
//!
//! Why put the seam there hor:
//!
//! * `kopitiam-ollama` deliberately depends on **nothing else in KOPITIAM**, and
//!   the GGUF reader belongs to `kopitiam-loader`. Pulling a reader in here
//!   would invert that.
//! * The layer that decides *whether a model fits on your GPU* must be testable
//!   **without a model file**. Every test below runs on a hand-built [`Kv`] --
//!   no download, no GPU, no network. That is the whole point.
//!
//! What it costs: the caller must fill the [`Kv`] faithfully, including the
//! quirk that `tokenizer.ggml.tokens` carries a **declared array size** which
//! may be larger than the values actually read (upstream truncates big arrays
//! but keeps `size`). [`KvArray`] models that split on purpose -- see
//! [`Kv::vocab_size`].
//!
//! Nothing in `GraphSize` needs a GGUF reader. The one thing a caller cannot
//! fake is the *content* of the metadata, and that is exactly the caller's job.
//!
//! ## Reading the arithmetic
//!
//! The formulas are per-architecture and genuinely different from each other --
//! llama is not gemma3 is not deepseek2 is not gptoss. They came out of
//! upstream's measurements against real models; they are **empirical**, not
//! derived. Where a constant is a fudge factor (`105/128`, `9/16`), the doc says
//! so plainly rather than pretending got a derivation. A number with no source
//! is a bug that has not fired yet, so every one of them names where it came
//! from.
//!
//! All the byte arithmetic runs in [`std::num::Wrapping<u64>`]. That is
//! deliberate: Go's `uint64` wraps silently, so wrapping is the *faithful*
//! semantics, and it also means a rubbish [`Kv`] can never panic a debug build
//! with an overflow. Realistic model shapes are nowhere near `u64::MAX` -- the
//! biggest term is roughly `embedding * vocab * 105`, about 2e14 for a 405B
//! model -- so wraparound is a safety net, not a working condition.

use std::collections::BTreeMap;
use std::num::Wrapping;

use crate::format::{human_bytes2, MEBIBYTE};

/// Shorthand for the wrapping arithmetic used all through [`graph_size`].
///
/// See the module header: Go wraps, so we wrap, and that keeps a rubbish GGUF
/// from panicking a debug build.
#[inline]
const fn w(n: u64) -> Wrapping<u64> {
    Wrapping(n)
}

/// Our own sanity bound on `block_count` (number of transformer layers).
///
/// **Deliberate divergence from upstream.** Go does `make([]uint64, nLayers)`
/// straight from the file's `block_count`, so a rubbish GGUF claiming 4 billion
/// layers makes it try to allocate 32 GiB. `block_count` comes from a `u32` KV
/// entry, so it is fully attacker-controlled when the model file is untrusted.
///
/// 65536 is ours, not upstream's -- no provenance to claim, and it says so. It
/// is chosen to be absurdly generous: the deepest real model anybody ships is
/// around 126 layers (Llama-3.1 405B), so this is ~500x headroom, while still
/// bounding the allocation to 512 KiB.
///
/// **What would make this wrong:** if some future architecture genuinely has
/// more than 65536 blocks, [`graph_size`] would refuse a legitimate model. Raise
/// the bound then -- don't remove it.
pub const MAX_BLOCK_COUNT: u64 = 1 << 16;

/// Things that can go wrong working out a model's memory footprint.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MemoryError {
    /// `tokenizer.ggml.tokens` is missing, or is not an array of strings.
    ///
    /// **Deliberate divergence.** Upstream does a bare Go type assertion
    /// (`f.KV()["tokenizer.ggml.tokens"].(*array[string]).size`) which
    /// **panics** when the key is absent or the wrong type. Library code in
    /// Rust must not panic on bad input, so we return this instead. Same
    /// condition, different delivery.
    #[error("`tokenizer.ggml.tokens` missing or not a string array -- cannot size the vocab graph")]
    MissingVocab,

    /// `block_count` is past [`MAX_BLOCK_COUNT`]; the file is almost certainly
    /// corrupt. See [`MAX_BLOCK_COUNT`] for why this guard exists at all.
    #[error("block_count {block_count} exceeds the sanity bound {max} -- corrupt model metadata?")]
    AbsurdBlockCount {
        /// What the metadata claimed.
        block_count: u64,
        /// The bound it broke ([`MAX_BLOCK_COUNT`]).
        max: u64,
    },
}

// ---------------------------------------------------------------------------
// KV metadata
// ---------------------------------------------------------------------------

/// One GGUF metadata value.
///
/// **Upstream:** the `any` stored in `KV map[string]any` (`fs/ggml/ggml.go:33`),
/// which in practice is one of the scalar Go types or a `*array[T]`.
///
/// The variants are exact about width on purpose. Upstream's `keyValue[T]` does
/// a Go **type assertion**, so `kv.Uint("block_count")` only matches a value
/// stored as `uint32` -- a `u64` sitting under the same key returns the default,
/// not a converted number. We copy that strictness exactly; loosening it would
/// silently change which models get sized correctly.
#[derive(Debug, Clone, PartialEq)]
pub enum KvValue {
    /// `uint8`
    U8(u8),
    /// `int8`
    I8(i8),
    /// `uint16`
    U16(u16),
    /// `int16`
    I16(i16),
    /// `uint32` -- what nearly every model hyperparameter is stored as.
    U32(u32),
    /// `int32`
    I32(i32),
    /// `uint64` -- e.g. `general.parameter_count`.
    U64(u64),
    /// `int64`
    I64(i64),
    /// `float32`
    F32(f32),
    /// `float64`
    F64(f64),
    /// `bool`
    Bool(bool),
    /// UTF-8 string, e.g. `general.architecture`, `tokenizer.chat_template`.
    String(String),
    /// An array. See [`KvArray`] for the declared-size-vs-values wrinkle.
    Array(KvArray),
}

impl From<&str> for KvValue {
    fn from(v: &str) -> Self {
        KvValue::String(v.to_string())
    }
}
impl From<String> for KvValue {
    fn from(v: String) -> Self {
        KvValue::String(v)
    }
}
impl From<u32> for KvValue {
    fn from(v: u32) -> Self {
        KvValue::U32(v)
    }
}
impl From<u64> for KvValue {
    fn from(v: u64) -> Self {
        KvValue::U64(v)
    }
}
impl From<i32> for KvValue {
    fn from(v: i32) -> Self {
        KvValue::I32(v)
    }
}
impl From<f32> for KvValue {
    fn from(v: f32) -> Self {
        KvValue::F32(v)
    }
}
impl From<bool> for KvValue {
    fn from(v: bool) -> Self {
        KvValue::Bool(v)
    }
}
impl From<KvArray> for KvValue {
    fn from(v: KvArray) -> Self {
        KvValue::Array(v)
    }
}

/// The element list of a [`KvArray`], tagged by element type.
///
/// **Upstream:** the `values []T` field of `array[T]` (`fs/ggml/gguf.go:457`).
/// The type tag survives even when the list is empty, because upstream's type
/// assertions distinguish `*array[uint32]` from `*array[int32]` and take
/// different code paths for each.
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)] // every variant is just "a vector of that GGUF element type"
pub enum KvArrayValues {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    Bool(Vec<bool>),
    String(Vec<String>),
}

impl KvArrayValues {
    /// How many elements are actually present. **Not** the declared size --
    /// see [`KvArray::size`].
    pub fn len(&self) -> usize {
        match self {
            KvArrayValues::U8(v) => v.len(),
            KvArrayValues::I8(v) => v.len(),
            KvArrayValues::U16(v) => v.len(),
            KvArrayValues::I16(v) => v.len(),
            KvArrayValues::U32(v) => v.len(),
            KvArrayValues::I32(v) => v.len(),
            KvArrayValues::U64(v) => v.len(),
            KvArrayValues::I64(v) => v.len(),
            KvArrayValues::F32(v) => v.len(),
            KvArrayValues::F64(v) => v.len(),
            KvArrayValues::Bool(v) => v.len(),
            KvArrayValues::String(v) => v.len(),
        }
    }

    /// True when no element was materialised. Usually means the array got
    /// truncated on read, not that the array is genuinely empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A GGUF array value: a **declared size** plus the elements actually read.
///
/// **Upstream:** `array[T]` (`fs/ggml/gguf.go:457`), whose own comment says
/// `values` "is nil if the array is larger than configured maxSize" while
/// `size` stays "the actual size of the array".
///
/// This split is load-bearing, not bookkeeping. The vocabulary array
/// `tokenizer.ggml.tokens` has 32k-260k entries and is routinely read with the
/// values thrown away -- but [`graph_size`] needs the **count** to size the
/// output-projection graph. So `size` is what [`Kv::vocab_size`] reads, and it
/// must be right even when `values` is empty.
#[derive(Debug, Clone, PartialEq)]
pub struct KvArray {
    /// The array length as declared in the file header. Authoritative.
    pub size: usize,
    /// The elements that were materialised. May be empty while `size` is huge.
    pub values: KvArrayValues,
}

impl KvArray {
    /// Array whose declared size matches the values you hand in. The normal case.
    pub fn new(values: KvArrayValues) -> Self {
        Self {
            size: values.len(),
            values,
        }
    }

    /// Array whose declared size is stated separately from its values -- the
    /// truncated-on-read case (`size = 151936`, `values = []`).
    pub fn with_declared_size(size: usize, values: KvArrayValues) -> Self {
        Self { size, values }
    }

    /// A string array declared `size` long but carrying no values -- exactly how
    /// a big `tokenizer.ggml.tokens` arrives after truncation. Use this to model
    /// a real vocabulary without carrying 150k strings around.
    pub fn truncated_strings(size: usize) -> Self {
        Self::with_declared_size(size, KvArrayValues::String(Vec::new()))
    }
}

/// The model's GGUF metadata: every `key -> value` from the file header.
///
/// **Upstream:** `type KV map[string]any` (`fs/ggml/ggml.go:33`) plus its whole
/// accessor set.
///
/// Two behaviours you must know before touching the accessors, because both
/// bite:
///
/// 1. **Keys get the architecture prefixed automatically.** Ask for
///    `"block_count"` on a llama model and the lookup is really
///    `"llama.block_count"`. The exceptions are keys already starting with
///    `"tokenizer."` or `"general."`, which are looked up as-is. Upstream:
///    `keyValue` (`ggml.go:318`).
/// 2. **Type match is exact.** A value stored as `u64` will *not* satisfy
///    [`Kv::uint`] (which wants `u32`); you get the default back instead.
///    Upstream relies on Go type assertions and behaves the same way.
///
/// Ordered map (`BTreeMap`) so iteration and debug output are deterministic --
/// Go's `map` is deliberately randomised and nothing here depends on order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Kv(BTreeMap<String, KvValue>);

impl Kv {
    /// Empty metadata.
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }

    /// Insert a raw key. The key is stored **verbatim** -- no architecture
    /// prefixing happens here, only on lookup. So a llama model's block count
    /// goes in as `"llama.block_count"`.
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<KvValue>) -> &mut Self {
        self.0.insert(key.into(), value.into());
        self
    }

    /// Raw, unprefixed lookup. **Upstream:** `KV.Value` (`ggml.go:274`).
    pub fn value(&self, key: &str) -> Option<&KvValue> {
        self.0.get(key)
    }

    /// Number of entries. **Upstream:** `KV.Len` (`ggml.go:266`).
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// True when got no metadata at all.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Every key, sorted. **Upstream:** `KV.Keys` (`ggml.go:270`, unordered there).
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.0.keys().map(String::as_str)
    }

    /// Apply the architecture prefix, then look up.
    ///
    /// **Upstream:** the first three lines of `keyValue` (`ggml.go:318`) --
    /// anything not already under `tokenizer.` or `general.` is namespaced by
    /// the architecture.
    fn lookup(&self, key: &str) -> Option<&KvValue> {
        if key.starts_with("tokenizer.") || key.starts_with("general.") {
            self.0.get(key)
        } else {
            self.0.get(&format!("{}.{}", self.architecture(), key))
        }
    }
}

impl FromIterator<(String, KvValue)> for Kv {
    fn from_iter<T: IntoIterator<Item = (String, KvValue)>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

// ---------------------------------------------------------------------------
// KV: the general typed getters
// ---------------------------------------------------------------------------

impl Kv {
    /// **Upstream:** `KV.String` (`ggml.go:187`). Exact match on a string value,
    /// else `default_value`.
    pub fn string<'a>(&'a self, key: &str, default_value: &'a str) -> &'a str {
        match self.lookup(key) {
            Some(KvValue::String(s)) => s,
            _ => default_value,
        }
    }

    /// **Upstream:** `KV.Uint` (`ggml.go:192`). Only matches a value stored as
    /// `u32` -- a `u64` under the same key gives you `default_value`, same as Go.
    pub fn uint(&self, key: &str, default_value: u32) -> u32 {
        match self.lookup(key) {
            Some(KvValue::U32(v)) => *v,
            _ => default_value,
        }
    }

    /// **Upstream:** `KV.Float` (`ggml.go:197`). `f32` only.
    pub fn float(&self, key: &str, default_value: f32) -> f32 {
        match self.lookup(key) {
            Some(KvValue::F32(v)) => *v,
            _ => default_value,
        }
    }

    /// **Upstream:** `KV.Bool` (`ggml.go:202`).
    ///
    /// Named `boolean` because `bool` is a primitive type name and reads badly
    /// as a method. Behaviour is upstream's exactly.
    pub fn boolean(&self, key: &str, default_value: bool) -> bool {
        match self.lookup(key) {
            Some(KvValue::Bool(v)) => *v,
            _ => default_value,
        }
    }

    /// **Upstream:** `KV.Strings` (`ggml.go:241`). Returns the array's
    /// **values**, not its declared size.
    pub fn strings(&self, key: &str, default_value: &[String]) -> Vec<String> {
        match self.lookup(key) {
            Some(KvValue::Array(KvArray {
                values: KvArrayValues::String(v),
                ..
            })) => v.clone(),
            _ => default_value.to_vec(),
        }
    }

    /// **Upstream:** `KV.Ints` (`ggml.go:246`) -- `int32` arrays.
    pub fn ints(&self, key: &str, default_value: &[i32]) -> Vec<i32> {
        match self.lookup(key) {
            Some(KvValue::Array(KvArray {
                values: KvArrayValues::I32(v),
                ..
            })) => v.clone(),
            _ => default_value.to_vec(),
        }
    }

    /// **Upstream:** `KV.Uints` (`ggml.go:251`) -- `uint32` arrays.
    pub fn uints(&self, key: &str, default_value: &[u32]) -> Vec<u32> {
        match self.lookup(key) {
            Some(KvValue::Array(KvArray {
                values: KvArrayValues::U32(v),
                ..
            })) => v.clone(),
            _ => default_value.to_vec(),
        }
    }

    /// **Upstream:** `KV.Floats` (`ggml.go:256`) -- `float32` arrays.
    pub fn floats(&self, key: &str, default_value: &[f32]) -> Vec<f32> {
        match self.lookup(key) {
            Some(KvValue::Array(KvArray {
                values: KvArrayValues::F32(v),
                ..
            })) => v.clone(),
            _ => default_value.to_vec(),
        }
    }

    /// **Upstream:** `KV.Bools` (`ggml.go:261`) -- `bool` arrays.
    pub fn bools(&self, key: &str, default_value: &[bool]) -> Vec<bool> {
        match self.lookup(key) {
            Some(KvValue::Array(KvArray {
                values: KvArrayValues::Bool(v),
                ..
            })) => v.clone(),
            _ => default_value.to_vec(),
        }
    }

    /// **Upstream:** `KV.UintOrArrayValueAsArray` (`ggml.go:222`).
    ///
    /// One key, three shapes, because GGUF is not consistent hor: a hyper-
    /// parameter like `attention.head_count` can be a single `u32` (same for
    /// every layer), a `u32` array (per-layer), or a `i32` array (per-layer,
    /// some exporters do this). Upstream tries them in exactly that order.
    ///
    /// Negative `i32` entries are nonsense for a head count; upstream logs a
    /// warning and casts anyway. We cast the same way, silently -- this crate
    /// has no logger, and changing the number would change the estimate.
    ///
    /// **Deliberate divergence:** an array present but with **no values**
    /// (truncated on read, or genuinely length 0) returns `[default_value]`
    /// here. Upstream returns an empty slice, which then makes `slices.Min` /
    /// `slices.Max` **panic** inside `UintOrArrayValue`. Returning the default
    /// keeps [`Kv::head_count`] byte-identical to Go (it treats a length-1
    /// result as "same for every layer") while removing the panic.
    pub fn uint_or_array_value_as_array(&self, key: &str, default_value: u32) -> Vec<u32> {
        match self.lookup(key) {
            Some(KvValue::U32(v)) => return vec![*v],
            Some(KvValue::Array(a)) => match &a.values {
                KvArrayValues::U32(v) if !v.is_empty() => return v.clone(),
                KvArrayValues::I32(v) if !v.is_empty() => {
                    return v.iter().map(|&x| x as u32).collect()
                }
                _ => {}
            },
            _ => {}
        }
        vec![default_value]
    }

    /// `(min, max)` over [`Kv::uint_or_array_value_as_array`].
    /// **Upstream:** `KV.UintOrArrayValue` (`ggml.go:217`).
    pub fn uint_or_array_value(&self, key: &str, default_value: u32) -> (u32, u32) {
        let values = self.uint_or_array_value_as_array(key, default_value);
        // Never empty: see the divergence note above.
        let min = values.iter().copied().min().unwrap_or(default_value);
        let max = values.iter().copied().max().unwrap_or(default_value);
        (min, max)
    }

    /// **Upstream:** `KV.UintOrMaxArrayValue` (`ggml.go:207`).
    pub fn uint_or_max_array_value(&self, key: &str, default_value: u32) -> u32 {
        self.uint_or_array_value(key, default_value).1
    }

    /// **Upstream:** `KV.UintOrMinArrayValue` (`ggml.go:212`).
    pub fn uint_or_min_array_value(&self, key: &str, default_value: u32) -> u32 {
        self.uint_or_array_value(key, default_value).0
    }
}

// ---------------------------------------------------------------------------
// KV: the model-shape accessors
// ---------------------------------------------------------------------------

impl Kv {
    /// The architecture name -- `"llama"`, `"gemma3"`, `"qwen3"`, `"deepseek2"`,
    /// ... **Upstream:** `KV.Architecture` (`ggml.go:35`).
    ///
    /// This is *the* load-bearing string in the whole module: it namespaces
    /// every non-`general.`/non-`tokenizer.` key, **and** it picks which
    /// `graph_size` formula runs. Default `"unknown"` (upstream's), which lands
    /// in no formula branch and so gives zero offload estimates.
    pub fn architecture(&self) -> &str {
        match self.0.get("general.architecture") {
            Some(KvValue::String(s)) => s,
            _ => "unknown",
        }
    }

    /// `general.type` -- `"model"`, `"adapter"`, ... **Upstream:** `KV.Kind`.
    pub fn kind(&self) -> &str {
        match self.0.get("general.type") {
            Some(KvValue::String(s)) => s,
            _ => "unknown",
        }
    }

    /// Total parameter count, as declared. **Upstream:** `KV.ParameterCount`.
    ///
    /// Stored as `u64`, unlike almost everything else here, so it does **not**
    /// go through [`Kv::uint`].
    pub fn parameter_count(&self) -> u64 {
        match self.0.get("general.parameter_count") {
            Some(KvValue::U64(v)) => *v,
            _ => 0,
        }
    }

    /// The file's quantisation. **Upstream:** `KV.FileType` (`ggml.go:48`).
    ///
    /// **Quirk worth knowing:** upstream only accepts a value `> 0`, and
    /// `FileTypeF32` *is* 0 -- so a genuine F32 model reports
    /// [`FileType::UNKNOWN`], not `F32`. That is upstream behaviour, kept
    /// deliberately. Do not "fix" it without checking who depends on it.
    pub fn file_type(&self) -> FileType {
        let t = self.uint("general.file_type", 0);
        if t > 0 {
            FileType(t)
        } else {
            FileType::UNKNOWN
        }
    }

    /// Number of transformer blocks (layers). **Upstream:** `KV.BlockCount`.
    ///
    /// Sizes every per-layer vector in this module, so an absurd value is an
    /// allocation hazard -- see [`MAX_BLOCK_COUNT`].
    pub fn block_count(&self) -> u64 {
        u64::from(self.uint("block_count", 0))
    }

    /// Model dimension `d_model`, in **elements** not bytes.
    /// **Upstream:** `KV.EmbeddingLength`.
    pub fn embedding_length(&self) -> u64 {
        u64::from(self.uint("embedding_length", 0))
    }

    /// Attention head count **per layer**, always exactly `block_count` long.
    /// **Upstream:** `KV.HeadCount` (`ggml.go:64`).
    ///
    /// The padding rule is upstream's and slightly surprising: if the metadata
    /// gave exactly **one** value, that value becomes the fill for every layer;
    /// otherwise layers past the end of the array get **1**. So a scalar
    /// `head_count = 32` yields `[32; block_count]`, but a 4-element array on a
    /// 32-layer model yields those 4 values then 28 ones.
    ///
    /// A layer with head count **0** is how a recurrent (SSM/Mamba) block
    /// announces itself -- [`graph_size`] keys off exactly that.
    ///
    /// Allocates `block_count` elements; validate untrusted metadata against
    /// [`MAX_BLOCK_COUNT`] first.
    pub fn head_count(&self) -> Vec<u64> {
        self.per_layer("attention.head_count", 1)
    }

    /// Largest attention head count over all layers.
    /// **Upstream:** `KV.HeadCountMax` (`ggml.go:85`).
    pub fn head_count_max(&self) -> u64 {
        u64::from(self.uint_or_max_array_value("attention.head_count", 1))
    }

    /// Smallest attention head count over all layers.
    /// **Upstream:** `KV.HeadCountMin` (`ggml.go:89`).
    pub fn head_count_min(&self) -> u64 {
        u64::from(self.uint_or_min_array_value("attention.head_count", 1))
    }

    /// KV head count **per layer** (grouped-query attention: fewer K/V heads
    /// than Q heads). **Upstream:** `KV.HeadCountKV` (`ggml.go:93`). Same
    /// padding rule as [`Kv::head_count`].
    pub fn head_count_kv(&self) -> Vec<u64> {
        self.per_layer("attention.head_count_kv", 1)
    }

    /// **Upstream:** `KV.HeadCountKVMax` (`ggml.go:114`).
    pub fn head_count_kv_max(&self) -> u64 {
        u64::from(self.uint_or_max_array_value("attention.head_count_kv", 1))
    }

    /// **Upstream:** `KV.HeadCountKVMin` (`ggml.go:118`).
    pub fn head_count_kv_min(&self) -> u64 {
        u64::from(self.uint_or_min_array_value("attention.head_count_kv", 1))
    }

    /// Largest per-head dimension, in **elements**.
    /// **Upstream:** `KV.EmbeddingHeadCountMax` (`ggml.go:122`).
    ///
    /// Read the formula carefully, the naming is upstream's and it trips people:
    /// it divides `embedding_length` by [`Kv::head_count_min`] -- the **min**
    /// head count -- because fewest heads over a fixed model dimension means the
    /// **widest** head. Returns 0 when the min head count is 0 (a fully
    /// recurrent model), which is upstream's guard against dividing by zero.
    pub fn embedding_head_count_max(&self) -> u64 {
        // Upstream writes this as `if heads > 0 { embedding / heads } else { 0 }`
        // (`ggml.go:122`) -- a guard against a fully recurrent model, where the
        // min head count is 0. `checked_div` says exactly the same thing.
        self.embedding_length()
            .checked_div(self.head_count_min())
            .unwrap_or(0)
    }

    /// Key head dimension in **elements**, `attention.key_length`, falling back
    /// to [`Kv::embedding_head_count_max`].
    /// **Upstream:** `KV.EmbeddingHeadCountK` (`ggml.go:130`).
    ///
    /// The fallback is narrowed `u64 -> u32` before being used as the default,
    /// exactly like upstream's `uint32(kv.EmbeddingHeadCountMax())`. That
    /// truncates above 4294967295, which no real head dimension gets near.
    pub fn embedding_head_count_k(&self) -> u64 {
        u64::from(self.uint("attention.key_length", self.embedding_head_count_max() as u32))
    }

    /// Value head dimension in **elements**, `attention.value_length`.
    /// **Upstream:** `KV.EmbeddingHeadCountV` (`ggml.go:134`). Same fallback and
    /// same narrowing as [`Kv::embedding_head_count_k`].
    pub fn embedding_head_count_v(&self) -> u64 {
        u64::from(self.uint("attention.value_length", self.embedding_head_count_max() as u32))
    }

    /// Training context length in **tokens**. **Upstream:** `KV.ContextLength`.
    ///
    /// Note this is what the model was *trained* for. [`graph_size`] takes the
    /// context it must actually size for as an argument instead, because a
    /// runner can be configured shorter or (with rope scaling) longer.
    pub fn context_length(&self) -> u64 {
        u64::from(self.uint("context_length", 0))
    }

    /// The Jinja chat template baked into the GGUF, or `""`.
    /// **Upstream:** `KV.ChatTemplate` (`ggml.go:142`). Not architecture-
    /// prefixed -- it lives under `tokenizer.`.
    pub fn chat_template(&self) -> &str {
        self.string("tokenizer.chat_template", "")
    }

    /// SSM convolution kernel width `d_conv`, in **elements**.
    /// **Upstream:** `KV.SSMConvKernel` (`ggml.go:148`).
    pub fn ssm_conv_kernel(&self) -> u64 {
        u64::from(self.uint("ssm.conv_kernel", 0))
    }

    /// SSM inner dimension `d_inner`, in **elements**.
    /// **Upstream:** `KV.SSMInnerSize` (`ggml.go:152`).
    pub fn ssm_inner_size(&self) -> u64 {
        u64::from(self.uint("ssm.inner_size", 0))
    }

    /// SSM recurrent state width `d_state`, in **elements**.
    /// **Upstream:** `KV.SSMStateSize` (`ggml.go:156`).
    pub fn ssm_state_size(&self) -> u64 {
        u64::from(self.uint("ssm.state_size", 0))
    }

    /// SSM group count `n_groups`. **Upstream:** `KV.SSMGroupCount`.
    pub fn ssm_group_count(&self) -> u64 {
        u64::from(self.uint("ssm.group_count", 0))
    }

    /// Feed-forward hidden width **per layer**, in elements.
    /// **Upstream:** `KV.FFNLength` (`ggml.go:164`).
    ///
    /// Same padding rule as [`Kv::head_count`] except the fill default is **0**,
    /// not 1 -- a layer with no declared FFN has no FFN.
    pub fn ffn_length(&self) -> Vec<u64> {
        self.per_layer("feed_forward_length", 0)
    }

    /// Vocabulary size, from the **declared size** of `tokenizer.ggml.tokens`.
    ///
    /// **Upstream:** the inline assertion in `GraphSize` (`ggml.go:656`):
    /// `uint64(f.KV()["tokenizer.ggml.tokens"].(*array[string]).size)`.
    ///
    /// Three things here are deliberate and all three matter:
    ///
    /// * The key is looked up **raw**, no architecture prefix -- it is already
    ///   under `tokenizer.`.
    /// * It reads `size`, **not** `values.len()`. Big vocab arrays are routinely
    ///   read with values discarded, and the count is what sizes the output
    ///   projection graph. Using `values.len()` would silently report 0 for
    ///   every real model.
    /// * Missing or wrong-typed is an [`Err`], where upstream panics. See
    ///   [`MemoryError::MissingVocab`].
    pub fn vocab_size(&self) -> Result<u64, MemoryError> {
        match self.0.get("tokenizer.ggml.tokens") {
            Some(KvValue::Array(a)) if matches!(a.values, KvArrayValues::String(_)) => {
                Ok(a.size as u64)
            }
            _ => Err(MemoryError::MissingVocab),
        }
    }

    /// Shared body of [`Kv::head_count`] / [`Kv::head_count_kv`] /
    /// [`Kv::ffn_length`] -- upstream repeats this loop three times verbatim
    /// (`ggml.go:64`, `:93`, `:164`), only the key and the fill default change.
    ///
    /// Upstream also logs a warning when the array is **longer** than the layer
    /// count; we drop the extra silently, same numeric result, because this
    /// crate carries no logger.
    fn per_layer(&self, key: &str, fill_default: u32) -> Vec<u64> {
        let values = self.uint_or_array_value_as_array(key, fill_default);
        // A single value means "same for every layer" -- it *replaces* the fill.
        let fill = if values.len() == 1 {
            values[0]
        } else {
            fill_default
        };
        let n_layers = self.block_count() as usize;
        (0..n_layers)
            .map(|i| u64::from(values.get(i).copied().unwrap_or(fill)))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Quantisation types -- fs/ggml/type.go
// ---------------------------------------------------------------------------

/// A `ggml_type`: how one tensor's numbers are packed.
///
/// **Upstream:** `type TensorType uint32` (`fs/ggml/type.go:250`).
///
/// Kept as a newtype over `u32` rather than a Rust `enum` **on purpose**. GGUF
/// stores the kind as a raw `u32` and new ggml types appear all the time; a
/// closed enum would need a fallible conversion at every read and would reject
/// files that ggml itself handles. Upstream has the same shape and the same
/// reason -- unknown kinds fall through to the `default` arms below.
///
/// A quantised tensor is stored in **blocks**: `block_size()` elements packed
/// into `type_size()` bytes. Both numbers come from ggml's own table, and the
/// pair is what makes [`Tensor::size`] exact rather than approximate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorType(pub u32);

#[allow(missing_docs)] // one const per ggml_type; the names are ggml's own
impl TensorType {
    // Discriminants are ggml's `enum ggml_type` in declaration order --
    // upstream's `const (... iota)` block at fs/ggml/type.go:252. The order is
    // fixed by the GGUF format; never renumber.
    pub const F32: Self = Self(0);
    pub const F16: Self = Self(1);
    pub const Q4_0: Self = Self(2);
    pub const Q4_1: Self = Self(3);
    pub const Q4_2: Self = Self(4); // removed from GGML
    pub const Q4_3: Self = Self(5); // removed from GGML
    pub const Q5_0: Self = Self(6);
    pub const Q5_1: Self = Self(7);
    pub const Q8_0: Self = Self(8);
    pub const Q8_1: Self = Self(9);
    pub const Q2_K: Self = Self(10);
    pub const Q3_K: Self = Self(11);
    pub const Q4_K: Self = Self(12);
    pub const Q5_K: Self = Self(13);
    pub const Q6_K: Self = Self(14);
    pub const Q8_K: Self = Self(15);
    pub const IQ2_XXS: Self = Self(16);
    pub const IQ2_XS: Self = Self(17);
    pub const IQ3_XXS: Self = Self(18);
    pub const IQ1_S: Self = Self(19);
    pub const IQ4_NL: Self = Self(20);
    pub const IQ3_S: Self = Self(21);
    pub const IQ2_S: Self = Self(22);
    pub const IQ4_XS: Self = Self(23);
    pub const I8: Self = Self(24);
    pub const I16: Self = Self(25);
    pub const I32: Self = Self(26);
    pub const I64: Self = Self(27);
    pub const F64: Self = Self(28);
    pub const IQ1_M: Self = Self(29);
    pub const BF16: Self = Self(30);
    pub const Q4_0_4_4: Self = Self(31); // unused by GGML
    pub const Q4_0_4_8: Self = Self(32); // unused by GGML
    pub const Q4_0_8_8: Self = Self(33); // unused by GGML
    pub const TQ1_0: Self = Self(34);
    pub const TQ2_0: Self = Self(35);
    pub const IQ4_NL_4_4: Self = Self(36); // unused by GGML
    pub const IQ4_NL_4_8: Self = Self(37); // unused by GGML
    pub const IQ4_NL_8_8: Self = Self(38); // unused by GGML
    pub const MXFP4: Self = Self(39);
    pub const NVFP4: Self = Self(40);
    pub const Q1_0: Self = Self(41);
}

impl TensorType {
    /// How many **elements** share one quantisation block.
    ///
    /// **Upstream:** `TensorType.BlockSize` (`fs/ggml/ggml.go:407`), which
    /// mirrors ggml's `ggml.c` type traits table.
    ///
    /// The numbers, and what they physically mean:
    ///
    /// * **1** -- not block-quantised at all (F32/F16/BF16/F64 and the integer
    ///   types). Every element stands alone.
    /// * **32** -- the classic "legacy" quant block: one scale (and sometimes a
    ///   min) shared by 32 weights. Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1, IQ4_NL, MXFP4.
    /// * **64** -- NVFP4's block.
    /// * **128** -- Q1_0's block.
    /// * **256** -- the K-quant superblock: 256 weights under one super-scale
    ///   with 8 or 16 sub-scales inside. Everything Q*_K and IQ*.
    ///
    /// Unknown kinds fall to 256, upstream's `default`. That is a guess, but it
    /// is *upstream's* guess and changing it would change every size estimate
    /// for a type we do not yet know.
    pub fn block_size(self) -> u64 {
        match self {
            TensorType::F32
            | TensorType::F16
            | TensorType::I8
            | TensorType::I16
            | TensorType::I32
            | TensorType::I64
            | TensorType::F64
            | TensorType::BF16 => 1,
            TensorType::Q4_0
            | TensorType::Q4_1
            | TensorType::Q5_0
            | TensorType::Q5_1
            | TensorType::Q8_0
            | TensorType::Q8_1
            | TensorType::IQ4_NL
            | TensorType::MXFP4 => 32,
            TensorType::NVFP4 => 64,
            TensorType::Q1_0 => 128,
            _ => 256,
        }
    }

    /// **Bytes** one block occupies on disk / in memory.
    ///
    /// **Upstream:** `TensorType.TypeSize` (`fs/ggml/ggml.go:442`). Every arm
    /// below is the literal ggml block struct laid out in bytes, so the `+ 2`s
    /// are `ggml_fp16_t` scales, the `+ 4`s are `float` scales, and the
    /// `blockSize/N` terms are the packed weights themselves. A few worked
    /// examples so the pattern is readable:
    ///
    /// * `Q4_0 = 2 + 32/2 = 18` -- one f16 scale `d`, then 32 weights at 4 bits.
    /// * `Q4_1 = 2 + 2 + 32/2 = 20` -- f16 scale `d` **and** f16 min `m`.
    /// * `Q8_0 = 2 + 32 = 34` -- f16 scale, 32 weights at a full byte each.
    /// * `Q4_K = 2 + 2 + 12 + 256/2 = 144` -- f16 super-scale `d`, f16 super-min
    ///   `dmin`, 12 bytes of packed 6-bit sub-scales/mins, then 256 4-bit weights.
    /// * `Q6_K = 256/2 + 256/4 + 256/16 + 2 = 210` -- low 4 bits, high 2 bits,
    ///   16 int8 sub-scales, f16 super-scale.
    ///
    /// Unknown kinds return **0**, upstream's `default`. Callers that divide by
    /// this must handle 0 -- [`Tensor::size`] does not, matching upstream, so a
    /// tensor of an unknown kind sizes to 0 rather than blowing up.
    ///
    /// Cross-check: upstream's own `TestTensorTypes` table is ported verbatim
    /// into the tests below, and it is itself a copy of llama.cpp's
    /// `ggml/src/ggml.c` type-traits table (link in that test's doc).
    pub fn type_size(self) -> u64 {
        let block_size = self.block_size();
        match self {
            TensorType::F32 => 4,
            TensorType::F16 => 2,
            TensorType::Q4_0 => 2 + block_size / 2,
            TensorType::Q4_1 => 2 + 2 + block_size / 2,
            TensorType::Q5_0 => 2 + 4 + block_size / 2,
            TensorType::Q5_1 => 2 + 2 + 4 + block_size / 2,
            TensorType::Q8_0 => 2 + block_size,
            TensorType::Q8_1 => 2 + 2 + block_size,
            TensorType::Q2_K => block_size / 16 + block_size / 4 + 2 + 2,
            TensorType::Q3_K => block_size / 8 + block_size / 4 + 12 + 2,
            TensorType::Q4_K => 2 + 2 + 12 + block_size / 2,
            TensorType::Q5_K => 2 + 2 + 12 + block_size / 8 + block_size / 2,
            TensorType::Q6_K => block_size / 2 + block_size / 4 + block_size / 16 + 2,
            TensorType::Q8_K => 4 + block_size + 2 * block_size / 16,
            TensorType::IQ2_XXS => 2 + 2 * block_size / 8,
            TensorType::IQ2_XS => 2 + 2 * block_size / 8 + block_size / 32,
            TensorType::IQ3_XXS => 2 + block_size / 4 + block_size / 8,
            TensorType::IQ1_S => 2 + block_size / 8 + block_size / 16,
            TensorType::IQ4_NL => 2 + block_size / 2,
            TensorType::IQ3_S => 2 + block_size / 4 + block_size / 8 + block_size / 32 + 4,
            TensorType::IQ2_S => 2 + block_size / 4 + block_size / 16,
            TensorType::IQ4_XS => 2 + 2 + block_size / 2 + block_size / 64,
            TensorType::I8 => 1,
            TensorType::I16 => 2,
            TensorType::I32 => 4,
            TensorType::I64 => 8,
            TensorType::F64 => 8,
            TensorType::IQ1_M => block_size / 8 + block_size / 16 + block_size / 32,
            TensorType::BF16 => 2,
            TensorType::MXFP4 => 1 + block_size / 2,
            TensorType::NVFP4 => 4 + block_size / 2,
            TensorType::Q1_0 => 2 + block_size / 8,
            _ => 0,
        }
    }

    /// Bytes taken by one row of `ne` elements.
    /// **Upstream:** `TensorType.RowSize` (`fs/ggml/type.go:349`).
    pub fn row_size(self, ne: u64) -> u64 {
        self.type_size() * ne / self.block_size()
    }

    /// Anything that is not F32/F16/BF16.
    /// **Upstream:** `TensorType.IsQuantized` (`fs/ggml/type.go:340`).
    pub fn is_quantized(self) -> bool {
        !matches!(self, TensorType::F32 | TensorType::F16 | TensorType::BF16)
    }

    /// The ggml name, e.g. `"Q4_K"`. **Upstream:** `TensorType.String`.
    pub fn name(self) -> &'static str {
        match self {
            TensorType::F32 => "F32",
            TensorType::F16 => "F16",
            TensorType::Q4_0 => "Q4_0",
            TensorType::Q4_1 => "Q4_1",
            TensorType::Q5_0 => "Q5_0",
            TensorType::Q5_1 => "Q5_1",
            TensorType::Q8_0 => "Q8_0",
            TensorType::Q8_1 => "Q8_1",
            TensorType::Q2_K => "Q2_K",
            TensorType::Q3_K => "Q3_K",
            TensorType::Q4_K => "Q4_K",
            TensorType::Q5_K => "Q5_K",
            TensorType::Q6_K => "Q6_K",
            TensorType::Q8_K => "Q8_K",
            TensorType::IQ2_XXS => "IQ2_XXS",
            TensorType::IQ2_XS => "IQ2_XS",
            TensorType::IQ3_XXS => "IQ3_XXS",
            TensorType::IQ1_S => "IQ1_S",
            TensorType::IQ4_NL => "IQ4_NL",
            TensorType::IQ3_S => "IQ3_S",
            TensorType::IQ2_S => "IQ2_S",
            TensorType::IQ4_XS => "IQ4_XS",
            TensorType::I8 => "I8",
            TensorType::I16 => "I16",
            TensorType::I32 => "I32",
            TensorType::I64 => "I64",
            TensorType::F64 => "F64",
            TensorType::IQ1_M => "IQ1_M",
            TensorType::BF16 => "BF16",
            TensorType::TQ1_0 => "TQ1_0",
            TensorType::TQ2_0 => "TQ2_0",
            TensorType::MXFP4 => "MXFP4",
            TensorType::NVFP4 => "NVFP4",
            TensorType::Q1_0 => "Q1_0",
            _ => "unknown",
        }
    }
}

impl std::fmt::Display for TensorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// A `llama_ftype`: the quantisation label for a **whole file**.
///
/// **Upstream:** `type FileType uint32` (`fs/ggml/type.go:10`).
///
/// Not the same numbering as [`TensorType`], and confusing the two is a classic
/// bug -- `FileType(12)` is `Q4_K_M` while `TensorType(12)` is `Q4_K`. A file
/// type is a *recipe* (`Q4_K_M` means "mostly Q4_K, but Q6_K for the important
/// tensors"), a tensor type is a *layout*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FileType(pub u32);

#[allow(missing_docs)] // one const per llama_ftype; names are llama.cpp's own
impl FileType {
    // Upstream's `const (... iota)` at fs/ggml/type.go:12. Fixed by the format.
    pub const F32: Self = Self(0);
    pub const F16: Self = Self(1);
    pub const Q4_0: Self = Self(2);
    pub const Q4_1: Self = Self(3);
    pub const Q4_1_F16: Self = Self(4); // removed from GGUF files
    pub const Q4_2: Self = Self(5); // removed from GGUF files
    pub const Q4_3: Self = Self(6); // removed from GGUF files
    pub const Q8_0: Self = Self(7);
    pub const Q5_0: Self = Self(8);
    pub const Q5_1: Self = Self(9);
    pub const Q2_K: Self = Self(10);
    pub const Q3_K_S: Self = Self(11);
    pub const Q3_K_M: Self = Self(12);
    pub const Q3_K_L: Self = Self(13);
    pub const Q4_K_S: Self = Self(14);
    pub const Q4_K_M: Self = Self(15);
    pub const Q5_K_S: Self = Self(16);
    pub const Q5_K_M: Self = Self(17);
    pub const Q6_K: Self = Self(18);
    pub const IQ2_XXS: Self = Self(19);
    pub const IQ2_XS: Self = Self(20);
    pub const Q2_K_S: Self = Self(21);
    pub const IQ3_XS: Self = Self(22);
    pub const IQ3_XXS: Self = Self(23);
    pub const IQ1_S: Self = Self(24);
    pub const IQ4_NL: Self = Self(25);
    pub const IQ3_S: Self = Self(26);
    pub const IQ3_M: Self = Self(27);
    pub const IQ2_S: Self = Self(28);
    pub const IQ2_M: Self = Self(29);
    pub const IQ4_XS: Self = Self(30);
    pub const IQ1_M: Self = Self(31);
    pub const BF16: Self = Self(32);
    pub const Q4_0_4_4: Self = Self(33); // unused by GGML
    pub const Q4_0_4_8: Self = Self(34); // unused by GGML
    pub const Q4_0_8_8: Self = Self(35); // unused by GGML
    pub const TQ1_0: Self = Self(36);
    pub const TQ2_0: Self = Self(37);
    pub const MXFP4_MOE: Self = Self(38);
    pub const NVFP4: Self = Self(39);
    pub const Q1_0: Self = Self(40);

    /// Sentinel for "not a file type we recognise".
    ///
    /// **Upstream:** `FileTypeUnknown = 1024` (`fs/ggml/type.go:55`). 1024 is
    /// deliberately far above the `iota` run so a future real ftype cannot
    /// collide with it.
    pub const UNKNOWN: Self = Self(1024);

    /// The label, e.g. `"Q4_K_M"`. **Upstream:** `FileType.String`
    /// (`fs/ggml/type.go:92`), whose own note says it deliberately returns a
    /// broader set of names than [`FileType::parse`] will accept, because old
    /// models in the wild carry types ollama no longer produces.
    pub fn name(self) -> &'static str {
        match self {
            FileType::F32 => "F32",
            FileType::F16 => "F16",
            FileType::Q4_0 => "Q4_0",
            FileType::Q4_1 => "Q4_1",
            FileType::Q8_0 => "Q8_0",
            FileType::Q5_0 => "Q5_0",
            FileType::Q5_1 => "Q5_1",
            FileType::Q2_K => "Q2_K",
            FileType::Q3_K_S => "Q3_K_S",
            FileType::Q3_K_M => "Q3_K_M",
            FileType::Q3_K_L => "Q3_K_L",
            FileType::Q4_K_S => "Q4_K_S",
            FileType::Q4_K_M => "Q4_K_M",
            FileType::Q5_K_S => "Q5_K_S",
            FileType::Q5_K_M => "Q5_K_M",
            FileType::Q6_K => "Q6_K",
            FileType::IQ2_XXS => "IQ2_XXS",
            FileType::IQ2_XS => "IQ2_XS",
            FileType::Q2_K_S => "Q2_K_S",
            FileType::IQ3_XS => "IQ3_XS",
            FileType::IQ3_XXS => "IQ3_XXS",
            FileType::IQ1_S => "IQ1_S",
            FileType::IQ4_NL => "IQ4_NL",
            FileType::IQ3_S => "IQ3_S",
            FileType::IQ3_M => "IQ3_M",
            FileType::IQ2_S => "IQ2_S",
            FileType::IQ2_M => "IQ2_M",
            FileType::IQ4_XS => "IQ4_XS",
            FileType::IQ1_M => "IQ1_M",
            FileType::BF16 => "BF16",
            FileType::TQ1_0 => "TQ1_0",
            FileType::TQ2_0 => "TQ2_0",
            FileType::MXFP4_MOE => "MXFP4_MOE",
            FileType::NVFP4 => "NVFP4",
            FileType::Q1_0 => "Q1_0",
            _ => "unknown",
        }
    }

    /// Parse a quantisation the platform will actually *produce*.
    ///
    /// **Upstream:** `ParseFileType` (`fs/ggml/type.go:60`) -- deliberately
    /// narrow. Upstream can *read* far more types than this; the short list is
    /// what it is willing to quantise **to**. `"Q4_K"` is accepted as an alias
    /// for `"Q4_K_M"`, which is upstream's own convenience.
    ///
    /// Returns [`FileType::UNKNOWN`] as the error payload's companion so a
    /// caller that wants to carry on can.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "F32" => Ok(FileType::F32),
            "F16" => Ok(FileType::F16),
            "Q8_0" => Ok(FileType::Q8_0),
            "Q4_K_S" => Ok(FileType::Q4_K_S),
            "Q4_K_M" | "Q4_K" => Ok(FileType::Q4_K_M),
            "BF16" => Ok(FileType::BF16),
            other => Err(format!(
                "unsupported quantization type {other} - supported types are F32, F16, Q4_K_S, Q4_K_M, Q8_0"
            )),
        }
    }

    /// Which tensor layout this file-level recipe mostly uses.
    /// **Upstream:** `FileType.ToTensorType` (`fs/ggml/type.go:174`).
    ///
    /// Lossy on purpose: `Q3_K_S`, `Q3_K_M` and `Q3_K_L` all collapse to
    /// [`TensorType::Q3_K`], because the S/M/L only differ in *which* tensors
    /// get bumped to a wider type, not in the layout of the Q3_K ones.
    /// An unrecognised ftype falls back to [`TensorType::F32`] (upstream logs a
    /// warning there; we have no logger, so it is silent).
    pub fn to_tensor_type(self) -> TensorType {
        match self {
            FileType::F32 => TensorType::F32,
            FileType::F16 => TensorType::F16,
            FileType::Q4_0 => TensorType::Q4_0,
            FileType::Q4_1 => TensorType::Q4_1,
            FileType::Q8_0 => TensorType::Q8_0,
            FileType::Q5_0 => TensorType::Q5_0,
            FileType::Q5_1 => TensorType::Q5_1,
            FileType::Q2_K | FileType::Q2_K_S => TensorType::Q2_K,
            FileType::Q3_K_S | FileType::Q3_K_M | FileType::Q3_K_L => TensorType::Q3_K,
            FileType::Q4_K_S | FileType::Q4_K_M => TensorType::Q4_K,
            FileType::Q5_K_S | FileType::Q5_K_M => TensorType::Q5_K,
            FileType::Q6_K => TensorType::Q6_K,
            FileType::IQ2_XXS => TensorType::IQ2_XXS,
            FileType::IQ2_XS => TensorType::IQ2_XS,
            FileType::IQ3_XS | FileType::IQ3_S | FileType::IQ3_M => TensorType::IQ3_S,
            FileType::IQ3_XXS => TensorType::IQ3_XXS,
            FileType::IQ1_S => TensorType::IQ1_S,
            FileType::IQ4_NL => TensorType::IQ4_NL,
            FileType::IQ2_S | FileType::IQ2_M => TensorType::IQ2_S,
            FileType::IQ4_XS => TensorType::IQ4_XS,
            FileType::IQ1_M => TensorType::IQ1_M,
            FileType::BF16 => TensorType::BF16,
            FileType::TQ1_0 => TensorType::TQ1_0,
            FileType::TQ2_0 => TensorType::TQ2_0,
            FileType::MXFP4_MOE => TensorType::MXFP4,
            FileType::NVFP4 => TensorType::NVFP4,
            FileType::Q1_0 => TensorType::Q1_0,
            _ => TensorType::F32,
        }
    }
}

impl std::fmt::Display for FileType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// Tensors -- names, shapes, kinds. No bytes, no reader.
// ---------------------------------------------------------------------------

/// One tensor's *description* -- name, quant kind, shape. **Not its data.**
///
/// **Upstream:** `type Tensor` (`fs/ggml/ggml.go:384`), minus the `io.WriterTo`
/// that upstream embeds for writing GGUF back out. We do not write GGUF here,
/// so that field has no counterpart -- see the module header on the seam.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Tensor {
    /// The GGUF tensor name, e.g. `"blk.0.ffn_gate.weight"`. Its dotted shape is
    /// what [`Tensors::group_layers`] parses.
    pub name: String,
    /// The raw `ggml_type` discriminant. Wrap in [`TensorType`] to interpret.
    pub kind: u32,
    /// Byte offset into the file's tensor-data section. Carried for
    /// completeness; [`graph_size`] never reads it.
    pub offset: u64,
    /// Elements per dimension, **fastest-varying first** (ggml's `ne`). So a
    /// `[4096, 14336]` FFN weight has 4096 down a row.
    pub shape: Vec<u64>,
}

impl Tensor {
    /// Convenience for building test/consumer fixtures: name + kind + shape.
    pub fn new(name: impl Into<String>, kind: TensorType, shape: impl Into<Vec<u64>>) -> Self {
        Self {
            name: name.into(),
            kind: kind.0,
            offset: 0,
            shape: shape.into(),
        }
    }

    /// This tensor's quant type.
    pub fn tensor_type(&self) -> TensorType {
        TensorType(self.kind)
    }

    /// Total **elements** (product of the shape).
    /// **Upstream:** `Tensor.Elements` (`fs/ggml/ggml.go:515`).
    ///
    /// An empty shape gives 1 -- the empty product -- which is upstream's
    /// behaviour and matters because a scalar tensor is legal.
    pub fn elements(&self) -> u64 {
        self.shape
            .iter()
            .fold(w(1), |acc, &n| acc * w(n))
            .0
    }

    /// Total **bytes** on disk: `elements * type_size / block_size`.
    /// **Upstream:** `Tensor.Size` (`fs/ggml/ggml.go:534`).
    ///
    /// Ordered exactly as upstream writes it -- multiply *then* divide -- so a
    /// row length that is not a whole number of blocks rounds the same way Go
    /// rounds. Reordering to divide first would change results for such tensors.
    ///
    /// A tensor of an unknown [`TensorType`] sizes to **0** (its `type_size` is
    /// 0). Upstream does the same; treat 0 as "cannot size this", not "empty".
    pub fn size(&self) -> u64 {
        let t = self.tensor_type();
        let block_size = t.block_size();
        if block_size == 0 {
            return 0;
        }
        (w(self.elements()) * w(t.type_size()) / w(block_size)).0
    }
}

/// One logical layer: the tensors sharing a `blk.N` / `mm.N` / bare prefix,
/// keyed by the rest of their name (`"attn_k.weight"`, `"ffn_gate.weight"`).
///
/// **Upstream:** `type Layer map[string]*Tensor` (`fs/ggml/ggml.go:374`).
/// Borrowed, not owned, so grouping never copies tensor descriptions.
pub type Layer<'a> = BTreeMap<String, &'a Tensor>;

/// Total **bytes** of every tensor in one layer.
///
/// **Upstream:** `Layer.Size` (`fs/ggml/ggml.go:376`). A free function here
/// because [`Layer`] is a plain `BTreeMap` alias and Rust does not let us hang
/// an inherent method off that.
///
/// This is the weights-on-disk figure the scheduler adds to
/// [`GraphSize::kv_cache`] to work out what one layer really costs in VRAM.
/// Saturating, so a tensor of an unknown kind (which sizes to 0) under-reports
/// rather than wrapping.
pub fn layer_size(layer: &Layer<'_>) -> u64 {
    layer
        .values()
        .fold(0u64, |acc, t| acc.saturating_add(t.size()))
}

/// Every tensor description in a model.
///
/// **Upstream:** `type Tensors` (`fs/ggml/ggml.go:331`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Tensors {
    /// The tensor descriptions, file order.
    pub items: Vec<Tensor>,
    /// Byte offset where the tensor data section starts. Carried for
    /// completeness; unused by [`graph_size`].
    pub offset: u64,
}

impl Tensors {
    /// Build from a list of tensor descriptions.
    pub fn new(items: Vec<Tensor>) -> Self {
        Self { items, offset: 0 }
    }

    /// All tensors, or only those whose name starts with `prefix`.
    /// **Upstream:** `Tensors.Items` (`fs/ggml/ggml.go:336`) -- its variadic
    /// prefix becomes an explicit [`Option`] here.
    pub fn items(&self, prefix: Option<&str>) -> Vec<&Tensor> {
        match prefix {
            None => self.items.iter().collect(),
            Some(p) => self.items.iter().filter(|t| t.name.starts_with(p)).collect(),
        }
    }

    /// Group tensors into layers by name.
    ///
    /// **Upstream:** `Tensors.GroupLayers` (`fs/ggml/ggml.go:351`).
    ///
    /// The rule, in full, because the naming scheme is not obvious:
    ///
    /// * Split the name on `.`.
    /// * If any part is literally `blk` or `mm`, the **next** part is its index,
    ///   so glue those two together into one key -- `blk.0.attn_k.weight`
    ///   becomes layer `blk.0`, entry `attn_k.weight`. This survives a prefix:
    ///   `v.blk.0.attn_k.weight` becomes layer `v.blk.0`.
    /// * The gluing only happens if there is something left after the index; a
    ///   name that *is* just `blk.0` stays as-is.
    /// * Anything else groups on its first part alone -- `token_embd.weight`
    ///   becomes layer `token_embd`, entry `weight`; `v.patch_embd.weight`
    ///   becomes layer `v`, entry `patch_embd.weight`.
    ///
    /// [`graph_size`] uses this to peek at `blk.0` and tell a dense llama from a
    /// mixtral MoE, so the grouping is not cosmetic -- get it wrong and the
    /// wrong formula runs.
    pub fn group_layers(&self) -> BTreeMap<String, Layer<'_>> {
        let mut layers: BTreeMap<String, Layer<'_>> = BTreeMap::new();
        for t in &self.items {
            let mut parts: Vec<String> = t.name.split('.').map(str::to_string).collect();
            if let Some(index) = parts.iter().position(|s| s == "blk" || s == "mm")
                && parts.len() > index + 2
            {
                let head = parts[..index + 2].join(".");
                let mut rebuilt = vec![head];
                rebuilt.extend_from_slice(&parts[index + 2..]);
                parts = rebuilt;
            }
            let (layer_name, rest) = parts.split_at(1);
            layers
                .entry(layer_name[0].clone())
                .or_default()
                .insert(rest.join("."), t);
        }
        layers
    }
}

// ---------------------------------------------------------------------------
// Cache type + flash attention
// ---------------------------------------------------------------------------

/// **Bytes per KV-cache element** for a given cache quantisation.
///
/// **Upstream:** `kvCacheBytesPerElement` (`fs/ggml/ggml.go:959`). The values,
/// and what they physically are:
///
/// | cache type | bytes/elem | why |
/// |---|---|---|
/// | `"q8_0"` | 1.0 | 8-bit quantised cache -- half of fp16 |
/// | `"q4_0"` | 0.5 | 4-bit quantised cache -- a quarter of fp16 |
/// | `"f32"` | 4.0 | full float; llama.cpp forces this for **recurrent** state |
/// | anything else (incl. `""`) | 2.0 | fp16, the default |
///
/// Fractional on purpose: 0.5 B/element is exactly why the result is computed in
/// `f64` and truncated once at the end, not accumulated in integers.
///
/// **What would make this wrong:** these are the *storage* widths only. The
/// per-block scale a real q8_0/q4_0 cache carries is not counted here -- that is
/// upstream's simplification, and copying it is the point.
pub fn kv_cache_bytes_per_element(cache_type: &str) -> f64 {
    match cache_type {
        "q8_0" => 1.0,
        "q4_0" => 0.5,
        "f32" => 4.0,
        _ => 2.0,
    }
}

/// Can the KV cache be stored in this quantisation?
/// **Upstream:** `GGML.SupportsKVCacheType` (`fs/ggml/ggml.go:899`).
pub fn supports_kv_cache_type(cache_type: &str) -> bool {
    matches!(cache_type, "" | "f16" | "q8_0" | "q4_0")
}

/// Is this cache type quantised (as opposed to a plain float)?
/// **Upstream:** `GGML.KVCacheTypeIsQuantized` (`fs/ggml/ggml.go:908`).
pub fn kv_cache_type_is_quantized(cache_type: &str) -> bool {
    !matches!(cache_type, "" | "f16" | "f32" | "bf16")
}

/// Tri-state flash-attention switch.
///
/// **Upstream:** `ml.FlashAttentionType` (`ml/device.go:602`), whose own comment
/// says the numbering is "aligned with `llama_flash_attn_type`" -- so the
/// discriminants are llama.cpp's, not ollama's invention, and must not be
/// renumbered.
///
/// Only [`FlashAttentionType::Enabled`] changes anything in [`graph_size`]
/// (`Auto` has not yet been resolved to a decision at that point, so it costs
/// the non-flash estimate, same as `Disabled`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum FlashAttentionType {
    /// Decide later, from the device. Upstream value `-1`.
    #[default]
    Auto = -1,
    /// Off. Upstream value `0`.
    Disabled = 0,
    /// On. Upstream value `1`.
    Enabled = 1,
}

impl std::fmt::Display for FlashAttentionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Upstream: ml.FlashAttentionType.String (ml/device.go:615).
        f.write_str(match self {
            FlashAttentionType::Auto => "Auto",
            FlashAttentionType::Disabled => "Disabled",
            FlashAttentionType::Enabled => "Enabled",
        })
    }
}

impl Kv {
    /// Does the model's shape allow flash attention at all?
    /// **Upstream:** `GGML.SupportsFlashAttention` (`fs/ggml/ggml.go:916`).
    ///
    /// Three gates, in upstream's order:
    /// 1. An **embedding** model (it has `<arch>.pooling_type`) -- never.
    /// 2. A hardcoded yes-list, then a hardcoded no-list (`gemma2`, `grok`).
    /// 3. Otherwise: K and V head dims must be non-zero **and equal**, because
    ///    the fused kernel assumes one head width for both.
    pub fn supports_flash_attention(&self) -> bool {
        let pooling_key = format!("{}.pooling_type", self.architecture());
        if self.0.contains_key(&pooling_key) {
            return false;
        }
        let arch = self.architecture();
        if matches!(arch, "qwen35" | "qwen35moe" | "qwen3next") {
            return true;
        }
        if matches!(arch, "gemma2" | "grok") {
            return false;
        }
        let k = self.embedding_head_count_k();
        let v = self.embedding_head_count_v();
        k != 0 && v != 0 && k == v
    }

    /// Should flash attention be **on by default** for this architecture?
    /// **Upstream:** `GGML.FlashAttention` (`fs/ggml/ggml.go:938`) -- a plain
    /// allow-list of architectures upstream has validated it against. It is a
    /// curated list, not a rule; do not try to derive it.
    pub fn flash_attention_default(&self) -> bool {
        matches!(
            self.string("general.architecture", ""),
            "bert"
                | "gemma3"
                | "gemma4"
                | "glm4moelite"
                | "glmocr"
                | "gptoss"
                | "gpt-oss"
                | "lfm2"
                | "lfm2moe"
                | "mistral3"
                | "nemotron_h"
                | "nemotron_h_moe"
                | "nemotron_h_omni"
                | "olmo3"
                | "qwen3"
                | "qwen3moe"
                | "qwen35"
                | "qwen35moe"
                | "qwen3next"
                | "qwen3vl"
                | "qwen3vlmoe"
        )
    }

    /// Must this model run on ollama's own engine rather than the llama.cpp
    /// runner? **Upstream:** `KV.OllamaEngineRequired` (`fs/ggml/ggml.go:278`).
    ///
    /// Relevant here because [`graph_size`]'s *default* per-layer cache formula
    /// explicitly assumes the llamarunner's caching behaviour -- an architecture
    /// on this list without its own branch in the `match` is a sizing risk, not
    /// just a routing note.
    pub fn ollama_engine_required(&self) -> bool {
        matches!(
            self.architecture(),
            "bert"
                | "deepseek2"
                | "deepseekocr"
                | "gemma3"
                | "gemma3n"
                | "gemma4"
                | "gptoss"
                | "gpt-oss"
                | "laguna"
                | "llama4"
                | "mistral3"
                | "mllama"
                | "nemotron_h"
                | "nemotron_h_moe"
                | "nemotron_h_omni"
                | "nomic-bert"
                | "olmo3"
                | "qwen25vl"
                | "qwen3"
                | "qwen3moe"
                | "qwen35"
                | "qwen35moe"
                | "qwen3next"
                | "qwen3vl"
                | "qwen3vlmoe"
                | "glm4moelite"
                | "glmocr"
                | "lfm2"
                | "lfm2moe"
        )
    }
}

// ---------------------------------------------------------------------------
// GraphSize
// ---------------------------------------------------------------------------

/// What a model will cost in memory, in **bytes**.
///
/// **Upstream:** the three named return values of `GGML.GraphSize`
/// (`fs/ggml/ggml.go:648`), collected into a struct so callers cannot swap
/// `partial_offload` and `full_offload` by accident -- an easy mistake with a
/// bare `(Vec<u64>, u64, u64)`, and an expensive one, since the two differ by
/// several hundred MiB on a normal model.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GraphSize {
    /// KV cache bytes **per layer**, exactly `block_count` long.
    ///
    /// Per-layer because a modern model is not uniform: gemma3 alternates local
    /// sliding-window layers with global full-context ones, gptoss alternates by
    /// parity, mllama's cross-attention layers cache vision tokens instead of
    /// text, and a Mamba layer caches recurrent state that does not scale with
    /// context at all. A single average would be wrong for all four.
    ///
    /// Upstream's `kv []uint64`.
    pub kv_cache: Vec<u64>,
    /// Scratch bytes for the compute graph when only **some** layers are on the
    /// GPU, so activations must cross the CPU/GPU boundary mid-graph. Usually
    /// the bigger of the two. Upstream's `partialOffload`.
    pub partial_offload: u64,
    /// Scratch bytes for the compute graph when **every** layer is on the GPU.
    /// Upstream's `fullOffload`.
    pub full_offload: u64,
}

impl GraphSize {
    /// Total KV cache over all layers, in bytes. Saturating, so a nonsense model
    /// reports `u64::MAX` rather than wrapping round to something small and
    /// plausible-looking -- a too-big number gets a layer rejected, a wrapped
    /// one gets it accepted and then OOMs.
    pub fn kv_cache_total(&self) -> u64 {
        self.kv_cache
            .iter()
            .fold(0u64, |acc, &n| acc.saturating_add(n))
    }

    /// One-line human summary, for logs and `kopitiam` CLI output.
    ///
    /// Uses [`human_bytes2`] (binary units -- KiB/MiB/GiB), never `human_bytes`:
    /// these are allocator numbers, so they must speak the allocator's units.
    /// See `format.rs` on why the crate carries both.
    pub fn summary(&self) -> String {
        format!(
            "kv cache {} over {} layers, partial offload {}, full offload {}",
            human_bytes2(self.kv_cache_total()),
            self.kv_cache.len(),
            human_bytes2(self.partial_offload),
            human_bytes2(self.full_offload),
        )
    }
}

/// Work out the KV-cache and compute-graph memory a model needs.
///
/// **Upstream:** `GGML.GraphSize` (`fs/ggml/ggml.go:648`). This is the whole
/// reason the module exists: the scheduler compares these numbers against free
/// VRAM to decide **how many layers fit on the GPU**.
///
/// # Arguments
///
/// * `kv` -- the model's GGUF metadata. See [`Kv`] for the prefixing and
///   exact-type rules.
/// * `tensors` -- tensor names/shapes/kinds. Only a few branches read it (to
///   tell dense llama from mixtral MoE, and for chatglm's qkv bias / mllama's
///   rope freqs). Pass `&Tensors::default()` if the architecture does not need
///   it -- the result is identical for every branch that never looks.
/// * `context` -- context length in **tokens, per sequence**. Multiplied by
///   `num_parallel` on the first line, exactly like upstream, so everything
///   downstream sees the *aggregate* context.
/// * `batch` -- batch size in **tokens** (ollama's `num_batch`, default 512).
/// * `num_parallel` -- how many sequences run concurrently.
/// * `kv_cache_type` -- `""`/`"f16"`/`"q8_0"`/`"q4_0"`; see
///   [`kv_cache_bytes_per_element`].
/// * `flash_attention` -- only [`FlashAttentionType::Enabled`] changes anything,
///   and only for gptoss.
///
/// # Returns
///
/// [`GraphSize`], or [`MemoryError::MissingVocab`] if the vocabulary count
/// cannot be read (upstream panics there instead), or
/// [`MemoryError::AbsurdBlockCount`] on metadata claiming more layers than
/// [`MAX_BLOCK_COUNT`].
///
/// # What the numbers are, and are not
///
/// These formulas are **empirical fits** from upstream's measurements against
/// real models, not derivations from the transformer's algebra. Constants like
/// `105/128` and `9/16` are fudge factors with no closed form -- they are
/// documented as such at each use rather than dressed up with invented
/// reasoning. The correct way to change any of them is to check what ollama
/// does now.
///
/// **What would make this wrong:** an architecture that is *not* in the `match`
/// below falls back to the default per-layer cache loop and gets
/// `partial_offload = full_offload = 0`. That is upstream's behaviour and it is
/// silent. A brand new architecture will therefore be sized as if its compute
/// graph is free, which under-estimates. If KOPITIAM starts serving an
/// architecture upstream has not special-cased, that is the first thing to check.
#[allow(clippy::too_many_arguments)]
pub fn graph_size(
    kv: &Kv,
    tensors: &Tensors,
    context: u64,
    batch: u64,
    num_parallel: u64,
    kv_cache_type: &str,
    flash_attention: FlashAttentionType,
) -> Result<GraphSize, MemoryError> {
    let block_count = kv.block_count();
    if block_count > MAX_BLOCK_COUNT {
        return Err(MemoryError::AbsurdBlockCount {
            block_count,
            max: MAX_BLOCK_COUNT,
        });
    }
    let n_layers = block_count as usize;

    // Upstream line 649: `context *= uint64(numParallel)`. From here on
    // `context` is the AGGREGATE token count across all sequences, not the
    // per-sequence one. Every formula below depends on that -- including the
    // `context>>10` in the gptoss flash-attention estimate.
    let context = w(context.wrapping_mul(num_parallel));
    let batch = w(batch);
    let num_parallel_w = w(num_parallel);

    let embedding = w(kv.embedding_length());
    let heads = w(kv.head_count_max());
    let heads_arr = kv.head_count();
    let heads_kv = w(kv.head_count_kv_max());
    let heads_kv_arr = kv.head_count_kv();
    let vocab = w(kv.vocab_size()?);

    let embedding_heads = w(kv.embedding_head_count_max());
    let embedding_heads_k = w(kv.embedding_head_count_k());
    let embedding_heads_v = w(kv.embedding_head_count_v());

    let layers = tensors.group_layers();

    let bytes_per_element = kv_cache_bytes_per_element(kv_cache_type);

    // ---- the default per-layer cache loop (upstream lines 676-708) ----
    //
    // Upstream's own comment: these defaults mirror llama.cpp's cache usage, on
    // the assumption that any architecture without a special case below runs on
    // the llamarunner and lets llama.cpp handle caching.
    //
    // It also assumes a layer with NO heads and NO kv-heads is **recurrent**,
    // which is usually right. Where it is not -- nemotronh uses "blocks" and
    // some of them are plain MLP with no cache at all -- the architecture needs
    // its own branch below or the estimate is wrong.
    let mut kv_cache = vec![0u64; n_layers];
    let mut kv_total = w(0);
    for (i, slot) in kv_cache.iter_mut().enumerate() {
        let heads_l = heads_arr[i];
        let heads_kv_l = heads_kv_arr[i];
        if heads_l > 0 && heads_kv_l > 0 {
            // Full attention layer. Elements cached = context tokens x (K width
            // + V width) x number of KV heads. NOTE upstream's own caveat: this
            // assumes every attention layer has the same K/V widths.
            let elements = context * (embedding_heads_k + embedding_heads_v) * w(heads_kv_l);
            *slot = (elements.0 as f64 * bytes_per_element) as u64;
        } else {
            // Recurrent (SSM / Mamba) layer. Its state does NOT grow with
            // context -- that is the whole selling point of the architecture --
            // so `context` does not appear here at all.
            let ssm_d_conv = w(kv.ssm_conv_kernel());
            let ssm_d_state = w(kv.ssm_state_size());
            let ssm_d_inner = w(kv.ssm_inner_size());
            let ssm_n_groups = w(kv.ssm_group_count());
            // n_embd_r: the rolling conv window. (d_conv - 1) positions of
            // history, over the inner channels plus the B/C projections
            // (2 x n_groups x d_state).
            let n_embd_r = if ssm_d_conv.0 > 0 {
                (ssm_d_conv - w(1)) * (ssm_d_inner + w(2) * ssm_n_groups * ssm_d_state)
            } else {
                w(0)
            };
            // n_embd_s: the SSM state itself, d_state per inner channel.
            let n_embd_s = ssm_d_state * ssm_d_inner;
            // Recurrent state is ALWAYS f32 in the llama.cpp backend, whatever
            // the user asked for the attention cache. Upstream cites
            // https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp#L18644
            let bytes_per_element_recurrent = kv_cache_bytes_per_element("f32");
            *slot = ((n_embd_r + n_embd_s) * w(bytes_per_element_recurrent as u64)).0;
        }
        kv_total += w(*slot);
    }

    let mut partial_offload = w(0);
    let mut full_offload = w(0);

    // ---- per-architecture graph estimates (upstream lines 711-893) ----
    //
    // Two fudge factors recur; both are upstream's, neither has a derivation:
    //   * `* 9 / 16`   (= 0.5625) on square-ish weight terms.
    //   * `* 105 / 128` (= 0.8203) on the `embedding * vocab` output-projection
    //     term -- the "vocab graph".
    // They are empirical fits. Do not "simplify" them.
    match kv.architecture() {
        "llama" | "llama4" => {
            full_offload = (w(4) * batch * (w(1) + w(4) * embedding + context * (w(1) + heads)))
                .max(w(4) * batch * (embedding + vocab));

            partial_offload = w(4) * batch * embedding;
            partial_offload += (w(4) * batch * (w(1) + embedding + context.max(embedding))
                + embedding * embedding * w(9) / w(16)
                + w(4) * context * (batch * heads + embedding_heads * heads_kv))
                .max(w(4) * batch * (embedding + vocab) + embedding * vocab * w(105) / w(128));

            if let Some(t) = layers
                .get("blk.0")
                .and_then(|l| l.get("ffn_gate_exps.weight"))
            {
                // mixtral 8x22b -- MoE with the experts fused into one tensor.
                let ff = w(u64::from(kv.uint("feed_forward_length", 0)));
                partial_offload = (w(3) * w(t.size())
                    + w(4)
                        * batch
                        * (w(2) * ff
                            + heads_kv
                            + embedding
                            + context
                            + embedding_heads * heads_kv))
                    .max(
                        w(4)
                            * (context * batch * heads
                                + context * embedding_heads * heads_kv
                                // 1024: upstream literal, no stated source.
                                + batch * w(1024)
                                + embedding_heads * heads_kv * batch),
                    );
            } else if let Some(t) = layers.get("blk.0").and_then(|l| l.get("ffn_gate.0.weight")) {
                // mixtral 8x7b -- MoE with one tensor per expert, so the FFN
                // width has to be read off the gate tensor's shape rather than
                // from `feed_forward_length`.
                let ffn_gate_weight1 = w(t.shape.get(1).copied().unwrap_or(0));
                full_offload = w(4)
                    * batch
                    * (w(2)
                        + w(3) * embedding
                        + context * (w(1) + heads)
                        + w(2) * heads_kv
                        + ffn_gate_weight1);
                // Upstream divides by `heads` here. `heads` is
                // `HeadCountMax(..., default 1)`, so it is only 0 if the file
                // explicitly says `attention.head_count = 0` -- on which Go
                // panics with an integer divide by zero. DELIBERATE DIVERGENCE:
                // we substitute 1 and carry on. A zero-head MoE is nonsense
                // metadata either way; this just refuses to crash over it.
                let heads_nz = if heads.0 == 0 { w(1) } else { heads };
                partial_offload = (w(4)
                    * batch
                    * (w(3)
                        + embedding_heads * heads_kv
                        + embedding
                        + context * (w(1) + heads)
                        + ffn_gate_weight1)
                    + (embedding * embedding + w(3) * embedding * heads_kv * ffn_gate_weight1)
                        * w(9)
                        / w(16))
                .max(
                    w(4) * batch * (w(1) + w(2) * embedding + context * (w(1) + heads))
                        + embedding
                            * (w(6) * context * heads_kv / heads_nz + embedding * w(9) / w(16)),
                );
            }
        }
        "mllama" => {
            // Llama 3.2 Vision. Its cross-attention layers cache *image* tokens,
            // not text, so their size does not scale with context at all.
            //
            // 1601 = 1600 patch tokens + 1 CLS token, the ViT output for one
            // image tile; 4 = the tile grid. Both are upstream literals
            // (ggml.go:741) fixed by mllama's vision encoder configuration.
            let vision_tokens = w(1601);
            let tiles = w(4);

            let cross_attention_layers = kv.ints("attention.cross_attention_layers", &[]);
            for (i, slot) in kv_cache.iter_mut().enumerate() {
                if cross_attention_layers.contains(&(i as i32)) {
                    *slot = (heads_kv
                        * (embedding_heads_k + embedding_heads_v)
                        // 4 = sizeof(float32). Cross-attention cache is always
                        // f32 here, so `bytes_per_element` is NOT applied.
                        * w(4)
                        * vision_tokens
                        * tiles)
                        .0;
                }
            }

            full_offload = (w(4)
                * batch
                * (w(2) + w(3) * embedding + embedding_heads_k * heads + context * (w(1) + heads)))
                // the vocab graph
                .max(w(4) * batch * (embedding + vocab));

            let rope_freqs_count = layers
                .get("rope_freqs")
                .and_then(|l| l.get("weights"))
                .map(|t| t.elements())
                .unwrap_or(0);

            partial_offload = (w(4)
                * (batch
                    * (w(2) * embedding
                        + w(1)
                        + context * (w(1) + heads)
                        + embedding_heads_k * heads)
                    + w(rope_freqs_count)
                    + embedding_heads_k * context * heads_kv))
                // the vocab graph
                .max(w(4) * batch * (embedding + vocab) + embedding * vocab * w(105) / w(128));
        }
        "gemma" | "gemma2" | "gemma3" | "gemma3n" => {
            full_offload = (w(4) * batch * (embedding + vocab)).max(
                w(4)
                    * batch
                    * (w(2)
                        + context
                        + context * heads
                        + w(2) * embedding
                        + w(2) * embedding_heads_k * heads),
            );

            partial_offload = (w(4) * embedding * batch
                + embedding * vocab * w(105) / w(128)
                + w(4) * vocab * batch)
                .max(
                    w(4)
                        * batch
                        * (w(2) * embedding
                            + w(1)
                            + w(2) * embedding_heads_k * heads
                            + context
                            + context * heads)
                        // 8: upstream literal (ggml.go:783), no stated source.
                        + w(4) * embedding_heads_k * context * w(8)
                        + embedding * embedding_heads_k * heads * w(9) / w(16),
                );

            if kv.architecture() == "gemma3n" {
                // Upstream multiplies both by 4 for gemma3n. No derivation is
                // given; it is a measured correction for the per-layer-embedding
                // architecture. (ggml.go:787)
                full_offload *= w(4);
                partial_offload *= w(4);
            }

            // gemma2 also has sliding-window attention, but upstream only has an
            // optimised implementation in the ollama engine, and gemma3 always
            // uses that engine -- so only gemma3 gets the smaller local cache.
            if kv.architecture() == "gemma3" {
                // Every 6th layer is GLOBAL (full context); the other 5 are
                // LOCAL sliding-window layers. 6 is gemma3's published
                // local:global interleave ratio (5 local : 1 global).
                // Upstream: `const gemma3GlobalCacheCount = 6` (ggml.go:795).
                const GEMMA3_GLOBAL_CACHE_COUNT: usize = 6;
                let sliding_window = num_parallel_w
                    * w(u64::from(kv.uint("attention.sliding_window", 0)))
                    + batch;
                for (i, slot) in kv_cache.iter_mut().enumerate() {
                    if (i + 1) % GEMMA3_GLOBAL_CACHE_COUNT != 0 {
                        let elements =
                            sliding_window * (embedding_heads_k + embedding_heads_v) * heads_kv;
                        *slot = (elements.0 as f64 * bytes_per_element) as u64;
                    }
                }
            }
        }
        "command-r" => {
            full_offload = (w(4) * batch * (embedding + vocab))
                .max(w(4) * batch * (w(2) + w(4) * embedding + context * (w(1) + heads)));

            partial_offload = (w(4) * batch * (embedding + vocab)
                + embedding * vocab * w(105) / w(128))
            .max(
                w(4) * batch * (w(1) + w(2) * embedding + context * (w(1) + heads))
                    + w(4) * embedding * context
                    + embedding * embedding * w(9) / w(16),
            );
        }
        "qwen2" => {
            full_offload = (w(4) * batch * (embedding + vocab))
                .max(w(4) * batch * (w(1) + w(2) * embedding + context + context * heads));

            partial_offload = (w(4) * batch * (embedding + vocab)
                + embedding * vocab * w(105) / w(128))
            .max(
                w(4)
                    * (batch * (w(1) + w(2) * embedding + context * (w(1) + heads))
                        + embedding * (w(1) + context)),
            );
        }
        "phi2" => {
            full_offload = (w(4) * batch * (embedding + vocab))
                .max(w(4) * batch * (w(1) + w(4) * embedding + context + context * heads));

            partial_offload = (w(4) * batch * (w(2) * embedding + vocab)
                + embedding * vocab * w(105) / w(128))
            .max(w(4) * batch * (w(2) + w(3) * embedding + context + context * heads));
        }
        "stablelm" => {
            full_offload = w(4) * batch * (context * (w(1) + heads) + w(3) * embedding + w(2));
            // Note the shape: stablelm's partial estimate is floored at its own
            // full estimate, which no other architecture does.
            partial_offload = (w(4) * batch * (vocab + w(2) * embedding)).max(full_offload);
        }
        "deepseek2" => {
            // Deepseek-V2/V3 MLA. Note every attention term keys off `heads_kv`,
            // not `heads` -- the latent KV projection is shared, so the cache is
            // sized by the KV heads alone.
            full_offload = (w(4) * batch * (w(3) * embedding + vocab)).max(
                w(4)
                    * batch
                    * (w(3) * embedding
                        + w(2)
                        + context * (w(1) + heads_kv)
                        + w(2) * embedding_heads_k * heads_kv),
            );

            partial_offload = (w(4) * batch * (w(3) * embedding + vocab)
                + embedding * vocab * w(105) / w(128))
            .max(
                w(4)
                    * batch
                    * (w(2) * embedding
                        + w(1)
                        + w(2) * embedding_heads_k * heads_kv
                        + context
                        + context * heads_kv)
                    + w(4) * embedding_heads_k * context * heads_kv
                    + embedding * embedding_heads_k * heads_kv * w(9) / w(16),
            );
        }
        "chatglm" => {
            full_offload = w(4) * batch * (embedding + vocab);
            partial_offload = w(4) * batch * (embedding + vocab)
                + embedding * vocab * w(105) / w(128);
            // ChatGLM's fused QKV bias only exists on some conversions; when it
            // is there the attention graph is bigger, so both estimates get
            // floored upward.
            if let Some(qkv_bias) = layers.get("blk.0").and_then(|l| l.get("attn_qkv.bias")) {
                // Upstream indexes Shape[0] directly and panics on an empty
                // shape. DELIBERATE DIVERGENCE: a shapeless bias contributes 0.
                let qkv_bias0 = w(qkv_bias.shape.first().copied().unwrap_or(0));

                full_offload = full_offload.max(
                    w(4)
                        * batch
                        * (w(2)
                            + w(2) * embedding
                            + context
                            + context * heads
                            + embedding_heads_k * heads
                            + qkv_bias0),
                );

                partial_offload = partial_offload.max(
                    w(4)
                        * batch
                        * (w(1)
                            + w(2) * embedding
                            + embedding_heads_k * heads
                            + context
                            + context * heads)
                        // These two terms are identical up to operand order in
                        // upstream too (ggml.go:872-873). Kept as written rather
                        // than folded to 8*..., so the port stays diffable.
                        + w(4) * embedding_heads_k * context
                        + w(4) * context * embedding_heads_k
                        + w(4) * qkv_bias0,
                );
            }
        }
        "gptoss" | "gpt-oss" => {
            // GPT-OSS alternates sliding-window and full-context attention by
            // layer PARITY: even layers are windowed, odd layers see everything.
            for (i, slot) in kv_cache.iter_mut().enumerate() {
                let per_token = (embedding_heads_k + embedding_heads_v) * heads_kv;
                let mut bytes = w((per_token.0 as f64 * bytes_per_element) as u64);
                if i % 2 == 0 {
                    // 4096 = gpt-oss's sliding window in tokens, one window per
                    // parallel sequence, plus the in-flight batch. Upstream
                    // literal (ggml.go:882).
                    bytes *= num_parallel_w * w(4096) + batch;
                } else {
                    bytes *= context;
                }
                *slot = bytes.0;
            }

            // `kv_total` here is the total from the DEFAULT loop above, not from
            // the gptoss cache just written. That is upstream's behaviour
            // (ggml.go:888 reads the `kvTotal` accumulated at line 707) and it
            // matters: the two totals differ.
            //
            // Evaluation order is left-to-right and the integer divisions are
            // NOT associative -- ((2*headsMax)/headsKVMin)*kvTotal/6. Reordering
            // changes the answer. `cmp.Or(HeadCountKVMin(), 1)` guards a zero.
            let heads_kv_min = kv.head_count_kv_min();
            let divisor = w(if heads_kv_min == 0 { 1 } else { heads_kv_min });
            partial_offload = w(2) * w(kv.head_count_max()) / divisor * kv_total / w(6);

            if flash_attention == FlashAttentionType::Enabled {
                // Upstream calls this "a rough estimate of graph size with flash
                // attention on" (ggml.go:890). All three literals are its own,
                // with no derivation offered:
                //   4 MiB per parallel sequence,
                //   1 MiB per 1024 tokens of AGGREGATE context (`context>>10`,
                //     and remember context was already multiplied by
                //     num_parallel at the top),
                //   110 MiB fixed overhead.
                partial_offload =
                    (w(4) * num_parallel_w + context / w(1024) + w(110)) * w(MEBIBYTE);
            }
        }
        // Everything else keeps the default per-layer cache and zero graph
        // estimates. See the "what would make this wrong" note on this function.
        _ => {}
    }

    Ok(GraphSize {
        kv_cache,
        partial_offload: partial_offload.0,
        full_offload: full_offload.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a [`Kv`] from literal pairs. Keys go in **raw** -- so a llama
    /// model's block count must be written `"llama.block_count"`, same as it
    /// sits in the file.
    fn kv_of(pairs: Vec<(&str, KvValue)>) -> Kv {
        pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
    }

    /// A vocabulary of `n` tokens, modelled the way a real GGUF arrives: the
    /// count is declared, the 32k strings are not materialised.
    fn vocab(n: usize) -> KvValue {
        KvArray::truncated_strings(n).into()
    }

    /// Llama-2-7B-ish shape: 32 layers, d_model 4096, 32 query heads, 8 KV
    /// heads (GQA), 32k vocab. Head dim falls out as 4096/32 = 128.
    fn llama_7b_kv() -> Kv {
        kv_of(vec![
            ("general.architecture", "llama".into()),
            ("llama.block_count", 32u32.into()),
            ("llama.embedding_length", 4096u32.into()),
            ("llama.attention.head_count", 32u32.into()),
            ("llama.attention.head_count_kv", 8u32.into()),
            ("tokenizer.ggml.tokens", vocab(32000)),
        ])
    }

    // -- Tensors::group_layers, ported from upstream's TestTensorLayers ------

    fn named(names: &[&str]) -> Tensors {
        Tensors::new(
            names
                .iter()
                .map(|n| Tensor::new(*n, TensorType::F32, vec![]))
                .collect(),
        )
    }

    const TEXT_TENSORS: &[&str] = &[
        "token_embd.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_output.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_norm.weight",
        "blk.0.ffn_down.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_norm.weight",
        "output_norm.weight",
    ];

    const VISION_TENSORS: &[&str] = &[
        "mm.0.bias",
        "mm.0.weight",
        "v.blk.0.attn_k.weight",
        "v.blk.0.attn_output.weight",
        "v.blk.0.attn_q.weight",
        "v.blk.0.attn_v.weight",
        "v.blk.0.attn_norm.weight",
        "v.blk.0.ffn_down.weight",
        "v.blk.0.ffn_gate.weight",
        "v.blk.0.ffn_up.weight",
        "v.blk.0.ffn_norm.weight",
        "v.patch_embd.weight",
        "v.position_embd.gate",
        "v.position_embd.weight",
    ];

    /// Layer name -> its sorted entry keys, which is all the upstream test
    /// actually asserts on (it compares whole maps of pointers).
    fn layout(tensors: &Tensors) -> BTreeMap<String, Vec<String>> {
        tensors
            .group_layers()
            .into_iter()
            .map(|(k, v)| (k, v.keys().cloned().collect()))
            .collect()
    }

    #[test]
    fn group_layers_splits_text_tensors_into_blk_and_top_level_layers() {
        // Upstream: TestTensorLayers/"text" (fs/ggml/ggml_test.go:52).
        let got = layout(&named(TEXT_TENSORS));
        assert_eq!(got.len(), 3, "blk.0, token_embd, output_norm");
        assert_eq!(
            got["blk.0"],
            vec![
                "attn_k.weight",
                "attn_norm.weight",
                "attn_output.weight",
                "attn_q.weight",
                "attn_v.weight",
                "ffn_down.weight",
                "ffn_gate.weight",
                "ffn_norm.weight",
                "ffn_up.weight",
            ]
        );
        assert_eq!(got["token_embd"], vec!["weight"]);
        assert_eq!(got["output_norm"], vec!["weight"]);
    }

    #[test]
    fn group_layers_splits_vision_tensors_by_v_and_mm_prefix() {
        // Upstream: TestTensorLayers/"vision" (ggml_test.go:79). The point is
        // that `mm` and `blk` both glue to their index, and that `v.` on its own
        // is a layer name, so `v.patch_embd.weight` lands in layer "v".
        let got = layout(&named(VISION_TENSORS));
        assert_eq!(got.len(), 3, "mm.0, v.blk.0, v");
        assert_eq!(got["mm.0"], vec!["bias", "weight"]);
        assert_eq!(got["v.blk.0"].len(), 9);
        assert_eq!(
            got["v"],
            vec![
                "patch_embd.weight",
                "position_embd.gate",
                "position_embd.weight",
            ]
        );
    }

    #[test]
    fn group_layers_keeps_text_and_vision_apart_in_one_model() {
        // Upstream: TestTensorLayers/"vision and text" (ggml_test.go:113).
        let all: Vec<&str> = TEXT_TENSORS
            .iter()
            .chain(VISION_TENSORS.iter())
            .copied()
            .collect();
        let got = layout(&named(&all));
        let mut names: Vec<&str> = got.keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            vec!["blk.0", "mm.0", "output_norm", "token_embd", "v", "v.blk.0"]
        );
    }

    // -- TensorType, ported from upstream's TestTensorTypes ------------------

    #[test]
    fn tensor_block_and_type_sizes_match_ggml_type_traits_table() {
        // Upstream: TestTensorTypes (ggml_test.go:164), which itself cites
        // llama.cpp ggml/src/ggml.c#L572 (commit a82c9e7c23ef). Every row is
        // (ggml_type discriminant, elements per block, bytes per block).
        let cases: &[(u32, u64, u64)] = &[
            (0, 1, 4),      // F32
            (1, 1, 2),      // F16
            (2, 32, 18),    // Q4_0: f16 scale + 32x4bit
            (3, 32, 20),    // Q4_1: f16 scale + f16 min + 32x4bit
            (6, 32, 22),    // Q5_0
            (7, 32, 24),    // Q5_1
            (8, 32, 34),    // Q8_0: f16 scale + 32 bytes
            (9, 32, 36),    // Q8_1
            (10, 256, 84),  // Q2_K
            (11, 256, 110), // Q3_K
            (12, 256, 144), // Q4_K
            (13, 256, 176), // Q5_K
            (14, 256, 210), // Q6_K
            (15, 256, 292), // Q8_K
            (16, 256, 66),  // IQ2_XXS
            (17, 256, 74),  // IQ2_XS
            (18, 256, 98),  // IQ3_XXS
            (19, 256, 50),  // IQ1_S
            (20, 32, 18),   // IQ4_NL
            (21, 256, 110), // IQ3_S
            (22, 256, 82),  // IQ2_S
            (23, 256, 136), // IQ4_XS
            (24, 1, 1),     // I8
            (25, 1, 2),     // I16
            (26, 1, 4),     // I32
            (27, 1, 8),     // I64
            (28, 1, 8),     // F64
            (29, 256, 56),  // IQ1_M
            (30, 1, 2),     // BF16
        ];
        for &(kind, block_size, type_size) in cases {
            let t = TensorType(kind);
            assert_eq!(t.block_size(), block_size, "block size of kind {kind}");
            assert_eq!(t.type_size(), type_size, "type size of kind {kind}");
        }
    }

    #[test]
    fn tensor_size_is_elements_scaled_by_the_block_layout() {
        // A Q4_K FFN weight: 4096 x 14336 elements = 58,720,256, packed 256 to
        // a 144-byte superblock -> 229,376 blocks -> 33,030,144 bytes.
        let t = Tensor::new("blk.0.ffn_down.weight", TensorType::Q4_K, vec![4096, 14336]);
        assert_eq!(t.elements(), 58_720_256);
        assert_eq!(t.size(), 33_030_144);

        // Empty shape is the empty product: one element, not zero.
        let scalar = Tensor::new("scalar", TensorType::F32, vec![]);
        assert_eq!(scalar.elements(), 1);
        assert_eq!(scalar.size(), 4);
    }

    #[test]
    fn layer_size_sums_the_weights_of_one_block() {
        let tensors = Tensors::new(vec![
            Tensor::new("blk.0.ffn_down.weight", TensorType::Q4_K, vec![4096, 14336]),
            Tensor::new("blk.0.attn_norm.weight", TensorType::F32, vec![4096]),
        ]);
        let layers = tensors.group_layers();
        assert_eq!(
            layer_size(&layers["blk.0"]),
            33_030_144 + 4096 * 4,
            "the Q4_K FFN weight plus an f32 norm vector"
        );
    }

    #[test]
    fn an_unknown_tensor_kind_sizes_to_zero_rather_than_guessing() {
        // Upstream's TypeSize default arm returns 0. Treat 0 as "cannot size",
        // never as "empty tensor".
        let t = Tensor::new("mystery", TensorType(9999), vec![4096]);
        assert_eq!(t.tensor_type().name(), "unknown");
        assert_eq!(t.tensor_type().block_size(), 256, "upstream's default");
        assert_eq!(t.size(), 0);
    }

    #[test]
    fn file_type_zero_reports_unknown_because_f32_collides_with_absent() {
        // Upstream KV.FileType only accepts > 0, and FileTypeF32 IS 0. Quirk
        // kept on purpose -- see Kv::file_type.
        let f32_model = kv_of(vec![
            ("general.architecture", "llama".into()),
            ("general.file_type", 0u32.into()),
        ]);
        assert_eq!(f32_model.file_type(), FileType::UNKNOWN);

        let q4km = kv_of(vec![
            ("general.architecture", "llama".into()),
            ("general.file_type", 15u32.into()),
        ]);
        assert_eq!(q4km.file_type(), FileType::Q4_K_M);
        assert_eq!(q4km.file_type().name(), "Q4_K_M");
        // The recipe Q4_K_M mostly lays tensors out as Q4_K.
        assert_eq!(q4km.file_type().to_tensor_type(), TensorType::Q4_K);
    }

    #[test]
    fn file_type_parsing_accepts_only_what_ollama_will_quantise_to() {
        assert_eq!(FileType::parse("Q4_K_M"), Ok(FileType::Q4_K_M));
        assert_eq!(FileType::parse("Q4_K"), Ok(FileType::Q4_K_M), "alias");
        assert_eq!(FileType::parse("Q8_0"), Ok(FileType::Q8_0));
        // Readable, but not something upstream will produce -- so parse rejects.
        assert!(FileType::parse("Q6_K").is_err());
        assert_eq!(FileType::Q6_K.name(), "Q6_K", "but it still has a name");
    }

    // -- KV accessors, ported from upstream's TestKeyValue / TestHeadCount ---

    #[test]
    fn typed_getters_match_on_exact_type_and_fall_back_otherwise() {
        // Upstream: TestKeyValue (ggml_test.go:215).
        let kv = kv_of(vec![
            ("general.architecture", "test".into()),
            (
                "test.strings",
                KvArray::new(KvArrayValues::String(vec![
                    "a".into(),
                    "b".into(),
                    "c".into(),
                ]))
                .into(),
            ),
            (
                "test.float32s",
                KvArray::new(KvArrayValues::F32(vec![1.0, 2.0, 3.0])).into(),
            ),
            (
                "test.int32s",
                KvArray::new(KvArrayValues::I32(vec![1, 2, 3])).into(),
            ),
            (
                "test.uint32s",
                KvArray::new(KvArrayValues::U32(vec![1, 2, 3])).into(),
            ),
        ]);

        assert_eq!(kv.strings("strings", &[]), vec!["a", "b", "c"]);
        assert!(kv.strings("nonexistent.strings", &[]).is_empty());
        assert_eq!(
            kv.strings("default.strings", &["ollama".to_string()]),
            vec!["ollama"]
        );

        assert_eq!(kv.floats("float32s", &[]), vec![1.0, 2.0, 3.0]);
        assert!(kv.floats("nonexistent.float32s", &[]).is_empty());
        assert_eq!(kv.floats("default.float32s", &[f32::MAX]), vec![f32::MAX]);

        assert_eq!(kv.ints("int32s", &[]), vec![1, 2, 3]);
        assert!(kv.ints("nonexistent.int32s", &[]).is_empty());
        assert_eq!(kv.ints("default.int32s", &[i32::MAX]), vec![i32::MAX]);

        assert_eq!(kv.uints("uint32s", &[]), vec![1, 2, 3]);
        assert!(kv.uints("nonexistent.uint32s", &[]).is_empty());
        assert_eq!(kv.uints("default.uint32s", &[u32::MAX]), vec![u32::MAX]);

        // Exact-type match, upstream's Go type assertion: a u64 does not
        // satisfy `uint`, which wants u32. You get the default back.
        let widened = kv_of(vec![
            ("general.architecture", "test".into()),
            ("test.block_count", 32u64.into()),
        ]);
        assert_eq!(widened.uint("block_count", 7), 7, "u64 must not match u32");
    }

    #[test]
    fn head_count_max_takes_the_largest_whether_scalar_or_array() {
        // Upstream: TestHeadCount (ggml_test.go:273).
        let arr = kv_of(vec![
            ("general.architecture", "abc".into()),
            (
                "abc.attention.head_count",
                KvArray::new(KvArrayValues::I32(vec![1, 5, 3, 4])).into(),
            ),
        ]);
        assert_eq!(arr.head_count_max(), 5);
        assert_eq!(arr.head_count_min(), 1);

        let scalar = kv_of(vec![
            ("general.architecture", "abc".into()),
            ("abc.attention.head_count", 3u32.into()),
        ]);
        assert_eq!(scalar.head_count_max(), 3);
        assert_eq!(scalar.head_count_min(), 3);
    }

    #[test]
    fn per_layer_head_counts_pad_out_to_the_block_count() {
        // One scalar means "same for every layer".
        let uniform = kv_of(vec![
            ("general.architecture", "abc".into()),
            ("abc.block_count", 4u32.into()),
            ("abc.attention.head_count", 8u32.into()),
        ]);
        assert_eq!(uniform.head_count(), vec![8, 8, 8, 8]);

        // A short array does NOT repeat -- the tail fills with the accessor's
        // own default, which is 1 for head counts and 0 for FFN lengths.
        let ragged = kv_of(vec![
            ("general.architecture", "abc".into()),
            ("abc.block_count", 4u32.into()),
            (
                "abc.attention.head_count",
                KvArray::new(KvArrayValues::U32(vec![8, 0])).into(),
            ),
            (
                "abc.feed_forward_length",
                KvArray::new(KvArrayValues::U32(vec![14336, 0])).into(),
            ),
        ]);
        assert_eq!(ragged.head_count(), vec![8, 0, 1, 1]);
        assert_eq!(ragged.ffn_length(), vec![14336, 0, 0, 0]);
    }

    #[test]
    fn embedding_head_count_divides_by_the_minimum_head_count() {
        // The naming is upstream's and it reads backwards: FEWEST heads over a
        // fixed d_model means the WIDEST head, hence min in the denominator.
        let kv = kv_of(vec![
            ("general.architecture", "abc".into()),
            ("abc.embedding_length", 4096u32.into()),
            (
                "abc.attention.head_count",
                KvArray::new(KvArrayValues::U32(vec![32, 16])).into(),
            ),
        ]);
        assert_eq!(kv.head_count_min(), 16);
        assert_eq!(kv.embedding_head_count_max(), 256, "4096 / 16");
        assert_eq!(kv.embedding_head_count_k(), 256, "no explicit key_length");

        // An explicit key/value length wins over the fallback.
        let explicit = kv_of(vec![
            ("general.architecture", "abc".into()),
            ("abc.embedding_length", 4096u32.into()),
            ("abc.attention.head_count", 32u32.into()),
            ("abc.attention.key_length", 192u32.into()),
        ]);
        assert_eq!(explicit.embedding_head_count_k(), 192);
        assert_eq!(explicit.embedding_head_count_v(), 128, "falls back to 4096/32");
    }

    #[test]
    fn a_fully_recurrent_model_reports_zero_head_width_without_dividing_by_zero() {
        let kv = kv_of(vec![
            ("general.architecture", "mamba".into()),
            ("mamba.embedding_length", 2048u32.into()),
            ("mamba.attention.head_count", 0u32.into()),
        ]);
        assert_eq!(kv.head_count_min(), 0);
        assert_eq!(kv.embedding_head_count_max(), 0);
    }

    // -- graph_size, against real model shapes -------------------------------

    #[test]
    fn a_llama_7b_shape_sizes_its_kv_cache_and_both_offloads() {
        let got = graph_size(
            &llama_7b_kv(),
            &Tensors::default(),
            2048,
            512,
            1,
            "f16",
            FlashAttentionType::Disabled,
        )
        .expect("llama shape sizes cleanly");

        // Per layer: 2048 tokens x (128 K + 128 V) x 8 KV heads x 2 B = 8 MiB.
        assert_eq!(got.kv_cache.len(), 32);
        assert!(got.kv_cache.iter().all(|&n| n == 8_388_608));
        assert_eq!(got.kv_cache_total(), 268_435_456, "256 MiB of KV cache");

        // max( 4*512*(1 + 4*4096 + 2048*33), 4*512*(4096+32000) )
        //   = max(171_968_512, 73_924_608)
        assert_eq!(got.full_offload, 171_968_512);

        // 4*512*4096  +  max(attention graph 168_822_784, vocab graph 181_444_608)
        assert_eq!(got.partial_offload, 189_833_216);
    }

    #[test]
    fn num_parallel_multiplies_the_context_before_anything_else() {
        // Upstream's very first line is `context *= numParallel`, so two
        // sequences of 1024 must cost exactly what one sequence of 2048 costs.
        let two_seqs = graph_size(
            &llama_7b_kv(),
            &Tensors::default(),
            1024,
            512,
            2,
            "f16",
            FlashAttentionType::Disabled,
        )
        .unwrap();
        let one_long = graph_size(
            &llama_7b_kv(),
            &Tensors::default(),
            2048,
            512,
            1,
            "f16",
            FlashAttentionType::Disabled,
        )
        .unwrap();
        assert_eq!(two_seqs, one_long);
    }

    #[test]
    fn a_quantised_kv_cache_halves_then_quarters_the_f16_footprint() {
        let sized = |cache_type: &str| {
            graph_size(
                &llama_7b_kv(),
                &Tensors::default(),
                2048,
                512,
                1,
                cache_type,
                FlashAttentionType::Disabled,
            )
            .unwrap()
            .kv_cache[0]
        };
        assert_eq!(sized("f16"), 8_388_608);
        assert_eq!(sized(""), 8_388_608, "empty string means f16");
        assert_eq!(sized("q8_0"), 4_194_304, "1 B/elem");
        assert_eq!(sized("q4_0"), 2_097_152, "0.5 B/elem");
        assert_eq!(kv_cache_bytes_per_element("q4_0"), 0.5);
        assert!(supports_kv_cache_type("q8_0"));
        assert!(!supports_kv_cache_type("q5_1"));
        assert!(kv_cache_type_is_quantized("q4_0"));
        assert!(!kv_cache_type_is_quantized("f16"));
    }

    #[test]
    fn mixtral_8x7b_is_detected_from_its_per_expert_gate_tensor() {
        // Same llama metadata; what switches the formula is one tensor NAME.
        // `blk.0.ffn_gate.0.weight` (per-expert) says 8x7b, and its shape[1] is
        // where the FFN width comes from -- feed_forward_length is not used.
        let tensors = Tensors::new(vec![Tensor::new(
            "blk.0.ffn_gate.0.weight",
            TensorType::Q4_K,
            vec![4096, 14336],
        )]);
        let got = graph_size(
            &llama_7b_kv(),
            &tensors,
            2048,
            512,
            1,
            "f16",
            FlashAttentionType::Disabled,
        )
        .unwrap();

        assert_eq!(got.full_offload, 192_974_848);
        assert_eq!(got.partial_offload, 980_424_704);

        // Sanity: the dense-llama path gives different numbers, so the branch
        // really did fire.
        let dense = graph_size(
            &llama_7b_kv(),
            &Tensors::default(),
            2048,
            512,
            1,
            "f16",
            FlashAttentionType::Disabled,
        )
        .unwrap();
        assert_ne!(dense.full_offload, got.full_offload);
    }

    #[test]
    fn gemma3_gives_every_sixth_layer_the_full_context_cache() {
        let kv = kv_of(vec![
            ("general.architecture", "gemma3".into()),
            ("gemma3.block_count", 6u32.into()),
            ("gemma3.embedding_length", 1152u32.into()),
            ("gemma3.attention.head_count", 4u32.into()),
            ("gemma3.attention.head_count_kv", 1u32.into()),
            ("gemma3.attention.key_length", 256u32.into()),
            ("gemma3.attention.value_length", 256u32.into()),
            ("gemma3.attention.sliding_window", 512u32.into()),
            ("tokenizer.ggml.tokens", vocab(262144)),
        ]);
        let got = graph_size(
            &kv,
            &Tensors::default(),
            1024,
            256,
            1,
            "f16",
            FlashAttentionType::Disabled,
        )
        .unwrap();

        // Local layers cache (512 window + 256 batch) = 768 tokens; the global
        // one (index 5, because (5+1) % 6 == 0) caches the full 1024.
        // tokens x (256 K + 256 V) x 1 KV head x 2 B per f16 element.
        let local = 768 * 512 * 2;
        let global = 1024 * 512 * 2;
        assert_eq!(
            got.kv_cache,
            vec![local, local, local, local, local, global]
        );
        assert_eq!(got.full_offload, 269_615_104);
        assert_eq!(got.partial_offload, 517_341_184);
    }

    #[test]
    fn gemma3n_scales_both_offloads_by_four() {
        let base = |arch: &str| {
            let kv = kv_of(vec![
                ("general.architecture", arch.into()),
                (&format!("{arch}.block_count"), 4u32.into()),
                (&format!("{arch}.embedding_length"), 2048u32.into()),
                (&format!("{arch}.attention.head_count"), 8u32.into()),
                (&format!("{arch}.attention.head_count_kv"), 2u32.into()),
                ("tokenizer.ggml.tokens", vocab(262144)),
            ]);
            graph_size(
                &kv,
                &Tensors::default(),
                1024,
                256,
                1,
                "f16",
                FlashAttentionType::Disabled,
            )
            .unwrap()
        };
        let g2 = base("gemma2");
        let g3n = base("gemma3n");
        assert_eq!(g3n.full_offload, g2.full_offload * 4);
        assert_eq!(g3n.partial_offload, g2.partial_offload * 4);
        // gemma2 shares the formula but NOT the sliding-window cache override,
        // so its layers all cost the full context.
        assert!(g2.kv_cache.iter().all(|&n| n == g2.kv_cache[0]));
    }

    #[test]
    fn gptoss_alternates_window_and_full_context_by_layer_parity() {
        let kv = gptoss_kv();
        let got = graph_size(
            &kv,
            &Tensors::default(),
            4096,
            512,
            2,
            "f16",
            FlashAttentionType::Disabled,
        )
        .unwrap();

        // per token: (64 K + 64 V) x 8 KV heads x 2 B = 2048 B.
        // even layers: window 4096 x 2 sequences + 512 batch = 8704 tokens.
        // odd layers: the aggregate context, 4096 x 2 = 8192 tokens.
        assert_eq!(
            got.kv_cache,
            vec![17_825_792, 16_777_216, 17_825_792, 16_777_216]
        );

        // ((2 * 64) / 8) * kvTotal / 6, where kvTotal is the DEFAULT loop's
        // total (4 x 16_777_216 = 67_108_864), not the gptoss cache above.
        assert_eq!(got.partial_offload, 178_956_970);
        // gptoss never assigns fullOffload -- upstream leaves it at Go's zero.
        assert_eq!(got.full_offload, 0);
    }

    #[test]
    fn gptoss_with_flash_attention_switches_to_the_flat_mib_estimate() {
        let got = graph_size(
            &gptoss_kv(),
            &Tensors::default(),
            4096,
            512,
            2,
            "f16",
            FlashAttentionType::Enabled,
        )
        .unwrap();
        // (4 per sequence x 2) + (8192 aggregate tokens >> 10) + 110 = 126 MiB.
        assert_eq!(got.partial_offload, 126 * MEBIBYTE);
        assert_eq!(got.partial_offload, 132_120_576);

        // Auto is NOT Enabled: it has not been resolved yet, so it costs the
        // non-flash estimate.
        let auto = graph_size(
            &gptoss_kv(),
            &Tensors::default(),
            4096,
            512,
            2,
            "f16",
            FlashAttentionType::Auto,
        )
        .unwrap();
        assert_eq!(auto.partial_offload, 178_956_970);
    }

    fn gptoss_kv() -> Kv {
        kv_of(vec![
            ("general.architecture", "gptoss".into()),
            ("gptoss.block_count", 4u32.into()),
            ("gptoss.embedding_length", 2880u32.into()),
            ("gptoss.attention.head_count", 64u32.into()),
            ("gptoss.attention.head_count_kv", 8u32.into()),
            ("gptoss.attention.key_length", 64u32.into()),
            ("gptoss.attention.value_length", 64u32.into()),
            ("tokenizer.ggml.tokens", vocab(201088)),
        ])
    }

    #[test]
    fn recurrent_layers_cache_ssm_state_and_ignore_context_entirely() {
        let kv = kv_of(vec![
            ("general.architecture", "mamba2".into()),
            ("mamba2.block_count", 2u32.into()),
            ("mamba2.embedding_length", 2048u32.into()),
            ("mamba2.attention.head_count", 0u32.into()),
            ("mamba2.attention.head_count_kv", 0u32.into()),
            ("mamba2.ssm.conv_kernel", 4u32.into()),
            ("mamba2.ssm.inner_size", 4096u32.into()),
            ("mamba2.ssm.state_size", 16u32.into()),
            ("mamba2.ssm.group_count", 1u32.into()),
            ("tokenizer.ggml.tokens", vocab(32000)),
        ]);
        let sized = |context: u64| {
            graph_size(
                &kv,
                &Tensors::default(),
                context,
                256,
                1,
                "f16",
                FlashAttentionType::Disabled,
            )
            .unwrap()
        };

        // n_embd_r = (4-1) * (4096 + 2*1*16) = 12_384 conv-history elements.
        // n_embd_s = 16 * 4096 = 65_536 state elements.
        // Recurrent state is ALWAYS f32, so x4 B -> 311_680 B per layer.
        let got = sized(2048);
        assert_eq!(got.kv_cache, vec![311_680, 311_680]);

        // The whole point of an SSM: 64x the context, identical cache.
        assert_eq!(sized(131_072).kv_cache, got.kv_cache);
    }

    #[test]
    fn a_hybrid_model_mixes_attention_and_recurrent_layers() {
        // nemotron-h style: layer 0 recurrent (0 heads), layer 1 full attention.
        // The per-layer head-count ARRAY is what selects the branch.
        let kv = kv_of(vec![
            ("general.architecture", "nemotron_h".into()),
            ("nemotron_h.block_count", 2u32.into()),
            ("nemotron_h.embedding_length", 2048u32.into()),
            (
                "nemotron_h.attention.head_count",
                KvArray::new(KvArrayValues::I32(vec![0, 8])).into(),
            ),
            (
                "nemotron_h.attention.head_count_kv",
                KvArray::new(KvArrayValues::I32(vec![0, 8])).into(),
            ),
            ("nemotron_h.attention.key_length", 128u32.into()),
            ("nemotron_h.attention.value_length", 128u32.into()),
            ("nemotron_h.ssm.conv_kernel", 4u32.into()),
            ("nemotron_h.ssm.inner_size", 4096u32.into()),
            ("nemotron_h.ssm.state_size", 16u32.into()),
            ("nemotron_h.ssm.group_count", 1u32.into()),
            ("tokenizer.ggml.tokens", vocab(32000)),
        ]);
        let got = graph_size(
            &kv,
            &Tensors::default(),
            512,
            128,
            1,
            "f16",
            FlashAttentionType::Disabled,
        )
        .unwrap();

        // layer 0: recurrent, 311_680 B, context-independent.
        // layer 1: 512 tokens x (128+128) x 8 heads x 2 B = 2_097_152 B.
        assert_eq!(got.kv_cache, vec![311_680, 2_097_152]);
    }

    #[test]
    fn an_architecture_with_no_special_case_gets_zero_graph_estimates() {
        // Upstream's silent fallback, documented on graph_size: the cache is
        // still sized, but the compute graph is estimated as free. If KOPITIAM
        // ever serves an arch upstream has not special-cased, this is why the
        // scheduler under-estimates.
        let kv = kv_of(vec![
            ("general.architecture", "some-new-arch".into()),
            ("some-new-arch.block_count", 2u32.into()),
            ("some-new-arch.embedding_length", 2048u32.into()),
            ("some-new-arch.attention.head_count", 16u32.into()),
            ("some-new-arch.attention.head_count_kv", 16u32.into()),
            ("tokenizer.ggml.tokens", vocab(32000)),
        ]);
        let got = graph_size(
            &kv,
            &Tensors::default(),
            1024,
            256,
            1,
            "f16",
            FlashAttentionType::Disabled,
        )
        .unwrap();
        assert!(got.kv_cache.iter().all(|&n| n > 0), "cache still sized");
        assert_eq!(got.partial_offload, 0);
        assert_eq!(got.full_offload, 0);
    }

    #[test]
    fn a_missing_vocabulary_is_an_error_not_a_panic() {
        // Upstream type-asserts and panics here. We refuse instead.
        let mut kv = llama_7b_kv();
        kv.insert("tokenizer.ggml.tokens", KvValue::U32(32000));
        assert_eq!(
            graph_size(
                &kv,
                &Tensors::default(),
                2048,
                512,
                1,
                "f16",
                FlashAttentionType::Disabled
            ),
            Err(MemoryError::MissingVocab)
        );
    }

    #[test]
    fn the_vocabulary_count_comes_from_the_declared_size_not_the_values() {
        // Real GGUFs truncate the token array on read but keep its size. If we
        // read values.len() instead, every real model would size its vocab
        // graph as zero.
        let kv = llama_7b_kv();
        assert_eq!(kv.vocab_size(), Ok(32000));
        match kv.value("tokenizer.ggml.tokens") {
            Some(KvValue::Array(a)) => assert!(a.values.is_empty(), "no strings materialised"),
            other => panic!("expected a string array, got {other:?}"),
        }
    }

    #[test]
    fn an_absurd_block_count_is_rejected_before_anything_is_allocated() {
        let mut kv = llama_7b_kv();
        kv.insert("llama.block_count", u32::MAX);
        assert_eq!(
            graph_size(
                &kv,
                &Tensors::default(),
                2048,
                512,
                1,
                "f16",
                FlashAttentionType::Disabled
            ),
            Err(MemoryError::AbsurdBlockCount {
                block_count: u64::from(u32::MAX),
                max: MAX_BLOCK_COUNT,
            })
        );
    }

    #[test]
    fn flash_attention_support_follows_the_head_widths_and_the_allow_lists() {
        // Equal, non-zero K and V head widths -> supported.
        assert!(llama_7b_kv().supports_flash_attention());

        // An embedding model (it declares a pooling type) is never supported.
        let mut embedder = llama_7b_kv();
        embedder.insert("llama.pooling_type", 1u32);
        assert!(!embedder.supports_flash_attention());

        // Hardcoded no-list beats the head-width check.
        let mut grok = llama_7b_kv();
        grok.insert("general.architecture", "grok");
        assert!(!grok.supports_flash_attention());

        // Mismatched K/V widths -> the fused kernel cannot be used.
        let mut mla = llama_7b_kv();
        mla.insert("llama.attention.key_length", 192u32);
        mla.insert("llama.attention.value_length", 128u32);
        assert!(!mla.supports_flash_attention());

        // "supported" and "on by default" are different questions.
        assert!(!llama_7b_kv().flash_attention_default());
        let mut qwen3 = llama_7b_kv();
        qwen3.insert("general.architecture", "qwen3");
        assert!(qwen3.flash_attention_default());
        assert!(qwen3.ollama_engine_required());
    }

    #[test]
    fn the_summary_reports_memory_in_binary_units() {
        let got = graph_size(
            &llama_7b_kv(),
            &Tensors::default(),
            2048,
            512,
            1,
            "f16",
            FlashAttentionType::Disabled,
        )
        .unwrap();
        let summary = got.summary();
        assert!(summary.contains("256.0 MiB"), "kv cache total: {summary}");
        assert!(summary.contains("over 32 layers"), "{summary}");
        // MiB/GiB, never MB/GB -- these are allocator numbers.
        assert!(!summary.contains("MB"), "{summary}");
    }

    #[test]
    fn the_metadata_map_prefixes_keys_with_the_architecture() {
        let kv = llama_7b_kv();
        assert_eq!(kv.architecture(), "llama");
        // "block_count" really means "llama.block_count" ...
        assert_eq!(kv.block_count(), 32);
        // ... while general.* and tokenizer.* are looked up verbatim.
        assert_eq!(kv.string("general.architecture", ""), "llama");
        assert_eq!(kv.chat_template(), "");
        // An unknown architecture still resolves, just to a namespace nothing
        // lives in -- which is how a bad general.architecture silently zeroes
        // every hyperparameter.
        let anon = kv_of(vec![("llama.block_count", 32u32.into())]);
        assert_eq!(anon.architecture(), "unknown");
        assert_eq!(anon.block_count(), 0);
    }
}


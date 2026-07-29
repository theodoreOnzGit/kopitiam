//! # `convert` -- HuggingFace safetensors -> GGUF, the whole remapping
//!
//! **Upstream:** the `convert/` package of ollama
//! `4713800b08b2ddf5e14acf8398953cf7b12f169b` (2026-07-28), MIT, Copyright (c)
//! Ollama. This is a **port**, not inspiration. Where we and ollama disagree,
//! ollama win -- and every place we deliberately go our own way say so right at
//! the point of divergence.
//!
//! Each section below names the Go file it come from. Each function name its Go
//! counterpart. Every metadata key and magic number name where it come from.
//!
//! ## What this module actually do
//!
//! You got a HuggingFace checkpoint on disk: `config.json`, `tokenizer.json`,
//! one or more `*.safetensors`. You want a GGUF. Three jobs:
//!
//! 1. **Pick the converter** from `config.json`'s `architectures[0]`
//!    (`LlamaForCausalLM` -> llama, `Qwen2ForCausalLM` -> qwen2, ...).
//! 2. **Rename every tensor** from HF's naming into ggml's
//!    (`model.layers.0.self_attn.q_proj.weight` -> `blk.0.attn_q.weight`), and
//!    for some architectures **repack the bytes** while you at it.
//! 3. **Write the metadata** each architecture need -- `llama.block_count`,
//!    `gemma3.attention.sliding_window_pattern`, the whole tokenizer block.
//!
//! Job 2 and job 3 are the *knowledge* here. The name maps and the metadata keys
//! are not derivable from first principles -- they are conventions that ggml and
//! HuggingFace arrived at separately, and getting one wrong produce a GGUF that
//! load fine and then answer rubbish. So they all sit in code with their
//! upstream cited.
//!
//! ## The seam: this crate got no GGUF writer, and that is on purpose
//!
//! `kopitiam-loader` own GGUF **reading**, and `kopitiam-ollama` deliberately
//! depend on nothing else in KOPITIAM (see `docs/ai-decisions/AID-0055`). So
//! there is no `WriteGGUF` here to call, and inventing one inside this crate
//! would invert that dependency direction.
//!
//! The fix is a **trait seam**. [`GgufWriter`] is the whole contract:
//! `write_kv` + `write_tensor`. The conversion logic is ported against the
//! trait, so:
//!
//! * every mapping is unit-testable **today** against [`RecordingGgufWriter`] --
//!   no GGUF encoder, no model download, no GPU;
//! * a real writer can live in `kopitiam-loader` (or anywhere) later and drop
//!   straight in.
//!
//! Reading got the same treatment. [`ModelFiles`] is the read seam: name a file,
//! get bytes; or name a byte range inside a file, get those bytes. A checkpoint
//! that live entirely in a `BTreeMap` ([`MemoryFiles`]) is as valid an input as
//! one on disk ([`DirFiles`]), so the safetensors header parsing and the whole
//! tensor-name mapping can be tested from literal bytes.
//!
//! **What the writer must do that this module does NOT do for it** (upstream do
//! these inside `ggml.WriteGGUF`, `fs/ggml/gguf.go:620`):
//!
//! * **Prefix unqualified keys with the architecture.** A converter that write
//!   `block_count` mean `qwen3.block_count`. Keys already starting with
//!   `<arch>.`, `general.`, `adapter.` or `tokenizer.` are left alone. Use
//!   [`qualify_key`] -- [`RecordingGgufWriter`] already do.
//! * **Sort tensors** by `blk.N` then by name, and pad each tensor's data to
//!   `general.alignment` (default 32).
//! * **Sort the KV pairs** by key.
//!
//! ## No new dependencies
//!
//! Everything here is `serde` / `serde_json` / `sha2` / `std`. That mean a few
//! wheels got reinvented on purpose, and each one say why where it live:
//!
//! * the safetensors header is 8 bytes of length + JSON, so `serde_json` is the
//!   whole parser -- no `safetensors` crate needed;
//! * `f16` and `bf16` conversion are ~20 lines each, ported bit-for-bit;
//! * the SentencePiece `tokenizer.model` is protobuf, so there is a **minimal**
//!   wire-format reader here that understand exactly the three fields
//!   `sentencepiece_model.proto` need. No `prost`, no `protobuf`.
//!
//! ## Status -- this port is PARTIAL, and here is the line
//!
//! Ported: the entry point, model-type detection, the safetensors reader
//! (including the fp8 block-scale path), the tokenizer (BPE + SentencePiece),
//! and these ten architectures -- **llama, mixtral, qwen2, qwen3/qwen3moe,
//! gemma, gemma2, gemma3, phi3, command-r, mistral3**.
//!
//! NOT ported: the pytorch `.bin` reader (`reader_torch.go` -- need a pickle
//! parser, which is a dependency this crate cannot take), LoRA adapter
//! conversion (`ConvertAdapter`, `convert_llama_adapter.go`,
//! `convert_gemma2_adapter.go`), the generic `splitDim` machinery from
//! `tensor.go` (qwen3's two uses of it are ported by hand instead -- see
//! [`Qwen3Model::tensors`]), and the remaining ~25 architectures.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::memory::{Kv, KvArray, KvArrayValues, KvValue, TensorType};

// ===========================================================================
// §0  Errors
// ===========================================================================

/// Everything that can go wrong turning a checkpoint into a GGUF.
///
/// No `unwrap` in this module -- a malformed `config.json` off the internet is
/// **input**, not a programmer mistake, so it come back as an error every time.
#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    /// A file the converter need is not in the checkpoint.
    #[error("{0} not found in checkpoint")]
    MissingFile(String),

    /// The checkpoint got no `architectures` entry we can dispatch on.
    /// **Upstream:** `errors.New("unknown architecture")` (`convert.go:279`).
    #[error("unknown architecture")]
    UnknownArchitecture,

    /// We know the name but got no converter for it yet.
    /// **Upstream:** `fmt.Errorf("unsupported architecture %q")` (`convert.go:355`).
    #[error("unsupported architecture {0:?}")]
    UnsupportedArchitecture(String),

    /// Two input tensors mapped onto the same GGUF name. Fatal -- one would
    /// silently overwrite the other. **Upstream:** `ensureUniqueTensorNames`.
    #[error("duplicate tensor name '{0}' was found for this model")]
    DuplicateTensorName(String),

    /// No `*.safetensors`, no `pytorch_model*.bin`, nothing we recognise.
    /// **Upstream:** `errors.New("unknown tensor format")` (`reader.go:98`).
    #[error("unknown tensor format")]
    UnknownTensorFormat,

    /// A safetensors `dtype` we cannot decode.
    /// **Upstream:** `fmt.Errorf("unknown data type: %s")`.
    #[error("unknown data type: {0}")]
    UnknownDataType(String),

    /// The fp8 block-scale metadata is missing or inconsistent.
    #[error("fp8: {0}")]
    Fp8(String),

    /// Tensor shape does not match the bytes behind it, or a repack got handed
    /// a shape it cannot work with.
    #[error("tensor {name}: {reason}")]
    Shape {
        /// Which tensor.
        name: String,
        /// What is wrong with it.
        reason: String,
    },

    /// `tokenizer.json` / `tokenizer.model` / `tokenizer_config.json` is not
    /// shaped the way the format say.
    #[error("tokenizer: {0}")]
    Tokenizer(String),

    /// The SentencePiece protobuf is truncated, or use a wire type we do not
    /// expect for that field.
    #[error("sentencepiece: {0}")]
    Protobuf(String),

    /// A `rope_scaling.type` no converter know what to do with. Upstream
    /// `panic("unknown rope scaling type")` here; we return instead --
    /// **deliberate divergence**, a library must not kill the process over a
    /// config file it downloaded.
    #[error("unknown rope scaling type {0:?}")]
    UnknownRopeScaling(String),

    /// The safetensors header is malformed.
    #[error("safetensors: {0}")]
    Safetensors(String),

    /// Bytes could not be read out of the checkpoint.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// JSON could not be parsed.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

// ===========================================================================
// §1  json_compat.go -- HuggingFace configs that are not legal JSON
// ===========================================================================

/// Rewrite the non-standard numeric tokens some HF configs emit -- `Infinity`,
/// `-Infinity`, `NaN` -- into `0`.
///
/// **Upstream:** `sanitizeNonFiniteJSON` (`json_compat.go:14`).
///
/// Python's `json.dump` happily write bare `Infinity`; the JSON spec do not
/// allow it and neither Go's `encoding/json` nor `serde_json` accept it. The
/// fields that carry these values are model-side metadata the converter never
/// read, so mapping them to `0` lose nothing.
///
/// Deliberately conservative, exactly like upstream:
///
/// * only rewrite **outside** quoted strings, so a chat template containing the
///   word `NaN` survive untouched;
/// * only rewrite **whole tokens** -- `NaNny` is not `NaN`, and a key like
///   `"Infinity_scale"` is not touched.
///
/// What would make this wrong: relaxing the boundary check. `[-Infinity]` must
/// become `[0]`, but `myNaN` must stay `myNaN`.
pub fn sanitize_non_finite_json(input: &[u8]) -> Vec<u8> {
    if input.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(input.len());
    let mut in_string = false;
    let mut escape = false;
    let mut i = 0usize;

    while i < input.len() {
        let c = input[i];

        if in_string {
            out.push(c);
            if escape {
                escape = false;
            } else if c == b'\\' {
                escape = true;
            } else if c == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }

        if c == b'"' {
            in_string = true;
            out.push(c);
            i += 1;
            continue;
        }

        // Order matter: `-Infinity` before `Infinity`, else the minus sign get
        // emitted and only `Infinity` become `0`, giving `-0`.
        let mut matched = false;
        for token in [&b"-Infinity"[..], &b"Infinity"[..], &b"NaN"[..]] {
            if has_token(input, i, token) {
                out.push(b'0');
                i += token.len();
                matched = true;
                break;
            }
        }
        if matched {
            continue;
        }

        out.push(c);
        i += 1;
    }

    out
}

/// **Upstream:** `hasToken` (`json_compat.go:76`). Whole-token match with JSON
/// value boundaries on both sides.
fn has_token(input: &[u8], at: usize, token: &[u8]) -> bool {
    let end = at + token.len();
    if end > input.len() || &input[at..end] != token {
        return false;
    }
    if at > 0 && !is_json_value_prefix_boundary(input[at - 1]) {
        return false;
    }
    if end < input.len() && !is_json_value_suffix_boundary(input[end]) {
        return false;
    }
    true
}

/// **Upstream:** `isJSONWhitespace` (`json_compat.go:92`).
fn is_json_whitespace(b: u8) -> bool {
    b == b' ' || b == b'\t' || b == b'\n' || b == b'\r'
}

/// **Upstream:** `isJSONValuePrefixBoundary` (`json_compat.go:96`). A value can
/// only start after `:`, `,`, `[` or whitespace.
fn is_json_value_prefix_boundary(b: u8) -> bool {
    is_json_whitespace(b) || b == b':' || b == b',' || b == b'['
}

/// **Upstream:** `isJSONValueSuffixBoundary` (`json_compat.go:100`).
fn is_json_value_suffix_boundary(b: u8) -> bool {
    is_json_whitespace(b) || b == b',' || b == b']' || b == b'}'
}

/// Drop every `null`-valued key from a JSON object tree, recursively.
///
/// **DELIBERATE DIVERGENCE from upstream** -- there is no Go counterpart,
/// because Go do not need one.
///
/// Go's `encoding/json` treat `"sliding_window": null` as "leave the Go field
/// alone", so a `uint32` field just stay `0`. `serde` treat the same input as a
/// **type error** and fail the whole parse. HuggingFace configs are full of
/// explicit nulls (`rope_scaling: null`, `head_dim: null`,
/// `sliding_window: null`), so without this every second checkpoint would refuse
/// to convert.
///
/// Stripping the key first make `#[serde(default)]` kick in, which give exactly
/// Go's zero-value semantics. A field that genuinely want to tell "absent" apart
/// from "null" must therefore not rely on this -- but none in the ported
/// converters do; they all treat both as "not set".
pub fn strip_json_nulls(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            map.retain(|_, v| !v.is_null());
            for v in map.values_mut() {
                strip_json_nulls(v);
            }
        }
        serde_json::Value::Array(items) => {
            for v in items.iter_mut() {
                strip_json_nulls(v);
            }
        }
        _ => {}
    }
}

/// Read a JSON config the way the Go converter would: sanitize the non-finite
/// tokens, drop the nulls, then deserialise.
fn parse_config<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, ConvertError> {
    let mut value: serde_json::Value = serde_json::from_slice(&sanitize_non_finite_json(bytes))?;
    strip_json_nulls(&mut value);
    Ok(serde_json::from_value(value)?)
}

// ===========================================================================
// §2  The two seams -- reading a checkpoint, writing a GGUF
// ===========================================================================

/// Where the checkpoint's bytes come from.
///
/// **Upstream:** the `fs.FS` that `ConvertModel` take (`convert.go:397`).
///
/// Two things a caller must provide: the list of file names (so `*.safetensors`
/// can be found) and the bytes. Tensor payloads are read as **ranges** so a 16
/// GB shard never has to sit in memory whole -- [`DirFiles`] seek, and only the
/// in-memory implementation cheat by slicing.
pub trait ModelFiles {
    /// Every file name in the checkpoint root, sorted.
    ///
    /// Upstream use `fs.Glob`, which return sorted matches; the sort matter
    /// because shard order decide tensor order in the output.
    fn names(&self) -> Vec<String>;

    /// Whole file, or `None` when it is not there.
    ///
    /// `None` stand in for Go's `errors.Is(err, fs.ErrNotExist)` -- every caller
    /// upstream branch on exactly that, so it belong in the type.
    fn read(&self, name: &str) -> Result<Option<Vec<u8>>, ConvertError>;

    /// `len` bytes starting at `offset` inside `name`.
    ///
    /// Default implementation read the whole file and slice, which is correct
    /// but greedy; back it with a seek when the file is big.
    fn read_range(&self, name: &str, offset: u64, len: u64) -> Result<Vec<u8>, ConvertError> {
        let bytes = self
            .read(name)?
            .ok_or_else(|| ConvertError::MissingFile(name.to_string()))?;
        let start = usize::try_from(offset).unwrap_or(usize::MAX);
        let end = start.saturating_add(usize::try_from(len).unwrap_or(usize::MAX));
        if end > bytes.len() {
            return Err(ConvertError::Safetensors(format!(
                "{name}: want bytes [{start}, {end}) but file only got {}",
                bytes.len()
            )));
        }
        Ok(bytes[start..end].to_vec())
    }

    /// Whether the file exist. Default go through [`ModelFiles::names`] so an
    /// implementation never have to read a file just to answer this.
    fn exists(&self, name: &str) -> bool {
        self.names().iter().any(|n| n == name)
    }
}

/// A checkpoint that live entirely in memory. The test workhorse.
///
/// No Go counterpart -- upstream test against `os.DirFS(t.TempDir())`. Keeping a
/// pure in-memory implementation mean the safetensors header parsing and the
/// whole name-mapping suite run from literal bytes with no filesystem at all.
#[derive(Debug, Default, Clone)]
pub struct MemoryFiles(BTreeMap<String, Vec<u8>>);

impl MemoryFiles {
    /// Empty checkpoint.
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }

    /// Add or replace one file.
    pub fn insert(&mut self, name: impl Into<String>, bytes: impl Into<Vec<u8>>) -> &mut Self {
        self.0.insert(name.into(), bytes.into());
        self
    }
}

impl FromIterator<(String, Vec<u8>)> for MemoryFiles {
    fn from_iter<T: IntoIterator<Item = (String, Vec<u8>)>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl ModelFiles for MemoryFiles {
    fn names(&self) -> Vec<String> {
        self.0.keys().cloned().collect()
    }

    fn read(&self, name: &str) -> Result<Option<Vec<u8>>, ConvertError> {
        Ok(self.0.get(name).cloned())
    }

    fn exists(&self, name: &str) -> bool {
        self.0.contains_key(name)
    }
}

/// A checkpoint sitting in one directory on disk.
///
/// **Upstream:** `os.DirFS(path)`. Non-recursive on purpose -- HuggingFace
/// checkpoints are flat, and walking deeper would pick up cache directories.
#[derive(Debug, Clone)]
pub struct DirFiles {
    root: std::path::PathBuf,
}

impl DirFiles {
    /// Point at a checkpoint directory. Nothing is read until you ask.
    pub fn new(root: impl Into<std::path::PathBuf>) -> Self {
        Self { root: root.into() }
    }
}

impl ModelFiles for DirFiles {
    fn names(&self) -> Vec<String> {
        let Ok(entries) = std::fs::read_dir(&self.root) else {
            return Vec::new();
        };
        let mut names: Vec<String> = entries
            .flatten()
            .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
            .filter_map(|e| e.file_name().into_string().ok())
            .collect();
        names.sort();
        names
    }

    fn read(&self, name: &str) -> Result<Option<Vec<u8>>, ConvertError> {
        match std::fs::read(self.root.join(name)) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    fn read_range(&self, name: &str, offset: u64, len: u64) -> Result<Vec<u8>, ConvertError> {
        use std::io::{Read, Seek, SeekFrom};

        let mut f = std::fs::File::open(self.root.join(name))?;
        f.seek(SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; usize::try_from(len).unwrap_or(0)];
        f.read_exact(&mut buf)?;
        Ok(buf)
    }

    fn exists(&self, name: &str) -> bool {
        self.root.join(name).is_file()
    }
}

/// Where the GGUF go.
///
/// **Upstream:** `ggml.WriteGGUF(f, kv, ts)` (`fs/ggml/gguf.go:620`), which this
/// crate cannot call -- see the module header on why the seam sit here.
///
/// The contract, and all of it matter:
///
/// * `write_kv` is handed the key **as the converter wrote it**. Unqualified
///   keys like `block_count` must be prefixed with the architecture before they
///   hit the file. Call [`qualify_key`].
/// * `write_tensor`'s `shape` is already in **GGUF order** -- fastest-varying
///   dimension first, i.e. the safetensors shape reversed. [`write_file`] do
///   that reversal, so a writer must not do it again.
/// * `data` is the tensor's payload already encoded to `kind`. A writer only
///   place it and pad it.
/// * A real writer must also sort KV by key, sort tensors by `blk.N` then name,
///   and pad each tensor's data to `general.alignment` (default 32).
pub trait GgufWriter {
    /// One metadata pair.
    fn write_kv(&mut self, key: &str, value: KvValue) -> Result<(), ConvertError>;

    /// One tensor: name, ggml type, GGUF-order shape, encoded bytes.
    fn write_tensor(
        &mut self,
        name: &str,
        kind: TensorType,
        shape: &[u64],
        data: &[u8],
    ) -> Result<(), ConvertError>;
}

/// Apply GGUF's architecture prefixing rule to a metadata key.
///
/// **Upstream:** the first five lines of `ggufWriteKV` (`fs/ggml/gguf.go:689`).
///
/// A key is left alone when it already start with `<arch>.`, or with any of the
/// three global namespaces `general.`, `adapter.`, `tokenizer.`. Everything else
/// get `<arch>.` glued on the front. That is why [`Qwen3Model::kv`] can write a
/// bare `block_count` and still end up with `qwen3.block_count` in the file.
///
/// What would make this wrong: dropping the `<arch>.` check. Then
/// `llama.block_count` would become `llama.llama.block_count`.
pub fn qualify_key(arch: &str, key: &str) -> String {
    if key.starts_with(&format!("{arch}."))
        || key.starts_with("general.")
        || key.starts_with("adapter.")
        || key.starts_with("tokenizer.")
    {
        key.to_string()
    } else {
        format!("{arch}.{key}")
    }
}

/// A [`GgufWriter`] that keep everything in memory so a test can assert on it.
///
/// No Go counterpart -- upstream write a real file and decode it back
/// (`convertFull`, `convert_test.go:31`). Recording instead mean the whole
/// mapping suite need no GGUF encoder and no temp files.
///
/// Apply [`qualify_key`] exactly like the real writer, so the recorded keys are
/// the keys that would land in the file.
#[derive(Debug, Default)]
pub struct RecordingGgufWriter {
    /// The architecture, taken from the `general.architecture` value as it go
    /// past. Prefixing cannot happen before that key arrive, so [`write_file`]
    /// always send it first.
    pub architecture: String,
    /// Every metadata pair, qualified. [`Kv`] is a `BTreeMap`, so this end up
    /// sorted by key like the real writer's output.
    pub kv: Kv,
    /// Every tensor: name, kind, GGUF-order shape, encoded bytes.
    pub tensors: Vec<RecordedTensor>,
}

/// One tensor as recorded by [`RecordingGgufWriter`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedTensor {
    /// GGUF tensor name, e.g. `blk.0.attn_q.weight`.
    pub name: String,
    /// ggml type the payload is encoded in.
    pub kind: TensorType,
    /// Shape in GGUF order (fastest-varying first).
    pub shape: Vec<u64>,
    /// The encoded payload.
    pub data: Vec<u8>,
}

impl RecordingGgufWriter {
    /// Fresh recorder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Find a recorded tensor by GGUF name. Handy in tests.
    pub fn tensor(&self, name: &str) -> Option<&RecordedTensor> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

impl GgufWriter for RecordingGgufWriter {
    fn write_kv(&mut self, key: &str, value: KvValue) -> Result<(), ConvertError> {
        if key == "general.architecture"
            && let KvValue::String(arch) = &value
        {
            self.architecture = arch.clone();
        }
        self.kv.insert(qualify_key(&self.architecture, key), value);
        Ok(())
    }

    fn write_tensor(
        &mut self,
        name: &str,
        kind: TensorType,
        shape: &[u64],
        data: &[u8],
    ) -> Result<(), ConvertError> {
        self.tensors.push(RecordedTensor {
            name: name.to_string(),
            kind,
            shape: shape.to_vec(),
            data: data.to_vec(),
        });
        Ok(())
    }
}

// ===========================================================================
// §3  reader.go + tensor.go -- names, kinds, and the tensor plumbing
// ===========================================================================

/// Go's `strings.Replacer`, ported, because the tensor-name mapping *is* a
/// `strings.Replacer` and its exact semantics decide every output name.
///
/// **Upstream:** `strings.NewReplacer(conv.Replacements()...)` (`convert.go:407`).
///
/// The semantics, and they are NOT "longest match wins":
///
/// * scan the input left to right;
/// * at each position, try the patterns **in the order they were given** and
///   take the **first** that match;
/// * replacements never overlap -- after a match, carry on **after** the
///   replacement, so a replacement's own text is never re-scanned.
///
/// Argument order therefore carry meaning. Look at [`Gemma3Model::replacements`]:
/// `vision_tower.vision_model.embeddings -> v` sit **before**
/// `vision_tower.vision_model -> v`, so the longer one win. Swap them and every
/// vision-embedding tensor land on the wrong name. Same story in
/// [`Mistral3Model::replacements`], where `language_model.model.norm` must come
/// before `language_model.model.` and that before `language_model.`.
///
/// And because a replacement is never re-scanned, `model.layers -> blk` then
/// `input_layernorm -> attn_norm` compose safely: `blk` cannot be eaten by a
/// later rule.
///
/// **Limitation, stated plainly:** an empty `from` is not supported. Go's
/// `Replacer` match the empty string at every position; nothing upstream use it,
/// and supporting it would only add a way to loop forever.
#[derive(Debug, Clone, Default)]
pub struct Replacer {
    pairs: Vec<(String, String)>,
}

impl Replacer {
    /// Build from `(from, to)` pairs, **in priority order**.
    pub fn new<'a>(pairs: impl IntoIterator<Item = (&'a str, &'a str)>) -> Self {
        Self {
            pairs: pairs
                .into_iter()
                .filter(|(from, _)| !from.is_empty())
                .map(|(from, to)| (from.to_string(), to.to_string()))
                .collect(),
        }
    }

    /// Apply every rule once, left to right, non-overlapping.
    pub fn replace(&self, input: &str) -> String {
        let mut out = String::with_capacity(input.len());
        let mut rest = input;

        'outer: while !rest.is_empty() {
            for (from, to) in &self.pairs {
                if let Some(tail) = rest.strip_prefix(from.as_str()) {
                    out.push_str(to);
                    rest = tail;
                    continue 'outer;
                }
            }
            // No rule fire here -- copy one whole char (never a byte, so a
            // multi-byte name cannot be split mid-codepoint) and move on.
            let mut chars = rest.chars();
            match chars.next() {
                Some(c) => {
                    out.push(c);
                    rest = chars.as_str();
                }
                None => break,
            }
        }

        out
    }
}

/// The ggml type a converted tensor get stored as, before any per-reader
/// override.
///
/// **Upstream:** `tensorBase.Kind` (`tensor.go` -> actually `reader.go:41`).
///
/// Two rules, and the first one override the second:
///
/// 1. **A hard-coded list is always F32.** These are tensors whose *values* are
///    read directly by the runtime rather than fed through a matmul, so f16
///    rounding would corrupt them outright:
///    * `*.ffn_gate_inp.weight` -- the MoE router. Rounding the router's logits
///      change which expert get picked.
///    * `*.bias` -- biases are added, never quantised.
///    * `*.shortconv.conv.weight`, `*.ssm_conv1d.weight` -- SSM conv kernels;
///      upstream note Metal require F32 here.
///    * `a.feature_extractor.*` -- audio constants read with `BackendGet`, so
///      they must be real F32 values, not a lossy re-encode.
///    * `a.conv1d.*`, `a.subsampling.*`, `*.conv_dw.*` -- audio conv weights
///      kept F32 for im2col / conv stability. Upstream's own comment say this
///      probably slow audio down and should be revisited; we keep the behaviour
///      *and* the caveat.
///    * `token_types.weight`, the whole `v.*_position_embd` family,
///      `s.position_embd`, `*rel_pos_h`, `*rel_pos_w` -- position tables that
///      get indexed, not multiplied.
/// 2. **Otherwise it is rank-based.** Rank 1 (a vector: norms, biases) -> F32.
///    Rank >= 2 -> F16. Rank 0 is not legal and upstream `panic` -- we return
///    an error instead (**deliberate divergence**: a library must not kill the
///    process because a checkpoint carry a rubbish shape).
///
/// What would make this wrong: adding a suffix to the F32 list that is actually
/// a matmul weight. That would silently double the file size for no gain.
pub fn base_tensor_kind(name: &str, shape: &[u64]) -> Result<TensorType, ConvertError> {
    let always_f32 = name.ends_with(".ffn_gate_inp.weight")
        || name.ends_with(".bias")
        || name.ends_with(".shortconv.conv.weight")
        || name.ends_with(".ssm_conv1d.weight")
        || name.starts_with("a.feature_extractor.")
        || name.starts_with("a.conv1d.")
        || name.starts_with("a.subsampling.")
        || name.contains(".conv_dw.")
        || name == "token_types.weight"
        || name == "v.positional_embedding_vlm"
        || name == "v.position_embd.weight"
        || name == "v.tile_position_embd.weight"
        || name == "v.pre_tile_position_embd.weight"
        || name == "v.post_tile_position_embd.weight"
        || name == "s.position_embd"
        || name.ends_with("rel_pos_h")
        || name.ends_with("rel_pos_w");

    if always_f32 {
        return Ok(TensorType::F32);
    }

    match shape.len() {
        0 => Err(ConvertError::Shape {
            name: name.to_string(),
            reason: "invalid tensor shape: rank 0".to_string(),
        }),
        1 => Ok(TensorType::F32),
        _ => Ok(TensorType::F16),
    }
}

/// A function that rewrite a tensor's values on the way out.
///
/// **Upstream:** `type Repacker func(string, []float32, []uint64) ([]float32, error)`
/// (`reader.go:76`).
///
/// It is handed the **GGUF** name, the decoded values, and the **safetensors**
/// shape (the input shape, not the output one). It return the flattened values
/// to write. Used for llama's rope permute, gemma's `+1` on norms, gemma3's
/// vocabulary truncation, and qwen3's expert transposes.
pub type Repacker =
    Arc<dyn Fn(&str, Vec<f32>, &[u64]) -> Result<Vec<f32>, ConvertError> + Send + Sync>;

/// One tensor as it exist in the **input** checkpoint: already renamed into GGUF
/// naming, but still pointing at safetensors bytes.
///
/// **Upstream:** the `Tensor` interface plus the `safetensor` struct that
/// implement it (`reader.go:10`, `reader_safetensors.go:118`).
#[derive(Clone)]
pub struct SourceTensor {
    /// GGUF name -- the [`Replacer`] already ran.
    pub name: String,
    /// Shape as safetensors state it: **slowest-varying dimension first**, the
    /// opposite of GGUF. [`write_file`] reverse it at the very end.
    pub shape: Vec<u64>,
    /// safetensors `dtype` string: `F32`, `F16`, `BF16`, `U8`, `F8_E4M3`.
    pub dtype: String,
    /// Which `*.safetensors` file the bytes live in.
    pub file: String,
    /// Absolute byte offset into that file (already includes the 8-byte length
    /// prefix and the JSON header -- see [`safetensors_pad`]).
    pub offset: u64,
    /// Byte length of the payload.
    pub size: u64,
    /// The fp8 scale companion, when `dtype == "F8_E4M3"`.
    pub scale: Option<SafetensorScale>,
    /// The fp8 block size from `config.json`, when `dtype == "F8_E4M3"`.
    pub fp8_block: Option<Fp8BlockSize>,
    /// Optional value rewrite, set by the architecture converter.
    pub repacker: Option<Repacker>,
}

impl std::fmt::Debug for SourceTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SourceTensor")
            .field("name", &self.name)
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("file", &self.file)
            .field("offset", &self.offset)
            .field("size", &self.size)
            .field("repacker", &self.repacker.is_some())
            .finish()
    }
}

impl SourceTensor {
    /// The ggml type this tensor get written as.
    ///
    /// **Upstream:** `safetensor.Kind` (`reader_safetensors.go:135`), which wrap
    /// `tensorBase.Kind`.
    ///
    /// On top of [`base_tensor_kind`], two source-dtype overrides:
    ///
    /// * **`BF16` stay `BF16`** rather than being narrowed to F16 -- but *only*
    ///   for the text model. Anything named `v.*` (vision), `s.*` (speech),
    ///   `mm.*` (multimodal projector), or containing
    ///   `ffn_gate_inp_shexp.weight`, is excluded and go to F16. Upstream give no
    ///   reason; the effect is that projector weights stay in the format the
    ///   vision code path expect.
    /// * **`F8_E4M3` become `BF16`**, because there is no fp8 tensor type in
    ///   GGUF -- the values are dequantised through their block scales and
    ///   re-encoded. See `SourceTensor::decode_fp8_e4m3`.
    ///
    /// Both override are skipped when the base kind is F32: an F32-by-name
    /// tensor stay F32 no matter what the source dtype was.
    pub fn kind(&self) -> Result<TensorType, ConvertError> {
        let base = base_tensor_kind(&self.name, &self.shape)?;

        if self.dtype == "BF16"
            && !self.name.starts_with("v.")
            && !self.name.starts_with("s.")
            && !self.name.starts_with("mm.")
            && !self.name.contains("ffn_gate_inp_shexp.weight")
            && base != TensorType::F32
        {
            return Ok(TensorType::BF16);
        }

        if self.dtype == "F8_E4M3" && base != TensorType::F32 {
            return Ok(TensorType::BF16);
        }

        Ok(base)
    }

    /// **Upstream:** `tensorBase.SetRepacker` (`reader.go:72`).
    pub fn set_repacker(&mut self, f: Repacker) {
        self.repacker = Some(f);
    }

    /// Turn this input tensor into an output tensor with its own name and shape
    /// unchanged. The overwhelmingly common case -- most converters do exactly
    /// this for most tensors.
    fn passthrough(self) -> Result<OutTensor, ConvertError> {
        let kind = self.kind()?;
        Ok(OutTensor {
            name: self.name.clone(),
            kind,
            shape: self.shape.clone(),
            source: TensorSource::Input(Box::new(self)),
        })
    }
}

/// Where an output tensor's bytes come from.
///
/// **Upstream:** the `io.WriterTo` field of `ggml.Tensor`. Go can hold any
/// writer there; we enumerate the three that the ported converters actually
/// produce, which keep the whole thing inspectable.
#[derive(Debug, Clone)]
pub enum TensorSource {
    /// Straight from one input tensor (possibly repacked).
    Input(Box<SourceTensor>),
    /// A literal `f32` vector computed by the converter -- llama's
    /// `rope_freqs.weight`, phi3's `rope_factors_{long,short}.weight`.
    /// **Upstream:** `ropeFactor` (`convert_phi3.go:117`).
    Literal(Vec<f32>),
    /// Several input tensors concatenated in order, which is how per-expert
    /// weights get merged into one `ffn_*_exps` tensor.
    /// **Upstream:** `mergeGroup` (`tensor.go:141`).
    Merge(Vec<SourceTensor>),
}

/// One tensor as it will be written.
///
/// **Upstream:** `ggml.Tensor` (`fs/ggml/ggml.go:384`) as the `convert` package
/// build it -- name, kind, shape, and something that can produce the bytes.
///
/// `shape` is still in **safetensors order** here. [`write_file`] reverse it.
#[derive(Debug, Clone)]
pub struct OutTensor {
    /// GGUF tensor name.
    pub name: String,
    /// ggml type the payload will be encoded to.
    pub kind: TensorType,
    /// Shape, slowest-varying first (reversed at write time).
    pub shape: Vec<u64>,
    /// Where the bytes come from.
    pub source: TensorSource,
}

impl OutTensor {
    /// A literal `f32` vector, kind F32, rank 1.
    ///
    /// **Upstream:** how `llamaModel.Tensors` build `rope_freqs.weight` with
    /// `Kind: 0` (`convert_llama.go:110`) -- `0` being `TensorType::F32`.
    pub fn literal(name: impl Into<String>, values: Vec<f32>) -> Self {
        let len = values.len() as u64;
        Self {
            name: name.into(),
            kind: TensorType::F32,
            shape: vec![len],
            source: TensorSource::Literal(values),
        }
    }
}

/// Reject a model whose tensors would collide on name.
///
/// **Upstream:** `ensureUniqueTensorNames` (`convert.go:443`).
///
/// This is not paranoia. The replacement rules are aggressive substring edits,
/// and a checkpoint carrying both `model.layers.0.self_attn.q_proj.weight` and
/// some other tensor that happen to map onto `blk.0.attn_q.weight` would produce
/// a GGUF where one silently shadow the other. Better to refuse.
pub fn ensure_unique_tensor_names(ts: &[SourceTensor]) -> Result<(), ConvertError> {
    let mut seen = BTreeSet::new();
    for t in ts {
        if !seen.insert(t.name.as_str()) {
            return Err(ConvertError::DuplicateTensorName(t.name.clone()));
        }
    }
    Ok(())
}

/// One merge rule: every input tensor whose name match `pattern` become one
/// output tensor called `name`.
///
/// **Upstream:** `type merge struct { pattern, name string }` (`tensor.go:86`).
#[derive(Debug, Clone)]
pub struct Merge {
    /// A glob, matched with [`glob_match`]. Mixtral use `blk.0.*.w1.weight`.
    pub pattern: String,
    /// The merged GGUF name, e.g. `blk.0.ffn_gate_exps.weight`.
    pub name: String,
}

impl Merge {
    /// Build one rule.
    pub fn new(pattern: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
            name: name.into(),
        }
    }
}

/// `path.Match` semantics, cut down to what the merge patterns need.
///
/// **Upstream:** `path.Match(merges[i].pattern, t.Name())` (`tensor.go:93`).
///
/// The important quirk, and it is the whole reason this is not a plain glob:
/// in Go's `path.Match`, **`*` does not cross a `/`**. Tensor names contain no
/// `/`, so here `*` match any run of characters including `.` -- which is what
/// make `blk.0.*.w1.weight` match `blk.0.7.w1.weight` (expert 7). Only `*` is
/// supported; upstream's patterns use nothing else.
pub fn glob_match(pattern: &str, name: &str) -> bool {
    fn helper(p: &[u8], n: &[u8]) -> bool {
        if p.is_empty() {
            return n.is_empty();
        }
        if p[0] == b'*' {
            // Try every split point. Patterns are tiny, so the quadratic
            // fallback cost nothing.
            for i in 0..=n.len() {
                if helper(&p[1..], &n[i..]) {
                    return true;
                }
            }
            return false;
        }
        if n.is_empty() || p[0] != n[0] {
            return false;
        }
        helper(&p[1..], &n[1..])
    }
    helper(pattern.as_bytes(), name.as_bytes())
}

/// Collapse per-expert tensors into single stacked tensors.
///
/// **Upstream:** `mergeTensors` (`tensor.go:89`).
///
/// Returns `(merged, leftover)`. Each rule consume the tensors it match, so a
/// later rule never see them again.
///
/// Two details that decide whether the result is correct:
///
/// * **The sort.** Matched tensors are sorted by splitting the name on `.` and
///   comparing component by component, **numerically when both components parse
///   as integers**. Plain lexicographic order would put expert `10` before
///   expert `2` and scramble the stack. Upstream sort first by *number of
///   components*, then component-wise.
/// * **The shape.** The merged tensor gain a new leading dimension equal to the
///   number of tensors merged, so `n` experts of shape `[a, b]` become
///   `[n, a, b]`. Kind is taken from the first member.
pub fn merge_tensors(
    unmatched: Vec<SourceTensor>,
    merges: &[Merge],
) -> Result<(Vec<OutTensor>, Vec<SourceTensor>), ConvertError> {
    let mut out = Vec::new();
    let mut rest = unmatched;

    for m in merges {
        let (mut matched, remaining): (Vec<_>, Vec<_>) =
            rest.into_iter().partition(|t| glob_match(&m.pattern, &t.name));
        rest = remaining;

        matched.sort_by(|a, b| compare_tensor_names(&a.name, &b.name));

        if let Some(first) = matched.first() {
            let kind = first.kind()?;
            let mut shape = vec![matched.len() as u64];
            shape.extend_from_slice(&first.shape);
            out.push(OutTensor {
                name: m.name.clone(),
                kind,
                shape,
                source: TensorSource::Merge(matched),
            });
        }
    }

    Ok((out, rest))
}

/// **Upstream:** the `slices.SortStableFunc` comparator inside `mergeTensors`
/// (`tensor.go:98`). Component-wise, numeric where possible.
fn compare_tensor_names(a: &str, b: &str) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    let x: Vec<&str> = a.split('.').collect();
    let y: Vec<&str> = b.split('.').collect();
    if x.len() != y.len() {
        return x.len().cmp(&y.len());
    }

    for (xi, yi) in x.iter().zip(y.iter()) {
        let ord = match (xi.parse::<i64>(), yi.parse::<i64>()) {
            (Ok(m), Ok(n)) => m.cmp(&n),
            _ => xi.cmp(yi),
        };
        if ord != Ordering::Equal {
            return ord;
        }
    }

    Ordering::Equal
}

/// Record which tensors arrived as fp8, so a downstream consumer know the
/// weights went through a lossy dequantise.
///
/// **Upstream:** `sourceTensorKV` (`tensor.go:161`). Emit nothing at all when no
/// tensor was fp8 -- an all-bf16 model must not gain these keys.
///
/// * `source_quantization = "hf_fp8"` -- a plain marker string.
/// * `source_fp8_tensors` -- the sorted list of GGUF names that were fp8.
///
/// A merged tensor count as fp8 only when **every** member was fp8; a mixed
/// group report nothing, because "half of this tensor was fp8" is not a fact
/// anyone can act on.
pub fn source_tensor_kv(ts: &[OutTensor]) -> Option<(String, Vec<String>)> {
    let mut names = BTreeSet::new();

    for t in ts {
        let dtype = match &t.source {
            TensorSource::Input(st) => st.dtype.clone(),
            TensorSource::Literal(_) => String::new(),
            TensorSource::Merge(group) => match group.split_first() {
                None => String::new(),
                Some((head, tail)) => {
                    if tail.iter().all(|m| m.dtype == head.dtype) {
                        head.dtype.clone()
                    } else {
                        String::new()
                    }
                }
            },
        };
        if dtype == "F8_E4M3" {
            names.insert(t.name.clone());
        }
    }

    if names.is_empty() {
        None
    } else {
        Some(("hf_fp8".to_string(), names.into_iter().collect()))
    }
}

/// Find and parse the checkpoint's tensors.
///
/// **Upstream:** `parseTensors` (`reader.go:80`).
///
/// Patterns are tried **in order** and the first one that match anything win:
///
/// 1. `*.safetensors`
/// 2. `pytorch_model-*-of-*.bin`
/// 3. `pytorch_model.bin`
/// 4. `consolidated.*.pth`
///
/// **Only pattern 1 is ported.** Patterns 2-4 go through `reader_torch.go`,
/// which need a Python pickle parser (`gopickle` upstream) -- a dependency this
/// crate cannot take. A checkpoint that only ship `.bin` files therefore come
/// back as [`ConvertError::UnknownTensorFormat`], with the same error upstream
/// would give for a checkpoint it does not recognise at all. That is a **known
/// gap**, not a silent one.
pub fn parse_tensors(
    files: &dyn ModelFiles,
    replacer: &Replacer,
) -> Result<Vec<SourceTensor>, ConvertError> {
    let names = files.names();

    let safetensors: Vec<String> = names
        .iter()
        .filter(|n| n.ends_with(".safetensors"))
        .cloned()
        .collect();
    if !safetensors.is_empty() {
        return parse_safetensors(files, replacer, &safetensors);
    }

    Err(ConvertError::UnknownTensorFormat)
}

// ===========================================================================
// §4  reader_safetensors.go -- the safetensors container, byte for byte
// ===========================================================================

/// The safetensors file format, all of it, since this is the one place the
/// knowledge live:
///
/// ```text
/// [0 .. 8)          u64 little-endian: N, the byte length of the JSON header
/// [8 .. 8+N)        the JSON header
/// [8+N .. EOF)      the tensor data, one blob after another
/// ```
///
/// The header is a JSON object mapping tensor name -> `{dtype, shape,
/// data_offsets}`. `data_offsets` is `[start, end)` **relative to the start of
/// the data section**, i.e. relative to `8+N`. There is also an optional
/// `__metadata__` key whose value is a free-form string map and carry no
/// `dtype`; it is skipped.
///
/// **Upstream:** `safetensorMetadata` + `parseSafetensors`
/// (`reader_safetensors.go:21`).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct SafetensorMetadata {
    #[serde(rename = "dtype")]
    dtype: String,
    shape: Vec<u64>,
    data_offsets: Vec<u64>,
}

/// Turn a data-section-relative offset into an absolute file offset.
///
/// **Upstream:** `safetensorsPad` (`reader_safetensors.go:111`), whose name is a
/// bit of a misnomer -- nothing is padded, it just add the 8-byte length prefix
/// and the header length.
///
/// `absolute = 8 + header_len + relative`.
pub fn safetensors_pad(header_len: u64, offset: u64) -> u64 {
    8 + header_len + offset
}

/// The fp8 block-scale grid: one scale per `rows x cols` block of weights.
///
/// **Upstream:** `safetensorFP8BlockSize` (`reader_safetensors.go:373`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp8BlockSize {
    /// Rows per block.
    pub rows: usize,
    /// Columns per block.
    pub cols: usize,
}

/// The companion tensor holding an fp8 weight's per-block scales.
///
/// **Upstream:** `safetensorScale` (`reader_safetensors.go:115`).
#[derive(Debug, Clone)]
pub struct SafetensorScale {
    /// The scale tensor's own name in the header, e.g. `...weight_scale_inv`.
    pub name: String,
    /// Its dtype -- `F32`, `F16` or `BF16`.
    pub dtype: String,
    /// Its shape, which must be `ceil(rows/block.rows) x ceil(cols/block.cols)`.
    pub shape: Vec<u64>,
    /// Absolute byte offset.
    pub offset: u64,
    /// Byte length.
    pub size: u64,
}

/// Read every `*.safetensors` shard's header and produce the input tensors.
///
/// **Upstream:** `parseSafetensors` (`reader_safetensors.go:27`).
///
/// Things that are easy to get wrong and are therefore spelled out:
///
/// * Header keys are visited in **sorted order**, not JSON order. Tensor order
///   in the output follow from it.
/// * A **0-dim** tensor (a scalar, e.g. a clipped-linear min/max) is promoted to
///   shape `[1]`, because GGUF got no rank-0 tensor.
/// * The name is renamed by `replacer` **here**, and duplicates are rejected
///   per-shard as well as globally.
/// * An fp8 weight's scale companion is **consumed** -- it never become a tensor
///   of its own.
pub fn parse_safetensors(
    files: &dyn ModelFiles,
    replacer: &Replacer,
    paths: &[String],
) -> Result<Vec<SourceTensor>, ConvertError> {
    let fp8_block = safetensors_fp8_block_size(files)?;
    let mut ts = Vec::new();

    for path in paths {
        let raw = files
            .read(path)?
            .ok_or_else(|| ConvertError::MissingFile(path.clone()))?;
        if raw.len() < 8 {
            return Err(ConvertError::Safetensors(format!(
                "{path}: file is {} bytes, too short for the 8-byte header length",
                raw.len()
            )));
        }

        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&raw[..8]);
        let header_len = u64::from_le_bytes(len_bytes);
        let header_end = 8usize.saturating_add(usize::try_from(header_len).unwrap_or(usize::MAX));
        if header_end > raw.len() {
            return Err(ConvertError::Safetensors(format!(
                "{path}: header claim {header_len} bytes but only {} left in the file",
                raw.len().saturating_sub(8)
            )));
        }

        let headers: BTreeMap<String, SafetensorMetadata> =
            serde_json::from_slice(&raw[8..header_end])?;

        let scales = collect_safetensors_fp8_scales(header_len, &headers)?;
        let mut names_this_shard = BTreeSet::new();

        for (key, value) in &headers {
            // `__metadata__` and anything else without a dtype is not a tensor.
            if value.dtype.is_empty() {
                continue;
            }
            if scales.consumed.contains(key) {
                continue;
            }

            let mut shape = value.shape.clone();
            if shape.is_empty() {
                // Scalar tensors are 0-dim in safetensors; GGUF cannot hold
                // rank 0, so promote to a 1-element vector.
                shape.push(1);
            }

            if value.data_offsets.len() < 2 {
                return Err(ConvertError::Safetensors(format!(
                    "{path}: tensor {key:?} got no data_offsets pair"
                )));
            }

            let mut scale = None;
            if value.dtype == "F8_E4M3" {
                if fp8_block.is_none() {
                    return Err(ConvertError::Fp8(format!(
                        "missing fp8 block size metadata for tensor {key:?}"
                    )));
                }
                scale = Some(scales.by_weight.get(key).cloned().ok_or_else(|| {
                    ConvertError::Fp8(format!("missing fp8 scale companion for tensor {key:?}"))
                })?);
            }

            let gguf_name = replacer.replace(key);
            if !names_this_shard.insert(gguf_name.clone()) {
                return Err(ConvertError::DuplicateTensorName(gguf_name));
            }

            let start = safetensors_pad(header_len, value.data_offsets[0]);
            let end = safetensors_pad(header_len, value.data_offsets[1]);

            ts.push(SourceTensor {
                name: gguf_name,
                shape,
                dtype: value.dtype.clone(),
                file: path.clone(),
                offset: start,
                size: end.saturating_sub(start),
                scale,
                fp8_block,
                repacker: None,
            });
        }
    }

    Ok(ts)
}

/// The fp8 weight -> scale-companion index for one shard.
///
/// **Upstream:** `safetensorsFP8Scales` (`reader_safetensors.go:292`).
#[derive(Debug, Default)]
struct Fp8Scales {
    by_weight: BTreeMap<String, SafetensorScale>,
    consumed: BTreeSet<String>,
}

/// **Upstream:** `collectSafetensorsFP8Scales` (`reader_safetensors.go:297`).
fn collect_safetensors_fp8_scales(
    header_len: u64,
    headers: &BTreeMap<String, SafetensorMetadata>,
) -> Result<Fp8Scales, ConvertError> {
    let mut scales = Fp8Scales::default();

    for (key, value) in headers {
        if value.dtype != "F8_E4M3" {
            continue;
        }

        let Some((scale_key, scale_value)) = safetensors_fp8_scale(key, headers)? else {
            continue;
        };
        if scales.consumed.contains(&scale_key) {
            return Err(ConvertError::Fp8(format!(
                "fp8 scale companion {scale_key:?} is used by multiple tensors"
            )));
        }
        if scale_value.data_offsets.len() < 2 {
            return Err(ConvertError::Safetensors(format!(
                "fp8 scale {scale_key:?} got no data_offsets pair"
            )));
        }

        let start = safetensors_pad(header_len, scale_value.data_offsets[0]);
        let end = safetensors_pad(header_len, scale_value.data_offsets[1]);
        scales.by_weight.insert(
            key.clone(),
            SafetensorScale {
                name: scale_key.clone(),
                dtype: scale_value.dtype.clone(),
                shape: scale_value.shape.clone(),
                offset: start,
                size: end.saturating_sub(start),
            },
        );
        scales.consumed.insert(scale_key);
    }

    Ok(scales)
}

/// Find the one scale tensor belonging to an fp8 weight.
///
/// **Upstream:** `safetensorsFP8Scale` + `safetensorsFP8ScaleCandidates`
/// (`reader_safetensors.go:335`).
///
/// The naming is not standardised across exporters, so four suffix forms are
/// tried on the weight's own key -- `_scale`, `_scale_inv`, `.scale`,
/// `.scale_inv` -- plus, when the key end in `.weight`, the two
/// compressed-tensors forms that put the scale name *before* the suffix:
/// `<base>.weight_scale` and `<base>.weight_scale_inv`.
///
/// Finding **more than one** is an error, not a pick-the-first: two candidates
/// mean we cannot tell which grid the weights were quantised against, and
/// guessing would corrupt the values silently.
fn safetensors_fp8_scale(
    key: &str,
    headers: &BTreeMap<String, SafetensorMetadata>,
) -> Result<Option<(String, SafetensorMetadata)>, ConvertError> {
    let mut candidates: Vec<String> = Vec::new();
    let push_unique = |c: String, v: &mut Vec<String>| {
        if !v.contains(&c) {
            v.push(c);
        }
    };

    push_unique(format!("{key}_scale"), &mut candidates);
    push_unique(format!("{key}_scale_inv"), &mut candidates);
    push_unique(format!("{key}.scale"), &mut candidates);
    push_unique(format!("{key}.scale_inv"), &mut candidates);

    if let Some(base) = key.strip_suffix(".weight") {
        push_unique(format!("{base}.weight_scale"), &mut candidates);
        push_unique(format!("{base}.weight_scale_inv"), &mut candidates);
    }

    let mut found: Option<(String, SafetensorMetadata)> = None;
    for candidate in candidates {
        if let Some(value) = headers.get(&candidate)
            && !value.dtype.is_empty()
        {
            if let Some((existing, _)) = &found {
                return Err(ConvertError::Fp8(format!(
                    "multiple fp8 scale companions for tensor {key:?}: {existing:?} and {candidate:?}"
                )));
            }
            found = Some((candidate, value.clone()));
        }
    }

    Ok(found)
}

/// The four shapes a `config.json` can state an fp8 block size in.
///
/// **Upstream:** `safetensorsSourceQuantization` (`reader_safetensors.go:379`).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct SourceQuantization {
    quant_method: String,
    format: String,
    weight_block_size: Vec<i64>,
    config_groups: BTreeMap<String, ConfigGroup>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct ConfigGroup {
    format: String,
    weights: ConfigGroupWeights,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct ConfigGroupWeights {
    block_structure: Vec<i64>,
    num_bits: i64,
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct SafetensorsModelConfig {
    quantization: SourceQuantization,
    quantization_config: SourceQuantization,
    compression_config: SourceQuantization,
    text_config: SafetensorsTextConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct SafetensorsTextConfig {
    quantization: SourceQuantization,
    quantization_config: SourceQuantization,
    compression_config: SourceQuantization,
}

/// Dig the fp8 block size out of `config.json`.
///
/// **Upstream:** `safetensorsFP8BlockSize` (`reader_safetensors.go:409`).
///
/// Six places are checked, because three exporters each use a different key and
/// a multimodal checkpoint may bury it under `text_config`:
/// `quantization`, `quantization_config`, `compression_config`, and the same
/// three again inside `text_config`.
///
/// Two source shapes are understood:
///
/// * `quant_method == "fp8"` with a 2-element `weight_block_size` (DeepSeek's
///   form -- typically `[128, 128]`);
/// * compressed-tensors: `quant_method == "compressed-tensors"` **or**
///   `format == "float-quantized"`, then each entry of `config_groups` whose
///   weights are `float`, 8-bit, with a 2-element `block_structure`.
///
/// If two of them disagree, that is an error -- a single file cannot have two
/// block grids, and picking one would dequantise half the model wrongly.
/// No fp8 metadata at all is fine and give `None`; only an actual `F8_E4M3`
/// tensor then turn that into an error.
fn safetensors_fp8_block_size(
    files: &dyn ModelFiles,
) -> Result<Option<Fp8BlockSize>, ConvertError> {
    let Some(bytes) = files.read("config.json")? else {
        return Ok(None);
    };
    let cfg: SafetensorsModelConfig = parse_config(&bytes)?;

    let mut blocks: Vec<Fp8BlockSize> = Vec::new();
    let candidates = [
        &cfg.quantization,
        &cfg.quantization_config,
        &cfg.compression_config,
        &cfg.text_config.quantization,
        &cfg.text_config.quantization_config,
        &cfg.text_config.compression_config,
    ];

    for q in candidates {
        if q.quant_method.eq_ignore_ascii_case("fp8") && q.weight_block_size.len() == 2 {
            blocks.push(new_fp8_block_size(
                q.weight_block_size[0],
                q.weight_block_size[1],
            )?);
        }

        if !q.quant_method.eq_ignore_ascii_case("compressed-tensors")
            && !q.format.eq_ignore_ascii_case("float-quantized")
        {
            continue;
        }
        for group in q.config_groups.values() {
            if !group.format.eq_ignore_ascii_case("float-quantized")
                || group.weights.num_bits != 8
                || !group.weights.kind.eq_ignore_ascii_case("float")
                || group.weights.block_structure.len() != 2
            {
                continue;
            }
            blocks.push(new_fp8_block_size(
                group.weights.block_structure[0],
                group.weights.block_structure[1],
            )?);
        }
    }

    let Some((first, rest)) = blocks.split_first() else {
        return Ok(None);
    };
    for other in rest {
        if other != first {
            return Err(ConvertError::Fp8(format!(
                "multiple fp8 block sizes in config.json: {}x{} and {}x{}",
                first.rows, first.cols, other.rows, other.cols
            )));
        }
    }
    Ok(Some(*first))
}

/// **Upstream:** `newSafetensorFP8BlockSize` (`reader_safetensors.go:472`).
fn new_fp8_block_size(rows: i64, cols: i64) -> Result<Fp8BlockSize, ConvertError> {
    if rows <= 0 || cols <= 0 {
        return Err(ConvertError::Fp8(format!(
            "invalid fp8 block size {rows}x{cols}"
        )));
    }
    Ok(Fp8BlockSize {
        rows: rows as usize,
        cols: cols as usize,
    })
}

// ---------------------------------------------------------------------------
// §4a  The float codecs, hand-rolled so this crate take no new dependency
// ---------------------------------------------------------------------------

/// IEEE-754 binary16 -> f32.
///
/// **Upstream:** `float16.Frombits(u).Float32()` from `github.com/x448/float16`.
/// Rust's `f32::from_bits` plus the standard widening -- exact in every case, no
/// rounding to worry about, since every f16 is representable as an f32.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;

    let out = if exp == 0 {
        if mant == 0 {
            sign << 31 // signed zero
        } else {
            // Subnormal f16 -> normal f32. A subnormal's value is
            // `mant * 2^-24`; shift until bit 10 is set, drop that implicit
            // bit, and the exponent field become `113 - shifts`:
            //   mant * 2^-24 = (m / 1024) * 2^(-14 - shifts)
            //   => biased exponent = 127 + (-14 - shifts) = 113 - shifts.
            let mut shifts = 0u32;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                shifts += 1;
            }
            m &= 0x3ff;
            (sign << 31) | ((113 - shifts) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        // Inf / NaN: exponent all ones in both formats.
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };

    f32::from_bits(out)
}

/// f32 -> IEEE-754 binary16, round-to-nearest-even.
///
/// **Upstream:** `float16.Fromfloat32(f).Bits()` from `github.com/x448/float16`.
/// The bit-trick form here is Fabian Giesen's `float_to_half_fast3_rtne` (public
/// domain, <https://gist.github.com/rygorous/2156668>), which is also what the
/// x448 package compute -- same rounding mode, same overflow-to-Inf behaviour.
///
/// The three branches:
///
/// * `>= (127+16) << 23` -- magnitude too big for f16. NaN in, NaN out
///   (`0x7e00`); anything else saturate to Inf (`0x7c00`).
/// * `< 113 << 23` -- the f16 result is subnormal. Adding the "denorm magic"
///   constant force the FPU to do the subnormal rounding for us, then the same
///   constant is subtracted back out of the bit pattern.
/// * otherwise -- normal: rebias the exponent, then add `0xfff + mantissa_odd`
///   before truncating 13 bits, which is exactly round-half-to-even. The carry
///   out of the mantissa propagate into the exponent by itself, which is why the
///   exponent rebias happen *before* the rounding add.
pub fn f32_to_f16(value: f32) -> u16 {
    /// `255 << 23` -- everything above this exponent is Inf or NaN.
    const F32_INFTY: u32 = 255 << 23;
    /// `(127 + 16) << 23` -- the smallest f32 magnitude that overflow f16.
    const F16_MAX: u32 = (127 + 16) << 23;
    /// `((127 - 15) + (23 - 10) + 1) << 23` -- the subnormal rounding magic.
    const DENORM_MAGIC: u32 = ((127 - 15) + (23 - 10) + 1) << 23;

    let mut u = value.to_bits();
    let sign = u & 0x8000_0000;
    u ^= sign;

    let o: u32 = if u >= F16_MAX {
        if u > F32_INFTY { 0x7e00 } else { 0x7c00 }
    } else if u < (113 << 23) {
        let f = f32::from_bits(u) + f32::from_bits(DENORM_MAGIC);
        f.to_bits().wrapping_sub(DENORM_MAGIC)
    } else {
        let mant_odd = (u >> 13) & 1;
        // Rebias: f32 bias 127 -> f16 bias 15, i.e. subtract 112 from the
        // exponent field.
        u = u.wrapping_sub(112u32 << 23);
        u = u.wrapping_add(0xfff);
        u = u.wrapping_add(mant_odd);
        u >> 13
    };

    ((sign >> 16) | o) as u16
}

/// bfloat16 -> f32: put the 16 bits in the **high** half and zero the rest.
///
/// **Upstream:** `bfloat16.DecodeFloat32` from `github.com/d4l3k/go-bfloat16`.
///
/// bf16 is literally the top half of an f32 -- same 8-bit exponent, mantissa
/// truncated from 23 bits to 7. That is why it is the format of choice for
/// training: same dynamic range as f32, conversion is a shift.
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// f32 -> bfloat16 by **truncation**, not rounding.
///
/// **Upstream:** `bfloat16.EncodeFloat32`, which do
/// `PutUint16(b, uint16(math.Float32bits(f) >> 16))` -- a plain shift, so the
/// low 16 bits are dropped, never rounded.
///
/// **This is a real difference from `f32_to_f16`**, which round-to-nearest-even.
/// We keep the truncation because the values must match ollama's output
/// bit-for-bit; switching to RNE here would produce a GGUF that differ from
/// ollama's for the same checkpoint, which make every cross-check useless.
pub fn f32_to_bf16(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

/// One byte of `float8_e4m3fn` -> f32.
///
/// **Upstream:** `decodeFloat8E4M3FN` (`reader_safetensors.go:596`).
///
/// The layout is `[sign:1][exponent:4][mantissa:3]` with an exponent bias of 7,
/// and the `fn` suffix mean **finite-only**: there is no Inf encoding, and the
/// single NaN is `exponent == 0xf && mantissa == 0x7`.
///
/// * `exp == 0` -> subnormal: `sign * (mant/8) * 2^-6`.
/// * `exp == 0xf && mant == 0x7` -> NaN.
/// * otherwise -> `sign * (1 + mant/8) * 2^(exp-7)`.
///
/// What would make this wrong: treating `exp == 0xf` as Inf the way IEEE binary
/// formats do. In `e4m3fn` that exponent is a normal value except for the one
/// NaN pattern, so an Inf reading would throw away the largest magnitudes the
/// format can hold (up to 448).
pub fn decode_float8_e4m3fn(v: u8) -> f32 {
    let sign = if v & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((v >> 3) & 0x0f) as i32;
    let mant = (v & 0x07) as i32;

    if exp == 0 {
        if mant == 0 {
            return 0.0 * sign;
        }
        return sign * ((mant as f64 / 8.0) * (2.0f64).powi(-6)) as f32;
    }
    if exp == 0x0f && mant == 0x07 {
        return f32::NAN;
    }

    sign * ((1.0 + mant as f64 / 8.0) * (2.0f64).powi(exp - 7)) as f32
}

// ---------------------------------------------------------------------------
// §4b  Getting a tensor's bytes out, decoded and re-encoded
// ---------------------------------------------------------------------------

/// Decode a raw safetensors payload into `f32`.
///
/// **Upstream:** the `switch st.dtype` inside `safetensor.WriteTo`
/// (`reader_safetensors.go:215`). Everything is little-endian; safetensors do
/// not have a big-endian variant.
fn decode_dtype(dtype: &str, bytes: &[u8]) -> Result<Vec<f32>, ConvertError> {
    match dtype {
        "F32" => Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        "F16" => Ok(bytes
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        "BF16" => Ok(bytes
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        other => Err(ConvertError::UnknownDataType(other.to_string())),
    }
}

/// Encode `f32` values into the ggml type the tensor is going out as.
///
/// **Upstream:** the `switch st.Kind()` at the end of `safetensor.WriteTo`
/// (`reader_safetensors.go:250`). Only F32, F16 and BF16 can come out of a
/// conversion -- quantisation is a separate pass upstream (`quantizeLayer` in
/// `server/create.go`), not part of `convert`.
fn encode_kind(kind: TensorType, values: &[f32]) -> Result<Vec<u8>, ConvertError> {
    match kind {
        TensorType::F32 => {
            let mut out = Vec::with_capacity(values.len() * 4);
            for v in values {
                out.extend_from_slice(&v.to_le_bytes());
            }
            Ok(out)
        }
        TensorType::F16 => {
            let mut out = Vec::with_capacity(values.len() * 2);
            for v in values {
                out.extend_from_slice(&f32_to_f16(*v).to_le_bytes());
            }
            Ok(out)
        }
        TensorType::BF16 => {
            let mut out = Vec::with_capacity(values.len() * 2);
            for v in values {
                out.extend_from_slice(&f32_to_bf16(*v).to_le_bytes());
            }
            Ok(out)
        }
        other => Err(ConvertError::UnknownDataType(format!(
            "unknown storage type: {}",
            other.0
        ))),
    }
}

impl SourceTensor {
    /// Produce this tensor's payload, decoded, repacked and re-encoded.
    ///
    /// **Upstream:** `safetensor.WriteTo` (`reader_safetensors.go:180`).
    ///
    /// The **fast path** matter for speed on a multi-GB checkpoint: when there
    /// is no repacker and the input dtype already equal the output kind (F32 ->
    /// F32, F16 -> F16), or the dtype is `U8`, the bytes are copied straight
    /// through with no float round-trip.
    ///
    /// Note what the fast path deliberately leave out: **BF16 -> BF16 still go
    /// the long way**, decoding to f32 and re-encoding. That is upstream's
    /// behaviour and it is lossless (bf16 -> f32 is exact, and f32 -> bf16 by
    /// truncation give the original bits back), so it cost time, not accuracy.
    pub fn materialise(&self, files: &dyn ModelFiles) -> Result<Vec<u8>, ConvertError> {
        let kind = self.kind()?;
        let raw = files.read_range(&self.file, self.offset, self.size)?;

        let fast_path = self.repacker.is_none()
            && ((self.dtype == "F32" && kind == TensorType::F32)
                || (self.dtype == "F16" && kind == TensorType::F16)
                || self.dtype == "U8");
        if fast_path {
            return Ok(raw);
        }

        let mut values = if self.dtype == "F8_E4M3" {
            self.decode_fp8_e4m3(files, &raw)?
        } else {
            decode_dtype(&self.dtype, &raw)?
        };

        if let Some(repacker) = &self.repacker {
            values = repacker(&self.name, values, &self.shape)?;
        }

        encode_kind(kind, &values)
    }

    /// Dequantise a block-scaled fp8 tensor.
    ///
    /// **Upstream:** `safetensor.decodeFP8E4M3` (`reader_safetensors.go:509`).
    ///
    /// The layout: the weight is a 2-D `rows x cols` matrix of `e4m3fn` bytes.
    /// The scale companion is a smaller 2-D matrix, one entry per
    /// `block.rows x block.cols` tile, so its shape must be exactly
    /// `ceil(rows/block.rows) x ceil(cols/block.cols)`. Every weight is
    /// multiplied by the scale of the tile it fall in:
    ///
    /// ```text
    /// value[r][c] = e4m3(data[r][c]) * scale[r / block.rows][c / block.cols]
    /// ```
    ///
    /// Every one of the shape checks below is load-bearing -- a mismatched scale
    /// grid would still "work" arithmetically and silently scale whole rows of
    /// weights by the wrong factor, which no downstream test would catch.
    fn decode_fp8_e4m3(
        &self,
        files: &dyn ModelFiles,
        data: &[u8],
    ) -> Result<Vec<f32>, ConvertError> {
        let scale_meta = self.scale.as_ref().ok_or_else(|| {
            ConvertError::Fp8(format!("missing fp8 scale companion for tensor {:?}", self.name))
        })?;
        let block = self.fp8_block.ok_or_else(|| {
            ConvertError::Fp8(format!(
                "missing fp8 block size metadata for tensor {:?}",
                self.name
            ))
        })?;

        if self.shape.len() != 2 {
            return Err(ConvertError::Shape {
                name: self.name.clone(),
                reason: format!("expected 2D fp8 tensor, got shape {:?}", self.shape),
            });
        }
        let rows = self.shape[0] as usize;
        let cols = self.shape[1] as usize;
        if rows * cols != data.len() {
            return Err(ConvertError::Shape {
                name: self.name.clone(),
                reason: format!("shape {:?} does not match {} bytes", self.shape, data.len()),
            });
        }

        // The scale always live in the **same shard** as the weight it belong
        // to -- safetensors offsets are file-local, so a cross-file scale could
        // not be addressed at all.
        let scale_raw = files.read_range(&self.file, scale_meta.offset, scale_meta.size)?;
        let scale = decode_dtype(&scale_meta.dtype, &scale_raw).map_err(|_| {
            ConvertError::Fp8(format!(
                "unsupported fp8 scale dtype {:?} for tensor {:?}",
                scale_meta.dtype, scale_meta.name
            ))
        })?;

        if scale_meta.shape.len() != 2 {
            return Err(ConvertError::Shape {
                name: scale_meta.name.clone(),
                reason: format!("expected 2D fp8 scale tensor, got shape {:?}", scale_meta.shape),
            });
        }
        let scale_rows = scale_meta.shape[0] as usize;
        let scale_cols = scale_meta.shape[1] as usize;
        let expected_rows = rows.div_ceil(block.rows);
        let expected_cols = cols.div_ceil(block.cols);
        if scale_rows != expected_rows || scale_cols != expected_cols {
            return Err(ConvertError::Fp8(format!(
                "unexpected fp8 scale shape {:?} for tensor {:?} shape {:?}; want [{expected_rows} {expected_cols}]",
                scale_meta.shape, self.name, self.shape
            )));
        }
        if scale.len() != scale_rows * scale_cols {
            return Err(ConvertError::Fp8(format!(
                "fp8 scale tensor {:?} shape {:?} does not match decoded length {}",
                scale_meta.name,
                scale_meta.shape,
                scale.len()
            )));
        }

        let mut out = vec![0.0f32; data.len()];
        for r in 0..rows {
            let scale_row = r / block.rows;
            let row_offset = r * cols;
            for c in 0..cols {
                out[row_offset + c] =
                    decode_float8_e4m3fn(data[row_offset + c]) * scale[scale_row * scale_cols + c / block.cols];
            }
        }
        Ok(out)
    }
}

// ===========================================================================
// §5  tokenizer.go + tokenizer_spm.go -- vocabulary, merges, special tokens
// ===========================================================================

/// GGUF token types, written into `tokenizer.ggml.token_type`.
///
/// **Upstream:** the `const (_ int32 = iota; tokenTypeNormal; ...)` block
/// (`tokenizer.go:18`). Numbering start at **1**, not 0 -- the blank `_`
/// consume 0 -- and it match SentencePiece's own `ModelProto.SentencePiece.Type`
/// enum one for one, which is exactly why the SPM path can cast straight across.
pub mod token_type {
    /// An ordinary piece of text.
    pub const NORMAL: i32 = 1;
    /// `<unk>`.
    pub const UNKNOWN: i32 = 2;
    /// A control token -- `<s>`, `</s>`, `<start_of_turn>`.
    pub const CONTROL: i32 = 3;
    /// Added by the user / by `added_tokens`.
    pub const USER_DEFINED: i32 = 4;
    /// Present in the vocabulary but never emitted.
    pub const UNUSED: i32 = 5;
    /// A raw byte, used when `byte_fallback` is on.
    pub const BYTE: i32 = 6;
}

/// The vocabulary as GGUF want it: three parallel arrays plus a model name.
///
/// **Upstream:** `type Vocabulary` (`tokenizer.go:216`).
///
/// `model` is either `"gpt2"` (BPE, from `tokenizer.json`) or `"llama"`
/// (SentencePiece, from `tokenizer.model`), and it decide which tokeniser the
/// runtime instantiate. The three arrays are index-aligned and **must stay the
/// same length** -- token id is the index.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Vocabulary {
    /// `tokenizer.ggml.model`: `"gpt2"` or `"llama"`.
    pub model: String,
    /// `tokenizer.ggml.tokens`.
    pub tokens: Vec<String>,
    /// `tokenizer.ggml.scores`. For BPE this is **the token id as a float**, not
    /// a real score -- see [`parse_vocabulary_from_tokenizer`].
    pub scores: Vec<f32>,
    /// `tokenizer.ggml.token_type`, values from [`token_type`].
    pub types: Vec<i32>,
}

/// One special token the model care about (`bos`, `eos`, `unk`, ...).
///
/// **Upstream:** `type SpecialVocabulary` (`tokenizer.go:283`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpecialVocabulary {
    /// `"bos"`, `"eos"`, `"unk"`, `"sep"`, `"pad"`, `"cls"`, `"mask"`.
    pub kind: String,
    /// Token id, resolved through `tokenizer.json`'s `added_tokens`.
    pub id: i64,
    /// The literal token text, e.g. `"<|eot_id|>"`.
    pub content: String,
    /// Value of `add_<kind>_token` from `tokenizer_config.json`.
    pub add_token: bool,
    /// Whether `add_<kind>_token` was **explicitly present**.
    ///
    /// Missing and explicit-`false` are NOT the same thing in GGUF: some
    /// tokenisers default the flag to true, so writing an explicit `false` where
    /// the config said nothing would change behaviour. Upstream carry the same
    /// flag for the same reason (`tokenizer.go:290`).
    pub add_token_set: bool,
    /// Extra ids from `generation_config.json`, e.g. a model with several EOS
    /// tokens. Written as `tokenizer.ggml.<key>_token_ids`.
    pub ids: Vec<i32>,
}

impl SpecialVocabulary {
    /// The name this token type use **inside GGUF keys**, which is not always
    /// the name it use in `tokenizer_config.json`.
    ///
    /// **Upstream:** `SpecialVocabulary.Key` (`tokenizer.go:300`).
    ///
    /// Three renames, and they are conventions you cannot guess:
    ///
    /// * `unk` -> `unknown`
    /// * `sep` -> **`seperator`** -- misspelled. It is misspelled *in the GGUF
    ///   spec*, so we misspell it too; "fixing" it would write a key no runtime
    ///   read. Upstream carry a `//nolint:misspell` on that exact line.
    /// * `pad` -> `padding`
    ///
    /// `bos`, `eos`, `cls` and `mask` pass through unchanged.
    ///
    /// Upstream `panic` on anything else; we return an error instead
    /// (**deliberate divergence** -- same reasoning as everywhere else in this
    /// module, a library do not kill the process over its input).
    pub fn key(&self) -> Result<&'static str, ConvertError> {
        match self.kind.as_str() {
            "bos" => Ok("bos"),
            "eos" => Ok("eos"),
            "cls" => Ok("cls"),
            "mask" => Ok("mask"),
            "unk" => Ok("unknown"),
            "sep" => Ok("seperator"),
            "pad" => Ok("padding"),
            other => Err(ConvertError::Tokenizer(format!(
                "unknown special vocabulary type {other:?}"
            ))),
        }
    }
}

/// Everything the tokenizer block of a GGUF need.
///
/// **Upstream:** `type Tokenizer` (`tokenizer.go:27`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Tokenizer {
    /// The vocabulary arrays.
    pub vocabulary: Vocabulary,
    /// Special tokens, in the order the converter asked for them.
    pub special_vocabulary: Vec<SpecialVocabulary>,
    /// BPE merge rules, each `"a b"`. Empty for SentencePiece.
    pub merges: Vec<String>,
    /// `tokenizer.ggml.pre` -- which pre-tokenizer the runtime should use.
    /// Identified by hashing the regexes; see [`parse_tokenizer`].
    pub pre: String,
    /// The Jinja chat template, or empty.
    pub template: String,
}

// --- tokenizer.json ---------------------------------------------------------

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct TokenizerJson {
    added_tokens: Vec<AddedToken>,
    model: TokenizerModel,
    pre_tokenizer: PreTokenizerWrapper,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct TokenizerModel {
    vocab: BTreeMap<String, i64>,
    merges: serde_json::Value,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct PreTokenizerWrapper {
    pretokenizers: Vec<PreTokenizer>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct PreTokenizer {
    #[serde(rename = "type")]
    kind: String,
    pattern: PreTokenizerPattern,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct PreTokenizerPattern {
    /// Capital `R` -- that is genuinely the key HuggingFace emit.
    #[serde(rename = "Regex")]
    regex: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct AddedToken {
    id: i64,
    content: String,
    special: bool,
}

/// Build a BPE vocabulary out of `tokenizer.json`.
///
/// **Upstream:** `parseVocabularyFromTokenizer` (`tokenizer.go:223`).
///
/// Two things worth knowing:
///
/// * **`scores` hold the token id, not a score.** BPE has no per-token score, so
///   upstream write `float32(token.ID)` into the array. It is a placeholder that
///   keep the three arrays the same length. Do not "fix" it to zeros -- some
///   readers round-trip it.
/// * **`added_tokens` overwrite by id**, so an added token replace whatever the
///   base vocab had at that index, and get `USER_DEFINED` unless it is also
///   flagged `special`, in which case `CONTROL` win.
pub fn parse_vocabulary_from_tokenizer(bytes: &[u8]) -> Result<Vocabulary, ConvertError> {
    let t: TokenizerJson = parse_config(bytes)?;

    /// Which of the two sources a token came from, since that decide its type.
    struct Entry {
        content: String,
        special: bool,
        user_defined: bool,
    }

    let mut tokens: BTreeMap<i64, Entry> = BTreeMap::new();
    for (content, id) in &t.model.vocab {
        tokens.insert(
            *id,
            Entry {
                content: content.clone(),
                special: false,
                user_defined: false,
            },
        );
    }
    for added in &t.added_tokens {
        tokens.insert(
            added.id,
            Entry {
                content: added.content.clone(),
                special: added.special,
                user_defined: true,
            },
        );
    }

    let mut v = Vocabulary {
        model: "gpt2".to_string(),
        ..Default::default()
    };
    // `BTreeMap` iterate in key order, which is upstream's
    // `slices.Sorted(maps.Keys(tokens))` -- token id order, so index == id.
    for (id, entry) in &tokens {
        v.tokens.push(entry.content.clone());
        v.scores.push(*id as f32);
        v.types.push(if entry.special {
            token_type::CONTROL
        } else if entry.user_defined {
            token_type::USER_DEFINED
        } else {
            token_type::NORMAL
        });
    }

    Ok(v)
}

/// Pick the vocabulary source.
///
/// **Upstream:** `parseVocabulary` (`tokenizer.go:255`).
///
/// **`tokenizer.model` win over `tokenizer.json`.** That order is upstream's and
/// it is not obvious -- a checkpoint that ship both (gemma do) get the
/// SentencePiece vocabulary, with `tokenizer.json` only consulted afterwards for
/// merges, added tokens and the pre-tokenizer digest.
pub fn parse_vocabulary(files: &dyn ModelFiles) -> Result<Vocabulary, ConvertError> {
    if let Some(bytes) = files.read("tokenizer.model")? {
        let ast = parse_additional_special_tokens(files)?;
        let added = files.read("added_tokens.json")?;
        return parse_sentence_piece(&bytes, &ast, added.as_deref());
    }
    if let Some(bytes) = files.read("tokenizer.json")? {
        return parse_vocabulary_from_tokenizer(&bytes);
    }
    Err(ConvertError::Tokenizer("unknown tokenizer format".to_string()))
}

/// Read the whole tokenizer out of a checkpoint.
///
/// **Upstream:** `parseTokenizer` (`tokenizer.go:35`).
///
/// Four files, each optional except the vocabulary:
///
/// 1. `tokenizer.model` **or** `tokenizer.json` -- the vocabulary.
/// 2. `tokenizer.json` -- added tokens, merges, and the pre-tokenizer digest.
/// 3. `tokenizer_config.json` -- chat template and the `<kind>_token` /
///    `add_<kind>_token` settings.
/// 4. `generation_config.json` -- extra `<kind>_token_id` **lists**.
///
/// ### The pre-tokenizer digest, which is the clever bit
///
/// GGUF need to name the pre-tokenizer (`tokenizer.ggml.pre`) so the runtime can
/// reproduce the exact splitting regex, but `tokenizer.json` do not name it --
/// it only carry the regexes. So upstream **hash the regexes**: SHA-256 over the
/// `pattern.Regex` of every `Split` pre-tokenizer, concatenated in order, then
/// look the digest up in a table. Those digests are literal fingerprints of
/// known tokenisers and are copied verbatim from `tokenizer.go:97`.
///
/// `e3b0c442...` is the SHA-256 of the empty string -- it mean "no Split
/// pre-tokenizer at all", which is legitimate and leave `pre` at `"default"`.
/// An unrecognised digest also fall back to `"default"`; upstream log a warning,
/// we do not log (this crate take no logging dependency), which is a
/// **deliberate divergence** worth knowing when debugging a new model.
pub fn parse_tokenizer(
    files: &dyn ModelFiles,
    special_token_types: &[&str],
) -> Result<Tokenizer, ConvertError> {
    let vocabulary = parse_vocabulary(files)?;
    let mut t = Tokenizer {
        vocabulary,
        pre: "default".to_string(),
        ..Default::default()
    };

    let mut added_tokens: BTreeMap<String, AddedToken> = BTreeMap::new();

    if let Some(bytes) = files.read("tokenizer.json")? {
        let tt: TokenizerJson = parse_config(&bytes)?;

        for token in &tt.added_tokens {
            added_tokens.insert(token.content.clone(), token.clone());
        }

        // Merges arrive in one of two shapes and both are in the wild:
        //   older: ["a b", "c d"]           (already space-joined)
        //   newer: [["a","b"], ["c","d"]]   (pairs)
        // Upstream try `[]string` first, then `[][]string`, joining with a
        // single space. An empty/absent `merges` is fine and mean "no BPE".
        match &tt.model.merges {
            serde_json::Value::Null => {}
            value => {
                if let Ok(list) = serde_json::from_value::<Vec<String>>(value.clone()) {
                    t.merges = list;
                } else if let Ok(pairs) = serde_json::from_value::<Vec<Vec<String>>>(value.clone()) {
                    t.merges = pairs.into_iter().map(|p| p.join(" ")).collect();
                } else {
                    return Err(ConvertError::Tokenizer(
                        "could not parse tokenizer merges. expected []string or [][]string"
                            .to_string(),
                    ));
                }
            }
        }

        let mut hasher = Sha256::new();
        for pt in &tt.pre_tokenizer.pretokenizers {
            if pt.kind == "Split" && !pt.pattern.regex.is_empty() {
                hasher.update(pt.pattern.regex.as_bytes());
            }
        }
        let digest = hex_lower(&hasher.finalize());
        t.pre = pretokenizer_name(&digest).to_string();
    }

    if let Some(bytes) = files.read("tokenizer_config.json")? {
        let p: BTreeMap<String, serde_json::Value> = parse_config(&bytes)?;

        // `chat_template` is either a string, or a list of named templates of
        // which the one called "default" win. **Upstream:** `tokenizer.go:126`.
        if let Some(template) = p.get("chat_template") {
            if let Some(s) = template.as_str() {
                t.template = s.to_string();
            } else if let Some(list) = template.as_array() {
                for entry in list {
                    if entry.get("name").and_then(|n| n.as_str()) == Some("default")
                        && let Some(s) = entry.get("template").and_then(|s| s.as_str())
                    {
                        t.template = s.to_string();
                        break;
                    }
                }
            } else {
                return Err(ConvertError::Tokenizer("invalid chat_template".to_string()));
            }
        }

        for st in special_token_types {
            let mut sv = SpecialVocabulary {
                kind: (*st).to_string(),
                ..Default::default()
            };

            if let Some(value) = p.get(&format!("add_{st}_token")) {
                sv.add_token = value.as_bool().ok_or_else(|| {
                    ConvertError::Tokenizer(format!("add_{st}_token is not a bool"))
                })?;
                sv.add_token_set = true;
            }

            // `<kind>_token` is either the literal string, or an object with a
            // `content` field. Anything else is skipped, not an error --
            // upstream `continue` on both failures.
            if let Some(value) = p.get(&format!("{st}_token")) {
                if let Some(s) = value.as_str() {
                    sv.content = s.to_string();
                } else if let Some(s) = value.get("content").and_then(|c| c.as_str()) {
                    sv.content = s.to_string();
                } else {
                    continue;
                }
            }

            // Only keep the entry when the token actually exist in
            // `added_tokens`, because that is the only place its **id** come
            // from. A `<kind>_token` naming something not in `added_tokens` is
            // dropped entirely -- upstream do the same.
            if let Some(token) = added_tokens.get(&sv.content) {
                sv.id = token.id;
                t.special_vocabulary.push(sv);
            }
        }
    }

    if let Some(bytes) = files.read("generation_config.json")? {
        let p: BTreeMap<String, serde_json::Value> = parse_config(&bytes)?;
        for st in special_token_types {
            let Some(value) = p.get(&format!("{st}_token_id")) else {
                continue;
            };
            // Only a **list** is interesting. A scalar mean "the single id we
            // already got from tokenizer_config.json", so it is skipped.
            let Ok(ids) = serde_json::from_value::<Vec<i32>>(value.clone()) else {
                continue;
            };
            if let Some(sv) = t.special_vocabulary.iter_mut().find(|sv| sv.kind == *st) {
                sv.ids = ids;
            }
        }
    }

    Ok(t)
}

/// Map a pre-tokenizer regex digest onto the name GGUF use.
///
/// **Upstream:** the `switch digest` in `parseTokenizer` (`tokenizer.go:97`).
/// These hex strings are fingerprints of specific published tokenisers -- do not
/// invent new ones; they must be computed from a real `tokenizer.json`.
fn pretokenizer_name(digest: &str) -> &'static str {
    match digest {
        "d98f9631be1e9607a9848c26c1f9eac1aa9fc21ac6ba82a2fc0741af9780a48f" => "llama-bpe",
        "03df5c5863ad70781dcfdef491ead25140f895fe8010964be0daefe27be32b02" => "deepseek-llm",
        "21cde974d587f0d54dc8d56b183cc1e6239600172035c68fbd6d4b9f8da0576e" => "deepseek-coder",
        "1ff7f41064896984db5d1bb6ff64fa4bc29007d08c1b439e505b7392777a319e" => "qwen2",
        "00431aed57e696b747435f734d1e3b9b1bfd931a121fb5cac7129e97c181e9ba" => "qwen35",
        "b92c0824a58e1d8dc3221cf3e12c433c3a86f57e46d57229993489f0798e7702" => "laguna",
        // SHA-256 of the empty string: no Split pre-tokenizer at all.
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" => "default",
        _ => "default",
    }
}

/// Lower-case hex, because `sha2` hand back raw bytes and this crate got no
/// `hex` dependency.
fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

// --- tokenizer_spm.go -------------------------------------------------------

/// One entry of `special_tokens_map.json`'s `additional_special_tokens`.
///
/// **Upstream:** `type specialToken` (`tokenizer_spm.go:110`). Only `content` is
/// read; the rest of the fields exist so the object form parse at all.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct AdditionalSpecialToken {
    /// The token text.
    pub content: String,
}

/// Read `special_tokens_map.json`'s `additional_special_tokens`.
///
/// **Upstream:** `parseAdditionalSpecialTokens` (`tokenizer_spm.go:118`).
///
/// The list is either `["<tok>", ...]` or `[{"content": "<tok>", ...}, ...]`.
/// Absent file -> empty list, which is not an error.
pub fn parse_additional_special_tokens(
    files: &dyn ModelFiles,
) -> Result<Vec<AdditionalSpecialToken>, ConvertError> {
    let Some(bytes) = files.read("special_tokens_map.json")? else {
        return Ok(Vec::new());
    };

    #[derive(Debug, Default, Deserialize)]
    #[serde(default)]
    struct Map {
        additional_special_tokens: serde_json::Value,
    }

    let m: Map = parse_config(&bytes)?;
    let Some(items) = m.additional_special_tokens.as_array() else {
        return Ok(Vec::new());
    };

    let mut out = Vec::new();
    for item in items {
        if let Some(s) = item.as_str() {
            out.push(AdditionalSpecialToken {
                content: s.to_string(),
            });
        } else if item.is_object() {
            out.push(serde_json::from_value(item.clone())?);
        }
    }
    Ok(out)
}

/// Gemma's broken configs: these pieces are typed `NORMAL` in `tokenizer.model`
/// but are genuinely control tokens.
///
/// **Upstream:** the inline `slices.Contains([]string{...})` in
/// `parseSentencePiece` (`tokenizer_spm.go:52`), whose own comment call it a
/// "temporary fix to handle gemma3 broken configs". Kept verbatim, including the
/// fact that it apply to **every** SPM model, not only gemma -- narrowing it
/// would diverge from ollama's output.
pub const SPM_FORCED_CONTROL_PIECES: &[&str] = &[
    "<end_of_turn>",
    "<start_of_turn>",
    "<start_function_declaration>",
    "<end_function_declaration>",
    "<start_function_call>",
    "<end_function_call>",
    "<start_function_response>",
    "<end_function_response>",
    "<escape>",
];

/// Build a SentencePiece vocabulary from `tokenizer.model`.
///
/// **Upstream:** `parseSentencePiece` (`tokenizer_spm.go:19`).
///
/// The type mapping, which is where the subtlety live:
///
/// * `UNKNOWN`, `CONTROL`, `UNUSED`, `BYTE` pass straight through -- the SPM
///   enum and GGUF's `token_type` use the same numbers.
/// * **everything else become `NORMAL`**, including SPM's own `USER_DEFINED`.
///   That is upstream's `default` arm and it is deliberate.
/// * ...*unless* the piece is in [`SPM_FORCED_CONTROL_PIECES`], or appear in
///   `special_tokens_map.json`'s `additional_special_tokens`, in which case it
///   become `CONTROL`.
///
/// Then `added_tokens.json` is appended. Its entries must extend the vocabulary
/// **contiguously**: an id below the current length is only tolerated when the
/// text already match exactly (a duplicate, skipped), and any gap is an error.
/// That strictness is the point -- token id is an array index, so a hole would
/// silently shift every later token.
pub fn parse_sentence_piece(
    model_bytes: &[u8],
    additional: &[AdditionalSpecialToken],
    added_tokens_json: Option<&[u8]>,
) -> Result<Vocabulary, ConvertError> {
    let pieces = decode_sentencepiece_model(model_bytes)?;

    let mut v = Vocabulary {
        model: "llama".to_string(),
        ..Default::default()
    };

    for piece in pieces {
        v.tokens.push(piece.piece.clone());
        v.scores.push(piece.score);

        let t = piece.kind;
        if t == token_type::UNKNOWN
            || t == token_type::CONTROL
            || t == token_type::UNUSED
            || t == token_type::BYTE
        {
            v.types.push(t);
            continue;
        }

        let forced = SPM_FORCED_CONTROL_PIECES.contains(&piece.piece.as_str())
            || additional.iter().any(|a| a.content == piece.piece);
        v.types
            .push(if forced { token_type::CONTROL } else { token_type::NORMAL });
    }

    let Some(bytes) = added_tokens_json else {
        return Ok(v);
    };

    let atm: BTreeMap<String, i64> = parse_config(bytes)?;
    let mut entries: Vec<(i64, String)> = atm.into_iter().map(|(c, id)| (id, c)).collect();
    entries.sort_by_key(|(id, _)| *id);

    for (id, content) in entries {
        let pos = usize::try_from(id).unwrap_or(usize::MAX);
        if pos < v.tokens.len() {
            if v.tokens[pos] == content {
                // Duplicate -- upstream warn and carry on.
                continue;
            }
            return Err(ConvertError::Tokenizer(format!(
                "token mismatch: {content} != {} at pos [{id}]",
                v.tokens[pos]
            )));
        }
        if pos != v.tokens.len() {
            return Err(ConvertError::Tokenizer(format!(
                "invalid token id: [{id}] as pos [{}]",
                v.tokens.len()
            )));
        }

        v.tokens.push(content);
        // -1000.0 is upstream's literal (`tokenizer_spm.go:104`): far below any
        // real SentencePiece log-probability, so an added token never win a
        // merge on score.
        v.scores.push(-1000.0);
        v.types.push(token_type::USER_DEFINED);
    }

    Ok(v)
}

/// One `ModelProto.SentencePiece`.
#[derive(Debug, Clone, Default)]
struct SentencePiece {
    piece: String,
    score: f32,
    kind: i32,
}

/// Decode `tokenizer.model` -- a SentencePiece `ModelProto` in protobuf wire
/// format.
///
/// **Upstream:** `proto.Unmarshal(bts, &spm)` against the generated
/// `convert/sentencepiece/sentencepiece_model.pb.go`.
///
/// **DELIBERATE DIVERGENCE:** upstream link the full `google.golang.org/protobuf`
/// runtime. This crate take no new dependency, so what is here is a **minimal
/// wire-format reader** that understand exactly the fields
/// `sentencepiece_model.proto` define for this message and skip everything else.
///
/// ### The protobuf wire format, since this is the only place it live
///
/// A message is a flat sequence of fields. Each field start with a varint
/// **key** = `(field_number << 3) | wire_type`:
///
/// | wire type | meaning | how to skip |
/// |---|---|---|
/// | 0 | varint | read a varint |
/// | 1 | 64-bit | skip 8 bytes |
/// | 2 | length-delimited (string, bytes, nested message) | read varint length, skip that many |
/// | 5 | 32-bit | skip 4 bytes |
///
/// A varint is base-128 little-endian: 7 bits per byte, high bit set means "more
/// bytes follow".
///
/// The fields we read (`sentencepiece_model.proto:295`):
///
/// * `ModelProto.pieces` = field **1**, wire type 2 (repeated nested message)
/// * `SentencePiece.piece` = field **1**, wire type 2 (string)
/// * `SentencePiece.score` = field **2**, wire type 5 (`float`, little-endian)
/// * `SentencePiece.type` = field **3**, wire type 0 (enum, `default = NORMAL`)
///
/// Unknown fields are skipped, which is exactly what a protobuf reader must do
/// and is why this survive `trainer_spec`, `normalizer_spec` and friends being
/// present in the same file.
///
/// What would make this wrong: assuming the fields arrive in order, or assuming
/// `type` is always present. Neither is guaranteed -- an absent `type` mean
/// `NORMAL`, which is why [`SentencePiece::default`] start there.
fn decode_sentencepiece_model(bytes: &[u8]) -> Result<Vec<SentencePiece>, ConvertError> {
    let mut r = ProtoReader::new(bytes);
    let mut pieces = Vec::new();

    while !r.is_empty() {
        let (field, wire) = r.key()?;
        if field == 1 && wire == 2 {
            let body = r.length_delimited()?;
            pieces.push(decode_sentencepiece(body)?);
        } else {
            r.skip(wire)?;
        }
    }

    Ok(pieces)
}

/// **Upstream:** the `SentencePiece` message. See
/// [`decode_sentencepiece_model`] for the wire format and the field numbers.
fn decode_sentencepiece(bytes: &[u8]) -> Result<SentencePiece, ConvertError> {
    let mut r = ProtoReader::new(bytes);
    // `type` default to NORMAL when the field is absent -- `[default = NORMAL]`
    // in the .proto.
    let mut sp = SentencePiece {
        kind: token_type::NORMAL,
        ..Default::default()
    };

    while !r.is_empty() {
        let (field, wire) = r.key()?;
        match (field, wire) {
            (1, 2) => {
                let body = r.length_delimited()?;
                sp.piece = String::from_utf8(body.to_vec()).map_err(|e| {
                    ConvertError::Protobuf(format!("piece is not valid UTF-8: {e}"))
                })?;
            }
            (2, 5) => sp.score = f32::from_bits(r.fixed32()?),
            (3, 0) => sp.kind = r.varint()? as i32,
            _ => r.skip(wire)?,
        }
    }

    Ok(sp)
}

/// A cursor over protobuf wire-format bytes. See [`decode_sentencepiece_model`]
/// for the format itself.
struct ProtoReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> ProtoReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn is_empty(&self) -> bool {
        self.pos >= self.buf.len()
    }

    /// Base-128 little-endian varint. Capped at 10 bytes, which is the most a
    /// 64-bit value can take.
    fn varint(&mut self) -> Result<u64, ConvertError> {
        let mut result: u64 = 0;
        let mut shift = 0u32;
        for _ in 0..10 {
            let byte = *self
                .buf
                .get(self.pos)
                .ok_or_else(|| ConvertError::Protobuf("truncated varint".to_string()))?;
            self.pos += 1;
            result |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(result);
            }
            shift += 7;
        }
        Err(ConvertError::Protobuf("varint longer than 10 bytes".to_string()))
    }

    /// `(field_number, wire_type)`.
    fn key(&mut self) -> Result<(u64, u64), ConvertError> {
        let key = self.varint()?;
        Ok((key >> 3, key & 0x07))
    }

    fn length_delimited(&mut self) -> Result<&'a [u8], ConvertError> {
        let len = usize::try_from(self.varint()?)
            .map_err(|_| ConvertError::Protobuf("length does not fit in usize".to_string()))?;
        let end = self
            .pos
            .checked_add(len)
            .filter(|e| *e <= self.buf.len())
            .ok_or_else(|| ConvertError::Protobuf("truncated length-delimited field".to_string()))?;
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn fixed32(&mut self) -> Result<u32, ConvertError> {
        let end = self
            .pos
            .checked_add(4)
            .filter(|e| *e <= self.buf.len())
            .ok_or_else(|| ConvertError::Protobuf("truncated 32-bit field".to_string()))?;
        let mut b = [0u8; 4];
        b.copy_from_slice(&self.buf[self.pos..end]);
        self.pos = end;
        Ok(u32::from_le_bytes(b))
    }

    fn skip(&mut self, wire: u64) -> Result<(), ConvertError> {
        match wire {
            0 => {
                self.varint()?;
            }
            1 => {
                self.pos = self
                    .pos
                    .checked_add(8)
                    .filter(|e| *e <= self.buf.len())
                    .ok_or_else(|| ConvertError::Protobuf("truncated 64-bit field".to_string()))?;
            }
            2 => {
                self.length_delimited()?;
            }
            5 => {
                self.fixed32()?;
            }
            other => {
                return Err(ConvertError::Protobuf(format!(
                    "unsupported wire type {other}"
                )));
            }
        }
        Ok(())
    }
}

// ===========================================================================
// §6  convert.go -- the entry point, model detection, the converter contract
// ===========================================================================

/// The bit of `config.json` that every architecture share.
///
/// **Upstream:** `type ModelParameters` (`convert.go:20`).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct ModelParameters {
    /// `architectures` -- the dispatch key. `architectures[0]` decide
    /// everything; the rest are ignored, same as upstream.
    pub architectures: Vec<String>,
    /// `vocab_size` at the top level.
    pub vocab_size: u32,
    /// `model_type`. Carried because upstream carry it; nothing read it.
    pub model_type: String,
    /// `text_config` -- multimodal checkpoints hide the text model's numbers in
    /// here instead of at the top level.
    pub text_config: TextModelParameters,
}

/// **Upstream:** the anonymous `TextModel` struct inside `ModelParameters`
/// (`convert.go:27`).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct TextModelParameters {
    /// `text_config.vocab_size`.
    pub vocab_size: u32,
    /// `text_config.hidden_size`.
    pub hidden_size: u32,
    /// `text_config.model_type`.
    pub model_type: String,
}

/// The special token types every architecture ask for unless it override.
///
/// **Upstream:** `ModelParameters.specialTokenTypes` (`convert.go:169`).
pub const DEFAULT_SPECIAL_TOKEN_TYPES: &[&str] =
    &["bos", "eos", "unk", "sep", "pad", "cls", "mask"];

/// Build a `[]string` GGUF array value.
fn kv_strings(values: Vec<String>) -> KvValue {
    KvValue::Array(KvArray::new(KvArrayValues::String(values)))
}

/// Build a `[]float32` GGUF array value.
fn kv_f32s(values: Vec<f32>) -> KvValue {
    KvValue::Array(KvArray::new(KvArrayValues::F32(values)))
}

/// Build a `[]int32` GGUF array value.
fn kv_i32s(values: Vec<i32>) -> KvValue {
    KvValue::Array(KvArray::new(KvArrayValues::I32(values)))
}

/// Build a `[]bool` GGUF array value.
fn kv_bools(values: Vec<bool>) -> KvValue {
    KvValue::Array(KvArray::new(KvArrayValues::Bool(values)))
}

/// Go's `cmp.Or` for `u32`: the first non-zero, else zero.
///
/// Used everywhere upstream to fold the two or three names a config might use
/// for the same number (`n_layers` / `num_hidden_layers` / `n_layer`).
fn or_u32(values: &[u32]) -> u32 {
    values.iter().copied().find(|v| *v != 0).unwrap_or(0)
}

/// Go's `cmp.Or` for `f32`.
fn or_f32(values: &[f32]) -> f32 {
    values.iter().copied().find(|v| *v != 0.0).unwrap_or(0.0)
}

impl ModelParameters {
    /// The tokenizer half of the metadata, which every architecture start from.
    ///
    /// **Upstream:** `ModelParameters.KV` (`convert.go:126`).
    ///
    /// The fixed pair at the top:
    ///
    /// * `general.file_type = 1` -- ggml's `F16` file type. Conversion always
    ///   emit F16/BF16/F32; quantisation is a later pass.
    /// * `general.quantization_version = 2` -- the GGUF quantisation-scheme
    ///   version, a constant since GGUF v2.
    ///
    /// Then the tokenizer block:
    ///
    /// * `tokenizer.ggml.pre` -- pre-tokenizer name (see [`parse_tokenizer`]).
    /// * `tokenizer.ggml.model` -- `"gpt2"` or `"llama"`.
    /// * `tokenizer.ggml.tokens` / `.scores` / `.token_type` -- the three
    ///   index-aligned vocabulary arrays.
    /// * `tokenizer.ggml.merges` -- **only when non-empty**. Writing an empty
    ///   merges array would make a SentencePiece model look like a BPE one.
    /// * `tokenizer.chat_template` -- only when non-empty.
    ///
    /// And per special token, using [`SpecialVocabulary::key`] for the name:
    ///
    /// * `tokenizer.ggml.add_<key>_token` -- **only if the config said so
    ///   explicitly**; see [`SpecialVocabulary::add_token_set`].
    /// * `tokenizer.ggml.<key>_token_id` -- always, as `u32`.
    /// * `tokenizer.ggml.<key>_token_ids` -- only when
    ///   `generation_config.json` gave a list.
    pub fn kv(&self, t: &Tokenizer) -> Result<Kv, ConvertError> {
        let mut kv = Kv::new();
        kv.insert("general.file_type", 1u32);
        kv.insert("general.quantization_version", 2u32);
        kv.insert("tokenizer.ggml.pre", t.pre.clone());
        kv.insert("tokenizer.ggml.model", t.vocabulary.model.clone());
        kv.insert(
            "tokenizer.ggml.tokens",
            kv_strings(t.vocabulary.tokens.clone()),
        );
        kv.insert(
            "tokenizer.ggml.scores",
            kv_f32s(t.vocabulary.scores.clone()),
        );
        kv.insert(
            "tokenizer.ggml.token_type",
            kv_i32s(t.vocabulary.types.clone()),
        );

        if !t.merges.is_empty() {
            kv.insert("tokenizer.ggml.merges", kv_strings(t.merges.clone()));
        }

        if !t.template.is_empty() {
            kv.insert("tokenizer.chat_template", t.template.clone());
        }

        for sv in &t.special_vocabulary {
            let key = sv.key()?;
            if sv.add_token_set {
                kv.insert(format!("tokenizer.ggml.add_{key}_token"), sv.add_token);
            }
            kv.insert(
                format!("tokenizer.ggml.{key}_token_id"),
                u32::try_from(sv.id).unwrap_or(0),
            );
            if !sv.ids.is_empty() {
                kv.insert(
                    format!("tokenizer.ggml.{key}_token_ids"),
                    kv_i32s(sv.ids.clone()),
                );
            }
        }

        Ok(kv)
    }
}

/// What every architecture converter must provide.
///
/// **Upstream:** the `ModelConverter` interface (`convert.go:182`), plus the two
/// optional interfaces `tokenizerAdjuster` and `tokenizerAwareTensorConverter`
/// folded in as defaulted methods -- Rust got no "does this value implement that
/// other interface too" check, so an optional Go interface become a method with
/// a default.
pub trait ModelConverter {
    /// Map the config onto GGUF metadata.
    ///
    /// **Upstream:** `KV(*Tokenizer) KV`.
    fn kv(&self, t: &Tokenizer) -> Result<Kv, ConvertError>;

    /// Map input tensors onto output tensors, doing any per-architecture
    /// reshaping or repacking on the way.
    ///
    /// **Upstream:** `Tensors([]Tensor) []*ggml.Tensor`.
    fn tensors(&self, ts: Vec<SourceTensor>) -> Result<Vec<OutTensor>, ConvertError>;

    /// The tensor-name rewrite rules, **in priority order** -- see [`Replacer`].
    ///
    /// **Upstream:** `Replacements() []string`, a flat `from, to, from, to`
    /// list. Pairs here instead, which make an odd-length list impossible.
    fn replacements(&self) -> Vec<(&'static str, &'static str)>;

    /// Which special tokens to pull out of `tokenizer_config.json`.
    ///
    /// **Upstream:** `specialTokenTypes()`.
    fn special_token_types(&self) -> &'static [&'static str] {
        DEFAULT_SPECIAL_TOKEN_TYPES
    }

    /// Optional tweak to the tokenizer after it is parsed.
    ///
    /// **Upstream:** the `tokenizerAdjuster` interface (`convert.go:212`).
    fn adjust_tokenizer(&self, _t: &mut Tokenizer) {}

    /// Tensor mapping that need to see the tokenizer -- gemma3 use it to trim
    /// `token_embd.weight` down to the real vocabulary size.
    ///
    /// **Upstream:** the `tokenizerAwareTensorConverter` interface
    /// (`convert.go:216`). Default just forward to [`ModelConverter::tensors`],
    /// which is what a Go type that does not implement it get.
    fn tensors_with_tokenizer(
        &self,
        ts: Vec<SourceTensor>,
        _t: &Tokenizer,
    ) -> Result<Vec<OutTensor>, ConvertError> {
        self.tensors(ts)
    }
}

/// Read `config.json`, pick the converter, read the tokenizer.
///
/// **Upstream:** `LoadModelMetadata` (`convert.go:269`).
///
/// ### The vocabulary-size reconciliation, which is easy to miss
///
/// `config.json` state a `vocab_size` (or `text_config.vocab_size`), and the
/// tokenizer file state its own token count. They disagree more often than you
/// would think, and the three cases are handled differently:
///
/// * **config says 0** -- no opinion; the tokenizer's count win.
/// * **config > tokenizer** -- the embedding matrix is bigger than the
///   vocabulary, so the vocabulary is **padded** with `[PAD0]`, `[PAD1]`, ...
///   each scored `-1` and typed `USER_DEFINED`. Without this the embedding rows
///   and the token list would be different lengths and the runtime would index
///   past the end.
/// * **config < tokenizer** -- the tokenizer win and the config's number is
///   ignored. (Upstream also write the corrected number back into its local
///   `ModelParameters`, but that copy is never read afterwards, so there is
///   nothing to port.)
pub fn load_model_metadata(
    files: &dyn ModelFiles,
) -> Result<(Box<dyn ModelConverter>, Tokenizer), ConvertError> {
    let bytes = files
        .read("config.json")?
        .ok_or_else(|| ConvertError::MissingFile("config.json".to_string()))?;

    let p: ModelParameters = parse_config(&bytes)?;
    let arch = p
        .architectures
        .first()
        .cloned()
        .ok_or(ConvertError::UnknownArchitecture)?;

    let conv = converter_for(&arch, &bytes)?;

    let mut t = parse_tokenizer(files, conv.special_token_types())?;
    conv.adjust_tokenizer(&mut t);

    let vocab_size = or_u32(&[p.vocab_size, p.text_config.vocab_size]) as usize;
    if vocab_size > t.vocabulary.tokens.len() {
        for i in 0..(vocab_size - t.vocabulary.tokens.len()) {
            t.vocabulary.tokens.push(format!("[PAD{i}]"));
            t.vocabulary.scores.push(-1.0);
            t.vocabulary.types.push(token_type::USER_DEFINED);
        }
    }

    Ok((conv, t))
}

/// Map `architectures[0]` onto a converter.
///
/// **Upstream:** the big `switch p.Architectures[0]` (`convert.go:288`).
///
/// Only the ten ported architectures are listed. Everything else -- including
/// names upstream *do* support -- come back as
/// [`ConvertError::UnsupportedArchitecture`], which is honest: a wrong-but-close
/// converter would produce a GGUF that load and misbehave.
fn converter_for(arch: &str, config: &[u8]) -> Result<Box<dyn ModelConverter>, ConvertError> {
    // Each converter is the same `config.json` deserialised into its own shape,
    // exactly like upstream's `json.Unmarshal(bts, conv)` after the switch.
    Ok(match arch {
        "LlamaForCausalLM" => Box::new(parse_config::<LlamaModel>(config)?),
        "MixtralForCausalLM" => Box::new(parse_config::<MixtralModel>(config)?),
        "Qwen2ForCausalLM" => Box::new(parse_config::<Qwen2Model>(config)?),
        "Qwen3ForCausalLM" | "Qwen3MoeForCausalLM" => {
            Box::new(parse_config::<Qwen3Model>(config)?)
        }
        "GemmaForCausalLM" => Box::new(parse_config::<GemmaModel>(config)?),
        "Gemma2ForCausalLM" => Box::new(parse_config::<Gemma2Model>(config)?),
        "Gemma3ForCausalLM" | "Gemma3ForConditionalGeneration" => {
            let mut m = parse_config::<Gemma3Model>(config)?;
            // Upstream set this at construction: `&gemma3Model{Architecture:
            // p.Architectures[0]}` (`convert.go:302`). The two names take
            // genuinely different metadata paths -- see `Gemma3Model::kv`.
            m.architecture = arch.to_string();
            Box::new(m)
        }
        "Phi3ForCausalLM" => Box::new(parse_config::<Phi3Model>(config)?),
        "CohereForCausalLM" => Box::new(parse_config::<CommandRModel>(config)?),
        "Mistral3ForConditionalGeneration" => Box::new(parse_config::<Mistral3Model>(config)?),
        other => return Err(ConvertError::UnsupportedArchitecture(other.to_string())),
    })
}

/// Convert a whole checkpoint into a GGUF.
///
/// **Upstream:** `ConvertModel` (`convert.go:397`).
///
/// The pipeline:
///
/// 1. [`load_model_metadata`] -- config + converter + tokenizer.
/// 2. [`parse_tensors`] with the converter's [`Replacer`] -- every tensor is
///    renamed as it is discovered, not afterwards.
/// 3. [`ensure_unique_tensor_names`].
/// 4. [`ModelConverter::tensors_with_tokenizer`].
/// 5. [`write_file`].
///
/// **Not ported:** the multimodal split. Upstream, a `MultimodalConverter` with
/// projector weights write **two** files -- a text GGUF and a projector GGUF
/// (`convert.go:424`). That need a second [`GgufWriter`] and none of the ten
/// ported architectures implement the interface, so the branch is left out
/// rather than half-built. Gemma3's vision path therefore convert as one file.
pub fn convert_model(
    files: &dyn ModelFiles,
    writer: &mut dyn GgufWriter,
) -> Result<(), ConvertError> {
    let (conv, t) = load_model_metadata(files)?;
    let replacer = Replacer::new(conv.replacements());
    let ts = parse_tensors(files, &replacer)?;
    ensure_unique_tensor_names(&ts)?;
    let tensors = conv.tensors_with_tokenizer(ts, &t)?;
    let kv = conv.kv(&t)?;
    write_file(files, writer, kv, tensors)
}

/// Push the metadata and the tensors through a [`GgufWriter`].
///
/// **Upstream:** `writeFile` (`convert.go:454`).
///
/// Two jobs, and the second one is the one people get wrong:
///
/// * fold in [`source_tensor_kv`]'s fp8 provenance keys;
/// * **reverse every shape.** safetensors state a shape slowest-varying first
///   (`[out_features, in_features]`); ggml's `ne` is fastest-varying first. The
///   *data* is untouched -- it is row-major in both readings -- so reversing the
///   shape is the whole conversion. Get this backwards and every matmul in the
///   model transpose itself.
///
/// `general.architecture` is written **first**, before anything else, because a
/// writer cannot apply [`qualify_key`] until it know the architecture.
pub fn write_file(
    files: &dyn ModelFiles,
    writer: &mut dyn GgufWriter,
    mut kv: Kv,
    ts: Vec<OutTensor>,
) -> Result<(), ConvertError> {
    if let Some((quantization, names)) = source_tensor_kv(&ts) {
        kv.insert("source_quantization", quantization);
        kv.insert("source_fp8_tensors", kv_strings(names));
    }

    let arch = kv.string("general.architecture", "").to_string();
    if arch.is_empty() {
        return Err(ConvertError::UnknownArchitecture);
    }
    writer.write_kv("general.architecture", KvValue::String(arch))?;

    for key in kv.keys().map(str::to_string).collect::<Vec<_>>() {
        if key == "general.architecture" {
            continue;
        }
        if let Some(value) = kv.value(&key) {
            writer.write_kv(&key, value.clone())?;
        }
    }

    for t in &ts {
        let mut shape = t.shape.clone();
        shape.reverse();

        let data = match &t.source {
            TensorSource::Input(st) => st.materialise(files)?,
            TensorSource::Literal(values) => encode_kind(t.kind, values)?,
            TensorSource::Merge(group) => {
                let mut out = Vec::new();
                for member in group {
                    out.extend_from_slice(&member.materialise(files)?);
                }
                out
            }
        };

        writer.write_tensor(&t.name, t.kind, &shape, &data)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// §6a  The repack primitives the architecture converters share
// ---------------------------------------------------------------------------

/// The rope permute: HuggingFace's interleaved-half layout -> ggml's.
///
/// **Upstream:** `llamaModel.repack` (`convert_llama.go:181`) and
/// `mistral3Model.repack` (`convert_mistral.go:176`), which are the same
/// function. Also identical to llama.cpp's `permute()` in
/// `convert_hf_to_gguf.py` (MIT), which is the clearer statement of it:
///
/// ```text
/// w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:])
///  .swapaxes(1, 2)
///  .reshape(w.shape)
/// ```
///
/// ### Why it exist
///
/// Rotary embeddings pair up dimensions. HuggingFace store a head's rows as
/// `[first half | second half]` and rotate `x[i]` against `x[i + d/2]`; ggml
/// store them **interleaved** and rotate adjacent pairs. Same maths, different
/// row order, so `q_proj` and `k_proj` must be shuffled once at conversion time.
/// `v_proj` and `o_proj` are untouched -- they carry no rope.
///
/// ### Reading the Go
///
/// Upstream's version end with `n.Transpose()` then `native.SelectF32(n, 1)`
/// then concatenating the slices. That pair look like a real transpose but is
/// **not**: `Transpose()` physically transpose the `[d0, d1]` matrix, and
/// `SelectF32(_, 1)` then walk it along axis 1, which read out the *original*
/// rows in the original order. Net effect: materialise + flatten, i.e. identity
/// on the row-major byte order.
///
/// That reading is forced by what the output must be. Non-repacked tensors are
/// written as raw safetensors bytes with only the shape reversed, so a repacked
/// tensor must stay row-major too -- an actual transpose would make `attn_q`
/// disagree with `attn_v` about layout. Hence the plain permute here.
///
/// **What would make this wrong:** if that reading of gorgonia is off, every
/// llama/mistral `attn_q`/`attn_k` come out transposed and the model produce
/// fluent nonsense rather than failing loudly. A follow-up should check one real
/// Llama-3 `blk.0.attn_q.weight` against ollama's own output before this is
/// trusted in anger.
pub fn permute(name: &str, data: Vec<f32>, shape: &[u64], heads: u32) -> Result<Vec<f32>, ConvertError> {
    let bad = |reason: String| ConvertError::Shape {
        name: name.to_string(),
        reason,
    };

    if shape.is_empty() {
        return Err(bad("permute needs at least a 1-D shape".to_string()));
    }
    if heads == 0 {
        return Err(bad("permute needs a non-zero head count".to_string()));
    }

    let n0 = usize::try_from(shape[0]).unwrap_or(0);
    let rest: usize = shape[1..]
        .iter()
        .map(|d| usize::try_from(*d).unwrap_or(0))
        .product::<usize>()
        .max(1);
    let heads = heads as usize;

    if n0 == 0 || n0 % (heads * 2) != 0 {
        return Err(bad(format!(
            "cannot split leading dimension {n0} into {heads} heads of 2 halves"
        )));
    }
    let d = n0 / heads / 2;

    if data.len() != n0 * rest {
        return Err(bad(format!(
            "got {} values but shape {shape:?} want {}",
            data.len(),
            n0 * rest
        )));
    }

    let mut out = vec![0.0f32; data.len()];
    let mut w = 0usize;
    for h in 0..heads {
        for j in 0..d {
            for k in 0..2 {
                let src = ((h * 2 + k) * d + j) * rest;
                out[w..w + rest].copy_from_slice(&data[src..src + rest]);
                w += rest;
            }
        }
    }

    Ok(out)
}

/// Add 1 to every value.
///
/// **Upstream:** `gemmaModel.addOne` (`convert_gemma.go:80`).
///
/// Gemma's RMSNorm compute `x * (1 + w)` while ggml compute `x * w`, so the
/// stored weight is shifted once at conversion time and the runtime need no
/// gemma-specific norm. Applied to every `*_norm.weight` **except** the vision
/// tower's (`v.*`) -- see [`GemmaModel::tensors`].
///
/// What would make this wrong: applying it twice, or to a `v.*` norm. Both give
/// a model that run and drift.
pub fn add_one(data: Vec<f32>) -> Vec<f32> {
    data.into_iter().map(|v| v + 1.0).collect()
}

/// Transpose the last two axes of a rank-3 tensor: `[a, b, c] -> [a, c, b]`.
///
/// **Upstream:** `tensor.Transpose(t, 0, 2, 1)` as used by qwen3's expert
/// handling (`convert_qwen3.go:87` and `:105`).
///
/// Used because HuggingFace store MoE expert weights `[experts, in, out]` while
/// ggml want `[experts, out, in]`.
fn transpose_last_two(
    name: &str,
    data: &[f32],
    a: usize,
    b: usize,
    c: usize,
) -> Result<Vec<f32>, ConvertError> {
    if data.len() != a * b * c {
        return Err(ConvertError::Shape {
            name: name.to_string(),
            reason: format!("got {} values but [{a}, {b}, {c}] want {}", data.len(), a * b * c),
        });
    }
    let mut out = vec![0.0f32; data.len()];
    for i in 0..a {
        for j in 0..b {
            for k in 0..c {
                out[(i * c + k) * b + j] = data[(i * b + j) * c + k];
            }
        }
    }
    Ok(out)
}

// ===========================================================================
// §7  The architecture converters
// ===========================================================================
//
// Read this once and the rest of the section is repetition:
//
// * `replacements()` is the tensor-name map, **in priority order**. It is the
//   single most valuable thing in this file -- see [`Replacer`] for why order
//   carry meaning.
// * `kv()` is the metadata map. Every key is a GGUF convention; a wrong key is
//   silently ignored by the runtime, which then fall back to a default and
//   produce a model that work badly rather than not at all.
// * `tensors()` is usually a straight pass-through; the interesting ones repack.
//
// The name map that most architectures share, worth learning once:
//
// | HuggingFace | GGUF |
// |---|---|
// | `model.embed_tokens` | `token_embd` |
// | `model.layers` | `blk` |
// | `model.norm` | `output_norm` |
// | `lm_head` | `output` |
// | `input_layernorm` | `attn_norm` |
// | `post_attention_layernorm` | `ffn_norm` |
// | `self_attn.q_proj` / `k_proj` / `v_proj` / `o_proj` | `attn_q` / `attn_k` / `attn_v` / `attn_output` |
// | `mlp.gate_proj` / `down_proj` / `up_proj` | `ffn_gate` / `ffn_down` / `ffn_up` |
//
// So `model.layers.0.self_attn.q_proj.weight` -> `blk.0.attn_q.weight`.

/// A rope scaling `factor` that a config might state as a number **or** a list.
///
/// **DELIBERATE DIVERGENCE.** Upstream type this as `ropeFactor` (`[]float32`)
/// in qwen2 and qwen3 (`convert_qwen2.go:17`), which mean a config writing
/// `"factor": 4.0` fail to unmarshal and the whole conversion die. Real
/// HuggingFace yarn configs write a scalar. We accept both: a scalar become a
/// one-element list. Phi3's `long_factor` / `short_factor` are genuinely lists
/// and are unaffected.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum RopeFactor {
    /// `"factor": 4.0`
    Scalar(f32),
    /// `"long_factor": [1.0, 1.02, ...]`
    Vector(Vec<f32>),
}

impl Default for RopeFactor {
    fn default() -> Self {
        RopeFactor::Vector(Vec::new())
    }
}

impl RopeFactor {
    /// As a flat list.
    pub fn values(&self) -> Vec<f32> {
        match self {
            RopeFactor::Scalar(v) => vec![*v],
            RopeFactor::Vector(v) => v.clone(),
        }
    }

    /// True when nothing was stated.
    pub fn is_empty(&self) -> bool {
        match self {
            RopeFactor::Scalar(_) => false,
            RopeFactor::Vector(v) => v.is_empty(),
        }
    }
}

// ---------------------------------------------------------------------------
// convert_llama.go
// ---------------------------------------------------------------------------

/// `rope_scaling` as llama state it.
///
/// **Upstream:** the anonymous struct in `llamaModel` (`convert_llama.go:29`).
/// Note **both** `type` and `rope_type` are read: older configs use `type`,
/// newer ones `rope_type`, and they mean different things here -- `type` gate
/// the linear-scaling metadata, `rope_type` gate the llama3 factor table.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct LlamaRopeScaling {
    /// Older key. `"linear"` is the only value acted on.
    #[serde(rename = "type")]
    pub kind: String,
    /// Newer key. `"llama3"` trigger [`LlamaModel::rope_factors`].
    pub rope_type: String,
    /// Scaling factor.
    pub factor: f32,
    /// llama3 low-frequency cutoff factor.
    pub low_freq_factor: f32,
    /// llama3 high-frequency cutoff factor.
    pub high_freq_factor: f32,
    /// The context length the model was originally trained at.
    pub original_max_position_embeddings: u32,
}

/// `LlamaForCausalLM`. **Upstream:** `convert_llama.go`.
///
/// Also the base for [`MixtralModel`].
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct LlamaModel {
    /// The shared `architectures` / `vocab_size` / `text_config` block.
    #[serde(flatten)]
    pub params: ModelParameters,

    // Block count, under any of its three spellings.
    /// `n_layers`.
    pub n_layers: u32,
    /// `num_hidden_layers` -- the modern name.
    pub num_hidden_layers: u32,
    /// `n_layer`.
    pub n_layer: u32,

    /// `max_position_embeddings`.
    pub max_position_embeddings: u32,
    /// `n_ctx` -- the GPT-2-era name for the same thing.
    pub n_ctx: u32,
    /// `hidden_size`.
    pub hidden_size: u32,
    /// `n_embd` -- the GPT-2-era name.
    pub n_embd: u32,
    /// `intermediate_size`.
    pub intermediate_size: u32,
    /// `n_inner` -- the GPT-2-era name.
    pub n_inner: u32,
    /// `num_attention_heads`.
    pub num_attention_heads: u32,
    /// `n_head` -- the GPT-2-era name.
    pub n_head: u32,
    /// `num_key_value_heads` -- GQA. Absent means MHA.
    pub num_key_value_heads: u32,
    /// `rope_theta`.
    pub rope_theta: f32,
    /// `rope_scaling`.
    pub rope_scaling: LlamaRopeScaling,
    /// `rms_norm_eps`.
    pub rms_norm_eps: f32,
    /// `layer_norm_eps`.
    pub layer_norm_eps: f32,
    /// `layer_norm_epsilon`.
    pub layer_norm_epsilon: f32,
    /// `norm_epsilon`.
    pub norm_epsilon: f32,
    /// `head_dim`. When set it override `hidden_size / num_attention_heads`.
    pub head_dim: u32,

    /// Skip the rope permute. **Upstream:** the unexported `skipRepack` field
    /// (`convert_llama.go:43`), set by other converters that embed this one and
    /// have already laid their tensors out ggml-style.
    #[serde(skip)]
    pub skip_repack: bool,
}

impl LlamaModel {
    /// The llama3 rope frequency-scaling table, written as the tensor
    /// `rope_freqs.weight`.
    ///
    /// **Upstream:** `llamaModel.ropeFactors` (`convert_llama.go:136`).
    ///
    /// Only produced when `rope_scaling.rope_type == "llama3"`. This is the
    /// piecewise NTK-by-parts schedule from the Llama-3.1 release: for each rope
    /// dimension pair, work out the wavelength
    /// `lambda = 2*pi * theta^(i/dim)` and then
    ///
    /// * `lambda < original/high_freq_factor` -- high frequency, **no scaling**
    ///   (factor 1). These dimensions encode local position and must not stretch.
    /// * `lambda > original/low_freq_factor` -- low frequency, **full scaling**
    ///   (factor `factor`). These encode long-range position and stretch fully.
    /// * in between -- a linear ramp in `smooth`, interpolating between the two
    ///   in the *reciprocal* domain: `1 / ((1 - smooth)/factor + smooth)`.
    ///
    /// The defaults `factor=8`, `low=1`, `high=4`, `original=8192` are upstream's
    /// literals and match Meta's published Llama-3.1 config.
    pub fn rope_factors(&self) -> Option<Vec<f32>> {
        if self.rope_scaling.rope_type != "llama3"
            || self.hidden_size == 0
            || self.num_attention_heads == 0
            || self.rope_theta == 0.0
        {
            return None;
        }

        let dim = self.hidden_size / self.num_attention_heads;
        if dim == 0 {
            return None;
        }

        let factor = or_f32(&[self.rope_scaling.factor, 8.0]);
        let factor_low = or_f32(&[self.rope_scaling.low_freq_factor, 1.0]);
        let factor_high = or_f32(&[self.rope_scaling.high_freq_factor, 4.0]);
        let original = or_u32(&[self.rope_scaling.original_max_position_embeddings, 8192]) as f32;

        let lambda_low = original / factor_low;
        let lambda_high = original / factor_high;

        let mut factors = Vec::with_capacity(dim as usize / 2);
        let mut i = 0u32;
        while i < dim {
            let lambda = 2.0
                * std::f64::consts::PI
                * (self.rope_theta as f64).powf(f64::from(i) / f64::from(dim));

            if lambda < f64::from(lambda_high) {
                factors.push(1.0);
            } else if lambda > f64::from(lambda_low) {
                factors.push(factor);
            } else {
                let smooth = (original / lambda as f32 - factor_low) / (factor_high - factor_low);
                factors.push(1.0 / ((1.0 - smooth) / factor + smooth));
            }
            i += 2;
        }

        Some(factors)
    }

    /// Attach the rope permute to `attn_q` / `attn_k` and pass everything else
    /// through. Shared with [`MixtralModel`].
    ///
    /// **Upstream:** the loop in `llamaModel.Tensors` (`convert_llama.go:117`).
    /// The four suffixes checked are `attn_q.weight`, `attn_k.weight`,
    /// `attn_q_proj.weight`, `attn_k_proj.weight` -- the `_proj` pair catch
    /// checkpoints whose names the replacer did not fully rewrite.
    fn tensors_inner(&self, ts: Vec<SourceTensor>) -> Result<Vec<OutTensor>, ConvertError> {
        let mut out = Vec::new();

        if let Some(factors) = self.rope_factors() {
            out.push(OutTensor::literal("rope_freqs.weight", factors));
        }

        let q_heads = self.num_attention_heads;
        let k_heads = or_u32(&[self.num_key_value_heads, self.num_attention_heads]);

        for mut t in ts {
            let is_q = t.name.ends_with("attn_q.weight") || t.name.ends_with("attn_q_proj.weight");
            let is_k = t.name.ends_with("attn_k.weight") || t.name.ends_with("attn_k_proj.weight");

            if (is_q || is_k) && !self.skip_repack {
                let heads = if is_q { q_heads } else { k_heads };
                t.set_repacker(Arc::new(move |name, data, shape| {
                    permute(name, data, shape, heads)
                }));
            }

            out.push(t.passthrough()?);
        }

        Ok(out)
    }
}

impl ModelConverter for LlamaModel {
    /// **Upstream:** `llamaModel.KV` (`convert_llama.go:49`).
    ///
    /// Every key is conditional on its source being non-zero, which matter:
    /// writing `llama.context_length = 0` would make the runtime believe the
    /// model has no context at all, whereas leaving the key out let it fall back
    /// to a sane default.
    ///
    /// * `llama.vocab_size` -- always, even when zero (upstream's one
    ///   unconditional write besides the architecture).
    /// * `llama.block_count` <- `n_layers` / `num_hidden_layers` / `n_layer`.
    /// * `llama.context_length` <- `max_position_embeddings` / `n_ctx`.
    /// * `llama.embedding_length` <- `hidden_size` / `n_embd`.
    /// * `llama.feed_forward_length` <- `intermediate_size` / `n_inner`.
    /// * `llama.attention.head_count` <- `num_attention_heads` / `n_head`, and
    ///   **whenever that is set**, `llama.rope.dimension_count` =
    ///   `hidden_size / head_count`. Note it use `hidden_size` specifically, not
    ///   the `n_embd` fallback -- an `n_embd`-only config get `0` here, which is
    ///   upstream's behaviour.
    /// * `llama.attention.head_dim`, and `llama.attention.key_length` /
    ///   `value_length`, all from `head_dim`.
    /// * `llama.rope.freq_base` <- `rope_theta`.
    /// * `llama.rope.scaling.type` / `.factor` -- **only** when
    ///   `rope_scaling.type == "linear"`. yarn and llama3 are handled elsewhere
    ///   (llama3 through the `rope_freqs.weight` tensor, not metadata).
    /// * `llama.attention.head_count_kv` <- `num_key_value_heads`.
    /// * `llama.attention.layer_norm_rms_epsilon` <- `rms_norm_eps`.
    /// * `llama.attention.layer_norm_epsilon` <- `layer_norm_eps` /
    ///   `layer_norm_epsilon` / `norm_epsilon`.
    fn kv(&self, t: &Tokenizer) -> Result<Kv, ConvertError> {
        let mut kv = self.params.kv(t)?;
        kv.insert("general.architecture", "llama");
        kv.insert("llama.vocab_size", self.params.vocab_size);
        kv.insert(
            "llama.block_count",
            or_u32(&[self.n_layers, self.num_hidden_layers, self.n_layer]),
        );

        let context_length = or_u32(&[self.max_position_embeddings, self.n_ctx]);
        if context_length > 0 {
            kv.insert("llama.context_length", context_length);
        }

        let embedding_length = or_u32(&[self.hidden_size, self.n_embd]);
        if embedding_length > 0 {
            kv.insert("llama.embedding_length", embedding_length);
        }

        let feed_forward_length = or_u32(&[self.intermediate_size, self.n_inner]);
        if feed_forward_length > 0 {
            kv.insert("llama.feed_forward_length", feed_forward_length);
        }

        let head_count = or_u32(&[self.num_attention_heads, self.n_head]);
        if head_count > 0 {
            kv.insert("llama.attention.head_count", head_count);
            kv.insert("llama.rope.dimension_count", self.hidden_size / head_count);
        }

        if self.head_dim > 0 {
            kv.insert("llama.attention.head_dim", self.head_dim);
        }

        if self.rope_theta > 0.0 {
            kv.insert("llama.rope.freq_base", self.rope_theta);
        }

        if self.rope_scaling.kind == "linear" {
            kv.insert("llama.rope.scaling.type", self.rope_scaling.kind.clone());
            kv.insert("llama.rope.scaling.factor", self.rope_scaling.factor);
        }

        if self.num_key_value_heads > 0 {
            kv.insert("llama.attention.head_count_kv", self.num_key_value_heads);
        }

        if self.rms_norm_eps > 0.0 {
            kv.insert("llama.attention.layer_norm_rms_epsilon", self.rms_norm_eps);
        }

        let layer_norm_epsilon = or_f32(&[
            self.layer_norm_eps,
            self.layer_norm_epsilon,
            self.norm_epsilon,
        ]);
        if layer_norm_epsilon > 0.0 {
            kv.insert("llama.attention.layer_norm_epsilon", layer_norm_epsilon);
        }

        if self.head_dim > 0 {
            kv.insert("llama.attention.key_length", self.head_dim);
            kv.insert("llama.attention.value_length", self.head_dim);
        }

        Ok(kv)
    }

    fn tensors(&self, ts: Vec<SourceTensor>) -> Result<Vec<OutTensor>, ConvertError> {
        self.tensors_inner(ts)
    }

    /// **Upstream:** `llamaModel.Replacements` (`convert_llama.go:163`).
    /// The canonical map -- everything else in this section is a variation on it.
    fn replacements(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("lm_head", "output"),
            ("model.embed_tokens", "token_embd"),
            ("model.norm", "output_norm"),
            ("model.layers", "blk"),
            ("input_layernorm", "attn_norm"),
            ("self_attn.q_proj", "attn_q"),
            ("self_attn.k_proj", "attn_k"),
            ("self_attn.v_proj", "attn_v"),
            ("self_attn.o_proj", "attn_output"),
            ("mlp.gate_proj", "ffn_gate"),
            ("mlp.down_proj", "ffn_down"),
            ("mlp.up_proj", "ffn_up"),
            ("post_attention_layernorm", "ffn_norm"),
        ]
    }
}

// ---------------------------------------------------------------------------
// convert_mixtral.go
// ---------------------------------------------------------------------------

/// `MixtralForCausalLM` -- llama plus sparse MoE. **Upstream:**
/// `convert_mixtral.go`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct MixtralModel {
    /// Everything llama, inherited exactly.
    #[serde(flatten)]
    pub llama: LlamaModel,
    /// `num_local_experts` -- how many experts per layer.
    pub num_local_experts: u32,
    /// `num_experts_per_tok` -- how many are active per token.
    pub num_experts_per_tok: u32,
}

impl ModelConverter for MixtralModel {
    /// **Upstream:** `mixtralModel.KV` (`convert_mixtral.go:14`). llama's
    /// metadata plus `llama.expert_count` and `llama.expert_used_count` -- note
    /// the architecture stay **`llama`**, not `mixtral`; ggml treat MoE as a
    /// llama variant.
    fn kv(&self, t: &Tokenizer) -> Result<Kv, ConvertError> {
        let mut kv = self.llama.kv(t)?;
        if self.num_local_experts > 0 {
            kv.insert("llama.expert_count", self.num_local_experts);
        }
        if self.num_experts_per_tok > 0 {
            kv.insert("llama.expert_used_count", self.num_experts_per_tok);
        }
        Ok(kv)
    }

    /// **Upstream:** `mixtralModel.Tensors` (`convert_mixtral.go:28`).
    ///
    /// Mixtral ship each expert as its own tensor
    /// (`blk.0.3.w1.weight` after replacement); ggml want them **stacked** into
    /// one `blk.0.ffn_gate_exps.weight` of shape `[n_experts, ...]`. So six
    /// merge rules per layer -- `w1`/`w2`/`w3`, weight and bias:
    ///
    /// | pattern | merged into | what it is |
    /// |---|---|---|
    /// | `blk.N.*.w1.weight` | `blk.N.ffn_gate_exps.weight` | gate |
    /// | `blk.N.*.w2.weight` | `blk.N.ffn_up_exps.weight` | **up** |
    /// | `blk.N.*.w3.weight` | `blk.N.ffn_down_exps.weight` | **down** |
    ///
    /// Note `w2 -> up` and `w3 -> down`. Mistral's own naming call `w2` the
    /// down-projection, so upstream's mapping look swapped -- it is copied here
    /// exactly as ollama have it, because ollama's output is the reference. If
    /// this ever proves wrong it is wrong in ollama too, and the fix belong
    /// upstream first.
    ///
    /// Anything left unmatched fall through to llama's handling.
    fn tensors(&self, ts: Vec<SourceTensor>) -> Result<Vec<OutTensor>, ConvertError> {
        let layers = self.llama.num_hidden_layers;
        let mut merges = Vec::with_capacity(layers as usize * 6);
        for i in 0..layers {
            merges.push(Merge::new(
                format!("blk.{i}.*.w1.weight"),
                format!("blk.{i}.ffn_gate_exps.weight"),
            ));
            merges.push(Merge::new(
                format!("blk.{i}.*.w1.bias"),
                format!("blk.{i}.ffn_gate_exps.bias"),
            ));
            merges.push(Merge::new(
                format!("blk.{i}.*.w2.weight"),
                format!("blk.{i}.ffn_up_exps.weight"),
            ));
            merges.push(Merge::new(
                format!("blk.{i}.*.w2.bias"),
                format!("blk.{i}.ffn_up_exps.bias"),
            ));
            merges.push(Merge::new(
                format!("blk.{i}.*.w3.weight"),
                format!("blk.{i}.ffn_down_exps.weight"),
            ));
            merges.push(Merge::new(
                format!("blk.{i}.*.w3.bias"),
                format!("blk.{i}.ffn_down_exps.bias"),
            ));
        }

        let (mut out, rest) = merge_tensors(ts, &merges)?;
        out.extend(self.llama.tensors_inner(rest)?);
        Ok(out)
    }

    /// **Upstream:** `mixtralModel.Replacements` (`convert_mixtral.go:58`).
    ///
    /// llama's map, then three more **appended** (so they run at lower
    /// priority):
    ///
    /// * `model.layers -> blk` again (harmless, already in llama's list);
    /// * `block_sparse_moe.gate -> ffn_gate_inp` -- the router;
    /// * `block_sparse_moe.experts. -> .` -- collapse the expert path, turning
    ///   `blk.0.block_sparse_moe.experts.3.w1.weight` into `blk.0.3.w1.weight`,
    ///   which is exactly the shape the merge patterns above match.
    fn replacements(&self) -> Vec<(&'static str, &'static str)> {
        let mut v = self.llama.replacements();
        v.extend([
            ("model.layers", "blk"),
            ("block_sparse_moe.gate", "ffn_gate_inp"),
            ("block_sparse_moe.experts.", "."),
        ]);
        v
    }
}

// ---------------------------------------------------------------------------
// convert_qwen2.go
// ---------------------------------------------------------------------------

/// `rope_scaling` as qwen state it.
///
/// **Upstream:** the anonymous struct in `qwen2Model` (`convert_qwen2.go:14`).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct QwenRopeScaling {
    /// `""`, `"yarn"`, `"mrope"` or `"default"`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Scaling factor. See [`RopeFactor`] for why this accept both shapes.
    pub factor: RopeFactor,
    /// Original training context length.
    pub original_max_position_embeddings: u32,
    /// M-RoPE section split, for the vision-language variants.
    pub mrope_section: Vec<i32>,
}

/// `Qwen2ForCausalLM`. **Upstream:** `convert_qwen2.go`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Qwen2Model {
    /// Shared params.
    #[serde(flatten)]
    pub params: ModelParameters,
    /// `max_position_embeddings`.
    pub max_position_embeddings: u32,
    /// `hidden_size`.
    pub hidden_size: u32,
    /// `num_hidden_layers`.
    pub num_hidden_layers: u32,
    /// `intermediate_size`.
    pub intermediate_size: u32,
    /// `num_attention_heads`.
    pub num_attention_heads: u32,
    /// `num_key_value_heads`.
    pub num_key_value_heads: u32,
    /// `rope_theta`.
    pub rope_theta: f32,
    /// `rope_scaling`.
    pub rope_scaling: QwenRopeScaling,
    /// `rms_norm_eps`.
    pub rms_norm_eps: f32,
}

impl ModelConverter for Qwen2Model {
    /// **Upstream:** `qwen2Model.KV` (`convert_qwen2.go:25`).
    ///
    /// Unlike llama, **every key is written unconditionally** -- qwen2 configs
    /// always carry all of them, so upstream do not bother guarding.
    ///
    /// The rope-scaling switch:
    ///
    /// * `""` -- nothing written.
    /// * `"yarn"` -- `qwen2.rope.scaling.type` + `.factor`.
    /// * `"mrope"` / `"default"` -- `qwen2.rope.mrope_section` (the multimodal
    ///   rope split), **not** the scaling keys.
    /// * anything else -- upstream `panic`; we error
    ///   ([`ConvertError::UnknownRopeScaling`]).
    fn kv(&self, t: &Tokenizer) -> Result<Kv, ConvertError> {
        let mut kv = self.params.kv(t)?;
        kv.insert("general.architecture", "qwen2");
        kv.insert("qwen2.block_count", self.num_hidden_layers);
        kv.insert("qwen2.context_length", self.max_position_embeddings);
        kv.insert("qwen2.embedding_length", self.hidden_size);
        kv.insert("qwen2.feed_forward_length", self.intermediate_size);
        kv.insert("qwen2.attention.head_count", self.num_attention_heads);
        kv.insert("qwen2.attention.head_count_kv", self.num_key_value_heads);
        kv.insert("qwen2.rope.freq_base", self.rope_theta);
        kv.insert("qwen2.attention.layer_norm_rms_epsilon", self.rms_norm_eps);

        match self.rope_scaling.kind.as_str() {
            "" => {}
            "yarn" => {
                kv.insert("qwen2.rope.scaling.type", self.rope_scaling.kind.clone());
                kv.insert(
                    "qwen2.rope.scaling.factor",
                    kv_f32s(self.rope_scaling.factor.values()),
                );
            }
            "mrope" | "default" => {
                kv.insert(
                    "qwen2.rope.mrope_section",
                    kv_i32s(self.rope_scaling.mrope_section.clone()),
                );
            }
            other => return Err(ConvertError::UnknownRopeScaling(other.to_string())),
        }

        Ok(kv)
    }

    /// **Upstream:** `qwen2Model.Tensors` (`convert_qwen2.go:52`). Pure
    /// pass-through -- qwen store rope already interleaved, so no permute.
    fn tensors(&self, ts: Vec<SourceTensor>) -> Result<Vec<OutTensor>, ConvertError> {
        ts.into_iter().map(SourceTensor::passthrough).collect()
    }

    /// **Upstream:** `qwen2Model.Replacements` (`convert_qwen2.go:65`).
    /// The canonical map exactly, only the ordering differ.
    fn replacements(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("lm_head", "output"),
            ("model.embed_tokens", "token_embd"),
            ("model.layers", "blk"),
            ("input_layernorm", "attn_norm"),
            ("self_attn.k_proj", "attn_k"),
            ("self_attn.v_proj", "attn_v"),
            ("self_attn.q_proj", "attn_q"),
            ("self_attn.o_proj", "attn_output"),
            ("mlp.down_proj", "ffn_down"),
            ("mlp.gate_proj", "ffn_gate"),
            ("mlp.up_proj", "ffn_up"),
            ("post_attention_layernorm", "ffn_norm"),
            ("model.norm", "output_norm"),
        ]
    }
}

// ---------------------------------------------------------------------------
// convert_qwen3.go
// ---------------------------------------------------------------------------

/// `Qwen3ForCausalLM` / `Qwen3MoeForCausalLM`. **Upstream:** `convert_qwen3.go`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Qwen3Model {
    /// Shared params.
    #[serde(flatten)]
    pub params: ModelParameters,
    /// `max_position_embeddings`.
    pub max_position_embeddings: u32,
    /// `hidden_size`.
    pub hidden_size: u32,
    /// `num_hidden_layers`.
    pub num_hidden_layers: u32,
    /// `intermediate_size`.
    pub intermediate_size: u32,
    /// `num_attention_heads`.
    pub num_attention_heads: u32,
    /// `num_key_value_heads`.
    pub num_key_value_heads: u32,
    /// `head_dim` -- qwen3 state it explicitly rather than deriving it.
    pub head_dim: u32,
    /// `num_experts`. Non-zero switch the architecture to `qwen3moe`.
    pub num_experts: u32,
    /// `num_experts_per_tok`.
    pub num_experts_per_tok: u32,
    /// `norm_topk_prob`.
    pub norm_topk_prob: bool,
    /// `rope_theta`.
    pub rope_theta: f32,
    /// `rope_scaling`.
    pub rope_scaling: QwenRopeScaling,
    /// `rms_norm_eps`.
    pub rms_norm_eps: f32,
}

impl ModelConverter for Qwen3Model {
    /// **Upstream:** `qwen3Model.KV` (`convert_qwen3.go:34`).
    ///
    /// Two things here are unlike every other converter:
    ///
    /// 1. **The architecture is computed:** `"qwen3"`, or `"qwen3moe"` when
    ///    `num_experts > 0`. Same config struct, two GGUF architectures.
    /// 2. **The keys are written unqualified** -- `block_count`, not
    ///    `qwen3.block_count`. The writer prefix them; see [`qualify_key`].
    ///    That is what let one `kv()` serve both architecture names without
    ///    formatting a prefix into every string.
    ///
    /// MoE-only keys: `expert_count`, `expert_used_count`, `norm_top_k_prob`.
    ///
    /// `attention.key_length` and `attention.value_length` both come from
    /// `head_dim`.
    fn kv(&self, t: &Tokenizer) -> Result<Kv, ConvertError> {
        let arch = if self.num_experts > 0 { "qwen3moe" } else { "qwen3" };

        let mut kv = self.params.kv(t)?;
        kv.insert("general.architecture", arch);
        kv.insert("block_count", self.num_hidden_layers);
        kv.insert("context_length", self.max_position_embeddings);
        kv.insert("embedding_length", self.hidden_size);
        kv.insert("feed_forward_length", self.intermediate_size);
        kv.insert("attention.head_count", self.num_attention_heads);
        kv.insert("attention.head_count_kv", self.num_key_value_heads);
        kv.insert("attention.key_length", self.head_dim);
        kv.insert("attention.value_length", self.head_dim);

        if self.num_experts > 0 {
            kv.insert("expert_count", self.num_experts);
            kv.insert("expert_used_count", self.num_experts_per_tok);
            kv.insert("norm_top_k_prob", self.norm_topk_prob);
        }

        kv.insert("rope.freq_base", self.rope_theta);
        kv.insert("attention.layer_norm_rms_epsilon", self.rms_norm_eps);

        match self.rope_scaling.kind.as_str() {
            "" => {}
            "yarn" => {
                kv.insert("rope.scaling.type", self.rope_scaling.kind.clone());
                kv.insert(
                    "rope.scaling.factor",
                    kv_f32s(self.rope_scaling.factor.values()),
                );
            }
            "mrope" | "default" => {
                kv.insert(
                    "rope.mrope_section",
                    kv_i32s(self.rope_scaling.mrope_section.clone()),
                );
            }
            other => return Err(ConvertError::UnknownRopeScaling(other.to_string())),
        }

        Ok(kv)
    }

    /// **Upstream:** `qwen3Model.Tensors` (`convert_qwen3.go:76`).
    ///
    /// Dense qwen3 is a pass-through. The MoE variant need two rearrangements,
    /// both because HuggingFace store expert weights `[experts, in, out]` while
    /// ggml want `[experts, out, in]`:
    ///
    /// * **`ffn_gate_up_exps`** -- HF fuse gate and up into one tensor of shape
    ///   `[E, I, 2F]`. Split it in half along the **last** axis into
    ///   `ffn_gate_exps` and `ffn_up_exps` (the `gate_up -> gate` / `gate_up ->
    ///   up` name rewrite), then transpose each to `[E, F, I]`.
    /// * **`ffn_down_exps`** -- already one tensor of shape `[E, A, B]`, just
    ///   transposed to `[E, B, A]`.
    ///
    /// **DELIBERATE DIVERGENCE in form, not in result:** upstream do this
    /// through the generic `splitDim` iterator in `tensor.go`, which build lazy
    /// gorgonia slices. The two call sites are the only uses of it, so they are
    /// written out directly here rather than porting a general slicing engine
    /// with no second customer. The output tensors, names and shapes are the
    /// same.
    ///
    /// What would make this wrong: getting the split axis or the transpose order
    /// mixed up. Both produce correctly-shaped tensors full of wrong numbers.
    fn tensors(&self, ts: Vec<SourceTensor>) -> Result<Vec<OutTensor>, ConvertError> {
        let mut out = Vec::new();

        for t in ts {
            if t.name.contains("ffn_gate_up_exps") {
                if t.shape.len() != 3 {
                    return Err(ConvertError::Shape {
                        name: t.name.clone(),
                        reason: format!("ffn_gate_up_exps want rank 3, got {:?}", t.shape),
                    });
                }
                let e = t.shape[0] as usize;
                let i = t.shape[1] as usize;
                let two_f = t.shape[2] as usize;
                if !two_f.is_multiple_of(2) {
                    return Err(ConvertError::Shape {
                        name: t.name.clone(),
                        reason: format!("ffn_gate_up_exps last dim {two_f} is not even"),
                    });
                }
                let f = two_f / 2;

                for (half, from, to) in [(0usize, "gate_up", "gate"), (1usize, "gate_up", "up")] {
                    let mut piece = t.clone();
                    piece.name = Replacer::new([(from, to)]).replace(&t.name);
                    // Out shape: split dim 2 in half, then swap dims 1 and 2 --
                    // upstream's `t.Shape[1], t.Shape[2] = t.Shape[2], t.Shape[1]`.
                    piece.shape = vec![e as u64, f as u64, i as u64];
                    piece.set_repacker(Arc::new(move |name, data, _shape| {
                        // Slice `[E, I, 2F]` down the last axis, then transpose
                        // the trailing two axes to `[E, F, I]`.
                        let mut sliced = Vec::with_capacity(e * i * f);
                        for ei in 0..e {
                            for ii in 0..i {
                                let row = (ei * i + ii) * two_f + half * f;
                                let end = row + f;
                                if end > data.len() {
                                    return Err(ConvertError::Shape {
                                        name: name.to_string(),
                                        reason: "ffn_gate_up_exps payload shorter than its shape"
                                            .to_string(),
                                    });
                                }
                                sliced.extend_from_slice(&data[row..end]);
                            }
                        }
                        transpose_last_two(name, &sliced, e, i, f)
                    }));
                    out.push(OutTensor {
                        name: piece.name.clone(),
                        kind: piece.kind()?,
                        shape: piece.shape.clone(),
                        source: TensorSource::Input(Box::new(piece)),
                    });
                }
                continue;
            }

            if t.name.contains("ffn_down_exps") {
                if t.shape.len() != 3 {
                    return Err(ConvertError::Shape {
                        name: t.name.clone(),
                        reason: format!("ffn_down_exps want rank 3, got {:?}", t.shape),
                    });
                }
                let (e, a, b) = (
                    t.shape[0] as usize,
                    t.shape[1] as usize,
                    t.shape[2] as usize,
                );
                let mut t = t;
                let shape = vec![e as u64, b as u64, a as u64];
                t.set_repacker(Arc::new(move |name, data, _shape| {
                    transpose_last_two(name, &data, e, a, b)
                }));
                out.push(OutTensor {
                    name: t.name.clone(),
                    kind: t.kind()?,
                    shape,
                    source: TensorSource::Input(Box::new(t)),
                });
                continue;
            }

            out.push(t.passthrough()?);
        }

        Ok(out)
    }

    /// **Upstream:** `qwen3Model.Replacements` (`convert_qwen3.go:131`).
    ///
    /// The canonical map plus qwen3's own:
    ///
    /// * `self_attn.q_norm` / `k_norm` -> `attn_q_norm` / `attn_k_norm` --
    ///   qwen3 normalise Q and K, which qwen2 do not.
    /// * `mlp.gate.weight -> ffn_gate_inp.weight` -- the MoE router. Note the
    ///   `.weight` is part of the pattern, which is what stop it from also
    ///   eating `mlp.gate_proj`.
    /// * `mlp.experts.down_proj -> ffn_down_exps.weight` and
    ///   `mlp.experts.gate_up_proj -> ffn_gate_up_exps.weight` -- the stacked
    ///   expert tensors [`Qwen3Model::tensors`] then take apart.
    fn replacements(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("lm_head", "output"),
            ("model.embed_tokens", "token_embd"),
            ("model.layers", "blk"),
            ("input_layernorm", "attn_norm"),
            ("self_attn.k_proj", "attn_k"),
            ("self_attn.k_norm", "attn_k_norm"),
            ("self_attn.v_proj", "attn_v"),
            ("self_attn.q_proj", "attn_q"),
            ("self_attn.q_norm", "attn_q_norm"),
            ("self_attn.o_proj", "attn_output"),
            ("mlp.down_proj", "ffn_down"),
            ("mlp.gate_proj", "ffn_gate"),
            ("mlp.up_proj", "ffn_up"),
            ("mlp.gate.weight", "ffn_gate_inp.weight"),
            ("mlp.experts.down_proj", "ffn_down_exps.weight"),
            ("mlp.experts.gate_up_proj", "ffn_gate_up_exps.weight"),
            ("post_attention_layernorm", "ffn_norm"),
            ("model.norm", "output_norm"),
        ]
    }
}

// ---------------------------------------------------------------------------
// convert_gemma.go
// ---------------------------------------------------------------------------

/// `GemmaForCausalLM`. **Upstream:** `convert_gemma.go`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct GemmaModel {
    /// Shared params.
    #[serde(flatten)]
    pub params: ModelParameters,
    /// `max_position_embeddings`.
    pub max_position_embeddings: u32,
    /// `hidden_size`.
    pub hidden_size: u32,
    /// `num_hidden_layers`.
    pub num_hidden_layers: u32,
    /// `intermediate_size`.
    pub intermediate_size: u32,
    /// `num_attention_heads`.
    pub num_attention_heads: u32,
    /// `num_key_value_heads`.
    pub num_key_value_heads: u32,
    /// `rms_norm_eps`.
    pub rms_norm_eps: f32,
    /// `head_dim` -- gemma state it explicitly (256 for the 2B).
    pub head_dim: u32,
}

/// Attach [`add_one`] to every `*_norm.weight` that is not part of a vision
/// tower.
///
/// **Upstream:** the loop in `gemmaModel.Tensors` (`convert_gemma.go:45`) and
/// the same test inside `gemma3Model.TensorsWithTokenizer`
/// (`convert_gemma3.go:197`).
///
/// The `v.` exclusion is the load-bearing half: the vision encoder use ordinary
/// LayerNorm, so shifting its weights by 1 would corrupt the image path while
/// leaving text working -- the worst possible failure, because it look fine.
pub fn gemma_set_add_one(t: &mut SourceTensor) {
    if !t.name.starts_with("v.") && t.name.ends_with("_norm.weight") {
        t.set_repacker(Arc::new(|_name, data, _shape| Ok(add_one(data))));
    }
}

/// The four fill-in-the-middle / end-of-turn token ids every gemma variant
/// hard-code.
///
/// **Upstream:** the identical four lines in `gemmaModel.KV`
/// (`convert_gemma.go:38`) and `gemma2Model.KV` (`convert_gemma2.go:26`).
///
/// These are **literal token ids in gemma's vocabulary**, not derived from
/// anything: `eot = 107`, `middle = 68`, `prefix = 67`, `suffix = 69`. They are
/// only correct for gemma's own SentencePiece vocabulary -- if a gemma
/// derivative retrain the tokenizer, these become wrong and there is nothing in
/// the config to catch it.
pub fn gemma_fim_token_ids(kv: &mut Kv) {
    kv.insert("tokenizer.ggml.eot_token_id", 107u32);
    kv.insert("tokenizer.ggml.middle_token_id", 68u32);
    kv.insert("tokenizer.ggml.prefix_token_id", 67u32);
    kv.insert("tokenizer.ggml.suffix_token_id", 69u32);
}

impl ModelConverter for GemmaModel {
    /// **Upstream:** `gemmaModel.KV` (`convert_gemma.go:25`). Straightforward --
    /// every key unconditional, plus [`gemma_fim_token_ids`].
    fn kv(&self, t: &Tokenizer) -> Result<Kv, ConvertError> {
        let mut kv = self.params.kv(t)?;
        kv.insert("general.architecture", "gemma");
        kv.insert("gemma.context_length", self.max_position_embeddings);
        kv.insert("gemma.embedding_length", self.hidden_size);
        kv.insert("gemma.block_count", self.num_hidden_layers);
        kv.insert("gemma.feed_forward_length", self.intermediate_size);
        kv.insert("gemma.attention.head_count", self.num_attention_heads);
        kv.insert("gemma.attention.head_count_kv", self.num_key_value_heads);
        kv.insert("gemma.attention.layer_norm_rms_epsilon", self.rms_norm_eps);
        kv.insert("gemma.attention.key_length", self.head_dim);
        kv.insert("gemma.attention.value_length", self.head_dim);
        gemma_fim_token_ids(&mut kv);
        Ok(kv)
    }

    /// **Upstream:** `gemmaModel.Tensors` (`convert_gemma.go:44`).
    fn tensors(&self, ts: Vec<SourceTensor>) -> Result<Vec<OutTensor>, ConvertError> {
        ts.into_iter()
            .map(|mut t| {
                gemma_set_add_one(&mut t);
                t.passthrough()
            })
            .collect()
    }

    /// **Upstream:** `gemmaModel.Replacements` (`convert_gemma.go:62`). The
    /// canonical map **minus `lm_head -> output`** -- gemma tie the output
    /// projection to the input embedding, so there is no `lm_head` tensor.
    fn replacements(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("model.embed_tokens", "token_embd"),
            ("model.norm", "output_norm"),
            ("model.layers", "blk"),
            ("input_layernorm", "attn_norm"),
            ("self_attn.q_proj", "attn_q"),
            ("self_attn.k_proj", "attn_k"),
            ("self_attn.v_proj", "attn_v"),
            ("self_attn.o_proj", "attn_output"),
            ("mlp.gate_proj", "ffn_gate"),
            ("mlp.down_proj", "ffn_down"),
            ("mlp.up_proj", "ffn_up"),
            ("post_attention_layernorm", "ffn_norm"),
        ]
    }
}

// ---------------------------------------------------------------------------
// convert_gemma2.go
// ---------------------------------------------------------------------------

/// `Gemma2ForCausalLM`. **Upstream:** `convert_gemma2.go`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Gemma2Model {
    /// Everything gemma.
    #[serde(flatten)]
    pub gemma: GemmaModel,
    /// `sliding_window` -- gemma2 alternate local and global attention.
    pub sliding_window: u32,
    /// `attn_logit_softcapping`.
    pub attn_logit_softcapping: f32,
    /// `final_logit_softcapping`.
    pub final_logit_softcapping: f32,
}

impl ModelConverter for Gemma2Model {
    /// **Upstream:** `gemma2Model.KV` (`convert_gemma2.go:9`).
    ///
    /// Same shape as gemma but under the `gemma2.` prefix, plus three keys that
    /// are gemma2's whole architectural difference:
    ///
    /// * `gemma2.attention.sliding_window` -- the local-attention window.
    /// * `gemma2.attn_logit_softcapping` -- `tanh`-based cap on attention
    ///   logits.
    /// * `gemma2.final_logit_softcapping` -- the same cap on the output logits.
    ///
    /// Note it call `ModelParameters.KV` directly rather than gemma's, so
    /// nothing from `gemma.kv()` leak in besides these keys.
    fn kv(&self, t: &Tokenizer) -> Result<Kv, ConvertError> {
        let g = &self.gemma;
        let mut kv = g.params.kv(t)?;
        kv.insert("general.architecture", "gemma2");
        kv.insert("gemma2.context_length", g.max_position_embeddings);
        kv.insert("gemma2.embedding_length", g.hidden_size);
        kv.insert("gemma2.block_count", g.num_hidden_layers);
        kv.insert("gemma2.feed_forward_length", g.intermediate_size);
        kv.insert("gemma2.attention.head_count", g.num_attention_heads);
        kv.insert("gemma2.attention.head_count_kv", g.num_key_value_heads);
        kv.insert("gemma2.attention.layer_norm_rms_epsilon", g.rms_norm_eps);
        kv.insert("gemma2.attention.key_length", g.head_dim);
        kv.insert("gemma2.attention.value_length", g.head_dim);
        kv.insert("gemma2.attention.sliding_window", self.sliding_window);
        kv.insert("gemma2.attn_logit_softcapping", self.attn_logit_softcapping);
        kv.insert("gemma2.final_logit_softcapping", self.final_logit_softcapping);
        gemma_fim_token_ids(&mut kv);
        Ok(kv)
    }

    /// Same `+1` on the norms as gemma. **Upstream:** gemma2 inherit
    /// `gemmaModel.Tensors` through Go embedding.
    fn tensors(&self, ts: Vec<SourceTensor>) -> Result<Vec<OutTensor>, ConvertError> {
        self.gemma.tensors(ts)
    }

    /// **Upstream:** `gemma2Model.Replacements` (`convert_gemma2.go:33`).
    ///
    /// The difference from gemma matter: gemma2 got **four** norms per block,
    /// not two, so `post_attention_layernorm` no longer mean "the FFN norm":
    ///
    /// | HuggingFace | GGUF |
    /// |---|---|
    /// | `input_layernorm` | `attn_norm` |
    /// | `post_attention_layernorm` | **`post_attention_norm`** |
    /// | `pre_feedforward_layernorm` | **`ffn_norm`** |
    /// | `post_feedforward_layernorm` | `post_ffw_norm` |
    ///
    /// Carrying gemma's `post_attention_layernorm -> ffn_norm` over would put
    /// two different tensors on the same name and lose one.
    fn replacements(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("model.embed_tokens", "token_embd"),
            ("model.norm", "output_norm"),
            ("model.layers", "blk"),
            ("input_layernorm", "attn_norm"),
            ("self_attn.q_proj", "attn_q"),
            ("self_attn.k_proj", "attn_k"),
            ("self_attn.v_proj", "attn_v"),
            ("self_attn.o_proj", "attn_output"),
            ("mlp.gate_proj", "ffn_gate"),
            ("mlp.down_proj", "ffn_down"),
            ("mlp.up_proj", "ffn_up"),
            ("post_attention_layernorm", "post_attention_norm"),
            ("pre_feedforward_layernorm", "ffn_norm"),
            ("post_feedforward_layernorm", "post_ffw_norm"),
        ]
    }
}

// ---------------------------------------------------------------------------
// convert_gemma3.go
// ---------------------------------------------------------------------------

/// `text_config` as gemma3 state it.
/// **Upstream:** `gemma3Model.TextModel` (`convert_gemma3.go:15`).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Gemma3TextConfig {
    /// `head_dim`.
    pub head_dim: u32,
    /// `hidden_size`.
    pub hidden_size: u32,
    /// `num_hidden_layers`.
    pub num_hidden_layers: u32,
    /// `intermediate_size`.
    pub intermediate_size: u32,
    /// `sliding_window`.
    pub sliding_window: u32,
}

/// `vision_config` as gemma3 state it.
/// **Upstream:** `gemma3Model.VisionModel` (`convert_gemma3.go:22`).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Gemma3VisionConfig {
    /// `num_attention_heads` (16 for the SigLIP tower).
    pub num_attention_heads: u32,
    /// `layer_norm_eps` (1e-5).
    pub layer_norm_eps: f32,
    /// `num_hidden_layers` (32).
    pub num_hidden_layers: u32,
    /// `hidden_size` (1280).
    pub hidden_size: u32,
    /// `intermediate_size` (5120).
    pub intermediate_size: u32,
    /// `image_size` (560).
    pub image_size: u32,
    /// `num_channels` (3).
    pub num_channels: u32,
    /// `patch_size` (14).
    pub patch_size: u32,
}

/// gemma3's yarn block. **Upstream:** `gemma3Model.RopeScaling`
/// (`convert_gemma3.go:47`).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Gemma3RopeScaling {
    /// `rope_type`.
    #[serde(rename = "rope_type")]
    pub kind: String,
    /// `factor`.
    pub factor: f32,
    /// `original_max_position_embeddings`.
    pub original_max_position_embeddings: u32,
    /// `extrapolation_factor`.
    pub extrapolation_factor: f32,
    /// `beta_fast`.
    pub beta_fast: f32,
    /// `beta_slow`.
    pub beta_slow: f32,
}

/// `Gemma3ForCausalLM` / `Gemma3ForConditionalGeneration`. **Upstream:**
/// `convert_gemma3.go`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Gemma3Model {
    /// Shared params.
    #[serde(flatten)]
    pub params: ModelParameters,

    /// Which of the two `architectures[0]` names this came from. Set by
    /// `converter_for`, not by serde -- it is not a config field.
    #[serde(skip)]
    pub architecture: String,

    /// `num_hidden_layers` at the top level (text-only checkpoints).
    pub num_hidden_layers: u32,
    /// `hidden_size` at the top level.
    pub hidden_size: u32,
    /// `intermediate_size` at the top level.
    pub intermediate_size: u32,
    /// `text_config`.
    pub text_config: Gemma3TextConfig,
    /// `vision_config`.
    pub vision_config: Gemma3VisionConfig,
    /// `max_position_embeddings`.
    pub max_position_embeddings: u32,
    /// `num_attention_heads`.
    pub num_attention_heads: u32,
    /// `num_key_value_heads`.
    pub num_key_value_heads: u32,
    /// `rms_norm_eps`.
    pub rms_norm_eps: f32,
    /// `head_dim`.
    pub head_dim: u32,
    /// `final_logit_softcapping`.
    pub final_logit_softcapping: f32,
    /// `rope_local_base_freq` -- the theta for the *local* (sliding) layers.
    pub rope_local_base_freq: f32,
    /// `rope_theta` -- the theta for the *global* layers.
    pub rope_theta: f32,
    /// `sliding_window`.
    pub sliding_window: u32,
    /// `sliding_window_pattern` -- "every Nth layer is global".
    pub sliding_window_pattern: Option<u32>,
    /// `layer_types` -- the explicit per-layer list, newer than the pattern int.
    pub layer_types: Vec<String>,
    /// `mm_tokens_per_image`.
    pub mm_tokens_per_image: u32,
    /// `rope_scaling`.
    pub rope_scaling: Option<Gemma3RopeScaling>,
}

/// Layer counts that identify a published gemma3 size.
///
/// **Upstream:** `gemma4BLayerCount` / `gemma12BLayerCount` /
/// `gemma27BLayerCount` (`convert_gemma3.go:55`).
///
/// This is a genuine **workaround, not a design**: some released gemma3 configs
/// state the wrong `num_attention_heads` / `num_key_value_heads`, so upstream
/// identify the model by its layer count and substitute the known-good numbers.
/// A gemma3 derivative that happen to have 34, 48 or 62 layers and *different*
/// head counts would be silently mis-converted -- that is the risk upstream
/// accepted, and we match it rather than second-guess.
const GEMMA_4B_LAYER_COUNT: u32 = 34;
/// See [`GEMMA_4B_LAYER_COUNT`].
const GEMMA_12B_LAYER_COUNT: u32 = 48;
/// See [`GEMMA_4B_LAYER_COUNT`].
const GEMMA_27B_LAYER_COUNT: u32 = 62;

impl ModelConverter for Gemma3Model {
    /// **Upstream:** `gemma3Model.KV` (`convert_gemma3.go:61`).
    ///
    /// Head counts come from the layer-count table above; only an unrecognised
    /// layer count fall back to the config's own numbers.
    ///
    /// Then the two `architectures[0]` names take **different** paths:
    ///
    /// **`Gemma3ForCausalLM`** (text only) reads the top-level fields and write:
    /// `context_length`, `attention.layer_norm_rms_epsilon`,
    /// `attention.key_length` / `value_length` (both from `head_dim`),
    /// `attention.sliding_window`, `embedding_length`, `feed_forward_length`,
    /// optionally `final_logit_softcapping`, the two rope thetas, and the yarn
    /// block when `rope_scaling.rope_type == "yarn"`.
    ///
    /// * `gemma3.rope.local.freq_base` default **10000**, `gemma3.rope.freq_base`
    ///   default **1000000** -- gemma3's local layers use a short-range theta and
    ///   the global ones a long-range theta. Both literals are upstream's.
    /// * yarn defaults: `extrapolation_factor` 1.0, `beta_fast` 64.0,
    ///   `beta_slow` 1.0.
    /// * **`gemma3.attention.sliding_window_pattern`** is a **bool array, one
    ///   entry per block**, `true` = this layer is local/sliding. Two sources:
    ///   `layer_types[i] == "sliding_attention"` when the list is present, else
    ///   `(i+1) % sliding_window_pattern != 0` -- i.e. every Nth layer is global
    ///   and the rest are local.
    ///
    /// **`Gemma3ForConditionalGeneration`** (multimodal) instead read
    /// `text_config` for the text numbers, default `context_length` to 131072,
    /// default `key_length`/`value_length` to **256**, and add the whole
    /// `gemma3.vision.*` block (`num_channels` default 3,
    /// `attention.layer_norm_epsilon` default 1e-6).
    ///
    /// `gemma3.mm.tokens_per_image` is written by both paths when non-zero.
    fn kv(&self, t: &Tokenizer) -> Result<Kv, ConvertError> {
        let mut kv = self.params.kv(t)?;
        kv.insert("general.architecture", "gemma3");

        let num_blocks = or_u32(&[self.num_hidden_layers, self.text_config.num_hidden_layers]);
        kv.insert("gemma3.block_count", num_blocks);

        let (num_heads, num_kv_heads) = match num_blocks {
            GEMMA_4B_LAYER_COUNT => (8u32, 4u32),
            GEMMA_12B_LAYER_COUNT => (16, 8),
            GEMMA_27B_LAYER_COUNT => (32, 16),
            _ => (self.num_attention_heads, self.num_key_value_heads),
        };
        kv.insert("gemma3.attention.head_count", num_heads);
        kv.insert("gemma3.attention.head_count_kv", num_kv_heads);

        if self.architecture == "Gemma3ForCausalLM" {
            kv.insert("gemma3.context_length", self.max_position_embeddings);
            kv.insert("gemma3.attention.layer_norm_rms_epsilon", self.rms_norm_eps);
            kv.insert("gemma3.attention.key_length", self.head_dim);
            kv.insert("gemma3.attention.value_length", self.head_dim);
            kv.insert("gemma3.attention.sliding_window", self.sliding_window);

            if self.sliding_window_pattern.is_some() || !self.layer_types.is_empty() {
                let mut pattern = Vec::with_capacity(num_blocks as usize);
                for i in 0..num_blocks {
                    let is_local = if !self.layer_types.is_empty()
                        && (i as usize) < self.layer_types.len()
                    {
                        self.layer_types[i as usize] == "sliding_attention"
                    } else if let Some(n) = self.sliding_window_pattern.filter(|n| *n > 0) {
                        (i + 1) % n != 0
                    } else {
                        false
                    };
                    pattern.push(is_local);
                }
                kv.insert("gemma3.attention.sliding_window_pattern", kv_bools(pattern));
            }

            if self.final_logit_softcapping > 0.0 {
                kv.insert("gemma3.final_logit_softcapping", self.final_logit_softcapping);
            }
            kv.insert(
                "gemma3.rope.local.freq_base",
                or_f32(&[self.rope_local_base_freq, 10000.0]),
            );
            kv.insert(
                "gemma3.rope.freq_base",
                or_f32(&[self.rope_theta, 1000000.0]),
            );

            if let Some(rs) = self.rope_scaling.as_ref().filter(|rs| rs.kind == "yarn" && rs.factor > 0.0) {
                kv.insert("gemma3.rope.scaling.type", "yarn");
                kv.insert("gemma3.rope.scaling.factor", rs.factor);
                kv.insert(
                    "gemma3.rope.scaling.original_context_length",
                    rs.original_max_position_embeddings,
                );
                kv.insert(
                    "gemma3.rope.scaling.extrapolation_factor",
                    or_f32(&[rs.extrapolation_factor, 1.0]),
                );
                kv.insert("gemma3.rope.scaling.beta_fast", or_f32(&[rs.beta_fast, 64.0]));
                kv.insert("gemma3.rope.scaling.beta_slow", or_f32(&[rs.beta_slow, 1.0]));
            }

            kv.insert("gemma3.embedding_length", self.hidden_size);
            kv.insert("gemma3.feed_forward_length", self.intermediate_size);
        } else {
            kv.insert(
                "gemma3.context_length",
                or_u32(&[self.max_position_embeddings, 131072]),
            );
            kv.insert("gemma3.embedding_length", self.text_config.hidden_size);
            kv.insert(
                "gemma3.feed_forward_length",
                self.text_config.intermediate_size,
            );
            kv.insert(
                "gemma3.attention.sliding_window",
                self.text_config.sliding_window,
            );
            kv.insert(
                "gemma3.vision.block_count",
                self.vision_config.num_hidden_layers,
            );
            kv.insert(
                "gemma3.vision.embedding_length",
                self.vision_config.hidden_size,
            );
            kv.insert(
                "gemma3.vision.feed_forward_length",
                self.vision_config.intermediate_size,
            );
            kv.insert("gemma3.vision.image_size", self.vision_config.image_size);
            kv.insert("gemma3.vision.patch_size", self.vision_config.patch_size);
            kv.insert(
                "gemma3.vision.num_channels",
                or_u32(&[self.vision_config.num_channels, 3]),
            );
            kv.insert(
                "gemma3.vision.attention.head_count",
                self.vision_config.num_attention_heads,
            );
            kv.insert(
                "gemma3.vision.attention.layer_norm_epsilon",
                or_f32(&[self.vision_config.layer_norm_eps, 1e-6]),
            );
            kv.insert(
                "gemma3.attention.key_length",
                or_u32(&[self.text_config.head_dim, 256]),
            );
            kv.insert(
                "gemma3.attention.value_length",
                or_u32(&[self.text_config.head_dim, 256]),
            );
        }

        if self.mm_tokens_per_image > 0 {
            kv.insert("gemma3.mm.tokens_per_image", self.mm_tokens_per_image);
        }

        Ok(kv)
    }

    /// Without a tokenizer there is nothing to trim against, so this is just the
    /// gemma `+1` on the norms. The real work is in
    /// [`ModelConverter::tensors_with_tokenizer`].
    fn tensors(&self, ts: Vec<SourceTensor>) -> Result<Vec<OutTensor>, ConvertError> {
        ts.into_iter()
            .map(|mut t| {
                gemma_set_add_one(&mut t);
                t.passthrough()
            })
            .collect()
    }

    /// **Upstream:** `gemma3Model.TensorsWithTokenizer` (`convert_gemma3.go:190`).
    ///
    /// Beyond the gemma `+1`, one gemma3-specific fix: **`token_embd.weight` is
    /// trimmed down to the real vocabulary size.** Gemma3 checkpoints ship an
    /// embedding matrix padded out past the tokenizer's token count (padding to
    /// a nice multiple for TPU sharding), and carrying the padding through would
    /// make the GGUF disagree with `tokenizer.ggml.tokens` about how many tokens
    /// exist.
    ///
    /// So when `shape[0] > vocab_size`, the shape's leading dimension is cut to
    /// `vocab_size` and a repacker truncate the values to the first
    /// `vocab_size * embd_dim`. That the *first* rows are the real ones is what
    /// make a plain prefix truncation correct.
    fn tensors_with_tokenizer(
        &self,
        ts: Vec<SourceTensor>,
        t: &Tokenizer,
    ) -> Result<Vec<OutTensor>, ConvertError> {
        let vocab_size = t.vocabulary.tokens.len() as u64;
        let mut out = Vec::with_capacity(ts.len());

        for mut st in ts {
            gemma_set_add_one(&mut st);

            let mut shape = st.shape.clone();
            if vocab_size > 0
                && st.name == "token_embd.weight"
                && shape.len() >= 2
                && shape[0] > vocab_size
            {
                let embd_dim = shape[1];
                shape[0] = vocab_size;
                let want = vocab_size.saturating_mul(embd_dim) as usize;
                st.set_repacker(Arc::new(move |name, data, _shape| {
                    if data.len() < want {
                        return Err(ConvertError::Shape {
                            name: name.to_string(),
                            reason: format!(
                                "gemma3 token_embd.weight has {} values, need {want}",
                                data.len()
                            ),
                        });
                    }
                    Ok(data[..want].to_vec())
                }));
            }

            out.push(OutTensor {
                name: st.name.clone(),
                kind: st.kind()?,
                shape,
                source: TensorSource::Input(Box::new(st)),
            });
        }

        Ok(out)
    }

    /// **Upstream:** `gemma3Model.Replacements` (`convert_gemma3.go:160`).
    ///
    /// gemma2's four-norm map, plus the multimodal rules. **Order is critical
    /// here** -- the two `...embeddings` rules must precede their prefixes, else
    /// `vision_tower.vision_model` fire first and the embeddings land under the
    /// wrong name:
    ///
    /// | HuggingFace | GGUF |
    /// |---|---|
    /// | `vision_tower.vision_model.embeddings` | `v` |
    /// | `vision_tower.vision_model` | `v` |
    /// | `language_model.` | *(removed)* |
    /// | `encoder.layers` | `blk` |
    /// | `self_attn.out_proj` | `attn_output` |
    /// | `multi_modal_projector` | `mm` |
    ///
    /// `language_model. -> ""` is a **deletion**: the text tower's tensors are
    /// nested under `language_model.` in a conditional-generation checkpoint and
    /// must come out at the same names a text-only gemma3 use.
    fn replacements(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("lm_head", "output"),
            ("model.embed_tokens", "token_embd"),
            ("model.norm", "output_norm"),
            ("vision_tower.vision_model.embeddings", "v"),
            ("vision_tower.vision_model", "v"),
            ("vision_model.vision_model.embeddings", "v"),
            ("vision_model.vision_model", "v"),
            ("language_model.", ""),
            ("model.layers", "blk"),
            ("encoder.layers", "blk"),
            ("input_layernorm", "attn_norm"),
            ("self_attn.q_proj", "attn_q"),
            ("self_attn.q_norm", "attn_q_norm"),
            ("self_attn.k_proj", "attn_k"),
            ("self_attn.k_norm", "attn_k_norm"),
            ("self_attn.v_proj", "attn_v"),
            ("self_attn.o_proj", "attn_output"),
            ("self_attn.out_proj", "attn_output"),
            ("mlp.gate_proj", "ffn_gate"),
            ("mlp.down_proj", "ffn_down"),
            ("mlp.up_proj", "ffn_up"),
            ("post_attention_layernorm", "post_attention_norm"),
            ("pre_feedforward_layernorm", "ffn_norm"),
            ("post_feedforward_layernorm", "post_ffw_norm"),
            ("input_projection_weight", "input_projection.weight"),
            ("multi_modal_projector", "mm"),
        ]
    }
}

// ---------------------------------------------------------------------------
// convert_phi3.go
// ---------------------------------------------------------------------------

/// phi3's `rope_scaling`. **Upstream:** the anonymous struct in `phi3Model`
/// (`convert_phi3.go:25`).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Phi3RopeScaling {
    /// `""`, `"su"`, `"longrope"` or `"yarn"`.
    #[serde(rename = "type")]
    pub kind: String,
    /// `long_factor` -- per-rope-dimension scaling for long contexts.
    pub long_factor: RopeFactor,
    /// `short_factor` -- ditto for short contexts.
    pub short_factor: RopeFactor,
}

/// `Phi3ForCausalLM`. **Upstream:** `convert_phi3.go`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Phi3Model {
    /// Shared params.
    #[serde(flatten)]
    pub params: ModelParameters,
    /// `num_hidden_layers`.
    pub num_hidden_layers: u32,
    /// `n_layers`.
    pub n_layers: u32,
    /// `hidden_size`.
    pub hidden_size: u32,
    /// `n_embd`.
    pub n_embd: u32,
    /// `intermediate_size`.
    pub intermediate_size: u32,
    /// `num_attention_heads`.
    pub num_attention_heads: u32,
    /// `n_head`.
    pub n_head: u32,
    /// `num_key_value_heads`.
    pub num_key_value_heads: u32,
    /// `n_head_kv`.
    pub n_head_kv: u32,
    /// `rope_theta`.
    pub rope_theta: f32,
    /// `rope_scaling`.
    pub rope_scaling: Phi3RopeScaling,
    /// `rms_norm_eps`.
    pub rms_norm_eps: f32,
    /// `n_positions`.
    pub n_positions: u32,
    /// `max_position_embeddings`.
    pub max_position_embeddings: u32,
    /// `original_max_position_embeddings`.
    pub original_max_position_embeddings: u32,
    /// `sliding_window`.
    pub sliding_window: u32,
}

impl ModelConverter for Phi3Model {
    /// **Upstream:** `phi3Model.KV` (`convert_phi3.go:39`).
    ///
    /// The interesting key is `phi3.rope.scaling.attn_factor`, the attention
    /// temperature correction that go with LongRoPE. With
    /// `scale = max_position_embeddings / original_max_position_embeddings`:
    ///
    /// * `"su"` / `"longrope"` --
    ///   `max(sqrt(1 + ln(scale) / ln(original_max_position_embeddings)), 1.0)`
    /// * `"yarn"` -- `max(0.1 * ln(scale) + 1.0, 1.0)`
    ///
    /// Both formulas are from the LongRoPE and YaRN papers respectively and are
    /// copied from upstream exactly, including the `max(_, 1.0)` floor that stop
    /// a model with `scale <= 1` from getting an attenuating factor.
    ///
    /// Upstream `panic` on an unknown type; we error.
    fn kv(&self, t: &Tokenizer) -> Result<Kv, ConvertError> {
        let mut kv = self.params.kv(t)?;
        let head_count = or_u32(&[self.num_attention_heads, self.n_head]);

        kv.insert("general.architecture", "phi3");
        kv.insert("phi3.context_length", self.max_position_embeddings);
        kv.insert(
            "phi3.embedding_length",
            or_u32(&[self.hidden_size, self.n_embd]),
        );
        kv.insert("phi3.feed_forward_length", self.intermediate_size);
        kv.insert(
            "phi3.block_count",
            or_u32(&[self.num_hidden_layers, self.n_layers]),
        );
        kv.insert("phi3.attention.head_count", head_count);
        kv.insert(
            "phi3.attention.head_count_kv",
            or_u32(&[self.num_key_value_heads, self.n_head_kv]),
        );
        kv.insert("phi3.attention.layer_norm_rms_epsilon", self.rms_norm_eps);
        kv.insert(
            "phi3.rope.dimension_count",
            // Go divide straight here; a config with no head count would divide
            // by zero, which panic in Rust and give +Inf-ish nonsense in C. Zero
            // is the honest answer, and it is what an absent key would mean.
            self.hidden_size.checked_div(head_count).unwrap_or(0),
        );
        kv.insert("phi3.rope.freq_base", self.rope_theta);
        kv.insert(
            "phi3.rope.scaling.original_context_length",
            self.original_max_position_embeddings,
        );
        kv.insert("phi3.attention.sliding_window", self.sliding_window);

        let scale =
            f64::from(self.max_position_embeddings) / f64::from(self.original_max_position_embeddings);
        match self.rope_scaling.kind.as_str() {
            "" => {}
            "su" | "longrope" => {
                let v = (1.0 + scale.ln() / f64::from(self.original_max_position_embeddings).ln())
                    .sqrt()
                    .max(1.0);
                kv.insert("phi3.rope.scaling.attn_factor", v as f32);
            }
            "yarn" => {
                let v = (0.1 * scale.ln() + 1.0).max(1.0);
                kv.insert("phi3.rope.scaling.attn_factor", v as f32);
            }
            other => return Err(ConvertError::UnknownRopeScaling(other.to_string())),
        }

        Ok(kv)
    }

    /// **Upstream:** `phi3Model.Tensors` (`convert_phi3.go:74`).
    ///
    /// The two LongRoPE factor tables ride along as **tensors**, not metadata,
    /// because they are per-rope-dimension vectors. Upstream emit them exactly
    /// once, guarded by a `sync.Once` fired on the first `blk.0.` tensor; here
    /// that is a plain bool, because this loop is single-threaded.
    ///
    /// The `blk.0.` trigger is upstream's way of putting them next to the first
    /// block in the output order -- the GGUF writer sort tensors anyway, so it
    /// only decide ordering, not correctness. What **would** be wrong is
    /// emitting them for a checkpoint with no `blk.0.` tensor at all: they would
    /// be dropped, and a LongRoPE phi3 would silently lose its scaling.
    fn tensors(&self, ts: Vec<SourceTensor>) -> Result<Vec<OutTensor>, ConvertError> {
        let mut out = Vec::with_capacity(ts.len() + 2);
        let mut added_rope_factors = false;

        for t in ts {
            if t.name.starts_with("blk.0.") && !added_rope_factors {
                added_rope_factors = true;
                out.push(OutTensor::literal(
                    "rope_factors_long.weight",
                    self.rope_scaling.long_factor.values(),
                ));
                out.push(OutTensor::literal(
                    "rope_factors_short.weight",
                    self.rope_scaling.short_factor.values(),
                ));
            }
            out.push(t.passthrough()?);
        }

        Ok(out)
    }

    /// **Upstream:** `phi3Model.Replacements` (`convert_phi3.go:100`).
    ///
    /// phi3 **fuse** its projections, so two rules differ from the canonical map:
    ///
    /// * `self_attn.qkv_proj -> attn_qkv` -- Q, K and V in one tensor.
    /// * `mlp.gate_up_proj -> ffn_up` -- gate and up in one tensor, and the
    ///   fused thing is called `ffn_up`, not `ffn_gate`. There is deliberately
    ///   **no** `ffn_gate` for phi3.
    fn replacements(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("lm_head", "output"),
            ("model.embed_tokens", "token_embd"),
            ("model.norm", "output_norm"),
            ("model.layers", "blk"),
            ("input_layernorm", "attn_norm"),
            ("self_attn.qkv_proj", "attn_qkv"),
            ("self_attn.o_proj", "attn_output"),
            ("mlp.down_proj", "ffn_down"),
            ("mlp.gate_up_proj", "ffn_up"),
            ("post_attention_layernorm", "ffn_norm"),
        ]
    }
}

// ---------------------------------------------------------------------------
// convert_commandr.go
// ---------------------------------------------------------------------------

/// `CohereForCausalLM` -- Command-R. **Upstream:** `convert_commandr.go`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct CommandRModel {
    /// Shared params.
    #[serde(flatten)]
    pub params: ModelParameters,
    /// `max_position_embeddings`.
    pub max_position_embeddings: u32,
    /// `hidden_size`.
    pub hidden_size: u32,
    /// `num_hidden_layers`.
    pub num_hidden_layers: u32,
    /// `intermediate_size`.
    pub intermediate_size: u32,
    /// `num_attention_heads`.
    pub num_attention_heads: u32,
    /// `num_key_value_heads`.
    pub num_key_value_heads: u32,
    /// `layer_norm_eps` -- Command-R use plain LayerNorm, not RMSNorm.
    pub layer_norm_eps: f32,
    /// `rope_theta`.
    pub rope_theta: f32,
    /// `use_qk_norm`.
    pub use_qk_norm: bool,
    /// `model_max_length` -- Cohere's own name for the context length.
    pub model_max_length: u32,
    /// `logit_scale` -- Command-R scale its output logits.
    pub logit_scale: f32,
    /// `n_ctx`.
    pub n_ctx: u32,
}

impl ModelConverter for CommandRModel {
    /// **Upstream:** `commandrModel.KV` (`convert_commandr.go:27`).
    ///
    /// Notes worth carrying:
    ///
    /// * The architecture string is **`command-r`**, with a hyphen. It also
    ///   write `general.name = "command-r"`, which no other ported converter do.
    /// * `command-r.context_length` <- `model_max_length` / `max_position_embeddings`
    ///   / `n_ctx`, in that order -- Cohere's own key win.
    /// * `command-r.attention.layer_norm_epsilon` (not `_rms_`): Command-R use
    ///   LayerNorm.
    /// * `command-r.rope.scaling.type = "none"` is written **literally**, as a
    ///   positive statement that no scaling applies.
    /// * `command-r.max_position_embeddings` is written **as well as**
    ///   `context_length`, from the same source minus the `n_ctx` fallback.
    fn kv(&self, t: &Tokenizer) -> Result<Kv, ConvertError> {
        let mut kv = self.params.kv(t)?;
        kv.insert("general.architecture", "command-r");
        kv.insert("general.name", "command-r");
        kv.insert(
            "command-r.context_length",
            or_u32(&[
                self.model_max_length,
                self.max_position_embeddings,
                self.n_ctx,
            ]),
        );
        kv.insert("command-r.embedding_length", self.hidden_size);
        kv.insert("command-r.block_count", self.num_hidden_layers);
        kv.insert("command-r.feed_forward_length", self.intermediate_size);
        kv.insert("command-r.attention.head_count", self.num_attention_heads);
        kv.insert(
            "command-r.attention.head_count_kv",
            self.num_key_value_heads,
        );
        kv.insert(
            "command-r.attention.layer_norm_epsilon",
            self.layer_norm_eps,
        );
        kv.insert("command-r.rope.freq_base", self.rope_theta);
        kv.insert(
            "command-r.max_position_embeddings",
            or_u32(&[self.model_max_length, self.max_position_embeddings]),
        );
        kv.insert("command-r.logit_scale", self.logit_scale);
        kv.insert("command-r.rope.scaling.type", "none");
        Ok(kv)
    }

    /// **Upstream:** `commandrModel.Tensors` (`convert_commandr.go:45`).
    /// Pass-through -- Command-R store rope already interleaved.
    fn tensors(&self, ts: Vec<SourceTensor>) -> Result<Vec<OutTensor>, ConvertError> {
        ts.into_iter().map(SourceTensor::passthrough).collect()
    }

    /// **Upstream:** `commandrModel.Replacements` (`convert_commandr.go:58`).
    ///
    /// The canonical map plus `self_attn.q_norm` / `k_norm` (Command-R's
    /// optional QK-norm), and **no `lm_head`** -- Command-R tie its output
    /// projection to the embedding.
    ///
    /// The two `_norm` rules sit **first** on purpose: `self_attn.q_norm` must
    /// beat nothing here, but keeping upstream's exact order keep the two
    /// implementations bit-identical if a future rule ever overlap.
    fn replacements(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("self_attn.q_norm", "attn_q_norm"),
            ("self_attn.k_norm", "attn_k_norm"),
            ("model.layers", "blk"),
            ("input_layernorm", "attn_norm"),
            ("mlp.down_proj", "ffn_down"),
            ("mlp.gate_proj", "ffn_gate"),
            ("mlp.up_proj", "ffn_up"),
            ("self_attn.k_proj", "attn_k"),
            ("self_attn.o_proj", "attn_output"),
            ("self_attn.q_proj", "attn_q"),
            ("self_attn.v_proj", "attn_v"),
            ("model.norm", "output_norm"),
            ("model.embed_tokens", "token_embd"),
        ]
    }
}

// ---------------------------------------------------------------------------
// convert_mistral.go
// ---------------------------------------------------------------------------

/// mistral3's `text_config.rope_parameters`.
/// **Upstream:** `mistral3Model.TextModel.RopeParameters` (`convert_mistral.go:32`).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Mistral3RopeParameters {
    /// `beta_fast`.
    pub beta_fast: f32,
    /// `beta_slow`.
    pub beta_slow: f32,
    /// `factor`.
    pub factor: f32,
    /// `llama_4_scaling_beta` -- present only on the llama4-style variants.
    pub llama_4_scaling_beta: Option<f32>,
    /// `original_max_position_embeddings`.
    pub original_max_position_embeddings: u32,
    /// `rope_type`.
    pub rope_type: String,
    /// `rope_theta`.
    pub rope_theta: f32,
    /// `mscale`.
    pub mscale: Option<f32>,
    /// `mscale_all_dim`.
    pub mscale_all_dim: Option<f32>,
}

/// mistral3's `text_config`. **Upstream:** `mistral3Model.TextModel`
/// (`convert_mistral.go:18`).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Mistral3TextConfig {
    /// `num_hidden_layers`.
    pub num_hidden_layers: u32,
    /// `max_position_embeddings`.
    pub max_position_embeddings: u32,
    /// `hidden_size`.
    pub hidden_size: u32,
    /// `intermediate_size`.
    pub intermediate_size: u32,
    /// `num_attention_heads`.
    pub num_attention_heads: u32,
    /// `num_key_value_heads`.
    pub num_key_value_heads: u32,
    /// `rope_theta`.
    pub rope_theta: f32,
    /// `rms_norm_eps`.
    pub rms_norm_eps: f32,
    /// `head_dim`.
    pub head_dim: u32,
    /// `sliding_window`.
    pub sliding_window: Option<u32>,
    /// `hidden_act`.
    pub hidden_act: String,
    /// `vocab_size`.
    pub vocab_size: u32,
    /// `rope_parameters`.
    pub rope_parameters: Mistral3RopeParameters,
}

/// mistral3's `vision_config`. **Upstream:** `mistral3Model.VisionModel`
/// (`convert_mistral.go:44`).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Mistral3VisionConfig {
    /// `num_attention_heads`.
    pub num_attention_heads: u32,
    /// `num_hidden_layers`.
    pub num_hidden_layers: u32,
    /// `hidden_size`.
    pub hidden_size: u32,
    /// `intermediate_size`.
    pub intermediate_size: u32,
    /// `image_size`.
    pub image_size: u32,
    /// `num_channels`.
    pub num_channels: u32,
    /// `patch_size`.
    pub patch_size: u32,
    /// `head_dim`.
    pub head_dim: u32,
    /// `hidden_act`.
    pub hidden_act: String,
    /// `rope_theta`.
    pub rope_theta: f32,
    /// `rope_parameters`.
    pub rope_parameters: Mistral3RopeParameters,
}

/// `Mistral3ForConditionalGeneration`. **Upstream:** `convert_mistral.go`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Mistral3Model {
    /// Shared params.
    #[serde(flatten)]
    pub params: ModelParameters,
    /// `image_token_index`.
    pub image_token_index: u32,
    /// `spatial_merge_size`.
    pub spatial_merge_size: u32,
    /// `vision_feature_layer`.
    pub vision_feature_layer: i32,
    /// `text_config`.
    pub text_config: Mistral3TextConfig,
    /// `vision_config`.
    pub vision_config: Mistral3VisionConfig,
    /// `multimodal_projector_bias`.
    pub multimodal_projector_bias: bool,
    /// `projector_hidden_act`.
    pub projector_hidden_act: String,
}

impl ModelConverter for Mistral3Model {
    /// **Upstream:** `mistral3Model.KV` (`convert_mistral.go:64`).
    ///
    /// Everything come from `text_config` / `vision_config`, never the top
    /// level. Two derivations worth naming:
    ///
    /// * `mistral3.rope.dimension_count` <- `head_dim`, falling back to
    ///   `hidden_size / num_attention_heads`.
    /// * `mistral3.rope.freq_base` <- `text_config.rope_theta`, falling back to
    ///   `text_config.rope_parameters.rope_theta`. Newer configs moved the value
    ///   into the nested block; both spellings are live.
    ///
    /// Three optional keys, each written only when the config actually carry the
    /// value (they are `*float32` upstream, so absent and `0.0` are different):
    /// `rope.scaling.yarn_log_multiplier` <- `mscale_all_dim`,
    /// `rope.scaling.original_context_length`, and
    /// `attention.temperature_scale` <- `llama_4_scaling_beta`.
    ///
    /// `mistral3.vision.attention.layer_norm_epsilon` is **commented out
    /// upstream** with the note "Default value 1e-05", so it is deliberately not
    /// written here either.
    fn kv(&self, t: &Tokenizer) -> Result<Kv, ConvertError> {
        let tc = &self.text_config;
        let vc = &self.vision_config;
        let rp = &tc.rope_parameters;

        let mut kv = self.params.kv(t)?;
        kv.insert("general.architecture", "mistral3");
        kv.insert("mistral3.vocab_size", tc.vocab_size);

        kv.insert("mistral3.block_count", tc.num_hidden_layers);
        kv.insert("mistral3.context_length", tc.max_position_embeddings);
        kv.insert("mistral3.embedding_length", tc.hidden_size);
        kv.insert("mistral3.feed_forward_length", tc.intermediate_size);
        kv.insert("mistral3.attention.head_count", tc.num_attention_heads);
        kv.insert("mistral3.attention.head_count_kv", tc.num_key_value_heads);
        kv.insert("mistral3.attention.layer_norm_rms_epsilon", tc.rms_norm_eps);
        kv.insert("mistral3.attention.key_length", tc.head_dim);
        kv.insert("mistral3.attention.value_length", tc.head_dim);
        kv.insert(
            "mistral3.rope.dimension_count",
            or_u32(&[
                tc.head_dim,
                // Same divide-by-zero guard as phi3's above.
                tc.hidden_size.checked_div(tc.num_attention_heads).unwrap_or(0),
            ]),
        );
        kv.insert(
            "mistral3.rope.freq_base",
            or_f32(&[tc.rope_theta, rp.rope_theta]),
        );
        kv.insert("mistral3.rope.scaling.factor", rp.factor);
        kv.insert("mistral3.rope.scaling.type", rp.rope_type.clone());
        kv.insert("mistral3.rope.scaling.yarn_beta_fast", rp.beta_fast);
        kv.insert("mistral3.rope.scaling.yarn_beta_slow", rp.beta_slow);

        if let Some(v) = rp.mscale_all_dim {
            kv.insert("mistral3.rope.scaling.yarn_log_multiplier", v);
        }
        if rp.original_max_position_embeddings > 0 {
            kv.insert(
                "mistral3.rope.scaling.original_context_length",
                rp.original_max_position_embeddings,
            );
        }
        if let Some(v) = rp.llama_4_scaling_beta {
            kv.insert("mistral3.attention.temperature_scale", v);
        }

        kv.insert("mistral3.vision.block_count", vc.num_hidden_layers);
        kv.insert("mistral3.vision.embedding_length", vc.hidden_size);
        kv.insert("mistral3.vision.feed_forward_length", vc.intermediate_size);
        kv.insert(
            "mistral3.vision.attention.head_count",
            vc.num_attention_heads,
        );
        kv.insert("mistral3.vision.attention.key_length", vc.head_dim);
        kv.insert("mistral3.vision.image_size", vc.image_size);
        kv.insert("mistral3.vision.patch_size", vc.patch_size);
        kv.insert("mistral3.vision.num_channels", vc.num_channels);
        kv.insert(
            "mistral3.vision.rope.freq_base",
            or_f32(&[vc.rope_theta, vc.rope_parameters.rope_theta]),
        );

        kv.insert("mistral3.image_token_index", self.image_token_index);
        kv.insert("mistral3.spatial_merge_size", self.spatial_merge_size);
        kv.insert("mistral3.mm.projector_bias", self.multimodal_projector_bias);

        if !self.projector_hidden_act.is_empty() {
            kv.insert(
                "mistral3.mm.projector_hidden_act",
                self.projector_hidden_act.clone(),
            );
        }

        Ok(kv)
    }

    /// **Upstream:** `mistral3Model.Tensors` (`convert_mistral.go:120`).
    ///
    /// The same rope permute as llama, on `.attn_q.weight` / `.attn_k.weight`,
    /// **skipping anything under `v.`** -- the vision tower's attention use no
    /// rope, so permuting it would scramble it. Head counts come from
    /// `text_config`, and `attn_k` fall back to the Q head count when the config
    /// state no KV heads (MHA rather than GQA).
    ///
    /// Note the leading dot in the suffix (`.attn_q.weight`, not
    /// `attn_q.weight`): it stop the rule matching a top-level tensor literally
    /// named `attn_q.weight`.
    fn tensors(&self, ts: Vec<SourceTensor>) -> Result<Vec<OutTensor>, ConvertError> {
        let q_heads = self.text_config.num_attention_heads;
        let k_heads = or_u32(&[
            self.text_config.num_key_value_heads,
            self.text_config.num_attention_heads,
        ]);

        ts.into_iter()
            .map(|mut t| {
                if !t.name.starts_with("v.") {
                    let is_q = t.name.ends_with(".attn_q.weight");
                    let is_k = t.name.ends_with(".attn_k.weight");
                    if is_q || is_k {
                        let heads = if is_q { q_heads } else { k_heads };
                        t.set_repacker(Arc::new(move |name, data, shape| {
                            permute(name, data, shape, heads)
                        }));
                    }
                }
                t.passthrough()
            })
            .collect()
    }

    /// **Upstream:** `mistral3Model.Replacements` (`convert_mistral.go:140`).
    ///
    /// The longest map in the ported set, because mistral3 must handle **two**
    /// naming conventions at once -- HuggingFace's `self_attn.*` / `mlp.*` and
    /// Mistral's own `attention.*` / `feed_forward.*` -- plus the multimodal
    /// nesting.
    ///
    /// Order is doing real work at the top:
    ///
    /// 1. `language_model.model.norm -> output_norm` -- must beat both prefixes
    ///    below, else the final norm end up as `norm`.
    /// 2. `language_model.model. -> ""`
    /// 3. `language_model. -> ""`
    ///
    /// `ffn_norm -> ffn_norm` near the end is an **identity rule**, and it is not
    /// dead code: it stop `ffn_norm` (already produced by an earlier rule in the
    /// same pass? no -- present verbatim in some checkpoints) from being eaten
    /// by a later partial match. Upstream have it; removing it would be a
    /// behaviour change, so it stay.
    fn replacements(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("language_model.model.norm", "output_norm"),
            ("language_model.model.", ""),
            ("language_model.", ""),
            ("layers", "blk"),
            ("transformer.layers", "blk"),
            ("vision_tower", "v"),
            ("ln_pre", "encoder_norm"),
            ("input_layernorm", "attn_norm"),
            ("post_attention_layernorm", "ffn_norm"),
            ("embed_tokens", "token_embd"),
            ("self_attn.q_proj", "attn_q"),
            ("self_attn.k_proj", "attn_k"),
            ("self_attn.v_proj", "attn_v"),
            ("self_attn.o_proj", "attn_output"),
            ("mlp.down_proj", "ffn_down"),
            ("mlp.gate_proj", "ffn_gate"),
            ("mlp.up_proj", "ffn_up"),
            ("attention.q_proj", "attn_q"),
            ("attention.k_proj", "attn_k"),
            ("attention.v_proj", "attn_v"),
            ("attention.o_proj", "attn_output"),
            ("attention_norm", "attn_norm"),
            ("feed_forward.gate_proj", "ffn_gate"),
            ("feed_forward.down_proj", "ffn_down"),
            ("feed_forward.up_proj", "ffn_up"),
            ("multi_modal_projector", "mm"),
            ("ffn_norm", "ffn_norm"),
            ("lm_head", "output"),
        ]
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers ------------------------------------------------------------

    /// Build a safetensors file in memory. See [`SafetensorMetadata`] for the
    /// container layout this is producing.
    fn safetensors_file(entries: &[(&str, &str, Vec<u64>, Vec<u8>)]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();

        for (name, dtype, shape, bytes) in entries {
            let start = data.len() as u64;
            data.extend_from_slice(bytes);
            let end = data.len() as u64;
            header.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [start, end],
                }),
            );
        }

        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header))
            .expect("header serialises");
        let mut out = Vec::new();
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(&data);
        out
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// Encode one protobuf field key. See [`decode_sentencepiece_model`].
    fn proto_key(field: u64, wire: u64) -> Vec<u8> {
        proto_varint((field << 3) | wire)
    }

    fn proto_varint(mut v: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let byte = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(byte);
                return out;
            }
            out.push(byte | 0x80);
        }
    }

    fn proto_sentencepiece(piece: &str, score: f32, kind: Option<i32>) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend(proto_key(1, 2));
        body.extend(proto_varint(piece.len() as u64));
        body.extend_from_slice(piece.as_bytes());
        body.extend(proto_key(2, 5));
        body.extend_from_slice(&score.to_bits().to_le_bytes());
        if let Some(k) = kind {
            body.extend(proto_key(3, 0));
            body.extend(proto_varint(k as u64));
        }

        let mut out = Vec::new();
        out.extend(proto_key(1, 2));
        out.extend(proto_varint(body.len() as u64));
        out.extend_from_slice(&body);
        out
    }

    // -- §1 json_compat.go --------------------------------------------------

    #[test]
    fn sanitizing_json_turns_bare_infinity_and_nan_into_zero() {
        let input = br#"{"a": Infinity, "b": -Infinity, "c": NaN, "d": [NaN, 1]}"#;
        let got = sanitize_non_finite_json(input);
        assert_eq!(
            String::from_utf8_lossy(&got),
            r#"{"a": 0, "b": 0, "c": 0, "d": [0, 1]}"#
        );
    }

    #[test]
    fn sanitizing_json_leaves_those_words_alone_inside_strings() {
        let input = br#"{"note": "NaN and Infinity are words", "v": NaN}"#;
        let got = sanitize_non_finite_json(input);
        assert_eq!(
            String::from_utf8_lossy(&got),
            r#"{"note": "NaN and Infinity are words", "v": 0}"#
        );
    }

    #[test]
    fn sanitizing_json_only_rewrites_whole_tokens() {
        let input = br#"{"a": NaNny, "b": 1}"#;
        assert_eq!(sanitize_non_finite_json(input), input.to_vec());
    }

    #[test]
    fn stripping_nulls_gives_serde_the_go_zero_value_behaviour() {
        let mut v: serde_json::Value =
            serde_json::from_str(r#"{"sliding_window": null, "layers": [{"a": null, "b": 1}]}"#)
                .expect("parses");
        strip_json_nulls(&mut v);
        assert_eq!(v, serde_json::json!({"layers": [{"b": 1}]}));

        // ...and that is what let a config with explicit nulls deserialise.
        let m: LlamaModel =
            parse_config(br#"{"rope_scaling": null, "head_dim": null, "hidden_size": 8}"#)
                .expect("null-bearing config parses");
        assert_eq!(m.hidden_size, 8);
        assert_eq!(m.head_dim, 0);
    }

    // -- §3 reader.go / tensor.go -------------------------------------------

    #[test]
    fn the_replacer_takes_the_first_matching_rule_not_the_longest() {
        // Order decide the answer -- this is the whole reason `Replacer` exist.
        let first_wins = Replacer::new([("ab", "X"), ("abc", "Y")]);
        assert_eq!(first_wins.replace("abc"), "Xc");

        let longer_first = Replacer::new([("abc", "Y"), ("ab", "X")]);
        assert_eq!(longer_first.replace("abc"), "Y");
    }

    #[test]
    fn the_replacer_never_rescans_what_it_just_wrote() {
        let r = Replacer::new([("a", "b"), ("b", "c")]);
        assert_eq!(r.replace("a"), "b");
    }

    #[test]
    fn llama_names_map_onto_ggml_names() {
        let r = Replacer::new(LlamaModel::default().replacements());
        for (hf, gguf) in [
            ("model.embed_tokens.weight", "token_embd.weight"),
            ("model.norm.weight", "output_norm.weight"),
            ("lm_head.weight", "output.weight"),
            ("model.layers.0.self_attn.q_proj.weight", "blk.0.attn_q.weight"),
            ("model.layers.7.self_attn.k_proj.weight", "blk.7.attn_k.weight"),
            ("model.layers.7.self_attn.v_proj.weight", "blk.7.attn_v.weight"),
            (
                "model.layers.7.self_attn.o_proj.weight",
                "blk.7.attn_output.weight",
            ),
            ("model.layers.3.mlp.gate_proj.weight", "blk.3.ffn_gate.weight"),
            ("model.layers.3.mlp.down_proj.weight", "blk.3.ffn_down.weight"),
            ("model.layers.3.mlp.up_proj.weight", "blk.3.ffn_up.weight"),
            ("model.layers.3.input_layernorm.weight", "blk.3.attn_norm.weight"),
            (
                "model.layers.3.post_attention_layernorm.weight",
                "blk.3.ffn_norm.weight",
            ),
        ] {
            assert_eq!(r.replace(hf), gguf, "mapping {hf}");
        }
    }

    #[test]
    fn gemma3_vision_embeddings_win_over_the_shorter_vision_prefix() {
        let r = Replacer::new(Gemma3Model::default().replacements());
        assert_eq!(
            r.replace("vision_tower.vision_model.embeddings.patch_embedding.weight"),
            "v.patch_embedding.weight"
        );
        assert_eq!(
            r.replace("language_model.model.layers.0.self_attn.q_proj.weight"),
            "blk.0.attn_q.weight"
        );
    }

    #[test]
    fn mixtral_collapses_the_expert_path_so_the_merge_patterns_match() {
        let r = Replacer::new(MixtralModel::default().replacements());
        // Note the **double dot**: `model.layers -> blk` consume up to
        // `model.layers`, the following `.0.` is copied verbatim, and then
        // `block_sparse_moe.experts. -> .` add its own dot. Upstream produce
        // exactly the same string, and it is harmless because
        // `blk.0.*.w1.weight` match it (`*` span the `.3`) and the merged
        // tensor get a fresh name anyway.
        assert_eq!(
            r.replace("model.layers.0.block_sparse_moe.experts.3.w1.weight"),
            "blk.0..3.w1.weight"
        );
        assert!(glob_match("blk.0.*.w1.weight", "blk.0..3.w1.weight"));
        assert_eq!(
            r.replace("model.layers.0.block_sparse_moe.gate.weight"),
            "blk.0.ffn_gate_inp.weight"
        );
    }

    #[test]
    fn tensor_kind_is_f32_for_vectors_and_for_the_hard_coded_list() {
        assert_eq!(
            base_tensor_kind("blk.0.attn_norm.weight", &[4096]).expect("rank 1"),
            TensorType::F32
        );
        assert_eq!(
            base_tensor_kind("blk.0.attn_q.weight", &[4096, 4096]).expect("rank 2"),
            TensorType::F16
        );
        assert_eq!(
            base_tensor_kind("blk.0.ffn_gate_inp.weight", &[8, 4096]).expect("router"),
            TensorType::F32
        );
        assert_eq!(
            base_tensor_kind("blk.0.attn_q.bias", &[4096, 1]).expect("bias"),
            TensorType::F32
        );
        assert_eq!(
            base_tensor_kind("v.position_embd.weight", &[16, 16]).expect("position table"),
            TensorType::F32
        );
        assert!(base_tensor_kind("whatever", &[]).is_err());
    }

    #[test]
    fn a_bf16_text_tensor_stays_bf16_but_a_vision_one_narrows_to_f16() {
        let make = |name: &str| SourceTensor {
            name: name.to_string(),
            shape: vec![8, 8],
            dtype: "BF16".to_string(),
            file: "m.safetensors".to_string(),
            offset: 0,
            size: 128,
            scale: None,
            fp8_block: None,
            repacker: None,
        };
        assert_eq!(
            make("blk.0.attn_q.weight").kind().expect("kind"),
            TensorType::BF16
        );
        assert_eq!(make("v.blk.0.attn_q.weight").kind().expect("kind"), TensorType::F16);
        assert_eq!(make("mm.0.weight").kind().expect("kind"), TensorType::F16);
    }

    #[test]
    fn glob_star_spans_dots_so_expert_patterns_match() {
        assert!(glob_match("blk.0.*.w1.weight", "blk.0.7.w1.weight"));
        assert!(!glob_match("blk.0.*.w1.weight", "blk.1.7.w1.weight"));
        assert!(glob_match("blk.0.*.w1.weight", "blk.0.a.b.w1.weight"));
    }

    #[test]
    fn merged_experts_sort_numerically_not_lexicographically() {
        let make = |name: &str| SourceTensor {
            name: name.to_string(),
            shape: vec![2, 2],
            dtype: "F32".to_string(),
            file: "m.safetensors".to_string(),
            offset: 0,
            size: 16,
            scale: None,
            fp8_block: None,
            repacker: None,
        };
        let ts = vec![
            make("blk.0.10.w1.weight"),
            make("blk.0.2.w1.weight"),
            make("blk.0.1.w1.weight"),
            make("blk.0.attn_q.weight"),
        ];

        let (merged, rest) = merge_tensors(
            ts,
            &[Merge::new("blk.0.*.w1.weight", "blk.0.ffn_gate_exps.weight")],
        )
        .expect("merge");

        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].name, "blk.0.ffn_gate_exps.weight");
        // Leading dim gain the expert count.
        assert_eq!(merged[0].shape, vec![3, 2, 2]);
        let TensorSource::Merge(group) = &merged[0].source else {
            panic!("expected a merge group");
        };
        let order: Vec<&str> = group.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            order,
            vec!["blk.0.1.w1.weight", "blk.0.2.w1.weight", "blk.0.10.w1.weight"]
        );
        assert_eq!(rest.len(), 1);
    }

    #[test]
    fn a_model_with_no_fp8_tensors_gains_no_provenance_keys() {
        let t = OutTensor::literal("rope_freqs.weight", vec![1.0, 2.0]);
        assert!(source_tensor_kv(&[t]).is_none());
    }

    // -- §4 reader_safetensors.go -------------------------------------------

    #[test]
    fn f32_to_f16_matches_upstreams_expected_bytes() {
        // Straight from ollama's own `reader_test.go` "fp32-fp16" case: the
        // f16 encoding of 0.0 .. 31.0, little-endian.
        let want: Vec<u8> = vec![
            0x00, 0x00, 0x00, 0x3c, 0x00, 0x40, 0x00, 0x42, 0x00, 0x44, 0x00, 0x45, 0x00, 0x46,
            0x00, 0x47, 0x00, 0x48, 0x80, 0x48, 0x00, 0x49, 0x80, 0x49, 0x00, 0x4a, 0x80, 0x4a,
            0x00, 0x4b, 0x80, 0x4b, 0x00, 0x4c, 0x40, 0x4c, 0x80, 0x4c, 0xc0, 0x4c, 0x00, 0x4d,
            0x40, 0x4d, 0x80, 0x4d, 0xc0, 0x4d, 0x00, 0x4e, 0x40, 0x4e, 0x80, 0x4e, 0xc0, 0x4e,
            0x00, 0x4f, 0x40, 0x4f, 0x80, 0x4f, 0xc0, 0x4f,
        ];
        let values: Vec<f32> = (0..32).map(|i| i as f32).collect();
        assert_eq!(encode_kind(TensorType::F16, &values).expect("encode"), want);
    }

    #[test]
    fn f16_round_trips_through_f32_and_handles_the_edges() {
        for bits in [0u16, 0x8000, 0x3c00, 0x7bff, 0x0001, 0x0400] {
            assert_eq!(f32_to_f16(f16_to_f32(bits)), bits, "round trip {bits:#06x}");
        }
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
        // Anything past f16's range saturate to Inf rather than wrapping.
        assert_eq!(f32_to_f16(1.0e30), 0x7c00);
        assert_eq!(f32_to_f16(-1.0e30), 0xfc00);
    }

    #[test]
    fn bf16_is_the_top_half_of_an_f32_and_encoding_truncates() {
        assert_eq!(f32_to_bf16(1.0f32), 0x3f80);
        assert_eq!(bf16_to_f32(0x3f80), 1.0f32);
        // Truncation, not rounding: the low mantissa bits are simply dropped.
        let v = f32::from_bits(0x3f80_ffff);
        assert_eq!(f32_to_bf16(v), 0x3f80);
    }

    #[test]
    fn fp8_e4m3fn_decodes_its_special_patterns() {
        assert_eq!(decode_float8_e4m3fn(0x00), 0.0);
        // exponent 7 (bias), mantissa 0 -> 1.0
        assert_eq!(decode_float8_e4m3fn(0x38), 1.0);
        assert_eq!(decode_float8_e4m3fn(0xb8), -1.0);
        // The one NaN pattern: exponent 0xf, mantissa 0x7.
        assert!(decode_float8_e4m3fn(0x7f).is_nan());
        // 0x7e is NOT Inf -- e4m3fn is finite-only, so it is a large number.
        assert_eq!(decode_float8_e4m3fn(0x7e), 448.0);
    }

    #[test]
    fn the_safetensors_header_is_eight_bytes_of_length_then_json() {
        let file = safetensors_file(&[
            (
                "model.embed_tokens.weight",
                "F32",
                vec![2, 2],
                f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
            ),
            ("model.norm.weight", "F32", vec![2], f32_bytes(&[5.0, 6.0])),
        ]);

        let mut files = MemoryFiles::new();
        files.insert("model.safetensors", file);

        let r = Replacer::new(LlamaModel::default().replacements());
        let ts = parse_tensors(&files, &r).expect("parses");

        assert_eq!(ts.len(), 2);
        let embd = ts
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .expect("renamed");
        assert_eq!(embd.shape, vec![2, 2]);
        assert_eq!(embd.dtype, "F32");
        // The offset is absolute: 8 + header_len + data_offsets[0].
        assert_eq!(embd.size, 16);
        assert_eq!(
            embd.materialise(&files).expect("bytes"),
            // rank 2 -> F16 on the way out
            encode_kind(TensorType::F16, &[1.0, 2.0, 3.0, 4.0]).expect("encode")
        );

        let norm = ts
            .iter()
            .find(|t| t.name == "output_norm.weight")
            .expect("renamed");
        // rank 1 -> stays F32, and the fast path copy the bytes straight out.
        assert_eq!(norm.kind().expect("kind"), TensorType::F32);
        assert_eq!(norm.materialise(&files).expect("bytes"), f32_bytes(&[5.0, 6.0]));
    }

    #[test]
    fn a_zero_dim_safetensors_scalar_is_promoted_to_rank_one() {
        let file = safetensors_file(&[("scale", "F32", vec![], f32_bytes(&[2.0]))]);
        let mut files = MemoryFiles::new();
        files.insert("model.safetensors", file);

        let ts = parse_tensors(&files, &Replacer::default()).expect("parses");
        assert_eq!(ts[0].shape, vec![1]);
    }

    #[test]
    fn a_checkpoint_with_only_pytorch_bins_reports_an_unknown_format() {
        let mut files = MemoryFiles::new();
        files.insert("pytorch_model.bin", vec![0u8; 4]);
        let err = parse_tensors(&files, &Replacer::default()).expect_err("no safetensors");
        assert!(matches!(err, ConvertError::UnknownTensorFormat));
    }

    // -- §5 tokenizer -------------------------------------------------------

    #[test]
    fn a_bpe_vocabulary_is_sorted_by_id_and_scored_with_the_id() {
        let json = serde_json::json!({
            "added_tokens": [{"id": 3, "content": "<|eot|>", "special": true}],
            "model": {"vocab": {"a": 0, "b": 1, "c": 2}},
        });
        let v = parse_vocabulary_from_tokenizer(json.to_string().as_bytes()).expect("parses");

        assert_eq!(v.model, "gpt2");
        assert_eq!(v.tokens, vec!["a", "b", "c", "<|eot|>"]);
        assert_eq!(v.scores, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(
            v.types,
            vec![
                token_type::NORMAL,
                token_type::NORMAL,
                token_type::NORMAL,
                token_type::CONTROL
            ]
        );
    }

    #[test]
    fn merges_parse_from_both_the_flat_and_the_paired_shape() {
        let flat = serde_json::json!({
            "model": {"vocab": {"a": 0}, "merges": ["a b", "c d"]},
        });
        let paired = serde_json::json!({
            "model": {"vocab": {"a": 0}, "merges": [["a", "b"], ["c", "d"]]},
        });

        for value in [flat, paired] {
            let mut files = MemoryFiles::new();
            files.insert("tokenizer.json", value.to_string().into_bytes());
            let t = parse_tokenizer(&files, DEFAULT_SPECIAL_TOKEN_TYPES).expect("parses");
            assert_eq!(t.merges, vec!["a b", "c d"]);
            // No Split pre-tokenizer -> empty digest -> "default".
            assert_eq!(t.pre, "default");
        }
    }

    #[test]
    fn a_special_token_only_survives_when_added_tokens_give_it_an_id() {
        let tok = serde_json::json!({
            "added_tokens": [{"id": 2, "content": "</s>", "special": true}],
            "model": {"vocab": {"a": 0, "b": 1}},
        });
        let cfg = serde_json::json!({
            "add_bos_token": true,
            "eos_token": {"content": "</s>"},
            "bos_token": "<s>",
            "chat_template": "hello",
        });

        let mut files = MemoryFiles::new();
        files.insert("tokenizer.json", tok.to_string().into_bytes());
        files.insert("tokenizer_config.json", cfg.to_string().into_bytes());

        let t = parse_tokenizer(&files, &["bos", "eos"]).expect("parses");
        assert_eq!(t.template, "hello");
        // `<s>` is not in added_tokens, so bos is dropped entirely.
        assert_eq!(t.special_vocabulary.len(), 1);
        assert_eq!(t.special_vocabulary[0].kind, "eos");
        assert_eq!(t.special_vocabulary[0].id, 2);
        assert!(!t.special_vocabulary[0].add_token_set);
    }

    #[test]
    fn special_vocabulary_keys_keep_upstreams_misspelling() {
        let key = |kind: &str| {
            SpecialVocabulary {
                kind: kind.to_string(),
                ..Default::default()
            }
            .key()
            .expect("known kind")
        };
        assert_eq!(key("bos"), "bos");
        assert_eq!(key("unk"), "unknown");
        assert_eq!(key("sep"), "seperator");
        assert_eq!(key("pad"), "padding");
        assert!(
            SpecialVocabulary {
                kind: "nonsense".to_string(),
                ..Default::default()
            }
            .key()
            .is_err()
        );
    }

    #[test]
    fn the_sentencepiece_protobuf_decodes_pieces_scores_and_types() {
        let mut model = Vec::new();
        model.extend(proto_sentencepiece("<unk>", 0.0, Some(token_type::UNKNOWN)));
        model.extend(proto_sentencepiece("hello", -1.5, None));
        model.extend(proto_sentencepiece("<end_of_turn>", -2.0, None));

        let v = parse_sentence_piece(&model, &[], None).expect("decodes");
        assert_eq!(v.model, "llama");
        assert_eq!(v.tokens, vec!["<unk>", "hello", "<end_of_turn>"]);
        assert_eq!(v.scores, vec![0.0, -1.5, -2.0]);
        assert_eq!(
            v.types,
            vec![
                token_type::UNKNOWN,
                // absent `type` default to NORMAL
                token_type::NORMAL,
                // ...unless it is on gemma's forced-control list
                token_type::CONTROL,
            ]
        );
    }

    #[test]
    fn added_tokens_must_extend_the_sentencepiece_vocabulary_contiguously() {
        let mut model = Vec::new();
        model.extend(proto_sentencepiece("a", 0.0, None));
        model.extend(proto_sentencepiece("b", 0.0, None));

        let ok = parse_sentence_piece(&model, &[], Some(br#"{"c": 2}"#)).expect("contiguous");
        assert_eq!(ok.tokens, vec!["a", "b", "c"]);
        assert_eq!(ok.scores[2], -1000.0);
        assert_eq!(ok.types[2], token_type::USER_DEFINED);

        // A gap would silently shift every later token, so it is refused.
        assert!(parse_sentence_piece(&model, &[], Some(br#"{"c": 5}"#)).is_err());
        // A mismatch at an existing index is refused too.
        assert!(parse_sentence_piece(&model, &[], Some(br#"{"z": 0}"#)).is_err());
    }

    // -- §6 convert.go ------------------------------------------------------

    #[test]
    fn unqualified_keys_gain_the_architecture_prefix_but_namespaced_ones_do_not() {
        assert_eq!(qualify_key("qwen3", "block_count"), "qwen3.block_count");
        assert_eq!(qualify_key("llama", "llama.block_count"), "llama.block_count");
        assert_eq!(qualify_key("llama", "general.file_type"), "general.file_type");
        assert_eq!(
            qualify_key("llama", "tokenizer.ggml.tokens"),
            "tokenizer.ggml.tokens"
        );
    }

    #[test]
    fn the_rope_permute_interleaves_the_two_halves_of_each_head() {
        // One head, leading dim 4, one column: rows [0,1,2,3] are
        // [first-half-0, first-half-1, second-half-0, second-half-1] and must
        // come out interleaved as [0, 2, 1, 3].
        let got = permute("blk.0.attn_q.weight", vec![0.0, 1.0, 2.0, 3.0], &[4, 1], 1)
            .expect("permutes");
        assert_eq!(got, vec![0.0, 2.0, 1.0, 3.0]);

        // Two heads over 4 rows: each head is one first-half row and one
        // second-half row, so the order is unchanged.
        let got = permute("blk.0.attn_q.weight", vec![0.0, 1.0, 2.0, 3.0], &[4, 1], 2)
            .expect("permutes");
        assert_eq!(got, vec![0.0, 1.0, 2.0, 3.0]);

        // A leading dimension that cannot be split is refused, not fudged.
        assert!(permute("t", vec![0.0; 6], &[6, 1], 4).is_err());
    }

    #[test]
    fn adding_one_to_a_gemma_norm_only_touches_the_text_tower() {
        let make = |name: &str| SourceTensor {
            name: name.to_string(),
            shape: vec![4],
            dtype: "F32".to_string(),
            file: "m.safetensors".to_string(),
            offset: 0,
            size: 16,
            scale: None,
            fp8_block: None,
            repacker: None,
        };
        let mut text = make("blk.0.attn_norm.weight");
        let mut vision = make("v.blk.0.attn_norm.weight");
        gemma_set_add_one(&mut text);
        gemma_set_add_one(&mut vision);
        assert!(text.repacker.is_some());
        assert!(vision.repacker.is_none());
        assert_eq!(add_one(vec![0.0, 1.5]), vec![1.0, 2.5]);
    }

    #[test]
    fn transposing_the_last_two_axes_moves_experts_into_ggml_order() {
        // [1, 2, 3] -> [1, 3, 2]
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let got = transpose_last_two("t", &data, 1, 2, 3).expect("transposes");
        assert_eq!(got, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    #[test]
    fn llama_metadata_carries_the_keys_a_runtime_reads() {
        let config = serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": 32000,
            "num_hidden_layers": 32,
            "max_position_embeddings": 8192,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "rope_theta": 500000.0,
            "rms_norm_eps": 1.0e-5,
        });
        let m: LlamaModel = parse_config(config.to_string().as_bytes()).expect("parses");
        let kv = m.kv(&Tokenizer::default()).expect("kv");

        // NOTE: `Kv`'s typed getters prefix an unqualified key with the
        // architecture (see `memory::Kv::lookup`), so `block_count` here really
        // read `llama.block_count`. Asking for `"llama.block_count"` would look
        // up `llama.llama.block_count` and quietly give the default back.
        assert_eq!(kv.string("general.architecture", ""), "llama");
        assert_eq!(kv.uint("block_count", 0), 32);
        assert_eq!(kv.uint("context_length", 0), 8192);
        assert_eq!(kv.uint("embedding_length", 0), 4096);
        assert_eq!(kv.uint("feed_forward_length", 0), 14336);
        assert_eq!(kv.uint("attention.head_count", 0), 32);
        assert_eq!(kv.uint("attention.head_count_kv", 0), 8);
        // hidden_size / head_count
        assert_eq!(kv.uint("rope.dimension_count", 0), 128);
        assert_eq!(kv.float("rope.freq_base", 0.0), 500000.0);
        // No `rope_scaling.type == "linear"`, so no scaling keys at all.
        assert!(kv.value("llama.rope.scaling.type").is_none());
        // General block from ModelParameters::kv.
        assert_eq!(kv.uint("general.file_type", 0), 1);
        assert_eq!(kv.uint("general.quantization_version", 0), 2);
    }

    #[test]
    fn the_llama3_rope_factor_table_is_flat_at_both_ends_and_ramps_between() {
        let config = serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "rope_theta": 500000.0,
            "rope_scaling": {
                "rope_type": "llama3",
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
            },
        });
        let m: LlamaModel = parse_config(config.to_string().as_bytes()).expect("parses");
        let factors = m.rope_factors().expect("llama3 gives factors");

        // One entry per rope dimension pair: 128 dims -> 64 factors.
        assert_eq!(factors.len(), 64);
        // Highest frequency: untouched.
        assert_eq!(factors[0], 1.0);
        // Lowest frequency: fully scaled.
        assert_eq!(factors[63], 8.0);
        // Monotonically non-decreasing in between.
        assert!(factors.windows(2).all(|w| w[0] <= w[1]));

        // A model that is not llama3-scaled get no table at all.
        let plain: LlamaModel =
            parse_config(br#"{"hidden_size": 4096, "num_attention_heads": 32}"#).expect("parses");
        assert!(plain.rope_factors().is_none());
    }

    #[test]
    fn qwen3_writes_unqualified_keys_and_switches_architecture_on_experts() {
        let dense: Qwen3Model =
            parse_config(br#"{"num_hidden_layers": 28, "head_dim": 128}"#).expect("parses");
        let kv = dense.kv(&Tokenizer::default()).expect("kv");
        assert_eq!(kv.string("general.architecture", ""), "qwen3");
        // Written bare -- the writer is what prefix it.
        assert!(kv.value("block_count").is_some());
        assert!(kv.value("qwen3.block_count").is_none());

        let moe: Qwen3Model =
            parse_config(br#"{"num_hidden_layers": 48, "num_experts": 128}"#).expect("parses");
        let kv = moe.kv(&Tokenizer::default()).expect("kv");
        assert_eq!(kv.string("general.architecture", ""), "qwen3moe");
        assert!(kv.value("expert_count").is_some());
    }

    #[test]
    fn an_unknown_rope_scaling_type_errors_instead_of_panicking() {
        let m: Qwen2Model =
            parse_config(br#"{"rope_scaling": {"type": "nonsense"}}"#).expect("parses");
        let err = m.kv(&Tokenizer::default()).expect_err("rejects");
        assert!(matches!(err, ConvertError::UnknownRopeScaling(t) if t == "nonsense"));
    }

    #[test]
    fn gemma3_marks_every_layer_local_except_each_nth_one() {
        let config = serde_json::json!({
            "architectures": ["Gemma3ForCausalLM"],
            "num_hidden_layers": 6,
            "sliding_window_pattern": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
        });
        let mut m: Gemma3Model = parse_config(config.to_string().as_bytes()).expect("parses");
        m.architecture = "Gemma3ForCausalLM".to_string();
        let kv = m.kv(&Tokenizer::default()).expect("kv");

        // (i+1) % 3 != 0 -> layers 0,1 local, 2 global, 3,4 local, 5 global.
        assert_eq!(
            kv.bools("attention.sliding_window_pattern", &[]),
            vec![true, true, false, true, true, false]
        );
        // The two rope thetas fall back to upstream's literals.
        assert_eq!(kv.float("rope.local.freq_base", 0.0), 10000.0);
        assert_eq!(kv.float("rope.freq_base", 0.0), 1000000.0);
    }

    #[test]
    fn gemma3_prefers_the_explicit_layer_types_list_over_the_pattern_int() {
        let config = serde_json::json!({
            "architectures": ["Gemma3ForCausalLM"],
            "num_hidden_layers": 3,
            "layer_types": ["sliding_attention", "full_attention", "sliding_attention"],
        });
        let mut m: Gemma3Model = parse_config(config.to_string().as_bytes()).expect("parses");
        m.architecture = "Gemma3ForCausalLM".to_string();
        let kv = m.kv(&Tokenizer::default()).expect("kv");
        assert_eq!(
            kv.bools("attention.sliding_window_pattern", &[]),
            vec![true, false, true]
        );
    }

    #[test]
    fn gemma3_substitutes_the_head_counts_for_a_known_layer_count() {
        for (layers, heads, kv_heads) in [(34u32, 8u32, 4u32), (48, 16, 8), (62, 32, 16)] {
            let config = serde_json::json!({
                "architectures": ["Gemma3ForCausalLM"],
                "num_hidden_layers": layers,
                "num_attention_heads": 999,
                "num_key_value_heads": 999,
            });
            let mut m: Gemma3Model = parse_config(config.to_string().as_bytes()).expect("parses");
            m.architecture = "Gemma3ForCausalLM".to_string();
            let kv = m.kv(&Tokenizer::default()).expect("kv");
            assert_eq!(kv.uint("attention.head_count", 0), heads);
            assert_eq!(kv.uint("attention.head_count_kv", 0), kv_heads);
        }
    }

    #[test]
    fn phi3_emits_the_longrope_factor_tables_as_tensors() {
        let config = serde_json::json!({
            "architectures": ["Phi3ForCausalLM"],
            "max_position_embeddings": 131072,
            "original_max_position_embeddings": 4096,
            "rope_scaling": {
                "type": "longrope",
                "long_factor": [1.0, 1.1],
                "short_factor": [1.0, 1.0],
            },
        });
        let m: Phi3Model = parse_config(config.to_string().as_bytes()).expect("parses");

        let kv = m.kv(&Tokenizer::default()).expect("kv");
        // sqrt(1 + ln(32) / ln(4096)), floored at 1.0.
        let want = (1.0f64 + 32.0f64.ln() / 4096.0f64.ln()).sqrt() as f32;
        assert!((kv.float("rope.scaling.attn_factor", 0.0) - want).abs() < 1e-6);

        let ts = vec![SourceTensor {
            name: "blk.0.attn_q.weight".to_string(),
            shape: vec![4, 4],
            dtype: "F32".to_string(),
            file: "m.safetensors".to_string(),
            offset: 0,
            size: 64,
            scale: None,
            fp8_block: None,
            repacker: None,
        }];
        let out = m.tensors(ts).expect("tensors");
        let names: Vec<&str> = out.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "rope_factors_long.weight",
                "rope_factors_short.weight",
                "blk.0.attn_q.weight"
            ]
        );
    }

    #[test]
    fn command_r_writes_a_hyphenated_architecture_and_an_explicit_no_scaling() {
        let m: CommandRModel = parse_config(
            br#"{"model_max_length": 131072, "max_position_embeddings": 8192, "hidden_size": 8192}"#,
        )
        .expect("parses");
        let kv = m.kv(&Tokenizer::default()).expect("kv");
        assert_eq!(kv.string("general.architecture", ""), "command-r");
        assert_eq!(kv.string("general.name", ""), "command-r");
        // model_max_length win over max_position_embeddings.
        assert_eq!(kv.uint("context_length", 0), 131072);
        assert_eq!(kv.string("rope.scaling.type", ""), "none");
    }

    #[test]
    fn mistral3_reads_its_numbers_out_of_text_config_not_the_top_level() {
        let config = serde_json::json!({
            "architectures": ["Mistral3ForConditionalGeneration"],
            "text_config": {
                "num_hidden_layers": 40,
                "hidden_size": 5120,
                "num_attention_heads": 32,
                "head_dim": 128,
                "rope_parameters": {"rope_theta": 1000000000.0, "rope_type": "yarn"},
            },
            "vision_config": {"hidden_size": 1024, "patch_size": 14},
        });
        let m: Mistral3Model = parse_config(config.to_string().as_bytes()).expect("parses");
        let kv = m.kv(&Tokenizer::default()).expect("kv");

        assert_eq!(kv.uint("block_count", 0), 40);
        assert_eq!(kv.uint("embedding_length", 0), 5120);
        assert_eq!(kv.uint("rope.dimension_count", 0), 128);
        // rope_theta absent at text_config level -> nested rope_parameters win.
        assert_eq!(kv.float("rope.freq_base", 0.0), 1000000000.0);
        assert_eq!(kv.uint("vision.embedding_length", 0), 1024);
        // Upstream comment this one out; we must not write it either.
        assert!(kv.value("mistral3.vision.attention.layer_norm_epsilon").is_none());
    }

    #[test]
    fn duplicate_gguf_names_are_refused_rather_than_silently_shadowed() {
        let make = |name: &str| SourceTensor {
            name: name.to_string(),
            shape: vec![2],
            dtype: "F32".to_string(),
            file: "m.safetensors".to_string(),
            offset: 0,
            size: 8,
            scale: None,
            fp8_block: None,
            repacker: None,
        };
        let err = ensure_unique_tensor_names(&[make("blk.0.attn_q.weight"), make("blk.0.attn_q.weight")])
            .expect_err("duplicate");
        assert!(matches!(err, ConvertError::DuplicateTensorName(n) if n == "blk.0.attn_q.weight"));
    }

    // -- end to end ---------------------------------------------------------

    /// A whole (tiny) llama checkpoint, converted through the recording writer.
    /// This is the closest thing here to upstream's `TestConvertModel`, minus
    /// the real weights it need testdata for.
    fn tiny_llama_checkpoint() -> MemoryFiles {
        let config = serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": 4,
            "num_hidden_layers": 1,
            "max_position_embeddings": 128,
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1.0e-5,
        });
        let tokenizer = serde_json::json!({
            "added_tokens": [{"id": 3, "content": "</s>", "special": true}],
            "model": {"vocab": {"a": 0, "b": 1, "c": 2}, "merges": ["a b"]},
        });
        let tokenizer_config = serde_json::json!({
            "add_eos_token": false,
            "eos_token": "</s>",
            "chat_template": "{{ messages }}",
        });

        let tensors = safetensors_file(&[
            (
                "model.embed_tokens.weight",
                "F32",
                vec![4, 4],
                f32_bytes(&(0..16).map(|i| i as f32).collect::<Vec<_>>()),
            ),
            (
                "model.layers.0.self_attn.q_proj.weight",
                "F32",
                vec![4, 4],
                f32_bytes(&(0..16).map(|i| i as f32).collect::<Vec<_>>()),
            ),
            (
                "model.norm.weight",
                "F32",
                vec![4],
                f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
            ),
        ]);

        let mut files = MemoryFiles::new();
        files.insert("config.json", config.to_string().into_bytes());
        files.insert("tokenizer.json", tokenizer.to_string().into_bytes());
        files.insert(
            "tokenizer_config.json",
            tokenizer_config.to_string().into_bytes(),
        );
        files.insert("model.safetensors", tensors);
        files
    }

    #[test]
    fn a_whole_tiny_llama_checkpoint_converts_end_to_end() {
        let files = tiny_llama_checkpoint();
        let mut w = RecordingGgufWriter::new();
        convert_model(&files, &mut w).expect("converts");

        assert_eq!(w.architecture, "llama");
        assert_eq!(w.kv.string("general.architecture", ""), "llama");
        assert_eq!(w.kv.uint("block_count", 0), 1);
        assert_eq!(w.kv.uint("embedding_length", 0), 4);
        assert_eq!(w.kv.string("tokenizer.ggml.model", ""), "gpt2");
        assert_eq!(w.kv.strings("tokenizer.ggml.merges", &[]), vec!["a b"]);
        assert_eq!(w.kv.string("tokenizer.chat_template", ""), "{{ messages }}");
        // `add_eos_token` was explicitly present, so the flag is written.
        assert!(w.kv.value("tokenizer.ggml.add_eos_token").is_some());
        assert_eq!(w.kv.uint("tokenizer.ggml.eos_token_id", 99), 3);

        let names: BTreeSet<&str> = w.tensors.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains("token_embd.weight"));
        assert!(names.contains("blk.0.attn_q.weight"));
        assert!(names.contains("output_norm.weight"));

        // Shape is REVERSED on the way out: safetensors [4, 4] is symmetric, so
        // check the rank-1 one and the rank-2 kinds instead.
        let embd = w.tensor("token_embd.weight").expect("present");
        assert_eq!(embd.kind, TensorType::F16);
        assert_eq!(embd.shape, vec![4, 4]);
        assert_eq!(embd.data.len(), 16 * 2);

        let norm = w.tensor("output_norm.weight").expect("present");
        assert_eq!(norm.kind, TensorType::F32);
        assert_eq!(norm.shape, vec![4]);
        assert_eq!(norm.data, f32_bytes(&[1.0, 2.0, 3.0, 4.0]));

        // attn_q went through the rope permute, so its bytes are NOT the
        // input's even though both tensors started from the same 0..15 values.
        // With one head over a leading dim of 4, `d == 2`: rows
        // [0, 1, 2, 3] become [0, 2, 1, 3], each row being 4 values wide.
        let q = w.tensor("blk.0.attn_q.weight").expect("present");
        let want: Vec<f32> = [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15]
            .iter()
            .map(|i| *i as f32)
            .collect();
        assert_eq!(q.data, encode_kind(TensorType::F16, &want).expect("encode"));
        assert_ne!(q.data, embd.data);
    }

    #[test]
    fn a_shape_is_reversed_on_the_way_into_the_gguf() {
        let file = safetensors_file(&[(
            "model.embed_tokens.weight",
            "F32",
            vec![4, 2],
            f32_bytes(&(0..8).map(|i| i as f32).collect::<Vec<_>>()),
        )]);
        let mut files = tiny_llama_checkpoint();
        files.insert("model.safetensors", file);

        let mut w = RecordingGgufWriter::new();
        convert_model(&files, &mut w).expect("converts");

        // safetensors [4, 2] -> GGUF ne [2, 4].
        assert_eq!(w.tensor("token_embd.weight").expect("present").shape, vec![2, 4]);
    }

    #[test]
    fn a_short_vocabulary_is_padded_up_to_the_configs_vocab_size() {
        let mut files = tiny_llama_checkpoint();
        let config = serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": 6,
            "num_hidden_layers": 1,
            "hidden_size": 4,
        });
        files.insert("config.json", config.to_string().into_bytes());

        let (_, t) = load_model_metadata(&files).expect("loads");
        assert_eq!(t.vocabulary.tokens.len(), 6);
        assert_eq!(t.vocabulary.tokens[4], "[PAD0]");
        assert_eq!(t.vocabulary.tokens[5], "[PAD1]");
        assert_eq!(t.vocabulary.scores[5], -1.0);
        assert_eq!(t.vocabulary.types[5], token_type::USER_DEFINED);
    }

    #[test]
    fn an_architecture_we_have_not_ported_says_so_plainly() {
        let mut files = tiny_llama_checkpoint();
        files.insert(
            "config.json",
            br#"{"architectures": ["BertModel"]}"#.to_vec(),
        );
        // `Box<dyn ModelConverter>` is not `Debug`, so unwrap the error by hand.
        let Err(err) = load_model_metadata(&files) else {
            panic!("BertModel should not be supported yet");
        };
        assert!(matches!(err, ConvertError::UnsupportedArchitecture(a) if a == "BertModel"));

        files.insert("config.json", br#"{"architectures": []}"#.to_vec());
        let Err(err) = load_model_metadata(&files) else {
            panic!("an empty architectures list should not resolve");
        };
        assert!(matches!(err, ConvertError::UnknownArchitecture));
    }

    #[test]
    fn a_directory_backed_checkpoint_reads_the_same_as_an_in_memory_one() {
        let dir = tempfile::tempdir().expect("tempdir");
        let files = tiny_llama_checkpoint();
        for name in files.names() {
            let bytes = files.read(&name).expect("read").expect("present");
            std::fs::write(dir.path().join(&name), bytes).expect("write");
        }

        let on_disk = DirFiles::new(dir.path());
        assert_eq!(on_disk.names(), files.names());

        let mut a = RecordingGgufWriter::new();
        let mut b = RecordingGgufWriter::new();
        convert_model(&files, &mut a).expect("memory converts");
        convert_model(&on_disk, &mut b).expect("disk converts");
        assert_eq!(a.kv, b.kv);
        assert_eq!(a.tensors, b.tensors);
    }
}

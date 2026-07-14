//! The format-agnostic result of loading a model file, and the trait that
//! produces it.

use std::io::Read;
use std::path::Path;

use indexmap::IndexMap;
use kopitiam_core::{DType, Error, Result, Shape};

use crate::byte_source::ByteSource;
use crate::gguf::GgufLoader;
use crate::metadata::ModelMetadata;
use crate::safetensors::SafeTensorsLoader;

/// One tensor's identity and location, without its bytes.
///
/// Deliberately does *not* hold a `&[u8]` or a `Vec<u8>` directly. Doing so
/// would tie every `TensorEntry` to the lifetime of (or force a copy out
/// of) the [`LoadedModel`] that owns the backing storage, for no benefit —
/// nothing in this crate ever needs a tensor's bytes without also having
/// the `LoadedModel` at hand. Storing an offset and length instead, and
/// serving bytes through [`LoadedModel::tensor_bytes`], keeps `TensorEntry`
/// cheap to clone and keeps the bounds check in exactly one place
/// ([`ByteSource::slice`](crate::byte_source::ByteSource::slice)).
#[derive(Debug, Clone, PartialEq)]
pub struct TensorEntry {
    /// The tensor's name as it appears in the file, e.g. `"blk.0.attn_q.weight"`.
    pub name: String,
    pub dtype: DType,
    /// Dimensions in [`kopitiam_core::Shape`]'s convention: outermost
    /// first, row-major, last dimension contiguous. See the crate-level
    /// docs and [`crate::gguf`] for why GGUF's on-disk dimension order has
    /// to be reversed to produce this.
    pub shape: Shape,
    pub(crate) offset: usize,
    pub(crate) len: usize,
}

/// A parsed model file: metadata plus a directory of tensors, backed by one
/// open file.
///
/// This type intentionally never constructs a `Tensor`. `kopitiam-tensor`
/// owns that type and is being developed independently of this crate; a
/// loader that returned tensors would have to depend on (and track the
/// in-flux API of) a crate it has no need to know about. Instead,
/// `LoadedModel` hands out exactly what a tensor needs to be built —
/// dtype, shape, and raw bytes — and lets whoever owns `Tensor` do the
/// building.
pub struct LoadedModel {
    pub(crate) metadata: ModelMetadata,
    pub(crate) tensors: IndexMap<String, TensorEntry>,
    pub(crate) source: ByteSource,
    pub(crate) format: &'static str,
}

impl std::fmt::Debug for LoadedModel {
    /// Deliberately hand-written rather than `#[derive]`d: the backing
    /// storage can be a multi-gigabyte `mmap`, and a derived `Debug` would
    /// either fail to compile (`memmap2::Mmap` is not `Debug`) or, worse,
    /// succeed and dump gigabytes of tensor bytes into a log line. This
    /// prints only what is useful for diagnostics: the format, tensor
    /// count and names, and the metadata.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedModel")
            .field("format", &self.format)
            .field("metadata", &self.metadata)
            .field("tensors", &self.tensors.values().collect::<Vec<_>>())
            .finish()
    }
}

impl LoadedModel {
    /// The format-agnostic hyperparameters and raw metadata bag.
    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    /// `"gguf"` or `"safetensors"` — matches the `format` field of any
    /// [`Error::MalformedModel`] this model's loader could have produced.
    pub fn format(&self) -> &'static str {
        self.format
    }

    /// Number of tensors in the file.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Tensor names, in file order.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// Looks up a tensor's name/dtype/shape by name, without touching its
    /// bytes.
    pub fn tensor(&self, name: &str) -> Option<&TensorEntry> {
        self.tensors.get(name)
    }

    /// All tensor entries, in file order.
    pub fn tensors(&self) -> impl Iterator<Item = &TensorEntry> {
        self.tensors.values()
    }

    /// The raw bytes of the named tensor.
    ///
    /// This is the loader/tensor boundary: whoever builds a
    /// `kopitiam_tensor::Tensor` from a loaded model calls this to get the
    /// bytes, combines them with [`TensorEntry::dtype`] and
    /// [`TensorEntry::shape`] (available via [`LoadedModel::tensor`]), and
    /// constructs the tensor on their side of the boundary. The bytes
    /// returned are exactly the on-disk encoding for `dtype` — still
    /// block-quantized if `dtype` is quantized, still `f16`/`bf16` if the
    /// file stored it that way. This loader does not dequantize or convert
    /// anything.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let entry = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::MissingTensor { name: name.to_string() })?;
        self.source.slice(self.format, entry.offset, entry.len)
    }
}

/// A parser for one on-disk model format.
///
/// Implemented by [`crate::GgufLoader`] and [`crate::SafeTensorsLoader`].
/// The trait exists (rather than two unrelated `load` free functions) so
/// that callers who *do* know their format up front can hold a
/// `Box<dyn ModelLoader>` chosen once, and so that [`load_model`] itself is
/// just "try each known loader" instead of hand-rolled dispatch logic
/// duplicated at every call site that needs to support both formats.
pub trait ModelLoader {
    /// A short, lowercase name for the format this loader parses — always
    /// one of the `format` strings this loader's errors carry.
    fn format_name(&self) -> &'static str;

    /// Cheap, non-destructive sniff: does `bytes` (the start of the file)
    /// look like this loader's format? Used by [`load_model`] to pick a
    /// loader by content rather than by file extension.
    ///
    /// A `true` result is not a promise the file will load successfully —
    /// only that it is worth trying. A `false` result *is* a promise this
    /// loader will refuse the file, so [`load_model`] can skip straight to
    /// the next candidate without wasting a full parse attempt.
    fn probe(&self, bytes: &[u8]) -> bool;

    /// Parses the file at `path` into a [`LoadedModel`].
    fn load(&self, path: &Path) -> Result<LoadedModel>;
}

/// Loads a model file, choosing GGUF or SafeTensors by sniffing its
/// content.
///
/// # Why content, not extension
///
/// GGUF files always start with the four-byte magic `b"GGUF"`
/// ([`GgufLoader::probe`](crate::GgufLoader)), so sniffing it is exact.
/// SafeTensors has no magic number at all — a file starting with anything
/// other than the GGUF magic is *assumed* to be SafeTensors and handed to
/// [`SafeTensorsLoader`], which will itself fail with
/// [`Error::MalformedModel`] if that assumption is wrong. This is still
/// preferable to trusting a `.gguf`/`.safetensors` extension: an extension
/// is metadata about the file, supplied by whoever named it, not a fact
/// about the bytes — sniffing means a renamed or extension-less file still
/// loads correctly, and a genuinely unrecognized file still fails loudly
/// rather than silently mis-parsing.
pub fn load_model(path: impl AsRef<Path>) -> Result<LoadedModel> {
    let path = path.as_ref();

    // Read only the handful of bytes probing needs — not the whole file.
    // A file shorter than 8 bytes legitimately yields fewer; that is not an
    // I/O error, it just means every probe below sees a shorter (possibly
    // empty) slice and correctly reports "not this format". `take(8)` plus
    // `read_to_end` (rather than one `read` call into a fixed buffer) is
    // used because a single `read` is allowed to return short even when
    // more bytes are available.
    let file = std::fs::File::open(path)?;
    let mut probe_bytes = Vec::with_capacity(8);
    file.take(8).read_to_end(&mut probe_bytes)?;

    let gguf = GgufLoader;
    if gguf.probe(&probe_bytes) {
        return gguf.load(path);
    }

    SafeTensorsLoader.load(path)
}

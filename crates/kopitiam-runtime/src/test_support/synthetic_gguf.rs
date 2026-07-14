//! Hand-builds a tiny, valid, on-disk GGUF file describing a Qwen-shaped
//! transformer with random-but-fixed weights.
//!
//! This mirrors `crates/kopitiam-loader/tests/common/mod.rs`'s byte-level
//! GGUF-writing helpers (that module cannot be reused directly: it is
//! private to `kopitiam-loader`'s integration test binary, not exported by
//! the crate), field-for-field against the wire format documented in
//! `kopitiam_loader::gguf`. Writing a fixture from scratch here, in a
//! format this module and `kopitiam_loader::gguf` both independently agree
//! on, is what lets [`crate::model`]'s end-to-end test prove the whole
//! `kopitiam-runtime` stack (load -> build weights -> forward pass) works
//! without needing a multi-gigabyte real model file as a test asset — see
//! the Model Runtime task brief's "no real model on disk -> build a tiny
//! synthetic GGUF" instruction.

const ALIGNMENT: usize = 32;

const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_Q8_0: u32 = 8;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_U32: u32 = 4;
const GGUF_TYPE_F32: u32 = 6;

/// The shape of a synthetic model. Every field mirrors one
/// [`kopitiam_loader::ModelMetadata`] field; see [`crate::config::QwenConfig`]
/// for what each becomes once resolved.
#[derive(Clone)]
pub(crate) struct SyntheticModelSpec {
    pub architecture: &'static str,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub hidden_size: usize,
    pub ffn_hidden_size: usize,
    pub vocab_size: usize,
    pub context_length: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
    /// Whether to omit `output.weight`, exercising the tied-embedding
    /// fallback (see [`crate::weights::ModelWeights::output_weight`]).
    pub tie_embeddings: bool,
    /// Whether attention Q/K/V get bias tensors (Qwen2's actual
    /// configuration) or not (plain LLaMA's).
    pub with_qkv_bias: bool,
    /// Whether every matmul-operand weight (`wq`/`wk`/`wv`/`wo`,
    /// `ffn_gate`/`ffn_up`/`ffn_down`, and `output.weight` when present)
    /// is written as real on-disk `Q8_0` tensors instead of `f32` — see
    /// [`quantize_q8_0_blocks`]. Token embeddings and norm weights are
    /// never quantized regardless of this flag (see
    /// [`crate::weights::ModelWeights`]'s docs on why those two are not
    /// matmul operands). `false` by default, matching every GGUF export
    /// this fixture produced before quantized loading existed.
    ///
    /// Every weight this flag affects has an `in_features` dimension
    /// (`hidden_size` or `ffn_hidden_size`) that must be a multiple of 32
    /// for a `Q8_0` row to be block-aligned — [`SyntheticModelSpec::default`]'s
    /// `hidden_size` (16) is deliberately *not*, so this flag should only
    /// be combined with a spec built from [`SyntheticModelSpec::quantized_benchmark`].
    pub quantize_matmul_weights: bool,
}

impl Default for SyntheticModelSpec {
    /// A GQA-shaped (`n_heads=4`, `n_kv_heads=2`) 2-layer toy model: small
    /// enough that a unit test builds and parses it in milliseconds, large
    /// enough that `hidden_size` (16) evenly divides into 4 heads of
    /// `head_dim=4`, which itself is even (required for RoPE's split-half
    /// pairing — see [`crate::rope`]).
    fn default() -> Self {
        Self {
            architecture: "qwen2",
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            hidden_size: 16,
            ffn_hidden_size: 32,
            vocab_size: 37,
            context_length: 64,
            rope_theta: 10_000.0,
            norm_eps: 1e-6,
            tie_embeddings: true,
            with_qkv_bias: true,
            quantize_matmul_weights: false,
        }
    }
}

impl SyntheticModelSpec {
    pub(crate) fn head_dim(&self) -> usize {
        self.hidden_size / self.n_heads
    }

    /// A larger, `Q8_0`-block-alignment-friendly spec: `hidden_size` (256)
    /// and `ffn_hidden_size` (512) are both multiples of 32, so every
    /// matmul-operand weight's rows are block-aligned when
    /// [`SyntheticModelSpec::quantize_matmul_weights`] is set. Also large
    /// enough (4 layers, 512-token vocab) that a wall-clock benchmark
    /// comparing this spec's `f32` and quantized variants (see
    /// `crate::model::tests`' `bench_` functions) measures something more
    /// representative than [`SyntheticModelSpec::default`]'s
    /// milliseconds-scale toy model.
    pub(crate) fn quantized_benchmark() -> Self {
        Self {
            hidden_size: 256,
            n_heads: 8,
            n_kv_heads: 4,
            ffn_hidden_size: 512,
            vocab_size: 512,
            n_layers: 4,
            context_length: 512,
            ..Self::default()
        }
    }
}

/// A deterministic, dependency-free PRNG (xorshift64*) used only to fill
/// synthetic weight tensors with small, reproducible, non-degenerate
/// values. Not cryptographic, not `rand` — a fixed seed plus a simple
/// recurrence is all a test fixture needs, and pulling in a `rand`
/// dependency for this alone would violate CLAUDE.md's "avoid unnecessary
/// dependencies".
struct Xorshift64(u64);

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// A pseudo-random `f32` in `[-0.1, 0.1]` — small enough that the
    /// forward pass's `f32` accumulations stay well away from overflow
    /// across every op this crate chains together.
    fn next_weight(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        let unit = (self.0 >> 40) as u32 as f32 / (1u32 << 24) as f32; // in [0, 1)
        (unit - 0.5) * 0.2
    }
}

fn push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn push_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn push_string(buf: &mut Vec<u8>, s: &str) {
    push_u64(buf, s.len() as u64);
    buf.extend_from_slice(s.as_bytes());
}

fn pad_to_alignment(buf: &mut Vec<u8>, alignment: usize) {
    let pad = alignment.wrapping_sub(buf.len() % alignment) % alignment;
    buf.extend(std::iter::repeat_n(0u8, pad));
}

/// Encodes `data` (length a multiple of 32) as raw Q8_0 block bytes,
/// matching `kopitiam_tensor::quant`'s decode formula (`qs[j] * d`, `d =
/// max(|block|) / 127`) exactly — see that module's docs for the on-disk
/// block layout this reproduces byte-for-byte.
///
/// This is a small, from-scratch, test-only encoder, not a use of
/// `kopitiam-tensor`'s internals: that crate deliberately has no *public*
/// `f32 -> quantized` encoder (see its crate docs on why requantizing
/// weights is a model-export concern, not a forward-pass one), so this
/// fixture — which needs to hand the real `kopitiam_loader` GGUF parser
/// genuine quantized bytes, not a shortcut — writes its own, independently,
/// the same way [`Xorshift64`] above is its own independent PRNG rather
/// than a borrowed one.
fn quantize_q8_0_blocks(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / 32 * 34);
    for chunk in data.chunks_exact(32) {
        let amax = chunk.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&kopitiam_tensor::f32_to_f16(d).to_le_bytes());
        for &v in chunk {
            let q = (v * id).round().clamp(-127.0, 127.0) as i8;
            out.push(q as u8);
        }
    }
    out
}

/// Accumulates metadata KVs, tensor info records, and tensor data
/// separately (matching GGUF's three-section layout) and assembles them
/// into one file in [`GgufBuilder::build`].
struct GgufBuilder {
    kv_count: u64,
    kvs: Vec<u8>,
    tensor_count: u64,
    tensor_infos: Vec<u8>,
    /// Tensor bytes, laid out exactly as they will appear in the file's
    /// tensor-data section: each tensor's bytes start at an
    /// [`ALIGNMENT`]-aligned offset *within this buffer*, which is what
    /// makes every `relative_offset` recorded in `tensor_infos` valid
    /// regardless of where the tensor-data section itself lands in the
    /// final file (see `kopitiam_loader::gguf`'s `tensor_data_start`
    /// computation, which independently aligns the section start).
    tensor_data: Vec<u8>,
}

impl GgufBuilder {
    fn new() -> Self {
        Self { kv_count: 0, kvs: Vec::new(), tensor_count: 0, tensor_infos: Vec::new(), tensor_data: Vec::new() }
    }

    fn kv_string(&mut self, key: &str, value: &str) {
        push_string(&mut self.kvs, key);
        push_u32(&mut self.kvs, GGUF_TYPE_STRING);
        push_string(&mut self.kvs, value);
        self.kv_count += 1;
    }

    fn kv_u32(&mut self, key: &str, value: u32) {
        push_string(&mut self.kvs, key);
        push_u32(&mut self.kvs, GGUF_TYPE_U32);
        push_u32(&mut self.kvs, value);
        self.kv_count += 1;
    }

    fn kv_f32(&mut self, key: &str, value: f32) {
        push_string(&mut self.kvs, key);
        push_u32(&mut self.kvs, GGUF_TYPE_F32);
        self.kvs.extend_from_slice(&value.to_le_bytes());
        self.kv_count += 1;
    }

    /// Adds an `f32` tensor named `name` with [`kopitiam_core::Shape`]
    /// convention dims (outermost first) `shape`, whose elements are
    /// `data` in row-major order.
    fn tensor_f32(&mut self, name: &str, shape: &[usize], data: &[f32]) {
        assert_eq!(shape.iter().product::<usize>(), data.len(), "tensor {name}: shape/data length mismatch");

        pad_to_alignment(&mut self.tensor_data, ALIGNMENT);
        let relative_offset = self.tensor_data.len() as u64;

        // GGUF's ne[] is fastest-varying-first: the reverse of Shape's
        // outermost-first convention (see kopitiam_loader::gguf's module
        // docs, "the dimension-order trap").
        let ne: Vec<u64> = shape.iter().rev().map(|&d| d as u64).collect();

        push_string(&mut self.tensor_infos, name);
        push_u32(&mut self.tensor_infos, ne.len() as u32);
        for &d in &ne {
            push_u64(&mut self.tensor_infos, d);
        }
        push_u32(&mut self.tensor_infos, GGML_TYPE_F32);
        push_u64(&mut self.tensor_infos, relative_offset);
        self.tensor_count += 1;

        for &v in data {
            self.tensor_data.extend_from_slice(&v.to_le_bytes());
        }
    }

    /// Adds a real on-disk `Q8_0` tensor named `name`, quantizing `data`
    /// (row-major, `shape` outermost-first) via [`quantize_q8_0_blocks`].
    /// `shape`'s last dimension (`in_features`, in `[out, in]` weight
    /// convention) must be a multiple of 32 — see
    /// [`SyntheticModelSpec::quantize_matmul_weights`]'s docs.
    fn tensor_q8_0(&mut self, name: &str, shape: &[usize], data: &[f32]) {
        let elems: usize = shape.iter().product();
        assert_eq!(elems, data.len(), "tensor {name}: shape/data length mismatch");
        let in_features = *shape.last().expect("Q8_0 weight tensors are at least rank 1");
        assert!(
            in_features.is_multiple_of(32),
            "tensor {name}: Q8_0 requires each row ({in_features} elements) to be a whole number of 32-element blocks"
        );

        pad_to_alignment(&mut self.tensor_data, ALIGNMENT);
        let relative_offset = self.tensor_data.len() as u64;
        let ne: Vec<u64> = shape.iter().rev().map(|&d| d as u64).collect();

        push_string(&mut self.tensor_infos, name);
        push_u32(&mut self.tensor_infos, ne.len() as u32);
        for &d in &ne {
            push_u64(&mut self.tensor_infos, d);
        }
        push_u32(&mut self.tensor_infos, GGML_TYPE_Q8_0);
        push_u64(&mut self.tensor_infos, relative_offset);
        self.tensor_count += 1;

        self.tensor_data.extend(quantize_q8_0_blocks(data));
    }

    /// Writes `name` as `Q8_0` if `quantize` is set, `f32` otherwise — the
    /// single call site every matmul-operand weight in [`build`] goes
    /// through, so the quantized/unquantized choice lives in one place
    /// instead of being duplicated at every weight tensor's call site.
    fn matmul_weight(&mut self, quantize: bool, name: &str, shape: &[usize], data: &[f32]) {
        if quantize {
            self.tensor_q8_0(name, shape, data);
        } else {
            self.tensor_f32(name, shape, data);
        }
    }

    fn build(self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        push_u32(&mut buf, 3); // version
        push_u64(&mut buf, self.tensor_count);
        push_u64(&mut buf, self.kv_count);
        buf.extend_from_slice(&self.kvs);
        buf.extend_from_slice(&self.tensor_infos);
        pad_to_alignment(&mut buf, ALIGNMENT);
        buf.extend_from_slice(&self.tensor_data);
        buf
    }
}

/// Builds the raw bytes of a synthetic GGUF file matching `spec`: every
/// metadata key [`crate::config::QwenConfig::from_metadata`] needs, plus a
/// full set of weight tensors (embedding, per-layer attention/MLP/norm
/// weights, output norm, and — unless `spec.tie_embeddings` — a separate
/// output projection) named per GGUF's standardized tensor-naming
/// convention (`token_embd.weight`, `blk.N.attn_q.weight`, ...).
pub(crate) fn build(spec: &SyntheticModelSpec) -> Vec<u8> {
    let mut rng = Xorshift64::new(0xC0FFEE_u64);
    let fill = |n: usize, rng: &mut Xorshift64| -> Vec<f32> { (0..n).map(|_| rng.next_weight()).collect() };

    let mut g = GgufBuilder::new();
    let arch = spec.architecture;
    g.kv_string("general.architecture", arch);
    g.kv_u32(&format!("{arch}.block_count"), spec.n_layers as u32);
    g.kv_u32(&format!("{arch}.attention.head_count"), spec.n_heads as u32);
    g.kv_u32(&format!("{arch}.attention.head_count_kv"), spec.n_kv_heads as u32);
    g.kv_u32(&format!("{arch}.embedding_length"), spec.hidden_size as u32);
    g.kv_u32(&format!("{arch}.feed_forward_length"), spec.ffn_hidden_size as u32);
    g.kv_u32(&format!("{arch}.context_length"), spec.context_length as u32);
    g.kv_u32(&format!("{arch}.vocab_size"), spec.vocab_size as u32);
    g.kv_f32(&format!("{arch}.rope.freq_base"), spec.rope_theta);
    g.kv_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"), spec.norm_eps);

    let hidden = spec.hidden_size;
    let kv_dim = spec.n_kv_heads * spec.head_dim();
    let ffn = spec.ffn_hidden_size;
    let vocab = spec.vocab_size;

    g.tensor_f32("token_embd.weight", &[vocab, hidden], &fill(vocab * hidden, &mut rng));

    for layer in 0..spec.n_layers {
        let p = |suffix: &str| format!("blk.{layer}.{suffix}");

        let q = spec.quantize_matmul_weights;
        g.tensor_f32(&p("attn_norm.weight"), &[hidden], &fill(hidden, &mut rng));
        g.matmul_weight(q, &p("attn_q.weight"), &[hidden, hidden], &fill(hidden * hidden, &mut rng));
        g.matmul_weight(q, &p("attn_k.weight"), &[kv_dim, hidden], &fill(kv_dim * hidden, &mut rng));
        g.matmul_weight(q, &p("attn_v.weight"), &[kv_dim, hidden], &fill(kv_dim * hidden, &mut rng));
        if spec.with_qkv_bias {
            g.tensor_f32(&p("attn_q.bias"), &[hidden], &fill(hidden, &mut rng));
            g.tensor_f32(&p("attn_k.bias"), &[kv_dim], &fill(kv_dim, &mut rng));
            g.tensor_f32(&p("attn_v.bias"), &[kv_dim], &fill(kv_dim, &mut rng));
        }
        g.matmul_weight(q, &p("attn_output.weight"), &[hidden, hidden], &fill(hidden * hidden, &mut rng));

        g.tensor_f32(&p("ffn_norm.weight"), &[hidden], &fill(hidden, &mut rng));
        g.matmul_weight(q, &p("ffn_gate.weight"), &[ffn, hidden], &fill(ffn * hidden, &mut rng));
        g.matmul_weight(q, &p("ffn_up.weight"), &[ffn, hidden], &fill(ffn * hidden, &mut rng));
        g.matmul_weight(q, &p("ffn_down.weight"), &[hidden, ffn], &fill(hidden * ffn, &mut rng));
    }

    g.tensor_f32("output_norm.weight", &[hidden], &fill(hidden, &mut rng));
    if !spec.tie_embeddings {
        g.matmul_weight(spec.quantize_matmul_weights, "output.weight", &[vocab, hidden], &fill(vocab * hidden, &mut rng));
    }

    g.build()
}

/// A ready-to-use tiny model: [`SyntheticModelSpec::default`].
pub(crate) fn tiny_model_bytes() -> Vec<u8> {
    build(&SyntheticModelSpec::default())
}

/// Writes `bytes` to a fresh, uniquely-named temp file and returns its
/// path. GGUF loading is inherently file-based (`kopitiam_loader`
/// memory-maps the file), so every test that needs a
/// [`kopitiam_loader::LoadedModel`] needs real bytes on disk, not just an
/// in-memory `Vec<u8>`.
///
/// `disambiguator` becomes part of the filename purely to make a failing
/// test's leftover fixture identifiable by eye; it is *not* what makes the
/// name unique — [`tempfile::Builder`] is (via `mkstemp`-style random
/// suffixing). That distinction matters: `cargo test` runs tests in
/// parallel by default, and several tests in this crate call a shared
/// `load_tiny()`-style helper concurrently, so a filename derived only
/// from a caller-supplied string plus the (single, shared) process id
/// collides across threads. The first version of this helper did exactly
/// that and it was a real, reproduced bug: one thread's
/// `std::fs::write` truncating the file while another thread had it
/// `mmap`-ed produced a `SIGBUS` (`kopitiam_loader::byte_source` maps the
/// file and does not defend against concurrent truncation from *this*
/// process — see that module's docs on why it accepts that risk for
/// external processes; here the "external" writer was our own test suite).
/// `tempfile::Builder::tempfile()` sidesteps the whole problem by
/// guaranteeing a fresh, never-before-used path per call.
pub(crate) fn write_temp_gguf(bytes: &[u8], disambiguator: &str) -> std::path::PathBuf {
    use std::io::Write;
    let mut tmp = tempfile::Builder::new()
        .prefix(&format!("kopitiam-runtime-test-{disambiguator}-"))
        .suffix(".gguf")
        .tempfile()
        .expect("create a uniquely-named temp file");
    tmp.write_all(bytes).expect("write synthetic GGUF fixture");
    tmp.flush().expect("flush synthetic GGUF fixture");
    let (_file, path) = tmp.keep().expect("persist temp GGUF fixture past this function's return");
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_model_bytes_parses_as_valid_gguf_with_expected_shapes() {
        let bytes = tiny_model_bytes();
        let path = write_temp_gguf(&bytes, "self-check");
        let model = kopitiam_loader::load_model(&path).unwrap();
        assert_eq!(model.format(), "gguf");
        let embd = model.tensor("token_embd.weight").unwrap();
        assert_eq!(embd.shape.dims(), &[37, 16]);
        assert!(model.tensor("blk.0.attn_q.weight").is_some());
        assert!(model.tensor("blk.1.ffn_down.weight").is_some());
        // Default spec ties embeddings, so no separate output.weight.
        assert!(model.tensor("output.weight").is_none());
    }

    /// Round-trips a `quantize_matmul_weights: true` fixture through the
    /// *real* `kopitiam_loader` GGUF parser (not a shortcut) and checks
    /// that matmul-operand weights actually landed as `Q8_0` on disk while
    /// the embedding table did not — the same distinction
    /// `crate::weights::ModelWeights`'s docs describe, exercised at the
    /// file-format level this time instead of the loading-code level.
    #[test]
    fn quantized_spec_writes_real_q8_0_tensors_the_loader_recognizes() {
        let spec = SyntheticModelSpec { quantize_matmul_weights: true, ..SyntheticModelSpec::quantized_benchmark() };
        let bytes = build(&spec);
        let path = write_temp_gguf(&bytes, "quantized-self-check");
        let model = kopitiam_loader::load_model(&path).unwrap();

        let wq = model.tensor("blk.0.attn_q.weight").unwrap();
        assert_eq!(wq.dtype, kopitiam_core::DType::Q8_0);
        assert_eq!(wq.shape.dims(), &[spec.hidden_size, spec.hidden_size]);

        let embd = model.tensor("token_embd.weight").unwrap();
        assert_eq!(embd.dtype, kopitiam_core::DType::F32, "embeddings must never be quantized by this fixture");
    }

    #[test]
    fn quantize_q8_0_blocks_round_trips_within_one_quantization_step() {
        let data: Vec<f32> = (0..32).map(|j| (j as f32 - 16.0) * 0.3).collect();
        let bytes = quantize_q8_0_blocks(&data);
        assert_eq!(bytes.len(), 34); // one block: 2-byte f16 scale + 32 signed bytes.

        let d = kopitiam_tensor::f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
        for (j, &orig) in data.iter().enumerate() {
            let decoded = f32::from(bytes[2 + j] as i8) * d;
            assert!((decoded - orig).abs() <= d / 2.0 + 1e-6, "index {j}: {decoded} vs {orig} (d={d})");
        }
    }
}

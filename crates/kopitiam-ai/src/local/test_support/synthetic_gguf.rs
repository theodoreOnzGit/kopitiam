//! Hand-builds a tiny, valid, on-disk GGUF file describing a Qwen-shaped
//! transformer *with a full embedded byte-level tokenizer*, random-but-fixed
//! weights, and no real model weights anywhere.
//!
//! # Why this exists as its own copy, not a reuse of `kopitiam-runtime`'s
//!
//! `kopitiam-runtime` already has a byte-level-identical synthetic GGUF
//! builder at `crates/kopitiam-runtime/src/test_support/synthetic_gguf.rs`
//! (this module mirrors its architecture-metadata and weight-tensor
//! sections field-for-field, against the same wire format documented in
//! `kopitiam_loader::gguf`). It cannot be reused directly: that module is
//! `#[cfg(test)] pub(crate)`, private to `kopitiam-runtime`'s own test
//! binary, and this task's brief is explicit that `kopitiam-runtime` is
//! frozen and out of scope to modify (which would be required to expose a
//! `test-support` feature downstream). Duplicating ~150 lines of
//! GGUF-writing plumbing here is the honest cost of that constraint, not
//! an oversight.
//!
//! What this module adds beyond `kopitiam-runtime`'s fixture is the part
//! `LocalAdapter::load` actually needs and the other fixture has no reason
//! to carry: a complete `tokenizer.ggml.tokens` vocabulary (every one of
//! the 256 base bytes, mapped through the same GPT-2 byte-to-unicode
//! alphabet `kopitiam_tokenizer::byte_map` uses — see
//! `kopitiam_runtime::gguf_tokenizer`'s docs for why that mapping is what a
//! real Qwen2 GGUF export actually stores) plus the three ChatML control
//! tokens (`<|endoftext|>`, `<|im_start|>`, `<|im_end|>`), each marked
//! `tokenizer.ggml.token_type == CONTROL` so `tokenizer_from_gguf`
//! registers them as atomic specials, exactly like a real Qwen2 GGUF file
//! does.

use kopitiam_tokenizer::byte_map::byte_to_unicode;

const ALIGNMENT: usize = 32;

const GGML_TYPE_F32: u32 = 0;
const GGUF_VALUE_TYPE_I32: u32 = 5;
const GGUF_VALUE_TYPE_U32: u32 = 4;
const GGUF_VALUE_TYPE_F32: u32 = 6;
const GGUF_VALUE_TYPE_STRING: u32 = 8;
const GGUF_VALUE_TYPE_ARRAY: u32 = 9;

/// `llama.cpp`'s `LLAMA_TOKEN_TYPE_NORMAL`/`_CONTROL` — see
/// `kopitiam_runtime::gguf_tokenizer`'s `TOKEN_TYPE_CONTROL` doc for where
/// this value comes from.
const TOKEN_TYPE_NORMAL: i32 = 1;
const TOKEN_TYPE_CONTROL: i32 = 3;

const CONTROL_TOKENS: [&str; 3] = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"];

fn push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn push_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn push_i32(buf: &mut Vec<u8>, v: i32) {
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

/// A deterministic, dependency-free PRNG (xorshift64*), used only to fill
/// synthetic weight tensors with small, reproducible, non-degenerate
/// values — mirrors `kopitiam-runtime`'s own fixture; see that module's
/// docs on why a real `rand` dependency is not justified for a test-only
/// fixture like this one.
struct Xorshift64(u64);

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// A pseudo-random `f32` in `[-0.1, 0.1]` — small enough that the
    /// forward pass's `f32` accumulations stay well away from overflow
    /// across every op the chain calls into.
    fn next_weight(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        let unit = (self.0 >> 40) as u32 as f32 / (1u32 << 24) as f32; // in [0, 1)
        (unit - 0.5) * 0.2
    }
}

/// Accumulates metadata KVs, tensor info records, and tensor data
/// separately (matching GGUF's three-section layout) and assembles them
/// into one file in [`GgufBuilder::build`].
struct GgufBuilder {
    kv_count: u64,
    kvs: Vec<u8>,
    tensor_count: u64,
    tensor_infos: Vec<u8>,
    tensor_data: Vec<u8>,
}

impl GgufBuilder {
    fn new() -> Self {
        Self { kv_count: 0, kvs: Vec::new(), tensor_count: 0, tensor_infos: Vec::new(), tensor_data: Vec::new() }
    }

    fn kv_string(&mut self, key: &str, value: &str) {
        push_string(&mut self.kvs, key);
        push_u32(&mut self.kvs, GGUF_VALUE_TYPE_STRING);
        push_string(&mut self.kvs, value);
        self.kv_count += 1;
    }

    fn kv_u32(&mut self, key: &str, value: u32) {
        push_string(&mut self.kvs, key);
        push_u32(&mut self.kvs, GGUF_VALUE_TYPE_U32);
        push_u32(&mut self.kvs, value);
        self.kv_count += 1;
    }

    fn kv_f32(&mut self, key: &str, value: f32) {
        push_string(&mut self.kvs, key);
        push_u32(&mut self.kvs, GGUF_VALUE_TYPE_F32);
        self.kvs.extend_from_slice(&value.to_le_bytes());
        self.kv_count += 1;
    }

    /// A GGUF array-of-string KV: `key -> [elem_type=STRING, len, values...]`.
    fn kv_string_array(&mut self, key: &str, values: &[String]) {
        push_string(&mut self.kvs, key);
        push_u32(&mut self.kvs, GGUF_VALUE_TYPE_ARRAY);
        push_u32(&mut self.kvs, GGUF_VALUE_TYPE_STRING);
        push_u64(&mut self.kvs, values.len() as u64);
        for v in values {
            push_string(&mut self.kvs, v);
        }
        self.kv_count += 1;
    }

    /// A GGUF array-of-i32 KV: `key -> [elem_type=I32, len, values...]`.
    fn kv_i32_array(&mut self, key: &str, values: &[i32]) {
        push_string(&mut self.kvs, key);
        push_u32(&mut self.kvs, GGUF_VALUE_TYPE_ARRAY);
        push_u32(&mut self.kvs, GGUF_VALUE_TYPE_I32);
        push_u64(&mut self.kvs, values.len() as u64);
        for &v in values {
            push_i32(&mut self.kvs, v);
        }
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

/// The full byte-level vocabulary a real Qwen2 GGUF export carries: the
/// 256 base bytes (mapped through the GPT-2 byte-to-unicode alphabet, id
/// == byte value) followed by the three ChatML control tokens at ids
/// 256/257/258. [`build_local_adapter_fixture`]'s model is sized to this
/// exact vocab size (259).
fn vocab_size() -> usize {
    256 + CONTROL_TOKENS.len()
}

/// Builds the raw bytes of a synthetic GGUF file: a tiny (2-layer,
/// GQA-shaped, tied-embedding) Qwen2 model plus a complete embedded
/// byte-level tokenizer with ChatML's three control tokens — everything
/// [`crate::local::LocalAdapter::load`] needs, with no real model weights
/// anywhere. See this module's docs for why it duplicates (rather than
/// reuses) `kopitiam-runtime`'s own synthetic-GGUF fixture.
pub(crate) fn build_local_adapter_fixture() -> Vec<u8> {
    let n_layers = 2usize;
    let n_heads = 4usize;
    let n_kv_heads = 2usize;
    let hidden = 16usize;
    let head_dim = hidden / n_heads;
    let ffn = 32usize;
    let vocab = vocab_size();
    let context_length = 64u32;

    let mut rng = Xorshift64::new(0xC0FFEE_u64);
    let fill = |n: usize, rng: &mut Xorshift64| -> Vec<f32> { (0..n).map(|_| rng.next_weight()).collect() };

    let mut g = GgufBuilder::new();

    // --- Architecture metadata -- everything QwenConfig::from_metadata needs.
    let arch = "qwen2";
    g.kv_string("general.architecture", arch);
    g.kv_string("general.name", "kopitiam-test-qwen");
    g.kv_u32(&format!("{arch}.block_count"), n_layers as u32);
    g.kv_u32(&format!("{arch}.attention.head_count"), n_heads as u32);
    g.kv_u32(&format!("{arch}.attention.head_count_kv"), n_kv_heads as u32);
    g.kv_u32(&format!("{arch}.embedding_length"), hidden as u32);
    g.kv_u32(&format!("{arch}.feed_forward_length"), ffn as u32);
    g.kv_u32(&format!("{arch}.context_length"), context_length);
    g.kv_f32(&format!("{arch}.rope.freq_base"), 10_000.0);
    g.kv_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"), 1e-6);

    // --- Embedded tokenizer: full byte-level vocab + ChatML control tokens.
    let mut tokens: Vec<String> = (0u16..=255).map(|b| byte_to_unicode(b as u8).to_string()).collect();
    for &special in &CONTROL_TOKENS {
        tokens.push(special.to_string());
    }
    g.kv_string_array("tokenizer.ggml.tokens", &tokens);
    g.kv_string_array("tokenizer.ggml.merges", &[]); // no merges: pure byte-level tokens suffice for this fixture.
    let mut token_types = vec![TOKEN_TYPE_NORMAL; 256];
    token_types.extend(std::iter::repeat_n(TOKEN_TYPE_CONTROL, CONTROL_TOKENS.len()));
    g.kv_i32_array("tokenizer.ggml.token_type", &token_types);

    // --- Weights: token_embd, per-layer attention/MLP, output_norm.
    // tie_embeddings = true (no separate output.weight -- see
    // ModelWeights::output_weight's docs), with_qkv_bias = true (Qwen2's
    // real configuration).
    g.tensor_f32("token_embd.weight", &[vocab, hidden], &fill(vocab * hidden, &mut rng));

    let kv_dim = n_kv_heads * head_dim;
    for layer in 0..n_layers {
        let p = |suffix: &str| format!("blk.{layer}.{suffix}");

        g.tensor_f32(&p("attn_norm.weight"), &[hidden], &fill(hidden, &mut rng));
        g.tensor_f32(&p("attn_q.weight"), &[hidden, hidden], &fill(hidden * hidden, &mut rng));
        g.tensor_f32(&p("attn_q.bias"), &[hidden], &fill(hidden, &mut rng));
        g.tensor_f32(&p("attn_k.weight"), &[kv_dim, hidden], &fill(kv_dim * hidden, &mut rng));
        g.tensor_f32(&p("attn_k.bias"), &[kv_dim], &fill(kv_dim, &mut rng));
        g.tensor_f32(&p("attn_v.weight"), &[kv_dim, hidden], &fill(kv_dim * hidden, &mut rng));
        g.tensor_f32(&p("attn_v.bias"), &[kv_dim], &fill(kv_dim, &mut rng));
        g.tensor_f32(&p("attn_output.weight"), &[hidden, hidden], &fill(hidden * hidden, &mut rng));

        g.tensor_f32(&p("ffn_norm.weight"), &[hidden], &fill(hidden, &mut rng));
        g.tensor_f32(&p("ffn_gate.weight"), &[ffn, hidden], &fill(ffn * hidden, &mut rng));
        g.tensor_f32(&p("ffn_up.weight"), &[ffn, hidden], &fill(ffn * hidden, &mut rng));
        g.tensor_f32(&p("ffn_down.weight"), &[hidden, ffn], &fill(hidden * ffn, &mut rng));
    }

    g.tensor_f32("output_norm.weight", &[hidden], &fill(hidden, &mut rng));
    // tied embeddings: no separate output.weight tensor.

    g.build()
}

/// Writes `bytes` to a fresh, uniquely-named temp file and returns its
/// path. GGUF loading is inherently file-based (`kopitiam_loader`
/// memory-maps the file), so a test needing a real load needs real bytes
/// on disk. `disambiguator` is only for making a leftover fixture
/// identifiable by eye; uniqueness itself comes from
/// [`tempfile::Builder`] (mirrors `kopitiam-runtime`'s own fixture helper
/// and its documented reason: `cargo test`'s default parallelism means a
/// filename derived only from a caller string collides across threads).
pub(crate) fn write_temp_gguf(bytes: &[u8], disambiguator: &str) -> std::path::PathBuf {
    use std::io::Write;
    let mut tmp = tempfile::Builder::new()
        .prefix(&format!("kopitiam-ai-test-{disambiguator}-"))
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
    use kopitiam_tokenizer::Tokenizer as _;

    #[test]
    fn fixture_parses_as_valid_gguf_with_expected_shapes_and_vocab() {
        let bytes = build_local_adapter_fixture();
        let path = write_temp_gguf(&bytes, "self-check");
        let model = kopitiam_loader::load_model(&path).unwrap();
        assert_eq!(model.format(), "gguf");

        let embd = model.tensor("token_embd.weight").unwrap();
        assert_eq!(embd.shape.dims(), &[vocab_size(), 16]);
        assert!(model.tensor("blk.0.attn_q.weight").is_some());
        assert!(model.tensor("blk.1.ffn_down.weight").is_some());
        assert!(model.tensor("output.weight").is_none(), "fixture ties embeddings");

        assert_eq!(model.metadata().vocab_size, Some(vocab_size() as u64));
        assert_eq!(model.metadata().name.as_deref(), Some("kopitiam-test-qwen"));
    }

    #[test]
    fn fixture_tokenizer_round_trips_and_exposes_chatml_control_tokens() {
        let bytes = build_local_adapter_fixture();
        let path = write_temp_gguf(&bytes, "tokenizer-self-check");
        let model = kopitiam_loader::load_model(&path).unwrap();
        let tokenizer = kopitiam_runtime::tokenizer_from_gguf(&model).unwrap();

        assert_eq!(tokenizer.special_token_id("<|im_start|>"), Some(257));
        assert_eq!(tokenizer.special_token_id("<|im_end|>"), Some(258));
        assert_eq!(tokenizer.special_token_id("<|endoftext|>"), Some(256));

        let text = "hello world";
        let ids = tokenizer.encode(text).unwrap();
        assert_eq!(tokenizer.decode(&ids).unwrap(), text);
    }
}

//! End-to-end tests for the GGUF loader: a byte-exact minimal valid file
//! proves the parser round-trips metadata, dtypes, shapes and tensor bytes
//! correctly; the rest hand-craft malformed variants and assert every one
//! fails gracefully (`Error`, never a panic).

mod common;

use common::*;
use kopitiam_core::{DType, Error};
use kopitiam_loader::{GgufLoader, ModelLoader, load_model};

/// Builds a small-but-representative GGUF file: enough metadata keys to
/// exercise every [`kopitiam_loader::ModelMetadata`] field this loader
/// promotes, plus two tensors — one plain `f32` and one block-quantized
/// `q8_0` — to prove both the unquantized and quantized paths round-trip.
///
/// Returns the file bytes alongside the exact tensor bytes expected back
/// out, so the test asserts byte-for-byte equality rather than "looks
/// about right".
fn build_minimal_valid_gguf() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut kvs = Vec::new();
    let mut kv_count = 0u64;
    macro_rules! kv {
        ($push:expr) => {{
            $push;
            kv_count += 1;
        }};
    }
    kv!(push_kv_string(&mut kvs, "general.architecture", "llama"));
    kv!(push_kv_string(&mut kvs, "general.name", "test-model"));
    kv!(push_kv_u32(&mut kvs, "llama.block_count", 2));
    kv!(push_kv_u32(&mut kvs, "llama.context_length", 128));
    kv!(push_kv_u32(&mut kvs, "llama.embedding_length", 8));
    kv!(push_kv_u32(&mut kvs, "llama.feed_forward_length", 32));
    kv!(push_kv_u32(&mut kvs, "llama.attention.head_count", 2));
    kv!(push_kv_u32(&mut kvs, "llama.attention.head_count_kv", 1));
    kv!(push_kv_f32(&mut kvs, "llama.rope.freq_base", 10000.0));
    kv!(push_kv_u32(&mut kvs, "llama.rope.dimension_count", 4));
    kv!(push_kv_f32(&mut kvs, "llama.attention.layer_norm_rms_epsilon", 1e-5));
    kv!(push_kv_u32(&mut kvs, "general.quantization_version", 2));
    kv!(push_kv_u32(&mut kvs, "general.file_type", 7));
    kv!(push_kv_string_array(&mut kvs, "tokenizer.ggml.tokens", &["<unk>", "a", "b", "c"]));

    let mut tensor_infos = Vec::new();
    // ne = [4, 2] is ggml's fastest-varying-first order, so the logical
    // (kopitiam_core::Shape) shape is the reverse: [2, 4].
    push_tensor_info(&mut tensor_infos, "weight.a", &[4, 2], GGML_TYPE_F32, 0);
    // ne = [32]: one whole Q8_0 block (block_size = 32 elements).
    push_tensor_info(&mut tensor_infos, "weight.b", &[32], GGML_TYPE_Q8_0, 32);

    let mut buf = Vec::new();
    push_header(&mut buf, 3, 2, kv_count);
    buf.extend_from_slice(&kvs);
    buf.extend_from_slice(&tensor_infos);
    pad_to_alignment(&mut buf, 32);

    // weight.a: 8 f32 elements (2*4), values 0.0..=7.0 -> 32 bytes.
    let a_bytes: Vec<u8> = (0..8u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    assert_eq!(a_bytes.len(), 32);
    buf.extend_from_slice(&a_bytes);

    // weight.b: one Q8_0 block = 34 bytes, arbitrary-but-deterministic content.
    let b_bytes: Vec<u8> = (0..34u8).collect();
    assert_eq!(b_bytes.len(), 34);
    buf.extend_from_slice(&b_bytes);

    (buf, a_bytes, b_bytes)
}

#[test]
fn loads_a_minimal_valid_gguf_file_end_to_end() {
    let (bytes, expected_a, expected_b) = build_minimal_valid_gguf();
    let dir = tempfile::tempdir().unwrap();
    let path = write_temp_file(&dir, "model.gguf", &bytes);

    let model = load_model(&path).expect("well-formed fixture must load");
    assert_eq!(model.format(), "gguf");
    assert_eq!(model.tensor_count(), 2);
    assert_eq!(model.tensor_names().collect::<Vec<_>>(), ["weight.a", "weight.b"]);

    let meta = model.metadata();
    assert_eq!(meta.architecture.as_deref(), Some("llama"));
    assert_eq!(meta.name.as_deref(), Some("test-model"));
    assert_eq!(meta.n_layers, Some(2));
    assert_eq!(meta.context_length, Some(128));
    assert_eq!(meta.embedding_length, Some(8));
    assert_eq!(meta.feed_forward_length, Some(32));
    assert_eq!(meta.n_heads, Some(2));
    assert_eq!(meta.n_kv_heads, Some(1));
    assert_eq!(meta.rope_theta, Some(10000.0));
    assert_eq!(meta.rope_dimension_count, Some(4));
    assert_eq!(meta.norm_epsilon, Some(1e-5));
    assert_eq!(meta.quantization_version, Some(2));
    assert_eq!(meta.file_type, Some(7));
    assert_eq!(meta.vocab_size, Some(4));
    assert_eq!(meta.raw.get_str("general.architecture"), Some("llama"));

    let a = model.tensor("weight.a").expect("weight.a present");
    assert_eq!(a.dtype, DType::F32);
    assert_eq!(a.shape.dims(), &[2, 4], "ggml ne=[4,2] must reverse to Shape [2,4]");
    assert_eq!(model.tensor_bytes("weight.a").unwrap(), expected_a.as_slice());

    let b = model.tensor("weight.b").expect("weight.b present");
    assert_eq!(b.dtype, DType::Q8_0);
    assert_eq!(b.shape.dims(), &[32]);
    assert_eq!(model.tensor_bytes("weight.b").unwrap(), expected_b.as_slice());

    assert!(model.tensor("does.not.exist").is_none());
    assert!(matches!(
        model.tensor_bytes("does.not.exist"),
        Err(Error::MissingTensor { .. })
    ));
}

#[test]
fn load_model_sniffs_gguf_by_magic_regardless_of_extension() {
    let (bytes, _, _) = build_minimal_valid_gguf();
    let dir = tempfile::tempdir().unwrap();
    // Deliberately wrong extension: dispatch must not rely on it.
    let path = write_temp_file(&dir, "model.bin", &bytes);
    let model = load_model(&path).unwrap();
    assert_eq!(model.format(), "gguf");
}

#[test]
fn gguf_loader_probe_matches_the_magic_only() {
    let loader = GgufLoader;
    assert!(loader.probe(b"GGUF and then more bytes"));
    assert!(!loader.probe(b"NOPE"));
    assert!(!loader.probe(b"GG"));
    assert!(!loader.probe(b""));
}

/// A truncated file must fail gracefully at every truncation point tried,
/// never panic. This sweeps a range of cut points across the header,
/// metadata and tensor-info sections of the valid fixture.
#[test]
fn truncated_files_fail_gracefully_at_every_cut_point() {
    let (bytes, _, _) = build_minimal_valid_gguf();
    let dir = tempfile::tempdir().unwrap();

    for cut in [0, 1, 3, 4, 8, 16, 24, 25, 40, 100, bytes.len() / 2, bytes.len() - 1] {
        let truncated = &bytes[..cut];
        let path = write_temp_file(&dir, "truncated.gguf", truncated);
        let result = load_model(&path);
        assert!(result.is_err(), "cut at {cut} bytes should not load successfully");
    }
}

#[test]
fn bad_magic_is_rejected() {
    let (mut bytes, _, _) = build_minimal_valid_gguf();
    bytes[0] = b'X';
    let dir = tempfile::tempdir().unwrap();
    let path = write_temp_file(&dir, "bad_magic.gguf", &bytes);
    let err = GgufLoader.load(&path).unwrap_err();
    assert!(matches!(err, Error::MalformedModel { format: "gguf", .. }), "got {err:?}");
}

#[test]
fn unsupported_version_is_rejected_without_panicking() {
    let mut buf = Vec::new();
    push_header(&mut buf, 1, 0, 0); // v1 is explicitly unsupported (see gguf module docs)
    let dir = tempfile::tempdir().unwrap();
    let path = write_temp_file(&dir, "v1.gguf", &buf);
    let err = GgufLoader.load(&path).unwrap_err();
    assert!(matches!(err, Error::UnsupportedModelFeature { format: "gguf", .. }), "got {err:?}");

    let mut buf2 = Vec::new();
    push_header(&mut buf2, 99, 0, 0);
    let path2 = write_temp_file(&dir, "v99.gguf", &buf2);
    let err2 = GgufLoader.load(&path2).unwrap_err();
    assert!(matches!(err2, Error::UnsupportedModelFeature { format: "gguf", .. }), "got {err2:?}");
}

#[test]
fn an_absurd_metadata_kv_count_fails_fast_instead_of_hanging_or_oom() {
    let mut buf = Vec::new();
    // Header claims ~u64::MAX metadata entries but the file has none of
    // that data; the very first (missing) key read must fail immediately.
    push_header(&mut buf, 3, 0, u64::MAX);
    let dir = tempfile::tempdir().unwrap();
    let path = write_temp_file(&dir, "huge_kv_count.gguf", &buf);
    let err = GgufLoader.load(&path).unwrap_err();
    assert!(matches!(err, Error::MalformedModel { format: "gguf", .. }), "got {err:?}");
}

#[test]
fn an_absurd_tensor_count_fails_fast_instead_of_hanging_or_oom() {
    let mut buf = Vec::new();
    push_header(&mut buf, 3, u64::MAX, 0);
    let dir = tempfile::tempdir().unwrap();
    let path = write_temp_file(&dir, "huge_tensor_count.gguf", &buf);
    let err = GgufLoader.load(&path).unwrap_err();
    assert!(matches!(err, Error::MalformedModel { format: "gguf", .. }), "got {err:?}");
}

#[test]
fn a_tensor_offset_past_end_of_file_is_rejected() {
    let mut buf = Vec::new();
    push_header(&mut buf, 3, 1, 0);
    // Offset far beyond anything the (empty) tensor data section could hold.
    push_tensor_info(&mut buf, "w", &[4], GGML_TYPE_F32, 1_000_000);
    pad_to_alignment(&mut buf, 32);
    // No tensor data appended at all.
    let dir = tempfile::tempdir().unwrap();
    let path = write_temp_file(&dir, "oob_offset.gguf", &buf);
    let err = GgufLoader.load(&path).unwrap_err();
    assert!(matches!(err, Error::MalformedModel { format: "gguf", .. }), "got {err:?}");
}

#[test]
fn a_misaligned_tensor_offset_is_rejected() {
    let mut buf = Vec::new();
    push_header(&mut buf, 3, 1, 0);
    // offset=4 is not a multiple of the default alignment (32).
    push_tensor_info(&mut buf, "w", &[4], GGML_TYPE_F32, 4);
    pad_to_alignment(&mut buf, 32);
    buf.extend_from_slice(&[0u8; 32]);
    let dir = tempfile::tempdir().unwrap();
    let path = write_temp_file(&dir, "misaligned.gguf", &buf);
    let err = GgufLoader.load(&path).unwrap_err();
    assert!(matches!(err, Error::MalformedModel { format: "gguf", .. }), "got {err:?}");
}

#[test]
fn an_unsupported_ggml_tensor_type_is_reported_not_misdecoded() {
    let mut buf = Vec::new();
    push_header(&mut buf, 3, 1, 0);
    push_tensor_info(&mut buf, "w", &[1], GGML_TYPE_Q2_K, 0);
    pad_to_alignment(&mut buf, 32);
    let dir = tempfile::tempdir().unwrap();
    let path = write_temp_file(&dir, "unsupported_type.gguf", &buf);
    let err = GgufLoader.load(&path).unwrap_err();
    assert!(
        matches!(err, Error::UnsupportedModelFeature { format: "gguf", .. }),
        "got {err:?}"
    );
}

#[test]
fn a_partial_quantized_block_is_rejected() {
    let mut buf = Vec::new();
    push_header(&mut buf, 3, 1, 0);
    // Q8_0 packs 32 elements per block; 5 elements is not a whole block.
    push_tensor_info(&mut buf, "w", &[5], GGML_TYPE_Q8_0, 0);
    pad_to_alignment(&mut buf, 32);
    buf.extend_from_slice(&[0u8; 34]);
    let dir = tempfile::tempdir().unwrap();
    let path = write_temp_file(&dir, "partial_block.gguf", &buf);
    let err = GgufLoader.load(&path).unwrap_err();
    assert!(matches!(err, Error::PartialQuantizedBlock { .. }), "got {err:?}");
}

#[test]
fn a_duplicate_metadata_key_is_rejected() {
    let mut kvs = Vec::new();
    push_kv_string(&mut kvs, "general.architecture", "llama");
    push_kv_string(&mut kvs, "general.architecture", "gpt2");
    let mut buf = Vec::new();
    push_header(&mut buf, 3, 0, 2);
    buf.extend_from_slice(&kvs);
    let dir = tempfile::tempdir().unwrap();
    let path = write_temp_file(&dir, "dup_key.gguf", &buf);
    let err = GgufLoader.load(&path).unwrap_err();
    assert!(matches!(err, Error::MalformedModel { format: "gguf", .. }), "got {err:?}");
}

#[test]
fn a_duplicate_tensor_name_is_rejected() {
    let mut buf = Vec::new();
    push_header(&mut buf, 3, 2, 0);
    push_tensor_info(&mut buf, "w", &[4], GGML_TYPE_F32, 0);
    push_tensor_info(&mut buf, "w", &[4], GGML_TYPE_F32, 32);
    pad_to_alignment(&mut buf, 32);
    buf.extend_from_slice(&[0u8; 64]);
    let dir = tempfile::tempdir().unwrap();
    let path = write_temp_file(&dir, "dup_tensor.gguf", &buf);
    let err = GgufLoader.load(&path).unwrap_err();
    assert!(matches!(err, Error::MalformedModel { format: "gguf", .. }), "got {err:?}");
}

#[test]
fn an_invalid_bool_byte_is_rejected() {
    let mut kvs = Vec::new();
    push_kv_bool_byte(&mut kvs, "some.flag", 42); // only 0x00/0x01 are valid
    let mut buf = Vec::new();
    push_header(&mut buf, 3, 0, 1);
    buf.extend_from_slice(&kvs);
    let dir = tempfile::tempdir().unwrap();
    let path = write_temp_file(&dir, "bad_bool.gguf", &buf);
    let err = GgufLoader.load(&path).unwrap_err();
    assert!(matches!(err, Error::MalformedModel { format: "gguf", .. }), "got {err:?}");
}

#[test]
fn a_zero_alignment_is_rejected() {
    let mut kvs = Vec::new();
    push_kv_u32(&mut kvs, "general.alignment", 0);
    let mut buf = Vec::new();
    push_header(&mut buf, 3, 0, 1);
    buf.extend_from_slice(&kvs);
    let dir = tempfile::tempdir().unwrap();
    let path = write_temp_file(&dir, "zero_alignment.gguf", &buf);
    let err = GgufLoader.load(&path).unwrap_err();
    assert!(matches!(err, Error::MalformedModel { format: "gguf", .. }), "got {err:?}");
}

#[test]
fn an_empty_file_fails_gracefully() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_temp_file(&dir, "empty.gguf", &[]);
    let err = GgufLoader.load(&path).unwrap_err();
    assert!(matches!(err, Error::MalformedModel { format: "gguf", .. }), "got {err:?}");
}

/// Deeply nested one-element arrays are a cheap way to try to blow the
/// parser's call stack (~12 bytes of header per nesting level). This must
/// fail as a graceful error well before the configured nesting cap, not
/// crash the process.
#[test]
fn deeply_nested_arrays_are_rejected_rather_than_overflowing_the_stack() {
    let mut kvs = Vec::new();
    push_string(&mut kvs, "evil.nested");
    // 10_000 levels of `Array(elem_type=ARRAY, len=1)`, then nothing —
    // the innermost level is simply missing its element, which is also a
    // fine way for this to fail if the nesting cap doesn't trigger first.
    for _ in 0..10_000 {
        push_u32(&mut kvs, TYPE_ARRAY);
        push_u32(&mut kvs, TYPE_ARRAY); // element type of this array: another array
        push_u64(&mut kvs, 1); // length 1
    }
    let mut buf = Vec::new();
    push_header(&mut buf, 3, 0, 1);
    buf.extend_from_slice(&kvs);
    let dir = tempfile::tempdir().unwrap();
    let path = write_temp_file(&dir, "nested_arrays.gguf", &buf);
    // The important assertion is simply "did not crash the test process";
    // `load` returning any Err here is success.
    let err = GgufLoader.load(&path).unwrap_err();
    assert!(matches!(err, Error::MalformedModel { format: "gguf", .. }), "got {err:?}");
}

/// If a real GGUF file happens to be present on this machine (this repo
/// vendors `llama.cpp`'s tiny vocab-only test fixtures for tokenizer
/// testing), load it end-to-end and cross-check a handful of facts against
/// values read out-of-band with an independent script. `#[ignore]`d: this
/// is a bonus integration check, not part of the crate's guaranteed test
/// surface, since it depends on a vendored file rather than a hand-built
/// fixture.
#[test]
#[ignore = "depends on a real vendored GGUF file being present on disk"]
fn loads_a_real_vendored_gguf_vocab_file() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../kopitiam-ai/vendor/llama.cpp/models/ggml-vocab-gpt-2.gguf"
    );
    let model = load_model(path).expect("real vendored GGUF vocab file should load");

    assert_eq!(model.format(), "gguf");
    // This particular fixture carries only tokenizer metadata, no tensors.
    assert_eq!(model.tensor_count(), 0);

    let meta = model.metadata();
    assert_eq!(meta.architecture.as_deref(), Some("gpt2"));
    assert_eq!(meta.n_layers, Some(12));
    assert_eq!(meta.context_length, Some(1024));
    assert_eq!(meta.embedding_length, Some(8 * 96)); // 768
    assert_eq!(meta.feed_forward_length, Some(3072));
    assert_eq!(meta.n_heads, Some(12));
    assert_eq!(meta.vocab_size, Some(50257));
    // gpt2 has no RMS-norm epsilon key, only the plain one — exercises the
    // fallback in `gguf::build_metadata`.
    assert!(meta.norm_epsilon.is_some());
}

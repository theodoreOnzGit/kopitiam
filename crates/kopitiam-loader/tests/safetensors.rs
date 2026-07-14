//! End-to-end tests for the SafeTensors loader: a byte-exact minimal valid
//! file proves the parser round-trips metadata, dtypes, shapes and tensor
//! bytes correctly; the rest hand-craft malformed variants and assert
//! every one fails gracefully.

use kopitiam_core::{DType, Error};
use kopitiam_loader::{ModelLoader, SafeTensorsLoader, load_model};
use serde_json::json;

fn write_temp_file(dir: &tempfile::TempDir, name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path = dir.path().join(name);
    std::fs::write(&path, bytes).expect("write temp fixture file");
    path
}

/// Assembles a SafeTensors file from a header `serde_json::Value` and raw
/// tensor bytes: an 8-byte little-endian header length, the header JSON
/// itself, then the bytes verbatim.
fn assemble(header: &serde_json::Value, data: &[u8]) -> Vec<u8> {
    let header_bytes = serde_json::to_vec(header).unwrap();
    let mut buf = Vec::with_capacity(8 + header_bytes.len() + data.len());
    buf.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(&header_bytes);
    buf.extend_from_slice(data);
    buf
}

/// Two tensors — `weight` (`F32`, shape `[2, 3]`) and `bias` (`I32`, shape
/// `[3]`) — plus a free-form `__metadata__` entry, so the fixture exercises
/// two different dtypes and the metadata escape hatch in one file.
fn build_minimal_valid_safetensors() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let weight_vals: Vec<f32> = (0..6).map(|i| i as f32 * 1.5).collect();
    let weight_bytes: Vec<u8> = weight_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(weight_bytes.len(), 24);

    let bias_vals: [i32; 3] = [-1, 0, 42];
    let bias_bytes: Vec<u8> = bias_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(bias_bytes.len(), 12);

    let header = json!({
        "weight": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},
        "bias": {"dtype": "I32", "shape": [3], "data_offsets": [24, 36]},
        "__metadata__": {"note": "test fixture", "producer": "kopitiam-loader tests"},
    });

    let mut data = weight_bytes.clone();
    data.extend_from_slice(&bias_bytes);
    let bytes = assemble(&header, &data);
    (bytes, weight_bytes, bias_bytes)
}

#[test]
fn loads_a_minimal_valid_safetensors_file_end_to_end() {
    let (bytes, expected_weight, expected_bias) = build_minimal_valid_safetensors();
    let dir = tempfile::tempdir().unwrap();
    let path = write_temp_file(&dir, "model.safetensors", &bytes);

    let model = load_model(&path).expect("well-formed fixture must load");
    assert_eq!(model.format(), "safetensors");
    assert_eq!(model.tensor_count(), 2);

    let weight = model.tensor("weight").expect("weight present");
    assert_eq!(weight.dtype, DType::F32);
    assert_eq!(weight.shape.dims(), &[2, 3], "safetensors shape needs no reversal");
    assert_eq!(model.tensor_bytes("weight").unwrap(), expected_weight.as_slice());

    let bias = model.tensor("bias").expect("bias present");
    assert_eq!(bias.dtype, DType::I32);
    assert_eq!(bias.shape.dims(), &[3]);
    assert_eq!(model.tensor_bytes("bias").unwrap(), expected_bias.as_slice());

    let meta = model.metadata();
    assert_eq!(meta.raw.get_str("note"), Some("test fixture"));
    assert_eq!(meta.raw.get_str("producer"), Some("kopitiam-loader tests"));
    // SafeTensors carries no standardized hyperparameter keys.
    assert_eq!(meta.architecture, None);

    assert!(matches!(
        model.tensor_bytes("nonexistent"),
        Err(Error::MissingTensor { .. })
    ));
}

#[test]
fn load_model_falls_back_to_safetensors_by_content_regardless_of_extension() {
    let (bytes, _, _) = build_minimal_valid_safetensors();
    let dir = tempfile::tempdir().unwrap();
    let path = write_temp_file(&dir, "model.bin", &bytes);
    let model = load_model(&path).unwrap();
    assert_eq!(model.format(), "safetensors");
}

#[test]
fn safetensors_loader_probe_heuristic() {
    let loader = SafeTensorsLoader;
    let (bytes, _, _) = build_minimal_valid_safetensors();
    assert!(loader.probe(&bytes));
    assert!(!loader.probe(b"GGUF-magic-here-not-json"));
    assert!(!loader.probe(&[0u8; 4])); // too short even for the length prefix
}

#[test]
fn a_tensor_with_a_dtype_kopitiam_core_cannot_represent_is_reported_not_misdecoded() {
    for dtype in ["I64", "U8", "U16", "U32", "BOOL", "F64", "F8_E4M3"] {
        let header = json!({
            "t": {"dtype": dtype, "shape": [1], "data_offsets": [0, 8]},
        });
        let bytes = assemble(&header, &[0u8; 8]);
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp_file(&dir, "unsupported_dtype.safetensors", &bytes);
        let err = SafeTensorsLoader.load(&path).unwrap_err();
        assert!(
            matches!(err, Error::UnsupportedModelFeature { format: "safetensors", .. }),
            "dtype {dtype}: got {err:?}"
        );
    }
}

#[test]
fn header_length_past_end_of_file_is_rejected() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&1_000_000u64.to_le_bytes());
    buf.extend_from_slice(b"{}"); // far short of the claimed 1,000,000-byte header
    let dir = tempfile::tempdir().unwrap();
    let path = write_temp_file(&dir, "huge_header.safetensors", &buf);
    let err = SafeTensorsLoader.load(&path).unwrap_err();
    assert!(matches!(err, Error::MalformedModel { format: "safetensors", .. }), "got {err:?}");
}

#[test]
fn invalid_json_header_is_rejected() {
    let mut buf = Vec::new();
    let junk = b"not valid json {{{";
    buf.extend_from_slice(&(junk.len() as u64).to_le_bytes());
    buf.extend_from_slice(junk);
    let dir = tempfile::tempdir().unwrap();
    let path = write_temp_file(&dir, "bad_json.safetensors", &buf);
    let err = SafeTensorsLoader.load(&path).unwrap_err();
    assert!(matches!(err, Error::MalformedModel { format: "safetensors", .. }), "got {err:?}");
}

#[test]
fn data_offsets_past_end_of_file_are_rejected() {
    let header = json!({
        "t": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
    });
    // Only 4 bytes of actual tensor data follow, not the declared 16.
    let bytes = assemble(&header, &[0u8; 4]);
    let dir = tempfile::tempdir().unwrap();
    let path = write_temp_file(&dir, "oob_offsets.safetensors", &bytes);
    let err = SafeTensorsLoader.load(&path).unwrap_err();
    assert!(matches!(err, Error::MalformedModel { format: "safetensors", .. }), "got {err:?}");
}

#[test]
fn data_offsets_length_mismatched_with_dtype_and_shape_is_rejected() {
    let header = json!({
        // F32 x 4 elements needs 16 bytes; this declares only 12.
        "t": {"dtype": "F32", "shape": [4], "data_offsets": [0, 12]},
    });
    let bytes = assemble(&header, &[0u8; 12]);
    let dir = tempfile::tempdir().unwrap();
    let path = write_temp_file(&dir, "mismatched_len.safetensors", &bytes);
    let err = SafeTensorsLoader.load(&path).unwrap_err();
    assert!(matches!(err, Error::MalformedModel { format: "safetensors", .. }), "got {err:?}");
}

#[test]
fn data_offsets_with_start_after_end_are_rejected() {
    let header = json!({
        "t": {"dtype": "F32", "shape": [1], "data_offsets": [8, 4]},
    });
    let bytes = assemble(&header, &[0u8; 8]);
    let dir = tempfile::tempdir().unwrap();
    let path = write_temp_file(&dir, "backwards_offsets.safetensors", &bytes);
    let err = SafeTensorsLoader.load(&path).unwrap_err();
    assert!(matches!(err, Error::MalformedModel { format: "safetensors", .. }), "got {err:?}");
}

#[test]
fn metadata_with_a_non_string_value_is_rejected() {
    let header = json!({
        "__metadata__": {"note": 42},
    });
    let bytes = assemble(&header, &[]);
    let dir = tempfile::tempdir().unwrap();
    let path = write_temp_file(&dir, "bad_metadata.safetensors", &bytes);
    let err = SafeTensorsLoader.load(&path).unwrap_err();
    assert!(matches!(err, Error::MalformedModel { format: "safetensors", .. }), "got {err:?}");
}

#[test]
fn a_tensor_entry_missing_required_fields_is_rejected() {
    let header = json!({
        "t": {"dtype": "F32"}, // no shape, no data_offsets
    });
    let bytes = assemble(&header, &[]);
    let dir = tempfile::tempdir().unwrap();
    let path = write_temp_file(&dir, "missing_fields.safetensors", &bytes);
    let err = SafeTensorsLoader.load(&path).unwrap_err();
    assert!(matches!(err, Error::MalformedModel { format: "safetensors", .. }), "got {err:?}");
}

#[test]
fn files_shorter_than_the_length_prefix_fail_gracefully() {
    let dir = tempfile::tempdir().unwrap();
    for bytes in [&b""[..], &b"\x01\x02\x03"[..]] {
        let path = write_temp_file(&dir, "too_short.safetensors", bytes);
        let err = SafeTensorsLoader.load(&path).unwrap_err();
        assert!(matches!(err, Error::MalformedModel { format: "safetensors", .. }), "got {err:?}");
    }
}

/// If a real SafeTensors file happens to be present on this machine (this
/// repo vendors one of `candle`'s example assets), load it end-to-end.
/// This particular fixture's only tensor is `I64`, which
/// `kopitiam_core::DType` cannot represent, so the meaningful assertion is
/// that loading fails with `UnsupportedModelFeature` naming that dtype —
/// gracefully, not a panic — rather than that it succeeds.
/// `#[ignore]`d for the same reason as the GGUF equivalent: it depends on a
/// vendored file rather than a hand-built fixture.
#[test]
#[ignore = "depends on a real vendored SafeTensors file being present on disk"]
fn a_real_vendored_safetensors_file_with_an_unsupported_dtype_fails_gracefully() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../kopitiam-ai/vendor/candle/candle-examples/examples/encodec/jfk-codes.safetensors"
    );
    let err = load_model(path).expect_err("this fixture's only tensor is I64, which is unsupported");
    assert!(
        matches!(err, Error::UnsupportedModelFeature { format: "safetensors", .. }),
        "got {err:?}"
    );
}

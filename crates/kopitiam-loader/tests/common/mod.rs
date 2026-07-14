//! Byte-level helpers for building GGUF fixtures by hand in tests.
//!
//! These mirror the wire format documented in
//! `crates/kopitiam-loader/src/gguf.rs` field-for-field (magic, version,
//! counts, typed KV entries, tensor info records), so a test that builds a
//! file with these and a test that parses it with `kopitiam_loader::gguf`
//! are exercising the same spec from opposite ends — that symmetry is the
//! whole point of hand-building fixtures instead of shipping a real
//! multi-gigabyte model as a test asset.

#![allow(dead_code)] // not every helper is used by every test binary that includes this module.

pub const TYPE_U8: u32 = 0;
pub const TYPE_I8: u32 = 1;
pub const TYPE_U16: u32 = 2;
pub const TYPE_I16: u32 = 3;
pub const TYPE_U32: u32 = 4;
pub const TYPE_I32: u32 = 5;
pub const TYPE_F32: u32 = 6;
pub const TYPE_BOOL: u32 = 7;
pub const TYPE_STRING: u32 = 8;
pub const TYPE_ARRAY: u32 = 9;
pub const TYPE_U64: u32 = 10;
pub const TYPE_I64: u32 = 11;
pub const TYPE_F64: u32 = 12;

pub const GGML_TYPE_F32: u32 = 0;
pub const GGML_TYPE_F16: u32 = 1;
pub const GGML_TYPE_Q4_0: u32 = 2;
pub const GGML_TYPE_Q5_0: u32 = 6;
pub const GGML_TYPE_Q8_0: u32 = 8;
pub const GGML_TYPE_Q2_K: u32 = 10; // deliberately unsupported by kopitiam-loader

pub fn push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

pub fn push_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

pub fn push_f32(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// A `gguf_string_t`: u64 length prefix, then raw UTF-8 bytes.
pub fn push_string(buf: &mut Vec<u8>, s: &str) {
    push_u64(buf, s.len() as u64);
    buf.extend_from_slice(s.as_bytes());
}

pub fn push_header(buf: &mut Vec<u8>, version: u32, tensor_count: u64, kv_count: u64) {
    buf.extend_from_slice(b"GGUF");
    push_u32(buf, version);
    push_u64(buf, tensor_count);
    push_u64(buf, kv_count);
}

pub fn push_kv_string(buf: &mut Vec<u8>, key: &str, value: &str) {
    push_string(buf, key);
    push_u32(buf, TYPE_STRING);
    push_string(buf, value);
}

pub fn push_kv_u32(buf: &mut Vec<u8>, key: &str, value: u32) {
    push_string(buf, key);
    push_u32(buf, TYPE_U32);
    push_u32(buf, value);
}

pub fn push_kv_f32(buf: &mut Vec<u8>, key: &str, value: f32) {
    push_string(buf, key);
    push_u32(buf, TYPE_F32);
    push_f32(buf, value);
}

pub fn push_kv_bool_byte(buf: &mut Vec<u8>, key: &str, raw_byte: u8) {
    push_string(buf, key);
    push_u32(buf, TYPE_BOOL);
    buf.push(raw_byte);
}

pub fn push_kv_string_array(buf: &mut Vec<u8>, key: &str, values: &[&str]) {
    push_string(buf, key);
    push_u32(buf, TYPE_ARRAY);
    push_u32(buf, TYPE_STRING);
    push_u64(buf, values.len() as u64);
    for v in values {
        push_string(buf, v);
    }
}

/// A `gguf_tensor_info_t`: name, dimension count, dimensions in ggml's
/// `ne[]` (fastest-varying-first) order, ggml type id, and offset relative
/// to the tensor data section.
pub fn push_tensor_info(buf: &mut Vec<u8>, name: &str, ne: &[u64], ggml_type: u32, offset: u64) {
    push_string(buf, name);
    push_u32(buf, ne.len() as u32);
    for &d in ne {
        push_u64(buf, d);
    }
    push_u32(buf, ggml_type);
    push_u64(buf, offset);
}

/// Pads `buf` with zero bytes up to the next multiple of `alignment`.
pub fn pad_to_alignment(buf: &mut Vec<u8>, alignment: usize) {
    let pad = alignment.wrapping_sub(buf.len() % alignment) % alignment;
    buf.extend(std::iter::repeat_n(0u8, pad));
}

pub fn write_temp_file(dir: &tempfile::TempDir, name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path = dir.path().join(name);
    std::fs::write(&path, bytes).expect("write temp fixture file");
    path
}

// y = x @ w^T, where w is BLOCK-SCALED INT8 held resident on the device.
//
// Each row of w (one output feature, k wide) is split into k/32 blocks of 32
// int8 quants, each block carrying one f32 scale. An element is
// `scale[block] * quant`, which is ggml's Q8_0 scheme expressed generically —
// this crate deliberately knows nothing about GGUF; the caller separates the
// scales and quants and hands them over (see ops/block_q8_matmul_nt.rs).
//
// WHY THIS KERNEL EXISTS. The output projection of even a 360M model is
// 49152 x 960: 188 MB as f32, which does not fit a 128 MB
// `max_storage_buffer_binding_size` and so cannot reach the GPU at all. Held as
// block-scaled int8 the same weight is ~47 MB of quants plus ~6 MB of scales,
// which fits comfortably. It is also the single most expensive matmul in a
// decode step (25.67 ms on CPU, more than every other decode matmul combined),
// so it is the one worth the trouble.
//
// Bindings (must match the bind group in ops/block_q8_matmul_nt.rs):
//   @binding(0) x      -- read-only storage, [m, k] f32 row-major (activations)
//   @binding(1) quants -- read-only storage, k*n int8 PACKED 4-per-u32
//   @binding(2) scales -- read-only storage, [n, k/32] f32
//   @binding(3) y      -- read_write storage, [m, n] f32 row-major
//   @binding(4) dims   -- uniform (m, k, n, _pad)
//
// PACKING. WGSL has no i8 type and storage buffers are typed, so the quants
// arrive as `array<u32>`, four int8s per word, little-endian: element t of the
// row lives in byte (t % 4) of word (t / 4). The sign extension below
// (`<< 24 >> 24` on a signed i32) is not decoration — without it, quant 0xFF
// reads as +255 instead of -1 and every negative weight silently flips sign.
// Done manually rather than via `unpack4xI8` so this loads on older naga
// backends too, which matters for the low-end adapters this project targets.
//
// ACCUMULATION ORDER. The scale multiplies the whole block's integer dot
// product once, rather than each element — that is both how the format is
// defined and cheaper. The block sub-total accumulates in f32, matching the CPU
// twin exactly so the two remain comparable.

struct Dims {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
};

const BLOCK: u32 = 32u;

@group(0) @binding(0) var<storage, read>       x:      array<f32>;
@group(0) @binding(1) var<storage, read>       quants: array<u32>;
@group(0) @binding(2) var<storage, read>       scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> y:      array<f32>;
@group(0) @binding(4) var<uniform>             dims:   Dims;

fn quant_at(index: u32) -> f32 {
    let word = quants[index / 4u];
    let shift = (index % 4u) * 8u;
    let byte = (word >> shift) & 0xFFu;
    // Sign-extend the low 8 bits into a signed value in [-128, 127].
    let signed = (i32(byte) << 24u) >> 24u;
    return f32(signed);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x; // 0..m — activation row
    let col = gid.y; // 0..n — output feature
    if (row >= dims.m || col >= dims.n) {
        return;
    }

    let blocks_per_row = dims.k / BLOCK;
    let x_base = row * dims.k;
    let q_base = col * dims.k;
    let s_base = col * blocks_per_row;

    var acc: f32 = 0.0;
    for (var b: u32 = 0u; b < blocks_per_row; b = b + 1u) {
        var block_sum: f32 = 0.0;
        let offset = b * BLOCK;
        for (var t: u32 = 0u; t < BLOCK; t = t + 1u) {
            block_sum = block_sum + x[x_base + offset + t] * quant_at(q_base + offset + t);
        }
        acc = acc + scales[s_base + b] * block_sum;
    }
    y[row * dims.n + col] = acc;
}

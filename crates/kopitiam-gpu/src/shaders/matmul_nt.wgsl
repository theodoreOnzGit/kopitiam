// y = x @ w^T  — the shape a transformer linear layer actually needs.
//
// x is [m, k] (m activation rows of width k), w is [n, k] (n output features,
// each a k-wide row), y is [m, n]:
//
//     y[i][j] = dot(x[i][..], w[j][..])
//
// WHY TRANSPOSED-B AND NOT A GENERIC MATMUL. GGUF stores a linear layer's
// weight row-major as [out_features, in_features], and `kopitiam-runtime`'s
// `linear()` computes `x @ weight^T`. Implementing that shape directly means
// both operands are read along their contiguous axis — row i of x and row j of
// w are each a contiguous k-element run — so every read is coalesced. A generic
// `a @ b` kernel would force either a transposed copy of the weight on upload
// or a strided (cache-hostile) read per element. Same reason ggml stores
// weights this way and dots rows.
//
// Buffer bindings (must match the bind-group built in ops/matmul_nt.rs):
//   @binding(0) x    -- read-only storage, [m, k] row-major
//   @binding(1) w    -- read-only storage, [n, k] row-major
//   @binding(2) y    -- read_write storage, [m, n] row-major
//   @binding(3) dims -- uniform, (m, k, n, _pad)
//
// WORKGROUP SHAPE. 16x16 = 256 invocations, one per output element. 256 is the
// downlevel guaranteed maximum for `maxComputeInvocationsPerWorkgroup`, so this
// still loads on low-end and mobile adapters (which matters: Termux is a
// first-class target, even though it usually falls back to CPU for lack of a
// Vulkan driver). 16x16 tiles the [m, n] output squarely, so both a tall-thin
// prefill ([33, 2560]) and a single decode row ([1, 2560]) map sensibly.
//
// BOUNDS GUARD. The dispatch rounds both dimensions up, so the edge workgroups
// spawn invocations past the output. Anything outside [m, n] must return before
// touching memory — not optional, and not merely defensive: without it the tail
// invocations of a [1, n] decode step would stomp past the end of y.
//
// NUMERICS. The accumulator is f32 and the sum runs in index order, exactly as
// the CPU twin does, so the two agree to within f32 addition-order noise. This
// kernel deliberately does NOT use a tree reduction or fused multiply-add,
// because a different summation order would make GPU/CPU disagreement expected
// and therefore un-testable — and this crate's whole discipline is that the two
// twins are comparable.

struct Dims {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read>       x:    array<f32>;
@group(0) @binding(1) var<storage, read>       w:    array<f32>;
@group(0) @binding(2) var<storage, read_write> y:    array<f32>;
@group(0) @binding(3) var<uniform>             dims: Dims;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x; // 0..m  — which activation row
    let col = gid.y; // 0..n  — which output feature
    if (row >= dims.m || col >= dims.n) {
        return;
    }

    let x_base = row * dims.k;
    let w_base = col * dims.k;
    var acc: f32 = 0.0;
    for (var t: u32 = 0u; t < dims.k; t = t + 1u) {
        acc = acc + x[x_base + t] * w[w_base + t];
    }
    y[row * dims.n + col] = acc;
}

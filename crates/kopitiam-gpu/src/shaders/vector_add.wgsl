// Elementwise vector add: out[i] = a[i] + b[i].
//
// This is the demonstrator kernel for the GPU->CPU cascade. It is deliberately
// the simplest useful compute shader: one GPU thread ("invocation") per output
// element, no cross-thread communication, no shared memory.
//
// Buffer bindings (must match the bind-group layout built in ops/vector_add.rs,
// binding index for binding index):
//   @binding(0) a   -- read-only storage, the first input vector
//   @binding(1) b   -- read-only storage, the second input vector
//   @binding(2) out -- read_write storage, where the sum lands (COPY_SRC on host)
//
// WORKGROUP SHAPE. @workgroup_size(64) means each workgroup runs 64 invocations.
// The host dispatches ceil(n / 64) workgroups in the x dimension (see
// dispatch_workgroups in the Rust side), so total invocations = 64 * ceil(n/64),
// which is >= n. 64 is a safe, portable choice: it is a multiple of the 32/64
// hardware wavefront/warp width on every backend wgpu targets (Vulkan, Metal,
// DX12, GL, WebGPU) and stays well under the 256-invocation downlevel limit, so
// this shader loads on low-end and mobile adapters too.
//
// BOUNDS GUARD. Because we round the dispatch UP, the last workgroup spawns
// invocations past the end of the arrays. `global_invocation_id.x` is the flat
// element index; any invocation with i >= arrayLength(&out) must do nothing, or
// it would read/write out of bounds. That early-return is not optional.

@group(0) @binding(0) var<storage, read>       a:   array<f32>;
@group(0) @binding(1) var<storage, read>       b:   array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&out)) {
        return;
    }
    out[i] = a[i] + b[i];
}

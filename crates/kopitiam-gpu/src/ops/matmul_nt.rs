//! `y = x @ w^T` — the matmul a transformer linear layer actually performs.
//!
//! `x` is `[m, k]`, `w` is `[n, k]` (GGUF's `[out_features, in_features]`
//! row-major convention), `y` is `[m, n]`. This is the shape
//! `kopitiam_runtime::linear` computes, and the reason it is worth a GPU kernel
//! at all: the attention projections (q/k/v/o), the MLP's gate/up/down, and the
//! output projection are where a decoder-only model spends nearly all of its
//! time.
//!
//! # Where offload pays, and where it costs — measured, not assumed
//!
//! This op uploads `w` on **every call**, so whether it helps depends entirely
//! on how much arithmetic that upload is amortised over. Measured on an Intel
//! integrated GPU against 14 CPU cores, at SmolLM2-360M's real projection
//! shapes (`tests/matmul_timing.rs`, timings INCLUDE upload and readback):
//!
//! ```text
//! decode  attn q/o     (1 tok)   cpu   443µs   gpu 1.88ms   0.24x   LOSS
//! decode  mlp gate/up  (1 tok)   cpu  1.28ms   gpu 3.20ms   0.40x   LOSS
//! decode  mlp down     (1 tok)   cpu  1.29ms   gpu 2.86ms   0.45x   LOSS
//! decode  output head  (1 tok)   cpu 25.67ms   n/a — exceeds binding limit
//! prefill attn q/o    (33 tok)   cpu 12.89ms   gpu 3.72ms   3.46x   WIN
//! prefill mlp gate/up (33 tok)   cpu 40.83ms   gpu 7.77ms   5.26x   WIN
//! ```
//!
//! The split is the whole story: **a decode step is one row of activations**,
//! far too little work to pay for moving a weight matrix, while a 33-token
//! prefill does 33x the arithmetic against the same upload and wins outright.
//! Anyone wiring this into a forward pass should offload prefill and leave
//! decode alone; doing it uniformly would make chat 2-4x *slower*, which is the
//! opposite of the thing that prompted the work.
//!
//! Two further facts that shape the real design:
//!
//! * **The most expensive decode op cannot run here at all.** The output head
//!   is `49152 x 960` — 188 MB as `f32`, against a 128 MB
//!   `max_storage_buffer_binding_size` on this adapter — and at 25.67 ms it
//!   dominates every other decode matmul combined. Getting *it* onto the GPU
//!   needs the weight kept **quantized** on the device (Q8_0 is ~47 MB, which
//!   fits), not merely resident.
//! * Making decode pay at all needs the weight resident across calls. That is
//!   deliberately a separate change: a resident-weight cache is an
//!   ownership/lifetime design question, not something to smuggle in alongside
//!   a first kernel.
//!
//! So treat this as the correctness-checked kernel and building block, with the
//! numbers above as the map of where to point it.

use crate::context::GpuContext;
use crate::executor::{ComputeOp, GpuOpError};
use wgpu::util::DeviceExt;

/// Must match `@workgroup_size(16, 16)` in `shaders/matmul_nt.wgsl`.
const WORKGROUP_X: u32 = 16;
const WORKGROUP_Y: u32 = 16;

/// `x` `[m, k]` times the transpose of `w` `[n, k]`.
///
/// `k` is carried explicitly rather than inferred, because inferring it from
/// `x.len() / m` would silently accept a ragged input and produce a plausible
/// wrong answer instead of an error.
pub struct MatmulNtInput<'a> {
    pub x: &'a [f32],
    pub w: &'a [f32],
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

/// `y = x @ w^T`. Zero-sized; it names the operation for the [`ComputeOp`] impl.
pub struct MatmulNt;

impl ComputeOp for MatmulNt {
    type Input<'a> = MatmulNtInput<'a>;
    type Output = Vec<f32>;

    fn compute_gpu(
        &self,
        ctx: &GpuContext,
        input: &Self::Input<'_>,
    ) -> Result<Self::Output, GpuOpError> {
        matmul_nt_gpu(ctx, input.x, input.w, input.m, input.k, input.n)
    }

    fn compute_cpu(&self, input: &Self::Input<'_>) -> Self::Output {
        matmul_nt_cpu(input.x, input.w, input.m, input.k, input.n)
    }
}

/// The pure-Rust twin, and the floor of the cascade.
///
/// Sums in index order, matching the WGSL exactly, so the two paths are
/// comparable rather than merely both "about right". Returns an all-zero
/// `[m, n]` if the inputs are too short — the GPU path rejects that case as
/// [`GpuOpError::InvalidInput`], and callers should not rely on either
/// behaviour; validate before calling.
#[must_use]
pub fn matmul_nt_cpu(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0f32; m * n];
    if x.len() < m * k || w.len() < n * k {
        return y;
    }
    for i in 0..m {
        let x_row = &x[i * k..i * k + k];
        for j in 0..n {
            let w_row = &w[j * k..j * k + k];
            let mut acc = 0f32;
            for t in 0..k {
                acc += x_row[t] * w_row[t];
            }
            y[i * n + j] = acc;
        }
    }
    y
}

/// The wgpu compute path. Any wgpu-level failure returns `Err` so
/// [`crate::Executor`] falls back to [`matmul_nt_cpu`].
///
/// Mirrors `shaders/matmul_nt.wgsl` binding for binding: `x`, `w`, `y`, and a
/// `dims` uniform carrying `(m, k, n)`. The dispatch is
/// `ceil(m/16) x ceil(n/16)` workgroups; the shader guards the rounded-up tail.
pub fn matmul_nt_gpu(
    ctx: &GpuContext,
    x: &[f32],
    w: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, GpuOpError> {
    if x.len() != m * k {
        return Err(GpuOpError::InvalidInput(format!(
            "x has {} elements, expected m*k = {}*{} = {}",
            x.len(),
            m,
            k,
            m * k
        )));
    }
    if w.len() != n * k {
        return Err(GpuOpError::InvalidInput(format!(
            "w has {} elements, expected n*k = {}*{} = {}",
            w.len(),
            n,
            k,
            n * k
        )));
    }
    // Zero-sized buffers are a validation error on some backends, and an empty
    // dispatch computes nothing anyway.
    if m == 0 || n == 0 || k == 0 {
        return Ok(vec![0f32; m * n]);
    }

    let device = ctx.device();
    let queue = ctx.queue();
    let out_bytes = (m * n * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

    // Refuse anything the adapter cannot bind, BEFORE asking wgpu to do it.
    //
    // This is not defensive padding: wgpu treats an over-limit binding as a
    // validation error and **panics** rather than returning `Err`, which would
    // abort the process instead of cascading to the CPU twin. And the limit is
    // reachable with ordinary weights — SmolLM2-360M's output head is
    // 49152 x 960 f32 = 188 MB against a 128 MB
    // `max_storage_buffer_binding_size` on this Intel adapter, so the single
    // largest matmul in the model is exactly the one that blows it.
    let limit = device.limits().max_storage_buffer_binding_size;
    let biggest = [
        (m * k * std::mem::size_of::<f32>()) as u64,
        (n * k * std::mem::size_of::<f32>()) as u64,
        out_bytes,
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    if biggest > limit {
        return Err(GpuOpError::InvalidInput(format!(
            "matmul {m}x{k}x{n} needs a {biggest}-byte storage binding but this \
             adapter's max_storage_buffer_binding_size is {limit}; falling back to CPU"
        )));
    }

    let x_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("matmul_nt.x"),
        contents: bytemuck::cast_slice(x),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let w_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("matmul_nt.w"),
        contents: bytemuck::cast_slice(w),
        usage: wgpu::BufferUsages::STORAGE,
    });
    // WGSL's `Dims` is four u32s; the fourth is padding so the struct meets the
    // 16-byte alignment a uniform buffer requires.
    let dims: [u32; 4] = [m as u32, k as u32, n as u32, 0];
    let dims_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("matmul_nt.dims"),
        contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let y_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("matmul_nt.y"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("matmul_nt.staging"),
        size: out_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("matmul_nt.wgsl"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/matmul_nt.wgsl").into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("matmul_nt.pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("matmul_nt.bind_group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: w_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: y_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: dims_buf.as_entire_binding() },
        ],
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("matmul_nt") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matmul_nt.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            (m as u32).div_ceil(WORKGROUP_X),
            (n as u32).div_ceil(WORKGROUP_Y),
            1,
        );
    }
    encoder.copy_buffer_to_buffer(&y_buf, 0, &staging, 0, out_bytes);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| GpuOpError::Backend(format!("device poll failed: {e:?}")))?;
    rx.recv()
        .map_err(|e| GpuOpError::Backend(format!("map callback dropped: {e}")))?
        .map_err(|e| GpuOpError::Backend(format!("buffer map failed: {e:?}")))?;

    let data = slice
        .get_mapped_range()
        .map_err(|e| GpuOpError::Backend(format!("get_mapped_range failed: {e:?}")))?;
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Executor;

    /// Deterministic pseudo-random fill — no `rand` dependency, and the same
    /// values on every run so a failure is reproducible.
    fn fill(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * 0.37 + seed).sin()).collect()
    }

    #[test]
    fn cpu_matches_a_hand_computed_product() {
        // x = [[1, 2], [3, 4]]  (m=2, k=2)
        // w = [[5, 6], [7, 8]]  (n=2, k=2)  -> w^T = [[5, 7], [6, 8]]
        // y = [[1*5+2*6, 1*7+2*8], [3*5+4*6, 3*7+4*8]] = [[17, 23], [39, 53]]
        let y = matmul_nt_cpu(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 2, 2, 2);
        assert_eq!(y, vec![17.0, 23.0, 39.0, 53.0]);
    }

    /// A non-square case, because square shapes hide index-order bugs: a kernel
    /// that transposed its output would still pass a symmetric test.
    #[test]
    fn cpu_handles_non_square_shapes() {
        // x [2,3], w [4,3] -> y [2,4]
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w: Vec<f32> = (1..=12).map(|v| v as f32).collect();
        let y = matmul_nt_cpu(&x, &w, 2, 3, 4);
        assert_eq!(y.len(), 8);
        // y[0][0] = 1*1 + 2*2 + 3*3 = 14; y[1][3] = 4*10 + 5*11 + 6*12 = 167
        assert_eq!(y[0], 14.0);
        assert_eq!(y[7], 167.0);
    }

    #[test]
    fn gpu_rejects_ragged_inputs_instead_of_guessing() {
        let Ok(ctx) = crate::GpuContext::new() else {
            eprintln!("skipped: no GPU on this machine");
            return;
        };
        let err = matmul_nt_gpu(&ctx, &[1.0, 2.0], &[1.0, 2.0], 2, 2, 1).unwrap_err();
        assert!(matches!(err, GpuOpError::InvalidInput(_)), "got {err:?}");
    }

    /// The property that matters: the GPU kernel and the CPU twin agree, at the
    /// shapes a real transformer uses, not just on a toy 2x2.
    ///
    /// Tolerance is relative and small but not zero — both paths sum in index
    /// order, so they should agree closely, but a GPU may still contract
    /// `a*b + c` into an FMA and change the last bit.
    #[test]
    fn gpu_matches_cpu_at_real_transformer_shapes() {
        let Ok(ctx) = crate::GpuContext::new() else {
            eprintln!("skipped: no GPU on this machine");
            return;
        };
        // (m, k, n): a 33-token prefill and a 1-token decode step through
        // SmolLM2-360M's actual projections — attention q/o (960x960), the MLP
        // gate/up (960 -> 2560) and down (2560 -> 960).
        for &(m, k, n) in &[(33usize, 960usize, 960usize), (1, 960, 2560), (33, 2560, 960), (1, 64, 64)] {
            let x = fill(m * k, 0.1);
            let w = fill(n * k, 0.7);
            let gpu = matmul_nt_gpu(&ctx, &x, &w, m, k, n).expect("gpu matmul");
            let cpu = matmul_nt_cpu(&x, &w, m, k, n);
            assert_eq!(gpu.len(), cpu.len(), "shape {m}x{k}x{n}");

            let mut worst = 0f32;
            for (g, c) in gpu.iter().zip(&cpu) {
                worst = worst.max((g - c).abs());
            }
            let scale = cpu.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
            assert!(
                worst / scale < 1e-4,
                "shape {m}x{k}x{n}: GPU and CPU disagree, worst {worst} (relative {})",
                worst / scale
            );
        }
    }

    /// The cascade must produce the same answer whichever way it went, so a
    /// machine with no GPU is not quietly running different maths.
    #[test]
    fn the_executor_cascade_agrees_with_the_forced_cpu_path() {
        let (m, k, n) = (8usize, 64usize, 32usize);
        let x = fill(m * k, 0.3);
        let w = fill(n * k, 0.9);
        let input = MatmulNtInput { x: &x, w: &w, m, k, n };

        let cascade = Executor::new().run(&MatmulNt, &input);
        let cpu_only = Executor::cpu_only().run(&MatmulNt, &input);
        assert_eq!(cascade.len(), cpu_only.len());
        let worst = cascade
            .iter()
            .zip(&cpu_only)
            .fold(0f32, |acc, (a, b)| acc.max((a - b).abs()));
        let scale = cpu_only.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
        assert!(worst / scale < 1e-4, "cascade disagreed with CPU: worst {worst}");
    }
}

#[cfg(test)]
mod limit_tests {
    use super::*;

    /// A weight too large for the adapter's binding limit must come back as an
    /// error the cascade can catch — NOT a panic.
    ///
    /// wgpu reports an over-limit binding as a validation error and aborts the
    /// process, so without the up-front check this exact shape (SmolLM2-360M's
    /// output head, 49152 x 960 f32 = 188 MB) would kill the caller instead of
    /// quietly running on the CPU. Found by benchmarking, not by review.
    #[test]
    fn an_oversized_weight_falls_back_instead_of_panicking() {
        let Ok(ctx) = crate::GpuContext::new() else {
            eprintln!("skipped: no GPU on this machine");
            return;
        };
        let limit = ctx.device().limits().max_storage_buffer_binding_size as usize;
        let k = 960usize;
        // One row past the limit, so this is over on every adapter rather than
        // relying on any particular device's numbers.
        let n = limit / (k * std::mem::size_of::<f32>()) + 1;

        // Allocating the host-side weight would cost the same memory, so assert
        // on the guard's arithmetic via a deliberately short slice: the length
        // check fires first for a ragged input, so pass a correctly-sized `x`
        // and a `w` that is correctly sized but oversized for the device.
        let x = vec![0.0f32; k];
        let w = vec![0.0f32; n * k];
        let err = matmul_nt_gpu(&ctx, &x, &w, 1, k, n).expect_err("must refuse, not panic");
        assert!(
            matches!(err, GpuOpError::InvalidInput(ref m) if m.contains("max_storage_buffer_binding_size")),
            "expected a binding-size refusal, got {err:?}"
        );

        // And the cascade must then still produce the right answer on CPU.
        let out = crate::Executor::new().run(&MatmulNt, &MatmulNtInput { x: &x, w: &w, m: 1, k, n });
        assert_eq!(out.len(), n);
    }
}

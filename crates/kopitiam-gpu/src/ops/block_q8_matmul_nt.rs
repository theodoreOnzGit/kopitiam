//! `y = x @ w^T` with `w` **block-scaled int8 and resident on the device**.
//!
//! This is the op that makes GPU offload actually pay for a decode step, and it
//! exists because of two measurements (see `tests/matmul_timing.rs` and
//! [`super::matmul_nt`]):
//!
//! * Re-uploading an `f32` weight per token is a 0.24–0.45x **loss** — a single
//!   activation row is nowhere near enough arithmetic to amortise the transfer.
//!   So the weight has to stay on the device across calls. That is what
//!   [`ResidentBlockQ8Weight`] is: upload once, dispatch many.
//! * The weight that matters most cannot be uploaded as `f32` at all. A 360M
//!   model's output projection is `49152 x 960` — 188 MB, against a 128 MB
//!   `max_storage_buffer_binding_size` — while costing 25.67 ms per token on
//!   CPU, more than every other decode matmul combined. Held as block-scaled
//!   int8 it is ~47 MB of quants plus ~6 MB of scales, which fits.
//!
//! # The format, and why this crate does not name it "Q8_0"
//!
//! Each row of `w` (one output feature, `k` wide) is `k / [`BLOCK`]` blocks of
//! 32 int8 quants, each block carrying one `f32` scale; an element is
//! `scale[block] * quant`. That is exactly ggml's Q8_0 scheme, but
//! `kopitiam-gpu` is domain-agnostic parallel-compute infrastructure and has no
//! business knowing about GGUF. The caller separates the scales from the quants
//! and passes both — format knowledge stays in `kopitiam-runtime`, where the
//! GGUF block layout is already understood and verified against ggml.

use crate::context::GpuContext;
use crate::executor::GpuOpError;
use wgpu::util::DeviceExt;

/// Quants per scale block. Must match `const BLOCK` in
/// `shaders/block_q8_matmul_nt.wgsl`.
pub const BLOCK: usize = 32;

const WORKGROUP_X: u32 = 16;
const WORKGROUP_Y: u32 = 16;

/// The pure-Rust twin: `y = x @ w^T` from separated scales and int8 quants.
///
/// Accumulates a block's integer dot product first and applies the scale once
/// per block, exactly as the WGSL does, so the two paths stay comparable rather
/// than merely both "about right".
///
/// Returns an all-zero `[m, n]` on ragged input; the GPU path rejects that case
/// explicitly, and callers should validate rather than depend on either.
#[must_use]
pub fn block_q8_matmul_nt_cpu(
    x: &[f32],
    quants: &[i8],
    scales: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut y = vec![0f32; m * n];
    let blocks_per_row = k / BLOCK;
    if x.len() < m * k || quants.len() < n * k || scales.len() < n * blocks_per_row {
        return y;
    }
    for i in 0..m {
        let x_row = &x[i * k..i * k + k];
        for j in 0..n {
            let q_row = &quants[j * k..j * k + k];
            let s_row = &scales[j * blocks_per_row..j * blocks_per_row + blocks_per_row];
            let mut acc = 0f32;
            for (b, &scale) in s_row.iter().enumerate() {
                let off = b * BLOCK;
                let mut block_sum = 0f32;
                for t in 0..BLOCK {
                    block_sum += x_row[off + t] * f32::from(q_row[off + t]);
                }
                acc += scale * block_sum;
            }
            y[i * n + j] = acc;
        }
    }
    y
}

/// A block-scaled int8 weight living in GPU memory, plus its compiled pipeline.
///
/// Build one per weight at model load and call [`Self::matmul_nt`] per token.
/// Building it per call would reintroduce exactly the per-call upload this type
/// exists to remove, and would also recompile the shader every time.
#[derive(Debug)]
pub struct ResidentBlockQ8Weight {
    quants: wgpu::Buffer,
    scales: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    n: usize,
    k: usize,
}

impl ResidentBlockQ8Weight {
    /// Uploads `quants` (`[n, k]`, row-major) and `scales` (`[n, k/BLOCK]`) once.
    ///
    /// # Errors
    ///
    /// [`GpuOpError::InvalidInput`] if the lengths do not match `n`/`k`, if `k`
    /// is not a multiple of [`BLOCK`], or if either buffer exceeds the adapter's
    /// `max_storage_buffer_binding_size` — checked up front because wgpu
    /// **panics** on an over-limit binding instead of returning an error, which
    /// would abort the process rather than let the caller fall back to CPU.
    pub fn upload(
        ctx: &GpuContext,
        quants: &[i8],
        scales: &[f32],
        n: usize,
        k: usize,
    ) -> Result<Self, GpuOpError> {
        if k == 0 || !k.is_multiple_of(BLOCK) {
            return Err(GpuOpError::InvalidInput(format!(
                "k ({k}) must be a non-zero multiple of BLOCK ({BLOCK})"
            )));
        }
        if quants.len() != n * k {
            return Err(GpuOpError::InvalidInput(format!(
                "quants has {} elements, expected n*k = {}",
                quants.len(),
                n * k
            )));
        }
        let blocks_per_row = k / BLOCK;
        if scales.len() != n * blocks_per_row {
            return Err(GpuOpError::InvalidInput(format!(
                "scales has {} elements, expected n*(k/{BLOCK}) = {}",
                scales.len(),
                n * blocks_per_row
            )));
        }

        let device = ctx.device();
        let limit = device.limits().max_storage_buffer_binding_size;
        // Quants are packed 4-per-u32, so the buffer is n*k bytes exactly.
        let quant_bytes = (n * k) as u64;
        let scale_bytes = std::mem::size_of_val(scales) as u64;
        if quant_bytes.max(scale_bytes) > limit {
            return Err(GpuOpError::InvalidInput(format!(
                "weight needs a {}-byte storage binding but this adapter's \
                 max_storage_buffer_binding_size is {limit}",
                quant_bytes.max(scale_bytes)
            )));
        }

        // Pack four int8s per u32, little-endian, matching `quant_at` in the
        // shader. `as u8 as u32` keeps the two's-complement bit pattern; the
        // shader sign-extends it back.
        let mut packed = vec![0u32; (n * k).div_ceil(4)];
        for (i, &q) in quants.iter().enumerate() {
            packed[i / 4] |= (q as u8 as u32) << ((i % 4) * 8);
        }

        let quants_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("block_q8.quants"),
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let scales_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("block_q8.scales"),
            contents: bytemuck::cast_slice(scales),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("block_q8_matmul_nt.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/block_q8_matmul_nt.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("block_q8.pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self { quants: quants_buf, scales: scales_buf, pipeline, n, k })
    }

    /// Output features (`n`) this weight produces.
    #[must_use]
    pub fn out_features(&self) -> usize {
        self.n
    }

    /// Input width (`k`) this weight consumes.
    #[must_use]
    pub fn in_features(&self) -> usize {
        self.k
    }

    /// `y = x @ w^T` for `m` rows of activations. Only `x` crosses the bus.
    ///
    /// # Errors
    ///
    /// [`GpuOpError::InvalidInput`] if `x` is not `m * k` long, and
    /// [`GpuOpError::Backend`] for any wgpu-level failure — both of which the
    /// caller should treat as "fall back to the CPU twin".
    pub fn matmul_nt(
        &self,
        ctx: &GpuContext,
        x: &[f32],
        m: usize,
    ) -> Result<Vec<f32>, GpuOpError> {
        if x.len() != m * self.k {
            return Err(GpuOpError::InvalidInput(format!(
                "x has {} elements, expected m*k = {}*{} = {}",
                x.len(),
                m,
                self.k,
                m * self.k
            )));
        }
        if m == 0 {
            return Ok(Vec::new());
        }

        let device = ctx.device();
        let queue = ctx.queue();
        let out_bytes = (m * self.n * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

        let x_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("block_q8.x"),
            contents: bytemuck::cast_slice(x),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let dims: [u32; 4] = [m as u32, self.k as u32, self.n as u32, 0];
        let dims_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("block_q8.dims"),
            contents: bytemuck::cast_slice(&dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let y_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("block_q8.y"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("block_q8.staging"),
            size: out_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("block_q8.bind_group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.quants.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.scales.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: y_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: dims_buf.as_entire_binding() },
            ],
        });

        let mut encoder = device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("block_q8") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("block_q8.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                (m as u32).div_ceil(WORKGROUP_X),
                (self.n as u32).div_ceil(WORKGROUP_Y),
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic_quants(n: usize) -> Vec<i8> {
        // Spans the full signed range INCLUDING -128 and -1, because the whole
        // sign-extension risk lives in the negative half: a kernel that read the
        // byte unsigned would pass a test using only small positives.
        (0..n).map(|i| ((i as i32 * 37 % 255) - 128) as i8).collect()
    }

    fn deterministic_scales(n: usize) -> Vec<f32> {
        (0..n).map(|i| 0.001 + (i as f32 * 0.13).sin().abs() * 0.01).collect()
    }

    #[test]
    fn cpu_twin_matches_a_hand_computation() {
        // n=1 output feature, k=32 (one block), scale 0.5, quants all 2, x all 3.
        // dot = 32 * (3 * 2) = 192; scaled = 0.5 * 192 = 96.
        let x = vec![3.0f32; 32];
        let quants = vec![2i8; 32];
        let scales = vec![0.5f32];
        let y = block_q8_matmul_nt_cpu(&x, &quants, &scales, 1, 32, 1);
        assert_eq!(y, vec![96.0]);
    }

    /// Negative quants must survive. A kernel (or a twin) that treated the byte
    /// as unsigned would read -1 as +255 and silently flip the sign of every
    /// negative weight — the single most likely bug in this file.
    #[test]
    fn cpu_twin_handles_negative_quants() {
        let mut quants = vec![0i8; 32];
        quants[0] = -1;
        quants[1] = -128;
        let x = vec![1.0f32; 32];
        let y = block_q8_matmul_nt_cpu(&x, &quants, &[1.0], 1, 32, 1);
        assert_eq!(y, vec![-129.0]);
    }

    #[test]
    fn gpu_matches_the_cpu_twin_including_negative_quants() {
        let Ok(ctx) = crate::GpuContext::new() else {
            eprintln!("skipped: no GPU on this machine");
            return;
        };
        // A realistic decode step through a transformer projection, small enough
        // to stay quick: 1 activation row, k=960, n=512.
        let (m, k, n) = (1usize, 960usize, 512usize);
        let quants = deterministic_quants(n * k);
        let scales = deterministic_scales(n * (k / BLOCK));
        let x: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.017).sin()).collect();

        let resident = ResidentBlockQ8Weight::upload(&ctx, &quants, &scales, n, k).expect("upload");
        assert_eq!(resident.out_features(), n);
        assert_eq!(resident.in_features(), k);

        let gpu = resident.matmul_nt(&ctx, &x, m).expect("gpu matmul");
        let cpu = block_q8_matmul_nt_cpu(&x, &quants, &scales, m, k, n);
        assert_eq!(gpu.len(), cpu.len());

        let worst = gpu.iter().zip(&cpu).fold(0f32, |a, (g, c)| a.max((g - c).abs()));
        let scale = cpu.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
        assert!(
            worst / scale < 1e-4,
            "GPU and CPU disagree: worst {worst} (relative {})",
            worst / scale
        );
    }

    /// The residency property: one upload, many dispatches, same answer every
    /// time. If a later call read stale or clobbered device memory this drifts.
    #[test]
    fn repeated_dispatches_reuse_the_resident_weight_consistently() {
        let Ok(ctx) = crate::GpuContext::new() else {
            eprintln!("skipped: no GPU on this machine");
            return;
        };
        let (k, n) = (64usize, 32usize);
        let quants = deterministic_quants(n * k);
        let scales = deterministic_scales(n * (k / BLOCK));
        let resident = ResidentBlockQ8Weight::upload(&ctx, &quants, &scales, n, k).expect("upload");

        let x1: Vec<f32> = (0..k).map(|i| (i as f32 * 0.03).sin()).collect();
        let x2: Vec<f32> = (0..k).map(|i| (i as f32 * 0.07).cos()).collect();

        let a1 = resident.matmul_nt(&ctx, &x1, 1).expect("dispatch 1");
        let b = resident.matmul_nt(&ctx, &x2, 1).expect("dispatch 2");
        let a2 = resident.matmul_nt(&ctx, &x1, 1).expect("dispatch 3");
        assert_eq!(a1, a2, "the same input must give the same answer after an intervening call");
        assert_ne!(a1, b, "different inputs must give different answers");
    }

    #[test]
    fn upload_rejects_a_k_that_is_not_a_whole_number_of_blocks() {
        let Ok(ctx) = crate::GpuContext::new() else {
            eprintln!("skipped: no GPU on this machine");
            return;
        };
        let err = ResidentBlockQ8Weight::upload(&ctx, &[0i8; 40], &[1.0], 1, 40).unwrap_err();
        assert!(matches!(err, GpuOpError::InvalidInput(ref m) if m.contains("multiple of BLOCK")));
    }
}

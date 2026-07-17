//! Elementwise vector add — the demonstrator op that proves the cascade.
//!
//! `out[i] = a[i] + b[i]`. Trivial arithmetic on purpose: the point of this op
//! is not the maths but to exercise the whole GPU->CPU machinery end to end —
//! host data into storage buffers, a WGSL compute dispatch, readback through a
//! staging buffer, and a pure-Rust twin that gives the identical answer when
//! there is no GPU.
//!
//! Both paths return `Vec<f32>` of the same length as the inputs, and for
//! elementwise `f32` addition the GPU and CPU results are **bit-for-bit equal**
//! (IEEE-754 addition is exact and deterministic here — no reordering, no
//! fused-multiply-add), which is why the tests can assert plain equality when a
//! GPU is present.

use crate::context::GpuContext;
use crate::executor::{ComputeOp, GpuOpError};
use wgpu::util::DeviceExt;

/// Must match `@workgroup_size(64)` in `shaders/vector_add.wgsl`. The host
/// dispatches `ceil(n / WORKGROUP_SIZE)` workgroups; the shader guards the tail.
const WORKGROUP_SIZE: u32 = 64;

/// The two input vectors, borrowed. Equal length is a precondition (checked in
/// both paths); mismatched lengths are a caller bug, not something to paper over.
pub struct VectorAddInput<'a> {
    pub a: &'a [f32],
    pub b: &'a [f32],
}

/// Elementwise vector addition. Zero-sized: it carries no state, it just names
/// the operation so the [`ComputeOp`] impl and its two kernels hang off a type.
pub struct VectorAdd;

impl ComputeOp for VectorAdd {
    type Input<'a> = VectorAddInput<'a>;
    type Output = Vec<f32>;

    fn compute_gpu(
        &self,
        ctx: &GpuContext,
        input: &Self::Input<'_>,
    ) -> Result<Self::Output, GpuOpError> {
        vector_add_gpu(ctx, input.a, input.b)
    }

    fn compute_cpu(&self, input: &Self::Input<'_>) -> Self::Output {
        vector_add_cpu(input.a, input.b)
    }
}

/// The pure-Rust floor of the cascade. Infallible and correct.
///
/// Length rule: the output is as long as the SHORTER input (`zip` stops there).
/// The GPU path enforces equal lengths and errors on mismatch; to keep the two
/// twins returning the same thing, callers should only ever pass equal-length
/// slices. Given equal lengths, this and the GPU kernel agree exactly.
pub fn vector_add_cpu(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// The wgpu compute path. Fallible: any wgpu-level problem returns `Err` so the
/// [`crate::Executor`] can fall back to [`vector_add_cpu`].
///
/// The dispatch shape, binding by binding, mirrors `shaders/vector_add.wgsl`:
///
/// 1. **Upload** `a` and `b` into read-only STORAGE buffers via
///    `create_buffer_init` (bytemuck reinterprets `&[f32]` as the `&[u8]` wgpu
///    wants — same bytes, no copy beyond the upload itself).
/// 2. **Allocate** an `out` STORAGE buffer with `COPY_SRC` (the shader writes it;
///    we then copy it to a mappable staging buffer — you cannot map a STORAGE
///    buffer directly for reading).
/// 3. **Bind** all three at group 0, bindings 0/1/2, matching the WGSL exactly.
/// 4. **Dispatch** `ceil(n / 64)` workgroups in x; the shader's bounds guard
///    handles the rounded-up tail invocations.
/// 5. **Read back**: copy `out` -> staging, submit, `map_async` + `poll(Wait)`
///    to block until the GPU is done, then `bytemuck`-cast the mapped bytes back
///    to `f32`. `poll(Wait)` is the synchronous join point — without it the map
///    callback may not have fired yet.
pub fn vector_add_gpu(ctx: &GpuContext, a: &[f32], b: &[f32]) -> Result<Vec<f32>, GpuOpError> {
    if a.len() != b.len() {
        return Err(GpuOpError::InvalidInput(format!(
            "vector_add needs equal-length inputs, got {} and {}",
            a.len(),
            b.len()
        )));
    }
    let n = a.len();
    // An empty dispatch is legal but pointless (and a zero-sized buffer is a
    // validation error on some backends), so short-circuit it.
    if n == 0 {
        return Ok(Vec::new());
    }

    let device = ctx.device();
    let queue = ctx.queue();
    // Bytes needed for one vector: a and b are equal length (checked above), so
    // size_of_val(a) is the byte length of each buffer. u64 for BufferAddress.
    let byte_len = std::mem::size_of_val(a) as wgpu::BufferAddress;

    // (1) input buffers, uploaded at creation.
    let a_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("vector_add.a"),
        contents: bytemuck::cast_slice(a),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let b_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("vector_add.b"),
        contents: bytemuck::cast_slice(b),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // (2) output buffer: the shader writes it (STORAGE) and we copy it out
    // afterwards (COPY_SRC). Not mappable itself — hence the staging buffer.
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vector_add.out"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    // Staging buffer we CAN map for reading: COPY_DST target + MAP_READ.
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vector_add.staging"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Shader + pipeline. `layout: None` lets wgpu derive the bind-group layout
    // from the WGSL, so the binding indices come straight from the shader.
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("vector_add.wgsl"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/vector_add.wgsl").into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("vector_add.pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // (3) bind group: group 0, bindings 0/1/2 = a, b, out. Order and indices
    // MUST match the WGSL @binding numbers.
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("vector_add.bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    // (4) encode the dispatch. ceil(n / WORKGROUP_SIZE) workgroups in x.
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("vector_add") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vector_add.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (n as u32).div_ceil(WORKGROUP_SIZE);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    // (5) copy result out and submit.
    encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, byte_len);
    queue.submit(Some(encoder.finish()));

    // Map the staging buffer and block until the GPU has actually finished.
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        // Ignore send errors: if the receiver is gone we're tearing down anyway.
        let _ = tx.send(res);
    });
    // poll(Wait) is the synchronous join: it drives the GPU work and the map
    // callback to completion. Without it the recv below could block forever.
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| GpuOpError::Backend(format!("device poll failed: {e:?}")))?;
    rx.recv()
        .map_err(|e| GpuOpError::Backend(format!("map callback dropped: {e}")))?
        .map_err(|e| GpuOpError::Backend(format!("buffer map failed: {e:?}")))?;

    // wgpu 30's get_mapped_range is fallible (the map could have raced a device
    // loss); propagate that as a Backend error so the cascade falls to CPU.
    let data = slice
        .get_mapped_range()
        .map_err(|e| GpuOpError::Backend(format!("get_mapped_range failed: {e:?}")))?;
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    // Drop the mapped view before unmapping (wgpu requires the view be gone).
    drop(data);
    staging.unmap();

    Ok(result)
}

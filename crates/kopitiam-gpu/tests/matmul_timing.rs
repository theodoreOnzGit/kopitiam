//! Does GPU offload actually pay for a transformer linear layer?
//!
//! Gated behind `KOPITIAM_GPU_BENCH=1` because it is a measurement, not an
//! assertion — timings vary per machine and a test that failed on a slow GPU
//! would be noise. Run it, read the numbers, decide:
//!
//! ```bash
//! KOPITIAM_GPU_BENCH=1 cargo test --release -p kopitiam-gpu \
//!   --test matmul_timing -- --nocapture
//! ```
//!
//! It reports the GPU path **including** upload and readback, because that is
//! what a caller actually pays. A kernel-only number would flatter the GPU and
//! answer a question nobody has.

use kopitiam_gpu::{matmul_nt_cpu, matmul_nt_gpu, GpuContext};
use std::time::Instant;

fn fill(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * 0.37 + seed).sin()).collect()
}

#[test]
fn report_gpu_versus_cpu_at_real_projection_shapes() {
    if std::env::var("KOPITIAM_GPU_BENCH").is_err() {
        println!("SKIPPED: set KOPITIAM_GPU_BENCH=1 to measure");
        return;
    }
    let Ok(ctx) = GpuContext::new() else {
        println!("SKIPPED: no GPU adapter on this machine");
        return;
    };

    // SmolLM2-360M's real projections, at both a prefill batch and a single
    // decode row — decode is the case the maintainer feels as chat lag.
    let cases: &[(&str, usize, usize, usize)] = &[
        ("decode  attn q/o   ", 1, 960, 960),
        ("decode  mlp gate/up", 1, 960, 2560),
        ("decode  mlp down   ", 1, 2560, 960),
        ("decode  output head", 1, 960, 49152),
        ("prefill attn q/o    (33 tok)", 33, 960, 960),
        ("prefill mlp gate/up (33 tok)", 33, 960, 2560),
    ];

    println!("\n{:<30} {:>12} {:>12} {:>10}", "case", "cpu", "gpu(total)", "speedup");
    for &(label, m, k, n) in cases {
        let x = fill(m * k, 0.1);
        let w = fill(n * k, 0.7);

        // Warm up both paths: the first GPU call compiles the pipeline, and the
        // first CPU call pays cache misses. Timing either cold measures setup,
        // not throughput.
        let _ = matmul_nt_cpu(&x, &w, m, k, n);
        let warm = matmul_nt_gpu(&ctx, &x, &w, m, k, n);

        let reps = 3;
        let t0 = Instant::now();
        for _ in 0..reps {
            let _ = matmul_nt_cpu(&x, &w, m, k, n);
        }
        let cpu = t0.elapsed() / reps;

        // A refusal is a RESULT, not a failure: the biggest matmul in the model
        // does not fit this adapter's binding limit, and the cascade is supposed
        // to hand it to the CPU. Reporting it keeps that visible in the table
        // instead of hiding it behind an `expect`.
        if let Err(e) = &warm {
            println!("{label:<30} {:>10.2?} {:>12} {:>10}", cpu, "n/a", "cpu-only");
            println!("{:<30}   -> {e}", "");
            continue;
        }

        let t1 = Instant::now();
        for _ in 0..reps {
            let _ = matmul_nt_gpu(&ctx, &x, &w, m, k, n).expect("gpu matmul");
        }
        let gpu = t1.elapsed() / reps;

        let speedup = cpu.as_secs_f64() / gpu.as_secs_f64().max(1e-9);
        println!("{label:<30} {:>10.2?} {:>10.2?} {speedup:>9.2}x", cpu, gpu);
    }
    println!(
        "\nGPU timings INCLUDE per-call weight upload + readback. A speedup below\n\
         1.0 means offloading this op as-is would make chat slower, not faster."
    );
}

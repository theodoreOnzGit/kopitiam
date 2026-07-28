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

use kopitiam_gpu::{
    block_q8_matmul_nt_cpu, matmul_nt_cpu, matmul_nt_gpu, GpuContext, ResidentBlockQ8Weight, BLOCK,
};

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

/// The op that is supposed to fix chat lag: the output head, held **quantized
/// and resident**, timed the way it would actually be used — upload once at
/// model load, dispatch once per token.
///
/// Reported separately from the table above because it answers a different
/// question. That table asks "does naive per-call offload pay?" (no). This asks
/// "does residency plus quantization pay for the one op that dominates decode?"
#[test]
fn report_resident_quantized_output_head() {
    if std::env::var("KOPITIAM_GPU_BENCH").is_err() {
        println!("SKIPPED: set KOPITIAM_GPU_BENCH=1 to measure");
        return;
    }
    let Ok(ctx) = GpuContext::new() else {
        println!("SKIPPED: no GPU adapter on this machine");
        return;
    };

    // SmolLM2-360M's output projection, exactly: vocab 49152 x hidden 960.
    let (m, k, n) = (1usize, 960usize, 49152usize);
    let quants: Vec<i8> = (0..n * k).map(|i| ((i as i32 * 37 % 255) - 128) as i8).collect();
    let scales: Vec<f32> =
        (0..n * (k / BLOCK)).map(|i| 0.001 + (i as f32 * 0.13).sin().abs() * 0.01).collect();
    let x: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.017).sin()).collect();

    let t_up = Instant::now();
    let resident = match ResidentBlockQ8Weight::upload(&ctx, &quants, &scales, n, k) {
        Ok(r) => r,
        Err(e) => {
            println!("output head could not be made resident: {e}");
            return;
        }
    };
    let upload = t_up.elapsed();

    // Warm both paths — the first dispatch compiles the pipeline.
    let _ = resident.matmul_nt(&ctx, &x, m).expect("warm");
    let _ = block_q8_matmul_nt_cpu(&x, &quants, &scales, m, k, n);

    let reps = 3;
    let t0 = Instant::now();
    for _ in 0..reps {
        let _ = block_q8_matmul_nt_cpu(&x, &quants, &scales, m, k, n);
    }
    let cpu = t0.elapsed() / reps;

    let t1 = Instant::now();
    for _ in 0..reps {
        let _ = resident.matmul_nt(&ctx, &x, m).expect("dispatch");
    }
    let gpu = t1.elapsed() / reps;

    println!("\noutput head {n} x {k}, quantized + resident");
    println!("  one-time upload : {upload:.2?}");
    println!("  cpu  per token  : {cpu:.2?}");
    println!("  gpu  per token  : {gpu:.2?}   ({:.2}x)", cpu.as_secs_f64() / gpu.as_secs_f64().max(1e-9));
    println!(
        "  upload amortises after ~{} tokens",
        (upload.as_secs_f64() / (cpu.as_secs_f64() - gpu.as_secs_f64()).max(1e-9)).ceil() as i64
    );
}

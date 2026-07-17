//! End-to-end proof of the GPU->CPU cascade.
//!
//! The guarantee this suite pins down: **kopitiam-gpu builds and its tests pass
//! on a machine with NO GPU** (headless CI, a GPU-less container). On such a
//! machine every op lands on the pure-Rust CPU path, and we assert the CPU
//! answer is correct. When a GPU *is* present we additionally assert the GPU
//! result equals the CPU result — but a missing GPU must NEVER fail the suite.
//!
//! This is why nothing here calls `unwrap()` on a GPU handle: the tests are
//! written so that "no adapter" is a normal, passing outcome.

use kopitiam_gpu::ops::{vector_add_cpu, VectorAdd, VectorAddInput};
use kopitiam_gpu::{Executor, GpuContext};

/// A small deterministic pair of vectors and their known sum.
fn sample() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // 300 elements: spans several 64-wide workgroups (300 = 4*64 + 44), so the
    // GPU path's bounds guard on the rounded-up tail is actually exercised.
    let a: Vec<f32> = (0..300).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..300).map(|i| (i as f32) * 0.5).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    (a, b, expected)
}

/// THE guarantee: the cascade returns the correct answer on any machine.
///
/// On a no-GPU box this runs entirely on CPU. On a GPU box it runs on GPU. Either
/// way the result must equal the known-good sum. This test must pass headless.
#[test]
fn cascade_is_always_correct() {
    let (a, b, expected) = sample();
    let exec = Executor::new();

    // Report which way the cascade went, so CI logs show it.
    if exec.has_gpu() {
        eprintln!(
            "cascade_is_always_correct: GPU present ({})",
            exec.gpu_context().unwrap().describe_backend()
        );
    } else {
        eprintln!("cascade_is_always_correct: no GPU, running on CPU path");
    }

    let got = exec.run(&VectorAdd, &VectorAddInput { a: &a, b: &b });
    assert_eq!(got, expected);
}

/// The CPU floor is correct on its own terms, independent of any GPU.
///
/// Uses `Executor::cpu_only()`, which forces the CPU path even on a machine that
/// *does* have a GPU — this is the "prove the fallback" test.
#[test]
fn cpu_fallback_is_correct() {
    let (a, b, expected) = sample();

    // Directly:
    assert_eq!(vector_add_cpu(&a, &b), expected);

    // And through the cascade with the GPU forced off:
    let exec = Executor::cpu_only();
    assert!(!exec.has_gpu());
    let got = exec.run(&VectorAdd, &VectorAddInput { a: &a, b: &b });
    assert_eq!(got, expected);
}

/// If (and only if) a GPU is present, the GPU kernel must agree with the CPU
/// twin bit-for-bit. If there is no GPU, this test skips its assertion and
/// passes — a missing GPU never fails CI.
#[test]
fn gpu_matches_cpu_when_present() {
    let (a, b, expected) = sample();

    let ctx = match GpuContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("gpu_matches_cpu_when_present: no GPU ({e}); skipping GPU==CPU check");
            return; // headless: pass without asserting the GPU path
        }
    };

    let gpu = kopitiam_gpu::ops::vector_add_gpu(&ctx, &a, &b)
        .expect("GPU present but the vector_add kernel failed");
    // Elementwise f32 add is exact, so demand exact equality, not a tolerance.
    assert_eq!(gpu, expected, "GPU result diverged from the known-good sum");
    assert_eq!(gpu, vector_add_cpu(&a, &b), "GPU and CPU results disagree");
}

/// Mismatched input lengths are a caller bug; the GPU path reports it rather
/// than silently truncating. (The cascade would still fall back to CPU, but the
/// direct GPU call surfaces the error — this checks that guard exists.)
#[test]
fn gpu_rejects_mismatched_lengths() {
    let ctx = match GpuContext::new() {
        Ok(ctx) => ctx,
        Err(_) => return, // no GPU: nothing to check here, pass
    };
    let a = [1.0f32, 2.0, 3.0];
    let b = [1.0f32, 2.0];
    let err = kopitiam_gpu::ops::vector_add_gpu(&ctx, &a, &b);
    assert!(err.is_err(), "unequal lengths should be rejected by the GPU path");
}

//! The compute operations, each implemented on BOTH the GPU and the CPU.
//!
//! Every op here implements [`crate::ComputeOp`] (a GPU kernel + a pure-Rust
//! twin) and is runnable through the [`crate::Executor`] cascade. The first cut
//! ships one demonstrator, [`vector_add`], enough to prove the whole GPU->CPU
//! machinery end to end; more kernels (tiled matmul, reductions, plot shaders)
//! land alongside it later.

pub mod vector_add;

pub use vector_add::{vector_add_cpu, vector_add_gpu, VectorAdd, VectorAddInput};

//! The compute operations, each implemented on BOTH the GPU and the CPU.
//!
//! Every op here implements [`crate::ComputeOp`] (a GPU kernel + a pure-Rust
//! twin) and is runnable through the [`crate::Executor`] cascade.
//!
//! * [`vector_add`] — the demonstrator that proves the cascade end to end.
//! * [`matmul_nt`] — `y = x @ w^T`, the shape a transformer linear layer
//!   actually performs, and where a decoder-only model spends nearly all of its
//!   time. Read that module's warning before assuming it makes anything faster:
//!   it re-uploads the weight on every call, and making GPU offload *pay* needs
//!   the weight resident on the device across calls.

pub mod matmul_nt;
pub mod vector_add;

pub use matmul_nt::{matmul_nt_cpu, matmul_nt_gpu, MatmulNt, MatmulNtInput};
pub use vector_add::{vector_add_cpu, vector_add_gpu, VectorAdd, VectorAddInput};

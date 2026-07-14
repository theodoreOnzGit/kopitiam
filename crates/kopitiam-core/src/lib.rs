//! Core primitives for the **Kopitiam Runtime** — KOPITIAM's CPU-first,
//! local-first, Rust-native inference engine.
//!
//! This crate is to the runtime what `kopitiam-ontology` is to the Semantic
//! Runtime: shared vocabulary, no logic, no allocation strategy, no I/O.
//! Every runtime crate (`kopitiam-tensor`, `kopitiam-loader`,
//! `kopitiam-runtime`, and the kernel/graph/scheduler crates that follow)
//! agrees on the types here, so none of them has to depend on each other
//! merely to name a dtype.
//!
//! What lives here:
//!
//! * [`DType`] — element types, including the block-quantized GGUF formats.
//! * [`Shape`] — dimensions, strides, reshape and broadcast rules.
//! * [`Device`] — where tensors live (CPU, and only CPU, by design).
//! * [`Error`] / [`Result`] — the runtime's single shared error type.
//!
//! # Relationship to `kopitiam-ai`
//!
//! `kopitiam-ai` owns the [`ModelAdapter`] boundary that the Semantic Runtime
//! talks to; this crate is far below that line. `kopitiam-workflow` should
//! never see a `DType`. The layering is:
//!
//! ```text
//! kopitiam-workflow  ->  kopitiam-ai (ModelAdapter)  ->  Kopitiam Runtime  ->  this crate
//! ```
//!
//! [`ModelAdapter`]: https://docs.rs/kopitiam-ai

mod device;
mod dtype;
mod error;
mod shape;

pub use device::Device;
pub use dtype::DType;
pub use error::{Error, Result};
pub use shape::Shape;

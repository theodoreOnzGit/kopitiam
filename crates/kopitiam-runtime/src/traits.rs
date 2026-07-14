//! The Model Runtime boundary: [`Model`] and [`Backend`].
//!
//! Everything above this line ([`crate::generate`], and eventually
//! `kopitiam-ai`'s real `ModelAdapter` — see this crate's parent epic) is
//! written against these two traits, not against [`crate::model::QwenModel`]
//! directly, the same way `kopitiam-workflow` is written against
//! `kopitiam_ai::ModelAdapter` rather than any one concrete adapter. That
//! mirroring is deliberate: it is the same "own the boundary, not the
//! implementation" shape CLAUDE.md's Architecture section asks every layer
//! of this platform to follow.

use kopitiam_core::{Device, Result};
use kopitiam_tensor::Tensor;

use crate::kv_cache::KvCache;

/// A causal (decoder-only) language model that can run a forward pass and
/// produce next-token logits.
///
/// # Why this trait exists with exactly one implementation
///
/// [`crate::model::QwenModel`] is the only [`Model`] today. That is not
/// reason enough on its own to add a trait — CLAUDE.md is explicit that
/// unused abstraction is a cost, not a virtue — but this one earns its
/// keep on two counts: [`crate::generate`] is written against `Model`, not
/// `QwenModel`, so it does not have to change when a second architecture
/// (a non-GQA LLaMA checkpoint, say, or a future non-transformer
/// architecture) arrives; and it marks precisely the seam a second
/// architecture would implement against, the same role
/// [`kopitiam_core::Device`]'s single `Cpu` variant plays for backends (see
/// that type's docs, which this trait's shape deliberately echoes).
pub trait Model {
    /// Runs a forward pass over `token_ids` — new tokens only (a multi-token
    /// prompt on the first call, ordinarily one token per call thereafter)
    /// — using and extending `cache`. Returns logits shaped
    /// `[token_ids.len(), vocab_size]`: one row of unnormalized
    /// per-vocabulary scores for every input position, in the same order as
    /// `token_ids`.
    fn forward(&self, token_ids: &[u32], cache: &mut KvCache) -> Result<Tensor>;

    /// The vocabulary size logits' last dimension is sized to.
    fn vocab_size(&self) -> usize;

    /// The context window `cache` should be constructed with via
    /// [`KvCache::new`] to run this model without hitting
    /// [`kopitiam_core::Error::IndexOutOfBounds`] from
    /// [`KvCache::append`](crate::kv_cache::KvCache::append).
    fn max_context(&self) -> usize;

    /// Builds a fresh, empty [`KvCache`] sized correctly for this model
    /// (layer count and context window). This is the only correct way to
    /// construct a cache for a given model — [`KvCache::new`] takes those
    /// two numbers as bare `usize`s and trusts the caller to get them
    /// right, which is fine for `KvCache`'s own unit tests but exactly the
    /// kind of easy-to-mismatch call [`crate::generate::generate`] (generic
    /// over any [`Model`], not just [`crate::model::QwenModel`]) should
    /// never have to get right by hand.
    fn new_cache(&self) -> KvCache;
}

/// The execution backend a [`Model`] runs its tensor ops on.
///
/// # Why this trait exists with exactly one implementation
///
/// Today every [`Model`] impl computes directly against
/// `kopitiam_tensor::Tensor`'s plain-CPU `f32` kernels — there is no
/// dispatch through this trait yet, and adding one purely to satisfy this
/// task brief's ask for a `Backend` seam, without a second backend that
/// would use it, is exactly the unnecessary abstraction CLAUDE.md warns
/// against. So this trait is deliberately minimal: it names the seam (what
/// a caller would ask a backend for — its [`Device`]) without inventing a
/// dispatch mechanism no code calls yet. A real second backend (a
/// SIMD-tuned kernel set, a threaded scheduler — see the parent epic's
/// Phase 2/3) is expected to grow this trait's surface *and* wire `Model`
/// impls to route their tensor ops through it; until then, [`CpuBackend`]
/// exists so [`crate::model::QwenModel`] has something concrete to report
/// through [`Model`]'s eventual `device()` accessor rather than hardcoding
/// [`Device::Cpu`] as a bare literal at every call site.
pub trait Backend {
    /// Where this backend's tensors live and compute.
    fn device(&self) -> Device;
}

/// The only [`Backend`] the Kopitiam Runtime implements: plain CPU
/// execution via `kopitiam-tensor`'s `f32` kernels. See [`Backend`]'s docs
/// for why this is a real (if minimal) type rather than a bare `Device`
/// literal.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn device(&self) -> Device {
        Device::Cpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_backend_reports_the_cpu_device() {
        assert_eq!(CpuBackend.device(), Device::Cpu);
    }
}

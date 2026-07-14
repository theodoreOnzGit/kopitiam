//! The key/value cache: what makes autoregressive decoding `O(1)` in
//! attention width per new token instead of `O(n)`.
//!
//! # Why this exists at all
//!
//! Without a cache, generating token `N` means re-running the *entire*
//! prefix `[0..N)` through every layer's attention from scratch, because
//! attention at position `N` needs the keys and values of every earlier
//! position. Generating a `T`-token completion this way costs `O(T^2)`
//! total attention work. A KV cache instead remembers each layer's
//! already-computed keys and values across calls, so decoding step `N`
//! only computes the *new* token's K/V and reuses the rest — `O(T)` total.
//! This is the difference between a usable CPU-side generation loop and one
//! that is quadratically, unusably slow past a few dozen tokens.
//!
//! # Design: one growable tensor per layer, not a fixed ring buffer
//!
//! [`KvCache::append`] concatenates a layer's new K/V onto what is already
//! cached ([`kopitiam_tensor::Tensor::concat`]), which reallocates and
//! copies the whole accumulated tensor on every call. That is the
//! "correct before fast" choice this crate's brief calls for: a
//! pre-sized ring buffer that writes new positions in place without
//! reallocating is a real optimization (and the natural next step once a
//! scheduler/memory-manager crate exists to own buffer reuse — see the
//! parent epic's Phase 2), but it is a performance change with no effect
//! on *what* gets computed, so it does not belong in the first working
//! version. [`KvCache::max_context`] is still enforced up front so a
//! caller finds out generation has hit the model's context window as a
//! clean error rather than silent truncation or an out-of-memory crash.

use kopitiam_core::{Error, Result};
use kopitiam_tensor::Tensor;

/// One layer's accumulated keys and values, each shaped
/// `[n_kv_heads, cached_len, head_dim]`. `None` until the first
/// [`KvCache::append`] for that layer.
struct LayerCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
}

/// Per-layer, growable key/value cache for one generation session.
///
/// A fresh [`KvCache`] holds zero cached positions. Every call to
/// [`crate::traits::Model::forward`] appends that call's new positions
/// to every layer's cache (via [`KvCache::append`], one call per layer) and
/// reads back the full accumulated K/V for attention. The cache is specific
/// to one generation session — call [`KvCache::new`] again (or
/// [`KvCache::clear`]) to start a new, unrelated prompt.
pub struct KvCache {
    layers: Vec<LayerCache>,
    max_context: usize,
}

impl KvCache {
    /// A fresh, empty cache for a model with `n_layers` transformer blocks
    /// and a `max_context`-token context window (from
    /// [`crate::config::QwenConfig::max_context`]).
    pub fn new(n_layers: usize, max_context: usize) -> Self {
        let layers = (0..n_layers).map(|_| LayerCache { k: None, v: None }).collect();
        Self { layers, max_context }
    }

    /// Number of positions currently cached (the same for every layer,
    /// since every [`KvCache::append`] call for a given forward pass
    /// appends the same number of new positions to each layer in turn).
    /// `0` for a fresh cache.
    pub fn len(&self) -> usize {
        self.layers
            .first()
            .and_then(|l| l.k.as_ref())
            .map(|k| k.shape().dims()[1])
            .unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn max_context(&self) -> usize {
        self.max_context
    }

    /// Drops every layer's cached K/V, returning this cache to its
    /// freshly-[`KvCache::new`]d state so it can be reused for a new,
    /// unrelated prompt without reallocating the outer `Vec`.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.k = None;
            layer.v = None;
        }
    }

    /// Appends `new_k`/`new_v` (each `[n_kv_heads, new_len, head_dim]`) to
    /// `layer`'s cache and returns the full accumulated `(k, v)`, each
    /// `[n_kv_heads, cached_len + new_len, head_dim]`.
    ///
    /// # Errors
    ///
    /// [`Error::IndexOutOfBounds`] if appending would make the cache exceed
    /// [`KvCache::max_context`] positions — the closest fit among
    /// `kopitiam-core`'s error variants (this crate cannot add a new one;
    /// see this crate's brief) for "position is past the end of the
    /// context window", read as `index` = the position that would have
    /// been reached and `len` = the window size that rejected it.
    pub(crate) fn append(&mut self, layer: usize, new_k: Tensor, new_v: Tensor) -> Result<(Tensor, Tensor)> {
        let new_len = new_k.shape().dims()[1];
        let existing_len = self.layers[layer].k.as_ref().map(|k| k.shape().dims()[1]).unwrap_or(0);
        let total_len = existing_len + new_len;
        if total_len > self.max_context {
            return Err(Error::IndexOutOfBounds { dim: 1, index: total_len, len: self.max_context });
        }

        let (full_k, full_v) = match (&self.layers[layer].k, &self.layers[layer].v) {
            (Some(prev_k), Some(prev_v)) => {
                (Tensor::concat(&[prev_k.clone(), new_k], 1)?, Tensor::concat(&[prev_v.clone(), new_v], 1)?)
            }
            _ => (new_k, new_v),
        };

        self.layers[layer].k = Some(full_k.clone());
        self.layers[layer].v = Some(full_v.clone());
        Ok((full_k, full_v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kv(seq: usize, fill: f32) -> Tensor {
        // shape [n_kv_heads=1, seq, head_dim=1] for simplicity.
        Tensor::from_f32(vec![fill; seq], [1, seq, 1]).unwrap()
    }

    #[test]
    fn a_fresh_cache_is_empty() {
        let cache = KvCache::new(2, 128);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn append_accumulates_across_calls_and_reports_the_new_length() {
        let mut cache = KvCache::new(1, 128);
        let (k, _v) = cache.append(0, kv(3, 1.0), kv(3, 2.0)).unwrap();
        assert_eq!(k.shape().dims(), &[1, 3, 1]);
        assert_eq!(cache.len(), 3);

        let (k, v) = cache.append(0, kv(1, 9.0), kv(1, 9.0)).unwrap();
        assert_eq!(k.shape().dims(), &[1, 4, 1]);
        assert_eq!(cache.len(), 4);
        // The earlier three positions are still there, unmodified, ahead
        // of the newly appended one.
        assert_eq!(k.to_vec_f32().unwrap(), vec![1.0, 1.0, 1.0, 9.0]);
        assert_eq!(v.to_vec_f32().unwrap(), vec![2.0, 2.0, 2.0, 9.0]);
    }

    #[test]
    fn exceeding_max_context_is_rejected() {
        let mut cache = KvCache::new(1, 4);
        cache.append(0, kv(4, 1.0), kv(4, 1.0)).unwrap();
        assert!(matches!(
            cache.append(0, kv(1, 1.0), kv(1, 1.0)),
            Err(Error::IndexOutOfBounds { .. })
        ));
    }

    #[test]
    fn clear_resets_every_layer_to_empty() {
        let mut cache = KvCache::new(2, 128);
        cache.append(0, kv(3, 1.0), kv(3, 1.0)).unwrap();
        cache.append(1, kv(3, 1.0), kv(3, 1.0)).unwrap();
        cache.clear();
        assert_eq!(cache.len(), 0);
        // A fresh append after clear() starts from zero again, not from 3.
        let (k, _) = cache.append(0, kv(2, 1.0), kv(2, 1.0)).unwrap();
        assert_eq!(k.shape().dims(), &[1, 2, 1]);
    }

    #[test]
    fn different_layers_are_independent() {
        let mut cache = KvCache::new(2, 128);
        cache.append(0, kv(3, 1.0), kv(3, 1.0)).unwrap();
        assert_eq!(cache.len(), 3); // len() reads layer 0
        // Layer 1 has never been appended to; appending 2 there should
        // start from zero, independent of layer 0's length.
        let (k, _) = cache.append(1, kv(2, 5.0), kv(2, 5.0)).unwrap();
        assert_eq!(k.shape().dims(), &[1, 2, 1]);
    }
}

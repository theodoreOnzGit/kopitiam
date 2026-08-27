//! A small, generic, bounded recency-order eviction policy -- the "which
//! rendered page gets thrown out to make room for a new one" decision a
//! continuous-scroll viewer needs (see gh-88's performance concern: opening
//! a long document must not hold every page's full-resolution texture in
//! GPU memory at once), kept deliberately separate from what is actually
//! cached: [`Lru`] only tracks **which key was touched when** and decides
//! what to evict once the tracked set grows past capacity. The caller
//! (`kpdf`'s page-texture cache) owns the real storage (an
//! `egui::TextureHandle` per key, in its own `HashMap`) -- that separation
//! is what keeps this struct free of any GUI dependency, and therefore
//! unit-testable without a display.

/// Bounded most-recently-used-order tracker over up to `capacity` keys.
///
/// Not a cache itself -- `K` only needs [`PartialEq`] + [`Clone`], and
/// lookups are a linear scan, which is entirely fine at the small
/// capacities (a double-digit number of on-screen-or-nearby pages) this is
/// meant for. A caller pairs this with its own `HashMap<K, V>` for the
/// actual cached values: call [`Lru::touch`] on every access (hit or fresh
/// insert), and remove whatever key it returns (if any) from that map.
#[derive(Debug, Clone)]
pub struct Lru<K> {
    capacity: usize,
    /// Recency order, oldest (least-recently-used) first, most-recently-used
    /// last.
    order: Vec<K>,
}

impl<K: PartialEq + Clone> Lru<K> {
    /// `capacity` is clamped to at least `1` -- a zero-capacity cache would
    /// evict the very key it was just asked to keep, which is never useful
    /// to a caller.
    pub fn new(capacity: usize) -> Self {
        Lru {
            capacity: capacity.max(1),
            order: Vec::new(),
        }
    }

    /// Record that `key` was just accessed (a cache hit, or a fresh insert)
    /// -- moves it to the most-recently-used end, and if that pushes the
    /// tracked set over capacity, evicts and returns the least-recently-used
    /// key that no longer fits. `key` itself was just used, so it is never
    /// the one evicted by its own touch.
    pub fn touch(&mut self, key: K) -> Option<K> {
        if let Some(pos) = self.order.iter().position(|k| *k == key) {
            self.order.remove(pos);
        }
        self.order.push(key);
        if self.order.len() > self.capacity {
            Some(self.order.remove(0))
        } else {
            None
        }
    }

    /// Whether `key` is currently tracked -- does not itself count as an
    /// access; call [`Lru::touch`] to record one.
    pub fn contains(&self, key: &K) -> bool {
        self.order.iter().any(|k| k == key)
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn touching_within_capacity_evicts_nothing() {
        let mut lru: Lru<i32> = Lru::new(3);
        assert_eq!(lru.touch(1), None);
        assert_eq!(lru.touch(2), None);
        assert_eq!(lru.touch(3), None);
        assert_eq!(lru.len(), 3);
    }

    #[test]
    fn touching_beyond_capacity_evicts_the_least_recently_used() {
        let mut lru: Lru<i32> = Lru::new(2);
        lru.touch(1);
        lru.touch(2);
        // 1 is now the least-recently-used of the two; adding a third must
        // evict it, not 2.
        assert_eq!(lru.touch(3), Some(1));
        assert!(lru.contains(&2));
        assert!(lru.contains(&3));
        assert!(!lru.contains(&1));
    }

    #[test]
    fn re_touching_an_existing_key_protects_it_from_eviction() {
        let mut lru: Lru<i32> = Lru::new(2);
        lru.touch(1);
        lru.touch(2);
        // Re-touching 1 makes 2 the least-recently-used instead.
        assert_eq!(lru.touch(1), None);
        assert_eq!(lru.touch(3), Some(2));
        assert!(lru.contains(&1));
        assert!(lru.contains(&3));
    }

    #[test]
    fn capacity_of_one_always_keeps_only_the_last_touched() {
        let mut lru: Lru<&str> = Lru::new(1);
        assert_eq!(lru.touch("a"), None);
        assert_eq!(lru.touch("b"), Some("a"));
        assert_eq!(lru.touch("c"), Some("b"));
        assert_eq!(lru.len(), 1);
    }

    #[test]
    fn zero_capacity_is_clamped_to_one() {
        let mut lru: Lru<i32> = Lru::new(0);
        assert_eq!(lru.touch(1), None);
        assert_eq!(lru.touch(2), Some(1));
    }

    #[test]
    fn touching_the_same_key_repeatedly_never_evicts_it() {
        let mut lru: Lru<i32> = Lru::new(2);
        lru.touch(1);
        for _ in 0..5 {
            assert_eq!(lru.touch(1), None);
        }
        assert_eq!(lru.len(), 1);
    }

    #[test]
    fn is_empty_reflects_the_tracked_set() {
        let mut lru: Lru<i32> = Lru::new(2);
        assert!(lru.is_empty());
        lru.touch(1);
        assert!(!lru.is_empty());
    }
}

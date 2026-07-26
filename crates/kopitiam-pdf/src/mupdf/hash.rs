//! Ported from MuPDF `source/fitz/hash.c` + `include/mupdf/fitz/hash.h`
//! (commit 19f1284, AGPL-3.0, © Artifex Software, Inc.), translated to Rust for
//! KOPITIAM (AGPL-3.0-only). Close adaptation: the algorithms and numeric
//! behaviour follow MuPDF; the code is re-expressed in idiomatic Rust. See
//! docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references").
//!
//! # What this is
//!
//! MuPDF's `fz_hash_table`: a simple hash table with **open addressing and
//! linear probing**, and **fixed-length binary keys** (one key length, chosen at
//! creation, for the whole table) mapping to a value. Unlike textbook open
//! addressing, its removal path backfills the probe run rather than leaving
//! tombstones, so repeated insert/remove cycles do not degrade.
//!
//! Here it becomes a generic [`HashTable<V>`] backed by a `Vec<Option<Entry>>`.
//! An occupied slot is `Some`, an empty slot is `None` -- that `Option` is the
//! safe stand-in for MuPDF's "`val == NULL` means empty" convention. No raw
//! pointers, no `unsafe`.
//!
//! # Translation notes (per AID-0051)
//!
//! * **The hash function is copied bit-for-bit.** It is the Jenkins
//!   one-at-a-time byte hash MuPDF uses (`hash()` in `hash.c`). C computes it in
//!   32-bit `unsigned`, where every `+`/`<<` wraps mod 2^32, so we use `u32` with
//!   `wrapping_add` and plain shifts. Reproducing it exactly is what makes probe
//!   positions -- and therefore iteration order and collision behaviour -- match
//!   MuPDF. See [`hash`], which is pinned by tests.
//! * **Insert does NOT overwrite.** `fz_hash_insert` returns the *existing*
//!   value when the key is already present and leaves the table unchanged (the
//!   header spells this out). [`HashTable::insert`] mirrors it: `None` on a fresh
//!   insert (the value moves into the table), `Some(&existing)` when the key was
//!   already there (the passed value is dropped, since the table never takes it).
//! * **Growth.** The table doubles once it is more than 80% full
//!   (`load > size * 8 / 10`), rehashing every live entry -- exactly MuPDF's
//!   trigger and factor. See [`HashTable::insert`] / [`HashTable::resize`].
//! * **Removal backfills.** [`HashTable::do_removal`] is a direct translation of
//!   `do_removal`, including the three-way wrap-around window test that decides
//!   whether a later probe entry must slide back into the hole.
//!
//! ## Not ported from this file (deliberately)
//!
//! `hash.c` threads an `fz_context` and an optional `FZ_LOCK` through every call,
//! and takes a `drop_val` callback so values can be freed when the table is
//! dropped. None of that survives translation: there is no global context, the
//! lock/threading concern is out of scope for this single-threaded port, and Rust
//! ownership drops the owned values ([`Vec`] of `Option<Entry>`) automatically,
//! which subsumes `drop_val`. The `Memento_*` allocation-debug labels are C-only.
//! The C API's `fz_hash_for_each` / `fz_hash_filter` callbacks become the
//! idiomatic [`HashTable::iter`] and [`HashTable::filter`].

/// `FZ_HASH_TABLE_KEY_LENGTH` -- MuPDF's compile-time cap on the fixed key length
/// (`hash.h`). A table's `keylen` may not exceed this.
pub const FZ_HASH_TABLE_KEY_LENGTH: usize = 48;

/// One occupied slot: the fixed-length key bytes and the owned value.
///
/// Mirrors `fz_hash_entry { unsigned char key[..]; void *val; }`. We store the
/// key as an owned `Vec<u8>` of exactly `keylen` bytes rather than a fixed 48-byte
/// array, but the semantics (compare/copy exactly `keylen` bytes) are identical.
struct Entry<V> {
    key: Vec<u8>,
    val: V,
}

/// MuPDF's `fz_hash_table`: open-addressing, linear-probe hash with fixed-length
/// binary keys.
///
/// See the module docs for the porting decisions. `V` is the value type; keys are
/// byte slices of the table's fixed `keylen`.
pub struct HashTable<V> {
    /// Fixed key length in bytes for every key in this table.
    keylen: usize,
    /// Number of occupied slots (`load` in `fz_hash_table`).
    load: usize,
    /// The slot array. `ents.len()` is MuPDF's `size`; `None` is an empty slot
    /// (MuPDF's `val == NULL`).
    ents: Vec<Option<Entry<V>>>,
}

/// The MuPDF byte-mixing hash (Jenkins one-at-a-time).
///
/// A faithful copy of `static unsigned hash(const unsigned char *s, int len)` in
/// `hash.c`. C evaluates this in 32-bit `unsigned`, where `+` and `<<` wrap mod
/// 2^32; `u32` with `wrapping_add` (and plain shifts, whose amounts are all < 32)
/// reproduces it exactly. Reproducing it byte-for-byte is what keeps probe
/// positions, iteration order and collision behaviour identical to MuPDF.
///
/// The hash covers the whole slice passed; callers hand it exactly `keylen` bytes.
// MuPDF: hash (hash.c:52)
fn hash(s: &[u8]) -> u32 {
    let mut val: u32 = 0;
    for &b in s {
        val = val.wrapping_add(b as u32);
        val = val.wrapping_add(val << 10);
        val ^= val >> 6;
    }
    val = val.wrapping_add(val << 3);
    val ^= val >> 11;
    val = val.wrapping_add(val << 15);
    val
}

impl<V> HashTable<V> {
    /// Create a new hash table with `initial_size` slots and a fixed key length.
    ///
    /// `initial_size` must be non-zero (MuPDF indexes `hash % size`, undefined
    /// for zero); `keylen` must not exceed [`FZ_HASH_TABLE_KEY_LENGTH`]. MuPDF's
    /// `fz_new_hash_table` throws `FZ_ERROR_ARGUMENT` on an over-long key; that is
    /// a caller precondition, so here it is an assertion.
    // MuPDF: fz_new_hash_table (hash.c:68)
    pub fn new(initial_size: usize, keylen: usize) -> Self {
        assert!(initial_size > 0, "hash table initial size must be non-zero");
        assert!(
            keylen <= FZ_HASH_TABLE_KEY_LENGTH,
            "hash table key length too large"
        );
        HashTable {
            keylen,
            load: 0,
            ents: (0..initial_size).map(|_| None).collect(),
        }
    }

    /// The fixed key length for this table.
    pub fn keylen(&self) -> usize {
        self.keylen
    }

    /// Number of live entries (MuPDF's `load`).
    pub fn len(&self) -> usize {
        self.load
    }

    /// Whether the table holds no entries.
    pub fn is_empty(&self) -> bool {
        self.load == 0
    }

    /// Current slot-array size (MuPDF's `size`). Grows as the table fills.
    pub fn capacity(&self) -> usize {
        self.ents.len()
    }

    /// Look up `key` and return a reference to its value, or `None`.
    ///
    /// Linear probe from the hash slot; an empty slot ends the search (a run of
    /// occupied slots is never broken by removal, which is what makes this sound).
    // MuPDF: fz_hash_find (hash.c:208)
    pub fn find(&self, key: &[u8]) -> Option<&V> {
        assert_eq!(key.len(), self.keylen, "hash key length mismatch");
        let size = self.ents.len();
        let mut pos = hash(key) as usize % size;
        loop {
            match &self.ents[pos] {
                None => return None,
                Some(e) if e.key.as_slice() == key => return Some(&e.val),
                Some(_) => pos = (pos + 1) % size,
            }
        }
    }

    /// Insert `key`/`val`, growing the table first if it is over 80% full.
    ///
    /// Returns `None` if the entry was newly inserted (and `val` moved into the
    /// table). If an entry with `key` already exists, the table is left unchanged
    /// and a reference to the **existing** value is returned; `val` is dropped.
    /// This is MuPDF's non-overwrite / return-existing contract.
    // MuPDF: fz_hash_insert (hash.c:230)
    pub fn insert(&mut self, key: &[u8], val: V) -> Option<&V> {
        if self.load > self.ents.len() * 8 / 10 {
            self.resize(self.ents.len() * 2);
        }
        self.do_insert(key, val)
    }

    /// The probing insert primitive, without the growth check.
    ///
    /// Also used by [`resize`](Self::resize) to re-home entries, which is why the
    /// growth check lives in [`insert`](Self::insert) and not here.
    // MuPDF: do_hash_insert (hash.c:117)
    fn do_insert(&mut self, key: &[u8], val: V) -> Option<&V> {
        assert_eq!(key.len(), self.keylen, "hash key length mismatch");
        let size = self.ents.len();
        let mut pos = hash(key) as usize % size;
        let existing = loop {
            match &self.ents[pos] {
                None => break false,
                Some(e) if e.key.as_slice() == key => break true,
                Some(_) => pos = (pos + 1) % size,
            }
        };
        if existing {
            // Legal, but should rarely happen: return the value already present.
            return self.ents[pos].as_ref().map(|e| &e.val);
        }
        self.ents[pos] = Some(Entry {
            key: key.to_vec(),
            val,
        });
        self.load += 1;
        None
    }

    /// Grow (or shrink) to `newsize` slots and rehash every live entry.
    ///
    /// A faithful port of `fz_resize_hash` minus the lock/threading dance: it
    /// refuses to shrink below 80% of the current load, allocates fresh empty
    /// slots, then reinserts each old entry via [`do_insert`](Self::do_insert).
    // MuPDF: fz_resize_hash (hash.c:153)
    fn resize(&mut self, newsize: usize) {
        let oldload = self.load;
        if newsize < oldload * 8 / 10 {
            // MuPDF warns "assert: resize hash too small" and leaves the table be.
            return;
        }
        let old = std::mem::take(&mut self.ents);
        self.ents = (0..newsize).map(|_| None).collect();
        self.load = 0;
        for entry in old.into_iter().flatten() {
            self.do_insert(&entry.key, entry.val);
        }
    }

    /// Remove the entry for `key`, if present.
    ///
    /// The value is dropped (Rust ownership). MuPDF warns on a missing key; here
    /// removing an absent key is simply a no-op.
    // MuPDF: fz_hash_remove (hash.c:274)
    pub fn remove(&mut self, key: &[u8]) {
        assert_eq!(key.len(), self.keylen, "hash key length mismatch");
        let size = self.ents.len();
        let mut pos = hash(key) as usize % size;
        loop {
            match &self.ents[pos] {
                None => return, // "assert: remove non-existent hash entry"
                Some(e) if e.key.as_slice() == key => {
                    self.do_removal(pos);
                    return;
                }
                Some(_) => {
                    pos += 1;
                    if pos == size {
                        pos = 0;
                    }
                }
            }
        }
    }

    /// Backfilling removal: empty slot `hole`, then slide back any later entry in
    /// the probe run whose ideal position makes it reachable through the hole.
    ///
    /// A direct translation of `do_removal`, including the three-way wrap-around
    /// window test. Keeping probe runs contiguous is what lets [`find`](Self::find)
    /// stop at the first empty slot.
    // MuPDF: do_removal (hash.c:238)
    fn do_removal(&mut self, mut hole: usize) {
        let size = self.ents.len();
        self.ents[hole] = None;

        let mut look = hole + 1;
        if look == size {
            look = 0;
        }

        while self.ents[look].is_some() {
            let code = hash(&self.ents[look].as_ref().unwrap().key) as usize % size;
            if (code <= hole && hole < look)
                || (look < code && code <= hole)
                || (hole < look && look < code)
            {
                self.ents[hole] = self.ents[look].take();
                hole = look;
            }

            look += 1;
            if look == size {
                look = 0;
            }
        }

        self.load -= 1;
    }

    /// Iterate over every live `(key, value)` pair, in slot order.
    ///
    /// The idiomatic replacement for `fz_hash_for_each`. Order follows the slot
    /// array, so it depends on the hash and on insertion/removal history exactly
    /// as MuPDF's iteration does.
    // MuPDF: fz_hash_for_each (hash.c:304)
    pub fn iter(&self) -> impl Iterator<Item = (&[u8], &V)> {
        self.ents
            .iter()
            .filter_map(|slot| slot.as_ref().map(|e| (e.key.as_slice(), &e.val)))
    }

    /// Remove every entry for which `callback(key, value)` returns `true`.
    ///
    /// The idiomatic replacement for `fz_hash_filter` (whose callback likewise
    /// returns "true = remove"). Because [`do_removal`](Self::do_removal) may move
    /// later slots back, the scan restarts from the beginning after each removal,
    /// exactly as MuPDF's `goto restart` does.
    // MuPDF: fz_hash_filter (hash.c:313)
    pub fn filter<F>(&mut self, mut callback: F)
    where
        F: FnMut(&[u8], &V) -> bool,
    {
        'restart: loop {
            for i in 0..self.ents.len() {
                let remove = match &self.ents[i] {
                    Some(e) => callback(e.key.as_slice(), &e.val),
                    None => false,
                };
                if remove {
                    self.do_removal(i);
                    continue 'restart;
                }
            }
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Keys must all be exactly `keylen` bytes; helper builds a keylen-4 key.
    fn k(bytes: &[u8; 4]) -> Vec<u8> {
        bytes.to_vec()
    }

    #[test]
    fn hash_matches_mupdf() {
        // Pinned outputs of MuPDF's `hash()` (hash.c:52), computed from the exact
        // C algorithm. If these drift, the byte-mixing hash has been altered.
        assert_eq!(hash(b""), 0);
        assert_eq!(hash(b"A"), 2181104624);
        assert_eq!(hash(b"MuPDF"), 1173734170);
        assert_eq!(hash(b"kopitiam"), 3669521261);
    }

    #[test]
    fn insert_find_remove() {
        let mut t: HashTable<i32> = HashTable::new(8, 4);
        assert!(t.is_empty());

        assert!(t.insert(&k(b"aaaa"), 1).is_none());
        assert!(t.insert(&k(b"bbbb"), 2).is_none());
        assert_eq!(t.len(), 2);

        assert_eq!(t.find(&k(b"aaaa")), Some(&1));
        assert_eq!(t.find(&k(b"bbbb")), Some(&2));
        assert_eq!(t.find(&k(b"cccc")), None);

        t.remove(&k(b"aaaa"));
        assert_eq!(t.find(&k(b"aaaa")), None);
        assert_eq!(t.find(&k(b"bbbb")), Some(&2));
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn insert_existing_returns_prior_value_and_does_not_overwrite() {
        let mut t: HashTable<i32> = HashTable::new(8, 4);
        assert!(t.insert(&k(b"key0"), 100).is_none());

        // Inserting the same key returns the existing value and leaves it intact.
        assert_eq!(t.insert(&k(b"key0"), 999), Some(&100));
        assert_eq!(t.find(&k(b"key0")), Some(&100));
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn remove_then_find_misses() {
        let mut t: HashTable<u32> = HashTable::new(4, 4);
        t.insert(&k(b"zzzz"), 7);
        assert_eq!(t.find(&k(b"zzzz")), Some(&7));
        t.remove(&k(b"zzzz"));
        assert_eq!(t.find(&k(b"zzzz")), None);
        // Removing an absent key is a no-op (MuPDF warns; we ignore).
        t.remove(&k(b"zzzz"));
        assert!(t.is_empty());
    }

    #[test]
    fn growth_past_initial_capacity_keeps_all_entries() {
        // Start tiny so we cross the 80% threshold and rehash several times.
        let mut t: HashTable<u32> = HashTable::new(2, 4);
        let n = 500u32;
        for i in 0..n {
            let key = i.to_le_bytes().to_vec();
            assert!(t.insert(&key, i).is_none());
        }
        assert_eq!(t.len(), n as usize);
        assert!(t.capacity() > 2, "table should have grown");

        // Every entry still findable after all the rehashing.
        for i in 0..n {
            let key = i.to_le_bytes().to_vec();
            assert_eq!(t.find(&key), Some(&i), "lost entry {i} after growth");
        }
    }

    #[test]
    fn collisions_resolve_via_linear_probe() {
        // Force many keys into a small fixed table so probe runs form, then make
        // sure lookups and a mid-run removal (with backfill) stay correct.
        let mut t: HashTable<u32> = HashTable::new(64, 4);
        let mut keys = Vec::new();
        for i in 0..40u32 {
            let key = i.to_le_bytes().to_vec();
            t.insert(&key, i * 10);
            keys.push(key);
        }
        for (i, key) in keys.iter().enumerate() {
            assert_eq!(t.find(key), Some(&(i as u32 * 10)));
        }

        // Remove every third key, then confirm the survivors are all still found
        // (this exercises do_removal's backfill of the probe runs).
        for (i, key) in keys.iter().enumerate() {
            if i % 3 == 0 {
                t.remove(key);
            }
        }
        for (i, key) in keys.iter().enumerate() {
            if i % 3 == 0 {
                assert_eq!(t.find(key), None);
            } else {
                assert_eq!(t.find(key), Some(&(i as u32 * 10)));
            }
        }
    }

    #[test]
    fn iter_visits_every_live_entry() {
        let mut t: HashTable<u32> = HashTable::new(8, 4);
        for i in 0..10u32 {
            t.insert(i.to_le_bytes().as_ref(), i);
        }
        let mut seen: Vec<u32> = t.iter().map(|(_, &v)| v).collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn filter_removes_matching_entries() {
        let mut t: HashTable<u32> = HashTable::new(8, 4);
        for i in 0..20u32 {
            t.insert(i.to_le_bytes().as_ref(), i);
        }
        // Drop even values.
        t.filter(|_key, &val| val % 2 == 0);
        assert_eq!(t.len(), 10);
        for i in 0..20u32 {
            let key = i.to_le_bytes().to_vec();
            if i % 2 == 0 {
                assert_eq!(t.find(&key), None);
            } else {
                assert_eq!(t.find(&key), Some(&i));
            }
        }
    }
}

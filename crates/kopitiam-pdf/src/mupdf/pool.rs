//! Ported from MuPDF `source/fitz/pool.c` + `include/mupdf/fitz/pool.h`
//! (commit 19f1284, AGPL-3.0, © Artifex Software, Inc.), translated to Rust for
//! KOPITIAM (AGPL-3.0-only). Close adaptation: the algorithms and numeric
//! behaviour follow MuPDF; the code is re-expressed in idiomatic Rust. See
//! docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references").
//!
//! # What `fz_pool` is
//!
//! `fz_pool` is a **block-chained bump allocator** (arena). It allocates
//! fixed-size blocks and hands out bump-pointer slices from the current block,
//! chaining to a fresh block when the current one can't fit the request;
//! everything is freed at once when the pool is dropped. There is *no*
//! per-object free. MuPDF builds the structured-text (`stext`) page in such a
//! pool, so this is a load-bearing foundation for the extraction port.
//!
//! The allocation strategy, copied exactly (see the `// MuPDF:` breadcrumbs):
//!
//! * blocks are `POOL_SIZE` = `4 << 10` = 4096 payload bytes (`pool.c:34`);
//! * a request `>= POOL_SELF` = `1 << 10` = 1024 bytes gets its **own** block
//!   (oversize path, `pool.c:35`, `pool.c:67`);
//! * a small request is rounded up to `FZ_POINTER_ALIGN_MOD` and bumped out of
//!   the current block; when `pos + size > end` a new `POOL_SIZE` block is
//!   chained on (`pool.c:88`..`pool.c:99`);
//! * memory is zero-initialised (MuPDF uses `fz_calloc`; the header promises
//!   "Block will be inited to 0's").
//!
//! # Where this safe Rust arena deliberately diverges from the C API
//!
//! MuPDF's API is pointer-based: `fz_pool_alloc` returns a raw `void*` the
//! caller casts and keeps. That does not map onto safe Rust, which cannot hand
//! out many coexisting borrows into a *growing* arena without `unsafe` (the
//! `typed-arena` trick). We preserve the **observable semantics and allocation
//! strategy** while re-expressing the interface:
//!
//! 1. **`void*` -> handles + slices.** The core [`Pool::alloc`] returns a
//!    [`Handle`] (block index + offset + length). Resolve it to bytes with
//!    [`Pool::bytes`] / [`Pool::bytes_mut`] (both fully safe; a `&[u8]` handle
//!    lookup can coexist with others, which is what lets callers hold several
//!    allocations at once — the C usage pattern). Convenience wrappers
//!    [`Pool::alloc_bytes`] and [`Pool::strdup`] return a borrow directly for
//!    the single-allocation case (they borrow `&mut self`, so only one is live
//!    at a time — use handles when you need several).
//! 2. **Bulk free on drop is implicit.** Dropping the owned `Vec<Vec<u8>>`
//!    frees every block at once, mirroring `fz_drop_pool` (`pool.c:125`). No
//!    per-object free exists; no explicit `Drop` impl is needed.
//! 3. **Alignment.** MuPDF rounds every request up to `FZ_POINTER_ALIGN_MOD`
//!    (4 on x86_64, `sizeof(void*)` elsewhere — `system.h:342`) so the bump
//!    offset stays aligned within a block. We reproduce that offset rounding
//!    exactly. We do *not* additionally guarantee the absolute address of a
//!    returned `&[u8]` is 4-aligned, because safe Rust never reinterprets these
//!    bytes as pointer-bearing structs (that reinterpretation — the reason C
//!    needs the guarantee — would itself require `unsafe`). The meaningful,
//!    preserved invariant is that successive allocations start at aligned
//!    *offsets*, identical to MuPDF's `pos` arithmetic.
//! 4. **Oversize block ordering is unobservable.** C links oversize blocks at
//!    the list *head*; we push to the blocks vector (ordering is irrelevant
//!    with index handles) and, exactly like C, leave the bump cursor untouched
//!    so later small allocations keep filling the current block.
//! 5. **`fz_pool_size` quirk preserved.** [`Pool::size`] mirrors `fz_pool_size`
//!    (`pool.c:120`) including its quirk that the *initial* block is not counted
//!    (`fz_new_pool` never adds it to `size`): each *subsequent* block adds
//!    `NODE_OVERHEAD + payload`, reproducing the "increases in lumps" wording of
//!    the header and matching C's values on a 64-bit build.
//! 6. **`strdup` keeps C's consumption, drops the NUL from the length.** C
//!    allocates `strlen+1` and copies the terminating NUL (`pool.c:105`). We
//!    reserve the same `len+1` from the bump (so block consumption matches C)
//!    but report `len = s.len()`, since a Rust `&str` carries its own length and
//!    needs no terminator.
//! 7. **No `fz_context` / no `Result`.** Allocation failure in C throws via the
//!    context; Rust `Vec` growth aborts on OOM, so nothing is threaded through —
//!    matching the infallible feel from the caller's side (AID-0051 §5).
//!
//! **No `unsafe` is used in this module.**
//!
//! ## Not ported from this file (deliberately)
//!
//! `pool.c` also carries the `fz_pool_array` family (`fz_new_pool_array_imp`,
//! `fz_pool_array_append`, `fz_pool_array_lookup`, `fz_pool_array_len`) — a
//! variable-length array *built on top of* the pool. It is a separate facility
//! and not on the immediate stext critical path, so it is out of scope here
//! (port it if and when a later module needs it), just as `geometry.rs` left out
//! the checked-integer and pixel-blend helpers.

// ---------------------------------------------------------------------------
// Constants (pool.c:34-35, system.h:342-348)
// ---------------------------------------------------------------------------

/// `POOL_SIZE` — default payload size of a pool block, `4 << 10` (`pool.c:34`).
const POOL_SIZE: usize = 4 << 10; // 4096

/// `POOL_SELF` — requests `>= POOL_SELF` get their own block, `1 << 10`
/// (`pool.c:35`).
const POOL_SELF: usize = 1 << 10; // 1024

/// `FZ_POINTER_ALIGN_MOD` — each request is rounded up to this before bumping.
/// MuPDF picks `4` on x86_64 and `sizeof(void*)` otherwise (`system.h:342`); we
/// mirror the same `#if` so the rounding matches on whatever target we build.
#[cfg(target_arch = "x86_64")]
const FZ_POINTER_ALIGN_MOD: usize = 4;
#[cfg(not(target_arch = "x86_64"))]
const FZ_POINTER_ALIGN_MOD: usize = core::mem::size_of::<usize>();

/// `offsetof(fz_pool_node, mem)` — the C block header (a single `next` pointer
/// before the flexible array). It exists here *only* to reproduce
/// `fz_pool_size`'s per-block "lump" accounting bit-for-bit on a 64-bit build;
/// the Rust blocks carry no such header.
const NODE_OVERHEAD: usize = core::mem::size_of::<usize>();

// ---------------------------------------------------------------------------
// Handle
// ---------------------------------------------------------------------------

/// An opaque reference to one allocation within a [`Pool`].
///
/// This is the safe stand-in for the raw `void*` that `fz_pool_alloc` returns.
/// A handle records the owning block, the byte offset within it, and the
/// requested length; resolve it back to bytes with [`Pool::bytes`] /
/// [`Pool::bytes_mut`]. A handle is only valid for the `Pool` that produced it,
/// and only until that pool is dropped (all storage is freed at once).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Handle {
    /// Index of the owning block in the pool's block list.
    block: usize,
    /// Byte offset of this allocation within its block.
    offset: usize,
    /// Number of usable bytes (as requested by the caller).
    len: usize,
}

impl Handle {
    /// Number of usable bytes this handle refers to.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether this handle refers to a zero-length allocation.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// ---------------------------------------------------------------------------
// Pool
// ---------------------------------------------------------------------------

/// A block-chained bump allocator. The safe Rust equivalent of `fz_pool`.
///
/// Allocate with [`alloc`](Pool::alloc) / [`alloc_bytes`](Pool::alloc_bytes) /
/// [`strdup`](Pool::strdup); everything is freed together when the `Pool` is
/// dropped (mirroring `fz_drop_pool` — there is no per-object free).
pub struct Pool {
    /// Owned block storage. Each block is zero-filled; index-based [`Handle`]s
    /// stay valid across pushes because we never remove or reorder blocks.
    blocks: Vec<Vec<u8>>,
    /// Index of the current bump block (the `tail` of MuPDF's chain).
    cur_block: usize,
    /// Bump cursor (byte offset) within `blocks[cur_block]` — MuPDF's `pos`.
    cur_pos: usize,
    /// Storage accounted to the pool, in the same "lumps" as `fz_pool_size`.
    size: usize,
}

impl Pool {
    /// Create a new, empty pool with one initial block.
    ///
    /// Faithful to `fz_new_pool`: the initial block exists but is **not** added
    /// to [`size`](Pool::size) (C's `fz_new_pool` zeroes `pool->size` and never
    /// counts the first block), so `pool.size()` reads `0` here.
    // MuPDF: fz_new_pool (pool.c:45)
    pub fn new() -> Self {
        Pool {
            blocks: vec![vec![0u8; POOL_SIZE]],
            cur_block: 0,
            cur_pos: 0,
            size: 0,
        }
    }

    /// Allocate `len` zero-initialised bytes and return a [`Handle`] to them.
    ///
    /// This is the safe analogue of `fz_pool_alloc`.
    // MuPDF: fz_pool_alloc (pool.c:80)
    pub fn alloc(&mut self, len: usize) -> Handle {
        if len >= POOL_SELF {
            return self.alloc_oversize(len);
        }

        // Round the request up to pointer alignment (pool.c:88).
        let rounded = (len + FZ_POINTER_ALIGN_MOD - 1) & !(FZ_POINTER_ALIGN_MOD - 1);

        // Chain a new block if the current one can't fit it (pool.c:90).
        if self.cur_pos + rounded > POOL_SIZE {
            self.blocks.push(vec![0u8; POOL_SIZE]);
            self.cur_block = self.blocks.len() - 1;
            self.cur_pos = 0;
            self.size += NODE_OVERHEAD + POOL_SIZE; // pool.c:96
        }

        let offset = self.cur_pos;
        self.cur_pos += rounded; // bump by the rounded size (pool.c:99)
        // Report the *requested* length; the rounding is invisible padding.
        Handle {
            block: self.cur_block,
            offset,
            len,
        }
    }

    /// Oversize path: give a large request its own block, without disturbing the
    /// bump cursor. C links it at the list head; ordering is irrelevant here.
    // MuPDF: fz_pool_alloc_oversize (pool.c:67)
    fn alloc_oversize(&mut self, len: usize) -> Handle {
        self.blocks.push(vec![0u8; len]);
        let block = self.blocks.len() - 1;
        self.size += NODE_OVERHEAD + len; // pool.c:75
        Handle {
            block,
            offset: 0,
            len,
        }
    }

    /// Allocate `len` zero-initialised bytes and return the fresh slice directly.
    ///
    /// Convenience over [`alloc`](Pool::alloc) + [`bytes_mut`](Pool::bytes_mut)
    /// for the common "allocate, then immediately fill" case. Because the slice
    /// borrows `&mut self`, only one such borrow is live at a time; when you need
    /// several allocations alive at once, keep [`Handle`]s and resolve them with
    /// [`bytes`](Pool::bytes).
    pub fn alloc_bytes(&mut self, len: usize) -> &mut [u8] {
        let h = self.alloc(len);
        self.bytes_mut(h)
    }

    /// Copy `s` into the pool and return the interned string slice.
    ///
    /// Safe analogue of `fz_pool_strdup`. Borrows `&mut self`; for several
    /// interned strings alive simultaneously use [`strdup_handle`](Pool::strdup_handle).
    // MuPDF: fz_pool_strdup (pool.c:103)
    pub fn strdup(&mut self, s: &str) -> &str {
        let h = self.strdup_handle(s);
        self.str_at(h)
    }

    /// Copy `s` into the pool and return a [`Handle`] to the interned bytes.
    ///
    /// Like C, this reserves `s.len() + 1` bytes (the extra byte is the NUL C
    /// would write; it stays zero here) so block consumption matches MuPDF, but
    /// the handle length is `s.len()` because a Rust `&str` carries its length.
    // MuPDF: fz_pool_strdup (pool.c:103)
    pub fn strdup_handle(&mut self, s: &str) -> Handle {
        let bytes = s.as_bytes();
        // C: n = strlen(s) + 1; p = fz_pool_alloc(ctx, pool, n) (pool.c:105-106).
        let reserved = self.alloc(bytes.len() + 1);
        let dst = &mut self.blocks[reserved.block][reserved.offset..reserved.offset + bytes.len()];
        dst.copy_from_slice(bytes); // C: memcpy(p, s, n); the trailing NUL is already 0.
        Handle {
            block: reserved.block,
            offset: reserved.offset,
            len: bytes.len(),
        }
    }

    /// Resolve a handle to its bytes (shared borrow — several may coexist).
    #[inline]
    pub fn bytes(&self, h: Handle) -> &[u8] {
        &self.blocks[h.block][h.offset..h.offset + h.len]
    }

    /// Resolve a handle to its bytes for mutation.
    #[inline]
    pub fn bytes_mut(&mut self, h: Handle) -> &mut [u8] {
        &mut self.blocks[h.block][h.offset..h.offset + h.len]
    }

    /// Resolve a handle produced by [`strdup_handle`](Pool::strdup_handle) (or
    /// any handle whose bytes are valid UTF-8) to a string slice.
    ///
    /// # Panics
    /// If the handle's bytes are not valid UTF-8. `strdup`/`strdup_handle`
    /// always store valid UTF-8, so this is infallible for those handles.
    #[inline]
    pub fn str_at(&self, h: Handle) -> &str {
        core::str::from_utf8(self.bytes(h)).expect("pool handle bytes are not valid UTF-8")
    }

    /// Storage currently accounted to the pool, in "lumps" (`fz_pool_size`).
    ///
    /// See the module-level note: the initial block is not counted (a faithful
    /// `fz_new_pool` quirk); each subsequent block adds `NODE_OVERHEAD + payload`.
    // MuPDF: fz_pool_size (pool.c:120)
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Default for Pool {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_pool_size_quirk() {
        // Faithful to fz_new_pool: the initial block is not counted.
        let pool = Pool::new();
        assert_eq!(pool.size(), 0);
    }

    #[test]
    fn fresh_alloc_is_zeroed() {
        let mut pool = Pool::new();
        let h = pool.alloc(16);
        assert_eq!(pool.bytes(h), &[0u8; 16]);
    }

    #[test]
    fn small_allocs_are_pointer_aligned() {
        // Each request rounds up to FZ_POINTER_ALIGN_MOD, so offsets stay aligned.
        let mut pool = Pool::new();
        let a = pool.alloc(3);
        let b = pool.alloc(3);
        assert_eq!(a.offset, 0);
        assert_eq!(b.offset, FZ_POINTER_ALIGN_MOD); // 3 rounded up to the align mod
        assert_eq!(b.offset % FZ_POINTER_ALIGN_MOD, 0);
    }

    #[test]
    fn many_small_allocs_span_blocks_without_overlap() {
        let mut pool = Pool::new();
        // 16 bytes rounds to 16; 4096/16 = 256 per block. 600 allocs => 3 blocks.
        const N: usize = 600;
        const SZ: usize = 16;
        let mut handles = Vec::with_capacity(N);
        for i in 0..N {
            let h = pool.alloc(SZ);
            // Write a distinct byte pattern into each region.
            pool.bytes_mut(h).fill((i & 0xff) as u8);
            handles.push(h);
        }
        // Read every region back; any overlap would have corrupted a neighbour.
        for (i, &h) in handles.iter().enumerate() {
            assert_eq!(pool.bytes(h), &[(i & 0xff) as u8; SZ][..], "region {i} clobbered");
        }
        // Two blocks were chained beyond the (uncounted) initial one.
        assert_eq!(pool.size(), 2 * (NODE_OVERHEAD + POOL_SIZE));
    }

    #[test]
    fn returned_slices_do_not_alias() {
        let mut pool = Pool::new();
        let h1 = pool.alloc(32);
        let h2 = pool.alloc(32);
        // Both are shared borrows, so they can be held at once.
        let s1 = pool.bytes(h1);
        let s2 = pool.bytes(h2);
        let r1 = s1.as_ptr_range();
        let r2 = s2.as_ptr_range();
        // Disjoint address ranges: end of one <= start of the other.
        assert!(r1.end <= r2.start || r2.end <= r1.start, "slices alias");
    }

    #[test]
    fn oversize_alloc_succeeds_and_keeps_own_block() {
        let mut pool = Pool::new();
        let small_before = pool.alloc(16);
        let big = pool.alloc(2000); // >= POOL_SELF -> own block
        pool.bytes_mut(big).fill(0xAB);
        let small_after = pool.alloc(16);

        // The oversize allocation got its own, exact-sized block.
        assert_eq!(big.len(), 2000);
        assert_eq!(pool.bytes(big), &[0xABu8; 2000][..]);
        assert_ne!(big.block, small_before.block);

        // The bump cursor was undisturbed: the small allocs stayed contiguous.
        assert_eq!(small_before.block, small_after.block);
        assert_eq!(small_before.offset, 0);
        assert_eq!(small_after.offset, 16);

        // size() counts the oversize block: NODE_OVERHEAD + its exact payload.
        assert_eq!(pool.size(), NODE_OVERHEAD + 2000);
    }

    #[test]
    fn strdup_round_trips() {
        let mut pool = Pool::new();
        let h_hello = pool.strdup_handle("hello");
        let h_empty = pool.strdup_handle("");
        // A long string exercises the oversize path too.
        let long = "x".repeat(2048);
        let h_long = pool.strdup_handle(&long);

        assert_eq!(pool.str_at(h_hello), "hello");
        assert_eq!(pool.str_at(h_empty), "");
        assert_eq!(pool.str_at(h_long), long);

        // The direct &str-returning form round-trips as well.
        let mut pool2 = Pool::new();
        assert_eq!(pool2.strdup("world"), "world");
    }

    #[test]
    fn capacity_grows_by_block() {
        let mut pool = Pool::new();
        assert_eq!(pool.size(), 0);
        // Fill exactly one block (256 * 16 == 4096); still no *extra* block yet.
        for _ in 0..256 {
            pool.alloc(16);
        }
        assert_eq!(pool.size(), 0);
        // One more small alloc forces a second block.
        pool.alloc(16);
        assert_eq!(pool.size(), NODE_OVERHEAD + POOL_SIZE);
    }

    #[test]
    fn drop_releases_no_leak_smoke() {
        // Repeatedly build and drop pools; bulk-free on drop must reclaim
        // everything (implicit Vec drop == fz_drop_pool). Smoke test: it runs.
        for _ in 0..1000 {
            let mut pool = Pool::new();
            for i in 0..500 {
                let h = pool.alloc(24);
                pool.bytes_mut(h).fill(i as u8);
            }
            pool.alloc(4096); // an oversize block each round, too
            drop(pool);
        }
    }
}

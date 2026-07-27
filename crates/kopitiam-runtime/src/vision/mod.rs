//! The vision path: running an image through a vision tower so a model can see
//! a page, not just read text about it.
//!
//! # Status — partial, and honest about it
//!
//! Only step 1 ([`preprocess`]) exists. The runtime is otherwise **GGUF
//! text-only**, so no vision tower can execute here yet. The remaining pieces,
//! in dependency order:
//!
//! | # | Piece | Notes |
//! |---|---|---|
//! | 1 | Preprocess (resize, normalize, patchify) | **done** — pure, testable without weights |
//! | 2 | Patch embedding | reshape + matmul; both ops exist |
//! | 3 | Position embeddings | learned table, gather + add |
//! | 4 | Tower blocks | needs a **bidirectional** attention path — `attention.rs` applies `causal_mask` unconditionally |
//! | 5 | Projector (vision dim → LLM dim) | matmul |
//! | 6 | Embedding splice into the token stream | **API change**, see below |
//! | 7 | `mmproj` tensor naming | needs a real file to pin key names |
//!
//! # The decision that gates the rest (step 6)
//!
//! [`crate::generate`] takes **token IDs** and embeds them internally. A vision
//! model cannot work that way: image patches become embeddings that have no
//! token ID at all, and they must be spliced into the sequence at the position
//! of an image placeholder. So the generation path has to accept
//! *pre-computed embeddings*, not only IDs.
//!
//! That is a change to the core text path every existing caller shares, which is
//! why it is called out here rather than quietly attempted: it wants a
//! deliberate decision (and probably an ADR) about whether `generate` grows an
//! embedding-input variant, or whether the vision path gets its own entry point
//! that converges lower down.
//!
//! # Why no `conv2d` is needed anywhere here
//!
//! Patch embedding is conventionally a `conv2d` with `kernel = stride =
//! patch_size`. For **non-overlapping** patches that is arithmetically identical
//! to flattening each tile and running one matmul — every input element lands in
//! exactly one window, so there is no sliding-window reuse to exploit.
//! `kopitiam-tensor` ships `matmul` but no `conv2d`, and this identity is why
//! that costs nothing. It would stop being true for a tower with
//! `stride < kernel`.

pub mod preprocess;

pub use preprocess::{
    gray_to_rgb8, preprocess, resize_square, PreprocessConfig, Rgb8, SIGLIP_MEAN, SIGLIP_STD,
};

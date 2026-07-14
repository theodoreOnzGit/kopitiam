//! Kopitiam Runtime: byte-level BPE tokenizer.
//!
//! This crate turns text into the token ids a GPT-2/GPT-3/GPT-4- or
//! Qwen-family model was trained on, and back. It is a from-scratch,
//! original Rust implementation -- not a wrapper around the `tokenizers`
//! crate -- because the whole point of the Kopitiam Runtime (see
//! `docs/ai-decisions/AID-0001`) is to own this layer rather than depend
//! on it, in keeping with this workspace's Pure Rust Core commitment.
//!
//! # Byte-level BPE, in one paragraph
//!
//! Classic word-level BPE needs an `<UNK>` token for anything outside its
//! training vocabulary -- a stray emoji, a typo, a byte sequence that
//! isn't valid text at all. Byte-level BPE sidesteps this entirely by
//! making the *256 possible byte values* the base alphabet instead of
//! characters or words: every possible input, valid UTF-8 or not, is some
//! sequence of bytes, and every byte already has a token id. There is
//! therefore no `<UNK>` in this crate's design and no failure mode for
//! encoding -- only [`Tokenizer::decode`] can fail, and only on a
//! genuinely unknown id (see its docs). This is the scheme GPT-2 through
//! GPT-4 and the Qwen family all use.
//!
//! # How the pieces fit together
//!
//! * [`byte_map`] -- the reversible byte <-> "printable Unicode character"
//!   mapping that lets a byte-level vocab be written as JSON text at all.
//!   Used only when loading/saving `tokenizer.json`; the runtime path
//!   never touches it.
//! * [`vocab`] -- [`vocab::Vocab`], the token (raw bytes) <-> id table.
//! * [`merges`] -- [`merges::MergeTable`], the ordered "which adjacent pair
//!   merges first, into what" rules learned during BPE training.
//! * [`pretokenize`] -- splits text into words/numbers/punctuation/
//!   whitespace-run chunks *before* BPE runs, matching the GPT-2 pattern
//!   without `regex`'s unsupported lookahead (see that module's docs --
//!   this is the single easiest place to introduce a silent correctness
//!   bug in a tokenizer).
//! * [`specials`] -- atomic special-token matching (`<|endoftext|>` and
//!   friends), so they are pulled out before pre-tokenization ever sees
//!   them.
//! * [`bpe`] -- [`bpe::BpeTokenizer`], the concrete [`Tokenizer`] that
//!   wires the above into `encode`/`decode`.
//! * [`loader`] -- parses the HuggingFace `tokenizer.json` shape into a
//!   [`bpe::BpeTokenizer`].
//!
//! # What this crate does *not* do
//!
//! No model inference, no chat templating, no attention-mask bookkeeping.
//! Those belong to `kopitiam-runtime`, which will code against the
//! [`Tokenizer`] trait rather than `BpeTokenizer` directly, the same way
//! it will code against traits from `kopitiam-tensor` and `kopitiam-loader`
//! rather than their concrete types.

pub mod bpe;
pub mod byte_map;
pub mod loader;
pub mod merges;
pub mod pretokenize;
pub mod specials;
pub mod vocab;

pub use bpe::BpeTokenizer;
pub use loader::from_tokenizer_json;

use kopitiam_core::Result;

/// What every tokenizer implementation in the Kopitiam Runtime must
/// provide. `kopitiam-runtime` codes against this trait, not
/// [`BpeTokenizer`] directly, so a future tokenizer scheme (e.g.
/// SentencePiece for a model family that needs it) can be dropped in
/// without touching the runtime's call sites.
pub trait Tokenizer {
    /// Encodes `text` into token ids. Byte-level implementations never
    /// fail here -- every possible `&str` (any valid UTF-8, including
    /// emoji, CJK, and arbitrary punctuation runs) is representable as
    /// some sequence of byte-derived tokens -- so this returning `Result`
    /// is about leaving room for non-byte-level implementations, not
    /// because `BpeTokenizer::encode` can actually fail today.
    fn encode(&self, text: &str) -> Result<Vec<u32>>;

    /// Decodes token ids back into text. Fails gracefully (rather than
    /// panicking) on an id outside the vocab; never fails just because the
    /// concatenated bytes happen not to be valid UTF-8 on their own (see
    /// [`bpe::BpeTokenizer::decode`]'s docs for why that case is handled
    /// losslessly-in-practice-but-not-by-contract via lossy UTF-8
    /// conversion instead of an error).
    fn decode(&self, ids: &[u32]) -> Result<String>;

    /// Total number of ids in the vocabulary, including any registered
    /// special tokens.
    fn vocab_size(&self) -> usize;
}

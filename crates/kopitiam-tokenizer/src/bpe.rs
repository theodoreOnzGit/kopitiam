//! [`BpeTokenizer`]: the byte-level BPE [`crate::Tokenizer`] implementation.
//!
//! This ties together the other modules in this crate into the pipeline a
//! GPT-2/Qwen-family tokenizer actually runs:
//!
//! ```text
//! text
//!   -> split off special tokens atomically      (specials::SpecialTokens)
//!   -> pre-tokenize the remaining text spans     (pretokenize::split)
//!   -> map each chunk's raw UTF-8 bytes to base-vocab ids
//!   -> repeatedly apply the highest-priority merge until none apply
//! ```
//!
//! Two design choices are worth calling out because they are easy to get
//! wrong by copying GPT-2's Python reference too literally:
//!
//! * **The byte-to-unicode alphabet never appears here.** [`crate::vocab::Vocab`]
//!   stores every token's *canonical raw bytes*, decoded once at load time
//!   by [`crate::loader`] (or supplied directly by [`BpeTokenizer::from_vocab_and_merges`]).
//!   So encoding a chunk is simply "look up the vocab id for each of its
//!   raw UTF-8 bytes" -- no per-token mapped-character round trip, at
//!   encode or decode time.
//! * **Decoding needs no special-token branch.** A special token's id maps
//!   to the raw UTF-8 bytes of its own content (see [`crate::specials`]),
//!   so plain byte concatenation already reconstructs it.

use crate::Tokenizer;
use crate::merges::MergeTable;
use crate::pretokenize;
use crate::specials::{Segment, SpecialTokens};
use crate::vocab::Vocab;
use kopitiam_core::{Error, Result};

/// A byte-level BPE tokenizer: the scheme used by the GPT-2/GPT-3/GPT-4 and
/// Qwen model families.
///
/// Build one via [`BpeTokenizer::from_vocab_and_merges`] (plain data --
/// what tests and a future GGUF loader use) or
/// [`crate::loader::from_tokenizer_json`] (HuggingFace `tokenizer.json`).
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    vocab: Vocab,
    /// `byte_ids[b as usize]` is the vocab id of the single-byte token for
    /// raw byte `b`, or `None` when this vocab simply never got a token for
    /// that byte. Precomputed once at construction (rather than doing a
    /// `vocab.id_of(&[b])` hash lookup per input byte) since every encode
    /// call starts by mapping raw bytes to ids.
    ///
    /// **Why `Option` and not a flat `[u32; 256]`.** The textbook says a
    /// byte-level BPE vocab must carry all 256 bytes as base tokens, and this
    /// crate used to enforce exactly that. Real published weights say
    /// otherwise, so we sked and had to relax it — see
    /// [`BpeTokenizer::missing_byte_tokens`] for the full story and for which
    /// bytes are actually reachable from valid UTF-8 input.
    byte_ids: [Option<u32>; 256],
    merges: MergeTable,
    /// Whether to prepend a single space to the input before
    /// pre-tokenizing, if it does not already start with one. See
    /// [`BpeTokenizer::with_add_prefix_space`] for why this exists and
    /// its (documented) scope limitation.
    add_prefix_space: bool,
    specials: SpecialTokens,
}

impl BpeTokenizer {
    /// Builds a tokenizer directly from a dense, id-ordered vocab and an
    /// ordered merge list -- no JSON, no byte-to-unicode mapping involved.
    ///
    /// `vocab_entries[i]` is the raw-byte token for id `i`. Ideally every
    /// one of the 256 possible single bytes appears somewhere in
    /// `vocab_entries` as its own one-byte token -- that completeness is
    /// what makes byte-level BPE total (every input encodable, no `<UNK>`).
    /// A vocab with holes is accepted anyway, because real shipped models
    /// have them; see [`BpeTokenizer::missing_byte_tokens`].
    pub fn from_vocab_and_merges(
        vocab_entries: Vec<Vec<u8>>,
        merge_pairs: Vec<(Vec<u8>, Vec<u8>)>,
    ) -> Result<Self> {
        Self::from_vocab(Vocab::from_entries(vocab_entries)?, merge_pairs)
    }

    /// Shared construction path for an already-built [`Vocab`]. Used by
    /// [`BpeTokenizer::from_vocab_and_merges`] (dense array input) and by
    /// [`crate::loader`] (sparse, id-keyed JSON input, which cannot go
    /// through `Vocab::from_entries`'s "index is the id" assumption).
    pub(crate) fn from_vocab(vocab: Vocab, merge_pairs: Vec<(Vec<u8>, Vec<u8>)>) -> Result<Self> {
        let merges = MergeTable::build(&merge_pairs, |b| vocab.id_of(b))?;

        // A missing single-byte token is NOT an error -- see
        // `missing_byte_tokens` for why refusing here was wrong.
        let mut byte_ids = [None; 256];
        for b in 0u16..=255 {
            let b = b as u8;
            byte_ids[b as usize] = vocab.id_of(&[b]);
        }

        Ok(Self {
            vocab,
            byte_ids,
            merges,
            add_prefix_space: false,
            specials: SpecialTokens::new(),
        })
    }

    /// Sets whether a leading space is prepended before pre-tokenizing an
    /// input that does not already start with one.
    ///
    /// GPT-2 treats a word at the very start of a sequence differently
    /// from the same word later on (the pre-tokenization pattern's
    /// `" ?"` prefix means "dog" and " dog" are different tokens), so
    /// most fast tokenizers add this synthetic leading space to make the
    /// first word behave like every other word. Whether it defaults to on
    /// varies by model family; a loaded `tokenizer.json`'s
    /// `pre_tokenizer.add_prefix_space` field (see [`crate::loader`])
    /// takes precedence over this default when present.
    ///
    /// Scope limitation: this crate applies the prefix space to the very
    /// start of the whole input passed to [`BpeTokenizer::encode`], before
    /// special-token splitting. The reference implementation technically
    /// scopes it to the first *non-special* span, which only differs when
    /// a special token is the very first thing in the input -- a rare
    /// enough case that documenting it here beats the added complexity of
    /// tracking it precisely.
    #[must_use]
    pub fn with_add_prefix_space(mut self, add_prefix_space: bool) -> Self {
        self.add_prefix_space = add_prefix_space;
        self
    }

    /// Registers a token that must always be matched atomically, never
    /// split by pre-tokenization or BPE -- e.g. Qwen's `<|endoftext|>`,
    /// `<|im_start|>`, `<|im_end|>`.
    ///
    /// `id` becomes the token's vocab id; if it falls outside the current
    /// vocab it grows the vocab to fit (special tokens are conventionally
    /// appended after the base vocab). Errors if `id` is already assigned
    /// to different content.
    pub fn add_special_token(&mut self, content: impl Into<String>, id: u32) -> Result<()> {
        let content = content.into();
        self.vocab.insert(id, content.clone().into_bytes())?;
        self.specials.register(content, id);
        Ok(())
    }

    /// The id of a registered special token by its exact content, if any.
    pub fn special_token_id(&self, content: &str) -> Option<u32> {
        self.specials.id_of(content)
    }

    /// The id of any exact token (special or ordinary) by its raw bytes.
    pub fn token_id(&self, token: &[u8]) -> Option<u32> {
        self.vocab.id_of(token)
    }

    /// The raw bytes this vocab has **no** single-byte token for, ascending.
    ///
    /// # Why a byte-level vocab can have holes at all
    ///
    /// Textbook byte-level BPE carries all 256 bytes as base tokens, which is
    /// what makes it total: any input at all is encodable, never mind valid
    /// UTF-8 or not, so no `<UNK>` is needed. This crate used to *enforce*
    /// that in [`BpeTokenizer::from_vocab`], returning
    /// [`Error::MalformedModel`] on the first missing byte.
    ///
    /// That check was too strict, and it blocked a real model. SmolLM2-360M-Instruct
    /// as HuggingFaceTB actually ships it (`smollm2-360m-instruct-q8_0.gguf`,
    /// 49152 tokens) is missing **21** of the 256:
    ///
    /// ```text
    /// 0x04 0x06 0x13 0x14 0x16 0x1d 0xc0 0xc1 0xf1 0xf2 0xf5..=0xff
    /// ```
    ///
    /// and it carries no SentencePiece-style `<0xNN>` byte-fallback spelling
    /// for them either, so there is genuinely nothing to fall back on — the
    /// tokens were never trained because those bytes never showed up in the
    /// corpus. `kopitiam models inspect <file.gguf>` prints exactly this
    /// verdict for any GGUF, which is how the diagnosis was made.
    ///
    /// # What a hole actually costs you (the precise part)
    ///
    /// [`Tokenizer::encode`] takes a `&str`, so its input is **always valid
    /// UTF-8**. Of the 21 bytes above, **13 cannot occur in valid UTF-8 at
    /// all** — `0xc0` / `0xc1` are the forbidden overlong lead bytes, and
    /// `0xf5..=0xff` are above the largest legal lead byte `0xf4`. Those are
    /// unreachable, so their absence costs nothing, full stop.
    ///
    /// The 8 remaining reachable ones are `0x04 0x06 0x13 0x14 0x16 0x1d`
    /// (C0 controls EOT, ACK, DC3, DC4, SYN, GS) and `0xf1` / `0xf2` (lead
    /// bytes for U+40000..=U+BFFFF, planes 4-11, which Unicode leaves entirely
    /// unassigned). So in practice only six rare control characters are
    /// affected.
    ///
    /// For those, [`Tokenizer::encode`] **drops the byte** rather than
    /// failing (see [`BpeTokenizer::chunk_to_symbols`]). Be clear-eyed about
    /// what that means: `decode(encode(s)) == s` holds for every `s` that
    /// contains none of the 8 reachable bytes, and for any `s` that does
    /// contain one, that byte is silently gone from the round trip. Refusing
    /// to build the tokenizer instead would trade that narrow, documented
    /// lossiness for "this model cannot be used at all" — a far worse deal
    /// for a model whose only sin is not having seen an EOT character.
    ///
    /// Callers who need to know: this method is the supported way to find out.
    #[must_use]
    pub fn missing_byte_tokens(&self) -> Vec<u8> {
        (0u16..=255)
            .filter(|&b| self.byte_ids[b as usize].is_none())
            .map(|b| b as u8)
            .collect()
    }

    /// Maps a pre-tokenized chunk's raw UTF-8 bytes to their base
    /// (single-byte) vocab ids.
    ///
    /// A byte this vocab has no token for is **dropped**, not substituted:
    /// there is no `<UNK>` in byte-level BPE to substitute *with*, and
    /// inventing one would put an id into the stream that the model was never
    /// trained on — worse than a missing control character.
    /// [`BpeTokenizer::missing_byte_tokens`] documents which bytes this can
    /// bite, and why the answer is "almost never".
    fn chunk_to_symbols(&self, chunk: &str) -> Vec<u32> {
        chunk.bytes().filter_map(|b| self.byte_ids[b as usize]).collect()
    }

    /// Repeatedly merges the highest-priority (lowest-rank) adjacent pair
    /// until no pair in `symbols` has a merge rule.
    ///
    /// This rescans every adjacent pair from scratch on every merge, so it
    /// is O(merges x symbols) per chunk -- quadratic in the chunk length
    /// in the worst case. That is the "naive but correct first"
    /// implementation named in this crate's design brief. The standard
    /// production technique (see e.g. the vendored reference's
    /// `models/bpe/word.rs`) keeps a doubly-linked list of symbols plus a
    /// binary heap of candidate merges keyed by rank, so each merge is
    /// found in O(log symbols) and applying it only re-examines the two
    /// new neighboring pairs instead of rescanning everything. That
    /// optimization is a pure performance change with no effect on the
    /// output and belongs here, behind this same function signature, once
    /// tokenizer throughput actually matters for the runtime.
    fn merge(&self, mut symbols: Vec<u32>) -> Vec<u32> {
        loop {
            let mut best: Option<(usize, u32, u32)> = None; // (position, rank, merged_id)
            for i in 0..symbols.len().saturating_sub(1) {
                if let Some(rule) = self.merges.get(symbols[i], symbols[i + 1])
                    && best.is_none_or(|(_, best_rank, _)| rule.rank < best_rank)
                {
                    best = Some((i, rule.rank, rule.merged_id));
                }
            }
            let Some((i, _, merged_id)) = best else {
                return symbols;
            };
            symbols[i] = merged_id;
            symbols.remove(i + 1);
        }
    }
}

impl Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let prefixed;
        let text = if self.add_prefix_space && !text.starts_with(' ') {
            prefixed = format!(" {text}");
            prefixed.as_str()
        } else {
            text
        };

        let mut ids = Vec::new();
        for segment in self.specials.split(text) {
            match segment {
                Segment::Special(id) => ids.push(id),
                Segment::Text(span) => {
                    for (start, end) in pretokenize::split(span) {
                        let symbols = self.chunk_to_symbols(&span[start..end]);
                        ids.extend(self.merge(symbols));
                    }
                }
            }
        }
        Ok(ids)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            let token_bytes = self.vocab.bytes_of(id).ok_or(Error::IndexOutOfBounds {
                dim: 0,
                index: id as usize,
                len: self.vocab.len(),
            })?;
            bytes.extend_from_slice(token_bytes);
        }
        // Lossy, not `from_utf8`: for any id sequence produced by this
        // tokenizer's own `encode`, the concatenated bytes are always
        // valid UTF-8 (they reconstruct the original valid `&str`), so the
        // lossy path never actually triggers there. But `decode` is also
        // called on model output during generation, where a caller may
        // reasonably decode a prefix of ids that splits a multi-byte
        // character mid-sequence (e.g. streaming token-by-token before a
        // multi-token emoji is complete); erroring on that would make
        // streaming decode unusable, so this matches the universal
        // tokenizer convention of never failing decode on well-formed ids.
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny hand-built vocab/merges pair standing in for a trained BPE
    /// model. Base bytes 'a', 'b', 'c' (plus every other byte, since
    /// this is the complete-alphabet case), and two merges: `("a","b")` ->
    /// `"ab"` (rank 0, highest priority) and `("ab","c")` -> `"abc"`
    /// (rank 1).
    fn tiny_tokenizer() -> BpeTokenizer {
        let mut vocab: Vec<Vec<u8>> = (0u16..=255).map(|b| vec![b as u8]).collect();
        vocab.push(b"ab".to_vec()); // id 256
        vocab.push(b"abc".to_vec()); // id 257
        let merges = vec![
            (b"a".to_vec(), b"b".to_vec()),
            (b"ab".to_vec(), b"c".to_vec()),
        ];
        BpeTokenizer::from_vocab_and_merges(vocab, merges).expect("valid tiny tokenizer")
    }

    fn byte_id(tok: &BpeTokenizer, b: u8) -> u32 {
        tok.token_id(&[b]).unwrap()
    }

    #[test]
    fn a_plus_b_then_ab_plus_c_merges_produce_the_single_token_abc() {
        let tok = tiny_tokenizer();
        let ids = tok.encode("abc").unwrap();
        assert_eq!(ids, vec![tok.token_id(b"abc").unwrap()]);
    }

    /// The property that actually needs the rank comparison, not just the
    /// existence check: build a vocab where the *leftmost* adjacent pair
    /// has the *lower priority* (higher rank number) merge, and a
    /// non-leftmost pair has the higher-priority (rank 0) merge. A
    /// tokenizer that greedily merges "the first pair it happens to find"
    /// while scanning left to right -- instead of comparing ranks across
    /// every candidate pair before merging any of them -- would merge
    /// `(a, b)` first because it is encountered first, producing `["ab",
    /// "c"]` with no further merges possible. The correct, rank-ordered
    /// result instead prefers `(b, c)` (rank 0, taught first during BPE
    /// training) even though it is not the leftmost pair, producing `["a",
    /// "bc"]`.
    #[test]
    fn merges_apply_in_rank_order_not_leftmost_scan_order() {
        let mut vocab: Vec<Vec<u8>> = (0u16..=255).map(|b| vec![b as u8]).collect();
        vocab.push(b"bc".to_vec()); // id 256
        vocab.push(b"ab".to_vec()); // id 257
        let merges = vec![
            (b"b".to_vec(), b"c".to_vec()), // rank 0: highest priority
            (b"a".to_vec(), b"b".to_vec()), // rank 1: lower priority
        ];
        let tok = BpeTokenizer::from_vocab_and_merges(vocab, merges).unwrap();

        let ids = tok.encode("abc").unwrap();
        let expected = vec![tok.token_id(b"a").unwrap(), tok.token_id(b"bc").unwrap()];
        assert_eq!(
            ids, expected,
            "expected rank-0 (b,c) to win over the leftmost-but-lower-priority (a,b) pair"
        );

        // Pin down exactly what the *wrong* leftmost-greedy answer would
        // have been, so this test fails loudly (not just "doesn't match
        // expected") if the implementation regresses to scan order.
        let wrong_leftmost_greedy_answer =
            vec![tok.token_id(b"ab").unwrap(), tok.token_id(b"c").unwrap()];
        assert_ne!(ids, wrong_leftmost_greedy_answer);
    }

    #[test]
    fn vocab_size_counts_every_registered_id() {
        let tok = tiny_tokenizer();
        assert_eq!(tok.vocab_size(), 258);
    }

    #[test]
    fn empty_string_encodes_to_empty() {
        let tok = tiny_tokenizer();
        assert_eq!(tok.encode("").unwrap(), Vec::<u32>::new());
    }

    #[test]
    fn decode_of_empty_is_empty() {
        let tok = tiny_tokenizer();
        assert_eq!(tok.decode(&[]).unwrap(), "");
    }

    #[test]
    fn decode_rejects_unknown_id_gracefully() {
        let tok = tiny_tokenizer();
        let err = tok.decode(&[99_999]).unwrap_err();
        assert!(matches!(err, Error::IndexOutOfBounds { .. }));
    }

    #[test]
    fn special_tokens_are_never_split_by_bpe() {
        let mut tok = tiny_tokenizer();
        tok.add_special_token("<|endoftext|>", 300).unwrap();
        let ids = tok.encode("ab<|endoftext|>c").unwrap();
        assert_eq!(ids.last().copied(), Some(byte_id(&tok, b'c')));
        assert!(ids.contains(&300));
        // The special token contributes exactly one id, not one id per
        // byte of "<|endoftext|>".
        assert_eq!(ids.iter().filter(|&&id| id == 300).count(), 1);
    }

    #[test]
    fn decode_of_encode_is_byte_exact_for_ascii() {
        let tok = tiny_tokenizer();
        let s = "abcabc hello world";
        assert_eq!(tok.decode(&tok.encode(s).unwrap()).unwrap(), s);
    }

    /// Same tiny vocab as [`tiny_tokenizer`], but with the single-byte tokens
    /// for `holes` left out — a scale model of what SmolLM2-360M actually
    /// ships. The merges still refer only to bytes that ARE present, exactly
    /// as a real trained merge list would.
    fn tokenizer_missing_bytes(holes: &[u8]) -> BpeTokenizer {
        let vocab: Vec<Vec<u8>> = (0u16..=255)
            .map(|b| b as u8)
            // Keep the id space dense: a hole becomes a harmless multi-byte
            // filler token rather than shifting every later id.
            .map(|b| if holes.contains(&b) { vec![b, b] } else { vec![b] })
            .collect();
        BpeTokenizer::from_vocab_and_merges(vocab, Vec::new()).expect("holes must not be fatal")
    }

    /// The regression this whole change exists for: a vocab with holes used
    /// to be rejected outright with `MalformedModel`, which took a perfectly
    /// usable SmolLM2-360M out of service. Building must now succeed.
    #[test]
    fn a_vocab_missing_some_single_byte_tokens_still_builds() {
        let tok = tokenizer_missing_bytes(&[0x04, 0x06, 0xff]);
        assert_eq!(tok.missing_byte_tokens(), vec![0x04, 0x06, 0xff]);
    }

    #[test]
    fn a_complete_vocab_reports_no_missing_bytes() {
        assert!(tiny_tokenizer().missing_byte_tokens().is_empty());
    }

    /// Ordinary text must be completely unaffected by the holes — that is the
    /// claim that makes tolerating them acceptable rather than merely lenient.
    #[test]
    fn text_that_avoids_the_holes_round_trips_exactly() {
        let tok = tokenizer_missing_bytes(&[0x04, 0x06, 0x1d]);
        let s = "the capital of France is";
        assert_eq!(tok.decode(&tok.encode(s).unwrap()).unwrap(), s);
    }

    /// And when the text DOES contain a hole byte, pin the documented
    /// behaviour precisely: that byte alone disappears, everything around it
    /// survives, and no bogus id is invented in its place.
    #[test]
    fn a_hole_byte_is_dropped_and_its_neighbours_survive() {
        let tok = tokenizer_missing_bytes(&[0x04]);
        let with_hole = tok.encode("ab\u{4}c").unwrap();
        let without = tok.encode("abc").unwrap();
        assert_eq!(
            with_hole, without,
            "byte 0x04 should vanish, leaving the surrounding text encoded identically"
        );
        assert_eq!(tok.decode(&with_hole).unwrap(), "abc");
    }
}

//! The token <-> id table shared by encode and decode.
//!
//! A vocab entry's canonical form is its raw bytes (`Vec<u8>`), never a
//! `String`. Byte-level BPE tokens are not guaranteed to be valid UTF-8 on
//! their own -- a single token can be one byte out of a multi-byte UTF-8
//! sequence -- so storing `String` would either panic or lossily mangle
//! data the moment a loader handed us a real vocab. Bytes are also what
//! [`crate::byte_map`] produces once a JSON-mapped token string has been
//! decoded, so `Vocab` never needs to think about the byte-to-unicode
//! alphabet again after construction.

use kopitiam_core::{Error, Result};
use std::collections::HashMap;

/// Bidirectional token <-> id table.
///
/// Ids are expected to be dense (0..len), matching how every BPE vocab
/// format in the wild (GPT-2 `vocab.json`, HF `tokenizer.json`, GGUF
/// metadata arrays) assigns them, but [`Vocab::insert`] tolerates inserting
/// an id past the current end (growing the table with placeholder empty
/// entries) because special tokens are conventionally appended with ids
/// starting right after the base vocab -- see [`crate::BpeTokenizer::add_special_token`].
#[derive(Debug, Clone, Default)]
pub struct Vocab {
    id_to_bytes: Vec<Vec<u8>>,
    bytes_to_id: HashMap<Vec<u8>, u32>,
}

impl Vocab {
    /// An empty vocab, built up one entry at a time via [`Vocab::insert`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a vocab from a dense, id-ordered list of token byte strings:
    /// `entries[i]` is the token for id `i`.
    ///
    /// This is the "plain constructor" path -- GGUF metadata and hand-built
    /// test vocabs already have exactly this shape (an array indexed by
    /// id), so there is no JSON or byte-map involved here at all.
    pub fn from_entries(entries: Vec<Vec<u8>>) -> Result<Self> {
        let mut vocab = Self::new();
        for (id, bytes) in entries.into_iter().enumerate() {
            vocab.insert(id as u32, bytes)?;
        }
        Ok(vocab)
    }

    /// Inserts (or overwrites-with-the-same-value) the token for `id`,
    /// growing the table if `id` is past the current end.
    ///
    /// Errors if `id` already names a *different* token -- that is always a
    /// caller bug (a malformed vocab file, or two special tokens registered
    /// under the same id), never a legitimate case to silently overwrite.
    pub fn insert(&mut self, id: u32, bytes: Vec<u8>) -> Result<()> {
        let idx = id as usize;
        if idx < self.id_to_bytes.len() {
            let existing = &self.id_to_bytes[idx];
            if !existing.is_empty() && existing != &bytes {
                return Err(Error::MalformedModel {
                    format: "bpe-vocab",
                    reason: format!(
                        "id {id} is already assigned to a different token \
                         ({existing:?} vs {bytes:?})"
                    ),
                });
            }
        } else {
            self.id_to_bytes.resize(idx + 1, Vec::new());
        }
        self.id_to_bytes[idx] = bytes.clone();
        self.bytes_to_id.insert(bytes, id);
        Ok(())
    }

    /// Number of ids in the table (including the highest id inserted, even
    /// if some lower slot was never filled -- see the doc comment on
    /// [`Vocab::insert`] about growth).
    pub fn len(&self) -> usize {
        self.id_to_bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.id_to_bytes.is_empty()
    }

    /// The raw bytes for `id`, or `None` if it is out of range or an
    /// unfilled placeholder slot left behind by growth.
    pub fn bytes_of(&self, id: u32) -> Option<&[u8]> {
        self.id_to_bytes
            .get(id as usize)
            .filter(|b| !b.is_empty())
            .map(Vec::as_slice)
    }

    /// The id for an exact token byte string, if present.
    pub fn id_of(&self, bytes: &[u8]) -> Option<u32> {
        self.bytes_to_id.get(bytes).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_entries() {
        let vocab = Vocab::from_entries(vec![b"a".to_vec(), b"b".to_vec(), b"ab".to_vec()])
            .expect("valid vocab");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.id_of(b"ab"), Some(2));
        assert_eq!(vocab.bytes_of(2), Some(b"ab".as_slice()));
        assert_eq!(vocab.id_of(b"missing"), None);
        assert_eq!(vocab.bytes_of(99), None);
    }

    #[test]
    fn insert_grows_past_the_end_for_appended_special_tokens() {
        let mut vocab = Vocab::from_entries(vec![b"a".to_vec(), b"b".to_vec()]).unwrap();
        vocab
            .insert(10, b"<|endoftext|>".to_vec())
            .expect("growth insert should succeed");
        assert_eq!(vocab.len(), 11);
        assert_eq!(vocab.id_of(b"<|endoftext|>"), Some(10));
        // The gap between id 2 and id 10 is left as unfilled placeholders.
        assert_eq!(vocab.bytes_of(5), None);
    }

    #[test]
    fn insert_rejects_conflicting_reassignment() {
        let mut vocab = Vocab::from_entries(vec![b"a".to_vec()]).unwrap();
        let err = vocab.insert(0, b"z".to_vec()).unwrap_err();
        assert!(matches!(err, Error::MalformedModel { .. }));
    }

    #[test]
    fn insert_tolerates_reinserting_the_same_value() {
        let mut vocab = Vocab::from_entries(vec![b"a".to_vec()]).unwrap();
        vocab.insert(0, b"a".to_vec()).expect("idempotent insert");
        assert_eq!(vocab.len(), 1);
    }
}

//! The ordered BPE merge rules.
//!
//! Training a BPE vocab produces merges in the order they were learned:
//! the first merge discovered is the most frequent pair in the training
//! corpus, so it has the highest priority at encode time. `tokenizer.json`
//! and classic `merges.txt` files preserve this as a plain ordered list.
//! [`MergeTable`] turns that list into "given a pair of adjacent token ids,
//! what is its priority (lower = merge first) and what id does merging it
//! produce?" -- the two questions [`crate::bpe::BpeTokenizer::encode`] asks
//! on every iteration of the merge loop.

use kopitiam_core::{Error, Result};
use std::collections::HashMap;

/// A learned merge's priority (rank) and the id it produces.
///
/// Rank is a `u32` purely to keep this struct small and `Copy`; ranks come
/// from a merge list's position (`enumerate()`), so `u32::MAX` merges is
/// not a real-world constraint (GPT-2/Qwen vocabs have on the order of
/// 10^5 merges).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MergeRule {
    pub rank: u32,
    pub merged_id: u32,
}

/// Maps an adjacent `(left_id, right_id)` pair to its [`MergeRule`].
#[derive(Debug, Clone, Default)]
pub struct MergeTable {
    rules: HashMap<(u32, u32), MergeRule>,
}

impl MergeTable {
    /// Builds a merge table from an ordered list of `(left_bytes,
    /// right_bytes)` pairs, resolving each side and the concatenated
    /// result to vocab ids via `lookup`.
    ///
    /// `lookup` is a closure rather than a `&Vocab` borrow so this stays
    /// decoupled from [`crate::vocab::Vocab`]'s concrete type -- the only
    /// thing a merge table needs from a vocab is "what id is this byte
    /// string?".
    pub fn build(
        merges: &[(Vec<u8>, Vec<u8>)],
        mut lookup: impl FnMut(&[u8]) -> Option<u32>,
    ) -> Result<Self> {
        let mut rules = HashMap::with_capacity(merges.len());
        for (rank, (left, right)) in merges.iter().enumerate() {
            let left_id = lookup(left).ok_or_else(|| Error::MalformedModel {
                format: "bpe-merges",
                reason: format!(
                    "merge rule {rank} references left-hand token {left:?}, which is not in the vocab"
                ),
            })?;
            let right_id = lookup(right).ok_or_else(|| Error::MalformedModel {
                format: "bpe-merges",
                reason: format!(
                    "merge rule {rank} references right-hand token {right:?}, which is not in the vocab"
                ),
            })?;
            let mut merged = left.clone();
            merged.extend_from_slice(right);
            let merged_id = lookup(&merged).ok_or_else(|| Error::MalformedModel {
                format: "bpe-merges",
                reason: format!(
                    "merge rule {rank} produces token {merged:?}, which is not in the vocab"
                ),
            })?;

            // A pair should only ever be taught once; a duplicate would
            // silently make one of the two ranks unreachable. Treat it as
            // a malformed merge list rather than quietly keeping the
            // first (or last) one.
            if rules
                .insert(
                    (left_id, right_id),
                    MergeRule {
                        rank: rank as u32,
                        merged_id,
                    },
                )
                .is_some()
            {
                return Err(Error::MalformedModel {
                    format: "bpe-merges",
                    reason: format!(
                        "pair ({left_id}, {right_id}) is taught by more than one merge rule"
                    ),
                });
            }
        }
        Ok(Self { rules })
    }

    /// The merge rule for an adjacent pair, if the pair has one.
    pub fn get(&self, left_id: u32, right_id: u32) -> Option<MergeRule> {
        self.rules.get(&(left_id, right_id)).copied()
    }

    pub fn len(&self) -> usize {
        self.rules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::Vocab;

    fn test_vocab() -> Vocab {
        Vocab::from_entries(vec![
            b"a".to_vec(),
            b"b".to_vec(),
            b"c".to_vec(),
            b"ab".to_vec(),
            b"abc".to_vec(),
        ])
        .unwrap()
    }

    #[test]
    fn resolves_ranks_and_merged_ids() {
        let vocab = test_vocab();
        let merges = vec![
            (b"a".to_vec(), b"b".to_vec()),
            (b"ab".to_vec(), b"c".to_vec()),
        ];
        let table = MergeTable::build(&merges, |b| vocab.id_of(b)).unwrap();
        assert_eq!(table.len(), 2);
        let ab = table.get(0, 1).unwrap(); // ("a","b")
        assert_eq!(ab.rank, 0);
        assert_eq!(ab.merged_id, 3); // "ab"
        let abc = table.get(3, 2).unwrap(); // ("ab","c")
        assert_eq!(abc.rank, 1);
        assert_eq!(abc.merged_id, 4); // "abc"
        assert!(table.get(0, 2).is_none()); // ("a","c") was never taught
    }

    #[test]
    fn rejects_merge_producing_a_token_outside_the_vocab() {
        let vocab = Vocab::from_entries(vec![b"a".to_vec(), b"b".to_vec()]).unwrap();
        let merges = vec![(b"a".to_vec(), b"b".to_vec())]; // "ab" is not in vocab
        let err = MergeTable::build(&merges, |b| vocab.id_of(b)).unwrap_err();
        assert!(matches!(err, Error::MalformedModel { .. }));
    }

    #[test]
    fn rejects_duplicate_pair() {
        let vocab = test_vocab();
        let merges = vec![
            (b"a".to_vec(), b"b".to_vec()),
            (b"a".to_vec(), b"b".to_vec()),
        ];
        let err = MergeTable::build(&merges, |b| vocab.id_of(b)).unwrap_err();
        assert!(matches!(err, Error::MalformedModel { .. }));
    }
}

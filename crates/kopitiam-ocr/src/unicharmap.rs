//! Ported from Tesseract `src/ccutil/unicharmap.cpp` +
//! `src/ccutil/unicharmap.h` (commit db0ec62, Apache-2.0, © 2006 Google Inc.,
//! Author: Thomas Kielbus), translated to Rust for KOPITIAM (AGPL-3.0-only).
//! Close adaptation: the byte-indexed 256-way tree and its lookup/insert walks
//! follow Tesseract exactly; the code is re-expressed in idiomatic Rust (owned
//! `Box` children in place of raw `new[]`/`delete[]`). See docs/ACKNOWLEDGEMENTS.md.
//!
//! # What this is
//!
//! [`Unicharmap`] is the UTF-8 → [`UnicharId`] lookup structure used by
//! `UNICHARSET` ([`crate::unicharset`]). It is a trie keyed one byte at a time:
//! each node has 256 slots (one per possible next byte), each slot carrying an
//! id (`-1` if none) and an optional child node. A unichar's id lives in the
//! slot reached by consuming all but the last byte of its representation and
//! indexing the final node by the last byte.

use crate::unichar::{INVALID_UNICHAR_ID, UNICHAR_LEN, UnicharId};

/// One node of the [`Unicharmap`] trie: 256 byte-indexed slots.
///
/// Tesseract: `UNICHARMAP::UNICHARMAP_NODE` (unicharmap.h:61). Each slot holds
/// an `id` (defaulting to [`INVALID_UNICHAR_ID`]) and, lazily, a child node
/// array.
#[derive(Debug)]
struct Node {
    children: Option<Box<[Node]>>,
    id: UnicharId,
}

impl Node {
    fn new() -> Self {
        Node {
            children: None,
            id: INVALID_UNICHAR_ID,
        }
    }

    /// A fresh array of 256 empty nodes (Tesseract: `new UNICHARMAP_NODE[256]`).
    fn new_array() -> Box<[Node]> {
        (0..256)
            .map(|_| Node::new())
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }
}

/// A store of unique unichar representations, each associated with one
/// [`UnicharId`].
///
/// Tesseract: `class UNICHARMAP` (unicharmap.h:29).
#[derive(Debug, Default)]
pub struct Unicharmap {
    nodes: Option<Box<[Node]>>,
}

impl Unicharmap {
    /// Create an empty map.
    pub fn new() -> Self {
        Unicharmap { nodes: None }
    }

    /// Insert `repr` and associate it with `id`. `repr` must be non-empty.
    ///
    /// Tesseract: `UNICHARMAP::insert` (unicharmap.cpp:59). Each byte indexes a
    /// node; the id is stored at the slot for the final byte.
    pub fn insert(&mut self, repr: &[u8], id: UnicharId) {
        if repr.is_empty() {
            return;
        }
        let mut current = &mut self.nodes;
        let mut index = 0;
        loop {
            if current.is_none() {
                *current = Some(Node::new_array());
            }
            let arr = current.as_mut().unwrap();
            let byte = repr[index] as usize;
            // `current_char[1] == '\0'`: this is the last byte of repr.
            if index + 1 >= repr.len() {
                arr[byte].id = id;
                return;
            }
            current = &mut arr[byte].children;
            index += 1;
        }
    }

    /// The id associated with `repr` (using at most `length` bytes), which MUST
    /// exist in the map. `length` must be non-zero.
    ///
    /// Tesseract: `UNICHARMAP::unichar_to_id` (unicharmap.cpp:36). Returns
    /// [`INVALID_UNICHAR_ID`] if the path runs off the tree.
    pub fn unichar_to_id(&self, repr: &[u8], length: usize) -> UnicharId {
        if length == 0 || repr.is_empty() {
            return INVALID_UNICHAR_ID;
        }
        let mut current = self.nodes.as_deref();
        let mut index = 0;
        loop {
            let arr = match current {
                Some(a) => a,
                None => return INVALID_UNICHAR_ID,
            };
            let byte = repr[index] as usize;
            if index + 1 >= length || index + 1 >= repr.len() {
                return arr[byte].id;
            }
            current = arr[byte].children.as_deref();
            index += 1;
        }
    }

    /// Whether `repr` (using at most `length` bytes) is present in the map.
    ///
    /// Tesseract: `UNICHARMAP::contains` (unicharmap.cpp:83).
    pub fn contains(&self, repr: &[u8], length: usize) -> bool {
        if repr.is_empty() {
            return false;
        }
        if length == 0 || length > UNICHAR_LEN {
            return false;
        }
        let mut current = self.nodes.as_deref();
        let mut index = 0;
        // Walk down while there is a deeper byte to consume.
        while let Some(arr) = current {
            if !(index + 1 < length && index + 1 < repr.len()) {
                break;
            }
            let byte = repr[index] as usize;
            current = arr[byte].children.as_deref();
            index += 1;
        }
        match current {
            Some(arr) => {
                let byte = repr[index] as usize;
                (index + 1 >= length || index + 1 >= repr.len()) && arr[byte].id >= 0
            }
            None => false,
        }
    }

    /// The minimum number of bytes of `repr` needed to reach a match, or `0`.
    ///
    /// Tesseract: `UNICHARMAP::minmatch` (unicharmap.cpp:106).
    pub fn minmatch(&self, repr: &[u8]) -> usize {
        if repr.is_empty() {
            return 0;
        }
        let mut current = self.nodes.as_deref();
        let mut index = 0;
        while let Some(arr) = current {
            if index >= repr.len() {
                break;
            }
            let byte = repr[index] as usize;
            if arr[byte].id >= 0 {
                return index + 1;
            }
            current = arr[byte].children.as_deref();
            index += 1;
        }
        0
    }

    /// Clear the map, dropping all data.
    ///
    /// Tesseract: `UNICHARMAP::clear` (unicharmap.cpp:123).
    pub fn clear(&mut self) {
        self.nodes = None;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_lookup_ascii_and_multibyte() {
        let mut map = Unicharmap::new();
        map.insert(b"A", 0);
        map.insert(b" ", 1);
        map.insert("标".as_bytes(), 2); // 3-byte CJK
        map.insert("가".as_bytes(), 3); // 3-byte Hangul

        assert_eq!(map.unichar_to_id(b"A", 1), 0);
        assert_eq!(map.unichar_to_id(b" ", 1), 1);
        assert_eq!(map.unichar_to_id("标".as_bytes(), 3), 2);
        assert_eq!(map.unichar_to_id("가".as_bytes(), 3), 3);

        assert!(map.contains(b"A", 1));
        assert!(map.contains("标".as_bytes(), 3));
        assert!(!map.contains(b"B", 1));
        // A shared 3-byte prefix that was never inserted is absent.
        assert!(!map.contains("中".as_bytes(), 3));
    }

    #[test]
    fn overwrite_id() {
        let mut map = Unicharmap::new();
        map.insert(b"x", 5);
        assert_eq!(map.unichar_to_id(b"x", 1), 5);
        map.insert(b"x", 9);
        assert_eq!(map.unichar_to_id(b"x", 1), 9);
    }

    #[test]
    fn minmatch_reports_prefix_length() {
        let mut map = Unicharmap::new();
        map.insert("标".as_bytes(), 2); // bytes E6 A0 87
        // No shorter prefix is a match, so minmatch is the full 3 bytes.
        assert_eq!(map.minmatch("标".as_bytes()), 3);
        // A single ASCII entry matches at its first (only) byte.
        map.insert(b"A", 0);
        assert_eq!(map.minmatch(b"A"), 1);
        // Unknown string: no match.
        assert_eq!(map.minmatch(b"Z"), 0);
    }

    #[test]
    fn empty_and_bounds() {
        let map = Unicharmap::new();
        assert!(!map.contains(b"A", 1));
        assert_eq!(map.unichar_to_id(b"A", 1), INVALID_UNICHAR_ID);
        // length guards.
        let mut map = Unicharmap::new();
        map.insert(b"A", 0);
        assert!(!map.contains(b"A", 0));
        assert!(!map.contains(b"A", UNICHAR_LEN + 1));
    }

    #[test]
    fn clear_empties() {
        let mut map = Unicharmap::new();
        map.insert(b"A", 0);
        map.clear();
        assert!(!map.contains(b"A", 1));
    }
}

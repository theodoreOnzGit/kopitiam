//! Stable, content-derived **segment identifiers** over the [`Document`]
//! model — the basis for translation memory (Task III-4,
//! `kopitiam_token_max.md` §13).
//!
//! ## The id is derived, never embedded (sidesteps §2.1)
//!
//! A segment's id is the FNV-1a content hash of its *normalized text*. It is
//! **computed on demand and written nowhere**: no anchor, no comment, nothing
//! is added to the converted Markdown. That is deliberate. `validate()`
//! measures a recovery ratio by comparing extracted vs. rendered character
//! counts (§2.1); any in-band id we emitted would inflate `rendered_chars`,
//! push the ratio above 1.0, and silently mask genuine content loss. Deriving
//! the id from content the document already contains keeps the ratio honest
//! while still giving every segment a stable name.
//!
//! ## Stable across runs and machines
//!
//! FNV-1a is fixed by its constants, so identical text hashes to an identical
//! id on every run, machine and toolchain — unlike `std::hash::DefaultHasher`
//! (SipHash), whose output is explicitly not stable across Rust releases. That
//! stability is the whole point: a translation cache keyed on these ids
//! (`kopitiam-workspace`) survives between sessions, so a revised document
//! only re-translates the segments whose text — and therefore id — changed.
//!
//! ## Granularity: one block, one segment
//!
//! Splitting happens at the already-structured [`Block`] level, never by
//! re-parsing rendered Markdown text. Each block that carries translatable
//! text becomes exactly one segment; blocks with no text (e.g. a figure with
//! no caption) are skipped. A block is a natural translation unit — a
//! paragraph, a heading, a list, a table — and using the existing structure
//! keeps the split deterministic and order-stable without a second parser.

use std::fmt;
use std::ops::Range;

use crate::{Block, Document};

/// FNV-1a 64-bit offset basis and prime. Identical to the constants
/// `kopitiam_workspace::content_hash` uses (Task II-5), so a [`SegmentId`]
/// produced here is byte-for-byte the same hex string the workspace
/// translation-memory store keys on. The tiny algorithm is duplicated rather
/// than shared to keep the two crates decoupled — no cross-crate dependency
/// edge, and therefore no `Cargo.lock` churn.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// A stable identifier for a translation segment: the FNV-1a content hash of
/// the segment's normalized text, as lower-case hex.
///
/// See the module docs for why this is derived from content rather than
/// embedded in the output.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SegmentId(String);

impl SegmentId {
    /// Computes the id of `text`: normalize it (trim, and collapse every run
    /// of whitespace to a single space), then hash the result with FNV-1a.
    ///
    /// Normalizing before hashing makes the id stable under trivial reflow —
    /// re-wrapping a paragraph, or a differently line-broken extraction of the
    /// same prose, keeps the same translation identity. Identical normalized
    /// text always yields an identical id.
    pub fn of(text: &str) -> Self {
        let normalized = normalize(text);
        let mut hash = FNV_OFFSET_BASIS;
        for &byte in normalized.as_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        SegmentId(format!("{hash:016x}"))
    }

    /// The id as a hex string — the exact key a translation cache stores it
    /// under.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SegmentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// One translation-sized unit of a [`Document`]: a run of source text with a
/// content-derived [`SegmentId`], its position in the segment sequence, and
/// the source block(s) it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Segment {
    /// 0-based position of this segment in the sequence returned by
    /// [`segments`]. Contiguous even when source blocks were skipped.
    pub index: usize,

    /// Stable content hash of [`Self::text`]. See [`SegmentId`].
    pub id: SegmentId,

    /// The segment's translatable source text, extracted from its block(s).
    pub text: String,

    /// Half-open range of source-block indices (into [`Document::blocks`])
    /// this segment covers. With one-block-per-segment granularity this is
    /// always a single block (`i..i + 1`); the range type leaves room to group
    /// several blocks into one unit later without changing the shape.
    pub block_range: Range<usize>,
}

/// Splits `document` into deterministic, order-stable translation segments.
///
/// Walks [`Document::blocks`] in order; each block that carries translatable
/// text becomes one [`Segment`] whose [`SegmentId`] is the content hash of
/// that text. Blocks with no text after normalization (e.g. a captionless
/// figure) are skipped. Called twice on the same document it returns identical
/// ids in identical order.
pub fn segments(document: &Document) -> Vec<Segment> {
    let mut segments = Vec::new();
    for (block_index, block) in document.blocks.iter().enumerate() {
        let text = block_text(block);
        if normalize(&text).is_empty() {
            // No translatable content (e.g. a figure with no caption): nothing
            // to name, cache, or translate.
            continue;
        }
        let index = segments.len();
        segments.push(Segment {
            index,
            id: SegmentId::of(&text),
            text,
            block_range: block_index..block_index + 1,
        });
    }
    segments
}

/// Extracts a block's translatable text. Structural blocks (lists, tables) are
/// flattened deterministically so the same block always yields the same
/// string, and therefore the same id.
fn block_text(block: &Block) -> String {
    match block {
        Block::Heading(heading) => heading.text.clone(),
        Block::Paragraph(paragraph) => paragraph.text.clone(),
        Block::List(list) => list.items.join("\n"),
        Block::Table(table) => {
            let mut lines = Vec::with_capacity(table.rows.len() + 1);
            if !table.headers.is_empty() {
                lines.push(table.headers.join(" | "));
            }
            for row in &table.rows {
                lines.push(row.join(" | "));
            }
            lines.join("\n")
        }
        Block::Figure(figure) => figure.caption.clone().unwrap_or_default(),
        Block::CodeBlock(code) => code.text.clone(),
        Block::Quote(quote) => quote.text.clone(),
    }
}

/// Trims `text` and collapses every run of whitespace to a single space, so
/// the id is stable under trivial reflow. Uses [`str::split_whitespace`],
/// which handles Unicode whitespace and needs no dependency.
fn normalize(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Heading, List, Metadata, Paragraph};

    fn doc(blocks: Vec<Block>) -> Document {
        Document {
            title: None,
            metadata: Metadata::default(),
            blocks,
            block_pages: Vec::new(),
            citations: Vec::new(),
        }
    }

    fn sample() -> Document {
        doc(vec![
            Block::Heading(Heading {
                level: 1,
                text: "Digital Twins".to_string(),
            }),
            Block::Paragraph(Paragraph {
                text: "A digital twin mirrors a physical asset.".to_string(),
            }),
            Block::List(List {
                ordered: false,
                items: vec!["sensor feeds".to_string(), "a model".to_string()],
            }),
        ])
    }

    #[test]
    fn segment_id_is_deterministic_and_content_sensitive() {
        assert_eq!(SegmentId::of("hello world"), SegmentId::of("hello world"));
        assert_ne!(SegmentId::of("hello world"), SegmentId::of("hello worlD"));
    }

    #[test]
    fn segment_id_is_stable_under_whitespace_reflow() {
        // Re-wrapping / re-spacing the same prose keeps the same id.
        assert_eq!(
            SegmentId::of("A digital twin  mirrors\na physical asset."),
            SegmentId::of("A digital twin mirrors a physical asset."),
        );
    }

    #[test]
    fn segment_id_known_answer_pins_the_hash() {
        // Pins the algorithm + normalization so a refactor cannot silently
        // change persisted ids (which would invalidate every cached
        // translation). Empty text normalizes to "" -> the FNV-1a basis.
        assert_eq!(
            SegmentId::of("   ").as_str(),
            format!("{:016x}", 0xcbf2_9ce4_8422_2325u64),
        );
        // A fixed non-empty answer, computed once and locked in.
        assert_eq!(SegmentId::of("digital twin").as_str(), "976c60764a911fc7");
    }

    #[test]
    fn segments_are_deterministic_and_order_stable() {
        let document = sample();
        let a = segments(&document);
        let b = segments(&document);
        assert_eq!(a, b);
        assert_eq!(a.len(), 3);
        assert_eq!(a[0].index, 0);
        assert_eq!(a[1].index, 1);
        assert_eq!(a[2].index, 2);
        // Ids follow content, in document order.
        assert_eq!(a[0].id, SegmentId::of("Digital Twins"));
        assert_eq!(a[2].id, SegmentId::of("sensor feeds\na model"));
    }

    #[test]
    fn changing_one_block_changes_exactly_one_segment_id() {
        let original = segments(&sample());

        // Revise exactly the middle block's text.
        let mut revised_doc = sample();
        revised_doc.blocks[1] = Block::Paragraph(Paragraph {
            text: "A digital twin mirrors a physical asset in real time.".to_string(),
        });
        let revised = segments(&revised_doc);

        assert_eq!(revised.len(), original.len());
        assert_eq!(revised[0].id, original[0].id); // heading unchanged
        assert_ne!(revised[1].id, original[1].id); // the edited block
        assert_eq!(revised[2].id, original[2].id); // list unchanged
    }

    #[test]
    fn blocks_without_text_are_skipped_but_indices_stay_contiguous() {
        use crate::Figure;
        let document = doc(vec![
            Block::Heading(Heading {
                level: 2,
                text: "Method".to_string(),
            }),
            Block::Figure(Figure {
                caption: None,
                image_path: None,
            }),
            Block::Paragraph(Paragraph {
                text: "We measured throughput.".to_string(),
            }),
        ]);
        let segs = segments(&document);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].index, 0);
        assert_eq!(segs[0].block_range, 0..1);
        assert_eq!(segs[1].index, 1);
        // The captionless figure at block 1 was skipped; this segment maps to
        // block 2.
        assert_eq!(segs[1].block_range, 2..3);
    }
}

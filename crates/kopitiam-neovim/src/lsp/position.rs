//! Converting between kvim's grapheme-cluster [`Position`](crate::core::Position)
//! and the position encodings the Language Server Protocol uses on the wire.
//!
//! # Why this file exists
//!
//! [`crate::core::Position`] is deliberately grapheme-indexed (see that
//! module's doc comment) because that is the unit a human moves the cursor
//! by. The Language Server Protocol was not designed with grapheme clusters
//! in mind at all: LSP 3.17 defines three interoperable `PositionEncodingKind`s
//! for `Position.character`, and a client and server negotiate which one to
//! use during `initialize`:
//!
//! * `"utf-8"` — count **UTF-8 code units**, i.e. bytes.
//! * `"utf-16"` — count **UTF-16 code units** (the historical LSP default,
//!   used by every server that predates the 3.17 negotiation capability, or
//!   that simply ignores it).
//! * `"utf-32"` — count **UTF-32 code units**, which the spec calls out as
//!   numerically identical to Unicode scalar values, i.e. plain `char`s.
//!
//! None of the three is grapheme clusters. A multi-`char` grapheme cluster
//! — a ZWJ emoji sequence, a base letter plus combining marks — is several
//! units under every one of these encodings, but exactly one grapheme to a
//! kvim cursor. Converting correctly means: find the byte range of the
//! requested grapheme cluster within its line, then measure (or locate) that
//! boundary in whichever unit the wire is speaking. Get this wrong and the
//! bug is silent: rename lands one or more graphemes away from the cursor on
//! any line containing CJK or emoji, and nowhere else — exactly the failure
//! mode this module exists to prevent.
//!
//! # A note on what `kopitiam-semantic` currently does
//!
//! [`kopitiam_semantic::RustAnalyzerSession::rename`] (used by
//! [`super::client`]) takes a `character` argument that its own
//! documentation and its caller (`apps/cli/src/rename.rs`) describe as a
//! "Unicode scalar value" (`char`) offset, and it asks the server to
//! negotiate `positionEncodings: ["utf-8", "utf-16"]`, treating a server's
//! choice of `"utf-8"` as meaning "char offsets, no surrogate-pair math".
//! Per the LSP 3.17 spec text quoted above, that is not what `"utf-8"`
//! means — `"utf-8"` means *byte* offsets; the encoding whose semantics
//! match "Unicode scalar value offset" is `"utf-32"`. On any server that
//! implements the spec literally (rust-analyzer among them) and negotiates
//! real byte-oriented `"utf-8"` positions, a char-count sent as `character`
//! will land in the wrong place on any line with multi-byte UTF-8 content
//! before the target column — the same silent-corruption failure mode this
//! module is designed to avoid, just one layer further down. This module
//! therefore treats [`PositionEncoding::Utf32`] as its own first-class
//! variant (`char` offsets) rather than conflating it with
//! [`PositionEncoding::Utf8`] (byte offsets), and [`super::client`] uses
//! `Utf32` specifically when talking to `RustAnalyzerSession::rename`. See
//! the top-level report for the exact upstream fix this suggests.

use unicode_segmentation::UnicodeSegmentation;

use crate::core::Position;

/// Which unit a `Position.character` value is measured in on the wire, per
/// LSP 3.17's `PositionEncodingKind`. `Utf32` is included even though the
/// LSP default fallback (used by servers that don't negotiate at all) is
/// [`Utf16`](Self::Utf16), because [`super::client`] needs to talk to
/// [`kopitiam_semantic::RustAnalyzerSession`] in exactly this unit — see the
/// module-level doc comment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PositionEncoding {
    /// UTF-8 code units (bytes). LSP's `"utf-8"`.
    Utf8,
    /// UTF-16 code units. LSP's `"utf-16"` — the historical default, sent by
    /// any server that does not negotiate a `positionEncoding` at all.
    Utf16,
    /// UTF-32 code units, i.e. Unicode scalar values (`char`s). LSP's
    /// `"utf-32"`.
    Utf32,
}

impl PositionEncoding {
    /// Parses the `capabilities.positionEncoding` string a server returns
    /// from `initialize`. Any value this client doesn't recognise — most
    /// importantly, a server that omits the field entirely because it
    /// predates LSP 3.17 — falls back to [`Self::Utf16`], which is correct:
    /// UTF-16 is the wire encoding the spec mandates when a server does not
    /// opt into anything else.
    pub fn from_capability(s: &str) -> Self {
        match s {
            "utf-8" => Self::Utf8,
            "utf-32" => Self::Utf32,
            _ => Self::Utf16,
        }
    }
}

/// A raw `{ line, character }` pair exactly as it appears on the LSP wire —
/// `character`'s unit depends on the negotiated [`PositionEncoding`], which
/// is why converting to/from [`Position`] always takes one explicitly rather
/// than being a `From`/`Into` impl (those can't smuggle in the extra
/// argument, and a silent default would be exactly the kind of implicit
/// behaviour that causes this class of bug).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LspPosition {
    pub line: u32,
    pub character: u32,
}

/// Converts a grapheme column within `line` to an offset in `encoding`'s
/// unit.
///
/// `col` is clamped to `line`'s grapheme length, matching
/// [`Position`]'s "one past the last cluster is end-of-line, and valid"
/// convention (see `text::grapheme::col_to_byte` for the sibling
/// implementation this mirrors on the text-engine side; this module does not
/// depend on it — see the crate-level report for why).
pub fn grapheme_col_to_unit(line: &str, col: usize, encoding: PositionEncoding) -> u32 {
    // The byte offset at which grapheme cluster `col` begins (or, if `col`
    // is at/past the end, `line.len()`).
    let byte_offset = line.grapheme_indices(true).nth(col).map(|(b, _)| b).unwrap_or(line.len());
    let prefix = &line[..byte_offset];
    match encoding {
        PositionEncoding::Utf8 => prefix.len() as u32,
        PositionEncoding::Utf16 => prefix.chars().map(|c| c.len_utf16() as u32).sum(),
        PositionEncoding::Utf32 => prefix.chars().count() as u32,
    }
}

/// Inverse of [`grapheme_col_to_unit`]: given a `character` offset in
/// `encoding`'s unit, returns the grapheme column it falls in.
///
/// An offset that lands strictly inside a multi-unit encoding of a single
/// `char` (e.g. a UTF-16 offset pointing between the two surrogate halves of
/// an astral-plane emoji) cannot correspond to any real boundary; rather
/// than panic, this treats that `char` as un-splittable and resolves to the
/// grapheme boundary *after* it — the same offset a target sitting exactly
/// on that `char`'s far edge would produce. This is the same "never panic,
/// always land somewhere defined" contract
/// [`text::grapheme`](crate::text) documents for its own mid-cluster case.
/// A well-behaved server never actually sends such an offset, so this path
/// is a safety net, not a normal one — see the
/// `mid_surrogate_pair_offset_does_not_panic` test.
pub fn unit_to_grapheme_col(line: &str, unit: u32, encoding: PositionEncoding) -> usize {
    let byte_offset = match encoding {
        PositionEncoding::Utf8 => (unit as usize).min(line.len()),
        PositionEncoding::Utf32 => line.char_indices().nth(unit as usize).map(|(b, _)| b).unwrap_or(line.len()),
        PositionEncoding::Utf16 => {
            let mut units_so_far = 0u32;
            let mut found = line.len();
            for (byte, ch) in line.char_indices() {
                if units_so_far >= unit {
                    found = byte;
                    break;
                }
                units_so_far += ch.len_utf16() as u32;
            }
            found
        }
    };
    // Grapheme index whose cluster starts at or after `byte_offset` — i.e.
    // the count of cluster-start boundaries strictly before it. This is
    // correct regardless of whether `byte_offset` points at a grapheme
    // boundary, mid-cluster (a combining mark's own `char`), or at
    // end-of-line (`byte_offset == line.len()`, which counts every cluster
    // and so yields exactly `grapheme_len(line)`, the documented
    // end-of-line column).
    line.grapheme_indices(true).take_while(|(b, _)| *b < byte_offset).count()
}

/// Converts a full [`Position`] to wire format, given the text of the line
/// it is on.
pub fn to_lsp(line: &str, pos: Position, encoding: PositionEncoding) -> LspPosition {
    LspPosition { line: pos.line as u32, character: grapheme_col_to_unit(line, pos.col, encoding) }
}

/// Converts a wire [`LspPosition`] back to a [`Position`], given the text of
/// the line it is on. The caller is responsible for locating that line
/// (`lsp.line` indexes the same way [`Position::line`] does — LSP lines are
/// also zero-based).
pub fn from_lsp(line: &str, lsp: LspPosition, encoding: PositionEncoding) -> Position {
    Position::new(lsp.line as usize, unit_to_grapheme_col(line, lsp.character, encoding))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cross-checks a grapheme-column conversion against ground truth
    /// computed independently via `str`'s own `encode_utf16`/`len`/`chars`
    /// on the substring up to that grapheme boundary — not by re-deriving
    /// the same arithmetic the code under test performs, so a shared bug in
    /// both would not go unnoticed.
    fn assert_unit(line: &str, col: usize, encoding: PositionEncoding, expected_via_stdlib: u32) {
        let got = grapheme_col_to_unit(line, col, encoding);
        assert_eq!(got, expected_via_stdlib, "line={line:?} col={col} encoding={encoding:?}");
        // And the inverse must land back on the same grapheme column.
        assert_eq!(
            unit_to_grapheme_col(line, got, encoding),
            col.min(line.graphemes(true).count()),
            "round trip failed: line={line:?} col={col} encoding={encoding:?}"
        );
    }

    fn prefix_of(line: &str, col: usize) -> String {
        line.graphemes(true).take(col).collect()
    }

    #[test]
    fn ascii_is_one_unit_per_column_in_every_encoding() {
        let line = "hello world";
        for col in 0..=line.len() {
            let prefix = prefix_of(line, col);
            assert_unit(line, col, PositionEncoding::Utf8, prefix.len() as u32);
            assert_unit(line, col, PositionEncoding::Utf16, prefix.encode_utf16().count() as u32);
            assert_unit(line, col, PositionEncoding::Utf32, prefix.chars().count() as u32);
        }
    }

    #[test]
    fn cjk_is_one_utf16_unit_but_three_utf8_bytes_per_character() {
        let line = "日本語abc";
        for col in 0..=6 {
            let prefix = prefix_of(line, col);
            assert_unit(line, col, PositionEncoding::Utf8, prefix.len() as u32);
            assert_unit(line, col, PositionEncoding::Utf16, prefix.encode_utf16().count() as u32);
            assert_unit(line, col, PositionEncoding::Utf32, prefix.chars().count() as u32);
        }
        // Concretely: "本" (col 1 -> col 2) is 1 UTF-16 unit and 3 UTF-8 bytes.
        assert_eq!(grapheme_col_to_unit(line, 1, PositionEncoding::Utf16), 1);
        assert_eq!(grapheme_col_to_unit(line, 1, PositionEncoding::Utf8), 3);
    }

    #[test]
    fn astral_emoji_needs_a_surrogate_pair_in_utf16() {
        // U+1F600 GRINNING FACE: one grapheme, one `char`, two UTF-16 code
        // units (a surrogate pair), four UTF-8 bytes.
        let line = "a\u{1F600}b";
        assert_eq!(line.graphemes(true).count(), 3);
        for col in 0..=3 {
            let prefix = prefix_of(line, col);
            assert_unit(line, col, PositionEncoding::Utf8, prefix.len() as u32);
            assert_unit(line, col, PositionEncoding::Utf16, prefix.encode_utf16().count() as u32);
            assert_unit(line, col, PositionEncoding::Utf32, prefix.chars().count() as u32);
        }
        // The emoji itself: col 1->2 is 2 UTF-16 units, 4 UTF-8 bytes, 1 char.
        assert_eq!(unit_to_grapheme_col(line, 1, PositionEncoding::Utf16), 1); // 'a' ends
        assert_eq!(grapheme_col_to_unit(line, 2, PositionEncoding::Utf16), 3); // 'a' (1) + emoji (2)
        assert_eq!(grapheme_col_to_unit(line, 2, PositionEncoding::Utf8), 5); // 'a' (1) + emoji (4)
        assert_eq!(grapheme_col_to_unit(line, 2, PositionEncoding::Utf32), 2); // 'a' (1) + emoji (1 char)
    }

    #[test]
    fn zwj_family_emoji_is_one_grapheme_but_five_chars_and_a_surrogate_heavy_utf16_run() {
        // Man + ZWJ + Woman + ZWJ + Girl: one grapheme cluster to a human,
        // five `char`s, three astral-plane emoji (2 UTF-16 units each) plus
        // two ZWJs (1 UTF-16 unit each, U+200D is in the BMP).
        let line = "👨\u{200D}👩\u{200D}👧";
        assert_eq!(line.graphemes(true).count(), 1, "the whole sequence must be ONE grapheme cluster");
        let expected_utf16 = line.encode_utf16().count() as u32;
        let expected_utf8 = line.len() as u32;
        let expected_utf32 = line.chars().count() as u32;
        assert_eq!(expected_utf16, 8, "3 emoji * 2 units + 2 ZWJ * 1 unit");
        assert_eq!(expected_utf32, 5);

        // Column 0 (before the cluster) is unit 0 in every encoding.
        assert_eq!(grapheme_col_to_unit(line, 0, PositionEncoding::Utf16), 0);
        // Column 1 (after the cluster, end-of-line) is the FULL width in
        // every encoding -- this is the case an off-by-N bug would blur
        // with "column 0", landing a rename one grapheme early.
        assert_eq!(grapheme_col_to_unit(line, 1, PositionEncoding::Utf16), expected_utf16);
        assert_eq!(grapheme_col_to_unit(line, 1, PositionEncoding::Utf8), expected_utf8);
        assert_eq!(grapheme_col_to_unit(line, 1, PositionEncoding::Utf32), expected_utf32);

        // And the inverse: any unit offset strictly between 0 and the full
        // width lands back on... there is no such thing as "half a
        // grapheme", so every offset in (0, expected_utf16] except an
        // internal one must resolve to column 0 or 1 only. Check the
        // documented endpoints round-trip.
        assert_eq!(unit_to_grapheme_col(line, 0, PositionEncoding::Utf16), 0);
        assert_eq!(unit_to_grapheme_col(line, expected_utf16, PositionEncoding::Utf16), 1);
    }

    #[test]
    fn combining_acute_accent_is_one_grapheme_two_chars() {
        let line = "e\u{0301}"; // 'e' + COMBINING ACUTE ACCENT
        assert_eq!(line.graphemes(true).count(), 1);
        assert_unit(line, 0, PositionEncoding::Utf16, 0);
        assert_unit(line, 1, PositionEncoding::Utf16, line.encode_utf16().count() as u32);
    }

    #[test]
    fn mixed_line_round_trips_at_every_grapheme_boundary_in_every_encoding() {
        let line = "a語👨\u{200D}👩\u{200D}👧e\u{0301}z";
        let n = line.graphemes(true).count();
        for encoding in [PositionEncoding::Utf8, PositionEncoding::Utf16, PositionEncoding::Utf32] {
            for col in 0..=n {
                let unit = grapheme_col_to_unit(line, col, encoding);
                assert_eq!(
                    unit_to_grapheme_col(line, unit, encoding),
                    col,
                    "round trip failed at col {col} encoding {encoding:?}"
                );
            }
        }
    }

    #[test]
    fn mid_surrogate_pair_offset_does_not_panic_and_snaps_past_the_undividable_char() {
        let line = "a\u{1F600}b"; // 'a' (unit 0), 2-unit emoji (units 1-2), 'b' (unit 3)
        // Unit 2 sits between the emoji's two surrogate halves -- there is no
        // real boundary there. Rather than panic, or arbitrarily round back
        // to before the emoji, this resolves the same as landing exactly on
        // 'b' (unit 3): the un-splittable emoji is treated as fully crossed.
        let mid_surrogate = unit_to_grapheme_col(line, 2, PositionEncoding::Utf16);
        let exactly_at_b = unit_to_grapheme_col(line, 3, PositionEncoding::Utf16);
        assert_eq!(mid_surrogate, exactly_at_b);
        assert_eq!(mid_surrogate, 2, "column 2 is 'b', the grapheme right after the emoji");
        // And it must differ from landing before the emoji (unit 1, column 1).
        assert_ne!(mid_surrogate, unit_to_grapheme_col(line, 1, PositionEncoding::Utf16));
    }

    #[test]
    fn an_offset_past_the_end_of_the_line_clamps_to_end_of_line_rather_than_panicking() {
        let line = "abc";
        assert_eq!(unit_to_grapheme_col(line, 999, PositionEncoding::Utf8), 3);
        assert_eq!(unit_to_grapheme_col(line, 999, PositionEncoding::Utf16), 3);
        assert_eq!(unit_to_grapheme_col(line, 999, PositionEncoding::Utf32), 3);
    }

    #[test]
    fn empty_line_has_a_single_valid_column_at_unit_zero() {
        assert_eq!(grapheme_col_to_unit("", 0, PositionEncoding::Utf16), 0);
        assert_eq!(unit_to_grapheme_col("", 0, PositionEncoding::Utf16), 0);
    }

    #[test]
    fn position_level_helpers_round_trip() {
        let line = "日本語 café";
        let pos = Position::new(4, 2);
        let lsp = to_lsp(line, pos, PositionEncoding::Utf16);
        assert_eq!(from_lsp(line, lsp, PositionEncoding::Utf16), pos);
    }

    #[test]
    fn capability_string_parsing_defaults_to_utf16() {
        assert_eq!(PositionEncoding::from_capability("utf-8"), PositionEncoding::Utf8);
        assert_eq!(PositionEncoding::from_capability("utf-32"), PositionEncoding::Utf32);
        assert_eq!(PositionEncoding::from_capability("utf-16"), PositionEncoding::Utf16);
        assert_eq!(PositionEncoding::from_capability("anything-else"), PositionEncoding::Utf16);
    }
}

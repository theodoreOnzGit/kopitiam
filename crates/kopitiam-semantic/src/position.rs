//! Converting between this crate's public position unit — Unicode scalar
//! value (`char`) offsets — and the units the Language Server Protocol
//! itself uses on the wire.
//!
//! # Why `char` offsets are the public unit
//!
//! Every public entry point in this crate ([`crate::RustAnalyzerSession::rename`],
//! [`crate::RustAnalyzerSession::code_actions`]) documents its `character`
//! argument as a Unicode scalar value offset — plain `chars()` indexing —
//! and `apps/cli`'s own `--character` flag promises callers the same thing.
//! That is a deliberate API choice: it lets every caller of this crate count
//! characters with the standard library and never think about the wire
//! protocol at all. This module is the one place that boundary is crossed:
//! [`crate::lsp_client::LspClient`] negotiates a wire encoding during
//! `initialize`, and [`crate::session::RustAnalyzerSession`] uses the
//! functions here to translate a `char` offset to that encoding before a
//! request goes out, and back to a `char` offset once a response (or a
//! server-initiated `workspace/applyEdit`) comes back.
//!
//! # What the wire actually uses
//!
//! LSP 3.17 defines three interoperable `PositionEncodingKind`s for
//! `Position.character`, negotiated between client and server during
//! `initialize` via the client's `general.positionEncodings` capability and
//! the server's `capabilities.positionEncoding` response field:
//!
//! * `"utf-8"`  — UTF-8 code units, i.e. **bytes**.
//! * `"utf-16"` — UTF-16 code units. This is the encoding every server must
//!   fall back to if it does not send `positionEncoding` in its response at
//!   all (e.g. because it predates LSP 3.17's negotiation capability) — and
//!   it is what rust-analyzer negotiates today.
//! * `"utf-32"` — UTF-32 code units, which the spec calls out as numerically
//!   identical to Unicode scalar values, i.e. plain `char`s.
//!
//! Of the three, **`"utf-32"` — not `"utf-8"` — is the one that matches
//! `char` offsets.** An earlier version of this crate got this backwards: it
//! asked the server to negotiate `positionEncodings: ["utf-8", "utf-16"]`
//! and, if the server picked `"utf-8"`, sent `char` counts straight over the
//! wire as if that were what `"utf-8"` meant. It never actually corrupted a
//! rename in practice only because rust-analyzer negotiates `"utf-16"`
//! today and every other code path fell back to refusing to run — "the
//! server happens to choose the encoding we mishandle least" is not a
//! defence, and a spec-literal server that picked real byte-oriented
//! `"utf-8"` would have silently sent a rename to the wrong offset on any
//! line with multi-byte UTF-8 content before the target column. See bead
//! `kopitiam-q7f` and the `regression_rename_target_after_multibyte_text_under_utf8_encoding`
//! test below for the exact failure this module now prevents.
//!
//! # A note on what this module deliberately does *not* do
//!
//! `kopitiam-neovim`'s `lsp::position` module (a sibling crate, not a
//! dependency of this one — see `CLAUDE.md`'s dependency-direction rules)
//! solves a related but different problem: it converts between LSP wire
//! encodings and a *grapheme-cluster*-indexed cursor position, because that
//! is the unit a human moves an editor cursor by. This crate has no cursor
//! and no editor UI; its public contract is `char` offsets end to end (see
//! above), so there is no grapheme layer here — only wire-unit conversion.
//! Do not "fix" this module to match that one; they solve different
//! problems on purpose.

/// Which unit a `Position.character` value is measured in on the wire, per
/// LSP 3.17's `PositionEncodingKind`, as negotiated by
/// [`crate::lsp_client::LspClient::spawn`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PositionEncoding {
    /// UTF-8 code units, i.e. bytes. LSP's `"utf-8"`.
    Utf8,
    /// UTF-16 code units. LSP's `"utf-16"` — the encoding every server must
    /// support and the one implied when a server sends no
    /// `positionEncoding` at all.
    Utf16,
    /// UTF-32 code units, i.e. Unicode scalar values (`char`s). LSP's
    /// `"utf-32"`.
    Utf32,
}

impl PositionEncoding {
    /// Parses the `capabilities.positionEncoding` string a server returns
    /// from `initialize` (`None` if the server omitted the field entirely).
    /// Anything this client doesn't recognise — including a missing field —
    /// falls back to [`Self::Utf16`], which is correct: UTF-16 is the wire
    /// encoding the spec mandates when a server does not opt into anything
    /// else.
    pub(crate) fn from_capability(s: Option<&str>) -> Self {
        match s {
            Some("utf-8") => Self::Utf8,
            Some("utf-32") => Self::Utf32,
            _ => Self::Utf16,
        }
    }
}

/// Converts a `char` column within `line` — this crate's public position
/// unit — to an offset in `encoding`'s wire unit.
///
/// `col` is clamped to `line`'s `char` length: a column at or past the end
/// of the line maps to the line's full width in `encoding`, matching LSP's
/// convention that `character == line length` is the valid end-of-line
/// position.
pub(crate) fn char_col_to_unit(line: &str, col: u32, encoding: PositionEncoding) -> u32 {
    // The byte offset at which `char` column `col` begins (or, if `col` is
    // at/past the end of the line, `line.len()`).
    let byte_offset = line.char_indices().nth(col as usize).map(|(b, _)| b).unwrap_or(line.len());
    let prefix = &line[..byte_offset];
    match encoding {
        PositionEncoding::Utf8 => prefix.len() as u32,
        PositionEncoding::Utf16 => prefix.chars().map(|c| c.len_utf16() as u32).sum(),
        PositionEncoding::Utf32 => prefix.chars().count() as u32,
    }
}

/// Inverse of [`char_col_to_unit`]: given a wire `unit` offset in
/// `encoding`, returns the `char` column it corresponds to.
///
/// A well-behaved server only ever sends offsets that fall on a real
/// boundary in the negotiated encoding. If one doesn't — a `"utf-16"`
/// offset landing between the two surrogate halves of an astral-plane
/// `char`, or a `"utf-8"` offset landing mid-way through a multi-byte
/// character — this never panics: it rounds down to the nearest valid
/// boundary rather than trusting server- or bug-controlled input to be
/// well-formed. A well-behaved server never actually sends such an offset,
/// so this path is a safety net, not a normal one.
pub(crate) fn unit_to_char_col(line: &str, unit: u32, encoding: PositionEncoding) -> u32 {
    let byte_offset = match encoding {
        PositionEncoding::Utf8 => {
            let mut boundary = (unit as usize).min(line.len());
            while boundary > 0 && !line.is_char_boundary(boundary) {
                boundary -= 1;
            }
            boundary
        }
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
    line[..byte_offset].chars().count() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `char` column of `line`, from `0` to the line's full length,
    /// must round-trip through `char_col_to_unit` -> `unit_to_char_col` back
    /// to itself, in every encoding. This is the property that makes the
    /// conversion safe to use in both directions (request and response).
    fn assert_round_trips(line: &str) {
        let n = line.chars().count() as u32;
        for encoding in [PositionEncoding::Utf8, PositionEncoding::Utf16, PositionEncoding::Utf32] {
            for col in 0..=n {
                let unit = char_col_to_unit(line, col, encoding);
                assert_eq!(unit_to_char_col(line, unit, encoding), col, "line={line:?} col={col} encoding={encoding:?}");
            }
        }
    }

    #[test]
    fn ascii_is_one_unit_per_char_in_every_encoding() {
        let line = "hello world";
        for col in 0..=line.chars().count() as u32 {
            let prefix: String = line.chars().take(col as usize).collect();
            assert_eq!(char_col_to_unit(line, col, PositionEncoding::Utf8), prefix.len() as u32);
            assert_eq!(char_col_to_unit(line, col, PositionEncoding::Utf16), prefix.encode_utf16().count() as u32);
            assert_eq!(char_col_to_unit(line, col, PositionEncoding::Utf32), prefix.chars().count() as u32);
        }
        assert_round_trips(line);
    }

    #[test]
    fn cjk_is_one_utf16_and_utf32_unit_but_three_utf8_bytes_per_character() {
        let line = "日本語abc";
        // "本" (col 1 -> col 2): 1 UTF-16 unit, 1 UTF-32 unit, 3 UTF-8 bytes.
        assert_eq!(char_col_to_unit(line, 1, PositionEncoding::Utf16), 1);
        assert_eq!(char_col_to_unit(line, 1, PositionEncoding::Utf8), 3);
        assert_eq!(char_col_to_unit(line, 1, PositionEncoding::Utf32), 1);
        assert_round_trips(line);
    }

    #[test]
    fn astral_emoji_needs_a_surrogate_pair_in_utf16_but_is_one_char() {
        // U+1F600 GRINNING FACE: one `char`, two UTF-16 code units (a
        // surrogate pair), four UTF-8 bytes.
        let line = "a\u{1F600}b";
        assert_eq!(line.chars().count(), 3);
        // Column 2 = just after 'a' and the emoji.
        assert_eq!(char_col_to_unit(line, 2, PositionEncoding::Utf16), 3); // 'a' (1) + emoji (2)
        assert_eq!(char_col_to_unit(line, 2, PositionEncoding::Utf8), 5); // 'a' (1) + emoji (4)
        assert_eq!(char_col_to_unit(line, 2, PositionEncoding::Utf32), 2); // 'a' (1) + emoji (1 char)
        assert_round_trips(line);
    }

    #[test]
    fn zwj_family_emoji_is_five_chars_with_a_surrogate_heavy_utf16_run() {
        // Man + ZWJ + Woman + ZWJ + Girl: a human reads this as one glyph,
        // but this crate's public unit is plain `char`s (see the module
        // docs), not grapheme clusters, so it is legitimately FIVE columns
        // here -- unlike `kopitiam-neovim`'s grapheme-indexed cursor, this
        // is the documented, correct contract for this crate, not a gap.
        let line = "👨\u{200D}👩\u{200D}👧";
        assert_eq!(line.chars().count(), 5);
        assert_eq!(line.encode_utf16().count(), 8, "3 emoji * 2 units + 2 ZWJ * 1 unit");
        assert_round_trips(line);
    }

    #[test]
    fn combining_acute_accent_is_two_chars_precomposed_is_one() {
        let decomposed = "e\u{0301}"; // 'e' + COMBINING ACUTE ACCENT: 2 chars
        let precomposed = "\u{00E9}"; // 'é': 1 char
        assert_eq!(decomposed.chars().count(), 2);
        assert_eq!(precomposed.chars().count(), 1);
        assert_round_trips(decomposed);
        assert_round_trips(precomposed);
        // Both spell the same visual glyph, but reaching "the end" of the
        // decomposed form takes 2 columns where the precomposed form takes 1
        // -- exactly the kind of unit mismatch this crate's `char`-offset
        // contract makes visible to callers rather than hiding.
        assert_eq!(char_col_to_unit(decomposed, 2, PositionEncoding::Utf32), 2);
        assert_eq!(char_col_to_unit(precomposed, 1, PositionEncoding::Utf32), 1);
    }

    #[test]
    fn the_same_logical_column_maps_to_different_numbers_in_each_encoding() {
        // This is the entire point of the three encodings existing: past
        // multi-byte content, "the same position" is a different integer
        // depending on which encoding the server negotiated.
        let line = "日本語x";
        let col = 4; // just after the trailing 'x' (end of line)
        let utf8 = char_col_to_unit(line, col, PositionEncoding::Utf8);
        let utf16 = char_col_to_unit(line, col, PositionEncoding::Utf16);
        let utf32 = char_col_to_unit(line, col, PositionEncoding::Utf32);
        assert_eq!(utf8, 10, "3 CJK chars * 3 bytes + 1 ASCII byte");
        assert_eq!(utf16, 4, "every char here is in the BMP: 1 UTF-16 unit each");
        assert_eq!(utf32, 4, "4 chars");
        assert_ne!(utf8, utf16, "utf-8 and utf-16 must disagree once multi-byte content is involved");
        assert_eq!(utf16, utf32, "utf-16 and utf-32 happen to agree here because every char is in the BMP");
    }

    #[test]
    fn unit_to_char_col_clamps_past_end_of_line_instead_of_panicking() {
        let line = "abc";
        assert_eq!(unit_to_char_col(line, 999, PositionEncoding::Utf8), 3);
        assert_eq!(unit_to_char_col(line, 999, PositionEncoding::Utf16), 3);
        assert_eq!(unit_to_char_col(line, 999, PositionEncoding::Utf32), 3);
    }

    #[test]
    fn utf8_offset_mid_char_boundary_rounds_down_instead_of_panicking() {
        // Byte offsets 1 and 2 sit inside "日" (bytes 0..3). A compliant
        // server never sends these, but this must not panic on a slice
        // index that isn't a UTF-8 char boundary.
        let line = "日本";
        assert_eq!(unit_to_char_col(line, 1, PositionEncoding::Utf8), 0);
        assert_eq!(unit_to_char_col(line, 2, PositionEncoding::Utf8), 0);
        assert_eq!(unit_to_char_col(line, 3, PositionEncoding::Utf8), 1);
    }

    #[test]
    fn empty_line_has_a_single_valid_column_at_unit_zero() {
        assert_eq!(char_col_to_unit("", 0, PositionEncoding::Utf16), 0);
        assert_eq!(unit_to_char_col("", 0, PositionEncoding::Utf16), 0);
    }

    #[test]
    fn capability_string_parsing_defaults_to_utf16() {
        assert_eq!(PositionEncoding::from_capability(Some("utf-8")), PositionEncoding::Utf8);
        assert_eq!(PositionEncoding::from_capability(Some("utf-32")), PositionEncoding::Utf32);
        assert_eq!(PositionEncoding::from_capability(Some("utf-16")), PositionEncoding::Utf16);
        assert_eq!(PositionEncoding::from_capability(Some("anything-else")), PositionEncoding::Utf16);
        assert_eq!(PositionEncoding::from_capability(None), PositionEncoding::Utf16);
    }

    /// The regression test: this is the exact bug described in the module
    /// docs. A rename target sits after multi-byte UTF-8 content on the
    /// line, and the server has negotiated real byte-oriented `"utf-8"`
    /// positions. Before the fix, this crate treated `"utf-8"` as meaning
    /// `char` offsets and would have sent the raw `char` column straight
    /// over the wire; a byte-oriented server reads that number as "this
    /// many bytes into the line", which lands inside the preceding
    /// multi-byte character, not on the target -- silently corrupting the
    /// rename.
    #[test]
    fn regression_rename_target_after_multibyte_text_under_utf8_encoding() {
        let line = "日本語x"; // target 'x' sits at char column 3
        let char_col = 3;

        // What the pre-fix code effectively did: pass the `char` offset
        // straight through as if `"utf-8"` meant `char` count.
        let wrongly_assumed_wire_value = char_col;

        // What `"utf-8"` actually means per LSP 3.17: a byte offset.
        let correct_wire_value = char_col_to_unit(line, char_col, PositionEncoding::Utf8);
        assert_eq!(correct_wire_value, 9, "'x' is preceded by 3 CJK characters, 3 bytes each");
        assert_ne!(
            correct_wire_value, wrongly_assumed_wire_value,
            "if these matched, the original bug would have happened to be silently correct here too"
        );

        // Converting the correct wire value back must land exactly on 'x'.
        assert_eq!(unit_to_char_col(line, correct_wire_value, PositionEncoding::Utf8), char_col);

        // Whereas naively treating the WRONG (char-count) value as a byte
        // offset lands one or more characters early, inside "本"/"語"
        // territory rather than on 'x' -- exactly the silent corruption
        // this module exists to prevent.
        let corrupted_col = unit_to_char_col(line, wrongly_assumed_wire_value, PositionEncoding::Utf8);
        assert_ne!(corrupted_col, char_col, "the bug: byte offset 3 must NOT land on 'x' (char column 3)");
    }
}

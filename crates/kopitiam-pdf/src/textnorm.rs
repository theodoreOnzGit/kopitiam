//! Per-character text hygiene at the extraction boundary.
//!
//! Ported from the vendored MIT reference
//! `crates/kopitiam-document/vendor/pdf-to-markdown/pdf2md/textnorm.py`
//! (`normalize_char`, lines 42-53, with the `LIGATURES`/`_SPACES` tables at
//! lines 13-24), translated to Rust for KOPITIAM (AGPL-3.0-only). Per the port
//! conventions in `docs/ai-decisions/AID-0051-mupdf-port-conventions.md`: the
//! observable behaviour follows the Python reference; the code is re-expressed
//! idiomatically, and each deliberate divergence is called out below.
//!
//! # Why this exists (kopitiam_token_max.md Task I-A, §2.1)
//!
//! PDF text streams routinely carry ligature glyphs (a single `ﬁ`), exotic
//! whitespace (NBSP, thin/figure spaces), a byte-order mark, and stray control
//! characters. Worst of these is a raw `U+0000`: `pdf-extract`'s `decode_char`
//! can return NUL when a `/ToUnicode` CMap maps a code to it, and a single NUL
//! in the output makes ripgrep classify the whole file as **binary** and refuse
//! to print matches — silently disabling the grep-then-slice access pattern.
//!
//! Fixing this at *extraction* (not rendering) keeps the recovery ratio honest:
//! a NUL counts as non-whitespace on **both** sides of the validator's ratio, so
//! the validator reports `PASS` on an unsearchable file — it is structurally
//! blind to the bug. Dropping the character here removes it *before*
//! `extracted_content_chars` counts it, so the ratio is unchanged or improved,
//! never depressed.
//!
//! # Behaviour table
//!
//! | Input                                   | Output   | Rule                     |
//! |-----------------------------------------|----------|--------------------------|
//! | `ﬁ ﬂ ﬀ ﬃ ﬄ ﬅ ﬆ`                         | `fi fl …` | ligature expansion       |
//! | `U+00A0` NBSP, `U+2009` thin, `U+2007` … | `' '`    | Zs space → ASCII space   |
//! | `U+FEFF` BOM / ZWNBSP                    | dropped  | format char (Cf)         |
//! | `U+0000`..`U+001F` (C0), `U+007F`, C1    | dropped  | control (Cc), except `\t`,`\n` |
//! | `U+200B` ZWSP, `U+200E` LRM, `U+202A` …  | dropped  | format char (Cf)         |
//! | `U+E000`..`U+F8FF` (+ supplementary PUA) | dropped  | private use (Co)         |
//! | `\t`, `\n`                              | kept     | see divergence below     |
//! | any other char                          | kept     | pass through             |
//!
//! # Deliberate divergences from `textnorm.py`
//!
//! * **Newline is preserved.** The Python `normalize_char` excepts only `\t`
//!   (`text != "\t"`), so it would drop `\n` as category `Cc`. It operates per
//!   glyph, where a newline never appears. In kopitiam this function also runs on
//!   the MuPDF path's line text, and callers legitimately hold `\n`/`\t`; Task
//!   I-A's acceptance is explicit that both are preserved. So [`normalize_char`]
//!   excepts `\t` **and** `\n`.
//! * **Category `Cn` (unassigned) is not dropped.** Python leans on
//!   `unicodedata.category(...).startswith("C")`, which covers `Cn`. Rust's std
//!   ships no Unicode general-category database, and this crate takes no new
//!   dependency (Task I-A: pure std). We reproduce the *practically occurring*
//!   members of category `C`: control (`Cc`, via [`char::is_control`]), the
//!   well-known format chars (`Cf`), surrogates (`Cs`, unrepresentable in a Rust
//!   `char` — so already impossible), and private-use (`Co`). Detecting the full
//!   `Cn` set faithfully would need a generated Unicode table — exactly the kind
//!   of generated-data port AID-0051 §6 defers until a phase demands it. The
//!   bug this card targets (NUL, C0/C1, BOM) is fully covered without it.

/// Ligature glyphs that expand to their ASCII letter sequence.
/// Mirrors `textnorm.py::LIGATURES` (lines 13-21).
fn ligature(c: char) -> Option<&'static str> {
    Some(match c {
        '\u{FB00}' => "ff", // ﬀ
        '\u{FB01}' => "fi", // ﬁ
        '\u{FB02}' => "fl", // ﬂ
        '\u{FB03}' => "ffi", // ﬃ
        '\u{FB04}' => "ffl", // ﬄ
        '\u{FB05}' => "st", // ﬅ (long-s t)
        '\u{FB06}' => "st", // ﬆ
        _ => return None,
    })
}

/// Unicode space separators (general category `Zs`) other than the ASCII space,
/// plus NBSP. These collapse to a single ASCII space. This is a superset of
/// `textnorm.py::_SPACES` (line 24): the reference lists NBSP + seven exotic
/// spaces; we cover the complete `Zs` block so figure/narrow/ideographic spaces
/// are handled too.
fn is_exotic_space(c: char) -> bool {
    matches!(
        c,
        '\u{00A0}'                 // NO-BREAK SPACE (NBSP)
        | '\u{1680}'               // OGHAM SPACE MARK
        | '\u{2000}'..='\u{200A}'  // EN QUAD .. HAIR SPACE (thin, figure, en, em, …)
        | '\u{202F}'               // NARROW NO-BREAK SPACE
        | '\u{205F}'               // MEDIUM MATHEMATICAL SPACE
        | '\u{3000}' // IDEOGRAPHIC SPACE
    )
}

/// Well-known format characters (`Cf`) and private-use characters (`Co`) — the
/// members of Unicode category `C` that [`char::is_control`] does *not* catch.
/// Includes `U+FEFF` (BOM / zero-width no-break space). Surrogates (`Cs`) cannot
/// exist in a Rust `char`, and unassigned (`Cn`) is deferred (see module docs).
fn is_format_or_private(c: char) -> bool {
    matches!(
        c,
        // --- Format characters (Cf) ---
        '\u{00AD}'                 // SOFT HYPHEN
        | '\u{0600}'..='\u{0605}'  // ARABIC NUMBER SIGN .. ARABIC NUMBER MARK ABOVE
        | '\u{061C}'               // ARABIC LETTER MARK
        | '\u{06DD}'               // ARABIC END OF AYAH
        | '\u{070F}'               // SYRIAC ABBREVIATION MARK
        | '\u{08E2}'               // ARABIC DISPUTED END OF AYAH
        | '\u{180E}'               // MONGOLIAN VOWEL SEPARATOR
        | '\u{200B}'..='\u{200F}'  // ZERO WIDTH SPACE .. RIGHT-TO-LEFT MARK
        | '\u{202A}'..='\u{202E}'  // bidi embedding / override controls
        | '\u{2060}'..='\u{2064}'  // WORD JOINER .. INVISIBLE PLUS
        | '\u{2066}'..='\u{206F}'  // bidi isolates / deprecated formatting
        | '\u{FEFF}'               // ZERO WIDTH NO-BREAK SPACE (BOM)
        | '\u{FFF9}'..='\u{FFFB}'  // interlinear annotation anchors
        | '\u{110BD}' | '\u{110CD}'
        | '\u{13430}'..='\u{1343F}'
        | '\u{1BCA0}'..='\u{1BCA3}'
        | '\u{1D173}'..='\u{1D17A}'
        | '\u{E0001}' | '\u{E0020}'..='\u{E007F}'
        // --- Private use (Co) ---
        | '\u{E000}'..='\u{F8FF}'      // BMP Private Use Area
        | '\u{F0000}'..='\u{FFFFD}'    // Supplementary Private Use Area-A
        | '\u{100000}'..='\u{10FFFD}'  // Supplementary Private Use Area-B
    )
}

/// Normalize a single character, appending its cleaned form to `out`.
///
/// This is the single source of truth applied on **both** extraction engines
/// (the legacy `pdf-extract` path in `extractor.rs` and the default MuPDF path
/// in `mupdf_extract.rs`), so a stray control char from either engine can never
/// reach the output. A dropped character appends nothing.
pub fn normalize_char(c: char, out: &mut String) {
    if let Some(expanded) = ligature(c) {
        out.push_str(expanded);
        return;
    }
    if is_exotic_space(c) {
        out.push(' ');
        return;
    }
    // Drop category-C characters. `\t` and `\n` are the two exceptions (see the
    // module-level divergence note): they are control chars that carry real
    // structure at kopitiam's extraction boundary.
    if c == '\t' || c == '\n' {
        out.push(c);
        return;
    }
    if c.is_control() || is_format_or_private(c) {
        return;
    }
    out.push(c);
}

/// Append the normalized form of every character in `s` to `out`. Zero-copy for
/// the common case where nothing needs changing is not attempted — callers
/// building a fresh span string pay one `String` growth either way.
pub fn normalize_into(s: &str, out: &mut String) {
    for c in s.chars() {
        normalize_char(c, out);
    }
}

/// Convenience wrapper: normalize `s` into a fresh `String`. Reserved for
/// callers that want a one-shot normalization (and used by this module's tests);
/// the extraction hot paths use [`normalize_into`]/[`normalize_char`] to append
/// into a span string they are already building.
#[allow(dead_code)]
pub fn normalize_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    normalize_into(s, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The core acceptance case (Task I-A §6): a string carrying a NUL, a C0
    /// control, a C1 control, NBSP, a ligature and a BOM comes out clean, while
    /// tab and newline survive.
    #[test]
    fn strips_controls_expands_ligatures_collapses_spaces() {
        let input = "a\u{0}b\u{1}c\u{0085}\u{00A0}d\u{FB01}e\u{FEFF}f\tg\nh";
        let out = normalize_str(input);
        // NUL, U+0001 (C0), U+0085 (C1 NEL) and BOM are gone; NBSP became ' ';
        // ﬁ expanded to "fi"; \t and \n preserved. Walk: a b c [NBSP→' '] d
        // [ﬁ→fi] e f \t g \n h.
        assert_eq!(out, "abc dfief\tg\nh");
    }

    /// Assert the strong invariant the card demands: no `0x00` and no C0/C1
    /// control char (besides `\n`/`\t`) survives, across a hostile input that
    /// includes every C0 code, DEL, and the C1 range.
    #[test]
    fn no_control_char_survives_except_tab_and_newline() {
        let mut input = String::from("visible");
        for cp in 0x00u32..=0x1F {
            input.push(char::from_u32(cp).unwrap());
        }
        input.push('\u{7F}'); // DEL
        for cp in 0x80u32..=0x9F {
            input.push(char::from_u32(cp).unwrap());
        }
        let out = normalize_str(&input);
        assert!(!out.contains('\u{0}'), "NUL must never survive");
        for c in out.chars() {
            let is_c0_c1 = c.is_control();
            assert!(
                !is_c0_c1 || c == '\t' || c == '\n',
                "control char U+{:04X} leaked through",
                c as u32
            );
        }
        // The only control chars present in the input were \t and \n (added via
        // the 0x00..=0x1F loop), so exactly those two survive.
        assert!(out.contains('\t'));
        assert!(out.contains('\n'));
    }

    #[test]
    fn all_ligatures_expand() {
        assert_eq!(normalize_str("\u{FB00}"), "ff");
        assert_eq!(normalize_str("\u{FB01}"), "fi");
        assert_eq!(normalize_str("\u{FB02}"), "fl");
        assert_eq!(normalize_str("\u{FB03}"), "ffi");
        assert_eq!(normalize_str("\u{FB04}"), "ffl");
        assert_eq!(normalize_str("\u{FB05}"), "st");
        assert_eq!(normalize_str("\u{FB06}"), "st");
    }

    #[test]
    fn exotic_spaces_collapse_to_ascii_space() {
        for sp in [
            '\u{00A0}', '\u{1680}', '\u{2000}', '\u{2007}', '\u{2009}', '\u{200A}', '\u{202F}',
            '\u{205F}', '\u{3000}',
        ] {
            assert_eq!(normalize_str(&sp.to_string()), " ", "U+{:04X}", sp as u32);
        }
    }

    #[test]
    fn bom_and_zero_width_and_private_use_are_dropped() {
        assert_eq!(normalize_str("\u{FEFF}"), ""); // BOM
        assert_eq!(normalize_str("\u{200B}"), ""); // ZWSP
        assert_eq!(normalize_str("\u{200E}"), ""); // LRM
        assert_eq!(normalize_str("\u{E000}"), ""); // PUA
        assert_eq!(normalize_str("\u{F8FF}"), ""); // PUA (Apple logo)
    }

    #[test]
    fn ordinary_text_passes_through_unchanged() {
        let s = "Hello, 世界! naïve café — résumé 1.0";
        assert_eq!(normalize_str(s), s);
    }

    #[test]
    fn a_word_of_only_droppable_chars_becomes_empty() {
        assert_eq!(normalize_str("\u{0}\u{FEFF}\u{200B}"), "");
    }
}

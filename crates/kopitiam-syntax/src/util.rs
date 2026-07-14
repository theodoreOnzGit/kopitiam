//! Small, language-agnostic scanning helpers shared by every per-language
//! lexer in this crate.
//!
//! None of this is a general-purpose lexer generator. Each language module
//! hand-writes its own `highlight_line` function; these helpers only factor
//! out the handful of sub-scans (identifiers, numbers, operator runs) that
//! are genuinely identical across C-family-ish languages, so that six
//! lexers don't reimplement "scan an identifier" six times with six subtly
//! different bugs.

/// A single character set membership test used to classify the "operator"
/// and "punctuation" buckets. Kept as plain `char` predicates rather than a
/// regex (or a dependency on one) because the alphabet is tiny and fixed.
pub fn is_ident_start(c: char) -> bool {
    c.is_alphabetic() || c == '_'
}

pub fn is_ident_continue(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Scans the identifier (or keyword, before classification) starting at
/// byte offset `start` in `line`. Returns the exclusive end byte offset.
///
/// Precondition: `line.as_bytes()[start]` (as a `char`) satisfies
/// [`is_ident_start`]. If it doesn't, returns `start` (an empty span) —
/// callers are expected to have already checked this, but a defensive
/// no-op is safer than a panic in a highlighter that runs on every
/// keystroke over untrusted buffer content.
pub fn scan_ident(line: &str, start: usize) -> usize {
    let rest = &line[start..];
    let mut end = start;
    for (i, c) in rest.char_indices() {
        if i == 0 {
            if !is_ident_start(c) {
                return start;
            }
        } else if !is_ident_continue(c) {
            break;
        }
        end = start + i + c.len_utf8();
    }
    end
}

/// Scans a run of ASCII whitespace (spaces and tabs only — line content
/// never contains the newline itself, see the crate-level docs) starting
/// at `start`, returning the exclusive end offset. Used to peek past
/// whitespace when deciding e.g. whether an identifier is followed by `(`
/// (a function call/definition) without consuming it as part of any span.
pub fn skip_inline_whitespace(line: &str, start: usize) -> usize {
    let rest = &line[start..];
    let mut end = start;
    for (i, c) in rest.char_indices() {
        if c == ' ' || c == '\t' {
            end = start + i + c.len_utf8();
        } else {
            break;
        }
    }
    end
}

/// Best-effort numeric literal scanner shared by Rust, Lua, Python and
/// TOML, all of which use essentially the same alphabet for number
/// literals: an optional `0x`/`0o`/`0b` radix prefix, digits, `_`
/// separators, at most one `.`, an optional exponent (`e`/`E` for decimal,
/// `p`/`P` for hex floats), and a run of trailing alphanumeric "suffix"
/// characters (Rust's `1u32`, `1.0f64`; Python's `1j`; Lua has no numeric
/// suffixes but tolerates the same scan harmlessly).
///
/// This is deliberately *not* a validating numeric-literal parser — it
/// will happily "accept" a malformed number like `1.2.3` as one token. A
/// highlighter's job is to colour code that a human is actively typing,
/// which is malformed far more often than compiler input; erring toward
/// "highlight the plausible extent of the number" over "reject anything
/// not spec-exact" gives a calmer editing experience and, critically, is
/// still a single linear pass — no backtracking, so pathological input
/// only ever costs O(line length).
///
/// Precondition: the character at `start` is an ASCII digit. Returns the
/// exclusive end byte offset.
pub fn scan_number(line: &str, start: usize) -> usize {
    let bytes = line.as_bytes();
    let mut i = start;
    let len = bytes.len();

    // Radix prefix: 0x, 0o, 0b (case-insensitive). Only a hex prefix
    // changes how the rest of the scan behaves (hex digits include a-f,
    // and a hex float's exponent marker is `p`/`P` rather than `e`/`E` --
    // Lua's `0x1p4` -- so treating `e`/`p` uniformly regardless of radix
    // would either mis-scan `0xE` as starting an exponent or `0x1p4`'s
    // `p` as an ordinary trailing suffix letter).
    let mut is_hex = false;
    if bytes.get(i) == Some(&b'0') && i + 1 < len {
        match bytes[i + 1].to_ascii_lowercase() {
            b'x' => {
                is_hex = true;
                i += 2;
            }
            b'o' | b'b' => i += 2,
            _ => {}
        }
    }
    let exponent_markers: &[u8] = if is_hex { b"pP" } else { b"eE" };

    let mut seen_dot = false;
    while i < len {
        let c = bytes[i];
        if c.is_ascii_digit() || c == b'_' || (is_hex && c.is_ascii_hexdigit()) {
            i += 1;
        } else if c == b'.' && !seen_dot && bytes.get(i + 1).is_some_and(u8::is_ascii_digit) {
            // Don't swallow a trailing `.` that's actually a method
            // call / range operator (`1.` followed by non-digit is
            // ambiguous in Rust; we only treat `.` as part of the number
            // if a digit follows it, e.g. `1.0` but not `1.method()`).
            seen_dot = true;
            i += 1;
        } else if exponent_markers.contains(&c) {
            // Exponent marker, possibly with a sign, requiring at least
            // one digit to follow -- otherwise this "e"/"p" is the start
            // of a trailing suffix (or, for hex, just not a digit) rather
            // than a real exponent, and the scan should stop here.
            let mut j = i + 1;
            if bytes.get(j).is_some_and(|b| *b == b'+' || *b == b'-') {
                j += 1;
            }
            if bytes.get(j).is_some_and(u8::is_ascii_digit) {
                i = j;
            } else {
                break;
            }
        } else if c.is_ascii_alphabetic() {
            i += 1; // numeric suffix, e.g. `u32`, `f64`, Python's `2j`.
        } else {
            break;
        }
    }
    i
}

/// Consumes a maximal run of characters drawn from `alphabet` starting at
/// `start`. Used to group multi-character operators (`==`, `->`, `..=`)
/// into a single [`crate::HighlightSpan`] rather than one span per
/// character, which would be technically correct but visually noisy and
/// wasteful (three spans of the same colour instead of one).
pub fn scan_run(line: &str, start: usize, alphabet: &str) -> usize {
    let rest = &line[start..];
    let mut end = start;
    for (i, c) in rest.char_indices() {
        if alphabet.contains(c) {
            end = start + i + c.len_utf8();
        } else {
            break;
        }
    }
    end
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_ident_stops_at_non_ident_char() {
        assert_eq!(scan_ident("foo_bar(baz)", 0), 7);
        assert_eq!(scan_ident("x", 0), 1);
    }

    #[test]
    fn scan_ident_handles_unicode_identifiers() {
        // Rust and Python both permit non-ASCII identifiers.
        let line = "café + 1";
        let end = scan_ident(line, 0);
        assert_eq!(&line[0..end], "café");
    }

    #[test]
    fn scan_number_handles_hex_float_and_suffix() {
        assert_eq!(scan_number("0xFFu32 rest", 0), "0xFFu32".len());
        assert_eq!(scan_number("1.5e-10 rest", 0), "1.5e-10".len());
        assert_eq!(scan_number("1_000_000", 0), 9);
    }

    #[test]
    fn scan_number_does_not_swallow_method_call_dot() {
        // `1.method()` -- the `.` must not be absorbed into the number,
        // since `1method` is not a thing but `1.method()` very much is.
        assert_eq!(scan_number("1.method()", 0), 1);
    }

    #[test]
    fn scan_run_groups_multi_char_operators() {
        assert_eq!(scan_run("==foo", 0, "=!<>&|+-*/%^.:?~"), 2);
        assert_eq!(scan_run("->foo", 0, "=!<>&|+-*/%^.:?~"), 2);
    }
}

//! Line-ending detection and preservation.
//!
//! Vim (and every other serious editor) treats "does this file use `\n` or
//! `\r\n`" as a property of the file, not a preference of the editor.
//! Silently normalizing a CRLF file to LF on save turns every line of a
//! diff into a change — exactly the kind of noise a version-control-aware
//! editor must not introduce. [`LineEnding`] is detected once, on load
//! ([`LineEnding::detect`]), and [`super::buffer::Buffer`] never rewrites
//! existing bytes because of it. The only thing it *does* rewrite is
//! **freshly inserted** text ([`LineEnding::normalize`]), so that pressing
//! Enter in a CRLF file produces `\r\n` instead of leaving a single stray
//! `\n` behind — see [`super::buffer::Buffer::apply`].

use std::borrow::Cow;

/// Which line terminator a buffer's file used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum LineEnding {
    /// `\n`. The default for a brand-new buffer, and for any content whose
    /// first line break is a bare `\n` (or that has no line break at all).
    #[default]
    Lf,
    /// `\r\n`.
    CrLf,
}

impl LineEnding {
    /// Detects the dominant line ending in `text` by inspecting the first
    /// line break found — the same "first line wins" heuristic vim's own
    /// `fileformat` autodetection uses.
    ///
    /// Text with no line breaks at all (a single line, or empty) has
    /// nothing to detect, so it defaults to `Lf` — the more portable choice,
    /// and the one every buffer created with [`Buffer::new`](super::buffer::Buffer::new)
    /// or [`Buffer::from_str`](super::buffer::Buffer::from_str) starts with.
    pub(crate) fn detect(text: &str) -> Self {
        if let Some(nl) = text.find('\n')
            && nl > 0 && text.as_bytes()[nl - 1] == b'\r' {
                return LineEnding::CrLf;
            }
        LineEnding::Lf
    }

    /// Rewrites `text` so every line break in it matches this line ending:
    /// for `CrLf`, bare `\n` becomes `\r\n`; for `Lf`, `\r\n` collapses to
    /// `\n`. Returns the input unchanged (borrowed, no allocation) when it
    /// already matches.
    ///
    /// This is applied to an [`Edit`](crate::Edit)'s *inserted* text before
    /// it reaches the rope. It never touches text already in the buffer —
    /// preserving an existing file's line endings is a matter of simply not
    /// rewriting them, which `Buffer` achieves by never running this over
    /// anything but the new text of an edit.
    pub(crate) fn normalize<'a>(self, text: &'a str) -> Cow<'a, str> {
        match self {
            LineEnding::Lf => {
                if text.contains('\r') {
                    Cow::Owned(text.replace("\r\n", "\n"))
                } else {
                    Cow::Borrowed(text)
                }
            }
            LineEnding::CrLf => {
                if text.contains('\n') {
                    // Normalize to LF first so an existing `\r\n` in the
                    // inserted text (e.g. a paste from a CRLF source)
                    // doesn't double up into `\r\r\n`.
                    let lf_only: Cow<'_, str> =
                        if text.contains('\r') { Cow::Owned(text.replace("\r\n", "\n")) } else { Cow::Borrowed(text) };
                    Cow::Owned(lf_only.replace('\n', "\r\n"))
                } else {
                    Cow::Borrowed(text)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_lf() {
        assert_eq!(LineEnding::detect("a\nb\n"), LineEnding::Lf);
    }

    #[test]
    fn detects_crlf() {
        assert_eq!(LineEnding::detect("a\r\nb\r\n"), LineEnding::CrLf);
    }

    #[test]
    fn defaults_to_lf_with_no_line_break_at_all() {
        assert_eq!(LineEnding::detect("no newlines here"), LineEnding::Lf);
        assert_eq!(LineEnding::detect(""), LineEnding::Lf);
    }

    #[test]
    fn lf_normalize_collapses_crlf_and_leaves_lf_alone() {
        assert_eq!(LineEnding::Lf.normalize("a\r\nb\nc"), "a\nb\nc");
        assert_eq!(LineEnding::Lf.normalize("a\nb"), "a\nb");
    }

    #[test]
    fn crlf_normalize_expands_bare_lf_without_doubling_existing_crlf() {
        assert_eq!(LineEnding::CrLf.normalize("a\nb"), "a\r\nb");
        assert_eq!(LineEnding::CrLf.normalize("a\r\nb"), "a\r\nb");
    }

    #[test]
    fn normalize_borrows_when_already_matching() {
        assert!(matches!(LineEnding::Lf.normalize("plain text"), Cow::Borrowed(_)));
        assert!(matches!(LineEnding::CrLf.normalize("plain text"), Cow::Borrowed(_)));
    }
}

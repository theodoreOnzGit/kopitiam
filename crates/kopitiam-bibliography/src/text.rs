//! Turning printed text back into the string the author typed.
//!
//! A reference list in a PDF is *typeset*, and typesetting is lossy. Before any
//! of it can be parsed, three transformations a typesetter applied have to be
//! undone. Each one is a place where a naive `lines.join(" ")` silently corrupts
//! a citation, and all three occur routinely in real typeset reference lists
//! — none of them would have appeared in a synthetic fixture written from
//! scratch, because you do not think to synthesise the things you do not know
//! about.
//!
//! # 1. Hyphenation at a line break
//!
//! ```text
//!     "Investigation on cross-lingual transfer characteris-
//!      tics of subword tokenizers in a multi-encoder pipeline"
//! ```
//!
//! The hyphen after `characteris` is not in the title. TeX put it there because
//! the line ran out. Joining with a space gives `characteris- tics`; joining
//! naively without one gives `characteristics`, which is right *here* — but the
//! same rule applied to
//!
//! ```text
//!     "context-aware, high-
//!      precision alignment output"
//! ```
//!
//! must produce `high-precision`, **keeping** the hyphen, because that one is
//! part of the word.
//!
//! These two cases are not distinguishable from the text alone. They are only
//! distinguishable with a dictionary, and this crate does not have one and will
//! not guess with an LLM (CLAUDE.md: never ask a model to infer what tooling can
//! derive — and when tooling *cannot* derive it, the honest move is to say so,
//! not to promote a model to an oracle).
//!
//! So [`normalise`] applies the rule that is right **when the hyphen is a
//! soft (typesetter-inserted) hyphen**, and *records the assumption*: see
//! [`HyphenJoin`], which is returned alongside the text so the caller can
//! surface it as an anomaly. The current rule:
//!
//! * A line-final `-` preceded by a letter and followed by a **lower-case**
//!   letter is treated as a soft hyphen and removed (`characteris-|tics` →
//!   `characteristics`).
//! * A line-final `-` followed by an **upper-case** letter or a digit keeps the
//!   hyphen (`high-|Temperature`, `ISO-|9001`), because a compound whose second
//!   element is capitalised is a real compound.
//!
//! That is right far more often than it is wrong, and being *wrong* here
//! produces a title with a missing hyphen — visible, correctable, and not a
//! misattribution. It is the least dangerous available failure.
//!
//! # 2. URLs broken across lines
//!
//! ```text
//!     "https://github.com/
//!      openalign/mtat_toolkit"
//! ```
//!
//! LaTeX's `url` package breaks long URLs at `/` with **no hyphen**. Joining
//! with a space produces `https://github.com/ openalign/...`, which is not
//! a URL and will not resolve. A URL is a claim about where something lives; a
//! corrupted one is a broken claim. So a line that ends inside a URL is joined
//! to the next **without** a space.
//!
//! # 3. Digit-group separators inside numbers
//!
//! `biblatex` prints large numbers with a group separator, so a journal's
//! article number `111144` is typeset as
//! `p. 111 144`. Left alone, that parses as page `111`, followed by the
//! mysterious token `144`. This is the same class of bug the plot engine hit
//! with comma-grouped axis labels (`11,000`) — a real producer groups digits,
//! and a synthetic one does not.
//!
//! Digit regrouping is **not** done here, because at this level a space between
//! two numbers is ambiguous. It is done in [`crate::fields`], where the
//! surrounding `p.` / `pp.` tells us we are looking at a page number, and it is
//! always reported as an assumption.

use std::sync::LazyLock;

use regex::Regex;

/// Matches the tail of a line that is in the middle of a URL, so that the next
/// line must be joined to it without an intervening space.
///
/// Deliberately anchored on a URL *scheme* appearing in the trailing token
/// rather than on "ends with `/`": plenty of ordinary prose ends a line with a
/// slash, and gluing the next word onto it would invent a compound.
static URL_TAIL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:https?://|www\.|doi\.org/|arxiv\.org/|10\.\d{4,9}/)\S*$").unwrap()
});

/// Whether a line break inside a word was joined by deleting a hyphen, keeping
/// one, or was not a hyphen break at all.
///
/// Returned by [`normalise_reporting`] so that a caller can declare the
/// assumption rather than bury it. See the module docs for why the assumption
/// is unavoidable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HyphenJoin {
    /// A line-final hyphen was **removed**, on the assumption that the
    /// typesetter inserted it. The `String` is the reconstructed word, so a
    /// caller can show a human exactly what was assumed.
    SoftHyphenRemoved(String),
    /// A line-final hyphen was **kept**, because the following fragment began
    /// with a capital or a digit and so reads as a real compound.
    HardHyphenKept(String),
}

/// Joins the printed lines of a reference back into the string it was typeset
/// from.
///
/// See [`normalise_reporting`] for the same thing with the assumptions it made
/// reported back, which is what [`crate::extract`] uses. This is the
/// convenience form for callers that only want the text.
pub fn normalise(text: &str) -> String {
    normalise_reporting(text).0
}

/// [`normalise`], plus the list of hyphen decisions it had to make.
///
/// Every entry in the returned `Vec` is a place where the text on the page and
/// the text the author typed **cannot both be recovered**, and this function
/// picked one. That is worth surfacing, not hiding: see [`crate::Anomaly`].
pub fn normalise_reporting(text: &str) -> (String, Vec<HyphenJoin>) {
    let mut out = String::with_capacity(text.len());
    let mut joins = Vec::new();

    let lines: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();

    for (index, line) in lines.iter().enumerate() {
        let is_last = index + 1 == lines.len();
        let next = if is_last { "" } else { lines[index + 1] };

        // -- Case 1: this line ends mid-hyphenated-word.
        if !is_last
            && let Some(stem) = line.strip_suffix('-')
            && stem.chars().next_back().is_some_and(char::is_alphabetic)
            && let Some(first) = next.chars().next()
        {
            if first.is_lowercase() {
                // Soft hyphen: the typesetter put it there. Delete it.
                out.push_str(stem);
                joins.push(HyphenJoin::SoftHyphenRemoved(format!(
                    "{}{}",
                    last_word(stem),
                    first_word(next)
                )));
            } else {
                // Hard hyphen: a real compound (`high-Temperature`, `ISO-9001`).
                out.push_str(line);
                joins.push(HyphenJoin::HardHyphenKept(format!(
                    "{}-{}",
                    last_word(stem),
                    first_word(next)
                )));
            }
            continue;
        }

        out.push_str(line);

        // -- Case 2: this line ends inside a URL. Join with no space, or the
        // URL is destroyed.
        if is_last {
            continue;
        }
        if URL_TAIL.is_match(line) {
            continue;
        }

        // -- Ordinary line wrap.
        out.push(' ');
    }

    (out, joins)
}

/// The final whitespace-delimited word of `text` (used only to describe a
/// hyphen decision to a human).
fn last_word(text: &str) -> &str {
    text.rsplit(|c: char| c.is_whitespace()).next().unwrap_or(text)
}

/// The leading whitespace-delimited word of `text`.
fn first_word(text: &str) -> &str {
    text.split(|c: char| c.is_whitespace()).next().unwrap_or(text)
}

/// Replaces the typographic characters a typesetter substitutes for ASCII, so
/// that downstream matching does not have to spell every one of them.
///
/// This is applied to *parsed field values*, never to
/// [`crate::Provenance::verbatim`] — the source keeps its own glyphs.
///
/// | Printed | ASCII | Why it matters |
/// |---|---|---|
/// | `\u{2018}` `\u{2019}` | `'` | `Vega et al\u{2019}s` must match an apostrophe |
/// | `\u{201c}` `\u{201d}` | `"` | quoted article titles |
/// | `\u{2013}` `\u{2014}` | `-` | **page ranges**: `281\u{2013}301` |
/// | `\u{2212}` | `-` | a minus sign, used as a dash by some producers |
/// | `\u{00a0}` `\u{2009}` `\u{202f}` | ` ` | non-breaking/thin spaces inside `pp. 281` |
///
/// The en-dash row is the load-bearing one. A page range printed with `\u{2013}`
/// and split on ASCII `-` yields a single page number of `281\u{2013}301`, which is
/// wrong, and wrong in a way that looks fine until someone tries to follow it.
pub fn fold_typography(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' | '\u{02bc}' => '\'',
            '\u{201c}' | '\u{201d}' => '"',
            '\u{2013}' | '\u{2014}' | '\u{2212}' | '\u{2010}' | '\u{2011}' => '-',
            '\u{00a0}' | '\u{2009}' | '\u{202f}' | '\u{2007}' | '\u{2002}' | '\u{2003}' => ' ',
            other => other,
        })
        .collect()
}

/// Collapses runs of whitespace to a single space and trims. Used for
/// comparison and for field values, never for verbatim source text.
pub fn squeeze(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_soft_hyphen_at_a_line_break_is_removed() {
        // The kind of soft hyphen a typesetter inserts mid-word at a line break.
        let (text, joins) = normalise_reporting(
            "Investigation on cross-lingual transfer characteris-\ntics of subword tokenizers",
        );
        assert_eq!(text, "Investigation on cross-lingual transfer characteristics of subword tokenizers");
        assert_eq!(
            joins,
            vec![HyphenJoin::SoftHyphenRemoved("characteristics".to_string())]
        );
    }

    #[test]
    fn a_hyphen_before_a_capital_is_kept_as_a_real_compound() {
        let (text, joins) = normalise_reporting("high-\nLevel grammar");
        assert_eq!(text, "high-Level grammar");
        assert_eq!(
            joins,
            vec![HyphenJoin::HardHyphenKept("high-Level".to_string())]
        );
    }

    #[test]
    fn a_hyphen_before_a_digit_is_kept() {
        let (text, _) = normalise_reporting("ISO-\n9001 conformance");
        assert_eq!(text, "ISO-9001 conformance");
    }

    #[test]
    fn a_url_broken_across_lines_is_joined_without_a_space() {
        // A repository reference whose URL was broken across a line. LaTeX's
        // `url` package breaks at `/` with no hyphen; a space here destroys it.
        let (text, _) = normalise_reporting(
            "Mtat alignment toolkit, https://github.com/\nopenalign/mtat_toolkit, 2024.",
        );
        assert_eq!(
            text,
            "Mtat alignment toolkit, https://github.com/openalign/mtat_toolkit, 2024."
        );
    }

    #[test]
    fn an_ordinary_wrap_is_joined_with_one_space() {
        let (text, _) = normalise_reporting("M. R. Chen, S. Novak,\nand J. P. Alvarez");
        assert_eq!(text, "M. R. Chen, S. Novak, and J. P. Alvarez");
    }

    #[test]
    fn a_line_ending_in_a_slash_that_is_not_a_url_is_not_glued() {
        // "and/" is not a URL tail, so gluing would invent the word "and/or".
        // (Here it genuinely would be "and/ or", which is still not a word --
        // the point is that we do not silently create compounds out of prose.)
        let (text, _) = normalise_reporting("context and/\nor style");
        assert_eq!(text, "context and/ or style");
    }

    #[test]
    fn a_hyphen_not_at_a_line_break_is_untouched() {
        let (text, joins) = normalise_reporting("a well-known part-of-speech tagger");
        assert_eq!(text, "a well-known part-of-speech tagger");
        assert!(joins.is_empty());
    }

    #[test]
    fn blank_lines_do_not_produce_double_spaces() {
        let (text, _) = normalise_reporting("first line\n\n   \nsecond line");
        assert_eq!(text, "first line second line");
    }

    #[test]
    fn en_dashes_fold_to_ascii_so_page_ranges_can_be_split() {
        // The load-bearing case: `281–301` split on ASCII '-' is one page
        // number, not a range, and it looks fine until someone follows it.
        assert_eq!(fold_typography("pp. 281\u{2013}301"), "pp. 281-301");
        assert_eq!(fold_typography("Vega et al\u{2019}s"), "Vega et al's");
        assert_eq!(
            fold_typography("\u{201c}An open-source toolkit,\u{201d}"),
            "\"An open-source toolkit,\""
        );
        assert_eq!(fold_typography("pp.\u{00a0}281"), "pp. 281");
    }

    #[test]
    fn squeeze_collapses_runs_of_whitespace() {
        assert_eq!(squeeze("  a   b \n c  "), "a b c");
    }
}

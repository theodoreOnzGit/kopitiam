//! Finding text on a page — the `Ctrl+F` half of a reader.
//!
//! Ported from MuPDF `source/fitz/stext-search.c` (`fz_search_stext_page`,
//! `canon`, `add_quad`, `hdist`/`vdist`, commit `0b8fd1c`, AGPL-3.0,
//! © Artifex). [`super::structured_text`] already produces the same
//! `StextPage`/`StextLine`/`StextChar` shapes MuPDF searches over, so this is
//! the matching layer on top rather than a new extraction path.
//!
//! # What a hit is
//!
//! A hit is a run of characters, returned as **quads** rather than a string
//! offset, because that is what a viewer needs: quads are what you draw a
//! highlight over. A single hit can carry several quads — a phrase broken
//! across a line break is one hit in two boxes, and drawing one box around
//! both would cover half the paragraph.
//!
//! Consecutive characters merge into one quad while they stay adjacent along
//! the line, within a fuzz proportional to the character size:
//! `hfuzz = 0.5`, `vfuzz = 0.1` (MuPDF's own constants, `stext-search.c:461`
//! and `:1356`, commented there as "merge large gaps"). The distances are
//! measured **along** and **across** the line's own direction
//! ([`StextLine::dir`]), not in page axes, so rotated and skewed text merges
//! correctly rather than fragmenting into one quad per glyph.
//!
//! # Two deliberate divergences from MuPDF, both for the searcher's benefit
//!
//! MuPDF's `canon` folds CR/LF/TAB and the Unicode space-equivalents to a
//! plain space, and stops there. This port adds two things on top, because
//! without them the obvious searches fail on real PDFs:
//!
//! * **A line break reads as a space.** A PDF stores "sound speed" broken
//!   across two lines as two lines with no space character anywhere, so a
//!   literal search for `sound speed` finds nothing at all — the single most
//!   common "search is broken" complaint against a viewer.
//! * **Runs of whitespace collapse to one.** Justified text is full of
//!   multi-space runs, and a reader typing one space means one space.
//!
//! Both are handled by *mapping*, not by rewriting text: the haystack keeps a
//! parallel index back to the character each position came from, and a
//! synthetic or collapsed position maps to nothing. So a match that consumes
//! a line break simply contributes no quad for it, and highlight boxes stay
//! exactly on real glyphs.
//!
//! # Known ceiling
//!
//! Case folding is per-character ([`char::to_lowercase`]), so the one-to-many
//! folds (`ß` → `ss`, `ﬁ` → `fi`) do not match across the expansion. MuPDF
//! reaches full Unicode folding through `ucdn`; matching that needs a case-
//! folding table this crate does not yet carry. Diacritic-insensitive search
//! (MuPDF's `FZ_SEARCH_IGNORE_DIACRITICS`) is likewise not implemented.

use super::geometry::{Point, Quad};
use super::structured_text::{StextBlock, StextChar, StextLine, StextPage};

/// Horizontal merge tolerance, as a fraction of character size. MuPDF:
/// `hits.hfuzz = 0.5f; /* merge large gaps */` (`stext-search.c:461`).
const HFUZZ: f32 = 0.5;

/// Vertical merge tolerance, as a fraction of character size. MuPDF:
/// `hits.vfuzz = 0.1f` (`stext-search.c:462`). Much tighter than the
/// horizontal one: text moves along its line, so a vertical jump almost
/// always means a new line and therefore a new quad.
const VFUZZ: f32 = 0.1;

/// One match on the page.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    /// The hit's position in reading order, 0-based — MuPDF's `hit_mark`.
    pub index: usize,
    /// The boxes to highlight. More than one when the match spans a line
    /// break or a large intra-line gap; never empty.
    pub quads: Vec<Quad>,
}

impl SearchHit {
    /// The union of every quad — a single box covering the whole hit.
    ///
    /// Convenient for scrolling a hit into view, and **wrong for
    /// highlighting** a hit that spans lines, where it would cover the text
    /// between the two fragments as well.
    pub fn bounds(&self) -> Option<Quad> {
        let mut it = self.quads.iter();
        let first = *it.next()?;
        let (mut x0, mut y0) = (first.ul.x.min(first.ll.x), first.ul.y.min(first.ur.y));
        let (mut x1, mut y1) = (first.ur.x.max(first.lr.x), first.ll.y.max(first.lr.y));
        for q in it {
            x0 = x0.min(q.ul.x).min(q.ll.x);
            y0 = y0.min(q.ul.y).min(q.ur.y);
            x1 = x1.max(q.ur.x).max(q.lr.x);
            y1 = y1.max(q.ll.y).max(q.lr.y);
        }
        Some(Quad {
            ul: Point { x: x0, y: y0 },
            ur: Point { x: x1, y: y0 },
            ll: Point { x: x0, y: y1 },
            lr: Point { x: x1, y: y1 },
        })
    }
}

/// Search behaviour. [`Default`] is case-insensitive, which is what a reader's
/// find bar does unless told otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchOptions {
    /// MuPDF's `FZ_SEARCH_IGNORE_CASE`, and the default here.
    pub ignore_case: bool,
}

impl Default for SearchOptions {
    fn default() -> SearchOptions {
        SearchOptions { ignore_case: true }
    }
}

/// Find every occurrence of `needle` on `page`, case-insensitively.
pub fn search_page(page: &StextPage, needle: &str) -> Vec<SearchHit> {
    search_page_with(page, needle, SearchOptions::default())
}

/// Find every occurrence of `needle` on `page`.
///
/// Matches never overlap: after a hit, scanning resumes at its end, so
/// searching `aa` in `aaaa` yields two hits rather than three. An empty or
/// whitespace-only needle matches nothing — a find bar mid-typing must not
/// report every position on the page as a hit.
pub fn search_page_with(page: &StextPage, needle: &str, opts: SearchOptions) -> Vec<SearchHit> {
    let needle = canon_needle(needle, opts);
    if needle.is_empty() {
        return Vec::new();
    }
    let hay = Haystack::build(page, opts);
    if hay.chars.len() < needle.len() {
        return Vec::new();
    }

    let mut hits = Vec::new();
    let mut at = 0usize;
    let last_start = hay.chars.len() - needle.len();
    while at <= last_start {
        if hay.chars[at..at + needle.len()] == needle[..] {
            hits.push(SearchHit {
                index: hits.len(),
                quads: hay.quads_for(at, at + needle.len()),
            });
            at += needle.len();
        } else {
            at += 1;
        }
    }
    // A hit whose every position was synthetic (only possible for a
    // whitespace needle, which is refused above) would carry no quad and
    // could not be drawn; drop such a hit rather than emit an undrawable one.
    hits.retain(|h| !h.quads.is_empty());
    for (i, h) in hits.iter_mut().enumerate() {
        h.index = i;
    }
    hits
}

/// MuPDF `canon` (`stext-search.c:646`): CR/LF/TAB and every Unicode
/// space-equivalent fold to a plain space. Everything else is unchanged.
fn canon(c: char) -> char {
    if c == '\r' || c == '\n' || c == '\t' || is_unicode_space_equivalent(c) {
        ' '
    } else {
        c
    }
}

/// MuPDF `fz_is_unicode_space_equivalent` (`stext-search.c:600`) — verbatim,
/// including the omission of U+200B ZERO WIDTH SPACE (which is a *format*
/// character, not a space, and folding it would join words that are not
/// joined).
fn is_unicode_space_equivalent(c: char) -> bool {
    matches!(
        c,
        '\u{00a0}' // NO-BREAK SPACE
        | '\u{1680}' // OGHAM SPACE MARK
        | '\u{2000}'..='\u{200a}' // EN QUAD .. HAIR SPACE
        | '\u{202f}' // NARROW NO-BREAK SPACE
        | '\u{205f}' // MEDIUM MATHEMATICAL SPACE
        | '\u{3000}' // IDEOGRAPHIC SPACE
    )
}

fn fold(c: char, opts: SearchOptions) -> char {
    if opts.ignore_case {
        // Per-character folding: see the module docs' "Known ceiling" on the
        // one-to-many folds this cannot express.
        c.to_lowercase().next().unwrap_or(c)
    } else {
        c
    }
}

/// Canonicalise the needle the same way the haystack is: fold, canon, and
/// collapse whitespace runs, then trim — so a trailing space the user typed
/// does not stop the phrase matching at a line end.
fn canon_needle(needle: &str, opts: SearchOptions) -> Vec<char> {
    let mut out: Vec<char> = Vec::with_capacity(needle.len());
    for c in needle.chars() {
        let c = fold(canon(c), opts);
        if c == ' ' && out.last() == Some(&' ') {
            continue;
        }
        out.push(c);
    }
    while out.last() == Some(&' ') {
        out.pop();
    }
    while out.first() == Some(&' ') {
        out.remove(0);
    }
    out
}

/// Where a haystack position came from on the page. `None` for a synthetic
/// position (a line break read as a space) — those match, but contribute no
/// quad, so a highlight never covers anything that is not a real glyph.
type Origin = Option<(usize, usize)>;

/// The page flattened into a matchable character sequence, keeping each
/// position's origin so a match can be turned back into quads.
struct Haystack<'a> {
    chars: Vec<char>,
    origins: Vec<Origin>,
    lines: Vec<&'a StextLine>,
}

impl<'a> Haystack<'a> {
    fn build(page: &'a StextPage, opts: SearchOptions) -> Haystack<'a> {
        let mut lines: Vec<&StextLine> = Vec::new();
        for block in &page.blocks {
            if let StextBlock::Text(t) = block {
                lines.extend(t.lines.iter());
            }
        }

        let mut chars = Vec::new();
        let mut origins: Vec<Origin> = Vec::new();
        for (li, line) in lines.iter().enumerate() {
            // A line boundary reads as a space (see the module docs). Emitted
            // before the line rather than after, so no trailing space is left
            // dangling, and skipped when the previous position is already a
            // space so the collapse rule holds across the join too.
            if !chars.is_empty() && chars.last() != Some(&' ') {
                chars.push(' ');
                origins.push(None);
            }
            for (ci, ch) in line.chars.iter().enumerate() {
                let c = fold(canon(ch.c), opts);
                if c == ' ' && chars.last() == Some(&' ') {
                    continue;
                }
                chars.push(c);
                origins.push(Some((li, ci)));
            }
        }
        Haystack {
            chars,
            origins,
            lines,
        }
    }

    /// The quads covering haystack positions `start..end`, merged per
    /// MuPDF's `add_quad` (`stext-search.c:1657`).
    fn quads_for(&self, start: usize, end: usize) -> Vec<Quad> {
        let mut quads: Vec<Quad> = Vec::new();
        let mut prev_line: Option<usize> = None;
        for pos in start..end {
            let Some((li, ci)) = self.origins[pos] else {
                continue; // synthetic position: nothing to draw
            };
            let line = self.lines[li];
            let Some(ch) = line.chars.get(ci) else { continue };

            let merged = prev_line == Some(li)
                && quads
                    .last()
                    .is_some_and(|q| adjacent(q, ch, line.dir));
            if merged {
                // Extend the run's trailing edge, exactly as MuPDF does.
                let last = quads.last_mut().expect("merged implies a last quad");
                last.ur = ch.quad.ur;
                last.lr = ch.quad.lr;
            } else {
                quads.push(ch.quad);
            }
            prev_line = Some(li);
        }
        quads
    }
}

/// Is `ch` close enough to the running quad's trailing edge to be the same
/// visual run? MuPDF `add_quad`'s merge test, with its `hdist`/`vdist`
/// (`stext-search.c:87`): the gap is projected onto the line direction and
/// its perpendicular, so the test follows rotated text instead of assuming
/// page axes.
fn adjacent(run: &Quad, ch: &StextChar, dir: Point) -> bool {
    let hfuzz = ch.size * HFUZZ;
    let vfuzz = ch.size * VFUZZ;
    hdist(dir, run.lr, ch.quad.ll) < hfuzz
        && vdist(dir, run.lr, ch.quad.ll) < vfuzz
        && hdist(dir, run.ur, ch.quad.ul) < hfuzz
        && vdist(dir, run.ur, ch.quad.ul) < vfuzz
}

/// Distance between `a` and `b` **along** `dir`. MuPDF `hdist`.
fn hdist(dir: Point, a: Point, b: Point) -> f32 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    (dx * dir.x - dy * dir.y).abs()
}

/// Distance between `a` and `b` **across** `dir`. MuPDF `vdist`.
fn vdist(dir: Point, a: Point, b: Point) -> f32 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    (dx * dir.y - dy * dir.x).abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::structured_text::{StextTextBlock, StextPage};
    use super::super::geometry::Rect;

    /// A line of single-width chars laid out left to right at `y`, each 10
    /// units wide and `size` tall — enough geometry for the merge test to be
    /// meaningful without a real document.
    fn line(text: &str, y: f32, size: f32) -> StextLine {
        let chars = text
            .chars()
            .enumerate()
            .map(|(i, c)| {
                let x = i as f32 * 10.0;
                StextChar {
                    c,
                    origin: Point { x, y },
                    quad: Quad {
                        ul: Point { x, y: y - size },
                        ur: Point { x: x + 10.0, y: y - size },
                        ll: Point { x, y },
                        lr: Point { x: x + 10.0, y },
                    },
                    size,
                    font: 0,
                    flags: 0,
                    cid: 0,
                    wmode: 0,
                }
            })
            .collect();
        StextLine {
            wmode: 0,
            flags: 0,
            dir: Point { x: 1.0, y: 0.0 },
            bbox: Rect { x0: 0.0, y0: y - size, x1: text.chars().count() as f32 * 10.0, y1: y },
            chars,
        }
    }

    fn page(lines: Vec<StextLine>) -> StextPage {
        StextPage {
            mediabox: Rect { x0: 0.0, y0: 0.0, x1: 600.0, y1: 800.0 },
            blocks: vec![StextBlock::Text(StextTextBlock {
                bbox: Rect { x0: 0.0, y0: 0.0, x1: 600.0, y1: 800.0 },
                lines,
            })],
            fonts: Vec::new(),
        }
    }

    #[test]
    fn finds_a_simple_match() {
        let p = page(vec![line("hello world", 20.0, 10.0)]);
        let hits = search_page(&p, "world");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].index, 0);
        assert_eq!(hits[0].quads.len(), 1, "one run on one line = one quad");
    }

    #[test]
    fn matching_is_case_insensitive_by_default() {
        let p = page(vec![line("Hello World", 20.0, 10.0)]);
        assert_eq!(search_page(&p, "hello").len(), 1);
        assert_eq!(search_page(&p, "WORLD").len(), 1);
        let exact = SearchOptions { ignore_case: false };
        assert_eq!(search_page_with(&p, "hello", exact).len(), 0);
        assert_eq!(search_page_with(&p, "Hello", exact).len(), 1);
    }

    /// The complaint this module exists to answer: a phrase broken across a
    /// line break has no space character anywhere, so a literal search finds
    /// nothing.
    #[test]
    fn a_phrase_split_across_lines_is_found() {
        let p = page(vec![line("sound", 20.0, 10.0), line("speed", 40.0, 10.0)]);
        let hits = search_page(&p, "sound speed");
        assert_eq!(hits.len(), 1, "the line break must read as a space");
        assert_eq!(
            hits[0].quads.len(),
            2,
            "a hit spanning two lines needs two boxes, not one covering both"
        );
    }

    /// Justified text is full of multi-space runs; a reader types one space.
    #[test]
    fn runs_of_whitespace_collapse() {
        let p = page(vec![line("a    b", 20.0, 10.0)]);
        assert_eq!(search_page(&p, "a b").len(), 1);
        // And the needle's own extra spaces are collapsed the same way.
        assert_eq!(search_page(&p, "a    b").len(), 1);
    }

    /// Unicode space equivalents fold to a plain space (MuPDF `canon`).
    #[test]
    fn nbsp_and_friends_match_a_plain_space() {
        let p = page(vec![line("a\u{00a0}b", 20.0, 10.0)]);
        assert_eq!(search_page(&p, "a b").len(), 1);
        let p = page(vec![line("a\u{2003}b", 20.0, 10.0)]);
        assert_eq!(search_page(&p, "a b").len(), 1);
    }

    /// A quad is only drawn for a real glyph, so a match that consumes a line
    /// break contributes nothing for it — a highlight never covers blank
    /// space between fragments.
    #[test]
    fn synthetic_positions_contribute_no_quad() {
        let p = page(vec![line("ab", 20.0, 10.0), line("cd", 40.0, 10.0)]);
        let hits = search_page(&p, "b c");
        assert_eq!(hits.len(), 1);
        let total: usize = hits[0].quads.len();
        assert_eq!(total, 2, "one quad for 'b', one for 'c', none for the break");
    }

    #[test]
    fn overlapping_matches_are_not_double_counted() {
        let p = page(vec![line("aaaa", 20.0, 10.0)]);
        assert_eq!(search_page(&p, "aa").len(), 2, "hits must not overlap");
    }

    #[test]
    fn an_empty_or_blank_needle_matches_nothing() {
        let p = page(vec![line("hello", 20.0, 10.0)]);
        assert!(search_page(&p, "").is_empty());
        assert!(search_page(&p, "   ").is_empty());
        assert!(search_page(&p, "\n\t").is_empty());
    }

    #[test]
    fn a_needle_longer_than_the_page_matches_nothing() {
        let p = page(vec![line("hi", 20.0, 10.0)]);
        assert!(search_page(&p, "hi there everyone").is_empty());
    }

    #[test]
    fn multiple_hits_are_indexed_in_order() {
        let p = page(vec![line("cat dog cat", 20.0, 10.0)]);
        let hits = search_page(&p, "cat");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].index, 0);
        assert_eq!(hits[1].index, 1);
        // The second hit must be further along the line than the first.
        assert!(hits[1].quads[0].ll.x > hits[0].quads[0].ll.x);
    }

    /// A large intra-line gap (a table column, say) must break the run into
    /// separate quads rather than draw one box across the gap.
    #[test]
    fn a_wide_gap_within_a_line_splits_the_quad() {
        let mut l = line("ab", 20.0, 10.0);
        // Shove 'b' far to the right, well past hfuzz = 0.5 * size.
        let shift = 400.0;
        l.chars[1].quad.ul.x += shift;
        l.chars[1].quad.ur.x += shift;
        l.chars[1].quad.ll.x += shift;
        l.chars[1].quad.lr.x += shift;
        let p = page(vec![l]);
        let hits = search_page(&p, "ab");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].quads.len(), 2, "a big gap is two runs");
    }

    #[test]
    fn bounds_unions_every_quad() {
        let p = page(vec![line("sound", 20.0, 10.0), line("speed", 40.0, 10.0)]);
        let hits = search_page(&p, "sound speed");
        let b = hits[0].bounds().expect("a hit has bounds");
        assert!(b.ul.y <= 10.0 && b.ll.y >= 40.0, "spans both lines: {b:?}");
    }

    #[test]
    fn a_page_with_no_text_yields_nothing() {
        let p = page(Vec::new());
        assert!(search_page(&p, "anything").is_empty());
    }
}

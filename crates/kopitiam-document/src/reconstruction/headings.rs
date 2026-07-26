use std::collections::BTreeSet;
use std::sync::LazyLock;

use regex::Regex;

use super::Line;

static NUMBERED_SECTION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\d+(\.\d+)*\.?\s+\S").unwrap());

const MAX_HEADING_CHARS: usize = 160;

/// A font size must be at least this multiple of the body size to be considered
/// a heading tier at all -- "clearly above body" (task #16). This matches the
/// old fixed H3 threshold, so the *set* of size-based headings is unchanged; only
/// the *level* assigned to each becomes adaptive (see [`HeadingScale`]).
const HEADING_MIN_RATIO: f32 = 1.12;

/// Half-point bucket for a font size, mirroring `estimate_body_font_size` so the
/// two agree on what counts as "the same size".
fn size_bucket(size: f32) -> u32 {
    (size * 2.0).round() as u32
}

/// The document's discrete heading font-size tiers, largest first, adapted from
/// marker's `SectionHeaderProcessor` (Apache-2.0; clean-room adaptation, not a
/// line port).
///
/// Marker clusters every section-header line's height with k-means and ranks the
/// clusters by size to assign H1/H2/H3…. We do the same idea deterministically
/// and without a numeric-ML dependency: the *distinct* font sizes that appear on
/// heading-candidate lines (short, clearly larger than body) are already a
/// handful of discrete values once bucketed to the half point, so ranking those
/// buckets in descending order **is** the clustering. The largest tier is H1, the
/// next H2, and so on (capped at Markdown's `######`).
///
/// This is more correct than a fixed ratio ladder: two headings at 1.8x and 2.4x
/// body are H2 and H1 relative to *this* document, whereas a fixed
/// ">=1.6 => H1" rule would flatten both to H1.
pub(super) struct HeadingScale {
    /// Distinct heading-size buckets, sorted **descending** (largest = H1). At
    /// most six entries (deeper levels collapse into H6).
    tiers: Vec<u32>,
}

impl HeadingScale {
    /// An empty scale: no font-size tiers, so [`heading_level`] falls back to the
    /// fixed-ratio ladder. This is what a uniform-font document produces (the
    /// `build` iterator simply yields no candidates). Used by tests that exercise
    /// `build_blocks` without a document-wide clustering pass.
    #[cfg(test)]
    pub(super) fn empty() -> Self {
        Self { tiers: Vec::new() }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.tiers.is_empty()
    }

    /// Builds the scale from every line in the document. A line contributes its
    /// size as a heading tier only when it is a size heading *candidate*: short
    /// enough to be line-like (not a full paragraph) and clearly larger than
    /// body. Lines that merely *happen* to be large but are long prose are
    /// excluded, so a big pull-quote paragraph does not invent a heading tier.
    pub(super) fn build<'a>(lines: impl Iterator<Item = &'a Line>, body_font_size: f32) -> Self {
        let mut buckets: BTreeSet<u32> = BTreeSet::new();
        for line in lines {
            if is_size_heading_candidate(line, body_font_size) {
                buckets.insert(size_bucket(line.font_size));
            }
        }
        // BTreeSet yields ascending; reverse for descending (largest = H1).
        let mut tiers: Vec<u32> = buckets.into_iter().rev().collect();
        tiers.truncate(6);
        Self { tiers }
    }

    /// The heading level (1-based) for a given font size, or `None` when the size
    /// sits below the smallest heading tier (i.e. it is body text). A size at or
    /// above the largest tier is H1; between tiers it takes the deeper (larger-
    /// numbered) adjacent level.
    fn level_for(&self, size: f32) -> Option<usize> {
        let bucket = size_bucket(size);
        self.tiers
            .iter()
            .position(|&tier| bucket >= tier)
            .map(|index| (index + 1).min(6))
    }
}

/// True when a line's font size alone qualifies it as a heading candidate:
/// non-empty, short (line-like, not a paragraph), and clearly above body size.
/// Deliberately excludes the numbered-section case, which is size-independent.
fn is_size_heading_candidate(line: &Line, body_font_size: f32) -> bool {
    let char_count = line.text.trim().chars().count();
    char_count > 0
        && char_count <= MAX_HEADING_CHARS
        && line.font_size >= body_font_size * HEADING_MIN_RATIO
}

/// Heading detection: a font size that ranks as one of the document's heading
/// tiers (adaptive levels, see [`HeadingScale`]), or a numbered-section prefix
/// (e.g. "3.4 Hyperfocus") at body size.
///
/// Bold-text and centered-text cues are not available -- the PDF extraction
/// backend does not expose font style reliably here, and centering would require
/// per-column text width, which multi-column layouts don't have yet (kopitiam-q4f).
///
/// Precision over recall: a size is only a heading when it is clearly above body
/// *and* the line is short/line-like, so ordinary prose is not mis-promoted. When
/// the document has no size tiers (uniform font, or font sizes absent), this
/// falls back to the previous fixed-ratio ladder so behaviour is unchanged for
/// documents the adaptive path cannot inform.
pub(super) fn heading_level(
    line: &Line,
    body_font_size: f32,
    scale: &HeadingScale,
) -> Option<usize> {
    let text = line.text.trim();
    if text.is_empty() || text.chars().count() > MAX_HEADING_CHARS {
        return None;
    }

    // Size-based heading: adaptive tier rank when the document established tiers,
    // otherwise the legacy fixed-ratio ladder (uniform-font / no-tier fallback).
    let size_level = if scale.is_empty() {
        fixed_ratio_level(line.font_size / body_font_size)
    } else if is_size_heading_candidate(line, body_font_size) {
        scale.level_for(line.font_size)
    } else {
        None
    };
    if let Some(level) = size_level {
        return Some(level);
    }

    // A numbered section at (roughly) body size is a heading regardless of size.
    let ratio = line.font_size / body_font_size;
    if (0.98..HEADING_MIN_RATIO).contains(&ratio) && NUMBERED_SECTION.is_match(text) {
        return Some(section_depth(text));
    }

    None
}

/// The pre-task-#16 fixed-ratio ladder, kept as the fallback used only when the
/// document has no adaptive size tiers.
fn fixed_ratio_level(ratio: f32) -> Option<usize> {
    if ratio >= 1.6 {
        Some(1)
    } else if ratio >= 1.3 {
        Some(2)
    } else if ratio >= HEADING_MIN_RATIO {
        Some(3)
    } else {
        None
    }
}

fn section_depth(text: &str) -> usize {
    text.split_whitespace()
        .next()
        .map(|number| number.trim_end_matches('.').matches('.').count() + 1)
        .unwrap_or(1)
        .min(6)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(text: &str, font_size: f32) -> Line {
        Line {
            text: text.to_string(),
            y: 0.0,
            font_size,
            cells: Vec::new(),
        }
    }

    /// Builds the scale the way the pipeline does: from the same lines heading
    /// detection will run over.
    fn scale_of(lines: &[Line], body: f32) -> HeadingScale {
        HeadingScale::build(lines.iter(), body)
    }

    #[test]
    fn large_font_is_a_top_level_heading() {
        let lines = [line("Chapter One", 20.0)];
        let scale = scale_of(&lines, 10.0);
        assert_eq!(heading_level(&lines[0], 10.0, &scale), Some(1));
    }

    #[test]
    fn body_sized_text_is_not_a_heading() {
        let lines = [line("Just a normal sentence.", 10.0)];
        let scale = scale_of(&lines, 10.0);
        assert_eq!(heading_level(&lines[0], 10.0, &scale), None);
    }

    #[test]
    fn numbered_section_at_body_size_is_a_heading() {
        let lines = [line("3.4 Hyperfocus", 10.0)];
        let scale = scale_of(&lines, 10.0);
        assert_eq!(heading_level(&lines[0], 10.0, &scale), Some(2));
    }

    #[test]
    fn overly_long_line_is_never_a_heading_even_if_large() {
        let text = "word ".repeat(60);
        let lines = [line(&text, 20.0)];
        let scale = scale_of(&lines, 10.0);
        assert_eq!(heading_level(&lines[0], 10.0, &scale), None);
    }

    #[test]
    fn three_font_tiers_map_to_h1_h2_h3_in_descending_size_order() {
        // Task #16 acceptance: three distinct heading sizes above a 10pt body
        // become H1/H2/H3 by rank, not by a fixed ratio (which would have made
        // 18pt and 24pt BOTH H1 under the old >=1.6 rule).
        let lines = [
            line("Biggest", 24.0),
            line("Middle", 18.0),
            line("Smallest heading", 14.0),
            line("body text here", 10.0),
        ];
        let scale = scale_of(&lines, 10.0);
        assert_eq!(heading_level(&lines[0], 10.0, &scale), Some(1));
        assert_eq!(heading_level(&lines[1], 10.0, &scale), Some(2));
        assert_eq!(heading_level(&lines[2], 10.0, &scale), Some(3));
        assert_eq!(heading_level(&lines[3], 10.0, &scale), None);
    }

    #[test]
    fn uniform_font_page_has_no_size_headings() {
        // Task #16 acceptance: a uniform-font page must not sprout false
        // headings. Every line is body size, so the scale is empty and no line
        // is promoted.
        let lines = [
            line("A line of text", 10.0),
            line("Another line of text", 10.0),
            line("A third line of text", 10.0),
        ];
        let scale = scale_of(&lines, 10.0);
        assert!(scale.is_empty());
        for l in &lines {
            assert_eq!(heading_level(l, 10.0, &scale), None);
        }
    }

    #[test]
    fn two_tiers_rank_relative_to_the_document_not_a_fixed_ratio() {
        // Only two heading sizes exist (13pt and 15pt over a 10pt body). Both are
        // below the old >=1.6 H1 threshold, yet the larger must still be H1 and
        // the smaller H2 -- levels are relative to this document.
        let lines = [line("Larger", 15.0), line("Smaller", 13.0)];
        let scale = scale_of(&lines, 10.0);
        assert_eq!(heading_level(&lines[0], 10.0, &scale), Some(1));
        assert_eq!(heading_level(&lines[1], 10.0, &scale), Some(2));
    }
}

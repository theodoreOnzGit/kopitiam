/// PASS threshold for [`ConversionReport::recovery_ratio`], the
/// non-whitespace-character-based headline signal.
///
/// The old word-count ratio needed a lenient 0.95 threshold because
/// re-tokenization (hyphenation repair, OCR word-gap merges, table-cell
/// splitting) routinely shifted word counts by several percent even with
/// zero content loss -- see kopitiam-wwr. The character-based signal is
/// immune to re-tokenization by construction (concatenating text with a
/// different choice of whitespace never changes a non-whitespace character
/// count), so the only things that can now depress the ratio are: real
/// content loss, or a gap in [`strip_rendered_markdown_syntax`]'s
/// normalization (an unhandled scaffolding pattern, or one of the
/// documented conservative false-positives in `strip_list_marker`). Both
/// failure modes are rare and small once they occur, so the threshold is
/// tightened to 0.98 rather than kept at 0.95: a genuinely complete
/// conversion should now land within a fraction of a percent of 100%, and
/// a 2%+ shortfall is a meaningful signal worth investigating rather than
/// noise to be tolerated.
///
/// [`strip_rendered_markdown_syntax`]: super::strip_rendered_markdown_syntax
const MIN_RECOVERY_RATIO: f64 = 0.98;

/// Audits one PDF-to-Markdown conversion: how much of the extracted text
/// survived into the rendered output, plus a tally of the structural
/// blocks found, so every conversion is auditable rather than a silent
/// best-effort guess.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ConversionReport {
    pub pages: usize,

    /// `split_whitespace().count()` word totals, kept for diagnostics only.
    /// Do **not** use these to judge recovery -- hyphenation repair, OCR
    /// word-gap merges, and table-cell tokenization all legitimately shift
    /// word counts by several percent with zero content loss, which made
    /// this ratio produce false FAILs on real documents (kopitiam-wwr).
    /// [`ConversionReport::recovery_ratio`] is the authoritative signal;
    /// these fields exist so a human auditing a report still has the raw
    /// word counts to look at.
    pub extracted_words: usize,
    pub rendered_words: usize,

    /// Non-whitespace character totals underlying the headline
    /// [`recovery_ratio`](Self::recovery_ratio). `extracted_chars` counts
    /// every `TextSpan` on every page (soft line-wrap hyphens excluded, see
    /// `is_wrap_hyphen`); `rendered_chars` counts the rendered Markdown
    /// after `strip_rendered_markdown_syntax` removes syntax the renderer
    /// added (heading hashes, list markers, table pipes/separators,
    /// blockquote markers, code fences, the figure placeholder).
    pub extracted_chars: usize,
    pub rendered_chars: usize,

    pub headings_found: usize,
    pub lists_found: usize,
    pub tables_found: usize,
    pub citations_found: usize,
}

impl ConversionReport {
    /// The authoritative recovery signal: `rendered_chars / extracted_chars`,
    /// both counted as non-whitespace characters after normalizing away
    /// renderer-added Markdown syntax and PDF line-wrap hyphenation
    /// artifacts (see `extracted_content_chars` and
    /// `strip_rendered_markdown_syntax` in `validation::mod`).
    ///
    /// **What this can detect:** content that was extracted from the PDF
    /// but never made it into the rendered Markdown -- a dropped
    /// paragraph, a truncated table, a figure caption that got lost. A
    /// character deficit of any size shows up directly as a ratio below
    /// 1.0, because unlike a word count, non-whitespace character count
    /// cannot be inflated or deflated by re-tokenizing the same content
    /// differently.
    ///
    /// **What this cannot detect:** content that survived but was
    /// *corrupted* or *reordered* -- garbled characters from a bad font
    /// mapping, two paragraphs swapped, a table's rows shuffled, a heading
    /// demoted to body text. All of those preserve the character count
    /// while changing or misplacing the content, so this ratio stays at
    /// ~100% and reports PASS regardless. This metric answers "did we lose
    /// content?", not "is the rendered document correct?" -- correctness
    /// still requires the structural tallies below and, ultimately, human
    /// review.
    pub fn recovery_ratio(&self) -> f64 {
        if self.extracted_chars == 0 {
            return 1.0;
        }
        self.rendered_chars as f64 / self.extracted_chars as f64
    }

    /// Secondary, informational-only word-count ratio. See the field docs
    /// on [`extracted_words`](Self::extracted_words) for why this is not
    /// used for PASS/FAIL: it is reported so a human can still see it, not
    /// because it is trustworthy on its own.
    pub fn word_recovery_ratio(&self) -> f64 {
        if self.extracted_words == 0 {
            return 1.0;
        }
        self.rendered_words as f64 / self.extracted_words as f64
    }

    /// PASS/FAIL against [`MIN_RECOVERY_RATIO`], using the character-based
    /// [`recovery_ratio`](Self::recovery_ratio) -- never the word-count
    /// ratio, which is informational only (see kopitiam-wwr).
    pub fn passes(&self) -> bool {
        self.recovery_ratio() >= MIN_RECOVERY_RATIO
    }
}

impl std::fmt::Display for ConversionReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Pages: {}", self.pages)?;
        writeln!(f)?;
        writeln!(f, "Recovery (characters, authoritative for PASS/FAIL):")?;
        writeln!(f, "  Extracted: {} chars", self.extracted_chars)?;
        writeln!(f, "  Rendered:  {} chars", self.rendered_chars)?;
        writeln!(f, "  Ratio:     {:.1}%", self.recovery_ratio() * 100.0)?;
        writeln!(f)?;
        writeln!(f, "Words (informational only, not authoritative -- see kopitiam-wwr):")?;
        writeln!(f, "  Extracted: {}", self.extracted_words)?;
        writeln!(f, "  Rendered:  {}", self.rendered_words)?;
        writeln!(f)?;
        writeln!(f, "Headings:  {}", self.headings_found)?;
        writeln!(f, "Lists:     {}", self.lists_found)?;
        writeln!(f, "Tables:    {}", self.tables_found)?;
        writeln!(f, "Citations: {}", self.citations_found)?;
        writeln!(f)?;
        write!(f, "Status: {}", if self.passes() { "PASS" } else { "FAIL" })
    }
}

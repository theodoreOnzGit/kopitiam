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

/// Character recovery for a single source page, attributed via the
/// `<!-- page N -->` anchors the renderer emits at page boundaries
/// (`kopitiam_token_max.md` §8 card I-B).
///
/// The document-wide [`ConversionReport::recovery_ratio`] answers "did we lose
/// content *somewhere*?"; these answer "*which page*?" — turning "97% overall,
/// re-check everything" into "page 4 is 71%, check only page 4". Their
/// `extracted_chars`/`rendered_chars` partition the document-wide totals
/// exactly (they sum back to [`ConversionReport::extracted_chars`] /
/// [`ConversionReport::rendered_chars`]), because the same normalization is
/// applied and the anchor lines themselves contribute zero content.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct PageRecovery {
    /// 1-based page number, matching the block's `Document::block_pages` value
    /// and the `<!-- page N -->` anchor.
    pub page: usize,
    pub extracted_chars: usize,
    pub rendered_chars: usize,
}

impl PageRecovery {
    /// This page's `rendered_chars / extracted_chars`, with the same
    /// empty-extraction convention as [`ConversionReport::recovery_ratio`]: a
    /// page from which nothing was extracted (e.g. one that was entirely a
    /// running head, now stripped) is full recovery, not a divide-by-zero.
    pub fn recovery_ratio(&self) -> f64 {
        if self.extracted_chars == 0 {
            return 1.0;
        }
        self.rendered_chars as f64 / self.extracted_chars as f64
    }
}

/// Audits one PDF-to-Markdown conversion: how much of the extracted text
/// survived into the rendered output, plus a tally of the structural
/// blocks found, so every conversion is auditable rather than a silent
/// best-effort guess.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
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

    /// Per-page character recovery, one entry per source page, in page order.
    /// Populated only when the `Document` carries page provenance
    /// (`Document::block_pages`); empty otherwise (a hand-built document has no
    /// page to attribute content to, and guessing would be dishonest). See
    /// [`PageRecovery`].
    pub per_page: Vec<PageRecovery>,
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

    /// A machine-readable JSON view of this report (`kopitiam pdf2md
    /// --report-json`, card I-F).
    ///
    /// # Why a bespoke view rather than the bare `#[derive(Serialize)]`
    ///
    /// The point of `--report-json` is that a caller gates on the recovery
    /// ratio *without parsing prose* (`kopitiam_token_max.md` §0.2). But the
    /// headline signals a caller actually gates on — [`recovery_ratio`], whether
    /// it [`passes`], and the **per-page** ratios from card I-B — are computed
    /// methods, not stored fields, so the raw field derive omits exactly them.
    /// This view emits the stored fields *and* those computed signals, so
    /// `recovery_ratio`, `passes`, and each page's `recovery_ratio` are present
    /// as first-class numbers a script can test directly.
    ///
    /// [`recovery_ratio`]: Self::recovery_ratio
    /// [`passes`]: Self::passes
    #[cfg(feature = "serde")]
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "pages": self.pages,
            "recovery_ratio": self.recovery_ratio(),
            "passes": self.passes(),
            "extracted_chars": self.extracted_chars,
            "rendered_chars": self.rendered_chars,
            "word_recovery_ratio": self.word_recovery_ratio(),
            "extracted_words": self.extracted_words,
            "rendered_words": self.rendered_words,
            "headings_found": self.headings_found,
            "lists_found": self.lists_found,
            "tables_found": self.tables_found,
            "citations_found": self.citations_found,
            "per_page": self
                .per_page
                .iter()
                .map(|p| {
                    serde_json::json!({
                        "page": p.page,
                        "extracted_chars": p.extracted_chars,
                        "rendered_chars": p.rendered_chars,
                        "recovery_ratio": p.recovery_ratio(),
                    })
                })
                .collect::<Vec<_>>(),
        })
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
        if !self.per_page.is_empty() {
            writeln!(f)?;
            writeln!(f, "  Per page:")?;
            for page in &self.per_page {
                writeln!(
                    f,
                    "    Page {:>3}: {:>5.1}%  ({}/{} chars)",
                    page.page,
                    page.recovery_ratio() * 100.0,
                    page.rendered_chars,
                    page.extracted_chars,
                )?;
            }
        }
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

#[cfg(all(test, feature = "serde"))]
mod json_tests {
    use super::*;

    fn sample() -> ConversionReport {
        ConversionReport {
            pages: 2,
            extracted_words: 10,
            rendered_words: 10,
            extracted_chars: 100,
            rendered_chars: 98,
            headings_found: 3,
            lists_found: 1,
            tables_found: 2,
            citations_found: 4,
            per_page: vec![
                PageRecovery { page: 1, extracted_chars: 60, rendered_chars: 60 },
                PageRecovery { page: 2, extracted_chars: 40, rendered_chars: 38 },
            ],
        }
    }

    #[test]
    fn to_json_carries_the_computed_signals_a_caller_gates_on() {
        // Card I-F / §0.2: the computed recovery_ratio + passes must be present
        // as first-class numbers, not left for the caller to re-derive.
        let json = sample().to_json();
        assert_eq!(json["pages"], 2);
        // 98 / 100 = 0.98 -> exactly the PASS threshold.
        assert!((json["recovery_ratio"].as_f64().unwrap() - 0.98).abs() < 1e-9);
        assert_eq!(json["passes"], true);
        assert_eq!(json["extracted_chars"], 100);
        assert_eq!(json["rendered_chars"], 98);
        assert_eq!(json["headings_found"], 3);
        assert_eq!(json["tables_found"], 2);
        assert_eq!(json["citations_found"], 4);
    }

    #[test]
    fn to_json_includes_per_page_ratios_from_card_i_b() {
        let json = sample().to_json();
        let per_page = json["per_page"].as_array().unwrap();
        assert_eq!(per_page.len(), 2);
        assert_eq!(per_page[0]["page"], 1);
        assert!((per_page[0]["recovery_ratio"].as_f64().unwrap() - 1.0).abs() < 1e-9);
        assert_eq!(per_page[1]["page"], 2);
        // 38 / 40 = 0.95: the per-page view localizes the shortfall to page 2.
        assert!((per_page[1]["recovery_ratio"].as_f64().unwrap() - 0.95).abs() < 1e-9);
    }
}

use kopitiam_document::{
    Block, CodeBlock, Document, Figure, Heading, List, Paragraph, Quote, Table,
};

pub trait RenderMarkdown {
    fn render(&self) -> String;
}

/// Options controlling [`render_document_with`].
///
/// A struct (rather than a bare `bool` parameter) so future rendering knobs
/// can be added without breaking every call site again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderOptions {
    /// Emit a `<!-- page N -->` HTML comment at each page boundary (see
    /// [`render_document_with`]). Defaults to **on**.
    ///
    /// # Why on by default
    ///
    /// KOPITIAM's primary consumer of `pdf2md` output is an agent that must
    /// *cite what it reads* — jump straight to a source page to verify a
    /// claim (`kopitiam_token_max.md` §8 card I-B). A page anchor is the
    /// cheapest possible affordance for that: an HTML comment renders to
    /// nothing in a Markdown viewer, costs a handful of tokens per page, and
    /// is trivially greppable (`rg -n "<!-- page 4 -->"`). The anchors are
    /// also what makes **per-page** recovery ratios computable in
    /// `kopitiam-document`'s validator, so leaving them on by default keeps
    /// that diagnostic available for every conversion. Turn them off only for
    /// a human-facing export where even invisible scaffolding is unwanted.
    pub page_anchors: bool,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self { page_anchors: true }
    }
}

/// Renders a full document with page anchors **on** (the agent default; see
/// [`RenderOptions::page_anchors`]). Back-compatible entry point kept so the
/// common call site stays `render_document(&doc)`.
///
/// Blocks are joined by a single blank line and the output always ends in
/// exactly one trailing newline, so re-rendering an unchanged `Document`
/// byte-for-byte reproduces the same file (deterministic output, stable diffs).
pub fn render_document(document: &Document) -> String {
    render_document_with(document, RenderOptions::default())
}

/// Formats the page-boundary anchor for page `n`.
///
/// The literal `<!-- page N -->` form is **duplicated** in
/// `kopitiam-document`'s validator, which must recognize and strip it before
/// counting rendered characters and must split the rendered side on it to
/// compute per-page recovery (`kopitiam_token_max.md` §2.1, §2.3). If this
/// wording changes, `validation/mod.rs`'s `parse_page_anchor` must change with
/// it — the two are kept in sync by tests on both sides, not by a shared
/// constant (dependencies flow markdown -> document, not the other way).
fn page_anchor(n: usize) -> String {
    format!("<!-- page {n} -->")
}

/// Renders a full document, optionally emitting a `<!-- page N -->` anchor at
/// each page boundary.
///
/// Blocks are joined by a single blank line and the output ends in exactly one
/// trailing newline. When [`RenderOptions::page_anchors`] is set and the
/// `Document` carries page provenance ([`Document::block_pages`]), a single
/// anchor is emitted at every boundary — the point where a block starts a
/// higher-numbered page than its predecessor — carrying that new page's number.
/// No anchor precedes the first page (its content is simply everything before
/// the first anchor), so there is exactly **one anchor per boundary**. A
/// `Document` without page information (e.g. one built by hand or via
/// `Default`) emits no anchors regardless of the option, because there is no
/// honest page number to attach.
pub fn render_document_with(document: &Document, options: RenderOptions) -> String {
    let mut pieces: Vec<String> = Vec::with_capacity(document.blocks.len());
    let mut previous_page: Option<usize> = None;

    for (index, block) in document.blocks.iter().enumerate() {
        if options.page_anchors {
            // `page_of` is `None` for a document without provenance, and only
            // then; in that case we never emit an anchor (nothing to cite).
            if let Some(page) = document.page_of(index) {
                if previous_page.is_some_and(|prev| prev != page) {
                    // A boundary: this block begins a new page. Page numbers are
                    // monotonic non-decreasing (blocks are in reading order), so
                    // this fires once per boundary and never emits a leading
                    // anchor for the very first page.
                    pieces.push(page_anchor(page));
                }
                previous_page = Some(page);
            }
        }
        pieces.push(block.render());
    }

    let mut out = pieces.join("\n\n");
    if !out.is_empty() {
        out.push('\n');
    }
    out
}

impl RenderMarkdown for Block {
    fn render(&self) -> String {
        match self {
            Block::Heading(heading) => heading.render(),
            Block::Paragraph(paragraph) => paragraph.render(),
            Block::List(list) => list.render(),
            Block::Table(table) => table.render(),
            Block::Figure(figure) => figure.render(),
            Block::CodeBlock(code) => code.render(),
            Block::Quote(quote) => quote.render(),
        }
    }
}

impl RenderMarkdown for Heading {
    fn render(&self) -> String {
        let level = self.level.clamp(1, 6);
        format!("{} {}", "#".repeat(level), self.text)
    }
}

impl RenderMarkdown for Paragraph {
    fn render(&self) -> String {
        self.text.clone()
    }
}

impl RenderMarkdown for List {
    /// Renders a (possibly nested) list. Each item is indented by two spaces per
    /// nesting level ([`List::depth`]) and marked with `- ` (unordered) or an
    /// `N. ` counter (ordered). An ordered list restarts its numbering within
    /// each nested sub-list, exactly as Markdown renders it, by keeping a
    /// per-depth counter that is discarded whenever the indent steps back out.
    ///
    /// A flat list (`depths` empty) is unaffected: every item is depth 0, so the
    /// output is byte-identical to the pre-nesting `- item` / `1. item` form.
    fn render(&self) -> String {
        // counters[d] = items already emitted at depth d in the current branch.
        let mut counters: Vec<usize> = Vec::new();
        self.items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let depth = self.depth(i);
                // Drop any deeper counters (a sub-list ended); the counter at
                // `depth` is preserved so siblings keep counting up.
                counters.truncate(depth + 1);
                if counters.len() <= depth {
                    counters.resize(depth + 1, 0);
                }
                counters[depth] += 1;

                let indent = "  ".repeat(depth);
                if self.ordered {
                    format!("{indent}{}. {item}", counters[depth])
                } else {
                    format!("{indent}- {item}")
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl RenderMarkdown for Table {
    fn render(&self) -> String {
        // Cells are not padded for visual column alignment: padding every
        // cell in a column to match the widest one means a single future
        // edit can reflow whitespace across the whole table. Unpadded rows
        // keep diffs to the lines that actually changed.
        let mut lines = Vec::with_capacity(self.rows.len() + 2);
        lines.push(render_row(&self.headers));
        lines.push(render_separator(self.headers.len().max(1)));
        for row in &self.rows {
            lines.push(render_row(row));
        }
        lines.join("\n")
    }
}

fn render_row(cells: &[String]) -> String {
    let escaped: Vec<String> = cells.iter().map(|cell| escape_cell(cell)).collect();
    format!("| {} |", escaped.join(" | "))
}

fn render_separator(column_count: usize) -> String {
    format!("| {} |", vec!["---"; column_count].join(" | "))
}

fn escape_cell(cell: &str) -> String {
    cell.replace('|', "\\|")
}

/// Single-token marker for a figure that has no caption. The image itself is
/// never extracted (the extraction layer recovers text spans only), so a
/// caption-less figure would otherwise vanish entirely; this keeps a minimal,
/// grep-able "a figure was here" signal at essentially zero token cost.
///
/// Must stay identical to `FIGURE_PLACEHOLDER` in
/// `kopitiam-document`'s `validation/mod.rs` (`kopitiam_token_max.md` §2.3):
/// validation strips exactly this string before counting rendered characters,
/// so if one changes the other must change with it.
const FIGURE_PLACEHOLDER: &str = "[figure]";

impl RenderMarkdown for Figure {
    fn render(&self) -> String {
        // A captioned figure renders the caption *alone*. The old
        // `{caption}\n\n[Figure omitted from Markdown output]` emission spent a
        // 37-char placeholder plus a blank line per figure carrying no
        // information (`kopitiam_token_max.md` §7 card I-D); the caption is the
        // only recovered content, so it is all we emit.
        match &self.caption {
            Some(caption) => caption.clone(),
            None => FIGURE_PLACEHOLDER.to_string(),
        }
    }
}

impl RenderMarkdown for CodeBlock {
    fn render(&self) -> String {
        let language = self.language.as_deref().unwrap_or("");
        format!("```{language}\n{}\n```", self.text)
    }
}

impl RenderMarkdown for Quote {
    fn render(&self) -> String {
        self.text
            .lines()
            .map(|line| format!("> {line}"))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use kopitiam_document::Metadata;

    use super::*;

    #[test]
    fn heading_renders_with_hashes_for_its_level() {
        let heading = Heading {
            level: 2,
            text: "Section".to_string(),
        };
        assert_eq!(heading.render(), "## Section");
    }

    #[test]
    fn ordered_list_numbers_items() {
        let list = List::flat(true, vec!["First".to_string(), "Second".to_string()]);
        assert_eq!(list.render(), "1. First\n2. Second");
    }

    #[test]
    fn unordered_list_uses_dashes() {
        let list = List::flat(false, vec!["A".to_string(), "B".to_string()]);
        assert_eq!(list.render(), "- A\n- B");
    }

    #[test]
    fn nested_unordered_list_indents_by_depth() {
        // Task #16: two-level nested bullet list renders with two-space indents
        // per level.
        let list = List {
            ordered: false,
            items: vec![
                "Top one".to_string(),
                "Nested a".to_string(),
                "Nested b".to_string(),
                "Top two".to_string(),
            ],
            depths: vec![0, 1, 1, 0],
        };
        assert_eq!(
            list.render(),
            "- Top one\n  - Nested a\n  - Nested b\n- Top two"
        );
    }

    #[test]
    fn nested_ordered_list_restarts_numbering_per_sublist() {
        // The sub-list numbering restarts at 1 and the parent resumes at 3.
        let list = List {
            ordered: true,
            items: vec![
                "First".to_string(),
                "Second".to_string(),
                "Sub of second".to_string(),
                "Third".to_string(),
            ],
            depths: vec![0, 0, 1, 0],
        };
        assert_eq!(
            list.render(),
            "1. First\n2. Second\n  1. Sub of second\n3. Third"
        );
    }

    #[test]
    fn table_escapes_pipes_and_has_no_column_padding() {
        let table = Table {
            headers: vec!["Metric".to_string(), "Value".to_string()],
            rows: vec![vec!["A|B".to_string(), "1".to_string()]],
        };
        assert_eq!(
            table.render(),
            "| Metric | Value |\n| --- | --- |\n| A\\|B | 1 |"
        );
    }

    #[test]
    fn figure_without_caption_still_renders_placeholder() {
        // Updated deliberately for I-D (`kopitiam_token_max.md` §7, §2.5): the
        // placeholder was shortened from the 37-char
        // "[Figure omitted from Markdown output]" to the single-token "[figure]"
        // to stop spending tokens on a per-figure boilerplate line. A
        // caption-less figure still emits *a* marker (the image is never
        // extracted, so without it the figure would vanish silently); it is just
        // as short as possible now. Must match `FIGURE_PLACEHOLDER` in
        // kopitiam-document's validation/mod.rs (§2.3).
        let figure = Figure {
            caption: None,
            image_path: None,
        };
        assert_eq!(figure.render(), "[figure]");
    }

    #[test]
    fn captioned_figure_renders_the_caption_alone_with_no_placeholder() {
        // The I-D win: a captioned figure is now just its caption -- no
        // "[Figure omitted ...]" line, no extra blank line.
        let figure = Figure {
            caption: Some("Fig. 1 System overview".to_string()),
            image_path: None,
        };
        assert_eq!(figure.render(), "Fig. 1 System overview");
    }

    #[test]
    fn document_blocks_are_joined_by_a_blank_line_with_trailing_newline() {
        // Updated deliberately for I-B (`kopitiam_token_max.md` §8, §2.5). The
        // renderer now emits a `<!-- page N -->` anchor at each page boundary
        // when `page_anchors` is on (the default). This test guards the
        // *block-joining* invariant — blocks separated by exactly one blank
        // line, output ending in one trailing newline — which anchors must not
        // disturb, so it now pins the anchors-**off** path explicitly. The
        // fixture is given real page provenance that spans a boundary
        // (`block_pages = [1, 2]`) precisely so this asserts anchors are
        // suppressed when off, rather than being vacuously absent. The
        // anchors-on behavior is covered by the two tests below.
        let document = Document {
            title: None,
            metadata: Metadata::default(),
            block_pages: vec![1, 2],
            blocks: vec![
                Block::Heading(Heading {
                    level: 1,
                    text: "Title".to_string(),
                }),
                Block::Paragraph(Paragraph {
                    text: "Body text.".to_string(),
                }),
            ],
            citations: Vec::new(),
        };
        let rendered = render_document_with(&document, RenderOptions { page_anchors: false });
        assert_eq!(rendered, "# Title\n\nBody text.\n");
    }

    #[test]
    fn page_anchors_default_on_emit_one_anchor_per_boundary_matching_block_pages() {
        // Multi-page fixture: three blocks on pages 1, 2, 2. There is exactly
        // one page boundary (1 -> 2), so exactly one anchor is emitted, and it
        // carries the new page's number from `block_pages`. No leading anchor
        // precedes page 1, and the second page-2 block adds no second anchor.
        let document = Document {
            title: None,
            metadata: Metadata::default(),
            block_pages: vec![1, 2, 2],
            blocks: vec![
                Block::Paragraph(Paragraph {
                    text: "Page one body.".to_string(),
                }),
                Block::Heading(Heading {
                    level: 2,
                    text: "Second Page".to_string(),
                }),
                Block::Paragraph(Paragraph {
                    text: "Page two body.".to_string(),
                }),
            ],
            citations: Vec::new(),
        };
        let rendered = render_document(&document);
        assert_eq!(
            rendered,
            "Page one body.\n\n<!-- page 2 -->\n\n## Second Page\n\nPage two body.\n"
        );
        // Exactly one anchor, and it is literal + greppable (acceptance:
        // `rg -n "<!-- page 2 -->"` locates the page).
        assert_eq!(rendered.matches("<!-- page ").count(), 1);
        assert!(rendered.contains("<!-- page 2 -->"));
    }

    #[test]
    fn page_anchor_number_jumps_when_a_page_has_no_blocks() {
        // A boundary that skips a page number (page 2 produced no blocks) still
        // emits exactly one anchor, carrying the *next* page that actually has
        // content (page 3), so the anchor number always matches `block_pages`.
        let document = Document {
            title: None,
            metadata: Metadata::default(),
            block_pages: vec![1, 3],
            blocks: vec![
                Block::Paragraph(Paragraph {
                    text: "First.".to_string(),
                }),
                Block::Paragraph(Paragraph {
                    text: "Third.".to_string(),
                }),
            ],
            citations: Vec::new(),
        };
        assert_eq!(
            render_document(&document),
            "First.\n\n<!-- page 3 -->\n\nThird.\n"
        );
    }

    #[test]
    fn a_document_without_page_provenance_emits_no_anchors_even_with_anchors_on() {
        // `Default`/hand-built documents carry no page numbers; there is no
        // honest page to cite, so no anchor is emitted regardless of the option.
        let document = Document {
            title: None,
            metadata: Metadata::default(),
            block_pages: Vec::new(),
            blocks: vec![
                Block::Paragraph(Paragraph {
                    text: "One.".to_string(),
                }),
                Block::Paragraph(Paragraph {
                    text: "Two.".to_string(),
                }),
            ],
            citations: Vec::new(),
        };
        assert_eq!(render_document(&document), "One.\n\nTwo.\n");
    }
}

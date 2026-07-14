use kopitiam_document::{
    Block, CodeBlock, Document, Figure, Heading, List, Paragraph, Quote, Table,
};

pub trait RenderMarkdown {
    fn render(&self) -> String;
}

/// Renders a full document. Blocks are joined by a single blank line and the
/// output always ends in exactly one trailing newline, so re-rendering an
/// unchanged `Document` byte-for-byte reproduces the same file (deterministic
/// output, stable diffs).
pub fn render_document(document: &Document) -> String {
    let mut out = document
        .blocks
        .iter()
        .map(RenderMarkdown::render)
        .collect::<Vec<_>>()
        .join("\n\n");
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
    fn render(&self) -> String {
        self.items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                if self.ordered {
                    format!("{}. {item}", i + 1)
                } else {
                    format!("- {item}")
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

impl RenderMarkdown for Figure {
    fn render(&self) -> String {
        match &self.caption {
            Some(caption) => format!("{caption}\n\n[Figure omitted from Markdown output]"),
            None => "[Figure omitted from Markdown output]".to_string(),
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
        let list = List {
            ordered: true,
            items: vec!["First".to_string(), "Second".to_string()],
        };
        assert_eq!(list.render(), "1. First\n2. Second");
    }

    #[test]
    fn unordered_list_uses_dashes() {
        let list = List {
            ordered: false,
            items: vec!["A".to_string(), "B".to_string()],
        };
        assert_eq!(list.render(), "- A\n- B");
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
        let figure = Figure {
            caption: None,
            image_path: None,
        };
        assert_eq!(figure.render(), "[Figure omitted from Markdown output]");
    }

    #[test]
    fn document_blocks_are_joined_by_a_blank_line_with_trailing_newline() {
        let document = Document {
            title: None,
            metadata: Metadata::default(),
            // Rendering does not need page provenance; this fixture is
            // hand-built rather than reconstructed, so it honestly has none.
            block_pages: Vec::new(),
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
        assert_eq!(render_document(&document), "# Title\n\nBody text.\n");
    }
}

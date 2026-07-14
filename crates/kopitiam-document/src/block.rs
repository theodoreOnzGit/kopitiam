use crate::{Figure, Heading, List, Paragraph, Table};

#[derive(Debug, Clone, PartialEq)]
pub enum Block {
    Heading(Heading),
    Paragraph(Paragraph),
    List(List),
    Table(Table),
    Figure(Figure),
    CodeBlock(CodeBlock),
    Quote(Quote),
}

/// No reconstruction pass produces this yet -- code listings currently fall
/// through to `Paragraph`. Kept in the AST so renderers and future
/// reconstruction heuristics have a target to produce/consume.
#[derive(Debug, Clone, PartialEq)]
pub struct CodeBlock {
    pub text: String,
    pub language: Option<String>,
}

/// No reconstruction pass produces this yet -- block quotes currently fall
/// through to `Paragraph`.
#[derive(Debug, Clone, PartialEq)]
pub struct Quote {
    pub text: String,
}

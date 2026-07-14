//! Markdown rendering: turns a `kopitiam_document::Document` AST into
//! deterministic, human-readable, git-diff-friendly Markdown text.

mod renderer;

pub use renderer::{RenderMarkdown, render_document};

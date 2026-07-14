//! Extract a bibliography from a PDF and report, honestly, what was recovered.
//!
//! ```text
//! cargo run --release -p kopitiam-bibliography --example extract -- paper.pdf
//! ```
//!
//! This is the crate's own audit tool, and it is deliberately blunt: it prints
//! what parsed, what only partly parsed, what did not parse at all, and every
//! assumption made along the way. A tool that showed only the successes would be
//! the exact failure this crate is written to avoid.

use std::process::ExitCode;

use kopitiam_bibliography::bibtex::{Dialect, emit_references};
use kopitiam_bibliography::hayagriva::emit_hayagriva;
use kopitiam_bibliography::{ParsedReference, extract_pdf, to_graph};

fn main() -> ExitCode {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: extract <paper.pdf> [--bib | --yaml | --graph]");
        return ExitCode::FAILURE;
    };
    let mode = std::env::args().nth(2).unwrap_or_default();

    let bibliography = match extract_pdf(&path) {
        Ok(bibliography) => bibliography,
        Err(error) => {
            eprintln!("could not extract {path}: {error}");
            return ExitCode::FAILURE;
        }
    };

    let references: Vec<_> = bibliography.references().cloned().collect();

    match mode.as_str() {
        "--bib" => {
            print!("{}", emit_references(&references, Dialect::Biblatex));
            return ExitCode::SUCCESS;
        }
        "--yaml" => {
            print!("{}", emit_hayagriva(&references));
            return ExitCode::SUCCESS;
        }
        "--graph" => {
            let graph = to_graph(&bibliography);
            println!(
                "{} entities, {} relationships",
                graph.entities.len(),
                graph.relationships.len()
            );
            for entity in &graph.entities {
                println!("  {:?}  {}", entity.kind, entity.name);
            }
            return ExitCode::SUCCESS;
        }
        _ => {}
    }

    println!("=== {} ===", bibliography.document());
    println!();

    let parsed = bibliography
        .entries()
        .iter()
        .filter(|e| matches!(e, ParsedReference::Parsed(_)))
        .count();
    let partial = bibliography.partial().count();
    let unparsed = bibliography.unparsed().count();

    println!("REFERENCE LIST: {} entries", bibliography.entries().len());
    println!("  fully parsed:     {parsed}");
    println!("  partially parsed: {partial}");
    println!("  not parsed:       {unparsed}");
    println!();

    for (index, entry) in bibliography.entries().iter().enumerate() {
        let label = index + 1;
        match entry {
            ParsedReference::Parsed(reference) | ParsedReference::Partial(reference) => {
                let status = if entry.is_partial() { "PARTIAL" } else { "ok     " };
                println!(
                    "[{label:>2}] {status} {:?}",
                    reference.title().unwrap_or("(no title)")
                );
                println!(
                    "          kind={} year={} authors={}{}",
                    reference.kind(),
                    reference
                        .year()
                        .map(|y| y.to_string())
                        .unwrap_or_else(|| "?".into()),
                    reference.authors().len(),
                    if reference.authors().is_truncated() {
                        "+ (et al.)"
                    } else {
                        ""
                    },
                );
                for author in reference.authors().authors() {
                    println!(
                        "          - {:?} (family: {:?})",
                        author.as_written(),
                        author.family()
                    );
                }
                if let Some(container) = reference.container() {
                    println!("          in: {container:?}");
                }
                if let Some(institution) = reference.institution() {
                    println!("          institution: {institution:?}");
                }
                if let Some(pages) = reference.pages() {
                    println!("          pages: {pages}");
                }
                if !reference.identifiers().any() {
                    println!("          NO IDENTIFIER -- would need a network resolver");
                }
                if let Some(remainder) = reference.unparsed() {
                    println!("          NOT UNDERSTOOD: {remainder:?}");
                }
            }
            ParsedReference::Unparsed(raw) => {
                println!("[{label:>2}] UNPARSED {:?}", raw.text());
            }
        }
    }

    println!();
    println!("CITATIONS: {} found in the body", bibliography.citations().len());
    println!("  resolved to an entry: {}", bibliography.resolve_citations().len());
    println!("  unresolved:           {}", bibliography.unresolved_citations().len());
    println!("  distinct works cited: {}", bibliography.cited_entry_count());
    let uncited = bibliography.uncited_entries();
    if !uncited.is_empty() {
        println!(
            "  listed but never cited in the body: {:?}",
            uncited.iter().map(|i| i + 1).collect::<Vec<_>>()
        );
    }

    println!();
    println!("ANOMALIES: {}", bibliography.anomalies().len());
    for anomaly in bibliography.anomalies() {
        println!("  - {}", anomaly.summary());
    }

    ExitCode::SUCCESS
}

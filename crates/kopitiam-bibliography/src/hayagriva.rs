//! Hayagriva — **Typst's** native bibliography format.
//!
//! CLAUDE.md lists Typst among the Long-Term Goals, and Typst is the more
//! plausible destination for KOPITIAM's scientific publishing than LaTeX is: it
//! is pure Rust, it compiles in milliseconds, and it does not require a
//! `texlive` install measured in gigabytes.
//!
//! # Typst reads `.bib` too, so why this?
//!
//! Typst accepts BibTeX/BibLaTeX `.bib` files directly, so [`crate::bibtex`]
//! already serves it. Hayagriva YAML is emitted as well because it is Typst's
//! *native* format and carries two things BibTeX cannot:
//!
//! * a **structured parent** (`parent: { type: periodical, title: ..., volume:
//!   ... }`), which models "this article appeared in that journal" as a
//!   relationship rather than as three loose fields; and
//! * a **`serial-number` map** holding DOI, ISBN, ISSN and arXiv id side by
//!   side, rather than BibTeX's ad-hoc field-per-identifier convention that no
//!   two styles agree on.
//!
//! # Why the YAML is hand-emitted
//!
//! No `serde_yaml`, for two reasons that both come from CLAUDE.md. First,
//! "avoid unnecessary dependencies": the subset of YAML needed here is small and
//! entirely within a few dozen lines. Second, and more important,
//! **determinism** — a serializer's key order is its own business, and this
//! crate promises byte-identical output on every run. Hand-emission makes that a
//! property of the code rather than a hope about a dependency.
//!
//! Every string is emitted double-quoted with `\` and `"` escaped, which is
//! valid YAML for any content and removes the entire class of "did this need
//! quoting?" bugs.

use std::fmt::Write as _;

use crate::author::{Author, NameConfidence};
use crate::bibtex::disambiguate_keys;
use crate::reference::Reference;

/// Emits a bibliography as Hayagriva YAML, for `#bibliography("refs.yml")` in
/// Typst.
///
/// Deterministic: the same references always produce byte-identical output.
/// Keys are disambiguated exactly as in [`crate::bibtex::emit_references`], so a
/// document can be moved between the two formats without its `\cite` keys
/// changing.
///
/// ```
/// use kopitiam_bibliography::{
///     DocumentId, EntryKind, Provenance, Reference, Year,
///     author::parse_printed_name_list, hayagriva::emit_hayagriva,
/// };
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let doc = DocumentId::new("paper.pdf")?;
/// let provenance = Provenance::from_page(&doc, 15, "M. R. Chen, MTAT, 2024.")?;
///
/// let reference = Reference::builder(provenance)
///     .kind(EntryKind::Article)
///     .authors(parse_printed_name_list("M. R. Chen, S. Novak"))
///     .title("An open-source toolkit")
///     .container("IJCLTP")
///     .year(Year::new(2024)?)
///     .build();
///
/// let yaml = emit_hayagriva(&[reference]);
/// assert!(yaml.starts_with("chen2024open:\n"));
/// assert!(yaml.contains("  type: article\n"));
/// assert!(yaml.contains("    title: \"IJCLTP\"\n"), "{yaml}");
/// # Ok(())
/// # }
/// ```
pub fn emit_hayagriva(references: &[Reference]) -> String {
    let keys = disambiguate_keys(references);
    let mut out = String::new();

    for (reference, key) in references.iter().zip(&keys) {
        let _ = writeln!(out, "{key}:");
        let _ = writeln!(out, "  type: {}", reference.kind().hayagriva_type());

        if let Some(title) = reference.title() {
            let _ = writeln!(out, "  title: {}", quote(title));
        }

        if !reference.authors().is_empty() {
            let _ = writeln!(out, "  author:");
            for author in reference.authors().authors() {
                let _ = writeln!(out, "    - {}", quote(&hayagriva_name(author)));
            }
            if reference.authors().is_truncated() {
                // Hayagriva has no `et al.` marker, so the truncation is recorded
                // in the note rather than being silently dropped -- an author
                // list that LOOKS complete when it is not is a lie about the
                // provenance of a result.
                let _ = writeln!(out, "    # author list truncated in the source (et al.)");
            }
        }

        if !reference.editors().is_empty() {
            let _ = writeln!(out, "  editor:");
            for editor in reference.editors().authors() {
                let _ = writeln!(out, "    - {}", quote(&hayagriva_name(editor)));
            }
        }

        if let Some(year) = reference.year() {
            let _ = writeln!(out, "  date: {year}");
        }

        if let Some(pages) = reference.pages() {
            let printed = match pages.end() {
                Some(end) => format!("{}-{}", pages.start(), end),
                None => pages.start().to_string(),
            };
            let _ = writeln!(out, "  page-range: {}", quote(&printed));
        }

        if let Some(publisher) = reference.publisher() {
            let _ = writeln!(out, "  publisher: {}", quote(publisher));
        }
        if let Some(institution) = reference.institution() {
            let _ = writeln!(out, "  organization: {}", quote(institution));
        }
        if let Some(edition) = reference.edition() {
            let _ = writeln!(out, "  edition: {}", quote(edition));
        }

        // The structured parent: the thing that makes Hayagriva richer than a
        // `.bib` entry. "This article appeared in that journal" is a
        // relationship, not three loose fields.
        if let Some(container) = reference.container() {
            let _ = writeln!(out, "  parent:");
            let parent_type = match reference.kind() {
                crate::EntryKind::InProceedings => "proceedings",
                crate::EntryKind::InCollection => "book",
                _ => "periodical",
            };
            let _ = writeln!(out, "    type: {parent_type}");
            let _ = writeln!(out, "    title: {}", quote(container));
            if let Some(volume) = reference.volume() {
                let _ = writeln!(out, "    volume: {}", quote(volume));
            }
            if let Some(issue) = reference.issue() {
                let _ = writeln!(out, "    issue: {}", quote(issue));
            }
            if let Some(issn) = &reference.identifiers().issn {
                let _ = writeln!(out, "    serial-number:");
                let _ = writeln!(out, "      issn: {}", quote(issn.as_str()));
            }
        }

        // The serial-number map: every identifier in one place, which BibTeX has
        // never managed to agree on.
        let ids = reference.identifiers();
        if ids.doi.is_some() || ids.arxiv.is_some() || ids.isbn.is_some() {
            let _ = writeln!(out, "  serial-number:");
            if let Some(doi) = &ids.doi {
                let _ = writeln!(out, "    doi: {}", quote(doi.as_str()));
            }
            if let Some(arxiv) = &ids.arxiv {
                let _ = writeln!(out, "    arxiv: {}", quote(arxiv.as_str()));
            }
            if let Some(isbn) = &ids.isbn {
                let _ = writeln!(out, "    isbn: {}", quote(isbn.as_str()));
            }
        }
        if let Some(url) = &ids.url {
            let _ = writeln!(out, "  url: {}", quote(url.as_str()));
        }

        // What we could not work out travels with the entry, here as everywhere.
        let mut notes: Vec<String> = Vec::new();
        if let Some(note) = reference.note() {
            notes.push(note.to_string());
        }
        if let Some(unparsed) = reference.unparsed() {
            notes.push(format!("unparsed remainder: {unparsed}"));
        }
        if !notes.is_empty() {
            let _ = writeln!(out, "  note: {}", quote(&notes.join("; ")));
        }

        out.push('\n');
    }

    out
}

/// Renders an author for Hayagriva.
///
/// # The same name-safety rule as [`crate::bibtex`]
///
/// Hayagriva splits `"Family, Given"` on the comma, exactly as BibTeX does. So
/// the rule is identical and for the identical reason: a name whose split is only
/// [`NameConfidence::Assumed`] is written **as it appeared**, with no comma, so
/// that nothing downstream is told a family name we do not actually know.
fn hayagriva_name(author: &Author) -> String {
    match author {
        Author::Organization(name) | Author::Literal(name) => name.clone(),
        Author::Person(person) => match person.confidence() {
            NameConfidence::Explicit | NameConfidence::Conventional => {
                let given = person
                    .given()
                    .iter()
                    .map(crate::GivenName::as_display)
                    .collect::<Vec<_>>()
                    .join(" ");
                if given.is_empty() {
                    person.full_family()
                } else {
                    format!("{}, {given}", person.full_family())
                }
            }
            // We do not know which part is the family name. Say the name, not our
            // guess at its structure.
            NameConfidence::Assumed => person.as_written().to_string(),
        },
    }
}

/// Double-quotes a YAML scalar, escaping `\` and `"`.
///
/// Always quoting removes the entire class of "did this string need quoting?"
/// bugs — a title beginning with `-`, containing `: `, or consisting of the word
/// `yes` are all YAML traps, and none of them are hypothetical in a
/// bibliography.
fn quote(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::author::parse_printed_name_list;
    use crate::identifier::{Doi, Identifiers};
    use crate::provenance::{DocumentId, Provenance};
    use crate::reference::{EntryKind, Page, PageRange, Year};

    fn provenance() -> Provenance {
        let doc = DocumentId::new("paper.pdf").unwrap();
        Provenance::from_page(&doc, 15, "a reference line").unwrap()
    }

    #[test]
    fn emits_a_journal_article_with_a_structured_parent() {
        let reference = Reference::builder(provenance())
            .kind(EntryKind::Article)
            .authors(parse_printed_name_list("M. R. Chen, S. Novak, and J. P. Alvarez"))
            .title("An open-source toolkit for multilingual text alignment")
            .container("International Journal of Computational Linguistics")
            .volume("6")
            .issue("4")
            .pages(PageRange::new(Page::new("281").unwrap(), Page::new("301")))
            .year(Year::new(2024).unwrap())
            .identifiers(Identifiers {
                doi: Some(Doi::parse("10.1016/j.x").unwrap()),
                ..Default::default()
            })
            .build();

        let yaml = emit_hayagriva(&[reference]);

        assert!(yaml.starts_with("chen2024open:\n"), "{yaml}");
        assert!(yaml.contains("  type: article\n"), "{yaml}");
        assert!(yaml.contains("    - \"Chen, M. R.\"\n"), "{yaml}");
        assert!(yaml.contains("  date: 2024\n"), "{yaml}");
        assert!(yaml.contains("  page-range: \"281-301\"\n"), "{yaml}");
        // The structured parent -- what BibTeX cannot say.
        assert!(yaml.contains("  parent:\n    type: periodical\n"), "{yaml}");
        assert!(yaml.contains("    volume: \"6\"\n"), "{yaml}");
        // The serial-number map.
        assert!(yaml.contains("  serial-number:\n    doi: \"10.1016/j.x\"\n"), "{yaml}");
    }

    #[test]
    fn an_assumed_name_is_never_given_a_comma_it_did_not_have() {
        // The same safety rule as BibTeX: Hayagriva splits on the comma too, so
        // writing "Zedong, Mao" would rename him in a Typst bibliography.
        let reference = Reference::builder(provenance())
            .authors(parse_printed_name_list("Mao Zedong"))
            .title("On Practice")
            .year(Year::new(1937).unwrap())
            .build();

        let yaml = emit_hayagriva(&[reference]);
        assert!(yaml.contains("    - \"Mao Zedong\"\n"), "{yaml}");
        assert!(!yaml.contains("Zedong, Mao"), "must not reorder an assumed name");
    }

    #[test]
    fn a_trustworthy_name_is_written_family_first_so_typst_can_alphabetise_it() {
        let reference = Reference::builder(provenance())
            .authors(parse_printed_name_list("J. D. van der Waals"))
            .title("On the continuity")
            .year(Year::new(1873).unwrap())
            .build();
        let yaml = emit_hayagriva(&[reference]);
        assert!(yaml.contains("    - \"van der Waals, J. D.\"\n"), "{yaml}");
    }

    #[test]
    fn an_organisation_keeps_its_name_whole() {
        let reference = Reference::builder(provenance())
            .authors(parse_printed_name_list("European Bioinformatics Institute"))
            .title("A report")
            .year(Year::new(2019).unwrap())
            .build();
        let yaml = emit_hayagriva(&[reference]);
        assert!(yaml.contains("    - \"European Bioinformatics Institute\"\n"), "{yaml}");
    }

    #[test]
    fn a_truncated_author_list_is_marked_and_not_passed_off_as_complete() {
        let reference = Reference::builder(provenance())
            .authors(parse_printed_name_list("K. R. Fulton et al."))
            .title("A paper")
            .year(Year::new(2021).unwrap())
            .build();
        let yaml = emit_hayagriva(&[reference]);
        assert!(yaml.contains("truncated"), "{yaml}");
    }

    #[test]
    fn strings_are_quoted_and_escaped_so_yaml_traps_cannot_fire() {
        // A title beginning with `-`, containing `: `, or consisting of the word
        // `yes` are all YAML traps, and none are hypothetical in a bibliography.
        assert_eq!(quote("a: b"), "\"a: b\"");
        assert_eq!(quote("- leading dash"), "\"- leading dash\"");
        assert_eq!(quote(r#"He said "hi""#), r#""He said \"hi\"""#);
        assert_eq!(quote(r"back\slash"), r#""back\\slash""#);
    }

    #[test]
    fn emission_is_byte_identical_across_runs() {
        let reference = Reference::builder(provenance())
            .kind(EntryKind::Article)
            .authors(parse_printed_name_list("M. R. Chen, S. Novak"))
            .title("MTAT")
            .container("IJCLTP")
            .volume("6")
            .year(Year::new(2024).unwrap())
            .build();
        let references = vec![reference];

        let first = emit_hayagriva(&references);
        for _ in 0..500 {
            assert_eq!(emit_hayagriva(&references), first);
        }
    }

    #[test]
    fn keys_match_the_bibtex_emitters_so_a_document_can_move_between_formats() {
        let make = |title: &str| {
            Reference::builder(provenance())
                .authors(parse_printed_name_list("M. R. Chen"))
                .title(title)
                .year(Year::new(2024).unwrap())
                .build()
        };
        let references = vec![make("Digital twins one"), make("Digital twins two")];

        let yaml = emit_hayagriva(&references);
        let bib = crate::bibtex::emit_references(&references, crate::bibtex::Dialect::Bibtex);

        for key in ["chen2024digital", "chen2024digitalb"] {
            assert!(yaml.contains(&format!("{key}:")), "yaml missing {key}");
            assert!(bib.contains(&format!("{{{key},")), "bib missing {key}");
        }
    }
}

//! BibTeX / BibLaTeX emission — deterministic, and safe with names.
//!
//! # Determinism
//!
//! CLAUDE.md requires deterministic behaviour, and a bibliography emitter is a
//! place where non-determinism does concrete damage: a `.bib` file whose field
//! order shuffles between runs produces a diff on every save, which makes it
//! impossible to see the change you actually made.
//!
//! So **nothing here iterates a `HashMap`**. Two orderings exist, and both are
//! total:
//!
//! * Round-tripping a parsed file — fields come out **in the order they went
//!   in**, because that order is the user's and not ours to normalise.
//! * Emitting a [`Reference`] this crate built — fields come out in
//!   [`CANONICAL_FIELD_ORDER`], a fixed `const` array.
//!
//! Byte-identical output across runs is asserted in the tests, not merely
//! intended.
//!
//! # The name-safety rule
//!
//! This is the module where the author-name work in [`crate::author`] either
//! pays off or is thrown away, so it is worth stating in one place:
//!
//! > **A name whose split is [`NameConfidence::Assumed`] is emitted exactly as
//! > it was written, never reordered into `Family, Given` form.**
//!
//! `Mao Zedong` is emitted as `Mao Zedong`. It is **not** emitted as
//! `Zedong, Mao` — which is what a naive "last token is the surname" reordering
//! would produce, and which BibTeX would then typeset in a published
//! bibliography as *"M. Zedong"*, renaming a real person in a document that
//! carries someone else's name as author.
//!
//! When we *do* know the family name — because the source used a comma
//! ([`NameConfidence::Explicit`]) or because the shape said so
//! ([`NameConfidence::Conventional`]: `van der Waals`, `M. R. Chen`) —
//! reordering is safe and we do it, because `van der Waals, J. D.` is what lets
//! a bibliography style alphabetise him correctly.
//!
//! Organisations and unsplittable names are wrapped in `{{double braces}}`,
//! which is **BibTeX's own idiom** for "this is one unit, do not take it
//! apart". We are not inventing a convention; we are using the one the format
//! already has.

use std::fmt::Write as _;

use super::{BibDatabase, Component, Item, Value};
use crate::author::{Author, AuthorList, NameConfidence};
use crate::reference::Reference;

/// Which flavour of the format to emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Dialect {
    /// Classic BibTeX (`bibtex`). Entry types limited to the traditional set;
    /// `doi`/`url` are tolerated by most styles but are not standard.
    #[default]
    Bibtex,

    /// BibLaTeX (`biber`). Richer entry types (`@online`, `@software`,
    /// `@report`), and `date`/`journaltitle` in place of `year`/`journal`.
    ///
    /// Typst reads BibTeX/BibLaTeX `.bib` files natively, so this output is
    /// also a perfectly good Typst bibliography — see [`crate::hayagriva`] for
    /// Typst's *native* format, which carries a little more.
    Biblatex,
}

/// The order fields are emitted in for a [`Reference`] this crate built.
///
/// A `const` array, so emission is deterministic by construction rather than by
/// discipline. The order is the conventional reading order of a citation — who,
/// what, where, when, how to find it — which is also, not coincidentally, the
/// order that makes a `.bib` file readable by a human.
pub const CANONICAL_FIELD_ORDER: &[&str] = &[
    "author",
    "editor",
    "title",
    "journal",
    "journaltitle",
    "booktitle",
    "school",
    "institution",
    "publisher",
    "edition",
    "volume",
    "number",
    "pages",
    "year",
    "date",
    "doi",
    "eprint",
    "eprinttype",
    "isbn",
    "issn",
    "url",
    "note",
];

/// Emits a parsed database back to `.bib` source.
///
/// The output re-parses to a database **equal to the input** — the fixed point
/// property that `tests/roundtrip.rs` asserts over hundreds of generated cases.
///
/// Items are separated by exactly one blank line, and [`Item::Ignored`] text is
/// written verbatim. That combination is what closes the loop: the parser drops
/// whitespace-only runs, so the blank lines we add here do not accumulate as
/// `Ignored` items on the next parse.
pub fn emit_database(db: &BibDatabase) -> String {
    let mut out = String::new();

    for item in db.items() {
        match item {
            Item::Entry(entry) => {
                let _ = write!(out, "@{}{{{}", entry.entry_type, entry.key);
                for (name, value) in &entry.fields {
                    let _ = write!(out, ",\n  {name} = {}", emit_value(value));
                }
                out.push_str(if entry.fields.is_empty() { "}" } else { ",\n}" });
            }
            Item::String { name, value } => {
                let _ = write!(out, "@string{{{name} = {}}}", emit_value(value));
            }
            Item::Preamble(value) => {
                let _ = write!(out, "@preamble{{{}}}", emit_value(value));
            }
            Item::Comment(text) => {
                let _ = write!(out, "@comment{{{text}}}");
            }
            Item::Ignored(text) => {
                out.push_str(text);
            }
        }
        out.push_str("\n\n");
    }

    out
}

/// Emits one field value, components joined by ` # `.
fn emit_value(value: &Value) -> String {
    value
        .components()
        .iter()
        .map(|component| match component {
            Component::Braced(text) => format!("{{{text}}}"),
            Component::Quoted(text) => format!("\"{text}\""),
            Component::Number(text) | Component::Macro(text) => text.clone(),
        })
        .collect::<Vec<_>>()
        .join(" # ")
}

/// Emits one [`Reference`] as a `.bib` entry, using its
/// [`suggested_key`](Reference::suggested_key).
pub fn emit_reference(reference: &Reference, dialect: Dialect) -> String {
    emit_reference_with_key(reference, reference.suggested_key().as_str(), dialect)
}

/// Emits a whole bibliography, with **collision-free** citation keys.
///
/// Two papers by the same first author in the same year with the same leading
/// title word would otherwise get the same key, and BibTeX would silently use
/// only one of them. They are disambiguated with `a`, `b`, `c` suffixes — the
/// same way a human would, in the same deterministic order (source order),
/// every time.
pub fn emit_references(references: &[Reference], dialect: Dialect) -> String {
    let keys = disambiguate_keys(references);
    references
        .iter()
        .zip(&keys)
        .map(|(reference, key)| emit_reference_with_key(reference, key, dialect))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Assigns each reference a unique citation key, deterministically.
///
/// Not a `HashMap` iteration: the input order fixes the output, so the same
/// bibliography always produces the same keys, and adding a reference at the
/// end never renumbers the ones before it.
pub fn disambiguate_keys(references: &[Reference]) -> Vec<String> {
    let mut counts: Vec<(String, usize)> = Vec::new();
    let mut keys = Vec::with_capacity(references.len());

    for reference in references {
        let base = reference.suggested_key().as_str().to_string();
        let seen = match counts.iter_mut().find(|(key, _)| *key == base) {
            Some((_, count)) => {
                *count += 1;
                *count
            }
            None => {
                counts.push((base.clone(), 0));
                0
            }
        };

        keys.push(if seen == 0 {
            base
        } else {
            // 1 -> 'b', 2 -> 'c', ... The first occurrence keeps the bare key,
            // matching what a human does when they discover a clash.
            let suffix = char::from(b'a' + (seen % 26) as u8);
            format!("{base}{suffix}")
        });
    }

    keys
}

fn emit_reference_with_key(reference: &Reference, key: &str, dialect: Dialect) -> String {
    let entry_type = match dialect {
        Dialect::Bibtex => reference.kind().bibtex_type(),
        Dialect::Biblatex => reference.kind().biblatex_type(),
    };

    let mut fields: Vec<(&str, String)> = Vec::new();

    if !reference.authors().is_empty() {
        fields.push(("author", emit_author_list(reference.authors())));
    }
    if !reference.editors().is_empty() {
        fields.push(("editor", emit_author_list(reference.editors())));
    }
    if let Some(title) = reference.title() {
        fields.push(("title", escape(title)));
    }

    // The container's field name depends on what kind of container it is --
    // which is exactly the information `EntryKind` carries and a single
    // `container` string does not. This is where the one-field model is
    // resolved back into BibTeX's three.
    if let Some(container) = reference.container() {
        let field = match (reference.kind(), dialect) {
            (crate::EntryKind::InProceedings | crate::EntryKind::InCollection, _) => "booktitle",
            (_, Dialect::Biblatex) => "journaltitle",
            (_, Dialect::Bibtex) => "journal",
        };
        fields.push((field, escape(container)));
    }

    if let Some(institution) = reference.institution() {
        // The issuing body's field name depends on the entry type, and BibTeX
        // insists on the distinction — a `@book` with an `institution` field is
        // not a valid entry, and most styles will simply not print it, silently
        // dropping the publisher from the bibliography.
        //
        //   @phdthesis / @thesis  -> school
        //   @techreport / @report -> institution
        //   everything else       -> publisher
        //
        // Caught by dogfooding the emitter against a real reference list: several
        // university-published works came out as
        // `@book{..., institution = {University of California, Berkeley}}`,
        // which would have typeset with no publisher at all.
        let field = match reference.kind() {
            crate::EntryKind::Thesis => "school",
            crate::EntryKind::TechReport => "institution",
            _ => "publisher",
        };
        fields.push((field, escape(institution)));
    }
    if let Some(publisher) = reference.publisher() {
        fields.push(("publisher", escape(publisher)));
    }
    if let Some(edition) = reference.edition() {
        fields.push(("edition", escape(edition)));
    }
    if let Some(volume) = reference.volume() {
        fields.push(("volume", escape(volume)));
    }
    if let Some(issue) = reference.issue() {
        fields.push(("number", escape(issue)));
    }
    if let Some(pages) = reference.pages() {
        fields.push(("pages", pages.to_string()));
    }
    if let Some(year) = reference.year() {
        let field = match dialect {
            Dialect::Bibtex => "year",
            Dialect::Biblatex => "date",
        };
        fields.push((field, year.to_string()));
    }

    let ids = reference.identifiers();
    if let Some(doi) = &ids.doi {
        fields.push(("doi", doi.as_str().to_string()));
    }
    if let Some(arxiv) = &ids.arxiv {
        fields.push(("eprint", arxiv.as_str().to_string()));
        fields.push(("eprinttype", "arxiv".to_string()));
    }
    if let Some(isbn) = &ids.isbn {
        fields.push(("isbn", isbn.as_str().to_string()));
    }
    if let Some(issn) = &ids.issn {
        fields.push(("issn", issn.as_str().to_string()));
    }
    if let Some(url) = &ids.url {
        fields.push(("url", url.as_str().to_string()));
    }

    // The note carries what we could not work out. It travels with the entry,
    // because a `.bib` file is where a human will next look at this reference,
    // and hiding our own uncertainty from them there would defeat the point.
    let mut notes: Vec<String> = Vec::new();
    if let Some(note) = reference.note() {
        notes.push(note.to_string());
    }
    if reference.kind() == crate::EntryKind::Unknown {
        notes.push("entry type not determined from the source".to_string());
    }
    if reference.authors().is_truncated() {
        notes.push("author list truncated in the source (et al.)".to_string());
    }
    if let Some(unparsed) = reference.unparsed() {
        notes.push(format!("unparsed remainder: {unparsed}"));
    }
    if !notes.is_empty() {
        fields.push(("note", escape(&notes.join("; "))));
    }

    // Canonical order. A `const` array, not a hash map.
    fields.sort_by_key(|(name, _)| {
        CANONICAL_FIELD_ORDER
            .iter()
            .position(|f| f == name)
            .unwrap_or(usize::MAX)
    });

    let mut out = format!("@{entry_type}{{{key}");
    for (name, value) in &fields {
        let _ = write!(out, ",\n  {name} = {{{value}}}");
    }
    out.push_str(if fields.is_empty() { "}\n" } else { ",\n}\n" });
    out
}

/// Renders an author list for a BibTeX `author` field.
///
/// **This is where the name-safety rule is enforced.** See the module docs.
pub fn emit_author_list(authors: &AuthorList) -> String {
    let mut names: Vec<String> = authors.authors().iter().map(emit_author).collect();

    // BibTeX's own spelling of `et al.`. Emitting the authors we know and
    // saying "and others" is the truthful rendering of a truncated list; padding
    // it out with invented names, or silently pretending the list is complete,
    // are the two ways to get this wrong.
    if authors.is_truncated() {
        names.push("others".to_string());
    }

    names.join(" and ")
}

fn emit_author(author: &Author) -> String {
    match author {
        // `{{...}}` is BibTeX's own "one unit, do not take it apart" idiom. An
        // organisation put through the name grammar becomes "Institute,
        // European Bioinformatics", which appears in real bibliographies and is
        // exactly as silly as it looks.
        Author::Organization(name) | Author::Literal(name) => format!("{{{{{}}}}}", escape(name)),

        Author::Person(person) => match person.confidence() {
            // We know which part is the family name -- either the source told
            // us, or the shape did. Reordering is safe and useful: it is what
            // lets a style alphabetise `van der Waals` under W.
            NameConfidence::Explicit | NameConfidence::Conventional => {
                escape(&person.to_bibtex_reordered())
            }

            // We do NOT know. Emit it exactly as written and let BibTeX apply
            // its own rule -- which is the same rule we would have applied, so
            // we add no information and, crucially, assert none.
            //
            // Reordering here would turn `Mao Zedong` into `Zedong, Mao`, which
            // BibTeX typesets as "M. Zedong". That is a real person, renamed in
            // a published document, by us.
            NameConfidence::Assumed => escape(person.as_written()),
        },
    }
}

/// Escapes characters that would otherwise change a `.bib` file's structure or
/// a LaTeX document's meaning.
///
/// # What is escaped, and what deliberately is not
///
/// Escaped: the ten characters TeX reserves (`# $ % & _ { } ~ ^ \`), because an
/// unescaped `&` in a journal name (*Language & Speech*) is an
/// alignment tab and will not compile, and an unescaped `%` comments out the
/// rest of the line — silently truncating the entry.
///
/// **Not** escaped: accented and non-Latin characters. `Müller` is emitted as
/// `Müller`, not as `M\"{u}ller`. Modern LaTeX (`inputenc`, or XeLaTeX/LuaLaTeX,
/// or Typst) reads UTF-8 directly, and mangling a person's name into escape
/// sequences to satisfy a 1980s toolchain is both unnecessary and the kind of
/// disrespect this crate is written to avoid.
///
/// Braces that are already balanced in the input are left alone, because they
/// are almost certainly deliberate brace-protection (`{DNA}`) rather than an
/// accident — escaping them would break exactly the case-protection the author
/// asked for.
fn escape(text: &str) -> String {
    if braces_are_balanced(text) {
        // Trust the author's braces; escape everything else.
        return text
            .chars()
            .map(|c| match c {
                '#' => "\\#".to_string(),
                '$' => "\\$".to_string(),
                '%' => "\\%".to_string(),
                '&' => "\\&".to_string(),
                '_' => "\\_".to_string(),
                '~' => "\\textasciitilde{}".to_string(),
                '^' => "\\textasciicircum{}".to_string(),
                other => other.to_string(),
            })
            .collect();
    }

    // Unbalanced braces would corrupt the file's structure. Escape them.
    text.chars()
        .map(|c| match c {
            '{' => "\\{".to_string(),
            '}' => "\\}".to_string(),
            '#' => "\\#".to_string(),
            '$' => "\\$".to_string(),
            '%' => "\\%".to_string(),
            '&' => "\\&".to_string(),
            '_' => "\\_".to_string(),
            '~' => "\\textasciitilde{}".to_string(),
            '^' => "\\textasciicircum{}".to_string(),
            other => other.to_string(),
        })
        .collect()
}

fn braces_are_balanced(text: &str) -> bool {
    let mut depth = 0i32;
    for c in text.chars() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth < 0 {
                    return false;
                }
            }
            _ => {}
        }
    }
    depth == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::author::parse_printed_name_list;
    use crate::bibtex::parse_database;
    use crate::provenance::{DocumentId, Provenance};
    use crate::reference::{EntryKind, Page, PageRange, Year};

    fn provenance() -> Provenance {
        let doc = DocumentId::new("paper.pdf").unwrap();
        Provenance::from_page(&doc, 14, "a reference line").unwrap()
    }

    // -- the name-safety rule, end to end -------------------------------

    #[test]
    fn a_known_family_name_is_reordered_for_bibtex() {
        let authors = parse_printed_name_list("J. D. van der Waals");
        assert_eq!(emit_author_list(&authors), "van der Waals, J. D.");
    }

    #[test]
    fn an_assumed_name_is_emitted_exactly_as_written_and_never_reordered() {
        // THE test. Reordering here would put "Zedong, Mao" in a .bib file,
        // which BibTeX typesets as "M. Zedong" -- a real person, renamed in a
        // published document, by us.
        let authors = parse_printed_name_list("Mao Zedong");
        assert_eq!(emit_author_list(&authors), "Mao Zedong");
        assert_ne!(emit_author_list(&authors), "Zedong, Mao");

        let authors = parse_printed_name_list("Kim Jong-un");
        assert_eq!(emit_author_list(&authors), "Kim Jong-un");
    }

    #[test]
    fn a_cjk_name_is_double_braced_so_latex_cannot_split_it_either() {
        let authors = parse_printed_name_list("\u{6BDB}\u{6CFD}\u{4E1C}");
        assert_eq!(emit_author_list(&authors), "{{\u{6BDB}\u{6CFD}\u{4E1C}}}");
    }

    #[test]
    fn an_organisation_is_double_braced() {
        let authors = parse_printed_name_list("European Bioinformatics Institute");
        assert_eq!(
            emit_author_list(&authors),
            "{{European Bioinformatics Institute}}"
        );
    }

    #[test]
    fn a_truncated_list_says_others_and_does_not_invent_the_missing_authors() {
        let authors = parse_printed_name_list("K. R. Fulton et al.");
        assert_eq!(emit_author_list(&authors), "Fulton, K. R. and others");
    }

    // -- escaping --------------------------------------------------------

    #[test]
    fn tex_reserved_characters_are_escaped() {
        // An unescaped `&` in "Language & Speech" is an alignment tab and will
        // not compile. An unescaped `%` comments out the rest of the line,
        // silently truncating the entry.
        assert_eq!(escape("Language & Speech"), "Language \\& Speech");
        assert_eq!(escape("50% complete"), "50\\% complete");
        assert_eq!(escape("a_b"), "a\\_b");
        assert_eq!(escape("C#"), "C\\#");
    }

    #[test]
    fn deliberate_brace_protection_is_left_alone() {
        // Escaping `{DNA}` would destroy exactly the case-protection the author
        // asked for.
        assert_eq!(escape("The {DNA} of {HTML}"), "The {DNA} of {HTML}");
    }

    #[test]
    fn unbalanced_braces_are_escaped_because_they_would_corrupt_the_file() {
        assert_eq!(escape("a { b"), "a \\{ b");
    }

    #[test]
    fn accented_names_are_left_as_utf8_not_mangled_into_escape_sequences() {
        // Modern LaTeX reads UTF-8. Turning `Müller` into `M\"{u}ller` to
        // satisfy a 1980s toolchain is unnecessary and disrespectful.
        assert_eq!(escape("M\u{fc}ller"), "M\u{fc}ller");
    }

    // -- emission --------------------------------------------------------

    #[test]
    fn emits_a_reference_in_canonical_field_order() {
        let reference = Reference::builder(provenance())
            .kind(EntryKind::Article)
            .authors(parse_printed_name_list("M. R. Chen, S. Novak, and J. P. Alvarez"))
            .title("An open-source toolkit for multilingual text alignment")
            .container("International Journal of Computational Linguistics and Text Processing")
            .volume("6")
            .issue("4")
            .pages(PageRange::new(Page::new("281").unwrap(), Page::new("301")))
            .year(Year::new(2024).unwrap())
            .build();

        let bib = emit_reference(&reference, Dialect::Bibtex);
        assert_eq!(
            bib,
            "@article{chen2024open,\n  \
             author = {Chen, M. R. and Novak, S. and Alvarez, J. P.},\n  \
             title = {An open-source toolkit for multilingual text alignment},\n  \
             journal = {International Journal of Computational Linguistics and Text Processing},\n  \
             volume = {6},\n  \
             number = {4},\n  \
             pages = {281--301},\n  \
             year = {2024},\n}\n"
        );
    }

    #[test]
    fn emission_is_byte_identical_across_many_runs() {
        // Determinism, asserted rather than intended. A .bib file whose field
        // order shuffles between runs produces a diff on every save.
        let reference = Reference::builder(provenance())
            .kind(EntryKind::Article)
            .authors(parse_printed_name_list("M. R. Chen, S. Novak"))
            .title("MTAT")
            .container("IJCLTP")
            .volume("6")
            .year(Year::new(2024).unwrap())
            .build();

        let first = emit_reference(&reference, Dialect::Bibtex);
        for _ in 0..500 {
            assert_eq!(emit_reference(&reference, Dialect::Bibtex), first);
        }
    }

    #[test]
    fn biblatex_uses_its_own_richer_vocabulary() {
        let reference = Reference::builder(provenance())
            .kind(EntryKind::TechReport)
            .institution("European Bioinformatics Institute")
            .title("Parser validation")
            .year(Year::new(2019).unwrap())
            .build();

        let bib = emit_reference(&reference, Dialect::Biblatex);
        assert!(bib.starts_with("@report{"), "got: {bib}");
        assert!(bib.contains("date = {2019}"), "biblatex uses `date`: {bib}");

        let bib = emit_reference(&reference, Dialect::Bibtex);
        assert!(bib.starts_with("@techreport{"), "got: {bib}");
        assert!(bib.contains("year = {2019}"), "bibtex uses `year`: {bib}");
    }

    #[test]
    fn a_conference_papers_container_is_a_booktitle_not_a_journal() {
        let reference = Reference::builder(provenance())
            .kind(EntryKind::InProceedings)
            .title("Benefits and drawbacks of adopting a secure programming language")
            .container("Seventeenth Symposium on Usable Privacy and Security")
            .year(Year::new(2021).unwrap())
            .build();
        let bib = emit_reference(&reference, Dialect::Bibtex);
        assert!(bib.contains("booktitle = {Seventeenth"), "got: {bib}");
        assert!(!bib.contains("journal ="));
    }

    #[test]
    fn the_issuing_bodys_field_name_depends_on_the_entry_type() {
        // A `@book` with an `institution` field is not a valid entry, and most
        // styles simply do not print it -- silently dropping the publisher from
        // the bibliography. Caught by dogfooding the emitter against a real
        // reference list, where several university-published works came out that way.
        let build = |kind: EntryKind| {
            Reference::builder(provenance())
                .kind(kind)
                .institution("University of California, Berkeley")
                .year(Year::new(2024).unwrap())
                .build()
        };

        assert!(
            emit_reference(&build(EntryKind::Thesis), Dialect::Bibtex)
                .contains("school = {University")
        );
        assert!(
            emit_reference(&build(EntryKind::TechReport), Dialect::Bibtex)
                .contains("institution = {University")
        );
        // The one that was wrong.
        let book = emit_reference(&build(EntryKind::Book), Dialect::Bibtex);
        assert!(book.contains("publisher = {University"), "got: {book}");
        assert!(
            !book.contains("institution ="),
            "a @book has no `institution` field: {book}"
        );
    }

    #[test]
    fn what_we_could_not_work_out_travels_with_the_entry_in_its_note() {
        // Hiding our own uncertainty from the .bib file -- the place a human
        // will next look at this reference -- would defeat the point.
        let reference = Reference::builder(provenance())
            .title("Something")
            .unparsed("Tech. Rep., Advanced ...")
            .build();
        let bib = emit_reference(&reference, Dialect::Bibtex);
        assert!(bib.contains("entry type not determined"), "got: {bib}");
        assert!(bib.contains("unparsed remainder: Tech. Rep."), "got: {bib}");
    }

    #[test]
    fn colliding_keys_are_disambiguated_deterministically() {
        let make = |title: &str| {
            Reference::builder(provenance())
                .authors(parse_printed_name_list("M. R. Chen"))
                .title(title)
                .year(Year::new(2024).unwrap())
                .build()
        };
        // Same author, same year, same leading title word: the same base key.
        let references = vec![make("Digital twins one"), make("Digital twins two"), make("Digital twins three")];
        let keys = disambiguate_keys(&references);
        assert_eq!(keys, ["chen2024digital", "chen2024digitalb", "chen2024digitalc"]);

        // ...and it is stable.
        assert_eq!(disambiguate_keys(&references), keys);
    }

    // -- round trip -------------------------------------------------------

    #[test]
    fn parse_emit_parse_is_a_fixed_point_on_a_hand_written_file() {
        let source = r#"
% my notes about these references
@string{jlt = "Journal of Language Technology"}

@article{vega2021,
  author  = {Vega, L. and Park, G. and O'Brien, D. and Kaur, R.},
  title   = {Corpus validation of a parser using treebank data from {UD}},
  journal = jlt,
  volume  = 377,
  pages   = {111144},
  year    = 2021,
}

@preamble{"\newcommand{\noop}[1]{}"}

@misc{mtat, author = {{{Chen, Novak and Alvarez}}}, note = "vol. " # 6 # {, 2024}}
"#;
        let once = parse_database(source).unwrap();
        let emitted = emit_database(&once);
        let twice = parse_database(&emitted).unwrap();

        // The fixed point.
        assert_eq!(once, twice, "parse(emit(parse(s))) must equal parse(s)");
        // And emission is idempotent from there.
        assert_eq!(emit_database(&twice), emitted);
    }

    #[test]
    fn brace_protection_survives_a_round_trip() {
        // The single most important thing to preserve: `{DNA}` tells BibTeX not
        // to case-fold, and a style that lowercases titles prints "dna" without
        // it.
        let source = "@article{k, title = {The {DNA} of {HTML} in a {REST} {API}}}";
        let db = parse_database(source).unwrap();
        let reparsed = parse_database(&emit_database(&db)).unwrap();

        assert_eq!(db, reparsed);
        assert_eq!(
            reparsed
                .entry("k")
                .unwrap()
                .field("title")
                .unwrap()
                .brace_protected_spans(),
            ["DNA", "HTML", "REST", "API"]
        );
    }

    #[test]
    fn string_macros_survive_a_round_trip_unexpanded() {
        let source = r#"@string{jan = "January"} @article{k, month = jan # "~1", year = 2024}"#;
        let db = parse_database(source).unwrap();
        let emitted = emit_database(&db);

        // Still a macro reference, not its expansion. Expanding on save would
        // silently rewrite the user's file.
        assert!(emitted.contains("month = jan # \"~1\""), "got: {emitted}");
        assert_eq!(db, parse_database(&emitted).unwrap());
    }

    #[test]
    fn emitting_a_database_is_byte_identical_across_runs() {
        let db = parse_database(
            "@article{a, z = {1}, a = {2}, m = {3}} @article{b, q = {4}}",
        )
        .unwrap();
        let first = emit_database(&db);
        for _ in 0..500 {
            assert_eq!(emit_database(&db), first);
        }
        // Field order is the SOURCE's, not alphabetised behind the user's back.
        assert!(first.find("z = ").unwrap() < first.find("a = ").unwrap());
    }
}

//! The BibTeX round-trip property: **parse → emit → parse is a fixed point.**
//!
//! ```text
//!     parse(emit(parse(source))) == parse(source)
//! ```
//!
//! # Why this one test is worth twenty hand-written ones
//!
//! A `.bib` file is the *user's data*. If this crate rewrites it on a save — drops
//! a `@string` macro, expands a concatenation, strips the braces that protect
//! `{DNA}` from case-folding, silently deletes a comment — then the second time
//! they run KOPITIAM their bibliography is different, and their paper typesets
//! differently, and they have no idea why.
//!
//! Hand-written tests find the cases you thought of. This finds the cases you did
//! not: it drives a generator over the whole grammar — nested braces, quoted
//! values containing braces containing quotes, macro concatenation, empty
//! entries, duplicated fields, free text between entries, parenthesised
//! delimiters, trailing commas — and asserts the fixed point on every one.
//!
//! # The generator is seeded, and that is deliberate
//!
//! CLAUDE.md demands deterministic behaviour. A `proptest`/`quickcheck` run that
//! finds a failure on one machine and not another, or on Tuesdays and not
//! Wednesdays, is a test that cannot be acted on — and it would also have meant a
//! new dependency to buy something a thirty-line LCG provides.
//!
//! So the generator is a fixed-seed linear congruential generator. The same 600
//! databases are tested on every run, on every machine, forever. When one fails,
//! it fails for everybody, and the seed that produced it is printed.

use kopitiam_bibliography::bibtex::{BibDatabase, emit_database, parse_database};

/// A small, fast, entirely deterministic PRNG (Numerical Recipes' LCG
/// constants). Not cryptographic — it does not need to be. It needs to be
/// *reproducible*, which `rand` explicitly is not across versions.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 16
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len())]
    }

    fn chance(&mut self, one_in: usize) -> bool {
        self.below(one_in) == 0
    }
}

/// Field-value fragments, chosen to cover every construct the format has.
///
/// Each one is here because getting it wrong corrupts a real bibliography:
const VALUE_FRAGMENTS: &[&str] = &[
    "A Simple Title",
    // Brace-protected capitalisation. Strip these and a title-lowercasing style
    // prints "the dna of html".
    "The {DNA} of Formal Syntax",
    "{HTML} and {JSON} in a {REST}",
    "A {study of {HTML}} tags", // nested protection
    // TeX escapes: `\{` is a literal brace and must not count towards nesting.
    r"A \{literal\} brace",
    // Characters that break a .bib file if mis-escaped.
    "Language & Speech",
    "50\\% of the corpus",
    "Vega and Kaur",    // the word `and` inside a value
    "\u{fc}ber alles",  // multi-byte: a byte-wise scanner splits this and panics
    "\u{6BDB}\u{6CFD}\u{4E1C}", // CJK
    "281\u{2013}301",   // an en-dash page range
    "",                 // an empty value is legal
    "a, b, c",          // commas inside a value
];

const ENTRY_TYPES: &[&str] = &[
    "article",
    "ARTICLE",
    "InProceedings",
    "book",
    "phdthesis",
    "techreport",
    "misc",
    "software",
];

const FIELD_NAMES: &[&str] = &[
    "author",
    "title",
    "journal",
    "year",
    "volume",
    "pages",
    "doi",
    "note",
    "month",
    "x-custom-field",
];

const MACRO_NAMES: &[&str] = &["jan", "feb", "jcl", "jlt"];

/// Generates one pseudo-random but entirely reproducible `.bib` source file.
fn generate(rng: &mut Lcg) -> String {
    let mut out = String::new();

    // Some @string macros, so entries below can reference them.
    for _ in 0..rng.below(3) {
        let name = rng.pick(MACRO_NAMES);
        let value = rng.pick(VALUE_FRAGMENTS);
        out.push_str(&format!("@string{{{name} = \"{value}\"}}\n\n"));
    }

    if rng.chance(4) {
        out.push_str("@preamble{\"\\newcommand{\\noop}[1]{}\"}\n\n");
    }
    if rng.chance(5) {
        out.push_str("@comment{a note the user wrote}\n\n");
    }
    // Free text outside an entry. bibtex ignores it; we must preserve it.
    if rng.chance(4) {
        out.push_str("% these three are the ones the reviewer asked for\n\n");
    }

    let entries = 1 + rng.below(4);
    for index in 0..entries {
        let entry_type = rng.pick(ENTRY_TYPES);
        let open = if rng.chance(6) { '(' } else { '{' };
        let close = if open == '(' { ')' } else { '}' };

        out.push_str(&format!("@{entry_type}{open}key{index}"));

        let fields = rng.below(5);
        for _ in 0..fields {
            let name = rng.pick(FIELD_NAMES);
            out.push_str(&format!(",\n  {name} = "));

            // A value is one or more components concatenated with `#`.
            let components = 1 + rng.below(3);
            for component in 0..components {
                if component > 0 {
                    out.push_str(" # ");
                }
                match rng.below(4) {
                    // A braced literal.
                    0 => out.push_str(&format!("{{{}}}", rng.pick(VALUE_FRAGMENTS))),
                    // A quoted literal. Sometimes containing a brace-wrapped
                    // quote, which must NOT terminate it.
                    1 => {
                        if rng.chance(5) {
                            out.push_str("\"a {\"} inside braces\"");
                        } else {
                            out.push_str(&format!("\"{}\"", rng.pick(VALUE_FRAGMENTS)));
                        }
                    }
                    // A bare number.
                    2 => out.push_str(&(1900 + rng.below(200)).to_string()),
                    // A macro reference, left unexpanded.
                    _ => out.push_str(rng.pick(MACRO_NAMES)),
                }
            }
        }

        // A trailing comma is legal, extremely common, and must not change
        // anything.
        if fields > 0 && rng.chance(3) {
            out.push(',');
        }
        out.push_str(&format!("\n{close}\n\n"));
    }

    out
}

#[test]
fn parse_emit_parse_is_a_fixed_point_over_six_hundred_generated_databases() {
    let mut failures = 0;

    for seed in 0..600u64 {
        let mut rng = Lcg(seed.wrapping_mul(2_654_435_761).wrapping_add(1));
        let source = generate(&mut rng);

        let once: BibDatabase = match parse_database(&source) {
            Ok(db) => db,
            Err(error) => {
                panic!("seed {seed}: the generator produced source we cannot parse: {error}\n{source}");
            }
        };

        let emitted = emit_database(&once);

        let twice = match parse_database(&emitted) {
            Ok(db) => db,
            Err(error) => {
                panic!(
                    "seed {seed}: OUR OWN OUTPUT does not re-parse: {error}\n\
                     --- emitted ---\n{emitted}"
                );
            }
        };

        if once != twice {
            failures += 1;
            eprintln!(
                "seed {seed}: NOT A FIXED POINT\n--- source ---\n{source}\n\
                 --- emitted ---\n{emitted}\n--- first parse ---\n{once:#?}\n\
                 --- second parse ---\n{twice:#?}"
            );
        }

        // Emission must also be idempotent from the fixed point onwards: having
        // round-tripped once, emitting again must produce the same bytes.
        assert_eq!(
            emit_database(&twice),
            emitted,
            "seed {seed}: emission is not idempotent"
        );
    }

    assert_eq!(failures, 0, "{failures} of 600 databases failed the fixed-point property");
}

#[test]
fn emission_is_byte_identical_across_runs_for_every_generated_database() {
    // Determinism, asserted over the whole generated corpus rather than one
    // hand-picked case. A .bib file whose field order shuffles between runs
    // produces a spurious diff on every save.
    for seed in 0..200u64 {
        let mut rng = Lcg(seed.wrapping_mul(2_654_435_761).wrapping_add(1));
        let source = generate(&mut rng);
        let db = parse_database(&source).expect("must parse");

        let first = emit_database(&db);
        for _ in 0..5 {
            assert_eq!(emit_database(&db), first, "seed {seed}: emission is not deterministic");
        }
    }
}

#[test]
fn brace_protection_survives_every_round_trip_in_the_corpus() {
    // Called out separately because it is the single most damaging thing to lose:
    // `{DNA}` tells BibTeX not to case-fold, and a style that lowercases titles
    // prints "dna" without it. A user would not notice until the paper was
    // submitted.
    for seed in 0..600u64 {
        let mut rng = Lcg(seed.wrapping_mul(2_654_435_761).wrapping_add(1));
        let source = generate(&mut rng);

        let db = parse_database(&source).expect("must parse");
        let reparsed = parse_database(&emit_database(&db)).expect("must re-parse");

        for (before, after) in db.entries().zip(reparsed.entries()) {
            for ((name, value), (_, reparsed_value)) in before.fields.iter().zip(&after.fields) {
                assert_eq!(
                    value.brace_protected_spans(),
                    reparsed_value.brace_protected_spans(),
                    "seed {seed}: brace protection lost in field `{name}`"
                );
            }
        }
    }
}

#[test]
fn string_macros_are_never_expanded_by_a_round_trip() {
    // Expanding a macro on save silently rewrites the user's file. The value is
    // the same; the FILE is not, and it is theirs.
    let source = r#"
@string{jcl = "Journal of Computational Linguistics"}
@article{zhang2021, journal = jcl, month = jan # "~1", year = 2021}
"#;
    let db = parse_database(source).unwrap();
    let emitted = emit_database(&db);

    assert!(emitted.contains("journal = jcl"), "the macro must survive: {emitted}");
    assert!(emitted.contains("month = jan # \"~1\""), "so must the concatenation: {emitted}");
    assert_eq!(db, parse_database(&emitted).unwrap());
}

#[test]
fn a_users_comments_are_not_deleted_by_a_round_trip() {
    // A tool that silently deleted your comments the first time you saved through
    // it would be a tool you never used again.
    let source = "% the reviewer asked for these\n@article{k, title = {T}}\n% and not this one\n";
    let db = parse_database(source).unwrap();
    let emitted = emit_database(&db);

    assert!(emitted.contains("% the reviewer asked for these"), "{emitted}");
    assert!(emitted.contains("% and not this one"), "{emitted}");
    assert_eq!(db, parse_database(&emitted).unwrap());
}

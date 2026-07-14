//! The BibTeX parser.
//!
//! A hand-written recursive scanner rather than a parser-generator, for three
//! reasons: BibTeX's grammar is small, it is *irregular* in ways a grammar file
//! expresses badly (the "a `"` inside braces does not close a quoted value"
//! rule, in particular), and adding a parser-generator dependency to buy 200
//! lines is a poor trade against CLAUDE.md's "avoid unnecessary dependencies".
//!
//! # Error philosophy
//!
//! The parser is **total on well-formed input and honest on the rest**. It does
//! not "recover" from a malformed entry by guessing what the author meant —
//! guessing produces a `.bib` file that silently differs from the one on disk,
//! which is the single most damaging thing this module could do.
//!
//! But it also does not abort a whole file over one stray `@` in a comment.
//! Text that is not a valid entry is captured as [`Item::Ignored`] and
//! re-emitted verbatim, which is precisely what `bibtex` itself does (it skips
//! to the next `@`) — with the difference that we *keep* the skipped text
//! instead of discarding it.
//!
//! A genuinely broken entry — one that *starts* validly and then runs off the
//! end of the file with an unclosed brace — is a [`BibtexError`], because
//! continuing past it would mean inventing structure that is not there.

use super::{BibDatabase, BibEntry, Component, Item, Value};

/// A `.bib` file could not be parsed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BibtexError {
    /// A brace group was opened and never closed.
    #[error("unclosed brace opened at byte {position} (the entry runs off the end of the file)")]
    UnclosedBrace {
        /// Byte offset of the offending `{`.
        position: usize,
    },

    /// A quoted value was opened and never closed.
    #[error("unclosed quotation opened at byte {position}")]
    UnclosedQuote {
        /// Byte offset of the offending `"`.
        position: usize,
    },

    /// An entry began but its body never closed.
    #[error("entry `{key}` (`@{entry_type}`) is not closed")]
    UnclosedEntry {
        /// The entry type.
        entry_type: String,
        /// The key, as far as it was read.
        key: String,
    },

    /// A field name was expected and something else was found.
    #[error("expected a field name in `@{entry_type}{{{key}}}` at byte {position}, found {found:?}")]
    ExpectedFieldName {
        /// The entry type.
        entry_type: String,
        /// The entry key.
        key: String,
        /// Byte offset.
        position: usize,
        /// What was actually there.
        found: String,
    },

    /// A field value was expected after `=`.
    #[error("expected a value after `{field} =` in `@{entry_type}{{{key}}}` at byte {position}")]
    ExpectedValue {
        /// The entry type.
        entry_type: String,
        /// The entry key.
        key: String,
        /// The field whose value is missing.
        field: String,
        /// Byte offset.
        position: usize,
    },
}

/// Parses a `.bib` file.
///
/// # Errors
///
/// [`BibtexError`] only for input that cannot be read without inventing
/// structure — an unclosed brace, an unclosed quote, a field with no value.
/// Text that is merely *not an entry* is preserved as [`Item::Ignored`], not
/// rejected.
///
/// ```
/// use kopitiam_bibliography::bibtex::parse_database;
///
/// let db = parse_database(r#"
///     @string{jlt = "Journal of Language Technology"}
///
///     @article{vega2021,
///       author  = {Vega, L. and Park, G. and O'Brien, D. and Kaur, R.},
///       title   = {Corpus validation of a parser using {UD} treebank data},
///       journal = jlt,
///       year    = 2021,
///     }
/// "#)?;
///
/// let entry = db.entry("vega2021").expect("the entry is there");
/// // The macro is NOT expanded at parse time -- that would rewrite the file.
/// assert_eq!(entry.field("journal").unwrap().as_text(), "jlt");
/// // ...but it can be resolved on demand.
/// assert_eq!(
///     db.expand_value(entry.field("journal").unwrap()),
///     "Journal of Language Technology",
/// );
/// // And the brace-protected acronym survived.
/// assert_eq!(
///     entry.field("title").unwrap().brace_protected_spans(),
///     ["UD"],
/// );
/// # Ok::<(), kopitiam_bibliography::bibtex::BibtexError>(())
/// ```
pub fn parse_database(source: &str) -> Result<BibDatabase, BibtexError> {
    Parser::new(source).parse()
}

struct Parser<'a> {
    src: &'a [u8],
    text: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            src: text.as_bytes(),
            text,
            pos: 0,
        }
    }

    fn parse(mut self) -> Result<BibDatabase, BibtexError> {
        let mut items = Vec::new();
        let mut pending = String::new();

        while self.pos < self.src.len() {
            if self.src[self.pos] != b'@' {
                pending.push(self.current_char());
                self.advance_char();
                continue;
            }

            // At an '@'. Try to read an item from here.
            let start = self.pos;
            match self.parse_item() {
                Ok(Some(item)) => {
                    flush(&mut pending, &mut items);
                    items.push(item);
                }
                Ok(None) => {
                    // Not an item after all (e.g. a bare `@` in prose). Treat
                    // the '@' as ordinary text and carry on -- exactly what
                    // bibtex(1) does, except that we keep the text.
                    self.pos = start;
                    pending.push('@');
                    self.advance_char();
                }
                Err(error) => return Err(error),
            }
        }

        flush(&mut pending, &mut items);
        Ok(BibDatabase::new(items))
    }

    /// Reads one `@...` item. `Ok(None)` means "this `@` does not begin an
    /// item" — recoverable, and the caller treats the `@` as text.
    fn parse_item(&mut self) -> Result<Option<Item>, BibtexError> {
        debug_assert_eq!(self.src[self.pos], b'@');
        self.pos += 1;
        self.skip_whitespace();

        let entry_type = self.read_identifier();
        if entry_type.is_empty() {
            return Ok(None);
        }

        self.skip_whitespace();
        let Some(open) = self.peek() else {
            return Ok(None);
        };
        if open != b'{' && open != b'(' {
            return Ok(None);
        }
        let close = if open == b'{' { b'}' } else { b')' };
        self.pos += 1;

        match entry_type.to_ascii_lowercase().as_str() {
            "comment" => {
                // `@comment{...}` -- content preserved verbatim, braces balanced.
                let body = self.read_until_matching(close)?;
                Ok(Some(Item::Comment(body)))
            }
            "preamble" => {
                let value = self.read_value("preamble", "", "preamble")?;
                self.skip_whitespace();
                self.expect_close(close, "preamble", "")?;
                Ok(Some(Item::Preamble(value)))
            }
            "string" => {
                self.skip_whitespace();
                let name = self.read_identifier();
                self.skip_whitespace();
                if self.peek() != Some(b'=') {
                    return Ok(None);
                }
                self.pos += 1;
                let value = self.read_value("string", &name, &name)?;
                self.skip_whitespace();
                // A trailing comma is legal here too.
                if self.peek() == Some(b',') {
                    self.pos += 1;
                    self.skip_whitespace();
                }
                self.expect_close(close, "string", &name)?;
                Ok(Some(Item::String { name, value }))
            }
            _ => self.parse_entry(entry_type, close).map(Some),
        }
    }

    fn parse_entry(&mut self, entry_type: String, close: u8) -> Result<Item, BibtexError> {
        self.skip_whitespace();

        // The citation key runs to the first comma (or to the closing brace,
        // for a key-only entry).
        let key_start = self.pos;
        while let Some(c) = self.peek() {
            if c == b',' || c == close {
                break;
            }
            self.advance_char();
        }
        let key = self.text[key_start..self.pos].trim().to_string();

        let mut fields = Vec::new();

        loop {
            self.skip_whitespace();
            match self.peek() {
                None => {
                    return Err(BibtexError::UnclosedEntry { entry_type, key });
                }
                Some(c) if c == close => {
                    self.pos += 1;
                    break;
                }
                Some(b',') => {
                    self.pos += 1;
                    continue;
                }
                Some(_) => {}
            }

            let name_start = self.pos;
            let name = self.read_identifier();
            if name.is_empty() {
                return Err(BibtexError::ExpectedFieldName {
                    entry_type,
                    key,
                    position: name_start,
                    found: self.snippet(name_start),
                });
            }

            self.skip_whitespace();
            if self.peek() != Some(b'=') {
                return Err(BibtexError::ExpectedFieldName {
                    entry_type,
                    key,
                    position: self.pos,
                    found: self.snippet(self.pos),
                });
            }
            self.pos += 1;

            let value = self.read_value(&entry_type, &key, &name)?;
            fields.push((name, value));
        }

        Ok(Item::Entry(BibEntry {
            entry_type,
            key,
            fields,
        }))
    }

    /// Reads a field value: components separated by `#`.
    fn read_value(
        &mut self,
        entry_type: &str,
        key: &str,
        field: &str,
    ) -> Result<Value, BibtexError> {
        let mut components = Vec::new();

        loop {
            self.skip_whitespace();
            let start = self.pos;
            let Some(c) = self.peek() else {
                return Err(BibtexError::ExpectedValue {
                    entry_type: entry_type.to_string(),
                    key: key.to_string(),
                    field: field.to_string(),
                    position: start,
                });
            };

            match c {
                b'{' => {
                    self.pos += 1;
                    let body = self.read_until_matching(b'}')?;
                    components.push(Component::Braced(body));
                }
                b'"' => {
                    self.pos += 1;
                    let body = self.read_quoted(start)?;
                    components.push(Component::Quoted(body));
                }
                c if c.is_ascii_digit() => {
                    let digits = self.read_while(|c| c.is_ascii_digit());
                    components.push(Component::Number(digits));
                }
                _ => {
                    let name = self.read_identifier();
                    if name.is_empty() {
                        return Err(BibtexError::ExpectedValue {
                            entry_type: entry_type.to_string(),
                            key: key.to_string(),
                            field: field.to_string(),
                            position: start,
                        });
                    }
                    components.push(Component::Macro(name));
                }
            }

            self.skip_whitespace();
            if self.peek() == Some(b'#') {
                self.pos += 1;
                continue;
            }
            break;
        }

        Ok(Value::new(components))
    }

    /// Reads to the brace/paren that matches one already consumed, tracking
    /// nesting. Returns the **inner content, braces intact**.
    ///
    /// The nesting is the whole reason this is not `find('}')`:
    /// `{The {DNA} of it}` must come back as `The {DNA} of it`, not as
    /// `The {DNA`.
    fn read_until_matching(&mut self, close: u8) -> Result<String, BibtexError> {
        let open = if close == b'}' { b'{' } else { b'(' };
        let start = self.pos;
        let mut depth = 1usize;

        while let Some(c) = self.peek() {
            match c {
                b'\\' => {
                    // A TeX escape: `\{` and `\}` are literal braces and must
                    // not count towards nesting. Skip the backslash AND the
                    // character it escapes.
                    self.pos += 1;
                    if self.pos < self.src.len() {
                        self.advance_char();
                    }
                    continue;
                }
                c if c == open => depth += 1,
                c if c == close => {
                    depth -= 1;
                    if depth == 0 {
                        let body = self.text[start..self.pos].to_string();
                        self.pos += 1;
                        return Ok(body);
                    }
                }
                _ => {}
            }
            self.advance_char();
        }

        Err(BibtexError::UnclosedBrace {
            position: start.saturating_sub(1),
        })
    }

    /// Reads a `"`-delimited value.
    ///
    /// # The rule everyone gets wrong
    ///
    /// A `"` inside a **brace group** does not close the value:
    ///
    /// ```text
    ///     title = "The {"} character"
    /// ```
    ///
    /// is one value containing a literal quotation mark. A parser that scans for
    /// the next `"` truncates it, and then the rest of the entry is parsed as
    /// garbage — usually *without* an error, because what follows still looks
    /// vaguely like fields. So brace depth is tracked here too.
    fn read_quoted(&mut self, open_pos: usize) -> Result<String, BibtexError> {
        let start = self.pos;
        let mut depth = 0usize;

        while let Some(c) = self.peek() {
            match c {
                b'\\' => {
                    self.pos += 1;
                    if self.pos < self.src.len() {
                        self.advance_char();
                    }
                    continue;
                }
                b'{' => depth += 1,
                b'}' => depth = depth.saturating_sub(1),
                b'"' if depth == 0 => {
                    let body = self.text[start..self.pos].to_string();
                    self.pos += 1;
                    return Ok(body);
                }
                _ => {}
            }
            self.advance_char();
        }

        Err(BibtexError::UnclosedQuote {
            position: open_pos,
        })
    }

    fn expect_close(&mut self, close: u8, entry_type: &str, key: &str) -> Result<(), BibtexError> {
        if self.peek() == Some(close) {
            self.pos += 1;
            Ok(())
        } else {
            Err(BibtexError::UnclosedEntry {
                entry_type: entry_type.to_string(),
                key: key.to_string(),
            })
        }
    }

    // -- primitives -----------------------------------------------------

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn current_char(&self) -> char {
        self.text[self.pos..].chars().next().unwrap_or('\0')
    }

    /// Advances by one **character**, not one byte, so that a multi-byte UTF-8
    /// sequence (an accented name, a CJK name, an en-dash in a page range) is
    /// never split down the middle — which would panic on the next slice.
    fn advance_char(&mut self) {
        let len = self.text[self.pos..]
            .chars()
            .next()
            .map_or(1, char::len_utf8);
        self.pos += len;
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_ascii_whitespace() {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn read_while(&mut self, predicate: impl Fn(u8) -> bool) -> String {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if predicate(c) {
                self.pos += 1;
            } else {
                break;
            }
        }
        self.text[start..self.pos].to_string()
    }

    /// Reads an entry type, field name, or macro name.
    ///
    /// BibTeX is permissive here: anything that is not whitespace, a delimiter,
    /// or an operator. `journal-title`, `x_custom` and `IEEEauthorrefmark` all
    /// occur in the wild.
    fn read_identifier(&mut self) -> String {
        self.read_while(|c| {
            !c.is_ascii_whitespace()
                && !matches!(
                    c,
                    b'{' | b'}' | b'(' | b')' | b',' | b'=' | b'#' | b'"' | b'@' | b'%'
                )
        })
    }

    fn snippet(&self, at: usize) -> String {
        self.text[at..]
            .chars()
            .take(16)
            .collect::<String>()
            .replace('\n', "\\n")
    }
}

/// Pushes accumulated free text as an [`Item::Ignored`], if it is not purely
/// whitespace.
///
/// **Whitespace-only runs are dropped**, and that is what makes the round trip a
/// fixed point rather than a slow leak: the emitter puts a blank line between
/// items, and if the parser turned that blank line back into an `Ignored("\n\n")`
/// item, every save would grow the file.
fn flush(pending: &mut String, items: &mut Vec<Item>) {
    let text = pending.trim();
    if !text.is_empty() {
        items.push(Item::Ignored(text.to_string()));
    }
    pending.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_plain_entry() {
        let db = parse_database("@article{key, title = {A Title}, year = 2024}").unwrap();
        let entry = db.entry("key").unwrap();
        assert_eq!(entry.entry_type, "article");
        assert_eq!(entry.fields.len(), 2);
        assert_eq!(entry.field("title").unwrap().as_text(), "A Title");
        assert_eq!(
            entry.field("year").unwrap().components(),
            [Component::Number("2024".to_string())]
        );
    }

    #[test]
    fn field_names_and_entry_types_are_case_insensitive() {
        let db = parse_database("@ARTICLE{k, TITLE = {T}}").unwrap();
        let entry = db.entry("k").unwrap();
        assert_eq!(entry.kind(), crate::EntryKind::Article);
        assert_eq!(entry.field("title").unwrap().as_text(), "T");
        // ...but the source's own casing is preserved for the round trip.
        assert_eq!(entry.entry_type, "ARTICLE");
    }

    #[test]
    fn nested_braces_survive_and_that_is_the_whole_point() {
        let db = parse_database("@article{k, title = {The {DNA} of {HTML} Markup}}").unwrap();
        let title = db.entry("k").unwrap().field("title").unwrap();
        assert_eq!(title.as_text(), "The {DNA} of {HTML} Markup");
        assert_eq!(title.brace_protected_spans(), ["DNA", "HTML"]);
    }

    #[test]
    fn a_quote_inside_braces_does_not_close_a_quoted_value() {
        // The rule everyone gets wrong. A parser that scans for the next `"`
        // truncates this and then parses the rest of the entry as garbage --
        // usually WITHOUT an error, which is the dangerous part.
        let db = parse_database(r#"@article{k, title = "The {"} character", year = 2024}"#).unwrap();
        let entry = db.entry("k").unwrap();
        assert_eq!(entry.field("title").unwrap().as_text(), r#"The {"} character"#);
        assert_eq!(entry.field("year").unwrap().as_text(), "2024");
    }

    #[test]
    fn escaped_braces_do_not_count_towards_nesting() {
        let db = parse_database(r"@article{k, title = {A \{literal\} brace}}").unwrap();
        assert_eq!(
            db.entry("k").unwrap().field("title").unwrap().as_text(),
            r"A \{literal\} brace"
        );
    }

    #[test]
    fn parses_string_macros_and_concatenation() {
        let db = parse_database(
            r#"
            @string{jlt = "Journal of Language Technology"}
            @string{vol = "vol. "}
            @article{k, journal = jlt, note = vol # 377 # {, 2021}}
            "#,
        )
        .unwrap();

        assert_eq!(db.macros().len(), 2);
        let entry = db.entry("k").unwrap();

        // Not expanded at parse time.
        assert_eq!(entry.field("journal").unwrap().as_text(), "jlt");
        // Expanded on demand.
        assert_eq!(
            db.expand_value(entry.field("journal").unwrap()),
            "Journal of Language Technology"
        );
        // Concatenation of three components.
        assert_eq!(entry.field("note").unwrap().components().len(), 3);
        assert_eq!(db.expand_value(entry.field("note").unwrap()), "vol. 377, 2021");
    }

    #[test]
    fn an_unknown_macro_expands_to_itself_rather_than_vanishing() {
        // bibtex(1) warns and carries on. A silently-vanished field is a much
        // harder bug to notice than a field containing the word `nosuch`.
        let db = parse_database("@article{k, journal = nosuch}").unwrap();
        assert_eq!(
            db.expand_value(db.entry("k").unwrap().field("journal").unwrap()),
            "nosuch"
        );
    }

    #[test]
    fn a_macro_cycle_terminates_instead_of_hanging() {
        let db = parse_database("@string{a = b} @string{b = a} @article{k, x = a}").unwrap();
        // We do not care what it says; we care that it RETURNS.
        let _ = db.expand_value(db.entry("k").unwrap().field("x").unwrap());
    }

    #[test]
    fn parses_preamble_and_comment() {
        let db =
            parse_database(r#"@preamble{"\newcommand{\x}{y}"} @comment{a note} @article{k}"#)
                .unwrap();
        assert!(matches!(db.items()[0], Item::Preamble(_)));
        assert_eq!(db.items()[1], Item::Comment("a note".to_string()));
        assert!(matches!(db.items()[2], Item::Entry(_)));
    }

    #[test]
    fn parenthesised_entries_are_legal_bibtex() {
        let db = parse_database("@article(k, title = {T})").unwrap();
        assert_eq!(db.entry("k").unwrap().field("title").unwrap().as_text(), "T");
    }

    #[test]
    fn a_trailing_comma_is_legal_and_extremely_common() {
        let db = parse_database("@article{k, title = {T}, year = 2024,}").unwrap();
        assert_eq!(db.entry("k").unwrap().fields.len(), 2);
    }

    #[test]
    fn free_text_between_entries_is_preserved_not_deleted() {
        // A tool that silently deleted your comments the first time you saved
        // through it would be a tool you never used again.
        let db = parse_database("% my notes\n@article{k, title={T}}\n% more notes\n").unwrap();
        assert_eq!(db.items()[0], Item::Ignored("% my notes".to_string()));
        assert!(matches!(db.items()[1], Item::Entry(_)));
        assert_eq!(db.items()[2], Item::Ignored("% more notes".to_string()));
    }

    #[test]
    fn a_stray_at_sign_in_prose_does_not_abort_the_file() {
        let db = parse_database("email me @ home\n@article{k, title={T}}").unwrap();
        assert_eq!(db.entries().count(), 1);
        assert_eq!(db.items()[0], Item::Ignored("email me @ home".to_string()));
    }

    #[test]
    fn a_duplicate_field_keeps_both_and_reads_the_first_like_bibtex_does() {
        let db = parse_database("@article{k, year = 2024, year = 2025}").unwrap();
        let entry = db.entry("k").unwrap();
        assert_eq!(entry.fields.len(), 2, "both are kept for the round trip");
        assert_eq!(entry.field("year").unwrap().as_text(), "2024", "the FIRST wins");
    }

    #[test]
    fn utf8_names_are_not_split_down_the_middle() {
        // A byte-wise scanner panics here. An accented name, a CJK name, and an
        // en-dash in a page range are all multi-byte.
        let db = parse_database(
            "@article{k, author = {M\u{fc}ller, Hans and \u{6BDB}\u{6CFD}\u{4E1C}}, pages = {281\u{2013}301}}",
        )
        .unwrap();
        let entry = db.entry("k").unwrap();
        assert_eq!(
            entry.field("author").unwrap().as_text(),
            "M\u{fc}ller, Hans and \u{6BDB}\u{6CFD}\u{4E1C}"
        );
        assert_eq!(entry.field("pages").unwrap().as_text(), "281\u{2013}301");
    }

    #[test]
    fn an_entry_with_no_fields_is_fine() {
        let db = parse_database("@misc{lonely}").unwrap();
        assert_eq!(db.entry("lonely").unwrap().fields.len(), 0);
    }

    // -- the errors -----------------------------------------------------

    #[test]
    fn an_unclosed_brace_is_an_error_not_a_guess() {
        assert!(matches!(
            parse_database("@article{k, title = {never closed"),
            Err(BibtexError::UnclosedBrace { .. })
        ));
    }

    #[test]
    fn an_unclosed_quote_is_an_error() {
        assert!(matches!(
            parse_database(r#"@article{k, title = "never closed"#),
            Err(BibtexError::UnclosedQuote { .. })
        ));
    }

    #[test]
    fn an_unclosed_entry_is_an_error() {
        assert!(matches!(
            parse_database("@article{k, title = {T}"),
            Err(BibtexError::UnclosedEntry { .. })
        ));
    }

    #[test]
    fn a_field_with_no_value_is_an_error() {
        assert!(matches!(
            parse_database("@article{k, title = }"),
            Err(BibtexError::ExpectedValue { .. })
        ));
    }

    #[test]
    fn an_empty_file_parses_to_an_empty_database() {
        assert_eq!(parse_database("").unwrap(), BibDatabase::default());
        assert_eq!(parse_database("   \n\n  ").unwrap(), BibDatabase::default());
    }
}

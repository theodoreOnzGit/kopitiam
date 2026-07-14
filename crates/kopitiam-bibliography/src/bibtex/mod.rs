//! BibTeX: parse **and** emit, losslessly.
//!
//! BibTeX is the interchange format of the LaTeX world, it is forty years old,
//! and it is fiddlier than it looks. This module implements it properly rather
//! than approximately, because a bibliography tool that mangles a `.bib` file
//! on a round trip is a tool nobody can afford to run twice.
//!
//! # The lossless-round-trip design
//!
//! The central decision: **a parsed `.bib` file is stored as its own syntax,
//! not as a list of [`Reference`](crate::Reference)s.**
//!
//! It would have been tempting to parse straight into `Reference` and emit from
//! that. It would also have been wrong. `Reference` is a *semantic* model — it
//! has an [`Author`](crate::Author) list, a [`Year`](crate::Year), a
//! [`PageRange`](crate::PageRange) — and every one of those conversions throws
//! something away:
//!
//! * `title = {The {DNA} of Formal Syntax}` carries **brace-protected
//!   capitalisation**. Those inner braces tell BibTeX "do not case-fold `DNA`",
//!   and a style that lowercases titles will produce *"The dna of formal syntax"*
//!   without them. They are semantically load-bearing punctuation, and a
//!   `String` title loses them.
//! * `month = jan # "~1"` is a `@string` macro concatenated with a literal.
//!   Expanding it at parse time and re-emitting the expansion silently rewrites
//!   the user's file.
//! * Field order, entry order, and the user's own comments are theirs, not ours.
//!
//! So [`BibDatabase`] holds [`Item`]s of [`Value`]s of [`Component`]s — the
//! literal syntax — and **[`Reference`](crate::Reference) is a *view* derived
//! from it** ([`BibEntry::to_reference`]). Round-tripping goes through the
//! syntax; interpretation goes through the view. Neither contaminates the other,
//! and the round-trip is a **fixed point**:
//!
//! ```text
//!   parse(emit(parse(source))) == parse(source)
//! ```
//!
//! asserted in `tests/roundtrip.rs` against a seeded generator over hundreds of
//! synthetic databases, plus every hand-written case in this module.
//!
//! # What the parser handles, and why each one is here
//!
//! | Construct | Why it is not optional |
//! |---|---|
//! | `@article{key, ...}` | the basic entry |
//! | `@ARTICLE`, `@Article` | entry types are case-insensitive; real files use all three |
//! | `{...}` and `"..."` values | both are legal and both appear |
//! | `@entry(...)` parentheses | legal, and emitted by some tools |
//! | nested braces | `{The {DNA} of it}` — the whole point |
//! | `"a {"} b"` | a quote inside braces does **not** end a quoted value |
//! | `@string{jan = "January"}` | macro definitions |
//! | `jan # " 1"` | concatenation |
//! | `@preamble{"\newcommand..."}` | LaTeX preamble injection |
//! | `@comment{...}` | explicit comments |
//! | free text between entries | BibTeX ignores it; we **preserve** it, because it is the user's |
//! | a trailing comma before `}` | legal, extremely common |
//!
//! # Emission is deterministic
//!
//! CLAUDE.md requires deterministic behaviour, so nothing here iterates a
//! `HashMap`. Fields are held in a `Vec` and emitted in the order they were
//! parsed (for a round-tripped file, so the user's own ordering survives) or in
//! a fixed canonical order (for a `Reference` synthesised by this crate — see
//! [`emit::CANONICAL_FIELD_ORDER`]). Byte-identical output on every run is
//! asserted in the tests, not merely intended.

mod emit;
mod parse;

use std::fmt;

use serde::{Deserialize, Serialize};

pub use emit::{
    CANONICAL_FIELD_ORDER, Dialect, disambiguate_keys, emit_author_list, emit_database,
    emit_reference, emit_references,
};
pub use parse::{BibtexError, parse_database};

/// One piece of a field value.
///
/// A BibTeX field value is a `#`-separated concatenation of these. Keeping the
/// pieces distinct — rather than eagerly concatenating them into a `String` —
/// is what makes the round trip lossless: `jan # " 1"` must come back out as
/// `jan # " 1"`, not as `"January 1"`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Component {
    /// A brace-delimited literal, stored **with its inner braces intact**.
    ///
    /// `{The {DNA} of Formal Syntax}` is stored as
    /// `Braced("The {DNA} of Formal Syntax")`. Those inner braces are not noise;
    /// they instruct BibTeX not to case-fold `DNA`, and stripping them changes
    /// what a bibliography prints. See [`Value::brace_protected_spans`].
    Braced(String),

    /// A quote-delimited literal, stored verbatim.
    Quoted(String),

    /// A bare number: `volume = 6`.
    Number(String),

    /// A bare identifier, i.e. a reference to a `@string` macro: `month = jan`.
    ///
    /// **Not expanded at parse time.** Expanding it would rewrite the user's
    /// file on the next save. Use [`BibDatabase::expand`] when you want the
    /// text.
    Macro(String),
}

impl Component {
    /// The literal text of this component, **without** resolving macros.
    ///
    /// A [`Self::Macro`] yields its *name*, which is almost never what a caller
    /// wants — use [`BibDatabase::expand_value`] instead, which resolves it
    /// against the database's macro table.
    pub fn raw(&self) -> &str {
        match self {
            Self::Braced(s) | Self::Quoted(s) | Self::Number(s) | Self::Macro(s) => s,
        }
    }
}

/// A field's value: one or more [`Component`]s concatenated with `#`.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Value(Vec<Component>);

impl Value {
    /// Builds a value from its components.
    pub fn new(components: Vec<Component>) -> Self {
        Self(components)
    }

    /// A value that is a single braced literal — the form this crate emits.
    pub fn braced(text: impl Into<String>) -> Self {
        Self(vec![Component::Braced(text.into())])
    }

    /// The components.
    pub fn components(&self) -> &[Component] {
        &self.0
    }

    /// The concatenated text with macros left **unresolved** (a macro
    /// contributes its own name).
    ///
    /// Adequate for the overwhelmingly common case of a value with no macros in
    /// it; use [`BibDatabase::expand_value`] when correctness against `@string`
    /// definitions matters.
    pub fn as_text(&self) -> String {
        self.0.iter().map(Component::raw).collect()
    }

    /// The text with **brace-protection markers removed** — the string a human
    /// means when they say "the title".
    ///
    /// `{The {DNA} of Formal Syntax}` yields `The DNA of Formal Syntax`. The
    /// protection is not lost, merely not shown: it is still in the
    /// [`Component`], and [`Self::brace_protected_spans`] reports it.
    pub fn as_plain_text(&self) -> String {
        self.0
            .iter()
            .map(|component| match component {
                Component::Braced(s) => s.replace(['{', '}'], ""),
                other => other.raw().to_string(),
            })
            .collect()
    }

    /// The substrings the author brace-protected against case-folding.
    ///
    /// For `{The {DNA} of {HTTP} Markup}` this is `["DNA", "HTTP"]`.
    ///
    /// # Why this is worth exposing
    ///
    /// These are exactly the tokens the author considered *not ordinary words*:
    /// acronyms, standards names, protocol names, product names. In a
    /// computer-science bibliography that is a remarkably high-value signal —
    /// `{HTML}`, `{JSON}`, `{REST}`, `{API}` — and it is free, because the
    /// author already marked them up for the typesetter. It feeds the knowledge
    /// graph in [`crate::knowledge`].
    pub fn brace_protected_spans(&self) -> Vec<String> {
        let mut spans = Vec::new();
        for component in &self.0 {
            if let Component::Braced(text) = component {
                collect_braced_spans(text, &mut spans);
            }
        }
        spans
    }

    /// Whether the value has no components.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Collects the contents of top-level `{...}` groups inside an already-braced
/// value.
fn collect_braced_spans(text: &str, out: &mut Vec<String>) {
    let mut depth = 0usize;
    let mut current = String::new();
    for c in text.chars() {
        match c {
            '{' => {
                if depth > 0 {
                    current.push('{');
                }
                depth += 1;
            }
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    if !current.is_empty() {
                        out.push(std::mem::take(&mut current));
                    }
                } else {
                    current.push('}');
                }
            }
            other if depth > 0 => current.push(other),
            _ => {}
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.as_plain_text())
    }
}

/// One `@type{key, ...}` entry.
///
/// Fields are a `Vec`, not a map: order is the user's, duplicates are possible
/// in real files (and BibTeX takes the *first*), and a `HashMap` would both
/// destroy the order and silently drop the duplicate. It would also make
/// emission non-deterministic, which CLAUDE.md forbids outright.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BibEntry {
    /// The entry type as written (`article`, `ARTICLE`, `InProceedings`).
    /// Preserved with its original case so a round trip does not re-case the
    /// user's file; compare with [`Self::kind`], which is case-insensitive.
    pub entry_type: String,
    /// The citation key.
    pub key: String,
    /// The fields, in source order.
    pub fields: Vec<(String, Value)>,
}

impl BibEntry {
    /// The first value of the named field, matched case-insensitively (BibTeX
    /// field names are case-insensitive).
    ///
    /// The *first*, because that is what `bibtex` itself does with a duplicated
    /// field, and quietly disagreeing with the tool the file is destined for is
    /// how a bibliography ends up different from the one the author proofread.
    pub fn field(&self, name: &str) -> Option<&Value> {
        self.fields
            .iter()
            .find(|(field, _)| field.eq_ignore_ascii_case(name))
            .map(|(_, value)| value)
    }

    /// The entry's semantic kind.
    pub fn kind(&self) -> crate::EntryKind {
        crate::EntryKind::from_bibtex_type(&self.entry_type)
    }
}

/// A top-level item in a `.bib` file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Item {
    /// A bibliography entry.
    Entry(BibEntry),

    /// A `@string{name = value}` macro definition.
    String {
        /// The macro's name.
        name: String,
        /// Its expansion.
        value: Value,
    },

    /// A `@preamble{...}`: LaTeX injected into the bibliography.
    Preamble(Value),

    /// An explicit `@comment{...}`, content preserved verbatim.
    Comment(String),

    /// Free text outside any entry.
    ///
    /// `bibtex` ignores this. **We keep it**, because it is very often the
    /// user's own notes ("% these three are the ones the reviewer wanted"), and
    /// a tool that silently deleted your comments the first time you saved
    /// through it would be a tool you never used again.
    ///
    /// Stored trimmed of surrounding whitespace, which is what makes the
    /// round-trip a fixed point: see [`emit_database`].
    Ignored(String),
}

/// A parsed `.bib` file: its items, in source order.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BibDatabase {
    items: Vec<Item>,
}

impl BibDatabase {
    /// Builds a database from items.
    pub fn new(items: Vec<Item>) -> Self {
        Self { items }
    }

    /// Every item, in source order.
    pub fn items(&self) -> &[Item] {
        &self.items
    }

    /// Just the bibliography entries.
    pub fn entries(&self) -> impl Iterator<Item = &BibEntry> {
        self.items.iter().filter_map(|item| match item {
            Item::Entry(entry) => Some(entry),
            _ => None,
        })
    }

    /// The entry with the given citation key, if any.
    pub fn entry(&self, key: &str) -> Option<&BibEntry> {
        self.entries().find(|entry| entry.key == key)
    }

    /// The `@string` macro table, in definition order.
    ///
    /// A `Vec` rather than a map, deliberately: BibTeX macros are *scoped by
    /// position* (a `@string` only applies to entries after it), and a map
    /// would flatten that away.
    pub fn macros(&self) -> Vec<(&str, &Value)> {
        self.items
            .iter()
            .filter_map(|item| match item {
                Item::String { name, value } => Some((name.as_str(), value)),
                _ => None,
            })
            .collect()
    }

    /// Resolves a value against this database's `@string` macros, returning the
    /// concatenated text.
    ///
    /// # Unknown macros
    ///
    /// An unresolvable macro name expands to **itself**, which is exactly what
    /// `bibtex` does (it warns and carries on). It is not silently dropped: a
    /// vanished field is a much harder bug to notice than a field containing
    /// the literal word `jan`.
    ///
    /// Macros expanding to other macros are resolved to a depth of 8, after
    /// which the name is used as-is. A `.bib` file with a macro cycle is broken;
    /// hanging on it would be worse than declining to.
    pub fn expand_value(&self, value: &Value) -> String {
        let macros = self.macros();
        value
            .components()
            .iter()
            .map(|component| match component {
                Component::Macro(name) => self.expand_macro(name, &macros, 0),
                Component::Braced(text) => text.replace(['{', '}'], ""),
                other => other.raw().to_string(),
            })
            .collect()
    }

    fn expand_macro(&self, name: &str, macros: &[(&str, &Value)], depth: usize) -> String {
        if depth >= 8 {
            return name.to_string();
        }
        match macros
            .iter()
            .find(|(macro_name, _)| macro_name.eq_ignore_ascii_case(name))
        {
            Some((_, value)) => value
                .components()
                .iter()
                .map(|component| match component {
                    Component::Macro(inner) => self.expand_macro(inner, macros, depth + 1),
                    Component::Braced(text) => text.replace(['{', '}'], ""),
                    other => other.raw().to_string(),
                })
                .collect(),
            None => name.to_string(),
        }
    }

    /// A copy of this database with every `@string` macro expanded in place and
    /// the `@string` definitions removed.
    ///
    /// Destructive, and offered only because some downstream tools cannot handle
    /// macros. It is **not** what a round trip does.
    pub fn expand(&self) -> Self {
        let items = self
            .items
            .iter()
            .filter_map(|item| match item {
                Item::String { .. } => None,
                Item::Entry(entry) => Some(Item::Entry(BibEntry {
                    entry_type: entry.entry_type.clone(),
                    key: entry.key.clone(),
                    fields: entry
                        .fields
                        .iter()
                        .map(|(name, value)| {
                            (name.clone(), Value::braced(self.expand_value(value)))
                        })
                        .collect(),
                })),
                other => Some(other.clone()),
            })
            .collect();
        Self::new(items)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brace_protected_spans_are_recovered() {
        let value = Value::braced("The {DNA} of {HTTP} Markup");
        assert_eq!(value.brace_protected_spans(), vec!["DNA", "HTTP"]);
        // The plain text drops the markers but not the words.
        assert_eq!(value.as_plain_text(), "The DNA of HTTP Markup");
        // ...and the raw text keeps them, which is what round-trips.
        assert_eq!(value.as_text(), "The {DNA} of {HTTP} Markup");
    }

    #[test]
    fn nested_protection_reports_the_outer_span() {
        let value = Value::braced("A {study of {HTML}} tags");
        assert_eq!(value.brace_protected_spans(), vec!["study of {HTML}"]);
        assert_eq!(value.as_plain_text(), "A study of HTML tags");
    }
}

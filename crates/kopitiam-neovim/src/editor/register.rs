//! Registers: named storage for yanked/deleted text.
//!
//! # Why granularity has to live here, not just on the motion
//!
//! `dd` then `p` pastes a whole line *below* the cursor. `dw` then `p`
//! pastes the word *inline, after* the cursor. Both go through the exact
//! same `p` keystroke — the only thing that tells `p` which behaviour to use
//! is what got written into the register that fed it. So
//! [`Granularity`](crate::core::Granularity) is not a property of the
//! *delete* — by the time `p` runs, the deletion is long over — it is a
//! property of the *register's contents*, remembered until the next paste.
//! Dropping it here (e.g. by storing registers as bare `String`s) is exactly
//! the bug the brief calls out as "the classic way a vim clone ends up
//! pasting text in the wrong place."

use std::collections::HashMap;

use crate::core::Granularity;

/// One register's contents: the text, and how `p`/`P` should place it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RegisterContent {
    pub text: String,
    pub granularity: Granularity,
}

/// All of the editor's registers.
///
/// Deliberately scoped to what the brief asks for: the unnamed register
/// (`""`), named `a`-`z` (with `A`-`Z` appending), and the yank register
/// (`"0`). Real vim also has numbered delete-history registers `"1`-`"9`,
/// the small-delete register `"-`, and several read-only registers (`"%`,
/// `"/`, `"+`, ...). Those are a deliberate omission — they add real value
/// but no new *architecture*, and can be layered on by extending
/// [`Registers::write_delete`] later without touching its callers.
#[derive(Debug, Default)]
pub struct Registers {
    unnamed: RegisterContent,
    named: HashMap<char, RegisterContent>,
    /// Register `"0`: mirrors the most recent yank that did *not* specify an
    /// explicit register (vim's actual rule — a `"ayy` does not touch `"0`).
    yank: RegisterContent,
}

impl Registers {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records the result of a delete or change. `selected` is the register
    /// name the user typed after `"`, if any (`'A'`..`'Z'` appends instead
    /// of overwriting). The unnamed register always ends up mirroring
    /// whatever was written, matching vim: `"ax` still lets a bare `p`
    /// paste what `x` deleted.
    pub fn write_delete(&mut self, selected: Option<char>, text: String, granularity: Granularity) {
        let content = self.store_named(selected, text, granularity);
        self.unnamed = content;
    }

    /// Records the result of a yank. Like [`Self::write_delete`], but also
    /// mirrors into `"0` — and only when no explicit register was given,
    /// since `"0` means "the last *plain* yank", not "the last yank into any
    /// register".
    pub fn write_yank(&mut self, selected: Option<char>, text: String, granularity: Granularity) {
        let content = self.store_named(selected, text, granularity);
        if selected.is_none() {
            self.yank = content.clone();
        }
        self.unnamed = content;
    }

    fn store_named(&mut self, selected: Option<char>, text: String, granularity: Granularity) -> RegisterContent {
        let content = RegisterContent { text, granularity };
        match selected {
            None | Some('"') => {}
            Some(c) if c.is_ascii_uppercase() => {
                let key = c.to_ascii_lowercase();
                let entry = self.named.entry(key).or_default();
                entry.text.push_str(&content.text);
                entry.granularity = content.granularity;
                return entry.clone();
            }
            Some(c) => {
                self.named.insert(c, content.clone());
            }
        }
        content
    }

    /// Reads the register `"{name}` refers to, or the unnamed register if
    /// `name` is `None`. Uppercase names read the same underlying register
    /// as their lowercase form (append is a write-time concept only).
    pub fn read(&self, name: Option<char>) -> Option<&RegisterContent> {
        match name {
            None | Some('"') => Some(&self.unnamed),
            Some('0') => Some(&self.yank),
            Some(c) => self.named.get(&c.to_ascii_lowercase()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unnamed_mirrors_a_delete_into_a_named_register() {
        let mut r = Registers::new();
        r.write_delete(Some('a'), "hello".into(), Granularity::Charwise);
        assert_eq!(r.read(Some('a')).unwrap().text, "hello");
        assert_eq!(r.read(None).unwrap().text, "hello");
    }

    #[test]
    fn uppercase_register_appends() {
        let mut r = Registers::new();
        r.write_delete(Some('a'), "foo".into(), Granularity::Charwise);
        r.write_delete(Some('A'), "bar".into(), Granularity::Charwise);
        assert_eq!(r.read(Some('a')).unwrap().text, "foobar");
    }

    #[test]
    fn plain_yank_updates_register_zero_but_explicit_register_does_not() {
        let mut r = Registers::new();
        r.write_yank(None, "plain".into(), Granularity::Charwise);
        assert_eq!(r.read(Some('0')).unwrap().text, "plain");

        r.write_yank(Some('b'), "explicit".into(), Granularity::Charwise);
        // "0 still holds the last *plain* yank.
        assert_eq!(r.read(Some('0')).unwrap().text, "plain");
        assert_eq!(r.read(Some('b')).unwrap().text, "explicit");
    }

    #[test]
    fn granularity_survives_a_round_trip() {
        let mut r = Registers::new();
        r.write_delete(None, "line\n".into(), Granularity::Linewise);
        assert_eq!(r.read(None).unwrap().granularity, Granularity::Linewise);
    }
}

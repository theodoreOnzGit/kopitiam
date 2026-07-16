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
//!
//! # What lives here, and what does not
//!
//! This struct own the registers that are *pure storage*: the unnamed
//! register (`""`), the named `a`-`z` (`A`-`Z` append), the yank register
//! (`"0`), the numbered delete-ring (`"1`-`"9`), and the small-delete
//! register (`"-`). All of those are plain in-memory state with vim's routing
//! rules, and they get unit-tested hard right here.
//!
//! The registers that are *not* pure storage stay out of this struct and are
//! resolved one layer up, in [`crate::editor`]:
//!
//! * the system-clipboard registers `"+`/`"*`, because reading and writing
//!   them is terminal / OS I/O (see [`crate::editor::clipboard`]);
//! * the read-only registers `"%` (filename), `".` (last insert), `":` (last
//!   ex command), `"/` (last search), because they are *computed* from editor
//!   state, not stored.
//!
//! Keeping the storage registers here and the derived/IO registers in `Editor`
//! is what lets this whole module stay a pure, deterministic, side-effect-free
//! unit.
//!
//! # The numbered delete-ring (`"1`-`"9`), the vim rules
//!
//! vim's rules for the numbered registers are fiddly and easy to get subtly
//! wrong, so they are spelled out here at the point they are enforced
//! (`:help registers`):
//!
//! * `"0` holds the text of the most recent **yank** — but only a yank that
//!   did *not* name an explicit register (`"ayy` leaves `"0` untouched).
//! * `"1` holds the text of the most recent **delete or change** — but only
//!   when it deleted **at least one line** (linewise, or a charwise delete
//!   that spanned a newline) *and* named no explicit register. Each such
//!   delete pushes the ring down: `"1`→`"2`→…→`"8`→`"9`, and the old `"9`
//!   falls off the end. This is what makes `"1p"2p"3p…` walk back through
//!   recent line deletes, and dot-`.`-after-`"1p` cycle them.
//! * `"-` (the small-delete register) holds a delete of **less than one line**
//!   that named no explicit register. Crucially this means a small charwise
//!   delete does **not** pollute the numbered ring — without a home for small
//!   deletes, `"1` would be wrong after every `x`.
//! * Whenever the command **named an explicit register** (`"a`, `"+`, a
//!   digit, …), the yank/ring/small-delete machinery is all skipped — the
//!   text went where the user asked. The unnamed register still mirrors it,
//!   because unnamed always points at "the text of the last edit, wherever it
//!   went".

use std::collections::HashMap;

use crate::core::Granularity;

/// One register's contents: the text, and how `p`/`P` should place it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RegisterContent {
    pub text: String,
    pub granularity: Granularity,
}

/// All of the editor's *stored* registers.
///
/// See the module docs for the split between the storage registers modelled
/// here and the clipboard / read-only registers resolved in [`crate::editor`].
#[derive(Debug, Default)]
pub struct Registers {
    unnamed: RegisterContent,
    named: HashMap<char, RegisterContent>,
    /// Register `"0`: mirrors the most recent yank that did *not* specify an
    /// explicit register (vim's actual rule — a `"ayy` does not touch `"0`).
    yank: RegisterContent,
    /// Registers `"1`-`"9`: the delete/change ring. Index 0 is `"1` (most
    /// recent big delete), index 8 is `"9` (oldest). A big unnamed delete
    /// shifts the whole array down by one and writes the new text into index
    /// 0. See the module docs for what counts as "big".
    numbered: [RegisterContent; 9],
    /// Register `"-`: the most recent unnamed delete of *less than one line*.
    /// Keeps small deletes out of the numbered ring.
    small_delete: RegisterContent,
}

impl Registers {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records the result of a delete or change. `selected` is the register
    /// name the user typed after `"`, if any (`'A'`..`'Z'` appends instead of
    /// overwriting). The unnamed register always ends up mirroring whatever
    /// was written, matching vim: `"ax` still lets a bare `p` paste what `x`
    /// deleted.
    ///
    /// The blackhole register `"_` is *not* handled here — a `"_`-selected
    /// delete must touch nothing at all, so [`crate::editor`] drops it before
    /// this is ever called. (Routing it here would still have to special-case
    /// it before the unnamed mirror, so the cut is cleaner at the caller.)
    pub fn write_delete(&mut self, selected: Option<char>, text: String, granularity: Granularity) {
        let content = self.store_named(selected, text, granularity);
        // The numbered ring, small-delete register and (for yanks) "0 are only
        // touched when the command named no explicit register — see the module
        // docs. `Some('"')` is the way to *name* the unnamed register
        // explicitly and vim treats it the same as leaving it off, so it also
        // counts as "unnamed" here.
        if matches!(selected, None | Some('"')) {
            if is_big_delete(&content) {
                self.push_numbered(content.clone());
            } else {
                self.small_delete = content.clone();
            }
        }
        self.unnamed = content;
    }

    /// Records the result of a yank. Like [`Self::write_delete`], but mirrors
    /// into `"0` (not the delete-ring) — and only when no explicit register
    /// was given, since `"0` means "the last *plain* yank", not "the last yank
    /// into any register".
    pub fn write_yank(&mut self, selected: Option<char>, text: String, granularity: Granularity) {
        let content = self.store_named(selected, text, granularity);
        if matches!(selected, None | Some('"')) {
            self.yank = content.clone();
        }
        self.unnamed = content;
    }

    /// Shift the numbered ring down (`"1`→`"2`→…, `"9` dropped) and write
    /// `content` into `"1`.
    fn push_numbered(&mut self, content: RegisterContent) {
        self.numbered.rotate_right(1);
        self.numbered[0] = content;
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
            Some(c) if c.is_ascii_lowercase() => {
                self.named.insert(c, content.clone());
            }
            // An explicit numbered register (`"1p` reads; `"3"dd` writes) lands
            // in that slot directly, without rotating the ring — writing a
            // numbered register by name is a direct poke, not a delete event.
            Some(c @ '1'..='9') => {
                self.numbered[c as usize - '1' as usize] = content.clone();
            }
            Some('0') => {
                self.yank = content.clone();
            }
            Some('-') => {
                self.small_delete = content.clone();
            }
            // Any other explicit register (a clipboard `+`/`*` or a read-only
            // name) is not stored here — the caller in `Editor` has already
            // handled the ones that mean anything. We still return `content`
            // so the unnamed register mirrors it, matching vim.
            Some(_) => {}
        }
        content
    }

    /// Reads the register `"{name}` refers to among the *stored* registers, or
    /// the unnamed register if `name` is `None`. Returns `None` for names this
    /// struct does not own (clipboard `+`/`*`, read-only `%`/`.`/`:`/`/`) —
    /// [`crate::editor`] resolves those before falling back here. Uppercase
    /// names read the same underlying register as their lowercase form (append
    /// is a write-time concept only).
    pub fn read(&self, name: Option<char>) -> Option<&RegisterContent> {
        match name {
            None | Some('"') => Some(&self.unnamed),
            Some('0') => Some(&self.yank),
            Some('-') => Some(&self.small_delete),
            Some(c @ '1'..='9') => Some(&self.numbered[c as usize - '1' as usize]),
            Some(c) if c.is_ascii_alphabetic() => self.named.get(&c.to_ascii_lowercase()),
            Some(_) => None,
        }
    }
}

/// Does this delete belong in the numbered ring (`"1`-`"9`) rather than the
/// small-delete register (`"-`)? vim's rule: a whole line or more. Linewise is
/// always "big"; a charwise delete is big only if it crossed a line boundary
/// (its text contains a newline). See the module docs.
fn is_big_delete(content: &RegisterContent) -> bool {
    matches!(content.granularity, Granularity::Linewise | Granularity::Blockwise) || content.text.contains('\n')
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

    #[test]
    fn register_zero_is_the_last_yank_not_the_last_delete() {
        // The everyday footgun `"0` exists to solve: yank something, delete
        // something else, and `"0p` must still paste the yank.
        let mut r = Registers::new();
        r.write_yank(None, "yanked".into(), Granularity::Linewise);
        r.write_delete(None, "deleted\n".into(), Granularity::Linewise);
        assert_eq!(r.read(Some('0')).unwrap().text, "yanked");
        // ...while the unnamed register followed the more recent delete.
        assert_eq!(r.read(None).unwrap().text, "deleted\n");
    }

    #[test]
    fn a_big_delete_shifts_the_numbered_ring() {
        let mut r = Registers::new();
        // Three successive whole-line deletes. The newest is "1, oldest is "3.
        r.write_delete(None, "first\n".into(), Granularity::Linewise);
        r.write_delete(None, "second\n".into(), Granularity::Linewise);
        r.write_delete(None, "third\n".into(), Granularity::Linewise);
        assert_eq!(r.read(Some('1')).unwrap().text, "third\n");
        assert_eq!(r.read(Some('2')).unwrap().text, "second\n");
        assert_eq!(r.read(Some('3')).unwrap().text, "first\n");
        // And "1 always equals the unnamed register right after a big delete,
        // which is why a plain `p` and `"1p` paste the same thing.
        assert_eq!(r.read(None).unwrap().text, r.read(Some('1')).unwrap().text);
    }

    #[test]
    fn the_ring_falls_off_the_end_at_nine() {
        let mut r = Registers::new();
        for i in 1..=10 {
            r.write_delete(None, format!("d{i}\n"), Granularity::Linewise);
        }
        // "1 is the tenth (newest) delete; "9 is the second; the first fell off.
        assert_eq!(r.read(Some('1')).unwrap().text, "d10\n");
        assert_eq!(r.read(Some('9')).unwrap().text, "d2\n");
    }

    #[test]
    fn a_small_delete_goes_to_the_dash_register_and_spares_the_ring() {
        let mut r = Registers::new();
        r.write_delete(None, "aline\n".into(), Granularity::Linewise); // fills "1
        r.write_delete(None, "x".into(), Granularity::Charwise); // small: -> "-
        assert_eq!(r.read(Some('-')).unwrap().text, "x");
        // The numbered ring was NOT disturbed by the small delete.
        assert_eq!(r.read(Some('1')).unwrap().text, "aline\n");
        // Unnamed still mirrors the most recent (small) delete.
        assert_eq!(r.read(None).unwrap().text, "x");
    }

    #[test]
    fn a_multiline_charwise_delete_counts_as_big() {
        let mut r = Registers::new();
        // Charwise granularity, but the text crossed a line boundary — vim
        // treats that as a whole-line-or-more delete for ring purposes.
        r.write_delete(None, "end\nstart".into(), Granularity::Charwise);
        assert_eq!(r.read(Some('1')).unwrap().text, "end\nstart");
    }

    #[test]
    fn a_named_delete_leaves_the_ring_and_small_register_untouched() {
        let mut r = Registers::new();
        r.write_delete(None, "ring\n".into(), Granularity::Linewise); // "1
        r.write_delete(None, "small".into(), Granularity::Charwise); // "-
        r.write_delete(Some('a'), "named\n".into(), Granularity::Linewise);
        // Naming register a routes the text to a and to unnamed only.
        assert_eq!(r.read(Some('a')).unwrap().text, "named\n");
        assert_eq!(r.read(None).unwrap().text, "named\n");
        // Neither the ring nor "- moved.
        assert_eq!(r.read(Some('1')).unwrap().text, "ring\n");
        assert_eq!(r.read(Some('-')).unwrap().text, "small");
    }

    #[test]
    fn clipboard_and_readonly_names_are_not_owned_here() {
        let r = Registers::new();
        // Stored-register storage returns None for names Editor resolves.
        assert!(r.read(Some('+')).is_none());
        assert!(r.read(Some('*')).is_none());
        assert!(r.read(Some('%')).is_none());
        assert!(r.read(Some('/')).is_none());
    }
}

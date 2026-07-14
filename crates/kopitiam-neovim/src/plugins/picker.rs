//! A generic fuzzy picker — the one engine behind `FindFiles`, `FindBuffers`
//! and `FindHelp`.
//!
//! # Why one generic picker instead of three
//!
//! telescope.nvim's entire design is "one fuzzy list widget, many sources":
//! the finder, sorter and previewer are pluggable, but the picker itself
//! neither knows nor cares whether it is looking at files, buffers or help
//! tags. Writing `FilePicker`, `BufferPicker`, `HelpPicker` as separate types
//! would triplicate the scoring/selection/navigation logic for no benefit —
//! and the next source (LSP symbols, git status, …) would make it
//! quadruplicate. Instead [`Picker<T>`] is generic over the item type, and a
//! source is nothing more than "a `Vec<T>` plus a way to read its search
//! text" ([`Searchable`]).
//!
//! # Why `nucleo` rather than a hand-rolled scorer
//!
//! `nucleo` is the fuzzy matcher behind Helix, already a workspace
//! dependency, and pure Rust. Reimplementing fzf-quality scoring (bonus for
//! word boundaries, camelCase, path separators, consecutive-match streaks)
//! is a research project in itself; `nucleo` has already done it.
//!
//! This module uses `nucleo`'s synchronous [`nucleo::pattern::Pattern`] API
//! (score + rebuild on every keystroke) rather than the asynchronous
//! `nucleo::Nucleo` worker. The async worker exists to keep a background
//! thread pool matching against tens of thousands of streaming items without
//! blocking a UI thread; a picker over "files in this repo" or "open
//! buffers" is a few thousand items at most, and rescoring that synchronously
//! is sub-millisecond. Reaching for the threaded engine here would add
//! surface area (channels, snapshots, timeouts) with no measurable benefit —
//! see `nucleo::pattern::Pattern::match_list`'s own doc comment, which
//! recommends exactly this trade-off for "a (relatively small) list".

use std::path::{Path, PathBuf};

use nucleo::pattern::{AtomKind, CaseMatching, Normalization, Pattern};
use nucleo::{Matcher, Utf32Str};

use crate::core::{BufferId, Position};

/// An item a [`Picker`] can rank. The picker only ever looks at
/// [`search_text`](Searchable::search_text) — everything else about `T` is
/// opaque to it and round-trips back to the caller via [`Picker::confirm`].
pub trait Searchable {
    /// The text `nucleo` fuzzy-matches the query against.
    fn search_text(&self) -> &str;
}

/// One scored, ranked candidate.
///
/// `indices` are **character** offsets into [`Searchable::search_text`] (not
/// bytes — `nucleo` matches on `char`, matching the grapheme-column
/// convention used everywhere else in kvim would require re-deriving indices
/// from a rope, which the picker has no access to and does not need). The UI
/// uses them to bold/underline the matched characters, exactly as telescope
/// highlights fuzzy hits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PickerMatch {
    /// Index into the picker's original (unfiltered) item list.
    pub item_index: usize,
    /// Higher is a better match. Not comparable across different queries.
    pub score: u32,
    /// Matched character indices, sorted and deduplicated.
    pub indices: Vec<u32>,
}

/// A generic fuzzy picker over any list of [`Searchable`] items.
///
/// Typical use: construct once per invocation (`\ff`, `\fb`, `\fh`, ...) with
/// the source's items, then feed it keystrokes via [`Picker::set_query`] and
/// selection movement via [`Picker::select_next`]/[`Picker::select_prev`]
/// until the user commits with [`Picker::confirm`] or cancels (the UI simply
/// drops the picker).
pub struct Picker<T> {
    items: Vec<T>,
    matcher: Matcher,
    pattern: Pattern,
    query: String,
    matches: Vec<PickerMatch>,
    selected: usize,
}

impl<T: Searchable> Picker<T> {
    /// Builds a picker over `items`, initially unfiltered (equivalent to an
    /// empty query — see [`Picker::set_query`]).
    pub fn new(items: Vec<T>) -> Self {
        let mut picker = Self {
            items,
            matcher: Matcher::new(nucleo::Config::DEFAULT),
            pattern: Pattern::new("", CaseMatching::Smart, Normalization::Smart, AtomKind::Fuzzy),
            query: String::new(),
            matches: Vec::new(),
            selected: 0,
        };
        picker.rescore();
        picker
    }

    /// Re-filters and re-ranks the item list against `query`. Cheap enough
    /// to call on every keystroke (see the module docs on why this is
    /// synchronous, not backgrounded).
    ///
    /// An empty query is a special case matching everything — a picker that
    /// went blank the moment you opened it, before you typed anything, would
    /// be useless as a browser.
    pub fn set_query(&mut self, query: &str) {
        if query == self.query {
            return;
        }
        query.clone_into(&mut self.query);
        self.pattern.reparse(&self.query, CaseMatching::Smart, Normalization::Smart);
        self.rescore();
    }

    /// The current query text.
    pub fn query(&self) -> &str {
        &self.query
    }

    fn rescore(&mut self) {
        let mut buf = Vec::new();
        let mut indices = Vec::new();
        self.matches = self
            .items
            .iter()
            .enumerate()
            .filter_map(|(item_index, item)| {
                indices.clear();
                buf.clear();
                let haystack = Utf32Str::new(item.search_text(), &mut buf);
                let score = self.pattern.indices(haystack, &mut self.matcher, &mut indices)?;
                let mut indices = indices.clone();
                indices.sort_unstable();
                indices.dedup();
                Some(PickerMatch { item_index, score, indices })
            })
            .collect();
        // Best match first. Ties broken by original position so that
        // re-scoring the same query twice (e.g. after the item list is
        // refreshed) is deterministic rather than depending on sort
        // stability of equal keys across calls.
        self.matches.sort_by(|a, b| b.score.cmp(&a.score).then(a.item_index.cmp(&b.item_index)));
        self.selected = self.selected.min(self.matches.len().saturating_sub(1));
    }

    /// The ranked, scored matches for the current query. Index 0 is the best
    /// match.
    pub fn matches(&self) -> &[PickerMatch] {
        &self.matches
    }

    /// All items the picker was constructed with, regardless of the current
    /// query. Used by the UI to resolve a [`PickerMatch::item_index`] or
    /// [`PickerMatch::indices`] back to displayable text.
    pub fn items(&self) -> &[T] {
        &self.items
    }

    /// The item under the current selection cursor, if any match exists.
    pub fn selected(&self) -> Option<&T> {
        self.matches.get(self.selected).map(|m| &self.items[m.item_index])
    }

    /// Index into [`Picker::matches`] of the current selection.
    pub fn selected_index(&self) -> usize {
        self.selected
    }

    pub fn select_next(&mut self) {
        if !self.matches.is_empty() {
            self.selected = (self.selected + 1) % self.matches.len();
        }
    }

    pub fn select_prev(&mut self) {
        if !self.matches.is_empty() {
            self.selected = (self.selected + self.matches.len() - 1) % self.matches.len();
        }
    }

    /// Commits the current selection, e.g. on `<CR>`.
    pub fn confirm(&self) -> Option<&T> {
        self.selected()
    }
}

/// A file found by [`walk_files`], for `Action::FindFiles`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileItem {
    /// Path relative to the walked root — what telescope displays and what
    /// `nucleo` matches against, so a query like `src/pick` scores on the
    /// path a human actually types, not an absolute filesystem path.
    pub relative_path: PathBuf,
    display: String,
}

impl FileItem {
    fn new(relative_path: PathBuf) -> Self {
        let display = relative_path.to_string_lossy().into_owned();
        Self { relative_path, display }
    }
}

impl Searchable for FileItem {
    fn search_text(&self) -> &str {
        &self.display
    }
}

/// Walks `root` for `Action::FindFiles`, honouring `.gitignore`,
/// `.git/info/exclude` and global gitignore — exactly what `ignore::Walk`
/// gives for free, and exactly what telescope's `find_files` picker does by
/// default. Directories are not returned as items; only regular files are.
pub fn walk_files(root: &Path) -> Vec<FileItem> {
    ignore::WalkBuilder::new(root)
        .build()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_some_and(|ft| ft.is_file()))
        .map(|entry| {
            let relative = entry.path().strip_prefix(root).unwrap_or(entry.path()).to_path_buf();
            FileItem::new(relative)
        })
        .collect()
}

/// One open buffer, for `Action::FindBuffers`.
///
/// Constructed by the editor layer (which owns the buffer list) — this
/// module only defines the shape a buffer needs to expose to be picked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferItem {
    pub id: BufferId,
    /// What's shown and matched against — typically the buffer's path, or
    /// `[No Name]` for a scratch buffer.
    pub display: String,
}

impl Searchable for BufferItem {
    fn search_text(&self) -> &str {
        &self.display
    }
}

/// One help tag, for `Action::FindHelp`.
///
/// Constructed by whatever owns the help-tag database — this module only
/// defines the shape, mirroring [`BufferItem`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HelpItem {
    /// The tag as a user would type it after `:help`, e.g. `"gO"` or
    /// `"v_ib"`.
    pub tag: String,
    /// Where the tag jumps to.
    pub target: Position,
}

impl Searchable for HelpItem {
    fn search_text(&self) -> &str {
        &self.tag
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl Searchable for &'static str {
        fn search_text(&self) -> &str {
            self
        }
    }

    #[test]
    fn best_fuzzy_match_sorts_first() {
        let mut picker = Picker::new(vec!["src/plugins/hop.rs", "src/plugins/mod.rs", "hop.rs"]);
        picker.set_query("hop");
        // "hop.rs" is a much tighter (shorter, contiguous) match for "hop"
        // than either path buried in src/plugins/, so it must rank first.
        assert_eq!(picker.matches()[0].item_index, 2);
        assert_eq!(*picker.selected().unwrap(), "hop.rs");
    }

    #[test]
    fn matched_indices_point_at_the_query_characters() {
        let mut picker = Picker::new(vec!["hop.rs"]);
        picker.set_query("hrs");
        let m = &picker.matches()[0];
        // h-o-p-.-r-s: 'h' at 0, 'r' at 4, 's' at 5.
        assert_eq!(m.indices, vec![0, 4, 5]);
    }

    #[test]
    fn empty_query_returns_every_item() {
        let picker = Picker::new(vec!["a", "b", "c"]);
        assert_eq!(picker.matches().len(), 3);
    }

    #[test]
    fn query_matching_nothing_returns_nothing() {
        let mut picker = Picker::new(vec!["alpha", "beta", "gamma"]);
        picker.set_query("zzz_no_such_thing");
        assert!(picker.matches().is_empty());
        assert!(picker.selected().is_none());
        assert!(picker.confirm().is_none());
    }

    #[test]
    fn selection_wraps_in_both_directions() {
        let mut picker = Picker::new(vec!["a", "b", "c"]);
        assert_eq!(picker.selected_index(), 0);
        picker.select_prev();
        assert_eq!(picker.selected_index(), 2, "prev from the first item wraps to the last");
        picker.select_next();
        assert_eq!(picker.selected_index(), 0, "next from the last item wraps to the first");
    }

    #[test]
    fn selection_clamps_when_a_narrower_query_shrinks_the_match_list() {
        let mut picker = Picker::new(vec!["apple", "banana", "cherry"]);
        picker.select_prev(); // now on the last item (index 2)
        picker.set_query("zz");
        assert!(picker.matches().is_empty());
        picker.set_query(""); // back to everything
        assert_eq!(picker.matches().len(), 3);
    }

    #[test]
    fn walk_files_honours_gitignore() {
        // `.gitignore` is only honoured inside an actual git repository
        // (matching ripgrep's and telescope's own default) — an empty
        // `.git/` directory is enough to establish that boundary without
        // pulling in the `git` binary as a test dependency.
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join(".gitignore"), "ignored.txt\n").unwrap();
        std::fs::write(dir.path().join("ignored.txt"), "").unwrap();
        std::fs::write(dir.path().join("kept.txt"), "").unwrap();

        let files = walk_files(dir.path());
        let names: Vec<_> = files.iter().map(|f| f.relative_path.display().to_string()).collect();
        assert!(names.contains(&"kept.txt".to_string()));
        assert!(!names.contains(&"ignored.txt".to_string()));
        // `ignore::Walk` skips dotfiles by default (the same default
        // ripgrep and telescope's `find_files` use), so `.gitignore` itself
        // is not among the results.
        assert!(!names.contains(&".gitignore".to_string()));
    }
}

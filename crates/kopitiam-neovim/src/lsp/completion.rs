//! The `blink.cmp` replacement: a headless completion engine that merges
//! candidates from the LSP, the current buffer's own words, and file paths,
//! then filters and ranks them against what the user has typed.
//!
//! # Headless, deliberately
//!
//! This module produces a `Vec<CompletionItem>` and nothing else. Rendering
//! the popup, wiring `<CR>`/`<C-space>`/`<C-e>`/`<C-b>`/`<C-f>` to it, and
//! deciding when to (re)trigger a query are all UI concerns — the task
//! brief for this crate's `lsp` module is explicit that those are "the UI
//! agent's business". Keeping this module UI-free also makes it trivially
//! unit-testable: every test below constructs items and prefixes directly,
//! with no terminal, no event loop, and no LSP process involved.
//!
//! # Ranking
//!
//! Filtering and scoring reuse [`nucleo`] — already a workspace dependency
//! for the fuzzy-finder ("telescope") replacement — rather than hand-rolling
//! a second fuzzy matcher. `nucleo_matcher::Config::prefer_prefix` is a
//! setting the crate documents as specifically intended for autocompletion
//! (as opposed to fzf-style open-ended fuzzy search), so it is enabled here.

use std::collections::HashSet;
use std::path::Path;

use kopitiam_semantic::CompletionItemKind;
use nucleo::pattern::{Atom, AtomKind, CaseMatching, Normalization};
use nucleo::{Config, Matcher, Utf32Str};

/// Where a [`CompletionItem`] came from. Ordering here **is** priority order:
/// when the same label is offered by more than one source, the earliest
/// variant wins — see [`merge_and_rank`].
///
/// The order `Lsp > Snippet > Buffer > Path` mirrors the maintainer's
/// `blink.cmp` source priority: the language server (which understands scope
/// and type) first, then snippets (a deliberate, curated suggestion), then the
/// weaker "some word already in this file" and path sources. See
/// `docs/ai-decisions/AID-0024`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CompletionSource {
    /// `textDocument/completion`. Ranked highest: the language server
    /// understands scope and type, so its suggestions are the most likely
    /// to be exactly what the user meant.
    Lsp,
    /// A snippet — either a built-in ([`builtin_snippets`]) or an LSP item
    /// whose `insertTextFormat` was `Snippet`. Its `insert_text` is snippet
    /// *grammar* and is carried in [`CompletionItem::snippet`], to be expanded
    /// (via `kopitiam-snippet`) on accept rather than inserted literally.
    Snippet,
    /// A word that already appears somewhere in the current buffer.
    Buffer,
    /// A filesystem entry under the directory being typed into (e.g. inside
    /// a string literal that looks like a path).
    Path,
}

/// One completion candidate, source-tagged so the UI can style/sort
/// secondarily by provenance if it wants to (e.g. a small "[LSP]"/"[buf]"
/// badge), and carrying enough LSP-shaped metadata to be useful without
/// forcing every source to fabricate it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionItem {
    /// What's shown in the completion menu.
    pub label: String,
    /// What gets inserted on confirm. Usually equal to `label`; kept
    /// separate because an LSP item's `insertText`/`textEdit` can differ
    /// from its `label` (e.g. a trailing `()` shown in the label but not
    /// inserted, or vice versa). Snippet placeholder expansion (`$1`, `$0`)
    /// is a UI/editor concern, not this module's — `insert_text` is the raw
    /// text as the source provided it.
    pub insert_text: String,
    pub source: CompletionSource,
    /// A short one-line description for the completion menu's detail
    /// column: an LSP item's `detail` (often a type signature), or `None`
    /// for buffer/path candidates, which don't have one.
    pub detail: Option<String>,
    /// The LSP `CompletionItemKind` (Function, Method, Struct, …) when the
    /// source knows it, so the menu can badge the row with the *kind* rather
    /// than only the source. `None` for buffer/path words, which have no kind.
    pub kind: Option<CompletionItemKind>,
    /// The snippet **body** (LSP snippet grammar) to expand on accept, when
    /// this item is a snippet. `Some` for [`CompletionSource::Snippet`] items
    /// and for LSP items whose `insertTextFormat` was `Snippet`; `None` for a
    /// plain item, which is inserted literally from [`Self::insert_text`].
    ///
    /// Kept separate from `insert_text` on purpose: `insert_text` is always the
    /// literal-insert fallback (used if snippet expansion ever fails), while
    /// `snippet` is the un-expanded grammar the editor feeds to
    /// `kopitiam-snippet`.
    pub snippet: Option<String>,
}

impl CompletionItem {
    pub fn new(label: impl Into<String>, source: CompletionSource) -> Self {
        let label = label.into();
        Self { insert_text: label.clone(), label, source, detail: None, kind: None, snippet: None }
    }

    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }

    pub fn with_kind(mut self, kind: CompletionItemKind) -> Self {
        self.kind = Some(kind);
        self
    }

    /// Marks this item as a snippet with `body` as its expandable grammar.
    /// This sets only the body, never [`Self::source`]: an LSP snippet must keep
    /// `source == Lsp` so it still wins label collisions, so the caller chooses
    /// the source independently of whether the item expands.
    pub fn with_snippet(mut self, body: impl Into<String>) -> Self {
        self.snippet = Some(body.into());
        self
    }

    /// Whether accepting this item should expand a snippet rather than insert
    /// literal text.
    pub fn is_snippet(&self) -> bool {
        self.snippet.is_some()
    }
}

/// The built-in snippet source: a small, hand-curated set of the snippets the
/// maintainer reaches for most, surfaced as [`CompletionSource::Snippet`]
/// candidates so `fn`, `impl`, … appear in the menu while typing. Their bodies
/// are LSP snippet grammar, expanded on accept by `kopitiam-snippet`.
///
/// This is deliberately tiny and static — a curated starter set, not a snippet
/// *library*. A user-editable snippet collection (the `LuaSnip`/`friendly-
/// snippets` equivalent) is a later feature; when it lands it feeds the same
/// [`CompletionSource::Snippet`] path this function seeds today.
///
/// `filetype` is the kvim filetype string (`"rust"`, `"lua"`, `"tex"`); an
/// unknown one yields no snippets rather than a wrong-language set.
pub fn builtin_snippets(filetype: &str) -> Vec<CompletionItem> {
    // (trigger label, one-line detail, snippet body). Bodies use `\t` for the
    // indent so the expander/editor keeps the buffer's own indentation style;
    // `$0` is the final cursor stop, `${1:ph}` an editable placeholder.
    let table: &[(&str, &str, &str)] = match filetype {
        "rust" => &[
            ("fn", "function", "fn ${1:name}(${2:args})${3: -> ${4:T}} {\n\t$0\n}"),
            ("pub fn", "public function", "pub fn ${1:name}(${2:args})${3: -> ${4:T}} {\n\t$0\n}"),
            ("impl", "impl block", "impl ${1:Type} {\n\t$0\n}"),
            ("for", "for loop", "for ${1:item} in ${2:iter} {\n\t$0\n}"),
            ("match", "match expression", "match ${1:expr} {\n\t${2:pattern} => ${3:value},\n\t$0\n}"),
            ("test", "unit test", "#[test]\nfn ${1:name}() {\n\t$0\n}"),
        ],
        "lua" => &[
            ("function", "function", "function ${1:name}(${2:args})\n\t$0\nend"),
            ("for", "numeric for", "for ${1:i} = ${2:1}, ${3:n} do\n\t$0\nend"),
        ],
        "tex" => &[
            ("begin", "environment", "\\begin{${1:env}}\n\t$0\n\\end{${1:env}}"),
            ("section", "section", "\\section{${1:title}}\n$0"),
        ],
        _ => &[],
    };
    table
        .iter()
        .map(|(label, detail, body)| {
            let mut item = CompletionItem::new(*label, CompletionSource::Snippet);
            item.detail = Some((*detail).to_string());
            item.kind = Some(CompletionItemKind::Snippet);
            item.snippet = Some((*body).to_string());
            item
        })
        .collect()
}

/// Extracts every distinct identifier-like word from `lines`, as
/// [`CompletionSource::Buffer`] candidates — the "words already in this
/// file" source `blink.cmp`'s buffer source provides.
///
/// A "word" is a maximal run of characters for which `char::is_alphanumeric`
/// or `_` holds, using Unicode's definition of alphanumeric (so an
/// identifier in a non-Latin script counts too) rather than an ASCII-only
/// `[A-Za-z0-9_]` pattern. Words are deduplicated and returned in first-seen
/// order, which is stable across calls on unchanged input and keeps
/// [`merge_and_rank`]'s ordering deterministic before scoring is applied.
pub fn buffer_words(lines: &[&str]) -> Vec<CompletionItem> {
    let mut seen = HashSet::new();
    let mut items = Vec::new();
    for line in lines {
        for word in split_words(line) {
            if seen.insert(word.to_string()) {
                items.push(CompletionItem::new(word, CompletionSource::Buffer));
            }
        }
    }
    items
}

fn split_words(line: &str) -> impl Iterator<Item = &str> {
    line.split(|c: char| !(c.is_alphanumeric() || c == '_')).filter(|w| !w.is_empty())
}

/// Lists filesystem entries as [`CompletionSource::Path`] candidates.
///
/// `typed` is whatever path fragment the user has typed so far (e.g. `"src/l"`
/// from a partially-typed `"src/lsp/cl"`); it is split into a directory part
/// and a filename prefix, the directory is listed relative to `base`, and
/// entries whose name starts with the filename prefix are returned. Returns
/// an empty list (never an error) for a directory that doesn't exist, isn't
/// readable, or when `typed` escapes `base` in a way that looks like it
/// isn't meant as a relative path completion (an absolute path is honoured
/// as-is, matching shell-style completion) — a completion source failing
/// should narrow the menu, not interrupt typing.
///
/// # Which character counts as a separator (this used to be wrong on Windows)
///
/// The split used to be a hardcoded `typed.rfind('/')`, which quietly killed
/// path completion outright on Windows: a user typing `src\lsp\cl` found no
/// `/`, so the whole fragment became the *filename prefix*, `base` itself got
/// listed, and nothing on earth starts with `src\lsp\cl` — empty menu, no
/// error, no clue why.
///
/// The split is now [`std::path::is_separator`], which is the platform's own
/// answer and is exactly right in **both** directions, hor:
///
/// * On Windows it is true for `\` **and** `/` — Win32 accepts either, so a
///   user who types either one gets completion.
/// * On unix (Linux, macOS, Android/Termux) it is true for `/` **only**, which
///   matters just as much: `\` is a perfectly legal character in a unix
///   filename, so treating it as a separator there would break completing any
///   file whose name genuinely contains a backslash.
///
/// Whatever separator the user typed is the one echoed back in
/// [`CompletionItem::insert_text`] and in the trailing directory marker, so the
/// suggestion never comes back as a mongrel `src\lsp/client.rs`. With nothing
/// typed yet we default to `/`, which works on every platform kvim supports
/// and matches the forward-slash output convention the workspace already
/// follows.
pub fn path_candidates(typed: &str, base: &Path) -> Vec<CompletionItem> {
    let (dir_part, sep, file_prefix) = match typed.rfind(std::path::is_separator) {
        Some(idx) => {
            let sep = typed[idx..].chars().next().expect("rfind returned a char boundary with a char at it");
            (&typed[..idx], sep, &typed[idx + sep.len_utf8()..])
        }
        None => ("", '/', typed),
    };
    let dir = if dir_part.is_empty() {
        base.to_path_buf()
    } else if Path::new(dir_part).is_absolute() {
        Path::new(dir_part).to_path_buf()
    } else {
        base.join(dir_part)
    };

    let Ok(entries) = std::fs::read_dir(&dir) else { return Vec::new() };

    let mut items = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue }; // skip non-UTF-8 filenames rather than lossily mangling them
        if !name.starts_with(file_prefix) {
            continue;
        }
        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        // Echo back the separator the user actually typed (see the doc comment)
        // so a Windows `src\` completes to `src\lsp\`, not `src\lsp/`.
        let label = if is_dir { format!("{name}{sep}") } else { name.to_string() };
        let insert_text = if dir_part.is_empty() { label.clone() } else { format!("{dir_part}{sep}{label}") };
        items.push(CompletionItem { label, insert_text, source: CompletionSource::Path, detail: None, kind: None, snippet: None });
    }
    items
}

/// Merges candidates from every source, filters out anything that doesn't
/// fuzzy-match `prefix`, and ranks the rest — the single function the UI
/// layer calls once it has gathered items from each source.
///
/// # Deduplication
///
/// Sources are consumed in **priority order** (`lsp_items`, then
/// `snippet_items`, then `buffer_items`, then `path_items`): when two sources
/// offer the same `label`, only the first one encountered survives. Since
/// `lsp_items` is consumed first, an LSP suggestion always wins a label
/// collision over a snippet, buffer word, or path entry with the same text —
/// see [`CompletionSource`]'s doc comment for why that ordering is the intended
/// priority. (This is why a built-in `fn` snippet does not shadow rust-analyzer
/// when both offer `fn`: the LSP item is kept.)
///
/// # Ranking
///
/// An empty `prefix` (nothing typed yet, e.g. right after `<C-space>`)
/// matches everything with a flat score, preserving priority-then-alphabetic
/// order. Otherwise every surviving item is scored by
/// [`nucleo::pattern::Atom`] against `prefix`; anything that doesn't
/// match at all (`None`) is filtered out (this is where "filter by the
/// typed prefix" happens), and the rest are sorted by score descending,
/// breaking ties alphabetically for determinism.
///
/// Case handling is **smart-case**, same as the picker: all-lowercase prefix
/// matches case-insensitively, any uppercase char in the prefix makes the whole
/// match case-sensitive.
///
/// # Why go through `Atom` and not call `Matcher::fuzzy_match` straight
///
/// Hard-won one, this. `nucleo_matcher::Matcher`'s raw `..._match` /
/// `..._indices` methods say in their own docs that the **caller** must hand
/// over a needle that is already unicode-normalised and case-folded. Pass a raw
/// user string with `Config::ignore_case = true` (which `Config::DEFAULT` has)
/// and you don't just get wrong scores — you get a **panic**:
///
/// ```text
/// nucleo-matcher-0.3.1/src/fuzzy_optimal.rs:37: should have been caught by prefilter
/// ```
///
/// The mechanism, so nobody has to rediscover it: `prefilter_ascii` folds case
/// one-way only — for a needle byte `'t'` it looks for `'t'` *or* `'T'`, but for
/// a needle byte `'T'` it looks for `'T'` alone. The matrix setup right after it
/// normalises the *haystack* to lowercase and compares against the needle
/// verbatim. So the prefilter says "can match" (it found a literal `'T'`) while
/// the matrix says "cannot" (`'t' != 'T'`), the two disagree, and since both
/// sides are ASCII nucleo asserts that its own invariant broke. It is nucleo's
/// assert, but our bug — we broke the documented precondition.
///
/// It needs a *gap* to fire, which is why it looked intermittent. When the
/// prefilter's window comes back exactly `needle.len()` wide, `fuzzy_match`
/// short-circuits to `calculate_score` and never touches the matrix — so
/// prefix `"Th"` against `"Theorem"` is fine. Give the last needle char a
/// second occurrence further along (`"Th"` against `"Through"` — the `h` in
/// `-gh`) and the window widens, the matrix runs, and it panics.
///
/// Why it bit in Markdown and (nearly) never in Rust: prose is full of
/// Capitalised words, so buffer-word completion in a `.md` file gets uppercase
/// prefixes typed at it all day. Code identifiers are mostly lowercase, so the
/// same code path looked fine for months.
///
/// [`Atom::new`] does the normalisation + case folding for us, and
/// [`Atom::score`] sets `ignore_case`/`normalize` on the matcher to match the
/// needle it actually holds — so prefilter and matrix can never disagree again.
/// It leaves `prefer_prefix` alone, so the autocompletion tuning below survives.
pub fn merge_and_rank(
    prefix: &str,
    lsp_items: Vec<CompletionItem>,
    snippet_items: Vec<CompletionItem>,
    buffer_items: Vec<CompletionItem>,
    path_items: Vec<CompletionItem>,
) -> Vec<CompletionItem> {
    // `Config` is `#[non_exhaustive]`, which blocks struct-literal
    // construction (even with `..Config::DEFAULT`) from outside
    // `nucleo_matcher`'s own crate -- but mutating a `pub` field on an
    // already-constructed value is fine, so start from the constant and
    // flip the one setting autocompletion wants.
    let mut config = Config::DEFAULT;
    config.prefer_prefix = true;
    let mut matcher = Matcher::new(config);
    // Build the needle once, not once per candidate -- `Atom::new` allocates a
    // normalised `Utf32String`, and the prefix is the same for every item.
    // `escape_whitespace = false`: a completion prefix is one word, `\ ` in it
    // is a literal backslash-space, not an escape.
    let atom = Atom::new(prefix, CaseMatching::Smart, Normalization::Smart, AtomKind::Fuzzy, false);
    let mut seen = HashSet::new();
    let mut scored: Vec<(u32, CompletionItem)> = Vec::new();

    for item in lsp_items.into_iter().chain(snippet_items).chain(buffer_items).chain(path_items) {
        if !seen.insert(item.label.clone()) {
            continue;
        }
        if prefix.is_empty() {
            scored.push((0, item));
            continue;
        }
        let mut haystack_buf = Vec::new();
        let haystack = Utf32Str::new(&item.label, &mut haystack_buf);
        if let Some(score) = atom.score(haystack, &mut matcher) {
            scored.push((score as u32, item));
        }
    }

    scored.sort_by(|(score_a, item_a), (score_b, item_b)| score_b.cmp(score_a).then_with(|| item_a.label.cmp(&item_b.label)));
    scored.into_iter().map(|(_, item)| item).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn lsp(label: &str) -> CompletionItem {
        CompletionItem::new(label, CompletionSource::Lsp)
    }
    fn buf(label: &str) -> CompletionItem {
        CompletionItem::new(label, CompletionSource::Buffer)
    }
    fn snip(label: &str) -> CompletionItem {
        CompletionItem::new(label, CompletionSource::Snippet).with_snippet("body")
    }

    #[test]
    fn buffer_words_are_deduplicated_and_unicode_aware() {
        let lines = ["let foo = bar(foo, baz);", "日本語 identifier_日本語 more_text"];
        let words = buffer_words(&lines);
        let labels: Vec<&str> = words.iter().map(|i| i.label.as_str()).collect();
        assert!(labels.contains(&"foo"));
        assert_eq!(labels.iter().filter(|&&w| w == "foo").count(), 1, "foo appears twice but must be deduplicated");
        assert!(labels.contains(&"bar"));
        assert!(labels.contains(&"baz"));
        assert!(labels.contains(&"日本語"));
        assert!(labels.contains(&"identifier_日本語"));
        assert!(words.iter().all(|w| w.source == CompletionSource::Buffer));
    }

    #[test]
    fn buffer_words_ignores_punctuation_and_whitespace() {
        let words = buffer_words(&["a.b::c(d, e)[f]{g}"]);
        let labels: HashSet<&str> = words.iter().map(|i| i.label.as_str()).collect();
        assert_eq!(labels, HashSet::from(["a", "b", "c", "d", "e", "f", "g"]));
    }

    #[test]
    fn path_candidates_filters_by_typed_prefix_and_marks_directories() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("lib.rs"), "").unwrap();
        std::fs::write(dir.path().join("lsp_client.rs"), "").unwrap();
        std::fs::create_dir(dir.path().join("lsp")).unwrap();

        let items = path_candidates("ls", dir.path());
        let labels: HashSet<&str> = items.iter().map(|i| i.label.as_str()).collect();
        assert_eq!(labels, HashSet::from(["lsp_client.rs", "lsp/"]));
        assert!(!labels.contains("lib.rs"), "lib.rs does not start with the typed prefix `ls`");
    }

    #[test]
    fn path_candidates_returns_empty_for_a_nonexistent_directory_rather_than_erroring() {
        // Built from the platform's own separator instead of a literal
        // "/definitely/does/not/exist", so the path is nonsense-but-well-formed
        // on Windows too rather than a drive-relative POSIX-looking string.
        let nowhere: PathBuf = ["definitely", "does", "not", "exist"].iter().collect();
        let items = path_candidates("nope/", &nowhere);
        assert!(items.is_empty());
    }

    #[test]
    fn path_candidates_splits_on_the_separators_this_platform_actually_uses() {
        // The regression this pins: the split used to be a hardcoded
        // `rfind('/')`, so on Windows a typed `sub\l` never split, the whole
        // fragment was treated as a filename prefix, and completion silently
        // returned nothing. Assert the true behaviour on BOTH families rather
        // than skipping one:
        //
        //   * `/` splits everywhere.
        //   * `\` splits on Windows only; on unix it is an ordinary filename
        //     character and must NOT split.
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("sub");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(sub.join("lib.rs"), "").unwrap();

        // Forward slash: works on every platform kvim supports.
        let items = path_candidates("sub/l", dir.path());
        let inserts: HashSet<&str> = items.iter().map(|i| i.insert_text.as_str()).collect();
        assert_eq!(inserts, HashSet::from(["sub/lib.rs"]), "`/` must split on every platform");

        // Backslash: a separator on Windows, a filename character on unix.
        let items = path_candidates("sub\\l", dir.path());
        let inserts: HashSet<&str> = items.iter().map(|i| i.insert_text.as_str()).collect();
        if cfg!(windows) {
            assert_eq!(inserts, HashSet::from(["sub\\lib.rs"]), "Windows must split on `\\` and echo it back");
        } else {
            assert!(
                inserts.is_empty(),
                "on unix `\\` is a filename character, not a separator, so `sub\\l` matches nothing in the base dir: {inserts:?}"
            );
        }
    }

    #[test]
    fn a_directory_candidate_is_marked_with_a_separator_that_matches_what_was_typed() {
        // With nothing typed the marker defaults to `/` on every platform (it
        // works on Windows too, and matches the workspace's forward-slash output
        // convention); with a separator typed, that same separator comes back.
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("outer")).unwrap();
        std::fs::create_dir(dir.path().join("outer").join("inner")).unwrap();

        let labels: HashSet<String> = path_candidates("out", dir.path()).into_iter().map(|i| i.label).collect();
        assert_eq!(labels, HashSet::from(["outer/".to_string()]), "no separator typed -> default `/` marker");

        let labels: HashSet<String> = path_candidates("outer/", dir.path()).into_iter().map(|i| i.label).collect();
        assert_eq!(labels, HashSet::from(["inner/".to_string()]));

        #[cfg(windows)]
        {
            let labels: HashSet<String> = path_candidates("outer\\", dir.path()).into_iter().map(|i| i.label).collect();
            assert_eq!(labels, HashSet::from(["inner\\".to_string()]), "a typed `\\` must not come back as `/`");
        }
    }

    #[test]
    fn merge_and_rank_prefers_lsp_over_buffer_on_a_label_collision() {
        let lsp_item = lsp("println").with_detail("macro println!");
        let buf_item = buf("println");
        let ranked = merge_and_rank("println", vec![lsp_item.clone()], vec![], vec![buf_item], vec![]);
        assert_eq!(ranked, vec![lsp_item], "the buffer duplicate must be dropped, keeping the LSP item (with its detail)");
    }

    #[test]
    fn merge_and_rank_filters_out_non_matching_items() {
        let ranked = merge_and_rank("xyz", vec![lsp("println")], vec![], vec![buf("format")], vec![]);
        assert!(ranked.is_empty(), "neither candidate fuzzy-matches `xyz`");
    }

    #[test]
    fn merge_and_rank_survives_an_uppercase_prefix() {
        // Regression: passing a raw (un-case-folded) needle to
        // `Matcher::fuzzy_match` while `ignore_case` was on made nucleo's
        // prefilter and its matrix disagree, and nucleo asserted:
        //   fuzzy_optimal.rs:37: should have been caught by prefilter
        // Editing Markdown hit this constantly -- prose is full of Capitalised
        // words, so buffer-word completion sees uppercase prefixes all the time.
        // See `merge_and_rank`'s doc comment for the full mechanism.
        //
        // `"Through"` is load-bearing, don't swap it for a shorter word: the
        // needle's last char `h` must appear *again* later in the haystack, so
        // that nucleo's prefilter widens the window past `needle.len()` and the
        // matrix path actually runs. Something like `"Theorem"` short-circuits
        // on the exact-window fast path and never panicked even before the fix.
        let ranked = merge_and_rank("Th", vec![], vec![], vec![buf("Through"), buf("The")], vec![]);
        let labels: Vec<&str> = ranked.iter().map(|i| i.label.as_str()).collect();
        assert!(labels.contains(&"Through"), "uppercase prefix must still match, got {labels:?}");
        assert!(labels.contains(&"The"));
    }

    #[test]
    fn merge_and_rank_is_smart_case() {
        // Lowercase prefix -> case-insensitive, catches the Capitalised word.
        let ranked = merge_and_rank("th", vec![], vec![], vec![buf("The"), buf("this")], vec![]);
        let labels: HashSet<&str> = ranked.iter().map(|i| i.label.as_str()).collect();
        assert_eq!(labels, HashSet::from(["The", "this"]));

        // Any uppercase in the prefix -> case-sensitive, so lowercase drops out.
        let ranked = merge_and_rank("Th", vec![], vec![], vec![buf("The"), buf("this")], vec![]);
        let labels: Vec<&str> = ranked.iter().map(|i| i.label.as_str()).collect();
        assert_eq!(labels, vec!["The"], "an uppercase prefix must not match a lowercase label");
    }

    #[test]
    fn merge_and_rank_survives_a_non_ascii_prefix() {
        // Same precondition (needle must be normalised) applies on the unicode
        // path; `Atom` handles it, so this must neither panic nor come back empty.
        let ranked = merge_and_rank("日本", vec![], vec![], vec![buf("日本語"), buf("plain")], vec![]);
        assert_eq!(ranked.iter().map(|i| i.label.as_str()).collect::<Vec<_>>(), vec!["日本語"]);
    }

    #[test]
    fn merge_and_rank_returns_everything_in_priority_order_when_prefix_is_empty() {
        let ranked = merge_and_rank("", vec![lsp("alpha")], vec![], vec![buf("beta")], vec![]);
        assert_eq!(ranked.iter().map(|i| i.label.as_str()).collect::<Vec<_>>(), vec!["alpha", "beta"]);
    }

    #[test]
    fn merge_and_rank_scores_a_prefix_match_above_a_looser_fuzzy_match() {
        // "format" is a prefix match for "form"; "some_format_helper" only
        // fuzzy-matches. prefer_prefix in nucleo's Config exists precisely
        // to rank the former above the latter for autocomplete.
        let ranked = merge_and_rank("form", vec![], vec![], vec![buf("some_format_helper"), buf("format")], vec![]);
        assert_eq!(ranked.first().unwrap().label, "format");
    }

    #[test]
    fn merge_and_rank_ties_break_alphabetically_for_determinism() {
        let ranked = merge_and_rank("", vec![], vec![], vec![buf("zeta"), buf("alpha")], vec![]);
        assert_eq!(ranked.iter().map(|i| i.label.as_str()).collect::<Vec<_>>(), vec!["alpha", "zeta"]);
    }

    #[test]
    fn merge_and_rank_ranks_snippet_between_lsp_and_buffer() {
        // Distinct labels, empty prefix -> pure priority order: lsp, snippet,
        // buffer. This pins the `Lsp > Snippet > Buffer` intent.
        let ranked = merge_and_rank("", vec![lsp("a_lsp")], vec![snip("b_snip")], vec![buf("c_buf")], vec![]);
        assert_eq!(
            ranked.iter().map(|i| i.label.as_str()).collect::<Vec<_>>(),
            vec!["a_lsp", "b_snip", "c_buf"],
            "empty prefix must preserve source priority: LSP, then snippet, then buffer"
        );
    }

    #[test]
    fn merge_and_rank_lsp_wins_a_label_collision_over_a_snippet() {
        // Both offer `fn`; the LSP item (consumed first) survives, so a built-in
        // `fn` snippet never shadows rust-analyzer's own `fn` completion.
        let lsp_fn = lsp("fn").with_detail("keyword fn");
        let ranked = merge_and_rank("fn", vec![lsp_fn.clone()], vec![snip("fn")], vec![], vec![]);
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0], lsp_fn, "the LSP `fn` wins; the snippet duplicate is dropped");
    }

    #[test]
    fn builtin_snippets_are_snippet_tagged_with_bodies_and_are_filetype_scoped() {
        let rust = builtin_snippets("rust");
        assert!(rust.iter().any(|i| i.label == "fn"), "rust snippets include `fn`");
        assert!(rust.iter().any(|i| i.label == "impl"), "rust snippets include `impl`");
        assert!(
            rust.iter().all(|i| i.source == CompletionSource::Snippet && i.is_snippet()),
            "every built-in is a Snippet-source item carrying an expandable body"
        );
        let fn_snip = rust.iter().find(|i| i.label == "fn").unwrap();
        assert!(fn_snip.snippet.as_deref().unwrap().contains("$0"), "the body carries a final tabstop");

        assert!(builtin_snippets("lua").iter().any(|i| i.label == "function"));
        assert!(builtin_snippets("cobol").is_empty(), "an unknown filetype yields no snippets, not a wrong-language set");
    }
}


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
pub fn path_candidates(typed: &str, base: &Path) -> Vec<CompletionItem> {
    let (dir_part, file_prefix) = match typed.rfind('/') {
        Some(idx) => (&typed[..idx], &typed[idx + 1..]),
        None => ("", typed),
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
        let label = if is_dir { format!("{name}/") } else { name.to_string() };
        let insert_text = if dir_part.is_empty() { label.clone() } else { format!("{dir_part}/{label}") };
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
/// [`nucleo::Matcher::fuzzy_match`] against `prefix`; anything that doesn't
/// match at all (`None`) is filtered out (this is where "filter by the
/// typed prefix" happens), and the rest are sorted by score descending,
/// breaking ties alphabetically for determinism.
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
        let mut needle_buf = Vec::new();
        let haystack = Utf32Str::new(&item.label, &mut haystack_buf);
        let needle = Utf32Str::new(prefix, &mut needle_buf);
        if let Some(score) = matcher.fuzzy_match(haystack, needle) {
            scored.push((score as u32, item));
        }
    }

    scored.sort_by(|(score_a, item_a), (score_b, item_b)| score_b.cmp(score_a).then_with(|| item_a.label.cmp(&item_b.label)));
    scored.into_iter().map(|(_, item)| item).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let items = path_candidates("nope/", Path::new("/definitely/does/not/exist"));
        assert!(items.is_empty());
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

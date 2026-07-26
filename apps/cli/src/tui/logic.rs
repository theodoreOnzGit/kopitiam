//! Pure, UI-free logic for the TUI app.
//!
//! Everything here is a plain function over paths and strings with no ratatui,
//! no crossterm, and no terminal I/O, so it is unit-testable without a TTY.
//! The views (`convert`, `explorer`, `home`, ...) call into these helpers; the
//! router (`super::App::handle_key`) is tested separately in `super`'s tests.
//!
//! Four families of logic live here:
//!
//! * **Discovery** — [`discover_pdfs`] recursively finds `*.pdf` under a root
//!   (via `walkdir`, pure Rust, Android/Termux-safe).
//! * **Fuzzy ranking** — [`fuzzy_rank`] filters+ranks candidate strings against
//!   a query with `nucleo-matcher` (pure Rust), returning indices so callers
//!   keep their own richer candidate type.
//! * **Output-path derivation** — [`derive_md_name`] (stem → `.md`),
//!   [`mirror_output_path`] (mirror a subtree into an output root), and
//!   [`default_batch_output`] (the sensible default output folder).

use std::path::{Path, PathBuf};

use nucleo_matcher::pattern::{CaseMatching, Normalization, Pattern};
use nucleo_matcher::{Config, Matcher, Utf32Str};
use walkdir::WalkDir;

/// Recursively find every `*.pdf` file under `root` (case-insensitive on the
/// extension), returned sorted by path for a stable, predictable listing.
///
/// Uses `walkdir`, which is pure Rust over `std::fs` — no platform syscalls of
/// our own — so it walks the same way on Linux, macOS, Windows, and Android
/// (Termux). Unreadable entries are silently skipped rather than aborting the
/// whole walk, so one permission-denied directory never sinks discovery.
pub fn discover_pdfs(root: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| entry.into_path())
        .filter(|path| is_pdf(path))
        .collect();
    out.sort();
    out
}

/// True when `path` has a `.pdf` extension, compared case-insensitively so
/// `Paper.PDF` and `paper.pdf` both count.
pub fn is_pdf(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("pdf"))
}

/// Fuzzy-filter and rank `candidates` against `query`, returning the indices of
/// the matching candidates best-first.
///
/// An empty (or whitespace-only) query matches everything and preserves the
/// original order — the natural "no filter yet" state of a live search box. A
/// non-empty query is parsed by `nucleo-matcher`; candidates that do not match
/// at all are dropped, and the rest are sorted by descending match score with
/// the original index as a stable tie-breaker.
///
/// Returning indices (rather than cloned strings) lets a caller keep a richer
/// parallel array — e.g. the absolute `PathBuf` behind each display string —
/// and simply index into it with the result.
pub fn fuzzy_rank(query: &str, candidates: &[String]) -> Vec<usize> {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return (0..candidates.len()).collect();
    }

    // `match_paths` biases scoring toward path-like haystacks (favouring matches
    // against the final component), which is exactly what a PDF picker wants.
    let mut matcher = Matcher::new(Config::DEFAULT.match_paths());
    let pattern = Pattern::parse(trimmed, CaseMatching::Ignore, Normalization::Smart);

    let mut buf: Vec<char> = Vec::new();
    let mut scored: Vec<(usize, u32)> = candidates
        .iter()
        .enumerate()
        .filter_map(|(idx, candidate)| {
            let haystack = Utf32Str::new(candidate, &mut buf);
            pattern.score(haystack, &mut matcher).map(|score| (idx, score))
        })
        .collect();

    // Highest score first; ties broken by original order so the list does not
    // jitter as scores collide.
    scored.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    scored.into_iter().map(|(idx, _)| idx).collect()
}

/// Derive the default output Markdown filename for a PDF: its file stem plus a
/// `.md` extension (e.g. `paper.pdf` → `paper.md`). A path with no usable stem
/// falls back to `output.md`, so this never returns an empty name.
pub fn derive_md_name(pdf: &Path) -> String {
    let stem = pdf
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("output");
    format!("{stem}.md")
}

/// The sensible default output folder for a recursive batch convert: a
/// `markdown` subdirectory of the input folder. Kept a pure function so the
/// default is testable and consistent between the UI prefill and any headless
/// use.
pub fn default_batch_output(input_root: &Path) -> PathBuf {
    input_root.join("markdown")
}

/// Map an input PDF path to its mirrored output `.md` path under `output_root`,
/// preserving the subdirectory structure relative to `input_root`.
///
/// For example `mirror_output_path("in", "out", "in/a/b/x.pdf")` yields
/// `out/a/b/x.md`. If `pdf` is not actually under `input_root` (which should
/// not happen for discovered files, but is handled defensively), only the file
/// name is mirrored so the output never escapes `output_root`.
pub fn mirror_output_path(input_root: &Path, output_root: &Path, pdf: &Path) -> PathBuf {
    let relative = pdf.strip_prefix(input_root).unwrap_or_else(|_| {
        // Not under the root — fall back to just the file name.
        Path::new(pdf.file_name().unwrap_or(pdf.as_os_str()))
    });
    output_root.join(relative).with_extension("md")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_md_name_swaps_extension_to_md() {
        assert_eq!(derive_md_name(Path::new("paper.pdf")), "paper.md");
        assert_eq!(derive_md_name(Path::new("/a/b/report.PDF")), "report.md");
        assert_eq!(derive_md_name(Path::new("no_ext")), "no_ext.md");
    }

    #[test]
    fn derive_md_name_falls_back_on_empty_stem() {
        assert_eq!(derive_md_name(Path::new("")), "output.md");
    }

    #[test]
    fn mirror_output_path_preserves_subdirectories() {
        let got = mirror_output_path(
            Path::new("in"),
            Path::new("out"),
            Path::new("in/a/b/x.pdf"),
        );
        assert_eq!(got, PathBuf::from("out/a/b/x.md"));
    }

    #[test]
    fn mirror_output_path_handles_root_level_file() {
        let got = mirror_output_path(Path::new("in"), Path::new("out"), Path::new("in/x.pdf"));
        assert_eq!(got, PathBuf::from("out/x.md"));
    }

    #[test]
    fn mirror_output_path_defends_against_paths_outside_root() {
        // A path not under `input_root` mirrors only its file name, never
        // escaping the output root.
        let got = mirror_output_path(
            Path::new("in"),
            Path::new("out"),
            Path::new("elsewhere/y.pdf"),
        );
        assert_eq!(got, PathBuf::from("out/y.md"));
    }

    #[test]
    fn default_batch_output_is_a_markdown_subdir() {
        assert_eq!(
            default_batch_output(Path::new("/data/papers")),
            PathBuf::from("/data/papers/markdown")
        );
    }

    #[test]
    fn empty_query_returns_every_candidate_in_order() {
        let candidates = vec!["a.pdf".to_string(), "b.pdf".to_string(), "c.pdf".to_string()];
        assert_eq!(fuzzy_rank("", &candidates), vec![0, 1, 2]);
        assert_eq!(fuzzy_rank("   ", &candidates), vec![0, 1, 2]);
    }

    #[test]
    fn fuzzy_rank_picks_the_expected_file_first() {
        let candidates = vec![
            "alpha/quarterly_report.pdf".to_string(),
            "beta/executive_summary.pdf".to_string(),
            "gamma/report_final_draft.pdf".to_string(),
        ];
        // "summary" clearly targets the second candidate.
        let ranked = fuzzy_rank("summary", &candidates);
        assert_eq!(ranked.first().copied(), Some(1), "ranked = {ranked:?}");
    }

    #[test]
    fn fuzzy_rank_drops_non_matching_candidates() {
        let candidates = vec!["report.pdf".to_string(), "invoice.pdf".to_string()];
        // "zzzzz" matches nothing.
        assert!(fuzzy_rank("zzzzz", &candidates).is_empty());
    }

    #[test]
    fn fuzzy_rank_ranks_a_subsequence_match_above_a_non_match() {
        let candidates = vec![
            "annual_meeting_notes.pdf".to_string(),
            "budget.pdf".to_string(),
        ];
        // "amn" is a subsequence of the first (Annual Meeting Notes) and not of
        // the second, so the first must rank and the second must be excluded.
        let ranked = fuzzy_rank("amn", &candidates);
        assert_eq!(ranked, vec![0]);
    }

    #[test]
    fn discover_pdfs_finds_nested_pdfs_only() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("sub/deep")).unwrap();
        std::fs::write(root.join("top.pdf"), b"x").unwrap();
        std::fs::write(root.join("sub/mid.PDF"), b"x").unwrap();
        std::fs::write(root.join("sub/deep/leaf.pdf"), b"x").unwrap();
        std::fs::write(root.join("notes.txt"), b"x").unwrap();

        let found = discover_pdfs(root);
        assert_eq!(found.len(), 3, "found = {found:?}");
        assert!(found.iter().all(|p| is_pdf(p)));
        assert!(found.iter().any(|p| p.ends_with("top.pdf")));
        assert!(found.iter().any(|p| p.ends_with("sub/mid.PDF") || p.ends_with("sub\\mid.PDF")));
        assert!(found.iter().any(|p| p.file_name().unwrap() == "leaf.pdf"));
    }
}

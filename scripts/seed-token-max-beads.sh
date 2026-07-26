#!/usr/bin/env bash
#
# seed-token-max-beads.sh -- fold the kopitiam Token-Efficiency Work Order
# (kopitiam_token_max.md) into the beads (bd) tracker: 26 task cards + the
# dependency edges from the doc's "Suggested order" (§13).
#
# DETERMINISTIC in the commands it issues (same doc -> same bd invocations), but
# NOT idempotent: `bd` assigns fresh IDs on each run, so re-running would create
# duplicates. A guard aborts if it detects a prior seed. Run once.
#
# `beads-rs` (the `bd` binary) must be installed and on PATH. NOTE: beads-rs
# v0.1.26 currently fails to compile on this toolchain -- install a working
# version first, then run this.
#
# Descriptions are deliberately terse and point back to the doc section rather
# than duplicating it -- per the doc's own principle §0.1 ("coordinates over
# content"). The doc is the source of truth.
#
# Two assumptions to verify against `bd create --help` / `bd dep --help` once a
# working bd exists (adjust if the CLI differs):
#   * `bd create ... --json` prints an object containing "id":"<assigned-id>".
#   * `bd dep add <A> <B>` means "A depends on / is blocked by B".
set -euo pipefail

command -v bd >/dev/null 2>&1 || {
  echo "error: 'bd' (beads-rs) not on PATH." >&2
  echo "       beads-rs v0.1.26 fails to build on the current toolchain; install a working beads-rs first." >&2
  exit 1
}
cd "$(git rev-parse --show-toplevel)"
DOC="kopitiam_token_max.md"

# --- guard against double-seeding ---
if bd list --json 2>/dev/null | grep -q 'I-A --'; then
  echo "error: an 'I-A --' issue already exists; the tracker looks seeded. Aborting to avoid duplicates." >&2
  exit 1
fi

declare -A ID   # card code -> bd-assigned issue id

mk() { # mk <code> <type> <prio> <labels> <title> <description> <acceptance>
  local code="$1" typ="$2" prio="$3" labels="$4" title="$5" desc="$6" acc="$7" out id
  out="$(bd create "$title" -t "$typ" -p "$prio" -l "$labels" -d "$desc" --acceptance "$acc" --json)"
  id="$(printf '%s' "$out" | grep -oE '"id"[[:space:]]*:[[:space:]]*"[^"]+"' | head -1 | sed -E 's/.*"([^"]+)"$/\1/')"
  [ -n "$id" ] || { echo "error: could not parse an id for $code from bd output:" >&2; printf '%s\n' "$out" >&2; exit 1; }
  ID["$code"]="$id"
  printf '  %-5s -> %s\n' "$code" "$id"
}

dep() { # dep <child> <blocker>   (child depends on / is blocked by blocker)
  printf '  %-5s blocked-by %s\n' "$1" "$2"
  bd dep add "${ID[$1]}" "${ID[$2]}"
}

echo "== Part I -- pdf2md output efficiency (high confidence) =="
mk I-0 task p1 "token-max,part-i,wave-0" \
  "I-0 -- Baseline measurement harness" \
  "Repeatable pdf2md measurement over a fixed corpus (synthetic + local PDFs by path only). $DOC §5. Owns: scripts/." \
  "Two runs on an unchanged tree byte-identical; per-file bytes/lines/nonws-chars/recovery/PASS-FAIL/bare-page-# count/short-paragraph count as JSON|TSV."
mk I-A task p1 "token-max,part-i,wave-1" \
  "I-A -- Strip control characters at the extraction boundary" \
  "Drop U+0000/C0-C1 (keep \\t\\n), expand ligatures, normalize spaces in WordBuilder::push. $DOC §6. Owns: kopitiam-pdf/src/extractor.rs (+ textnorm.rs). Port vendor/pdf-to-markdown textnorm.py." \
  "No NUL/control (besides \\n\\t) in any fixture output; rg (not grep -a) searches every output; recovery ratio unchanged or better."
mk I-E task p2 "token-max,part-i,wave-1" \
  "I-E -- Table detection: emit longest valid prefix instead of bailing" \
  "try_table currently returns None on one ragged row -> whole run becomes one-paragraph-per-cell. Emit the valid prefix as a Table. $DOC §6. Owns: kopitiam-document/src/reconstruction/tables.rs." \
  "5-row table with ragged row 4 -> rows 1-3 render as a table; table_escapes_pipes_and_has_no_column_padding still passes; ratio <= 1.0; report line-count reduction."
mk I-C task p1 "token-max,part-i,wave-2" \
  "I-C -- Strip running heads, footers, bare page numbers" \
  "Top/bottom 10% zones by Page::height; digit-normalized signature; drop recurring + bare-number lines. $DOC §7. Owns: new kopitiam-document/src/reconstruction/headers.rs + mod.rs. Port pdf-to-markdown headers.py." \
  "6-page fixture with running head + footer numbers -> zero remain; 1-page fixture unchanged; ratio honest per §2.1; bare-page-# count ~0 on corpus (report token delta)."
mk I-D task p1 "token-max,part-i,wave-2" \
  "I-D -- Collapse figure regions to captions (highest payoff)" \
  "Heuristic figure-region detection (short-line runs, x-scatter, adjacency to Fig.N, no sentence structure) -> emit caption only. Precision over recall; flag if low-confidence. $DOC §7. Owns: reconstruction/figures.rs,mod.rs + FIGURE_PLACEHOLDER (renderer.rs + validation/mod.rs:16)." \
  "~30 scattered labels + Fig.1 -> caption alone; prose list NOT collapsed; placeholder consistent; ratio honest; >50% line-count reduction on a diagram-heavy fixture."
mk I-B task p2 "token-max,part-i,wave-3" \
  "I-B -- Emit page-boundary anchors + per-page recovery" \
  "Emit <!-- page N --> at boundaries from block_pages; add skip rule to strip_rendered_markdown_syntax; per-page recovery in report; render_document options struct (anchors default on). $DOC §8. Owns: kopitiam-markdown/src/renderer.rs, kopitiam-document/src/validation/{mod,report}.rs." \
  "One anchor/boundary matching block_pages; ratio test for anchor scaffolding; per-page ratios in report; rg locates a page anchor."
mk I-F task p2 "token-max,part-i,wave-4" \
  "I-F -- pdf2md --report-json" \
  "serde-derive ConversionReport (feature-gated dep) + --report-json flag; include per-page ratios if I-B landed. $DOC §9. Owns: apps/cli/src/main.rs (+ mirror tui/convert.rs, §2.2)." \
  "--help documents the flag; a test; existing two-flag invocation identical."
mk I-G task p1 "token-max,part-i,wave-4" \
  "I-G -- Sidecar index (<output>.index.json)" \
  "Emit a sidecar mapping heading->line-range and page->line-range (grep-and-probe -> lookup-then-slice). Sidecar, not in-band (md char count/ratio untouched). $DOC §9. Owns: apps/cli/src/main.rs (+ tui/convert.rs)." \
  "--help documents it; a test; .md unchanged vs no-index; index locates a chapter by heading + by page."
mk I-H task p2 "token-max,part-i,wave-4" \
  "I-H -- --pages A-B and --split-by heading-level N" \
  "Convert only a page range; split a multi-chapter output into per-chapter files (reuse tui/logic.rs naming helpers). $DOC §9. Owns: apps/cli/src/main.rs (+ tui/convert.rs)." \
  "--help documents both; each has a test; existing invocation identical."

echo "== Part II -- agentic coding (medium confidence; survey first) =="
mk II-0 task p1 "token-max,part-ii,survey" \
  "II-0 -- Survey scan/rename/code-actions/status/plan internals (BLOCKING)" \
  "Written survey of rust-analyzer/rustdoc caching, .kopitiam/state.redb schema, LSP client reuse, code-actions position addressing, plan's local-model feed. $DOC §10. Report findings back into the doc. Blocks all Part II." \
  "Survey subsection added to $DOC; each Part-II card re-specified with verified internals; flags any already-built pieces."
mk II-1 feature p1 "token-max,part-ii" \
  "II-1 -- Semantic queries (refs/def/sig/callers/callees/impls)" \
  "Expose rust-analyzer knowledge as coordinate-returning commands (~40 tokens vs thousands of grep+read). --json mandatory. $DOC §11. Highest Part-II value." \
  "Each query returns file:line coordinates (no bodies) as JSON; matches rust-analyzer ground truth on a sample."
mk II-2 feature p2 "token-max,part-ii" \
  "II-2 -- Outline / skeleton mode (outline <file>)" \
  "Items only (fn sigs, struct fields, impl blocks) with line numbers, no bodies. ~10x reduction on orientation. $DOC §11." \
  "outline of a large file lists all items + line numbers, zero bodies; --json."
mk II-3 feature p2 "token-max,part-ii" \
  "II-3 -- Cached architecture digest" \
  "Compact persistent crate -> responsibility/key-types/dep-edges digest in .kopitiam/state.redb; regen on Cargo.toml/source hash change. $DOC §11." \
  "Digest generated once, read cheaply; invalidates on content-hash change, not time."
mk II-4 feature p2 "token-max,part-ii" \
  "II-4 -- Compact diagnostics (check/test --compact)" \
  "One line per DISTINCT diagnostic, deduped, sorted by file, noise stripped; failures as name+assertion+file:line. $DOC §11." \
  "40-diagnostic-from-one-cause output collapses to the distinct set; failures show file:line."
mk II-5 feature p2 "token-max,part-ii" \
  "II-5 -- Persistent conclusion memory" \
  "Record conclusions (invariants, flaky tests) in state.redb; invalidate on content hash, never time; status --stale view. $DOC §11." \
  "Entries store derived-from hashes and drop when they change; status --stale lists stale ones."
mk II-6 feature p2 "token-max,part-ii" \
  "II-6 -- Local-model preprocessing" \
  "Route high-volume low-judgment work to local weights (summarize, triage grep hits, draft commit msgs, classify diagnostics); zero cloud tokens; always report what it dropped. $DOC §11." \
  "A high-volume task runs on the local model with a report of dropped items; never the final authority on correctness."
mk II-7 feature p3 "token-max,part-ii" \
  "II-7 -- Token accounting (tokens <path>)" \
  "BPE-approx token estimate for a file/dir so read-vs-outline is informed (§0.7). $DOC §11." \
  "tokens <path> returns a defensible estimate; --json."
mk II-8 feature p3 "token-max,part-ii" \
  "II-8 -- Deterministic refactors (preview-then-apply)" \
  "Mechanical verifiable transforms like rename: move item + import fixup, extract fn, add derive across types, apply a clippy-fix class. $DOC §11." \
  "Each refactor previews a diff then --apply; behaviour-preserving; matches rename ergonomics."

echo "== Part III -- translation (medium-low confidence) =="
mk III-1 feature p1 "token-max,part-iii" \
  "III-1 -- Port ledger (port status)" \
  "Machine-maintained source->target symbol ledger with status + divergence notes + upstream drift. $DOC §12. Highest Part-III value (mupdf/ + vendored pdf-to-markdown are the port fronts)." \
  "port status shows the ledger; upstream-drift flags source changes since a symbol was ported."
mk III-2 feature p1 "token-max,part-iii" \
  "III-2 -- Differential equivalence harness" \
  "Feed the same input to the reference impl and the Rust port, diff outputs (correctness == identical output). $DOC §12. Biggest translation token saver. Check vendored fixture licences (§2.4); prefer synthetic." \
  "Harness runs reference vs port on inputs and reports diffs; a ported symbol gains cheap regression coverage."
mk III-3 feature p2 "token-max,part-iii" \
  "III-3 -- Skeleton-first translation" \
  "Mechanically generate Rust signatures/stubs from the source structure so the agent writes only bodies; consistent naming. $DOC §12." \
  "Stub generation produces compiling signatures for a source module; agent fills bodies."
mk III-4 feature p2 "token-max,part-iii" \
  "III-4 -- Segment IDs + translation memory" \
  "Stable segment IDs in converted md (or the §9 sidecar); cache translations by content hash; re-translate only changed segments. $DOC §13. Depends on I-B/I-G." \
  "Revised doc re-translates only changed segments; unchanged segments cost zero."
mk III-5 feature p3 "token-max,part-iii" \
  "III-5 -- Terminology glossary enforcement" \
  "Deterministic project glossary application to prevent per-occurrence re-decision + drift. $DOC §13." \
  "Glossary applied deterministically across a document; no per-occurrence model spend."
mk III-6 feature p3 "token-max,part-iii" \
  "III-6 -- Local-first two-pass translation" \
  "Local model drafts every segment; cloud reviews only low-confidence ones (length/glossary/perplexity/round-trip). $DOC §13. Report the split." \
  "A confidence signal routes segments; the local/cloud split is measured and reported; conservative default."
mk III-7 feature p3 "token-max,part-iii" \
  "III-7 -- Bilingual aligned output" \
  "Side-by-side/interleaved source+target with segment anchors so a reviewer checks specific segments. $DOC §13." \
  "Aligned output lets a reviewer verify one segment without reading the whole document."

echo
echo "== Dependency edges (per $DOC §13 Suggested order) =="
# Part I: I-0 -> (I-A, I-E) -> I-C -> I-D -> I-B -> I-F/I-G/I-H
dep I-A I-0
dep I-E I-0
dep I-C I-0
dep I-D I-C
dep I-B I-D
dep I-F I-B
dep I-G I-B
dep I-H I-B
# Part II: II-0 blocks II-1..II-8
for c in II-1 II-2 II-3 II-4 II-5 II-6 II-7 II-8; do dep "$c" II-0; done
# Part III: ledger informs skeleton; segment-memory needs page anchors + sidecar index
dep III-3 III-1
dep III-4 I-B
dep III-4 I-G
dep III-6 III-4
dep III-7 III-4

echo
echo "Done. 26 cards + edges seeded. Review with:  bd list --json  |  bd list -l token-max"
bd list -l token-max 2>/dev/null | head -30 || true

#!/usr/bin/env bash
#
# token-harness.sh -- Task I-0 baseline measurement harness.
# SPDX-License-Identifier: AGPL-3.0-only
#
# The measurement foundation for every Part I claim in kopitiam_token_max.md.
# It converts a FIXED corpus with `kopitiam pdf2md` and reports, per output
# file, a small set of token-relevant numbers so that before/after comparisons
# across a Part I wave are mechanical (§12: "every token-reduction claim must
# be backed by a number from this harness").
#
# Per output file it reports:
#   bytes            output file size in bytes
#   lines            line count
#   nonws_chars      non-whitespace character count
#   extracted_chars  \ parsed from pdf2md's own validation report
#   rendered_chars   /
#   recovery_ratio   rendered_chars / extracted_chars, 4 dp (recomputed)
#   status           PASS / FAIL / ERROR (PASS/FAIL from the report)
#   bare_page_lines  lines matching ^\W*\d{1,4}\W*$ (bare page numbers)
#   short_paras      paragraphs under 5 words (figure-label / fragment soup
#                    proxy); page-anchor-only blocks (<!-- page N -->) are
#                    excluded, as they are fixed scaffolding, not soup.
# Plus a corpus-wide summary row (label __CORPUS__).
#
# DETERMINISM (acceptance §5): two runs on an unchanged tree must produce
# byte-identical output. To that end: LC_ALL=C, rows sorted by label, floats
# printed with a fixed %.4f, and the output carries no timestamps, hostnames,
# or absolute paths (local corpus PDFs appear by basename only).
#
# ---------------------------------------------------------------------------
# USAGE
#   scripts/token-harness.sh              # TSV to stdout (default)
#   scripts/token-harness.sh --json       # JSON to stdout
#   scripts/token-harness.sh --tsv        # explicit TSV
#
# It builds ONLY the `kopitiam` package (`cargo build -q -p kopitiam`), never
# the workspace, so an in-flight fork of crates/kopitiam-bds (which `kopitiam`
# does not depend on) cannot break it.
#
# CORPUS
#   1. Synthetic fixtures in scripts/token-harness/fixtures/*.pdf (committed;
#      synthetic per §2.4). Regenerate them with:
#        rustc -O scripts/token-harness/gen_fixtures.rs -o /tmp/gen && \
#          /tmp/gen scripts/token-harness/fixtures
#   2. OPTIONAL local PDFs by PATH ONLY (never copied into the repo, §2.4):
#      - env var KOPITIAM_TOKEN_CORPUS: paths separated by ';' or newlines, or
#      - scripts/token-harness.corpus: one path per line, '#' comments allowed.
#      Missing paths are skipped silently (degrade gracefully). See
#      scripts/token-harness.corpus.example for the format.
# ---------------------------------------------------------------------------

set -euo pipefail
export LC_ALL=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FIX_DIR="$SCRIPT_DIR/token-harness/fixtures"
CORPUS_FILE="$SCRIPT_DIR/token-harness.corpus"

FORMAT=tsv
for arg in "$@"; do
  case "$arg" in
    --json) FORMAT=json ;;
    --tsv) FORMAT=tsv ;;
    -h|--help) sed -n '2,45p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

# --- Build only the kopitiam package (never --workspace). -------------------
echo "token-harness: building kopitiam (-p kopitiam) ..." >&2
( cd "$REPO_ROOT" && cargo build -q -p kopitiam ) >&2

BIN="$REPO_ROOT/target/debug/kopitiam"
if [ -x "$BIN.exe" ]; then
  BIN="$BIN.exe"
elif [ ! -x "$BIN" ]; then
  echo "token-harness: kopitiam binary not found at $BIN(.exe)" >&2
  exit 1
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# --- Assemble the corpus: (path, label) pairs. ------------------------------
# Synthetic fixtures first, sorted; then optional local PDFs by path.
declare -a PATHS=() LABELS=()

add_path() {
  # $1 = path. Skip silently if it is not an existing file (graceful degrade).
  local p="$1"
  [ -n "$p" ] || return 0
  [ -f "$p" ] || return 0
  PATHS+=("$p")
  LABELS+=("$(basename "$p")")
}

if [ -d "$FIX_DIR" ]; then
  while IFS= read -r f; do
    add_path "$f"
  done < <(find "$FIX_DIR" -maxdepth 1 -type f -name '*.pdf' | LC_ALL=C sort)
fi

# Local PDFs from the env var (';' or newline separated).
if [ -n "${KOPITIAM_TOKEN_CORPUS:-}" ]; then
  while IFS= read -r line; do
    line="${line#"${line%%[![:space:]]*}"}"   # ltrim
    line="${line%"${line##*[![:space:]]}"}"   # rtrim
    add_path "$line"
  done < <(printf '%s' "$KOPITIAM_TOKEN_CORPUS" | tr ';' '\n')
fi

# Local PDFs from the corpus file (one path per line, '#' comments allowed).
if [ -f "$CORPUS_FILE" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%#*}"                         # strip comment
    line="${line#"${line%%[![:space:]]*}"}"   # ltrim
    line="${line%"${line##*[![:space:]]}"}"   # rtrim
    add_path "$line"
  done < "$CORPUS_FILE"
fi

# --- gawk metric extractor over a rendered .md file. ------------------------
# Prints: "<lines> <nonws_chars> <bare_page_lines> <short_paras>".
# A "paragraph" is a run of non-blank lines delimited by blank lines; blocks
# whose every line is an HTML page anchor (<!-- ... -->) are not paragraphs.
md_metrics() {
  gawk '
    function is_blank(s) { return s ~ /^[ \t\r]*$/ }
    function flush() {
      if (block_has_content && !block_all_anchor && block_words > 0 && block_words < 5)
        short++
      block_has_content = 0; block_all_anchor = 1; block_words = 0
    }
    BEGIN { lines = 0; nws = 0; bare = 0; short = 0
            block_has_content = 0; block_all_anchor = 1; block_words = 0 }
    {
      lines++
      line = $0; sub(/\r$/, "", line)
      tmp = line; gsub(/[ \t]/, "", tmp); nws += length(tmp)
      if (line ~ /^[^A-Za-z0-9_]*[0-9]{1,4}[^A-Za-z0-9_]*$/) bare++
      if (is_blank(line)) { flush(); next }
      block_has_content = 1
      if (line !~ /^<!--.*-->[ \t]*$/) block_all_anchor = 0
      n = split(line, a, /[ \t]+/)
      for (i = 1; i <= n; i++) if (a[i] != "") block_words++
    }
    END { flush(); printf "%d %d %d %d\n", lines, nws, bare, short }
  ' "$1"
}

# --- Convert + measure each corpus entry, building sorted TSV data rows. -----
declare -a ROWS=()
sum_bytes=0; sum_lines=0; sum_nws=0; sum_ext=0; sum_ren=0; sum_bare=0; sum_short=0

idx=0
count=${#PATHS[@]}
while [ "$idx" -lt "$count" ]; do
  path="${PATHS[$idx]}"
  label="${LABELS[$idx]}"
  out="$WORK/$idx.md"
  rep="$WORK/$idx.report"

  if "$BIN" pdf2md "$path" -o "$out" >"$rep" 2>"$WORK/$idx.err"; then
    bytes=$(wc -c < "$out" | tr -d '[:space:]')
    read -r lines nws bare short < <(md_metrics "$out")
    ext=$(grep -oE 'Extracted:[[:space:]]+[0-9]+ chars' "$rep" | grep -oE '[0-9]+' | head -1)
    ren=$(grep -oE 'Rendered:[[:space:]]+[0-9]+ chars' "$rep" | grep -oE '[0-9]+' | head -1)
    status=$(grep -E '^Status:' "$rep" | awk '{print $2}')
    ext=${ext:-0}; ren=${ren:-0}; status=${status:-ERROR}
    ratio=$(awk -v e="$ext" -v r="$ren" 'BEGIN { if (e+0==0) printf "%.4f", 1.0; else printf "%.4f", r/e }')
    sum_bytes=$((sum_bytes + bytes)); sum_lines=$((sum_lines + lines))
    sum_nws=$((sum_nws + nws)); sum_ext=$((sum_ext + ext)); sum_ren=$((sum_ren + ren))
    sum_bare=$((sum_bare + bare)); sum_short=$((sum_short + short))
  else
    # Graceful: a PDF that fails to convert is reported, not fatal.
    bytes=0; lines=0; nws=0; ext=0; ren=0; bare=0; short=0
    ratio="0.0000"; status="ERROR"
  fi

  ROWS+=("$(printf '%s\t%d\t%d\t%d\t%d\t%d\t%s\t%s\t%d\t%d' \
    "$label" "$bytes" "$lines" "$nws" "$ext" "$ren" "$ratio" "$status" "$bare" "$short")")
  idx=$((idx + 1))
done

# Corpus-wide summary.
corpus_ratio=$(awk -v e="$sum_ext" -v r="$sum_ren" 'BEGIN { if (e+0==0) printf "%.4f", 1.0; else printf "%.4f", r/e }')
corpus_status=$(awk -v ratio="$corpus_ratio" 'BEGIN { print (ratio+0 >= 0.98 ? "PASS" : "FAIL") }')

# --- Emit, sorted by label for determinism. ---------------------------------
HEADER="file	bytes	lines	nonws_chars	extracted_chars	rendered_chars	recovery_ratio	status	bare_page_lines	short_paras"

if [ "$FORMAT" = "tsv" ]; then
  printf '%s\n' "$HEADER"
  if [ "${#ROWS[@]}" -gt 0 ]; then
    printf '%s\n' "${ROWS[@]}" | LC_ALL=C sort -t $'\t' -k1,1
  fi
  printf '%s\t%d\t%d\t%d\t%d\t%d\t%s\t%s\t%d\t%d\n' \
    "__CORPUS__" "$sum_bytes" "$sum_lines" "$sum_nws" "$sum_ext" "$sum_ren" \
    "$corpus_ratio" "$corpus_status" "$sum_bare" "$sum_short"
else
  # JSON: sorted files array + summary object.
  emit_json_row() {
    # $1..$10 columns; prints one object.
    printf '    {"file":"%s","bytes":%d,"lines":%d,"nonws_chars":%d,"extracted_chars":%d,"rendered_chars":%d,"recovery_ratio":%s,"status":"%s","bare_page_lines":%d,"short_paras":%d}' \
      "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}"
  }
  printf '{\n  "files": [\n'
  if [ "${#ROWS[@]}" -gt 0 ]; then
    mapfile -t SORTED < <(printf '%s\n' "${ROWS[@]}" | LC_ALL=C sort -t $'\t' -k1,1)
    n=${#SORTED[@]}
    for i in "${!SORTED[@]}"; do
      IFS=$'\t' read -r c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 <<< "${SORTED[$i]}"
      emit_json_row "$c1" "$c2" "$c3" "$c4" "$c5" "$c6" "$c7" "$c8" "$c9" "$c10"
      if [ "$((i + 1))" -lt "$n" ]; then printf ',\n'; else printf '\n'; fi
    done
  fi
  printf '  ],\n  "summary": '
  emit_json_row "__CORPUS__" "$sum_bytes" "$sum_lines" "$sum_nws" "$sum_ext" \
    "$sum_ren" "$corpus_ratio" "$corpus_status" "$sum_bare" "$sum_short" | sed 's/^    //'
  printf '\n}\n'
fi

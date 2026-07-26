#!/usr/bin/env bash
#
# diff-harness.sh -- token-max card III-2 (differential equivalence harness)
#                    AND Task #8 (golden-oracle regression set), one tool.
# SPDX-License-Identifier: AGPL-3.0-only
#
# These two cards are the same regression/reference machinery, so they live in
# one harness with two modes:
#
#   Mode A -- Golden-oracle regression (Task #8; RUNS here).
#     For each synthetic fixture the harness runs `kopitiam pdf2md` and diffs the
#     fresh output against a committed golden snapshot under scripts/golden/,
#     failing on any byte-level drift. It snapshots BOTH the rendered `.md` and
#     the `--report-json` (recovery ratio etc.) per fixture, and asserts the
#     recovery ratio stays in [0.98, 1.0] with passes=true. This is a whole-
#     pipeline regression guard across the four format archetypes.
#
#   Mode B -- Differential vs reference (card III-2; --diff-reference).
#     The III-2 core: feed the same input to the reference implementation and
#     the Rust port, then diff -- replacing "an agent reads both implementations
#     and reasons about equivalence" with "run the harness" (kopitiam_token_max
#     .md §0.3). Focused on the FAITHFUL port `textnorm.py -> textnorm.rs`
#     (docs/port-ledger.json). Because the live Python reference is unavailable
#     on this host (the `python` here is the Windows Store stub), the external
#     cross-language diff is GATED: it runs where a real python3 exists and
#     SKIPS with a documented run-elsewhere command otherwise. An in-process
#     Rust-only stand-in (the crate's textnorm assertions + a control-char scan
#     of every real fixture output) always runs here.
#
# DETERMINISM (§12): with an unchanged tree and unchanged goldens, two runs
# produce byte-identical harness output and the golden `.md`/`.report.json`
# snapshots are stable (pdf2md is deterministic). To that end: LC_ALL=C, fixtures
# processed in sorted order, floats taken verbatim from pdf2md's own JSON, and
# the output carries no timestamps, hostnames, or absolute paths (fixtures and
# goldens appear by basename / repo-relative path only).
#
# The committed goldens are snapshots of SYNTHETIC fixtures only, so committing
# them is allowed (§2.4: never commit oracles of copyrighted papers). A
# scripts/golden/.gitattributes pins them as `-text` so autocrlf (this host has
# core.autocrlf=true) can never rewrite LF->CRLF and desync them from pdf2md's
# LF output on checkout.
#
# ---------------------------------------------------------------------------
# USAGE
#   scripts/diff-harness.sh                 # Mode A regression check (default)
#   scripts/diff-harness.sh --bless         # (re)generate the golden snapshots
#   scripts/diff-harness.sh --report        # Mode A as machine-readable JSON
#   scripts/diff-harness.sh --diff-reference# Mode B differential vs reference
#   scripts/diff-harness.sh --help
#
# Default / --report bless the goldens on first run when they are missing, then
# diff on every run after. --bless always (re)writes them. Exit is nonzero on any
# drift, out-of-band ratio, failed report, or dirty (control-char) output.
#
# It builds ONLY the `kopitiam` package (`cargo build -q -p kopitiam`), never the
# workspace, and reuses that one binary for every fixture (same convention as
# scripts/token-harness.sh).
#
# CORPUS (reused, never duplicated): the token-harness synthetic fixtures in
#   scripts/token-harness/fixtures/*.pdf -- the four format archetypes
#   (a_two_column, b_figure, c_ragged_table, d_prose).
# ---------------------------------------------------------------------------

set -euo pipefail
export LC_ALL=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FIX_DIR="$SCRIPT_DIR/token-harness/fixtures"
GOLDEN_DIR="$SCRIPT_DIR/golden"
LEDGER="$REPO_ROOT/docs/port-ledger.json"
VENDOR_TEXTNORM_DIR="$REPO_ROOT/crates/kopitiam-document/vendor/pdf-to-markdown"

MODE=check
for arg in "$@"; do
  case "$arg" in
    --bless|--update)  MODE=bless ;;
    --report)          MODE=report ;;
    --diff-reference)  MODE=diffref ;;
    -h|--help) sed -n '2,72p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "diff-harness: unknown argument: $arg (try --help)" >&2; exit 2 ;;
  esac
done

# --- Build only the kopitiam package (never --workspace). -------------------
echo "diff-harness: building kopitiam (-p kopitiam) ..." >&2
( cd "$REPO_ROOT" && cargo build -q -p kopitiam ) >&2

BIN="$REPO_ROOT/target/debug/kopitiam"
if [ -x "$BIN.exe" ]; then
  BIN="$BIN.exe"
elif [ ! -x "$BIN" ]; then
  echo "diff-harness: kopitiam binary not found at $BIN(.exe)" >&2
  exit 1
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# --- Collect the synthetic fixtures, sorted (determinism). ------------------
declare -a FIX_PATHS=() FIX_NAMES=()
if [ -d "$FIX_DIR" ]; then
  while IFS= read -r f; do
    FIX_PATHS+=("$f")
    FIX_NAMES+=("$(basename "$f" .pdf)")
  done < <(find "$FIX_DIR" -maxdepth 1 -type f -name '*.pdf' | LC_ALL=C sort)
fi
if [ "${#FIX_PATHS[@]}" -eq 0 ]; then
  echo "diff-harness: no fixtures found in $FIX_DIR" >&2
  exit 1
fi

# --- Helpers ----------------------------------------------------------------

# Run pdf2md once, capturing BOTH the rendered .md (-o) and the --report-json
# (stdout). Default engine (mupdf) -- what an ordinary user gets. "Wrote ..."
# notices go to stderr in --report-json mode, so stdout is clean JSON.
run_pdf2md() { # $1 input.pdf  $2 out.md  $3 out.report.json
  "$BIN" pdf2md "$1" -o "$2" --report-json >"$3" 2>/dev/null
}

# Top-level recovery_ratio from a report JSON (2-space indent = document level;
# the 6-space-indented per_page ratios are deliberately not matched).
report_ratio() { grep -E '^  "recovery_ratio":' "$1" | grep -oE '[0-9]+\.[0-9]+|[0-9]+' | head -1; }
report_passes() { grep -E '^  "passes":' "$1" | grep -oE 'true|false' | head -1; }

# A recovery ratio is in band iff 0.98 <= r <= 1.0 (a ratio above 1.0 is a
# failure per §12: emitted scaffolding masking content loss).
ratio_in_band() { awk -v r="$1" 'BEGIN { exit !(r+0 >= 0.98 && r+0 <= 1.0 + 1e-9) }'; }

# True (rc 0) iff the .md is free of forbidden control characters: any C0/C1
# control or DEL, plus stray CR, but ALLOWING tab (0x09) and newline (0x0A).
# This is the pipeline-level textnorm guarantee (Task I-A) checked on real
# fixture output -- a NUL here would make ripgrep treat the file as binary.
control_clean() { # $1 md
  local leftover
  leftover="$(LC_ALL=C tr -d '\011\012' <"$1" | LC_ALL=C tr -cd '[:cntrl:]' | wc -c | tr -d '[:space:]')"
  [ "$leftover" = "0" ]
}

# --- Mode A: golden-oracle regression --------------------------------------
# Populates parallel arrays; caller renders as table (default) or JSON (--report).
declare -a R_NAME=() R_MD=() R_REPORT=() R_RATIO=() R_PASSES=() R_CLEAN=() R_STATUS=()
FAILED=0

run_mode_a() { # $1 = "bless" to force-write goldens, else diff-or-bootstrap
  local force="${1:-}"
  mkdir -p "$GOLDEN_DIR"

  local i
  for i in "${!FIX_PATHS[@]}"; do
    local name="${FIX_NAMES[$i]}" path="${FIX_PATHS[$i]}"
    local fresh_md="$WORK/$name.md" fresh_rep="$WORK/$name.report.json"
    local gold_md="$GOLDEN_DIR/$name.md" gold_rep="$GOLDEN_DIR/$name.report.json"

    run_pdf2md "$path" "$fresh_md" "$fresh_rep"

    local ratio passes clean md_st rep_st status
    ratio="$(report_ratio "$fresh_rep")"; ratio="${ratio:-0}"
    passes="$(report_passes "$fresh_rep")"; passes="${passes:-false}"
    if control_clean "$fresh_md"; then clean=clean; else clean=DIRTY; fi

    if [ "$force" = "bless" ] || [ ! -f "$gold_md" ] || [ ! -f "$gold_rep" ]; then
      cp -f "$fresh_md" "$gold_md"
      cp -f "$fresh_rep" "$gold_rep"
      md_st=$([ "$force" = "bless" ] && echo blessed || echo new)
      rep_st="$md_st"
      status=BLESSED
    else
      if cmp -s "$fresh_md" "$gold_md"; then md_st=match; else md_st=DRIFT; fi
      if cmp -s "$fresh_rep" "$gold_rep"; then rep_st=match; else rep_st=DRIFT; fi
      status=OK
      [ "$md_st" = DRIFT ] && status=FAIL
      [ "$rep_st" = DRIFT ] && status=FAIL
    fi

    # Ratio-band and control-clean gates apply in every mode.
    ratio_in_band "$ratio" || status=FAIL
    [ "$passes" = true ] || status=FAIL
    [ "$clean" = clean ] || status=FAIL
    [ "$status" = FAIL ] && FAILED=1

    R_NAME+=("$name"); R_MD+=("$md_st"); R_REPORT+=("$rep_st")
    R_RATIO+=("$ratio"); R_PASSES+=("$passes"); R_CLEAN+=("$clean"); R_STATUS+=("$status")
  done
}

emit_table() { # human-readable, deterministic
  local title="$1"
  printf '%s\n' "$title"
  printf '%s\n' "-------------------------------------------------------------------------------"
  printf '%-16s %-7s %-8s %-8s %-7s %-7s %s\n' \
    fixture md report ratio passes clean status
  local i
  for i in "${!R_NAME[@]}"; do
    printf '%-16s %-7s %-8s %-8s %-7s %-7s %s\n' \
      "${R_NAME[$i]}" "${R_MD[$i]}" "${R_REPORT[$i]}" "${R_RATIO[$i]}" \
      "${R_PASSES[$i]}" "${R_CLEAN[$i]}" "${R_STATUS[$i]}"
  done
  printf '%s\n' "-------------------------------------------------------------------------------"
  if [ "$FAILED" -eq 0 ]; then
    printf 'result: OK  (%d/%d fixtures match goldens; ratios in [0.98,1.0]; outputs control-clean)\n' \
      "${#R_NAME[@]}" "${#R_NAME[@]}"
  else
    printf 'result: FAIL  (drift, out-of-band ratio, or dirty output above -- re-bless deliberately with --bless if the change is intended)\n'
  fi
}

emit_report_json() { # machine-readable, deterministic
  printf '{\n  "mode": "golden-regression",\n  "fixtures": [\n'
  local i n=${#R_NAME[@]}
  for i in "${!R_NAME[@]}"; do
    printf '    {"fixture":"%s","md":"%s","report":"%s","recovery_ratio":%s,"passes":%s,"control_clean":%s,"status":"%s"}' \
      "${R_NAME[$i]}" "${R_MD[$i]}" "${R_REPORT[$i]}" "${R_RATIO[$i]}" "${R_PASSES[$i]}" \
      "$([ "${R_CLEAN[$i]}" = clean ] && echo true || echo false)" "${R_STATUS[$i]}"
    if [ "$((i + 1))" -lt "$n" ]; then printf ',\n'; else printf '\n'; fi
  done
  printf '  ],\n  "ok": %s\n}\n' "$([ "$FAILED" -eq 0 ] && echo true || echo false)"
}

# --- Mode B: differential vs reference -------------------------------------

# The shared textnorm contract as crafted (input, expected) pairs. Inputs use
# only the rules on which the port is FAITHFUL, deliberately excluding the two
# documented divergences of textnorm.rs (newline preservation, and category Cn),
# so the reference (Python) and the Rust port must produce identical output.
# Encoding: TAB in a value is written as the two chars '\t' on both sides before
# comparison, so every cell is printable.
declare -a TN_IN=() TN_EXP=()
tn_case() { TN_IN+=("$1"); TN_EXP+=("$2"); }
# Invisible/exotic chars are written as bash $'\xHH' UTF-8 BYTE escapes so this
# source stays pure ASCII (visible ligatures/CJK are kept literal). No real NUL:
# bash and env strings truncate at a NUL byte, so NUL coverage lives in [B1] (the
# Rust textnorm tests) and [B2] (the fixture control-char scan), not here.
#         input                                          expected            rule
tn_case $'ﬀﬁﬂﬃﬄﬅﬆ'                                     'fffiflffifflstst' # ligatures
tn_case $'a\xC2\xA0\xE2\x80\x89b'                       'a  b'             # exotic spaces (NBSP,U+2009: both in Python _SPACES)
tn_case $'a\x01\x1Fb'                                   'ab'               # C0 controls
tn_case $'a\tb'                                         'a\tb'             # TAB preserved
tn_case $'a\xC2\x85b'                                   'ab'               # C1 (NEL)
tn_case $'a\xEF\xBB\xBFb'                               'ab'               # BOM (U+FEFF)
tn_case $'a\xE2\x80\x8Bb'                               'ab'               # ZWSP (U+200B)
tn_case $'a\xEE\x80\x80b'                               'ab'               # private use (U+E000)
tn_case $'世界'                                         '世界'            # CJK passes through
tn_case 'cafe resume'                                   'cafe resume'      # ordinary ASCII

# escape a raw string's TAB to the literal two chars \t (for printable diffs)
esc_tab() { printf '%s' "$1" | sed 's/\t/\\t/g'; }

find_python() {
  # The Windows Store stub prints a "not found" banner and yields NO stdout for
  # `-c`, while a real interpreter prints its major version. Detect by output,
  # never by exit code (the stub exits 0). KOPITIAM_PYTHON overrides.
  local cand out
  for cand in "${KOPITIAM_PYTHON:-}" python3 python; do
    [ -n "$cand" ] || continue
    command -v "$cand" >/dev/null 2>&1 || continue
    out="$("$cand" -c 'import sys; sys.stdout.write(str(sys.version_info[0]))' 2>/dev/null || true)"
    case "$out" in 2|3) printf '%s' "$cand"; return 0 ;; esac
  done
  return 1
}

# Print the faithful-vs-diverged pdf-to-markdown port map, straight from the
# ledger so it stays honest (docs/port-ledger.json is the source of truth).
emit_port_map() {
  printf 'Port map (pdf-to-markdown -> Rust), from docs/port-ledger.json:\n'
  if [ -f "$LEDGER" ]; then
    awk -F'"' '
      /"upstream": "pdf-to-markdown"/ && /"target": "crates/ {
        src=""; tgt=""; st=""
        for (i = 1; i <= NF; i++) {
          if ($i == "source") src = $(i + 2)
          if ($i == "target") tgt = $(i + 2)
          if ($i == "status") st  = $(i + 2)
        }
        note = (st == "ported") ? "FAITHFUL  (differential is meaningful)" \
             : (st == "deliberately-diverged") ? "DIVERGED  (output NOT expected to match, by design)" \
             : st
        printf "  %-18s -> %-52s %s\n", src, tgt, note
      }' "$LEDGER" | LC_ALL=C sort
  else
    printf '  (ledger not found at docs/port-ledger.json)\n'
  fi
  printf '\n'
  printf 'The differential below targets the FAITHFUL port only (textnorm). The\n'
  printf 'diverged ports (headers.py, regions.py) are clean-room adaptations whose\n'
  printf 'output is expected to differ from the Python -- diffing them is not a bug.\n\n'
}

run_mode_b() {
  local b_failed=0
  emit_port_map

  # (B1) In-process Rust-only stand-in: the crate's textnorm assertions encode
  # exactly this crafted-input -> expected contract for the Rust port. Green =
  # textnorm.rs matches the documented normalization. Runs here.
  printf '[B1] Rust textnorm oracle (in-process, runs here)\n'
  local tn_out tn_rc
  tn_out="$( ( cd "$REPO_ROOT" && cargo test -q -p kopitiam-pdf --lib textnorm ) 2>/dev/null )" && tn_rc=0 || tn_rc=$?
  local passed
  passed="$(printf '%s\n' "$tn_out" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' | head -1)"
  passed="${passed:-0}"
  if [ "${tn_rc:-1}" -eq 0 ] && [ "$passed" -gt 0 ]; then
    printf '  PASS  cargo test -p kopitiam-pdf --lib textnorm (%s assertions)\n\n' "$passed"
  else
    printf '  FAIL  cargo test -p kopitiam-pdf --lib textnorm\n\n'
    b_failed=1
  fi

  # (B2) Pipeline textnorm on real fixtures: every golden/fresh .md must be
  # control-char clean. Runs here.
  printf '[B2] Fixture output control-char scan (in-process, runs here)\n'
  local i any_dirty=0
  for i in "${!FIX_PATHS[@]}"; do
    local name="${FIX_NAMES[$i]}" md="$WORK/${FIX_NAMES[$i]}.md"
    run_pdf2md "${FIX_PATHS[$i]}" "$md" "$WORK/$name.rep"
    if control_clean "$md"; then
      printf '  clean  %s.md\n' "$name"
    else
      printf '  DIRTY  %s.md\n' "$name"; any_dirty=1; b_failed=1
    fi
  done
  printf '\n'
  [ "$any_dirty" -eq 0 ] || true

  # (B3) External cross-language differential: Python textnorm.py vs the shared
  # contract. Gated on a usable python3 (absent on this host).
  printf '[B3] Reference cross-language differential (textnorm.py vs Rust port)\n'
  local py
  if py="$(find_python)"; then
    printf '  reference interpreter: %s\n' "$py"
    local diff_seen=0 j
    for j in "${!TN_IN[@]}"; do
      # Apply the vendored normalize_char per character (matches Rust
      # normalize_into), then compare to the shared expected value.
      local got
      got="$(
        KOPITIAM_TN_INPUT="${TN_IN[$j]}" "$py" - "$VENDOR_TEXTNORM_DIR" <<'PY' 2>/dev/null || true
import os, sys
sys.path.insert(0, sys.argv[1])
from pdf2md.textnorm import normalize_char
s = os.environ.get("KOPITIAM_TN_INPUT", "")
sys.stdout.write("".join(normalize_char(ch) for ch in s))
PY
      )"
      local got_e exp_e
      got_e="$(esc_tab "$got")"; exp_e="${TN_EXP[$j]}"
      if [ "$got_e" = "$exp_e" ]; then
        printf '  ok    case %02d  -> %s\n' "$((j + 1))" "$exp_e"
      else
        printf '  DIFF  case %02d  expected[%s] got[%s]\n' "$((j + 1))" "$exp_e" "$got_e"
        diff_seen=1; b_failed=1
      fi
    done
    [ "$diff_seen" -eq 0 ] && printf '  reference matches the Rust port on all %d cases.\n' "${#TN_IN[@]}"
  else
    printf '  SKIP  no usable python3 on this host (the `python` here is the\n'
    printf '        Windows Store stub: it prints a "not found" banner and yields\n'
    printf '        no output, so the live reference side cannot run).\n'
    printf '  To run the cross-language differential where a real Python 3 exists:\n'
    printf '        # from the repo root, on a host with python3 (or set KOPITIAM_PYTHON):\n'
    printf '        KOPITIAM_PYTHON=python3 scripts/diff-harness.sh --diff-reference\n'
    printf '        # it imports crates/kopitiam-document/vendor/pdf-to-markdown/pdf2md/textnorm.py\n'
    printf '        # and diffs normalize_char output against the shared contract (and the\n'
    printf '        # Rust oracle in [B1]); a three-way agreement proves textnorm.rs == textnorm.py.\n'
    printf '  (This is a SKIP, not a failure: Mode B degrades gracefully -- §12.)\n'
  fi
  printf '\n'

  if [ "$b_failed" -eq 0 ]; then
    printf 'Mode B result: OK  (Rust oracle green; fixtures clean; reference side ran or skipped cleanly)\n'
  else
    printf 'Mode B result: FAIL\n'
    return 1
  fi
}

# --- Dispatch ---------------------------------------------------------------
case "$MODE" in
  check)
    run_mode_a ""
    emit_table "diff-harness: Mode A -- golden-oracle regression"
    [ "$FAILED" -eq 0 ]
    ;;
  bless)
    run_mode_a bless
    emit_table "diff-harness: Mode A -- goldens (re)blessed"
    # A bless run writes fresh goldens; the ratio/clean gates still apply.
    [ "$FAILED" -eq 0 ]
    ;;
  report)
    run_mode_a ""
    emit_report_json
    [ "$FAILED" -eq 0 ]
    ;;
  diffref)
    printf 'diff-harness: Mode B -- differential vs reference (card III-2)\n'
    printf '=============================================================\n\n'
    run_mode_b
    ;;
esac

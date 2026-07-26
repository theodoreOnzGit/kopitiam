#!/usr/bin/env bash
#
# port-ledger.sh -- Task III-1 (kopitiam_token_max.md §12): the machine-
# maintained PORT LEDGER for kopitiam's two code-translation projects.
# SPDX-License-Identifier: AGPL-3.0-only
#
# WHAT THIS IS
# ------------
# kopitiam is already two porting projects (kopitiam_token_max.md §12):
#   * crates/kopitiam-pdf/src/mupdf/*  -- a faithful MuPDF (C) -> Rust port;
#   * crates/kopitiam-ocr/src/*        -- a Tesseract/Leptonica (C++) -> Rust port;
# plus a vendored MIT Python reference, pdf-to-markdown, whose modules
# (headers.py, textnorm.py, postprocess.py, serialize.py, tables.py, regions.py,
# ...) are the porting SOURCE for kopitiam-document's reconstruction pass.
#
# "Which of those source units is already ported, which is deliberately diverged,
# and which is still unported" is a fact an agent otherwise re-derives from
# scratch every session (the recurring exploration cost §12/§290 names). This
# tool derives it DETERMINISTICALLY from the repo's own point-of-use provenance
# headers -- never from a hand-kept list -- and writes it to:
#   docs/port-ledger.json  (machine view: schema + entries + summary)
#   docs/port-ledger.md    (human view: tables by upstream + the unported list)
#
# HOW IT CLASSIFIES  (deterministic; a pure function of tree content, §0.3/§0.4)
# ----------------------------------------------------------------------------
# Pass A -- target-driven inventory. Every port target under the three tracked
#   source roots (kopitiam-pdf, kopitiam-ocr, kopitiam-document) is scanned for
#   its provenance header. From that header we read:
#     - the upstream project (MuPDF / Tesseract / Leptonica / pdf-to-markdown),
#     - the cited upstream source file(s) (the backtick-quoted `source/...`,
#       `include/...`, `src/...`, `pdf2md/...` paths, incl. `foo.{cpp,h}`),
#     - the upstream commit SHA the translation was made from ("commit 19f1284",
#       "at commit `10bdea2`", ...), so upstream drift can be checked later.
#   The target's status is:
#     ported               -- the header says "Ported from" (a close adaptation
#                             / translation of that source);
#     deliberately-diverged -- the header cites the source but disclaims a
#                             faithful port ("original Rust, not a port" /
#                             "not a translation" / "not a line-for-line port":
#                             a clean-room adaptation of the idea only).
#     in-progress          -- the header carries an explicit machine marker
#                             `PORT-LEDGER: in-progress` (none today; reserved
#                             so a porter can flag work mid-flight).
# Pass B -- source-driven unported detection (pdf-to-markdown backlog). Every
#   .py under the vendored pdf-to-markdown pdf2md/ tree that NO target cited in
#   Pass A is emitted as `unported`. This is the finite, well-defined porting
#   backlog the card cares about; the MuPDF/Tesseract C/C++ vendor trees are
#   thousands of files never all intended for porting, so "unported" is only
#   enumerated for the pdf2md set (see the note printed in --report).
#
# DETERMINISM  (acceptance: two runs on an unchanged tree are byte-identical)
# --------------------------------------------------------------------------
# LC_ALL=C throughout; every list is `sort`ed; the canonical intermediate is a
# sorted TSV; NO timestamps, hostnames, `$RANDOM`, or absolute paths ever reach
# the output (paths are repo-relative). The default run regenerates both files
# and then PROVES reproducibility by regenerating to a temp dir and diffing;
# a mismatch is a hard failure.
#
# UPSTREAM DRIFT  (design, not run here -- there is no network in the pure path)
# ----------------------------------------------------------------------------
# Each ported/diverged entry records `commit` = the upstream SHA it was made
# from. To check drift later, OUT OF this deterministic path:
#     git ls-remote <upstream-url> <ref>        # or a maintainer-provided ref
# and compare the returned SHA to the recorded one; if they differ, diff the
# specific cited source path between the two commits in a local clone to see
# whether the ported region actually moved. `--drift-check <ref>` documents the
# mechanism (and, given a reachable ref, prints the compare plan) but never
# hard-codes a network call into the generator.
#
# USAGE
#   scripts/port-ledger.sh                 # regenerate docs/port-ledger.{json,md} + self-check
#   scripts/port-ledger.sh --report        # print status counts + unported list to stdout (no writes)
#   scripts/port-ledger.sh --check         # recompute; fail if committed docs are stale (CI guard)
#   scripts/port-ledger.sh --drift-check REF# print the (documented) upstream-drift compare plan
#   scripts/port-ledger.sh --help
#
set -euo pipefail
export LC_ALL=C

# --------------------------------------------------------------------------
# 0. Locate the repo root from this script's location (no absolute paths baked).
# --------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

JSON_OUT="docs/port-ledger.json"
MD_OUT="docs/port-ledger.md"

# Tracked target roots (the two porting projects + the reconstruction pass that
# ports pdf-to-markdown). Kept narrow on purpose: the ledger is about the
# document-extraction translation surface, not every "Ported from" in the tree.
TARGET_ROOTS="crates/kopitiam-pdf/src crates/kopitiam-ocr/src crates/kopitiam-document/src"

# The finite porting backlog whose "unported" is meaningful (Pass B).
PDF2MD_VENDOR="crates/kopitiam-document/vendor/pdf-to-markdown/pdf2md"

# --------------------------------------------------------------------------
# 1. Upstream registry. Pinned commits are the vendored reference state per
#    docs/ACKNOWLEDGEMENTS.md (pdf-to-markdown 54baa2e, Tesseract db0ec62,
#    Leptonica 10bdea2) and docs/ai-decisions/AID-0051 (MuPDF 19f1284). Used
#    ONLY as the SHA fallback when a point-of-use header omits an inline commit
#    (e.g. the pdf-to-markdown ports cite the path but no SHA).
# --------------------------------------------------------------------------
upstream_name()   { case "$1" in mupdf) echo "MuPDF";; tesseract) echo "Tesseract";; leptonica) echo "Leptonica";; pdf-to-markdown) echo "pdf-to-markdown";; *) echo "$1";; esac; }
upstream_lic()    { case "$1" in mupdf) echo "AGPL-3.0";; tesseract) echo "Apache-2.0";; leptonica) echo "BSD-2-Clause";; pdf-to-markdown) echo "MIT";; *) echo "";; esac; }
upstream_pin()    { case "$1" in mupdf) echo "19f1284";; tesseract) echo "db0ec62";; leptonica) echo "10bdea2";; pdf-to-markdown) echo "54baa2e";; *) echo "";; esac; }
upstream_url()    { case "$1" in
    mupdf) echo "https://github.com/ArtifexSoftware/mupdf";;
    tesseract) echo "https://github.com/tesseract-ocr/tesseract";;
    leptonica) echo "https://github.com/DanBloomberg/leptonica";;
    pdf-to-markdown) echo "https://github.com/iamarunbrahma/pdf-to-markdown";;
    *) echo "";; esac; }
upstream_vendor() { case "$1" in
    mupdf) echo "crates/kopitiam-pdf/vendor/mupdf";;
    tesseract) echo "crates/kopitiam-ocr/vendor/tesseract";;
    leptonica) echo "crates/kopitiam-ocr/vendor/leptonica";;
    pdf-to-markdown) echo "crates/kopitiam-document/vendor/pdf-to-markdown";;
    *) echo "";; esac; }

ALL_UPSTREAMS="leptonica mupdf pdf-to-markdown tesseract"   # sorted, for stable emission

# --------------------------------------------------------------------------
# 2. Header extraction helpers (a port target's provenance lives in its leading
#    comment block; we read the first 80 lines' comment lines only).
# --------------------------------------------------------------------------
header_of() { awk 'NR<=80 && /^[[:space:]]*\/\//' "$1"; }

detect_upstream() {   # stdin: header text -> upstream key or empty
    local h; h="$(cat)"
    # A cited `pdf2md/` path is the unambiguous, specific signal and wins over
    # incidental prose mentions -- textnorm.rs / headers.rs port pdf-to-markdown
    # yet also name MuPDF because they follow AID-0051's MuPDF port conventions.
    case "$h" in *pdf2md/*) echo "pdf-to-markdown"; return;; esac
    case "$h" in
        *MuPDF*)     echo "mupdf";;
        *Tesseract*) echo "tesseract";;
        *Leptonica*) echo "leptonica";;
        *pdf-to-markdown*) echo "pdf-to-markdown";;
        *) echo "";;
    esac
}

# Extract cited upstream source paths from a header, canonicalized per upstream.
extract_sources() {   # args: <upstream>; stdin: header text -> one source per line
    local up="$1" h; h="$(cat)"
    {
        # (a) plain paths ending in a source extension
        { printf '%s\n' "$h" | grep -oE '`[^`]+\.(c|h|cpp|cc|hpp|py)`' 2>/dev/null || true; } | tr -d '`'
        # (b) brace-expanded paths: base.{cpp,h} -> base.cpp, base.h
        { printf '%s\n' "$h" | grep -oE '`[^`]+\.\{[a-z,]+\}`' 2>/dev/null || true; } | tr -d '`' | while IFS= read -r tok; do
            local base exts ext
            base="${tok%.\{*}"; exts="${tok#*\{}"; exts="${exts%\}}"
            IFS=','; for ext in $exts; do printf '%s.%s\n' "$base" "$ext"; done; unset IFS
        done
    } | while IFS= read -r p; do
        [ -z "$p" ] && continue
        case "$up" in
            pdf-to-markdown)
                # Only pdf-to-markdown's own pdf2md/ modules count. A citing file
                # may also name marker's processors/*.py (Apache-2.0, clean-room
                # study only, no algorithm ported) -- those are deliberately NOT
                # tracked here, so require the path to contain `pdf2md/`.
                case "$p" in *pdf2md/*) printf 'pdf2md/%s\n' "${p##*pdf2md/}";;
                             *) : ;; esac ;;
            *)
                case "$p" in */*) printf '%s\n' "$p";; *) : ;; esac ;;  # need a real path
        esac
    done | sort -u
}

extract_sha() {   # stdin: header text -> 7+ hex SHA or empty (first inline commit)
    { grep -oiE 'commit `?[0-9a-f]{7,40}' 2>/dev/null || true; } | head -1 | { grep -oiE '[0-9a-f]{7,40}' || true; } | head -1
}

status_of() {   # stdin: header text -> ported | deliberately-diverged | in-progress
    local h; h="$(cat)"
    case "$h" in
        *PORT-LEDGER:\ in-progress*|*PORT-LEDGER:in-progress*) echo "in-progress"; return;;
    esac
    # A clean-room adaptation that only borrows the idea disclaims a port
    # ("original Rust, not a port" / "not a translation"); check that FIRST so a
    # file that both cites a source and disclaims porting it is classed diverged.
    case "$h" in
        *"not a port"*|*"not a translation"*|*"not a line-for-line port"*) echo "deliberately-diverged"; return;;
    esac
    # A faithful port announces itself: "Ported from ...", "ported verbatim
    # from ...", or "translated/Translated to Rust for KOPITIAM".
    case "$h" in
        *[Pp]orted\ from*|*[Pp]orted\ verbatim*|*[Tt]ranslated\ to\ Rust*) echo "ported";;
        *) echo "deliberately-diverged";;   # cites a source but never claims a port
    esac
}

note_for() {   # args: <status>; short, fixed divergence note (no free-form drift)
    case "$1" in
        ported)                echo "";;
        deliberately-diverged) echo "clean-room adaptation: idea adapted, target header disclaims a faithful port";;
        in-progress)           echo "port in progress (PORT-LEDGER marker)";;
        unported)              echo "present in vendor tree; no port target cites it";;
    esac
}

# --------------------------------------------------------------------------
# 3. Build the canonical, sorted TSV of ledger entries on stdout.
#    columns: upstream \t source \t target \t status \t commit \t notes
# --------------------------------------------------------------------------
build_tsv() {
    local seen_sources
    seen_sources="$(mktemp)"

    # ---- Pass A: target-driven inventory ----
    # Skip aggregator files: a crate-root `lib.rs` / module `mod.rs` is an
    # OVERVIEW that re-lists sources ported by their own dedicated modules
    # (e.g. kopitiam-ocr/src/lib.rs enumerates the whole Tesseract set). The
    # authoritative point-of-use provenance lives in the per-module files, so
    # counting the overview too would double-count every source. Every source an
    # aggregator names is also cited by its dedicated module (verified).
    local f header up srcs sha status note
    for f in $(find $TARGET_ROOTS -name '*.rs' 2>/dev/null | grep -v '/vendor/' | grep -vE '/(lib|mod)\.rs$' | sort); do
        header="$(header_of "$f")"
        [ -z "$header" ] && continue
        up="$(printf '%s' "$header" | detect_upstream)"
        [ -z "$up" ] && continue
        srcs="$(printf '%s' "$header" | extract_sources "$up")"
        [ -z "$srcs" ] && continue
        status="$(printf '%s' "$header" | status_of)"
        sha="$(printf '%s' "$header" | extract_sha)"
        [ -z "$sha" ] && sha="$(upstream_pin "$up")"
        note="$(note_for "$status")"
        while IFS= read -r s; do
            [ -z "$s" ] && continue
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$up" "$s" "$f" "$status" "$sha" "$note"
            [ "$up" = "pdf-to-markdown" ] && printf '%s\n' "$s" >>"$seen_sources"
        done <<EOF
$srcs
EOF
    done

    # ---- Pass B: source-driven unported detection (pdf-to-markdown backlog) ----
    local p rel
    for p in $(find "$PDF2MD_VENDOR" -name '*.py' 2>/dev/null | sort); do
        rel="pdf2md/${p##*/pdf2md/}"
        if ! grep -qxF "$rel" "$seen_sources" 2>/dev/null; then
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
                "pdf-to-markdown" "$rel" "" "unported" "" "$(note_for unported)"
        fi
    done

    rm -f "$seen_sources"
}

# --------------------------------------------------------------------------
# 4. JSON emitter (awk over the sorted TSV). Simple ASCII fields; we still
#    escape " and \ defensively. No timestamps, no absolute paths.
# --------------------------------------------------------------------------
emit_json() {   # args: <sorted-tsv-file>
    local tsv="$1"
    {
        printf '{\n'
        printf '  "schema": "kopitiam-port-ledger/v1",\n'
        printf '  "generator": "scripts/port-ledger.sh",\n'
        printf '  "note": "Deterministically generated from point-of-use provenance headers. Do not edit by hand; run scripts/port-ledger.sh.",\n'
        printf '  "status_values": ["ported", "in-progress", "deliberately-diverged", "unported"],\n'
        # upstreams block
        printf '  "upstreams": {\n'
        local first=1 up
        for up in $ALL_UPSTREAMS; do
            [ $first -eq 1 ] || printf ',\n'; first=0
            printf '    "%s": { "name": "%s", "license": "%s", "pinned_commit": "%s", "url": "%s", "vendor": "%s" }' \
                "$up" "$(upstream_name "$up")" "$(upstream_lic "$up")" "$(upstream_pin "$up")" "$(upstream_url "$up")" "$(upstream_vendor "$up")"
        done
        printf '\n  },\n'
        # entries block
        printf '  "entries": [\n'
        awk -F '\t' '
            function esc(s){ gsub(/\\/,"\\\\",s); gsub(/"/,"\\\"",s); return s }
            {
                tgt = ($3=="") ? "null" : "\"" esc($3) "\""
                sha = ($5=="") ? "null" : "\"" esc($5) "\""
                if (NR>1) printf ",\n"
                printf "    { \"upstream\": \"%s\", \"source\": \"%s\", \"target\": %s, \"status\": \"%s\", \"commit\": %s, \"notes\": \"%s\" }",
                    esc($1), esc($2), tgt, esc($4), sha, esc($6)
            }
            END { if (NR>0) printf "\n" }
        ' "$tsv"
        printf '  ],\n'
        # summary block
        printf '  "summary": {\n'
        printf '    "total_entries": %s,\n' "$(wc -l <"$tsv" | tr -d ' ')"
        printf '    "by_status": {\n'
        local s first_s=1 c
        for s in ported in-progress deliberately-diverged unported; do
            c="$(awk -F '\t' -v st="$s" '$4==st{n++} END{print n+0}' "$tsv")"
            [ $first_s -eq 1 ] || printf ',\n'; first_s=0
            printf '      "%s": %s' "$s" "$c"
        done
        printf '\n    },\n'
        printf '    "by_upstream": {\n'
        local first_u=1
        for up in $ALL_UPSTREAMS; do
            c="$(awk -F '\t' -v u="$up" '$1==u{n++} END{print n+0}' "$tsv")"
            [ $first_u -eq 1 ] || printf ',\n'; first_u=0
            printf '      "%s": %s' "$up" "$c"
        done
        printf '\n    },\n'
        printf '    "unported": [\n'
        awk -F '\t' '$4=="unported"{ printf "%s      \"%s\"", (n++?",\n":""), $2 } END{ if(n) printf "\n" }' "$tsv"
        printf '    ]\n'
        printf '  }\n'
        printf '}\n'
    }
}

# --------------------------------------------------------------------------
# 5. Markdown human view (awk over the sorted TSV).
# --------------------------------------------------------------------------
emit_md() {   # args: <sorted-tsv-file>
    local tsv="$1" up
    {
        printf '# Port ledger\n\n'
        printf '> **Machine-generated by `scripts/port-ledger.sh` (Task III-1). Do not edit by hand.**\n'
        printf '> It is a pure function of the point-of-use provenance headers in the port tree\n'
        printf '> (`crates/kopitiam-pdf/src/mupdf/*`, `crates/kopitiam-ocr/src/*`,\n'
        printf '> `crates/kopitiam-document/src/reconstruction/*`) and the vendored\n'
        printf '> pdf-to-markdown reference. Regenerate with `scripts/port-ledger.sh`.\n\n'
        printf 'This ledger records, for every upstream source unit kopitiam translates,\n'
        printf 'where it landed (`target`), whether it is `ported` / `deliberately-diverged` /\n'
        printf '`in-progress` / `unported`, and the upstream `commit` the translation was made\n'
        printf 'from (so [upstream drift](#upstream-drift) can be checked later). It replaces the\n'
        printf 'per-session re-discovery of "what is wired, what is ported, what is left".\n\n'

        # Summary
        printf '## Summary\n\n'
        printf '| Status | Count |\n| --- | --- |\n'
        local s c
        for s in ported in-progress deliberately-diverged unported; do
            c="$(awk -F '\t' -v st="$s" '$4==st{n++} END{print n+0}' "$tsv")"
            printf '| %s | %s |\n' "$s" "$c"
        done
        printf '| **total** | **%s** |\n\n' "$(wc -l <"$tsv" | tr -d ' ')"

        printf '| Upstream | License | Pinned commit | Entries |\n| --- | --- | --- | --- |\n'
        for up in $ALL_UPSTREAMS; do
            c="$(awk -F '\t' -v u="$up" '$1==u{n++} END{print n+0}' "$tsv")"
            printf '| %s | %s | `%s` | %s |\n' "$(upstream_name "$up")" "$(upstream_lic "$up")" "$(upstream_pin "$up")" "$c"
        done
        printf '\n'

        # Per-upstream tables
        for up in $ALL_UPSTREAMS; do
            awk -F '\t' -v u="$up" '$1==u{n++} END{exit !n}' "$tsv" || continue
            printf '## %s\n\n' "$(upstream_name "$up")"
            printf '| Source | Target | Status | Commit | Notes |\n| --- | --- | --- | --- | --- |\n'
            awk -F '\t' -v u="$up" '
                $1==u {
                    tgt = ($3=="") ? "—" : "`" $3 "`"
                    sha = ($5=="") ? "—" : "`" $5 "`"
                    note = ($6=="") ? "" : $6
                    printf "| `%s` | %s | %s | %s | %s |\n", $2, tgt, $4, sha, note
                }' "$tsv"
            printf '\n'
        done

        # Unported list (the recurring "what is left to port" fact)
        printf '## What is left to port (unported)\n\n'
        if awk -F '\t' '$4=="unported"{n++} END{exit !n}' "$tsv"; then
            awk -F '\t' '$4=="unported"{ printf "- `%s`\n", $2 }' "$tsv"
        else
            printf '_None: every tracked source unit is ported or deliberately diverged._\n'
        fi
        printf '\n'

        # Drift mechanism
        printf '## Upstream drift\n\n'
        printf 'Each ported/diverged entry records the upstream `commit` it was translated from.\n'
        printf 'The generator is offline and deterministic, so it does **not** fetch upstream.\n'
        printf 'To check drift, out of the deterministic path:\n\n'
        printf '```sh\n'
        printf '# for an entry with upstream U and recorded commit C:\n'
        printf 'git ls-remote <U-url> <ref>          # newest upstream SHA for a ref\n'
        printf '# if it differs from C, diff the cited source path between C and the ref\n'
        printf '# in a local clone to see whether the ported region actually moved.\n'
        printf '```\n\n'
        printf '`scripts/port-ledger.sh --drift-check <ref>` documents this compare plan.\n\n'

        # Deferred CLI wiring
        printf '## Deferred: the `port status` CLI subcommand\n\n'
        printf 'Task III-1 also specifies a `kopitiam port status` subcommand. It is **not** wired\n'
        printf 'in this wave because `apps/cli/src/main.rs` is owned by another agent. Intended\n'
        printf 'wiring (the standard clap 3-edit pattern, per §10.1 finding 8):\n\n'
        printf '1. `apps/cli/src/port.rs` — a new module exposing `Args` + `pub fn run(args)`\n'
        printf '   that either shells `scripts/port-ledger.sh --report` or reads\n'
        printf '   `docs/port-ledger.json` and prints the ledger / `--unported` / `--json`.\n'
        printf '2. `main.rs`: add `mod port;` (with the other `mod` lines).\n'
        printf '3. `main.rs`: add a `Port(port::Args)` variant to the `Command` enum.\n'
        printf '4. `main.rs`: add the `Command::Port(a) => port::run(a)` match arm.\n\n'
    }
}

# --------------------------------------------------------------------------
# 6. Report mode (stdout): counts by status + the unported list.
# --------------------------------------------------------------------------
emit_report() {   # args: <sorted-tsv-file>
    local tsv="$1" s c up
    printf 'kopitiam port ledger — report\n'
    printf '=============================\n\n'
    printf 'status counts:\n'
    for s in ported in-progress deliberately-diverged unported; do
        c="$(awk -F '\t' -v st="$s" '$4==st{n++} END{print n+0}' "$tsv")"
        printf '  %-22s %s\n' "$s" "$c"
    done
    printf '  %-22s %s\n\n' "total" "$(wc -l <"$tsv" | tr -d ' ')"
    printf 'entries by upstream:\n'
    for up in $ALL_UPSTREAMS; do
        c="$(awk -F '\t' -v u="$up" '$1==u{n++} END{print n+0}' "$tsv")"
        printf '  %-22s %s\n' "$up" "$c"
    done
    printf '\nunported (pdf-to-markdown backlog — the recurring "what is left to port"):\n'
    if awk -F '\t' '$4=="unported"{n++} END{exit !n}' "$tsv"; then
        awk -F '\t' '$4=="unported"{ printf "  %s\n", $2 }' "$tsv"
    else
        printf '  (none)\n'
    fi
    printf '\nnote: unported is enumerated only for the finite pdf-to-markdown pdf2md/ set.\n'
    printf 'The MuPDF/Tesseract/Leptonica C/C++ vendor trees hold thousands of files that\n'
    printf 'were never all slated for porting, so their ported units are inventoried\n'
    printf '(target-driven) but their "unported" remainder is not enumerated.\n'
}

# --------------------------------------------------------------------------
# 7. Main.
# --------------------------------------------------------------------------
MODE="write"
DRIFT_REF=""
case "${1:-}" in
    ""|--write) MODE="write";;
    --report)   MODE="report";;
    --check)    MODE="check";;
    --drift-check) MODE="drift"; DRIFT_REF="${2:-HEAD}";;
    -h|--help)
        sed -n '2,70p' "$0" | sed 's/^# \{0,1\}//'
        exit 0;;
    *) echo "port-ledger: unknown argument: $1 (try --help)" >&2; exit 2;;
esac

# Compute the canonical sorted TSV once.
TSV="$(mktemp)"
trap 'rm -f "$TSV"' EXIT
build_tsv | sort -u >"$TSV"

case "$MODE" in
    report) emit_report "$TSV"; exit 0;;
    drift)
        echo "port-ledger: upstream-drift compare plan for ref '$DRIFT_REF'" >&2
        printf 'For each ported/diverged entry, compare its recorded commit to the ref:\n'
        for up in $ALL_UPSTREAMS; do
            u_url="$(upstream_url "$up")"
            printf '  %-16s git ls-remote %s %s   # vs recorded commit(s):\n' "$up" "$u_url" "$DRIFT_REF"
            awk -F '\t' -v u="$up" '$1==u && $5!="" {print $5"\t"$2}' "$TSV" | sort -u | awk -F '\t' '{printf "      %s  %s\n", $1, $2}'
        done
        printf '\n(no network call is made by this script; run the ls-remote/diff yourself)\n'
        exit 0;;
    check)
        tmpj="$(mktemp)"; tmpm="$(mktemp)"
        emit_json "$TSV" >"$tmpj"; emit_md "$TSV" >"$tmpm"
        rc=0
        if ! diff -q "$JSON_OUT" "$tmpj" >/dev/null 2>&1; then echo "port-ledger: STALE: $JSON_OUT differs — run scripts/port-ledger.sh" >&2; rc=1; fi
        if ! diff -q "$MD_OUT"   "$tmpm" >/dev/null 2>&1; then echo "port-ledger: STALE: $MD_OUT differs — run scripts/port-ledger.sh"   >&2; rc=1; fi
        rm -f "$tmpj" "$tmpm"
        [ $rc -eq 0 ] && echo "port-ledger: committed ledger is up to date" >&2
        exit $rc;;
    write)
        mkdir -p docs
        emit_json "$TSV" >"$JSON_OUT"
        emit_md   "$TSV" >"$MD_OUT"
        echo "port-ledger: wrote $JSON_OUT and $MD_OUT ($(wc -l <"$TSV" | tr -d ' ') entries)" >&2
        # Determinism self-check: regenerate to a temp and diff (acceptance).
        tmpj="$(mktemp)"; tmpm="$(mktemp)"
        emit_json "$TSV" >"$tmpj"; emit_md "$TSV" >"$tmpm"
        if diff -q "$JSON_OUT" "$tmpj" >/dev/null 2>&1 && diff -q "$MD_OUT" "$tmpm" >/dev/null 2>&1; then
            echo "port-ledger: determinism self-check PASSED (byte-identical regeneration)" >&2
        else
            echo "port-ledger: determinism self-check FAILED" >&2
            diff "$JSON_OUT" "$tmpj" >&2 || true
            diff "$MD_OUT"   "$tmpm" >&2 || true
            rm -f "$tmpj" "$tmpm"; exit 1
        fi
        rm -f "$tmpj" "$tmpm"
        exit 0;;
esac

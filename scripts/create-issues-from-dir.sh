#!/usr/bin/env bash
# Create GitHub issues from a directory of Markdown files with YAML front matter.
#
# Each file looks like:
#
#     ---
#     title: '[pdf2md] F-1 — Some title'
#     labels: bug
#     ---
#
#     Body markdown...
#
# Files are processed in sorted filename order. The FIRST file is treated as the
# tracking issue: it is created first, and every subsequent issue gets a
# "Tracking: #N" line appended so the children point back at it.
#
# Usage:
#     scripts/create-issues-from-dir.sh docs/issues/field-report-2026-08-01
#     scripts/create-issues-from-dir.sh <dir> --dry-run
#
# Requires: gh (authenticated), awk, sed.

set -euo pipefail

DIR="${1:-}"
DRY_RUN="${2:-}"

if [[ -z "$DIR" || ! -d "$DIR" ]]; then
  echo "usage: $0 <dir-of-issue-md-files> [--dry-run]" >&2
  exit 1
fi

command -v gh >/dev/null || { echo "error: gh not found" >&2; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "error: gh not authenticated — run 'gh auth login'" >&2; exit 1; }

# Front matter is delimited by the first two '---' lines.
# extract_field <file> <key>  -> value with surrounding quotes stripped
extract_field() {
  awk -v key="$2" '
    NR==1 && $0=="---" { infm=1; next }
    infm && $0=="---"  { exit }
    infm {
      # split on the FIRST colon only, so colons inside the title survive
      i = index($0, ":")
      if (i > 0 && substr($0, 1, i-1) == key) {
        v = substr($0, i+1)
        sub(/^[ \t]+/, "", v)
        # strip one layer of matching single or double quotes
        if (v ~ /^\x27.*\x27$/ || v ~ /^".*"$/) v = substr(v, 2, length(v)-2)
        print v
        exit
      }
    }
  ' "$1"
}

# Everything after the closing '---' of the front matter.
extract_body() {
  awk 'NR==1 && $0=="---" { infm=1; next }
       infm && $0=="---"  { infm=0; body=1; next }
       body' "$1"
}

TRACKING_NUM=""
CREATED=()

shopt -s nullglob
FILES=("$DIR"/*.md)
shopt -u nullglob

if (( ${#FILES[@]} == 0 )); then
  echo "error: no .md files in $DIR" >&2
  exit 1
fi

echo "Found ${#FILES[@]} issue file(s) in $DIR"
if [[ "$DRY_RUN" == "--dry-run" ]]; then echo "(dry run — nothing will be created)"; fi
echo

for f in "${FILES[@]}"; do
  title="$(extract_field "$f" title)"
  labels="$(extract_field "$f" labels)"

  if [[ -z "$title" ]]; then
    echo "skip: $(basename "$f") has no 'title' in front matter" >&2
    continue
  fi

  # Children link back to the tracking issue.
  tmp="$(mktemp)"
  extract_body "$f" > "$tmp"
  if [[ -n "$TRACKING_NUM" ]]; then
    printf '\n---\n\nTracking: #%s\n' "$TRACKING_NUM" >> "$tmp"
  fi

  if [[ "$DRY_RUN" == "--dry-run" ]]; then
    echo "would create: [$labels] $title  ($(wc -l < "$tmp") lines)"
    rm -f "$tmp"
    continue
  fi

  # --label is repeatable; split a comma-separated list into several flags.
  label_args=()
  IFS=',' read -ra parts <<< "$labels"
  for l in "${parts[@]}"; do
    l="$(echo "$l" | sed 's/^[ \t]*//;s/[ \t]*$//')"
    [[ -n "$l" ]] && label_args+=(--label "$l")
  done

  url="$(gh issue create --title "$title" --body-file "$tmp" "${label_args[@]}")"
  rm -f "$tmp"

  num="${url##*/}"
  [[ -z "$TRACKING_NUM" ]] && TRACKING_NUM="$num"
  CREATED+=("$num  $title")
  echo "created #$num  $title"
  echo "        $url"

  # Be gentle with the API on a long run.
  sleep 1
done

echo
echo "Done. Created ${#CREATED[@]} issue(s)."
if [[ -n "$TRACKING_NUM" ]]; then echo "Tracking issue: #$TRACKING_NUM"; fi

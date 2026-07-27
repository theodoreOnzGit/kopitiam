#!/usr/bin/env bash
#
# gen-kopitiam-skill.sh — deterministically generate `kopitiam_skill.md`,
# an agent-facing "skill" doc that teaches any Claude/AI agent to drive the
# kopitiam CLI non-interactively.
#
# DETERMINISM CONTRACT
# --------------------
# The generated file is a PURE FUNCTION of exactly two inputs:
#   (a) the kopitiam binary's own `--help` output — clap is the single source
#       of truth for the command surface, so the doc can never drift from the
#       real commands; and
#   (b) the static, curated agent-guidance prose embedded in this script.
#
# Consequences enforced here:
#   * NO `date`/timestamps, NO `$RANDOM`, NO hostnames, NO machine-varying
#     paths ever reach the output. The only place a path is printed is the
#     "using binary: ..." line, which goes to STDERR and never to the file.
#   * Subcommands are emitted in the exact order `kopitiam --help` lists them
#     (stable, single ordering — no re-sorting).
#   * Running this twice against the same binary yields a byte-identical file.
#     Step 4 proves this with a self-check: it regenerates to a temp file and
#     `diff`s; a mismatch is a hard failure (exit non-zero).
#
# USAGE
#   bash scripts/gen-kopitiam-skill.sh [OUTPUT_PATH]
#   KOPITIAM_BIN=/path/to/kopitiam bash scripts/gen-kopitiam-skill.sh
#
# Default OUTPUT_PATH is `kopitiam_skill.md` in the current directory.

set -euo pipefail

# --------------------------------------------------------------------------
# 1. Resolve the kopitiam binary: $KOPITIAM_BIN, else PATH, else build it.
#    The chosen path is announced on STDERR so it never pollutes the output.
# --------------------------------------------------------------------------
resolve_bin() {
	if [ -n "${KOPITIAM_BIN:-}" ]; then
		printf '%s\n' "$KOPITIAM_BIN"
		return 0
	fi
	if command -v kopitiam >/dev/null 2>&1; then
		command -v kopitiam
		return 0
	fi
	echo "gen-kopitiam-skill: kopitiam not on PATH; building (cargo build -q --release -p kopitiam)..." >&2
	cargo build -q --release -p kopitiam >&2
	for cand in target/release/kopitiam target/release/kopitiam.exe; do
		if [ -f "$cand" ]; then
			printf '%s\n' "$cand"
			return 0
		fi
	done
	echo "gen-kopitiam-skill: error: could not locate a built kopitiam binary under target/release/" >&2
	return 1
}

# --------------------------------------------------------------------------
# 2. Help parsing helpers.
# --------------------------------------------------------------------------

# Extract the subcommand names from a `--help` text's `Commands:` block, in
# listed order. Awk scopes strictly to the region between `^Commands:` and the
# next `^Options:` line, and takes the first token of each indented row.
list_subcommands() {
	printf '%s\n' "$1" | awk '
		/^Commands:/ { in_cmds = 1; next }
		/^Options:/  { in_cmds = 0 }
		in_cmds && /^[[:space:]]+[A-Za-z]/ { print $1 }
	'
}

# Static, curated classification: the INTERACTIVE commands an agent must never
# spawn. Everything else is non-interactive and agent-safe. `view` is a
# full-screen, on-screen PDF viewer that owns the terminal and runs its own
# blocking event loop (it is what `kmux latex` shells out to for a human's live
# preview), so — like `tui` and `ai chat` — it will HANG a non-interactive agent.
is_interactive() {
	case "$1" in
	"tui" | "ai chat" | "view") return 0 ;;
	*) return 1 ;;
	esac
}

# Emit one command's reference subsection: heading, safety marker, and its
# `--help` verbatim in a fenced block. Recurses one level into command groups
# (e.g. `ai`, `models`), which is detected by a nested `Commands:` block.
emit_cmd() {
	local bin="$1" path="$2" level="$3"
	local help_txt nested marker

	# shellcheck disable=SC2086
	help_txt="$("$bin" $path --help 2>&1)"
	# Clap prints argv[0]'s FILE NAME in every `Usage:` line, so a Windows build
	# says `kopitiam.exe` and a Linux/Termux one says `kopitiam` — same command,
	# different bytes. Without this normalisation the determinism self-check
	# below only holds *per platform*: regenerating on Windows then on Linux
	# rewrites every Usage line, and the committed skill churns back and forth
	# between the maintainer's box and any agent's. Force the platform-neutral
	# spelling so regeneration is byte-identical everywhere. Same spirit as
	# CLAUDE.md "Cross-platform paths" (forward slashes regardless of OS).
	help_txt="${help_txt//kopitiam.exe/kopitiam}"
	nested="$(list_subcommands "$help_txt")"

	if [ -n "$nested" ]; then
		marker='**Command group** — not invoked directly; dispatch to one of its subcommands below.'
	elif is_interactive "$path"; then
		marker='**INTERACTIVE — DO NOT RUN from an agent.** Full-screen / token-streamed; it blocks on stdin and will HANG a non-interactive session.'
	else
		marker='**Agent-safe** — non-interactive: takes flags, writes files, and returns an exit code.'
	fi

	printf '%s `kopitiam %s`\n\n' "$level" "$path"
	printf '%s\n\n' "$marker"
	printf '```text\n%s\n```\n\n' "$help_txt"

	if [ -n "$nested" ]; then
		local sub
		while IFS= read -r sub; do
			[ -z "$sub" ] && continue
			[ "$sub" = "help" ] && continue
			emit_cmd "$bin" "$path $sub" "####"
		done <<-EOF
			$nested
		EOF
	fi
}

# --------------------------------------------------------------------------
# 3. Generate the whole document to STDOUT. Static prose lives in quoted
#    heredocs (`<<'EOF'`) so nothing expands; dynamic help is interleaved.
# --------------------------------------------------------------------------
generate() {
	local bin="$1"

	cat <<'EOF'
---
name: kopitiam-cli
description: Use when you need to convert PDFs to semantic Markdown or otherwise drive the kopitiam Rust CLI non-interactively — batch document conversion, Rust-project scans/refactors, and local model management from scripts or agents.
---

# kopitiam CLI skill

## What kopitiam is

`kopitiam` is a Rust command-line tool. Its headline job is turning PDFs into
semantic Markdown via the `pdf2md` command (a text-extract → structure-recover
→ Markdown-render pipeline), and it also ships Rust-project tooling (scan,
rename, code-actions, plan) and a local-AI layer (offline model store plus a
chat/TUI front end that runs even with no weights present).

## CRITICAL agent guidance (read first)

- **Use the non-interactive subcommands.** They accept flags, read/write
  files, and return meaningful exit codes — exactly what an agent needs.
- **NEVER run `kopitiam tui`, `kopitiam ai chat`, or `kopitiam view` from an
  agent.** All three are INTERACTIVE programs: `tui` is a full-screen terminal
  UI, `ai chat` streams tokens while blocking on stdin, and `view` is a
  full-screen, on-screen PDF viewer that owns the terminal and runs its own
  blocking event loop (it is what `kmux latex` shells out to for a human's live
  preview). From a non-interactive agent they do not return — they will **HANG
  the session**. They exist for humans at a real terminal, not for automation.
- **The primary agent workflow is `pdf2md`.** Convert one file, or loop over a
  folder, then read the printed validation report to confirm nothing dropped.
- Everything else in the Command reference below is marked **Agent-safe** or
  **INTERACTIVE** so you can tell at a glance what is drivable.

## Recipes

All commands are non-interactive unless noted. **Always quote paths that
contain spaces.**

Convert a single PDF to a chosen output file:

```bash
kopitiam pdf2md "in.pdf" -o "out.md"
```

Convert every PDF in the current folder into a `markdown/` subfolder,
preserving base names:

```bash
mkdir -p markdown
for f in *.pdf; do
  kopitiam pdf2md "$f" -o "markdown/${f%.pdf}.md"
done
```

Read the validation / recovery report: `pdf2md` prints a report comparing the
extracted word count against the rendered Markdown word count, ending in a
PASS/FAIL line. After a batch, check exit codes and scan the output for FAIL to
find documents that need a second look:

```bash
kopitiam pdf2md "paper.pdf" -o "paper.md" || echo "conversion returned non-zero for paper.pdf"
```

Manage local AI model weights (all non-interactive):

```bash
kopitiam models list          # show the catalog and what is present on disk
kopitiam models pull <id>     # download + SHA-256 verify a model by catalog id
kopitiam models path <id>     # print expected on-disk paths (bring-your-own)
kopitiam models verify <id>   # re-check a present model against its checksums
```

### Navigate Rust code without reading whole files (the token-max loop)

When you are working *in a Rust codebase* (including kopitiam's own source),
do NOT open files blind. The whole point of these commands is to spend a few
dozen tokens on coordinates and skeletons instead of thousands on file bodies.
The loop is **measure → skeleton → coordinates → read only the slice you need**:

**1. `kopitiam tokens <path>...` BEFORE you read anything.** It prints a
deterministic BPE token estimate for a file or a whole directory (no
rust-analyzer, instant). Let the number decide read-vs-outline: a 2k-token file
you can afford to read; a 130k-token directory you must *not* — target its call
sites instead.

Takes **many paths in one call**, so budgeting a few places at once costs one
invocation, not one each. Naming the same file twice (directly, or once
directly and once through its parent directory) counts it **once** — the grand
total never double-counts.

```bash
kopitiam tokens crates/kopitiam-document/src/reconstruction   # a directory: total to budget against
kopitiam tokens apps/cli/src/main.rs                          # one file: is it cheap enough to read whole?
kopitiam tokens apps/cli/src/ocr_fallback.rs crates/kopitiam-ocr/src   # several at once: one budget for the job
kopitiam tokens --json src/ | jq '.total'                     # gate programmatically, no prose parsing
kopitiam tokens src/ --tree                                   # per-directory rollup: find the heavy subtree in one call
```

**2. `kopitiam outline <file>` instead of reading the file for orientation.**
It prints a body-free skeleton — every declaration with its line number, no
function bodies — roughly a 10x reduction versus a full read. Read the outline,
pick the one item you actually need, then read just those lines.

```bash
kopitiam outline crates/kopitiam-document/src/reconstruction/mod.rs
kopitiam outline --json src/foo.rs        # machine-readable items[] with line/kind/name
```

**3. `kopitiam refs` / `def` / `sig` / `callers` / `callees` / `impls` instead
of grep-then-read.** These answer "who calls this / where is it defined / what
is its signature" as `file:line:character` coordinates (and a signature), not
code bodies — the deterministic, exact answer rust-analyzer already knows. Pass
`--file <FILE>` naming the file that declares the symbol (its identifier
position is the query anchor):

```bash
kopitiam def  --file crates/kopitiam-document/src/reconstruction/tables.rs try_table   # where + signature
kopitiam sig  --file src/tables.rs try_table                                           # signature alone
kopitiam refs --file src/tables.rs try_table                                           # every call site, as coordinates
kopitiam callers --file src/tables.rs try_table --depth 2                              # up the call graph
```

Then read only the coordinates that matter — never the whole file to find them.

**4. `kopitiam slice <file> <A-B>` to READ only the lines you need.** This is
the last step of the loop: once `outline`/`refs` hand back `file:line`
coordinates, slice the exact range instead of reading the whole file (and
instead of shelling `sed`). It prints the lines with a `(~N tokens)` cost so the
read stays budget-aware. `--grep <pat>` fuses grep-then-slice — it prints each
match's ±context neighbourhood as merged slices, so you grep and read only the
hit windows in ONE call. Ranges accept `A-B`, a bare `A`, `A-` (to EOF), or
`-B`; `--json` for the machine form.

```bash
kopitiam slice src/tables.rs 120-145                 # read exactly the function outline pointed you at
kopitiam slice src/tables.rs --grep try_table        # each hit + context, merged, in one call
kopitiam slice src/tables.rs 120-145 --json          # {file, slices:[{start,end,tokens,lines}], total_tokens}
```

**5. `kopitiam check --compact` / `test --compact` instead of raw cargo
output.** `check --compact` collapses cargo's diagnostics to one deduplicated
line per distinct problem, sorted by file (one bad type stops spamming forty
identical errors); `test --compact` reports each failure as
`name — assertion @ file:line` rather than the full captured stdout. Add
`--json` to gate on the result without parsing prose:

```bash
kopitiam check --compact          # dedup'd diagnostics, one line each
kopitiam test  --compact          # failures as name — assertion @ file:line
kopitiam check --json | jq length # zero == clean, gate on it
```

**On a large workspace, `outline` and `refs` DEFAULT to the instant syntactic
scan** — kopitiam's own tree is large enough that rust-analyzer cannot index in
bounded time, so waiting for it is wasted. They print a one-line stderr note
(`large workspace: syntactic outline; --lsp for rust-analyzer`) and answer
immediately from a deterministic, dependency-free scan. That scan is textual
(grep-grade) rather than semantic, so `refs` hits are labelled
`(syntactic, not semantic — verify)`. Pass **`--lsp`** to *require* rust-analyzer
(real signatures / semantic references) and fail hard on timeout instead of
answering syntactically; on a small workspace rust-analyzer is tried first
automatically. `--no-lsp` (alias `--syntactic`) always forces the scan.

The other semantic queries — `def`/`sig`/`callers`/`callees`/`impls` — have no
syntactic fallback (a definition/signature/call-graph answer needs real
resolution), so they always use rust-analyzer and wait for its index (default
**180 s**, overridable with `KOPITIAM_RA_TIMEOUT_SECS`).

```bash
kopitiam outline src/foo.rs                          # instant syntactic skeleton on a large workspace
kopitiam refs --file src/tables.rs try_table         # instant textual call sites, verify each
kopitiam outline --lsp src/foo.rs                    # require rust-analyzer (real signatures), fail hard on timeout
KOPITIAM_RA_TIMEOUT_SECS=20 kopitiam def --file src/tables.rs try_table  # shorten the RA wait
```

## Command reference

Each subsection embeds the command's real `kopitiam ... --help` output verbatim
(clap is the source of truth) and flags whether it is agent-safe or interactive.

EOF

	# --- deterministic core: iterate top-level subcommands in listed order ---
	local top c
	top="$(list_subcommands "$("$bin" --help 2>&1)")"
	while IFS= read -r c; do
		[ -z "$c" ] && continue
		[ "$c" = "help" ] && continue
		emit_cmd "$bin" "$c" "###"
	done <<-EOF
		$top
	EOF

	cat <<'EOF'
## Gotchas

- **Quote paths with spaces.** `kopitiam pdf2md "My Paper (v2).pdf" -o "My Paper.md"`.
  Unquoted paths split on whitespace and the command will read the wrong file
  or fail.
- **`pdf2md` prints a PASS/FAIL recovery report.** It compares extracted vs.
  rendered word counts as a cheap check that structure reconstruction did not
  silently drop content. Treat FAIL (or a large word-count gap) as a signal to
  inspect that document, not necessarily a hard error.
- **Some PDFs skip unreadable pages.** A page that cannot be extracted may be
  omitted from the Markdown; the report is how you notice.
- **Scanned pages are OCR'd automatically.** `pdf2md` detects image-only
  (low-text) pages and recognizes them with the built-in Tesseract LSTM engine —
  `--ocr auto` (the default). It needs the language `.traineddata` in the model
  store; the default `--ocr-lang eng,chi_sim,jpn`, so first run
  `kopitiam models pull tessdata-eng` (and `tessdata-chi_sim` / `tessdata-jpn` as
  needed), or the run fails telling you which to pull. Use `--ocr off` to disable
  it or `--ocr on` to force OCR on every page. Born-digital PDFs are untouched
  (the fallback never triggers on a page with real text).
- **`tui`, `ai chat`, and `view` are interactive** and must never be launched by
  an agent (see CRITICAL agent guidance above). `view` in particular is the
  on-screen page viewer `kmux latex` opens for a human's live preview.
EOF
}

# --------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------
OUT="${1:-kopitiam_skill.md}"

BIN="$(resolve_bin)"
echo "gen-kopitiam-skill: using binary: $BIN" >&2

generate "$BIN" >"$OUT"

# --------------------------------------------------------------------------
# 4. Determinism self-check: regenerate to a temp file and diff. A second,
#    guarded run (KOPITIAM_SKILL_SELFCHECK=1) skips this block so there is no
#    infinite recursion. KOPITIAM_BIN is pinned so the inner run uses the same
#    binary without re-resolving or rebuilding.
# --------------------------------------------------------------------------
if [ -z "${KOPITIAM_SKILL_SELFCHECK:-}" ]; then
	tmp="$(mktemp 2>/dev/null || echo "${TMPDIR:-/tmp}/kopitiam_skill_selfcheck.$$")"
	KOPITIAM_SKILL_SELFCHECK=1 KOPITIAM_BIN="$BIN" bash "$0" "$tmp"
	if diff -q "$OUT" "$tmp" >/dev/null 2>&1; then
		echo "gen-kopitiam-skill: determinism self-check PASSED (byte-identical regeneration)" >&2
		rm -f "$tmp"
	else
		echo "gen-kopitiam-skill: determinism self-check FAILED — output is not reproducible:" >&2
		diff "$OUT" "$tmp" >&2 || true
		rm -f "$tmp"
		exit 1
	fi
fi

echo "gen-kopitiam-skill: wrote $OUT" >&2

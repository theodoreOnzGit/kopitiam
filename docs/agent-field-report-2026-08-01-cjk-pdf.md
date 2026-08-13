# Field report → work order — `pdf2md` on CJK annotated PDFs (2026-08-01)

Written by Claude after using `kopitiam pdf2md` for real work: five Cambridge
IGCSE Japanese (0716) specimen papers — born-digital, furigana (ruby) annotated,
some subset-embedded fonts. Formatted as a dispatch document in the style of
`kopitiam_token_max.md` so it can be handed to Claude Code directly.

**Corpus:** `729148` (10pp), `729150` (18pp), `729152` (10pp), `729154` (6pp),
`730209` (12pp). Command: `kopitiam pdf2md <in> -o <out> --index`. All exited 0,
all reported `Status: PASS`, `recovery_ratio: 1.0`.

---

## 0. Corrections to the first draft of this report

The first draft was written from output alone, before reading any source. Two
claims in it were **wrong**, and are corrected here. Recorded because §2.1 of the
work order warns about exactly this failure mode, and because it is the reason
every card below carries a confidence level.

| First-draft claim | Reality | Evidence |
|---|---|---|
| "The ratio is word-count based; CJK has no whitespace so it degenerates" | **Already fixed.** Validation uses non-whitespace *character* counts, with a doc comment explaining that tokenization is a PDF artifact and char counts are immune to it | `crates/kopitiam-document/src/validation/mod.rs:189` `content_char_count`, doc comment at `:193-205`, references `kopitiam-wwr` |
| "Per-page ratios are not computable" | **Already shipped.** Page anchors land in output and per-page recovery exists | `validation/mod.rs:113` `per_page_recovery`, `:177` `parse_page_anchor`; `<!-- page N -->` present in all five outputs |
| "The `���` runs are undecodable *image* bytes" | **Font mapping failure**, not images. U+FFFD is the deliberate graceful fallback when a CID→Unicode lookup misses | `crates/kopitiam-pdf/src/mupdf/font.rs:554`, `agl.rs:40` `REPLACEMENT_CHARACTER` |

What survives from the first draft, and is the core finding, is narrower and
stronger than what was originally written — see §1.

---

## 1. The finding: the gate cannot fail on this input class

All five documents scored `recovery_ratio: 1.0` and `PASS` while the Japanese
body text was materially corrupted.

This is **not** the §2.1 recovery-ratio trap, and **not** the word-count problem
already solved by `kopitiam-wwr`. It is a scope limit that remains after both
fixes: recovery is a **conservation** check — it asks whether content survived
extraction → reconstruction → render. Every defect below is **additive or
substitutive**:

- **Duplication** (Defect A) inflates the extracted *and* rendered sides
  equally. A conserved-content check is structurally incapable of seeing it.
- **U+FFFD substitution** (Defect C) replaces one character with one character.
  Count preserved on both sides.
- **Ruby displacement** (Defect B) moves text within the document. Nothing is
  lost, so nothing is measured.

The ratio is behaving exactly as designed. The gap is that **nothing else is
checked**, so `PASS` reads as "the output is good" when it only means "nothing
vanished." I took it at face value and consumed all five documents; the
corruption was found later by reading the Japanese.

**Consequence to fix:** a gate that cannot fail on an input class is worse than
no gate, because it is trusted.

---

## 2. Shared contract for these cards

**2.1 No copyrighted fixtures (§2.4 of the work order applies).** The Cambridge
PDFs are free to download but are not ours to commit. Every card below specifies
a **synthetic** fixture built from primitives — a `Page` of `TextSpan`s
constructed in test code, or a tiny generated PDF. Reproduce locally against the
real corpus; commit only synthetic.

**2.2 Confidence levels.**
- **High** — defect reproduced and counted in output; fix site identified in source.
- **Medium** — defect reproduced and counted; fix site inferred, needs survey.
- **Low** — design suggestion, not a defect. Judgement call.

**2.3 Re-verify every line number before editing.** All coordinates below were
read on 2026-08-01 via `kopitiam outline` / `slice` / `grep`. They were accurate
then. They are orientation, not gospel.

**2.4 Additive-corruption principle.** Cards F-1 and F-2 share a premise worth
stating once: **conservation checks cannot detect additive corruption.** Any new
validation signal here must be an *absolute* property of the output (a count, a
density, a rate), not a comparison between two sides of the pipeline.

**2.5 Contended files.**

| File | Touched by |
|---|---|
| `crates/kopitiam-pdf/src/mupdf_extract.rs` | F-3, F-4 |
| `crates/kopitiam-document/src/reconstruction/mod.rs` | F-5 |
| `crates/kopitiam-document/src/validation/mod.rs` | F-1, F-2 |
| `apps/cli/src/main.rs` | F-7, F-8 |
| `kopitiam_skill.md` | F-6 |

---

## 3. Dispatch

| Wave | Cards | Parallel? | Rationale |
|---|---|---|---|
| 0 | F-0 | single | Survey — three cards depend on it |
| 1 | F-6 | single | Docs only, no code, land immediately |
| 2 | F-1, F-3, F-5 | parallel | Disjoint files |
| 3 | F-2, F-4 | serialize after wave 2 | F-2 shares `validation/mod.rs` with F-1; F-4 shares `mupdf_extract.rs` with F-3 |
| 4 | F-7, F-8, F-9 | one agent | All touch `main.rs` |
| 5 | F-10 | single | Depends on F-7 |

---

## Task F-0 — Survey (blocking)

**Confidence:** n/a. **Owns:** nothing; read-only.

Answer these before wave 2 dispatches. Each answer is one or two lines plus
coordinates.

1. Where exactly are `TextSpan`s constructed on the mupdf path?
   (`crates/kopitiam-pdf/src/mupdf_extract.rs:154` is the candidate; confirm, and
   confirm whether `extractor.rs:315` on the legacy path needs the same fix.)
2. Is glyph *origin* (x, y) available at that point, or only a bounding box?
   F-3 needs per-glyph origin to dedup positionally.
3. Is `font_size` available per span, and is a page-level median or modal body
   size computed anywhere? F-4 needs a body-size baseline to identify ruby.
4. At `reconstruction/mod.rs:289` (`blocks.push(Block::Heading(...))`), what
   decides heading-ness and what sets `level`? F-5 needs the predicate.
5. Does `ConversionReport` have anywhere to carry non-fatal warnings, or does
   F-1 need to add the field?
6. Does anything already collapse runs of identical adjacent glyphs? (grep for
   `dedup` found nothing in `kopitiam-pdf` / `kopitiam-document`.)

---

## Task F-1 — Validation: detect substitutive corruption (U+FFFD density)

**Confidence:** High. **Owns:** `crates/kopitiam-document/src/validation/{mod.rs,report.rs}`

**Problem.** `729152` emits runs of `���` and scores `recovery_ratio: 1.0`,
`passes: true`. U+FFFD is a legitimate graceful fallback at the font layer
(`mupdf/font.rs:554`), but by the time a document is *rendered*, a page dense in
replacement characters is a failed page and nothing says so.

**Change.** Compute replacement-character density per page (count of U+FFFD ÷
content chars). Surface in `ConversionReport` per page and document-wide. Gate:
any nonzero count emits a warning; above a threshold (suggest 1% of a page's
content chars, tunable) the page fails and the document does not `PASS`.

Per §2.4 this is an **absolute** property of the rendered side — do not implement
it as an extracted-vs-rendered comparison, which would cancel out.

**Acceptance.**
- Synthetic fixture: a `Page` of spans where 5% of characters are `\u{FFFD}` →
  document does not pass; report names the page.
- Negative fixture: a clean page with a single legitimate `\u{FFFD}` in body text
  → warns, does not fail.
- Existing tests in `validation/mod.rs` (`dropped_content_fails`,
  `repaired_hyphenation_still_passes`, …) still pass unchanged.
- Re-run the real corpus: `729152` must no longer report `PASS`.

---

## Task F-2 — Validation: detect additive corruption (duplication rate)

**Confidence:** High (defect), Medium (statistic choice).
**Owns:** `crates/kopitiam-document/src/validation/{mod.rs,report.rs}`. **After F-1.**

**Problem.** Defect A (see F-3) doubles glyphs. Conservation checks cannot see
it — both sides inflate equally.

**Change.** Add a cheap language-agnostic statistic over the *rendered* side:
rate of adjacent repeated tokens. Two candidates, pick one and justify:

- **Repeated-bigram rate** — for CJK, the fraction of positions where
  `text[i..i+2] == text[i+2..i+4]`. Catches `会話会話` and, with a space-tolerant
  variant, `会 会話 話`.
- **Repeated-word rate** — fraction of whitespace-delimited tokens equal to their
  predecessor. Catches `スーパーで スーパーで`, misses the interleaved CJK form.

The bigram form is the more general of the two; the interleaving in the observed
output (`会 会話 話` rather than `会話 会話`) is what makes the word-level check
insufficient.

Warn above a threshold. **Do not fail the build on this** — legitimate repetition
exists (`々`, `いろいろ`, tables of identical cells). This is a signal for a
human or agent to look, not a gate.

**Acceptance.**
- Synthetic fixture reproducing `会 会話 話 1: 先生が 会話を はじめます。` →
  warning raised, naming the page.
- Negative fixture with legitimate repetition (`いろいろな人が来ました。`,
  a 3-row table with a repeated cell value) → no warning.
- Real corpus: `730209` (18 occurrences) warns; a clean Latin PDF does not.

---

## Task F-3 — Extraction: positionally dedup double-struck glyphs

**Confidence:** High (defect), Medium (fix site — see F-0 Q1/Q2).
**Owns:** `crates/kopitiam-pdf/src/mupdf_extract.rs` (+ possibly `extractor.rs`)

**Problem.** PDFs render bold by stroking the same glyph twice at a small offset.
Both copies are emitted. Measured across the corpus:

| Output string | Count | File |
|---|---|---|
| `会 会話 話` | 18 | `730209` |
| `会 会話 話` | 2 | `729150` |
| `日本 本語 語` | 3 | `729154` |
| `前 半 半` | 1 | `729150` |
| `スーパーで スーパーで` | 1 | `729152` |
| `二つ 二つ` | 1 | `729152` |

**This is not CJK-specific.** It affects every bold run in every PDF; Latin
merely degrades more visibly (`bboolldd`), so it is likelier to have been noticed
and dismissed as cosmetic. In CJK it produces plausible-but-wrong strings and
**defeats grep on any phrase longer than the doubled unit**.

**Change.** At span construction, drop a glyph whose codepoint equals the
previous glyph's *and* whose origin is within ε of it, where ε is a small
fraction of font size (start at 0.15 × font_size; make it a named constant with a
comment, not a magic number). Purely positional — no language knowledge.

**Precision over recall.** Two genuinely adjacent identical characters
(`ここ`, `々`, `1000`) sit a full advance width apart, far outside ε. If origin is
unavailable per glyph (F-0 Q2), say so and stop rather than approximating with
bounding boxes.

**Acceptance.**
- Synthetic fixture: spans for `会話` where each glyph is duplicated at
  +0.05 × font_size → extraction yields `会話`, not `会 会話 話`.
- Negative fixture: `ここ` (two identical glyphs a full advance apart) →
  both retained.
- Negative fixture: Latin `bold` double-struck → `bold`.
- Real corpus: all six strings in the table above disappear; recovery ratio is
  *expected to move* — state the new value and why it is correct (per §2.1
  ratio honesty: removing duplicates legitimately reduces the extracted side).

---

## Task F-4 — Extraction: identify ruby / annotation runs

**Confidence:** High (defect), Low (fix design — needs F-0 Q3).
**Owns:** `crates/kopitiam-pdf/src/mupdf_extract.rs` + a new reconstruction module. **After F-3.**

**Problem.** Furigana — small kana above kanji giving the reading — are real text
objects at reduced size on a raised baseline. They are treated as body text and
emitted *before* the run they annotate:

```
ぶんしょう つぎの 文  章 を 読みなさい。
```

`ぶんしょう` is the reading *of* `文章`; it now sits at the head of the sentence
as a stray word, and `文章` has acquired internal spaces. Further examples:

```
はくぶつかん 町には、にんじゃの 博 物 館もあります。
い が 伊 伊賀 賀市へ ようこそ 市へ ようこそ        ← ruby + Defect A together
ねんれい せいと                                      ← a line that is ENTIRELY ruby
```

**Impact is worst precisely where this tool is most useful.** For a
language-learning corpus the ruby is the highest-value content, and it has been
detached from what it annotates.

**Change.** Detect an annotation run: font size below ~60% of the page's body
size (F-0 Q3), baseline raised relative to a larger run, horizontal overlap with
that run. Expose `--ruby=drop|inline|html`:

- `drop` (**default**) — omit. Already a large improvement over emitting inline.
- `inline` — `文章（ぶんしょう）`.
- `html` — `<ruby>文章<rt>ぶんしょう</rt></ruby>`.

**Generalise the rule in the work order:** *small text on a raised baseline that
horizontally overlaps a larger run is an annotation, not a paragraph.* The same
predicate covers superscript footnote markers, which currently become stray
digits in body text.

**Acceptance.**
- Synthetic fixture: body spans `文章` at 12pt plus spans `ぶんしょう` at 6pt,
  raised, horizontally overlapping → `drop` yields `文章`; `inline` yields
  `文章（ぶんしょう）`.
- Negative fixture: a genuine small-print footnote *below* body text with no
  horizontal overlap → retained as a paragraph.
- Ratio honesty per §2.1: `drop` removes real extracted content, so the ratio
  moves. Either exclude classified ruby from the extracted side, or document why
  the new number is correct.
- Real corpus: the standalone `ねんれい せいと` line disappears; `文章` is
  contiguous.

---

## Task F-5 — Never promote undecodable text to a heading

**Confidence:** High. **Owns:** `crates/kopitiam-document/src/reconstruction/mod.rs` (heading push at `:289`)

**Problem.** `729152` page 3 produced seven headings at six different levels:

```markdown
### ���
#### ����
## ��
# ��
#### ���
###### ���
```

**This poisons `--index`,** which keys on headings. The sidecar contains:

```json
{"text": "��", "level": 1, "line_start": 19, "line_end": 59}
```

A level-1 heading with no retrievable name claiming lines 19–59 — **two thirds of
a 59-line document**. The one deterministic navigation aid is unusable on this
file, so I read whole documents instead. Directly counter to the token-max goal,
and it fails *silently*: `--index` still writes a well-formed sidecar.

Distinct from Task I-D (collapse figure regions to captions): the problem here is
that undecodable content was classified as a heading at all.

**Change.** Invariant at the heading-construction site: **a candidate heading
whose text is empty after stripping U+FFFD, control characters, whitespace and
punctuation is not a heading.** Demote to paragraph (or drop, if it is also
empty as a paragraph). Cheap, independent of every other card, and worth
asserting regardless of what upstream produces.

Consider the same invariant on the `--index` writer as defence in depth — a
heading that survives to the sidecar with an unusable name is worse than absent.

**Acceptance.**
- Synthetic fixture: a large-font span of `\u{FFFD}\u{FFFD}\u{FFFD}` → no
  `Block::Heading` emitted.
- Negative fixture: a legitimate heading containing one `\u{FFFD}` among real
  text → still a heading, text preserved.
- Real corpus: `729152.md.index.json` contains no heading whose text is empty
  after stripping; no heading spans two thirds of the document.

---

## Task F-6 — Correct the claim in `kopitiam_skill.md`

**Confidence:** High. **Owns:** `kopitiam_skill.md`. **Land first — docs only.**

**Problem.** The skill doc instructs agents:

> Read the validation / recovery report: `pdf2md` prints a report comparing the
> extracted word count against the rendered Markdown word count, ending in a
> PASS/FAIL line.

Two problems. It says *word* count (stale since `kopitiam-wwr`; it is character
count now), and it implies PASS means the output is good. Agents follow this
literally — I did.

**Change.** State what PASS does and does not mean: **PASS means no content was
lost; it does not mean the output is correct.** Name the classes it cannot see
(duplication, character substitution, displacement) and advise a visual check for
CJK or annotation-heavy documents. Fix "word count" → "non-whitespace character
count". Once F-1/F-2 land, add the new warnings to the same paragraph.

**Acceptance.** An agent reading only the skill doc would not conclude that PASS
means the output is trustworthy. Cheapest fix on this list; do it first.

---

## Task F-7 — `pdf2md --headings-only`

**Confidence:** Low (design). **Owns:** `apps/cli/src/main.rs`

**Problem.** Facing five unfamiliar PDFs, there was no way to learn their shape
without converting all five in full and reading them. `--pages` (shipped) helps
once you know where to look; nothing helps you *find out*.

**Change.** Emit only the heading tree with page and line ranges — the `--index`
content, without the body. Should cost a few dozen tokens against a document
whose full conversion is thousands.

**Acceptance.** On a multi-heading fixture, output contains every heading, no
body paragraph, and page numbers matching `--index`. Depends on F-5, or it
faithfully reproduces garbage headings.

---

## Task F-8 — Cross-document boilerplate stripping

**Confidence:** Low (design). **Owns:** `apps/cli/src/main.rs` + new module

**Problem.** The five papers share ~200 words of identical front matter
(instructions, "Dictionaries are not allowed", the copyright block) plus an
identical per-page footer — roughly 1,000 duplicated words across the batch, paid
for five times.

Task I-C strips *intra*-document running heads by signature recurrence across
pages. The extension is *inter*-document.

**Change.** Accept multiple inputs in one invocation. Shingle-hash paragraphs,
drop those appearing verbatim in a majority of documents, write them once to a
shared `boilerplate.md` alongside the outputs.

```bash
kopitiam pdf2md *.pdf --out-dir markdown/ --strip-shared-boilerplate
```

Secondary benefit: makes batch conversion first-class. Today the skill doc's
recipe is a shell `for` loop, so nothing can reason across the set.

**Acceptance.** Three synthetic PDFs sharing a header paragraph → it appears in
`boilerplate.md` and in none of the three outputs. A paragraph in only one
document is never stripped.

---

## Task F-9 — `tokens --tree` should exclude vendored trees by default

**Confidence:** High (reproduced this session). **Owns:** `apps/cli/src/main.rs`

**Problem.** Orienting in this repo, `kopitiam tokens crates/ --tree` reported
**74,135,961 tokens across 22,673 files** — of which 66,959,903 (90%) was
`kopitiam-ai/vendor/OpenFOAM-dev`, mostly mesh geometry. The output's first 25
lines were entirely vendored CFD tutorials. The tool built to make budgeting
cheap gave an answer dominated by files no agent will ever read, and I had to
re-run with explicit paths.

**Change.** Default-exclude `vendor/`, `target/`, `node_modules/`, `.git/`; add
`--include-vendor` to restore. Alternatively report them as a single collapsed
line (`vendor/ 66,959,903 tokens (20,813 files) [excluded]`) so the number is
visible without dominating. Prefer the collapsed line — it is honest and cheap.

**Acceptance.** `tokens crates/ --tree` surfaces the real source tree in the
first screen; vendored totals shown as one line; `--include-vendor` restores
current behaviour.

---

## Task F-10 — Extend `outline` / `slice` to prose documents

**Confidence:** Low (design), but **highest leverage on the stated goal**.
**Owns:** `apps/cli/src/main.rs`, shared with the `--index` writer. **After F-7.**

**Problem.** The token-max loop (`tokens → outline → refs → slice`) is
Rust/rust-analyzer-only. Follow-up P6 plans Python — more *languages*, still the
code half.

The gap hit this session was different: **the same loop applied to converted
Markdown.** After `pdf2md` there is no `outline` for the result. I fell back to
reading whole documents — the exact behaviour the loop exists to prevent.

**Change.** Make `kopitiam outline <file.md>` return the heading skeleton with
line ranges. The `--index` sidecar already computes this; it is not exposed
through the same verb. Confirm `slice` works on non-Rust files (it appeared to
this session) and document it.

**Why this over more languages.** The product has two halves — documents and
code. The deterministic navigation loop exists only for code. Unifying `outline`
across both makes the document half navigable **with no new analysis machinery**,
because the data structure is already being written to disk. Adding Python
extends the half that already works; this fixes the half that has nothing.

**Acceptance.** `kopitiam outline paper.md` returns headings + line ranges
matching `paper.md.index.json`. `kopitiam tokens paper.md && kopitiam outline
paper.md && kopitiam slice paper.md 40-80` works end to end on a converted PDF.

---

## Ranked summary

| Card | Confidence | Value | Cost |
|---|---|---|---|
| F-6 skill-doc correction | High | High — prevents misplaced trust today | Trivial |
| F-5 no garbage headings | High | High — restores `--index` | Small |
| F-1 U+FFFD density gate | High | High — gate can fail again | Small |
| F-3 double-struck dedup | High | High — corrupts all bold, every language | Medium |
| F-9 `tokens --tree` vendor | High | Medium — the budgeting tool mis-budgets | Trivial |
| F-10 `outline` for prose | Low | High — extends the loop to the other half | Medium |
| F-2 duplication warning | High | Medium — catches F-3 regressions | Small |
| F-4 ruby detection | High | Medium (High for CJK users) | Large |
| F-7 `--headings-only` | Low | Medium | Small |
| F-8 shared boilerplate | Low | Medium | Medium |

## Reproduction

The five PDFs are free downloads from Cambridge International (0716 specimen
papers, for examination from 2027). Per §2.4 they are **local reproduction only**
— commit synthetic fixtures as specified in each card.

```bash
for f in *.pdf; do kopitiam pdf2md "$f" -o "markdown/${f%.pdf}.md" --index; done
grep -c "会 会話 話" markdown/*.md                    # F-3
grep -nE "^#{1,6} [^[:alnum:]]*$" markdown/*.md       # F-5
python3 -c "import json;print(json.load(open('markdown/x.md.index.json')))"  # F-5
kopitiam pdf2md x.pdf -o /tmp/x.md --report-json | jq '.recovery_ratio, .passes'  # F-1
```

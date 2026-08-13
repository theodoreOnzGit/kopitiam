---
title: '[pdf2md] Tracking: CJK / annotated-PDF field report — the validation gate cannot fail on this input class'
labels: bug
---

Field report from using `kopitiam pdf2md` on five Cambridge IGCSE Japanese (0716) specimen papers — born-digital, furigana (ruby) annotated, some subset-embedded fonts.

**All five reported `recovery_ratio: 1.0`, `passes: true`, `Status: PASS` while the Japanese body text was materially corrupted.**

Full write-up: `docs/agent-field-report-2026-08-01-cjk-pdf.md`.

## §0 — Corrections to the first draft

The first draft of this report was written from output alone, before reading source. Three claims were **wrong**. Recorded because §2.1 of the work order warns about exactly this failure mode, and because it is why every card carries a confidence level.

| First-draft claim | Reality | Evidence |
|---|---|---|
| "Ratio is word-count based; CJK has no whitespace so it degenerates" | **Already fixed.** Non-whitespace *character* counts, with a doc comment explaining tokenization is a PDF artifact | `crates/kopitiam-document/src/validation/mod.rs:189`, doc comment `:193-205`, refs `kopitiam-wwr` |
| "Per-page ratios not computable" | **Already shipped** | `validation/mod.rs:113` `per_page_recovery`, `:177` `parse_page_anchor` |
| "The `���` runs are undecodable *image* bytes" | **Font CID→Unicode mapping failure.** U+FFFD is the deliberate graceful fallback | `crates/kopitiam-pdf/src/mupdf/font.rs:554`, `agl.rs:40` |

## §1 — The finding

This is **not** the §2.1 recovery-ratio trap, and **not** the word-count problem solved by `kopitiam-wwr`. Recovery is a **conservation** check: did content survive extraction → reconstruction → render. Every defect found is **additive or substitutive**:

- **Duplication** (#F-3) inflates extracted *and* rendered sides equally — structurally invisible to a conservation check.
- **U+FFFD substitution** (#F-1) replaces one character with one character. Count preserved on both sides.
- **Ruby displacement** (#F-4) moves text within the document. Nothing lost, nothing measured.

The ratio behaves exactly as designed. The gap is that **nothing else is checked**, so `PASS` reads as "output is good" when it only means "nothing vanished."

**A gate that cannot fail on an input class is worse than no gate, because it is trusted.** I took it at face value and consumed all five documents; the corruption was found later by reading the Japanese.

## §2.4 — Shared principle for validation cards

**Conservation checks cannot detect additive corruption.** Any new validation signal must be an **absolute** property of the output — a count, a density, a rate — not a comparison between two sides of the pipeline.

## §2.1 — No copyrighted fixtures

The Cambridge PDFs are free to download but not ours to commit. Every card specifies a **synthetic** fixture built from `TextSpan` primitives. Reproduce locally against the real corpus; commit only synthetic.

## §2.3 — Re-verify line numbers

All coordinates read 2026-08-01 via `kopitiam outline` / `slice` / `grep`. Orientation, not gospel.

## Dispatch

| Wave | Cards | Parallel? |
|---|---|---|
| 0 | F-0 survey | single (blocking) |
| 1 | F-6 skill doc | single — docs only, land immediately |
| 2 | F-1, F-3, F-5 | parallel (disjoint files) |
| 3 | F-2, F-4 | after wave 2 (shared files) |
| 4 | F-7, F-8, F-9 | one agent (all touch `main.rs`) |
| 5 | F-10 | after F-7 |

**Contended files:** `mupdf_extract.rs` (F-3, F-4) · `validation/mod.rs` (F-1, F-2) · `main.rs` (F-7, F-8, F-9)

## Reproduction

```bash
for f in *.pdf; do kopitiam pdf2md "$f" -o "markdown/${f%.pdf}.md" --index; done
grep -c "会 会話 話" markdown/*.md                    # F-3
grep -nE "^#{1,6} [^[:alnum:]]*$" markdown/*.md       # F-5
kopitiam pdf2md x.pdf -o /tmp/x.md --report-json | jq '.recovery_ratio, .passes'  # F-1
```

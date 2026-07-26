# KOPITIAM 0.2.2 — release notes

> **Status: not yet published.** These notes cover the work landed on `main`
> since 0.2.1; the version has not been tagged or released. Dates and the exact
> commit set are in `git log`; provenance for every ported subsystem is in
> `docs/ACKNOWLEDGEMENTS.md` and `docs/port-ledger.md`.

This is a large wave, dominated by two threads: the pure-Rust PDF **rendering**
engine reaching a usable viewer, and a "token-max" push to make the CLI cheap for
an AI agent to drive. Grouped by area below.

## PDF rendering — a pure-Rust rasterizer and viewer

The `kopitiam-pdf` MuPDF port grew from text extraction into a near-complete
rendering vertical, all pure Rust with no C toolchain:

* **Draw device + rasterizer.** A `Pixmap` raster target and an anti-aliased
  polygon fill (Global-Edge-List scan conversion, nonzero + even-odd winding),
  path flattening and stroking, and an inverse-mapped image blit — the pixel
  counterpart to the existing structured-text device.
* **The content interpreter now drives the draw device** — paths, colour, and
  images from the PDF content stream paint straight into a `Pixmap`.
* **Real letterforms.** Embedded-font **glyph-outline** decoding: TrueType
  `glyf` and CFF / Type2 charstrings, so the viewer fills actual glyph shapes
  instead of advance boxes. Re-implemented from the OpenType and Adobe specs —
  no FreeType (see AID-0052).
* **Embedded images.** DCTDecode (JPEG) and the 1/2/4/8/16-bpc / indexed-palette
  image paths decode for both the viewer and the OCR layer.
* **A terminal PDF viewer** in the CLI/TUI at "tdf-parity" — rasterised pages via
  `ratatui-image` (kitty / sixel / iTerm2, half-block fallback on Termux).

## Token efficiency — a CLI an agent can drive cheaply

The "token-max" work makes extraction output smaller and adds agent-facing
semantic/token subcommands so an AI caller reads summaries, not whole files:

* **`pdf2md` surface:** `--report-json` (machine-readable run report), a sidecar
  `--index`, `--pages` range selection, and `--split-by` for chunked output.
* **Cleaner extraction:** control characters stripped at the extraction boundary
  (no more NUL-broken `grep`), page anchors + per-page recovery, running-head /
  footer / bare-page-number stripping, figure-label soup collapsed to captions,
  and table detection that emits the longest valid prefix.
* **Semantic / token subcommands:** `refs` / `def` / `sig` / `outline` for code
  navigation, `tokens` for token-count estimation (`kopitiam-tokenizer`), and
  `check` — with `--compact`, `translate`, `digest`, and `port` modes — plus an
  outline/skeleton engine, a cached architecture digest, and persistent
  conclusion memory (content-hash invalidated).

## OCR and translation

* **OCR stack:** the pure-Rust Tesseract LSTM engine (eng / chi_sim / jpn) with
  Leptonica image preprocessing runs end-to-end as the no-embedded-text fallback.
* **Translation stack:** local-first two-pass translation, deterministic
  skeleton-first generation, segment IDs + translation memory (content-hash
  invalidation), terminology-glossary enforcement, and bilingual aligned output
  for targeted review.

## Tooling and platform

* **`kopitiam-bds`** — beads-rs (the `bd` issue tracker) forked into the
  workspace and ported to **Windows** (the reason for the fork) and
  **Android/Termux**. Every AID is filed as a `bd` issue, so this unblocks the
  tracker on the maintainer's platforms.
* **kmux LaTeX workflow** — a split-pane live-preview: `kvim` editing on one side,
  a PDF viewer on the other, recompiling on save.
* **Provenance tooling** — a deterministic port ledger (`docs/port-ledger.md`,
  162 source→target units across MuPDF / Tesseract / Leptonica / pdf-to-markdown)
  generated from the point-of-use provenance headers.

## Provenance

New upstreams credited this wave in `docs/ACKNOWLEDGEMENTS.md`: the MuPDF
draw-device / rasterizer / glyph / image ports (Artifex, AGPL-3.0, `19f1284`),
the `zune-jpeg` and `miniz_oxide` pure-Rust codec substitutions, the `tdf` viewer
parity reference (AGPL-3.0), the `beads-rs` fork (MIT → `kopitiam-bds`), and the
`nucleo` / `ratatui-image` shipped dependencies. Two new decision records:
**AID-0051** (MuPDF port conventions) and **AID-0052** (why the port substitutes
pure-Rust crates / spec re-implementations for MuPDF's zlib / libjpeg / FreeType
delegations).

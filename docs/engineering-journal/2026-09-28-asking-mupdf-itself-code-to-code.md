# Asking MuPDF itself: the 0.4.2 code-to-code sweep

**Date:** 2026-09-28
**About:** `crates/kopitiam-pdf` 0.4.2. Covers `examples/mupdf_oracle.rs`, `scripts/mupdf-feature-corpus.py`, `tests/mupdf_parity.rs`, `tests/repair.rs`, and most of `src/mupdf/`.
**Related:** [Scanned paper open blank](2026-09-28-scanned-paper-blank-indirect-contents.md) · [The guard you drop when porting](2026-08-25-dropped-guards-in-a-port.md) · [Silent wrongness and the reference oracle](2026-07-28-silent-wrongness-and-the-reference-oracle.md)
**Tracks:** bd-x0b / gh-111 · **Numbers:** [`docs/mupdf-code-to-code.md`](../mupdf-code-to-code.md) · **Map:** [`docs/mupdf-port-coverage.md`](../mupdf-port-coverage.md)

## What happen

We found the 0.4.1 bugs one at a time, from one user's scanned paper.
Maintainer's ask after that: "exhaustively translate mupdf to kopitiam-pdf,
with code to code verification". The approach was the same one that made the
llama.cpp oracle work. Do not ask a reader whether the Rust looks like the C.
Build the upstream program from the pinned commit (`mutool`, 19f1284), give
it and the port the **same file**, and diff the answers at three levels:
objects and stream bytes, structured text, and pixels.

Then work in order:

1. A read-only coverage audit, one row per MuPDF C file, stating what the
   user sees when that file is missing.
2. A harness baseline on openly licensed PDFs (NRC reports, CC BY PHYSOR
   papers, the maintainer's own arXiv paper).
3. Fixes in tranches. Each tranche is prioritised by what the harness
   shows, each fix comes with a synthetic test that fails on the tree
   before it, and each lands as its own commit.

## The numbers, before and after

Both columns were measured with the **final** harness (`mutool -N -M 0`,
per-channel gross metric) on the same 1368 open-corpus pages:

| | 0.4.1 | 0.4.2 |
|---|---|---|
| object kinds (42,771 objects) | 0 mismatches | 0 mismatches |
| text pages passing | 1245 / 1368 | **1368 / 1368** |
| chars missing / extra (of 2,023,821) | 974 / 412 | **0 / 0** |
| raster pages within 1 % | 1099 / 1368 | **1368 / 1368** |
| synthetic feature files passing | 2 / 23 | **33 / 34** (the miss is font shape, 1.84 %) |

The last text page to pass was NUREG/CR-7289 p. 2. Its 144 extra chars sat in
tagged spans whose ActualText is `()`. That ActualText lives on the
**structure element** the span's `/MCID` points to, not in the BDC. Porting
ActualText without MuPDF's `lookup_mcid` changed nothing on that page. The
harness showed it at once, and the structure-tree lookup closed it.

## What was ported

The tranches, in commit order. Each has a section in the code-to-code doc.

- **Structured text:** one-to-many ToUnicode fillers, the no-glyph/`Mn` pen
  rule, the glyph sign that turns fake-bold dropping back on, and
  presentation-form decomposition. Every "fi" of the TeX papers had been
  coming out as "f".
- **Images:** MuPDF's gridfit, subsample and scale pipeline plus the affine
  painters. JPX via hayro-jpeg2000 and JBIG2 via hayro-jbig2.
- **Graphics state:** `gs`, dashes, caps, joins, stroke text, and MuPDF's
  no-ICC CMYK.
- **CropBox is the page.** Maintainer report: "my iaea tecdoc document is
  unusually wide". The MediaBox was a two-page spread; the CropBox was the
  page.
- **Stencils, inline images, and generation-correct stream decryption.**
- **Clipping:** real masks, text clip, Form `/BBox`, and `gbot`.
- **Functions, colour spaces, shadings (all seven types), optional
  content, Type3.**
- **Tiling patterns:** both branches of `pdf_show_pattern`, including the
  draw device's tile cache.
- **Repair:** `pdf-repair.c`, plus the endstream filter.
- **Image `/Mask`:** colour keys and stencil mask streams.
- **AES-256** (`/R 5`, `/R 6`).
- **ActualText**, including the MCID structure-element lookup.
- **Transparency**: all 16 blend modes, luminosity/alpha soft masks with `/BC` and `/TR`, isolated and non-isolated groups (real `group_alpha`).

## The lessons, hor

**1. Half the early "divergences" were configuration, not code.** The first
harness run compared `mutool draw -F stext` against our plain extraction.
But mudraw switches on COLLECT_STYLES, and with that MuPDF merges text
printed twice. So we were measuring an option, not a port. The same
happened twice more:

- luma-only "gross" pixels hid a red-for-green error;
- a default `mutool draw` does ICC and spot simulation, which the port
  does not have.

Each time the fix was to make the ORACLE's configuration match what the port
claims to translate, and to write the amendment down before any number
depended on it. Never loosen the gate.

**2. Integer placement is behaviour.** My first tiling-pattern cut ran the
pattern cell once per repeat at its exact sub-pixel position. That seemed
"more correct", and it came out 3.9 % gross off MuPDF on a rotated pattern.
MuPDF draws the cell once and pastes copies at `(int)` offsets
(`dest->x = ttm.e`). Porting that truncation took the page to MuPDF's own
floor. That cut was withdrawn, not tuned.

**3. The rule you miss is a guard, again.** The AES-256 key derivation
formulas went in correctly the first time. What was wrong was the fixture
that should have been refused: user password `secret`, empty owner password.
My code opened it and mutool refused it. The reason is three lines in
`pdf_authenticate_password`: "To match Acrobat, we choose not to allow an
empty owner password, unless the user password is also the empty one". A
missing limit, not a wrong formula. That is the same lesson as the
dropped-guards entry, and it was caught the same way, by running upstream on
the same bytes.

**4. A corpus that passes stops being evidence.** After the image tranche,
every open-corpus page rendered inside 1 %. That was not because the port was
done: those 12 files simply do not contain shadings, patterns, dashes or
stencils in any size that matters. From then on the harness also ran
`scripts/mupdf-feature-corpus.py`, where each file exercises one feature over
a big area, so a missing feature fails loudly. At 0.4.1 only 2 of those
files passed.

**5. Doc claims rot in both directions.** The audit found fourteen doc
claims that the code contradicts. The worst was in `docs/port-ledger.md`,
which is machine-generated from headers. It listed the Adobe CJK CMaps as
**ported** because the generator counts any backticked path in a header, and
this one sat inside a "Not ported" section. It is fixed in the header, with a
note explaining why that path has no backticks.

## What downstream (kovan etc.) needs to know

- **Page geometry is now the CropBox.** `page_size_points`,
  `page_media_box_points`, the raster size and stext coordinates all
  report the cropped page, as MuPDF does. A caller that assumed MediaBox
  coordinates sees a shift on cropped pages; that shift is the fix.
- **Behaviour changes on existing APIs** (the 0.4.0 precedent would call
  these minor-bump territory; the maintainer named 0.4.2):
  - `page_images` / `page_full_image` now return pixels for JPX and JBIG2
    images instead of Unsupported;
  - damaged files open (repair) where `PdfDocument::open` used to fail;
  - AES-256 files open.
- **New public items, all additive:**
  - defaulted `TextDevice` methods (clip, shade, tile, stroke glyph,
    filler, alpha, image mask, group, mask, ActualText, flush_text);
  - `draw_path::StrokeStyle`;
  - `page_run::{page_transform, page_bounds}`;
  - `PdfDocument::{xref_len, was_repaired}`;
  - `Font::{is_type3, type3_info}` and `Type3Info`;
  - the new modules (`function`, `shade`, `layer`, `repair`, `draw_scale`,
    `draw_affine`, `draw_blend`, `op_transparency`, `marked_content`).
- `ColorSpace` stays crate-internal.
- `crypt::EncryptDict` gained `oe`/`ue` fields. This is a breaking change
  for anyone constructing it by hand. Nothing in the workspace does, since
  it is built from the `/Encrypt` dict inside `xref.rs`.
- kovan (outram-park-backend) uses `rasterize_page`, `page_to_stext`,
  `StextLine`, `PdfDocument`, `incremental_update`,
  `page_edit::locate_page_slot` and `gui_frontend::PdfReader`. None of
  their signatures changed, so its bump is `0.4.1` -> `0.4.2` in its
  Cargo.toml and nothing else.
- One thing to watch: MuPDF refuses incremental saves on a REPAIRED file,
  but `write.rs` still appends, with `/Prev` pointing at the broken xref.
  The result reopens fine here, because repair runs again and the last
  definition wins. Still, a caller that saves should check
  `PdfDocument::was_repaired()` first.

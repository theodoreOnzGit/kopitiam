# kopitiam-pdf vs MuPDF: the code-to-code harness

**Started:** 2026-09-28 (SGT) · **Tracks:** bd-x0b / gh-111 · **Coverage map:** [`mupdf-port-coverage.md`](mupdf-port-coverage.md)

kopitiam-pdf is a translation of MuPDF (`source/fitz` + `source/pdf`, commit
`19f1284`). Reading the C beside the Rust only catches what the reader thinks
to look for -- the 0.4.1 blank-scan bug (an indirect `/Contents` array never
run) sat there for months because no fixture had that shape. So this harness
asks the upstream program itself, on the **same file**, and diffs the answers.
Same idea as the llama.cpp reference oracle ([`REFERENCE-ORACLE.md`](REFERENCE-ORACLE.md)),
applied to the PDF port.

## The oracle: `mutool` built from the pinned commit

The comparison is only meaningful against the MuPDF the port was translated
*from*, not whatever `mutool` a distro ships. Build it once, into the
gitignored vendor tree (read-only reference; never linked, never shipped):

```sh
cd crates/kopitiam-pdf/vendor
git clone https://github.com/ArtifexSoftware/mupdf.git   # or a local mirror
cd mupdf && git checkout 19f1284 && git submodule update --init
make -j16 build=release HAVE_X11=no HAVE_GLUT=no HAVE_CURL=no html=no extract=no tools
build/release/mutool -v    # mutool version 1.29.0
```

Build notes, so nobody has to rediscover them:

* `html=no extract=no` drops the HTML/EPUB engine (gumbo, harfbuzz for HTML
  layout, cmark) and the docx writer. None of them are on the PDF
  parse/render/extract path, and at `19f1284` the pinned `extract` and
  `gumbo-parser` submodule commits are no longer fetchable from the local
  mirror used here.
* The `mujs` checkout used on this machine is newer than the pinned submodule
  (the pinned one was not in the mirror either). mujs is only the JavaScript
  interpreter behind `mutool run`; it cannot change any PDF result, only how
  the object-walk script below runs.
* Everything that CAN change a PDF result -- `source/fitz`, `source/pdf`,
  FreeType, libjpeg, zlib, openjpeg, jbig2dec, lcms2 -- is exactly at
  `19f1284` and its pinned submodules.

## What runs, and the pass criteria (fixed BEFORE the first measurement)

```sh
cargo build --release -p kopitiam-pdf --example mupdf_oracle
target/release/examples/mupdf_oracle \
    --mutool crates/kopitiam-pdf/vendor/mupdf/build/release/mutool \
    [--dpi 72] [--max-pages N] [--tsv per-page.tsv] [--show-unmatched N] FILE.pdf ...
```

Three layers per file:

| Layer | MuPDF side | kopitiam side | PASS criterion | Why that criterion |
|---|---|---|---|---|
| **Objects + stream decode** | `mutool run` script: for every object `1..countObjects()`, its kind (null/bool/int/real/string/name/array/dict/stream) and, for a stream, the decoded bytes from `readStream` (= `pdf_load_stream`) | `PdfDocument::resolve` / `open_stream_num` on the same numbers | same kind; **byte-identical** decoded stream | Filters are lossless; any tolerance would hide a real divergence. |
| **Structured text** | `mutool draw -F stext -O accurate-bboxes=no,collect-styles=no` (so the option set is MuPDF's `FZ_STEXT_CLIP` only) | `page_to_stext` with `StextOptions::CLIP` | every non-space char of MuPDF's matched to an unused char of ours with the same code point within **1.0 pt** of its origin, none left over on either side, AND the non-space text identical in reading order | 1 pt is ~a fifth of an average advance at 10 pt: tight enough that two neighbouring identical letters cannot cross-match. Max origin error is reported, not gated. |
| **Raster** | `mutool draw -r 72 -c rgb` (PPM) | `rasterize_page_ex` -- the **native** engine, never the hayro fallback (it is the port under test) | pixels with \|Δluma\| > 128 ("gross": one side ink, other side paper) ≤ **1 %** of the page | AID-0052: our glyph outlines come from from-spec decoders / skrifa, not FreeType, and the scan converter is not a fixed-point GEL clone, so anti-aliased edges legitimately differ by partial coverage. A missing feature (a shading, an image codec, a dash pattern) moves whole regions and blows straight through 1 %. |

Why `mutool draw -F stext` needs the `-O`: mudraw switches on
`FZ_STEXT_ACCURATE_BBOXES | FZ_STEXT_COLLECT_STYLES` for every non-plain-text
format (`mudraw.c:872-878`). With COLLECT_STYLES, MuPDF's `check_for_fake_bold`
merges a block of text printed twice anywhere on the page, which plain CLIP
extraction (MuPDF's API default, and ours) keeps. Without the `-O` the diff
measures configuration, not code -- the first harness run did exactly that on
NUREG/KM-0004 p. 43 (a figure drawn twice) before this was caught.

## Inputs: openly licensed only

| Input | Pages | Licence basis |
|---|---|---|
| `reactor-literature/kovan-standard-open-corpus/nrc/*.pdf` (7 NRC reports) | 1258 | U.S. Government Work, 17 U.S.C. § 105; NRC site disclaimer (quoted in that corpus's README) |
| `reactor-literature/kovan-standard-open-corpus/physor-2026/*.pdf` (4 papers) | 32 | CC BY 4.0 per each paper's Zenodo DOI record (README lists authors + DOIs) |
| `testdata/arxiv-2608.17504v1.pdf` | 78 | the maintainer's own paper, committed with their say-so (testdata/README.md); the only input with embedded Type1 + CID-CFF programs |

The corpus lives in the outram-park-backend repo's `reactor-literature`
submodule; nothing from it is copied into kopitiam. The PDFs are runtime
arguments only. Proprietary literature is never fed to this harness for
recorded results, and never committed.

## Results

Total: 1368 pages, 12 files. `mean |Δluma|` is the page-average over the file.

### Baseline -- kopitiam-pdf 0.4.1 (`9066d1b`), measured 2026-09-28

| file | pages | objects kind-mismatch | streams identical | stream diffs by filter | text pages pass | chars missing / extra (of MuPDF's) | order-mismatch pages | max origin err (pt) | raster pages pass (≤1% gross) | worst gross % (page) | mean \|Δluma\| | fallback glyphs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ML070740002.pdf | 36 | 0/258 | 149/149 | - | 36/36 | 0 / 0 (of 76931) | 0 | 0.000 | 36/36 | 0.07 (1) | 3.94 | 0 |
| ML12338A215.pdf | 279 | 0/8727 | 1095/1117 | CCITTFaxDecode(err):3 DCTDecode(err):19 | 279/279 | 0 / 0 (of 289775) | 0 | 0.000 | 274/279 | 4.57 (275) | 3.62 | 227 |
| ML13028A421.pdf | 56 | 0/1154 | 377/377 | - | 56/56 | 0 / 0 (of 61258) | 0 | 0.000 | 55/56 | 4.15 (51) | 2.98 | 103 |
| ML13325A086.pdf | 349 | 0/24808 | 918/920 | CCITTFaxDecode(err):2 | 328/349 | 209 / 234 (of 420628) | 4 | 0.595 | 347/349 | 4.37 (345) | 2.24 | 885 |
| ML15334A199.pdf | 228 | 0/979 | 244/473 | CCITTFaxDecode(err):228 DCTDecode(err):1 | 228/228 | 0 / 0 (of 455536) | 0 | 0.000 | 34/228 | 8.72 (74) | 12.72 | 0 |
| ML16245A032.pdf | 101 | 0/3884 | 319/323 | CCITTFaxDecode(err):4 | 101/101 | 0 / 0 (of 169431) | 0 | 0.000 | 99/101 | 7.03 (2) | 3.62 | 746 |
| ML22063A060.pdf | 209 | 0/1108 | 332/548 | JBIG2Decode(err):5 JPXDecode(err):211 | 195/209 | 33 / 177 (of 309291) | 1 | 0.201 | 170/209 | 11.34 (114) | 5.24 | 2 |
| physor2026-206 | 8 | 0/94 | 33/49 | DCTDecode(err):16 | 8/8 | 0 / 0 (of 14515) | 0 | 0.000 | 8/8 | 0.05 (1) | 3.41 | 0 |
| physor2026-306 | 8 | 0/276 | 78/81 | DCTDecode(err):3 | 0/8 | 37 / 0 (of 21758) | 8 | 0.000 | 8/8 | 0.36 (3) | 4.62 | 0 |
| physor2026-343 | 8 | 0/67 | 20/30 | DCTDecode(err):10 | 8/8 | 0 / 0 (of 20304) | 0 | 0.000 | 8/8 | 0.35 (5) | 4.78 | 0 |
| physor2026-449 | 8 | 0/186 | 44/45 | DCTDecode(err):1 | 0/8 | 60 / 0 (of 13847) | 8 | 0.000 | 7/8 | 1.37 (2) | 5.48 | 0 |
| arxiv-2608.17504v1 | 78 | 0/1230 | 145/145 | - | 6/78 | 635 / 1 (of 170547) | 72 | 0.000 | 77/78 | 1.31 (26) | 3.34 | 0 |
| **total** | **1368** | **0/42771** | **3754/4257** | | **1245/1368** | **974 / 412 (of 2023821)** | **93** | | **1123/1368** | | | |

What the baseline says, in one breath each:

* **Parsing is already there lah.** Zero object-kind mismatches over 42,771
  objects; every non-image stream (Flate, LZW, ASCIIHex/85, predictors, object
  streams, xref streams) decodes byte-identical. The only stream diffs are the
  image codecs, where `open_stream` stops and `page_image.rs` takes over
  (MuPDF's `pdf_load_stream` runs the codec).
* **Text positions match to the last digit** wherever a char exists on both
  sides (max origin error 0.000 pt on 10 of 12 files). The misses are control
  flow, not arithmetic: one-to-many ToUnicode fillers dropped (every "fi" of
  the TeX/InDesign papers came out "f"), the no-glyph/combining-mark pen rule
  missing, fake-bold suppression disabled, ActualText ignored.
* **Raster** is within 1 % on 82 % of pages. The failures cluster on image
  codecs (JPX blank on NUREG/CR-7289, JBIG2 blank) and on the WASH-1400 CCITT
  scan (1-bit image downscaling, bd-6lx), plus a handful of vector pages still
  to be diagnosed.

### After tranche 1 (structured text) -- measured 2026-09-28

Fixes (all in `pdf-op-run.c` / `stext-device.c` control flow; the formulas
were already right):

1. **One-to-many ToUnicode fillers** (`pdf_show_char`, pdf-op-run.c:1449): the
   2nd+ code points of a ToUnicode entry are now shown as zero-advance filler
   chars (new defaulted `TextDevice::show_filler_char`; the draw device paints
   nothing for them, same as MuPDF's `gid = -1`).
2. **The no-glyph / non-spacing-mark pen rule** (stext-device.c:850-858):
   fillers, ligature tails and `Mn` combining marks sit on the pen and do not
   move it.
3. **`glyph` sign convention**: real glyphs now carry `glyph >= 0` into
   `add_char_imp`, which re-enables MuPDF's fake-bold overprint drop (it was
   silently off because every glyph arrived as `-1`).
4. **Presentation-form decomposition** (stext-device.c:1097-1104).

Only the text columns moved; the raster and object columns are identical to
the baseline row for row, as they should be.

| file | text pages pass (before -> after) | chars missing / extra, after (of MuPDF's) | max origin err (pt) |
|---|---|---|---|
| ML13325A086.pdf | 328 -> **349/349** | 0 / 0 (of 420628) | 0.000 |
| ML22063A060.pdf | 195 -> 208/209 | 0 / 144 (of 309291) | 0.000 |
| physor2026-306 | 0 -> **8/8** | 0 / 0 | 0.000 |
| physor2026-449 | 0 -> **8/8** | 0 / 0 | 0.000 |
| arxiv-2608.17504v1 | 6 -> **78/78** | 0 / 0 (of 170547) | 0.000 |
| every other file | unchanged, all pages pass | 0 / 0 | 0.000 |
| **total** | 1245 -> **1367/1368** | **0 / 144 (of 2023821)** | **0.000** |

The one remaining page (NUREG/CR-7289 p. 2) is **ActualText**: the paragraph
sits in a marked-content span `/ActualText ()` (empty), which MuPDF emits
instead of the glyphs, so its text vanishes from MuPDF's stext. We ignore
marked content, so we keep the 144 glyph chars. Recorded as a gap in the
coverage map, not fixed in this tranche.

### After tranche 2 (images) -- measured 2026-09-28

Fixes:

1. **The image drawing pipeline is MuPDF's now** (`fz_draw_fill_image`,
   draw-device.c:1836): grid-fit the image matrix (`fz_gridfit_matrix`),
   box-subsample by `l2factor` (`fz_subsample_pixblock`), smooth-scale to the
   device footprint with the "simple" filter (`fz_scale_pixmap`,
   draw-scale-simple.c), then paint in 14-bit fixed point with MuPDF's
   near/bilinear decision (`fz_paint_image_imp`, draw-affine.c). New modules
   `draw_scale.rs`, `draw_affine.rs`. Before: nearest-neighbour at pixel
   centres, which shredded 1-bit scans (bd-6lx).
2. **JPXDecode** via `hayro-jpeg2000` and **JBIG2Decode** via `hayro-jbig2`
   (AID-0052 substitutions for openjpeg / jbig2dec; the exact versions hayro
   already links), with MuPDF's colour-space rule for JPX and the
   1 = black -> 0 = black inversion for JBIG2.

| file | raster pages pass (before -> after) | worst gross % (after) | mean \|Δluma\| (before -> after) |
|---|---|---|---|
| ML15334A199 (WASH-1400, CCITT scan) | 34 -> **228/228** | 0.00 | 12.72 -> **0.00** (pixel-identical) |
| ML22063A060 (NUREG/CR-7289, 211 JPX figures) | 170 -> **209/209** | 0.15 | 5.24 -> 2.88 |
| ML12338A215 | 274 -> **279/279** | 0.12 | 3.62 -> 2.31 |
| ML13028A421 | 55 -> **56/56** | 0.23 | 2.98 -> 2.01 |
| ML13325A086 | 347 -> **349/349** | 0.34 | 2.24 -> 2.16 |
| ML16245A032 | 99 -> **101/101** | 0.17 | 3.62 -> 3.00 |
| physor2026-449 | 7 -> **8/8** | 0.25 | 5.48 -> 2.84 |
| arxiv-2608.17504v1 | 77 -> **78/78** | 0.27 | 3.34 -> 2.99 |
| ML070740002, physor-206/-306/-343 | all pass before and after | <= 0.12 | small drops |
| **total** | 1123 -> **1368/1368** | **0.34** | |

WASH-1400 is worth a sentence: 228 pages of 1-bit CCITT scan now come out
**pixel-identical** to MuPDF (mean |Δluma| 0.00, zero gross pixels), because
the subsample + simple-filter + gridfit chain is integer arithmetic end to
end and the CCITT decoder was already byte-identical (AID-0058).

**The open corpus no longer discriminates rasters.** Every page is inside
1 %; the features still missing from the port (shadings, `gs` alpha, dashes,
patterns, Type3, stencil masks, inline images, ...) simply do not occur, or
occur too small, in these 12 files. From here on the harness is fed
synthetic feature PDFs as well -- see the next section.

Tests (each checked to fail on the pre-fix tree): `downscaled_image_is_filtered_not_point_sampled`
(expected value **measured with mutool**: 127 in every pixel -- the first
draft of the test assumed MuPDF would also filter a 64x4 -> 16x4 shrink, and
mutool showed it point-samples that one, because `fz_default_image_scale`
needs BOTH axes to shrink), `jpx_image_decodes`, `jbig2_image_decodes`
(fixtures from `tests/fixtures/make-image-codecs.py`, cross-rendered with
mutool before use).

## The synthetic feature corpus

`scripts/mupdf-feature-corpus.py OUTDIR` writes 18 one-feature PDFs (line
style, ExtGState, stencil masks, inline images, shadings via `sh` and via a
pattern, a circular clip path, Type3, text render modes, fill colour spaces,
CropBox, tiling pattern, Form `/BBox`, optional content, blend + soft mask,
CMYK fill, hairlines, and one deliberately damaged xref). Each feature covers
a large area, so a missing feature fails the raster gate loudly. They are
generated from the script alone -- nothing third-party, nothing committed but
the generator.

**Metric amendment (2026-09-28, before any feature-corpus number was
recorded).** The raster "gross" test was `|Δluma| > 128`. The first feature
run showed that hides hue errors: a red square where MuPDF paints green is
only 74 apart in luma, so `optional-content` "passed" while plainly wrong.
From here on a pixel is gross when **any RGB channel** differs by more than
128; the 1 % page criterion is unchanged. The open-corpus results above were
measured with the luma rule; the final table at the end of this document
re-measures everything, both corpora, with the per-channel rule.

### After tranche 3 (graphics state) -- measured 2026-09-28

Fixes: `gs` (ExtGState `LW`/`LC`/`LJ`/`ML`/`D`/`Font`/`CA`/`ca`, per
`pdf_process_extgstate`), `J`/`j`/`M`/`d`, MuPDF's dash walker
(`fz_dash_moveto`/`fz_dash_lineto`/`fz_dash_bezier`, draw-path.c), the stroke
width floor (`0.2 px`, not the invented `0.7 px`), fill/stroke alpha on paths,
text and images, text render modes 1/2/5/6 stroking the glyph outline
(`fz_stroke_text`) and mode 1/5 no longer filling, and MuPDF's no-ICC CMYK
formula `1 - min(1, c + k)`.

| feature file | gross % (tranche 2 -> 3) | mean \|Δluma\| (2 -> 3) | note |
|---|---|---|---|
| line-style | 2.08 -> **0.00** | 4.97 -> 0.07 | dashes, butt/round/square caps, miter/round/bevel joins |
| extgstate | 2.96 -> **0.00** | 32.48 -> 0.01 | `/ca` `/CA`, `/LW`, `/D`, `/LC` through `gs` |
| hairline | 0.00 -> 0.00 | 13.30 -> 1.50 | `0 w` lines now 0.2 px like MuPDF |
| text-render-modes | 37.99 -> 32.98 | 67.82 -> 56.70 | stroke + fill/stroke right; mode 7 (clip) is the clip tranche |
| cmyk-fill | 0.00 -> 0.00 | 31.00 -> 31.00 | residual is ICC: mutool converts through lcms2 + its default CMYK profile; this port has no CMS |

(Tranche-2 figures for these rows are luma-gross, tranche-3 per-channel; for
these five files the two rules give the same verdict.)

### Tranche 4 (page boxes) -- the CropBox is the page -- measured 2026-09-28

**Maintainer report, same day:** "my iaea tecdoc document is unusually wide".
The IAEA TECDOC covers (restricted documents, *not* used as inputs or
fixtures) have a MediaBox that is a two-page spread, `[0 0 1340.74 898.53]`,
and a CropBox that is the A4 right half; poppler says page 1 is
595.25 x 841.84 pt. kopitiam-pdf sized, rendered and positioned text from the
MediaBox, so kovan laid the page out 1340 pt wide.

Fix, ported from `pdf_page_obj_transform_box` / `pdf_bound_page`
(pdf-page.c:666, 742) and the CropBox clip in
`pdf_run_page_contents_with_usage_imp` (pdf-run.c:179): the page is the
CropBox intersected with the MediaBox (`/CropBox` inheritable, default the
MediaBox), `/Rotate` snapped to quarter turns, `/UserUnit` applied, the
CropBox corner moved to the origin; content is clipped to the CropBox when it
is smaller than the MediaBox. One `page_transform` feeds the raster size, the
page CTM (so stext coordinates and annotation placement), the stext page
`mediabox`, and `page_geom::{page_size_points, page_media_box_points}`.
`FZ_STEXT_CLIP` now culls glyphs entirely outside the page box, as mutool's
stext output does.

| check | 0.4.1 | now | MuPDF (mutool 19f1284) |
|---|---|---|---|
| spread-cover shape: page size | 1340 x 898 | **595 x 842** | 595 x 842 |
| same, red square inside the crop at user (850,150) | device (850, 748) | **(133, 720)** | (133, 720) |
| same, first char of "Inside" (stext) | x 800 | **x 83, y 270** | x 83, y 270 |
| same with `/Rotate 90`: raster + stext vs mutool | -- | **0.00 % gross, text 6/6 chars** | -- |
| feature corpus `feat-cropbox.pdf` | 66.67 % gross | **0.00 %** (mean 0.00) | -- |

Tests: six `cropbox_*` cases in `tests/mupdf_parity.rs` (size, raster
offset + clip, stext space, inherited from `/Pages`, larger than MediaBox,
`/Rotate 90`); five fail on the 0.4.1 tree, the sixth pins the
intersect-with-MediaBox rule.

### Tranche 5 (stencils, inline images, encrypted streams) -- measured 2026-09-28

1. **Stencil `/ImageMask`s paint the fill colour** through the mask
   (`fz_fill_image_mask` -> `fz_paint_image_with_color`, the
   `template_affine_color_N_near/_lerp` painters with MuPDF's
   `FZ_EXPAND`/`FZ_COMBINE`/`FZ_BLEND` arithmetic), after the same subsample
   + scale pipeline as images. 0.4.1 drew them as opaque black-and-white
   pictures.
2. **Inline images decode and paint** (`parse_inline_image` +
   `pdf_load_inline_image`): abbreviated keys and colour spaces
   (`/G /RGB /CMYK /I`, other names via `/ColorSpace` resources); unfiltered
   data cut at exactly `stride x H` bytes; filtered data ends at the first
   candidate `EI` (MuPDF's delimiter rule) up to which it decodes to a whole
   image -- the buffer-decoder equivalent of MuPDF letting the decoder
   consume the stream. 0.4.1 skipped every inline image.
3. **Encrypted streams use their object's generation** in the RC4/AES key
   (§7.6.2 algorithm 1, `pdf_open_crypt(num, gen)`). 0.4.1 always used 0, so
   a stream in a generation-1 object decrypted to garbage (strings were fine).
   The coverage audit found this by reading, not the harness -- no open-corpus
   file is encrypted.

| feature file / fixture | before | after |
|---|---|---|
| `feat-image-mask` | 100.00 % gross | **0.00 %** (mean 0.00: pixel-identical) |
| `feat-inline-image` | 50.00 % gross | **0.00 %** (mean 0.25) |
| `encrypted-rc4-gen1.pdf` (MuPDF-encrypted fixture) | blank page | red square, as mutool |

Tests: 4 new, all fail on the pre-fix tree.

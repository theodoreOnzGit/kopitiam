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

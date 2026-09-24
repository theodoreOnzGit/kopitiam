# AID-0058 — The `CCITTFaxDecode` port is verified byte-identical against two independent decoders

**Status:** Pending review
**Date:** 2026-09-24
**Upstream commit:** MuPDF `1ca9d1788` (1.28.0-235, released as 1.29.0)

## Context

A 1990 scanned report (Verfondern & Nabielek, *PANAMA-I*) rendered as **blank
pages** in kovan's PDF reader. The cause was not a rendering defect: every image
in the file is `CCITTFaxDecode`, and `page_image.rs` listed that filter in
`is_deferred_codec`, so `decode_image_base` returned `ErrorKind::Unsupported`
before reaching any sample. The page had no ink because nothing had been
decoded.

CCITT Group 3/4 (ITU-T T.4 / T.6) is how bilevel scans of that era are stored —
a 600-dpi A4 page is ~35 megapixels of 1-bit data compressed to tens of
kilobytes. Without it, scanned literature is unreadable in the reader.

## Decision

**Port `source/fitz/filter-fax.c` rather than write a T.4/T.6 decoder from the
specification**, consistent with AID-0056, and **verify it code-to-code against
MuPDF's own output** rather than by inspection.

Two units landed, both pinned at `1ca9d1788`:

| Unit | Target | Nature |
| --- | --- | --- |
| `source/fitz/filter-fax.c` | `src/mupdf/filter_fax.rs` | ported (state machine, `dec1d`, `dec2d`, row/EOL control flow) |
| `source/fitz/filter-fax.c` | `src/mupdf/filter_fax_tables.rs` | ported verbatim (776 Huffman table entries, machine-generated from the C) |

This is a **third** pinned MuPDF commit alongside `19f1284` (AID-0051) and
`5fe54ce` (AID-0056). It is recorded rather than harmonised: `filter-fax.c` has
not changed between those commits, and re-pinning the other 106 units to chase
a single file would be a large untested diff for no behavioural gain.

## Verification — methodology

MuPDF 1.29.0 was **compiled from source on this machine** so the reference is
this exact code, not a distribution build. The bundled third-party tree was
required: the system `lcms2` is incompatible (Artifex ships a forked
`lcms2mt.h`) and the system `gumbo` 0.13.2 is too old, so the build is
`make -j16 build=release HAVE_X11=no HAVE_GLUT=no HAVE_CURL=no tools` against
`thirdparty/`.

Corpus: **all 34 CCITT images in the 39-page PANAMA-I report**, each
7008 × 4960 (G4, `/K -1`, `EncodedByteAlign false`) — 1.16 gigapixels in total.
The PDF is proprietary and is **not** committed; the harness
(`examples/fax_verify.rs`) takes the file as a runtime argument, so the
procedure is reproducible while the corpus stays out of the repository, per
`DATA_POLICY.md`.

Three independent comparisons, each on the raw decoded bitmap:

1. **vs MuPDF itself** — `mutool extract`, the code this was ported from.
2. **vs poppler** — `pdfimages` 26.08.0, an unrelated implementation. This is
   the check that matters: agreeing with the code you copied can mean you
   copied a bug, whereas agreeing with a second lineage cannot.
3. **vs ITU-T T.6 by hand** — a 4-white/4-black row coded from the
   specification's own tables (`H` = `001`, white-4 = `1011`, black-4 = `011`),
   as a unit test that depends on neither reference binary.

## Verification — results

| Comparison | Images | Byte-identical | Differing |
| --- | --- | --- | --- |
| vs MuPDF 1.29.0 `mutool extract` | 34 | **34** | 0 |
| vs poppler 26.08.0 `pdfimages` | 34 | **34** | 0 |

Byte-identical on the decoded sample payload, all 34, both references — not
"visually equivalent" and not within a tolerance. End-to-end, the six sampled
pages now rasterise to the correct dimensions with ink present where it
belongs; row- and column-ink profiles correlate **0.94–0.995** against
`pdftoppm`. The residual difference in *total* ink is this port's sparse
downsampling at 12× reduction versus poppler's box filter — a pre-existing
`draw_device` property, not a codec one, and bounded away from the codec by the
byte-identical result above.

### The defect verification found, which inspection had not

The first run decoded **4999, 4980 and 4973 rows** for images declaring 4960,
and threw `invalid code in 2d faxd` on others. The end-of-block test had been
written as six EOLs; T.6 specifies **two** for G4 (`K < 0`), six being the G3
figure. With six, the RTC is never recognised, so the decoder runs past the
final row into whatever bytes follow. Fixing it gave 34/34.

This is the failure mode `docs/engineering-journal/2026-08-25-dropped-guards-in-a-port.md`
describes: the *formula* translated correctly and a *bound* did not. It
produced plausible output — a slightly too-tall image — rather than an obvious
crash, and no amount of reading the Rust would have revealed it. Only
comparison against the reference did.

### A second defect, found by unit-testing rather than by the corpus

Writing `g4_end_of_block_is_two_eols_not_six` exposed that MuPDF's EOL handling
is an `if / else if` **chain** and the port had flattened it into a sequence:
after consuming an EOL, control fell through into `dec1d`/`dec2d`, which reset
`eolc` — destroying the counter the end-of-block test reads — and then tried to
decode the bits after the EOL as a run code. The 34-image corpus could not
catch it, because pure G4 streams carry no embedded EOLs; only a synthetic
EOL-bearing input reaches that path. Restructured to the chain, all 34 images
re-decode byte-identically, confirming the fix is a no-op on data that never
took the broken branch.

## Consequences

- `CCITTFaxDecode`/`CCF` leave `is_deferred_codec`. The decoded bits are
  inverted into the PDF sample convention (`/BlackIs1` defaults false, so 0 is
  black) and handed to the **existing** raw-sample path, so `/ImageMask`,
  `/Decode`, `/BitsPerComponent` and the colourspace are handled once, in
  common with every other filter — mirroring MuPDF, where the fax filter is a
  stream filter and not an image loader.
- A short decode is an **error**, not a silent resize: an image that returns
  fewer rows than `/Height` would misplace everything laid out against it.
- `JPXDecode` and `JBIG2Decode` remain deferred, and still return a clear
  `Unsupported` error rather than a blank page.
- 712 `kopitiam-pdf` library tests pass, 13 of them new.

## What this does not establish

The corpus is **one document, one encoder, one mode**: 34 images, all G4
(`/K < 0`), all `EncodedByteAlign false`, all from a single 1990s scanner
toolchain. The G3 paths — `/K 0` (pure 1-D) and `/K > 0` (mixed, with the
per-row mode bit) — and `EncodedByteAlign true` are exercised **only** by unit
tests, not against any reference decoder, because no file in reach uses them.
They are ported, not verified. `UNCOMPRESSED` (the T.4 extension escape) is
rejected with a format error exactly as upstream rejects it, and so is
implemented in neither code.

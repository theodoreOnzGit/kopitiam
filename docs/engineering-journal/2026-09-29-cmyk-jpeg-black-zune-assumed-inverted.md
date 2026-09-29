# CMYK JPEG figure come out solid black: zune-jpeg assume the CMYK is inverted

**Date:** 2026-09-29
**About:** `crates/kopitiam-pdf/src/mupdf/filter_dct.rs` (new), `page_image.rs` `decode_jpeg`; kopitiam-pdf 0.4.2 -> 0.4.3; gh-116
**Related:** [The guard you drop when porting is the one you never had a fixture for](2026-08-25-dropped-guards-in-a-port.md), [asking MuPDF itself](2026-09-28-asking-mupdf-itself-code-to-code.md)

## What happen

The maintainer opened a paper in kovan and one figure was a solid black box. The paper is proprietary, so it was diagnosed locally and nothing from it is committed. The image is `/DeviceCMYK` + DCTDecode with no `/Decode`. Inside the JPEG there is an Adobe APP14 marker with transform 0 and component ids `C M Y K`, and the samples are stored plain (0 = no ink).

The first guess was an Adobe-inverted JPEG with the inversion not honoured. Wrong way round: nothing was inverted at all. zune-jpeg's default output colour space is **RGB**, so for a 4-component JPEG it did the CMYK->RGB itself as `r = c * k / 255` (`worker.rs` `color_convert_cymk_to_rgb`). That formula only works for Photoshop-inverted storage. On plain CMYK, white `(0,0,0,0)` gives 0, i.e. black. Our own `ColorSpace::CMYK` branch, the one that looked at `/Decode`, could never run, because `output_colorspace()` reported RGB.

MuPDF does it like this: `pdf-stream.c:164` says `invert_cmyk = 0` for every PDF DCT stream. `filter-dct.c:255-279` picks the colour space (the Adobe transform overrides `/ColorTransform`; 0 forces CMYK/RGB). libjpeg hands back the raw CMYK. `image.c:726` then applies `/Decode` (`fz_decode_tile`), and only after that does the colour converter run.

## Lessons

* **A substitute crate has defaults, and those defaults are a spec too.** AID-0052 substituted libjpeg's *decoder*. It never said zune-jpeg should also choose the *colour* policy, yet leaving the output colour space on its default did exactly that. When you substitute, pin every knob the upstream sets explicitly. Here, ask for the raw components.
* **The coverage map said "handled via `/Decode`".** It was true of the code as written, but that code was unreachable. Reading the branch is not the same as knowing it runs: a doc claim needs a fixture that goes through that branch.
* **No open-corpus file had a plain-CMYK JPEG,** so the harness could not see it, same as the indirect-`/Contents` bug. The fixture `tests/fixtures/dct-colour.pdf` (generator `make-dct-colour.py`) now has six shapes (plain CMYK, inverted CMYK + `/Decode`, YCCK, partial `/Decode`, `/ColorTransform 0` RGB, gray `/Decode`), with mutool's own pixels committed beside it. Before the fix 4 of the 6 were wrong, max |Δ| 255. After, max |Δ| is 0 on every pixel.

## Numbers (mutool 19f1284, `-N -M 0`)

On the paper's page at 72 dpi, gross went from 8.80 % to 0.000 % and mean |Δluma| from 23.71 to 1.81. In the figure's box our mean luma went from 1.3 to 245.4, against mutool's 244.7. Across the 1368 raster pages of the open corpus, no page moved by 0.001, even though YCbCr->RGB now uses libjpeg's tables instead of zune-jpeg's.

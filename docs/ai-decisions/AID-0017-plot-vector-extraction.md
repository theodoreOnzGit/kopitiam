# AID-0017: kopitiam-plot reads PDF vector paths itself, rather than rasterising or reusing pdf-extract's callbacks

* **Status:** Pending review
* **Bead:** `kopitiam-szg`
* **Date:** 2026-07-14
* **Decided by:** AI (Claude), maintainer absent

## The brief

> Build plot digitisation for KOPITIAM: recover the underlying DATA from graphs
> printed in PDFs. `crates/kopitiam-plot` is yours.

The brief itself flagged the crux correctly and asked me to settle it before
building anything:

> **So your first job is to find out whether you can get the vector paths.**
> [...] If vector extraction proves impossible, fall back to raster [...] **Do not
> start here.** If you end up here, write an AID explaining why.

## The finding: vector extraction works, and this is a much better position than the brief allowed for

**We can read the path geometry, and it is essentially exact.** No rasterisation,
no colour segmentation, no anti-aliasing heuristics, no `image` crate. The
round-trip test recovers known data from a generated plot to within **1e-5**
(and that bound is the test's, not the method's — the true error is ~1e-7,
limited only by the `f32` precision the producer wrote its coordinates at).

The reason is worth stating plainly, because it changes what this crate *is*: a
PDF plot is not a picture of data. It is the data, drawn. The producer wrote the
points into the content stream as coordinates (`m`, `l`, `c`, `re`) and they are
still sitting there, transformed by an affine map that the tick marks and tick
labels let us recover. Digitisation is therefore a *geometry* problem, not an
image-processing one. Every pixel-based digitiser on the market is solving a
harder problem than it needs to.

So there is **no raster fallback in this crate, and that is deliberate**. A
figure that is a scanned or embedded image yields no plot and no numbers, which
is the honest outcome. See "what would make this wrong" below.

## The decision

Three ways to get path geometry. I took the third.

### (a) Rasterise and trace — rejected

The conventional approach, and the brief's stated fallback. Lossy by
construction; cannot recover a dash pattern; cannot recover a curve that passes
under another curve; needs anti-aliasing heuristics that are themselves a source
of silent error. Unnecessary, given (c) works.

### (b) Reuse `pdf-extract`'s `OutputDev::stroke` / `fill` callbacks — rejected, and this one is a trap

`pdf-extract` (already in the tree, already used by `kopitiam-pdf`) *does* expose
stroke and fill callbacks carrying path geometry, a CTM, a colour space and a
colour. It looks like exactly what we want, and it is less code. It is not
adequate, for two reasons that are fatal rather than stylistic:

1. **No line width and no dash pattern.** The callbacks pass colour and geometry
   and nothing else. `pdf-extract` tracks `line_width` internally but never
   surfaces it, and does not track the dash pattern at all. Colour, width and
   dash are *precisely* the three cues that distinguish one series from another —
   it is how the figure was authored and how a reader tells the curves apart. A
   digitiser built on these callbacks cannot separate a solid 1pt black curve
   from a dashed 2pt black one. That is not a corner case; it is what a
   two-series figure normally looks like.

2. **Five paint operators are silently dropped.** `pdf-extract`'s interpreter
   matches `"s" | "f*" | "B" | "B*" | "b"` and logs them as unhandled — no
   callback fires, **and the path buffer is not cleared**, so the discarded
   path's segments leak into whatever path is built next. Only `S` and `f` reach
   a callback. `B` (fill-then-stroke) is how a solid scatter marker is normally
   painted. This would lose entire series outright and corrupt their neighbours
   on the way past.

### (c) Walk the content stream ourselves with `lopdf` — **chosen**

`crates/kopitiam-plot/src/content.rs`. Full graphics state: CTM, stroke/fill
colour across DeviceGray/RGB/CMYK and `sc`/`scn`, line width, dash (including via
ExtGState `/LW` and `/D`), clipping bbox, `q`/`Q`, and Form XObject recursion
with the form's `/Matrix` applied.

`kopitiam-pdf` had already set this exact precedent, for this exact class of gap:
`font_resources.rs` re-walks the same content stream with `lopdf` to recover font
state that `OutputDev` does not expose. This is the same manoeuvre applied to
path state, and deliberately mirrors its shape.

**The cost, stated plainly:** the workspace now contains a second, partial PDF
content-stream interpreter, which must be maintained alongside `pdf-extract`'s.
That is a real, permanent maintenance liability and the maintainer should know
they have acquired it. I judge it worth paying, because the alternative is a
digitiser that cannot tell two curves apart — which is not a digitiser.

## A load-bearing fact about coordinate spaces, recorded because a dependency bump could silently break it

Everything here depends on the geometry we extract and the tick-label text
`kopitiam-pdf` extracts landing in the **same coordinate space**. If they do not,
every tick match is garbage and every calibration is silently wrong.

They do, and the reason is not obvious:

> `pdf-extract` computes a `flip_ctm` that would convert to a top-left-origin,
> y-down space — but it only applies that flip inside its **own** HTML/SVG output
> devices. The generic `OutputDev` path that `kopitiam-pdf` builds on passes the
> text matrix straight through: `show_text` takes the flip matrix as an argument
> named **`_flip_ctm` and never uses it**, and `Processor`'s CTM starts at the
> identity. So `output_character` receives `Trm = Tsm × Tm × CTM` in **raw PDF
> user space, y-up**.

Our walk also starts its CTM at the identity, so paths and text agree *by
construction*. This is asserted directly in `tests/coordinate_space.rs` rather
than trusted — if a future `pdf-extract` release starts applying its flip, that
test fails loudly instead of the crate quietly producing upside-down data.

## What the crate refuses to do, and why that is the design

The failure mode that matters in plot digitisation is not "it didn't work". It is
**silent confident wrongness**: a plausible dataset that is quietly garbage. A
fabricated validation dataset is worse than no dataset, because someone will
publish against it and nobody can tell whether a solver's disagreement with it is
the solver's fault.

So `DigitisedPlot::warnings` is a first-class output, and the crate declines to
produce numbers it cannot justify:

* **An axis with fewer than two labelled ticks produces no data values at all.**
  Not approximate values — none. The series still carries its page geometry as
  evidence that a curve exists.
* **An axis with exactly two ticks warns loudly.** Two points fit *any*
  two-parameter model exactly, so linear and logarithmic are both consistent with
  the figure and the scale is genuinely undecidable. We assume linear and say so
  in terms ("ASSUMED LINEAR — if this axis is logarithmic, every value is wrong").
* **Log axes are detected from evidence**, by comparing the normalised residual of
  a linear fit against a fit in log space — not by pattern-matching the labels.
  A log axis read as linear is the single most dangerous silent failure available
  here, and it is common in exactly this project's domains.
* **Bézier segments are never flattened.** Only on-curve anchors are reported.
  Flattening would *invent* coordinates that were never in anyone's dataset, and a
  caller would have no way to tell an invented point from a measured one. Where
  anchors might be spline knots rather than data, that is warned about.
* **Every `DataPoint` carries `page_xy`**, and every plot carries the calibration
  and the tick observations it was fitted from, with the label text as printed.
  Any number can be traced back to a position on the page and checked by a human
  against the printed figure. That is the Scientific Standards requirement, and it
  is why the API looks the way it does.

## What a real paper taught us that the synthetic corpus could not

The crate was developed against PDFs it writes itself (`tests/common`), which is
the only way to get ground truth — any real PDF has, by definition, lost the
numbers you would be checking against. But the synthetic corpus can only contain
cases we thought of. Pointing the finished crate at a real published paper's
figures found **four** bugs, every one of which produced a *plausible wrong
answer* rather than a failure:

1. Its pressure axis is labelled `0`, `1,000` … `11,000`. **Comma digit grouping**,
   which `f64::from_str` rejects. The axis collapsed to one usable tick.
2. Its plot frame is a single closed 5-vertex subpath, so the (single-segment)
   furniture test walked past it and reported **the figure's own border as a data
   series**.
3. Its y-axis title is rotated, arriving as ~15 single-glyph fragments; the title
   logic picked `"essur"` (from "pressure") as the axis name and would have put
   that into the knowledge graph as an entity name.
4. Every measured point is drawn as **eight separate strokes** — an error-bar
   cross — so each measurement was reported as eight data points, scattered around
   the true value at the ends of its own error bars.

All four are fixed, and the case is pinned as an `#[ignore]`d regression test
(`tests/real_paper.rs`; ignored because the PDF is not ours to redistribute).

The general lesson, which is the part worth keeping: **synthetic ground truth
proves the pipeline; only real documents find the assumptions.** Both are needed.

### The one judgment call inside that list

Comma grouping. `1,000` is one thousand in the English-language scientific
literature this crate targets — and one, to three decimal places, in a
decimal-comma locale. Nothing in the figure disambiguates it. I read it as one
thousand (the pattern test requires *every* comma to be followed by exactly three
digits, so `1,5` is refused rather than silently becoming 15), and I raise a
warning on any axis where it applies, because the consequence of being wrong is a
silent 1000× error. The printed label is preserved verbatim in the tick
observation so a reader can check.

## Alternatives considered and rejected

| Option | Why not |
| --- | --- |
| Raster / pixel tracing | Unnecessary — the vectors are right there. Lossy, cannot recover dash or occluded curves. |
| `pdf-extract`'s stroke/fill callbacks | No line width, no dash; silently drops `s`/`f*`/`B`/`B*`/`b` and corrupts the following path. |
| Flatten Béziers into polylines | Fabricates coordinates that were never data, indistinguishable to the caller from real ones. |
| Emit values from an uncalibrated axis | The whole point of the crate is not to do this. |
| Guess the scale when two ticks allow both | Same. Warn instead. |
| Add a `csv` crate dependency | Long-format CSV with one quoted field is ~20 lines. Not worth a dependency. |
| Take an `image`/OpenCV dependency | Would breach the Pure Rust Core, and buys nothing given the above. |

## What would make this decision wrong

Listed honestly, because these are the conditions under which the maintainer
should revisit it:

* **If the corpus turns out to be mostly raster figures.** Everything above rests
  on the observation that scientific PDFs are overwhelmingly vector. That is true
  of the papers on this machine, and true of anything produced by LaTeX,
  matplotlib, gnuplot, Word or Excel — but a corpus of *scanned* papers (older
  literature, photocopied reports) would be entirely opaque to this crate, and the
  raster path the brief described would then be needed after all. It would be
  additive, not a replacement: vector first, raster only when there are no paths.
* **If `pdf-extract` grows line-width and dash on its callbacks and fixes the five
  dropped operators.** Then option (b) becomes viable and our interpreter becomes
  redundant maintenance. Worth re-checking on each `pdf-extract` bump; the tests
  in `content.rs` name the exact operators to look for.
* **If `pdf-extract` starts applying its `flip_ctm` on the `OutputDev` path.**
  Then text and geometry land in different spaces and every calibration silently
  inverts. `tests/coordinate_space.rs` exists to catch precisely this, and it will
  fail rather than let it through.
* **If the warnings turn out to be so numerous on real papers that people stop
  reading them.** Warning fatigue would defeat the entire safety design. The
  current set fires only on genuine ambiguity (I checked against a real paper: 3-4
  warnings per figure, all of them true and all of them actionable). If that
  number climbs, the fix is to tighten the conditions, not to delete the warnings.

## What is deliberately not done

Stated so nobody assumes otherwise:

* Error bar **magnitudes**. The central value of a point with error bars is
  recovered exactly (from the vertex the bars cross at); the bar *lengths* — i.e.
  the stated uncertainty — are not. For validation work this is a real gap, since
  the uncertainty is half the point of the measurement. It is warned about, and it
  is the most valuable next thing to build.
* Glyph-drawn scatter markers (markers drawn as font characters, not paths). They
  arrive as text and are lost. The symptom is detected and warned about.
* Fills, contours, heatmaps, shading.
* Wiring into `apps/cli`. The crate ships an `examples/digitise.rs` that runs
  against any PDF in one command; promoting that to a real CLI subcommand is the
  obvious follow-up and is what CLAUDE.md's dogfooding rule asks for.

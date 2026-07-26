# AID-0051: Conventions for the faithful MuPDF text-extraction port (`kopitiam-pdf::mupdf`)

* **Status:** Pending review
* **Date:** 2026-07-26
* **Decided by:** AI (Claude), maintainer absent
* **Scope:** the whole MuPDF port -- which lives as the **`mupdf` module inside
  `kopitiam-pdf`** (`crates/kopitiam-pdf/src/mupdf/…`), not as a standalone
  crate. This AID is the pattern-setter and the FOUNDATION `geometry` module
  (`crates/kopitiam-pdf/src/mupdf/geometry.rs`) is its first application.

## The brief

> Start a faithful Rust port of MuPDF's text-extraction engine inside the
> `kopitiam` workspace. The port lives as the `mupdf` module inside
> `kopitiam-pdf` (`crates/kopitiam-pdf/src/mupdf/…`), not as a standalone crate.
> First task: port the FOUNDATION geometry module (`fitz/geometry.h` +
> `fitz/geometry.c`) as the pattern-setter for the whole port. Keep scope tight:
> module scaffold + geometry + base types. Build and test. Don't port other
> modules yet.

MuPDF is **AGPL-3.0**, © Artifex Software (vendored read-only at
`crates/kopitiam-pdf/vendor/mupdf`, commit `19f1284`). KOPITIAM is
**AGPL-3.0-only**, which is exactly what makes this port permissible -- an AGPL
upstream adapts into an AGPLv3 work. This is a **close adaptation / translation**
(middle tier of `docs/ACKNOWLEDGEMENTS.md`), not clean-room study.

## Decisions (follow these in every later module)

### 1. Provenance is recorded twice: at the file top, and at the point of use

Every ported source file opens with this header (the exact wording, only the
source-file name changing):

```rust
//! Ported from MuPDF `source/fitz/<file>.c` + `include/mupdf/fitz/<file>.h`
//! (commit 19f1284, AGPL-3.0, © Artifex Software, Inc.), translated to Rust for
//! KOPITIAM (AGPL-3.0-only). Close adaptation: the algorithms and numeric
//! behaviour follow MuPDF; the code is re-expressed in idiomatic Rust. See
//! docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references").
```

Where a Rust function tracks a specific C function, it carries a one-line
breadcrumb at that function so the 1:1 map stays discoverable:

```rust
// MuPDF: fz_transform_rect (geometry.c:519)
```

`docs/ACKNOWLEDGEMENTS.md` already lists MuPDF; no new row needed per module, but
the per-file + per-function provenance above is mandatory.

### 2. Numbers match MuPDF, not "textbook correct"

MuPDF geometry is C `float`, so KOPITIAM uses `f32`. Three subtleties bite, and
we copy MuPDF rather than "fixing" it:

* **`FZ_PI` is the truncated `3.14159265f`**, not `std::f32::consts::PI`. MuPDF
  hard-codes it in `fitz/system.h`; using the fuller pi drifts the last ULP of
  every rotation. Copy the truncated constant.
* **`fz_min`/`fz_max` are the ternary `a < b ? a : b`,** reimplemented by hand --
  *not* `f32::min`/`f32::max`, whose NaN handling differs (std returns the
  non-NaN operand; the ternary propagates a second-operand NaN). It matters for
  quads, which legitimately carry NaN (invalid) and inf (infinite) ordinates.
* **Widen where MuPDF widens.** Matrix inversion and the `fmod` in `rotate`
  compute in `double` in C; do the same in `f64` and narrow back. Everything
  else stays `f32`.

### 3. Free functions become methods, but the C name stays findable

`fz_transform_rect(r, m)` -> `r.transform(m)`; `fz_concat(a, b)` -> `a.concat(b)`
(and `a * b` via `impl Mul`, where `a * b` applies `a` first); `fz_scale` ->
`Matrix::scale`, etc. Every one keeps its `// MuPDF: fz_<name>` breadcrumb. Base
value types derive `Clone, Copy, Debug, PartialEq` -- they are the plain-old-data
structs MuPDF passes by value.

Name map for this module: `fz_matrix`->`Matrix`, `fz_point`->`Point`,
`fz_rect`->`Rect`, `fz_irect`->`IRect`, `fz_quad`->`Quad`.

### 4. Empty / infinite / invalid rect semantics are preserved exactly

This is the subtle heart of MuPDF geometry and the reason it picked its
representation. Do not conflate:

* **valid** = `x0 <= x1 && y0 <= y1` (zero-area rects are valid);
* **infinite** = all four ordinates pinned to the `FZ_{MIN,MAX}_INF_RECT`
  sentinels (`i32::MIN` .. `0x7fffff80`, the largest int that round-trips
  through `f32`);
* **invalid** = not valid (the `{0,0,-1,-1}` sentinel, or any swapped-corner
  rect);
* **empty** (`is_empty`) = zero-or-negative area (`x0>=x1 || y0>=y1`), which
  *includes* all invalid rects but is a different question from validity.

Consequences that later modules must not "tidy away": `intersect` does **not**
pre-check emptiness (disjoint rects yield an empty result downstream code relies
on); `union` checks emptiness *before* infiniteness; `transform` special-cases
infinite rects and re-swaps corners to keep an invalid input invalid; point-in-
rect is **half-open** (top/right excluded).

### 5. `fz_context` and error handling (declared here, implemented later)

Geometry is pure math -- no `fz_context`. For later modules the convention is:

* `fz_context` -> Rust ownership / explicit allocators (no global context
  threaded through every call);
* `fz_try` / `fz_catch` / `fz_throw` -> `Result` (and `Option` where MuPDF
  returns a "degenerate" flag, e.g. `fz_try_invert_matrix` -> `Option<Matrix>`).

Not implemented in the geometry phase; recorded so the next porter follows it.

### 6. What is deliberately left out of `geometry.rs`

`geometry.c`'s checked-integer helpers (`fz_ckd_*`) and `geometry.h`'s 0..255
pixel-blend macros (`fz_mul255`, `FZ_BLEND`, ...) are not geometry and not on the
text-extraction path -- std's `checked_mul`/`saturating_add` cover us when
needed. `fz_gridfit_matrix` is *declared* in the header but has **no body in
geometry.c** (it lives in another translation unit), so it is out of scope for
this file. Port these only if and when a later module actually needs them.

## Where the conventions live in code

The load-bearing summary of this AID is duplicated as the module-doc at the top
of `crates/kopitiam-pdf/src/mupdf/geometry.rs` (with the port-wide framing in
`crates/kopitiam-pdf/src/mupdf/mod.rs`), so a porter reading the code sees the
rules without leaving the file. This AID is the durable, reviewable record.

## Translate from the MuPDF C source only

Every phase translates from MuPDF's **C** source (`vendor/mupdf/source/**`,
`vendor/mupdf/include/**`), never from PyMuPDF's Python. PyMuPDF is a thin
binding over the same C engine -- it carries no extraction algorithm of its own,
so translating it would only reproduce the C at one remove. It stays vendored
purely as an API-shape reference (what `get_text` ultimately yields); the code we
port is always the underlying C. (Maintainer's instruction, 2026-07-26.)

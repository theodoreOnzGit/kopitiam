# AID-0052: The MuPDF port does not port MuPDF's C-library delegations — it substitutes pure-Rust crates or re-implements from spec

* **Status:** Pending review
* **Date:** 2026-07-26
* **Decided by:** AI (Claude), maintainer absent
* **Scope:** the rendering half of the MuPDF port inside `kopitiam-pdf`
  (`crates/kopitiam-pdf/src/mupdf/…`) — specifically the three subsystems where
  MuPDF hands off to a bundled C library: **FreeType** (embedded-font outlines,
  `glyph.rs` / `glyph_truetype.rs` / `glyph_cff.rs`), **libjpeg** (DCTDecode
  images, `page_image.rs`), and **zlib** (FlateDecode, `filter_flate.rs`). This
  extends AID-0051 (the port conventions); it does not restate it.

## The brief

The wave that took the MuPDF port from text-extraction into a full rendering
vertical — the draw device / rasterizer (`pixmap.rs`, `draw_edge.rs`,
`draw_path.rs`, `draw_device.rs`), embedded-font glyph outlines, and embedded
JPEG image decode. AID-0051 already fixes *how* to translate MuPDF's own C. The
open question this wave forced is **what to do at the points where MuPDF's C does
not do the work itself, but calls out to a C library** (`<jpeglib.h>`,
`<zlib.h>`, FreeType). Porting those libraries too would be a large translation
of a *different* upstream, and — for FreeType — would pull a fourth AGPL/FTL
codebase's provenance into the tree.

## The decision

At every point where MuPDF delegates to a bundled C library, the port does
**not** translate that library. It does one of two things instead, chosen by
whether a mature pure-Rust equivalent already exists:

### 1. Substitute a pure-Rust crate where one exists (zlib, libjpeg)

* **FlateDecode / `<zlib.h>` → `miniz_oxide`.** The `fz_stream` filter *plumbing*
  around it is still translated from MuPDF (`filter_flate.rs`); only the DEFLATE
  codec itself is the crate.
* **DCTDecode / libjpeg → `zune-jpeg`** (`MIT OR Apache-2.0 OR Zlib`, pure Rust,
  `zune-core` its only dep). MuPDF's `pdf-image.c` / `image.c` field-reading and
  sample unpacking are translated faithfully in `page_image.rs`; the JPEG
  bitstream decode that `load-jpeg.c` gives to libjpeg is handed to `zune-jpeg`.

The substituted crate is a **linked, shipped dependency**, credited in
`docs/ACKNOWLEDGEMENTS.md` under "Notable shipped Rust dependencies" and named at
the point of use — a different provenance relationship from the MuPDF
translation, and recorded as such.

### 2. Re-implement from the format specification where no clean substitute fits (FreeType)

FreeType has no drop-in pure-Rust equivalent that the port wants to take on as a
dependency, and font-outline extraction is small and well-specified. So the port
**avoids FreeType entirely** and re-implements the two embedded-font programs
**clean-room from their public specifications**:

* the outline→`Path` *callback shape* (`move_to` / `line_to` / `conic_to` /
  `cubic_to`) is kept from MuPDF's `font.c` decompose — that part is still a
  MuPDF translation;
* the **TrueType `glyf`/`loca`** parse and on-/off-curve reconstruction
  (`glyph_truetype.rs`) is written from the **OpenType specification**;
* the **CFF / Type2 charstring interpreter** (`glyph_cff.rs`) is written from the
  **Adobe Type2 Charstring Format, Technical Note #5177**.
* **One recorded exception:** the CFF *container* parse (INDEX / DICT / charset /
  FDSelect / `subr_bias`) is a close adaptation of MuPDF's **own** non-FreeType
  CFF reader in `source/fitz/subset-cff.c` (still MuPDF `19f1284`), not of
  FreeType. It is cited at the point of use.

## Why this is the maintainer's call, and why it went this way

The alternatives were: (a) port FreeType/libjpeg/zlib too — a large translation
of unrelated upstreams, and for FreeType a licensing entanglement (FTL/GPL) the
project has no reason to acquire; or (b) link the system C libraries via FFI —
which breaks the **Pure Rust Core** commitment (`PROVENANCE.md`) that the build
needs no mandatory C toolchain. Substitution/re-implementation keeps the build
pure-Rust *and* keeps the provenance honest: the glyph decoders are **not**
derivative of FreeType (they are spec-based original Rust plus one
MuPDF-`subset-cff.c` adaptation), and the JPEG/zlib paths are permissive shipped
crates, not translations of anyone's C.

**This would be wrong if** a byte-exact match to MuPDF's rendering were a
requirement — a different JPEG decoder or a spec-built rasterizer will not be
bit-identical to libjpeg + FreeType + MuPDF's GEL. It is not a requirement: the
target is a legible terminal PDF viewer at tdf-parity, and the anti-aliased fill
is deliberately a coverage-accumulating sweep, "the same nonzero/even-odd result,
numerically simpler" (per `draw_edge.rs`), not a fixed-point-exact GEL clone. If
pixel-exactness ever *is* required, this decision is the thing to revisit.

## Relationship to AID-0051

AID-0051 governs the translation of MuPDF's **own** C (numbers match MuPDF,
free functions become methods, `fz_context`→ownership, translate from the C not
PyMuPDF). AID-0052 governs the **boundary** of that translation: where MuPDF's C
stops doing the work and calls a C library, the port leaves MuPDF's algorithm
behind and either substitutes a Rust crate or re-implements from spec. Every
ported file still carries the AID-0051 provenance header; these three carry an
*additional* sentence naming the substitution/spec, as they already do.
